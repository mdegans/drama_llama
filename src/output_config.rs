//! `Prompt::output_config` → [`SamplingMode`] compiler.
//!
//! Reads [`OutputConfig`] from a [`Prompt`] and emits a GBNF that
//! forces the model's response to match the configured JSON Schema.
//! Mirrors the `tool_choice` module's shape —
//! same schema compiler, same optional `<think>...</think>` preamble,
//! different wrapper rule.
//!
//! # Thought preamble
//!
//! [`OutputConfigOptions::allow_thought`] defaults to `true` — the
//! opposite of [`ToolChoiceOptions::allow_thought`](crate::ToolChoiceOptions).
//! Structured-output callers typically want the model to reason about
//! the shape before committing to JSON, and reasoning-capable models
//! (cogito, Qwen3, DeepSeek-R1) emit `<think>...</think>` by habit.
//! Flip it off when you want to reject any prelude and start directly
//! with the JSON body.
//!
//! # Interaction with `tool_choice`
//!
//! `Session` treats `tool_choice` and `output_config` as mutually
//! exclusive at grammar-resolution time, with `tool_choice` winning
//! when both are set. Library callers using this module directly are
//! expected to enforce their own priority if they mix the two.
//!
//! # Framing
//!
//! Where the JSON body sits is the chat format's business, so the
//! grammar follows [`OutputConfigOptions::framing`], which `Session`
//! fills from its dialect on every call. [`ResponseFraming::Bare`] is
//! the `<think>…</think>`-then-JSON shape above;
//! [`ResponseFraming::Harmony`] puts the body in gpt-oss's final
//! channel. A grammar framed for the wrong format does not merely
//! constrain badly — a phase-split trigger the model never writes
//! leaves the body unconstrained (2026-10-01: gpt-oss returned invalid
//! JSON with a 200, because `</think>` never fired).
//!
//! [`OutputConfig`]: misanthropic::prompt::output::OutputConfig
//! [`Prompt`]: crate::Prompt
//! [`SamplingMode`]: crate::SamplingMode

use std::fmt::Write;

use misanthropic::prompt::output::{OutputConfig, OutputFormat};

use crate::dialect::harmony;
use crate::grammar_compile::{
    emit_think_body_rules, emit_thought_rules, emit_until_rules,
    escape_for_gbnf_string, schema_to_gbnf, JSON_GRAMMAR,
};
use crate::{DeferredGrammar, GrammarError, Prompt, SamplingMode};

/// Byte sequence that triggers deferred-grammar promotion when
/// [`OutputConfigOptions::phase_split`] is on. Matches the closing tag of
/// the thought preamble emitted by reasoning models.
pub const THINK_CLOSE_TRIGGER: &[u8] = b"</think>";

/// The [`ResponseFraming::Harmony`] phase-split trigger: the final
/// channel's header up to the channel name. What follows it — an
/// optional ` <|constrain|>json`, then `<|message|>` — is the grammar's.
pub const HARMONY_FINAL_TRIGGER: &str = "<|channel|>final";

/// How the chat format frames a structured response — where the JSON
/// body starts, and what reasoning may precede it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponseFraming {
    /// The body follows an optional `<think>…</think>` thought (cogito,
    /// Qwen, DeepSeek-R1) or starts the response.
    #[default]
    Bare,
    /// OpenAI Harmony (gpt-oss): the generation prompt ends at
    /// `<|start|>assistant`, so the body lives in the final channel —
    /// `<|channel|>final<|message|>{…}`, or with the ` <|constrain|>json`
    /// gpt-oss writes for JSON — after at most one analysis block
    /// (closed by `<|end|>`, reopened by `<|start|>assistant`).
    Harmony,
}

/// Options for [`grammar_for_output_config`].
#[derive(Clone, Debug)]
pub struct OutputConfigOptions {
    /// Permit an optional `<think>…</think>` block before the JSON
    /// body. Defaults to `true` because reasoning-capable models
    /// (cogito, Qwen3, DeepSeek-R1) emit thought tags naturally and
    /// structured-output callers usually want the reasoning preserved
    /// as a [`Block::Thought`](crate::Block) on the assistant message.
    pub allow_thought: bool,
    /// When `true` *and* `allow_thought` is also `true`, compile the
    /// grammar as a JSON-only body and return a [`DeferredGrammar`]
    /// triggered by `</think>` instead of a single unified grammar. The
    /// caller (typically `TokenPredictor`) runs unconstrained during the
    /// thought preamble and only activates the JSON grammar once the
    /// trigger fires — which restores pure-inference tok/s during the
    /// otherwise-permissive `<think>` body. Defaults to `true`; flip off
    /// to keep the old unified-grammar behaviour (useful for callers that
    /// need the matcher to also guard the thought structure itself).
    pub phase_split: bool,
    /// The bytes between `</think>` and the JSON body, spelled
    /// literally so the constrained turn re-renders byte-for-byte —
    /// a fact about the template, not a preference: `Session` fills it
    /// from the dialect's measured
    /// [`ReasoningSyntax::separator`](crate::dialect::ReasoningSyntax::separator)
    /// on every call. `None` keeps the permissive single-byte `ws` gap,
    /// which cannot express Qwen's `\n\n` (#112).
    pub thought_separator: Option<String>,
    /// Where the JSON body sits in the response — a fact about the chat
    /// format, filled by `Session` from its dialect on every call, like
    /// [`Self::thought_separator`].
    pub framing: ResponseFraming,
}

impl Default for OutputConfigOptions {
    fn default() -> Self {
        Self {
            allow_thought: true,
            phase_split: true,
            thought_separator: None,
            framing: ResponseFraming::Bare,
        }
    }
}

impl OutputConfigOptions {
    /// The GBNF fragment that follows a closed thought: the literal
    /// separator when known, else the permissive `ws`.
    fn after_thought(&self) -> String {
        match &self.thought_separator {
            Some(sep) if sep.is_empty() => String::new(),
            Some(sep) => format!(r#" "{}""#, escape_for_gbnf_string(sep)),
            None => " ws".to_string(),
        }
    }
}

/// Output of [`compile_output_config`] — either a single unified grammar
/// (run it from the start) or a thought/JSON phase-split pair (run
/// unconstrained until the trigger, then promote the JSON grammar).
#[derive(Clone, Debug)]
pub enum CompiledOutputConfig {
    /// Standard single-grammar shape. Push this into `SamplerConfig::modes`.
    Single(SamplingMode),
    /// Phase-split shape. Install into `SamplerConfig::deferred_grammar`
    /// and let `TokenPredictor` promote it when the trigger is emitted.
    Deferred(DeferredGrammar),
}

impl CompiledOutputConfig {
    /// Flatten to a single `SamplingMode` by discarding the deferred
    /// wrapper. Callers that haven't been updated to handle the deferred
    /// path can use this to stay on the legacy code path.
    pub fn into_grammar(self) -> SamplingMode {
        match self {
            Self::Single(g) => g,
            Self::Deferred(d) => SamplingMode::Grammar(d.grammar),
        }
    }
}

/// Build a [`SamplingMode::Grammar`] that constrains the model's
/// response to match `config`'s JSON Schema, optionally preceded by a
/// `<think>...</think>` block. Ignores [`OutputConfigOptions::phase_split`]
/// — always emits the unified grammar. Use [`compile_output_config`] for
/// the phase-split path.
pub fn grammar_for_output_config(
    config: &OutputConfig,
    opts: &OutputConfigOptions,
    thought_pre_opened: bool,
) -> Result<SamplingMode, OutputConfigError> {
    let schema = match &config.format {
        Some(OutputFormat::JsonSchema(f)) => &f.schema,
        _ => return Err(OutputConfigError::UnsupportedFormat),
    };
    let source = build_grammar_source(schema, opts, thought_pre_opened);
    Ok(SamplingMode::grammar(&source)?)
}

/// Compile an [`OutputConfig`] into a [`CompiledOutputConfig`] that either
/// holds a single unified grammar or a `</think>`-triggered
/// [`DeferredGrammar`], depending on `opts.phase_split` and
/// `opts.allow_thought`. Phase-split applies only when both are `true`.
///
/// `thought_pre_opened`: the rendered generation prompt already opened
/// the thought (Qwen-style `<think>\n` scaffold, or a resumed open
/// thought). The unified grammar's root must then anchor close-first
/// and must not spell the opener literal — a root offering `"<think>"`
/// at position 0 masks the model's real preference and *forces* a
/// duplicate opener (the #107 mechanism). The deferred path is
/// unaffected: its trigger is the closer, which the model must emit
/// either way.
pub fn compile_output_config(
    config: &OutputConfig,
    opts: &OutputConfigOptions,
    thought_pre_opened: bool,
) -> Result<CompiledOutputConfig, OutputConfigError> {
    let schema = match &config.format {
        Some(OutputFormat::JsonSchema(f)) => &f.schema,
        _ => return Err(OutputConfigError::UnsupportedFormat),
    };
    if opts.phase_split && opts.allow_thought {
        let source = build_json_only_grammar_source(schema, opts);
        let trigger = match opts.framing {
            ResponseFraming::Bare => THINK_CLOSE_TRIGGER,
            ResponseFraming::Harmony => HARMONY_FINAL_TRIGGER.as_bytes(),
        };
        Ok(CompiledOutputConfig::Deferred(DeferredGrammar {
            grammar: crate::CompiledGrammar::parse(&source)?,
            activate_after: vec![trigger.to_vec()],
            // The JSON-body grammar starts *after* the trigger, which
            // itself stays outside the constrained span.
            feed_trigger: false,
        }))
    } else {
        let source = build_grammar_source(schema, opts, thought_pre_opened);
        Ok(CompiledOutputConfig::Single(SamplingMode::grammar(
            &source,
        )?))
    }
}

/// The prompt's `output_config` iff it asks for structured output. An
/// `output_config` carrying only an [`effort`](OutputConfig::effort)
/// (`format: None`) constrains nothing — the effort reaches the chat
/// template instead — so it must not claim the grammar slot. Treating it
/// as a format request failed every effort-only request with
/// [`OutputConfigError::UnsupportedFormat`] (the first Agora run with
/// `thinking_effort`, 2026-09-23).
fn structured(prompt: &Prompt) -> Option<&OutputConfig> {
    prompt
        .output_config
        .as_ref()
        .filter(|config| config.format.is_some())
}

/// The JSON Schema the prompt's structured output must satisfy, if it
/// asks for any — what the post-generation schema check validates
/// against.
pub(crate) fn json_schema(prompt: &Prompt) -> Option<&serde_json::Value> {
    match &structured(prompt)?.format {
        Some(OutputFormat::JsonSchema(f)) => Some(&f.schema),
        _ => None,
    }
}

/// Derive the output-config grammar directly from a [`Prompt`]. Reads
/// `prompt.output_config`; returns `Ok(None)` when unset. Legacy entry
/// point — ignores `phase_split`. Use [`compile_prompt_output_config`] for
/// the phase-split-aware shape.
pub fn grammar_for_prompt(
    prompt: &Prompt,
    opts: &OutputConfigOptions,
    thought_pre_opened: bool,
) -> Result<Option<SamplingMode>, OutputConfigError> {
    let Some(config) = structured(prompt) else {
        return Ok(None);
    };
    Ok(Some(grammar_for_output_config(
        config,
        opts,
        thought_pre_opened,
    )?))
}

/// Phase-split-aware equivalent of [`grammar_for_prompt`]. Returns the
/// compiled output config (either unified grammar or deferred) when
/// `prompt.output_config` is set.
///
/// `phase_split` is auto-disabled for this call when the prompt has
/// no thinking enabled (absent or `Thinking::Disabled`). Phase-split is
/// a performance optimization that defers the JSON grammar until
/// `</think>` appears in the output — with thinking off, `</think>`
/// never appears, so the deferred grammar would never activate and
/// the model would generate unconstrained free text. Auto-disabling
/// here gives callers the correct behavior (structured output works
/// regardless of whether thinking is on) without needing to know the
/// phase-split knob exists. Session-level `output_config_opts` is
/// still honored when thinking IS enabled.
pub fn compile_prompt_output_config(
    prompt: &Prompt,
    opts: &OutputConfigOptions,
    thought_pre_opened: bool,
) -> Result<Option<CompiledOutputConfig>, OutputConfigError> {
    let Some(config) = structured(prompt) else {
        return Ok(None);
    };
    let effective = if !crate::chat_template::thinking_enabled(prompt) {
        OutputConfigOptions {
            phase_split: false,
            ..opts.clone()
        }
    } else {
        opts.clone()
    };
    Ok(Some(compile_output_config(
        config,
        &effective,
        thought_pre_opened,
    )?))
}

/// Emit the GBNF source text for an output-config constraint. Kept
/// `pub(crate)` so tests can inspect the grammar text directly.
pub(crate) fn build_grammar_source(
    schema: &serde_json::Value,
    opts: &OutputConfigOptions,
    thought_pre_opened: bool,
) -> String {
    let mut src = String::with_capacity(512);

    if opts.framing == ResponseFraming::Harmony {
        // gpt-oss never pre-opens its analysis channel (the generation
        // prompt ends at `<|start|>assistant`), so `thought_pre_opened`
        // has nothing to anchor here. At most one analysis block, never
        // `(analysis | commentary)*`: every extra block the grammar
        // admits is somewhere a model can go instead of answering
        // (see the forced-call grammar in `dialect::emit`).
        let analysis = if opts.allow_thought {
            "h_analysis? "
        } else {
            ""
        };
        let _ = writeln!(src, "root ::= {analysis}h_final output_schema");
        let _ = writeln!(
            src,
            r#"h_analysis ::= "{open}" h_end "{start}""#,
            open = escape_for_gbnf_string(harmony::ANALYSIS_OPEN),
            start = escape_for_gbnf_string(harmony::START_ASSISTANT),
        );
        emit_until_rules("h_end", harmony::END, &mut src);
        emit_harmony_final_header("h_final", HARMONY_FINAL_TRIGGER, &mut src);
    } else if thought_pre_opened {
        // The template already emitted the opener, so the tag IS open:
        // the thought is mandatory (close-first), the opener literal
        // must not appear (it would force a duplicate — #107), and
        // this dominates `allow_thought = false`, mirroring the tool
        // grammars' `EagerThoughtPreOpened` precedent — a caller
        // cannot forbid a thought the render already started.
        let after = opts.after_thought();
        let _ = writeln!(
            src,
            r#"root ::= think_body "</think>"{after} output_schema"#
        );
        emit_think_body_rules(&mut src);
    } else if opts.allow_thought {
        let after = opts.after_thought();
        let _ = writeln!(src, "root ::= ( thought{after} | ws ) output_schema");
        emit_thought_rules(&mut src);
    } else {
        let _ = writeln!(src, "root ::= ws output_schema");
    }

    schema_to_gbnf(schema, "output_schema", &mut src);
    src.push_str(JSON_GRAMMAR);
    src
}

/// Emit the JSON-only grammar used by the deferred / phase-split path.
/// Root starts right after the `</think>` trigger, so it opens with the
/// thought separator; thought rules are omitted entirely because
/// `TokenPredictor` doesn't run the matcher during the thought preamble.
pub(crate) fn build_json_only_grammar_source(
    schema: &serde_json::Value,
    opts: &OutputConfigOptions,
) -> String {
    let mut src = String::with_capacity(512);
    match opts.framing {
        ResponseFraming::Bare => {
            let _ =
                writeln!(src, "root ::={} output_schema", opts.after_thought());
        }
        ResponseFraming::Harmony => {
            // The trigger consumed `<|channel|>final`; the rest of the
            // header is constrained.
            let _ = writeln!(src, "root ::= h_final output_schema");
            emit_harmony_final_header("h_final", "", &mut src);
        }
    }
    schema_to_gbnf(schema, "output_schema", &mut src);
    src.push_str(JSON_GRAMMAR);
    src
}

/// The Harmony final-channel header, from `lead` (what of
/// `<|channel|>final` the rule still has to spell) through
/// `<|message|>`, with the ` <|constrain|>json` gpt-oss writes for JSON
/// optional: both spellings are the model's, and the parser reads both.
/// The body follows `<|message|>` directly, as the template renders it.
fn emit_harmony_final_header(rule: &str, lead: &str, out: &mut String) {
    let lead = match lead {
        "" => String::new(),
        lead => format!(r#""{}" "#, escape_for_gbnf_string(lead)),
    };
    let _ = writeln!(
        out,
        r#"{rule} ::= {lead}( " {constrain}json" )? "{msg}""#,
        constrain = escape_for_gbnf_string(harmony::CONSTRAIN),
        msg = escape_for_gbnf_string(harmony::MESSAGE),
    );
}

/// Errors from [`grammar_for_output_config`].
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum OutputConfigError {
    /// The [`OutputFormat`] variant is not one this crate knows how to
    /// compile to a grammar. Reserved for future upstream variants —
    /// today only [`OutputFormat::JsonSchema`] exists.
    #[error(
        "unsupported OutputFormat variant; only JsonSchema is handled today"
    )]
    UnsupportedFormat,
    /// The compiled GBNF source failed to parse.
    #[error("compiled grammar is invalid: {0}")]
    Grammar(#[from] GrammarError),
}

static_assertions::assert_impl_all!(OutputConfigError: Send, Sync);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grammar, GrammarState};
    use serde_json::json;
    use std::sync::Arc;

    fn accepts(source: &str, input: &str) -> bool {
        let grammar = match Grammar::parse(source) {
            Ok(g) => g,
            Err(e) => panic!("grammar failed: {e}\n--- source ---\n{source}"),
        };
        let mut state = GrammarState::new(Arc::new(grammar));
        if state.advance_bytes(input.as_bytes()).is_err() {
            return false;
        }
        state.is_complete()
    }

    fn cfg(schema: serde_json::Value) -> OutputConfig {
        OutputConfig::json_schema(schema)
    }

    /// #112: with the template's separator known, every root that
    /// follows a thought spells it — and only it. The permissive `ws`
    /// is at most one byte, so Qwen's `</think>\n\n{…}` was
    /// unreachable and the model wrote `</think>\n{…}`, `</think> {…}`
    /// or `</think>{…}` instead; each re-rendered differently and lost
    /// the tip.
    /// An effort-only `output_config` (`format: None`) requests no
    /// grammar: it must fall through to the tool grammar, not fail the
    /// request as an unsupported format (2026-09-23 Agora outage).
    #[test]
    fn effort_only_output_config_compiles_to_nothing() {
        use misanthropic::prompt::Effort;
        let prompt = Prompt::default()
            .add_message((misanthropic::prompt::message::Role::User, "hi"))
            .unwrap()
            .effort(Effort::Medium);
        assert!(prompt.output_config.is_some(), "precondition");
        let opts = OutputConfigOptions::default();
        assert!(compile_prompt_output_config(&prompt, &opts, false)
            .unwrap()
            .is_none());
        assert!(grammar_for_prompt(&prompt, &opts, false).unwrap().is_none());
    }

    #[test]
    fn thought_separator_is_spelled_after_every_thought() {
        let schema = cfg(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }))
        .format_schema();
        let opts = OutputConfigOptions {
            thought_separator: Some("\n\n".into()),
            ..Default::default()
        };
        let body = r#"{"x":1}"#;

        // Deferred: the grammar starts right after the `</think>` trigger.
        let deferred = build_json_only_grammar_source(&schema, &opts);
        assert!(accepts(&deferred, &format!("\n\n{body}")));
        for bad in ["", " ", "\n", "\n\n\n"] {
            assert!(!accepts(&deferred, &format!("{bad}{body}")), "{bad:?}");
        }
        // Without the separator: the single permissive byte, unchanged.
        let loose = build_json_only_grammar_source(
            &schema,
            &OutputConfigOptions::default(),
        );
        assert!(accepts(&loose, &format!("\n{body}")));
        assert!(!accepts(&loose, &format!("\n\n{body}")));

        // Unified, pre-opened (Qwen's `<think>\n` scaffold).
        let pre = build_grammar_source(&schema, &opts, true);
        assert!(accepts(&pre, &format!("hmm\n</think>\n\n{body}")));
        assert!(!accepts(&pre, &format!("hmm\n</think>\n{body}")));

        // Unified, optional thought: forced after one, free without.
        let optional = build_grammar_source(&schema, &opts, false);
        assert!(accepts(&optional, &format!("<think>hmm</think>\n\n{body}")));
        assert!(!accepts(&optional, &format!("<think>hmm</think>{body}")));
        assert!(accepts(&optional, body));
    }

    #[test]
    fn flat_schema_allows_thought_by_default() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }));
        let src = build_grammar_source(
            &config.format_schema(),
            &OutputConfigOptions::default(),
            false,
        );
        assert!(accepts(&src, r#"<think>hmm</think> {"x":1}"#));
        assert!(accepts(&src, r#"{"x":1}"#));
    }

    #[test]
    fn allow_thought_false_rejects_prefix() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }));
        let src = build_grammar_source(
            &config.format_schema(),
            &OutputConfigOptions {
                allow_thought: false,
                phase_split: false,
                ..Default::default()
            },
            false,
        );
        assert!(accepts(&src, r#"{"x":1}"#));
        assert!(!accepts(&src, r#"<think>hmm</think> {"x":1}"#));
    }

    #[test]
    fn grammar_for_prompt_none_when_output_config_unset() {
        let prompt = Prompt::default();
        let mode =
            grammar_for_prompt(&prompt, &OutputConfigOptions::default(), false)
                .expect("compile");
        assert!(mode.is_none());
    }

    #[test]
    fn grammar_for_prompt_some_when_output_config_set() {
        let prompt = Prompt::default().json_schema(json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }));
        let mode =
            grammar_for_prompt(&prompt, &OutputConfigOptions::default(), false)
                .expect("compile");
        assert!(mode.is_some());
    }

    #[test]
    fn compile_output_config_defers_when_phase_split_and_allow_thought() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }));
        let compiled = compile_output_config(
            &config,
            &OutputConfigOptions::default(),
            false,
        )
        .expect("compile");
        let CompiledOutputConfig::Deferred(deferred) = compiled else {
            panic!("expected Deferred variant on default options");
        };
        assert_eq!(deferred.activate_after, vec![b"</think>".to_vec()]);
        // JSON-only grammar accepts bare JSON…
        let state = deferred.grammar;
        let source = state.source().to_string();
        assert!(source.contains("output_schema"));
        assert!(
            !source.contains("think_body"),
            "phase-split grammar must omit thought rules: {source}"
        );
        // …and indeed parses bare JSON as a sanity check.
        assert!(accepts(&source, r#"{"ok":true}"#));
    }

    #[test]
    fn compile_output_config_single_when_phase_split_off() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }));
        let opts = OutputConfigOptions {
            allow_thought: true,
            phase_split: false,
            ..Default::default()
        };
        let compiled =
            compile_output_config(&config, &opts, false).expect("compile");
        let CompiledOutputConfig::Single(SamplingMode::Grammar(state)) =
            compiled
        else {
            panic!("expected Single(Grammar) variant");
        };
        let source = state.source().to_string();
        assert!(source.contains("think_body"));
    }

    #[test]
    fn compile_prompt_single_when_thinking_disabled() {
        // With `phase_split: true` (the default) but no thinking on
        // the prompt, `compile_prompt_output_config` auto-disables
        // phase_split — otherwise the deferred JSON grammar would
        // wait forever for `</think>` and the model would produce
        // unconstrained output.
        let prompt = Prompt::default().json_schema(json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }));
        assert!(prompt.thinking.is_none(), "precondition");
        let compiled = compile_prompt_output_config(
            &prompt,
            &OutputConfigOptions::default(),
            false,
        )
        .expect("compile")
        .expect("output_config set");
        assert!(
            matches!(
                compiled,
                CompiledOutputConfig::Single(SamplingMode::Grammar(_))
            ),
            "expected Single variant when thinking is disabled on prompt"
        );
    }

    #[test]
    fn compile_prompt_deferred_when_thinking_enabled() {
        // When thinking IS enabled on the prompt, Session-level
        // phase_split=true is honored and the grammar is deferred.
        use misanthropic::prompt::thinking::Thinking;
        use std::num::NonZeroU32;
        let prompt = Prompt::default()
            .json_schema(json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
            }))
            .thinking(Thinking::Enabled {
                budget_tokens: NonZeroU32::new(1024).unwrap(),
                display: None,
            });
        assert!(prompt.thinking.is_some(), "precondition");
        let compiled = compile_prompt_output_config(
            &prompt,
            &OutputConfigOptions::default(),
            false,
        )
        .expect("compile")
        .expect("output_config set");
        assert!(
            matches!(compiled, CompiledOutputConfig::Deferred(_)),
            "expected Deferred variant when thinking is enabled and \
             phase_split is on"
        );
    }

    #[test]
    fn compile_output_config_single_when_allow_thought_off() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
        }));
        let opts = OutputConfigOptions {
            allow_thought: false,
            phase_split: true, // ignored since allow_thought is off
            ..Default::default()
        };
        let compiled =
            compile_output_config(&config, &opts, false).expect("compile");
        assert!(matches!(
            compiled,
            CompiledOutputConfig::Single(SamplingMode::Grammar(_))
        ));
    }

    /// #107: under a pre-opened render the root must anchor
    /// close-first — the JSON alone is *incomplete* (the open tag must
    /// close), and the opener literal must not appear in the source
    /// (offering it at position 0 is what forced the duplicate).
    /// Byte-spelled opener text inside the body remains grammar-legal
    /// (body chars are free); the emit-side opener ban owns the
    /// single-token form.
    #[test]
    fn pre_opened_root_closes_first_and_omits_opener() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }));
        let src = build_grammar_source(
            &config.format_schema(),
            &OutputConfigOptions::default(),
            true,
        );
        assert!(
            !src.contains(r#""<think>""#),
            "pre-opened root must not spell the opener: {src}"
        );
        assert!(accepts(&src, "hmm</think> {\"x\":1}"));
        // Empty thought body: closing immediately is legal.
        assert!(accepts(&src, "</think>{\"x\":1}"));
        // Bare JSON is incomplete — the open tag never closed.
        assert!(!accepts(&src, "{\"x\":1}"));
    }

    /// #107: pre-opened dominates `allow_thought = false` — a caller
    /// cannot forbid a thought the render already started (the tool
    /// grammars' `EagerThoughtPreOpened` precedent).
    #[test]
    fn pre_opened_dominates_allow_thought_off() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }));
        let src = build_grammar_source(
            &config.format_schema(),
            &OutputConfigOptions {
                allow_thought: false,
                phase_split: false,
                ..Default::default()
            },
            true,
        );
        assert!(accepts(&src, "hmm</think> {\"x\":1}"));
        assert!(!accepts(&src, "{\"x\":1}"));
    }

    /// Harmony framing: the deferred grammar waits for the final
    /// channel (never `</think>`, which gpt-oss does not write) and
    /// constrains the rest of its header; the unified grammar admits at
    /// most one analysis block — none with `allow_thought` off — and no
    /// commentary detour.
    #[test]
    fn harmony_framing_puts_the_body_in_the_final_channel() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }));
        let harmony = OutputConfigOptions {
            framing: ResponseFraming::Harmony,
            ..Default::default()
        };
        let CompiledOutputConfig::Deferred(d) =
            compile_output_config(&config, &harmony, false).unwrap()
        else {
            panic!("thinking on defers");
        };
        assert_eq!(d.activate_after, vec![b"<|channel|>final".to_vec()]);
        assert!(!d.feed_trigger);
        let tail = d.grammar.source();
        assert!(accepts(tail, r#"<|message|>{"x":1}"#));
        assert!(accepts(tail, r#" <|constrain|>json<|message|>{"x":1}"#));
        assert!(!accepts(tail, r#"<|message|> {"x":1}"#));
        assert!(!accepts(tail, r#"<|message|>{"y":1}"#));

        let analysis = "<|channel|>analysis<|message|>hmm<|end|>\
                        <|start|>assistant";
        let body = r#"<|channel|>final<|message|>{"x":1}"#;
        let unified =
            build_grammar_source(&config.format_schema(), &harmony, false);
        assert!(accepts(&unified, body));
        assert!(accepts(&unified, &format!("{analysis}{body}")));
        assert!(!accepts(&unified, &format!("{analysis}{analysis}{body}")));
        let preamble = "<|channel|>commentary<|message|>hi<|end|>\
                        <|start|>assistant";
        assert!(!accepts(&unified, &format!("{preamble}{body}")));
        assert!(!accepts(&unified, r#"{"x":1}"#));
        let no_thought = build_grammar_source(
            &config.format_schema(),
            &OutputConfigOptions {
                allow_thought: false,
                ..harmony.clone()
            },
            false,
        );
        assert!(accepts(&no_thought, body));
        assert!(!accepts(&no_thought, &format!("{analysis}{body}")));
    }

    /// Small helper so the tests above don't each re-extract the
    /// inner schema value.
    trait OutputConfigSchemaExt {
        fn format_schema(&self) -> serde_json::Value;
    }

    impl OutputConfigSchemaExt for OutputConfig {
        fn format_schema(&self) -> serde_json::Value {
            match &self.format {
                Some(OutputFormat::JsonSchema(f)) => f.schema.clone(),
                _ => serde_json::Value::Null,
            }
        }
    }
}
