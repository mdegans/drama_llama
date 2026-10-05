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
//! the thought-then-JSON shape above, its thought spelled with the
//! dialect's own markers ([`OutputConfigOptions::thought_open`] /
//! [`OutputConfigOptions::thought_close`]: `<think>…</think>`, Gemma 4's
//! `<|channel>thought…<channel|>`, Mistral 4's `[THINK]…[/THINK]`);
//! [`ResponseFraming::Harmony`] puts the body in gpt-oss's final
//! channel. A grammar framed for the wrong format does not merely
//! constrain badly — a phase-split trigger the model never writes
//! leaves the body unconstrained (2026-10-01: gpt-oss returned invalid
//! JSON with a 200, because `</think>` never fired; Gemma 4 and
//! Mistral 4 never write it either).
//!
//! [`OutputConfig`]: misanthropic::prompt::output::OutputConfig
//! [`Prompt`]: crate::Prompt
//! [`SamplingMode`]: crate::SamplingMode

use std::fmt::Write;

use misanthropic::prompt::output::{OutputConfig, OutputFormat};

use crate::dialect::harmony;
use crate::grammar_compile::{
    emit_until_rules, escape_for_gbnf_string, schema_to_gbnf, JSON_GRAMMAR,
};
use crate::{DeferredGrammar, GrammarError, Prompt, SamplingMode};

/// The default [`OutputConfigOptions::thought_close`], and so the default
/// [`ResponseFraming::Bare`] phase-split trigger: the `</think>` that
/// cogito, Qwen and DeepSeek-R1 close their thoughts with.
pub const THINK_CLOSE_TRIGGER: &[u8] = b"</think>";

/// The default [`OutputConfigOptions::thought_open`].
pub const THINK_OPEN: &str = "<think>";

/// The most whitespace the gap after a thought admits besides the
/// measured separator (see [`OutputConfigOptions::thought_separator`]):
/// enough for `""`, `" "`, `"\n"` and `"\n\n"`.
pub const THOUGHT_GAP_MAX: usize = 2;

/// The [`ResponseFraming::Harmony`] phase-split trigger for a final
/// that opens the turn: the final channel's header up to the channel
/// name. What follows it — `<|message|>`, plain — is the grammar's.
pub const HARMONY_FINAL_TRIGGER: &str = "<|channel|>final";

/// The [`ResponseFraming::Harmony`] phase-split trigger after the
/// turn's first block (the analysis, as a rule): its close and the next
/// message's start. What follows it is the grammar's — the final's
/// header (`<|channel|>final`, an optional ` <|constrain|>json`, then
/// `<|message|>`) and nothing else: no commentary preamble and no call
/// after the analysis, as the unified grammar admits none, so the
/// answer is the final alone. Ends earlier than
/// [`HARMONY_FINAL_TRIGGER`] there, so it fires first.
pub const HARMONY_NEXT_BLOCK_TRIGGER: &str = "<|end|><|start|>assistant";

/// How the chat format frames a structured response — where the JSON
/// body starts, and what reasoning may precede it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ResponseFraming {
    /// The body follows an optional thought (cogito, Qwen and
    /// DeepSeek-R1's `<think>…</think>`, or the dialect's own markers —
    /// see [`OutputConfigOptions::thought_close`]) or starts the
    /// response.
    #[default]
    Bare,
    /// OpenAI Harmony (gpt-oss): the generation prompt ends at
    /// `<|start|>assistant`, so the body lives in the final channel —
    /// `<|channel|>final<|message|>{…}`, or after another block with the
    /// ` <|constrain|>json` gpt-oss writes for JSON — after at most one
    /// analysis block (closed by `<|end|>`, reopened by
    /// `<|start|>assistant`). A commentary preamble or a call after
    /// that block is refused: the answer is one text block.
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
    /// triggered by the thought's close ([`Self::thought_close`], or
    /// Harmony's final channel) instead of a single unified grammar. The
    /// caller (typically `TokenPredictor`) runs unconstrained during the
    /// thought preamble and only activates the JSON grammar once the
    /// trigger fires — which restores pure-inference tok/s during the
    /// otherwise-permissive `<think>` body. Defaults to `true`; flip off
    /// to keep the old unified-grammar behaviour (useful for callers that
    /// need the matcher to also guard the thought structure itself).
    ///
    /// Honoured only where the trigger is certain to come: a render that
    /// pre-opened the thought (the model must close it) or
    /// [`ResponseFraming::Harmony`] (the body lives in the final
    /// channel). A [`ResponseFraming::Bare`] thought the render did not
    /// open is optional, and a model that answers without one never
    /// writes the trigger — its body would run unconstrained, a refusal
    /// on every draw for a greedy or seeded caller. Those calls get the
    /// unified `( thought | ws ) body` grammar, which steers instead.
    pub phase_split: bool,
    /// The bytes the template renders between `</think>` and the JSON
    /// body — a fact about the template, not a preference: `Session`
    /// fills it from the dialect's measured
    /// [`ReasoningSyntax::separator`](crate::dialect::ReasoningSyntax::separator)
    /// on every call. The gap after a thought always admits it, plus
    /// any other run of up to [`THOUGHT_GAP_MAX`] whitespace bytes:
    /// bounded, so the model cannot idle in whitespace, but permissive,
    /// because a literal gap masks the spellings a model really writes
    /// (`[/THINK]\n{` against Mistral 4's measured empty gap), and a
    /// token that closes the thought *and* carries the gap
    /// (`>\n\n` after `</`) wakes a deferred body grammar mid-token —
    /// where a refused gap ends the turn instead of steering it. A
    /// single-byte gap could not express Qwen's `\n\n` (#112).
    pub thought_separator: Option<String>,
    /// Where the JSON body sits in the response — a fact about the chat
    /// format, filled by `Session` from its dialect on every call, like
    /// [`Self::thought_separator`].
    pub framing: ResponseFraming,
    /// How a [`ResponseFraming::Bare`] thought opens (default `<think>`)
    /// — the dialect's measured
    /// [`ReasoningSyntax::start`](crate::dialect::ReasoningSyntax::start)
    /// when it has one, filled by `Session` like [`Self::thought_separator`].
    /// Empty: the format only closes thoughts. Unused by
    /// [`ResponseFraming::Harmony`], whose channels are its own.
    pub thought_open: String,
    /// How a [`ResponseFraming::Bare`] thought closes (default
    /// `</think>`), whitespace-trimmed as the parser reads it: the
    /// dialect's [`ReasoningSyntax::end`](crate::dialect::ReasoningSyntax::end)
    /// when it has one, filled by `Session`. It is also the phase-split
    /// trigger, so it must be the closer the model writes — `</think>`
    /// for a Gemma 4 or Mistral 4 left their bodies unconstrained. Empty
    /// stands for the default.
    pub thought_close: String,
    /// The most the schema may measure, checked before anything
    /// compiles it ([`OutputConfigError::SchemaBudget`]). Default
    /// [`SchemaLimits::default`](crate::SchemaLimits::default); `Session`
    /// fills in its own
    /// ([`Session::with_schema_limits`](crate::Session::with_schema_limits)).
    pub schema_limits: crate::SchemaLimits,
}

impl Default for OutputConfigOptions {
    fn default() -> Self {
        Self {
            allow_thought: true,
            phase_split: true,
            thought_separator: None,
            framing: ResponseFraming::Bare,
            thought_open: THINK_OPEN.to_string(),
            thought_close: String::from_utf8_lossy(THINK_CLOSE_TRIGGER)
                .into_owned(),
            schema_limits: crate::SchemaLimits::default(),
        }
    }
}

impl OutputConfigOptions {
    /// Emit `thought_gap`, the whitespace that follows a closed
    /// thought: up to [`THOUGHT_GAP_MAX`] bytes of it, or the measured
    /// separator when that is longer or spelled outside `[ \t\n\r]`.
    fn emit_thought_gap(&self, out: &mut String) {
        let ws = r"[ \t\n\r]";
        let bounded = (0..THOUGHT_GAP_MAX)
            .fold(String::new(), |inner, _| format!("( {ws} {inner})? "));
        let measured = self.thought_separator.as_deref().filter(|sep| {
            sep.len() > THOUGHT_GAP_MAX
                || !sep.bytes().all(|b| b" \t\n\r".contains(&b))
        });
        let _ = match measured {
            Some(sep) => writeln!(
                out,
                r#"thought_gap ::= {bounded}| "{}""#,
                escape_for_gbnf_string(sep)
            ),
            None => writeln!(out, "thought_gap ::= {}", bounded.trim_end()),
        };
    }

    /// [`Self::thought_close`], the default standing in for an empty
    /// one: a thought must close on *something*.
    fn close(&self) -> &str {
        match self.thought_close.as_str() {
            "" => "</think>",
            close => close,
        }
    }

    /// Emit `thought_close`: a thought's body through [`Self::close`],
    /// which it contains nowhere else — the shape the tool grammars give
    /// a thought (`dialect::emit`).
    fn emit_thought_close(&self, out: &mut String) {
        emit_until_rules("thought_close", self.close(), out);
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
    let source = build_grammar_source(schema, opts, thought_pre_opened)?;
    Ok(SamplingMode::grammar(&source)?)
}

/// Compile an [`OutputConfig`] into a [`CompiledOutputConfig`] that either
/// holds a single unified grammar or a thought-close-triggered
/// [`DeferredGrammar`], depending on `opts.phase_split` and
/// `opts.allow_thought`. Phase-split applies only when both are `true`
/// and the trigger is certain: `thought_pre_opened`, or
/// [`ResponseFraming::Harmony`] (see [`OutputConfigOptions::phase_split`]).
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
    // A deferred body waits for a trigger the model is sure to write:
    // the closer of a thought the render opened, or Harmony's final
    // channel. An optional Bare thought is not that — see
    // `OutputConfigOptions::phase_split`.
    let trigger_certain =
        thought_pre_opened || opts.framing == ResponseFraming::Harmony;
    if opts.phase_split && opts.allow_thought && trigger_certain {
        let source = build_json_only_grammar_source(schema, opts)?;
        let (triggers, feed_trigger) = match opts.framing {
            // The JSON-body grammar starts *after* the trigger, which
            // itself stays outside the constrained span.
            ResponseFraming::Bare => (vec![opts.close()], false),
            // The header grammar reads the trigger: after the first
            // block only the final, which may carry the constraint (see
            // `emit_harmony_final_header`).
            ResponseFraming::Harmony => (
                vec![HARMONY_NEXT_BLOCK_TRIGGER, HARMONY_FINAL_TRIGGER],
                true,
            ),
        };
        Ok(CompiledOutputConfig::Deferred(DeferredGrammar {
            grammar: crate::CompiledGrammar::parse(&source)?,
            activate_after: triggers
                .iter()
                .map(|t| t.as_bytes().to_vec())
                .collect(),
            feed_trigger,
        }))
    } else {
        let source = build_grammar_source(schema, opts, thought_pre_opened)?;
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
pub(crate) fn structured(prompt: &Prompt) -> Option<&OutputConfig> {
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
) -> Result<String, OutputConfigError> {
    crate::schema_budget::check_schemas([], Some(schema), &opts.schema_limits)?;
    let mut src = String::with_capacity(512);

    if opts.framing == ResponseFraming::Harmony {
        // gpt-oss never pre-opens its analysis channel (the generation
        // prompt ends at `<|start|>assistant`), so `thought_pre_opened`
        // has nothing to anchor here. At most one analysis block, never
        // `(analysis | commentary)*`: every extra block the grammar
        // admits is somewhere a model can go instead of answering
        // (see the forced-call grammar in `dialect::emit`).
        // The constraint only after the analysis, which records it (see
        // `emit_harmony_final_header`).
        let header = if opts.allow_thought {
            "( h_analysis h_final | h_final_plain )"
        } else {
            "h_final_plain"
        };
        let _ = writeln!(src, "root ::= {header} output_schema");
        let _ = writeln!(
            src,
            r#"h_analysis ::= "{open}" h_end "{start}""#,
            open = escape_for_gbnf_string(harmony::ANALYSIS_OPEN),
            start = escape_for_gbnf_string(harmony::START_ASSISTANT),
        );
        emit_until_rules("h_end", harmony::END, &mut src);
        emit_harmony_final_header(
            "h_final",
            HARMONY_FINAL_TRIGGER,
            true,
            &mut src,
        );
        emit_harmony_final_header(
            "h_final_plain",
            HARMONY_FINAL_TRIGGER,
            false,
            &mut src,
        );
    } else if thought_pre_opened {
        // The template already emitted the opener, so the tag IS open:
        // the thought is mandatory (close-first), the opener literal
        // must not appear (it would force a duplicate — #107), and
        // this dominates `allow_thought = false`, mirroring the tool
        // grammars' `EagerThoughtPreOpened` precedent — a caller
        // cannot forbid a thought the render already started.
        let _ =
            writeln!(src, "root ::= thought_close thought_gap output_schema");
        opts.emit_thought_close(&mut src);
        opts.emit_thought_gap(&mut src);
    } else if opts.allow_thought {
        let open = match opts.thought_open.as_str() {
            "" => String::new(),
            open => format!(r#""{}" "#, escape_for_gbnf_string(open)),
        };
        let _ = writeln!(
            src,
            "root ::= ( {open}thought_close thought_gap | ws ) output_schema"
        );
        opts.emit_thought_close(&mut src);
        opts.emit_thought_gap(&mut src);
    } else {
        let _ = writeln!(src, "root ::= ws output_schema");
    }

    schema_to_gbnf(schema, "output_schema", &mut src)?;
    src.push_str(JSON_GRAMMAR);
    Ok(src)
}

/// Emit the JSON-only grammar used by the deferred / phase-split path.
/// Root starts right after the thought-close trigger, so it opens with the
/// thought separator; thought rules are omitted entirely because
/// `TokenPredictor` doesn't run the matcher during the thought preamble.
pub(crate) fn build_json_only_grammar_source(
    schema: &serde_json::Value,
    opts: &OutputConfigOptions,
) -> Result<String, OutputConfigError> {
    crate::schema_budget::check_schemas([], Some(schema), &opts.schema_limits)?;
    let mut src = String::with_capacity(512);
    match opts.framing {
        ResponseFraming::Bare => {
            let _ = writeln!(src, "root ::= thought_gap output_schema");
            opts.emit_thought_gap(&mut src);
        }
        ResponseFraming::Harmony => {
            // The grammar reads the trigger: after the first block the
            // final comes next — no preamble, no call, no second
            // analysis — and its header may carry the constraint;
            // opening the turn it may not (see
            // `emit_harmony_final_header`).
            let _ =
                writeln!(src, "root ::= ( h_after | h_first ) output_schema");
            emit_harmony_final_header(
                "h_after",
                &format!("{HARMONY_NEXT_BLOCK_TRIGGER}{HARMONY_FINAL_TRIGGER}"),
                true,
                &mut src,
            );
            emit_harmony_final_header(
                "h_first",
                HARMONY_FINAL_TRIGGER,
                false,
                &mut src,
            );
        }
    }
    schema_to_gbnf(schema, "output_schema", &mut src)?;
    src.push_str(JSON_GRAMMAR);
    Ok(src)
}

/// The Harmony final-channel header, from `lead` (the header up to the
/// channel name, and whatever precedes it the rule must spell) through
/// `<|message|>`. With `constrain`, the ` <|constrain|>json` gpt-oss
/// writes for JSON is optional: both spellings are the model's, and the
/// parser reads both and records which in the analysis block before
/// the final, so the re-render spells the one written. A final with no
/// analysis before it has nowhere to record it and re-renders plain,
/// so there the rule is plain: a constrained final the grammar let
/// through would lose its turn's tip on the next request. The body
/// follows `<|message|>` directly, as the template renders it.
///
/// The deferred grammar cannot see past its trigger, which ends a
/// preamble as it does an analysis (`<|end|><|start|>assistant`): a
/// preamble that opens the turn, before the trigger, runs free, and a
/// constrained final after it still parts there. Such a turn is two
/// text blocks, which the session refuses as a schema violation
/// whatever its header (pinned in the session's gpt-oss tests).
fn emit_harmony_final_header(
    rule: &str,
    lead: &str,
    constrain: bool,
    out: &mut String,
) {
    let constraint = match constrain {
        true => format!(
            r#"( " {}json" )? "#,
            escape_for_gbnf_string(harmony::CONSTRAIN)
        ),
        false => String::new(),
    };
    let _ = writeln!(
        out,
        r#"{rule} ::= "{lead}" {constraint}"{msg}""#,
        lead = escape_for_gbnf_string(lead),
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
    /// The JSON Schema has no grammar: too complex, or unsatisfiable.
    #[error("output_config.format.schema: {0}")]
    Schema(#[from] crate::grammar_compile::SchemaError),
    /// The schema measures past [`OutputConfigOptions::schema_limits`],
    /// so nothing compiled it: the request's fault, a 400.
    #[error("schema limits: {0}")]
    SchemaBudget(#[from] crate::SchemaBudgetError),
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

    /// Whether `source` refuses `input` before its end: no continuation
    /// of it could complete.
    fn refuses(source: &str, input: &str) -> bool {
        let grammar = Grammar::parse(source).expect("grammar parses");
        GrammarState::new(Arc::new(grammar))
            .advance_bytes(input.as_bytes())
            .is_err()
    }

    fn cfg(schema: serde_json::Value) -> OutputConfig {
        OutputConfig::json_schema(schema)
    }

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

    /// The gap after a thought, in every root that follows one: the
    /// measured separator (Qwen's `\n\n`, #112 — once unreachable
    /// behind a single-byte `ws`) and the spellings a model really
    /// writes beside it (`[/THINK]\n{` against Mistral 4's measured
    /// empty gap), but bounded, so whitespace cannot run on.
    #[test]
    fn thought_gap_is_bounded_whitespace() {
        let schema = cfg(json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }))
        .format_schema();
        let body = r#"{"x":1}"#;
        let good = ["", " ", "\n", "\n\n", " \n", "\r\n"];
        let bad = ["\n\n\n", "   ", "\n \n"];
        for sep in [None, Some(""), Some("\n\n")] {
            let opts = OutputConfigOptions {
                thought_separator: sep.map(str::to_string),
                ..Default::default()
            };
            let deferred =
                build_json_only_grammar_source(&schema, &opts).unwrap();
            let pre = build_grammar_source(&schema, &opts, true).unwrap();
            let optional = build_grammar_source(&schema, &opts, false).unwrap();
            for gap in good {
                let at = format!("{sep:?}, {gap:?}");
                assert!(accepts(&deferred, &format!("{gap}{body}")), "{at}");
                assert!(
                    accepts(&pre, &format!("hmm\n</think>{gap}{body}")),
                    "{at}"
                );
                assert!(
                    accepts(
                        &optional,
                        &format!("<think>hmm</think>{gap}{body}")
                    ),
                    "{at}"
                );
            }
            for gap in bad {
                let at = format!("{sep:?}, {gap:?}");
                assert!(!accepts(&deferred, &format!("{gap}{body}")), "{at}");
                assert!(
                    !accepts(&pre, &format!("hmm\n</think>{gap}{body}")),
                    "{at}"
                );
            }
            // Without a thought, the body's own single `ws` as before.
            assert!(accepts(&optional, body));
            assert!(!accepts(&optional, &format!("\n\n{body}")));
        }
        // A measured gap the bound cannot express stays reachable.
        let long = OutputConfigOptions {
            thought_separator: Some("\n\n\n".into()),
            ..Default::default()
        };
        let deferred = build_json_only_grammar_source(&schema, &long).unwrap();
        assert!(accepts(&deferred, &format!("\n\n\n{body}")));
        assert!(accepts(&deferred, &format!("\n{body}")));
        assert!(!accepts(&deferred, &format!("\n\n\n\n{body}")));
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
        )
        .unwrap();
        assert!(accepts(&src, r#"<think>hmm</think> {"x":1}"#));
        assert!(accepts(&src, r#"{"x":1}"#));
    }

    /// A recursive `$ref` (a tree, as schemars emits one) compiles as
    /// a structured-output schema — unified and deferred grammars both
    /// admit a valid three-level tree and refuse an invalid one, as
    /// the schema check does — where it used to overflow the stack.
    #[test]
    fn recursive_ref_schema_compiles() {
        let config = cfg(json!({
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
            "required": ["root"],
            "$defs": {"Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Node"},
                    },
                },
            }},
        }));
        let schema = config.format_schema();
        assert!(schema.pointer("/$defs/Node").is_some(), "{schema}");
        let opts = OutputConfigOptions::default();
        let unified = build_grammar_source(&schema, &opts, false).unwrap();
        let deferred = build_json_only_grammar_source(&schema, &opts).unwrap();
        let valid = r#"{"root":{"name":"a","children":[{"name":"b","children":[{"name":"c","children":[]}]},{"name":"d"}]}}"#;
        let invalid = r#"{"root":{"name":"a","children":[{"name":"b","children":[{"name":3}]}]}}"#;
        assert!(accepts(&unified, valid));
        assert!(accepts(&unified, &format!("<think>hmm</think>{valid}")));
        assert!(accepts(&deferred, valid));
        assert!(crate::schema_check::check_text(&schema, valid).is_ok());
        for src in [&unified, &deferred] {
            assert!(!accepts(src, invalid));
        }
        assert!(crate::schema_check::check_text(&schema, invalid).is_err());
        assert!(grammar_for_output_config(&config, &opts, false).is_ok());
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
        )
        .unwrap();
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
            true,
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
            !source.contains("thought_close"),
            "phase-split grammar must omit thought rules: {source}"
        );
        // …and indeed parses bare JSON as a sanity check.
        assert!(accepts(&source, r#"{"ok":true}"#));
    }

    /// A Bare thought the render did not open is optional, so a model
    /// may answer without the trigger: a deferred body would then run
    /// unconstrained and refuse every greedy draw. Unified instead, the
    /// grammar steers a thought-less answer into the schema.
    #[test]
    fn compile_output_config_unified_when_thought_is_optional() {
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
        let CompiledOutputConfig::Single(SamplingMode::Grammar(state)) =
            compiled
        else {
            panic!("an optional thought must not defer the body");
        };
        let source = state.source();
        assert!(accepts(source, r#"{"ok":true}"#));
        assert!(accepts(source, "<think>hmm</think>\n\n{\"ok\":true}"));
        assert!(!accepts(source, r#"{"ok":1}"#));
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
        assert!(source.contains("thought_close"));
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
        // When thinking IS enabled on the prompt and the render opened
        // the thought, Session-level phase_split=true is honored and
        // the grammar is deferred.
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
            true,
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
        )
        .unwrap();
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
        )
        .unwrap();
        assert!(accepts(&src, "hmm</think> {\"x\":1}"));
        assert!(!accepts(&src, "{\"x\":1}"));
    }

    /// Harmony framing: the deferred grammar waits for the final
    /// channel or the end of the turn's first block (never `</think>`,
    /// which gpt-oss does not write); the unified grammar admits at most
    /// one analysis block — none with `allow_thought` off. Neither
    /// admits a commentary detour after it, a preamble or a call: the
    /// answer is the final alone. Either admits ` <|constrain|>json`
    /// only where the parse can record it: after another block. A final
    /// opening the turn is plain, so it re-renders as written.
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
        let (next, first) = ("<|end|><|start|>assistant", "<|channel|>final");
        let after = format!("{next}{first}");
        assert_eq!(
            d.activate_after,
            vec![next.as_bytes().to_vec(), first.as_bytes().to_vec()]
        );
        assert!(d.feed_trigger, "the header grammar reads its trigger");
        let tail = d.grammar.source();
        let body = |header: &str, value: &str| format!("{header}{value}");
        let constrained = " <|constrain|>json<|message|>";
        assert!(accepts(tail, &body(&after, r#"<|message|>{"x":1}"#)));
        assert!(accepts(
            tail,
            &body(&after, &format!(r#"{constrained}{{"x":1}}"#))
        ));
        assert!(accepts(tail, &body(first, r#"<|message|>{"x":1}"#)));
        assert!(
            !accepts(tail, &body(first, &format!(r#"{constrained}{{"x":1}}"#))),
            "opening the turn, a constrained final would re-render plain"
        );
        assert!(!accepts(tail, &body(&after, r#"<|message|> {"x":1}"#)));
        assert!(!accepts(tail, &body(&after, r#"<|message|>{"y":1}"#)));
        // After the first block the final, and nothing else, comes next.
        for detour in [
            "<|channel|>commentary<|message|>Hi.",
            "<|channel|>commentary to=functions.f <|constrain|>json\
             <|message|>{}",
            " to=functions.f<|channel|>commentary json<|message|>{}",
            "<|channel|>analysis<|message|>More.",
        ] {
            let text = format!("{next}{detour}");
            assert!(refuses(tail, &text), "{text}");
        }
        // After a block the trigger ends before the final's header, so
        // it fires first, and the grammar reads it.
        let text = format!("<|channel|>analysis<|message|>hmm{next}");
        let fired = crate::predictor::find_any_deferred_trigger_end(
            text.as_bytes(),
            &d.activate_after,
            text.len(),
            |_, _| true,
        );
        assert_eq!(fired, Some((text.len(), next.len())));

        let analysis = "<|channel|>analysis<|message|>hmm<|end|>\
                        <|start|>assistant";
        let body = r#"<|channel|>final<|message|>{"x":1}"#;
        let json_body =
            r#"<|channel|>final <|constrain|>json<|message|>{"x":1}"#;
        let unified =
            build_grammar_source(&config.format_schema(), &harmony, false)
                .unwrap();
        assert!(accepts(&unified, body));
        assert!(accepts(&unified, &format!("{analysis}{body}")));
        assert!(accepts(&unified, &format!("{analysis}{json_body}")));
        assert!(!accepts(&unified, json_body), "nowhere to record it");
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
        )
        .unwrap();
        assert!(accepts(&no_thought, body));
        assert!(!accepts(&no_thought, json_body));
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
