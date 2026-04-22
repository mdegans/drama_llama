//! `Prompt::output_config` → [`SamplingMode`] compiler.
//!
//! Reads [`OutputConfig`] from a [`Prompt`] and emits a GBNF that
//! forces the model's response to match the configured JSON Schema.
//! Mirrors the [`tool_choice`](crate::tool_choice) module's shape —
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
//! [`OutputConfig`]: misanthropic::prompt::output::OutputConfig
//! [`Prompt`]: crate::Prompt
//! [`SamplingMode`]: crate::SamplingMode

use std::fmt::Write;

use misanthropic::prompt::output::{OutputConfig, OutputFormat};

use crate::grammar_compile::{emit_thought_rules, schema_to_gbnf, JSON_GRAMMAR};
use crate::{DeferredGrammar, GrammarError, Prompt, SamplingMode};

/// Byte sequence that triggers deferred-grammar promotion when
/// [`OutputConfigOptions::phase_split`] is on. Matches the closing tag of
/// the thought preamble emitted by reasoning models.
pub const THINK_CLOSE_TRIGGER: &[u8] = b"</think>";

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
}

impl Default for OutputConfigOptions {
    fn default() -> Self {
        Self {
            allow_thought: true,
            phase_split: true,
        }
    }
}

/// Output of [`compile_output_config`] — either a single unified grammar
/// (run it from the start) or a thought/JSON phase-split pair (run
/// unconstrained until the trigger, then promote the JSON grammar).
#[derive(Clone, Debug)]
pub enum CompiledOutputConfig {
    /// Standard single-grammar shape. Push this into `SampleOptions::modes`.
    Single(SamplingMode),
    /// Phase-split shape. Install into `SampleOptions::deferred_grammar`
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
            Self::Deferred(d) => d.grammar,
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
) -> Result<SamplingMode, OutputConfigError> {
    let schema = match &config.format {
        OutputFormat::JsonSchema(f) => &f.schema,
        _ => return Err(OutputConfigError::UnsupportedFormat),
    };
    let source = build_grammar_source(schema, opts);
    Ok(SamplingMode::grammar(&source)?)
}

/// Compile an [`OutputConfig`] into a [`CompiledOutputConfig`] that either
/// holds a single unified grammar or a `</think>`-triggered
/// [`DeferredGrammar`], depending on `opts.phase_split` and
/// `opts.allow_thought`. Phase-split applies only when both are `true`.
pub fn compile_output_config(
    config: &OutputConfig,
    opts: &OutputConfigOptions,
) -> Result<CompiledOutputConfig, OutputConfigError> {
    let schema = match &config.format {
        OutputFormat::JsonSchema(f) => &f.schema,
        _ => return Err(OutputConfigError::UnsupportedFormat),
    };
    if opts.phase_split && opts.allow_thought {
        let source = build_json_only_grammar_source(schema);
        let grammar = SamplingMode::grammar(&source)?;
        Ok(CompiledOutputConfig::Deferred(DeferredGrammar {
            grammar,
            activate_after: THINK_CLOSE_TRIGGER.to_vec(),
        }))
    } else {
        let source = build_grammar_source(schema, opts);
        Ok(CompiledOutputConfig::Single(SamplingMode::grammar(&source)?))
    }
}

/// Derive the output-config grammar directly from a [`Prompt`]. Reads
/// `prompt.output_config`; returns `Ok(None)` when unset. Legacy entry
/// point — ignores `phase_split`. Use [`compile_prompt_output_config`] for
/// the phase-split-aware shape.
pub fn grammar_for_prompt(
    prompt: &Prompt,
    opts: &OutputConfigOptions,
) -> Result<Option<SamplingMode>, OutputConfigError> {
    let Some(config) = prompt.output_config.as_ref() else {
        return Ok(None);
    };
    Ok(Some(grammar_for_output_config(config, opts)?))
}

/// Phase-split-aware equivalent of [`grammar_for_prompt`]. Returns the
/// compiled output config (either unified grammar or deferred) when
/// `prompt.output_config` is set.
pub fn compile_prompt_output_config(
    prompt: &Prompt,
    opts: &OutputConfigOptions,
) -> Result<Option<CompiledOutputConfig>, OutputConfigError> {
    let Some(config) = prompt.output_config.as_ref() else {
        return Ok(None);
    };
    Ok(Some(compile_output_config(config, opts)?))
}

/// Emit the GBNF source text for an output-config constraint. Kept
/// `pub(crate)` so tests can inspect the grammar text directly.
pub(crate) fn build_grammar_source(
    schema: &serde_json::Value,
    opts: &OutputConfigOptions,
) -> String {
    let mut src = String::with_capacity(512);

    if opts.allow_thought {
        let _ = writeln!(src, "root ::= thought? ws output_schema");
        emit_thought_rules(&mut src);
    } else {
        let _ = writeln!(src, "root ::= ws output_schema");
    }

    schema_to_gbnf(schema, "output_schema", &mut src);
    src.push_str(JSON_GRAMMAR);
    src
}

/// Emit the JSON-only grammar used by the deferred / phase-split path.
/// Root starts at the JSON body (leading whitespace tolerated); thought
/// rules are omitted entirely because `TokenPredictor` doesn't run the
/// matcher during the thought preamble.
pub(crate) fn build_json_only_grammar_source(
    schema: &serde_json::Value,
) -> String {
    let mut src = String::with_capacity(512);
    let _ = writeln!(src, "root ::= ws output_schema");
    schema_to_gbnf(schema, "output_schema", &mut src);
    src.push_str(JSON_GRAMMAR);
    src
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
            },
        );
        assert!(accepts(&src, r#"{"x":1}"#));
        assert!(!accepts(&src, r#"<think>hmm</think> {"x":1}"#));
    }

    #[test]
    fn grammar_for_prompt_none_when_output_config_unset() {
        let prompt = Prompt::default();
        let mode = grammar_for_prompt(&prompt, &OutputConfigOptions::default())
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
        let mode = grammar_for_prompt(&prompt, &OutputConfigOptions::default())
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
        let compiled =
            compile_output_config(&config, &OutputConfigOptions::default())
                .expect("compile");
        let CompiledOutputConfig::Deferred(deferred) = compiled else {
            panic!("expected Deferred variant on default options");
        };
        assert_eq!(deferred.activate_after.as_slice(), b"</think>");
        // JSON-only grammar accepts bare JSON…
        let SamplingMode::Grammar(state) = deferred.grammar else {
            panic!("deferred grammar must be SamplingMode::Grammar");
        };
        let source = state.lock().unwrap().grammar().source().to_string();
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
        };
        let compiled =
            compile_output_config(&config, &opts).expect("compile");
        let CompiledOutputConfig::Single(SamplingMode::Grammar(state)) =
            compiled
        else {
            panic!("expected Single(Grammar) variant");
        };
        let source = state.lock().unwrap().grammar().source().to_string();
        assert!(source.contains("think_body"));
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
        };
        let compiled =
            compile_output_config(&config, &opts).expect("compile");
        assert!(matches!(
            compiled,
            CompiledOutputConfig::Single(SamplingMode::Grammar(_))
        ));
    }

    /// Small helper so the tests above don't each re-extract the
    /// inner schema value.
    trait OutputConfigSchemaExt {
        fn format_schema(&self) -> serde_json::Value;
    }

    impl OutputConfigSchemaExt for OutputConfig {
        fn format_schema(&self) -> serde_json::Value {
            match &self.format {
                OutputFormat::JsonSchema(f) => f.schema.clone(),
                _ => serde_json::Value::Null,
            }
        }
    }
}
