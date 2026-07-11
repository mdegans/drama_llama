//! GBNF emitter + canonical reference renderer driven by
//! [`CallSyntax`].
//!
//! Two outputs from one dialect value, kept consistent by
//! construction:
//!
//! * [`grammar_source`] — the GBNF constraining generation to the
//!   dialect's call shape (Phase D of the tool-dialects plan).
//! * [`render_reference`] — the canonical byte serialization of a set
//!   of calls, i.e. what the chat template's re-render will produce.
//!   Used by the reconstruction harness and Session's
//!   canonicalization check (cache-stability layer 2).
//!
//! ## Argument order: sorted, not declaration order
//!
//! llama.cpp emits required args in schema-declaration order. We
//! deliberately diverge: minijinja alphabetizes object keys on output
//! (and serde_json's default map is a BTreeMap), so the template's
//! re-render of `tool_calls` is *sorted by key* — and round-trip
//! byte-stability pins emission to re-render. Sorting also lets
//! optionals sit in place (`a? b c?` for required `b`) instead of
//! trailing any-order, which closes upstream's duplicate-optional
//! grammar hole entirely.

use std::fmt::Write;

use serde_json::Value;

use crate::grammar_compile::{
    dict_encode_value, emit_dict_value_rules, emit_until_rules,
    escape_for_gbnf_string, schema_to_dict_gbnf, schema_to_gbnf, JSON_GRAMMAR,
};
use crate::Tool;

use super::{harmony, CallSyntax, Family, ReasoningMode};

/// Errors from dialect emission / reference rendering.
#[derive(Debug, thiserror::Error)]
pub enum DialectError {
    /// A raw (tagged) string value contains the dialect's close
    /// delimiter and therefore cannot round-trip: tagged dialects
    /// have no in-band escape the model was trained on, and render
    /// is the template's job. Callers may catch this and substitute
    /// (e.g. replace the input with a warning to the model) — policy
    /// belongs to the consuming app, per the plan amendments.
    #[error(
        "argument {param:?} of tool {tool:?} contains the dialect \
         delimiter {delimiter:?} and cannot be represented"
    )]
    UnrepresentableValue {
        tool: String,
        param: String,
        delimiter: String,
    },
    /// The dialect has no tool-call format (Family::None) — nothing
    /// to emit. The `Instructed` dialect (deferred follow-up to
    /// Phase F) will own this case.
    #[error("dialect has no tool-call format (family = None)")]
    NoToolFormat,
    /// Emitted GBNF failed to compile — an emitter bug or a marker
    /// set the grammar engine can't express. Carries the source for
    /// debugging.
    #[error("emitted grammar failed to compile: {source_err}")]
    Grammar {
        source_err: crate::GrammarError,
        gbnf: String,
    },
}

static_assertions::assert_impl_all!(DialectError: Send, Sync);

/// How the grammar root anchors on the generation prompt — the
/// dialect-generic form of `tool_choice::RootShape`, using the
/// dialect's own reasoning tags rather than hardcoded `<think>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Anchor {
    /// Grammar active from token 0; reasoning block optional
    /// (emitted only when the dialect has reasoning tags).
    Eager,
    /// Grammar active from token 0 and the rendered prompt already
    /// ends with the open reasoning tag: body + close are required
    /// before the call.
    EagerThoughtPreOpened,
    /// Trigger-activated: the root begins exactly at the dialect
    /// trigger ([`CallSyntax::trigger`]); no reasoning prefix.
    Lazy,
}

/// Emission options.
#[derive(Clone, Debug)]
pub struct EmitOptions {
    pub anchor: Anchor,
    /// Allow more than one call per turn. Gated on the caller's
    /// parallel-tool-calls setting.
    pub parallel: bool,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            anchor: Anchor::Eager,
            parallel: false,
        }
    }
}

/// Build the GBNF source constraining generation to `syntax`'s call
/// shape over `tools`.
pub fn grammar_source(
    syntax: &CallSyntax,
    tools: &[&Tool],
    opts: &EmitOptions,
) -> Result<String, DialectError> {
    if syntax.family == Family::None {
        return Err(DialectError::NoToolFormat);
    }
    if syntax.family == Family::Harmony {
        // The channel-block structure doesn't decompose into the
        // generic section/per-call root below — hand-built, like the
        // parser side.
        return harmony_grammar_source(tools, opts);
    }
    let mut src = String::with_capacity(2048);
    let mut until_counter = 0usize;

    // Root.
    let has_reasoning = syntax.reasoning.mode != ReasoningMode::None
        && !syntax.reasoning.end.is_empty();
    match (opts.anchor, has_reasoning) {
        (Anchor::Lazy, _) => {
            let _ = writeln!(src, "root ::= calls");
        }
        (Anchor::EagerThoughtPreOpened, _) => {
            // Close tag required; body is raw-until-close.
            emit_until_rules("thought_close", &syntax.reasoning.end, &mut src);
            let _ = writeln!(src, "root ::= thought_close ws calls");
        }
        (Anchor::Eager, true) => {
            let start_lit = escape_for_gbnf_string(&syntax.reasoning.start);
            emit_until_rules("thought_close", &syntax.reasoning.end, &mut src);
            if syntax.reasoning.start.is_empty() {
                let _ = writeln!(src, "root ::= thought_close? ws calls");
            } else {
                let _ = writeln!(
                    src,
                    r#"root ::= ( "{start_lit}" thought_close )? ws calls"#
                );
            }
        }
        (Anchor::Eager, false) => {
            let _ = writeln!(src, "root ::= ws calls");
        }
    }

    // Section wrapper + call multiplicity. Dialects whose template
    // keeps the call turn open (`tool_response_start` non-empty,
    // Gemma 4) REQUIRE the response opener as the turn exit: it is
    // the model's trained continuation after its last call — masking
    // it makes the model loop emitting more calls — and it is the
    // canonical re-render byte. After the exit nothing else is
    // legal, so the sampler's complete-constraint logic forces EOG:
    // a deterministic stop.
    let exit = if syntax.tool_response_start.is_empty() {
        String::new()
    } else {
        format!(
            r#" "{}""#,
            escape_for_gbnf_string(&syntax.tool_response_start)
        )
    };
    let sec_open = escape_for_gbnf_string(&syntax.section_start);
    let sec_close = escape_for_gbnf_string(&syntax.section_end);
    let call_seq = if opts.parallel { "call+" } else { "call" };
    match (
        syntax.section_start.is_empty(),
        syntax.section_end.is_empty(),
    ) {
        (true, true) => {
            let _ = writeln!(src, "calls ::= {call_seq}{exit}");
        }
        _ => {
            let _ = writeln!(
                src,
                r#"calls ::= "{sec_open}" {call_seq} "{sec_close}"{exit}"#
            );
        }
    }

    // Per-tool alternatives.
    let alts = (0..tools.len())
        .map(|i| format!("call_{i}"))
        .collect::<Vec<_>>()
        .join(" | ");
    let _ = writeln!(src, "call ::= {alts}");

    let per_open = escape_for_gbnf_string(&syntax.per_call_start);
    let per_close = escape_for_gbnf_string(&syntax.per_call_end);

    for (i, tool) in tools.iter().enumerate() {
        match syntax.family {
            Family::TagWithTagged => emit_tagged_call(
                syntax,
                tool,
                i,
                (&per_open, &per_close),
                &mut src,
                &mut until_counter,
            ),
            Family::TagWithJson => {
                emit_tag_json_call(
                    syntax,
                    tool,
                    i,
                    (&per_open, &per_close),
                    &mut src,
                );
            }
            Family::JsonNative => {
                emit_json_native_call(
                    syntax,
                    tool,
                    i,
                    (&per_open, &per_close),
                    &mut src,
                );
            }
            Family::TagWithDict => {
                emit_dict_call(
                    syntax,
                    tool,
                    i,
                    (&per_open, &per_close),
                    &mut src,
                );
            }
            Family::None | Family::Harmony => unreachable!("checked above"),
        }
    }

    if syntax.family == Family::TagWithDict {
        emit_dict_value_rules(&syntax.arguments.string_quote, &mut src);
    }
    src.push_str(JSON_GRAMMAR);
    Ok(src)
}

/// Hand-built Harmony (gpt-oss) grammar.
///
/// Eager (`Any`/`Method`): the generation prompt ends at
/// `<|start|>assistant`, and every Harmony message is a channel
/// block, so the grammar covers the *whole* emission — any number of
/// analysis / commentary-preamble blocks (each closed by `<|end|>`
/// and reopened by `<|start|>assistant`), then the forced call in its
/// canonical trained shape:
/// `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}`.
/// After the args complete nothing further is legal, so the sampler's
/// complete-constraint logic admits only EOG — the model's `<|call|>`
/// (see `extra_eos_tokens`), a deterministic stop that is also the
/// canonical re-render byte.
///
/// Lazy (`Auto`): activated by one of [`CallSyntax::triggers`] (both
/// recipient positions); the root accepts either header shape from
/// the trigger's first byte, with the constraint clause lenient
/// (optional, `<|constrain|>` literal optional, any `[A-Za-z0-9_-]+`
/// type — upstream parity) because the pre-trigger bytes were sampled
/// free and canonical-byte forcing is pointless mid-header.
fn harmony_grammar_source(
    tools: &[&Tool],
    opts: &EmitOptions,
) -> Result<String, DialectError> {
    let mut src = String::with_capacity(2048);
    let start = escape_for_gbnf_string(harmony::START_ASSISTANT);
    let chan = escape_for_gbnf_string(harmony::CHANNEL);
    let msg = escape_for_gbnf_string(harmony::MESSAGE);
    let to_fn = escape_for_gbnf_string(harmony::TO_FUNCTIONS);
    let constrain = escape_for_gbnf_string(harmony::CONSTRAIN);
    let analysis_open = escape_for_gbnf_string(harmony::ANALYSIS_OPEN);
    let commentary_open = escape_for_gbnf_string(harmony::COMMENTARY_OPEN);

    match opts.anchor {
        Anchor::Lazy => {
            let _ = writeln!(src, "root ::= h_role_form | h_chan_form");
            let role_alts = (0..tools.len())
                .map(|i| format!("h_role_{i}"))
                .collect::<Vec<_>>()
                .join(" | ");
            let chan_alts = (0..tools.len())
                .map(|i| format!("h_chan_{i}"))
                .collect::<Vec<_>>()
                .join(" | ");
            let _ = writeln!(
                src,
                r#"h_role_form ::= "{start}{to_fn}" ( {role_alts} )"#
            );
            let _ = writeln!(
                src,
                r#"h_chan_form ::= ( "{chan}commentary{to_fn}" | "{chan}analysis{to_fn}" ) ( {chan_alts} )"#
            );
            let _ = writeln!(
                src,
                r#"h_channel ::= "{chan}" ( "commentary" | "analysis" )"#
            );
            let _ = writeln!(
                src,
                r#"h_constraint ::= " " ( "{constrain}" )? h_ctype"#
            );
            let _ = writeln!(src, "h_ctype ::= [A-Za-z0-9_-]+");
            for (i, tool) in tools.iter().enumerate() {
                let name_lit = escape_for_gbnf_string(tool.name.as_ref());
                // Recipient in the role header: the channel clause
                // follows the name.
                let _ = writeln!(
                    src,
                    r#"h_role_{i} ::= "{name_lit}" h_channel h_constraint? "{msg}" h_args_{i}"#
                );
                // Recipient in the channel header: the channel was
                // consumed by the trigger.
                let _ = writeln!(
                    src,
                    r#"h_chan_{i} ::= "{name_lit}" h_constraint? "{msg}" h_args_{i}"#
                );
            }
        }
        Anchor::Eager | Anchor::EagerThoughtPreOpened => {
            // The gpt-oss generation prompt never pre-opens a
            // reasoning block, so both eager anchors share one shape.
            emit_until_rules("h_end", harmony::END, &mut src);
            let call_alts = (0..tools.len())
                .map(|i| format!("h_call_{i}"))
                .collect::<Vec<_>>()
                .join(" | ");
            let _ = writeln!(src, "root ::= h_seg* ( {call_alts} )");
            let _ = writeln!(
                src,
                r#"h_seg ::= ( "{analysis_open}" | "{commentary_open}" ) h_end "{start}""#
            );
            for (i, tool) in tools.iter().enumerate() {
                let name_lit = escape_for_gbnf_string(tool.name.as_ref());
                let _ = writeln!(
                    src,
                    r#"h_call_{i} ::= "{chan}commentary{to_fn}{name_lit} {constrain}json{msg}" h_args_{i}"#
                );
            }
        }
    }
    for (i, tool) in tools.iter().enumerate() {
        schema_to_gbnf(&tool.schema, &format!("h_args_{i}"), &mut src);
    }
    src.push_str(JSON_GRAMMAR);
    Ok(src)
}

/// Sorted required/optional key partition from a tool's input
/// schema. Unknown / empty schemas yield no keys (permissive object
/// downstream for JSON families; zero args for tagged).
fn schema_args(tool: &Tool) -> (Vec<(String, Value)>, Vec<(String, Value)>) {
    let props = tool
        .schema
        .get("properties")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let required: std::collections::HashSet<String> = tool
        .schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    // serde_json's default map is a BTreeMap: iteration is already
    // key-sorted, matching minijinja's re-render order.
    let mut req = Vec::new();
    let mut opt = Vec::new();
    for (k, v) in props {
        if required.contains(&k) {
            req.push((k, v));
        } else {
            opt.push((k, v));
        }
    }
    (req, opt)
}

fn schema_is_string(schema: &Value) -> bool {
    schema.get("type").and_then(|t| t.as_str()) == Some("string")
        && schema.get("enum").is_none()
}

/// TAG_WITH_TAGGED: literal name; args sorted-by-key in place
/// (optionals wrapped in `( ... )?`); string values raw-until-close,
/// other values schema-compiled JSON + literal close.
fn emit_tagged_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
    until_counter: &mut usize,
) {
    let name_lit = escape_for_gbnf_string(tool.name.as_ref());
    let fn_pre = escape_for_gbnf_string(&syntax.function.name_prefix);
    let fn_suf = escape_for_gbnf_string(&syntax.function.name_suffix);
    let fn_close = escape_for_gbnf_string(&syntax.function.close);
    let arg_pre = escape_for_gbnf_string(&syntax.arguments.name_prefix);
    let arg_suf = escape_for_gbnf_string(&syntax.arguments.name_suffix);
    let val_pre = escape_for_gbnf_string(&syntax.arguments.value_prefix);
    let val_suf_lit = escape_for_gbnf_string(&syntax.arguments.value_suffix);
    let sep = escape_for_gbnf_string(&syntax.arguments.separator);

    let (req, opt) = schema_args(tool);
    // Merge back to one sorted list, remembering optionality.
    let mut all: Vec<(String, Value, bool)> = req
        .into_iter()
        .map(|(k, v)| (k, v, true))
        .chain(opt.into_iter().map(|(k, v)| (k, v, false)))
        .collect();
    all.sort_by(|a, b| a.0.cmp(&b.0));

    let mut body = String::new();
    for (key, schema, required) in &all {
        *until_counter += 1;
        let key_lit = escape_for_gbnf_string(key);
        let arg_rule = format!("arg_{i}_{c}", c = *until_counter);
        if schema_is_string(schema) {
            // Raw value: the until-rule consumes value bytes AND the
            // closing delimiter.
            let until_rule = format!("val_{i}_{c}", c = *until_counter);
            emit_until_rules(&until_rule, &syntax.arguments.value_suffix, src);
            let _ = writeln!(
                src,
                r#"{arg_rule} ::= "{arg_pre}{key_lit}{arg_suf}{val_pre}" {until_rule}"#,
            );
        } else {
            let typed_rule = format!("typed_{i}_{c}", c = *until_counter);
            schema_to_gbnf(schema, &typed_rule, src);
            let _ = writeln!(
                src,
                r#"{arg_rule} ::= "{arg_pre}{key_lit}{arg_suf}{val_pre}" {typed_rule} "{val_suf_lit}""#,
            );
        }
        if !body.is_empty() && !sep.is_empty() {
            let _ = write!(body, r#" "{sep}""#);
        }
        if *required {
            let _ = write!(body, " {arg_rule}");
        } else {
            // In-place optional. NOTE: with a non-empty separator
            // this admits a dangling separator when the optional is
            // skipped — acceptable for now; no known non-empty-
            // separator tagged dialect. Revisit if one appears.
            let _ = write!(body, " {arg_rule}?");
        }
    }

    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}{fn_pre}{name_lit}{fn_suf}"{body} "{fn_close}{per_close}""#,
    );
}

/// TAG_WITH_DICT (Gemma 4): literal `call:` + name, then the whole
/// argument dict compiled from the schema in dict encoding — braces
/// included, keys bare and sorted in place, strings quoted by the
/// dialect's quote marker, compact separators. The generic `dvalue`
/// rules it references are appended once per grammar by
/// [`grammar_source`].
fn emit_dict_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
) {
    let name_lit = escape_for_gbnf_string(tool.name.as_ref());
    let fn_pre = escape_for_gbnf_string(&syntax.function.name_prefix);
    let args_rule = format!("args_{i}");
    schema_to_dict_gbnf(
        &tool.schema,
        &args_rule,
        &syntax.arguments.string_quote,
        src,
    );
    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}{fn_pre}{name_lit}" {args_rule} "{per_close}""#,
    );
}

/// TAG_WITH_JSON: tagged name, JSON args object.
fn emit_tag_json_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
) {
    let name_lit = escape_for_gbnf_string(tool.name.as_ref());
    let fn_pre = escape_for_gbnf_string(&syntax.function.name_prefix);
    let fn_suf = escape_for_gbnf_string(&syntax.function.name_suffix);
    let fn_close = escape_for_gbnf_string(&syntax.function.close);
    let args_rule = format!("args_{i}");
    schema_to_gbnf(&tool.schema, &args_rule, src);
    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}{fn_pre}{name_lit}{fn_suf}" {args_rule} "{fn_close}{per_close}""#,
    );
}

/// JSON_NATIVE: the call is a JSON object; field order follows the
/// analyzed `parameter_order` (defaults to name-then-args).
fn emit_json_native_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
) {
    let name_lit = escape_for_gbnf_string(
        &serde_json::to_string(tool.name.as_ref()).expect("string"),
    );
    let args_rule = format!("args_{i}");
    schema_to_gbnf(&tool.schema, &args_rule, src);

    if syntax.json.fun_name_is_key {
        let _ = writeln!(
            src,
            r#"call_{i} ::= "{per_open}" "{{" ws "{name_lit}" ws ":" ws {args_rule} ws "}}" "{per_close}""#,
        );
        return;
    }

    let name_field = if syntax.json.name_field.is_empty() {
        "name"
    } else {
        &syntax.json.name_field
    };
    let args_field = if syntax.json.args_field.is_empty() {
        "arguments"
    } else {
        &syntax.json.args_field
    };
    // Honor analyzed field order; unknown/extra fields (ids) are not
    // emitted — the model has no source of truth for them and JSON
    // dialects tolerate their absence on re-ingest.
    let mut order: Vec<&str> = syntax
        .json
        .parameter_order
        .iter()
        .map(String::as_str)
        .filter(|f| *f == name_field || *f == args_field)
        .collect();
    if order.is_empty() {
        order = vec![name_field, args_field];
    }
    let mut fields = String::new();
    for (j, field) in order.iter().enumerate() {
        if j > 0 {
            fields.push_str(r#" ws "," ws"#);
        }
        let field_lit = escape_for_gbnf_string(
            &serde_json::to_string(field).expect("string"),
        );
        if *field == name_field {
            let _ = write!(fields, r#" "{field_lit}" ws ":" ws "{name_lit}""#);
        } else {
            let _ = write!(fields, r#" "{field_lit}" ws ":" ws {args_rule}"#);
        }
    }
    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}" "{{" ws{fields} ws "}}" "{per_close}""#,
    );
}

/// Verify every argument of `input` is representable in `syntax` —
/// for tagged dialects, raw string values must not contain the value
/// close delimiter. JSON families escape natively and always pass.
pub fn validate_representable(
    syntax: &CallSyntax,
    tool_name: &str,
    input: &Value,
) -> Result<(), DialectError> {
    match syntax.family {
        Family::TagWithTagged => {
            let delimiter = &syntax.arguments.value_suffix;
            if delimiter.is_empty() {
                return Ok(());
            }
            if let Some(obj) = input.as_object() {
                for (key, val) in obj {
                    if let Some(s) = val.as_str() {
                        if s.contains(delimiter.as_str()) {
                            return Err(DialectError::UnrepresentableValue {
                                tool: tool_name.to_string(),
                                param: key.clone(),
                                delimiter: delimiter.clone(),
                            });
                        }
                    }
                }
            }
            Ok(())
        }
        Family::TagWithDict => {
            let quote = &syntax.arguments.string_quote;
            if quote.is_empty() {
                return Ok(());
            }
            // Recursive: nested containers render inline, so *every*
            // string is quote-delimited and *every* key is bare.
            fn check(
                tool: &str,
                param: &str,
                v: &Value,
                quote: &str,
            ) -> Result<(), DialectError> {
                let err = |delimiter: &str| {
                    Err(DialectError::UnrepresentableValue {
                        tool: tool.to_string(),
                        param: param.to_string(),
                        delimiter: delimiter.to_string(),
                    })
                };
                match v {
                    Value::String(s) if s.contains(quote) => err(quote),
                    Value::Object(map) => {
                        for (k, val) in map {
                            // Bare keys: the dict terminators (and the
                            // quote marker) cannot appear in a key.
                            if let Some(c) = k
                                .chars()
                                .find(|c| matches!(c, ':' | '{' | '}' | ','))
                            {
                                return err(&c.to_string());
                            }
                            if k.contains(quote) || k.is_empty() {
                                return err(quote);
                            }
                            check(tool, k, val, quote)?;
                        }
                        Ok(())
                    }
                    Value::Array(items) => {
                        for val in items {
                            check(tool, param, val, quote)?;
                        }
                        Ok(())
                    }
                    _ => Ok(()),
                }
            }
            check(tool_name, "", input, quote)
        }
        _ => Ok(()),
    }
}

/// Canonical byte serialization of `calls` in `syntax` — the exact
/// bytes the chat template's re-render will produce for these calls,
/// and the exact bytes [`grammar_source`]'s grammar forces. One call
/// is `(name, input)`.
///
/// Errors with [`DialectError::UnrepresentableValue`] on raw values
/// containing the close delimiter (tagged dialects only).
pub fn render_reference(
    syntax: &CallSyntax,
    calls: &[(&str, &Value)],
) -> Result<String, DialectError> {
    if syntax.family == Family::None {
        return Err(DialectError::NoToolFormat);
    }
    let mut out = String::new();
    out.push_str(&syntax.section_start);
    for (call_idx, (name, input)) in calls.iter().enumerate() {
        validate_representable(syntax, name, input)?;
        out.push_str(&syntax.per_call_start);
        match syntax.family {
            Family::Harmony => {
                // Canonical trained shape (channel-header recipient);
                // the cache-stable sidecar re-renders the same bytes.
                // A generation only ever produces one call (`<|call|>`
                // is EOG); client-constructed parallel calls become
                // consecutive messages, separated here by the same
                // `<|call|><|start|>assistant` framing the sidecar
                // renders. The *trailing* `<|call|>` is EOG framing,
                // not call bytes, so it is excluded (Gemma's
                // `tool_response_start` precedent).
                if call_idx > 0 {
                    out.push_str(harmony::CALL);
                    out.push_str(harmony::START_ASSISTANT);
                }
                out.push_str(harmony::COMMENTARY);
                out.push_str(harmony::TO_FUNCTIONS);
                out.push_str(name);
                out.push(' ');
                out.push_str(harmony::CONSTRAIN);
                out.push_str("json");
                out.push_str(harmony::MESSAGE);
                out.push_str(
                    &serde_json::to_string(input).expect("serializable"),
                );
            }
            Family::TagWithTagged => {
                out.push_str(&syntax.function.name_prefix);
                out.push_str(name);
                out.push_str(&syntax.function.name_suffix);
                if let Some(obj) = input.as_object() {
                    // BTreeMap: already key-sorted, like minijinja.
                    let mut first = true;
                    for (key, val) in obj {
                        if !first {
                            out.push_str(&syntax.arguments.separator);
                        }
                        first = false;
                        out.push_str(&syntax.arguments.name_prefix);
                        out.push_str(key);
                        out.push_str(&syntax.arguments.name_suffix);
                        out.push_str(&syntax.arguments.value_prefix);
                        match val {
                            // Raw strings render unquoted; everything
                            // else renders as JSON — the `| string` /
                            // `tojson` split templates use.
                            Value::String(s) => out.push_str(s),
                            other => out.push_str(
                                &serde_json::to_string(other)
                                    .expect("serializable"),
                            ),
                        }
                        out.push_str(&syntax.arguments.value_suffix);
                    }
                }
                out.push_str(&syntax.function.close);
            }
            Family::TagWithJson => {
                out.push_str(&syntax.function.name_prefix);
                out.push_str(name);
                out.push_str(&syntax.function.name_suffix);
                out.push_str(
                    &serde_json::to_string(input).expect("serializable"),
                );
                out.push_str(&syntax.function.close);
            }
            Family::JsonNative => {
                if syntax.json.fun_name_is_key {
                    let mut obj = serde_json::Map::new();
                    obj.insert((*name).to_string(), (*input).clone());
                    out.push_str(
                        &serde_json::to_string(&Value::Object(obj))
                            .expect("serializable"),
                    );
                } else {
                    let name_field = if syntax.json.name_field.is_empty() {
                        "name"
                    } else {
                        &syntax.json.name_field
                    };
                    let args_field = if syntax.json.args_field.is_empty() {
                        "arguments"
                    } else {
                        &syntax.json.args_field
                    };
                    // serde_json Map preserves insertion order only
                    // with preserve_order; default sorts. Render
                    // manually to honor parameter_order.
                    let mut order: Vec<&str> = syntax
                        .json
                        .parameter_order
                        .iter()
                        .map(String::as_str)
                        .filter(|f| *f == name_field || *f == args_field)
                        .collect();
                    if order.is_empty() {
                        order = vec![name_field, args_field];
                    }
                    out.push('{');
                    for (j, field) in order.iter().enumerate() {
                        if j > 0 {
                            out.push_str(", ");
                        }
                        if *field == name_field {
                            let _ = write!(
                                out,
                                "{}: {}",
                                serde_json::to_string(field).unwrap(),
                                serde_json::to_string(name).unwrap(),
                            );
                        } else {
                            let _ = write!(
                                out,
                                "{}: {}",
                                serde_json::to_string(field).unwrap(),
                                serde_json::to_string(input).unwrap(),
                            );
                        }
                    }
                    out.push('}');
                }
            }
            Family::TagWithDict => {
                out.push_str(&syntax.function.name_prefix);
                out.push_str(name);
                // The whole args object, braces included — compact,
                // key-sorted (BTreeMap ⇒ dictsort), quote-marked
                // strings. Matches the template's `format_argument`
                // with `escape_keys=False`.
                dict_encode_value(
                    input,
                    &syntax.arguments.string_quote,
                    &mut out,
                );
            }
            Family::None => unreachable!("checked above"),
        }
        out.push_str(&syntax.per_call_end);
    }
    out.push_str(&syntax.section_end);
    Ok(out)
}
