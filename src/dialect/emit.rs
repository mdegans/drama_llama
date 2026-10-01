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
//! ## Argument order: schema declaration order (#60)
//!
//! With `serde_json/preserve_order` + `minijinja/preserve_order` on
//! (unconditionally — see Cargo.toml), JSON maps keep insertion
//! order end-to-end: schemars derives `properties` in field
//! declaration order, the grammar emits fields in that order (so a
//! small model conditions later args on earlier, reasoning-ish
//! ones), the parser inserts in parse order, and minijinja's
//! `tojson` re-renders it unchanged. Matches llama.cpp. Two
//! refinements:
//!
//! * Optionals sit *in place* (`a? b c?` for required `b`) rather
//!   than trailing any-order. Any *fixed* order — each key appearing
//!   exactly once in the grammar — is what closes upstream's
//!   duplicate-optional grammar hole; alphabetization never was the
//!   load-bearing part.
//! * [`render_reference`] takes `(name, input)` with no schema, so
//!   it renders the input Map's own order. That agrees with the
//!   template re-render by construction (same Map), and with the
//!   grammar for model-generated calls because parse order equals
//!   grammar order. Caller-constructed inputs are canonical in
//!   whatever order their Map carries.
//!
//! The dict family (Gemma 4) is the exception: its templates pipe
//! arguments through `| dictsort`, so that family stays explicitly
//! alphabetical everywhere.

use std::collections::{HashMap, HashSet};
use std::fmt::Write;
use std::sync::Arc;

use serde_json::Value;

use crate::grammar_compile::{
    def_target, dict_encode_value, emit_dict_value_rules, emit_until_rules,
    escape_for_gbnf_string, json_grammar_canonical, schema_to_dict_gbnf,
    schema_to_gbnf, Compiler, RuleTally, SchemaError, FIELD_SEP, KV_SEP,
};
use crate::Tool;

use super::{harmony, CallSyntax, Family, ReasoningMode};

/// Errors from dialect emission / reference rendering.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
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
    /// A tool's `input_schema` has no grammar ([`SchemaError`]): the
    /// request's fault, like a 400 from Anthropic.
    #[error("tool {tool:?}: {source}")]
    Schema {
        tool: String,
        #[source]
        source: SchemaError,
    },
    /// The tools' schemas measure past [`EmitOptions::schema_limits`],
    /// so nothing compiled them: the request's fault, a 400.
    #[error("schema limits: {0}")]
    SchemaBudget(#[from] crate::SchemaBudgetError),
}

/// Tag a tool's [`SchemaError`] with the tool's name.
fn schema_err(tool: &Tool) -> impl FnOnce(SchemaError) -> DialectError + '_ {
    move |source| DialectError::Schema {
        tool: tool.name.to_string(),
        source,
    }
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
#[non_exhaustive]
pub struct EmitOptions {
    pub anchor: Anchor,
    /// Allow more than one call per turn. Gated on the caller's
    /// parallel-tool-calls setting.
    pub parallel: bool,
    /// The most the tools' schemas may measure, checked before anything
    /// classifies or compiles them ([`DialectError::SchemaBudget`]).
    /// Default [`SchemaLimits::default`](crate::SchemaLimits::default);
    /// `Session` passes its own
    /// ([`Session::with_schema_limits`](crate::Session::with_schema_limits)).
    pub schema_limits: crate::SchemaLimits,
}

impl Default for EmitOptions {
    fn default() -> Self {
        Self {
            anchor: Anchor::Eager,
            parallel: false,
            schema_limits: crate::SchemaLimits::default(),
        }
    }
}

/// Build the GBNF source constraining generation to `syntax`'s call
/// shape over `tools`, once their schemas measure inside
/// [`EmitOptions::schema_limits`].
pub fn grammar_source(
    syntax: &CallSyntax,
    tools: &[&Tool],
    opts: &EmitOptions,
) -> Result<String, DialectError> {
    if syntax.family == Family::None {
        return Err(DialectError::NoToolFormat);
    }
    crate::schema_budget::check_schemas(
        tools.iter().copied(),
        None,
        &opts.schema_limits,
    )?;
    if syntax.family == Family::Harmony {
        // The channel-block structure doesn't decompose into the
        // generic section/per-call root below — hand-built, like the
        // parser side.
        return harmony_grammar_source(
            tools,
            opts,
            syntax.arguments.json_spacing,
        );
    }
    let mut src = String::with_capacity(2048);
    let mut until_counter = 0usize;
    // Whether the shared raw-value rule is written yet.
    let mut raw_written = false;

    // Root.
    let has_reasoning = syntax.reasoning.mode != ReasoningMode::None
        && !syntax.reasoning.end.is_empty();
    // The until-delimiter for the thought body is the WHITESPACE-
    // TRIMMED close marker, matching the parser's leniency
    // (`parse_thought` scans for `reasoning.end.trim()`). Canonical
    // ends carry leading whitespace for byte-stable re-rendering
    // (Gemma's `"\n<channel|>"`), but an until-rule over the full
    // form is a trap: a model that closes without the newline can
    // NEVER exit the region — every subsequent byte is legal thought
    // content — and generation free-runs to the token budget
    // (observed e2e on Gemma 4, plan Phase G postmortem). A
    // non-canonical close now costs a one-turn canonicalization
    // repair instead, same as the parse side always did.
    let thought_delim = syntax.reasoning.end.trim();
    // What follows the close. A measured separator is spelled
    // literally: `fws` admits at most one whitespace byte, so a
    // template rendering `</think>\n\n` could never be matched under
    // the grammar and every forced thinking turn missed the tip (#112).
    let after_thought = match &syntax.reasoning.separator {
        Some(sep) if sep.is_empty() => String::new(),
        Some(sep) => format!(r#" "{}""#, escape_for_gbnf_string(sep)),
        None => " fws".to_string(),
    };
    match (opts.anchor, has_reasoning) {
        (Anchor::Lazy, _) => {
            let _ = writeln!(src, "root ::= calls");
        }
        (Anchor::EagerThoughtPreOpened, _) => {
            // Close tag required; body is raw-until-close.
            emit_until_rules("thought_close", thought_delim, &mut src);
            let _ =
                writeln!(src, "root ::= thought_close{after_thought} calls");
        }
        (Anchor::Eager, true) => {
            let start_lit = escape_for_gbnf_string(&syntax.reasoning.start);
            emit_until_rules("thought_close", thought_delim, &mut src);
            let open = if syntax.reasoning.start.is_empty() {
                String::new()
            } else {
                format!(r#""{start_lit}" "#)
            };
            let _ = writeln!(
                src,
                "root ::= ( {open}thought_close{after_thought} | fws ) calls"
            );
        }
        (Anchor::Eager, false) => {
            let _ = writeln!(src, "root ::= fws calls");
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
    // Parallel calls are joined by the template's inter-call separator
    // (Qwen3-Coder's `"\n"`), not concatenated — so the grammar forces
    // `call ( SEP call )*` and re-render byte-matches. Empty separator
    // collapses to `call+`.
    let sep = escape_for_gbnf_string(&syntax.call_separator);
    let call_seq = if opts.parallel {
        if sep.is_empty() {
            "call+".to_string()
        } else {
            format!(r#"call ( "{sep}" call )*"#)
        }
    } else {
        "call".to_string()
    };
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

    // Rules across every tool: each tool's compiler counts its own, and
    // many tools under the limit can still pass it together.
    let mut tally = RuleTally::default();
    for (i, tool) in tools.iter().enumerate() {
        match syntax.family {
            Family::TagWithTagged => emit_tagged_call(
                syntax,
                tool,
                i,
                (&per_open, &per_close),
                &mut src,
                &mut until_counter,
                &mut raw_written,
            ),
            Family::TagWithJson => emit_tag_json_call(
                syntax,
                tool,
                i,
                (&per_open, &per_close),
                &mut src,
            ),
            Family::JsonNative => emit_json_native_call(
                syntax,
                tool,
                i,
                (&per_open, &per_close),
                &mut src,
            ),
            Family::TagWithDict => emit_dict_call(
                syntax,
                tool,
                i,
                (&per_open, &per_close),
                &mut src,
            ),
            Family::None | Family::Harmony => unreachable!("checked above"),
        }?;
        tally.update(&src).map_err(schema_err(tool))?;
    }

    if syntax.family == Family::TagWithDict {
        emit_dict_value_rules(&syntax.arguments.string_quote, &mut src);
    }
    src.push_str(&json_grammar_canonical(syntax.arguments.json_spacing));
    Ok(src)
}

/// Hand-built Harmony (gpt-oss) grammar.
///
/// Eager (`Any`/`Method`): the generation prompt ends at
/// `<|start|>assistant`, and every Harmony message is a channel
/// block, so the grammar covers the *whole* emission — at most one
/// analysis block, then at most one commentary preamble (each closed
/// by `<|end|>` and reopened by `<|start|>assistant`), then the forced
/// call in its canonical trained shape:
/// `<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}`.
/// After the args complete nothing further is legal, so the sampler's
/// complete-constraint logic admits only EOG — the model's `<|call|>`
/// (see [`Model::eog_tokens`]), a deterministic stop that is also the
/// canonical re-render byte.
///
/// At most one of each, never `(analysis | commentary)*`: EOG is
/// illegal until the call completes and `final` is not a channel this
/// grammar offers, so every extra block it admits is somewhere a model
/// that does not want to call can go instead — indefinitely. A forced
/// call after the model had already answered (gpt-oss-120b, 2026-09-30)
/// alternated analysis and commentary blocks ("Now final.", "pong",
/// "We need to stop.", "[END]") to `max_tokens`. Bounded, the call is
/// at most two blocks away: the shape every other dialect's eager root
/// already has (one optional thought, then the calls).
///
/// [`Model::eog_tokens`]: crate::backend::Model::eog_tokens
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
    spacing: crate::JsonSpacing,
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
            let _ = writeln!(
                src,
                "root ::= h_analysis? h_preamble? ( {call_alts} )"
            );
            let _ = writeln!(
                src,
                r#"h_analysis ::= "{analysis_open}" h_end "{start}""#
            );
            let _ = writeln!(
                src,
                r#"h_preamble ::= "{commentary_open}" h_end "{start}""#
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
    let mut tally = RuleTally::default();
    for (i, tool) in tools.iter().enumerate() {
        schema_to_gbnf(&tool.schema, &format!("h_args_{i}"), &mut src)
            .map_err(schema_err(tool))?;
        tally.update(&src).map_err(schema_err(tool))?;
    }
    src.push_str(&json_grammar_canonical(spacing));
    Ok(src)
}

/// The tool's argument slots in `properties` iteration order
/// (declaration order under `preserve_order`), each flagged required
/// by *membership* in `required:` — never by that array's order.
/// Unknown / empty schemas yield no keys (permissive object
/// downstream for JSON families; zero args for tagged).
fn schema_args(tool: &Tool) -> Vec<(String, Value, bool)> {
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
    props
        .into_iter()
        .map(|(k, v)| {
            let req = required.contains(&k);
            (k, v, req)
        })
        .collect()
}

/// How a TAG_WITH_TAGGED parameter's value is spelled between its
/// tags: what [`grammar_source`] generates and what the parser reads
/// back, decided in one place so the two agree by construction.
///
/// The rule is the template's: Qwen renders a string argument raw
/// (`args_value | string if args_value is string else tojson`), so a
/// string value is never JSON-quoted at the top of a parameter — the
/// model was trained on `<parameter=detail>\nfull\n</parameter>`, and
/// a quoted `"full"` re-renders without its quotes, a tip miss on
/// every such call.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum TaggedValue {
    /// Any string, raw until the close tag. When `nullable`, a bare
    /// `null` is JSON null — its one non-string value.
    Raw { nullable: bool },
    /// One of finitely many values, at least one a string (an `enum`,
    /// a `const`, or a union of them, nullable or not): each spelled
    /// as the template re-renders it — a string raw, anything else
    /// as JSON — and read back by exact match. Shared: parameters
    /// with the same members (every one that `$ref`s the same def)
    /// hold the same set, which the grammar writes once.
    Choice(Choice),
    /// Schema-compiled JSON: no string in it, or no raw spelling that
    /// reads back unambiguously.
    Json,
}

/// A [`TaggedValue::Choice`]'s members, in declaration order.
pub(crate) type Choice = Arc<[Arc<Member>]>;

/// One member of a finite set, as a tagged value spells it.
#[derive(Debug, PartialEq)]
pub(crate) struct Member {
    /// The template's rendering: a string raw, anything else as JSON.
    pub(crate) spelling: String,
    pub(crate) value: Value,
}

/// Most work classifying one tool's parameters, all of them together:
/// a step per schema visited, a member's bytes for each member spelled,
/// a step per member a parameter takes from a `$ref`'d def. A parameter
/// met once the budget is spent is [`TaggedValue::Json`], which is
/// always a correct spelling, merely not the raw one.
///
/// Each def is classified once per tool and its members spelled once,
/// so a parameter that `$ref`s one costs a step a member, not a walk of
/// it: two thousand parameters naming a 100 KB `enum` spell it once
/// (they took 200 MB of spelling and 400 MB of members, the hostile
/// recheck). The budget bounds what is left — every parameter with an
/// inline set of its own, or a union of defs — at about a mebibyte of
/// member text per tool. Real tools take a few steps a parameter.
const TAGGED_BUDGET: usize = 1 << 20;

/// Most distinct members a finite set may have and still be spelled
/// raw ([`TaggedValue::Choice`]); a larger one is JSON. The matcher's
/// cap on stacks alive at once (`MAX_STACKS`), which a set past it
/// would reach anyway: so every set inside
/// [`SchemaLimits::max_width`](crate::SchemaLimits::max_width) is raw,
/// as the template writes it. At 1024 a set of 1025–2048 members passed
/// the measure and came out JSON — `"Zone_01024"` where the template
/// writes `Zone_01024` — a spelling the cache could not match.
const TAGGED_MAX_MEMBERS: usize = crate::sample::grammar::MAX_STACKS;

/// Deepest a classification nests (`$ref` chains, unions in unions)
/// before it gives up as JSON rather than recurse further.
const TAGGED_MAX_DEPTH: usize = 32;

/// Every parameter of `tool_schema`, in `properties` order, with how
/// it is spelled for `syntax` ([`TaggedValue`]).
///
/// `$ref`s resolve against the tool schema's `$defs`; `anyOf` and
/// `oneOf` are read as unions (their members are disjoint in every
/// shape schemars emits — a `oneOf` of unit-variant `const`s). A
/// finite set falls back to [`TaggedValue::Json`] when its raw
/// spellings collide (the string `"1"` beside the number `1`, `"null"`
/// beside `null`) or a member would end its own value early (it
/// contains the close tag); a mixed set (`["a", 1, null]`) spells its
/// strings raw and the rest as JSON. Leading or trailing whitespace is
/// kept: the template renders it verbatim and the parser reads the
/// value byte-exact, so such a member round-trips as written.
///
/// The emitter and the parser both classify a whole tool through
/// here, in this order, so the shared [`TAGGED_BUDGET`] runs out at
/// the same parameter for both.
pub(crate) fn tagged_values(
    syntax: &CallSyntax,
    tool_schema: &Value,
) -> Vec<(String, TaggedValue)> {
    let mut classifier = Classifier::new(syntax, tool_schema);
    tool_schema
        .get("properties")
        .and_then(Value::as_object)
        .into_iter()
        .flatten()
        .map(|(name, param)| (name.clone(), classifier.classify(param)))
        .collect()
}

/// One parameter of `tool_schema` as [`tagged_values`] spells it, on a
/// budget of its own.
#[cfg(test)]
pub(crate) fn tagged_value(
    syntax: &CallSyntax,
    tool_schema: &Value,
    param: &Value,
) -> TaggedValue {
    Classifier::new(syntax, tool_schema).classify(param)
}

/// What a schema admits, as far as a tagged value's spelling cares.
#[derive(Clone, Default)]
struct Class {
    /// Any string (`"type": "string"`).
    any_string: bool,
    /// Unboundedly many non-string values: a non-string type, or an
    /// unconstrained schema — or the classification gave up.
    any_other: bool,
    /// Finitely many values (`enum`, `const`, `"type": "null"`), as
    /// [`Classifier`] member ids, in declaration order, distinct.
    members: Vec<usize>,
    /// `members`, for O(1) dedup.
    seen: HashSet<usize>,
    /// Deepest nesting the walk reached below where it started.
    height: usize,
    /// The walk hit [`TAGGED_MAX_DEPTH`]: the answer depends on where
    /// it started, so a def's is not memoized.
    truncated: bool,
    /// The tool's budget ran out during the walk: likewise.
    starved: bool,
}

impl Class {
    fn add(&mut self, member: usize) {
        if self.seen.insert(member) {
            self.members.push(member);
        }
        if self.members.len() > TAGGED_MAX_MEMBERS {
            self.any_other = true;
        }
    }
}

/// A spelled member and what [`Classifier::finish`] checks of it,
/// computed once per tool.
struct Spelled {
    member: Arc<Member>,
    /// The spelling's id: two members of one set that share one
    /// collide.
    spelling: usize,
    /// The close tag right after the member is the first one.
    delimited: bool,
    /// A JSON string (or `null`, the nullable case).
    is_string: bool,
    is_null: bool,
}

/// One tool's classifier: each def's [`Class`] once, each member's
/// spelling once, each distinct member set's [`Choice`] once.
struct Classifier<'s> {
    syntax: &'s CallSyntax,
    defs: Option<&'s serde_json::Map<String, Value>>,
    /// Work left of the tool's [`TAGGED_BUDGET`].
    budget: usize,
    /// Every member met, deduped by JSON spelling.
    members: Vec<Spelled>,
    /// JSON spelling → member id.
    by_json: HashMap<String, usize>,
    /// Template spelling → spelling id.
    by_spelling: HashMap<String, usize>,
    /// Member value (by address in the schema) → member id, so a
    /// member is spelled once however often it is reached.
    by_node: HashMap<*const Value, usize>,
    /// Each def's class, once computed (and not truncated).
    def_classes: HashMap<&'s str, Arc<Class>>,
    /// Each distinct member set's choice, so parameters share one.
    choices: HashMap<Vec<usize>, Choice>,
}

/// How far a classification has got through a def.
#[derive(Clone, Copy, PartialEq)]
enum Visit {
    Open,
    Done,
}

impl<'s> Classifier<'s> {
    fn new(syntax: &'s CallSyntax, tool_schema: &'s Value) -> Self {
        Self {
            syntax,
            defs: tool_schema.get("$defs").and_then(Value::as_object),
            budget: TAGGED_BUDGET,
            members: Vec::new(),
            by_json: HashMap::new(),
            by_spelling: HashMap::new(),
            by_node: HashMap::new(),
            def_classes: HashMap::new(),
            choices: HashMap::new(),
        }
    }

    /// How `param` is spelled.
    fn classify(&mut self, param: &'s Value) -> TaggedValue {
        let mut class = Class::default();
        self.collect(param, 0, &mut class, &mut HashMap::new());
        self.finish(class)
    }

    /// Spend `cost`; out of budget, `class` is JSON.
    fn spend(&mut self, class: &mut Class, cost: usize) -> bool {
        if !class.any_other {
            match self.budget.checked_sub(cost) {
                Some(left) => self.budget = left,
                None => {
                    self.budget = 0;
                    class.any_other = true;
                    class.starved = true;
                }
            }
        }
        !class.any_other
    }

    fn collect(
        &mut self,
        schema: &'s Value,
        depth: usize,
        class: &mut Class,
        visits: &mut HashMap<&'s str, Visit>,
    ) {
        // Once anything non-string goes, the answer is JSON whatever
        // else turns up. Too deep (a chain of thousands of aliases)
        // gives up as JSON too, rather than nest that far.
        class.height = class.height.max(depth);
        if depth > TAGGED_MAX_DEPTH {
            class.any_other = true;
            class.truncated = true;
        }
        if !self.spend(class, 1) {
            return;
        }
        // The `$ref` shape the grammar compiler resolves; anything else
        // falls through to the schema's other keywords, as it does there.
        if let Some((name, def)) = def_target(self.defs, schema) {
            match visits.get(name) {
                // A `$ref` cycle is unconstrained (see `Defs`): JSON.
                Some(Visit::Open) => class.any_other = true,
                Some(Visit::Done) => {}
                None => {
                    visits.insert(name, Visit::Open);
                    let sub = self.def_class(name, def, visits);
                    visits.insert(name, Visit::Done);
                    self.merge(class, &sub, depth + 1);
                }
            }
            return;
        }
        let union = schema
            .get("anyOf")
            .or_else(|| schema.get("oneOf"))
            .and_then(Value::as_array);
        if let Some(variants) = union {
            match variants.is_empty() {
                true => class.any_other = true,
                false => variants
                    .iter()
                    .for_each(|v| self.collect(v, depth + 1, class, visits)),
            }
            return;
        }
        if let Some(values) = schema.get("enum").and_then(Value::as_array) {
            values.iter().for_each(|v| self.member(class, v));
            return;
        }
        if let Some(value) = schema.get("const") {
            return self.member(class, value);
        }
        match schema.get("type") {
            Some(Value::String(t)) => self.of_type(class, t),
            Some(Value::Array(types)) => types.iter().for_each(|t| match t {
                Value::String(t) => self.of_type(class, t),
                _ => class.any_other = true,
            }),
            _ => class.any_other = true,
        }
    }

    /// `def`'s class, from the memo or walked on its own. Its own walk
    /// keeps only the defs open around it (`visits`' `Open`s), so a
    /// cycle is still a cycle, and drops the caller's finished ones, so
    /// the class holds everything the def reaches. That is the same
    /// wherever the def is reached — a def that reaches an open one is
    /// on a cycle, which is JSON from anywhere — unless the walk was cut
    /// short by depth or budget, so only those are not kept.
    fn def_class(
        &mut self,
        name: &'s str,
        def: &'s Value,
        visits: &HashMap<&'s str, Visit>,
    ) -> Arc<Class> {
        if let Some(class) = self.def_classes.get(name) {
            return class.clone();
        }
        let mut own: HashMap<&'s str, Visit> = visits
            .iter()
            .filter(|(_, v)| **v == Visit::Open)
            .map(|(k, v)| (*k, *v))
            .collect();
        let mut sub = Class::default();
        self.collect(def, 0, &mut sub, &mut own);
        let sub = Arc::new(sub);
        if !sub.truncated && !sub.starved {
            self.def_classes.insert(name, sub.clone());
        }
        sub
    }

    /// Fold a def's class into `class`, as if walked from `depth`.
    fn merge(&mut self, class: &mut Class, sub: &Class, depth: usize) {
        class.height = class.height.max(depth + sub.height);
        class.truncated |= sub.truncated;
        class.starved |= sub.starved;
        if depth + sub.height > TAGGED_MAX_DEPTH {
            class.any_other = true;
            class.truncated = true;
        }
        class.any_string |= sub.any_string;
        class.any_other |= sub.any_other;
        if !self.spend(class, sub.members.len()) {
            return;
        }
        sub.members.iter().for_each(|&m| class.add(m));
    }

    fn of_type(&mut self, class: &mut Class, t: &str) {
        match t {
            "string" => class.any_string = true,
            "null" => self.member(class, &Value::Null),
            _ => class.any_other = true,
        }
    }

    fn member(&mut self, class: &mut Class, value: &Value) {
        if class.any_other {
            return;
        }
        let id = match self.by_node.get(&(value as *const Value)) {
            Some(&id) => {
                if !self.spend(class, 1) {
                    return;
                }
                id
            }
            None => {
                let json = value.to_string();
                if !self.spend(class, json.len()) {
                    return;
                }
                let id = self.spell(json, value);
                // `Value::Null` from `"type": "null"` is not in the
                // schema; it costs one spelling all the same.
                self.by_node.insert(value as *const Value, id);
                id
            }
        };
        class.add(id);
    }

    /// The member id of `value`, whose JSON spelling is `json`.
    fn spell(&mut self, json: String, value: &Value) -> usize {
        if let Some(&id) = self.by_json.get(&json) {
            return id;
        }
        let spelling = match value {
            Value::String(s) => s.clone(),
            other => crate::json_canon::to_string(
                other,
                self.syntax.arguments.json_spacing,
            ),
        };
        let close = self.syntax.arguments.value_suffix.as_str();
        // The parser ends a value at the first close tag, so the close
        // after a member must be the first one: none inside the member,
        // and none that starts in its tail (`a\n</parameter>` + the
        // close).
        let delimited = !close.is_empty()
            && format!("{spelling}{close}").find(close) == Some(spelling.len());
        let next = self.by_spelling.len();
        let spelling_id =
            *self.by_spelling.entry(spelling.clone()).or_insert(next);
        let id = self.members.len();
        self.members.push(Spelled {
            member: Arc::new(Member {
                spelling,
                value: value.clone(),
            }),
            spelling: spelling_id,
            delimited,
            is_string: value.is_string(),
            is_null: value.is_null(),
        });
        self.by_json.insert(json, id);
        id
    }

    /// The spelling `class` comes to.
    fn finish(&mut self, class: Class) -> TaggedValue {
        if class.any_other {
            return TaggedValue::Json;
        }
        let members: Vec<&Spelled> =
            class.members.iter().map(|&m| &self.members[m]).collect();
        if class.any_string {
            let nullable = members.iter().any(|m| m.is_null);
            return match members.iter().all(|m| m.is_string || m.is_null) {
                true => TaggedValue::Raw { nullable },
                false => TaggedValue::Json,
            };
        }
        if let Some(choice) = self.choices.get(&class.members) {
            return TaggedValue::Choice(choice.clone());
        }
        if !members.iter().any(|m| m.is_string) {
            return TaggedValue::Json;
        }
        let mut seen = HashSet::new();
        let distinct = members.iter().all(|m| seen.insert(m.spelling));
        if !(distinct && members.iter().all(|m| m.delimited)) {
            return TaggedValue::Json;
        }
        let choice: Choice = members.iter().map(|m| m.member.clone()).collect();
        self.choices.insert(class.members, choice.clone());
        TaggedValue::Choice(choice)
    }
}

/// Whether a JSON-spelled tagged parameter's `schema` allows `null`
/// only through a `type` array the compiler collapses to its one other
/// type (`effective_type`): no `enum` or `const` (whose members decide,
/// null among them or not) or `anyOf` (compiled whole, a `null` variant
/// included). A `$ref` is read as the def it names in `defs`, as the
/// compiler reads it — through an alias chain, a cycle reading as no.
fn bare_nullable(
    defs: Option<&serde_json::Map<String, Value>>,
    schema: &Value,
) -> bool {
    let mut schema = schema;
    let mut hops = 0;
    while let Some((_, def)) = def_target(defs, schema) {
        hops += 1;
        if hops > defs.map_or(0, |d| d.len()) {
            return false;
        }
        schema = def;
    }
    let decided_elsewhere = ["enum", "const", "$ref", "anyOf"]
        .iter()
        .any(|k| schema.get(k).is_some());
    let null_in_types = schema
        .get("type")
        .and_then(Value::as_array)
        .is_some_and(|ts| ts.iter().any(|t| t == "null"));
    !decided_elsewhere
        && null_in_types
        && crate::grammar_compile::effective_type(schema)
            .is_some_and(|t| t != "null")
}

/// The until-rule every raw tagged value shares ([`emit_tagged_call`]).
const RAW_VALUE_RULE: &str = "val_raw";

/// TAG_WITH_TAGGED: literal name; args in schema declaration order
/// in place (optionals wrapped in `( ... )?`); each value spelled as
/// [`tagged_values`] decides — a string raw-until-close, a finite set
/// of strings raw by alternation, anything else schema-compiled JSON —
/// then the literal close.
fn emit_tagged_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
    until_counter: &mut usize,
    raw_written: &mut bool,
) -> Result<(), DialectError> {
    let name_lit = escape_for_gbnf_string(tool.name.as_ref());
    let fn_pre = escape_for_gbnf_string(&syntax.function.name_prefix);
    let fn_suf = escape_for_gbnf_string(&syntax.function.name_suffix);
    let fn_close = escape_for_gbnf_string(&syntax.function.close);
    let arg_pre = escape_for_gbnf_string(&syntax.arguments.name_prefix);
    let arg_suf = escape_for_gbnf_string(&syntax.arguments.name_suffix);
    let val_pre = escape_for_gbnf_string(&syntax.arguments.value_prefix);
    let val_suf_lit = escape_for_gbnf_string(&syntax.arguments.value_suffix);
    let sep = escape_for_gbnf_string(&syntax.arguments.separator);

    let all = schema_args(tool);
    let values = tagged_values(syntax, &tool.schema);
    // One compiler for the tool's JSON parameters: a parameter's schema
    // has no `$defs` of its own, so its `$ref`s resolve in the tool's,
    // and each def they reach is written once for the tool, not once
    // per parameter naming it.
    let prefix = format!("tool_{i}");
    let defs = tool.schema.get("$defs").and_then(Value::as_object);
    let mut compiler = Compiler::new(defs, &prefix, None);
    // Each distinct set's alternation, written once for every parameter
    // holding it (the classifier shares one `Choice` among them).
    let mut choice_rules: HashMap<*const Arc<Member>, String> = HashMap::new();

    let mut body = String::new();
    for ((key, schema, required), (_, value)) in all.iter().zip(values) {
        if compiler.halted(src) {
            break;
        }
        *until_counter += 1;
        let key_lit = escape_for_gbnf_string(key);
        let arg_rule = format!("arg_{i}_{c}", c = *until_counter);
        let typed_rule = format!("typed_{i}_{c}", c = *until_counter);
        let value = match value {
            TaggedValue::Raw { .. } => {
                // Raw value: the until-rule consumes value bytes AND
                // the closing delimiter. A nullable string's `null` is
                // one such value. One rule serves every raw parameter
                // of every tool — it depends only on the delimiter —
                // written the first time one needs it: a copy per
                // parameter (~1.5 KB of KMP states each) made a large
                // tool's grammar mostly duplicates.
                if !std::mem::replace(raw_written, true) {
                    emit_until_rules(
                        RAW_VALUE_RULE,
                        &syntax.arguments.value_suffix,
                        src,
                    );
                }
                RAW_VALUE_RULE.to_string()
            }
            TaggedValue::Choice(choice) => {
                let rule =
                    choice_rules.entry(choice.as_ptr()).or_insert_with(|| {
                        let alts = choice
                            .iter()
                            .map(|m| {
                                format!(
                                    r#""{}""#,
                                    escape_for_gbnf_string(&m.spelling)
                                )
                            })
                            .collect::<Vec<_>>()
                            .join(" | ");
                        let _ = writeln!(src, "{typed_rule} ::= {alts}");
                        typed_rule.clone()
                    });
                format!(r#"{rule} "{val_suf_lit}""#)
            }
            TaggedValue::Json if bare_nullable(defs, schema) => {
                // A nullable type (`["integer", "null"]`, schemars'
                // `Option<T>`) compiles to its base type
                // (`effective_type`); as a parameter it takes the bare
                // `null` its schema allows too, which reads back null.
                let base = format!("{typed_rule}_base");
                compiler.add(schema, &base, src);
                let _ = writeln!(src, r#"{typed_rule} ::= {base} | "null""#);
                format!(r#"{typed_rule} "{val_suf_lit}""#)
            }
            TaggedValue::Json => {
                compiler.add(schema, &typed_rule, src);
                format!(r#"{typed_rule} "{val_suf_lit}""#)
            }
        };
        let _ = writeln!(
            src,
            r#"{arg_rule} ::= "{arg_pre}{key_lit}{arg_suf}{val_pre}" {value}"#,
        );
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
    compiler.finish(src).map_err(schema_err(tool))?;

    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}{fn_pre}{name_lit}{fn_suf}"{body} "{fn_close}{per_close}""#,
    );
    Ok(())
}

/// TAG_WITH_DICT (Gemma 4): literal `call:` + name, then the whole
/// argument dict compiled from the schema in dict encoding — braces
/// included, keys bare and explicitly sorted in place (the Gemma
/// templates `dictsort` their re-renders, so alphabetical is
/// upstream's decree, not ours), strings quoted by the dialect's
/// quote marker, compact separators. The generic `dvalue`
/// rules it references are appended once per grammar by
/// [`grammar_source`].
fn emit_dict_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
) -> Result<(), DialectError> {
    let name_lit = escape_for_gbnf_string(tool.name.as_ref());
    let fn_pre = escape_for_gbnf_string(&syntax.function.name_prefix);
    let args_rule = format!("args_{i}");
    schema_to_dict_gbnf(
        &tool.schema,
        &args_rule,
        &syntax.arguments.string_quote,
        src,
    )
    .map_err(schema_err(tool))?;
    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}{fn_pre}{name_lit}" {args_rule} "{per_close}""#,
    );
    Ok(())
}

/// TAG_WITH_JSON: tagged name, JSON args object.
fn emit_tag_json_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
) -> Result<(), DialectError> {
    let name_lit = escape_for_gbnf_string(tool.name.as_ref());
    let fn_pre = escape_for_gbnf_string(&syntax.function.name_prefix);
    let fn_suf = escape_for_gbnf_string(&syntax.function.name_suffix);
    let fn_close = escape_for_gbnf_string(&syntax.function.close);
    let args_rule = format!("args_{i}");
    schema_to_gbnf(&tool.schema, &args_rule, src).map_err(schema_err(tool))?;
    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}{fn_pre}{name_lit}{fn_suf}" {args_rule} "{fn_close}{per_close}""#,
    );
    Ok(())
}

/// JSON_NATIVE: the call is a JSON object; field order follows the
/// analyzed `parameter_order` (defaults to name-then-args).
fn emit_json_native_call(
    syntax: &CallSyntax,
    tool: &Tool,
    i: usize,
    (per_open, per_close): (&str, &str),
    src: &mut String,
) -> Result<(), DialectError> {
    let name_lit = escape_for_gbnf_string(
        &serde_json::to_string(tool.name.as_ref()).expect("string"),
    );
    let args_rule = format!("args_{i}");
    schema_to_gbnf(&tool.schema, &args_rule, src).map_err(schema_err(tool))?;

    if syntax.json.fun_name_is_key {
        let _ = writeln!(
            src,
            r#"call_{i} ::= "{per_open}" "{{" "{name_lit}" "{KV_SEP}" {args_rule} "}}" "{per_close}""#,
        );
        return Ok(());
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
            let _ = write!(fields, r#" "{FIELD_SEP}""#);
        }
        let field_lit = escape_for_gbnf_string(
            &serde_json::to_string(field).expect("string"),
        );
        if *field == name_field {
            let _ = write!(fields, r#" "{field_lit}" "{KV_SEP}" "{name_lit}""#);
        } else {
            let _ = write!(fields, r#" "{field_lit}" "{KV_SEP}" {args_rule}"#);
        }
    }
    let _ = writeln!(
        src,
        r#"call_{i} ::= "{per_open}" "{{"{fields} "}}" "{per_close}""#,
    );
    Ok(())
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
        // Inter-call separator the template weaves between adjacent
        // calls (empty for Harmony, whose own call framing is handled
        // in its match arm below). Mirrors the grammar's
        // `call ( SEP call )*` so an N-call turn round-trips.
        if call_idx > 0 {
            out.push_str(&syntax.call_separator);
        }
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
                out.push_str(&crate::json_canon::to_string(
                    input,
                    syntax.arguments.json_spacing,
                ));
            }
            Family::TagWithTagged => {
                out.push_str(&syntax.function.name_prefix);
                out.push_str(name);
                out.push_str(&syntax.function.name_suffix);
                if let Some(obj) = input.as_object() {
                    // Map insertion order (preserve_order): parse
                    // order == grammar order, and minijinja renders
                    // the same Map — agreement by construction.
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
                            other => {
                                out.push_str(&crate::json_canon::to_string(
                                    other,
                                    syntax.arguments.json_spacing,
                                ))
                            }
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
                out.push_str(&crate::json_canon::to_string(
                    input,
                    syntax.arguments.json_spacing,
                ));
                out.push_str(&syntax.function.close);
            }
            Family::JsonNative => {
                if syntax.json.fun_name_is_key {
                    // NOTE: under `Compact` the grammar's envelope
                    // (`KV_SEP` after the name key) and this render
                    // disagree — a pre-existing latent mismatch for
                    // fun-name-is-key templates, out of #88 phase 2's
                    // scope. `Spaced` makes them agree.
                    let mut obj = serde_json::Map::new();
                    obj.insert((*name).to_string(), (*input).clone());
                    out.push_str(&crate::json_canon::to_string(
                        &Value::Object(obj),
                        syntax.arguments.json_spacing,
                    ));
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
                    // Manual render: honors parameter_order for the
                    // top-level fields and emits the spaced
                    // separators (`", "`, `": "`) this shape is
                    // probed with — compact `to_string` couldn't.
                    // The inner args object still rides the Map
                    // (insertion order under preserve_order).
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
                                crate::json_canon::to_string(
                                    input,
                                    syntax.arguments.json_spacing,
                                ),
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
                // explicitly key-sorted (the template `dictsort`s,
                // so we must too), quote-marked strings. Matches the
                // template's `format_argument` with
                // `escape_keys=False`.
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
