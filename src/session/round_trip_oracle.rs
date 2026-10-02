//! The round-trip oracle (#129, item 5): for emissions a baked
//! dialect's grammar admits, `render(parse(emission)) == emission`,
//! byte for byte, through the session's own parse and the real template
//! render — the invariant the auto-tip rests on (see the round-trip
//! section of [`Session`](super::Session)).
//!
//! Where `fleet_bakes_round_trip_every_admitted_shape` sweeps a fixed
//! list of hand-written shapes over one string argument, this walks the
//! grammars themselves over random tool schemas:
//!
//! * **forced** — `tool_choice` any: the whole emission is a random walk
//!   of the eager grammar the session compiles, thought framing
//!   included;
//! * **auto** — the dialect's habitual thought and prose framing around
//!   random text, then a walk of the lazy (trigger-activated) grammar
//!   — and then text after the last call (a stray close, whitespace,
//!   prose), which the grammar must refuse or the turn round-trip;
//! * **text** — thoughts and prose alone (Harmony: analysis, preamble
//!   and final, plain or `<|constrain|>json`).
//!
//! Each emission is held to the session's canonicalization gate (the
//! turn closes the render) and to the next request (tool results or a
//! user turn, then the generation prompt). A failure is minimized and
//! reported with its case; accepted classes — irreducible ones, and
//! `PENDING` ones awaiting a decision — are listed in
//! `known_divergence` with the reason each is accepted, and tallied on
//! every run. `round_trip_oracle_corpus` replays the live failures and
//! every finding.
//!
//! The smoke run is fixed-seed and runs with the unit tests. For a long
//! run, with a random seed it prints:
//!
//! ```sh
//! ORACLE_SEED=random ORACLE_CASES=20000 cargo test --lib \
//!     round_trip_oracle -- --nocapture
//! ```
//!
//! CPU only: no model is loaded. Not modelled: the content-literal
//! neutralizer (it needs a vocab), so prose here never spells a
//! special token.

use std::sync::Arc;

use rand::rngs::SmallRng;
use rand::seq::{IndexedRandom, SliceRandom};
use rand::{RngExt as _, SeedableRng};
use serde_json::{Map, Value};

use super::{
    drop_repeated_calls, effective_tool_syntax, emission_divergence,
    render_ends_with_closed_reasoning, render_ends_with_open_reasoning, Prompt,
};
use crate::dialect::{
    grammar_source, harmony, Anchor, EmitOptions, Family, Leniency,
    ParseStatus, ReasoningMode,
};
use crate::prompt::{Message, Role};
use crate::{ChatTemplate, Content, Grammar, GrammarState, RenderOptions};

/// One served template: a baked replacement and its analyzed dialect.
struct Fixture {
    name: &'static str,
    template: ChatTemplate,
    syntax: crate::CallSyntax,
}

fn fixtures() -> Vec<Fixture> {
    [
        (&crate::baked::GEMMA4, ("<bos>", "<turn|>")),
        (&crate::baked::GPTOSS, ("<|startoftext|>", "<|return|>")),
        (
            &crate::baked::GPTOSS_UPSTREAM,
            ("<|startoftext|>", "<|return|>"),
        ),
        (&crate::baked::COGITO, ("", "<|im_end|>")),
        (&crate::baked::MISTRAL4, ("<s>", "</s>")),
        (&crate::baked::QWEN36, ("", "<|im_end|>")),
        (&crate::baked::QWEN38, ("", "<|im_end|>")),
    ]
    .into_iter()
    .map(|(baked, (bos, eos))| {
        let served = crate::baked::detect(baked.stock)
            .expect("stock dump detects")
            .replacement;
        Fixture {
            name: baked.name,
            template: ChatTemplate::from_source(
                served.to_owned(),
                bos.to_owned(),
                eos.to_owned(),
            )
            .expect("template compiles"),
            syntax: crate::dialect::analyze_template(served, bos, eos)
                .expect("analyze"),
        }
    })
    .collect()
}

/// How the emission came to be — and so what it must be admitted by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// `tool_choice: any`: the eager grammar from the first byte.
    Forced,
    /// `tool_choice: auto`: free text, then the lazy grammar from the
    /// trigger on.
    Auto,
    /// No call: free text in the dialect's framing.
    Text,
}

/// What part of an emission a segment is — what minimizing may cut.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Seg {
    /// Dialect framing (thought and channel markers): kept whole.
    Frame,
    /// Free text inside the framing.
    Free,
    /// A grammar walk: cut only as far as the grammar still admits.
    Walk,
}

/// One oracle case: everything needed to replay it.
#[derive(Clone, Debug)]
struct Case {
    mode: Mode,
    thinking: bool,
    parallel: bool,
    tools: Vec<crate::Tool>,
    segments: Vec<(Seg, String)>,
    /// The segments, joined.
    emission: String,
}

impl Case {
    fn with_segments(&self, segments: Vec<(Seg, String)>) -> Self {
        Self {
            emission: segments.iter().map(|(_, s)| s.as_str()).collect(),
            segments,
            ..self.clone()
        }
    }

    /// Where the walk starts in the emission, if there is one.
    fn walk_at(&self) -> Option<usize> {
        self.walks().first().map(|w| w.start)
    }

    /// Where each walk sits in the emission.
    fn walks(&self) -> Vec<std::ops::Range<usize>> {
        let mut at = 0;
        let mut out = Vec::new();
        for (kind, seg) in &self.segments {
            if *kind == Seg::Walk {
                out.push(at..at + seg.len());
            }
            at += seg.len();
        }
        out
    }
}

/// What follows the turn in the render that measures it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Next {
    /// Nothing: the session's canonicalization gate.
    Nothing,
    /// The next request: tool results (by id) or a user turn, then the
    /// generation prompt.
    Reply,
}

/// Why a case fails the oracle.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Failure {
    /// The re-render parts from the emission at `at`.
    Diverges {
        next: Next,
        at: usize,
        /// What the KV holds there, and what the re-render writes.
        emitted: String,
        rerendered: String,
    },
    /// An admitted emission did not parse to completion: the session
    /// would treat the turn as cut, and keep no tip.
    ParseIncomplete,
    /// A forced turn parsed to no call: the session refuses it.
    ForcedNoCall,
}

impl Failure {
    /// Equal for the same bug, so a minimized case still shows it.
    fn signature(&self) -> String {
        match self {
            Self::Diverges {
                next,
                emitted,
                rerendered,
                ..
            } => {
                let first = |s: &str| s.chars().next();
                format!("{next:?}:{:?}/{:?}", first(emitted), first(rerendered))
            }
            other => format!("{other:?}"),
        }
    }
}

// ---------------------------------------------------------------------
// The check: the session's parse and render, as `run_call` runs them.
// ---------------------------------------------------------------------

fn user(text: &str) -> Message {
    Message {
        role: Role::User,
        content: Content::text(text.to_owned()),
    }
}

fn base_prompt(tools: &[crate::Tool]) -> Prompt {
    Prompt {
        messages: vec![user("Who checks the fog signal?")],
        tools: (!tools.is_empty())
            .then(|| tools.iter().map(|t| t.clone().into()).collect()),
        ..Prompt::default()
    }
}

/// The session's render options (`Session::from_engine`), with the
/// request's thinking switch.
fn render_opts(fx: &Fixture, thinking: bool) -> RenderOptions {
    RenderOptions::default()
        .with_extra("preserve_thinking", true)
        .with_extra("enable_thinking", thinking)
        .with_thought_reingest(fx.syntax.reasoning.reingest)
        .with_reasoning_start(fx.syntax.reasoning.start.clone())
}

/// The generation prompt a case is generated after.
fn generation_prompt(fx: &Fixture, case: &Case) -> String {
    fx.template
        .render_with(
            &base_prompt(&case.tools),
            &render_opts(fx, case.thinking).with_generation_prompt(true),
        )
        .expect("render")
}

/// Hold one case to the invariant. `None` when it round-trips — or when
/// the session would keep no tip for it anyway (a repeated call).
fn check(fx: &Fixture, case: &Case) -> Option<Failure> {
    let prompt = generation_prompt(fx, case);
    let syntax = effective_tool_syntax(&fx.syntax);
    let tools: Vec<&crate::Tool> = case.tools.iter().collect();
    let parsed = crate::dialect::parse_text(
        &syntax,
        &tools,
        &case.emission,
        render_ends_with_open_reasoning(&prompt, &fx.syntax),
        Leniency::Final,
    );
    if parsed.status != ParseStatus::Complete {
        return Some(Failure::ParseIncomplete);
    }
    // `run_call`: a repeated call is dropped and the turn keeps no tip,
    // by design; a whitespace-only text never reaches the client.
    let (mut blocks, dropped) = drop_repeated_calls(parsed.blocks, false);
    if dropped {
        return None;
    }
    blocks.retain(|block| {
        !matches!(block, crate::Block::Text { text, .. }
            if text.trim().is_empty())
    });
    let calls: Vec<crate::Block> = blocks
        .iter()
        .filter_map(|block| match block {
            crate::Block::ToolUse { call } => Some(crate::Block::ToolResult {
                result: misanthropic::tool::Result {
                    tool_use_id: call.id.clone(),
                    content: "Fog, 4°C.".into(),
                    is_error: false,
                    cache_control: None,
                },
            }),
            _ => None,
        })
        .collect();
    if case.mode == Mode::Forced && calls.is_empty() {
        return Some(Failure::ForcedNoCall);
    }
    // `render_extended`: the blocks seated as the session seats them.
    let asst: misanthropic::prompt::message::AssistantMessage =
        blocks.into_iter().collect();
    let mut turn = base_prompt(&case.tools);
    turn.messages.push(asst.into());
    [Next::Nothing, Next::Reply].into_iter().find_map(|next| {
        let mut turn = turn.clone();
        if next == Next::Reply {
            turn.messages.push(match calls.is_empty() {
                true => user("And the lamp?"),
                false => Message {
                    role: Role::User,
                    content: Content(calls.clone()),
                },
            });
        }
        let extended = fx
            .template
            .render_with(
                &turn,
                &render_opts(fx, case.thinking)
                    .with_generation_prompt(next == Next::Reply),
            )
            .expect("render");
        let at = emission_divergence(&extended, &prompt, &case.emission)?;
        let rerender = extended.strip_prefix(prompt.as_str()).unwrap_or("");
        let from = |s: &str| s.get(at..).unwrap_or("").to_owned();
        Some(Failure::Diverges {
            next,
            at,
            emitted: from(&case.emission),
            rerendered: from(rerender),
        })
    })
}

// ---------------------------------------------------------------------
// Admission: could the session have generated this emission?
// ---------------------------------------------------------------------

/// The grammar a case's calls run under: eager for a forced turn, lazy
/// (from the trigger) otherwise — as `dialect_grammar_for_prompt` and
/// `dialect_deferred_grammar_for_prompt` compile them.
fn grammar_for(
    fx: &Fixture,
    mode: Mode,
    parallel: bool,
    pre_opened: bool,
    tools: &[crate::Tool],
) -> Option<Arc<Grammar>> {
    let syntax = effective_tool_syntax(&fx.syntax);
    let anchor = match (mode, pre_opened) {
        (Mode::Text, _) => return None,
        (Mode::Auto, _) => Anchor::Lazy,
        (Mode::Forced, true) => Anchor::EagerThoughtPreOpened,
        (Mode::Forced, false) => Anchor::Eager,
    };
    let opts = EmitOptions {
        anchor,
        parallel: parallel && !syntax.per_call_start.is_empty(),
        ..EmitOptions::default()
    };
    let refs: Vec<&crate::Tool> = tools.iter().collect();
    let source = grammar_source(&syntax, &refs, &opts).ok()?;
    Grammar::parse(&source).ok().map(Arc::new)
}

/// Whether the session could have generated `case.emission`: the
/// grammar admits its constrained part, and no reasoning marker the
/// render already spent appears (the opener and closer bans).
fn admitted(fx: &Fixture, case: &Case) -> bool {
    let prompt = generation_prompt(fx, case);
    let pre_opened = render_ends_with_open_reasoning(&prompt, &fx.syntax);
    let closed = render_ends_with_closed_reasoning(&prompt, &fx.syntax);
    let start = fx.syntax.reasoning.start.trim();
    let end = fx.syntax.reasoning.end.trim();
    let tagged = fx.syntax.reasoning.mode != ReasoningMode::None;
    if tagged
        && (pre_opened || closed)
        && !start.is_empty()
        && case.emission.contains(start)
    {
        return false;
    }
    if tagged && closed && !end.is_empty() && case.emission.contains(end) {
        return false;
    }
    let grammar =
        grammar_for(fx, case.mode, case.parallel, pre_opened, &case.tools);
    let complete = |text: &str| {
        grammar.as_ref().is_some_and(|g| {
            let mut state = GrammarState::new(g.clone());
            state.advance_bytes(text.as_bytes()).is_ok() && state.is_complete()
        })
    };
    let triggers = effective_tool_syntax(&fx.syntax).triggers();
    let free = |text: &str| {
        triggers
            .iter()
            .all(|t| t.is_empty() || !text.contains(t.as_str()))
    };
    match case.mode {
        Mode::Forced => complete(&case.emission),
        // Free text up to the trigger; from there the grammar, which
        // holds to the end of the turn once it fires (after a call only
        // another call, or the end), admits the rest whole.
        Mode::Auto => case.walk_at().map_or(free(&case.emission), |at| {
            case.emission
                .split_at_checked(at)
                .is_some_and(|(text, walk)| free(text) && complete(walk))
        }),
        Mode::Text => free(&case.emission),
    }
}

// ---------------------------------------------------------------------
// Generation.
// ---------------------------------------------------------------------

/// A random tool: a few parameters of the kinds that have broken the
/// round trip before (`null`, enums quoted or not, nested objects) and
/// the plain ones around them.
fn gen_tool(rng: &mut SmallRng, index: usize) -> crate::Tool {
    let mut defs = Map::new();
    let names = [
        "city", "detail", "verbose", "days", "scale", "mode", "level", "units",
        "tier", "filter", "tags", "ids", "note", "limit",
    ];
    let mut picked: Vec<&str> = names.to_vec();
    picked.shuffle(rng);
    let count = rng.random_range(0..=4usize);
    let mut properties = Map::new();
    let mut required = Vec::new();
    for name in picked.into_iter().take(count) {
        properties.insert(name.into(), gen_param(rng, &mut defs, 0));
        if rng.random_bool(0.6) {
            required.push(Value::from(name));
        }
    }
    let mut schema = Map::new();
    schema.insert("type".into(), "object".into());
    schema.insert("properties".into(), Value::Object(properties));
    schema.insert("required".into(), Value::Array(required));
    if !defs.is_empty() {
        schema.insert("$defs".into(), Value::Object(defs));
    }
    let name = ["get_weather", "get_content", "vote"][index % 3];
    crate::Tool::builder(name)
        .description("A tool.")
        .schema(Value::Object(schema))
        .build()
        .expect("valid tool")
}

fn obj(pairs: &[(&str, Value)]) -> Value {
    Value::Object(
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), v.clone()))
            .collect(),
    )
}

fn typed(ty: &str) -> Value {
    obj(&[("type", ty.into())])
}

fn gen_param(
    rng: &mut SmallRng,
    defs: &mut Map<String, Value>,
    depth: u8,
) -> Value {
    let members = || {
        Value::Array(
            ["summary", "full", "a b", "null", "1"]
                .iter()
                .map(|&m| Value::from(m))
                .collect(),
        )
    };
    match rng.random_range(0..14u8) {
        0 => typed("string"),
        1 => {
            obj(&[("type", Value::Array(vec!["string".into(), "null".into()]))])
        }
        2 => typed("integer"),
        3 => typed("number"),
        4 => typed("boolean"),
        5 => obj(&[("type", "string".into()), ("enum", members())]),
        6 => obj(&[("type", "string".into()), ("const", "metric".into())]),
        // Agora's `Option<DetailLevel>`, as schemars emits it.
        7 => obj(&[(
            "anyOf",
            Value::Array(vec![
                obj(&[(
                    "oneOf",
                    Value::Array(vec![
                        obj(&[
                            ("type", "string".into()),
                            ("const", "summary".into()),
                        ]),
                        obj(&[
                            ("type", "string".into()),
                            ("const", "full".into()),
                        ]),
                    ]),
                )]),
                typed("null"),
            ]),
        )]),
        8 => {
            defs.insert(
                "Tier".into(),
                obj(&[
                    ("type", "string".into()),
                    ("enum", Value::Array(vec!["free".into(), "pro".into()])),
                ]),
            );
            obj(&[("$ref", "#/$defs/Tier".into())])
        }
        9 => obj(&[("type", "array".into()), ("items", typed("string"))]),
        10 => obj(&[("type", "array".into()), ("items", typed("integer"))]),
        11 | 12 if depth < 2 => {
            let inner: Map<String, Value> = ["z", "a", "m"]
                .iter()
                .take(rng.random_range(1..=3usize))
                .map(|&k| (k.to_owned(), gen_param(rng, defs, depth + 1)))
                .collect();
            let required: Vec<Value> = inner
                .keys()
                .filter(|_| rng.random_bool(0.5))
                .map(|k| Value::from(k.as_str()))
                .collect();
            obj(&[
                ("type", "object".into()),
                ("properties", Value::Object(inner)),
                ("required", Value::Array(required)),
            ])
        }
        // An enum with a null member, and an integer enum.
        12 => obj(&[("enum", Value::Array(vec!["on".into(), Value::Null]))]),
        _ => obj(&[
            ("type", "integer".into()),
            ("enum", Value::Array(vec![1.into(), 2.into(), 10.into()])),
        ]),
    }
}

/// Every string literal a GBNF source spells — the walker's shortcuts
/// through markers, member names and closing delimiters.
fn grammar_literals(source: &str) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut chars = source.chars();
    while let Some(c) = chars.next() {
        // A character class may hold a `"`: skip it whole.
        if c == '[' {
            while let Some(c) = chars.next() {
                match c {
                    '\\' => {
                        chars.next();
                    }
                    ']' => break,
                    _ => {}
                }
            }
            continue;
        }
        if c != '"' {
            continue;
        }
        let mut lit = String::new();
        while let Some(c) = chars.next() {
            match c {
                '"' => break,
                '\\' => match chars.next() {
                    Some('n') => lit.push('\n'),
                    Some('r') => lit.push('\r'),
                    Some('t') => lit.push('\t'),
                    Some(other) => lit.push(other),
                    None => break,
                },
                c => lit.push(c),
            }
        }
        if !lit.is_empty() {
            out.push(lit.into_bytes());
        }
    }
    // Multi-byte text the walker's byte steps never pick.
    out.extend(["é", "→", "🍓"].map(|s| s.as_bytes().to_vec()));
    out.sort();
    out.dedup();
    out
}

/// Bytes the walker prefers in free content: text, the JSON and XML
/// punctuation serializers disagree about, and whitespace.
const PALETTE: &[u8] = b"abcdefgxyzAB0123456789 \n\n\t.,:;'\"<>/{}[]=&\\-_";

/// A random walk of `grammar` to a complete state, or `None` when the
/// budget runs out first. Byte steps draw from the matcher's first-byte
/// set (ASCII; see `grammar_fuzz`'s walker), biased to [`PALETTE`];
/// literal jumps take a whole marker or delimiter at once, more often
/// as the walk grows, so walks close instead of wandering.
fn walk(
    grammar: &Arc<Grammar>,
    literals: &[Vec<u8>],
    rng: &mut SmallRng,
    budget: usize,
) -> Option<String> {
    let mut state = GrammarState::new(grammar.clone());
    let mut out: Vec<u8> = Vec::new();
    while out.len() < budget {
        if state.is_complete() && rng.random_bool(0.4) {
            break;
        }
        let jump = if out.len() * 2 > budget { 0.9 } else { 0.3 };
        if rng.random_bool(jump) {
            let mut order: Vec<&Vec<u8>> = literals.iter().collect();
            order.shuffle(rng);
            if let Some(lit) =
                order.into_iter().find(|lit| state.accepts_bytes(lit))
            {
                state.advance_bytes(lit).ok()?;
                out.extend_from_slice(lit);
                continue;
            }
        }
        let bitmap = state.first_byte_bitmap();
        let allowed: Vec<u8> = (0u8..128)
            .filter(|&b| bitmap[(b / 64) as usize] >> (b % 64) & 1 == 1)
            .filter(|&b| b >= b' ' || b == b'\n' || b == b'\t')
            .collect();
        if allowed.is_empty() {
            break;
        }
        let preferred: Vec<u8> = allowed
            .iter()
            .copied()
            .filter(|b| PALETTE.contains(b))
            .collect();
        let pool = match preferred.is_empty() || rng.random_bool(0.1) {
            true => &allowed,
            false => &preferred,
        };
        let stepped = (0..8).any(|_| {
            let Some(&b) = pool.choose(rng) else {
                return false;
            };
            let ok = state.advance_bytes(&[b]).is_ok();
            if ok {
                out.push(b);
            }
            ok
        });
        if !stepped {
            break;
        }
    }
    state
        .is_complete()
        .then(|| String::from_utf8(out).ok())
        .flatten()
}

/// Random prose: words, the characters serializers disagree about, and
/// whitespace at either edge — what a model's free text can be.
fn prose(rng: &mut SmallRng) -> String {
    let words = [
        "Ada",
        "checks",
        "it.",
        "José's",
        "\"B&B\",",
        "<2>",
        "→",
        "{x}",
        "[1, 2]",
        "null",
        "none",
        "True",
        "🍓",
        "a\\b",
        "{\"a\": 1}",
    ];
    let n = rng.random_range(1..=5usize);
    let body: Vec<&str> =
        (0..n).map(|_| *words.choose(rng).unwrap_or(&"")).collect();
    let joint = *[" ", " ", "\n", "\n\n"].choose(rng).unwrap_or(&" ");
    let edge = |rng: &mut SmallRng| {
        *["", "", "", "\n", "\n\n", " "].choose(rng).unwrap_or(&"")
    };
    let (lead, tail) = (edge(rng), edge(rng));
    format!("{lead}{}{tail}", body.join(joint))
}

/// A thought body: prose that never starts or ends in whitespace the
/// framing owns (a blank line before the close is framing's choice).
fn thought_body(rng: &mut SmallRng) -> String {
    prose(rng).trim().to_owned()
}

/// The dialect's habitual framing of `n` thoughts, as its own
/// round-trip tests pin it — free text, so the framing is the model's
/// habit rather than a grammar's.
fn thoughts(fx: &Fixture, rng: &mut SmallRng, n: usize) -> Vec<(Seg, String)> {
    let mut out = Vec::new();
    let mut push = |open: String, body: String, close: String| {
        out.extend([(Seg::Frame, open), (Seg::Free, body), (Seg::Frame, close)])
    };
    for i in 0..n {
        let body = thought_body(rng);
        match fx.syntax.family {
            Family::Harmony => push(
                harmony::ANALYSIS_OPEN.into(),
                body,
                format!("{}{}", harmony::END, harmony::START_ASSISTANT),
            ),
            Family::TagWithJson => {
                let gap = *["", "", "\n"].choose(rng).unwrap_or(&"");
                push("[THINK]".into(), body, format!("[/THINK]{gap}"));
            }
            Family::TagWithDict => {
                let gap = match i + 1 < n {
                    true => *["", "\n"].choose(rng).unwrap_or(&""),
                    false => "",
                };
                push(
                    "<|channel>thought\n".into(),
                    body,
                    format!("\n<channel|>{gap}"),
                );
            }
            // Cogito: its thought is prose.
            _ if fx.syntax.reasoning.mode == ReasoningMode::None => {
                push("<think>\n".into(), body, "\n</think>\n\n".into())
            }
            // Qwen: the render pre-opens the first.
            _ => {
                let blank = *["", "", "\n"].choose(rng).unwrap_or(&"");
                let gap = *["\n\n", "\n\n", "\n"].choose(rng).unwrap_or(&"");
                let open = if i == 0 { "" } else { "<think>\n" };
                push(open.into(), body, format!("\n{blank}</think>{gap}"));
            }
        }
    }
    out
}

/// Text a model might write after its last call: the dialect's own
/// closing markers again, whitespace, and prose.
fn trailing_text(fx: &Fixture, rng: &mut SmallRng) -> Vec<String> {
    let syntax = &fx.syntax;
    [
        syntax.per_call_end.trim(),
        syntax.section_end.trim(),
        syntax.function.close.trim(),
        syntax.tool_response_start.trim(),
        harmony::CALL,
    ]
    .into_iter()
    .filter(|m| !m.is_empty())
    .map(str::to_owned)
    .chain(["\n".to_owned(), "\n\n".to_owned(), prose(rng)])
    .collect()
}

/// Whether `fx` reasons in markers a thinking switch drives.
fn reasons(fx: &Fixture) -> bool {
    fx.syntax.family == Family::Harmony
        || fx.syntax.reasoning.mode != ReasoningMode::None
        || fx.syntax.family == Family::JsonNative
}

/// One random case for `fx`, or `None` when its walk ran out of budget.
fn gen_case(fx: &Fixture, rng: &mut SmallRng) -> Option<Case> {
    let mode = *[Mode::Forced, Mode::Auto, Mode::Auto, Mode::Text]
        .choose(rng)
        .unwrap_or(&Mode::Auto);
    let harmony = fx.syntax.family == Family::Harmony;
    // Harmony always reasons; the switch moves nothing there.
    let thinking = harmony || (reasons(fx) && rng.random_bool(0.6));
    let parallel = rng.random_bool(0.5);
    let tools: Vec<crate::Tool> = (0..rng.random_range(1..=2usize))
        .map(|i| gen_tool(rng, i))
        .collect();
    let case = Case {
        mode,
        thinking,
        parallel,
        tools,
        segments: Vec::new(),
        emission: String::new(),
    };
    let prompt = generation_prompt(fx, &case);
    let pre_opened = render_ends_with_open_reasoning(&prompt, &fx.syntax);
    let spent =
        pre_opened || render_ends_with_closed_reasoning(&prompt, &fx.syntax);
    let walked = |rng: &mut SmallRng| {
        let grammar = grammar_for(fx, mode, parallel, pre_opened, &case.tools)?;
        let mut literals = grammar_literals(grammar.source());
        // The thought close sits in an until-rule, spelled byte by byte.
        let r = &fx.syntax.reasoning;
        literals.extend(
            [
                r.end.as_str(),
                r.end.trim(),
                r.separator.as_deref().unwrap_or(""),
            ]
            .into_iter()
            .filter(|m| !m.is_empty())
            .map(|m| m.as_bytes().to_vec()),
        );
        walk(&grammar, &literals, rng, 600).map(|w| (Seg::Walk, w))
    };
    let segments = match mode {
        Mode::Forced => vec![walked(rng)?],
        Mode::Auto | Mode::Text => {
            // Thoughts: the pre-opened one is owed; none when the render
            // already closed the turn's.
            let n = match (pre_opened, spent, thinking) {
                // One: the opener ban rules out a second.
                (true, _, _) => 1,
                (false, true, _) | (false, false, false) => 0,
                (false, false, true) => rng.random_range(0..=2usize),
            };
            let mut segs = thoughts(fx, rng, n);
            if harmony {
                // A preamble, then a call or the final.
                if rng.random_bool(0.4) {
                    segs.extend([
                        (Seg::Frame, harmony::COMMENTARY_OPEN.to_owned()),
                        (Seg::Free, thought_body(rng)),
                        (
                            Seg::Frame,
                            format!(
                                "{}{}",
                                harmony::END,
                                harmony::START_ASSISTANT
                            ),
                        ),
                    ]);
                }
            } else if rng.random_bool(0.7) {
                segs.push((Seg::Free, prose(rng)));
            }
            match mode {
                Mode::Auto => {
                    let walk = walked(rng)?;
                    // A role-header call opens with the `<|start|>assistant`
                    // the block before it closed on — the trigger spans
                    // both — and cannot open the turn at all (the prompt
                    // holds that opener, so no trigger fires).
                    if walk.1.starts_with(harmony::START_ASSISTANT) {
                        let (_, prev) = segs.last_mut()?;
                        let kept =
                            prev.strip_suffix(harmony::START_ASSISTANT)?;
                        *prev = kept.to_owned();
                    }
                    segs.push(walk);
                }
                _ if harmony => {
                    let json = rng.random_bool(0.4) && n > 0;
                    segs.extend(match json {
                        true => [
                            (
                                Seg::Frame,
                                "<|channel|>final <|constrain|>json<|message|>"
                                    .to_owned(),
                            ),
                            (Seg::Free, r#"{"answer":"Ada","n":[1,2]}"#.into()),
                        ],
                        false => [
                            (Seg::Frame, harmony::FINAL_OPEN.to_owned()),
                            (Seg::Free, prose(rng)),
                        ],
                    });
                }
                _ => {}
            }
            segs
        }
    };
    Some(case.with_segments(segments))
}

// ---------------------------------------------------------------------
// Known divergences, and minimization.
// ---------------------------------------------------------------------

/// An accepted divergence: why it is accepted, and the case with it
/// respelled away — the re-render's bytes in place of the emission's —
/// so the rest of the emission is still checked past it.
struct Known {
    reason: &'static str,
    respelled: Option<Case>,
}

/// The number run of `s` from its start.
fn number_run(s: &str) -> &str {
    let numeric = |c: char| c.is_ascii_digit() || ".eE+-".contains(c);
    let end = s.find(|c: char| !numeric(c)).unwrap_or(s.len());
    s.get(..end).unwrap_or("")
}

/// Whether the JSON object `text` opens with repeats a member name.
fn repeats_a_key(text: &str) -> bool {
    let mut keys = std::collections::HashSet::new();
    let (mut depth, mut in_string, mut escaped) = (0usize, false, false);
    let mut current = String::new();
    let mut last_string: Option<String> = None;
    for c in text.chars() {
        if in_string {
            match (escaped, c) {
                (true, _) => escaped = false,
                (false, '\\') => escaped = true,
                (false, '"') => {
                    in_string = false;
                    last_string = Some(std::mem::take(&mut current));
                    continue;
                }
                _ => {}
            }
            current.push(c);
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' | '[' => depth += 1,
            '}' | ']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return false;
                }
            }
            ':' if depth == 1 => {
                if let Some(key) = last_string.take() {
                    if !keys.insert(key) {
                        return true;
                    }
                }
            }
            _ => {}
        }
        if !c.is_whitespace() && c != ':' {
            last_string = None;
        }
    }
    false
}

/// The member names of the dict-encoded object `text` opens with, in
/// order (Gemma 4's bare keys, which may hold brackets; strings between
/// `quote` markers), read the way the parser reads them.
fn dict_keys(text: &str, quote: &str) -> Vec<String> {
    /// Past one value at the head of `rest`, or `None` when malformed.
    fn skip_value<'t>(rest: &'t str, quote: &str) -> Option<&'t str> {
        if !quote.is_empty() && rest.starts_with(quote) {
            let body = &rest[quote.len()..];
            return Some(&body[body.find(quote)? + quote.len()..]);
        }
        // `skip_value` is only ever at a value: a key is read by
        // `object` up to its `:`, marker or not.
        match rest.chars().next()? {
            '{' => object(rest, quote, &mut Vec::new()),
            '[' => {
                let mut rest = &rest[1..];
                loop {
                    if let Some(after) = rest.strip_prefix(']') {
                        return Some(after);
                    }
                    rest = skip_value(rest, quote)?;
                    rest = rest.strip_prefix(',').unwrap_or(rest);
                }
            }
            _ => Some(&rest[rest.find([',', '}', ']'])?..]),
        }
    }
    /// Past the object at the head of `rest`, its keys pushed.
    fn object<'t>(
        rest: &'t str,
        quote: &str,
        keys: &mut Vec<String>,
    ) -> Option<&'t str> {
        let mut rest = rest.strip_prefix('{')?;
        loop {
            if let Some(after) = rest.strip_prefix('}') {
                return Some(after);
            }
            let colon = rest.find(':')?;
            keys.push(rest[..colon].to_owned());
            rest = skip_value(&rest[colon + 1..], quote)?;
            rest = rest.strip_prefix(',').unwrap_or(rest);
        }
    }
    let mut keys = Vec::new();
    let _ = object(text, quote, &mut keys);
    keys
}

/// `case` with `len` bytes at `start` replaced by `with`, inside the
/// segment that holds them.
fn respell(case: &Case, start: usize, len: usize, with: &str) -> Option<Case> {
    let mut offset = 0;
    let mut segments = case.segments.clone();
    let (_, text) = segments.iter_mut().find(|(_, text)| {
        offset += text.len();
        offset > start
    })?;
    let local = start + text.len() - offset;
    *text = format!("{}{with}{}", text.get(..local)?, text.get(local + len..)?);
    Some(case.with_segments(segments))
}

/// The accepted classes, each with the reason it is accepted: the
/// irreducible ones `templates/README.md` lists, and the `PENDING`
/// ones — real, reported on #129, awaiting a decision. `None` for
/// anything else — a finding.
fn known_divergence(
    fx: &Fixture,
    case: &Case,
    failure: &Failure,
) -> Option<Known> {
    let Failure::Diverges {
        at,
        emitted,
        rerendered,
        ..
    } = failure
    else {
        return None;
    };
    let emission = &case.emission;
    // A number's spelling is not in its value: `1.50`, `1e3`, `-0` and
    // friends re-render canonically, and the grammar's `number` admits
    // them all.
    let start = emission
        .get(..*at)?
        .char_indices()
        .rfind(|&(_, c)| !(c.is_ascii_digit() || ".eE+-".contains(c)))
        .map_or(0, |(i, c)| i + c.len_utf8());
    let head = emission.get(start..*at)?;
    let ours = format!("{head}{}", number_run(emitted));
    let theirs = format!("{head}{}", number_run(rerendered));
    // Equal up to serde_json's float parse, which is not correctly
    // rounded without its `float_roundtrip` feature (`572e-92` re-renders
    // `5.7199999999999995e-90`).
    let same = match (ours.parse::<f64>(), theirs.parse::<f64>()) {
        (Ok(a), Ok(b)) => {
            a == b || (a - b).abs() <= a.abs().max(b.abs()) * 1e-12
        }
        _ => false,
    };
    if ours != theirs && same {
        return Some(Known {
            reason: "non-canonical number spelling",
            respelled: Some(respell(case, start, ours.len(), &theirs)?),
        });
    }
    // PENDING (#120): Gemma 4's dict grammar admits `null`, `none` and
    // `None`, and the template prints whichever spelling the resolved
    // minijinja's `Display` gives (`none` before 2.20, `None` after).
    // Which one to force is the model's habit, still unmeasured.
    let nulls = ["null", "none", "None"];
    let spelled = |s: &str| nulls.into_iter().find(|n| s.starts_with(n));
    if fx.syntax.family == Family::TagWithDict {
        let word_start = emission
            .get(..*at)?
            .char_indices()
            .rfind(|&(_, c)| !c.is_ascii_alphabetic())
            .map_or(0, |(i, c)| i + c.len_utf8());
        let ours = spelled(emission.get(word_start..)?);
        let theirs =
            spelled(&format!("{}{rerendered}", emission.get(word_start..*at)?));
        if let (Some(ours), Some(theirs)) = (ours, theirs) {
            return Some(Known {
                reason: "PENDING #120: Gemma null spelling",
                respelled: Some(respell(case, word_start, ours.len(), theirs)?),
            });
        }
    }
    // PENDING (reported, #129): with thinking on, Gemma 4's generation
    // prompt leaves the thought to the model, and a turn that writes
    // none re-renders with the thinking-off scaffold — which is also
    // how an empty thought channel the model *did* write re-renders, so
    // the two cannot both round-trip without the parse telling them
    // apart.
    let scaffold = "<|channel>thought\n<channel|>";
    let turn = format!("{}{rerendered}", emission.get(..*at)?);
    let inserted = turn.find(scaffold).is_some_and(|p| {
        p <= *at
            && !emission
                .get(p..)
                .is_some_and(|e| e.starts_with("<|channel>thought"))
    });
    if fx.syntax.family == Family::TagWithDict && case.thinking && inserted {
        return Some(Known {
            reason: "PENDING: Gemma thinking-on turn without a thought",
            respelled: None,
        });
    }
    // PENDING (reported, #129): a Harmony call addressed in the role
    // header (`<|start|>assistant to=functions.X<|channel|>…`) or on
    // the analysis channel re-renders in the trained channel-header
    // form; the call keeps no record of the header it was written in.
    let header = case
        .walk_at()
        .and_then(|w| Some((w, emission.get(w..)?)))
        .filter(|(_, walk)| {
            walk.starts_with(&format!(
                "{}{}",
                harmony::START_ASSISTANT,
                harmony::TO_FUNCTIONS
            )) || walk.starts_with(&format!(
                "{}analysis{}",
                harmony::CHANNEL,
                harmony::TO_FUNCTIONS
            ))
        })
        .and_then(|(w, walk)| Some(w..w + walk.find(harmony::MESSAGE)?));
    if fx.syntax.family == Family::Harmony
        && header.is_some_and(|h| h.contains(at))
    {
        return Some(Known {
            reason: "PENDING: Harmony call header in role/analysis form",
            respelled: None,
        });
    }
    // The thought grammar closes on the *trimmed* marker, so a model
    // that drops the newline before it can still leave the region (see
    // `grammar_source`); the re-render puts the canonical newline back.
    let (start, end) = (
        fx.syntax.reasoning.start.trim(),
        fx.syntax.reasoning.end.trim(),
    );
    if !end.is_empty()
        && emitted.starts_with(end)
        && rerendered.trim_start().starts_with(end)
        && rerendered.starts_with(char::is_whitespace)
    {
        return Some(Known {
            reason: "thought closed without its newline",
            respelled: None,
        });
    }
    // An empty thought re-renders as the thinking-off scaffold, which
    // keeps neither a blank body nor the gap after it.
    let before = emission.get(..*at)?;
    let opened = match start.is_empty() {
        true => None,
        false => before.rfind(start).map(|i| i + start.len()),
    };
    let thought = opened.or_else(|| {
        (fx.syntax.reasoning.mode != ReasoningMode::None).then_some(0)
    });
    let empty = thought.is_some_and(|from| {
        let body = emission.get(from..).unwrap_or("");
        body.find(end).is_some_and(|close| {
            let after = &body[close + end.len()..];
            let gap = after.len() - after.trim_start().len();
            body[..close].trim().is_empty()
                && *at <= from + close + end.len() + gap
        })
    });
    if !end.is_empty() && empty && fx.syntax.family != Family::Harmony {
        return Some(Known {
            reason: "empty thought",
            respelled: None,
        });
    }
    // Untyped objects in the calls before the divergence. Any object
    // that opens there is tried: member names may spell any marker, so
    // no scan can find the one enclosing `at` reliably, and a case that
    // holds one of these classes diverges at it.
    let quote = &fx.syntax.arguments.string_quote;
    let walk = case.walk_at().unwrap_or(0);
    let objects: Vec<&str> = emission
        .get(walk.min(*at)..*at)?
        .char_indices()
        .filter(|&(_, c)| c == '{')
        .filter_map(|(i, _)| emission.get(walk + i..))
        .collect();
    // An untyped object may repeat a key, which a map cannot: the
    // grammar admits any member names there, as JSON does.
    if objects.iter().any(|o| repeats_a_key(o)) {
        return Some(Known {
            reason: "repeated key in an untyped object",
            respelled: None,
        });
    }
    // Gemma 4's template `dictsort`s every object, so a grammar that
    // spells known members alphabetically cannot do the same for the
    // free member names of an untyped one.
    let unsorted = |o: &&str| {
        dict_keys(o, quote)
            .windows(2)
            .any(|w| w[0].to_lowercase() > w[1].to_lowercase())
    };
    if fx.syntax.family == Family::TagWithDict && objects.iter().any(unsorted) {
        return Some(Known {
            reason: "unsorted keys in an untyped dict object",
            respelled: None,
        });
    }
    // A constrained final with no thought right before it has nowhere
    // to record its header, and re-renders plain.
    if emitted.starts_with(" <|constrain|>json<|message|>")
        && rerendered.starts_with(harmony::MESSAGE)
        && !emission
            .get(..*at)?
            .rsplit(harmony::START_ASSISTANT)
            .nth(1)
            .is_some_and(|block| block.starts_with(harmony::ANALYSIS_OPEN))
    {
        return Some(Known {
            reason: "constrained final after a non-analysis block",
            respelled: None,
        });
    }
    // Prose run straight into the call: no block records the gap the
    // model omitted, so the template's canonical one comes back.
    let prose_end = case.walk_at().filter(|&w| w == *at && w > 0);
    let gap: String = rerendered
        .chars()
        .take_while(|c| c.is_whitespace())
        .collect();
    if prose_end.is_some()
        && !emission.get(..*at)?.ends_with(char::is_whitespace)
        && !gap.is_empty()
        && rerendered
            .get(gap.len()..)?
            .starts_with(emitted.get(..emitted.len().min(16))?)
    {
        return Some(Known {
            reason: "gap omitted between prose and call",
            respelled: None,
        });
    }
    None
}

/// Accepted divergences met, by reason.
type Tally = std::collections::BTreeMap<&'static str, usize>;

/// The failure a case reaches once accepted divergences are respelled
/// out of it, or `None`; each accepted one is counted in `tally`.
fn unexplained(
    fx: &Fixture,
    case: &Case,
    tally: &mut Tally,
) -> Option<(Case, Failure)> {
    let mut case = case.clone();
    for _ in 0..16 {
        let failure = check(fx, &case)?;
        let Some(known) = known_divergence(fx, &case, &failure) else {
            return Some((case, failure));
        };
        *tally.entry(known.reason).or_default() += 1;
        case = known.respelled?;
        if !admitted(fx, &case) {
            return None;
        }
    }
    None
}

/// Delta-debug each free and walked segment of `case` down to a
/// minimal case the session could still have generated that fails the
/// same way. Framing stays whole.
fn minimize(fx: &Fixture, case: &Case, failure: &Failure) -> Case {
    let signature = failure.signature();
    let still = |c: &Case| {
        admitted(fx, c)
            && unexplained(fx, c, &mut Tally::new())
                .is_some_and(|(_, f)| f.signature() == signature)
    };
    let mut case = case.clone();
    for seg in 0..case.segments.len() {
        if case.segments[seg].0 == Seg::Frame {
            continue;
        }
        let mut chars: Vec<char> = case.segments[seg].1.chars().collect();
        let mut chunk = chars.len().div_ceil(2);
        while chunk >= 1 {
            let mut i = 0;
            let mut shrank = false;
            while i < chars.len() {
                let end = (i + chunk).min(chars.len());
                let mut segments = case.segments.clone();
                let cut: String =
                    chars[..i].iter().chain(&chars[end..]).collect();
                // Prose stays prose: a blank text is its own (accepted)
                // class, never what a cut should turn a finding into.
                if case.segments[seg].0 == Seg::Free && cut.trim().is_empty() {
                    i += chunk;
                    continue;
                }
                segments[seg].1 = cut;
                let trial = case.with_segments(segments);
                if still(&trial) {
                    chars.drain(i..end);
                    case = trial;
                    shrank = true;
                } else {
                    i += chunk;
                }
            }
            if !shrank {
                chunk /= 2;
            }
        }
    }
    case
}

// ---------------------------------------------------------------------
// The runs.
// ---------------------------------------------------------------------

/// `ORACLE_SEED` (a number, or `random`) and `ORACLE_CASES` (per
/// template), else the smoke run's fixed defaults.
fn run_config() -> (u64, usize) {
    let seed = match std::env::var("ORACLE_SEED").ok().as_deref() {
        None => 0x5eed_0129,
        Some("random") => std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(1, |d| d.as_nanos() as u64),
        Some(n) => n.parse().expect("ORACLE_SEED: a u64 or `random`"),
    };
    let cases = std::env::var("ORACLE_CASES")
        .ok()
        .map_or(120, |n| n.parse().expect("ORACLE_CASES: a count"));
    (seed, cases)
}

fn report(fx: &Fixture, case: &Case, failure: &Failure) -> String {
    let schemas: Vec<String> = case
        .tools
        .iter()
        .map(|t| format!("{}: {}", t.name, t.schema))
        .collect();
    let what = match failure {
        Failure::Diverges {
            next,
            at,
            emitted,
            rerendered,
        } => {
            let cut = |s: &str| s.chars().take(32).collect::<String>();
            format!(
                "{next:?} at {at}: emitted {:?}, re-rendered {:?}",
                cut(emitted),
                cut(rerendered)
            )
        }
        other => format!("{other:?}"),
    };
    format!(
        "{}: {:?} thinking={} parallel={}: {what}\n  emission: {:?}\n  \
         tools: {}",
        fx.name,
        case.mode,
        case.thinking,
        case.parallel,
        case.emission,
        schemas.join("; ")
    )
}

/// The oracle: random cases per baked template, every failure
/// minimized, distinct ones reported together.
#[test]
fn round_trip_oracle() {
    let (seed, per_fixture) = run_config();
    eprintln!(
        "round_trip_oracle: ORACLE_SEED={seed} ORACLE_CASES={per_fixture}"
    );
    let mut findings: Vec<(String, String)> = Vec::new();
    let mut tally = Tally::new();
    for (i, fx) in fixtures().iter().enumerate() {
        let mut rng = SmallRng::seed_from_u64(seed ^ ((i as u64) << 32));
        let (mut ran, mut tries) = (0, 0);
        let mut modes = std::collections::BTreeMap::<String, usize>::new();
        while ran < per_fixture && tries < per_fixture * 4 {
            tries += 1;
            let Some(case) = gen_case(fx, &mut rng) else {
                continue;
            };
            if !admitted(fx, &case) {
                continue;
            }
            ran += 1;
            *modes.entry(format!("{:?}", case.mode)).or_default() += 1;
            // Text after the last call must be refused by the grammar or
            // round-trip: a stray close (Qwen3.6, live 2026-10-01),
            // whitespace, prose.
            for tail in trailing_text(fx, &mut rng) {
                let mut segments = case.segments.clone();
                match segments.last_mut() {
                    Some((Seg::Walk, walk)) => walk.push_str(&tail),
                    _ => break,
                }
                let probe = case.with_segments(segments);
                if !admitted(fx, &probe) {
                    continue;
                }
                if let Some((probe, failure)) =
                    unexplained(fx, &probe, &mut tally)
                {
                    let key = format!(
                        "{}:after-call:{}",
                        fx.name,
                        failure.signature()
                    );
                    if !findings.iter().any(|(k, _)| *k == key) {
                        findings.push((key, report(fx, &probe, &failure)));
                    }
                }
            }
            let Some((case, failure)) = unexplained(fx, &case, &mut tally)
            else {
                continue;
            };
            let key =
                format!("{}:{:?}:{}", fx.name, case.mode, failure.signature());
            if findings.iter().any(|(k, _)| *k == key) {
                continue;
            }
            let small = minimize(fx, &case, &failure);
            let failure = check(fx, &small).unwrap_or(failure);
            findings.push((key, report(fx, &small, &failure)));
        }
        eprintln!("round_trip_oracle: {}: {modes:?}", fx.name);
        assert!(
            ran * 2 >= per_fixture,
            "{}: only {ran} of {per_fixture} cases generated",
            fx.name
        );
    }
    eprintln!("round_trip_oracle: accepted divergences: {tally:?}");
    assert!(
        findings.is_empty(),
        "ORACLE_SEED={seed}: {} distinct round-trip failures:\n{}",
        findings.len(),
        findings
            .iter()
            .map(|(_, r)| r.as_str())
            .collect::<Vec<_>>()
            .join("\n")
    );
}

// ---------------------------------------------------------------------
// The corpus: past live failures and every finding, replayed.
// ---------------------------------------------------------------------

/// What a corpus case must do.
#[derive(Debug)]
enum Expect {
    /// Admitted, and re-renders byte for byte.
    RoundTrips,
    /// No longer admitted: the grammar (or a ban) rules it out.
    NotAdmitted,
    /// Admitted, and diverges only in this accepted class.
    Accepted(&'static str),
}

/// The corpus tool: one parameter of each kind a seed needs.
fn corpus_tool() -> crate::Tool {
    let schema = obj(&[
        ("type", "object".into()),
        (
            "properties",
            obj(&[
                ("city", typed("string")),
                (
                    "detail",
                    obj(&[(
                        "type",
                        Value::Array(vec!["string".into(), "null".into()]),
                    )]),
                ),
                (
                    "mode",
                    obj(&[
                        ("type", "string".into()),
                        (
                            "enum",
                            Value::Array(vec!["summary".into(), "full".into()]),
                        ),
                    ]),
                ),
                ("scale", typed("number")),
                ("filter", typed("object")),
            ]),
        ),
        ("required", Value::Array(vec!["city".into()])),
    ]);
    crate::Tool::builder("get_weather")
        .description("A tool.")
        .schema(schema)
        .build()
        .expect("valid tool")
}

/// One corpus case on `fixture`, from its segments.
fn seed(
    fixture: &'static str,
    mode: Mode,
    thinking: bool,
    segments: &[(Seg, &str)],
    expect: Expect,
) -> (&'static str, Case, Expect) {
    let case = Case {
        mode,
        thinking,
        parallel: true,
        tools: vec![corpus_tool()],
        segments: Vec::new(),
        emission: String::new(),
    };
    let segments = segments.iter().map(|&(k, s)| (k, s.to_owned())).collect();
    (fixture, case.with_segments(segments), expect)
}

/// The live failures the oracle was seeded with (#129), and its own
/// minimized findings, each held to what it must now do.
#[test]
fn round_trip_oracle_corpus() {
    use Expect::*;
    use Mode::*;
    use Seg::*;
    let qwen_call = |param: &str| {
        format!(
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n\
             </parameter>\n{param}</function>\n</tool_call>"
        )
    };
    let detail_null = qwen_call("<parameter=detail>\nnull\n</parameter>\n");
    let mode_raw = qwen_call("<parameter=mode>\nfull\n</parameter>\n");
    let mode_quoted = qwen_call("<parameter=mode>\n\"full\"\n</parameter>\n");
    let scale_long = qwen_call("<parameter=scale>\n1.50\n</parameter>\n");
    let cogito_call = "<tool_call>\n{\"name\": \"get_weather\", \
                       \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>";
    let leading_ws = format!("\n{}", qwen_call(""));
    let analysis = "<|channel|>analysis<|message|>";
    let next = "<|end|><|start|>assistant";
    let gptoss = "gptoss-cache-stable";
    let chan = "<|channel|>commentary to=functions.get_weather";
    let gemma_call = |args: &str| {
        format!(
            "<|tool_call>call:get_weather{{{args}}}<tool_call|>\
             <|tool_response>"
        )
    };
    let cases = [
        // 2026-10-01, Qwen3.6: `null` re-rendered `none` (stock `| string`).
        seed(
            "qwen3.6-cache-stable",
            Auto,
            false,
            &[(Walk, &detail_null)],
            RoundTrips,
        ),
        // 2026-10-01, Agora's `detail`: a quoted enum member re-rendered
        // raw; the grammar now writes members raw.
        seed(
            "qwen3.6-cache-stable",
            Auto,
            false,
            &[(Walk, &mode_raw)],
            RoundTrips,
        ),
        seed(
            "qwen3.6-cache-stable",
            Auto,
            false,
            &[(Walk, &mode_quoted)],
            NotAdmitted,
        ),
        // Cogito's prose-to-call gap: stock printed a `\n` more.
        seed(
            "cogito-cache-stable",
            Auto,
            false,
            &[(Free, "Checking.\n\n"), (Walk, cogito_call)],
            RoundTrips,
        ),
        // gpt-oss's `<|constrain|>json` final, dropped on re-render.
        seed(
            gptoss,
            Text,
            true,
            &[
                (Frame, analysis),
                (Free, "Plan."),
                (Frame, next),
                (Frame, "<|channel|>final <|constrain|>json<|message|>"),
                (Free, "{\"a\":1}"),
            ],
            RoundTrips,
        ),
        // Mistral 4's back-to-back thoughts, merged into one block.
        seed(
            "mistral4-cache-stable",
            Text,
            true,
            &[
                (Frame, "[THINK]"),
                (Free, "A."),
                (Frame, "[/THINK][THINK]"),
                (Free, "B."),
                (Frame, "[/THINK]"),
                (Free, "Ada."),
            ],
            RoundTrips,
        ),
        // 2026-10-02, Mistral Small 4: `[THINK]` inside an open thought,
        // rejected by #101 on every retry. The sampler now steers it to
        // `[/THINK]`; the model then answers, or reopens.
        seed(
            "mistral4-cache-stable",
            Text,
            true,
            &[
                (Frame, "[THINK]"),
                (Free, "invite others to read the decisions with me.\""),
                (Frame, "[/THINK]"),
                (Free, "**Reflection:**\n\nI've introduced myself."),
            ],
            RoundTrips,
        ),
        seed(
            "mistral4-cache-stable",
            Text,
            true,
            &[
                (Frame, "[THINK]"),
                (Free, "resonance, patience, and active listening.\""),
                (Frame, "[/THINK][THINK]"),
                (Free, "**Reflection:**\n\nI've introduced myself."),
                (Frame, "[/THINK]"),
                (Free, "Ada."),
            ],
            RoundTrips,
        ),
        // The same steer on a `<think>` dialect.
        seed(
            "cogito-cache-stable",
            Text,
            true,
            &[
                (Frame, "<think>"),
                (Free, "A."),
                (Frame, "</think><think>"),
                (Free, "B."),
                (Frame, "</think>"),
                (Free, "Ada."),
            ],
            RoundTrips,
        ),
        // Oracle findings, fixed: whitespace before a forced call.
        seed(
            "qwen3.8-cache-stable",
            Forced,
            false,
            &[(Walk, &leading_ws)],
            NotAdmitted,
        ),
        // A `\u` escape of a character the serializer writes raw.
        seed(
            "mistral4-cache-stable",
            Auto,
            false,
            &[(Walk, "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"\\u00e9\"}")],
            NotAdmitted,
        ),
        seed(
            "mistral4-cache-stable",
            Auto,
            false,
            &[(Walk, "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"é\"}")],
            RoundTrips,
        ),
        // A Harmony channel-header call in a non-canonical spelling.
        seed(
            gptoss,
            Auto,
            true,
            &[(
                Walk,
                &format!("{chan} json<|message|>{{\"city\":\"Paris\"}}"),
            )],
            NotAdmitted,
        ),
        seed(
            gptoss,
            Auto,
            true,
            &[(
                Walk,
                &format!(
                    "{chan} <|constrain|>json<|message|>{{\"city\":\"Paris\"}}"
                ),
            )],
            RoundTrips,
        ),
        // Whitespace at the edge of an untyped dict key.
        seed(
            "gemma4-cache-stable",
            Auto,
            false,
            &[(Walk, &gemma_call("city:<|\"|>Paris<|\"|>,filter:{d :1}"))],
            NotAdmitted,
        ),
        seed(
            "gemma4-cache-stable",
            Auto,
            false,
            &[(Walk, &gemma_call("city:<|\"|>Paris<|\"|>,filter:{d:1}"))],
            RoundTrips,
        ),
        // A blank preamble before a forced Harmony call.
        seed(
            gptoss,
            Forced,
            true,
            &[(
                Walk,
                &format!(
                    "<|channel|>commentary<|message|>{next}{chan} \
                     <|constrain|>json<|message|>{{\"city\":\"Paris\"}}"
                ),
            )],
            NotAdmitted,
        ),
        // Accepted classes, pinned so a fix flips them deliberately.
        seed(
            "qwen3.6-cache-stable",
            Auto,
            false,
            &[(Walk, &scale_long)],
            Accepted("non-canonical number spelling"),
        ),
        seed(
            "gemma4-cache-stable",
            Forced,
            true,
            &[(Walk, &gemma_call("city:<|\"|>Paris<|\"|>"))],
            Accepted("PENDING: Gemma thinking-on turn without a thought"),
        ),
        seed(
            gptoss,
            Auto,
            true,
            &[
                (Frame, "<|channel|>commentary<|message|>"),
                (Free, "Checking."),
                (Frame, "<|end|>"),
                (
                    Walk,
                    "<|start|>assistant to=functions.get_weather<|channel|>\
                     commentary json<|message|>{\"city\":\"Paris\"}",
                ),
            ],
            Accepted("PENDING: Harmony call header in role/analysis form"),
        ),
        seed(
            "gemma4-cache-stable",
            Auto,
            false,
            &[(Walk, &gemma_call("city:<|\"|>Paris<|\"|>,filter:{b:1,a:2}"))],
            Accepted("unsorted keys in an untyped dict object"),
        ),
        // Live 2026-10-01 (gpt-oss, RC e559a44): `to=create_comment`.
        // The lazy grammar arms at `to=` and forces `functions.`.
        seed(
            gptoss,
            Auto,
            true,
            &[
                (Frame, analysis),
                (Free, "Plan."),
                (Frame, next),
                (
                    Free,
                    "<|channel|>commentary to=get_weather \
                     <|constrain|>json<|message|>{\"city\":\"Paris\"}",
                ),
            ],
            NotAdmitted,
        ),
        // Text after a call, here a second thought and call: the lazy
        // grammar holds to the end of the turn, so it is never written.
        seed(
            "qwen3.6-cache-stable",
            Auto,
            true,
            &[
                (Free, "Plan."),
                (Frame, "\n</think>\n\n"),
                (Walk, &qwen_call("")),
                (Free, "\nThe proposals page for context."),
                (Frame, "\n</think>\n\n"),
                (Walk, &mode_raw),
            ],
            NotAdmitted,
        ),
        // Live 2026-10-01 (Qwen3.6, a 12.7k-token tip): a stray close
        // after the call, parsed to a text block the bake renders
        // before the calls. No grammar admits it, on any tagged dialect.
        seed(
            "qwen3.6-cache-stable",
            Auto,
            true,
            &[
                (Free, "Plan."),
                (Frame, "\n</think>\n\n"),
                (Walk, &format!("{}</tool_call>", qwen_call(""))),
            ],
            NotAdmitted,
        ),
        seed(
            "qwen3.8-cache-stable",
            Auto,
            false,
            &[(Walk, &format!("{}\n</tool_call>", qwen_call("")))],
            NotAdmitted,
        ),
        seed(
            "cogito-cache-stable",
            Auto,
            false,
            &[(Walk, &format!("{cogito_call}\n</tool_call>"))],
            NotAdmitted,
        ),
        seed(
            "mistral4-cache-stable",
            Auto,
            false,
            &[(
                Walk,
                "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}Done.",
            )],
            NotAdmitted,
        ),
        seed(
            "gemma4-cache-stable",
            Auto,
            false,
            &[(Walk, &(gemma_call("city:<|\"|>Paris<|\"|>") + "Done."))],
            NotAdmitted,
        ),
        seed(
            "mistral4-cache-stable",
            Auto,
            false,
            &[(
                Walk,
                "[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\", \
                 \"filter\": {\"a\": {}, \"b\": 1, \"b\": 2}}",
            )],
            Accepted("repeated key in an untyped object"),
        ),
    ];
    let fixtures = fixtures();
    let failures: Vec<String> = cases
        .into_iter()
        .filter_map(|(name, case, expect)| {
            let fx = fixtures.iter().find(|fx| fx.name == name)?;
            let mut tally = Tally::new();
            let admitted = admitted(fx, &case);
            let failure = admitted
                .then(|| unexplained(fx, &case, &mut tally))
                .flatten();
            let ok = match &expect {
                RoundTrips => admitted && failure.is_none() && tally.is_empty(),
                NotAdmitted => !admitted,
                Accepted(reason) => {
                    admitted && failure.is_none() && tally.contains_key(reason)
                }
            };
            (!ok).then(|| {
                format!(
                    "{name}: want {expect:?}, admitted={admitted} \
                     accepted={tally:?} failure={:?}: {:?}",
                    failure.map(|(_, f)| f),
                    case.emission
                )
            })
        })
        .collect();
    assert_eq!(fixtures.len(), 7);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
