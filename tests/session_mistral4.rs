//! Mistral Small 4 e2e (#88): the `TagWithJson` dialect —
//! `[TOOL_CALLS]name[ARGS]{…}` calls, `[THINK]` reasoning — against
//! the real model. Mirrors the Gemma 4 (`session_gemma4.rs`) and
//! gpt-oss (`session_gptoss.rs`) suites: forced/auto calls, thinking,
//! round-trip byte-stability, prefix-cache survival.
//!
//! **What is deliberately different here: this suite installs no
//! template sidecar.** Both sibling suites copy their
//! `*-cache-stable.jinja` next to the model and therefore exercise
//! **rung 1** of the loading ladder (explicit sidecar). That left the
//! gap noted in `.claude/memory/plan_template_ownership.md` — *no test
//! observes rung 2 end-to-end on a real model*. Rung 2 is the whole
//! point of `src/baked.rs`: a model whose embedded template is
//! byte-equal to a stock we recognize silently receives our owned
//! replacement, with nothing on disk to say so. Here the only thing
//! standing between the GGUF's own template and a cache-stable session
//! is `baked::detect` + `baked::MISTRAL4`, and
//! [`rung_two_baked_template_applies_without_a_sidecar`] is the
//! witness. Every other test in this file rides on that: if rung 2
//! silently stopped firing, the stock template would take over and
//! *most* of these would still pass — which is exactly why the witness
//! asserts the rung explicitly rather than leaving it implied.
//!
//! The rung-2 tell is cheap and unambiguous: the stock template
//! accepts a thought only as a `thinking`-typed content chunk, never
//! as a message field, so the analyzer measures `ReasoningMode::None`
//! on it (pinned FFI-free in `dialect_analyzer.rs::mistral4_stock`).
//! Only the baked replacement exposes `[THINK]`/`[/THINK]` as a
//! `TagBased` channel with `ReasoningReingest::Field`. A session whose
//! dialect carries those markers can only have gotten them from
//! `baked::MISTRAL4`.
//!
//! All tests load `models/Mistral-Small-4-119B-2603-UD-IQ3_S.gguf`
//! (override with `$DRAMA_LLAMA_MISTRAL_MODEL`) and are `#[ignore]`d.
//! Absent that model they **skip loudly rather than substituting**
//! whatever `models/model.gguf` happens to point at — the banned
//! pattern from `hash_cache_smoke.rs`. Run with
//! `cargo nextest run --features llama-cpp --test session_mistral4 \
//!  --run-ignored all`.

// Double-gated: Cargo.toml carries `required-features = ["llama-cpp"]`
// (so featureless configurations skip the build entirely, per the
// test-topology preference) and the file also gates itself — harmless
// belt-and-suspenders left from before the Cargo.toml entry existed.
#![cfg(feature = "llama-cpp")]

mod common;

use std::{borrow::Cow, num::NonZeroU32, path::PathBuf};

use drama_llama::{
    dialect::{Family, ReasoningMode, ReasoningReingest},
    prompt::{ToolResult, ToolUse},
    Block, Content, FromPath, JsonSpacing, Message, Prompt, RenderOptions,
    Role, Tool, ToolChoice,
};
use serde_json::json;

/// Resolve the Mistral Small 4 GGUF: `$DRAMA_LLAMA_MISTRAL_MODEL` if
/// set and present, else the conventional path under `models/`. `None`
/// means skip — never substitute `model.gguf`. Same helper shape as
/// `probe_unforced_habit.rs`, and the same reason.
fn model_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("DRAMA_LLAMA_MISTRAL_MODEL") {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let conventional = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/Mistral-Small-4-119B-2603-UD-IQ3_S.gguf");
    conventional.exists().then_some(conventional)
}

/// Load a session, or `None` to skip. **No sidecar is installed** —
/// see the module doc: this suite is the rung-2 witness, and a
/// `*.template.jinja` next to the model would silently promote it to
/// rung 1 and void every claim in here. If one exists (someone ran an
/// experiment, or copied the gemma4 helper), that is not a thing to
/// paper over: fail loudly and let the human delete it, because a
/// green run under a stray sidecar is the failure this whole file
/// exists to prevent.
fn load_session() -> Option<drama_llama::LlamaCppSession> {
    let path = model_path()?;
    let sidecar = path.with_extension("template.jinja");
    assert!(
        !sidecar.exists(),
        "a template sidecar exists at {}, which would make this suite \
         exercise rung 1 and silently void its rung-2 claim. Delete it \
         and re-run.",
        sidecar.display()
    );
    // `n_ubatch = 31` is load-bearing, not tuning: on Metal this model
    // returns an all-NaN vocabulary for any micro-batch of >=32 tokens
    // (f16 overflow of its layer-32 activations in the half-precision
    // `mul_mm_id` MoE matmul; the `mul_mv_id` path below 32 carries the
    // same values in f32 and is correct). Without this, every test here
    // dies on `DecodeError::NonFinite` before it can assert anything.
    // See `.claude/memory/mistral4_support_and_metal_nan.md`; drop it
    // once upstream fixes the kernel.
    Some(
        drama_llama::LlamaCppSession::from_path_with(
            path,
            drama_llama::LlamaCppOptions::default()
                .with_n_ctx(8192)
                .with_n_ubatch(31),
        )
        .expect("session load")
        .quiet(),
    )
}

/// Shared skip preamble: every test in this file starts with it.
macro_rules! session_or_skip {
    () => {
        match load_session() {
            Some(s) => s,
            None => {
                eprintln!(
                    "SKIP: needs a Mistral Small 4 model \
                     (DRAMA_LLAMA_MISTRAL_MODEL or \
                     models/Mistral-Small-4-119B-2603-UD-IQ3_S.gguf)"
                );
                return;
            }
        }
    };
}

fn count_letters_prompt() -> Prompt {
    let tool = Tool::builder("count_letters")
        .description(
            "Count the number of times a letter appears in a string.",
        )
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "the letter to count"},
                "string": {"type": "string", "description": "the string to search"}
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool");
    Prompt {
        system: Some(Content::text(
            "You are a helpful assistant. You cannot count letters in a \
             word reliably on your own because you see in tokens, not \
             letters. Use the `count_letters` tool when asked to count \
             characters.",
        )),
        messages: vec![Message {
            role: Role::User,
            content: Content::text("Count the number of r's in 'strawberry'"),
        }],
        tools: Some(vec![tool.into()]),
        tool_choice: Some(ToolChoice::method("count_letters")),
        ..Default::default()
    }
}

/// **The rung-2 witness** — the one test in the repo that observes
/// `baked::detect` firing end-to-end on a real model, with nothing on
/// disk helping it.
///
/// Two halves, and both matter:
///
/// 1. **Rung 2 fired.** `reasoning.mode == TagBased` with
///    `[THINK]`/`[/THINK]` and `ReasoningReingest::Field` is
///    producible *only* by `baked::MISTRAL4`'s replacement. The stock
///    template exposes reasoning solely as a `thinking`-typed content
///    chunk, so the analyzer measures `ReasoningMode::None` against it
///    (`dialect_analyzer.rs::mistral4_stock`). `None` here therefore
///    means the ladder fell through to rung 3 (embedded, best-effort)
///    — the template got byte-drifted by a re-quant, or the registry
///    entry rotted. Read `baked::nearest_stock`'s warning in the log:
///    it names the family when the bytes drifted but the dialect
///    didn't, which is the likely diagnosis.
/// 2. **It fired on the right thing.** The call path is asserted too,
///    because a dialect can carry the right reasoning markers and
///    still be wrong everywhere the grammar and parser key on.
///    `[TOOL_CALLS]` / `[ARGS]` / empty close / compact JSON is the
///    whole `TagWithJson` contract for this model, derived by the
///    analyzer with no hand-built `PATCHES` arm for the call shape.
///
/// The FFI-free half of this pin lives in
/// `dialect_analyzer.rs::mistral4_cache_stable`; what this adds is
/// that the *real GGUF's* embedded template is still byte-equal to our
/// detection key, which no model-free test can know.
#[test]
#[ignore = "long running - requires Mistral Small 4 model"]
fn rung_two_baked_template_applies_without_a_sidecar() {
    let session = session_or_skip!();
    let d = session.dialect();
    println!("=== dialect ===\n{d:#?}\n===");

    // Half 1: only the baked replacement produces these.
    assert_eq!(
        d.reasoning.mode,
        ReasoningMode::TagBased,
        "rung 2 did not fire: the stock template measures \
         ReasoningMode::None, so a non-TagBased reasoning mode means \
         `baked::detect` missed and we fell through to the embedded \
         best-effort tier. {d:#?}"
    );
    assert_eq!(d.reasoning.start, "[THINK]", "{d:#?}");
    assert_eq!(d.reasoning.end, "[/THINK]", "{d:#?}");
    assert_eq!(
        d.reasoning.reingest,
        ReasoningReingest::Field,
        "`[THINK]` must round-trip through the reasoning_content \
         *field* — the one thing the probes cannot see, corrected by a \
         PATCHES entry. {d:#?}"
    );

    // Half 2: the call path the grammar and parser key on.
    assert_eq!(d.family, Family::TagWithJson, "{d:#?}");
    assert_eq!(d.per_call_start, "[TOOL_CALLS]", "{d:#?}");
    assert_eq!(d.function.name_suffix, "[ARGS]", "{d:#?}");
    assert_eq!(d.function.close, "", "{d:#?}");
    assert_eq!(d.call_separator, "", "{d:#?}");
    // `Spaced` is itself a rung-2 witness: only the owned template
    // renders arguments through `json_dumps`. Stock's `tojson` measures
    // `Compact`, so this asserting Spaced means the baked replacement
    // is the one in force.
    assert_eq!(d.arguments.json_spacing, JsonSpacing::Spaced, "{d:#?}");
    assert_eq!(d.user_start, "[INST]", "{d:#?}");
    assert_eq!(d.tool_response_start, "", "{d:#?}");
    assert_eq!(d.trigger(), "[TOOL_CALLS]", "{d:#?}");

    // The framing tokens the emit-ban and the KV cache depend on must
    // survive as preserved specials; losing either turns a call into
    // prose that happens to look like one.
    for tok in ["[THINK]", "[TOOL_CALLS]"] {
        assert!(
            d.preserved_tokens.iter().any(|t| t == tok),
            "{tok} must be a preserved special: {d:#?}"
        );
    }
}

/// Method-forced call: the grammar forces the tag envelope
/// (`[TOOL_CALLS]count_letters[ARGS]{…}`) and the parser returns a
/// typed `ToolUse`. Note there is no per-call *close* in this dialect
/// — the JSON object's `}` is the whole terminator — so a parser that
/// expects one would either hang on the tail or swallow the following
/// prose. Typed arguments are asserted, not just presence: a
/// `TagWithJson` parse that fell back to treating the args as an
/// opaque string would still produce a `ToolUse`.
#[test]
#[ignore = "long running - requires Mistral Small 4 model"]
fn forced_call_parses_to_tool_use() {
    let prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(512).unwrap());
    let mut session = session_or_skip!();

    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== forced blocks ===\n{blocks:#?}\n===");

    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .expect("Method tool_choice must produce a ToolUse block");
    assert_eq!(call.name, "count_letters");
    assert!(
        call.input.get("letter").and_then(|v| v.as_str()).is_some(),
        "letter arg missing/untyped: {:?}",
        call.input
    );
    assert!(
        call.input.get("string").and_then(|v| v.as_str()).is_some(),
        "string arg missing/untyped: {:?}",
        call.input
    );
    // Framing must never survive into prose. `[TOOL_CALLS]` and
    // `[ARGS]` are single special tokens in this vocab, so seeing
    // either inside a Text block means the dialect split failed, not
    // that the model wrote the words.
    assert!(
        !blocks.iter().any(|b| matches!(
            b,
            Block::Text { text, .. } if text.contains("[TOOL_CALLS]")
                || text.contains("[ARGS]")
        )),
        "call framing leaked into a Text block: {blocks:#?}"
    );
}

/// Auto tool choice: the lazy grammar arms on the `[TOOL_CALLS]`
/// trigger only, so everything before it is free generation. This is
/// the path that proves the trigger is the right one — a wrong or
/// over-eager trigger shows up as either no call at all (never armed)
/// or mangled prose (armed on a substring the model writes naturally).
#[test]
#[ignore = "long running - requires Mistral Small 4 model"]
fn auto_tool_choice_parses_native_call() {
    let mut prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(1024).unwrap());
    prompt.tool_choice = Some(ToolChoice::auto());
    let mut session = session_or_skip!();

    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== auto blocks ===\n{blocks:#?}\n===");

    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call),
            _ => None,
        })
        .expect("unforced path must still parse the native tag call");
    assert_eq!(call.name, "count_letters");
    assert!(
        call.input.get("letter").is_some()
            && call.input.get("string").is_some(),
        "arguments must coerce to typed JSON, got: {:?}",
        call.input
    );
}

/// Thinking on a plain prose turn (no tools, no grammar). Two
/// properties, and the second is the load-bearing one:
///
/// 1. The `[THINK]` channel is reachable at all. `enable_thinking` is
///    derived from `prompt.thinking` and lands in the template's
///    `[MODEL_SETTINGS]{"reasoning_effort": …}` block — which sits in
///    the prompt *prefix*, so this also incidentally exercises the
///    render path that a thinking toggle invalidates the cache
///    through. A `Thought` block coming back means dialect → template
///    → model → parser all agreed on the channel.
/// 2. `[THINK]` / `[/THINK]` must never appear inside a `Block::Text`.
///    That is not cosmetic: these are single special tokens in the
///    vocab, and a marker surviving into content means the parser
///    failed to split the channel. The next turn would then re-ingest
///    reasoning as prose (the `Field` reingest puts thoughts in
///    `reasoning_content`, prose in `content`), the re-render would
///    differ from the emission, and the prefix cache would die
///    silently — a wrong answer that looks like a slow one.
///
/// Deliberately toolless: `forced_call_parses_to_tool_use` covers the
/// grammar path, and mixing the two would make a failure ambiguous
/// between "thinking broke" and "the grammar prefix rejected it".
#[test]
#[ignore = "long running - requires Mistral Small 4 model"]
fn thinking_turn_yields_thought_without_leaking_markers() {
    use misanthropic::prompt::thinking::Thinking;

    let prompt = Prompt {
        system: Some(Content::text("You are a helpful assistant.")),
        messages: vec![Message {
            role: Role::User,
            content: Content::text(
                "Alice has three apples and gives one to Bob. How many \
                 does she have left? Think it through, then answer.",
            ),
        }],
        ..Default::default()
    }
    .thinking(Thinking::Enabled {
        budget_tokens: NonZeroU32::new(512).unwrap(),
        display: None,
    })
    .max_tokens(NonZeroU32::new(1024).unwrap());

    let mut session = session_or_skip!();
    let blocks = session.complete_blocks(&prompt).expect("complete_blocks");
    println!("=== thinking blocks ===\n{blocks:#?}\n===");

    let thought = blocks
        .iter()
        .find_map(|b| match b {
            Block::Thought { thought, .. } => Some(thought),
            _ => None,
        })
        .unwrap_or_else(|| {
            panic!(
                "thinking enabled but no Thought block came back — the \
                 [THINK] channel is unreachable, which is also what a \
                 rung-2 miss looks like (stock measures \
                 ReasoningMode::None). blocks: {blocks:#?}"
            )
        });
    assert!(
        !thought.trim().is_empty(),
        "reasoning parsed as an empty Thought: {blocks:#?}"
    );
    assert!(
        !blocks.iter().any(|b| matches!(
            b,
            Block::Text { text, .. } if text.contains("[THINK]")
                || text.contains("[/THINK]")
        )),
        "think markers leaked into a Text block — the channel split \
         failed and the next re-render will not reproduce these bytes: \
         {blocks:#?}"
    );
    // The turn must still answer. A Thought-only turn means the model
    // ran out of budget inside the channel, which is a real failure
    // for a prose turn even though the parse succeeded.
    assert!(
        blocks.iter().any(|b| matches!(b, Block::Text { .. })),
        "thinking turn produced no prose answer: {blocks:#?}"
    );
}

/// Round-trip byte-stability e2e — the cache-correctness invariant
/// (#30/#88) for the tag dialect: the raw emission must be a byte
/// *prefix* of the canonical template re-render of the parsed blocks.
///
/// This is the property the whole baked template exists for, and it is
/// exactly what stock cannot deliver here: stock closes every
/// assistant message with `</s>` unconditionally and has no
/// `add_generation_prompt` branch at all, so its generation-prompt
/// render is not a prefix of the follow-up render and the
/// `strip_prefix` below would fail outright. That makes this test a
/// second, independent rung-2 witness — one that fails at the byte
/// level rather than the dialect level.
#[test]
#[ignore = "long running - requires Mistral Small 4 model"]
fn emission_round_trips_through_parse_and_render() {
    use drama_llama::AssistantMessage;

    let prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(512).unwrap());
    let mut session = session_or_skip!();
    println!("=== dialect ===\n{:#?}\n===", session.dialect());

    // Mirror the session's own render defaults, including the
    // dialect-driven thought re-ingest convention (`Field` here).
    let render_opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_extra("preserve_thinking", true)
        .with_thought_reingest(session.dialect().reasoning.reingest);
    let rendered_original = session
        .template()
        .render_with(&prompt, &render_opts)
        .expect("render original");

    let raw = session.complete_text(&prompt).expect("complete_text");
    println!("=== raw emission ===\n{raw:?}\n===");

    let tool_refs: Vec<&Tool> = prompt
        .tools
        .iter()
        .flatten()
        .filter_map(|def| def.as_method())
        .collect();
    let blocks = drama_llama::dialect::parse_text(
        session.dialect(),
        &tool_refs,
        &raw,
        // Mistral's generation prompt ends after `[MODEL_SETTINGS]` /
        // the last turn — there is no assistant header and no
        // pre-opened `[THINK]`, so reasoning is never already-open in
        // the prompt tail.
        false,
        drama_llama::dialect::Leniency::Final,
    )
    .blocks;
    assert!(
        blocks.iter().any(|b| matches!(b, Block::ToolUse { .. })),
        "Method tool_choice must parse to a ToolUse block; got {blocks:?}"
    );
    let assistant: AssistantMessage = blocks.into_iter().collect();

    let mut follow_up = prompt.clone();
    follow_up.messages.push(assistant.into());
    follow_up.tool_choice = None;
    let rendered_follow_up = session
        .template()
        .render_with(
            &follow_up,
            &RenderOptions::default()
                .with_generation_prompt(false)
                .with_extra("preserve_thinking", true)
                .with_thought_reingest(session.dialect().reasoning.reingest),
        )
        .expect("render follow_up");

    // STRICT byte-prefix, no LCP fallback: the baked template emits
    // the `</s>` close per *message*, so the generation prompt is
    // simply the render up to that point and the completed turn
    // extends it exactly. Failure here means either the baked template
    // is not in play (rung 2 missed — check the load log) or the
    // re-render lost bytes the model emitted.
    let suffix = rendered_follow_up
        .strip_prefix(&rendered_original)
        .unwrap_or_else(|| {
            panic!(
                "follow-up must extend the original prefix exactly. The \
                 stock Mistral template CANNOT do this (unconditional \
                 </s>, no add_generation_prompt branch), so suspect a \
                 rung-2 miss first.\n\
                 --- original ---\n{rendered_original}\n\
                 --- follow-up ---\n{rendered_follow_up}"
            )
        });
    assert!(
        suffix.starts_with(&raw),
        "emission is not a byte prefix of the canonical re-render.\n\
         --- emission ---\n{raw:?}\n--- re-rendered suffix ---\n{suffix:?}"
    );
}

/// Tool turn 2: assistant call + tool result re-ingest through the
/// `[TOOL_RESULTS]` framing, then free prose. The answer should
/// surface the result — a model that ignores it is usually a sign the
/// result landed in the wrong slot, not that it reasoned badly.
#[test]
#[ignore = "long running - requires Mistral Small 4 model"]
fn tool_result_turn_produces_prose_answer() {
    let call_id = "call_0_count_letters";
    let mut prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(512).unwrap());
    prompt.tool_choice = None;
    prompt.messages.push(Message {
        role: Role::Assistant,
        content: Content(vec![Block::ToolUse {
            call: ToolUse {
                id: Cow::Borrowed(call_id),
                name: Cow::Borrowed("count_letters"),
                input: json!({"letter": "r", "string": "strawberry"}),
                cache_control: None,
                caller: None,
            },
        }]),
    });
    prompt.messages.push(Message {
        role: Role::User,
        content: Content(vec![Block::ToolResult {
            result: ToolResult {
                tool_use_id: Cow::Borrowed(call_id),
                content: Content::text("3"),
                is_error: false,
                cache_control: None,
            },
        }]),
    });

    let mut session = session_or_skip!();
    let out = session.complete_text(&prompt).expect("complete_text");
    println!("=== turn 2 ===\n{out}\n===");
    assert!(!out.trim().is_empty(), "got empty output");
    assert!(
        out.contains('3') || out.to_lowercase().contains("three"),
        "expected the count (3) in the answer, got: {out:?}"
    );
}

/// Prefix cache across a tool turn — the accounting proof that the
/// byte-stability above actually buys reuse at the KV level.
///
/// Turn 1 is grammar-constrained, and per
/// `.claude/memory/plan_template_ownership.md` a grammar-constrained
/// turn only reuses via the hash path with a breakpoint at the tip —
/// hence the explicit `cache_control` on the tool definition, which
/// puts a breakpoint at the end of the static tools prefix (this
/// template renders `[AVAILABLE_TOOLS]` ahead of every message, so
/// that breakpoint covers the system prompt, the tool schemas and
/// `[MODEL_SETTINGS]`). Mirrors `session_gemma4.rs` exactly; nonzero
/// `cache_read` on turn 2 is the assertion.
///
/// Zero here means emission/re-render drift somewhere in the tool
/// turn: the turn-1 bytes we recorded are not the bytes turn 2's
/// prompt renders to, so the LCP dies before the breakpoint and the
/// whole prefix re-prefills. On a 119B IQ3_S that is minutes, not
/// milliseconds — this test is cheap insurance on an expensive
/// failure.
#[test]
#[ignore = "long running - requires Mistral Small 4 model"]
fn prefix_cache_survives_tool_turn() {
    use misanthropic::prompt::message::AssistantMessage;

    let mut prompt =
        count_letters_prompt().max_tokens(NonZeroU32::new(1024).unwrap());
    if let Some(tools) = prompt.tools.as_mut() {
        if let Some(def) = tools.first_mut() {
            if let Some(tool) = def.as_method_mut() {
                tool.cache_control = Some(
                    misanthropic::prompt::message::CacheControl::ephemeral(),
                );
            }
        }
    }
    let mut session = session_or_skip!().with_prefix_cache(true);

    let blocks = session.complete_blocks(&prompt).expect("turn 1");
    let call = blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call.clone()),
            _ => None,
        })
        .expect("turn 1 must produce a ToolUse block");

    let assistant: AssistantMessage = blocks.iter().cloned().collect();
    prompt.messages.push(assistant.into());
    prompt.messages.push(Message {
        role: Role::User,
        content: Content(vec![Block::ToolResult {
            result: ToolResult {
                tool_use_id: call.id.clone(),
                content: Content::text("3"),
                is_error: false,
                cache_control: None,
            },
        }]),
    });
    prompt.tool_choice = None;

    let out = session.complete_text(&prompt).expect("turn 2");
    println!("=== turn 2 ===\n{out}\n===");
    let read = session
        .last_usage()
        .cache_read_input_tokens
        .unwrap_or_default();
    assert!(
        read > 0,
        "turn 2 reused no prefix — emission/re-render drift broke the \
         cache across the tool turn (usage: {:?})",
        session.last_usage()
    );
}

/// #96, the downstream (agentkit) shape against the `TagWithJson`
/// dialect: sliding markers, forced tool-call turns, every
/// continuation resuming past the entire previous prompt via the tip.
/// This is the exact scenario the issue measured missing on this
/// model (`l_hit` pinned at the last marker as `new_len` grew).
#[test]
#[ignore = "long running; needs the Mistral Small 4 model"]
fn tip_anchors_across_tool_rounds_issue_96() {
    let session = session_or_skip!();
    common::tip::assert_tip_anchors_across_tool_rounds(
        session.with_prefix_cache(true),
        3,
    );
}

/// #96's probe scenario on Mistral Small 4: a continuation adding no
/// new `cache_control` anywhere may only be covered by the tip via
/// the LCP walk.
#[test]
#[ignore = "long running; needs the Mistral Small 4 model"]
fn tip_anchors_unmarked_continuation_issue_96() {
    let session = session_or_skip!();
    common::tip::assert_tip_anchors_unmarked_continuation(
        session.with_prefix_cache(true),
    );
}
