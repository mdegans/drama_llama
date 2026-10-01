//! Integration test for `Prompt::output_config` → Session structured
//! output.
//!
//! Loads a real model (cogito-family recommended) and exercises the
//! full path: optional `<think>...</think>` prefix, nested `$ref`
//! structs, `anyOf` enum, arrays, boolean fields, and round-trip via
//! misanthropic's typed `Message::json()`.
//!
//! All tests here are `#[ignore = "requires model"]`. Run with:
//! `cargo test --test output_config --features json-schema -- --ignored`.
//!
//! The scenario is a tight micro-whodunit: three suspects, five
//! evidence lines, exactly one of the three is the answer. The model
//! reasons inside the thought block and commits to the structured
//! verdict. Thinking is where the reasoning actually lives — the JSON
//! only holds the conclusion — so this is a real stress test of the
//! thought-prefix behavior rather than a vanity demo.
#![cfg(feature = "json-schema")]

use std::{
    num::{NonZeroU128, NonZeroU32},
    path::PathBuf,
};

use drama_llama::{
    Block, Content, Prompt, RenderOptions, Role, SamplingMode, Session,
};
use misanthropic::prompt::message::CacheControl;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf")
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
#[allow(dead_code)] // fields appear in schema + Debug output, not all asserted on
struct Suspect {
    name: String,
    motive: String,
    had_opportunity: bool,
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug, PartialEq)]
enum Confidence {
    /// Evidence is thin; a jury would not convict.
    Low,
    /// The case is plausible but not airtight.
    Medium,
    /// The evidence conclusively identifies the culprit.
    High,
}

#[derive(schemars::JsonSchema, serde::Deserialize, Debug)]
#[allow(dead_code)]
struct CaseFile {
    /// Every suspect considered — one entry per named suspect in the
    /// scenario, with their motive and whether they had physical
    /// access to the scene. Never empty: even exonerated suspects
    /// were considered and belong here. (`length(min = 1)` survives
    /// the Anthropic schema sanitizer and compiles into the grammar,
    /// so the empty-array exit is closed at the sampling level.)
    #[schemars(length(min = 1))]
    suspects_considered: Vec<Suspect>,
    /// The evidence items the detective weighed, in the order
    /// considered. At least one — a verdict without evidence is not a
    /// verdict.
    #[schemars(length(min = 1))]
    key_evidence: Vec<String>,
    /// The suspect name (must match one of `suspects_considered`).
    culprit: String,
    /// How certain the detective is of the verdict.
    confidence: Confidence,
    /// One-sentence summary of the deductive chain.
    reasoning_summary: String,
}

const SCENARIO: &str = "\
Scenario: Sir Harold was found dead in his study at 11 PM, poisoned.

Suspects and verified facts:
- BUTLER (Mr. Finch): disliked Sir Harold. Served the nightcap at 9 PM, \
  but as a precaution (Sir Harold was paranoid) he took a sip from the \
  same glass in front of the house physician. He is alive and unharmed, \
  so the glass was not yet poisoned when it left his hands.
- NIECE (Lady Elsie): stood to inherit if Sir Harold died. She attended \
  the village charity gala from 8 PM to midnight; twenty named guests \
  place her there continuously. She cannot have been at the mansion.
- BUSINESS PARTNER (Mr. Crane): Sir Harold's ledger, found open on the \
  desk, showed Mr. Crane had been embezzling for two years and Sir \
  Harold intended to report him in the morning. Mr. Crane has a copy of \
  the study key (Sir Harold gave him one years ago). Two staff saw him \
  alone in the study from 10:30 to 10:50 PM. The poison is one Mr. \
  Crane keeps for his prize rose bushes.

All three had a motive. Only one had both opportunity (access to the \
glass after the butler's safe sip) AND means (possession of the \
specific poison used). Identify that suspect.";

#[test]
#[ignore = "requires model"]
fn whodunit_verdict() {
    // Default n_ctx (512) truncates long before the thought block
    // finishes. Bump to 8192 so scenario + thinking + verdict fit.
    let mut session = Session::from_path_with_n_ctx(model_path(), 8192)
        .expect("session load")
        .quiet()
        // Seeded deliberately. The assertions below are about the
        // *shape* of the structured output — thought block present,
        // `$ref` array populated, typed round-trip — but they are
        // read off stochastic generation, so unseeded they measure
        // the model's mood as much as the code. Seeding makes a
        // failure mean "the structured-output path changed", which is
        // the only thing this test can honestly report on.
        //
        // A seed is safe here in a way it would not be in the cache
        // suites: those rely on unseeded runs to exercise the
        // sampler-state resume path (a forced seed forks every call
        // and discards the KV-paired snapshot), whereas this is a
        // single completion with no reuse.
        .with_seed(NonZeroU128::new(42))
        .with_render_opts(
            RenderOptions::default().with_extra("enable_thinking", true),
        );

    let prompt = Prompt::default()
        .max_tokens(NonZeroU32::new(4096).unwrap())
        .structured_output::<CaseFile>()
        .system(
            "You are a brief, decisive detective. Think before \
             answering, in under 300 tokens: note which suspects are \
             ruled out by their alibis, identify the one remaining with \
             motive, means, and opportunity. Then output the structured \
             verdict as JSON matching the given schema, listing ALL \
             named suspects in `suspects_considered`.",
        )
        .add_message((Role::User, Content::text(SCENARIO)))
        .expect("add_message");

    let response = session
        .complete_response(&prompt)
        .expect("complete_response");

    // (1) Response is multipart and contains at least one non-empty
    // Thought block — the model reasoned before committing.
    let blocks = &response.inner.content.0;
    let has_thought = blocks.iter().any(|b| {
        matches!(b, Block::Thought { thought, .. } if !thought.trim().is_empty())
    });
    assert!(
        has_thought,
        "expected at least one non-empty Block::Thought, got: {blocks:#?}"
    );

    // (2) Typed round-trip: the final Block::Text is valid JSON that
    // deserializes into the CaseFile schema via misanthropic's
    // Message::json() helper.
    let verdict: CaseFile = response.json().unwrap_or_else(|e| {
        panic!(
            "structured output should deserialize into CaseFile: {e:?}; \
             blocks: {blocks:#?}"
        )
    });

    println!("=== verdict ===\n{verdict:#?}\n===");

    // (3) The array of suspects deserialized through the $ref →
    // Suspect path, and is non-empty.
    //
    // Non-emptiness is the strongest claim this assertion can
    // honestly make: `length(min = 1)` is enforced in the grammar,
    // but counts beyond 1 are deliberately NOT — misanthropic's
    // sanitizer strips `minItems >= 2` (Anthropic only enforces
    // non-emptiness server-side) and `schema_to_gbnf` ignores larger
    // counts on purpose, because forcing N items makes the model
    // manufacture filler entries. See
    // `.claude/memory/schema_constraint_keywords_decision.md` — whose
    // trigger was *this test*, answering `suspects_considered: []`.
    //
    // So asserting ≥2 here, as this used to, claimed more than the
    // grammar guarantees and left the difference resting on the
    // model's mood; it duly broke when the sidecar's default chain
    // got hotter. Multi-element `$ref`-array coverage lives where it
    // can be deterministic: `compiles_ref_array_accepts_populated` in
    // `src/grammar_compile.rs` accepts a two-element array with no
    // model in the loop.
    assert!(
        !verdict.suspects_considered.is_empty(),
        "suspects_considered must be non-empty (grammar enforces \
         minItems=1): {:#?}",
        verdict.suspects_considered,
    );

    // (4) Culprit matches one of the considered suspects.
    let names: Vec<&str> = verdict
        .suspects_considered
        .iter()
        .map(|s| s.name.as_str())
        .collect();
    let matched = names.iter().any(|n| {
        n.eq_ignore_ascii_case(&verdict.culprit)
            || n.contains(&verdict.culprit)
            || verdict.culprit.contains(n)
    });
    assert!(
        matched,
        "culprit {:?} not present in suspects_considered {:?}",
        verdict.culprit, names,
    );

    // (5) Evidence array is populated.
    assert!(
        !verdict.key_evidence.is_empty(),
        "expected non-empty key_evidence array"
    );

    // (6) Confidence is a valid enum variant (tests anyOf
    // alternation compiled from schemars-emitted enum-with-docs).
    assert!(
        matches!(
            verdict.confidence,
            Confidence::Low | Confidence::Medium | Confidence::High
        ),
        "confidence deserialized to an unexpected variant: {:?}",
        verdict.confidence,
    );
}

/// A structured-output answer must survive being replayed as history.
///
/// This is the claim that closes #88 phase 5a. A structured-output
/// answer parses to a [`Block::Text`], and `Block::Text` renders
/// **verbatim** (`chat_template.rs`: `out.push_str(text)`), so nothing
/// re-serializes the JSON and the emission re-renders byte-identically
/// whatever whitespace spelling the model chose. That is *why* the
/// permissive `JSON_GRAMMAR` is correct here, and why moving structured
/// output to the canonical prelude would repeat the
/// `grammar_for_tool_choice` regression — pinning bytes where no
/// re-render contract exists, which we measured as degraded generation.
///
/// Round 2's `cache_read` exceeding round 1's entire prompt means reuse
/// captured the assistant turn — the structured answer round-tripped.
///
/// Doubles as the end-to-end witness for the phase-5b tip fix on a
/// NON-tool-call path: a structured-output turn ends on
/// **grammar-complete** (the JSON completes the schema), which is
/// exactly the ending whose tip prediction used to drop its last
/// content token.
///
/// # Bytes round-trip; token ids do not
///
/// The thing this test cost a session to learn. Measured with the
/// assistant turn UNMARKED: the tip was created at the right position
/// and its hash was present (so the emission genuinely re-rendered
/// byte-stable — the claim above holds), and reuse was still **zero**,
/// because the LCP walk died 2 entries into a 254-token emission.
///
/// Grammar-constrained generation is emitted in a NON-canonical BPE
/// segmentation: the grammar masks a longer merged token whenever it
/// would overshoot the allowed next characters, so the model's token
/// sequence is not the one the tokenizer produces from the same bytes.
/// `prev_entries` holds the emitted ids, `new_entries` holds the
/// re-tokenized render, and `compute_l_hit` compares ids.
///
/// Since "trust the emission" (`PrefixCacheConfig::adopt_emitted_tokens`,
/// on by default) the next call reads round 1's turn in the slot's own
/// ids wherever they spell the same bytes as the render, so the walk
/// no longer dies in the JSON: round 2 reuses round 1's whole prompt
/// and its emission. That reuse is only sound if the KV it keeps holds
/// exactly the ids round 2 then feeds, so the test checks it two ways:
/// the cache tripwire is armed (a spliced prompt that does not read as
/// its render panics), and round 2 is re-run *cold* — the KV wiped
/// under the session, so the restore fails and the same ids are
/// prefilled from position 0 — and must produce the warm output under
/// greedy sampling.
///
/// Greedy rather than a seed — a forced seed forks every call and
/// discards the KV-paired snapshot the resume path needs (see
/// `whodunit_verdict` above for the contrast).
#[test]
#[ignore = "requires model"]
fn structured_output_round_trips_as_history() {
    // OnceLock reads this on first use; nextest's process-per-test
    // isolation keeps it scoped to this test.
    std::env::set_var("DRAMA_LLAMA_CACHE_TRIPWIRE", "1");
    const SYSTEM: &str = "You are a brief, decisive detective. Answer \
                          ONLY with the structured verdict as JSON \
                          matching the given schema. Do not explain.";

    let mut session = Session::from_path_with_n_ctx(model_path(), 8192)
        .expect("session load")
        .quiet()
        .without_repetition()
        .with_sampling([SamplingMode::Greedy])
        // Thinking off so the emission is pure JSON. The thought path
        // has its own normalization (`parse_thought` trims, the
        // renderer re-emits bare markers) which is shared with every
        // prose turn and is NOT what this test is about.
        .with_render_opts(
            RenderOptions::default()
                .with_generation_prompt(true)
                .with_extra("enable_thinking", false)
                .with_extra("preserve_thinking", true),
        )
        .with_prefix_cache(true);

    let round1 = Prompt::default()
        .max_tokens(NonZeroU32::new(1024).unwrap())
        .structured_output::<CaseFile>()
        .system(SYSTEM)
        .add_message((Role::User, Content::text(SCENARIO)))
        .expect("add_message");

    let r1 = session.complete_response(&round1).expect("round 1");
    // `round1` carries no `cache_control` marker anywhere, so
    // `breakpoint_cells` is 0 and the whole prompt lands in
    // `input_tokens` (the three-way split — read / creation / input —
    // only pulls cells out of `input_tokens` when a breakpoint exists
    // to pull them up to). That's what makes `r1_input` usable below
    // as "round 1's whole prompt" rather than just its post-breakpoint
    // tail.
    let r1_input = r1.usage.input_tokens;
    let r1_output = r1.usage.output_tokens;
    eprintln!(
        "round 1: input_tokens={}, output_tokens={}",
        r1_input, r1_output
    );
    let _: CaseFile = r1.json().expect("round 1 deserializes");

    // Round 2 replays round 1's content VERBATIM — substituting a
    // re-serialized value would test our serializer, not the
    // round-trip.
    //
    // The assistant turn's last block carries `cache_control`, and
    // that is load-bearing rather than decorative: it puts a
    // *hash-keyed* breakpoint at the tip position. Grammar-constrained
    // output is emitted in a NON-canonical BPE segmentation (the
    // grammar masks a longer merged token whenever it would overshoot
    // the allowed next characters), so the re-render's tokenization
    // differs from the emitted token ids even though the bytes are
    // identical — measured: LCP died 2 tokens into an unmarked
    // 254-token JSON emission. The LCP walk compares token ids and
    // cannot survive that; the hash path compares renders and can.
    let mut assistant = r1.inner.content.clone();
    if let Some(Block::Text { cache_control, .. }) = assistant.last_mut() {
        *cache_control = Some(CacheControl::ephemeral());
    }

    let round2 = Prompt::default()
        .max_tokens(NonZeroU32::new(1024).unwrap())
        .structured_output::<CaseFile>()
        .system(SYSTEM)
        .add_message((Role::User, Content::text(SCENARIO)))
        .expect("add_message")
        .add_message((Role::Assistant, assistant))
        .expect("add_message")
        .add_message((
            Role::User,
            Content::text(
                "Re-check that verdict against the alibis and answer \
                 again.",
            ),
        ))
        .expect("add_message");

    let r2 = session.complete_response(&round2).expect("round 2");
    let r2_read = r2.usage.cache_read_input_tokens.unwrap_or(0);
    eprintln!(
        "round 2: input_tokens={}, cache_read={}",
        r2.usage.input_tokens, r2_read
    );

    // The test's name still holds — the BYTES round-trip, which is
    // what `r2.json()` below proves. What does not round-trip is the
    // *segmentation*: the grammar masks merged tokens, so the
    // tokenizer re-reads the emission in a different split.
    //
    // History of this bound. It was `r2_read > r1_input` when the test
    // landed (#88 phase 5b), green *on corruption*: the tip's render
    // hash matched, so reuse proceeded at the tip's cached entry index
    // while the same bytes ended three entries earlier in the new
    // render, and the first tokens of the new user turn were never
    // decoded (#91, found here). The #91 fix refused that hit, and with
    // the LCP walk dying two tokens into the JSON this pinned zero
    // reuse. Adoption resolves the hit the other way: the prompt is
    // read in the slot's own ids up to the edit, so the ids before the
    // tip are equal and the reuse is real again.
    //
    // So: everything round 1 left in the KV is reused — its prompt and
    // its emission. The tip is the emission's last token, which was
    // sampled but never decoded, so the KV ends one short of it. More
    // than that would reach into the new user turn round 1 never saw.
    assert_eq!(
        r2_read,
        r1_input + r1_output - 1,
        "round 2 should reuse round 1's prompt ({r1_input}) and every \
         decoded token of its {r1_output}-token emission"
    );
    assert!(
        r2.usage.input_tokens > 0,
        "the new user turn must be prefilled, not reused"
    );

    // Cache stats are not a proxy for output (the 2026-07-24
    // stale-matcher-carry regression read green on reuse while
    // emitting nothing).
    assert!(
        r2.usage.output_tokens > 0,
        "round 2 produced no tokens (reused {r2_read})"
    );
    let _: CaseFile = r2.json().expect("round 2 deserializes");

    // Warm == cold. Wipe the KV under the session: its slot still
    // records round 2's prompt, so the call reads the same ids, the
    // restore fails (no KV, no checkpoint), and the whole prompt is
    // prefilled from position 0. Greedy output must match the warm
    // call's — the reused KV was the KV of the ids it claimed.
    session.engine_mut().memory_clear();
    let cold = session.complete_response(&round2).expect("round 2, cold");
    assert_eq!(
        cold.usage.cache_read_input_tokens.unwrap_or(0),
        0,
        "the wiped KV must not be reused"
    );
    // Read + written + fresh is the whole prompt, in the ids it was
    // read in: the same total both ways.
    let prompt_len = |u: &misanthropic::response::TokenCounts| {
        u.input_tokens
            + u.cache_read_input_tokens.unwrap_or(0)
            + u.cache_creation_input_tokens.unwrap_or(0)
    };
    assert_eq!(
        prompt_len(&cold.usage),
        prompt_len(&r2.usage),
        "the cold call must prefill the same prompt the warm one read"
    );
    assert_eq!(
        cold.inner.content.to_string(),
        r2.inner.content.to_string(),
        "warm round 2 (reused {r2_read}) diverged from the same ids \
         prefilled cold"
    );
}
