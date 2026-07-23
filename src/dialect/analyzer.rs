//! Differential template analyzer: derive a [`CallSyntax`] from a
//! chat template by rendering sentinel payloads and diffing.
//!
//! Rust port of llama.cpp's `common/chat-diff-analyzer.cpp`
//! (b9754 / 52b3df00, MIT), with the tagged-PEG extraction replaced
//! by direct segment scans ([`super::segment`]) — we deliberately do
//! not port the PEG arena (see plan). The analyzer core is FFI-free:
//! it needs only the template source and the BOS/EOS piece strings,
//! so it is testable against `.jinja` fixtures alone.
//!
//! The method: render pairs of variant payloads that differ in
//! exactly one dimension (reasoning present/absent, one call vs two,
//! `FFF_FIRST_FUN_F` vs `SSS_SECOND_FUN_S`, …), take the
//! marker-aware diff, and read the markers off the boundary. Family
//! classification (`JSON_NATIVE` / `TAG_WITH_JSON` /
//! `TAG_WITH_TAGGED`) falls out of where the sentinels appear
//! quoted. Misdetections on unseen finetunes are expected — the
//! sidecar override and the patches-as-data list are the escape
//! hatches, and load must never hard-fail on analysis (Session falls
//! back to content-only; deliberate divergence from upstream's hard
//! error).

use minijinja::{Environment, UndefinedBehavior};
use serde_json::{json, Value};

use super::segment::{
    after_common_suffix, calculate_diff_split, find_first_marker,
    find_last_marker, marker_after, marker_before, prune_whitespace_segments,
    segmentize_markers, until_common_prefix, DiffSplit, Segment,
};
use super::{
    CallIdPosition, CallSyntax, ContentMode, ContentSyntax, Family, JsonFields,
    ReasoningMode, ReasoningSyntax,
};

// Sentinel payloads, verbatim from upstream — chosen to be
// unambiguous in rendered output and free of marker characters.
const FUN_FIRST: &str = "FFF_FIRST_FUN_F";
const FUN_SECOND: &str = "SSS_SECOND_FUN_S";
const ARG_FIRST: &str = "AA_ARG_FST_AA";
const ARG_SECOND: &str = "BB_ARG_SND_BB";
const USER_MSG: &str = "U_USER_MSG Hello END_U";
const ASSISTANT_MSG: &str = "A_ASST_MSG I can help END_A";
const THINKING_CONTENT: &str = "REASON_PART I am thinking END_R";
const CALL_ID_001: &str = "call00001";
const CALL_ID_002: &str = "call00002";
const CALL_ID_999: &str = "call99999";

#[derive(Debug, thiserror::Error)]
pub enum AnalyzeError {
    #[error("template failed to compile: {0}")]
    Template(#[from] minijinja::Error),
}

/// Optional post-analysis pass over a loaded model's vocabulary: the
/// outer structural markers of a detected dialect (the call trigger
/// and the reasoning tags) are trained as single special tokens in
/// every family we know — a marker that tokenizes to several pieces
/// is a strong misdetection signal for a finetune's template.
///
/// Returns the suspect markers (empty = all high-confidence). Callers
/// decide severity; [`Session::from_engine`](crate::Session) logs
/// them at debug level and continues — analysis never gates a load.
/// Inner markers (`<function=`, `<parameter=`) are deliberately not
/// checked: tagged dialects compose those from ordinary text tokens.
pub fn vocab_cross_check<M: crate::backend::Model + ?Sized>(
    syntax: &super::CallSyntax,
    model: &M,
) -> Vec<String> {
    let mut suspects = Vec::new();
    let candidates = [
        syntax.trigger(),
        syntax.reasoning.start.as_str(),
        syntax.reasoning.end.as_str(),
    ];
    for marker in candidates {
        let core = marker.trim();
        if core.is_empty() {
            continue;
        }
        if model.tokenize(core, true).len() != 1
            && !suspects.iter().any(|s| s == core)
        {
            suspects.push(core.to_string());
        }
    }
    suspects
}

/// A template environment prepared for sentinel probing.
struct Probe<'s> {
    env: Environment<'s>,
    bos: String,
    eos: String,
}

/// Render parameters for one probe variant (upstream
/// `template_params`).
#[derive(Clone)]
struct Params {
    messages: Value,
    tools: Value,
    add_generation_prompt: bool,
    enable_thinking: bool,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            messages: json!([]),
            tools: Value::Null,
            add_generation_prompt: false,
            enable_thinking: true,
        }
    }
}

struct Comparison {
    diff: DiffSplit,
    output_a: String,
    output_b: String,
}

impl<'s> Probe<'s> {
    fn new(source: &str, bos: &str, eos: &str) -> Result<Self, AnalyzeError> {
        let mut env = Environment::new();
        env.set_unknown_method_callback(
            minijinja_contrib::pycompat::unknown_method_callback,
        );
        // Chainable so partially-supported payload shapes degrade to
        // empty renders (which the probes treat as "skip") instead of
        // erroring the whole analysis.
        env.set_undefined_behavior(UndefinedBehavior::Chainable);
        env.add_function(
            "raise_exception",
            |msg: String| -> Result<String, minijinja::Error> {
                Err(minijinja::Error::new(
                    minijinja::ErrorKind::InvalidOperation,
                    msg,
                ))
            },
        );
        env.add_function("strftime_now", |_fmt: String| -> String {
            // Deterministic: probing must not depend on wall clock.
            "01 Jan 2026".to_string()
        });
        env.add_template_owned("probe", source.to_string())?;
        Ok(Self {
            env,
            bos: bos.to_string(),
            eos: eos.to_string(),
        })
    }

    /// Render one probe variant. `None` on any render error —
    /// upstream treats failures as "skip this probe".
    fn apply(&self, params: &Params) -> Option<String> {
        let ctx = minijinja::context! {
            bos_token => &self.bos,
            eos_token => &self.eos,
            messages => &params.messages,
            tools => &params.tools,
            add_generation_prompt => params.add_generation_prompt,
            enable_thinking => params.enable_thinking,
            date_string => "01 Jan 2026",
        };
        // `catch_unwind` in addition to `.ok()`: some templates index
        // `messages[0]` unconditionally and minijinja *panics* (not
        // errors) on the empty-messages probe (observed: Qwen3 chat).
        // Rendering is a pure function of its inputs, so unwinding is
        // state-safe; the cost is one panic-hook line on stderr for
        // quirky templates. Upstream's try/catch equivalent.
        let tmpl = self.env.get_template("probe").ok()?;
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            tmpl.render(ctx).ok()
        }))
        .ok()
        .flatten()
    }

    /// Render A and B (B = `mutate` applied to a copy of A) and diff.
    fn compare(
        &self,
        params: &Params,
        mutate: impl FnOnce(&mut Params),
    ) -> Option<Comparison> {
        let mut params_b = params.clone();
        mutate(&mut params_b);
        let output_a = self.apply(params)?;
        let output_b = self.apply(&params_b)?;
        let diff = calculate_diff_split(&output_a, &output_b);
        Some(Comparison {
            diff,
            output_a,
            output_b,
        })
    }
}

// ---------------------------------------------------------------------------
// Sentinel payload builders (upstream's static json globals)
// ---------------------------------------------------------------------------

fn sentinel_tools() -> Value {
    let params_schema = json!({
        "type": "object",
        "properties": {
            ARG_FIRST: {"type": "string", "description": "First argument"},
            ARG_SECOND: {"type": "string", "description": "Second argument"},
        },
        "required": [],
    });
    json!([
        {"type": "function", "function": {
            "name": FUN_FIRST, "description": "Test function foo",
            "parameters": params_schema}},
        {"type": "function", "function": {
            "name": FUN_SECOND, "description": "Test function bar",
            "parameters": params_schema}},
    ])
}

fn user_msg() -> Value {
    json!({"role": "user", "content": USER_MSG})
}

fn assistant_msg() -> Value {
    json!({"role": "assistant", "content": ASSISTANT_MSG})
}

fn tool_call(name: &str, args: Value, id: &str) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": args},
    })
}

fn assistant_calls(calls: Vec<Value>) -> Value {
    json!({"role": "assistant", "content": "", "tool_calls": calls})
}

fn first_call() -> Value {
    tool_call(
        FUN_FIRST,
        json!({ARG_FIRST: "XXXX", ARG_SECOND: "YYYY"}),
        CALL_ID_001,
    )
}

fn second_call() -> Value {
    tool_call(
        FUN_SECOND,
        json!({ARG_FIRST: "XXXX", ARG_SECOND: "YYYY"}),
        CALL_ID_002,
    )
}

// ---------------------------------------------------------------------------
// Top level
// ---------------------------------------------------------------------------

/// Derive a [`CallSyntax`] from a chat template's Jinja source.
///
/// ~30 cheap minijinja renders; no FFI, no model required. Analysis
/// never hard-fails on a *supported-compile* template: undetectable
/// aspects come back as their `None`/empty defaults and callers fall
/// back (content-only + warning). Only a template that fails to
/// compile at all errors.
pub fn analyze_template(
    source: &str,
    bos: &str,
    eos: &str,
) -> Result<CallSyntax, AnalyzeError> {
    // Wholesale source sniffs run BEFORE probing (upstream order:
    // `common_chat_try_specialized_template` precedes the diff
    // analyzer). These templates are exactly the ones whose formats
    // the differential probes can't segment — and whose guard
    // `raise_exception`s the probe payloads may trip — so any probe
    // result would be discarded noise anyway.
    if let Some(hand_built) = sniff_hand_built(source) {
        return Ok(hand_built);
    }

    let probe = Probe::new(source, bos, eos)?;

    let reasoning = analyze_reasoning(&probe);
    let content = analyze_content(&probe, &reasoning);
    let mut syntax = analyze_tools(&probe, &reasoning);
    syntax.reasoning = reasoning;
    syntax.content = content;
    syntax.assistant_start = detect_assistant_start(&probe, &syntax.reasoning);
    syntax.user_start = detect_user_start(&probe);
    collect_preserved_tokens(&mut syntax);

    for patch in PATCHES {
        patch(source, &mut syntax);
    }

    Ok(syntax)
}

/// Post-analysis patches — data, not architecture (upstream
/// `workarounds`). Each inspects the template *source* for a
/// distinctive fragment and corrects fields the differential probes
/// can't see. Grows as fixtures expose misdetections.
type Patch = fn(&str, &mut CallSyntax);

static PATCHES: &[Patch] = &[
    // Old Qwen/DeepSeek reasoning templates: they strip reasoning on
    // re-render (`content.split('</think>')`) so the presence probe
    // can't see it, but the model still emits <think> blocks.
    |src, syntax| {
        if src.contains("content.split('</think>')")
            && !src.contains("reasoning_content")
            && !src.contains("<SPECIAL_12>")
            && syntax.reasoning.mode == ReasoningMode::None
        {
            syntax.reasoning.mode = ReasoningMode::TagBased;
            syntax.reasoning.start = "<think>".into();
            syntax.reasoning.end = "</think>".into();
            push_preserved(&mut syntax.preserved_tokens, "<think>");
            push_preserved(&mut syntax.preserved_tokens, "</think>");
        }
    },
];

/// Source sniffs selecting hand-built dialects wholesale, checked
/// before any differential probing (upstream's specialized-template
/// dispatch). Order matters only if fragments could collide — they
/// don't: Gemma 4's `<|tool_call>call:` / `<|"|>` and Harmony's
/// `<|channel|>` (double pipe — Gemma's channel token is the
/// single-pipe `<|channel>`) are mutually exclusive.
fn sniff_hand_built(src: &str) -> Option<CallSyntax> {
    // Gemma 4 (`common_chat_params_init_gemma4` upstream): bare-key
    // dict format with the asymmetric `<|…>` / `<…|>` marker
    // convention.
    if src.contains("<|tool_call>call:") && src.contains("<|\"|>") {
        return Some(CallSyntax::gemma4());
    }
    // gpt-oss / Harmony: upstream detects by this exact substring
    // (`chat.cpp:2350`).
    if src.contains("<|channel|>") {
        return Some(CallSyntax::gpt_oss());
    }
    None
}

fn push_preserved(tokens: &mut Vec<String>, token: &str) {
    let t = token.trim();
    if !t.is_empty() && !tokens.iter().any(|x| x == t) {
        tokens.push(t.to_string());
    }
}

fn collect_preserved_tokens(syntax: &mut CallSyntax) {
    let mut tokens = std::mem::take(&mut syntax.preserved_tokens);
    for value in [
        &syntax.reasoning.start,
        &syntax.reasoning.end,
        &syntax.content.start,
        &syntax.content.end,
        &syntax.section_start,
        &syntax.section_end,
        &syntax.per_call_start,
        &syntax.per_call_end,
        &syntax.function.name_prefix,
        &syntax.function.name_suffix,
        &syntax.function.close,
        &syntax.arguments.start,
        &syntax.arguments.end,
        &syntax.arguments.name_prefix,
        &syntax.arguments.name_suffix,
        &syntax.arguments.separator,
        &syntax.arguments.value_prefix,
        &syntax.arguments.value_suffix,
        &syntax.arguments.string_quote,
        &syntax.call_id.prefix,
        &syntax.call_id.suffix,
    ] {
        push_preserved(&mut tokens, value);
    }
    syntax.preserved_tokens = tokens;
}

// ---------------------------------------------------------------------------
// Phase 1: reasoning
// ---------------------------------------------------------------------------

fn analyze_reasoning(probe: &Probe) -> ReasoningSyntax {
    let mut r = ReasoningSyntax::default();
    compare_reasoning_presence(probe, &mut r);
    compare_thinking_enabled(probe, &mut r);
    compare_reasoning_scope(probe, &mut r);
    r
}

/// Assistant message with vs without `reasoning_content`: if the
/// reasoning text renders, the markers around it are the reasoning
/// tags.
fn compare_reasoning_presence(probe: &Probe, r: &mut ReasoningSyntax) {
    let params = Params {
        messages: json!([user_msg(), assistant_msg()]),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([user_msg(), {
            "role": "assistant",
            "content": ASSISTANT_MSG,
            "reasoning_content": THINKING_CONTENT,
        }]);
    }) else {
        return;
    };

    if cmp.diff.right.is_empty() || !cmp.diff.right.contains(THINKING_CONTENT) {
        return;
    }
    // Wrapped: marker immediately before AND after the reasoning
    // text in the full B output. Delimiter-style: only after.
    let pos = cmp
        .output_b
        .find(THINKING_CONTENT)
        .expect("checked via diff.right");
    let end = pos + THINKING_CONTENT.len();
    let pre = marker_before(&cmp.output_b, pos);
    let post = marker_after(&cmp.output_b, end);
    match (pre, post) {
        (Some(pre), Some(post)) => {
            r.mode = ReasoningMode::TagBased;
            r.start = pre;
            r.end = post;
        }
        (None, Some(post)) => {
            r.mode = ReasoningMode::TagBased;
            r.end = post;
        }
        _ => {}
    }
}

/// Generation prompt with `enable_thinking` off vs on: templates that
/// pre-open the think tag reveal the start marker here; templates
/// that append a *closed* empty think block when thinking is off
/// reveal the end marker.
fn compare_thinking_enabled(probe: &Probe, r: &mut ReasoningSyntax) {
    let params = Params {
        messages: json!([user_msg()]),
        add_generation_prompt: true,
        enable_thinking: false,
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| p.enable_thinking = true) else {
        return;
    };
    let diff = &cmp.diff;
    let left_trimmed = diff.left.trim();
    let right_trimmed = diff.right.trim();

    if left_trimmed.is_empty() && !diff.right.is_empty() {
        // Thinking-on appends extra tail (e.g. Qwen's "<think>\n").
        if !right_trimmed.is_empty()
            && cmp.output_b.trim_end().ends_with(right_trimmed)
            && r.start.is_empty()
        {
            r.start = diff.right.clone();
            r.mode = ReasoningMode::TagBased;
        }
    } else if right_trimmed.is_empty() && !diff.left.is_empty() {
        // Thinking-OFF appends extra tail (e.g. a pre-closed
        // "<think>\n\n</think>" stub): the extra piece is the end
        // marker, and the marker before it is the start.
        if !left_trimmed.is_empty()
            && cmp.output_a.trim_end().ends_with(left_trimmed)
            && r.end.is_empty()
        {
            let seg =
                prune_whitespace_segments(&segmentize_markers(&cmp.output_a));
            if seg.len() >= 2
                && seg[seg.len() - 1].value() == left_trimmed
                && seg[seg.len() - 2].is_marker()
            {
                r.start = seg[seg.len() - 2].value().to_string();
            }
            r.end = diff.left.clone();
            r.mode = ReasoningMode::TagBased;
        }
    } else if !left_trimmed.is_empty() && !right_trimmed.is_empty() {
        // Noisy full-output diff (e.g. the system message also
        // changes). Tail-anchor: one output's tail may appear in the
        // other with extra reasoning markers appended.
        const ANCHOR_LEN: usize = 64;
        for (base, extended) in [
            (&cmp.output_b, &cmp.output_a),
            (&cmp.output_a, &cmp.output_b),
        ] {
            let len = base.len().min(ANCHOR_LEN);
            // Align to a char boundary for slicing safety.
            let mut start = base.len() - len;
            while !base.is_char_boundary(start) {
                start += 1;
            }
            let anchor = &base[start..];
            let Some(pos) = extended.rfind(anchor) else {
                continue;
            };
            if pos + anchor.len() >= extended.len() {
                continue;
            }
            let extra = extended[pos + anchor.len()..].trim();
            if extra.is_empty() {
                continue;
            }
            let seg = prune_whitespace_segments(&segmentize_markers(extra));
            if seg.len() == 2 && seg[0].is_marker() && seg[1].is_marker() {
                if r.start.is_empty() {
                    r.start = seg[0].value().to_string();
                }
                if r.end.is_empty() {
                    r.end = seg[1].value().to_string();
                }
                r.mode = ReasoningMode::TagBased;
                break;
            }
        }
    }

    if r.mode == ReasoningMode::None && r.start.is_empty() && !r.end.is_empty()
    {
        r.mode = ReasoningMode::TagBased;
    }
}

/// Reasoning-with-content vs reasoning-with-tool-calls: some
/// templates only render reasoning on tool turns.
fn compare_reasoning_scope(probe: &Probe, r: &mut ReasoningSyntax) {
    let params = Params {
        messages: json!([user_msg(), {
            "role": "assistant",
            "content": ASSISTANT_MSG,
            "reasoning_content": THINKING_CONTENT,
        }]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([user_msg(), {
            "role": "assistant",
            "content": null,
            "reasoning_content": THINKING_CONTENT,
            "tool_calls": [tool_call(
                FUN_FIRST,
                json!({ARG_FIRST: "VVVV", ARG_SECOND: "XXXX"}),
                CALL_ID_001,
            )],
        }]);
    }) else {
        return;
    };

    let in_a = cmp.output_a.contains(THINKING_CONTENT);
    let in_b = cmp.output_b.contains(THINKING_CONTENT);
    if !in_a && in_b {
        r.mode = ReasoningMode::ToolsOnly;
        let pos = cmp.output_b.find(THINKING_CONTENT).unwrap();
        let end = pos + THINKING_CONTENT.len();
        match (
            marker_before(&cmp.output_b, pos),
            marker_after(&cmp.output_b, end),
        ) {
            (Some(pre), Some(post)) => {
                r.start = pre;
                r.end = post;
            }
            (None, Some(post)) => r.end = post,
            _ => {
                r.mode = ReasoningMode::None;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2: content
// ---------------------------------------------------------------------------

fn analyze_content(
    probe: &Probe,
    reasoning: &ReasoningSyntax,
) -> ContentSyntax {
    let mut c = ContentSyntax::default();

    let params = Params {
        messages: json!([user_msg(), assistant_msg()]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let cmp_tools = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                "test_func",
                json!({"arg1": "value1"}),
                CALL_ID_001,
            )])
        ]);
    });
    let cmp_reasoning = probe.compare(&params, |p| {
        p.messages = json!([user_msg(), {
            "role": "assistant",
            "content": "",
            "reasoning_content": THINKING_CONTENT,
        }]);
    });
    let (Some(cmp_tools), Some(cmp_reasoning)) = (cmp_tools, cmp_reasoning)
    else {
        return c;
    };

    let mut found_plain = false;
    if cmp_tools.diff.left.trim() == ASSISTANT_MSG {
        // Content appears bare in the diff — check the reasoning
        // diff agrees (content alone, possibly after reasoning.end).
        let rdiff_left = cmp_reasoning.diff.left.trim();
        if rdiff_left == ASSISTANT_MSG
            || rdiff_left.is_empty()
            || segmentize_markers(rdiff_left).len() <= 2
        {
            c.mode = ContentMode::Plain;
            found_plain = true;
        } else if reasoning.mode != ReasoningMode::None
            && !reasoning.end.is_empty()
        {
            // "…</think> CONTENT" shape still counts as plain.
            let e = reasoning.end.trim();
            if let Some(pos) = cmp_reasoning.diff.left.find(e) {
                if cmp_reasoning.diff.left[pos + e.len()..]
                    .contains(ASSISTANT_MSG)
                {
                    c.mode = ContentMode::Plain;
                    found_plain = true;
                }
            }
        }
    }
    if !found_plain {
        // Look for wrapper markers immediately around the content.
        let mut rdiff = cmp_reasoning.diff.left.clone();
        if !reasoning.end.is_empty() {
            if let Some(pos) = rdiff.find(reasoning.end.trim()) {
                rdiff = rdiff[pos + reasoning.end.trim().len()..].to_string();
            }
        }
        let pure = if rdiff.len() > cmp_tools.diff.left.len() {
            rdiff
        } else {
            cmp_tools.diff.left.clone()
        };
        if let Some(pos) = pure.find(ASSISTANT_MSG) {
            let end = pos + ASSISTANT_MSG.len();
            if let (Some(pre), Some(post)) =
                (marker_before(&pure, pos), marker_after(&pure, end))
            {
                c.start = pre;
                c.end = post;
            }
        }
    }

    if !c.start.is_empty() || !c.end.is_empty() {
        c.mode = ContentMode::AlwaysWrapped;
    }
    c
}

// ---------------------------------------------------------------------------
// Phase 3: tools
// ---------------------------------------------------------------------------

fn analyze_tools(probe: &Probe, reasoning: &ReasoningSyntax) -> CallSyntax {
    let mut syntax = CallSyntax::default();

    let Some(haystack) = tool_call_haystack(probe) else {
        return syntax;
    };
    if haystack.is_empty() || !haystack.contains(FUN_FIRST) {
        // Template drops tool calls entirely → Family::None; the
        // Instructed dialect owns this case.
        return syntax;
    }

    classify_and_extract(probe, &haystack, reasoning, &mut syntax);

    if syntax.family == Family::TagWithJson
        || syntax.family == Family::TagWithTagged
    {
        check_per_call_markers(probe, &mut syntax);
        extract_function_markers(probe, &mut syntax);
        if syntax.family == Family::TagWithTagged {
            extract_argument_name_markers(probe, &mut syntax);
            extract_argument_value_markers(probe, &mut syntax);
        }
        extract_argument_separator(probe, &mut syntax);
        extract_args_markers(probe, &mut syntax);
        extract_call_id_markers(probe, &mut syntax);
    } else if syntax.family == Family::JsonNative {
        analyze_json_native_parallel_calls(probe, &mut syntax);
        extract_args_markers(probe, &mut syntax);
    }

    syntax
}

/// The tool-call section: diff of assistant-with-content vs
/// assistant-with-one-call.
fn tool_call_haystack(probe: &Probe) -> Option<String> {
    let params = Params {
        messages: json!([user_msg(), assistant_msg()]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let cmp = probe.compare(&params, |p| {
        p.messages = json!([user_msg(), assistant_calls(vec![first_call()])]);
    })?;
    Some(cmp.diff.right)
}

/// Family classification + section/per-call markers.
fn classify_and_extract(
    probe: &Probe,
    haystack: &str,
    reasoning: &ReasoningSyntax,
    syntax: &mut CallSyntax,
) {
    // Quoted-needle-in-JSON-context test: `"NEEDLE"` present with a
    // `{` or `:` somewhere before it.
    let in_json = |needle: &str| -> bool {
        let quoted = format!("\"{needle}\"");
        match haystack.find(&quoted) {
            Some(pos) => haystack[..pos].contains(['{', ':']),
            None => false,
        }
    };

    syntax.family = if in_json(FUN_FIRST) {
        Family::JsonNative
    } else if in_json(ARG_FIRST) {
        Family::TagWithJson
    } else {
        Family::TagWithTagged
    };

    // Strip reasoning markers before structural extraction.
    let mut clean = haystack.to_string();
    for marker in [&reasoning.start, &reasoning.end] {
        if !marker.is_empty() {
            if let Some(pos) = clean.find(marker.as_str()) {
                clean.replace_range(pos..pos + marker.len(), "");
            }
        }
    }

    if syntax.family == Family::JsonNative {
        extract_json_native(&clean, syntax);
    } else {
        extract_non_json_sections(&clean, syntax);
    }
    let _ = probe; // parallel-call refinement happens in callers

    // Trailing whitespace after ending markers is turn-end layout,
    // not call structure: drop it. LEADING whitespace stays — it is
    // the canonical separator between the last value and the closer
    // (e.g. Hermes' `{json}\n</tool_call>`), and render_reference
    // must reproduce it byte-for-byte. The parser matches closers
    // whitespace-tolerantly, so keeping it costs nothing there.
    syntax.section_end = syntax.section_end.trim_end().to_string();
    syntax.per_call_end = syntax.per_call_end.trim_end().to_string();
}

/// JSON_NATIVE: parse the call object out of the haystack and map
/// fields by sentinel values; text before/after the JSON is the
/// section wrapper.
fn extract_json_native(clean: &str, syntax: &mut CallSyntax) {
    let Some(json_start) = clean.find('{') else {
        return;
    };
    let Some(json_end) = clean.rfind('}') else {
        return;
    };
    let cut = &clean[json_start..=json_end];
    let Ok(call_struct) = serde_json::from_str::<Value>(cut) else {
        return;
    };
    let Some(obj) = call_struct.as_object() else {
        return;
    };

    syntax.json = JsonFields {
        name_field: String::new(),
        args_field: String::new(),
        ..JsonFields::default()
    };
    let register =
        |prefix: &str, key: &str, val: &Value, json: &mut JsonFields| {
            let path = if prefix.is_empty() {
                key.to_string()
            } else {
                format!("{prefix}.{key}")
            };
            if val.as_str().is_some_and(|s| s.contains("call0000")) {
                json.id_field = path;
            } else if val.as_str() == Some(FUN_FIRST) {
                json.name_field = path;
            } else if val.to_string().contains(ARG_FIRST) {
                json.args_field = path;
            } else if key.contains("id") {
                json.gen_id_field = path;
            }
        };
    for (key, val) in obj {
        if key == FUN_FIRST {
            syntax.json.fun_name_is_key = true;
            syntax.json.name_field.clear();
            syntax.json.args_field.clear();
        } else {
            if val.is_object() && !val.to_string().contains(ARG_FIRST) {
                syntax.json.function_field = key.clone();
                for (sub_key, sub_val) in val.as_object().unwrap() {
                    register(key, sub_key, sub_val, &mut syntax.json);
                }
            }
            register("", key, val, &mut syntax.json);
        }
    }

    // Array-wrapped parallel calls: `[ {call} ]`.
    let mut start = json_start;
    let mut end_excl = json_end + 1;
    let before = clean[..json_start].trim_end();
    let after = clean[json_end + 1..].trim_start();
    if before.ends_with('[') && after.starts_with(']') {
        syntax.json.tools_array_wrapped = true;
        start = clean[..json_start].rfind('[').expect("checked ends_with");
        end_excl = json_end
            + 1
            + clean[json_end + 1..]
                .find(']')
                .expect("checked starts_with")
            + 1;
    }

    // Field render order for emitter parity.
    let mut located: Vec<(usize, String)> = Vec::new();
    for field in [
        &syntax.json.name_field,
        &syntax.json.args_field,
        &syntax.json.id_field,
        &syntax.json.gen_id_field,
    ] {
        if !field.is_empty() {
            if let Some(pos) = clean.find(field.as_str()) {
                located.push((pos, field.clone()));
            }
        }
    }
    located.sort();
    syntax.json.parameter_order = located.into_iter().map(|(_, f)| f).collect();

    syntax.section_start = clean[..start].trim_start().to_string();
    syntax.section_end = clean[end_excl..].trim_end().to_string();
    if syntax.json.tools_array_wrapped && syntax.section_end.trim() == "]" {
        syntax.section_end.clear();
    }
}

/// Tagged formats: leading markers before the function marker are
/// section/per-call openers; trailing markers after it are closers.
fn extract_non_json_sections(clean: &str, syntax: &mut CallSyntax) {
    // The function marker: the `<...>`/`[...]` tag containing
    // FUN_FIRST, or the bare needle when untagged.
    let segs = segmentize_markers(clean);
    let fun_marker = segs
        .iter()
        .find(|s| s.is_marker() && s.value().contains(FUN_FIRST))
        .map(|s| s.value().to_string())
        .unwrap_or_else(|| FUN_FIRST.to_string());

    let Some(fun_pos) = clean.find(&fun_marker) else {
        return;
    };

    // Leading markers (whitespace between them allowed): up to two.
    // Text between the last leading marker and the function marker
    // becomes the function-name prefix seed.
    let head = &clean[..fun_pos];
    let head_segs = segmentize_markers(head);
    let mut leading: Vec<&Segment> = Vec::new();
    let mut trailing_text = String::new();
    for seg in &head_segs {
        match seg {
            Segment::Marker(_) if trailing_text.trim().is_empty() => {
                leading.push(seg);
                trailing_text.clear();
            }
            Segment::Marker(_) => break,
            Segment::Text(t) => trailing_text.push_str(t),
        }
    }
    let fun_pre = if trailing_text.trim().is_empty() {
        String::new()
    } else {
        trailing_text.trim_start().to_string()
    };

    // Attach each leading marker's trailing whitespace (up to the
    // next segment) by slicing the original text.
    let marker_with_ws = |idx: usize| -> String {
        let mut offset = 0usize;
        for seg in head_segs.iter().take(idx) {
            offset += seg.value().len();
        }
        let start = offset;
        let mut end = start + head_segs[idx].value().len();
        // Extend through whitespace to the next non-ws byte.
        while end < head.len() && head.as_bytes()[end].is_ascii_whitespace() {
            end += 1;
        }
        head[start..end].to_string()
    };
    // Map leading segment refs back to indices in head_segs.
    let leading_indices: Vec<usize> = head_segs
        .iter()
        .enumerate()
        .filter(|(_, s)| s.is_marker())
        .map(|(i, _)| i)
        .take(leading.len())
        .collect();

    match leading_indices.len() {
        0 => {}
        1 => {
            syntax.section_start = marker_with_ws(leading_indices[0]);
        }
        _ => {
            syntax.section_start = marker_with_ws(leading_indices[0]);
            syntax.per_call_start = marker_with_ws(leading_indices[1]);
        }
    }
    syntax.function.name_prefix = fun_pre;

    // Trailing markers at the very end of the haystack: one → section
    // end; two → per-call end + section end.
    let rest = &clean[fun_pos + fun_marker.len()..];
    let rest_segs = prune_whitespace_segments(&segmentize_markers(rest));
    let mut tail_markers: Vec<&str> = Vec::new();
    for seg in rest_segs.iter().rev() {
        if seg.is_marker() && tail_markers.len() < 2 {
            tail_markers.push(seg.value());
        } else {
            break;
        }
    }
    match tail_markers.len() {
        0 => {}
        1 => syntax.section_end = tail_markers[0].to_string(),
        _ => {
            // tail_markers is reversed: [last, second-to-last].
            syntax.section_end = tail_markers[0].to_string();
            syntax.per_call_end = tail_markers[1].to_string();
        }
    }
}

/// One call vs two: when the second call repeats the "section"
/// marker, it was actually a per-call marker.
fn check_per_call_markers(probe: &Probe, syntax: &mut CallSyntax) {
    let params = Params {
        messages: json!([user_msg(), assistant_calls(vec![first_call()])]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![first_call(), second_call()]),
        ]);
    }) else {
        return;
    };

    let filtered = calculate_diff_split(&cmp.diff.suffix, &cmp.diff.right);
    let second = filtered.right.trim_start();
    if !syntax.section_start.is_empty()
        && second.starts_with(syntax.section_start.trim_end())
    {
        // The whitespace between call 1's end marker and call 2's
        // start marker is the template's inter-call separator (the
        // `loop.first`-gated `"\n"`). It was being discarded by
        // `trim_start`; capture it so the grammar and re-render weave
        // it back in and an N-call turn round-trips byte-exact (#58).
        let sep_len = filtered.right.len() - second.len();
        syntax.call_separator = filtered.right[..sep_len].to_string();
        syntax.per_call_start = std::mem::take(&mut syntax.section_start);
        syntax.per_call_end = std::mem::take(&mut syntax.section_end);
    }
}

/// JSON native, one call vs two: same flip as
/// [`check_per_call_markers`] but keyed on the second call's diff
/// containing the section marker.
fn analyze_json_native_parallel_calls(probe: &Probe, syntax: &mut CallSyntax) {
    let params = Params {
        messages: json!([user_msg(), assistant_calls(vec![first_call()])]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![first_call(), second_call()]),
        ]);
    }) else {
        return;
    };
    if !syntax.section_start.is_empty() {
        let marker = syntax.section_start.trim();
        if let Some(pos) = cmp.diff.right.find(marker) {
            // Bytes before the second call's start marker are the
            // inter-call separator, when they're pure whitespace
            // (#58); anything else means the diff isn't a clean call
            // boundary, so leave the separator empty.
            let lead = &cmp.diff.right[..pos];
            if !lead.is_empty() && lead.chars().all(char::is_whitespace) {
                syntax.call_separator = lead.to_string();
            }
            syntax.per_call_start = std::mem::take(&mut syntax.section_start);
            syntax.per_call_end = std::mem::take(&mut syntax.section_end);
        }
    }
}

/// FUN_FIRST vs FUN_SECOND: the diff brackets the function name;
/// extract name prefix/suffix and the function close marker.
fn extract_function_markers(probe: &Probe, syntax: &mut CallSyntax) {
    let params = Params {
        messages: json!([user_msg(), assistant_calls(vec![first_call()])]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_SECOND,
                json!({ARG_FIRST: "XXXX", ARG_SECOND: "YYYY"}),
                CALL_ID_002,
            )]),
        ]);
    }) else {
        return;
    };
    let diff = &cmp.diff;
    if !diff.left.contains(FUN_FIRST) || !diff.right.contains(FUN_SECOND) {
        return;
    }

    // Name prefix from the shared prefix after the opener marker...
    let opener = if !syntax.per_call_start.is_empty() {
        syntax.per_call_start.clone()
    } else {
        syntax.section_start.clone()
    };
    let mut name_prefix = String::new();
    if !opener.is_empty() {
        if let Some(pos) = diff.prefix.rfind(&opener) {
            name_prefix = diff.prefix[pos + opener.len()..].to_string();
        }
    }
    // ...plus the part of diff.left before the needle.
    let fun_pos = diff.left.find(FUN_FIRST).unwrap();
    name_prefix.push_str(&diff.left[..fun_pos]);
    if !name_prefix.is_empty() {
        syntax.function.name_prefix = name_prefix;
    }

    // Name suffix: after the needle in diff.left up to a marker
    // boundary, extended into diff.suffix.
    let after_needle = &diff.left[fun_pos + FUN_FIRST.len()..];
    let mut name_suffix = text_before_first_marker(after_needle);
    if syntax.family == Family::TagWithJson {
        // Extend to the first `{` / `[`, markers included, but only
        // if a marker follows the JSON later (upstream guard).
        let ext = json_prefix_extension(&diff.suffix);
        name_suffix.push_str(&ext);
    } else {
        name_suffix.push_str(&text_before_first_marker(&diff.suffix));
    }
    syntax.function.name_suffix = name_suffix;

    // Close marker: between the last arg value (YYYY) and the
    // per-call/section end.
    let closer_marker = if !syntax.per_call_end.is_empty() {
        syntax.per_call_end.clone()
    } else {
        syntax.section_end.clone()
    };
    let closer_suffix = if closer_marker.is_empty() {
        // No end marker: diff against a no-call render instead.
        probe
            .compare(&params, |p| {
                p.messages = json!([user_msg(), assistant_msg()]);
            })
            .map(|c| {
                let l = &c.diff.left;
                match l.find("YYYY") {
                    Some(pos) => l[pos + 4..].to_string(),
                    None => String::new(),
                }
            })
            .unwrap_or_default()
    } else {
        match diff.suffix.find(&closer_marker) {
            Some(pos) => diff.suffix[..pos].to_string(),
            None => String::new(),
        }
    };
    if !closer_suffix.is_empty() {
        match syntax.family {
            Family::TagWithTagged => {
                // After the last value: skip the value-closing
                // marker, the rest is the function close.
                if let Some(pos) = closer_suffix.find("YYYY") {
                    let after = &closer_suffix[pos + 4..];
                    let segs = segmentize_markers(after);
                    let mut consumed = 0usize;
                    let mut seen_marker = false;
                    for seg in &segs {
                        consumed += seg.value().len();
                        if seg.is_marker() {
                            seen_marker = true;
                            break;
                        }
                        if !seg.value().trim().is_empty() {
                            break;
                        }
                    }
                    if seen_marker {
                        syntax.function.close =
                            after[consumed..].trim_start().to_string();
                    }
                }
            }
            Family::TagWithJson => {
                if let Some(pos) = closer_suffix.find("YYYY") {
                    let post = &closer_suffix[pos + 4..];
                    if let Some(brace) = post.rfind(['}', ']']) {
                        if brace < post.len() - 1 {
                            syntax.function.close =
                                post[brace + 1..].trim_start().to_string();
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

/// Leading text of `s` up to (not including) the first marker.
fn text_before_first_marker(s: &str) -> String {
    let mut out = String::new();
    for seg in segmentize_markers(s) {
        match seg {
            Segment::Text(t) => out.push_str(&t),
            Segment::Marker(_) => break,
        }
    }
    out
}

/// For TAG_WITH_JSON name suffixes: everything (markers included) up
/// to the first `{`/`[` in text, provided at least one marker follows
/// later (mirrors upstream's guard).
fn json_prefix_extension(suffix: &str) -> String {
    let segs = segmentize_markers(suffix);
    let mut out = String::new();
    let mut cut_at_json = None;
    for seg in &segs {
        match seg {
            Segment::Marker(m) => out.push_str(m),
            Segment::Text(t) => match t.find(['{', '[']) {
                Some(pos) => {
                    out.push_str(&t[..pos]);
                    cut_at_json = Some(());
                    break;
                }
                None => out.push_str(t),
            },
        }
    }
    if cut_at_json.is_none() {
        return String::new();
    }
    // Guard: a marker must follow the JSON content later in suffix.
    let after_out = &suffix[out.len()..];
    if find_first_marker(after_out).is_none() {
        return String::new();
    }
    out
}

/// ARG_FIRST vs ARG_SECOND (single-arg calls): the diff brackets the
/// argument name.
fn extract_argument_name_markers(probe: &Probe, syntax: &mut CallSyntax) {
    let params = Params {
        messages: json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_FIRST,
                json!({ARG_FIRST: "XXXX"}),
                CALL_ID_001,
            )])
        ]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_FIRST,
                json!({ARG_SECOND: "YYYY"}),
                CALL_ID_001,
            )])
        ]);
    }) else {
        return;
    };
    let diff = &cmp.diff;
    if diff.left.is_empty() || diff.right.is_empty() {
        return;
    }

    let left_pos = diff.left.find(ARG_FIRST);
    let right_pos = diff.right.find(ARG_SECOND);
    let (Some(lp), Some(rp)) = (left_pos, right_pos) else {
        return;
    };

    let l_pre = &diff.left[..lp];
    let r_pre = &diff.right[..rp];
    let l_suf = until_any(&diff.left[lp + ARG_FIRST.len()..], &['"', 'X']);
    let r_suf = until_any(&diff.right[rp + ARG_SECOND.len()..], &['"', 'Y']);

    if !l_pre.is_empty() && l_pre == r_pre && l_suf == r_suf {
        // Name inside shared structure (e.g. `<parameter=`).
        syntax.arguments.name_prefix = l_pre.to_string();
        syntax.arguments.name_suffix = l_suf;
    } else if lp == 0 && rp == 0 {
        // Name at the diff start: prefix is the last marker (plus
        // trailing text) of diff.prefix.
        syntax.arguments.name_prefix = last_marker_through_end(&diff.prefix)
            .unwrap_or_else(|| diff.prefix.clone());
        // Suffix runs to the first marker (inclusive) in
        // left-remainder + shared suffix.
        let after_first =
            format!("{}{}", &diff.left[ARG_FIRST.len()..], diff.suffix);
        if let Some(pos) = after_first.find(ARG_FIRST) {
            // Shouldn't happen; guard against pathological repeats.
            let _ = pos;
        }
        let mut consumed = String::new();
        for seg in segmentize_markers(&after_first) {
            consumed.push_str(seg.value());
            if seg.is_marker() {
                break;
            }
        }
        // Include trailing whitespace after the marker, like
        // upstream's `p.marker() + p.space()`.
        let rest = &after_first[consumed.len()..];
        let ws_len = rest.len() - rest.trim_start().len();
        consumed.push_str(&rest[..ws_len]);
        syntax.arguments.name_suffix = consumed;
    }
}

/// `s` up to the first occurrence of any needle char.
fn until_any(s: &str, needles: &[char]) -> String {
    match s.find(|c| needles.contains(&c)) {
        Some(pos) => s[..pos].to_string(),
        None => s.to_string(),
    }
}

/// The last marker in `s` plus everything after it, or `None` when
/// `s` has no marker (upstream's `last_marker + trailing text`).
fn last_marker_through_end(s: &str) -> Option<String> {
    let segs = segmentize_markers(s);
    let mut offset = 0usize;
    let mut last_marker_start = None;
    for seg in &segs {
        if seg.is_marker() {
            last_marker_start = Some(offset);
        }
        offset += seg.value().len();
    }
    last_marker_start.map(|p| s[p..].to_string())
}

/// XXXX vs YYYY (same arg): the diff brackets the value.
fn extract_argument_value_markers(probe: &Probe, syntax: &mut CallSyntax) {
    let params = Params {
        messages: json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_FIRST,
                json!({ARG_FIRST: "XXXX"}),
                CALL_ID_001,
            )])
        ]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_FIRST,
                json!({ARG_FIRST: "YYYY"}),
                CALL_ID_001,
            )])
        ]);
    }) else {
        return;
    };
    let diff = &cmp.diff;
    if diff.left != "XXXX" || diff.right != "YYYY" {
        return;
    }

    let arg_name_ending =
        format!("{ARG_FIRST}{}", syntax.arguments.name_suffix);
    let mut prefix = diff.prefix.clone();
    if let Some(pos) = prefix.rfind(&arg_name_ending) {
        prefix = prefix[pos + arg_name_ending.len()..].to_string();
    }
    if !prefix.is_empty() {
        syntax.arguments.value_prefix =
            last_marker_through_end(&prefix).unwrap_or(prefix);
    }

    let mut value_suffix = diff.suffix.clone();
    if !syntax.function.close.is_empty() {
        if let Some(pos) = value_suffix.find(&syntax.function.close) {
            value_suffix.truncate(pos);
        }
    } else {
        let end_marker = if !syntax.per_call_end.is_empty() {
            syntax.per_call_end.clone()
        } else {
            syntax.section_end.clone()
        };
        if !end_marker.is_empty() {
            if let Some(pos) = value_suffix.find(&end_marker) {
                value_suffix.truncate(pos);
            }
        }
    }
    if !value_suffix.trim().is_empty() {
        syntax.arguments.value_suffix = value_suffix;
    }
}

/// One arg vs two args: text before the second arg's name is the
/// separator.
fn extract_argument_separator(probe: &Probe, syntax: &mut CallSyntax) {
    let params = Params {
        messages: json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_FIRST,
                json!({ARG_FIRST: "XXXX"}),
                CALL_ID_001,
            )])
        ]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([user_msg(), assistant_calls(vec![first_call()])]);
    }) else {
        return;
    };
    if !cmp.diff.right.is_empty() {
        syntax.arguments.separator =
            until_common_prefix(&cmp.diff.right, ARG_FIRST, ARG_SECOND);
    }
}

/// Zero args vs one arg: wrapper markers around the argument list.
fn extract_args_markers(probe: &Probe, syntax: &mut CallSyntax) {
    if syntax.family != Family::JsonNative {
        return;
    }
    let params = Params {
        messages: json!([
            user_msg(),
            assistant_calls(vec![
                tool_call(FUN_FIRST, json!({}), CALL_ID_001,)
            ])
        ]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_FIRST,
                json!({ARG_FIRST: "XXXX"}),
                CALL_ID_001,
            )])
        ]);
    }) else {
        return;
    };
    let diff = &cmp.diff;

    let prefix_marker = if !syntax.section_start.is_empty() {
        syntax.section_start.clone()
    } else {
        syntax.per_call_start.clone()
    };
    let suffix_marker = if !syntax.section_end.is_empty() {
        syntax.section_end.clone()
    } else {
        syntax.per_call_end.clone()
    };
    // Cut past the marker only when it is actually *found* — the old
    // shape skipped `marker.len()` unrelated bytes on a miss
    // (`rfind(..).unwrap_or(0)` then an unconditional advance), which
    // both chopped real content and could slice mid-char on
    // non-ASCII.
    let prefix_cut = if prefix_marker.is_empty() {
        diff.prefix.as_str()
    } else {
        match diff.prefix.rfind(&prefix_marker) {
            Some(pos) => &diff.prefix[pos + prefix_marker.len()..],
            None => diff.prefix.as_str(),
        }
    };
    let suffix_cut = if suffix_marker.is_empty() {
        diff.suffix.as_str()
    } else {
        &diff.suffix[..diff
            .suffix
            .find(&suffix_marker)
            .unwrap_or(diff.suffix.len())]
    };

    let mut args_start = until_common_prefix(prefix_cut, "{}", "{\"first\":");
    let args_end = after_common_suffix(suffix_cut, "{}", "\"XXXX\"}");

    if !args_start.is_empty() || !args_end.is_empty() {
        if let Some(pos) = args_start.find(FUN_FIRST) {
            args_start = args_start[pos + FUN_FIRST.len()..].to_string();
        }
        if let Some(pos) = args_start.find(CALL_ID_001) {
            args_start = args_start[pos + CALL_ID_001.len()..].to_string();
        }
        syntax.arguments.start = args_start;
        syntax.arguments.end = args_end;
    }
}

/// call00001 vs call99999: locate the call-ID slot for tagged
/// formats.
fn extract_call_id_markers(probe: &Probe, syntax: &mut CallSyntax) {
    let params = Params {
        messages: json!([user_msg(), assistant_calls(vec![first_call()])]),
        tools: sentinel_tools(),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([
            user_msg(),
            assistant_calls(vec![tool_call(
                FUN_FIRST,
                json!({ARG_FIRST: "XXXX", ARG_SECOND: "YYYY"}),
                CALL_ID_999,
            )])
        ]);
    }) else {
        return;
    };
    let diff = &cmp.diff;
    if diff.left.is_empty() && diff.right.is_empty() {
        // Template drops call IDs (Qwen does).
        return;
    }

    // "call" is the common prefix of the two sentinels.
    let common_id_part = "call";
    let func_in_prefix = diff.prefix.rfind(FUN_FIRST);
    let func_in_suffix = diff.suffix.find(FUN_FIRST);

    match (func_in_prefix, func_in_suffix) {
        (Some(fp), None) => {
            let args_in_prefix = diff.prefix[fp..].find('{').map(|p| p + fp);
            let args_in_suffix = diff.suffix.find('{');
            if args_in_suffix.is_some() && args_in_prefix.is_none() {
                syntax.call_id.position = CallIdPosition::BetweenFuncAndArgs;
                let after_func = &diff.prefix[fp + FUN_FIRST.len()..];
                // Marker preceding the id (common part may be in
                // prefix when ids share the "call" prefix).
                let id_pos =
                    after_func.find(common_id_part).unwrap_or(after_func.len());
                syntax.call_id.prefix = marker_before(after_func, id_pos)
                    .or_else(|| find_last_marker(after_func))
                    .unwrap_or_default();
                // First marker in the suffix before args.
                let before_args = &diff.suffix[..args_in_suffix.unwrap_or(0)];
                syntax.call_id.suffix =
                    find_first_marker(before_args).unwrap_or_default();
            } else if let Some(ap) = args_in_prefix {
                syntax.call_id.position = CallIdPosition::PostArgs;
                let after_args = &diff.prefix[ap..];
                if let Some(cb) = after_args.rfind('}') {
                    syntax.call_id.prefix =
                        find_last_marker(&after_args[cb + 1..])
                            .unwrap_or_default();
                }
                syntax.call_id.suffix =
                    find_first_marker(&diff.suffix).unwrap_or_default();
            }
        }
        (None, Some(fs)) => {
            syntax.call_id.position = CallIdPosition::PreFuncName;
            syntax.call_id.prefix =
                find_last_marker(&diff.prefix).unwrap_or_default();
            syntax.call_id.suffix =
                find_first_marker(&diff.suffix[..fs]).unwrap_or_default();
        }
        _ => {}
    }

    if syntax.call_id.prefix == syntax.arguments.end {
        syntax.call_id.prefix.clear();
    }
    if syntax.call_id.suffix == syntax.arguments.start {
        syntax.call_id.suffix.clear();
    }
    if syntax.call_id.position != CallIdPosition::None
        && !syntax.call_id.suffix.is_empty()
        && syntax.per_call_end.starts_with(&syntax.call_id.suffix)
    {
        syntax.per_call_end.clear();
    }
}

// ---------------------------------------------------------------------------
// Message-start markers
// ---------------------------------------------------------------------------

fn detect_assistant_start(
    probe: &Probe,
    reasoning: &ReasoningSyntax,
) -> String {
    let params = Params {
        messages: json!([user_msg()]),
        ..Params::default()
    };
    let Some(cmp) = probe.compare(&params, |p| {
        p.messages = json!([user_msg(), assistant_msg()]);
    }) else {
        return String::new();
    };
    let block = &cmp.diff.right;
    let Some(pos) = block.find(ASSISTANT_MSG) else {
        return String::new();
    };
    let mut prefix = block[..pos].to_string();
    for marker in [reasoning.start.trim(), reasoning.end.trim()] {
        if !marker.is_empty() {
            if let Some(mp) = prefix.find(marker) {
                prefix.truncate(mp);
            }
        }
    }
    prefix.trim().to_string()
}

fn detect_user_start(probe: &Probe) -> String {
    let params = Params::default();
    let cmp = probe
        .compare(&params, |p| p.messages = json!([user_msg()]))
        .filter(|c| c.diff.right.contains(USER_MSG))
        .or_else(|| {
            // Templates that reject empty message lists: diff a
            // two-message base against three.
            let base = Params {
                messages: json!([
                    {"role": "user", "content": "V_USER_MSG Hello END_V"},
                    assistant_msg(),
                ]),
                ..Params::default()
            };
            probe.compare(&base, |p| {
                p.messages = json!([
                    {"role": "user", "content": "V_USER_MSG Hello END_V"},
                    assistant_msg(),
                    user_msg(),
                ]);
            })
        });
    let Some(cmp) = cmp else {
        return String::new();
    };
    let mut block = cmp.diff.right;
    if let Some(pos) = block.find(ASSISTANT_MSG) {
        block = block[pos + ASSISTANT_MSG.len()..].to_string();
    }
    let Some(pos) = block.find(USER_MSG) else {
        return String::new();
    };
    let candidate = &block[..pos];

    // Weed out leading end/close markers (they belong to the prior
    // message), keep the rest.
    let mut result = String::new();
    let mut encountered_marker = false;
    for seg in segmentize_markers(candidate) {
        let lower = seg.value().to_lowercase();
        if seg.is_marker()
            && !encountered_marker
            && (lower.contains("end") || lower.contains("close"))
        {
            continue;
        }
        if !seg.is_marker()
            && !encountered_marker
            && seg.value().trim().is_empty()
        {
            continue;
        }
        encountered_marker |= seg.is_marker();
        result.push_str(seg.value());
    }
    result.trim().to_string()
}
