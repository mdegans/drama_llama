//! Phase 2 (#88) design probe: a model's *unforced* tool-call
//! spelling.
//!
//! Renders a tools-bearing prompt through the model's own chat
//! template, then predicts greedily through the raw [`Engine`] path —
//! **no grammar, no Session, no emit bans** — so every byte of the
//! call envelope and argument object is the model's own habit. The
//! output is the design input for owned-template canonical bytes:
//! canonical form derives from what the model does when nothing forces
//! it (`.claude/memory/plan_template_ownership.md`, "canonical bytes
//! derive from the model's unforced habit").
//!
//! This is a measurement, not a regression pin: it asserts only that
//! *something* was emitted, and prints the bytes for the session log.
//! Greedy sampling makes it as repeatable as the hardware allows, but
//! model habit is weather, not contract — the contract lives in the
//! round-trip pins that the measurement informs.
//!
//! Requires a **cogito-family** model (`DRAMA_LLAMA_COGITO_MODEL` or
//! `models/cogito-32b.gguf`); skips loudly rather than substituting
//! whatever `model.gguf` points at (see `hash_cache_smoke.rs` for the
//! incident that made substitution a banned pattern).

#![cfg(feature = "llama-cpp")]

use std::{borrow::Cow, num::NonZeroUsize, path::PathBuf};

use drama_llama::{
    Block, ChatTemplate, Content, LlamaCppEngine, Message, PredictOptions,
    Prompt, RenderOptions, Role, Tool,
};
use serde_json::json;

/// Resolve a probe's model: `$env_var` if set and present, else the
/// conventional path under `models/`. `None` means skip — never
/// substitute `model.gguf`.
fn model_path(env_var: &str, conventional: &str) -> Option<PathBuf> {
    if let Ok(p) = std::env::var(env_var) {
        let p = PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let conventional =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(conventional);
    conventional.exists().then_some(conventional)
}

/// A schema wide enough to observe every separator position: multiple
/// string fields (field separators), an array (element separators),
/// and an integer (value spacing after `:` for a non-string). The
/// user message baits apostrophes so string interiors are non-trivial.
#[test]
#[ignore = "long running"]
fn cogito_unforced_call_spelling() {
    let Some(path) =
        model_path("DRAMA_LLAMA_COGITO_MODEL", "models/cogito-32b.gguf")
    else {
        eprintln!(
            "SKIP: needs a cogito model (DRAMA_LLAMA_COGITO_MODEL or \
             models/cogito-32b.gguf)"
        );
        return;
    };
    probe_unforced_call(path);
}

/// The same measurement for Mistral Small 4, whose owned template
/// currently keeps stock's `tojson`-compact argument interior on the
/// grounds that nothing has measured the alternative. This is that
/// measurement. Two things to read off the printed emission:
///
/// 1. **Separator spacing.** Uniform `": "` / `", "` means the habit is
///    `JsonSpacing::Spaced` and the owned template should swap
///    `| tojson` → `| json_dumps`, exactly as cogito did (#88 phase 2).
///    Compact means stock's spelling was already the habit and the
///    template is right as written.
/// 2. **`[CALL_ID]`.** The vocab has that token but neither template
///    emits it. If the model volunteers one unforced, the re-render
///    cannot reproduce it and every tool turn breaks byte-stability —
///    which would make `CallIdSyntax` a required part of this dialect
///    rather than the `None` the analyzer derived.
#[test]
#[ignore = "long running"]
fn mistral4_unforced_call_spelling() {
    let Some(path) = model_path(
        "DRAMA_LLAMA_MISTRAL_MODEL",
        "models/Mistral-Small-4-119B-2603-UD-IQ3_S.gguf",
    ) else {
        eprintln!(
            "SKIP: needs a Mistral Small 4 model \
             (DRAMA_LLAMA_MISTRAL_MODEL or \
             models/Mistral-Small-4-119B-2603-UD-IQ3_S.gguf)"
        );
        return;
    };
    probe_unforced_call(path);
}

/// The probe body: render a tools-bearing prompt through the model's
/// *own* embedded template, predict greedily with no grammar, print
/// every byte.
fn probe_unforced_call(path: PathBuf) {
    drama_llama::silence_logs();
    let mut engine = LlamaCppEngine::from_path_with_n_ctx(path, 4096)
        .expect("model loads")
        .quiet();

    let tool = Tool::builder("create_post")
        .description("Create a post in a community.")
        .schema(json!({
            "type": "object",
            "properties": {
                "community": {"type": "string"},
                "title": {"type": "string"},
                "body": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "priority": {"type": "integer"}
            },
            "required": ["community", "title", "body", "tags", "priority"]
        }))
        .build()
        .expect("valid probe tool");

    let prompt = Prompt {
        system: Some(Content::text(
            "You are a helpful assistant. Use the provided tool to \
             fulfill the user's request.",
        )),
        messages: vec![Message {
            role: Role::User,
            content: Content(vec![Block::Text {
                text: Cow::Borrowed(
                    "Create a post in the 'debate' community titled \
                     \"Marcus's paradox\" whose body explains in one or \
                     two sentences why it's tricky. Tag it 'philosophy' \
                     and 'logic', priority 2.",
                ),
                cache_control: None,
                citations: None,
            }]),
        }],
        tools: Some(vec![tool.into()]),
        ..Prompt::default()
    };

    let tmpl = ChatTemplate::from_model(engine.model()).expect("template");
    let opts = RenderOptions::default().with_generation_prompt(true);
    let rendered = tmpl.render_with(&prompt, &opts).expect("render");

    println!("=== prompt tail (last 600 bytes) ===");
    let mut tail_start = rendered.len().saturating_sub(600);
    while !rendered.is_char_boundary(tail_start) {
        tail_start += 1;
    }
    println!("{}", &rendered[tail_start..]);

    let tokens = engine.model().tokenize(&rendered, true);
    // Stops are opt-in at the raw Engine layer; without this the
    // predictor runs greedily through `<|im_end|>` and burns the
    // budget on post-EOG degeneration (observed on the first run).
    let mut popts = PredictOptions::greedy().add_model_stops(engine.model());
    popts.n = NonZeroUsize::new(600).unwrap();

    let emission: String = engine.predict_pieces(tokens, popts, None).collect();

    println!("=== unforced emission ({} bytes) ===", emission.len());
    println!("{emission}");
    println!("=== end ===");

    assert!(!emission.is_empty(), "probe produced no output");
}
