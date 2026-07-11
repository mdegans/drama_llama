//! gpt-oss e2e (#30 Phase G): the Harmony dialect against the real
//! model. Mirrors the Phase E/F suites — forced/auto calls, thinking
//! under grammar, round-trip byte-stability, prefix-cache survival —
//! for the channel-structured Harmony format
//! (`<|channel|>commentary to=functions.NAME <|constrain|>json<|message|>{args}<|call|>`).
//!
//! All tests load `models/gpt-oss-20b-UD-Q8_K_XL.gguf` and are
//! `#[ignore]`d. Run with
//! `cargo test --features serde,cuda --test session_gptoss -- --ignored`.

use std::path::PathBuf;

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("models/gpt-oss-20b-UD-Q8_K_XL.gguf")
}

/// The EOG contract Phase G's grammar and stop logic rely on, pinned
/// against the real vocab: `<|return|>` and `<|call|>` end
/// generation, while `<|end|>` (the in-stream channel separator) must
/// NOT — libllama's o200k_harmony workaround removes it from
/// `special_eog_ids`, and `extra_eos_tokens` must surface exactly
/// that set. CPU-only load: vocab introspection needs no GPU.
#[test]
#[ignore = "long running - requires gpt-oss model"]
fn gptoss_eog_token_set() {
    use drama_llama::Model as _;

    let mut params = unsafe { llama_cpp_sys_3::llama_model_default_params() };
    params.n_gpu_layers = 0;
    let model =
        drama_llama::LlamaCppModel::from_file(model_path(), Some(params))
            .expect("model load");

    let piece_of = |t| drama_llama::Model::token_to_piece(&model, t);
    let by_piece = |s: &str| {
        let toks = model.tokenize(s, true);
        assert_eq!(toks.len(), 1, "{s:?} must be a single token: {toks:?}");
        toks[0]
    };

    // GGUF metadata: eos = <|return|>, eot = <|endoftext|>.
    assert_eq!(piece_of(model.eos()), "<|return|>");

    let call = by_piece("<|call|>");
    let end = by_piece("<|end|>");
    let ret = by_piece("<|return|>");

    let extra = model.extra_eos_tokens();
    assert!(
        extra.contains(&call),
        "<|call|> must stop generation (tool-call turn exit); extra = \
         {:?}",
        extra.iter().map(|&t| piece_of(t)).collect::<Vec<_>>()
    );
    assert!(
        !extra.contains(&end) && model.eos() != end && model.eot() != end,
        "<|end|> is the in-stream channel separator and must stay \
         generatable"
    );
    assert!(model.eos() == ret || extra.contains(&ret));
}
