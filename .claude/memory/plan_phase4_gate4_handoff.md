# Phase 4 gate 4 handoff

Gates 1–3 landed on 2026-04-24 across moeflux commits `32dd06e`
through `6e600ef`. This doc captures what gate 4 needs so a future
session can pick up without re-reading the whole arc.

## What's already available

- `moeflux` safe wrapper (`~/Projects/moeflux/crates/moeflux`)
  exposes `Ctx` with: `open`, `n_vocab`, `n_ctx`, `eos`,
  `model_name`, `eval_prompt`, `eval_token`, `memory_clear`,
  `memory_seq_rm`, `memory_seq_pos_max`, `state_size`,
  `state_save`, `state_load`. Send + !Sync. Drop calls
  `mf_free_model`. Auto-sets `MOEFLUX_SHADERS_PATH`.
- `moeflux-sys` for raw FFI if anything the safe wrapper doesn't
  cover comes up.
- Model-variant feature flags `model-qwen3-5-a17b` and
  `model-qwen3-6-35b-a3b` — exactly one required.
- Ready-to-load 35B-A3B artifacts at
  `/Volumes/Temp Backup/models/moeflux/qwen3-6-35b-a3b-{root,artifacts}`
  (MLX-side) and `/Volumes/Temp Backup/models/gguf/qwen3-6-35b-a3b-q4_k_m.gguf`
  (GGUF-side for cross-backend).

## What's missing

### 1. `MoefluxDecoder` impl (easy)

One-to-one over the existing `Decoder` trait:

| trait method       | moeflux::Ctx         |
| ------------------ | -------------------- |
| `prefill`          | `eval_prompt`        |
| `step`             | `eval_token`         |
| `memory_clear`     | `memory_clear`       |
| `memory_seq_rm`    | `memory_seq_rm`      |
| `n_vocab`          | `n_vocab`            |
| `n_ctx`            | `n_ctx`              |
| `eos`              | `eos`                |

`Ctx::Drop` runs on struct drop; nothing extra needed. The logits
slice needs attention — `moeflux::Ctx::eval_*` takes a caller-owned
`&mut [f32]`, while `Decoder::step` returns `Result<&[f32]>`. Store
a `Vec<f32>` in the decoder struct sized to `n_vocab`, hand out
`&self.logits[..]`.

### 2. `MoefluxModel` impl (harder — tokenization)

moeflux takes token IDs only, so the Model trait's
tokenize/detokenize/token_to_piece/special_tokens/chat_template
surface needs a Rust tokenizer. Plan is the HuggingFace
[`tokenizers`](https://crates.io/crates/tokenizers) crate:

- Load `tokenizer.json` from the MLX model dir (same file we export
  to BPET for the C side).
- Load `chat_template.jinja` from the same dir; apply via
  [`minijinja`](https://crates.io/crates/minijinja) or similar.
- Special tokens: parse `added_tokens` section of tokenizer.json.
- `n_vocab` / `context_size` / `embedding_size` / `get_meta` come
  from the `model_weights.json` manifest moeflux already consumes.
  Thin wrapper around a cached `serde_json::Value`.
- `eos` / `bos` / `eot`: from compile-time moeflux constants
  (exposed via `Ctx::eos`) + tokenizer.json's special tokens.

`moeflux-sys` does not currently re-export the `EOS_TOKEN_*` /
`THINK_*` consts. Either add a getter function to the C API
(`mf_think_start_token` etc.) or read them from tokenizer.json.
Probably the latter — keeps moeflux focused on inference.

### 3. `Engine<D, M>` wiring (mechanical)

Type alias when the feature is on:
```rust
#[cfg(all(feature = "moeflux", target_os = "macos"))]
pub type MoefluxEngine = Engine<MoefluxDecoder, MoefluxModel>;
```

Cargo.toml: `moeflux = { path = "../../moeflux/crates/moeflux",
optional = true, features = ["model-qwen3-6-35b-a3b"] }`. Note the
feature forwarding — drama_llama picks one model at compile time
too, matching moeflux's constraint.

### 4. Cross-backend regression harness

The existing `tests/regression.rs` captures a golden on
`LlamaCppEngine` via `DRAMA_LLAMA_UPDATE_GOLDEN=1`. Plan:

- Extend it with a parallel `moeflux_golden.json` at the same 4-token
  prompt + 32 greedy steps.
- New test `moeflux_matches_llama_cpp` that runs both backends on
  the same prompt and asserts token-argmax agreement ≥ 95% and
  top-20 logit set overlap ≥ 80% (thresholds from the plan — tune
  on first run).
- Gate behind `#[ignore]` since it needs ~40 GB of artifacts and
  both backends linked.

The trickiest bit: both backends in the same test binary. moeflux is
feature-gated `macos`-only; llama.cpp is always available. Use
`#[cfg(all(feature = "moeflux", target_os = "macos"))]` on the
cross-test and run via
`cargo test --features "llama-cpp,moeflux" -- --ignored`.

## Open questions to answer before starting

- Does drama_llama's `Model::chat_template` return the applied
  template (a string) or a closure that applies it? Check
  `src/backend.rs` / `src/llama_cpp/model.rs` for the LlamaCppModel
  side before writing the Moeflux side so we match the contract.
- Same for `Model::get_meta` — probably a key-value lookup; might
  need to invent keys for config fields that only exist in
  model_weights.json, not llama.cpp's gguf metadata.
- Does anything in drama_llama's existing Session / Predictor code
  assume llama.cpp-specific sampling details we haven't surfaced
  through traits? The "deferred work" note in the v0.8.0 plan
  flagged sample_token / repetition / grammar / json as still pinned
  to LlamaCppModel. Touching MoefluxModel will force that unpinning
  — budget time for it.

## Suggested session shape

- Hour 1: Read the trait contracts + existing LlamaCpp impl pattern
  carefully. Write MoefluxDecoder (mechanical).
- Hour 2: MoefluxModel with tokenizers crate. Start simple; skip
  chat_template until the basics work.
- Hour 3: Wire into Engine + fix the sample.rs etc. pins.
- Hour 4: Cross-backend regression test.

If the sample.rs unpinning turns out to be deep, stop and discuss
with Mike — that's where scope can balloon.
