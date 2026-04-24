---
name: Rust-ification + unsafe audit (post-Phase-5)
description: Final QA pass after v0.8.0 ships. Tear out C where safely possible; audit remaining unsafe in drama_llama and moeflux-sys.
type: project
---

# Post-v0.8.0 cleanup / QA pass

After the Phase 4/5 moeflux integration lands and Cogito 600B is
running on Agora's Council, we'll do a dedicated session on:

1. **Tear out C where we can.** moeflux's C/Obj-C has places that
   could reasonably be Rust without touching the Metal kernels:
   - The tokenizer (`tokenizer.h`, `export_tokenizer.py`,
     `export_vocab.py`, `init_tokenizer` path) — redo in Rust
     via the `tokenizers` crate, keep only token-ID flow crossing
     the FFI.
   - The manifest loader + `WeightFile` mmap logic in infer.m —
     small, self-contained, reasonable Rust port behind the FFI.
   - `load_vocab` / `Vocabulary` struct in infer.m — only used for
     display decoding; could be Rust-side in the safe wrapper.
   - The Python prep pipeline (`repack_experts.py`,
     `extract_weights.py`, `gen_expert_index.py`) could eventually
     be a Rust CLI too, though Python is fine there since it's not
     on the hot path.
   Metal shader dispatch + Obj-C pipeline setup stay C — they live
   on top of the Metal framework and translating them buys
   nothing except risk.

2. **Audit remaining unsafe.** Focus on:
   - `drama_llama/src/backend.rs` (Token / TokenData ABI-
     compatibility claim vs the C layouts).
   - `drama_llama/src/llama_cpp/` — the llama.cpp FFI surface.
   - `moeflux-sys` raw bindgen types.
   - `moeflux` safe wrapper's `Drop` + slice-from-pointer paths.
   Write down each unsafe block's invariants; flag ones where
   the invariant is load-bearing but not obviously stated.

3. **Cross-backend fuzz.** Regression harness should already be
   running both backends post-Phase-4. Add a property-style test
   that drives random token sequences through both and asserts
   token-argmax agreement stays within the plan's ≥95% bound.

This is a separate session per Mike's preference — not folded
into the Phase 4/5 work where the priority is landing the
feature. Session should be scheduled after Cogito probe baseline
lands on Agora.
