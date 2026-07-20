---
name: Unsafe audit — drama_llama's own FFI surface
description: The one surviving item from the old post-v0.8.0 cleanup memo. moeflux was audited in 2026-04-28; drama_llama's llama.cpp FFI never was.
type: project
---

# Unsafe audit: `src/llama_cpp/`

Narrowed 2026-07-20. This file used to carry three asks; two are
resolved and were removed rather than left as ghosts:

- ~~Tear out C where we can (moeflux tokenizer, manifest loader,
  `load_vocab`)~~ — **superseded** by the full RIIR arc, which ported
  essentially all of moeflux's C/Obj-C to Rust and retired the C path
  to a `diff-oracle`-gated dev dependency. See
  `riir_moeflux_strategy.md`.
- ~~Cross-backend fuzz with ≥0.95 argmax agreement~~ — **done**:
  `tests/cross_backend.rs:113`, `ARGMAX_AGREEMENT_MIN: f32 = 0.95`,
  the plan's exact bar, asserted on decisive steps.

## What actually remains

The unsafe audit was completed for **moeflux only**
(`riir_unsafe_audit.md`, 2026-04-28, scoped to
`crates/moeflux/src/riir/`). drama_llama's own FFI surface has never
had the same pass. Current inventory:

| file | `unsafe` occurrences |
|---|---|
| `src/llama_cpp/mtmd.rs` | 46 |
| `src/llama_cpp/model.rs` | 44 |
| `src/llama_cpp/decoder.rs` | 42 |
| `src/llama_cpp/engine.rs` | 5 |

`src/llama_cpp/mtmd.rs` is the priority: it is the newest of the four
(landed 2026-07-11 for #31), the least exercised, and it is where the
hand-assembled `llama_batch` view lives — `EmbdBatch` builds a
`llama_batch` by hand rather than through `llama_batch_init`, with
Rust-owned buffers and a documented no-`llama_batch_free` invariant.
That is exactly the shape where a load-bearing invariant goes unstated.

**Correction to the original memo's second target**: it named
`src/backend.rs` for "the Token / TokenData ABI-compatibility claim vs
the C layouts". That file now contains **zero** `unsafe` blocks — it
holds the *claim* (`:11` transmute-compatibility, `:14` ABI
compatibility, `:24-29` `#[repr(C)]` with field order matching
llama.cpp) while the `unsafe` that *relies* on the claim lives
elsewhere (`candidates.rs` and the llama_cpp modules). Audit the pair
together; a doc comment asserting ABI compatibility is only as good as
the transmute sites trusting it, and they are not co-located.

## Method

Per unsafe block: write down its invariant, then flag the ones where
the invariant is load-bearing but not obviously stated at the site.
Same format as `riir_unsafe_audit.md`, which is the worked example.

Still a dedicated session, not folded into feature work.
