---
name: llama.cpp FFI audit — drama_llama's unsafe surface
description: 2026-07-20 audit of src/llama_cpp/, batch.rs, candidates.rs, backend.rs against llama.h 0.8.1. Findings, invariants, and what was checked and found clean.
type: project
---

# llama.cpp FFI audit (2026-07-20)

Supersedes `future_work_rust_audit.md` (deleted — the artifact it
asked for is this file). Format follows `riir_unsafe_audit.md`.

**Header of record**: `llama-cpp-sys-3` **v0.8.1** from the registry
(`~/.cargo/registry/src/index.crates.io-*/llama-cpp-sys-3-0.8.1/`),
`external/llama.cpp/include/llama.h` + `tools/mtmd/mtmd.h`. Where the
header comment was stale or silent, the vendored `.cpp` was traced and
that is noted per finding.

**Sync note**: at audit time the local `~/Projects/llama-cpp-sys`
checkout was 8 commits behind `origin/main` and predated the mtmd
feature entirely — the `v0.8.0`/`v0.8.1` tags existed only on the
remote (that release was cut from another machine). Fast-forwarded to
`f60efcc` and the submodule to `52b3df002`; both headers are now
byte-identical to the registry copy. **Before grepping the local
checkout, verify parity** — a header that disagrees with the linked
binary produces confident wrong answers.

Method: per `unsafe` block, state the invariant; read the C doc comment
verbatim; flag divergence. Every finding below marked CONFIRMED was
re-verified by hand in header + C++ source, not taken on a subagent's
word.

## Findings

### 1. Safe-code use-after-free: `&self` on both the mutator and the borrow

`decoder.rs:409` `decode(&self)`, `:424` `prefill_inherent(&self)`,
`:449` `logits(&self, i) -> &[f32]`, `:463` `embeddings(&self)`.
Re-exported on the public `Engine` at `engine.rs:294/300/310/321`.

llama.h:998 — *"Token logits obtained from the last call to
llama_decode()"*. `llama-context.cpp:2121` (`output_reserve`, called
from `decode`) frees and reallocates when `n_outputs` grows:
`buf_output = nullptr; logits.data = nullptr;`.

Both the borrow and the mutation take `&self`, so two shared borrows
coexist and borrowck permits:

```rust
let l = engine.logits(0);            // &self
engine.prefill(&big_tokens, 0, 0)?;  // &self — allowed; frees buf_output
let x = l[0];                        // read of freed memory
```

No `unsafe` at the call site. The crate already states the rule
correctly at `backend.rs:64-66` and the `Decoder` **trait** enforces it
with `&mut self` — the *inherent* methods bypass the trait and void the
guarantee. Weave uses `Engine` directly, so this is a live surface.

Fix: `&mut self` on `decode` / `prefill_inherent` / `memory_*`. They
mutate C-side state regardless.

### 2. `llama_get_*_ith` NULL return never checked

`decoder.rs:449-454, 457-460, 463-466, 469-472` — all four feed the
pointer straight into `slice::from_raw_parts{,_mut}`, no null check.

llama.h:1009 / :1025, verbatim: *"returns NULL for invalid ids."*
`llama-context.cpp:867` and `:891`:

```cpp
} catch (const std::exception & err) {
#ifndef NDEBUG
    GGML_ABORT("fatal error");
#else
    return nullptr;
#endif
}
```

**Build-configuration split (verified in our own CMakeCache, and the
subagent had this backwards — recheck rather than trust):**
`CMAKE_BUILD_TYPE=Debug` → `CMAKE_CXX_FLAGS_DEBUG=-g`, NDEBUG
**undefined** → loud `GGML_ABORT`. `Release` → `-O3 -DNDEBUG` →
**returns nullptr** → we build a slice over NULL. So debug/test builds
abort loudly and release builds take the UB. Tests will not catch this.

The rustdoc at `decoder.rs:447-448` — *"# Panics: If the index is
invalid (panics come from the C side)"* — is wrong in both configs: an
abort is not an unwinding panic, and release does not abort at all.

Most reachable instance is not exotic: `llama-context.cpp:891` throws
`"no embeddings"` when the context was built without `embeddings=true`,
so `Engine::embeddings(0)` on an ordinary generative context is a
null-slice construction on the *first* call.

### 3. `llama_decode` KV-dirty return codes neither distinguished nor reconciled

`decoder.rs:410-415` collapses everything non-`0`/`1` into an opaque
`ErrorCode`. llama.h:950-958 partitions failures:

> `//    0 - success`
> `//    1 - could not find a KV slot for the batch`
> `//    2 - aborted (processed ubatches will remain in the context's memory)`
> `//   -1 - invalid input batch`
> `// < -1 - fatal error (processed ubatches will remain in the context's memory)`
> `// Upon fatal-error or abort ... query the memory state using`
> `// llama_memory_seq_pos_min() and llama_memory_seq_pos_max()`

`1` and `-1` roll the KV back; `2` and `< -1` leave partial ubatches
in the cache. We never call the prescribed reconciliation. `-2` is not
rare — `llama-context.cpp:1772/1785/1803/1814` all return it.

Failure: prefill a 4k chunk, ubatch 3 of 8 fails → `-2`; Session leaves
`pos` at the pre-decode value while the KV holds cells at
`[start_pos, start_pos + 3*n_ubatch)`; a later prefix-cache hit trusts
`pos` and resumes over stale cells → silently wrong logits. Same class
as `blallama_session_state_pollution.md`.

This is the one finding that is a design decision, not a small fix:
`DecodeError` needs to carry KV-dirtiness.

### 4. `prefill_inherent` doesn't bounds-check `n_batch`; overflow aborts

`decoder.rs:433` builds `Batch::new(tokens.len(), 0, 1)` for arbitrary
`tokens.len()`. `llama-context.cpp:1748`:
`GGML_ASSERT(n_tokens_all <= cparams.n_batch);`

`GGML_ASSERT` is **not** NDEBUG-gated (`ggml.h:288` → `GGML_ABORT`), so
this aborts the process in *every* build config. `n_batch()` exists at
`decoder.rs:224` and is never consulted. A caller passing an over-long
slice gets `SIGABRT` from a fn whose signature promises `Result`.

### 5. `token_to_text` / `token_to_score` missing the guard their sibling has

`model.rs:767`, `:791`. `llama-vocab.cpp:3862-3870`:

```cpp
return pimpl->id_to_token.at(id).text.c_str();   // .at() throws
```

`llama_token` is signed; `LLAMA_TOKEN_NULL` is `-1` → huge `size_t` →
`std::out_of_range` unwinding through `extern "C"`.

The guard already exists for `token_to_piece` (`model.rs:38`, added
after the gemma-4 no-EOT incident, with the hazard spelled out in the
comment) and was never replicated to the two siblings, nor to the
`tokens_to_text` / `tokens_to_scores` iterator wrappers.
`model.token_to_text(model.eot())` on a model lacking EOT does it.
Public-API-reachable only; no in-crate caller passes an unbounded id.

### 6. `tokenize()` appends a second EOS when the vocab already added one

`model.rs:648-655` (and the trait impl at `:850-862`).

llama.h:1132 — *"@param add_special Allow to add BOS **and EOS** tokens
if model is configured to do so."* We pass `add_special = add_bos()`
then unconditionally append EOS ourselves. llama.cpp already did it
(`llama-vocab.cpp:3340` SPM, `:3439` BPE):

```cpp
if (add_special && add_eos) { output.push_back(special_eos_id); }
```

`add_bos` / `add_eos` are **independent GGUF KVs**
(`tokenizer.ggml.add_bos_token` / `add_eos_token`), so this fires
per-file, decided by whoever converted the GGUF. When both are set:
`[BOS, …, EOS, EOS]`.

The asymmetry that hides it: with `add_bos=false, add_eos=true` our
append is *correct*, which is presumably why it was written this way.
Relevant to cache work — a stray trailing EOS changes the prompt hash
and silently degrades prefix-cache hit rate rather than announcing
itself. **Open**: whether any GGUF currently in `models/` sets both.

### 7. Two safe mtmd APIs reach UB without `unsafe` at the call site

**(a) `Engine::set_vision` with a foreign-model `Mtmd`** —
`engine.rs:99`, reaching `mtmd.rs:434`. Nothing binds the `decoder`
argument to the model the `Mtmd` was built from, and two quantities are
read from *opposite* objects: `n_embd` from
`llama_get_model(decoder.context)` (`mtmd.rs:471`, sizes the slice at
`:474` and the `embd` stride at `:713`) but the buffer was sized by
mtmd's own `n_embd_out` (`mtmd.cpp:1449`); and `mrope` from the *mtmd*
context (`:494`) while llama.cpp reads `pos` using the *decoder's*
`n_pos_per_embd` (`llama-batch.cpp:713-719`).

Upstream enforces agreement only at construction (`mtmd.cpp:375`
rejects `n_embd_text != n_embd_clip`), which evaporates on reseat.
Not reachable on shipped paths — both auto-load and `load_mmproj`
build from `&self.model` — `set_vision` alone opens it. Cheap fix:
store `*const llama_model` on `Mtmd`, compare at the top of
`eval_media_chunk`, return a typed error.

**(b) `Mtmd::from_path` establishes an unenforced borrow** —
`mtmd.rs:235-266`. `mtmd.cpp:311` caches `llama_model_get_vocab(...)`
and `mtmd_tokenize` dereferences it. `from_path` takes
`model: &LlamaCppModel` but stores no lifetime, so `Mtmd` is
`'static`; dropping the model then tokenizing is a UAF. `Engine`'s
field order (`vision, decoder, model`) makes drop order correct for
engine-owned instances — verified — but the standalone constructor is
public and safe.

**FIXED 2026-07-21 (#54), together with the `LlamaCppDecoder::new`
twin below** — see the resolution note at the end of this memo. The
audit *understated* this one: the shortest path to the UAF needed
neither standalone constructor, because `Engine.model` was a `pub`
field and assignment drops in place.

### 8. ABI claim is enforced against field *addition*, not *reorder*

`backend.rs:8-11` claims "same size, same alignment, **same field
order**" and credits `static_assertions` with verifying it.
Actually asserted (`backend.rs:43-44`): size and align only.

All three fields (`id: i32`, `logit: f32`, `p: f32`) are 4 bytes, so
**any permutation passes both asserts**. If upstream swaps `logit` and
`p`, the crate compiles clean and every sample treats probabilities as
logits. Note the `// TODO: simplify` sitting directly above the struct
at llama.h:206 — upstream is contemplating changes here.

Size+align *do* catch field addition and width changes, which are the
likelier mutations, so this is partial cover, not none. Separately,
`pub type Token = i32` (`backend.rs:20`) has **no** assertion tying it
to `llama_token` at all.

Sites relying on the claim (complete list — no `mem::transmute`
anywhere in the crate):
- `candidates.rs:536-537` — outbound `*mut llama_token_data`
- `candidates.rs:86-91`, `:99-104` — inbound `from_raw_parts{,_mut}`

Fix is ~4 lines: `memoffset::offset_of` per field, plus a size assert
on `Token`.

### 9. `&mut [bool]` constructed over uninitialized malloc memory

`batch.rs:243-250`. llama.h:922-928, verbatim: *"The rest of the
llama_batch members are allocated with size n_tokens / **All members
are left uninitialized**"*. Confirmed `malloc`, not `calloc`
(`llama-batch.cpp`). `Batch::new` does not zero it.

`add_token` increments `n_tokens` *first*, then `logits_mut()[i] = ...`
builds a `&mut [bool]` of length `i+1` spanning a slot that still holds
an arbitrary malloc byte. `bool` has a validity invariant (0 or 1); a
reference to an invalid `bool` is UB regardless of whether it is read.
The three `static_assertions` at `batch.rs:230-232` establish size and
encoding, not byte validity — which is the actual load-bearing
invariant.

Benign under current codegen (write-only path, no niche exploitation);
Miri would flag it. Fix: `ptr.add(i).write(logits as i8)` instead of
materializing the slice, or zero `logits` once in `Batch::new`.

### 10. `ENGINE_COUNT` panic path poisons the mutex and turns every later Drop into an abort

`decoder.rs:149-165`, `:478-482`. The mutex correctly serializes
`backend_init`/`backend_free` and the rollback at `:173-177` is sound —
the defect is the panic path. `numa_strategy` is caller-supplied and
`.try_into().unwrap()` runs *after* `*count += 1` and *while the guard
is held*: the count is left permanently high (backend never freed) and
`ENGINE_COUNT` is **poisoned**, so every later `new` panics at
`.lock().unwrap()` and every later `Drop` panics at `:478` — a panic in
a destructor during unwinding is an immediate abort.

Fix: `unwrap_or_else(|e| e.into_inner())` at both lock sites (the
pattern `log.rs` tests already use at `:187`), and validate
`numa_strategy` before taking the guard.

### 11. Embedding row stride uses `n_embd`, contract says `n_embd_out`

`decoder.rs:465`, `:471` size with `self.embedding_size` (from
`llama_model_n_embd`). The header comment at llama.h:1021 says `n_embd`
but is **stale**; `llama-context.cpp:896`:

```cpp
const uint32_t n_embd_out = model.hparams.n_embd_out();
return embd.data + j*n_embd_out;
```

`n_embd_out` comes from the optional GGUF key `%s.embedding_length_out`
and is unconditionally overridden for `LLM_ARCH_WAVTOKENIZER_DEC`.
There is a dedicated `llama_model_n_embd_out` we don't call. Any model
publishing `embedding_length_out < embedding_length` gives an OOB read
of `(n_embd - n_embd_out)` floats per call. Contract violation
confirmed; whether a model in our fleet triggers it is not.

### 12. Lower-severity, confirmed

- **`meta()` wrong failure check then underflow** — `model.rs:543`,
  `:554`. llama.h:586 says *"-1 on failure"*; we compute
  `required = ret + 1` then test `required < 0`, so a failure gives
  `0`, passes, and `truncate(0usize - 1)` underflows.
  `get_meta_by_key` gets this right (`< 1`), so it's an
  inconsistency. Unreachable today (loop bounded by `meta_count()`).
- **`INT32_MIN` sentinel treated as a length** — `model.rs:699-703`.
  llama.h:1131 documents *"Returns INT32_MIN on overflow"* as distinct
  from the negative-required-size convention; `-i32::MIN` overflows.
  Needs >2^31 tokens, so practically out of reach — but it is exactly
  the "never treat a negative return as a length" case and the header
  names it.
- **`token_to_piece` destroys split codepoints** — `model.rs:44`
  returns `"[Invalid UTF-8]"` for bytes that are merely an *incomplete*
  UTF-8 sequence spanning byte-fallback tokens. Already leaking into
  output: `session/mod.rs:5581`, `:5622` and `tool_choice.rs:1324` all
  `trim_end_matches("[Invalid UTF-8]")`. Real fix is buffering
  incomplete sequences across tokens, above this layer; the
  byte-preserving `token_to_piece_ref` is the primitive to thread
  through. Four public methods also document a `# Panics` that cannot
  fire (`unwrap_or`, not `unwrap`).
- **`Batch::new` ignores allocation failure** — `batch.rs:51-68`.
  `llama_batch_init` null-checks none of its seven `malloc`s.
- **`Batch::add_tokens` assigns every token the same position** —
  `batch.rs:327-342`. Not memory-unsafe, currently dead, but `pub`.
- **`log.rs` trampoline has no `catch_unwind`** — `log.rs:69-90`.
  Since Rust 1.81 `extern "C"` carries an abort-on-unwind shim, so this
  is abort-not-UB, but undocumented: a stray `.unwrap()` in a
  consumer's log sink kills the process with no diagnostic. Also
  SUSPECTED reentrancy deadlock — `GLOBAL_LOG_CALLBACK` is held across
  the user callback, so a closure that logs (or calls
  `set_log_callback`) self-deadlocks on a non-reentrant `Mutex`.
- **`LlamaCppDecoder::new` doesn't tie its lifetime to the model** —
  `decoder.rs:143-147`. Same shape as 7(b); `Engine` is saved only by
  field declaration order, which is load-bearing and unwritten.
  **FIXED 2026-07-21 (#54)** — resolution note at the end of this memo.
- **`moeflux/model.rs:80,83`** — `unsafe impl Send`/`Sync` are no-ops
  (all fields already `Send + Sync`) that would silently mask a future
  `Rc`/`Cell`/raw-pointer addition. Delete both; `engine.rs:32` already
  articulates the better pattern.
- **`decoder.rs:169`** uses `llama_new_context_with_model`, DEPRECATED
  in this header (llama.h:518, use `llama_init_from_model`).

## Checked and found clean (do not re-litigate)

- **`EmbdBatch` hand-assembly — REFUTED as a hazard.** The prior memo
  flagged it as the likeliest bug site; it is the *best* code in the
  audit. `llama.h:240-249` has exactly seven fields; `mtmd.rs:710-718`
  sets all seven as an **exhaustive struct literal** with no
  `..Default::default()`, so an upstream field addition is a *compile
  error* rather than a silently-null pointer. Copy this pattern.
- **The missing `seq_id[n_tokens] = NULL` sentinel is correct** — and
  load-bearing in a second, non-obvious way. `llama_batch_init`
  allocates `n_tokens_alloc + 1` pointers and writes the terminator;
  `EmbdBatch::new` allocates exactly `n_tokens`. That sentinel is read
  by **exactly one** consumer: `llama_batch_free`'s
  `for (i = 0; batch.seq_id[i] != nullptr; ++i)` (`llama-batch.cpp:913`).
  `llama_batch_allocr::init` never indexes past `n_tokens`. So the
  omission is sound *precisely because* the batch is never freed —
  a future refactor that "fixes" this by routing through
  `llama_batch_free` would walk off the end of a Rust `Vec`.
- **mtmd no-free invariant holds on every path** including unwind:
  the `llama_batch` exists only as a temporary from `EmbdBatch::view`,
  consumed by `llama_decode` in the same iteration; `EmbdBatch` has no
  `Drop`; `Bitmap`/`Chunks` free only what they own and are not
  `Clone`.
- **Pixel ownership**: `mtmd.cpp:42-49` `memcpy`s into its own vector.
  We hand over `rgb8().as_ptr()`, read once, never retained. The
  `nx*ny*3` length is guaranteed by `Image::from_rgb8`
  (`backend.rs:226-237`, private fields, rejects mismatch).
- **`llama_batch_init`/`free` pairing in `batch.rs` is exact** on all
  paths including unwind and the `InvalidSequenceLength` rollback;
  capacity is checked before increment; `Batch` is not `Clone`.
- **Ownership/lifetimes in `model.rs` are correct throughout** —
  nothing model-owned is freed, nothing C-owned outlives the model,
  every escaping string is an owned copy or lifetime-bound to `&self`.
  `token_to_text` is the one deliberate borrow and it is correctly
  tied to `&'a self`.
- **Tokenizer NUL handling is correct**: `model.rs:690` passes
  `ptr + len` and llama.cpp reconstructs with `std::string(text, len)`,
  so interior NULs are tokenized as data, never truncated.
- **`seq_rm` return handling is right** — `decoder.rs:598` checks the
  bool on partial truncate; `:607` correctly ignores it on
  whole-sequence removal because llama.h:713 says *"Removing a whole
  sequence never fails"*. Reads like the ignored-return bug it isn't;
  the justification belongs at the site.
- **`n_vocab` row stride is correct** (`model.rs:379` uses
  `llama_vocab_n_tokens`, matching `llama-context.cpp:866`) — checked
  specifically because finding 11 is the same bug class.
- **`candidates.rs` inbound casts are bounds-guarded** by the
  `min`-based `len()`; sort/softmax state is invalidated on every
  mutable exposure.
- **State save/load is exception-safe**: the inner
  `llama_context::state_set_data` catches and returns 0, so nothing
  unwinds across `extern "C"`.
- **`log.rs` drop-then-swap ordering is sound** despite the module doc
  describing it backwards — the old `Box` is dropped under the same
  mutex the trampoline holds while calling `cb`.

## Status (2026-07-20, same session)

Fixed in this pass: **1** (`&mut self` on `decode`/`prefill_inherent`,
which makes the `&self` on `logits`/`embeddings` sound), **2** (null
checks + accurate `# Panics` on all four accessors), **4**
(`BatchTooLarge` pre-check), **5** (range guards on `token_to_text` /
`token_to_score`, now `Option`-returning), **6** (delegate BOS/EOS to
`add_special`), **8** (per-field `offset_of` asserts + `Token` size /
align), **9** (raw write instead of a `&mut [bool]` over uninit),
**10** (poison-safe `engine_count()`; note the finding's "NUMA
`try_into().unwrap()` panics" premise was *false* — `ggml_numa_strategy`
is `c_uint`, so it was a `u32 → u32` infallible conversion — but the
poisoning concern it rode in on is real and now fixed, and the spurious
conversion was simplified away), **11** (`n_embd_out` row stride), **12**'s `meta()` underflow,
`INT32_MIN` sentinel, `moeflux` no-op `unsafe impl`s, the deprecated
`llama_new_context_with_model`, and the `log.rs` unwind guard.

Partially fixed: **3** — `DecodeError` now distinguishes `Aborted` /
`Fatal` / `InvalidBatch` and exposes `kv_dirty()`, but nothing acts on
it yet. Reconciliation is [#52](https://github.com/mdegans/drama_llama/issues/52).

Also fixed: **7a**, via a stored `*const llama_model` on `Mtmd`
compared at the top of `eval_media_chunk` (`ModelMismatch`).

Deferred to issues: [#52](https://github.com/mdegans/drama_llama/issues/52)
(KV-dirty reconciliation), [#54](https://github.com/mdegans/drama_llama/issues/54)
(lifetime coupling — 7b and the `LlamaCppDecoder::new` twin),
[#55](https://github.com/mdegans/drama_llama/issues/55) (split-codepoint
loss; wants doing with the streaming work).

**Follow-up pass (2026-07-20, later session) — [#56](https://github.com/mdegans/drama_llama/issues/56)
CLOSED:** all three leftovers fixed. Log reentrancy: `LogFn` is now an
`Arc`; `trampoline` clones it out and drops the guard before the call,
so a sink that logs or touches `set/clear_log_callback` no longer
deadlocks (regression test: a sink that clears itself mid-callback).
`Batch::new` rejects OOM (null `token`/`embd`/`pos`/`n_seq_id`/`logits`
when `capacity > 0`; `capacity == 0` exempt — `malloc(0)` may return
NULL legitimately; a failed `seq_id` alloc crashes inside
`llama_batch_init` at the terminator write and never reaches Rust).
`Batch::add_tokens` deleted (dead, `pub`, wrote every token at one
position). Same session closed [#57](https://github.com/mdegans/drama_llama/issues/57)
(replayable test seeds: `tests/common/mod.rs::test_seed()`, env-seeded +
printed, wired into the round-trip fuzzer — the seed *replayed* #53's
failure deterministically, which is the proof it works).

**Follow-up pass (2026-07-21) — [#54](https://github.com/mdegans/drama_llama/issues/54)
CLOSED, finding 7(b) + the `LlamaCppDecoder::new` twin.**
`LlamaCppModel` is now a refcounted handle — `LlamaCppModel(Arc<ModelInner>)`,
where the private `ModelInner` owns the `*mut llama_model` and frees it
on drop. `LlamaCppDecoder` and `Mtmd` each store a clone, so a
`llama_context` structurally cannot outlive the weights it references.
Notes for whoever audits next:

- **The `pub model` field was the real hole**, not the constructors.
  `engine.model = other` dropped the old model in place while the
  decoder's context still held `const llama_model &` to it — two lines,
  all `pub`, no `unsafe`. Now `pub(crate)` + `Engine::model()`. Post-fix
  the reason to keep it private is *coherence*, not safety: swapping the
  handle would leave a tokenizer disagreeing with a KV cache built from
  the previous weights (silent garbage, worse than a crash).
- **How it got there, per Mike (2026-07-21): co-owning the decoder and
  the model so their lifetimes coincide was the entire original purpose
  of `Engine`.** That rationale was never written down, so it decayed —
  the field went `pub` at some later point and nothing flagged it,
  because nothing recorded what the type was *for*. The general lesson
  is the one worth carrying: an invariant that lives only in a
  reviewer's head is one refactor from gone, and the failure is silent
  by construction. The purpose is now stated on the `Engine` struct doc
  itself (`src/engine.rs`), where a future editor reaching for `pub`
  will hit it.
- **`Sync` was re-verified, not assumed**, because `Clone` makes shared
  access reachable rather than hypothetical. `llama_context` stores
  `const llama_model &` (llama-context.h:276); no `mutable` members in
  llama-model.h / llama-vocab.h; `llama_init_from_model`'s non-const
  `llama_model *` is vestigial (body is reads-only before binding to
  the const ref). **The one model-mutating context call upstream is
  `llama_opt_init`** (llama-context.cpp:3245, finetune): it writes
  `hparams.n_ctx_train` and `llama_set_param`s every weight tensor.
  Binding it into the safe surface would invalidate `unsafe impl Sync`.
  That named exclusion now lives in the safety comment.
- **The `Sync` claim is about llama.cpp, not about us, and we bump
  `llama-cpp-sys` often — so it can rot with nothing going red.**
  Warning at the bump site (`Cargo.toml`, on the dependency line, since
  that is the line your hand is on); CI tripwire tracked upstream at
  **mdegans/llama-cpp-sys#6**, which is where the submodule bump
  actually happens and where 3-OS CI already exists. Until that lands
  the comment is the only defence, which is weak — treat a bump as a
  prompt to re-read the SAFETY block, not a routine version edit.
- **Not a "never bind finetuning" rule** (Mike wants to try it
  eventually). The `Arc` is what makes it *tractable*: `Arc::get_mut`
  yields `Some` only when exactly one handle exists — i.e. no context
  and no projector holds the model — so a future
  `LlamaCppModel::try_get_mut()` hands out mutable access precisely
  when it is sound. Mutation must be gated on exclusive access; it
  need not be forbidden.
- **Drop order is no longer load-bearing anywhere.** Both `Drop` impls
  (`LlamaCppDecoder`, `Mtmd`) run to completion before their fields
  drop, so the C handle is freed and only then is the model handle
  released. `Engine`'s field-order comment was rewritten to say so —
  leaving a comment claiming a load-bearing order that isn't is its own
  trap.
- **Upside, not just a fix:** cloning the handle gives N contexts over
  one copy of the weights (on Metal, one unified-memory allocation
  instead of N). `two_decoders_share_one_model` covers it.
- **Test discipline:** `decoder_outlives_model_handle` asserts
  `model.into_raw().is_none()` — a *deterministic* proof the decoder
  holds a strong ref. Do not rely on the prefill-after-drop half alone:
  a UAF is UB and is not guaranteed to manifest as a crash you can
  assert on.
- API breaks taken (zero in-tree callers): `into_raw` → `Option`,
  `as_ptr_mut` → `pub unsafe fn (&self)` (a `&mut` on a shared handle
  would be a lie), `context_ptr_mut` → `&mut self`.
- Also in this pass: `webchat` and `egui` joined the justfile feature
  sets. They were omitted as "UI glue", which meant a library API change
  could break the bins with no gate noticing — this change did exactly
  that, and only `cargo check --all-targets` caught it.

Still open: #52 (design — decode hook + `Session` reconciliation),
#55 (streaming session).

### Methodology note — do not repeat this mistake

A model-backed test (`complete_text_round_trips_through_parse_and_render`)
failed on the changed tree, passed on `HEAD`, and passed again when the
`tokenize` change was reverted. Three data points, all pointing at a
regression that **did not exist**: the test fails ~1 run in 3 on its
own, because `Session` picks a random seed when `with_seed` is unset.

Two lessons. First, on a suite with model-backed tests, a single run
per side of a bisect carries almost no information — the token ids
were later compared directly and proved identical at `add_special`
true and false (this GGUF has `add_bos=false` and no `add_eos` key),
which is the evidence that should have been gathered first. Second,
the flake is *worth keeping*: the random seed is fuzzing emission
shapes, and it surfaced a genuine parse/render asymmetry
([#53](https://github.com/mdegans/drama_llama/issues/53)) — a tool call
emitted inside an unclosed `<think>` block does not round-trip.
Seeding the test would have hidden a real bug.

Also: the parallel `--test-threads=1` rule in `validation_runbook.md`
is real. Running the ignored sweep in parallel produced 5 bogus
`-3` failures before the runbook was consulted.

## Verdict and fix order

Ownership — the thing this audit was chartered to check — is in good
shape. Nothing frees what llama.cpp owns, nothing owned by us leaks,
and the mtmd batch work is exemplary. **The defects are concentrated in
API shape, not resource handling**: `&self` where `&mut self` was
meant, unchecked NULL/negative returns, and doc comments asserting
guarantees nothing enforces.

Recommended order:
1. **#1** (`&mut self`) and **#2** (null checks) — can corrupt a
   running process; #1 is trippable by a downstream consumer with no
   `unsafe`. Both small.
2. **#5**, **#10**, **#4** — abort-instead-of-`Err` paths; small.
3. **#6** — silent correctness bug with cache-hit-rate consequences.
   Check `models/` for a GGUF setting both flags.
4. **#8** (offset asserts), **#7a** (model-identity check), **#9**
   (raw write) — cheap hardening.
5. **#3** — genuine design work: `DecodeError` carrying KV-dirtiness.

`decoder.rs` carries 42 `unsafe` blocks and **zero** `// SAFETY`
comments, which is the structural reason #1 could drift from the rule
`backend.rs:64` already states correctly.

## Addendum: 0.8.0 pre-publish delta audit (2026-07-23)

A second pass over everything unsafe-bearing that changed since this
audit (delta from `54b8d15`): `options.rs` (new), `log.rs` (rewritten),
`engine.rs` (from_path_with refactor), plus a from-scratch verification
of the mtmd wrapper's load-bearing C-side facts against the vendored
source at 0.8.1 parity. **Verdict: sound; published as-is.**

Verified and now settled (extend the do-not-re-litigate list):

- All prior-audit fixes confirmed present, not assumed (decode/prefill
  `&mut self`; four `llama_get_*_ith` null checks; Mtmd model-identity
  check; the #54 Arc keepalive; moeflux has zero unsafe).
- mtmd C-side facts pinned: NULL-bitmap ctor is memcpy-safe; encoder
  output slice length exactly matches the C buffer resize; M-RoPE
  `rel` buffer count and (t, y, x, z) plane order match
  `mtmd-helper.cpp`; `mtmd_tokenize`/`mtmd_encode_chunk`/
  `mtmd_init_from_file` are exception-caught C-side.
- `log.rs` rewrite: catch_unwind around the consumer closure, Arc
  cloned out and mutex released before the call (reentrancy-safe),
  clear/swap is drop-after-swap — no callback-outlives-data hazard.

Open, minor, deliberately not release blockers:

- **F1**: `mtmd_input_chunks_init` / `mtmd_bitmap_init` are bare
  `new`/resize with no try/catch upstream (`mtmd.cpp:1707-1712,
  1782-1784`), so C++ `bad_alloc` would unwind across `extern "C"`
  into Rust — OOM-only, upstream defect; the siblings all catch.
  **File upstream** (llama.cpp), optionally mirror in llama-cpp-sys#
  tracker.
- **F2**: `CausalAttnGuard` restores `true`, not the prior value —
  wrong only if a non-causal (embeddings) context ever runs image
  prefill; nothing in-crate does today.
- **F3**: audio chunks on an M-RoPE model fail closed with a
  misleading `NoMediaChunk` error (upstream would take `mrope_1d`).
  Matters only if a `Block` audio variant ever appears.
- **F4**: `set_log_callback` vs `set_log_callback_raw`/`clear` can
  interleave to misroute logs (Rust slot Some while raw C callback
  registered, or trampoline against empty slot). No UAF in any
  interleaving; purely logs-may-be-lost on a process-global race.
- **F5**: `LlamaCppOptions::numa` is a serde-settable raw `u32` fed
  to `llama_numa_init` unvalidated — defined-but-meaningless on
  out-of-range, not UB. CLI correctly skips the arg.
- **F6**: `start_pos as llama_pos` wraps past `i32::MAX`; unreachable
  with any real `n_ctx`.
