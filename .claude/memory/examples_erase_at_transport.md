# The examples erase at `Transport`, not at `Session` (#48)

Landed 2026-07-22 on `v0.8.0` (`b8a6b23`, `f557ad3`). Read this before
touching `examples/utils/args.rs` or wondering why four examples still
name `LlamaCpp*`.

## The decision, and why it isn't what #48 proposed

Issue #48 offered two options: make every example generic over `B` (like
`bin/blallama/blallama.rs`), or add a dyn-compatible `Session` wrapper
trait and pass `Box<dyn Session>`.

Neither was needed. An inventory of all 18 examples found **8 already
drove `misanthropic::Transport` and never touched `Session` after
construction**, and 4 more converted cheaply. `Transport` is already
dyn-compatible and is the *entire* interface for driving a completion, so
backend polymorphism above the session needs no drama_llama trait at all.
The erasure boundary is a trait misanthropic owns and maintains.

Generic-over-`B` was rejected on a specific ground, not taste: it does not
avoid the dispatch, it *duplicates* it. blallama still ends in
`match args.backend { LlamaCpp => run::<LlamaCppBackend>(…) }`; making the
examples generic would put that match in every one of them — ~13 copies
instead of one — and `fn ask<B: Backend>(…) where Session<B>: Send` is
noise for someone reading an example to learn the crate.

## What `LocalTransport` is for

Erasing to a bare `dyn Transport` would lose
`SessionTransport::scan_text_for_specials`, which has no API-side
counterpart: a remote endpoint cannot be prompt-injected with *its own*
framing tokens, and a local one can. So it rides along as
`LocalTransport`'s single method (`src/session/transport.rs`).

Both prompt-type supertraits are listed (`Transport<Prompt>` and
`Transport<CachedPrompt>`) because `SessionTransport` serves both and
`examples/prompt_caching.rs` uses the second. A trait object implements
its supertraits, so `Arc<dyn LocalTransport>` is simultaneously a valid
`T: Transport` for anything generic over one **and** carries the scan —
one erased type, not two. The `assert_impl_all!` in that file pins exactly
this and is the load-bearing assertion for the whole design.

**Do not push the boundary lower.** dyn `Model` would put a virtual call
on `tokenize` / `token_to_piece_ref`, which are per-token in the sampling
loop. `Transport` sits above the hot path; "dispatch cost is nil" is true
there and false one layer down.

> **Superseded in part (2026-07-22), see
> [`per_backend_load_options.md`](per_backend_load_options.md).** The
> constructor zoo this memo describes is gone: `from_path_sync` is now
> `FromPath::from_path`, and the five `from_path_*` variants collapsed
> into `FromPath::from_path_with(path, B::Options)`. Where this file
> says `from_path_sync`, read `from_path`. The erasure design below —
> `Arc<dyn LocalTransport>`, one `match`, the flags-vs-builder split —
> is unchanged and still current.

## Traps found the hard way

- **`from_path_with_cache_slots(path, n_ctx, 1)` is NOT
  `from_path_with_n_ctx(path, n_ctx)`.** The multi-slot constructor also
  sets `kv_unified = true`. Routing every session through it would have
  silently changed KV behaviour for the eight examples that never asked
  for slots.
  **Now expressed as a field, not two constructors:**
  `LlamaCppOptions::cache_slots` is `None` vs `Some(1)`, and `Some(1)`
  still flips `kv_unified` — deliberately, since asking for one slot is
  asking for the unified pool. Pinned by
  `one_cache_slot_still_unifies_the_kv_cache` in
  `src/llama_cpp/options.rs`, so the branch no longer has to be
  remembered by hand.
- **`quiet()` is llama.cpp-only and that is correct**, not an oversight.
  It calls `silence_logs()`, which muzzles a C library writing to stderr,
  process-globally. moeflux logs through the `log` crate, which
  `env_logger` already governs, so a backend-agnostic `quiet` would be a
  no-op on half its backends. It lives in the llama-cpp arm of `build`,
  not the shared finisher. (Also worth knowing: it only silences
  *post-load* spew — every caller applies it after construction, so model
  load logs print regardless. Pre-existing, all the way back. See the
  future-work note below.)

  **Future work (Mike, 2026-07-22): `Backend::quiet()` with a default
  no-op body.** The real defect is ordering, not placement: llama.cpp is
  loudest during load, and `.quiet()` always runs after it. `Backend` is
  a type-level bundle with no `self`, so an associated fn fits; llama.cpp
  implements it with `silence_logs()`, moeflux keeps the default (its
  stray `eprintln!`s want moving to `tracing`/`log` regardless, after
  which `RUST_LOG` governs them and the no-op is the *correct* impl, not
  a gap). Call it from the caller *before* construction — one line at the
  top of each arm of `TransportBuilder::build`, which is already the only
  place naming a concrete backend. Deliberately do **not** make
  `FromPath` generic over `B` to do this: the per-backend constructors
  do not line up (llama.cpp has five, moeflux one), so a generic
  `FromPath` forces either a lowest-common-denominator signature or an
  options struct, and it would decide quiet-ness for callers who
  legitimately want load logs when a model won't load.
- **`Box<T>` cannot take a blanket trait impl; `Arc<T>` can.** `Box` is
  `#[fundamental]`, so downstream `impl Trait for Box<TheirType>` is legal
  and a blanket `Box<T>` impl collides with it — breaking, not additive.
  `Arc` is not fundamental, so the orphan rule already forbids the
  conflicting downstream impl. Verified empirically:
  `impl ForeignTrait for Arc<Local>` is rejected with E0117 while the same
  over `Box<Local>` compiles. This is why misanthropic#140 ships `Arc`
  only. Generalizes to any smart-pointer blanket impl.

## Flags vs. builder arguments

Across all 18 examples the total session config outside `CommonArgs` was
six calls. Two evaporated (`quiet` was already universal;
`with_prefix_cache` is forced on by `SessionTransport::new`). The rest
split:

- **User preference** → a flag, one global default. No example needed a
  *different* default, so `CommonArgs` needs no per-example defaults
  mechanism (a `CommonArgs<D: ArgDefaults>` ZST-parameter trick was
  considered and is unnecessary — revisit only if two examples ever want
  different *displayed* defaults, which `Option<T>` + `unwrap_or` cannot
  give you).
- **Example requirement** → a `TransportBuilder` argument, in code,
  deliberately not a flag. `strawberry` must echo the digit from its tool
  result, so `--repetition` would let a user silently break the demo and
  get a wrong letter count from the example whose whole job is a right
  one.

## The four that stay backend-concrete

`whoami`, `dump_template`, `inspect_prompt` sit *below* the erasure
boundary (raw `Engine` / `CandidatePredictor` / `LlamaCppModel`).
`whodunit` is different: it exists to demonstrate
`Session::complete_stream`, and `Transport` has **no streaming method** —
converting it to `send` would delete the example. Revisit only if
`Transport` grows a `stream`; reconciling local `Block` streaming with SSE
deltas is its own design project.

Each of the four now says so in its module docs, because the low-level
examples were once ported away wholesale and the crate was left with
nothing demonstrating that layer.

## Verified (2026-07-22)

`just check`, 462 unignored + 119/119 ignored model tests green.

- `strawberry` (llama.cpp) — correct letter count, so the repetition knob
  survived the move from example body into the builder.
- `unhelpful` (llama.cpp, blocking bridge) — `cache read: 84`, so
  `SessionTransport::new` really does enable the prefix cache now that
  the example no longer sets it. A zero here would have been the council
  zero-cache-reads bug again.
- `neologism --backend moeflux` against `qwen3-6-a3b` — coherent output.
  This is the point of the whole issue. (a3b deliberately: cogito-v2-671b
  is the blocked one at ~12 s/token, a17b has the mmap working-set
  problem.) Building both backends together also compiles clean, which is
  what would catch the `E0034` bare-`Session::from_path` hazard.
- `--backend moeflux --n-ctx 4096` warns, then fails with a typed error.

The **`Arc<T>: Transport` blanket impl from misanthropic#140 is
runtime-verified**, not merely compile-verified: `transport.send(..)` on
an `Arc<dyn LocalTransport>` resolves at the `Arc` level (shallower than
the deref to `dyn LocalTransport`), and that is the path strawberry and
neologism took on both backends. `BlockingTransport::send` is the
exception — it derefs explicitly (`&*self.inner`), so it does *not*
exercise the blanket.

**Not verified: `swarm`.** It is the only example driving misanthropic's
`Chat` over the erased `Arc`. Two blockers, both incidental: rustyline
rejects a pipe with `Errno(ENOTTY)` (a pty via
`script -q /dev/null` gets past it), and `bee`/`moth` then die because
`DockerSandbox`'s default image is
`concat!("mdegans/misan-bashd:", env!("CARGO_PKG_VERSION"))` — so the
misanthropic version bump retargeted it to a tag not yet on Docker Hub.
With two seats dead the beat never dispatched and the run stalled.
Residual risk is small: `Chat` only calls `Transport::send` on its `T`,
which is the same call the verified examples make. Remedy if it needs
running before the image publishes: `just build-bashd` in misanthropic
(note that pins a *local* image at that tag, shadowing the published one
until pulled).
