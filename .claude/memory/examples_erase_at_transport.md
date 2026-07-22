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

## Traps found the hard way

- **`from_path_with_cache_slots(path, n_ctx, 1)` is NOT
  `from_path_with_n_ctx(path, n_ctx)`.** The multi-slot constructor also
  sets `kv_unified = true` (`src/llama_cpp/engine.rs:167`). Routing every
  session through it would have silently changed KV behaviour for the
  eight examples that never asked for slots. `TransportBuilder::build`
  branches on slot count for this reason — leave the branch alone.
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

## Verified

`just check`, 462 unignored + 119 ignored model tests green.
`strawberry` returns the correct letter count (proves the repetition knob
survived the move); `unhelpful` reports `cache read: 84` (proves
`SessionTransport::new` enables the prefix cache now that the example no
longer does). `swarm` is the only example driving `Chat` over the erased
`Arc`, and it is rustyline-interactive, so it was not run — the bound is
compile-proven and the forwarding is unit-tested upstream.
