# Future work: examples broken after Session<B: Backend> landing

## Status

`cargo build --features "webchat,cli,stats,toml,serde,egui" --examples`
fails on `strawberry` (and likely others that hold a `Session` in a
closure or local binding) because Phase 4 of the v0.8.0 backend split
made `Session` generic over `B: Backend` (`src/session/mod.rs:382`).

Pre-existing as of 2026-05-11 (confirmed against `v0.8.0` HEAD
`4ed4fa1`, before any work in this session). Not caused by the
`tool_choice` per-tool-alternatives refactor.

## Repro

```sh
cargo build --features "webchat,cli,stats,toml,serde,egui" \
  --example strawberry
```

```
error[E0107]: missing generics for struct `Session`
   --> examples/strawberry.rs:120:50
    |
120 |     let load_session = |path: PathBuf| -> Result<Session, _> {
    |                                                  ^^^^^^^ expected 1 generic argument
```

`inspect_prompt` also fails (E0603: `chat_template` module is private
since 0.8). Likely the broader set of `examples/*.rs` files is stale.

## Fix sketch

Either:

1. **Pin the concrete backend.** `Session<LlamaCppBackend>` (or
   whatever the type-alias ended up named) in the closure annotation,
   plus a matching turbofish on `load_session_inner`. Cheapest fix;
   keeps the example readable.
2. **Type-alias.** Add `pub type DefaultSession = Session<Llama…>` in
   the crate root and consume that from examples. Spares examples the
   backend-noise.

For `inspect_prompt`'s `chat_template` privacy: either re-export the
needed types at the crate root (already done for `ChatTemplate` etc.)
or drop the example's reliance on the private module path.

## Don't touch as part of grammar / tool_choice work

The fix is mechanical but it pulls in backend-shape decisions that
belong with whoever owns the v0.8.0 example-cleanup pass, not the
grammar-bug fix this memory was filed alongside.
