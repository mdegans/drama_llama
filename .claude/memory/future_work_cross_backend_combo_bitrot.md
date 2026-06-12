# Future work: moeflux+llama-cpp combo bit-rot (cross_backend.rs)

**Found 2026-06-12** during v0.8.0 release prep. Pre-existing, not
caused by the misanthropic alpha.2 port.

`cargo check --features "moeflux-model-qwen3-6-35b-a3b,llama-cpp"
--all-targets` fails with ~36 errors. Two classes:

1. **`tests/cross_backend.rs` predates the `Session<B: Backend>` /
   `Engine<B>` split** — it's still written against the old
   two-parameter `Engine<D, M>` (D: Decoder, M: Model) shape. The
   whole capture loop needs rewriting against `Engine<B>`. This file
   has been uncompilable since the Phase 1–4 backend split landed;
   nobody noticed because the moeflux+llama-cpp *combination* isn't in
   any default, CI, or docs.rs feature set.

2. **`from_path` ambiguity** (~24 sites): with both backends enabled,
   unqualified `Engine::from_path` / `Session::from_path` in
   model-gated tests is ambiguous between the
   `Engine<LlamaCppBackend>` and `Engine<MoefluxBackend>` inherent
   impls. Fix is mechanical: `Engine::<LlamaCppBackend>::from_path`
   etc. Sites: src/chat_template.rs, src/session/mod.rs,
   src/tool_choice.rs, tests/session.rs, tests/hash_cache_smoke.rs.

Each feature alone is green (all targets, zero warnings as of
v0.8.0). Fix both classes in one pass when the cross-backend A/B
coherence test becomes load-bearing again.
