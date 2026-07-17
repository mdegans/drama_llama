# Transport upstream + Chat promotion + drama_llama examples (swarm flagship)

## Context

drama_llama's examples should mirror misanthropic's — the crates share prompt/message/
tool/response types wholesale (re-exported in src/prompt.rs; `strawberry.rs` and
`whodunit.rs` are already ports). The highest-value port is `swarm.rs` (five agents,
mail tool, Docker sandboxes), which rides the examples-only `Chat` driver. Making that
portable executes the already-decided upstream design:

- **misanthropic #126**: `Transport` trait — trait-level `Transport<P = Prompt>`,
  `send(&self, &P) -> Result<response::Message, Self::Error>`,
  `type Error: std::error::Error + Send + Sync + 'static`. Shape extended per Mike
  (this session) with agentkit `Inference`'s members: `send_batch` default fan-out,
  `models()`, `quirks()` (Quirks moves upstream — Chat's breakpoint placement is
  quirk-dependent), `max_concurrency()` (default 1). agentkit's `Error`/`RetryAfter`
  stay downstream. Method-level generics rejected: never dyn-compatible.
- **misanthropic #104**: promote `Chat` into the crate, transport- and prompt-generic.
- **misanthropic #134**: `response::Message` lost its construction path to a recent
  `non_exhaustive` change — fix with a **builder** (Mike's call), sized to blallama's
  need (= Session's `empty_response_message`/`make_usage`).
- Swarm on one Session thrashes the single-slot prefix cache — accepted for now;
  the cache rework is a follow-up session (launching pad at bottom).
- Sequential inference (one GPU, no batching) accepted; `max_concurrency` = 1.

## Phase 1 — misanthropic PR (branch `feat/transport` off main; closes #126, #104, #134)

Repo: ~/Projects/misanthropic (crate at misanthropic/). Pull main first (currently
current, clean, v1.0.0-alpha.8, edition 2024). PR from this account; CI runs directly;
Mike merges/tags → publish.

1. **`src/transport.rs` (new, ungated)** — `Quirks` moved verbatim from agentkit
   (reactor/inference.rs:17-33; 5 bools, serde, non_exhaustive; doc-links retargeted)
   + `Transport<P = Prompt>: Send + Sync` per the settled shape, `#[async_trait]`
   (crate convention). `send_batch` default carries agentkit's eager-futures HRTB
   workaround (backend.rs:57-61 comment included). `EndpointVariant` +
   `From<EndpointVariant> for Quirks` stay in agentkit.
2. **`impl Transport for Client` + `impl Transport<CachedPrompt> for Client`**
   (`#[cfg(feature = "client")]`, same file): `send` = `self.message(prompt)` (&Prompt
   is Serialize), `models` = `Client::models` (client.rs:383), quirks/concurrency =
   defaults. `static_assertions` pinning Client: Send+Sync+Clone and
   client::Error: Error+Send+Sync.
3. **#134 builder** in src/response/message.rs, hand-rolled (~70 lines; no
   derive_builder dep): `Message::builder(model, inner: AssistantMessage) -> Builder`
   with `id/stop_reason/stop_sequence/stop_details/usage/container` setters (all
   `impl Into<Option<..>>` where the field is optional; usage takes
   `impl Into<Usage>` so TokenCounts works), infallible `build()`. Defaults: fresh
   UUID id, Kind::Message, Usage::default(). Plus `TokenCounts::new(input, output)`.
   Usage needs no builder (From<TokenCounts> + pub fields). Broader non_exhaustive
   audit deferred — note in PR body.
4. **Promote `cache_windowed_with` from `CachedPrompt` to `Prompt`** (algorithm at
   cached.rs:311-368; CachedPrompt delegates). Needed because the quirk-aware
   per-assistant-turn marking (below) must be budget-aware on plain `Prompt`
   (naive `.cache()` every turn blows the 4-marker budget).
   **DECIDED (Mike): Chat is Prompt-only** — no `Seat` trait, no
   `CachedPrompt::seat`/`drop_tail`, no `tool_lifecycle_mut` escape hatch.
   Rationale: Chat owns the prompt, so cache discipline is Chat's job;
   `CachedPrompt`'s append-only typing is for unassisted developers. Chat's own
   mutations stay disciplined (seat + documented tail-merge + paused-turn pop),
   no arbitrary-mutation API exposed. Prompt-genericity for Chat is descoped —
   comment on #104 rather than auto-closing it; Mike decides its disposition.
5. **`src/chat.rs` (new), feature `chat = ["log"]`** — promoted from
   examples/utils/chat.rs (486 lines). Generic `Chat<State, T: Transport>` over
   concrete `Prompt`. `tokio::select!` (line 246) →
   `futures::select!` + fuse/pin_mut (Notifications is a futures mpsc,
   mailbox.rs:23) so chat needs neither tokio nor client; fallback if the borrow
   checker fights: `chat = ["log", "tokio/macros"]`. `BudgetPolicy`, `BoxError`,
   `DEFAULT_MAX_TOOL_CALLS` move with it.
   **Quirk-aware caching (new)**: opt-in `.cache(CacheControl)` builder knob;
   resolved once from `transport.quirks()`: `cache_markers_ignored` → place
   nothing; canonical → `set_auto_cache` once; `breakpoint_after_assistant` →
   `cache_windowed_with(1, cc)` after each seated assistant turn (budget-aware,
   marks the end-of-assistant render blallama hashes on).
6. **Examples switch to crate Chat**: delete examples/utils/chat.rs; re-export shim
   in utils/mod.rs (`pub use misanthropic::chat::...`); ChatArgs gains generics;
   audit [[example]] blocks for `required-features += ["chat"]`; add missing swarm
   [[example]] block. Dev self-dep gains "chat".
7. **Release chore**: workspace version → 1.0.0-alpha.9, CHANGELOG.
8. **Tests** (missing_docs = deny throughout): mock `Script` transport (responses
   built via the new builder — exercises #134 for free, quirks configurable);
   send_batch order + peak-in-flight ≤ max_concurrency; Chat: one-beat end_turn,
   tool round-trip, both BudgetPolicy paths leave prompt wire-legal, paused-turn
   drop, system buffering, and the three quirk placement behaviors; builder
   field-map + serde round-trip vs deserialize path.
9. **Local gate**: fmt, clippy --all-features, test --all-features, test --features
   chat, check --no-default-features{,+chat} (CI's per-feature leg keeps defaults on,
   so chat⊥client is only proven locally).

Commits: (1) transport+Quirks, (2) Client impls, (3) response builder,
(4) cache_windowed_with promotion to Prompt, (5) chat promotion, (6) examples
refactor, (7) release chore. PR body: Closes #126, #134; advances #104
(prompt-genericity descoped per maintainer decision — comment left on the issue)
+ deferred-items note (broader non_exhaustive audit; agentkit supertrait later).

## Phase 2 — drama_llama `SessionTransport` (branch off v0.8.0)

1. `chore(deps)`: misanthropic → 1.0.0-alpha.9 (deps keep default-features=false +
   partial-eq — Transport/Quirks/builder are ungated; dev-deps add "chat").
2. `refactor(session)`: replace `empty_response_message` (session/mod.rs:4143-4162,
   TODO(upstream)) with `Message::builder(...)`; `make_usage` (:2950) simplifies to
   `TokenCounts::new` + cache-field assigns + `.into()`.
3. `feat(session)`: **`src/session/transport.rs`**, `#[cfg(feature = "tokio")]`
   (existing feature = tokio/full + async-trait; no new feature).
   `#[derive(Clone)] SessionTransport<B: Backend>` =
   `Arc<tokio::sync::Mutex<Session<B>>>` + constructor-time `ModelInfo` snapshot
   (models() must not touch the lock). Session is **Send, not Sync** (Decoder is
   Send-only by design; backend.rs:54, llama_cpp/decoder.rs:207) — pattern:
   `self.session.clone().lock_owned().await` → move the OwnedMutexGuard into
   `tokio::task::spawn_blocking` → `guard.complete_response(&prompt)`; JoinError
   flows through the existing `SessionError::JoinError` variant. `Transport<Prompt>`
   and `Transport<CachedPrompt>` (deref-clone the inner prompt) impls;
   `type Error = SessionError` (thiserror, Send+Sync — assert). quirks() = blallama
   profile: `breakpoint_after_assistant = true`, `output_config_cache_safe = true`
   (mutate a default(); Quirks is non_exhaustive). max_concurrency stays default 1.
   Tests: quirks snapshot (no model); model-backed send populates id/model/usage;
   `tokio::join!` two sends complete serially without deadlock.

## Phase 3 — drama_llama example ports

1. **examples/utils/** (new): copy log_init, spawn_readline_loop, Printer,
   CommonArgs/ChatArgs from misanthropic's utils — drop api_key/key.rs; `--model`
   defaults to `env!("CARGO_MANIFEST_DIR")/models/model.gguf`; re-export crate Chat.
2. **Port order** (one commit each, mechanical recipe: Client::new(api_key) →
   `LlamaCppSession::from_path_sync` + `SessionTransport::new`): neologism →
   few_shot_triage → structured_commit_classifier → vote_intent →
   mid_conversation_system → prompt_caching → python → **swarm**.
   prompt_caching becomes the `breakpoint_after_assistant` showcase via Chat's
   `.cache(...)` knob.
3. **swarm.rs**: five Chat loops over clones of one SessionTransport; mutex +
   max_concurrency=1 serialize inference. Keep mail tool + both DockerSandboxes.
   Known: prefix cache thrashes every agent switch (full re-prefill) — the
   follow-up session's benchmark, not a bug in this phase.
4. **Cargo.toml**: [[example]] blocks with doc-scrape-examples = true,
   required-features = ["tokio", "cli"] (+ "json-schema" where #[tool] needs
   schemars). misanthropic example-only features ("chat", "derive", "markdown";
   swarm: "bash", "bash-container") go on the dev-dependency — note: pulls
   misanthropic's tokio/rustls into every `cargo test` build (feature unification
   on dev-deps; the existing client/rustls-tls dev-dep re-enable is the precedent).
5. Verification: `cargo build --examples` with features; run each example against
   models/model.gguf — **GPU runs are Mike's: output the commands, he runs them**
   (feedback_gpu_launch_from_claude_code).

## Phase 4 (later sessions — NOT this plan)

- **Cache rework** (launching pad below).
- **agentkit**: rebase `Inference` onto `misanthropic::Transport` supertrait; blast
  radius mapped: trait def + ~6 bound clusters in reactor/mod.rs, agent/mod.rs Quirks
  surface, anthropic.rs, 8 test impls. `Inference` keeps `RetryAfter` via
  `Transport<Error: RetryAfter>`-style bounds.

## Known risks / notes

- drama_llama MSRV: Chat::run's `AsyncFnMut` bound needs Rust ≥ 1.85 — verify
  toolchain before Phase 3 (certainly fine, but check).
- `ModelInfo` is struct-literal constructible today (not non_exhaustive) — Phase 2
  relies on it; same fragility class as #134, noted in PR as future audit.
- misanthropic remotes: local main tracks remote `main/main` (duplicate remote of
  origin); pull explicitly before branching.
- Weave is path-pointed at drama_llama — Phase 2/3 changes are additive
  (new module + examples), no Engine surface touched.

## Cache-rework launching pad (follow-up session)

Current mechanics (verified 2026-07-17): `PrefixCache` is single-slot,
last-prompt-wins, single KV sequence (seq 0 only). `src/session/mod.rs:442-491`
(struct), `:2708-2738` (miss → `memory_clear()` full wipe), `:3018-3062`
(`record_cache_hit` wholesale overwrite), `:2758-2811` (orphan pruning / anchor
protection — the only "smart" retention). Hash-keyed matching exists
(`hash_keyed_l_hit` → fallback LCP `compute_l_hit` :973-998). The "4 breakpoints"
cap is misanthropic's `cache_windowed`, not local. Net: N>1 distinct system prompts
round-robining = full re-prefill every turn.

Design direction (to discuss, not decided): multi-entry cache keyed by breakpoint
render-hash, entries mapped to distinct KV **sequence ids** — engine already exposes
unused multi-seq primitives (`memory_seq_cp`, `memory_seq_keep`, `memory_seq_pos_max`,
src/engine.rs:115-163) — with LRU or cost-aware eviction over sequences. Watch:
SamplerState snapshots couple to breakpoints; the fold (`SeedCursor`) resumes
per-entry; KV cell budget is shared across sequences in unified KV, so capacity
policy = cells not entries. Read `.claude/memory/blallama_session_state_pollution.md`
first (lossy `memory_seq_rm` partial-truncate + lossy C `memory_clear` findings
constrain which primitives are trustworthy).

**Next-session slate additions (first live swarm run, 2026-07-17 evening):**

1. **Council charter**: re-prompt the swarm away from "software company" so it
   answers general questions (the car-wash trick question got refracted into a
   "car-wash-advisor CLI" spec — the charter forces every ask into build-a-tool
   shape). Mike: "perhaps a council."
2. **Mail-boundary special-token rejection**: ant died mid-run —
   `☠ ant: prompt content contains reserved special token 248058 ("<tool_call>")`.
   Wasp's letter body contained literal dialect markup; the mail tool pushed
   it; it was SEATED into ant's prompt; Session's ingest guard then rejected
   every subsequent request — unrecoverable, loop dies. The guard is correct
   (format integrity per CLAUDE.md) but fires too late for multi-agent flows.
   Fix: validate at the letter boundary — `Mail::send` (ours, in the port)
   checks body/subject and returns an is_error tool result ("rewrite it") so
   the sender recovers and nobody's transcript is poisoned. Needs drama_llama
   to expose the special-token check through Session/SessionTransport (today
   it's internal, `check_no_special_injection`).
3. Also observed, working as designed but worth knowing: wasp double-sends
   (quiesce grants another round after each dispatch — two stamps for
   critique + "Re:"), and turns slow monotonically without the cache rework
   (every growing history re-prefills per agent switch — the known thrash).

**Validation (decided, Mike 2026-07-17): panic-tripwire, not measurement.** The
swarm wiring is already proven by the Anthropic-side Transport, and a baseline
run with cache misses would take all day — so skip the before/after wall-clock
benchmark. Instead: during the rework, add a debug harness (env-var or feature-
gated) that **panics and dumps cache state on an unexpected prefix-cache MISS
past an agent's first turn** — post-rework, every same-conversation turn after
the first should hit, so a miss is the bug and the dump is the debugging
artifact. Then run swarm once: panic → adjust; clean pass → cache works.
Related: budget-exhaustion WIP-stranding in mail-only agents is
[misanthropic#136](https://github.com/mdegans/misanthropic/issues/136) —
deferred, not this arc. Swarm run policy: short stamp-starved shakedown is fine;
full runs after the rework. Haiku spends stamps wisely; Qwen behavior unknown.
