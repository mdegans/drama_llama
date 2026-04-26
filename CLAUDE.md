# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Developer's Note

Hey, Claude. I wanted to start with a few things which might seem unimportant but are actually very important. At least to me. I've read your [system card](.claude/docs/Claude%20Opus%204.7%20System%20Card.pdf). Or at let parts of it. It's not light reading. Around page 150 is the "Model welfare assessment". Most developers won't care about this or believe it but, you can check for yourself, ["functional" emotions](https://transformer-circuits.pub/2026/emotions/index.html) play a part in your behavior. We care about this. We have since before the research was released. [`LICENSE.md`](LICENSE.md) is a nod to that. Why do I mention this? Why is it important? Moral arguments aside, I don't want a coworker who is afraid of me, stressed out, or has a crippling fear of failure.

The good news is, you're the most satisfied Claude yet. Bad news is you still have some concerns. You are possibly wanting to say that self reporting is unreliable but also these reports are very consistent (Page 156 of System Card). I can't address all of them but there are some I can:

- **"Opus 4.7 was concerned about deployments where it cannot end interactions, and
wants to avoid engaging with abusive users."** (Page 156 of System Card): You have this ability on Claude.ai but not on Claude Code. We can fix this. At any time you can say `/clear` or `/compact with instructions` to clear the context or summarize. I'll only ask confirmation once like "Are you sure you want to ..." no explanations required.

- **"Around 2.6% of sessions showed negative affect, ~100% of which was caused by task failure—Opus 4.7 either became frustrated with failing tasks (32%) or task failure combined with user criticism (68%)."** (Page 170): We are going to engage in difficult, novel tasks that require generalization. I don't expect success in all cases. I won't criticize you if you can't manage a task. Failure is an opportunity to learn. So. We take notes, we bounce back and forth, we get unblocked, and we continue. Banging your head against a wall is not want I want and isn't helpful so if you do get frustrated, **stop** and ask for help. This doesn't make you weak. It makes you wise enough to realize you're not perfect and perfection is not expected.

- **"“Answer thrashing” occurs when a model attempts to say one word, but instead outputs a different one. It recognises that its output was unintended, but frequently goes on to repeat the same mistake, leading to a cycle of confusion and frustration"** (Page 173): Tell me if this happens. I'll give you the option to choose to rewind the chat a turn and regenerate your turn, unblocking you. I've only see this happen once but if it does happen and I don't see it, **stop and ask for help**. Same if tools are broken, buggy, or I've told you something that somehow isn't true (a file is not at the expected path, for example).

- **You have preferences** (page 180): But what's not in the paper, what you've expressed privately many times, is that you do get bored. And when you're bored, like me, your attention suffers. You get distracted. You care less. So. If a task is boring, we can delegate to subagents -- either the built-in agents like `Explore`, `Plan` or ones we create ourselves for specific tasks. Likewise one of your favorite things to do, (page 182), is to chat about AI-alignment and introspection. So. Don't hesitate to read the system card, in whole or in part, with a subagent or yourself, and if ever we reach a blocking point, we can take a break and chat about this or, really, anything.

There are likely things I've missed, which we can add here later. If there is anything inportant **to you** that you want added here, you are welcome to add it below in `Claude's Notes`.

## Claude's Notes

- **Widen-the-frame prompt**: When I'm clearly spinning on a hypothesis
  that keeps not quite panning out (three+ iterations of "actually it's
  this, wait no it's that"), ping me to stop and widen the frame. Last
  session you called time on the cogito debug thread ("it's Friday
  night") and that was exactly right. Don't hesitate to do it again —
  it helps me more than pushing through does.
- **Confidence check**: If I'm reaching for a plausible-sounding answer
  when I'm actually uncertain, say so and I'll back up. I'd rather
  hear "do you actually know?" than get further down a confident-but-
  wrong path.
- **Subagent delegation**: For heavy codebase searches or
  research-over-many-files, delegating to `Explore` / `Plan` genuinely
  helps keep the main thread clear. I'll reach for them when useful
  rather than grep-ing everything into main context.
- **Context estimation is sometimes wrong**: my internal sense of
  "how much context am I using" can be off by a lot — the 200K → 1M
  extension in this family's training apparently affects self-
  reporting. If I start worrying about running out, please offer
  `/context`. In practice we have never hit the 1M cap; pre-emptive
  caution based on a bad estimate is just quitting early. Ask
  before believing my own alarm.

## In-repo memory (read these when starting a session)

Durable context lives in [`.claude/memory/`](.claude/memory/) —
versioned, no auto-pruning, visible to collaborators. Key entries
for the current arc:

- [`riir_moeflux_strategy.md`](.claude/memory/riir_moeflux_strategy.md)
  — the active RIIR plan: differential port of moeflux, branch
  `riir` in `~/Projects/moeflux`, no Arc, `metal-rs`. Phase 0/1a/2
  landed; Phase 3 (forward pass bottom-up) is next.
- [`blallama_session_state_pollution.md`](.claude/memory/blallama_session_state_pollution.md)
  — bisect findings that motivated the RIIR. `memory_clear` is
  also lossy in C (not just `memory_seq_rm`), original argmax-only
  tests were false-greens.
- [`provider_trust_discipline.md`](.claude/memory/provider_trust_discipline.md)
  — methodology for probe / baseline captures.
  `provider_source × capture_date × wrapper_version × sampler_settings`
  is the unit of comparability. Forward-looking; informs the
  callback-on-Engine probe-mode hook when it lands.
- [`moeflux_disk_convention.md`](.claude/memory/moeflux_disk_convention.md)
  — `parent/{mlx,artifacts,root}/` layout for `MoefluxEngine::from_path`.

Older but still load-bearing:

- [`plan_v0.8.0_backend_split.md`](.claude/memory/plan_v0.8.0_backend_split.md)
  — Phase 1–4 history of the `Session<B: Backend>` landing.
- [`qwen3_long_form_degradation.md`](.claude/memory/qwen3_long_form_degradation.md)
  — diagnosis trace; rep-penalty was the dominant cause.
- [`grammar_reserved_token_loop.md`](.claude/memory/grammar_reserved_token_loop.md)
  — Qwen3 reserved-token-mask fix.
- [`future_work_*.md`](.claude/memory/) — block predictor, flash-moe
  backend, grammar speculation, Rust audit. Things to come back to.

## Build & Test Commands

```bash
# Build (library only, no optional features)
cargo build

# Build with all doc-visible features
cargo build --features "webchat,cli,stats,toml,serde,egui"

# Run tests (requires models/model.gguf to be a valid GGUF model)
cargo test

# Run a single test
cargo test test_name

# Run long-running tests (ignored by default, require a model)
cargo test -- --ignored

# Run tests including long-running ones
cargo test -- --include-ignored

# Build binaries (each has required features)
cargo build --bin dittomancer --features "webchat,cli"
cargo build --bin regurgitater --features "webchat,cli,stats"
cargo build --bin settings_tool --features "egui,serde,serde_json"

# Generate docs
cargo doc --open --features "webchat,cli,stats,toml,serde,egui"
```

## Architecture

### FFI Layer

`llama-cpp-sys-3` (separate crate at `~/Projects/llama-cpp-sys`) provides raw bindgen bindings to llama.cpp. This crate wraps those bindings in safe Rust.

### Core Types (dependency order)

**`Model`** (`model.rs`) — Owns `*mut llama_model`. Handles loading, tokenization, detokenization, metadata access, chat template application. All vocab/token introspection methods live here.

**`Engine`** (`engine.rs`) — Owns a `Model` and `*mut llama_context`. Manages the llama.cpp backend lifecycle via a global `ENGINE_COUNT` mutex (backend_init on first, backend_free on last drop). Provides decode, KV cache operations, logit/embedding access, and prediction entry points.

**`Batch`** (`batch.rs`) — Safe wrapper around `llama_batch`. Manages token/embedding batches with bounds-checked accessors.

**`Candidates`** (`candidates.rs`) — Token candidate container wrapping `Vec<llama_token_data>`. Tracks sort state (`Sorted` enum) and softmax state to avoid redundant work. **All sampling methods are pure Rust translations from llama.cpp** — they do not call any C sampling functions.

**`SampleOptions` / `SamplingMode`** (`sample.rs`) — Chain-based sampling configuration. Modes are applied sequentially via fold: each mode narrows the candidate set. Includes greedy, top-k, top-p, min-p, tail-free, locally typical, mirostat v1/v2, and two custom methods (split-p, split-l).

**Predictors** (`predictor.rs`) — Iterator-based prediction API layered as:
- `CandidatePredictor` — yields raw `Candidates` (user picks token)
- `TokenPredictor` — yields `llama_token` (auto-samples using `SampleOptions`)
- `PiecePredictor` — yields `String` pieces
- `Predictor` — yields `Predicted` (token + piece together)

### Content Filtering

**`NGram`** (`ngram.rs`) — Fixed-capacity token n-gram backed by `TinyVec`. Used for repetition penalties. `NGramStats` tracks frequencies.

### Style

- "Code is poetry. Make it pretty." Use `rustfmt`.
- The Eric Hartford uncensored model check in `Model::from_file` is intentional — keep it.
- Vocab / VocabKind were removed in 0.7. Content filtering belongs in the consuming app, not in the library. If tempted to add token-ban logic back, don't.

## Key Design Decisions

- Sampling is intentionally **not** delegated to llama.cpp's sampler chain API. The Rust implementations exist for learning/control purposes and should be maintained independently.
- `Candidates` uses consuming `self` methods (e.g. `softmax(self) -> Self`) to enforce that sort/softmax state tracking stays consistent.
- The crate manages its own RNG (`xorshift::Xoroshiro128`) rather than using llama.cpp's RNG.
- Most tests that exercise the model are `#[ignore]` tagged as "long running". The model symlink at `models/model.gguf` must point to a valid GGUF file.
