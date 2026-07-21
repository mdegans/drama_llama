# Open thought blocks (#59) — read before touching thought parse/render

**Landed 2026-07-21.** A `Block::Thought` can now represent "the model
never closed this `<think>`", and one such thought can be *rendered* so
the model resumes from it.

## The flag

`prompt::OPEN_THOUGHT_SIGNATURE` in misanthropic's otherwise-unused
`signature` field. Helpers: `open_thought`, `is_open_thought`,
`prune_open_thoughts` (all `src/prompt.rs`).

**Polarity is inverted vs upstream and that is deliberate.**
`Block::is_complete_thought` / `Message::remove_incomplete_thought` read
an *empty* signature as incomplete; we read empty as **closed** and the
sentinel as **open**. Rationale: mis-flagging a closed thought as open
makes the renderer omit a close the model really wrote (destructive);
the converse merely loses a continuation. Neither upstream helper is
called anywhere in this crate — if you call one on a local prompt it
does the opposite of what you want.

Why the signature field and not a new one: no `misanthropic` schema
change (it models the Anthropic wire format), and it survives JSON
round-trip — Agora persists prompts, so a `#[serde(skip)]` Rust-side
flag would silently vanish. A body sentinel was rejected outright:
framing bytes inside content is the `[Invalid UTF-8]` / #38 class.

## The invariant: renderable ⟺ sole block of the trailing assistant turn

An open thought renders by being **withheld from the chat template** and
appended to the finished generation prompt
(`chat_template::open_thought_tail` → `render_with_env`). That is the
only position expressible as a suffix of the generation prompt.
Everything else is `SessionError::UnrenderableOpenThought`, checked
beside `check_no_special_injection` on both prepare paths.

This rejects shapes a caller can build in good faith — prose before a
spontaneous `<think>` leaves a leading `Text`, giving
`[Text, Thought(open)]`. Those mis-render today anyway; the error is the
honest version. The error message names `prune_open_thoughts`.

## Why divert-and-append, not strip-the-close

**Do not "render it closed and strip the `</think>`".** Verified from
the Qwen3.6 template (`tests/fixtures/templates/qwen3.6-gguf.jinja:89`):

```jinja
{%- set content = render_content(message.content, true)|trim %}
{%- set reasoning_content = content.split('</think>')[0].rstrip('\n')
                                   .split('<think>')[-1].lstrip('\n') %}
```

An unconditional `|trim` on content plus lstrip/rstrip on the split
halves. Anything that *reaches* the template loses its exact trailing
whitespace irrecoverably — `\n` and `\n\n\n` become indistinguishable —
and no post-hoc stripping gets them back. Diverting means the template
never sees those bytes.

Corollary, load-bearing in both directions: **closed** thoughts
round-trip precisely *because* the parser's `strip_prefix('\n')` +
`trim_end()` mirrors that normalization. Keep the two in lockstep. Open
thoughts store the body **raw** (`Parser::push_open_thought`) for the
same reason inverted — nothing re-supplies what a trim eats.

## Three parser sites stamp openness

`parse.rs`: `run`'s pre-opened `Final` arm, `parse_thought`'s
`None`/`None` arm (which replaced `incomplete(start)` — this fixed the
thought-half of #38, where the literal `<think>` was seated in a
`Block::Text`), and `harmony_analysis`'s `Final` arm. Streaming gets it
free: `StreamParser::finish` reparses under `Final`.

Harmony is flagged but **not renderable** — gpt-oss's generation prompt
never pre-opens a channel (`emit.rs` collapses both eager anchors), so
`dialect_renders_open_thought` excludes it. A gpt-oss truncation is
rejected at ingest rather than silently re-rendered with a fabricated
`<|end|>`. Resuming Harmony would mean appending an `analysis` channel
header; possible, not done.

## Two non-obvious couplings

1. **`render_extended` must merge, not append** (`session/mod.rs`). It
   is the canonicalization gate's renderer; appending the continuation
   as a *new* assistant message would push the seed's open thought into
   mid-history and fail the gate on **every** continued turn — losing
   the auto-tip, which is the exact cost this feature exists to avoid.
   After merging, a continued turn is byte-identical to one that ran to
   completion in a single call. This is the *only* place that merges;
   the public API never does (see below).

2. **`pre_opened_reasoning` is an OR, not a widened scan.**
   `render_ends_with_open_reasoning` stays narrow (bare trailing marker
   only) and is OR'd with `prompt_resumes_open_reasoning`. Widening it
   to "last open marker after last close" was considered and
   **rejected**: an unmatched `<think>` in a final user or tool message
   would flip the grammar anchor — content deciding framing.

## No implicit merge across the public API

`complete_blocks` returns only what that call generated. Seating a
continuation beside its seed without merging is a hard error at the next
render, not a silent mis-render. `Block::Thought` has no `cache_control`
field, so a caller **cannot** place a breakpoint at the seed boundary;
reuse there rides the LCP walk and the auto-tip.

## Proof it works

`examples/unhelpful.rs` — prefilled reasoning makes the model refuse
every request (control run answers normally), then truncates a thought
and resumes it: measured **cache read 84 / created 1**. Pinned as
`truncated_thought_resumes_from_cache` and
`unrenderable_open_thought_is_rejected` in `tests/session.rs`, plus
byte-exactness (`open_thought_tail_appends_raw_after_generation_prompt`,
body ending in `\n\n\n`) in `chat_template.rs`.

Related: [[byte_exact_round_trip_invariant]], #38 (tool-call half still
open), #62 (the release-only `debug_assert!` parser bug found while
doing this).
