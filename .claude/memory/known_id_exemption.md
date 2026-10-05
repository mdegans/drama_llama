# Known-id exemption — why the repetition penalty spares copied ids, not id *shapes*

**Landed 2026-09-22.** Read before touching `sample::ids`, before
proposing a shape/regex-on-the-*emission* detector for identifiers, or
before "fixing" the seeding fold to mirror the live pass.

## The problem, as observed

Agora agents cite post/comment UUIDs and `GOV-2026.7`-style ids
constantly, and must re-emit them verbatim. Stacked n-gram penalties
(sizes 2–5, `penalty_repeat^n` multiplicative on positive logits —
`1.06^14 ≈ 2.3×` under the Qwen3.8 sidecar) push the model off the exact
string after a few sightings; it then says "I can't trust my own memory"
and eventually surrenders (`placeholderAGENDAPOST01` in a real tool call).

## What was tried first, and why it was the wrong tool

`4b929dd` (same day, removed the same day, never released) suspended the
penalty by *shape*: a byte tracker in `SamplerState` that fired from a
UUID's first dash to its 36th char. On device it worked exactly as
designed — every post-dash tail became byte-exact — and the failure
moved to the first group. The blallama log showed why shape detection
cannot close that: the model cites ids as **bare 8-hex prefixes**
(`[05676b9d]`, `c966b...`) dozens of times per turn. No dash, so nothing
to detect; `05676b9d` and `deadbeef` are the same shape. The prefix is
poisoned by its own short-form repetitions before any full UUID is
written. Retracting completed UUIDs from the n-gram store has the same
blind spot. Don't reopen either.

## What shipped (balerion's proposal)

Exempt **faithful copies of ids the context already contains**:

1. `RepetitionOptions::id_patterns` — consumer-supplied regexes
   (`IdPattern`, compiled at the sidecar door; invalid ⇒ strict error).
   Default empty: which strings are identifiers is the consumer's
   knowledge, not the crate's.
2. `Session::predict_options_for` runs them over the prompt's text
   leaves (`prompt_known_ids`, via `block_free_text`) into the per-call
   `RepetitionOptions::known_ids` (`serde(skip)` — config, not state,
   not sidecar, not cached tip). **Prior `Thought` blocks are skipped**:
   that is where the model's own miscopied ids live; protecting them
   would launder last turn's mistake into this turn's known set.
3. `IdGuard: RegionGuard` — the same hook the constrained pass uses to
   protect region-exit tokens. Per step it derives the partial word from
   the trailing pieces of `tokens` (nothing in `SamplerState`; pure
   function of history, so snapshot/restore are untouched) and exempts
   a penalized target iff `word ++ piece` is a prefix of a known id.
   Composed with the region guard by `region::Either` in regime (b).

Properties: covers the entry token and bare prefixes (a prefix of a
known id is exempt by construction); **only** copies of known ids — a
wrong start keeps its full penalty, so the exemption steers toward the
set of ids the context holds; ids are still *recorded* in the corpus,
just never penalized.

## #113 (2026-09-23): multi-word ids, and numbers as a category

**Multi-word ids.** The first cut derived "the partial word" by walking
back to the last byte outside `[A-Za-z0-9._-]`, so a space ended every
copy and `Sept. 7` / `September 22, 2026` could never be exempt past
their first word. Now the guard keeps *copies in progress*: every suffix
of the last `min(64, longest id)` history bytes that begins at a word
start and is a prefix of some known id (plus the empty copy at a word
start). A piece is exempt iff it extends one of them, starts a fresh
copy at a word start inside itself (` 05`), or completes one exactly
and then crosses only non-word bytes (`9d]`, `7,`). The word class now
only decides where a copy may *begin*; the ids decide where it ends.

**Numbers were the bigger bug.** A tokenizer probe (Qwen3.8, Gemma 4,
Mistral Small 4, gpt-oss — all four) showed ` 22` is always a bare ` `
token + digit tokens (single digits; gpt-oss groups ≤3). So eleven
tokens carry *every* number in the context, and in surgical mode the
bare ` ` accumulates penalty from every n-gram that starts with it —
the mock battery drove it down by ~40 logits in 20 steps. #113's
"today is Sept. Sept." is exactly the model unable to emit the space
that begins the day number. Fix: `IgnoreCategory::Numbers` (` `, the
digits, every 1–3 digit spelling with and without a leading space),
**default on**. Numbers are facts; words do the loop-breaking.

Sidecar patterns (Agora, matching agora-agentkit's renderer — note the
old `GOV-2026.7` pattern was stale; agentkit renders `GOV-2026-0006`
and also AMD/KEY/REC): UUID, `(?:GOV|APP|AMD|KEY|REC)-\d{4}[-.]\d+`,
ISO date/RFC 3339 timestamp, English dates, kebab-case handles/slugs.
Pinned by `ids::tests::agora_sidecar_patterns`. balerion's sidecar is
separate — copy the `[repetition]` block over.

## #113 part 3 (2026-09-23): the ids were fine — the *close* was penalized

The 2026-09-22 Agora trial transcripts (five Qwen3.8 agents; private
data, never in the repo) showed the dominant "id failure" was not a
miscopy at all: the UUID is exact, then the value **does not end** —
`…e2f5\n】\n\n</invoke>…`, `…4a0b3a8d6f7e1b2c…`, `…\nWait — recheck…`.
It clusters by call position within a turn: broken on 7 of 11 *third*
calls vs 2 of 18 second calls (and those two are the stale-GOV-pattern
class). Cause: the region guard protected only tokens that *leave* the
free region, and Qwen3.8's string values are `until("\n</parameter>")`,
a multi-token exit whose middle states are still permissive. The
delimiter's n-grams repeat once per call, so by call three the close was
suppressed and the model reached for lookalike closers. region.rs had
documented this as a "v1 limitation, bounded". It was not bounded enough.

Fix (`65cc608`): each grammar region finds its *home* (the permissive
fixed point a plain content byte returns to) and any token whose walk
ends elsewhere is protected. Also covers `until("</think>")` thought
regions under a tool grammar — which may be part of why thoughts ran
4–8k tokens against a 4096 "budget".

Lesson for next time: when a copy failure looks like a miscopy, check
whether the copy was *right and unterminated* first. Tabulate failures
by call index within the turn — position-dependence points at
per-call-repeated structure (delimiters), not at the content.

## Sidecar coverage: every served model, or it silently doesn't apply

2026-09-24: the #113 block went on the Qwen sidecars only. gpt-oss,
Mistral 4 and cogito spell out `ignored_categories`, which **replaces**
the library default (so they didn't get `Numbers` either), and had no
`id_patterns`. They ran a whole Agora night with #113's bug:
- ids mangled into near-copies (dashes dropped, wrong digits, U+2011
  hyphens);
- about 37% of their sessions stalled (3 rounds without a successful
  tool call), against 1/56 on Qwen.

A stall produces no blallama error, so watching blallama's log reported
"healthy" all night. When a repetition-block change is meant for Agora,
grep every `models/*.sampling.toml` the cohort uses, and measure
**runner-side stalls**, not just blallama errors.

## Edges, documented, accepted

- At a word start, any token whose piece starts some known id is
  exempt (now including word starts *inside* a multi-word copy — after
  `Sept. `, `Sept` is exempt as the start of a fresh copy): with dozens of UUIDs in context that is every space-prefixed
  hex digit and short hex word (` 1`, ` be`, ` bad`) as the *first*
  token of a word. The next token is judged against the longer word and
  re-penalized; surgical mode already spares single digits. Bounded, not
  zero.
- A merged piece that completes an id and crosses a boundary (`9d]`) is
  exempt iff the id is completed exactly.
- On an exact-tip resume `tokens` is the suffix, so the partial word is
  empty at the first step — same tolerance class as Phase 1's truncated
  trailing n-grams.
- A pattern as loose as `\w+` makes every word a known id and disables
  the penalty. Keep patterns specific.

## Not done, on purpose

- The seeding fold is untouched. Live and fold corpora already differ
  (template-markup n-grams, BOS, `windows(max)` shape, step counting)
  and **nothing asserts them equal** — the cold≡incremental oracle is
  fold-vs-fold at a prompt breakpoint; `tests/sampler_state_cache.rs`
  is resume-vs-resume. (I believed otherwise in a plan draft; the
  reviewer corrected it.)
- Thinking budget: `budget_tokens` is accepted and silently discarded —
  only `enable_thinking: bool` reaches the template. Enforcing it is a
  separate feature.
