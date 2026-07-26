# Prefix-cache loss on tool-call turns (#85) — Phase 0 findings

**Read before touching `compute_l_hit`, the auto-tip, `render_reference`,
or proposing a chat-template change.** Measured 2026-07-26/27 against
`models/cogito-32b.gguf` on the M2 Max, reproducing a downstream Agora
seed-runner report. Captures and repro scripts were in the session
scratchpad (ephemeral); everything durable is here.

## Symptom

`blallama` loses prefix reuse on exactly the turns where an assistant
message containing a `tool_use` block is replayed as history. Qwen3.6 and
gpt-oss are unaffected. Deficits compound as the conversation grows —
3, 59, 1115, 3964, 4705 tokens across a 13-request trace — always
"everything after the last surviving anchor".

Reproduced exactly on this box: replaying the two captured request
bodies gives request 2 `cache_read = 11589`, `cache_creation = 403`,
against request 1's `input_tokens = 11592`. Same constant as the
downstream capture.

## Ruled out by measurement (do not re-litigate)

- **Not model-family misdetection.** Cogito is `general.architecture:
  qwen2`, base `Qwen2.5 32B`. The crate does no name/architecture
  sniffing at all — dialect comes from sniffing the Jinja source — and
  Cogito classifies correctly as `Family::JsonNative`, the same arm as
  Qwen3/Hermes.
- **Not a render or tokenization divergence in the prompt.** Rendering
  and tokenizing both requests with the real cogito vocab shows request
  1's full 11592-token prompt present *verbatim* in request 2's prefix.
  `<tool_call>` is **special token 151657**, so the `\n` before it
  tokenizes standalone and there is no BPE merge at the assistant
  header. Cogito's template asymmetry (`<|im_start|>assistant` with the
  newline inside the `if content` arm, so a content-less tool-call turn
  renders `<|im_start|>assistant` + `\n<tool_call>`) is **harmless** —
  byte- and token-identical to the generation prompt.
- **Not `output_config`** — the prior leading hypothesis, open since
  2026-05-12. Trace rows that set it are healthy.

## Actual mechanism

1. `slot.prev_entries` = prompt entries + **raw generated tokens**
   (`src/session/mod.rs:4667-4669`). Next call's `new_entries` = the
   **canonical re-render**.
2. Cogito's template re-serializes call arguments with
   `{{ tool_call.arguments | tojson }}`, which minijinja emits
   **compact**. The model emits **spaced** JSON.
3. The LCP walk must cross the re-rendered assistant turn to reach the
   auto-tip (which sits at the end of the emission). It dies at the
   whitespace divergence ~45 bytes into a 1139-byte emission, so `safe`
   drops below the tip position and the tip becomes ineligible.
4. `compute_l_hit` (`:1507-1531`) computes the true LCP, backs off one
   entry for BPE safety, then returns the largest **anchor** at or below
   it — never `safe` itself. With the tip unreachable, reuse snaps back
   to the last user `cache_control` breakpoint.

Confirmed directly in the logs (`RUST_LOG=drama_llama::session=debug`),
on both tool turns:

    "emission does not re-render byte-stable; auto-tip hash skipped
     (LCP fallback)"  — src/session/mod.rs:5116

Note `hash: None` means "LCP-matchable only", so the tip being un-hashed
is *not* what kills it — being unreachable by LCP is.

### The divergence, exactly — TWO of them

Measured with a model-free round-trip test (now in
`tests/template_rendering.rs`, `#[ignore]`d, tagged #85). All three
byte streams disagree:

    model emitted    {"community": "debate", "body": "x's belief …"}
    render_reference {"community":"debate","body":"x's belief …"}
    template render  {"community":"debate","body":"x\u0027s belief …"}

1. **Whitespace.** The model emits spaced inner JSON (`": "`), both
   `render_reference` and the template emit compact (`":"`). This one
   hits first, ~46 bytes in, at `{"community":` — which is why
   essentially the whole 1139-byte emission is unreachable and reuse
   snaps clear back to the last breakpoint.

   Cause: `ws ::= [ \t\n\r]?` in the shared JSON grammar prelude
   (`src/grammar_compile.rs:874`), referenced throughout the emitted
   tool-call rules (`emit.rs:573`, `:604`, `:610-617`). **Our own
   grammar under-determines the emission** — the model may legally emit
   `":"` or `": "`, and no renderer can know which it chose.

   Note the spaced-separator code at `emit.rs:817-836` applies only to
   the *top-level* `name`/`arguments` fields; the inner args object
   rides `serde_json`'s compact Map serialization.

2. **HTML-safe escaping.** minijinja's `tojson` follows Jinja2 and
   escapes `'` → `\u0027`, `&` → `\u0026`, `<` → `\u003c`,
   `>` → `\u003e`. Neither the model nor `render_reference` does.
   Non-ASCII (`→`) is *not* escaped by either, so `ensure_ascii` is
   off — it is specifically the HTML-safety set.

   The production payload contains `x's belief`, so this fires in
   practice. It is a *second*, independent break: fixing whitespace
   alone would still leave apostrophe-bearing calls diverging.

**Not a minijinja bug — verified.** Python jinja2 (the reference impl,
and what HuggingFace's `apply_chat_template` uses) renders this template
**byte-identically to minijinja**, compact *and* HTML-escaped:

    uv run --script tests/fixtures/render_jinja.py \
        tests/fixtures/cogito_14b_template.jinja <vars>.json

Jinja2's `tojson` is `htmlsafe_json_dumps`, not plain `json.dumps` — so
compact separators and `'&<>` escaping are *correct* Jinja behaviour, not
drift. There is no implementation bug to fix; the whole Jinja ecosystem
renders tool-call history one way and the model writes it another.

Consequence for the fix direction: **do not** pin the grammar to the
template's output. That would require the model to emit `\u0027` for
every apostrophe — six characters fighting the model for no benefit. The
render must move to the model, which means owning the tool-call
serialization rather than deferring to the template's `tojson`.

Worth noting independently of #85: rendering the model's own apostrophe
back to it as `\u0027` is a *fidelity* bug. History the model reads
differs from what it wrote, on every `tojson`-based template. HTML-safety
escaping in a context with no HTML.

**Why this went unnoticed:** every pre-existing test payload is clean
ASCII with no `'&<>`, and the whitespace question never arose because
nothing compared `render_reference` against a real template render.
Swap the new test's payload for a plain string and it passes.

Templates are NOT the variable here: `tests/fixtures/cogito_14b_template.jinja`
is **byte-identical** to the 32B GGUF's embedded template (verified by
diff), so 14B fixture results transfer to the 32B model directly.

## Governing requirement (Mike, 2026-07-27)

**On resume, session state must be identical to where the previous turn
left off — down to the RNG position.** Agora deliberately runs a mixed
fleet, so this must hold on every backend.

This is currently violated. `seed_prose_fold` / `seed_prose_block`
(`:1052-1095`) only mutate `ngram_stats`; they never advance the RNG.
The RNG position comes solely from the inherited `SamplerState.rng`, so:

| resume anchor | RNG state obtained |
|---|---|
| **the tip** | exactly where the last turn ended — the requirement |
| an earlier breakpoint | that breakpoint's *pre-generation* snapshot |
| nothing | fresh entropy |

**Only a tip match satisfies it.** The tip is lost on two independent
paths: (1) tool-call turns, above; (2) **any** turn ending on max-tokens
or grammar-complete — `compute_tip_extension` (`:4643`) creates a tip
only in the stop-sequence branch. (2) is backend- and model-independent
and affects every model.

## Load-bearing facts about rewind (I got this wrong once)

- `Engine::restore_to` does **not** require a snapshot on attention
  models. `src/llama_cpp/decoder.rs:800-804` tries
  `llama_memory_seq_rm(seq, pos, -1)` first — lossless and copy-free —
  and only falls back to a stored snapshot when llama.cpp refuses
  partial-range removal (recurrent/hybrid; see
  [[qwen3_a3b_llama_cpp_rewind_diagnosis]]).
- `seq_snapshots_enabled` = `llama_model_is_recurrent ||
  llama_model_is_hybrid` (`decoder.rs:269-275`). **On cogito,
  snapshotting is off entirely** — `checkpoint_pos` early-returns, no
  snapshots exist, and every restore already *is* a bare truncate. The
  anchor restriction buys nothing on dense attention.
- `checkpoint_pos` snapshots the **current** head keyed by `pos`. It
  cannot retroactively snapshot a past position — this kills any
  "checkpoint at the divergence point after the fact" design.

## Direction

Mike has approved (2026-07-27) eventually shipping **our own chat
templates** for all models, noting stock GGUF templates are of mixed
quality and HF has better ones.

The framing that fell out of the measurements: **the grammar and the
renderer must be two views of one canonical byte string.**
`render_reference` already *claims* to be that string
(`emit.rs:704-710`) — the bug is that nothing makes either side honour
it. So the fix is two-sided and both sides are ours:

1. **Pin the emission.** Stop `ws` from floating inside the emitted
   tool call, so the model cannot choose whitespace the renderer can't
   reproduce. Emission then equals `render_reference` by construction
   rather than by habit. (Do *not* just teach `render_reference` to
   emit spaces to match the model's current habit — the grammar would
   still permit the other choice, so it would be a coin-flip fix.)
2. **Make the render match.** The template's `tojson` HTML-escaping has
   to go, or the tool-call turn has to render through our own
   serialization instead of the stock template's filter.

Together these make the byte-stability gate at `session/mod.rs:5080`
pass by construction, which keeps the auto-tip alive, which is what the
governing requirement actually needs — **no KV surgery required.** This
supersedes the plan's Phase 2 ("canonicalize the KV"), which was
designed before these measurements existed and is strictly more
machinery.

**Do not assume `render_reference` matches any template today.** Its
doc comment says it does; for cogito it does not, in two ways. A
round-trip test per fixture template is the missing guard — one now
exists for cogito (`#[ignore]`d) and should be generalized to
gemma4 / gpt-oss / Qwen when the fix lands.

Watch out: `<tool_call>` is a *special* token, so any design that feeds
canonical call bytes back through a text path will trip
`check_no_special_injection` and needs an internal bypass.

## Related

- Issue #86 — `render_partial` drops `prompt.thinking`, so partials
  render with `enable_thinking = false`. Cogito is the one template
  where `enable_thinking` rewrites the *front* of the prompt, so its
  partials stop being prefixes. Real and latent, but **not** #85's
  cause: breakpoints demonstrably survive in the trace.
- `examples/inspect_prompt.rs` is the tool for this class of bug —
  renders + tokenizes a saved prompt JSON and prints breakpoints,
  per-message token ranges, and the generation-prompt tail. Its
  assistant token dump was dead (`role == "assistant"` compared against
  `Role::as_str()`, which yields `"Assistant"`); fixed with
  `as_lowercase()`.
