# How far logits are comparable across backends (MoE routing is a cliff)

**Measured 2026-07-14** on Qwen3.6-35B-A3B (IQ4_XS), CUDA golden vs
Metal run, `tests/regression.rs`. Directly relevant to the **moeflux
diff-oracle**: it tells you which signals can be held to a tight
tolerance and which cannot, and why widening the tolerance won't save
the latter.

## The measurements

| signal | CUDA vs Metal |
|---|---|
| `prompt_tokens` (tokenization) | identical |
| `tokens` — 32-step greedy stream | **identical, byte-for-byte** |
| `logits_step_0` (prefill, top-20) | same ids, max drift < 0.5 nats |
| `logits_step_n` (31 decodes deep, top-20) | **10/20 shared ids**, max drift **1.86 nats**, id-aligned cosine **0.58** |

Metal is deterministic run-to-run (identical diagnostics across
invocations). So the step-n divergence is *between backends*, not noise
within one.

## Why the tail collapses but the argmax doesn't

Almost certainly **MoE routing**. a3b activates 8 of 256 experts. A hair
of numeric difference in the router flips *which* experts fire, and that
is a **discrete** divergence, not smooth float drift — a different expert
mix rewrites the tail of the distribution wholesale. The argmax survives
because it leads by 7.18 nats; nothing in the tail can climb that.

It also **compounds with depth**: step 0 is one forward pass over 4
prompt tokens (agrees fine), step 31 has had 31 chances for a routing
flip. Expect divergence to grow with context length, not stay flat.

## Consequences

- **A tolerance cannot fix a membership change.** Half of step-n's top-20
  is *different tokens*, not the same tokens with different values.
  Widening `LOGIT_TOL` is not a fix, it is a way of not noticing.
- **Deep-context logit comparisons across backends are not a contract.**
  For the diff oracle this means: compare early positions tightly;
  compare late positions loosely or not at all; and never conclude
  "moeflux is broken" from a divergent deep-context *tail* alone. Check
  whether the argmax and the greedy stream still agree first — those are
  the signals with teeth.
- **The greedy stream is the real oracle.** 32 steps of greedy is 32
  argmax assertions over a 248k vocab, and it survived a backend change
  intact. It is a far stronger and far more portable signal than any
  single top-K snapshot.
- **Same-device pinning is not a rescue either.** Drivers, llama.cpp
  kernels, and macOS updates all move these numbers. So `regression.rs`
  keeps ONE cross-backend golden and simply doesn't assert the parts that
  aren't invariant: step-0 top-K by id within tolerance, step-n argmax
  only, tail recorded + reported but not enforced. Assert invariants,
  record everything else.

## Diagnostics, and a note on cosine

`regression.rs` prints `max_drift` / `shared_ids` / id-aligned cosine on
the unasserted tail. Cosine is a **diagnostic, never an assertion** —
tested against this data, a rank-ordered top-K vector scores 1.00000000
even when *every token id is wrong* (position in the vector is "whatever
came 3rd", not a token identity), and even id-aligned it puts benign
noise (0.99999) and a real 2-nat regression (0.99944) 0.0006 apart while
diluting one exploding token across nineteen others. It works in moeflux
because activation tensors are indexed *by dimension* — that property is
what's missing in a rank-ordered top-K. Per-id delta in nats is a bound,
is interpretable, and is what we assert on.

See also [[provider_trust_discipline]] (comparability units) and
[[riir_moeflux_strategy]].
