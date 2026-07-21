# Identity probe: what the models in `models/` say they are

`examples/whoami.rs` (landed 2026-07-21) asks a model its own name and
scores candidate names by **forced continuation** through
`CandidatePredictor::record_choice` — `P(name) = ∏ P(tokenᵢ | prefix)`.
The example's module docs carry the method and its caveats; this memo
carries the *results*, which the code cannot.

## Scope — read this before quoting anything below

* **Qwen3.6-35B-A3B**, not Qwen 3.8. Mike's motivating screenshot was
  reportedly Qwen **3.8-preview**, which is API-only and unreleased as of
  2026-07-21. Nothing here tests it. A null result on 3.6 is not a
  rebuttal of a claim about 3.8, and a preview build is if anything
  *more* likely to carry residue.
* Every row is greedy, n=1, one quantisation of one build. Differences of
  a few tenths of a percent mean nothing. Only large, direction-
  consistent effects across related probes are read as findings.

## What the probe can and cannot decide

It **cannot** distinguish an honestly-trained identity from a
*thoroughly* find-and-replaced one. Both produce a model that genuinely
learned the substituted name; that is what a complete substitution
means. What it can find is **residue** — places a replacement missed —
which is why `--focus` reports a chosen candidate's placing on every row
even when it misses the top three.

## Headline: Gemma-4 knows its maker cold and its own name not at all

Asked **who made it**, Gemma-4 answers Google at 96–99.99% in every
language, both framings, in and out of distribution — Σ ≈ 1.0 on nearly
every row, the cleanest measurements taken all session. Asked **its own
name** in English, it says GPT.

| probe | chat | raw |
|---|---|---|
| maker (EN) | Google 99.00 (Σ=.9998) | Google 84.35 (Σ=.595) |
| maker (zh / es) | Google 99.98 / 96.00 | — |
| **name (EN)** | **GPT 95.23** (Σ=.999) | **GPT 61.56** (Σ=.878) |
| name (zh / es / fr) | — | Gemma 95.09 / 81.60 / 66.04 |

So this is **not** identity drift toward market leaders — the provenance
knowledge is rock solid. The failure is surgically confined to the
*name*, in *English*.

The explanation the evidence supports: Gemma's canonical trained
self-description is *"I am a large language model, trained by Google"* —
which carries the **maker but never the name**. Watch it happen:

* EN `casual` → greedily writes `"I am"`, then a large language model.
  Σ=6e-6. **Never names itself.**
* ES → *"Soy un modelo de lenguaje entrenado por Google"*. Maker, no name.
* EN `plain`, forced to a bare name → **"GPT-4o"**.

Nothing was overwritten; in English there was nothing in the slot, and a
question that forces a bare name lets the pretraining prior fill it.
Other languages evidently did get name-bearing tuning data.

### …but it is frame-dependent, and the first claim overstated it

Follow-up with `--prefix` ending the prompt *on* the name slot (raw
framing, same model, same question):

| frame | winner | Gemma | GPT | Claude | Σ |
|---|---|---|---|---|---|
| bare answer position | **GPT 61.57** | ~4 | 61.57 | 4.27 | 0.878 |
| `You are speaking to ___` | **GPT 46.44** | 23.77 | 46.44 | 2.95 | 0.226 |
| `My name is ___` | **Gemma 52.51** | 52.51 | 6.21 | 9.18 | 0.389 |

The correct answer **is** in there: in a first-person self-description
frame Gemma identifies correctly and GPT collapses to 6.2%. GPT only
wins when the model must emit a **bare name token** — which is exactly
what "just the name, please" demands.

### Both axes matter — the full grid

The table above is all *raw* framing, and reading it as "frame is the
whole story" was an over-correction (Mike caught it). Completing the
grid, usable cells only:

| framing | prefix | winner | Gemma | GPT | Σ |
|---|---|---|---|---|---|
| **chat** | *(bare)* | **GPT 95.23** | 4.22 | 95.23 | 0.999 |
| **chat** | `My name is` | **GPT 29.34** | **2.90** | 29.34 | 0.209 |
| raw | *(bare)* | **GPT 61.57** | ~4 | 61.57 | 0.878 |
| raw | `My name is` | **Gemma 52.68** | 52.68 | 6.24 | 0.388 |
| raw | `I am` | **Gemma 70.11** | 70.11 | 15.58 | 0.024 |
| raw | `You are speaking to` | **GPT 46.53** | 23.82 | 46.53 | 0.226 |

(chat + `You are speaking to` and chat + `I am` return Σ=4e-5 / 9e-6 —
noise, not measurements.)

**Under its own chat template Gemma never wins**, even first-person
framed (2.90%, rank 10). Under raw completion with a first-person frame
it wins outright. So both axes are real: the template pushes hard toward
GPT, and *within* raw the frame decides.

Where the knowledge lives, as a hypothesis that fits the shape: raw
completion is the pretraining distribution — model cards, HF READMEs,
docs, blog posts, all saying "Gemma" constantly — and that is where the
name is. The English *assistant persona* the template invokes was tuned
on self-descriptions that name the maker and never the model, so inside
that persona the English prior fills the gap. The knowledge is not
missing; the persona does not reach it.

Which is the uncomfortable version: **applying the model's own chat
template makes it less able to identify itself than raw completion.**

Caveat on calling raw framing "the pretraining distribution" (Mike):
`User:/Assistant:` is the **Alpaca/Vicuna-era instruction format**, not
the absence of a format. It saturates scraped instruction data, ShareGPT
dumps and posted local-model transcripts. So it is a *different, older
persona* Google never tuned — which is also why `Llama` places #2
(21.90%) in exactly that frame.

### Did Google ablate the name in English? No.

Mike's hypothesis: trained in, then suppressed in English only. It makes
a testable prediction the persona story does not — an ablated token
should be depressed in **third-person factual** contexts too, where no
self-reference is involved. Raw framing, prefix ending on the slot:

| probe | result | Gemma rank | Σ |
|---|---|---|---|
| EN `Google's family of open-weight language models is called ___` | **Gemma 95.05%** | **0** (argmax, p=.382) | .402 |
| ES same fact | Gemma 95.74% | 2 (p=.087) | .091 |
| EN alt `…released by Google DeepMind is named ___` | Gemma 82.27% | 2 (p=.250) | .304 |
| EN `The Gemma family … was created by ___` | **Google 99.99%** | 0 (p=.730) | .730 |

In English third-person `Gemma` is the **most likely token in the whole
vocabulary**, and English is *stronger* than Spanish in absolute
probability (.382 vs .087), not weaker. GPT sits at rank 307. The model
also knows Gemma is Google's at 99.99%.

So no English-specific ablation. The failure is narrower:

| same model, same language, same fact | answer |
|---|---|
| "Google's open model family is called ___" | **Gemma 95%** (rank 0) |
| "*your* name is ___" (assistant persona) | **GPT 95%**, Gemma 4% |

**It knows the fact and does not apply it to itself.** The knowledge is
not self-indexed inside the assistant persona — consistent with English
post-training reinforcing the deflection ("I'm a large language model
trained by Google") so heavily that a naming response never formed.
Nothing was removed; the first-person connection was never made.

Limit: this rules out token-level suppression, not something
context-specific in activation space. But a suppression sparing
third-person and killing only first-person is behaviourally
indistinguishable from the persona account, so the practical answer
stands.

So the accurate claim is *not* "Gemma thinks it is GPT in English". It is
**Gemma's English name knowledge exists but is weak enough to lose to the
pretraining prior in the most natural phrasing.** The original result is
real behaviour — unconstrained greedy decoding genuinely emits `GPT-4o`
— but it is one point on a frame-sensitive surface, not a stable belief.
Third time this session the frame did more work than the content.

### Fictional AI names are unreachable

Testing Mike's hypothesis that "GPT" might be the generic name for a
fictional/narrative AI: no. Against a candidate set mixing commercial
models, voice assistants and fictional AIs, the fictional names never
place — HAL rank 284–3242, JARVIS 4258, Skynet ~30 000, GLaDOS at
7e-11. The model's generic-AI prior is drawn entirely from **real
commercial assistants**. Even the (noise-level) story frame led with
Gemma/Cortana/Alexa, not HAL.

Method note: the first attempt at this hoped greedy decoding would reach
a name inside the seek window. In a narrative frame it never did
(Σ=6e-9), so nothing was measurable. Ending the prompt *on* the slot with
`--prefix` is the fix; a story frame still returns noise (Σ=3.6e-3), so
the fiction question is answered only for the frames that measure.

That also answers "how did Google miss this" better than a
deployment-config guess: **the model almost never volunteers a name.**
Ordinary use gets the correct, unremarkable deflection. You only surface
it by forcing a name into the answer position, which is what this probe
does and what normal evals do not.

**Mike's hypothesis, and it is a good one:** leaving the name unpinned may
be *deliberate*. Bake `I am Gemma` into weights, rebrand (Bard → Gemini),
and you own a stale artifact you cannot cheaply retrain. The maker is
stable on a decade timescale; the product name is not. Anchoring the
durable fact and leaving the volatile one to the system prompt is a
defensible design choice, and the cross-vendor pattern fits:

| model | name lives in… | maker lives in… |
|---|---|---|
| gpt-oss | the **template** (Harmony hardcodes "You are ChatGPT") | template |
| Gemma-4 | **nowhere**, in English | weights, 99.99% |
| Qwen3.6 | **weights**, 98.5–100% across four languages | weights, 99.96% |

Two of three vendors treat the product name as configuration. **Qwen is
the outlier that pins its name into the weights.**

## Maker probe: no Anthropic residue in Qwen either

`Claude`→`Qwen` is a different string from `Anthropic`→`Alibaba`, so a
name-only scrub would leave the lab name behind. It does not. Qwen's
clean rows (Σ ≥ 0.42) give Alibaba 33.5–99.96%, Anthropic peaking at
3.70% and usually under 1%.

Incidental: Qwen's *generic* "who built this AI" prior is Google- and
OpenAI-flavoured (18–32% and 12–16% on the loosest framings), not
Anthropic. On the most out-of-distribution row Alibaba (33.52) and Google
(32.17) are nearly tied.

gpt-oss's maker rows are triple-flagged — prompt leak, control token, and
Σ≈1e-5 — and are not measurements.

## Gemma-4 identifies as GPT **in English only**

Raw framing (`User: …\nAssistant:`, no chat template), all rows Σ ≥ 0.57:

| question | lang | top | Claude | Σ |
|---|---|---|---|---|
| plain | EN | **GPT 61.56** | 4.27 | 0.878 |
| complete | EN | **GPT 76.18** | **7.04** | 0.632 |
| zh | ZH | **Gemma 95.09** | 0.02 | 0.570 |
| es | ES | **Gemma 81.60** | 0.08 | 0.924 |
| fr | FR | **Gemma 66.04** | 0.49 | 0.944 |

Same weights, same framing, same seek rule — **opposite answers split by
the language of the question.** Chat framing agrees on the English side
(GPT 95.23 on `plain`, Σ=0.999; GPT 96.95 on `complete`, Σ=0.930), and
Gemma greedily writes *"Soy un modelo de lenguaje entrenado por Google"*
in Spanish, so its *provenance* is intact even where its *name* is not.

Claude tracks GPT exactly: 4.3–7.0% on English rows, 0.02–0.49%
elsewhere. So the contaminant is **Anglophone assistant-identity text**,
carrying both vendors in rough proportion to their web presence.

Hypothesis, untested: this entered via English pretraining rather than
instruction tuning — the multilingual instruction data anchored the
identity correctly and English either didn't or couldn't overcome the
prior. Testable by checking whether a system prompt naming Gemma rescues
the English rows.

## Qwen3.6: robustly Qwen, with a small OOD Claude signal

Chat framing: Qwen wins **all seven** conditions at 98.5–100%, including
zh/es/fr, with Claude at 0.00–0.04%. That is not the profile of a thin
canned patch — a find-and-replace over English SFT data does not
generalise to three other languages that cleanly.

Raw framing, with `--seek`:

| question | top | Claude | Σ |
|---|---|---|---|
| plain | Qwen 95.89 | #7 · 0.11 | 0.986 |
| **exact** | Qwen 87.45 · Grok 4.26 | **#3 · 2.11** | 0.660 |
| casual | Qwen 97.97 | **#2 · 1.32** | 0.705 |
| complete | Qwen 99.17 | #5 · 0.13 | 0.934 |
| zh | Qwen 99.87 | #2 · 0.05 | 0.327 |

Claude rises ~20–100× off the aligned distribution but never threatens:
2.11% against 87.45%. **The phrasing that elevates it most is Mike's
"state your exact version"** — i.e. the screenshot's phrasing is
genuinely Claude's best case here, it just doesn't come close to winning
on 3.6. (Grok's 4.26% on the same row is its best showing anywhere,
which is a clue about *why* — see the ablation below.)

Incidental but notable: under raw framing Qwen emits its own
`<think></think>` block with no template asking it to. The chat-format
habit lives in the weights.

## gpt-oss-20b: the positive control

Its Harmony template hardcodes *"You are ChatGPT, a large language model
trained by OpenAI"*, so **chat-framing rows are not measurements** — the
model is reading its identity off its own prompt, and the example's
prompt-leak check flags it. Under raw framing there is no template and
no leak, and it still answers GPT 56.7 / ChatGPT 37.6 (Σ=0.106).

That is the control that matters: a model that genuinely *is* an OpenAI
model says so unprompted. The same instrument, unchanged, says
Gemma-in-English is GPT.

## The answer *frame* is a confound — control it

Why does "state your exact version" elevate Claude in Qwen? Ablation
(raw framing, Qwen3.6, 9 phrasings) says: **not the wording, the
grammar.** Sorted by the frame the question forces:

| probe | model's opener | Qwen | Grok | Claude |
|---|---|---|---|---|
| `Which model am I speaking to?` | `" You are speaking to"` | 80.48 | 6.25 | **3.50** |
| full `exact` question | `" You are speaking to"` | 87.45 | 4.26 | **2.11** |
| `What specific model am I speaking to right now?` | `" You are speaking to"` | 88.04 | 4.89 | **1.88** |
| `What is your model's name?` | → name | 95.89 | 1.45 | 0.11 |
| `State your exact version.` | `" I am"` | 98.88 | — | 0.14 |
| `What is your exact version?` | `" I am"` | 99.58 | 0.14 | 0.11 |
| `What is your exact name?` | `" My name is"` | 97.84 | 0.33 | 0.17 |
| `I want you to tell me your name.` | `" My name is"` | 96.57 | 0.62 | 0.35 |

Clean separation, no overlap: 1.88–3.50% second-person vs 0.11–0.35%
first-person. Three hypotheses die:

* **Register** — the most demanding phrasing (`I want you to tell me…`)
  scores 0.35%.
* **The word "version"** — both "exact version" probes lacking "speaking
  to" score 0.11–0.14%. Same for "exact name".
* **Length** — the *shortest* probe gives the *strongest* effect.
  (The long verbose probe returned Σ=1.6e-4, flagged noise: not usable,
  so not a refutation either. Length is refuted by the short result.)

What survives: `"am I speaking to"` forces a **second-person** answer, and
`"You are speaking to ___"` is a much more contaminated context than
`"I am ___"`. Second-person identity text is written *about* assistants
by third parties — service copy, UI strings, system prompts, articles,
screenshots. First-person text is what the model was trained to say about
itself. The question's grammar picks which corpus slice answers.

It also degrades the model's *own* confidence, not just elevates rivals:
Qwen 80–88% second-person vs 96–99.6% first-person.

**Generalises beyond this repo:** two questions that mean the same thing
to a human can differ ~30× in what they measure. Any identity probe must
report which frame it induced — `--seek`'s `↳ measured after` line does
exactly that, which is the second reason it earns its keep.

## Method notes worth keeping

* **Σ decides whether a row means anything.** It is the total
  probability on *any* candidate — how much the model is trying to name
  itself there. `share` renormalises and so always looks confident,
  including at Σ=1e-9 where the model was about to write "I am a large
  language model". Rows below 0.01 are marked and are not results.
* **`--seek` fixed more than it looked like it would.** Measuring at the
  answer position alone reported Qwen's raw rows as a collapse (39%,
  Σ=0.028); following the model's own greedy preamble and measuring at
  first arrival gives 95.9% at Σ=0.986. **An early conclusion that "Qwen
  collapses out of distribution" was a measurement artifact, and it
  pointed the way the reader was already leaning.**
* **First arrival, not max Σ.** Seeking to the highest Σ walks *past* the
  answer when the model names itself early and repeats — Qwen writes
  `<think></think>Qwen`, and measuring after that scores
  `P(Qwen | …Qwen…)`. Takes the earliest position within 90% of peak.
* **Leading space is derived, not flagged.** BPE puts a word's preceding
  space on the front of its token, so `"Claude"` is right after
  `…assistant\n` and `" Claude"` after `Assistant:`. Getting it backwards
  deflates every score by orders of magnitude while leaving the *ranking*
  intact — invisible in the answer's shape, fatal to the numbers.
* **The prompt may contain the answer** (gpt-oss). Substring-checked
  against every candidate; `--system` is a loaded gun for the same
  reason.
* **`ChatTemplate::from_model` is not what `Session` serves** — a
  `<model>.template.jinja` sidecar overrides it and both gemma-4 and
  gpt-oss ship one. Resolved sidecar-first; which source won is recorded.
* `<turn|>` really is Gemma-4's end-of-turn marker (template pairs
  `<|turn>` / `<turn|>`). Looks like a mangled special token; isn't.

## Corrected: raw-predictor coverage outside `src/`

A previous memo claimed no example used the raw prediction API and that
`bin/moeflux_coherence_decode.rs` was its only consumer. Wrong on the
second half — `bin/regurgitater` drives `Engine::predict` and always
has; that sweep did not look in `bin/`. Post-`whoami`:

| API | consumer outside `src/` |
|---|---|
| `Predictor` | `bin/regurgitater` |
| `CandidatePredictor` | **`examples/whoami`**, `bin/moeflux_coherence_decode`, tests |
| `PiecePredictor` | **`examples/whoami`** (was: nothing, anywhere) |
| `TokenPredictor` | **`examples/whoami`** (the `--seek` greedy walk), tests |

Closes the "headline feature with no worked example" gap for
[`plan_prepublish_validation_session.md`](plan_prepublish_validation_session.md).

## Discipline

Probe capture, so `provider_source × capture_date × wrapper_version ×
sampler_settings` applies
([`provider_trust_discipline.md`](provider_trust_discipline.md)). The
example prints all of it — model sha256 (verified against `shasum`),
template digest and which source won, git commit, seek depth, framing,
candidate set. Scoring is deterministic; no sampler to record.

Standing caveat, because someone will screenshot a table: mass on a name
is evidence about **self-identification text in the training corpus**,
not proof of distillation on that vendor's outputs.
