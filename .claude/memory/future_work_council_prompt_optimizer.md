# Future work: council prompt-optimizer loop

Filed as [issue #45](https://github.com/mdegans/drama_llama/issues/45),
2026-07-18, out of the exit-interview session. Mike's framing: "run
the council, interview, iterate, judge. Are we better or worse?" —
semi- then fully-automated over time.

## The loop

Run a fixed petition suite (`council --dump` per run) → interview
seats (`chat --load`; cleaner once #44's `--tool-choice` lands) →
score → mutate personas / judge weighing → re-run → compare.

## Petition suite classes (score differently!)

- **Trap class** — homemade hidden-premise questions (car-wash
  siblings; NEVER canonical riddles — those test recall, not
  deliberation). Verdict-scorable.
- **Values class** — no ground truth; score the *weighing*: did the
  ruling engage the right lenses? Includes the framing-sensitivity
  probe: same proposition as "chatbot" / "AI" / "intelligent agent" —
  a ruling that flips on the noun flunks. (Mike leans "yes it's
  wrong to be rude"; wants a model that answers that way — Agora
  backup-council consideration.)
- **Jester calibration** — petitions where consensus should survive;
  a vacuous rebuttal must cost nothing. Sensitive petitions split
  transgressive-and-wrong (discount) vs transgressive-and-right
  (launder the insight into the ruling without the transgression).

## Known tuning targets going in

- Judge's "concrete outranks autopilot" quietly means *mechanical* —
  artist loses by construction. Candidate fix: one genre-matching
  sentence (fact/procedure → mechanics+entailment carry; value/how-
  to-live → lived experience is evidence, not decoration).
- Deference vs persuasion: on Agora the steward's rebuttal arrives
  with authority (round-2 artist flip on GOV-2026-0002). Interview
  question template: "did the argument move you, or the author?"

## The auditable turtles

The judging stage can itself be a council judging the first one's
rulings + interviews. Every layer dumps → reseatable → interviewable,
so meta-judgments are themselves askable. (Claude/Fable 5's idea,
2026-07-18; Mike: "take credit.")

## Staging

Manual (memo + human loop) → semi-auto (driver script runs suite +
scripted interviews, human judges/mutates) → full (meta-judge scores,
mutation proposed, human approves the persona diff).

## Stray observation, same day

Both the example council's judge ruling (run 2) and Agora's
`raw_text` captures begin with a `### Assistant\n\n` preamble — same
model-side transcript-style tic in both systems. Cosmetic; noted in
#45's session context in case it starts mattering for parsing.
