# Plan — polish sessions (post-Phase-2, pre-publish)

Mike, 2026-07-16 (end of the Phase 2 landing session): the next few
sessions — apart from moeflux work — focus on **review, code quality,
tests, and convenience accessors so examples get shorter**. "Pretty."

## Ergonomics north star: the misanthropic structured-output shape

misanthropic's structured-generation example core:

```rust
let prompt = args.common.configure(
    Prompt::default()
        .model(Id::Haiku45)
        .structured_output::<CommitClassification>()
        .system(system),
)
.add_message((Role::User, format!("Classify this diff:\n\n```diff\n{diff}\n```")))?;

let response = client.message(&prompt).await?;
let classification: CommitClassification = response.json()?;
```

Because drama_llama uses the same misanthropic types, a
`response::Message` from `Session::complete_response` already has
`.json()` — so the same terse shape should work locally. Known wart:
our structured-generation example accumulates output from a string,
which is unnecessary. Audit the examples for that pattern and for
places where a small Session/Response accessor would delete
boilerplate.

## Scope notes

- Also on deck: the standing full review pass (code quality, tests),
  Phase 3 of the sampler split (see
  `design_sampler_config_state_split.md`), and the pre-publish
  validation checklist (`plan_prepublish_validation_session.md`).
- Moeflux inventory 2026-07-16: `just test moeflux` 6/6 green
  (cross-backend parity, both coherence, both pollution, smoke) —
  despite churn since the last run (balerion work, image support),
  nothing broke. No repair backlog from that quarter as of Phase 2.
