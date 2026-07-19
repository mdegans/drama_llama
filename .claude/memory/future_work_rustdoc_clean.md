# Future work: `cargo doc` is not lint-clean

Filed as [issue #47](https://github.com/mdegans/drama_llama/issues/47)
(2026-07-19, noticed while fixing #44).

Rustdoc reports **11 broken intra-doc links** + **8 public-doc→private-item**
warnings. Reproduce:

```sh
RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --no-deps --lib
```

Audited on default features only — the full documented set
(`webchat,cli,stats,toml,serde,egui`) may surface more.

## Buckets (see #47 for the full table + locations)

- **Unresolved links:** `Self::from_path` ×5 in `session/mod.rs`
  (Session has no `from_path` — renamed to `from_path_sync` /
  `from_path_with_n_ctx`); `SamplerConfig` ×2 + a `crate::crate::…`
  typo in `sidecar.rs`; `field` placeholder + `sidecar::load_sample_options`
  in `session/mod.rs`; `AfterTools` in `chat_template.rs` (reference-style
  def scoped to a sibling variant's doc).
- **Private-item links:** 8 `pub` items linking private symbols
  (`grammar_filter`, `apply_sample_repetition_ngram`, `analyzer`, …) —
  widen visibility, drop the link, or plain code font.

Already fixed on `v0.8.0` (in #44): stale `Prompt::functions` →
`Prompt::tools` (misanthropic renamed the field/builder to `tools`).

## The durable fix

Gate it in CI / a `just doc` recipe so it can't regress:

```sh
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps \
  --features "webchat,cli,stats,toml,serde,egui"
```

Delete this memo when #47 lands (per the resolved-future-work rule).
