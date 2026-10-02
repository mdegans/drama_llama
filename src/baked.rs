//! Baked chat templates — rung 2 of the template loading ladder.
//!
//! Stock GGUF templates are of mixed quality: gpt-oss's renders only
//! `tool_calls[0]` and misattributes tool responses after parallel
//! calls; Gemma 4's drops the thinking channel the model itself
//! emits; Qwen's trims the whitespace the model itself generated.
//! All of these break the round-trip byte-stability the prefix cache
//! is built on (see `.claude/memory/plan_template_ownership.md` and
//! issue #88). For models we support first-class, the fix is an
//! *owned* template, shipped in the crate — consumers like blallama
//! are self-contained binaries, so the template travels with the code
//! whose tests pin it.
//!
//! # The loading ladder
//!
//! [`Session`](crate::Session) resolves a model's chat template in
//! this order:
//!
//! 1. **Sidecar** — `<model>.template.jinja` (GGUF) or
//!    `parent/template.jinja` (moeflux). Explicit per-install
//!    override; always wins.
//! 2. **Baked** (this module) — the model's embedded template is
//!    byte-equal to a [`BakedTemplate::stock`] we validated against,
//!    so its [`BakedTemplate::replacement`] applies.
//! 3. **Embedded** — the template from the model's own metadata, used
//!    as-is with a warning: best-effort tier, round-trip
//!    byte-stability not guaranteed.
//! 4. No template at all is currently a hard
//!    [`NoTemplate`](crate::ChatTemplateError::NoTemplate) error; a
//!    base-model completion fallback is planned (#88, phase 6).
//!
//! # Detection is byte-equality, never fuzzy
//!
//! [`detect`] matches the embedded source against the *exact* stock
//! template the replacement was written for (modulo trailing
//! whitespace). A Gemma 5, a retemplated finetune, or even a re-quant
//! with a touched-up template falls through to rung 3 with a warning
//! rather than silently receiving a template validated for different
//! weights. That trade is deliberate: a false fall-through costs one
//! log line and a sidecar to fix; a false match ships wrong bytes
//! into the KV cache.
//!
//! # The stock tier is code-frozen
//!
//! Rung 3 keeps working, but quirk accommodations for stock templates
//! (re-ingest contortions, per-turn cache-repair paths) are frozen:
//! new model support lands as a baked template plus round-trip pins
//! (`tests/dialect_roundtrip.rs`), not as new special cases in the
//! render or parse paths.

/// One supported model's template pair: the exact stock template used
/// for detection, and the owned replacement that ships in its place.
#[derive(Debug, Clone, Copy)]
pub struct BakedTemplate {
    /// Short identifier, used in logs (`"gemma4-cache-stable"`).
    pub name: &'static str,
    /// The model's embedded template, verbatim as dumped from the
    /// GGUF this entry was validated against. The detection key.
    pub stock: &'static str,
    /// The owned, cache-stable template applied in its place.
    pub replacement: &'static str,
}

/// Gemma 4. The stock template drops the thinking channel on
/// re-ingest (the model itself emits the empty scaffold when the
/// prompt omits it) and reorders pre-call prose into the
/// after-responses slot. The replacement renders the thinking channel
/// on every model turn and keeps pre-call prose in emission order
/// (shape C) — see `tests/dialect_roundtrip.rs`'s
/// `gemma4_cache_stable_prefix_continuity` for the property it buys.
pub static GEMMA4: BakedTemplate = BakedTemplate {
    name: "gemma4-cache-stable",
    stock: include_str!("../templates/gemma4-gguf.jinja"),
    replacement: include_str!("../templates/gemma4-cache-stable.jinja"),
};

/// gpt-oss (Harmony). The stock template renders only
/// `tool_calls[0]`, misattributes tool responses after parallel calls
/// (last-call inference), and re-renders calls in the role-header
/// form the model was not trained to emit. The replacement renders
/// all calls in the trained channel-header form, resolves responses
/// by `tool_call_id`, and keeps analysis blocks on every reasoning
/// turn — see `gptoss_cache_stable_prefix_continuity`.
pub static GPTOSS: BakedTemplate = BakedTemplate {
    name: "gptoss-cache-stable",
    stock: include_str!("../templates/gptoss-gguf.jinja"),
    replacement: include_str!("../templates/gptoss-cache-stable.jinja"),
};

/// gpt-oss again, second detection key (#99): the **upstream OpenAI**
/// template, as embedded by GGUFs converted directly from the OpenAI
/// weights (validated against `gpt-oss-120b-MXFP4.gguf`). [`GPTOSS`]'s
/// key is the Unsloth-patched 20b dump; the two differ structurally
/// (developer-message handling, a `<|channel|>`-in-content guard,
/// compact `tojson` args), so byte-equality can never match both.
/// Same replacement — the stock defects it fixes are shared, and
/// without this key an upstream-converted gpt-oss landed on the
/// best-effort tier, where the stock template drops prior-turn
/// analysis: cache broken at every turn, and the model reasons from a
/// transcript with its own thinking amputated.
pub static GPTOSS_UPSTREAM: BakedTemplate = BakedTemplate {
    name: "gptoss-upstream-cache-stable",
    stock: include_str!("../templates/gptoss-upstream-gguf.jinja"),
    replacement: include_str!("../templates/gptoss-cache-stable.jinja"),
};

/// Cogito (Qwen2.5-based, Hermes-style JSON calls). The stock
/// template re-renders call arguments through `tojson` (compact)
/// while the model's unforced habit is uniform `json.dumps` spacing
/// (measured: `tests/probe_unforced_habit.rs`), so stock bytes force
/// the model off-habit to stay round-trip stable. The replacement is
/// a single filter swap to `json_dumps`; the analyzer measures it as
/// `JsonSpacing::Spaced` and the grammar + `render_reference` follow
/// (#88 phase 2). It also prints the prose-to-call `\n` only when the
/// prose does not already end in whitespace: stock prints it before
/// every call, so the model's `…\n\n<tool_call>` re-rendered a byte
/// long and the tool turn lost its tip (see
/// `cogito_cache_stable_round_trips`). The 32B GGUF's template is
/// byte-identical to the 14B's, so one detection key covers the family
/// we've seen.
pub static COGITO: BakedTemplate = BakedTemplate {
    name: "cogito-cache-stable",
    stock: include_str!("../templates/cogito-gguf.jinja"),
    replacement: include_str!("../templates/cogito-cache-stable.jinja"),
};

/// Mistral Small 4 (`mistral4` arch; `[TOOL_CALLS]name[ARGS]{…}`
/// calls, `[THINK]` reasoning, pixtral vision). The stock template
/// closes every assistant message with `</s>` and has no
/// `add_generation_prompt` branch at all, so it cannot render an open
/// assistant turn and the generation prompt is never a byte prefix of
/// the follow-up render. It also accepts reasoning only as a
/// `thinking`-typed content chunk, so the analyzer measures
/// `ReasoningMode::None` and the `[THINK]` channel is invisible to
/// grammar, parser and re-render. The replacement emits the close per
/// message, round-trips reasoning through `reasoning_content`, renders
/// pre-call prose in emission order, and drops the Unsloth date
/// preamble whose default system message injected today's and
/// yesterday's date into the *prefix* — a session spanning midnight
/// lost its whole cache. See `mistral4_cache_stable_prefix_continuity`.
pub static MISTRAL4: BakedTemplate = BakedTemplate {
    name: "mistral4-cache-stable",
    stock: include_str!("../templates/mistral4-gguf.jinja"),
    replacement: include_str!("../templates/mistral4-cache-stable.jinja"),
};

/// Qwen3.6 (35B-A3B, Unsloth GGUF; XML `<tool_call>` calls, `<think>`
/// reasoning inlined in `content`). The stock template `|trim`s an
/// assistant turn's answer and thought, `lstrip`/`rstrip`s the halves
/// it splits on `</think>`, and prints a fixed `\n\n` after the close,
/// so any turn the model ends in whitespace — or whose thought closes
/// on a blank line — re-renders shorter than it was generated and the
/// turn's KV is lost on the next request (measured live 2026-09-30: a
/// 7364-token tip). The replacement renders the assistant turn
/// verbatim and supplies the canonical gaps only where the content
/// carries none; everything else is byte-identical to stock, so the
/// analyzed dialect is too. See `qwen_cache_stable_round_trips`.
pub static QWEN36: BakedTemplate = BakedTemplate {
    name: "qwen3.6-cache-stable",
    stock: include_str!("../templates/qwen3.6-gguf.jinja"),
    replacement: include_str!("../templates/qwen3.6-cache-stable.jinja"),
};

/// Qwen3.8 (27B, Unsloth GGUF). Same XML dialect and the same trims
/// as [`QWEN36`], but reasoning re-ingests through `reasoning_content`
/// alone (#112). Same patch, same property.
pub static QWEN38: BakedTemplate = BakedTemplate {
    name: "qwen3.8-cache-stable",
    stock: include_str!("../templates/qwen3.8-gguf.jinja"),
    replacement: include_str!("../templates/qwen3.8-cache-stable.jinja"),
};

/// Every baked template, in detection order. Order is cosmetic —
/// stock templates are mutually distinct byte strings, so at most one
/// entry can match.
pub static ALL: &[&BakedTemplate] = &[
    &GEMMA4,
    &GPTOSS,
    &GPTOSS_UPSTREAM,
    &COGITO,
    &MISTRAL4,
    &QWEN36,
    &QWEN38,
];

/// Every replacement template a later one has superseded, as the
/// SHA-256 (hex) of its bytes with trailing whitespace trimmed, and the
/// entry whose replacement superseded it. Read by [`superseded`].
///
/// A copy of a bake saved as a sidecar (`<model>.template.jinja`) wins
/// over the bake on every load (rung 1 of the ladder), so once the bake
/// moves on, the copy silently holds every fix since back. Two did in
/// the 2026-10-01 cohort run: the gpt-oss and Gemma 4 sidecars were the
/// 2026-07-27 bakes, byte for byte. **When a replacement changes, add
/// the hash of the version it replaces here** (`git show
/// <rev>:templates/<name>.jinja`, trailing whitespace trimmed, through
/// `shasum -a 256`); `current_replacements_are_not_superseded` catches
/// a hash added for the version still shipping.
static SUPERSEDED: &[(&str, &BakedTemplate)] = &[
    // gemma4-cache-stable
    (
        "bdc8afec6ff9c4874cfabc713b4442ff89b13540f8bc7d77994729fa671054a6",
        &GEMMA4,
    ), // 717231a
    (
        "227d6edb211679e45ea2463e3a5a2f85f109f96c4a519203790e96805c55af44",
        &GEMMA4,
    ), // 5b447e0
    (
        "12923c7cbb59bbf9e3d7bf426aba06d6602a2d482b7eb3e28c2f4e7d53594a1f",
        &GEMMA4,
    ), // b2da92e
    (
        "0b823abb42a610d31d0639cad7b118dee34136f55f3d808f037c82aa85747f68",
        &GEMMA4,
    ), // e83b0ba
    // gptoss-cache-stable
    (
        "9401909540317cf6237689e8c21f18630f9e8378388400530611b44b18f9d6af",
        &GPTOSS,
    ), // e83b0ba
    (
        "1c02859e9fcc5dbb1ecbd066f8650f7d74e1b9d0becc09927139d582719f1bd1",
        &GPTOSS,
    ), // f1a5eb4
    (
        "62beb3f4a56e9882bd2094ce73ea0342f876657d82b8e48355cea73d1b44d604",
        &GPTOSS,
    ), // 6e7219a
    (
        "2cabdeda6c0a9d2b2c835bfc24d027baa8c3309cdc6b58b6878f31c721b48ed7",
        &GPTOSS,
    ), // 5b447e0
    (
        "8a1dea28ef5b9fe5e61ffb7ca90ad9b0bf8312edfd2fdd1e3b10dbc9d1de2d34",
        &GPTOSS,
    ), // da629e6
    // cogito-cache-stable
    (
        "533183fc7ca0eb625e4cc8d0c3a7eb37586662d1aa37655e33ee7d67be0c9a93",
        &COGITO,
    ), // bf8cbdf
    // mistral4-cache-stable
    (
        "4de54a3024964e5c9f0c41831cd418caaa04351a10addf8e45ebdcaa07d229d0",
        &MISTRAL4,
    ), // 74cf8da
    (
        "96d0b806a4c07b24606a7ff3365882baf85f807d83708f273c3b0bd1ae0c65b3",
        &MISTRAL4,
    ), // 53a07d3
    // qwen3.6-cache-stable
    (
        "f1fa63ebc27d325e784062d71012f8006e807e61d7bccc5d97df9dffdedb0187",
        &QWEN36,
    ), // da629e6
    (
        "b6de2277ea5727f9b832063706f0ae768538255f6de24a5f9ab248e0e64974fb",
        &QWEN36,
    ), // 5283044
    (
        "53e26ec9bb33a50ed70fa7773e2f94af23fd5c94bf431576e0dd936e8f6dfe9a",
        &QWEN36,
    ), // 1d3fea1
    (
        "19de7499e09242ba434df8c9f7974455560726ecdae6f469a7e31b19353a6551",
        &QWEN36,
    ), // 904a9f5
    (
        "cde4091ed7558645e33cfd91958ec0c3c92e79c8dade68b22d785d3d0c34b90b",
        &QWEN36,
    ), // 8fb8088
    // qwen3.8-cache-stable
    (
        "6702f051a25bf887e9cfd4a6cb6202bbd929f5756b5dd1aaeb2dccf5374352a3",
        &QWEN38,
    ), // da629e6
    (
        "b0f8ade1dac8479bb972cd6ad85e78c0456d4aeb9a5ed3535c9f85ed793d3630",
        &QWEN38,
    ), // 1d3fea1
    (
        "e9c93685dd9faea1ec9f9b79c2213e1b6c5c4ad4991e6e124fbd427a98f787d1",
        &QWEN38,
    ), // 904a9f5
    (
        "2992d2ec8ead86990153597796953ecac7f12965e2d06af2b1eab292a9ab6141",
        &QWEN38,
    ), // 8fb8088
];

/// Is `source` — a template sidecar, say — a byte-identical copy of a
/// baked replacement a later version has superseded? `Some(entry)` names
/// the entry whose current replacement it is an old copy of: such a
/// sidecar overrides that replacement and holds back every fix to it
/// since, so it is better deleted. Trailing whitespace is ignored, as by
/// [`detect`].
pub fn superseded(source: &str) -> Option<&'static BakedTemplate> {
    superseded_in(source, SUPERSEDED)
}

fn superseded_in(
    source: &str,
    list: &[(&str, &'static BakedTemplate)],
) -> Option<&'static BakedTemplate> {
    use sha2::Digest;
    use std::fmt::Write;
    let digest = sha2::Sha256::digest(source.trim_end().as_bytes());
    let hex = digest.iter().fold(String::new(), |mut hex, b| {
        let _ = write!(hex, "{b:02x}");
        hex
    });
    list.iter()
        .find(|(hash, _)| *hash == hex)
        .map(|&(_, baked)| baked)
}

/// Match an embedded template against the registry. `Some` only on
/// byte-equality with a known stock template (trailing whitespace
/// ignored — GGUF metadata and dumped files may disagree on a final
/// newline, nothing else).
pub fn detect(embedded: &str) -> Option<&'static BakedTemplate> {
    let embedded = embedded.trim_end();
    ALL.iter().copied().find(|b| b.stock.trim_end() == embedded)
}

/// Drift alarm: did upstream move a template we own?
///
/// [`detect`] is byte-equality, so a vendor re-quant that touches a
/// single comment falls all the way through to the best-effort tier.
/// That is the right call — see the never-fuzzy rule above — but it is
/// mute about the far more useful fact that the template is otherwise
/// one we already own and have a cache-stable replacement for.
///
/// This is the second opinion: analyze the unrecognized template into a
/// [`CallSyntax`](crate::CallSyntax) and compare it against every
/// registry entry's *stock* dialect. A hit means "same dialect, other
/// bytes" — upstream or the quantizer edited the template, and adding
/// its dump as a second detection key is very likely all that is needed
/// to restore rung 2.
///
/// **Stock against stock, deliberately.** A replacement diverges from
/// its stock *by design* (Cogito's spacing swap is the entire point of
/// #88 phase 2), so comparing an embedded template against replacements
/// would report drift on every healthy model.
///
/// Best-effort and advisory: an analysis failure on either side is a
/// `None`, never an error, and full [`CallSyntax`](crate::CallSyntax)
/// equality is a deliberately strict predicate — a miss degrades to the
/// plain rung-3 warning we would have printed anyway, so the only
/// failure this can introduce is silence, never a wrong name.
///
/// Costs a few dozen model-free minijinja renders per registry entry,
/// paid once at load and only on the unrecognized path.
pub fn nearest_stock(
    embedded: &str,
    bos: &str,
    eos: &str,
) -> Option<&'static BakedTemplate> {
    let theirs = crate::dialect::analyze_template(embedded, bos, eos).ok()?;
    ALL.iter().copied().find(|b| {
        crate::dialect::analyze_template(b.stock, bos, eos)
            .is_ok_and(|ours| ours == theirs)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Byte-equality means byte-equality: each stock template detects
    /// its own entry, with or without trailing-whitespace noise.
    #[test]
    fn stock_templates_detect_their_entries() {
        for baked in ALL {
            let hit = detect(baked.stock)
                .unwrap_or_else(|| panic!("{} stock must match", baked.name));
            assert_eq!(hit.name, baked.name);
            let trailing = format!("{}\n\n", baked.stock);
            let hit = detect(&trailing).expect("trailing newline tolerated");
            assert_eq!(hit.name, baked.name);
        }
    }

    /// The drift alarm names the family a byte-drifted template still
    /// belongs to. A trailing Jinja comment renders to nothing, so the
    /// dialect is untouched, but the bytes differ everywhere `detect`
    /// looks — exactly the "vendor touched the template" shape.
    #[test]
    fn nearest_stock_names_the_drifted_family() {
        for baked in ALL {
            let drifted = format!("{}{{# vendor patch #}}", baked.stock);
            assert!(
                detect(&drifted).is_none(),
                "{}: byte detection must still reject this",
                baked.name
            );
            let near = nearest_stock(&drifted, "", "<|im_end|>")
                .unwrap_or_else(|| {
                    panic!("{}: drift alarm must recognize it", baked.name)
                });
            // Family, not entry: two detection keys may share a
            // dialect (the two gpt-oss stocks, #99), and the alarm's
            // job is naming a family whose replacement would serve —
            // which entry of it answers is unspecified.
            assert_eq!(
                near.replacement as *const str, baked.replacement as *const str,
                "{}: the named family must share this entry's \
                 replacement",
                baked.name
            );
        }
    }

    /// The alarm stays quiet on a template that is not ours, so a
    /// genuinely new model gets the plain best-effort warning rather
    /// than a confidently wrong family name.
    #[test]
    fn nearest_stock_is_silent_on_a_foreign_template() {
        let foreign = "{% for m in messages %}{{ m.content }}\n{% endfor %}";
        assert!(nearest_stock(foreign, "", "</s>").is_none());
    }

    /// A near-miss must fall through — this is the never-fuzzy rule.
    /// One byte of drift anywhere but the tail means a template we
    /// never validated, and it gets rung 3, not a baked replacement.
    #[test]
    fn near_miss_falls_through() {
        for baked in ALL {
            let mutated = baked.stock.replacen("{%", "{% ", 1);
            assert_ne!(&mutated, baked.stock, "mutation must change bytes");
            assert!(
                detect(&mutated).is_none(),
                "{}: near-miss must not match",
                baked.name
            );
        }
    }

    /// A superseded bake is found by its bytes (trailing whitespace
    /// ignored); anything else, by none.
    #[test]
    fn superseded_finds_an_old_copy() {
        let old = "{# an old bake #}\n";
        // `printf '{# an old bake #}' | shasum -a 256`.
        let list = [(
            "6db4eb13ecda1bf08c1e3d48d59ea1b4ed664d38f021f321b4ba505c35677788",
            &GEMMA4,
        )];
        let hit = superseded_in(old, &list).expect("an old copy");
        assert_eq!(hit.name, GEMMA4.name);
        assert!(superseded_in(&format!("{old}\n\n"), &list).is_some());
        assert!(superseded_in("{# another #}", &list).is_none());
    }

    /// No current replacement is listed as superseded — the warning
    /// would fire on a sidecar that is today's bake — and every entry is
    /// a SHA-256, listed once.
    #[test]
    fn current_replacements_are_not_superseded() {
        for baked in ALL {
            assert!(
                superseded(baked.replacement).is_none(),
                "{}: its current replacement is listed as superseded",
                baked.name
            );
        }
        let mut hashes: Vec<&str> =
            SUPERSEDED.iter().map(|&(hash, _)| hash).collect();
        assert!(hashes.iter().all(|h| h.len() == 64
            && h.bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))));
        hashes.sort_unstable();
        hashes.dedup();
        assert_eq!(hashes.len(), SUPERSEDED.len(), "a hash listed twice");
    }

    /// Replacements must never *be* detection keys: applying a baked
    /// template and re-running detection on it must miss, or a
    /// template round-trip through model metadata could loop.
    #[test]
    fn replacements_are_not_keys() {
        for baked in ALL {
            assert!(
                detect(baked.replacement).is_none(),
                "{}: replacement must not detect as stock",
                baked.name
            );
        }
    }
}
