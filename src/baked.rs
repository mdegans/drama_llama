//! Baked chat templates — rung 2 of the template loading ladder.
//!
//! Stock GGUF templates are of mixed quality: gpt-oss's renders only
//! `tool_calls[0]` and misattributes tool responses after parallel
//! calls; Gemma 4's drops the thinking channel the model itself
//! emits. Both break the round-trip byte-stability the prefix cache
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
/// (#88 phase 2). The 32B GGUF's template is byte-identical to the
/// 14B's, so one detection key covers the family we've seen.
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

/// Every baked template, in detection order. Order is cosmetic —
/// stock templates are mutually distinct byte strings, so at most one
/// entry can match.
pub static ALL: &[&BakedTemplate] =
    &[&GEMMA4, &GPTOSS, &GPTOSS_UPSTREAM, &COGITO, &MISTRAL4];

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
