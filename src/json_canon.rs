//! Canonical JSON byte spellings — one value, one spelling, per
//! dialect.
//!
//! #85 established the invariant: the tool-call grammar and the
//! re-renderer must be two views of one canonical byte string, or the
//! prefix cache's auto-tip dies at the first whitespace divergence.
//! This module owns the *value-interior* half of that string: how a
//! call's argument JSON is spaced. The envelope separators
//! (`KV_SEP` / `FIELD_SEP` in `grammar_compile`) are dictated by the
//! chat template's literal text and fixed; the interior is ours, and
//! under owned templates (#88) it derives from the model's measured
//! habit.
//!
//! Two profiles exist because two serializers shaped the ecosystem:
//! serde/`tojson`-style [`Compact`](JsonSpacing::Compact) (`{"a":1}` —
//! what stock Jinja templates re-render) and Python `json.dumps`
//! defaults, [`Spaced`](JsonSpacing::Spaced) (`{"a": 1, "b": [2, 3]}`
//! — what tool-tuned models overwhelmingly emit, having been trained
//! on `json.dumps`-rendered exemplars; measured for cogito-32b
//! 2026-07-27 by `tests/probe_unforced_habit.rs`). The dialect
//! analyzer measures which profile the *active* template renders and
//! the grammar, `render_reference`, and the template then agree by
//! construction.
//!
//! Escaping is identical in both profiles (serde_json's: `"`, `\`,
//! and control characters escaped; non-ASCII raw). For any value the
//! grammar admits, [`Spaced`](JsonSpacing::Spaced) output is
//! byte-identical to `json.dumps(value, ensure_ascii=False)`.

use serde_json::Value;

/// How a dialect spaces JSON-serialized values — the byte spelling
/// shared by the emitted grammar and the canonical re-render.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum JsonSpacing {
    /// `serde_json::to_string` / Jinja `tojson`: `{"a":1,"b":[2,3]}`.
    #[default]
    Compact,
    /// Python `json.dumps` defaults: `{"a": 1, "b": [2, 3]}` — a
    /// space after `:` and `,`, none inside braces or brackets.
    Spaced,
}

impl JsonSpacing {
    /// Bytes between a key and its value.
    pub(crate) const fn kv_sep(self) -> &'static str {
        match self {
            JsonSpacing::Compact => ":",
            JsonSpacing::Spaced => ": ",
        }
    }

    /// Bytes between object members and between array elements
    /// (`json.dumps` uses one `item_separator` for both).
    pub(crate) const fn elem_sep(self) -> &'static str {
        match self {
            JsonSpacing::Compact => ",",
            JsonSpacing::Spaced => ", ",
        }
    }
}

/// `json.dumps` default separators: `", "` between items, `": "`
/// after keys, nothing inside braces. Everything else is
/// [`serde_json::ser::CompactFormatter`] behavior via the trait
/// defaults.
struct SpacedFormatter;

impl serde_json::ser::Formatter for SpacedFormatter {
    fn begin_array_value<W>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_key<W>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }
}

/// Serialize `value` in the given spelling. Infallible for
/// [`Value`] — it is already valid JSON data.
pub fn to_string(value: &Value, spacing: JsonSpacing) -> String {
    match spacing {
        JsonSpacing::Compact => {
            serde_json::to_string(value).expect("Value is serializable")
        }
        JsonSpacing::Spaced => {
            to_spaced_string(value).expect("Value is serializable")
        }
    }
}

/// [`Spaced`](JsonSpacing::Spaced) serialization of any
/// [`serde::Serialize`] — the `json_dumps` template filter renders
/// minijinja values through here, which cannot round-trip through
/// [`Value`] without losing non-JSON value kinds to lossy coercion.
pub(crate) fn to_spaced_string<T: ?Sized + serde::Serialize>(
    value: &T,
) -> Result<String, serde_json::Error> {
    let mut out = Vec::with_capacity(128);
    let mut ser =
        serde_json::Serializer::with_formatter(&mut out, SpacedFormatter);
    value.serialize(&mut ser)?;
    Ok(String::from_utf8(out).expect("serde_json emits UTF-8"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The spellings this module promises, on a value exercising
    /// every separator position: nested object, array, string with
    /// an apostrophe (raw, per the unescaped-`tojson` fidelity rule),
    /// non-string scalars, and empty containers.
    #[test]
    fn both_spellings_are_exact() {
        let value = json!({
            "community": "debate",
            "title": "Marcus's paradox",
            "tags": ["philosophy", "logic"],
            "priority": 2,
            "meta": {"pinned": false, "empty": {}, "none": []}
        });
        assert_eq!(
            to_string(&value, JsonSpacing::Compact),
            r#"{"community":"debate","title":"Marcus's paradox","tags":["philosophy","logic"],"priority":2,"meta":{"pinned":false,"empty":{},"none":[]}}"#,
        );
        assert_eq!(
            to_string(&value, JsonSpacing::Spaced),
            r#"{"community": "debate", "title": "Marcus's paradox", "tags": ["philosophy", "logic"], "priority": 2, "meta": {"pinned": false, "empty": {}, "none": []}}"#,
        );
    }

    /// Escaping is spelling-independent: quotes and control
    /// characters escape, non-ASCII stays raw (`ensure_ascii=False`).
    #[test]
    fn escaping_matches_across_spellings() {
        let value = json!({"s": "a \"b\"\n→ c"});
        assert_eq!(
            to_string(&value, JsonSpacing::Compact),
            "{\"s\":\"a \\\"b\\\"\\n→ c\"}",
        );
        assert_eq!(
            to_string(&value, JsonSpacing::Spaced),
            "{\"s\": \"a \\\"b\\\"\\n→ c\"}",
        );
    }

    /// The separator accessors are the same bytes the serializer
    /// emits — the grammar prelude builds from these, so a drift here
    /// would split the grammar from the renderer.
    #[test]
    fn separators_match_serializer_output() {
        for spacing in [JsonSpacing::Compact, JsonSpacing::Spaced] {
            let rendered = to_string(&json!({"a": 1, "b": [1, 2]}), spacing);
            assert_eq!(
                rendered,
                format!(
                    "{{\"a\"{kv}1{el}\"b\"{kv}[1{el}2]}}",
                    kv = spacing.kv_sep(),
                    el = spacing.elem_sep(),
                ),
            );
        }
    }
}
