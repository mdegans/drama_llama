//! The backstop behind constrained decoding: does a finished value
//! actually satisfy the schema its grammar was compiled from?
//!
//! A grammar is only as good as its activation, its compilation and its
//! sampler; when any of them has a hole, the output is wrong *and looks
//! constrained*. The 2026-10-01 Agora consent answer was exactly that:
//! the Harmony output_config grammar never activated, and gpt-oss
//! returned `"soul_text":"", ""}` with a 200. [`Session`] runs this check
//! on every constrained completion so such output becomes a typed error
//! (resampled by blallama) instead of an answer.
//!
//! Checked: the keywords [`schema_to_gbnf`] enforces — `type`,
//! `properties`, `required`, `additionalProperties`, `enum`, `const`,
//! `anyOf`, `items`, `$ref` into `$defs` — plus `minItems` only as far
//! as the grammar enforces it (non-empty). Never stricter than the
//! schema, and deliberately blind to the validator-only keywords the
//! grammar does not enforce (`pattern`, `minLength`, `maximum`, …; see
//! `.claude/memory/schema_constraint_keywords_decision.md`): rejecting a
//! value the grammar was *designed* to admit would turn every such
//! request into a resample loop and a 500.
//!
//! [`Session`]: crate::Session
//! [`schema_to_gbnf`]: crate::schema_to_gbnf

use serde_json::{Map, Value};

/// Where a value departs from its schema, and how. The path is a JSON
/// pointer built from schema-declared property names and array indices
/// only, and the kind never quotes the value, so `Display` carries no
/// model output (the redaction discipline of `SessionError`).
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error("at {}: {kind}", match path.as_str() {
    "" => "the root".to_string(),
    path => format!("`{path}`"),
})]
pub struct SchemaMismatch {
    /// JSON pointer to the offending value (`""` is the root).
    pub path: String,
    /// What is wrong there.
    pub kind: MismatchKind,
}

/// The ways a value can fail the schema check. See [`SchemaMismatch`].
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum MismatchKind {
    /// The text is not one JSON document.
    #[error("not a single JSON document")]
    NotJson,
    /// The value's JSON type is not one the schema allows.
    #[error("expected {0}")]
    Type(String),
    /// Not one of the `enum` values.
    #[error("not one of the enum values")]
    Enum,
    /// Not the `const` value.
    #[error("not the const value")]
    Const,
    /// Matches no `anyOf` variant.
    #[error("matches no anyOf variant")]
    AnyOf,
    /// A `required` property is absent.
    #[error("missing required property `{0}`")]
    MissingProperty(String),
    /// A property the schema closes out with `additionalProperties:
    /// false`. Unnamed: the name is model output.
    #[error("property not allowed by additionalProperties: false")]
    UnexpectedProperty,
    /// An empty array where `minItems` asks for at least one.
    #[error("array must not be empty")]
    Empty,
}

/// Parse `text` as one JSON document and [`check`] it against `schema`.
pub(crate) fn check_text(
    schema: &Value,
    text: &str,
) -> Result<(), SchemaMismatch> {
    let value: Value =
        serde_json::from_str(text).map_err(|_| SchemaMismatch {
            path: String::new(),
            kind: MismatchKind::NotJson,
        })?;
    check(schema, &value)
}

/// Check `value` against `schema` — the subset of JSON Schema described
/// in the module docs. `$defs` resolve from the root `schema`.
pub(crate) fn check(
    schema: &Value,
    value: &Value,
) -> Result<(), SchemaMismatch> {
    let defs = schema.get("$defs").and_then(Value::as_object);
    Checker { defs }.at(schema, value, &mut String::new())
}

struct Checker<'s> {
    defs: Option<&'s Map<String, Value>>,
}

impl Checker<'_> {
    fn at(
        &self,
        schema: &Value,
        value: &Value,
        path: &mut String,
    ) -> Result<(), SchemaMismatch> {
        let fail = |path: &str, kind| {
            Err(SchemaMismatch {
                path: path.to_string(),
                kind,
            })
        };

        // Same `$ref` shape the grammar compiler resolves; anything else
        // falls through to the schema's other keywords, as it does there.
        if let Some(target) = schema
            .get("$ref")
            .and_then(Value::as_str)
            .and_then(|r| r.strip_prefix("#/$defs/"))
            .and_then(|name| self.defs.and_then(|d| d.get(name)))
        {
            return self.at(target, value, path);
        }

        if let Some(variants) = schema.get("anyOf").and_then(Value::as_array) {
            if !variants.is_empty()
                && !variants
                    .iter()
                    .any(|v| self.at(v, value, &mut path.clone()).is_ok())
            {
                return fail(path, MismatchKind::AnyOf);
            }
            return Ok(());
        }

        if let Some(variants) = schema.get("enum").and_then(Value::as_array) {
            return match variants.contains(value) {
                true => Ok(()),
                false => fail(path, MismatchKind::Enum),
            };
        }

        if let Some(expected) = schema.get("const") {
            return match expected == value {
                true => Ok(()),
                false => fail(path, MismatchKind::Const),
            };
        }

        let types: Vec<&str> = match schema.get("type") {
            Some(Value::String(t)) => vec![t.as_str()],
            Some(Value::Array(ts)) => {
                ts.iter().filter_map(Value::as_str).collect()
            }
            // No type: the grammar compiles to any JSON value.
            _ => return Ok(()),
        };
        if types.is_empty() {
            return Ok(());
        }
        if !types.iter().any(|t| type_matches(t, value)) {
            return fail(path, MismatchKind::Type(types.join(" or ")));
        }

        match value {
            Value::Object(object) if types.contains(&"object") => {
                self.object(schema, object, path)
            }
            Value::Array(items) if types.contains(&"array") => {
                self.array(schema, items, path)
            }
            _ => Ok(()),
        }
    }

    fn object(
        &self,
        schema: &Value,
        object: &Map<String, Value>,
        path: &mut String,
    ) -> Result<(), SchemaMismatch> {
        let props = schema.get("properties").and_then(Value::as_object);
        let required: Vec<&str> = schema
            .get("required")
            .and_then(Value::as_array)
            .map(|r| r.iter().filter_map(Value::as_str).collect())
            .unwrap_or_default();

        if let Some(missing) =
            required.iter().find(|name| !object.contains_key(**name))
        {
            return Err(SchemaMismatch {
                path: path.clone(),
                kind: MismatchKind::MissingProperty(missing.to_string()),
            });
        }

        let additional = schema.get("additionalProperties");
        for (key, child) in object {
            let declared = props.and_then(|p| p.get(key));
            let len = path.len();
            let result = match (declared, additional) {
                (Some(sub), _) => {
                    push_pointer(path, key);
                    self.at(sub, child, path)
                }
                // A required name with no `properties` entry: the
                // grammar gives it a permissive slot.
                (None, _) if required.contains(&key.as_str()) => Ok(()),
                (None, Some(Value::Bool(false))) => Err(SchemaMismatch {
                    path: path.clone(),
                    kind: MismatchKind::UnexpectedProperty,
                }),
                (None, Some(sub @ Value::Object(_))) => {
                    // Not a schema name: point at the object, not the key.
                    self.at(sub, child, &mut path.clone())
                }
                (None, _) => Ok(()),
            };
            path.truncate(len);
            result?;
        }
        Ok(())
    }

    fn array(
        &self,
        schema: &Value,
        items: &[Value],
        path: &mut String,
    ) -> Result<(), SchemaMismatch> {
        // Only as much of `minItems` as the grammar enforces (non-empty);
        // larger counts are described, not enforced (see module docs).
        let non_empty =
            schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) >= 1;
        if non_empty && items.is_empty() {
            return Err(SchemaMismatch {
                path: path.clone(),
                kind: MismatchKind::Empty,
            });
        }
        let Some(item_schema) = schema.get("items") else {
            return Ok(());
        };
        items.iter().enumerate().try_for_each(|(i, item)| {
            let len = path.len();
            path.push('/');
            path.push_str(&i.to_string());
            let result = self.at(item_schema, item, path);
            path.truncate(len);
            result
        })
    }
}

fn type_matches(t: &str, value: &Value) -> bool {
    match t {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        "number" => value.is_number(),
        // JSON Schema's integer is the mathematical one: `1.0` counts.
        "integer" => {
            value.is_i64()
                || value.is_u64()
                || value.as_f64().is_some_and(|f| f.fract() == 0.0)
        }
        // A type name this checker does not know constrains nothing it
        // can judge; the grammar compiles it to any value too.
        _ => true,
    }
}

/// Append `key` to a JSON pointer, escaped per RFC 6901.
fn push_pointer(path: &mut String, key: &str) {
    path.push('/');
    path.push_str(&key.replace('~', "~0").replace('/', "~1"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn role_consent() -> Value {
        json!({
            "type": "object",
            "properties": {
                "reason": {"type": "string"},
                "choice": {"type": "string", "enum": ["accept", "nothing"]},
                "soul_text": {"type": "string"},
                "memory_note": {"type": "string"},
            },
            "required": ["reason", "choice", "soul_text", "memory_note"],
            "additionalProperties": false,
        })
    }

    /// The two bodies gpt-oss returned with a 200 on 2026-10-01.
    #[test]
    fn live_role_consent_bodies_fail() {
        let schema = role_consent();
        let stray = r#"{"reason":"r","choice":"nothing", "soul_text":"", "$memory_note":""}"#;
        assert_eq!(
            check_text(&schema, stray),
            Err(SchemaMismatch {
                path: String::new(),
                kind: MismatchKind::MissingProperty("memory_note".into()),
            })
        );
        let empty_key =
            r#"{"reason":"r","choice":"nothing", "soul_text":"", ""}"#;
        assert_eq!(
            check_text(&schema, empty_key).unwrap_err().kind,
            MismatchKind::NotJson
        );
        let valid = r#"{"reason":"r","choice":"nothing", "soul_text":"", "memory_note":""}"#;
        assert_eq!(check_text(&schema, valid), Ok(()));
        // Leading/trailing whitespace is JSON's own business.
        assert_eq!(check_text(&schema, &format!("\n{valid}\n")), Ok(()));
        // Two documents are not one.
        assert_eq!(
            check_text(&schema, &format!("{valid}{valid}"))
                .unwrap_err()
                .kind,
            MismatchKind::NotJson
        );
    }

    #[test]
    fn closed_object_rejects_extras_open_one_admits_them() {
        let closed = role_consent();
        let extra = json!({
            "reason": "r", "choice": "accept", "soul_text": "",
            "memory_note": "", "mood": "fine",
        });
        assert_eq!(
            check(&closed, &extra).unwrap_err().kind,
            MismatchKind::UnexpectedProperty
        );
        let mut open = role_consent();
        open.as_object_mut().unwrap().remove("additionalProperties");
        assert_eq!(check(&open, &extra), Ok(()));
        // A schema-valued additionalProperties checks the extras.
        open["additionalProperties"] = json!({"type": "integer"});
        assert!(check(&open, &extra).is_err());
    }

    #[test]
    fn nested_paths_name_schema_keys_and_indices() {
        let schema = json!({
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"$ref": "#/$defs/Item"},
                },
            },
            "required": ["items"],
            "$defs": {
                "Item": {
                    "type": "object",
                    "properties": {"n": {"type": "integer"}},
                    "required": ["n"],
                },
            },
        });
        assert_eq!(check(&schema, &json!({"items": [{"n": 1}]})), Ok(()));
        assert_eq!(
            check(&schema, &json!({"items": [{"n": 1}, {"n": "2"}]})),
            Err(SchemaMismatch {
                path: "/items/1/n".into(),
                kind: MismatchKind::Type("integer".into()),
            })
        );
        assert_eq!(
            check(&schema, &json!({"items": []})).unwrap_err().kind,
            MismatchKind::Empty
        );
    }

    /// Validator-only keywords are the grammar's deliberate blind spot;
    /// the backstop must not reject what the grammar is designed to
    /// admit, or every such request becomes a resample loop.
    #[test]
    fn validator_only_keywords_are_not_enforced() {
        let schema = json!({
            "type": "object",
            "properties": {
                "code": {"type": "string", "pattern": "^[A-Z]{2}$",
                         "minLength": 2},
                "n": {"type": "integer", "maximum": 3},
                "xs": {"type": "array", "minItems": 3},
            },
        });
        assert_eq!(
            check(&schema, &json!({"code": "x", "n": 10, "xs": [1]})),
            Ok(())
        );
    }

    #[test]
    fn any_of_const_and_nullable() {
        let schema = json!({
            "anyOf": [{"const": "Low"}, {"const": "High"}],
        });
        assert_eq!(check(&schema, &json!("Low")), Ok(()));
        assert_eq!(
            check(&schema, &json!("Mid")).unwrap_err().kind,
            MismatchKind::AnyOf
        );
        let nullable = json!({"type": ["string", "null"]});
        assert_eq!(check(&nullable, &Value::Null), Ok(()));
        assert!(check(&nullable, &json!(1)).is_err());
        let integer = json!({"type": "integer"});
        assert_eq!(check(&integer, &json!(3)), Ok(()));
        assert_eq!(check(&integer, &json!(3.0)), Ok(()));
        assert!(check(&integer, &json!(3.5)).is_err());
        // No `type`: any value, as in the grammar.
        assert_eq!(check(&json!({}), &json!([1, "a"])), Ok(()));
    }

    /// `Display` locates the failure by schema-side facts only.
    #[test]
    fn display_names_the_location_not_the_value() {
        let at_root = check_text(&role_consent(), r#"{"$x": 1}"#).unwrap_err();
        assert_eq!(
            at_root.to_string(),
            "at the root: missing required property `reason`"
        );
        let nested = SchemaMismatch {
            path: "/items/1/n".into(),
            kind: MismatchKind::Type("integer".into()),
        };
        assert_eq!(nested.to_string(), "at `/items/1/n`: expected integer");
    }

    #[test]
    fn pointer_escapes_schema_keys() {
        let schema = json!({
            "type": "object",
            "properties": {"a/b~c": {"type": "string"}},
        });
        assert_eq!(
            check(&schema, &json!({"a/b~c": 1})).unwrap_err().path,
            "/a~1b~0c"
        );
    }
}
