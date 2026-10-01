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

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use serde_json::{Map, Value};

use crate::grammar_compile::Defs;

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
    let value = parse_document(text).ok_or(SchemaMismatch {
        path: String::new(),
        kind: MismatchKind::NotJson,
    })?;
    check(schema, &value)
}

/// `text` as one JSON document, read no stricter than the grammar writes
/// it. serde_json refuses a number whose magnitude overflows `f64`
/// ("number out of range"), and [`JSON_GRAMMAR`]'s `number` admits one:
/// its integer part is unbounded (only the exponent, at two digits, and
/// `integer`, at eighteen, are capped), so a long enough run of digits is
/// grammar-legal and would be a `NotJson` on every draw. Such a number
/// reads as `0` — the checker judges only its type, and a number it
/// stays. The grammar's other refusals match serde_json's (a lone
/// surrogate escape is unwritable: `\uD800` must pair with a low
/// surrogate), so nothing else needs reading around.
///
/// [`JSON_GRAMMAR`]: crate::grammar_compile::JSON_GRAMMAR
fn parse_document(text: &str) -> Option<Value> {
    serde_json::from_str(text).ok().or_else(|| {
        // Syntax first: serde's skip-parse checks structure, not ranges.
        serde_json::from_str::<serde::de::IgnoredAny>(text).ok()?;
        serde_json::from_str(&finite_numbers(text)).ok()
    })
}

/// `text` with every number outside a string that overflows `f64`
/// replaced by `0`. Assumes `text` is syntactically JSON.
fn finite_numbers(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) =
        rest.find(|c: char| c == '"' || c == '-' || c.is_ascii_digit())
    {
        out.push_str(&rest[..at]);
        rest = &rest[at..];
        let len = match rest.as_bytes()[0] {
            b'"' => string_len(rest),
            _ => rest
                .find(|c: char| {
                    !(c.is_ascii_digit()
                        || matches!(c, '-' | '+' | '.' | 'e' | 'E'))
                })
                .unwrap_or(rest.len()),
        };
        let (token, tail) = rest.split_at(len);
        let overflows = !token.starts_with('"')
            && token.parse::<f64>().is_ok_and(|n| n.is_infinite());
        out.push_str(if overflows { "0" } else { token });
        rest = tail;
    }
    out.push_str(rest);
    out
}

/// The length of the JSON string `text` opens with, quotes included.
fn string_len(text: &str) -> usize {
    let bytes = text.as_bytes();
    let mut i = 1;
    while i < bytes.len() {
        match bytes[i] {
            b'\\' => i += 2,
            b'"' => return i + 1,
            _ => i += 1,
        }
    }
    bytes.len()
}

/// Check `value` against `schema` — the subset of JSON Schema described
/// in the module docs. `$defs` resolve from the root `schema`.
///
/// Bounded by [`step_budget`]: a check that runs out passes, with a
/// warning — the grammar already constrained the bytes, and a false
/// mismatch would be a resample loop and a 500 for valid output.
pub(crate) fn check(
    schema: &Value,
    value: &Value,
) -> Result<(), SchemaMismatch> {
    let (verdict, steps, budget) = check_counted(schema, value);
    if steps <= budget {
        return verdict;
    }
    tracing::warn!(
        target: "drama_llama::schema_check",
        event = "schema_check_budget",
        steps,
        budget,
        "schema check ran out of steps; accepting the value, which the \
         grammar constrained",
    );
    Ok(())
}

/// [`check`]'s verdict before the budget is applied, the steps it took
/// (past the budget, it stopped judging there) and the budget.
pub(crate) fn check_counted(
    schema: &Value,
    value: &Value,
) -> (Result<(), SchemaMismatch>, usize, usize) {
    let defs = Defs::new(schema.get("$defs").and_then(Value::as_object));
    let mut checker = Checker {
        defs,
        memo: HashMap::new(),
        names: HashMap::new(),
        depth: 0,
        steps: 0,
        budget: step_budget(value),
        speculative: 0,
    };
    let verdict = checker.at(schema, value, &mut String::new(), None);
    (verdict, checker.steps, checker.budget)
}

/// The most steps [`check`] takes on `value` — one per subschema judged
/// and per `enum` member, distinct `required` name or distinct `type`
/// compared, and one per entry of a schema's `required` and `type`
/// arrays, the once their distinct names are gathered: 2^20, plus 2^14
/// per JSON value in it.
///
/// A value is judged by the subschemas that can reach it, and inside
/// [`SchemaLimits`](crate::SchemaLimits) those are few. An `anyOf`'s
/// variants are tried in turn, nested ones multiplying, an object's
/// property is judged by every object variant that declares it, and an
/// `enum`'s members are compared one by one — but every one of those is
/// counted by [`SchemaLimits::max_width`](crate::SchemaLimits::max_width),
/// as is each alternation itself, so they are at most 2048 per value;
/// and each def is judged at most twice per value (its verdict is
/// memoized, inside an `anyOf` and out), at most 1024 defs. Under 8192
/// per value, then: 2^14 is room to spare. A `required` or `type` array
/// is gathered once per check, whatever its duplicates: its entries are
/// values the schema counts toward
/// [`SchemaLimits::max_nodes`](crate::SchemaLimits::max_nodes), 2^17 at
/// most, inside the 2^20. Its distinct names are what each value is
/// judged against, and those the width counts (a `required` name is a
/// property; an `"object"` named a hundred thousand times is one). So a
/// request inside the limits never runs out
/// (`schema_check_at_the_limits_stays_in_budget`), while a schema past
/// them (only a caller that skips the measure can send one) costs
/// seconds, not hours.
fn step_budget(value: &Value) -> usize {
    let mut nodes = 0usize;
    let mut stack = vec![value];
    while let Some(v) = stack.pop() {
        nodes += 1;
        match v {
            Value::Array(items) => stack.extend(items),
            Value::Object(map) => stack.extend(map.values()),
            _ => {}
        }
    }
    (1 << 20) + nodes.saturating_mul(1 << 14)
}

/// How deep [`Checker::at`] may nest before it stops judging. A value
/// is at most serde_json's 128 levels, but each level can also descend
/// a def's `anyOf`s and a chain of `$ref`s, and the checker must never
/// overflow the stack on a client's schema. Past the cap a value
/// passes: the backstop goes lenient, never wrong.
const MAX_DEPTH: usize = 256;

struct Checker<'s> {
    defs: Defs<'s>,
    /// A def's verdict on a value, by `(def, the value's address)`:
    /// the same value always sits at the same path, so the verdict is
    /// reusable whole. Without it, `anyOf`s of `$ref`s fan out
    /// exponentially in the chain's length.
    memo: HashMap<(usize, usize, bool), Result<(), SchemaMismatch>>,
    /// Each schema's distinct `type` and `required` names, by the
    /// schema's address, gathered the first time a value meets it — a
    /// `required` naming one property a hundred thousand times was a
    /// hundred thousand lookups for every object judged.
    names: HashMap<usize, Rc<Names>>,
    /// The current nesting of [`Self::at`].
    depth: usize,
    /// Steps taken ([`step_budget`]).
    steps: usize,
    /// Past this many steps every judgement passes.
    budget: usize,
    /// How many `anyOf`s the current judgement is a variant of. Inside
    /// one a mismatch is only ever discarded (the `anyOf` reports its
    /// own), so it is built without its path or names: a failing
    /// variant costs a step, not a copy of a path as long as the
    /// schema's keys.
    speculative: usize,
}

/// A schema's distinct `type` and `required` names, in order.
struct Names {
    /// `None` when the schema has no `type` (or a malformed one).
    types: Option<Vec<String>>,
    required: Vec<String>,
    /// `required`, for lookups.
    required_set: HashSet<String>,
}

impl Names {
    /// `schema`'s names, and the entries read to gather them.
    fn gather(schema: &Value) -> (Self, usize) {
        let distinct = |names: &[Value]| {
            let mut seen = HashSet::new();
            let names: Vec<String> = names
                .iter()
                .filter_map(Value::as_str)
                .filter(|n| seen.insert(*n))
                .map(str::to_string)
                .collect();
            names
        };
        let (types, typed) = match schema.get("type") {
            Some(Value::String(t)) => (Some(vec![t.clone()]), 1),
            Some(Value::Array(ts)) => (Some(distinct(ts)), ts.len()),
            _ => (None, 0),
        };
        let listed = schema.get("required").and_then(Value::as_array);
        let required = listed.map_or_else(Vec::new, |r| distinct(r));
        let read = typed + listed.map_or(0, Vec::len);
        let required_set = required.iter().cloned().collect();
        let names = Self {
            types,
            required,
            required_set,
        };
        (names, read)
    }
}

impl Checker<'_> {
    /// `schema`'s [`Names`], gathered once per check.
    fn names(&mut self, schema: &Value) -> Rc<Names> {
        let key = schema as *const Value as usize;
        if let Some(names) = self.names.get(&key) {
            return names.clone();
        }
        let (names, read) = Names::gather(schema);
        self.steps = self.steps.saturating_add(read);
        let names = Rc::new(names);
        self.names.insert(key, names.clone());
        names
    }

    /// A mismatch of `kind` at `path` — or, speculative, at none.
    fn mismatch(&self, path: &str, kind: MismatchKind) -> SchemaMismatch {
        SchemaMismatch {
            path: match self.speculative {
                0 => path.to_string(),
                _ => String::new(),
            },
            kind,
        }
    }
}

impl Checker<'_> {
    /// Check `value` at `path` against `schema`. `left_of` is the def
    /// whose body `schema` is at the left of — no byte of `value`
    /// consumed since — which decides whether a `$ref` closes a cycle
    /// (see [`Defs`]).
    fn at(
        &mut self,
        schema: &Value,
        value: &Value,
        path: &mut String,
        left_of: Option<usize>,
    ) -> Result<(), SchemaMismatch> {
        if self.depth >= MAX_DEPTH || self.steps > self.budget {
            return Ok(());
        }
        self.steps += 1;
        self.depth += 1;
        let result = self.at_inner(schema, value, path, left_of);
        self.depth -= 1;
        result
    }

    fn at_inner(
        &mut self,
        schema: &Value,
        value: &Value,
        path: &mut String,
        left_of: Option<usize>,
    ) -> Result<(), SchemaMismatch> {
        let fail = |checker: &Self, path: &str, kind| {
            Err(checker.mismatch(path, kind))
        };

        // Same `$ref` shape the grammar compiler resolves; anything else
        // falls through to the schema's other keywords, as it does there.
        // A reference closing a left cycle is `value` there: anything.
        if let Some(id) = self.defs.target(schema) {
            let Some(id) = self.defs.resolve(left_of, id) else {
                return Ok(());
            };
            // Keyed by mode too: a speculative verdict has no path.
            let key =
                (id, value as *const Value as usize, self.speculative > 0);
            if let Some(verdict) = self.memo.get(&key) {
                return verdict.clone();
            }
            let verdict = self.at(self.defs.schema(id), value, path, Some(id));
            self.memo.insert(key, verdict.clone());
            return verdict;
        }

        if let Some(variants) = schema.get("anyOf").and_then(Value::as_array) {
            // Each variant leaves `path` as it found it.
            self.speculative += 1;
            let any = variants.is_empty()
                || variants
                    .iter()
                    .any(|v| self.at(v, value, path, left_of).is_ok());
            self.speculative -= 1;
            return match any {
                true => Ok(()),
                false => fail(self, path, MismatchKind::AnyOf),
            };
        }

        if let Some(variants) = schema.get("enum").and_then(Value::as_array) {
            self.steps = self.steps.saturating_add(variants.len());
            return match variants.contains(value) {
                true => Ok(()),
                false => fail(self, path, MismatchKind::Enum),
            };
        }

        if let Some(expected) = schema.get("const") {
            return match expected == value {
                true => Ok(()),
                false => fail(self, path, MismatchKind::Const),
            };
        }

        let names = self.names(schema);
        // No type: the grammar compiles to any JSON value.
        let Some(types) = names.types.as_ref().filter(|t| !t.is_empty()) else {
            return Ok(());
        };
        self.steps = self.steps.saturating_add(types.len());
        if !types.iter().any(|t| type_matches(t, value)) {
            let expected = match self.speculative {
                0 => types.join(" or "),
                _ => String::new(),
            };
            return fail(self, path, MismatchKind::Type(expected));
        }
        let names_type = |name: &str| types.iter().any(|t| t == name);

        match value {
            Value::Object(object) if names_type("object") => {
                self.object(schema, &names, object, path)
            }
            Value::Array(items) if names_type("array") => {
                self.array(schema, items, path)
            }
            _ => Ok(()),
        }
    }

    fn object(
        &mut self,
        schema: &Value,
        names: &Names,
        object: &Map<String, Value>,
        path: &mut String,
    ) -> Result<(), SchemaMismatch> {
        let props = schema.get("properties").and_then(Value::as_object);
        let required = &names.required;
        self.steps = self.steps.saturating_add(required.len());

        if let Some(missing) = required
            .iter()
            .find(|name| !object.contains_key(name.as_str()))
        {
            let missing = match self.speculative {
                0 => missing.to_string(),
                _ => String::new(),
            };
            return Err(
                self.mismatch(path, MismatchKind::MissingProperty(missing))
            );
        }

        // Each undeclared key looks itself up here, not in the list.
        let required_set = &names.required_set;
        let additional = schema.get("additionalProperties");
        for (key, child) in object {
            let declared = props.and_then(|p| p.get(key));
            let len = path.len();
            let result = match (declared, additional) {
                (Some(sub), _) => {
                    push_pointer(path, key);
                    self.at(sub, child, path, None)
                }
                // A required name with no `properties` entry: the
                // grammar gives it a permissive slot.
                (None, _) if required_set.contains(key.as_str()) => Ok(()),
                (None, Some(Value::Bool(false))) => {
                    Err(self.mismatch(path, MismatchKind::UnexpectedProperty))
                }
                (None, Some(sub @ Value::Object(_))) => {
                    // Not a schema name: point at the object, not the key.
                    self.at(sub, child, path, None)
                }
                (None, _) => Ok(()),
            };
            path.truncate(len);
            result?;
        }
        Ok(())
    }

    fn array(
        &mut self,
        schema: &Value,
        items: &[Value],
        path: &mut String,
    ) -> Result<(), SchemaMismatch> {
        // Only as much of `minItems` as the grammar enforces (non-empty);
        // larger counts are described, not enforced (see module docs).
        let non_empty =
            schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) >= 1;
        if non_empty && items.is_empty() {
            return Err(self.mismatch(path, MismatchKind::Empty));
        }
        let Some(item_schema) = schema.get("items") else {
            return Ok(());
        };
        items.iter().enumerate().try_for_each(|(i, item)| {
            let len = path.len();
            path.push('/');
            path.push_str(&i.to_string());
            let result = self.at(item_schema, item, path, None);
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

    /// What the grammar can write, the checker must read: an integer part
    /// too long for `f64` is legal in [`crate::grammar_compile::JSON_GRAMMAR`]'s
    /// `number` (serde_json: "number out of range"), and was a `NotJson`
    /// on every draw — a deterministic 500. Still judged by type; real
    /// syntax errors and the out-of-grammar forms stay refused.
    #[test]
    fn grammar_legal_overflowing_numbers_are_json() {
        let schema = json!({
            "type": "object",
            "properties": {
                "n": {"type": "number"},
                "s": {"type": "string"},
            },
            "required": ["n", "s"],
        });
        let big = format!("1{}", "0".repeat(400));
        for n in [big.clone(), format!("-{big}.5"), format!("{big}e99")] {
            let text = format!(r#"{{"n": {n}, "s": "1{big} \"x\""}}"#);
            assert_eq!(check_text(&schema, &text), Ok(()), "{n:.12}");
        }
        // Inside a string it is text, untouched.
        let as_string = json!({"type": "string"});
        assert_eq!(check_text(&as_string, &format!(r#""{big}""#)), Ok(()));
        // Still a number, so still not a string.
        let wrong = format!(r#"{{"n": 1, "s": {big}}}"#);
        assert_eq!(
            check_text(&schema, &wrong).unwrap_err(),
            SchemaMismatch {
                path: "/s".into(),
                kind: MismatchKind::Type("string".into()),
            }
        );
        // Broken syntax is still not JSON.
        assert_eq!(
            check_text(&schema, &format!(r#"{{"n": {big},}}"#))
                .unwrap_err()
                .kind,
            MismatchKind::NotJson
        );
        // A lone surrogate is not grammar-legal, and stays refused.
        assert_eq!(
            check_text(&as_string, r#""\ud800""#).unwrap_err().kind,
            MismatchKind::NotJson
        );
    }

    /// The grammar refuses what serde_json does, for the forms that
    /// matter: a lone surrogate escape and a three-digit exponent are
    /// unwritable, while a surrogate pair is fine.
    #[test]
    fn the_grammar_cannot_write_a_lone_surrogate_or_1e999() {
        let source =
            format!("root ::= value\n{}", crate::grammar_compile::JSON_GRAMMAR);
        let admits = |text: &str| {
            let mut state = crate::GrammarState::from_source(&source)
                .expect("grammar parses");
            state.advance_bytes(text.as_bytes()).is_ok() && state.is_complete()
        };
        assert!(admits(r#""\ud83c\udf53""#));
        assert!(!admits(r#""\ud800""#));
        assert!(!admits(r#""\udc00""#));
        assert!(!admits("1e999"));
        assert!(admits("1e99"));
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

    /// Past its step budget the check stops judging and passes the
    /// value, valid or not: 300 values each compared against a
    /// 30,000-member `enum` (far past the width limit) are nine million
    /// steps against a budget of six million.
    #[test]
    fn over_its_budget_the_check_passes() {
        let members: Vec<usize> = (0..30_000).collect();
        let schema = json!({"type": "array", "items": {"enum": members}});
        let mut items = vec![json!(29_999); 300];
        let (verdict, steps, budget) =
            check_counted(&schema, &Value::Array(items.clone()));
        assert!(verdict.is_ok() && steps > budget, "{steps} of {budget}");
        items[299] = json!(-1);
        let invalid = Value::Array(items);
        let (_, steps, budget) = check_counted(&schema, &invalid);
        assert!(steps > budget);
        assert_eq!(check(&schema, &invalid), Ok(()));
        // Within it, the same invalid value is caught.
        let short = Value::Array(vec![json!(-1)]);
        assert_eq!(
            check(&schema, &short).unwrap_err().kind,
            MismatchKind::Enum
        );
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
