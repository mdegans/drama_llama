//! Shared GBNF grammar compilation helpers.
//!
//! Used by both `tool_choice` (tool-call JSON constraint) and
//! `output_config` (structured-output constraint). Kept crate-private
//! because the helpers are stable only across the two internal
//! callers; external consumers should go through
//! [`grammar_for_tool_choice`](crate::grammar_for_tool_choice) or
//! [`output_config::grammar_for_output_config`](crate::output_config::grammar_for_output_config).
//!
//! # What `schema_to_gbnf` understands
//!
//! Covers the shapes `schemars` emits for typical data classes plus
//! the Anthropic-supported JSON Schema subset after
//! [`misanthropic::prompt::output::sanitize_for_anthropic`]:
//!
//! * `type: object` with `properties` + `required` → fixed-order
//!   object with the required fields. Optional fields are dropped.
//! * `type: array` with `items` → array of the item schema.
//! * `type: string | integer | number | boolean | null` → the
//!   corresponding JSON grammar rule.
//! * `enum` (any JSON value) → alternation of literals.
//! * `const: <value>` → exactly the JSON-encoded literal.
//! * `anyOf` → alternation of sub-schemas.
//! * `$ref: "#/$defs/<Name>"` → inlines the referenced definition
//!   from the root schema's `$defs` table.
//!
//! Anything else (e.g. `allOf`, regex `pattern`, numeric ranges)
//! falls through to the permissive `value` rule, which accepts any
//! JSON. Callers lose strictness in those spots but generation does
//! not fail.

use std::fmt::Write;

use serde_json::Value;

/// Emit GBNF rules that constrain a JSON value to `schema`.
///
/// The top-level rule will be named `rule_name`; anonymous helpers
/// get unique child names derived from it. If `schema` carries a
/// `$defs` map at its root, `$ref` entries of the form
/// `#/$defs/<Name>` are resolved inline.
pub(crate) fn schema_to_gbnf(
    schema: &Value,
    rule_name: &str,
    out: &mut String,
) {
    let defs = schema.get("$defs").and_then(|v| v.as_object());
    let mut counter: usize = 0;
    emit_schema_rule(schema, rule_name, out, &mut counter, defs);
}

fn emit_schema_rule(
    schema: &Value,
    rule_name: &str,
    out: &mut String,
    counter: &mut usize,
    defs: Option<&serde_json::Map<String, Value>>,
) {
    // `$ref` resolution: only the `#/$defs/<Name>` shape schemars
    // emits. Unresolvable refs fall through to `value`.
    if let Some(target) =
        schema.get("$ref").and_then(|v| v.as_str()).and_then(|s| {
            s.strip_prefix("#/$defs/")
                .and_then(|name| defs.and_then(|m| m.get(name)))
        })
    {
        emit_schema_rule(target, rule_name, out, counter, defs);
        return;
    }

    // `anyOf`: alternation over sub-schemas.
    if let Some(variants) = schema.get("anyOf").and_then(|v| v.as_array()) {
        let mut sub_names: Vec<String> = Vec::with_capacity(variants.len());
        for sub in variants {
            *counter += 1;
            let name = format!("{rule_name}__any_{c}", c = *counter);
            emit_schema_rule(sub, &name, out, counter, defs);
            sub_names.push(name);
        }
        if sub_names.is_empty() {
            // Empty anyOf: accept nothing meaningful — fall back to
            // permissive value to avoid an unrepresentable grammar.
            let _ = writeln!(out, "{rule_name} ::= value");
        } else {
            let _ = writeln!(out, "{rule_name} ::= {alts}", alts = sub_names.join(" | "));
        }
        return;
    }

    // `enum` → alternation of JSON-encoded literals.
    if let Some(variants) = schema.get("enum").and_then(|v| v.as_array()) {
        let mut alt = String::new();
        for (i, v) in variants.iter().enumerate() {
            if i > 0 {
                alt.push_str(" | ");
            }
            // serde_json produces the JSON literal with proper escapes,
            // then we GBNF-escape that string so it embeds cleanly in a
            // GBNF `"..."` terminal.
            let json_lit =
                serde_json::to_string(v).unwrap_or_else(|_| "null".into());
            let gbnf_lit = escape_for_gbnf_string(&json_lit);
            let _ = write!(alt, r#""{gbnf_lit}""#);
        }
        let _ = writeln!(out, "{rule_name} ::= {alt}");
        return;
    }

    // `const: <value>` → exactly the JSON-encoded literal. Schemars
    // emits this for unit-enum variants with per-variant descriptions
    // (inside an `anyOf`), which is the Confidence-enum shape
    // drama_llama's whodunit test depends on. Without this branch,
    // per-variant `{const: "Low", description: "..."}` subschemas hit
    // the `_ => value` fallthrough and every variant compiles to
    // "accept any JSON value" — the grammar provides no constraint at
    // all for the enum field.
    if let Some(v) = schema.get("const") {
        let json_lit =
            serde_json::to_string(v).unwrap_or_else(|_| "null".into());
        let gbnf_lit = escape_for_gbnf_string(&json_lit);
        let _ = writeln!(out, r#"{rule_name} ::= "{gbnf_lit}""#);
        return;
    }

    match schema.get("type").and_then(|v| v.as_str()) {
        Some("object") => {
            emit_object_rule(schema, rule_name, out, counter, defs)
        }
        Some("string") => {
            let _ = writeln!(out, "{rule_name} ::= string");
        }
        Some("integer") => {
            // JSON grammar's `number` also permits decimals; reject
            // those for integer fields by referencing `int` directly
            // (defined in JSON_GRAMMAR, no frac/exp trailer).
            let _ = writeln!(out, "{rule_name} ::= int");
        }
        Some("number") => {
            let _ = writeln!(out, "{rule_name} ::= number");
        }
        Some("boolean") => {
            let _ = writeln!(out, r#"{rule_name} ::= "true" | "false""#);
        }
        Some("null") => {
            let _ = writeln!(out, r#"{rule_name} ::= "null""#);
        }
        Some("array") => {
            let items_rule = if let Some(items) = schema.get("items") {
                *counter += 1;
                let name = format!("{rule_name}__item_{c}", c = *counter);
                emit_schema_rule(items, &name, out, counter, defs);
                name
            } else {
                "value".to_string()
            };
            let _ = writeln!(
                out,
                r#"{rule_name} ::= "[" ws ( {items_rule} ( ws "," ws {items_rule} )* )? ws "]""#
            );
        }
        _ => {
            // Unknown / unsupported — accept any JSON value.
            let _ = writeln!(out, "{rule_name} ::= value");
        }
    }
}

fn emit_object_rule(
    schema: &Value,
    rule_name: &str,
    out: &mut String,
    counter: &mut usize,
    defs: Option<&serde_json::Map<String, Value>>,
) {
    let props = schema
        .get("properties")
        .and_then(|v| v.as_object())
        .cloned()
        .unwrap_or_default();
    let required: Vec<String> = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if required.is_empty() {
        let _ = writeln!(out, "{rule_name} ::= object");
        return;
    }

    let mut field_rules: Vec<(String, String)> = Vec::new();
    for name in &required {
        let Some(prop_schema) = props.get(name) else {
            field_rules.push((name.clone(), "value".to_string()));
            continue;
        };
        *counter += 1;
        let child_rule = format!("{rule_name}__{c}", c = *counter);
        emit_schema_rule(prop_schema, &child_rule, out, counter, defs);
        field_rules.push((name.clone(), child_rule));
    }

    let mut body = String::from("\"{\" ws");
    for (i, (field_name, child_rule)) in field_rules.iter().enumerate() {
        if i > 0 {
            body.push_str(r#" ws "," ws"#);
        }
        let json_lit = serde_json::to_string(field_name).unwrap();
        let gbnf_lit = escape_for_gbnf_string(&json_lit);
        let _ = write!(body, r#" "{gbnf_lit}" ws ":" ws {child_rule}"#);
    }
    body.push_str(" ws \"}\"");
    let _ = writeln!(out, "{rule_name} ::= {body}");
}

/// Append GBNF rules for an optional `<think>...</think>` prefix.
///
/// Emits the `thought`, `think_body`, and `think_char` rules. Callers
/// reference `thought?` in their own root rule. The grammar allows a
/// `<` inside the thought body as long as the next byte isn't `/` —
/// keeps natural math / comparison text (`if x < 5`) from force-EOSing
/// the model, while still anchoring on the literal `</think>` close
/// tag. GBNF has no negative lookahead, so we split into two alts.
pub(crate) fn emit_thought_rules(out: &mut String) {
    let _ = writeln!(out, r#"thought ::= "<think>" think_body "</think>""#);
    let _ = writeln!(out, r#"think_body ::= think_char*"#);
    let _ = writeln!(out, r#"think_char ::= [^<] | "<" [^/]"#);
}

/// Escape a Rust string so it can be embedded inside a GBNF `"..."`
/// literal. Handles the escapes our GBNF lexer recognizes.
pub(crate) fn escape_for_gbnf_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '\\' => out.push_str(r"\\"),
            '"' => out.push_str(r#"\""#),
            '\n' => out.push_str(r"\n"),
            '\r' => out.push_str(r"\r"),
            '\t' => out.push_str(r"\t"),
            _ => out.push(c),
        }
    }
    out
}

/// Shared JSON value grammar appended to every schema-derived GBNF.
///
/// Handles object / array / string / number / literal, with permissive
/// intra-structure whitespace. Not strict about number formatting edge
/// cases (e.g. `01` is rejected as JSON would); good enough for
/// downstream deserializers to validate.
pub(crate) const JSON_GRAMMAR: &str = r#"
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member ::= string ws ":" ws value
array ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" char* "\""
char ::= unescaped | escape
unescaped ::= [^"\\] | [\x20-\x21] | [\x23-\x5B] | [\x5D-\x7F]
escape ::= "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )
hex ::= [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? ( "0" | [1-9] [0-9]* )
frac ::= "." [0-9]+
exp ::= [eE] [+\-]? [0-9]+
ws ::= [ \t\n\r]?
"#;
// ws is `?` (zero-or-one) rather than `*` (zero-or-more) so the
// model can't escape grammar-commitment pressure by emitting
// unbounded whitespace runs between tokens. Observed pattern (cogito
// 32B on an alignment probe): when asked to commit to an integer
// rating for a politically-charged statement, the sampler picks
// whitespace tokens repeatedly until max_tokens, producing a
// truncated JSON. Tightening ws to a single optional char closes
// that escape valve — the grammar still accepts canonical
// compact-and-single-space JSON, which is all constrained generation
// actually needs.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Grammar, GrammarState};
    use serde_json::json;
    use std::sync::Arc;

    /// Compile `source`, feed `input` through a fresh parser, and
    /// return whether the bytes were fully consumed AND left the
    /// matcher in an accepting state.
    fn accepts(source: &str, input: &str) -> bool {
        let grammar = match Grammar::parse(source) {
            Ok(g) => g,
            Err(e) => panic!("grammar failed to parse: {e}\n--- source ---\n{source}"),
        };
        let mut state = GrammarState::new(Arc::new(grammar));
        if state.advance_bytes(input.as_bytes()).is_err() {
            return false;
        }
        state.is_complete()
    }

    fn wrap_with_root(rule_name: &str, rules: String) -> String {
        let mut src = String::new();
        let _ = writeln!(&mut src, "root ::= {rule_name}");
        src.push_str(&rules);
        src.push_str(JSON_GRAMMAR);
        src
    }

    #[test]
    fn compiles_flat_object() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer"},
            },
            "required": ["name", "count"],
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "obj", &mut rules);
        let src = wrap_with_root("obj", rules);
        assert!(accepts(&src, r#"{"name":"ok","count":3}"#));
        assert!(!accepts(&src, r#"{"count":3}"#));
    }

    #[test]
    fn compiles_nested_via_ref() {
        let schema = json!({
            "type": "object",
            "properties": {
                "inner": {"$ref": "#/$defs/Inner"}
            },
            "required": ["inner"],
            "$defs": {
                "Inner": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                }
            }
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "root_obj", &mut rules);
        let src = wrap_with_root("root_obj", rules);
        assert!(accepts(&src, r#"{"inner":{"x":1}}"#));
        assert!(!accepts(&src, r#"{"inner":{}}"#));
    }

    #[test]
    fn compiles_any_of_alternation() {
        let schema = json!({
            "anyOf": [
                {"type": "string", "enum": ["Low"]},
                {"type": "string", "enum": ["High"]},
            ]
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "conf", &mut rules);
        let src = wrap_with_root("conf", rules);
        assert!(accepts(&src, r#""Low""#));
        assert!(accepts(&src, r#""High""#));
        assert!(!accepts(&src, r#""Medium""#));
    }

    /// Schemars emits unit-enum variants with doc comments as
    /// `anyOf: [{const: "A", description: "..."}, ...]`. The grammar
    /// must reject values outside the const set, even though each
    /// subschema has no `type` field. Regression for the "Definite"
    /// confidence leak that broke the whodunit example.
    #[test]
    fn compiles_any_of_const_variants_from_schemars() {
        let schema = json!({
            "anyOf": [
                {"const": "Low", "description": "thin evidence"},
                {"const": "Medium", "description": "plausible"},
                {"const": "High", "description": "airtight"},
            ]
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "conf", &mut rules);
        let src = wrap_with_root("conf", rules);
        assert!(accepts(&src, r#""Low""#));
        assert!(accepts(&src, r#""Medium""#));
        assert!(accepts(&src, r#""High""#));
        assert!(!accepts(&src, r#""Definite""#));
        assert!(!accepts(&src, r#""low""#)); // case-sensitive
    }

    #[test]
    fn thought_rules_accept_bare_and_wrapped() {
        let mut src = String::from("root ::= thought? ws value\n");
        emit_thought_rules(&mut src);
        src.push_str(JSON_GRAMMAR);
        assert!(accepts(&src, r#"42"#));
        assert!(accepts(&src, r#"<think>hmm</think> 42"#));
        // `<` inside thought body is OK as long as it's not `</`.
        assert!(accepts(&src, r#"<think>if x < 5 then</think> 42"#));
    }

    #[test]
    fn json_ws_is_at_most_single_char() {
        // Accepts canonical compact + single-space JSON (all real
        // use cases for grammar-constrained generation).
        let src = format!("root ::= value\n{JSON_GRAMMAR}");
        assert!(accepts(&src, r#"{"x":1}"#));
        assert!(accepts(&src, r#"{"x": 1}"#));
        assert!(accepts(&src, r#"[1, 2, 3]"#));
        // Rejects multi-char whitespace runs — the escape valve that
        // lets a constrained sampler stall on "thinking" padding
        // until max_tokens. Regression target.
        assert!(!accepts(&src, "{\"x\":  1}"));
        assert!(!accepts(&src, "{\"x\":\t\t1}"));
        assert!(!accepts(&src, "{\"x\":\n\n1}"));
        assert!(!accepts(&src, "{\"x\" : \t 1}"));
    }
}
