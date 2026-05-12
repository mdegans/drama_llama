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
//! * `type: object` with `properties` (+ optional `required`) →
//!   required fields first (in `required:`-array order), then
//!   optional fields each wrapped in `( ws "," ws ... )?` so they
//!   may be omitted but must match the declared type when present.
//!   The all-optional case (no `required`) emits N "chain"
//!   alternatives so all 2^N inclusion patterns are reachable.
//!
//!   The required-first / optional-after layout matches Anthropic's
//!   structured-outputs documentation ("properties in objects
//!   maintain their defined ordering from your schema, with one
//!   important caveat: required properties appear first, followed by
//!   optional properties"), so a model emitting in that canonical
//!   order will not force-EOS against this grammar.
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
//!
//! # What's intentionally NOT supported
//!
//! `minLength`, `maxLength`, `pattern`, `minimum`, `maximum`,
//! `multipleOf`, `oneOf`, and `allOf` are deliberately ignored. This
//! matches what Anthropic's own SDKs do (Python / TypeScript / Ruby /
//! PHP all strip these keywords before sending the schema and reword
//! them into the field's `description`). Grammar-level enforcement of
//! value-bound constraints replaces the model's *reasoning about
//! value* with structural padding that *looks* valid:
//!
//! * `pattern: "^[A-Z]{2}_\d{4}$"` → model emits `"AB_0000"`. Pattern
//!   satisfied, semantics empty.
//! * `minLength: 5` on a 3-char answer → model emits `"yesyy"` to
//!   pad. Garbage that passes validation.
//! * `maximum: 10` when model wanted 100 → emits `10`. Off by 10×.
//! * `oneOf` has been observed to break Anthropic's structured
//!   generation entirely — model forced to emit `null`.
//!
//! Document constraints in the field's `description` instead;
//! validate post-generation in the tool runtime. See
//! `.claude/memory/schema_constraint_keywords_decision.md` for the
//! full reasoning. Don't add support without revisiting that memo.

use std::fmt::Write;

use serde_json::Value;

/// Emit GBNF rules that constrain a JSON value to `schema`.
///
/// The top-level rule will be named `rule_name`; anonymous helpers
/// get unique child names derived from it. If `schema` carries a
/// `$defs` map at its root, `$ref` entries of the form
/// `#/$defs/<Name>` are resolved inline.
///
/// Exposed as `#[doc(hidden)] pub` (re-exported at the crate root)
/// so the in-tree fuzzer can compile schemas directly without going
/// through the tool-choice wrapper rules. Not part of the stable
/// surface — callers outside the fuzzer should use
/// [`grammar_for_tool_choice`](crate::grammar_for_tool_choice) or
/// [`output_config::grammar_for_output_config`](crate::output_config::grammar_for_output_config).
#[doc(hidden)]
pub fn schema_to_gbnf(schema: &Value, rule_name: &str, out: &mut String) {
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
            let _ = writeln!(
                out,
                "{rule_name} ::= {alts}",
                alts = sub_names.join(" | ")
            );
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
    let required_vec: Vec<String> = schema
        .get("required")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    let required_set: std::collections::HashSet<&String> =
        required_vec.iter().collect();

    // Empty `properties` (and therefore no slots) → permissive object.
    if props.is_empty() && required_vec.is_empty() {
        let _ = writeln!(out, "{rule_name} ::= object");
        return;
    }

    // Layout: required first (in `required:`-array order), then
    // optionals. This matches Anthropic's documented tool-call
    // convention (optionals trailing), and lets the leading-comma
    // logic stay simple — every required slot anchors the chain, so
    // each optional gets a self-contained `( ws "," ws ... )?` slot.
    //
    // Trade: declaration order from `properties` is not preserved for
    // required props (we use `required:`-array order instead). In
    // practice the two are aligned in real schemas and serde_json's
    // default Map iterator is alphabetical anyway, so neither order
    // is "the schema author's intent" with high confidence.
    let mut required_slots: Vec<(String, String)> = Vec::new();
    for name in &required_vec {
        *counter += 1;
        let child_rule = format!("{rule_name}__{c}", c = *counter);
        if let Some(prop_schema) = props.get(name) {
            emit_schema_rule(prop_schema, &child_rule, out, counter, defs);
            required_slots.push((name.clone(), child_rule));
        } else {
            // Required prop absent from `properties` — rare but legal;
            // permissive `value` is the safest fallback.
            required_slots.push((name.clone(), "value".to_string()));
        }
    }
    let mut optional_slots: Vec<(String, String)> = Vec::new();
    for (name, prop_schema) in props.iter() {
        if required_set.contains(name) {
            continue;
        }
        *counter += 1;
        let child_rule = format!("{rule_name}__{c}", c = *counter);
        emit_schema_rule(prop_schema, &child_rule, out, counter, defs);
        optional_slots.push((name.clone(), child_rule));
    }

    if required_slots.is_empty() {
        // All-optional case. Emit chain alternatives so all 2^N
        // include/skip combinations are reachable: for each starting
        // position K, emit prop[K] followed by `(",", prop[K+1])?` ...
        // `(",", prop[N-1])?`. The outer wrapping is
        // `(chain_0 | chain_1 | ... | chain_{N-1})?` so the
        // empty-object case is also matched.
        let n = optional_slots.len();
        let mut chain_names: Vec<String> = Vec::with_capacity(n);
        for k in 0..n {
            let chain_name = format!("{rule_name}__chain_{k}");
            let mut tail = String::new();
            for (i, (name, child)) in optional_slots.iter().enumerate().skip(k)
            {
                let lit = escape_for_gbnf_string(
                    &serde_json::to_string(name).unwrap(),
                );
                if i == k {
                    let _ = write!(&mut tail, r#""{lit}" ws ":" ws {child}"#);
                } else {
                    let _ = write!(
                        &mut tail,
                        r#" ( ws "," ws "{lit}" ws ":" ws {child} )?"#
                    );
                }
            }
            let _ = writeln!(out, "{chain_name} ::= {tail}");
            chain_names.push(chain_name);
        }
        let alts = chain_names.join(" | ");
        let _ = writeln!(out, r#"{rule_name} ::= "{{" ws ( {alts} )? ws "}}""#);
        return;
    }

    // Mixed required + (zero or more) optional. Required first, then
    // optionals each as `( ws "," ws "<name>" ws ":" ws <typed> )?`.
    let mut body = String::from("\"{\" ws");
    for (i, (field_name, child_rule)) in required_slots.iter().enumerate() {
        if i > 0 {
            body.push_str(r#" ws "," ws"#);
        }
        let lit =
            escape_for_gbnf_string(&serde_json::to_string(field_name).unwrap());
        let _ = write!(body, r#" "{lit}" ws ":" ws {child_rule}"#);
    }
    for (field_name, child_rule) in optional_slots.iter() {
        let lit =
            escape_for_gbnf_string(&serde_json::to_string(field_name).unwrap());
        let _ =
            write!(body, r#" ( ws "," ws "{lit}" ws ":" ws {child_rule} )?"#);
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
/// Standard JSON grammar appended to every schema-derived GBNF.
///
/// Exposed as `#[doc(hidden)] pub` for the fuzzer (paired with
/// [`schema_to_gbnf`]). Not part of the stable surface.
#[doc(hidden)]
pub const JSON_GRAMMAR: &str = r#"
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" ws ( member ( ws "," ws member )* )? ws "}"
member ::= string ws ":" ws value
array ::= "[" ws ( value ( ws "," ws value )* )? ws "]"
string ::= "\"" char* "\""
char ::= unescaped | escape
unescaped ::= [^"\\\x00-\x1F]
escape ::= "\\" ( ["\\/bfnrt] | "u" non_surrogate_hex4 | "u" high_surrogate "\\u" low_surrogate )
non_surrogate_hex4 ::= [0-9a-cA-C] hex hex hex | [dD] [0-7] hex hex | [e-fE-F] hex hex hex
high_surrogate ::= [dD] [89aAbB] hex hex
low_surrogate ::= [dD] [c-fC-F] hex hex
hex ::= [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? ( "0" | [1-9] [0-9]* )
frac ::= "." [0-9]+
exp ::= [eE] [+\-]? [0-9] [0-9]?
ws ::= [ \t\n\r]?
"#;
// `exp` allows 1-2 exponent digits (vs the original `[0-9]+`) so a
// grammar-emitted number is guaranteed to fit in `f64`'s ±E308 range.
// 1e99 is the largest representable; in practice tool-arg numbers
// almost never use scientific notation at all, so capping at 2 digits
// is the right tradeoff. Without this cap, the fuzzer trivially emits
// things like `5E481` that the grammar accepts but
// `serde_json::from_slice` rejects with "number out of range" —
// generation force-EOSes downstream.
//
// `escape`'s `\u` branch is split into a non-surrogate code-unit and
// a paired high+low surrogate alternative. The original
// `"u" hex hex hex hex` admitted lone surrogates (`\uD800` with no
// low pair) and surrogate prefixes followed by string-close, both of
// which serde_json rejects per RFC 8259 §7. The split lets all valid
// JSON escapes through while excluding the malformed shapes.
//
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
            Err(e) => {
                panic!("grammar failed to parse: {e}\n--- source ---\n{source}")
            }
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

    #[test]
    fn json_string_rejects_raw_control_chars() {
        // RFC 8259 §7: raw control characters (U+0000–U+001F) inside a
        // string are forbidden — they must be escaped (\n, \t, \uXXXX).
        // Pre-fix the `unescaped` rule had `[^"\\]` as a first
        // alternative, which accepted raw control bytes; downstream
        // serde_json::from_str then rejected them with "Invalid control
        // character". Regression target for that failure mode.
        let src = format!("root ::= string\n{JSON_GRAMMAR}");
        assert!(!accepts(&src, "\"foo\nbar\""));
        assert!(!accepts(&src, "\"foo\tbar\""));
        assert!(!accepts(&src, "\"foo\rbar\""));
        assert!(!accepts(&src, "\"\x01\""));
        // Escaped forms still accepted.
        assert!(accepts(&src, r#""foo\nbar""#));
        assert!(accepts(&src, r#""foo\tbar""#));
        assert!(accepts(&src, r#""foo\rbar""#));
    }

    #[test]
    fn json_string_accepts_multibyte_utf8() {
        // The negated-set form must still admit non-ASCII codepoints
        // (Cogito tool args carry CJK / emoji routinely). Belt-and-
        // braces against future tightening that loses UTF-8 support.
        let src = format!("root ::= string\n{JSON_GRAMMAR}");
        assert!(accepts(&src, "\"你好\""));
        assert!(accepts(&src, "\"🍓\""));
    }

    /// Surfaced by the differential fuzzer (2026-05-12). The original
    /// `exp ::= [eE] [+\-]? [0-9]+` admitted unbounded exponent
    /// magnitude, so the grammar accepted numbers like `5E481` that
    /// `serde_json::from_slice` rejects with "number out of range"
    /// (overflows `f64`'s ±E308). Cap is 1-2 exponent digits — well
    /// inside `f64` range, covers any realistic tool-arg number.
    #[test]
    fn json_number_exp_capped_to_fit_f64() {
        let src = format!("root ::= value\n{JSON_GRAMMAR}");
        // 1-2 digit exponents accepted.
        assert!(accepts(&src, "1e0"));
        assert!(accepts(&src, "1e3"));
        assert!(accepts(&src, "1.5E99"));
        assert!(accepts(&src, "-2.5e-99"));
        // 3+ digit exponents rejected (the bug class — could overflow
        // f64). Trades the legitimate 1e100..1e308 range for safety;
        // tool args essentially never use exponents that large.
        assert!(!accepts(&src, "1E308"));
        assert!(!accepts(&src, "1E1234"));
        assert!(!accepts(&src, "5E481"));
    }

    /// Option A landing (2026-05-12): optional properties are now
    /// type-enforced when present. Mixed required+optional schema:
    /// the optional must match its declared type if included, and is
    /// omittable. Wrong type for the optional rejects.
    #[test]
    fn optional_property_type_enforced_when_present() {
        let schema = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "verbose": {"type": "boolean"}
            },
            "required": ["name"]
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "obj", &mut rules);
        let src = wrap_with_root("obj", rules);
        // Required-only — optional omitted.
        assert!(accepts(&src, r#"{"name":"x"}"#));
        // Required + optional with correct type.
        assert!(accepts(&src, r#"{"name":"x","verbose":true}"#));
        assert!(accepts(&src, r#"{"name":"x","verbose":false}"#));
        // Required + optional with WRONG type — the bug class
        // we're closing.
        assert!(!accepts(&src, r#"{"name":"x","verbose":1}"#));
        assert!(!accepts(&src, r#"{"name":"x","verbose":"yes"}"#));
        // Missing required still rejected.
        assert!(!accepts(&src, r#"{"verbose":true}"#));
        assert!(!accepts(&src, "{}"));
    }

    /// All-optional schema: every 2^N inclusion combination must be
    /// reachable, including the empty object. Wrong types rejected
    /// when present.
    #[test]
    fn all_optional_object_permits_every_subset() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "boolean"}
            }
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "obj", &mut rules);
        let src = wrap_with_root("obj", rules);
        // All four combinations of include/skip — empty, a, b, both.
        assert!(accepts(&src, "{}"));
        assert!(accepts(&src, r#"{"a":1}"#));
        assert!(accepts(&src, r#"{"b":true}"#));
        assert!(accepts(&src, r#"{"a":1,"b":true}"#));
        // Wrong type rejected.
        assert!(!accepts(&src, r#"{"a":"oops"}"#));
        assert!(!accepts(&src, r#"{"b":1}"#));
        // Order not relevant when both required-first/optional-after
        // collapse to all-optional — but reverse order isn't supported
        // (chains follow declaration / alphabetical iteration). Don't
        // assert on `{"b":true,"a":1}` — that's a known limitation
        // documented in the module header.
    }

    /// `default:`-bearing optional behaves the same as any other
    /// optional. The grammar lets the model pick either alternative
    /// (or omit) — it doesn't pre-judge which value the model
    /// "should" emit when it includes the field. This preserves
    /// neutral behavior for both omit-defaulting and explicit-
    /// defaulting model training styles.
    #[test]
    fn optional_with_default_keeps_full_value_alternation() {
        let schema = json!({
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "verbose": {"type": "boolean", "default": false}
            },
            "required": ["action"]
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "obj", &mut rules);
        let src = wrap_with_root("obj", rules);
        // Both true and false accepted when included.
        assert!(accepts(&src, r#"{"action":"go","verbose":true}"#));
        assert!(accepts(&src, r#"{"action":"go","verbose":false}"#));
        // Omission still allowed.
        assert!(accepts(&src, r#"{"action":"go"}"#));
    }

    /// Surfaced by the differential fuzzer (2026-05-12). The original
    /// `escape ::= "\\" ( ["\\/bfnrt] | "u" hex hex hex hex )` admitted
    /// lone high surrogates (`\uD800`) and surrogate prefixes followed
    /// by string-close, both of which RFC 8259 §7 / `serde_json`
    /// reject. Replaced with a non-surrogate alternative plus a
    /// paired-surrogate alternative.
    #[test]
    fn json_escape_rejects_lone_surrogates() {
        let src = format!("root ::= value\n{JSON_GRAMMAR}");
        // Non-surrogate \u escapes still accepted.
        assert!(accepts(&src, r#""A""#)); // 'A'
        assert!(accepts(&src, r#""é""#)); // 'é'
        assert!(accepts(&src, r#""中""#)); // '中'
                                           // Surrogate range D800-DFFF rejected as a lone code unit.
        assert!(!accepts(&src, r#""\uD800""#));
        assert!(!accepts(&src, r#""\uDBFF""#));
        assert!(!accepts(&src, r#""\uDC00""#));
        assert!(!accepts(&src, r#""\uDFFF""#));
        // Lowercase hex of a surrogate also rejected.
        assert!(!accepts(&src, r#""\udabc""#));
        // Just-below and just-above surrogate range still accepted.
        assert!(accepts(&src, r#""퟿""#));
        assert!(accepts(&src, r#""""#));
        // Properly paired surrogates accepted (encodes U+10000+,
        // i.e. astral-plane codepoints like emoji).
        assert!(accepts(&src, r#""🍓""#)); // 🍓
        assert!(accepts(&src, r#""𝄞""#)); // 𝄞
                                          // Half-pair (high without low) rejected — the bug class.
        assert!(!accepts(&src, r#""\uD83C""#));
        // High surrogate followed by something other than \u low
        // surrogate is rejected.
        assert!(!accepts(&src, r#""\uD83Cx""#));
        assert!(!accepts(&src, r#""\uD83CA""#));
    }
}
