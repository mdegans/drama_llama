//! Hostile-schema regressions: no client-supplied schema — a tool's
//! `input_schema` (strict or not, any dialect) or an `output_config`
//! `json_schema` — can abort, hang, or exhaust memory compiling,
//! constraining, or parsing. Each test started as a probe in the
//! hostile recheck; the engine-level ones (stack and state caps) live
//! beside the matcher in `sample::grammar`, the encoder-level ones in
//! `grammar_compile`.

use std::{sync::Arc, time::Instant};

use serde_json::{json, Map, Value};

use crate::dialect::{grammar_source, Anchor, DialectError, EmitOptions};
use crate::grammar_compile::{SchemaError, JSON_GRAMMAR, MAX_GRAMMAR_BYTES};
use crate::{
    grammar_for_tool_choice, output_config::grammar_for_output_config,
    schema_to_gbnf, CallSyntax, Grammar, GrammarState, OutputConfigError,
    OutputConfigOptions, Tool, ToolChoice, ToolChoiceError, ToolChoiceOptions,
};
use misanthropic::prompt::output::OutputConfig;

fn tool(schema: Value) -> Tool {
    Tool::builder("t")
        .description("d")
        .schema(schema)
        .build()
        .unwrap()
}

/// Every dialect family's grammar.
fn syntaxes() -> [CallSyntax; 4] {
    [
        CallSyntax::qwen_xml(),
        CallSyntax::hermes_json(),
        CallSyntax::gemma4(),
        CallSyntax::gpt_oss(),
    ]
}

fn lazy() -> EmitOptions {
    EmitOptions {
        anchor: Anchor::Lazy,
        parallel: false,
    }
}

/// `schema` as a standalone JSON grammar.
fn compile(schema: &Value) -> String {
    let mut rules = String::new();
    schema_to_gbnf(schema, "s", &mut rules).unwrap();
    format!("root ::= s\n{rules}{JSON_GRAMMAR}")
}

fn accepts(src: &str, input: &str) -> bool {
    let grammar = Arc::new(Grammar::parse(src).unwrap());
    let mut state = GrammarState::new(grammar);
    state.advance_bytes(input.as_bytes()).is_ok() && state.is_complete()
}

/// Run `f` on a thread with a `kib` KiB stack: a compile that recursed
/// as deep as its input would overflow it (and abort the process).
fn on_small_stack<T: Send + 'static>(
    kib: usize,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    std::thread::Builder::new()
        .stack_size(kib * 1024)
        .spawn(f)
        .unwrap()
        .join()
        .unwrap()
}

/// A schema too complex for a grammar is a schema error on every path
/// a client reaches — each dialect, strict `tool_choice`, and
/// `output_config` — never a half-gigabyte grammar.
#[test]
fn too_complex_schema_fails_cleanly_everywhere() {
    let n = 400_000;
    let props: Map<String, Value> =
        (0..n).map(|i| (format!("p{i}"), json!({}))).collect();
    let schema = json!({"type": "object", "properties": props});
    let tool = tool(schema.clone());
    for syntax in syntaxes() {
        let family = syntax.family;
        match grammar_source(&syntax, &[&tool], &lazy()) {
            Err(DialectError::Schema {
                source: SchemaError::TooComplex { limit },
                ..
            }) => assert_eq!(limit, MAX_GRAMMAR_BYTES),
            other => panic!("{family:?}: {:?}", other.map(|s| s.len())),
        }
    }
    let err = grammar_for_tool_choice(
        &ToolChoice::any(),
        &[tool],
        &ToolChoiceOptions::default(),
        false,
    )
    .unwrap_err();
    assert!(matches!(err, ToolChoiceError::Schema { .. }), "{err}");
    assert!(err.to_string().contains("too complex"), "{err}");
    let err = grammar_for_output_config(
        &OutputConfig::json_schema(schema),
        &OutputConfigOptions::default(),
        false,
    )
    .unwrap_err();
    assert!(matches!(err, OutputConfigError::Schema(_)), "{err}");
    assert!(err.to_string().contains("too complex"), "{err}");
}

/// `{"enum": []}` admits nothing: a schema error on every path, not a
/// GBNF syntax error from an empty rule body.
#[test]
fn empty_enum_fails_cleanly_everywhere() {
    let schema = json!({
        "type": "object",
        "properties": {"x": {"enum": []}},
        "required": ["x"],
    });
    let tool = tool(schema.clone());
    for syntax in syntaxes() {
        let family = syntax.family;
        match grammar_source(&syntax, &[&tool], &lazy()) {
            Err(DialectError::Schema {
                source: SchemaError::EmptyEnum,
                ..
            }) => {}
            other => panic!("{family:?}: {other:?}"),
        }
    }
    let err = grammar_for_tool_choice(
        &ToolChoice::any(),
        &[tool],
        &ToolChoiceOptions::default(),
        false,
    )
    .unwrap_err();
    assert!(matches!(err, ToolChoiceError::Schema { .. }), "{err}");
    let err = grammar_for_output_config(
        &OutputConfig::json_schema(schema),
        &OutputConfigOptions::default(),
        false,
    )
    .unwrap_err();
    assert!(
        matches!(err, OutputConfigError::Schema(SchemaError::EmptyEnum)),
        "{err}"
    );
}

/// Every left-recursion shape the recheck tried — self-aliases, alias
/// cycles, recursive `anyOf`/`allOf`/`oneOf`, `items`- and
/// `additionalProperties`-only recursion, odd `$defs` — compiles on a
/// 512 KiB stack to a grammar that parses, in every dialect, or fails
/// as a schema error. None aborts.
#[test]
fn left_recursion_variants_compile_everywhere() {
    let cases = [
        json!({"$ref":"#/$defs/A","$defs":{"A":{"anyOf":[{"anyOf":[{"$ref":"#/$defs/A"}]},{"type":"string"}]}}}),
        json!({"$ref":"#/$defs/A","$defs":{"A":{"anyOf":[{"$ref":"#/$defs/B"}]},"B":{"anyOf":[{"$ref":"#/$defs/A"},{"type":"integer"}]}}}),
        json!({"anyOf":[{"$ref":"#/$defs/A"}],"$defs":{"A":{"$ref":"#/$defs/A"}}}),
        json!({"type":"object","properties":{"x":{"$ref":"#"}}}),
        json!({"type":"object","properties":{"x":{"$ref":"#/definitions/N"}},"definitions":{"N":{"$ref":"#/definitions/N"}}}),
        json!({"$ref":"#/$defs/","$defs":{"":{"$ref":"#/$defs/"}}}),
        json!({"$ref":"#/$defs/L","$defs":{"L":{"type":"array","items":{"$ref":"#/$defs/L"}}}}),
        json!({"$ref":"#/$defs/M","$defs":{"M":{"type":"object","additionalProperties":{"$ref":"#/$defs/M"}}}}),
        json!({"$ref":"#/$defs/M","$defs":{"M":{"allOf":[{"$ref":"#/$defs/M"}]}}}),
        json!({"$ref":"#/$defs/M","$defs":{"M":{"oneOf":[{"$ref":"#/$defs/M"}]}}}),
        json!({"$ref":"#/$defs/E","$defs":{"E":{"anyOf":[]}}}),
        json!({"$ref":"#/$defs/A","$defs":{"A":{"type":"object","$ref":"#/$defs/A"}}}),
        json!({"$ref":"#/$defs/A","$defs":{"A":{"type":["array","null"],"items":{"$ref":"#/$defs/A"}}}}),
        json!({"$ref":"#/$defs/A","$defs":[1,2]}),
        json!({"$ref":"#/$defs/A","$defs":{"A": 5}}),
        json!({"$ref":"#/$defs/A","$defs":{"A": true}}),
    ];
    for schema in cases {
        let s = schema.clone();
        let src = on_small_stack(512, move || compile(&s));
        assert!(Grammar::parse(&src).is_ok(), "{schema}");
        let param = schema
            .get("$ref")
            .map(|r| json!({"$ref": r}))
            .unwrap_or(json!({}));
        let defs = schema.get("$defs").cloned().unwrap_or(json!({}));
        let wrapped = json!({
            "type": "object",
            "properties": {"x": param},
            "$defs": defs,
        });
        for syntax in syntaxes() {
            let t = tool(wrapped.clone());
            let ok = on_small_stack(512, move || {
                grammar_source(&syntax, &[&t], &lazy())
                    .map(|src| Grammar::parse(&src).is_ok())
            });
            assert!(matches!(ok, Ok(true)), "{schema}: {ok:?}");
        }
    }
}

/// A tree schema's grammar matches a 2000-deep tree, on a worker-sized
/// stack: deep output recursion lives in the matcher's heap stacks.
#[test]
fn deep_tree_output_matches() {
    let schema = json!({
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/Node"}},
        "required": ["root"],
        "$defs": {"Node": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
            },
        }},
    });
    let src = compile(&schema);
    let depth = 2000;
    let input = format!(
        "{{\"root\":{}{{}}{}}}",
        r#"{"children":["#.repeat(depth),
        "]}".repeat(depth)
    );
    assert!(on_small_stack(2048, move || accepts(&src, &input)));
}

/// A 100,000-way `anyOf` of one `$ref`: a 4 MB grammar, within the
/// limit, that matches in time and memory bounded by the stack caps.
#[test]
fn wide_any_of_fanout_matches() {
    let n = 100_000;
    let variants: Vec<Value> =
        (0..n).map(|_| json!({"$ref": "#/$defs/A"})).collect();
    let schema = json!({"anyOf": variants, "$defs": {"A": {"type": "string"}}});
    let src = compile(&schema);
    assert!(src.len() < MAX_GRAMMAR_BYTES);
    assert!(accepts(&src, r#""x""#));
    assert!(!accepts(&src, "1"));
}

/// The post-generation schema check over a 60-deep diamond of defs
/// (each def three ways to reach the next) stays finite and on a small
/// stack.
#[test]
fn schema_check_diamond_stays_bounded() {
    let n = 60;
    let mut defs = Map::new();
    for i in 0..n {
        let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
        defs.insert(
            format!("D{i}"),
            json!({"anyOf": [
                next.clone(),
                {"type": "array", "items": next},
                {"type": "object", "additionalProperties": next},
            ]}),
        );
    }
    defs.insert(format!("D{n}"), json!({"type": "integer"}));
    let schema = json!({"$ref": "#/$defs/D0", "$defs": defs});
    let value = format!("{}\"x\"{}", "[".repeat(30), "]".repeat(30));
    let start = Instant::now();
    let ok = on_small_stack(512, move || {
        crate::schema_check::check_text(&schema, &value).is_ok()
    });
    assert!(!ok);
    assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
}
