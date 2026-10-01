//! Hostile-schema regressions: no client-supplied schema — a tool's
//! `input_schema` (strict or not, any dialect) or an `output_config`
//! `json_schema` — can abort, hang, or exhaust memory compiling,
//! constraining, or parsing. Each test started as a probe in the
//! hostile recheck; the engine-level ones (stack, state and cache caps)
//! live beside the matcher in `sample::grammar`, the encoder-level ones
//! in `grammar_compile`, the up-front measure in `schema_budget`.
//!
//! Two layers: the request's schemas are measured before anything
//! touches them ([`SchemaLimits`]), and every pipeline keeps its own
//! caps for a caller that skips the measure. The probes below that pass
//! the default limits must be cheap anyway; the ones past them must be
//! refused up front.

use std::{sync::Arc, time::Instant};

use serde_json::{json, Map, Value};

use crate::dialect::emit::{tagged_values, TaggedValue};
use crate::dialect::{
    grammar_source, parse_text, Anchor, DialectError, EmitOptions, Leniency,
};
use crate::grammar_compile::{
    SchemaError, JSON_GRAMMAR, MAX_GRAMMAR_BYTES, MAX_GRAMMAR_RULES,
};
use crate::schema_budget::{check_schemas, SchemaLimit};
use crate::{
    grammar_for_tool_choice, output_config::grammar_for_output_config,
    schema_to_gbnf, Block, CallSyntax, Grammar, GrammarState,
    OutputConfigError, OutputConfigOptions, SchemaLimits, Tool, ToolChoice,
    ToolChoiceError, ToolChoiceOptions,
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

/// Qwen's tagged dialect compiles each parameter's schema on its own,
/// and every parameter `$ref`ing the head of a D-long chain of defs used
/// to rewrite all D for each of the P parameters: 800 × 800 was 143 MB
/// of grammar and 5 million rules. One compiler a tool writes each def
/// once.
#[test]
fn qwen_writes_each_def_once_per_tool() {
    let (p, d) = (800, 800);
    let props: Map<String, Value> = (0..p)
        .map(|i| (format!("p{i}"), json!({"$ref": "#/$defs/D0"})))
        .collect();
    let mut defs: Map<String, Value> = (0..d)
        .map(|i| {
            let next = format!("#/$defs/D{}", i + 1);
            (
                format!("D{i}"),
                json!({"type": "object", "properties": {"n": {"$ref": next}}}),
            )
        })
        .collect();
    defs.insert(format!("D{d}"), json!({"type": "integer"}));
    let schema = json!({"type": "object", "properties": props, "$defs": defs});
    let src =
        grammar_source(&CallSyntax::qwen_xml(), &[&tool(schema)], &lazy())
            .unwrap();
    assert!(src.len() < 1 << 20, "{} bytes", src.len());
    for i in 0..=d {
        let head = format!("tool_0__def{i}_D{i} ::=");
        assert_eq!(src.matches(&head).count(), 1, "{head}");
    }
    Grammar::parse(&src).unwrap();
}

/// Thousands of parameters beside thousands of (unreferenced) defs:
/// the defs table is read once per tool, not once per parameter.
#[test]
fn qwen_many_params_and_defs_compile_and_parse_quickly() {
    let n = 6000;
    let props: Map<String, Value> = (0..n)
        .map(|i| (format!("p{i}"), json!({"type": "string"})))
        .collect();
    let defs: Map<String, Value> =
        (0..n).map(|i| (format!("D{i}"), json!({}))).collect();
    let tool =
        tool(json!({"type": "object", "properties": props, "$defs": defs}));
    let start = Instant::now();
    let src =
        grammar_source(&CallSyntax::qwen_xml(), &[&tool], &lazy()).unwrap();
    Grammar::parse(&src).unwrap();

    let params: String = (0..n)
        .map(|i| format!("<parameter=p{i}>\nv{i}\n</parameter>\n"))
        .collect();
    let text =
        format!("<tool_call>\n<function=t>\n{params}</function>\n</tool_call>");
    let parsed = parse_text(
        &CallSyntax::qwen_xml(),
        &[&tool],
        &text,
        false,
        Leniency::Final,
    );
    let input = parsed
        .blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(&call.input),
            _ => None,
        })
        .expect("a call");
    assert_eq!(input["p5999"], json!("v5999"));
    assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
}

/// The first input of a tagged call, as `parse_text` reads it.
fn first_input(syntax: &CallSyntax, tool: &Tool, text: &str) -> Value {
    let parsed = parse_text(syntax, &[tool], text, false, Leniency::Final);
    parsed
        .blocks
        .iter()
        .find_map(|b| match b {
            Block::ToolUse { call } => Some(call.input.clone()),
            _ => None,
        })
        .expect("a call")
}

/// A Qwen call to `t` with `params` in order.
fn qwen_call(params: &[(&str, &str)]) -> String {
    let params: String = params
        .iter()
        .map(|(k, v)| format!("<parameter={k}>\n{v}\n</parameter>\n"))
        .collect();
    format!("<tool_call>\n<function=t>\n{params}</function>\n</tool_call>")
}

/// A tool's parameters are classified on one shared budget, in
/// declaration order, by the emitter and the parser alike — so when
/// every parameter has a large set of its own, the budget runs out at
/// the same parameter for both: the early ones are spelled raw, the
/// late ones JSON, and each reads back as written.
#[test]
fn tagged_budget_agrees_between_grammar_and_parser() {
    // 100 sets of 50 members ~300 bytes each: 1.5 MB, half again the
    // budget.
    let pad = "x".repeat(300);
    let props: Map<String, Value> = (0..100)
        .map(|p| {
            let members: Vec<String> =
                (0..50).map(|m| format!("{pad}{p}_{m}")).collect();
            (format!("p{p}"), json!({"enum": members}))
        })
        .collect();
    let schema = json!({
        "type": "object",
        "properties": props,
        "required": ["p0", "p99"],
    });
    let syntax = CallSyntax::qwen_xml();
    let values = tagged_values(&syntax, &schema);
    assert!(matches!(values[0].1, TaggedValue::Choice(_)));
    assert!(matches!(values[99].1, TaggedValue::Json));

    let tool = tool(schema);
    let src = grammar_source(&syntax, &[&tool], &lazy()).unwrap();
    let (first, last) = (format!("{pad}0_7"), format!("{pad}99_7"));
    let quoted = format!("\"{last}\"");
    let raw_then_json = qwen_call(&[("p0", &first), ("p99", &quoted)]);
    assert!(accepts(&src, &raw_then_json));
    let both_json =
        qwen_call(&[("p0", &format!("\"{first}\"")), ("p99", &quoted)]);
    assert!(!accepts(&src, &both_json));
    assert!(!accepts(
        &src,
        &qwen_call(&[("p0", &first), ("p99", &last)])
    ));
    assert_eq!(
        first_input(&syntax, &tool, &raw_then_json),
        json!({"p0": first, "p99": last})
    );
}

/// 2000 parameters naming one def whose one member is 100 KB (the
/// hostile recheck): classified once and shared, the member spelled
/// once — it was 200 MB of spelling and 400 MB of copies — and the
/// grammar writes its alternation once.
#[test]
fn qwen_shared_def_is_classified_and_written_once() {
    let member = "x".repeat(100_000);
    let props: Map<String, Value> = (0..2000)
        .map(|i| (format!("p{i}"), json!({"$ref": "#/$defs/E"})))
        .collect();
    let schema = json!({
        "type": "object",
        "properties": props,
        "$defs": {"E": {"enum": [member]}},
    });
    let syntax = CallSyntax::qwen_xml();
    let start = Instant::now();
    let values = tagged_values(&syntax, &schema);
    let TaggedValue::Choice(shared) = &values[0].1 else {
        panic!("{:?}", values[0].1);
    };
    for (name, value) in &values {
        match value {
            TaggedValue::Choice(c) => assert!(Arc::ptr_eq(c, shared), "{name}"),
            other => panic!("{name}: {other:?}"),
        }
    }
    let tool = tool(schema);
    let src = grammar_source(&syntax, &[&tool], &lazy()).unwrap();
    assert_eq!(src.matches(member.as_str()).count(), 1);
    Grammar::parse(&src).unwrap();
    let text = qwen_call(&[("p0", &member), ("p1999", &member)]);
    assert!(accepts(&src, &text));
    let input = first_input(&syntax, &tool, &text);
    assert_eq!(input["p1999"], json!(member));
    assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
}

/// The same, with an object member, which is JSON: one def rule for
/// every parameter, its 100 KB written once.
#[test]
fn qwen_shared_object_def_is_compiled_once() {
    let long = "x".repeat(100_000);
    let props: Map<String, Value> = (0..2000)
        .map(|i| (format!("p{i}"), json!({"$ref": "#/$defs/E"})))
        .collect();
    let schema = json!({
        "type": "object",
        "properties": props,
        "$defs": {"E": {"enum": [{"k": long}]}},
    });
    let syntax = CallSyntax::qwen_xml();
    let start = Instant::now();
    assert!(tagged_values(&syntax, &schema)
        .iter()
        .all(|(_, v)| *v == TaggedValue::Json));
    let tool = tool(schema);
    let src = grammar_source(&syntax, &[&tool], &lazy()).unwrap();
    assert_eq!(src.matches(long.as_str()).count(), 1);
    let value = format!(r#"{{"k":"{long}"}}"#);
    let text = qwen_call(&[("p7", &value)]);
    assert!(accepts(&src, &text));
    assert_eq!(first_input(&syntax, &tool, &text)["p7"], json!({"k": long}));
    assert!(start.elapsed().as_secs() < 30, "{:?}", start.elapsed());
}

/// Every hostile shape the rechecks found, sized as they found it, is
/// refused by the default [`SchemaLimits`] before anything compiles
/// it — each in milliseconds, naming the limit it passed.
#[test]
fn hostile_requests_are_refused_up_front() {
    let limits = SchemaLimits::default();
    let props = |n: usize, schema: Value| -> Map<String, Value> {
        (0..n).map(|i| (format!("p{i}"), schema.clone())).collect()
    };
    let object = |props: Map<String, Value>| json!({"type": "object", "properties": props});
    let chain = |n: usize| -> Map<String, Value> {
        (0..n)
            .map(|i| {
                let next = format!("#/$defs/D{}", i + 1);
                (
                    format!("D{i}"),
                    json!({"anyOf": [{"$ref": next}, {"$ref": next}]}),
                )
            })
            .chain([(format!("D{n}"), json!({"const": "leaf"}))])
            .collect()
    };
    let big = "x".repeat(100_000);
    let mut memberless_chain = chain(60);
    memberless_chain.insert("D60".into(), json!({"type": "string"}));
    let cases: Vec<(&str, Vec<Tool>, Option<Value>, SchemaLimit)> = vec![
        (
            "400,000 optional properties",
            vec![tool(object(props(400_000, json!({}))))],
            None,
            SchemaLimit::Nodes,
        ),
        (
            "400,000 optional properties, as output",
            vec![],
            Some(object(props(400_000, json!({})))),
            SchemaLimit::Nodes,
        ),
        (
            "6000 params beside 6000 defs",
            vec![tool(json!({
                "type": "object",
                "properties": props(6000, json!({"type": "string"})),
                "$defs": props(6000, json!({})),
            }))],
            None,
            SchemaLimit::Params,
        ),
        (
            "500 params naming a 100 KB member",
            vec![tool(json!({
                "type": "object",
                "properties": props(500, json!({"$ref": "#/$defs/E"})),
                "$defs": {"E": {"enum": [big]}},
            }))],
            None,
            SchemaLimit::MemberBytes,
        ),
        (
            "500 params naming 16,000 bytes of members",
            vec![tool(json!({
                "type": "object",
                "properties": props(500, json!({"$ref": "#/$defs/E"})),
                "$defs": {"E": {"enum": ["y".repeat(16_000)]}},
            }))],
            None,
            SchemaLimit::TotalMemberBytes,
        ),
        (
            "15,000 tools",
            (0..15_000)
                .map(|i| {
                    Tool::builder(format!("t{i}"))
                        .description("d")
                        .schema(object(props(2, json!({"type": "string"}))))
                        .build()
                        .unwrap()
                })
                .collect(),
            None,
            SchemaLimit::Tools,
        ),
        (
            "an enum of 180,000 numbers",
            vec![],
            Some(json!({"enum": (0..180_000).collect::<Vec<_>>()})),
            SchemaLimit::Nodes,
        ),
        (
            "a doubling $ref chain 60 deep",
            vec![tool(json!({
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/D0"}},
                "$defs": chain(60),
            }))],
            None,
            SchemaLimit::TotalMemberBytes,
        ),
        (
            "a doubling $ref chain 60 deep, no members",
            vec![tool(json!({
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/D0"}},
                "$defs": memberless_chain,
            }))],
            None,
            SchemaLimit::Nodes,
        ),
        (
            "an enum of 3000",
            vec![],
            Some(json!({"enum": (0..3000).collect::<Vec<_>>()})),
            SchemaLimit::Width,
        ),
        (
            "100 objects in an anyOf, each alive through an integer",
            vec![],
            Some(json!({"anyOf": (0..100).map(|i| json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, format!("z{i}"): {}},
            })).collect::<Vec<_>>()})),
            SchemaLimit::Width,
        ),
        (
            "16 x 16 nested anyOf objects over 16 members",
            vec![tool(json!({
                "type": "object",
                "properties": {"x": nested_any_of(16, 16)},
            }))],
            None,
            SchemaLimit::Width,
        ),
        (
            "5000 defs",
            vec![tool(json!({
                "type": "object",
                "properties": {"x": {"$ref": "#/$defs/p0"}},
                "$defs": props(5000, json!({"type": "integer"})),
            }))],
            None,
            SchemaLimit::Defs,
        ),
    ];
    for (name, tools, output, limit) in cases {
        let start = Instant::now();
        let err =
            check_schemas(&tools, output.as_ref(), &limits).expect_err(name);
        assert_eq!(err.limit, limit, "{name}: {err}");
        assert!(
            start.elapsed().as_millis() < 2000,
            "{name}: {:?}",
            start.elapsed()
        );
    }
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
                source: SchemaError::TooComplex { what, limit },
                ..
            }) => assert!(
                (what, limit) == ("bytes", MAX_GRAMMAR_BYTES)
                    || (what, limit) == ("rules", MAX_GRAMMAR_RULES),
                "{what} {limit}"
            ),
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

/// Requests at the default [`SchemaLimits`] — the most a client may
/// send — compile in every dialect, classify, parse and match quickly,
/// or fail cleanly as too complex: the limits are set so that no
/// request inside them is a hang or a gigabyte.
#[test]
fn requests_at_the_limits_stay_cheap() {
    let limits = SchemaLimits::default();
    // 2,000 optional properties in one nested object.
    let wide = {
        let inner: Map<String, Value> =
            (0..2_000).map(|i| (format!("k{i}"), json!({}))).collect();
        json!({"type": "object", "properties": {
            "o": {"type": "object", "properties": inner},
        }})
    };
    // 512 tools of 8 parameters, two of them small enums.
    let many: Vec<Tool> = (0..limits.max_tools)
        .map(|t| {
            let props: Map<String, Value> = (0..8)
                .map(|p| match p {
                    0 | 1 => {
                        (format!("a{p}"), json!({"enum": ["x", "y", "z"]}))
                    }
                    _ => (format!("a{p}"), json!({"type": "string"})),
                })
                .collect();
            Tool::builder(format!("t{t}"))
                .description("d")
                .schema(json!({"type": "object", "properties": props}))
                .build()
                .unwrap()
        })
        .collect();
    let cases: Vec<(&str, Vec<Tool>)> = vec![
        (
            "512 params x 8 shared defs",
            vec![tool(shared_defs_schema())],
        ),
        ("2,000 nested optionals", vec![tool(wide.clone())]),
        ("512 tools", many),
    ];
    for (name, tools) in cases {
        check_schemas(&tools, None, &limits).expect(name);
        let refs: Vec<&Tool> = tools.iter().collect();
        for syntax in syntaxes() {
            let family = syntax.family;
            let start = Instant::now();
            let src = grammar_source(&syntax, &refs, &lazy());
            let compiled = src.as_ref().ok().map(|s| Grammar::parse(s));
            let elapsed = start.elapsed();
            eprintln!(
                "{name} / {family:?}: {} in {elapsed:?}",
                match (&src, &compiled) {
                    (Ok(s), Some(Ok(_))) =>
                        format!("{} bytes, parsed", s.len()),
                    (Err(e), _) => format!("refused: {e}"),
                    (_, Some(Err(e))) => format!("grammar refused: {e}"),
                    _ => unreachable!(),
                }
            );
            if let Err(e) = &src {
                assert!(e.to_string().contains("too complex"), "{name}: {e}");
            }
            assert!(elapsed.as_secs() < 10, "{name} / {family:?}: {elapsed:?}");
        }
    }
    // The shared case end to end in the tagged dialect: a call naming
    // the last parameter matches and reads back.
    let tool = tool(shared_defs_schema());
    let syntax = CallSyntax::qwen_xml();
    let start = Instant::now();
    let src = grammar_source(&syntax, &[&tool], &lazy()).unwrap();
    let text = qwen_call(&[("p0", "member_0_000"), ("p511", "member_7_099")]);
    assert!(accepts(&src, &text));
    let input = first_input(&syntax, &tool, &text);
    assert_eq!(input["p511"], json!("member_7_099"));
    eprintln!("shared tagged end to end: {:?}", start.elapsed());
    assert!(start.elapsed().as_secs() < 10, "{:?}", start.elapsed());
    // A wide object's matcher: 2,000 ways to go on after `{` and after
    // each member, inside [`SchemaLimits::max_width`] and so under the
    // matcher's own cap.
    let inner: Map<String, Value> =
        (0..2000).map(|i| (format!("k{i}"), json!({}))).collect();
    let src = compile(&json!({"type": "object", "properties": {
        "o": {"type": "object", "properties": inner},
    }}));
    let start = Instant::now();
    for ok in [r#"{"o":{}}"#, r#"{"o":{"k0":1,"k7":[],"k1999":{}}}"#] {
        assert!(accepts(&src, ok), "{ok}");
    }
    assert!(!accepts(&src, r#"{"o":{"k7":1,"k0":1}}"#));
    eprintln!("wide match: {:?}", start.elapsed());
    assert!(start.elapsed().as_secs() < 10, "{:?}", start.elapsed());
}

/// 512 parameters, each a `$ref` to one of 8 defs of 100 members:
/// ~820 KB of members counted per reference, inside the default 1 MiB.
fn shared_defs_schema() -> Value {
    let defs: Map<String, Value> = (0..8)
        .map(|d| {
            let members: Vec<String> =
                (0..100).map(|m| format!("member_{d}_{m:03}")).collect();
            (format!("E{d}"), json!({"enum": members}))
        })
        .collect();
    let props: Map<String, Value> = (0..512)
        .map(|p| {
            (
                format!("p{p}"),
                json!({"$ref": format!("#/$defs/E{}", p % 8)}),
            )
        })
        .collect();
    json!({"type": "object", "properties": props, "$defs": defs})
}

/// `{"anyOf": k objects {"a": {"anyOf": k objects {"b": enum of m}}}}`,
/// each object with a property of its own beside: every variant at a
/// level shares the prefix `{"a":` (`{"b":`), so all k × k × m members
/// are alive at once inside — nested alternatives multiply.
fn nested_any_of(k: usize, m: usize) -> Value {
    let members: Vec<String> = (0..m).map(|i| format!("m{i:04}")).collect();
    let level = |key: &str, inner: Value| -> Value {
        let variants: Vec<Value> = (0..k)
            .map(|i| {
                json!({
                    "type": "object",
                    "properties": {key: inner, format!("{key}{i}"): {"type": "integer"}},
                    "required": [key],
                })
            })
            .collect();
        json!({"anyOf": variants})
    };
    level("a", level("b", json!({"enum": members})))
}

/// The most matcher stacks alive at once while `src` reads `input`,
/// and whether it accepts it.
fn peak_stacks(src: &str, input: &str) -> (usize, bool) {
    let mut state = GrammarState::new(Arc::new(Grammar::parse(src).unwrap()));
    let mut peak = state.stack_depth();
    for b in input.bytes() {
        if state.advance_bytes(&[b]).is_err() {
            return (peak, false);
        }
        peak = peak.max(state.stack_depth());
    }
    (peak, state.is_complete())
}

/// The width limit keeps every grammar under the matcher's stack cap,
/// so a request inside [`SchemaLimits`] is never truncated (over-
/// restricted) by it. Each shape is filled to the default limit — the
/// widest an `enum`, an object, an `anyOf` of objects alive through an
/// integer (the most stacks per counted alternative measured), an
/// `anyOf` of arrays, and `anyOf`s nested to multiply — as a tool
/// parameter in every dialect and as structured output, and read with
/// a value naming the *last* alternative: never past `MAX_STACKS`, and
/// never past the width counted for it.
#[test]
fn width_limit_keeps_the_matcher_under_its_cap() {
    use crate::dialect::render_reference;
    use crate::sample::grammar::MAX_STACKS;
    use crate::schema_budget::width;
    let limit = SchemaLimits::default().max_width;
    assert!(2 * limit <= MAX_STACKS);
    // `(name, shape(n), the value naming the last alternative)`.
    type Shape = (&'static str, fn(usize) -> Value, fn(usize) -> Value);
    let shapes: [Shape; 6] = [
        (
            "enum",
            |n| json!({"enum": (0..n).map(|i| format!("m{i:05}")).collect::<Vec<_>>()}),
            |n| json!(format!("m{:05}", n - 1)),
        ),
        (
            "optional integers",
            |n| {
                json!({"type": "object", "properties": (0..n)
                .map(|i| (format!("k{i:05}"), json!({"type": "integer"})))
                .collect::<Map<String, Value>>()})
            },
            |n| json!({format!("k{:05}", n - 1): 12}),
        ),
        (
            "required strings",
            |n| {
                let keys: Vec<String> =
                    (0..n).map(|i| format!("k{i:05}")).collect();
                json!({"type": "object", "required": keys, "properties": keys
                    .iter()
                    .map(|k| (k.clone(), json!({"type": "string"})))
                    .collect::<Map<String, Value>>()})
            },
            |n| {
                Value::Object(
                    (0..n).map(|i| (format!("k{i:05}"), json!("v"))).collect(),
                )
            },
        ),
        (
            "anyOf of optional-integer objects",
            |n| {
                json!({"anyOf": (0..n).map(|i| json!({
                "type": "object",
                "properties": {"a": {"type": "integer"}, format!("z{i}"): {"type": "integer"}},
            })).collect::<Vec<_>>()})
            },
            |n| json!({"a": 1234, format!("z{}", n - 1): 5}),
        ),
        (
            "anyOf of integer arrays",
            |n| {
                json!({"anyOf": (0..n).map(|i| json!({
                "type": "array", "items": {"type": "integer"}, "description": i,
            })).collect::<Vec<_>>()})
            },
            |_| json!([12, 345]),
        ),
        (
            "nested anyOf, k = 4",
            |m| nested_any_of(4, m),
            |m| json!({"a": {"b": format!("m{:04}", m - 1), "b3": 1}, "a3": 2}),
        ),
    ];
    // The widest `n` whose `wrap`ped schema is within the limit.
    let fill = |shape: fn(usize) -> Value, wrap: &dyn Fn(Value) -> Value| {
        let (mut lo, mut hi) = (1, 2 * limit);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            match width(&wrap(shape(mid))) <= limit {
                true => lo = mid,
                false => hi = mid - 1,
            }
        }
        lo
    };
    for (name, shape, value) in shapes {
        // Structured output: the schema as it is.
        let n = fill(shape, &|s| s);
        let schema = shape(n);
        check_schemas([], Some(&schema), &SchemaLimits::default()).expect(name);
        let counted = width(&schema);
        let (peak, ok) = peak_stacks(&compile(&schema), &value(n).to_string());
        eprintln!("{name} (n = {n}) as output: {peak} stacks of {counted}");
        assert!(ok, "{name}: output refused");
        assert!(peak <= counted && peak < MAX_STACKS, "{name}: {peak}");

        // A tool parameter, in every dialect.
        let wrap = |s: Value| json!({"type": "object", "properties": {"x": s}, "required": ["x"]});
        let n = fill(shape, &wrap);
        let t = tool(wrap(shape(n)));
        check_schemas([&t], None, &SchemaLimits::default()).expect(name);
        let counted = width(&t.schema);
        let input = json!({"x": value(n)});
        for syntax in syntaxes() {
            let family = syntax.family;
            let src = grammar_source(&syntax, &[&t], &lazy()).unwrap();
            let mut text = render_reference(&syntax, &[("t", &input)]).unwrap();
            if family == crate::dialect::Family::TagWithDict {
                text.push_str("<|tool_response>");
            }
            let (mut peak, mut ok) = peak_stacks(&src, &text);
            if !ok && family == crate::dialect::Family::TagWithTagged {
                // A set past 1024 members is spelled as JSON, not raw.
                let json = value(n).to_string();
                (peak, ok) = peak_stacks(&src, &qwen_call(&[("x", &json)]));
            }
            eprintln!(
                "{name} (n = {n}) in {family:?}: {peak} stacks of {counted}"
            );
            assert!(ok, "{name} / {family:?}: call refused");
            assert!(
                peak <= counted && peak < MAX_STACKS,
                "{name} / {family:?}: {peak}"
            );
        }
    }
}
