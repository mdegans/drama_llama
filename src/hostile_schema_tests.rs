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
        ..Default::default()
    }
}

/// [`lazy`] without the up-front measure ([`SchemaLimits::unlimited`]):
/// a caller that skips it, whom the pipelines' own caps still bound.
fn unmeasured() -> EmitOptions {
    EmitOptions {
        schema_limits: SchemaLimits::unlimited(),
        ..lazy()
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
    let src = grammar_source(
        &CallSyntax::qwen_xml(),
        &[&tool(schema)],
        &unmeasured(),
    )
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
    let src = grammar_source(&CallSyntax::qwen_xml(), &[&tool], &unmeasured())
        .unwrap();
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
    let src = grammar_source(&syntax, &[&tool], &unmeasured()).unwrap();
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
    let src = grammar_source(&syntax, &[&tool], &unmeasured()).unwrap();
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
    let src = grammar_source(&syntax, &[&tool], &unmeasured()).unwrap();
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
            "200 one-variant anyOf chains, 200 deep",
            vec![],
            Some(json!({"anyOf": (0..200).map(|i| (0..200)
                .fold(json!({"const": i}), |s, _| json!({"anyOf": [s]})))
                .collect::<Vec<_>>()})),
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

/// A schema too complex for a grammar never reaches a compiler from a
/// public entry point: each one — every dialect's `grammar_source`,
/// strict `tool_choice`, `output_config`, unified and phase-split —
/// measures it against its options' [`SchemaLimits`] first and refuses
/// it as a budget error. A caller that lifts the limits gets a schema
/// error from the compiler's own caps instead — never a half-gigabyte
/// grammar.
#[test]
fn too_complex_schema_fails_cleanly_everywhere() {
    use crate::output_config::compile_output_config;
    let n = 400_000;
    let props: Map<String, Value> =
        (0..n).map(|i| (format!("p{i}"), json!({}))).collect();
    let schema = json!({"type": "object", "properties": props});
    let tool = tool(schema.clone());
    let config = OutputConfig::json_schema(schema);
    let unlimited = SchemaLimits::unlimited();
    // Phase-split needs a trigger it is sure of: a pre-opened thought.
    let split = OutputConfigOptions::default();
    for syntax in syntaxes() {
        let family = syntax.family;
        match grammar_source(&syntax, &[&tool], &lazy()) {
            Err(DialectError::SchemaBudget(e)) => {
                assert_eq!(e.limit, SchemaLimit::Nodes, "{family:?}")
            }
            other => panic!("{family:?}: {:?}", other.map(|s| s.len())),
        }
        match grammar_source(&syntax, &[&tool], &unmeasured()) {
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
    let choice = |schema_limits| {
        grammar_for_tool_choice(
            &ToolChoice::any(),
            std::slice::from_ref(&tool),
            &ToolChoiceOptions {
                schema_limits,
                ..ToolChoiceOptions::default()
            },
            false,
        )
        .unwrap_err()
    };
    let err = choice(SchemaLimits::default());
    assert!(matches!(err, ToolChoiceError::SchemaBudget(_)), "{err}");
    let err = choice(unlimited);
    assert!(matches!(err, ToolChoiceError::Schema { .. }), "{err}");
    assert!(err.to_string().contains("too complex"), "{err}");
    let output = |schema_limits| OutputConfigOptions {
        schema_limits,
        ..split.clone()
    };
    for pre_opened in [false, true] {
        let err = grammar_for_output_config(
            &config,
            &output(SchemaLimits::default()),
            pre_opened,
        )
        .unwrap_err();
        assert!(matches!(err, OutputConfigError::SchemaBudget(_)), "{err}");
        let Err(err) = compile_output_config(
            &config,
            &output(SchemaLimits::default()),
            pre_opened,
        ) else {
            panic!("compiled");
        };
        assert!(matches!(err, OutputConfigError::SchemaBudget(_)), "{err}");
        let Err(err) =
            compile_output_config(&config, &output(unlimited), pre_opened)
        else {
            panic!("compiled");
        };
        assert!(matches!(err, OutputConfigError::Schema(_)), "{err}");
        assert!(err.to_string().contains("too complex"), "{err}");
    }
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
                    "additionalProperties": false,
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

/// The schema check's step budget never binds a request inside
/// [`SchemaLimits`]: its worst cases — an `enum`, `anyOf`s nested to
/// multiply, a wide `anyOf` of one-variant chains, a chain of defs each
/// an alternation, objects that all declare the property being judged
/// — filled to the width limit and checked over 2,000 values each, stay
/// well inside it, valid output passing and invalid output (only the
/// last value wrong) caught, so the check judged every value.
#[test]
fn schema_check_at_the_limits_stays_in_budget() {
    use crate::schema_budget::width;
    use crate::schema_check::check_counted;
    let limit = SchemaLimits::default().max_width;
    // An array of `items`, its `$defs` hoisted to the root they resolve
    // from.
    let array_of = |mut items: Value| {
        let defs = items.as_object_mut().and_then(|o| o.remove("$defs"));
        let mut array = json!({"type": "array", "items": items});
        if let Some(defs) = defs {
            array["$defs"] = defs;
        }
        array
    };
    // `(name, shape(n), a value naming its last alternative, a value
    // matching none)`.
    type Shape = (
        &'static str,
        fn(usize) -> Value,
        fn(usize) -> Value,
        fn(usize) -> Value,
    );
    let shapes: [Shape; 5] = [
        (
            "enum",
            |n| json!({"enum": (0..n).map(|i| format!("m{i:05}")).collect::<Vec<_>>()}),
            |n| json!(format!("m{:05}", n - 1)),
            |_| json!("zz"),
        ),
        (
            "nested anyOf, k = 4",
            |m| nested_any_of(4, m),
            |m| json!({"a": {"b": format!("m{:04}", m - 1), "b3": 1}, "a3": 2}),
            |_| json!({"a": {"b": "zz", "b3": 1}, "a3": 2}),
        ),
        (
            "anyOf of one-variant chains 50 deep",
            |n| {
                let chain = |i: usize| {
                    (0..50)
                        .fold(json!({"const": i}), |s, _| json!({"anyOf": [s]}))
                };
                json!({"anyOf": (0..n).map(chain).collect::<Vec<_>>()})
            },
            |n| json!(n - 1),
            |_| json!(-1),
        ),
        (
            // 18 members a level, so the chain fills the width before
            // the check's depth cap (where it stops judging) is near.
            "defs chained through anyOf",
            |n| {
                let mut defs: Map<String, Value> = (0..n)
                    .map(|i| {
                        let next = format!("#/$defs/D{}", i + 1);
                        let members: Vec<usize> = (i * 18..i * 18 + 18).collect();
                        (
                            format!("D{i}"),
                            json!({"anyOf": [{"$ref": next}, {"enum": members}]}),
                        )
                    })
                    .collect();
                defs.insert(format!("D{n}"), json!({"const": "end"}));
                json!({"$ref": "#/$defs/D0", "$defs": defs})
            },
            |_| json!("end"),
            |_| json!("x"),
        ),
        (
            "objects all declaring the judged property",
            |n| {
                json!({"anyOf": (0..n).map(|i| json!({
                    "type": "object",
                    "properties": {
                        "a": {"enum": [0, 1, 2, 3, 4, 5, 6, 7]},
                        format!("z{i}"): {"type": "integer"},
                    },
                    "required": ["a", format!("z{i}")],
                    "additionalProperties": false,
                })).collect::<Vec<_>>()})
            },
            |n| json!({"a": 7, format!("z{}", n - 1): 1}),
            |_| json!({"a": 7, "zz": 1}),
        ),
    ];
    for (name, shape, valid, invalid) in shapes {
        let (mut lo, mut hi) = (1, 2 * limit);
        while lo < hi {
            let mid = (lo + hi).div_ceil(2);
            match width(&array_of(shape(mid))) <= limit {
                true => lo = mid,
                false => hi = mid - 1,
            }
        }
        let schema = array_of(shape(lo));
        check_schemas([], Some(&schema), &SchemaLimits::default()).expect(name);
        let mut items = vec![valid(lo); 2000];
        let start = Instant::now();
        let (verdict, steps, budget) =
            check_counted(&schema, &Value::Array(items.clone()));
        eprintln!(
            "{name} (n = {lo}, width {}): {steps} steps of {budget}, {} per \
             value, in {:?}",
            width(&schema),
            steps / items.len(),
            start.elapsed(),
        );
        assert_eq!(verdict, Ok(()), "{name}");
        assert!(steps <= budget / 2, "{name}: {steps} of {budget}");
        *items.last_mut().unwrap() = invalid(lo);
        let (verdict, steps, budget) =
            check_counted(&schema, &Value::Array(items));
        assert!(steps <= budget / 2, "{name}: {steps} of {budget}");
        assert_eq!(verdict.unwrap_err().path, "/1999", "{name}");
        assert!(start.elapsed().as_secs() < 10, "{:?}", start.elapsed());
    }
}

/// The most resident memory, in MB, the footprint guards allow the test
/// process: the hostile rechecks' line for "exhausts memory".
pub(crate) const FOOTPRINT_LIMIT_MB: u64 = 1200;

/// The most any one step of a footprint guard may take. The rechecks'
/// line is a second; this leaves room for a slower machine.
pub(crate) const FOOTPRINT_STEP_SECS: f64 = 2.0;

/// This process's resident set, in MB, as `ps` reports it (Linux and
/// macOS alike, no `unsafe`), or 0 when `ps` is unavailable.
pub(crate) fn rss_mb() -> u64 {
    std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &std::process::id().to_string()])
        .output()
        .ok()
        .and_then(|out| {
            String::from_utf8_lossy(&out.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map_or(0, |kib| kib / 1024)
}

/// Run `step`, then hold it to [`FOOTPRINT_STEP_SECS`] and the process
/// to [`FOOTPRINT_LIMIT_MB`], logging both.
pub(crate) fn guarded<T>(name: &str, step: impl FnOnce() -> T) -> T {
    let start = Instant::now();
    let out = step();
    let (took, rss) = (start.elapsed(), rss_mb());
    eprintln!("footprint: {name}: {took:?}, {rss} MB resident");
    assert!(took.as_secs_f64() < FOOTPRINT_STEP_SECS, "{name}: {took:?}");
    assert!(rss < FOOTPRINT_LIMIT_MB, "{name}: {rss} MB resident");
    out
}

/// Footprint guard for the schema pipelines — the hostile rechecks'
/// probes (`adv`, `guard.sh`), committed. Requests at the default
/// [`SchemaLimits`] and past them go through every pipeline a client's
/// schema reaches: the up-front measure, each dialect's compile (with
/// the tagged dialect's classifier) and the grammar parse, the parse of
/// a large call, the schema check over ~32k tokens of output, and the
/// matcher over a value naming the last alternative. Each step must
/// finish in [`FOOTPRINT_STEP_SECS`] and the process stay under
/// [`FOOTPRINT_LIMIT_MB`] resident. Its matcher counterpart, which
/// feeds adversarial byte streams through a vocabulary filter, is
/// `sample::grammar::tests::footprint_guard_matcher`.
///
/// Ignored because it measures the whole process, so it must run
/// alone: under nextest (a process per test), e.g. in the nightly/GPU
/// window, `cargo nextest run --run-ignored only -E
/// 'test(footprint_guard)'`; under `cargo test`, with
/// `--test-threads=1`. CPU only, no model.
#[test]
#[ignore = "footprint guard: measures process RSS, so run it alone"]
fn footprint_guard_pipelines() {
    use crate::schema_check::check_counted;
    let limits = SchemaLimits::default();
    let wide = {
        let inner: Map<String, Value> =
            (0..2_000).map(|i| (format!("k{i}"), json!({}))).collect();
        json!({"type": "object", "properties": {
            "o": {"type": "object", "properties": inner},
        }})
    };
    let many: Vec<Tool> = (0..512)
        .map(|i| {
            Tool::builder(format!("t{i}"))
                .description("d")
                .schema(json!({
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"enum": ["x", "y"]}},
                    "required": ["a"],
                }))
                .build()
                .unwrap()
        })
        .collect();
    let enum_tool = tool(json!({
        "type": "object",
        "properties": {"x": {"enum": (0..2040)
            .map(|i| format!("m{i:05}"))
            .collect::<Vec<_>>()}},
        "required": ["x"],
    }));
    let requests: Vec<(&str, Vec<Tool>)> = vec![
        (
            "512 params x 8 shared defs",
            vec![tool(shared_defs_schema())],
        ),
        ("2,000 nested optionals", vec![tool(wide)]),
        ("512 tools", many),
        ("a 2,040-member enum", vec![enum_tool]),
        (
            "nested anyOf at the width limit",
            vec![tool(json!({
                "type": "object",
                "properties": {"x": nested_any_of(4, 119)},
                "required": ["x"],
            }))],
        ),
    ];
    for (name, tools) in &requests {
        let refs: Vec<&Tool> = tools.iter().collect();
        guarded(&format!("{name}: measure"), || {
            check_schemas(refs.iter().copied(), None, &limits).expect(name)
        });
        for syntax in syntaxes() {
            let family = syntax.family;
            let src = guarded(&format!("{name}: {family:?} compile"), || {
                grammar_source(&syntax, &refs, &lazy())
            });
            if let Ok(src) = src {
                guarded(&format!("{name}: {family:?} grammar parse"), || {
                    Grammar::parse(&src).expect(name)
                });
            }
        }
    }

    // A Qwen call naming all 512 parameters, ~100 KB, parsed.
    let shared = tool(shared_defs_schema());
    let params: Vec<(String, String)> = (0..512)
        .map(|p| (format!("p{p}"), format!("member_{}_{:03}", p % 8, p % 100)))
        .collect();
    let refs: Vec<(&str, &str)> = params
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect();
    let call = qwen_call(&refs).repeat(8);
    let parsed = guarded("512-param Qwen calls x 8: parse", || {
        parse_text(
            &CallSyntax::qwen_xml(),
            &[&shared],
            &call,
            false,
            Leniency::Final,
        )
    });
    assert_eq!(
        parsed
            .blocks
            .iter()
            .filter(|b| matches!(b, Block::ToolUse { .. }))
            .count(),
        8
    );

    // The schema check and the matcher over ~32k tokens of output for
    // the widest nested anyOf: 3,200 values each naming the last
    // alternative (~40 bytes apiece, ~128 KB), the check's last wrong.
    let schema = json!({"type": "array", "items": nested_any_of(4, 119)});
    check_schemas([], Some(&schema), &limits).expect("inside the limits");
    let item = json!({"a": {"b": "m0118", "b3": 1}, "a3": 2});
    let mut items = vec![item; 3_200];
    let text = Value::Array(items.clone()).to_string();
    let mut state =
        GrammarState::new(Arc::new(Grammar::parse(&compile(&schema)).unwrap()));
    // A step is 4 KB (~1k tokens), each token's advance its own.
    for (i, chunk) in text.as_bytes().chunks(4 << 10).enumerate() {
        guarded(&format!("nested anyOf: match 4 KB #{i}"), || {
            state.advance_bytes(chunk).expect("admitted")
        });
    }
    assert!(state.is_complete());
    *items.last_mut().unwrap() = json!({"a": {"b": "zz", "b3": 1}, "a3": 2});
    let (verdict, steps, budget) =
        guarded("nested anyOf: schema check", || {
            check_counted(&schema, &Value::Array(items))
        });
    assert!(verdict.is_err() && steps <= budget, "{steps} of {budget}");

    // Past the limits: refused up front, cheaply.
    let props: Map<String, Value> =
        (0..400_000).map(|i| (format!("p{i}"), json!({}))).collect();
    let hostile = tool(json!({"type": "object", "properties": props}));
    guarded("400,000 properties: refused", || {
        check_schemas([&hostile], None, &limits).unwrap_err()
    });
}

/// A call to `t` whose untyped parameter `p` is the text `value`,
/// spliced into `render_reference`'s bytes — a value too deep to build
/// as a [`Value`] (serializing one recurses) still reaches the parser.
fn call_with_raw_param(syntax: &CallSyntax, value: &str) -> String {
    let call = crate::dialect::render_reference(
        syntax,
        &[("t", &json!({"p": 12345}))],
    )
    .unwrap();
    call.replacen("12345", value, 1)
}

/// `levels` containers around `1`: arrays, or objects keyed `k` in
/// `syntax`'s spelling (bare keys in the dict encoding).
fn nested_text(syntax: &CallSyntax, levels: usize, objects: bool) -> String {
    let (open, close) = match (objects, syntax.family) {
        (false, _) => ("[", "]"),
        (true, crate::dialect::Family::TagWithDict) => ("{k:", "}"),
        (true, _) => (r#"{"k":"#, "}"),
    };
    format!("{}1{}", open.repeat(levels), close.repeat(levels))
}

/// Brackets nested far past what any reader takes, in an untyped
/// parameter: every dialect's batch, streamed and clipped parses, on a
/// tokio worker's 2 MiB stack, refuse it rather than recurse. The Gemma
/// 4 dict reader overflowed at about 2,500 levels — inside what its
/// grammar then admitted — and the readers of a call cut short at about
/// 3,800; an overflow aborts the process, every request on the server
/// with it. Nesting the grammar admits still reads back exactly.
#[test]
fn deep_nesting_never_overflows_a_parser() {
    use crate::dialect::StreamParser;
    use crate::grammar_compile::UNTYPED_DEPTH;
    let syntaxes = [
        CallSyntax::qwen_xml(),
        CallSyntax::hermes_json(),
        CallSyntax::llama31_json(),
        CallSyntax::gemma4(),
        CallSyntax::gpt_oss(),
    ];
    let t = tool(json!({
        "type": "object",
        "properties": {"p": {}},
        "required": ["p"],
    }));
    for syntax in syntaxes {
        for objects in [false, true] {
            // Exactly what the grammar admits, and no deeper: read back
            // whole.
            let grammar = Arc::new(
                Grammar::parse(
                    &grammar_source(&syntax, &[&t], &lazy()).unwrap(),
                )
                .unwrap(),
            );
            let admits = |levels| {
                let value = nested_text(&syntax, levels, objects);
                let text = call_with_raw_param(&syntax, &value);
                // Gemma's grammar goes on past the call (its turn exit),
                // so admitted is all a call can be.
                let mut state = GrammarState::new(grammar.clone());
                state.advance_bytes(text.as_bytes()).is_ok()
            };
            assert!(admits(UNTYPED_DEPTH), "{:?}", syntax.family);
            assert!(!admits(UNTYPED_DEPTH + 1), "{:?}", syntax.family);
            let text = call_with_raw_param(
                &syntax,
                &nested_text(&syntax, UNTYPED_DEPTH, objects),
            );
            let mut expected = json!(1);
            for _ in 0..UNTYPED_DEPTH {
                expected = match objects {
                    false => json!([expected]),
                    true => json!({"k": expected}),
                };
            }
            assert_eq!(
                first_input(&syntax, &t, &text),
                json!({"p": expected}),
                "{:?}",
                syntax.family
            );

            let deep = call_with_raw_param(
                &syntax,
                &nested_text(&syntax, 5_000, objects),
            );
            let (syntax, t) = (syntax.clone(), t.clone());
            on_small_stack(2048, move || {
                for leniency in [Leniency::Final, Leniency::Clipped] {
                    for text in [&deep[..], &deep[..deep.len() / 2]] {
                        parse_text(&syntax, &[&t], text, false, leniency);
                    }
                }
                let mut stream =
                    StreamParser::new(syntax.clone(), vec![t.clone()], false);
                let half = deep.len() / 2;
                for chunk in deep.as_bytes()[..half].chunks(1024) {
                    stream.push(std::str::from_utf8(chunk).unwrap());
                }
                stream.clone().finish_clipped();
                stream.push(&deep[half..]);
                stream.finish();
            });
        }
    }
}

/// The depth bounds add up: a schema at
/// [`SchemaLimits::max_depth`](crate::SchemaLimits::max_depth), an
/// untyped value at its deepest point nested as deep as the grammar
/// lets it ([`UNTYPED_DEPTH`](crate::grammar_compile::UNTYPED_DEPTH)),
/// and the call envelope around it (two levels at most: an array of
/// `{"name", "arguments"}` objects) stay inside what every parser reads
/// ([`MAX_NESTING`](crate::grammar_compile::MAX_NESTING), serde_json's
/// own limit) — so whatever the grammar of a schema inside the limits
/// admits, every dialect reads back.
#[test]
fn depth_budget_fits_the_parsers() {
    use crate::grammar_compile::{MAX_NESTING, UNTYPED_DEPTH};
    const ENVELOPE: usize = 2;
    let max_depth = SchemaLimits::default().max_depth;
    assert!(max_depth + UNTYPED_DEPTH + ENVELOPE <= MAX_NESTING);
    // serde_json's limit is MAX_NESTING, as the parsers assume.
    let nested = |n: usize| format!("{}{}", "[".repeat(n), "]".repeat(n));
    assert!(serde_json::from_str::<Value>(&nested(MAX_NESTING)).is_ok());
    assert!(serde_json::from_str::<Value>(&nested(MAX_NESTING + 1)).is_err());

    // `{"root": D0}`, `D0`…`D62` objects each requiring the next, `D63`
    // untyped: 64 levels of schema.
    let n = max_depth - 1;
    let mut defs: Map<String, Value> = (0..n)
        .map(|i| {
            let next = format!("#/$defs/D{}", i + 1);
            let def = json!({
                "type": "object",
                "properties": {"a": {"$ref": next}},
                "required": ["a"],
            });
            (format!("D{i}"), def)
        })
        .collect();
    defs.insert(format!("D{n}"), json!({}));
    let schema = json!({
        "type": "object",
        "properties": {"root": {"$ref": "#/$defs/D0"}},
        "required": ["root"],
        "$defs": defs,
    });
    let t = tool(schema.clone());
    check_schemas([&t], None, &SchemaLimits::default()).expect("at the limit");
    let mut deepest = json!(1);
    for _ in 0..UNTYPED_DEPTH {
        deepest = json!([deepest]);
    }
    for _ in 0..n {
        deepest = json!({"a": deepest});
    }
    let value = json!({"root": deepest});
    let wrapped = json!([{"name": "t", "arguments": value}]).to_string();
    assert!(serde_json::from_str::<Value>(&wrapped).is_ok());

    for syntax in [
        CallSyntax::qwen_xml(),
        CallSyntax::hermes_json(),
        CallSyntax::llama31_json(),
        CallSyntax::gemma4(),
        CallSyntax::gpt_oss(),
    ] {
        let family = syntax.family;
        let src = grammar_source(&syntax, &[&t], &lazy()).unwrap();
        let mut state =
            GrammarState::new(Arc::new(Grammar::parse(&src).unwrap()));
        let text = crate::dialect::render_reference(&syntax, &[("t", &value)])
            .unwrap();
        state.advance_bytes(text.as_bytes()).expect("admitted");
        assert_eq!(first_input(&syntax, &t, &text), value, "{family:?}");
    }
    let text = value.to_string();
    assert!(accepts(&compile(&schema), &text));
    assert_eq!(crate::schema_check::check_text(&schema, &text), Ok(()));
}

/// What the grammars admit past serde's reach is refused by every
/// parser, never read back as a string. A recursive `$ref` nests as
/// deep as the model takes it (a back-reference costs no depth), and a
/// JSON number has no digit limit: Qwen XML read either as the raw
/// text, a *string* — which a union admitting strings even passed the
/// schema check as. At the boundary, the deepest value serde reads
/// comes back whole.
#[test]
fn values_past_serde_are_refused_not_retyped() {
    use crate::grammar_compile::MAX_NESTING;
    let calls = |syntax: &CallSyntax, t: &Tool, text: &str| -> Vec<Value> {
        parse_text(syntax, &[t], text, false, Leniency::Final)
            .blocks
            .into_iter()
            .filter_map(|b| match b {
                Block::ToolUse { call } => Some(call.input),
                _ => None,
            })
            .collect()
    };
    let admits = |syntax: &CallSyntax, t: &Tool, text: &str| {
        let src = grammar_source(syntax, &[t], &lazy()).unwrap();
        let mut state =
            GrammarState::new(Arc::new(Grammar::parse(&src).unwrap()));
        state.advance_bytes(text.as_bytes()).is_ok()
    };
    let nested = |n: usize| (0..n).fold(json!([]), |inner, _| json!([inner]));

    // A tree of arrays, alone and in a union with a string.
    let node = json!({"type": "array", "items": {"$ref": "#/$defs/Node"}});
    for root in [
        json!({"$ref": "#/$defs/Node"}),
        json!({"anyOf": [{"type": "string"}, {"$ref": "#/$defs/Node"}]}),
    ] {
        let schema = json!({
            "type": "object",
            "properties": {"root": root},
            "required": ["root"],
            "$defs": {"Node": node},
        });
        let t = tool(schema);
        check_schemas([&t], None, &SchemaLimits::default()).expect("in limits");
        let qwen = CallSyntax::qwen_xml();
        // `nested(n)` is n + 1 levels; the parameter's value is read on
        // its own, so its last readable depth is serde's.
        let deepest = json!({"root": nested(MAX_NESTING - 1)});
        let text = crate::dialect::render_reference(&qwen, &[("t", &deepest)])
            .unwrap();
        assert!(admits(&qwen, &t, &text));
        assert_eq!(calls(&qwen, &t, &text), [deepest], "{root}");

        let past = json!({"root": nested(MAX_NESTING)});
        for syntax in syntaxes() {
            let text =
                crate::dialect::render_reference(&syntax, &[("t", &past)])
                    .unwrap();
            assert!(admits(&syntax, &t, &text), "{:?}", syntax.family);
            assert_eq!(
                calls(&syntax, &t, &text),
                Vec::<Value>::new(),
                "{:?} {root}",
                syntax.family
            );
        }
    }

    // A number past `f64`, in every parameter shape that admits one.
    let huge = "9".repeat(401);
    for x in [
        json!({"type": "number"}),
        json!({"type": ["number", "string"]}),
        json!({"anyOf": [{"type": "number"}, {"type": "string"}]}),
        json!({}),
    ] {
        let t = tool(json!({
            "type": "object",
            "properties": {"x": x},
            "required": ["x"],
        }));
        let qwen = CallSyntax::qwen_xml();
        let text = qwen_call(&[("x", &huge)]);
        assert!(admits(&qwen, &t, &text), "{x}");
        assert_eq!(calls(&qwen, &t, &text), Vec::<Value>::new(), "{x}");
    }
}

/// A `required` naming one property 100,000 times, or a `type` listing
/// `"object"` as often, is inside every limit (one property, one type),
/// and must cost as little to judge: the check rebuilt and walked the
/// whole list for every object it judged — 4.4 s on 64 KB of output,
/// 8.4 s on 128 KB, past its step budget, which no request inside the
/// limits may reach. The names are gathered once per check. The
/// compilers read each name once too: an undeclared name listed twice
/// was a key the grammar made the model write twice, and a type listed
/// twice a union that compiled to any value at all.
#[test]
fn duplicate_names_cost_once() {
    use crate::schema_check::check_counted;
    let n = 100_000;
    let required: Vec<Value> = (0..n).map(|_| json!("a")).collect();
    let item = json!({
        "type": "object",
        "properties": {"a": {"type": "integer"}},
        "required": required,
    });
    let mut types: Vec<Value> = (0..n).map(|_| json!("object")).collect();
    types.push(json!("array"));
    for item in [item, json!({"type": types})] {
        let schema = json!({
            "type": "object",
            "properties": {"xs": {"type": "array", "items": item}},
            "required": ["xs"],
        });
        check_schemas([&tool(schema.clone())], None, &SchemaLimits::default())
            .expect("inside the limits");
        // ~128 KB of output, ~32k tokens; the last element wrong.
        let mut xs: Vec<Value> = (0..16_000).map(|i| json!({"a": i})).collect();
        *xs.last_mut().unwrap() = json!({"b": 1});
        let start = Instant::now();
        let (verdict, steps, budget) =
            check_counted(&schema, &json!({"xs": xs}));
        let elapsed = start.elapsed();
        assert!(steps <= budget, "{steps} of {budget}");
        assert!(elapsed.as_secs() < 2, "{elapsed:?}");
        // The `type` list admits `{"b": 1}`; `required` does not.
        let typed = schema["properties"]["xs"]["items"].get("required");
        assert_eq!(verdict.is_err(), typed.is_some());
    }

    // Undeclared, twice: one slot.
    let twice = json!({"type": "object", "required": ["x", "x"]});
    let src = compile(&twice);
    assert!(accepts(&src, r#"{"x":1}"#));
    assert!(!accepts(&src, r#"{"x":1,"x":1}"#));
    // A type, twice: still that type.
    let src = compile(&json!({"type": ["string", "string", "null"]}));
    assert!(accepts(&src, r#""s""#));
    assert!(!accepts(&src, "1"));
}

/// A string `enum` as wide as the measure lets one be is spelled raw on
/// Qwen XML, as the template writes it — not JSON-quoted, as one of
/// 1025 to 2048 members was while the raw spelling stopped at 1024.
#[test]
fn qwen_widest_string_set_is_raw() {
    let members: Vec<String> =
        (0..2040).map(|i| format!("Zone_{i:05}")).collect();
    let schema = json!({
        "type": "object",
        "properties": {"tz": {"enum": members}},
        "required": ["tz"],
    });
    let t = tool(schema.clone());
    check_schemas([&t], None, &SchemaLimits::default()).expect("inside");
    let syntax = CallSyntax::qwen_xml();
    assert!(matches!(
        tagged_values(&syntax, &schema)[0].1,
        TaggedValue::Choice(_)
    ));
    let src = grammar_source(&syntax, &[&t], &lazy()).unwrap();
    let raw = qwen_call(&[("tz", "Zone_02039")]);
    assert!(accepts(&src, &raw));
    assert!(!accepts(&src, &qwen_call(&[("tz", "\"Zone_02039\"")])));
    assert_eq!(first_input(&syntax, &t, &raw), json!({"tz": "Zone_02039"}));
}

/// A generation that floods its call trigger — 384 KB of it, ~32k
/// tokens, alone or inside an unclosed thought — parses in linear time,
/// in every dialect. Each trigger is a malformed call and a block
/// boundary, and the parser rescanned the rest of the text for every
/// landmark from each: quadratic, 10 s for one parse of Gemma 4's
/// `<|tool_call>` flood, and a stream re-parses per token.
#[test]
fn trigger_flood_parses_in_linear_time() {
    let t = tool(json!({
        "type": "object",
        "properties": {"p": {"type": "string"}},
        "required": ["p"],
    }));
    for syntax in [
        CallSyntax::qwen_xml(),
        CallSyntax::hermes_json(),
        CallSyntax::llama31_json(),
        CallSyntax::gemma4(),
        CallSyntax::gpt_oss(),
    ] {
        let trigger = syntax.trigger();
        if trigger.is_empty() {
            continue;
        }
        let flood = trigger.repeat((384 << 10) / trigger.len());
        let thought = format!("{}{flood}", syntax.reasoning.start);
        for text in [&flood, &thought] {
            let start = Instant::now();
            parse_text(&syntax, &[&t], text, false, Leniency::Final);
            let elapsed = start.elapsed();
            assert!(elapsed.as_secs() < 1, "{:?}: {elapsed:?}", syntax.family);
        }
    }
}
