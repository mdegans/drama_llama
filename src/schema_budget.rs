//! An up-front bound on a request's client-supplied JSON Schemas — every
//! custom tool's `input_schema` and an `output_config` `json_schema` —
//! checked before anything classifies or compiles them.
//!
//! Each pipeline downstream (the grammar compiler, the tagged dialect's
//! value classifier, the parser, the matcher) has caps of its own, but
//! those are per-path and were found one hostile schema at a time. This
//! is the structural one: a single linear, iterative walk that measures a
//! request in the dimensions its cost grows with, and refuses it — a 400
//! `invalid_request_error`, as Anthropic refuses tool definitions too
//! large for it — when any is past its [`SchemaLimits`].
//!
//! The dimensions:
//!
//! * **tools** advertised, and **parameters** (top-level `properties`)
//!   per tool: grammars and classifiers are per tool and per parameter.
//! * **nodes**: every JSON value in every schema, summed over the
//!   request — counted as written, and again with each `$ref` counted
//!   at its target's size, once per reference: the work a pipeline that
//!   reads a schema per parameter, or per use, would do.
//! * **`$defs`** (with `definitions`) per schema.
//! * **member bytes**: an `enum` member's or `const` value's size as
//!   compact JSON, alone and in total, the total also counting a `$ref`
//!   at its target's size per reference — so a large `enum` behind a
//!   `$ref` that two thousand parameters name is two thousand copies of
//!   it, not one.
//!
//! The defaults ([`SchemaLimits::default`]) leave real schemas far
//! inside every limit: see each field for what was measured.

use std::collections::{BTreeMap, HashMap};

use serde_json::{Map, Value};

use crate::Tool;

/// The most a request's schemas may measure ([`check_schemas`]).
///
/// The defaults are generous for real requests and tight enough that a
/// request at them compiles, classifies, parses and matches in well
/// under a second. Measured (2026-10-01) against Agora's seed-agent
/// request — its 15 tools plus the `Soul` output schema — every tool and
/// structured-output schema in misanthropic's captured request fixtures,
/// Anthropic's documented tool examples, and a heavier synthetic tool (a
/// 600-member time-zone `enum` behind a `$ref` four parameters name,
/// beside a 249-member country `enum`): the largest of them is 21× inside
/// every limit, Agora's request at least 34×. Each field says what the
/// largest measured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct SchemaLimits {
    /// Custom tools in one request. Default 512; Agora sends 15.
    pub max_tools: usize,
    /// Top-level `properties` of one tool's `input_schema`. Default
    /// 512; the most measured is 5.
    pub max_params: usize,
    /// JSON values across all of a request's schemas, both as written
    /// and with each `$ref` counted at its target's size per reference
    /// (a reference cycle counts its target once). Default 2^17
    /// (131,072); Agora's request has 406 (420 with its references
    /// counted), the synthetic tool 882 (2,690).
    pub max_nodes: usize,
    /// `$defs` plus `definitions` entries of one schema. Default 1024;
    /// the most measured is 5 (`Soul`).
    pub max_defs: usize,
    /// One `enum` member or `const` value, as compact JSON. Default 16
    /// KiB; the longest measured is 24 bytes.
    pub max_member_bytes: usize,
    /// Every `enum` member and `const` value of a request, a `$ref`
    /// counted at its target's size once per reference. Default 1 MiB;
    /// Agora's request has 304 bytes, the synthetic tool ~49 KB.
    pub max_total_member_bytes: usize,
}

impl Default for SchemaLimits {
    fn default() -> Self {
        Self {
            max_tools: 512,
            max_params: 512,
            max_nodes: 1 << 17,
            max_defs: 1024,
            max_member_bytes: 16 << 10,
            max_total_member_bytes: 1 << 20,
        }
    }
}

impl SchemaLimits {
    /// No limits: every schema passes ([`check_schemas`] still walks
    /// them).
    pub fn unlimited() -> Self {
        Self {
            max_tools: usize::MAX,
            max_params: usize::MAX,
            max_nodes: usize::MAX,
            max_defs: usize::MAX,
            max_member_bytes: usize::MAX,
            max_total_member_bytes: usize::MAX,
        }
    }

    /// Set [`Self::max_tools`].
    pub fn with_max_tools(mut self, n: usize) -> Self {
        self.max_tools = n;
        self
    }

    /// Set [`Self::max_params`].
    pub fn with_max_params(mut self, n: usize) -> Self {
        self.max_params = n;
        self
    }

    /// Set [`Self::max_nodes`].
    pub fn with_max_nodes(mut self, n: usize) -> Self {
        self.max_nodes = n;
        self
    }

    /// Set [`Self::max_defs`].
    pub fn with_max_defs(mut self, n: usize) -> Self {
        self.max_defs = n;
        self
    }

    /// Set [`Self::max_member_bytes`].
    pub fn with_max_member_bytes(mut self, n: usize) -> Self {
        self.max_member_bytes = n;
        self
    }

    /// Set [`Self::max_total_member_bytes`].
    pub fn with_max_total_member_bytes(mut self, n: usize) -> Self {
        self.max_total_member_bytes = n;
        self
    }
}

/// Which of the [`SchemaLimits`] a request is past.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SchemaLimit {
    /// [`SchemaLimits::max_tools`].
    Tools,
    /// [`SchemaLimits::max_params`].
    Params,
    /// [`SchemaLimits::max_nodes`].
    Nodes,
    /// [`SchemaLimits::max_defs`].
    Defs,
    /// [`SchemaLimits::max_member_bytes`].
    MemberBytes,
    /// [`SchemaLimits::max_total_member_bytes`].
    TotalMemberBytes,
}

impl SchemaLimit {
    /// What the limit counts, for the error message.
    fn counts(self) -> &'static str {
        match self {
            Self::Tools => "custom tools",
            Self::Params => "top-level properties",
            Self::Nodes => {
                "JSON values across the request's schemas, each `$ref` \
                 counted at its target's size"
            }
            Self::Defs => "`$defs` and `definitions` entries",
            Self::MemberBytes => "bytes in one `enum` member or `const` value",
            Self::TotalMemberBytes => {
                "bytes of `enum` members and `const` values across the \
                 request's schemas, each `$ref` counted at its target's size"
            }
        }
    }
}

/// A request whose schemas measure past a [`SchemaLimits`] field — the
/// request's fault, a 400 `invalid_request_error`.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "{location}: more than {max} {counts} (at least {actual}); simplify \
     the schema or split the request",
    counts = limit.counts()
)]
pub struct SchemaBudgetError {
    /// Where: `tools`, ``tool `name` input_schema``,
    /// `output_config.format.schema`, or `request` for a request-wide
    /// total.
    pub location: String,
    /// The limit passed.
    pub limit: SchemaLimit,
    /// How much was counted before the walk stopped — at least
    /// `max + 1`; a walk stops as soon as it is past.
    pub actual: usize,
    /// The limit's value.
    pub max: usize,
}

static_assertions::assert_impl_all!(SchemaBudgetError: Send, Sync);

/// Check every custom tool's `input_schema` in `prompt`, and its
/// `output_config` `json_schema` if any, against `limits`
/// ([`check_schemas`]).
pub fn check_prompt(
    prompt: &crate::Prompt,
    limits: &SchemaLimits,
) -> Result<(), SchemaBudgetError> {
    let tools = prompt.tools.iter().flatten().filter_map(|d| d.as_method());
    check_schemas(tools, crate::output_config::json_schema(prompt), limits)
}

/// Measure `tools`' input schemas and `output` (an `output_config`
/// schema) against `limits`, before anything compiles them. A few
/// iterative walks, each linear in the schemas (the first stops as soon
/// as the request is past [`SchemaLimits::max_nodes`], bounding the
/// rest), so a hostile request costs no more to refuse than to read.
pub fn check_schemas<'a>(
    tools: impl IntoIterator<Item = &'a Tool>,
    output: Option<&Value>,
    limits: &SchemaLimits,
) -> Result<(), SchemaBudgetError> {
    let tools: Vec<&Tool> = tools.into_iter().collect();
    let over = |location: String, limit, actual, max| SchemaBudgetError {
        location,
        limit,
        actual,
        max,
    };
    if tools.len() > limits.max_tools {
        return Err(over(
            "tools".into(),
            SchemaLimit::Tools,
            tools.len(),
            limits.max_tools,
        ));
    }
    let schemas: Vec<(String, &Value)> = tools
        .iter()
        .map(|t| (format!("tool `{}` input_schema", t.name), &t.schema))
        .chain(output.map(|s| ("output_config.format.schema".into(), s)))
        .collect();

    // Nodes as written first, across the request: it bounds every walk
    // after it.
    let mut nodes = 0usize;
    for (_, schema) in &schemas {
        nodes = count_nodes(schema, nodes, limits.max_nodes);
        if nodes > limits.max_nodes {
            return Err(over(
                "request".into(),
                SchemaLimit::Nodes,
                nodes,
                limits.max_nodes,
            ));
        }
    }

    for tool in &tools {
        let params = tool
            .schema
            .get("properties")
            .and_then(Value::as_object)
            .map_or(0, Map::len);
        if params > limits.max_params {
            return Err(over(
                format!("tool `{}` input_schema", tool.name),
                SchemaLimit::Params,
                params,
                limits.max_params,
            ));
        }
    }

    let (mut bytes, mut expanded) = (0usize, 0usize);
    for (location, schema) in &schemas {
        let units = Units::new(schema);
        if units.defs() > limits.max_defs {
            return Err(over(
                location.clone(),
                SchemaLimit::Defs,
                units.defs(),
                limits.max_defs,
            ));
        }
        let measured =
            units.measure(limits.max_member_bytes).map_err(|bytes| {
                over(
                    location.clone(),
                    SchemaLimit::MemberBytes,
                    bytes,
                    limits.max_member_bytes,
                )
            })?;
        bytes = bytes.saturating_add(measured.effective(&measured.bytes));
        if bytes > limits.max_total_member_bytes {
            return Err(over(
                "request".into(),
                SchemaLimit::TotalMemberBytes,
                bytes,
                limits.max_total_member_bytes,
            ));
        }
        expanded = expanded.saturating_add(measured.effective(&measured.nodes));
        if expanded > limits.max_nodes {
            return Err(over(
                "request".into(),
                SchemaLimit::Nodes,
                expanded,
                limits.max_nodes,
            ));
        }
    }
    Ok(())
}

/// `so_far` plus the JSON values in `value`, stopping once past `max`.
fn count_nodes(value: &Value, so_far: usize, max: usize) -> usize {
    let mut count = so_far;
    let mut stack = vec![value];
    while let Some(v) = stack.pop() {
        count += 1;
        if count > max {
            break;
        }
        match v {
            Value::Array(items) => stack.extend(items),
            Value::Object(map) => stack.extend(map.values()),
            _ => {}
        }
    }
    count
}

/// Keys whose values are data, not subschemas: a `$ref` or `enum`
/// inside them means nothing.
const DATA_KEYS: [&str; 5] =
    ["enum", "const", "default", "examples", "example"];

/// A schema split into units a `$ref` can name: the root (`#`) and each
/// root-level `$defs` / `definitions` entry.
struct Units<'s> {
    /// `(name as a $ref spells it, schema)`; index 0 is the root.
    units: Vec<(String, &'s Value)>,
    /// `$ref` string → unit index.
    by_ref: HashMap<String, usize>,
}

/// What [`Units::measure`] counts, per unit, and the order to combine
/// the units in.
struct Measured {
    /// Member bytes in each unit's own subtree.
    bytes: Vec<usize>,
    /// JSON values in each unit's own subtree.
    nodes: Vec<usize>,
    /// The units each unit references, with multiplicity, ordered (so a
    /// cycle is cut at the same edge every time).
    refs: Vec<Vec<(usize, usize)>>,
    /// The units the root reaches, in the order a depth-first walk of
    /// the reference graph from the root finishes them: each after
    /// every unit it references, except a reference back into a cycle
    /// still open — which is how a cycle counts its target once.
    order: Vec<usize>,
}

impl Measured {
    /// The root's `local` total, each reference counted at its target's
    /// total (saturating), a reference back into an open cycle at
    /// nothing.
    fn effective(&self, local: &[usize]) -> usize {
        let mut total: Vec<Option<usize>> = vec![None; local.len()];
        for &unit in &self.order {
            let sum =
                self.refs[unit].iter().fold(local[unit], |sum, &(t, n)| {
                    sum.saturating_add(total[t].unwrap_or(0).saturating_mul(n))
                });
            total[unit] = Some(sum);
        }
        total[0].expect("the order ends at the root")
    }
}

impl<'s> Units<'s> {
    fn new(root: &'s Value) -> Self {
        let mut units = vec![("#".to_string(), root)];
        for table in ["$defs", "definitions"] {
            let entries = root.get(table).and_then(Value::as_object);
            units.extend(
                entries
                    .into_iter()
                    .flatten()
                    .map(|(name, def)| (format!("#/{table}/{name}"), def)),
            );
        }
        let by_ref = units
            .iter()
            .enumerate()
            .map(|(i, (name, _))| (name.clone(), i))
            .collect();
        Self { units, by_ref }
    }

    /// Entries in the root's `$defs` and `definitions`.
    fn defs(&self) -> usize {
        self.units.len() - 1
    }

    /// The unit `node` references, if it is a schema with a `$ref` to
    /// one.
    fn target(&self, node: &Value) -> Option<usize> {
        let r = node.get("$ref")?.as_str()?;
        self.by_ref.get(r).copied()
    }

    /// Whether `key` of `node` in `unit` is the root's own `$defs` or
    /// `definitions` table, whose entries are units of their own.
    fn is_root_table(&self, unit: usize, node: &Value, key: &str) -> bool {
        unit == 0
            && std::ptr::eq(node, self.units[0].1)
            && (key == "$defs" || key == "definitions")
    }

    /// Each unit's member bytes, values and references: one walk over
    /// each unit's subtree. `Err(bytes)` for a member past
    /// `max_member`.
    fn measure(&self, max_member: usize) -> Result<Measured, usize> {
        let n = self.units.len();
        let mut bytes = vec![0usize; n];
        let mut nodes = vec![0usize; n];
        let mut refs: Vec<BTreeMap<usize, usize>> = vec![BTreeMap::new(); n];
        for unit in 0..n {
            let mut stack: Vec<&Value> = vec![self.units[unit].1];
            while let Some(node) = stack.pop() {
                nodes[unit] += 1;
                let Value::Object(map) = node else {
                    if let Value::Array(items) = node {
                        stack.extend(items);
                    }
                    continue;
                };
                let members = map
                    .get("enum")
                    .and_then(Value::as_array)
                    .into_iter()
                    .flatten()
                    .chain(map.get("const"));
                for member in members {
                    let len = json_len(member);
                    if len > max_member {
                        return Err(len);
                    }
                    bytes[unit] = bytes[unit].saturating_add(len);
                }
                if let Some(target) = self.target(node) {
                    *refs[unit].entry(target).or_default() += 1;
                }
                for (key, value) in map {
                    if self.is_root_table(unit, node, key) {
                        continue;
                    }
                    match DATA_KEYS.contains(&key.as_str()) {
                        // Data is values all the same, if not schemas.
                        true => {
                            nodes[unit] =
                                count_nodes(value, nodes[unit], usize::MAX)
                        }
                        false => stack.push(value),
                    }
                }
            }
        }
        let refs: Vec<Vec<(usize, usize)>> =
            refs.into_iter().map(|m| m.into_iter().collect()).collect();
        let order = finish_order(&refs);
        Ok(Measured {
            bytes,
            nodes,
            refs,
            order,
        })
    }
}

/// The units the root reaches, in depth-first finish order
/// ([`Measured::order`]).
fn finish_order(refs: &[Vec<(usize, usize)>]) -> Vec<usize> {
    let mut seen = vec![false; refs.len()];
    let mut order = Vec::new();
    // `(unit, next edge)`: the recursion's frames.
    let mut frames = vec![(0usize, 0usize)];
    seen[0] = true;
    while let Some(top) = frames.last_mut() {
        let (unit, edge) = *top;
        match refs[unit].get(edge) {
            Some(&(target, _)) => {
                top.1 += 1;
                if !seen[target] {
                    seen[target] = true;
                    frames.push((target, 0));
                }
            }
            None => {
                order.push(unit);
                frames.pop();
            }
        }
    }
    order
}

/// `value`'s length as compact JSON, as `serde_json` writes it, without
/// serializing.
fn json_len(value: &Value) -> usize {
    let mut len = 0usize;
    let mut stack = vec![value];
    while let Some(v) = stack.pop() {
        len = len.saturating_add(match v {
            Value::Null => 4,
            Value::Bool(true) => 4,
            Value::Bool(false) => 5,
            Value::Number(n) => n.to_string().len(),
            Value::String(s) => string_len(s),
            Value::Array(items) => {
                stack.extend(items);
                2 + items.len().saturating_sub(1)
            }
            Value::Object(map) => {
                stack.extend(map.values());
                let keys: usize = map.keys().map(|k| string_len(k) + 1).sum();
                2 + keys + map.len().saturating_sub(1)
            }
        });
    }
    len
}

/// A JSON string literal's length: quotes, and each escape at its
/// written length — two bytes for `"`, `\` and the control characters
/// with a short form (`\n`, `\t`, …), six for the rest (`\u001f`).
fn string_len(s: &str) -> usize {
    let escapes: usize = s
        .bytes()
        .map(|b| match b {
            b'"' | b'\\' | b'\x08' | b'\x0c' | b'\n' | b'\r' | b'\t' => 1,
            0..=0x1f => 5,
            _ => 0,
        })
        .sum();
    s.len() + 2 + escapes
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tool(name: &str, schema: Value) -> Tool {
        Tool::builder(name.to_string())
            .description("d")
            .schema(schema)
            .build()
            .unwrap()
    }

    fn check(schema: Value, limits: &SchemaLimits) -> Result<(), SchemaLimit> {
        check_schemas([&tool("t", schema)], None, limits).map_err(|e| e.limit)
    }

    fn effective_bytes(schema: &Value) -> usize {
        let measured = Units::new(schema).measure(usize::MAX).unwrap();
        measured.effective(&measured.bytes)
    }

    fn effective_nodes(schema: &Value) -> usize {
        let measured = Units::new(schema).measure(usize::MAX).unwrap();
        measured.effective(&measured.nodes)
    }

    #[test]
    fn json_len_is_compact_length() {
        for value in [
            json!(null),
            json!(true),
            json!(-12.5),
            json!("a\"b\\c\nd"),
            json!([1, "x", [], {}]),
            json!({"k": {"l": [null, false]}, "m": "é"}),
            // Every control character, `\u{7f}` (written raw) and the
            // short escapes, in a value and in a key.
            Value::String((0u8..0x20).chain([0x7f]).map(char::from).collect()),
            json!({"\u{1}\t\u{8}\u{c}\r": "\u{1f}"}),
        ] {
            let compact = serde_json::to_string(&value).unwrap();
            assert_eq!(json_len(&value), compact.len(), "{compact}");
        }
    }

    /// A `$ref` counts its target's values too, per reference, so a
    /// fan-out of memberless leaves is measured; a cycle counts once.
    #[test]
    fn ref_counts_target_values_per_reference() {
        let leaf = json!({"type": "string"});
        // The root `{` and its `$ref` string; `E` two values more.
        let once = json!({"$ref": "#/$defs/E", "$defs": {"E": leaf}});
        assert_eq!(effective_nodes(&once), 2 + 2);
        // Root, array, two `{"$ref"}`s of two values; `E` twice.
        let twice = json!({
            "anyOf": [{"$ref": "#/$defs/E"}, {"$ref": "#/$defs/E"}],
            "$defs": {"E": leaf},
        });
        assert_eq!(effective_nodes(&twice), 6 + 2 * 2);
        // Data counts as values: the root, `enum` (array, members) and
        // `default` (object, array, number).
        let data = json!({"enum": ["a", "b"], "default": {"x": [1]}});
        assert_eq!(effective_nodes(&data), 1 + 3 + 3);
        let cyclic = json!({"$ref": "#", "type": "string"});
        assert_eq!(effective_nodes(&cyclic), 3);

        // A doubling chain with no members at all is past the limit
        // (saturated: 2^70 copies of its leaf).
        let n = 70;
        let mut defs: Map<String, Value> = (0..n)
            .map(|i| {
                let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
                (format!("D{i}"), json!({"anyOf": [next.clone(), next]}))
            })
            .collect();
        defs.insert(format!("D{n}"), json!({"type": "string"}));
        let schema = json!({"$ref": "#/$defs/D0", "$defs": defs});
        assert_eq!(effective_nodes(&schema), usize::MAX);
        assert_eq!(
            check(schema, &SchemaLimits::default()),
            Err(SchemaLimit::Nodes)
        );
    }

    #[test]
    fn ref_counts_at_target_size_per_reference() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"$ref": "#/$defs/E"},
                "b": {"$ref": "#/$defs/E"},
                "c": {"anyOf": [{"$ref": "#/$defs/E"}, {"const": 1}]},
            },
            "$defs": {"E": {"enum": ["xy", "z"]}},
        });
        // E is 4 + 3 bytes, named three times; `1` is one byte.
        assert_eq!(effective_bytes(&schema), 3 * 7 + 1);
    }

    #[test]
    fn ref_chains_multiply_and_cycles_count_once() {
        // D0 names D1 twice, D1 names D2 twice: D2's member counts 4×.
        let schema = json!({
            "$ref": "#/$defs/D0",
            "$defs": {
                "D0": {"anyOf": [{"$ref": "#/$defs/D1"}, {"$ref": "#/$defs/D1"}]},
                "D1": {"anyOf": [{"$ref": "#/$defs/D2"}, {"$ref": "#/$defs/D2"}]},
                "D2": {"const": "abc"},
            },
        });
        assert_eq!(effective_bytes(&schema), 4 * 5);
        let cyclic = json!({
            "$ref": "#/$defs/A",
            "$defs": {
                "A": {"anyOf": [{"$ref": "#/$defs/B"}, {"const": 1}]},
                "B": {"anyOf": [{"$ref": "#/$defs/A"}, {"const": 22}]},
            },
        });
        assert_eq!(effective_bytes(&cyclic), 1 + 2);
        let root = json!({"type": "object", "properties": {"x": {"$ref": "#"}}, "enum": [{}]});
        assert_eq!(effective_bytes(&root), 2);
    }

    #[test]
    fn data_keys_and_unreferenced_defs_count_nothing() {
        let schema = json!({
            "type": "object",
            "default": {"enum": ["no"]},
            "examples": [{"$ref": "#/$defs/E"}],
            "properties": {"a": {"type": "string"}},
            "$defs": {"E": {"enum": ["unused"]}},
        });
        assert_eq!(effective_bytes(&schema), 0);
    }

    /// An exponential `$ref` fan-out is measured in O(defs), not walked.
    #[test]
    fn exponential_fanout_saturates_quickly() {
        let n = 200;
        let mut defs: Map<String, Value> = (0..n)
            .map(|i| {
                let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
                (format!("D{i}"), json!({"anyOf": [next.clone(), next]}))
            })
            .collect();
        defs.insert(format!("D{n}"), json!({"const": "x"}));
        let schema = json!({"$ref": "#/$defs/D0", "$defs": defs});
        assert_eq!(effective_bytes(&schema), usize::MAX);
        assert_eq!(
            check(schema, &SchemaLimits::default()),
            Err(SchemaLimit::TotalMemberBytes)
        );
    }

    #[test]
    fn each_limit_refuses_with_its_name() {
        let limits = SchemaLimits::default()
            .with_max_tools(2)
            .with_max_params(2)
            .with_max_nodes(64)
            .with_max_defs(2)
            .with_max_member_bytes(8)
            .with_max_total_member_bytes(20);
        let ok =
            json!({"type": "object", "properties": {"a": {"enum": ["x"]}}});
        assert_eq!(check(ok.clone(), &limits), Ok(()));

        let tools: Vec<Tool> =
            (0..3).map(|i| tool(&format!("t{i}"), ok.clone())).collect();
        let err = check_schemas(&tools, None, &limits).unwrap_err();
        assert_eq!(
            (err.limit, err.actual, err.max),
            (SchemaLimit::Tools, 3, 2)
        );

        let params = json!({"type": "object", "properties": {"a": {}, "b": {}, "c": {}}});
        assert_eq!(check(params, &limits), Err(SchemaLimit::Params));

        let nodes: Vec<Value> = (0..64).map(|i| json!(i)).collect();
        assert_eq!(
            check(json!({"type": "object", "examples": nodes}), &limits),
            Err(SchemaLimit::Nodes)
        );
        let defs =
            json!({"$defs": {"A": {}, "B": {}}, "definitions": {"C": {}}});
        assert_eq!(check(defs, &limits), Err(SchemaLimit::Defs));

        let long = json!({"type": "object", "properties": {"a": {"const": "1234567"}}});
        assert_eq!(check(long, &limits), Err(SchemaLimit::MemberBytes));

        // Under the per-member limit, over the total once it is shared.
        let shared = json!({
            "type": "object",
            "properties": {"a": {"$ref": "#/$defs/E"}, "b": {"$ref": "#/$defs/E"}},
            "$defs": {"E": {"enum": ["abcd", "efgh"]}},
        });
        assert_eq!(check(shared, &limits), Err(SchemaLimit::TotalMemberBytes));

        // The output schema counts toward the request's totals.
        let output = json!({"enum": ["abcdef", "ghijkl", "mnopqr"]});
        let err =
            check_schemas([&tool("t", ok.clone())], Some(&output), &limits)
                .unwrap_err();
        assert_eq!(err.limit, SchemaLimit::TotalMemberBytes);
        assert_eq!(err.location, "request");
    }

    /// A prompt is measured on its custom tools and its structured
    /// output's schema; an effort-only `output_config` has none.
    #[test]
    fn prompt_counts_custom_tools_and_output_schema() {
        use misanthropic::prompt::output::OutputConfig;
        use misanthropic::tool::MethodDef;
        let limits = SchemaLimits::default().with_max_member_bytes(4);
        let small =
            json!({"type": "object", "properties": {"a": {"const": 1}}});
        let mut prompt = crate::Prompt {
            tools: Some(vec![MethodDef::Custom(tool("t", small))]),
            ..Default::default()
        };
        assert_eq!(check_prompt(&prompt, &limits), Ok(()));
        prompt.output_config = Some(OutputConfig::json_schema(json!({
            "enum": ["too long"],
        })));
        let err = check_prompt(&prompt, &limits).unwrap_err();
        assert_eq!(err.limit, SchemaLimit::MemberBytes);
        assert_eq!(err.location, "output_config.format.schema");
    }

    #[test]
    fn message_names_the_limit_and_where() {
        let limits = SchemaLimits::default().with_max_params(1);
        let schema =
            json!({"type": "object", "properties": {"a": {}, "b": {}}});
        let err = check_schemas([&tool("lookup", schema)], None, &limits)
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "tool `lookup` input_schema: more than 1 top-level properties \
             (at least 2); simplify the schema or split the request"
        );
    }
}
