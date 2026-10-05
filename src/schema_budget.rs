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
//! * **width**: how many ways a schema's grammar can go on at once —
//!   the matcher's stacks at one position, which it caps at 4096 by
//!   refusing the excess ([`SchemaLimits::max_width`] says how it is
//!   counted). A request inside the limit never reaches that cap.
//! * **depth**: how many levels of objects and arrays a value the
//!   schema's grammar writes nests, through `$ref`s — what the parsers
//!   must read back ([`SchemaLimits::max_depth`]).
//!
//! The defaults ([`SchemaLimits::default`]) leave real schemas far
//! inside every limit: see each field for what was measured.
//!
//! Every entry point that takes client schemas measures them first:
//! [`Session`](crate::Session)'s `complete*` calls and
//! [`count_tokens`](crate::Session::count_tokens) against
//! [`Session::with_schema_limits`](crate::Session::with_schema_limits),
//! and the public compilers —
//! [`dialect::grammar_source`](crate::dialect::grammar_source),
//! [`grammar_for_tool_choice`](crate::grammar_for_tool_choice) and the
//! `output_config` ones — against their options' `schema_limits`.

use std::collections::{BTreeMap, HashMap, HashSet};

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
/// beside a 249-member country `enum`): the synthetic tool is 3× inside
/// [`Self::max_width`] and 21× inside the rest, Agora's request at least
/// 21× inside every limit. Each field says what the largest measured.
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
    /// How wide one schema's grammar can branch: an upper bound on the
    /// matcher stacks alive at once inside it, which the matcher caps at
    /// 4096 by refusing the excess (`MAX_STACKS`). Counted per schema
    /// from the shape, bottom up:
    ///
    /// * an `enum` member or `const` is 1 (a literal, one stack until
    ///   its first distinguishing byte); `boolean` and `null` 4,
    ///   `number` 8, `string` 16, `integer` 24, anything untyped 16 —
    ///   each the most measured in any dialect, with margin;
    /// * an object its properties (required ones included) + 4, plus
    ///   its widest property; an array 4 plus its `items`;
    /// * an `anyOf` or `oneOf` the *sum* of its variants, plus one —
    ///   variants sharing a prefix (objects all opening with `{"a":`)
    ///   are all alive inside it, so nested ones multiply, and the one
    ///   is the alternation's own step in the schema check, so a chain
    ///   of one-variant alternations costs its length;
    /// * a `$ref` its target's width, each def counted once; a
    ///   reference back into its own cycle counts as untyped. (So
    ///   ambiguity *through* recursion — two interchangeable recursive
    ///   defs, doubling at every level of output — is not bounded by
    ///   this, only by the matcher's cap, which then refuses
    ///   alternatives and keeps the close.)
    ///
    /// Default 2048, half the matcher's cap. Every shape filled to it
    /// (an `enum`, optional and required properties, `anyOf`s of objects
    /// and of arrays alive through an integer, `anyOf`s nested to
    /// multiply) peaked at or under its count in every dialect — an
    /// `enum` exactly at it, the `anyOf`s at 60–95% — so a request
    /// inside the limit never reaches the cap, with room for the
    /// dialect's own framing. Agora's widest schema counts 70, the
    /// synthetic tool 613.
    pub max_width: usize,
    /// How many levels of objects and arrays a value the schema's
    /// grammar writes may nest — through `$ref`s, each reference at its
    /// target's depth (one back into its own cycle at none: recursion
    /// goes as deep as the model takes it, see below). Counted from the
    /// shape: an object or array is one level more than its deepest
    /// property or `items`, an `anyOf`/`oneOf` variant or `allOf` part
    /// the same level as its schema, an `enum` member or `const` its own
    /// depth as JSON. An untyped value counts nothing here: the grammar
    /// nests it at most 32 levels on its own.
    ///
    /// Default 64. With the 32 an untyped value may add and the two the
    /// dialects' call envelopes do, a value stays under the 127 levels
    /// serde_json reads (its recursion limit, which every dialect parser
    /// shares) — a deeper `required` chain, which a `$ref` per level
    /// spells in a shallow request, compiled to a grammar whose every
    /// value no parser could read: each draw failed, and the request
    /// ended in resampling and a 500. Raised past 93, a value can nest
    /// past what the parsers read. The deepest schema measured is 3
    /// levels (Agora's, and 40 real MCP and Claude Code tools).
    pub max_depth: usize,
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
            max_width: 2048,
            max_depth: 64,
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
            max_width: usize::MAX,
            max_depth: usize::MAX,
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

    /// Set [`Self::max_width`].
    pub fn with_max_width(mut self, n: usize) -> Self {
        self.max_width = n;
        self
    }

    /// Set [`Self::max_depth`].
    pub fn with_max_depth(mut self, n: usize) -> Self {
        self.max_depth = n;
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
    /// [`SchemaLimits::max_width`].
    Width,
    /// [`SchemaLimits::max_depth`].
    Depth,
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
            Self::Width => {
                "ways to continue at once (`enum` members, properties and \
                 `anyOf`/`oneOf` variants, nested variants multiplying)"
            }
            Self::Depth => {
                "levels of objects and arrays nested in a value, each \
                 `$ref` at its target's depth"
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
        let width = units.width(&measured);
        if width > limits.max_width {
            return Err(over(
                location.clone(),
                SchemaLimit::Width,
                width,
                limits.max_width,
            ));
        }
        let depth = units.depth(&measured);
        if depth > limits.max_depth {
            return Err(over(
                location.clone(),
                SchemaLimit::Depth,
                depth,
                limits.max_depth,
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

    /// The root's width ([`SchemaLimits::max_width`]): each unit's in
    /// `measured`'s order, so a `$ref` reads its target's, done — or,
    /// back into a cycle still open, counts as untyped.
    fn width(&self, measured: &Measured) -> usize {
        let mut widths: Vec<Option<usize>> = vec![None; self.units.len()];
        for &unit in &measured.order {
            widths[unit] = Some(self.unit_width(unit, &widths));
        }
        widths[0].expect("the order ends at the root")
    }

    /// One unit's subschemas in pre-order, so children follow their
    /// parent: `(schema, parent, how the parent combines it)`, the root's
    /// parent `usize::MAX`.
    fn subschemas(&self, unit: usize) -> Vec<(&'s Value, usize, Part)> {
        let root = self.units[unit].1;
        let mut nodes: Vec<(&Value, usize, Part)> =
            vec![(root, usize::MAX, Part::Other)];
        let mut i = 0;
        while i < nodes.len() {
            if let Value::Object(map) = nodes[i].0 {
                for (key, value) in map {
                    if DATA_KEYS.contains(&key.as_str())
                        || self.is_root_table(unit, nodes[i].0, key)
                    {
                        continue;
                    }
                    let part = match key.as_str() {
                        "anyOf" => Part::AnyOf,
                        "oneOf" => Part::OneOf,
                        "properties" => Part::Property,
                        "items" => Part::Items,
                        _ => Part::Other,
                    };
                    let children: Box<dyn Iterator<Item = &Value>> =
                        match (part, value) {
                            (Part::Property, Value::Object(props)) => {
                                Box::new(props.values())
                            }
                            (_, Value::Array(items)) => Box::new(items.iter()),
                            (_, value) => Box::new(std::iter::once(value)),
                        };
                    // Elsewhere only objects are schemas worth a look
                    // (`required`'s names, a `description`, are not).
                    nodes.extend(
                        children
                            .filter(|c| part != Part::Other || c.is_object())
                            .map(|c| (c, i, part)),
                    );
                }
            }
            i += 1;
        }
        nodes
    }

    /// One unit's width, given those of the units done before it: an
    /// iterative post-order over the unit's subschemas.
    fn unit_width(&self, unit: usize, widths: &[Option<usize>]) -> usize {
        let nodes = self.subschemas(unit);
        let mut parts = vec![Parts::default(); nodes.len()];
        let mut width = 0;
        for i in (0..nodes.len()).rev() {
            let (node, parent, part) = nodes[i];
            let target = self.target(node).map(|t| widths[t].unwrap_or(W_ANY));
            let w = own_width(node, &parts[i], target);
            match parts.get_mut(parent) {
                Some(p) => p.add(part, w),
                None => width = w,
            }
        }
        width
    }

    /// The root's depth ([`SchemaLimits::max_depth`]), as
    /// [`Self::width`] its width: each unit's in `measured`'s order, so a
    /// `$ref` reads its target's — or, back into a cycle still open,
    /// counts nothing.
    fn depth(&self, measured: &Measured) -> usize {
        let mut depths: Vec<Option<usize>> = vec![None; self.units.len()];
        for &unit in &measured.order {
            depths[unit] = Some(self.unit_depth(unit, &depths));
        }
        depths[0].expect("the order ends at the root")
    }

    /// One unit's depth, given those of the units done before it: an
    /// iterative post-order over the unit's subschemas.
    fn unit_depth(&self, unit: usize, depths: &[Option<usize>]) -> usize {
        let nodes = self.subschemas(unit);
        // Per subschema, the deepest of its children a level down (a
        // property, `items`) and of those at its own level.
        let mut below: Vec<Option<usize>> = vec![None; nodes.len()];
        let mut level = vec![0usize; nodes.len()];
        let mut depth = 0;
        for i in (0..nodes.len()).rev() {
            let (node, parent, part) = nodes[i];
            let target = self.target(node).map(|t| depths[t].unwrap_or(0));
            let d = own_depth(node, below[i], level[i], target);
            if parent == usize::MAX {
                depth = d;
            } else if matches!(part, Part::Property | Part::Items) {
                below[parent] = Some(below[parent].unwrap_or(0).max(d));
            } else {
                level[parent] = level[parent].max(d);
            }
        }
        depth
    }
}

/// `schema`'s width ([`SchemaLimits::max_width`]), for the tests that
/// fill shapes to the limit.
#[cfg(test)]
pub(crate) fn width(schema: &Value) -> usize {
    let units = Units::new(schema);
    units.width(&units.measure(usize::MAX).expect("no member limit"))
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

/// Width of an untyped (any JSON value) or unknown schema: the most
/// stacks the generic value grammar holds at once, measured 12, plus
/// margin. See [`SchemaLimits::max_width`] for the rest.
const W_ANY: usize = 16;
/// `string`: 4 in JSON, 11 as a raw tagged (Qwen) value.
const W_STRING: usize = 16;
/// `integer`: its 18 optional digits are a stack each (19 measured).
const W_INTEGER: usize = 24;
/// `number` (5 measured).
const W_NUMBER: usize = 8;
/// `boolean` or `null` (2 measured, 3 in Harmony).
const W_LITERAL: usize = 4;
/// An object's or array's own continuations: a separator, the close,
/// whitespace (3 measured).
const W_CONTAINER: usize = 4;

/// How a subschema's width combines into its parent's.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Part {
    /// An `anyOf` variant: summed.
    AnyOf,
    /// A `oneOf` variant: summed.
    OneOf,
    /// A property's schema: the widest counts.
    Property,
    /// `items` (or a tuple's): the widest counts.
    Items,
    /// Anything else (`additionalProperties`, `allOf`, …): the widest
    /// counts.
    Other,
}

/// A schema's subschemas' widths, combined per [`Part`].
#[derive(Clone, Copy, Default)]
struct Parts {
    any_of: Option<usize>,
    one_of: Option<usize>,
    property: usize,
    items: Option<usize>,
    other: usize,
}

impl Parts {
    fn add(&mut self, part: Part, w: usize) {
        let sum = |acc: &mut Option<usize>| {
            *acc = Some(acc.unwrap_or(0).saturating_add(w));
        };
        match part {
            Part::AnyOf => sum(&mut self.any_of),
            Part::OneOf => sum(&mut self.one_of),
            Part::Property => self.property = self.property.max(w),
            Part::Items => self.items = Some(self.items.unwrap_or(0).max(w)),
            Part::Other => self.other = self.other.max(w),
        }
    }
}

/// A schema's width from its own keywords and its subschemas' `parts`
/// (`target`: its `$ref`'s): the widest reading of it, so a schema
/// that is several things at once (an `anyOf` beside `properties`)
/// counts as the widest of them.
fn own_width(node: &Value, parts: &Parts, target: Option<usize>) -> usize {
    let Value::Object(map) = node else {
        return W_ANY;
    };
    // Each name once, here as in the compiler and the check: a name a
    // `type` or `required` array repeats is no wider.
    let types: Vec<&str> = match map.get("type") {
        Some(Value::String(t)) => vec![t.as_str()],
        Some(Value::Array(ts)) => {
            let mut seen = HashSet::new();
            ts.iter()
                .filter_map(Value::as_str)
                .filter(|t| seen.insert(*t))
                .collect()
        }
        _ => Vec::new(),
    };
    let props = map.get("properties").and_then(Value::as_object);
    let required = map.get("required").and_then(Value::as_array);
    let object = (props.is_some() || types.contains(&"object")).then(|| {
        let declared = props.map_or(0, Map::len);
        let undeclared: HashSet<&str> = required
            .into_iter()
            .flatten()
            .filter_map(Value::as_str)
            .filter(|n| !props.is_some_and(|p| p.contains_key(*n)))
            .collect();
        match declared + undeclared.len() {
            0 => W_ANY,
            slots => slots
                .saturating_add(W_CONTAINER)
                .saturating_add(parts.property),
        }
    });
    let array =
        (map.contains_key("items") || types.contains(&"array")).then(|| {
            match parts.items {
                Some(items) => items.saturating_add(W_CONTAINER),
                None => W_ANY,
            }
        });
    let scalars = types
        .iter()
        .map(|t| match *t {
            "object" | "array" => 0,
            "string" => W_STRING,
            "integer" => W_INTEGER,
            "number" => W_NUMBER,
            "boolean" | "null" => W_LITERAL,
            _ => W_ANY,
        })
        .fold(0usize, usize::saturating_add);
    // One more than the variants: the alternation's own step in the
    // schema check, so a chain of them is no free depth there.
    let alternatives = |key: &str, sum: Option<usize>| {
        map.get(key)
            .and_then(Value::as_array)
            .map(|_| sum.map_or(W_ANY, |sum| sum.saturating_add(1)))
    };
    let readings = [
        target,
        alternatives("anyOf", parts.any_of),
        alternatives("oneOf", parts.one_of),
        map.get("enum")
            .and_then(Value::as_array)
            .map(|members| members.len().max(1)),
        map.contains_key("const").then_some(1),
        object,
        array,
        (scalars > 0).then_some(scalars),
    ];
    let typed = readings.iter().flatten().copied().max();
    // Untyped: any value, unless something above says otherwise.
    typed.unwrap_or(W_ANY).max(parts.other)
}

/// A schema's depth ([`SchemaLimits::max_depth`]) from its children's:
/// the deepest a level `below` it (`None`: no such child), the deepest
/// at its own `level`, and its `$ref`'s `target`'s.
fn own_depth(
    node: &Value,
    below: Option<usize>,
    level: usize,
    target: Option<usize>,
) -> usize {
    let Value::Object(map) = node else {
        return 0;
    };
    let typed_container = match map.get("type") {
        Some(Value::String(t)) => t == "object" || t == "array",
        Some(Value::Array(ts)) => {
            ts.iter().any(|t| t == "object" || t == "array")
        }
        _ => false,
    };
    let container = below.is_some()
        || typed_container
        || map.contains_key("properties")
        || map.contains_key("items");
    let literals = map
        .get("enum")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .chain(map.get("const"))
        .map(json_depth)
        .max()
        .unwrap_or(0);
    let own = match container {
        true => below.unwrap_or(0).saturating_add(1),
        false => 0,
    };
    [target.unwrap_or(0), level, literals, own]
        .into_iter()
        .max()
        .unwrap_or(0)
}

/// Levels of arrays and objects in `value` (a scalar is 0).
fn json_depth(value: &Value) -> usize {
    let mut deepest = 0;
    let mut stack = vec![(value, 0usize)];
    while let Some((v, depth)) = stack.pop() {
        let children: Box<dyn Iterator<Item = &Value>> = match v {
            Value::Array(items) => Box::new(items.iter()),
            Value::Object(map) => Box::new(map.values()),
            _ => continue,
        };
        deepest = deepest.max(depth + 1);
        stack.extend(children.map(|c| (c, depth + 1)));
    }
    deepest
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

    /// The width of each shape, as [`SchemaLimits::max_width`] counts it.
    #[test]
    fn width_counts_alternatives() {
        let members: Vec<String> = (0..100).map(|i| format!("m{i}")).collect();
        assert_eq!(width(&json!({"enum": members})), 100);
        assert_eq!(width(&json!({"const": "x"})), 1);
        assert_eq!(width(&json!({})), W_ANY);
        assert_eq!(width(&json!(true)), W_ANY);
        assert_eq!(width(&json!({"type": "integer"})), W_INTEGER);
        assert_eq!(
            width(&json!({"type": ["string", "null"]})),
            W_STRING + W_LITERAL
        );
        // Properties (required ones without a schema too) + 4, plus
        // the widest property.
        let object = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"enum": [1, 2]}},
            "required": ["a", "c"],
        });
        assert_eq!(width(&object), 3 + W_CONTAINER + W_INTEGER);
        assert_eq!(width(&json!({"type": "object"})), W_ANY);
        let array = json!({"type": "array", "items": {"enum": [1, 2, 3]}});
        assert_eq!(width(&array), 3 + W_CONTAINER);
        // Variants add up, so nested ones multiply; the alternation
        // itself is one more.
        let inner = json!({"anyOf": [array, {"const": 0}]});
        assert_eq!(width(&inner), 3 + W_CONTAINER + 1 + 1);
        let outer = json!({"oneOf": [inner, inner, inner]});
        assert_eq!(width(&outer), 3 * (3 + W_CONTAINER + 2) + 1);
        // So a chain of one-variant alternations is as long as it is.
        let chain =
            (0..10).fold(json!({"const": 0}), |s, _| json!({"anyOf": [s]}));
        assert_eq!(width(&chain), 1 + 10);
        assert_eq!(width(&json!({"anyOf": []})), W_ANY);
        // A schema that is several things counts as the widest.
        let both = json!({"anyOf": [{"const": 1}], "enum": [1, 2, 3]});
        assert_eq!(width(&both), 3);
        // Data never counts; neither do unreferenced defs.
        let data = json!({
            "type": "string",
            "default": {"anyOf": [{}, {}, {}]},
            "$defs": {"Wide": {"enum": members}},
        });
        assert_eq!(width(&data), W_STRING);
    }

    /// A `$ref` is its target's width, each def counted once; one back
    /// into its cycle counts as untyped, so a tree is finite.
    #[test]
    fn width_follows_refs_and_cuts_cycles() {
        let members: Vec<usize> = (0..50).collect();
        let shared = json!({
            "type": "object",
            "properties": {
                "a": {"$ref": "#/$defs/E"},
                "b": {"anyOf": [{"$ref": "#/$defs/E"}, {"$ref": "#/$defs/E"}]},
            },
            "$defs": {"E": {"enum": members}},
        });
        assert_eq!(width(&shared), 2 + W_CONTAINER + 2 * 50 + 1);
        let tree = json!({
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
            }},
        });
        assert_eq!(width(&tree), 2 + W_CONTAINER + (W_ANY + W_CONTAINER));
        let root =
            json!({"type": "object", "properties": {"x": {"$ref": "#"}}});
        assert_eq!(width(&root), 1 + W_CONTAINER + W_ANY);
        // An exponential fan-out saturates in O(defs).
        let n = 200;
        let mut defs: Map<String, Value> = (0..n)
            .map(|i| {
                let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
                (format!("D{i}"), json!({"anyOf": [next.clone(), next]}))
            })
            .collect();
        defs.insert(format!("D{n}"), json!({"type": "string"}));
        let schema = json!({"$ref": "#/$defs/D0", "$defs": defs});
        assert_eq!(width(&schema), usize::MAX);
    }

    fn depth(schema: &Value) -> usize {
        let units = Units::new(schema);
        units.depth(&units.measure(usize::MAX).expect("no member limit"))
    }

    /// The depth of each shape, as [`SchemaLimits::max_depth`] counts it.
    #[test]
    fn depth_counts_nested_containers() {
        assert_eq!(depth(&json!({})), 0);
        assert_eq!(depth(&json!(true)), 0);
        assert_eq!(depth(&json!({"type": "integer"})), 0);
        // An object or array is a level, with or without its insides.
        assert_eq!(depth(&json!({"type": "object"})), 1);
        assert_eq!(depth(&json!({"type": ["array", "null"]})), 1);
        let object = json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {
                "type": "array",
                "items": {"type": "object", "properties": {}},
            }},
        });
        assert_eq!(depth(&object), 3);
        // Variants are at their schema's level.
        let any = json!({"anyOf": [object, {"type": "string"}]});
        assert_eq!(depth(&any), 3);
        // Literals are as deep as their JSON.
        assert_eq!(depth(&json!({"enum": [1, [[2]], {"k": [3]}]})), 2);
        assert_eq!(depth(&json!({"const": [[[0]]]})), 3);
        // Data never counts.
        assert_eq!(depth(&json!({"default": [[[[0]]]]})), 0);
    }

    /// A `$ref` is its target's depth; one back into its cycle counts
    /// nothing, so a tree is finite.
    #[test]
    fn depth_follows_refs_and_cuts_cycles() {
        let tree = json!({
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {
                "type": "object",
                "properties": {
                    "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}},
                },
            }},
        });
        assert_eq!(depth(&tree), 2);
        let chain = required_chain(10);
        assert_eq!(depth(&chain), 11);
    }

    /// `{"root": D0}`, each `D<i>` an object requiring its one property,
    /// a `$ref` to the next, `D<n>` an integer: a value `n + 1` levels
    /// deep, spelled in a request a few levels deep.
    fn required_chain(n: usize) -> Value {
        let mut defs: Map<String, Value> = (0..n)
            .map(|i| {
                let next = format!("#/$defs/D{}", i + 1);
                (
                    format!("D{i}"),
                    json!({
                        "type": "object",
                        "properties": {"a": {"$ref": next}},
                        "required": ["a"],
                    }),
                )
            })
            .collect();
        defs.insert(format!("D{n}"), json!({"type": "integer"}));
        json!({
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/D0"}},
            "required": ["root"],
            "$defs": defs,
        })
    }

    /// A `required` chain deeper than any parser reads back — 127 levels
    /// past serde_json's limit — is a 400, not a grammar whose every
    /// value fails to parse. Inside the default it passes.
    #[test]
    fn depth_past_the_limit_is_refused() {
        let limits = SchemaLimits::default();
        assert_eq!(check(required_chain(63), &limits), Ok(()));
        for n in [64, 126, 127, 300] {
            let err =
                check_schemas([&tool("t", required_chain(n))], None, &limits)
                    .unwrap_err();
            assert_eq!(
                (err.limit, err.actual, err.max),
                (SchemaLimit::Depth, n + 1, 64),
            );
        }
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
            .with_max_total_member_bytes(20)
            .with_max_width(30)
            .with_max_depth(3);
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

        // Width is per schema: an `enum` of 31.
        let wide: Vec<u8> = (0..31).collect();
        let limits = limits.with_max_total_member_bytes(usize::MAX);
        let output = json!({"enum": wide});
        let err = check_schemas([&tool("t", ok)], Some(&output), &limits)
            .unwrap_err();
        assert_eq!(err.limit, SchemaLimit::Width);
        assert_eq!(err.location, "output_config.format.schema");

        let deep = json!({"type": "array", "items": {"type": "array", "items": {
            "type": "array", "items": {"type": "array"},
        }}});
        assert_eq!(check(deep, &limits), Err(SchemaLimit::Depth));
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
