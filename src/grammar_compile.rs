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
//!   fields in `properties` iteration order (declaration order under
//!   `preserve_order`), required-ness by membership in `required:`.
//!   Optionals sit *in place*, wrapped in `( ... )?`, so they may be
//!   omitted but must match the declared type when present. The
//!   all-optional case (no `required`) reaches all 2^N inclusion
//!   patterns with a grammar linear in N (`optional_subsets`).
//!
//!   Anthropic's structured outputs order the same way: optionals stay
//!   in place, in `properties` order. Its docs say "required properties
//!   appear first, followed by optional properties", but the wire
//!   disagrees — probed live with `zulu` required, `alpha` optional,
//!   `mike` required, every sample on four models emitted
//!   `zulu → alpha → mike`, never the hoisted order (misanthropic
//!   `9be105f`). An interleaved schema renders the same on both
//!   engines.
//! * `type: array` with `items` → array of the item schema.
//!   `minItems >= 1` additionally forces non-emptiness — matching
//!   what Anthropic's structured outputs enforce (its sanitizer
//!   passes `minItems: 0 | 1` through). Counts beyond non-emptiness
//!   are ignored like other value-bound keywords (see below).
//! * `type: string | integer | number | boolean | null` → the
//!   corresponding JSON grammar rule.
//! * `enum` (any JSON value) → alternation of literals. An empty
//!   `enum` admits nothing and is a [`SchemaError`].
//! * `const: <value>` → exactly the JSON-encoded literal.
//! * `anyOf` → alternation of sub-schemas.
//! * `$ref: "#/$defs/<Name>"` → a reference to the definition's own
//!   named rule, compiled once from the root schema's `$defs` table,
//!   so a recursive type (a tree) is a recursive grammar. A reference
//!   that would be left recursion compiles to `value` (see `Defs`).
//!
//! Anything else (e.g. `allOf`, regex `pattern`, numeric ranges)
//! falls through to the permissive `value` rule, which accepts any
//! JSON. Callers lose strictness in those spots but generation does
//! not fail.
//!
//! A schema is a client's input, so its grammar is bounded: past 8
//! MiB (`MAX_GRAMMAR_BYTES`) compilation stops and the schema is
//! [`SchemaError::TooComplex`] — a 400, as Anthropic answers a schema
//! its own grammar compiler refuses.
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

use std::{collections::HashMap, fmt::Write};

use serde_json::{Map, Value};

use crate::json_canon::JsonSpacing;
pub(crate) use crate::sample::grammar::{
    rule_count, MAX_GRAMMAR_BYTES, MAX_GRAMMAR_RULES,
};

/// Why a JSON Schema has no grammar. Each is the request's fault — a
/// 400 `invalid_request_error`, the answer Anthropic gives a schema
/// its own compiler refuses — never a retryable server error.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SchemaError {
    /// The grammar would pass `limit` `what`s — 8 MiB of source, or
    /// 2^18 rules: compilation stops there rather than build it.
    #[error(
        "schema is too complex: its compiled grammar would exceed \
         {limit} {what}"
    )]
    TooComplex { what: &'static str, limit: usize },
    /// An `enum` with no members: no value satisfies it.
    #[error("schema has an empty `enum`, which no value can satisfy")]
    EmptyEnum,
}

static_assertions::assert_impl_all!(SchemaError: Send, Sync);

/// The rules a grammar's source has grown by, counted as it grows: a
/// scan of each new stretch ([`rule_count`]), so a long grammar is
/// counted once, not once per check.
#[derive(Debug, Default)]
pub(crate) struct RuleTally {
    /// Bytes of the source already counted.
    scanned: usize,
    rules: usize,
}

impl RuleTally {
    /// A tally that counts from byte `start` of the source on.
    pub(crate) fn from(start: usize) -> Self {
        Self {
            scanned: start,
            rules: 0,
        }
    }

    /// Count `src` up to its end, which must not split a rule (a tool's
    /// rules, written whole); [`SchemaError::TooComplex`] past
    /// [`MAX_GRAMMAR_RULES`].
    pub(crate) fn update(&mut self, src: &str) -> Result<(), SchemaError> {
        self.count(src, src.len())
    }

    /// Count `src` up to its last newline — a rule's end, where a
    /// grammar still being written may be cut.
    fn lines(&mut self, src: &str) -> Result<(), SchemaError> {
        match src[self.scanned..].rfind('\n') {
            Some(end) => self.count(src, self.scanned + end + 1),
            None => Ok(()),
        }
    }

    fn count(&mut self, src: &str, end: usize) -> Result<(), SchemaError> {
        self.rules += rule_count(&src[self.scanned..end]);
        self.scanned = end;
        match self.rules > MAX_GRAMMAR_RULES {
            true => Err(SchemaError::TooComplex {
                what: "rules",
                limit: MAX_GRAMMAR_RULES,
            }),
            false => Ok(()),
        }
    }
}

/// Emit GBNF rules that constrain a JSON value to `schema`.
///
/// The top-level rule will be named `rule_name`; anonymous helpers
/// get unique child names derived from it. If `schema` carries a
/// `$defs` map at its root, each `#/$defs/<Name>` a `$ref` reaches
/// compiles to one named rule (see `Defs`), so a recursive type
/// is a recursive grammar.
///
/// Fails, writing a partial grammar, when the schema has no grammar
/// ([`SchemaError`]); `out` past [`MAX_GRAMMAR_BYTES`] fails it too.
///
/// Exposed as `#[doc(hidden)] pub` (re-exported at the crate root)
/// so the in-tree fuzzer can compile schemas directly without going
/// through the tool-choice wrapper rules. Not part of the stable
/// surface — callers outside the fuzzer should use
/// [`grammar_for_tool_choice`](crate::grammar_for_tool_choice) or
/// [`output_config::grammar_for_output_config`](crate::output_config::grammar_for_output_config).
#[doc(hidden)]
pub fn schema_to_gbnf(
    schema: &Value,
    rule_name: &str,
    out: &mut String,
) -> Result<(), SchemaError> {
    let defs = schema.get("$defs").and_then(Value::as_object);
    let mut compiler = Compiler::new(defs, rule_name, None);
    compiler.add(schema, rule_name, out);
    compiler.finish(out)
}

/// A `$defs` table as the compiler, the checker and the tagged-value
/// classifier resolve `$ref`s against it: only the `#/$defs/<Name>`
/// shape schemars emits. Any other `$ref` is no reference at all —
/// the schema's other keywords apply, as if it were absent.
///
/// A reference is *left* when it is met before any byte of the value
/// it describes: at a schema's top, or down its `anyOf` branches, but
/// not in an object's properties or an array's items, which sit
/// behind a `{` or a `[`. A cycle of left references (`A = anyOf[$ref
/// A, …]`, an alias loop `A → B → A`) is left recursion as a grammar
/// and an endless loop as a check, so a reference that closes one
/// ([`Defs::resolve`]) reads as unconstrained: the compiler
/// writes `value` and the checker passes, in step and both finite.
/// Recursion behind a `{` or `[` — a tree's `children` — is ordinary
/// and stays exact.
pub(crate) struct Defs<'s> {
    /// `(name, schema)` in table order; a def's id is its index.
    entries: Vec<(&'s str, &'s Value)>,
    /// Name → id.
    ids: HashMap<&'s str, usize>,
    /// Each def's strongly connected component in the graph of left
    /// references.
    component: Vec<usize>,
}

impl<'s> Defs<'s> {
    pub(crate) fn new(table: Option<&'s Map<String, Value>>) -> Self {
        let entries: Vec<(&str, &Value)> = table
            .into_iter()
            .flatten()
            .map(|(name, schema)| (name.as_str(), schema))
            .collect();
        let ids = entries
            .iter()
            .enumerate()
            .map(|(id, (name, _))| (*name, id))
            .collect();
        let mut defs = Self {
            entries,
            ids,
            component: Vec::new(),
        };
        let edges: Vec<Vec<usize>> = defs
            .entries
            .iter()
            .map(|(_, schema)| {
                let mut targets = Vec::new();
                defs.left_refs(schema, &mut targets);
                targets
            })
            .collect();
        defs.component = components(&edges);
        defs
    }

    /// The id of the def `schema`'s `$ref` names, if it names one.
    pub(crate) fn target(&self, schema: &Value) -> Option<usize> {
        self.ids.get(ref_name(schema)?).copied()
    }

    /// Def `id`'s schema.
    pub(crate) fn schema(&self, id: usize) -> &'s Value {
        self.entries[id].1
    }

    /// Def `id`'s name.
    fn name(&self, id: usize) -> &'s str {
        self.entries[id].0
    }

    /// The number of defs.
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether a left reference to def `to`, met at a left position of
    /// def `from`'s body, closes a cycle. `from` is `None` at the
    /// root's top and behind any `{` or `[`, where no cycle can close.
    fn closes_cycle(&self, from: Option<usize>, to: usize) -> bool {
        from.is_some_and(|from| self.component[from] == self.component[to])
    }

    /// The def a left reference to `id` from `from` (as in
    /// [`Self::closes_cycle`]) lands on, through every alias — a def
    /// that is itself a bare `$ref` — in a loop rather than nested,
    /// however long the chain. `None` when the chain closes a cycle:
    /// the reference is unconstrained.
    pub(crate) fn resolve(
        &self,
        mut from: Option<usize>,
        mut id: usize,
    ) -> Option<usize> {
        loop {
            if self.closes_cycle(from, id) {
                return None;
            }
            match self.target(self.schema(id)) {
                Some(next) => (from, id) = (Some(id), next),
                None => return Some(id),
            }
        }
    }

    /// The defs `schema` references from its left positions. Walks the
    /// same precedence the compiler does: a resolvable `$ref` ends the
    /// schema, else an `anyOf` branches; nothing else is left.
    fn left_refs(&self, schema: &Value, targets: &mut Vec<usize>) {
        if let Some(id) = self.target(schema) {
            targets.push(id);
        } else if let Some(variants) =
            schema.get("anyOf").and_then(Value::as_array)
        {
            variants.iter().for_each(|v| self.left_refs(v, targets));
        }
    }
}

/// The def name `schema`'s `$ref` spells, in the one shape [`Defs`]
/// resolves (`#/$defs/<Name>`).
fn ref_name(schema: &Value) -> Option<&str> {
    schema.get("$ref")?.as_str()?.strip_prefix("#/$defs/")
}

/// The `(name, schema)` of the def in `table` that `schema`'s `$ref`
/// names, if it names one — [`Defs::target`] without building a
/// [`Defs`], for a walk that only follows references.
pub(crate) fn def_target<'s>(
    table: Option<&'s Map<String, Value>>,
    schema: &Value,
) -> Option<(&'s str, &'s Value)> {
    let (name, def) = table?.get_key_value(ref_name(schema)?)?;
    Some((name.as_str(), def))
}

/// Each node's strongly connected component (Tarjan), iteratively: a
/// client's `$defs` can chain thousands deep, and a recursive walk
/// would put that depth on the stack.
fn components(edges: &[Vec<usize>]) -> Vec<usize> {
    const UNSEEN: usize = usize::MAX;
    let n = edges.len();
    let mut index = vec![UNSEEN; n];
    let mut low = vec![0; n];
    let mut on_stack = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();
    let mut component = vec![UNSEEN; n];
    let (mut next_index, mut next_component) = (0, 0);
    for root in 0..n {
        if index[root] != UNSEEN {
            continue;
        }
        // `(node, next edge to follow)`: the recursion's frames.
        let mut frames: Vec<(usize, usize)> = vec![(root, 0)];
        index[root] = next_index;
        low[root] = next_index;
        next_index += 1;
        stack.push(root);
        on_stack[root] = true;
        while let Some(&(v, edge)) = frames.last() {
            if let Some(&w) = edges[v].get(edge) {
                let top = frames.len() - 1;
                frames[top].1 += 1;
                if index[w] == UNSEEN {
                    index[w] = next_index;
                    low[w] = next_index;
                    next_index += 1;
                    stack.push(w);
                    on_stack[w] = true;
                    frames.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
                continue;
            }
            frames.pop();
            if let Some(&(parent, _)) = frames.last() {
                low[parent] = low[parent].min(low[v]);
            }
            if low[v] == index[v] {
                while let Some(w) = stack.pop() {
                    on_stack[w] = false;
                    component[w] = next_component;
                    if w == v {
                        break;
                    }
                }
                next_component += 1;
            }
        }
    }
    component
}

/// One schema's compilation into rules, JSON or dict-encoded.
///
/// A def is compiled once, to its own named rule, which every `$ref`
/// to it names — so recursion lives in the grammar, where GBNF
/// handles it, instead of in an inlining that never ends (a tree
/// schema overflowed the stack and aborted the server). The named
/// rules come off a worklist rather than out of the reference that
/// first meets them: a chain of defs each naming the next would
/// otherwise nest the compiler as deep as the chain is long.
///
/// One compiler can write several schemas that share a `$defs` table
/// ([`Self::add`] each, then [`Self::finish`]): a tagged dialect's
/// parameters, which all resolve against their tool's defs, write
/// each def once for the tool rather than once per parameter.
///
/// Every rule is written into the caller's `out`, the whole grammar
/// so far, and the compiler stops once that passes
/// [`MAX_GRAMMAR_BYTES`] ([`SchemaError::TooComplex`]): a schema can't
/// make it build a grammar much larger than the limit first.
pub(crate) struct Compiler<'a> {
    /// Uniquifies child rule names.
    counter: usize,
    defs: Defs<'a>,
    /// The rule each def compiles to, once a reference names it.
    def_rules: Vec<Option<String>>,
    /// Defs named but not yet written: `(id, rule)`.
    pending: Vec<(usize, String)>,
    /// What the def rules' names start with: the root rule's name, so
    /// two schemas' defs in one grammar (two tools' `Node`s) never
    /// collide.
    prefix: &'a str,
    /// The dict encoding's string quote (Gemma 4), or `None` for JSON.
    quote: Option<&'a str>,
    /// The first reason this schema has no grammar. Once set, nothing
    /// more is written.
    error: Option<SchemaError>,
    /// The rules this compiler has written ([`MAX_GRAMMAR_RULES`]),
    /// from its first write on.
    tally: Option<RuleTally>,
}

impl<'a> Compiler<'a> {
    /// A compiler for schemas whose `$ref`s resolve against `defs`,
    /// naming its def rules `<prefix>__def…` and writing the dict
    /// encoding when `quote` is set.
    pub(crate) fn new(
        defs: Option<&'a Map<String, Value>>,
        prefix: &'a str,
        quote: Option<&'a str>,
    ) -> Self {
        let defs = Defs::new(defs);
        Self {
            counter: 0,
            def_rules: vec![None; defs.len()],
            defs,
            pending: Vec::new(),
            prefix,
            quote,
            error: None,
            tally: None,
        }
    }

    /// Write `schema` as `rule_name`. The defs it reaches wait for
    /// [`Self::finish`].
    pub(crate) fn add(
        &mut self,
        schema: &Value,
        rule_name: &str,
        out: &mut String,
    ) {
        self.rule(schema, rule_name, None, out);
    }

    /// Write every def the added schemas reach; the first reason one
    /// of them has no grammar, if any.
    pub(crate) fn finish(
        mut self,
        out: &mut String,
    ) -> Result<(), SchemaError> {
        while let Some((id, name)) = self.pending.pop() {
            self.rule(self.defs.schema(id), &name, Some(id), out);
        }
        self.halted(out);
        self.error.map_or(Ok(()), Err)
    }

    /// Whether to write nothing more: a schema already failed, the
    /// grammar so far is past [`MAX_GRAMMAR_BYTES`], or what this
    /// compiler wrote would build more than [`MAX_GRAMMAR_RULES`] — each
    /// fails it. Counting rules here, not leaving them to
    /// [`Grammar::parse`](crate::Grammar::parse), makes a grammar too
    /// big either way the same [`SchemaError::TooComplex`].
    pub(crate) fn halted(&mut self, out: &str) -> bool {
        if self.error.is_none() && out.len() > MAX_GRAMMAR_BYTES {
            self.error = Some(SchemaError::TooComplex {
                what: "bytes",
                limit: MAX_GRAMMAR_BYTES,
            });
        }
        if self.error.is_none() {
            let tally = self.tally.get_or_insert(RuleTally::from(out.len()));
            if let Err(e) = tally.lines(out) {
                self.error = Some(e);
            }
        }
        self.error.is_some()
    }

    /// Record why the schema has no grammar (the first reason wins).
    fn fail(&mut self, error: SchemaError) {
        self.error.get_or_insert(error);
    }

    /// The permissive fallback: any value, in this encoding.
    fn any(&self) -> &'static str {
        match self.quote {
            None => "value",
            Some(_) => "dvalue",
        }
    }

    /// Write `name ::= …` for `schema`. `left_of` is the def whose
    /// body this position is at the left of (see [`Defs`]).
    fn rule(
        &mut self,
        schema: &Value,
        name: &str,
        left_of: Option<usize>,
        out: &mut String,
    ) {
        if self.halted(out) {
            return;
        }
        if let Some(id) = self.defs.target(schema) {
            let target = self.def_rule(id, left_of);
            let _ = writeln!(out, "{name} ::= {target}");
            return;
        }

        // `anyOf`: alternation over sub-schemas, each still at the left.
        if let Some(variants) = schema.get("anyOf").and_then(Value::as_array) {
            let mut alts: Vec<String> = Vec::with_capacity(variants.len());
            for sub in variants {
                if self.halted(out) {
                    return;
                }
                self.counter += 1;
                let sub_name = format!("{name}__any_{c}", c = self.counter);
                self.rule(sub, &sub_name, left_of, out);
                alts.push(sub_name);
            }
            // Empty anyOf: accept nothing meaningful — fall back to
            // permissive value to avoid an unrepresentable grammar.
            let alts = match alts.is_empty() {
                true => self.any().to_string(),
                false => alts.join(" | "),
            };
            let _ = writeln!(out, "{name} ::= {alts}");
            return;
        }

        match self.quote {
            None => self.json_rule(schema, name, out),
            Some(quote) => self.dict_rule(schema, name, quote, out),
        }
    }

    /// The rule a `$ref` to def `id` compiles to: the def's own (an
    /// alias's target's), written once from the worklist — or, when
    /// the reference closes a left cycle, the permissive fallback.
    fn def_rule(&mut self, id: usize, left_of: Option<usize>) -> String {
        let Some(id) = self.defs.resolve(left_of, id) else {
            return self.any().to_string();
        };
        if let Some(rule) = &self.def_rules[id] {
            return rule.clone();
        }
        // GBNF names are `[A-Za-z0-9_-]`; the id keeps a lossy
        // spelling unique.
        let spelled: String = self
            .defs
            .name(id)
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        let rule = format!("{}__def{id}_{spelled}", self.prefix);
        self.def_rules[id] = Some(rule.clone());
        self.pending.push((id, rule.clone()));
        rule
    }

    /// The JSON rule for a schema past `$ref` and `anyOf`.
    fn json_rule(&mut self, schema: &Value, rule_name: &str, out: &mut String) {
        // `enum` → alternation of JSON-encoded literals.
        if let Some(variants) = schema.get("enum").and_then(|v| v.as_array()) {
            if variants.is_empty() {
                return self.fail(SchemaError::EmptyEnum);
            }
            let mut alt = String::new();
            for (i, v) in variants.iter().enumerate() {
                if i > 0 {
                    alt.push_str(" | ");
                }
                // serde_json produces the JSON literal with proper
                // escapes, then we GBNF-escape that string so it embeds
                // cleanly in a GBNF `"..."` terminal.
                let json_lit =
                    serde_json::to_string(v).unwrap_or_else(|_| "null".into());
                let gbnf_lit = escape_for_gbnf_string(&json_lit);
                let _ = write!(alt, r#""{gbnf_lit}""#);
            }
            let _ = writeln!(out, "{rule_name} ::= {alt}");
            return;
        }

        // `const: <value>` → exactly the JSON-encoded literal. Schemars
        // emits this for unit-enum variants with per-variant
        // descriptions (inside an `anyOf`), which is the Confidence-enum
        // shape drama_llama's whodunit test depends on. Without this
        // branch, per-variant `{const: "Low", description: "..."}`
        // subschemas hit the `_ => value` fallthrough and every variant
        // compiles to "accept any JSON value" — the grammar provides no
        // constraint at all for the enum field.
        if let Some(v) = schema.get("const") {
            let json_lit =
                serde_json::to_string(v).unwrap_or_else(|_| "null".into());
            let gbnf_lit = escape_for_gbnf_string(&json_lit);
            let _ = writeln!(out, r#"{rule_name} ::= "{gbnf_lit}""#);
            return;
        }

        match effective_type(schema) {
            Some("object") => self.object_rule(schema, rule_name, out),
            Some("string") => {
                let _ = writeln!(out, "{rule_name} ::= string");
            }
            Some("integer") => {
                // JSON grammar's `number` also permits decimals; reject
                // those for integer fields by referencing `int` directly
                // (defined in JSON_GRAMMAR, no frac/exp trailer).
                let _ = writeln!(out, "{rule_name} ::= integer");
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
                let items_rule = self.items_rule(schema, rule_name, out);
                // `minItems >= 1` forces a non-empty array — exactly as
                // much as Anthropic's own structured outputs enforce (the
                // misanthropic sanitizer passes `minItems: 0 | 1` through
                // and strips larger values). Counts beyond non-emptiness
                // are deliberately NOT enforced: forcing N items
                // manufactures filler entries, the value-bound failure
                // mode documented in
                // `.claude/memory/schema_constraint_keywords_decision.md`.
                // `maxItems` remains unenforced (permissive) for the same
                // reason.
                if non_empty(schema) {
                    let _ = writeln!(
                        out,
                        r#"{rule_name} ::= "[" pad {items_rule} ( elem_sep {items_rule} )* pad "]""#
                    );
                } else {
                    let _ = writeln!(
                        out,
                        r#"{rule_name} ::= "[" pad ( {items_rule} ( elem_sep {items_rule} )* )? pad "]""#
                    );
                }
            }
            _ => {
                // Unknown / unsupported — accept any JSON value.
                let _ = writeln!(out, "{rule_name} ::= value");
            }
        }
    }

    /// The rule an array's elements match: its `items` schema, written
    /// behind the `[` (no longer at the left), or any value.
    fn items_rule(
        &mut self,
        schema: &Value,
        rule_name: &str,
        out: &mut String,
    ) -> String {
        match schema.get("items") {
            Some(items) => {
                self.counter += 1;
                let name = format!("{rule_name}__item_{c}", c = self.counter);
                self.rule(items, &name, None, out);
                name
            }
            None => self.any().to_string(),
        }
    }

    /// JSON object layout: see the comments inside.
    fn object_rule(
        &mut self,
        schema: &Value,
        rule_name: &str,
        out: &mut String,
    ) {
        let no_props = Map::new();
        let props = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .unwrap_or(&no_props);
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

        // Layout: slots in `properties` iteration order (declaration
        // order under `preserve_order`), required-ness by *membership* in
        // `required:` — never by the array's order. Optionals sit in
        // place — before the first required slot as `( member "," )?`
        // (comma trailing), after it as `( "," member )?` — so the
        // accepted order is exactly the re-render order (the Map's own),
        // and every subset containing the required keys parses with
        // correct commas. Each key appears exactly once in the grammar;
        // that fixed order is what closes the duplicate-optional hole —
        // any fixed order does, alphabetization was never the
        // load-bearing part.
        //
        // Required names absent from `properties` are rare but legal;
        // they get a permissive `value` slot up front (their position is
        // arbitrary — no schema entry defines one).
        let mut slots: Vec<(String, String, bool)> = Vec::new();
        for name in &required_vec {
            if !props.contains_key(name) {
                slots.push((name.clone(), "value".to_string(), true));
            }
        }
        for (name, prop_schema) in props.iter() {
            if self.halted(out) {
                return;
            }
            self.counter += 1;
            let child_rule = format!("{rule_name}__{c}", c = self.counter);
            self.rule(prop_schema, &child_rule, None, out);
            slots.push((name.clone(), child_rule, required_set.contains(name)));
        }

        let member = |name: &str, child: &str| {
            let lit =
                escape_for_gbnf_string(&serde_json::to_string(name).unwrap());
            format!(r#""{lit}" kv_sep {child}"#)
        };

        let first_required = slots.iter().position(|(_, _, req)| *req);
        match first_required {
            None => {
                // All-optional: every subset of the slots, in slot
                // order, the empty one included — linear in the slots
                // (see `optional_subsets`).
                let members: Vec<String> = slots
                    .iter()
                    .map(|(name, child, _)| member(name, child))
                    .collect();
                let subsets =
                    optional_subsets(rule_name, &members, "elem_sep", out);
                let _ = writeln!(
                    out,
                    r#"{rule_name} ::= "{{" pad {subsets}? pad "}}""#
                );
            }
            Some(r) => {
                let mut body = String::from("\"{\" pad");
                for (name, child, _) in &slots[..r] {
                    let _ = write!(
                        body,
                        r#" ( {} elem_sep )?"#,
                        member(name, child)
                    );
                }
                let (name, child, _) = &slots[r];
                let _ = write!(body, " {}", member(name, child));
                for (name, child, req) in &slots[r + 1..] {
                    if *req {
                        let _ = write!(
                            body,
                            r#" elem_sep {}"#,
                            member(name, child)
                        );
                    } else {
                        let _ = write!(
                            body,
                            r#" ( elem_sep {} )?"#,
                            member(name, child)
                        );
                    }
                }
                body.push_str(" pad \"}\"");
                let _ = writeln!(out, "{rule_name} ::= {body}");
            }
        }
    }

    /// The dict-encoded rule for a schema past `$ref` and `anyOf`.
    fn dict_rule(
        &mut self,
        schema: &Value,
        rule_name: &str,
        quote: &str,
        out: &mut String,
    ) {
        if let Some(variants) = schema.get("enum").and_then(|v| v.as_array()) {
            if variants.is_empty() {
                return self.fail(SchemaError::EmptyEnum);
            }
            let mut alt = String::new();
            for (i, v) in variants.iter().enumerate() {
                if i > 0 {
                    alt.push_str(" | ");
                }
                let mut lit = String::new();
                dict_encode_value(v, quote, &mut lit);
                let _ = write!(alt, r#""{}""#, escape_for_gbnf_string(&lit));
            }
            let _ = writeln!(out, "{rule_name} ::= {alt}");
            return;
        }

        if let Some(v) = schema.get("const") {
            let mut lit = String::new();
            dict_encode_value(v, quote, &mut lit);
            let _ = writeln!(
                out,
                r#"{rule_name} ::= "{}""#,
                escape_for_gbnf_string(&lit)
            );
            return;
        }

        match effective_type(schema) {
            Some("object") => self.dict_object_rule(schema, rule_name, out),
            Some("string") => {
                let _ = writeln!(out, "{rule_name} ::= dstring");
            }
            Some("integer") => {
                let _ = writeln!(out, "{rule_name} ::= integer");
            }
            Some("number") => {
                let _ = writeln!(out, "{rule_name} ::= number");
            }
            Some("boolean") => {
                let _ = writeln!(out, r#"{rule_name} ::= "true" | "false""#);
            }
            Some("null") => {
                let _ = writeln!(out, "{rule_name} ::= dnull");
            }
            Some("array") => {
                let items_rule = self.items_rule(schema, rule_name, out);
                if non_empty(schema) {
                    let _ = writeln!(
                        out,
                        r#"{rule_name} ::= "[" {items_rule} ( "," {items_rule} )* "]""#
                    );
                } else {
                    let _ = writeln!(
                        out,
                        r#"{rule_name} ::= "[" ( {items_rule} ( "," {items_rule} )* )? "]""#
                    );
                }
            }
            _ => {
                let _ = writeln!(out, "{rule_name} ::= dvalue");
            }
        }
    }

    /// Dict object layout: keys explicitly sorted in place (the Gemma
    /// templates `dictsort` their re-renders, which alphabetizes
    /// regardless of Map iteration order), compact separators. Optionals *before* the first
    /// required slot render as `( "key:" child "," )?` (comma trailing);
    /// from the first required onward, each later slot carries its
    /// leading comma (`( "," "key:" child )?` when optional). All
    /// subsets containing every required key are reachable with correct
    /// commas, and — unlike a trailing-optionals layout — the accepted
    /// order is exactly the re-render order.
    fn dict_object_rule(
        &mut self,
        schema: &Value,
        rule_name: &str,
        out: &mut String,
    ) {
        let no_props = Map::new();
        let props = schema
            .get("properties")
            .and_then(|v| v.as_object())
            .unwrap_or(&no_props);
        let required: std::collections::HashSet<String> = schema
            .get("required")
            .and_then(|v| v.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        if props.is_empty() {
            let _ = writeln!(out, "{rule_name} ::= dobject");
            return;
        }

        // Explicit sort: `dictsort` alphabetizes no matter what order the
        // Map yields, so the grammar must too.
        let mut entries: Vec<(&String, &Value)> = props.iter().collect();
        entries.sort_unstable_by_key(|(k, _)| *k);
        let mut slots: Vec<(String, String, bool)> = Vec::new();
        for (key, prop_schema) in entries {
            if self.halted(out) {
                return;
            }
            self.counter += 1;
            let child = format!("{rule_name}__{c}", c = self.counter);
            self.rule(prop_schema, &child, None, out);
            slots.push((key.clone(), child, required.contains(key)));
        }

        let kv = |key: &str, child: &str| {
            format!(r#""{}:" {child}"#, escape_for_gbnf_string(key))
        };

        let first_required = slots.iter().position(|(_, _, req)| *req);
        let mut body = String::new();
        match first_required {
            Some(r) => {
                for (key, child, _) in &slots[..r] {
                    let _ = write!(body, r#"( {} "," )? "#, kv(key, child));
                }
                let (key, child, _) = &slots[r];
                body.push_str(&kv(key, child));
                for (key, child, req) in &slots[r + 1..] {
                    if *req {
                        let _ = write!(body, r#" "," {}"#, kv(key, child));
                    } else {
                        let _ = write!(body, r#" ( "," {} )?"#, kv(key, child));
                    }
                }
                let _ = writeln!(out, r#"{rule_name} ::= "{{" {body} "}}""#);
            }
            None => {
                // All optional: every subset in sorted order, the empty
                // dict included — linear, as in `object_rule`.
                let members: Vec<String> = slots
                    .iter()
                    .map(|(key, child, _)| kv(key, child))
                    .collect();
                let subsets =
                    optional_subsets(rule_name, &members, r#"",""#, out);
                let _ =
                    writeln!(out, r#"{rule_name} ::= "{{" {subsets}? "}}""#);
            }
        }
    }
}

/// Write the rules for a non-empty run of optional `members` (GBNF
/// sequences, in their fixed order) joined by `sep`, and return the
/// rule that matches any non-empty subset of them in order:
///
/// ```text
/// pick_k ::= member_k rest_{k+1} | pick_{k+1}    (pick_{n-1} ::= member_{n-1})
/// rest_k ::= ( sep pick_k )?
/// ```
///
/// `pick_k` is "the first member present is one of `k..`", `rest_k`
/// "maybe a separator and another, past the last one present". Each
/// member is written once, so the grammar is linear in the members —
/// writing each subset's tail in full was quadratic: 4000 optional
/// properties compiled to half a gigabyte. The separator is matched
/// once, before the choice of the next member, so after a member the
/// matcher holds a handful of stacks rather than one per later member
/// (each with its own separator in flight). Stops early once `out` is
/// past [`MAX_GRAMMAR_BYTES`], leaving the compiler to fail the schema.
fn optional_subsets(
    rule_name: &str,
    members: &[String],
    sep: &str,
    out: &mut String,
) -> String {
    let n = members.len();
    for (k, member) in members.iter().enumerate() {
        if out.len() > MAX_GRAMMAR_BYTES {
            break;
        }
        match k + 1 < n {
            true => {
                let next = k + 1;
                let _ = writeln!(
                    out,
                    "{rule_name}__pick_{k} ::= {member} {rule_name}__rest_{next} \
                     | {rule_name}__pick_{next}"
                );
                let _ = writeln!(
                    out,
                    "{rule_name}__rest_{next} ::= ( {sep} {rule_name}__pick_{next} )?"
                );
            }
            false => {
                let _ = writeln!(out, "{rule_name}__pick_{k} ::= {member}");
            }
        }
    }
    format!("{rule_name}__pick_0")
}

/// Whether an array schema's `minItems` asks for at least one element.
fn non_empty(schema: &Value) -> bool {
    schema.get("minItems").and_then(Value::as_u64).unwrap_or(0) >= 1
}

/// The schema's effective type, seeing through nullability: a bare
/// `"type": "T"`, or a type array whose non-`"null"` entries collapse
/// to exactly one `T` — schemars 1.x renders `Option<T>` as
/// `"type": ["T", "null"]`, and the council's optional-only tool
/// parameter compiled to a dead-end without this (the tagged-dialect
/// emitter fell through to a JSON `value` rule inside an XML
/// parameter; the model's prose was fully masked and the
/// grammar-exempt EOG won argmax, truncating the call mid-parameter).
///
/// Collapsing drops the `null` alternative deliberately: optionality
/// is expressed at the *property* level (omit the key / the
/// `arg_rule?` wrapper), and a model that wants `null` omits instead.
/// The tightened grammar still satisfies the original schema, so the
/// fuzzer's Class-2 property (grammar output validates against the
/// schema) holds by construction. Genuine multi-type unions
/// (`["string", "integer"]`) stay `None` → permissive fallthrough.
pub(crate) fn effective_type(schema: &Value) -> Option<&str> {
    match schema.get("type")? {
        Value::String(s) => Some(s.as_str()),
        Value::Array(arr) => {
            if !arr.iter().all(|v| v.is_string()) {
                return None;
            }
            let mut non_null = arr
                .iter()
                .filter_map(|v| v.as_str())
                .filter(|s| *s != "null");
            let first = non_null.next()?;
            non_null.next().is_none().then_some(first)
        }
        _ => None,
    }
}

// ===========================================================================
// Dict value encoding (Family::TagWithDict — Gemma 4)
// ===========================================================================
//
// JSON-shaped values with two twists the template trains the model
// on: dict keys are *bare* (`city:` not `"city":`) and strings are
// delimited by a dedicated quote marker (`<|"|>`) instead of `"` —
// the marker is a single special token, so string content needs no
// in-band escaping. Rendering is compact (no whitespace): that is
// what the template's `format_argument` re-renders, and round-trip
// byte-stability pins emission to re-render.
//
// Value-type canonical bytes were probed against our minijinja setup
// (pycompat) rendering the Gemma 4 template:
//   * null → `none` (minijinja lowercases; Python jinja says `None`,
//     upstream llama.cpp parses `null`). We *render* `none` and
//     *accept* all three in the grammar — if the model picks another
//     spelling, Session's canonicalization layer repairs the bytes.
//   * floats → serde_json/ryu shortest form matches minijinja
//     (`1.5e10` ⇒ `15000000000.0`, `3.0` ⇒ `3.0`).

/// Append `value` in dict encoding. Objects render explicitly
/// key-sorted: the Gemma templates pipe arguments through
/// `| dictsort`, which alphabetizes regardless of the Map's own
/// iteration order, and re-render byte-stability pins us to it.
pub(crate) fn dict_encode_value(v: &Value, quote: &str, out: &mut String) {
    match v {
        Value::String(s) => {
            out.push_str(quote);
            out.push_str(s);
            out.push_str(quote);
        }
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Null => out.push_str("none"),
        Value::Number(n) => {
            let _ = write!(out, "{n}");
        }
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_unstable_by_key(|(k, _)| *k);
            out.push('{');
            for (i, (k, val)) in entries.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(k);
                out.push(':');
                dict_encode_value(val, quote, out);
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, val) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                dict_encode_value(val, quote, out);
            }
            out.push(']');
        }
    }
}

/// Append the generic (schema-free) dict value rules: `dvalue`,
/// `dobject`, `darray`, `dstring`, `dnull`. References `number` from
/// [`JSON_GRAMMAR`], which callers append separately. Emit at most
/// once per grammar.
///
/// Like [`JSON_GRAMMAR`]'s `value`, `dvalue` nests at most
/// [`UNTYPED_DEPTH`] levels: a set of rules per level (`dvalue_1` …),
/// the last level's values scalars only.
pub(crate) fn emit_dict_value_rules(quote: &str, out: &mut String) {
    // Level `k`'s rule names end in this; the top level's in nothing.
    let level = |k: usize| match k {
        0 => String::new(),
        k => format!("_{k}"),
    };
    for k in 0..UNTYPED_DEPTH {
        let (this, next) = (level(k), level(k + 1));
        let _ = writeln!(
            out,
            r#"dvalue{this} ::= dstring | dobject{this} | darray{this} | number | "true" | "false" | dnull"#
        );
        let _ = writeln!(
            out,
            r#"dobject{this} ::= "{{" ( dmember{this} ( "," dmember{this} )* )? "}}""#
        );
        let _ = writeln!(out, r#"dmember{this} ::= dkey ":" dvalue{next}"#);
        let _ = writeln!(
            out,
            r#"darray{this} ::= "[" ( dvalue{next} ( "," dvalue{next} )* )? "]""#
        );
    }
    let _ = writeln!(
        out,
        r#"dvalue{} ::= dstring | number | "true" | "false" | dnull"#,
        level(UNTYPED_DEPTH)
    );
    let _ = writeln!(out, r#"dnull ::= "null" | "none" | "None""#);
    // Bare keys: anything but the key/dict terminators (upstream
    // parity: `chars("[^:}]", 1, -1)`).
    let _ = writeln!(out, r#"dkey ::= [^:}}]+"#);
    let quote_lit = escape_for_gbnf_string(quote);
    // The until-rule consumes string content AND the closing quote.
    let _ = writeln!(out, r#"dstring ::= "{quote_lit}" dstring__body"#);
    emit_until_rules("dstring__body", quote, out);
}

/// Dict-encoded counterpart of [`schema_to_gbnf`]: compile `schema`
/// into rules producing dict-encoded values. Objects lay out
/// explicitly key-sorted *in place* (required anchoring, optionals as
/// self-contained comma groups) to match the template's `dictsort`
/// re-render byte-for-byte.
pub(crate) fn schema_to_dict_gbnf(
    schema: &Value,
    rule_name: &str,
    quote: &str,
    out: &mut String,
) -> Result<(), SchemaError> {
    let defs = schema.get("$defs").and_then(Value::as_object);
    let mut compiler = Compiler::new(defs, rule_name, Some(quote));
    compiler.add(schema, rule_name, out);
    compiler.finish(out)
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
    emit_think_body_rules(out);
}

/// The body-and-char half of [`emit_thought_rules`], for roots that
/// anchor a *pre-opened* thought: the template already emitted the
/// opener, so the grammar must demand close-first (`think_body
/// "</think>" …`) and must NOT spell the opener literal — a root that
/// offers `"<think>"` at position 0 forces the model to *duplicate*
/// the template's tag, which is exactly the #107 duplicate-opener
/// containment class.
pub(crate) fn emit_think_body_rules(out: &mut String) {
    let _ = writeln!(out, r#"think_body ::= think_char*"#);
    let _ = writeln!(out, r#"think_char ::= [^<] | "<" [^/]"#);
}

/// Append GBNF rules matching raw content terminated by `delim`: the
/// emitted language is every string ending in exactly one occurrence
/// of `delim` — the delimiter appears nowhere except as the final
/// suffix. Content before the terminator is unrestricted.
///
/// This is the multi-char generalization of the `think_char` trick
/// and the GBNF encoding of llama.cpp's `until()` combinator
/// (`gbnf_excluding_grammar`, upstream PR #24839): the KMP prefix
/// automaton of `delim` emitted as right-linear rules, one rule per
/// automaton state. Each state gets an explicit branch per distinct
/// char of `delim` (advancing or falling back per KMP) and a
/// catch-all negated class returning to state 0; completing the
/// match terminates the rule. Exact — no lookahead required, so it
/// compiles to plain GBNF.
///
/// The root rule is `{rule_name}`; helpers are `{rule_name}__s{i}`.
/// Tagged dialects (Phase D) embed it as e.g.
/// `param_value ::= until_param_close` where the parsed value is
/// everything before the delimiter. The delimiter itself is part of
/// the matched text.
///
/// States are over Unicode scalar values, matching the grammar
/// engine's codepoint-based matcher (multi-byte UTF-8 delimiters
/// work). Practical dialect delimiters are ASCII.
///
/// # Panics
///
/// Panics if `delim` is empty — an "until nothing" rule is
/// meaningless and a caller bug.
///
/// Exposed as `#[doc(hidden)] pub` (re-exported at the crate root)
/// so the in-tree fuzzer can compile `until` grammars directly. Not
/// part of the stable surface — dialect callers should go through
/// the tagged-dialect emitter once it lands (Phase D), not this
/// function.
#[doc(hidden)]
pub fn emit_until_rules(rule_name: &str, delim: &str, out: &mut String) {
    let d: Vec<char> = delim.chars().collect();
    let n = d.len();
    assert!(n > 0, "emit_until_rules: empty delimiter");

    // Distinct delimiter chars in first-appearance order, for
    // deterministic output.
    let mut distinct: Vec<char> = Vec::new();
    for &c in &d {
        if !distinct.contains(&c) {
            distinct.push(c);
        }
    }

    // KMP transition: from state `i` (i chars of `delim` matched) on
    // char `c`, the next state is the longest prefix of `delim` that
    // is a suffix of `delim[..i] + c`. O(n²) per lookup; delimiters
    // are tiny.
    let delta = |i: usize, c: char| -> usize {
        let mut k = (i + 1).min(n);
        loop {
            if k == 0 {
                return 0;
            }
            if d[k - 1] == c && d[..k - 1] == d[i - (k - 1)..i] {
                return k;
            }
            k -= 1;
        }
    };

    // Catch-all class: any char not in the delimiter's alphabet
    // always resets to state 0 (delta is 0 for chars outside the
    // pattern), so one negated class covers all of them.
    let class: String =
        distinct.iter().map(|&c| escape_for_gbnf_class(c)).collect();

    let _ = writeln!(out, "{rule_name} ::= {rule_name}__s0");
    for i in 0..n {
        let mut alts: Vec<String> = Vec::with_capacity(distinct.len() + 1);
        for &c in &distinct {
            let lit = escape_for_gbnf_string(&c.to_string());
            let next = delta(i, c);
            if next == n {
                alts.push(format!(r#""{lit}""#));
            } else {
                alts.push(format!(r#""{lit}" {rule_name}__s{next}"#));
            }
        }
        alts.push(format!("[^{class}] {rule_name}__s0"));
        let _ = writeln!(
            out,
            "{rule_name}__s{i} ::= {alts}",
            alts = alts.join(" | ")
        );
    }
}

/// Escape a char for embedding inside a GBNF `[...]` character
/// class. Beyond the lexer's named escapes, `-` and `^` are emitted
/// as `\xNN` since they carry meaning inside a class.
fn escape_for_gbnf_class(c: char) -> String {
    match c {
        '\\' => r"\\".into(),
        ']' => r"\]".into(),
        '[' => r"\[".into(),
        '\n' => r"\n".into(),
        '\r' => r"\r".into(),
        '\t' => r"\t".into(),
        '-' | '^' => format!(r"\x{:02X}", c as u32),
        c if (c as u32) < 0x20 => format!(r"\x{:02X}", c as u32),
        c => c.to_string(),
    }
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

/// How many levels of objects and arrays an *untyped* value may nest:
/// one whose schema says nothing of its shape (`{}`, an unknown
/// `type`, `{"type": "object"}` without properties), which every
/// grammar writes with [`JSON_GRAMMAR`]'s `value` (or the dict
/// encoding's `dvalue`). Past it, an untyped value takes only scalars.
///
/// Unbounded, a degenerate bracket loop in an untyped parameter was
/// grammar-legal thousands of levels deep, past what any parser reads:
/// serde_json refuses nesting past [`MAX_NESTING`], and the recursive
/// readers overflowed the stack and aborted the process. With this bound
/// and [`SchemaLimits::max_depth`](crate::SchemaLimits::max_depth) on the
/// schema around it, no value the grammar of a schema inside the default
/// limits admits nests past [`MAX_NESTING`], wrapper levels included
/// (`depth_budget_fits_the_parsers`). Recursion through `$ref`s is the
/// exception: it nests as deep as the model takes it, and a value past
/// [`MAX_NESTING`] is refused by the parsers, never read.
pub(crate) const UNTYPED_DEPTH: usize = 32;

/// The most levels of objects and arrays a parsed value may nest:
/// serde_json's own limit (it refuses the 128th), which the dialect
/// parsers that read values themselves (the dict encoding, a call's
/// input read up to a cut) share, so that every reader refuses the
/// same values — and none recurses deeper than this on model output.
pub(crate) const MAX_NESTING: usize = 127;

/// [`JSON_GRAMMAR`]: the generic value rules unrolled one set per level
/// (`value_1` … `value_N`), each level's containers holding the next
/// level's values and the last level scalars only — GBNF has no depth
/// counter, so the bound ([`UNTYPED_DEPTH`]) is the rule names.
macro_rules! json_grammar {
    (last $last:literal; $($level:literal => $next:literal),* $(,)?) => {
        concat!(
            r#"
value ::= object | array | string | number | "true" | "false" | "null"
object ::= "{" pad ( member ( elem_sep member )* )? pad "}"
member ::= string kv_sep value_1
array ::= "[" pad ( value_1 ( elem_sep value_1 )* )? pad "]"
"#,
            $(
                "value_", $level, " ::= object_", $level, " | array_",
                $level, r#" | string | number | "true" | "false" | "null""#,
                "\n",
                "object_", $level, r#" ::= "{" pad ( member_"#, $level,
                " ( elem_sep member_", $level, r#" )* )? pad "}""#, "\n",
                "member_", $level, " ::= string kv_sep value_", $next, "\n",
                "array_", $level, r#" ::= "[" pad ( value_"#, $next,
                " ( elem_sep value_", $next, r#" )* )? pad "]""#, "\n",
            )*
            "value_", $last,
            r#" ::= string | number | "true" | "false" | "null""#,
            "\n",
            r#"string ::= "\"" char* "\""
char ::= unescaped | escape
unescaped ::= [^"\\\x00-\x1F]
escape ::= "\\" ( ["\\/bfnrt] | "u" non_surrogate_hex4 | "u" high_surrogate "\\u" low_surrogate )
non_surrogate_hex4 ::= [0-9a-cA-C] hex hex hex | [dD] [0-7] hex hex | [e-fE-F] hex hex hex
high_surrogate ::= [dD] [89aAbB] hex hex
low_surrogate ::= [dD] [c-fC-F] hex hex
hex ::= [0-9a-fA-F]
number ::= int frac? exp?
int ::= "-"? ( "0" | [1-9] [0-9]* )
integer ::= "0" | "-"? [1-9] [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]? [0-9]?
frac ::= "." [0-9]+
exp ::= [eE] [+\-]? [0-9] [0-9]?
ws ::= [ \t\n\r]?
kv_sep ::= ws ":" ws
elem_sep ::= ws "," ws
pad ::= ws
"#
        )
    };
}

/// Standard JSON grammar appended to every schema-derived GBNF.
///
/// Handles object / array / string / number / literal, with permissive
/// intra-structure whitespace. Not strict about number formatting edge
/// cases (e.g. `01` is rejected as JSON would); good enough for
/// downstream deserializers to validate. `value` nests at most 32
/// levels of objects and arrays (`UNTYPED_DEPTH`).
///
/// Exposed as `#[doc(hidden)] pub` for the fuzzer (paired with
/// [`schema_to_gbnf`]). Not part of the stable surface.
#[doc(hidden)]
pub const JSON_GRAMMAR: &str = json_grammar! {
    last 32;
    1 => 2,
    2 => 3,
    3 => 4,
    4 => 5,
    5 => 6,
    6 => 7,
    7 => 8,
    8 => 9,
    9 => 10,
    10 => 11,
    11 => 12,
    12 => 13,
    13 => 14,
    14 => 15,
    15 => 16,
    16 => 17,
    17 => 18,
    18 => 19,
    19 => 20,
    20 => 21,
    21 => 22,
    22 => 23,
    23 => 24,
    24 => 25,
    25 => 26,
    26 => 27,
    27 => 28,
    28 => 29,
    29 => 30,
    30 => 31,
    31 => 32,
};

/// The exact separators the JSON-envelope dialects put between a
/// call's top-level fields, shared by the grammar emitter and
/// `dialect::emit::render_reference` so the two cannot drift.
///
/// These are *not* pinnable to nothing the way the argument object's
/// interior is: the surrounding bytes are literal text in the model's
/// own chat template (cogito hardcodes `{"name": "` and
/// `", "arguments": `), so the canonical form is mixed — the template
/// dictates the envelope, our serializer dictates the interior. Change
/// one of these and the grammar stops admitting `render_reference`'s
/// output, which `canonical_call_grammar_admits_render_reference`
/// catches.
pub(crate) const KV_SEP: &str = ": ";
/// Sibling of [`KV_SEP`]; separates top-level fields.
pub(crate) const FIELD_SEP: &str = ", ";

/// The permissive separator productions inside [`JSON_GRAMMAR`], as
/// literals, so [`json_grammar_canonical`] can swap them. Kept honest
/// by `canonical_json_grammar_pins_separators`, which fails if any
/// drifts out of sync with the grammar text above.
///
/// Separators are *position-aware* — distinct named productions
/// rather than one generic `ws` — because a canonical spelling is not
/// uniform: `json.dumps` puts a space after `:` and `,` but none
/// inside braces. One `ws` rule cannot express that; three named
/// positions can (#88 phase 2).
const WS_PERMISSIVE: &str = r"ws ::= [ \t\n\r]?";
/// Between a key and its value.
const KV_SEP_PERMISSIVE: &str = r#"kv_sep ::= ws ":" ws"#;
/// Between object members and between array elements.
const ELEM_SEP_PERMISSIVE: &str = r#"elem_sep ::= ws "," ws"#;
/// Just inside `{`/`}` and `[`/`]`.
const PAD_PERMISSIVE: &str = r"pad ::= ws";
/// Framing whitespace, permissive, under a name the JSON rules never
/// reference — root rules use it for the layout *around* the JSON
/// (e.g. the `\n\n` a thinking model puts between `</think>` and its
/// call). Appended by both prelude builders below so roots written
/// against one compile against the other.
const FWS_PERMISSIVE: &str = r"fws ::= [ \t\n\r]?";

/// [`JSON_GRAMMAR`] with JSON-*internal* whitespace pinned to exactly
/// one spelling per [`JsonSpacing`], plus a separate `fws` for
/// framing.
///
/// Tool calls must re-render byte-identically to what the model
/// emitted or the prefix cache's auto-tip is discarded (#85):
/// permissive separators let the model choose `": "` where the
/// serializer re-renders `":"`, and nothing downstream can know which
/// it picked. Pinning makes a `serde_json::Value` have exactly one
/// legal byte spelling, so grammar and serializer become inverses —
/// [`crate::json_canon::to_string`] with the same [`JsonSpacing`] is
/// that inverse, and the pinned productions are built from the same
/// [`JsonSpacing::kv_sep`] / [`JsonSpacing::elem_sep`] bytes it
/// emits, so the two cannot drift.
///
/// Which spelling to pin is per-dialect data: the analyzer measures
/// how the *active* chat template spaces its re-render (stock
/// `tojson` templates are [`Compact`](JsonSpacing::Compact); owned
/// templates match the model's measured habit — cogito's is
/// [`Spaced`](JsonSpacing::Spaced), `tests/probe_unforced_habit.rs`).
///
/// Rules generated by [`schema_to_gbnf`] reference `kv_sep` /
/// `elem_sep` / `pad` **by name** and are therefore prelude-agnostic
/// — the same generated text is permissive under [`JSON_GRAMMAR`] and
/// canonical under this one. That is why pinning is a prelude swap
/// rather than a change to the emitter. (One narrow exception:
/// container-valued `enum:` / `const:` schema literals embed compact
/// bytes directly in the rule, so they only match their emission
/// under `Compact`. Schemars-derived tools only produce *scalar*
/// literals, which are spelling-invariant.)
///
/// **`fws` is the reason this isn't a one-line override.** The root
/// rules use whitespace for *framing* — the `\n\n` a model puts
/// between `</think>` and its call — which legitimately varies and is
/// not part of the byte-stability problem. Pinning that too would
/// mask tokens the model is trained to emit, for no cache benefit.
/// Root rules use `fws`; everything inside the JSON uses the pinned
/// productions.
///
/// Structured output keeps [`JSON_GRAMMAR`] for now — it has the same
/// latent divergence when a JSON response is replayed as history, but
/// that is a separate change with its own blast radius.
#[doc(hidden)]
pub fn json_grammar_canonical(spacing: JsonSpacing) -> String {
    let mut out = JSON_GRAMMAR
        .replace(WS_PERMISSIVE, r#"ws ::= """#)
        .replace(
            KV_SEP_PERMISSIVE,
            &format!(r#"kv_sep ::= "{}""#, spacing.kv_sep()),
        )
        .replace(
            ELEM_SEP_PERMISSIVE,
            &format!(r#"elem_sep ::= "{}""#, spacing.elem_sep()),
        )
        .replace(PAD_PERMISSIVE, r#"pad ::= """#);
    out.push_str(FWS_PERMISSIVE);
    out.push('\n');
    out
}

/// [`JSON_GRAMMAR`] as-is (every spelling admitted) plus the `fws`
/// framing rule — for grammars with **no canonical-bytes contract**.
///
/// The deprecated [`grammar_for_tool_choice`] path is the consumer:
/// nothing re-renders its emissions (the byte-stability invariant
/// belongs to `Session`'s dialect emitter, which never routes through
/// it), so pinning a spelling there buys no cache property — and
/// measurably hurts. The #85 pin made it force `{"location":"` where
/// Qwen3.6's habit is `{"location": "`, and the model, boxed out of
/// its trained bytes, flailed inside the string's *free* region
/// (`"}}<|im_end|>…"` — grammar-legal garbage). Deterministic repro:
/// `DRAMA_LLAMA_SEED=4` on
/// `tool_choice_forces_call_against_real_model`; caught by the first
/// full ignored-tier run after the pin landed. Constrain exactly what
/// the contract needs, nothing more.
///
/// [`grammar_for_tool_choice`]: crate::grammar_for_tool_choice
#[doc(hidden)]
pub fn json_grammar_lenient() -> String {
    let mut out = String::from(JSON_GRAMMAR);
    out.push_str(FWS_PERMISSIVE);
    out.push('\n');
    out
}
// `integer` (used by `type: integer` fields; `int` stays permissive
// because `number` composes it and must express `-0.5`) forbids `-0`
// and caps at 18 digits so every grammar-emitted integer fits `i64` —
// serde_json parses 19+-digit literals (and `-0`) as `f64`, which the
// schema validator then type-rejects. Both were fuzzer Class-3
// findings (2026-07-17). Style follows `exp` below: explicit
// optionals, no `{m,n}` (unsupported by the matcher).
//
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

    /// The canonical prelude must actually differ from the permissive
    /// one. `json_grammar_canonical` works by string replacement, so if
    /// any `*_PERMISSIVE` literal ever drifts out of sync with the
    /// grammar text the replace silently no-ops and every tool call
    /// goes back to being un-pinned — with no other symptom until a
    /// prefix cache quietly stops matching. Fail loudly here instead.
    #[test]
    fn canonical_json_grammar_pins_separators() {
        const PERMISSIVE: [&str; 4] = [
            WS_PERMISSIVE,
            KV_SEP_PERMISSIVE,
            ELEM_SEP_PERMISSIVE,
            PAD_PERMISSIVE,
        ];
        for production in PERMISSIVE {
            assert!(
                JSON_GRAMMAR.contains(production),
                "`{production}` no longer matches JSON_GRAMMAR; \
                 json_grammar_canonical is silently a no-op for it",
            );
        }
        for spacing in [JsonSpacing::Compact, JsonSpacing::Spaced] {
            let canonical = json_grammar_canonical(spacing);
            // Line-exact: `fws ::= ...` *contains* `ws ::= ...`, so a
            // substring check here silently passes for the wrong
            // reason.
            let lines: Vec<&str> = canonical
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty())
                .collect();
            assert!(lines.contains(&r#"ws ::= """#));
            assert!(lines.contains(&r#"pad ::= """#));
            let kv = format!(r#"kv_sep ::= "{}""#, spacing.kv_sep());
            let elem = format!(r#"elem_sep ::= "{}""#, spacing.elem_sep());
            assert!(lines.contains(&kv.as_str()), "{spacing:?}: {kv}");
            assert!(lines.contains(&elem.as_str()), "{spacing:?}: {elem}");
            for production in PERMISSIVE {
                assert!(
                    !lines.contains(&production),
                    "permissive `{production}` survived into the \
                     {spacing:?} canonical prelude",
                );
            }
            // Framing whitespace survives, under a name the JSON rules
            // never reference.
            assert!(lines.contains(&r"fws ::= [ \t\n\r]?"));
        }
    }

    /// The point of the canonical prelude: exactly one legal spelling
    /// per [`JsonSpacing`].
    ///
    /// `schema_to_gbnf`'s output is prelude-agnostic — it references
    /// `kv_sep` / `elem_sep` / `pad` by name — so the *same* generated
    /// rules accept every spelling under `JSON_GRAMMAR` and exactly
    /// one under each `json_grammar_canonical` profile. That property
    /// is what lets tool calls be pinned without touching structured
    /// output, and what lets the pinned spelling follow the model's
    /// measured habit (#88 phase 2).
    #[test]
    fn canonical_prelude_admits_exactly_one_spelling() {
        let schema = json!({
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["a", "b"],
        });
        let mut rules = String::from("root ::= args\n");
        schema_to_gbnf(&schema, "args", &mut rules).unwrap();

        let compact = r#"{"a":"x","b":[1,2]}"#;
        let spaced = r#"{"a": "x", "b": [1, 2]}"#;
        let mixed = r#"{"a": "x","b":[1, 2]}"#;
        let padded = r#"{ "a": "x", "b": [1, 2] }"#;

        let permissive = format!("{rules}{JSON_GRAMMAR}");
        for input in [compact, spaced, mixed, padded] {
            assert!(
                accepts(&permissive, input),
                "permissive prelude must keep accepting every spelling — \
                 structured output still depends on it: {input}",
            );
        }

        let canonical =
            format!("{rules}{}", json_grammar_canonical(JsonSpacing::Compact));
        assert!(
            accepts(&canonical, compact),
            "Compact prelude must accept what the compact serializer emits",
        );
        for input in [spaced, mixed, padded] {
            assert!(
                !accepts(&canonical, input),
                "Compact prelude must reject other spellings — this is \
                 the divergence that cost 4705 tokens/turn in #85: {input}",
            );
        }

        let canonical =
            format!("{rules}{}", json_grammar_canonical(JsonSpacing::Spaced));
        assert!(
            accepts(&canonical, spaced),
            "Spaced prelude must accept the model's measured habit \
             (`json.dumps` spacing — the cogito probe)",
        );
        for input in [compact, mixed, padded] {
            assert!(
                !accepts(&canonical, input),
                "Spaced prelude must reject other spellings: {input}",
            );
        }
    }

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

    /// Canary for `serde_json/preserve_order` (#60): the whole
    /// declaration-order chain — schemars properties, grammar
    /// emission, parse, template re-render — rides on the Map keeping
    /// insertion order. If someone drops the feature from Cargo.toml,
    /// this fails before the round-trip fixtures start flaking.
    #[test]
    fn serde_json_preserves_insertion_order() {
        assert_eq!(
            serde_json::to_string(&json!({"b": 1, "a": 2})).unwrap(),
            r#"{"b":1,"a":2}"#,
        );
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
        schema_to_gbnf(&schema, "obj", &mut rules).unwrap();
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
        schema_to_gbnf(&schema, "root_obj", &mut rules).unwrap();
        let src = wrap_with_root("root_obj", rules);
        assert!(accepts(&src, r#"{"inner":{"x":1}}"#));
        assert!(!accepts(&src, r#"{"inner":{}}"#));
    }

    /// A schemars-derived shape with a `$ref`-array and a string
    /// array (the whodunit CaseFile pattern) must accept a POPULATED
    /// instance — regression probe for suspects_considered=[] on
    /// Qwen3.6: is it the model's choice or a grammar hole?
    #[test]
    fn compiles_ref_array_accepts_populated() {
        let schema = json!({
            "type": "object",
            "properties": {
                "suspects": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Suspect"}
                },
                "evidence": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "culprit": {"type": "string"},
            },
            "required": ["suspects", "evidence", "culprit"],
            "$defs": {
                "Suspect": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "had_access": {"type": "boolean"},
                    },
                    "required": ["name", "had_access"],
                }
            }
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "case", &mut rules).unwrap();
        let src = wrap_with_root("case", rules);
        assert!(
            accepts(
                &src,
                r#"{"suspects":[{"name":"Crane","had_access":true},{"name":"Elsie","had_access":false}],"evidence":["poison","ledger"],"culprit":"Crane"}"#
            ),
            "grammar must accept populated $ref arrays:\n{src}"
        );
        assert!(accepts(
            &src,
            r#"{"suspects":[],"evidence":[],"culprit":"Crane"}"#
        ));
    }

    /// `minItems >= 1` enforces non-emptiness and nothing more —
    /// Anthropic-API parity (its sanitizer passes only `0 | 1`
    /// through). Larger counts stay validator territory per
    /// `.claude/memory/schema_constraint_keywords_decision.md`.
    #[test]
    fn min_items_enforces_non_empty_only() {
        let schema = json!({
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "arr1", &mut rules).unwrap();
        let src = wrap_with_root("arr1", rules);
        assert!(!accepts(&src, "[]"), "empty must be rejected");
        assert!(accepts(&src, r#"["a"]"#));
        assert!(accepts(&src, r#"["a", "b"]"#));

        // Counts beyond non-emptiness are NOT enforced: minItems 3
        // still admits a single element (forcing more manufactures
        // filler — the value-bound failure mode).
        let schema = json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "arr3", &mut rules).unwrap();
        let src = wrap_with_root("arr3", rules);
        assert!(!accepts(&src, "[]"));
        assert!(accepts(&src, "[1]"), "counts beyond 1 are permissive");

        // minItems 0 (and absent) keep the empty form.
        let schema = json!({
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 0,
        });
        let mut rules = String::new();
        schema_to_gbnf(&schema, "arr0", &mut rules).unwrap();
        let src = wrap_with_root("arr0", rules);
        assert!(accepts(&src, "[]"));
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
        schema_to_gbnf(&schema, "conf", &mut rules).unwrap();
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
        schema_to_gbnf(&schema, "conf", &mut rules).unwrap();
        let src = wrap_with_root("conf", rules);
        assert!(accepts(&src, r#""Low""#));
        assert!(accepts(&src, r#""Medium""#));
        assert!(accepts(&src, r#""High""#));
        assert!(!accepts(&src, r#""Definite""#));
        assert!(!accepts(&src, r#""low""#)); // case-sensitive
    }

    /// Exhaustive differential check of `emit_until_rules` against a
    /// naive matcher: over a 3-char alphabet, every string up to
    /// length 7, for delimiters exercising self-overlap (`aa`, `aba`)
    /// and the trivial single char. The grammar must accept exactly
    /// the strings whose only occurrence of the delimiter is the
    /// final suffix.
    #[test]
    fn until_rules_match_naive_exhaustively() {
        const ALPHABET: [char; 3] = ['a', 'b', 'c'];
        for delim in ["a", "ab", "aa", "aba", "abc"] {
            let mut rules = String::new();
            emit_until_rules("u", delim, &mut rules);
            let src = format!("root ::= u\n{rules}");
            let grammar = Arc::new(
                Grammar::parse(&src).expect("until grammar must parse"),
            );

            // Enumerate all strings of length 0..=7 by counting in
            // base 3.
            for len in 0..=7usize {
                for mut idx in 0..3usize.pow(len as u32) {
                    let mut s = String::with_capacity(len);
                    for _ in 0..len {
                        s.push(ALPHABET[idx % 3]);
                        idx /= 3;
                    }
                    let naive = s.ends_with(delim)
                        && s.find(delim) == Some(s.len() - delim.len());
                    let mut state = GrammarState::new(Arc::clone(&grammar));
                    let by_grammar = state.advance_bytes(s.as_bytes()).is_ok()
                        && state.is_complete();
                    assert_eq!(
                        by_grammar, naive,
                        "delim {delim:?}, input {s:?}: grammar said \
                         {by_grammar}, naive said {naive}\n{src}"
                    );
                }
            }
        }
    }

    /// The Phase D use case: raw parameter values terminated by the
    /// Qwen XML close tag, including partial-overlap content the
    /// naive `[^<]*`-style approximations get wrong.
    #[test]
    fn until_rules_handle_dialect_close_tags() {
        let mut rules = String::new();
        emit_until_rules("val", "</parameter>", &mut rules);
        let src = format!("root ::= val\n{rules}");

        // Empty content: just the delimiter.
        assert!(accepts(&src, "</parameter>"));
        // Plain content.
        assert!(accepts(&src, "42 rue de la Paix\n</parameter>"));
        // Content with partial-overlap teasers: `<`, `</`, `</param`.
        assert!(accepts(&src, "a < b and c </ d </param e</parameter>"));
        assert!(accepts(&src, "<</parameter>"));
        assert!(accepts(&src, "</</parameter>"));
        // Trailing whitespace inside the value survives (the
        // awkward-but-legal class from the plan amendments).
        assert!(accepts(&src, "value ends in newline\n\n</parameter>"));
        // A full delimiter mid-content must reject: the value ended
        // earlier, the rest is trailing garbage.
        assert!(!accepts(&src, "x</parameter>y</parameter>"));
        // No terminator at all: incomplete, not accepted.
        assert!(!accepts(&src, "dangling"));
        // Bare prefix of the delimiter at end: incomplete.
        assert!(!accepts(&src, "value</param"));
    }

    /// Multi-byte UTF-8 delimiter chars work (the automaton runs on
    /// codepoints, matching the engine's matcher).
    #[test]
    fn until_rules_unicode_delimiter() {
        let mut rules = String::new();
        emit_until_rules("u", "→end", &mut rules);
        let src = format!("root ::= u\n{rules}");
        assert!(accepts(&src, "before →end"));
        assert!(accepts(&src, "→ not yet →end"));
        assert!(!accepts(&src, "→end trailing"));
    }

    /// Delimiter chars that are metacharacters inside GBNF classes /
    /// literals must be escaped, not break the emitted grammar.
    #[test]
    fn until_rules_escapes_metacharacters() {
        for delim in ["]", "[x]", "a-b", "^", "\\", "\"", "\n\n"] {
            let mut rules = String::new();
            emit_until_rules("u", delim, &mut rules);
            let src = format!("root ::= u\n{rules}");
            let content = format!("some content{delim}");
            assert!(accepts(&src, &content), "delim {delim:?} failed:\n{src}");
            assert!(!accepts(&src, "no terminator"));
        }
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
        schema_to_gbnf(&schema, "obj", &mut rules).unwrap();
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
    /// The all-optional encoding `optional_subsets` replaced: `chain_k`
    /// writes member `k` and every later one in full — quadratic.
    fn quadratic_chains(
        rule_name: &str,
        members: &[String],
        sep: &str,
        out: &mut String,
    ) -> String {
        (0..members.len())
            .map(|k| {
                let mut tail = members[k].clone();
                for member in &members[k + 1..] {
                    let _ = write!(tail, " ( {sep} {member} )?");
                }
                let _ = writeln!(out, "{rule_name}__chain_{k} ::= {tail}");
                format!("{rule_name}__chain_{k}")
            })
            .collect::<Vec<_>>()
            .join(" | ")
    }

    /// The linear all-optional encoding admits exactly the quadratic
    /// one's language: random member sets (overlapping, duplicated, one
    /// a prefix of another), every input over their alphabet up to 7
    /// bytes, and every in-order subset.
    #[test]
    fn optional_subsets_match_quadratic_encoding() {
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let mut inputs: Vec<String> = vec![String::new()];
        let mut frontier = inputs.clone();
        for _ in 0..7 {
            frontier = frontier
                .iter()
                .flat_map(|s| ["a", "b", ","].map(|c| format!("{s}{c}")))
                .collect();
            inputs.extend(frontier.iter().cloned());
        }
        let grammar = |encode: &dyn Fn(&mut String) -> String| {
            let mut rules = String::new();
            let alts = encode(&mut rules);
            let src = format!("root ::= \"{{\" ( {alts} )? \"}}\"\n{rules}");
            GrammarState::new(Arc::new(Grammar::parse(&src).unwrap()))
        };
        let accepts = |root: &GrammarState, text: &str| {
            let mut state = root.clone();
            state
                .advance_bytes(format!("{{{text}}}").as_bytes())
                .is_ok()
                && state.is_complete()
        };
        for _ in 0..60 {
            let n = 1 + (next() % 5) as usize;
            let spelled: Vec<String> = (0..n)
                .map(|_| {
                    (0..1 + next() % 2)
                        .map(|_| if next() % 2 == 0 { 'a' } else { 'b' })
                        .collect()
                })
                .collect();
            let members: Vec<String> =
                spelled.iter().map(|m| format!("\"{m}\"")).collect();
            let sep = r#"",""#;
            let linear =
                grammar(&|out| optional_subsets("o", &members, sep, out));
            let quadratic =
                grammar(&|out| quadratic_chains("o", &members, sep, out));
            let subsets = (1..1u32 << n).map(|mask| {
                spelled
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask & (1 << i) != 0)
                    .map(|(_, m)| m.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            });
            for input in inputs.iter().cloned().chain(subsets) {
                assert_eq!(
                    accepts(&linear, &input),
                    accepts(&quadratic, &input),
                    "members {spelled:?}, input {input:?}"
                );
            }
        }
    }

    /// All-optional objects compile linearly in both encodings: 4000
    /// optional properties were ~530 MB of grammar.
    #[test]
    fn all_optional_object_is_linear() {
        let n = 4000;
        let props: Map<String, Value> =
            (0..n).map(|i| (format!("p{i}"), json!({}))).collect();
        let schema = json!({"type": "object", "properties": props});
        let mut rules = String::new();
        schema_to_gbnf(&schema, "obj", &mut rules).unwrap();
        assert!(rules.len() < 1 << 20, "{} bytes", rules.len());
        let src = wrap_with_root("obj", rules);
        assert!(accepts(&src, "{}"));
        assert!(accepts(&src, r#"{"p0":1,"p17":true,"p3999":null}"#));
        assert!(!accepts(&src, r#"{"p17":true,"p0":1}"#));

        let mut dict = String::new();
        schema_to_dict_gbnf(&schema, "obj", "<|\"|>", &mut dict).unwrap();
        assert!(dict.len() < 1 << 20, "{} bytes", dict.len());
    }

    /// The dict encoding's all-optional object: every subset in sorted
    /// order, nothing out of it.
    #[test]
    fn dict_all_optional_object_permits_every_sorted_subset() {
        let schema = json!({
            "type": "object",
            "properties": {
                "c": {"type": "integer"},
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            }
        });
        let mut rules = String::new();
        schema_to_dict_gbnf(&schema, "obj", "'", &mut rules).unwrap();
        emit_dict_value_rules("'", &mut rules);
        let src = wrap_with_root("obj", rules);
        for ok in ["{}", "{a:1}", "{c:3}", "{a:1,c:3}", "{a:1,b:2,c:3}"] {
            assert!(accepts(&src, ok), "{ok}");
        }
        for bad in ["{c:3,a:1}", "{a:1,}", "{,a:1}", "{a:1,a:1}", "{a:1b:2}"] {
            assert!(!accepts(&src, bad), "{bad}");
        }
    }

    /// A schema whose grammar would pass [`MAX_GRAMMAR_BYTES`] or
    /// [`MAX_GRAMMAR_RULES`] fails as [`SchemaError::TooComplex`] once the
    /// grammar gets there, without building the rest: a 400,000-way
    /// `anyOf` (17 MB of grammar), a 400,000-property object — both
    /// pass the rule limit first — and a 1,500,000-way `anyOf` of one
    /// short name, which passes the byte limit first.
    #[test]
    fn too_complex_schema_stops_at_the_limit() {
        let n = 400_000;
        let variants: Vec<Value> =
            (0..n).map(|_| json!({"$ref": "#/$defs/A"})).collect();
        let fanout =
            json!({"anyOf": variants, "$defs": {"A": {"type": "string"}}});
        let props: Map<String, Value> =
            (0..n).map(|i| (format!("p{i}"), json!({}))).collect();
        let wide = json!({"type": "object", "properties": props});
        for schema in [fanout, wide] {
            let mut out = String::new();
            assert_eq!(
                schema_to_gbnf(&schema, "s", &mut out),
                Err(SchemaError::TooComplex {
                    what: "rules",
                    limit: MAX_GRAMMAR_RULES
                })
            );
            // Stopped near the limit: what one rule line can add past it.
            assert!(rule_count(&out) < MAX_GRAMMAR_RULES + 64);
            assert!(out.len() < MAX_GRAMMAR_BYTES, "{}", out.len());
            let mut dict = String::new();
            assert!(schema_to_dict_gbnf(&schema, "s", "'", &mut dict).is_err());
        }
    }

    /// Long rules pass the byte limit before the rule limit: still
    /// [`SchemaError::TooComplex`], stopped there.
    #[test]
    fn too_many_bytes_stops_at_the_limit() {
        let long = "x".repeat(4096);
        let variants: Vec<Value> = (0..4096)
            .map(|i| json!({"const": format!("{long}{i}")}))
            .collect();
        let schema = json!({"anyOf": variants});
        let mut out = String::new();
        assert_eq!(
            schema_to_gbnf(&schema, "s", &mut out),
            Err(SchemaError::TooComplex {
                what: "bytes",
                limit: MAX_GRAMMAR_BYTES
            })
        );
        assert!(out.len() < MAX_GRAMMAR_BYTES + (1 << 16), "{}", out.len());
    }

    /// [`rule_count`] is the number of rules [`Grammar::parse`]
    /// builds — literals, groups, repetitions and comments included —
    /// so the compiler's rule limit is the parser's.
    #[test]
    fn rule_count_matches_the_parser() {
        let schemas = [
            json!({"type": "object", "properties": {
                "a": {"type": "string"},
                "b": {"type": "array", "items": {"enum": ["x\"y", 1, null]}},
                "c": {"anyOf": [{"type": "integer"}, {"const": "(?*+)"}]},
            }, "required": ["a"]}),
            json!({"$ref": "#/$defs/N", "$defs": {"N": {"type": "object",
                "properties": {"kids": {"type": "array",
                    "items": {"$ref": "#/$defs/N"}}}}}}),
        ];
        for schema in schemas {
            for dict in [false, true] {
                let mut src = String::from("# a comment: \"(\n");
                src.push_str("root ::= s [\\]\"(]?\n");
                match dict {
                    false => schema_to_gbnf(&schema, "s", &mut src),
                    true => schema_to_dict_gbnf(&schema, "s", "'", &mut src),
                }
                .unwrap();
                if dict {
                    emit_dict_value_rules("'", &mut src);
                }
                src.push_str(JSON_GRAMMAR);
                let grammar = crate::Grammar::parse(&src).unwrap();
                assert_eq!(rule_count(&src), grammar.rule_count(), "{src}");
            }
        }
    }

    /// `{"enum": []}` admits no value. It used to compile to an empty
    /// rule body, a GBNF syntax error; now it is a schema error, as deep
    /// in the schema as it sits.
    #[test]
    fn empty_enum_is_a_schema_error() {
        for schema in [
            json!({"enum": []}),
            json!({"type": "object", "properties": {"x": {"enum": []}}}),
            json!({"anyOf": [{"type": "string"}, {"enum": []}]}),
            json!({"$ref": "#/$defs/E", "$defs": {"E": {"enum": []}}}),
        ] {
            let mut out = String::new();
            assert_eq!(
                schema_to_gbnf(&schema, "s", &mut out),
                Err(SchemaError::EmptyEnum),
                "{schema}"
            );
            let mut dict = String::new();
            assert_eq!(
                schema_to_dict_gbnf(&schema, "s", "'", &mut dict),
                Err(SchemaError::EmptyEnum),
                "{schema}"
            );
        }
    }

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
        schema_to_gbnf(&schema, "obj", &mut rules).unwrap();
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
        schema_to_gbnf(&schema, "obj", &mut rules).unwrap();
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

    /// `Node`, as schemars emits a tree type: a `$ref` back to itself
    /// behind an array's `[`, reached from a root property.
    fn tree_schema() -> Value {
        json!({
            "type": "object",
            "properties": {"root": {"$ref": "#/$defs/Node"}},
            "required": ["root"],
            "$defs": {"Node": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "children": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Node"},
                    },
                },
            }},
        })
    }

    /// Compile `schema`, and judge `input` by its grammar and by the
    /// schema check, which must agree.
    fn judge(schema: &Value, input: &str) -> bool {
        let mut rules = String::new();
        schema_to_gbnf(schema, "s", &mut rules).unwrap();
        let grammar = accepts(&wrap_with_root("s", rules), input);
        let checked = crate::schema_check::check_text(schema, input).is_ok();
        assert_eq!(grammar, checked, "grammar and check disagree: {input}");
        grammar
    }

    /// Run `f` on a thread with only `kib` KiB of stack: what a
    /// recursion bug would overflow (an abort, not a panic — the
    /// whole server, in production).
    fn on_small_stack<T: Send + 'static>(
        kib: usize,
        f: impl FnOnce() -> T + Send + 'static,
    ) -> T {
        std::thread::Builder::new()
            .stack_size(kib * 1024)
            .spawn(f)
            .expect("spawn")
            .join()
            .expect("no panic")
    }

    /// A recursive `$ref` is one named rule the grammar recurses
    /// through, not an inlining without end (it overflowed the stack
    /// and aborted blallama).
    #[test]
    fn recursive_ref_compiles_to_one_named_rule() {
        let schema = tree_schema();
        let rules = on_small_stack(256, {
            let schema = schema.clone();
            move || {
                let mut rules = String::new();
                schema_to_gbnf(&schema, "s", &mut rules).unwrap();
                rules
            }
        });
        assert_eq!(rules.matches("s__def0_Node ::=").count(), 1, "{rules}");
        let three_levels = r#"{"root":{"name":"a","children":[{"name":"b","children":[{"name":"c","children":[]}]},{"name":"d"}]}}"#;
        assert!(judge(&schema, three_levels));
        assert!(judge(&schema, r#"{"root":{}}"#));
        for wrong in [
            r#"{"root":{"name":"a","children":[{"name":"b","children":[{"name":3}]}]}}"#,
            r#"{"root":{"children":[{"children":{}}]}}"#,
            r#"{"root":[]}"#,
        ] {
            assert!(!judge(&schema, wrong), "{wrong}");
        }
    }

    /// Mutual recursion (`A → B → A`) through properties.
    #[test]
    fn mutual_recursion_compiles() {
        let schema = json!({
            "$ref": "#/$defs/A",
            "$defs": {
                "A": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        "b": {"anyOf": [{"$ref": "#/$defs/B"}, {"type": "null"}]},
                    },
                    "required": ["label", "b"],
                },
                "B": {
                    "type": "object",
                    "properties": {"a": {"$ref": "#/$defs/A"}},
                    "required": ["a"],
                },
            },
        });
        assert!(judge(
            &schema,
            r#"{"label":"x","b":{"a":{"label":"y","b":{"a":{"label":"z","b":null}}}}}"#
        ));
        for wrong in [
            r#"{"label":"x","b":{"a":{"label":5,"b":null}}}"#,
            r#"{"label":"x","b":{"a":{"label":"y"}}}"#,
            r#"{"label":"x","b":{}}"#,
        ] {
            assert!(!judge(&schema, wrong), "{wrong}");
        }
    }

    /// An alias chain resolves to its end; a reference that loops back
    /// before any byte of the value — a self-alias, an alias cycle, a
    /// left-recursive `anyOf` — is unconstrained, in the grammar and
    /// the check alike, and compiles to no left recursion.
    #[test]
    fn ref_chains_and_left_cycles() {
        let schema = json!({
            "type": "object",
            "properties": {
                "chain": {"$ref": "#/$defs/A"},
                "selfie": {"$ref": "#/$defs/Me"},
                "loop": {"$ref": "#/$defs/P"},
                "either": {"$ref": "#/$defs/G"},
            },
            "$defs": {
                "A": {"$ref": "#/$defs/B"},
                "B": {"$ref": "#/$defs/C"},
                "C": {"type": "integer"},
                "Me": {"$ref": "#/$defs/Me"},
                "P": {"$ref": "#/$defs/Q"},
                "Q": {"$ref": "#/$defs/P"},
                "G": {"anyOf": [{"$ref": "#/$defs/G"}, {"type": "string"}]},
            },
        });
        assert!(judge(&schema, r#"{"chain":3}"#));
        assert!(!judge(&schema, r#"{"chain":"3"}"#));
        for anything in [r#""x""#, "[1,{}]", "null", r#"{"q":1}"#] {
            for key in ["selfie", "loop", "either"] {
                let input = format!(r#"{{"{key}":{anything}}}"#);
                assert!(judge(&schema, &input), "{input}");
            }
        }
        // The chain names `C`'s rule directly: no alias rule for `A`/`B`.
        let mut rules = String::new();
        schema_to_gbnf(&schema, "s", &mut rules).unwrap();
        assert!(!rules.contains("_A ::="), "{rules}");
        assert!(rules.contains("s__def2_C ::= integer"), "{rules}");
    }

    /// Thousands of defs, each naming the next — through a property,
    /// or as a bare alias — compile, match and check without nesting
    /// that deep; a diamond of `anyOf`s (`D_i = anyOf[D_{i+1},
    /// D_{i+1}]`) compiles and checks in linear time, not 2^n.
    #[test]
    fn long_and_wide_ref_chains_stay_flat() {
        fn chain(n: usize, link: fn(Value) -> Value) -> Value {
            let mut defs = serde_json::Map::new();
            for i in 0..n {
                let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
                defs.insert(format!("D{i}"), link(next));
            }
            defs.insert(format!("D{n}"), json!({"type": "integer"}));
            json!({"$ref": "#/$defs/D0", "$defs": defs})
        }
        const N: usize = 5000;
        let nested =
            |next| json!({"type": "object", "properties": {"n": next}});
        let diamond = |next: Value| json!({"anyOf": [next.clone(), next]});
        let alias = chain(N, |next| next);
        let (nested, long_diamond) = (chain(N, nested), chain(N, diamond));
        // Deep enough that 2^n is forever, shallow enough to be judged.
        let diamond = chain(64, diamond);
        on_small_stack(256, move || {
            for schema in [&nested, &alias, &long_diamond] {
                let mut rules = String::new();
                schema_to_gbnf(schema, "s", &mut rules).unwrap();
                assert!(rules.len() < 200 * N, "{} bytes", rules.len());
            }
            assert!(judge(&nested, r#"{"n":{"n":{}}}"#));
            assert!(!judge(&nested, r#"{"n":{"n":1}}"#));
            assert!(judge(&alias, "7"));
            assert!(!judge(&alias, r#""7""#));
            let check = crate::schema_check::check_text;
            assert!(check(&diamond, "7").is_ok());
            assert!(check(&diamond, r#""7""#).is_err());
            // Past the checker's depth cap: lenient, not a crash.
            assert!(check(&long_diamond, r#""7""#).is_ok());
        });
    }

    /// A schema as deep as a request can carry (serde_json refuses
    /// nesting past 128) compiles, matches and checks on the stack a
    /// tokio worker has.
    #[test]
    fn deepest_parseable_schema_fits_a_worker_stack() {
        // Two JSON levels per schema level: `properties` and the schema.
        let depth = 63;
        let schema_text = format!(
            "{}{{\"type\":\"integer\"}}{}",
            r#"{"type":"object","required":["a"],"properties":{"a":"#
                .repeat(depth),
            "}}".repeat(depth),
        );
        let schema: Value =
            serde_json::from_str(&schema_text).expect("within the limit");
        let value =
            format!("{}1{}", r#"{"a":"#.repeat(depth), "}".repeat(depth));
        let past_limit = format!("{}{}", "[".repeat(129), "]".repeat(129));
        assert!(serde_json::from_str::<Value>(&past_limit).is_err());
        on_small_stack(2048, move || {
            assert!(judge(&schema, &value));
            assert!(!judge(&schema, &value.replace('1', "true")));
        });
    }
}
