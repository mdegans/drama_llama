//! Generic envelope parser driven by [`CallSyntax`]: raw model text →
//! [`Block`]s, for any analyzed dialect.
//!
//! ## Re-parse-per-tick (deliberate)
//!
//! [`parse_text`] is a pure function over the *entire accumulated*
//! generation text — streaming callers re-invoke it per tick rather
//! than feeding deltas into a stateful machine. Total work is O(n²)
//! over a generation; outputs are a few KB of string scanning, so
//! this is microseconds, and llama.cpp ships exactly this design. Do
//! NOT "optimize" it back into an incremental state machine — that is
//! the partial-tag-holdback `BlockParser` this replaces, FIXMEs and
//! all. A full re-parse also yields a complete, consistent partial
//! AST at every tick, which is what streaming events (#26) need.
//!
//! ## Leniency
//!
//! [`Leniency::Streaming`] suppresses an incomplete trailing
//! structure (atomicity: no half-parsed calls surface) and reports
//! [`ParseStatus::NeedMoreInput`]. [`Leniency::Final`] converts the
//! incomplete tail into a [`Block::Text`] fallback — the historic
//! `BlockParser::finish` contract; Session decides whether that is a
//! grammar violation.
//!
//! ## Coercion & healing (llama.cpp mapper parity)
//!
//! Tagged raw values are schema-coerced: params typed `string` (or
//! unknown) stay raw strings; anything else is parsed as JSON after
//! normalizing pythonisms (`True`/`False`/`None`, single-quoted
//! strings) with bounded brace-healing, falling back to a JSON
//! string of the raw bytes when parsing still fails. Parse never
//! hard-errors on content — worst case a call degrades to text.

use std::borrow::Cow;

use serde_json::Value;

use crate::prompt::{Block, ToolUse};
use crate::Tool;

use super::{CallSyntax, Family, ReasoningMode};

/// Whether the parse saw a complete structure or ran out of input
/// mid-call / mid-thought.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ParseStatus {
    Complete,
    NeedMoreInput,
}

/// How to treat an incomplete trailing structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Leniency {
    /// Mid-stream: suppress the partial, report `NeedMoreInput`.
    Streaming,
    /// End of generation: partials degrade to [`Block::Text`].
    Final,
}

#[derive(Debug)]
pub struct Parsed {
    pub blocks: Vec<Block>,
    pub status: ParseStatus,
}

/// Parse `text` (the full accumulated generation) into blocks per
/// `syntax`.
///
/// `pre_opened_reasoning`: the rendered generation prompt ended with
/// the open reasoning tag, so `text` *begins inside* the reasoning
/// block (no open tag will appear) — everything up to
/// `reasoning.end` is thought. This is the unforced-path fix for
/// issue #27.
pub fn parse_text(
    syntax: &CallSyntax,
    tools: &[&Tool],
    text: &str,
    pre_opened_reasoning: bool,
    leniency: Leniency,
) -> Parsed {
    let mut p = Parser {
        syntax,
        tools,
        text,
        pos: 0,
        blocks: Vec::new(),
        next_id: 0,
        status: ParseStatus::Complete,
        leniency,
    };
    p.run(pre_opened_reasoning);
    Parsed {
        blocks: p.blocks,
        status: p.status,
    }
}

struct Parser<'a> {
    syntax: &'a CallSyntax,
    tools: &'a [&'a Tool],
    text: &'a str,
    pos: usize,
    blocks: Vec<Block>,
    next_id: usize,
    status: ParseStatus,
    leniency: Leniency,
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    fn eat(&mut self, literal: &str) -> bool {
        if self.rest().starts_with(literal) {
            self.pos += literal.len();
            true
        } else {
            false
        }
    }

    /// Consume `marker` allowing whitespace drift before it: skips
    /// leading whitespace in the input, then matches the marker's
    /// non-whitespace form. End markers carry canonical leading
    /// whitespace for rendering; parsing stays lenient.
    fn eat_ws_tolerant(&mut self, marker: &str) -> bool {
        let core = marker.trim_start();
        let ws = self.rest().len() - self.rest().trim_start().len();
        if self.text[self.pos + ws..].starts_with(core) {
            self.pos += ws + core.len();
            true
        } else {
            false
        }
    }

    fn push_text(&mut self, text: &str) {
        if text.is_empty() {
            return;
        }
        // Merge with a trailing Text block (partial-tail fallbacks
        // concatenate naturally).
        if let Some(Block::Text { text: prev, .. }) = self.blocks.last_mut() {
            let mut merged = prev.to_string();
            merged.push_str(text);
            *prev = merged.into();
            return;
        }
        self.blocks.push(text.to_string().into());
    }

    fn push_thought(&mut self, body: &str) {
        self.blocks.push(Block::Thought {
            thought: body.to_string().into(),
            signature: Cow::Borrowed(""),
        });
    }

    /// The incomplete tail starting at `from`: suppress or degrade
    /// per leniency.
    fn incomplete(&mut self, from: usize) {
        match self.leniency {
            Leniency::Streaming => {
                self.status = ParseStatus::NeedMoreInput;
            }
            Leniency::Final => {
                let tail = self.text[from..].to_string();
                self.push_text(&tail);
            }
        }
        self.pos = self.text.len();
    }

    fn run(&mut self, pre_opened_reasoning: bool) {
        let reasoning_on = self.syntax.reasoning.mode != ReasoningMode::None
            && !self.syntax.reasoning.end.is_empty();

        // Pre-opened reasoning: thought body runs to reasoning.end.
        if pre_opened_reasoning && reasoning_on {
            let end = self.syntax.reasoning.end.trim_start();
            match self.rest().find(end) {
                Some(at) => {
                    let body = &self.rest()[..at];
                    self.push_thought(body.trim_end());
                    self.pos += at + end.len();
                }
                None => {
                    // Entire text so far is thought-in-progress.
                    match self.leniency {
                        Leniency::Streaming => {
                            self.status = ParseStatus::NeedMoreInput;
                            self.pos = self.text.len();
                        }
                        Leniency::Final => {
                            // Unclosed thought at end of generation:
                            // surface what we have as a Thought — the
                            // model was cut off mid-reasoning.
                            let body = self.rest().to_string();
                            self.push_thought(body.trim_end());
                            self.pos = self.text.len();
                        }
                    }
                    return;
                }
            }
        }

        let trigger = self.syntax.trigger().to_string();
        let reasoning_start = self.syntax.reasoning.start.trim().to_string();

        while self.pos < self.text.len() {
            // Next structural landmark: reasoning open or call
            // trigger, whichever comes first.
            let rest = self.rest();
            let think_at = if reasoning_on && !reasoning_start.is_empty() {
                rest.find(&reasoning_start)
            } else {
                None
            };
            let trigger_at = if trigger.is_empty() {
                // Bare JSON-native dialects have no marker trigger:
                // the JSON opener itself is the call landmark. A
                // prose `{` costs a parse attempt that degrades back
                // to Text on failure — same trade upstream makes.
                if self.syntax.family == Family::JsonNative {
                    rest.find(['{', '['])
                } else {
                    None
                }
            } else {
                rest.find(trigger.as_str())
            };

            match (think_at, trigger_at) {
                (Some(t), None) => {
                    let prose = rest[..t].to_string();
                    self.push_text(&prose);
                    self.pos += t;
                    self.parse_thought(&reasoning_start);
                }
                (Some(t), Some(c)) if t < c => {
                    let prose = rest[..t].to_string();
                    self.push_text(&prose);
                    self.pos += t;
                    self.parse_thought(&reasoning_start);
                }
                (_, Some(c)) => {
                    let prose = rest[..c].to_string();
                    self.push_text(&prose);
                    self.pos += c;
                    self.parse_calls();
                }
                (None, None) => {
                    // Pure prose to the end. A trailing *partial*
                    // landmark prefix is possible mid-stream, but
                    // re-parse-per-tick makes holding back
                    // unnecessary for correctness of the final
                    // parse; streaming callers display ticks
                    // provisionally by design.
                    let prose = rest.to_string();
                    self.push_text(&prose);
                    self.pos = self.text.len();
                }
            }
        }
    }

    fn parse_thought(&mut self, open: &str) {
        let start = self.pos;
        debug_assert!(self.eat(open));
        let end = self.syntax.reasoning.end.trim();
        match self.rest().find(end) {
            Some(at) => {
                let body = &self.rest()[..at];
                let body = body
                    .strip_prefix('\n')
                    .unwrap_or(body)
                    .trim_end()
                    .to_string();
                self.push_thought(&body);
                self.pos += at + end.len();
                // Swallow one newline after the close, mirroring how
                // templates lay the tag out.
                let _ = self.eat("\n");
            }
            None => self.incomplete(start),
        }
    }

    /// Parse the call section beginning at the trigger.
    fn parse_calls(&mut self) {
        let start = self.pos;
        let has_section = !self.syntax.section_start.is_empty();
        if has_section {
            debug_assert!(self.rest().starts_with(&self.syntax.section_start));
            self.pos += self.syntax.section_start.len();
        }

        loop {
            // Per-call opener (when distinct from the section).
            if !self.syntax.per_call_start.is_empty()
                && !self.eat(&self.syntax.per_call_start.clone())
            {
                // For repeat calls the trigger may re-occur with
                // leading whitespace between calls.
                let ws = self.rest().len() - self.rest().trim_start().len();
                let after_ws = self.pos + ws;
                if self.text[after_ws..]
                    .starts_with(&self.syntax.per_call_start)
                {
                    self.pos = after_ws + self.syntax.per_call_start.len();
                } else {
                    break;
                }
            }

            match self.parse_one_call() {
                CallOutcome::Parsed => {
                    if !self.syntax.per_call_end.is_empty() {
                        let pce = self.syntax.per_call_end.clone();
                        self.eat_ws_tolerant(&pce);
                    }
                    // Another call? Only when a per-call opener
                    // exists to delimit it.
                    if self.syntax.per_call_start.is_empty() {
                        break;
                    }
                    let ws = self.rest().len() - self.rest().trim_start().len();
                    if !self.text[self.pos + ws..]
                        .starts_with(&self.syntax.per_call_start)
                    {
                        break;
                    }
                    // Loop continues; opener consumed at loop head.
                }
                CallOutcome::Incomplete => {
                    self.incomplete(start);
                    return;
                }
                CallOutcome::Malformed => {
                    // Degrade the whole section from the trigger to
                    // the next landmark (or end) into Text — nothing
                    // silently dropped; Session decides severity.
                    let upto = match self.text[start + 1..]
                        .find(self.syntax.trigger())
                    {
                        Some(next) => start + 1 + next,
                        None => self.text.len(),
                    };
                    let chunk = self.text[start..upto].to_string();
                    self.push_text(&chunk);
                    self.pos = upto;
                    return;
                }
            }
        }

        if has_section && !self.syntax.section_end.is_empty() {
            let se = self.syntax.section_end.clone();
            if !self.eat_ws_tolerant(&se) && self.rest().trim().is_empty() {
                // Closer not yet generated.
                self.incomplete(start);
            }
        }
        // Swallow one trailing newline after the call section.
        let _ = self.eat("\n");
    }

    fn parse_one_call(&mut self) -> CallOutcome {
        match self.syntax.family {
            Family::TagWithTagged => self.parse_tagged_call(),
            Family::TagWithJson => self.parse_tag_json_call(),
            Family::JsonNative => self.parse_json_native_call(),
            Family::None => CallOutcome::Malformed,
        }
    }

    fn schema_for(&self, tool: &str, param: &str) -> Option<Value> {
        self.tools
            .iter()
            .find(|t| t.name.as_ref() == tool)?
            .schema
            .get("properties")?
            .get(param)
            .cloned()
    }

    fn push_call(&mut self, name: String, input: Value) {
        let call = ToolUse {
            id: Cow::Owned(format!("call_{}_{}", self.next_id, name)),
            name: Cow::Owned(name),
            input,
            cache_control: None,
            caller: None,
        };
        self.next_id += 1;
        self.blocks.push(Block::ToolUse { call });
    }

    fn parse_tagged_call(&mut self) -> CallOutcome {
        let f = &self.syntax.function;
        let a = &self.syntax.arguments;

        if !self.eat(&f.name_prefix.clone()) {
            return if self.rest().is_empty()
                || f.name_prefix.starts_with(self.rest())
            {
                CallOutcome::Incomplete
            } else {
                CallOutcome::Malformed
            };
        }
        let Some(name_end) = self.rest().find(&f.name_suffix) else {
            return CallOutcome::Incomplete;
        };
        let name = self.rest()[..name_end].to_string();
        if name.is_empty() || name.len() > 256 {
            return CallOutcome::Malformed;
        }
        self.pos += name_end + f.name_suffix.len();

        let mut args = serde_json::Map::new();
        loop {
            // Function close ends the argument list.
            if !f.close.is_empty() {
                let ws = self.rest().len() - self.rest().trim_start().len();
                if self.text[self.pos + ws..].starts_with(f.close.trim_end()) {
                    self.pos += ws + f.close.trim_end().len();
                    // Absorb the close's own trailing whitespace.
                    let _ = self.eat("\n");
                    break;
                }
            }
            if !self.eat(&a.name_prefix.clone()) {
                // Neither an argument nor a close: incomplete if we
                // could still be mid-marker, else malformed.
                return if f.close.starts_with(self.rest())
                    || a.name_prefix.starts_with(self.rest())
                    || self.rest().trim().is_empty()
                {
                    CallOutcome::Incomplete
                } else {
                    CallOutcome::Malformed
                };
            }
            let Some(key_end) = self.rest().find(&a.name_suffix) else {
                return CallOutcome::Incomplete;
            };
            let key = self.rest()[..key_end].to_string();
            self.pos += key_end + a.name_suffix.len();
            if !a.value_prefix.is_empty() && !self.eat(&a.value_prefix.clone())
            {
                return CallOutcome::Incomplete;
            }
            let Some(val_end) = self.rest().find(&a.value_suffix) else {
                return CallOutcome::Incomplete;
            };
            let raw = self.rest()[..val_end].to_string();
            self.pos += val_end + a.value_suffix.len();

            let value = self.coerce_value(&name, &key, &raw);
            args.insert(key, value);

            if !a.separator.is_empty() {
                let _ = self.eat(&a.separator.clone());
            }
        }

        self.push_call(name, Value::Object(args));
        CallOutcome::Parsed
    }

    /// Schema-guided coercion of a raw tagged value (llama.cpp
    /// mapper parity): `string`-typed (or unknown) params stay raw;
    /// otherwise parse as JSON after pythonism normalization with
    /// bounded brace healing; fall back to a raw string.
    fn coerce_value(&self, tool: &str, param: &str, raw: &str) -> Value {
        let schema = self.schema_for(tool, param);
        let is_string = match &schema {
            Some(s) => s.get("type").and_then(|t| t.as_str()) == Some("string"),
            None => true,
        };
        if is_string {
            return Value::String(raw.to_string());
        }
        let trimmed = raw.trim();
        if let Ok(v) = serde_json::from_str::<Value>(trimmed) {
            return v;
        }
        let healed = heal_json(trimmed);
        if let Ok(v) = serde_json::from_str::<Value>(&healed) {
            return v;
        }
        Value::String(raw.to_string())
    }

    fn parse_tag_json_call(&mut self) -> CallOutcome {
        let f = &self.syntax.function;
        if !self.eat(&f.name_prefix.clone()) {
            return if self.rest().is_empty()
                || f.name_prefix.starts_with(self.rest())
            {
                CallOutcome::Incomplete
            } else {
                CallOutcome::Malformed
            };
        }
        let Some(name_end) = self.rest().find(&f.name_suffix) else {
            return CallOutcome::Incomplete;
        };
        let name = self.rest()[..name_end].to_string();
        self.pos += name_end + f.name_suffix.len();

        let Some((json_len, complete)) = balanced_json_len(self.rest()) else {
            return CallOutcome::Malformed;
        };
        if !complete {
            return CallOutcome::Incomplete;
        }
        let body = &self.rest()[..json_len];
        let input = match parse_json_healed(body) {
            Some(v) => v,
            None => return CallOutcome::Malformed,
        };
        self.pos += json_len;

        if !f.close.is_empty() {
            let ws = self.rest().len() - self.rest().trim_start().len();
            if self.text[self.pos + ws..].starts_with(f.close.trim_end()) {
                self.pos += ws + f.close.trim_end().len();
            } else if self.rest().trim().is_empty() {
                return CallOutcome::Incomplete;
            }
        }
        self.push_call(name, input);
        CallOutcome::Parsed
    }

    fn parse_json_native_call(&mut self) -> CallOutcome {
        // Skip leading whitespace inside the section.
        let ws = self.rest().len() - self.rest().trim_start().len();
        self.pos += ws;
        let Some((json_len, complete)) = balanced_json_len(self.rest()) else {
            return if self.rest().trim().is_empty() {
                CallOutcome::Incomplete
            } else {
                CallOutcome::Malformed
            };
        };
        if !complete {
            return CallOutcome::Incomplete;
        }
        let body = &self.rest()[..json_len];
        let Some(parsed) = parse_json_healed(body) else {
            return CallOutcome::Malformed;
        };
        self.pos += json_len;

        // Array-wrapped parallel calls.
        let calls: Vec<Value> = match parsed {
            Value::Array(items) => items,
            other => vec![other],
        };
        for call in calls {
            let Some((name, input)) = self.map_json_call(&call) else {
                return CallOutcome::Malformed;
            };
            self.push_call(name, input);
        }
        CallOutcome::Parsed
    }

    /// Map a parsed JSON call object to (name, args) via the
    /// dialect's field names, handling one-level `function` nesting
    /// and the name-is-key shape.
    fn map_json_call(&self, call: &Value) -> Option<(String, Value)> {
        let obj = call.as_object()?;
        if self.syntax.json.fun_name_is_key {
            let (name, args) = obj.iter().next()?;
            return Some((name.clone(), args.clone()));
        }
        let inner = if !self.syntax.json.function_field.is_empty() {
            obj.get(&self.syntax.json.function_field)
                .and_then(|v| v.as_object())
                .unwrap_or(obj)
        } else {
            obj
        };
        let name_field = leaf(&self.syntax.json.name_field, "name");
        let args_field = leaf(&self.syntax.json.args_field, "arguments");
        let name = inner
            .get(name_field)
            .or_else(|| obj.get(name_field))
            .and_then(|v| v.as_str())?
            .to_string();
        let args = inner
            .get(args_field)
            .or_else(|| obj.get(args_field))
            .cloned()
            .unwrap_or(Value::Object(serde_json::Map::new()));
        // Tolerate stringified args (some models double-encode).
        let args = match args {
            Value::String(s) => {
                serde_json::from_str(&s).unwrap_or(Value::String(s))
            }
            v => v,
        };
        Some((name, args))
    }
}

enum CallOutcome {
    Parsed,
    Incomplete,
    Malformed,
}

/// Last path component of an analyzed dotted field path
/// (`"function.name"` → `"name"`), with a default for empty.
fn leaf<'x>(path: &'x str, default: &'x str) -> &'x str {
    let p = path.rsplit('.').next().unwrap_or(path);
    if p.is_empty() {
        default
    } else {
        p
    }
}

/// Length of the balanced JSON value at the start of `s` (after
/// optional whitespace… no — caller trims). Returns `(len,
/// complete)`; `complete = false` when the input ends mid-value.
/// `None` when `s` doesn't start with a JSON value opener.
fn balanced_json_len(s: &str) -> Option<(usize, bool)> {
    let bytes = s.as_bytes();
    let first = *bytes.first()?;
    if first != b'{' && first != b'[' {
        return None;
    }
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    for (i, &b) in bytes.iter().enumerate() {
        if in_string {
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some((i + 1, true));
                }
            }
            _ => {}
        }
    }
    Some((s.len(), false))
}

/// Parse `body` as JSON, retrying with pythonism normalization and
/// bounded brace-healing. `None` when nothing works.
fn parse_json_healed(body: &str) -> Option<Value> {
    let body = body.trim();
    if let Ok(v) = serde_json::from_str(body) {
        return Some(v);
    }
    let healed = heal_json(body);
    serde_json::from_str(&healed).ok()
}

/// Normalize pythonisms (`True`/`False`/`None` outside strings,
/// single-quoted strings) and close up to 8 unbalanced braces /
/// brackets (llama.cpp's bounded heal on tool-close).
fn heal_json(body: &str) -> String {
    let mut out = String::with_capacity(body.len() + 8);
    let bytes = body.as_bytes();
    let mut i = 0usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut depth_stack: Vec<u8> = Vec::new();
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            out.push(b as char);
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += 1;
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                out.push('"');
                i += 1;
            }
            b'\'' => {
                // Single-quoted string → double-quoted, escaping
                // inner double quotes.
                out.push('"');
                i += 1;
                while i < bytes.len() {
                    let c = bytes[i];
                    if c == b'\\' && i + 1 < bytes.len() {
                        out.push('\\');
                        out.push(bytes[i + 1] as char);
                        i += 2;
                        continue;
                    }
                    if c == b'\'' {
                        i += 1;
                        break;
                    }
                    if c == b'"' {
                        out.push_str("\\\"");
                    } else {
                        out.push(c as char);
                    }
                    i += 1;
                }
                out.push('"');
            }
            b'{' => {
                depth_stack.push(b'}');
                out.push('{');
                i += 1;
            }
            b'[' => {
                depth_stack.push(b']');
                out.push('[');
                i += 1;
            }
            b'}' | b']' => {
                depth_stack.pop();
                out.push(b as char);
                i += 1;
            }
            _ => {
                // Word-boundary pythonism literals.
                let rest = &body[i..];
                let prev_boundary = i == 0
                    || !bytes[i - 1].is_ascii_alphanumeric()
                        && bytes[i - 1] != b'_';
                let mut replaced = false;
                if prev_boundary {
                    for (py, js) in
                        [("True", "true"), ("False", "false"), ("None", "null")]
                    {
                        if rest.starts_with(py)
                            && !rest[py.len()..].starts_with(|c: char| {
                                c.is_ascii_alphanumeric() || c == '_'
                            })
                        {
                            out.push_str(js);
                            i += py.len();
                            replaced = true;
                            break;
                        }
                    }
                }
                if !replaced {
                    // Preserve multi-byte UTF-8 intact.
                    let ch_len = body[i..]
                        .chars()
                        .next()
                        .map(char::len_utf8)
                        .unwrap_or(1);
                    out.push_str(&body[i..i + ch_len]);
                    i += ch_len;
                }
            }
        }
    }
    // Bounded close.
    for closer in depth_stack.into_iter().rev().take(8) {
        out.push(closer as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::DialectError;
    use crate::dialect::{render_reference, validate_representable};

    fn tool(name: &'static str) -> Tool {
        Tool::builder(name)
            .description("test")
            .schema(serde_json::json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"},
                    "detail": {"type": "string"},
                },
                "required": ["city", "days"],
            }))
            .build()
            .expect("valid test tool")
    }

    fn calls_of(blocks: &[Block]) -> Vec<(&str, &Value)> {
        blocks
            .iter()
            .filter_map(|b| match b {
                Block::ToolUse { call } => {
                    Some((call.name.as_ref(), &call.input))
                }
                _ => None,
            })
            .collect()
    }

    /// render_reference → parse_text is the identity on calls, for
    /// every family. The core Phase D invariant.
    #[test]
    fn reference_roundtrip_all_families() {
        let input = serde_json::json!({
            "city": "Paris\nFrance",   // embedded newline round-trips
            "days": 3,
        });
        let t = tool("get_weather");
        for syntax in [
            CallSyntax::qwen_xml(),
            CallSyntax::hermes_json(),
            CallSyntax::llama31_json(),
        ] {
            let emission =
                render_reference(&syntax, &[("get_weather", &input)])
                    .expect("representable");
            let parsed =
                parse_text(&syntax, &[&t], &emission, false, Leniency::Final);
            assert_eq!(
                parsed.status,
                ParseStatus::Complete,
                "{:?}: {emission:?} → {:#?}",
                syntax.family,
                parsed.blocks
            );
            let calls = calls_of(&parsed.blocks);
            assert_eq!(
                calls.len(),
                1,
                "{:?}: {emission:?} → {:#?}",
                syntax.family,
                parsed.blocks
            );
            assert_eq!(calls[0].0, "get_weather");
            assert_eq!(
                calls[0].1, &input,
                "{:?} emission {emission:?}",
                syntax.family
            );
        }
    }

    /// Adversarial raw values (plan amendments): trailing newlines
    /// and JSON-looking strings round-trip; unicode round-trips;
    /// the embedded close delimiter is a typed error at render.
    #[test]
    fn qwen_xml_adversarial_values() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");

        for value in [
            "trailing newline\n",
            "\nleading newline",
            "{\"looks\": \"like json\"}",
            "emoji 🍓 and 中文",
            "closing bracket ] and tag </function",
        ] {
            let input = serde_json::json!({"city": value, "days": 1});
            let emission =
                render_reference(&syntax, &[("get_weather", &input)])
                    .expect("representable");
            let parsed =
                parse_text(&syntax, &[&t], &emission, false, Leniency::Final);
            let calls = calls_of(&parsed.blocks);
            assert_eq!(calls.len(), 1, "value {value:?}: {parsed:#?}");
            assert_eq!(
                calls[0].1["city"],
                serde_json::json!(value),
                "value {value:?} must round-trip byte-exact"
            );
        }

        // The unrepresentable case: typed error, not silent damage.
        let evil = serde_json::json!({
            "city": "sneaky\n</parameter>\ninjected",
            "days": 1,
        });
        let err = render_reference(&syntax, &[("get_weather", &evil)])
            .expect_err("must be unrepresentable");
        assert!(matches!(err, DialectError::UnrepresentableValue { .. }));
        let err = validate_representable(&syntax, "get_weather", &evil)
            .expect_err("validate too");
        assert!(matches!(err, DialectError::UnrepresentableValue { .. }));
    }

    /// Thought + prose + two calls, Qwen XML: full structure parses;
    /// schema coercion types the integer.
    #[test]
    fn qwen_xml_full_turn() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let text = "<think>\nplanning the calls\n</think>\nSure, checking \
                    both cities.\n<tool_call>\n<function=get_weather>\n\
                    <parameter=city>\nParis\n</parameter>\n\
                    <parameter=days>\n3\n</parameter>\n</function>\n\
                    </tool_call>\n<tool_call>\n<function=get_weather>\n\
                    <parameter=city>\nLondon\n</parameter>\n\
                    <parameter=days>\n5\n</parameter>\n</function>\n\
                    </tool_call>";
        // Pre-opened form: the same text minus the literal <think>.
        for (input_text, pre_opened) in
            [(text, false), (text.strip_prefix("<think>").unwrap(), true)]
        {
            let parsed = parse_text(
                &syntax,
                &[&t],
                input_text,
                pre_opened,
                Leniency::Final,
            );
            let blocks = &parsed.blocks;
            assert!(
                matches!(&blocks[0], Block::Thought { thought, .. }
                    if thought.contains("planning")),
                "pre_opened={pre_opened}: {blocks:#?}"
            );
            assert!(
                matches!(&blocks[1], Block::Text { text, .. }
                    if text.contains("checking")),
                "pre_opened={pre_opened}: {blocks:#?}"
            );
            let calls = calls_of(blocks);
            assert_eq!(calls.len(), 2, "{blocks:#?}");
            assert_eq!(calls[0].1["city"], serde_json::json!("Paris"));
            // Schema-guided coercion: integer, not string.
            assert_eq!(calls[0].1["days"], serde_json::json!(3));
            assert_eq!(calls[1].1["city"], serde_json::json!("London"));
        }
    }

    /// Prefix-chop atomicity: parsing any prefix of a full emission
    /// in streaming mode never surfaces a call that isn't a prefix
    /// of the final call list, and never errors.
    #[test]
    fn streaming_prefixes_are_atomic() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let input = serde_json::json!({"city": "Paris", "days": 3});
        let mut full = String::from("<think>\nhm\n</think>\nCalling.\n");
        full.push_str(
            &render_reference(&syntax, &[("get_weather", &input)]).unwrap(),
        );
        let final_calls: Vec<String> = {
            let parsed =
                parse_text(&syntax, &[&t], &full, false, Leniency::Final);
            assert_eq!(parsed.status, ParseStatus::Complete);
            calls_of(&parsed.blocks)
                .iter()
                .map(|(n, _)| n.to_string())
                .collect()
        };
        for i in 0..=full.len() {
            if !full.is_char_boundary(i) {
                continue;
            }
            let parsed = parse_text(
                &syntax,
                &[&t],
                &full[..i],
                false,
                Leniency::Streaming,
            );
            let calls = calls_of(&parsed.blocks);
            assert!(
                calls.len() <= final_calls.len(),
                "prefix {i}: {:#?}",
                parsed.blocks
            );
            for (j, (name, _)) in calls.iter().enumerate() {
                assert_eq!(*name, final_calls[j], "prefix {i}");
            }
        }
    }

    /// Final-mode leniency: a dangling call degrades to Text, matching
    /// the BlockParser::finish contract.
    #[test]
    fn final_mode_degrades_partial_call_to_text() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let text = "ok\n<tool_call>\n<function=get_weather>\n<parameter=ci";
        let parsed = parse_text(&syntax, &[&t], text, false, Leniency::Final);
        assert_eq!(parsed.status, ParseStatus::Complete);
        assert!(calls_of(&parsed.blocks).is_empty());
        let joined: String = parsed
            .blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text, .. } => Some(text.to_string()),
                _ => None,
            })
            .collect();
        assert!(joined.contains("<tool_call>"), "{parsed:#?}");
    }

    /// JSON healing: pythonisms + unbalanced tails.
    #[test]
    fn heal_json_pythonisms_and_braces() {
        assert_eq!(
            heal_json(r#"{"a": True, "b": None, "c": False}"#),
            r#"{"a": true, "b": null, "c": false}"#
        );
        assert_eq!(heal_json(r#"{'a': 'x"y'}"#), r#"{"a": "x\"y"}"#);
        assert_eq!(heal_json(r#"{"a": [1, 2"#), r#"{"a": [1, 2]}"#);
        // Inside strings, pythonisms are untouched.
        assert_eq!(heal_json(r#"{"a": "True None"}"#), r#"{"a": "True None"}"#);
        // Identifier boundaries respected.
        assert_eq!(heal_json(r#"{"a": Truex}"#), r#"{"a": Truex}"#);
    }

    /// JsonNative with array-wrapped parallel calls (upstream's
    /// tools_array_wrapped shape) maps every element.
    #[test]
    fn json_native_array_wrapped_parallel() {
        let mut syntax = CallSyntax::llama31_json();
        syntax.json.tools_array_wrapped = true;
        let t = tool("get_weather");
        let text = r#"[{"name": "get_weather", "parameters": {"city": "Paris", "days": 1}}, {"name": "get_weather", "parameters": {"city": "Rome", "days": 2}}]"#;
        let parsed = parse_text(&syntax, &[&t], text, false, Leniency::Final);
        let calls = calls_of(&parsed.blocks);
        assert_eq!(calls.len(), 2, "{parsed:#?}");
        assert_eq!(calls[1].1["city"], serde_json::json!("Rome"));
    }

    /// The emitted grammar accepts its own reference render — the
    /// emitter/renderer consistency half of round-trip stability.
    #[test]
    fn grammar_accepts_reference_render() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        use crate::{Grammar, GrammarState};
        use std::sync::Arc;

        let t = tool("get_weather");
        let input = serde_json::json!({
            "city": "Paris",
            "days": 3,
            "detail": "with wind\nand rain",
        });
        for syntax in [
            CallSyntax::qwen_xml(),
            CallSyntax::hermes_json(),
            CallSyntax::llama31_json(),
        ] {
            let emission =
                render_reference(&syntax, &[("get_weather", &input)])
                    .expect("representable");
            let src = grammar_source(
                &syntax,
                &[&t],
                &EmitOptions {
                    anchor: Anchor::Lazy,
                    parallel: false,
                },
            )
            .expect("emit");
            let grammar = Arc::new(Grammar::parse(&src).unwrap_or_else(|e| {
                panic!("{:?} grammar: {e}\n{src}", syntax.family)
            }));
            let mut state = GrammarState::new(grammar);
            assert!(
                state.advance_bytes(emission.as_bytes()).is_ok()
                    && state.is_complete(),
                "{:?}: grammar must accept its reference render\n\
                 emission: {emission:?}\ngrammar:\n{src}",
                syntax.family
            );
        }
    }
}
