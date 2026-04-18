//! Streaming parser: raw model bytes → [`Block`]s.
//!
//! The parser is a small state machine over three fixed literal tags:
//!
//! * `<think>...</think>` → [`Block::Thought`]
//! * `<tool_call>...</tool_call>` → [`Block::ToolUse`] (serde-parsed
//!   JSON body)
//! * everything else → [`Block::Text`]
//!
//! Tag scanning is byte-level because the tags are fixed ASCII
//! literals. JSON bodies are parsed via serde using a
//! [`#[serde(alias)]`][serde-alias]-annotated struct, so the parser is
//! tolerant of `arguments` vs `parameters` keys without any hand-rolled
//! field lookups.
//!
//! # Streaming contract
//!
//! [`BlockParser::push`] takes a chunk of bytes and returns every
//! complete block that could be extracted from the accumulated buffer.
//! If a tag (open or close) is only partially present at the tail,
//! bytes up to the partial tag are emitted as [`Block::Text`] and the
//! partial prefix is held over for the next `push`.
//!
//! [`BlockParser::finish`] drains whatever remains:
//!
//! * Clean `Prose` state with empty buffer → no extra blocks.
//! * Clean `Prose` state with leftover bytes → one final
//!   [`Block::Text`].
//! * Mid-`<think>` / `<tool_call>` with no closing tag yet → one
//!   [`Block::Text`] containing the opening tag + whatever body
//!   arrived. Downstream code (e.g. [`Session::complete`]) can
//!   upgrade this into a `GrammarViolation` error when the completion
//!   was grammar-forced.
//!
//! Malformed JSON inside a well-framed `<tool_call>` falls back to a
//! [`Block::Text`] containing the full tagged literal, rather than
//! panicking or silently dropping. Same reasoning: the Session layer
//! decides whether that's fatal.
//!
//! [serde-alias]: https://serde.rs/field-attrs.html#alias
//! [`Session::complete`]: crate::Session

use std::borrow::Cow;

use serde::Deserialize;

use crate::prompt::{Block, ToolUse};

type CowStr = Cow<'static, str>;

const THINK_OPEN: &str = "<think>";
const THINK_CLOSE: &str = "</think>";
const TOOL_OPEN: &str = "<tool_call>";
const TOOL_CLOSE: &str = "</tool_call>";

/// Longest open tag — we only need to hold back this much at the tail
/// of the buffer when looking for a partial tag match.
const MAX_OPEN_TAG_LEN: usize = 11; // "<tool_call>"

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Prose,
    InThink,
    InToolCall,
}

/// Pushed-byte streaming block parser. See module docs for the
/// contract.
#[derive(Debug)]
pub struct BlockParser {
    buffer: String,
    state: State,
    /// Synthetic ID counter. Model outputs like cogito's don't include
    /// an `id` with each tool call, but downstream tool-result
    /// correlation (in a follow-up `ToolResult` block) requires one.
    /// Seeding per-completion with a counter keeps IDs unique within
    /// the message.
    next_id: usize,
}

impl BlockParser {
    /// New parser in the `Prose` state with an empty buffer.
    pub fn new() -> Self {
        Self {
            buffer: String::new(),
            state: State::Prose,
            next_id: 0,
        }
    }

    /// Feed a chunk of bytes. Returns every [`Block`] that could be
    /// fully resolved from the accumulated buffer.
    pub fn push(&mut self, bytes: &str) -> Vec<Block<'static>> {
        self.buffer.push_str(bytes);
        self.drain()
    }

    /// Drain the parser. Call after the completion stream ends.
    ///
    /// * Clean state with no leftover bytes returns empty.
    /// * Incomplete `<think>` / `<tool_call>` emits a single
    ///   [`Block::Text`] wrapping the opening tag + body so the raw
    ///   completion text is preserved and the caller can decide
    ///   how to handle the truncation.
    pub fn finish(mut self) -> Vec<Block<'static>> {
        let mut out = self.drain();

        if !self.buffer.is_empty() {
            let remainder = match self.state {
                State::Prose => std::mem::take(&mut self.buffer),
                State::InThink => {
                    let body = std::mem::take(&mut self.buffer);
                    format!("{THINK_OPEN}{body}")
                }
                State::InToolCall => {
                    let body = std::mem::take(&mut self.buffer);
                    format!("{TOOL_OPEN}{body}")
                }
            };
            if !remainder.is_empty() {
                out.push(text_block(remainder));
            }
        }
        out
    }

    /// Whether the parser currently holds a non-empty buffer OR is
    /// mid-tag. Useful for Session's grammar-violation check.
    pub fn is_idle(&self) -> bool {
        self.state == State::Prose && self.buffer.is_empty()
    }

    fn drain(&mut self) -> Vec<Block<'static>> {
        let mut out = Vec::new();
        loop {
            match self.state {
                State::Prose => {
                    if !self.advance_prose(&mut out) {
                        break;
                    }
                }
                State::InThink => {
                    if !self.advance_in_think(&mut out) {
                        break;
                    }
                }
                State::InToolCall => {
                    if !self.advance_in_tool_call(&mut out) {
                        break;
                    }
                }
            }
        }
        out
    }

    /// In `Prose`, advance to the next opening tag. Return `true` if
    /// the loop should keep draining (state changed), `false` if
    /// nothing more can be done with the current buffer.
    fn advance_prose(&mut self, out: &mut Vec<Block<'static>>) -> bool {
        let think = self.buffer.find(THINK_OPEN);
        let tool = self.buffer.find(TOOL_OPEN);
        let next = match (think, tool) {
            (Some(t), Some(c)) if t <= c => Some((t, Tag::Think)),
            (Some(_), Some(c)) => Some((c, Tag::Tool)),
            (Some(t), None) => Some((t, Tag::Think)),
            (None, Some(c)) => Some((c, Tag::Tool)),
            (None, None) => None,
        };
        if let Some((pos, tag)) = next {
            if pos > 0 {
                out.push(text_block(self.buffer[..pos].to_owned()));
            }
            let open_len = match tag {
                Tag::Think => THINK_OPEN.len(),
                Tag::Tool => TOOL_OPEN.len(),
            };
            self.buffer.drain(..pos + open_len);
            self.state = match tag {
                Tag::Think => State::InThink,
                Tag::Tool => State::InToolCall,
            };
            return true;
        }
        // No open tag in buffer. Emit prose, hold back anything that
        // could still grow into an open tag.
        let hold = trailing_open_tag_prefix_len(&self.buffer);
        let emit_end = self.buffer.len() - hold;
        if emit_end > 0 {
            let emitted: String = self.buffer.drain(..emit_end).collect();
            out.push(text_block(emitted));
        }
        false
    }

    fn advance_in_think(&mut self, out: &mut Vec<Block<'static>>) -> bool {
        let Some(close_at) = self.buffer.find(THINK_CLOSE) else {
            return false;
        };
        let body: String = self.buffer.drain(..close_at).collect();
        self.buffer.drain(..THINK_CLOSE.len());
        out.push(Block::Thought {
            thought: CowStr::from(body),
            signature: Cow::Borrowed(""),
        });
        self.state = State::Prose;
        true
    }

    fn advance_in_tool_call(&mut self, out: &mut Vec<Block<'static>>) -> bool {
        let Some(close_at) = self.buffer.find(TOOL_CLOSE) else {
            return false;
        };
        let body: String = self.buffer.drain(..close_at).collect();
        self.buffer.drain(..TOOL_CLOSE.len());
        match parse_tool_call_body(body.trim(), self.next_id) {
            Ok(call) => {
                self.next_id += 1;
                out.push(Block::ToolUse { call });
            }
            Err(_) => {
                // Malformed JSON inside a well-framed tool_call — fall
                // back to Text with the full tagged literal so nothing
                // is silently dropped. Session-level logic decides
                // whether to treat this as GrammarViolation.
                out.push(text_block(format!("{TOOL_OPEN}{body}{TOOL_CLOSE}")));
            }
        }
        self.state = State::Prose;
        true
    }
}

impl Default for BlockParser {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Copy)]
enum Tag {
    Think,
    Tool,
}

/// Non-streaming shortcut. Feeds the whole `output` and drains.
pub fn parse_completion(output: &str) -> Vec<Block<'static>> {
    let mut p = BlockParser::new();
    let mut out = p.push(output);
    out.extend(p.finish());
    out
}

fn text_block(text: String) -> Block<'static> {
    Block::Text {
        text: CowStr::from(text),
        cache_control: None,
    }
}

/// Longest prefix of any open tag present at the end of `buf`. Used
/// to know how many trailing bytes in `Prose` state we must hold back
/// (they could still grow into a complete open tag on the next push).
fn trailing_open_tag_prefix_len(buf: &str) -> usize {
    let mut best = 0;
    for tag in &[THINK_OPEN, TOOL_OPEN] {
        // Only partial prefixes count; a complete match would have
        // been found by `find` already.
        for k in 1..tag.len() {
            let prefix = &tag[..k];
            if buf.len() >= k && buf.is_char_boundary(buf.len() - k) {
                if buf[buf.len() - k..] == *prefix {
                    best = best.max(k);
                }
            }
        }
    }
    best
}

/// Wire type for a tool call JSON body. `#[serde(alias)]` lets us
/// accept either `"arguments"` (cogito/qwen/hermes) or `"parameters"`
/// (Llama 3.1) without any runtime dispatch.
#[derive(Deserialize)]
struct ToolCallWire {
    name: String,
    #[serde(alias = "parameters")]
    arguments: serde_json::Value,
}

fn parse_tool_call_body(
    body: &str,
    synth_id: usize,
) -> Result<ToolUse<'static>, serde_json::Error> {
    let wire: ToolCallWire = serde_json::from_str(body)?;
    Ok(ToolUse {
        id: Cow::Owned(format!("call_{synth_id}_{}", wire.name)),
        name: Cow::Owned(wire.name),
        input: wire.arguments,
        cache_control: None,
    })
}

// The `CowStr` type from misanthropic — re-exported for internal use.
// It's a `Cow<'static, str>` alias that `Block::Text` / `Block::Thought`
// want, so we avoid a Cow::Owned everywhere.
#[cfg(test)]
mod tests {
    use super::*;

    fn text_content<'a>(block: &'a Block<'a>) -> &'a str {
        match block {
            Block::Text { text, .. } => text,
            _ => panic!("expected Text, got {block:?}"),
        }
    }

    fn thought_content<'a>(block: &'a Block<'a>) -> &'a str {
        match block {
            Block::Thought { thought, .. } => thought,
            _ => panic!("expected Thought, got {block:?}"),
        }
    }

    fn tool_use<'a>(block: &'a Block<'a>) -> &'a ToolUse<'a> {
        match block {
            Block::ToolUse { call } => call,
            _ => panic!("expected ToolUse, got {block:?}"),
        }
    }

    // --- batch parse_completion ---

    #[test]
    fn pure_prose_is_one_text_block() {
        let blocks = parse_completion("just some text");
        assert_eq!(blocks.len(), 1);
        assert_eq!(text_content(&blocks[0]), "just some text");
    }

    #[test]
    fn empty_input_returns_no_blocks() {
        assert!(parse_completion("").is_empty());
    }

    #[test]
    fn think_block_plus_trailing_text() {
        let blocks = parse_completion("<think>plan</think>done");
        assert_eq!(blocks.len(), 2);
        assert_eq!(thought_content(&blocks[0]), "plan");
        assert_eq!(text_content(&blocks[1]), "done");
    }

    #[test]
    fn tool_call_block_basic() {
        let src = "<tool_call>\n{\"name\":\"x\",\"arguments\":{\"a\":1}}\n</tool_call>";
        let blocks = parse_completion(src);
        assert_eq!(blocks.len(), 1);
        let call = tool_use(&blocks[0]);
        assert_eq!(call.name, "x");
        assert_eq!(call.input, serde_json::json!({"a": 1}));
    }

    #[test]
    fn think_then_tool_call() {
        let src = "<think>reasoning</think>\n<tool_call>\n{\"name\":\"x\",\"arguments\":{}}\n</tool_call>";
        let blocks = parse_completion(src);
        assert_eq!(blocks.len(), 3); // think, "\n", tool_call
        assert_eq!(thought_content(&blocks[0]), "reasoning");
        assert_eq!(text_content(&blocks[1]), "\n");
        assert_eq!(tool_use(&blocks[2]).name, "x");
    }

    #[test]
    fn parameters_field_also_parses() {
        // Llama-3.1-style tool call uses `parameters` instead of
        // `arguments`. The serde alias handles both without any
        // runtime dispatch.
        let src =
            "<tool_call>\n{\"name\":\"x\",\"parameters\":{\"a\":1}}\n</tool_call>";
        let blocks = parse_completion(src);
        let call = tool_use(&blocks[0]);
        assert_eq!(call.input, serde_json::json!({"a": 1}));
    }

    #[test]
    fn malformed_tool_call_falls_back_to_text() {
        let src = "<tool_call>\nnot json\n</tool_call>";
        let blocks = parse_completion(src);
        assert_eq!(blocks.len(), 1);
        // Full tagged literal is preserved.
        assert!(
            text_content(&blocks[0]).contains("<tool_call>"),
            "expected tag preserved, got {:?}",
            text_content(&blocks[0])
        );
        assert!(text_content(&blocks[0]).contains("not json"));
    }

    #[test]
    fn adversarial_inline_json_stays_text() {
        // Prose that resembles a tool call but isn't wrapped in the
        // real tags must stay Text.
        let src = r#"Can you call {"name": "foo"}? It's fine."#;
        let blocks = parse_completion(src);
        assert_eq!(blocks.len(), 1);
        assert_eq!(text_content(&blocks[0]), src);
    }

    #[test]
    fn unclosed_think_drains_as_text() {
        let blocks = parse_completion("<think>never closed...");
        assert_eq!(blocks.len(), 1);
        assert_eq!(text_content(&blocks[0]), "<think>never closed...");
    }

    #[test]
    fn unclosed_tool_call_drains_as_text() {
        let blocks = parse_completion("<tool_call>\n{\"name\":\"x\"");
        assert_eq!(blocks.len(), 1);
        let text = text_content(&blocks[0]);
        assert!(text.starts_with("<tool_call>"));
        assert!(text.contains("\"name\":\"x\""));
    }

    // --- streaming behavior ---

    #[test]
    fn streaming_split_tag_across_pushes() {
        // Open tag split in the middle — parser must hold until it
        // completes.
        let mut p = BlockParser::new();
        let a = p.push("<thi");
        assert!(a.is_empty(), "no block should emit yet, got {a:?}");
        let b = p.push("nk>plan</th");
        // At this point we have the opening consumed and `plan</th` in
        // the buffer, so nothing should be emitted yet.
        assert!(b.is_empty(), "still should be empty, got {b:?}");
        let c = p.push("ink>done");
        let mut all = c;
        all.extend(p.finish());
        assert_eq!(all.len(), 2);
        assert_eq!(thought_content(&all[0]), "plan");
        assert_eq!(text_content(&all[1]), "done");
    }

    #[test]
    fn streaming_prose_emits_without_hold() {
        let mut p = BlockParser::new();
        let out = p.push("hello world");
        assert_eq!(out.len(), 1);
        assert_eq!(text_content(&out[0]), "hello world");
    }

    #[test]
    fn streaming_prose_holds_back_tag_prefix() {
        // Trailing `<` could grow into a real tag — parser must hold
        // until next push disambiguates.
        let mut p = BlockParser::new();
        let out = p.push("hello<");
        assert_eq!(out.len(), 1);
        assert_eq!(
            text_content(&out[0]),
            "hello",
            "`<` should be held back pending tag disambiguation"
        );
        let out = p.push("not a tag");
        assert_eq!(out.len(), 1);
        assert_eq!(text_content(&out[0]), "<not a tag");
    }

    #[test]
    fn is_idle_reflects_state() {
        let mut p = BlockParser::new();
        assert!(p.is_idle());
        let _ = p.push("<think>");
        assert!(!p.is_idle());
        let _ = p.push("plan</think>");
        assert!(p.is_idle());
    }
}
