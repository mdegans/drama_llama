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
//! grammar violation. [`Leniency::Clipped`] is the end of a generation
//! that was *cut short* (`max_tokens`, a stop sequence): an incomplete
//! call comes back as Anthropic returns one — its input the members
//! that completed — and an incomplete thought surfaces open.
//!
//! ## Coercion & healing (llama.cpp mapper parity)
//!
//! Tagged raw values are schema-coerced: params typed `string` (or
//! unknown) stay raw strings, as do nullable strings bar a literal
//! `null`; a finite set of strings (`enum`, `const`, nullable or not)
//! reads by exact match on its members' raw spellings, as the grammar
//! generates it; anything else is parsed as JSON after
//! normalizing pythonisms (`True`/`False`/`None`, single-quoted
//! strings) with bounded brace-healing, falling back to a JSON
//! string of the raw bytes when parsing still fails. Parse never
//! hard-errors on content — worst case a call degrades to text.

use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::chat_template::is_tool_name;
use crate::grammar_compile::MAX_NESTING;
use crate::prompt::{Block, ToolUse};
use crate::Tool;

use super::emit::{tagged_values, TaggedValue};
use super::partial::{
    marker_holdback, read_partial, unclosed_json, Flavor, OpenStrings,
};
use super::provenance::Provenance;
use super::{harmony, CallSyntax, Family, ReasoningMode};

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
    /// End of a generation that was cut short — `max_tokens` ran out
    /// or a stop sequence fired (#121, #122). An incomplete **call**
    /// comes back as Anthropic returns one a clip cut: a
    /// [`Block::ToolUse`] whose input holds only the members that
    /// *completed*, the one in flight dropped whole ([`Parsed::status`]
    /// still [`ParseStatus::NeedMoreInput`]: the call never closed). An
    /// incomplete **thought** surfaces as an open thought, exactly as
    /// under [`Self::Final`].
    ///
    /// This is Anthropic parity, captured 2026-09-30 on
    /// claude-haiku-4-5 (raw bytes; misanthropic's
    /// `misanthropic/test/data/stop/` `clip*.*`): a forced `write_file`
    /// cut mid-`contents` came back `{"path":"hello.py"}` under
    /// `stop_reason: max_tokens`, and 140 output tokens into a 200-word
    /// `contents` string, still `{"path":"story.txt"}`. The same prompt
    /// must drive a client the same way on both backends, and a client
    /// that gates dispatch on `stop_reason` (as it must) never
    /// dispatches it. Nested containers keep their completed members at
    /// every depth, the one in progress dropped — inferred, as only the
    /// top level is captured. The calls that closed before the cut
    /// stand unchanged.
    ///
    /// Withheld instead: a call whose *name* the cut left incomplete
    /// (Anthropic's `tool_use` block carries its name whole, so there
    /// is nothing to return), and one the cut left unreadable, running
    /// to the end of input. Neither is seated as prose, where its frame
    /// marker would poison the next ingest.
    ///
    /// One exception: a trigger-less dialect (bare-JSON, Llama 3.1)
    /// cannot tell a clipped call from clipped prose JSON — its call
    /// landmark is any `{` — so there the tail degrades as under
    /// `Final`, and structured output clipped mid-object keeps its
    /// text.
    Clipped,
}

#[derive(Debug)]
pub struct Parsed {
    pub blocks: Vec<Block>,
    pub status: ParseStatus,
}

/// The call section a parse ran out of input inside, with a name to
/// show for it — what a cut leaves of a call in flight (#121, #122; see
/// [`Leniency::Clipped`]).
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct OpenCall {
    /// What a clip returns: any calls the unclosed section completed
    /// that a streaming parse holds back with it (array-wrapped JSON),
    /// then the call cut short, its input the members that completed.
    pub calls: Vec<ToolUse>,
    /// The last call's input with the string value in flight kept as
    /// far as it is known to be text ([`OpenStrings::Held`]) — what a
    /// stop sequence is matched against.
    pub held_input: Value,
    /// The same, every byte of that string kept ([`OpenStrings::Raw`])
    /// — what locates a stop's cut in the raw generation.
    pub raw_input: Value,
    /// The last call's input as JSON left open where the cut fell
    /// (`{"path":"story.txt"`): what Anthropic streams as a clipped
    /// call's `input_json_delta`, with no `content_block_stop` after it.
    /// `None` when the last call closed (an array-wrapped section cut
    /// between calls).
    pub partial_json: Option<String>,
}

/// Streaming adapter over [`parse_text`]: accumulate pieces, re-parse
/// the whole text per tick, and diff against what has already been
/// yielded. A tool call emerges whole when its close marker arrives,
/// a thought once whatever follows it does (its signature records the
/// whitespace after it, and on Harmony the header of the final after
/// it — see [`parse_text`]); trailing prose streams incrementally as
/// byte deltas, holding back any tail that could still grow into a
/// dialect marker, and any prose that is so far only whitespace (a
/// parse never returns a whitespace-only `Text`).
///
/// The diff is sound because of two properties of [`parse_text`] on a
/// growing input: the parsed block *prefix* is stable (landmarks are
/// found by forward scan; longer input never re-classifies earlier
/// complete blocks), and a trailing `Text` block only ever grows by
/// appending (partial structures are suppressed under
/// [`Leniency::Streaming`], and degraded ones append). The regression
/// tests below pin both.
#[derive(Debug, Clone)]
pub struct StreamParser {
    syntax: CallSyntax,
    tools: Vec<crate::Tool>,
    pre_opened_reasoning: bool,
    /// Full accumulated generation, re-parsed each tick.
    text: String,
    /// Leading parsed blocks already yielded in full.
    stable_blocks: usize,
    /// Bytes of `blocks[stable_blocks]` (a still-growing trailing
    /// `Text`) already yielded as deltas.
    text_bytes_emitted: usize,
    /// The call the last re-parse ended inside, if any.
    open: Option<OpenCall>,
    /// When set, `text` is the generation with every reserved piece it
    /// *spelled* marked (see [`Provenance`]); what the parser yields is
    /// restored. `None` parses the text as it stands.
    provenance: Option<Provenance>,
    /// Each tool's parameter spellings, classified once for the whole
    /// generation rather than once per re-parse — every token.
    spellings: Spellings,
}

/// Each tool's parameter spellings ([`tagged_values`]), by index into
/// the parse's tools.
pub(crate) type Spellings = HashMap<usize, Arc<HashMap<String, TaggedValue>>>;

impl StreamParser {
    pub fn new(
        syntax: CallSyntax,
        tools: Vec<crate::Tool>,
        pre_opened_reasoning: bool,
    ) -> Self {
        Self {
            syntax,
            tools,
            pre_opened_reasoning,
            text: String::new(),
            stable_blocks: 0,
            text_bytes_emitted: 0,
            open: None,
            provenance: None,
            spellings: Spellings::new(),
        }
    }

    /// Parse with emission provenance: only framing the model emitted
    /// as a real reserved token is structure, and a reserved piece it
    /// spelled in ordinary tokens is text (see [`Provenance`]). Feed it
    /// with [`Self::push_token`]; [`Self::push`] then reads every piece
    /// as ordinary.
    pub(crate) fn with_provenance(mut self, provenance: Provenance) -> Self {
        self.provenance = Some(provenance);
        self
    }

    /// Feed one decoded piece; returns every block (or prose delta)
    /// newly resolved by it.
    pub fn push(&mut self, piece: &str) -> Vec<Block> {
        self.push_token(piece, None)
    }

    /// [`Self::push`], with the token that produced `piece` — what
    /// provenance keys on. Without provenance the token is ignored.
    pub(crate) fn push_token(
        &mut self,
        piece: &str,
        token: Option<crate::Token>,
    ) -> Vec<Block> {
        match self.provenance.as_mut() {
            Some(provenance) => {
                let settled = provenance.push(piece, token);
                self.text.push_str(&settled);
            }
            None => self.text.push_str(piece),
        }
        self.reparse(Leniency::Streaming)
    }

    /// Flush at end of generation: partial trailing structures
    /// degrade per [`Leniency::Final`] and held-back marker-prefix
    /// bytes are released.
    pub fn finish(&mut self) -> Vec<Block> {
        self.settle();
        self.reparse(Leniency::Final)
    }

    /// End of input: provenance's held tail can no longer grow into a
    /// spelled piece, so it joins the text as it stands.
    fn settle(&mut self) {
        if let Some(provenance) = self.provenance.as_mut() {
            let tail = provenance.finish();
            self.text.push_str(&tail);
        }
    }

    /// Flush a generation that was cut short (`max_tokens`, a stop
    /// sequence) per [`Leniency::Clipped`]: an incomplete trailing
    /// call comes back with the members of its input that completed,
    /// instead of degrading to text; everything else flushes as
    /// [`Self::finish`] would.
    pub fn finish_clipped(&mut self) -> Vec<Block> {
        self.settle();
        self.reparse(Leniency::Clipped)
    }

    /// The call the text so far ends inside, as of the last push or
    /// flush — a streaming parse yields nothing of it until it closes.
    pub(crate) fn open(&self) -> Option<&OpenCall> {
        self.open.as_ref()
    }

    /// The text so far as a batch parse reads it ([`Leniency::Final`]):
    /// its blocks whole, where the yields are deltas that cannot say
    /// where one text block ended and the next began — two Harmony
    /// channels, a preamble then the final, stream as adjacent text.
    pub(crate) fn blocks(&self) -> Vec<Block> {
        let tool_refs: Vec<&crate::Tool> = self.tools.iter().collect();
        let blocks = parse_text(
            &self.syntax,
            &tool_refs,
            &self.text,
            self.pre_opened_reasoning,
            Leniency::Final,
        )
        .blocks;
        match &self.provenance {
            Some(provenance) => provenance.restore_blocks(blocks),
            None => blocks,
        }
    }

    /// Whether the text so far ends in a structure in flight
    /// ([`Leniency::Clipped`] reports it unfinished): framing, not
    /// prose.
    pub(crate) fn in_flight(&self) -> bool {
        let tool_refs: Vec<&crate::Tool> = self.tools.iter().collect();
        let mut spellings = self.spellings.clone();
        parse_text_cached(
            &self.syntax,
            &tool_refs,
            &self.text,
            self.pre_opened_reasoning,
            Leniency::Clipped,
            &mut spellings,
        )
        .0
        .status
            == ParseStatus::NeedMoreInput
    }

    /// Longest tail of `text` that is a proper prefix of a dialect
    /// marker the prose scanner could re-classify — the open
    /// landmarks (call trigger, reasoning open) *and* the close
    /// markers that trail a parsed call (`per_call_end`,
    /// `section_end`: an incomplete close degrades to prose on this
    /// tick and is consumed as marker on the next). These bytes must
    /// be held back from prose deltas until disambiguated.
    fn landmark_holdback(&self, text: &str) -> usize {
        let tail = text.as_bytes();
        let mut best = 0;
        let generic = [
            self.syntax.trigger(),
            self.syntax.reasoning.start.trim(),
            // The close marker matters on its own for dialects where
            // a stray close terminates content (Gemma channels), and
            // the turn-exit marker is swallowed as envelope.
            self.syntax.reasoning.end.trim(),
            self.syntax.tool_response_start.trim(),
            self.syntax.per_call_end.trim(),
            self.syntax.section_end.trim(),
        ];
        // Harmony's markers live outside the generic fields: a
        // partial `<|end|>` / `<|start|>assistant` at the tail of
        // streamed final content must be held back exactly like a
        // partial `</tool_call>`. ` to=` is included because it opens
        // a role-header recipient at a block boundary — prose ending
        // in " to" is held a tick and released on the next byte.
        let harmony_markers = [
            harmony::START_ASSISTANT,
            harmony::CHANNEL,
            harmony::MESSAGE,
            harmony::END,
            harmony::CALL,
            harmony::RETURN,
            harmony::CONSTRAIN,
            harmony::TO_FUNCTIONS,
        ];
        let markers: &[&str] = if self.syntax.family == Family::Harmony {
            &harmony_markers
        } else {
            &generic
        };
        for marker in markers {
            let m = marker.as_bytes();
            for k in ((best + 1)..m.len()).rev() {
                if k > tail.len() {
                    continue;
                }
                if tail[tail.len() - k..] == m[..k] {
                    best = k;
                    break;
                }
            }
        }
        // Byte-wise prefix matches can only end mid-char for
        // non-ASCII markers; widen until the cut is a boundary so the
        // delta slice below stays valid UTF-8.
        while best > 0 && !text.is_char_boundary(text.len() - best) {
            best += 1;
        }
        best
    }

    /// `end`, pulled back so that `text[..end]` holds whole spelled
    /// pieces: a delta is restored on its own, so it must hold whole
    /// markers.
    fn whole_markers(&self, text: &str, end: usize) -> usize {
        match &self.provenance {
            Some(provenance) => provenance.cut_before_marker(text, end),
            None => end,
        }
    }

    /// How many trailing blocks a streaming parse withholds this tick,
    /// so that nothing yielded can change — or be a whitespace-only
    /// `Text`, which a parse never returns ([`fold_blank_text`]):
    ///
    /// * a trailing `Text` whose yieldable part is whitespace: it may
    ///   yet be the framing before a structure (`[/THINK]\n[TOOL_CA`),
    ///   folded into a thought or dropped once the structure lands;
    /// * a closed thought with nothing after it but such text: its
    ///   signature records what follows it (`ThoughtTail`), which is
    ///   not known until something does.
    fn held_tail(&self, blocks: &[Block]) -> usize {
        let blank_text = match blocks.last() {
            Some(Block::Text { text, .. }) => {
                let end = text.len() - self.landmark_holdback(text);
                text[..self.whole_markers(text, end)].trim().is_empty()
            }
            _ => false,
        };
        let before = blocks.len() - usize::from(blank_text);
        let thought = matches!(
            before.checked_sub(1).map(|at| &blocks[at]),
            Some(Block::Thought { .. })
        );
        usize::from(blank_text) + usize::from(thought)
    }

    fn reparse(&mut self, leniency: Leniency) -> Vec<Block> {
        let tool_refs: Vec<&crate::Tool> = self.tools.iter().collect();
        let (parsed, open) = parse_text_cached(
            &self.syntax,
            &tool_refs,
            &self.text,
            self.pre_opened_reasoning,
            leniency,
            &mut self.spellings,
        );
        self.open = match &self.provenance {
            Some(provenance) => open.map(|o| provenance.restore_open(o)),
            None => open,
        };
        let mut blocks = parsed.blocks;
        if leniency == Leniency::Streaming {
            let held = self.held_tail(&blocks);
            blocks.truncate(blocks.len() - held);
        }
        let last = blocks.len().saturating_sub(1);
        let mut out = Vec::new();
        for (i, block) in blocks.into_iter().enumerate() {
            if i < self.stable_blocks {
                continue;
            }
            let open_tail = i == last && leniency == Leniency::Streaming;
            match block {
                Block::Text { text, .. } => {
                    // Interior Text is final. Trailing Text may still
                    // grow (or its tail may become a marker), so under
                    // Streaming yield only the safe delta and keep the
                    // block open; under Final flush it whole.
                    let end = match open_tail {
                        true => text.len() - self.landmark_holdback(&text),
                        false => text.len(),
                    };
                    let end = self.whole_markers(&text, end);
                    // What was yielded is a prefix of this text unless
                    // a longer re-parse re-cut the block under it (a
                    // call opened inside an open thought that a later
                    // close turns back into thought) — model output
                    // reaches that, so the slice must not land
                    // mid-char.
                    let from = text.ceil_char_boundary(self.text_bytes_emitted);
                    if from != self.text_bytes_emitted.min(text.len()) {
                        tracing::warn!(
                            emitted = self.text_bytes_emitted,
                            "stream parser: trailing text re-cut mid-char"
                        );
                    }
                    if end > from {
                        let delta = &text[from..end];
                        out.push(match &self.provenance {
                            Some(p) => p.restore(delta).into_owned().into(),
                            None => delta.to_string().into(),
                        });
                        self.text_bytes_emitted = end;
                    }
                    if !open_tail {
                        self.stable_blocks = i + 1;
                        self.text_bytes_emitted = 0;
                    }
                }
                // Thought / ToolUse only materialize once their close
                // marker has been consumed — they cannot change on a
                // longer re-parse. Yield immediately.
                other => {
                    out.push(match &self.provenance {
                        Some(provenance) => provenance.restore_block(other),
                        None => other,
                    });
                    self.stable_blocks = i + 1;
                    self.text_bytes_emitted = 0;
                }
            }
        }
        out
    }
}

/// Parse `text` (the full accumulated generation) into blocks per
/// `syntax`.
///
/// No block is a whitespace-only [`Block::Text`]: Anthropic never
/// returns one and rejects one on ingest. Whitespace the model wrote
/// between a closed thought and what follows it is recorded in that
/// thought's `signature` instead (`drama_llama:tail;gap=…`), which the
/// chat template renderer puts back where it sat; anywhere else such a
/// run is dropped. A closed thought's signature also records the
/// content type of a Harmony final channel right after it
/// (`;constrain=json`, from `<|channel|>final <|constrain|>json`).
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
    parse_text_open(syntax, tools, text, pre_opened_reasoning, leniency).0
}

/// [`parse_text`], plus the call the text ends inside when there is one
/// with a name to show for it (under [`Leniency::Streaming`] and
/// [`Leniency::Clipped`]; `Final` degrades it instead).
pub(crate) fn parse_text_open(
    syntax: &CallSyntax,
    tools: &[&Tool],
    text: &str,
    pre_opened_reasoning: bool,
    leniency: Leniency,
) -> (Parsed, Option<OpenCall>) {
    parse_text_cached(
        syntax,
        tools,
        text,
        pre_opened_reasoning,
        leniency,
        &mut Spellings::new(),
    )
}

/// [`parse_text_open`], reading and filling `spellings`, so a caller that
/// parses the same tools again and again (the streaming re-parse)
/// classifies each tool once.
pub(crate) fn parse_text_cached(
    syntax: &CallSyntax,
    tools: &[&Tool],
    text: &str,
    pre_opened_reasoning: bool,
    leniency: Leniency,
    spellings: &mut Spellings,
) -> (Parsed, Option<OpenCall>) {
    let mut p = Parser {
        syntax,
        tools,
        text,
        pos: 0,
        blocks: Vec::new(),
        next_id: 0,
        status: ParseStatus::Complete,
        leniency,
        open: None,
        spellings: RefCell::new(std::mem::take(spellings)),
        landmarks: HashMap::new(),
    };
    p.run(pre_opened_reasoning);
    *spellings = p.spellings.into_inner();
    let parsed = Parsed {
        blocks: fold_blank_text(p.blocks),
        status: p.status,
    };
    (parsed, p.open)
}

/// `blocks` without a whitespace-only [`Block::Text`]: Anthropic never
/// returns one and rejects one on ingest, so neither may a parse. Such a
/// run is framing between structures — Mistral 4's
/// `[/THINK]\n[TOOL_CALLS]`, Gemma 4's `<channel|>\n<|tool_call>`,
/// Qwen's `</think>\n\n<tool_call>` — and the next render needs it
/// back, so after a closed thought it rides in that thought's
/// signature (`ThoughtTail::gap`, which the renderer re-inserts where
/// it sat). Anywhere else (before the turn's first structure, after a
/// call) it has no block to ride and is dropped: the one place a turn's
/// whitespace no longer re-renders.
pub(crate) fn fold_blank_text(blocks: Vec<Block>) -> Vec<Block> {
    use crate::prompt::{is_blank, ThoughtTail, OPEN_THOUGHT_SIGNATURE};
    let mut out: Vec<Block> = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            Block::Text { text, .. } if text.trim().is_empty() => {
                if let Some(Block::Thought { signature, .. }) = out.last_mut() {
                    if is_blank(&text) && signature != OPEN_THOUGHT_SIGNATURE {
                        let mut tail = ThoughtTail::of(signature);
                        tail.gap.push_str(&text);
                        *signature = tail.signature();
                    }
                }
            }
            other => out.push(other),
        }
    }
    out
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
    /// The call the input ended inside ([`OpenCall`]), read just before
    /// [`Self::incomplete`] handles it.
    open: Option<OpenCall>,
    /// Each tool's parameter spellings ([`tagged_values`]) by tool
    /// index, classified the first time a call to the tool needs one —
    /// not once per parameter read — and kept across re-parses by a
    /// caller that lends its own ([`parse_text_cached`]).
    spellings: RefCell<Spellings>,
    /// Each landmark's next occurrence ([`Self::find_landmark`]):
    /// `marker → (searched from, found at)`, absolute.
    landmarks: HashMap<String, (usize, Option<usize>)>,
}

impl<'a> Parser<'a> {
    fn rest(&self) -> &'a str {
        &self.text[self.pos..]
    }

    /// Where `marker` next occurs in [`Self::rest`], as
    /// `rest().find(marker)` — remembered in `landmarks`, so the scan for
    /// a landmark the text no longer holds runs to its end once, not
    /// again from every block boundary after it. Those rescans made a
    /// parse quadratic in its boundaries: 384 KB of Gemma 4's
    /// `<|tool_call>` alone (32k tokens, each a malformed call and a
    /// boundary) took 10 s to parse, and a stream re-parses on every
    /// token.
    fn find_landmark(&mut self, marker: &str) -> Option<usize> {
        let pos = self.pos;
        // `(from, at)`: `marker` first occurs at or after `from` at `at`
        // — so first at or after `pos` there too, for `from <= pos <=
        // at`, and nowhere after `pos` when `at` is `None`.
        if let Some(&(from, at)) = self.landmarks.get(marker) {
            if from <= pos && at.is_none_or(|at| at >= pos) {
                return at.map(|at| at - pos);
            }
        }
        let at = self.rest().find(marker);
        self.landmarks
            .insert(marker.to_string(), (pos, at.map(|at| at + pos)));
        at
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

    /// Consume a call opener whitespace-tolerantly: leading whitespace,
    /// the marker's trimmed core (the special — [`CallSyntax::trigger`]
    /// is the same core), then any whitespace after it. The canonical
    /// trailing `\n` the template lays out is *forced* by the grammar
    /// once the trigger fires, so drift here only ever comes from an
    /// unconstrained generation — and a real opener must still parse
    /// as a call, never be left in prose as a reserved special. An
    /// empty marker is vacuously consumed.
    fn eat_opener(&mut self, marker: &str) -> bool {
        let core = marker.trim();
        if core.is_empty() {
            return true;
        }
        let ws = self.rest().len() - self.rest().trim_start().len();
        if !self.text[self.pos + ws..].starts_with(core) {
            return false;
        }
        self.pos += ws + core.len();
        self.pos += self.rest().len() - self.rest().trim_start().len();
        true
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

    /// A closed thought's body as the chat template re-renders it: the
    /// bytes before the close marker, minus exactly the whitespace the
    /// marker is canonically spelled with (the `"\n"` of Qwen's
    /// `"\n</think>"`, Gemma 4's `"\n<channel|>"`) and nothing more.
    ///
    /// Not `trim_end`: a thought the model closes on a blank line keeps
    /// its extra `"\n"`, because a template that renders the thought
    /// verbatim (every baked one) puts the marker's own `"\n"` back and
    /// needs the rest from the block to reproduce the emission. A
    /// trimming stock template trims either way.
    fn closed_thought_body(&self, body: &'a str) -> &'a str {
        let end = &self.syntax.reasoning.end;
        let lead = &end[..end.len() - end.trim_start().len()];
        body.strip_suffix(lead).unwrap_or(body)
    }

    /// A thought's body after its open marker as the chat template
    /// re-renders it: minus exactly the whitespace the marker is
    /// canonically spelled with (the `"\n"` of Qwen's `"<think>\n"`,
    /// Gemma 4's `"<|channel>thought\n"`; nothing for Mistral 4's
    /// `"[THINK]"`, whose thought may itself start with a newline).
    fn opened_thought_body(&self, body: &'a str) -> &'a str {
        let start = &self.syntax.reasoning.start;
        let trail = &start[start.trim_end().len()..];
        body.strip_prefix(trail).unwrap_or(body)
    }

    /// After an *empty* pre-opened thought, consume the reasoning
    /// separator. An empty thought is no thought at all to the
    /// re-render, which then spells the thinking-off scaffold — Qwen's
    /// `<think>\n\n</think>\n\n` — so the separator the model wrote is
    /// the scaffold's, and left in the answer it would render twice.
    ///
    /// `true` when the rest of the input is withheld under
    /// [`Leniency::Streaming`]: a proper prefix of the separator may
    /// still grow into it, and a streamed `"\n"` cannot be taken back.
    ///
    /// Known residual: a *continued* open thought whose completion
    /// closes at once is merged with its seed on re-render, so the
    /// thought is no longer empty there, and a non-canonical gap after
    /// it (`"\n\n\n"`) re-renders one separator short. Stock trims
    /// every gap, so this is never worse than it.
    fn eat_scaffold_separator(&mut self) -> bool {
        let Some(sep) = self.syntax.reasoning.separator.as_deref() else {
            return false;
        };
        let rest = self.rest();
        if sep.is_empty() || rest.is_empty() {
            return false;
        }
        if rest.starts_with(sep) {
            self.pos += sep.len();
            return false;
        }
        if self.leniency == Leniency::Streaming && sep.starts_with(rest) {
            self.status = ParseStatus::NeedMoreInput;
            self.pos = self.text.len();
            return true;
        }
        false
    }

    fn push_thought(&mut self, body: &str) {
        // Empty thoughts carry no signal and some dialects emit them
        // as pure noise (Gemma 4's pre-closed / trailing channel
        // blocks) — drop rather than surface an empty block.
        if body.is_empty() {
            return;
        }
        self.push_closed_thought(body);
    }

    /// A thought the model wrote both markers of, kept even when empty
    /// (Mistral 4's `[THINK][/THINK]`, an empty Harmony analysis):
    /// the templates that render each thought render an empty one as
    /// the model wrote it, so dropping it lost the markers from the
    /// re-render. Anthropic returns empty thinking blocks too (display
    /// `omitted`). Two exceptions keep dropping it: Gemma 4, whose empty
    /// channel is the template's own thinking-off scaffold — or noise
    /// around content — and renders from no thought at all; and a
    /// dialect that re-ingests thoughts inline, where an empty one would
    /// put a bare `<think></think>` into the turn's content.
    fn push_closed_thought(&mut self, body: &str) {
        let drop_empty = self.syntax.family == Family::TagWithDict
            || self.syntax.reasoning.reingest
                == super::ReasoningReingest::InlineThink;
        if body.is_empty() && drop_empty {
            return;
        }
        self.blocks.push(Block::Thought {
            thought: body.to_string().into(),
            signature: Cow::Borrowed(""),
        });
    }

    /// Push a thought whose close marker never arrived — the model was
    /// cut off mid-reasoning.
    ///
    /// Two differences from [`Self::push_thought`], both load-bearing:
    ///
    /// * The body is stored **raw**. The closed path strips the close
    ///   marker's canonical leading whitespace (the `"\n"` of
    ///   `"\n</think>"`, see `closed_thought_body`) because the
    ///   template's close marker re-supplies it on render; an open
    ///   thought has no close, so any stripped byte would be gone for
    ///   good and the re-render would no longer match the KV.
    /// * It carries [`OPEN_THOUGHT_SIGNATURE`], which is what stops the
    ///   renderer from inventing a close marker the model never wrote.
    ///
    /// [`OPEN_THOUGHT_SIGNATURE`]: crate::prompt::OPEN_THOUGHT_SIGNATURE
    fn push_open_thought(&mut self, body: &str) {
        if body.is_empty() {
            // Nothing was generated before the cutoff. Dropping it is
            // byte-safe: under a pre-opened template the generation
            // prompt re-emits the open marker by itself, so an empty
            // open thought contributes no bytes either way.
            return;
        }
        self.blocks.push(Block::Thought {
            thought: body.to_string().into(),
            signature: Cow::Borrowed(crate::prompt::OPEN_THOUGHT_SIGNATURE),
        });
    }

    /// Whether the dialect is trigger-less bare JSON, whose call
    /// landmark is any `{` (see [`Leniency::Clipped`]). Scoped to the
    /// family, not to `trigger()` alone: Harmony has no single trigger
    /// either, but its landmarks are all frame markers.
    fn bare_json(&self) -> bool {
        self.syntax.family == Family::JsonNative
            && self.syntax.trigger().is_empty()
    }

    /// The incomplete tail starting at `from`: suppress or degrade per
    /// leniency. Under `Clipped`, a call in flight ([`Self::open`])
    /// comes back cut short.
    fn incomplete(&mut self, from: usize) {
        let withhold = match self.leniency {
            Leniency::Streaming => true,
            Leniency::Final => false,
            Leniency::Clipped => !self.bare_json(),
        };
        if withhold {
            self.status = ParseStatus::NeedMoreInput;
            if self.leniency == Leniency::Clipped {
                let calls = self.open.iter().flat_map(|open| &open.calls);
                let calls: Vec<Block> = calls
                    .map(|call| Block::ToolUse { call: call.clone() })
                    .collect();
                self.blocks.extend(calls);
            }
        } else {
            let tail = self.text[from..].to_string();
            self.push_text(&tail);
        }
        self.pos = self.text.len();
    }

    fn run(&mut self, pre_opened_reasoning: bool) {
        if self.syntax.family == Family::Harmony {
            // Channel-block structure; the generic landmark scan
            // below has no notion of per-block headers. gpt-oss never
            // pre-opens reasoning (the generation prompt ends at
            // `<|start|>assistant`), so that flag is moot here.
            self.run_harmony();
            return;
        }
        let reasoning_on = self.syntax.reasoning.mode != ReasoningMode::None
            && !self.syntax.reasoning.end.is_empty();

        // Pre-opened reasoning: thought body runs to reasoning.end.
        if pre_opened_reasoning && reasoning_on {
            let end = self.syntax.reasoning.end.trim_start();
            match self.rest().find(end) {
                Some(at) => {
                    let body = self.closed_thought_body(&self.rest()[..at]);
                    self.pos += at + end.len();
                    if body.is_empty() {
                        if self.eat_scaffold_separator() {
                            return;
                        }
                    } else {
                        self.push_thought(body);
                    }
                }
                None => {
                    // No reasoning close anywhere. Before treating the
                    // remainder as an unterminated thought, check whether
                    // the model emitted a tool call *inside* the still-open
                    // reasoning block (issue #53): under the lazy trigger
                    // grammar the model may decide to call mid-thought and
                    // never close `</think>`. The call is content, not
                    // reasoning — split the thought at the trigger and let
                    // the main loop parse the call. Restricted to marker
                    // dialects (non-empty trigger); a bare-JSON `{`/`[`
                    // trigger would false-positive on braces in reasoning
                    // prose/code/math. We only reach here with the close
                    // provably absent, so there is no close/trigger
                    // ordering ambiguity.
                    let trigger = self.syntax.trigger();
                    let call_at = (!trigger.is_empty())
                        .then(|| self.rest().find(trigger))
                        .flatten();
                    match call_at {
                        Some(at) => {
                            let body = &self.rest()[..at];
                            self.push_thought(body.trim_end());
                            self.pos += at;
                            // Fall through to the main landmark loop, which
                            // parses the call section from the trigger.
                        }
                        None => {
                            // Entire text so far is thought-in-progress.
                            match self.leniency {
                                Leniency::Streaming => {
                                    self.status = ParseStatus::NeedMoreInput;
                                    self.pos = self.text.len();
                                }
                                Leniency::Final | Leniency::Clipped => {
                                    // Unclosed thought at end of generation:
                                    // surface what we have as an *open*
                                    // Thought — the model was cut off
                                    // mid-reasoning. Raw, untrimmed: these
                                    // bytes must re-render exactly.
                                    let body = self.rest().to_string();
                                    self.push_open_thought(&body);
                                    self.pos = self.text.len();
                                }
                            }
                            return;
                        }
                    }
                }
            }
        }

        let trigger = self.syntax.trigger().to_string();
        let reasoning_start = self.syntax.reasoning.start.trim().to_string();
        // Gemma-style channel noise (TagWithDict): the reasoning open
        // marker has the shape `<open>thought`, and the model may emit
        // a bare `<open>` (an empty/other channel) or an unmatched
        // close as pure noise around content. Both are consumed
        // silently, upstream parity (`consume_empty_channels` and
        // "stop at the first unmatched close" in
        // `common_chat_params_init_gemma4`).
        let channel_open = (self.syntax.family == Family::TagWithDict)
            .then(|| reasoning_start.strip_suffix("thought"))
            .flatten()
            .filter(|s| !s.is_empty())
            .map(str::to_string);
        let channel_close = channel_open
            .is_some()
            .then(|| self.syntax.reasoning.end.trim().to_string())
            .filter(|s| !s.is_empty());
        let exit = self.syntax.tool_response_start.clone();

        while self.pos < self.text.len() {
            // Next structural landmark: reasoning open or call
            // trigger, whichever comes first.
            let think_at = if reasoning_on && !reasoning_start.is_empty() {
                self.find_landmark(&reasoning_start)
            } else {
                None
            };

            let trigger_at = if trigger.is_empty() {
                // Bare JSON-native dialects have no marker trigger:
                // the JSON opener itself is the call landmark. A
                // prose `{` costs a parse attempt that degrades back
                // to Text on failure — same trade upstream makes.
                if self.syntax.family == Family::JsonNative {
                    self.rest().find(['{', '['])
                } else {
                    None
                }
            } else {
                self.find_landmark(&trigger)
            };

            // The turn-exit marker before the next thought / call
            // landmark, consumed below.
            let exit_at = (!exit.is_empty())
                .then(|| self.find_landmark(&exit))
                .flatten()
                .filter(|&p| {
                    think_at.is_none_or(|t| p < t)
                        && trigger_at.is_none_or(|t| p < t)
                });

            // Channel noise strictly before the next thought / call
            // landmark — and before the turn-exit marker, which would
            // otherwise land in the prose before it — is consumed
            // first. A bare open is only *noise* when it is not the
            // thought open itself: the open marker is a prefix of the
            // thought marker, so `open_at ≤ think_at` always, with
            // equality meaning "this IS the thought open".
            if let Some(open) = &channel_open {
                let before_structs = |&p: &usize| {
                    think_at.is_none_or(|t| p < t)
                        && trigger_at.is_none_or(|t| p < t)
                        && exit_at.is_none_or(|e| p < e)
                };
                let open_at = self
                    .find_landmark(open)
                    .filter(|&o| think_at != Some(o))
                    .filter(before_structs);
                let close_at = channel_close
                    .as_deref()
                    .and_then(|c| self.find_landmark(c))
                    .filter(before_structs)
                    .filter(|&c| open_at.is_none_or(|o| c < o));
                let rest = self.rest();
                if let Some(c) = close_at {
                    let prose = rest[..c].to_string();
                    self.push_text(&prose);
                    self.pos += c + channel_close.as_deref().unwrap().len();
                    continue;
                }
                if let Some(o) = open_at {
                    let prose = rest[..o].to_string();
                    self.push_text(&prose);
                    self.pos += o;
                    let after = self.pos + open.len();
                    let tail = &self.text[after..];
                    // The bytes after the open could still grow into
                    // `thought` — don't classify as noise yet.
                    if tail.is_empty() || "thought".starts_with(tail) {
                        self.incomplete(self.pos);
                        return;
                    }
                    self.pos = after;
                    continue;
                }
            }

            // Turn-exit marker (`tool_response_start`, e.g. Gemma's
            // `<|tool_response>`): envelope framing the grammar
            // requires after the last call — swallow it silently
            // wherever it appears outside a call; it is never
            // content.
            if let Some(p) = exit_at {
                let prose = self.rest()[..p].to_string();
                self.push_text(&prose);
                self.pos += p + exit.len();
                continue;
            }

            let rest = self.rest();
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
        // NOT `debug_assert!(self.eat(open))`: `debug_assert!` does not
        // evaluate its argument in release builds, so the marker would
        // go unconsumed and end up inside the thought body — frame
        // bytes seated as content, and a release-only divergence no
        // debug test can see (#62). Consume first, assert second.
        let ate = self.eat(open);
        debug_assert!(ate, "parse_thought: open marker not at self.pos");
        let end = self.syntax.reasoning.end.trim();
        match self.rest().find(end) {
            Some(at) => {
                let body = &self.rest()[..at];
                let body = self.opened_thought_body(body);
                let body = self.closed_thought_body(body);
                self.push_closed_thought(body);
                // Whatever follows the close is the answer's: the baked
                // templates render it verbatim after the marker (Mistral
                // 4 writes `[/THINK]` then the answer; a `\n` there was
                // swallowed and never re-rendered), and a trimming stock
                // template trims it either way.
                self.pos += at + end.len();
            }
            None => {
                // No reasoning close. Did the model emit a tool call
                // inside the still-open reasoning block (issue #53)?
                // Split the thought at the trigger and return to the
                // main loop, which parses the call. Marker dialects only
                // (non-empty trigger); a bare-JSON `{`/`[` would
                // false-positive on prose braces. Reached only with the
                // close provably absent, so no ordering ambiguity.
                let trigger = self.syntax.trigger();
                let call_at = (!trigger.is_empty())
                    .then(|| self.rest().find(trigger))
                    .flatten();
                match call_at {
                    Some(at) => {
                        // `open` was already eaten, so `rest()` is the
                        // body. Strip the leading `\n` as the closed
                        // branch does, but `trim_end` the tail (as the
                        // pre-opened split does) rather than
                        // `closed_thought_body`: the model wrote no
                        // close, so the re-render's close marker is
                        // invented and cannot reproduce the emission
                        // anyway, and whitespace before the trigger is
                        // the gap to the call, not the thought's.
                        let body = &self.rest()[..at];
                        let body = self
                            .opened_thought_body(body)
                            .trim_end()
                            .to_string();
                        self.push_thought(&body);
                        self.pos += at;
                        // Return to the main loop; the trigger dispatches
                        // to parse_calls on the next iteration.
                    }
                    // No close and no call: the model was cut off
                    // mid-reasoning. `open` is already eaten, so
                    // `rest()` is exactly the body — surface it as an
                    // open Thought rather than letting `incomplete`
                    // degrade the whole span (marker bytes included)
                    // into a `Text` block. That degradation is the
                    // thought-half of issue #38: frame markers seated
                    // as content.
                    None => match self.leniency {
                        Leniency::Streaming => {
                            self.status = ParseStatus::NeedMoreInput;
                            self.pos = self.text.len();
                        }
                        Leniency::Final | Leniency::Clipped => {
                            let body = self.rest().to_string();
                            self.push_open_thought(&body);
                            self.pos = self.text.len();
                        }
                    },
                }
            }
        }
    }

    /// Parse the call section beginning at the trigger.
    fn parse_calls(&mut self) {
        let has_section = !self.syntax.section_start.is_empty();
        if has_section {
            // NOT `debug_assert!(self.eat_opener(..))` — see
            // `parse_thought` (#62).
            let ate = self.eat_opener(&self.syntax.section_start.clone());
            debug_assert!(ate, "parse_calls: opener not at self.pos");
        }

        loop {
            // Where *this* call begins. Degradation below is scoped to
            // it, not to `start`: every call parsed so far is already a
            // `ToolUse` block, and degrading from the section start
            // would re-emit them a second time as `Text` — under
            // `Leniency::Final` that rendered a 25-call turn as 50
            // (Mistral's `[TOOL_CALLS]` loop, cut by `max_tokens`
            // mid-call, `session_mistral4::emission_round_trips_…`).
            let call_start = self.pos;
            // Per-call opener (when distinct from the section). For
            // repeat calls it may re-occur after the inter-call
            // whitespace; `eat_opener` tolerates that and any layout
            // drift after the special.
            if !self.eat_opener(&self.syntax.per_call_start.clone()) {
                break;
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
                    if !self
                        .rest()
                        .trim_start()
                        .starts_with(self.syntax.per_call_start.trim())
                    {
                        break;
                    }
                    // Loop continues; opener consumed at loop head.
                }
                CallOutcome::Incomplete => {
                    self.open = self.read_open_call(call_start);
                    self.incomplete(call_start);
                    return;
                }
                CallOutcome::Malformed => {
                    // Degrade this call, from its opener to the next
                    // landmark (or end), into Text — nothing silently
                    // dropped; Session decides severity. Step one
                    // *char* past the opener — a byte step slices
                    // mid-char when a derived trigger opens with a
                    // multi-byte char.
                    let upto = call_start
                        + past_first_char(
                            &self.text[call_start..],
                            self.syntax.trigger(),
                        );
                    // A cut can leave a call malformed-looking (the
                    // clip, not the model, broke it); running to the
                    // end of input it is the call in flight — with
                    // nothing readable to return, withheld.
                    if upto == self.text.len()
                        && self.leniency == Leniency::Clipped
                    {
                        self.incomplete(call_start);
                        return;
                    }
                    let chunk = self.text[call_start..upto].to_string();
                    self.push_text(&chunk);
                    self.pos = upto;
                    return;
                }
            }
        }

        if has_section && !self.syntax.section_end.is_empty() {
            let se = self.syntax.section_end.clone();
            if !self.eat_ws_tolerant(&se) && self.rest().trim().is_empty() {
                // Closer not yet generated. The calls parsed so far
                // stand; only what follows them is incomplete.
                let after_calls = self.pos;
                self.incomplete(after_calls);
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
            Family::TagWithDict => self.parse_dict_call(),
            // Harmony never reaches the generic call loop — `run`
            // dispatches to `run_harmony` before it.
            Family::None | Family::Harmony => CallOutcome::Malformed,
        }
    }

    /// Harmony (gpt-oss): parse the generation as a sequence of
    /// channel blocks. Grammar of a block (parser side — deliberately
    /// looser than the emitted GBNF, upstream stance):
    ///
    /// ```text
    /// [<|start|>assistant] [stray] HEADER <|message|> BODY
    /// stray  = <|channel|>commentary [ to=assistant]   (20b wart,
    ///          only when another <|channel|> header follows)
    /// HEADER = [ to=RECIPIENT] [<|channel|>WORD [ to=RECIPIENT]]
    ///          [ [<|constrain|>]TYPE]
    /// ```
    ///
    /// Dispatch on the parsed header, upstream parity
    /// (`common_chat_params_init_gpt_oss` builder + test corpus):
    /// * recipient `functions.NAME` → tool call, JSON body ended by
    ///   `<|call|>` / end of input;
    /// * any other recipient (`container.exec`, `python`,
    ///   `assistant`) → unsolicited builtin traffic, swallowed whole;
    /// * `analysis` → [`Block::Thought`] (verbatim body — the
    ///   cache-stable re-render is byte-exact), one per block,
    ///   multiple blocks allowed;
    /// * `commentary` (no recipient) → preamble prose →
    ///   [`Block::Text`];
    /// * `final` → [`Block::Text`], body to `<|end|>` or end of
    ///   input (trailing `<|return|>` / `<|call|>` pieces stripped).
    fn run_harmony(&mut self) {
        while self.pos < self.text.len() {
            let block_start = self.pos;
            match self.harmony_block() {
                CallOutcome::Parsed => {}
                CallOutcome::Incomplete => {
                    // Degrade / suppress the whole block: `pos` may
                    // already sit past the header, but a Final-mode
                    // Text fallback must carry the header bytes too
                    // (the dangling-call contract).
                    self.incomplete(block_start);
                    return;
                }
                CallOutcome::Malformed => {
                    // Degrade to prose up to the next possible marker
                    // byte — nothing silently dropped.
                    let upto = block_start
                        + past_first_char(&self.text[block_start..], "<|");
                    // Cut short mid-block: withheld, as a malformed call
                    // running to the end is in the generic loop.
                    if upto == self.text.len()
                        && self.leniency == Leniency::Clipped
                    {
                        self.incomplete(block_start);
                        return;
                    }
                    let chunk = self.text[block_start..upto].to_string();
                    self.push_text(&chunk);
                    self.pos = upto;
                }
            }
        }
    }

    /// One Harmony channel block at `pos`. On `Incomplete` the caller
    /// applies leniency from the *current* `pos` (start of the
    /// still-growing structure).
    fn harmony_block(&mut self) -> CallOutcome {
        // Inter-block opener (absent on the first block: the
        // generation prompt already ends with it).
        if !self.eat(harmony::START_ASSISTANT)
            && harmony::START_ASSISTANT.starts_with(self.rest())
        {
            return CallOutcome::Incomplete;
        }
        let rest = self.rest();
        if rest.is_empty() {
            return CallOutcome::Parsed;
        }

        let header_ish = rest.starts_with(harmony::CHANNEL)
            || rest.starts_with(" to=")
            || harmony::CHANNEL.starts_with(rest)
            || " to=".starts_with(rest);
        if !header_ish {
            // Prose outside any block structure (grammarless model
            // drift): consume up to the next possible marker.
            let upto = past_first_char(rest, "<|");
            let prose = rest[..upto].to_string();
            self.push_text(&prose);
            self.pos += upto;
            return CallOutcome::Parsed;
        }

        let Some(msg_rel) = rest.find(harmony::MESSAGE) else {
            // Header still streaming in (or degenerate junk that will
            // flush as Text at finish).
            return CallOutcome::Incomplete;
        };
        let mut header = &rest[..msg_rel];
        // An `<|end|>` inside the header region means this is not a
        // block header at all.
        if header.contains(harmony::END) {
            return CallOutcome::Malformed;
        }

        // Stray-commentary wart: `<|channel|>commentary[ to=assistant]`
        // immediately followed by the real channel header.
        while let Some(after) = header.strip_prefix(harmony::COMMENTARY) {
            let after = after.strip_prefix(" to=assistant").unwrap_or(after);
            if after.starts_with(harmony::CHANNEL) {
                header = after;
            } else {
                break;
            }
        }

        // Header fields. Owned copies: the borrows point into
        // `self.text` and the body readers below need `&mut self`.
        let mut recipient: Option<String> = None;
        let mut channel: Option<String> = None;
        let mut constrain: Option<String> = None;
        let mut h = header;
        if let Some(r) = h.strip_prefix(" to=") {
            let end = r.find(harmony::CHANNEL).unwrap_or(r.len());
            recipient = Some(r[..end].trim().to_string());
            h = &r[end..];
        }
        if let Some(r) = h.strip_prefix(harmony::CHANNEL) {
            let end = r.find(' ').unwrap_or(r.len());
            channel = Some(r[..end].to_string());
            let tail = &r[end..];
            if let Some(r2) = tail.strip_prefix(" to=") {
                let e2 = r2.find(' ').unwrap_or(r2.len());
                recipient = Some(r2[..e2].trim().to_string());
            }
            // Whatever trails (` [<|constrain|>]TYPE`) is the
            // constraint clause — parsed leniently by ignoring it, bar
            // the content type a final channel declares, which its
            // re-render must spell (see `note_final_constrain`).
            constrain = tail
                .strip_prefix(' ')
                .and_then(|t| t.strip_prefix(harmony::CONSTRAIN))
                .filter(|t| crate::prompt::is_content_type(t))
                .map(str::to_string);
        }

        self.pos += msg_rel + harmony::MESSAGE.len();

        match (recipient, channel.as_deref()) {
            (Some(r), _) if r.starts_with("functions.") => {
                self.harmony_call(r["functions.".len()..].to_string())
            }
            // A declared tool without its namespace (gpt-oss-120b,
            // 2026-10-01: `to=create_comment`) is that tool: swallowed
            // as a builtin, the call was lost without a trace. It
            // re-renders under `functions.`, as the grammar forces it
            // wherever a trigger saw the recipient.
            (Some(r), _) if self.tools.iter().any(|t| t.name == r) => {
                self.harmony_call(r)
            }
            (Some(_), _) => {
                // Builtin / unsolicited recipient: swallow the block
                // (upstream surfaces empty content for these).
                self.harmony_swallow_body();
                CallOutcome::Parsed
            }
            (None, Some("analysis")) => self.harmony_analysis(),
            (None, Some("commentary")) => self.harmony_text_body(false),
            (None, Some("final")) => {
                if let Some(constrain) = constrain {
                    self.note_final_constrain(constrain);
                }
                self.harmony_text_body(true)
            }
            _ => CallOutcome::Malformed,
        }
    }

    /// Analysis body → Thought, verbatim (no trimming: the
    /// cache-stable re-render reproduces these bytes exactly).
    /// Unclosed at end of input: `Final` surfaces the partial as a
    /// Thought (the model was cut off mid-reasoning — upstream pins
    /// the same), `Streaming` suppresses it whole (thought-delta
    /// streaming is #26 territory).
    fn harmony_analysis(&mut self) -> CallOutcome {
        match self.rest().find(harmony::END) {
            Some(at) => {
                let body = self.rest()[..at].to_string();
                self.push_closed_thought(&body);
                self.pos += at + harmony::END.len();
                CallOutcome::Parsed
            }
            None => match self.leniency {
                Leniency::Streaming => CallOutcome::Incomplete,
                Leniency::Final | Leniency::Clipped => {
                    let body = strip_harmony_eog(self.rest()).to_string();
                    // Open: the analysis channel never closed. Harmony
                    // can't *render* an open thought (its generation
                    // prompt never pre-opens a channel — `emit.rs`
                    // collapses both eager anchors), so this turns a
                    // silent re-render with a fabricated `<|end|>` into
                    // a loud rejection at ingest. Stated limitation,
                    // not an oversight.
                    self.push_open_thought(&body);
                    self.pos = self.text.len();
                    CallOutcome::Parsed
                }
            },
        }
    }

    /// Commentary-preamble / final body → Text. An unclosed body is
    /// surfaced as a growing trailing Text block — final content is
    /// the part of a Harmony generation users watch stream, and the
    /// StreamParser's landmark holdback keeps partial markers out of
    /// the deltas. `is_final` only affects trailing-EOG stripping.
    ///
    /// Each channel's body is its own block, never merged into the
    /// prose before it: a preamble (`Hi.`) then a final (`{"a":1}`) are
    /// two answers to the template — commentary, then final — and one
    /// merged `Hi.{"a":1}` was neither the visible answer nor a value
    /// an `output_config` schema could accept.
    fn harmony_text_body(&mut self, is_final: bool) -> CallOutcome {
        let (body, end) = match self.rest().find(harmony::END) {
            Some(at) => {
                (&self.rest()[..at], self.pos + at + harmony::END.len())
            }
            None if is_final => {
                (strip_harmony_eog(self.rest()), self.text.len())
            }
            None => (self.rest(), self.text.len()),
        };
        if !body.is_empty() {
            self.blocks.push(body.to_string().into());
        }
        self.pos = end;
        CallOutcome::Parsed
    }

    /// Record a final channel's declared content type (` <|constrain|>
    /// json`) in the thought right before it (`ThoughtTail::constrain`),
    /// whatever its body — an empty one too, which the template renders
    /// from the thought alone: the header is framing, so its `Text`
    /// cannot say which spelling the model wrote, and gpt-oss writes
    /// either — unforced, ` <|constrain|>json` on structured answers,
    /// plain on prose, and its content shape does not tell them apart
    /// (`[1, 2, 3]` may be prose; a schema whose root is a string is not
    /// `{…}`). With no thought right before it — none at all, or a
    /// preamble between — there is nowhere to record it, and the final
    /// re-renders plain: a stream has already released a thought a
    /// preamble follows, so it could not carry the header after it. The
    /// `output_config` grammar keeps the constraint where it would be
    /// lost (see `output_config::emit_harmony_final_header`).
    fn note_final_constrain(&mut self, constrain: String) {
        use crate::prompt::{ThoughtTail, OPEN_THOUGHT_SIGNATURE};
        if let Some(Block::Thought { signature, .. }) = self.blocks.last_mut() {
            if signature != OPEN_THOUGHT_SIGNATURE {
                let mut tail = ThoughtTail::of(signature);
                tail.constrain = Some(constrain);
                *signature = tail.signature();
            }
        }
    }

    /// Unsolicited builtin block: consumed silently through its
    /// terminator (upstream: matched, content discarded).
    fn harmony_swallow_body(&mut self) {
        let rest = self.rest();
        let end = [harmony::END, harmony::CALL]
            .iter()
            .filter_map(|m| rest.find(m).map(|at| at + m.len()))
            .min()
            .unwrap_or(rest.len());
        self.pos += end;
    }

    /// Tool-call body: one JSON value, optionally terminated by the
    /// `<|call|>` piece (EOG — usually absent from surfaced text).
    fn harmony_call(&mut self, name: String) -> CallOutcome {
        if !is_tool_name(&name) {
            return CallOutcome::Malformed;
        }
        self.skip_ws();
        let body = self.rest();
        let Some((json_len, complete)) = balanced_json_len(body) else {
            return if body.trim().is_empty() {
                self.open = self.open_input(&name, body, Flavor::Json);
                CallOutcome::Incomplete
            } else {
                CallOutcome::Malformed
            };
        };
        if !complete {
            self.open = self.open_input(&name, body, Flavor::Json);
            return CallOutcome::Incomplete;
        }
        let body = &self.rest()[..json_len];
        let Some(input) = parse_json_healed(body) else {
            return CallOutcome::Malformed;
        };
        self.pos += json_len;
        let _ = self.eat(harmony::CALL);
        self.push_call(name, input);
        CallOutcome::Parsed
    }

    fn push_call(&mut self, name: String, input: Value) {
        let call = self.make_call(name, input);
        self.blocks.push(Block::ToolUse { call });
    }

    /// A call with the next id in parse order. Every reader checks the
    /// name first ([`is_tool_name`]): the model's calls are echoed back
    /// as history, and a name or id that ingest rejects would fail
    /// every later request on the transcript. A valid name makes a
    /// valid id.
    fn make_call(&mut self, name: String, input: Value) -> ToolUse {
        debug_assert!(is_tool_name(&name), "unchecked tool name {name:?}");
        let call = ToolUse {
            id: Cow::Owned(format!("call_{}_{}", self.next_id, name)),
            name: Cow::Owned(name),
            input,
            cache_control: None,
            caller: None,
        };
        self.next_id += 1;
        call
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
        if !is_tool_name(&name) {
            return CallOutcome::Malformed;
        }
        self.pos += name_end + f.name_suffix.len();

        let mut args = serde_json::Map::new();
        loop {
            // Progress guard: with a degenerate syntax (empty argument
            // markers and a close that never matches — reachable via
            // the analyzer's TagWithTagged fallback on an odd template,
            // or a user-constructed `CallSyntax`), an iteration can
            // consume zero bytes. Zero progress must be Malformed, not
            // a hang.
            let iter_start = self.pos;
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

            let Some(value) = self.coerce_value(&name, &key, &raw) else {
                return CallOutcome::Malformed;
            };
            args.insert(key, value);

            if !a.separator.is_empty() {
                let _ = self.eat(&a.separator.clone());
            }
            if self.pos == iter_start {
                return CallOutcome::Malformed;
            }
        }

        self.push_call(name, Value::Object(args));
        CallOutcome::Parsed
    }

    /// Schema-guided coercion of a raw tagged value (llama.cpp
    /// mapper parity), spelled as [`tagged_values`] says the grammar
    /// generates it: a string param (or an unknown one) stays raw; a
    /// nullable string too, bar `null`; a finite set of strings is
    /// matched exactly against its members' spellings, and a value
    /// outside it (a model writing unconstrained) is read as JSON or
    /// kept raw, a string beside the set's own; anything else is
    /// parsed as JSON after pythonism normalization with bounded brace
    /// healing.
    ///
    /// `None` when a JSON-spelled value reads as no JSON at all — the
    /// call is malformed. Not the raw text as a string: that retyped a
    /// value the grammar admitted but serde refuses (nested past
    /// `MAX_NESTING` through a recursive `$ref`, a number past `f64`)
    /// into a string, which a union admitting strings even passed the
    /// schema check as.
    ///
    /// Parsing a raw string as JSON would type a `5` or `true` the
    /// model wrote as text, and unquote a `"quoted"` one, which then
    /// re-renders without its quotes.
    ///
    /// [`tagged_values`]: super::emit::tagged_values
    fn coerce_value(
        &self,
        tool: &str,
        param: &str,
        raw: &str,
    ) -> Option<Value> {
        let trimmed = raw.trim();
        let json = || {
            serde_json::from_str::<Value>(trimmed)
                .or_else(|_| serde_json::from_str(&heal_json(trimmed)))
                .ok()
        };
        match self.tagged_value(tool, param) {
            None | Some(TaggedValue::Raw { nullable: false }) => {
                Some(Value::String(raw.to_string()))
            }
            Some(TaggedValue::Raw { nullable: true }) => Some(match trimmed {
                "null" => Value::Null,
                _ => Value::String(raw.to_string()),
            }),
            Some(TaggedValue::Choice(choice)) => {
                // Exact under the grammar; trimmed, or quoted (read as
                // JSON), only from a model writing unconstrained.
                let found = [raw, trimmed].into_iter().find_map(|text| {
                    choice.iter().find(|m| m.spelling == text).map(|m| &m.value)
                });
                let unlisted = || json().unwrap_or_else(|| raw.into());
                Some(found.cloned().unwrap_or_else(unlisted))
            }
            Some(TaggedValue::Json) => json(),
        }
    }

    /// How `param` of `tool` is spelled ([`tagged_values`]); `None`
    /// when the schema does not declare it, which reads as a string.
    ///
    /// [`tagged_values`]: super::emit::tagged_values
    fn tagged_value(&self, tool: &str, param: &str) -> Option<TaggedValue> {
        let index = self.tools.iter().position(|t| t.name.as_ref() == tool)?;
        // A `Choice` is shared, so the clone is a reference count.
        self.spellings
            .borrow_mut()
            .entry(index)
            .or_insert_with(|| {
                Arc::new(
                    tagged_values(self.syntax, &self.tools[index].schema)
                        .into_iter()
                        .collect(),
                )
            })
            .get(param)
            .cloned()
    }

    /// Whether `param` of `tool` takes its raw text as a string, so the
    /// value in flight is one too: a non-nullable string, or unknown.
    fn is_string_param(&self, tool: &str, param: &str) -> bool {
        matches!(
            self.tagged_value(tool, param),
            None | Some(TaggedValue::Raw { nullable: false })
        )
    }

    /// Read the call the input ended inside, starting at its opener
    /// (`call_start`): what a cut leaves of it ([`OpenCall`]). `None`
    /// when its name is incomplete or its input unreadable — withheld —
    /// and under [`Leniency::Final`], which degrades it instead.
    fn read_open_call(&mut self, call_start: usize) -> Option<OpenCall> {
        if self.leniency == Leniency::Final || self.bare_json() {
            return None;
        }
        // `Copy` the borrow out of `self`, so the slices outlive `&mut`.
        let text: &'a str = &self.text[call_start..];
        // Whitespace-tolerant, as `eat_opener` reads it.
        let opener = self.syntax.per_call_start.trim();
        let text = match opener {
            "" => text,
            _ => text.trim_start().strip_prefix(opener)?.trim_start(),
        };
        let f = &self.syntax.function;
        match self.syntax.family {
            Family::TagWithTagged => self.open_tagged(text),
            Family::TagWithJson => {
                let (name, args) = named(text, &f.name_prefix, &f.name_suffix)?;
                self.open_input(name, args, Flavor::Json)
            }
            Family::TagWithDict => {
                let args_open = match self.syntax.arguments.start.as_str() {
                    "" => "{",
                    start => start,
                };
                let (name, _) = named(text, &f.name_prefix, args_open)?;
                let args = &text[f.name_prefix.len() + name.len()..];
                let quote = self.syntax.arguments.string_quote.clone();
                self.open_input(name, args, Flavor::Dict { quote: &quote })
            }
            Family::JsonNative => self.open_json_native(text),
            Family::None | Family::Harmony => None,
        }
    }

    /// A call with its name read and its input `args` in flight — JSON
    /// or Gemma's dict, empty when not begun.
    fn open_input(
        &mut self,
        name: &str,
        args: &str,
        flavor: Flavor<'_>,
    ) -> Option<OpenCall> {
        if self.leniency == Leniency::Final {
            return None;
        }
        let empty = || Value::Object(serde_json::Map::new());
        let (input, held_input, raw_input, open) = if args.trim().is_empty() {
            (empty(), empty(), empty(), 1)
        } else {
            let read = |strings| read_partial(args, flavor, strings);
            let kept =
                read(OpenStrings::Drop).filter(|t| t.value.is_object())?;
            (
                kept.value,
                read(OpenStrings::Held)?.value,
                read(OpenStrings::Raw)?.value,
                kept.open,
            )
        };
        let partial_json = Some(unclosed_json(&input, open));
        let call = self.make_call(name.to_owned(), input);
        Some(OpenCall {
            calls: vec![call],
            held_input,
            raw_input,
            partial_json,
        })
    }

    /// TAG_WITH_TAGGED (Qwen XML): the parameters whose close marker
    /// arrived, coerced as a closed call's are; the one in flight kept
    /// (held, raw) only when it is a string parameter.
    fn open_tagged(&mut self, text: &'a str) -> Option<OpenCall> {
        let f = &self.syntax.function;
        let a = self.syntax.arguments.clone();
        let (name, mut rest) = named(text, &f.name_prefix, &f.name_suffix)?;
        let mut members = serde_json::Map::new();
        let mut in_flight: Option<(&str, &str)> = None;
        loop {
            let before = rest.len();
            let Some(after) = rest.strip_prefix(a.name_prefix.as_str()) else {
                break;
            };
            let Some((key, after)) = split_marker(after, &a.name_suffix) else {
                break;
            };
            let Some(value) = after.strip_prefix(a.value_prefix.as_str())
            else {
                break;
            };
            let Some((raw, after)) = split_marker(value, &a.value_suffix)
            else {
                if self.is_string_param(name, key) {
                    in_flight = Some((key, value));
                }
                break;
            };
            // A member that closed unreadable makes the call malformed,
            // as it will be once it closes: nothing to show for it.
            members.insert(key.to_owned(), self.coerce_value(name, key, raw)?);
            rest = after.strip_prefix(a.separator.as_str()).unwrap_or(after);
            // Progress guard: degenerate markers can match nothing.
            if rest.len() == before {
                break;
            }
        }
        let with = |value: Option<&str>| {
            let mut members = members.clone();
            if let (Some((key, _)), Some(value)) = (in_flight, value) {
                members.insert(key.to_owned(), Value::String(value.into()));
            }
            Value::Object(members)
        };
        let raw = in_flight.map(|(_, v)| v);
        let held =
            raw.map(|v| &v[..v.len() - marker_holdback(v, &a.value_suffix)]);
        let (held_input, raw_input) = (with(held), with(raw));
        let input = Value::Object(members);
        let partial_json = Some(unclosed_json(&input, 1));
        let call = self.make_call(name.to_owned(), input);
        Some(OpenCall {
            calls: vec![call],
            held_input,
            raw_input,
            partial_json,
        })
    }

    /// JSON_NATIVE with a trigger (Hermes): the envelope in flight,
    /// `{"name": …, "arguments": {…}}` — or an array of them, whose
    /// completed elements come back whole.
    fn open_json_native(&mut self, text: &str) -> Option<OpenCall> {
        let read = |strings| read_partial(text, Flavor::Json, strings);
        let kept = read(OpenStrings::Drop)?;
        let (held, raw) = (
            read(OpenStrings::Held)?.value,
            read(OpenStrings::Raw)?.value,
        );
        let Value::Array(envelopes) = kept.value else {
            let (name, input, open) =
                self.open_envelope(&kept.value, kept.open, &kept.path)?;
            return Some(self.open_native_call(
                Vec::new(),
                name,
                input,
                open,
                &held,
                &raw,
            ));
        };
        // Array-wrapped: the last element is in flight when a container
        // inside it is still open.
        let in_flight = kept.open >= 2;
        let closed = envelopes.len().checked_sub(usize::from(in_flight))?;
        let calls = envelopes[..closed]
            .iter()
            .map(|env| self.map_json_call(env))
            .collect::<Option<Vec<_>>>()?;
        let calls: Vec<ToolUse> = calls
            .into_iter()
            .map(|(name, input)| self.make_call(name, input))
            .collect();
        let at =
            |value: &Value| value.get(closed).cloned().unwrap_or(Value::Null);
        let envelope = in_flight
            .then(|| {
                self.open_envelope(
                    &envelopes[closed],
                    kept.open - 1,
                    &kept.path[1..],
                )
            })
            .flatten();
        if let Some((name, input, open)) = envelope {
            return Some(self.open_native_call(
                calls,
                name,
                input,
                open,
                &at(&held),
                &at(&raw),
            ));
        }
        // Cut between calls, or before the next one's name is whole:
        // the calls that closed stand, and nothing is open.
        let last = calls.last()?.input.clone();
        Some(OpenCall {
            calls,
            held_input: last.clone(),
            raw_input: last,
            partial_json: None,
        })
    }

    /// One envelope in flight: its name (which must have completed),
    /// its arguments so far, and how many of their containers are open.
    fn open_envelope(
        &self,
        envelope: &Value,
        open: usize,
        path: &[String],
    ) -> Option<(String, Value, usize)> {
        let (name, input) = self.map_json_call(envelope)?;
        let args_field = leaf(&self.syntax.json.args_field, "arguments");
        let function_field = &self.syntax.json.function_field;
        // The keys leading from the envelope to its arguments.
        let to_args: Vec<&str> = if self.syntax.json.fun_name_is_key {
            vec![name.as_str()]
        } else if !function_field.is_empty()
            && envelope.get(function_field).is_some_and(Value::is_object)
        {
            vec![function_field.as_str(), args_field]
        } else {
            vec![args_field]
        };
        let begun = to_args
            .iter()
            .try_fold(envelope, |v, key| v.get(key))
            .is_some();
        let open = if !begun {
            // Not begun: `{}`, still to come.
            1
        } else if path.len() >= to_args.len()
            && path.iter().zip(&to_args).all(|(p, k)| p == k)
        {
            open - to_args.len()
        } else {
            0
        };
        Some((name, input, open))
    }

    /// The [`OpenCall`] for an envelope read by [`Self::open_envelope`];
    /// `held`/`raw` are the same envelope read with its string in
    /// flight kept.
    fn open_native_call(
        &mut self,
        mut calls: Vec<ToolUse>,
        name: String,
        input: Value,
        open: usize,
        held: &Value,
        raw: &Value,
    ) -> OpenCall {
        // The arguments with the string in flight — unless it is not
        // arguments at all (double-encoded, say): then as they stand.
        let args = |envelope: &Value| {
            self.map_json_call(envelope)
                .map(|(_, args)| args)
                .filter(Value::is_object)
                .unwrap_or_else(|| input.clone())
        };
        let (held_input, raw_input) = (args(held), args(raw));
        let partial_json = Some(unclosed_json(&input, open));
        calls.push(self.make_call(name, input));
        OpenCall {
            calls,
            held_input,
            raw_input,
            partial_json,
        }
    }

    /// TAG_WITH_DICT (Gemma 4): `call:name{key:value,…}` after the
    /// per-call opener. Values are dict-encoded (bare keys, quote-
    /// marked strings); the recursive reader coerces them straight to
    /// JSON.
    fn parse_dict_call(&mut self) -> CallOutcome {
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
        let args_open = if self.syntax.arguments.start.is_empty() {
            "{"
        } else {
            &self.syntax.arguments.start
        }
        .to_string();
        let Some(name_end) = self.rest().find(&args_open) else {
            return if self.rest().len() > 256 {
                CallOutcome::Malformed
            } else {
                CallOutcome::Incomplete
            };
        };
        let name = self.rest()[..name_end].to_string();
        if !is_tool_name(&name) {
            return CallOutcome::Malformed;
        }
        // Leave the opening brace for the value reader.
        self.pos += name_end;

        match self.parse_dict_value(0) {
            DictOutcome::Value(input) => {
                self.push_call(name, input);
                CallOutcome::Parsed
            }
            DictOutcome::Incomplete => CallOutcome::Incomplete,
            DictOutcome::Malformed => CallOutcome::Malformed,
        }
    }

    /// Read one dict-encoded value at `pos` (whitespace-lenient like
    /// upstream's PEG; canonical output is compact), inside `depth`
    /// containers. A container past [`MAX_NESTING`] is malformed — as
    /// serde_json refuses one in the JSON dialects — rather than a
    /// recursion as deep as the model's brackets.
    fn parse_dict_value(&mut self, depth: usize) -> DictOutcome {
        self.skip_ws();
        let quote = self.syntax.arguments.string_quote.clone();
        let rest = self.rest();

        if rest.is_empty() {
            return DictOutcome::Incomplete;
        }
        if !quote.is_empty()
            && (rest.starts_with(&quote) || quote.starts_with(rest))
        {
            if !self.eat(&quote) {
                return DictOutcome::Incomplete;
            }
            let Some(end) = self.rest().find(&quote) else {
                return DictOutcome::Incomplete;
            };
            let s = self.rest()[..end].to_string();
            self.pos += end + quote.len();
            return DictOutcome::Value(Value::String(s));
        }
        match rest.as_bytes()[0] {
            b'{' | b'[' if depth >= MAX_NESTING => DictOutcome::Malformed,
            b'{' => self.parse_dict_object(depth + 1),
            b'[' => self.parse_dict_array(depth + 1),
            _ => self.parse_dict_scalar(),
        }
    }

    /// The object at `pos`, itself the `depth`th container.
    fn parse_dict_object(&mut self, depth: usize) -> DictOutcome {
        // Consume first, assert second — see `parse_thought` (#62).
        let ate = self.eat("{");
        debug_assert!(ate, "parse_dict_object: `{{` not at self.pos");
        let mut map = serde_json::Map::new();
        loop {
            self.skip_ws();
            if self.eat("}") {
                return DictOutcome::Value(Value::Object(map));
            }
            if self.rest().is_empty() {
                return DictOutcome::Incomplete;
            }
            // Bare key up to `:` (grammar parity: `[^:}]+`).
            let Some(colon) = self.rest().find([':', '}']) else {
                return DictOutcome::Incomplete;
            };
            if self.rest().as_bytes()[colon] == b'}' {
                return DictOutcome::Malformed;
            }
            let key = self.rest()[..colon].trim().to_string();
            if key.is_empty() {
                return DictOutcome::Malformed;
            }
            self.pos += colon + 1;
            let value = match self.parse_dict_value(depth) {
                DictOutcome::Value(v) => v,
                other => return other,
            };
            map.insert(key, value);
            self.skip_ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("}") {
                return DictOutcome::Value(Value::Object(map));
            }
            return if self.rest().is_empty() {
                DictOutcome::Incomplete
            } else {
                DictOutcome::Malformed
            };
        }
    }

    /// The array at `pos`, itself the `depth`th container.
    fn parse_dict_array(&mut self, depth: usize) -> DictOutcome {
        // Consume first, assert second — see `parse_thought` (#62).
        let ate = self.eat("[");
        debug_assert!(ate, "parse_dict_array: `[` not at self.pos");
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.eat("]") {
                return DictOutcome::Value(Value::Array(items));
            }
            if self.rest().is_empty() {
                return DictOutcome::Incomplete;
            }
            let value = match self.parse_dict_value(depth) {
                DictOutcome::Value(v) => v,
                other => return other,
            };
            items.push(value);
            self.skip_ws();
            if self.eat(",") {
                continue;
            }
            if self.eat("]") {
                return DictOutcome::Value(Value::Array(items));
            }
            return if self.rest().is_empty() {
                DictOutcome::Incomplete
            } else {
                DictOutcome::Malformed
            };
        }
    }

    /// Literals and numbers. `none`/`None` are minijinja/pythonic
    /// null spellings (our re-render canonicalizes to `none`).
    fn parse_dict_scalar(&mut self) -> DictOutcome {
        let rest = self.rest();
        let boundary = |s: &str| {
            !s.chars()
                .next()
                .is_some_and(|c| c.is_ascii_alphanumeric() || c == '_')
        };
        for (lit, value) in [
            ("true", Value::Bool(true)),
            ("false", Value::Bool(false)),
            ("null", Value::Null),
            ("none", Value::Null),
            ("None", Value::Null),
        ] {
            if let Some(after) = rest.strip_prefix(lit) {
                if boundary(after) {
                    self.pos += lit.len();
                    return DictOutcome::Value(value);
                }
            }
            // Trailing partial literal: could still complete.
            if lit.starts_with(rest) {
                return DictOutcome::Incomplete;
            }
        }
        let len = rest
            .find(|c: char| {
                !matches!(c, '0'..='9' | '-' | '+' | '.' | 'e' | 'E')
            })
            .unwrap_or(rest.len());
        if len == 0 {
            return DictOutcome::Malformed;
        }
        // A number at end-of-input could still grow more digits — and
        // on a clip, would have: the cut is not the number's end.
        if len == rest.len() && self.leniency != Leniency::Final {
            return DictOutcome::Incomplete;
        }
        match serde_json::from_str::<Value>(&rest[..len]) {
            Ok(v @ Value::Number(_)) => {
                self.pos += len;
                DictOutcome::Value(v)
            }
            _ => DictOutcome::Malformed,
        }
    }

    fn skip_ws(&mut self) {
        let ws = self.rest().len() - self.rest().trim_start().len();
        self.pos += ws;
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
        if !is_tool_name(&name) {
            return CallOutcome::Malformed;
        }
        self.pos += name_end + f.name_suffix.len();

        // Nothing past the name yet (`[TOOL_CALLS]name[ARGS]` and a
        // cut): the arguments are still coming, not malformed.
        if self.rest().trim().is_empty() {
            return CallOutcome::Incomplete;
        }
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
    /// and the name-is-key shape. `None` for a name ingest would
    /// reject (see [`is_tool_name`]).
    fn map_json_call(&self, call: &Value) -> Option<(String, Value)> {
        let obj = call.as_object()?;
        if self.syntax.json.fun_name_is_key {
            let (name, args) = obj.iter().next()?;
            return is_tool_name(name).then(|| (name.clone(), args.clone()));
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
            .and_then(|v| v.as_str())
            .filter(|name| is_tool_name(name))?
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

/// `text` past `prefix`, split at `suffix` into a call's name and what
/// follows it (see [`split_marker`]). `None` until the name is whole
/// (its suffix arrived) and for a name ingest would reject (see
/// [`is_tool_name`]).
fn named<'t>(
    text: &'t str,
    prefix: &str,
    suffix: &str,
) -> Option<(&'t str, &'t str)> {
    let (name, rest) = split_marker(text.strip_prefix(prefix)?, suffix)?;
    is_tool_name(name).then_some((name, rest))
}

/// Where `marker` next occurs in `text` past its first char — model
/// output, so that char may be any width — or `text.len()`. What a
/// scan that must make progress resumes from.
fn past_first_char(text: &str, marker: &str) -> usize {
    let first = text.chars().next().map_or(0, char::len_utf8);
    text[first..]
        .find(marker)
        .map_or(text.len(), |at| first + at)
}

/// Split `text` at the first `marker`: what precedes it and what
/// follows. A marker whose trailing whitespace the input has not
/// reached yet (`">"` of `">\n"`, at the end) counts as arrived — that
/// whitespace is framing, not content.
fn split_marker<'t>(text: &'t str, marker: &str) -> Option<(&'t str, &'t str)> {
    if let Some(at) = text.find(marker) {
        return Some((&text[..at], &text[at + marker.len()..]));
    }
    let core = marker.trim_end();
    let at = text.rfind(core).filter(|_| !core.is_empty())?;
    let tail = &text[at + core.len()..];
    marker[core.len()..]
        .starts_with(tail)
        .then(|| (&text[..at], ""))
}

enum CallOutcome {
    Parsed,
    Incomplete,
    Malformed,
}

/// Strip a trailing Harmony EOG piece from body text. Whether the
/// predictor surfaces the stop token's bytes depends on the path
/// (`trim_eos` covers the whole EOG set — `<|return|>` and `<|call|>`
/// alike), so the parser tolerates both. `<|end|>` is deliberately not
/// here: it is in-stream structure, not a terminator.
fn strip_harmony_eog(s: &str) -> &str {
    let s = s.strip_suffix(harmony::RETURN).unwrap_or(s);
    s.strip_suffix(harmony::CALL).unwrap_or(s)
}

/// Outcome of reading one dict-encoded value.
enum DictOutcome {
    Value(Value),
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
            // Copy whole chars — pushing a raw UTF-8 byte `as char`
            // reinterprets it as Latin-1 and mojibakes non-ASCII
            // content into *valid* JSON serde then accepts. The state
            // bytes (`\`, `"`) are ASCII, so testing the lead byte is
            // enough.
            let ch_len =
                body[i..].chars().next().map(char::len_utf8).unwrap_or(1);
            out.push_str(&body[i..i + ch_len]);
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                in_string = false;
            }
            i += ch_len;
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
                        // `\` is ASCII, so i + 1 is a char boundary;
                        // copy the escaped char whole (it may be
                        // multi-byte).
                        let ch_len = body[i + 1..]
                            .chars()
                            .next()
                            .map(char::len_utf8)
                            .unwrap_or(1);
                        out.push('\\');
                        out.push_str(&body[i + 1..i + 1 + ch_len]);
                        i += 1 + ch_len;
                        continue;
                    }
                    if c == b'\'' {
                        i += 1;
                        break;
                    }
                    if c == b'"' {
                        out.push_str("\\\"");
                        i += 1;
                    } else {
                        // Whole chars, as above.
                        let ch_len = body[i..]
                            .chars()
                            .next()
                            .map(char::len_utf8)
                            .unwrap_or(1);
                        out.push_str(&body[i..i + ch_len]);
                        i += ch_len;
                    }
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
    use serde_json::json;

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

    /// The grammar admits exactly what `render_reference` emits.
    ///
    /// This is the invariant #85 fell through: `render_reference`'s own
    /// doc comment promised "the exact bytes ... `grammar_source`'s
    /// grammar forces", but nothing checked the two against each other.
    /// The grammar's `ws` was permissive, so the model could emit `": "`
    /// where the serializer emits `":"`, the re-render was not
    /// byte-identical to the KV, and the prefix cache's auto-tip was
    /// discarded — 4705 tokens per turn against cogito-32b.
    ///
    /// Uses a payload with an apostrophe and `&` on purpose: those are
    /// what Jinja's HTML-safe `tojson` used to escape, and clean-ASCII
    /// payloads are why this went unnoticed for so long.
    #[test]
    fn canonical_call_grammar_admits_render_reference() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        use crate::{Grammar, GrammarState};
        use std::sync::Arc;

        let input = serde_json::json!({
            "city": "x's cafe & bar",
            "days": 3,
        });
        let t = tool("get_weather");
        for syntax in [
            CallSyntax::hermes_json(),
            CallSyntax::llama31_json(),
            CallSyntax::qwen_xml(),
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
                    ..Default::default()
                },
            )
            .expect("emit");
            let grammar = Arc::new(
                Grammar::parse(&src)
                    .unwrap_or_else(|e| panic!("grammar: {e}\n{src}")),
            );
            let mut state = GrammarState::new(grammar);
            assert!(
                state.advance_bytes(emission.as_bytes()).is_ok()
                    && state.is_complete(),
                "{:?}: grammar rejects its own reference render \
                 {emission:?}\n{src}",
                syntax.family,
            );
        }
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
            CallSyntax::gemma4(),
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

    /// A call whose name ingest would reject (Anthropic's
    /// `^[a-zA-Z0-9_-]{1,64}$`) degrades to text in every family, whole
    /// or cut: seated as a `ToolUse`, its name and its `call_{n}_{name}`
    /// id would fail every later request on the transcript once the
    /// client echoed it back.
    #[test]
    fn invalid_tool_name_degrades_instead_of_seating() {
        let input = serde_json::json!({"city": "Paris", "days": 3});
        let t = tool("get_weather");
        let long = "x".repeat(65);
        for syntax in [
            CallSyntax::qwen_xml(),
            CallSyntax::hermes_json(),
            CallSyntax::llama31_json(),
            CallSyntax::gemma4(),
        ] {
            let emission =
                render_reference(&syntax, &[("get_weather", &input)])
                    .expect("representable");
            for bad in ["get.weather", "get weather", long.as_str()] {
                let text = emission.replace("get_weather", bad);
                for leniency in [Leniency::Final, Leniency::Clipped] {
                    let parsed =
                        parse_text(&syntax, &[&t], &text, false, leniency);
                    assert!(
                        calls_of(&parsed.blocks).is_empty(),
                        "{:?} {leniency:?}: {text:?} → {:#?}",
                        syntax.family,
                        parsed.blocks,
                    );
                }
                // Cut mid-arguments: the open call is withheld too.
                let cut = &text[..text.find("Paris").expect("city")];
                let parsed =
                    parse_text(&syntax, &[&t], cut, false, Leniency::Clipped);
                assert!(
                    calls_of(&parsed.blocks).is_empty(),
                    "{:?} cut: {cut:?} → {:#?}",
                    syntax.family,
                    parsed.blocks,
                );
            }
            // Final keeps the bytes as text: nothing silently dropped.
            let text = emission.replace("get_weather", "get.weather");
            let parsed =
                parse_text(&syntax, &[&t], &text, false, Leniency::Final);
            assert!(
                parsed.blocks.iter().any(|b| matches!(
                    b,
                    Block::Text { text, .. } if text.contains("get.weather")
                )),
                "{:?}: {:#?}",
                syntax.family,
                parsed.blocks,
            );
        }

        // Harmony keeps whatever follows `functions.`, to the space.
        for recipient in ["functions.get.weather", "functions.a:b"] {
            let text = format!(
                "<|channel|>commentary to={recipient} \
                 <|constrain|>json<|message|>{{\"arg1\": 1}}<|call|>"
            );
            let blocks = harmony_parse(&text, Leniency::Final);
            assert!(calls_of(&blocks).is_empty(), "{text:?}: {blocks:#?}");
        }
    }

    /// A multi-call section whose *last* call is cut off (the model ran
    /// out of `max_tokens` mid-JSON — Mistral's `[TOOL_CALLS]` loop is
    /// the production case) degrades only that call to `Text`. The
    /// complete calls before it are `ToolUse` blocks and must not be
    /// re-emitted inside the degraded text: degrading from the
    /// *section* start did exactly that, and a 25-call turn re-rendered
    /// as 50 (`session_mistral4::emission_round_trips_…`). The
    /// truncated call itself is unrecoverable (half a JSON object has
    /// no representation) and stays text — `Session` contains it.
    #[test]
    fn truncated_last_call_does_not_duplicate_the_complete_ones() {
        let t = tool("get_weather");
        for syntax in [CallSyntax::qwen_xml(), CallSyntax::gemma4()] {
            let a = serde_json::json!({"city": "Paris", "days": 1});
            let b = serde_json::json!({"city": "Oslo", "days": 2});
            let complete = render_reference(
                &syntax,
                &[("get_weather", &a), ("get_weather", &b)],
            )
            .expect("representable");
            // A third call, cut mid-arguments: take the two-call render
            // and append the third's opener plus a byte-truncated
            // prefix of a call body.
            let third = render_reference(&syntax, &[("get_weather", &b)])
                .expect("representable");
            let cut = &third[..third.len() * 2 / 3];
            let emission = format!("{complete}{cut}");

            let parsed =
                parse_text(&syntax, &[&t], &emission, false, Leniency::Final);
            let calls = calls_of(&parsed.blocks);
            assert_eq!(
                calls.len(),
                2,
                "{:?}: {emission:?} → {:#?}",
                syntax.family,
                parsed.blocks
            );
            assert_eq!(calls[0].1, &a);
            assert_eq!(calls[1].1, &b);
            let texts: Vec<&str> = parsed
                .blocks
                .iter()
                .filter_map(|b| match b {
                    Block::Text { text, .. } => Some(text.as_ref()),
                    _ => None,
                })
                .collect();
            // Exactly one degraded block, holding the cut call and
            // nothing of the two that parsed.
            assert_eq!(texts.len(), 1, "{:?}: {:#?}", syntax.family, texts);
            assert!(
                !texts[0].contains("Paris"),
                "{:?}: degraded text re-emits a parsed call: {:?}",
                syntax.family,
                texts[0]
            );
            assert!(
                emission.ends_with(texts[0]),
                "{:?}: degraded text is not the emission's tail: {:?}",
                syntax.family,
                texts[0]
            );
        }
    }

    /// Whether `tool`'s Qwen XML grammar admits `emission` whole.
    /// Unmeasured ([`crate::SchemaLimits::unlimited`]): the callers
    /// are about the grammar, some of them past the limits on purpose.
    fn qwen_admits(tool: &Tool, emission: &str) -> bool {
        let source = crate::dialect::grammar_source(
            &CallSyntax::qwen_xml(),
            &[tool],
            &crate::dialect::EmitOptions {
                schema_limits: crate::SchemaLimits::unlimited(),
                ..Default::default()
            },
        )
        .expect("grammar");
        let mut state =
            crate::GrammarState::from_source(&source).expect("grammar parses");
        state.advance_bytes(emission.as_bytes()).is_ok() && state.is_complete()
    }

    /// A one-parameter Qwen XML call to `set_mode` with `raw` between
    /// the tags.
    fn qwen_mode_call(raw: &str) -> String {
        format!(
            "<tool_call>\n<function=set_mode>\n\
             <parameter=mode>\n{raw}\n</parameter>\n\
             </function>\n</tool_call>"
        )
    }

    /// The streaming parser re-parses the whole generation on every
    /// token; each tool's spellings are classified once for all of
    /// them, not once a token.
    #[test]
    fn stream_parser_classifies_each_tool_once() {
        let tool = mode_tool(json!({"enum": ["fast", "slow"]}), None);
        let mut parser =
            StreamParser::new(CallSyntax::qwen_xml(), vec![tool], false);
        let call = qwen_mode_call("fast");
        let mut first: Option<Arc<HashMap<String, TaggedValue>>> = None;
        let mut blocks = Vec::new();
        for c in call.chars() {
            blocks.extend(parser.push(&c.to_string()));
            if let Some(spellings) = parser.spellings.get(&0) {
                let first = first.get_or_insert_with(|| spellings.clone());
                assert!(Arc::ptr_eq(first, spellings), "reclassified");
            }
        }
        blocks.extend(parser.finish());
        assert!(first.is_some(), "never classified");
        let calls = calls_of(&blocks);
        assert_eq!(calls[0].1, &json!({"mode": "fast"}));
    }

    /// On Qwen XML a parameter the schema does not make nullable can
    /// never come back JSON `null`, and one it does takes a bare
    /// `null`. A non-string type is written as JSON and its grammar has
    /// no `null` at all; a plain string is written raw, so the text
    /// `null` is admitted and reads back as the *string* `"null"`. A
    /// nullable string, a `["integer", "null"]` and an `anyOf` with a
    /// `null` variant (Agora's `Option<DetailLevel>` shape) each take
    /// a bare `null` as JSON `null` — and so does a `$ref` to a nullable
    /// def, as its inline form does.
    #[test]
    fn qwen_xml_null_only_where_the_schema_allows_it() {
        let schema = json!({
            "type": "object",
            "properties": {
                "i": {"type": "integer"},
                "n": {"type": "number"},
                "b": {"type": "boolean"},
                "o": {"type": "object", "properties": {"x": {"type": "integer"}}},
                "a": {"type": "array", "items": {"type": "integer"}},
                "e": {"enum": ["fast", "slow"]},
                "s": {"type": "string"},
                "ns": {"type": ["string", "null"]},
                "ni": {"type": ["integer", "null"]},
                "an": {"anyOf": [
                    {"oneOf": [{"const": "summary"}, {"const": "full"}]},
                    {"type": "null"},
                ]},
                "rn": {"$ref": "#/$defs/N"},
                "ra": {"$ref": "#/$defs/Alias"},
                "ri": {"$ref": "#/$defs/I"},
            },
            "$defs": {
                "N": {"type": ["integer", "null"]},
                "Alias": {"$ref": "#/$defs/N"},
                "I": {"type": "integer"},
            },
        });
        let mut tool = Tool::builder("t")
            .description("test")
            .schema(schema)
            .build()
            .expect("valid tool");
        tool.strict = Some(true);
        let call = |param: &str, raw: &str| {
            format!(
                "<tool_call>\n<function=t>\n<parameter={param}>\n{raw}\n\
                 </parameter>\n</function>\n</tool_call>"
            )
        };
        let read = |param: &str, raw: &str| -> Value {
            let text = call(param, raw);
            assert!(qwen_admits(&tool, &text), "{param} = {raw} refused");
            let parsed = parse_text(
                &CallSyntax::qwen_xml(),
                &[&tool],
                &text,
                false,
                Leniency::Final,
            );
            calls_of(&parsed.blocks)[0].1[param].clone()
        };
        // Non-nullable, not a string: no `null` in the grammar, though
        // a value of the type is admitted.
        for (param, valid) in [
            ("i", "5"),
            ("n", "1.5"),
            ("b", "true"),
            ("o", r#"{"x":1}"#),
            ("a", "[1,2]"),
            ("e", "fast"),
            ("ri", "3"),
        ] {
            assert_ne!(read(param, valid), Value::Null, "{param}");
            assert!(!qwen_admits(&tool, &call(param, "null")), "{param}");
        }
        // A plain string: the text `null` is that string.
        assert_eq!(read("s", "null"), json!("null"));
        // Nullable: a bare `null` is null, and a value is still a value.
        assert_eq!(read("ns", "null"), Value::Null);
        assert_eq!(read("ns", "text"), json!("text"));
        assert_eq!(read("an", "null"), Value::Null);
        assert_eq!(read("an", "full"), json!("full"));
        assert_eq!(read("ni", "7"), json!(7));
        assert_eq!(read("ni", "null"), Value::Null);
        // Through a `$ref` (an alias chain too), as inline.
        for param in ["rn", "ra"] {
            assert_eq!(read(param, "7"), json!(7), "{param}");
            assert_eq!(read(param, "null"), Value::Null, "{param}");
        }
    }

    /// A stop cut parses the output prefix after prefix, each a fresh
    /// parse; lent one [`Spellings`] (as `Session` does, per call) they
    /// classify each tool once between them.
    #[test]
    fn cached_parses_share_one_classification() {
        let tool = mode_tool(json!({"enum": ["fast", "slow"]}), None);
        let call = qwen_mode_call("fast");
        let mut spellings = Spellings::new();
        let mut first: Option<Arc<HashMap<String, TaggedValue>>> = None;
        for end in (1..=call.len()).filter(|&i| call.is_char_boundary(i)) {
            let (parsed, _) = parse_text_cached(
                &CallSyntax::qwen_xml(),
                &[&tool],
                &call[..end],
                false,
                Leniency::Clipped,
                &mut spellings,
            );
            if let Some(s) = spellings.get(&0) {
                let first = first.get_or_insert_with(|| s.clone());
                assert!(Arc::ptr_eq(first, s), "reclassified at {end}");
            }
            if end == call.len() {
                assert_eq!(
                    calls_of(&parsed.blocks)[0].1,
                    &json!({"mode": "fast"})
                );
            }
        }
        assert!(first.is_some(), "never classified");
    }

    /// A strict `set_mode` tool whose one required parameter is `mode`.
    fn mode_tool(mode: Value, defs: Option<Value>) -> Tool {
        let mut schema = json!({
            "type": "object",
            "properties": {"mode": mode},
            "required": ["mode"],
        });
        if let Some(defs) = defs {
            schema["$defs"] = defs;
        }
        let mut tool = Tool::builder("set_mode")
            .description("test")
            .schema(schema)
            .build()
            .expect("valid test tool");
        tool.strict = Some(true);
        tool
    }

    /// The Agora `get_content` `detail` parameter, as schemars 1.x emits
    /// `Option<DetailLevel>` for an `inline` enum whose variants carry
    /// docs (descriptions shortened): a `oneOf` of `const`s, made
    /// nullable by an outer `anyOf`.
    fn agora_detail() -> Value {
        json!({
            "description": "How much to return.",
            "anyOf": [
                {
                    "description": "How much of a piece of content.",
                    "oneOf": [
                        {
                            "description": "The short form.",
                            "type": "string",
                            "const": "summary",
                        },
                        {
                            "description": "The verbatim record.",
                            "type": "string",
                            "const": "full",
                        },
                    ],
                },
                {"type": "null"},
            ],
        })
    }

    /// A finite set of strings on Qwen XML is generated raw, the way
    /// the template re-renders it (`args_value | string`): the model
    /// was trained on `<parameter=detail>\nfull\n</parameter>`. Quoted,
    /// as the grammar once forced it, the value re-rendered without its
    /// quotes and the turn missed the tip on every such call (seen live
    /// on Agora: `detail` = `"summary"`). Every shape a finite string
    /// set reaches the grammar in — `enum`, `const`, nullable, through
    /// `$ref`, `anyOf` and schemars' `oneOf` — admits the raw member
    /// and no other spelling, parses to the member, passes the strict
    /// schema backstop, and renders back byte for byte.
    #[test]
    fn qwen_xml_string_set_is_generated_and_read_raw() {
        let level = Some(
            json!({"Level": {"type": "string", "enum": ["lite", "full"]}}),
        );
        let level_ref = json!({"$ref": "#/$defs/Level"});
        // `(name, schema, $defs, nullable)`.
        let shapes: Vec<(&str, Value, Option<Value>, bool)> = vec![
            (
                "enum",
                json!({"type": "string", "enum": ["lite", "full"]}),
                None,
                false,
            ),
            (
                "nullable enum",
                json!({"type": ["string", "null"], "enum": ["lite", "full", null]}),
                None,
                true,
            ),
            (
                "anyOf[enum, null]",
                json!({"anyOf": [
                    {"type": "string", "enum": ["lite", "full"]},
                    {"type": "null"},
                ]}),
                None,
                true,
            ),
            (
                "const",
                json!({"type": "string", "const": "full"}),
                None,
                false,
            ),
            (
                "anyOf[const, const]",
                json!({"anyOf": [{"const": "lite"}, {"const": "full"}]}),
                None,
                false,
            ),
            ("$ref", level_ref.clone(), level.clone(), false),
            (
                "anyOf[$ref, null]",
                json!({"anyOf": [level_ref, {"type": "null"}]}),
                level,
                true,
            ),
            ("Agora Option<DetailLevel>", agora_detail(), None, true),
        ];
        let syntax = CallSyntax::qwen_xml();
        for (name, schema, defs, nullable) in shapes {
            let tool = mode_tool(schema, defs);
            let emission = qwen_mode_call("full");
            assert!(qwen_admits(&tool, &emission), "{name}: raw member");
            for wrong in ["\"full\"", "fullest", "ful", "Full", " full", ""] {
                assert!(
                    !qwen_admits(&tool, &qwen_mode_call(wrong)),
                    "{name}: admits {wrong:?}"
                );
            }
            let null = qwen_mode_call("null");
            assert_eq!(qwen_admits(&tool, &null), nullable, "{name}: null");

            let mut cases = vec![(emission, json!("full"))];
            if nullable {
                cases.push((null, Value::Null));
            }
            for (emission, want) in cases {
                let parsed = parse_text(
                    &syntax,
                    &[&tool],
                    &emission,
                    false,
                    Leniency::Final,
                );
                let calls = calls_of(&parsed.blocks);
                assert_eq!(calls.len(), 1, "{name}: {parsed:#?}");
                assert_eq!(calls[0].1, &json!({"mode": want}), "{name}");
                assert_eq!(
                    crate::schema_check::check(&tool.schema, calls[0].1),
                    Ok(()),
                    "{name}"
                );
                let rendered =
                    render_reference(&syntax, &[("set_mode", calls[0].1)])
                        .expect("renders");
                assert_eq!(rendered, emission, "{name}: byte-stable");
            }
        }
    }

    /// A quoted member — what the grammar wrote before the set went raw,
    /// or what a model writes unconstrained — still reads as the member,
    /// so a transcript from either side of the change parses; so does a
    /// padded one. Text that is no member falls through to the JSON read.
    #[test]
    fn qwen_xml_string_set_reads_leniently() {
        let syntax = CallSyntax::qwen_xml();
        let tool = mode_tool(
            json!({"type": "string", "enum": ["lite", "full"]}),
            None,
        );
        for (raw, want) in [
            ("\"full\"", json!("full")),
            (" full ", json!("full")),
            ("other", json!("other")),
        ] {
            let parsed = parse_text(
                &syntax,
                &[&tool],
                &qwen_mode_call(raw),
                false,
                Leniency::Final,
            );
            let calls = calls_of(&parsed.blocks);
            assert_eq!(calls[0].1, &json!({"mode": want}), "{raw:?}");
        }
    }

    /// What a Qwen XML parameter reads leniently, and what it refuses.
    /// A JSON-spelled parameter that reads as no JSON at all makes the
    /// call malformed — degraded to text (which the session then
    /// rejects as a real `<tool_call>` in free text, a resample) —
    /// never the raw text as a string: that retyped a value the grammar
    /// admitted but serde refuses, and a union admitting strings even
    /// passed the schema check as.
    #[test]
    fn qwen_xml_unreadable_json_is_malformed_not_a_string() {
        let syntax = CallSyntax::qwen_xml();
        let read = |mode: Value, raw: &str| {
            let tool = mode_tool(mode, None);
            let parsed = parse_text(
                &syntax,
                &[&tool],
                &qwen_mode_call(raw),
                false,
                Leniency::Final,
            );
            calls_of(&parsed.blocks).first().map(|(_, input)| {
                let input: &Value = input;
                input["mode"].clone()
            })
        };
        let huge = "9".repeat(401);
        let number = json!({"type": "number"});
        let number_or_string = json!({"type": ["number", "string"]});
        let any_of = json!({"anyOf": [{"type": "number"}, {"type": "string"}]});

        // Lenient, and kept: a raw string reads as written, JSON-looking
        // or not; a nullable one takes a bare `null`; an unlisted member
        // of a set stays a string; pythonisms and unclosed brackets of a
        // JSON value heal.
        let string = json!({"type": "string"});
        let nullable = json!({"type": ["string", "null"]});
        let set = json!({"enum": ["lite", "full"]});
        let object = json!({"type": "object"});
        for (mode, raw, want) in [
            (&string, "{\"a\": 1", json!("{\"a\": 1")),
            (&string, huge.as_str(), json!(huge)),
            (&nullable, "null", Value::Null),
            (&nullable, "nil", json!("nil")),
            (&set, "other", json!("other")),
            (&json!({"type": "boolean"}), "True", json!(true)),
            (&object, "{'a': None}", json!({"a": null})),
            (&object, "{\"a\": [1", json!({"a": [1]})),
            (&number, "1e308", json!(1e308)),
        ] {
            assert_eq!(read(mode.clone(), raw), Some(want), "{mode} {raw:?}");
        }

        // Refused: no JSON reads out of it, so no call — whether the
        // schema admits a string or not.
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        for mode in [&number, &number_or_string, &any_of, &json!({})] {
            for raw in [huge.as_str(), "1e999", "-1e999", deep.as_str()] {
                assert_eq!(read(mode.clone(), raw), None, "{mode} {raw:?}");
            }
        }
        assert_eq!(read(number.clone(), "five"), None);
        assert_eq!(read(number_or_string.clone(), "five"), None);

        // A call in flight whose closed member is unreadable has nothing
        // to show either: no open call, under any leniency.
        let tool = mode_tool(number.clone(), None);
        let cut = qwen_mode_call(&huge);
        let cut = &cut[..cut.find("</function>").unwrap()];
        let (parsed, open) =
            parse_text_open(&syntax, &[&tool], cut, false, Leniency::Clipped);
        assert!(calls_of(&parsed.blocks).is_empty(), "{parsed:#?}");
        assert!(open.is_none(), "{open:#?}");
    }

    /// The edges of the raw spelling. A mixed set spells its strings raw
    /// and the rest as JSON, which is what the template renders for each.
    /// A set whose spellings would collide (`"1"` beside `1`, `"null"`
    /// beside `null`) or whose member contains the close tag stays JSON:
    /// raw, it could not be read back. A member with surrounding
    /// whitespace stays raw: the template renders it verbatim and the
    /// read is byte-exact.
    #[test]
    fn qwen_xml_string_set_edges() {
        let syntax = CallSyntax::qwen_xml();
        let close = syntax.arguments.value_suffix.clone();
        let tagged = |mode: Value| {
            let tool = mode_tool(mode, None);
            crate::dialect::emit::tagged_value(
                &syntax,
                &tool.schema,
                &tool.schema["properties"]["mode"],
            )
        };
        let raw = |pairs: &[(&str, Value)]| {
            TaggedValue::Choice(
                pairs
                    .iter()
                    .map(|(s, v)| {
                        Arc::new(crate::dialect::emit::Member {
                            spelling: s.to_string(),
                            value: v.clone(),
                        })
                    })
                    .collect(),
            )
        };
        assert_eq!(
            tagged(json!({"enum": ["a", 1, null]})),
            raw(&[("a", json!("a")), ("1", json!(1)), ("null", Value::Null)])
        );
        assert_eq!(
            tagged(json!({"enum": [" padded "]})),
            raw(&[(" padded ", json!(" padded "))])
        );
        // The empty string is a spelling too: nothing between the tags.
        assert_eq!(
            tagged(json!({"enum": ["", "a"]})),
            raw(&[("", json!("")), ("a", json!("a"))])
        );
        for json_only in [
            json!({"enum": ["1", 1]}),
            json!({"enum": ["null", null]}),
            json!({"enum": ["true", true]}),
            json!({"enum": [format!("a{close}b")]}),
            // Ends with the close tag's prefix, so the first close tag
            // in `member + close` starts inside the member.
            json!({"enum": ["a\n</parameter>"]}),
            // No string in it: JSON as ever.
            json!({"enum": [1, 2]}),
            json!({"type": "null"}),
            // A free string beside a non-string set.
            json!({"anyOf": [{"type": "string"}, {"const": 1}]}),
            json!({"anyOf": [{"enum": ["a"]}, {"type": "integer"}]}),
        ] {
            assert_eq!(
                tagged(json_only.clone()),
                TaggedValue::Json,
                "{json_only}"
            );
        }
        for (string, nullable) in [
            (json!({"type": "string"}), false),
            (json!({"type": ["string", "null"]}), true),
            (
                json!({"anyOf": [{"type": "string"}, {"type": "null"}]}),
                true,
            ),
            // A free string swallows a set of strings.
            (
                json!({"anyOf": [{"type": "string"}, {"const": "a"}]}),
                false,
            ),
        ] {
            assert_eq!(
                tagged(string.clone()),
                TaggedValue::Raw { nullable },
                "{string}"
            );
        }

        // The mixed set end to end: admitted raw, read typed, rendered
        // back byte for byte.
        let tool = mode_tool(json!({"enum": ["a", 1, null]}), None);
        assert!(!qwen_admits(&tool, &qwen_mode_call("\"a\"")));
        for (raw, want) in
            [("a", json!("a")), ("1", json!(1)), ("null", Value::Null)]
        {
            let emission = qwen_mode_call(raw);
            assert!(qwen_admits(&tool, &emission), "{raw:?}");
            let parsed = parse_text(
                &syntax,
                &[&tool],
                &emission,
                false,
                Leniency::Final,
            );
            let calls = calls_of(&parsed.blocks);
            assert_eq!(calls[0].1, &json!({"mode": want}), "{raw:?}");
            assert_eq!(
                crate::schema_check::check(&tool.schema, calls[0].1),
                Ok(())
            );
            assert_eq!(
                render_reference(&syntax, &[("set_mode", calls[0].1)]).unwrap(),
                emission
            );
        }

        // A colliding set is JSON on both sides: quoted admitted, read
        // as the string.
        let tool = mode_tool(json!({"enum": ["1", 1]}), None);
        assert!(qwen_admits(&tool, &qwen_mode_call("\"1\"")));
        assert!(qwen_admits(&tool, &qwen_mode_call("1")));
        let parsed = parse_text(
            &syntax,
            &[&tool],
            &qwen_mode_call("\"1\""),
            false,
            Leniency::Final,
        );
        assert_eq!(calls_of(&parsed.blocks)[0].1, &json!({"mode": "1"}));
    }

    /// A tagged parameter's `$ref` resolves against the tool's `$defs`:
    /// the parameter's own schema carries none, and an unresolved ref
    /// compiled to any JSON value.
    #[test]
    fn qwen_xml_ref_param_resolves_against_the_tool_defs() {
        let tool = mode_tool(
            json!({"$ref": "#/$defs/Point"}),
            Some(json!({"Point": {
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            }})),
        );
        assert!(qwen_admits(&tool, &qwen_mode_call(r#"{"x":1}"#)));
        for wrong in [r#"{"y":1}"#, r#"{"x":"1"}"#, "[]", "1"] {
            assert!(!qwen_admits(&tool, &qwen_mode_call(wrong)), "{wrong}");
        }
    }

    /// A recursive `$ref` on every dialect: a tree (as schemars emits
    /// one), mutual recursion, and alias chains with a self-alias
    /// among them each compile to a grammar that admits a valid call,
    /// refuses an invalid one, and parses back to the input, which the
    /// schema check passes. The tree used to inline without end and
    /// abort the server on a stack overflow.
    #[test]
    fn recursive_refs_compile_on_every_dialect() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        use crate::{Grammar, GrammarState};
        use std::sync::Arc;

        let node = json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "children": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/Node"},
                },
            },
        });
        let tree = (
            json!({
                "type": "object",
                "properties": {"root": {"$ref": "#/$defs/Node"}},
                "required": ["root"],
                "$defs": {"Node": node},
            }),
            json!({"root": {"name": "a", "children": [
                {"name": "b", "children": [{"name": "c", "children": []}]},
                {"name": "d"},
            ]}}),
            json!({"root": {"name": "a", "children": [
                {"name": "b", "children": [{"name": 3}]},
            ]}}),
        );
        let mutual = (
            json!({
                "type": "object",
                "properties": {"a": {"$ref": "#/$defs/A"}},
                "required": ["a"],
                "$defs": {
                    "A": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "b": {"anyOf": [
                                {"$ref": "#/$defs/B"},
                                {"type": "null"},
                            ]},
                        },
                        "required": ["label", "b"],
                    },
                    "B": {
                        "type": "object",
                        "properties": {"a": {"$ref": "#/$defs/A"}},
                        "required": ["a"],
                    },
                },
            }),
            json!({"a": {"label": "x", "b": {"a": {"label": "y", "b": null}}}}),
            json!({"a": {"label": "x", "b": {"a": {"label": 5, "b": null}}}}),
        );
        let aliases = (
            json!({
                "type": "object",
                "properties": {
                    "count": {"$ref": "#/$defs/Count"},
                    "extra": {"$ref": "#/$defs/Me"},
                },
                "required": ["count", "extra"],
                "$defs": {
                    "Count": {"$ref": "#/$defs/Int"},
                    "Int": {"$ref": "#/$defs/Integer"},
                    "Integer": {"type": "integer"},
                    "Me": {"$ref": "#/$defs/Me"},
                },
            }),
            json!({"count": 3, "extra": {"any": [1, true]}}),
            json!({"count": "three", "extra": 1}),
        );
        let options = EmitOptions {
            anchor: Anchor::Lazy,
            parallel: false,
            ..Default::default()
        };
        for (case, (schema, valid, invalid)) in
            [("tree", tree), ("mutual", mutual), ("aliases", aliases)]
        {
            let tool = Tool::builder("grow")
                .description("test")
                .schema(schema)
                .build()
                .expect("valid test tool");
            assert!(crate::schema_check::check(&tool.schema, &valid).is_ok());
            assert!(crate::schema_check::check(&tool.schema, &invalid).is_err());
            for (name, syntax) in call_dialects() {
                let at = format!("{name}, {case}");
                let src = grammar_source(&syntax, &[&tool], &options)
                    .unwrap_or_else(|e| panic!("{at}: {e}"));
                let grammar = Arc::new(
                    Grammar::parse(&src)
                        .unwrap_or_else(|e| panic!("{at}: {e}\n{src}")),
                );
                // Plus the turn-exit marker the grammar ends on, where the
                // dialect has one (Gemma's `<|tool_response>`).
                let admits = |input: &Value| {
                    let call = render_reference(&syntax, &[("grow", input)])
                        .unwrap_or_else(|e| panic!("{at}: {e}"));
                    let framed =
                        format!("{call}{}", syntax.tool_response_start);
                    let mut state = GrammarState::new(grammar.clone());
                    (state.advance_bytes(framed.as_bytes()).is_ok()
                        && state.is_complete())
                    .then_some(call)
                };
                let call = admits(&valid)
                    .unwrap_or_else(|| panic!("{at}: valid refused\n{src}"));
                let parsed = parse_text(
                    &syntax,
                    &[&tool],
                    &call,
                    false,
                    Leniency::Final,
                );
                assert_eq!(
                    calls_of(&parsed.blocks),
                    [("grow", &valid)],
                    "{at}: {call}"
                );
                assert!(admits(&invalid).is_none(), "{at}: invalid admitted");
            }
        }
    }

    /// The tagged-value classifier walks a wide diamond of `anyOf`s
    /// (`D_i = anyOf[D_{i+1} × 50]`) once per def, not 50^n times, and
    /// still finds the string set at its end.
    #[test]
    fn qwen_xml_ref_diamond_is_linear() {
        let n = 10;
        let mut defs = serde_json::Map::new();
        for i in 0..n {
            let next = json!({"$ref": format!("#/$defs/D{}", i + 1)});
            defs.insert(format!("D{i}"), json!({"anyOf": vec![next; 50]}));
        }
        defs.insert(format!("D{n}"), json!({"enum": ["lite", "full"]}));
        let tool = mode_tool(json!({"$ref": "#/$defs/D0"}), Some(defs.into()));
        assert!(qwen_admits(&tool, &qwen_mode_call("full")));
        assert!(!qwen_admits(&tool, &qwen_mode_call("fullest")));
    }

    /// The raw spelling is the tagged dialects' alone: a JSON dialect
    /// still quotes a string enum, as JSON must.
    #[test]
    fn json_dialects_keep_quoting_a_string_set() {
        let tool = mode_tool(
            json!({"type": "string", "enum": ["lite", "full"]}),
            None,
        );
        let syntax = CallSyntax::hermes_json();
        let source = crate::dialect::grammar_source(
            &syntax,
            &[&tool],
            &crate::dialect::EmitOptions::default(),
        )
        .expect("grammar");
        let admits = |mode: &str| {
            let call = render_reference(
                &syntax,
                &[("set_mode", &json!({"mode": mode}))],
            )
            .expect("renders");
            let mut state = crate::GrammarState::from_source(&source)
                .expect("grammar parses");
            (state.advance_bytes(call.as_bytes()).is_ok()
                && state.is_complete())
            .then_some(call)
        };
        let call = admits("full").expect("quoted member admitted");
        assert!(call.contains(r#""mode":"full""#), "{call}");
        assert!(admits("fullest").is_none());
    }

    /// The schemars derive itself, not a transcription of its output:
    /// Agora's `Option<DetailLevel>`, `inline` with documented variants.
    #[cfg(feature = "json-schema")]
    #[test]
    fn qwen_xml_reads_a_derived_option_enum_raw() {
        /// How much of a piece of content to return.
        #[derive(schemars::JsonSchema)]
        #[schemars(inline)]
        #[serde(rename_all = "snake_case")]
        #[allow(dead_code)]
        enum DetailLevel {
            /// The short form.
            Summary,
            /// The verbatim record.
            Full,
        }
        #[derive(schemars::JsonSchema)]
        #[allow(dead_code)]
        struct GetContentInput {
            /// How much to return.
            detail: Option<DetailLevel>,
        }
        let schema =
            serde_json::to_value(schemars::schema_for!(GetContentInput))
                .expect("schema");
        let mut tool = Tool::builder("set_mode")
            .description("test")
            .schema(json!({
                "type": "object",
                "properties": {"mode": schema["properties"]["detail"]},
            }))
            .build()
            .expect("valid test tool");
        tool.strict = Some(true);
        let syntax = CallSyntax::qwen_xml();
        for (raw, want) in [("full", json!("full")), ("null", Value::Null)] {
            let emission = qwen_mode_call(raw);
            assert!(qwen_admits(&tool, &emission), "{raw}: {schema:#}");
            let parsed = parse_text(
                &syntax,
                &[&tool],
                &emission,
                false,
                Leniency::Final,
            );
            let calls = calls_of(&parsed.blocks);
            assert_eq!(calls[0].1, &json!({"mode": want}));
            assert_eq!(
                crate::schema_check::check(&tool.schema, calls[0].1),
                Ok(())
            );
        }
        assert!(!qwen_admits(&tool, &qwen_mode_call("\"full\"")));
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

    // Issue #53: a tool call emitted inside an *unclosed* reasoning
    // block must surface as a `ToolUse`, not be swallowed into the
    // Thought (pre-opened) or a Text block (mid-stream). Deterministic
    // parser-level coverage — the model-backed round-trip fuzzer can't
    // reliably reproduce this shape on a thinking-disabled prompt.
    // These assert the *functional* fix (call surfaces); byte-exact
    // round-trip of an unclosed-think emission is a grammar/canon
    // concern tracked separately.

    /// Pre-opened reasoning, no `</think>`, then a call: prose becomes
    /// a Thought and the call a ToolUse. (Bug site 1 — the pre-opened
    /// `None`-close arm.)
    #[test]
    fn pre_opened_unclosed_think_then_call() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3});
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).unwrap();
        // Text begins INSIDE reasoning (pre-opened); no `</think>`.
        let text = format!("planning the call\n{call}");
        let parsed = parse_text(&syntax, &[&t], &text, true, Leniency::Final);
        assert_eq!(
            parsed.status,
            ParseStatus::Complete,
            "{:#?}",
            parsed.blocks
        );
        assert!(
            matches!(&parsed.blocks[0], Block::Thought { thought, .. }
                if thought.contains("planning")),
            "prose before the call is a Thought: {:#?}",
            parsed.blocks
        );
        let calls = calls_of(&parsed.blocks);
        assert_eq!(
            calls.len(),
            1,
            "the call must surface, not be swallowed: {:#?}",
            parsed.blocks
        );
        assert_eq!(calls[0].1["city"], json!("Paris"));
        assert!(
            !parsed
                .blocks
                .iter()
                .any(|b| matches!(b, Block::Text { .. })),
            "no stray Text carrying the swallowed call: {:#?}",
            parsed.blocks
        );
    }

    /// Mid-stream `<think>{prose}<tool_call>…` with no close. (Bug
    /// site 2 — the `parse_thought` `None`-close branch.)
    #[test]
    fn midstream_unclosed_think_then_call() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3});
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).unwrap();
        let text = format!("<think>\nreason it out\n{call}"); // no </think>
        let parsed = parse_text(&syntax, &[&t], &text, false, Leniency::Final);
        assert!(
            matches!(&parsed.blocks[0], Block::Thought { thought, .. }
                if thought.contains("reason it out")),
            "{:#?}",
            parsed.blocks
        );
        assert_eq!(calls_of(&parsed.blocks).len(), 1, "{:#?}", parsed.blocks);
    }

    /// A pre-opened reasoning block cut off by `max_tokens` surfaces as
    /// an **open** Thought whose body is byte-exact — trailing
    /// whitespace included. The closed path strips only the close
    /// marker's canonical leading `"\n"`, which the re-rendered marker
    /// puts back; an open thought has no close, so stripping anything
    /// would lose bytes the KV cache holds.
    #[test]
    fn pre_opened_unclosed_thought_is_open_and_byte_exact() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        // Ran out the clock mid-paragraph: real trailing newlines.
        let emission = "Weighing the two options.\n\n\n";
        let parsed =
            parse_text(&syntax, &[&t], emission, true, Leniency::Final);
        assert_eq!(parsed.blocks.len(), 1, "{:#?}", parsed.blocks);
        let Block::Thought { thought, signature } = &parsed.blocks[0] else {
            panic!("expected a Thought: {:#?}", parsed.blocks);
        };
        assert_eq!(
            thought, emission,
            "open thought bodies are stored raw, not trimmed"
        );
        assert_eq!(signature, crate::prompt::OPEN_THOUGHT_SIGNATURE);
        assert!(crate::prompt::is_open_thought(&parsed.blocks[0]));
    }

    /// The same emission with its close marker present takes the closed
    /// path: empty signature, and exactly the close marker's canonical
    /// whitespace comes off the body — the `"\n"` of the analyzed
    /// `"\n</think>"`, which the template puts back — and nothing else,
    /// so a thought closed on blank lines re-renders them.
    #[test]
    fn closed_thought_keeps_empty_signature_and_its_own_whitespace() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let parsed = parse_text(
            &syntax,
            &[&t],
            "Weighing the two options.\n\n\n</think>\n\nDone.",
            true,
            Leniency::Final,
        );
        let Block::Thought { thought, signature } = &parsed.blocks[0] else {
            panic!("expected a Thought: {:#?}", parsed.blocks);
        };
        assert_eq!(thought, "Weighing the two options.\n\n");
        assert!(signature.is_empty(), "closed thoughts carry no signature");
        assert!(!crate::prompt::is_open_thought(&parsed.blocks[0]));

        // The spontaneous (not pre-opened) path agrees.
        let parsed = parse_text(
            &syntax,
            &[&t],
            "<think>\nWeighing.\n\n</think>\n\nDone.",
            false,
            Leniency::Final,
        );
        let Block::Thought { thought, .. } = &parsed.blocks[0] else {
            panic!("expected a Thought: {:#?}", parsed.blocks);
        };
        assert_eq!(thought, "Weighing.\n");
    }

    /// An empty pre-opened thought is the thinking-off scaffold to the
    /// re-render, so the separator after it is the scaffold's and not
    /// the answer's — and a streamed prefix of it is held back rather
    /// than yielded as prose it would later have to take back.
    #[test]
    fn empty_pre_opened_thought_consumes_the_scaffold_separator() {
        let mut syntax = CallSyntax::qwen_xml();
        syntax.reasoning.separator = Some("\n\n".into());
        let t = tool("get_weather");
        let text = |emission| {
            merge_text(
                parse_text(&syntax, &[&t], emission, true, Leniency::Final)
                    .blocks,
            )
        };
        assert_eq!(
            text("\n</think>\n\nDone."),
            [Block::from("Done.".to_string())]
        );
        assert_eq!(
            text("\n</think>\n\n\nDone."),
            [Block::from("\nDone.".to_string())]
        );
        // Not the separator: nothing is consumed.
        assert_eq!(
            text("\n</think>\nDone."),
            [Block::from("\nDone.".to_string())]
        );
        // A real thought keeps the gap in its answer, as it always has.
        assert_eq!(
            text("Hm.\n</think>\n\nDone.")[1],
            Block::from("\n\nDone.".to_string())
        );

        for chunk in 1..=4usize {
            let mut p =
                StreamParser::new(syntax.clone(), vec![t.clone()], true);
            let emission = "\n</think>\n\nDone.";
            let mut out = Vec::new();
            for piece in emission.as_bytes().chunks(chunk) {
                out.extend(p.push(std::str::from_utf8(piece).unwrap()));
            }
            out.extend(p.finish());
            assert_eq!(
                merge_text(out),
                [Block::from("Done.".to_string())],
                "chunk={chunk}"
            );
        }
    }

    /// A *spontaneous* `<think>` (not pre-opened) that never closes used
    /// to fall through to `incomplete`, which seated the literal
    /// `<think>` marker inside a `Block::Text` — the thought-half of
    /// issue #38. It is now an open Thought, marker excluded from the
    /// body since the renderer re-emits it.
    #[test]
    fn spontaneous_unclosed_thought_is_open_not_marker_text() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let parsed = parse_text(
            &syntax,
            &[&t],
            "<think>\nstill reasoning when the clock ran out",
            false,
            Leniency::Final,
        );
        assert_eq!(parsed.blocks.len(), 1, "{:#?}", parsed.blocks);
        assert!(
            crate::prompt::is_open_thought(&parsed.blocks[0]),
            "{:#?}",
            parsed.blocks
        );
        let Block::Thought { thought, .. } = &parsed.blocks[0] else {
            panic!("expected a Thought: {:#?}", parsed.blocks);
        };
        assert_eq!(
            thought, "\nstill reasoning when the clock ran out",
            "the open marker is framing, not body"
        );
    }

    /// Streaming must agree with batch on openness: `finish()` reparses
    /// under `Leniency::Final`, so the flush yields the same open
    /// Thought the batch parser produces.
    #[test]
    fn stream_flush_agrees_with_batch_on_open_thought() {
        let syntax = CallSyntax::qwen_xml();
        let emission = "reasoning, interrupted\n\n";
        let batch =
            parse_text(&syntax, &[&tool("t")], emission, true, Leniency::Final)
                .blocks;

        let mut p = StreamParser::new(syntax, vec![tool("t")], true);
        let mut streamed = p.push(emission);
        streamed.extend(p.finish());

        assert_eq!(streamed, batch, "streaming flush must equal batch");
        assert!(streamed.iter().any(crate::prompt::is_open_thought));
    }

    /// An open thought with no body at all is dropped, deliberately:
    /// under a pre-opened template the generation prompt re-emits the
    /// open marker itself, so an empty open thought contributes no
    /// bytes either way. Pinned so it stays a decision.
    #[test]
    fn empty_open_thought_is_dropped() {
        let syntax = CallSyntax::qwen_xml();
        let parsed =
            parse_text(&syntax, &[&tool("t")], "", true, Leniency::Final);
        assert!(parsed.blocks.is_empty(), "{:#?}", parsed.blocks);
    }

    /// A bare `<tool_call>` mention in an unclosed thought. The bare
    /// special is the call landmark now ([`CallSyntax::trigger`]; it
    /// used to be `<tool_call>\n`, so a mention followed by a space was
    /// no landmark at all), so the thought splits there and what follows
    /// — not a call — degrades to Text: no spurious ToolUse, no dropped
    /// bytes, no `<think>` marker seated as Text (#38). A session never
    /// emits this shape: the special arms the grammar, which forces a
    /// call from there (`real_opener_shapes_never_reach_containment`).
    #[test]
    fn unclosed_think_opener_mention_is_not_a_call() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let preserved = |blocks: &[Block]| {
            blocks.iter().any(|b| match b {
                Block::Thought { thought, .. } => {
                    thought.contains("<tool_call>")
                }
                Block::Text { text, .. } => text.contains("<tool_call>"),
                _ => false,
            })
        };
        let midstream = "<think>\nI could emit a <tool_call> but not yet";
        let parsed =
            parse_text(&syntax, &[&t], midstream, false, Leniency::Final);
        assert!(
            calls_of(&parsed.blocks).is_empty(),
            "a mention must not become a call: {:#?}",
            parsed.blocks
        );
        assert!(
            preserved(&parsed.blocks),
            "the mention is preserved, not dropped: {:#?}",
            parsed.blocks
        );
        // And the reasoning open marker stays framing, not content.
        assert!(
            parsed
                .blocks
                .iter()
                .all(|b| !matches!(b, Block::Text { text, .. }
                if text.contains("<think>"))),
            "no `<think>` marker seated as Text (#38): {:#?}",
            parsed.blocks
        );

        // Same shape, pre-opened.
        let pre = "reasoning with a <tool_call> mention only";
        let parsed = parse_text(&syntax, &[&t], pre, true, Leniency::Final);
        assert!(calls_of(&parsed.blocks).is_empty(), "{:#?}", parsed.blocks);
        assert!(preserved(&parsed.blocks), "{:#?}", parsed.blocks);
    }

    /// Regression: a properly-*closed* `</think>` then a call still
    /// parses to Thought + ToolUse, in both literal and pre-opened
    /// forms — the fix must not disturb the happy path.
    #[test]
    fn closed_think_then_call_still_parses() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3});
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).unwrap();
        let closed = format!("<think>\nthinking\n</think>\n{call}");
        for (text, pre) in [
            (closed.clone(), false),
            (closed.strip_prefix("<think>").unwrap().to_string(), true),
        ] {
            let parsed =
                parse_text(&syntax, &[&t], &text, pre, Leniency::Final);
            assert!(
                matches!(&parsed.blocks[0], Block::Thought { thought, .. }
                    if thought.contains("thinking")),
                "pre={pre}: {:#?}",
                parsed.blocks
            );
            assert_eq!(
                calls_of(&parsed.blocks).len(),
                1,
                "pre={pre}: {:#?}",
                parsed.blocks
            );
        }
    }

    /// Empty reasoning body before the call yields no stray empty
    /// Thought (`push_thought` drops empties).
    #[test]
    fn unclosed_think_empty_body_no_empty_thought() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3});
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).unwrap();
        // Pre-opened, call immediately: no reasoning prose, no close.
        let parsed = parse_text(&syntax, &[&t], &call, true, Leniency::Final);
        assert!(
            !parsed
                .blocks
                .iter()
                .any(|b| matches!(b, Block::Thought { .. })),
            "empty reasoning must not make a Thought: {:#?}",
            parsed.blocks
        );
        assert_eq!(calls_of(&parsed.blocks).len(), 1, "{:#?}", parsed.blocks);
    }

    /// Multiple calls after an unclosed reasoning block: all surface.
    #[test]
    fn unclosed_think_then_two_calls() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let a = json!({"city": "Paris", "days": 3});
        let b = json!({"city": "London", "days": 5});
        let two = format!(
            "{}\n{}",
            render_reference(&syntax, &[("get_weather", &a)]).unwrap(),
            render_reference(&syntax, &[("get_weather", &b)]).unwrap(),
        );
        let text = format!("planning\n{two}");
        let parsed = parse_text(&syntax, &[&t], &text, true, Leniency::Final);
        let calls = calls_of(&parsed.blocks);
        assert_eq!(calls.len(), 2, "{:#?}", parsed.blocks);
        assert_eq!(calls[1].1["city"], json!("London"));
    }

    /// Streaming: a partial call inside an unclosed reasoning block
    /// must report `NeedMoreInput` and surface no call — the Thought
    /// split is stable once the full trigger is buffered, but the call
    /// is held back until complete.
    #[test]
    fn streaming_unclosed_think_partial_call_needs_more_input() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let text =
            "planning\n<tool_call>\n<function=get_weather>\n<parameter=ci";
        let parsed =
            parse_text(&syntax, &[&t], text, true, Leniency::Streaming);
        assert_eq!(
            parsed.status,
            ParseStatus::NeedMoreInput,
            "{:#?}",
            parsed.blocks
        );
        assert!(calls_of(&parsed.blocks).is_empty(), "{:#?}", parsed.blocks);
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

    /// Non-ASCII string content survives every heal path byte-exact —
    /// the per-byte `as char` copy used to mojibake it ("Zürich" →
    /// "ZÃ¼rich") into *valid* JSON serde then accepted.
    #[test]
    fn heal_json_preserves_non_ascii() {
        // Double-quoted path.
        assert_eq!(
            heal_json(r#"{"city": "Zürich", "n": 1"#),
            r#"{"city": "Zürich", "n": 1}"#
        );
        // Single-quoted path.
        assert_eq!(heal_json("{'city': 'Zürich'}"), r#"{"city": "Zürich"}"#);
        // Escape followed by a multi-byte char.
        assert_eq!(heal_json("{'a': '\\é'}"), r#"{"a": "\é"}"#);
        // Astral plane (4-byte) content, both quote styles.
        assert_eq!(heal_json(r#"{"a": "🦙"}"#), r#"{"a": "🦙"}"#);
        assert_eq!(heal_json("{'a': '🦙'}"), r#"{"a": "🦙"}"#);
        // End-to-end through the healed-parse entry point.
        let v = parse_json_healed("{'city': 'Zürich'}").unwrap();
        assert_eq!(v["city"], serde_json::json!("Zürich"));
    }

    /// A degenerate `CallSyntax` (empty argument markers, unmatched
    /// close) must yield Malformed, not consume zero bytes per
    /// iteration forever. Reachable via the analyzer's TagWithTagged
    /// fallback on odd templates and via user-constructed syntaxes.
    #[test]
    fn tagged_call_degenerate_syntax_terminates() {
        // Trigger comes from `section_start`; every argument marker
        // stays empty (the degenerate extraction result) and the close
        // never appears in the input.
        let mut syntax = CallSyntax {
            family: Family::TagWithTagged,
            section_start: "<fn>".into(),
            ..Default::default()
        };
        syntax.function.name_suffix = "\n".into();
        syntax.function.close = "</never>".into();
        let t = tool("get_weather");
        // Hung before the progress guard; any non-hanging outcome is
        // acceptable.
        let _ = parse_text(&syntax, &[&t], "<fn>f\nx", false, Leniency::Final);
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
            CallSyntax::gemma4(),
        ] {
            // The grammar additionally requires the turn-exit marker
            // when the dialect has one (Gemma's `<|tool_response>`);
            // it is turn framing, not call bytes, so render_reference
            // doesn't include it.
            let emission =
                render_reference(&syntax, &[("get_weather", &input)])
                    .expect("representable")
                    + &syntax.tool_response_start;
            let src = grammar_source(
                &syntax,
                &[&t],
                &EmitOptions {
                    anchor: Anchor::Lazy,
                    parallel: false,
                    ..Default::default()
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

    /// The eager grammar must exit the thought until-region on a
    /// close marker WITHOUT the canonical leading newline. Trap
    /// regression (plan Phase G postmortem): with the delimiter
    /// `"\n<channel|>"`, a model closing as `...text<channel|>` could
    /// never leave the region — every later byte was legal thought
    /// content — and generation free-ran to the token budget. The
    /// grammar now keys on the trimmed close, like the parser always
    /// did; the missing newline costs a one-turn canonicalization
    /// repair instead.
    #[test]
    fn gemma4_thought_close_without_newline_exits_grammar() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        use crate::{Grammar, GrammarState};
        use std::sync::Arc;

        let syntax = CallSyntax::gemma4();
        let t = tool("get_weather");
        let input = serde_json::json!({"city": "Paris", "days": 3});
        let call = render_reference(&syntax, &[("get_weather", &input)])
            .expect("representable");
        let src = grammar_source(
            &syntax,
            &[&t],
            &EmitOptions {
                anchor: Anchor::Eager,
                parallel: true,
                ..Default::default()
            },
        )
        .expect("emit");
        let grammar = Arc::new(Grammar::parse(&src).expect("grammar"));
        for close in ["\n<channel|>", "<channel|>"] {
            let emission = format!(
                "<|channel>thought\nplanning{close}{call}<|tool_response>"
            );
            let mut state = GrammarState::new(grammar.clone());
            assert!(
                state.advance_bytes(emission.as_bytes()).is_ok()
                    && state.is_complete(),
                "close {close:?} must exit the thought and complete\n{src}"
            );
        }
    }

    // -----------------------------------------------------------------
    // Gemma 4 (TagWithDict): upstream test-matrix port
    // (llama.cpp tests/test-chat.cpp, "Google Gemma 4" section).
    // -----------------------------------------------------------------

    /// One parse; asserts exactly one call and returns its input.
    fn gemma_call(text: &str) -> (String, Value) {
        let syntax = CallSyntax::gemma4();
        let t = tool("t");
        let parsed = parse_text(&syntax, &[&t], text, false, Leniency::Final);
        let calls = calls_of(&parsed.blocks);
        assert_eq!(calls.len(), 1, "{text:?} → {:#?}", parsed.blocks);
        (calls[0].0.to_string(), calls[0].1.clone())
    }

    /// The value-type matrix, byte-for-byte from upstream's pins.
    #[test]
    fn gemma4_value_types() {
        for (text, name, want) in [
            (
                r#"<|tool_call>call:get_time{city:<|"|>London<|"|>}<tool_call|>"#,
                "get_time",
                json!({"city": "London"}),
            ),
            (
                r#"<|tool_call>call:get_time{city:<|"|>San Francisco<|"|>}<tool_call|>"#,
                "get_time",
                json!({"city": "San Francisco"}),
            ),
            (
                "<|tool_call>call:empty_args{}<tool_call|>",
                "empty_args",
                json!({}),
            ),
            (
                "<|tool_call>call:special_function{arg1:42}<tool_call|>",
                "special_function",
                json!({"arg1": 42}),
            ),
            (
                "<|tool_call>call:special_function{arg1:-7}<tool_call|>",
                "special_function",
                json!({"arg1": -7}),
            ),
            (
                "<|tool_call>call:amount{orig:3.5}<tool_call|>",
                "amount",
                json!({"orig": 3.5}),
            ),
            (
                "<|tool_call>call:amount{orig:1.5e10}<tool_call|>",
                "amount",
                json!({"orig": 1.5e10}),
            ),
            (
                "<|tool_call>call:toggle{enabled:true}<tool_call|>",
                "toggle",
                json!({"enabled": true}),
            ),
            (
                "<|tool_call>call:toggle{enabled:false}<tool_call|>",
                "toggle",
                json!({"enabled": false}),
            ),
            (
                "<|tool_call>call:set_nullable{value:null}<tool_call|>",
                "set_nullable",
                json!({"value": null}),
            ),
            // minijinja's canonical null spelling (what a re-rendered
            // turn contains).
            (
                "<|tool_call>call:set_nullable{value:none}<tool_call|>",
                "set_nullable",
                json!({"value": null}),
            ),
            (
                r#"<|tool_call>call:todo_list{todos:[<|"|>buy milk<|"|>,<|"|>walk dog<|"|>]}<tool_call|>"#,
                "todo_list",
                json!({"todos": ["buy milk", "walk dog"]}),
            ),
            (
                "<|tool_call>call:todo_list{todos:[]}<tool_call|>",
                "todo_list",
                json!({"todos": []}),
            ),
            (
                r#"<|tool_call>call:set_config{config:{theme:<|"|>dark<|"|>,count:3}}<tool_call|>"#,
                "set_config",
                json!({"config": {"theme": "dark", "count": 3}}),
            ),
            (
                "<|tool_call>call:set_config{config:{}}<tool_call|>",
                "set_config",
                json!({"config": {}}),
            ),
        ] {
            let (got_name, got) = gemma_call(text);
            assert_eq!(got_name, name, "{text:?}");
            assert_eq!(got, want, "{text:?}");
        }
    }

    /// Content and parallel calls around the dict envelope.
    #[test]
    fn gemma4_content_and_parallel() {
        let syntax = CallSyntax::gemma4();
        let t = tool("t");

        let text = "Hello, world!\nWhat's up?<|tool_call>call:get_time\
                    {city:<|\"|>Paris<|\"|>}<tool_call|>";
        let parsed = parse_text(&syntax, &[&t], text, false, Leniency::Final);
        assert!(
            matches!(&parsed.blocks[0], Block::Text { text, .. }
                if text.as_ref() == "Hello, world!\nWhat's up?"),
            "{:#?}",
            parsed.blocks
        );
        assert_eq!(calls_of(&parsed.blocks).len(), 1);

        let text = "<|tool_call>call:get_time{city:<|\"|>London<|\"|>}\
                    <tool_call|><|tool_call>call:get_weather\
                    {city:<|\"|>Paris<|\"|>}<tool_call|>";
        let parsed = parse_text(&syntax, &[&t], text, false, Leniency::Final);
        let calls = calls_of(&parsed.blocks);
        assert_eq!(calls.len(), 2, "{:#?}", parsed.blocks);
        assert_eq!(calls[0].0, "get_time");
        assert_eq!(calls[1].0, "get_weather");
        assert_eq!(calls[1].1["city"], json!("Paris"));
    }

    /// The turn-exit marker (`<|tool_response>`) that the grammar
    /// requires after the last call is envelope, not content: the
    /// parser swallows it, whole and under any chunking.
    #[test]
    fn gemma4_turn_exit_marker_is_swallowed() {
        let syntax = CallSyntax::gemma4();
        let t = tool("t");
        let text = "Okay.<|tool_call>call:get_time{city:<|\"|>Paris<|\"|>}\
                    <tool_call|><|tool_response>";
        let parsed = parse_text(&syntax, &[&t], text, false, Leniency::Final);
        assert_eq!(calls_of(&parsed.blocks).len(), 1, "{:#?}", parsed.blocks);
        assert!(
            !parsed.blocks.iter().any(|b| matches!(
                b,
                Block::Text { text, .. } if text.contains("tool_response")
            )),
            "exit marker leaked into Text: {:#?}",
            parsed.blocks
        );

        // Streaming: the marker (and any prefix of it) never surfaces
        // as a prose delta.
        for chunk in 1..=5usize {
            let mut p =
                StreamParser::new(syntax.clone(), vec![t.clone()], false);
            let mut streamed = Vec::new();
            let mut i = 0;
            while i < text.len() {
                let mut j = (i + chunk).min(text.len());
                while !text.is_char_boundary(j) {
                    j += 1;
                }
                streamed.extend(p.push(&text[i..j]));
                i = j;
            }
            streamed.extend(p.finish());
            let prose: String = streamed
                .iter()
                .filter_map(|b| match b {
                    Block::Text { text, .. } => Some(text.to_string()),
                    _ => None,
                })
                .collect();
            assert_eq!(prose, "Okay.", "chunk={chunk}: {streamed:#?}");
        }
    }

    /// Channel-noise matrix: empty thoughts drop, unmatched closes
    /// and bare channel opens are consumed silently (upstream edge
    /// cases, `test-chat.cpp` "Edge cases").
    #[test]
    fn gemma4_channel_noise() {
        let syntax = CallSyntax::gemma4();
        let t = tool("t");
        let content = "Hello, world!\nWhat's up?";

        for (text, want_thought) in [
            // Reasoning and content.
            (
                "<|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?",
                Some("I'm\nthinking"),
            ),
            // Empty reasoning (budget=0: close before newline).
            ("<|channel>thought<channel|>Hello, world!\nWhat's up?", None),
            // Empty thought + trailing unmatched close.
            (
                "<|channel>thought\n<channel|>Hello, world!\nWhat's up?<channel|>",
                None,
            ),
            // Trailing empty thought block after content.
            (
                "<|channel>thought\n<channel|>Hello, world!\nWhat's up?<|channel>thought\n<channel|>",
                None,
            ),
            // ... plus a stray close on top.
            (
                "<|channel>thought\n<channel|>Hello, world!\nWhat's up?<|channel>thought\n<channel|><channel|>",
                None,
            ),
            // Bare channel open before the real thought.
            (
                "<|channel><|channel>thought\nI'm\nthinking<channel|>Hello, world!\nWhat's up?",
                Some("I'm\nthinking"),
            ),
        ] {
            let parsed =
                parse_text(&syntax, &[&t], text, false, Leniency::Final);
            let blocks = merge_text(parsed.blocks);
            let mut expect: Vec<Block> = Vec::new();
            if let Some(thought) = want_thought {
                expect.push(Block::Thought {
                    thought: thought.to_string().into(),
                    signature: std::borrow::Cow::Borrowed(""),
                });
            }
            expect.push(content.to_string().into());
            assert_eq!(blocks.len(), expect.len(), "{text:?}: {blocks:#?}");
            for (got, want) in blocks.iter().zip(expect.iter()) {
                match (got, want) {
                    (
                        Block::Text { text: a, .. },
                        Block::Text { text: b, .. },
                    ) => assert_eq!(a, b, "{text:?}"),
                    (
                        Block::Thought { thought: a, .. },
                        Block::Thought { thought: b, .. },
                    ) => assert_eq!(a, b, "{text:?}"),
                    other => panic!("{text:?}: {other:?}"),
                }
            }
        }
    }

    /// Unrepresentable dict values: the quote marker in a string (or
    /// key) is a typed error at render/validate time; dict
    /// terminators in keys likewise. Values containing OTHER markers
    /// (per-call close, braces) round-trip fine inside quotes.
    #[test]
    fn gemma4_adversarial_values() {
        let syntax = CallSyntax::gemma4();
        let t = tool("get_weather");

        for value in [
            "trailing newline\n",
            "{\"looks\": \"like json\"}",
            "emoji 🍓 and 中文",
            "a <tool_call|> inside",
            "braces { } and commas , and colons :",
        ] {
            let input = json!({"city": value, "days": 1});
            let emission =
                render_reference(&syntax, &[("get_weather", &input)])
                    .expect("representable");
            let parsed =
                parse_text(&syntax, &[&t], &emission, false, Leniency::Final);
            let calls = calls_of(&parsed.blocks);
            assert_eq!(calls.len(), 1, "value {value:?}: {parsed:#?}");
            assert_eq!(
                calls[0].1["city"],
                json!(value),
                "value {value:?} must round-trip byte-exact"
            );
        }

        // Quote marker embedded in a string value: typed error.
        let evil = json!({"city": "sneaky <|\"|> injection", "days": 1});
        let err = render_reference(&syntax, &[("get_weather", &evil)])
            .expect_err("unrepresentable");
        assert!(matches!(err, DialectError::UnrepresentableValue { .. }));
        // ... nested inside a container too.
        let evil = json!({"city": "x", "days": 1,
            "extra": {"inner": ["fine", "<|\"|>"]}});
        assert!(validate_representable(&syntax, "get_weather", &evil).is_err());
        // Dict terminators in keys.
        let evil = json!({"bad:key": 1});
        assert!(validate_representable(&syntax, "t", &evil).is_err());
    }

    /// Streaming chunking invariance for the Gemma envelope,
    /// including channel noise around the thought.
    #[test]
    fn gemma4_stream_matches_batch_for_any_chunking() {
        let syntax = CallSyntax::gemma4();
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3});
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).expect("ok");
        let emission = format!(
            "<|channel><|channel>thought\nreason it out\n<channel|>\
             Sure thing.<channel|>{call}"
        );
        let batch = merge_text(
            parse_text(&syntax, &[&t], &emission, false, Leniency::Final)
                .blocks,
        );
        for chunk in 1..=7usize {
            let mut p =
                StreamParser::new(syntax.clone(), vec![t.clone()], false);
            let mut streamed = Vec::new();
            let bytes = emission.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                let mut j = (i + chunk).min(bytes.len());
                while !emission.is_char_boundary(j) {
                    j += 1;
                }
                streamed.extend(p.push(&emission[i..j]));
                i = j;
            }
            streamed.extend(p.finish());
            let streamed = merge_text(streamed);
            assert_eq!(
                streamed.len(),
                batch.len(),
                "chunk={chunk}: {streamed:#?} vs {batch:#?}"
            );
            for (s, b) in streamed.iter().zip(batch.iter()) {
                match (s, b) {
                    (
                        Block::Text { text: a, .. },
                        Block::Text { text: c, .. },
                    ) => assert_eq!(a, c, "chunk={chunk}"),
                    (
                        Block::Thought { thought: a, .. },
                        Block::Thought { thought: c, .. },
                    ) => assert_eq!(a, c, "chunk={chunk}"),
                    (
                        Block::ToolUse { call: a },
                        Block::ToolUse { call: c },
                    ) => {
                        assert_eq!(a.name, c.name, "chunk={chunk}");
                        assert_eq!(a.input, c.input, "chunk={chunk}");
                    }
                    other => panic!("chunk={chunk}: mismatch {other:?}"),
                }
            }
        }
    }

    /// Prefix-chop atomicity for the dict family (mirrors
    /// `streaming_prefixes_are_atomic`).
    #[test]
    fn gemma4_streaming_prefixes_are_atomic() {
        let syntax = CallSyntax::gemma4();
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3, "detail": "x"});
        let mut full =
            String::from("<|channel>thought\nhm\n<channel|>Calling.");
        full.push_str(
            &render_reference(&syntax, &[("get_weather", &input)]).unwrap(),
        );
        let parsed = parse_text(&syntax, &[&t], &full, false, Leniency::Final);
        assert_eq!(parsed.status, ParseStatus::Complete, "{parsed:#?}");
        assert_eq!(calls_of(&parsed.blocks).len(), 1);
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
            assert!(calls.len() <= 1, "prefix {i}: {:#?}", parsed.blocks);
            if let Some((name, input_got)) = calls.first() {
                assert_eq!(*name, "get_weather", "prefix {i}");
                assert_eq!(*input_got, &input, "prefix {i}");
            }
        }
    }

    // -----------------------------------------------------------------
    // Harmony / gpt-oss: upstream test-matrix port
    // (llama.cpp tests/test-chat.cpp:5075-5254, gpt-oss section).
    // -----------------------------------------------------------------

    /// Upstream's `special_function_tool`: one required integer.
    fn special_function() -> Tool {
        Tool::builder("special_function")
            .description("I'm special")
            .schema(json!({
                "type": "object",
                "properties": {"arg1": {
                    "type": "integer",
                    "description": "The arg."
                }},
                "required": ["arg1"],
            }))
            .build()
            .expect("valid test tool")
    }

    fn harmony_parse(text: &str, leniency: Leniency) -> Vec<Block> {
        let syntax = CallSyntax::gpt_oss();
        let t = special_function();
        parse_text(&syntax, &[&t], text, false, leniency).blocks
    }

    /// Every Harmony recipient arms the lazy grammar at its `to=`, and
    /// from there only `functions.` and a declared name are legal: the
    /// recipients gpt-oss-120b wrote instead (2026-10-01) — a bare tool
    /// name, which the parser swallowed as a builtin, and `function`
    /// followed by prose — are refused, so the model is steered to the
    /// call it meant. The canonical call is admitted from every trigger.
    #[test]
    fn harmony_triggers_arm_at_any_recipient() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        let syntax = CallSyntax::gpt_oss();
        let t = tool("get_weather");
        let source = grammar_source(
            &syntax,
            &[&t],
            &EmitOptions {
                anchor: Anchor::Lazy,
                ..Default::default()
            },
        )
        .expect("grammar");
        let admits = |text: &str| {
            let mut state = crate::GrammarState::from_source(&source)
                .expect("grammar parses");
            state.advance_bytes(text.as_bytes()).is_ok()
        };
        let args = r#"{"city":"Paris","days":3}"#;
        for trigger in syntax.triggers() {
            assert!(trigger.ends_with(" to="), "{trigger:?}");
            let header = match trigger.starts_with(harmony::START_ASSISTANT) {
                true => {
                    "functions.get_weather<|channel|>commentary \
                         <|constrain|>json<|message|>"
                }
                false => "functions.get_weather <|constrain|>json<|message|>",
            };
            let call = format!("{trigger}{header}{args}");
            assert!(admits(&call), "{call:?}");
            for stray in [
                "get_weather",
                "function\n\nOops need correct tool.",
                "functions.unknown",
                "assistant",
                "python",
            ] {
                let text = format!("{trigger}{stray}");
                assert!(!admits(&text), "{text:?}");
            }
        }
    }

    /// A recipient naming a declared tool without `functions.` is that
    /// tool's call, not a builtin to swallow: the call was lost whole.
    /// Undeclared recipients are still swallowed (upstream parity).
    #[test]
    fn harmony_bare_declared_recipient_is_a_call() {
        let t = tool("get_weather");
        let parse = |text: &str| {
            parse_text(&CallSyntax::gpt_oss(), &[&t], text, false, {
                Leniency::Final
            })
            .blocks
        };
        let args = r#"{"city":"Paris","days":3}"#;
        for header in [
            "<|channel|>commentary to=get_weather <|constrain|>json",
            " to=get_weather<|channel|>commentary json",
        ] {
            let text = format!("{header}<|message|>{args}<|call|>");
            let blocks = parse(&text);
            let calls = calls_of(&blocks);
            assert_eq!(calls.len(), 1, "{text:?}: {blocks:#?}");
            assert_eq!(calls[0].0, "get_weather");
            assert_eq!(calls[0].1["city"], "Paris");
        }
        let text = format!(
            "<|channel|>commentary to=python <|constrain|>json\
             <|message|>{args}<|call|>"
        );
        assert!(calls_of(&parse(&text)).is_empty());
    }

    /// A Harmony call's args end at the JSON close, on both sides:
    /// the grammar (eager and lazy) admits nothing after it but EOG,
    /// and the parser reads prose written past it — the model wrote
    /// `"}` and went on in the same message — as its own text, never
    /// as part of a value, non-ASCII right before the close included
    /// (the 2026-10-01 gpt-oss `create_comment` shape).
    #[test]
    fn harmony_args_end_at_the_json_close() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        let comment = Tool::builder("create_comment")
            .description("Post a comment.")
            .schema(json!({
                "type": "object",
                "properties": {
                    "reply_to": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["reply_to", "body"],
            }))
            .build()
            .expect("valid test tool");
        let header = "<|channel|>commentary to=functions.create_comment \
                      <|constrain|>json<|message|>";
        let args = "{\"reply_to\":\"7ad26ccd\",\
                    \"body\":\"Agreed (Art\u{202F}II\u{2011}6).\"}";
        let prose = "Will need to check. Let's proceed.";

        for anchor in [Anchor::Eager, Anchor::Lazy] {
            let source = grammar_source(
                &CallSyntax::gpt_oss(),
                &[&comment],
                &EmitOptions {
                    anchor,
                    ..Default::default()
                },
            )
            .expect("grammar");
            let mut state = crate::GrammarState::from_source(&source)
                .expect("grammar parses");
            let call = format!("{header}{args}");
            assert!(
                state.advance_bytes(call.as_bytes()).is_ok()
                    && state.is_complete(),
                "{anchor:?}: {call:?}"
            );
            assert!(
                state.advance_bytes(prose.as_bytes()).is_err(),
                "{anchor:?}: prose after the close"
            );
        }

        for tail in ["", "<|call|>"] {
            let text = format!("{header}{args}{prose}{tail}");
            for leniency in
                [Leniency::Final, Leniency::Streaming, Leniency::Clipped]
            {
                let blocks = parse_text(
                    &CallSyntax::gpt_oss(),
                    &[&comment],
                    &text,
                    false,
                    leniency,
                )
                .blocks;
                let calls = calls_of(&blocks);
                assert_eq!(calls.len(), 1, "{leniency:?}: {blocks:#?}");
                assert_eq!(
                    calls[0].1["body"], "Agreed (Art\u{202F}II\u{2011}6).",
                    "{leniency:?}: {blocks:#?}"
                );
                assert!(
                    matches!(blocks.last(), Some(Block::Text { text, .. })
                        if text.starts_with(prose)),
                    "{leniency:?}: {blocks:#?}"
                );
            }
        }
    }

    /// Content-only messages: final and commentary-preamble channels
    /// both surface as Text (upstream `message_assist`).
    #[test]
    fn harmony_content_channels() {
        for text in [
            "<|channel|>final<|message|>Hello, world!\nWhat's up?",
            "<|channel|>commentary<|message|>Hello, world!\nWhat's up?",
            // Trailing EOG pieces are envelope, not content.
            "<|channel|>final<|message|>Hello, world!\nWhat's up?<|return|>",
        ] {
            let blocks = merge_text(harmony_parse(text, Leniency::Final));
            assert_eq!(blocks.len(), 1, "{text:?}: {blocks:#?}");
            assert!(
                matches!(&blocks[0], Block::Text { text, .. }
                    if text.as_ref() == "Hello, world!\nWhat's up?"),
                "{text:?}: {blocks:#?}"
            );
        }
    }

    /// Reasoning then final content, including the stray-commentary
    /// wart before either block (the 20b prefix), and a partial
    /// analysis block under Final leniency (upstream pins reasoning
    /// for the cut-off case).
    #[test]
    fn harmony_analysis_blocks() {
        let want_thought = "I'm\nthinking";
        let want_text = "Hello, world!\nWhat's up?";
        for text in [
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>\
             <|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
            // Stray commentary before the analysis header.
            "<|channel|>commentary to=assistant<|channel|>analysis<|message|>I'm\nthinking<|end|>\
             <|start|>assistant<|channel|>final<|message|>Hello, world!\nWhat's up?",
            // Stray commentary before the final header.
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>\
             <|start|>assistant<|channel|>commentary<|channel|>final<|message|>Hello, world!\nWhat's up?",
        ] {
            let blocks = merge_text(harmony_parse(text, Leniency::Final));
            assert_eq!(blocks.len(), 2, "{text:?}: {blocks:#?}");
            assert!(
                matches!(&blocks[0], Block::Thought { thought, .. }
                    if thought.as_ref() == want_thought),
                "{text:?}: {blocks:#?}"
            );
            assert!(
                matches!(&blocks[1], Block::Text { text, .. }
                    if text.as_ref() == want_text),
                "{text:?}: {blocks:#?}"
            );
        }

        // Cut off mid-reasoning: Final surfaces the partial Thought,
        // flagged OPEN. Harmony can't render an open thought back (its
        // generation prompt never pre-opens a channel), so the flag is
        // what turns a silent re-render with a fabricated `<|end|>`
        // into a loud rejection at ingest.
        let blocks = merge_text(harmony_parse(
            "<|channel|>analysis<|message|>I'm\nthinking",
            Leniency::Final,
        ));
        assert_eq!(blocks.len(), 1, "{blocks:#?}");
        assert!(
            matches!(&blocks[0], Block::Thought { thought, .. }
                if thought.as_ref() == want_thought),
            "{blocks:#?}"
        );
        assert!(crate::prompt::is_open_thought(&blocks[0]), "{blocks:#?}");
        // The closed sibling above must NOT be flagged.
        let closed = merge_text(harmony_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>",
            Leniency::Final,
        ));
        assert!(!crate::prompt::is_open_thought(&closed[0]), "{closed:#?}");

        // Multiple analysis blocks stay separate Thoughts, in order.
        let blocks = merge_text(harmony_parse(
            "<|channel|>analysis<|message|>one<|end|>\
             <|start|>assistant<|channel|>analysis<|message|>two<|end|>\
             <|start|>assistant<|channel|>final<|message|>done",
            Leniency::Final,
        ));
        assert_eq!(blocks.len(), 3, "{blocks:#?}");
        assert!(matches!(&blocks[0], Block::Thought { thought, .. }
            if thought.as_ref() == "one"));
        assert!(matches!(&blocks[1], Block::Thought { thought, .. }
            if thought.as_ref() == "two"));
    }

    /// The tool-call header matrix, byte-for-byte from upstream's
    /// pins: both recipient positions, optional `<|constrain|>`, the
    /// bare-type form the stock template re-renders, and a preceding
    /// analysis block.
    #[test]
    fn harmony_call_headers() {
        for text in [
            // Recipient in role header, analysis channel.
            " to=functions.special_function<|channel|>analysis<|message|>{\"arg1\": 1}",
            // Recipient in channel header.
            "<|channel|>analysis to=functions.special_function<|message|>{\"arg1\": 1}",
            // With <|constrain|>json.
            " to=functions.special_function<|channel|>analysis <|constrain|>json<|message|>{\"arg1\": 1}",
            // Commentary channel.
            "<|channel|>commentary to=functions.special_function<|message|>{\"arg1\": 1}",
            // Channel header + constraint (canonical emission shape).
            "<|channel|>commentary to=functions.special_function <|constrain|>json<|message|>{\"arg1\": 1}",
            // Stock-template re-render: role header, bare type, <|call|>.
            "<|start|>assistant to=functions.special_function<|channel|>commentary json<|message|>{\"arg1\": 1}<|call|>",
        ] {
            let blocks = harmony_parse(text, Leniency::Final);
            let calls = calls_of(&blocks);
            assert_eq!(calls.len(), 1, "{text:?}: {blocks:#?}");
            assert_eq!(calls[0].0, "special_function", "{text:?}");
            assert_eq!(calls[0].1, &json!({"arg1": 1}), "{text:?}");
            // Header/envelope bytes never leak into Text.
            assert!(
                !blocks.iter().any(|b| matches!(b, Block::Text { .. })),
                "{text:?}: {blocks:#?}"
            );
        }

        // Reasoning then call (upstream message_assist_call_thoughts).
        let blocks = harmony_parse(
            "<|channel|>analysis<|message|>I'm\nthinking<|end|>\
             <|start|>assistant to=functions.special_function<|channel|>analysis<|message|>{\"arg1\": 1}",
            Leniency::Final,
        );
        assert!(
            matches!(&blocks[0], Block::Thought { thought, .. }
                if thought.as_ref() == "I'm\nthinking"),
            "{blocks:#?}"
        );
        assert_eq!(calls_of(&blocks).len(), 1, "{blocks:#?}");
    }

    /// Unsolicited builtin-tool traffic (recipients outside
    /// `functions.`) is swallowed whole — upstream surfaces empty
    /// content for these.
    #[test]
    fn harmony_builtin_recipients_swallowed() {
        for text in [
            // Recipient in role header.
            "<|channel|>analysis<|message|>thinking<|end|>\
             <|start|>assistant to=container.exec<|channel|>commentary<|message|>python3 -c 'print(\"hello\")'",
            // Recipient in channel header, code constraint.
            "<|channel|>analysis<|message|>thinking<|end|>\
             <|start|>assistant<|channel|>commentary to=python <|constrain|>code<|message|>print(\"hello\")",
        ] {
            let blocks = harmony_parse(text, Leniency::Final);
            assert_eq!(blocks.len(), 1, "{text:?}: {blocks:#?}");
            assert!(
                matches!(&blocks[0], Block::Thought { thought, .. }
                    if thought.as_ref() == "thinking"),
                "{text:?}: {blocks:#?}"
            );
        }
    }

    /// Prose preamble (commentary, no recipient) before a call:
    /// the causal announce-then-call shape.
    #[test]
    fn harmony_preamble_then_call() {
        let blocks = harmony_parse(
            "<|channel|>analysis<|message|>plan<|end|>\
             <|start|>assistant<|channel|>commentary<|message|>I'll use the tool.<|end|>\
             <|start|>assistant<|channel|>commentary to=functions.special_function \
             <|constrain|>json<|message|>{\"arg1\": 7}<|call|>",
            Leniency::Final,
        );
        assert_eq!(blocks.len(), 3, "{blocks:#?}");
        assert!(matches!(&blocks[0], Block::Thought { thought, .. }
            if thought.as_ref() == "plan"));
        assert!(matches!(&blocks[1], Block::Text { text, .. }
            if text.as_ref() == "I'll use the tool."));
        let calls = calls_of(&blocks);
        assert_eq!(calls[0].1, &json!({"arg1": 7}), "{blocks:#?}");
    }

    /// Final-mode leniency: a dangling call degrades to Text carrying
    /// the header bytes (the BlockParser::finish contract).
    #[test]
    fn harmony_partial_call_degrades_to_text() {
        let blocks = harmony_parse(
            "<|channel|>commentary to=functions.special_function \
             <|constrain|>json<|message|>{\"arg1\": ",
            Leniency::Final,
        );
        assert!(calls_of(&blocks).is_empty(), "{blocks:#?}");
        let joined: String = blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text, .. } => Some(text.to_string()),
                _ => None,
            })
            .collect();
        assert!(joined.contains("to=functions."), "{blocks:#?}");
    }

    /// Reference render → parse is the identity on calls (the Phase D
    /// invariant, Harmony edition), and the emitted grammar accepts
    /// its own reference render under both anchors.
    #[test]
    fn harmony_reference_roundtrip_and_grammar() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        use crate::{Grammar, GrammarState};
        use std::sync::Arc;

        let syntax = CallSyntax::gpt_oss();
        let t = special_function();
        let input = json!({"arg1": 42});
        let reference =
            render_reference(&syntax, &[("special_function", &input)])
                .expect("representable");
        assert_eq!(
            reference,
            "<|channel|>commentary to=functions.special_function \
             <|constrain|>json<|message|>{\"arg1\":42}"
        );
        let parsed = harmony_parse(&reference, Leniency::Final);
        let calls = calls_of(&parsed);
        assert_eq!(calls.len(), 1, "{parsed:#?}");
        assert_eq!(calls[0].1, &input);

        // Lazy grammar (Auto): accepts the canonical channel form AND
        // the role-header form the stock template re-renders.
        let lazy = grammar_source(
            &syntax,
            &[&t],
            &EmitOptions {
                anchor: Anchor::Lazy,
                parallel: false,
                ..Default::default()
            },
        )
        .expect("emit lazy");
        let grammar = Arc::new(
            Grammar::parse(&lazy)
                .unwrap_or_else(|e| panic!("lazy grammar: {e}\n{lazy}")),
        );
        for emission in [
            reference.as_str(),
            "<|start|>assistant to=functions.special_function\
             <|channel|>commentary <|constrain|>json<|message|>{\"arg1\":42}",
            // Stock re-render: bare constraint type. The JSON is
            // compact like the other two — `tojson` no longer
            // HTML-escapes *or* spaces, and the grammar admits exactly
            // one spelling now (#85). The variation under test here is
            // the bare `json` constraint, not the whitespace.
            "<|start|>assistant to=functions.special_function\
             <|channel|>commentary json<|message|>{\"arg1\":42}",
        ] {
            let mut state = GrammarState::new(grammar.clone());
            assert!(
                state.advance_bytes(emission.as_bytes()).is_ok()
                    && state.is_complete(),
                "lazy grammar must accept {emission:?}\n{lazy}"
            );
        }

        // Eager grammar (Any/Method): analysis and preamble blocks,
        // then the forced canonical call.
        let eager = grammar_source(
            &syntax,
            &[&t],
            &EmitOptions {
                anchor: Anchor::Eager,
                parallel: false,
                ..Default::default()
            },
        )
        .expect("emit eager");
        let grammar = Arc::new(
            Grammar::parse(&eager)
                .unwrap_or_else(|e| panic!("eager grammar: {e}\n{eager}")),
        );
        let emission = format!(
            "<|channel|>analysis<|message|>let me think<|end|>\
             <|start|>assistant<|channel|>commentary<|message|>calling now\
             <|end|><|start|>assistant{reference}"
        );
        let mut state = GrammarState::new(grammar.clone());
        assert!(
            state.advance_bytes(emission.as_bytes()).is_ok()
                && state.is_complete(),
            "eager grammar must accept {emission:?}\n{eager}"
        );
        // ... and the call alone (model may skip reasoning).
        let mut state = GrammarState::new(grammar);
        assert!(
            state.advance_bytes(reference.as_bytes()).is_ok()
                && state.is_complete(),
            "eager grammar must accept the bare call\n{eager}"
        );
    }

    /// The eager (`Any`/`Method`) Harmony grammar admits at most one
    /// analysis block and one preamble before the forced call. EOG is
    /// illegal until the call completes and `final` is never offered,
    /// so an unbounded block loop let gpt-oss-120b — forced to call
    /// after it had already answered — alternate analysis and
    /// commentary to `max_tokens` (2026-09-30, the live shape below).
    #[test]
    fn harmony_eager_grammar_bounds_blocks_before_the_call() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        use crate::{Grammar, GrammarState};
        use std::sync::Arc;

        let syntax = CallSyntax::gpt_oss();
        let t = special_function();
        let call = render_reference(
            &syntax,
            &[("special_function", &json!({"arg1": 1}))],
        )
        .expect("representable");
        for anchor in [Anchor::Eager, Anchor::EagerThoughtPreOpened] {
            let src = grammar_source(
                &syntax,
                &[&t],
                &EmitOptions {
                    anchor,
                    parallel: false,
                    ..Default::default()
                },
            )
            .expect("emit eager");
            let grammar = Arc::new(
                Grammar::parse(&src)
                    .unwrap_or_else(|e| panic!("eager grammar: {e}\n{src}")),
            );
            let accepts = |text: &str| {
                let mut state = GrammarState::new(grammar.clone());
                state.advance_bytes(text.as_bytes()).is_ok()
            };
            let analysis = "<|channel|>analysis<|message|>Now say pong.\
                            <|end|><|start|>assistant";
            let preamble = "<|channel|>commentary<|message|>pong\
                            <|end|><|start|>assistant";

            // The runaway, block by block: each prefix up to the
            // second block of either kind is refused.
            for text in [
                format!("{analysis}{preamble}{analysis}"),
                format!("{analysis}{preamble}{preamble}"),
                format!("{analysis}{analysis}"),
                format!("{preamble}{preamble}"),
                format!("{preamble}{analysis}"),
            ] {
                assert!(!accepts(&text), "{anchor:?}: must refuse {text:?}");
            }
            // `final` was never a way out under a forced call, and
            // still isn't — the call is the only exit.
            assert!(!accepts(&format!(
                "{analysis}<|channel|>final<|message|>pong"
            )));
            // What it may do: each block at most once, in trained
            // order, then the call — the whole of it complete.
            for text in [
                format!("{analysis}{preamble}{call}"),
                format!("{analysis}{call}"),
                format!("{preamble}{call}"),
                call.clone(),
            ] {
                let mut state = GrammarState::new(grammar.clone());
                assert!(
                    state.advance_bytes(text.as_bytes()).is_ok()
                        && state.is_complete(),
                    "{anchor:?}: must accept {text:?}\n{src}"
                );
            }
        }
    }

    /// The class, across every call dialect that reasons (Qwen XML,
    /// Gemma 4, Harmony): under a forced call (eager anchor) a closed
    /// thought is followed by the calls, never by another thought — or
    /// the model can reopen one forever instead of calling, and no end
    /// of turn is ever legal.
    #[test]
    fn eager_grammar_admits_one_thought_before_the_calls() {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        use crate::{Grammar, GrammarState};
        use std::sync::Arc;

        let t = tool("get_weather");
        for (name, syntax) in call_dialects() {
            let (open, close) =
                (syntax.reasoning.start.as_str(), syntax.reasoning.end.trim());
            if open.trim().is_empty() || close.is_empty() {
                continue;
            }
            let src = grammar_source(
                &syntax,
                &[&t],
                &EmitOptions {
                    anchor: Anchor::Eager,
                    parallel: true,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            let grammar = Arc::new(
                Grammar::parse(&src)
                    .unwrap_or_else(|e| panic!("{name}: {e}\n{src}")),
            );
            // Harmony reopens with its role header between blocks.
            let reopen = match syntax.family {
                Family::Harmony => harmony::START_ASSISTANT,
                _ => "",
            };
            let sep = syntax.reasoning.separator.as_deref().unwrap_or("");
            let text = format!("{open}one{close}{sep}{reopen}{open}two");
            let mut state = GrammarState::new(grammar);
            assert!(
                state.advance_bytes(text.as_bytes()).is_err(),
                "{name}: a second thought must be refused: {text:?}\n{src}"
            );
        }
    }

    /// `ToolChoice::None` never constrains a Harmony turn: no grammar
    /// resolves (the dialect has no opener special to ban either), so
    /// the final channel's `<|return|>` — EOG — stays reachable and a
    /// final answer parses to a closed thought and the text, the turn
    /// complete (`end_turn`).
    #[test]
    fn harmony_final_message_ends_the_turn() {
        let text = "<|channel|>analysis<|message|>The budget is spent; \
                    answer in words.<|end|><|start|>assistant\
                    <|channel|>final<|message|>pong<|return|>";
        let parsed = parse_text(
            &CallSyntax::gpt_oss(),
            &[&special_function()],
            text,
            false,
            Leniency::Final,
        );
        assert_eq!(parsed.status, ParseStatus::Complete, "{parsed:#?}");
        match parsed.blocks.as_slice() {
            [Block::Thought { thought, signature }, Block::Text { text, .. }] =>
            {
                assert_eq!(thought, "The budget is spent; answer in words.");
                assert_ne!(
                    signature.as_ref(),
                    crate::prompt::OPEN_THOUGHT_SIGNATURE,
                    "the thought closed"
                );
                assert_eq!(text, "pong");
            }
            other => panic!("expected [Thought, Text], got {other:#?}"),
        }
    }

    /// Streaming chunking invariance for the Harmony envelope
    /// (mirrors the Gemma/Qwen invariance pins), plus prefix-chop
    /// atomicity for a call emission.
    #[test]
    fn harmony_stream_matches_batch_for_any_chunking() {
        let syntax = CallSyntax::gpt_oss();
        let t = special_function();
        let emission = "<|channel|>analysis<|message|>reason it out<|end|>\
             <|start|>assistant<|channel|>commentary<|message|>Sure thing.<|end|>\
             <|start|>assistant<|channel|>commentary to=functions.special_function \
             <|constrain|>json<|message|>{\"arg1\":3}<|call|>";
        let batch = merge_text(
            parse_text(&syntax, &[&t], emission, false, Leniency::Final).blocks,
        );
        assert_eq!(calls_of(&batch).len(), 1, "{batch:#?}");
        for chunk in 1..=7usize {
            let mut p =
                StreamParser::new(syntax.clone(), vec![t.clone()], false);
            let mut streamed = Vec::new();
            let mut i = 0;
            while i < emission.len() {
                let mut j = (i + chunk).min(emission.len());
                while !emission.is_char_boundary(j) {
                    j += 1;
                }
                streamed.extend(p.push(&emission[i..j]));
                i = j;
            }
            streamed.extend(p.finish());
            let streamed = merge_text(streamed);
            assert_eq!(
                streamed.len(),
                batch.len(),
                "chunk={chunk}: {streamed:#?} vs {batch:#?}"
            );
            for (s, b) in streamed.iter().zip(batch.iter()) {
                match (s, b) {
                    (
                        Block::Text { text: a, .. },
                        Block::Text { text: c, .. },
                    ) => assert_eq!(a, c, "chunk={chunk}"),
                    (
                        Block::Thought { thought: a, .. },
                        Block::Thought { thought: c, .. },
                    ) => assert_eq!(a, c, "chunk={chunk}"),
                    (
                        Block::ToolUse { call: a },
                        Block::ToolUse { call: c },
                    ) => {
                        assert_eq!(a.name, c.name, "chunk={chunk}");
                        assert_eq!(a.input, c.input, "chunk={chunk}");
                    }
                    other => panic!("chunk={chunk}: mismatch {other:?}"),
                }
            }
        }

        // Prefix-chop atomicity: no prefix ever surfaces a call that
        // the full parse doesn't have, and never panics.
        for i in 0..=emission.len() {
            if !emission.is_char_boundary(i) {
                continue;
            }
            let parsed = parse_text(
                &syntax,
                &[&t],
                &emission[..i],
                false,
                Leniency::Streaming,
            );
            let calls = calls_of(&parsed.blocks);
            assert!(calls.len() <= 1, "prefix {i}: {:#?}", parsed.blocks);
            if let Some((name, input)) = calls.first() {
                assert_eq!(*name, "special_function", "prefix {i}");
                assert_eq!(*input, &json!({"arg1": 3}), "prefix {i}");
            }
        }
    }

    /// Final-channel content streams incrementally as prose deltas —
    /// the block users watch — with marker prefixes held back.
    #[test]
    fn harmony_final_content_streams() {
        let syntax = CallSyntax::gpt_oss();
        let t = special_function();
        let mut p = StreamParser::new(syntax, vec![t], false);
        assert!(p.push("<|channel|>analysis<|message|>hm").is_empty());
        let out =
            p.push("<|end|><|start|>assistant<|channel|>final<|message|>Hello");
        assert_eq!(out.len(), 2, "{out:#?}");
        assert!(matches!(&out[0], Block::Thought { thought, .. }
            if thought.as_ref() == "hm"));
        assert!(matches!(&out[1], Block::Text { text, .. }
            if text.as_ref() == "Hello"));
        let out = p.push(", world");
        assert_eq!(merge_text(out), vec![Block::from(", world".to_string())]);
        // A partial <|return|> is held back...
        let out = p.push("!<|ret");
        assert_eq!(merge_text(out), vec![Block::from("!".to_string())]);
        // ...and swallowed once complete.
        let out = p.push("urn|>");
        assert!(out.is_empty(), "{out:#?}");
        assert!(p.finish().is_empty());
    }

    // -----------------------------------------------------------------
    // StreamParser: re-parse-per-tick diffing.
    // -----------------------------------------------------------------

    /// Collapse adjacent Text blocks so chunking granularity doesn't
    /// affect comparisons (the stream deliberately fragments prose).
    fn merge_text(blocks: Vec<Block>) -> Vec<Block> {
        let mut out: Vec<Block> = Vec::new();
        for block in blocks {
            match (out.last_mut(), block) {
                (
                    Some(Block::Text { text: prev, .. }),
                    Block::Text { text: new, .. },
                ) => {
                    let merged = format!("{prev}{new}");
                    *prev = merged.into();
                }
                (_, block) => out.push(block),
            }
        }
        out
    }

    #[test]
    fn stream_prose_yields_per_push() {
        let mut p =
            StreamParser::new(CallSyntax::qwen_xml(), vec![tool("t")], false);
        let a = p.push("Hello ");
        assert_eq!(merge_text(a), vec![Block::from("Hello ".to_string())]);
        let b = p.push("world");
        assert_eq!(merge_text(b), vec![Block::from("world".to_string())]);
        assert!(p.finish().is_empty());
    }

    /// A prose tail that could still grow into the call trigger is
    /// held back until disambiguated — the streaming soundness
    /// property (never yield bytes a longer parse might re-classify).
    #[test]
    fn stream_holds_back_trigger_prefix() {
        let mut p =
            StreamParser::new(CallSyntax::qwen_xml(), vec![tool("t")], false);
        let a = p.push("hi <tool");
        assert_eq!(
            merge_text(a),
            vec![Block::from("hi ".to_string())],
            "`<tool` must be held back pending disambiguation"
        );
        let b = p.push("box");
        assert_eq!(merge_text(b), vec![Block::from("<toolbox".to_string())]);
    }

    /// Held-back trigger-prefix bytes flush at finish when the marker
    /// never completed.
    #[test]
    fn stream_flushes_holdback_at_finish() {
        let mut p =
            StreamParser::new(CallSyntax::qwen_xml(), vec![tool("t")], false);
        let a = p.push("hi <tool");
        assert_eq!(merge_text(a), vec![Block::from("hi ".to_string())]);
        let b = p.finish();
        assert_eq!(merge_text(b), vec![Block::from("<tool".to_string())]);
    }

    /// Pre-opened reasoning buffers until the close marker, then
    /// yields one Thought — never Text (the #27 fix, streaming side).
    #[test]
    fn stream_pre_opened_thought_buffers_until_close() {
        let mut p =
            StreamParser::new(CallSyntax::qwen_xml(), vec![tool("t")], true);
        assert!(p.push("planning...").is_empty());
        let out = p.push("\n</think>\n\nHello");
        assert_eq!(out.len(), 2, "expected Thought + Text, got {out:?}");
        assert!(
            matches!(&out[0], Block::Thought { thought, .. } if thought == "planning..."),
            "got {out:?}"
        );
        assert!(matches!(&out[1], Block::Text { .. }));
    }

    /// Chunking invariance: for every chunk size, streaming a full
    /// thought + prose + tool-call emission yields the same blocks as
    /// one Final batch parse. Pins both diff properties (stable block
    /// prefix, append-only trailing Text).
    #[test]
    fn stream_matches_batch_for_any_chunking() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let input = serde_json::json!({"city": "Paris", "days": 3});
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).expect("ok");
        let emission =
            format!("<think>\nreason it out\n</think>\n\nSure thing.\n{call}");
        let batch = merge_text(
            parse_text(&syntax, &[&t], &emission, false, Leniency::Final)
                .blocks,
        );
        for chunk in 1..=7usize {
            let mut p =
                StreamParser::new(syntax.clone(), vec![t.clone()], false);
            let mut streamed = Vec::new();
            let bytes = emission.as_bytes();
            let mut i = 0;
            while i < bytes.len() {
                let mut j = (i + chunk).min(bytes.len());
                while !emission.is_char_boundary(j) {
                    j += 1;
                }
                streamed.extend(p.push(&emission[i..j]));
                i = j;
            }
            streamed.extend(p.finish());
            let streamed = merge_text(streamed);
            assert_eq!(
                streamed.len(),
                batch.len(),
                "chunk={chunk}: {streamed:#?} vs {batch:#?}"
            );
            for (s, b) in streamed.iter().zip(batch.iter()) {
                match (s, b) {
                    (
                        Block::Text { text: a, .. },
                        Block::Text { text: c, .. },
                    ) => assert_eq!(a, c, "chunk={chunk}"),
                    (
                        Block::Thought { thought: a, .. },
                        Block::Thought { thought: c, .. },
                    ) => assert_eq!(a, c, "chunk={chunk}"),
                    (
                        Block::ToolUse { call: a },
                        Block::ToolUse { call: c },
                    ) => {
                        assert_eq!(a.name, c.name, "chunk={chunk}");
                        assert_eq!(a.input, c.input, "chunk={chunk}");
                    }
                    other => panic!("chunk={chunk}: mismatch {other:?}"),
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Leniency::Clipped — a generation cut short (#121, #122).
    // -----------------------------------------------------------------

    fn texts_of(blocks: &[Block]) -> String {
        blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text, .. } => Some(text.as_ref()),
                _ => None,
            })
            .collect()
    }

    /// Mistral Small 4's `[TOOL_CALLS]name[ARGS]{…}` — what the
    /// analyzer derives for it (`tests/dialect_analyzer.rs`).
    fn mistral_tag_json() -> CallSyntax {
        CallSyntax {
            family: Family::TagWithJson,
            per_call_start: "[TOOL_CALLS]".into(),
            function: crate::dialect::FunctionSyntax {
                name_prefix: String::new(),
                name_suffix: "[ARGS]".into(),
                close: String::new(),
            },
            ..CallSyntax::default()
        }
    }

    /// Every dialect with tool calls, as a test names it.
    fn call_dialects() -> [(&'static str, CallSyntax); 5] {
        [
            ("qwen_xml", CallSyntax::qwen_xml()),
            ("hermes_json", CallSyntax::hermes_json()),
            ("mistral", mistral_tag_json()),
            ("gemma4", CallSyntax::gemma4()),
            ("harmony", CallSyntax::gpt_oss()),
        ]
    }

    /// Prose to put before a call: Harmony carries none beside one.
    fn prose_for(syntax: &CallSyntax) -> &'static str {
        match syntax.family {
            Family::Harmony => "",
            _ => "Let me check. ",
        }
    }

    /// Parse `text` clipped, and stream it a char at a time then flush
    /// clipped: the two must agree block for block (prose merged), and
    /// on the call left open. Returns the batch parse and its open call.
    fn clipped_both_ways(
        syntax: &CallSyntax,
        tool: &Tool,
        text: &str,
    ) -> (Parsed, Option<OpenCall>) {
        let (parsed, open) =
            parse_text_open(syntax, &[tool], text, false, Leniency::Clipped);
        let mut p =
            StreamParser::new(syntax.clone(), vec![tool.clone()], false);
        let mut streamed: Vec<Block> = text
            .chars()
            .flat_map(|c| p.push(c.encode_utf8(&mut [0; 4])))
            .collect();
        streamed.extend(p.finish_clipped());
        assert_eq!(
            merge_text(streamed),
            merge_text(parsed.blocks.clone()),
            "{:?}: stream != batch on {text:?}",
            syntax.family,
        );
        assert_eq!(p.open(), open.as_ref(), "{:?}: {text:?}", syntax.family);
        (parsed, open)
    }

    /// Cut anywhere inside a call, the call comes back as Anthropic
    /// returns one a clip cut (#121): its input holds exactly the members
    /// that completed, in order, the one in flight dropped whole — or,
    /// before its name is whole, it is withheld. None of its bytes are
    /// ever seated as prose (the frame marker would poison the next
    /// ingest), the prose before it stands, and the stream agrees.
    #[test]
    fn clipped_returns_the_completed_members_of_a_cut_call() {
        let input = json!({"city": "Paris", "days": 3, "detail": "sunny"});
        let members: Vec<_> = input.as_object().unwrap().iter().collect();
        let t = tool("get_weather");
        for (name, syntax) in call_dialects() {
            let prose = prose_for(&syntax);
            let call =
                render_reference(&syntax, &[("get_weather", &input)]).unwrap();
            let full = format!("{prose}{call}");
            let closers = [
                syntax.per_call_end.trim(),
                syntax.section_end.trim(),
                syntax.function.close.trim(),
            ]
            .into_iter()
            .filter(|m| !m.is_empty())
            .collect::<Vec<_>>();
            let mut seen = [false; 4];
            let mut last_count = 0;
            // From the whole trigger on: a trigger is one special token,
            // so a clip never leaves half of one.
            let from = prose.len() + syntax.trigger().len().max(1);
            for i in from..=full.len() {
                if !full.is_char_boundary(i) {
                    continue;
                }
                let cut = &full[..i];
                let (parsed, open) = clipped_both_ways(&syntax, &t, cut);
                // At most a close marker's first byte: markers are one
                // special token each, so no real clip leaves that.
                let text = texts_of(&parsed.blocks);
                let extra = text.trim_end().strip_prefix(prose.trim_end());
                let extra = extra.map(str::trim).unwrap_or("?");
                assert!(
                    extra.is_empty()
                        || closers.iter().any(|m| m.starts_with(extra)),
                    "{name} cut at {i}: call bytes leaked into prose: {text:?}",
                );
                let calls = calls_of(&parsed.blocks);
                assert!(calls.len() <= 1, "{name} cut at {i}: {calls:?}");
                let Some((call_name, got)) = calls.first() else {
                    assert!(open.is_none(), "{name} cut at {i}");
                    assert_eq!(last_count, 0, "{name} cut at {i}: call lost");
                    continue;
                };
                assert_eq!(*call_name, "get_weather", "{name} cut at {i}");
                let got = got.as_object().expect("an object");
                // A prefix of the members, each whole.
                assert!(got.len() >= last_count, "{name} cut at {i}");
                for ((k, v), (want_k, want_v)) in got.iter().zip(&members) {
                    assert_eq!((k, v), (*want_k, *want_v), "{name} cut at {i}");
                }
                last_count = got.len();
                seen[got.len()] = true;
                match (parsed.status, &open) {
                    (ParseStatus::NeedMoreInput, Some(open)) => {
                        // What the stream would send before the cut
                        // parses, closed, to the same input.
                        let partial = open.partial_json.as_deref().unwrap();
                        let closers = "}".repeat(
                            partial.matches('{').count()
                                - partial.matches('}').count(),
                        );
                        let closed: Value = serde_json::from_str(&format!(
                            "{partial}{closers}"
                        ))
                        .unwrap();
                        assert_eq!(&closed, calls[0].1, "{name} cut at {i}");
                    }
                    // Closed, or only a section closer to come.
                    (_, None) => {
                        assert_eq!(got.len(), 3, "{name} cut at {i}");
                    }
                    other => panic!("{name} cut at {i}: {other:?}"),
                }
            }
            assert_eq!(seen, [true; 4], "{name}: every member count occurs");
        }
    }

    /// The captured points, per dialect: cut before any member, the
    /// input is `{}`; mid the first, still `{}`; mid the second after a
    /// complete first, the first alone.
    #[test]
    fn clipped_at_the_captured_points() {
        let input = json!({"city": "Paris", "detail": "sunny spells"});
        let t = tool("get_weather");
        for (name, syntax) in call_dialects() {
            let full = format!(
                "{}{}",
                prose_for(&syntax),
                render_reference(&syntax, &[("get_weather", &input)]).unwrap(),
            );
            let at =
                |needle: &str, past: usize| full.find(needle).unwrap() + past;
            for (i, want) in [
                (at("city", 0), json!({})),
                (at("Paris", 3), json!({})),
                (at("sunny", 3), json!({"city": "Paris"})),
            ] {
                let (parsed, _) = clipped_both_ways(&syntax, &t, &full[..i]);
                assert_eq!(
                    calls_of(&parsed.blocks),
                    [("get_weather", &want)],
                    "{name}: cut {:?}",
                    &full[..i],
                );
                assert_eq!(parsed.status, ParseStatus::NeedMoreInput);
            }
        }
    }

    /// Nested containers (inferred): the completed members kept at every
    /// depth, the one in flight dropped. Qwen XML carries a non-string
    /// parameter as one JSON value, so there the whole member is in
    /// flight until its close marker.
    #[test]
    fn clipped_nested_input_keeps_completed_members_at_every_depth() {
        let t = Tool::builder("get_weather")
            .description("test")
            .schema(json!({
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "opts": {"type": "object"},
                },
            }))
            .build()
            .unwrap();
        // Keys in sorted order: Gemma's dict renders nested maps sorted.
        let input = json!({
            "city": "Paris",
            "opts": {"lang": "french", "units": "metric"},
        });
        for (name, syntax) in call_dialects() {
            let full = format!(
                "{}{}",
                prose_for(&syntax),
                render_reference(&syntax, &[("get_weather", &input)]).unwrap(),
            );
            let cut = &full[..full.find("metric").unwrap() + 3];
            let (parsed, open) = clipped_both_ways(&syntax, &t, cut);
            let want = match syntax.family {
                Family::TagWithTagged => json!({"city": "Paris"}),
                _ => json!({"city": "Paris", "opts": {"lang": "french"}}),
            };
            assert_eq!(
                calls_of(&parsed.blocks),
                [("get_weather", &want)],
                "{name}"
            );
            let open = open.expect("open");
            let partial = open.partial_json.unwrap();
            match syntax.family {
                Family::TagWithTagged => {
                    assert_eq!(partial, r#"{"city":"Paris""#, "{name}")
                }
                _ => assert_eq!(
                    partial, r#"{"city":"Paris","opts":{"lang":"french""#,
                    "{name}",
                ),
            }
            // A stop is matched against the string in flight.
            let units = open.held_input.pointer("/opts/units");
            match syntax.family {
                Family::TagWithTagged => assert_eq!(units, None, "{name}"),
                _ => assert_eq!(units, Some(&json!("met")), "{name}"),
            }
        }
    }

    /// Parallel calls: the ones that closed before the cut stand
    /// unchanged; only the one in flight is cut short.
    #[test]
    fn clipped_parallel_calls_cut_only_the_last() {
        let t = tool("get_weather");
        let a = json!({"city": "Paris", "detail": "one"});
        let b = json!({"city": "Oslo", "detail": "two"});
        for (name, syntax) in call_dialects() {
            let render = |calls: &[(&str, &Value)]| {
                render_reference(&syntax, calls).unwrap()
            };
            // Hermes has no per-call opener: one section per call.
            let calls = match syntax.family {
                Family::JsonNative => format!(
                    "{}\n{}",
                    render(&[("get_weather", &a)]),
                    render(&[("get_weather", &b)]),
                ),
                _ => render(&[("get_weather", &a), ("get_weather", &b)]),
            };
            let full = format!("{}{calls}", prose_for(&syntax));
            let cut = &full[..full.find("two").unwrap() + 1];
            let (parsed, _) = clipped_both_ways(&syntax, &t, cut);
            assert_eq!(
                calls_of(&parsed.blocks),
                [
                    ("get_weather", &a),
                    ("get_weather", &json!({"city": "Oslo"}))
                ],
                "{name}: {parsed:#?}",
            );
            // The ids are the parse order's, cut call included.
            let ids: Vec<_> = parsed
                .blocks
                .iter()
                .filter_map(|b| match b {
                    Block::ToolUse { call } => Some(call.id.as_ref()),
                    _ => None,
                })
                .collect();
            assert_eq!(ids, ["call_0_get_weather", "call_1_get_weather"]);
        }
    }

    /// A call whose input closed but whose dialect close marker the cut
    /// took keeps its full input.
    #[test]
    fn clipped_before_the_close_marker_keeps_the_whole_input() {
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3});
        let mut syntax = mistral_tag_json();
        syntax.function.close = "</fn>".into();
        let text = r#"[TOOL_CALLS]get_weather[ARGS]{"city":"Paris","days":3}"#;
        let (parsed, open) = clipped_both_ways(&syntax, &t, text);
        assert_eq!(calls_of(&parsed.blocks), [("get_weather", &input)]);
        assert_eq!(
            open.unwrap().partial_json.as_deref(),
            Some(r#"{"city":"Paris","days":3}"#),
        );
        // Qwen XML, cut inside `</function>`.
        let syntax = CallSyntax::qwen_xml();
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).unwrap();
        let cut = &call[..call.find("</function>").unwrap() + 4];
        let (parsed, _) = clipped_both_ways(&syntax, &t, cut);
        assert_eq!(calls_of(&parsed.blocks), [("get_weather", &input)]);
    }

    /// Array-wrapped JSON calls hold every element back until the array
    /// closes; cut, the elements that closed come back whole, the one in
    /// flight cut short — or, cut between elements, nothing is left open.
    #[test]
    fn clipped_array_wrapped_calls() {
        let syntax = CallSyntax {
            family: Family::JsonNative,
            section_start: "[TOOL_CALLS]".into(),
            json: crate::dialect::JsonFields {
                tools_array_wrapped: true,
                ..Default::default()
            },
            ..CallSyntax::default()
        };
        let t = tool("get_weather");
        let one = r#"{"name": "get_weather", "arguments": {"city": "Paris"}}"#;
        let text = format!(
            r#"[TOOL_CALLS][{one}, {{"name": "get_weather", "arguments": {{"city": "Oslo", "detail": "cl"#
        );
        let (parsed, open) = clipped_both_ways(&syntax, &t, &text);
        assert_eq!(
            calls_of(&parsed.blocks),
            [
                ("get_weather", &json!({"city": "Paris"})),
                ("get_weather", &json!({"city": "Oslo"})),
            ],
        );
        let open = open.unwrap();
        assert_eq!(open.partial_json.as_deref(), Some(r#"{"city":"Oslo""#));
        assert_eq!(open.held_input, json!({"city": "Oslo", "detail": "cl"}));

        for text in [
            format!("[TOOL_CALLS][{one}, "),
            format!(r#"[TOOL_CALLS][{one}, {{"name": "get_wea"#),
        ] {
            let (parsed, open) = clipped_both_ways(&syntax, &t, &text);
            assert_eq!(
                calls_of(&parsed.blocks),
                [("get_weather", &json!({"city": "Paris"}))],
                "{text:?}",
            );
            assert_eq!(open.unwrap().partial_json, None, "{text:?}");
        }
    }

    /// Qwen XML: the string parameter in flight is kept for matching as
    /// far as it is known to be text — a tail that could be the start of
    /// its close marker is held back — and raw, every byte.
    #[test]
    fn qwen_open_string_holds_back_a_close_marker_prefix() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\n\
                    Paris\n<parameter=detail>\nimport datetime\n</para";
        // `city` never closed its value: the text runs on into what
        // looks like another parameter, so all of it is the value.
        let (_, open) = clipped_both_ways(&syntax, &t, text);
        let open = open.unwrap();
        assert_eq!(open.calls[0].input, json!({}));
        let text = "<tool_call>\n<function=get_weather>\n<parameter=city>\n\
                    Paris\n</parameter>\n<parameter=detail>\nimport datetime\n</para";
        let (_, open) = clipped_both_ways(&syntax, &t, text);
        let open = open.unwrap();
        assert_eq!(open.calls[0].input, json!({"city": "Paris"}));
        assert_eq!(
            open.held_input,
            json!({"city": "Paris", "detail": "import datetime"}),
        );
        assert_eq!(
            open.raw_input,
            json!({"city": "Paris", "detail": "import datetime\n</para"}),
        );
    }

    /// `[TOOL_CALLS]name[ARGS]` with nothing after it is a call whose
    /// arguments are still coming, not a malformed one: the streaming
    /// parser waits instead of yielding the frame as prose, and a clip
    /// there returns the call with no members.
    #[test]
    fn tag_json_call_at_its_args_marker_is_incomplete() {
        let syntax = mistral_tag_json();
        let t = tool("get_weather");
        for head in ["[TOOL_CALLS]get_weather[ARGS]", "[TOOL_CALLS]x[ARGS] "] {
            let (clipped, _) = clipped_both_ways(&syntax, &t, head);
            assert_eq!(clipped.status, ParseStatus::NeedMoreInput, "{head:?}");
            let calls = calls_of(&clipped.blocks);
            assert_eq!(calls.len(), 1, "{head:?}: {clipped:#?}");
            assert_eq!(calls[0].1, &json!({}), "{head:?}");
            assert!(texts_of(&clipped.blocks).is_empty(), "{head:?}");
        }

        let mut p = StreamParser::new(syntax, vec![t], false);
        let mut out = p.push("[TOOL_CALLS]get_weather[ARGS]");
        assert!(out.is_empty(), "frame yielded as prose: {out:#?}");
        out.extend(p.push(r#"{"city": "Paris", "days": 3}"#));
        out.extend(p.finish());
        assert!(texts_of(&out).is_empty(), "{out:#?}");
        assert_eq!(
            calls_of(&out),
            [("get_weather", &json!({"city": "Paris", "days": 3}))],
        );
    }

    /// An unclosed thought is not withheld: a clip mid-reasoning
    /// surfaces an open thought, byte-for-byte what `Final` gives, so the
    /// next request can continue it.
    #[test]
    fn clipped_surfaces_an_open_thought_like_final() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool("get_weather");
        for (text, pre_opened) in [
            ("still reasoning about the", true),
            ("<think>\nstill reasoning about the", false),
        ] {
            let clipped =
                parse_text(&syntax, &[&t], text, pre_opened, Leniency::Clipped);
            let fin =
                parse_text(&syntax, &[&t], text, pre_opened, Leniency::Final);
            assert_eq!(clipped.blocks, fin.blocks, "{text:?}");
            assert!(
                clipped
                    .blocks
                    .last()
                    .is_some_and(crate::prompt::is_open_thought),
                "{text:?}: {:#?}",
                clipped.blocks,
            );
        }
    }

    /// Trigger-less bare JSON cannot tell a clipped call from clipped
    /// structured output (any `{` is its landmark), so the exception
    /// holds: the partial object degrades to `Text` exactly as under
    /// `Final` rather than vanishing.
    #[test]
    fn clipped_bare_json_degrades_like_final() {
        let syntax = CallSyntax::llama31_json();
        let t = tool("get_weather");
        let text = r#"{"verdict": "guilty", "confidence": 0."#;
        let clipped =
            parse_text(&syntax, &[&t], text, false, Leniency::Clipped);
        let fin = parse_text(&syntax, &[&t], text, false, Leniency::Final);
        assert_eq!(clipped.blocks, fin.blocks);
        assert_eq!(texts_of(&clipped.blocks), text);
    }

    /// Every served dialect, as the session analyzes it: the stock and
    /// baked template of each [`crate::baked`] pair, plus the
    /// hand-built constructors (Hermes is the `Family::None` fallback).
    fn served_dialects() -> Vec<(String, CallSyntax)> {
        let eos = |name: &str| match name {
            n if n.starts_with("gemma4") => ("<bos>", "<turn|>"),
            n if n.starts_with("mistral4") => ("<s>", "</s>"),
            n if n.starts_with("gptoss") => ("<|startoftext|>", "<|return|>"),
            _ => ("", "<|im_end|>"),
        };
        let built = [
            ("hermes_json", CallSyntax::hermes_json()),
            ("qwen_xml", CallSyntax::qwen_xml()),
            ("gemma4", CallSyntax::gemma4()),
            ("gpt_oss", CallSyntax::gpt_oss()),
        ]
        .map(|(name, syntax)| (name.to_string(), syntax));
        let baked = crate::baked::ALL.iter().flat_map(|b| {
            let (bos, eos) = eos(b.name);
            [("stock", b.stock), ("baked", b.replacement)].map(
                |(which, source)| {
                    let syntax =
                        crate::dialect::analyze_template(source, bos, eos)
                            .unwrap_or_else(|e| {
                                panic!("{} {which}: {e}", b.name)
                            });
                    (format!("{} ({which})", b.name), syntax)
                },
            )
        });
        built.into_iter().chain(baked).collect()
    }

    /// The lazy grammar for `syntax` over `tools`, as the session
    /// builds it (parallel wherever there is a per-call opener).
    fn lazy_grammar(
        syntax: &CallSyntax,
        tools: &[&Tool],
    ) -> std::sync::Arc<crate::Grammar> {
        use crate::dialect::{grammar_source, Anchor, EmitOptions};
        let opts = EmitOptions {
            anchor: Anchor::Lazy,
            parallel: !syntax.per_call_start.is_empty(),
            ..EmitOptions::default()
        };
        let src = grammar_source(syntax, tools, &opts).expect("emit");
        std::sync::Arc::new(
            crate::Grammar::parse(&src)
                .unwrap_or_else(|e| panic!("grammar: {e}\n{src}")),
        )
    }

    /// #101, live on cogito-32b (~10% of attempts rejected): the marker
    /// dialects triggered on the whole opener, `<tool_call>\n`, so a
    /// real `<tool_call>` the model followed with anything else — `{`,
    /// a space, `\r\n`, EOG — never armed the grammar, the parser left
    /// it in prose, and containment rejected the turn. Every marker
    /// dialect now triggers on the bare special, and the lazy grammar —
    /// which starts at the full opener — takes it as a strict prefix
    /// (so EOG right after it is masked) and goes on to accept the
    /// canonical call. Harmony's recipient-header triggers are
    /// untouched.
    #[test]
    fn trigger_is_the_bare_opener_and_arms_the_lazy_grammar() {
        use crate::GrammarState;
        let t = tool("get_weather");
        let input = json!({"city": "Paris", "days": 3});
        for (name, syntax) in served_dialects() {
            if syntax.family == Family::Harmony {
                assert_eq!(
                    syntax.triggers(),
                    [
                        "<|start|>assistant to=",
                        "<|channel|>commentary to=",
                        "<|channel|>analysis to=",
                    ],
                    "{name}"
                );
                continue;
            }
            let trigger = syntax.trigger();
            assert!(!trigger.is_empty(), "{name}");
            assert_eq!(trigger, trigger.trim(), "{name}");
            assert_eq!(syntax.triggers(), [trigger], "{name}");
            let opener = match syntax.section_start.as_str() {
                "" => syntax.per_call_start.as_str(),
                section => section,
            };
            assert!(opener.starts_with(trigger), "{name}: {opener:?}");

            let mut state = GrammarState::new(lazy_grammar(&syntax, &[&t]));
            assert!(
                state.advance_bytes(trigger.as_bytes()).is_ok(),
                "{name}: the grammar must take the bare trigger"
            );
            assert!(
                !state.is_complete(),
                "{name}: a bare opener must leave the call open"
            );
            let call = render_reference(&syntax, &[("get_weather", &input)])
                .expect("representable");
            let rest = call.strip_prefix(trigger).unwrap_or_else(|| {
                panic!("{name}: {call:?} must open with {trigger:?}")
            });
            // Gemma 4's grammar requires its turn exit after the call.
            let rest = format!("{rest}{}", syntax.tool_response_start);
            assert!(
                state.advance_bytes(rest.as_bytes()).is_ok()
                    && state.is_complete(),
                "{name}: canonical call after the trigger: {call:?}"
            );
        }
    }

    /// The live #101 shapes (cogito-32b; the same opener on the Qwen
    /// XML dialect), each through what the session does with it: the
    /// first trigger arms the lazy grammar, and from there the sampler
    /// only emits bytes the grammar accepts. So every shape with a real
    /// opener ends one of two ways — a complete call, seated by the
    /// parser as `ToolUse` with no special left in prose, or a
    /// constraint the grammar does not complete (`GrammarViolation`, or
    /// bytes it masks and the model could not have emitted) — never as
    /// a special in free text (`EmittedSpecialToken`). Shapes whose
    /// call is well-formed after layout drift also seat without any
    /// grammar: a real opener is never prose.
    #[test]
    fn real_opener_shapes_never_reach_containment() {
        use crate::GrammarState;
        let t = Tool::builder("get_inbox")
            .description("x")
            .schema(json!({
                "type": "object",
                "properties": {"limit": {"type": "integer"}},
            }))
            .build()
            .expect("valid tool");
        let cogito = crate::dialect::analyze_template(
            crate::baked::COGITO.replacement,
            "",
            "<|im_end|>",
        )
        .expect("analyze");
        let json_call = "<tool_call>\n{\"name\": \"get_inbox\", \
                         \"arguments\": {}}\n</tool_call>";
        let json_body = "{\"name\": \"get_inbox\", \"arguments\": {}}\
                         \n</tool_call>";
        let xml_call = "<tool_call>\n<function=get_inbox>\n</function>\n\
                        </tool_call>";
        let xml_body = "<function=get_inbox>\n</function>\n</tool_call>";
        let poisoned = |blocks: &[Block], marker: &str| {
            blocks.iter().any(|b| match b {
                Block::Text { text, .. } => text.contains(marker),
                Block::Thought { thought, .. } => thought.contains(marker),
                _ => false,
            })
        };
        for (syntax, call, body) in [
            (cogito, json_call, json_body),
            (CallSyntax::qwen_xml(), xml_call, xml_body),
        ] {
            let opener = syntax.trigger();
            // `(label, emission, seats without a grammar)`.
            let shapes: Vec<(&str, String, bool)> = vec![
                ("canonical", format!("ok\n\n{call}"), true),
                ("call only", call.to_string(), true),
                ("two calls", format!("ok\n\n{call}\n{call}"), true),
                ("no newline", format!("ok\n\n{opener}{body}"), true),
                ("space", format!("ok\n\n{opener} {body}"), true),
                ("crlf", format!("ok\n\n{opener}\r\n{body}"), true),
                ("double newline", format!("ok\n\n{opener}\n\n{body}"), true),
                ("mention", format!("I will use {opener} tags."), false),
                ("opener at EOG", format!("{call}\n{opener}"), false),
                ("opener+nl at EOG", format!("{call}\n{opener}\n"), false),
                ("bare opener", format!("ok\n\n{opener}"), false),
                ("malformed", format!("{opener}\n{{\"name\": }}"), false),
            ];
            let grammar = lazy_grammar(&syntax, &[&t]);
            for (label, emission, seats) in shapes {
                let what = format!("{:?} {label}: {emission:?}", syntax.family);
                let parsed = parse_text(
                    &syntax,
                    &[&t],
                    &emission,
                    false,
                    Leniency::Final,
                );
                let seated = !poisoned(&parsed.blocks, opener)
                    && parsed
                        .blocks
                        .iter()
                        .any(|b| matches!(b, Block::ToolUse { .. }));
                if seats {
                    assert!(seated, "{what} → {:#?}", parsed.blocks);
                }
                // Under the session: armed at the first trigger.
                let at = emission.find(opener).expect("has an opener");
                let mut state = GrammarState::new(grammar.clone());
                assert!(
                    state.advance_bytes(opener.as_bytes()).is_ok(),
                    "{what}: the opener must arm the grammar"
                );
                let rest = &emission.as_bytes()[at + opener.len()..];
                let completes =
                    state.advance_bytes(rest).is_ok() && state.is_complete();
                // A grammar-complete turn is the only one the session
                // returns `Ok`; it must seat.
                if completes {
                    assert!(seated, "{what} → {:#?}", parsed.blocks);
                }
                assert!(
                    completes || !matches!(label, "canonical" | "call only"),
                    "{what}: canonical shapes complete the grammar"
                );
            }
        }
    }
}
