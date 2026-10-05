//! Request `stop_sequences` (#122), matched against what a client sees
//! as **text**: prose ([`Block::Text`]) and the string values of a tool
//! call's input.
//!
//! - A match in prose ends the turn there: the text before it stands,
//!   everything after it goes.
//! - A match in a call's input ends the turn there too, and the call
//!   comes back **cut at the match**, as Anthropic returns it: the
//!   string is cut right before the match, the members before it stand,
//!   and the JSON is closed. The members after it never existed — the
//!   match is found while the call is still being generated, and
//!   generation stops there. Only string values are matched, each on
//!   its own — never the keys, the JSON around them, or the dialect's
//!   spelling of them.
//! - Either way the turn reads `stop_reason: stop_sequence`.
//!
//! That is Anthropic's behavior, captured 2026-09-30 on
//! claude-haiku-4-5: with `stop_sequences: ["print("]` and a forced
//! `write_file`, it answered
//! `{"path":"hello.py","contents":"import datetime\n"}` under
//! `stop_reason: stop_sequence`, `stop_sequence: "print("` (misanthropic's
//! `misanthropic/test/data/stop/stop_sequence_tool.*`; streamed, the block
//! gets its `content_block_stop` and the last `input_json_delta` closes
//! the object; `stop_sequence_text_tool.*`: the same after a text block
//! under `tool_choice: auto`). The same prompt must drive a client the
//! same way on both backends. The call looks complete and is not; a
//! client must gate dispatch on `stop_reason`, as it must for a clip.
//!
//! Never matched: **framing** and **thinking**. Framing is the dialect's
//! markers (`<tool_call>`, `<function=…>`, `[TOOL_CALLS]`/`[ARGS]`,
//! Harmony headers, EOG pieces) *and the whitespace that separates prose
//! from a structure*: models write `"Sure, checking.\n\n<tool_call>"`
//! and `"</think>\n\nAnswer"`, and a stop of `"\n"` matched there killed
//! the call, or ended the turn before its answer began. So whitespace at
//! either edge of a prose run that touches a thought or a call is
//! framing; whitespace inside a run is prose, and so is whitespace that
//! ends a turn that finished cleanly (`"Hello!\n"`, then EOG). A turn cut
//! short (`max_tokens`, or a call in flight) matches none of its trailing
//! whitespace — what would have followed is unknown. Thinking is not the
//! answer; matched there, a stop would end a turn before the answer
//! began. (Anthropic's behavior inside `thinking` is uncaptured.)
//!
//! A *whitespace-only* stop (`"\n"`) never reaches this through
//! blallama: Anthropic rejects one with a 400 (`stop_sequences: each
//! stop sequence must contain non-whitespace`, captured 2026-09-30 in
//! misanthropic's `misanthropic/test/data/stop/whitespace_stop.error.json`),
//! and
//! so does blallama. The framing rule stays for two reasons: a library
//! caller can still pass one to [`Session`](super::Session) directly,
//! and a stop that merely *begins or ends* with whitespace
//! (`"\nObservation:"`) must not match across the whitespace beside a
//! structure either.
//!
//! A match is per prose *run*: a structure between two stretches of
//! text ends one run and starts the next, so a stop never straddles one.

use serde_json::Value;

use crate::{
    dialect::{
        cut_value, OpenCall, ParseStatus, Parsed, Provenance, StreamParser,
    },
    predictor::{first_stop_string, stop_string_holdback},
    Block,
};

/// The request's stop sequences, empties dropped (they would match
/// before the first token).
pub(super) fn request_stops(prompt: &crate::Prompt) -> Vec<String> {
    prompt
        .stop_sequences
        .iter()
        .flatten()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
        .collect()
}

/// One prose run's stop matching: a stream cannot take back text it
/// has yielded, so text that could still grow into a stop sequence is
/// held until the next delta settles it.
#[derive(Debug, Default, Clone)]
pub(super) struct StopCutter {
    stops: Vec<String>,
    /// Admitted text not yet passed on: the tail that could still grow
    /// into a stop sequence.
    held: String,
    /// The stop sequence that completed and was cut out. Everything
    /// after it is past the stop.
    hit: Option<String>,
}

impl StopCutter {
    pub(super) fn new(stops: Vec<String>) -> Self {
        Self {
            stops,
            ..Self::default()
        }
    }

    /// The stop sequence that matched, once one has.
    pub(super) fn hit(&self) -> Option<&str> {
        self.hit.as_deref()
    }

    /// Admit one delta of the run; returns the text now safe to pass
    /// on. Once a stop sequence completes, the text before it and
    /// nothing more, ever.
    pub(super) fn push(&mut self, delta: &str) -> String {
        if self.hit.is_some() {
            return String::new();
        }
        self.held.push_str(delta);
        let mut held = std::mem::take(&mut self.held);
        if let Some((at, i)) = first_stop_string(&held, &self.stops) {
            self.hit = Some(self.stops[i].clone());
            held.truncate(at);
            return held;
        }
        let keep = stop_string_holdback(&held, &self.stops);
        self.held = held.split_off(held.len() - keep);
        held
    }

    /// Cut a call's input at the first stop in it ([`cut_value`]); a
    /// match is a hit.
    fn cut_input(&mut self, input: &Value) -> Option<Value> {
        if self.hit.is_some() {
            return None;
        }
        let (cut, i) = cut_value(input, &self.stops)?;
        self.hit = Some(self.stops[i].clone());
        Some(cut)
    }

    /// Whether any stop sequence is in one of `inputs`' string values.
    fn in_any<'v>(&self, mut inputs: impl Iterator<Item = &'v Value>) -> bool {
        inputs.any(|input| cut_value(input, &self.stops).is_some())
    }

    /// End of the run: the held tail never completed a stop sequence,
    /// so it is output after all.
    pub(super) fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// Stop sequences over a generation in flight: pieces go through the
/// dialect's [`StreamParser`], and only what it releases as text —
/// prose, and the string values of a call's input, the call in flight
/// included — is matched; never framing held back as a possible marker,
/// never whitespace between prose and a structure, never a thought.
/// Yields the parser's blocks with everything past a stop dropped (a
/// call whose input matched is cut there), so [`super::BlockStream`]
/// streams from it directly and the batch paths use it to know when to
/// stop.
#[derive(Debug, Clone)]
pub(super) struct StopFilter {
    parser: StreamParser,
    cutter: StopCutter,
    /// Whitespace ending the prose run so far, held back unmatched:
    /// framing if a structure follows, prose if more text does.
    trailing_ws: String,
    /// A structure was the last thing out and no prose has followed
    /// but whitespace — framing too, passed on unmatched.
    after_structure: bool,
}

impl StopFilter {
    /// The parser the filter reads.
    pub(super) fn parser(&self) -> &StreamParser {
        &self.parser
    }

    pub(super) fn new(parser: StreamParser, stops: Vec<String>) -> Self {
        Self {
            parser,
            cutter: StopCutter::new(stops),
            trailing_ws: String::new(),
            after_structure: false,
        }
    }

    /// The stop sequence that matched, once one has.
    pub(super) fn hit(&self) -> Option<&str> {
        self.cutter.hit()
    }

    /// Feed one piece and the token behind it (what the parser's
    /// emission provenance keys on); returns the blocks (or prose
    /// deltas) it resolved.
    pub(super) fn push(
        &mut self,
        piece: &str,
        token: Option<crate::Token>,
    ) -> Vec<Block> {
        if self.hit().is_some() {
            return Vec::new();
        }
        let blocks = self.parser.push_token(piece, token);
        let mut out = self.admit(blocks);
        if self.hit().is_none() {
            out.extend(self.stop_in_flight());
        }
        out
    }

    /// Once the budget cut a turn inside a call and the flush returned
    /// it ([`Self::finish`] with `clipped`): that call's input as JSON
    /// left open where the cut fell — what an Anthropic stream sends as
    /// its `input_json_delta`, with no `content_block_stop` after
    /// ([`OpenCall::partial_json`]). A call a stop sequence cut is
    /// closed, so `None` then.
    pub(super) fn open_call_json(&self) -> Option<&str> {
        self.hit()
            .is_none()
            .then(|| self.parser.open()?.partial_json.as_deref())
            .flatten()
    }

    /// A stop in the input of the call still being generated ends the
    /// turn now, as on Anthropic — not when (or whether) the call
    /// closes. The flush returns what the parser holds: prose it held
    /// back (which may match first), and the call with its string in
    /// flight, which [`Self::admit`] cuts at the match.
    fn stop_in_flight(&mut self) -> Vec<Block> {
        let Some(open) = self.parser.open() else {
            return Vec::new();
        };
        let Some((last, before)) = open.calls.split_last() else {
            return Vec::new();
        };
        let inputs = before.iter().map(|call| &call.input);
        if !self.cutter.in_any(inputs.chain([&open.held_input])) {
            return Vec::new();
        }
        let (id, held) = (last.id.clone(), open.held_input.clone());
        let mut blocks = self.parser.finish_clipped();
        if let Some(Block::ToolUse { call }) = blocks.last_mut() {
            if call.id == id {
                call.input = held;
            }
        }
        self.admit(blocks)
    }

    /// End of generation. `clipped`: the generation was cut short, so
    /// an incomplete trailing call comes back with the members that
    /// completed ([`StreamParser::finish_clipped`]) and the run's
    /// trailing whitespace is not matched.
    ///
    /// Otherwise the flush degrades an incomplete structure to text
    /// ([`StreamParser::finish`]) — framing, not prose the model
    /// finished, so it is never matched. Only the prose the flush
    /// releases (the tail held back as a possible marker) and the
    /// whitespace that ended the answer are: a stop found there, with
    /// any call in flight cut short, stops the turn as though it had
    /// matched mid-stream.
    pub(super) fn finish(&mut self, clipped: bool) -> Vec<Block> {
        if clipped || self.hit().is_some() {
            return self.flush_matching(false);
        }
        let mut probe = self.clone();
        let out = probe.flush_matching(true);
        if probe.hit().is_some() {
            *self = probe;
            return out;
        }
        let blocks = self.parser.finish();
        self.release().into_iter().chain(blocks).collect()
    }

    /// Flush the parser clipped and match what it releases. `turn_end`:
    /// the turn finished cleanly, so unless a structure is in flight the
    /// run's trailing whitespace ended the answer and is prose.
    fn flush_matching(&mut self, turn_end: bool) -> Vec<Block> {
        let blocks = self.parser.finish_clipped();
        let mut out = self.admit(blocks);
        if turn_end && self.hit().is_none() && !self.parser.in_flight() {
            let ws = std::mem::take(&mut self.trailing_ws);
            out.extend(prose(self.cutter.push(&ws)));
            self.clear_on_hit();
        }
        out.extend(self.release());
        out
    }

    /// Held text that never completed a stop: output after all,
    /// unmatched — the run's tail, then its trailing whitespace.
    fn release(&mut self) -> Vec<Block> {
        let tail = self.cutter.finish();
        let ws = std::mem::take(&mut self.trailing_ws);
        prose(tail).into_iter().chain(prose(ws)).collect()
    }

    /// Past a stop nothing is output, held whitespace included.
    fn clear_on_hit(&mut self) {
        if self.hit().is_some() {
            self.trailing_ws.clear();
        }
    }

    /// Match the text among `blocks`; a structure ends the run.
    fn admit(&mut self, blocks: Vec<Block>) -> Vec<Block> {
        let mut out = Vec::new();
        for block in blocks {
            if self.hit().is_some() {
                break;
            }
            match block {
                Block::Text { text, .. } => {
                    let mut text = text.as_ref();
                    if self.after_structure {
                        // Whitespace after a structure is framing.
                        let body = text.trim_start();
                        out.extend(prose(
                            text[..text.len() - body.len()].to_owned(),
                        ));
                        if body.is_empty() {
                            continue;
                        }
                        self.after_structure = false;
                        text = body;
                    }
                    // Whitespace ending the run is held unmatched until
                    // prose follows it (then it is prose too).
                    self.trailing_ws.push_str(text);
                    let body_len = self.trailing_ws.trim_end().len();
                    if body_len > 0 {
                        let ws = self.trailing_ws.split_off(body_len);
                        let body = std::mem::replace(&mut self.trailing_ws, ws);
                        out.extend(prose(self.cutter.push(&body)));
                        self.clear_on_hit();
                    }
                }
                mut other => {
                    // The run's held tail never completed a stop, and
                    // its trailing whitespace is framing.
                    out.extend(self.release());
                    self.after_structure = true;
                    if let Block::ToolUse { call } = &mut other {
                        if let Some(cut) = self.cutter.cut_input(&call.input) {
                            // Cut at the match; nothing follows it.
                            call.input = cut;
                            out.push(other);
                            break;
                        }
                    }
                    out.push(other);
                }
            }
        }
        out
    }
}

/// `text` as a [`Block::Text`], or nothing when empty.
fn prose(text: String) -> Option<Block> {
    (!text.is_empty()).then(|| text.into())
}

/// A clipped parse's blocks as a stop sequence sees them: the call in
/// flight with its string value kept — every byte when `raw`
/// ([`OpenCall::raw_input`]), else as far as it is known to be text
/// ([`OpenCall::held_input`], what [`StopFilter`] matched). The batch
/// paths cut this with
/// [`cut_at_stop`].
pub(super) fn stop_view(
    (parsed, open): (Parsed, Option<OpenCall>),
    raw: bool,
) -> Vec<Block> {
    let mut blocks = parsed.blocks;
    if let (Some(open), Some(Block::ToolUse { call })) =
        (open, blocks.last_mut())
    {
        if open.calls.last().is_some_and(|last| last.id == call.id) {
            call.input = if raw { open.raw_input } else { open.held_input };
        }
    }
    // Not merged: a parse merges its own prose, so text blocks side by
    // side are Harmony channels, which stay two blocks; `cut_at_stop`
    // matches across them, as the stream does.
    blocks
}

/// Cut `blocks` (as [`stop_view`] leaves them) at the first stop
/// sequence, by [`StopFilter`]'s rules — the batch half of the same
/// policy. A match in prose keeps the text before it (the block dropped
/// when nothing is left); a match in a call's input keeps the call, its
/// input cut at the match ([`cut_value`]). Every block after the cut
/// goes. Returns the matched stop, if any. Pass a call in flight as
/// [`stop_view`] leaves it.
///
/// Prose is matched per run, as the filter matches it: text blocks side
/// by side (Harmony channels — a preamble, then the final) are one run,
/// since a stream's text yields cannot mark where one ended and the
/// next began, so a stop the client sees across them matches here too.
/// The blocks stay apart; a match is cut in the block it starts in.
///
/// The last block's trailing whitespace is matched: `blocks` is taken
/// to end a turn that finished cleanly. A caller holding the stop the
/// filter reported can pass just that one — the filter found it first,
/// so it is the first match here too.
pub(super) fn cut_at_stop<S: AsRef<str>>(
    mut blocks: Vec<Block>,
    stops: &[S],
) -> (Vec<Block>, Option<String>) {
    fn text(b: &Block) -> Option<&str> {
        match b {
            Block::Text { text, .. } => Some(text.as_ref()),
            _ => None,
        }
    }
    /// Where a stop matched: at a byte of prose, or in a call's input.
    enum Match {
        Prose(usize),
        Input(Value),
    }
    let n = blocks.len();
    let hit = (0..n).find_map(|i| match &blocks[i] {
        // Inside a run: its first block matched the whole of it.
        Block::Text { .. } if i > 0 && text(&blocks[i - 1]).is_some() => None,
        Block::Text { .. } => {
            let run: Vec<&str> = blocks[i..].iter().map_while(text).collect();
            let end = i + run.len();
            let joined = run.concat();
            // Whitespace touching a structure is framing.
            let start = match i {
                0 => 0,
                _ => joined.len() - joined.trim_start().len(),
            };
            let stop = match end < n {
                true => joined.trim_end().len(),
                false => joined.len(),
            };
            let body = joined.get(start..stop.max(start)).unwrap_or("");
            let (at, s) = first_stop_string(body, stops)?;
            let at = start + at;
            // The block the match starts in: the last to start at or
            // before it.
            let (k, from) = run
                .iter()
                .scan(0, |pos, t| {
                    let from = *pos;
                    *pos += t.len();
                    Some(from)
                })
                .enumerate()
                .take_while(|&(_, from)| from <= at)
                .last()?;
            Some((i + k, Match::Prose(at - from), s))
        }
        Block::ToolUse { call } => cut_value(&call.input, stops)
            .map(|(cut, s)| (i, Match::Input(cut), s)),
        _ => None,
    });
    let Some((i, at, s)) = hit else {
        return (blocks, None);
    };
    match at {
        Match::Prose(0) => blocks.truncate(i),
        Match::Prose(at) => {
            blocks.truncate(i + 1);
            if let Some(Block::Text { text, .. }) = blocks.last_mut() {
                text.to_mut().truncate(at);
            }
        }
        // In a call's input: the call stands, cut at the match.
        Match::Input(cut) => {
            blocks.truncate(i + 1);
            if let Some(Block::ToolUse { call }) = blocks.last_mut() {
                call.input = cut;
            }
        }
    }
    (blocks, Some(stops[s].as_ref().to_owned()))
}

/// Byte offset at which to cut `raw` — the generation's raw bytes,
/// framing and all — so that it parses to `kept`, the output
/// [`cut_at_stop`] left. `view` is a prefix's parse as [`stop_view`]
/// leaves it with `raw` (every byte of a string in flight kept), or
/// `None` when it leaves out a structure in flight.
///
/// The shortest of the run of such prefixes nearest the end: a match in
/// a call's input leaves the call's raw bytes up to the match, and an
/// escape the match began with (`\n` of `"a\nb"`) decodes to nothing
/// until complete, so the prefix ending mid-escape parses the same —
/// its dangling byte is not kept.
///
/// Each step parses a prefix, so the walk is bounded: it starts at the
/// end, where generation stopped (within a piece or a held-back marker
/// of the match), and goes no lower than [`text_floor`] — a prefix
/// shorter than the text `kept` shows cannot parse to it. A walk that
/// finds no match then costs the framing and what followed the match,
/// not every byte.
pub(super) fn raw_stop_cut(
    raw: &str,
    kept: &[Block],
    view: impl Fn(&str) -> Option<Vec<Block>>,
) -> Option<usize> {
    let floor = text_floor(kept);
    let mut prefixes = raw
        .char_indices()
        .map(|(i, _)| i)
        .chain([raw.len()])
        .rev()
        .take_while(|&i| i >= floor)
        .skip_while(|&i| view(&raw[..i]).as_deref() != Some(kept));
    let longest = prefixes.next()?;
    Some(
        prefixes
            .take_while(|&i| view(&raw[..i]).as_deref() == Some(kept))
            .last()
            .unwrap_or(longest),
    )
}

/// Bytes of `blocks` that each come from a distinct raw byte of the
/// text they were parsed from: prose and thoughts verbatim, and a
/// call's input keys and string values, which decode to no more bytes
/// than their spelling. Ids, names and numbers are left out (an id is
/// made up, a number can re-serialize longer), so this never exceeds
/// the length of a raw prefix that parses to `blocks`.
fn text_floor(blocks: &[Block]) -> usize {
    fn strings(v: &Value) -> usize {
        match v {
            Value::String(s) => s.len(),
            Value::Array(items) => items.iter().map(strings).sum(),
            Value::Object(members) => {
                members.iter().map(|(k, v)| k.len() + strings(v)).sum()
            }
            _ => 0,
        }
    }
    blocks
        .iter()
        .map(|b| match b {
            Block::Text { text, .. } => text.len(),
            Block::Thought { thought, .. } => thought.len(),
            Block::ToolUse { call } => strings(&call.input),
            _ => 0,
        })
        .sum()
}

/// A stop cut of a generation read with emission provenance: the
/// output [`cut_at_stop`] keeps, and where [`raw_stop_cut`] cuts the
/// raw bytes — `provenance.restore(marked)` — to match it. `parse` is
/// the clipped parse of marked text. Each raw prefix is parsed as the
/// marked text it restores from (`Provenance::marked_prefix`): a stop
/// can start inside a piece the model spelled, where no marked prefix
/// ends.
pub(super) fn marked_stop_cut(
    provenance: &Provenance,
    marked: &str,
    stop: &str,
    parse: impl Fn(&str) -> (Parsed, Option<OpenCall>),
) -> (Vec<Block>, Option<usize>) {
    let raw = provenance.restore(marked);
    let parse_to = |end: usize| {
        provenance.restore_parse(parse(&provenance.marked_prefix(marked, end)))
    };
    // A prefix withholding a structure in flight, with no call to show
    // for it, is not where any stop fell.
    let view = |prefix: &str| {
        let (parsed, open) = parse_to(prefix.len());
        let complete = parsed.status == ParseStatus::Complete;
        (complete || open.is_some()).then(|| stop_view((parsed, open), true))
    };
    let (kept, _) = cut_at_stop(stop_view(parse_to(raw.len()), false), &[stop]);
    let at = raw_stop_cut(&raw, &kept, view);
    (kept, at)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{
        parse_text, parse_text_open, render_reference, CallSyntax, Family,
        FunctionSyntax, Leniency, StreamParser,
    };
    use crate::Tool;
    use serde_json::json;

    /// What a client sees as text.
    fn texts(blocks: &[Block]) -> String {
        blocks
            .iter()
            .filter_map(|b| match b {
                Block::Text { text, .. } => Some(text.as_ref()),
                _ => None,
            })
            .collect()
    }

    fn tool() -> Tool {
        Tool::builder("get_weather")
            .description("test")
            .schema(json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }))
            .build()
            .expect("valid test tool")
    }

    /// Mistral Small 4's `[TOOL_CALLS]name[ARGS]{…}`, as the analyzer
    /// derives it.
    fn mistral() -> CallSyntax {
        CallSyntax {
            family: Family::TagWithJson,
            per_call_start: "[TOOL_CALLS]".into(),
            function: FunctionSyntax {
                name_prefix: String::new(),
                name_suffix: "[ARGS]".into(),
                close: String::new(),
            },
            ..CallSyntax::default()
        }
    }

    /// Every dialect with prose beside its calls: Qwen XML, Hermes JSON
    /// (Qwen chat, Cogito), Mistral's `[TOOL_CALLS]`, Gemma 4.
    fn dialects() -> [(&'static str, CallSyntax); 4] {
        [
            ("qwen_xml", CallSyntax::qwen_xml()),
            ("hermes_json", CallSyntax::hermes_json()),
            ("mistral", mistral()),
            ("gemma4", CallSyntax::gemma4()),
        ]
    }

    fn call(syntax: &CallSyntax, input: serde_json::Value) -> String {
        render_reference(syntax, &[("get_weather", &input)]).unwrap()
    }

    fn filter(syntax: CallSyntax, stops: &[&str]) -> StopFilter {
        StopFilter::new(
            StreamParser::new(syntax, vec![tool()], false),
            stops.iter().map(|s| s.to_string()).collect(),
        )
    }

    /// Feed `text` a char at a time — every piece boundary a tokenizer
    /// could produce — until a stop fires (generation stops there, as
    /// in the session), then finish a turn that ended cleanly (EOG).
    /// Returns the output and the bytes generated.
    fn run_to_stop<'t>(
        f: &mut StopFilter,
        text: &'t str,
    ) -> (Vec<Block>, &'t str) {
        let mut out = Vec::new();
        let mut generated = text.len();
        for (i, c) in text.char_indices() {
            out.extend(f.push(c.encode_utf8(&mut [0; 4]), None));
            if f.hit().is_some() {
                generated = i + c.len_utf8();
                break;
            }
        }
        out.extend(f.finish(false));
        (super::super::merge_adjacent_prose(out), &text[..generated])
    }

    fn run(f: &mut StopFilter, text: &str) -> Vec<Block> {
        run_to_stop(f, text).0
    }

    fn calls(blocks: &[Block]) -> usize {
        blocks
            .iter()
            .filter(|b| matches!(b, Block::ToolUse { .. }))
            .count()
    }

    /// Stream `text` through the filter, then check the batch half
    /// agrees, as the session runs it: the clipped parse of what was
    /// generated — a call in flight with its string kept — cut at the
    /// stop the filter reported, is the streamed output, block for
    /// block (as a client assembles a stream's text yields: two Harmony
    /// channels side by side are one run there). With no stop hit, the
    /// batch finds none either. Returns the output.
    fn stream_and_batch(
        syntax: CallSyntax,
        stops: &[&str],
        text: &str,
    ) -> (Vec<Block>, Option<String>) {
        let mut f = filter(syntax.clone(), stops);
        let (streamed, generated) = run_to_stop(&mut f, text);
        let hit = f.hit().map(str::to_owned);
        let t = tool();
        let parsed = parse_text_open(
            &syntax,
            &[&t],
            generated,
            false,
            Leniency::Clipped,
        );
        let view = stop_view(parsed, false);
        match &hit {
            Some(hit) => {
                let (cut, found) = cut_at_stop(view, &[hit]);
                assert_eq!(found.as_ref(), Some(hit), "{text:?}");
                assert_eq!(
                    super::super::merge_adjacent_prose(cut),
                    streamed,
                    "batch vs stream on {text:?}"
                );
            }
            None => {
                let (_, found) = cut_at_stop(view, stops);
                assert_eq!(found, None, "batch only on {text:?}");
            }
        }
        (streamed, hit)
    }

    /// The input of the one call in `blocks`.
    fn input_of(blocks: &[Block]) -> &serde_json::Value {
        let mut inputs = blocks.iter().filter_map(|b| match b {
            Block::ToolUse { call } => Some(&call.input),
            _ => None,
        });
        let input = inputs.next().expect("a call");
        assert!(inputs.next().is_none(), "one call: {blocks:#?}");
        input
    }

    /// The review's case: models put a blank line between their prose
    /// and a call (`"Sure, checking.\n\n<tool_call>…"`). That whitespace
    /// is framing, not text, so a stop of `"\n"` must not match it and
    /// cut the call — in any dialect, nor anywhere in the call's own
    /// framing. The prose stands, the call parses whole.
    #[test]
    fn newline_stop_never_matches_framing_around_a_call() {
        for (name, syntax) in dialects() {
            let text = format!(
                "Sure, checking.\n\n{}",
                call(&syntax, json!({"city": "Paris"})),
            );
            let (out, hit) = stream_and_batch(syntax, &["\n"], &text);
            assert_eq!(hit, None, "{name}: {out:#?}");
            assert_eq!(calls(&out), 1, "{name}: {out:#?}");
            assert_eq!(texts(&out), "Sure, checking.\n\n", "{name}");
        }
    }

    /// Parallel calls, and whitespace after the last one before EOG:
    /// all framing. The turn is a finished tool turn, not a stop.
    #[test]
    fn newline_stop_never_matches_between_or_after_calls() {
        for (name, syntax) in dialects() {
            let paris = call(&syntax, json!({"city": "Paris"}));
            let oslo = call(&syntax, json!({"city": "Oslo"}));
            let text = format!("Checking both.\n\n{paris}\n{oslo}\n");
            let (out, hit) = stream_and_batch(syntax, &["\n"], &text);
            assert_eq!(hit, None, "{name}: {out:#?}");
            assert_eq!(calls(&out), 2, "{name}: {out:#?}");
        }
    }

    /// Harmony carries no prose beside a call; its headers, its
    /// reasoning and its call are all out of a `"\n"` stop's reach.
    #[test]
    fn newline_stop_never_matches_harmony_framing() {
        let syntax = CallSyntax::gpt_oss();
        let text = format!(
            "<|channel|>analysis<|message|>Need the\nweather.<|end|>\
             <|start|>assistant{}",
            call(&syntax, json!({"city": "Paris"})),
        );
        let (out, hit) = stream_and_batch(syntax, &["\n"], &text);
        assert_eq!(hit, None, "{out:#?}");
        assert_eq!(calls(&out), 1, "{out:#?}");
    }

    /// gpt-oss may write a commentary preamble, then its final, with no
    /// call between: two text blocks in a batch, one run of text yields
    /// in a stream, which cannot mark where one ended. A stop matches
    /// what the client sees on either path — across the two as well —
    /// and the batch cut keeps the channels apart.
    #[test]
    fn stop_matches_across_a_harmony_preamble_and_its_final() {
        let syntax = CallSyntax::gpt_oss();
        let text = "<|channel|>commentary<|message|>Hi.<|end|>\
                    <|start|>assistant<|channel|>final<|message|>Ok then.";
        let t = tool();
        let blocks =
            parse_text(&syntax, &[&t], text, false, Leniency::Final).blocks;
        assert_eq!(blocks.len(), 2, "two channels: {blocks:#?}");
        for (stop, want, n) in [
            // Across the boundary: cut in the preamble.
            (".O", "Hi", 1),
            // At the boundary: the preamble stands whole.
            ("Ok", "Hi.", 1),
            // Inside the final: both channels, the final cut.
            ("then", "Hi.Ok ", 2),
        ] {
            let (out, hit) = stream_and_batch(syntax.clone(), &[stop], text);
            assert_eq!(hit.as_deref(), Some(stop));
            assert_eq!(texts(&out), want, "{stop:?}");
            let (cut, found) = cut_at_stop(blocks.clone(), &[stop]);
            assert_eq!(found.as_deref(), Some(stop));
            assert_eq!(cut.len(), n, "{stop:?}: {cut:#?}");
            assert_eq!(texts(&cut), want, "{stop:?}");
        }
        let (out, hit) = stream_and_batch(syntax, &["absent"], text);
        assert_eq!((texts(&out).as_str(), hit), ("Hi.Ok then.", None));
    }

    /// The raw cut of a stop across two Harmony channels ends inside the
    /// preamble, where the kept output does: the session cuts the
    /// response there, not the stop bytes left in place.
    #[test]
    fn raw_stop_cut_across_harmony_channels() {
        let syntax = CallSyntax::gpt_oss();
        let preamble = "<|channel|>commentary<|message|>Hi";
        let raw = format!(
            "{preamble}.<|end|><|start|>assistant<|channel|>final\
             <|message|>O"
        );
        let parse = |prefix: &str| {
            parse_text_open(&syntax, &[], prefix, false, Leniency::Clipped)
        };
        let view = |prefix: &str| {
            let (parsed, open) = parse(prefix);
            let complete =
                parsed.status == crate::dialect::ParseStatus::Complete;
            (complete || open.is_some())
                .then(|| stop_view((parsed, open), true))
        };
        let (kept, hit) = cut_at_stop(stop_view(parse(&raw), false), &[".O"]);
        assert_eq!(hit.as_deref(), Some(".O"));
        assert_eq!(texts(&kept), "Hi");
        assert_eq!(raw_stop_cut(&raw, &kept, view), Some(preamble.len()));
    }

    /// The separator after a thought (`"</think>\n\n"`) is framing too:
    /// a `"\n"` stop does not end the turn before its answer begins.
    #[test]
    fn newline_stop_never_matches_the_separator_after_a_thought() {
        let text = "<think>\nhmm\n</think>\n\nLine one.\nLine two.";
        let (out, hit) =
            stream_and_batch(CallSyntax::qwen_xml(), &["\n"], text);
        assert_eq!(hit.as_deref(), Some("\n"));
        assert_eq!(texts(&out).trim_start(), "Line one.", "{out:#?}");

        // Harmony's final channel: the stop is inside the answer.
        let text = "<|channel|>analysis<|message|>Think.<|end|>\
                    <|start|>assistant<|channel|>final<|message|>A\nB";
        let (out, hit) = stream_and_batch(CallSyntax::gpt_oss(), &["\n"], text);
        assert_eq!(hit.as_deref(), Some("\n"));
        assert_eq!(texts(&out), "A", "{out:#?}");
    }

    /// A stop genuinely inside prose still stops — whitespace inside a
    /// run is text, and so is whitespace that ends a turn that finished
    /// cleanly. Only a turn cut short leaves its trailing whitespace
    /// unmatched.
    #[test]
    fn stop_inside_prose_still_stops() {
        for (name, syntax) in dialects() {
            let (out, hit) =
                stream_and_batch(syntax.clone(), &["\n"], "line1\nline2");
            assert_eq!(hit.as_deref(), Some("\n"), "{name}");
            assert_eq!(texts(&out), "line1", "{name}");

            let (out, hit) = stream_and_batch(syntax.clone(), &["\n"], "Hi!\n");
            assert_eq!(hit.as_deref(), Some("\n"), "{name}");
            assert_eq!(texts(&out), "Hi!", "{name}");

            let mut f = filter(syntax, &["\n"]);
            let mut out: Vec<Block> = "Hi!\n"
                .chars()
                .flat_map(|c| f.push(&c.to_string(), None))
                .collect();
            out.extend(f.finish(true));
            assert_eq!(f.hit(), None, "{name}: clipped");
            assert_eq!(texts(&out), "Hi!\n", "{name}");
        }
    }

    /// Prose before the call is text output, so a stop there does fire —
    /// and everything past it, the call included, is gone.
    #[test]
    fn stop_in_prose_before_a_call_cuts_the_call() {
        for (name, syntax) in dialects() {
            let text = format!(
                "Sure. Let me check.\n\n{}",
                call(&syntax, json!({"city": "Oslo"})),
            );
            let (out, hit) = stream_and_batch(syntax, &["Let me"], &text);
            assert_eq!(hit.as_deref(), Some("Let me"), "{name}");
            assert_eq!(calls(&out), 0, "{name}: {out:#?}");
            assert_eq!(texts(&out), "Sure. ", "{name}");
        }
    }

    /// Every dialect with calls, Harmony included (no prose beside its
    /// calls).
    fn call_dialects() -> [(&'static str, CallSyntax, &'static str); 5] {
        let [qwen, hermes, mistral, gemma] = dialects()
            .map(|(name, syntax)| (name, syntax, "Sure, checking.\n\n"));
        [
            qwen,
            hermes,
            mistral,
            gemma,
            ("harmony", CallSyntax::gpt_oss(), ""),
        ]
    }

    /// A call's input values are text: a stop in one ends the turn, and
    /// the call comes back cut at the match — Anthropic's captured
    /// behavior (`{"path":"hello.py","contents":"import datetime\n"}` for
    /// `["print("]`). The prose before it stands; keys and JSON syntax
    /// are never matched.
    #[test]
    fn stop_in_a_call_input_cuts_the_call_there() {
        for (name, syntax, prose) in call_dialects() {
            let text = format!(
                "{prose}{}",
                call(&syntax, json!({"city": "Paris\nFrance"})),
            );
            let (out, hit) = stream_and_batch(syntax.clone(), &["\n"], &text);
            assert_eq!(hit.as_deref(), Some("\n"), "{name}");
            assert_eq!(input_of(&out), &json!({"city": "Paris"}), "{name}");
            assert_eq!(texts(&out), prose, "{name}");

            let (out, hit) =
                stream_and_batch(syntax.clone(), &["France"], &text);
            assert_eq!(hit.as_deref(), Some("France"), "{name}");
            assert_eq!(input_of(&out), &json!({"city": "Paris\n"}), "{name}");

            let (out, hit) = stream_and_batch(syntax, &["city"], &text);
            assert_eq!(hit, None, "{name}: a key is not text");
            assert_eq!(calls(&out), 1, "{name}: {out:#?}");
        }
    }

    /// Multibyte text on both sides of a stop, and multibyte stops, in
    /// prose and in a call's input: the stream holds back and cuts on
    /// char boundaries, and agrees with the batch cut, in every
    /// dialect.
    #[test]
    fn multibyte_stops_cut_on_char_boundaries() {
        for (name, syntax, prose) in call_dialects() {
            let text = format!(
                "{prose}{}",
                call(&syntax, json!({"city": "Zürich🦀日本é"})),
            );
            for (stop, want) in [
                ("🦀", "Zürich"),
                ("日本", "Zürich🦀"),
                ("é", "Zürich🦀日本"),
            ] {
                let (out, hit) =
                    stream_and_batch(syntax.clone(), &[stop], &text);
                assert_eq!(hit.as_deref(), Some(stop), "{name} {stop}");
                assert_eq!(
                    input_of(&out),
                    &json!({"city": want}),
                    "{name} {stop}"
                );
            }
            // A stop sharing a lead byte with the text (`ü` vs `é`,
            // both `C3 …`) never matches half a char.
            let (out, hit) = stream_and_batch(syntax.clone(), &["è"], &text);
            assert_eq!(hit, None, "{name}");
            assert_eq!(calls(&out), 1, "{name}");
        }
        for (name, syntax) in dialects() {
            let text = format!(
                "Grüße 🦀 日本語 done.\n\n{}",
                call(&syntax, json!({"city": "Paris"})),
            );
            let (out, hit) = stream_and_batch(syntax, &["日本"], &text);
            assert_eq!(hit.as_deref(), Some("日本"), "{name}");
            assert_eq!(texts(&out), "Grüße 🦀 ", "{name}");
            assert_eq!(calls(&out), 0, "{name}");
        }
    }

    /// The captured case, in every dialect: the stop fires *while the
    /// call is generated* — generation stops there, not when the call
    /// closes — the members before the match stand, the string it fell
    /// in is cut before it, and what would have followed never exists.
    #[test]
    fn stop_in_a_call_input_fires_before_the_call_closes() {
        let input = json!({
            "path": "hello.py",
            "contents": "import datetime\nprint(datetime.now())\n",
            "mode": "w",
        });
        for (name, syntax, prose) in call_dialects() {
            let text = format!("{prose}{}", call(&syntax, input.clone()));
            let mut f = filter(syntax.clone(), &["print("]);
            let (_, generated) = run_to_stop(&mut f, &text);
            assert!(
                generated.ends_with("print("),
                "{name}: the stop waited for more: {generated:?}",
            );
            let (out, hit) =
                stream_and_batch(syntax.clone(), &["print("], &text);
            assert_eq!(hit.as_deref(), Some("print("), "{name}");
            // Generation order: Gemma's dict renders keys sorted, so
            // `contents` comes before `path` there.
            let want = match syntax.family {
                Family::TagWithDict => r#"{"contents":"import datetime\n"}"#,
                _ => r#"{"path":"hello.py","contents":"import datetime\n"}"#,
            };
            let cut = serde_json::to_string(input_of(&out)).unwrap();
            assert_eq!(cut, want, "{name}");
        }
    }

    /// Nested input: the cut keeps the containers around the match
    /// with the members before it. (Qwen XML carries a non-string
    /// parameter as one JSON value, matched only once it closes.)
    #[test]
    fn stop_in_nested_input_cuts_at_every_depth() {
        let input = json!({
            "a": 1,
            "b": {"x": ["keep", "cut STOP here", "gone"], "y": "gone"},
            "c": "gone",
        });
        for (name, syntax, prose) in call_dialects() {
            if syntax.family == Family::TagWithTagged {
                continue;
            }
            let text = format!("{prose}{}", call(&syntax, input.clone()));
            let (out, hit) = stream_and_batch(syntax, &["STOP"], &text);
            assert_eq!(hit.as_deref(), Some("STOP"), "{name}");
            assert_eq!(
                input_of(&out),
                &json!({"a": 1, "b": {"x": ["keep", "cut "]}}),
                "{name}",
            );
        }
    }

    /// Parallel calls: a stop in the second cuts it; the first, which
    /// closed before it, stands unchanged.
    #[test]
    fn stop_in_a_later_call_keeps_the_earlier_one() {
        for (name, syntax) in dialects() {
            let paris = call(&syntax, json!({"city": "Paris"}));
            let oslo = call(&syntax, json!({"city": "Oslo STOP x"}));
            let text = format!("Both.\n\n{paris}\n{oslo}");
            let (out, hit) = stream_and_batch(syntax, &["STOP"], &text);
            assert_eq!(hit.as_deref(), Some("STOP"), "{name}");
            let inputs: Vec<_> = out
                .iter()
                .filter_map(|b| match b {
                    Block::ToolUse { call } => Some(&call.input),
                    _ => None,
                })
                .collect();
            assert_eq!(
                inputs,
                [&json!({"city": "Paris"}), &json!({"city": "Oslo "})],
                "{name}",
            );
        }
    }

    /// A clip inside a call returns it with the members that completed
    /// and leaves it open for a stream (no `content_block_stop`); a stop
    /// closes the call it cuts.
    #[test]
    fn a_clipped_call_is_left_open_and_a_stopped_one_closed() {
        for (name, syntax, prose) in call_dialects() {
            let text = format!(
                "{prose}{}",
                call(&syntax, json!({"city": "Paris", "detail": "sunny"})),
            );
            let clip = &text[..text.find("sunny").unwrap() + 2];
            let mut f = filter(syntax.clone(), &["zzz"]);
            let mut out: Vec<Block> = clip
                .chars()
                .flat_map(|c| f.push(&c.to_string(), None))
                .collect();
            out.extend(f.finish(true));
            assert_eq!(f.hit(), None, "{name}");
            assert_eq!(input_of(&out), &json!({"city": "Paris"}), "{name}");
            assert_eq!(
                f.open_call_json(),
                Some(r#"{"city":"Paris""#),
                "{name}"
            );

            let mut f = filter(syntax, &["un"]);
            let _ = run(&mut f, &text);
            assert_eq!(f.hit(), Some("un"), "{name}");
            assert_eq!(f.open_call_json(), None, "{name}");
        }
    }

    /// Reasoning is not text output: a stop inside a thought is not
    /// matched, and the answer after it is.
    #[test]
    fn stop_never_matches_thinking() {
        let text = "<think>\nwe END here\n</think>\nAnswer END x";
        let (out, hit) =
            stream_and_batch(CallSyntax::qwen_xml(), &["END"], text);
        assert_eq!(hit.as_deref(), Some("END"));
        assert!(
            out.iter().any(|b| matches!(
                b,
                Block::Thought { thought, .. } if thought.contains("END")
            )),
            "{out:#?}",
        );
        assert_eq!(texts(&out).trim(), "Answer");
    }

    /// A stop that overlaps a marker's prefix is held by the parser
    /// until the next piece settles it; at a clean end of turn the flush
    /// releases the tail as prose, and a stop there still counts.
    #[test]
    fn stop_in_the_flushed_marker_prefix_counts() {
        let (out, hit) =
            stream_and_batch(CallSyntax::qwen_xml(), &["<tool"], "Hi <tool");
        assert_eq!(hit.as_deref(), Some("<tool"));
        assert_eq!(texts(&out), "Hi ");
    }

    /// An incomplete call degraded to text by a clean-end flush is
    /// framing the model never finished, not prose: never matched — nor
    /// is the whitespace before it.
    #[test]
    fn degraded_call_is_never_matched() {
        let mut f = filter(CallSyntax::hermes_json(), &["city", "\n"]);
        let partial = concat!(
            "ok\n\n<tool_call>\n",
            r#"{"name": "get_weather", "arguments": {"city""#,
        );
        let out = run(&mut f, partial);
        assert_eq!(f.hit(), None, "{out:#?}");
        assert!(texts(&out).contains("city"), "Final degrades it: {out:#?}");
    }

    /// The batch cut: first match in prose or a call's input, thoughts
    /// skipped, whitespace beside a structure skipped, an emptied block
    /// dropped.
    #[test]
    fn cut_at_stop_cuts_the_first_match() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool();
        let parse = |text: &str| {
            parse_text(&syntax, &[&t], text, false, Leniency::Final).blocks
        };
        let blocks = parse("<think>\nEND?\n</think>\nfine END more");
        let (cut, hit) = cut_at_stop(blocks.clone(), &["END"]);
        assert_eq!(hit.as_deref(), Some("END"));
        assert_eq!(cut.len(), blocks.len());
        assert_eq!(texts(&cut).trim(), "fine");

        let (cut, hit) =
            cut_at_stop(vec!["END x".to_string().into()], &["END"]);
        assert_eq!(hit.as_deref(), Some("END"));
        assert!(cut.is_empty(), "{cut:#?}");

        let (cut, hit) = cut_at_stop(blocks.clone(), &["absent"]);
        assert_eq!((cut, hit), (blocks, None));

        let text = format!("A\n\n{}", call(&syntax, json!({"city": "x\ny"})));
        let blocks = parse(&text);
        let (cut, hit) = cut_at_stop(blocks.clone(), &["\n"]);
        assert_eq!(hit.as_deref(), Some("\n"));
        assert_eq!(cut[..1], blocks[..1], "the prose stands");
        assert_eq!(input_of(&cut), &json!({"city": "x"}), "the call is cut");
    }

    /// The raw cut lands where the cut output ends: just before a stop
    /// in prose (a copy of it in framing before that intact), or inside
    /// the call whose input matched, just before the match — before an
    /// escape the match began with, never after its dangling backslash.
    #[test]
    fn raw_stop_cut_ends_where_the_output_does() {
        let t = tool();
        let cut = |syntax: &CallSyntax, raw: &str, stop: &str| {
            let parse = |prefix: &str| {
                parse_text_open(syntax, &[&t], prefix, false, Leniency::Clipped)
            };
            let view = |prefix: &str| {
                let (parsed, open) = parse(prefix);
                let complete =
                    parsed.status == crate::dialect::ParseStatus::Complete;
                (complete || open.is_some())
                    .then(|| stop_view((parsed, open), true))
            };
            let (kept, hit) =
                cut_at_stop(stop_view(parse(raw), false), &[stop]);
            hit.and_then(|_| raw_stop_cut(raw, &kept, view))
        };

        let qwen = CallSyntax::qwen_xml();
        let call_ok = call(&qwen, json!({"city": "Paris"}));
        let raw = format!("A\n\n{call_ok}done\nlater");
        let want = format!("A\n\n{call_ok}done").len();
        assert_eq!(cut(&qwen, &raw, "\n"), Some(want));
        assert_eq!(cut(&qwen, "no stop", "\n"), None);

        // Generation stopped at the match, inside the call.
        for syntax in [qwen, CallSyntax::hermes_json()] {
            let full =
                call(&syntax, json!({"city": "import datetime\nprint(x)"}));
            let at = full.find("print(").unwrap();
            let raw = &full[..at + "print(".len()];
            assert_eq!(
                cut(&syntax, raw, "print("),
                Some(at),
                "{:?}",
                syntax.family
            );
        }
        // An escaped match: `\n` in JSON is two raw bytes.
        let syntax = CallSyntax::hermes_json();
        let full = call(&syntax, json!({"city": "Paris\nFrance"}));
        let at = full.find("\\n").unwrap();
        assert_eq!(cut(&syntax, &full[..at + 2], "\n"), Some(at));
    }

    /// The walk parses one prefix a step, so it stops at the text the
    /// cut output shows: no shorter prefix can parse to it. Without a
    /// match that is where it ends, not at the first byte.
    #[test]
    fn raw_stop_cut_walks_no_lower_than_the_kept_text() {
        let kept: Vec<Block> = vec!["a".repeat(1000).into()];
        let raw = format!("{}<junk>", "a".repeat(1000));
        let steps = std::cell::Cell::new(0);
        let never = |_: &str| {
            steps.set(steps.get() + 1);
            None
        };
        assert_eq!(raw_stop_cut(&raw, &kept, never), None);
        assert_eq!(steps.get(), "<junk>".len() + 1);
        assert_eq!(text_floor(&kept), 1000);
    }

    /// #122, streaming: a stop sequence split across deltas is held
    /// until it completes, then cut with everything after it; text that
    /// only looked like the start of one is released.
    #[test]
    fn stop_cutter_holds_back_and_cuts() {
        let mut c = StopCutter::new(vec!["</answer>".into()]);
        assert_eq!(c.push("The answer is 42"), "The answer is 42");
        assert_eq!(c.push(" </"), " ");
        assert_eq!(c.push("ans"), "");
        assert_eq!(c.push("wer> and more"), "");
        assert_eq!(c.hit(), Some("</answer>"));
        // Past the stop: nothing, ever.
        assert_eq!(c.push("still more"), "");
        assert_eq!(c.finish(), "");

        // A false start is released once it diverges.
        let mut c = StopCutter::new(vec!["</answer>".into()]);
        assert_eq!(c.push("a </a"), "a ");
        assert_eq!(c.push("bbr>"), "</abbr>");
        // A dangling prefix at the end of the run is output.
        assert_eq!(c.push(" </an"), " ");
        assert_eq!(c.finish(), "</an");
        assert_eq!(c.hit(), None);

        // No stops: a pass-through.
        let mut c = StopCutter::new(Vec::new());
        assert_eq!(c.push("x"), "x");
        assert_eq!(c.finish(), "");
    }
}
