//! Request `stop_sequences` (#122), matched against what a client sees
//! as **text**: prose ([`Block::Text`]) and the string values of a tool
//! call's input.
//!
//! - A match in prose ends the turn there: the text before it stands,
//!   everything after it goes.
//! - A match in a call's input **withholds that call**, as a clip does
//!   (#121): the prose before it stands; the call and everything after
//!   it go. Only whole string values are matched, each on its own —
//!   never the keys, the JSON around them, or the dialect's spelling of
//!   them.
//! - Either way the turn reads `stop_reason: stop_sequence`.
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
//! stop sequence must contain non-whitespace`, captured 2026-09-30), and
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
    dialect::StreamParser,
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

    /// Match a call's input (see [`input_stop`]); a match is a hit.
    fn check_input(&mut self, input: &Value) -> bool {
        if self.hit.is_none() {
            self.hit = input_stop(input, &self.stops).map(str::to_owned);
        }
        self.hit.is_some()
    }

    /// End of the run: the held tail never completed a stop sequence,
    /// so it is output after all.
    pub(super) fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// The first stop sequence found in a call's input: in its string
/// values, each matched on its own, keys and JSON syntax never.
fn input_stop<'s, S: AsRef<str>>(
    value: &Value,
    stops: &'s [S],
) -> Option<&'s str> {
    match value {
        Value::String(s) => {
            first_stop_string(s, stops).map(|(_, i)| stops[i].as_ref())
        }
        Value::Array(items) => items.iter().find_map(|v| input_stop(v, stops)),
        Value::Object(map) => map.values().find_map(|v| input_stop(v, stops)),
        _ => None,
    }
}

/// Stop sequences over a generation in flight: pieces go through the
/// dialect's [`StreamParser`], and only what it releases as text — prose
/// and, once a call closes, its input's string values — is matched;
/// never framing held back as a possible marker, never whitespace
/// between prose and a structure, never a thought. Yields the parser's
/// blocks with everything past a stop dropped (a call whose input
/// matched included), so [`super::BlockStream`] streams from it directly
/// and the batch paths use it to know when to stop.
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

    /// Feed one piece; returns the blocks (or prose deltas) it resolved.
    pub(super) fn push(&mut self, piece: &str) -> Vec<Block> {
        if self.hit().is_some() {
            return Vec::new();
        }
        let blocks = self.parser.push(piece);
        self.admit(blocks)
    }

    /// End of generation. `clipped`: the generation was cut short, so
    /// an incomplete trailing call is withheld
    /// ([`StreamParser::finish_clipped`]) and the run's trailing
    /// whitespace is not matched.
    ///
    /// Otherwise the flush degrades an incomplete structure to text
    /// ([`StreamParser::finish`]) — framing, not prose the model
    /// finished, so it is never matched. Only the prose the flush
    /// releases (the tail held back as a possible marker) and the
    /// whitespace that ended the answer are: a stop found there, with
    /// calls withheld, stops the turn as though it had matched
    /// mid-stream.
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
        if turn_end && self.hit().is_none() && !self.parser.withholds() {
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
                other => {
                    // The run's held tail never completed a stop, and
                    // its trailing whitespace is framing.
                    out.extend(self.release());
                    self.after_structure = true;
                    if let Block::ToolUse { call } = &other {
                        if self.cutter.check_input(&call.input) {
                            // Withheld, as a clip withholds a call.
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

/// Cut `blocks` (adjacent prose already merged) at the first stop
/// sequence, by [`StopFilter`]'s rules — the batch half of the same
/// policy. A match in prose keeps the text before it (the block dropped
/// when nothing is left); a match in a call's input drops the call.
/// Every block after the cut goes. Returns the matched stop, if any.
///
/// The last block's trailing whitespace is matched: `blocks` is taken
/// to end a turn that finished cleanly. A caller holding the stop the
/// filter reported can pass just that one — the filter found it first,
/// so it is the first match here too.
pub(super) fn cut_at_stop<S: AsRef<str>>(
    mut blocks: Vec<Block>,
    stops: &[S],
) -> (Vec<Block>, Option<String>) {
    let is_structure =
        |b: Option<&Block>| b.is_some_and(|b| !matches!(b, Block::Text { .. }));
    let hit = blocks.iter().enumerate().find_map(|(i, b)| match b {
        Block::Text { text, .. } => {
            // Whitespace touching a structure is framing.
            let start = if i > 0 && is_structure(blocks.get(i - 1)) {
                text.len() - text.trim_start().len()
            } else {
                0
            };
            let end = if is_structure(blocks.get(i + 1)) {
                text.trim_end().len()
            } else {
                text.len()
            };
            let body = text.get(start..end.max(start)).unwrap_or("");
            first_stop_string(body, stops)
                .map(|(at, s)| (i, Some(start + at), s))
        }
        Block::ToolUse { call } => input_stop(&call.input, stops).map(|hit| {
            let s = stops.iter().position(|s| s.as_ref() == hit);
            (i, None, s.expect("the stop came from `stops`"))
        }),
        _ => None,
    });
    let Some((i, at, s)) = hit else {
        return (blocks, None);
    };
    match at {
        // In a call's input: the call goes.
        None => blocks.truncate(i),
        Some(0) => blocks.truncate(i),
        Some(at) => {
            blocks.truncate(i + 1);
            if let Some(Block::Text { text, .. }) = blocks.last_mut() {
                text.to_mut().truncate(at);
            }
        }
    }
    (blocks, Some(stops[s].as_ref().to_owned()))
}

/// Byte offset at which to cut `raw` — the generation's raw bytes,
/// framing and all — so that it parses to `kept`, the output
/// [`cut_at_stop`] left. `view` is a prefix's parse (adjacent prose
/// merged), or `None` when it withholds a structure in flight.
///
/// The longest such prefix: the walk back from the end is short, since
/// generation stops within a piece, a held-back marker, or (a match in
/// a call's input) the call of the match.
pub(super) fn raw_stop_cut(
    raw: &str,
    kept: &[Block],
    view: impl Fn(&str) -> Option<Vec<Block>>,
) -> Option<usize> {
    raw.char_indices()
        .map(|(i, _)| i)
        .chain([raw.len()])
        .rev()
        .find(|&i| view(&raw[..i]).as_deref() == Some(kept))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{
        parse_text, render_reference, CallSyntax, Family, FunctionSyntax,
        Leniency, StreamParser,
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
    /// could produce — then finish a turn that ended cleanly (EOG).
    fn run(f: &mut StopFilter, text: &str) -> Vec<Block> {
        let mut out: Vec<Block> = text
            .chars()
            .flat_map(|c| f.push(c.encode_utf8(&mut [0; 4])))
            .collect();
        out.extend(f.finish(false));
        super::super::merge_adjacent_prose(out)
    }

    fn calls(blocks: &[Block]) -> usize {
        blocks
            .iter()
            .filter(|b| matches!(b, Block::ToolUse { .. }))
            .count()
    }

    /// Stream `text` through the filter, then check the batch half
    /// agrees: the clipped parse cut at the stop the filter reported
    /// is the streamed output, block for block. Returns the output.
    fn stream_and_batch(
        syntax: CallSyntax,
        stops: &[&str],
        text: &str,
    ) -> (Vec<Block>, Option<String>) {
        let mut f = filter(syntax.clone(), stops);
        let streamed = run(&mut f, text);
        let hit = f.hit().map(str::to_owned);
        if let Some(hit) = &hit {
            let t = tool();
            let parsed =
                parse_text(&syntax, &[&t], text, false, Leniency::Clipped);
            let blocks = super::super::merge_adjacent_prose(parsed.blocks);
            let (cut, found) = cut_at_stop(blocks, &[hit]);
            assert_eq!(found.as_ref(), Some(hit), "{text:?}");
            assert_eq!(cut, streamed, "batch vs stream on {text:?}");
        }
        (streamed, hit)
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
                .flat_map(|c| f.push(&c.to_string()))
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

    /// A call's input values are text: a stop in one withholds that
    /// call, as a clip does, and the turn stops there. The prose before
    /// it stands; keys and JSON syntax are never matched.
    #[test]
    fn stop_in_a_call_input_withholds_the_call() {
        for (name, syntax) in dialects() {
            let text = format!(
                "Sure, checking.\n\n{}",
                call(&syntax, json!({"city": "Paris\nFrance"})),
            );
            let (out, hit) = stream_and_batch(syntax.clone(), &["\n"], &text);
            assert_eq!(hit.as_deref(), Some("\n"), "{name}");
            assert_eq!(calls(&out), 0, "{name}: {out:#?}");
            assert_eq!(texts(&out), "Sure, checking.\n\n", "{name}");

            let (out, hit) =
                stream_and_batch(syntax.clone(), &["France"], &text);
            assert_eq!(hit.as_deref(), Some("France"), "{name}");
            assert_eq!(calls(&out), 0, "{name}: {out:#?}");

            let (out, hit) = stream_and_batch(syntax, &["city"], &text);
            assert_eq!(hit, None, "{name}: a key is not text");
            assert_eq!(calls(&out), 1, "{name}: {out:#?}");
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
        assert_eq!(cut, blocks[..1], "the call goes, the prose stands");
    }

    /// The raw cut lands where the cut output ends: just before a stop
    /// in prose (a copy of it in framing before that intact), or before
    /// the call whose input matched.
    #[test]
    fn raw_stop_cut_ends_where_the_output_does() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool();
        let view = |prefix: &str| {
            let parsed =
                parse_text(&syntax, &[&t], prefix, false, Leniency::Clipped);
            (parsed.status == crate::dialect::ParseStatus::Complete)
                .then(|| super::super::merge_adjacent_prose(parsed.blocks))
        };
        let cut = |raw: &str| {
            let blocks = view(raw).expect("complete");
            let (kept, hit) = cut_at_stop(blocks, &["\n"]);
            hit.and_then(|_| raw_stop_cut(raw, &kept, view))
        };

        let call_ok = call(&syntax, json!({"city": "Paris"}));
        let raw = format!("A\n\n{call_ok}done\nlater");
        assert_eq!(cut(&raw), Some(format!("A\n\n{call_ok}done").len()));

        let raw = format!("A\n\n{}", call(&syntax, json!({"city": "a\nb"})));
        assert_eq!(cut(&raw), Some("A\n\n".len()));

        assert_eq!(cut("no stop"), None);
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
