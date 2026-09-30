//! Request `stop_sequences` (#122), matched against what a client sees
//! as **text output**: prose [`Block::Text`] only.
//!
//! Never matched: dialect framing (`<tool_call>`, `<function=…>`,
//! `[TOOL_CALLS]`/`[ARGS]`, Harmony headers, EOG pieces), tool calls —
//! their input included — and thinking. A stop sequence exists to end
//! the *answer*: matched in framing, a stop of `"\n"` killed every Qwen
//! call at its opener; matched in reasoning, it would end a turn before
//! the answer began. (Anthropic's behavior inside `thinking` and
//! `tool_use` input is uncaptured; this is the conservative mapping —
//! a call the client asked for is never withheld on a stop.)
//!
//! A match is per prose *run*: a structure (thought, call) between two
//! stretches of text ends one run and starts the next, so a stop never
//! straddles one.

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

    /// End of the run: the held tail never completed a stop sequence,
    /// so it is output after all.
    pub(super) fn finish(&mut self) -> String {
        std::mem::take(&mut self.held)
    }
}

/// Stop sequences over a generation in flight: pieces go through the
/// dialect's [`StreamParser`], and only the prose it releases — never
/// framing held back as a possible marker, never a call or a thought —
/// is matched. Yields the parser's blocks with everything past a stop
/// dropped, so [`super::BlockStream`] streams from it directly and the
/// batch paths use it to know when to stop.
#[derive(Debug, Clone)]
pub(super) struct StopFilter {
    parser: StreamParser,
    cutter: StopCutter,
}

impl StopFilter {
    pub(super) fn new(parser: StreamParser, stops: Vec<String>) -> Self {
        Self {
            parser,
            cutter: StopCutter::new(stops),
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
    /// ([`StreamParser::finish_clipped`]).
    ///
    /// Otherwise the flush degrades an incomplete structure to text
    /// ([`StreamParser::finish`]) — framing, not prose the model
    /// finished, so it is never matched. Only the prose the flush
    /// releases (the tail held back as a possible marker) is: a stop
    /// found there, with calls withheld, stops the turn as though it
    /// had matched mid-stream.
    pub(super) fn finish(&mut self, clipped: bool) -> Vec<Block> {
        if clipped || self.hit().is_some() {
            return self.flush(true, true);
        }
        let mut probe = self.clone();
        let out = probe.flush(true, true);
        if probe.hit().is_some() {
            *self = probe;
            return out;
        }
        self.flush(false, false)
    }

    fn flush(&mut self, clipped: bool, matching: bool) -> Vec<Block> {
        let blocks = if clipped {
            self.parser.finish_clipped()
        } else {
            self.parser.finish()
        };
        let mut out = if matching {
            self.admit(blocks)
        } else {
            // The held run tail comes first: the flush continues it.
            prose(self.cutter.finish())
                .into_iter()
                .chain(blocks)
                .collect()
        };
        out.extend(prose(self.cutter.finish()));
        out
    }

    /// Match the prose among `blocks`; a structure ends the run.
    fn admit(&mut self, blocks: Vec<Block>) -> Vec<Block> {
        let mut out = Vec::new();
        for block in blocks {
            if self.hit().is_some() {
                break;
            }
            match block {
                Block::Text { text, .. } => {
                    out.extend(prose(self.cutter.push(&text)));
                }
                other => {
                    // The run's held tail never completed a stop.
                    out.extend(prose(self.cutter.finish()));
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
/// sequence in any prose run: that block keeps the text before the
/// match — dropped when nothing is left — and every block after it
/// goes. Returns the matched stop, if any.
pub(super) fn cut_at_stop<S: AsRef<str>>(
    mut blocks: Vec<Block>,
    stops: &[S],
) -> (Vec<Block>, Option<String>) {
    let hit = blocks.iter().enumerate().find_map(|(i, b)| match b {
        Block::Text { text, .. } => {
            first_stop_string(text, stops).map(|(at, s)| (i, at, s))
        }
        _ => None,
    });
    let Some((i, at, s)) = hit else {
        return (blocks, None);
    };
    blocks.truncate(i + 1);
    if at == 0 {
        blocks.pop();
    } else if let Some(Block::Text { text, .. }) = blocks.last_mut() {
        text.to_mut().truncate(at);
    }
    (blocks, Some(stops[s].as_ref().to_owned()))
}

/// Byte offset at which to cut `raw` — the generation's raw bytes,
/// framing and all — so it ends just before its stop sequence, or
/// `None` when no stop is visible. `visible` names the stop a prefix of
/// `raw` shows as text output ([`cut_at_stop`] over its parse).
///
/// The stop's last byte is where the smallest prefix still showing it
/// ends; the walk back from the end is short, since generation stops
/// within a piece or a held-back marker of the match.
pub(super) fn raw_stop_cut(
    raw: &str,
    visible: impl Fn(&str) -> Option<String>,
) -> Option<usize> {
    visible(raw)?;
    let end = raw
        .char_indices()
        .rev()
        .map(|(i, _)| i)
        .take_while(|&i| visible(&raw[..i]).is_some())
        .last()
        .unwrap_or(raw.len());
    visible(&raw[..end]).map(|stop| end - stop.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dialect::{
        parse_text, render_reference, CallSyntax, Leniency, StreamParser,
    };
    use crate::Tool;

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
            .schema(serde_json::json!({
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            }))
            .build()
            .expect("valid test tool")
    }

    fn filter(syntax: CallSyntax, stops: &[&str]) -> StopFilter {
        StopFilter::new(
            StreamParser::new(syntax, vec![tool()], false),
            stops.iter().map(|s| s.to_string()).collect(),
        )
    }

    /// Feed `text` a char at a time — every piece boundary a tokenizer
    /// could produce — then finish.
    fn run(f: &mut StopFilter, text: &str) -> Vec<Block> {
        let mut out: Vec<Block> = text
            .chars()
            .flat_map(|c| f.push(c.encode_utf8(&mut [0; 4])))
            .collect();
        out.extend(f.finish(false));
        out
    }

    fn calls(blocks: &[Block]) -> usize {
        blocks
            .iter()
            .filter(|b| matches!(b, Block::ToolUse { .. }))
            .count()
    }

    /// A stop of `"\n"` must not kill a Qwen XML call at its opener
    /// (`<tool_call>\n`), nor anywhere in its framing or input: none of
    /// that is text output. The call parses whole.
    #[test]
    fn newline_stop_never_matches_qwen_call_framing() {
        let syntax = CallSyntax::qwen_xml();
        let input = serde_json::json!({"city": "Paris\nFrance"});
        let call =
            render_reference(&syntax, &[("get_weather", &input)]).unwrap();
        assert!(call.contains('\n'), "the fixture must carry newlines");

        let mut f = filter(syntax, &["\n"]);
        let out = run(&mut f, &call);
        assert_eq!(f.hit(), None, "{out:#?}");
        assert_eq!(calls(&out), 1, "{out:#?}");
    }

    /// Prose before the call is text output, so a stop there does fire —
    /// and everything past it, the call included, is gone.
    #[test]
    fn stop_in_prose_before_a_call_cuts_the_call() {
        let syntax = CallSyntax::qwen_xml();
        let call = render_reference(
            &syntax,
            &[("get_weather", &serde_json::json!({"city": "Oslo"}))],
        )
        .unwrap();
        let mut f = filter(syntax, &["Let me"]);
        let out = run(&mut f, &format!("Sure. Let me check.{call}"));
        assert_eq!(f.hit(), Some("Let me"));
        assert_eq!(calls(&out), 0, "{out:#?}");
        assert_eq!(texts(&out), "Sure. ");
    }

    /// Reasoning is not text output: a stop inside a thought is not
    /// matched, and the answer after it is.
    #[test]
    fn stop_never_matches_thinking() {
        let mut f = filter(CallSyntax::qwen_xml(), &["END"]);
        let out = run(&mut f, "<think>\nwe END here\n</think>\nAnswer END x");
        assert_eq!(f.hit(), Some("END"));
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
        let mut f = filter(CallSyntax::qwen_xml(), &["<tool"]);
        let out = run(&mut f, "Hi <tool");
        assert_eq!(f.hit(), Some("<tool"));
        assert_eq!(texts(&out), "Hi ");
    }

    /// An incomplete call degraded to text by a clean-end flush is
    /// framing the model never finished, not prose: never matched.
    #[test]
    fn degraded_call_is_never_matched() {
        let syntax = CallSyntax::hermes_json();
        let mut f = filter(syntax, &["city"]);
        let partial = concat!(
            "ok <tool_call>\n",
            r#"{"name": "get_weather", "arguments": {"city""#,
        );
        let out = run(&mut f, partial);
        assert_eq!(f.hit(), None, "{out:#?}");
        assert!(texts(&out).contains("city"), "Final degrades it: {out:#?}");
    }

    /// Batch-side cut agrees with the stream: first match in prose,
    /// thoughts and calls skipped, an emptied block dropped.
    #[test]
    fn cut_at_stop_cuts_the_first_prose_match() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool();
        let text = "<think>\nEND?\n</think>\nfine END more";
        let blocks =
            parse_text(&syntax, &[&t], text, false, Leniency::Final).blocks;
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
    }

    /// The raw cut lands just before the stop's bytes, framing before it
    /// intact, even when a copy of the stop sits in that framing.
    #[test]
    fn raw_stop_cut_skips_framing() {
        let syntax = CallSyntax::qwen_xml();
        let t = tool();
        let call = render_reference(
            &syntax,
            &[("get_weather", &serde_json::json!({"city": "a\nb"}))],
        )
        .unwrap();
        let raw = format!("{call}done\nlater");
        let visible = |prefix: &str| {
            let parsed =
                parse_text(&syntax, &[&t], prefix, false, Leniency::Clipped);
            cut_at_stop(parsed.blocks, &["\n"]).1
        };
        let at = raw_stop_cut(&raw, visible).expect("stop is visible");
        assert_eq!(&raw[..at], format!("{call}done"));
        assert_eq!(raw_stop_cut("no stop", |_| None::<String>), None);
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
