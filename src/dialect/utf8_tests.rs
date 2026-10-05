//! Model output is arbitrary UTF-8: no parser may slice it mid-char.
//!
//! Every dialect parser is driven, batch (every [`Leniency`], with and
//! without pre-opened reasoning) and streaming (pushed in uneven
//! pieces, flushed both ways), over its own reference emission with a
//! multibyte char inserted at every char boundary, and over fuzzed
//! concatenations of its markers, marker prefixes and multibyte chars.
//! The only assertion is that nothing panics — a panic in a parser is a
//! panic in the server's request.

use super::{
    parse_text, render_reference, CallSyntax, Family, FunctionSyntax, Leniency,
    StreamParser,
};
use crate::Tool;
use serde_json::json;

/// A two-byte Latin, a four-byte emoji and three-byte CJK.
const MULTIBYTE: [&str; 3] = ["é", "🦀", "日本"];

fn tool() -> Tool {
    Tool::builder("get_weather")
        .description("test")
        .schema(json!({
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer"},
                "detail": {"type": "object"},
            },
            "required": ["city", "days"],
        }))
        .build()
        .expect("valid test tool")
}

/// Mistral Small 4's `[TOOL_CALLS]name[ARGS]{…}`.
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

fn dialects() -> [(&'static str, CallSyntax); 6] {
    [
        ("qwen_xml", CallSyntax::qwen_xml()),
        ("hermes_json", CallSyntax::hermes_json()),
        ("llama31_json", CallSyntax::llama31_json()),
        ("mistral", mistral()),
        ("gemma4", CallSyntax::gemma4()),
        ("harmony", CallSyntax::gpt_oss()),
    ]
}

/// A whole turn in `syntax`: a thought where the dialect has one,
/// prose, and a call with nested and escaped values.
fn emission(syntax: &CallSyntax) -> String {
    let input = json!({
        "city": "Pa\"ris\\",
        "days": 3,
        "detail": {"k": [1, "v"]},
    });
    let call =
        render_reference(syntax, &[("get_weather", &input)]).expect("renders");
    match syntax.family {
        Family::Harmony => format!(
            "<|channel|>analysis<|message|>think<|end|>\
             <|start|>assistant<|channel|>commentary<|message|>ok<|end|>\
             <|start|>assistant{}<|start|>assistant\
             <|channel|>final<|message|>done<|end|>",
            call.strip_prefix("<|start|>assistant").unwrap_or(&call),
        ),
        _ if !syntax.reasoning.start.is_empty() => format!(
            "{}hm{}Sure. {call} after",
            syntax.reasoning.start, syntax.reasoning.end
        ),
        _ => format!("Sure. {call} after"),
    }
}

/// Every marker of `syntax` a parser keys on.
fn markers(syntax: &CallSyntax) -> Vec<String> {
    let mut out: Vec<String> = [
        syntax.trigger(),
        &syntax.section_start,
        &syntax.section_end,
        &syntax.per_call_start,
        &syntax.per_call_end,
        &syntax.function.name_prefix,
        &syntax.function.name_suffix,
        &syntax.function.close,
        &syntax.reasoning.start,
        &syntax.reasoning.end,
        &syntax.tool_response_start,
    ]
    .into_iter()
    .chain(syntax.preserved_tokens.iter().map(String::as_str))
    .filter(|m| !m.is_empty())
    .map(str::to_string)
    .collect();
    if syntax.family == Family::Harmony {
        use super::harmony::*;
        out.extend(
            [
                START_ASSISTANT,
                CHANNEL,
                MESSAGE,
                END,
                CALL,
                RETURN,
                CONSTRAIN,
                TO_FUNCTIONS,
                COMMENTARY,
                "analysis",
                "final",
                " to=",
                "functions.get_weather",
            ]
            .map(str::to_string),
        );
    }
    out
}

/// Parse `text` every way a session can: batch under each leniency,
/// with and without pre-opened reasoning, and streamed in pieces of
/// `1 + (i % chunk)` chars, flushed both final and clipped.
fn parse_every_way(syntax: &CallSyntax, tool: &Tool, text: &str, chunk: usize) {
    for pre in [false, true] {
        for leniency in
            [Leniency::Final, Leniency::Streaming, Leniency::Clipped]
        {
            parse_text(syntax, &[tool], text, pre, leniency);
        }
        for clipped in [false, true] {
            let mut p =
                StreamParser::new(syntax.clone(), vec![tool.clone()], pre);
            let chars: Vec<(usize, char)> = text.char_indices().collect();
            let mut i = 0;
            let mut n = 0;
            while i < chars.len() {
                let j = (i + 1 + n % chunk).min(chars.len());
                let end = chars.get(j).map_or(text.len(), |&(at, _)| at);
                p.push(&text[chars[i].0..end]);
                i = j;
                n += 1;
            }
            if clipped {
                p.finish_clipped();
            } else {
                p.finish();
            }
        }
    }
}

/// The reported repro: Harmony prose outside a block that starts with a
/// non-ASCII char.
#[test]
fn harmony_prose_starting_non_ascii() {
    let syntax = CallSyntax::gpt_oss();
    let t = tool();
    for text in [
        "Über alles",
        "é",
        "<|channel|>final<|message|>hi<|end|>é",
        "<|start|>assistantñ",
        "<|start|>assistant🦀<|end|>",
    ] {
        parse_every_way(&syntax, &t, text, 3);
    }
}

/// A multibyte char at every boundary of each dialect's own turn, and
/// the turn cut right after it (what a stream re-parses mid-char-run).
#[test]
fn multibyte_at_every_boundary() {
    let t = tool();
    for (name, syntax) in dialects() {
        let full = emission(&syntax);
        let calls = parse_text(&syntax, &[&t], &full, false, Leniency::Final)
            .blocks
            .iter()
            .filter(|b| matches!(b, crate::Block::ToolUse { .. }))
            .count();
        assert_eq!(calls, 1, "{name}: the reference turn parses: {full:?}");
        let bounds: Vec<usize> = full
            .char_indices()
            .map(|(i, _)| i)
            .chain([full.len()])
            .collect();
        for &at in &bounds {
            for mb in MULTIBYTE {
                let text = format!("{}{mb}{}", &full[..at], &full[at..]);
                // Streaming over the whole turn at every insertion is
                // quadratic; the batch parses cover every prefix a
                // stream re-parses, the streams cover the deltas.
                for leniency in
                    [Leniency::Final, Leniency::Streaming, Leniency::Clipped]
                {
                    for pre in [false, true] {
                        parse_text(&syntax, &[&t], &text, pre, leniency);
                        let cut = &text[..at + mb.len()];
                        parse_text(&syntax, &[&t], cut, pre, leniency);
                    }
                }
                let mut p =
                    StreamParser::new(syntax.clone(), vec![t.clone()], false);
                p.push(&full[..at]);
                p.push(mb);
                p.push(&full[at..]);
                p.finish();
            }
        }
        // Whole streams, char by char and in uneven pieces, with
        // multibyte chars everywhere.
        let spiced: String = full
            .chars()
            .enumerate()
            .flat_map(|(i, c)| {
                [c.to_string(), MULTIBYTE[i % 3].to_string()]
                    .into_iter()
                    .take(if i % 4 == 0 { 2 } else { 1 })
            })
            .collect();
        for chunk in 1..=4 {
            parse_every_way(&syntax, &t, &spiced, chunk);
        }
    }
}

/// Cases per dialect for [`fuzzed_markers_and_multibyte`]:
/// `UTF8_FUZZ_CASES`, for a longer hunt than the gate's 3000.
fn fuzz_cases() -> usize {
    std::env::var("UTF8_FUZZ_CASES")
        .ok()
        .and_then(|n| n.parse().ok())
        .unwrap_or(3000)
}

/// Fuzzed concatenations of each dialect's markers, their proper
/// prefixes, JSON punctuation and multibyte chars.
#[test]
fn fuzzed_markers_and_multibyte() {
    let t = tool();
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for (name, syntax) in dialects() {
        let mut pool: Vec<String> = markers(&syntax)
            .iter()
            .flat_map(|m| {
                let prefixes =
                    m.char_indices().skip(1).map(|(i, _)| m[..i].to_string());
                prefixes.chain([m.clone()]).collect::<Vec<_>>()
            })
            .collect();
        pool.extend(
            [
                "get_weather",
                "{",
                "}",
                "[",
                "]",
                "\"",
                ":",
                ",",
                "\\",
                "\\u",
                "\\ud83e",
                "city",
                "days",
                "3",
                "null",
                "True",
                " ",
                "\n",
                "<",
                ">",
                "|",
                "=",
                "'",
            ]
            .map(str::to_string),
        );
        pool.extend(MULTIBYTE.map(str::to_string));
        for case in 0..fuzz_cases() {
            let len = 1 + (next() % 40) as usize;
            let text: String = (0..len)
                .map(|_| pool[(next() % pool.len() as u64) as usize].as_str())
                .collect();
            let run = std::panic::catch_unwind(|| {
                parse_every_way(&syntax, &t, &text, 1 + case % 3)
            });
            if let Err(panic) = run {
                eprintln!("{name}: case {case}: {text:?}");
                std::panic::resume_unwind(panic);
            }
        }
    }
}

/// Gemma 4: a turn-exit marker ahead of a channel open at the tail was
/// read as prose once the open arrived, so the stream's trailing text
/// was re-cut under it — mid-char past the `🦀`, a panic.
#[test]
fn gemma4_exit_marker_before_channel_open() {
    let syntax = CallSyntax::gemma4();
    let t = tool();
    let text = "<|tool_response>🦀<|tool_cal<|t<|channel>";
    parse_every_way(&syntax, &t, text, 3);
    for leniency in [Leniency::Final, Leniency::Streaming, Leniency::Clipped] {
        let parsed = parse_text(&syntax, &[&t], text, false, leniency);
        for block in &parsed.blocks {
            if let crate::Block::Text { text, .. } = block {
                assert!(
                    !text.contains("<|tool_response>"),
                    "{leniency:?}: the exit marker is envelope: {text:?}"
                );
            }
        }
    }
}
