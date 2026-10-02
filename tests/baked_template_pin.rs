//! Byte pins for every baked template (#120).
//!
//! Each [`baked::ALL`](drama_llama::baked::ALL) replacement renders a
//! fixed conversation dense in bools and nulls — tool-call arguments,
//! schema enums and defaults, thinking on and off — and the output must
//! equal `tests/fixtures/baked_pin/<name>.<case>.txt` byte for byte.
//! The pins were captured under minijinja 2.19, before 2.22 started
//! printing a bare bool or none Python-style (`True`/`None`): a served
//! render that drifts with the template engine breaks the prefix cache
//! for every transcript already in it.
//!
//! The date is the one input not fixed here (`strftime_now`), so its
//! value is masked before comparing. `PIN_BLESS=1` rewrites the pins —
//! only for a deliberate template change, never to absorb a library
//! bump.

use std::{borrow::Cow, num::NonZeroU32, path::PathBuf};

use drama_llama::dialect::analyze_template;
use drama_llama::prompt::{Content, Message, Role, ToolResult, ToolUse};
use drama_llama::{baked, Block, ChatTemplate, Prompt, RenderOptions, Tool};
use misanthropic::prompt::thinking::Thinking;
use serde_json::json;

/// BOS and EOS pieces each baked replacement is served with.
fn specials(name: &str) -> (&'static str, &'static str) {
    match name {
        "gemma4-cache-stable" => ("<bos>", "<turn|>"),
        "gptoss-cache-stable" | "gptoss-upstream-cache-stable" => {
            ("<|startoftext|>", "<|return|>")
        }
        "cogito-cache-stable" => ("", "<|im_end|>"),
        "mistral4-cache-stable" => ("<s>", "</s>"),
        "qwen3.6-cache-stable" | "qwen3.8-cache-stable" => ("", "<|im_end|>"),
        other => panic!("{other}: new baked template — add it to the pins"),
    }
}

fn pin_path(name: &str, case: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/baked_pin")
        .join(format!("{name}.{case}.txt"))
}

/// Masks the `YYYY-MM-DD` after each `Current date: `.
fn mask_date(render: &str) -> String {
    const KEY: &str = "Current date: ";
    render
        .split(KEY)
        .enumerate()
        .map(|(i, part)| match i {
            0 => part.to_owned(),
            _ => format!("{KEY}YYYY-MM-DD{}", part.get(10..).unwrap_or("")),
        })
        .collect()
}

fn tool() -> Tool {
    Tool::builder("configure")
        .description("Configure a thing.")
        .schema(json!({
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "flag": {"type": "boolean", "default": false},
                "maybe": {"type": ["string", "null"]},
                "mode": {"enum": ["on", null]},
                "level": {"type": "string", "enum": ["on", "off", null]},
                "nested": {"type": "object", "properties": {
                    "z": {"type": "integer"}, "a": {"type": "boolean"}}},
            },
            "required": ["name"],
        }))
        .build()
        .expect("valid test tool")
}

fn text(text: &'static str) -> Block {
    Block::Text {
        text: Cow::Borrowed(text),
        citations: None,
        cache_control: None,
    }
}

fn thought(thought: &'static str) -> Block {
    Block::Thought {
        thought: Cow::Borrowed(thought),
        signature: Cow::Borrowed(""),
    }
}

/// A finished tool turn, then a user follow-up, every scalar kind in
/// the call's arguments.
fn conversation(thinking: bool) -> Prompt {
    let think = |blocks: Vec<Block>, t: &'static str| match thinking {
        true => std::iter::once(thought(t)).chain(blocks).collect(),
        false => blocks,
    };
    let call = Block::ToolUse {
        call: ToolUse {
            id: Cow::Borrowed("call00001"),
            name: Cow::Borrowed("configure"),
            input: json!({
                "name": "unit",
                "flag": true,
                "off": false,
                "maybe": null,
                "mode": null,
                "list": [true, null, 1.5, "x"],
                "nested": {"z": 2, "a": false, "n": null},
            }),
            cache_control: None,
            caller: None,
        },
    };
    let result = Block::ToolResult {
        result: ToolResult {
            tool_use_id: Cow::Borrowed("call00001"),
            content: Content::text("ok"),
            is_error: false,
            cache_control: None,
        },
    };
    Prompt {
        system: Some(Content::text("Be brief.")),
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text("Configure it."),
            },
            Message {
                role: Role::Assistant,
                content: Content(think(
                    vec![text("Configuring."), call],
                    "weighing the flags",
                )),
            },
            Message {
                role: Role::User,
                content: Content(vec![result]),
            },
            Message {
                role: Role::Assistant,
                content: Content(think(vec![text("Done.")], "it worked")),
            },
            Message {
                role: Role::User,
                content: Content::text("Again?"),
            },
        ],
        tools: Some(vec![tool().into()]),
        thinking: thinking.then(|| Thinking::Enabled {
            budget_tokens: NonZeroU32::new(1024).expect("nonzero"),
            display: None,
        }),
        ..Default::default()
    }
}

#[test]
fn baked_templates_render_pinned_bytes() {
    let bless = std::env::var_os("PIN_BLESS").is_some();
    let drift: Vec<String> = baked::ALL
        .iter()
        .flat_map(|b| [(b, "thinking", true), (b, "plain", false)])
        .filter_map(|(b, case, thinking)| {
            let (bos, eos) = specials(b.name);
            let syntax = analyze_template(b.replacement, bos, eos)
                .expect("baked template analyzes");
            let template = ChatTemplate::from_source(
                b.replacement.to_owned(),
                bos.to_owned(),
                eos.to_owned(),
            )
            .expect("baked template compiles");
            // `Session::from_engine`'s options.
            let opts = RenderOptions::default()
                .with_generation_prompt(true)
                .with_extra("preserve_thinking", true)
                .with_thought_reingest(syntax.reasoning.reingest)
                .with_reasoning_start(syntax.reasoning.start.clone())
                .with_efforts(syntax.reasoning.efforts.clone());
            let render = template
                .render_with(&conversation(thinking), &opts)
                .map(|r| mask_date(&r))
                .unwrap_or_else(|e| panic!("{}.{case}: {e}", b.name));
            let path = pin_path(b.name, case);
            if bless {
                std::fs::create_dir_all(path.parent().expect("dir"))
                    .expect("create pin dir");
                std::fs::write(&path, &render).expect("write pin");
                return None;
            }
            let pinned = std::fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("{path:?}: {e}"));
            (render != pinned).then(|| {
                format!(
                    "{}.{case} drifted\n--- pinned ---\n{pinned}\n\
                     --- rendered ---\n{render}",
                    b.name
                )
            })
        })
        .collect();
    assert!(drift.is_empty(), "{}", drift.join("\n\n"));
}
