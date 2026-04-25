//! Byte-exact template rendering fixtures for Phase 0.3.
//!
//! Each shape under `tests/fixtures/shapes/` is a pair:
//!
//! * `NN_<name>.vars.json` — the Jinja variables (messages, tools,
//!   bos/eos, date_string, extras) for the Python jinja2 cross-check.
//! * `NN_<name>.expected.txt` — the rendered prompt we expect
//!   drama_llama's minijinja pipeline to produce. Generated via the
//!   Python jinja2 reference renderer at
//!   `tests/fixtures/render_jinja.py` and committed verbatim.
//!
//! Regular unit tests (no `#[ignore]`) construct an equivalent
//! [`Prompt`] and render it via [`ChatTemplate::from_source`] +
//! [`ChatTemplate::render_with`], then assert byte-equality with the
//! committed `expected.txt`. No model load required — the pinned
//! template source lives at
//! `tests/fixtures/cogito_14b_template.jinja`.
//!
//! Ignored tests (`--ignored`) re-run the Python jinja2 renderer and
//! cross-check byte-equality. Useful for catching drift if the
//! committed fixture ever falls out of sync with the Python output
//! (e.g. after a jinja2 upstream fix we want to match). Requires
//! `uv` on PATH.
//!
//! [`Prompt`]: drama_llama::Prompt

use std::{borrow::Cow, path::PathBuf};

use drama_llama::{
    ChatTemplate, Content, Message, Prompt, RenderOptions, Role, Tool,
};
use serde_json::json;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_template() -> ChatTemplate {
    let source = std::fs::read_to_string(
        fixtures_dir().join("cogito_14b_template.jinja"),
    )
    .expect("cogito template fixture missing");
    // bos/eos match what cogito:14b's GGUF advertises — the template
    // doesn't actually use `{{ bos_token }}` so the empty string is
    // fine; `{{ eos_token }}` never fires in the tested shapes.
    ChatTemplate::from_source(source, String::new(), "<|im_end|>".to_string())
        .expect("template compiles")
}

fn load_expected(name: &str) -> String {
    let path = fixtures_dir()
        .join("shapes")
        .join(format!("{name}.expected.txt"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("missing fixture {path:?}: {e}"))
}

/// Shape 3: strawberry turn 1 — system + user + one tool, grammar-
/// forced call comes on the next inference pass. This is the cogito
/// production path for tool use.
#[test]
fn shape_03_strawberry_turn_1_matches_fixture() {
    let tool = Tool {
        name: Cow::Borrowed("count_letters"),
        description: Cow::Borrowed(
            "Count the number of times a letter appears in a string.",
        ),
        schema: json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "the letter to count"},
                "string": {"type": "string", "description": "the string to search"}
            },
            "required": ["letter", "string"]
        }),
        cache_control: None,
        strict: None,
    };
    let prompt = Prompt {
        system: Some(Content::SinglePart(Cow::Borrowed(
            "You are a helpful assistant. You cannot count letters in a \
             word reliably on your own because you see in tokens, not \
             letters. Use the `count_letters` tool when asked to count \
             characters.",
        ))),
        messages: vec![Message {
            role: Role::User,
            content: Content::SinglePart(Cow::Borrowed(
                "Count the number of r's in 'strawberry'",
            )),
        }],
        functions: Some(vec![tool]),
        ..Default::default()
    };
    let opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_date("17 Apr 2026")
        .with_extra("enable_thinking", true);

    let actual = load_template().render_with(&prompt, &opts).expect("render");
    let expected = load_expected("03_strawberry_turn_1");

    assert_eq!(
        actual, expected,
        "rendered output did not match fixture\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    );
}

/// Shape 4: strawberry turn 2 — the follow-up inference pass after
/// the tool call resolved. Exercises the assistant `ToolUse` +
/// subsequent `ToolResult` rendering path, which is a separate branch
/// in `build_messages` from shape 3.
#[test]
fn shape_04_strawberry_turn_2_matches_fixture() {
    use drama_llama::prompt::{ToolResult, ToolUse};
    use drama_llama::Block;

    let tool = Tool {
        name: Cow::Borrowed("count_letters"),
        description: Cow::Borrowed(
            "Count the number of times a letter appears in a string.",
        ),
        schema: json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "the letter to count"},
                "string": {"type": "string", "description": "the string to search"}
            },
            "required": ["letter", "string"]
        }),
        cache_control: None,
        strict: None,
    };
    let call_id = "call_3_r";
    let prompt = Prompt {
        system: Some(Content::SinglePart(Cow::Borrowed(
            "You are a helpful assistant.",
        ))),
        messages: vec![
            Message {
                role: Role::User,
                content: Content::SinglePart(Cow::Borrowed(
                    "Count the number of r's in 'strawberry'",
                )),
            },
            Message {
                role: Role::Assistant,
                content: Content::MultiPart(vec![Block::ToolUse {
                    call: ToolUse {
                        id: Cow::Borrowed(call_id),
                        name: Cow::Borrowed("count_letters"),
                        input: json!({"letter": "r", "string": "strawberry"}),
                        cache_control: None,
                    },
                }]),
            },
            Message {
                role: Role::User,
                content: Content::MultiPart(vec![Block::ToolResult {
                    result: ToolResult {
                        tool_use_id: Cow::Borrowed(call_id),
                        content: Content::SinglePart(Cow::Borrowed("3")),
                        is_error: false,
                        cache_control: None,
                    },
                }]),
            },
        ],
        functions: Some(vec![tool]),
        ..Default::default()
    };
    let opts = RenderOptions::default()
        .with_generation_prompt(true)
        .with_date("17 Apr 2026")
        .with_extra("enable_thinking", true);

    let actual = load_template().render_with(&prompt, &opts).expect("render");
    let expected = load_expected("04_strawberry_turn_2");

    assert_eq!(
        actual, expected,
        "rendered output did not match fixture\n--- actual ---\n{actual}\n--- expected ---\n{expected}"
    );
}

/// Cross-check: each committed `expected.txt` must also round-trip
/// through Python jinja2. Catches fixture drift vs the reference
/// implementation. One ignored test drives all shapes.
#[test]
#[ignore = "requires uv on PATH"]
fn all_shapes_match_python_jinja2() {
    let fixtures = fixtures_dir();
    let script = fixtures.join("render_jinja.py");
    let tmpl = fixtures.join("cogito_14b_template.jinja");
    let shapes_dir = fixtures.join("shapes");

    let shape_names = ["03_strawberry_turn_1", "04_strawberry_turn_2"];
    for name in shape_names {
        let vars = shapes_dir.join(format!("{name}.vars.json"));
        let output = std::process::Command::new(&script)
            .arg(&tmpl)
            .arg(&vars)
            .output()
            .unwrap_or_else(|e| {
                panic!("exec render_jinja.py for {name}: {e} (is uv on PATH?)")
            });
        assert!(
            output.status.success(),
            "render_jinja.py {name} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let py_output = String::from_utf8(output.stdout).expect("utf8 output");
        let expected = load_expected(name);
        assert_eq!(
            py_output, expected,
            "python jinja2 output drifted from committed fixture for {name}"
        );
    }
}

/// `prompt.thinking` drives the `enable_thinking` Jinja variable in
/// templates that gate a `<think>` block on it (Qwen3 family).
/// Anthropic semantics: `None` → disabled, `Some(_)` → enabled.
/// Caller's `with_extra("enable_thinking", _)` wins over the derived
/// value so opt-out and explicit-override paths both work.
#[test]
fn enable_thinking_derives_from_prompt_thinking() {
    use misanthropic::prompt::thinking::{Kind, Thinking};
    use std::num::NonZeroU32;

    // Tiny template that echoes whatever value `enable_thinking` ends
    // up bound to in the Jinja context. Independent of any model
    // template — we're testing the wiring, not a downstream template.
    let tmpl = ChatTemplate::from_source(
        "thinking={{ enable_thinking }}".to_string(),
        String::new(),
        String::new(),
    )
    .expect("template compiles");

    let user_msg = Message {
        role: Role::User,
        content: Content::SinglePart(Cow::Borrowed("hi")),
    };

    // 1. thinking=None (default) → enable_thinking=false
    let prompt_off = Prompt {
        messages: vec![user_msg.clone()],
        ..Default::default()
    };
    assert_eq!(
        tmpl.render_with(&prompt_off, &RenderOptions::default())
            .expect("render"),
        "thinking=false",
        "prompt.thinking=None must render as enable_thinking=false"
    );

    // 2. thinking=Some(...) → enable_thinking=true
    let prompt_on = Prompt {
        messages: vec![user_msg.clone()],
        thinking: Some(Thinking {
            budget_tokens: NonZeroU32::new(1024).unwrap(),
            kind: Kind::Enabled,
        }),
        ..Default::default()
    };
    assert_eq!(
        tmpl.render_with(&prompt_on, &RenderOptions::default())
            .expect("render"),
        "thinking=true",
        "prompt.thinking=Some(_) must render as enable_thinking=true"
    );

    // 3. Caller-set extra wins — None + extras=true → true.
    assert_eq!(
        tmpl.render_with(
            &prompt_off,
            &RenderOptions::default().with_extra("enable_thinking", true)
        )
        .expect("render"),
        "thinking=true",
        "explicit with_extra=true must override derived false"
    );

    // 4. Caller-set extra wins — Some + extras=false → false.
    assert_eq!(
        tmpl.render_with(
            &prompt_on,
            &RenderOptions::default().with_extra("enable_thinking", false)
        )
        .expect("render"),
        "thinking=false",
        "explicit with_extra=false must override derived true"
    );
}
