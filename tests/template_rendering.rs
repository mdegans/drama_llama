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
    let tool = Tool::builder("count_letters")
        .description(
            "Count the number of times a letter appears in a string.",
        )
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "the letter to count"},
                "string": {"type": "string", "description": "the string to search"}
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool");
    let prompt = Prompt {
        system: Some(Content::text(
            "You are a helpful assistant. You cannot count letters in a \
             word reliably on your own because you see in tokens, not \
             letters. Use the `count_letters` tool when asked to count \
             characters.",
        )),
        messages: vec![Message {
            role: Role::User,
            content: Content::text("Count the number of r's in 'strawberry'"),
        }],
        tools: Some(vec![tool.into()]),
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

    let tool = Tool::builder("count_letters")
        .description(
            "Count the number of times a letter appears in a string.",
        )
        .schema(json!({
            "type": "object",
            "properties": {
                "letter": {"type": "string", "description": "the letter to count"},
                "string": {"type": "string", "description": "the string to search"}
            },
            "required": ["letter", "string"]
        }))
        .build()
        .expect("valid test tool");
    let call_id = "call_3_r";
    let prompt = Prompt {
        system: Some(Content::text("You are a helpful assistant.")),
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text(
                    "Count the number of r's in 'strawberry'",
                ),
            },
            Message {
                role: Role::Assistant,
                content: Content(vec![Block::ToolUse {
                    call: ToolUse {
                        id: Cow::Borrowed(call_id),
                        name: Cow::Borrowed("count_letters"),
                        input: json!({"letter": "r", "string": "strawberry"}),
                        cache_control: None,
                        caller: None,
                    },
                }]),
            },
            Message {
                role: Role::User,
                content: Content(vec![Block::ToolResult {
                    result: ToolResult {
                        tool_use_id: Cow::Borrowed(call_id),
                        content: Content::text("3"),
                        is_error: false,
                        cache_control: None,
                    },
                }]),
            },
        ],
        tools: Some(vec![tool.into()]),
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
    use misanthropic::prompt::thinking::Thinking;
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
        content: Content::text("hi"),
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
        thinking: Some(Thinking::Enabled {
            budget_tokens: NonZeroU32::new(1024).unwrap(),
            display: None,
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

/// `render_reference` claims, in its own doc comment, to produce "the
/// exact bytes the chat template's re-render will produce for these
/// calls, and the exact bytes `grammar_source`'s grammar forces". The
/// grammar half of that is exercised all over `dialect::parse`'s
/// tests; the *template* half never was — every caller of
/// `render_reference` is a test that compares it against itself or
/// against the parser, never against a real template render.
///
/// Written failing as the acceptance criterion for the #85 fix; if it
/// trips again, a replayed tool-call turn no longer re-renders
/// byte-stable, the prefix cache's auto-tip is discarded, and reuse
/// collapses to the last `cache_control` breakpoint.
///
/// Two independent divergences, both measured 2026-07-27, both fixed:
///
/// 1. **Whitespace.** The shared JSON prelude's `ws ::= [ \t\n\r]?`
///    let the model emit `": "` where the serializer emits `":"` —
///    the emission was under-determined by our own grammar. Fixed by
///    `json_grammar_canonical` pinning the argument interior, with
///    `KV_SEP`/`FIELD_SEP` shared between the grammar emitter and
///    `render_reference` so the envelope cannot drift either.
/// 2. **Escaping.** minijinja's `tojson` followed Jinja2 in being
///    HTML-safe, escaping `'`, `&`, `<`, `>` to `\u0027` etc., which
///    neither the model nor `render_reference` does. Fixed by
///    `tojson_unescaped`.
///
/// The payload carries all of those characters deliberately — every
/// pre-#85 payload in this suite was clean ASCII, which is why the gap
/// survived so long.
///
/// Model-free: the pinned fixture template is the whole input.
#[test]
fn render_reference_matches_template_tool_call_render() {
    use drama_llama::dialect::{analyze_template, render_reference};

    let source = std::fs::read_to_string(
        fixtures_dir().join("cogito_14b_template.jinja"),
    )
    .expect("cogito template fixture missing");
    let syntax = analyze_template(&source, "", "<|im_end|>")
        .expect("cogito template analyzes");

    // Shaped like a real Agora call: string, and a bool to catch the
    // `": "` vs `":"` divergence on a non-string value too.
    // Characters the real Agora payload carries and that JSON
    // serializers disagree about: an apostrophe and `<`/`>`/`&`
    // (Jinja2's `tojson` is HTML-safe and escapes them) and a
    // non-ASCII arrow (escaped when `ensure_ascii` is on).
    let input = json!({
        "community": "debate",
        "body": "x's belief in P & Q <br> stability \u{2192} contradiction",
        "is_proposal": false,
    });
    let reference = render_reference(&syntax, &[("create_post", &input)])
        .expect("call is representable");

    let tmpl = load_template();
    let prompt = Prompt {
        messages: vec![
            Message {
                role: Role::User,
                content: Content::text("post something"),
            },
            Message {
                role: Role::Assistant,
                content: Content(vec![drama_llama::Block::ToolUse {
                    call: misanthropic::tool::Use {
                        id: Cow::Borrowed("call_0_create_post"),
                        name: Cow::Borrowed("create_post"),
                        input: input.clone(),
                        cache_control: None,
                        caller: None,
                    },
                }]),
            },
        ],
        ..Prompt::default()
    };
    let rendered = tmpl
        .render_with(
            &prompt,
            &RenderOptions::default().with_generation_prompt(false),
        )
        .expect("render");

    assert!(
        rendered.contains(&reference),
        "template render does not contain render_reference's canonical \
         bytes — the documented invariant is violated, and every \
         replayed tool-call turn loses its auto-tip (#85).\n\
         \n--- render_reference ---\n{reference}\
         \n\n--- template render ---\n{rendered}",
    );
}
