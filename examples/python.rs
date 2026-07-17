//! An example of a tool use. The assistant will use `python` to answer the
//! user's questions. Ported from `misanthropic`, driven through a local
//! [`SessionTransport`](drama_llama::SessionTransport) — same tool
//! definitions and manual dispatch loop, no API key.
//!
//! ```sh
//! cargo run --example python --features "tokio,cli"
//! ```

// Note: This example uses blocking calls for simplicity such as `println!()`
// and `stdin().lock()`. In a real application, these should *usually* be
// replaced with async alternatives.
use std::{
    io::{Read, Seek, Write},
    time::Duration,
    vec,
};

use clap::Parser;
use subprocess::{Exec, Redirection};

use drama_llama::SessionTransport;
use misanthropic::{
    json,
    markdown::ToMarkdown,
    prompt::{
        message::{Content, Role},
        Message,
    },
    tool::{self, CustomMethodDef},
    Prompt, Transport,
};

/// Use Python to answer the user's questions.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[command(flatten)]
    common: utils::CommonArgs,

    /// User prompt.
    #[arg(
        short,
        long,
        default_value = "Count the number of r's in 'strawberry'"
    )]
    prompt: String,
}

mod utils;

/// Returns true if the user wants to run the Python script.
fn prompt_user(script: &str) -> bool {
    // There is no sandboxing in this example for simplicity's sake so we ask
    // the user instead.
    println!(
        "Run the following Python script? y/n:\n\n```python\n{}\n\nDo not run this unless you fully understand the code!\n```",
        script
    );
    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    input.trim().eq_ignore_ascii_case("y")
}

/// Handle the tool call. Returns a [`User`] [`Message`] with the result.
///
/// [`User`]: Role::User
pub fn handle_tool_call(call: &tool::Use) -> Result<Message, Message> {
    if call.name != "python" {
        let content = format!("Unknown tool: {}", call.name);
        return Err(tool::Result::new(call.id.to_string(), content)
            .error()
            .into());
    }

    if let Some(script) = call.input["script"].as_str() {
        if !prompt_user(script) {
            // User declined to run the Python script. Inform the Assistant.

            return Err(tool::Result::new(
                call.id.to_string(),
                "User declined to run the Python script. Do you really need Python for this?",
            )
            .error()
            .into());
        }

        // Write the code to a temporary file.
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(script.as_bytes()).unwrap();

        // Ensure there is a newline at the end of the file.
        if !script.ends_with('\n') {
            file.seek(std::io::SeekFrom::End(0)).unwrap();
            file.write_all(b"\n").unwrap();
        }

        // Run the Python script.
        let mut p = Exec::cmd("python3")
            .arg(file.path())
            .stdout(Redirection::Pipe)
            .stderr(Redirection::Pipe)
            .popen()
            .unwrap();
        if let Ok(Some(status)) = p.wait_timeout(Duration::from_secs(5)) {
            // Read the output to a string.
            let mut output = String::new();
            if status.success() {
                // Send stdout to the Assistant.
                p.stdout
                    .as_ref()
                    .unwrap()
                    .read_to_string(&mut output)
                    .unwrap();

                Ok(tool::Result::new(call.id.to_string(), output).into())
            } else {
                // Send stderr to the Assistant (the exception).
                p.stderr
                    .as_ref()
                    .unwrap()
                    .read_to_string(&mut output)
                    .unwrap();

                Err(tool::Result::new(call.id.to_string(), output)
                    .error()
                    .into())
            }
        } else {
            // The Python script timed out.
            Ok(tool::Result::new(
                call.id.to_string(),
                "Python script timed out.",
            )
            .error()
            .into())
        }
    } else {
        // The Assistant did not use the `script` key. This should never happen.
        Err(tool::Result::new(call.id.to_string(), "Invalid input.")
            .error()
            .into())
    }
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Read the command line arguments.
    let args = Args::parse();
    utils::log_init(args.common.verbose);

    // Load the local model and wrap it as a `Transport` — the local
    // stand-in for `Client::new(key)`.
    let transport = SessionTransport::new(args.common.session()?);

    // Get the Python version so the Assistant can write code for the correct
    // version.
    let python_version: String = Exec::cmd("python3")
        .arg("--version")
        .stdout(Redirection::Pipe)
        .stderr(Redirection::Pipe)
        .capture()?
        .stdout_str()
        .trim()
        .to_string();

    // Craft our `Prompt`, providing a Tool definition to call `python`.
    // Note: We can't use the new async lifecycle methods here since we're not using a ToolBox,
    // but this shows the traditional approach still works.
    let mut chat = Prompt::default()
        .add_tool(
            CustomMethodDef::builder("python")
                .description("Run a Python script.")
                .schema(json!({
                    "type": "object",
                    "properties": {
                        "script": {
                            "type": "string",
                            "description": "Python script to run.",
                        },
                    },
                    "required": ["script"],
                }))
                .build()?,
        )
        // Inform the assistant about their limitations.
        .system(include_str!("python_system.md"))
        .add_system(format!("## Python Environment\n\n{}", python_version))
        // The example has some examples of the Assistant using Python and some
        // without to help guide the assistant to use Python when necessary and
        // not when it isn't. The more examples here, with more varied prompts,
        // the better the Assistant will be at this.
        .messages([
            Message {
                role: Role::User,
                content: "Write a haiku about Python.".into(),
            },
            Message {
                role: Role::Assistant,
                content: "Elegant syntax\rPowerful and versatile\nPython, my delight.".into(),
            },
            Message {
                role: Role::User,
                content: "Count the number of r's in 'strawberry'".into(),
            },
            Message {
                role: Role::Assistant,
                content: Content(vec![
                    r#"<thinking>I can't do that myself, but I can run a Python script to count the number of r's in "strawberry". The user did not specify case sensitivity so I will default to case insensitive.</thinking>"#.into(),
                    tool::Use::new(
                        "python",
                        json!({
                            "script": r#"print("strawberry".lower().count("r"))"#
                        }),
                    )
                    .with_id("calibration_000")
                    .into()
                ]),
            },
            tool::Result::new("calibration_000", "3").into(),
            (Role::Assistant, r#"The number of r's in "strawberry" is 3.""#).into(),
            (Role::User, "List the permutations of the first four letters of the alphabet.").into(),
            Message {
                role: Role::Assistant,
                content: Content(vec![
                    r#"<thinking>This request is complex enough to need Python. I should use the itertools module for this.</thinking>"#.into(),
                    tool::Use::new(
                        "python",
                        json!({
                            "script": r#"import itertools\nprint(','.join("".join(t) for t in itertools.permutations(('a', 'b', 'c', 'd'))))"#
                        }),
                    )
                    .with_id("calibration_001")
                    .into()
                ]),
            },
            tool::Result::new(
                "calibration_001",
                "abcd,abdc,acbd,acdb,adbc,adcb,bacd,badc,bcad,bcda,bdac,bdca,cabd,cadb,cbad,cbda,cdab,cdba,dabc,dacb,dbac,dbca,dcab,dcba",
            ).into(),
            (Role::Assistant, "The permutations of the first four letters of the alphabet are:\n\nabcd, abdc, acbd, acdb, adbc, adcb, bacd, badc, bcad, bcda, bdac, bdca, cabd, cadb, cbad, cbda, cdab, cdba, dabc, dacb, dbac, dbca, dcab, dcba.").into(),
            (Role::User, "What is the capital of France?").into(),
            (Role::Assistant, "Paris.").into(),
            (Role::User, "Thanks for all your help. I have to go now.").into(),
            (Role::Assistant, "You're welcome. Have a great day!<narrator>A new user enters the chat</narrator>").into(),
        ])?
        // Insert cache breakpoint. It won't do anything in this example, but if
        // the system prompt and examples are very long, it can be useful to
        // cache everything up to the user input.
        .cache()
        .add_message((Role::User, args.prompt))?;

    // Call the tool and retry up to 3 times.
    for retry in 0..3 {
        let message = transport.send(&chat).await?;

        if args.common.verbose {
            println!("Assistant reply:\n\n{}", message.markdown_verbose());
        }

        if let Some(call) = message.tool_use() {
            match handle_tool_call(call) {
                Ok(result) => {
                    // Tool use was successful
                    //
                    // If the agent retried, we pop the incorrect tool use. This
                    // way the assistant "got it right" the first time and the
                    // context isn't polluted incorrect tool use.
                    if retry > 0 {
                        chat.messages
                            .truncate(chat.messages.len() - (retry * 2));
                    }

                    let _ = chat.push_message(message);
                    let _ = chat.push_message(result);

                    // Generate a message with the result.
                    let message = transport.send(&chat).await?;
                    let _ = chat.push_message(message);
                    break;
                }
                Err(error) => {
                    // Something went wrong with the tool use. We'll append the
                    // error message so the Assistant can learn from it and try
                    // again.
                    let _ = chat.push_message(message);
                    let _ = chat.push_message(error);
                }
            }
        } else {
            // Tool was not called. This is fine if the user didn't ask for
            // something that requires Python.
            let _ = chat.push_message(message);
            break;
        }
    }

    println!(
        "{}",
        if args.common.verbose {
            chat.markdown_verbose().to_string()
        } else {
            chat.markdown().to_string()
        }
    );

    Ok(())
}
