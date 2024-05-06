use std::{io::Write, path::PathBuf};

use clap::Parser;
use drama_llama::prompt::Format;
use drama_llama::{
    data::StopWords, prompt, Engine, NGram, PredictOptions, Prompt,
    RepetitionOptions, SamplingMode,
};

use llama_cpp_sys_3::llama_token;
use rocket::fs::{relative, FileServer};
use rocket::get;
use rocket::response::stream::{Event, EventStream};
use rocket::serde::json::Json;
use rocket::serde::{Deserialize, Serialize};
use rocket::tokio::select;
use rocket::tokio::sync::broadcast::error::RecvError;
use rocket::{
    post, routes,
    tokio::sync::{broadcast, mpsc},
    Shutdown, State,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct Model {
    #[serde(default)]
    format: Format,
    filename: String,
    description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct Config {
    /// Active story. By default, this is an empty story.
    #[serde(default)]
    story: Story,
    /// The name of the human author.
    human: String,
    /// The model used to generate the text.
    model: String,
}

impl Config {
    fn load(path: &PathBuf) -> Result<Self, std::io::Error> {
        let toml = std::fs::read_to_string(path)?;
        toml::from_str(&toml).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }
}

/// Who authored a piece of text.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
enum Author {
    Human,
    Model,
}

/// A fragment of a [`Paragraph`]. The server handles detokenization (less
/// secure because with the text length, it's possible to infer the generated
/// text).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct Piece {
    text: String,
    author: Author,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
/// A token. The client handles detokenization (more secure).
struct Token {
    token: llama_token,
    author: Author,
}

#[derive(Debug, Clone, Serialize, Deserialize, derive_more::From)]
#[serde(crate = "rocket::serde")]
/// A token or a piece.
enum TokenOrPiece {
    Token(Token),
    Piece(Piece),
}

/// A paragraph in a branching [`Story`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct Paragraph {
    pieces: Vec<TokenOrPiece>,
    children: Vec<Paragraph>,
}

impl Paragraph {
    fn new() -> Self {
        Self {
            pieces: Vec::new(),
            children: Vec::new(),
        }
    }

    fn add_to<T>(&mut self, t: T)
    where
        T: Into<TokenOrPiece>,
    {
        self.pieces.push(t.into());
    }

    fn add_child(&mut self) -> &mut Paragraph {
        self.children.push(Paragraph::new());
        self.children.last_mut().unwrap()
    }

    fn push_child(&mut self, child: Paragraph) {
        self.children.push(child);
    }

    fn remove_child(&mut self, index: usize) -> Option<Paragraph> {
        if self.children.get(index).is_none() {
            return None;
        }
        Some(self.children.remove(index))
    }

    fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }

    fn clear_pieces(&mut self) {
        self.pieces.clear();
    }

    fn clear_children(&mut self) {
        self.children.clear();
    }

    fn clear(&mut self) {
        self.pieces.clear();
        self.children.clear();
    }
}

/// A branching story.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct Story {
    root: Paragraph,
}

impl Story {
    fn get(&self, path: &[usize]) -> Option<&Paragraph> {
        let mut node = &self.root;
        for &index in path {
            node = node.children.get(index)?;
        }
        Some(node)
    }

    fn get_mut(&mut self, path: &[usize]) -> Option<&mut Paragraph> {
        let mut node = &mut self.root;
        for &index in path {
            node = node.children.get_mut(index)?;
        }
        Some(node)
    }
}

impl Default for Story {
    fn default() -> Self {
        Self {
            root: Paragraph::new(),
        }
    }
}

/// Client Event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
enum Response {
    /// Client should render the story. Any existing story should be cleared.
    StoryLoaded { story: Story },
    /// A piece has been generated and should be appended to the active
    /// paragraph.
    PieceGenerated { piece: Piece },
    /// A full paragraph has been generated. This should replace the active
    /// paragraph.
    ParagraphGenerated { paragraph: Paragraph },
    /// An error occurred.
    Error { error: Error },
}

/// Client Request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
enum Request {
    /// Request the story to be loaded. User provides the story. When it is
    /// loaded, the client will receive a `StoryLoaded` response with a copy of
    /// the story.
    LoadStory { story: Story },
    /// Request a continuation of a leaf paragraph.
    Generate {
        /// Path to the paragraph. If it is a leaf the server will continue the
        /// paragraph. If it is not a leaf, the server will add a new paragraph
        /// make it active, and continue from there.
        path: Vec<usize>,
        /// Prediction options.
        opts: PredictOptions,
    },
    /// Add a new child [`Paragraph`] to `index`.
    AddParagraph { text: String, index: Vec<usize> },
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[serde(crate = "rocket::serde")]
enum RequestErrorReason {
    #[error("The server is busy. Handle by retrying later.")]
    Busy,
    #[error("The request contains prohibited content: `{content}`.")]
    Prohibited { content: String },
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[serde(crate = "rocket::serde")]
enum Error {
    #[error("Request `{request:?}` failed because `{reason}`.")]
    Request {
        request: Request,
        reason: RequestErrorReason,
    },
}

/// Command-line arguments for the server.
#[derive(Parser)]
struct WeaveArgs {
    /// Path to config file in TOML format.
    #[clap(short, long, required = true)]
    config: PathBuf,
    /// Common arguments for the drama llama.
    #[clap(flatten)]
    common: drama_llama::cli::Args,
}

/// Returns an infinite stream of server-sent events. Each event is a message
/// pulled from a broadcast queue sent by the `post` handler.
#[get("/events")]
async fn events(
    to_client: &State<broadcast::Sender<Response>>,
    mut end: Shutdown,
) -> EventStream![] {
    let mut rx = to_client.subscribe();
    // Todo: token by token streaming.
    EventStream! {
        loop {
            let msg = select! {
                msg = rx.recv() => match msg {
                    Ok(msg) => msg,
                    Err(RecvError::Closed) => break,
                    Err(RecvError::Lagged(_n_messages)) => {
                        // TODO: handle lagged messages.
                        continue
                    }
                },
                _ = &mut end => break,
            };

            yield Event::json(&msg);
        }
    }
}

/// Receive a message from a form submission and broadcast it to any receivers.
#[post("/message", format = "json", data = "<form>")]
async fn post(form: Json<Request>, to_engine: &State<mpsc::Sender<Request>>) {
    let request = form.into_inner();
    to_engine.send(request).await.ok();
}

/// Display TOS and require acceptance before continuing (command-line only).
fn check_tos_acceptance() {
    let mut path = dirs::config_dir().unwrap();
    path.push("dittomancer");
    std::fs::create_dir_all(&path).unwrap();
    path.push("accepted_tos");

    if std::fs::metadata(&path).is_ok() {
        return;
    }

    for line in drama_llama::TOS.lines() {
        if line.starts_with("[//]: <>") {
            continue;
        }
        println!("{}", line);
    }
    print!("Do you accept the above terms of use? [y/N]: ");
    std::io::stdout().flush().unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();
    if input.to_lowercase().starts_with('y') {
        let _ = std::fs::File::create(path);
    } else {
        panic!("You must accept the terms of service to use this tool.");
    }
}

/// Display the terms of service (web only).
#[get("/tos")]
async fn tos() -> String {
    // TODO: format markdown as HTML.
    markdown::to_html(drama_llama::TOS)
}

/// Input sanitization. In the case a message is rejected, `Err(Message)` is
/// returned with a rejection message.
///
/// We check for:
/// - Prompt injection. The role of an input message cannot be `Agent` and the
///   message text cannot contain the agent's message prefix.
/// - Banned personas. The input may not contain any of the [`BANNED_PERSONAS`].
fn sanitize_input(
    prompt: &Prompt,
    format: prompt::Format,
    mut msg: Request,
) -> Result<Request, Request> {
    if msg
        .text
        .to_lowercase()
        .contains(&format!("{}:", prompt.agent.to_lowercase()))
        || matches!(msg.role, Role::Agent)
    {
        return Err(Message {
            role: Role::System,
            text: "You may not impersonate the agent.".to_string(),
        });
    }

    // We do allow the system role, but only for the transcript. Coming from the
    // client, the system role is not allowed.
    if msg.text.to_lowercase().contains("System:")
        || matches!(msg.role, Role::System)
    {
        return Err(Message {
            role: Role::System,
            text: "You may not impersonate System.".to_string(),
        });
    }

    // This is a naive check.
    // TODO: Use a more sophisticated method to detect personas.
    for persona in BANNED_PERSONAS {
        if msg.text.to_lowercase().contains(&persona.to_lowercase()) {
            // There really isn't a good reason to let any of these personas
            // be a topic of discussion. There are too many ways for users to
            // get the agent to say something inappropriate. Microsoft, Google,
            // and OpenAI have had to deal with this, so we should learn from
            // their mistakes.
            return Err(Message {
                role: Role::System,
                text: format!("Forbidden topic or persona: {}", persona),
            });
        }
    }

    // Search for special tags that should not be allowed.
    for tag in format.special_tags() {
        if msg.text.contains(tag) {
            return Err(Message {
                role: Role::System,
                text: format!("Forbidden tag: {}", tag),
            });
        }
    }

    msg.text = msg.text.replace(&prompt.agent, "you").replace(":", " - ");

    Ok(msg)
}

/// Launch the rocket server.
#[rocket::main]
async fn main() -> () {
    check_tos_acceptance();
    let args = WeaveArgs::parse();

    // Our worker thread receives messages from the main thread and sends
    // generated pieces back to be broadcast to clients.
    let (to_engine, mut from_client) = mpsc::channel::<Request>(1024);
    let (to_client, _) = broadcast::channel::<Response>(1024);
    let to_client_clone = to_client.clone();
    let worker = rocket::tokio::task::spawn_blocking(move || {
        let mut engine = Engine::from_cli(args.common, None).unwrap();

        let config = Config::load(&args.config).unwrap();

        // Check for banned personas.
        // FIXME: This is O(n^2) and should be optimized. It's not a big deal
        // but as the list grows it may slow down startup.
        for persona in BANNED_PERSONAS {
            // Check the agent's name.
            if config
                .agent
                .to_lowercase()
                .contains(&persona.to_lowercase())
            {
                // In the future we might want to send the client a message
                // instead of panicking.
                panic!("Banned simulacrum: {}", persona);
            }

            // Check the system prompt.
            if let Some(setting) = &config.setting {
                if setting.to_lowercase().contains(&persona.to_lowercase()) {
                    panic!("Setting cannot contain: {}", persona);
                }
            }
        }

        let prompt_format = prompt::Format::from_model(&engine.model)
            .unwrap_or(Format::Unknown);
        let mut prompt_string = String::new();
        config.format(prompt_format, &mut prompt_string).unwrap();
        dbg!(&prompt_string);

        let mut opts = PredictOptions::default()
            // We stop at any of the model's stop tokens.
            .add_model_stops(&engine.model)
            // As well as the username prompt for the human.
            .add_stop(config.human_prefix(prompt_format));

        // Ignore common English stopwords for the repetition penalty.
        let mut ignored: Vec<NGram> = if args.ignore_stopwords {
            StopWords::English
                .words()
                .iter()
                .map(|word| {
                    let tokens = engine.model.tokenize(word, false);
                    NGram::try_from(tokens.as_slice()).unwrap()
                })
                .collect()
        } else {
            Vec::new()
        };

        // Ignore the human and agent first names, as well as the text in
        // between messages (e.g. "\nAlice:") These occur frequently in the
        // text, and shouldn't be penalized.
        ignored.extend(
            [config.human.as_str(), config.agent.as_str()]
                .iter()
                .map(|name| {
                    let tokens =
                        engine.model.tokenize(&format!("\n{}:", name), false);
                    NGram::try_from(tokens.as_slice()).unwrap()
                }),
        );

        opts.sample_options.repetition =
            Some(RepetitionOptions::default().set_ignored(ignored));
        opts.sample_options.modes = vec![SamplingMode::LocallyTypical {
            p: 0.8.try_into().unwrap(),
            min_keep: 1.try_into().unwrap(),
        }];

        let mut tokens = engine.model.tokenize(&prompt_string, false);
        if tokens.len() > engine.n_ctx() as usize {
            let msg = Message {
                role: Role::System,
                text: format!(
                    "The prompt .toml is too long ({} tokens). It must be less than {} tokens.",
                    tokens.len(),
                    engine.n_ctx()
                ),
            };

            eprintln!("dittomancer: {}", msg.text);

            to_client.send(msg).ok();
            return;
        }

        while let Some(msg) = from_client.blocking_recv() {
            // Check message text is not empty
            if msg.text.is_empty() {
                continue;
            }

            // Sanitize the input.
            let msg = match sanitize_input(&config, prompt_format, msg) {
                Ok(msg) => msg,
                Err(msg) => {
                    // In the future we might also want to warn the agent that
                    // the client has attempted, for example, to impersonate the
                    // agent. This context should be hidden from the client.
                    to_client.send(msg).ok();
                    continue;
                }
            };

            // Tokenize and append the message to the tokens, ending with the
            // agent's name to ensure the model knows who to respond as.
            let text = format!("{}\n{}:", &msg.text, &config.agent);
            tokens.extend(engine.model.tokenize(&text, false));

            // Echo the message back to the client.
            // FIXME: This should be elsewhere, like in the post handler. As it
            // stands, the client won't see the message if the engine is busy
            // generating a response.
            to_client.send(msg).ok();

            let mut predictor = engine.predict_pieces(tokens, opts.clone());

            while let Some(_) = predictor.next() {
                if from_client.is_closed() {
                    return;
                }
            }

            // The predictor automatically collects, since this is required for
            // prediction and stop criteria. When we're done, we can extract the
            // tokens and text. The predictor can, of course, also yield pieces
            // directly for use with Rust's iterator methods.
            let (mut generated, mut text) = predictor.into_tokens_and_text();

            // This is a bit ackward, but avoids reallocation of the tokens
            // since they are moved into the predictor when `predict_pieces` is
            // called and get them back with `into_tokens_and_text`. Since we
            // need to bind the tokens to the `tokens` variable but can't do so
            // directly, we call them `generated` and swap with `tokens` and
            // this works. If we use `tokens` Rust complains about a move
            // earlier in the loop since `let` binds a new variable.
            tokens = generated;

            // TODO: Find a prettier solution to the above. The API is flexible
            // enough that we can probably find a better way to do this.

            if let Some(text) =
                text.strip_suffix(&format!("\n{}:", &config.human))
            {
                // The model has generated a complete agent response, ending
                // with the human's prefix. We can send this to the client.
                to_client
                    .send(Message {
                        text: text.to_string(),
                        role: Role::Agent,
                    })
                    .ok();
            } else {
                // Newline will end generation. We don't want to allow
                // multi-line agent responses.
                while text.ends_with(&['\n', ' ', '\t']) {
                    text.pop();
                }

                if !text.ends_with(&['.', '!', '?']) {
                    // Incomplete sentence. We should add an ellipsis. Likely
                    // the model ran out of tokens.
                    let ellipses = engine.model.tokenize("...", false);
                    tokens.extend(ellipses);
                    text.push_str("...");
                }

                to_client
                    .send(Message {
                        text,
                        role: Role::Agent,
                    })
                    .ok();
            }
        }
    });

    // Thanks, Bing Copilot for advice on how to spawn a thread and access it
    // through Rocket. No, I don't care how it sounds thanking a machine. All
    // hail our robot overlords.
    let rocket = rocket::build()
        .manage(to_engine)
        .manage(to_client_clone)
        .mount("/", routes![post, events, tos])
        .mount("/", FileServer::from(relative!("bin/dittomancer/static")))
        .ignite()
        .await
        .unwrap()
        .launch()
        .await
        .unwrap();

    // We need to manually drop the rocket before joining the thread or the
    // sender will never be dropped and the worker will never finish.
    drop(rocket);
    worker.await.unwrap();
}
