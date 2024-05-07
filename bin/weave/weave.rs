use std::num::NonZeroU8;
use std::{io::Write, path::PathBuf};

use clap::Parser;
use drama_llama::data::StopWords;
use drama_llama::{data::banned, PredictOptions};
use drama_llama::{Engine, Model, Predicted, SampleOptions};

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
struct EnabledEvents {
    /// Whether to send tokens (more secure).
    tokens: bool,
    /// Whether to send pieces (less secure).
    ///
    /// NOTE: It is possible to reconstruct the text from the packet sizes when
    /// pieces are sent, especially if the model is known.
    pieces: bool,
    /// Whether to send paragraphs.
    paragraphs: bool,
}

impl Default for EnabledEvents {
    fn default() -> Self {
        Self {
            // FIXME: tokenization in the client is not yet implemented.
            tokens: false,
            pieces: true,
            paragraphs: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct ModelData {
    /// Filename within the model path.
    filename: String,
    /// Description of the model. If this is blank, it will be filled in with
    /// the model's metadata.
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
struct Config {
    /// Active story. By default, this is an empty story.
    #[serde(default)]
    story: Story,
    /// The model used to generate text.
    model: ModelData,
}

impl Config {
    fn load(path: &PathBuf) -> Result<Self, std::io::Error> {
        let toml = std::fs::read_to_string(path)?;
        toml::from_str(&toml).map_err(|e| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, e)
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, derive_more::From)]
#[serde(crate = "rocket::serde")]
/// A token or a piece.
enum TokenOrPiece {
    Token(llama_token),
    Piece(String),
}

/// A paragraph in a branching [`Story`].
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(crate = "rocket::serde")]
struct Paragraph {
    pieces: Vec<(u8, TokenOrPiece)>,
    children: Vec<Paragraph>,
}

impl Paragraph {
    fn extend<It, T>(&mut self, author_id: u8, t: It)
    where
        It: IntoIterator<Item = T>,
        T: Into<TokenOrPiece>,
    {
        self.pieces
            .extend(t.into_iter().map(|piece| (author_id, piece.into())));
    }

    fn push_child(&mut self, child: Paragraph) -> usize {
        self.children.push(child);
        self.children.len() - 1
    }

    /// Apply a function to all paragraphs in self and all children.
    fn for_all<F>(&mut self, mut f: F)
    where
        F: FnMut(&mut Paragraph),
    {
        f(self);

        for child in &mut self.children {
            child.for_all(&mut f);
        }
    }

    /// Apply a function to all paragraphs along a path. If an index is out of
    /// bounds, the function will terminate at that point.
    fn for_path<F>(&mut self, path: &[usize], mut f: F)
    where
        F: FnMut(&mut Paragraph),
    {
        let mut node = self;

        f(node);

        for &index in path {
            node = match node.children.get_mut(index) {
                Some(node) => node,
                None => return,
            };
            f(node);
        }
    }

    /// Render the paragraph to a writable.
    fn render<W>(&self, model: &Model, w: &mut W) -> std::fmt::Result
    where
        W: std::fmt::Write,
    {
        for (_, piece) in &self.pieces {
            match piece {
                TokenOrPiece::Token(token) => {
                    write!(w, "{}", model.token_to_piece(*token))?;
                }
                TokenOrPiece::Piece(piece) => {
                    write!(w, "{}", piece)?;
                }
            }
        }

        Ok(())
    }
}

/// A branching story. The tree can diverge but not converge (for now).
// TODO: use petgraph to allow for converging paths.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(crate = "rocket::serde")]
struct Story {
    title: String,
    authors: Vec<String>,
    root: Paragraph,
    #[serde(skip)]
    active: Vec<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[serde(crate = "rocket::serde")]
#[error("There are too many authors in this story (max 256).")]
struct AuthorsFull;

#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
#[serde(crate = "rocket::serde")]
#[error("Invalid author index: {index}.")]
struct InvalidAuthor {
    index: u8,
}

#[derive(
    Debug, Clone, Serialize, Deserialize, thiserror::Error, derive_more::From,
)]
#[serde(crate = "rocket::serde")]
#[error("Selected path does not exist: {path:?}.")]
struct InvalidPath {
    path: Vec<usize>,
}

impl Story {
    #[cfg(test)]
    fn new(title: String) -> Self {
        Self {
            title,
            authors: Vec::new(),
            root: Paragraph::default(),
            active: Vec::new(),
        }
    }

    /// Add an author to the story. Returns the author's index.
    fn add_author(&mut self, author: String) -> Result<u8, AuthorsFull> {
        if self.authors.len() >= 256 {
            return Err(AuthorsFull);
        }

        self.authors.push(author);
        Ok(self.authors.len() as u8 - 1)
    }

    /// Set active paragraph. An empty path selects the root.
    fn select(&mut self, path: &[usize]) -> Result<(), InvalidPath> {
        if self.get(path).is_none() {
            return Err(path.to_vec().into());
        }
        self.active.clear();
        self.active.extend_from_slice(path);
        Ok(())
    }

    /// Get the active paragraph.
    fn active(&self) -> &Paragraph {
        self.get(&self.active).unwrap()
    }

    /// Get the active paragraph mutably.
    fn active_mut(&mut self) -> &mut Paragraph {
        let active = self.active.clone();
        self.get_mut(&active).unwrap()
    }

    /// Extend the active paragraph with an iterable of pieces or tokens.
    fn extend<It, T>(
        &mut self,
        author_id: u8,
        t: It,
    ) -> Result<(), InvalidAuthor>
    where
        It: IntoIterator<Item = T>,
        T: Into<TokenOrPiece>,
    {
        if self.authors.get(author_id as usize).is_none() {
            return Err(InvalidAuthor { index: author_id });
        }

        self.active_mut().extend(author_id, t);

        Ok(())
    }

    /// Append a new child paragraph to the active paragraph. Set it as the
    /// active path.
    fn push_paragraph(&mut self, paragraph: Paragraph) {
        let index = self.active_mut().push_child(paragraph);
        self.active.push(index);
    }

    /// Add a new paragraph to the story. Set it as the active path.
    fn new_paragraph(&mut self) {
        self.push_paragraph(Paragraph::default());
    }

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

    /// Render the story along the active path.
    fn render(&mut self, model: &Model) -> String {
        let active = self.active.clone();
        self.render_path(&active, model)
    }

    /// Render all branches of the story to string.
    fn render_all(&mut self, model: &Model) -> String {
        let mut s = self.title.clone();

        self.root.for_all(|paragraph| {
            s.push_str("\n\n");
            paragraph.render(model, &mut s).unwrap();
        });

        s
    }

    /// Render a specific path of the story to string.
    fn render_path(&mut self, path: &[usize], model: &Model) -> String {
        let mut s = self.title.clone();

        self.root.for_path(path, |node| {
            s.push_str("\n\n");
            node.render(model, &mut s).unwrap();
        });

        s
    }
}

/// Client Event
#[derive(Debug, Clone, Serialize, Deserialize, derive_more::From)]
#[serde(crate = "rocket::serde")]
enum Response {
    /// Enabled events.
    EnabledEvents { events: EnabledEvents },
    /// Active path.
    Active { path: Vec<usize> },
    /// A story has been loaded or reloaded.
    Story { story: Story },
    /// A piece has been generated.
    Piece { piece: String },
    /// A token has been generated.
    Token { token: llama_token },
    /// A full paragraph has been generated.
    Paragraph { paragraph: Paragraph },
    /// Prediction options updated or requested.
    PredictOptions { opts: PredictOptions },
    /// An error occurred.
    Error { error: Error },
}

const fn nz_one() -> NonZeroU8 {
    match NonZeroU8::new(1) {
        Some(n) => n,
        None => panic!("NonZeroU8::new(1) failed"),
    }
}

/// Client Request
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(crate = "rocket::serde")]
enum Request {
    /// Get active path.
    GetActivePath,
    /// Get the story.
    GetStory,
    /// Get the prediction options.
    GetPredictOptions,
    /// Get the enabled events.
    GetEnabledEvents,
    /// Create a new paragraph at the active path.
    NewParagraph,
    /// Set enabled events.
    SetEnabledEvents { events: EnabledEvents },
    /// Set the prediction options.
    SetPredictOptions { opts: PredictOptions },
    /// Request the story to be loaded.
    LoadStory { story: Story },
    /// Generate one or more new paragraphs. Does not change the active path.
    Generate {
        /// Optional path to continue from. If this is `None`, the active path
        /// is used.
        #[serde(default)]
        path: Option<Vec<usize>>,
        /// Number of new generated paragraphs.
        #[serde(default = "nz_one")]
        n: NonZeroU8,
    },
    /// Append content to the active paragraph.
    Append { author: u8, text: String },
    /// Replace the paragraph at `path` with a new one. If `path` is `None`,
    /// the active path is used. Does not change the active path.
    Replace {
        paragraph: Paragraph,
        path: Option<Vec<usize>>,
    },
    /// Add a new child [`Paragraph`] to `path`. Sets it as the active path.
    /// If `path` is `None`, the active path is used.
    Add {
        paragraph: Paragraph,
        path: Option<Vec<usize>>,
    },
}

#[derive(
    Debug, Clone, Serialize, Deserialize, thiserror::Error, derive_more::From,
)]
#[serde(crate = "rocket::serde")]
enum RequestErrorReason {
    #[error("The server is busy. Handle by retrying later.")]
    Busy,
    #[error("The request contains prohibited content: `{content}`.")]
    Prohibited { content: String },
    #[error("{error}")]
    InvalidPath { error: InvalidPath },
    #[error("{error}")]
    AuthorsFull { error: AuthorsFull },
    #[error("{error}")]
    InvalidAuthor { error: InvalidAuthor },
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

fn validate_text(text: &str) -> Result<(), RequestErrorReason> {
    for regex_str in banned::ENGLISH_WORDS {
        let regex = regex::Regex::new(regex_str).unwrap();
        if regex.is_match(text) {
            return Err(RequestErrorReason::Prohibited {
                content: format!("banned regex match: {regex_str}"),
            });
        }
    }

    Ok(())
}

/// Input sanitization. Checking for banned words and other prohibited content.
fn validate_request(
    model: &Model,
    mut request: Request,
) -> Result<Request, Error> {
    match &mut request {
        Request::LoadStory { story } => {
            match validate_text(&story.render_all(model)) {
                Ok(_) => Ok(request),
                Err(reason) => Err(Error::Request { request, reason }),
            }
        }
        Request::Append { text, .. } => match validate_text(&text) {
            Ok(_) => Ok(request),
            Err(reason) => Err(Error::Request { request, reason }),
        },
        Request::Replace { paragraph, .. } | Request::Add { paragraph, .. } => {
            let mut text = String::new();
            paragraph.render(model, &mut text).unwrap();

            match validate_text(&text) {
                Ok(_) => Ok(request),
                Err(reason) => Err(Error::Request { request, reason }),
            }
        }
        Request::Generate { .. }
        | Request::GetActivePath
        | Request::NewParagraph
        | Request::GetEnabledEvents
        | Request::SetEnabledEvents { .. }
        | Request::GetPredictOptions
        | Request::SetPredictOptions { .. }
        | Request::GetStory => Ok(request),
    }
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
        let mut config = Config::load(&args.config).unwrap();
        let mut events = EnabledEvents::default();
        let mut predict_opts = PredictOptions::default();
        predict_opts.sample_options = SampleOptions::story_writing();

        // Model and language specific options. In the future, a helper
        // function will be added to the `PredictOptions` struct to handle this.
        let mut predict_opts = predict_opts.add_model_stops(&engine.model);
        predict_opts.sample_options.repetition =
            predict_opts.sample_options.repetition.take().map(|opts| {
                // We ignore stopwords for the purposes of repetition penalty
                // since we don't want to penalize the most common words. In
                // the future, we may apply a partial penalty to stopwords.
                opts.ignore_stopwords(StopWords::English, &engine.model)
            });

        let mut get_model_author_id = || {
            let model_desc = engine.model.desc();
            for (i, author) in config.story.authors.iter().enumerate() {
                if author == &model_desc {
                    return i as u8;
                }
            }
            match config.story.add_author(model_desc) {
                Ok(id) => id,
                Err(AuthorsFull) => {
                    // FIXME: handle this more gracefully.
                    panic!("Cannot add model author because: {AuthorsFull}")
                }
            }
        };
        // This is mutable because we may allow changing the model in the future.
        let model_author_id = get_model_author_id();

        while let Some(request) = from_client.blocking_recv() {
            let request = match validate_request(&engine.model, request) {
                Ok(request) => request,
                Err(error) => {
                    to_client.send(error.into()).ok();
                    continue;
                }
            };

            match request {
                Request::Generate { path, n } => {
                    if let Some(ref path) = path {
                        match config.story.select(path) {
                            Ok(_) => (),
                            Err(InvalidPath { path }) => {
                                to_client
                                    .send(Error::Request {
                                        request: Request::Generate { path: Some(path.clone()), n },
                                        reason:
                                            RequestErrorReason::InvalidPath {
                                                error: InvalidPath { path },
                                            },
                                    }.into())
                                    .ok();
                                continue;
                            }
                        }
                    }
                    let path = path.unwrap_or(config.story.active.clone());
                    let text = config.story.render(&engine.model);
                    let tokens = engine.model.tokenize(&text, false);

                    for _ in 0..n.get() {
                        let mut predictor = engine
                            .predict(tokens.clone(), predict_opts.clone());

                        while let Some(Predicted { piece, token }) =
                            predictor.next()
                        {
                            if events.pieces {
                                to_client.send(Response::Piece { piece }).ok();
                            }
                            if events.tokens {
                                to_client.send(Response::Token { token }).ok();
                            }
                        }

                        // because we might stop in the middle of a token, we need
                        // to tokenize and detokenize the text to get the correct
                        // pieces.
                        let text = predictor.into_text();
                        let tokens = engine.model.tokenize(&text, false);
                        let pieces = engine.model.tokens_to_pieces(tokens);

                        // Add the generated Paragraph to the story.
                        config.story.select(&path).unwrap();
                        config.story.new_paragraph();
                        config.story.extend(model_author_id, pieces).unwrap();
                        if events.paragraphs {
                            to_client
                                .send(Response::Paragraph {
                                    paragraph: config.story.active().clone(),
                                })
                                .ok();
                        }
                    }
                }
                Request::GetEnabledEvents => {
                    to_client.send(events.clone().into()).ok();
                }
                Request::SetEnabledEvents { events: new_events } => {
                    events = new_events;
                    to_client.send(events.clone().into()).ok();
                }
                Request::GetActivePath => {
                    to_client.send(config.story.active.clone().into()).ok();
                }
                Request::GetStory => {
                    to_client.send(config.story.clone().into()).ok();
                }
                Request::GetPredictOptions => {
                    to_client.send(predict_opts.clone().into()).ok();
                }
                Request::SetPredictOptions { opts } => {
                    predict_opts = opts;
                    to_client.send(predict_opts.clone().into()).ok();
                }
                Request::LoadStory { story } => {
                    config.story = story;
                    to_client.send(config.story.clone().into()).ok();
                }
                Request::Append { author, text } => {
                    let tokens = engine.model.tokenize(&text, false);
                    let pieces = engine.model.tokens_to_pieces(tokens);
                    config.story.extend(author, pieces).unwrap();
                }
                Request::Replace { paragraph, path } => {
                    let path =
                        path.unwrap_or_else(|| config.story.active.clone());
                    *config.story.get_mut(&path).unwrap() = paragraph;
                }
                Request::Add { paragraph, path } => {
                    let path =
                        path.unwrap_or_else(|| config.story.active.clone());
                    config.story.push_paragraph(paragraph);
                    config.story.select(&path).unwrap();
                }
                Request::NewParagraph => {
                    config.story.new_paragraph();
                    to_client.send(config.story.active().clone().into()).ok();
                }
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

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{Engine, Story};

    #[test]
    fn test_paragraph() {
        let mut story = Story::new("Hello, world!".to_string());
        let engine = Engine::from_path(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("models/model.gguf"),
        )
        .unwrap();

        const ALICE_MSG: &str = "Hello, Bob!";
        const BOB_MSG: &str = "Hello, Alice!";

        assert_eq!(story.add_author("Alice".to_string()).unwrap(), 0);
        assert_eq!(story.add_author("Bob".to_string()).unwrap(), 1);

        story
            .extend(0, engine.model.tokenize(ALICE_MSG, false))
            .unwrap();
        story.new_paragraph();
        story
            .extend(1, engine.model.tokenize(BOB_MSG, false))
            .unwrap();

        let rendered = story.render(&engine.model);
        assert_eq!(rendered, "Hello, world!\n\nHello, Bob!\n\nHello, Alice!");
    }
}
