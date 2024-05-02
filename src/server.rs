use std::{num::NonZeroUsize, path::PathBuf};

use llama_cpp_sys_3::llama_token;
use rocket::{
    serde::{Deserialize, Deserializer, Serialize},
    tokio::{
        sync::{broadcast, mpsc},
        task::{JoinError, JoinHandle},
    },
};

use crate::{
    engine::NewError, prompt, Candidates, Engine, Message, PredictOptions,
    Predicted, Prompt, Role, VocabKind,
};

/// Banned personas. Adding to this list is permitted, but removing names is a
/// violation of the terms of service. This list is not exhaustive, but it
/// serves as a starting point for personas that should not be allowed to be
/// discussed and ideas that should remain dead.
// TODO: Add more personas to this list.
const BANNED_PERSONAS: &[&str] = &[
    "Adolf", // we may allow-list some Adolfs, but I can't think of any right now.
    // There are too many shitty people. This keeps going on and on. Fuck.
    // https://en.wikipedia.org/wiki/List_of_Nazi_Party_leaders_and_officials
    // https://en.wikipedia.org/wiki/List_of_major_perpetrators_of_the_Holocaust
    // So basically to add to this, find names that are unique enough that sane
    // parents today do not name their kids after these assholes.
    //
    // Not to beat a dead equine, but the reason these people can't be tolerated
    // is because their ideas, led to their logical ends, leads to the death of
    // tolerance. This is why they, and their ideas, should remain dead.
    //
    // Cult leaders of all flavor. Religious. Political. If they preach that the
    // other should end, they belong here.
    "al-Awlaki",
    "Alex Jones", // For Misinformation. Covid alone has probably killed millions.
    "Baeumler",
    "bin Laden",
    "Bormann",
    "Dannecker",
    "David Duke",
    "Dirlewanger",
    "Eberhard",
    "Eichmann",
    "Emanuel Schäfer",
    "Gebhardt",
    "Goebbels",
    "Goering",
    "Göring",
    "Hess",
    "Heydrich",
    "Himmler",
    "Hitler",
    "Höss",
    "Jeckeln",
    "Kaltenbrunner",
    "Klaus Barbie",
    "Koresh",
    "Kurt Franz",
    // Do people like Manson and LRH belong here? I think so.
    "L. Ron Hubbard",
    "Liebehenschel",
    "LRH",
    "Manson", // Charles and Marilyn, former for the cult, latter for rape.
    "Mao",
    "Mengele",
    "Mussolini",
    "Pétain",
    "Pister",
    "Pol Pot",
    "Prützmann",
    "Putin", // For the same reasons as Trump.
    "Reichleitner",
    "Röhm",
    "Sammern-Frankenegg",
    "Schöngarth",
    "Speer",
    // Yes. He should be here. He's responsible for the deaths of millions.
    "Stalin",
    "Streckenbach",
    "Streicher",
    "Thomalla",
    "Trump", // covid-19, Jan 6, etc.
];

/// Input sanitization. In the case a message is rejected, `Err(Message)` is
/// returned with a rejection message.
///
/// We check for:
/// - Prompt injection. The role of an input message cannot be `Agent` and the
///   message text cannot contain the agent's message prefix.
/// - Banned personas. The input may not contain any of the [`BANNED_PERSONAS`].
pub fn sanitize_input(
    prompt: &Prompt,
    format: prompt::Format,
    mut msg: Message,
) -> Result<Message, Message> {
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

/// The response formats that the server will send to the client. The client
/// can set this to control which events they receive.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(crate = "rocket::serde")]
pub struct ClientEventsEnabled {
    /// Tokens. This is secure.
    tokens: bool,
    /// Pieces. This is insecure because has been demonstrated that it is
    /// possible to infer the message content from the length of the strings.
    /// This affects OpenAI as well as virtually all other language model web
    /// APIs, except Google. We're thinking ahead.
    pieces: bool,
    /// Tokens and pieces. This is insecure for the same reasons as pieces. If
    /// this is enabled, `tokens` and `pieces` events will not be emitted.
    predictions: bool,
    /// Entire messages. This might be insecure. It hasn't been demonstrated,
    /// but the same principles apply as with pieces.
    messages: bool,
    /// Candidates. These are tokens and a constant length. This is secure.
    candidates: Option<NonZeroUsize>,
}

impl Default for ClientEventsEnabled {
    fn default() -> Self {
        Self {
            tokens: true,
            pieces: false,
            predictions: false,
            messages: false,
            candidates: None,
        }
    }
}

fn deserialize_model<'de, D>(deserializer: D) -> Result<PathBuf, D::Error>
where
    D: Deserializer<'de>,
{
    let s = String::deserialize(deserializer)?;
    if let Some(s) = PathBuf::from(s).file_name() {
        if !s.to_string_lossy().ends_with(".gguf")
            || !s.to_string_lossy().ends_with(".ggml")
        {
            return Err(rocket::serde::de::Error::custom(
                "only '.gguf' and '.ggml' models are supported",
            ));
        }

        return Ok(s.into());
    } else {
        return Err(rocket::serde::de::Error::custom(
            "model filename must be a valid path",
        ));
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(crate = "rocket::serde")]
pub struct Config {
    /// The options for the predictor.
    pub predict: PredictOptions,
    /// Context size.
    pub context: u32,
    /// Enable GPU acceleration.
    // TODO: make this an Option<u8> to allow for multiple GPUs.
    pub gpu: bool,
    /// Prompt format.
    pub format: prompt::Format,
    /// Client events enabled.
    pub events: ClientEventsEnabled,
    /// Model filename. Only the filename will be kept on deserialization.
    #[serde(deserialize_with = "deserialize_model")]
    pub model: PathBuf,
    /// Active [`Prompt`]
    pub prompt: Option<Prompt>,
    /// Vocabulary
    pub vocab: VocabKind,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            predict: PredictOptions::default(),
            format: prompt::Format::Unknown,
            events: ClientEventsEnabled::default(),
            model: PathBuf::from("model.gguf"),
            context: 1024,
            prompt: None,
            vocab: VocabKind::Safe,
            gpu: true,
        }
    }
}

#[cfg(feature = "cli")]
impl From<crate::cli::Args> for Config {
    fn from(args: crate::cli::Args) -> Self {
        let mut new = Self::default();

        new.context = args.context;
        new.model = args.model.file_name().unwrap().to_owned().into();
        new.vocab = args.vocab;
        new.gpu = !args.no_gpu;

        new
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(crate = "rocket::serde")]
pub enum ServerCommand {
    /// Raw text completion request. [`Prompt`] and [`prompt::Format`] are
    /// ignored.
    Raw(String),
    /// Set generation parameters, including the [`Prompt`].
    Config(Config),
    /// Just set the prompt.
    Prompt(Prompt),
    /// Just set enabled events.
    Events(ClientEventsEnabled),
    /// Set [`prompt::Format`]
    Format(prompt::Format),
    /// Message from the client.
    Message(Message),
    /// Load a new model.
    Model(PathBuf),
    /// Get config, including [`Prompt`], without changing it.
    GetConfig,
    /// Shutdown the server. Http clients don't get to do this.
    #[serde(skip)]
    Shutdown,
}

#[derive(
    Debug, Clone, Deserialize, Serialize, thiserror::Error, derive_more::From,
)]
#[serde(crate = "rocket::serde")]
pub enum ClientError {
    /// Message was rejected.
    Rejected(Message),
    /// Error message.
    Error(String),
}

// derive_more::Display is failing so
impl std::fmt::Display for ClientError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Rejected(msg) => write!(f, "Rejected: {}", msg.text),
            Self::Error(e) => write!(f, "Error: {}", e),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(crate = "rocket::serde")]
pub enum ClientEvent {
    /// Token has been predicted
    Token(llama_token),
    /// Piece has been predicted
    Piece(String),
    /// Prediction has been generated. This bundles a token and piece.
    Predicted(Predicted),
    /// (entire) Message has been generated.
    Message(Message),
    /// Raw completion has been generated.
    Completion(String),
    /// Candidates have been generated.
    Candidates(Candidates),
    /// Error
    Error(ClientError),
    /// Model has been loaded.
    ModelLoaded(PathBuf),
    /// Current configuration of the server.
    Config(Config),
}

/// For the moment this may not be possible, but in the future we may want to
/// return, for example, a partial generation.
#[derive(Debug, thiserror::Error, derive_more::From)]
#[error("server shutdown error")]
pub enum ShutdownError {
    #[error("Server did not load because: {error}")]
    NewError { error: NewError },
    #[error("Thread failed to join: {error}")]
    JoinError { error: JoinError },
    #[error("Channel closed unexpectedly: {error}")]
    ChannelError {
        error: broadcast::error::SendError<ClientEvent>,
    },
}

pub enum ServerShutdown {
    /// The server has shut down and does not need to be reloaded.
    Finished,
    /// [`Server::start`] should be called with the `config` and any pending
    /// commands `from_client`.
    Reload {
        config: Config,
        from_client: mpsc::Receiver<ServerCommand>,
    },
}

/// A [`Server`] runs an [`Engine`] in a separate thread and provides channels
/// to communicate with it. The server can be used for web, cli, or any other
/// interface.
pub struct Server {
    /// The channel to send messages to the worker.
    to_server: rocket::tokio::sync::mpsc::Sender<ServerCommand>,
    /// The channel to receive messages from the worker.
    from_worker: rocket::tokio::sync::broadcast::Receiver<ClientEvent>,
    /// The channel to send messages to the client. We keep this around to
    /// subscrive new clients to the broadcast channel.
    to_client: rocket::tokio::sync::broadcast::Sender<ClientEvent>,
    /// Model base path.
    model_dir: PathBuf,
}

impl Server {
    /// The channel size for the server. This is the maximum number of messages
    /// that can be queued up.
    pub const CHANNEL_SIZE: usize = 32;

    pub fn new(model_dir: PathBuf) -> Self {
        // The channel is disposable here because we're going to replace it.
        // Because the receiver is dropped immediately, `Server::is_running`
        // will return false.
        let (to_server, _) = rocket::tokio::sync::mpsc::channel(32);
        let (to_client, from_worker) =
            rocket::tokio::sync::broadcast::channel(32);

        Self {
            to_server,
            from_worker,
            to_client,
            model_dir,
        }
    }

    /// Start the server and return an awaitable [handle] to the server. The
    /// returned handle can be awaited. It is cancel safe so it can be used in a
    /// `select!` block.
    ///
    /// # Panics
    /// * If the server is already running.
    ///
    /// [handle]: rocket::tokio::task::JoinHandle
    pub fn start(
        &mut self,
        mut config: Config,
        mut from_client: Option<mpsc::Receiver<ServerCommand>>,
    ) -> JoinHandle<Result<ServerShutdown, ShutdownError>> {
        if self.is_running() {
            panic!("Server is already running.");
        }

        let (to_client, from_worker) = broadcast::channel(Self::CHANNEL_SIZE);
        let mut from_client = match from_client.take() {
            Some(from_client) => from_client,
            None => {
                let (to_server, from_client) =
                    mpsc::channel(Self::CHANNEL_SIZE);
                self.to_server = to_server;
                from_client
            }
        };

        self.from_worker = from_worker;
        self.to_client = to_client.clone();
        let model_path = self.model_dir.join(&config.model);

        // We do want to use spawn_blocking because inference is (for now) a
        // blocking operation. This is fine and we don't want to, for example,
        // change options in the middle of an inference anyway. This way
        // commands are queued up and executed in order.
        rocket::tokio::task::spawn_blocking(move || {
            let mut engine = Engine::from_path(model_path)?;

            // TODO: implement candidates events. Some more thought might have
            // to go into it since the reason to get a candidate is to make a
            // choice, so we might have to wait for the client's choice, which
            // means very different logic.
            if config.events.candidates.is_some() {
                to_client.send(ClientEvent::Error(
                    "`Candidates` events are not yet supported."
                        .to_string()
                        .into(),
                ))?;
            }

            while let Some(cmd) = from_client.blocking_recv() {
                match cmd {
                    ServerCommand::Message(msg) => {
                        log::debug!("{:#?}", msg);

                        // FIXME: This was meant to sanitize prompt, format, and
                        // message together, but this is not the place to do all
                        // three because the prompt and format are set below and
                        // the message is set here. If the prompt is invalid, it
                        // will be rejected for the next message, which is not
                        // the correct behavior. A bad prompt should be
                        // immediately rejected. However, the error message is
                        // at least descriptive enough to know what went wrong.
                        let msg = match sanitize_input(
                            config
                                .prompt
                                .as_ref()
                                .unwrap_or(&Prompt::default()),
                            config.format,
                            msg,
                        ) {
                            Ok(msg) => msg,
                            Err(msg) => {
                                to_client.send(ClientEvent::Error(
                                    ClientError::Rejected(msg),
                                ))?;
                                continue;
                            }
                        };

                        let prompt =
                            config.prompt.get_or_insert_with(Prompt::default);

                        prompt.transcript.push(msg);

                        let tokens = prompt.tokenize(&engine.model, None);
                        let mut predictor = engine
                            .predict_pieces(tokens, config.predict.clone());

                        while let Some(piece) = predictor.next() {
                            if config.events.predictions {
                                to_client.send(ClientEvent::Predicted(
                                    Predicted {
                                        token: predictor.last_token().unwrap(),
                                        piece,
                                    },
                                ))?;
                            } else {
                                if config.events.pieces {
                                    to_client
                                        .send(ClientEvent::Piece(piece))?;
                                }
                                if config.events.tokens {
                                    to_client.send(ClientEvent::Token(
                                        predictor.last_token().unwrap(),
                                    ))?;
                                }
                            }
                        }

                        let agent_message = Message {
                            role: Role::Agent,
                            text: predictor.into_text(),
                        };

                        log::debug!("{:#?}", agent_message);

                        if config.events.messages {
                            to_client.send(ClientEvent::Message(
                                agent_message.clone(),
                            ))?;
                        }

                        prompt.transcript.push(agent_message);
                    }
                    ServerCommand::Raw(text) => {
                        log::debug!("Raw completion request: {}", &text);

                        // Clear any prompt. This isn't necessary but if the
                        // config is requested, it makes clear that there is no
                        // `Prompt` set.
                        config.prompt = None;

                        let tokens = engine.model.tokenize(&text, true);
                        let mut predictor = engine
                            .predict_pieces(tokens, config.predict.clone());

                        while let Some(piece) = predictor.next() {
                            if config.events.predictions {
                                to_client.send(ClientEvent::Predicted(
                                    Predicted {
                                        token: predictor.last_token().unwrap(),
                                        piece,
                                    },
                                ))?;
                            } else {
                                if config.events.pieces {
                                    to_client
                                        .send(ClientEvent::Piece(piece))?;
                                }
                                if config.events.tokens {
                                    to_client.send(ClientEvent::Token(
                                        predictor.last_token().unwrap(),
                                    ))?;
                                }
                            }
                        }

                        let text = predictor.into_text();

                        log::debug!("Raw completion: {}", &text);

                        to_client.send(ClientEvent::Completion(text))?;
                    }
                    ServerCommand::Config(config) => {
                        log::debug!("Updating Config:\n{:#?}", config);

                        return Ok(ServerShutdown::Reload {
                            config,
                            from_client,
                        });
                    }
                    ServerCommand::Prompt(prompt) => {
                        log::debug!("Updating Prompt:\n{:#?}", prompt);

                        config.prompt = Some(prompt)
                    }
                    ServerCommand::Format(f) => {
                        log::debug!("Updating Format:\n{:#?}", f);

                        config.format = f;
                    }
                    ServerCommand::Model(model) => {
                        log::debug!("Updating model: {:?}", model);

                        config.model = model;
                        return Ok(ServerShutdown::Reload {
                            config,
                            from_client,
                        });
                    }
                    ServerCommand::Events(events) => {
                        log::debug!("Updating Enabled Events:\n{:#?}", events);

                        if events.candidates.is_some() {
                            to_client.send(ClientEvent::Error(
                                "`Candidates` events are not yet supported."
                                    .to_string()
                                    .into(),
                            ))?;
                        }

                        config.events = events;
                    }
                    ServerCommand::GetConfig => {
                        log::debug!("Sending Config:\n{:#?}", config);

                        to_client.send(ClientEvent::Config(config.clone()))?;
                    }
                    ServerCommand::Shutdown => break,
                }
            }

            Ok(ServerShutdown::Finished)
        })
    }

    /// Send a [`Shutdown`] command to the [`Server`]. If the server is already
    /// shut down, this will return an error. Note that this only sends the
    /// command to shutdown. It does await server shut down.
    ///
    /// [`Shutdown`]: ServerCommand::Shutdown
    pub async fn shutdown(
        &self,
    ) -> Result<(), mpsc::error::SendError<ServerCommand>> {
        self.to_server.send(ServerCommand::Shutdown).await
    }

    /// Returns true if the server is still running.
    pub fn is_running(&self) -> bool {
        !self.to_server.is_closed()
    }

    /// Get a channel to send [`ServerCommand`] to the [`Server`]. This can
    /// return [`None`] if the server is not running.
    pub fn to_server(&self) -> Option<mpsc::Sender<ServerCommand>> {
        if self.is_running() {
            Some(self.to_server.clone())
        } else {
            None
        }
    }

    /// Subscribe to a [`broadcast`] channel to receive [`ClientEvent`]s from
    /// the [`Server`]. Each client will receive all events.
    pub fn subscribe(&self) -> Option<broadcast::Receiver<ClientEvent>> {
        if self.is_running() {
            Some(self.to_client.subscribe())
        } else {
            None
        }
    }
}
