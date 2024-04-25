use std::{io::Write, path::PathBuf};

use clap::Parser;
use drama_llama::prompt::Format;
use drama_llama::{
    data::StopWords, prompt, Engine, Message, NGram, PredictOptions, Prompt,
    RepetitionOptions, Role, SamplingMode,
};

use rocket::form::Form;
use rocket::fs::{relative, FileServer};
use rocket::get;
use rocket::response::stream::{Event, EventStream};
use rocket::tokio::select;
use rocket::tokio::sync::broadcast::error::RecvError;
use rocket::{
    post, routes,
    tokio::sync::{broadcast, mpsc},
    Shutdown, State,
};

/// Banned personas. Adding to this list is permitted, but removing names is a
/// violation of the terms of service. This list is not exhaustive, but it
/// serves as a starting point for personas that should not be allowed to be
/// discussed and ideas that should remain dead.
// TODO: move this to the library itself.
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

/// Command-line arguments for the server.
#[derive(Parser)]
struct DittomancerArgs {
    /// Path to prompt file in TOML format.
    #[clap(short, long, required = true)]
    prompt: PathBuf,
    /// Whether to ignore common english words for the repetition penalty.
    #[clap(long)]
    ignore_stopwords: bool,
    /// Common arguments for the drama llama.
    #[clap(flatten)]
    common: drama_llama::cli::Args,
}

/// Returns an infinite stream of server-sent events. Each event is a message
/// pulled from a broadcast queue sent by the `post` handler.
#[get("/events")]
async fn events(
    to_client: &State<broadcast::Sender<Message>>,
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
#[post("/message", data = "<form>")]
async fn post(form: Form<Message>, to_engine: &State<mpsc::Sender<Message>>) {
    let msg = form.into_inner();

    // A send 'fails' if there are no active subscribers. That's okay.
    let _res = to_engine.send(msg).await;
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

/// Launch the rocket server.
#[rocket::main]
async fn main() -> () {
    check_tos_acceptance();
    let args = DittomancerArgs::parse();

    // Our worker thread receives messages from the main thread and sends
    // generated pieces back to be broadcast to clients.
    let (to_engine, mut from_client) = mpsc::channel::<Message>(1024);
    let (to_client, _) = broadcast::channel::<Message>(1024);
    let to_client_clone = to_client.clone();
    let worker = rocket::tokio::task::spawn_blocking(move || {
        let mut engine = Engine::from_cli(args.common, None).unwrap();

        let prompt = Prompt::load(args.prompt).unwrap();

        // Check for banned personas.
        // FIXME: This is O(n^2) and should be optimized. It's not a big deal
        // but as the list grows it may slow down startup.
        for persona in BANNED_PERSONAS {
            // Check the agent's name.
            if prompt
                .agent
                .to_lowercase()
                .contains(&persona.to_lowercase())
            {
                // In the future we might want to send the client a message
                // instead of panicking.
                panic!("Banned simulacrum: {}", persona);
            }

            // Check the system prompt.
            if let Some(setting) = &prompt.setting {
                if setting.to_lowercase().contains(&persona.to_lowercase()) {
                    panic!("Setting cannot contain: {}", persona);
                }
            }
        }

        let prompt_format = prompt::Format::from_model(&engine.model)
            .unwrap_or(Format::Unknown);
        let mut prompt_string = String::new();
        prompt.format(prompt_format, &mut prompt_string).unwrap();
        dbg!(&prompt_string);

        let mut opts = PredictOptions::default()
            // We stop at any of the model's stop tokens.
            .add_model_stops(&engine.model)
            // As well as the username prompt for the human.
            .add_stop(prompt.human_prefix(prompt_format));

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
            [prompt.human.as_str(), prompt.agent.as_str()]
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
            let msg = match sanitize_input(&prompt, prompt_format, msg) {
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
            let text = format!("{}\n{}:", &msg.text, &prompt.agent);
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
                text.strip_suffix(&format!("\n{}:", &prompt.human))
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
