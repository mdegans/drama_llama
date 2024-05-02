use std::{io::Write, path::PathBuf};

use clap::Parser;
use drama_llama::data::StopWords;
use drama_llama::server::{self, ServerShutdown};
use drama_llama::{Message, RepetitionOptions};

use rocket::form::Form;
use rocket::fs::{relative, FileServer};
use rocket::get;
use rocket::response::stream::{Event, EventStream};
use rocket::tokio::select;
use rocket::tokio::sync::broadcast::error::RecvError;
use rocket::tokio::task::JoinHandle;
use rocket::{
    post, routes,
    tokio::sync::{broadcast, mpsc},
    Shutdown, State,
};

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

/// Launch the rocket server.
#[rocket::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    check_tos_acceptance();
    let mut args = DittomancerArgs::parse();

    // TODO: this could be cleaner elsewhere. The reason we're separating the
    // model path and filename is because we allow swapping models and we don't
    // want to enable path traversal attacks.
    let mut model_path = args.common.model.clone().canonicalize()?;
    args.common.model = args
        .common
        .model
        .canonicalize()?
        .file_name()
        .expect("Model path must end with a file.")
        .into();
    assert!(model_path.pop());

    let mut server = server::Server::new(model_path);
    let mut config = server::Config::from(args.common);
    config.prompt =
        Some(toml::from_str(&std::fs::read_to_string(args.prompt)?)?);
    let mut worker = Some(server.start(config, None));
    let to_server = server.to_server();
    let to_client = server.subscribe();

    // Thanks, Bing Copilot for advice on how to spawn a thread and access it
    // through Rocket. No, I don't care how it sounds thanking a machine. All
    // hail our robot overlords.
    let rocket = rocket::build()
        .manage(to_server)
        .manage(to_client)
        .mount("/", routes![post, events, tos])
        .mount("/", FileServer::from(relative!("bin/dittomancer/static")))
        .ignite()
        .await?;

    let mut shutdown = rocket.shutdown();

    let rocket = rocket.launch();

    // FIXME: This is a bit of a mess. We should probably put this elsewhere.
    let server_task: JoinHandle<Result<(), String>> = rocket::tokio::spawn(
        async move {
            loop {
                select! {
                    ret = worker.unwrap() => match ret {
                        Ok(Ok(server_shutdown)) => {
                            match server_shutdown {
                                ServerShutdown::Reload {
                                    config,
                                    from_client,
                                } => {
                                    worker = Some(server.start(config, Some(from_client)));
                                }
                                ServerShutdown::Finished => {
                                    // We need to notify the Rocket server to
                                    // shut down.
                                    shutdown.notify();
                                    break;
                                }
                            }
                        },
                        Ok(Err(err)) => {
                            return Err(format!("Server worker failed: {:?}", err).into());
                        },
                        Err(join_error) => {
                            return Err(format!("Server worker failed: {:?}", join_error).into());
                        }
                    },
                    _ = &mut shutdown => {
                        server.shutdown().await.ok();
                        break;
                    }
                };
            }

            Ok(())
        },
    );

    select! {
        ret = server_task => {
            let _ = ret?;
        },
        ret = rocket => {
            let _ = ret?;
        },
    }

    Ok(())
}
