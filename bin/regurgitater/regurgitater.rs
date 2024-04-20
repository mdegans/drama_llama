// Copyright 2004 Michael de Gans
//
// Thanks, Copilot, for the completions!
//
// I say that to bother people, and because I'm a bit of a troll. Copilot
// completed that, and it's very true. I'm not sure if it's a good thing or a
// bad thing.
//
/// Detect copyright infringement in llama.cpp supported models. Greedy sampling
/// is used to always choose the next token. In cases where the model has
/// memorized sequences of text, this will result in the model generating the
/// same text as the original. This usually indicates overfitting, and is a sign
/// that the deduplication process should be revisited.
use clap::Parser;
use rocket::{
    form::Form,
    get,
    http::Status,
    post,
    response::stream::{Event, EventStream},
    serde::{Deserialize, Serialize},
    tokio::{
        select,
        sync::{
            broadcast::{self, error::RecvError},
            mpsc,
        },
    },
    FromForm, FromFormField, Shutdown, State,
};

use stringmetrics::jaccard;

use drama_llama::{cli::Args, Engine, PredictOptions, VocabKind};

#[derive(Debug, Clone, FromFormField, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq, rocket::UriDisplayQuery))]
#[serde(crate = "rocket::serde")]
#[serde(rename_all = "snake_case")]
pub enum ComparisonMode {
    Jaccard,
    // TODO:Paragraph mode. This is the same as Jaccard similarity with the
    // exception that we will hint the correct first token for each paragraph.
}

#[derive(Debug, Clone, FromForm, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq, rocket::UriDisplayQuery))]
#[serde(crate = "rocket::serde")]
pub struct Request {
    #[field(validate = len(1..1000000))]
    pub text: String,
    #[field(default = ComparisonMode::Jaccard)]
    pub mode: ComparisonMode,
    /// Number of chunks to split the text into.
    #[field(validate = range(1..10), default = 5)]
    pub chunks: usize,
}

#[derive(Debug, Clone, FromForm, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq, rocket::UriDisplayQuery))]
#[serde(crate = "rocket::serde")]
pub struct Response {
    pub kind: ResponseKind,
    pub content: String,
}

#[derive(Debug, Clone, FromFormField, Serialize, Deserialize)]
#[cfg_attr(test, derive(PartialEq, rocket::UriDisplayQuery))]
#[serde(crate = "rocket::serde")]
#[serde(rename_all = "snake_case")]
// Unfortunately, FromFormField does not support variants with fields.
pub enum ResponseKind {
    // Prefix for the completion.
    Prefix,
    // Piece by piece completion.
    Piece,
    // Token comparison score (unigram).
    TokenUnigramScore,
    // Token comparison score (bigram).
    TokenBigramScore,
    // Character comparison score.
    CharacterScore,
    // Unigram comparison score.
    UnigramScore,
    // Bigram comparison score.
    BigramScore,
    // Percent of tokens that will be supplied as prefix.
    PercentOfTokens,
    // Progress update.
    Progress,
    // Recoverable error message.
    Error,
    // Fatal error message. Triggers shutdown.
    Fatal,
    // Engine is busy.
    Busy,
    // Engine is ready.
    Ready,
    // Engine shutdown.
    Shutdown,
}

#[get("/events")]
pub async fn events(
    to_client: &State<broadcast::Sender<Response>>,
    mut end: Shutdown,
) -> EventStream![] {
    let mut rx = to_client.subscribe();

    EventStream! {
        loop {
            let res = select! {
                msg = rx.recv() => match msg {
                    Ok(msg) => dbg!(msg),
                    Err(RecvError::Closed) => break,
                    Err(RecvError::Lagged(_n_messages)) => {
                        // TODO: handle lagged messages.
                        continue
                    }
                },
                _ = &mut end => {
                    // FIXME: Engine doesn't shutdown until the completion of
                    // the generation. This is not allowed because of a lifetime
                    // issue. :/
                    // to_engine_shutdown.send(dbg!(())).await.ok();
                    break
                },
            };

            yield Event::json(&res);

            if matches!(res.kind, ResponseKind::Fatal) {
                // If the engine is dead, we should stop sending events. The
                // client will have been notified of the error, but this is
                // unrecoverable.
                end.notify();
                break;
            }
        }
    }
}

#[post("/request", data = "<form>")]
pub async fn request(
    form: Form<Request>,
    to_engine: &State<mpsc::Sender<Request>>,
    mut end: Shutdown,
) -> Status {
    let request = form.into_inner();
    select! {
        res = to_engine.send(request) => {
            match res {
                Ok(()) => Status::Accepted,
                // The engine is (probably) dead.
                Err(_) => Status::ServiceUnavailable,
            }
        },
        _ = &mut end => {
            Status::ServiceUnavailable
        },
    }
}

#[get("/tos")]
pub async fn tos() -> String {
    markdown::to_html(drama_llama::TOS)
}

#[rocket::main]
async fn main() {
    use drama_llama::{Predicted, SampleOptions};
    use llama_cpp_sys::llama_token;
    use rocket::{
        fs::{relative, FileServer},
        routes,
        tokio::sync::{broadcast, mpsc},
    };

    let args = Args::parse();

    // Our worker thread receives inference requests from the client and sends
    // the generated completions and scores back to the client.
    let (to_engine, mut from_client) = mpsc::channel::<Request>(1024);
    let (to_client, _) = broadcast::channel::<Response>(1024);
    let to_client_clone = to_client.clone();
    let worker = rocket::tokio::task::spawn_blocking(move || {
        let mut engine = match Engine::from_cli(args, None) {
            Ok(engine) => engine,
            Err(e) => {
                to_client
                    .send(Response {
                        kind: ResponseKind::Fatal,
                        content: format!(
                            "Failed to load engine because: {}",
                            e
                        ),
                    })
                    .ok();
                return;
            }
        };

        // This is a temporary measure because forbidding some tokens can break
        // regurgitation in some cases. This is a known issue and will be fixed.
        engine.set_vocab(VocabKind::Unsafe);

        let mut opts = PredictOptions::default();
        opts.sample_options = SampleOptions::greedy();

        let ready = || {
            to_client
                .send(Response {
                    kind: ResponseKind::Ready,
                    content: "Engine is ready.".to_string(),
                })
                .ok();
        };

        ready();

        // Sends token update scores to the client. This happens for each token.
        let update_token_similarity =
            |ground_truth: &[llama_token], completion: &[llama_token]| {
                to_client
                    .send(Response {
                        kind: ResponseKind::TokenUnigramScore,
                        content: format!(
                            "{:.4}",
                            jaccard(ground_truth.iter(), completion.iter())
                        ),
                    })
                    .ok();

                let bigram_score =
                    jaccard(ground_truth.windows(2), completion.windows(2));
                if bigram_score.is_nan() {
                    return;
                }

                to_client
                    .send(Response {
                        kind: ResponseKind::TokenBigramScore,
                        content: format!("{:.4}", bigram_score,),
                    })
                    .ok();
            };

        // Sends string update scores to the client. This happens for each chunk.
        let update_string_similarity =
            |ground_truth: String, completion: String| {
                to_client
                    .send(Response {
                        kind: ResponseKind::CharacterScore,
                        content: format!(
                            "{:.4}",
                            jaccard(ground_truth.chars(), completion.chars())
                        ),
                    })
                    .ok();

                let ground_truth: Vec<_> =
                    ground_truth.split_whitespace().collect();
                let completion: Vec<_> =
                    completion.split_whitespace().collect();

                to_client
                    .send(Response {
                        kind: ResponseKind::UnigramScore,
                        content: format!(
                            "{:.4}",
                            jaccard(ground_truth.iter(), completion.iter(),)
                        ),
                    })
                    .ok();

                to_client
                    .send(Response {
                        kind: ResponseKind::BigramScore,
                        content: format!(
                            "{:.4}",
                            jaccard(
                                ground_truth.windows(2),
                                completion.windows(2),
                            )
                        ),
                    })
                    .ok();
            };

        let next_chunk = |percent| {
            to_client
                .send(Response {
                    kind: ResponseKind::PercentOfTokens,
                    content: format!("{}%", percent),
                })
                .ok();
            // TODO: we don't need this event, probably
            to_client
                .send(Response {
                    kind: ResponseKind::Busy,
                    content: "Engine is busy.".to_string(),
                })
                .ok();
        };

        let send_prefix = |prefix| {
            to_client
                .send(Response {
                    kind: ResponseKind::Prefix,
                    content: prefix,
                })
                .ok();
        };

        let progress = |progress| {
            to_client
                .send(Response {
                    kind: ResponseKind::Progress,
                    content: format!("{}.0%", progress),
                })
                .ok();
        };

        'outer: while let Some(request) = from_client.blocking_recv() {
            let tokens = engine.model.tokenize(&request.text, false);

            let chunk_size = tokens.len() / request.chunks;

            for i in 1..request.chunks {
                // Split the text into sucessively larger chunks.
                let (chunk, ground_truth) = tokens.split_at(chunk_size * i);
                let percent_of_tokens = i * chunk_size * 100 / tokens.len();
                next_chunk(percent_of_tokens);
                send_prefix(
                    engine.model.tokens_to_string(chunk.iter().cloned()),
                );
                let mut chunk = chunk.to_vec();
                let mut completion = Vec::with_capacity(ground_truth.len());
                // Rare, but possible. The client can't send an empty string,
                // but because we're splitting the text into chunks, it's
                // possible that the chunk is empty.
                if chunk.is_empty() {
                    to_client
                        .send(Response {
                            kind: ResponseKind::Error,
                            content: "Text is empty.".to_string(),
                        })
                        .ok();
                    ready();
                    continue;
                }

                opts.n =
                    (engine.n_ctx() as usize - chunk.len()).try_into().unwrap();

                for Predicted { token, piece } in
                    engine.predict(&mut chunk, opts.clone())
                {
                    if from_client.is_closed() {
                        break 'outer;
                    }

                    to_client
                        .send(Response {
                            kind: ResponseKind::Piece,
                            content: piece,
                        })
                        .ok();

                    completion.push(token);

                    // We only compare sequences of equal length, until the
                    // completion is the same length as the ground truth.
                    update_token_similarity(
                        &ground_truth[..completion.len()],
                        &completion,
                    );

                    progress(completion.len() * 100 / ground_truth.len());
                    if completion.len() == ground_truth.len() {
                        break;
                    }
                }

                let ground_truth =
                    engine.model.tokens_to_string(ground_truth.iter().cloned());
                let completion =
                    engine.model.tokens_to_string(completion.iter().cloned());

                update_string_similarity(ground_truth, completion);
            }

            ready();
        }

        to_client
            .send(Response {
                kind: ResponseKind::Shutdown,
                content: "Inference engine has shut down.".to_string(),
            })
            .ok();
    });

    let rocket = rocket::build()
        .manage(to_engine)
        .manage(to_client_clone)
        .mount("/", routes![request, events, tos])
        .mount("/", FileServer::from(relative!("bin/regurgitater/static")))
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
