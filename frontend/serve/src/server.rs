//! The HTTP routes: health, models, completions and chat.

use crate::wire::{self, Failed, Head, Kind, Models, Usage};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event as Sse, KeepAlive, Sse as Stream};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use bundle::Message;
use engine::{Event, Reason, Request};
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{SystemTime, UNIX_EPOCH};
use tasks::{Generate, Spool};
use tokio::sync::mpsc as async_mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::ReceiverStream;

/// A request for the engine thread, and its event channel.
pub type Job = (Request, async_mpsc::Sender<Event>);

/// What every handler shares.
pub struct Shared {
    pub generate: Generate,
    /// The engine thread's queue. Answers share each step.
    pub jobs: Mutex<mpsc::Sender<Job>>,
    /// The name clients see, and when it loaded.
    pub model: String,
    pub created: u64,
    /// The last answer id handed out.
    pub answers: AtomicU64,
}

/// Seconds since the Unix epoch.
pub fn now() -> u64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |d| d.as_secs())
}

#[derive(Deserialize)]
pub struct Completion {
    prompt: String,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
}

#[derive(Deserialize)]
pub struct Chat {
    messages: Vec<Message>,
    #[serde(default = "default_max_tokens")]
    max_tokens: usize,
    #[serde(default)]
    stream: bool,
    /// Whether the model may think first. Left out: its default.
    #[serde(default = "yes")]
    think: bool,
}

fn yes() -> bool {
    true
}

fn default_max_tokens() -> usize {
    256
}

fn reason(r: Reason) -> &'static str {
    match r {
        Reason::Stop => "stop",
        Reason::Length | Reason::Context => "length",
    }
}

pub fn router(shared: Arc<Shared>) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/v1/completions", post(complete))
        .route("/v1/chat/completions", post(chat))
        .with_state(shared)
}

/// Why a request was refused, with its status.
type Refused = (StatusCode, String);

fn refuse((status, message): Refused) -> Response {
    (status, Json(Failed::new(status.as_u16(), message))).into_response()
}

/// Queues the prompt. Returns the event stream, or why not.
fn submit(
    shared: &Shared,
    prompt: &str,
    max_tokens: usize,
) -> Result<(async_mpsc::Receiver<Event>, usize), Refused> {
    let request = (shared.generate.request(prompt, max_tokens))
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    let prompt_tokens = request.tokens.len();
    let (events, receiver) = async_mpsc::channel(64);
    let gone = || {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            "the engine stopped".to_string(),
        )
    };
    let jobs = shared.jobs.lock().map_err(|_| gone())?;
    jobs.send((request, events)).map_err(|_| gone())?;
    Ok((receiver, prompt_tokens))
}

async fn models(State(shared): State<Arc<Shared>>) -> Json<Models> {
    Json(Models::one(shared.model.clone(), shared.created))
}

/// The fields this answer's bodies share.
fn head(shared: &Shared, kind: Kind) -> Head {
    let n = shared.answers.fetch_add(1, Ordering::Relaxed) + 1;
    let prefix = match kind {
        Kind::Text => "cmpl",
        Kind::Chat => "chatcmpl",
    };
    Head {
        kind,
        id: format!("{prefix}-{n}"),
        created: now(),
        model: shared.model.clone(),
    }
}

async fn complete(State(shared): State<Arc<Shared>>, Json(c): Json<Completion>) -> Response {
    match submit(&shared, &c.prompt, c.max_tokens) {
        Ok(queued) => {
            let head = head(&shared, Kind::Text);
            answer(shared, head, queued, c.stream).await
        }
        Err(refused) => refuse(refused),
    }
}

async fn chat(State(shared): State<Arc<Shared>>, Json(c): Json<Chat>) -> Response {
    let prompt = match shared.generate.chat(&c.messages, c.think) {
        Ok(prompt) => prompt,
        Err(e) => return refuse((StatusCode::BAD_REQUEST, e.to_string())),
    };
    match submit(&shared, &prompt, c.max_tokens) {
        Ok(queued) => {
            let head = head(&shared, Kind::Chat);
            answer(shared, head, queued, c.stream).await
        }
        Err(refused) => refuse(refused),
    }
}

/// One server-sent event with `body` as its data.
fn event(body: &wire::Body) -> Result<Sse, Infallible> {
    Ok(Sse::default().json_data(body).expect("a plain struct"))
}

/// The events as text: one body, or streamed chunks.
async fn answer(
    shared: Arc<Shared>,
    head: Head,
    (mut events, prompt_tokens): (async_mpsc::Receiver<Event>, usize),
    stream: bool,
) -> Response {
    let decode = move |ids: &[u32]| shared.generate.tokenizer.decode(ids);
    let mut spool = Spool::default();
    if stream {
        let first = match head.kind {
            Kind::Chat => Some(event(&head.role())),
            Kind::Text => None,
        };
        let chunks = ReceiverStream::new(events).map(move |e| {
            let body = match e {
                Event::Token(id) => head.chunk(spool.push(id, &decode), None),
                Event::Done(r) => head.chunk(String::new(), Some(reason(r))),
            };
            event(&body)
        });
        let done = Ok(Sse::default().data("[DONE]"));
        let all = tokio_stream::iter(first)
            .chain(chunks)
            .chain(tokio_stream::once(done));
        return Stream::new(all)
            .keep_alive(KeepAlive::default())
            .into_response();
    }
    let (mut text, mut tokens, mut finish) = (String::new(), 0, "length");
    while let Some(e) = events.recv().await {
        match e {
            Event::Token(id) => {
                text += &spool.push(id, &decode);
                tokens += 1;
            }
            Event::Done(r) => finish = reason(r),
        }
    }
    let usage = Usage::new(prompt_tokens, tokens);
    Json(head.whole(text, finish, usage)).into_response()
}
