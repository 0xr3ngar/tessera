//! OpenAI-shaped HTTP routes over the `tasks::api` traits.

mod answer;
#[cfg(test)]
mod answer_test;
#[cfg(test)]
mod lib_test;
#[cfg(test)]
mod mock;
mod wire;

use axum::Json;
use axum::Router;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use bundle::Message;
use serde_json::{Value, json};
use std::sync::{Arc, OnceLock};
use tasks::api::{Chat, Embed, Generator, Info, Tokens};
use wire::{Detokenize, Embeddings, Messages, Refusal, Tokenize, bad, refuse};

/// The loaded model, as the routes see it.
pub struct Model {
    pub info: Info,
    pub tokens: Arc<dyn Tokens>,
    pub chat: Option<Arc<dyn Chat>>,
    pub generator: Option<Arc<dyn Generator>>,
    pub embed: Option<Arc<dyn Embed>>,
    /// Unknown request fields refuse the request. Else they warn.
    pub strict: bool,
}

/// The model once it loads. Until then routes answer 503.
pub type Slot = Arc<OnceLock<Model>>;

/// Runs `f` on the model, off the async threads.
async fn with_model<T: Send + 'static>(
    slot: &Slot,
    f: impl FnOnce(&Model) -> Result<T, Refusal> + Send + 'static,
) -> Result<T, Refusal> {
    let slot = slot.clone();
    let loading = || {
        refuse(
            StatusCode::SERVICE_UNAVAILABLE,
            "the model is loading",
            None,
        )
    };
    let ran = tokio::task::spawn_blocking(move || f(slot.get().ok_or_else(loading)?));
    let failed =
        |e: tokio::task::JoinError| refuse(StatusCode::INTERNAL_SERVER_ERROR, e.to_string(), None);
    ran.await.map_err(failed)?
}

/// The messages as one prompt, by the model's template.
fn render(model: &Model, m: Messages) -> Result<String, Refusal> {
    let chat = model
        .chat
        .as_ref()
        .ok_or_else(|| bad("the model has no chat template", "messages"))?;
    let messages: Vec<Message> = m.messages.into_iter().map(Message::from).collect();
    let think = m.chat_template_kwargs.enable_thinking.unwrap_or(true);
    chat.render(&messages, think)
        .map_err(|e| bad(e.to_string(), "messages"))
}

/// A JSON answer, or the refusal.
async fn reply(
    slot: &Slot,
    f: impl FnOnce(&Model) -> Result<Value, Refusal> + Send + 'static,
) -> Response {
    match with_model(slot, f).await {
        Ok(v) => Json(v).into_response(),
        Err(refused) => refused.into_response(),
    }
}

/// Ready or loading, as llama.cpp reports it.
async fn health(State(slot): State<Slot>) -> Response {
    match slot.get() {
        Some(_) => Json(json!({"status": "ok"})).into_response(),
        None => {
            let error = json!({"code": 503, "message": "the model is loading", "type": "unavailable_error"});
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Json(json!({"error": error})),
            )
                .into_response()
        }
    }
}

async fn version() -> Json<Value> {
    Json(json!({"version": env!("CARGO_PKG_VERSION")}))
}

async fn models(State(slot): State<Slot>) -> Response {
    reply(&slot, |m| {
        let card = json!({
            "id": m.info.name,
            "object": "model",
            "created": wire::now(),
            "owned_by": "tessera",
            "root": m.info.name,
            "max_model_len": m.info.context,
        });
        Ok(json!({"object": "list", "data": [card]}))
    })
    .await
}

async fn tokenize(State(slot): State<Slot>, body: Bytes) -> Response {
    reply(&slot, move |m| {
        let text = match wire::parse(m, &body)? {
            Tokenize::Prompt { prompt, .. } => prompt,
            Tokenize::Chat { messages, .. } => render(m, messages)?,
        };
        let tokens = m
            .tokens
            .encode(&text)
            .map_err(|e| bad(e.to_string(), "prompt"))?;
        Ok(json!({"count": tokens.len(), "max_model_len": m.info.context, "tokens": tokens}))
    })
    .await
}

async fn detokenize(State(slot): State<Slot>, body: Bytes) -> Response {
    reply(&slot, move |m| {
        let d: Detokenize = wire::parse(m, &body)?;
        Ok(json!({"prompt": m.tokens.decode(&d.tokens)}))
    })
    .await
}

async fn embeddings(State(slot): State<Slot>, body: Bytes) -> Response {
    reply(&slot, move |m| {
        let embed = m
            .embed
            .as_ref()
            .ok_or_else(|| refuse(StatusCode::NOT_FOUND, "the model does not embed text", None))?;
        let e: Embeddings = wire::parse(m, &body)?;
        let texts = e.input.list();
        let got = embed
            .embed(&texts)
            .map_err(|e| bad(e.to_string(), "input"))?;
        let mut data = Vec::with_capacity(got.vectors.len());
        for (index, v) in got.vectors.iter().enumerate() {
            data.push(json!({"object": "embedding", "index": index, "embedding": v}));
        }
        let usage = json!({"prompt_tokens": got.tokens, "total_tokens": got.tokens});
        Ok(json!({"object": "list", "data": data, "model": m.info.name, "usage": usage}))
    })
    .await
}

pub fn router(slot: Slot) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/health", get(health))
        .route("/v1/health/ready", get(health))
        .route("/ping", get(health))
        .route("/version", get(version))
        .route("/v1/models", get(models))
        .route("/tokenize", post(tokenize))
        .route("/detokenize", post(detokenize))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/completions", post(answer::completions))
        .route("/v1/chat/completions", post(answer::chat_completions))
        .with_state(slot)
}
