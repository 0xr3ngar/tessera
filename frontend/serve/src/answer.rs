//! Completions and chat completions, whole or streamed.

use crate::wire::{self, Chat, Common, Completion, Prompt, Refusal, bad, refuse};
use crate::{Model, Slot, with_model};
use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use std::convert::Infallible;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tasks::api::{Finish, Params, Step};
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::UnboundedReceiverStream;

/// Which OpenAI shape an answer takes.
#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Text,
    Chat,
}

/// An answer in flight, and how to report it.
struct Started {
    kind: Kind,
    id: String,
    model: String,
    prompt_tokens: usize,
    stream: bool,
    usage: bool,
    steps: mpsc::UnboundedReceiver<Step>,
}

fn finish(f: Finish) -> &'static str {
    match f {
        Finish::Stop => "stop",
        Finish::Length => "length",
    }
}

/// An id no other answer from this process shares.
fn fresh_id(kind: Kind) -> String {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let prefix = match kind {
        Kind::Text => "cmpl",
        Kind::Chat => "chatcmpl",
    };
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    format!("{prefix}-{:x}{n:x}", wire::now())
}

/// The tokens asked for, or the room the context leaves.
fn budget(model: &Model, prompt: usize, asked: Option<usize>) -> Result<usize, Refusal> {
    let context = model.info.context;
    let room = context.saturating_sub(prompt);
    match asked {
        Some(0) => Err(bad("max_tokens must be at least 1", "max_tokens")),
        Some(n) if n <= room => Ok(n),
        None if room > 0 => Ok(room),
        _ => {
            let n = asked.unwrap_or(0);
            let message = format!(
                "This model's maximum context length is {context} tokens. However, you \
                 requested {} tokens ({prompt} in the messages, {n} in the completion). \
                 Please reduce the length of the messages or completion.",
                prompt + n
            );
            Err(bad(message, "max_tokens"))
        }
    }
}

/// Starts the answer to `prompt` on the model's generator.
fn start(
    model: &Model,
    kind: Kind,
    prompt: Vec<u32>,
    asked: Option<usize>,
    common: Common,
) -> Result<Started, Refusal> {
    let generator = model.generator.as_ref().ok_or_else(|| {
        refuse(
            StatusCode::NOT_FOUND,
            "the model does not generate text",
            None,
        )
    })?;
    if prompt.is_empty() {
        return Err(bad("the prompt is empty", "prompt"));
    }
    let params = Params {
        max_tokens: budget(model, prompt.len(), asked)?,
        stop: common.stop.map(wire::Strings::list).unwrap_or_default(),
    };
    let (sink, steps) = mpsc::unbounded_channel();
    let prompt_tokens = prompt.len();
    generator
        .generate(
            prompt,
            params,
            Box::new(move |step| sink.send(step).is_ok()),
        )
        .map_err(|e| refuse(StatusCode::SERVICE_UNAVAILABLE, e.to_string(), None))?;
    Ok(Started {
        kind,
        id: fresh_id(kind),
        model: model.info.name.clone(),
        prompt_tokens,
        stream: common.stream,
        usage: common.stream_options.is_some_and(|o| o.include_usage),
        steps,
    })
}

fn complete(model: &Model, body: &Bytes) -> Result<Started, Refusal> {
    let c: Completion = wire::parse(model, body)?;
    let prompt = match c.prompt {
        Prompt::Ids(ids) => ids,
        Prompt::Text(texts) => match texts.list().as_slice() {
            [text] => model
                .tokens
                .encode(text)
                .map_err(|e| bad(e.to_string(), "prompt"))?,
            _ => return Err(bad("give one prompt per request", "prompt")),
        },
    };
    start(model, Kind::Text, prompt, c.common.max_tokens, c.common)
}

fn chat(model: &Model, body: &Bytes) -> Result<Started, Refusal> {
    let c: Chat = wire::parse(model, body)?;
    let prompt = crate::render(model, c.messages)?;
    let prompt = model
        .tokens
        .encode(&prompt)
        .map_err(|e| bad(e.to_string(), "messages"))?;
    let asked = c.max_completion_tokens.or(c.common.max_tokens);
    start(model, Kind::Chat, prompt, asked, c.common)
}

impl Started {
    fn object(&self) -> &'static str {
        match (self.kind, self.stream) {
            (Kind::Text, _) => "text_completion",
            (Kind::Chat, false) => "chat.completion",
            (Kind::Chat, true) => "chat.completion.chunk",
        }
    }

    /// One choice: whole text, or a piece of a stream.
    fn choice(&self, text: &str, finish: Option<Finish>) -> Value {
        let reason = finish.map(self::finish);
        match (self.kind, self.stream) {
            (Kind::Text, _) => {
                json!({"index": 0, "text": text, "logprobs": null, "finish_reason": reason})
            }
            (Kind::Chat, false) => json!({
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": null,
                "finish_reason": reason,
            }),
            (Kind::Chat, true) => {
                let delta = match text.is_empty() {
                    true => json!({}),
                    false => json!({"content": text}),
                };
                json!({"index": 0, "delta": delta, "logprobs": null, "finish_reason": reason})
            }
        }
    }

    fn body(&self, choices: Value, usage: Option<usize>) -> Value {
        let mut body = json!({
            "id": self.id,
            "object": self.object(),
            "created": wire::now(),
            "model": self.model,
            "choices": choices,
        });
        if let Some(tokens) = usage {
            body["usage"] = json!({
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": tokens,
                "total_tokens": self.prompt_tokens + tokens,
            });
        }
        body
    }

    /// Waits for the end and answers in one body.
    async fn whole(mut self) -> Response {
        let mut text = String::new();
        while let Some(step) = self.steps.recv().await {
            match step {
                Step::Token { text: piece, .. } => text += &piece,
                Step::Done {
                    text: rest,
                    finish,
                    tokens,
                } => {
                    text += &rest;
                    let choice = self.choice(&text, Some(finish));
                    return Json(self.body(json!([choice]), Some(tokens))).into_response();
                }
            }
        }
        let gone = refuse(StatusCode::SERVICE_UNAVAILABLE, "the engine stopped", None);
        gone.into_response()
    }

    /// Streams each step as a chunk, then `[DONE]`.
    fn streamed(mut self) -> Response {
        let (out, events) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let send = |v: Value| out.send(Event::default().data(v.to_string())).is_ok();
            if self.kind == Kind::Chat {
                let delta = json!([{"index": 0, "delta": {"role": "assistant", "content": ""},
                    "logprobs": null, "finish_reason": null}]);
                if !send(self.body(delta, None)) {
                    return;
                }
            }
            while let Some(step) = self.steps.recv().await {
                let (text, end) = match step {
                    Step::Token { text, .. } if text.is_empty() => continue,
                    Step::Token { text, .. } => (text, None),
                    Step::Done {
                        text,
                        finish,
                        tokens,
                    } => (text, Some((finish, tokens))),
                };
                let choice = self.choice(&text, end.map(|(f, _)| f));
                if !send(self.body(json!([choice]), None)) {
                    return;
                }
                if let Some((_, tokens)) = end {
                    if self.usage {
                        send(self.body(json!([]), Some(tokens)));
                    }
                    let _ = out.send(Event::default().data("[DONE]"));
                    return;
                }
            }
            // The steps stopped before `Done`.
            send(refuse(StatusCode::SERVICE_UNAVAILABLE, "the engine stopped", None).body());
            let _ = out.send(Event::default().data("[DONE]"));
        });
        let events = UnboundedReceiverStream::new(events).map(Ok::<Event, Infallible>);
        Sse::new(events)
            .keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_secs(5))
                    .text("keep-alive"),
            )
            .into_response()
    }

    async fn respond(self) -> Response {
        match self.stream {
            true => self.streamed(),
            false => self.whole().await,
        }
    }
}

pub(crate) async fn completions(State(slot): State<Slot>, body: Bytes) -> Response {
    match with_model(&slot, move |m| complete(m, &body)).await {
        Ok(started) => started.respond().await,
        Err(refused) => refused.into_response(),
    }
}

pub(crate) async fn chat_completions(State(slot): State<Slot>, body: Bytes) -> Response {
    match with_model(&slot, move |m| chat(m, &body)).await {
        Ok(started) => started.respond().await,
        Err(refused) => refused.into_response(),
    }
}
