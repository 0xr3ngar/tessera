//! Request bodies, their checks, and the OpenAI error object.

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::{Json, body::Bytes};
use bundle::Message;
use serde::Deserialize;
use serde::de::DeserializeOwned;
use serde_json::{Map, Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
use tasks::api::Info;

/// Why a request was refused, as OpenAI reports it.
#[derive(Debug)]
pub(crate) struct Refusal {
    pub(crate) status: StatusCode,
    pub(crate) message: String,
    pub(crate) param: Option<String>,
}

impl IntoResponse for Refusal {
    fn into_response(self) -> Response {
        let kind = match self.status {
            StatusCode::NOT_FOUND => "NotFoundError",
            StatusCode::SERVICE_UNAVAILABLE => "ServiceUnavailableError",
            StatusCode::INTERNAL_SERVER_ERROR => "InternalServerError",
            _ => "BadRequestError",
        };
        let error = json!({"error": {
            "message": self.message,
            "type": kind,
            "param": self.param,
            "code": self.status.as_u16(),
        }});
        (self.status, Json(error)).into_response()
    }
}

pub(crate) fn refuse(
    status: StatusCode,
    message: impl Into<String>,
    param: Option<&str>,
) -> Refusal {
    Refusal {
        status,
        message: message.into(),
        param: param.map(str::to_string),
    }
}

pub(crate) fn bad(message: impl Into<String>, param: &str) -> Refusal {
    refuse(StatusCode::BAD_REQUEST, message, Some(param))
}

/// Seconds since 1970, for `created`.
pub(crate) fn now() -> u64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |d| d.as_secs())
}

/// Whether `value` keeps greedy decoding. `None`: not sampling.
fn greedy(key: &str, value: &Value) -> Option<bool> {
    let neutral: &[f64] = match key {
        "temperature" | "presence_penalty" | "frequency_penalty" | "min_p" => &[0.0],
        "top_p" | "n" | "best_of" | "repetition_penalty" => &[1.0],
        "top_k" => &[-1.0, 0.0],
        "logprobs" | "echo" => return Some(value.is_null() || *value == json!(false)),
        _ => return None,
    };
    Some(value.is_null() || value.as_f64().is_some_and(|v| neutral.contains(&v)))
}

/// The body as `T`. `known` lists the fields read.
pub(crate) fn parse<T: DeserializeOwned>(
    info: &Info,
    body: &Bytes,
    known: &[&str],
) -> Result<T, Refusal> {
    let body: Map<String, Value> = serde_json::from_slice(body)
        .map_err(|e| refuse(StatusCode::BAD_REQUEST, e.to_string(), None))?;
    match body.get("model") {
        None | Some(Value::Null) => {}
        Some(Value::String(m)) if *m == info.name => {}
        Some(m) => {
            let m = m.as_str().map_or_else(|| m.to_string(), str::to_string);
            let message = format!("The model `{m}` does not exist.");
            return Err(refuse(StatusCode::NOT_FOUND, message, Some("model")));
        }
    }
    for (key, value) in &body {
        if key == "model" || known.contains(&key.as_str()) {
            continue;
        }
        match greedy(key, value) {
            Some(true) => {}
            Some(false) => {
                return Err(bad(
                    format!("`{key}`: only greedy decoding is supported"),
                    key,
                ));
            }
            None => return Err(bad(format!("`{key}` is not supported"), key)),
        }
    }
    serde_json::from_value(Value::Object(body))
        .map_err(|e| refuse(StatusCode::BAD_REQUEST, e.to_string(), None))
}

/// One string or a list of them.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Strings {
    One(String),
    Many(Vec<String>),
}

impl Strings {
    pub(crate) fn list(self) -> Vec<String> {
        match self {
            Strings::One(s) => vec![s],
            Strings::Many(v) => v,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct StreamOptions {
    #[serde(default)]
    pub(crate) include_usage: bool,
}

/// What completions and chat share.
#[derive(Deserialize)]
pub(crate) struct Common {
    pub(crate) max_tokens: Option<usize>,
    #[serde(default)]
    pub(crate) stream: bool,
    pub(crate) stream_options: Option<StreamOptions>,
    pub(crate) stop: Option<Strings>,
}

pub(crate) const COMMON: [&str; 6] = [
    "max_tokens",
    "stream",
    "stream_options",
    "stop",
    "seed",
    "user",
];

/// A prompt as ids, a text or texts.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Prompt {
    Ids(Vec<u32>),
    Text(Strings),
}

#[derive(Deserialize)]
pub(crate) struct Completion {
    pub(crate) prompt: Prompt,
    #[serde(flatten)]
    pub(crate) common: Common,
}

#[derive(Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Part {
    Text { text: String },
}

#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Content {
    Text(String),
    Parts(Vec<Part>),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Said {
    role: String,
    content: Content,
}

impl From<Said> for Message {
    fn from(said: Said) -> Message {
        let content = match said.content {
            Content::Text(text) => text,
            Content::Parts(parts) => parts.into_iter().map(|Part::Text { text }| text).collect(),
        };
        Message {
            role: said.role,
            content,
        }
    }
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct Kwargs {
    pub(crate) enable_thinking: Option<bool>,
}

/// Messages and how to lay them out.
#[derive(Deserialize)]
pub(crate) struct Messages {
    pub(crate) messages: Vec<Said>,
    #[serde(default)]
    pub(crate) chat_template_kwargs: Kwargs,
}

#[derive(Deserialize)]
pub(crate) struct Chat {
    #[serde(flatten)]
    pub(crate) messages: Messages,
    pub(crate) max_completion_tokens: Option<usize>,
    #[serde(flatten)]
    pub(crate) common: Common,
}

/// What `/tokenize` takes: a prompt or messages.
#[derive(Deserialize)]
#[serde(untagged)]
pub(crate) enum Tokenize {
    Prompt { prompt: String },
    Chat(Messages),
}

#[derive(Deserialize)]
pub(crate) struct Detokenize {
    pub(crate) tokens: Vec<u32>,
}

#[derive(Deserialize)]
pub(crate) struct Embeddings {
    pub(crate) input: Strings,
    pub(crate) encoding_format: Option<String>,
}
