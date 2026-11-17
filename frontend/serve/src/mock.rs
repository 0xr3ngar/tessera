//! A mock model, and a client for the routes.

use crate::{Model, Slot, router};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use bundle::Message;
use http_body_util::BodyExt;
use serde_json::Value;
use std::sync::{Arc, Mutex, OnceLock};
use tasks::Error;
use tasks::api::{Chat, Embed, Embedded, Finish, Generator, Info, Params, Sink, Step, Tokens};
use tower::ServiceExt;

/// One id per character. Answers "hello" in two tokens.
#[derive(Default)]
pub struct Mock {
    /// The params of the last answer.
    pub asked: Mutex<Option<Params>>,
}

impl Tokens for Mock {
    fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        Ok(text.chars().map(u32::from).collect())
    }

    fn decode(&self, ids: &[u32]) -> String {
        ids.iter().filter_map(|&id| char::from_u32(id)).collect()
    }
}

impl Chat for Mock {
    fn render(&self, messages: &[Message], _think: bool) -> Result<String, Error> {
        Ok(messages
            .iter()
            .map(|m| format!("{}:{}", m.role, m.content))
            .collect())
    }
}

impl Generator for Mock {
    fn generate(&self, _prompt: Vec<u32>, params: Params, mut sink: Sink) -> Result<(), Error> {
        *self.asked.lock().unwrap() = Some(params);
        sink(Step::Token {
            id: 1,
            text: "he".into(),
        });
        sink(Step::Token {
            id: 2,
            text: String::new(),
        });
        sink(Step::Done {
            text: "llo".into(),
            finish: Finish::Stop,
            tokens: 2,
        });
        Ok(())
    }
}

impl Embed for Mock {
    fn embed(&self, texts: &[String]) -> Result<Embedded, Error> {
        Ok(Embedded {
            vectors: texts.iter().map(|t| vec![t.len() as f32]).collect(),
            tokens: texts.len(),
        })
    }
}

/// The mock as model `m`, 16 tokens of context.
pub fn loaded(mock: &Arc<Mock>, strict: bool) -> Slot {
    let slot: Slot = Arc::new(OnceLock::new());
    let _ = slot.set(Model {
        info: Info {
            name: "m".into(),
            context: 16,
        },
        tokens: mock.clone(),
        chat: Some(mock.clone()),
        generator: Some(mock.clone()),
        embed: Some(mock.clone()),
        strict,
    });
    slot
}

/// The status and body of a call. No body: GET.
pub async fn call(slot: &Slot, path: &str, body: Option<Value>) -> (StatusCode, String) {
    let request = match body {
        Some(b) => Request::post(path).body(Body::from(b.to_string())),
        None => Request::get(path).body(Body::empty()),
    };
    let response = router(slot.clone())
        .oneshot(request.unwrap())
        .await
        .unwrap();
    let status = response.status();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// The same, with the body as JSON.
pub async fn json(slot: &Slot, path: &str, body: Option<Value>) -> (StatusCode, Value) {
    let (status, text) = call(slot, path, body).await;
    (status, serde_json::from_str(&text).unwrap())
}
