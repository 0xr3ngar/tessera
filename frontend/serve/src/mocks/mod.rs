//! Mocks of each trait, and a client for the routes.

mod chat;
mod embed;
mod generator;
mod tokens;

use crate::{Model, Slot, router};
use axum::body::Body;
use axum::http::{Request, StatusCode};
use chat::MockChat;
use embed::MockEmbed;
pub use generator::MockGenerator;
use http_body_util::BodyExt;
use serde_json::Value;
use std::sync::{Arc, OnceLock};
use tasks::api::Info;
use tokens::MockTokens;
use tower::ServiceExt;

/// The mocks as model `m`, 16 tokens of context.
pub fn loaded(generator: &Arc<MockGenerator>, strict: bool) -> Slot {
    let slot: Slot = Arc::new(OnceLock::new());
    let _ = slot.set(Model {
        info: Info {
            name: "m".into(),
            context: 16,
        },
        tokens: Arc::new(MockTokens),
        chat: Some(Arc::new(MockChat)),
        generator: Some(generator.clone()),
        embed: Some(Arc::new(MockEmbed)),
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
