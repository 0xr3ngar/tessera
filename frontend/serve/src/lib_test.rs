use crate::Slot;
use crate::mock::{Mock, call, json, loaded};
use axum::http::StatusCode;
use serde_json::json;
use std::sync::{Arc, OnceLock};

#[tokio::test]
async fn health_waits() {
    let empty: Slot = Arc::new(OnceLock::new());
    let (status, loading) = json(&empty, "/v1/health", None).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(loading["error"]["type"], "unavailable_error");
    let (status, _) = call(&empty, "/v1/completions", Some(json!({"prompt": "a"}))).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let slot = loaded(&Arc::new(Mock::default()), false);
    assert_eq!(json(&slot, "/ping", None).await.1, json!({"status": "ok"}));
    let (_, version) = json(&slot, "/version", None).await;
    assert_eq!(version["version"], env!("CARGO_PKG_VERSION"));
}

#[tokio::test]
async fn small_routes() {
    let slot = loaded(&Arc::new(Mock::default()), false);
    let (_, models) = json(&slot, "/v1/models", None).await;
    assert_eq!(
        (
            &models["data"][0]["id"],
            &models["data"][0]["max_model_len"]
        ),
        (&json!("m"), &json!(16))
    );
    let (_, tokens) = json(&slot, "/tokenize", Some(json!({"prompt": "ab"}))).await;
    assert_eq!(
        tokens,
        json!({"count": 2, "max_model_len": 16, "tokens": [97, 98]})
    );
    let (_, text) = json(&slot, "/detokenize", Some(json!({"tokens": [97, 98]}))).await;
    assert_eq!(text["prompt"], "ab");
    let (_, embedded) = json(
        &slot,
        "/v1/embeddings",
        Some(json!({"input": ["a", "bcd"]})),
    )
    .await;
    assert_eq!(embedded["data"][1]["embedding"], json!([3.0]));
    assert_eq!(embedded["usage"]["prompt_tokens"], 2);
}
