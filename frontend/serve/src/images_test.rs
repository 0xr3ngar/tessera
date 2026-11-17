use crate::mocks::{MockGenerator, json, loaded};
use serde_json::json;
use std::sync::Arc;

#[test]
fn pictures() {
    let slot = loaded(&Arc::new(MockGenerator::default()), false);
    let body = json!({"prompt": "a duck", "n": 2, "size": "4x4", "seed": 7});
    let (status, got) = json(&slot, "/v1/images/generations", Some(body));
    assert_eq!((status, got["data"].as_array().unwrap().len()), (200, 2));
    // The PNG signature, in base64.
    assert!(
        got["data"][1]["b64_json"]
            .as_str()
            .unwrap()
            .starts_with("iVBORw0KGgo")
    );
    for (field, value) in [
        ("size", json!("4x2")),
        ("n", json!(11)),
        ("response_format", json!("url")),
    ] {
        let body = json!({"prompt": "a", field: value});
        let (status, got) = json(&slot, "/v1/images/generations", Some(body));
        assert_eq!((status, got["error"]["param"].as_str()), (400, Some(field)));
    }
}
