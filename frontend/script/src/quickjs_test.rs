use crate::{Error, Limits, Script};
use serde_json::json;
use std::time::Duration;

fn load(source: &str) -> Script {
    let limits = Limits {
        time: Duration::from_millis(50),
        ..Limits::default()
    };
    Script::load("test.js", source, limits).unwrap()
}

#[test]
fn throw_fails() {
    let mut s = load("export function step(s) { throw new Error('no ' + s.at) }");
    let got = s.call("step", &json!({ "at": 3 }), &mut []);
    assert!(
        matches!(&got, Err(Error::Thrown(_, m)) if m.contains("no 3")),
        "{got:?}"
    );
}

#[test]
fn loop_times_out() {
    let mut s = load("export function step(s) { for (;;) { s.x[0] += 1 } }");
    let mut x = [0.0f32];
    let got = s.call("step", &json!({}), &mut [("x", &mut x)]);
    assert!(matches!(got, Err(Error::Time(..))), "{got:?}");
    assert_eq!(x, [0.0]);
}
