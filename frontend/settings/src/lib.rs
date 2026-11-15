//! Settings read once from the environment, passed down.

use engine_settings::{Backend, Pick, Settings};
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The `TESSERA_` variables. Unlisted names are a model's reference files.
#[derive(Deserialize)]
struct Env {
    backend: Option<String>,
    lanes: Option<usize>,
    costs: Option<PathBuf>,
    names: Option<PathBuf>,
    emulate: Option<Backend>,
    #[serde(flatten)]
    references: BTreeMap<String, PathBuf>,
}

/// Variables without the prefix.
#[derive(Deserialize)]
struct Host {
    ci: Option<String>,
}

/// The only reader of the process environment. Bad values panic.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn read() -> Settings {
    let env: Env = envy::prefixed("TESSERA_")
        .from_env()
        .unwrap_or_else(|e| panic!("TESSERA_ environment: {e}"));
    let host: Host = envy::from_env().unwrap_or_else(|e| panic!("environment: {e}"));
    // `CI` picks the host: hosted macOS runners lack Metal.
    let host =
        env.backend.as_deref() == Some("host") || (cfg!(target_os = "macos") && host.ci.is_some());
    let cpu = env.backend.as_deref() == Some("cpu");
    let costs = env
        .costs
        .or_else(|| Some(std::env::home_dir()?.join(".cache/tessera/costs")));
    Settings {
        backend: match (host, cpu) {
            (true, _) => Pick::Host,
            (false, true) => Pick::Cpu,
            (false, false) => Pick::Auto,
        },
        host_lanes: env.lanes.unwrap_or(32),
        costs,
        references: env.references,
        names: env.names,
        emulate: env.emulate,
    }
}

#[cfg(test)]
mod lib_test;
