//! Settings read once from the environment, passed down.

use engine_settings::{Backend, Pick, Settings};
use serde::Deserialize;
use std::path::PathBuf;

/// The `TESSERA_` variables.
#[derive(Deserialize)]
struct Env {
    backend: Option<String>,
    lanes: Option<usize>,
    costs: Option<PathBuf>,
    names: Option<PathBuf>,
    emulate: Option<Backend>,
}

/// The only reader of the process environment. Bad values panic.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn read() -> Settings {
    let env: Env = envy::prefixed("TESSERA_")
        .from_env()
        .unwrap_or_else(|e| panic!("TESSERA_ environment: {e}"));
    let backend = match env.backend.as_deref() {
        Some("host") => Pick::Host,
        Some("cpu") => Pick::Cpu,
        _ => gpukit::backend::picked().map_or(Pick::Host, Pick::Gpu),
    };
    let costs = env
        .costs
        .or_else(|| Some(std::env::home_dir()?.join(".cache/tessera/costs")));
    Settings {
        backend,
        host_lanes: env.lanes.unwrap_or(32),
        costs,
        names: env.names,
        emulate: env.emulate,
    }
}

#[cfg(test)]
mod lib_test;
