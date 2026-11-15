//! Settings read once from the environment, passed down.

use gpukit::backend::Backend;
use gpukit::{Pick, Settings};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// The only reader of the process environment.
#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
pub fn read() -> Settings {
    use std::env::{home_dir, var, var_os};
    // `CI` picks the host: hosted macOS runners lack Metal.
    let host = var("TESSERA_BACKEND").as_deref() == Ok("cpu")
        || (cfg!(target_os = "macos") && var_os("CI").is_some());
    // The `FIELDS` variables fill fields. Other TESSERA_ names are references.
    let lanes = var("TESSERA_LANES").ok().and_then(|v| v.parse().ok());
    let costs = var_os("TESSERA_COSTS")
        .map(PathBuf::from)
        .or_else(|| Some(home_dir()?.join(".cache/tessera/costs")));
    Settings {
        backend: if host { Pick::Host } else { Pick::Auto },
        host_lanes: lanes.unwrap_or(32),
        costs,
        references: references(),
        names: var_os("TESSERA_NAMES").map(PathBuf::from),
        emulate: match var("TESSERA_EMULATE").as_deref() {
            Ok("metal") => Some(Backend::Metal),
            Ok("cuda") => Some(Backend::Cuda),
            Ok("rocm") => Some(Backend::Rocm),
            _ => None,
        },
    }
}

/// Variables that set a field, not a model's files.
const FIELDS: [&str; 5] = ["BACKEND", "LANES", "COSTS", "NAMES", "EMULATE"];

#[allow(clippy::disallowed_methods, reason = "the one sanctioned reader")]
fn references() -> BTreeMap<String, PathBuf> {
    let mut out = BTreeMap::new();
    for (key, value) in std::env::vars_os() {
        let key = key.to_string_lossy();
        if let Some(model) = key.strip_prefix("TESSERA_")
            && !FIELDS.contains(&model)
        {
            out.insert(model.to_lowercase(), PathBuf::from(value));
        }
    }
    out
}
