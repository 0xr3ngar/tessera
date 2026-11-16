use std::path::PathBuf;

/// `model`'s reference files, when this host keeps them.
#[allow(clippy::disallowed_methods, reason = "the cache lies under home")]
pub fn reference(model: &str) -> Option<PathBuf> {
    let cache = std::env::home_dir()?.join(".cache/tessera/references");
    let dir = cache.join(model);
    dir.is_dir().then_some(dir)
}
