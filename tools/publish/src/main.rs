//! Writes the merged crates.io crates to `target/publish` or `[dir]`.

mod emit;
mod manifest;
mod meta;
mod plan;
mod rewrite;
mod tokens;

#[cfg(test)]
mod rewrite_test;

use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use toml::{Table, Value};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-publish: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let meta = meta::load(&std::env::current_dir()?)?;
    let base = std::env::args()
        .nth(1)
        .map_or_else(|| meta.workspace_root.join("target/publish"), PathBuf::from);
    let outs = plan::plan(&meta)?;
    let engine = meta
        .packages
        .iter()
        .find(|p| p.name == plan::ENGINE)
        .ok_or("no engine")?;
    let version = engine.version.clone();
    let mut owners = manifest::Owners::new();
    for o in &outs {
        for m in &o.members {
            owners.insert(m.pkg, o.name.as_str());
        }
    }
    if base.exists() {
        fs::remove_dir_all(&base)?;
    }
    for o in &outs {
        emit::emit(&meta, o, &owners, &version, &base)?;
    }
    let mut workspace = Table::new();
    workspace.insert("resolver".into(), "3".into());
    let members = outs.iter().map(|o| Value::from(o.name.clone())).collect();
    workspace.insert("members".into(), Value::Array(members));
    let mut doc = Table::new();
    doc.insert("workspace".into(), workspace.into());
    fs::write(base.join("Cargo.toml"), toml::to_string(&doc)?)?;
    fs::copy(
        meta.workspace_root.join("Cargo.lock"),
        base.join("Cargo.lock"),
    )?;
    println!(
        "tessera-publish: {} crates in {}",
        outs.len(),
        base.display()
    );
    Ok(())
}
