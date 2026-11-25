//! FLUX.2 klein pictures from a diffusers repository.

use super::{Entry, Flag, Load, Pick};
use std::path::PathBuf;
use tessera_engine::Settings;
use tessera_tasks::Worker;
use tessera_tasks::klein::Klein;

pub static ENTRY: Entry = Entry {
    task: "image",
    pick: Pick::Class("Flux2KleinPipeline"),
    flags: &[
        Flag::count("--image-size", "<px>"),
        Flag::text("--script", "<file.js>"),
    ],
    open,
};

fn open(load: &Load, settings: Settings) -> Result<(), String> {
    let o = load.options;
    let largest = load.count("--image-size", 512);
    let (dir, config, progress) = (o.model.clone(), o.config.clone(), o.progress.clone());
    let script = load.given("--script").map(PathBuf::from);
    let klein = Worker::<Klein>::open(dir, settings, largest, config, script, progress);
    load.picture(klein.map_err(|e| e.to_string())?);
    Ok(())
}
