//! A decoder's residual rows after chosen layers, as text features.

use crate::tokenize::hf::Hf;
use crate::{Template, config, pull};
use checkpoint::File;
use engine::decoder::{Swiglu, taps};
use engine::{Graph, Norm, Pins, Weight};
use std::error::Error;
use std::path::Path;

type Result<T> = std::result::Result<T, Box<dyn Error + Send + Sync>>;

pub struct Tapped {
    pub graph: Graph,
    pub weights: Vec<Weight>,
    pub pins: Pins,
    pub tokenizer: Hf,
    pub template: Template,
    /// The id prompts are padded with.
    pub pad: u32,
}

/// The tokenizer config's pad token, as an id.
fn pad(dir: &Path, tokenizer: &Hf) -> Result<u32> {
    let text = std::fs::read_to_string(dir.join("tokenizer_config.json"))?;
    let config: serde_json::Value = serde_json::from_str(&text)?;
    let token = config["pad_token"].as_str().ok_or("no pad token")?;
    Ok(tokenizer.id(token).ok_or("the pad token has no id")?)
}

/// The model in `dir`, cut after its deepest tap.
pub fn tapped(dir: &Path, [seqs, context]: [u64; 2], layers: &[u64]) -> Result<Tapped> {
    let (arch, mut config) = config(dir, seqs, context)?;
    let deepest = layers.iter().copied().max().ok_or("no layers to tap")?;
    if deepest > config.stack.layers {
        return Err(format!("layer {deepest} of {}", config.stack.layers).into());
    }
    config.stack.layers = deepest;
    let graph = match arch.as_str() {
        "qwen3" => taps::<Norm, Swiglu>(&config, layers),
        "llama" => taps::<(), Swiglu>(&config, layers),
        _ => return Err(format!("{arch} has no taps here").into()),
    };
    let file = File::open(&dir.join("model.safetensors"))?;
    let (weights, pins) = pull(&graph, &file, deepest, &[])?;
    let tokenizer = Hf::open(&dir.join("tokenizer.json"))?;
    let template = Template::load(dir)?.ok_or("no chat template")?;
    Ok(Tapped {
        graph,
        weights,
        pins,
        pad: pad(dir, &tokenizer)?,
        tokenizer,
        template,
    })
}
