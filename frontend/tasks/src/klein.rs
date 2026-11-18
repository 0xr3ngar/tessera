//! FLUX.2 klein: a prompt to a picture.

use crate::Error;
use crate::api::Image;
use crate::features::Features;
use crate::flow::Flow;
use crate::pictures::Pictures;
use sensai_engine::Settings;
use std::path::PathBuf;

/// Text rows klein pads every prompt to.
const ROWS: usize = 512;
/// The Qwen3 layers klein reads.
const LAYERS: [u64; 3] = [9, 18, 27];
/// Channels per latent token.
const LATENT: usize = 128;

/// Diffusers' fitted shift for `tokens` rows over `steps`.
fn mu(tokens: usize, steps: usize) -> f64 {
    let (a1, b1, a2, b2) = (8.738_095_24e-5, 1.898_333_33, 0.000_169_27, 0.456_666_66);
    let n = tokens as f64;
    if n > 4300.0 {
        return a2 * n + b2;
    }
    let (m200, m10) = (a2 * n + b2, a1 * n + b1);
    let a = (m200 - m10) / 190.0;
    a * steps as f64 + (m200 - 200.0 * a)
}

/// Times from 1 to 0, shifted toward noise.
pub(crate) fn sigmas(tokens: usize, steps: usize) -> Vec<f32> {
    let shift = mu(tokens, steps).exp();
    let mut out: Vec<f32> = (0..steps)
        .map(|i| 1.0 - i as f64 * (1.0 - 1.0 / steps as f64) / (steps - 1).max(1) as f64)
        .map(|s| (shift / (shift + (1.0 / s - 1.0))) as f32)
        .collect();
    out.push(0.0);
    out
}

pub struct Klein {
    /// The diffusers repository: `text_encoder`, `tokenizer`, `transformer`, `vae`.
    pub dir: PathBuf,
    pub settings: Settings,
    pub steps: usize,
}

impl Klein {
    /// A picture `16 side` pixels a side, from `noise` latents.
    pub fn paint(&self, prompt: &str, side: usize, noise: &[f32]) -> Result<Image, Error> {
        let rows = side * side;
        if noise.len() != rows * LATENT {
            return Err(format!("{} noise values for {rows} tokens", noise.len()).into());
        }
        let dirs = [&self.dir.join("text_encoder"), &self.dir.join("tokenizer")];
        let text = Features::open(
            dirs.map(|d| d.as_path()),
            &self.settings,
            &LAYERS,
            [1, ROWS],
        )?
        .encode(&[prompt])?
        .remove(0);
        // Each stage drops its model before the next opens.
        let mut flow = Flow::open(&self.dir, &self.settings, [1, ROWS, rows])?;
        let mut x = noise.to_vec();
        let s = sigmas(rows, self.steps);
        for pair in s.windows(2) {
            let v = flow.velocity(&x, &text, side, pair[0])?;
            x.iter_mut()
                .zip(&v)
                .for_each(|(x, v)| *x += (pair[1] - pair[0]) * v);
        }
        drop(flow);
        Ok(Pictures::open(&self.dir, &self.settings, side, 1)?
            .decode(&x)?
            .remove(0))
    }
}
