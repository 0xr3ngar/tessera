//! FLUX.2's VAE decoder from a diffusers directory.

use checkpoint::File;
use engine::vae::{Config, program};
use engine::{Graph, PathSeg, Tensor, Weight, bind};
use std::error::Error;
use std::path::Path;

/// The batch norm's floor under the variance, from `vae/config.json`.
const BN_EPS: f32 = 1e-4;

pub struct Decoder {
    pub graph: Graph,
    pub weights: Vec<Weight>,
}

fn dotted(path: &[PathSeg]) -> String {
    let seg = |p: &PathSeg| match p {
        PathSeg::Field(f) => f.clone(),
        PathSeg::Index(i) => i.to_string(),
    };
    path.iter().map(seg).collect::<Vec<_>>().join(".")
}

/// `[out, in, k, k]` kernels as `[out, k, k, in]`.
fn taps(v: &[f32], shape: &[usize]) -> Vec<f32> {
    let [out, cin, kh, kw] = shape[..] else {
        return v.to_vec();
    };
    let mut t = vec![0.0; v.len()];
    for o in 0..out {
        for c in 0..cin {
            for k in 0..kh * kw {
                t[(o * kh * kw + k) * cin + c] = v[(o * cin + c) * kh * kw + k];
            }
        }
    }
    t
}

/// `dir/vae`, `side` tokens a side, `seqs` per launch.
pub fn decoder(dir: &Path, side: u64, seqs: u64) -> Result<Decoder, Box<dyn Error + Send + Sync>> {
    let graph = program(&Config::flux2(side, seqs));
    let file = File::open(&dir.join("vae").join("diffusion_pytorch_model.safetensors"))?;
    let find = |p: &[PathSeg]| {
        let name = dotted(p);
        let v = match name.as_str() {
            "bn.std" => (file.floats("bn.running_var").ok()?.iter())
                .map(|v| (v + BN_EPS).sqrt())
                .collect(),
            _ => taps(&file.floats(&name).ok()?, file.shape(&name)?),
        };
        Some(Tensor::f32(vec![v.len()], v))
    };
    let weights = bind(&graph, find)?
        .into_iter()
        .map(Weight::Floats)
        .collect();
    Ok(Decoder { graph, weights })
}
