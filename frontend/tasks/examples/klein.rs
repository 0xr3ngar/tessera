//! A FLUX.2 klein picture for a prompt, written to PNG.

use sensai_checkpoint::File;
use sensai_tasks::klein::Klein;
use sensai_tasks::png;
use std::time::Instant;

/// Standard normal values from a seed, by Box-Muller.
fn noise(seed: u64, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    let mut uniform = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| {
            ((-2.0 * uniform().ln()).sqrt() * (std::f64::consts::TAU * uniform()).cos()) as f32
        })
        .collect()
}

fn main() -> Result<(), sensai_tasks::Error> {
    let args: Vec<String> = std::env::args().collect();
    let usage =
        "usage: klein <model dir> <out.png> <latent side> <seed | noise.safetensors> <prompt>";
    let [_, dir, out, side, from, prompt @ ..] = args.as_slice() else {
        return Err(usage.into());
    };
    let side: usize = side.parse()?;
    let start = match from.parse::<u64>() {
        Ok(seed) => noise(seed, side * side * 128),
        Err(_) => File::open(from.as_ref())?.floats("noise")?,
    };
    let klein = Klein {
        dir: dir.into(),
        settings: sensai_frontend_settings::read(),
        steps: 4,
    };
    let started = Instant::now();
    let picture = klein.paint(&prompt.join(" "), side, &start)?;
    eprintln!(
        "painted {}x{} in {:?}",
        picture.width,
        picture.height,
        started.elapsed()
    );
    std::fs::write(out, png::encode(&picture))?;
    Ok(())
}
