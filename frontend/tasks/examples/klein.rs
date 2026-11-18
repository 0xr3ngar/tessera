//! A FLUX.2 klein picture for a prompt, written to PNG.

use checkpoint::File;
use std::time::Instant;
use tasks::klein::{Klein, noise};
use tasks::png;

fn main() -> Result<(), tasks::Error> {
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
    let started = Instant::now();
    let mut klein = Klein::open(dir.into(), frontend_settings::read(), side * 16)?;
    eprintln!("opened in {:?}", started.elapsed());
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
