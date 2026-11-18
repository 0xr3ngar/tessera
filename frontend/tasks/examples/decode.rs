//! Decodes saved FLUX.2 latents to PNG files through the VAE.

use checkpoint::File;
use std::time::Instant;
use tasks::pictures::Pictures;
use tasks::png;

fn main() -> Result<(), tasks::Error> {
    let args: Vec<String> = std::env::args().collect();
    let [_, dir, file, name, side, out] = args.as_slice() else {
        return Err(
            "usage: decode <model dir> <latents.safetensors> <tensor> <side> <out.png>".into(),
        );
    };
    let tokens = File::open(file.as_ref())?.floats(name)?;
    let side: usize = side.parse()?;
    let mut p = Pictures::open(dir.as_ref(), &frontend_settings::read(), side, 1)?;
    let started = Instant::now();
    let picture = p.decode(&tokens)?.remove(0);
    eprintln!(
        "decoded {}x{} in {:?}",
        picture.width,
        picture.height,
        started.elapsed()
    );
    std::fs::write(out, png::encode(&picture))?;
    Ok(())
}
