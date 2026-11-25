//! A FLUX.2 klein picture for a prompt, written to PNG.

use std::sync::Arc;
use std::time::{Duration, Instant};
use tessera_bundle::Overrides;
use tessera_checkpoint::File;
use tessera_recipe::Recipe;
use tessera_recipe::progress::{Log, Progress};
use tessera_tasks::Error;
use tessera_tasks::flow::{Hold, Hook, Scripted, Step};
use tessera_tasks::klein::{Klein, noise};
use tessera_tasks::png;

/// Times each step and the inner hook inside the step.
struct Timed {
    name: String,
    inner: Option<Box<dyn Hook>>,
    started: Instant,
    /// Each step, then the time inside the inner hook.
    steps: Vec<(Duration, Duration)>,
    hooked: Duration,
}

impl Hook for Timed {
    fn before(&mut self, step: &Step, x: &mut [f32]) -> Result<(), Error> {
        self.started = Instant::now();
        let out = self.inner.as_mut().map_or(Ok(()), |h| h.before(step, x));
        self.hooked = self.started.elapsed();
        out
    }

    fn after(&mut self, step: &mut Step, x: &mut [f32], v: &mut [f32]) -> Result<(), Error> {
        let at = Instant::now();
        let out = self.inner.as_mut().map_or(Ok(()), |h| h.after(step, x, v));
        let hooked = self.hooked + at.elapsed();
        self.steps.push((self.started.elapsed(), hooked));
        if step.index + 1 == step.steps {
            let steps = std::mem::take(&mut self.steps);
            eprintln!("hook {} steps {steps:?}", self.name);
        }
        out
    }
}

/// `none`, `hold` (left half at zero) or a script path.
fn hook(name: &str, side: usize) -> Result<Option<Box<dyn Hook>>, Error> {
    match name {
        "none" => return Ok(None),
        "hold" => {}
        path => return Ok(Some(Box::new(Scripted::open(path.as_ref())?))),
    }
    let mask = (0..side * side).map(|t| t % side < side / 2).collect();
    Ok(Some(Box::new(Hold::new(
        mask,
        vec![0.0; side * side * 128],
    ))))
}

fn main() -> Result<(), Error> {
    let mut args: Vec<String> = std::env::args().collect();
    let usage = "usage: klein [--hook none|hold|<file.js>,...] [--runs <n>] <recipe> <out.png> \
<latent side> <seed | noise.safetensors> <prompt>";
    let mut take = |flag: &str| {
        let at = args.iter().position(|a| a == flag)?;
        let value = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        value
    };
    let (named, runs) = (take("--hook"), take("--runs"));
    let runs: usize = runs.as_deref().unwrap_or("1").parse()?;
    let [_, recipe, out, side, from, prompt @ ..] = args.as_slice() else {
        return Err(usage.into());
    };
    let side: usize = side.parse()?;
    let start = match from.parse::<u64>() {
        Ok(seed) => noise(seed, side * side * 128),
        Err(_) => File::open(from.as_ref())?.floats("noise")?,
    };
    let settings = tessera_frontend_settings::read();
    let (cache, token) = (settings.recipes.clone(), settings.hf_token.clone());
    let recipe = Recipe::open(recipe, cache.as_deref())?;
    let progress = Progress::new(Arc::new(Log));
    let dir = recipe.weights(cache.as_deref(), token.as_deref(), &progress)?;
    let config = Overrides::new(recipe.overrides());
    let started = Instant::now();
    let mut klein = Klein::open(dir, settings, side * 16, config, &progress)?;
    eprintln!("opened in {:?}", started.elapsed());
    let names: Vec<String> = named
        .iter()
        .flat_map(|n| n.split(','))
        .map(String::from)
        .collect();
    for run in 0..runs {
        if let Some(name) = names.get(run % names.len().max(1)) {
            klein.hook = Some(Box::new(Timed {
                name: name.clone(),
                inner: hook(name, side)?,
                started: Instant::now(),
                steps: Vec::new(),
                hooked: Duration::ZERO,
            }));
        }
        let started = Instant::now();
        let picture = klein.paint(&prompt.join(" "), side, &start)?;
        let elapsed = started.elapsed();
        eprintln!(
            "painted {}x{} in {elapsed:?}",
            picture.width, picture.height
        );
        std::fs::write(out, png::encode(&picture))?;
    }
    Ok(())
}
