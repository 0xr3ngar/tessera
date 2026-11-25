//! FLUX.2's transformer as a velocity field over latent rows.

use crate::Error;
use serde_json::Value;
use std::path::Path;
use tessera_bundle::Overrides;
use tessera_bundle::flux2::transformer;
use tessera_engine::flux2::{AXES, LATENT, TIME};
use tessera_engine::progress::{Progress, Unit};
use tessera_engine::{Model, Recipe, Settings, Tensor};
use tessera_script::Array;

pub struct Flow {
    model: Model,
    /// Text rows per canvas, pads included.
    text: usize,
}

/// The timestep's sinusoids, cosines first, matching diffusers.
fn sinusoids(t: f32) -> Vec<f32> {
    let half = TIME as usize / 2;
    let at = |i: usize| t * 1000.0 * (-(10_000f32).ln() * i as f32 / half as f32).exp();
    let cos = (0..half).map(|i| at(i).cos());
    cos.chain((0..half).map(|i| at(i).sin())).collect()
}

/// Each axis's positions: text by index, then the grid.
fn positions(text: usize, side: usize) -> [Vec<u32>; AXES] {
    let mut ids: [Vec<u32>; AXES] = Default::default();
    for l in 0..text {
        [0, 0, 0, l]
            .iter()
            .zip(&mut ids)
            .for_each(|(&v, a)| a.push(v as u32));
    }
    for cell in 0..side * side {
        let at = [0, cell / side, cell % side, 0];
        at.iter().zip(&mut ids).for_each(|(&v, a)| a.push(v as u32));
    }
    ids
}

impl Flow {
    /// The diffusers model in `dir`, sized for one launch.
    pub fn open(
        dir: &Path,
        settings: &Settings,
        [seqs, text, rows]: [usize; 3],
        config: &Overrides,
        progress: &Progress,
    ) -> Result<Self, Error> {
        let reading = progress.step("read model", Unit::Items, None);
        let t = transformer(dir, [seqs as u64, text as u64, rows as u64], config)?;
        reading.finish();
        let recipe = Recipe {
            pins: t.pins,
            ..Recipe::default()
        };
        let model = Model::open_loaded(t.graph, &recipe, settings, &t.weights, progress)?;
        Ok(Self { model, text })
    }

    /// Velocity for square latents at time `t`.
    pub fn velocity(
        &mut self,
        latents: &[f32],
        text: &[f32],
        side: usize,
        t: f32,
    ) -> Result<Vec<f32>, Error> {
        let rows = side * side;
        let n = latents.len() / (rows * LATENT as usize);
        let width = text.len() / (n * self.text);
        let joint = self.text + rows;
        let mut inputs = vec![
            Tensor::f32(vec![n, rows, LATENT as usize], latents.to_vec()),
            Tensor::f32(vec![n, self.text, width], text.to_vec()),
            Tensor::f32(vec![n, TIME as usize], sinusoids(t).repeat(n)),
        ];
        for axis in positions(self.text, side) {
            inputs.push(Tensor::u32(vec![n, joint], axis.repeat(n)));
        }
        Ok(self.model.run(&inputs)?[0].floats()?.to_vec())
    }
}

/// Where the flow loop is, for a [`Hook`].
pub struct Step {
    pub index: usize,
    pub steps: usize,
    pub sigma: f32,
    /// The time this step moves to, which `after` may change.
    pub next: f32,
    /// Latent tokens a side.
    pub side: usize,
    /// Values per latent token.
    pub channels: usize,
}

/// Code that runs around each velocity call.
pub trait Hook {
    fn before(&mut self, step: &Step, latents: &mut [f32]) -> Result<(), Error>;
    fn after(
        &mut self,
        step: &mut Step,
        latents: &mut [f32],
        velocity: &mut [f32],
    ) -> Result<(), Error>;
}

/// Euler steps from `x` over `sigmas`, through `velocity`.
pub(crate) fn euler(
    x: &mut [f32],
    sigmas: &mut [f32],
    side: usize,
    mut hook: Option<&mut dyn Hook>,
    mut velocity: impl FnMut(&[f32], f32) -> Result<Vec<f32>, Error>,
) -> Result<(), Error> {
    let steps = sigmas.len() - 1;
    let channels = x.len() / (side * side).max(1);
    for index in 0..steps {
        let (sigma, next) = (sigmas[index], sigmas[index + 1]);
        let mut step = Step {
            index,
            steps,
            sigma,
            next,
            side,
            channels,
        };
        if let Some(hook) = hook.as_deref_mut() {
            hook.before(&step, x)?;
        }
        let mut v = velocity(x, sigma)?;
        if let Some(hook) = hook.as_deref_mut() {
            hook.after(&mut step, x, &mut v)?;
            if !step.next.is_finite() {
                return Err(format!("step {index}: the hook set time {}", step.next).into());
            }
            sigmas[index + 1] = step.next;
        }
        x.iter_mut()
            .zip(&v)
            .for_each(|(x, v)| *x += (sigmas[index + 1] - sigma) * v);
    }
    Ok(())
}

/// Moves masked tokens on a straight path to known latents.
pub struct Hold {
    /// Whether each latent token is held.
    pub mask: Vec<bool>,
    /// Latents the held tokens end at.
    pub known: Vec<f32>,
    /// The first time and latents.
    start: (f32, Vec<f32>),
}

impl Hold {
    pub fn new(mask: Vec<bool>, known: Vec<f32>) -> Self {
        let start = (1.0, Vec::new());
        Self { mask, known, start }
    }
}

impl Hook for Hold {
    fn before(&mut self, step: &Step, latents: &mut [f32]) -> Result<(), Error> {
        if step.index == 0 {
            self.start = (step.sigma, latents.to_vec());
        }
        Ok(())
    }

    fn after(&mut self, step: &mut Step, _: &mut [f32], v: &mut [f32]) -> Result<(), Error> {
        let (sigma, start) = (self.start.0, &self.start.1);
        for (token, _) in self.mask.iter().enumerate().filter(|m| *m.1) {
            for i in token * step.channels..(token + 1) * step.channels {
                v[i] = (start[i] - self.known[i]) / sigma;
            }
        }
        Ok(())
    }
}

/// A script's `step` function as a [`Hook`].
pub struct Scripted(tessera_script::Script);

impl Scripted {
    /// The script at `path`.
    pub fn open(path: &Path) -> Result<Self, Error> {
        let script = tessera_script::Script::open(path, Default::default())?;
        let named = Self::new(script);
        named.map_err(|e| format!("{}: {e}", path.display()).into())
    }

    /// A loaded script. It must export `step`.
    pub fn new(script: tessera_script::Script) -> Result<Self, String> {
        match script.has("step") {
            true => Ok(Self(script)),
            false => Err("exports no function `step`".into()),
        }
    }

    /// Calls `step`, and returns what it gave back.
    fn call(&mut self, phase: &str, step: &Step, arrays: &mut [Array]) -> Result<Value, Error> {
        let context = serde_json::json!({
            "phase": phase,
            "index": step.index,
            "steps": step.steps,
            "sigma": step.sigma,
            "next": step.next,
            "side": step.side,
            "channels": step.channels,
        });
        Ok(self.0.call("step", &context, arrays)?)
    }
}

impl Hook for Scripted {
    fn before(&mut self, step: &Step, latents: &mut [f32]) -> Result<(), Error> {
        // Each picture starts with fresh script state.
        if step.index == 0 {
            self.0.reset()?;
        }
        self.call("before", step, &mut [("latents", latents)])?;
        Ok(())
    }

    fn after(&mut self, step: &mut Step, x: &mut [f32], v: &mut [f32]) -> Result<(), Error> {
        let back = self.call("after", step, &mut [("latents", x), ("velocity", v)])?;
        if let Some(next) = back.get("next").and_then(Value::as_f64) {
            step.next = next as f32;
        }
        Ok(())
    }
}
