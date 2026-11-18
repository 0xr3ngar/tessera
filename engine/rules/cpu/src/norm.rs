//! RMS norm, a thread per row in small groups.

use sensai_codegen::prelude::*;
use sensai_lower::{Draft, Kernel, Launch, Step, T, Tree, ZoneKind, sites};
use sensai_lower::{GUESS, small, source};

/// The squares in one sum, then each element scaled.
#[kernel]
fn cpu_rms_norm(
    x: Buf<F32>,
    w: Buf<F32>,
    mut out: BufMut<F32>,
    rows: u32,
    width: u32,
    eps: f32,
    #[build] weighted: u32,
) {
    if gid() >= rows {
        return;
    }
    let at = gid() as u64 * width as u64;
    let inv = rsqrt(x.dot(at, x, at, width) / width as f32 + eps);
    for i in 0..width {
        let gain = if weighted == 1 { w[i as u64] } else { 1.0 };
        out[at + i as u64] = x[at + i as u64] * inv * gain;
    }
}

/// Rows scaled by their root mean square and gains.
#[derive(Clone, Copy, Debug, Kernel)]
#[launch(
    crate = sensai_lower,
    name = "cpu_rms_norm",
    sizes = [self.rows, self.width],
    work = self.rows * self.width,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    source = source("cpu_rms_norm", cpu_rms_norm_ir(u32::from(self.weighted))),
    launches = norm_launches,
)]
pub struct RmsNorm {
    pub rows: usize,
    pub width: usize,
    pub eps: f32,
    pub weighted: bool,
}

fn norm_launches(k: &RmsNorm, step: &Step) -> Vec<Launch> {
    let mut launch = Launch::plain(step, &[k.rows, k.width], small(k.rows));
    launch.args.push(sensai_lower::Arg::value(&k.eps));
    vec![launch]
}

fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Stage = tree.site(at)?;
    let (eps, weighted) = site.rms()?;
    let kernel = RmsNorm {
        rows: site.len / site.width,
        width: site.width,
        eps,
        weighted,
    };
    Some(vec![Draft::new(kernel, site.padded(), site.len)])
}

sensai_lower::rule!("cpu_rms_norm", propose);
