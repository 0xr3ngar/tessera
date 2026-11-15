//! Whole-sequence attention on the host, a thread per head.

use crate::{GUESS, Out, dot, floats, parallel};
use graph::Mask;
use lower::kernels::Map;
use lower::{Attn, Draft, Kernel, Native, T, Tree, ZoneKind, sites};

/// Softmax attention over `[seqs, rows, heads, width]` tensors.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_attend",
    sizes = self.sizes.sizes(),
    work = self.sizes.seqs * self.sizes.heads * self.sizes.queries * self.sizes.keys.most() * self.sizes.width,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    native,
)]
pub struct Attend {
    pub scale: f32,
    pub sizes: Attn,
}

/// The keys `[from, to)` a query at `position` sees.
fn seen(mask: Mask, position: usize, keys: usize) -> (usize, usize) {
    match mask {
        Mask::Causal => (0, (position + 1).min(keys)),
        Mask::Bidirectional => (0, keys),
        Mask::Within(n) => {
            let n = n as usize;
            (position.saturating_sub(n), (position + n + 1).min(keys))
        }
    }
}

impl Native for Attend {
    fn run(&self, reads: &[&[u8]], write: &mut [u8]) -> Result<(), String> {
        let (q, k, v) = (floats(reads[0]), floats(reads[1]), floats(reads[2]));
        let out = Out::new(write);
        let a = &self.sizes;
        let (queries, keys, width) = (a.queries, a.keys.most(), a.width);
        let (heads, kv_heads) = (a.heads, a.kv_heads);
        let group = heads / kv_heads;
        // Query and key ends align, as in decode.
        let past = keys.saturating_sub(queries);
        parallel(a.seqs * heads, |jobs| {
            let mut scores = vec![0.0f32; keys];
            for job in jobs {
                let (s, h) = (job / heads, job % heads);
                for t in 0..queries {
                    let qo = ((s * queries + t) * heads + h) * width;
                    let qr = &q[qo..qo + width];
                    let (from, to) = seen(a.mask, past + t, keys);
                    let key_at = |j: usize| ((s * keys + j) * kv_heads + h / group) * width;
                    let mut most = f32::NEG_INFINITY;
                    for j in from..to {
                        scores[j] = dot(qr, &k[key_at(j)..key_at(j) + width]) * self.scale;
                        most = most.max(scores[j]);
                    }
                    let mut total = 0.0;
                    for s in &mut scores[from..to] {
                        *s = (*s - most).exp();
                        total += *s;
                    }
                    let o = out.part(qo, width);
                    o.fill(0.0);
                    for j in from..to {
                        let p = scores[j] / total;
                        let vr = &v[key_at(j)..key_at(j) + width];
                        for (oi, vi) in o.iter_mut().zip(vr) {
                            *oi += p * vi;
                        }
                    }
                }
            }
        });
        Ok(())
    }
}

/// Whole sequences with fixed keys. Ragged rows go to trees.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Attend = tree.site(at)?;
    if site.rows.is_some() || site.sizes.keys.runtime() {
        return None;
    }
    let out = site.sizes.seqs * site.out();
    let kernel = Attend {
        scale: site.scale,
        sizes: site.sizes,
    };
    Some(vec![Draft::new(kernel, [site.q, site.k, site.v], out)])
}

lower::rule!("cpu_attend", propose);
