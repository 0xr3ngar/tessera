//! `x[m, k]` times `w[n, k]` transposed, dense or picked.

use crate::rows::weight_row;
use crate::{GUESS, Out, dot, finished, floats, ids, parallel};
use lower::kernels::Map;
use lower::sites::{After, Picked};
use lower::{Draft, Form, Kernel, Native, T, Tree, ZoneKind, sites};

/// A matmul and its finishing stages, on the host.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_dense",
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    accepts = self.accepts(input, form),
    source = Map::plain(Vec::new()).source(),
    launches = Vec::new(),
    native,
)]
pub struct Dense {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Form,
    pub after: After,
    /// Set when each row reads its own matrix.
    pub pick: Option<Pick>,
}

/// Which matrix each row reads, as the site has it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pick {
    pub count: usize,
    pub stride: usize,
    pub offset: usize,
    pub repeat: usize,
    pub first: usize,
}

impl From<Picked> for Pick {
    fn from(p: Picked) -> Self {
        Self {
            count: p.count,
            stride: p.stride,
            offset: p.offset,
            repeat: p.repeat,
            first: p.first.unwrap_or(0),
        }
    }
}

impl Dense {
    /// Weights read as stored. Rows and operands come as f32.
    fn accepts(&self, input: usize, form: &dyn Form) -> bool {
        input == 1 || !form.narrow()
    }
}

impl Native for Dense {
    fn run(&self, reads: &[&[u8]], write: &mut [u8]) -> Result<(), String> {
        let (x, w) = (floats(reads[0]), reads[1]);
        let picked = self.pick.map(|p| (p, ids(reads[2])));
        let skip = 2 + usize::from(picked.is_some());
        let operands: Vec<&[f32]> = reads[skip..].iter().map(|r| floats(r)).collect();
        let out = Out::new(write);
        let (m, n, k) = (self.m, self.n, self.k);
        match picked {
            None => parallel(n, |cols| {
                let mut row = vec![0.0f32; k];
                for col in cols {
                    weight_row(self.form, w, col, k, &mut row);
                    for r in 0..m {
                        let sum = dot(&x[r * k..(r + 1) * k], &row);
                        let at = r * n + col;
                        out.part(at, 1)[0] = finished(&self.after, &operands, at, sum);
                    }
                }
            }),
            Some((p, ids)) => parallel(m, |rows| {
                let mut row = vec![0.0f32; k];
                for r in rows {
                    let line_out = out.part(r * n, n);
                    let id = ids[r] as usize;
                    let held = id >= p.first && id - p.first < p.count;
                    let xr = &x[(r / p.repeat) * k..(r / p.repeat + 1) * k];
                    for (col, o) in line_out.iter_mut().enumerate() {
                        let mut sum = 0.0;
                        if held {
                            let line = (id - p.first) * p.stride + p.offset + col;
                            weight_row(self.form, w, line, k, &mut row);
                            sum = dot(xr, &row);
                        }
                        *o = finished(&self.after, &operands, r * n + col, sum);
                    }
                }
            }),
        }
        Ok(())
    }
}

/// Any finished matmul, dense or picked.
fn propose(tree: &Tree, at: T) -> Option<Vec<Draft>> {
    let site: sites::Finished = tree.site(at)?;
    let sites::Matmul {
        m,
        n,
        k,
        form,
        picked,
        ..
    } = site.product;
    let kernel = Dense {
        m,
        n,
        k,
        form,
        after: site.after.clone(),
        pick: picked.map(Pick::from),
    };
    Some(vec![Draft::new(kernel, site.reads(), m * n)])
}

lower::rule!("cpu_dense", propose);
