//! `x[m, k]` times `w[n, k]` transposed, dense or picked.

use crate::{GUESS, source};
use codegen::ir::{Body, Expr, Intrinsic, Ty};
use codegen::prelude::*;
use lower::kernels::finish;
use lower::sites::{After, Picked as Pick};
use lower::{Draft, Form as Stored, Kernel, Launch, Shape, Step, T, Tree, ZoneKind, sites};

/// One thread per output: its row dotted with its column.
#[kernel]
fn cpu_dense(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    out: BufMut<F32>,
    m: u32,
    n: u32,
    k: u32,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= m * n {
        return;
    }
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    let row = (gid() / n) as u64 * k as u64;
    let col = (gid() % n) as u64 * k as u64;
    sums[0] += x.dot(row, w, col, k);
}

/// Which matrix each row reads, and where this rank's start.
#[derive(Group)]
struct Picking {
    count: u32,
    stride: u32,
    offset: u32,
    repeat: u32,
    first: u32,
}

/// One thread per output, along the matrix its row picks.
#[kernel]
fn cpu_picked(
    x: Buf<F32>,
    #[form(form)] w: Buf<Dyn>,
    ids: Buf<U32>,
    out: BufMut<F32>,
    rows: u32,
    n: u32,
    k: u32,
    pick: Picking,
    #[build] form: &'static dyn Stored,
) {
    if gid() >= rows * n {
        return;
    }
    let row = gid() / n;
    let col = gid() % n;
    let id = ids[row as u64];
    let mut sums = [0.0f32; 1];
    sums[0] = 0.0;
    if id >= pick.first && id - pick.first < pick.count {
        let matrix = (id - pick.first) as u64 * pick.stride as u64 + pick.offset as u64;
        let line = (matrix + col as u64) * k as u64;
        sums[0] += x.dot((row / pick.repeat) as u64 * k as u64, w, line, k);
    }
}

/// The stages after the sum, into the thread's output.
fn finished(mut body: Body, after: &After, operands_at: usize) -> Body {
    lower::kernels::add_operands(&mut body, operands_at, after.others());
    let at = Expr::cast(Ty::U64, Expr::Call(Intrinsic::Gid, Vec::new()));
    let sum = Expr::index(Expr::var("sums"), Expr::Int(0));
    body.stmts.extend(finish(after, sum, at));
    body
}

/// Weights read as stored. Rows and operands come as f32.
fn accepts(input: usize, form: &dyn Stored) -> bool {
    input == 1 || !form.narrow()
}

/// A matmul and its finishing stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_dense",
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_dense", finished(cpu_dense_ir(self.form), &self.after, 2)),
    launches = dense_launches,
)]
pub struct Dense {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
}

fn dense_launches(k: &Dense, step: &Step) -> Vec<Launch> {
    vec![Launch::plain(
        step,
        &[k.m, k.n, k.k],
        Shape::Line(k.m * k.n),
    )]
}

/// Each row against its id's matrix, then the stages.
#[derive(Clone, Debug, Kernel)]
#[launch(
    crate = lower,
    name = "cpu_picked",
    sizes = [self.m, self.n, self.k],
    work = self.m * self.n * self.k,
    guess = GUESS,
    runs_on = ZoneKind::Cpu,
    accepts = accepts(input, form),
    source = source("cpu_picked", finished(cpu_picked_ir(self.form), &self.after, 3)),
    launches = picked_launches,
)]
pub struct Picked {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Stored,
    pub after: After,
    pub pick: Pick,
}

fn picked_launches(k: &Picked, step: &Step) -> Vec<Launch> {
    let p = k.pick;
    let counts = [
        k.m,
        k.n,
        k.k,
        p.count,
        p.stride,
        p.offset,
        p.repeat,
        p.first.unwrap_or(0),
    ];
    vec![Launch::plain(step, &counts, Shape::Line(k.m * k.n))]
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
    let after = site.after.clone();
    let kernel = match picked {
        None => Draft::new(
            Dense {
                m,
                n,
                k,
                form,
                after,
            },
            site.reads(),
            m * n,
        ),
        Some(pick) => Draft::new(
            Picked {
                m,
                n,
                k,
                form,
                after,
                pick,
            },
            site.reads(),
            m * n,
        ),
    };
    Some(vec![kernel])
}

lower::rule!("cpu_dense", propose);
