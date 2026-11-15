//! Native CPU kernels for matmuls and attention, as lowering rules.

mod attend;
mod dense;
mod rows;

pub use attend::Attend;
pub use dense::Dense;

use lower::kernels::MapStage;
use lower::sites::After;
use std::thread;

/// Guessed seconds per unit of work, under lockstep trees.
const GUESS: f64 = 1e-12;

/// Bytes as the f32 they hold.
fn floats(bytes: &[u8]) -> &[f32] {
    let (head, body, tail) = unsafe { bytes.align_to::<f32>() };
    assert!(head.is_empty() && tail.is_empty(), "misaligned floats");
    body
}

/// An output threads write disjoint parts of.
struct Out(*mut f32, usize);

// Each thread writes its own elements, never another's.
unsafe impl Sync for Out {}

impl Out {
    fn new(bytes: &mut [u8]) -> Self {
        let (head, body, tail) = unsafe { bytes.align_to_mut::<f32>() };
        assert!(head.is_empty() && tail.is_empty(), "misaligned floats");
        Self(body.as_mut_ptr(), body.len())
    }

    /// Elements `start..start + len`, this thread's alone.
    #[allow(clippy::mut_from_ref, reason = "threads own disjoint parts")]
    fn part(&self, start: usize, len: usize) -> &mut [f32] {
        assert!(start + len <= self.1, "an output past its end");
        unsafe { std::slice::from_raw_parts_mut(self.0.add(start), len) }
    }
}

fn ids(bytes: &[u8]) -> &[u32] {
    let (head, body, tail) = unsafe { bytes.align_to::<u32>() };
    assert!(head.is_empty() && tail.is_empty(), "misaligned ids");
    body
}

/// Runs `each` over `0..n` in chunks, one thread per core.
fn parallel(n: usize, each: impl Fn(std::ops::Range<usize>) + Sync) {
    let cores = thread::available_parallelism().map_or(1, |c| c.get());
    let chunk = n.div_ceil(cores).max(1);
    if n <= chunk {
        return each(0..n);
    }
    thread::scope(|s| {
        for start in (0..n).step_by(chunk) {
            let each = &each;
            s.spawn(move || each(start..(start + chunk).min(n)));
        }
    });
}

/// `sum` through the stages after a matmul, at output `at`.
fn finished(after: &After, operands: &[&[f32]], at: usize, mut sum: f32) -> f32 {
    let mut rest = operands.iter();
    for stage in &after.stages {
        sum = match stage {
            MapStage::Add => sum + rest.next().expect("an operand")[at],
            MapStage::Mul => sum * rest.next().expect("an operand")[at],
            MapStage::Scale(by) => sum * by,
            MapStage::Act(a) => rows::act(*a, sum),
        };
    }
    sum
}

/// Eight running sums, so the loop vectorizes.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0f32; 8];
    let (whole, rest) = (a.len() / 8 * 8, a.len() % 8);
    let (xs, _) = a[..whole].as_chunks::<8>();
    let (ys, _) = b[..whole].as_chunks::<8>();
    for (x, y) in xs.iter().zip(ys) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut sum = acc.iter().sum::<f32>();
    for i in 0..rest {
        sum += a[whole + i] * b[whole + i];
    }
    sum
}

#[cfg(test)]
mod cpu_test;
