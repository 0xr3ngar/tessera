//! Every rank in one process, summing through host memory.

use crate::Run;
use crate::open;
use engine_settings::Settings;
use lower::{Error, Plan};
use mem::{Tensor, View};

/// One runner per rank. Sums run in rank order.
pub struct Loopback {
    ranks: Vec<Box<dyn Run>>,
}

impl Loopback {
    /// A runner for each of the plan's ranks.
    pub fn open(plan: &Plan, settings: &Settings) -> Result<Self, Error> {
        let mut ranks = Vec::with_capacity(plan.ranks.max(1) as usize);
        for rank in 0..plan.ranks.max(1) {
            let mut runner = open(plan.clone(), settings)?;
            runner.base_mut().rank = rank;
            ranks.push(runner);
        }
        Ok(Self { ranks })
    }

    pub fn rank(&self, r: usize) -> &dyn Run {
        &*self.ranks[r]
    }

    /// Every rank takes its slice of the whole weight.
    pub fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
        for r in &mut self.ranks {
            r.load(weight, data.clone())?;
        }
        Ok(())
    }

    /// Runs every step on every rank. Sums meet between steps.
    pub fn run(&mut self, inputs: &[View]) -> Result<Vec<Tensor>, Error> {
        for r in &mut self.ranks {
            r.feed(inputs)?;
        }
        let steps = self.ranks[0].plan().steps.clone();
        for (i, step) in steps.iter().enumerate() {
            if step.kernel.name() != "all_reduce" {
                for r in &mut self.ranks {
                    r.submit(i..i + 1)?;
                }
                continue;
            }
            let mut sum: Option<Vec<f32>> = None;
            for r in &self.ranks {
                let part = r.read(step.reads[0])?;
                let part = part.floats()?;
                sum = Some(match sum {
                    None => part.to_vec(),
                    Some(s) => s.iter().zip(part).map(|(a, b)| a + b).collect(),
                });
            }
            let sum = Tensor::f32(vec![step.write.len], sum.expect("a rank"));
            for r in &mut self.ranks {
                r.write(step.write, sum.view())?;
            }
        }
        let outputs = self.ranks[0].plan().outputs.clone();
        let mut out = Vec::with_capacity(outputs.len());
        for o in &outputs {
            let first = self.ranks[0].read(o.slot)?;
            for r in &self.ranks[1..] {
                if r.read(o.slot)? != first {
                    return Err(Error::Unsupported("ranks disagree on an output".into()));
                }
            }
            out.push(first.reshaped(o.shape.clone()));
        }
        Ok(out)
    }
}
