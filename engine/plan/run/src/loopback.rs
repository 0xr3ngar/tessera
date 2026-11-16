//! Every rank in one process, summing through host memory.

use crate::Run;
use crate::open;
use engine_comm::summed;
use engine_settings::Settings;
use lower::{Error, Plan, Weight};
use mem::{Tensor, View};

/// One runner per rank. Sums run in rank order.
pub struct Loopback {
    ranks: Vec<Box<dyn Run>>,
}

impl Loopback {
    /// A runner per plan, one per rank in order.
    pub fn open(plans: &[Plan], settings: &Settings) -> Result<Self, Error> {
        let mut ranks = Vec::with_capacity(plans.len());
        for (rank, plan) in plans.iter().enumerate() {
            if plan.rank as usize != rank || plan.ranks.max(1) as usize != plans.len() {
                return Err(Error::Unsupported("plans out of rank order".into()));
            }
            ranks.push(open(plan.clone(), settings)?);
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

    /// Every rank takes its part of the weight as stored.
    pub fn load_weight(&mut self, weight: u32, w: &Weight) -> Result<(), Error> {
        for r in &mut self.ranks {
            r.load_weight(weight, w)?;
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
            let mut parts = Vec::with_capacity(self.ranks.len());
            for r in &self.ranks {
                parts.push(r.read(step.reads[0])?);
            }
            let sum = summed(parts.iter().map(|p| p.floats().expect("floats")));
            let sum = Tensor::f32(vec![step.write.len], sum);
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
