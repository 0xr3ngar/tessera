//! One process's rank of a plan, over a transport.

use crate::Run;
use crate::open;
use engine_comm::{Transport, summed};
use engine_settings::Settings;
use lower::{Error, Plan};
use mem::{Tensor, View};

/// One rank: its runner and its way to the others.
pub struct Rank<T: Transport> {
    runner: Box<dyn Run>,
    transport: T,
}

impl<T: Transport> Rank<T> {
    /// A runner for the transport's rank of `plan`.
    pub fn open(plan: Plan, settings: &Settings, transport: T) -> Result<Self, Error> {
        let fits = plan.ranks.max(1) == transport.ranks() && plan.rank == transport.rank();
        if !fits {
            let what = format!(
                "rank {} of {} on rank {} of {}",
                plan.rank,
                plan.ranks,
                transport.rank(),
                transport.ranks()
            );
            return Err(Error::Unsupported(what));
        }
        let runner = open(plan, settings)?;
        Ok(Self { runner, transport })
    }

    pub fn runner(&self) -> &dyn Run {
        &*self.runner
    }

    /// Takes this rank's slice of the whole weight.
    pub fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
        self.runner.load(weight, data)
    }

    /// Runs every step. Sums wait for every rank.
    pub fn run(&mut self, inputs: &[View]) -> Result<Vec<Tensor>, Error> {
        self.runner.feed(inputs)?;
        let steps = self.runner.plan().steps.clone();
        for (i, step) in steps.iter().enumerate() {
            if step.kernel.name() != "all_reduce" {
                self.runner.submit(i..i + 1)?;
                continue;
            }
            let part = self.runner.read(step.reads[0])?;
            let parts = self.transport.exchange(part.bytes())?;
            let floats = |p: &Vec<u8>| Tensor::from_bytes(vec![p.len() / 4], mem::DType::F32, p);
            let parts: Vec<Tensor> = parts.iter().map(floats).collect();
            let sum = summed(parts.iter().map(|p| p.floats().expect("floats")));
            let sum = Tensor::f32(vec![step.write.len], sum);
            self.runner.write(step.write, sum.view())?;
        }
        let outputs = self.runner.plan().outputs.clone();
        let read = |o: &lower::Output| Ok(self.runner.read(o.slot)?.reshaped(o.shape.clone()));
        outputs.iter().map(read).collect()
    }
}
