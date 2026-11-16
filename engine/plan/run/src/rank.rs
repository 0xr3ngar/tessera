//! One process's rank of a plan, over a transport.

use crate::Error;
use crate::Run;
use crate::open;
use engine_comm::reduce::{CUDA, ENTRY};
use engine_comm::{Reduce, Transport, summed};
use engine_settings::Settings;
use lower::{Arg, Plan, Shape, Step, Weight};
use mem::{Tensor, View};

/// One rank: its runner and its way to the others.
pub struct Rank<T: Transport> {
    runner: Box<dyn Run>,
    transport: T,
    /// The device's rounds, when it runs them itself.
    reduce: Option<Reduce>,
}

impl<T: Transport> Rank<T> {
    /// A runner for the transport's rank of `plan`.
    pub fn open(plan: Plan, settings: &Settings, transport: T) -> Result<Self, Error> {
        let fits = plan.ranks.max(1) == transport.ranks() && plan.rank == transport.rank();
        if !fits {
            return Err(Error::RankMismatch {
                plan: plan.rank,
                plans: plan.ranks,
                rank: transport.rank(),
                ranks: transport.ranks(),
            });
        }
        let runner = open(plan, settings)?;
        let mut transport = transport;
        let reduce = match runner.fixed() {
            true => transport.device(),
            false => None,
        };
        Ok(Self {
            runner,
            transport,
            reduce,
        })
    }

    /// Whether sums run on the device.
    pub fn on_device(&self) -> bool {
        self.reduce.is_some()
    }

    /// One sum on the device: copy, publish, wait and add.
    fn reduce(&mut self, step: &Step) -> Result<(), Error> {
        let r = self.reduce.as_mut().expect("a device");
        let n = step.write.len;
        if 16 + 4 * n > r.capacity {
            return Err(Error::SumTooLong(n));
        }
        let seq = r.round();
        let args = [
            Arg::value(&r.send),
            Arg::value(&r.recv),
            Arg::value(&r.flags),
            Arg::value(&r.ctl),
            step.read(0),
            Arg::at(step.write, 0),
            Arg::value(&(n as u32)),
            Arg::value(&r.rank),
            Arg::value(&r.ranks),
            Arg::value(&r.lanes),
            Arg::value(&seq),
            Arg::value(&(r.capacity as u64)),
        ];
        let blocks = (n / 1024).clamp(1, 8);
        let launched = self
            .runner
            .launch_fixed(CUDA, ENTRY, &args, Shape::Line(blocks * 256));
        if let Some(why) = self.reduce.as_ref().and_then(Reduce::failure) {
            return Err(engine_comm::Error::Link(why).into());
        }
        launched
    }

    pub fn runner(&self) -> &dyn Run {
        &*self.runner
    }

    /// Takes this rank's slice of the whole weight.
    pub fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
        self.runner.load(weight, data)
    }

    /// Takes this rank's part of the weight as stored.
    pub fn load_weight(&mut self, weight: u32, w: &Weight) -> Result<(), Error> {
        self.runner.load_weight(weight, w)
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
            if self.reduce.is_some() {
                self.reduce(step)?;
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
