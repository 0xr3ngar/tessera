//! Replays a plan on any GPU over the device traits.

use crate::kernels::Kernels;
use crate::run::{Base, Run};
use codegen::Dialect;
use gpukit::backend::Backend;
use gpukit::device::{Device, DeviceBuffer, cached_context, cached_source};
use gpukit::launch::{KernelArgs, Launch, launch_1d, launch_grid};
use graph::DimName;
use lower::{Arg, Error, Extent, Launch as Call, Launching, Plan, Shape, Slot};
use mem::Buffer;
use std::collections::HashMap;
use std::ops::Range;
use std::time::Instant;

pub(crate) struct Runner<D: Device> {
    ctx: &'static D,
    base: Base,
    kernels: Vec<D::Kernel>,
    compiled: Kernels<D::Kernel>,
    /// The whole plan's launches per binding, recorded once.
    graphs: HashMap<Vec<(DimName, u32)>, D::Graph>,
    launching: Launching,
    /// Whether the device overlaps dependent launches at all.
    overlaps: bool,
    /// Which steps' kernels wait on the one before them.
    waits: Vec<bool>,
}

/// The kernel text a backend compiles.
fn dialect(backend: Backend) -> Dialect {
    match backend {
        Backend::Metal => Dialect::Metal,
        Backend::Cuda | Backend::Rocm => Dialect::Cuda,
    }
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Runner<D> {
    pub fn new(plan: Plan) -> Result<Self, Error> {
        let ctx = cached_context::<D>()?;
        let mut runner = Self {
            ctx,
            base: Base::default(),
            kernels: Vec::new(),
            compiled: Kernels::default(),
            graphs: HashMap::new(),
            launching: Launching::default(),
            // Asked once: asking may launch, and captures come later.
            overlaps: ctx.platform().limits.pdl,
            waits: Vec::new(),
        };
        runner.replan(&plan)?;
        Ok(runner)
    }

    fn buffer(&self, slot: Slot) -> &DeviceBuffer<D> {
        let backing = self.base.held.get(slot).backing().as_any();
        backing
            .downcast_ref::<DeviceBuffer<D>>()
            .expect("this runner allocates device buffers")
    }

    /// Passes each argument by position, then enqueues.
    fn launch(&self, step: usize, launch: &Call) -> Result<(), Error> {
        let kernel = &self.kernels[step];
        let overlap = self.launching.pdl && self.overlaps && self.waits[step];
        self.ctx.programmatic(overlap);
        let mut args = KernelArgs::new();
        for arg in &launch.args {
            match arg {
                Arg::Buffer(slot, Extent::Fixed(skip)) => {
                    let at = slot.at(*skip) as u64;
                    args.device_ptr(self.buffer(*slot).device_ptr() + at)
                }
                Arg::Value(bytes) => args.bytes(bytes),
                Arg::Size(_) | Arg::Buffer(..) => {
                    unreachable!("bound before launching")
                }
            };
        }
        match launch.shape {
            Shape::Line(count) => launch_1d(self.ctx, kernel, count, &mut args)?,
            Shape::Grid {
                across,
                down,
                threads,
            } => {
                let threads = threads as u32;
                launch_grid(self.ctx, kernel, across, down, threads, &mut args)?
            }
            Shape::Runtime(_) => unreachable!("bound before launching"),
        }
        Ok(())
    }

    /// Every launch of `steps`, in order.
    fn launch_all(&self, steps: Range<usize>) -> Result<(), Error> {
        for i in steps {
            for launch in self.base.launches(i)? {
                self.launch(i, &launch)?;
            }
        }
        Ok(())
    }
}

impl<D: Device + Launch<Kernel = <D as Device>::Kernel>> Run for Runner<D> {
    fn base(&self) -> &Base {
        &self.base
    }

    fn base_mut(&mut self) -> &mut Base {
        &mut self.base
    }

    fn replan(&mut self, plan: &Plan) -> Result<(), Error> {
        if !self.base.fresh(plan)? {
            return Ok(());
        }
        // Buffers may move, so recorded launches go stale.
        self.graphs.clear();
        let ctx = self.ctx;
        self.base.held.bind(plan, |zone, bytes| {
            let buffer = DeviceBuffer::alloc(ctx, bytes.max(4))?;
            // Kernels may read what they are about to write.
            buffer.zero()?;
            Ok(Buffer::new(zone, buffer))
        })?;
        self.kernels.clear();
        self.waits.clear();
        for step in &plan.steps {
            let fixed = |text, entry| -> Result<_, Error> { Ok(cached_source::<D>(text, entry)?) };
            let made =
                |source: &str, entry: &str| -> Result<_, Error> { Ok(ctx.compile(source, entry)?) };
            // Generated kernels wait on `griddepcontrol`. Fixed ones do not.
            let (kernel, generated) = self.compiled.get(step, dialect(D::BACKEND), fixed, made)?;
            self.kernels.push(kernel);
            self.waits.push(generated);
        }
        self.base.adopt(plan);
        Ok(())
    }

    fn launching(&mut self, launching: Launching) {
        self.launching = launching;
        self.graphs.clear();
    }

    fn repeat(&mut self, step: usize, times: usize) -> Result<f64, Error> {
        self.ctx.finish()?;
        let start = Instant::now();
        let launches = self.base.launches(step)?;
        for _ in 0..times {
            for launch in &launches {
                self.launch(step, launch)?;
            }
        }
        let clocked = self.ctx.finish()?;
        let seconds = clocked.unwrap_or_else(|| start.elapsed().as_secs_f64());
        Ok(seconds / times as f64)
    }

    fn device(&self) -> String {
        format!("{} {}", D::BACKEND, self.ctx.name())
    }

    fn platform(&self) -> gpukit::Platform {
        self.ctx.platform()
    }

    fn precision(&self) -> codegen::Precision {
        crate::precision(D::BACKEND)
    }

    /// Launches every step, as a recorded graph when whole.
    fn submit(&mut self, steps: Range<usize>) -> Result<(), Error> {
        let whole = self.launching.graphs
            && self.ctx.graphs()
            && self.base.plan.id != 0
            && steps == (0..self.base.plan.steps.len());
        if !whole {
            self.launch_all(steps)?;
            self.ctx.finish()?;
            return Ok(());
        }
        let mut key: Vec<(DimName, u32)> = self.base.bound.iter().map(|(k, v)| (*k, *v)).collect();
        key.sort_unstable();
        if !self.graphs.contains_key(&key) {
            let graph = self.ctx.capture(|| self.launch_all(steps.clone()))?;
            self.graphs.insert(key.clone(), graph);
        }
        self.ctx.replay(&self.graphs[&key])?;
        self.ctx.finish()?;
        Ok(())
    }
}
