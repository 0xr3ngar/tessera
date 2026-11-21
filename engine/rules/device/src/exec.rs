//! Executes resident `tessera::table*` plan steps through `Resident`.

use crate::layout::{ARENAS, Form, Held, Item, Kept, LlmStep, Ref};
use crate::proxy::Op;
use crate::resident::{
    ConventionsPlan, Error, KvPlan, LanePlan, LocalLink, Plan, PoolPlan, Resident, USER,
};
use std::collections::HashMap;
use std::time::Duration;
use tessera_engine_comm::Link;
use tessera_gpukit::backend::Backend as Gpu;
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_gpukit::launch::Launch;
use tessera_graph::DimName;
use tessera_lower::{Collective, ItemData, LlmSpec, ResidentSpec, Slot, Source, Step};
use tessera_run::{Error as RunError, ResidentCall, ResidentExec};

/// Maximum user arenas available in `tessera::table.arenas`.
pub(crate) const USER_ARENAS: usize = ARENAS - USER as usize;
const TIMEOUT: Duration = Duration::from_secs(120);

/// Packed resident plan and total scratch bytes for one step.
#[derive(Clone, Debug)]
pub struct Packed {
    pub plan: Plan,
    pub scratch_bytes: usize,
    pub kv_lens: Vec<Vec<u32>>,
}

/// Maps plan buffers into deduplicated user arenas (`USER..ARENAS`).
struct ArenaPack {
    arenas: Vec<u64>,
}

impl ArenaPack {
    fn new() -> Self {
        Self { arenas: Vec::new() }
    }

    fn arena(&mut self, base: u64) -> Result<u32, Error> {
        if let Some(idx) = self.arenas.iter().position(|&a| a == base) {
            return Ok(USER + idx as u32);
        }
        if self.arenas.len() >= USER_ARENAS {
            return Err(Error::Plan(format!(
                "step needs more than {USER_ARENAS} unique user arenas \
                 (pack contiguous buffers into shared arenas)"
            )));
        }
        let idx = self.arenas.len();
        self.arenas.push(base);
        Ok(USER + idx as u32)
    }

    fn slot(&mut self, buffers: &[u64], slot: Slot) -> Result<Ref, Error> {
        let base = *buffers
            .get(slot.buffer as usize)
            .ok_or_else(|| Error::Plan(format!("missing buffer {}", slot.buffer)))?;
        Ok(Ref::at(self.arena(base)?, slot.start() as u64))
    }

    fn opt(&mut self, buffers: &[u64], slot: Option<Slot>) -> Result<Ref, Error> {
        match slot {
            Some(slot) => self.slot(buffers, slot),
            None => Ok(Ref::NONE),
        }
    }
}

fn collective_op(op: Collective) -> Op {
    match op {
        Collective::Sum => Op::Sum,
        Collective::Gather => Op::Gather,
        Collective::AllToAll => Op::AllToAll,
        Collective::Send => Op::Send,
        Collective::Recv => Op::Recv,
        Collective::Fetch => Op::Fetch,
    }
}

/// Computes total 16-byte aligned scratch bytes for `spec`.
pub(crate) fn scratch_size(spec: &ResidentSpec) -> usize {
    let mut total = 0usize;
    for item in &spec.items {
        if let ItemData::Scratch(bytes) = item.data {
            total = total.next_multiple_of(16) + bytes;
        }
    }
    total
}

/// Builds a `resident::Plan` from one lowered resident step.
pub(crate) fn pack_plan(
    call: &ResidentCall<'_>,
    spec: &ResidentSpec,
    scratch_base: u64,
) -> Result<Packed, Error> {
    let mut pack = ArenaPack::new();
    let mut items = Vec::with_capacity(spec.items.len());
    let mut scratch_at = 0usize;
    let mut scratch_arena = None;
    for item in &spec.items {
        let at = match item.data {
            ItemData::Slot(slot) => pack.slot(call.buffers, slot)?,
            ItemData::Scratch(bytes) => {
                let arena = match scratch_arena {
                    Some(a) => a,
                    None => {
                        let a = pack.arena(scratch_base)?;
                        scratch_arena = Some(a);
                        a
                    }
                };
                scratch_at = scratch_at.next_multiple_of(16);
                let r = Ref::at(arena, scratch_at as u64);
                scratch_at += bytes;
                r
            }
        };
        items.push(Item {
            at,
            kind: item.kind as u32,
            elem: item.elem as u32,
            dims: item.dims,
            shape: item.shape,
            share: item.share,
            form: item.form,
            forms: item.forms,
            unused: 0,
        });
    }
    let mut forms = Vec::with_capacity(spec.forms.len());
    for f in &spec.forms {
        forms.push(Form {
            codes: pack.slot(call.buffers, f.codes)?,
            suh: pack.opt(call.buffers, f.suh)?,
            svh: pack.opt(call.buffers, f.svh)?,
            kind: f.kind,
            rows: f.rows,
            cols: f.cols,
            unused: 0,
        });
    }
    let size = |d: DimName| call.bound.get(&d).copied();
    let mut kv = Vec::with_capacity(spec.conventions.kv.len());
    let mut kv_lens = Vec::with_capacity(spec.conventions.kv.len());
    let fallback_len = call.bound.get(&DimName::Context).copied().unwrap_or(0);
    for k in &spec.conventions.kv {
        let base = pack.slot(call.buffers, k.base)?;
        let pages = pack.opt(call.buffers, k.pages)?;
        kv.push(KvPlan {
            kept: Kept {
                base,
                host: Ref::NONE,
                pages,
                len: Ref::NONE,
                pages_per_slot: k.pages_per_slot,
                page_rows: k.page_rows,
                row_bytes: k.row_bytes,
                elem: k.elem as u32,
            },
            slots: k.slots,
            dense_pages: k.pages.is_none(),
        });
        let slots = k.slots.max(1) as usize;
        let mut lens = vec![fallback_len; slots];
        let copy = call.kv_lens.len().min(slots);
        if copy > 0 {
            lens[..copy].copy_from_slice(&call.kv_lens[..copy]);
        }
        kv_lens.push(lens);
    }
    let llm = match spec.conventions.llm {
        Some(l) => Some(pack_llm(&mut pack, call.buffers, l, &size)?),
        None => None,
    };
    let mut state = Vec::with_capacity(spec.conventions.state.len());
    for h in &spec.conventions.state {
        state.push(Held {
            base: pack.slot(call.buffers, h.base)?,
            fresh: pack.opt(call.buffers, h.fresh)?,
            slot_bytes: h.slot_bytes,
            elem: h.elem as u32,
        });
    }
    let ranks = call.plan.ranks.max(1);
    let mut pools = Vec::with_capacity(spec.pools.len());
    for p in &spec.pools {
        let group = match p.group.is_empty() {
            true => (0..ranks).collect(),
            false => p.group.clone(),
        };
        pools.push(PoolPlan {
            depth: p.depth,
            one_way: p.one_way,
            group,
            send_bytes: p.send_bytes,
            recv_bytes: p.recv_bytes,
        });
    }
    let mut lanes = Vec::with_capacity(spec.lanes.len());
    for l in &spec.lanes {
        lanes.push(LanePlan {
            op: collective_op(l.op),
            elem: l.elem as u32,
            pool: l.pool,
            block: l.block,
        });
    }
    let plan = Plan {
        pools,
        lanes,
        items,
        forms,
        conventions: ConventionsPlan { kv, llm, state },
        arenas: pack.arenas,
        threads: spec.threads,
        share: spec.share,
        shared_bytes: spec.shared_bytes,
    };
    Ok(Packed {
        plan,
        scratch_bytes: scratch_at,
        kv_lens,
    })
}

fn pack_llm(
    pack: &mut ArenaPack,
    buffers: &[u64],
    l: LlmSpec,
    size: &dyn Fn(DimName) -> Option<u32>,
) -> Result<LlmStep, Error> {
    let rows = l
        .rows
        .value(size)
        .ok_or_else(|| Error::Plan("unbound llm.rows".into()))? as u32;
    let out_rows = l
        .out_rows
        .value(size)
        .ok_or_else(|| Error::Plan("unbound llm.out_rows".into()))? as u32;
    Ok(LlmStep {
        rows,
        out_rows,
        vocab: l.vocab,
        unused: 0,
        token: pack.opt(buffers, l.token)?,
        slot: pack.opt(buffers, l.slot)?,
        position: pack.opt(buffers, l.position)?,
        out_row: pack.opt(buffers, l.out_row)?,
        temperature: pack.opt(buffers, l.temperature)?,
        top_k: pack.opt(buffers, l.top_k)?,
        seed: pack.opt(buffers, l.seed)?,
        logits: pack.opt(buffers, l.logits)?,
        sampled: pack.opt(buffers, l.sampled)?,
    })
}

/// Identifies one compiled resident kernel entry by source address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct StepKey {
    text_ptr: usize,
    entry: &'static str,
}

/// Structural layout signature for deciding in-place updates vs reopen.
#[derive(Clone, Debug, PartialEq, Eq)]
struct StepShape {
    threads: u32,
    share: u32,
    shared_bytes: u32,
    items: usize,
    forms: usize,
    kv_slots: Vec<(u32, bool)>,
    has_llm: bool,
    state: usize,
    pools: Vec<(u32, bool, Vec<u32>, u32, u32)>,
    lanes: Vec<(u32, u32, u32, u32)>,
}

impl StepShape {
    fn of(plan: &Plan) -> Self {
        Self {
            threads: plan.threads,
            share: plan.share,
            shared_bytes: plan.shared_bytes,
            items: plan.items.len(),
            forms: plan.forms.len(),
            kv_slots: plan
                .conventions
                .kv
                .iter()
                .map(|k| (k.slots, k.dense_pages))
                .collect(),
            has_llm: plan.conventions.llm.is_some(),
            state: plan.conventions.state.len(),
            pools: plan
                .pools
                .iter()
                .map(|p| {
                    (
                        p.depth,
                        p.one_way,
                        p.group.clone(),
                        p.send_bytes,
                        p.recv_bytes,
                    )
                })
                .collect(),
            lanes: plan
                .lanes
                .iter()
                .map(|l| (l.op as u32, l.elem, l.pool, l.block))
                .collect(),
        }
    }
}

struct StepState<D: Device, L> {
    resident: Resident<D, L>,
    scratch: Option<DeviceBuffer<D>>,
    shape: StepShape,
}

impl<D, L> StepState<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    fn start_fresh(
        resident: Resident<D, L>,
        scratch: Option<DeviceBuffer<D>>,
        packed: &Packed,
        values: &[u32],
    ) -> Result<Self, Error> {
        let mut state = Self {
            resident,
            scratch,
            shape: StepShape::of(&packed.plan),
        };
        for (i, lens) in packed.kv_lens.iter().enumerate() {
            state.resident.update_kv_len(i, lens)?;
        }
        state.resident.launch(values, TIMEOUT)?;
        Ok(state)
    }

    fn update_and_run(&mut self, packed: &Packed, values: &[u32]) -> Result<(), Error> {
        self.resident.update_arenas(&packed.plan.arenas)?;
        self.resident.update_items(&packed.plan.items)?;
        self.resident.update_forms(&packed.plan.forms)?;
        if let Some(llm) = packed.plan.conventions.llm {
            self.resident.update_llm(llm)?;
        }
        for (i, (kv, lens)) in packed
            .plan
            .conventions
            .kv
            .iter()
            .zip(&packed.kv_lens)
            .enumerate()
        {
            self.resident.update_kv(i, kv.kept)?;
            self.resident.update_kv_len(i, lens)?;
        }
        if !packed.plan.conventions.state.is_empty() {
            self.resident.update_state(&packed.plan.conventions.state)?;
        }
        self.resident.launch(values, TIMEOUT)
    }
}

/// Caches open `Resident` instances and executes resident steps.
pub struct Executor<D: Device, L> {
    ctx: D,
    local: HashMap<StepKey, StepState<D, LocalLink>>,
    linked: Option<(StepKey, StepState<D, L>)>,
    link: Option<L>,
}

fn kernel_text<D: Device>(step: &Step) -> Result<(&'static str, &'static str), Error> {
    let Source::Fixed {
        cuda, hip, entry, ..
    } = step.kernel.source()
    else {
        return Err(Error::Plan("resident step has non-fixed source".into()));
    };
    let text = match D::BACKEND {
        Gpu::Cuda => cuda,
        Gpu::Rocm => hip,
        Gpu::Metal => return Err(Error::Plan("Metal has no persistent grids".into())),
    };
    if text.is_empty() {
        return Err(Error::Plan(format!("{entry} lacks text for this backend")));
    }
    Ok((text, entry))
}

fn alloc_scratch<D: Device>(
    ctx: &D,
    existing: Option<DeviceBuffer<D>>,
    need: usize,
) -> Result<(Option<DeviceBuffer<D>>, u64), Error> {
    if need == 0 {
        return Ok((existing, 0));
    }
    if let Some(buf) = existing
        && buf.size() >= need
    {
        let ptr = buf.device_ptr();
        return Ok((Some(buf), ptr));
    }
    let buf = DeviceBuffer::alloc(ctx, need)?;
    buf.zero()?;
    let ptr = buf.device_ptr();
    Ok((Some(buf), ptr))
}

impl<D, L> Executor<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    fn run_local(
        &mut self,
        key: StepKey,
        text: &str,
        call: &ResidentCall<'_>,
        spec: &ResidentSpec,
        values: &[u32],
    ) -> Result<(), Error> {
        let need = scratch_size(spec);
        let can_reuse = self
            .local
            .get(&key)
            .is_some_and(|s| s.scratch.as_ref().map_or(0, DeviceBuffer::size) >= need);
        if can_reuse {
            let state = self.local.get_mut(&key).expect("checked");
            let ptr = state.scratch.as_ref().map_or(0, DeviceBuffer::device_ptr);
            let packed = pack_plan(call, spec, ptr)?;
            if state.shape == StepShape::of(&packed.plan) {
                return state.update_and_run(&packed, values);
            }
        }
        let prev = self.local.remove(&key).and_then(|s| s.scratch);
        let (scratch, ptr) = alloc_scratch(&self.ctx, prev, need)?;
        let packed = pack_plan(call, spec, ptr)?;
        let resident = Resident::open_local(&self.ctx, text, key.entry, &packed.plan)?;
        let state = StepState::start_fresh(resident, scratch, &packed, values)?;
        self.local.insert(key, state);
        Ok(())
    }

    fn run_linked(
        &mut self,
        key: StepKey,
        text: &str,
        call: &ResidentCall<'_>,
        spec: &ResidentSpec,
        values: &[u32],
    ) -> Result<(), Error> {
        let need = scratch_size(spec);
        if let Some((held_key, state)) = &mut self.linked
            && *held_key == key
            && state.scratch.as_ref().map_or(0, DeviceBuffer::size) >= need
        {
            let ptr = state.scratch.as_ref().map_or(0, DeviceBuffer::device_ptr);
            let packed = pack_plan(call, spec, ptr)?;
            if state.shape == StepShape::of(&packed.plan) {
                return state.update_and_run(&packed, values);
            }
        }
        let mut prev_scratch = None;
        if let Some((_, prev)) = self.linked.take() {
            prev_scratch = prev.scratch;
            self.link = Some(prev.resident.close()?);
        }
        let link = self
            .link
            .take()
            .ok_or_else(|| Error::Plan("collective resident step requires a rank link".into()))?;
        let (scratch, ptr) = alloc_scratch(&self.ctx, prev_scratch, need)?;
        let packed = pack_plan(call, spec, ptr)?;
        let resident = Resident::open(&self.ctx, text, key.entry, link, &packed.plan)?;
        let state = StepState::start_fresh(resident, scratch, &packed, values)?;
        self.linked = Some((key, state));
        Ok(())
    }

    fn run_inner(&mut self, call: &ResidentCall<'_>) -> Result<(), Error> {
        let step = &call.plan.steps[call.step];
        let spec = step
            .resident()
            .ok_or_else(|| Error::Plan("step has no resident spec".into()))?;
        let (text, entry) = kernel_text::<D>(step)?;
        let key = StepKey {
            text_ptr: text.as_ptr() as usize,
            entry,
        };
        let size = |d: DimName| call.bound.get(&d).copied();
        let values = spec
            .bound_values(&size)
            .ok_or_else(|| Error::Plan("unbound resident launch value".into()))?;
        if spec.pools.is_empty() {
            self.run_local(key, text, call, &spec, &values)
        } else {
            self.run_linked(key, text, call, &spec, &values)
        }
    }
}

impl<D, L> ResidentExec for Executor<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    fn run(&mut self, call: &ResidentCall<'_>) -> Result<(), RunError> {
        self.run_inner(call)
            .map_err(|e| RunError::Resident(Box::new(e)))
    }
}

/// Resident step executor over `link` on `ctx`.
pub fn on<D, L>(ctx: &D, link: L) -> Box<dyn ResidentExec>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Link + 'static,
{
    Box::new(Executor {
        ctx: ctx.clone(),
        local: HashMap::new(),
        linked: None,
        link: Some(link),
    })
}

/// Single-rank resident step executor on `ctx`, with no link.
#[cfg_attr(not(any(feature = "cuda", feature = "rocm")), allow(dead_code))]
pub(crate) fn on_local<D>(ctx: &D) -> Box<dyn ResidentExec>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    Box::new(Executor::<D, LocalLink> {
        ctx: ctx.clone(),
        local: HashMap::new(),
        linked: None,
        link: None,
    })
}

/// Single-rank resident executor on `gpu`, or `None` without CUDA/ROCm.
pub fn local(gpu: Option<Gpu>) -> Result<Option<Box<dyn ResidentExec>>, Error> {
    match gpu {
        #[cfg(feature = "cuda")]
        Some(Gpu::Cuda) => Ok(Some(on_local(tessera_gpukit::cuda::cached_context()?))),
        #[cfg(feature = "rocm")]
        Some(Gpu::Rocm) => Ok(Some(on_local(tessera_gpukit::rocm::cached_context()?))),
        _ => Ok(None),
    }
}

/// Multi-rank resident executor over `link` on `gpu`.
pub fn open<L: Link + 'static>(gpu: Option<Gpu>, link: L) -> Result<Box<dyn ResidentExec>, Error> {
    match gpu {
        #[cfg(feature = "cuda")]
        Some(Gpu::Cuda) => Ok(on(tessera_gpukit::cuda::cached_context()?, link)),
        #[cfg(feature = "rocm")]
        Some(Gpu::Rocm) => Ok(on(tessera_gpukit::rocm::cached_context()?, link)),
        _ => {
            drop(link);
            Err(Error::Plan(
                "persistent grids need a CUDA or ROCm GPU".into(),
            ))
        }
    }
}
