//! One persistent grid per rank, with its proxy beside it.

use crate::layout::{
    ARENAS, Barrier, CONV_KV, CONV_LLM, CONV_STATE, CONVENTIONS, ConventionRef, Counters,
    DOORBELL_ABORT, DTYPES, Fault, Form, Held, Item, Kept, KvTables, Lane, Launch as Record,
    LlmStep, MAX_DEPTH, MAX_GROUP, MAX_POOLS, NONE, PAGE, Pool, PoolSet, Ref, SIZES, STATUS,
    StateTables, Table, VALUES,
};
use crate::proxy::page::Page;
use crate::proxy::running::Running;
use crate::proxy::wired::{CREDITS, FLAGS, Place, RECV, REGIONS, SEND, Wired, regions};
use crate::proxy::{self, Op, Proxy};
use crate::{Backend, Options, flatten};
use std::ffi::c_int;
use std::fmt;
use std::mem::offset_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};
use tessera_engine_comm::{Error as CommError, Link as Comm, Put, Span};
use tessera_gpukit::Error as GpuError;
use tessera_gpukit::backend::Backend as Gpu;
use tessera_gpukit::device::{Device, DeviceBuffer};
use tessera_gpukit::launch::{Apart, KernelArgs, Launch};

/// Arenas the launcher fills: control, doorbell, then the link's.
const CONTROL: u32 = 0;
const DOORBELL: u32 = 1;
const LINK: u32 = 2;
/// The first arena a plan's items name.
pub const USER: u32 = LINK + REGIONS as u32;
/// Every pool starts at this ordinal.
const BASE: u64 = 1;

const CLOCK: &str = include_str!("clock.kernel");

unsafe extern "C" {
    fn getpagesize() -> c_int;
}

/// One pool: shape, member ranks, and bytes per slot.
#[derive(Clone, Debug)]
pub struct PoolPlan {
    pub depth: u32,
    pub one_way: bool,
    pub group: Vec<u32>,
    pub send_bytes: u32,
    pub recv_bytes: u32,
}

/// One lane: its operation, `dtype`, pool and all-to-all block.
#[derive(Clone, Copy, Debug)]
pub struct LanePlan {
    pub op: Op,
    pub elem: u32,
    pub pool: u32,
    pub block: u32,
}

/// One kept-row table with its slot count and dense flag.
#[derive(Clone, Debug)]
pub struct KvPlan {
    pub kept: Kept,
    pub slots: u32,
    /// When true and `kept.pages` is `NONE`, writes identity pages.
    pub dense_pages: bool,
}

/// Optional convention records written into the control arena.
#[derive(Clone, Debug, Default)]
pub struct ConventionsPlan {
    pub kv: Vec<KvPlan>,
    pub llm: Option<LlmStep>,
    pub state: Vec<Held>,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub pools: Vec<PoolPlan>,
    pub lanes: Vec<LanePlan>,
    /// Items whose refs name arenas from `USER` on.
    pub items: Vec<Item>,
    /// Packed or rotated weights referenced by `FORM` items.
    pub forms: Vec<Form>,
    /// Convention records populated in `table.conventions`.
    pub conventions: ConventionsPlan,
    /// Device addresses of those arenas, in order.
    pub arenas: Vec<u64>,
    pub threads: u32,
    /// Grids that run on the device at once.
    pub share: u32,
    /// Dynamic shared memory per block, in bytes.
    pub shared_bytes: u32,
}

/// Single-rank link for resident kernels with no collective pools.
#[derive(Clone, Copy, Debug, Default)]
pub struct LocalLink;

impl Comm for LocalLink {
    fn rank(&self) -> u32 {
        0
    }

    fn ranks(&self) -> u32 {
        1
    }

    fn put(&mut self, _: Put) -> Result<(), CommError> {
        Err(CommError::Fit("local link has no peers".into()))
    }

    fn word(&mut self, _: u32, _: Span, _: u64, _: u64) -> Result<(), CommError> {
        Err(CommError::Fit("local link has no peers".into()))
    }

    fn poll(&mut self, _: &mut Vec<u64>) -> Result<(), CommError> {
        Ok(())
    }

    fn reset(&mut self) -> Result<(), CommError> {
        Ok(())
    }

    fn load(&self, _: Span) -> Result<u64, CommError> {
        Err(CommError::Fit("local link has no regions".into()))
    }

    fn read(&self, _: Span) -> Result<Vec<u8>, CommError> {
        Err(CommError::Fit("local link has no regions".into()))
    }

    fn write(&mut self, _: Span, _: &[u8]) -> Result<(), CommError> {
        Err(CommError::Fit("local link has no regions".into()))
    }
}

impl Plan {
    pub(crate) fn proxy_pools(&self, rank: u32) -> Vec<proxy::Pool> {
        let pool = |p: &PoolPlan| proxy::Pool {
            depth: p.depth,
            one_way: p.one_way,
            group: p.group.len() as u32,
            me: p.group.iter().position(|&r| r == rank).unwrap_or(0) as u32,
        };
        self.pools.iter().map(pool).collect()
    }

    pub(crate) fn bytes(&self) -> Vec<(usize, usize)> {
        let bytes = |p: &PoolPlan| (p.send_bytes as usize, p.recv_bytes as usize);
        self.pools.iter().map(bytes).collect()
    }

    /// The region sizes for the rank's link.
    pub fn regions(&self) -> [usize; REGIONS] {
        regions(&self.proxy_pools(0), &self.bytes())
    }
}

/// A `status_code` from `core.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status(pub u32);

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match STATUS.get(self.0 as usize) {
            Some(name) => f.write_str(&name.to_lowercase().replace('_', " ")),
            None => write!(f, "status {}", self.0),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Comm(#[from] CommError),
    #[error("proxy: {0}")]
    Proxy(#[from] proxy::Error),
    #[error(transparent)]
    Kernel(#[from] crate::Error),
    #[error("plan: {0}")]
    Plan(String),
    #[error("the kernel stopped: {code} in pool {pool} at ordinal {ordinal}")]
    Fault {
        code: Status,
        pool: u32,
        ordinal: u64,
    },
}

fn plan(what: impl Into<String>) -> Error {
    Error::Plan(what.into())
}

/// The bytes of plain `repr(C)` values.
fn bytes<T: Copy>(values: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), size_of_val(values)) }
}

/// The control arena, built on the host.
#[derive(Default)]
struct Block(Vec<u8>);

impl Block {
    fn put<T: Copy>(&mut self, values: &[T]) -> u64 {
        let at = self.0.len().next_multiple_of(16);
        self.0.resize(at, 0);
        self.0.extend_from_slice(bytes(values));
        at as u64
    }
}

fn zeroed<T>() -> T {
    // Only for the plain integer structs of `layout`.
    unsafe { std::mem::zeroed() }
}

/// Where each record sits in the control arena.
#[derive(Clone, Debug, Default)]
pub struct ControlPlaces {
    pub launch: u64,
    pub status: u64,
    pub items: Option<u64>,
    pub items_len: usize,
    pub forms: Option<u64>,
    pub forms_len: usize,
    pub llm: Option<u64>,
    pub kv_tables: Option<u64>,
    /// Control offset and slot count per KV table's `len` array.
    pub kv_lens: Vec<(u64, u32)>,
    /// Control offset for each dense KV table's identity pages.
    pub kv_pages: Vec<Option<u64>>,
    pub state_tables: Option<u64>,
    pub state_len: usize,
}

/// Builds the host-side control arena bytes and record offsets.
pub(crate) fn layout_control(
    plan: &Plan,
    [rank, ranks]: [u32; 2],
    arenas: [u64; ARENAS],
    pool_places: &[Place],
) -> Result<(Vec<u8>, ControlPlaces), Error> {
    check(plan, rank)?;
    if pool_places.len() != plan.pools.len() {
        return Err(self::plan("pool place count mismatch"));
    }
    let mut block = Block::default();
    let table_at = block.put(&[zeroed::<Table>()]);
    let launch = block.put(&[zeroed::<Record>()]);
    let (items_ref, items_at) = match plan.items.is_empty() {
        true => (Ref::NONE, None),
        false => {
            let at = block.put(&plan.items);
            (Ref::at(CONTROL, at), Some(at))
        }
    };
    let (forms_ref, forms_at) = match plan.forms.is_empty() {
        true => (Ref::NONE, None),
        false => {
            let at = block.put(&plan.forms);
            (Ref::at(CONTROL, at), Some(at))
        }
    };
    let (lanes_ref, lanes_len, pools_ref) = if plan.pools.is_empty() {
        (Ref::NONE, 0, Ref::NONE)
    } else {
        let lanes: Vec<Lane> = plan.lanes.iter().map(|l| lane(plan, l, rank)).collect();
        let lanes_at = block.put(&lanes);
        let mut set: PoolSet = zeroed();
        set.count = plan.pools.len() as u32;
        for (p, (pool, &place)) in plan.pools.iter().zip(pool_places).enumerate() {
            set.pools[p] = pool_record(pool, p, place);
        }
        let mut counters: Counters = zeroed();
        counters.next = [BASE; MAX_POOLS];
        // The first bump must differ from the page's zero.
        counters.writes = 1;
        set.state = Ref::at(CONTROL, block.put(&[counters]));
        let pools = block.put(&[set]);
        (
            Ref::at(CONTROL, lanes_at),
            lanes.len() as u32,
            Ref::at(CONTROL, pools),
        )
    };
    let barrier = block.put(&[zeroed::<Barrier>()]);
    let status = block.put(&[zeroed::<Fault>()]);
    let none = ConventionRef {
        at: Ref::NONE,
        version: 0,
        bytes: 0,
    };
    let mut conventions = [none; CONVENTIONS];
    let mut places = ControlPlaces {
        launch,
        status,
        items: items_at,
        items_len: plan.items.len(),
        forms: forms_at,
        forms_len: plan.forms.len(),
        ..ControlPlaces::default()
    };
    if let Some(llm) = plan.conventions.llm {
        let at = block.put(&[llm]);
        places.llm = Some(at);
        conventions[CONV_LLM] = ConventionRef {
            at: Ref::at(CONTROL, at),
            version: crate::EDITION,
            bytes: size_of::<LlmStep>() as u32,
        };
    }
    if !plan.conventions.kv.is_empty() {
        let mut kepts = Vec::with_capacity(plan.conventions.kv.len());
        for kv in &plan.conventions.kv {
            let slots = kv.slots.max(1);
            let len_arr = vec![0u32; slots as usize];
            let len_at = block.put(&len_arr);
            places.kv_lens.push((len_at, slots));
            let mut kept = kv.kept;
            kept.len = Ref::at(CONTROL, len_at);
            if kv.dense_pages && kept.pages.arena == NONE {
                let dense: Vec<u32> = (0..slots).collect();
                let pages_at = block.put(&dense);
                kept.pages = Ref::at(CONTROL, pages_at);
                places.kv_pages.push(Some(pages_at));
            } else {
                places.kv_pages.push(None);
            }
            kepts.push(kept);
        }
        let tables_at = block.put(&kepts);
        places.kv_tables = Some(tables_at);
        let header = KvTables {
            count: kepts.len() as u32,
            unused: 0,
            tables: Ref::at(CONTROL, tables_at),
        };
        let kv_at = block.put(&[header]);
        conventions[CONV_KV] = ConventionRef {
            at: Ref::at(CONTROL, kv_at),
            version: crate::EDITION,
            bytes: size_of::<KvTables>() as u32,
        };
    }
    if !plan.conventions.state.is_empty() {
        let held_at = block.put(&plan.conventions.state);
        places.state_tables = Some(held_at);
        places.state_len = plan.conventions.state.len();
        let header = StateTables {
            count: plan.conventions.state.len() as u32,
            unused: 0,
            tables: Ref::at(CONTROL, held_at),
        };
        let state_at = block.put(&[header]);
        conventions[CONV_STATE] = ConventionRef {
            at: Ref::at(CONTROL, state_at),
            version: crate::EDITION,
            bytes: size_of::<StateTables>() as u32,
        };
    }
    let table = Table {
        edition: crate::EDITION,
        rank,
        ranks,
        items_len: plan.items.len() as u32,
        forms_len: plan.forms.len() as u32,
        lanes_len,
        values_len: VALUES as u32,
        unused: 0,
        arenas,
        now: Ref::at(CONTROL, launch),
        items: items_ref,
        forms: forms_ref,
        conventions,
        lanes: lanes_ref,
        pools: pools_ref,
        doorbell: Ref::at(DOORBELL, 0),
        barrier: Ref::at(CONTROL, barrier),
        status: Ref::at(CONTROL, status),
        marks: Ref::NONE,
    };
    block.0[table_at as usize..][..size_of::<Table>()].copy_from_slice(bytes(&[table]));
    Ok((block.0, places))
}

pub(crate) fn backend<D: Device>() -> Result<Backend, Error> {
    match D::BACKEND {
        Gpu::Cuda => Ok(Backend::Cuda),
        Gpu::Rocm => Ok(Backend::Hip),
        Gpu::Metal => Err(plan("Metal has no persistent grids")),
    }
}

/// The device's clock now, from a one-thread kernel.
fn clock<D: Device + Launch<Kernel = <D as Device>::Kernel>>(ctx: &D) -> Result<u64, Error> {
    let options = Options {
        backend: backend::<D>()?,
        remote_kv: false,
        marks: false,
    };
    let flat = flatten("clock.kernel", CLOCK, &|_| None, &options)?;
    let kernel = ctx.compile(&flat.text, "tessera_clock")?;
    let out = DeviceBuffer::alloc(ctx, 8)?;
    let mut args = KernelArgs::new();
    args.device_ptr(out.device_ptr());
    ctx.launch(&kernel, (1, 1, 1), (1, 1, 1), 0, &mut args)?;
    ctx.finish()?;
    let mut word = [0u8; 8];
    out.read_bytes(&mut word)?;
    Ok(u64::from_ne_bytes(word))
}

/// A persistent kernel, its control arena and its proxy.
pub struct Resident<D: Device, L> {
    ctx: D,
    kernel: D::Kernel,
    control: DeviceBuffer<D>,
    places: ControlPlaces,
    page: u64,
    registered: Vec<*mut u8>,
    running: Option<Running<L>>,
    /// Whether this resident owns a collective host proxy thread.
    has_proxy: bool,
    /// A stopped or unused link, kept until its pages unregister.
    parked: Option<Wired<L>>,
    grid: u32,
    threads: u32,
    shared_bytes: u32,
    /// The device clock and the host instant it matches.
    clock: (u64, Instant),
    seq: u64,
    broken: bool,
    /// A grid started apart, until `join`.
    apart: Option<Box<dyn Apart>>,
}

impl<D> Resident<D, LocalLink>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
{
    /// Opens a single-rank resident grid without a collective link.
    pub fn open_local(ctx: &D, text: &str, entry: &str, plan: &Plan) -> Result<Self, Error> {
        if !plan.pools.is_empty() {
            return Err(self::plan("collective pools require a multi-rank link"));
        }
        Self::open(ctx, text, entry, LocalLink, plan)
    }
}

impl<D, L> Resident<D, L>
where
    D: Device + Launch<Kernel = <D as Device>::Kernel>,
    L: Comm + 'static,
{
    /// Compiles `text`, lays out `plan` and starts any proxy.
    pub fn open(ctx: &D, text: &str, entry: &str, link: L, plan: &Plan) -> Result<Self, Error> {
        let rank = link.rank();
        check(plan, rank)?;
        let kernel = ctx.compile(text, entry)?;
        let fit = ctx.resident(&kernel, plan.threads, plan.shared_bytes)?;
        if fit == 0 {
            let threads = plan.threads;
            return Err(self::plan(format!(
                "{entry} cannot stay resident with {threads} threads"
            )));
        }
        let units = ctx.platform().units as u32;
        let grid = (units * fit / plan.share.max(1)).max(1);
        let ranks = link.ranks();
        let pools = plan.proxy_pools(rank);
        let groups = plan.pools.iter().map(|p| p.group.clone()).collect();
        let mut wired = Wired::new(link, &pools, &plan.bytes(), groups)?;
        let (page, device_page) = ctx.pin(PAGE)?;
        unsafe { std::ptr::write_bytes(page as *mut u8, 0, PAGE) };
        let has_proxy = !plan.pools.is_empty();
        let mut made = Resident {
            ctx: ctx.clone(),
            kernel,
            control: DeviceBuffer::alloc(ctx, 16)?,
            places: ControlPlaces::default(),
            page,
            registered: Vec::new(),
            running: None,
            has_proxy,
            parked: None,
            grid,
            threads: plan.threads,
            shared_bytes: plan.shared_bytes,
            clock: (0, Instant::now()),
            seq: 0,
            broken: false,
            apart: None,
        };
        let mut arenas = [0u64; ARENAS];
        arenas[DOORBELL as usize] = device_page;
        arenas[USER as usize..][..plan.arenas.len()].copy_from_slice(&plan.arenas);
        if has_proxy {
            let mapped = made.map(wired.link(), plan)?;
            arenas[LINK as usize..][..REGIONS].copy_from_slice(&mapped);
            let mut page_words = unsafe { Page::new(page as *mut u64, &mut wired)? };
            let proxy_lanes = plan.lanes.iter().map(|l| proxy::Lane {
                op: l.op,
                pool: l.pool,
                width: width(l.elem),
                block: l.block,
            });
            let proxy = Proxy::new(&mut page_words, pools, proxy_lanes.collect(), BASE)?;
            made.lay_out(&wired, plan, [rank, ranks], arenas)?;
            made.clock = (clock(ctx)?, Instant::now());
            made.running = Some(Running::spawn(proxy, page_words, wired));
        } else {
            made.lay_out(&wired, plan, [rank, ranks], arenas)?;
            made.clock = (clock(ctx)?, Instant::now());
            made.parked = Some(wired);
        }
        Ok(made)
    }

    /// Device addresses of the link's regions. Registers unpinned pages.
    fn map(&mut self, link: &mut L, plan: &Plan) -> Result<[u64; REGIONS], Error> {
        let page = unsafe { getpagesize() } as usize;
        let mut found = Vec::with_capacity(REGIONS);
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (region, need) in plan.regions().into_iter().enumerate() {
            let (host, device, len) = link
                .memory(region)
                .ok_or_else(|| self::plan(format!("the link has no fixed region {region}")))?;
            if len < need {
                return Err(self::plan(format!(
                    "region {region} holds {len} of {need} bytes"
                )));
            }
            let host = host as usize;
            if device.is_none() {
                spans.push((host / page * page, (host + len).next_multiple_of(page)));
            }
            found.push((host, device));
        }
        // Regions may share pages, so each page registers once.
        spans.sort_unstable();
        let mut merged: Vec<(usize, usize, u64)> = Vec::new();
        for (start, end) in spans {
            match merged.last_mut() {
                Some(last) if start <= last.1 => last.1 = last.1.max(end),
                _ => merged.push((start, end, 0)),
            }
        }
        for (start, end, base) in &mut merged {
            *base = self.ctx.register(*start as *mut u8, *end - *start)?;
            self.registered.push(*start as *mut u8);
        }
        let mut out = [0u64; REGIONS];
        for (slot, (host, device)) in out.iter_mut().zip(found) {
            let span = merged.iter().find(|m| (m.0..m.1).contains(&host));
            *slot = match (device, span) {
                (Some(device), _) => device,
                (None, Some(&(start, _, base))) => base + (host - start) as u64,
                (None, None) => unreachable!("every unpinned region has a span"),
            };
        }
        Ok(out)
    }

    /// Writes the table and the records it points to.
    fn lay_out(
        &mut self,
        wired: &Wired<L>,
        plan: &Plan,
        [rank, ranks]: [u32; 2],
        arenas: [u64; ARENAS],
    ) -> Result<(), Error> {
        let mut pool_places = Vec::with_capacity(plan.pools.len());
        for p in 0..plan.pools.len() {
            pool_places.push(wired.place(p).expect("a place per pool"));
        }
        let (mut raw, places) = layout_control(plan, [rank, ranks], arenas, &pool_places)?;
        let control = DeviceBuffer::alloc(&self.ctx, raw.len())?;
        let ptr_at = offset_of!(Table, arenas) + CONTROL as usize * size_of::<u64>();
        raw[ptr_at..ptr_at + 8].copy_from_slice(&control.device_ptr().to_ne_bytes());
        control.write_bytes(&raw)?;
        self.control = control;
        self.places = places;
        Ok(())
    }

    /// Updates user arena base addresses in the control table.
    pub fn update_arenas(&mut self, user_arenas: &[u64]) -> Result<(), Error> {
        if user_arenas.len() > ARENAS - USER as usize {
            return Err(plan(format!("{} arenas", user_arenas.len())));
        }
        let mut slots = [0u64; ARENAS - USER as usize];
        slots[..user_arenas.len()].copy_from_slice(user_arenas);
        let at = (offset_of!(Table, arenas) + USER as usize * size_of::<u64>()) as u64;
        self.ctx.upload(self.table() + at, bytes(&slots))?;
        Ok(())
    }

    /// Updates item records in place when the count matches.
    pub fn update_items(&mut self, items: &[Item]) -> Result<(), Error> {
        if items.len() != self.places.items_len {
            return Err(plan("item count changed"));
        }
        if let Some(at) = self.places.items {
            self.ctx.upload(self.table() + at, bytes(items))?;
        }
        Ok(())
    }

    /// Updates form records in place when the count matches.
    pub fn update_forms(&mut self, forms: &[Form]) -> Result<(), Error> {
        if forms.len() != self.places.forms_len {
            return Err(plan("form count changed"));
        }
        if let Some(at) = self.places.forms {
            self.ctx.upload(self.table() + at, bytes(forms))?;
        }
        Ok(())
    }

    /// Updates the `LLM` convention record in place.
    pub fn update_llm(&mut self, llm: LlmStep) -> Result<(), Error> {
        let at = self
            .places
            .llm
            .ok_or_else(|| plan("no LLM convention in table"))?;
        self.ctx.upload(self.table() + at, bytes(&[llm]))?;
        Ok(())
    }

    /// Updates one `KV` table record, keeping control-owned refs.
    pub fn update_kv(&mut self, index: usize, mut kept: Kept) -> Result<(), Error> {
        let base_at = self
            .places
            .kv_tables
            .ok_or_else(|| plan("no KV convention in table"))?;
        let &(len_at, _) = self
            .places
            .kv_lens
            .get(index)
            .ok_or_else(|| plan(format!("no KV table {index}")))?;
        kept.len = Ref::at(CONTROL, len_at);
        if kept.pages.arena == NONE
            && let Some(pages_at) = self.places.kv_pages[index]
        {
            kept.pages = Ref::at(CONTROL, pages_at);
        }
        let at = base_at + (index * size_of::<Kept>()) as u64;
        self.ctx.upload(self.table() + at, bytes(&[kept]))?;
        Ok(())
    }

    /// Updates per-slot kept lengths for KV table `index`.
    pub fn update_kv_len(&mut self, index: usize, lens: &[u32]) -> Result<(), Error> {
        let &(len_at, slots) = self
            .places
            .kv_lens
            .get(index)
            .ok_or_else(|| plan(format!("no KV table {index}")))?;
        if lens.len() > slots as usize {
            return Err(plan("too many KV slot lengths"));
        }
        let mut words = vec![0u32; slots as usize];
        words[..lens.len()].copy_from_slice(lens);
        self.ctx.upload(self.table() + len_at, bytes(&words))?;
        Ok(())
    }

    /// Updates the `STATE` convention records in place.
    pub fn update_state(&mut self, held: &[Held]) -> Result<(), Error> {
        if held.len() != self.places.state_len {
            return Err(plan("state table count changed"));
        }
        if let Some(at) = self.places.state_tables {
            self.ctx.upload(self.table() + at, bytes(held))?;
        }
        Ok(())
    }

    /// Blocks in the grid.
    pub fn grid(&self) -> u32 {
        self.grid
    }

    /// One launch with `values`. Waits for the grid to end.
    pub fn launch(&mut self, values: &[u32], timeout: Duration) -> Result<(), Error> {
        let deadline = self.deadline(timeout);
        self.record(values, deadline)?;
        let mut args = KernelArgs::new();
        args.device_ptr(self.table());
        let (grid, threads) = ((self.grid, 1, 1), (self.threads, 1, 1));
        self.ctx
            .launch(&self.kernel, grid, threads, self.shared_bytes, &mut args)?;
        self.ctx.finish()?;
        self.status()?;
        self.broken = false;
        Ok(())
    }

    /// Starts the grid on its own stream, with no deadline.
    pub fn start(&mut self, values: &[u32]) -> Result<(), Error> {
        if self.apart.is_some() {
            return Err(plan("the grid already runs"));
        }
        self.record(values, 0)?;
        let mut args = KernelArgs::new();
        args.device_ptr(self.table());
        let (grid, threads) = ((self.grid, 1, 1), (self.threads, 1, 1));
        let apart = self
            .ctx
            .launch_apart(&self.kernel, grid, threads, &mut args)?;
        self.apart = Some(apart);
        self.broken = false;
        Ok(())
    }

    /// Waits for a started grid, then reads its fault.
    pub fn join(&mut self) -> Result<(), Error> {
        if let Some(apart) = self.apart.take() {
            apart.wait()?;
        }
        self.status()
    }

    /// The control table's device address, the kernel's argument.
    pub fn table(&self) -> u64 {
        self.control.device_ptr()
    }

    /// The device clock `timeout` from now.
    pub fn deadline(&self, timeout: Duration) -> u64 {
        let ahead = (self.clock.1.elapsed() + timeout).as_secs_f64() * self.ctx.ticks();
        self.clock.0 + ahead as u64
    }

    /// Writes the launch record and clears the fault.
    fn record(&mut self, values: &[u32], deadline: u64) -> Result<(), Error> {
        if self.broken {
            return Err(plan("an earlier launch failed"));
        }
        if values.len() > VALUES {
            return Err(plan(format!("{} launch values", values.len())));
        }
        self.broken = true;
        self.alive()?;
        self.seq += 1;
        let mut record = Record {
            seq: self.seq,
            deadline,
            values: [0; VALUES],
        };
        record.values[..values.len()].copy_from_slice(values);
        let base = self.table();
        self.ctx
            .upload(base + self.places.launch, bytes(&[record]))?;
        self.ctx
            .upload(base + self.places.status, bytes(&[zeroed::<Fault>()]))?;
        Ok(())
    }

    /// The proxy's error, or the fault the grid recorded.
    pub fn status(&mut self) -> Result<(), Error> {
        self.alive()?;
        let mut fault = [zeroed::<Fault>()];
        let raw = unsafe {
            std::slice::from_raw_parts_mut(fault.as_mut_ptr().cast(), size_of::<Fault>())
        };
        self.ctx.download(self.table() + self.places.status, raw)?;
        let [fault] = fault;
        if fault.code != 0 {
            self.broken = true;
            return Err(Error::Fault {
                code: Status(fault.code),
                pool: fault.pool,
                ordinal: fault.ordinal,
            });
        }
        Ok(())
    }

    /// The proxy's error, if its thread stopped.
    pub fn alive(&mut self) -> Result<(), Error> {
        if !self.has_proxy {
            return Ok(());
        }
        if !self.running.as_ref().is_none_or(Running::failed) {
            return Ok(());
        }
        let Some((wired, ended)) = self.running.take().map(Running::stop) else {
            return Err(plan("the proxy is gone"));
        };
        self.parked = wired;
        Err(ended
            .err()
            .map_or_else(|| plan("the proxy is gone"), Error::from))
    }

    /// Stops any proxy and gives the link back.
    pub fn close(mut self) -> Result<L, Error> {
        if !self.has_proxy {
            let wired = self.parked.take().ok_or_else(|| plan("the link is gone"))?;
            return Ok(wired.into_link());
        }
        let running = self
            .running
            .take()
            .ok_or_else(|| plan("the proxy is gone"))?;
        let (wired, ended) = running.stop();
        let wired = wired.ok_or_else(|| plan("the proxy thread panicked"));
        ended?;
        Ok(wired?.into_link())
    }
}

impl<D: Device, L> Drop for Resident<D, L> {
    fn drop(&mut self) {
        // A grid still running sees the abort word and ends.
        if let Some(apart) = self.apart.take() {
            let abort = (self.page as usize + DOORBELL_ABORT) as *const AtomicU64;
            unsafe { &*abort }.store(1, Ordering::Release);
            let _ = apart.wait();
        }
        // The proxy reads the page and regions until it stops.
        if let Some(running) = self.running.take() {
            self.parked = running.stop().0;
        }
        // Unregister before `parked` drops and its link unmaps.
        for host in self.registered.drain(..) {
            self.ctx.unregister(host);
        }
        self.ctx.unpin(self.page);
    }
}

fn check(plan: &Plan, rank: u32) -> Result<(), Error> {
    if plan.pools.len() > MAX_POOLS {
        return Err(self::plan(format!("{} pools", plan.pools.len())));
    }
    if plan.arenas.len() > ARENAS - USER as usize {
        return Err(self::plan(format!("{} arenas", plan.arenas.len())));
    }
    if let Some(l) = plan.lanes.iter().find(|l| l.elem as usize >= DTYPES.len()) {
        return Err(self::plan(format!("a lane of dtype {}", l.elem)));
    }
    if let Some(l) = plan
        .lanes
        .iter()
        .find(|l| l.pool as usize >= plan.pools.len())
    {
        return Err(self::plan(format!("lane names missing pool {}", l.pool)));
    }
    for (p, pool) in plan.pools.iter().enumerate() {
        if !pool.group.contains(&rank) {
            return Err(self::plan(format!("rank {rank} is not in pool {p}")));
        }
    }
    Ok(())
}

/// Bytes per element of a `dtype`. `check` refused unknown ones.
fn width(elem: u32) -> u32 {
    DTYPES[elem as usize].1
}

fn lane(plan: &Plan, l: &LanePlan, rank: u32) -> Lane {
    let group = &plan.pools[l.pool as usize].group;
    Lane {
        what: l.op as u32,
        elem: l.elem,
        group_size: group.len() as u32,
        group_rank: group.iter().position(|&r| r == rank).unwrap_or(0) as u32,
        block: l.block,
        pool: l.pool,
        unused: [0; 2],
    }
}

fn pool_record(pool: &PoolPlan, p: usize, place: Place) -> Pool {
    let region = |r: usize, at: usize| Ref::at(LINK + r as u32, at as u64);
    let [send, recv, flags, credits] = [SEND, RECV, FLAGS, CREDITS];
    let mut record: Pool = zeroed();
    record.depth = pool.depth;
    record.one_way = u32::from(pool.one_way);
    record.send_bytes = pool.send_bytes;
    record.recv_bytes = pool.recv_bytes;
    for slot in 0..pool.depth as usize {
        record.send[slot] = region(send, place.send + slot * place.send_bytes);
        for k in 0..MAX_GROUP {
            let at = place.recv + (slot * MAX_GROUP + k) * place.recv_bytes;
            record.recv[slot * MAX_GROUP + k] = region(recv, at);
        }
    }
    record.arrived = region(flags, place.flags);
    record.credit = region(credits, place.credits);
    let sizes = SIZES + p * MAX_DEPTH * MAX_GROUP * 4;
    record.sizes = Ref::at(DOORBELL, sizes as u64);
    record
}
