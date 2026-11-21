//! A kernel's declaration: its arguments, shape and source.

use crate::extent::Extent;
use crate::plan::{Slot, Step};
use tessera_codegen::ir::{Body, Param};
use tessera_codegen::{Dialect, hash};
use tessera_graph::DimName;

/// One kernel argument, in the order its source declares.
#[derive(Clone, Debug, PartialEq)]
pub enum Arg {
    /// A slot's buffer, an extent of elements in.
    Buffer(Slot, Extent),
    /// A plain value by its bytes, any `repr(C)` type.
    Value(Vec<u8>),
    /// A u32 read from a dimension at launch.
    Size(Extent),
}

impl Arg {
    /// A `repr(C)` value by its bytes.
    pub fn value<T: Copy>(value: &T) -> Arg {
        let at = std::ptr::from_ref(value).cast::<u8>();
        Arg::Value(unsafe { std::slice::from_raw_parts(at, size_of::<T>()) }.to_vec())
    }

    pub fn count(n: usize) -> Arg {
        Arg::value(&(n as u32))
    }

    /// A count, plain when it is already known.
    pub fn size(e: Extent) -> Arg {
        match e {
            Extent::Fixed(n) => Arg::count(n),
            e => Arg::Size(e),
        }
    }

    /// A slot's buffer, `skip` elements in.
    pub fn at(slot: Slot, skip: usize) -> Arg {
        Arg::Buffer(slot, Extent::Fixed(skip))
    }

    /// The value, when it is a u32.
    pub fn u32(&self) -> Option<u32> {
        match self {
            Arg::Value(b) if b.len() == 4 => Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]])),
            _ => None,
        }
    }

    /// The value, when it is an f32.
    pub fn f32(&self) -> Option<f32> {
        self.u32().map(f32::from_bits)
    }
}

impl From<Extent> for Shape {
    fn from(e: Extent) -> Shape {
        Shape::line(e)
    }
}

/// A kernel struct's field as a scalar argument.
pub trait ToArg {
    fn arg(&self) -> Arg;
}

impl ToArg for usize {
    fn arg(&self) -> Arg {
        Arg::count(*self)
    }
}

impl ToArg for u32 {
    fn arg(&self) -> Arg {
        Arg::value(self)
    }
}

impl ToArg for f32 {
    fn arg(&self) -> Arg {
        Arg::value(self)
    }
}

impl ToArg for Extent {
    fn arg(&self) -> Arg {
        Arg::size(*self)
    }
}

impl Shape {
    /// A line of `e` threads, plain when known.
    pub fn line(e: Extent) -> Shape {
        match e {
            Extent::Fixed(n) => Shape::Line(n),
            e => Shape::Runtime(e),
        }
    }
}

/// How threads are laid out over the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, derive_more::From)]
pub enum Shape {
    /// One thread per element, in groups of 256.
    Line(usize),
    #[from(ignore)]
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
    /// A line whose length is read at launch.
    #[from(ignore)]
    Runtime(Extent),
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub args: Vec<Arg>,
    pub shape: Shape,
}

impl Launch {
    /// `body`'s arguments: buffers from `step`, scalars by name.
    pub fn of(
        body: &Body,
        step: &Step,
        overwrites: bool,
        scalar: &dyn Fn(&str) -> Arg,
        shape: Shape,
    ) -> Launch {
        let mut reads = 0..;
        let mut args = Vec::with_capacity(body.params.len());
        for p in &body.params {
            args.push(match p {
                Param::Buffer { writes: true, .. } if !overwrites => step.out(),
                Param::Buffer { .. } => step.read(reads.next().expect("reads")),
                Param::Scalar { name, .. } => scalar(name),
                _ => continue,
            });
        }
        Launch { args, shape }
    }

    /// One launch: the step's reads, its output, then `counts`.
    pub fn plain(step: &Step, counts: &[usize], shape: Shape) -> Launch {
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend(counts.iter().copied().map(Arg::count));
        Launch { args, shape }
    }

    /// The same launch with every runtime size filled in.
    pub fn bound(self, size: &dyn Fn(DimName) -> Option<u32>) -> Option<Launch> {
        let count = |e: &Extent| Some(e.value(size)? as u32);
        let mut args = Vec::with_capacity(self.args.len());
        for arg in self.args {
            args.push(match arg {
                Arg::Size(e) => Arg::count(count(&e)? as usize),
                Arg::Buffer(slot, e) => Arg::at(slot, e.value(size)?),
                fixed => fixed,
            });
        }
        let shape = match self.shape {
            Shape::Runtime(e) => Shape::Line(e.value(size)?),
            fixed => fixed,
        };
        Some(Launch { args, shape })
    }
}

/// Where a kernel's source comes from.
pub enum Source {
    /// Hand-written, with one text per backend.
    Fixed {
        metal: &'static str,
        cuda: &'static str,
        hip: &'static str,
        entry: &'static str,
    },
    /// A tree, rendered per dialect or run on the host.
    Made(Body),
}

impl Source {
    /// A tree under its own entry name.
    pub fn named(mut body: Body, entry: impl Into<String>) -> Source {
        body.entry = entry.into();
        Source::Made(body)
    }

    /// Changes whenever the kernel's text does.
    pub fn hash(&self) -> u64 {
        match self {
            Source::Fixed {
                metal,
                cuda,
                hip,
                entry,
            } => [metal, cuda, hip, entry]
                .iter()
                .fold(0, |h: u64, text| h.rotate_left(1) ^ hash(text)),
            Source::Made(body) => hash(&body.text().render(Dialect::Cuda)),
        }
    }
}

/// Collective kind on one resident lane, matching `collectives.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Collective {
    Sum = 0,
    Gather = 1,
    AllToAll = 2,
    Send = 3,
    Recv = 4,
    Fetch = 5,
}

/// Element kind for resident items and lanes, matching `core.h`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Elem {
    F32 = 0,
    Bf16 = 1,
    F16 = 2,
    E4M3 = 3,
    U32 = 4,
    Form = 5,
    Bytes = 6,
}

impl Elem {
    /// Element kind for a storage form name.
    pub fn of(name: &str) -> Self {
        match name {
            "f32" => Elem::F32,
            "bf16" => Elem::Bf16,
            "f16" => Elem::F16,
            "e4m3" => Elem::E4M3,
            "u32" => Elem::U32,
            "bytes" | "u8" => Elem::Bytes,
            _ => Elem::Form,
        }
    }

    /// Bytes per element, or 1 for packed forms.
    pub fn bytes(self) -> u32 {
        match self {
            Elem::F32 | Elem::U32 => 4,
            Elem::Bf16 | Elem::F16 => 2,
            Elem::E4M3 | Elem::Form | Elem::Bytes => 1,
        }
    }
}

/// Stable 32-bit identifier for a storage form name.
pub fn form_kind(name: &str) -> u32 {
    let mut h = 0x811c_9dc5u32;
    for &b in name.as_bytes() {
        h ^= u32::from(b);
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// One collective pool in a resident step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PoolSpec {
    pub depth: u32,
    pub one_way: bool,
    /// Member ranks, or empty for all ranks.
    pub group: Vec<u32>,
    pub send_bytes: u32,
    pub recv_bytes: u32,
}

/// One collective lane in a resident step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneSpec {
    pub op: Collective,
    pub elem: Elem,
    pub pool: u32,
    pub block: u32,
}

/// Role of one item in `tessera::table.items`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ItemKind {
    Input = 0,
    Output = 1,
    Scratch = 2,
}

/// Backing storage for one resident item.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ItemData {
    Slot(Slot),
    Scratch(usize),
}

/// One entry in `tessera::table.items`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ItemSpec {
    pub data: ItemData,
    pub kind: ItemKind,
    pub elem: Elem,
    pub dims: u32,
    pub shape: [u32; 4],
    pub share: [u32; 4],
    /// Index into `forms`, or `u32::MAX` if none.
    pub form: u32,
    pub forms: u32,
}

/// One packed or rotated weight in `tessera::table.forms`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FormSpec {
    pub codes: Slot,
    pub suh: Option<Slot>,
    pub svh: Option<Slot>,
    pub kind: u32,
    pub rows: u32,
    pub cols: u32,
}

/// One kept-row table in the `KV` convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeptSpec {
    pub index: u32,
    pub base: Slot,
    pub pages: Option<Slot>,
    pub pages_per_slot: u32,
    pub page_rows: u32,
    pub row_bytes: u32,
    pub elem: Elem,
    pub slots: u32,
}

/// One recurrent state buffer in the `STATE` convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HeldSpec {
    pub base: Slot,
    pub fresh: Option<Slot>,
    pub slot_bytes: u32,
    pub elem: Elem,
}

/// Token and sampling buffers in the `LLM` convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LlmSpec {
    pub rows: Extent,
    pub out_rows: Extent,
    pub vocab: u32,
    pub token: Option<Slot>,
    pub slot: Option<Slot>,
    pub position: Option<Slot>,
    pub out_row: Option<Slot>,
    pub temperature: Option<Slot>,
    pub top_k: Option<Slot>,
    pub seed: Option<Slot>,
    pub logits: Option<Slot>,
    pub sampled: Option<Slot>,
}

/// Optional conventions populated in `tessera::table.conventions`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Conventions {
    pub kv: Vec<KeptSpec>,
    pub llm: Option<LlmSpec>,
    pub state: Vec<HeldSpec>,
}

/// Persistent `tessera::table*` launch metadata for a step.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentSpec {
    pub threads: u32,
    pub share: u32,
    pub shared_bytes: u32,
    pub pools: Vec<PoolSpec>,
    pub lanes: Vec<LaneSpec>,
    pub items: Vec<ItemSpec>,
    pub forms: Vec<FormSpec>,
    pub values: Vec<Arg>,
    pub conventions: Conventions,
}

impl ResidentSpec {
    /// Resolves every launch value to a `u32`.
    pub fn bound_values(&self, size: &dyn Fn(DimName) -> Option<u32>) -> Option<Vec<u32>> {
        let mut out = Vec::with_capacity(self.values.len());
        for arg in &self.values {
            let word = match arg {
                Arg::Size(e) => e.value(size)? as u32,
                other => other.u32()?,
            };
            out.push(word);
        }
        Some(out)
    }
}
