//! The module's side of its JSON exchange with the host.

use serde::{Deserialize, Serialize};

/// The output of `manifest()`.
#[derive(Debug, Serialize)]
pub struct Manifest {
    pub abi: u32,
    pub kernels: Vec<Kernel>,
    pub rules: Vec<Rule>,
    /// Whether the module exports `graph` and `tables`.
    pub graph: bool,
    /// Recipe files that kernels from files read.
    pub files: Vec<File>,
}

/// One kernel's source, the same for every launch.
#[derive(Debug, Serialize)]
pub struct Kernel {
    pub name: &'static str,
    pub entry: &'static str,
    pub metal: &'static str,
    pub cuda: &'static str,
    /// A recipe file the host flattens into the texts.
    pub file: &'static str,
}

/// A kernel with CUDA text alone.
pub fn kernel(name: &'static str, entry: &'static str, cuda: &'static str) -> Kernel {
    Kernel {
        name,
        entry,
        metal: "",
        cuda,
        file: "",
    }
}

/// A kernel from recipe file `file`. It imports tessera headers.
pub fn imports(name: &'static str, entry: &'static str, file: &'static str) -> Kernel {
    Kernel {
        name,
        entry,
        metal: "",
        cuda: "",
        file,
    }
}

/// One recipe file, by its path inside the recipe.
#[derive(Debug, Serialize)]
pub struct File {
    pub path: &'static str,
    pub text: &'static str,
}

/// A rule, and the site kind it is asked at.
#[derive(Debug, Serialize)]
pub struct Rule {
    pub name: &'static str,
    pub site: Kind,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Matmul,
    Mlp,
    Moe,
    Attention,
    /// One decoder layer.
    Layer,
    /// Embedding to logits.
    Model,
}

/// The input of `propose`.
#[derive(Debug, Deserialize, Serialize)]
pub struct Ask {
    pub rule: String,
    pub site: Site,
}

/// A site's sizes. Offers name reads by place.
#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Site {
    /// `x[m, k]` times `w[n, k]` transposed. Reads `[x, w]`.
    Matmul {
        m: usize,
        n: usize,
        k: usize,
        form: String,
    },
    /// A whole region. Reads are its inputs, in order.
    Region {
        kind: Kind,
        inputs: Vec<Input>,
        /// Elements per output. Entries name the output they write.
        outputs: Vec<usize>,
        kept: Vec<Kept>,
        /// Elements per sum across ranks, in order. Offers place each.
        sums: Vec<usize>,
    },
}

/// Kept rows a region's launches append to.
#[derive(Debug, Deserialize, Serialize)]
pub struct Kept {
    pub index: u32,
    /// Slots, rows per slot, and elements per row.
    pub slots: usize,
    pub context: usize,
    pub width: usize,
    /// The row layout, its rows per page, and its scalars.
    pub layout: String,
    pub page: Option<usize>,
    pub params: Vec<String>,
    pub values: Vec<u32>,
    /// Input places of the slot and position ids.
    pub by: [Option<usize>; 2],
}

/// One value a region reads.
#[derive(Debug, Deserialize, Serialize)]
pub struct Input {
    pub shape: Vec<usize>,
    /// Per axis: the dimension read at launch, if any.
    pub dims: Vec<Option<String>>,
    pub form: String,
    /// A weight's dotted path.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weight: Option<String>,
    /// This rank's part: axis, first, count, whole.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub share: Option<[usize; 4]>,
    /// A rotated weight: the lengths of `suh` and `svh`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parts: Option<[usize; 2]>,
}

/// A site read, or an earlier offer's output.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum Place {
    Read(usize),
    Draft {
        draft: usize,
    },
    Kept {
        kept: usize,
    },
    /// An input, then 0, 1 or 2: codes, `suh`, `svh`.
    Part {
        part: [usize; 2],
    },
    /// This launch's own output, for convention bindings.
    Out {
        out: usize,
    },
}

/// A u32 argument, fixed or read at launch.
#[derive(Clone, Debug, Serialize)]
#[serde(untagged)]
pub enum Count {
    Fixed(u32),
    Dim { dim: String, mul: usize },
}

/// One entry `propose` offers. `null` offers none.
#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum Entry {
    Launch(Offer),
    /// Sums an earlier draft across ranks. Later entries read it.
    AllReduce {
        all_reduce: usize,
        #[serde(skip_serializing_if = "Option::is_none")]
        writes: Option<usize>,
    },
}

impl From<Offer> for Entry {
    fn from(offer: Offer) -> Self {
        Entry::Launch(offer)
    }
}

/// One launch. Its reads are all it depends on.
#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Offer<K = &'static str, C = Count> {
    pub kernel: K,
    /// In kernel argument order.
    pub reads: Vec<Place>,
    /// Elements written.
    pub len: usize,
    /// u32 arguments after the output.
    #[serde(default = "Vec::new")]
    pub counts: Vec<C>,
    #[serde(default = "default_shape")]
    pub shape: Shape,
    /// Work units, and seconds per unit until measured.
    pub work: f64,
    pub guess: f64,
    /// Output index it writes. With one output, the last entry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub writes: Option<usize>,
    /// Persistent `tessera::table*` launch settings, if resident.
    #[serde(default = "default_none", skip_serializing_if = "Option::is_none")]
    pub resident: Option<Box<Resident<C>>>,
}

fn default_none<T>() -> Option<T> {
    None
}

fn default_conventions<C>() -> Conventions<C> {
    Conventions::default()
}

fn default_shape() -> Shape {
    Shape::Line(1)
}

fn default_one() -> u32 {
    1
}

fn default_two() -> u32 {
    2
}

fn default_f32() -> String {
    "f32".into()
}

fn default_bytes() -> String {
    "bytes".into()
}

/// One collective pool for a resident kernel.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Pool {
    #[serde(default = "default_two")]
    pub depth: u32,
    #[serde(default)]
    pub one_way: bool,
    /// Member ranks, or empty for every rank.
    #[serde(default)]
    pub group: Vec<u32>,
    pub send_bytes: u32,
    pub recv_bytes: u32,
}

/// Collective operation kind on a resident lane.
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Op {
    Sum,
    Gather,
    AllToAll,
    Send,
    Recv,
    Fetch,
}

/// One collective lane for a resident kernel.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Lane {
    pub op: Op,
    #[serde(default = "default_f32")]
    pub elem: String,
    #[serde(default)]
    pub pool: u32,
    #[serde(default)]
    pub block: u32,
}

/// A scratch item in `tessera::table.items`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Scratch {
    pub bytes: usize,
    #[serde(default = "default_bytes")]
    pub elem: String,
    #[serde(default)]
    pub shape: Vec<u32>,
}

/// Token and sampling bindings for the `LLM` convention.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Llm<C = Count> {
    pub rows: C,
    pub out_rows: C,
    #[serde(default)]
    pub vocab: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub position: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub out_row: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logits: Option<Place>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sampled: Option<Place>,
}

/// One recurrent state table for the `STATE` convention.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct State {
    pub base: Place,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fresh: Option<Place>,
    pub slot_bytes: u32,
    #[serde(default = "default_f32")]
    pub elem: String,
}

/// Conventions requested in `tessera::table.conventions`.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Conventions<C = Count> {
    /// Kept indices from `site.kept` exposed in `kv_tables`.
    #[serde(default)]
    pub kv: Vec<usize>,
    #[serde(default = "default_none", skip_serializing_if = "Option::is_none")]
    pub llm: Option<Llm<C>>,
    #[serde(default)]
    pub state: Vec<State>,
}

impl<C> Default for Conventions<C> {
    fn default() -> Self {
        Self {
            kv: Vec::new(),
            llm: None,
            state: Vec::new(),
        }
    }
}

/// Persistent `tessera::table*` launch settings on an offer.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Resident<C = Count> {
    pub threads: u32,
    #[serde(default = "default_one")]
    pub share: u32,
    #[serde(default)]
    pub shared_bytes: u32,
    #[serde(default)]
    pub pools: Vec<Pool>,
    #[serde(default)]
    pub lanes: Vec<Lane>,
    #[serde(default)]
    pub scratch: Vec<Scratch>,
    #[serde(default = "default_conventions")]
    pub conventions: Conventions<C>,
    /// Launch values written into `launch.values`.
    #[serde(default = "Vec::new")]
    pub values: Vec<C>,
    /// Region `sums` absorbed internally by this kernel.
    #[serde(default)]
    pub absorbs_sums: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub enum Shape {
    Line(usize),
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
}
