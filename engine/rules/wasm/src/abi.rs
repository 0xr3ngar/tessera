//! The JSON a module and the host exchange.

use serde::{Deserialize, Serialize};

/// The version of this exchange the host reads.
pub(crate) const ABI: u32 = 1;

/// What `manifest()` returns.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Manifest {
    pub abi: u32,
    pub kernels: Vec<Text>,
    pub rules: Vec<Declared>,
}

/// One kernel's source, the same for every launch.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Text {
    pub name: String,
    pub entry: String,
    pub metal: String,
    pub cuda: String,
}

/// A rule, and the site kind it is asked at.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Declared {
    pub name: String,
    pub site: Kind,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Kind {
    Matmul,
}

/// What `propose` is given.
#[derive(Debug, Serialize)]
pub(crate) struct Ask<'a> {
    pub rule: &'a str,
    pub site: Site,
}

/// A site's sizes. Offers name reads by place.
#[derive(Debug, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Site {
    /// `x[m, k]` times `w[n, k]` transposed. Reads `[x, w]`.
    Matmul {
        m: usize,
        n: usize,
        k: usize,
        form: &'static str,
    },
}

/// One launch `propose` offers. `null` offers none.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Offer {
    pub kernel: String,
    /// Places in the site's reads, in kernel argument order.
    pub reads: Vec<usize>,
    /// Elements written.
    pub len: usize,
    /// u32 arguments after the output.
    #[serde(default)]
    pub counts: Vec<u32>,
    pub shape: Shape,
    /// Work units, and seconds per unit until measured.
    pub work: f64,
    pub guess: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub(crate) enum Shape {
    Line(usize),
    Grid {
        across: usize,
        down: usize,
        threads: usize,
    },
}
