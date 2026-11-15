//! Small graphs that reach every rule, at many sizes.

use crate::measure;
use graph::DimName::{self, *};
use lower::{Device, Error, Settings, Step, Tensor, View, lower};
use run::{Run, open};
use testkit::Noise;
use testkit::graphs::{attention, linear, rotation, routed, rows, table};
use trace::{Dim, Graph, TensorType};

/// A timed plan holds at most this many elements.
const ROOM: usize = 750_000_000;

type Sizes = Vec<(DimName, u64)>;

/// The grid times every rule at these sizes.
pub struct Grid {
    /// Row counts, for everything that works on rows.
    pub rows: Vec<u64>,
    /// Hidden and inner widths.
    pub widths: Vec<u64>,
    pub vocabs: Vec<u64>,
    /// Heads, key heads and head width.
    pub heads: Vec<[u64; 3]>,
    /// Queries and keys of one attention.
    pub spans: Vec<[u64; 2]>,
}

/// A graph, with each size to time it at.
struct Case {
    g: Graph,
    sizes: Vec<Sizes>,
}

/// Every name bounded far past any size the grid uses.
fn bounded(names: &[DimName]) -> Vec<Dim> {
    names.iter().map(|n| Dim::bounded(*n, 1 << 20)).collect()
}

/// Noise for `ty` as one row. Ids stay below eight.
fn fill(g: &Graph, ty: &TensorType, sizes: &Sizes, noise: &mut Noise) -> Result<Tensor, Error> {
    let unsized_shape = || Error::Size(ty.shape.to_string());
    testkit::fill(g, ty, sizes, noise).ok_or_else(unsized_shape)
}

impl Case {
    /// Times this case at each of its sizes that fits.
    fn run(
        &self,
        device: &mut Device,
        noise: &mut Noise,
        settings: &Settings,
    ) -> Result<Vec<(Step, f64)>, Error> {
        let mut runner: Option<Box<dyn Run>> = None;
        let mut added = Vec::new();
        for sizes in &self.sizes {
            let plan = lower(&self.g, 0, sizes, &Device::guess(settings))?;
            if plan.buffers.iter().sum::<usize>() > ROOM {
                continue;
            }
            let runner = match &mut runner {
                Some(runner) => runner,
                None => runner.insert(self.open(plan, noise, settings)?),
            };
            let ports = self.g.entries[0].inputs.iter();
            let fed: Result<Vec<Tensor>, Error> =
                ports.map(|p| fill(&self.g, &p.ty, sizes, noise)).collect();
            let fed = fed?;
            let feeds: Vec<View> = fed.iter().map(Tensor::view).collect();
            added.extend(measure(
                &mut **runner,
                (&self.g, 0),
                sizes,
                &feeds,
                device,
                settings,
            )?);
        }
        Ok(added)
    }

    /// A runner with noise for weights.
    fn open(
        &self,
        plan: lower::Plan,
        noise: &mut Noise,
        settings: &Settings,
    ) -> Result<Box<dyn Run>, Error> {
        let mut runner = open(plan, settings)?;
        for (at, w) in self.g.weights.iter().enumerate() {
            let data = fill(&self.g, &w.ty, &Vec::new(), noise)?;
            runner.load(at as u32, data.view())?;
        }
        Ok(runner)
    }
}

impl Grid {
    /// Sizes from one decoded row to a long prompt.
    pub fn full() -> Self {
        Self {
            rows: vec![1, 4, 16, 64, 256, 1024],
            widths: vec![256, 1024, 4096],
            vocabs: vec![32_768, 131_072],
            heads: vec![[16, 8, 128], [8, 8, 64]],
            spans: vec![
                [1, 16],
                [1, 128],
                [1, 1024],
                [16, 16],
                [128, 128],
                [512, 512],
            ],
        }
    }

    /// Enough to reach every rule, in well under a second.
    pub fn tiny() -> Self {
        Self {
            rows: vec![1, 3],
            widths: vec![16],
            vocabs: vec![300],
            heads: vec![[4, 2, 16]],
            spans: vec![[1, 5], [4, 4]],
        }
    }

    fn cases(&self) -> Vec<Case> {
        let by_rows = |g: Graph| {
            let sizes = self.rows.iter().map(|r| vec![(Rows, *r)]).collect();
            Case { g, sizes }
        };
        let (one, three) = (bounded(&[Rows]), bounded(&[Seqs, Rows, Context]));
        let mut cases = Vec::new();
        for &k in &self.widths {
            let outs = self.widths.iter().chain(&self.vocabs);
            cases.extend(outs.map(|n| by_rows(linear(&one, k, *n))));
        }
        cases.push(by_rows(rotation(&one)));
        cases.extend(
            self.widths
                .iter()
                .map(|&k| by_rows(routed(&one, [8, k, k]))),
        );
        for &width in self.widths.iter().chain(&self.vocabs) {
            cases.push(by_rows(rows(&one, width)));
            cases.extend((self.vocabs.iter()).map(|v| by_rows(table(&one, *v, width))));
        }
        for &heads in &self.heads {
            let span =
                |[queries, keys]: &[u64; 2]| vec![(Seqs, 1), (Rows, *queries), (Context, *keys)];
            let sizes = self.spans.iter().map(span).collect();
            let g = attention(&three, heads);
            cases.push(Case { g, sizes });
        }
        cases
    }

    /// Times every case on this GPU. Returns the launches timed.
    pub fn run(&self, device: &mut Device, settings: &Settings) -> Result<Vec<(Step, f64)>, Error> {
        let mut noise = Noise(17);
        let mut added = Vec::new();
        for case in self.cases() {
            added.extend(case.run(device, &mut noise, settings)?);
        }
        Ok(added)
    }
}
