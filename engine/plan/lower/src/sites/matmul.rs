use super::{Chain, Pattern};
use crate::form::Form;
use crate::kernels::MapStage;
use crate::rule::Read;
use crate::tree::{T, Term, Tree};
use graph::{Op, Select, View};

/// Each output row reads one matrix of a stored set.
#[derive(Clone, Copy, Debug)]
pub struct Picked {
    /// The matrix index per output row.
    pub ids: T,
    /// Matrices in the set, and the rows each holds.
    pub count: usize,
    pub stride: usize,
    /// The first row read of each matrix.
    pub offset: usize,
    /// Output rows that share one `x` row.
    pub repeat: usize,
}

/// `x[m, k]` times `w[n, k]` transposed, `w` stored as `form`.
#[derive(Clone, Copy, Debug)]
pub struct Matmul {
    pub x: T,
    /// The matrix, or the set `picked` reads.
    pub w: T,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub form: &'static dyn Form,
    pub picked: Option<Picked>,
}

impl Pattern for Matmul {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        if node.term != Term::Op(Op::Matmul) {
            return None;
        }
        let (x, w) = (node.args[0], node.args[1]);
        let k = *tree.node(w).shape.last()?;
        if let [n, _] = tree.node(w).shape[..] {
            let (m, form) = (tree.elements(x) / k, tree.node(w).form);
            return Some(Self {
                x,
                w,
                m,
                n,
                k,
                form,
                picked: None,
            });
        }
        let (set, ids, offset) = gathered(tree, w)?;
        let [count, stride, _] = tree.node(set).shape[..] else {
            return None;
        };
        let (m, n) = (
            tree.elements(ids),
            tree.elements(w) / tree.elements(ids) / k,
        );
        Some(Self {
            x,
            w: set,
            m,
            n,
            k,
            form: tree.node(set).form,
            picked: Some(Picked {
                ids,
                count,
                stride,
                offset,
                repeat: m / (tree.elements(x) / k),
            }),
        })
    }
}

/// The stored set, ids and row offset gathered at `w`.
fn gathered(tree: &Tree, w: T) -> Option<(T, T, usize)> {
    let gather = tree.node(w);
    if gather.term != Term::Op(Op::Select(Select::Gather)) {
        return None;
    }
    let (mut set, ids) = (gather.args[0], gather.args[1]);
    let mut offset = 0;
    if let Term::Op(Op::View(View::Slice { axis: 1, start, .. })) = &tree.node(set).term {
        let size = |name| tree.sizes.iter().find(|(n, _)| *n == name).map(|(_, n)| *n);
        offset = start.value(&tree.g.dims, &size)? as usize;
        set = tree.node(set).args[0];
    }
    tree.leaf(set).then_some((set, ids, offset))
}

impl Matmul {
    /// The rows, the weights, then any pick ids.
    pub fn reads(&self) -> Vec<Read> {
        let ids = self.picked.map(|p| Read::Node(p.ids));
        [Read::Node(self.x), Read::Node(self.w)]
            .into_iter()
            .chain(ids)
            .collect()
    }
}

/// Elementwise stages a sum runs through before it is stored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct After {
    pub stages: Vec<MapStage>,
    /// Which of the chain's reads the sum stands in for.
    pub sum: usize,
}

impl After {
    /// Operands the stages read besides the sum.
    pub fn others(&self) -> usize {
        crate::kernels::operands(&self.stages)
    }
}

/// A matmul with the stages that finish it, if any.
#[derive(Clone, Debug)]
pub struct Finished {
    pub product: Matmul,
    pub after: After,
    /// The stages' other operands, in their order.
    pub others: Vec<Read>,
}

impl Finished {
    /// The product's reads, then the stages' other operands.
    pub fn reads(&self) -> Vec<Read> {
        let mut reads = self.product.reads();
        reads.extend(self.others.iter().copied());
        reads
    }
}

impl Pattern for Finished {
    /// The matmul at `at`, or the one its chain finishes.
    fn find(tree: &Tree, at: T) -> Option<Self> {
        if let Some(product) = tree.site::<Matmul>(at) {
            return Some(Self {
                product,
                after: After::default(),
                others: Vec::new(),
            });
        }
        let Chain {
            stages,
            reads,
            spans,
            ..
        } = tree.site(at)?;
        if spans.iter().any(Option::is_some) {
            return None;
        }
        // A product read elsewhere is stored on its own.
        let lone = |r: &T| tree.site::<Matmul>(*r).filter(|_| !tree.root(*r));
        let sum = reads.iter().position(|r| lone(r).is_some())?;
        let others = reads.iter().filter(|r| **r != reads[sum]);
        Some(Self {
            product: lone(&reads[sum])?,
            after: After { stages, sum },
            others: others.map(|r| Read::Node(*r)).collect(),
        })
    }
}
