//! Resident `tessera::table*` kernel metadata built from wasm offers.

use crate::abi::{CollectiveOp, Count, Offer, Place, Resident};
use crate::site::{Reads, arg, extent};
use crate::text::Fixed;
use tessera_device::Import;
use tessera_device::layout::{MAX_DEPTH, NONE, VALUES};
use tessera_lower::{
    Arg, Collective, Conventions, Draft, Elem, Extent, FormSpec, HeldSpec, ItemData, ItemKind,
    ItemSpec, KeptSpec, LaneSpec, LlmSpec, PoolSpec, Read, ResidentSpec, Slot, Step, Tree,
    form_kind,
};

const MAX_POOLS: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SlotRef {
    Read(usize),
    Output,
}

impl SlotRef {
    fn slot(self, step: &Step) -> Slot {
        match self {
            SlotRef::Read(i) => step.reads[i],
            SlotRef::Output => step.write,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ItemAt {
    Read(usize),
    Output,
    Scratch(usize),
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ItemPlan {
    at: ItemAt,
    kind: ItemKind,
    elem: Elem,
    dims: u32,
    shape: [u32; 4],
    share: [u32; 4],
    form: u32,
    forms: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FormPlan {
    codes: usize,
    suh: Option<usize>,
    svh: Option<usize>,
    kind: u32,
    rows: u32,
    cols: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct KeptPlan {
    index: u32,
    base: usize,
    slots: u32,
    context: u32,
    width: u32,
    page: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HeldPlan {
    base: SlotRef,
    fresh: Option<SlotRef>,
    slot_bytes: u32,
    elem: Elem,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct LlmPlan {
    rows: Extent,
    out_rows: Extent,
    vocab: u32,
    token: Option<SlotRef>,
    slot: Option<SlotRef>,
    position: Option<SlotRef>,
    out_row: Option<SlotRef>,
    temperature: Option<SlotRef>,
    top_k: Option<SlotRef>,
    seed: Option<SlotRef>,
    logits: Option<SlotRef>,
    sampled: Option<SlotRef>,
}

/// Lowered resident launch template stored on a wasm `Text` kernel.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TablePlan {
    pub threads: u32,
    share: u32,
    shared_bytes: u32,
    pools: Vec<PoolSpec>,
    lanes: Vec<LaneSpec>,
    items: Vec<ItemPlan>,
    forms: Vec<FormPlan>,
    values: Vec<Arg>,
    kv: Vec<KeptPlan>,
    llm: Option<LlmPlan>,
    state: Vec<HeldPlan>,
}

fn shape4(dims: &[usize]) -> (u32, [u32; 4]) {
    let mut out = [1u32; 4];
    let n = dims.len().min(4);
    for (slot, &d) in out[..n].iter_mut().zip(dims) {
        *slot = d as u32;
    }
    (n as u32, out)
}

fn matrix_dims(dims: &[usize]) -> (u32, u32) {
    match dims {
        [] => (1, 1),
        [cols] => (1, *cols as u32),
        [head @ .., cols] => (head.iter().product::<usize>() as u32, *cols as u32),
    }
}

fn collective_op(op: CollectiveOp) -> Collective {
    match op {
        CollectiveOp::Sum => Collective::Sum,
        CollectiveOp::Gather => Collective::Gather,
        CollectiveOp::AllToAll => Collective::AllToAll,
        CollectiveOp::Send => Collective::Send,
        CollectiveOp::Recv => Collective::Recv,
        CollectiveOp::Fetch => Collective::Fetch,
    }
}

/// Ensures `read` is in `reads` and returns its index.
fn intern_read(reads: &mut Vec<Read>, read: Read) -> usize {
    if let Some(i) = reads.iter().position(|r| *r == read) {
        return i;
    }
    reads.push(read);
    reads.len() - 1
}

/// Resolves a `Place` into a `Read` for `reads`.
fn place_read(place: Place, from: &Reads, made: usize) -> Option<Read> {
    match place {
        Place::Read(i) => Some(Read::Node(*from.nodes.get(i)?)),
        Place::Draft { draft } if draft < made => Some(Read::Draft(draft)),
        Place::Part { part: [i, 0] } => Some(Read::Node(*from.nodes.get(i)?)),
        Place::Part { part: [i, p] } if p > 0 => Some(Read::Node(*from.scales(i)?.get(p - 1)?)),
        Place::Kept { kept: i } => Some(Read::Kept(from.kept.get(i)?.index)),
        Place::Draft { .. } | Place::Part { .. } | Place::Out { .. } => None,
    }
}

/// Resolves a convention `Place` into a read index or output.
fn place_slot(place: Place, reads: &mut Vec<Read>, from: &Reads, made: usize) -> Option<SlotRef> {
    if let Place::Out { out: 0 } = place {
        return Some(SlotRef::Output);
    }
    let read = place_read(place, from, made)?;
    Some(SlotRef::Read(intern_read(reads, read)))
}

fn opt_slot(
    place: Option<Place>,
    reads: &mut Vec<Read>,
    from: &Reads,
    made: usize,
) -> Option<Option<SlotRef>> {
    match place {
        Some(p) => Some(Some(place_slot(p, reads, from, made)?)),
        None => Some(None),
    }
}

fn check_imports(res: &Resident, fixed: &Fixed) -> bool {
    if res.threads == 0 || res.pools.len() > MAX_POOLS {
        return false;
    }
    for p in &res.pools {
        if p.depth == 0 || p.depth as usize > MAX_DEPTH {
            return false;
        }
    }
    for l in &res.lanes {
        if l.pool as usize >= res.pools.len() {
            return false;
        }
    }
    if res.absorbs_sums > 0 && (res.pools.is_empty() || res.lanes.is_empty()) {
        return false;
    }
    if !fixed.from_file {
        return true;
    }
    let has = |want: Import| fixed.imports.contains(&want);
    let has_collective = fixed.imports.iter().any(|i| i.collective());
    if !res.conventions.kv.is_empty() && !has(Import::Kept) {
        return false;
    }
    if res.conventions.llm.is_some() && !has(Import::Llm) {
        return false;
    }
    if !res.conventions.state.is_empty() && !has(Import::State) {
        return false;
    }
    if (!res.lanes.is_empty() || res.absorbs_sums > 0) && !has_collective {
        return false;
    }
    true
}

/// Builds reads and a `TablePlan` for a resident offer.
pub(crate) fn build(
    offer: &Offer,
    res: &Resident,
    fixed: &Fixed,
    tree: &Tree,
    from: &Reads,
    drafts: &[Draft],
) -> Option<(Vec<Read>, TablePlan)> {
    if !check_imports(res, fixed) {
        return None;
    }
    let made = drafts.len();
    let mut reads = Vec::with_capacity(offer.reads.len());
    let mut items = Vec::with_capacity(offer.reads.len() + 1 + res.scratch.len());
    let mut forms = Vec::new();
    for &place in &offer.reads {
        let read = place_read(place, from, made)?;
        let read_idx = reads.len();
        reads.push(read);
        let item = match place {
            Place::Read(i) => {
                let t = from.nodes[i];
                let node = tree.node(t);
                let (dims, shape) = shape4(&node.shape);
                let share = tree.share(t).map_or([0; 4], |s| {
                    [s.axis, s.first as u32, s.count as u32, s.whole as u32]
                });
                let mut elem = Elem::of(node.form.name());
                let (mut form, mut form_count) = (NONE, 0u32);
                if elem == Elem::Form || from.scales(i).is_some() {
                    let (suh, svh) = match from.scales(i) {
                        Some([s0, s1]) => (
                            Some(intern_read(&mut reads, Read::Node(s0))),
                            Some(intern_read(&mut reads, Read::Node(s1))),
                        ),
                        None => (None, None),
                    };
                    let (rows, cols) = matrix_dims(&node.shape);
                    form = forms.len() as u32;
                    form_count = 1;
                    elem = Elem::Form;
                    forms.push(FormPlan {
                        codes: read_idx,
                        suh,
                        svh,
                        kind: form_kind(node.form.name()),
                        rows,
                        cols,
                    });
                }
                ItemPlan {
                    at: ItemAt::Read(read_idx),
                    kind: ItemKind::Input,
                    elem,
                    dims,
                    shape,
                    share,
                    form,
                    forms: form_count,
                }
            }
            Place::Part { part: [i, p] } => {
                let t = match p {
                    0 => from.nodes[i],
                    _ => from.scales(i)?[p - 1],
                };
                let node = tree.node(t);
                let (dims, shape) = shape4(&node.shape);
                ItemPlan {
                    at: ItemAt::Read(read_idx),
                    kind: ItemKind::Input,
                    elem: Elem::of(node.form.name()),
                    dims,
                    shape,
                    share: [0; 4],
                    form: NONE,
                    forms: 0,
                }
            }
            Place::Draft { draft } => ItemPlan {
                at: ItemAt::Read(read_idx),
                kind: ItemKind::Input,
                elem: Elem::F32,
                dims: 1,
                shape: [drafts[draft].len as u32, 1, 1, 1],
                share: [0; 4],
                form: NONE,
                forms: 0,
            },
            Place::Kept { kept: i } => {
                let k = from.kept[i];
                ItemPlan {
                    at: ItemAt::Read(read_idx),
                    kind: ItemKind::Input,
                    elem: Elem::F32,
                    dims: 3,
                    shape: [k.slots as u32, k.context as u32, k.width as u32, 1],
                    share: [0; 4],
                    form: NONE,
                    forms: 0,
                }
            }
            Place::Out { .. } => return None,
        };
        items.push(item);
    }
    let out_node = offer
        .writes
        .and_then(|o| from.outputs.get(o))
        .or_else(|| (from.outputs.len() == 1).then(|| &from.outputs[0]));
    let (out_elem, out_dims, out_shape) = match out_node {
        Some(&t) => {
            let node = tree.node(t);
            let (dims, shape) = shape4(&node.shape);
            (Elem::of(node.form.name()), dims, shape)
        }
        None => (Elem::F32, 1, [offer.len as u32, 1, 1, 1]),
    };
    items.push(ItemPlan {
        at: ItemAt::Output,
        kind: ItemKind::Output,
        elem: out_elem,
        dims: out_dims,
        shape: out_shape,
        share: [0; 4],
        form: NONE,
        forms: 0,
    });
    for sc in &res.scratch {
        let mut shape = [1u32; 4];
        let dims = if sc.shape.is_empty() {
            shape[0] = sc.bytes as u32;
            1
        } else {
            let n = sc.shape.len().min(4);
            shape[..n].copy_from_slice(&sc.shape[..n]);
            n as u32
        };
        items.push(ItemPlan {
            at: ItemAt::Scratch(sc.bytes),
            kind: ItemKind::Scratch,
            elem: Elem::of(&sc.elem),
            dims,
            shape,
            share: [0; 4],
            form: NONE,
            forms: 0,
        });
    }
    let mut kv = Vec::with_capacity(res.conventions.kv.len());
    for &ki in &res.conventions.kv {
        let k = *from.kept.get(ki)?;
        let base = intern_read(&mut reads, Read::Kept(k.index));
        kv.push(KeptPlan {
            index: k.index,
            base,
            slots: k.slots as u32,
            context: k.context as u32,
            width: k.width as u32,
            page: tree.kept.page().map(|p| p as u32),
        });
    }
    let llm = match &res.conventions.llm {
        Some(l) => Some(LlmPlan {
            rows: extent(tree, &from.nodes, l.rows)?,
            out_rows: extent(tree, &from.nodes, l.out_rows)?,
            vocab: l.vocab,
            token: opt_slot(l.token, &mut reads, from, made)?,
            slot: opt_slot(l.slot, &mut reads, from, made)?,
            position: opt_slot(l.position, &mut reads, from, made)?,
            out_row: opt_slot(l.out_row, &mut reads, from, made)?,
            temperature: opt_slot(l.temperature, &mut reads, from, made)?,
            top_k: opt_slot(l.top_k, &mut reads, from, made)?,
            seed: opt_slot(l.seed, &mut reads, from, made)?,
            logits: opt_slot(l.logits, &mut reads, from, made)?,
            sampled: opt_slot(l.sampled, &mut reads, from, made)?,
        }),
        None => None,
    };
    let mut state = Vec::with_capacity(res.conventions.state.len());
    for st in &res.conventions.state {
        state.push(HeldPlan {
            base: place_slot(st.base, &mut reads, from, made)?,
            fresh: opt_slot(st.fresh, &mut reads, from, made)?,
            slot_bytes: st.slot_bytes,
            elem: Elem::of(&st.elem),
        });
    }
    let raw_values: &[Count] = match res.values.is_empty() {
        true => &offer.counts,
        false => &res.values,
    };
    if raw_values.len() > VALUES {
        return None;
    }
    let mut values = Vec::with_capacity(raw_values.len());
    for &c in raw_values {
        values.push(arg(tree, &from.nodes, c)?);
    }
    let pools = res
        .pools
        .iter()
        .map(|p| PoolSpec {
            depth: p.depth,
            one_way: p.one_way,
            group: p.group.clone(),
            send_bytes: p.send_bytes,
            recv_bytes: p.recv_bytes,
        })
        .collect();
    let lanes = res
        .lanes
        .iter()
        .map(|l| LaneSpec {
            op: collective_op(l.op),
            elem: Elem::of(&l.elem),
            pool: l.pool,
            block: l.block,
        })
        .collect();
    let plan = TablePlan {
        threads: res.threads,
        share: res.share,
        shared_bytes: res.shared_bytes,
        pools,
        lanes,
        items,
        forms,
        values,
        kv,
        llm,
        state,
    };
    Some((reads, plan))
}

impl TablePlan {
    /// Materializes `ResidentSpec` for lowered `step`.
    pub(crate) fn spec(&self, step: &Step) -> ResidentSpec {
        let items = self
            .items
            .iter()
            .map(|i| ItemSpec {
                data: match i.at {
                    ItemAt::Read(r) => ItemData::Slot(step.reads[r]),
                    ItemAt::Output => ItemData::Slot(step.write),
                    ItemAt::Scratch(bytes) => ItemData::Scratch(bytes),
                },
                kind: i.kind,
                elem: i.elem,
                dims: i.dims,
                shape: i.shape,
                share: i.share,
                form: i.form,
                forms: i.forms,
            })
            .collect();
        let forms = self
            .forms
            .iter()
            .map(|f| FormSpec {
                codes: step.reads[f.codes],
                suh: f.suh.map(|r| step.reads[r]),
                svh: f.svh.map(|r| step.reads[r]),
                kind: f.kind,
                rows: f.rows,
                cols: f.cols,
            })
            .collect();
        let kv = self
            .kv
            .iter()
            .map(|k| {
                let base = step.reads[k.base];
                let page_rows = k.page.unwrap_or(k.context);
                let pages_per_slot = match step.table {
                    Some(t) => (t.len / k.slots.max(1) as usize) as u32,
                    None => 1,
                };
                let elem = Elem::of(base.form.name());
                let row_bytes = base.form.bytes(k.width as usize) as u32;
                KeptSpec {
                    index: k.index,
                    base,
                    pages: step.table,
                    pages_per_slot,
                    page_rows,
                    row_bytes,
                    elem,
                    slots: k.slots,
                }
            })
            .collect();
        let llm = self.llm.map(|l| LlmSpec {
            rows: l.rows,
            out_rows: l.out_rows,
            vocab: l.vocab,
            token: l.token.map(|s| s.slot(step)),
            slot: l.slot.map(|s| s.slot(step)),
            position: l.position.map(|s| s.slot(step)),
            out_row: l.out_row.map(|s| s.slot(step)),
            temperature: l.temperature.map(|s| s.slot(step)),
            top_k: l.top_k.map(|s| s.slot(step)),
            seed: l.seed.map(|s| s.slot(step)),
            logits: l.logits.map(|s| s.slot(step)),
            sampled: l.sampled.map(|s| s.slot(step)),
        });
        let state = self
            .state
            .iter()
            .map(|h| HeldSpec {
                base: h.base.slot(step),
                fresh: h.fresh.map(|s| s.slot(step)),
                slot_bytes: h.slot_bytes,
                elem: h.elem,
            })
            .collect();
        ResidentSpec {
            threads: self.threads,
            share: self.share,
            shared_bytes: self.shared_bytes,
            pools: self.pools.clone(),
            lanes: self.lanes.clone(),
            items,
            forms,
            values: self.values.clone(),
            conventions: Conventions { kv, llm, state },
        }
    }
}
