//! Each rule's plans, and its launches against the CPU interpreter.

use kernels as _;

use frontend_settings::read as settings;
use graph::DimName::{Context, HeadDim, Heads, Hidden, Rows, Seqs, Vocab};
use graph::{DType, Dim, DimName, Graph, NodeId, TensorType, verify};
use interp::{Tensor, run};
use lower::form::{BF16_FORM, F32_FORM};
use lower::{Device, Limits, Pins, Plan, Platform, View, cut, lower, rules};
use run::open;
use testkit::{Noise, fill, max_abs_diff};
use trace::{Act, Embed, EntryKind, Mask, Model, Norm, Pairs, Port, Role, Weights, declare};

const SIZES: [(DimName, u64); 3] = [(Seqs, 2), (Rows, 3), (Context, 5)];

#[derive(Weights)]
struct Scale {
    #[shape(Hidden)]
    norm: Norm,
}

#[derive(Weights)]
struct Table {
    #[shape(Vocab, Hidden)]
    embed: Embed,
}

fn floats<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::F32);
    let role = Role::Logits;
    Port { role, ty }
}

fn ids<const N: usize>(shape: [DimName; N]) -> Port {
    let ty = TensorType::new(shape, DType::U32);
    let role = Role::Tokens;
    Port { role, ty }
}

/// Norms, maps, centering, a unit sum, softmax, argmax and mean.
fn norms() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Hidden, 32),
    ]);
    let w: Scale = declare(&m);
    let port = || floats([Seqs, Rows, Hidden]);
    let [x, y] = m.entry(EntryKind::Forward, [port(), port()]);
    let gated = x.rms_norm(&w.norm, 1e-6).act(Act::Silu).mul(&y.rms(1e-6));
    let sum = y.add(&gated).center();
    let top = sum.act(Act::Sigmoid).unit_sum().softmax().top_k(1u64);
    m.finish([(Role::Tokens, &top), (Role::Logits, &sum.mean(0))])
}

/// Rotated queries over more keys, then a softmax and mean.
fn attention() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::bounded(Context, 16),
        Dim::fixed(Heads, 2),
        Dim::fixed(HeadDim, 16),
    ]);
    let port = |rows| floats([Seqs, rows, Heads, HeadDim]);
    let ports = [port(Rows), port(Context), port(Context), ids([Seqs, Rows])];
    let [q, k, v, at] = m.entry(EntryKind::Forward, ports);
    let q = q.rope(&at, 1e4, Pairs::Halves);
    let out = m.attend(Mask::Causal, 0.5, [&q, &k, &v], None).softmax();
    m.finish([(Role::Logits, &out), (Role::Logits, &out.mean(1))])
}

/// Each row's likeliest entry, over more entries than threads.
fn likeliest() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 8),
        Dim::bounded(Rows, 8),
        Dim::fixed(Vocab, 600),
    ]);
    let [logits] = m.entry(EntryKind::Forward, [floats([Seqs, Rows, Vocab])]);
    m.finish([(Role::Tokens, &logits.top_k(1u64))])
}

/// Rows of 256, rotated in two blocks.
fn rotated() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 8),
        Dim::bounded(Rows, 8),
        Dim::fixed(Hidden, 256),
    ]);
    let [x] = m.entry(EntryKind::Forward, [floats([Seqs, Rows, Hidden])]);
    m.finish([(Role::Logits, &x.hadamard(128))])
}

/// Rows read from a table.
fn table() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Vocab, 32),
        Dim::fixed(Hidden, 16),
    ]);
    let w: Table = declare(&m);
    let [tokens] = m.entry(EntryKind::Forward, [ids([Seqs, Rows])]);
    m.finish([(Role::Logits, &m.embed(&tokens, &w.embed))])
}

/// Runs entry 0 both ways. Returns the largest difference.
fn gap(g: &Graph, inputs: &[Tensor], device: &Device) -> f32 {
    let want = run(g, &[], 0, inputs).unwrap();
    let mut runner = open(lower(g, 0, &SIZES, device).unwrap(), &settings()).unwrap();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let got = runner.run(&feeds).unwrap().remove(0);
    if want[0].dtype() != DType::U32 {
        return max_abs_diff(got.floats().unwrap(), want[0].floats().unwrap());
    }
    // Indices must match exactly.
    let same = got.indices().unwrap() == want[0].indices().unwrap();
    if same { 0.0 } else { f32::INFINITY }
}

#[test]
fn each_rule_matches_cpu() {
    let mut noise = Noise(7);
    let graphs = [norms(), attention(), likeliest(), rotated(), table()];
    let ours: Vec<&lower::Rule> = rules()
        .into_iter()
        .filter(|r| r.module.starts_with("kernels::"))
        .collect();
    let mut used: Vec<&str> = Vec::new();
    for (g, rule) in graphs.iter().flat_map(|g| ours.iter().map(move |r| (g, r))) {
        let device = Device::guess(&settings()).favoring(rule.name);
        let plan = lower(g, 0, &SIZES, &device).unwrap();
        let mut seen: Vec<&[NodeId]> = Vec::new();
        for step in &plan.steps {
            used.extend((step.rule == rule.name).then_some(rule.name));
            if seen.contains(&&step.covers[..]) {
                continue;
            }
            seen.push(&step.covers);
            // The nodes one rule covered, as their own graph.
            let sub = cut(g, &step.covers);
            verify(&sub).unwrap();
            let ports = &sub.entries[0].inputs;
            let inputs: Vec<Tensor> = ports
                .iter()
                .map(|p| fill(&sub, &p.ty, &SIZES, &mut noise).unwrap())
                .collect();
            let off = gap(&sub, &inputs, &device);
            assert!(off < 1e-4, "{:?} is off by {off:e}", step.kernel);
        }
    }
    // A new rule needs a graph here and in calibrate.
    for rule in ours {
        // Hand-written kernels wait for a compiler.
        if settings().gpu().is_none() && rule.name == "dense" {
            continue;
        }
        assert!(
            used.contains(&rule.name),
            "no graph exercises {}",
            rule.name
        );
    }
}

/// Two rotations by one set of positions, then a map.
fn rotations() -> Graph {
    let m = Model::new(vec![
        Dim::bounded(Seqs, 4),
        Dim::bounded(Rows, 8),
        Dim::fixed(Heads, 2),
        Dim::fixed(HeadDim, 16),
    ]);
    let port = || floats([Seqs, Rows, Heads, HeadDim]);
    let [x, y, at] = m.entry(EntryKind::Forward, [port(), port(), ids([Seqs, Rows])]);
    let turn = |t: &trace::Tensor| t.rope(&at, 1e4, Pairs::Halves);
    m.finish([(Role::Logits, &turn(&x).add(&turn(&y)).act(Act::Silu))])
}

fn count(plan: &Plan, kernel: &str) -> usize {
    let named = |s: &&lower::Step| s.kernel.name() == kernel;
    plan.steps.iter().filter(named).count()
}

fn names(plan: &Plan) -> Vec<&'static str> {
    plan.steps.iter().map(|s| s.kernel.name()).collect()
}

#[test]
fn cost_picks_attention() {
    let g = attention();
    let guess = Device::guess(&settings());
    // Dense attention is unproven, so a guess streams.
    let stream = lower(&g, 0, &SIZES, &guess).unwrap();
    let streams = count(&stream, "attend") + count(&stream, "flash");
    assert_eq!((streams, count(&stream, "scores")), (1, 0));
    // The host cannot run the hand-written dense gemm.
    if settings().gpu().is_none() {
        return;
    }
    let dense = lower(&g, 0, &SIZES, &guess.favoring("dense")).unwrap();
    assert_eq!((count(&dense, "attend"), count(&dense, "join")), (0, 0));
    let parts = ["scores", "causal", "values"].map(|k| count(&dense, k));
    assert_eq!(parts, [1, 1, 1]);
    let causal = dense.steps.iter().find(|s| s.kernel.name() == "causal");
    let causal = causal.unwrap();
    assert_eq!(causal.write.buffer, causal.reads[0].buffer, "in place");
}

#[test]
fn rotations_share_angles() {
    let plan = lower(&rotations(), 0, &SIZES, &Device::guess(&settings())).unwrap();
    assert_eq!((count(&plan, "angles"), count(&plan, "rope")), (1, 2));
}

#[test]
fn favoring_stays_local() {
    // No rule competes with rope, so favoring it changes nothing.
    let g = rotations();
    let plain = lower(&g, 0, &SIZES, &Device::guess(&settings())).unwrap();
    let favored = Device::guess(&settings()).favoring("rope");
    assert!(
        plain == lower(&g, 0, &SIZES, &favored).unwrap(),
        "a favored rule skewed its readers' choice"
    );
}

#[test]
fn trips_are_counted() {
    // Walking a row opens a line every sixteen floats.
    let norm = kernels::RmsNorm {
        rows: 4,
        width: 64,
        eps: 1e-5,
        weighted: true,
    };
    let trip = lower::Kernel::load(&norm, 64).unwrap().trip;
    assert_eq!((trip.loads, trip.stores), (3.0, 1.0));
    assert!((trip.fresh - 1.0 / 16.0).abs() < 1e-9, "{trip:?}");
    assert_eq!(trip.special, 0.0, "the root lies outside the loops");
}

#[test]
fn limits_bar_kernels() {
    let g = norms();
    let guess = Device::guess(&settings()).favoring("lane_norm");
    assert!(names(&lower(&g, 0, &SIZES, &guess).unwrap()).contains(&"lane_norm"));
    // A 64-lane device runs none of the 32-lane kernels.
    let wide = Limits {
        lanes: 64,
        ..Limits::assumed()
    };
    let narrow = guess.on(Platform::assumed(wide));
    let names = names(&lower(&g, 0, &SIZES, &narrow).unwrap());
    assert!(names.contains(&"rms_norm") && !names.contains(&"lane_norm"));
}

#[test]
fn tables_keep_forms() {
    let g = table();
    let guess = Device::guess(&settings());
    let narrow = guess.clone().pinned(Pins::all(BF16_FORM));
    assert!(names(&lower(&g, 0, &SIZES, &narrow).unwrap()).contains(&"gather_bf16"));
    // A table pinned f32 stays f32.
    let pins = Pins::all(BF16_FORM).pin("embed", F32_FORM);
    let plan = lower(&g, 0, &SIZES, &guess.pinned(pins)).unwrap();
    assert_eq!(names(&plan), ["gather"]);
}
