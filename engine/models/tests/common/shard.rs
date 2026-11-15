//! The small graph over two ranks matches one rank.

use super::{Spec, settings};
use graph::{Placement, shard};
use interp::{Tensor, run, run_ranks};
use lower::{Cluster, Device, Pick, View, lower};
use run::Loopback;
use testkit::{Noise, fill, max_abs_diff};
use trace::DimName::{Context, Seqs};
use trace::{DimName, TensorType};

const SIZES: [(DimName, u64); 2] = [(Seqs, 1), (Context, 8)];
const RANKS: usize = 2;

/// Two GPU zones, linked both ways.
pub fn pair() -> Cluster {
    Cluster::two("a", "b")
}

/// Within a thousandth of the largest value.
fn close(label: &str, got: &Tensor, want: &Tensor) {
    let (got, want) = (got.floats().unwrap(), want.floats().unwrap());
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let off = max_abs_diff(got, want) / scale;
    assert!(off < 1e-3, "{label}: off {off:e}");
}

/// Sharded, the interpreter and the plans over two ranks agree.
pub fn shards(spec: &Spec) {
    let Some(small) = spec.small else { return };
    let g = small();
    let sharded = shard(&g).unwrap();
    let mut noise = Noise(5);
    let mut draw = |ty: &TensorType| fill(&g, ty, &SIZES, &mut noise).unwrap();
    let weights: Vec<Tensor> = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let inputs: Vec<Tensor> = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    // Each rank holds its own matrices of every set.
    for w in sharded.weights.iter().filter(|w| w.ty.rank() == 3) {
        assert_eq!(w.ty.placement, Placement::Split { axis: 0 }, "{:?}", w.path);
    }
    let want = run(&g, &weights, 0, &inputs).unwrap();
    let two = run_ranks(&sharded, &weights, 0, &inputs, RANKS).unwrap();
    close(&format!("{} interpreted", spec.name), &two[0], &want[0]);
    let host = lower::Settings {
        backend: Pick::Host,
        ..settings()
    };
    let device = Device::guess(&host).zoned(pair()).ranked(vec![0, 1]);
    let plans = [0, 1].map(|r| lower(&sharded, 0, &SIZES, &device.clone().as_rank(r)).unwrap());
    let plan = &plans[0];
    let split = (sharded.weights.iter()).any(|w| w.ty.placement != Placement::Replicated);
    let sums = (plan.steps.iter()).filter(|s| s.kernel.name() == "all_reduce");
    assert_eq!(
        sums.count() > 0,
        split,
        "{}: sums without splits",
        spec.name
    );
    for settings in [host, settings()] {
        let mut ranks = Loopback::open(&plans, &settings).unwrap();
        for (at, w) in weights.iter().enumerate() {
            ranks.load(at as u32, w.view()).unwrap();
        }
        let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let got = ranks.run(&feeds).unwrap();
        let label = format!("{} on {}", spec.name, ranks.rank(0).device());
        close(&label, &got[0], &want[0]);
    }
}
