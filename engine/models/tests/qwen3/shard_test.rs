//! The toy decoder over two ranks matches one.

use graph::{Placement, shard};
use interp::{run, run_ranks};
use models::decoder::{Config, program};
use testkit::{Noise, fill, max_abs_diff};
use trace::DimName::{Context, Seqs};
use trace::{Norm, TensorType};

#[test]
fn two_ranks() {
    let g = program::<Norm>(&Config::toy(2, 8));
    let sharded = shard(&g).unwrap();
    let split = |axis| {
        let is = |w: &&graph::Weight| w.ty.placement == Placement::Split { axis };
        sharded.weights.iter().filter(is).count()
    };
    // Per layer: five column splits, two row splits.
    assert_eq!((split(0), split(1)), (10, 4));
    let sizes = [(Seqs, 1), (Context, 8)];
    let mut noise = Noise(5);
    let mut draw = |ty: &TensorType| fill(&g, ty, &sizes, &mut noise).unwrap();
    let weights: Vec<_> = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let inputs: Vec<_> = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    let one = run(&g, &weights, 0, &inputs).unwrap();
    let two = run_ranks(&sharded, &weights, 0, &inputs, 2).unwrap();
    let diff = max_abs_diff(one[0].floats().unwrap(), two[0].floats().unwrap());
    assert!(diff < 1e-4, "{diff}");
}
