//! The toy decoder over two ranks matches one.

use frontend_settings::read as settings;
use graph::{Graph, Placement, shard};
use interp::{Tensor, run, run_ranks};
use lower::{Cluster, Device, Link, Pick, Plan, View, Zone, ZoneKind, lower};
use models::decoder::{Config, program};
use run::{Loopback, Rank, Sockets};
use sim::Tag;
use std::net::TcpListener;
use testkit::{Noise, fill, max_abs_diff};
use trace::DimName::{Context, Seqs};
use trace::{DimName, Norm, TensorType};

const SIZES: [(DimName, u64); 2] = [(Seqs, 1), (Context, 8)];

/// The toy graph, sharded too, with faux weights and inputs.
fn toy() -> (Graph, Graph, Vec<Tensor>, Vec<Tensor>) {
    let g = program::<Norm>(&Config::toy(2, 8));
    let sharded = shard(&g).unwrap();
    let mut noise = Noise(5);
    let mut draw = |ty: &TensorType| fill(&g, ty, &SIZES, &mut noise).unwrap();
    let weights = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let inputs = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    (g, sharded, weights, inputs)
}

/// How far `got` is from the whole graph's logits, relative.
fn off(got: &Tensor, g: &Graph, weights: &[Tensor], inputs: &[Tensor]) -> f32 {
    let want = run(g, weights, 0, inputs).unwrap();
    let want = want[0].floats().unwrap();
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    max_abs_diff(got.floats().unwrap(), want) / scale
}

#[test]
fn two_ranks() {
    let (g, sharded, weights, inputs) = toy();
    let split = |axis| {
        let is = |w: &&graph::Weight| w.ty.placement == Placement::Split { axis };
        sharded.weights.iter().filter(is).count()
    };
    // Per layer: five column splits, two row splits.
    assert_eq!((split(0), split(1)), (10, 4));
    let two = run_ranks(&sharded, &weights, 0, &inputs, 2).unwrap();
    let diff = off(&two[0], &g, &weights, &inputs);
    assert!(diff < 1e-4, "{diff}");
}

/// Two GPU zones, linked both ways.
fn pair() -> Cluster {
    let mut cluster = Cluster::one("a");
    let b = cluster.add(Zone {
        name: "b".into(),
        kind: ZoneKind::Gpu,
        parent: None,
    });
    for (from, to) in [(0, b), (b, 0)] {
        cluster.links.push(Link {
            from,
            to,
            latency: 1e-5,
            rate: 1e10,
            tag: Tag::Assumed,
        });
    }
    cluster
}

#[test]
fn two_rank_plan() {
    let (g, sharded, weights, inputs) = toy();
    let host = lower::Settings {
        backend: Pick::Host,
        ..settings()
    };
    let device = Device::guess(&host).zoned(pair()).ranked(vec![0, 1]);
    let plan = lower(&sharded, 0, &SIZES, &device).unwrap();
    let sums = plan
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "all_reduce");
    assert_eq!((plan.ranks, sums.count()), (2, 4));
    // Each rank holds half of every split weight.
    let whole = lower(&g, 0, &SIZES, &Device::guess(&host)).unwrap();
    let held = |p: &Plan| p.weights.iter().map(|l| l.slot.len).sum::<usize>();
    let split: usize = (sharded.weights.iter().zip(&weights))
        .filter(|(w, _)| w.ty.placement != Placement::Replicated)
        .map(|(_, t)| t.len())
        .sum();
    assert_eq!(held(&whole) - held(&plan), split / 2);
    assert!(device.cost(&plan).seconds.is_finite());
    // On the host, then on any GPU the settings pick.
    for settings in [host, settings()] {
        let mut ranks = Loopback::open(&plan, &settings).unwrap();
        for (at, w) in weights.iter().enumerate() {
            ranks.load(at as u32, w.view()).unwrap();
        }
        let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
        let got = ranks.run(&feeds).unwrap();
        let diff = off(&got[0], &g, &weights, &inputs);
        assert!(diff < 1e-3, "{diff} on {}", ranks.rank(0).device());
    }
}

/// Each rank on its own thread, joined over TCP.
#[test]
fn two_sockets() {
    let (g, sharded, weights, inputs) = toy();
    let host = lower::Settings {
        backend: Pick::Host,
        ..settings()
    };
    let device = Device::guess(&host).zoned(pair()).ranked(vec![0, 1]);
    let plan = lower(&sharded, 0, &SIZES, &device).unwrap();
    let listeners: Vec<TcpListener> = (0..2)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addrs: Vec<_> = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    // Tensors stay on one thread, so threads get the numbers.
    let plain = |t: &Tensor| (t.shape.clone(), t.bytes().to_vec(), t.dtype());
    let rebuilt = |(shape, bytes, dtype): &(Vec<usize>, Vec<u8>, _)| {
        Tensor::from_bytes(shape.clone(), *dtype, bytes)
    };
    let (weights_plain, inputs_plain): (Vec<_>, Vec<_>) = (
        weights.iter().map(plain).collect(),
        inputs.iter().map(plain).collect(),
    );
    let outs: Vec<Vec<f32>> = std::thread::scope(|s| {
        let runs: Vec<_> = (listeners.into_iter().enumerate())
            .map(|(r, listener)| {
                let (plan, addrs, host) = (&plan, &addrs, &host);
                let (weights, inputs) = (&weights_plain, &inputs_plain);
                s.spawn(move || {
                    let sockets = Sockets::join(r as u32, listener, addrs).unwrap();
                    let mut rank = Rank::open(plan.clone(), host, sockets).unwrap();
                    for (at, w) in weights.iter().enumerate() {
                        rank.load(at as u32, rebuilt(w).view()).unwrap();
                    }
                    let inputs: Vec<Tensor> = inputs.iter().map(rebuilt).collect();
                    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
                    let out = rank.run(&feeds).unwrap().remove(0);
                    out.floats().unwrap().to_vec()
                })
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    assert_eq!(outs[0], outs[1]);
    let shape = run(&g, &weights, 0, &inputs).unwrap()[0].shape.clone();
    let got = Tensor::f32(shape, outs[0].clone());
    let diff = off(&got, &g, &weights, &inputs);
    assert!(diff < 1e-3, "{diff}");
}
