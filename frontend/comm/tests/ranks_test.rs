//! Ranks on threads, joined over handles the front end opened.

// Linking these registers their lowering rules.
use attention as _;
use kernels as _;
use linear as _;

use engine::comm::Transport;
use frontend_comm::{shm, sockets};
use frontend_settings::read as settings;
use graph::{Graph, shard};
use interp::{Tensor, run};
use lower::{Cluster, Device, Pick, Plan, Settings, View, lower};
use models::decoder::{Config, Swiglu, program};
use run::Rank;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use testkit::{Noise, fill, max_abs_diff};
use trace::DimName::{self, Context, Seqs};
use trace::{Norm, TensorType};

const SIZES: [(DimName, u64); 2] = [(Seqs, 1), (Context, 8)];

/// A tensor's parts, which may cross to another thread.
type Plain = (Vec<usize>, Vec<u8>, mem::DType);

fn plain(t: &Tensor) -> Plain {
    (t.shape.clone(), t.bytes().to_vec(), t.dtype())
}

fn rebuilt((shape, bytes, dtype): &Plain) -> Tensor {
    Tensor::from_bytes(shape.clone(), *dtype, bytes)
}

fn carried(tensors: &[Tensor]) -> Vec<Plain> {
    tensors.iter().map(plain).collect()
}

/// `n` listeners on the loopback, with their addresses.
fn listening(n: usize) -> (Vec<TcpListener>, Vec<SocketAddr>) {
    let listeners: Vec<TcpListener> = (0..n)
        .map(|_| TcpListener::bind("127.0.0.1:0").unwrap())
        .collect();
    let addrs = listeners.iter().map(|l| l.local_addr().unwrap()).collect();
    (listeners, addrs)
}

/// The toy graph with faux weights and inputs.
fn toy() -> (Graph, Vec<Tensor>, Vec<Tensor>) {
    let g = program::<Norm, Swiglu>(&Config::toy(2, 8));
    let mut noise = Noise(5);
    let mut draw = |ty: &TensorType| fill(&g, ty, &SIZES, &mut noise).unwrap();
    let weights = g.weights.iter().map(|w| draw(&w.ty)).collect();
    let inputs = g.entries[0].inputs.iter().map(|p| draw(&p.ty)).collect();
    (g, weights, inputs)
}

/// The sharded graph planned for each of two host ranks.
fn planned(g: &Graph, host: &Settings) -> [Plan; 2] {
    let device = Device::guess(host)
        .zoned(Cluster::two("a", "b"))
        .ranked(vec![0, 1]);
    let sharded = shard(g).unwrap();
    [0, 1].map(|r| lower(&sharded, 0, &SIZES, &device.clone().as_rank(r)).unwrap())
}

/// One rank of `plan` over `transport`, run once.
fn rank<T: Transport + 'static>(
    transport: T,
    plan: &Plan,
    host: &Settings,
    weights: &[Plain],
    inputs: &[Plain],
) -> Vec<f32> {
    let mut rank = Rank::open(plan.clone(), host, transport).unwrap();
    for (at, w) in weights.iter().enumerate() {
        rank.load(at as u32, rebuilt(w).view()).unwrap();
    }
    let inputs: Vec<Tensor> = inputs.iter().map(rebuilt).collect();
    let feeds: Vec<View> = inputs.iter().map(Tensor::view).collect();
    let out = rank.run(&feeds).unwrap().remove(0);
    out.floats().unwrap().to_vec()
}

/// Both ranks on threads, over the transports `open` gives.
fn both<T: Transport + 'static>(open: impl Fn(u32) -> T + Sync) {
    let (g, weights, inputs) = toy();
    let host = Settings {
        backend: Pick::Host,
        ..settings()
    };
    let plans = planned(&g, &host);
    let (weights_plain, inputs_plain) = (carried(&weights), carried(&inputs));
    let outs: Vec<Vec<f32>> = std::thread::scope(|s| {
        let runs: Vec<_> = (0..2u32)
            .map(|r| {
                let (plan, host, open) = (&plans[r as usize], &host, &open);
                let (weights, inputs) = (&weights_plain, &inputs_plain);
                s.spawn(move || rank(open(r), plan, host, weights, inputs))
            })
            .collect();
        runs.into_iter().map(|t| t.join().unwrap()).collect()
    });
    assert_eq!(outs[0], outs[1]);
    let want = run(&g, &weights, 0, &inputs).unwrap();
    let want = want[0].floats().unwrap();
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let diff = max_abs_diff(&outs[0], want) / scale;
    assert!(diff < 1e-3, "{diff}");
}

/// Ranks on threads over TCP, against one rank.
#[test]
fn two_sockets() {
    let (listeners, addrs) = listening(2);
    let listeners: Vec<_> = listeners.into_iter().map(std::sync::Mutex::new).collect();
    both(|r| {
        let listener = listeners[r as usize].lock().unwrap().try_clone().unwrap();
        sockets(r, listener, &addrs).unwrap()
    });
}

/// Ranks on threads over one mapped file, against one rank.
#[test]
fn two_shm() {
    let path: PathBuf = testkit::scratch(&format!("ranks-{}", std::process::id()));
    both(|r| shm(&path, r, 2, 1 << 16).unwrap());
    let _ = std::fs::remove_file(&path);
}
