//! One rank of Qwen3-0.6B, sharded over TCP, against the fixture.

// Linking these registers their lowering rules.
use kernels as _;
use linear as _;

use comm::Sockets;
use graph::DimName::{Context, Seqs};
use graph::shard;
use interp::Tensor;
use lower::{Cluster, Device, Link, Weight, Zone, ZoneKind, lower};
use models::decoder::{Config, Swiglu, program};
use run::Rank;
use sim::Tag;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use trace::Norm;

const USAGE: &str = "tessera-ranks --rank N --addrs HOST:PORT,... [--fixture DIR] [--listen ADDR]
  The fixture holds model.safetensors, tokens.txt and logits.f32.
  Its default is the TESSERA_QWEN3 directory.";

struct Args {
    rank: u32,
    addrs: Vec<SocketAddr>,
    fixture: Option<PathBuf>,
    listen: Option<SocketAddr>,
}

fn parse(mut given: impl Iterator<Item = String>) -> Option<Args> {
    let mut args = Args {
        rank: 0,
        addrs: Vec::new(),
        fixture: None,
        listen: None,
    };
    while let Some(flag) = given.next() {
        let value = given.next()?;
        match flag.as_str() {
            "--rank" => args.rank = value.parse().ok()?,
            "--addrs" => {
                let addrs = value.split(',').map(|a| a.parse().ok());
                args.addrs = addrs.collect::<Option<_>>()?;
            }
            "--fixture" => args.fixture = Some(value.into()),
            "--listen" => args.listen = Some(value.parse().ok()?),
            _ => return None,
        }
    }
    ((args.rank as usize) < args.addrs.len()).then_some(args)
}

/// Every rank a GPU zone, linked to every other.
fn cluster(ranks: usize) -> Cluster {
    let mut c = Cluster::one("rank0");
    for r in 1..ranks {
        c.add(Zone {
            name: format!("rank{r}"),
            kind: ZoneKind::Gpu,
            parent: None,
        });
    }
    for from in 0..ranks {
        for to in (0..ranks).filter(|t| *t != from) {
            c.links.push(Link {
                from,
                to,
                latency: 1e-4,
                rate: 1e9,
                tag: Tag::Assumed,
            });
        }
    }
    c
}

/// `w` as floats in the graph's shape.
fn floats(g: &graph::Graph, at: usize, w: &Weight) -> Tensor {
    let axis = |e: &graph::DimExpr| e.value(&g.dims, &|_| None).unwrap_or(1) as usize;
    let shape: Vec<usize> = g.weights[at].ty.shape.0.iter().map(axis).collect();
    match w {
        Weight::Floats(t) => Tensor::f32(shape, t.floats().unwrap_or_default().to_vec()),
        Weight::Packed(f, bytes) => {
            let n = f.elements(bytes.len());
            Tensor::f32(shape, (0..n).map(|i| f.unpack(bytes, i)).collect())
        }
    }
}

fn main() -> ExitCode {
    let Some(args) = parse(std::env::args().skip(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-ranks: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let settings = frontend_settings::read();
    let fixture = match &args.fixture {
        Some(dir) => dir.clone(),
        None => settings
            .reference("qwen3")
            .ok_or("no fixture: set TESSERA_QWEN3")?,
    };
    let text = std::fs::read_to_string(fixture.join("tokens.txt"))?;
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or("no tokens")?;
    let tokens: Vec<u32> = line
        .split_whitespace()
        .map(|t| t.parse())
        .collect::<Result<_, _>>()?;
    let n = tokens.len();
    let ranks = args.addrs.len();
    let g = program::<Norm, Swiglu>(&Config::qwen3(1, n as u64));
    let sharded = shard(&g)?;
    let split = sharded
        .weights
        .iter()
        .filter(|w| w.ty.placement != graph::Placement::Replicated);
    eprintln!(
        "rank {}: {} of {} weights split over {ranks} ranks",
        args.rank,
        split.count(),
        g.weights.len()
    );

    let file = checkpoint::File::open(&fixture.join("model.safetensors"))?;
    let (weights, pins) = bundle::pull(&g, &file, 0, &[]).map_err(|e| e.to_string())?;
    let device = Device::guess(&settings).pinned(pins).zoned(cluster(ranks));
    let device = device.ranked((0..ranks as u32).collect());
    let sizes = [(Seqs, 1), (Context, n as u64)];
    let start = Instant::now();
    let plan = lower(&sharded, 0, &sizes, &device)?;
    let sums = plan
        .steps
        .iter()
        .filter(|s| s.kernel.name() == "all_reduce")
        .count();
    let held: usize = plan.weights.iter().map(|l| l.slot.bytes()).sum();
    eprintln!(
        "rank {}: {} steps, {sums} sums, {:.1} MB of weights, lowered in {:.1} s",
        args.rank,
        plan.steps.len(),
        held as f64 / 1e6,
        start.elapsed().as_secs_f64()
    );

    let listen = args.listen.unwrap_or(args.addrs[args.rank as usize]);
    let listener = TcpListener::bind(listen)?;
    eprintln!(
        "rank {}: listening on {listen}, joining {} ranks",
        args.rank, ranks
    );
    let sockets = Sockets::join(args.rank, listener, &args.addrs)?;
    let mut rank = Rank::open(plan, &settings, sockets)?;
    eprintln!("rank {}: on {}", args.rank, rank.runner().device());
    let start = Instant::now();
    for (at, w) in weights.iter().enumerate() {
        rank.load(at as u32, floats(&g, at, w).view())?;
    }
    eprintln!(
        "rank {}: loaded in {:.1} s",
        args.rank,
        start.elapsed().as_secs_f64()
    );

    let ids = Tensor::u32(vec![1, n], tokens);
    let positions = Tensor::u32(vec![1, n], (0..n as u32).collect());
    let feeds = [ids.view(), positions.view()];
    let start = Instant::now();
    let out = rank.run(&feeds)?;
    let seconds = start.elapsed().as_secs_f64();
    let logits = out[0].floats()?;
    let vocab = logits.len() / n;
    let top = |row: &[f32]| {
        (0..row.len())
            .max_by(|a, b| row[*a].total_cmp(&row[*b]))
            .expect("a vocab")
    };
    let ours: Vec<usize> = logits.chunks(vocab).map(top).collect();
    eprintln!(
        "rank {}: forward of {n} tokens in {seconds:.3} s, top ids {ours:?}",
        args.rank
    );
    if let Ok(bytes) = std::fs::read(fixture.join("logits.f32")) {
        let (words, _) = bytes.as_chunks::<4>();
        let want: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let theirs: Vec<usize> = want.chunks(vocab).map(top).collect();
        let agree = ours.iter().zip(&theirs).filter(|(a, b)| a == b).count();
        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let off = logits
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
            / scale;
        eprintln!(
            "rank {}: top token agrees on {agree} of {n} rows, logits off {off:.2e}",
            args.rank
        );
    }
    Ok(())
}
