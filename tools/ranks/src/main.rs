//! One rank of Qwen3-0.6B, sharded over TCP, against the fixture.

// Linking these registers their lowering rules.
use cpu as _;
use kernels as _;
use linear as _;

use comm::{Shm, Sockets, Transport};
use graph::DimName::{Context, Seqs};
use graph::shard;
use interp::Tensor;
use lower::{Cluster, Device, Link, Pick, Weight, Zone, ZoneKind, lower};
use models::decoder::{Config, Moe, Swiglu, program};
use run::Rank;
use sim::Tag;
use std::net::{SocketAddr, TcpListener};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;
use trace::Norm;

const USAGE: &str = "tessera-ranks --rank N (--addrs HOST:PORT,... | --shm PATH --ranks N)
  [--model qwen3|granite] [--fixture DIR] [--listen ADDR] [--backend host|gpu] [--parts 3,1]
  The fixture holds model.safetensors, and tokens.txt with logits.f32,
  at its root or under cases/prompt. Its default is TESSERA_<MODEL>.";

struct Args {
    rank: u32,
    addrs: Vec<SocketAddr>,
    /// A mapped file joining the ranks, instead of sockets.
    shm: Option<PathBuf>,
    ranks: usize,
    model: String,
    fixture: Option<PathBuf>,
    listen: Option<SocketAddr>,
    host: bool,
    parts: Vec<u32>,
}

fn parse(mut given: impl Iterator<Item = String>) -> Option<Args> {
    let mut args = Args {
        rank: 0,
        addrs: Vec::new(),
        shm: None,
        ranks: 0,
        model: "qwen3".into(),
        fixture: None,
        listen: None,
        host: false,
        parts: Vec::new(),
    };
    while let Some(flag) = given.next() {
        let value = given.next()?;
        match flag.as_str() {
            "--rank" => args.rank = value.parse().ok()?,
            "--addrs" => {
                let addrs = value.split(',').map(|a| a.parse().ok());
                args.addrs = addrs.collect::<Option<_>>()?;
            }
            "--shm" => args.shm = Some(value.into()),
            "--ranks" => args.ranks = value.parse().ok()?,
            "--model" => args.model = value,
            "--fixture" => args.fixture = Some(value.into()),
            "--listen" => args.listen = Some(value.parse().ok()?),
            "--backend" => args.host = value == "host" || value == "cpu",
            "--parts" => {
                let parts = value.split(',').map(|p| p.parse().ok());
                args.parts = parts.collect::<Option<_>>()?;
            }
            _ => return None,
        }
    }
    if args.shm.is_none() {
        args.ranks = args.addrs.len();
    }
    ((args.rank as usize) < args.ranks).then_some(args)
}

/// Every rank a zone of `kind`, linked to every other.
fn cluster(ranks: usize, kind: ZoneKind) -> Cluster {
    let mut c = Cluster::one("rank0");
    c.zones[0].kind = kind;
    for r in 1..ranks {
        c.add(Zone {
            name: format!("rank{r}"),
            kind,
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

/// The model's forward graph for `n` tokens of one sequence.
fn graph(model: &str, n: u64) -> Option<graph::Graph> {
    Some(match model {
        "qwen3" => program::<Norm, Swiglu>(&Config::qwen3(1, n)),
        "granite" => program::<(), Moe>(&Config::granite(1, n)),
        _ => return None,
    })
}

/// Joins the ranks over the file, or over sockets.
fn join(args: &Args) -> Result<Box<dyn Transport>, Box<dyn std::error::Error>> {
    if let Some(path) = &args.shm {
        // Room for every token's logits, and then some.
        let capacity = 64 << 20;
        return Ok(Box::new(Shm::join(
            path,
            args.rank,
            args.ranks as u32,
            capacity,
        )?));
    }
    let listen = args.listen.unwrap_or(args.addrs[args.rank as usize]);
    let listener = TcpListener::bind(listen)?;
    eprintln!("rank {}: listening on {listen}", args.rank);
    Ok(Box::new(Sockets::join(args.rank, listener, &args.addrs)?))
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = frontend_settings::read();
    if args.host {
        settings.backend = Pick::Cpu;
    }
    let fixture = match &args.fixture {
        Some(dir) => dir.clone(),
        None => settings
            .reference(&args.model)
            .ok_or_else(|| format!("no fixture: set TESSERA_{}", args.model.to_uppercase()))?,
    };
    let case = match fixture.join("tokens.txt").is_file() {
        true => fixture.clone(),
        false => fixture.join("cases/prompt"),
    };
    let text = std::fs::read_to_string(case.join("tokens.txt"))?;
    let line = text
        .lines()
        .find(|l| !l.trim().is_empty())
        .ok_or("no tokens")?;
    let tokens: Vec<u32> = line
        .split_whitespace()
        .map(|t| t.parse())
        .collect::<Result<_, _>>()?;
    let n = tokens.len();
    let ranks = args.ranks;
    let g = graph(&args.model, n as u64).ok_or("a model this tool knows")?;
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
    let kind = match args.host {
        true => ZoneKind::Cpu,
        false => ZoneKind::Gpu,
    };
    let device = Device::guess(&settings)
        .pinned(pins)
        .zoned(cluster(ranks, kind));
    let device = device
        .ranked((0..ranks as u32).collect())
        .as_rank(args.rank);
    let device = match args.parts.is_empty() {
        true => device,
        false => device.uneven(args.parts.clone()),
    };
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

    let transport = join(args)?;
    let mut rank = Rank::open(plan, &settings, transport)?;
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
    if let Ok(bytes) = std::fs::read(case.join("logits.f32")) {
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
