//! One rank of Qwen3-0.6B, sharded over TCP, against the fixture.

// Linking these registers their lowering rules.
use cpu as _;
use kernels as _;
use linear as _;

use engine_comm::{Fabric, Transport, Verbs};
use frontend_comm::{ports, shm, sockets};
use graph::DimName::{Context, Seqs};
use graph::shard;
use interp::Tensor;
use lower::{Cluster, Device, Link, Pick, Weight, Zone, ZoneKind, lower};
use models::decoder::{Config, Moe, Swiglu, program};
use run::Rank;
use sim::Tag;
use std::net::{SocketAddr, TcpListener};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;
use trace::Norm;

const USAGE: &str = "tessera-ranks --rank N (--addrs HOST:PORT,... | --shm PATH --ranks N)
  [--model qwen3|granite|eg2|deepseek|glm53] [--fixture DIR] [--listen ADDR] [--backend host|gpu] [--parts 3,1]
  [--form q8_0] [--verbs DEV[:GID],... | --ring PREVDEV,.../NEXTDEV,...] [--repeat N]
  --verbs and --ring move the sums over RDMA, one lane per device, after the
  ranks meet over the sockets or the file.
  The fixture holds model.safetensors, and tokens.txt with logits.f32,
  at its root or under cases/prompt. Its default is ~/.cache/tessera/references/<model>.";

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
    /// A form every weight takes, instead of the file's.
    form: Option<String>,
    /// Forward passes to run. The last one is reported.
    repeat: usize,
    /// RDMA ports, after the ranks meet.
    fabric: Option<Fabric>,
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
        form: None,
        repeat: 1,
        fabric: None,
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
            "--form" => args.form = Some(value),
            "--repeat" => args.repeat = value.parse().ok().filter(|n| *n > 0)?,
            "--verbs" => args.fabric = Some(Fabric::Mesh(ports(&value).ok()?)),
            "--ring" => {
                let (prev, next) = value.split_once('/')?;
                let (prev, next) = (ports(prev).ok()?, ports(next).ok()?);
                args.fabric = Some(Fabric::Ring { prev, next });
            }
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

#[cfg(test)]
mod eg2_test;
#[cfg(test)]
mod mla_test;

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

type Loaded = (graph::Graph, Vec<Weight>, lower::Pins);

/// One sequence's graph for `n` tokens, its weights and forms.
fn model(name: &str, fixture: &Path, n: u64) -> Result<Loaded, Box<dyn std::error::Error>> {
    let g = match name {
        "qwen3" => program::<Norm, Swiglu>(&Config::qwen3(1, n)),
        "granite" => program::<(), Moe>(&Config::granite(1, n)),
        "deepseek" | "glm53" => {
            let r = bundle::mla::release(fixture, 1, n).map_err(|e| e.to_string())?;
            return Ok((r.graph, r.weights, r.pins));
        }
        "eg2" => {
            let e = bundle::encoder::encoder(&fixture.join("exlr8"), [1, n], false)
                .map_err(|e| e.to_string())?;
            return Ok((e.graph, e.weights, lower::Pins::default()));
        }
        _ => return Err("a model this tool knows".into()),
    };
    let file = checkpoint::File::open(&fixture.join("model.safetensors"))?;
    let (weights, pins) = bundle::pull(&g, &file, 0, &[]).map_err(|e| e.to_string())?;
    Ok((g, weights, pins))
}

/// Room for every token's logits, and then some.
const CAPACITY: usize = 64 << 20;

/// Joins the ranks over the file, or over sockets.
fn meet(args: &Args) -> Result<Box<dyn Transport>, Box<dyn std::error::Error>> {
    if let Some(path) = &args.shm {
        return Ok(Box::new(shm(path, args.rank, args.ranks as u32, CAPACITY)?));
    }
    let listen = args.listen.unwrap_or(args.addrs[args.rank as usize]);
    let listener = TcpListener::bind(listen)?;
    eprintln!("rank {}: listening on {listen}", args.rank);
    Ok(Box::new(sockets(args.rank, listener, &args.addrs)?))
}

/// The sums' transport: RDMA when ports are given.
fn join(
    args: &Args,
    settings: &lower::Settings,
) -> Result<Box<dyn Transport>, Box<dyn std::error::Error>> {
    let mut met = meet(args)?;
    let Some(fabric) = &args.fabric else {
        return Ok(met);
    };
    let (rank, ranks) = (args.rank, args.ranks as u32);
    let pinning = run::pinning(settings);
    let verbs = Verbs::connect(rank, ranks, fabric, CAPACITY, &mut *met, pinning)?;
    eprintln!(
        "rank {}: sums over {} RDMA lanes",
        args.rank,
        fabric.lanes()
    );
    Ok(Box::new(verbs))
}

fn run(args: &Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut settings = frontend_settings::read();
    if args.host {
        settings.backend = Pick::Cpu;
    }
    let fixture = match &args.fixture {
        Some(dir) => dir.clone(),
        None => testkit::reference(&args.model)
            .ok_or_else(|| format!("no fixture: {} has no reference here", args.model))?,
    };
    let case = ["", "cases/prompt", "cases/words"]
        .map(|at| fixture.join(at))
        .into_iter()
        .find(|dir| dir.join("tokens.txt").is_file())
        .ok_or("no tokens.txt in the fixture")?;
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
    let (g, weights, mut pins) = model(&args.model, &fixture, n as u64)?;
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

    if let Some(form) = &args.form {
        let form = lower::form::named(form).ok_or_else(|| format!("no form `{form}`"))?;
        pins = lower::Pins::all(form);
    }
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

    let transport = join(args, &settings)?;
    let mut rank = Rank::open(plan, &settings, transport)?;
    let sums = match rank.on_device() {
        true => "on the device",
        false => "on the host",
    };
    eprintln!(
        "rank {}: on {}, sums {sums}",
        args.rank,
        rank.runner().device()
    );
    let start = Instant::now();
    // Each rank cuts its part from the packed bytes.
    for (at, w) in weights.iter().enumerate() {
        rank.load_weight(at as u32, w)?;
    }
    eprintln!(
        "rank {}: loaded in {:.1} s",
        args.rank,
        start.elapsed().as_secs_f64()
    );

    let ids = Tensor::u32(vec![1, n], tokens);
    let positions = Tensor::u32(vec![1, n], (0..n as u32).collect());
    let feeds = [ids.view(), positions.view()];
    for _ in 1..args.repeat {
        rank.run(&feeds)?;
    }
    let start = Instant::now();
    let out = rank.run(&feeds)?;
    let seconds = start.elapsed().as_secs_f64();
    let logits = out[0].floats()?;
    // An embedding is one short row.
    let vocab = (logits.len() / n).max(1);
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
    if let Ok(bytes) = std::fs::read(case.join("embedding.f32")) {
        let (words, _) = bytes.as_chunks::<4>();
        let want: Vec<f32> = words.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let cos = testkit::cosine_f32(logits, &want);
        eprintln!("rank {}: embedding cos {cos:.5}", args.rank);
    }
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
