//! `tessera-serve --model <dir>`: OpenAI-shaped routes over one model.

use engine::{Batch, Engine, Fair, Pins, Recipe, Settings};
use serve::{Model, Slot};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::{Arc, Mutex, OnceLock, mpsc};
use std::time::Instant;
use tasks::api::Info;
use tasks::{Generate, Job, Queued, Threaded};

const USAGE: &str = "usage: tessera-serve --model <dir> [--host <addr>] [--port <n>] \
[--max-model-len <tokens>] [--slots <n>] [--served-model-name <name>] \
[--task generate|embed] [--reject-unknown-fields]";

struct Args {
    model: PathBuf,
    host: String,
    port: u16,
    /// Tokens one answer holds, prompt included.
    max_model_len: usize,
    /// Answers in flight at once.
    slots: usize,
    name: Option<String>,
    embed: bool,
    /// Unknown request fields refuse the request.
    strict: bool,
}

fn args() -> Option<Args> {
    let mut args = Args {
        model: PathBuf::new(),
        host: "127.0.0.1".into(),
        port: 8080,
        max_model_len: 4096,
        slots: 4,
        name: None,
        embed: false,
        strict: false,
    };
    let mut given = std::env::args().skip(1);
    while let Some(flag) = given.next() {
        if flag == "--reject-unknown-fields" {
            args.strict = true;
            continue;
        }
        let value = given.next()?;
        match flag.as_str() {
            "--model" => args.model = value.into(),
            "--host" => args.host = value,
            "--port" => args.port = value.parse().ok()?,
            "--max-model-len" => args.max_model_len = value.parse().ok()?,
            "--slots" => args.slots = value.parse().ok()?,
            "--served-model-name" => args.name = Some(value),
            "--task" if value == "generate" => args.embed = false,
            "--task" if value == "embed" => args.embed = true,
            _ => return None,
        }
    }
    (!args.model.as_os_str().is_empty()).then_some(args)
}

/// Loads the generator, fills `slot`, then steps jobs.
fn engine_thread(args: &Args, settings: &Settings, info: Info, slot: &Slot) -> Result<(), String> {
    let start = Instant::now();
    let rules = match &settings.names {
        Some(path) => std::fs::read_to_string(path)
            .map_err(|e| e.to_string())
            .and_then(|text| bundle::names::rules(&text).map_err(|e| e.to_string()))?,
        None => Vec::new(),
    };
    let (slots, context) = (args.slots as u64, args.max_model_len as u64);
    let loaded =
        bundle::load_named(&args.model, slots, context, &rules).map_err(|e| e.to_string())?;
    let recipe = Recipe {
        pins: loaded
            .pins
            .unwrap_or_else(|| Pins::all(engine::form::BF16_FORM)),
        ..Recipe::default()
    };
    let opened = Engine::open(loaded.graph, args.max_model_len, &recipe, settings);
    let mut engine = opened.map_err(|e| e.to_string())?;
    engine.load(&loaded.weights).map_err(|e| e.to_string())?;
    let seconds = start.elapsed().as_secs_f64();
    eprintln!("{}: loaded in {seconds:.1} s", engine.device());
    let generate = Arc::new(Generate {
        tokenizer: loaded.tokenizer,
        template: loaded.template,
        stop: loaded.stop,
    });
    let (queue, jobs) = mpsc::channel::<Job>();
    let queued = Queued {
        generate: generate.clone(),
        jobs: Mutex::new(queue),
    };
    let model = Model {
        info,
        tokens: generate.clone(),
        chat: Some(generate),
        generator: Some(Arc::new(queued)),
        embed: None,
        strict: args.strict,
    };
    let _ = slot.set(model);
    let mut batch = Batch::new(&mut engine, Box::new(Fair));
    loop {
        // Takes every job that has arrived. Waits only when idle.
        let job = match batch.busy() {
            true => jobs.try_recv().ok(),
            false => match jobs.recv() {
                Ok(job) => Some(job),
                Err(_) => return Ok(()),
            },
        };
        if let Some((request, sink)) = job {
            batch.submit(request, sink);
            continue;
        }
        batch.step().map_err(|e| e.to_string())?;
    }
}

/// Loads the model into `slot`. A failure ends the process.
fn load(args: Args, slot: Slot) {
    let settings = frontend_settings::read();
    let name = args.name.clone();
    let info = Info {
        name: name.unwrap_or_else(|| args.model.display().to_string()),
        context: args.max_model_len,
    };
    let strict = args.strict;
    let failed = match args.embed {
        false => engine_thread(&args, &settings, info, &slot),
        true => Threaded::open(args.model, settings)
            .map(|embedder| {
                let embedder = Arc::new(embedder);
                let _ = slot.set(Model {
                    info,
                    tokens: embedder.clone(),
                    chat: None,
                    generator: None,
                    embed: Some(embedder),
                    strict,
                });
            })
            .map_err(|e| e.to_string()),
    };
    if let Err(e) = failed {
        eprintln!("tessera-serve: {e}");
        std::process::exit(1);
    }
}

fn main() -> ExitCode {
    let Some(args) = args() else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let slot: Slot = Arc::new(OnceLock::new());
    let (host, port) = (args.host.clone(), args.port);
    let loading = slot.clone();
    std::thread::spawn(move || load(args, loading));
    let served = tokio::runtime::Runtime::new().and_then(|runtime| {
        runtime.block_on(async {
            let listener = tokio::net::TcpListener::bind((host.as_str(), port)).await?;
            eprintln!("listening on {host}:{port}");
            axum::serve(listener, serve::router(slot)).await
        })
    });
    match served {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("tessera-serve: {e}");
            ExitCode::FAILURE
        }
    }
}
