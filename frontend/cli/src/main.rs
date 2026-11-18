//! `tessera (--recipe <src> | --model <dir>) <verb>`.

use recipe::Recipe;
use serve::start::{FLAGS, Options};
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "usage: tessera (--recipe <dir | git+<url>#<commit>> | --model <dir>) <verb>
  serve [flags]  load the model and answer OpenAI-shaped requests
  fetch          download the recipe and its weights, then print where they are";

/// Why a run stops early.
enum Stop {
    Usage,
    Failed(String),
}

impl From<recipe::Error> for Stop {
    fn from(e: recipe::Error) -> Self {
        Stop::Failed(e.to_string())
    }
}

/// Where the model comes from.
enum Origin {
    Recipe(String),
    Model(String),
}

/// Takes one serve flag. `Usage` when it is bad.
fn apply(options: &mut Options, flag: &str, value: Option<String>) -> Result<(), Stop> {
    match value {
        None if options.switch(flag) => Ok(()),
        Some(value) => options.flag(flag, value).ok_or(Stop::Usage),
        None => Err(Stop::Usage),
    }
}

/// Serve options from the recipe, then from the command line.
fn options(
    recipe: Option<&Recipe>,
    cache: Option<&Path>,
    rest: &[String],
) -> Result<Options, Stop> {
    let mut options = Options::default();
    if let Some(r) = recipe {
        options.model = r.weights(cache)?;
        options.name = Some(r.name.clone());
        options.lower = r.lower.clone();
        options.names = r.names();
        for (flag, value) in r.flags() {
            apply(&mut options, &flag, value)?;
        }
    }
    let mut given = rest.iter().cloned();
    while let Some(flag) = given.next() {
        if options.switch(&flag) {
            continue;
        }
        let value = given.next().ok_or(Stop::Usage)?;
        apply(&mut options, &flag, Some(value))?;
    }
    Ok(options)
}

fn run(given: &[String]) -> Result<(), Stop> {
    let (from, verb, rest) = match given {
        [flag, src, verb, rest @ ..] if flag == "--recipe" => {
            (Origin::Recipe(src.clone()), verb, rest)
        }
        [flag, dir, verb, rest @ ..] if flag == "--model" => {
            (Origin::Model(dir.clone()), verb, rest)
        }
        _ => return Err(Stop::Usage),
    };
    let settings = frontend_settings::read();
    let cache = settings.recipes.as_deref();
    let recipe = match &from {
        Origin::Recipe(src) => Some(Recipe::open(src, cache)?),
        Origin::Model(_) => None,
    };
    match (verb.as_str(), &recipe) {
        ("fetch", Some(r)) if rest.is_empty() => {
            println!("recipe {}", r.dir.display());
            println!("weights {}", r.weights(cache)?.display());
            Ok(())
        }
        ("serve", _) => {
            let mut options = options(recipe.as_ref(), cache, rest)?;
            if let Origin::Model(dir) = from {
                options.model = dir.into();
            }
            serve::start::run(options).map_err(Stop::Failed)
        }
        _ => Err(Stop::Usage),
    }
}

fn main() -> ExitCode {
    let given: Vec<String> = std::env::args().skip(1).collect();
    match run(&given) {
        Ok(()) => ExitCode::SUCCESS,
        Err(Stop::Usage) => {
            eprintln!("{USAGE}\nserve flags: {FLAGS}");
            ExitCode::from(2)
        }
        Err(Stop::Failed(e)) => {
            eprintln!("tessera: {e}");
            ExitCode::FAILURE
        }
    }
}
