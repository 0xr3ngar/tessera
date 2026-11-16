//! Compiles the kernel bodies the small graphs lower to.

// Linking the crates registers their rules.
use cpu as _;
use kernels as _;
use linear as _;

use graph::DimName::{Context, Rows, Segments, Seqs};
use graph::{DimName, Graph};
use lower::form::named;
use lower::{Device, Pick, Pins, Recipe, Settings, Source, kept, lower};
use models::decoder::{Moe, Swiglu, program, stepper};
use models::{eg2, nanogpt, small, tagger};
use std::collections::BTreeMap;
use trace::{Linear, Norm};

const FORMS: [&str; 5] = ["f32", "bf16", "q8_0", "q4_0", "nvfp4"];

/// A graph to lower, at its sizes.
struct Case {
    graph: Graph,
    sizes: Vec<(DimName, u64)>,
    /// A stepper lowers under each kept layout.
    steps: bool,
}

/// Forward graphs, then steppers.
fn graphs() -> Vec<Case> {
    let forward = |graph: Graph| Case {
        graph,
        sizes: vec![(Seqs, 1), (Context, 8)],
        steps: false,
    };
    let step = |graph: Graph| {
        let slots = graph.dims.iter().find(|d| d.name == Seqs);
        let slots = slots.map_or(1, |d| d.size.max());
        Case {
            graph,
            sizes: vec![(Seqs, slots), (Rows, 1), (Context, 1), (Segments, 1)],
            steps: true,
        }
    };
    vec![
        forward(program::<Norm, Swiglu>(&small::qwen3())),
        forward(program::<(), Moe>(&small::granite())),
        forward(program::<(), Swiglu>(&small::smollm2())),
        forward(eg2::program::<Linear>(&small::eg2())),
        forward(tagger::program(&small::tagger())),
        forward(nanogpt::program(&small::nanogpt())),
        step(stepper::<Norm, Swiglu>(&small::qwen3())),
        step(stepper::<(), Moe>(&small::granite())),
        step(stepper::<(), Swiglu>(&small::smollm2())),
    ]
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut bodies = BTreeMap::new();
    // CPU zones pick native kernels, host zones whole trees.
    for (case, backend) in graphs()
        .iter()
        .flat_map(|c| [(c, Pick::Cpu), (c, Pick::Host)])
    {
        let settings = Settings {
            backend,
            ..Settings::default()
        };
        for form in FORMS {
            let pins = Pins::all(named(form).expect("a registered form"));
            let layouts: &[&dyn kept::Layout] = match case.steps {
                true => &[kept::DENSE, kept::INTERLEAVED],
                false => &[kept::DENSE],
            };
            for layout in layouts {
                let recipe = Recipe {
                    pins: pins.clone(),
                    kept: *layout,
                    ..Recipe::default()
                };
                let device = Device::guess(&settings).with(&recipe);
                // A form the graph cannot take leaves nothing to compile.
                let Ok(plan) = lower(&case.graph, 0, &case.sizes, &device) else {
                    continue;
                };
                for step in &plan.steps {
                    let source = step.kernel.source();
                    let hash = source.hash();
                    if let Source::Made(body) = source {
                        bodies.entry(hash).or_insert(body);
                    }
                }
            }
        }
    }
    let bodies: Vec<_> = bodies.into_iter().collect();
    let (text, skipped) = emit::module(&bodies);
    let mut out = String::new();
    for (entry, why) in &skipped {
        out.push_str(&format!("// {entry}: {why}\n"));
    }
    out.push_str(&text);
    let path = std::path::PathBuf::from(std::env::var("OUT_DIR").expect("OUT_DIR"));
    std::fs::write(path.join("kernels.rs"), out).expect("kernels.rs written");
}
