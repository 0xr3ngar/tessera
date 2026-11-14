//! Qwen3-0.6B on the GPU against the reference.

use super::{GOLDEN, LOGITS, release};
use crate::common::{interpreted, loaded};
use interp::Tensor;
use lower::{Cache, Device, Pins, Settings, View, lower};
use models::qwen3::{Config, stepper};
use run::Run;
use testkit::{argmax, max_abs_diff};
use trace::DimName::{self, Context, Rows, Segments, Seqs};
use trace::Graph;

/// Read from the directory `TESSERA_QWEN3` names.
struct Fixture {
    settings: Settings,
    weights: Vec<Tensor>,
    tokens: Vec<u32>,
    want: Vec<f32>,
}

fn fixture(g: &Graph) -> Option<Fixture> {
    let mut reference = GOLDEN.reference("qwen3", g)?;
    let mut one = reference.cases.remove(0);
    Some(Fixture {
        settings: settings::read(),
        weights: reference.weights,
        want: one.outputs.remove("logits").unwrap(),
        tokens: one.tokens,
    })
}

/// Rows of logits `tokens` give on the interpreter.
fn logits(g: &Graph, c: &Fixture, tokens: &[u32]) -> Vec<f32> {
    interpreted(g, &c.weights, 1, tokens)
}

/// Each position must pick the reference's top token.
fn close(label: &str, got: &[f32], c: &Fixture) {
    LOGITS.tolerance.check(label, got, &c.want, c.tokens.len());
}

#[test]
fn gpu_checks() {
    if settings::read().gpu().is_none() {
        return eprintln!("no GPU here; skipped");
    }
    let g = release();
    let Some(c) = fixture(&g) else {
        return eprintln!("TESSERA_QWEN3 is not set; skipped");
    };
    let mut gpu = Gpu::open(&g, &c);
    forward::check(&mut gpu, &g);
    decode::check(&mut gpu, &g);
}

/// The checkpoint is bf16, so plans read it as stored.
fn pinned(settings: &Settings) -> Device {
    Device::guess(settings).pinned(Pins::all(lower::form::BF16_FORM))
}

/// Plans for either entry on one runner. Weights load once.
struct Gpu<'a> {
    c: &'a Fixture,
    runner: Box<dyn Run>,
    /// This machine's stored samples.
    device: Device,
    /// Plans by sizes, reused where a launch repeats.
    plans: Cache,
}

/// One sequence's launch: its new tokens after `past` rows.
#[derive(Clone, Copy)]
struct Span<'a> {
    past: usize,
    tokens: &'a [u32],
}

impl Span<'_> {
    /// Sizes for either entry: the forward one ignores `Rows`.
    fn sizes(&self) -> [(DimName, u64); 4] {
        let (rows, all) = (self.tokens.len(), self.past + self.tokens.len());
        [
            (Seqs, 1),
            (Rows, rows as u64),
            (Context, all as u64),
            (Segments, 1),
        ]
    }

    fn positions(&self) -> Vec<u32> {
        (self.past as u32..(self.past + self.tokens.len()) as u32).collect()
    }

    /// Tokens, positions and slots as columns, all in slot 0.
    fn tensors(&self) -> [Tensor; 3] {
        let shape = vec![self.tokens.len()];
        let tokens = Tensor::u32(shape.clone(), self.tokens.to_vec());
        let slots = Tensor::u32(shape.clone(), vec![0; self.tokens.len()]);
        [tokens, Tensor::u32(shape, self.positions()), slots]
    }
}

/// A whole prompt for the forward entry.
fn prompt(tokens: &[u32]) -> Span<'_> {
    Span { past: 0, tokens }
}

impl<'a> Gpu<'a> {
    fn open(g: &Graph, c: &'a Fixture) -> Self {
        let first = prompt(&c.tokens);
        let plan = lower(g, 0, &first.sizes(), &pinned(&c.settings)).unwrap();
        let runner = loaded(plan, g, &c.weights, &c.settings);
        let device = Device::stored(runner.platform(), &c.settings).unwrap();
        let device = device.pinned(pinned(&c.settings).pins);
        let (name, samples) = (runner.device(), device.samples());
        eprintln!("{name}: {samples} stored samples");
        Self {
            c,
            runner,
            device,
            plans: Cache::default(),
        }
    }

    /// Lowers this launch for `device` and feeds it.
    fn plan(&mut self, g: &Graph, rows: Span, device: &Device) {
        let plan = self.plans.lower(g, 0, &rows.sizes(), device).unwrap();
        self.runner.replan(&plan).unwrap();
        let (tensors, ports) = (rows.tensors(), g.entries[0].inputs.len());
        let feeds: Vec<View> = tensors[..ports].iter().map(Tensor::view).collect();
        self.runner.feed(&feeds).unwrap();
    }
}

/// The forward entry as a plan, wholly on the GPU.
mod forward {
    use super::*;

    fn repeat(c: &Fixture, n: usize) -> Vec<u32> {
        c.tokens.iter().cycle().take(n).copied().collect()
    }

    /// Logits of `tokens` as planned for `device`.
    pub fn logits_of(gpu: &mut Gpu, g: &Graph, tokens: &[u32], device: &Device) -> Vec<f32> {
        gpu.plan(g, prompt(tokens), device);
        let all = 0..gpu.runner.plan().steps.len();
        let out = gpu.runner.plan().outputs[0].slot;
        gpu.runner.submit(all).unwrap();
        gpu.runner.read(out).unwrap().floats().unwrap().to_vec()
    }

    pub fn check(gpu: &mut Gpu, g: &Graph) {
        let c = gpu.c;
        let device = gpu.device.clone();
        let short = logits_of(gpu, g, &c.tokens, &device);
        close("plan", &short, c);
        // Past the reference's length, the CPU is the check.
        let longer = repeat(c, 64);
        let planned = logits_of(gpu, g, &longer, &device);
        let off = max_abs_diff(&planned, &logits(g, c, &longer));
        eprintln!("plan vs interpreter, 64 tokens: max_abs {off:.3e}");
        assert!(off < 1e-3, "the plan and the interpreter disagree");
        // Causal, so a long run's first rows match the reference.
        let long = logits_of(gpu, g, &repeat(c, 512), &device);
        close("plan, start of 512", &long[..c.want.len()], c);
    }
}

/// Greedy decoding: one plan per launch, rows kept on device.
mod decode {
    use super::*;

    /// Tokens decoded after the prompt.
    const NEW: usize = 32;

    /// A prompt and the tokens decoded after it.
    struct Decoded {
        tokens: Vec<u32>,
        /// The index chosen after each prompt token.
        prompt: Vec<u32>,
    }

    /// Runs one launch. Returns the chosen indices.
    fn launch(gpu: &mut Gpu, g: &Graph, rows: Span, device: &Device) -> Vec<u32> {
        // One plan per context range, its length bound at launch.
        let upto = [Context];
        let (plan, bound) = gpu.plans.upto(g, 0, &rows.sizes(), &upto, device).unwrap();
        let out = plan.outputs[0].slot;
        gpu.runner.replan(plan).unwrap();
        gpu.runner.bind_all(&bound);
        let tensors = rows.tensors();
        let feeds: Vec<View> = tensors.iter().map(Tensor::view).collect();
        gpu.runner.feed(&feeds).unwrap();
        gpu.runner.submit(0..gpu.runner.plan().steps.len()).unwrap();
        gpu.runner.ids(out).unwrap()
    }

    /// The prompt at once, then one row per launch.
    fn decode(gpu: &mut Gpu, g: &Graph, device: &Device) -> Decoded {
        let mut tokens = gpu.c.tokens.clone();
        let prompt = launch(gpu, g, super::prompt(&tokens), device);
        tokens.push(*prompt.last().unwrap());
        for _ in 1..NEW {
            let (past, last) = (tokens.len() - 1, [tokens[tokens.len() - 1]]);
            let rows = Span {
                past,
                tokens: &last,
            };
            tokens.push(launch(gpu, g, rows, device)[0]);
        }
        Decoded { tokens, prompt }
    }

    /// `chosen` must be a row's best, or tie with it.
    fn likeliest(label: &str, row: &[f32], chosen: u32) {
        let lead = row[argmax(row)] - row[chosen as usize];
        assert!(lead < 1e-3, "{label}: {chosen} trails the best by {lead}");
    }

    pub fn check(gpu: &mut Gpu, forward: &Graph) {
        let text = stepper(&Config::small(1, 1024)).to_json();
        let g = Graph::from_json(&text).unwrap();
        assert_eq!(g.weights, forward.weights);
        let c = gpu.c;
        let device = gpu.device.clone();
        let planned = decode(gpu, &g, &device);
        let vocab = c.want.len() / c.tokens.len();
        for (row, id) in c.want.chunks(vocab).zip(&planned.prompt) {
            likeliest("prefill", row, *id);
        }
        // One pass over everything must choose the same tokens.
        let tokens = &planned.tokens;
        eprintln!("  decoded {:?}", &tokens[c.tokens.len()..]);
        let whole = forward::logits_of(gpu, forward, &tokens[..tokens.len() - 1], &device);
        let rows = whole.chunks(vocab).zip(&tokens[1..]);
        for (row, id) in rows.skip(c.tokens.len() - 1) {
            likeliest("decode", row, *id);
        }
    }
}
