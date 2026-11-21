//! A module's kernel text, launched with plain arguments.

use crate::abi::{File, Offer, Shape};
use crate::leak;
use crate::resident::TablePlan;
use std::collections::BTreeSet;
use std::fmt;
use tessera_device::{Backend, Import, Options, Problem, flatten};
use tessera_lower::kept::Layout;
use tessera_lower::{Arg, Kernel, Launch, Needs, ResidentSpec, Source, Step};

/// A module kernel's text, never freed.
#[derive(Debug)]
pub struct Fixed {
    pub name: &'static str,
    pub entry: &'static str,
    pub metal: &'static str,
    pub cuda: &'static str,
    pub hip: &'static str,
    /// Platform backends with no text.
    lacks: &'static [&'static str],
    /// Imports flattened from recipe files, sorted.
    pub imports: &'static [Import],
    /// Whether this kernel came from a recipe file.
    pub from_file: bool,
}

/// Metal, CUDA and HIP text of `file`, plus its imports.
fn flat(file: &str, files: &[File]) -> Result<([String; 3], Vec<Import>), tessera_device::Error> {
    let load = |path: &str| {
        files
            .iter()
            .find(|f| f.path == path)
            .map(|f| f.text.clone())
    };
    let text = load(file).ok_or_else(|| tessera_device::Error {
        file: file.into(),
        line: 0,
        problem: Problem::Missing(file.into()),
    })?;
    let mut out: [String; 3] = Default::default();
    let mut imports = BTreeSet::new();
    let mut skipped = None;
    let backends = [Backend::Metal, Backend::Cuda, Backend::Hip];
    for (slot, backend) in out.iter_mut().zip(backends) {
        // No remote rows or marks yet.
        let options = Options {
            backend,
            remote_kv: false,
            marks: false,
        };
        match flatten(file, &text, &load, &options) {
            Ok(flat) => {
                imports.extend(flat.imports);
                *slot = flat.text;
            }
            Err(e) if matches!(e.problem, Problem::Extension(_) | Problem::Unsupported(..)) => {
                skipped = Some(e)
            }
            Err(e) => return Err(e),
        }
    }
    match skipped {
        Some(e) if out.iter().all(String::is_empty) => Err(e),
        _ => Ok((out, imports.into_iter().collect())),
    }
}

impl Fixed {
    /// Kernel `k`, with any recipe file flattened per backend.
    pub fn open(k: crate::abi::Text, files: &[File]) -> Result<Self, tessera_device::Error> {
        let from_file = !k.file.is_empty();
        let ([metal, cuda, hip], imports) = if !from_file {
            let cuda = leak(k.cuda);
            ([leak(k.metal), cuda, cuda], Vec::new())
        } else {
            let (texts, imports) = flat(&k.file, files)?;
            (texts.map(leak), imports)
        };
        let texts = [("metal", metal), ("cuda", cuda), ("rocm", hip)];
        let lacks = texts.iter().filter(|t| t.1.is_empty()).map(|t| t.0);
        Ok(Fixed {
            name: leak(k.name),
            entry: leak(k.entry),
            metal,
            cuda,
            hip,
            lacks: lacks.collect::<Vec<_>>().leak(),
            imports: imports.leak(),
            from_file,
        })
    }
}

/// One launch of a module kernel: reads, output, then `args`.
#[derive(Clone)]
pub struct Text {
    pub fixed: &'static Fixed,
    pub args: Vec<Arg>,
    pub shape: tessera_lower::Shape,
    pub work: f64,
    pub guess: f64,
    /// For kept-row launches: the layout and its scalars.
    pub kept: Option<(&'static dyn Layout, Vec<Arg>)>,
    /// Persistent `tessera::table*` plan, when resident.
    pub(crate) resident: Option<TablePlan>,
}

impl Text {
    pub fn new(fixed: &'static Fixed, offer: &Offer, args: Vec<Arg>) -> Self {
        let shape = match offer.shape {
            Shape::Line(n) => tessera_lower::Shape::Line(n),
            Shape::Grid {
                across,
                down,
                threads,
            } => tessera_lower::Shape::Grid {
                across,
                down,
                threads,
            },
        };
        Self {
            fixed,
            args,
            shape,
            work: offer.work,
            guess: offer.guess,
            kept: None,
            resident: None,
        }
    }
}

/// Name and launch, not text. Compiled kernels key on it.
impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Fixed { name, .. } = self.fixed;
        let layout = self.kept.as_ref().map(|(l, _)| l.name());
        write!(
            f,
            "{name}{:?}{:?}{layout:?}{:?}",
            self.args, self.shape, self.resident
        )
    }
}

impl Kernel for Text {
    fn name(&self) -> &'static str {
        self.fixed.name
    }

    fn source(&self) -> Source {
        let Fixed {
            entry,
            metal,
            cuda,
            hip,
            ..
        } = *self.fixed;
        Source::Fixed {
            metal,
            cuda,
            hip,
            entry,
        }
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        if let Some(plan) = &self.resident {
            return vec![Launch {
                args: self.args.clone(),
                shape: tessera_lower::Shape::Grid {
                    across: 1,
                    down: 1,
                    threads: plan.threads as usize,
                },
            }];
        }
        let mut args: Vec<Arg> = (0..step.reads.len()).map(|i| step.read(i)).collect();
        args.push(step.out());
        args.extend(self.args.iter().cloned());
        // Kept rows add any page table, then the layout's scalars.
        if let Some((layout, values)) = &self.kept {
            args.extend(step.kept(*layout));
            args.extend(values.iter().cloned());
        }
        vec![Launch {
            args,
            shape: self.shape,
        }]
    }

    fn resident(&self, step: &Step) -> Option<ResidentSpec> {
        self.resident.as_ref().map(|p| p.spec(step))
    }

    fn work(&self, _: usize) -> f64 {
        self.work
    }

    fn sizes(&self) -> Vec<usize> {
        let size = |a: &Arg| match a {
            Arg::Size(e) => Some(e.most()),
            other => other.u32().map(|n| n as usize),
        };
        self.args.iter().filter_map(size).collect()
    }

    fn guess(&self) -> f64 {
        self.guess
    }

    /// The module saw each read's form before it offered.
    fn accepts(&self, _: usize, _: &dyn tessera_lower::form::Load) -> bool {
        true
    }

    /// A module may bring text for some backends only.
    fn needs(&self) -> Needs {
        let group_threads = self.resident.as_ref().map_or(0, |r| r.threads as usize);
        Needs {
            lacks: self.fixed.lacks,
            group_threads,
            ..Needs::text()
        }
    }
}
