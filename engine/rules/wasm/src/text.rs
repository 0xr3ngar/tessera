//! A module's kernel text, launched with plain arguments.

use crate::abi::{Offer, Shape};
use sensai_lower::{Kernel, Launch, Source, Step};
use std::fmt;

/// A module kernel's text, never freed.
#[derive(Debug)]
pub struct Fixed {
    pub name: &'static str,
    pub entry: &'static str,
    pub metal: &'static str,
    pub cuda: &'static str,
}

/// One launch of a module kernel: reads, output, then `counts`.
#[derive(Clone)]
pub struct Text {
    pub fixed: &'static Fixed,
    pub counts: Vec<u32>,
    pub shape: sensai_lower::Shape,
    pub work: f64,
    pub guess: f64,
}

impl Text {
    pub fn new(fixed: &'static Fixed, offer: &Offer) -> Self {
        let shape = match offer.shape {
            Shape::Line(n) => sensai_lower::Shape::Line(n),
            Shape::Grid {
                across,
                down,
                threads,
            } => sensai_lower::Shape::Grid {
                across,
                down,
                threads,
            },
        };
        Self {
            fixed,
            counts: offer.counts.clone(),
            shape,
            work: offer.work,
            guess: offer.guess,
        }
    }
}

/// Name and launch, not text. Compiled kernels key on it.
impl fmt::Debug for Text {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let Fixed { name, .. } = self.fixed;
        write!(f, "{name}{:?}{:?}", self.counts, self.shape)
    }
}

impl Kernel for Text {
    fn name(&self) -> &'static str {
        self.fixed.name
    }

    fn source(&self) -> Source {
        let Fixed {
            entry, metal, cuda, ..
        } = *self.fixed;
        Source::Fixed { metal, cuda, entry }
    }

    fn launches(&self, step: &Step) -> Vec<Launch> {
        let counts: Vec<usize> = self.counts.iter().map(|c| *c as usize).collect();
        vec![Launch::plain(step, &counts, self.shape)]
    }

    fn work(&self, _: usize) -> f64 {
        self.work
    }

    fn sizes(&self) -> Vec<usize> {
        self.counts.iter().map(|c| *c as usize).collect()
    }

    fn guess(&self) -> f64 {
        self.guess
    }

    /// The module saw each read's form before it offered.
    fn accepts(&self, _: usize, _: &dyn sensai_lower::form::Load) -> bool {
        true
    }
}
