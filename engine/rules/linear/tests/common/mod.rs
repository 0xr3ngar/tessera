//! Shared graphs and inputs for the matmul tests.

use sensai_codegen::ir::Param;
use sensai_graph::DimName::{Ffn, Hidden, Rows};
use sensai_interp::Tensor;
use sensai_lower::{Plan, Source, Terms};
use sensai_testkit::Noise;
use sensai_trace::{DType, Dim, EntryKind, Graph, Linear, Model, Port, Role, TensorType, Weights};
use std::collections::HashMap;

#[derive(Weights)]
struct Layer {
    #[shape(Hidden, Ffn)]
    w: Linear,
}

/// A graph's work on its input and the product.
pub type Finish = fn(&sensai_trace::Tensor, sensai_trace::Tensor) -> sensai_trace::Tensor;

/// `k` inputs to `n` outputs, over any number of rows.
pub fn linear(k: u64, n: u64) -> Graph {
    finished(k, n, |_, product| product)
}

/// The same, with stages after the matmul.
pub fn finished(k: u64, n: u64, finish: Finish) -> Graph {
    let rows = Dim::bounded(Rows, 256);
    let m = Model::new(vec![rows, Dim::fixed(Hidden, k), Dim::fixed(Ffn, n)]);
    let w: Layer = sensai_trace::declare(&m);
    let ty = TensorType::new([Rows, Hidden], DType::F32);
    let role = Role::Logits;
    let [x] = m.entry(EntryKind::Forward, [Port { role, ty }]);
    m.finish([(Role::Logits, &finish(&x, x.matmul(&w.w)))])
}

pub fn noise(shape: Vec<usize>, noise: &mut Noise) -> Tensor {
    let data = (0..shape.iter().product()).map(|_| noise.draw()).collect();
    Tensor::f32(shape, data)
}

/// The sums one output of `plan`'s only step gathers.
pub fn terms(plan: &Plan) -> Terms {
    let step = &plan.steps[0];
    let Source::Made(body) = step.kernel.source() else {
        panic!("a generated kernel");
    };
    let launch = step.kernel.launches(step).remove(0);
    let mut params = HashMap::new();
    for (param, arg) in body.params.iter().zip(&launch.args) {
        let Param::Scalar { name, .. } = param else {
            continue;
        };
        let Param::Scalar { ty, .. } = param else {
            unreachable!()
        };
        let value = match ty {
            sensai_codegen::ir::Ty::F32 => arg.f32().map(f64::from),
            _ => arg.u32().map(f64::from),
        };
        let value = value.unwrap_or_else(|| panic!("{arg:?} is not a scalar"));
        params.insert(name.clone(), value);
    }
    sensai_lower::terms(&body, &params).expect("every loop bound is a parameter")
}

/// Every weight form whose blocks fit rows of `k`.
pub fn forms(k: usize) -> Vec<&'static dyn sensai_lower::form::Load> {
    let fits = |f: &&&dyn sensai_lower::form::Load| k.is_multiple_of(f.align());
    sensai_lower::form::LOADS
        .iter()
        .filter(fits)
        .copied()
        .filter(|f| f.dtype().is_float() && f.packs())
        .collect()
}
