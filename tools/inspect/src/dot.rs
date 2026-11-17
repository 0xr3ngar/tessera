//! A small Graphviz builder, and the graph drawn with it.

use crate::model::Model;
use crate::shards::{groups, local};
use crate::stages::{Pass, RankView};
use graph::{Graph, Op, Placement, Stage, TensorType, View};
use std::fmt::Write;

type Attrs = Vec<(&'static str, String)>;

pub enum Stmt {
    Node {
        id: String,
        attrs: Attrs,
    },
    Edge {
        from: String,
        to: String,
    },
    Cluster {
        id: String,
        label: String,
        body: Vec<Stmt>,
    },
}

/// Graphviz text quotes. Line breaks become `\n`.
fn quote(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn render(stmts: &[Stmt], depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    for s in stmts {
        match s {
            Stmt::Node { id, attrs } => {
                let attrs: Vec<String> = attrs
                    .iter()
                    .map(|(k, v)| format!("{k}={}", quote(v)))
                    .collect();
                let _ = writeln!(out, "{pad}{} [{}];", quote(id), attrs.join(", "));
            }
            Stmt::Edge { from, to } => {
                let _ = writeln!(out, "{pad}{} -> {};", quote(from), quote(to));
            }
            Stmt::Cluster { id, label, body } => {
                let _ = writeln!(out, "{pad}subgraph {} {{", quote(&format!("cluster_{id}")));
                let _ = writeln!(out, "{pad}  label={}; style=rounded;", quote(label));
                render(body, depth + 1, out);
                let _ = writeln!(out, "{pad}}}");
            }
        }
    }
}

/// The whole digraph, top to bottom.
pub fn digraph(name: &str, stmts: &[Stmt]) -> String {
    let mut out = format!("digraph {}", quote(name));
    out.push_str(" {");
    out.push('\n');
    out.push_str("  rankdir=TB;\n  node [fontname=\"Helvetica\", fontsize=10];\n");
    render(stmts, 1, &mut out);
    out.push_str("}\n");
    out
}

/// A stage's name without its fields.
fn stage(s: &Stage) -> String {
    let text = format!("{s:?}");
    let end = text.find([' ', '(']).unwrap_or(text.len());
    text[..end].to_string()
}

/// A weight's path, joined with dots.
pub fn path(g: &Graph, w: u32) -> String {
    let segs = g.weights[w as usize].path.iter().map(|s| match s {
        graph::PathSeg::Field(f) => f.clone(),
        graph::PathSeg::Index(i) => i.to_string(),
    });
    segs.collect::<Vec<_>>().join(".")
}

/// One line naming what a node does.
pub fn op_label(g: &Graph, op: &Op) -> String {
    match op {
        Op::Input { port, .. } => format!("input {:?}", g.entries[0].inputs[*port as usize].role),
        Op::Weight(w) => path(g, *w),
        Op::Row(r) => r.0.iter().map(stage).collect::<Vec<_>>().join(" · "),
        Op::View(View::Reshape(s)) => format!("reshape {s}"),
        Op::View(v) => format!("{v:?}").to_lowercase(),
        Op::Attend(a) => format!("attend {:?}", a.mask),
        Op::Append(k) => format!("append kept {k}"),
        other => format!("{other:?}").to_lowercase(),
    }
}

fn type_label(ty: &TensorType) -> String {
    let place = match ty.placement {
        Placement::Replicated => String::new(),
        p => format!(" {p:?}"),
    };
    format!("{} {:?}{place}", ty.shape, ty.dtype)
}

fn fill(ty: &TensorType, added: bool) -> &'static str {
    match (added, ty.placement) {
        (true, _) => "#f4a6a6",
        (_, Placement::Split { .. }) => "#a8d0f0",
        (_, Placement::Partial) => "#f6c88a",
        (_, Placement::Routed) => "#d4b3e8",
        (_, Placement::Replicated) => "#ffffff",
    }
}

/// One graph node, or one rank's copy of it.
fn node(model: &Model, sharded: bool, i: usize, copy: Option<(usize, &RankView)>) -> Stmt {
    let g = if sharded {
        &model.sharded
    } else {
        &model.source
    };
    let n = &g.nodes[i];
    let (src, added) = match sharded {
        true => (model.origin[i].0 as usize, model.added[i]),
        false => (i, false),
    };
    let ty = match &n.op {
        Op::Weight(w) => &g.weights[*w as usize].ty,
        _ => &n.ty,
    };
    let tag = match added {
        true => format!("sum of #{src}"),
        false => format!("#{src}"),
    };
    let shape = match &n.op {
        Op::Weight(_) => "note",
        Op::Input { .. } => "ellipse",
        _ if added => "doubleoctagon",
        _ => "box",
    };
    let held = match copy {
        Some((_, rank)) => local(model, rank, i),
        None => type_label(ty),
    };
    let label = format!("{tag} {}\n{held}", op_label(g, &n.op));
    Stmt::Node {
        id: id(i, copy.map(|c| c.0)),
        attrs: vec![
            ("label", label),
            ("shape", shape.into()),
            ("style", "filled".into()),
            ("fillcolor", fill(ty, added).into()),
            ("URL", format!("lowering.html#n{src}")),
        ],
    }
}

fn id(i: usize, rank: Option<usize>) -> String {
    match rank {
        Some(r) => format!("n{i}r{r}"),
        None => format!("n{i}"),
    }
}

/// How the drawing places split nodes.
struct Layout<'a> {
    model: &'a Model,
    sharded: bool,
    /// Per node, the sum that joins its group.
    group: Vec<Option<usize>>,
    pass: Option<&'a Pass>,
}

impl Layout<'_> {
    /// Plain nodes in `range`, then each group summed there.
    fn body(&self, range: std::ops::Range<usize>, sums: &[usize]) -> Vec<Stmt> {
        let mut out = Vec::new();
        for i in range.filter(|i| self.group[*i].is_none()) {
            out.push(node(self.model, self.sharded, i, None));
        }
        let Some(pass) = self.pass else {
            return out;
        };
        let total: u32 = pass.ranks.iter().map(|r| r.part).sum();
        for &sum in sums {
            for (r, rank) in pass.ranks.iter().enumerate() {
                let mut body = Vec::new();
                for i in (0..self.group.len()).filter(|i| self.group[*i] == Some(sum)) {
                    body.push(node(self.model, self.sharded, i, Some((r, rank))));
                }
                out.push(Stmt::Cluster {
                    id: format!("s{sum}r{r}"),
                    label: format!("{}, {} of {total}", rank.name, rank.part),
                    body,
                });
            }
        }
        out
    }

    /// One edge, or one per rank for split ends.
    fn edges(&self, from: usize, to: usize, out: &mut Vec<Stmt>) {
        let ranks = self.pass.map_or(0, |p| p.ranks.len());
        let (a, b) = (self.group[from].is_some(), self.group[to].is_some());
        if !a && !b {
            out.push(Stmt::Edge {
                from: id(from, None),
                to: id(to, None),
            });
            return;
        }
        for r in 0..ranks {
            out.push(Stmt::Edge {
                from: id(from, a.then_some(r)),
                to: id(to, b.then_some(r)),
            });
        }
    }
}

/// The source graph, or sharded with ranks side by side.
pub fn draw(model: &Model, sharded: bool, pass: Option<&Pass>) -> String {
    let g = if sharded {
        &model.sharded
    } else {
        &model.source
    };
    let group = match (sharded, pass) {
        (true, Some(_)) => groups(model),
        _ => vec![None; g.nodes.len()],
    };
    let layout = Layout {
        model,
        sharded,
        group,
        pass,
    };
    let mut sums: Vec<usize> = layout.group.iter().flatten().copied().collect();
    sums.sort_unstable();
    sums.dedup();
    let mut stmts = Vec::new();
    let mut inside = vec![false; g.nodes.len()];
    for (k, r) in g.regions.iter().enumerate() {
        let (start, end) = (r.start.0 as usize, r.end.0 as usize);
        let here: Vec<usize> = sums
            .iter()
            .copied()
            .filter(|s| (start..end).contains(s))
            .collect();
        sums.retain(|s| !here.contains(s));
        inside[start..end].iter_mut().for_each(|x| *x = true);
        stmts.push(Stmt::Cluster {
            id: k.to_string(),
            label: format!("{:?}", r.kind),
            body: layout.body(start..end, &here),
        });
    }
    let outside: Vec<usize> = (0..g.nodes.len()).filter(|i| !inside[*i]).collect();
    for i in outside
        .iter()
        .copied()
        .filter(|i| layout.group[*i].is_none())
    {
        stmts.push(node(model, sharded, i, None));
    }
    stmts.extend(layout.body(0..0, &sums));
    for (i, n) in g.nodes.iter().enumerate() {
        for a in &n.args {
            layout.edges(a.0 as usize, i, &mut stmts);
        }
    }
    for (k, o) in g.entries[0].outputs.iter().enumerate() {
        stmts.push(Stmt::Node {
            id: format!("out{k}"),
            attrs: vec![
                ("label", format!("{:?}", o.role)),
                ("shape", "ellipse".into()),
            ],
        });
        stmts.push(Stmt::Edge {
            from: format!("n{}", o.node.0),
            to: format!("out{k}"),
        });
    }
    digraph(&model.name, &stmts)
}
