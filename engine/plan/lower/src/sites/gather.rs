use super::Pattern;
use crate::form::Load;
use crate::tree::{T, Term, Tree};
use graph::{Op, Select};

/// Rows of `table` picked by `ids`, along its first axis.
#[derive(Clone, Copy, Debug)]
pub struct Gather {
    pub table: T,
    pub ids: T,
    pub rows: usize,
    pub width: usize,
    pub table_rows: usize,
    pub form: &'static dyn Load,
}

impl Pattern for Gather {
    fn find(tree: &Tree, at: T) -> Option<Self> {
        let node = tree.node(at);
        if node.term != Term::Op(Op::Select(Select::Gather)) {
            return None;
        }
        let (table, ids) = (node.args[0], node.args[1]);
        let rows = tree.elements(ids);
        // Stored tables are matrices. Computed ones may have more axes.
        let shape = &tree.node(table).shape;
        let table_rows = match shape[..] {
            [rows, _] => rows,
            [rows, _, ..] if !tree.leaf(table) => rows,
            _ => return None,
        };
        Some(Self {
            table,
            ids,
            rows,
            width: tree.elements(table) / table_rows,
            table_rows,
            form: tree.node(table).form,
        })
    }
}
