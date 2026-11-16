//! Re-exports of other crates that no other crate uses.

use std::collections::BTreeSet;

use syn::{Item, UseTree, Visibility};

use super::File;
use crate::Violation;
use crate::audit::unused::{is_target, leaves};
use crate::audit::workspace::Pkg;
use crate::rust::{attrs_of, is_test};

/// The first segment of a `use` tree.
fn root(tree: &UseTree) -> Option<String> {
    match tree {
        UseTree::Path(p) => Some(p.ident.to_string()),
        _ => None,
    }
}

/// Names and lines of `pub use` lines of `crates`.
fn reexports(items: &[Item], crates: &BTreeSet<String>, out: &mut Vec<(String, usize)>) {
    for item in items {
        if attrs_of(item).iter().any(is_test) {
            continue;
        }
        match item {
            Item::Use(u) if matches!(u.vis, Visibility::Public(_)) => {
                if root(&u.tree).is_some_and(|r| crates.contains(&r)) {
                    let mut found = Vec::new();
                    leaves(&u.tree, &mut found);
                    out.extend(found.iter().map(|i| (i.to_string(), i.span().start().line)));
                }
            }
            Item::Mod(m) if m.ident != "testing" => {
                let inner = m.content.iter().flat_map(|(_, i)| i.iter());
                reexports(&inner.cloned().collect::<Vec<_>>(), crates, out);
            }
            _ => {}
        }
    }
}

/// Names reached through `me`, or `None` after a glob.
fn reached<'a>(files: &'a [File], me: &Pkg) -> Option<BTreeSet<&'a str>> {
    let ident = me.name.replace('-', "_");
    let mut found = BTreeSet::new();
    for f in files {
        let outside = f.pkg.name != me.name || is_target(f.rel);
        let via = if outside { ident.as_str() } else { "$crate" };
        for n in f.code.iter().chain(&f.test).filter(|n| n.segs[0] == via) {
            if n.segs.last().is_some_and(|s| s == "*") {
                return None;
            }
            found.extend(n.segs[1..].iter().map(String::as_str));
        }
    }
    Some(found)
}

pub fn check(pkgs: &[Pkg], files: &[File], out: &mut Vec<Violation>) {
    let crates: BTreeSet<String> = pkgs.iter().map(|p| p.name.replace('-', "_")).collect();
    for p in pkgs {
        let Some(used) = reached(files, p) else {
            continue;
        };
        let mut others = crates.clone();
        others.remove(&p.name.replace('-', "_"));
        for f in files
            .iter()
            .filter(|f| f.pkg.name == p.name && !f.code.is_empty() && !is_target(f.rel))
        {
            let Ok(file) = syn::parse_file(&f.src.text) else {
                continue;
            };
            let mut names = Vec::new();
            reexports(&file.items, &others, &mut names);
            for (name, line) in names
                .into_iter()
                .filter(|(n, _)| !used.contains(n.as_str()))
            {
                out.push(Violation {
                    path: f.src.path.clone(),
                    line,
                    rule: "reexport",
                    message: format!(
                        "`{name}` is re-exported but no other crate names it through `{}`: make it private or remove it",
                        p.name
                    ),
                });
            }
        }
    }
}
