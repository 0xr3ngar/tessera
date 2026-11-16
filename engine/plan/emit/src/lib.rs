//! Rust source for kernel bodies: one function per body, lockstep.

mod expr;
mod stmt;

use codegen::ir::{Body, Param, Ty};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{format_ident, quote};
use std::collections::HashMap;

pub(crate) type Res = Result<TokenStream, String>;

/// What a name in scope is.
#[derive(Clone, Copy)]
pub(crate) enum Kind {
    /// One value per thread, indexed by `t`.
    Lanes(Ty),
    /// One value for every thread.
    One(Ty),
    Buffer(&'static dyn codegen::Form),
    /// A local array per thread.
    Array(Ty),
    /// Group memory.
    Shared(Ty),
    Tiles,
}

pub(crate) struct Emitter<'a> {
    scope: Vec<HashMap<String, Kind>>,
    /// Masks of the enclosing blocks, outermost first.
    masks: Vec<Ident>,
    fresh: usize,
    layout: Option<&'static dyn codegen::kept::Layout>,
    params: &'a [Param],
}

impl Emitter<'_> {
    pub(crate) fn kind(&self, name: &str) -> Result<Kind, String> {
        for scope in self.scope.iter().rev() {
            if let Some(k) = scope.get(name) {
                return Ok(*k);
            }
        }
        Err(format!("`{name}` is not in scope"))
    }

    pub(crate) fn bind(&mut self, name: &str, kind: Kind) {
        let last = self.scope.last_mut().expect("a scope");
        last.insert(name.into(), kind);
    }

    pub(crate) fn mask(&self) -> Ident {
        self.masks.last().expect("a mask").clone()
    }

    pub(crate) fn next(&mut self, prefix: &str) -> Ident {
        self.fresh += 1;
        format_ident!("{prefix}{}", self.fresh)
    }

    pub(crate) fn layout(&self) -> Result<&'static dyn codegen::kept::Layout, String> {
        self.layout
            .ok_or_else(|| "kept rows without a layout".into())
    }

    /// `stmts` under `mask`, in their own scope.
    pub(crate) fn block(&mut self, stmts: &[codegen::ir::Stmt], mask: Ident) -> Res {
        self.scope.push(HashMap::new());
        self.masks.push(mask);
        let mut out = TokenStream::new();
        for s in stmts {
            out.extend(self.stmt(s)?);
        }
        self.masks.pop();
        self.scope.pop();
        Ok(quote!({ #out }))
    }
}

/// The variable holding `name`.
pub(crate) fn var(name: &str) -> Ident {
    format_ident!("v_{name}")
}

pub(crate) fn rust_ty(ty: Ty) -> Res {
    Ok(match ty {
        Ty::F32 => quote!(f32),
        Ty::U32 => quote!(u32),
        Ty::U64 => quote!(u64),
        Ty::Bool => quote!(bool),
        Ty::F32x4 => quote!([f32; 4]),
        Ty::Bytes => return Err("a byte pointer as a value".into()),
    })
}

pub(crate) fn zero(ty: Ty) -> Res {
    Ok(match ty {
        Ty::F32 => quote!(0f32),
        Ty::U32 => quote!(0u32),
        Ty::U64 => quote!(0u64),
        Ty::Bool => quote!(false),
        Ty::F32x4 => quote!([0f32; 4]),
        Ty::Bytes => return Err("a byte pointer as a value".into()),
    })
}

/// The function running `body`, named by `hash`.
pub fn function(body: &Body, hash: u64) -> Res {
    let mut e = Emitter {
        scope: vec![HashMap::new()],
        masks: Vec::new(),
        fresh: 0,
        layout: body.layout,
        params: &body.params,
    };
    let mut binds = TokenStream::new();
    for (i, param) in e.params.iter().enumerate() {
        match param {
            Param::Buffer { name, form, .. } => {
                let (b, s) = (format_ident!("b_{name}"), format_ident!("s_{name}"));
                binds.extend(quote!(let (#b, #s) = run::compiled::lockstep::buf(&args[#i]);));
                e.bind(name, Kind::Buffer(*form));
            }
            Param::Scalar { name, ty } => {
                let v = var(name);
                let read = match ty {
                    Ty::U32 => quote!(run::compiled::lockstep::u32),
                    Ty::F32 => quote!(run::compiled::lockstep::f32),
                    other => return Err(format!("a {other:?} scalar")),
                };
                binds.extend(quote!(let #v = #read(&args[#i]);));
                e.bind(name, Kind::One(*ty));
            }
            other => return Err(format!("{other:?} as an entry parameter")),
        }
    }
    let m0 = Ident::new("m0", Span::call_site());
    let block = e.block(&body.stmts, m0.clone())?;
    let name = format_ident!("k_{hash:016x}");
    Ok(quote! {
        pub fn #name(
            args: &[run::compiled::exec::ArgVal],
            bufs: &mut [&mut [u8]],
            grid: run::compiled::exec::Grid,
            lanes: usize,
            precision: run::compiled::Precision,
        ) -> bool {
            const THREADS: usize = run::compiled::lockstep::THREADS;
            if grid.threads > THREADS {
                return false;
            }
            let n = grid.threads;
            #binds
            for g in 0..grid.groups {
                let first = g * n;
                let mut #m0 = [false; THREADS];
                for t in 0..n {
                    #m0[t] = true;
                }
                #block
            }
            true
        }
    })
}

/// A module of every body that emits, registered with `run`.
pub fn module(bodies: &[(u64, Body)]) -> (String, Vec<(String, String)>) {
    let mut items = TokenStream::new();
    let mut skipped = Vec::new();
    for (hash, body) in bodies {
        match function(body, *hash) {
            Ok(f) => {
                let name = format_ident!("k_{hash:016x}");
                items.extend(f);
                items.extend(quote!(run::compiled!(#hash, #name);));
            }
            Err(why) => skipped.push((body.entry.clone(), why)),
        }
    }
    let file: syn::File = syn::parse2(items).expect("emitted items parse");
    (prettyplease::unparse(&file), skipped)
}
