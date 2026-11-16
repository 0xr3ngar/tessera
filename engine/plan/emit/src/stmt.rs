//! Statements: a loop over live lanes, or a new mask.

use crate::{Emitter, Kind, Res, rust_ty, var, zero};
use codegen::ir::{Expr, Intrinsic, Stmt, Ty};
use proc_macro2::TokenStream;
use quote::quote;

impl Emitter<'_> {
    /// `e` under the mask, in a fresh array of `ty`.
    fn lanes(&mut self, e: &Expr, ty: Ty) -> Result<(TokenStream, proc_macro2::Ident), String> {
        let mut pre = TokenStream::new();
        let e = self.hoist(e, &mut pre)?;
        let value = self.expr_as(&e, ty)?;
        let (rty, z, mask) = (rust_ty(ty)?, zero(ty)?, self.mask());
        let name = self.next("x");
        pre.extend(quote! {
            let mut #name: [#rty; THREADS] = [#z; THREADS];
            for t in 0..n {
                if #mask[t] {
                    #name[t] = #value;
                }
            }
        });
        Ok((pre, name))
    }

    /// `target = value`, or `target op= value`, in one lane.
    fn store(&mut self, target: &Expr, op: Option<codegen::ir::BinOp>, value: &Expr) -> Res {
        let combined = |op| match op {
            Some(op) => Expr::Bin(op, Box::new(target.clone()), Box::new(value.clone())),
            None => value.clone(),
        };
        match target {
            Expr::Var(name) => {
                let Kind::Lanes(ty) = self.kind(name)? else {
                    return Err(format!("`{name}` is not a lane variable"));
                };
                let (v, value) = (var(name), self.expr_as(&combined(op), ty)?);
                Ok(quote!(#v[t] = #value;))
            }
            Expr::Index(base, at) => {
                let Expr::Var(name) = &**base else {
                    return Err("an index names what it writes".into());
                };
                let v = var(name);
                match self.kind(name)? {
                    Kind::Buffer(form) => {
                        if form.narrow() {
                            return Err("kernels write only f32".into());
                        }
                        let b = quote::format_ident!("b_{name}");
                        let (_, _, skip) = self.buffer(base)?;
                        let at = self.index(at)?;
                        let (st, ty) = match form.dtype().is_float() {
                            true => (quote!(st_f32), Ty::F32),
                            false => (quote!(st_u32), Ty::U32),
                        };
                        let value = self.expr_as(&combined(op), ty)?;
                        Ok(quote! {
                            let at = #skip + #at;
                            let value = #value;
                            run::compiled::lockstep::#st(&mut *bufs[#b], at, value);
                        })
                    }
                    Kind::Shared(ty) => {
                        let (at, value) = (self.index(at)?, self.expr_as(&combined(op), ty)?);
                        Ok(quote!(#v[#at] = #value;))
                    }
                    Kind::Array(ty) => {
                        let (at, value) = (self.index(at)?, self.expr_as(&combined(op), ty)?);
                        Ok(quote!(#v[t][#at] = #value;))
                    }
                    _ => Err(format!("`{name}` is not written by index")),
                }
            }
            Expr::Field(base, h) => {
                let Expr::Var(name) = &**base else {
                    return Err("a field names its value".into());
                };
                let v = var(name);
                let i = ['x', 'y', 'z', 'w']
                    .iter()
                    .position(|c| c == h)
                    .unwrap_or(3);
                let value = self.expr_as(&combined(op), Ty::F32)?;
                Ok(quote!(#v[t][#i] = #value;))
            }
            other => Err(format!("cannot assign to {other:?}")),
        }
    }

    /// Clears every mask where the current one is live.
    fn leave(&self) -> TokenStream {
        let mask = self.mask();
        let outer = self.masks.to_vec();
        quote! {
            for t in 0..n {
                if #mask[t] {
                    #(#outer[t] = false;)*
                }
            }
        }
    }

    pub(crate) fn stmt(&mut self, s: &Stmt) -> Res {
        let mask = self.mask();
        match s {
            Stmt::Let { name, init } => {
                let ty = self.ty(init)?;
                let (pre, x) = self.lanes(init, ty)?;
                let v = var(name);
                self.bind(name, Kind::Lanes(ty));
                Ok(quote!(#pre let mut #v = #x;))
            }
            Stmt::Array { name, ty, len } => {
                let (v, rty, z) = (var(name), rust_ty(*ty)?, zero(*ty)?);
                self.bind(name, Kind::Array(*ty));
                Ok(quote!(let mut #v: Vec<[#rty; #len]> = vec![[#z; #len]; THREADS];))
            }
            Stmt::Shared { name, ty, len } => {
                let (v, rty, z) = (var(name), rust_ty(*ty)?, zero(*ty)?);
                self.bind(name, Kind::Shared(*ty));
                Ok(quote!(let mut #v: [#rty; #len] = [#z; #len];))
            }
            Stmt::Assign {
                target,
                op,
                value: Expr::Call(Intrinsic::Sum32, args),
            } => {
                if op.is_some() || args[0] != *target {
                    return Err("`v = sum32(v)`".into());
                }
                // The target sums over lanes, then goes back.
                let (pre, x) = self.lanes(target, Ty::F32)?;
                let name = x.to_string();
                self.bind(&name, Kind::Lanes(Ty::F32));
                let summed = Expr::Var(name);
                let store = self.store(target, None, &summed)?;
                let v = var(&x.to_string());
                Ok(quote! {
                    #pre
                    let mut #v = #x;
                    run::compiled::lockstep::sum32(&mut #v, &#mask, n, lanes);
                    for t in 0..n {
                        if #mask[t] {
                            #store
                        }
                    }
                })
            }
            Stmt::Assign { target, op, value } => {
                let mut pre = TokenStream::new();
                let value = self.hoist(value, &mut pre)?;
                let target = self.hoist(target, &mut pre)?;
                let store = self.store(&target, *op, &value)?;
                Ok(quote! {
                    #pre
                    for t in 0..n {
                        if #mask[t] {
                            #store
                        }
                    }
                })
            }
            Stmt::For {
                var: name,
                from,
                to,
                step,
                body,
                ..
            } => {
                let (pre_from, from) = self.lanes(from, Ty::U32)?;
                let (pre_to, to) = self.lanes(to, Ty::U32)?;
                let (pre_step, step) = self.lanes(step, Ty::U32)?;
                let (v, live) = (var(name), self.next("m"));
                self.scope.push(Default::default());
                self.bind(name, Kind::Lanes(Ty::U32));
                let block = self.block(body, live.clone())?;
                self.scope.pop();
                Ok(quote! {
                    #pre_from #pre_to #pre_step
                    let mut #v = #from;
                    loop {
                        let mut #live = [false; THREADS];
                        let mut any = false;
                        for t in 0..n {
                            #live[t] = #mask[t] && #v[t] < #to[t];
                            any |= #live[t];
                        }
                        if !any {
                            break;
                        }
                        #block
                        for t in 0..n {
                            if #live[t] {
                                #v[t] = #v[t].wrapping_add(#step[t]);
                            }
                        }
                    }
                })
            }
            Stmt::If {
                cond: Expr::Bool(taken),
                then,
                els,
            } => {
                let branch = if *taken { then } else { els };
                self.block(branch, mask)
            }
            Stmt::If { cond, then, els } => {
                let (pre, c) = self.lanes(cond, Ty::Bool)?;
                let (yes, no) = (self.next("m"), self.next("m"));
                let then = self.block(then, yes.clone())?;
                let els = match els.is_empty() {
                    true => TokenStream::new(),
                    false => {
                        let els = self.block(els, no.clone())?;
                        quote! {
                            let mut #no = [false; THREADS];
                            let mut off = false;
                            for t in 0..n {
                                #no[t] = #mask[t] && !#c[t];
                                off |= #no[t];
                            }
                            if off {
                                #els
                            }
                        }
                    }
                };
                Ok(quote! {
                    #pre
                    let mut #yes = [false; THREADS];
                    let mut on = false;
                    for t in 0..n {
                        #yes[t] = #mask[t] && #c[t];
                        on |= #yes[t];
                    }
                    if on {
                        #then
                    }
                    #els
                })
            }
            Stmt::Return => Ok(self.leave()),
            // Lockstep leaves every thread at the barrier already.
            Stmt::Barrier => Ok(TokenStream::new()),
            Stmt::Expr(e) => {
                let mut pre = TokenStream::new();
                let e = self.hoist(e, &mut pre)?;
                let e = self.expr(&e)?;
                Ok(quote! {
                    #pre
                    for t in 0..n {
                        if #mask[t] {
                            let _ = #e;
                        }
                    }
                })
            }
            Stmt::TileAcc { name, rows, cols } => {
                let v = var(name);
                self.bind(name, Kind::Tiles);
                Ok(
                    quote!(let mut #v = run::compiled::lockstep::Tiles::new(#rows, #cols, n, lanes);),
                )
            }
            Stmt::TileZero { acc } => {
                let v = var(acc);
                Ok(quote!(#v.zero();))
            }
            Stmt::TileScale { acc, by } => {
                let v = var(acc);
                self.scope.push(Default::default());
                self.bind("r", Kind::One(Ty::U32));
                let by = self.expr_as(by, Ty::F32)?;
                self.scope.pop();
                let r = var("r");
                Ok(quote! {
                    for e in 0..#v.per() {
                        for t in 0..n {
                            if #mask[t] {
                                let #r: u32 = #v.row(t, e);
                                let by = #by;
                                #v.scale(t, e, by);
                            }
                        }
                    }
                })
            }
            Stmt::TileBlock {
                acc,
                a,
                a_at,
                b,
                b_at,
                stride,
                depth,
            } => {
                let (Expr::Var(a), Expr::Var(b)) = (a, b) else {
                    return Err("tiles multiply group memory by name".into());
                };
                let (v, a, b) = (var(acc), var(a), var(b));
                let (pre_a, a_at) = self.lanes(a_at, Ty::U32)?;
                let (pre_b, b_at) = self.lanes(b_at, Ty::U32)?;
                let (pre_s, stride) = self.lanes(stride, Ty::U32)?;
                Ok(quote! {
                    #pre_a #pre_b #pre_s
                    #v.block(&#a, &#a_at, &#b, &#b_at, &#stride, #depth, &#mask, n, precision);
                })
            }
            Stmt::TileEach { acc, body } => {
                let (v, live) = (var(acc), self.next("m"));
                let (r, c, val) = (var("r"), var("c"), var("v"));
                self.scope.push(Default::default());
                for (name, ty) in [("r", Ty::U32), ("c", Ty::U32), ("v", Ty::F32)] {
                    self.bind(name, Kind::Lanes(ty));
                }
                let block = self.block(body, live.clone())?;
                self.scope.pop();
                Ok(quote! {
                    for e in 0..#v.per() {
                        let mut #live = [false; THREADS];
                        let mut #r = [0u32; THREADS];
                        let mut #c = [0u32; THREADS];
                        let mut #val = [0f32; THREADS];
                        for t in 0..n {
                            let (on, row, col, value) = #v.each(t, e, #mask[t]);
                            #live[t] = on;
                            #r[t] = row;
                            #c[t] = col;
                            #val[t] = value;
                        }
                        #block
                    }
                })
            }
        }
    }
}
