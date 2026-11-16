//! Expressions, each lane's value inside a loop over `t`.

use crate::{Emitter, Kind, Res, var};
use codegen::ir::{BinOp, Expr, Intrinsic, Ty};
use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};

/// The wider of two types, as C promotes them.
fn wider(a: Ty, b: Ty) -> Ty {
    match (a, b) {
        (Ty::Bytes, _) | (_, Ty::Bytes) => Ty::Bytes,
        (Ty::F32x4, _) | (_, Ty::F32x4) => Ty::F32x4,
        (Ty::F32, _) | (_, Ty::F32) => Ty::F32,
        (Ty::U64, _) | (_, Ty::U64) => Ty::U64,
        _ => Ty::U32,
    }
}

/// `e` of type `from`, as `to`.
pub(crate) fn cast(e: TokenStream, from: Ty, to: Ty) -> Res {
    if from == to {
        return Ok(e);
    }
    Ok(match (from, to) {
        (Ty::F32, Ty::Bool) => quote!((#e != 0.0)),
        (Ty::U32 | Ty::U64, Ty::Bool) => quote!((#e != 0)),
        (Ty::Bool, Ty::F32) => quote!(((#e as u32) as f32)),
        (Ty::F32, Ty::F32x4) => quote!([#e; 4]),
        (_, Ty::F32x4) | (Ty::F32x4, _) | (Ty::Bytes, _) | (_, Ty::Bytes) => {
            return Err(format!("no cast from {from:?} to {to:?}"));
        }
        _ => {
            let ty = crate::rust_ty(to)?;
            quote!((#e as #ty))
        }
    })
}

fn float(x: f64) -> TokenStream {
    let x = x as f32;
    if x.is_infinite() {
        return match x > 0.0 {
            true => quote!(f32::INFINITY),
            false => quote!(f32::NEG_INFINITY),
        };
    }
    let lit = Literal::f32_suffixed(x);
    quote!(#lit)
}

/// The reader for one element of a form.
fn reader(form: &dyn codegen::Form) -> Res {
    Ok(match form.name() {
        "f32" => quote!(run::compiled::lockstep::rd_f32),
        "bf16" => quote!(run::compiled::lockstep::rd_bf16),
        "u32" => quote!(run::compiled::lockstep::rd_u32),
        "u8" => quote!(run::compiled::lockstep::rd_u8),
        "q8_0" => quote!(run::compiled::lockstep::rd_q8_0),
        "q4_0" => quote!(run::compiled::lockstep::rd_q4_0),
        "nvfp4" => quote!(run::compiled::lockstep::rd_nvfp4),
        other => return Err(format!("no reader for the {other} form")),
    })
}

fn element(form: &dyn codegen::Form) -> Ty {
    match form.dtype().is_float() {
        true => Ty::F32,
        false => Ty::U32,
    }
}

impl Emitter<'_> {
    /// The type an expression has, from the names it uses.
    pub(crate) fn ty(&self, e: &Expr) -> Result<Ty, String> {
        Ok(match e {
            Expr::Int(_) => Ty::U32,
            Expr::Float(_) => Ty::F32,
            Expr::Bool(_) => Ty::Bool,
            Expr::Var(name) => match self.kind(name)? {
                Kind::Lanes(ty) | Kind::One(ty) => ty,
                Kind::Buffer(_) => Ty::Bytes,
                _ => return Err(format!("`{name}` is not a value")),
            },
            Expr::Neg(a) | Expr::Not(a) => self.ty(a)?,
            Expr::Bin(op, a, b) => match op.compares() {
                true => Ty::Bool,
                false => wider(self.ty(a)?, self.ty(b)?),
            },
            Expr::Cast(ty, _) => *ty,
            Expr::Index(base, _) => match &**base {
                Expr::Var(name) => match self.kind(name)? {
                    Kind::Array(ty) | Kind::Shared(ty) => ty,
                    Kind::Buffer(form) => element(form),
                    _ => Ty::F32,
                },
                _ => Ty::F32,
            },
            Expr::Field(..) => Ty::F32,
            Expr::Select(c, a, b) => match **c {
                Expr::Bool(false) => self.ty(b)?,
                _ => self.ty(a)?,
            },
            Expr::Fn(_, ty, _) => *ty,
            Expr::Call(i, _) => i.ty(),
        })
    }

    /// `e` as `to`.
    pub(crate) fn expr_as(&mut self, e: &Expr, to: Ty) -> Res {
        let from = self.ty(e)?;
        cast(self.expr(e)?, from, to)
    }

    /// The bytes of the buffer `e` names.
    fn bytes(&self, e: &Expr) -> Res {
        let Expr::Var(name) = e else {
            return Err("a buffer is named directly".into());
        };
        let Kind::Buffer(_) = self.kind(name)? else {
            return Err(format!("`{name}` is not a buffer"));
        };
        let b = format_ident!("b_{name}");
        Ok(quote!(&*bufs[#b]))
    }

    /// The buffer `e` names: its form, bytes and skip.
    pub(crate) fn buffer(
        &self,
        e: &Expr,
    ) -> Result<(&'static dyn codegen::Form, TokenStream, TokenStream), String> {
        let Expr::Var(name) = e else {
            return Err("a buffer is named directly".into());
        };
        let Kind::Buffer(form) = self.kind(name)? else {
            return Err(format!("`{name}` is not a buffer"));
        };
        let s = format_ident!("s_{name}");
        Ok((form, self.bytes(e)?, quote!(#s)))
    }

    /// An index as a `usize`.
    pub(crate) fn index(&mut self, at: &Expr) -> Res {
        let at = self.expr_as(at, Ty::U64)?;
        Ok(quote!((#at as usize)))
    }

    pub(crate) fn expr(&mut self, e: &Expr) -> Res {
        match e {
            Expr::Int(n) => {
                let lit = Literal::u32_suffixed(*n as u32);
                Ok(quote!(#lit))
            }
            Expr::Float(x) => Ok(float(*x)),
            Expr::Bool(b) => Ok(quote!(#b)),
            Expr::Var(name) => {
                let v = var(name);
                match self.kind(name)? {
                    Kind::Lanes(_) => Ok(quote!(#v[t])),
                    Kind::One(_) => Ok(quote!(#v)),
                    _ => Err(format!("`{name}` is not a value")),
                }
            }
            Expr::Neg(a) => {
                let (ty, a) = (self.ty(a)?, self.expr(a)?);
                Ok(match ty {
                    Ty::F32 => quote!((-#a)),
                    Ty::U32 | Ty::U64 => quote!(#a.wrapping_neg()),
                    other => return Err(format!("negating a {other:?}")),
                })
            }
            Expr::Not(a) => {
                let a = self.expr(a)?;
                Ok(quote!((!#a)))
            }
            Expr::Bin(op, a, b) => self.binary(*op, a, b),
            Expr::Cast(ty, a) => self.expr_as(a, *ty),
            Expr::Index(base, at) => {
                let Expr::Var(name) = &**base else {
                    return Err("an index names what it reads".into());
                };
                let v = var(name);
                let at = self.index(at)?;
                match self.kind(name)? {
                    Kind::Buffer(form) => {
                        let (_, bytes, skip) = self.buffer(base)?;
                        let rd = reader(form)?;
                        Ok(quote!(#rd(#bytes, #skip + #at)))
                    }
                    Kind::Shared(_) => Ok(quote!(#v[#at])),
                    Kind::Array(_) => Ok(quote!(#v[t][#at])),
                    _ => Err(format!("`{name}` is not indexed")),
                }
            }
            Expr::Field(base, h) => {
                let base = self.expr(base)?;
                let i = ['x', 'y', 'z', 'w']
                    .iter()
                    .position(|c| c == h)
                    .unwrap_or(3);
                Ok(quote!(#base[#i]))
            }
            Expr::Select(c, a, b) if matches!(**c, Expr::Bool(_)) => {
                let taken = if **c == Expr::Bool(true) { a } else { b };
                self.expr(taken)
            }
            Expr::Select(c, a, b) => {
                let ty = self.ty(a)?;
                let (c, a, b) = (
                    self.expr_as(c, Ty::Bool)?,
                    self.expr(a)?,
                    self.expr_as(b, ty)?,
                );
                Ok(quote!((if #c { #a } else { #b })))
            }
            Expr::Call(i, args) => self.call(*i, args),
            Expr::Fn(name, ..) => Err(format!("`{name}` is a device helper")),
        }
    }

    fn binary(&mut self, op: BinOp, a: &Expr, b: &Expr) -> Res {
        let (ta, tb) = (self.ty(a)?, self.ty(b)?);
        if matches!(op, BinOp::And | BinOp::Or) {
            let (a, b) = (self.expr_as(a, Ty::Bool)?, self.expr_as(b, Ty::Bool)?);
            return Ok(match op {
                BinOp::And => quote!((#a && #b)),
                _ => quote!((#a || #b)),
            });
        }
        let w = wider(ta, tb);
        let (a, b) = (self.expr_as(a, w)?, self.expr_as(b, w)?);
        if op.compares() {
            let cmp = match op {
                BinOp::Lt => quote!(<),
                BinOp::Le => quote!(<=),
                BinOp::Gt => quote!(>),
                BinOp::Ge => quote!(>=),
                BinOp::Eq => quote!(==),
                _ => quote!(!=),
            };
            return Ok(quote!((#a #cmp #b)));
        }
        if w == Ty::F32x4 {
            let f = match op {
                BinOp::Add => quote!(add4),
                BinOp::Sub => quote!(sub4),
                BinOp::Mul => quote!(mul4),
                BinOp::Div => quote!(div4),
                other => return Err(format!("{other:?} on float4")),
            };
            return Ok(quote!(run::compiled::lockstep::#f(#a, #b)));
        }
        let float = w == Ty::F32;
        Ok(match op {
            BinOp::Add if float => quote!((#a + #b)),
            BinOp::Sub if float => quote!((#a - #b)),
            BinOp::Mul if float => quote!((#a * #b)),
            BinOp::Add => quote!(#a.wrapping_add(#b)),
            BinOp::Sub => quote!(#a.wrapping_sub(#b)),
            BinOp::Mul => quote!(#a.wrapping_mul(#b)),
            BinOp::Div => quote!((#a / #b)),
            BinOp::Rem => quote!((#a % #b)),
            BinOp::Shl => quote!((#a << #b)),
            BinOp::Shr => quote!((#a >> #b)),
            BinOp::BitAnd => quote!((#a & #b)),
            BinOp::BitOr => quote!((#a | #b)),
            BinOp::BitXor => quote!((#a ^ #b)),
            other => return Err(format!("{other:?} as arithmetic")),
        })
    }

    /// Float arguments, each as f32.
    fn floats(&mut self, args: &[Expr]) -> Result<Vec<TokenStream>, String> {
        args.iter().map(|a| self.expr_as(a, Ty::F32)).collect()
    }

    fn call(&mut self, i: Intrinsic, args: &[Expr]) -> Res {
        let f = |e: &mut Self| e.floats(args);
        Ok(match i {
            Intrinsic::Gid => quote!(((first + t) as u32)),
            Intrinsic::Lid => quote!((t as u32)),
            Intrinsic::Lane => quote!(((t % lanes) as u32)),
            Intrinsic::ShflXor | Intrinsic::Sum32 => {
                return Err("a lane shuffle outside a statement".into());
            }
            Intrinsic::Exp => {
                let a = f(self)?;
                quote!(#(#a)*.exp())
            }
            Intrinsic::Tanh => {
                let a = f(self)?;
                quote!(#(#a)*.tanh())
            }
            Intrinsic::Cos => {
                let a = f(self)?;
                quote!(#(#a)*.cos())
            }
            Intrinsic::Sin => {
                let a = f(self)?;
                quote!(#(#a)*.sin())
            }
            Intrinsic::Rsqrt => {
                let a = f(self)?;
                quote!((1.0f32 / #(#a)*.sqrt()))
            }
            Intrinsic::Max => {
                let a = f(self)?;
                quote!(f32::max(#(#a),*))
            }
            Intrinsic::Pow => {
                let a = f(self)?;
                quote!(f32::powf(#(#a),*))
            }
            Intrinsic::Clamp => {
                let a = f(self)?;
                let (x, lo, hi) = (&a[0], &a[1], &a[2]);
                quote!(#x.clamp(#lo, #hi))
            }
            Intrinsic::FromBits => {
                let a = self.expr_as(&args[0], Ty::U32)?;
                quote!(f32::from_bits(#a))
            }
            Intrinsic::ToBits => {
                let a = self.expr_as(&args[0], Ty::F32)?;
                quote!(#a.to_bits())
            }
            Intrinsic::Load4 => {
                let (form, bytes, skip) = self.buffer(&args[0])?;
                let rd = reader(form)?;
                let at = self.index(&args[1])?;
                quote!({
                    let at = #skip + #at;
                    [#rd(#bytes, at), #rd(#bytes, at + 1), #rd(#bytes, at + 2), #rd(#bytes, at + 3)]
                })
            }
            Intrinsic::Store4 => {
                let Expr::Var(name) = &args[0] else {
                    return Err("a store names its buffer".into());
                };
                let (b, s) = (format_ident!("b_{name}"), format_ident!("s_{name}"));
                let at = self.index(&args[1])?;
                let v = self.expr_as(&args[2], Ty::F32x4)?;
                quote!({
                    let at = #s + #at;
                    let v: [f32; 4] = #v;
                    for i in 0..4 {
                        run::compiled::lockstep::st_f32(&mut *bufs[#b], at + i, v[i]);
                    }
                })
            }
            Intrinsic::Zero4 => quote!([0f32; 4]),
            Intrinsic::Make4 => {
                let a = f(self)?;
                quote!([#(#a),*])
            }
            Intrinsic::KeptRow => {
                let layout = self.layout()?;
                let (s, p) = (
                    self.expr_as(&args[0], Ty::U32)?,
                    self.expr_as(&args[1], Ty::U32)?,
                );
                let values: Vec<_> = layout.params().iter().map(|p| var(p)).collect();
                let name = match layout.name() {
                    "dense" => quote!(run::compiled::kept::DENSE),
                    "interleaved" => quote!(run::compiled::kept::INTERLEAVED),
                    other => return Err(format!("no `{other}` layout on the host")),
                };
                quote!(#name.at(#s, #p, &[#(#values),*]))
            }
            Intrinsic::Word | Intrinsic::Half => {
                let (_, bytes, skip) = self.buffer(&args[0])?;
                let at = self.index(&args[1])?;
                let f = match i {
                    Intrinsic::Word => quote!(word),
                    _ => quote!(half),
                };
                quote!(run::compiled::lockstep::#f(#bytes, #skip + #at))
            }
            Intrinsic::Bits | Intrinsic::Signed => {
                let w = self.expr_as(&args[0], Ty::U32)?;
                let l = self.expr_as(&args[1], Ty::U32)?;
                let n = self.expr_as(&args[2], Ty::U32)?;
                let f = match i {
                    Intrinsic::Bits => quote!(field),
                    _ => quote!(signed),
                };
                quote!(run::compiled::lockstep::#f(#w, #l, #n))
            }
            Intrinsic::F16 | Intrinsic::E4M3 | Intrinsic::E2M1 => {
                let w = self.expr_as(&args[0], Ty::U32)?;
                let l = self.expr_as(&args[1], Ty::U32)?;
                let f = match i {
                    Intrinsic::F16 => quote!(f16),
                    Intrinsic::E4M3 => quote!(e4m3),
                    _ => quote!(e2m1),
                };
                quote!(run::compiled::lockstep::#f(#w, #l))
            }
        })
    }

    /// `e` with each lane shuffle computed first, into `pre`.
    pub(crate) fn hoist(&mut self, e: &Expr, pre: &mut TokenStream) -> Result<Expr, String> {
        let sub = |s: &mut Self, e: &Expr, pre: &mut TokenStream| s.hoist(e, pre).map(Box::new);
        Ok(match e {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Var(_) => e.clone(),
            Expr::Neg(a) => Expr::Neg(sub(self, a, pre)?),
            Expr::Not(a) => Expr::Not(sub(self, a, pre)?),
            Expr::Bin(op, a, b) => Expr::Bin(*op, sub(self, a, pre)?, sub(self, b, pre)?),
            Expr::Cast(ty, a) => Expr::Cast(*ty, sub(self, a, pre)?),
            Expr::Index(base, at) => Expr::Index(base.clone(), sub(self, at, pre)?),
            Expr::Field(base, h) => Expr::Field(sub(self, base, pre)?, *h),
            Expr::Select(c, a, b) => {
                Expr::Select(sub(self, c, pre)?, sub(self, a, pre)?, sub(self, b, pre)?)
            }
            Expr::Fn(name, ty, args) => Expr::Fn(name.clone(), *ty, self.hoist_all(args, pre)?),
            Expr::Call(Intrinsic::ShflXor, args) => {
                let args = self.hoist_all(args, pre)?;
                let (v, m) = (
                    self.expr_as(&args[0], Ty::F32)?,
                    self.expr_as(&args[1], Ty::U32)?,
                );
                let (name, hm, mask) = (self.next("h").to_string(), self.next("hm"), self.mask());
                let h = var(&name);
                pre.extend(quote! {
                    let mut #h = [0f32; THREADS];
                    let mut #hm = [0u32; THREADS];
                    for t in 0..n {
                        if #mask[t] {
                            #h[t] = #v;
                            #hm[t] = #m;
                        }
                    }
                    let #h = run::compiled::lockstep::shfl_xor(&#h, &#hm, n, lanes);
                });
                self.bind(&name, Kind::Lanes(Ty::F32));
                Expr::Var(name)
            }
            Expr::Call(i, args) => Expr::Call(*i, self.hoist_all(args, pre)?),
        })
    }

    fn hoist_all(&mut self, args: &[Expr], pre: &mut TokenStream) -> Result<Vec<Expr>, String> {
        args.iter().map(|a| self.hoist(a, pre)).collect()
    }
}
