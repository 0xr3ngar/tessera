//! Expressions, each lane's value inside a loop over `t`.

use crate::lower::{Kind, Lowerer, Res};
use codegen::ir::{BinOp, Expr, Intrinsic, Ty};
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, MemFlagsData, Type, Value, types};

/// One lane's value: a scalar, or four floats.
#[derive(Clone, Copy)]
pub(crate) enum Val {
    S(Value),
    V([Value; 4]),
}

impl Val {
    fn scalar(self) -> Res<Value> {
        match self {
            Val::S(v) => Ok(v),
            Val::V(_) => Err("a four-wide value where a scalar was due".into()),
        }
    }

    fn wide(self) -> Res<[Value; 4]> {
        match self {
            Val::V(v) => Ok(v),
            Val::S(_) => Err("a scalar where a four-wide value was due".into()),
        }
    }
}

/// The Cranelift type holding `ty` in a register.
pub(crate) fn clif(ty: Ty) -> Type {
    match ty {
        Ty::F32 => types::F32,
        Ty::U32 => types::I32,
        Ty::U64 => types::I64,
        Ty::Bool => types::I8,
        Ty::F32x4 | Ty::Bytes => panic!("{ty:?} is not one register"),
    }
}

/// Bytes one lane's value of `ty` takes.
pub(crate) fn elem_size(ty: Ty) -> usize {
    match ty {
        Ty::F32 | Ty::U32 => 4,
        Ty::U64 => 8,
        Ty::Bool => 1,
        Ty::F32x4 => 16,
        Ty::Bytes => panic!("a byte pointer has no lane size"),
    }
}

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

fn element(form: &dyn codegen::Form) -> Ty {
    match form.dtype().is_float() {
        true => Ty::F32,
        false => Ty::U32,
    }
}

/// The reader for one element of a form.
fn reader(form: &dyn codegen::Form) -> Res<&'static str> {
    Ok(match form.name() {
        "f32" => "rd_f32",
        "bf16" => "rd_bf16",
        "u32" => "rd_u32",
        "u8" => "rd_u8",
        "q8_0" => "rd_q8_0",
        "q4_0" => "rd_q4_0",
        "nvfp4" => "rd_nvfp4",
        other => return Err(format!("no reader for the {other} form")),
    })
}

impl Lowerer<'_, '_> {
    /// The type an expression has, from the names it uses.
    pub(crate) fn ty(&self, e: &Expr) -> Res<Ty> {
        Ok(match e {
            Expr::Int(_) => Ty::U32,
            Expr::Float(_) => Ty::F32,
            Expr::Bool(_) => Ty::Bool,
            Expr::Var(name) => match self.kind(name)? {
                Kind::Lanes(ty, _) | Kind::One(ty, _) => ty,
                Kind::Buffer { .. } => Ty::Bytes,
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
                    Kind::Array(ty, ..) | Kind::Shared(ty, ..) => ty,
                    Kind::Buffer { form, .. } => element(form),
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

    /// `v` of type `from`, as `to`.
    pub(crate) fn cast(&mut self, v: Val, from: Ty, to: Ty) -> Res<Val> {
        if from == to {
            return Ok(v);
        }
        Ok(Val::S(match (from, to) {
            (Ty::F32, Ty::F32x4) => {
                let v = v.scalar()?;
                return Ok(Val::V([v; 4]));
            }
            (Ty::F32, Ty::Bool) => {
                let zero = self.b.ins().f32const(0.0);
                self.b.ins().fcmp(FloatCC::NotEqual, v.scalar()?, zero)
            }
            (Ty::U32 | Ty::U64, Ty::Bool) => {
                self.b.ins().icmp_imm_u(IntCC::NotEqual, v.scalar()?, 0)
            }
            (Ty::Bool, Ty::U32) => self.b.ins().uextend(types::I32, v.scalar()?),
            (Ty::Bool, Ty::U64) => self.b.ins().uextend(types::I64, v.scalar()?),
            (Ty::Bool, Ty::F32) => {
                let wide = self.b.ins().uextend(types::I32, v.scalar()?);
                self.b.ins().fcvt_from_uint(types::F32, wide)
            }
            (Ty::U32, Ty::U64) => self.b.ins().uextend(types::I64, v.scalar()?),
            (Ty::U64, Ty::U32) => self.b.ins().ireduce(types::I32, v.scalar()?),
            (Ty::U32 | Ty::U64, Ty::F32) => self.b.ins().fcvt_from_uint(types::F32, v.scalar()?),
            (Ty::F32, Ty::U32) => self.b.ins().fcvt_to_uint_sat(types::I32, v.scalar()?),
            (Ty::F32, Ty::U64) => self.b.ins().fcvt_to_uint_sat(types::I64, v.scalar()?),
            _ => return Err(format!("no cast from {from:?} to {to:?}")),
        }))
    }

    /// `e` as `to`, for lane `t`.
    pub(crate) fn expr_as(&mut self, e: &Expr, to: Ty, t: Value) -> Res<Val> {
        let from = self.ty(e)?;
        let v = self.expr(e, t)?;
        self.cast(v, from, to)
    }

    /// An index as a pointer-wide integer.
    pub(crate) fn index(&mut self, at: &Expr, t: Value) -> Res<Value> {
        self.expr_as(at, Ty::U64, t)?.scalar()
    }

    /// Element `at` of the buffer `name`, read through its form.
    fn read(&mut self, name: &str, at: &Expr, t: Value) -> Res<Value> {
        let Kind::Buffer {
            form,
            ptr,
            len,
            skip,
        } = self.kind(name)?
        else {
            return Err(format!("`{name}` is not a buffer"));
        };
        let idx = self.index(at, t)?;
        let at = self.b.ins().iadd(skip, idx);
        let rd = reader(form)?;
        Ok(self.call(rd, &[ptr, len, at]).expect("a value"))
    }

    pub(crate) fn expr(&mut self, e: &Expr, t: Value) -> Res<Val> {
        match e {
            Expr::Int(n) => Ok(Val::S(self.iconst(types::I32, i64::from(*n as u32)))),
            Expr::Float(x) => Ok(Val::S(self.b.ins().f32const(*x as f32))),
            Expr::Bool(b) => Ok(Val::S(self.iconst(types::I8, i64::from(*b)))),
            Expr::Var(name) => match self.kind(name)? {
                Kind::Lanes(ty, slot) => Ok(self.load_lane(slot, ty, t)),
                Kind::One(_, v) => Ok(Val::S(v)),
                _ => Err(format!("`{name}` is not a value")),
            },
            Expr::Neg(a) => {
                let (ty, v) = (self.ty(a)?, self.expr(a, t)?.scalar()?);
                Ok(Val::S(match ty {
                    Ty::F32 => self.b.ins().fneg(v),
                    Ty::U32 | Ty::U64 => self.b.ins().ineg(v),
                    other => return Err(format!("negating a {other:?}")),
                }))
            }
            Expr::Not(a) => {
                let (ty, v) = (self.ty(a)?, self.expr(a, t)?.scalar()?);
                Ok(Val::S(match ty {
                    Ty::Bool => self.b.ins().bxor_imm_u(v, 1),
                    _ => self.b.ins().bnot(v),
                }))
            }
            Expr::Bin(op, a, b) => self.binary(*op, a, b, t),
            Expr::Cast(ty, a) => self.expr_as(a, *ty, t),
            Expr::Index(base, at) => {
                let Expr::Var(name) = &**base else {
                    return Err("an index names what it reads".into());
                };
                match self.kind(name)? {
                    Kind::Buffer { .. } => Ok(Val::S(self.read(name, at, t)?)),
                    Kind::Shared(ty, slot, _) => {
                        let idx = self.index(at, t)?;
                        let addr = self.at(slot, idx, elem_size(ty));
                        Ok(self.load_at(addr, ty))
                    }
                    Kind::Array(ty, slot, len) => {
                        let addr = self.array_at(slot, ty, len, t, at)?;
                        Ok(self.load_at(addr, ty))
                    }
                    _ => Err(format!("`{name}` is not indexed")),
                }
            }
            Expr::Field(base, h) => {
                let v = self.expr(base, t)?.wide()?;
                let i = ['x', 'y', 'z', 'w']
                    .iter()
                    .position(|c| c == h)
                    .unwrap_or(3);
                Ok(Val::S(v[i]))
            }
            Expr::Select(c, a, b) if matches!(**c, Expr::Bool(_)) => {
                let taken = if **c == Expr::Bool(true) { a } else { b };
                self.expr(taken, t)
            }
            Expr::Select(c, a, b) => self.select(c, a, b, t),
            Expr::Call(i, args) => self.call_intrinsic(*i, args, t),
            Expr::Fn(name, ..) => Err(format!("`{name}` is a device helper")),
        }
    }

    /// `if c { a } else { b }`, each side only where taken.
    fn select(&mut self, c: &Expr, a: &Expr, b: &Expr, t: Value) -> Res<Val> {
        let ty = self.ty(a)?;
        let c = self.expr_as(c, Ty::Bool, t)?.scalar()?;
        let (yes, no, merge) = (
            self.b.create_block(),
            self.b.create_block(),
            self.b.create_block(),
        );
        let width = if ty == Ty::F32x4 { 4 } else { 1 };
        for _ in 0..width {
            let reg = if ty == Ty::F32x4 {
                types::F32
            } else {
                clif(ty)
            };
            self.b.append_block_param(merge, reg);
        }
        self.b.ins().brif(c, yes, &[], no, &[]);
        for (block, e) in [(yes, a), (no, b)] {
            self.b.switch_to_block(block);
            let v = self.expr_as(e, ty, t)?;
            let args: Vec<_> = match v {
                Val::S(v) => vec![v.into()],
                Val::V(v) => v.into_iter().map(Into::into).collect(),
            };
            self.b.ins().jump(merge, &args);
        }
        self.b.switch_to_block(merge);
        let out = self.b.block_params(merge).to_vec();
        Ok(match ty {
            Ty::F32x4 => Val::V([out[0], out[1], out[2], out[3]]),
            _ => Val::S(out[0]),
        })
    }

    fn binary(&mut self, op: BinOp, a: &Expr, b: &Expr, t: Value) -> Res<Val> {
        let (ta, tb) = (self.ty(a)?, self.ty(b)?);
        if matches!(op, BinOp::And | BinOp::Or) {
            let (a, b) = (
                self.expr_as(a, Ty::Bool, t)?.scalar()?,
                self.expr_as(b, Ty::Bool, t)?.scalar()?,
            );
            return Ok(Val::S(match op {
                BinOp::And => self.b.ins().band(a, b),
                _ => self.b.ins().bor(a, b),
            }));
        }
        let w = wider(ta, tb);
        let (a, b) = (self.expr_as(a, w, t)?, self.expr_as(b, w, t)?);
        if w == Ty::F32x4 {
            let (a, b) = (a.wide()?, b.wide()?);
            let mut out = [a[0]; 4];
            for i in 0..4 {
                out[i] = self.arith(op, Ty::F32, a[i], b[i])?;
            }
            return Ok(Val::V(out));
        }
        let (a, b) = (a.scalar()?, b.scalar()?);
        if op.compares() {
            return Ok(Val::S(self.compare(op, w, a, b)));
        }
        Ok(Val::S(self.arith(op, w, a, b)?))
    }

    fn compare(&mut self, op: BinOp, ty: Ty, a: Value, b: Value) -> Value {
        if ty == Ty::F32 {
            let cc = match op {
                BinOp::Lt => FloatCC::LessThan,
                BinOp::Le => FloatCC::LessThanOrEqual,
                BinOp::Gt => FloatCC::GreaterThan,
                BinOp::Ge => FloatCC::GreaterThanOrEqual,
                BinOp::Eq => FloatCC::Equal,
                _ => FloatCC::NotEqual,
            };
            return self.b.ins().fcmp(cc, a, b);
        }
        let cc = match op {
            BinOp::Lt => IntCC::UnsignedLessThan,
            BinOp::Le => IntCC::UnsignedLessThanOrEqual,
            BinOp::Gt => IntCC::UnsignedGreaterThan,
            BinOp::Ge => IntCC::UnsignedGreaterThanOrEqual,
            BinOp::Eq => IntCC::Equal,
            _ => IntCC::NotEqual,
        };
        self.b.ins().icmp(cc, a, b)
    }

    fn arith(&mut self, op: BinOp, ty: Ty, a: Value, b: Value) -> Res<Value> {
        let ins = self.b.ins();
        Ok(match (op, ty == Ty::F32) {
            (BinOp::Add, true) => ins.fadd(a, b),
            (BinOp::Sub, true) => ins.fsub(a, b),
            (BinOp::Mul, true) => ins.fmul(a, b),
            (BinOp::Div, true) => ins.fdiv(a, b),
            (BinOp::Add, false) => ins.iadd(a, b),
            (BinOp::Sub, false) => ins.isub(a, b),
            (BinOp::Mul, false) => ins.imul(a, b),
            (BinOp::Div, false) => ins.udiv(a, b),
            (BinOp::Rem, false) => ins.urem(a, b),
            (BinOp::Shl, false) => ins.ishl(a, b),
            (BinOp::Shr, false) => ins.ushr(a, b),
            (BinOp::BitAnd, false) => ins.band(a, b),
            (BinOp::BitOr, false) => ins.bor(a, b),
            (BinOp::BitXor, false) => ins.bxor(a, b),
            (other, _) => return Err(format!("{other:?} on {ty:?}")),
        })
    }

    /// Float arguments, each as f32.
    fn floats(&mut self, args: &[Expr], t: Value) -> Res<Vec<Value>> {
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            out.push(self.expr_as(a, Ty::F32, t)?.scalar()?);
        }
        Ok(out)
    }

    fn u32s(&mut self, args: &[Expr], t: Value) -> Res<Vec<Value>> {
        let mut out = Vec::with_capacity(args.len());
        for a in args {
            out.push(self.expr_as(a, Ty::U32, t)?.scalar()?);
        }
        Ok(out)
    }

    fn call_intrinsic(&mut self, i: Intrinsic, args: &[Expr], t: Value) -> Res<Val> {
        let math = |s: &mut Self, name: &'static str| -> Res<Val> {
            let a = s.floats(args, t)?;
            Ok(Val::S(s.call(name, &a).expect("a float")))
        };
        Ok(match i {
            Intrinsic::Gid => {
                let first = self.b.use_var(self.first);
                let g = self.b.ins().iadd(first, t);
                Val::S(self.b.ins().ireduce(types::I32, g))
            }
            Intrinsic::Lid => Val::S(self.b.ins().ireduce(types::I32, t)),
            Intrinsic::Lane => {
                let lane = self.b.ins().urem(t, self.lanes);
                Val::S(self.b.ins().ireduce(types::I32, lane))
            }
            Intrinsic::ShflXor | Intrinsic::Sum32 => {
                return Err("a lane shuffle outside a statement".into());
            }
            Intrinsic::Exp => math(self, "exp")?,
            Intrinsic::Tanh => math(self, "tanh")?,
            Intrinsic::Cos => math(self, "cos")?,
            Intrinsic::Sin => math(self, "sin")?,
            Intrinsic::Max => math(self, "max")?,
            Intrinsic::Pow => math(self, "pow")?,
            Intrinsic::Clamp => math(self, "clamp")?,
            Intrinsic::Rsqrt => {
                let x = self.floats(args, t)?[0];
                let root = self.b.ins().sqrt(x);
                let one = self.b.ins().f32const(1.0);
                Val::S(self.b.ins().fdiv(one, root))
            }
            Intrinsic::FromBits => {
                let x = self.u32s(args, t)?[0];
                Val::S(self.b.ins().bitcast(types::F32, MemFlagsData::new(), x))
            }
            Intrinsic::ToBits => {
                let x = self.floats(args, t)?[0];
                Val::S(self.b.ins().bitcast(types::I32, MemFlagsData::new(), x))
            }
            Intrinsic::Load4 => {
                let Expr::Var(name) = &args[0] else {
                    return Err("a load names its buffer".into());
                };
                let mut out = [t; 4];
                for (k, slot) in out.iter_mut().enumerate() {
                    let at = Expr::bin(BinOp::Add, args[1].clone(), Expr::Int(k as i64));
                    *slot = self.read(name, &at, t)?;
                }
                Val::V(out)
            }
            Intrinsic::Store4 => {
                let Expr::Var(name) = &args[0] else {
                    return Err("a store names its buffer".into());
                };
                let Kind::Buffer { ptr, len, skip, .. } = self.kind(name)? else {
                    return Err(format!("`{name}` is not a buffer"));
                };
                let idx = self.index(&args[1], t)?;
                let at = self.b.ins().iadd(skip, idx);
                let v = self.expr_as(&args[2], Ty::F32x4, t)?.wide()?;
                for (k, part) in v.into_iter().enumerate() {
                    let k = self.iconst(self.ptr, k as i64);
                    let at = self.b.ins().iadd(at, k);
                    self.call("st_f32", &[ptr, len, at, part]);
                }
                Val::S(self.iconst(types::I8, 0))
            }
            Intrinsic::Zero4 => {
                let zero = self.b.ins().f32const(0.0);
                Val::V([zero; 4])
            }
            Intrinsic::Make4 => {
                let a = self.floats(args, t)?;
                Val::V([a[0], a[1], a[2], a[3]])
            }
            Intrinsic::KeptRow => {
                let layout = self.layout.ok_or("kept rows without a layout")?;
                let (s, p) = {
                    let v = self.u32s(args, t)?;
                    (v[0], v[1])
                };
                let mut values = Vec::new();
                for name in layout.params() {
                    let Kind::One(_, v) = self.kind(name)? else {
                        return Err(format!("layout scalar `{name}` is not a scalar"));
                    };
                    values.push(v);
                }
                let ins = self.b.ins();
                Val::S(match layout.name() {
                    "dense" => {
                        let row = ins.imul(s, values[0]);
                        self.b.ins().iadd(row, p)
                    }
                    "interleaved" => {
                        let row = ins.imul(p, values[0]);
                        self.b.ins().iadd(row, s)
                    }
                    other => return Err(format!("no `{other}` layout on the host")),
                })
            }
            Intrinsic::Word | Intrinsic::Half => {
                let Expr::Var(name) = &args[0] else {
                    return Err("a word read names its buffer".into());
                };
                let Kind::Buffer { ptr, len, skip, .. } = self.kind(name)? else {
                    return Err(format!("`{name}` is not a buffer"));
                };
                let idx = self.index(&args[1], t)?;
                let at = self.b.ins().iadd(skip, idx);
                let f = if i == Intrinsic::Word { "word" } else { "half" };
                Val::S(self.call(f, &[ptr, len, at]).expect("a word"))
            }
            Intrinsic::Bits | Intrinsic::Signed => {
                let v = self.u32s(args, t)?;
                let (w, l, n) = (v[0], v[1], v[2]);
                // The field, then its sign stretched over the word.
                let ins = self.b.ins();
                let shifted = ins.ushr(w, l);
                let wide = self.b.ins().uextend(types::I64, n);
                let one = self.iconst(types::I64, 1);
                let bit = self.b.ins().ishl(one, wide);
                let ones = self.b.ins().isub(bit, one);
                let ones = self.b.ins().ireduce(types::I32, ones);
                let field = self.b.ins().band(shifted, ones);
                match i {
                    Intrinsic::Bits => Val::S(field),
                    _ => {
                        let width = self.iconst(types::I32, 32);
                        let rest = self.b.ins().isub(width, n);
                        let up = self.b.ins().ishl(field, rest);
                        let down = self.b.ins().sshr(up, rest);
                        Val::S(self.b.ins().fcvt_from_sint(types::F32, down))
                    }
                }
            }
            Intrinsic::F16 | Intrinsic::E4M3 | Intrinsic::E2M1 => {
                let v = self.u32s(args, t)?;
                let f = match i {
                    Intrinsic::F16 => "f16",
                    Intrinsic::E4M3 => "e4m3",
                    _ => "e2m1",
                };
                Val::S(self.call(f, &[v[0], v[1]]).expect("a float"))
            }
        })
    }

    /// `e` with each lane shuffle computed first, into lane arrays.
    pub(crate) fn hoist(&mut self, e: &Expr) -> Res<Expr> {
        let sub = |s: &mut Self, e: &Expr| s.hoist(e).map(Box::new);
        Ok(match e {
            Expr::Int(_) | Expr::Float(_) | Expr::Bool(_) | Expr::Var(_) => e.clone(),
            Expr::Neg(a) => Expr::Neg(sub(self, a)?),
            Expr::Not(a) => Expr::Not(sub(self, a)?),
            Expr::Bin(op, a, b) => Expr::Bin(*op, sub(self, a)?, sub(self, b)?),
            Expr::Cast(ty, a) => Expr::Cast(*ty, sub(self, a)?),
            Expr::Index(base, at) => Expr::Index(base.clone(), sub(self, at)?),
            Expr::Field(base, h) => Expr::Field(sub(self, base)?, *h),
            Expr::Select(c, a, b) => Expr::Select(sub(self, c)?, sub(self, a)?, sub(self, b)?),
            Expr::Fn(name, ty, args) => Expr::Fn(name.clone(), *ty, self.hoist_all(args)?),
            Expr::Call(Intrinsic::ShflXor, args) => {
                let args = self.hoist_all(args)?;
                let (v, m) = (self.lane_slot(Ty::F32), self.lane_slot(Ty::U32));
                let mask = self.mask();
                self.lanes(Some(mask), |s, t| {
                    let value = s.expr_as(&args[0], Ty::F32, t)?;
                    s.store_lane(v, Ty::F32, t, value);
                    let by = s.expr_as(&args[1], Ty::U32, t)?;
                    s.store_lane(m, Ty::U32, t, by);
                    Ok(())
                })?;
                let out = self.lane_slot(Ty::F32);
                let (v, m, o) = (self.slot_addr(v), self.slot_addr(m), self.slot_addr(out));
                self.call("shfl_xor", &[v, m, o, self.n, self.lanes]);
                let name = format!("shfl__{}", out.as_u32());
                self.bind(&name, Kind::Lanes(Ty::F32, out));
                Expr::Var(name)
            }
            Expr::Call(i, args) => Expr::Call(*i, self.hoist_all(args)?),
        })
    }

    fn hoist_all(&mut self, args: &[Expr]) -> Res<Vec<Expr>> {
        args.iter().map(|a| self.hoist(a)).collect()
    }
}
