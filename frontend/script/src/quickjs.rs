//! The QuickJS engine, through rquickjs.

use crate::{Array, Engine, Error, Limits};
use rquickjs::function::Rest;
use rquickjs::{
    CatchResultExt, Coerced, Context, Ctx, Exception, Function, Module, Object, Persistent,
    Runtime, TypedArray, Value,
};
use std::cell::Cell;
use std::rc::Rc;
use std::time::Instant;

/// Runs calls under the limits, and names what stopped one.
struct Guard {
    name: String,
    limits: Limits,
    /// When the running call must stop.
    deadline: Rc<Cell<Option<Instant>>>,
    /// Whether the interrupt handler stopped the call.
    stopped: Rc<Cell<bool>>,
}

impl Guard {
    fn run<T>(
        &self,
        context: &Context,
        f: impl for<'js> FnOnce(&Ctx<'js>) -> rquickjs::Result<T>,
    ) -> Result<T, Error> {
        self.deadline.set(Some(Instant::now() + self.limits.time));
        self.stopped.set(false);
        let out = context.with(|ctx| f(&ctx).catch(&ctx).map_err(|e| e.to_string()));
        self.deadline.set(None);
        let name = self.name.clone();
        out.map_err(|message| match () {
            _ if self.stopped.get() => Error::Time(name, self.limits.time),
            _ if message.contains("out of memory") => Error::Memory(name, self.limits.memory),
            _ => Error::Thrown(name, message),
        })
    }
}

/// A runtime with one module loaded. Not `Send`.
pub(crate) struct Quick {
    context: Context,
    exports: Persistent<Object<'static>>,
    guard: Guard,
    /// Keeps the runtime alive for the context.
    _runtime: Runtime,
}

impl Quick {
    pub fn load(name: &str, source: &str, limits: Limits) -> Result<Self, Error> {
        let fail = |e: rquickjs::Error| Error::Thrown(name.into(), e.to_string());
        let runtime = Runtime::new().map_err(fail)?;
        runtime.set_memory_limit(limits.memory);
        let guard = Guard {
            name: name.into(),
            limits,
            deadline: Rc::new(Cell::new(None)),
            stopped: Rc::new(Cell::new(false)),
        };
        let (until, stop) = (guard.deadline.clone(), guard.stopped.clone());
        runtime.set_interrupt_handler(Some(Box::new(move || {
            let late = until.get().is_some_and(|d| Instant::now() > d);
            stop.set(stop.get() | late);
            late
        })));
        let context = Context::full(&runtime).map_err(fail)?;
        let exports = guard.run(&context, |ctx| {
            ctx.globals().set("log", log(ctx, name)?)?;
            let (module, done) = Module::declare(ctx.clone(), name, source)?.eval()?;
            done.finish::<()>()?;
            Ok(Persistent::save(ctx, module.namespace()?))
        })?;
        Ok(Self {
            context,
            exports,
            guard,
            _runtime: runtime,
        })
    }
}

/// `log(...)`: the words, to stderr after the script name.
fn log<'js>(ctx: &Ctx<'js>, name: &str) -> rquickjs::Result<Function<'js>> {
    let name = name.to_string();
    Function::new(ctx.clone(), move |words: Rest<Coerced<String>>| {
        let words: Vec<String> = words.0.into_iter().map(|w| w.0).collect();
        eprintln!("{name}: {}", words.join(" "));
    })
}

/// Copies edits back while the engine is idle.
fn take_back(
    ctx: &Ctx,
    name: &str,
    view: &TypedArray<f32>,
    to: &mut [f32],
) -> rquickjs::Result<()> {
    // SAFETY: the slice dies before the engine runs again.
    let bytes = unsafe { view.as_bytes() }.filter(|b| b.len() == to.len() * 4);
    let message = format!("the script detached `{name}`");
    let bytes = bytes.ok_or_else(|| Exception::throw_type(ctx, &message))?;
    for (to, from) in to.iter_mut().zip(bytes.as_chunks::<4>().0) {
        *to = f32::from_ne_bytes(*from);
    }
    Ok(())
}

impl Engine for Quick {
    fn has(&self, function: &str) -> bool {
        let exports = self.exports.clone();
        self.context.with(|ctx| {
            let exports = exports.restore(&ctx).ok();
            exports.is_some_and(|e| e.get::<_, Function>(function).is_ok())
        })
    }

    fn call(
        &mut self,
        function: &str,
        context: &str,
        arrays: &mut [Array],
    ) -> Result<serde_json::Value, Error> {
        let name = &self.guard.name;
        if !self.has(function) {
            return Err(Error::Missing(name.clone(), function.into()));
        }
        let exports = self.exports.clone();
        let out = self.guard.run(&self.context, |ctx| {
            let f: Function = exports.restore(ctx)?.get(function)?;
            let arg: Object = ctx.json_parse(context)?.get()?;
            let mut views = Vec::new();
            for (name, data) in arrays.iter() {
                let view = TypedArray::<f32>::new_copy(ctx.clone(), &**data)?;
                arg.set(*name, view.clone())?;
                views.push(view);
            }
            let back: Value = f.call((arg,))?;
            for ((name, data), view) in arrays.iter_mut().zip(&views) {
                take_back(ctx, name, view, data)?;
            }
            let text = ctx.json_stringify(back)?;
            text.map_or(Ok("null".into()), |t| t.to_string())
        })?;
        serde_json::from_str(&out).map_err(|_| Error::Return(name.clone(), function.into()))
    }
}
