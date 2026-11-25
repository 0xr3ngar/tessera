//! Operator scripts: call an exported function with data and arrays.

#[cfg(feature = "quickjs")]
mod quickjs;
#[cfg(all(test, feature = "quickjs"))]
mod quickjs_test;

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::time::Duration;

/// Why a script failed. The process goes on.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("this build has no script support: rebuild with the `script` feature")]
    Unsupported,
    #[error("{0}: {1}")]
    Read(PathBuf, std::io::Error),
    #[error("{0}: {1}")]
    Thrown(String, String),
    #[error("{0}: ran past the {1:?} limit")]
    Time(String, Duration),
    #[error("{0}: used more than {1} bytes")]
    Memory(String, usize),
    #[error("{0}: exports no function `{1}`")]
    Missing(String, String),
    #[error("{0}: `{1}` returned a value that is not JSON")]
    Return(String, String),
}

/// The time and memory limits of one call.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub time: Duration,
    /// Bytes for the whole runtime, copied arrays included.
    pub memory: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            time: Duration::from_secs(1),
            memory: 256 << 20,
        }
    }
}

/// One named float array that a call may change.
pub type Array<'a> = (&'a str, &'a mut [f32]);

/// The calls a script engine must answer.
trait Engine {
    fn has(&self, function: &str) -> bool;
    /// `context` is JSON text.
    fn call(&mut self, function: &str, context: &str, arrays: &mut [Array])
    -> Result<Value, Error>;
}

/// A loaded script. It stays on its loading thread.
pub struct Script {
    engine: Box<dyn Engine>,
}

impl Script {
    /// Reads and loads the script at `path`.
    pub fn open(path: &Path, limits: Limits) -> Result<Self, Error> {
        if !cfg!(feature = "quickjs") {
            return Err(Error::Unsupported);
        }
        let source = std::fs::read_to_string(path).map_err(|e| Error::Read(path.into(), e))?;
        let name = path.file_name().unwrap_or_default().to_string_lossy();
        Self::load(&name, &source, limits)
    }

    /// Loads `source` as an ES module called `name`.
    pub fn load(name: &str, source: &str, limits: Limits) -> Result<Self, Error> {
        #[cfg(feature = "quickjs")]
        {
            let engine = quickjs::Quick::load(name, source, limits)?;
            Ok(Self {
                engine: Box::new(engine),
            })
        }
        #[cfg(not(feature = "quickjs"))]
        {
            let _ = (name, source, limits);
            Err(Error::Unsupported)
        }
    }

    /// Whether the script exports `function`.
    pub fn has(&self, function: &str) -> bool {
        self.engine.has(function)
    }

    /// Calls `function`. Edits to the arrays come back.
    pub fn call(
        &mut self,
        function: &str,
        context: &Value,
        arrays: &mut [Array],
    ) -> Result<Value, Error> {
        self.engine.call(function, &context.to_string(), arrays)
    }
}
