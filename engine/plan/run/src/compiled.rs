//! Compiled bodies, found by the hash of their text.

use codegen::form::gather;
pub use codegen::form::scatter;
pub use codegen::lockstep::Kernel;
// What compiled bodies call, reached through this crate.
pub use codegen::{Precision, exec, kept, lockstep};

/// One compiled body.
pub struct Compiled {
    pub hash: u64,
    pub run: Kernel,
}

gather! {
    #[gather]
    pub static COMPILED: scattered_collect::ScatteredSlice<Compiled>;
}

/// Registers a compiled body under its source hash.
#[macro_export]
macro_rules! compiled {
    ($hash:literal, $f:path) => {
        const _: () = {
            $crate::compiled::scatter! {
                #[scatter($crate::compiled::COMPILED)]
                static COMPILED: $crate::compiled::Compiled = $crate::compiled::Compiled {
                    hash: $hash,
                    run: $f,
                };
            }
        };
    };
}

/// The compiled body whose source hashes to `hash`.
pub fn find(hash: u64) -> Option<Kernel> {
    COMPILED.iter().find(|c| c.hash == hash).map(|c| c.run)
}
