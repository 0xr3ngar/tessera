//! Runs kernel bodies on the host, each group in lockstep.

pub mod exec;
mod intrinsic {
    include!("../../codegen/src/intrinsic.rs");
}
pub mod lockstep;

#[cfg(test)]
mod exec_test;
#[cfg(test)]
mod lockstep_test;
