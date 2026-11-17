//! Guides: per-answer rules on which tokens come next.

pub type Fault = Box<dyn std::error::Error + Send + Sync>;

/// One answer's rules on its next tokens.
pub trait Guide: Send {
    /// Clears forbidden bits of an all-ones `out`. False: none.
    fn mask(&mut self, out: &mut [u32]) -> Result<bool, Fault>;

    /// Answer tokens to add with no pick, already accepted.
    fn forced(&mut self) -> Vec<u32> {
        Vec::new()
    }

    /// Moves past a picked token.
    fn accept(&mut self, id: u32) -> Result<(), Fault>;

    /// Whether the answer ends here.
    fn done(&self) -> bool {
        false
    }
}

/// Fixed tokens first, then any.
pub struct Prefix(pub Vec<u32>);

impl Guide for Prefix {
    fn mask(&mut self, _: &mut [u32]) -> Result<bool, Fault> {
        Ok(false)
    }

    fn forced(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.0)
    }

    fn accept(&mut self, _: u32) -> Result<(), Fault> {
        Ok(())
    }
}
