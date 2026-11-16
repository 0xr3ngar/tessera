//! The graph's JSON form, its version and its hash.

use crate::node::Graph;
use serde::Deserialize;

/// Bumped whenever the JSON form changes meaning.
pub const VERSION: u32 = 4;

#[derive(Debug, thiserror::Error)]
pub enum FormatError {
    #[error("graph JSON: {0}")]
    Json(#[source] serde_json::Error),
    #[error("graph format version {found}, this build reads {VERSION}")]
    Version { found: u32 },
}

#[derive(Deserialize)]
struct Header {
    version: u32,
}

impl Graph {
    /// Field order is fixed, so equal graphs give equal text.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("graphs always serialize")
    }

    /// Checks the version before reading anything else.
    pub fn from_json(text: &str) -> Result<Graph, FormatError> {
        let header: Header = serde_json::from_str(text).map_err(FormatError::Json)?;
        if header.version != VERSION {
            return Err(FormatError::Version {
                found: header.version,
            });
        }
        serde_json::from_str(text).map_err(FormatError::Json)
    }

    /// FNV-1a over the compact JSON form.
    pub fn hash(&self) -> u64 {
        let text = serde_json::to_vec(self).expect("graphs always serialize");
        text.iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
    }
}
