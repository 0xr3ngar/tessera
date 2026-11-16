//! Embeddings keyed by window text, kept between runs and checkouts.

use crate::fetch::REVISION;
use anyhow::{Result, ensure};
use std::collections::HashMap;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Changes with the file layout. Entries carry their last use.
const MAGIC: &[u8; 8] = b"simck002";
/// The layout before last uses. Its entries load as new.
const OLD_MAGIC: &[u8; 8] = b"simck001";
/// Saving drops entries unused for this long.
pub const KEEP_SECS: u64 = 7 * 24 * 60 * 60;
/// Keep in step with eg2cpu's `rev` in Cargo.toml.
const EG2CPU: &str = "a80156abbf8ec2e5a9b85ed0e5ff98454c1aaa09";

/// FNV-1a over the model revisions and the window text.
pub fn key(text: &str) -> u64 {
    let parts = [EG2CPU.as_bytes(), REVISION.as_bytes(), text.as_bytes()];
    parts
        .iter()
        .flat_map(|p| p.iter())
        .fold(0xcbf2_9ce4_8422_2325, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
}

pub fn now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Each key's last use in Unix seconds, and its embedding.
#[derive(Default)]
pub struct Cache {
    dim: usize,
    map: HashMap<u64, (u64, Vec<f32>)>,
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|f| f32::from_le_bytes(*f))
        .collect()
}

impl Cache {
    /// Empty when the file is missing or its layout unknown.
    pub fn load(path: &Path) -> Result<Cache> {
        let Ok(bytes) = std::fs::read(path) else {
            return Ok(Cache::default());
        };
        let old = bytes.get(..8) == Some(OLD_MAGIC);
        if bytes.len() < 12 || (&bytes[..8] != MAGIC && !old) {
            return Ok(Cache::default());
        }
        let dim = u32::from_le_bytes(bytes[8..12].try_into()?) as usize;
        let head = if old { 8 } else { 16 };
        let entry = head + dim * 4;
        let body = &bytes[12..];
        ensure!(
            dim > 0 && body.len() % entry == 0,
            "{}: truncated",
            path.display()
        );
        let loaded = now();
        let mut map = HashMap::new();
        for e in body.chunks_exact(entry) {
            let k = u64::from_le_bytes(e[..8].try_into()?);
            let used = if old {
                loaded
            } else {
                u64::from_le_bytes(e[8..16].try_into()?)
            };
            map.insert(k, (used, floats(&e[head..])));
        }
        Ok(Cache { dim, map })
    }

    pub fn get(&self, k: u64) -> Option<&Vec<f32>> {
        self.map.get(&k).map(|(_, v)| v)
    }

    pub fn insert(&mut self, k: u64, v: Vec<f32>) {
        self.dim = v.len();
        self.map.insert(k, (now(), v));
    }

    /// Marks `used` at `at`, then saves entries used this week.
    pub fn save_at(&mut self, path: &Path, used: &[u64], at: u64) -> Result<()> {
        for k in used {
            if let Some(e) = self.map.get_mut(k) {
                e.0 = at;
            }
        }
        let mut keys: Vec<u64> = self
            .map
            .iter()
            .filter(|(_, e)| at.saturating_sub(e.0) <= KEEP_SECS)
            .map(|(k, _)| *k)
            .collect();
        keys.sort_unstable();
        let mut out = MAGIC.to_vec();
        out.extend((self.dim as u32).to_le_bytes());
        for k in keys {
            let (when, v) = &self.map[&k];
            out.extend(k.to_le_bytes());
            out.extend(when.to_le_bytes());
            out.extend(v.iter().flat_map(|f| f.to_le_bytes()));
        }
        let tmp = path.with_extension(format!("tmp{}", std::process::id()));
        std::fs::write(&tmp, out)?;
        Ok(std::fs::rename(&tmp, path)?)
    }

    pub fn save(&mut self, path: &Path, used: &[u64]) -> Result<()> {
        self.save_at(path, used, now())
    }
}
