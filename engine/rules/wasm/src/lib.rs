//! Rules a wasm module proposes, given sizes as JSON.

mod abi;
mod guest;
mod text;

use abi::{ABI, Ask, Kind, Manifest, Offer, Site};
use guest::Guest;
use lower::{Draft, Propose, Read, Rule, T, Tree, sites};
use std::sync::Arc;
use text::{Fixed, Text};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("wasm: {0}")]
    Wasm(String),
    #[error("the module has no export {0}")]
    Missing(&'static str),
    #[error("the module's JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("the module speaks abi {0}, the host {ABI}")]
    Abi(u32),
    #[error("an earlier call panicked")]
    Poisoned,
}

impl Error {
    fn wasm(e: impl std::fmt::Display) -> Self {
        Error::Wasm(e.to_string())
    }
}

/// One declared rule, asking the shared module.
struct Asking {
    guest: Arc<Guest>,
    rule: &'static str,
    kind: Kind,
    kernels: &'static [Fixed],
}

impl Asking {
    /// The module's offers, or the reason it gave none.
    fn ask(&self, site: Site) -> Result<Option<Vec<Offer>>, Error> {
        let rule = self.rule;
        let input = serde_json::to_vec(&Ask { rule, site })?;
        let output = self.guest.call("propose", Some(&input))?;
        Ok(serde_json::from_slice(&output)?)
    }

    /// Drafts for the offers. `None` on unknown names.
    fn drafts(&self, offers: Vec<Offer>, reads: &[Read]) -> Option<Vec<Draft>> {
        let mut drafts = Vec::with_capacity(offers.len());
        for offer in offers {
            let fixed = self.kernels.iter().find(|k| k.name == offer.kernel)?;
            let mut taken = Vec::with_capacity(offer.reads.len());
            for i in &offer.reads {
                taken.push(*reads.get(*i)?);
            }
            drafts.push(Draft::new(Text::new(fixed, &offer), taken, offer.len));
        }
        Some(drafts)
    }
}

impl Propose for Asking {
    fn at(&self, tree: &Tree, at: T) -> Option<Vec<Draft>> {
        let m: sites::Matmul = tree.site(at)?;
        if self.kind != Kind::Matmul || m.picked.is_some() {
            return None;
        }
        let site = Site::Matmul {
            m: m.m,
            n: m.n,
            k: m.k,
            form: m.form.name(),
        };
        match self.ask(site) {
            Ok(offers) => self.drafts(offers?, &m.reads()),
            Err(e) => {
                eprintln!("wasm rule {}: {e}", self.rule);
                None
            }
        }
    }
}

fn leak(text: String) -> &'static str {
    Box::leak(text.into_boxed_str())
}

/// The rules `bytes` declares. They live until the process ends.
pub fn load(bytes: &[u8]) -> Result<Vec<&'static Rule>, Error> {
    let guest = Arc::new(Guest::new(bytes)?);
    let manifest: Manifest = serde_json::from_slice(&guest.call("manifest", None)?)?;
    if manifest.abi != ABI {
        return Err(Error::Abi(manifest.abi));
    }
    let mut kernels = Vec::with_capacity(manifest.kernels.len());
    for k in manifest.kernels {
        kernels.push(Fixed {
            name: leak(k.name),
            entry: leak(k.entry),
            metal: leak(k.metal),
            cuda: leak(k.cuda),
        });
    }
    let kernels: &'static [Fixed] = kernels.leak();
    let mut rules = Vec::with_capacity(manifest.rules.len());
    for r in manifest.rules {
        let rule = leak(r.name);
        let asking = Asking {
            guest: guest.clone(),
            rule,
            kind: r.site,
            kernels,
        };
        rules.push(Rule::made(rule, module_path!(), asking));
    }
    Ok(rules)
}
