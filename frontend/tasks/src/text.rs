//! Token ids to text, a piece at a time.

/// Decoded text, and how much of it was handed out.
#[derive(Default)]
pub(crate) struct Spool {
    ids: Vec<u32>,
    shown: usize,
}

impl Spool {
    /// Adds `id`. Returns the text it completes, if any.
    pub fn push(&mut self, id: u32, decode: impl Fn(&[u32]) -> String) -> String {
        self.ids.push(id);
        let text = decode(&self.ids);
        // A token can end mid-character: the replacement character.
        if text.ends_with('\u{fffd}') || text.len() < self.shown {
            return String::new();
        }
        let fresh = text[self.shown..].to_string();
        self.shown = text.len();
        fresh
    }
}

/// Text that stop strings can cut, held until they cannot.
pub(crate) struct Stops {
    stops: Vec<String>,
    text: String,
    /// Bytes of `text` handed out.
    sent: usize,
    /// Characters held back: the longest stop, less one.
    hold: usize,
}

/// The byte index `n` characters before byte `end`.
fn back(text: &str, end: usize, n: usize) -> usize {
    match n {
        0 => end,
        _ => text[..end]
            .char_indices()
            .rev()
            .nth(n - 1)
            .map_or(0, |(i, _)| i),
    }
}

impl Stops {
    pub fn new(stops: Vec<String>) -> Self {
        let stops: Vec<String> = stops.into_iter().filter(|s| !s.is_empty()).collect();
        let longest = stops.iter().map(|s| s.chars().count()).max();
        Self {
            stops,
            text: String::new(),
            sent: 0,
            hold: longest.unwrap_or(1) - 1,
        }
    }

    /// Adds `piece`. Returns text safe to show, and if cut.
    pub fn push(&mut self, piece: &str) -> (String, bool) {
        let before = self.text.len();
        self.text.push_str(piece);
        // Only a match that ends in `piece` is new.
        let from = back(&self.text, before, self.hold);
        let mut cut: Option<usize> = None;
        for stop in &self.stops {
            if let Some(at) = self.text[from..].find(stop.as_str()) {
                cut = Some(cut.map_or(from + at, |c| c.min(from + at)));
            }
        }
        let end = match cut {
            Some(at) => {
                self.text.truncate(at);
                at
            }
            None => back(&self.text, self.text.len(), self.hold).max(self.sent),
        };
        let out = self.text[self.sent..end].to_string();
        self.sent = end;
        (out, cut.is_some())
    }

    /// The text still held, once the answer ends.
    pub fn rest(&mut self) -> String {
        let out = self.text[self.sent..].to_string();
        self.sent = self.text.len();
        out
    }
}
