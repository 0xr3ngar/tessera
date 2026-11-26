//! How a split axis divides among ranks, evenly or not.

/// Each rank's weights: for most splits, then for routed sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Parts(pub Vec<u32>, pub Vec<u32>);

impl Parts {
    /// One list for every split.
    pub fn of(weights: Vec<u32>) -> Self {
        Self(weights, Vec::new())
    }

    pub fn even(ranks: usize) -> Self {
        Self::of(vec![1; ranks])
    }

    /// Routed sets divide by `sets`. Empty: by the first list.
    pub fn with_sets(mut self, sets: Vec<u32>) -> Self {
        self.1 = sets;
        self
    }

    pub fn ranks(&self) -> usize {
        self.0.len()
    }

    /// The weights that divide a routed set, or other splits.
    pub fn weights(&self, set: bool) -> &[u32] {
        match set && !self.1.is_empty() {
            true => &self.1,
            false => &self.0,
        }
    }

    /// Each rank's elements of a `whole` axis the sum divides.
    pub fn counts(&self, whole: usize) -> Option<Vec<usize>> {
        self.counts_of(false, whole)
    }

    /// Rank `rank`'s start and length along a `whole` axis.
    pub fn part(&self, whole: usize, rank: usize) -> Option<(usize, usize)> {
        self.part_of(false, whole, rank)
    }

    /// As `counts`, for a routed set when `set`.
    pub fn counts_of(&self, set: bool, whole: usize) -> Option<Vec<usize>> {
        let weights = self.weights(set);
        let sum: u32 = weights.iter().sum();
        if sum == 0 || !whole.is_multiple_of(sum as usize) {
            return None;
        }
        let unit = whole / sum as usize;
        Some(weights.iter().map(|w| *w as usize * unit).collect())
    }

    /// As `part`, for a routed set when `set`.
    pub fn part_of(&self, set: bool, whole: usize, rank: usize) -> Option<(usize, usize)> {
        let counts = self.counts_of(set, whole)?;
        let first = counts[..rank].iter().sum();
        Some((first, counts[rank]))
    }
}
