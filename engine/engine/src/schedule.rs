//! Many answers in flight, one step feeding them all.

use crate::{Engine, Error, Event, Fault, Reason, Request, Row};
use std::collections::VecDeque;

/// Takes each event. `false` ends the answer early.
pub type Sink = Box<dyn FnMut(Event) -> bool + Send>;

/// An answer in flight, in its slot.
pub struct Seq {
    pub slot: u32,
    request: Request,
    sink: Sink,
    /// Tokens fed: the prompt, then each answer token.
    fed: usize,
    /// Tokens to feed: the prompt, then picked and forced tokens.
    pending: VecDeque<u32>,
    made: usize,
}

impl Seq {
    /// The row feeding pending token `i`.
    pub fn row(&self, i: usize) -> Option<Row> {
        Some(Row {
            slot: self.slot,
            token: *self.pending.get(i)?,
            position: (self.fed + i) as u32,
        })
    }

    /// Tokens still to feed.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Whether more than one token waits, as with a prompt.
    pub fn prefilling(&self) -> bool {
        self.pending.len() > 1
    }

    /// Counts `n` tokens fed. `id` follows the last one.
    fn advance(&mut self, n: usize, id: u32) -> Result<Option<Reason>, Error> {
        self.pending.drain(..n);
        self.fed += n;
        if !self.pending.is_empty() {
            return Ok(None);
        }
        if let Some(guide) = &mut self.request.guide {
            guide.accept(id).map_err(fault)?;
        }
        if self.request.stop.contains(&id) {
            return Ok(Some(Reason::Stop));
        }
        if let Some(reason) = self.emit(id) {
            return Ok(Some(reason));
        }
        self.pending.push_back(id);
        self.force()
    }

    /// Sends an answer token.
    fn emit(&mut self, id: u32) -> Option<Reason> {
        self.made += 1;
        if !(self.sink)(Event::Token(id)) {
            return Some(Reason::Stop);
        }
        (self.made >= self.request.max_tokens).then_some(Reason::Length)
    }

    /// Queues and sends the guide's forced tokens.
    fn force(&mut self) -> Result<Option<Reason>, Error> {
        let Some(guide) = &mut self.request.guide else {
            return Ok(None);
        };
        let (forced, done) = (guide.forced(), guide.done());
        for id in forced {
            if self.request.stop.contains(&id) {
                return Ok(Some(Reason::Stop));
            }
            if let Some(reason) = self.emit(id) {
                return Ok(Some(reason));
            }
            self.pending.push_back(id);
        }
        Ok(done.then_some(Reason::Stop))
    }
}

fn fault(e: Fault) -> Error {
    Error::Guide(e.to_string())
}

/// Which rows the next step feeds, within a row budget.
pub trait Policy {
    fn pick(&mut self, seqs: &[Seq], budget: usize) -> Vec<Row>;
}

/// Decodes first, then a prompt token each in turn.
pub struct Fair;

impl Policy for Fair {
    fn pick(&mut self, seqs: &[Seq], budget: usize) -> Vec<Row> {
        let decodes = seqs.iter().filter(|s| !s.prefilling());
        let mut rows: Vec<Row> = decodes.filter_map(|s| s.row(0)).collect();
        let mut prompts: Vec<(&Seq, usize)> = seqs
            .iter()
            .filter(|s| s.prefilling())
            .map(|s| (s, 0))
            .collect();
        loop {
            let mut any = false;
            for (s, fed) in &mut prompts {
                let Some(row) = s.row(*fed).filter(|_| rows.len() < budget) else {
                    continue;
                };
                rows.push(row);
                *fed += 1;
                any = true;
            }
            if !any {
                return rows;
            }
        }
    }
}

/// Many answers over one engine, a step at a time.
pub struct Batch<'a> {
    engine: &'a mut Engine,
    policy: Box<dyn Policy>,
    waiting: VecDeque<(Request, Sink)>,
    active: Vec<Seq>,
    free: Vec<u32>,
    /// The most rows one step feeds.
    budget: usize,
}

impl<'a> Batch<'a> {
    pub fn new(engine: &'a mut Engine, policy: Box<dyn Policy>) -> Self {
        let free = (0..engine.slots as u32).rev().collect();
        let budget = engine.context;
        Self {
            engine,
            policy,
            waiting: VecDeque::new(),
            active: Vec::new(),
            free,
            budget,
        }
    }

    /// Queues an answer. An unfit prompt ends at once.
    pub fn submit(&mut self, request: Request, mut sink: Sink) {
        let long = request.tokens.len() > self.engine.context;
        if request.tokens.is_empty() || long {
            sink(Event::Done(Reason::Context));
            return;
        }
        if request.max_tokens == 0 {
            sink(Event::Done(Reason::Length));
            return;
        }
        self.waiting.push_back((request, sink));
    }

    /// Whether any answer is queued or in flight.
    pub fn busy(&self) -> bool {
        !self.waiting.is_empty() || !self.active.is_empty()
    }

    /// Admits waiting answers while slots are free.
    fn admit(&mut self) -> Result<(), Error> {
        while let Some(slot) = self.free.pop() {
            let Some((request, sink)) = self.waiting.pop_front() else {
                self.free.push(slot);
                return Ok(());
            };
            let mut seq = Seq {
                slot,
                pending: request.tokens.iter().copied().collect(),
                request,
                sink,
                fed: 0,
                made: 0,
            };
            match seq.force()? {
                Some(reason) => {
                    (seq.sink)(Event::Done(reason));
                    self.free.push(slot);
                }
                None => self.active.push(seq),
            }
        }
        Ok(())
    }

    /// The most rows one step feeds.
    pub fn budget(&mut self, rows: usize) {
        self.budget = rows;
    }

    /// Ends each answer whose next row lies past the context.
    fn fit(&mut self) {
        let context = self.engine.context;
        let mut i = 0;
        while i < self.active.len() {
            let seq = &self.active[i];
            if seq.fed + seq.pending() <= context {
                i += 1;
                continue;
            }
            (self.active[i].sink)(Event::Done(Reason::Context));
            self.finish(i);
        }
    }

    /// Each pick's mask, if any guide limits one.
    fn masks(&mut self, rows: &[Row], ends: &[u32]) -> Result<Option<Vec<u32>>, Error> {
        if self.active.iter().all(|s| s.request.guide.is_none()) {
            return Ok(None);
        }
        let words = self.engine.words().unwrap_or_default();
        let mut out = vec![u32::MAX; ends.len() * words];
        let mut limited = false;
        for seq in &mut self.active {
            let Some(guide) = &mut seq.request.guide else {
                continue;
            };
            let Some(k) = ends.iter().position(|&e| rows[e as usize].slot == seq.slot) else {
                continue;
            };
            // Only when the answer keeps this pick.
            let mine = rows.iter().filter(|r| r.slot == seq.slot).count();
            if mine != seq.pending.len() {
                continue;
            }
            let row = &mut out[k * words..(k + 1) * words];
            limited |= guide.mask(row).map_err(fault)?;
        }
        if limited && words == 0 {
            return Err(Error::Mask);
        }
        Ok(limited.then_some(out))
    }

    fn finish(&mut self, i: usize) {
        let seq = self.active.swap_remove(i);
        self.free.push(seq.slot);
    }

    /// One step of the policy's rows. False when nothing ran.
    pub fn step(&mut self) -> Result<bool, Error> {
        self.admit()?;
        self.fit();
        let rows = self.policy.pick(&self.active, self.budget);
        if rows.is_empty() {
            return Ok(false);
        }
        let ends = crate::ends(&rows);
        let ids = match self.masks(&rows, &ends)? {
            Some(masks) => self.engine.last_masked(&rows, &masks)?,
            None => self.engine.last(&rows)?,
        };
        // Each answer takes its rows and its last id.
        let mut i = 0;
        while i < self.active.len() {
            let slot = self.active[i].slot;
            let n = rows.iter().filter(|r| r.slot == slot).count();
            let mine = ends
                .iter()
                .zip(&ids)
                .find(|(e, _)| rows[**e as usize].slot == slot);
            match mine {
                None => i += 1,
                Some((_, &last)) => match self.active[i].advance(n, last)? {
                    Some(reason) => {
                        (self.active[i].sink)(Event::Done(reason));
                        self.finish(i);
                    }
                    None => i += 1,
                },
            }
        }
        Ok(true)
    }
}
