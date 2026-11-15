//! Ranks in separate processes, exchanging bytes over sockets.

use crate::Run;
use crate::open;
use engine_settings::Settings;
use lower::{Error, Plan};
use mem::{Tensor, View};
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

/// Moves bytes between ranks. One round trip per call.
pub trait Transport {
    fn rank(&self) -> u32;
    fn ranks(&self) -> u32;
    /// Every rank's bytes, in rank order, `mine` among them.
    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error>;
}

/// The parts summed in rank order. Every rank agrees.
pub(crate) fn summed<'a>(parts: impl Iterator<Item = &'a [f32]>) -> Vec<f32> {
    let mut sum: Option<Vec<f32>> = None;
    for part in parts {
        sum = Some(match sum {
            None => part.to_vec(),
            Some(s) => s.iter().zip(part).map(|(a, b)| a + b).collect(),
        });
    }
    sum.expect("a rank")
}

/// TCP to every other rank. Lower ranks call, higher answer.
pub struct Sockets {
    rank: u32,
    /// By rank. This rank's own entry is empty.
    peers: Vec<Option<TcpStream>>,
}

/// How long a rank waits for the others.
const PATIENCE: Duration = Duration::from_secs(10);

impl Sockets {
    /// Joins rank `rank` of `addrs`, listening on `listener`.
    pub fn join(rank: u32, listener: TcpListener, addrs: &[SocketAddr]) -> io::Result<Self> {
        let mut peers: Vec<Option<TcpStream>> = (0..addrs.len()).map(|_| None).collect();
        for (r, addr) in addrs.iter().enumerate().take(rank as usize) {
            let mut stream = call(*addr)?;
            stream.write_all(&rank.to_le_bytes())?;
            peers[r] = Some(stream);
        }
        for _ in rank as usize + 1..addrs.len() {
            let (mut stream, _) = listener.accept()?;
            stream.set_nodelay(true)?;
            let mut who = [0u8; 4];
            stream.read_exact(&mut who)?;
            let who = u32::from_le_bytes(who) as usize;
            if who <= rank as usize || who >= addrs.len() || peers[who].is_some() {
                return Err(io::Error::other(format!("rank {who} called twice")));
            }
            peers[who] = Some(stream);
        }
        Ok(Self { rank, peers })
    }
}

/// Connects to `addr`, retrying while it is not up yet.
fn call(addr: SocketAddr) -> io::Result<TcpStream> {
    let start = Instant::now();
    loop {
        match TcpStream::connect(addr) {
            Ok(stream) => {
                stream.set_nodelay(true)?;
                return Ok(stream);
            }
            Err(e) if start.elapsed() < PATIENCE => {
                let _ = e;
                thread::sleep(Duration::from_millis(20));
            }
            Err(e) => return Err(e),
        }
    }
}

/// A length, then the bytes.
fn send(stream: &mut TcpStream, bytes: &[u8]) -> io::Result<()> {
    stream.write_all(&(bytes.len() as u64).to_le_bytes())?;
    stream.write_all(bytes)
}

fn receive(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut len = [0u8; 8];
    stream.read_exact(&mut len)?;
    let mut bytes = vec![0u8; u64::from_le_bytes(len) as usize];
    stream.read_exact(&mut bytes)?;
    Ok(bytes)
}

impl Transport for Sockets {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.peers.len() as u32
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        let mut out: Vec<Vec<u8>> = vec![Vec::new(); self.peers.len()];
        out[self.rank as usize] = mine.to_vec();
        // Writers run aside, so no pair waits on each other.
        let mut writers: Vec<TcpStream> = Vec::new();
        for peer in self.peers.iter().flatten() {
            writers.push(peer.try_clone()?);
        }
        thread::scope(|s| -> Result<(), Error> {
            let writing = s.spawn(move || -> io::Result<()> {
                for w in &mut writers {
                    send(w, mine)?;
                }
                Ok(())
            });
            for (r, peer) in self.peers.iter_mut().enumerate() {
                if let Some(peer) = peer {
                    out[r] = receive(peer)?;
                }
            }
            writing.join().expect("the writer does not panic")?;
            Ok(())
        })?;
        Ok(out)
    }
}

/// One rank: its runner and its way to the others.
pub struct Rank<T: Transport> {
    runner: Box<dyn Run>,
    transport: T,
}

impl<T: Transport> Rank<T> {
    /// A runner for the transport's rank of `plan`.
    pub fn open(plan: Plan, settings: &Settings, transport: T) -> Result<Self, Error> {
        if plan.ranks.max(1) != transport.ranks() {
            let what = format!("a plan over {} ranks on {}", plan.ranks, transport.ranks());
            return Err(Error::Unsupported(what));
        }
        let mut runner = open(plan, settings)?;
        runner.base_mut().rank = transport.rank();
        Ok(Self { runner, transport })
    }

    pub fn runner(&self) -> &dyn Run {
        &*self.runner
    }

    /// Takes this rank's slice of the whole weight.
    pub fn load(&mut self, weight: u32, data: View) -> Result<(), Error> {
        self.runner.load(weight, data)
    }

    /// Runs every step. Sums wait for every rank.
    pub fn run(&mut self, inputs: &[View]) -> Result<Vec<Tensor>, Error> {
        self.runner.feed(inputs)?;
        let steps = self.runner.plan().steps.clone();
        for (i, step) in steps.iter().enumerate() {
            if step.kernel.name() != "all_reduce" {
                self.runner.submit(i..i + 1)?;
                continue;
            }
            let part = self.runner.read(step.reads[0])?;
            let parts = self.transport.exchange(part.bytes())?;
            let floats = |p: &Vec<u8>| Tensor::from_bytes(vec![p.len() / 4], mem::DType::F32, p);
            let parts: Vec<Tensor> = parts.iter().map(floats).collect();
            let sum = summed(parts.iter().map(|p| p.floats().expect("floats")));
            let sum = Tensor::f32(vec![step.write.len], sum);
            self.runner.write(step.write, sum.view())?;
        }
        let outputs = self.runner.plan().outputs.clone();
        let read = |o: &lower::Output| Ok(self.runner.read(o.slot)?.reshaped(o.shape.clone()));
        outputs.iter().map(read).collect()
    }
}
