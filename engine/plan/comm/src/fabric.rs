//! Which ranks a rank reaches, and over which ports.

use crate::Error;

/// Slots start with the length, then pad to 16.
pub(crate) const HEADER: usize = 16;

/// An RDMA device and the GID index to send from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Port {
    pub device: String,
    pub gid: u32,
}

/// How the ranks are cabled, one lane per port.
#[derive(Clone, Debug)]
pub enum Fabric {
    /// Every rank reaches every other over the same ports.
    Mesh(Vec<Port>),
    /// Rank r reaches r-1 over `prev` and r+1 over `next`.
    Ring { prev: Vec<Port>, next: Vec<Port> },
}

/// One link to a peer: a queue pair per lane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Channel {
    pub peer: u32,
    /// Port index per lane. Empty for the rank itself.
    pub ports: Vec<usize>,
    /// The peer's channel that faces this rank.
    pub meets: usize,
}

/// A rank's channels, its send order and its relay.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Wiring {
    pub channels: Vec<Channel>,
    /// Channel indexes, in the order a round posts to them.
    pub sends: Vec<usize>,
    /// Forward what arrives on one channel over another.
    pub relay: Option<(usize, usize)>,
}

impl Fabric {
    /// Every port, in lane order.
    pub(crate) fn ports(&self) -> Vec<Port> {
        match self {
            Fabric::Mesh(ports) => ports.clone(),
            Fabric::Ring { prev, next } => [prev.clone(), next.clone()].concat(),
        }
    }

    pub fn lanes(&self) -> usize {
        match self {
            Fabric::Mesh(ports) => ports.len(),
            Fabric::Ring { prev, .. } => prev.len(),
        }
    }

    /// The wiring of `rank` among `ranks`.
    pub(crate) fn wiring(&self, rank: u32, ranks: u32) -> Result<Wiring, Error> {
        let lanes = self.lanes();
        let fits = match self {
            Fabric::Mesh(_) => ranks >= 2,
            Fabric::Ring { prev, next } => (2..=4).contains(&ranks) && prev.len() == next.len(),
        };
        if !fits || lanes == 0 || rank >= ranks {
            let what = format!("rank {rank} of {ranks} over {lanes} lanes");
            return Err(Error::Fit(what));
        }
        Ok(match self {
            Fabric::Mesh(_) => {
                let channel = |peer: u32| Channel {
                    peer,
                    ports: match peer == rank {
                        true => Vec::new(),
                        false => (0..lanes).collect(),
                    },
                    meets: rank as usize,
                };
                Wiring {
                    channels: (0..ranks).map(channel).collect(),
                    sends: (1..ranks).map(|s| ((rank + s) % ranks) as usize).collect(),
                    relay: None,
                }
            }
            Fabric::Ring { .. } => {
                let prev = Channel {
                    peer: (rank + ranks - 1) % ranks,
                    ports: (0..lanes).collect(),
                    meets: 1,
                };
                let next = Channel {
                    peer: (rank + 1) % ranks,
                    ports: (lanes..2 * lanes).collect(),
                    meets: 0,
                };
                Wiring {
                    channels: vec![prev, next],
                    sends: match ranks {
                        2 => vec![1],
                        _ => vec![1, 0],
                    },
                    relay: (ranks == 4).then_some((0, 1)),
                }
            }
        })
    }
}

/// `total` bytes over `lanes`, as (start, length) in 16s.
pub(crate) fn parts(total: usize, lanes: usize) -> Vec<(usize, usize)> {
    let step = (total.div_ceil(lanes) + 15) & !15;
    (0..lanes)
        .map(|lane| {
            let start = (lane * step).min(total);
            (start, ((lane + 1) * step).min(total) - start)
        })
        .collect()
}

/// What a rank tells the others about one queue pair.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct Lane {
    pub qpn: u32,
    pub mtu: u32,
    pub recv_rkey: u32,
    pub flag_rkey: u32,
    pub gid: [u8; 16],
}

/// A rank's regions and queue pairs, for its peers.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
pub(crate) struct Peer {
    pub recv: u64,
    pub flags: u64,
    /// Per channel, per lane.
    pub channels: Vec<Vec<Lane>>,
}

impl Peer {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&self.recv.to_le_bytes());
        out.extend_from_slice(&self.flags.to_le_bytes());
        out.extend_from_slice(&(self.channels.len() as u32).to_le_bytes());
        for lanes in &self.channels {
            out.extend_from_slice(&(lanes.len() as u32).to_le_bytes());
            for l in lanes {
                for word in [l.qpn, l.mtu, l.recv_rkey, l.flag_rkey] {
                    out.extend_from_slice(&word.to_le_bytes());
                }
                out.extend_from_slice(&l.gid);
            }
        }
        out
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Option<Self> {
        let mut at = Cursor(bytes);
        let recv = at.u64()?;
        let flags = at.u64()?;
        let mut channels = Vec::new();
        for _ in 0..at.u32()? {
            let mut lanes = Vec::new();
            for _ in 0..at.u32()? {
                lanes.push(Lane {
                    qpn: at.u32()?,
                    mtu: at.u32()?,
                    recv_rkey: at.u32()?,
                    flag_rkey: at.u32()?,
                    gid: at.take(16)?.try_into().ok()?,
                });
            }
            channels.push(lanes);
        }
        at.0.is_empty().then_some(Self {
            recv,
            flags,
            channels,
        })
    }
}

struct Cursor<'a>(&'a [u8]);

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> Option<&[u8]> {
        let (head, rest) = self.0.split_at_checked(n)?;
        self.0 = rest;
        Some(head)
    }

    fn u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.take(4)?.try_into().ok()?))
    }

    fn u64(&mut self) -> Option<u64> {
        Some(u64::from_le_bytes(self.take(8)?.try_into().ok()?))
    }
}
