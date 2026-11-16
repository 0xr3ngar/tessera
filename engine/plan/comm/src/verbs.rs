//! Ranks over RDMA writes into each other's slots.

use crate::fabric::{Fabric, HEADER, Lane, Peer, Wiring, parts};
use crate::ibv::{self, Gid, Lib, Mr, QpAttr, QpInitAttr, SendWr, Sge, Wc, mask};
use crate::{Error, Transport};
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// How long a rank waits at a round.
const ROUND: Duration = Duration::from_secs(300);
/// Queue pair retries: 4.096 us times 2 to this.
const QP_TIMEOUT: u8 = 18;
/// Completions per device per poll.
const BATCH: usize = 16;

/// Bytes the devices register, page aligned.
struct Region {
    base: *mut u8,
    len: usize,
}

impl Region {
    fn new(len: usize) -> Self {
        let layout = Layout::from_size_align(len.max(1), 4096).expect("a page");
        let base = unsafe { alloc_zeroed(layout) };
        assert!(!base.is_null(), "{len} bytes to register");
        Self { base, len }
    }

    fn at(&self, offset: usize) -> *mut u8 {
        unsafe { self.base.add(offset) }
    }

    fn word(&self, at: usize) -> &AtomicU64 {
        unsafe { &*(self.at(8 * at) as *const AtomicU64) }
    }
}

impl Drop for Region {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len.max(1), 4096).expect("a page");
        unsafe { dealloc(self.base, layout) };
    }
}

/// One port: its context, domain and registrations.
struct Opened {
    ctx: *mut ibv::Context,
    pd: *mut ibv::Pd,
    cq: *mut ibv::Cq,
    /// The send, receive and flag regions.
    mrs: [*mut Mr; 3],
    gid: Gid,
    mtu: u32,
    sgid: u32,
}

struct Link {
    qp: *mut ibv::Qp,
    port: usize,
    posted: u64,
}

/// Two slots per source, by round parity.
pub struct Verbs {
    rank: u32,
    ranks: u32,
    lib: &'static Lib,
    wiring: Wiring,
    lanes: usize,
    opened: Vec<Opened>,
    /// Per channel, per lane.
    links: Vec<Vec<Link>>,
    peers: Vec<Peer>,
    /// Bytes per slot, header included.
    capacity: usize,
    generation: u64,
    /// The last round the relay forwarded.
    relayed: u64,
    send: Region,
    recv: Region,
    flags: Region,
}

// Each rank owns its queue pairs and regions.
unsafe impl Send for Verbs {}

fn failed(what: &str) -> Error {
    Error::Link(format!("{what}: {}", std::io::Error::last_os_error()))
}

impl Verbs {
    /// Rank `rank` of `ranks` over `fabric`, met through `boot`.
    pub fn connect(
        rank: u32,
        ranks: u32,
        fabric: &Fabric,
        capacity: usize,
        boot: &mut dyn Transport,
    ) -> Result<Self, Error> {
        let lib = Lib::get().map_err(Error::Link)?;
        let wiring = fabric.wiring(rank, ranks)?;
        if (boot.rank(), boot.ranks()) != (rank, ranks) {
            let what = format!(
                "rank {rank} of {ranks} met over {} of {}",
                boot.rank(),
                boot.ranks()
            );
            return Err(Error::Fit(what));
        }
        let capacity = (capacity.max(HEADER + 1) + 15) & !15;
        let lanes = fabric.lanes();
        let mut this = Self {
            rank,
            ranks,
            lib,
            wiring,
            lanes,
            opened: Vec::new(),
            links: Vec::new(),
            peers: Vec::new(),
            capacity,
            generation: 0,
            relayed: 0,
            send: Region::new(2 * capacity),
            recv: Region::new(2 * ranks as usize * capacity),
            flags: Region::new(8 * ranks as usize * lanes),
        };
        for port in fabric.ports() {
            let opened = this.open(&port.device, port.gid)?;
            this.opened.push(opened);
        }
        for c in 0..this.wiring.channels.len() {
            let mut links = Vec::new();
            for lane in 0..this.wiring.channels[c].ports.len() {
                links.push(this.link(c, lane)?);
            }
            this.links.push(links);
        }
        let mine = this.described();
        let all = boot.exchange(&mine.to_bytes())?;
        for (r, bytes) in all.iter().enumerate() {
            let peer = Peer::from_bytes(bytes).ok_or(Error::Fit(format!("rank {r}'s details")))?;
            this.peers.push(peer);
        }
        for c in 0..this.links.len() {
            for lane in 0..this.links[c].len() {
                this.wire(c, lane)?;
            }
        }
        // A barrier: every rank can receive now.
        boot.exchange(&[])?;
        Ok(this)
    }

    fn open(&self, device: &str, sgid: u32) -> Result<Opened, Error> {
        let lib = self.lib;
        let ctx = lib.device(device).map_err(Error::Link)?;
        let pd = unsafe { (lib.alloc_pd)(ctx) };
        if pd.is_null() {
            return Err(failed("alloc_pd"));
        }
        let access = (ibv::ACCESS_LOCAL_WRITE | ibv::ACCESS_REMOTE_WRITE) as i32;
        let mut mrs = [std::ptr::null_mut(); 3];
        for (mr, region) in mrs.iter_mut().zip([&self.send, &self.recv, &self.flags]) {
            *mr = unsafe { (lib.reg_mr)(pd, region.base as *mut c_void, region.len, access) };
            if mr.is_null() {
                return Err(failed("reg_mr"));
            }
        }
        let null = std::ptr::null_mut();
        let cq = unsafe { (lib.create_cq)(ctx, 4096, null, null, 0) };
        if cq.is_null() {
            return Err(failed("create_cq"));
        }
        let mut gid = Gid::default();
        if unsafe { (lib.query_gid)(ctx, 1, sgid as i32, &mut gid) } != 0 {
            return Err(failed("query_gid"));
        }
        let mut port = ibv::PortAttr::default();
        if unsafe { (lib.query_port)(ctx, 1, &mut port) } != 0 {
            return Err(failed("query_port"));
        }
        Ok(Opened {
            ctx,
            pd,
            cq,
            mrs,
            gid,
            mtu: port.active_mtu,
            sgid,
        })
    }

    /// A queue pair in its first state.
    fn link(&self, channel: usize, lane: usize) -> Result<Link, Error> {
        let port = self.wiring.channels[channel].ports[lane];
        let opened = &self.opened[port];
        let mut init = QpInitAttr {
            qp_context: std::ptr::null_mut(),
            send_cq: opened.cq,
            recv_cq: opened.cq,
            srq: std::ptr::null_mut(),
            cap: ibv::QpCap {
                max_send_wr: 1024,
                max_recv_wr: 1,
                max_send_sge: 1,
                max_recv_sge: 1,
                max_inline_data: 64,
            },
            qp_type: ibv::QPT_RC,
            sq_sig_all: 0,
        };
        let qp = unsafe { (self.lib.create_qp)(opened.pd, &mut init) };
        if qp.is_null() {
            return Err(failed("create_qp"));
        }
        let mut attr = QpAttr {
            qp_state: ibv::QPS_INIT,
            pkey_index: 0,
            port_num: 1,
            qp_access_flags: ibv::ACCESS_LOCAL_WRITE | ibv::ACCESS_REMOTE_WRITE,
            ..Default::default()
        };
        let bits = mask::STATE | mask::PKEY_INDEX | mask::PORT | mask::ACCESS_FLAGS;
        self.modify(qp, &mut attr, bits)?;
        Ok(Link {
            qp,
            port,
            posted: 0,
        })
    }

    fn modify(&self, qp: *mut ibv::Qp, attr: &mut QpAttr, bits: i32) -> Result<(), Error> {
        match unsafe { (self.lib.modify_qp)(qp, attr, bits) } {
            0 => Ok(()),
            e => Err(Error::Link(format!(
                "modify_qp to state {}: {}",
                attr.qp_state,
                std::io::Error::from_raw_os_error(e)
            ))),
        }
    }

    /// This rank, as its peers must see it.
    fn described(&self) -> Peer {
        let lane = |link: &Link| {
            let opened = &self.opened[link.port];
            Lane {
                qpn: unsafe { (*link.qp).qp_num },
                mtu: opened.mtu,
                recv_rkey: unsafe { (*opened.mrs[1]).rkey },
                flag_rkey: unsafe { (*opened.mrs[2]).rkey },
                gid: opened.gid.0,
            }
        };
        Peer {
            recv: self.recv.base as u64,
            flags: self.flags.base as u64,
            channels: self
                .links
                .iter()
                .map(|l| l.iter().map(lane).collect())
                .collect(),
        }
    }

    /// The far end of `channel`'s queue pair on `lane`.
    fn far(&self, channel: usize, lane: usize) -> Result<(&Peer, &Lane), Error> {
        let c = &self.wiring.channels[channel];
        let peer = &self.peers[c.peer as usize];
        let far = peer.channels.get(c.meets).and_then(|l| l.get(lane));
        let what = || format!("rank {} has no lane {lane} to rank {}", c.peer, self.rank);
        Ok((peer, far.ok_or_else(|| Error::Fit(what()))?))
    }

    /// Brings the queue pair up to send and receive.
    fn wire(&mut self, channel: usize, lane: usize) -> Result<(), Error> {
        let link = &self.links[channel][lane];
        let opened = &self.opened[link.port];
        let (_, far) = self.far(channel, lane)?;
        let mut attr = QpAttr {
            qp_state: ibv::QPS_RTR,
            path_mtu: opened.mtu.min(far.mtu),
            dest_qp_num: far.qpn,
            rq_psn: 0,
            max_dest_rd_atomic: 1,
            min_rnr_timer: 12,
            ..Default::default()
        };
        attr.ah_attr.is_global = 1;
        attr.ah_attr.port_num = 1;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.sgid_index = opened.sgid as u8;
        attr.ah_attr.grh.dgid = Gid(far.gid);
        let bits = mask::STATE
            | mask::AV
            | mask::PATH_MTU
            | mask::DEST_QPN
            | mask::RQ_PSN
            | mask::MAX_DEST_RD_ATOMIC
            | mask::MIN_RNR_TIMER;
        self.modify(link.qp, &mut attr, bits)?;
        let mut attr = QpAttr {
            qp_state: ibv::QPS_RTS,
            timeout: QP_TIMEOUT,
            retry_cnt: 7,
            rnr_retry: 7,
            sq_psn: 0,
            max_rd_atomic: 1,
            ..Default::default()
        };
        let bits = mask::STATE
            | mask::TIMEOUT
            | mask::RETRY_CNT
            | mask::RNR_RETRY
            | mask::SQ_PSN
            | mask::MAX_QP_RD_ATOMIC;
        self.modify(link.qp, &mut attr, bits)
    }

    fn slot(&self, parity: usize, source: u32) -> usize {
        (parity * self.ranks as usize + source as usize) * self.capacity
    }

    fn flag(&self, source: u32, lane: usize) -> &AtomicU64 {
        self.flags.word(source as usize * self.lanes + lane)
    }

    /// Every lane's flag from `source` reached `seq`.
    fn arrived(&self, source: u32, seq: u64) -> bool {
        (0..self.lanes).all(|lane| self.flag(source, lane).load(Ordering::Acquire) >= seq)
    }

    /// Writes a region's bytes to the peer, then flags `seq`.
    fn post(
        &mut self,
        channel: usize,
        from: (usize, usize),
        total: usize,
        slot: usize,
        source: u32,
        seq: u64,
    ) -> Result<(), Error> {
        let (which, at) = from;
        let region = match which {
            0 => &self.send,
            _ => &self.recv,
        };
        let c = &self.wiring.channels[channel];
        let peer = &self.peers[c.peer as usize];
        for (lane, (start, len)) in parts(total, self.lanes).into_iter().enumerate() {
            let link = &mut self.links[channel][lane];
            let far = peer.channels[c.meets][lane];
            let lkey = unsafe { (*self.opened[link.port].mrs[which]).lkey };
            let mut data = Sge {
                addr: region.at(at + start) as u64,
                length: len as u32,
                lkey,
            };
            let to = peer.recv + (slot + start) as u64;
            let mut write = SendWr::write(&mut data, to, far.recv_rkey, 0);
            let mut seq_word = seq;
            let mut word = Sge {
                addr: &mut seq_word as *mut u64 as u64,
                length: 8,
                lkey: 0,
            };
            link.posted += 1;
            let mut bits = ibv::SEND_INLINE;
            if link.posted.is_multiple_of(32) {
                bits |= ibv::SEND_SIGNALED;
            }
            let to = peer.flags + 8 * (source as usize * self.lanes + lane) as u64;
            let mut flag = SendWr::write(&mut word, to, far.flag_rkey, bits);
            let head = match len {
                0 => &mut flag,
                _ => {
                    write.next = &mut flag;
                    &mut write
                }
            };
            let e = self.lib.post_send(link.qp, head);
            if e != 0 {
                let why = std::io::Error::from_raw_os_error(e);
                return Err(Error::Link(format!("post to rank {}: {why}", c.peer)));
            }
        }
        Ok(())
    }

    /// Completions with errors stop the round.
    fn poll(&self) -> Result<(), Error> {
        let mut wc = [unsafe { std::mem::zeroed::<Wc>() }; BATCH];
        for opened in &self.opened {
            let n = self.lib.poll_cq(opened.cq, &mut wc);
            for c in wc.iter().take(n.max(0) as usize) {
                if c.status != ibv::WC_SUCCESS {
                    let what = format!("completion: {}", self.lib.status(c.status));
                    return Err(Error::Link(what));
                }
            }
        }
        Ok(())
    }

    /// Forwards the relay channel's arrival once per round.
    fn relay(&mut self, seq: u64) -> Result<(), Error> {
        let Some((from, to)) = self.wiring.relay else {
            return Ok(());
        };
        let source = self.wiring.channels[from].peer;
        if self.relayed >= seq || !self.arrived(source, seq) {
            return Ok(());
        }
        let slot = self.slot((seq % 2) as usize, source);
        let total = HEADER + self.recv.word(slot / 8).load(Ordering::Relaxed) as usize;
        self.relayed = seq;
        self.post(to, (1, slot), total, slot, source, seq)
    }

    fn wait(&mut self, seq: u64) -> Result<(), Error> {
        let start = Instant::now();
        let mut spins = 0u32;
        loop {
            self.relay(seq)?;
            if (0..self.ranks).all(|r| r == self.rank || self.arrived(r, seq)) {
                return self.poll();
            }
            spins = spins.wrapping_add(1);
            if spins.is_multiple_of(1024) {
                self.poll()?;
                if start.elapsed() > ROUND {
                    let late = (0..self.ranks).find(|r| !self.arrived(*r, seq));
                    let what = format!("rank {} never reached round {seq}", late.unwrap_or(0));
                    return Err(Error::Link(what));
                }
                thread::yield_now();
            }
        }
    }
}

impl Transport for Verbs {
    fn rank(&self) -> u32 {
        self.rank
    }

    fn ranks(&self) -> u32 {
        self.ranks
    }

    fn exchange(&mut self, mine: &[u8]) -> Result<Vec<Vec<u8>>, Error> {
        if HEADER + mine.len() > self.capacity {
            let what = format!(
                "{} bytes over a {} byte slot",
                mine.len(),
                self.capacity - HEADER
            );
            return Err(Error::Fit(what));
        }
        let next = self.generation + 1;
        let parity = (next % 2) as usize;
        let at = parity * self.capacity;
        self.send
            .word(at / 8)
            .store(mine.len() as u64, Ordering::Relaxed);
        unsafe {
            std::ptr::copy_nonoverlapping(mine.as_ptr(), self.send.at(at + HEADER), mine.len())
        };
        let total = HEADER + mine.len();
        let slot = self.slot(parity, self.rank);
        for c in self.wiring.sends.clone() {
            self.post(c, (0, at), total, slot, self.rank, next)?;
        }
        self.wait(next)?;
        let mut out = Vec::with_capacity(self.ranks as usize);
        for r in 0..self.ranks {
            if r == self.rank {
                out.push(mine.to_vec());
                continue;
            }
            let slot = self.slot(parity, r);
            let len = self.recv.word(slot / 8).load(Ordering::Relaxed) as usize;
            if HEADER + len > self.capacity {
                return Err(Error::Link(format!("rank {r} sent {len} bytes")));
            }
            let bytes = unsafe { std::slice::from_raw_parts(self.recv.at(slot + HEADER), len) };
            out.push(bytes.to_vec());
        }
        self.generation = next;
        Ok(out)
    }
}

impl Drop for Verbs {
    fn drop(&mut self) {
        let lib = self.lib;
        for links in &self.links {
            for link in links {
                unsafe { (lib.destroy_qp)(link.qp) };
            }
        }
        for opened in &self.opened {
            for mr in opened.mrs {
                unsafe { (lib.dereg_mr)(mr) };
            }
            unsafe { (lib.destroy_cq)(opened.cq) };
            unsafe { (lib.dealloc_pd)(opened.pd) };
            unsafe { (lib.close_device)(opened.ctx) };
        }
    }
}
