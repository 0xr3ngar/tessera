//! libibverbs, loaded at run time with dlopen.

use crate::dl::message;
use std::ffi::{c_char, c_int, c_void};
use std::io;
use std::sync::OnceLock;

pub const QPT_RC: u32 = 2;
const WR_RDMA_WRITE: u32 = 0;
pub const SEND_SIGNALED: u32 = 2;
pub const SEND_INLINE: u32 = 8;
pub const QPS_INIT: u32 = 1;
pub const QPS_RTR: u32 = 2;
pub const QPS_RTS: u32 = 3;
pub const ACCESS_LOCAL_WRITE: u32 = 1;
pub const ACCESS_REMOTE_WRITE: u32 = 2;
pub const WC_SUCCESS: u32 = 0;

/// `ibv_qp_attr_mask` bits.
pub mod mask {
    pub const STATE: i32 = 1 << 0;
    pub const ACCESS_FLAGS: i32 = 1 << 3;
    pub const PKEY_INDEX: i32 = 1 << 4;
    pub const PORT: i32 = 1 << 5;
    pub const AV: i32 = 1 << 7;
    pub const PATH_MTU: i32 = 1 << 8;
    pub const TIMEOUT: i32 = 1 << 9;
    pub const RETRY_CNT: i32 = 1 << 10;
    pub const RNR_RETRY: i32 = 1 << 11;
    pub const RQ_PSN: i32 = 1 << 12;
    pub const MAX_QP_RD_ATOMIC: i32 = 1 << 13;
    pub const MIN_RNR_TIMER: i32 = 1 << 15;
    pub const SQ_PSN: i32 = 1 << 16;
    pub const MAX_DEST_RD_ATOMIC: i32 = 1 << 17;
    pub const DEST_QPN: i32 = 1 << 20;
}

pub(crate) enum Device {}
pub enum Pd {}

/// The `ops` table follows the device pointer.
#[repr(C)]
pub struct Context {
    device: *mut Device,
    pub(crate) ops: [*const c_void; 32],
}

#[repr(C)]
pub struct Cq {
    pub context: *mut Context,
}

/// The prefix of `ibv_qp` up to its number.
#[repr(C)]
pub struct Qp {
    pub context: *mut Context,
    qp_context: *mut c_void,
    pd: *mut Pd,
    send_cq: *mut Cq,
    recv_cq: *mut Cq,
    srq: *mut c_void,
    handle: u32,
    pub qp_num: u32,
}

#[repr(C)]
pub struct Mr {
    context: *mut Context,
    pd: *mut Pd,
    addr: *mut c_void,
    length: usize,
    handle: u32,
    pub lkey: u32,
    pub rkey: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Sge {
    pub addr: u64,
    pub length: u32,
    pub lkey: u32,
}

/// `ibv_send_wr` with the RDMA write arm of its union.
#[repr(C)]
pub struct SendWr {
    wr_id: u64,
    pub next: *mut SendWr,
    sg_list: *mut Sge,
    num_sge: i32,
    opcode: u32,
    pub(crate) send_flags: u32,
    imm: u32,
    pub(crate) remote_addr: u64,
    pub(crate) rkey: u32,
    rest: [u32; 5],
    xrc: u32,
    pad: u32,
    tail: [u64; 6],
}

impl SendWr {
    /// An RDMA write of `sge` to `remote_addr` under `rkey`.
    pub fn write(sge: *mut Sge, remote_addr: u64, rkey: u32, flags: u32) -> Self {
        let mut wr: Self = unsafe { std::mem::zeroed() };
        wr.sg_list = sge;
        wr.num_sge = 1;
        wr.opcode = WR_RDMA_WRITE;
        wr.send_flags = flags;
        wr.remote_addr = remote_addr;
        wr.rkey = rkey;
        wr
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Wc {
    wr_id: u64,
    pub status: u32,
    opcode: u32,
    vendor_err: u32,
    byte_len: u32,
    imm: u32,
    pub(crate) qp_num: u32,
    src_qp: u32,
    wc_flags: u32,
    pkey_index: u16,
    slid: u16,
    sl: u8,
    dlid_path_bits: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QpCap {
    pub max_send_wr: u32,
    pub max_recv_wr: u32,
    pub max_send_sge: u32,
    pub max_recv_sge: u32,
    pub max_inline_data: u32,
}

#[repr(C)]
pub struct QpInitAttr {
    pub qp_context: *mut c_void,
    pub send_cq: *mut Cq,
    pub recv_cq: *mut Cq,
    pub srq: *mut c_void,
    pub cap: QpCap,
    pub qp_type: u32,
    pub sq_sig_all: i32,
}

#[repr(C, align(8))]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct Gid(pub [u8; 16]);

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct GlobalRoute {
    pub dgid: Gid,
    flow_label: u32,
    pub sgid_index: u8,
    pub hop_limit: u8,
    traffic_class: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct AhAttr {
    pub grh: GlobalRoute,
    pub dlid: u16,
    pub sl: u8,
    pub src_path_bits: u8,
    pub static_rate: u8,
    pub is_global: u8,
    pub port_num: u8,
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct QpAttr {
    pub qp_state: u32,
    pub cur_qp_state: u32,
    pub path_mtu: u32,
    pub path_mig_state: u32,
    pub qkey: u32,
    pub rq_psn: u32,
    pub sq_psn: u32,
    pub dest_qp_num: u32,
    pub qp_access_flags: u32,
    pub cap: QpCap,
    pub ah_attr: AhAttr,
    pub alt_ah_attr: AhAttr,
    pub pkey_index: u16,
    pub alt_pkey_index: u16,
    pub en_sqd_async_notify: u8,
    pub sq_draining: u8,
    pub max_rd_atomic: u8,
    pub max_dest_rd_atomic: u8,
    pub min_rnr_timer: u8,
    pub port_num: u8,
    pub timeout: u8,
    pub retry_cnt: u8,
    pub rnr_retry: u8,
    pub alt_port_num: u8,
    pub alt_timeout: u8,
    pub rate_limit: u32,
}

/// Room for `ibv_port_attr` of any version. The MTU leads.
#[repr(C)]
#[derive(Default)]
pub struct PortAttr {
    state: u32,
    max_mtu: u32,
    pub active_mtu: u32,
    rest: [u32; 13],
}

type GetDeviceList = unsafe extern "C" fn(*mut c_int) -> *mut *mut Device;
type FreeDeviceList = unsafe extern "C" fn(*mut *mut Device);
type GetDeviceName = unsafe extern "C" fn(*mut Device) -> *const c_char;
type OpenDevice = unsafe extern "C" fn(*mut Device) -> *mut Context;
type CloseDevice = unsafe extern "C" fn(*mut Context) -> c_int;
type AllocPd = unsafe extern "C" fn(*mut Context) -> *mut Pd;
type DeallocPd = unsafe extern "C" fn(*mut Pd) -> c_int;
type RegMr = unsafe extern "C" fn(*mut Pd, *mut c_void, usize, c_int) -> *mut Mr;
type DeregMr = unsafe extern "C" fn(*mut Mr) -> c_int;
type CreateCq =
    unsafe extern "C" fn(*mut Context, c_int, *mut c_void, *mut c_void, c_int) -> *mut Cq;
type DestroyCq = unsafe extern "C" fn(*mut Cq) -> c_int;
type CreateQp = unsafe extern "C" fn(*mut Pd, *mut QpInitAttr) -> *mut Qp;
type DestroyQp = unsafe extern "C" fn(*mut Qp) -> c_int;
type ModifyQp = unsafe extern "C" fn(*mut Qp, *mut QpAttr, c_int) -> c_int;
type QueryGid = unsafe extern "C" fn(*mut Context, u8, c_int, *mut Gid) -> c_int;
type QueryPort = unsafe extern "C" fn(*mut Context, u8, *mut PortAttr) -> c_int;
type WcStatusStr = unsafe extern "C" fn(c_int) -> *const c_char;
type PollCq = unsafe extern "C" fn(*mut Cq, c_int, *mut Wc) -> c_int;
type PostSend = unsafe extern "C" fn(*mut Qp, *mut SendWr, *mut *mut SendWr) -> c_int;

crate::dl::library! {
    /// The entry points the transport calls.
    pub struct Lib as "libibverbs" in ["libibverbs.so.1"] {
        get_device_list: GetDeviceList = "ibv_get_device_list",
        free_device_list: FreeDeviceList = "ibv_free_device_list",
        get_device_name: GetDeviceName = "ibv_get_device_name",
        open_device: OpenDevice = "ibv_open_device",
        pub close_device: CloseDevice = "ibv_close_device",
        pub alloc_pd: AllocPd = "ibv_alloc_pd",
        pub dealloc_pd: DeallocPd = "ibv_dealloc_pd",
        pub reg_mr: RegMr = "ibv_reg_mr",
        pub dereg_mr: DeregMr = "ibv_dereg_mr",
        pub create_cq: CreateCq = "ibv_create_cq",
        pub destroy_cq: DestroyCq = "ibv_destroy_cq",
        pub create_qp: CreateQp = "ibv_create_qp",
        pub destroy_qp: DestroyQp = "ibv_destroy_qp",
        pub modify_qp: ModifyQp = "ibv_modify_qp",
        pub query_gid: QueryGid = "ibv_query_gid",
        pub query_port: QueryPort = "ibv_query_port",
        wc_status_str: WcStatusStr = "ibv_wc_status_str",
    }
}

static LIB: OnceLock<Result<Lib, String>> = OnceLock::new();

impl Lib {
    /// Loaded once per process.
    pub fn get() -> Result<&'static Lib, String> {
        LIB.get_or_init(Lib::open).as_ref().map_err(Clone::clone)
    }

    /// Opens the device called `name`.
    pub fn device(&self, name: &str) -> Result<*mut Context, String> {
        let mut n: c_int = 0;
        let list = unsafe { (self.get_device_list)(&mut n) };
        if list.is_null() {
            return Err(format!("no RDMA devices: {}", io::Error::last_os_error()));
        }
        let mut found = std::ptr::null_mut();
        for i in 0..n as usize {
            let device = unsafe { *list.add(i) };
            if message(unsafe { (self.get_device_name)(device) }) == name {
                found = unsafe { (self.open_device)(device) };
            }
        }
        unsafe { (self.free_device_list)(list) };
        match found.is_null() {
            true => Err(format!("no RDMA device {name}")),
            false => Ok(found),
        }
    }

    /// `ibv_poll_cq` through the ops table.
    ///
    /// # Safety
    /// `cq` must be live.
    pub unsafe fn poll_cq(&self, cq: *mut Cq, wc: &mut [Wc]) -> c_int {
        let ops = unsafe { &(*(*cq).context).ops };
        let poll: PollCq = unsafe { std::mem::transmute(ops[11]) };
        unsafe { poll(cq, wc.len() as c_int, wc.as_mut_ptr()) }
    }

    /// # Safety
    /// `qp` and each linked `wr` must be live.
    pub unsafe fn post_send(&self, qp: *mut Qp, wr: &mut SendWr) -> c_int {
        let ops = unsafe { &(*(*qp).context).ops };
        let post: PostSend = unsafe { std::mem::transmute(ops[25]) };
        let mut bad = std::ptr::null_mut();
        unsafe { post(qp, wr, &mut bad) }
    }

    pub fn status(&self, code: u32) -> String {
        message(unsafe { (self.wc_status_str)(code as c_int) })
    }
}
