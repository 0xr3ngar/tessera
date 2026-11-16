//! The HIP runtime behind the shared context.

use std::cell::Cell;
use std::ffi::{CStr, c_char, c_int};
use std::sync::OnceLock;

use super::hip::{self, Runtime};
use crate::api::{self, Api, Call, Handle};
use crate::backend::Backend;
use crate::device::Device;
use crate::platform::{self, Asked, Platform};
use crate::{Error, Limits};

pub use crate::api::ContextConfig;

/// A HIP device plus a cache of loaded modules.
pub type Context = api::Context<Runtime>;
pub type Module = api::Module<Runtime>;
pub type Kernel = api::Kernel<Runtime>;
pub type Graph = api::Graph<Runtime>;

/// `warpSize`, asked once, and the platform, built once.
pub struct Own {
    lanes: OnceLock<usize>,
    platform: OnceLock<Platform>,
}

impl Context {
    /// Initialize the runtime and select one device.
    pub fn new(config: ContextConfig) -> Result<Self, Error> {
        let runtime = hip::runtime()?;
        runtime.check(unsafe { (runtime.hip_init)(0) }, "hipInit")?;

        let mut count: c_int = 0;
        runtime.check(
            unsafe { (runtime.hip_get_device_count)(&mut count) },
            "hipGetDeviceCount",
        )?;
        if config.device_ordinal >= count as usize {
            return Err(Error::Rocm(format!(
                "device ordinal {} out of range ({count} visible HIP devices)",
                config.device_ordinal
            )));
        }
        let device = config.device_ordinal as c_int;

        let mut name_buf = [0 as c_char; 256];
        runtime.check(
            unsafe { (runtime.hip_device_get_name)(name_buf.as_mut_ptr(), 256, device) },
            "hipDeviceGetName",
        )?;
        let mut name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();
        // APU GPU agents often report an empty marketing name.
        if name.is_empty() {
            name = format!("HIP device {device}");
        }

        let own = Own {
            lanes: OnceLock::new(),
            platform: OnceLock::new(),
        };
        let ctx = Self::made(runtime, own, device, name);
        ctx.set_current()?;
        Ok(ctx)
    }

    /// One `hipDeviceGetAttribute`.
    fn attr(&self, which: c_int) -> Result<i32, Error> {
        let mut v: c_int = 0;
        let r = self.runtime();
        r.check(
            unsafe { (r.hip_device_get_attribute)(&mut v, which, self.device()) },
            "hipDeviceGetAttribute",
        )?;
        Ok(v)
    }

    /// The device as asked, with its row. Zero when unasked.
    pub fn platform(&self) -> Platform {
        self.own()
            .platform
            .get_or_init(|| {
                let attr = |a| self.attr(a).unwrap_or(0);
                let mut v: c_int = 0;
                unsafe { (self.runtime().hip_runtime_get_version)(&mut v) };
                let (major, minor) = (
                    attr(hip::ATTR_COMPUTE_CAPABILITY_MAJOR),
                    attr(hip::ATTR_COMPUTE_CAPABILITY_MINOR),
                );
                Platform::new(Asked {
                    backend: "rocm",
                    arch: format!("gfx{major}_{minor}"),
                    name: self.name().to_string(),
                    driver: format!("{}.{}", v / 10_000_000, (v / 100_000) % 100),
                    limits: self.limits(),
                    units: attr(hip::ATTR_MULTIPROCESSOR_COUNT) as usize,
                    bandwidth: platform::bandwidth(
                        attr(hip::ATTR_MEMORY_CLOCK_RATE),
                        attr(hip::ATTR_MEMORY_BUS_WIDTH),
                    ),
                    integrated: attr(hip::ATTR_INTEGRATED) != 0,
                })
            })
            .clone()
    }

    /// Lanes come from the device. The rest are AMD constants.
    pub fn limits(&self) -> Limits {
        let lanes = *self
            .own()
            .lanes
            .get_or_init(|| self.probe_lanes().unwrap_or(64));
        Limits {
            lanes,
            group_bytes: 64 * 1024,
            group_threads: 1024,
            compiler: true,
            pdl: false,
        }
    }

    /// A kernel's `warpSize`: 32 on RDNA, 64 on CDNA.
    fn probe_lanes(&self) -> Result<usize, Error> {
        let source = "extern \"C\" __global__ void tessera_lanes(unsigned *out) {
    if (threadIdx.x == 0) {
        out[0] = warpSize;
    }
}
";
        let kernel = self.compile(source, "tessera_lanes")?;
        let out = self.alloc(4)?;
        let mut args = crate::launch::KernelArgs::new();
        args.device_ptr(out);
        let launched = crate::launch::launch_1d(self, &kernel, 64, &mut args).and_then(|_| {
            self.synchronize()?;
            let mut bytes = [0u8; 4];
            self.download(out, &mut bytes)?;
            Ok(u32::from_ne_bytes(bytes) as usize)
        });
        self.free(out);
        launched
    }

    pub fn runtime(&self) -> &'static Runtime {
        self.api()
    }
}

thread_local! {
    /// This thread's capture stream, or null outside a capture.
    static CAPTURING: Cell<Handle> = const { Cell::new(std::ptr::null_mut()) };
}

impl Api for Runtime {
    type Own = Own;

    const BACKEND: Backend = Backend::Rocm;

    api::shared_api! {
        context: Context,
        error: Error::Rocm,
        check: Runtime::check,
        synchronize: (hip_stream_synchronize, "hipStreamSynchronize"),
        begin_capture: (
            hip_stream_begin_capture, "hipStreamBeginCapture", hip::HIP_STREAM_CAPTURE_MODE_RELAXED
        ),
        end_capture: (hip_stream_end_capture, "hipStreamEndCapture"),
        replay: (hip_graph_launch, "hipGraphLaunch"),
        unload: (hip_module_unload, hip_graph_exec_destroy, hip_graph_destroy, hip_stream_destroy),
        load: (hip_module_load_data, "hipModuleLoadData"),
        function: (hip_module_get_function, "hipModuleGetFunction"),
        alloc: (hip_malloc, "hipMalloc"),
        free: hip_free,
        host_alloc: (
            hip_host_malloc, "hipHostMalloc", hip::HOST_MALLOC_MAPPED,
            hip_host_get_device_pointer, "hipHostGetDevicePointer"
        ),
        host_free: hip_host_free,
        zero: (hip_memset, "hipMemset"),
    }

    fn set_current(&self, _: &Own, device: c_int) -> Result<(), Error> {
        self.check(unsafe { (self.hip_set_device)(device) }, "hipSetDevice")
    }

    fn stream(&self) -> Result<Handle, Error> {
        let flags = hip::HIP_STREAM_NON_BLOCKING;
        self.out("hipStreamCreateWithFlags", |s| unsafe {
            (self.hip_stream_create)(s, flags)
        })
    }

    fn instantiate(&self, graph: Handle) -> Result<Handle, Error> {
        let (none, nothing) = (std::ptr::null_mut(), std::ptr::null_mut());
        self.out("hipGraphInstantiate", |e| unsafe {
            (self.hip_graph_instantiate)(e, graph, none, nothing, 0)
        })
    }

    fn launch(&self, call: &Call) -> Result<(), Error> {
        let code = unsafe { api::plain(self.hip_module_launch_kernel, call) };
        self.check(code, "hipModuleLaunchKernel")
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        let (dst, src) = (ptr as *mut _, bytes.as_ptr().cast());
        let to = hip::MEMCPY_HOST_TO_DEVICE;
        let code = unsafe { (self.hip_memcpy)(dst, src, bytes.len(), to) };
        self.check(code, "hipMemcpy to device")
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        let (dst, src) = (bytes.as_mut_ptr().cast(), ptr as *const _);
        let to = hip::MEMCPY_DEVICE_TO_HOST;
        let code = unsafe { (self.hip_memcpy)(dst, src, bytes.len(), to) };
        self.check(code, "hipMemcpy to host")
    }

    fn compile(ctx: &Context, source: &str) -> Result<Vec<u8>, Error> {
        ctx.set_current()?;
        // With no --offload-arch, hiprtc targets the current device.
        hip::HIPRTC.compile(source, &["-std=c++17"])
    }
}
