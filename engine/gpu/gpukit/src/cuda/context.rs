//! The CUDA driver behind the shared context.

use std::cell::Cell;
use std::ffi::{CStr, c_char, c_int};

use super::driver::{
    self, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH,
    CU_DEVICE_ATTRIBUTE_INTEGRATED, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION, CU_STREAM_CAPTURE_MODE_RELAXED,
    CU_STREAM_NON_BLOCKING, CUcontext, CUdevice, Driver, LaunchAttribute, LaunchConfig,
};
use crate::api::{self, Api, Call, Handle};
use crate::backend::Backend;
use crate::platform::{self, Asked, Platform};
use crate::{Error, Limits};

pub use crate::api::ContextConfig;

/// A CUDA context plus a cache of loaded modules.
pub type Context = api::Context<Driver>;
pub type Module = api::Module<Driver>;
pub type Kernel = api::Kernel<Driver>;
pub type Graph = api::Graph<Driver>;

/// The driver context and what the device reports.
pub struct Own {
    raw: CUcontext,
    compute_capability: (i32, i32),
    limits: Limits,
    platform: Platform,
}

/// The driver's version, `13.0` for 13000.
fn driver_version(driver: &Driver) -> Result<String, Error> {
    let mut v: c_int = 0;
    driver.check(
        unsafe { (driver.cu_driver_get_version)(&mut v) },
        "cuDriverGetVersion",
    )?;
    Ok(format!("{}.{}", v / 1000, (v % 1000) / 10))
}

fn device_attr(driver: &Driver, device: CUdevice, attr: c_int) -> Result<i32, Error> {
    let mut value: c_int = 0;
    driver.check(
        unsafe { (driver.cu_device_get_attribute)(&mut value, attr, device) },
        "cuDeviceGetAttribute",
    )?;
    Ok(value)
}

impl Context {
    /// Initialize the driver and open a context on one device.
    pub fn new(config: ContextConfig) -> Result<Self, Error> {
        let driver = driver::driver()?;
        driver.check(unsafe { (driver.cu_init)(0) }, "cuInit")?;

        let mut count: c_int = 0;
        driver.check(
            unsafe { (driver.cu_device_get_count)(&mut count) },
            "cuDeviceGetCount",
        )?;
        if config.device_ordinal >= count as usize {
            return Err(Error::Cuda(format!(
                "device ordinal {} out of range ({count} visible CUDA devices)",
                config.device_ordinal
            )));
        }

        let mut device: CUdevice = 0;
        driver.check(
            unsafe { (driver.cu_device_get)(&mut device, config.device_ordinal as c_int) },
            "cuDeviceGet",
        )?;

        let mut raw: CUcontext = std::ptr::null_mut();
        driver.check(
            unsafe { (driver.cu_ctx_create)(&mut raw, 0, device) },
            "cuCtxCreate",
        )?;

        let mut name_buf = [0 as c_char; 256];
        driver.check(
            unsafe { (driver.cu_device_get_name)(name_buf.as_mut_ptr(), 256, device) },
            "cuDeviceGetName",
        )?;
        let name = unsafe { CStr::from_ptr(name_buf.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let major = device_attr(driver, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = device_attr(driver, device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let limits = Limits {
            lanes: device_attr(driver, device, CU_DEVICE_ATTRIBUTE_WARP_SIZE)? as usize,
            group_bytes: device_attr(
                driver,
                device,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
            )? as usize,
            group_threads: device_attr(driver, device, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)?
                as usize,
            compiler: true,
            pdl: false,
        };
        let attr = |a| device_attr(driver, device, a);
        let platform = Platform::new(Asked {
            backend: "cuda",
            arch: format!("sm_{major}{minor}"),
            name: name.clone(),
            driver: driver_version(driver)?,
            limits,
            units: attr(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)? as usize,
            bandwidth: platform::bandwidth(
                attr(CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)?,
                attr(CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)?,
            ),
            integrated: attr(CU_DEVICE_ATTRIBUTE_INTEGRATED)? != 0,
        });
        let own = Own {
            raw,
            compute_capability: (major, minor),
            // The table says whether launches may overlap.
            limits: platform.limits,
            platform,
        };
        Ok(Self::made(driver, own, device, name))
    }

    /// Lanes, group memory and group size, from the device.
    pub fn limits(&self) -> Limits {
        self.own().limits
    }

    /// The device as asked, with its table row.
    pub fn platform(&self) -> Platform {
        self.own().platform.clone()
    }

    /// (major, minor), e.g. (12, 1) for sm_121.
    pub fn compute_capability(&self) -> (i32, i32) {
        self.own().compute_capability
    }

    pub fn driver(&self) -> &'static Driver {
        self.api()
    }
}

thread_local! {
    /// This thread's capture stream, or null outside a capture.
    static CAPTURING: Cell<Handle> = const { Cell::new(std::ptr::null_mut()) };
    /// Whether this thread's next launches may overlap the last.
    static PROGRAMMATIC: Cell<bool> = const { Cell::new(false) };
}

impl Driver {
    /// A launch that may overlap the kernel before it.
    fn overlap(&self, call: &Call) -> Result<(), Error> {
        let mut attr = LaunchAttribute {
            id: CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION,
            pad: 0,
            value: [1, 0, 0, 0, 0, 0, 0, 0],
        };
        let config = LaunchConfig {
            grid: call.grid,
            block: call.block,
            shared_bytes: call.shared_bytes,
            pad: 0,
            stream: call.stream,
            attrs: &mut attr,
            attr_count: 1,
            pad2: 0,
        };
        let null = std::ptr::null_mut();
        let code = unsafe { (self.cu_launch_kernel_ex)(&config, call.kernel, call.params, null) };
        self.check(code, "cuLaunchKernelEx")
    }
}

impl Api for Driver {
    type Own = Own;

    const BACKEND: Backend = Backend::Cuda;

    api::shared_api! {
        context: Context,
        error: Error::Cuda,
        check: Driver::check,
        synchronize: (cu_stream_synchronize, "cuStreamSynchronize"),
        begin_capture: (
            cu_stream_begin_capture, "cuStreamBeginCapture", CU_STREAM_CAPTURE_MODE_RELAXED
        ),
        end_capture: (cu_stream_end_capture, "cuStreamEndCapture"),
        replay: (cu_graph_launch, "cuGraphLaunch"),
        unload: (cu_module_unload, cu_graph_exec_destroy, cu_graph_destroy, cu_stream_destroy),
        load: (cu_module_load_data, "cuModuleLoadData"),
        function: (cu_module_get_function, "cuModuleGetFunction"),
        alloc: (cu_mem_alloc, "cuMemAlloc"),
        free: cu_mem_free,
        host_alloc: (
            cu_mem_host_alloc, "cuMemHostAlloc", driver::CU_MEMHOSTALLOC_DEVICEMAP,
            cu_mem_host_get_device_pointer, "cuMemHostGetDevicePointer"
        ),
        host_free: cu_mem_free_host,
        zero: (cu_memset_d8, "cuMemsetD8"),
    }

    fn programmatic(on: bool) {
        PROGRAMMATIC.with(|p| p.set(on));
    }

    fn close(&self, own: &Own) {
        unsafe { (self.cu_ctx_destroy)(own.raw) };
    }

    fn set_current(&self, own: &Own, _: c_int) -> Result<(), Error> {
        let code = unsafe { (self.cu_ctx_set_current)(own.raw) };
        self.check(code, "cuCtxSetCurrent")
    }

    fn stream(&self) -> Result<Handle, Error> {
        self.out("cuStreamCreate", |s| unsafe {
            (self.cu_stream_create)(s, CU_STREAM_NON_BLOCKING)
        })
    }

    fn instantiate(&self, graph: Handle) -> Result<Handle, Error> {
        self.out("cuGraphInstantiate", |e| unsafe {
            (self.cu_graph_instantiate)(e, graph, 0)
        })
    }

    fn launch(&self, call: &Call) -> Result<(), Error> {
        if PROGRAMMATIC.with(Cell::get) {
            return self.overlap(call);
        }
        let code = unsafe { api::plain(self.cu_launch_kernel, call) };
        self.check(code, "cuLaunchKernel")
    }

    fn upload(&self, ptr: u64, bytes: &[u8]) -> Result<(), Error> {
        let code = unsafe { (self.cu_memcpy_htod)(ptr, bytes.as_ptr().cast(), bytes.len()) };
        self.check(code, "cuMemcpyHtoD")
    }

    fn download(&self, ptr: u64, bytes: &mut [u8]) -> Result<(), Error> {
        let code = unsafe { (self.cu_memcpy_dtoh)(bytes.as_mut_ptr().cast(), ptr, bytes.len()) };
        self.check(code, "cuMemcpyDtoH")
    }

    fn compile(ctx: &Context, source: &str) -> Result<Vec<u8>, Error> {
        let (major, minor) = ctx.compute_capability();
        let arch = format!("--gpu-architecture=compute_{major}{minor}");
        driver::NVRTC.compile(source, &[&arch, "--std=c++17"])
    }
}
