/// Errors from device setup, shader compilation, and dispatch.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Device, queue, buffer, or encoder creation failed.
    #[error("gpu error: {0}")]
    Gpu(&'static str),
    /// Shader or pipeline compilation failure, with the compiler message.
    #[error("shader compile failed: {0}")]
    Compile(String),
    /// Pipeline archive open or persistence failed.
    #[error("pipeline cache: {0}")]
    Cache(String),
    /// CUDA driver, module, or launch failure, with the driver message.
    #[error("cuda: {0}")]
    Cuda(String),
    /// HIP runtime, module, or launch failure, with the runtime message.
    #[error("rocm: {0}")]
    Rocm(String),
    /// Malformed input data (a shape/size mismatch in a payload).
    #[error("format error: {0}")]
    Format(&'static str),
    /// A Metal device, buffer or dispatch failure.
    #[error("metal: {0}")]
    Metal(String),
}
