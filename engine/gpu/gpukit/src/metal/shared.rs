//! A shared-storage Metal buffer, mapped on unified memory.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

pub struct Shared(Retained<ProtocolObject<dyn MTLBuffer>>);

impl Shared {
    /// Zeroed, at least four bytes long.
    pub fn new(device: &ProtocolObject<dyn MTLDevice>, bytes: usize) -> Option<Self> {
        let shared = MTLResourceOptions::StorageModeShared;
        let buffer = device.newBufferWithLength_options(bytes.max(4), shared)?;
        let all = buffer.length();
        unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, all) };
        Some(Self(buffer))
    }

    pub fn buffer(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.0
    }

    pub fn bytes(&self) -> usize {
        self.0.length()
    }

    /// The bytes, in place.
    pub fn mapped(&self) -> &[u8] {
        let at = self.0.contents().as_ptr().cast::<u8>();
        unsafe { std::slice::from_raw_parts(at, self.bytes()) }
    }

    pub fn mapped_mut(&mut self) -> &mut [u8] {
        let at = self.0.contents().as_ptr().cast::<u8>();
        unsafe { std::slice::from_raw_parts_mut(at, self.bytes()) }
    }
}
