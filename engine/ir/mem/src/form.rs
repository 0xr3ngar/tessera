//! Storage forms: how buffers hold elements, and how bytes decode.

use graph::{Bf16 as Half, DType};
use std::fmt;

use scattered_collect::declarative::gather;

/// Packs and unpacks its elements on the host.
pub trait Form: Sync {
    fn name(&self) -> &'static str;
    /// The element type, for moving and converting bytes.
    fn dtype(&self) -> DType;
    /// Significant bits an element keeps.
    fn bits(&self) -> u32;
    /// Element `index` of `bytes`, as a float.
    fn unpack(&self, bytes: &[u8], index: usize) -> f32;
    /// `data` as the bytes of this form.
    fn pack(&self, data: &[f32]) -> Vec<u8>;

    /// Whether floats can become this form. Codes cannot.
    fn packs(&self) -> bool {
        true
    }

    /// Elements that share one scale, decoded together.
    fn block(&self) -> usize {
        1
    }

    /// Elements a row's length must be a multiple of.
    fn align(&self) -> usize {
        self.block()
    }

    /// `data` as this form keeps it.
    fn roundtrip(&self, data: &[f32]) -> Vec<f32> {
        let packed = self.pack(data);
        (0..data.len()).map(|i| self.unpack(&packed, i)).collect()
    }

    /// Bytes one element takes, averaged over a block.
    fn element_bytes(&self) -> f64 {
        self.dtype().bytes() as f64
    }

    /// The byte at which element `index` starts.
    fn offset(&self, index: usize) -> usize {
        index * self.dtype().bytes()
    }

    /// The element starting at byte `offset`.
    fn elements(&self, offset: usize) -> usize {
        offset / self.dtype().bytes()
    }

    /// Bytes `elements` take.
    fn bytes(&self, elements: usize) -> usize {
        elements * self.dtype().bytes()
    }

    /// `x` against the elements from `at`, one at a time.
    fn dot(&self, x: &[f32], bytes: &[u8], at: usize) -> f32 {
        let mut sum = 0.0;
        for (i, xi) in x.iter().enumerate() {
            sum += xi * self.unpack(bytes, at + i);
        }
        sum
    }

    /// Whether a float form needs widening before an f32 read.
    fn narrow(&self) -> bool {
        self.dtype().is_float() && self.dtype() != DType::F32
    }
}

impl fmt::Debug for dyn Form {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl PartialEq for dyn Form {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for dyn Form {}

gather! {
    #[gather]
    pub static FORMS: scattered_collect::ScatteredSlice<&'static dyn Form>;
}

/// Registers a form, so files name it.
macro_rules! form {
    ($form:expr) => {
        const _: () = {
            scattered_collect::declarative::scatter! {
                #[scatter($crate::form::FORMS)]
                static FORM: &'static dyn $crate::form::Form = &$form;
            }
        };
    };
}

pub(crate) use form;

/// The registered form called `name`.
pub fn named(name: &str) -> Option<&'static dyn Form> {
    FORMS.iter().copied().find(|f| f.name() == name)
}

/// Plain f32.
pub struct F32;
/// bf16, widened to f32 on read.
pub struct Bf16;
/// Unsigned 32-bit ids.
pub struct U32;
/// Raw bytes, as a decode helper walks them.
pub struct U8;

/// The methods naming a form and its element.
macro_rules! element {
    ($name:literal, $dtype:ident, $bits:literal) => {
        fn name(&self) -> &'static str {
            $name
        }
        fn dtype(&self) -> DType {
            DType::$dtype
        }
        fn bits(&self) -> u32 {
            $bits
        }
    };
}

fn word(bytes: &[u8], index: usize) -> u32 {
    let at = index * 4;
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

impl Form for F32 {
    element!("f32", F32, 24);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        f32::from_bits(word(bytes, index))
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|x| x.to_le_bytes()).collect()
    }
}

impl Form for Bf16 {
    element!("bf16", Bf16, 8);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        let at = index * 2;
        Half(u16::from_le_bytes([bytes[at], bytes[at + 1]])).into()
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter()
            .flat_map(|x| Half::from(*x).0.to_le_bytes())
            .collect()
    }
}

impl Form for U32 {
    element!("u32", U32, 32);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        word(bytes, index) as f32
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter()
            .flat_map(|x| (*x as u32).to_le_bytes())
            .collect()
    }
}

impl Form for U8 {
    element!("u8", U8, 8);
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        f32::from(bytes[index])
    }
    fn pack(&self, data: &[f32]) -> Vec<u8> {
        data.iter().map(|x| *x as u8).collect()
    }
}

form!(F32);
form!(Bf16);
form!(U32);
