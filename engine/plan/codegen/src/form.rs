//! Storage forms in kernel text: types, loads and decode helpers.

use crate::ir::Helper;
use std::fmt;
use std::sync::Mutex;

use mem::form::U8;
pub use mem::form::{Bf16, F32, Form, U32};
pub(crate) use scattered_collect::declarative::{gather, scatter};

/// Decodes its elements in kernel text.
pub trait Load: Form {
    /// A generated kernel's type for one element.
    fn ty(&self) -> &'static str;
    /// Kernel text: element `index` of `name` as a float.
    fn load(&self, name: &str, index: &str) -> String;

    /// Whether a load decodes at length, so loops stay rolled.
    fn heavy(&self) -> bool {
        false
    }

    /// Distance between a lane's two wide loads, in elements.
    fn pairs(&self) -> usize {
        0
    }

    /// The helpers `load` calls, each defined once.
    fn header(&self) -> Vec<Helper> {
        Vec::new()
    }

    /// Kernel text: elements `index` to `index + 3` as a `wide()`.
    fn load4(&self, name: &str, index: &str) -> String {
        let each = |i: u32| self.load(name, &format!("({index} + {i})"));
        format!(
            "f_make4({}, {}, {}, {})",
            each(0),
            each(1),
            each(2),
            each(3)
        )
    }

    /// The kernel type of a `load4` result.
    fn wide(&self) -> &'static str {
        "float4"
    }

    /// Kernel text: part `h` of a `wide()`, as a float.
    fn part(&self, wide: &str, h: char) -> String {
        format!("{wide}.{h}")
    }

    /// Whether `input` reads as `form`, when `weight` is this form.
    fn taken(&self, input: usize, weight: usize, form: &dyn Form) -> bool {
        !form.narrow() || (input == weight && form.name() == self.name())
    }

    /// Bytes a matmul reads: the weight, then the rows.
    fn matmul_bytes(&self, m: usize, n: usize, k: usize) -> Vec<f64> {
        vec![self.bytes(n * k) as f64, (4 * m * k) as f64]
    }

    /// `base` for f32, else `base` with the name appended.
    fn suffixed(&self, base: &'static str) -> &'static str {
        match self.name() {
            "f32" => base,
            name => interned(format!("{base}_{name}")),
        }
    }
}

impl fmt::Debug for dyn Load {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl PartialEq for dyn Load {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name()
    }
}

impl Eq for dyn Load {}

/// Kernel names that outlive the forms that made them.
static NAMES: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

fn interned(name: String) -> &'static str {
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    match names.iter().find(|n| ***n == *name) {
        Some(n) => n,
        None => {
            let name: &'static str = Box::leak(name.into_boxed_str());
            names.push(name);
            name
        }
    }
}

gather! {
    #[gather]
    pub static LOADS: scattered_collect::ScatteredSlice<&'static dyn Load>;
}

/// Registers a form's loads. Pins and recipes then name it.
macro_rules! form {
    ($form:expr) => {
        const _: () = {
            $crate::form::scatter! {
                #[scatter($crate::form::LOADS)]
                static FORM: &'static dyn $crate::form::Load = &$form;
            }
        };
    };
}

pub(crate) use form;

/// The registered form called `name`.
pub fn named(name: &str) -> Option<&'static dyn Load> {
    LOADS.iter().copied().find(|f| f.name() == name)
}

/// The form that stores `dtype` as is.
pub fn of(dtype: graph::DType) -> &'static dyn Load {
    let name = format!("{dtype:?}").to_lowercase();
    named(&name).unwrap_or_else(|| panic!("no form stores {dtype:?}"))
}

impl Load for F32 {
    fn ty(&self) -> &'static str {
        "float"
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("{name}[{index}]")
    }
    fn load4(&self, name: &str, index: &str) -> String {
        format!("f_f32x4({name} + {index})")
    }
}

impl Load for Bf16 {
    fn ty(&self) -> &'static str {
        "u16"
    }
    fn load(&self, name: &str, index: &str) -> String {
        format!("f_bits((u32){name}[{index}] << 16)")
    }
    /// Four bf16 as two words, decoded as they are used.
    fn load4(&self, name: &str, index: &str) -> String {
        format!("f_u32x2({name} + {index})")
    }
    fn wide(&self) -> &'static str {
        "uint2"
    }
    /// Two bf16 share each word: low half first.
    fn part(&self, wide: &str, h: char) -> String {
        match h {
            'x' => format!("f_bits({wide}.x << 16)"),
            'y' => format!("f_bits({wide}.x & 0xffff0000u)"),
            'z' => format!("f_bits({wide}.y << 16)"),
            _ => format!("f_bits({wide}.y & 0xffff0000u)"),
        }
    }
}

/// Integer forms: a plain index, under the kernel type `$ty`.
macro_rules! plain {
    ($($form:ident = $ty:literal),*) => {$(
        impl Load for $form {
            fn ty(&self) -> &'static str {
                $ty
            }
            fn load(&self, name: &str, index: &str) -> String {
                format!("{name}[{index}]")
            }
        }
    )*};
}

plain!(U32 = "u32", U8 = "u8");

form!(F32);
form!(Bf16);
form!(U32);

/// The builtin forms, by name.
pub const F32_FORM: &dyn Load = &F32;
pub const BF16_FORM: &dyn Load = &Bf16;
pub const U32_FORM: &dyn Load = &U32;
pub(crate) const U8_FORM: &dyn Load = &U8;
