//! The exlr8 trellis form: K-bit codes in tiles.

use crate::form::{Form, form};
use graph::{DType, E4M3, F16};

/// The codebook multiplier.
pub const MCG: u32 = 0xCBAC_1FED;

/// Header words: `in`, `out`, the sharded side, `1 / in`.
pub const HEAD: usize = 16;

/// Record field 4: entries hold 128 input channels.
pub const INPUT_SIDE: u32 = 2;

/// Channels one entry holds on its sharded side.
const ENTRY: usize = 128;

/// `K`-bit trellis codes of `[out][in]` matrices.
pub struct Trellis<const K: u32> {
    pub name: &'static str,
}

/// The matrix's shape and side, from the header.
struct Shape {
    inp: usize,
    out: usize,
    side: u32,
}

impl Shape {
    fn read(bytes: &[u8]) -> Self {
        let word =
            |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().expect("4 bytes"));
        Self {
            inp: word(0) as usize,
            out: word(1) as usize,
            side: word(2),
        }
    }

    fn write(&self) -> [u8; HEAD] {
        let mut head = [0u8; HEAD];
        let inverse = (1.0 / self.inp as f32).to_bits();
        for (i, v) in [self.inp as u32, self.out as u32, self.side, inverse]
            .iter()
            .enumerate()
        {
            head[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        head
    }

    /// The tile of `T[row][col]`. Later matrices follow at `out`.
    fn tile(&self, row: usize, col: usize) -> usize {
        let (m, col) = (col / self.out, col % self.out);
        m * (self.inp * self.out / 256) + self.local(row, col)
    }

    /// The tile within one matrix.
    fn local(&self, row: usize, col: usize) -> usize {
        // An entry: 128 channels by all of the other side.
        match self.side {
            INPUT_SIDE => {
                let (entry, r) = (row / ENTRY, row % ENTRY / 16);
                entry * (self.out / 2) + 64 * (col / ENTRY) + 8 * r + col % ENTRY / 16
            }
            _ => {
                let (entry, c) = (col / ENTRY, col % ENTRY / 16);
                entry * (self.inp / 2) + 8 * (row / 16) + c
            }
        }
    }
}

/// Stream position of tile element `(r, c)`.
fn position(r: usize, c: usize) -> usize {
    let t = 4 * (c % 8) + (r % 8) / 2;
    let s = 4 * usize::from(c >= 8) + 2 * usize::from(r >= 8) + r % 2;
    8 * t + s
}

/// Bit `b` of a tile: word pairs swapped, high first.
fn bit(tile: &[u8], b: usize) -> u32 {
    let word = (b / 16) ^ 1;
    let w = u16::from_le_bytes([tile[2 * word], tile[2 * word + 1]]);
    u32::from(w >> (15 - b % 16)) & 1
}

/// A state's weight: the mcg hash, then E4M3 rounding.
fn codebook(state: u32) -> f32 {
    let x = (state & 0xFFFF).wrapping_mul(MCG);
    let x = (x & 0x8FFF_8FFF) ^ 0x3B60_3B60;
    let low = f32::from(F16((x & 0xFFFF) as u16));
    let high = f32::from(F16((x >> 16) as u16));
    let sum = f32::from(F16::from(low + high));
    f32::from(E4M3::from(sum.clamp(-448.0, 448.0)))
}

impl<const K: u32> Trellis<K> {
    pub const fn named(name: &'static str) -> Self {
        Self { name }
    }

    /// Bytes one tile of 256 codes takes.
    const TILE: usize = 32 * K as usize;

    /// The code at stream position `j` of a tile.
    fn code(tile: &[u8], j: usize) -> u32 {
        let (k, b) = (K as usize, j * K as usize);
        if 16 % k != 0 {
            return (0..k).fold(0, |v, i| (v << 1) | bit(tile, b + i));
        }
        // Codes that divide a word sit in one, high first.
        let word = (b / 16) ^ 1;
        let w = u16::from_le_bytes([tile[2 * word], tile[2 * word + 1]]);
        u32::from(w >> (16 - k - b % 16)) & ((1 << k) - 1)
    }

    /// Element `index` of a `[out][in]` matrix in `bytes`.
    fn decode(bytes: &[u8], index: usize) -> f32 {
        let shape = Shape::read(bytes);
        let (col, row) = (index / shape.inp, index % shape.inp);
        let at = HEAD + shape.tile(row, col) * Self::TILE;
        let tile = &bytes[at..at + Self::TILE];
        let j = position(row % 16, col % 16);
        // A state: the 16 bits ending at its value, wrapping.
        let mut state = 0;
        for t in 0..16usize.div_ceil(K as usize) {
            state |= Self::code(tile, (j + 256 - t) % 256) << (t * K as usize);
        }
        codebook(state)
    }
}

impl<const K: u32> Form for Trellis<K> {
    fn name(&self) -> &'static str {
        self.name
    }
    fn dtype(&self) -> DType {
        DType::F32
    }
    fn bits(&self) -> u32 {
        K
    }
    fn narrow(&self) -> bool {
        true
    }
    /// A tile, so no float row copies in as is.
    fn block(&self) -> usize {
        256
    }
    fn packs(&self) -> bool {
        false
    }
    fn align(&self) -> usize {
        ENTRY
    }
    fn bytes(&self, elements: usize) -> usize {
        HEAD + elements * K as usize / 8
    }
    fn offset(&self, index: usize) -> usize {
        match index {
            0 => 0,
            _ => HEAD + index * K as usize / 8,
        }
    }
    fn elements(&self, offset: usize) -> usize {
        match offset {
            0 => 0,
            _ => (offset - HEAD) * 8 / K as usize,
        }
    }
    fn element_bytes(&self) -> f64 {
        f64::from(K) / 8.0
    }
    fn unpack(&self, bytes: &[u8], index: usize) -> f32 {
        Self::decode(bytes, index)
    }
    fn pack(&self, _: &[f32]) -> Vec<u8> {
        panic!("{} codes are read from a release, never packed", self.name)
    }
    /// Whole matrices of a set, or whole stored-side entries.
    fn share(
        &self,
        bytes: &[u8],
        [outer, _, inner]: [usize; 3],
        [first, count]: [usize; 2],
    ) -> Option<Vec<u8>> {
        let shape = Shape::read(bytes);
        let matrix = shape.inp * shape.out * K as usize / 8;
        if (outer, inner) == (1, shape.inp * shape.out) {
            let at = HEAD + first * matrix;
            let set = &bytes[at..at + count * matrix];
            return Some(packed(shape.inp, shape.out, shape.side, set));
        }
        let input = shape.side == INPUT_SIDE;
        // Outputs lead, so an output split has one outer part.
        let part = match (outer, inner) {
            (1, i) if i == shape.inp && !input => Shape {
                out: count,
                ..shape
            },
            (o, 1) if o == shape.out && input => Shape {
                inp: count,
                ..shape
            },
            _ => return None,
        };
        if !first.is_multiple_of(ENTRY) || !count.is_multiple_of(ENTRY) {
            return None;
        }
        let across = if input { shape.out } else { shape.inp };
        let entry = across * ENTRY * K as usize / 8;
        let at = HEAD + first / ENTRY * entry;
        Some(packed(
            part.inp,
            part.out,
            part.side,
            &bytes[at..at + count / ENTRY * entry],
        ))
    }
}

/// The header, then the entries as the release stores them.
pub fn packed(inp: usize, out: usize, side: u32, entries: &[u8]) -> Vec<u8> {
    let shape = Shape { inp, out, side };
    let mut bytes = Vec::with_capacity(HEAD + entries.len());
    bytes.extend_from_slice(&shape.write());
    bytes.extend_from_slice(entries);
    bytes
}

/// The form for `k` bits a weight.
pub fn form_for(k: u32) -> Option<&'static dyn Form> {
    match k {
        2 => Some(&TRELLIS_K2),
        3 => Some(&TRELLIS_K3),
        4 => Some(&TRELLIS_K4),
        _ => None,
    }
}

/// A trellis form as a registered constant.
macro_rules! trellis {
    ($(#[$doc:meta])* $name:ident = $text:literal, $k:literal) => {
        $(#[$doc])*
        pub const $name: Trellis<$k> = Trellis::named($text);
        form!($name);
    };
}

trellis! {
    /// exlr8 at two bits a weight.
    TRELLIS_K2 = "trellis_k2", 2
}

trellis! {
    /// exlr8 at three bits a weight.
    TRELLIS_K3 = "trellis_k3", 3
}

trellis! {
    /// exlr8 at four bits a weight.
    TRELLIS_K4 = "trellis_k4", 4
}
