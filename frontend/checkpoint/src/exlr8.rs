//! exlr8 v2 releases: trellis and fp8 matrices decoded to f32.

use crate::file::{Error, File};
use crate::offer::{Offer, Slice, Stored};
use engine::{E4M3, F16, trellis};
use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Record field 0: how the matrix is stored.
const TRELLIS: i64 = 1;
const INT4: i64 = 2;
const MXFP8: i64 = 3;
const FP8_BLOCK: i64 = 4;
const FP8_ROW: i64 = 5;
const BF16: i64 = 6;
const FP32: i64 = 7;

/// A matrix's record: format, shape, trellis width, sharded side.
struct Record {
    format: i64,
    out: usize,
    inp: usize,
    k: u32,
    side: i64,
}

fn record(f: &File, name: &str) -> Result<Record, Error> {
    let bytes = f.bytes(&format!("{name}.record"))?;
    let field = |i: usize| i64::from_le_bytes(bytes[8 * i..8 * i + 8].try_into().expect("8 bytes"));
    Ok(Record {
        format: field(0),
        out: field(1) as usize,
        inp: field(2) as usize,
        k: field(3) as u32,
        side: field(4),
    })
}

fn halves(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    let bytes = f.bytes(name)?;
    let half = |b: &[u8; 2]| f32::from(F16(u16::from_le_bytes(*b)));
    Ok(bytes.as_chunks::<2>().0.iter().map(half).collect())
}

fn e4m3(bits: u8) -> f32 {
    f32::from(E4M3(bits))
}

/// Trellis matrix `name` packed: a header, then its entries.
fn packed(f: &File, name: &str, r: &Record) -> Result<Vec<u8>, Error> {
    let entries = f.bytes(&format!("{name}.trellis"))?;
    Ok(trellis::packed(r.inp, r.out, r.side as u32, &entries))
}

/// Trellis matrix `name` as f32 `[out][in]`, before its rotations.
fn decoded(f: &File, name: &str, r: &Record) -> Result<Vec<f32>, Error> {
    let form = trellis::form_for(r.k).ok_or_else(|| Error::DType(format!("{name}: K{}", r.k)))?;
    let bytes = packed(f, name, r)?;
    Ok((0..r.inp * r.out).map(|i| form.unpack(&bytes, i)).collect())
}

/// The scaled Walsh-Hadamard transform of 128 values at `stride`.
fn hadamard(x: &mut [f32], start: usize, stride: usize) {
    let mut len = 1;
    while len < 128 {
        for i in (0..128).step_by(2 * len) {
            for j in i..i + len {
                let (a, b) = (x[start + j * stride], x[start + (j + len) * stride]);
                x[start + j * stride] = a + b;
                x[start + (j + len) * stride] = a - b;
            }
        }
        len *= 2;
    }
    let scale = 1.0 / 128f32.sqrt();
    for j in 0..128 {
        x[start + j * stride] *= scale;
    }
}

/// `diag(suh) H T H diag(svh)`, transposed to `[out][in]`.
fn rotate(mut t: Vec<f32>, r: &Record, suh: &[f32], svh: &[f32]) -> Vec<f32> {
    let (inp, out) = (r.inp, r.out);
    for block in 0..inp / 128 {
        for col in 0..out {
            hadamard(&mut t, 128 * block * out + col, out);
        }
    }
    for (row, s) in suh.iter().enumerate() {
        t[row * out..(row + 1) * out]
            .iter_mut()
            .for_each(|v| *v *= s);
    }
    for row in 0..inp {
        for block in 0..out / 128 {
            hadamard(&mut t, row * out + 128 * block, 1);
        }
    }
    let mut w = vec![0.0; inp * out];
    for row in 0..inp {
        for (col, s) in svh.iter().enumerate() {
            w[col * inp + row] = t[row * out + col] * s;
        }
    }
    w
}

/// `diag(suh) H T H diag(svh)` for a `[out][in]` matrix `t`.
pub fn rotated(t: &[f32], inp: usize, out: usize, suh: &[f32], svh: &[f32]) -> Vec<f32> {
    let r = Record {
        format: TRELLIS,
        out,
        inp,
        k: 0,
        side: 0,
    };
    rotate(transpose(t, out, inp), &r, suh, svh)
}

/// Matrix `name` as f32 `[out][in]`, decoded by its record.
fn matrix(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    let r = record(f, name)?;
    match r.format {
        TRELLIS => {
            let suh = halves(f, &format!("{name}.suh"))?;
            let svh = halves(f, &format!("{name}.svh"))?;
            let t = transpose(&decoded(f, name, &r)?, r.out, r.inp);
            Ok(rotate(t, &r, &suh, &svh))
        }
        other => plain(f, name, &r, other),
    }
}

/// Packed codes of `name`, or `rows` as `(first, count)`.
fn codes_packed(f: &File, name: &str, rows: Option<(usize, usize)>) -> Result<Vec<u8>, Error> {
    let r = record(f, name)?;
    if r.format != TRELLIS {
        let what = format!("{name} is in exlr8 format {}, not trellis", r.format);
        return Err(Error::DType(what));
    }
    let Some((first, count)) = rows else {
        return packed(f, name, &r);
    };
    // Output-side entries are 128 rows: a range is whole entries.
    let entry = r.inp * 128 * r.k as usize / 8;
    if r.side == trellis::INPUT_SIDE as i64 || first % 128 != 0 || count % 128 != 0 {
        return Err(Error::DType(format!(
            "{name}: rows {first}.. are not whole entries"
        )));
    }
    let entries = f.bytes(&format!("{name}.trellis"))?;
    let at = first / 128 * entry;
    let slice = &entries[at..at + count / 128 * entry];
    Ok(trellis::packed(r.inp, count, r.side as u32, slice))
}

/// Matrix `name` as f32 `[out][in]` before its rotations, if any.
pub fn codes(f: &File, name: &str) -> Result<Vec<f32>, Error> {
    let r = record(f, name)?;
    match r.format {
        TRELLIS => decoded(f, name, &r),
        other => plain(f, name, &r, other),
    }
}

/// The transpose of `t`, `rows` by `cols`.
fn transpose(t: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut w = vec![0.0; rows * cols];
    for row in 0..rows {
        for col in 0..cols {
            w[col * rows + row] = t[row * cols + col];
        }
    }
    w
}

/// A matrix in a format other than trellis, as `[out][in]`.
fn plain(f: &File, name: &str, r: &Record, format: i64) -> Result<Vec<f32>, Error> {
    let inp = r.inp;
    let fp8 = || f.bytes(&format!("{name}.fp8"));
    match format {
        FP8_ROW => {
            let scale = f.floats(&format!("{name}.scale"))?;
            let row = |(i, c): (usize, &u8)| e4m3(*c) * scale[i / inp];
            Ok(fp8()?.iter().enumerate().map(row).collect())
        }
        // One power of two per 32 inputs.
        MXFP8 => {
            let exp = f.bytes(&format!("{name}.e8m0"))?;
            let at = |i: usize| exp[i / inp * inp.div_ceil(32) + i % inp / 32];
            let value = |(i, c): (usize, &u8)| e4m3(*c) * 2f32.powi(i32::from(at(i)) - 127);
            Ok(fp8()?.iter().enumerate().map(value).collect())
        }
        // One scale per 128 by 128 block.
        FP8_BLOCK => {
            let scale = f.floats(&format!("{name}.scale"))?;
            let at = |i: usize| scale[i / inp / 128 * inp.div_ceil(128) + i % inp / 128];
            let value = |(i, c): (usize, &u8)| e4m3(*c) * at(i);
            Ok(fp8()?.iter().enumerate().map(value).collect())
        }
        INT4 => int4(f, name, r),
        BF16 => f.floats(&format!("{name}.bf16")),
        FP32 => f.floats(&format!("{name}.fp32")),
        other => Err(Error::DType(format!("{name} is in exlr8 format {other}"))),
    }
}

/// Nibbles as `(n - 8) * s`, one scale per 32 inputs.
fn int4(f: &File, name: &str, r: &Record) -> Result<Vec<f32>, Error> {
    let (inp, outp) = (r.inp, r.out);
    let q4 = f.bytes(&format!("{name}.q4"))?;
    let s4 = halves(f, &format!("{name}.s4"))?;
    let mut w = Vec::with_capacity(inp * outp);
    for o in 0..outp {
        for i in 0..inp {
            // Input-side shards lie in blocks of 128 inputs.
            let (byte, scale) = match r.side == trellis::INPUT_SIDE as i64 {
                true => {
                    let (g, j) = (i / 128, i % 128);
                    ((g * outp + o) * 64 + j / 2, (g * outp + o) * 4 + j / 32)
                }
                false => (o * inp / 2 + i / 2, o * inp / 32 + i / 32),
            };
            let n = (q4[byte] >> (4 * (i % 2))) & 0xF;
            w.push((f32::from(n) - 8.0) * s4[scale]);
        }
    }
    Ok(w)
}

/// An exlr8 release, read by HF tensor name.
pub struct Release {
    /// A directory, or a URL prefix ending in `/`.
    at: Where,
    files: RefCell<HashMap<String, File>>,
    /// GLM's layout: `dense/` files, layer names without their prefix.
    glm: bool,
}

/// GLM's vocabulary file, which marks its layout.
const GLM_VOCAB: &str = "dense/exlr8-vocab";

/// One file, and how its names become HF names.
struct Shard {
    file: String,
    prefix: String,
    /// Names kept as they are.
    bare: &'static [&'static str],
}

impl Shard {
    fn new(file: String, prefix: String) -> Self {
        Self {
            file,
            prefix,
            bare: &[],
        }
    }

    fn name(&self, local: &str) -> String {
        match self.bare.iter().any(|b| local.starts_with(b)) {
            true => local.into(),
            false => format!("{}{local}", self.prefix),
        }
    }
}

enum Where {
    Dir(PathBuf),
    Url(String),
}

impl Release {
    pub fn open(dir: &Path) -> Self {
        Self::new(Where::Dir(dir.to_path_buf()))
    }

    /// Streams `owner/name` on Hugging Face at `revision`.
    pub fn remote(repo: &str, revision: &str) -> Self {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/");
        Self::new(Where::Url(url))
    }

    fn new(at: Where) -> Self {
        let mut release = Self {
            at,
            files: RefCell::new(HashMap::new()),
            glm: false,
        };
        release.glm = release.with(GLM_VOCAB, |_| Ok(())).is_ok();
        release
    }

    /// The file holding `name`, and the name inside it.
    fn place(&self, name: &str) -> (String, String) {
        if self.glm {
            return glm_place(name);
        }
        if let Some(rest) = name.strip_prefix("language_model.layers.") {
            let (layer, local) = rest.split_once('.').expect("a layer index");
            let layer: usize = layer.parse().expect("a layer index");
            return (format!("exlr8-dense-text-{layer:03}"), local.into());
        }
        match name {
            "language_model.embed_tokens.weight" => {
                ("exlr8-vocab".into(), "embed_tokens.weight".into())
            }
            _ => ("exlr8-dense-global-000".into(), name.into()),
        }
    }

    fn with<T>(
        &self,
        file: &str,
        read: impl FnOnce(&File) -> Result<T, Error>,
    ) -> Result<T, Error> {
        let mut files = self.files.borrow_mut();
        if !files.contains_key(file) {
            let name = format!("{file}.safetensors");
            let opened = match &self.at {
                Where::Dir(dir) => File::open(&dir.join(name))?,
                Where::Url(base) => File::remote(&format!("{base}{name}"))?,
            };
            files.insert(file.into(), opened);
        }
        read(&files[file])
    }

    /// The packed codes of `M.codes`, or `rows` as `(first, count)`.
    pub fn packed(&self, name: &str, rows: Option<(usize, usize)>) -> Result<Vec<u8>, Error> {
        let (file, local) = self.place(name);
        let matrix_name = local.strip_suffix(".codes").unwrap_or(&local).to_string();
        self.with(&file, |f| codes_packed(f, &matrix_name, rows))
    }

    /// Tensor `name` as f32. `M.codes`, `M.suh`, `M.svh`: unrotated parts.
    pub fn read(&self, name: &str) -> Result<Vec<f32>, Error> {
        let (file, local) = self.place(name);
        self.with(&file, |f| {
            let (matrix_name, part) = match local.rsplit_once('.') {
                Some((m, part)) if ["weight", "codes", "suh", "svh"].contains(&part) => (m, part),
                _ => (local.as_str(), "weight"),
            };
            if f.shape(&format!("{matrix_name}.record")).is_none() {
                return f.floats(&local);
            }
            match part {
                "codes" => codes(f, matrix_name),
                "suh" | "svh" => halves(f, &format!("{matrix_name}.{part}")),
                _ => matrix(f, matrix_name),
            }
        })
    }
}

/// GLM: layer files under `dense/`, and one vocabulary file.
fn glm_place(name: &str) -> (String, String) {
    if let Some(rest) = name.strip_prefix("model.layers.") {
        let (layer, local) = rest.split_once('.').expect("a layer index");
        let layer: usize = layer.parse().expect("a layer index");
        return (format!("dense/exlr8-dense-{layer:03}"), local.into());
    }
    let local = name.strip_prefix("model.").unwrap_or(name);
    (GLM_VOCAB.into(), local.into())
}

impl Release {
    /// Every shard, and how its names become HF names.
    fn shards(&self) -> Vec<Shard> {
        let (mut out, layers, prefix) = match self.glm {
            true => {
                let vocab = Shard {
                    bare: &["lm_head"],
                    ..Shard::new(GLM_VOCAB.into(), "model.".into())
                };
                (vec![vocab], "dense/exlr8-dense-", "model.layers.")
            }
            false => (
                vec![
                    Shard::new("exlr8-vocab".into(), "language_model.".into()),
                    Shard::new("exlr8-dense-global-000".into(), String::new()),
                ],
                "exlr8-dense-text-",
                "language_model.layers.",
            ),
        };
        for layer in 0.. {
            let file = format!("{layers}{layer:03}");
            if self.with(&file, |_| Ok(())).is_err() {
                break;
            }
            out.push(Shard::new(file, format!("{prefix}{layer}.")));
        }
        out
    }

    /// A shard's slices: matrix parts and plain tensors.
    fn offered(f: &File, shard: &Shard) -> Vec<Slice> {
        let raw = f.slices();
        let shape = |name: &str| raw.iter().find(|s| s.name == name).map(|s| s.shape.clone());
        let mut out = Vec::new();
        for s in &raw {
            let Some(matrix) = s.name.strip_suffix(".record") else {
                continue;
            };
            let name = |part: &str| shard.name(&format!("{matrix}.{part}"));
            let (Some(suh), Some(svh)) = (
                shape(&format!("{matrix}.suh")),
                shape(&format!("{matrix}.svh")),
            ) else {
                // Other formats decode to the HF weight.
                if let Ok(r) = record(f, matrix) {
                    out.push(Slice {
                        name: name("weight"),
                        shape: vec![r.out, r.inp],
                        stored: Stored::F32,
                    });
                }
                continue;
            };
            let (inp, outp) = (suh[0], svh[0]);
            let Some(entries) = shape(&format!("{matrix}.trellis")) else {
                continue;
            };
            let k = (entries[0] * entries[1] * 8 / (inp * outp)) as u32;
            let Some(form) = trellis::form_for(k) else {
                continue;
            };
            out.push(Slice {
                name: name("weight"),
                shape: vec![outp, inp],
                stored: Stored::F32,
            });
            out.push(Slice {
                name: name("codes"),
                shape: vec![outp, inp],
                stored: Stored::Form(form),
            });
            out.push(Slice {
                name: name("suh"),
                shape: vec![inp],
                stored: Stored::F16,
            });
            out.push(Slice {
                name: name("svh"),
                shape: vec![outp],
                stored: Stored::F16,
            });
        }
        let part = |n: &str| {
            [
                ".record",
                ".checksum",
                ".trellis",
                ".suh",
                ".svh",
                ".fp8",
                ".scale",
                ".q4",
                ".s4",
                ".e8m0",
                ".bf16",
                ".fp32",
            ]
            .iter()
            .any(|p| n.ends_with(p))
                || n.starts_with("__pad")
        };
        for s in raw {
            if !part(&s.name) {
                out.push(Slice {
                    name: shard.name(&s.name),
                    ..s
                });
            }
        }
        out
    }
}

impl Offer for Release {
    fn slices(&self) -> Vec<Slice> {
        let mut out = Vec::new();
        for shard in self.shards() {
            let slices = self.with(&shard.file, |f| Ok(Self::offered(f, &shard)));
            out.extend(slices.unwrap_or_default());
        }
        out
    }

    fn floats(&self, name: &str) -> Result<Vec<f32>, Error> {
        self.read(name)
    }

    fn packed(&self, name: &str) -> Result<Vec<u8>, Error> {
        Release::packed(self, name, None)
    }

    fn packed_rows(&self, name: &str, rows: (usize, usize)) -> Result<Vec<u8>, Error> {
        Release::packed(self, name, Some(rows))
    }
}
