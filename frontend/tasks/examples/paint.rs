//! Writes sprites for a prompt to a PNG file.

use std::time::Instant;
use tasks::paint::{Options, Painter};

/// Each sprite pixel becomes this many.
const SCALE: usize = 8;
const N: usize = 4;

fn crc(bytes: &[u8]) -> u32 {
    let mut c = !0u32;
    for &b in bytes {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                0xedb8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
    }
    !c
}

/// zlib with stored blocks: no compression.
fn zlib(raw: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = raw.chunks(65_535).collect();
    for (i, b) in blocks.iter().enumerate() {
        out.push(u8::from(i + 1 == blocks.len()));
        let len = b.len() as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(b);
    }
    let (mut a, mut b) = (1u32, 0u32);
    for &x in raw {
        a = (a + x as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    out.extend_from_slice(&((b << 16) | a).to_be_bytes());
    out
}

fn png(width: usize, height: usize, rgba: &[u8]) -> Vec<u8> {
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let body = [kind, data].concat();
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc(&body).to_be_bytes());
    };
    let mut head = Vec::new();
    head.extend_from_slice(&(width as u32).to_be_bytes());
    head.extend_from_slice(&(height as u32).to_be_bytes());
    head.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &head);
    let mut raw = Vec::with_capacity(height * (width * 4 + 1));
    for row in rgba.chunks(width * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    chunk(b"IDAT", &zlib(&raw));
    chunk(b"IEND", &[]);
    out
}

fn main() -> Result<(), tasks::Error> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: paint <clip dir> <sprite.safetensors> <out.png> <seed|greedy> <prompt>";
    let [_, clip, weights, out, mode, prompt @ ..] = args.as_slice() else {
        return Err(usage.into());
    };
    let greedy = mode == "greedy";
    let seed = if greedy {
        0
    } else {
        mode.parse().map_err(|_| usage)?
    };
    let n = if greedy { 1 } else { N };
    let started = Instant::now();
    let mut painter = Painter::open(
        clip.as_ref(),
        weights.as_ref(),
        &frontend_settings::read(),
        n as u64,
    )?;
    eprintln!("{}: opened in {:?}", painter.device(), started.elapsed());
    let started = Instant::now();
    let options = Options {
        greedy,
        seed,
        ..Options::default()
    };
    let sprites = painter.paint(&prompt.join(" "), n, &options)?;
    eprintln!("{n} sprites in {:?}", started.elapsed());
    let side = 16 * SCALE;
    let width = side * n;
    let mut rgba = vec![0u8; width * side * 4];
    for (s, tokens) in sprites.iter().enumerate() {
        let px = painter.rgba(tokens);
        for y in 0..side {
            for x in 0..side {
                let from = ((y / SCALE) * 16 + x / SCALE) * 4;
                let to = (y * width + s * side + x) * 4;
                rgba[to..to + 4].copy_from_slice(&px[from..from + 4]);
            }
        }
    }
    if greedy {
        println!("{:?}", sprites[0]);
    }
    std::fs::write(out, png(width, side, &rgba))?;
    Ok(())
}
