//! PNG files without compression, for small pictures.

use crate::api::Image;

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

/// zlib with stored blocks.
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

/// An 8-bit RGBA PNG of `image`.
pub fn encode(image: &Image) -> Vec<u8> {
    let mut out = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let body = [kind, data].concat();
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc(&body).to_be_bytes());
    };
    let mut head = Vec::new();
    head.extend_from_slice(&(image.width as u32).to_be_bytes());
    head.extend_from_slice(&(image.height as u32).to_be_bytes());
    head.extend_from_slice(&[8, 6, 0, 0, 0]);
    chunk(b"IHDR", &head);
    let mut raw = Vec::with_capacity(image.height * (image.width * 4 + 1));
    for row in image.rgba.chunks(image.width * 4) {
        raw.push(0);
        raw.extend_from_slice(row);
    }
    chunk(b"IDAT", &zlib(&raw));
    chunk(b"IEND", &[]);
    out
}
