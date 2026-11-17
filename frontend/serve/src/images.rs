//! `/v1/images/generations` over `Paint`, as base64 PNGs.

use crate::http::Response;
use crate::wire::{self, Images, Refusal, bad, refuse};
use crate::{Slot, reply};
use serde_json::{Value, json};
use std::time::{SystemTime, UNIX_EPOCH};
use tasks::png;

/// The most pictures one request may ask for.
const MOST: usize = 10;

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Standard base64 with padding.
fn base64(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for group in bytes.chunks(3) {
        let b = [
            group[0],
            *group.get(1).unwrap_or(&0),
            *group.get(2).unwrap_or(&0),
        ];
        let n = u32::from(b[0]) << 16 | u32::from(b[1]) << 8 | u32::from(b[2]);
        for i in 0..4 {
            let pad = i > group.len();
            let c = ALPHABET[(n >> (18 - 6 * i) & 63) as usize];
            out.push(if pad { '=' } else { c as char });
        }
    }
    out
}

/// How many times `size` ("WxH", square) scales a `side` picture.
fn scale(size: Option<&str>, side: usize) -> Result<usize, Refusal> {
    let Some(size) = size else {
        return Ok(256 / side);
    };
    let wrong = || {
        let message = format!("`size` must be square, a multiple of {side} up to 1024");
        bad(message, "size")
    };
    let (w, h) = size.split_once('x').ok_or_else(wrong)?;
    let (w, h): (usize, usize) = (
        w.parse().map_err(|_| wrong())?,
        h.parse().map_err(|_| wrong())?,
    );
    match w == h && w > 0 && w <= 1024 && w.is_multiple_of(side) {
        true => Ok(w / side),
        false => Err(wrong()),
    }
}

/// A seed from the clock, when the request gives none.
fn fresh() -> u32 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |d| d.subsec_nanos())
}

pub(crate) fn generations(slot: &Slot, body: &[u8]) -> Response {
    reply(slot, |m| {
        let paint = m
            .paint
            .as_ref()
            .ok_or_else(|| refuse(404, "the model does not paint pictures", None))?;
        let r: Images = wire::parse(m, body)?;
        let n = r.n.unwrap_or(1);
        if n == 0 || n > MOST {
            return Err(bad(format!("`n` must be 1 to {MOST}"), "n"));
        }
        if r.response_format
            .as_deref()
            .is_some_and(|f| f != "b64_json")
        {
            return Err(bad("only `b64_json` is supported", "response_format"));
        }
        let k = scale(r.size.as_deref(), paint.side())?;
        let seed = r.seed.unwrap_or_else(fresh);
        let images = paint
            .paint(&r.prompt, n, seed)
            .map_err(|e| refuse(500, e.to_string(), None))?;
        let data: Vec<Value> = (images.iter())
            .map(|i| json!({"b64_json": base64(&png::encode(&i.scaled(k)))}))
            .collect();
        Ok(json!({"created": wire::now(), "data": data}))
    })
}
