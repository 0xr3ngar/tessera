//! Pictures for `/v1/images/generations`.

use tasks::Error;
use tasks::api::{Image, Paint};

/// 2x2 pictures. Red is the seed, blue the index.
pub(super) struct MockPaint;

impl Paint for MockPaint {
    fn side(&self) -> usize {
        2
    }

    fn paint(&self, _: &str, n: usize, seed: u32) -> Result<Vec<Image>, Error> {
        let one = |i: usize| Image {
            width: 2,
            height: 2,
            rgba: [seed as u8, 0, i as u8, 255].repeat(4),
        };
        Ok((0..n).map(one).collect())
    }
}
