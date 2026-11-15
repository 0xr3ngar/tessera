// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

use super::features::{Features, hash_string, shape, thinc_hash};
use bundle::tokenize::spacy::Word;
use std::collections::HashMap;

/// spaCy 3.8 and Thinc 8 give these values.
#[test]
fn hashes_match_spacy() {
    let loader = hash_string("loader");
    assert_eq!(loader, 9_424_920_502_403_081_132);
    assert_eq!(
        thinc_hash(loader, 8),
        [3_554_004_892, 3_057_108_156, 1_036_040_042, 2_438_178_587]
    );
    let f = Features::new(HashMap::new());
    assert_eq!((f.id("X"), f.id("Xxxx")), (101, 10_887_629_174_180_191_697));
    assert_eq!(
        (shape("C3PO"), shape("aaaaaa")),
        ("XdXX".into(), "xxxx".into())
    );
    let word = Word {
        span: 0..3,
        norm: None,
        space_after: false,
    };
    assert_eq!(f.norm("…", &word), "...");
}
