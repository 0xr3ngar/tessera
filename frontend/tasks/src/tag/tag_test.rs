// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

use super::Tagger;

/// Each case: the text, then spaCy's tokens as `[text, tag]`.
type Case = (String, Vec<(String, String)>);

/// Matches spaCy's tags on the texts in `$TESSERA_SPACY/tags.json`.
#[test]
fn spacy_tags() {
    let Some(dir) = settings::read().reference("spacy") else {
        eprintln!("TESSERA_SPACY unset: skipped");
        return;
    };
    let mut tagger = Tagger::open(&dir, &settings::read()).unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(dir.join("tags.json")).unwrap()).unwrap();
    let (mut wrong, mut tokens) = (0, 0);
    for (text, want) in &cases {
        let got: Vec<(String, String)> = tagger
            .tag(text)
            .unwrap()
            .into_iter()
            .map(|(w, t)| (text[w.span].into(), t.into()))
            .collect();
        tokens += want.len();
        let misses =
            got.iter().zip(want).filter(|(a, b)| a != b).count() + got.len().abs_diff(want.len());
        if misses > 0 {
            wrong += misses;
            eprintln!("{text:?}\n  got  {got:?}\n  want {want:?}");
        }
    }
    assert_eq!(wrong, 0, "{wrong} of {tokens} tokens differ");
}
