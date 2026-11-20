// SPDX-License-Identifier: MIT
// Ported from spaCy. Copyright ExplosionAI GmbH, see LICENSE.md.

use super::Tagger;

/// Each case: the text, then spaCy's tokens as `[text, tag]`.
type Case = (String, Vec<(String, String)>);

/// Matches spaCy's tags on the spacy reference's `tags.json`.
#[test]
fn spacy_tags() {
    let Some(dir) = tessera_testkit::reference("spacy") else {
        eprintln!("no spacy reference here: skipped");
        return;
    };
    let mut tagger = Tagger::open(&dir, &tessera_engine::Settings::default()).unwrap();
    let cases: Vec<Case> =
        serde_json::from_slice(&std::fs::read(dir.join("tags.json")).unwrap()).unwrap();
    for (text, want) in &cases {
        let got: Vec<(String, String)> = tagger
            .tag(text)
            .unwrap()
            .into_iter()
            .map(|(w, t)| (text[w.span].into(), t.into()))
            .collect();
        assert_eq!(&got, want, "{text:?}");
    }
}
