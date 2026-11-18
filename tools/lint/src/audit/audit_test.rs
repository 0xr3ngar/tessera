use super::level;
use sensai_findings::Level;

#[test]
fn listed_rules_advise() {
    let warn = ["unused-pub".to_string()];
    assert!(level("unused-pub", &warn) == Level::Note);
    assert!(level("file-lines", &warn) == Level::Fail);
}
