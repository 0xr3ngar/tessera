use std::path::PathBuf;

#[test]
fn reads_references() {
    // SAFETY: this crate's only test, so no other thread reads.
    unsafe {
        std::env::set_var("TESSERA_LANES", "8");
        std::env::set_var("TESSERA_EMULATE", "cuda");
        std::env::set_var("TESSERA_QWEN3", "/ref/qwen3");
    }
    let s = crate::read();
    assert_eq!(s.host_lanes, 8);
    assert_eq!(s.emulate, Some(engine_settings::Backend::Cuda));
    assert_eq!(s.reference("qwen3"), Some(PathBuf::from("/ref/qwen3")));
    assert_eq!(s.reference("lanes"), None);
}
