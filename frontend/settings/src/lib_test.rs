#[test]
fn reads_variables() {
    // SAFETY: this crate's only test, so no other thread reads.
    unsafe {
        std::env::set_var("TESSERA_LANES", "8");
        std::env::set_var("TESSERA_EMULATE", "cuda");
        std::env::set_var("TESSERA_BACKEND", "host");
        std::env::set_var("TESSERA_CHECK", "/shared/check");
        std::env::set_var("TESSERA_MEMORY", "24G");
    }
    let s = crate::read();
    assert_eq!(s.host_lanes, 8);
    assert_eq!(s.emulate, Some(engine::Backend::Cuda));
    assert_eq!(s.gpu(), None);
    assert_eq!(s.memory, engine::Memory::Bytes(24 << 30));
    assert_eq!(s.check, Some(std::path::PathBuf::from("/shared/check")));
}
