#[test]
fn reads_variables() {
    // SAFETY: this crate's only test, so no other thread reads.
    unsafe {
        std::env::set_var("TESSERA_LANES", "8");
        std::env::set_var("TESSERA_EMULATE", "cuda");
        std::env::set_var("TESSERA_BACKEND", "host");
    }
    let s = crate::read();
    assert_eq!(s.host_lanes, 8);
    assert_eq!(s.emulate, Some(engine_settings::Backend::Cuda));
    assert_eq!(s.gpu(), None);
}
