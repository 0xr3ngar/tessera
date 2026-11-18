use crate::klein::sigmas;

/// Diffusers' schedule for 512x512 in 4 steps.
#[test]
fn shifted_sigmas() {
    let want = [1.0, 0.958_085, 0.883_982, 0.717_497, 0.0];
    for (got, want) in sigmas(1024, 4).iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{got} against {want}");
    }
}
