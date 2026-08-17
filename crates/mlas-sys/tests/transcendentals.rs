//! Smoke tests pinning the `tanh`/`erf`/`gelu_erf` FFI at the crate boundary.
//!
//! The callers in `onnx-runtime-ep-cpu` compare these against a pure-Rust
//! reference, but that only runs when the `mlas` feature is on there. These
//! tests fix the shim signatures and the basic contract here, where the
//! bindings live, so a bad `extern "C"` declaration or a template
//! instantiation that silently stops linking fails in this crate.

/// Reference values are the correctly-rounded `f64` result of the same
/// function, so this pins the *function*, not a particular polynomial.
fn assert_close(actual: &[f32], expect: impl Fn(f64) -> f64, input: &[f32], tol: f64, what: &str) {
    for (&y, &x) in actual.iter().zip(input) {
        let want = expect(f64::from(x));
        let got = f64::from(y);
        assert!(
            (got - want).abs() <= tol,
            "{what}({x}) = {got}, expected ~{want} (tol {tol})"
        );
    }
}

/// Long enough that the vector body runs and leaves a ragged tail.
fn inputs() -> Vec<f32> {
    (0..1000).map(|i| (i as f32 - 500.0) / 64.0).collect()
}

#[test]
fn compute_tanh_matches_a_f64_reference() {
    let x = inputs();
    let mut y = vec![0.0f32; x.len()];
    mlas_sys::compute_tanh(&x, &mut y);
    assert_close(&y, f64::tanh, &x, 1e-6, "tanh");
}

#[test]
fn compute_erf_matches_a_f64_reference() {
    let x = inputs();
    let mut y = vec![0.0f32; x.len()];
    mlas_sys::compute_erf(&x, &mut y);
    assert_close(&y, libm::erf, &x, 1e-6, "erf");
}

#[test]
fn compute_gelu_erf_matches_a_f64_reference() {
    let x = inputs();
    let mut y = vec![0.0f32; x.len()];
    mlas_sys::compute_gelu_erf(&x, &mut y);
    assert_close(
        &y,
        |v| v * 0.5 * (1.0 + libm::erf(v / std::f64::consts::SQRT_2)),
        &x,
        1e-6,
        "gelu_erf",
    );
}

/// Empty input must not call into MLAS with a dangling pointer.
#[test]
fn empty_slices_are_a_no_op() {
    let mut empty: [f32; 0] = [];
    mlas_sys::compute_tanh(&[], &mut empty);
    mlas_sys::compute_erf(&[], &mut empty);
    mlas_sys::compute_gelu_erf(&[], &mut empty);
}

#[test]
#[should_panic(expected = "equal length")]
fn mismatched_lengths_panic_rather_than_read_out_of_bounds() {
    let x = vec![0.0f32; 8];
    let mut y = vec![0.0f32; 4];
    mlas_sys::compute_tanh(&x, &mut y);
}
