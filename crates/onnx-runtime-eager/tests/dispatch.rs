//! End-to-end dispatch tests for `onnx-runtime-eager` (`docs/EAGER.md` Phase-1).
//!
//! Each test drives a real op through the full 7-step dispatch flow — opset
//! resolution, device resolution, kernel compile+cache, shape inference, output
//! allocation, and kernel execution — and checks the numeric result against
//! hand-computed values.

use std::collections::HashMap;

use onnx_runtime_eager::{EagerContext, EagerError, Tensor};

/// f32 comparison helper with a tight tolerance for exact-arithmetic ops.
fn assert_close(got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(
        got.len(),
        want.len(),
        "length mismatch: {got:?} vs {want:?}"
    );
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() <= tol, "‖{g} - {w}‖ > {tol}");
    }
}

#[test]
fn dispatch_add_f32() {
    let ctx = EagerContext::new().unwrap();
    let a = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&[2, 2], &[10.0, 20.0, 30.0, 40.0]).unwrap();
    let out = ctx
        .dispatch("Add", "", &[&a, &b], &HashMap::new(), None)
        .unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_close(&out[0].to_vec_f32(), &[11.0, 22.0, 33.0, 44.0], 1e-6);
}

#[test]
fn dispatch_matmul_f32() {
    let ctx = EagerContext::new().unwrap();
    // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&[2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();
    let out = ctx
        .dispatch("MatMul", "", &[&a, &b], &HashMap::new(), None)
        .unwrap();
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_close(&out[0].to_vec_f32(), &[19.0, 22.0, 43.0, 50.0], 1e-6);
}

#[test]
fn dispatch_relu_f32() {
    let ctx = EagerContext::new().unwrap();
    let x = Tensor::from_f32(&[2, 2], &[-1.0, 0.0, 2.0, -3.0]).unwrap();
    let out = ctx
        .dispatch("Relu", "", &[&x], &HashMap::new(), None)
        .unwrap();
    assert_eq!(out[0].shape(), &[2, 2]);
    assert_close(&out[0].to_vec_f32(), &[0.0, 0.0, 2.0, 0.0], 1e-6);
}

#[test]
fn dispatch_custom_domain_gelu() {
    // `Gelu` is registered under `com.microsoft` in ep-cpu — exercises the
    // custom-domain routing path (opset resolution + shape inference + kernel
    // lookup keyed on a non-default domain).
    let ctx = EagerContext::new().unwrap();
    let x = Tensor::from_f32(&[4], &[0.0, 1.0, -1.0, 2.0]).unwrap();
    let out = ctx
        .dispatch("Gelu", "com.microsoft", &[&x], &HashMap::new(), None)
        .unwrap();
    assert_eq!(out[0].shape(), &[4]);
    // Exact GELU: 0.5·x·(1 + erf(x/√2)).
    // gelu(0)=0, gelu(1)=0.8413447, gelu(-1)=-0.1586553, gelu(2)=1.9544997.
    assert_close(
        &out[0].to_vec_f32(),
        &[0.0, 0.841_344_7, -0.158_655_3, 1.954_499_7],
        1e-4,
    );
}

#[test]
fn dispatch_unknown_op_is_no_kernel() {
    let ctx = EagerContext::new().unwrap();
    let x = Tensor::from_f32(&[2], &[1.0, 2.0]).unwrap();
    let err = ctx
        .dispatch("Conv", "", &[&x], &HashMap::new(), None)
        .unwrap_err();
    match err {
        EagerError::NoKernel {
            op_type, domain, ..
        } => {
            assert_eq!(op_type, "Conv");
            assert_eq!(domain, "");
        }
        other => panic!("expected NoKernel, got {other:?}"),
    }
}

#[test]
fn kernel_cache_reuses_same_op_and_shape() {
    let ctx = EagerContext::new().unwrap();
    let a = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();
    let b = Tensor::from_f32(&[2, 2], &[5.0, 6.0, 7.0, 8.0]).unwrap();

    let before = ctx.cache_stats();
    assert_eq!(before.entries, 0);

    let _ = ctx
        .dispatch("Add", "", &[&a, &b], &HashMap::new(), None)
        .unwrap();
    let after_first = ctx.cache_stats();
    assert_eq!(after_first.misses, 1, "first dispatch compiles a kernel");
    assert_eq!(after_first.hits, 0);
    assert_eq!(after_first.entries, 1);

    let _ = ctx
        .dispatch("Add", "", &[&a, &b], &HashMap::new(), None)
        .unwrap();
    let after_second = ctx.cache_stats();
    assert_eq!(
        after_second.misses, 1,
        "second dispatch reuses the cached kernel"
    );
    assert_eq!(after_second.hits, 1);
    assert_eq!(after_second.entries, 1);

    // A different shape is a distinct cache entry (kernels are shape-specialised).
    let c = Tensor::from_f32(&[3], &[1.0, 2.0, 3.0]).unwrap();
    let d = Tensor::from_f32(&[3], &[4.0, 5.0, 6.0]).unwrap();
    let _ = ctx
        .dispatch("Add", "", &[&c, &d], &HashMap::new(), None)
        .unwrap();
    let after_third = ctx.cache_stats();
    assert_eq!(after_third.misses, 2);
    assert_eq!(after_third.entries, 2);
}

#[test]
fn explicit_opset_is_accepted() {
    // A per-call opset override must still dispatch (priority: explicit > default).
    let ctx = EagerContext::new().unwrap();
    let a = Tensor::from_f32(&[2], &[1.0, 2.0]).unwrap();
    let b = Tensor::from_f32(&[2], &[3.0, 4.0]).unwrap();
    let out = ctx
        .dispatch("Add", "", &[&a, &b], &HashMap::new(), Some(17))
        .unwrap();
    assert_close(&out[0].to_vec_f32(), &[4.0, 6.0], 1e-6);
}
