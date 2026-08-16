//! End-to-end dispatch tests for `onnx-runtime-eager` (`docs/execution/EAGER.md` Phase-1).
//!
//! Each test drives a real op through the full 7-step dispatch flow — opset
//! resolution, device resolution, kernel compile+cache, shape inference, output
//! allocation, and kernel execution — and checks the numeric result against
//! hand-computed values.

use std::collections::HashMap;

use onnx_runtime_eager::{EagerContext, EagerError, Tensor};
use onnx_runtime_ir::Attribute;

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
        .dispatch("ThisOpDoesNotExist", "", &[&x], &HashMap::new(), None)
        .unwrap_err();
    match err {
        EagerError::NoKernel {
            op_type, domain, ..
        } => {
            assert_eq!(op_type, "ThisOpDoesNotExist");
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

#[test]
fn dispatch_split_materializes_all_requested_outputs_in_order() {
    let ctx = EagerContext::new().unwrap();
    let x = Tensor::from_f32(
        &[2, 3, 2],
        &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    )
    .unwrap();
    let attrs = HashMap::from([
        ("axis".to_string(), Attribute::Int(1)),
        ("split".to_string(), Attribute::Ints(vec![1, 2])),
    ]);

    let outputs = ctx
        .dispatch_with_outputs("Split", "", &[&x], &attrs, 2, None)
        .unwrap();

    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0].shape(), &[2, 1, 2]);
    assert_close(&outputs[0].to_vec_f32(), &[1., 2., 7., 8.], 1e-6);
    assert_eq!(outputs[1].shape(), &[2, 2, 2]);
    assert_close(
        &outputs[1].to_vec_f32(),
        &[3., 4., 5., 6., 9., 10., 11., 12.],
        1e-6,
    );
}

#[test]
fn dispatch_topk_returns_values_and_indices_for_largest_and_smallest() {
    let ctx = EagerContext::new().unwrap();
    let x = Tensor::from_f32(&[2, 4], &[2., 5., 5., 1., 4., 3., 2., 3.]).unwrap();
    let k = Tensor::from_i64(&[], &[2]).unwrap();

    let largest = ctx
        .dispatch_with_outputs(
            "TopK",
            "",
            &[&x, &k],
            &HashMap::from([
                ("axis".to_string(), Attribute::Int(1)),
                ("largest".to_string(), Attribute::Int(1)),
                ("sorted".to_string(), Attribute::Int(1)),
            ]),
            2,
            None,
        )
        .unwrap();
    assert_eq!(largest[0].shape(), &[2, 2]);
    assert_close(&largest[0].to_vec_f32(), &[5., 5., 4., 3.], 1e-6);
    assert_eq!(largest[1].to_vec_i64(), vec![1, 2, 0, 1]);

    let smallest = ctx
        .dispatch_with_outputs(
            "TopK",
            "",
            &[&x, &k],
            &HashMap::from([
                ("axis".to_string(), Attribute::Int(1)),
                ("largest".to_string(), Attribute::Int(0)),
                ("sorted".to_string(), Attribute::Int(0)),
            ]),
            2,
            None,
        )
        .unwrap();
    assert_close(&smallest[0].to_vec_f32(), &[1., 2., 2., 3.], 1e-6);
    assert_eq!(smallest[1].to_vec_i64(), vec![3, 0, 2, 1]);
}

#[test]
fn dispatch_allows_omitting_optional_trailing_output() {
    let ctx = EagerContext::new().unwrap();
    let x = Tensor::from_f32(&[2], &[3., 4.]).unwrap();

    let outputs = ctx
        .dispatch_with_outputs("Dropout", "", &[&x], &HashMap::new(), 1, None)
        .unwrap();

    assert_eq!(outputs.len(), 1);
    assert_close(&outputs[0].to_vec_f32(), &[3., 4.], 1e-6);
}

#[test]
fn dispatch_rejects_invalid_or_unsupported_output_counts_cleanly() {
    let ctx = EagerContext::new().unwrap();
    let x = Tensor::from_f32(&[2], &[1., 2.]).unwrap();
    let zero = ctx
        .dispatch_with_outputs("Relu", "", &[&x], &HashMap::new(), 0, None)
        .unwrap_err();
    assert!(matches!(zero, EagerError::InvalidOutputCount));

    let too_many = ctx
        .dispatch_with_outputs("Relu", "", &[&x], &HashMap::new(), 2, None)
        .unwrap_err();
    assert!(matches!(too_many, EagerError::ShapeInference { .. }));

    let k = Tensor::from_i64(&[], &[1]).unwrap();
    let too_few = ctx
        .dispatch_with_outputs("TopK", "", &[&x, &k], &HashMap::new(), 1, None)
        .unwrap_err();
    assert!(matches!(too_few, EagerError::Kernel(_)));
}
