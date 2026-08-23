//! `TensorScatter` (ai.onnx opset 24) against the ONNX reference implementation.
//!
//! Golden values were produced by `onnx.reference.ReferenceEvaluator` running a
//! real opset-24 `TensorScatter` node, not by reasoning about the spec text —
//! except for the one case where the reference is itself wrong, which is called
//! out explicitly below.

#[path = "../benches/common/mod.rs"]
mod common;

use common::{FloatDType, Tensor, assert_close, make_kernel};
use onnx_runtime_ir::Attribute;

/// Cache shaped (batch=2, heads=1, max_sequence_length=4, head_size=2) with the
/// sequence axis at -2 — the layout the op was standardized for.
const CACHE_SHAPE: [usize; 4] = [2, 1, 4, 2];
const UPDATE_SHAPE: [usize; 4] = [2, 1, 2, 2];

fn cache_prefilled_with_sentinels() -> Tensor {
    Tensor::floats(FloatDType::F32, &CACHE_SHAPE, &[-1.0; 16])
}

fn update_1_through_8() -> Tensor {
    Tensor::floats(
        FloatDType::F32,
        &UPDATE_SHAPE,
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    )
}

fn run(mode: &str, write_indices: Option<&[i64]>) -> Vec<f32> {
    let past = cache_prefilled_with_sentinels();
    let update = update_1_through_8();
    let mut present = Tensor::zeros(FloatDType::F32, &CACHE_SHAPE);

    let indices = write_indices.map(|values| Tensor::i64(&[values.len()], values));
    let mut shapes = vec![CACHE_SHAPE.to_vec(), UPDATE_SHAPE.to_vec()];
    if let Some(values) = write_indices {
        shapes.push(vec![values.len()]);
    }

    let mut inputs = vec![past.view(), update.view()];
    if let Some(tensor) = &indices {
        inputs.push(tensor.view());
    }

    make_kernel(
        "TensorScatter",
        [
            ("axis", Attribute::Int(-2)),
            ("mode", Attribute::String(mode.to_string().into_bytes())),
        ],
        &shapes,
        24,
    )
    .execute(&inputs, &mut [present.view_mut()])
    .expect("TensorScatter executes");
    present.to_f32()
}

#[test]
fn linear_writes_each_sample_at_its_own_offset() {
    // batch 0 writes at 0, batch 1 writes at 2 — the decode-phase shape, where
    // each sample appends after its own last valid token.
    assert_close(
        &run("linear", Some(&[0, 2])),
        &[
            1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0, // batch 0
            -1.0, -1.0, -1.0, -1.0, 5.0, 6.0, 7.0, 8.0, // batch 1
        ],
        0.0,
    );
}

#[test]
fn an_absent_write_indices_input_means_offset_zero() {
    // The prefill phase supplies only two inputs.
    assert_close(
        &run("linear", None),
        &[
            1.0, 2.0, 3.0, 4.0, -1.0, -1.0, -1.0, -1.0, // batch 0
            5.0, 6.0, 7.0, 8.0, -1.0, -1.0, -1.0, -1.0, // batch 1
        ],
        0.0,
    );
}

#[test]
fn circular_wraps_the_sequence_coordinate() {
    // Writing at index 3 of a 4-slot cache puts sequence 0 at slot 3 and
    // sequence 1 back at slot 0. This case matches the ONNX reference exactly:
    // every prefix coordinate here is smaller than max_sequence_length, so the
    // reference's whole-tuple modulo is the identity on them.
    assert_close(
        &run("circular", Some(&[3, 3])),
        &[
            3.0, 4.0, -1.0, -1.0, -1.0, -1.0, 1.0, 2.0, // batch 0
            7.0, 8.0, -1.0, -1.0, -1.0, -1.0, 5.0, 6.0, // batch 1
        ],
        0.0,
    );
}

#[test]
fn circular_does_not_wrap_the_batch_coordinate() {
    // The discriminating case, and a deliberate divergence from the current
    // ONNX reference implementation.
    //
    // batch = 5 exceeds max_sequence_length = 4, so the reference's whole-tuple
    // `np.mod` folds batch 4 onto batch 0: measured, it returns
    // [5, 2, 3, 4, -1] — sample 4's write lands in sample 0's cache and sample
    // 4 keeps stale contents. The spec prose says only the write index wraps,
    // which gives [1, 2, 3, 4, 5]. We implement the prose; reported upstream as
    // onnx/onnx#8353.
    //
    // Shape is (batch=5, max_sequence_length=4, 1) with the sequence axis at 1.
    let past = Tensor::floats(FloatDType::F32, &[5, 4, 1], &[-1.0; 20]);
    let update = Tensor::floats(FloatDType::F32, &[5, 1, 1], &[1.0, 2.0, 3.0, 4.0, 5.0]);
    let mut present = Tensor::zeros(FloatDType::F32, &[5, 4, 1]);
    let indices = Tensor::i64(&[5], &[0, 0, 0, 0, 0]);

    make_kernel(
        "TensorScatter",
        [
            ("axis", Attribute::Int(1)),
            (
                "mode",
                Attribute::String("circular".to_string().into_bytes()),
            ),
        ],
        &[vec![5, 4, 1], vec![5, 1, 1], vec![5]],
        24,
    )
    .execute(
        &[past.view(), update.view(), indices.view()],
        &mut [present.view_mut()],
    )
    .expect("TensorScatter executes");

    let slot_zero_per_batch: Vec<f32> = present.to_f32().chunks(4).map(|row| row[0]).collect();
    assert_close(&slot_zero_per_batch, &[1.0, 2.0, 3.0, 4.0, 5.0], 0.0);
}

#[test]
fn linear_refuses_a_write_that_would_run_past_the_cache() {
    let past = cache_prefilled_with_sentinels();
    let update = update_1_through_8();
    let mut present = Tensor::zeros(FloatDType::F32, &CACHE_SHAPE);
    // max_sequence_length is 4 and the update is 2 long, so offset 3 overflows.
    let indices = Tensor::i64(&[2], &[3, 0]);

    let error = make_kernel(
        "TensorScatter",
        [
            ("axis", Attribute::Int(-2)),
            ("mode", Attribute::String("linear".to_string().into_bytes())),
        ],
        &[CACHE_SHAPE.to_vec(), UPDATE_SHAPE.to_vec(), vec![2]],
        24,
    )
    .execute(
        &[past.view(), update.view(), indices.view()],
        &mut [present.view_mut()],
    )
    .expect_err("a linear write past the end must not silently wrap");
    let error = error.to_string();
    assert!(
        error.contains("exceeds cache capacity"),
        "the error must name the real cause: {error}"
    );
}

#[test]
fn the_untouched_region_is_carried_over_from_the_past_cache() {
    // A cache whose existing contents are distinct per slot: everything outside
    // the written window must survive byte-for-byte, which is what makes this a
    // functional model of an in-place update.
    let past = Tensor::floats(
        FloatDType::F32,
        &CACHE_SHAPE,
        &[
            10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            26.0, 27.0,
        ],
    );
    let update = Tensor::floats(
        FloatDType::F32,
        &[2, 1, 1, 2],
        &[100.0, 101.0, 200.0, 201.0],
    );
    let mut present = Tensor::zeros(FloatDType::F32, &CACHE_SHAPE);
    let indices = Tensor::i64(&[2], &[1, 3]);

    make_kernel(
        "TensorScatter",
        [
            ("axis", Attribute::Int(-2)),
            ("mode", Attribute::String("linear".to_string().into_bytes())),
        ],
        &[CACHE_SHAPE.to_vec(), vec![2, 1, 1, 2], vec![2]],
        24,
    )
    .execute(
        &[past.view(), update.view(), indices.view()],
        &mut [present.view_mut()],
    )
    .expect("TensorScatter executes");

    assert_close(
        &present.to_f32(),
        &[
            10.0, 11.0, 100.0, 101.0, 14.0, 15.0, 16.0, 17.0, // batch 0: slot 1 replaced
            20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 200.0, 201.0, // batch 1: slot 3 replaced
        ],
        0.0,
    );
}
