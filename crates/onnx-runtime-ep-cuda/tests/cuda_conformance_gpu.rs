#![allow(
    clippy::too_many_arguments,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings,
    clippy::doc_lazy_continuation,
    clippy::uninlined_format_args,
    clippy::cloned_ref_to_slice_refs,
    clippy::type_complexity,
    clippy::drop_non_drop,
    clippy::manual_repeat_n,
    clippy::manual_is_multiple_of,
    clippy::err_expect,
    clippy::clone_on_copy
)]
//! Data-driven CUDA EP **conformance profile**.
//!
//! This suite treats the CPU execution provider as the reference oracle and
//! validates that every op the CUDA EP *claims* to support (the
//! [`CUDA_COVERED_OPS`] table) is actually covered by a parity test, in a
//! declarative, maintainable way rather than ad-hoc per-op tests.
//!
//! A **conformance profile** is a table of [`ProfileEntry`] — one per covered op
//! — where each op is either:
//!
//! * [`Coverage::Sweep`] — inline `(op, dtype, shapes, attrs)` parity cases that
//!   the generic sweep here runs against the CPU EP; or
//! * [`Coverage::Dedicated`] — covered by a named dedicated GPU parity suite
//!   (a test source file that runs the op), recorded so the claim is auditable.
//!
//! The three profile guards run when `gpu-tests` is enabled on a CUDA runner:
//!
//! * [`every_covered_op_has_a_conformance_entry`] — the *coverage-of-coverage*
//!   check: it fails if any [`CUDA_COVERED_OPS`] op is missing a profile entry
//!   (i.e. "claimed but untested" — the exact class of miss that let
//!   `ReduceLogSumExp` and bf16 coverage gaps slip through), and if any profile
//!   entry references an op no longer covered.
//! * [`dedicated_suites_exist_and_name_their_op`] — verifies each `Dedicated`
//!   suite file exists and actually names its op, so deleting/renaming a suite
//!   cannot silently leave an op unverified.
//! * [`profile_has_no_duplicate_entries`].
//!
//! The GPU sweep ([`conformance_sweep_matches_cpu`]) is ignored unless
//! `gpu-tests` is enabled. Run it on a GPU box with:
//!
//! ```bash
//! CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-runtime-ep-cuda --features cuda,gpu-tests \
//!     --test cuda_conformance_gpu
//! ```
//!
//! See `docs/execution/CUDA_COVERAGE.md` ("Conformance profile & GPU parity sweep").

mod common;

use std::collections::HashSet;
use std::path::Path;

use common::{
    Tensor, absent_input, assert_close, build_graph, decode_floats, float_input, input,
    require_cuda, run_cpu, run_cuda, try_run_cuda,
};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cuda::{CUDA_COVERED_OPS, CudaExecutionProvider};
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, ValueId, static_shape};

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests"
)]
fn transpose_packed_cuda_matches_reference_bytes() {
    let ep = require_cuda();
    // Permute [3,1,5] -> [5,3,1]. Each column crosses packed source bytes,
    // and the dirty high bits of the final source byte never reach the output.
    let nibble_input = [0x70, 0x6e, 0x4d, 0x3c, 0x2a, 0x09, 0xf8, 0xf6];
    let nibble_expected = [0x40, 0x79, 0x0c, 0x3e, 0x68, 0xfa, 0x2d, 0x06];
    let two_bit_input = [0xac, 0xc1, 0x1a, 0xec];
    let two_bit_expected = [0xd0, 0xe0, 0xe8, 0x29];
    let float8_input = [0, 7, 14, 6, 13, 4, 12, 3, 10, 2, 9, 0, 8, 15, 6];
    let float8_expected = [0, 4, 9, 7, 12, 0, 14, 3, 8, 6, 10, 15, 13, 2, 6];
    let cases: [(DataType, u64, &[u8], &[u8]); 6] = [
        (DataType::Int4, 21, &nibble_input, &nibble_expected),
        (DataType::Uint4, 21, &nibble_input, &nibble_expected),
        (DataType::Float4E2M1, 23, &nibble_input, &nibble_expected),
        (DataType::Float8E8M0, 24, &float8_input, &float8_expected),
        (DataType::Int2, 25, &two_bit_input, &two_bit_expected),
        (DataType::Uint2, 25, &two_bit_input, &two_bit_expected),
    ];
    for (dtype, opset, payload, expected) in cases {
        let claim = ep.supports_op(
            &Node::new(NodeId(0), "Transpose", vec![], vec![]),
            opset - 1,
            &[static_shape([3, 1, 5])],
            &[dtype],
            &[],
        );
        assert!(
            claim
                .reason()
                .is_some_and(|reason| reason.contains(&format!("use opset {opset}"))),
            "CUDA must reject {dtype:?} before opset {opset}: {:?}",
            claim.reason()
        );
        let outputs = run_cuda(
            &ep,
            "Transpose",
            "",
            opset,
            &[input(dtype, &[3, 1, 5], payload)],
            &[(dtype, vec![5, 3, 1])],
            &[("perm", Attribute::Ints(vec![2, 0, 1]))],
        );
        assert_eq!(outputs[0], expected, "{dtype:?} at opset {opset}");
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable gpu-tests"
)]
fn packed_integer_cast_cuda_matches_strict_reference_bytes() {
    let ep = require_cuda();
    let shape = static_shape([7]);
    let mut node = Node::new(NodeId(0), "Cast", vec![], vec![]);
    node.attributes
        .insert("to".into(), Attribute::Int(DataType::Uint4 as i64));
    assert!(
        ep.supports_op(&node, 1, &[shape.clone()], &[DataType::Uint8], &[])
            .is_supported()
    );
    node.attributes
        .insert("to".into(), Attribute::Int(DataType::Int2 as i64));
    assert!(
        ep.supports_op(&node, 1, &[shape], &[DataType::Uint8], &[])
            .is_supported()
    );
    node.attributes
        .insert("to".into(), Attribute::Int(DataType::Float8E4M3FN as i64));
    assert!(
        ep.supports_op(&node, 25, &[static_shape([7])], &[DataType::Uint4], &[],)
            .reason()
            .is_some_and(|reason| reason.contains("does not support target dtype"))
    );
    node.attributes
        .insert("to".into(), Attribute::Int(DataType::Uint4 as i64));
    assert!(
        ep.supports_op(
            &node,
            25,
            &[static_shape([7])],
            &[DataType::Float8E4M3FN],
            &[],
        )
        .reason()
        .is_some_and(|reason| reason.contains("does not support source dtype"))
    );

    node.attributes
        .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
    assert!(
        ep.supports_op(
            &node,
            27,
            &[static_shape([3])],
            &[DataType::Float6E2M3],
            &[],
        )
        .reason()
        .is_some_and(|reason| reason.contains("opset 28"))
    );
    assert!(
        ep.supports_op(
            &node,
            28,
            &[static_shape([3])],
            &[DataType::Float6E2M3],
            &[],
        )
        .is_supported()
    );
    node.attributes
        .insert("to".into(), Attribute::Int(DataType::Int32 as i64));
    assert!(
        ep.supports_op(
            &node,
            28,
            &[static_shape([3])],
            &[DataType::Float6E2M3],
            &[],
        )
        .reason()
        .is_some_and(|reason| reason.contains("supports only"))
    );
    let cast_like = Node::new(NodeId(1), "CastLike", vec![], vec![]);
    assert!(
        ep.supports_op(
            &cast_like,
            15,
            &[static_shape([3]), static_shape([3])],
            &[DataType::Float6E2M3, DataType::Float32],
            &[],
        )
        .reason()
        .is_some_and(|reason| reason.contains("standard Cast"))
    );

    let cases = [
        (
            "uint8-to-uint4-codes-and-zero-points",
            1,
            input(DataType::Uint8, &[7], &[0u8, 1, 7, 8, 15, 16, 255]),
            DataType::Uint4,
            vec![0x10, 0x87, 0x0f, 0x0f],
        ),
        (
            "uint4-to-int2-between-packed-types",
            1,
            input(DataType::Uint4, &[5], &[0x10u8, 0x87, 0xff]),
            DataType::Int2,
            vec![0x34, 0x03],
        ),
        (
            "int2-to-int4-sign-extension",
            1,
            input(DataType::Int2, &[5], &[0x34u8, 0x03]),
            DataType::Int4,
            vec![0x10, 0x0f, 0x0f],
        ),
    ];
    for (label, opset, source, target, expected) in cases {
        let attrs = [("to", Attribute::Int(target as i64))];
        let outputs = [(target, source.shape.clone())];
        assert_eq!(
            run_cpu(
                "Cast",
                "",
                opset,
                std::slice::from_ref(&source),
                &outputs,
                &attrs
            )[0],
            expected,
            "{label}: CPU reference"
        );
        assert_eq!(
            run_cuda(
                &ep,
                "Cast",
                "",
                opset,
                std::slice::from_ref(&source),
                &outputs,
                &attrs,
            )[0],
            expected,
            "{label}: CUDA"
        );
    }

    let floats = float_input(
        DataType::Float32,
        &[6],
        &[-20.0, -8.9, -7.9, 7.9, 8.0, f32::NAN],
    );
    let attrs = [("to", Attribute::Int(DataType::Int4 as i64))];
    let outputs = [(DataType::Int4, vec![6])];
    let expected = vec![0x8c, 0x79, 0x08];
    assert_eq!(
        run_cpu(
            "Cast",
            "",
            1,
            std::slice::from_ref(&floats),
            &outputs,
            &attrs
        )[0],
        expected
    );
    assert_eq!(
        run_cuda(&ep, "Cast", "", 1, &[floats], &outputs, &attrs)[0],
        expected
    );

    let source = input(DataType::Uint8, &[5], &[0u8, 1, 2, 3, 4]);
    let target = input(DataType::Uint2, &[1], &[0u8]);
    let outputs = [(DataType::Uint2, vec![5])];
    let expected = vec![0xe4, 0x00];
    assert_eq!(
        run_cpu(
            "CastLike",
            "",
            15,
            &[source.clone(), target.clone()],
            &outputs,
            &[],
        )[0],
        expected
    );
    assert_eq!(
        run_cuda(&ep, "CastLike", "", 15, &[source, target], &outputs, &[])[0],
        expected
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Profile model
// ─────────────────────────────────────────────────────────────────────────────

/// How the outputs of an inline parity case are compared to the CPU oracle.
#[derive(Clone, Copy)]
enum Compare {
    /// Exact byte equality — integer / bool / metadata / data-movement ops.
    ExactBytes,
    /// Element-wise float parity within `tol`, decoding each output's dtype.
    Float { tol: f32 },
}

/// A single inline parity case: one node run on CUDA and on the CPU oracle.
struct Case {
    label: String,
    op: &'static str,
    domain: &'static str,
    opset: u64,
    inputs: Vec<Tensor>,
    outputs: Vec<(DataType, Vec<usize>)>,
    attrs: Vec<(&'static str, Attribute)>,
    compare: Compare,
}

/// How a covered op's conformance is validated.
enum Coverage {
    /// Inline parity cases executed by [`conformance_sweep_matches_cpu`].
    Sweep(Vec<Case>),
    /// Covered by a dedicated GPU parity suite (path relative to this crate's
    /// `tests` directory). The audit verifies the file exists and names the op.
    Dedicated {
        suite: &'static str,
        note: &'static str,
    },
}

/// One covered op and how it is validated.
struct ProfileEntry {
    op: &'static str,
    coverage: Coverage,
}

// ─────────────────────────────────────────────────────────────────────────────
// Case builders (DRY: one generator per op family, driven by data)
// ─────────────────────────────────────────────────────────────────────────────

const FLOAT_DTYPES: [DataType; 3] = [DataType::Float32, DataType::Float16, DataType::BFloat16];

/// Element-wise absolute tolerance per float dtype. f32 device intrinsics agree
/// with host libm to a few ulp; the half formats are dominated by their own
/// rounding step (values chosen small so the ulp stays tight).
fn float_tol(dtype: DataType) -> f32 {
    match dtype {
        DataType::Float32 => 1e-4,
        DataType::Float16 => 3e-3,
        DataType::BFloat16 => 3e-2,
        _ => unreachable!("non-float dtype {dtype:?}"),
    }
}

/// A unary float op fed domain-safe `values`, one case per f32/f16/bf16.
fn unary_float(op: &'static str, opset: u64, values: &[f32]) -> Vec<Case> {
    FLOAT_DTYPES
        .into_iter()
        .map(|dtype| {
            let shape = vec![values.len()];
            Case {
                label: format!("{op}[{dtype:?}]"),
                op,
                domain: "",
                opset,
                inputs: vec![float_input(dtype, &shape, values)],
                outputs: vec![(dtype, shape.clone())],
                attrs: vec![],
                compare: Compare::Float {
                    tol: float_tol(dtype),
                },
            }
        })
        .collect()
}

/// A binary float op with NumPy broadcasting, one case per f32/f16/bf16.
#[allow(clippy::too_many_arguments)]
fn binary_float(
    op: &'static str,
    opset: u64,
    a: &[f32],
    a_shape: &[usize],
    b: &[f32],
    b_shape: &[usize],
    out_shape: &[usize],
) -> Vec<Case> {
    FLOAT_DTYPES
        .into_iter()
        .map(|dtype| Case {
            label: format!("{op}[{dtype:?}]"),
            op,
            domain: "",
            opset,
            inputs: vec![
                float_input(dtype, a_shape, a),
                float_input(dtype, b_shape, b),
            ],
            outputs: vec![(dtype, out_shape.to_vec())],
            attrs: vec![],
            compare: Compare::Float {
                tol: float_tol(dtype),
            },
        })
        .collect()
}

fn quantize_linear_cases() -> Vec<Case> {
    vec![
        Case {
            label: "QuantizeLinear[u8,scalar]".into(),
            op: "QuantizeLinear",
            domain: "",
            opset: 13,
            inputs: vec![
                input(
                    DataType::Float32,
                    &[8],
                    &[-10.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 12.7, 30.0],
                ),
                input(DataType::Float32, &[], &[0.1f32]),
                input(DataType::Uint8, &[], &[128u8]),
            ],
            outputs: vec![(DataType::Uint8, vec![8])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "QuantizeLinear[i8,scalar]".into(),
            op: "QuantizeLinear",
            domain: "",
            opset: 13,
            inputs: vec![
                input(
                    DataType::Float32,
                    &[7],
                    &[-20.0f32, -2.5, -0.25, 0.0, 0.25, 2.5, 20.0],
                ),
                input(DataType::Float32, &[], &[0.25f32]),
                input(DataType::Int8, &[], &[-3i8]),
            ],
            outputs: vec![(DataType::Int8, vec![7])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
    ]
}

fn dequantize_linear_cases() -> Vec<Case> {
    vec![
        Case {
            label: "DequantizeLinear[u8,scalar]".into(),
            op: "DequantizeLinear",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Uint8, &[6], &[0u8, 1, 64, 128, 200, 255]),
                input(DataType::Float32, &[], &[0.125f32]),
                input(DataType::Uint8, &[], &[128u8]),
            ],
            outputs: vec![(DataType::Float32, vec![6])],
            attrs: vec![],
            compare: Compare::Float { tol: 0.0 },
        },
        Case {
            label: "DequantizeLinear[i8,scalar]".into(),
            op: "DequantizeLinear",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Int8, &[6], &[-128i8, -10, -3, 0, 64, 127]),
                input(DataType::Float32, &[], &[0.25f32]),
                input(DataType::Int8, &[], &[-3i8]),
            ],
            outputs: vec![(DataType::Float32, vec![6])],
            attrs: vec![],
            compare: Compare::Float { tol: 0.0 },
        },
    ]
}

fn qlinear_matmul_cases() -> Vec<Case> {
    vec![
        Case {
            label: "QLinearMatMul[uint8,per-tensor]".into(),
            op: "QLinearMatMul",
            domain: "",
            opset: 10,
            inputs: vec![
                input(DataType::Uint8, &[2, 3], &[120u8, 125, 131, 118, 129, 140]),
                input(DataType::Float32, &[], &[0.25f32]),
                input(DataType::Uint8, &[], &[123u8]),
                input(DataType::Uint8, &[3, 2], &[126u8, 119, 130, 121, 124, 135]),
                input(DataType::Float32, &[], &[0.5f32]),
                input(DataType::Uint8, &[], &[127u8]),
                input(DataType::Float32, &[], &[0.2f32]),
                input(DataType::Uint8, &[], &[111u8]),
            ],
            outputs: vec![(DataType::Uint8, vec![2, 2])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "QLinearMatMul[int8,per-row-column,negative-zp]".into(),
            op: "QLinearMatMul",
            domain: "",
            opset: 10,
            inputs: vec![
                input(DataType::Int8, &[2, 3], &[-9i8, 4, 17, -15, 3, 22]),
                input(DataType::Float32, &[2], &[0.2f32, 0.35]),
                input(DataType::Int8, &[2], &[-3i8, 5]),
                input(DataType::Int8, &[3, 2], &[-7i8, 11, 9, -5, 18, 4]),
                input(DataType::Float32, &[2], &[0.4f32, 0.15]),
                input(DataType::Int8, &[2], &[-4i8, 6]),
                input(DataType::Float32, &[], &[0.125f32]),
                input(DataType::Int8, &[], &[-7i8]),
            ],
            outputs: vec![(DataType::Int8, vec![2, 2])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "QLinearMatMul[batched-broadcast,per-axis]".into(),
            op: "QLinearMatMul",
            domain: "",
            opset: 10,
            inputs: vec![
                input(
                    DataType::Uint8,
                    &[2, 2, 3],
                    &[9u8, 11, 15, 7, 13, 20, 21, 17, 12, 18, 14, 10],
                ),
                input(DataType::Float32, &[2, 2, 1], &[0.2f32, 0.3, 0.4, 0.5]),
                input(DataType::Uint8, &[2, 2, 1], &[10u8, 9, 15, 12]),
                input(DataType::Int8, &[1, 3, 2], &[-4i8, 7, 5, -8, 11, 3]),
                input(DataType::Float32, &[1, 1, 2], &[0.25f32, 0.45]),
                input(DataType::Int8, &[1, 1, 2], &[-2i8, 4]),
                input(DataType::Float32, &[1], &[0.1f32]),
                input(DataType::Uint8, &[1], &[100u8]),
            ],
            outputs: vec![(DataType::Uint8, vec![2, 2, 2])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
    ]
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA")]
fn qlinear_matmul_scale_types_match_expected_integer_bytes() {
    let ep = require_cuda();
    for case in qlinear_matmul_cases() {
        run_case(&ep, &case);
    }
    for (opset, scale_dtype) in [
        (10, DataType::Float32),
        (10, DataType::Float16),
        (10, DataType::BFloat16),
        (21, DataType::Float16),
        (21, DataType::BFloat16),
    ] {
        let mut case = Case {
            label: format!("QLinearMatMul-{opset}[{scale_dtype:?},per-row-column,ties]"),
            op: "QLinearMatMul",
            domain: "",
            opset,
            inputs: vec![
                input(DataType::Uint8, &[2, 1], &[4u8, 6]),
                float_input(scale_dtype, &[2], &[0.5, 1.5]),
                input(DataType::Uint8, &[2], &[3u8, 3]),
                input(DataType::Int8, &[1, 2], &[2i8, -2]),
                float_input(scale_dtype, &[2], &[1.0, 0.5]),
                input(DataType::Int8, &[2], &[1i8, -1]),
                float_input(scale_dtype, &[], &[1.0]),
                input(DataType::Uint8, &[], &[10u8]),
            ],
            outputs: vec![(DataType::Uint8, vec![2, 2])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        };
        assert_eq!(
            run_case(&ep, &case),
            vec![vec![10, 10, 14, 8]],
            "{}",
            case.label
        );
        case.inputs[6] = float_input(scale_dtype, &[], &[f32::NAN]);
        let err = try_run_cuda(
            &ep,
            case.op,
            case.domain,
            case.opset,
            &case.inputs,
            &case.outputs,
            &case.attrs,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("y_scale values must be finite and positive"),
            "{err}"
        );
    }

    for (dtype, scales, accumulator, output_zero, expected) in [
        (DataType::Float16, [0.2, 0.75, 0.2], 2u8, 0u8, 1),
        (DataType::BFloat16, [0.5, 1024.0, 76.0], 2, 0, 14),
        (DataType::Float16, [256.0; 3], 1, 0, 255),
        (
            DataType::BFloat16,
            [f32::from_bits(0x5f80_0000); 3],
            1,
            0,
            255,
        ),
        (DataType::Float16, [256.0; 3], 0, 17, 0),
        (
            DataType::BFloat16,
            [f32::from_bits(0x5f80_0000); 3],
            0,
            17,
            0,
        ),
    ] {
        let [a_scale, b_scale, y_scale] = scales.map(|scale| float_input(dtype, &[], &[scale]));
        let case = Case {
            label: format!("QLinearMatMul-21[{dtype:?},scale-arithmetic,{scales:?}]"),
            op: "QLinearMatMul",
            domain: "",
            opset: 21,
            inputs: vec![
                input(DataType::Uint8, &[1, 1], &[accumulator]),
                a_scale,
                input(DataType::Uint8, &[], &[0u8]),
                input(DataType::Int8, &[1, 1], &[1i8]),
                b_scale,
                input(DataType::Int8, &[], &[0i8]),
                y_scale,
                input(DataType::Uint8, &[], &[output_zero]),
            ],
            outputs: vec![(DataType::Uint8, vec![1, 1])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        };
        assert_eq!(run_case(&ep, &case), vec![vec![expected]], "{}", case.label);
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA")]
fn qlinear_matmul_half_scale_admission_is_native_only() {
    let native = require_cuda();
    let legacy = CudaExecutionProvider::new_for_ort_plugin().expect("construct legacy CUDA EP");
    let node = Node::new(
        NodeId(0),
        "QLinearMatMul",
        (0..8).map(|id| Some(ValueId(id))).collect(),
        vec![ValueId(8)],
    );
    let mut case = qlinear_matmul_cases().remove(0);
    case.opset = 21;
    let shapes: Vec<_> = case
        .inputs
        .iter()
        .map(|tensor| static_shape(tensor.shape.iter().copied()))
        .collect();
    let mut dtypes: Vec<_> = case.inputs.iter().map(|tensor| tensor.dtype).collect();
    for dtype in [DataType::Float16, DataType::BFloat16] {
        for index in [1, 4, 6] {
            dtypes[index] = dtype;
        }
        assert!(
            native
                .supports_op(&node, 21, &shapes, &dtypes, &[])
                .is_supported()
        );
        assert!(
            !native
                .supports_op(&node, 9, &shapes, &dtypes, &[])
                .is_supported()
        );
        assert!(
            native
                .supports_op(&node, 10, &shapes, &dtypes, &[])
                .is_supported()
        );
        assert!(
            !legacy
                .supports_op(&node, 10, &shapes, &dtypes, &[])
                .is_supported()
        );
        assert!(
            !legacy
                .supports_op(&node, 21, &shapes, &dtypes, &[])
                .is_supported()
        );
    }

    let descriptors = onnx_runtime_ep_cuda::build_cuda_registry_descriptors(legacy.registry());
    let entries: Vec<_> = descriptors
        .iter()
        .filter(|op| op.op_type == "QLinearMatMul" && op.domain.is_empty())
        .collect();
    assert_eq!(entries.len(), 1);
    assert_eq!(
        (entries[0].since_version, entries[0].end_version),
        (10, u64::MAX)
    );
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        assert!(entries[0].supported_dtypes.contains(&dtype));
    }
    assert_eq!(run_case(&legacy, &case), run_case(&native, &case));
}

fn resize_cases() -> Vec<Case> {
    let image = [1.0, 2.0, 4.0, 8.0, 3.0, 5.0];
    let mut cases = vec![
        Case {
            label: "Resize[nearest,asymmetric,scales,upsample]".into(),
            op: "Resize",
            domain: "",
            opset: 10,
            inputs: vec![
                float_input(DataType::Float32, &[1, 1, 2, 3], &image),
                input(DataType::Float32, &[4], &[1.0f32, 1.0, 2.0, 2.0]),
            ],
            outputs: vec![(DataType::Float32, vec![1, 1, 4, 6])],
            attrs: vec![
                ("mode", Attribute::String(b"nearest".to_vec())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String(b"asymmetric".to_vec()),
                ),
            ],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Resize[nearest,align-corners,sizes,downsample]".into(),
            op: "Resize",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(
                    DataType::Float16,
                    &[1, 1, 4, 5],
                    &(0..20).map(|value| value as f32).collect::<Vec<_>>(),
                ),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Int64, &[4], &[1i64, 1, 2, 3]),
            ],
            outputs: vec![(DataType::Float16, vec![1, 1, 2, 3])],
            attrs: vec![
                ("mode", Attribute::String(b"nearest".to_vec())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String(b"align_corners".to_vec()),
                ),
                (
                    "nearest_mode",
                    Attribute::String(b"round_prefer_ceil".to_vec()),
                ),
            ],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Resize[nearest,half-pixel,floor]".into(),
            op: "Resize",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(DataType::Float32, &[4], &[1.0, 2.0, 4.0, 8.0]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[1], &[1.5f32]),
            ],
            outputs: vec![(DataType::Float32, vec![6])],
            attrs: vec![
                ("mode", Attribute::String(b"nearest".to_vec())),
                ("nearest_mode", Attribute::String(b"floor".to_vec())),
            ],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Resize[nearest,half-pixel,ceil]".into(),
            op: "Resize",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(DataType::Float32, &[4], &[1.0, 2.0, 4.0, 8.0]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[1], &[1.5f32]),
            ],
            outputs: vec![(DataType::Float32, vec![6])],
            attrs: vec![
                ("mode", Attribute::String(b"nearest".to_vec())),
                ("nearest_mode", Attribute::String(b"ceil".to_vec())),
            ],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Resize[linear,half-pixel,scales,upsample]".into(),
            op: "Resize",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(DataType::Float32, &[1, 1, 2, 3], &image),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[4], &[1.0f32, 1.0, 2.0, 2.0]),
            ],
            outputs: vec![(DataType::Float32, vec![1, 1, 4, 6])],
            attrs: vec![("mode", Attribute::String(b"linear".to_vec()))],
            compare: Compare::Float { tol: 1e-5 },
        },
        Case {
            label: "Resize[linear,align-corners,sizes,selected-axes]".into(),
            op: "Resize",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(DataType::BFloat16, &[1, 1, 2, 3], &image),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Int64, &[2], &[5i64, 4]),
            ],
            outputs: vec![(DataType::BFloat16, vec![1, 1, 5, 4])],
            attrs: vec![
                ("mode", Attribute::String(b"linear".to_vec())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String(b"align_corners".to_vec()),
                ),
                ("axes", Attribute::Ints(vec![2, 3])),
            ],
            compare: Compare::Float { tol: 3e-2 },
        },
        Case {
            label: "Resize[linear,asymmetric,scales,multi-axis-downsample]".into(),
            op: "Resize",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(
                    DataType::Float16,
                    &[1, 4, 4],
                    &(0..16).map(|value| value as f32 / 3.0).collect::<Vec<_>>(),
                ),
                input(DataType::Float32, &[0], &[] as &[f32]),
                input(DataType::Float32, &[2], &[0.5f32, 0.5]),
            ],
            outputs: vec![(DataType::Float16, vec![1, 2, 2])],
            attrs: vec![
                ("mode", Attribute::String(b"linear".to_vec())),
                (
                    "coordinate_transformation_mode",
                    Attribute::String(b"asymmetric".to_vec()),
                ),
                ("axes", Attribute::Ints(vec![-2, -1])),
            ],
            compare: Compare::Float { tol: 3e-3 },
        },
    ];
    cases.extend(resize_symmetric_cases());
    cases
}

fn resize_symmetric_cases() -> Vec<Case> {
    let case = |label: &str,
                data: Tensor,
                control,
                output_shape,
                mode: &str,
                axes: Option<Vec<i64>>,
                sizes| {
        let mut inputs = vec![data, input(DataType::Float32, &[0], &[] as &[f32])];
        if sizes {
            inputs.push(input(DataType::Float32, &[0], &[] as &[f32]));
        }
        inputs.push(control);
        let mut attrs = vec![
            ("mode", Attribute::String(mode.as_bytes().to_vec())),
            (
                "coordinate_transformation_mode",
                Attribute::String(b"half_pixel_symmetric".to_vec()),
            ),
        ];
        let nearest = mode == "nearest";
        if nearest {
            attrs.push(("nearest_mode", Attribute::String(b"floor".to_vec())));
        }
        if let Some(axes) = axes {
            attrs.push(("axes", Attribute::Ints(axes)));
        }
        Case {
            label: label.into(),
            op: "Resize",
            domain: "",
            opset: 19,
            inputs,
            outputs: vec![(DataType::Float32, output_shape)],
            attrs,
            compare: if nearest {
                Compare::ExactBytes
            } else {
                Compare::Float { tol: 1e-5 }
            },
        }
    };
    // ONNX backend test vectors from commit 622adf94d4e720142950aa8178f7357d52e63db0.
    vec![
        case(
            "Resize[linear,half-pixel-symmetric,scales,downsample]",
            float_input(DataType::Float32, &[1, 1, 1, 4], &[1., 2., 3., 4.]),
            input(DataType::Float32, &[4], &[1.0f32, 1., 1., 0.6]),
            vec![1, 1, 1, 2],
            "linear",
            None,
            false,
        ),
        case(
            "Resize[linear,half-pixel-symmetric,scales,upsample]",
            float_input(DataType::Float32, &[1, 1, 2, 2], &[1., 2., 3., 4.]),
            input(DataType::Float32, &[4], &[1.0f32, 1., 2.3, 2.94]),
            vec![1, 1, 4, 5],
            "linear",
            None,
            false,
        ),
        case(
            "Resize[nearest,half-pixel-symmetric,scales,negative-axis]",
            float_input(
                DataType::Float32,
                &[2, 4],
                &[1., 2., 3., 4., 5., 6., 7., 8.],
            ),
            input(DataType::Float32, &[1], &[0.6f32]),
            vec![2, 2],
            "nearest",
            Some(vec![-1]),
            false,
        ),
        case(
            "Resize[linear,half-pixel-symmetric,sizes,reordered-axes]",
            float_input(DataType::Float32, &[2, 1, 3], &[1., 2., 3., 4., 5., 6.]),
            input(DataType::Int64, &[2], &[5i64, 1]),
            vec![1, 1, 5],
            "linear",
            Some(vec![-1, 0]),
            true,
        ),
    ]
}

fn conv_transpose_cases() -> Vec<Case> {
    vec![
        Case {
            label: "ConvTranspose[1d,stride,dilation,output-padding,asymmetric-pads]".into(),
            op: "ConvTranspose",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(DataType::Float32, &[1, 1, 3], &[1.0, -2.0, 3.0]),
                float_input(DataType::Float32, &[1, 1, 2], &[0.5, 2.0]),
                float_input(DataType::Float32, &[1], &[0.25]),
            ],
            outputs: vec![(DataType::Float32, vec![1, 1, 7])],
            attrs: vec![
                ("strides", Attribute::Ints(vec![2])),
                ("dilations", Attribute::Ints(vec![2])),
                ("pads", Attribute::Ints(vec![1, 0])),
                ("output_padding", Attribute::Ints(vec![1])),
            ],
            compare: Compare::Float { tol: 1e-5 },
        },
        Case {
            label: "ConvTranspose[2d,depthwise,stride,asymmetric-pads]".into(),
            op: "ConvTranspose",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(
                    DataType::Float16,
                    &[1, 2, 2, 2],
                    &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0],
                ),
                float_input(
                    DataType::Float16,
                    &[2, 1, 2, 2],
                    &[1.0, 0.5, -1.0, 2.0, 0.25, -0.5, 1.5, 1.0],
                ),
            ],
            outputs: vec![(DataType::Float16, vec![1, 2, 2, 3])],
            attrs: vec![
                ("group", Attribute::Int(2)),
                ("strides", Attribute::Ints(vec![1, 2])),
                ("pads", Attribute::Ints(vec![0, 1, 1, 0])),
            ],
            compare: Compare::Float { tol: 3e-3 },
        },
        Case {
            label: "ConvTranspose[2d,grouped,multiple-output-channels]".into(),
            op: "ConvTranspose",
            domain: "",
            opset: 22,
            inputs: vec![
                float_input(
                    DataType::BFloat16,
                    &[1, 4, 2, 1],
                    &[1.0, 2.0, -1.0, 3.0, 0.5, -2.0, 4.0, 1.0],
                ),
                float_input(
                    DataType::BFloat16,
                    &[4, 2, 1, 2],
                    &[
                        1.0, 0.5, -1.0, 2.0, 0.25, 1.0, 1.5, -0.5, 2.0, 0.5, -0.5, 1.0, 0.75, -1.0,
                        1.25, 0.25,
                    ],
                ),
            ],
            outputs: vec![(DataType::BFloat16, vec![1, 4, 2, 2])],
            attrs: vec![("group", Attribute::Int(2))],
            compare: Compare::Float { tol: 3e-2 },
        },
        Case {
            label: "ConvTranspose[3d,grouped,dilated,asymmetric]".into(),
            op: "ConvTranspose",
            domain: "",
            opset: 22,
            inputs: vec![
                float_input(
                    DataType::Float32,
                    &[1, 2, 2, 1, 2],
                    &[1., 2., 3., 4., -1., -2., -3., -4.],
                ),
                float_input(
                    DataType::Float32,
                    &[2, 1, 2, 2, 1],
                    &[1., 2., 3., 4., 0.5, 1.5, -0.5, -1.5],
                ),
                float_input(DataType::Float32, &[2], &[0.25, 2.]),
            ],
            outputs: vec![(DataType::Float32, vec![1, 2, 4, 2, 3])],
            attrs: vec![
                ("group", Attribute::Int(2)),
                ("strides", Attribute::Ints(vec![2, 1, 2])),
                ("dilations", Attribute::Ints(vec![2, 1, 1])),
                ("pads", Attribute::Ints(vec![1, 0, 0, 1, 0, 1])),
                ("output_padding", Attribute::Ints(vec![1, 0, 1])),
            ],
            compare: Compare::Float { tol: 1e-5 },
        },
    ]
}

fn grid_sample_cases() -> Vec<Case> {
    let image = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let grid = [-1.4, -1.2, 0.0, 0.0, 1.3, 1.4, 0.6, -0.7];
    [
        (16, "bilinear", "zeros", 0, DataType::Float32, 1e-5),
        (20, "linear", "border", 1, DataType::Float16, 3e-3),
        (22, "linear", "reflection", 0, DataType::BFloat16, 3e-2),
        (16, "nearest", "zeros", 1, DataType::Float32, 1e-5),
        (20, "nearest", "border", 0, DataType::Float16, 3e-3),
        (22, "nearest", "reflection", 1, DataType::BFloat16, 3e-2),
    ]
    .into_iter()
    .map(
        |(opset, mode, padding, align_corners, dtype, tolerance)| Case {
            label: format!(
                "GridSample[{mode},{padding},align-corners={align_corners},out-of-bounds]"
            ),
            op: "GridSample",
            domain: "",
            opset,
            inputs: vec![
                float_input(dtype, &[1, 1, 2, 3], &image),
                float_input(dtype, &[1, 2, 2, 2], &grid),
            ],
            outputs: vec![(dtype, vec![1, 1, 2, 2])],
            attrs: vec![
                ("mode", Attribute::String(mode.as_bytes().to_vec())),
                (
                    "padding_mode",
                    Attribute::String(padding.as_bytes().to_vec()),
                ),
                ("align_corners", Attribute::Int(align_corners)),
            ],
            compare: Compare::Float { tol: tolerance },
        },
    )
    .collect()
}

#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]
#[test]
fn grid_sample_ort_plugin_keeps_existing_routes() {
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Node, NodeId, ValueId, static_shape};

    let plugin = CudaExecutionProvider::new_for_ort_plugin().expect("ORT plugin CUDA provider");
    let native = require_cuda();
    let versions = |ep: &CudaExecutionProvider| {
        onnx_runtime_ep_cuda::build_cuda_registry_descriptors(ep.registry())
            .into_iter()
            .filter(|entry| entry.domain.is_empty() && entry.op_type == "GridSample")
            .map(|entry| (entry.since_version, entry.end_version))
            .collect::<Vec<_>>()
    };
    assert_eq!(versions(&plugin), [(16, u64::MAX), (20, u64::MAX)]);
    assert_eq!(
        versions(&native),
        [(16, u64::MAX), (20, u64::MAX), (22, u64::MAX)]
    );

    let f32 = DataType::Float32;
    let image = [
        float_input(f32, &[1, 1, 2, 3], &[1., 2., 3., 4., 5., 6.]),
        float_input(f32, &[1, 1, 1, 2], &[0., 0.]),
    ];
    for (opset, mode) in [(19, "bilinear"), (20, "linear"), (22, "linear")] {
        let output = run_cuda(
            &plugin,
            "GridSample",
            "",
            opset,
            &image,
            &[(f32, vec![1, 1, 1, 1])],
            &[("mode", Attribute::String(mode.into()))],
        );
        assert_eq!(decode_floats(&output[0], f32), [3.5]);
    }

    let image = [static_shape([1, 1, 2, 3]), static_shape([1, 1, 1, 2])];
    let volume = [static_shape([1, 1, 2, 3, 4]), static_shape([1, 1, 1, 1, 3])];
    for (opset, mode, shapes, dtypes, reason) in [
        (19, "bicubic", &image, [f32, f32], "mode"),
        (20, "cubic", &image, [f32, f32], "mode"),
        (20, "nearest", &volume, [f32, f32], "rank 5"),
        (22, "linear", &image, [f32, DataType::BFloat16], "dtype"),
    ] {
        let mut node = Node::new(
            NodeId(0),
            "GridSample",
            vec![Some(ValueId(0)), Some(ValueId(1))],
            vec![ValueId(2)],
        );
        node.attributes
            .insert("mode".into(), Attribute::String(mode.into()));
        assert!(
            native
                .supports_op(&node, opset, shapes, &dtypes, &[])
                .is_supported()
        );
        let claim = plugin.supports_op(&node, opset, shapes, &dtypes, &[]);
        assert!(
            !claim.is_supported(),
            "{mode}/{opset} must remain standalone-only"
        );
        assert!(
            claim.reason().is_some_and(|text| text.contains(reason)),
            "{claim:?}"
        );
    }
}

#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]
#[test]
fn grid_sample_matches_pinned_onnx() {
    let ep = require_cuda();
    let check = |label: &str,
                 opset,
                 mode: &str,
                 padding: &str,
                 aligned,
                 inputs: &[Tensor],
                 dtype,
                 output_shape: &[usize],
                 expected: &[f32]| {
        let output = run_cuda(
            &ep,
            "GridSample",
            "",
            opset,
            inputs,
            &[(dtype, output_shape.to_vec())],
            &[
                ("mode", Attribute::String(mode.into())),
                ("padding_mode", Attribute::String(padding.into())),
                ("align_corners", Attribute::Int(aligned)),
            ],
        );
        assert_close(
            label,
            dtype,
            &decode_floats(&output[0], dtype),
            expected,
            1e-4,
        );
    };
    let f32 = DataType::Float32;
    // ONNX 622adf94d4e720142950aa8178f7357d52e63db0, backend/test/case/node/gridsample.py.
    let image = [
        float_input(f32, &[1, 1, 3, 2], &[0., 1., 2., 3., 4., 5.]),
        float_input(
            f32,
            &[1, 2, 4, 2],
            &[
                -1., -0.8, -0.6, -0.5, -0.1, -0.2, 0.7, 0., 0., 0.4, 0.2, -0.2, -0.3, 0.5, -1., 1.,
            ],
        ),
    ];
    for (opset, mode) in [(19, "bicubic"), (20, "cubic"), (22, "cubic")] {
        for (aligned, expected) in [
            (
                0,
                [
                    -0.173250, 0.284265, 1.923106, 2.568000, 5.170375, 2.284414, 4.744844, 1.046875,
                ],
            ),
            (
                1,
                [
                    0.304001, 1.128750, 2.266270, 3.144844, 4.531500, 2.455360, 4.599819, 4.000000,
                ],
            ),
        ] {
            check(
                &format!("pinned bicubic/{opset}/align={aligned}"),
                opset,
                mode,
                "zeros",
                aligned,
                &image,
                f32,
                &[1, 1, 2, 4],
                &expected,
            );
        }
    }
    let volume = [
        float_input(
            f32,
            &[1, 1, 3, 2, 2],
            &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
        ),
        float_input(
            f32,
            &[1, 2, 4, 2, 3],
            &[
                -1., -1., -1., -1., -0.5, 0.3, -0.5, -0.5, -0.5, 1., -0.6, -1., -0.2, -0.2, -0.2,
                0.4, 0.2, 0.6, 0., 0., 0., -1., 0., 0., 0., 0., 0., -1., 1., 0., -0.2, -0.2, -0.2,
                1., 0.4, -0.2, 0.5, 0.5, 0.5, -1., -0.8, 0.8, 1., 1., 1., 0.4, 0.6, -0.3,
            ],
        ),
    ];
    for (mode, aligned, expected) in [
        (
            "nearest",
            0,
            [
                1., 5., 1., 0., 5., 12., 5., 5., 5., 0., 5., 0., 12., 9., 0., 8.,
            ],
        ),
        (
            "nearest",
            1,
            [
                1., 5., 1., 2., 5., 12., 5., 5., 5., 7., 5., 8., 12., 9., 12., 8.,
            ],
        ),
        (
            "linear",
            0,
            [
                0.125, 3.4, 2., 0.45, 4.7, 10.9, 6.5, 3., 6.5, 1.75, 4.7, 3.3, 11., 2.52, 1.5, 5.49,
            ],
        ),
        (
            "linear",
            1,
            [
                1., 6.7, 3.75, 2.4, 5.4, 9.3, 6.5, 6., 6.5, 7., 5.4, 6.6, 9.25, 8.4, 12., 6.1,
            ],
        ),
    ] {
        check(
            &format!("pinned volume/{mode}/align={aligned}"),
            20,
            mode,
            "zeros",
            aligned,
            &volume,
            f32,
            &[1, 1, 2, 4, 2],
            &expected,
        );
    }

    // An impulse at one corner at the center sample uses the inner cubic weight on all three axes.
    let impulse = [
        float_input(f32, &[1, 1, 2, 2, 2], &[1., 0., 0., 0., 0., 0., 0., 0.]),
        float_input(DataType::Float16, &[1, 1, 1, 1, 3], &[0., 0., 0.]),
    ];
    check(
        "tricubic impulse",
        20,
        "cubic",
        "zeros",
        0,
        &impulse,
        f32,
        &[1, 1, 1, 1, 1],
        &[0.59375f32.powi(3)],
    );

    // At x=-0.5, the four cubic taps are at -2, -1, 0, 1. Border clamps
    // both negatives to 0; half-pixel reflection maps them to 1 and 0.
    let line = [
        float_input(f32, &[1, 1, 3], &[2., 6., 10.]),
        float_input(DataType::BFloat16, &[1, 2, 1], &[-1., -5.]),
    ];
    let border_line = [
        float_input(f32, &[1, 1, 3], &[2., 6., 10.]),
        float_input(DataType::BFloat16, &[1, 4, 1], &[-1., -1.25, 1.25, 1.]),
    ];
    check(
        "cubic line/border",
        22,
        "cubic",
        "border",
        0,
        &border_line,
        f32,
        &[1, 1, 4],
        &[1.625, 2., 10., 10.375],
    );
    check(
        "cubic line/reflection",
        22,
        "cubic",
        "reflection",
        0,
        &line,
        f32,
        &[1, 1, 2],
        &[1.25, 1.25],
    );
}

fn dropout_cases() -> Vec<Case> {
    vec![
        Case {
            label: "Dropout[f32,inference,data+mask]".into(),
            op: "Dropout",
            domain: "",
            opset: 13,
            inputs: vec![input(
                DataType::Float32,
                &[2, 3],
                &[1.0f32, -2.5, 0.0, 4.25, f32::NAN, f32::INFINITY],
            )],
            outputs: vec![
                (DataType::Float32, vec![2, 3]),
                (DataType::Bool, vec![2, 3]),
            ],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Dropout[bf16,inference]".into(),
            op: "Dropout",
            domain: "",
            opset: 13,
            inputs: vec![input(
                DataType::BFloat16,
                &[4],
                &[0xbfe0u16, 0, 0x8000, 0x7fc1],
            )],
            outputs: vec![(DataType::BFloat16, vec![4])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
    ]
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dropout_22_inference_inputs_on_physical_cuda() {
    let ep = require_cuda();
    use DataType::*;
    // Independent byte oracle: include signed zero and NaN payload encodings.
    let x = input(Float8E4M3FN, &[4], &[0x38u8, 0x80, 0x7f, 0xb8]);
    let ratio = input(Float8E5M2, &[], &[0x38u8]);
    let no_ratio = absent_input(Undefined);
    let evaluation = input(Bool, &[], &[0u8]);
    let training = input(Bool, &[], &[1u8]);
    let y = (Float8E4M3FN, vec![4]);
    let mask = (Bool, vec![4]);

    for (inputs, outputs, capture, expected) in [
        (
            vec![x.clone(), ratio.clone()],
            vec![y.clone(), mask.clone()],
            true,
            vec![x.bytes.clone(), vec![1; 4]],
        ),
        (
            vec![x.clone(), no_ratio.clone(), evaluation.clone()],
            vec![y.clone()],
            false,
            vec![x.bytes.clone()],
        ),
        (
            vec![x.clone(), ratio, evaluation],
            vec![y.clone(), mask],
            false,
            vec![x.bytes.clone(), vec![1; 4]],
        ),
    ] {
        let (graph, id) = build_graph("Dropout", "", 22, &inputs, &outputs, &[]);
        let kernel = ep.get_kernel(graph.node(id), &[], 22).unwrap();
        assert_eq!(
            kernel.capture_support(),
            if capture {
                onnx_runtime_ep_api::CaptureSupport::Supported
            } else {
                onnx_runtime_ep_api::CaptureSupport::unsupported(
                    "Dropout reads training_mode on the host; omit training_mode to capture inference",
                )
            }
        );
        assert_eq!(
            run_cuda(&ep, "Dropout", "", 22, &inputs, &outputs, &[]),
            expected
        );
    }

    let error =
        try_run_cuda(&ep, "Dropout", "", 22, &[x, no_ratio, training], &[y], &[]).unwrap_err();
    assert!(error.to_string().contains("training_mode=true"), "{error}");
}

fn instance_normalization_cases() -> Vec<Case> {
    let values = [-3.0, -1.0, 1.0, 3.0, 2.0, 4.0, 6.0, 8.0];
    let mut cases = FLOAT_DTYPES
        .into_iter()
        .map(|dtype| Case {
            label: format!("InstanceNormalization[{dtype:?}]"),
            op: "InstanceNormalization",
            domain: "",
            opset: 6,
            inputs: vec![
                float_input(dtype, &[1, 4, 2], &values),
                float_input(dtype, &[4], &[1.0, 0.5, 1.5, 2.0]),
                float_input(dtype, &[4], &[0.0, 1.0, -1.0, 0.25]),
            ],
            outputs: vec![(dtype, vec![1, 4, 2])],
            attrs: vec![("epsilon", Attribute::Float(1e-5))],
            compare: Compare::Float {
                tol: float_tol(dtype) * 2.0,
            },
        })
        .collect::<Vec<_>>();
    cases.push(Case {
        label: "InstanceNormalization[f32,large-offset]".into(),
        op: "InstanceNormalization",
        domain: "",
        opset: 6,
        inputs: vec![
            float_input(
                DataType::Float32,
                &[1, 1, 4],
                &[10_000.0, 10_001.0, 9_999.0, 10_002.0],
            ),
            float_input(DataType::Float32, &[1], &[1.0]),
            float_input(DataType::Float32, &[1], &[0.0]),
        ],
        outputs: vec![(DataType::Float32, vec![1, 1, 4])],
        attrs: vec![("epsilon", Attribute::Float(1e-5))],
        compare: Compare::Float { tol: 1e-4 },
    });
    cases
}

fn group_normalization_cases() -> Vec<Case> {
    let values = [-3.0, -1.0, 1.0, 3.0, 2.0, 4.0, 6.0, 8.0];
    let mut cases = Vec::new();
    for dtype in FLOAT_DTYPES {
        cases.push(Case {
            label: format!("GroupNormalization-18[{dtype:?}]"),
            op: "GroupNormalization",
            domain: "",
            opset: 18,
            inputs: vec![
                float_input(dtype, &[1, 4, 2], &values),
                float_input(dtype, &[2], &[1.0, 0.5]),
                float_input(dtype, &[2], &[0.25, -0.5]),
            ],
            outputs: vec![(dtype, vec![1, 4, 2])],
            attrs: vec![
                ("num_groups", Attribute::Int(2)),
                ("epsilon", Attribute::Float(1e-5)),
            ],
            compare: Compare::Float {
                tol: float_tol(dtype) * 2.0,
            },
        });
        cases.push(Case {
            label: format!("GroupNormalization-21[{dtype:?}]"),
            op: "GroupNormalization",
            domain: "",
            opset: 21,
            inputs: vec![
                float_input(dtype, &[1, 4, 2], &values),
                float_input(dtype, &[4], &[1.0, 0.5, 1.5, 2.0]),
                float_input(dtype, &[4], &[0.0, 1.0, -1.0, 0.25]),
            ],
            outputs: vec![(dtype, vec![1, 4, 2])],
            attrs: vec![
                ("num_groups", Attribute::Int(2)),
                ("epsilon", Attribute::Float(1e-5)),
                ("stash_type", Attribute::Int(1)),
            ],
            compare: Compare::Float {
                tol: float_tol(dtype) * 2.0,
            },
        });
    }
    cases
}

fn lp_pool_cases() -> Vec<Case> {
    vec![
        Case {
            label: "LpPool[p=1,pad]".into(),
            op: "LpPool",
            domain: "",
            opset: 18,
            inputs: vec![float_input(
                DataType::Float32,
                &[1, 1, 3, 4],
                &[
                    -1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0, 9.0, 10.0, -11.0, 12.0,
                ],
            )],
            outputs: vec![(DataType::Float32, vec![1, 1, 3, 4])],
            attrs: vec![
                ("kernel_shape", Attribute::Ints(vec![2, 2])),
                ("pads", Attribute::Ints(vec![1, 0, 0, 1])),
                ("p", Attribute::Int(1)),
            ],
            compare: Compare::Float { tol: 1e-4 },
        },
        Case {
            label: "LpPool[p=2,stride,ceil]".into(),
            op: "LpPool",
            domain: "",
            opset: 18,
            inputs: vec![float_input(
                DataType::Float16,
                &[1, 1, 4, 5],
                &[
                    1.0, -2.0, 3.0, -4.0, 5.0, 6.0, -7.0, 8.0, -9.0, 10.0, 11.0, -12.0, 13.0,
                    -14.0, 15.0, 16.0, -17.0, 18.0, -19.0, 20.0,
                ],
            )],
            outputs: vec![(DataType::Float16, vec![1, 1, 3, 3])],
            attrs: vec![
                ("kernel_shape", Attribute::Ints(vec![2, 3])),
                ("strides", Attribute::Ints(vec![2, 2])),
                ("pads", Attribute::Ints(vec![1, 0, 1, 1])),
                ("ceil_mode", Attribute::Int(1)),
                ("p", Attribute::Int(2)),
            ],
            compare: Compare::Float { tol: 6e-3 },
        },
        Case {
            label: "LpPool[p=2,dilation,bf16]".into(),
            op: "LpPool",
            domain: "",
            opset: 22,
            inputs: vec![float_input(
                DataType::BFloat16,
                &[1, 1, 4, 4],
                &[
                    1.0, 2.0, 3.0, 4.0, -5.0, -6.0, -7.0, -8.0, 9.0, 10.0, 11.0, 12.0, -13.0,
                    -14.0, -15.0, -16.0,
                ],
            )],
            outputs: vec![(DataType::BFloat16, vec![1, 1, 2, 2])],
            attrs: vec![
                ("kernel_shape", Attribute::Ints(vec![2, 2])),
                ("dilations", Attribute::Ints(vec![2, 2])),
                ("p", Attribute::Int(2)),
            ],
            compare: Compare::Float { tol: 6e-2 },
        },
    ]
}

fn center_crop_pad_cases() -> Vec<Case> {
    vec![
        Case {
            label: "CenterCropPad[crop-pad-odd]".into(),
            op: "CenterCropPad",
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Int32, &[3, 2], &[1i32, 2, 3, 4, 5, 6]),
                input(DataType::Int64, &[2], &[2i64, 5]),
            ],
            outputs: vec![(DataType::Int32, vec![2, 5])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "CenterCropPad[selected-negative-axes]".into(),
            op: "CenterCropPad",
            domain: "",
            opset: 18,
            inputs: vec![
                input(
                    DataType::Int16,
                    &[2, 3, 4],
                    &(0..24).map(|value| value as i16).collect::<Vec<_>>(),
                ),
                input(DataType::Int64, &[2], &[5i64, 3]),
            ],
            outputs: vec![(DataType::Int16, vec![2, 5, 3])],
            attrs: vec![("axes", Attribute::Ints(vec![-2, -1]))],
            compare: Compare::ExactBytes,
        },
    ]
}

fn col2im_cases() -> Vec<Case> {
    vec![
        Case {
            label: "Col2Im[overlap]".into(),
            op: "Col2Im",
            domain: "",
            opset: 18,
            inputs: vec![
                float_input(
                    DataType::Float32,
                    &[1, 4, 4],
                    &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
                ),
                input(DataType::Int64, &[2], &[3i64, 3]),
                input(DataType::Int64, &[2], &[2i64, 2]),
            ],
            outputs: vec![(DataType::Float32, vec![1, 1, 3, 3])],
            attrs: vec![],
            compare: Compare::Float { tol: 1e-4 },
        },
        Case {
            label: "Col2Im[padding-stride]".into(),
            op: "Col2Im",
            domain: "",
            opset: 18,
            inputs: vec![
                float_input(
                    DataType::Float16,
                    &[1, 4, 4],
                    &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
                ),
                input(DataType::Int64, &[2], &[3i64, 3]),
                input(DataType::Int64, &[2], &[2i64, 2]),
            ],
            outputs: vec![(DataType::Float16, vec![1, 1, 3, 3])],
            attrs: vec![
                ("pads", Attribute::Ints(vec![1, 1, 1, 1])),
                ("strides", Attribute::Ints(vec![2, 2])),
            ],
            compare: Compare::Float { tol: 6e-3 },
        },
        Case {
            label: "Col2Im[dilation-overlap]".into(),
            op: "Col2Im",
            domain: "",
            opset: 18,
            inputs: vec![
                float_input(
                    DataType::BFloat16,
                    &[1, 4, 9],
                    &(1..=36).map(|value| value as f32 / 4.0).collect::<Vec<_>>(),
                ),
                input(DataType::Int64, &[2], &[5i64, 5]),
                input(DataType::Int64, &[2], &[2i64, 2]),
            ],
            outputs: vec![(DataType::BFloat16, vec![1, 1, 5, 5])],
            attrs: vec![
                ("dilations", Attribute::Ints(vec![2, 2])),
                ("strides", Attribute::Ints(vec![1, 1])),
            ],
            compare: Compare::Float { tol: 6e-2 },
        },
    ]
}

fn nonzero_cases() -> Vec<Case> {
    vec![
        Case {
            label: "NonZero[f32,rank2]".into(),
            op: "NonZero",
            domain: "",
            opset: 13,
            inputs: vec![input(
                DataType::Float32,
                &[2, 3],
                &[0.0f32, -0.0, 2.5, -1.0, 0.0, 3.0],
            )],
            outputs: vec![(DataType::Int64, vec![2, 3])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "NonZero[f16,rank3]".into(),
            op: "NonZero",
            domain: "",
            opset: 13,
            inputs: vec![float_input(
                DataType::Float16,
                &[2, 2, 2],
                &[0.0, 4.0, 0.0, 0.0, -2.0, 0.0, 7.0, 0.0],
            )],
            outputs: vec![(DataType::Int64, vec![3, 3])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "NonZero[bf16,rank1]".into(),
            op: "NonZero",
            domain: "",
            opset: 13,
            inputs: vec![float_input(
                DataType::BFloat16,
                &[5],
                &[0.0, -0.0, 1.0, f32::NAN, -2.0],
            )],
            outputs: vec![(DataType::Int64, vec![1, 3])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "NonZero[bool,rank2]".into(),
            op: "NonZero",
            domain: "",
            opset: 13,
            inputs: vec![input::<u8>(DataType::Bool, &[2, 3], &[0u8, 1, 1, 0, 0, 1])],
            outputs: vec![(DataType::Int64, vec![2, 3])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
    ]
}

/// Output shape of a reduction over `axes` (negative allowed) honouring
/// `keepdims`.
fn reduce_out_shape(in_shape: &[usize], axes: &[i64], keepdims: bool) -> Vec<usize> {
    let rank = in_shape.len() as i64;
    let reduced: Vec<usize> = axes
        .iter()
        .map(|&a| {
            if a < 0 {
                (a + rank) as usize
            } else {
                a as usize
            }
        })
        .collect();
    let mut out = Vec::new();
    for (axis, &dim) in in_shape.iter().enumerate() {
        if reduced.contains(&axis) {
            if keepdims {
                out.push(1);
            }
        } else {
            out.push(dim);
        }
    }
    out
}

/// An f32 reduction op across a few axes/keepdims combinations. Reductions
/// accumulate in f32; the surviving difference is a few ulp scaled by the
/// reduced magnitude.
fn reduction_f32(op: &'static str) -> Vec<Case> {
    let in_shape = vec![2usize, 3, 4];
    let count: usize = in_shape.iter().product();
    // Mixed positive/negative magnitudes keep max/min/mean well conditioned.
    let values: Vec<f32> = (0..count)
        .map(|v| (v as f32 * 0.37) - 4.0 + (v % 5) as f32 * 0.11)
        .collect();
    let axes_cases: &[(&[i64], bool)] = &[
        (&[1], true),
        (&[1], false),
        (&[0, 2], false),
        (&[-1], false),
    ];
    axes_cases
        .iter()
        .map(|&(axes, keepdims)| {
            let out_shape = reduce_out_shape(&in_shape, axes, keepdims);
            Case {
                label: format!("{op}(axes={axes:?},keepdims={keepdims})"),
                op,
                domain: "",
                opset: 13,
                inputs: vec![input(DataType::Float32, &in_shape, &values)],
                outputs: vec![(DataType::Float32, out_shape)],
                attrs: vec![
                    ("axes", Attribute::Ints(axes.to_vec())),
                    ("keepdims", Attribute::Int(keepdims as i64)),
                ],
                compare: Compare::Float { tol: 2e-2 },
            }
        })
        .collect()
}

/// Extended floating-point reductions widen every input dtype to f32 for the
/// accumulator and post-op, then narrow once to the output dtype.
fn extended_reduction_float(op: &'static str) -> Vec<Case> {
    let in_shape = vec![2usize, 3, 4];
    let count: usize = in_shape.iter().product();
    let positive_only = matches!(op, "ReduceLogSum" | "ReduceProd");
    let values: Vec<f32> = (0..count)
        .map(|index| {
            if positive_only {
                0.25 + (index % 7) as f32 * 0.08
            } else {
                (index as f32 * 0.13) - 1.25 + (index % 3) as f32 * 0.07
            }
        })
        .collect();
    let axes_cases: &[(&[i64], bool)] =
        &[(&[1], true), (&[1], false), (&[0, 2], false), (&[-1], true)];

    FLOAT_DTYPES
        .into_iter()
        .flat_map(|dtype| {
            axes_cases.iter().map({
                let in_shape = in_shape.clone();
                let values = values.clone();
                move |&(axes, keepdims)| Case {
                    label: format!("{op}[{dtype:?}](axes={axes:?},keepdims={keepdims})"),
                    op,
                    domain: "",
                    opset: 17,
                    inputs: vec![float_input(dtype, &in_shape, &values)],
                    outputs: vec![(dtype, reduce_out_shape(&in_shape, axes, keepdims))],
                    attrs: vec![
                        ("axes", Attribute::Ints(axes.to_vec())),
                        ("keepdims", Attribute::Int(keepdims as i64)),
                    ],
                    compare: Compare::Float {
                        tol: float_tol(dtype) * 2.0,
                    },
                }
            })
        })
        .collect()
}

/// `Cast` from f32 to `to`, comparing bytes exactly (CUDA mirrors the CPU
/// truncate-and-saturate / round-to-nearest numerics).
fn cast_case(to: DataType) -> Case {
    let values: Vec<f32> = vec![-3.6, -0.4, 0.0, 1.5, 2.9, 130.0, -130.0, 5.5];
    Case {
        label: format!("Cast[f32->{to:?}]"),
        op: "Cast",
        domain: "",
        opset: 13,
        inputs: vec![float_input(DataType::Float32, &[values.len()], &values)],
        outputs: vec![(to, vec![values.len()])],
        attrs: vec![("to", Attribute::Int(to as i64))],
        compare: Compare::ExactBytes,
    }
}

/// `CastLike` casting f32 input to the second input's dtype (`target`).
fn cast_like_case(target: DataType) -> Case {
    let values: Vec<f32> = vec![-3.6, -0.4, 0.0, 1.5, 2.9, 42.0];
    // The target-type tensor only conveys its dtype; a single element is enough.
    let target_tensor = Tensor {
        dtype: target,
        shape: vec![1],
        bytes: vec![0u8; target.storage_bytes(1)],
        absent: false,
    };
    Case {
        label: format!("CastLike[f32->{target:?}]"),
        op: "CastLike",
        domain: "",
        opset: 15,
        inputs: vec![
            float_input(DataType::Float32, &[values.len()], &values),
            target_tensor,
        ],
        outputs: vec![(target, vec![values.len()])],
        attrs: vec![],
        compare: Compare::ExactBytes,
    }
}

/// Standard-domain `Gemm` cases: `Y = alpha·A'·B' + beta·C` (f32).
fn gemm_cases() -> Vec<Case> {
    let a: Vec<f32> = (0..6).map(|v| v as f32 * 0.5 - 1.0).collect(); // [2,3]
    let b: Vec<f32> = (0..12).map(|v| v as f32 * 0.25 - 0.5).collect(); // [3,4]
    let bias: Vec<f32> = vec![0.1, -0.2, 0.3, -0.4]; // per-N, broadcasts to [2,4]
    let b_t: Vec<f32> = (0..12).map(|v| v as f32 * 0.2 - 0.7).collect(); // [4,3] for transB
    vec![
        // Plain A·B, no bias.
        Case {
            label: "Gemm(no-C)".into(),
            op: "Gemm",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Float32, &[2, 3], &a),
                input(DataType::Float32, &[3, 4], &b),
            ],
            outputs: vec![(DataType::Float32, vec![2, 4])],
            attrs: vec![],
            compare: Compare::Float { tol: 1e-3 },
        },
        // alpha·A·B + beta·C with a broadcast per-N bias.
        Case {
            label: "Gemm(alpha,beta,C)".into(),
            op: "Gemm",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Float32, &[2, 3], &a),
                input(DataType::Float32, &[3, 4], &b),
                input(DataType::Float32, &[4], &bias),
            ],
            outputs: vec![(DataType::Float32, vec![2, 4])],
            attrs: vec![
                ("alpha", Attribute::Float(0.75)),
                ("beta", Attribute::Float(1.0)),
            ],
            compare: Compare::Float { tol: 1e-3 },
        },
        // transB: B stored [N,K] = [4,3].
        Case {
            label: "Gemm(transB)".into(),
            op: "Gemm",
            domain: "",
            opset: 13,
            inputs: vec![
                input(DataType::Float32, &[2, 3], &a),
                input(DataType::Float32, &[4, 3], &b_t),
            ],
            outputs: vec![(DataType::Float32, vec![2, 4])],
            attrs: vec![("transB", Attribute::Int(1))],
            compare: Compare::Float { tol: 1e-3 },
        },
    ]
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA with gpu-tests enabled"
)]
#[test]
fn gemm_29_shapes_and_values_match_onnx() {
    let ep = require_cuda();
    let a = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.];
    let b = input(DataType::Float32, &[2, 3], &[1.0f32, 0., 1., 0., 1., 0.]);
    let dtype = DataType::Float32;
    let inputs = |shape, a: &[f32]| vec![input(dtype, shape, a), b.clone()];
    let trans_b = ("transB", Attribute::Int(1));
    let trans_a = ("transA", Attribute::Int(1));
    let cases = [
        (
            "vector transA no-op",
            inputs(&[3], &a[..3]),
            vec![2],
            vec![trans_a.clone(), trans_b.clone()],
            vec![4., 2.],
        ),
        (
            "rank-4 flatten",
            inputs(&[2, 1, 2, 3], &a),
            vec![2, 1, 2, 2],
            vec![trans_b.clone()],
            vec![4., 2., 10., 5., 16., 8., 22., 11.],
        ),
        (
            "rank-4 transpose and batch bias",
            {
                let mut v = inputs(&[2, 1, 3, 2], &a);
                v.push(input(dtype, &[2, 1, 1, 2], &[10.0f32, 20., 30., 40.]));
                v
            },
            vec![2, 1, 2, 2],
            vec![
                trans_a,
                trans_b.clone(),
                ("alpha", Attribute::Float(0.25)),
                ("beta", Attribute::Float(0.5)),
            ],
            vec![6.5, 10.75, 7., 11., 19.5, 22.25, 20., 22.5],
        ),
        (
            "empty batch",
            inputs(&[0, 3, 3], &[]),
            vec![0, 3, 2],
            vec![trans_b],
            vec![],
        ),
        (
            "empty features with bias",
            vec![
                input::<f32>(dtype, &[2, 2, 0], &[]),
                input::<f32>(dtype, &[0, 2], &[]),
                input(dtype, &[2, 1, 1], &[2.0f32, 4.]),
            ],
            vec![2, 2, 2],
            vec![],
            vec![2., 2., 2., 2., 4., 4., 4., 4.],
        ),
    ];
    for (label, inputs, shape, attrs, expected) in cases {
        let case = Case {
            label: label.into(),
            op: "Gemm",
            domain: "",
            opset: 29,
            inputs,
            outputs: vec![(dtype, shape)],
            attrs,
            compare: Compare::Float { tol: 1e-6 },
        };
        let actual = run_case(&ep, &case);
        assert_close(
            label,
            dtype,
            &decode_floats(&actual[0], dtype),
            &expected,
            1e-6,
        );
    }
}

/// `com.microsoft::SkipLayerNormalization` (fused residual add + layernorm),
/// f32, exercising the optional `beta` and `bias` slots. Only the required `Y`
/// output is requested.
fn skip_layer_norm_case() -> Case {
    let rows = 2usize;
    let hidden = 4usize;
    let count = rows * hidden;
    let x: Vec<f32> = (0..count).map(|v| (v as f32 * 0.3) - 1.0).collect();
    let skip: Vec<f32> = (0..count).map(|v| (v as f32 * 0.1) - 0.5).collect();
    let gamma: Vec<f32> = vec![1.2, 0.8, 1.0, 0.5];
    let beta: Vec<f32> = vec![0.05, -0.1, 0.2, 0.0];
    let bias: Vec<f32> = vec![0.01, 0.02, -0.03, 0.04];
    Case {
        label: "SkipLayerNormalization[f32]".into(),
        op: "SkipLayerNormalization",
        domain: "com.microsoft",
        opset: 1,
        inputs: vec![
            input(DataType::Float32, &[rows, hidden], &x),
            input(DataType::Float32, &[rows, hidden], &skip),
            input(DataType::Float32, &[hidden], &gamma),
            input(DataType::Float32, &[hidden], &beta),
            input(DataType::Float32, &[hidden], &bias),
        ],
        outputs: vec![(DataType::Float32, vec![rows, hidden])],
        attrs: vec![("epsilon", Attribute::Float(1e-5))],
        compare: Compare::Float { tol: 2e-3 },
    }
}

/// `Not`: element-wise boolean negation.
fn not_case() -> Case {
    let values: Vec<u8> = vec![1, 0, 1, 0, 0, 1];
    Case {
        label: "Not[bool]".into(),
        op: "Not",
        domain: "",
        opset: 1,
        inputs: vec![input(DataType::Bool, &[values.len()], &values)],
        outputs: vec![(DataType::Bool, vec![values.len()])],
        attrs: vec![],
        compare: Compare::ExactBytes,
    }
}

/// Bitwise binary parity cases (`BitwiseAnd`/`BitwiseOr`/`BitwiseXor`) over a
/// signed and an unsigned dtype, including a broadcast case. Integer results are
/// compared byte-exactly against the CPU oracle (`bitwise.rs`).
fn bitwise_binary_cases(op: &'static str) -> Vec<Case> {
    let a32: Vec<i32> = vec![0x0f0f, -1, 0x1234, 0, 0x7fff_ffff, -256];
    let b32: Vec<i32> = vec![0x00ff, 0x0f0f, -1, 123, 0x0f0f_0f0f, 0xff];
    let a8: Vec<u8> = vec![0xf0, 0x3c, 0x55, 0x81];
    let b8: Vec<u8> = vec![0x0f, 0x33, 0xaa, 0x81];
    vec![
        Case {
            label: format!("{op}[i32]"),
            op,
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Int32, &[6], &a32),
                input(DataType::Int32, &[6], &b32),
            ],
            outputs: vec![(DataType::Int32, vec![6])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: format!("{op}[u8]"),
            op,
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Uint8, &[4], &a8),
                input(DataType::Uint8, &[4], &b8),
            ],
            outputs: vec![(DataType::Uint8, vec![4])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        // Broadcast [3,1] x [1,4] -> [3,4], u8.
        Case {
            label: format!("{op}[u8,broadcast]"),
            op,
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Uint8, &[3, 1], &[0xf0u8, 0x3c, 0x55]),
                input(DataType::Uint8, &[1, 4], &[0x0fu8, 0x33, 0xaa, 0x81]),
            ],
            outputs: vec![(DataType::Uint8, vec![3, 4])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
    ]
}

/// `BitwiseNot` parity over a signed and an unsigned dtype.
fn bitwise_not_cases() -> Vec<Case> {
    vec![
        Case {
            label: "BitwiseNot[i32]".into(),
            op: "BitwiseNot",
            domain: "",
            opset: 18,
            inputs: vec![input(DataType::Int32, &[5], &[0i32, -1, 0x1234, 255, -256])],
            outputs: vec![(DataType::Int32, vec![5])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "BitwiseNot[u16]".into(),
            op: "BitwiseNot",
            domain: "",
            opset: 18,
            inputs: vec![input(
                DataType::Uint16,
                &[4],
                &[0u16, 0x00ff, 0xf0f0, 0xffff],
            )],
            outputs: vec![(DataType::Uint16, vec![4])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
    ]
}

/// `BitShift` parity: LEFT and RIGHT over unsigned dtypes, a broadcast, and a
/// battery of width-guard overshift rows (amounts `==` and `>` the element
/// width, both directions, every dtype) that lock the CPU
/// `checked_shl`/`checked_shr` "overshift → 0" contract.
///
/// NOTE on the width guard's falsifiability (Bishop mutation probe #2): on GPU5
/// (Ampere) these overshift rows do **not** by themselves fail when the kernel's
/// `(amount >= bits) ? 0` guard is deleted — the hardware already yields `0` for
/// an out-of-range shift, so the guard is *value-redundant on this target*. This
/// was verified three ways (see `.squad/decisions/inbox/deckard-pr288-testfix-67`):
/// (a) SASS shows a non-wrap `SHF.L.U32` funnel shift, which clamps counts `>=`
/// the register width to `0`; (b) an exhaustive device brute force over every
/// overshift `(dtype, value, amount)` found no non-zero raw result — native
/// types clamp, and small types promote to `int` then narrow back to `0`; and
/// (c) a real-kernel neuter (guard deleted) left this whole sweep green. The
/// guard's removal *is* still caught by the source-contract lib unit test
/// `shift_guard_matches_cpu_checked_shift_contract`. These rows remain valuable:
/// they pin the overshift → 0 CPU contract against wrong-width / wrong-direction
/// / narrowing regressions and against ports to any target whose shift does not
/// clamp. The near-boundary in-range rows below additionally fail if the guard
/// threshold is mutated too low (e.g. `>= bits - 1`), which the hardware does
/// not mask.
fn bitshift_cases() -> Vec<Case> {
    let dir = |d: &str| ("direction", Attribute::String(d.as_bytes().to_vec()));
    vec![
        Case {
            label: "BitShift[u32,LEFT]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Uint32, &[5], &[1u32, 3, 8, 0x8000_0000, 255]),
                input(DataType::Uint32, &[5], &[0u32, 1, 4, 1, 3]),
            ],
            outputs: vec![(DataType::Uint32, vec![5])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "BitShift[u8,RIGHT,overshift]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            // Shift of 8 on a u8 must produce 0 (>= width).
            inputs: vec![
                input(DataType::Uint8, &[4], &[0x80u8, 0xff, 0x10, 0x01]),
                input(DataType::Uint8, &[4], &[1u8, 4, 8, 0]),
            ],
            outputs: vec![(DataType::Uint8, vec![4])],
            attrs: vec![dir("RIGHT")],
            compare: Compare::ExactBytes,
        },
        // Broadcast [2,1] x [1,3] -> [2,3], u16 LEFT.
        Case {
            label: "BitShift[u16,LEFT,broadcast]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Uint16, &[2, 1], &[3u16, 8]),
                input(DataType::Uint16, &[1, 3], &[0u16, 1, 2]),
            ],
            outputs: vec![(DataType::Uint16, vec![2, 3])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
        // ── Width-guard overshift cases (CPU checked-shift contract lock) ──
        //
        // Any amount `>=` the operand width must yield `0`, matching the CPU
        // `checked_shl`/`checked_shr(amount as u32)` contract (`None -> 0`). The
        // kernel enforces this with `(amount >= bits) ? 0`. On GPU5 the *raw* C
        // shift ALSO yields 0 for these (SASS `SHF` clamps native-type overshift
        // to 0; small types promote to `int` and narrow back to 0), so deleting
        // the guard does not change these outputs on this target — see the
        // function docstring. They are kept as a portable regression lock on the
        // overshift → 0 contract (amount `==` width AND `>` width, LEFT & RIGHT,
        // every dtype). The prior sole overshift case (`u8 >> 8`) already could
        // not diverge: under int promotion `(int)0x10 >> 8 == 0`.
        //
        // `amount == width` AND `amount > width`, for both LEFT and RIGHT.
        Case {
            label: "BitShift[u32,LEFT,overshift>=width]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Uint32, &[4], &[1u32, 3, 0x00FF_00FF, 7]),
                input(DataType::Uint32, &[4], &[32u32, 40, 33, 64]),
            ],
            outputs: vec![(DataType::Uint32, vec![4])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "BitShift[u32,RIGHT,overshift>=width]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(
                    DataType::Uint32,
                    &[4],
                    &[0x8000_0000u32, 0xFFFF_FFFF, 0x00FF_00FF, 7],
                ),
                input(DataType::Uint32, &[4], &[32u32, 40, 33, 64]),
            ],
            outputs: vec![(DataType::Uint32, vec![4])],
            attrs: vec![dir("RIGHT")],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "BitShift[u64,LEFT,overshift>=width]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(
                    DataType::Uint64,
                    &[3],
                    &[1u64, 0xFFFF_FFFF_FFFF_FFFF, 0x1234_5678],
                ),
                input(DataType::Uint64, &[3], &[64u64, 100, 64]),
            ],
            outputs: vec![(DataType::Uint64, vec![3])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
        // Small types promote to `int` before the shift and narrow back on
        // store; an overshift therefore lands as 0 (guard or not) on GPU5. Kept
        // to pin `u16`/`u8` overshift → 0 across LEFT and RIGHT.
        Case {
            label: "BitShift[u16,LEFT,overshift>width]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Uint16, &[3], &[0xFFFFu16, 0xF0F0, 0x0F0F]),
                input(DataType::Uint16, &[3], &[32u16, 40, 33]),
            ],
            outputs: vec![(DataType::Uint16, vec![3])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "BitShift[u8,LEFT,overshift>width]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Uint8, &[3], &[0xFFu8, 0x0F, 0x03]),
                input(DataType::Uint8, &[3], &[32u8, 33, 34]),
            ],
            outputs: vec![(DataType::Uint8, vec![3])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
        // ── Near-boundary IN-RANGE rows (guard-threshold falsifiability) ──
        //
        // Max valid shift `amount == width - 1` produces a non-zero result. The
        // hardware does NOT mask these (they are legal shifts), so they FAIL if
        // the guard threshold is mutated one too low (e.g. `amount >= bits - 1`),
        // which would wrongly zero a valid shift — the one guard-boundary bug
        // that IS behaviorally observable on this target. Both directions.
        Case {
            label: "BitShift[u32,LEFT,max-valid=31]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Uint32, &[3], &[1u32, 3, 0x0000_00FF]),
                input(DataType::Uint32, &[3], &[31u32, 31, 24]),
            ],
            outputs: vec![(DataType::Uint32, vec![3])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "BitShift[u32,RIGHT,max-valid=31]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(
                    DataType::Uint32,
                    &[3],
                    &[0x8000_0000u32, 0xFFFF_FFFF, 0xFF00_0000],
                ),
                input(DataType::Uint32, &[3], &[31u32, 31, 24]),
            ],
            outputs: vec![(DataType::Uint32, vec![3])],
            attrs: vec![dir("RIGHT")],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "BitShift[u8,LEFT,max-valid=7]".into(),
            op: "BitShift",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Uint8, &[3], &[1u8, 3, 0x03]),
                input(DataType::Uint8, &[3], &[7u8, 6, 5]),
            ],
            outputs: vec![(DataType::Uint8, vec![3])],
            attrs: vec![dir("LEFT")],
            compare: Compare::ExactBytes,
        },
    ]
}

/// `LogSoftmax` parity, f32/f16/bf16, opset-13 last-axis and legacy opset-11
/// coerce-to-2D, including a large-magnitude row to exercise the stable
/// shifted-logsumexp path (the #266 overflow lesson).
fn log_softmax_cases() -> Vec<Case> {
    // Mix of ordinary and large-magnitude logits (last row) so a naive
    // log(sum(exp(x))) would overflow while the stable path stays finite.
    let values: Vec<f32> = vec![
        1.0, 2.0, 3.0, 0.5, //
        -1.0, 0.0, 1.0, 2.0, //
        80.0, 79.0, 78.0, 81.0,
    ];
    let shape = vec![3usize, 4];
    let mut cases = Vec::new();
    for dtype in FLOAT_DTYPES {
        cases.push(Case {
            label: format!("LogSoftmax[{dtype:?},opset13]"),
            op: "LogSoftmax",
            domain: "",
            opset: 13,
            inputs: vec![float_input(dtype, &shape, &values)],
            outputs: vec![(dtype, shape.clone())],
            attrs: vec![("axis", Attribute::Int(-1))],
            compare: Compare::Float {
                tol: float_tol(dtype),
            },
        });
    }
    // Legacy opset-11 coerce-to-2D over a rank-3 input, f32.
    cases.push(Case {
        label: "LogSoftmax[Float32,opset11,coerce2d]".into(),
        op: "LogSoftmax",
        domain: "",
        opset: 11,
        inputs: vec![input(
            DataType::Float32,
            &[2, 2, 2],
            &[1.0f32, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        )],
        outputs: vec![(DataType::Float32, vec![2, 2, 2])],
        attrs: vec![("axis", Attribute::Int(1))],
        compare: Compare::Float { tol: 1e-4 },
    });
    // Interior-axis reduction (inner > 1), f32.
    cases.push(Case {
        label: "LogSoftmax[Float32,axis1,inner]".into(),
        op: "LogSoftmax",
        domain: "",
        opset: 13,
        inputs: vec![input(
            DataType::Float32,
            &[2, 3, 4],
            &(0..24).map(|v| (v as f32) * 0.3 - 3.0).collect::<Vec<_>>(),
        )],
        outputs: vec![(DataType::Float32, vec![2, 3, 4])],
        attrs: vec![("axis", Attribute::Int(1))],
        compare: Compare::Float { tol: 1e-4 },
    });
    // Overflow-stability proof (Bishop mutation probe / the #266 lesson).
    //
    // Logits reach 100, and the reduction widens every dtype to f32 before the
    // `exp`. A *naive* `log(sum(exp(x)))` therefore evaluates `exp(100) ~ 2.7e43`,
    // which overflows f32's ~3.4e38 range to `+inf` and drives the result to
    // `-inf`/`nan` — diverging hard from the CPU oracle. Only the stable
    // shifted-logsumexp (`x - max` before `exp`) stays finite here, so DELETING
    // the kernel's max-subtraction makes this row FAIL. The earlier ~81 logit
    // stayed inside f32 range (`exp(81) ~ 1.6e35 < 3.4e38`), which is exactly why
    // it could not falsify the guard.
    //
    // The stable outputs are `[0, -100, -200]`, all exactly representable in
    // f32/f16/bf16, so the CUDA-vs-CPU parity is byte-tight despite the large
    // magnitudes. Every float dtype widens to f32 for the reduction, so a naive
    // path overflows for all three and each row bites independently.
    let overflow_logits: Vec<f32> = vec![100.0, 0.0, -100.0];
    let overflow_shape = vec![1usize, 3];
    for dtype in FLOAT_DTYPES {
        cases.push(Case {
            label: format!("LogSoftmax[{dtype:?},overflow-stability]"),
            op: "LogSoftmax",
            domain: "",
            opset: 13,
            inputs: vec![float_input(dtype, &overflow_shape, &overflow_logits)],
            outputs: vec![(dtype, overflow_shape.clone())],
            attrs: vec![("axis", Attribute::Int(-1))],
            compare: Compare::Float {
                tol: float_tol(dtype),
            },
        });
    }
    cases
}

/// `Hardmax` parity, f32/f16/bf16: one-hot of the first maximum, including a tie
/// (first index wins) and an interior negative axis. Outputs are canonical
/// `0`/`1` so bytes match the CPU oracle exactly.
fn hardmax_cases() -> Vec<Case> {
    // Row 1 has a tie at indices 1 and 2 (both 5.0) -> index 1 must win.
    let tie_values: Vec<f32> = vec![1.0, 5.0, 5.0, 2.0, 7.0, 3.0, 6.0, 4.0];
    let mut cases = Vec::new();
    for dtype in FLOAT_DTYPES {
        cases.push(Case {
            label: format!("Hardmax[{dtype:?},last-axis,tie]"),
            op: "Hardmax",
            domain: "",
            opset: 13,
            inputs: vec![float_input(dtype, &[2, 4], &tie_values)],
            outputs: vec![(dtype, vec![2, 4])],
            attrs: vec![("axis", Attribute::Int(-1))],
            compare: Compare::ExactBytes,
        });
    }
    // Interior negative axis on a rank-3 input, f32.
    cases.push(Case {
        label: "Hardmax[Float32,axis-2]".into(),
        op: "Hardmax",
        domain: "",
        opset: 13,
        inputs: vec![input(
            DataType::Float32,
            &[2, 3, 2],
            &[
                1.0f32, 4.0, 3.0, 2.0, 3.0, 5.0, 6.0, 1.0, 6.0, 2.0, 1.0, 3.0,
            ],
        )],
        outputs: vec![(DataType::Float32, vec![2, 3, 2])],
        attrs: vec![("axis", Attribute::Int(-2))],
        compare: Compare::ExactBytes,
    });
    cases
}

/// Fused-GELU parity (`com.microsoft` `BiasGelu`/`FastGelu`), one case per
/// float dtype. `X` is `[2,4]` with a `[4]` bias broadcast over the last dim;
/// values span negative / zero / positive so both GELU branches are exercised.
/// The GELU is evaluated in `double` on both sides, so f32 stays within a few
/// ulp of the CPU oracle.
fn bias_gelu_cases(op: &'static str) -> Vec<Case> {
    let x = vec![-3.0f32, -0.7, 0.0, 0.5, 1.2, 2.5, -1.5, 3.0];
    let bias = vec![0.1f32, -0.2, 0.3, -0.4];
    FLOAT_DTYPES
        .into_iter()
        .map(|dtype| Case {
            label: format!("{op}[{dtype:?}]"),
            op,
            domain: "com.microsoft",
            opset: 1,
            inputs: vec![
                float_input(dtype, &[2, 4], &x),
                float_input(dtype, &[4], &bias),
            ],
            outputs: vec![(dtype, vec![2, 4])],
            attrs: vec![],
            compare: Compare::Float {
                tol: float_tol(dtype),
            },
        })
        .collect()
}

/// `com.microsoft::FastGelu` parity: the bias-free arity (single input) across
/// every float dtype, plus an f32 `-inf → 0` guard case.
fn fast_gelu_no_bias_cases() -> Vec<Case> {
    let x = vec![-3.0f32, -0.7, 0.0, 0.5, 1.2, 2.5, -1.5, 3.0];
    let mut cases: Vec<Case> = FLOAT_DTYPES
        .into_iter()
        .map(|dtype| Case {
            label: format!("FastGelu[{dtype:?},no-bias]"),
            op: "FastGelu",
            domain: "com.microsoft",
            opset: 1,
            inputs: vec![float_input(dtype, &[2, 4], &x)],
            outputs: vec![(dtype, vec![2, 4])],
            attrs: vec![],
            compare: Compare::Float {
                tol: float_tol(dtype),
            },
        })
        .collect();
    // The `-inf` element must map to 0 (matches the CPU EP's explicit guard);
    // if the guard were dropped, `0.5·(-inf)·(1+tanh(-inf))` is NaN and the
    // float comparison against the CPU's 0 fails.
    cases.push(Case {
        label: "FastGelu[Float32,neg-inf-guard]".into(),
        op: "FastGelu",
        domain: "com.microsoft",
        opset: 1,
        inputs: vec![float_input(
            DataType::Float32,
            &[4],
            &[f32::NEG_INFINITY, -1.0, 0.0, 2.0],
        )],
        outputs: vec![(DataType::Float32, vec![4])],
        attrs: vec![],
        compare: Compare::Float { tol: 1e-4 },
    });
    cases
}

/// `com.microsoft::QuickGelu` parity: `X·sigmoid(alpha·X)`, one case per float
/// dtype at the default alpha plus an explicit non-default alpha (f32).
fn quick_gelu_cases() -> Vec<Case> {
    let x = vec![-4.0f32, -1.0, -0.2, 0.0, 0.3, 1.5, 3.0, 5.0];
    let mut cases: Vec<Case> = FLOAT_DTYPES
        .into_iter()
        .map(|dtype| Case {
            label: format!("QuickGelu[{dtype:?}]"),
            op: "QuickGelu",
            domain: "com.microsoft",
            opset: 1,
            inputs: vec![float_input(dtype, &[2, 4], &x)],
            outputs: vec![(dtype, vec![2, 4])],
            attrs: vec![],
            compare: Compare::Float {
                tol: float_tol(dtype),
            },
        })
        .collect();
    cases.push(Case {
        label: "QuickGelu[Float32,alpha=1.5]".into(),
        op: "QuickGelu",
        domain: "com.microsoft",
        opset: 1,
        inputs: vec![float_input(DataType::Float32, &[2, 4], &x)],
        outputs: vec![(DataType::Float32, vec![2, 4])],
        attrs: vec![("alpha", Attribute::Float(1.5))],
        compare: Compare::Float { tol: 1e-4 },
    });
    cases
}

fn bitcast_cases() -> Vec<Case> {
    use DataType::*;
    [
        (Int32, Float32, vec![2], vec![1, 0, 192, 127, 0, 0, 0, 128]),
        (Uint8, Bool, vec![3], vec![0, 1, 255]),
        (Uint4, Float4E2M1, vec![3], vec![0x21, 0x03]),
        (Uint2, Int2, vec![5], vec![0xe4, 0x03]),
        (Complex128, Complex128, vec![], (0..16).collect()),
        (Uint64, Float64, vec![0, 4], vec![]),
    ]
    .into_iter()
    .map(|(from, to, shape, bytes)| Case {
        label: format!("BitCast[{from:?}->{to:?}]"),
        op: "BitCast",
        domain: "",
        opset: 26,
        inputs: vec![Tensor {
            dtype: from,
            shape: shape.clone(),
            bytes,
            absent: false,
        }],
        outputs: vec![(to, shape)],
        attrs: vec![("to", Attribute::Int(i64::from(to.to_onnx())))],
        compare: Compare::ExactBytes,
    })
    .collect()
}

/// `CumProd` parity across numeric dtypes, directions, axes, and scan modes.
fn cumprod_cases() -> Vec<Case> {
    let f = |label: &str,
             shape: &[usize],
             values: &[f32],
             axis: i64,
             attrs: Vec<(&'static str, Attribute)>|
     -> Case {
        Case {
            label: label.to_string(),
            op: "CumProd",
            domain: "",
            opset: 26,
            inputs: vec![
                float_input(DataType::Float32, shape, values),
                input(DataType::Int64, &[], &[axis]),
            ],
            outputs: vec![(DataType::Float32, shape.to_vec())],
            attrs,
            compare: Compare::Float { tol: 1e-4 },
        }
    };
    let row = vec![1.0f32, 2.0, 3.0, -1.0, 0.5, 4.0];
    let mut cases = vec![
        f("CumProd[f32,axis1]", &[2, 3], &row, 1, vec![]),
        f(
            "CumProd[f32,axis0,exclusive]",
            &[2, 3],
            &row,
            0,
            vec![("exclusive", Attribute::Int(1))],
        ),
        f(
            "CumProd[f32,axis-1,reverse]",
            &[2, 3],
            &row,
            -1,
            vec![("reverse", Attribute::Int(1))],
        ),
        f(
            "CumProd[f32,axis1,3d]",
            &[2, 2, 2],
            &[1.0, 2.0, 3.0, 0.5, -2.0, 4.0, 1.5, 2.0],
            1,
            vec![],
        ),
    ];
    // Int64 byte-exact scan (includes a zero and a negative factor).
    let ints = vec![1_i64, 2, 3, -1, 5, 0];
    cases.push(Case {
        label: "CumProd[i64,axis1,reverse]".into(),
        op: "CumProd",
        domain: "",
        opset: 26,
        inputs: vec![
            input(DataType::Int64, &[2, 3], &ints),
            input(DataType::Int64, &[], &[1_i64]),
        ],
        outputs: vec![(DataType::Int64, vec![2, 3])],
        attrs: vec![("reverse", Attribute::Int(1))],
        compare: Compare::ExactBytes,
    });
    for dtype in [
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float64,
        DataType::Int32,
        DataType::Uint32,
        DataType::Uint64,
    ] {
        let shape = vec![2, 5];
        let values = match dtype {
            DataType::Float16 | DataType::BFloat16 => float_input(dtype, &shape, &[1.13; 10]),
            DataType::Float64 => input(dtype, &shape, &[1.13_f64; 10]),
            DataType::Int32 => input(dtype, &shape, &[i32::MAX, 2, -3, 5, 0, -5, 4, 3, 2, 1]),
            DataType::Uint32 => input(dtype, &shape, &[u32::MAX, 2, 3, 5, 0, 5, 4, 3, 2, 1]),
            DataType::Uint64 => input(dtype, &shape, &[u64::MAX, 2, 3, 5, 0, 5, 4, 3, 2, 1]),
            _ => unreachable!(),
        };
        for (exclusive, reverse) in [(0, 0), (1, 1)] {
            cases.push(Case {
                label: format!("CumProd[{dtype:?},axis-1,exclusive={exclusive},reverse={reverse}]"),
                op: "CumProd",
                domain: "",
                opset: 26,
                inputs: vec![values.clone(), input(DataType::Int32, &[], &[-1_i32])],
                outputs: vec![(dtype, shape.clone())],
                attrs: vec![
                    ("exclusive", Attribute::Int(exclusive)),
                    ("reverse", Attribute::Int(reverse)),
                ],
                compare: Compare::ExactBytes,
            });
        }
    }
    cases
}

/// `ArgMax`/`ArgMin` parity vs the CPU oracle (byte-exact Int64 indices) over
/// keepdims/axis variants, dtypes, and the `select_last_index` tie-break.
fn arg_reduce_cases(op: &'static str) -> Vec<Case> {
    let in_shape = vec![2usize, 3, 4];
    let count: usize = in_shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|v| ((v * 7 + 3) % 11) as f32 - 5.0)
        .collect();
    let mut cases = Vec::new();
    for (axis, keepdims) in [(1i64, true), (1, false), (-1, false), (0, true)] {
        let out_shape = reduce_out_shape(&in_shape, &[axis], keepdims);
        cases.push(Case {
            label: format!("{op}(axis={axis},keepdims={keepdims})"),
            op,
            domain: "",
            opset: 13,
            inputs: vec![input(DataType::Float32, &in_shape, &values)],
            outputs: vec![(DataType::Int64, out_shape)],
            attrs: vec![
                ("axis", Attribute::Int(axis)),
                ("keepdims", Attribute::Int(keepdims as i64)),
            ],
            compare: Compare::ExactBytes,
        });
    }
    // Narrow float dtype paths (widened to f32 for comparison, like the CPU
    // oracle). Keep both declared CUDA input dtypes exercised.
    for dtype in [DataType::Float16, DataType::BFloat16] {
        cases.push(Case {
            label: format!("{op}[{dtype:?},axis-1]"),
            op,
            domain: "",
            opset: 13,
            inputs: vec![float_input(dtype, &[2, 4], &values[..8])],
            outputs: vec![(DataType::Int64, vec![2])],
            attrs: vec![
                ("axis", Attribute::Int(-1)),
                ("keepdims", Attribute::Int(0)),
            ],
            compare: Compare::ExactBytes,
        });
    }
    // Tie-break: repeated extremal values along the axis. Default keeps the
    // first index; select_last_index keeps the last — a falsifiable guard on
    // the tie rule (the two cases produce different bytes).
    let tie = vec![3.0f32, 5.0, 5.0, 2.0, 5.0, 5.0, 1.0, 0.0];
    for select_last in [0i64, 1] {
        cases.push(Case {
            label: format!("{op}[tie,select_last={select_last}]"),
            op,
            domain: "",
            opset: 13,
            inputs: vec![input(DataType::Float32, &[2, 4], &tie)],
            outputs: vec![(DataType::Int64, vec![2])],
            attrs: vec![
                ("axis", Attribute::Int(-1)),
                ("keepdims", Attribute::Int(0)),
                ("select_last_index", Attribute::Int(select_last)),
            ],
            compare: Compare::ExactBytes,
        });
    }
    cases
}

fn gather_nd_cases() -> Vec<Case> {
    vec![
        Case {
            label: "GatherND[i64-indices,negative-index]".into(),
            op: "GatherND",
            domain: "",
            opset: 13,
            inputs: vec![
                input(
                    DataType::Float32,
                    &[2, 3, 2],
                    &(0..12).map(|value| value as f32).collect::<Vec<_>>(),
                ),
                input(DataType::Int64, &[3, 2], &[0_i64, 1, 1, -1, -1, 0]),
            ],
            outputs: vec![(DataType::Float32, vec![3, 2])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "GatherND[i32-indices,batch-dims]".into(),
            op: "GatherND",
            domain: "",
            opset: 29,
            inputs: vec![
                input(
                    DataType::Int64,
                    &[2, 2, 3],
                    &(0_i64..12).collect::<Vec<_>>(),
                ),
                input(DataType::Int32, &[2, 2, 1], &[0_i32, 1, 1, 0]),
            ],
            outputs: vec![(DataType::Int64, vec![2, 2, 3])],
            attrs: vec![("batch_dims", Attribute::Int(1))],
            compare: Compare::ExactBytes,
        },
    ]
}

fn space_to_depth_cases() -> Vec<Case> {
    vec![
        Case {
            label: "SpaceToDepth[f32,multiple-channels]".into(),
            op: "SpaceToDepth",
            domain: "",
            opset: 13,
            inputs: vec![input(
                DataType::Float32,
                &[1, 2, 4, 4],
                &(0..32).map(|value| value as f32).collect::<Vec<_>>(),
            )],
            outputs: vec![(DataType::Float32, vec![1, 8, 2, 2])],
            attrs: vec![("blocksize", Attribute::Int(2))],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "SpaceToDepth[i64,empty-batch]".into(),
            op: "SpaceToDepth",
            domain: "",
            opset: 13,
            inputs: vec![input(DataType::Int64, &[0, 1, 2, 2], &[] as &[i64])],
            outputs: vec![(DataType::Int64, vec![0, 4, 1, 1])],
            attrs: vec![("blocksize", Attribute::Int(2))],
            compare: Compare::ExactBytes,
        },
    ]
}

fn eye_like_cases() -> Vec<Case> {
    let mut cases = vec![Case {
        label: "EyeLike[f32,negative-offset]".into(),
        op: "EyeLike",
        domain: "",
        opset: 9,
        inputs: vec![input(DataType::Float32, &[4, 3], &[7.0_f32; 12])],
        outputs: vec![(DataType::Float32, vec![4, 3])],
        attrs: vec![("k", Attribute::Int(-1))],
        compare: Compare::ExactBytes,
    }];
    for dtype in [
        DataType::Bool,
        DataType::Int8,
        DataType::Int16,
        DataType::Int32,
        DataType::Int64,
        DataType::Uint8,
        DataType::Uint16,
        DataType::Uint32,
        DataType::Uint64,
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float64,
    ] {
        cases.push(Case {
            label: format!("EyeLike[{dtype:?}-override,positive-offset]"),
            op: "EyeLike",
            domain: "",
            opset: if dtype == DataType::BFloat16 { 22 } else { 9 },
            inputs: vec![input(DataType::Float32, &[3, 4], &[9.0_f32; 12])],
            outputs: vec![(dtype, vec![3, 4])],
            attrs: vec![
                ("k", Attribute::Int(1)),
                ("dtype", Attribute::Int(dtype as i64)),
            ],
            compare: Compare::ExactBytes,
        });
    }
    cases.push(Case {
        label: "EyeLike[bool-override,empty-rows]".into(),
        op: "EyeLike",
        domain: "",
        opset: 9,
        inputs: vec![input(DataType::Float32, &[0, 3], &[] as &[f32])],
        outputs: vec![(DataType::Bool, vec![0, 3])],
        attrs: vec![("dtype", Attribute::Int(DataType::Bool as i64))],
        compare: Compare::ExactBytes,
    });
    cases
}

fn pad_cases() -> Vec<Case> {
    vec![
        Case {
            label: "Pad[f32,constant,broadcast-axes]".into(),
            op: "Pad",
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Float32, &[2, 2], &[1.0_f32, 2.0, 3.0, 4.0]),
                input(DataType::Int64, &[2], &[1_i64, 2]),
                input(DataType::Float32, &[], &[9.0_f32]),
                input(DataType::Int64, &[1], &[-1_i64]),
            ],
            outputs: vec![(DataType::Float32, vec![2, 5])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Pad[i64,negative-crop]".into(),
            op: "Pad",
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Int64, &[5], &[10_i64, 20, 30, 40, 50]),
                input(DataType::Int64, &[2], &[-1_i64, -1]),
            ],
            outputs: vec![(DataType::Int64, vec![3])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Pad[f16,reflect]".into(),
            op: "Pad",
            domain: "",
            opset: 18,
            inputs: vec![
                float_input(DataType::Float16, &[4], &[1.0, 2.0, 3.0, 4.0]),
                input(DataType::Int64, &[2], &[2_i64, 2]),
            ],
            outputs: vec![(DataType::Float16, vec![8])],
            attrs: vec![("mode", Attribute::String("reflect".into()))],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Pad[bf16,wrap]".into(),
            op: "Pad",
            domain: "",
            opset: 18,
            inputs: vec![
                float_input(DataType::BFloat16, &[3], &[1.0, -2.0, 3.5]),
                input(DataType::Int64, &[2], &[2_i64, 1]),
            ],
            outputs: vec![(DataType::BFloat16, vec![6])],
            attrs: vec![("mode", Attribute::String("wrap".into()))],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Pad[uint8,edge]".into(),
            op: "Pad",
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Uint8, &[3], &[2_u8, 4, 8]),
                input(DataType::Int64, &[2], &[2_i64, 1]),
            ],
            outputs: vec![(DataType::Uint8, vec![6])],
            attrs: vec![("mode", Attribute::String("edge".into()))],
            compare: Compare::ExactBytes,
        },
    ]
}

fn range_cases() -> Vec<Case> {
    let mut cases = vec![
        Case {
            label: "Range[i64,negative-delta]".into(),
            op: "Range",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Int64, &[], &[5_i64]),
                input(DataType::Int64, &[], &[-2_i64]),
                input(DataType::Int64, &[], &[-2_i64]),
            ],
            outputs: vec![(DataType::Int64, vec![4])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "Range[f32,fractional]".into(),
            op: "Range",
            domain: "",
            opset: 11,
            inputs: vec![
                input(DataType::Float32, &[], &[-0.5_f32]),
                input(DataType::Float32, &[], &[1.0_f32]),
                input(DataType::Float32, &[], &[0.25_f32]),
            ],
            outputs: vec![(DataType::Float32, vec![6])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
    ];
    for dtype in [DataType::Float16, DataType::BFloat16] {
        cases.push(Case {
            label: format!("Range[{dtype:?},fractional]"),
            op: "Range",
            domain: "",
            opset: 11,
            inputs: vec![
                float_input(dtype, &[], &[-1.0]),
                float_input(dtype, &[], &[1.0]),
                float_input(dtype, &[], &[0.5]),
            ],
            outputs: vec![(dtype, vec![4])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        });
    }
    cases
}

fn scatter_nd_cases() -> Vec<Case> {
    vec![
        Case {
            label: "ScatterND[f32,slice-update,negative-index]".into(),
            op: "ScatterND",
            domain: "",
            opset: 11,
            inputs: vec![
                input(
                    DataType::Float32,
                    &[3, 3],
                    &[1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                ),
                input(DataType::Int64, &[2, 1], &[-1_i64, 0]),
                input(
                    DataType::Float32,
                    &[2, 3],
                    &[10.0_f32, 11.0, 12.0, 20.0, 21.0, 22.0],
                ),
            ],
            outputs: vec![(DataType::Float32, vec![3, 3])],
            attrs: vec![],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "ScatterND[f16,duplicate-index,add]".into(),
            op: "ScatterND",
            domain: "",
            opset: 16,
            inputs: vec![
                float_input(DataType::Float16, &[3], &[10.0, 20.0, 30.0]),
                input(DataType::Int64, &[3, 1], &[1_i64, 1, -3]),
                float_input(DataType::Float16, &[3], &[2.0, 3.0, 4.0]),
            ],
            outputs: vec![(DataType::Float16, vec![3])],
            attrs: vec![("reduction", Attribute::String(b"add".to_vec()))],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "ScatterND[bf16,duplicate-index,max]".into(),
            op: "ScatterND",
            domain: "",
            opset: 18,
            inputs: vec![
                float_input(DataType::BFloat16, &[3], &[10.0, 20.0, 30.0]),
                input(DataType::Int64, &[3, 1], &[1_i64, 1, -3]),
                float_input(DataType::BFloat16, &[3], &[12.0, 25.0, 40.0]),
            ],
            outputs: vec![(DataType::BFloat16, vec![3])],
            attrs: vec![("reduction", Attribute::String(b"max".to_vec()))],
            compare: Compare::ExactBytes,
        },
        Case {
            label: "ScatterND[i64,slice-update,mul]".into(),
            op: "ScatterND",
            domain: "",
            opset: 18,
            inputs: vec![
                input(DataType::Int64, &[2, 2], &[2_i64, 3, 4, 5]),
                input(DataType::Int64, &[2, 1], &[0_i64, -1]),
                input(DataType::Int64, &[2, 2], &[10_i64, 20, 2, 3]),
            ],
            outputs: vec![(DataType::Int64, vec![2, 2])],
            attrs: vec![("reduction", Attribute::String(b"mul".to_vec()))],
            compare: Compare::ExactBytes,
        },
    ]
}

fn tensor_scatter_cases() -> Vec<Case> {
    // Distinct batch values and offsets catch cross-batch writes; the
    // circular case wraps only the first batch.
    [("linear", [0_i64, 2]), ("circular", [3, 1])]
        .into_iter()
        .map(|(mode, offsets)| Case {
            label: format!("TensorScatter_{mode}_batch_offsets"),
            op: "TensorScatter",
            domain: "",
            opset: 24,
            inputs: vec![
                input(DataType::Float32, &[2, 4], &[-1.0_f32; 8]),
                input(DataType::Float32, &[2, 2], &[1.0_f32, 2.0, 3.0, 4.0]),
                input(DataType::Int64, &[2], &offsets),
            ],
            outputs: vec![(DataType::Float32, vec![2, 4])],
            attrs: vec![
                ("axis", Attribute::Int(-1)),
                ("mode", Attribute::String(mode.as_bytes().to_vec())),
            ],
            compare: Compare::ExactBytes,
        })
        .collect()
}

fn window_cases(op: &'static str) -> Vec<Case> {
    [
        (DataType::Float32, true, 1e-6),
        (DataType::Float16, false, 3e-3),
        (DataType::BFloat16, true, 3e-2),
        (DataType::Float64, false, 1e-6),
    ]
    .into_iter()
    .map(|(dtype, periodic, tol)| Case {
        label: format!("{op}[{dtype:?},periodic={periodic}]"),
        op,
        domain: "",
        opset: 17,
        inputs: vec![input(DataType::Int64, &[], &[7_i64])],
        outputs: vec![(dtype, vec![7])],
        attrs: vec![
            ("periodic", Attribute::Int(i64::from(periodic))),
            ("output_datatype", Attribute::Int(dtype as i64)),
        ],
        compare: Compare::Float { tol },
    })
    .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// The conformance profile: one entry per CUDA_COVERED_OPS op.
// ─────────────────────────────────────────────────────────────────────────────

/// Convenience for a `Dedicated` entry.
fn dedicated(op: &'static str, suite: &'static str, note: &'static str) -> ProfileEntry {
    ProfileEntry {
        op,
        coverage: Coverage::Dedicated { suite, note },
    }
}

/// Convenience for a `Sweep` entry.
fn sweep(op: &'static str, cases: Vec<Case>) -> ProfileEntry {
    ProfileEntry {
        op,
        coverage: Coverage::Sweep(cases),
    }
}

/// The full conformance profile. Every op in [`CUDA_COVERED_OPS`] appears
/// exactly once. Inline `Sweep` entries carry new CUDA-vs-CPU parity cases
/// (concentrated on ops that previously had *no* parity test at all); the rest
/// are attributed to their dedicated GPU suite.
fn conformance_profile() -> Vec<ProfileEntry> {
    let mut p = Vec::with_capacity(CUDA_COVERED_OPS.len());

    // ── Inline parity sweep (previously untested ops get real coverage) ──────
    // Unary math.
    p.push(sweep(
        "Sqrt",
        unary_float("Sqrt", 13, &[0.0, 0.25, 1.0, 2.25, 9.0]),
    ));
    p.push(sweep(
        "Erf",
        unary_float("Erf", 13, &[-1.5, -0.5, 0.0, 0.7, 1.8]),
    ));
    p.push(sweep(
        "Tanh",
        unary_float("Tanh", 13, &[-2.0, -0.5, 0.0, 0.9, 2.5]),
    ));
    p.push(sweep(
        "Sigmoid",
        unary_float("Sigmoid", 13, &[-3.0, -0.7, 0.0, 0.8, 3.0]),
    ));
    p.push(sweep(
        "Abs",
        unary_float("Abs", 13, &[-2.5, -0.5, 0.0, 1.4, 2.5]),
    ));
    p.push(sweep(
        "Neg",
        unary_float("Neg", 13, &[-2.5, -0.5, 0.0, 1.4, 2.5]),
    ));
    p.push(sweep(
        "Reciprocal",
        unary_float("Reciprocal", 13, &[-2.0, -0.5, 0.5, 1.0, 4.0]),
    ));
    p.push(sweep(
        "Log",
        unary_float("Log", 13, &[0.1, 0.5, 1.0, 2.0, 5.0]),
    ));
    p.push(sweep(
        "Sign",
        unary_float("Sign", 13, &[-2.5, -0.5, 0.0, 1.4, 2.5]),
    ));
    p.push(sweep(
        "Floor",
        unary_float("Floor", 13, &[-2.5, -0.5, 0.0, 1.4, 2.6]),
    ));
    p.push(sweep(
        "Ceil",
        unary_float("Ceil", 13, &[-2.5, -0.5, 0.0, 1.4, 2.6]),
    ));
    p.push(sweep(
        "Round",
        unary_float("Round", 13, &[-2.5, -1.5, 0.0, 1.5, 2.6]),
    ));
    p.push(sweep(
        "Sin",
        unary_float("Sin", 22, &[-3.0, -1.0, 0.0, 1.0, 3.0]),
    ));
    p.push(sweep(
        "Cos",
        unary_float("Cos", 22, &[-3.0, -1.0, 0.0, 1.0, 3.0]),
    ));
    p.push(sweep(
        "Softplus",
        unary_float("Softplus", 13, &[-2.0, -0.5, 0.0, 1.0, 3.0]),
    ));

    // Binary math.
    p.push(sweep(
        "Pow",
        binary_float(
            "Pow",
            13,
            &[0.5, 1.0, 2.0, 3.0, 4.0],
            &[5],
            &[2.0, 0.5, 3.0, 1.0, 2.0],
            &[5],
            &[5],
        ),
    ));
    p.push(sweep(
        "Min",
        binary_float(
            "Min",
            13,
            &[-1.0, 2.0, 0.5, -3.0, 4.0],
            &[5],
            &[0.5, -2.0, 1.5, -1.0, 4.0],
            &[5],
            &[5],
        ),
    ));
    p.push(sweep(
        "Max",
        binary_float(
            "Max",
            13,
            &[-1.0, 2.0, 0.5, -3.0, 4.0],
            &[5],
            &[0.5, -2.0, 1.5, -1.0, 4.0],
            &[5],
            &[5],
        ),
    ));

    // Reductions.
    p.push(sweep("ReduceMean", reduction_f32("ReduceMean")));
    p.push(sweep("ReduceMax", reduction_f32("ReduceMax")));
    p.push(sweep("ReduceMin", reduction_f32("ReduceMin")));

    // Cast / CastLike.
    p.push(sweep(
        "Cast",
        vec![
            cast_case(DataType::Int32),
            cast_case(DataType::Int64),
            cast_case(DataType::Float16),
            cast_case(DataType::BFloat16),
        ],
    ));
    p.push(sweep(
        "CastLike",
        vec![
            cast_like_case(DataType::Int32),
            cast_like_case(DataType::Float16),
        ],
    ));

    // Logical, GEMM, fused norm.
    p.push(sweep("Not", vec![not_case()]));
    p.push(sweep("Gemm", gemm_cases()));
    p.push(sweep(
        "SkipLayerNormalization",
        vec![skip_layer_norm_case()],
    ));

    // Batch 4 (issue #67): integer bitwise + softmax-family axis reductions.
    p.push(sweep("BitwiseAnd", bitwise_binary_cases("BitwiseAnd")));
    p.push(sweep("BitwiseOr", bitwise_binary_cases("BitwiseOr")));
    p.push(sweep("BitwiseXor", bitwise_binary_cases("BitwiseXor")));
    p.push(sweep("BitwiseNot", bitwise_not_cases()));
    p.push(sweep("BitShift", bitshift_cases()));
    p.push(sweep("LogSoftmax", log_softmax_cases()));
    p.push(sweep("Hardmax", hardmax_cases()));

    // Batch 5 (issue #67): fused GELU activations, cumulative product, and the
    // index reductions ArgMax/ArgMin.
    p.push(sweep("BiasGelu", bias_gelu_cases("BiasGelu")));
    p.push(sweep("FastGelu", {
        let mut c = bias_gelu_cases("FastGelu");
        c.extend(fast_gelu_no_bias_cases());
        c
    }));
    p.push(sweep("QuickGelu", quick_gelu_cases()));
    p.push(sweep("CumProd", cumprod_cases()));
    p.push(sweep("BitCast", bitcast_cases()));
    p.push(sweep("ArgMax", arg_reduce_cases("ArgMax")));
    p.push(sweep("ArgMin", arg_reduce_cases("ArgMin")));

    // Batch 6 (issue #67): model-shaping/indexing structural operators.
    p.push(sweep("GatherND", gather_nd_cases()));
    p.push(sweep("SpaceToDepth", space_to_depth_cases()));
    p.push(dedicated(
        "DepthToSpace",
        "cuda_conformance_gpu.rs",
        "Independent DCR/CRD value fixtures in space_depth_modes_match_onnx",
    ));
    p.push(sweep("EyeLike", eye_like_cases()));

    // Batch 7 (issue #67): general padding/cropping and sequence construction.
    p.push(sweep("Pad", pad_cases()));
    p.push(sweep("Range", range_cases()));

    // Batch 8 (issue #67): indexed updates and signal-processing windows.
    p.push(sweep("ScatterND", scatter_nd_cases()));
    for op in ["HannWindow", "HammingWindow", "BlackmanWindow"] {
        p.push(sweep(op, window_cases(op)));
    }

    // Batch 9 (issue #67): per-tensor quantization, inference Dropout, and
    // data-dependent coordinate extraction.
    p.push(sweep("QuantizeLinear", quantize_linear_cases()));
    p.push(sweep("DequantizeLinear", dequantize_linear_cases()));
    p.push(dedicated(
        "DequantizePacked",
        "dequantize_packed_gpu.rs",
        "native device decode against all 27 pinned GGML oracle digests",
    ));
    p.push(sweep("QLinearMatMul", qlinear_matmul_cases()));
    p.push(sweep("Resize", resize_cases()));
    p.push(sweep("ConvTranspose", conv_transpose_cases()));
    p.push(dedicated(
        "RoiAlign",
        "session:roi_align.rs",
        "Pinned ONNX vectors and batch-index validation with CPU fallback disabled",
    ));
    p.push(sweep("GridSample", grid_sample_cases()));
    p.push(dedicated(
        "DeformConv",
        "deform_conv_gpu.rs",
        "Upstream ONNX and independently calculated N-D fixtures on both native EPs",
    ));
    p.push(dedicated(
        "MaxRoiPool",
        "max_roi_pool_gpu.rs",
        "Quantized inclusive ROI pooling, versioned dtypes, and clipping on both native EPs",
    ));
    p.push(sweep("Dropout", dropout_cases()));
    p.push(sweep("NonZero", nonzero_cases()));
    p.push(sweep(
        "InstanceNormalization",
        instance_normalization_cases(),
    ));
    p.push(sweep("GroupNormalization", group_normalization_cases()));
    p.push(dedicated(
        "MeanVarianceNormalization",
        "mean_variance_normalization_gpu.rs",
        "opset 29 native CPU/CUDA against independently calculated goldens",
    ));
    p.push(sweep("LpPool", lp_pool_cases()));
    p.push(sweep("CenterCropPad", center_crop_pad_cases()));
    p.push(sweep("Col2Im", col2im_cases()));

    // ── Dedicated GPU parity suites (verified to name their op) ──────────────
    // Batch 10 (issue #67): normalization, global reductions, quantization, and
    // low-complexity data transforms.
    for op in [
        "AffineGrid",
        "BatchNormalization",
        "Compress",
        "DynamicQuantizeLinear",
        "GlobalAveragePool",
        "GlobalLpPool",
        "GlobalMaxPool",
        "LpNormalization",
    ] {
        p.push(dedicated(
            op,
            "cuda_parity_batch10_gpu.rs",
            "issue #67 CUDA coverage batch 10 CPU parity",
        ));
    }

    p.push(dedicated(
        "GatherBlockQuantized",
        "gather_block_quantized_gpu.rs",
        "blockwise-quantized embedding lookup (issue #67)",
    ));
    p.push(dedicated(
        "CausalConvWithState",
        "causal_conv_with_state_gpu.rs",
        "depthwise causal short-conv with rolling state (issue #67 Qwen3.5 hybrid)",
    ));

    // GEMM / quantized-matmul family.
    p.push(dedicated(
        "MatMul",
        "matmul_gpu.rs",
        "cuBLASLt dense/batched GEMM",
    ));
    p.push(dedicated(
        "Einsum",
        "einsum_gpu.rs",
        "canonical-plan cuBLASLt contractions and zero-copy views",
    ));
    for op in ["SwiGLU", "GeGLU", "Expm1", "Log1p", "LogCumSumExp"] {
        p.push(dedicated(
            op,
            "standard_math_gpu.rs",
            "independent CPU and native CUDA math expectations",
        ));
    }
    for op in ["Pack", "Unpack"] {
        p.push(dedicated(
            op,
            "bit_packing_gpu.rs",
            "pinned opset-29 vectors and input validation on native CPU and CUDA",
        ));
    }
    p.push(dedicated(
        "GroupedMatMul",
        "grouped_matmul_gpu.rs",
        "native CPU and CUDA routing, dtype, bias and input validation vectors",
    ));
    for (op, suite) in [
        ("AffineQuantizedGather", "session:affine_gather.rs"),
        (
            "AffineQuantizedGroupedMatMul",
            "session:quantized_grouped.rs",
        ),
        (
            "DequantizePackedGather",
            "session:dequantize_packed_gather.rs",
        ),
        ("PackedGroupedMatMul", "session:packed_grouped.rs"),
    ] {
        p.push(dedicated(
            op,
            suite,
            "native session graph selection and numerical parity",
        ));
    }
    p.push(dedicated(
        "MatMulNBits",
        "matmul_nbits_gpu.rs",
        "packed INT4 block dequant + GEMM",
    ));
    p.push(dedicated(
        "QMoE",
        "qmoe_gpu.rs",
        "grouped block-dequant expert GEMM",
    ));
    p.push(dedicated(
        "LinearAttention",
        "linear_attention_gpu.rs",
        "gated delta-rule linear attention (Qwen3.5 hybrid)",
    ));
    p.push(dedicated(
        "BlockQuantizedMatMul",
        "block_quantized_matmul_gpu.rs",
        "block-quantized weights",
    ));
    p.push(dedicated(
        "BlockQuantizedMoE",
        "block_quantized_moe_gpu.rs",
        "block-quantized MoE expert GEMV + routing",
    ));
    p.push(dedicated(
        "FusedMatMulBias",
        "fused_epilogue_gpu.rs",
        "cuBLASLt BIAS epilogue",
    ));
    p.push(dedicated(
        "FusedGemm",
        "fused_epilogue_gpu.rs",
        "cuBLASLt BIAS/RELU/GELU epilogue",
    ));

    // Convolution / pooling.
    p.push(dedicated(
        "Conv",
        "conv_gpu.rs",
        "native 1-D / cuDNN 2-D conv",
    ));
    p.push(dedicated("MaxPool", "pooling_gpu.rs", "cuDNN pooling"));
    p.push(dedicated(
        "MaxUnpool",
        "pooling_gpu.rs",
        "NVRTC row-major scatter against pinned ONNX expectations",
    ));
    p.push(dedicated(
        "AveragePool",
        "pooling_gpu.rs",
        "cuDNN and dilated NVRTC pooling",
    ));

    // Attention / KV / sparse family.
    p.push(dedicated(
        "PagedAttention",
        "paged_attention_latent_gpu.rs",
        "latent prefill and decode against the CPU oracle",
    ));
    p.push(dedicated(
        "Attention",
        "standard_attention_gpu.rs",
        "flash-style attention",
    ));
    p.push(dedicated(
        "GroupQueryAttention",
        "group_query_attention_gpu.rs",
        "GQA with KV cache",
    ));
    p.push(dedicated(
        "MultiHeadAttention",
        "multi_head_attention_gpu.rs",
        "separate-QKV MHA with bias / mask / KV cache",
    ));
    p.push(dedicated(
        "VarlenAttention",
        "varlen_attention_gpu.rs",
        "variable-length attention",
    ));
    p.push(dedicated(
        "PackedVarlenAttention",
        "packed_varlen_attention_gpu.rs",
        "packed varlen attention",
    ));
    p.push(dedicated(
        "CompressedSparseAttention",
        "compressed_sparse_attention_gpu.rs",
        "CSA",
    ));
    p.push(dedicated(
        "SparseKvGather",
        "sparse_kv_gather_gpu.rs",
        "sparse KV gather",
    ));
    p.push(dedicated(
        "IndexShare",
        "index_share_gpu.rs",
        "shared index buffer",
    ));
    p.push(dedicated(
        "DsaIndexSelect",
        "dsa_index_select_gpu.rs",
        "query-dependent sparse index selection (DSA indexer)",
    ));
    p.push(dedicated(
        "RotaryEmbedding",
        "rope_capture_gpu.rs",
        "RoPE, graph-capture safe",
    ));
    p.push(dedicated(
        "KvCacheCapacityAppend",
        "kv_cache_capacity_append_gpu.rs",
        "S3 capacity-emission KV append, graph-capture safe",
    ));

    // Normalization / softmax.
    p.push(dedicated(
        "LayerNormalization",
        "normalization_fp16_gpu.rs",
        "fused layernorm",
    ));
    p.push(dedicated(
        "SimplifiedLayerNormalization",
        "simplified_layer_norm_gpu.rs",
        "RMS-style norm",
    ));
    p.push(dedicated(
        "SkipSimplifiedLayerNormalization",
        "skip_simplified_layer_norm_gpu.rs",
        "fused residual + RMS norm",
    ));
    p.push(dedicated(
        "RMSNormalization",
        "normalization_fp16_gpu.rs",
        "RMS norm",
    ));
    p.push(dedicated("Softmax", "indexing_gpu.rs", "row softmax"));

    // Activations (attribute-driven).
    p.push(dedicated(
        "Relu",
        "pointwise_gpu.rs",
        "elementwise activation",
    ));
    p.push(dedicated(
        "Gelu",
        "fused_epilogue_gpu.rs",
        "GELU activation",
    ));
    p.push(dedicated("Silu", "activations_gpu.rs", "SiLU activation"));
    p.push(dedicated("LeakyRelu", "activations_gpu.rs", "alpha-driven"));
    p.push(dedicated("Elu", "activations_gpu.rs", "alpha-driven"));
    p.push(dedicated(
        "Celu",
        "activations_gpu.rs",
        "scalar oracle parity across alpha values and NaN inputs",
    ));
    p.push(dedicated(
        "Mish",
        "activations_gpu.rs",
        "scalar oracle parity including the saturating tail",
    ));
    p.push(dedicated(
        "HardSigmoid",
        "activations_gpu.rs",
        "alpha/beta-driven",
    ));
    p.push(dedicated("Clip", "activations_gpu.rs", "min/max clamp"));
    p.push(dedicated("Softsign", "activations_gpu.rs", "x/(1+|x|)"));
    p.push(dedicated(
        "Selu",
        "activations_gpu.rs",
        "alpha/gamma-driven",
    ));

    // Elementwise binary / comparison / logical.
    p.push(dedicated("Add", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Sub", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Mul", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Div", "pointwise_gpu.rs", "broadcasting binary"));
    p.push(dedicated("Exp", "pointwise_gpu.rs", "unary math"));
    p.push(dedicated("And", "pointwise_gpu.rs", "boolean binary"));
    p.push(dedicated("Or", "pointwise_gpu.rs", "boolean binary"));
    p.push(dedicated("Xor", "pointwise_gpu.rs", "boolean binary"));
    p.push(dedicated("Equal", "pointwise_gpu.rs", "comparison -> bool"));
    p.push(dedicated(
        "Greater",
        "pointwise_gpu.rs",
        "comparison -> bool",
    ));
    p.push(dedicated("Less", "pointwise_gpu.rs", "comparison -> bool"));
    p.push(dedicated(
        "GreaterOrEqual",
        "pointwise_gpu.rs",
        "comparison -> bool",
    ));
    p.push(dedicated(
        "LessOrEqual",
        "pointwise_gpu.rs",
        "comparison -> bool",
    ));

    // Trigonometric / hyperbolic family (batch 1).
    for op in [
        "Tan", "Sinh", "Cosh", "Asin", "Acos", "Atan", "Asinh", "Acosh", "Atanh",
    ] {
        p.push(dedicated(
            op,
            "op_coverage_batch_gpu.rs",
            "trig/hyperbolic unary parity",
        ));
    }

    // Extended reductions / activations / variadic / misc (batches 2 & 3).
    for op in [
        "ReduceProd",
        "ReduceSumSquare",
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
    ] {
        p.push(ProfileEntry {
            op,
            coverage: Coverage::Sweep(extended_reduction_float(op)),
        });
    }
    p.push(dedicated(
        "Swish",
        "op_coverage_batch_gpu.rs",
        "extended activation",
    ));
    p.push(dedicated(
        "ThresholdedRelu",
        "op_coverage_batch_gpu.rs",
        "extended activation",
    ));
    p.push(dedicated(
        "Sum",
        "op_coverage_batch_gpu.rs",
        "variadic elementwise",
    ));
    p.push(dedicated(
        "Mean",
        "op_coverage_batch_gpu.rs",
        "variadic elementwise",
    ));
    p.push(dedicated("Mod", "op_coverage_batch_gpu.rs", "modulo"));
    p.push(dedicated(
        "IsInf",
        "op_coverage_batch_gpu.rs",
        "unary predicate",
    ));
    p.push(dedicated(
        "IsNaN",
        "op_coverage_batch_gpu.rs",
        "unary predicate",
    ));
    p.push(dedicated(
        "PRelu",
        "op_coverage_batch_gpu.rs",
        "parametric ReLU",
    ));
    p.push(dedicated(
        "Identity",
        "op_coverage_batch_gpu.rs",
        "byte copy",
    ));
    p.push(dedicated("Flatten", "op_coverage_batch_gpu.rs", "reshape"));
    p.push(dedicated(
        "Size",
        "op_coverage_batch_gpu.rs",
        "element count scalar",
    ));
    p.push(dedicated(
        "Trilu",
        "op_coverage_batch_gpu.rs",
        "triangular mask",
    ));

    // Reduction / metadata / movement covered elsewhere.
    p.push(dedicated("ReduceSum", "movement_gpu.rs", "reduction"));
    p.push(dedicated(
        "DFT",
        "dft_gpu.rs",
        "cuFFT real/complex forward/inverse and arbitrary-axis parity",
    ));
    p.push(dedicated(
        "NonMaxSuppression",
        "nms_gpu.rs",
        "CPU parity and bounded device workspace",
    ));
    p.push(dedicated(
        "Unique",
        "../src/kernels/unique.rs",
        "flattened CUDA Unique CPU parity and dynamic output transfer contract",
    ));
    p.push(dedicated(
        "STFT",
        "stft_gpu.rs",
        "batched cuFFT framed/windowed real and complex parity",
    ));
    p.push(dedicated("Gather", "movement_gpu.rs", "indexed gather"));
    p.push(dedicated("Shape", "movement_gpu.rs", "shape metadata"));
    p.push(dedicated(
        "Constant",
        "movement_gpu.rs",
        "constant materialize",
    ));
    p.push(dedicated(
        "ConstantOfShape",
        "opset24_ops_gpu.rs",
        "fill from shape",
    ));
    p.push(dedicated("OneHot", "opset24_ops_gpu.rs", "one-hot encode"));

    // Structural data-movement.
    for op in [
        "Concat",
        "Expand",
        "Reshape",
        "Slice",
        "Split",
        "Squeeze",
        "Tile",
        "Transpose",
        "Unsqueeze",
        "Where",
    ] {
        p.push(dedicated(op, "construction_gpu.rs", "shape/data movement"));
    }

    // Indexed / scan.
    p.push(dedicated("TopK", "indexing_gpu.rs", "top-k"));
    p.push(dedicated("CumSum", "indexing_gpu.rs", "prefix scan"));
    p.push(dedicated(
        "GatherElements",
        "indexing_gpu.rs",
        "gather along axis",
    ));
    p.push(dedicated(
        "ScatterElements",
        "indexing_gpu.rs",
        "scatter along axis",
    ));
    p.push(sweep("TensorScatter", tensor_scatter_cases()));

    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Coverage-of-coverage audits (ignored without gpu-tests; require CUDA when active)
// ─────────────────────────────────────────────────────────────────────────────

/// Every op the CUDA EP claims to cover must have a conformance profile entry,
/// and no profile entry may reference an op that is no longer covered.
///
/// This is the highest-value guard: it fails the moment an op is added to
/// `CUDA_COVERED_OPS` without a corresponding parity test — the "claimed but
/// untested" defect class (e.g. the `ReduceLogSumExp` and bf16 misses).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn every_covered_op_has_a_conformance_entry() {
    let _ep = require_cuda();
    let profile = conformance_profile();
    let profile_ops: HashSet<&str> = profile.iter().map(|e| e.op).collect();
    let covered: HashSet<&str> = CUDA_COVERED_OPS.iter().copied().collect();

    let missing: Vec<&str> = CUDA_COVERED_OPS
        .iter()
        .copied()
        .filter(|op| !profile_ops.contains(op))
        .collect();
    assert!(
        missing.is_empty(),
        "these CUDA_COVERED_OPS have no conformance profile entry (claimed but \
         untested — add them to conformance_profile()): {missing:?}"
    );

    let stale: Vec<&str> = profile_ops
        .iter()
        .copied()
        .filter(|op| !covered.contains(op))
        .collect();
    assert!(
        stale.is_empty(),
        "these profile entries reference ops not in CUDA_COVERED_OPS \
         (stale — remove or fix the entry): {stale:?}"
    );

    assert_eq!(
        profile.len(),
        CUDA_COVERED_OPS.len(),
        "profile must have exactly one entry per covered op"
    );
}

/// No op may appear twice in the profile, and every `Sweep` entry must carry at
/// least one case.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn profile_has_no_duplicate_entries() {
    let _ep = require_cuda();
    let profile = conformance_profile();
    let mut seen = HashSet::new();
    for entry in &profile {
        assert!(
            seen.insert(entry.op),
            "duplicate conformance profile entry for {:?}",
            entry.op
        );
        if let Coverage::Sweep(cases) = &entry.coverage {
            assert!(
                !cases.is_empty(),
                "Sweep entry for {:?} has no cases",
                entry.op
            );
        }
    }
}

/// Each `Dedicated` suite file must exist and actually name its op, so a
/// deleted, renamed, or gutted suite cannot silently leave an op unverified
/// while it stays in `CUDA_COVERED_OPS`.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn dedicated_suites_exist_and_name_their_op() {
    let _ep = require_cuda();
    let tests_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
    for entry in conformance_profile() {
        if let Coverage::Dedicated { suite, note } = entry.coverage {
            let path = match suite.strip_prefix("session:") {
                Some(file) => tests_dir
                    .join("..")
                    .join("..")
                    .join("onnx-runtime-session")
                    .join("tests")
                    .join(file),
                None => tests_dir.join(suite),
            };
            let src = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!(
                    "dedicated suite {suite} for op {:?} could not be read at {}: {error}",
                    entry.op,
                    path.display()
                )
            });
            let needle = format!("\"{}\"", entry.op);
            assert!(
                src.contains(&needle),
                "dedicated suite {suite} does not name op {:?} (looked for {needle}); \
                 the coverage claim is stale",
                entry.op
            );
            eprintln!("{:>32}  ✓ {suite}  ({note})", entry.op);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// GPU parity sweep (ignored unless `gpu-tests` is enabled)
// ─────────────────────────────────────────────────────────────────────────────

/// Run one inline case on CUDA and compare to the CPU oracle.
fn run_case(ep: &CudaExecutionProvider, case: &Case) -> Vec<Vec<u8>> {
    let cuda = run_cuda(
        ep,
        case.op,
        case.domain,
        case.opset,
        &case.inputs,
        &case.outputs,
        &case.attrs,
    );
    let cpu = run_cpu(
        case.op,
        case.domain,
        case.opset,
        &case.inputs,
        &case.outputs,
        &case.attrs,
    );
    assert_eq!(
        cuda.len(),
        cpu.len(),
        "{}: output count mismatch (CUDA {} vs CPU {})",
        case.label,
        cuda.len(),
        cpu.len()
    );
    match case.compare {
        Compare::ExactBytes => {
            for (index, (got, want)) in cuda.iter().zip(&cpu).enumerate() {
                assert_eq!(
                    got, want,
                    "{}: output {index} bytes differ (CUDA vs CPU oracle)",
                    case.label
                );
            }
        }
        Compare::Float { tol } => {
            for (index, ((got, want), (dtype, _))) in
                cuda.iter().zip(&cpu).zip(&case.outputs).enumerate()
            {
                let got_f = decode_floats(got, *dtype);
                let want_f = decode_floats(want, *dtype);
                assert_close(
                    &format!("{} out{index}", case.label),
                    *dtype,
                    &got_f,
                    &want_f,
                    tol,
                );
            }
        }
    }
    cuda
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn conv_transpose_native_cuda_conformance() {
    let ep = require_cuda();
    for case in conv_transpose_cases() {
        run_case(&ep, &case);
    }

    // Both SAME modes crop an odd cell on opposite sides in height, and
    // extend an odd cell on opposite sides in width (effective kernel < stride).
    let inputs = [
        float_input(DataType::Float32, &[1, 1, 1, 1], &[2.]),
        float_input(DataType::Float32, &[1, 1, 3, 1], &[1., 2., 3.]),
    ];
    for (mode, expected) in [
        ("SAME_UPPER", [0., 2., 0., 4.]),
        ("SAME_LOWER", [4., 0., 6., 0.]),
    ] {
        let outputs = run_cuda(
            &ep,
            "ConvTranspose",
            "",
            22,
            &inputs,
            &[(DataType::Float32, vec![1, 1, 2, 2])],
            &[
                ("strides", Attribute::Ints(vec![2, 2])),
                ("auto_pad", Attribute::String(mode.as_bytes().to_vec())),
            ],
        );
        assert_eq!(
            decode_floats(&outputs[0], DataType::Float32),
            expected,
            "{mode}"
        );
    }

    // Explicit output_shape ignores supplied pads and still uses output_padding
    // in the split. DOUBLE values here are deliberately not representable in f32.
    let x = 1.0_f64 + 2.0_f64.powi(-40);
    let outputs = run_cuda(
        &ep,
        "ConvTranspose",
        "",
        22,
        &[
            input(DataType::Float64, &[1, 1, 2], &[x, -2.0 * x]),
            input(DataType::Float64, &[1, 1, 2], &[1.0_f64, 0.5]),
        ],
        &[(DataType::Float64, vec![1, 1, 4])],
        &[
            ("strides", Attribute::Ints(vec![2])),
            ("output_shape", Attribute::Ints(vec![4])),
            ("output_padding", Attribute::Ints(vec![1])),
            ("pads", Attribute::Ints(vec![0, 2])),
        ],
    );
    assert_eq!(
        outputs[0],
        [0.5 * x, -2.0 * x, -x, 0.0].map(f64::to_ne_bytes).concat(),
        "explicit output_shape must shift the DOUBLE contribution without narrowing"
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn resize_19_symmetric_matches_onnx_and_cpu() {
    let ep = require_cuda();
    let cases = resize_symmetric_cases();
    let expected: [&[f32]; 4] = [
        &[1.6666667, 3.3333333],
        &[
            1., 1.159864, 1.5, 1.840136, 2., 1.5652174, 1.7250813, 2.0652174, 2.4053535, 2.5652175,
            2.4347825, 2.5946467, 2.9347825, 3.2749186, 3.4347825, 3., 3.159864, 3.5, 3.840136, 4.,
        ],
        &[1., 3., 5., 7.],
        &[2.5, 2.9, 3.5, 4.1, 4.5],
    ];
    assert_eq!(cases.len(), expected.len());
    for (case, expected) in cases.iter().zip(expected) {
        let output = run_case(&ep, case);
        assert_close(
            &case.label,
            DataType::Float32,
            &decode_floats(&output[0], DataType::Float32),
            expected,
            1e-5,
        );
    }
}
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn space_depth_modes_match_onnx() {
    let ep = require_cuda();
    let values: Vec<f32> = (0..16).map(|value| value as f32).collect();
    for (op, src, dst, dcr, crd) in [
        (
            "SpaceToDepth",
            [1, 2, 2, 4],
            [1, 8, 1, 2],
            [
                0., 2., 8., 10., 1., 3., 9., 11., 4., 6., 12., 14., 5., 7., 13., 15.,
            ],
            [
                0., 2., 1., 3., 4., 6., 5., 7., 8., 10., 9., 11., 12., 14., 13., 15.,
            ],
        ),
        (
            "DepthToSpace",
            [1, 8, 1, 2],
            [1, 2, 2, 4],
            [
                0., 4., 1., 5., 8., 12., 9., 13., 2., 6., 3., 7., 10., 14., 11., 15.,
            ],
            [
                0., 2., 1., 3., 4., 6., 5., 7., 8., 10., 9., 11., 12., 14., 13., 15.,
            ],
        ),
    ] {
        for (opset, mode, expected) in [
            (1, None, dcr),
            (28, None, dcr),
            (28, Some("DCR"), dcr),
            (28, Some("CRD"), crd),
        ] {
            let mut attrs = vec![("blocksize", Attribute::Int(2))];
            if let Some(mode) = mode {
                attrs.push(("mode", Attribute::String(mode.as_bytes().to_vec())));
            }
            let case = Case {
                label: format!("{op}-{opset}[{mode:?}]"),
                op,
                domain: "",
                opset,
                inputs: vec![input(DataType::Float32, &src, &values)],
                outputs: vec![(DataType::Float32, dst.to_vec())],
                attrs,
                compare: Compare::ExactBytes,
            };
            let output = run_case(&ep, &case);
            assert_eq!(
                decode_floats(&output[0], DataType::Float32),
                expected,
                "{}",
                case.label
            );
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn ort_plugin_keeps_the_space_to_depth_13_contract() {
    let ep = CudaExecutionProvider::new_for_ort_plugin().expect("CUDA device for ORT plugin");
    let descriptors = onnx_runtime_ep_cuda::build_cuda_registry_descriptors(ep.registry());
    let spatial: Vec<_> = descriptors
        .iter()
        .filter(|d| {
            d.domain.is_empty() && matches!(d.op_type.as_str(), "SpaceToDepth" | "DepthToSpace")
        })
        .collect();
    let [descriptor] = spatial.as_slice() else {
        panic!(
            "unexpected number of plugin spatial descriptors: {}",
            spatial.len()
        );
    };
    assert_eq!(descriptor.op_type, "SpaceToDepth");
    assert_eq!(descriptor.since_version, 13);
    assert_eq!(descriptor.end_version, u64::MAX);
    assert_eq!(
        descriptor.supported_dtypes,
        &[DataType::Float32, DataType::Float16, DataType::BFloat16]
    );
    assert!(!ep.registry().supports("SpaceToDepth", "", 12));
    assert!(!ep.registry().supports("DepthToSpace", "", 28));
    // At the legacy provider boundary, `mode` was ignored by both the claim
    // and kernel. Keep this probe below the plugin's independent shape filter.
    let output = run_cuda(
        &ep,
        "SpaceToDepth",
        "",
        28,
        &[input(
            DataType::Float32,
            &[1, 2, 2, 2],
            &[0_f32, 1., 2., 3., 4., 5., 6., 7.],
        )],
        &[(DataType::Float32, vec![1, 8, 1, 1])],
        &[
            ("blocksize", Attribute::Int(2)),
            ("mode", Attribute::String(b"CRD".to_vec())),
        ],
    );
    assert_eq!(
        decode_floats(&output[0], DataType::Float32),
        [0., 4., 1., 5., 2., 6., 3., 7.]
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn tensor_scatter_batch_offsets_match_cpu() {
    let ep = require_cuda();
    for case in tensor_scatter_cases() {
        run_case(&ep, &case);
    }
}

/// Execute every inline `Sweep` case against the CPU oracle on the real GPU.
/// Fails loudly on a host without a CUDA device when `gpu-tests` is enabled.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn conformance_sweep_matches_cpu() {
    let ep = require_cuda();
    let mut ran = 0usize;
    for entry in conformance_profile() {
        if let Coverage::Sweep(cases) = &entry.coverage {
            for case in cases {
                run_case(&ep, case);
                ran += 1;
            }
        }
    }
    assert!(ran > 0, "expected at least one inline conformance case");
    eprintln!("conformance sweep: {ran} inline CUDA-vs-CPU parity cases passed");
}

#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]
#[test]
fn opset26_native_cuda_conformance() {
    let ep = require_cuda();
    for case in bitcast_cases().into_iter().chain(cumprod_cases()) {
        run_case(&ep, &case);
    }
}

#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA device")]
#[test]
fn eye_like_22_native_cuda_matches_onnx_reference() {
    use DataType::{BFloat16, Float32};
    use onnx_runtime_ep_api::ExecutionProvider;
    let ep = require_cuda();
    // ONNX 1.24 ReferenceEvaluator, opset 22, 2x3 input filled with sevens.
    let bf16_one = 0x3f80_u16;
    let shape = [2, 3];
    let shape_for_claim = [onnx_runtime_ir::static_shape(shape)];
    for (input_tensor, output_dtype, attrs, edge, expected) in [
        (
            input(BFloat16, &shape, &[0x40e0_u16; 6]),
            BFloat16,
            vec![],
            "input",
            [bf16_one, 0, 0, 0, bf16_one, 0]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect::<Vec<_>>(),
        ),
        (
            input(BFloat16, &shape, &[0x40e0_u16; 6]),
            Float32,
            vec![
                ("dtype", Attribute::Int(i64::from(Float32.to_onnx()))),
                ("k", Attribute::Int(-1)),
            ],
            "input",
            [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect(),
        ),
        (
            input(Float32, &shape, &[7.0_f32; 6]),
            BFloat16,
            vec![
                ("dtype", Attribute::Int(i64::from(BFloat16.to_onnx()))),
                ("k", Attribute::Int(1)),
            ],
            "output",
            [0, bf16_one, 0, 0, 0, bf16_one]
                .into_iter()
                .flat_map(u16::to_le_bytes)
                .collect(),
        ),
    ] {
        let input_dtype = input_tensor.dtype;
        let inputs = [input_tensor];
        let outputs = [(output_dtype, shape.to_vec())];
        let (graph, id) = common::build_graph("EyeLike", "", 21, &inputs, &outputs, &attrs);
        let claim = ep.supports_op(graph.node(id), 21, &shape_for_claim, &[input_dtype], &[]);
        let reason = claim.reason().expect("EyeLike-9 must not claim BF16");
        assert!(
            reason.contains(&format!("{edge} dtype BFloat16")),
            "{reason}"
        );
        assert_eq!(
            run_cuda(&ep, "EyeLike", "", 22, &inputs, &outputs, &attrs),
            vec![expected],
            "{input_dtype:?} to {output_dtype:?}"
        );
    }
}
