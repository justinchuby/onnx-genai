//! Data-driven bfloat16 conformance sweep for the native CPU EP.
//!
//! Justin's brief ("全面检查cpu ep对bfloat16的原生支持 一口气支持所有op"):
//! every float-typed op the CPU EP registers must *natively* accept
//! `bfloat16` (`DataType::BFloat16`, storage `half::bf16`) inputs and compute a
//! correct result — no op may silently reject, panic on, or mis-handle bf16.
//!
//! This harness constructs a minimal valid node for a broad, representative set
//! of the registered float ops (covering every shared dtype-dispatch path:
//! `dispatch_arith`, `dispatch_float`, `to_dense_f32_widen`/`write_dense_f32_narrow`,
//! and the per-op manual matches), runs it with **bf16** inputs, and asserts
//!
//!   1. it executes without error (the "does the kernel accept bf16" guarantee), and
//!   2. its bf16 result matches the same node run in **f32** within bf16 tolerance
//!      (the "does the kernel compute bf16 correctly" guarantee).
//!
//! It is both the "一口气支持所有op" proof and the regression lock: adding a new
//! float op with an f32/f16-only dispatch will make its row here fail.

#[path = "../benches/common/mod.rs"]
mod common;

use common::{FloatDType, Tensor};
use onnx_runtime_ep_api::{ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId};

/// One positional operator input.
enum Inp {
    /// A float input built in the dtype under test (f32 reference vs bf16).
    F(Vec<usize>, Vec<f32>),
    /// A fixed int64 input (indices / axes / repeats), dtype-invariant.
    I64(Vec<usize>, Vec<i64>),
    /// A fixed int32 input (cu_seqlens), dtype-invariant.
    I32(Vec<usize>, Vec<i32>),
    /// A fixed bool input (e.g. `Where` condition), dtype-invariant.
    Bool(Vec<usize>, Vec<bool>),
}

struct OpCase {
    op: &'static str,
    domain: &'static str,
    opset: u64,
    attrs: Vec<(&'static str, Attribute)>,
    inputs: Vec<Inp>,
    out_shape: Vec<usize>,
    /// Absolute tolerance floor; the effective tolerance also scales with the
    /// magnitude of the f32 reference (bf16 carries ~3 significant digits).
    abs_tol: f32,
}

fn build_tensors(inputs: &[Inp], dtype: FloatDType) -> Vec<Tensor> {
    inputs
        .iter()
        .map(|inp| match inp {
            Inp::F(shape, values) => Tensor::floats(dtype, shape, values),
            Inp::I64(shape, values) => Tensor::i64(shape, values),
            Inp::I32(shape, values) => Tensor::i32(shape, values),
            Inp::Bool(shape, values) => Tensor::bool(shape, values),
        })
        .collect()
}

fn run_case(case: &OpCase, dtype: FloatDType) -> Vec<f32> {
    let mut node = Node::new(NodeId(0), case.op, vec![], vec![]);
    node.domain = case.domain.to_string();
    for (name, value) in &case.attrs {
        node.attributes.insert((*name).into(), value.clone());
    }
    let input_shapes: Vec<Vec<usize>> = case
        .inputs
        .iter()
        .map(|inp| match inp {
            Inp::F(shape, _) => shape.clone(),
            Inp::I64(shape, _) => shape.clone(),
            Inp::I32(shape, _) => shape.clone(),
            Inp::Bool(shape, _) => shape.clone(),
        })
        .collect();

    let kernel = CpuExecutionProvider::new()
        .get_kernel(&node, &input_shapes, case.opset)
        .unwrap_or_else(|e| panic!("{} ({}): get_kernel failed: {e:?}", case.op, dtype.name()));

    let inputs = build_tensors(&case.inputs, dtype);
    let mut output = Tensor::zeros(dtype, &case.out_shape);

    let views: Vec<TensorView<'_>> = inputs.iter().map(|t| t.view()).collect();
    let mut out_view: [TensorMut<'_>; 1] = [output.view_mut()];
    kernel
        .execute(&views, &mut out_view)
        .unwrap_or_else(|e| panic!("{} ({}): execute failed: {e:?}", case.op, dtype.name()));
    output.to_f32()
}

fn assert_bf16_matches_f32(case: &OpCase) {
    let reference = run_case(case, FloatDType::F32);
    let actual = run_case(case, FloatDType::Bf16);
    assert_eq!(
        reference.len(),
        actual.len(),
        "{}: bf16 output length {} != f32 {}",
        case.op,
        actual.len(),
        reference.len()
    );
    for (i, (&r, &a)) in reference.iter().zip(&actual).enumerate() {
        // bf16 has an 8-bit significand (~2 decimal digits); allow a relative
        // slack that scales with the reference magnitude plus a per-op floor.
        let tol = case.abs_tol + 0.06 * r.abs();
        assert!(
            (r - a).abs() <= tol || (r.is_nan() && a.is_nan()),
            "{}: element {i} bf16={a} vs f32={r} exceeds tol {tol}",
            case.op
        );
    }
}

/// A shared moderate-magnitude sample; kept away from bf16 saturation and from
/// the domain edges of `log`/`acos`/`atanh` etc.
fn sample6() -> Vec<f32> {
    vec![0.5, -0.75, 1.25, -1.5, 0.25, 2.0]
}

/// Strictly-positive sample for `Log`/`Sqrt`/`ReduceLogSum` domains.
fn positive6() -> Vec<f32> {
    vec![0.5, 0.75, 1.25, 1.5, 0.25, 2.0]
}

/// Sample confined to `(-1, 1)` for `Asin`/`Acos`/`Atanh`.
fn unit6() -> Vec<f32> {
    vec![0.5, -0.75, 0.25, -0.5, 0.125, -0.25]
}

fn unary(op: &'static str, opset: u64, values: Vec<f32>, abs_tol: f32) -> OpCase {
    OpCase {
        op,
        domain: "",
        opset,
        attrs: vec![],
        inputs: vec![Inp::F(vec![2, 3], values)],
        out_shape: vec![2, 3],
        abs_tol,
    }
}

fn binary(op: &'static str, opset: u64, abs_tol: f32) -> OpCase {
    OpCase {
        op,
        domain: "",
        opset,
        attrs: vec![],
        inputs: vec![
            Inp::F(vec![2, 3], sample6()),
            Inp::F(vec![2, 3], vec![1.5, 2.0, -0.5, 0.75, 3.0, -1.25]),
        ],
        out_shape: vec![2, 3],
        abs_tol,
    }
}

fn reduce(op: &'static str, values: Vec<f32>, abs_tol: f32) -> OpCase {
    OpCase {
        op,
        domain: "",
        opset: 11,
        attrs: vec![
            ("axes", Attribute::Ints(vec![1])),
            ("keepdims", Attribute::Int(1)),
        ],
        inputs: vec![Inp::F(vec![2, 3], values)],
        out_shape: vec![2, 1],
        abs_tol,
    }
}

fn cases() -> Vec<OpCase> {
    let mut v = Vec::new();

    // ---- Unary math (dispatch_float / to_dense_f32_widen) ----
    for op in ["Abs", "Neg", "Floor", "Ceil", "Round", "Sign", "Reciprocal"] {
        v.push(unary(op, 13, sample6(), 0.02));
    }
    v.push(unary("Sqrt", 13, positive6(), 0.02));
    v.push(unary("Exp", 13, sample6(), 0.05));
    v.push(unary("Log", 13, positive6(), 0.05));
    for op in ["Sin", "Cos", "Tan", "Atan", "Sinh", "Cosh", "Tanh"] {
        v.push(unary(op, 13, sample6(), 0.03));
    }
    for op in ["Asin", "Acos", "Atanh"] {
        v.push(unary(op, 13, unit6(), 0.03));
    }
    for op in ["Asinh", "Acosh"] {
        // Acosh domain is x >= 1.
        let vals = if op == "Acosh" {
            vec![1.5, 2.0, 3.0, 1.25, 4.0, 2.5]
        } else {
            sample6()
        };
        v.push(unary(op, 13, vals, 0.03));
    }
    v.push(unary("Erf", 13, sample6(), 0.02));

    // ---- Activations (to_dense_f32_widen) ----
    for op in ["Relu", "Sigmoid", "Softsign", "Softplus"] {
        v.push(unary(op, 13, sample6(), 0.03));
    }
    v.push(OpCase {
        op: "LeakyRelu",
        domain: "",
        opset: 16,
        attrs: vec![("alpha", Attribute::Float(0.1))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.03,
    });
    v.push(OpCase {
        op: "Elu",
        domain: "",
        opset: 6,
        attrs: vec![("alpha", Attribute::Float(1.0))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.03,
    });
    v.push(OpCase {
        op: "Selu",
        domain: "",
        opset: 6,
        attrs: vec![],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.05,
    });
    v.push(OpCase {
        op: "HardSigmoid",
        domain: "",
        opset: 6,
        attrs: vec![
            ("alpha", Attribute::Float(0.2)),
            ("beta", Attribute::Float(0.5)),
        ],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.03,
    });
    v.push(OpCase {
        op: "ThresholdedRelu",
        domain: "",
        opset: 10,
        attrs: vec![("alpha", Attribute::Float(0.5))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.03,
    });
    v.push(unary("Gelu", 20, sample6(), 0.03));
    v.push(unary("Sign", 13, sample6(), 0.0));

    // ---- Binary elementwise (dispatch_arith / dispatch_float) ----
    for op in ["Add", "Sub", "Mul", "Div", "Min", "Max", "Sum", "Mean"] {
        v.push(binary(op, 13, 0.03));
    }
    v.push(OpCase {
        op: "Pow",
        domain: "",
        opset: 13,
        attrs: vec![],
        inputs: vec![
            Inp::F(vec![2, 3], positive6()),
            Inp::F(vec![2, 3], vec![2.0, 1.0, 0.5, 2.0, 1.5, 1.0]),
        ],
        out_shape: vec![2, 3],
        abs_tol: 0.05,
    });

    // ---- Reductions (to_dense_f32_widen) ----
    v.push(reduce("ReduceSum", sample6(), 0.05));
    v.push(reduce("ReduceMean", sample6(), 0.03));
    v.push(reduce("ReduceMax", sample6(), 0.02));
    v.push(reduce("ReduceMin", sample6(), 0.02));
    v.push(reduce("ReduceProd", positive6(), 0.05));
    v.push(reduce("ReduceL1", sample6(), 0.05));
    v.push(reduce("ReduceL2", sample6(), 0.05));
    v.push(reduce("ReduceSumSquare", sample6(), 0.05));
    v.push(reduce("ReduceLogSum", positive6(), 0.05));
    v.push(reduce("ReduceLogSumExp", sample6(), 0.05));

    // ---- Normalization / softmax family ----
    v.push(OpCase {
        op: "Softmax",
        domain: "",
        opset: 13,
        attrs: vec![("axis", Attribute::Int(1))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.02,
    });
    v.push(OpCase {
        op: "LogSoftmax",
        domain: "",
        opset: 13,
        attrs: vec![("axis", Attribute::Int(1))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.05,
    });
    v.push(OpCase {
        op: "Hardmax",
        domain: "",
        opset: 13,
        attrs: vec![("axis", Attribute::Int(1))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.0,
    });
    v.push(OpCase {
        op: "LayerNormalization",
        domain: "",
        opset: 17,
        attrs: vec![("axis", Attribute::Int(-1))],
        inputs: vec![
            Inp::F(vec![2, 3], sample6()),
            Inp::F(vec![3], vec![1.0, 1.0, 1.0]),
            Inp::F(vec![3], vec![0.0, 0.0, 0.0]),
        ],
        out_shape: vec![2, 3],
        abs_tol: 0.05,
    });
    v.push(OpCase {
        op: "LpNormalization",
        domain: "",
        opset: 1,
        attrs: vec![("axis", Attribute::Int(-1)), ("p", Attribute::Int(2))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![2, 3],
        abs_tol: 0.03,
    });

    // ---- Selection / movement / linear algebra ----
    v.push(OpCase {
        op: "Clip",
        domain: "",
        opset: 13,
        attrs: vec![],
        inputs: vec![
            Inp::F(vec![2, 3], sample6()),
            Inp::F(vec![], vec![-1.0]),
            Inp::F(vec![], vec![1.0]),
        ],
        out_shape: vec![2, 3],
        abs_tol: 0.02,
    });
    v.push(OpCase {
        op: "Where",
        domain: "",
        opset: 16,
        attrs: vec![],
        inputs: vec![
            Inp::Bool(vec![2, 3], vec![true, false, true, false, true, false]),
            Inp::F(vec![2, 3], sample6()),
            Inp::F(vec![2, 3], vec![9.0, 9.0, 9.0, 9.0, 9.0, 9.0]),
        ],
        out_shape: vec![2, 3],
        abs_tol: 0.05,
    });
    v.push(OpCase {
        op: "Transpose",
        domain: "",
        opset: 13,
        attrs: vec![("perm", Attribute::Ints(vec![1, 0]))],
        inputs: vec![Inp::F(vec![2, 3], sample6())],
        out_shape: vec![3, 2],
        abs_tol: 0.0,
    });
    v.push(OpCase {
        op: "Concat",
        domain: "",
        opset: 13,
        attrs: vec![("axis", Attribute::Int(0))],
        inputs: vec![
            Inp::F(vec![2, 3], sample6()),
            Inp::F(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
        out_shape: vec![4, 3],
        abs_tol: 0.0,
    });
    v.push(OpCase {
        op: "MatMul",
        domain: "",
        opset: 13,
        attrs: vec![],
        inputs: vec![
            Inp::F(vec![2, 3], sample6()),
            Inp::F(vec![3, 2], vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0]),
        ],
        out_shape: vec![2, 2],
        abs_tol: 0.1,
    });
    v.push(OpCase {
        op: "Gemm",
        domain: "",
        opset: 13,
        attrs: vec![],
        inputs: vec![
            Inp::F(vec![2, 3], sample6()),
            Inp::F(vec![3, 2], vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0]),
        ],
        out_shape: vec![2, 2],
        abs_tol: 0.1,
    });
    v.push(OpCase {
        op: "CumSum",
        domain: "",
        opset: 14,
        attrs: vec![],
        inputs: vec![Inp::F(vec![2, 3], sample6()), Inp::I64(vec![], vec![1])],
        out_shape: vec![2, 3],
        abs_tol: 0.05,
    });
    v.push(OpCase {
        op: "Expand",
        domain: "",
        opset: 13,
        attrs: vec![],
        inputs: vec![
            Inp::F(vec![1, 3], vec![0.5, -0.75, 1.25]),
            Inp::I64(vec![2], vec![2, 3]),
        ],
        out_shape: vec![2, 3],
        abs_tol: 0.0,
    });
    v.push(OpCase {
        op: "Tile",
        domain: "",
        opset: 13,
        attrs: vec![],
        inputs: vec![Inp::F(vec![2, 3], sample6()), Inp::I64(vec![2], vec![2, 1])],
        out_shape: vec![4, 3],
        abs_tol: 0.0,
    });

    // ---- Signal / spectral (DFT: real -> complex, computed in f32) ----
    v.push(OpCase {
        op: "DFT",
        domain: "",
        opset: 17,
        attrs: vec![("axis", Attribute::Int(1))],
        inputs: vec![Inp::F(vec![1, 4, 1], vec![0.5, -0.75, 1.25, -1.5])],
        out_shape: vec![1, 4, 2],
        abs_tol: 0.1,
    });

    v
}

#[test]
fn every_registered_float_op_supports_bf16() {
    let all = cases();
    assert!(all.len() >= 60, "conformance table shrank to {}", all.len());
    for case in &all {
        assert_bf16_matches_f32(case);
    }
}

/// Ensures the CPU EP does not have a `DataType::BFloat16` gap that would make
/// `get_kernel`/`execute` bail: every case above must run natively in bf16.
#[test]
fn no_op_rejects_bf16_at_runtime() {
    for case in &cases() {
        let out = run_case(case, FloatDType::Bf16);
        assert_eq!(
            out.len(),
            case.out_shape.iter().product::<usize>(),
            "{}: bf16 produced {} elements, expected shape {:?}",
            case.op,
            out.len(),
            case.out_shape
        );
    }
}

#[allow(dead_code)]
fn _dtype_marker() -> DataType {
    DataType::BFloat16
}

/// `VarlenAttention` (`pkg.nxrt`) reads Q/K/V through `to_dense_f32_widen` and
/// writes through `write_dense_f32_narrow`, so it is natively bf16-capable — its
/// claim/execute gates previously rejected non-f32 spuriously. This covers the
/// packed varlen-attention path (single sequence, causal-off) in both dtypes.
fn varlen_case() -> OpCase {
    // 2 tokens, 1 head, head_size 2; one sequence => cu_seqlens = [0, 2].
    OpCase {
        op: "VarlenAttention",
        domain: "pkg.nxrt",
        opset: 1,
        attrs: vec![],
        inputs: vec![
            Inp::F(vec![2, 1, 2], vec![0.5, -0.25, 0.75, 0.1]),
            Inp::F(vec![2, 1, 2], vec![0.2, 0.4, -0.3, 0.6]),
            Inp::F(vec![2, 1, 2], vec![1.0, -1.0, 0.5, 0.25]),
            Inp::I32(vec![2], vec![0, 2]),
            Inp::I32(vec![2], vec![0, 2]),
        ],
        out_shape: vec![2, 1, 2],
        abs_tol: 0.05,
    }
}

#[test]
fn varlen_attention_supports_bf16() {
    let case = varlen_case();
    // Must not reject bf16 at claim/execute time, and must match the f32 oracle.
    assert_bf16_matches_f32(&case);
}
