mod common;

use common::{decode_floats, float_input, input, require_cuda, run_cpu, run_cuda};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, TensorLayout, ValueId, static_shape};

fn check_claims(ep: &dyn ExecutionProvider) {
    for (name, since) in onnx_runtime_ir::standard_math::OPS {
        let arity = if ["SwiGLU", "GeGLU"].contains(&name) {
            2
        } else {
            1
        };
        let node = Node::new(
            NodeId(0),
            name,
            (0..arity).map(|i| Some(ValueId(i))).collect(),
            vec![ValueId(arity)],
        );
        let shapes = vec![static_shape([2]); arity as usize];
        let layouts = vec![TensorLayout::contiguous(); arity as usize];
        let dtypes = vec![DataType::Float64; arity as usize];
        assert!(
            ep.supports_op(&node, since, &shapes, &dtypes, &layouts)
                .is_supported(),
            "{name}"
        );
        assert!(
            !ep.supports_op(&node, since - 1, &shapes, &dtypes, &layouts)
                .is_supported(),
            "{name}"
        );
        let invalid = vec![DataType::Int32; arity as usize];
        assert!(
            !ep.supports_op(&node, since, &shapes, &invalid, &layouts)
                .is_supported(),
            "{name}"
        );
    }
}

fn check(ep: Option<&CudaExecutionProvider>) {
    let dtypes = [
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float32,
        DataType::Float64,
    ];
    for dtype in dtypes {
        let run = |op, version, shape: &[usize], data: &[&[f32]], attrs: &[(&str, Attribute)]| {
            let inputs = data
                .iter()
                .map(|x| {
                    if dtype == DataType::Float64 {
                        input(
                            dtype,
                            shape,
                            &x.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
                        )
                    } else {
                        float_input(dtype, shape, x)
                    }
                })
                .collect::<Vec<_>>();
            let outputs = [(dtype, shape.to_vec())];
            let bytes = match ep {
                Some(ep) => run_cuda(ep, op, "", version, &inputs, &outputs, attrs),
                None => run_cpu(op, "", version, &inputs, &outputs, attrs),
            };
            decode_floats(&bytes[0], dtype)
        };
        let close = |value: f32, expected: f32| {
            let tol = if dtype == DataType::BFloat16 {
                0.008
            } else if dtype == DataType::Float16 {
                0.001
            } else {
                0.000_002
            };
            assert!(
                (value - expected).abs() <= tol * expected.abs(),
                "{dtype:?}: got {value}, expected {expected}"
            );
        };
        // Exactly representable even in f16; far below 1 + x's resolution in bf16/f16.
        let tiny = if dtype == DataType::Float16 {
            2f32.powi(-20)
        } else {
            2f32.powi(-30)
        };
        for (op, negative_inf_result) in [("Expm1", -1.), ("Log1p", f32::NAN)] {
            let y = run(
                op,
                29,
                &[7],
                &[&[
                    tiny,
                    -tiny,
                    -0.,
                    0.,
                    f32::NEG_INFINITY,
                    f32::INFINITY,
                    f32::NAN,
                ]],
                &[],
            );
            close(y[0], tiny);
            close(y[1], -tiny);
            assert_eq!(y[2].to_bits(), (-0.0f32).to_bits());
            assert_eq!(y[3].to_bits(), 0);
            if negative_inf_result.is_nan() {
                assert!(y[4].is_nan());
            } else {
                assert_eq!(y[4], negative_inf_result);
            }
            assert_eq!(y[5], f32::INFINITY);
            assert!(y[6].is_nan());
            assert!(run(op, 29, &[0, 3], &[&[]], &[]).is_empty());
        }
        assert_eq!(run("Log1p", 29, &[], &[&[-1.]], &[]), [f32::NEG_INFINITY]);
        let a = [-1., 1.];
        let b = [2., -3.];
        let y = run(
            "SwiGLU",
            28,
            &[2],
            &[&a, &b],
            &[("alpha", Attribute::Float(0.))],
        );
        assert_eq!(y, [-1., -1.5]);
        let y = run("SwiGLU", 28, &[2], &[&a, &b], &[]);
        close(y[0], -0.53788286);
        close(y[1], -2.1931758);
        for (approximate, expected) in [(b"none", -0.3173105), (b"tanh", -0.317616)] {
            let y = run(
                "GeGLU",
                29,
                &[2],
                &[&a, &b],
                &[("approximate", Attribute::String(approximate.to_vec()))],
            );
            close(y[0], expected);
            close(y[1], -3. * (1. + expected / 2.));
        }
        let y = run("GeGLU", 29, &[1], &[&[-4.], &[2.]], &[]);
        close(y[0], -0.00025337);
        let axis = [("axis", Attribute::Int(-2))];
        // Non-last axis: lanes are interleaved. NaN must override a previous +inf.
        let x = [
            -100.,
            f32::NEG_INFINITY,
            -100.,
            f32::INFINITY,
            -100.,
            f32::NAN,
        ];
        let y = run("LogCumSumExp", 29, &[3, 2], &[&x], &axis);
        assert_eq!(
            (y[0], y[1], y[3]),
            (-100., f32::NEG_INFINITY, f32::INFINITY)
        );
        assert!(y[5].is_nan());
        close(y[2], -99.30685);
        close(y[4], -98.90139);
        assert!(run("LogCumSumExp", 29, &[0, 2], &[&[]], &axis).is_empty());
        let y = run(
            "LogCumSumExp",
            29,
            &[1, 128, 1],
            &[&[0.; 128]],
            &[("axis", Attribute::Int(1))],
        );
        assert_eq!(y[0], 0.);
        close(y[127], 4.8520303);
    }

    // Must stay in f64, not underflow by conversion to f32.
    for op in ["Expm1", "Log1p"] {
        let inputs = [input(DataType::Float64, &[], &[1e-200f64])];
        let outputs = [(DataType::Float64, vec![])];
        let result = match ep {
            Some(ep) => run_cuda(ep, op, "", 29, &inputs, &outputs, &[]),
            None => run_cpu(op, "", 29, &inputs, &outputs, &[]),
        };
        assert_eq!(
            f64::from_ne_bytes(result[0].as_slice().try_into().unwrap()),
            1e-200
        );
    }
    let inputs = [input(DataType::Float64, &[2], &[1e12f64, 1e12])];
    let outputs = [(DataType::Float64, vec![2])];
    let result = match ep {
        Some(ep) => run_cuda(ep, "LogCumSumExp", "", 29, &inputs, &outputs, &[]),
        None => run_cpu("LogCumSumExp", "", 29, &inputs, &outputs, &[]),
    };
    assert_eq!(
        f64::from_ne_bytes(result[0][8..].try_into().unwrap()),
        1e12 + std::f64::consts::LN_2,
    );
}

#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires CUDA; enable gpu-tests")]
#[test]
fn standard_math_cuda_reference_values() {
    let ep = require_cuda();
    check_claims(&CpuExecutionProvider::new());
    check(None);
    check_claims(&ep);
    check(Some(&ep));
}
