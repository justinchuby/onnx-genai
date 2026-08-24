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
//! GPU-vs-CPU checks for the CUDA Wave-4 activation kernels.

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{DataType, DeviceId, Node, NodeId, compute_contiguous_strides};

fn f32_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: f32 is plain data and the byte slice retains the source lifetime.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|x| f32::from_ne_bytes([x[0], x[1], x[2], x[3]]))
        .collect()
}

fn run(
    ep: &CudaExecutionProvider,
    node: &Node,
    x: &[f32],
    bounds: Option<(Option<f32>, Option<f32>)>,
) -> Vec<f32> {
    let rt = ep.runtime();
    let dev: DeviceId = ep.device_id();
    let shape = [x.len()];
    let strides = compute_contiguous_strides(&shape);
    let x_buf = ep.allocate(std::mem::size_of_val(x), 256).unwrap();
    let mut y_buf = ep.allocate(std::mem::size_of_val(x), 256).unwrap();
    // SAFETY: x_buf was allocated for exactly this byte slice.
    unsafe { rt.htod(f32_bytes(x), cuptr(x_buf.as_ptr())).unwrap() };

    let x_view = TensorView::new(
        DevicePtr(x_buf.as_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        dev,
    );
    let y_view = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        DataType::Float32,
        &shape,
        &strides,
        dev,
    );
    let mut inputs = vec![x_view];
    let mut bound_buffers = Vec::new();
    let scalar_shape: [usize; 0] = [];
    let scalar_strides: [i64; 0] = [];
    if let Some((min, max)) = bounds {
        for bound in [min, max] {
            if let Some(value) = bound {
                let buf = ep.allocate(4, 256).unwrap();
                // SAFETY: buf is a four-byte allocation for one f32.
                unsafe { rt.htod(f32_bytes(&[value]), cuptr(buf.as_ptr())).unwrap() };
                inputs.push(TensorView::new(
                    DevicePtr(buf.as_ptr()),
                    DataType::Float32,
                    &scalar_shape,
                    &scalar_strides,
                    dev,
                ));
                bound_buffers.push(buf);
            } else if max.is_some() {
                inputs.push(TensorView::absent(DataType::Float32));
            }
        }
    }

    let kernel = ep.get_kernel(node, &[], 17).unwrap();
    kernel.execute(&inputs, &mut [y_view]).unwrap();
    let mut bytes = vec![0u8; std::mem::size_of_val(x)];
    // SAFETY: y_buf contains x.len() f32 values.
    unsafe { rt.dtoh(&mut bytes, cuptr(y_buf.as_ptr())).unwrap() };

    ep.deallocate(x_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    for buf in bound_buffers {
        ep.deallocate(buf).unwrap();
    }
    bytes_to_f32(&bytes)
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    for (index, (&got, &want)) in got.iter().zip(want).enumerate() {
        if want.is_nan() {
            assert!(got.is_nan(), "index {index}: got {got}, want NaN");
        } else {
            assert!(
                (got - want).abs() <= 2e-6,
                "index {index}: got {got}, want {want}"
            );
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn wave4_activations_match_cpu_references() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked (library unavailable)");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };
    let x = [-3.0, -1.0, -0.0, 0.0, 0.5, 2.0, f32::NAN];
    let node = |op| Node::new(NodeId(0), op, vec![], vec![]);

    assert_close(
        &run(&ep, &node("LeakyRelu"), &x, None),
        &x.map(|v| if v >= 0.0 { v } else { 0.01 * v }),
    );
    assert_close(
        &run(&ep, &node("Elu"), &x, None),
        &x.map(|v| if v >= 0.0 { v } else { v.exp_m1() }),
    );
    assert_close(
        &run(&ep, &node("HardSigmoid"), &x, None),
        &x.map(|v| (0.2 * v + 0.5).clamp(0.0, 1.0)),
    );
    assert_close(
        &run(&ep, &node("Clip"), &x, Some((Some(-1.0), Some(1.0)))),
        &x.map(|v| v.clamp(-1.0, 1.0)),
    );
    assert_close(
        &run(&ep, &node("Softsign"), &x, None),
        &x.map(|v| v / (1.0 + v.abs())),
    );
    assert_close(
        &run(&ep, &node("Selu"), &x, None),
        &x.map(|v| 1.0507 * if v >= 0.0 { v } else { 1.67326 * v.exp_m1() }),
    );

    let mut silu = node("Silu");
    silu.domain = "com.microsoft".into();
    assert_close(
        &run(&ep, &silu, &x, None),
        &x.map(|v| {
            if v >= 0.0 {
                v / (1.0 + (-v).exp())
            } else {
                let exp = v.exp();
                v * exp / (1.0 + exp)
            }
        }),
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn silu_matches_cpu_operation_order_exactly() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked (library unavailable)");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };
    let x: [f32; 6] = [
        -0.18738078,
        -0.19820021,
        0.52342105,
        -0.29911944,
        0.2953185,
        1.0913864,
    ];
    let expected = x.map(|value| {
        if value >= 0.0 {
            value / (1.0 + (-value).exp())
        } else {
            let exp = value.exp();
            value * exp / (1.0 + exp)
        }
    });
    let mut silu = Node::new(NodeId(0), "Silu", vec![], vec![]);
    silu.domain = "com.microsoft".into();

    assert_eq!(run(&ep, &silu, &x, None), expected);
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn clip_optional_bounds_match_cpu_reference() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked (library unavailable)");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };
    let x = [
        f32::NEG_INFINITY,
        -3.0,
        -1.0,
        0.5,
        2.0,
        f32::INFINITY,
        f32::NAN,
    ];
    let clip = Node::new(NodeId(0), "Clip", vec![], vec![]);
    let cases = [
        (Some((Some(-1.0), Some(1.0))), -1.0, 1.0),
        (Some((Some(-1.0), None)), -1.0, f32::MAX),
        (Some((None, Some(1.0))), f32::MIN, 1.0),
        (None, f32::MIN, f32::MAX),
    ];

    for (bounds, min, max) in cases {
        assert_close(
            &run(&ep, &clip, &x, bounds),
            &x.map(|value| value.clamp(min, max)),
        );
    }
}

/// `Mish(x) = x · tanh(softplus(x))`, opset 22.
///
/// The values deliberately include large positives. Written the obvious way as
/// `log1pf(expf(x))`, softplus overflows to `inf` around x ≈ 89 and Mish then
/// returns NaN, where the correct answer converges to x. The CPU kernel uses the
/// overflow-stable form and the CUDA one must agree — a test that only probed
/// small inputs would pass against the broken spelling.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn mish_matches_cpu_including_the_saturating_tail() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked (library unavailable)");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };
    let x = [
        -20.0f32, -3.0, -1.0, -0.0, 0.0, 0.5, 2.0, 20.0, 100.0, 200.0,
    ];
    let node = Node::new(NodeId(0), "Mish", vec![], vec![]);
    let got = run(&ep, &node, &x, None);

    let expected: Vec<f32> = x
        .iter()
        .map(|&v| {
            // Same stable spelling as the CPU kernel.
            let softplus = v.max(0.0) + (-v.abs()).exp().ln_1p();
            v * softplus.tanh()
        })
        .collect();
    assert_close(&got, &expected);

    // The property the tail is really about: for large x, Mish(x) -> x. If
    // softplus overflowed, these would be NaN rather than close to the input.
    for (input, output) in x.iter().zip(got.iter()).filter(|(v, _)| **v >= 20.0) {
        assert!(
            (output - input).abs() < 1e-3,
            "Mish({input}) = {output}; expected it to converge to the input"
        );
    }
}
