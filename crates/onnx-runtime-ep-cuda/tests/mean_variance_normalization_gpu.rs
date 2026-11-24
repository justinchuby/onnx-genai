//! Standalone CPU/CUDA MeanVarianceNormalization-29 against analytical goldens.

mod common;

use common::{
    assert_close, build_graph, decode_floats, float_input, input, require_cuda, run_cpu, run_cuda,
};
use onnx_runtime_ep_api::ExecutionProvider;
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ir::{Attribute, DataType, static_shape};

const OP: &str = "MeanVarianceNormalization";

fn goldens(cuda: Option<&CudaExecutionProvider>) {
    let check =
        |dtype, shape: &[usize], x: &[f32], attrs: &[(&str, Attribute)], expected: &[f32]| {
            let inputs = [float_input(dtype, shape, x)];
            let outputs = [(dtype, shape.to_vec())];
            let cpu = run_cpu(OP, "", 29, &inputs, &outputs, attrs);
            let cpu = decode_floats(&cpu[0], dtype);
            assert_close("CPU vs golden", dtype, &cpu, expected, 1e-6);
            if let Some(ep) = cuda {
                let gpu = run_cuda(ep, OP, "", 29, &inputs, &outputs, attrs);
                let gpu = decode_floats(&gpu[0], dtype);
                assert_close("CUDA vs golden", dtype, &gpu, expected, 1e-6);
                assert_close("CUDA vs CPU", dtype, &gpu, &cpu, 1e-6);
            }
        };

    // Default NCHW axes, population variance and cancellation: legacy
    // E[X²]-E[X]² rounds to zero for the first channel, not one.
    check(
        DataType::Float32,
        &[2, 2, 1, 1],
        &[9999., -2., 10001., 6.],
        &[],
        &[-1., -1., 1., 1.],
    );
    // Explicit epsilon of 2 goes outside the square root: 2 / (sqrt(4) + 2).
    check(
        DataType::Float32,
        &[2, 2],
        &[0., 4., 8., 4.],
        &[
            ("axes", Attribute::Ints(vec![-1])),
            ("epsilon", Attribute::Float(2.)),
        ],
        &[-0.5, 0.5, 0.5, -0.5],
    );
    // Empty axes reduce all dimensions. The small standard deviation makes
    // the default epsilon observable: 1e-9 / (1e-9 + 1e-9).
    let all = [("axes", Attribute::Ints(vec![]))];
    check(
        DataType::Float32,
        &[2, 1],
        &[-1e-9, 1e-9],
        &all,
        &[-0.5, 0.5],
    );
    check(DataType::Float32, &[], &[7.], &all, &[0.]);
    check(DataType::Float32, &[0, 3], &[], &all, &[]);
    // An f16 square of 1024 overflows; f32 intermediates do not.
    check(DataType::Float16, &[2], &[-1024., 1024.], &all, &[-1., 1.]);
    // In bf16 the mean 257 would round to 256 before the subtraction.
    check(DataType::BFloat16, &[2], &[256., 258.], &all, &[-1., 1.]);
    // Two separated reduction runs, more than one group, and more elements
    // per group than CUDA threads. Each group alternates -2 and +2 about its own mean.
    let values: Vec<f32> = (0..2 * 3 * 258)
        .map(|i| (i / 258 % 3) as f32 * 10. + if i % 2 == 0 { -2. } else { 2. })
        .collect();
    let expected: Vec<f32> = (0..values.len())
        .map(|i| if i % 2 == 0 { -1. } else { 1. })
        .collect();
    check(
        DataType::Float32,
        &[2, 3, 258],
        &values,
        &[("axes", Attribute::Ints(vec![0, -1]))],
        &expected,
    );

    // A real f64 check: casting either input to f32 would collapse the pair.
    let inputs = [input(DataType::Float64, &[2], &[1e12f64 - 2., 1e12 + 2.])];
    let outputs = [(DataType::Float64, vec![2])];
    let attrs = [
        ("axes", Attribute::Ints(vec![0])),
        ("epsilon", Attribute::Float(2.)),
    ];
    let cpu = run_cpu(OP, "", 29, &inputs, &outputs, &attrs);
    let expected: Vec<u8> = [-0.5f64, 0.5]
        .into_iter()
        .flat_map(f64::to_ne_bytes)
        .collect();
    assert_eq!(cpu[0], expected);
    if let Some(ep) = cuda {
        assert_eq!(run_cuda(ep, OP, "", 29, &inputs, &outputs, &attrs), cpu);
    }
}

fn claims(ep: &dyn ExecutionProvider) {
    for (opset, supported) in [(9, false), (13, false), (28, false), (29, true), (30, true)] {
        let (graph, id) = build_graph(
            OP,
            "",
            opset,
            &[input(DataType::Float64, &[2], &[0f64, 1.])],
            &[(DataType::Float64, vec![2])],
            &[("axes", Attribute::Ints(vec![0]))],
        );
        let claim = ep.supports_op(
            graph.node(id),
            opset,
            &[static_shape([2])],
            &[DataType::Float64],
            &[],
        );
        assert_eq!(
            claim.is_supported(),
            supported,
            "opset {opset}: {:?}",
            claim.reason()
        );
        assert!(
            !ep.supports_op(
                graph.node(id),
                opset,
                &[static_shape([2])],
                &[DataType::Int32],
                &[]
            )
            .is_supported()
        );
    }
    for (axes, reason) in [(vec![1], "out of range"), (vec![0, -1], "duplicate")] {
        let (graph, id) = build_graph(
            OP,
            "",
            29,
            &[input(DataType::Float32, &[2], &[0f32, 1.])],
            &[(DataType::Float32, vec![2])],
            &[("axes", Attribute::Ints(axes))],
        );
        let claim = ep.supports_op(
            graph.node(id),
            29,
            &[static_shape([2])],
            &[DataType::Float32],
            &[],
        );
        assert!(
            claim.reason().is_some_and(|text| text.contains(reason)),
            "{:?}",
            claim.reason()
        );
    }
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "CUDA test target requires gpu-tests")]
fn mean_variance_normalization_cpu_goldens_and_claims() {
    claims(&CpuExecutionProvider::new());
    goldens(None);
}

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn mean_variance_normalization_cuda_goldens_and_parity() {
    let ep = require_cuda();
    claims(&ep);
    goldens(Some(&ep));
}

#[test]
#[ignore = "release CUDA timing probe; requires an isolated GPU"]
fn mean_variance_normalization_cuda_release_probe() {
    use std::time::{Duration, Instant};

    use cudarc::driver::sys::CUevent_flags;
    use cudarc::driver::{DevicePtr as _, DevicePtrMut as _};
    use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
    use onnx_runtime_ir::{Node, NodeId, ValueId, compute_contiguous_strides};

    let ep = require_cuda();
    let rt = ep.runtime();
    let profile = |shape: &[usize], axes: Vec<i64>, ramp: Duration| {
        let dtype = DataType::Float32;
        let mut node = Node::new(NodeId(0), OP, vec![Some(ValueId(0))], vec![ValueId(1)]);
        node.attributes.insert("axes".into(), Attribute::Ints(axes));
        let kernel = ep.get_kernel(&node, &[shape.to_vec()], 29).unwrap();
        let n = shape.iter().product::<usize>();
        let strides = compute_contiguous_strides(shape);
        let data: Vec<f32> = (0..n).map(|i| if i % 2 == 0 { -2. } else { 2. }).collect();
        let x = rt.stream().clone_htod(&data).unwrap();
        let mut y = rt.stream().alloc_zeros::<f32>(n).unwrap();
        let (x_ptr, _) = x.device_ptr(rt.stream());
        let (y_ptr, _) = y.device_ptr_mut(rt.stream());
        let inputs = [TensorView::new(
            DevicePtr(x_ptr as *const _),
            dtype,
            shape,
            &strides,
            ep.device_id(),
        )];
        let mut outputs = [TensorMut::new(
            DevicePtrMut(y_ptr as *mut _),
            dtype,
            shape,
            &strides,
            ep.device_id(),
        )];
        kernel.execute(&inputs, &mut outputs).unwrap();
        rt.synchronize().unwrap();

        const BATCH: usize = 64;
        const REPLAYS: usize = 20;
        rt.begin_graph_capture(&[]).unwrap();
        for _ in 0..BATCH {
            kernel.execute(&inputs, &mut outputs).unwrap();
        }
        rt.end_graph_capture().unwrap();
        let warmup = Instant::now();
        while warmup.elapsed() < ramp {
            rt.replay_graph().unwrap();
            rt.synchronize().unwrap();
        }
        let mut us = (0..5)
            .map(|_| {
                let start = rt
                    .stream()
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                for _ in 0..REPLAYS {
                    rt.replay_graph().unwrap();
                }
                let end = rt
                    .stream()
                    .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                    .unwrap();
                f64::from(start.elapsed_ms(&end).unwrap()) * 1_000. / (BATCH * REPLAYS) as f64
            })
            .collect::<Vec<_>>();
        rt.reset_graph().unwrap();
        let actual = rt.stream().clone_dtoh(&y).unwrap();
        assert!(actual.iter().zip(&data).all(|(&y, &x)| y == x / 2.));
        us.sort_by(f64::total_cmp);
        eprintln!(
            "{OP} CUDA {shape:?}: median {:.3} us, range {:.3}-{:.3} us (n=5)",
            us[2], us[0], us[4]
        );
        assert!(us[2].is_finite() && us[2] > 0.);
        us[2]
    };
    let first = profile(&[512, 1024], vec![-1], Duration::from_secs(8));
    profile(&[8, 64, 32, 32], vec![0, 2, 3], Duration::from_secs(1));
    profile(&[512, 1024], vec![], Duration::from_secs(1));
    let last = profile(&[512, 1024], vec![-1], Duration::from_secs(1));
    eprintln!("Anchor repeat drift: {:.2}%", (last / first - 1.) * 100.);
}
