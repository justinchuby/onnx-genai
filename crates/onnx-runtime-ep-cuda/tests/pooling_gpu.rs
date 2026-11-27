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
//! Physical CUDA tests for ONNX pooling; CPU references live in cpu_fixture_checks.

mod common;
#[path = "fixtures/pooling.rs"]
mod fixtures;
use fixtures::{assert_close, build_pool_model, dilated_cases};

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, OutputMetadata, TensorMetadata, TensorMut,
    TensorView, WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, NodeId, as_static_shape, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn tensor_bytes(dtype: DataType, values: &[f32]) -> Vec<u8> {
    match dtype {
        DataType::Float32 => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect(),
        DataType::Float16 => values
            .iter()
            .flat_map(|&value| f16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        DataType::BFloat16 => values
            .iter()
            .flat_map(|&value| bf16::from_f32(value).to_bits().to_le_bytes())
            .collect(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn run_model(
    ep: &CudaExecutionProvider,
    model: &Model<'_>,
    node_id: NodeId,
    values: &[f32],
) -> Vec<f32> {
    run_model_with(ep, model, node_id, values, |run| run())
}

fn run_model_with(
    ep: &CudaExecutionProvider,
    model: &Model<'_>,
    node_id: NodeId,
    values: &[f32],
    execute: impl FnOnce(&mut dyn FnMut()),
) -> Vec<f32> {
    let graph = model.graph;
    let node = graph.node(node_id);
    let input_id = node.inputs[0].unwrap();
    let output_id = node.outputs[0].unwrap();
    let input_shape = as_static_shape(&graph.value(input_id).shape).unwrap();
    let output_shape = as_static_shape(&graph.value(output_id).shape).unwrap();
    let dtype = graph.value(input_id).dtype;
    let input_bytes = tensor_bytes(dtype, values);
    let input_buf = ep.allocate(input_bytes.len(), 256).unwrap();
    let mut output_buf = ep
        .allocate(
            output_shape.iter().product::<usize>() * dtype.byte_size(),
            256,
        )
        .unwrap();
    unsafe {
        ep.runtime()
            .htod(&input_bytes, cuptr(input_buf.as_ptr()))
            .unwrap();
    }
    let input_strides = compute_contiguous_strides(&input_shape)
        .expect("fixture shape has representable contiguous strides");
    let output_strides = compute_contiguous_strides(&output_shape)
        .expect("fixture shape has representable contiguous strides");
    let device = ep.device_id();
    let input = TensorView::new(
        DevicePtr(input_buf.as_ptr()),
        dtype,
        &input_shape,
        &input_strides,
        device,
    );
    let output = TensorMut::new(
        DevicePtrMut(output_buf.as_mut_ptr()),
        dtype,
        &output_shape,
        &output_strides,
        device,
    );
    let opset = graph.opset_imports[""];
    assert!(
        ep.supports_op(
            node,
            opset,
            &[graph.value(input_id).shape.clone()],
            &[dtype],
            &[]
        )
        .is_supported()
    );
    let kernel = ep
        .get_kernel(node, std::slice::from_ref(&input_shape), opset)
        .unwrap();
    let requirement = kernel
        .workspace_requirement(
            &[TensorMetadata::new(dtype, &input_shape, true)],
            &[OutputMetadata::from_view(&output)],
        )
        .and_then(WorkspacePreparation::require_ready)
        .unwrap();
    let mut buffer = (requirement.bytes != 0).then(|| {
        ep.allocate_workspace(
            requirement.bytes as usize,
            requirement.alignment,
            requirement.role(),
        )
        .unwrap()
    });
    let workspace = buffer
        .as_mut()
        .map(|buffer| WorkspaceView::new(DevicePtrMut(buffer.as_mut_ptr()), buffer.len()));
    assert_eq!(
        workspace.is_some(),
        node.attr("dilations").is_some_and(|attr| attr
            .as_ints()
            .unwrap()
            .iter()
            .any(|&value| value != 1))
    );
    let inputs = [input];
    let mut outputs = [output];
    execute(&mut || {
        kernel
            .execute_with_workspace(&inputs, &mut outputs, workspace)
            .unwrap();
    });
    if let Some(buffer) = buffer {
        ep.deallocate_workspace(buffer).unwrap();
    }

    let mut bytes = vec![0; output_shape.iter().product::<usize>() * dtype.byte_size()];
    unsafe {
        ep.runtime()
            .dtoh(&mut bytes, cuptr(output_buf.as_ptr()))
            .unwrap();
    }
    ep.deallocate(input_buf).unwrap();
    ep.deallocate(output_buf).unwrap();
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|bytes| f16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        DataType::BFloat16 => bytes
            .chunks_exact(2)
            .map(|bytes| bf16::from_bits(u16::from_le_bytes(bytes.try_into().unwrap())).to_f32())
            .collect(),
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn require_cuda() -> CudaExecutionProvider {
    match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => panic!(
            "CUDA test requires CUDA device/runtime; CPU-only runs must leave this test ignored: {error}"
        ),
        Err(_) => panic!(
            "CUDA test requires CUDA runtime libraries; CPU-only runs must leave this test ignored"
        ),
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn max_unpool_cuda_pinned_reference_and_bounds() {
    use common::{absent_input, float_input, input, raw, run_cuda, try_run_cuda};

    let ep = require_cuda();
    let f32_dtype = DataType::Float32;
    let i64_dtype = DataType::Int64;
    let attrs = [
        ("kernel_shape", Attribute::Ints(vec![2])),
        ("strides", Attribute::Ints(vec![2])),
    ];
    let shape = [1, 1, 2];
    let out = vec![1, 1, 4];
    // Pinned ONNX export_1d, including older schema versions and an omitted slot.
    for (version, dtype) in [
        (9, DataType::Float16),
        (11, f32_dtype),
        (22, DataType::BFloat16),
    ] {
        let actual = run_cuda(
            &ep,
            "MaxUnpool",
            "",
            version,
            &[
                float_input(dtype, &shape, &[1., 2.]),
                input(i64_dtype, &shape, &[1i64, 3]),
                absent_input(i64_dtype),
            ],
            &[(dtype, out.clone())],
            &attrs,
        );
        assert_eq!(actual[0], common::encode_floats(&[0., 1., 0., 2.], dtype));
    }

    // Pinned reference's last-source rule, with an explicit shape overriding
    // asymmetric pads and an f64 payload that cannot round-trip through f32.
    let attrs = [
        ("kernel_shape", Attribute::Ints(vec![2])),
        ("strides", Attribute::Ints(vec![2])),
        ("pads", Attribute::Ints(vec![0, 1])),
    ];
    let dtype = DataType::Float64;
    let precise = f64::from_bits(0x3ff0000000000001);
    let x = input(dtype, &shape, &[7., precise]);
    let control = input(i64_dtype, &[3], &[1i64, 1, 5]);
    let outputs = [(dtype, vec![1, 1, 5])];
    let actual = run_cuda(
        &ep,
        "MaxUnpool",
        "",
        22,
        &[
            x.clone(),
            input(i64_dtype, &shape, &[4i64, 4]),
            control.clone(),
        ],
        &outputs,
        &attrs,
    );
    assert_eq!(actual[0], raw(&[0., 0., 0., 0., precise]));
    for bad in [-1, 5] {
        let error = try_run_cuda(
            &ep,
            "MaxUnpool",
            "",
            22,
            &[
                x.clone(),
                input(i64_dtype, &shape, &[4i64, bad]),
                control.clone(),
            ],
            &outputs,
            &attrs,
        )
        .unwrap_err();
        assert!(
            error.to_string().contains(&format!("I[1]={bad}")),
            "{error}"
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cudnn_maxpool_matches_cpu_for_f32_and_f16() {
    let ep = require_cuda();
    let input = [
        1.0, 3.0, 2.0, 0.0, 4.0, 6.0, 5.0, 1.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 1.0, 0.0,
    ];
    let expected = [6.0, 5.0, 8.0, 9.0];
    for (dtype, tolerance) in [(DataType::Float32, 1e-6), (DataType::Float16, 1e-3)] {
        let (graph, node) = build_pool_model(
            "MaxPool",
            dtype,
            &[1, 1, 4, 4],
            &[1, 1, 2, 2],
            [2, 2],
            [2, 2],
            [0, 0, 0, 0],
            None,
        );
        let model = Model::new(&graph);
        assert_close(&run_model(&ep, &model, node, &input), &expected, tolerance);
    }
    println!("cuDNN MaxPool 2x2 stride-2 f32/f16 cases passed");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cudnn_averagepool_padding_count_modes_match_cpu() {
    let ep = require_cuda();
    let input = [1.0, 2.0, 3.0, 4.0];
    for (count_include_pad, expected) in [
        (0, [2.5, 2.5, 2.5, 2.5]),
        (1, [10.0 / 9.0, 10.0 / 9.0, 10.0 / 9.0, 10.0 / 9.0]),
    ] {
        let (mut graph, node) = build_pool_model(
            "AveragePool",
            DataType::Float32,
            &[1, 1, 2, 2],
            &[1, 1, 2, 2],
            [3, 3],
            [1, 1],
            [1, 1, 1, 1],
            Some(count_include_pad),
        );
        let model = Model::new(&graph);
        assert_close(&run_model(&ep, &model, node, &input), &expected, 1e-5);
        graph.opset_imports.insert(String::new(), 19);
        graph
            .node_mut(node)
            .attributes
            .insert("dilations".into(), Attribute::Ints(vec![1, 1]));
        assert_close(
            &run_model(&ep, &Model::new(&graph), node, &input),
            &expected,
            1e-5,
        );
    }

    println!("cuDNN AveragePool padded include/exclude-padding cases passed");
}

#[test]
#[ignore = "physical CUDA timing: run prebuilt under the host benchmark lock"]
fn averagepool_legacy_and_dilated_eager_latency() {
    use std::time::{Duration, Instant};

    let ep = require_cuda();
    let shape = [1, 16, 64, 64];
    let output_shape = [1, 16, 32, 32];
    let input = vec![2.; shape.iter().product()];
    let benchmark = |opset, dilation: Option<i64>, warm_for| {
        let pad = i64::from(dilation.is_some_and(|value| value != 1));
        let (mut graph, node) = build_pool_model(
            "AveragePool",
            DataType::Float32,
            &shape,
            &output_shape,
            [2, 2],
            [2, 2],
            [pad; 4],
            None,
        );
        graph.opset_imports.insert(String::new(), opset);
        if let Some(dilation) = dilation {
            graph
                .node_mut(node)
                .attributes
                .insert("dilations".into(), Attribute::Ints(vec![dilation; 2]));
        }
        let mut us = 0.0;
        let got = run_model_with(&ep, &Model::new(&graph), node, &input, |run| {
            let warm = Instant::now();
            loop {
                run();
                if warm.elapsed() >= warm_for {
                    break;
                }
            }
            // These are synchronous, uncaptured calls. The elapsed time
            // includes host dispatch and stream completion, not just kernels.
            let start = Instant::now();
            for _ in 0..100 {
                run();
            }
            us = start.elapsed().as_secs_f64() * 1e4;
        });
        assert_eq!(got.len(), output_shape.iter().product::<usize>());
        assert!(got.iter().all(|&value| value == 2.));
        us
    };
    let a0 = benchmark(17, None, Duration::from_secs(8));
    for iteration in 0..5 {
        let mut arms = [(17, None), (19, Some(1)), (19, Some(2))];
        if iteration % 2 == 1 {
            arms.reverse();
        }
        for (opset, dilation) in arms {
            let us = benchmark(opset, dilation, Duration::ZERO);
            println!("rep={iteration} opset={opset} dilation={dilation:?} {us:.2} us/run");
        }
    }
    let a1 = benchmark(17, None, Duration::ZERO);
    println!("Legacy A/A warm control: {a0:.2} / {a1:.2} us/run");
}
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cuda_dilated_averagepool_matches_cpu_and_expected_vectors() {
    let ep = require_cuda();
    for (graph, node, input, expected) in dilated_cases() {
        assert_close(
            &run_model(&ep, &Model::new(&graph), node, &input),
            &expected,
            1e-5,
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cuda_lp_pool_shared_window_keeps_power_reduction() {
    let ep = require_cuda();
    let (mut graph, node) = build_pool_model(
        "LpPool",
        DataType::Float32,
        &[1, 1, 4],
        &[1, 1, 2],
        [2],
        [1],
        [0, 0],
        None,
    );
    graph.opset_imports.insert(String::new(), 18);
    graph
        .node_mut(node)
        .attributes
        .insert("dilations".into(), Attribute::Ints(vec![2]));
    assert_close(
        &run_model(&ep, &Model::new(&graph), node, &[-3., -5., 4., 12.]),
        &[5., 13.],
        1e-5,
    );
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cuda_averagepool_claims_only_supported_dilated_variants() {
    let ep = require_cuda();
    let (mut graph, id) = build_pool_model(
        "AveragePool",
        DataType::Float32,
        &[1, 1, 4, 4],
        &[1, 1, 2, 2],
        [2, 2],
        [2, 2],
        [0, 0, 0, 0],
        None,
    );
    graph
        .node_mut(id)
        .attributes
        .insert("dilations".into(), Attribute::Ints(vec![2, 2]));
    let node = graph.node(id);
    let error = match ep.get_kernel(node, &[vec![1, 1, 4, 4]], 18) {
        Ok(_) => panic!("AveragePool-18 cannot use dilations"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("requires ONNX opset 19"));
    let shape = static_shape([1, 1, 4, 4]);
    for (opset, dtype, supported) in [
        (18, DataType::Float32, false),
        (19, DataType::Float32, true),
        (19, DataType::Float64, false),
        (19, DataType::BFloat16, false),
        (22, DataType::BFloat16, true),
    ] {
        let claim = ep.supports_op(node, opset, std::slice::from_ref(&shape), &[dtype], &[]);
        assert_eq!(
            claim.is_supported(),
            supported,
            "{opset} {dtype:?}: {claim:?}"
        );
    }
}
