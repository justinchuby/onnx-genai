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
//! CPU reference and physical CUDA tests for ONNX pooling.

use half::{bf16, f16};
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, OutputMetadata, TensorMetadata, TensorMut,
    TensorView, WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, as_static_shape, compute_contiguous_strides,
    static_shape,
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

fn build_pool_model(
    op_type: &str,
    dtype: DataType,
    input_shape: &[usize],
    output_shape: &[usize],
    kernel_shape: impl AsRef<[i64]>,
    strides: impl AsRef<[i64]>,
    pads: impl AsRef<[i64]>,
    count_include_pad: Option<i64>,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let input = graph.create_named_value("X", dtype, static_shape(input_shape.iter().copied()));
    graph.add_input(input);
    let output = graph.create_named_value("Y", dtype, static_shape(output_shape.iter().copied()));
    let mut pool = Node::new(NodeId(0), op_type, vec![Some(input)], vec![output]);
    pool.attributes.insert(
        "kernel_shape".into(),
        Attribute::Ints(kernel_shape.as_ref().to_vec()),
    );
    pool.attributes
        .insert("strides".into(), Attribute::Ints(strides.as_ref().to_vec()));
    pool.attributes
        .insert("pads".into(), Attribute::Ints(pads.as_ref().to_vec()));
    if let Some(value) = count_include_pad {
        pool.attributes
            .insert("count_include_pad".into(), Attribute::Int(value));
    }
    let node = graph.insert_node(pool);
    graph.add_output(output);
    (graph, node)
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
    let input_strides = compute_contiguous_strides(&input_shape);
    let output_strides = compute_contiguous_strides(&output_shape);
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
    let opset = graph.opset_imports[""] as u64;
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

fn assert_close(got: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(got.len(), expected.len());
    for (index, (&got, &expected)) in got.iter().zip(expected).enumerate() {
        assert!(
            (got - expected).abs() <= tolerance,
            "index {index}: got {got}, expected {expected}"
        );
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

#[test]
fn cpu_dilated_averagepool_matches_expected_vectors() {
    let ep = CpuExecutionProvider::new();
    for (graph, node_id, input, expected) in dilated_cases() {
        let node = graph.node(node_id);
        let shape = |id| as_static_shape(&graph.value(id).shape).unwrap();
        let input_shape = shape(node.inputs[0].unwrap());
        let output_shape = shape(node.outputs[0].unwrap());
        let dtype = graph.value(node.inputs[0].unwrap()).dtype;
        let narrow_input: Vec<u16> = match dtype {
            DataType::Float16 => input.iter().map(|&x| f16::from_f32(x).to_bits()).collect(),
            DataType::BFloat16 => input.iter().map(|&x| bf16::from_f32(x).to_bits()).collect(),
            _ => Vec::new(),
        };
        let input_ptr = if dtype == DataType::Float32 {
            input.as_ptr().cast()
        } else {
            narrow_input.as_ptr().cast()
        };
        // All test fixtures use f32 expected values representable in their input dtype.
        let mut got = vec![0.0_f32; expected.len()];
        let mut narrow = vec![0_u16; expected.len()];
        let output_ptr = if dtype == DataType::Float32 {
            got.as_mut_ptr().cast()
        } else {
            narrow.as_mut_ptr().cast()
        };
        let input_strides = compute_contiguous_strides(&input_shape);
        let output_strides = compute_contiguous_strides(&output_shape);
        let input = TensorView::new(
            DevicePtr(input_ptr),
            dtype,
            &input_shape,
            &input_strides,
            ep.device_id(),
        );
        let output = TensorMut::new(
            DevicePtrMut(output_ptr),
            dtype,
            &output_shape,
            &output_strides,
            ep.device_id(),
        );
        ep.get_kernel(node, &[input_shape.clone()], graph.opset_imports[""] as u64)
            .unwrap()
            .execute(&[input], &mut [output])
            .unwrap();
        if dtype != DataType::Float32 {
            for (value, bits) in got.iter_mut().zip(narrow) {
                *value = if dtype == DataType::Float16 {
                    f16::from_bits(bits).to_f32()
                } else {
                    bf16::from_bits(bits).to_f32()
                };
            }
        }
        assert_close(&got, &expected, 1e-5);
    }
}

fn dilated_cases() -> Vec<(Graph, NodeId, Vec<f32>, Vec<f32>)> {
    let mut cases = Vec::new();
    let mut add = |dtype,
                   input_shape: &[usize],
                   output_shape: &[usize],
                   kernel: &[i64],
                   strides: &[i64],
                   dilations: &[i64],
                   pads: &[i64],
                   auto_pad: Option<&str>,
                   ceil,
                   include_pad,
                   input,
                   expected| {
        let (mut graph, id) = build_pool_model(
            "AveragePool",
            dtype,
            input_shape,
            output_shape,
            kernel,
            strides,
            pads,
            Some(include_pad),
        );
        graph.opset_imports.insert(
            String::new(),
            if dtype == DataType::BFloat16 { 22 } else { 19 },
        );
        let node = graph.node_mut(id);
        node.attributes
            .insert("dilations".into(), Attribute::Ints(dilations.to_vec()));
        node.attributes
            .insert("ceil_mode".into(), Attribute::Int(ceil));
        if let Some(auto_pad) = auto_pad {
            node.attributes.remove("pads");
            node.attributes
                .insert("auto_pad".into(), Attribute::String(auto_pad.into()));
        }
        cases.push((graph, id, input, expected));
    };
    // ONNX v1.16 backend vector test_averagepool_2d_dilations (opset 19).
    for dtype in [DataType::Float32, DataType::Float16, DataType::BFloat16] {
        add(
            dtype,
            &[1, 1, 4, 4],
            &[1, 1, 2, 2],
            &[2, 2],
            &[1, 1],
            &[2, 2],
            &[0; 4],
            None,
            1,
            0,
            (1..=16).map(|i| i as f32).collect(),
            vec![6., 7., 10., 11.],
        );
    }
    // Two batches and channels, three spatial axes, mixed dilation, negative values.
    let groups = [0., 20., -20., 10.];
    add(
        DataType::Float32,
        &[2, 2, 4, 3, 5],
        &[2, 2, 2, 2, 2],
        &[2, 2, 2],
        &[1, 1, 1],
        &[2, 1, 3],
        &[],
        Some("VALID"),
        0,
        0,
        groups
            .into_iter()
            .flat_map(|bias| (0..60).map(move |i| i as f32 + bias - 30.))
            .collect(),
        groups
            .into_iter()
            .flat_map(|bias| {
                [-11., -10., -6., -5., 4., 5., 9., 10.]
                    .into_iter()
                    .map(move |x| x + bias)
            })
            .collect(),
    );
    // Explicit asymmetry: a retained partial window and a discarded trailing window.
    for (include_pad, expected) in [(0, vec![2., 3.]), (1, vec![1., 1.5])] {
        add(
            DataType::Float32,
            &[1, 1, 4],
            &[1, 1, 2],
            &[2],
            &[3],
            &[2],
            &[1, 3],
            None,
            1,
            include_pad,
            vec![1., 2., 3., 4.],
            expected,
        );
    }
    // VALID resolves to zero pads but still retains the ceil-mode overhang.
    add(
        DataType::Float32,
        &[1, 1, 4],
        &[1, 1, 2],
        &[2],
        &[2],
        &[2],
        &[],
        Some("VALID"),
        1,
        1,
        vec![1., 2., 7., 10.],
        vec![4., 3.5],
    );
    // Odd total padding splits to different sides for SAME_UPPER and SAME_LOWER.
    for (mode, expected) in [("SAME_UPPER", vec![4., 7.]), ("SAME_LOWER", vec![2., 6.])] {
        add(
            DataType::Float32,
            &[1, 1, 4],
            &[1, 1, 2],
            &[2],
            &[2],
            &[2],
            &[],
            Some(mode),
            0,
            0,
            vec![1., 2., 7., 10.],
            expected,
        );
    }
    cases
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
