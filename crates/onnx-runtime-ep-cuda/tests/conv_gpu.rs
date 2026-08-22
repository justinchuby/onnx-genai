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
//! GPU parity tests for the cuDNN-backed ONNX `Conv` kernel.

mod common;

use std::sync::Arc;

use half::f16;
use onnx_runtime_ep_api::{
    DevicePtr, DevicePtrMut, ExecutionProvider, TensorMetadata, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, as_static_shape,
    compute_contiguous_strides, static_shape,
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
        other => panic!("unsupported test dtype {other:?}"),
    }
}

fn initializer(
    g: &mut Graph,
    name: &str,
    dtype: DataType,
    shape: &[usize],
    values: &[f32],
) -> ValueId {
    let value = g.create_named_value(name, dtype, static_shape(shape.iter().copied()));
    g.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(
            dtype,
            shape.to_vec(),
            tensor_bytes(dtype, values),
        )),
    );
    value
}

fn build_conv_model(
    x_shape: &[usize],
    w_shape: &[usize],
    y_shape: &[usize],
    weights: &[f32],
    bias: &[f32],
    dtype: DataType,
    strides: [i64; 2],
    pads: [i64; 4],
    dilations: [i64; 2],
    group: i64,
) -> (Graph, NodeId) {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = graph.create_named_value("X", dtype, static_shape(x_shape.iter().copied()));
    graph.add_input(x);
    let w = initializer(&mut graph, "W", dtype, w_shape, weights);
    let b = initializer(&mut graph, "B", dtype, &[w_shape[0]], bias);
    let y = graph.create_named_value("Y", dtype, static_shape(y_shape.iter().copied()));
    let mut conv = Node::new(NodeId(0), "Conv", vec![Some(x), Some(w), Some(b)], vec![y]);
    conv.attributes
        .insert("strides".into(), Attribute::Ints(strides.to_vec()));
    conv.attributes
        .insert("pads".into(), Attribute::Ints(pads.to_vec()));
    conv.attributes
        .insert("dilations".into(), Attribute::Ints(dilations.to_vec()));
    conv.attributes
        .insert("group".into(), Attribute::Int(group));
    let node = graph.insert_node(conv);
    graph.add_output(y);
    (graph, node)
}

fn cpu_conv(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    x_shape: [usize; 4],
    w_shape: [usize; 4],
    y_shape: [usize; 4],
    stride: [usize; 2],
    pad: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> Vec<f32> {
    let mut y = vec![0.0; y_shape.iter().product()];
    let outputs_per_group = w_shape[0] / groups;
    for n in 0..y_shape[0] {
        for oc in 0..y_shape[1] {
            let group = oc / outputs_per_group;
            for oh in 0..y_shape[2] {
                for ow in 0..y_shape[3] {
                    let mut sum = bias[oc];
                    for ic_local in 0..w_shape[1] {
                        let ic = group * w_shape[1] + ic_local;
                        for kh in 0..w_shape[2] {
                            for kw in 0..w_shape[3] {
                                let ih = oh * stride[0] + kh * dilation[0];
                                let iw = ow * stride[1] + kw * dilation[1];
                                if ih >= pad[0] && iw >= pad[1] {
                                    let ih = ih - pad[0];
                                    let iw = iw - pad[1];
                                    if ih < x_shape[2] && iw < x_shape[3] {
                                        let xi = ((n * x_shape[1] + ic) * x_shape[2] + ih)
                                            * x_shape[3]
                                            + iw;
                                        let wi = ((oc * w_shape[1] + ic_local) * w_shape[2] + kh)
                                            * w_shape[3]
                                            + kw;
                                        sum += x[xi] * w[wi];
                                    }
                                }
                            }
                        }
                    }
                    let yi = ((n * y_shape[1] + oc) * y_shape[2] + oh) * y_shape[3] + ow;
                    y[yi] = sum;
                }
            }
        }
    }
    y
}

fn run_model(
    ep: &CudaExecutionProvider,
    model: &Model<'_>,
    node_id: NodeId,
    x: &[f32],
) -> Vec<f32> {
    let graph = model.graph;
    let node = graph.node(node_id);
    let x_id = node.inputs[0].unwrap();
    let w_id = node.inputs[1].unwrap();
    let b_id = node.inputs[2].unwrap();
    let y_id = node.outputs[0];
    let x_shape = as_static_shape(&graph.value(x_id).shape).unwrap();
    let y_shape = as_static_shape(&graph.value(y_id).shape).unwrap();
    let dtype = graph.value(x_id).dtype;
    let WeightRef::Inline(w) = &graph.initializers[&w_id] else {
        panic!("test weight must be inline")
    };
    let WeightRef::Inline(b) = &graph.initializers[&b_id] else {
        panic!("test bias must be inline")
    };

    let x_buf = ep.allocate(x.len() * dtype.byte_size(), 256).unwrap();
    let w_buf = ep.allocate(w.data.len(), 256).unwrap();
    let b_buf = ep.allocate(b.data.len(), 256).unwrap();
    let mut y_buf = ep
        .allocate(y_shape.iter().product::<usize>() * dtype.byte_size(), 256)
        .unwrap();
    unsafe {
        ep.runtime()
            .htod(&tensor_bytes(dtype, x), cuptr(x_buf.as_ptr()))
            .unwrap();
        ep.runtime().htod(&w.data, cuptr(w_buf.as_ptr())).unwrap();
        ep.runtime().htod(&b.data, cuptr(b_buf.as_ptr())).unwrap();
    }

    let x_strides = compute_contiguous_strides(&x_shape);
    let w_strides = compute_contiguous_strides(&w.dims);
    let b_strides = compute_contiguous_strides(&b.dims);
    let y_strides = compute_contiguous_strides(&y_shape);
    let dev = ep.device_id();
    let inputs = [
        TensorView::new(DevicePtr(x_buf.as_ptr()), dtype, &x_shape, &x_strides, dev),
        TensorView::new(DevicePtr(w_buf.as_ptr()), dtype, &w.dims, &w_strides, dev),
        TensorView::new(DevicePtr(b_buf.as_ptr()), dtype, &b.dims, &b_strides, dev),
    ];
    let output = TensorMut::new(
        DevicePtrMut(y_buf.as_mut_ptr()),
        dtype,
        &y_shape,
        &y_strides,
        dev,
    );
    let kernel = ep
        .get_kernel(node, &[x_shape.clone(), w.dims.clone(), b.dims.clone()], 17)
        .unwrap();
    let mut workspace = None;
    common::execute_kernel(ep, kernel.as_ref(), &inputs, &mut [output], &mut workspace).unwrap();

    let mut bytes = vec![0; y_shape.iter().product::<usize>() * dtype.byte_size()];
    unsafe {
        ep.runtime()
            .dtoh(&mut bytes, cuptr(y_buf.as_ptr()))
            .unwrap()
    };
    if let Some(workspace) = workspace {
        ep.deallocate_workspace(workspace).unwrap();
    }
    ep.deallocate(x_buf).unwrap();
    ep.deallocate(w_buf).unwrap();
    ep.deallocate(b_buf).unwrap();
    ep.deallocate(y_buf).unwrap();
    match dtype {
        DataType::Float32 => bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect(),
        DataType::Float16 => bytes
            .chunks_exact(2)
            .map(|v| f16::from_bits(u16::from_le_bytes(v.try_into().unwrap())).to_f32())
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

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cudnn_conv_matches_cpu_for_padding_groups_and_f16() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let x1: Vec<f32> = (1..=25).map(|v| v as f32 / 10.0).collect();
    let w1 = vec![1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0];
    let b1 = [0.25];
    let (g1, n1) = build_conv_model(
        &[1, 1, 5, 5],
        &[1, 1, 3, 3],
        &[1, 1, 5, 5],
        &w1,
        &b1,
        DataType::Float32,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
    );
    let model1 = Model::new(&g1);
    let got1 = run_model(&ep, &model1, n1, &x1);
    let expected1 = cpu_conv(
        &x1,
        &w1,
        &b1,
        [1, 1, 5, 5],
        [1, 1, 3, 3],
        [1, 1, 5, 5],
        [1, 1],
        [1, 1],
        [1, 1],
        1,
    );
    assert_close(&got1, &expected1, 1e-4);

    let x2: Vec<f32> = (0..32).map(|v| (v as f32 - 8.0) / 7.0).collect();
    let w2 = vec![
        1.0, 0.0, 0.0, 1.0, // group 0, output 0
        0.5, -0.5, -0.5, 0.5, // group 0, output 1
        -1.0, 0.0, 0.0, -1.0, // group 1, output 2
        0.25, 0.5, 0.75, 1.0, // group 1, output 3
    ];
    let b2 = [0.0, 0.1, -0.2, 0.3];
    let (g2, n2) = build_conv_model(
        &[1, 2, 4, 4],
        &[4, 1, 2, 2],
        &[1, 4, 2, 2],
        &w2,
        &b2,
        DataType::Float32,
        [2, 2],
        [0, 0, 0, 0],
        [1, 1],
        2,
    );
    let model2 = Model::new(&g2);
    let got2 = run_model(&ep, &model2, n2, &x2);
    let expected2 = cpu_conv(
        &x2,
        &w2,
        &b2,
        [1, 2, 4, 4],
        [4, 1, 2, 2],
        [1, 4, 2, 2],
        [2, 2],
        [0, 0],
        [1, 1],
        2,
    );
    assert_close(&got2, &expected2, 1e-4);

    let x3: Vec<f32> = x1
        .iter()
        .map(|&value| f16::from_f32(value).to_f32())
        .collect();
    let w3: Vec<f32> = w1
        .iter()
        .map(|&value| f16::from_f32(value).to_f32())
        .collect();
    let b3: Vec<f32> = b1
        .iter()
        .map(|&value| f16::from_f32(value).to_f32())
        .collect();
    let (g3, n3) = build_conv_model(
        &[1, 1, 5, 5],
        &[1, 1, 3, 3],
        &[1, 1, 5, 5],
        &w3,
        &b3,
        DataType::Float16,
        [1, 1],
        [1, 1, 1, 1],
        [1, 1],
        1,
    );
    let model3 = Model::new(&g3);
    let got3 = run_model(&ep, &model3, n3, &x3);
    let expected3 = cpu_conv(
        &x3,
        &w3,
        &b3,
        [1, 1, 5, 5],
        [1, 1, 3, 3],
        [1, 1, 5, 5],
        [1, 1],
        [1, 1],
        [1, 1],
        1,
    );
    assert_close(&got3, &expected3, 2e-2);
    println!("cuDNN Conv f32/f16 padded and grouped-strided cases passed");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cudnn_conv_matches_cpu_for_dilation() {
    let ep = match std::panic::catch_unwind(CudaExecutionProvider::new_default) {
        Ok(Ok(ep)) => ep,
        Ok(Err(error)) => {
            eprintln!("skip: no CUDA GPU/runtime available ({error})");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
        Err(_) => {
            eprintln!("skip: CUDA runtime library loading panicked");
            panic!(
                "CUDA test path did not run; this must be reported as a failed GPU test, not a pass"
            );
        }
    };

    let x: Vec<f32> = (1..=49).map(|v| v as f32 / 10.0).collect();
    let w = vec![1.0, 0.5, -1.0, 0.25, 0.0, -0.25, 1.0, -0.5, 0.75];
    let bias = [0.125];
    let (graph, node) = build_conv_model(
        &[1, 1, 7, 7],
        &[1, 1, 3, 3],
        &[1, 1, 3, 3],
        &w,
        &bias,
        DataType::Float32,
        [1, 1],
        [0, 0, 0, 0],
        [2, 2],
        1,
    );
    let model = Model::new(&graph);
    let got = run_model(&ep, &model, node, &x);
    let expected = cpu_conv(
        &x,
        &w,
        &bias,
        [1, 1, 7, 7],
        [1, 1, 3, 3],
        [1, 1, 3, 3],
        [1, 1],
        [0, 0],
        [2, 2],
        1,
    );
    assert_close(&got, &expected, 1e-4);
    println!("cuDNN Conv dilations=[2, 2] case passed");
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cuda_conv1d_matches_cpu_for_standard_grouped_and_causal_geometry() {
    let ep = common::require_cuda();
    let cases = [
        (
            "basic",
            DataType::Float32,
            vec![1, 2, 7],
            vec![3, 2, 3],
            vec![1, 3, 5],
            1,
            [0, 0],
            1,
            1,
            false,
        ),
        (
            "depthwise-causal",
            DataType::Float16,
            vec![1, 4, 8],
            vec![4, 1, 4],
            vec![1, 4, 8],
            1,
            [3, 0],
            1,
            4,
            true,
        ),
        (
            "grouped-strided-dilated",
            DataType::Float32,
            vec![1, 4, 12],
            vec![6, 2, 3],
            vec![1, 6, 6],
            2,
            [2, 1],
            2,
            2,
            true,
        ),
    ];

    for (label, dtype, x_shape, w_shape, y_shape, stride, pads, dilation, group, with_bias) in cases
    {
        let x_values = (0..x_shape.iter().product::<usize>())
            .map(|index| (index as f32 - 9.0) / 7.0)
            .collect::<Vec<_>>();
        let w_values = (0..w_shape.iter().product::<usize>())
            .map(|index| ((index % 11) as f32 - 5.0) / 9.0)
            .collect::<Vec<_>>();
        let bias_values = (0..w_shape[0])
            .map(|index| index as f32 / 13.0 - 0.2)
            .collect::<Vec<_>>();
        let mut inputs = vec![
            common::float_input(dtype, &x_shape, &x_values),
            common::float_input(dtype, &w_shape, &w_values),
        ];
        if with_bias {
            inputs.push(common::float_input(dtype, &[w_shape[0]], &bias_values));
        }
        let attrs = [
            ("strides", Attribute::Ints(vec![stride])),
            ("pads", Attribute::Ints(vec![pads[0], pads[1]])),
            ("dilations", Attribute::Ints(vec![dilation])),
            ("group", Attribute::Int(group)),
            ("kernel_shape", Attribute::Ints(vec![w_shape[2] as i64])),
        ];
        let outputs = [(dtype, y_shape)];
        let expected = common::run_cpu("Conv", "", 17, &inputs, &outputs, &attrs);
        let got = common::run_cuda(&ep, "Conv", "", 17, &inputs, &outputs, &attrs);
        common::assert_close(
            label,
            dtype,
            &common::decode_floats(&got[0], dtype),
            &common::decode_floats(&expected[0], dtype),
            if dtype == DataType::Float16 {
                2e-2
            } else {
                1e-4
            },
        );
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn cudnn_conv_prepared_workspace_reuses_the_injected_allocator_and_captures() {
    let injected = Arc::new(common::ExternalEagerAllocator::new(
        common::require_context("cuDNN Conv prepared-workspace allocator substitution"),
    ));
    let ep =
        common::require_cuda()
            .with_memory(
                Arc::clone(&injected) as Arc<dyn onnx_runtime_memory_governor::DeviceAllocator>
            )
            .expect("the injected allocator must be accepted for the visible CUDA device");
    let runtime = ep.runtime();
    let dtype = DataType::Float32;
    let candidates = [
        (
            vec![1usize, 16, 32, 32],
            vec![32usize, 16, 3, 3],
            vec![1usize, 32, 32, 32],
            [1, 1],
            [1, 1, 1, 1],
        ),
        (
            vec![1usize, 32, 64, 64],
            vec![64usize, 32, 5, 5],
            vec![1usize, 64, 64, 64],
            [1, 1],
            [2, 2, 2, 2],
        ),
        (
            vec![1usize, 32, 128, 128],
            vec![64usize, 32, 7, 7],
            vec![1usize, 64, 128, 128],
            [1, 1],
            [3, 3, 3, 3],
        ),
    ];

    for (x_shape, w_shape, y_shape, strides, pads) in candidates {
        let x_values = (0..x_shape.iter().product::<usize>())
            .map(|index| (index as f32 * 0.01).sin() + (index % 7) as f32 * 0.05)
            .collect::<Vec<_>>();
        let w_values = (0..w_shape.iter().product::<usize>())
            .map(|index| ((index % 17) as f32 - 8.0) * 0.02)
            .collect::<Vec<_>>();
        let bias_values = (0..w_shape[0])
            .map(|index| index as f32 * 0.01 - 0.2)
            .collect::<Vec<_>>();
        let (graph, node_id) = build_conv_model(
            &x_shape,
            &w_shape,
            &y_shape,
            &w_values,
            &bias_values,
            dtype,
            strides,
            pads,
            [1, 1],
            1,
        );
        let model = Model::new(&graph);
        let kernel = ep
            .get_kernel(
                model.graph.node(node_id),
                &[x_shape.clone(), w_shape.clone(), vec![w_shape[0]]],
                17,
            )
            .unwrap();
        let b_shape = [w_shape[0]];
        let metadata = [
            TensorMetadata::new(dtype, &x_shape, true),
            TensorMetadata::new(dtype, &w_shape, true),
            TensorMetadata::new(dtype, &b_shape, true),
        ];
        let requirement = kernel
            .workspace_requirement(&metadata)
            .expect("workspace requirement");
        if requirement.bytes == 0 {
            continue;
        }

        let x = ep
            .allocate(x_values.len() * dtype.byte_size(), 256)
            .expect("allocate X");
        let w = ep
            .allocate(w_values.len() * dtype.byte_size(), 256)
            .expect("allocate W");
        let b = ep
            .allocate(bias_values.len() * dtype.byte_size(), 256)
            .expect("allocate B");
        let mut eager = ep
            .allocate(y_shape.iter().product::<usize>() * dtype.byte_size(), 256)
            .expect("allocate eager Y");
        let mut captured = ep
            .allocate(y_shape.iter().product::<usize>() * dtype.byte_size(), 256)
            .expect("allocate captured Y");
        unsafe {
            runtime
                .htod(&tensor_bytes(dtype, &x_values), cuptr(x.as_ptr()))
                .unwrap();
            runtime
                .htod(&tensor_bytes(dtype, &w_values), cuptr(w.as_ptr()))
                .unwrap();
            runtime
                .htod(&tensor_bytes(dtype, &bias_values), cuptr(b.as_ptr()))
                .unwrap();
        }

        let x_strides = compute_contiguous_strides(&x_shape);
        let w_strides = compute_contiguous_strides(&w_shape);
        let b_strides = compute_contiguous_strides(&b_shape);
        let y_strides = compute_contiguous_strides(&y_shape);
        let device = ep.device_id();
        let inputs = [
            TensorView::new(DevicePtr(x.as_ptr()), dtype, &x_shape, &x_strides, device),
            TensorView::new(DevicePtr(w.as_ptr()), dtype, &w_shape, &w_strides, device),
            TensorView::new(DevicePtr(b.as_ptr()), dtype, &b_shape, &b_strides, device),
        ];
        let calls_after_buffers = injected.cumemalloc_calls();
        let mut workspace = None;
        {
            let output = TensorMut::new(
                DevicePtrMut(eager.as_mut_ptr()),
                dtype,
                &y_shape,
                &y_strides,
                device,
            );
            common::execute_kernel(&ep, kernel.as_ref(), &inputs, &mut [output], &mut workspace)
                .expect("first eager cuDNN Conv execute");
        }
        assert!(
            workspace.is_some(),
            "positive workspace requirement must allocate"
        );
        let calls_after_first = injected.cumemalloc_calls();
        assert_eq!(
            calls_after_first,
            calls_after_buffers + 1,
            "the first conv execute must allocate exactly one prepared workspace through the injected allocator"
        );
        {
            let output = TensorMut::new(
                DevicePtrMut(captured.as_mut_ptr()),
                dtype,
                &y_shape,
                &y_strides,
                device,
            );
            common::execute_kernel(&ep, kernel.as_ref(), &inputs, &mut [output], &mut workspace)
                .expect("second eager cuDNN Conv execute");
        }
        let calls_after_second = injected.cumemalloc_calls();
        assert_eq!(
            calls_after_second, calls_after_first,
            "repeated conv executes must reuse the prepared workspace"
        );
        let mut eager_bytes = vec![0u8; y_shape.iter().product::<usize>() * dtype.byte_size()];
        let mut repeat_bytes = vec![0u8; eager_bytes.len()];
        unsafe {
            runtime
                .dtoh(&mut eager_bytes, cuptr(eager.as_ptr()))
                .unwrap();
            runtime
                .dtoh(&mut repeat_bytes, cuptr(captured.as_ptr()))
                .unwrap();
        }
        assert_eq!(
            repeat_bytes, eager_bytes,
            "reusing the prepared workspace changed the convolution output"
        );
        assert!(
            kernel.cuda_graph_compatible(),
            "a warmed cuDNN Conv with prepared workspace must be capture-supported"
        );

        let kernels: [&dyn onnx_runtime_ep_api::Kernel; 1] = [kernel.as_ref()];
        runtime
            .begin_graph_capture(&kernels)
            .expect("begin Conv CUDA graph capture");
        {
            let output = TensorMut::new(
                DevicePtrMut(captured.as_mut_ptr()),
                dtype,
                &y_shape,
                &y_strides,
                device,
            );
            common::execute_kernel(&ep, kernel.as_ref(), &inputs, &mut [output], &mut workspace)
                .expect("record Conv with prepared workspace");
        }
        runtime
            .end_graph_capture()
            .expect("end Conv CUDA graph capture");
        runtime.replay_graph().expect("replay captured Conv");
        let calls_after_capture = injected.cumemalloc_calls();
        assert_eq!(
            calls_after_capture, calls_after_second,
            "recording or replaying the Conv graph must not allocate a new workspace"
        );
        let mut captured_bytes = vec![0u8; eager_bytes.len()];
        unsafe {
            runtime
                .dtoh(&mut captured_bytes, cuptr(captured.as_ptr()))
                .unwrap();
        }
        assert_eq!(
            captured_bytes, eager_bytes,
            "captured Conv output diverged from eager output"
        );
        assert!(runtime.reset_graph().unwrap(), "reset captured Conv graph");

        if let Some(workspace) = workspace.take() {
            ep.deallocate_workspace(workspace).unwrap();
        }
        for buffer in [captured, eager, b, w, x] {
            ep.deallocate(buffer).unwrap();
        }
        common::drain_releases(&ep, "Conv prepared workspace teardown");
        assert_eq!(
            injected.frees(),
            injected.cumemalloc_calls(),
            "the injected allocator must observe every Conv buffer/workspace free"
        );
        return;
    }

    panic!(
        "no cuDNN Conv candidate selected a positive prepared workspace; update the test geometries"
    );
}
