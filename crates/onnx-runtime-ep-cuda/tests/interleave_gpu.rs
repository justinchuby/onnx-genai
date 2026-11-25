//! One process-local lever fixture: no other tests in this binary mutate env.
use half::f16;
use onnx_runtime_ep_api::{
    DeviceGraphSlot, DevicePtr, DevicePtrMut, ExecutionProvider, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::{CudaExecutionProvider, runtime::cuptr};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

fn require_cuda() -> CudaExecutionProvider {
    CudaExecutionProvider::new_default().expect(
        "interleave GPU tests require a CUDA device/runtime; leave ignored on CPU-only hosts",
    )
}

fn bytes(values: &[f16]) -> &[u8] {
    // SAFETY: f16 is POD and the view covers exactly the input slice.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)) }
}

struct Levers(Vec<(&'static str, Option<std::ffi::OsString>)>);
impl Levers {
    fn install() -> Self {
        let mut saved = Vec::new();
        for (key, value) in [
            ("ONNX_GENAI_INTERLEAVE_DEQUANT", "1"),
            ("ONNX_GENAI_GENERAL_SPLITK", "off"),
            ("ONNX_GENAI_GEMV_WIDELOAD", "1"),
            ("ONNX_GENAI_GEMV_WIDE_MULTICOL", "1"),
            ("ONNX_GENAI_GEMV_FP16", "0"),
        ] {
            saved.push((key, std::env::var_os(key)));
            // SAFETY: this binary has exactly one test, with no concurrent env writers.
            unsafe {
                std::env::set_var(key, value);
            }
        }
        Self(saved)
    }
}
impl Drop for Levers {
    fn drop(&mut self) {
        for (key, value) in &self.0 {
            // SAFETY: the sole test restores its own changes before exit.
            unsafe {
                match value {
                    Some(value) => std::env::set_var(key, value),
                    None => std::env::remove_var(key),
                }
            }
        }
    }
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn immutable_interleave_pins_replay_and_dynamic_weights_bypass_cached_copies() {
    let _levers = Levers::install();
    let ep = require_cuda();
    let runtime = ep.runtime();
    let device = ep.device_id();
    let (k, n, block_size) = (256usize, 256usize, 64usize);
    let blocks = k / block_size;
    let blob = block_size / 2;
    let packed_len = n * blocks * blob;
    let shapes = [vec![1, k], vec![n, blocks, blob], vec![n, blocks]];
    let strides: Vec<_> = shapes
        .iter()
        .map(|shape| compute_contiguous_strides(shape))
        .collect();
    let out_shape = [1, n];
    let out_strides = compute_contiguous_strides(&out_shape);
    let mut graph = Graph::new();
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let values: Vec<_> = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            let value = graph.create_named_value(
                format!("input{index}"),
                if index == 1 {
                    DataType::Uint8
                } else {
                    DataType::Float16
                },
                static_shape(shape.iter().copied()),
            );
            graph.add_input(value);
            value
        })
        .collect();
    let output = graph.create_named_value("Y", DataType::Float16, static_shape(out_shape));
    let mut node = Node::new(
        NodeId(0),
        "MatMulNBits",
        values.iter().copied().map(Some).collect(),
        vec![output],
    );
    node.domain = "com.microsoft".into();
    for (name, value) in [
        ("K", k),
        ("N", n),
        ("block_size", block_size),
        ("bits", 4),
        ("accuracy_level", 4),
    ] {
        node.attributes
            .insert(name.into(), Attribute::Int(value as i64));
    }
    let id = graph.insert_node(node);
    graph.add_output(output);
    let model = Model::new(&graph);
    let mut kernel = ep.get_kernel(model.graph.node(id), &[], 1).unwrap();
    kernel.set_constant_inputs(&[false, true, true]);
    let mut dynamic = ep.get_kernel(model.graph.node(id), &[], 1).unwrap();
    dynamic.set_constant_inputs(&[false, false, true]);
    let activation = ep.allocate(k * 2, 256).unwrap();
    let scales = ep.allocate(n * blocks * 2, 256).unwrap();
    let packed_a = ep.allocate(packed_len, 256).unwrap();
    let mut result = ep.allocate(n * 2, 256).unwrap();
    // SAFETY: each allocation exactly covers its source slice.
    unsafe {
        runtime
            .htod(bytes(&vec![f16::ONE; k]), cuptr(activation.as_ptr()))
            .unwrap();
        runtime
            .htod(
                bytes(&vec![f16::from_f32(1.0 / 32.0); n * blocks]),
                cuptr(scales.as_ptr()),
            )
            .unwrap();
        runtime
            .htod(&vec![0x99; packed_len], cuptr(packed_a.as_ptr()))
            .unwrap();
    }
    macro_rules! execute {
        ($kernel:expr, $packed:expr) => {{
            let inputs = [
                TensorView::new(
                    DevicePtr(activation.as_ptr()),
                    DataType::Float16,
                    &shapes[0],
                    &strides[0],
                    device,
                ),
                TensorView::new(
                    DevicePtr($packed.as_ptr()),
                    DataType::Uint8,
                    &shapes[1],
                    &strides[1],
                    device,
                ),
                TensorView::new(
                    DevicePtr(scales.as_ptr()),
                    DataType::Float16,
                    &shapes[2],
                    &strides[2],
                    device,
                ),
            ];
            $kernel.execute(
                &inputs,
                &mut [TensorMut::new(
                    DevicePtrMut(result.as_mut_ptr()),
                    DataType::Float16,
                    &out_shape,
                    &out_strides,
                    device,
                )],
            )
        }};
    }
    macro_rules! assert_output {
        ($expected:expr) => {{
            let mut actual = vec![0; n * 2];
            // SAFETY: result contains exactly n fp16 values, and dtoh waits for the stream.
            unsafe {
                runtime.dtoh(&mut actual, cuptr(result.as_ptr())).unwrap();
            }
            assert_eq!(actual, bytes(&vec![f16::from_f32($expected); n]));
        }};
    }
    execute!(kernel, packed_a).unwrap();
    assert_output!(8.0);
    assert!(
        !kernel.device_graph_resources().is_empty(),
        "test must actually select interleave"
    );
    execute!(kernel, packed_a).unwrap();
    let a_identity = kernel.device_graph_resources()[0].identity();
    runtime.begin_graph_capture(&[kernel.as_ref()]).unwrap();
    execute!(kernel, packed_a).unwrap();
    runtime.end_graph_capture().unwrap();
    ep.deallocate(packed_a).unwrap(); // Invalidates source key, graph retains derived bytes.

    let packed_b = ep.allocate(packed_len, 256).unwrap();
    // SAFETY: packed_b covers packed_len bytes.
    unsafe {
        runtime
            .htod(&vec![0xaa; packed_len], cuptr(packed_b.as_ptr()))
            .unwrap();
    }
    execute!(kernel, packed_b).unwrap();
    assert_output!(16.0);
    assert_ne!(kernel.device_graph_resources()[0].identity(), a_identity);
    execute!(kernel, packed_b).unwrap();
    let warm_counts = runtime.allocation_counts();
    for _ in 0..4 {
        execute!(kernel, packed_b).unwrap();
    }
    assert_eq!(
        runtime.allocation_counts(),
        warm_counts,
        "warm interleave must not allocate or retire backing"
    );
    runtime.replay_graph().unwrap();
    assert_output!(8.0); // A survives cache invalidation and the kernel's switch to B.

    // The B conversion exists, but omitting its actual owner must refuse capture.
    runtime
        .begin_graph_capture_in(DeviceGraphSlot::Verify, &[])
        .unwrap();
    let refusal = execute!(kernel, packed_b).unwrap_err().to_string();
    runtime
        .abort_graph_capture_in(DeviceGraphSlot::Verify)
        .unwrap();
    assert!(refusal.contains("interleaved packed B"), "{refusal}");
    assert!(
        !kernel.cuda_graph_compatible(),
        "failed launch cannot advertise capture safety"
    );
    execute!(kernel, packed_b).unwrap();
    runtime
        .begin_graph_capture_in(DeviceGraphSlot::Verify, &[kernel.as_ref()])
        .unwrap();
    execute!(kernel, packed_b).unwrap();
    runtime
        .end_graph_capture_in(DeviceGraphSlot::Verify)
        .unwrap();
    runtime.replay_graph_in(DeviceGraphSlot::Verify).unwrap();
    assert_output!(16.0);

    // Dynamic B has the exact same address/size as a cached constant, then
    // changes in place. Neither dispatch may read that cached constant copy.
    execute!(dynamic, packed_b).unwrap();
    assert_output!(16.0);
    assert!(dynamic.device_graph_resources().is_empty());
    // SAFETY: packed_b covers the replacement bytes; no graph reads this source.
    unsafe {
        runtime
            .htod(&vec![0xbb; packed_len], cuptr(packed_b.as_ptr()))
            .unwrap();
    }
    execute!(dynamic, packed_b).unwrap();
    assert_output!(24.0);
    runtime.replay_graph().unwrap();
    assert_output!(8.0);
    runtime.replay_graph_in(DeviceGraphSlot::Verify).unwrap();
    assert_output!(16.0);
    runtime.reset_graph().unwrap();
    runtime.reset_graph_in(DeviceGraphSlot::Verify).unwrap();
    drop(kernel);
    drop(dynamic);
    ep.deallocate(packed_b).unwrap();
    ep.deallocate(activation).unwrap();
    ep.deallocate(scales).unwrap();
    ep.deallocate(result).unwrap();
}
