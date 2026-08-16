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
//! CUDA-graph regression coverage for unfused default-domain RotaryEmbedding decode.

use half::f16;
use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const DECODE_STEPS: usize = 3;
const CACHE_ROWS: usize = 8;

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: test inputs are fixed-width plain data.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
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

fn standalone_rope_kernel(ep: &CudaExecutionProvider) -> Box<dyn Kernel> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 23);
    let x = graph.create_named_value(
        "decode_hidden_states",
        DataType::Float16,
        static_shape([1, HEADS, 1, HEAD_DIM]),
    );
    let cos = graph.create_named_value(
        "cos_cache",
        DataType::Float16,
        static_shape([CACHE_ROWS, HEAD_DIM / 2]),
    );
    let sin = graph.create_named_value(
        "sin_cache",
        DataType::Float16,
        static_shape([CACHE_ROWS, HEAD_DIM / 2]),
    );
    let position_ids =
        graph.create_named_value("position_ids", DataType::Int64, static_shape([1, 1]));
    for input in [x, cos, sin, position_ids] {
        graph.add_input(input);
    }
    let y = graph.create_named_value(
        "rope_hidden_states",
        DataType::Float16,
        static_shape([1, HEADS, 1, HEAD_DIM]),
    );
    graph.add_output(y);
    let mut node = Node::new(
        NodeId(0),
        "RotaryEmbedding",
        vec![Some(x), Some(cos), Some(sin), Some(position_ids)],
        vec![y],
    );
    node.attributes.insert(
        "rotary_embedding_dim".into(),
        Attribute::Int(HEAD_DIM as i64),
    );
    let node_id = graph.insert_node(node);
    let model = Model::new(&graph);
    ep.get_kernel(model.graph.node(node_id), &[], 23)
        .expect("default-domain standalone RotaryEmbedding must be supported")
}

fn execute_decode(
    kernel: &dyn Kernel,
    x: &DeviceBuffer,
    cos: &DeviceBuffer,
    sin: &DeviceBuffer,
    position_ids: &DeviceBuffer,
    output: &mut DeviceBuffer,
) {
    let x_shape = [1, HEADS, 1, HEAD_DIM];
    let cache_shape = [CACHE_ROWS, HEAD_DIM / 2];
    let position_shape = [1, 1];
    let x_strides = compute_contiguous_strides(&x_shape);
    let cache_strides = compute_contiguous_strides(&cache_shape);
    let position_strides = compute_contiguous_strides(&position_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(x.as_ptr()),
            DataType::Float16,
            &x_shape,
            &x_strides,
            x.device(),
        ),
        TensorView::new(
            DevicePtr(cos.as_ptr()),
            DataType::Float16,
            &cache_shape,
            &cache_strides,
            cos.device(),
        ),
        TensorView::new(
            DevicePtr(sin.as_ptr()),
            DataType::Float16,
            &cache_shape,
            &cache_strides,
            sin.device(),
        ),
        TensorView::new(
            DevicePtr(position_ids.as_ptr()),
            DataType::Int64,
            &position_shape,
            &position_strides,
            position_ids.device(),
        ),
    ];
    let mut outputs = [TensorMut::new(
        DevicePtrMut(output.as_mut_ptr()),
        DataType::Float16,
        &x_shape,
        &x_strides,
        output.device(),
    )];
    kernel
        .execute(&inputs, &mut outputs)
        .expect("standalone RotaryEmbedding decode");
}

fn read(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
    let mut host = vec![0; bytes];
    // SAFETY: `buffer` owns at least `bytes` bytes in every caller.
    unsafe {
        ep.runtime()
            .dtoh(&mut host, cuptr(buffer.as_ptr()))
            .expect("copy CUDA output to host");
    }
    host
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn standalone_rope_decode_captures_and_matches_eager_without_fallback() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let kernel = standalone_rope_kernel(&ep);
    let token_elements = HEADS * HEAD_DIM;
    let token_bytes = token_elements * std::mem::size_of::<f16>();
    let cache_bytes = CACHE_ROWS * (HEAD_DIM / 2) * std::mem::size_of::<f16>();

    let x = ep
        .allocate(token_bytes, 256)
        .expect("allocate decode input");
    let cos = ep
        .allocate(cache_bytes, 256)
        .expect("allocate cosine cache");
    let sin = ep.allocate(cache_bytes, 256).expect("allocate sine cache");
    let position_ids = ep
        .allocate(std::mem::size_of::<i64>(), 256)
        .expect("allocate position ids");
    let mut eager = ep
        .allocate(token_bytes, 256)
        .expect("allocate eager output");
    let mut captured = ep
        .allocate(token_bytes, 256)
        .expect("allocate captured output");

    let cos_values = (0..CACHE_ROWS * (HEAD_DIM / 2))
        .map(|index| f16::from_f32(0.55 + (index % 5) as f32 * 0.07))
        .collect::<Vec<_>>();
    let sin_values = (0..CACHE_ROWS * (HEAD_DIM / 2))
        .map(|index| f16::from_f32(-0.35 + (index % 7) as f32 * 0.06))
        .collect::<Vec<_>>();
    // SAFETY: cache allocations exactly cover the fixed-size cache tensors.
    unsafe {
        runtime
            .htod(&bytes(&cos_values), cuptr(cos.as_ptr()))
            .unwrap();
        runtime
            .htod(&bytes(&sin_values), cuptr(sin.as_ptr()))
            .unwrap();
    }

    for step in 0..DECODE_STEPS {
        let token = (0..token_elements)
            .map(|index| f16::from_f32(-0.4 + step as f32 * 0.13 + index as f32 * 0.03125))
            .collect::<Vec<_>>();
        let position = (step + 1) as i64;
        // SAFETY: fixed decode input and scalar position allocations exactly cover
        // these host tensors, which are uploaded before graph capture begins.
        unsafe {
            runtime.htod(&bytes(&token), cuptr(x.as_ptr())).unwrap();
            runtime
                .htod(
                    &bytes(std::slice::from_ref(&position)),
                    cuptr(position_ids.as_ptr()),
                )
                .unwrap();
        }

        // Warm the exact f16 decode signature. This proves the graph path is the
        // standalone default-domain RoPE kernel, rather than a fused attention op.
        execute_decode(kernel.as_ref(), &x, &cos, &sin, &position_ids, &mut eager);
        assert!(
            kernel.cuda_graph_compatible(),
            "warmed standalone RoPE decode must be capture-supported"
        );

        let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
        runtime
            .begin_graph_capture(&kernels)
            .expect("begin standalone RoPE CUDA graph capture");
        execute_decode(
            kernel.as_ref(),
            &x,
            &cos,
            &sin,
            &position_ids,
            &mut captured,
        );
        runtime
            .end_graph_capture()
            .expect("standalone RoPE decode must record without host fallback");
        assert!(
            runtime.has_graph_executable().unwrap(),
            "standalone RoPE decode did not install a CUDA graph"
        );
        runtime
            .replay_graph()
            .expect("replay captured standalone RoPE decode");

        assert_eq!(
            read(&ep, &captured, token_bytes),
            read(&ep, &eager, token_bytes),
            "captured standalone RoPE output diverged at decode step {step}"
        );
        assert_eq!(
            runtime.check_capture_error().unwrap(),
            0,
            "standalone RoPE capture error latched at decode step {step}"
        );
        assert!(
            runtime.reset_graph().unwrap(),
            "captured standalone RoPE graph was not installed"
        );
    }

    for buffer in [captured, eager, position_ids, sin, cos, x] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}
