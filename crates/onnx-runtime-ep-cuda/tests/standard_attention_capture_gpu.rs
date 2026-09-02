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
//! CUDA-graph regression coverage for aliased default-domain Attention KV growth.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, Graph, Node, NodeId, compute_contiguous_strides, static_shape,
};
use onnx_runtime_loader::Model;

const HEADS: usize = 2;
const HEAD_DIM: usize = 8;
const INITIAL_PAST: usize = 1;
const DECODE_STEPS: usize = 3;

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
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

fn standard_attention_kernel(ep: &CudaExecutionProvider) -> Box<dyn Kernel> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 23);
    let q = graph.create_named_value(
        "q",
        DataType::Float32,
        static_shape([1, HEADS, 1, HEAD_DIM]),
    );
    let k = graph.create_named_value(
        "k",
        DataType::Float32,
        static_shape([1, HEADS, 1, HEAD_DIM]),
    );
    let v = graph.create_named_value(
        "v",
        DataType::Float32,
        static_shape([1, HEADS, 1, HEAD_DIM]),
    );
    let past_key = graph.create_named_value(
        "past_key",
        DataType::Float32,
        static_shape([1, HEADS, INITIAL_PAST, HEAD_DIM]),
    );
    let past_value = graph.create_named_value(
        "past_value",
        DataType::Float32,
        static_shape([1, HEADS, INITIAL_PAST, HEAD_DIM]),
    );
    for input in [q, k, v, past_key, past_value] {
        graph.add_input(input);
    }
    let y = graph.create_named_value(
        "y",
        DataType::Float32,
        static_shape([1, HEADS, 1, HEAD_DIM]),
    );
    let present_key = graph.create_named_value(
        "present_key",
        DataType::Float32,
        static_shape([1, HEADS, INITIAL_PAST + 1, HEAD_DIM]),
    );
    let present_value = graph.create_named_value(
        "present_value",
        DataType::Float32,
        static_shape([1, HEADS, INITIAL_PAST + 1, HEAD_DIM]),
    );
    let mut node = Node::new(
        NodeId(0),
        "Attention",
        vec![
            Some(q),
            Some(k),
            Some(v),
            None,
            Some(past_key),
            Some(past_value),
        ],
        vec![y, present_key, present_value],
    );
    node.attributes
        .insert("is_causal".into(), Attribute::Int(1));
    let node_id = graph.insert_node(node);
    let model = Model::new(&graph);
    ep.get_kernel(model.graph.node(node_id), &[], 23)
        .expect("default-domain Attention must be supported")
}

fn execute_decode(
    kernel: &dyn Kernel,
    q: &DeviceBuffer,
    k: &DeviceBuffer,
    v: &DeviceBuffer,
    key_cache: &mut DeviceBuffer,
    value_cache: &mut DeviceBuffer,
    y: &mut DeviceBuffer,
    past_seq: usize,
) -> onnx_runtime_ep_api::Result<()> {
    let device = DeviceId::cuda(0);
    let q_shape = [1, HEADS, 1, HEAD_DIM];
    let past_shape = [1, HEADS, past_seq, HEAD_DIM];
    let present_shape = [1, HEADS, past_seq + 1, HEAD_DIM];
    let q_strides = compute_contiguous_strides(&q_shape);
    let past_strides = compute_contiguous_strides(&past_shape);
    let present_strides = compute_contiguous_strides(&present_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(q.as_ptr()),
            DataType::Float32,
            &q_shape,
            &q_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(k.as_ptr()),
            DataType::Float32,
            &q_shape,
            &q_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(v.as_ptr()),
            DataType::Float32,
            &q_shape,
            &q_strides,
            device,
        ),
        TensorView::absent(DataType::Float32),
        TensorView::new(
            DevicePtr(key_cache.as_ptr()),
            DataType::Float32,
            &past_shape,
            &past_strides,
            device,
        ),
        TensorView::new(
            DevicePtr(value_cache.as_ptr()),
            DataType::Float32,
            &past_shape,
            &past_strides,
            device,
        ),
    ];
    let mut outputs = [
        TensorMut::new(
            DevicePtrMut(y.as_mut_ptr()),
            DataType::Float32,
            &q_shape,
            &q_strides,
            device,
        ),
        TensorMut::new(
            DevicePtrMut(key_cache.as_mut_ptr()),
            DataType::Float32,
            &present_shape,
            &present_strides,
            device,
        ),
        TensorMut::new(
            DevicePtrMut(value_cache.as_mut_ptr()),
            DataType::Float32,
            &present_shape,
            &present_strides,
            device,
        ),
    ];
    kernel.execute(&inputs, &mut outputs)
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
fn default_attention_aliased_dense_kv_growth_captures_and_matches_eager() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let kernel = standard_attention_kernel(&ep);
    let max_seq = INITIAL_PAST + DECODE_STEPS + 1;
    let cache_bytes = HEADS * max_seq * HEAD_DIM * std::mem::size_of::<f32>();
    let token_bytes = HEADS * HEAD_DIM * std::mem::size_of::<f32>();

    let initial_key = (0..HEADS * INITIAL_PAST * HEAD_DIM)
        .map(|index| index as f32 * 0.03125 - 0.25)
        .collect::<Vec<_>>();
    let initial_value = (0..HEADS * INITIAL_PAST * HEAD_DIM)
        .map(|index| index as f32 * -0.046875 + 0.5)
        .collect::<Vec<_>>();

    let q = ep.allocate(token_bytes, 256).expect("allocate query");
    let k = ep.allocate(token_bytes, 256).expect("allocate key");
    let v = ep.allocate(token_bytes, 256).expect("allocate value");
    let mut captured_key = ep
        .allocate(cache_bytes, 256)
        .expect("allocate captured key cache");
    let mut captured_value = ep
        .allocate(cache_bytes, 256)
        .expect("allocate captured value cache");
    let mut eager_key = ep
        .allocate(cache_bytes, 256)
        .expect("allocate eager key cache");
    let mut eager_value = ep
        .allocate(cache_bytes, 256)
        .expect("allocate eager value cache");
    let mut captured_y = ep
        .allocate(token_bytes, 256)
        .expect("allocate captured output");
    let mut eager_y = ep
        .allocate(token_bytes, 256)
        .expect("allocate eager output");

    // SAFETY: each cache allocation is at least as large as its initial dense
    // prefix. The output present cache aliases the corresponding past input.
    unsafe {
        runtime
            .htod(&f32_bytes(&initial_key), cuptr(captured_key.as_ptr()))
            .unwrap();
        runtime
            .htod(&f32_bytes(&initial_value), cuptr(captured_value.as_ptr()))
            .unwrap();
        runtime
            .htod(&f32_bytes(&initial_key), cuptr(eager_key.as_ptr()))
            .unwrap();
        runtime
            .htod(&f32_bytes(&initial_value), cuptr(eager_value.as_ptr()))
            .unwrap();
    }

    for step in 0..DECODE_STEPS {
        let past_seq = INITIAL_PAST + step;
        let token = |offset: f32| {
            (0..HEADS * HEAD_DIM)
                .map(|index| offset + step as f32 * 0.125 + index as f32 * 0.0078125)
                .collect::<Vec<_>>()
        };
        // SAFETY: the three fixed token buffers exactly cover these tensors.
        unsafe {
            runtime
                .htod(&f32_bytes(&token(0.25)), cuptr(q.as_ptr()))
                .unwrap();
            runtime
                .htod(&f32_bytes(&token(-0.5)), cuptr(k.as_ptr()))
                .unwrap();
            runtime
                .htod(&f32_bytes(&token(0.75)), cuptr(v.as_ptr()))
                .unwrap();
        }

        // This eager step both supplies the oracle and warms the exact staged
        // workspace shape before recording the same aliased decode under capture.
        execute_decode(
            kernel.as_ref(),
            &q,
            &k,
            &v,
            &mut eager_key,
            &mut eager_value,
            &mut eager_y,
            past_seq,
        )
        .unwrap();
        assert!(
            kernel.cuda_graph_compatible(),
            "eager staged decode must warm CUDA-graph capture support"
        );
        if step == 0 {
            let warmed_resources = kernel
                .device_graph_resources()
                .iter()
                .map(|resource| resource.identity())
                .collect::<Vec<_>>();
            assert!(
                !warmed_resources.is_empty(),
                "capture-eligible Attention must publish its private workspace owners"
            );
            let counts_before_failure = runtime.allocation_counts();
            let pooled_before_failure = runtime.raw_pool_retained_bytes();
            runtime.fail_warm_transaction_at_for_test(1);
            let failure = execute_decode(
                kernel.as_ref(),
                &q,
                &k,
                &v,
                &mut eager_key,
                &mut eager_value,
                &mut eager_y,
                past_seq + 1,
            )
            .unwrap_err()
            .to_string();
            assert!(
                failure.contains("injected staged warm-cache failure after Attention workspace"),
                "{failure}"
            );
            assert!(
                kernel.cuda_graph_compatible(),
                "a failed replacement call must preserve the successful Attention warm"
            );
            assert_eq!(
                kernel
                    .device_graph_resources()
                    .iter()
                    .map(|resource| resource.identity())
                    .collect::<Vec<_>>(),
                warmed_resources,
                "Attention capture eligibility and resources must remain one successful snapshot"
            );
            let counts_after_failure = runtime.allocation_counts();
            assert!(
                counts_after_failure.frees > counts_before_failure.frees
                    || runtime.raw_pool_retained_bytes() > pooled_before_failure,
                "the rejected Attention workspace candidate must return each allocation exactly once"
            );
        }

        let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
        runtime
            .begin_graph_capture(&kernels)
            .expect("begin default Attention CUDA graph capture");
        execute_decode(
            kernel.as_ref(),
            &q,
            &k,
            &v,
            &mut captured_key,
            &mut captured_value,
            &mut captured_y,
            past_seq,
        )
        .unwrap();
        runtime
            .end_graph_capture()
            .expect("aliased default Attention staged copy-back must capture");
        let present_bytes = HEADS * (past_seq + 1) * HEAD_DIM * std::mem::size_of::<f32>();
        let eager_y_expected = read(&ep, &eager_y, token_bytes);
        let eager_key_expected = read(&ep, &eager_key, present_bytes);
        let eager_value_expected = read(&ep, &eager_value, present_bytes);
        if step + 1 == DECODE_STEPS {
            execute_decode(
                kernel.as_ref(),
                &q,
                &k,
                &v,
                &mut eager_key,
                &mut eager_value,
                &mut eager_y,
                past_seq + 1,
            )
            .unwrap();
        }
        runtime
            .replay_graph()
            .expect("replay captured default Attention decode");

        assert_eq!(
            read(&ep, &captured_y, token_bytes),
            eager_y_expected,
            "captured Attention output diverged at decode step {step}"
        );
        assert_eq!(
            read(&ep, &captured_key, present_bytes),
            eager_key_expected,
            "captured aliased present key diverged at decode step {step}"
        );
        assert_eq!(
            read(&ep, &captured_value, present_bytes),
            eager_value_expected,
            "captured aliased present value diverged at decode step {step}"
        );
        assert_eq!(
            runtime.check_capture_error().unwrap(),
            0,
            "capture error latched at decode step {step}"
        );
        assert!(
            runtime.reset_graph().unwrap(),
            "captured graph was not installed"
        );
    }

    for buffer in [
        eager_y,
        captured_y,
        eager_value,
        eager_key,
        captured_value,
        captured_key,
        v,
        k,
        q,
    ] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}
