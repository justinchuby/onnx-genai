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
//! GPU regression coverage for `pkg.nxrt::KvCacheCapacityAppend` — the
//! CUDA-graph-capture-safe replacement for a decomposed-attention KV-cache
//! growth `Concat` (the "S3 capacity emission" gate).
//!
//! Unlike `ConcatKernel`, this op's launch geometry never depends on the
//! *logical* (growing) sequence length: `past`/`present` are always exposed
//! at their fixed `[batch, heads, capacity, head_dim]` physical shape, and the
//! only thing that varies decode-to-decode — the destination row — is read
//! from `position_ids`' *device memory contents* at execute time rather than
//! baked into host launch parameters. The defining property under test here
//! is therefore: **one CUDA graph capture, replayed many times with
//! different `position_ids`/`current` device content, must write each new
//! row correctly while leaving every other row untouched** — the exact
//! capability a captured `Concat` cannot offer.

use onnx_runtime_ep_api::{
    DeviceBuffer, DevicePtr, DevicePtrMut, ExecutionProvider, Kernel, TensorMut, TensorView,
};
use onnx_runtime_ep_cuda::runtime::cuptr;
use onnx_runtime_ep_cuda::{CudaExecutionProvider, KV_CAPACITY_APPEND_CAPTURE_ERROR_POSITION};
use onnx_runtime_ir::{Graph, Node, NodeId, compute_contiguous_strides, static_shape};
use onnx_runtime_loader::Model;

const DOMAIN: &str = "pkg.nxrt";
const OP: &str = "KvCacheCapacityAppend";

const BATCH: usize = 1;
const HEADS: usize = 2;
const HEAD_DIM: usize = 4;
const CAPACITY: usize = 4;

const ROW_ELEMS: usize = HEADS * HEAD_DIM;
const TOTAL_ELEMS: usize = BATCH * HEADS * CAPACITY * HEAD_DIM;
const SENTINEL: f32 = -999.0;

fn bytes<T: Copy>(values: &[T]) -> Vec<u8> {
    // SAFETY: test inputs are fixed-width plain data.
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast(), std::mem::size_of_val(values)).to_vec()
    }
}

fn floats(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn require_cuda() -> CudaExecutionProvider {
    CudaExecutionProvider::new_default().expect("CUDA runtime must be available for this GPU test")
}

fn upload(ep: &CudaExecutionProvider, bytes: &[u8]) -> DeviceBuffer {
    let buffer = ep.allocate(bytes.len(), 256).unwrap();
    // SAFETY: `buffer` was just allocated with exactly `bytes.len()` capacity.
    unsafe { ep.runtime().htod(bytes, cuptr(buffer.as_ptr())).unwrap() };
    buffer
}

fn overwrite(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, bytes: &[u8]) {
    // SAFETY: caller-provided buffer is guaranteed by every call site below to
    // have been allocated with at least `bytes.len()` capacity.
    unsafe {
        ep.runtime()
            .htod(bytes, cuptr(buffer.as_ptr()))
            .expect("overwrite CUDA test buffer")
    };
}

fn download(ep: &CudaExecutionProvider, buffer: &DeviceBuffer, len: usize) -> Vec<u8> {
    let mut host = vec![0_u8; len];
    // SAFETY: `buffer` owns at least `len` bytes in every caller below.
    unsafe {
        ep.runtime()
            .dtoh(&mut host, cuptr(buffer.as_ptr()))
            .expect("copy CUDA output to host")
    };
    host
}

/// Builds the single-node `KvCacheCapacityAppend` graph and returns its
/// kernel, mirroring how the executor's rewrite pass emits this op in place
/// of a KV-growth `Concat` (see `onnx-runtime-session::executor::geometry`'s
/// `kv_capacity_write_eligible_concats`).
fn kv_capacity_append_kernel(ep: &CudaExecutionProvider) -> Box<dyn Kernel> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 23);
    graph.opset_imports.insert(DOMAIN.into(), 1);
    let past = graph.create_named_value(
        "past_key",
        onnx_runtime_ir::DataType::Float32,
        static_shape([BATCH, HEADS, CAPACITY, HEAD_DIM]),
    );
    let current = graph.create_named_value(
        "current_key",
        onnx_runtime_ir::DataType::Float32,
        static_shape([BATCH, HEADS, 1, HEAD_DIM]),
    );
    let position_ids = graph.create_named_value(
        "position_ids",
        onnx_runtime_ir::DataType::Int64,
        static_shape([BATCH, 1]),
    );
    for input in [past, current, position_ids] {
        graph.add_input(input);
    }
    let present = graph.create_named_value(
        "present_key",
        onnx_runtime_ir::DataType::Float32,
        static_shape([BATCH, HEADS, CAPACITY, HEAD_DIM]),
    );
    graph.add_output(present);
    let mut node = Node::new(
        NodeId(0),
        OP,
        vec![Some(past), Some(current), Some(position_ids)],
        vec![present],
    );
    node.domain = DOMAIN.into();
    let node_id = graph.insert_node(node);
    let model = Model::new(&graph);
    let shapes = vec![
        vec![BATCH, HEADS, CAPACITY, HEAD_DIM],
        vec![BATCH, HEADS, 1, HEAD_DIM],
        vec![BATCH, 1],
    ];
    ep.get_kernel(model.graph.node(node_id), &shapes, 1)
        .expect("pkg.nxrt::KvCacheCapacityAppend must be supported on CUDA")
}

/// Executes one step, returning the kernel's raw `Result` instead of
/// panicking on failure — used directly by the eager-bounds-rejection test
/// below; every other call site uses [`execute_step`], which wraps this with
/// `.expect(...)` for the common "must succeed" case.
fn try_execute_step(
    kernel: &dyn Kernel,
    past_present: &DeviceBuffer,
    current: &DeviceBuffer,
    position_ids: &DeviceBuffer,
) -> onnx_runtime_ep_api::Result<()> {
    let past_shape = [BATCH, HEADS, CAPACITY, HEAD_DIM];
    let current_shape = [BATCH, HEADS, 1, HEAD_DIM];
    let position_shape = [BATCH, 1];
    let past_strides = compute_contiguous_strides(&past_shape);
    let current_strides = compute_contiguous_strides(&current_shape);
    let position_strides = compute_contiguous_strides(&position_shape);
    let inputs = [
        TensorView::new(
            DevicePtr(past_present.as_ptr()),
            onnx_runtime_ir::DataType::Float32,
            &past_shape,
            &past_strides,
            past_present.device(),
        ),
        TensorView::new(
            DevicePtr(current.as_ptr()),
            onnx_runtime_ir::DataType::Float32,
            &current_shape,
            &current_strides,
            current.device(),
        ),
        TensorView::new(
            DevicePtr(position_ids.as_ptr()),
            onnx_runtime_ir::DataType::Int64,
            &position_shape,
            &position_strides,
            position_ids.device(),
        ),
    ];
    let mut outputs = [TensorMut::new(
        DevicePtrMut(past_present.as_ptr() as *mut std::ffi::c_void),
        onnx_runtime_ir::DataType::Float32,
        &past_shape,
        &past_strides,
        past_present.device(),
    )];
    kernel.execute(&inputs, &mut outputs)
}

/// Executes one step: `past_present` is read as `past` *and* written as
/// `present` through the same device pointer, exactly as the executor's
/// persistent present==past KV binding aliases them in production.
fn execute_step(
    kernel: &dyn Kernel,
    past_present: &DeviceBuffer,
    current: &DeviceBuffer,
    position_ids: &DeviceBuffer,
) {
    try_execute_step(kernel, past_present, current, position_ids)
        .expect("KvCacheCapacityAppend decode step");
}

fn current_row_values(step: usize) -> Vec<f32> {
    (0..HEADS)
        .flat_map(|head| {
            (0..HEAD_DIM).map(move |d| step as f32 * 100.0 + head as f32 * 10.0 + d as f32)
        })
        .collect()
}

fn expect_row(expected: &mut [f32], row: usize, values: &[f32]) {
    for head in 0..HEADS {
        for d in 0..HEAD_DIM {
            let idx = (head * CAPACITY + row) * HEAD_DIM + d;
            expected[idx] = values[head * HEAD_DIM + d];
        }
    }
}

#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn kv_capacity_append_fills_capacity_row_by_row_without_capture() {
    let ep = require_cuda();
    let kernel = kv_capacity_append_kernel(&ep);

    let past_present = upload(&ep, &bytes(&[SENTINEL; TOTAL_ELEMS]));
    let current = ep.allocate(ROW_ELEMS * 4, 256).unwrap();
    let position_ids = ep.allocate(8, 256).unwrap();

    let mut expected = vec![SENTINEL; TOTAL_ELEMS];
    for step in 0..CAPACITY {
        let row_values = current_row_values(step);
        overwrite(&ep, &current, &bytes(&row_values));
        overwrite(&ep, &position_ids, &bytes(&[step as i64]));

        execute_step(kernel.as_ref(), &past_present, &current, &position_ids);

        expect_row(&mut expected, step, &row_values);
        let observed = floats(&download(&ep, &past_present, TOTAL_ELEMS * 4));
        assert_eq!(
            observed, expected,
            "row {step} write disturbed an unrelated row (eager, uncaptured)"
        );
        assert_eq!(
            ep.runtime().check_capture_error().unwrap(),
            0,
            "no bounds violation is expected while filling within capacity"
        );
    }

    for buffer in [position_ids, current, past_present] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}

/// The money test: capture the kernel launch **once**, then replay it many
/// times with fresh `current`/`position_ids` device content between replays
/// (no re-capture). This is exactly the steady-state decode loop the S3
/// capacity-emission mechanism exists to make possible, and is precisely
/// what a captured plain `Concat` cannot do (its launch geometry is derived
/// from the logical length at record time and cannot vary per replay).
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn kv_capacity_append_captures_once_and_replays_correctly_across_growing_logical_length() {
    let ep = require_cuda();
    let runtime = ep.runtime();
    let kernel = kv_capacity_append_kernel(&ep);

    let past_present = upload(&ep, &bytes(&[SENTINEL; TOTAL_ELEMS]));
    let current = ep.allocate(ROW_ELEMS * 4, 256).unwrap();
    let position_ids = ep.allocate(8, 256).unwrap();

    let mut expected = vec![SENTINEL; TOTAL_ELEMS];

    // Warm the exact signature with step 0, then capture that exact launch.
    let row0 = current_row_values(0);
    overwrite(&ep, &current, &bytes(&row0));
    overwrite(&ep, &position_ids, &bytes(&[0_i64]));
    execute_step(kernel.as_ref(), &past_present, &current, &position_ids);
    expect_row(&mut expected, 0, &row0);
    assert!(
        kernel.cuda_graph_compatible(),
        "warmed KvCacheCapacityAppend must be capture-supported"
    );

    let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
    runtime
        .begin_graph_capture(&kernels)
        .expect("begin KvCacheCapacityAppend CUDA graph capture");
    execute_step(kernel.as_ref(), &past_present, &current, &position_ids);
    runtime
        .end_graph_capture()
        .expect("KvCacheCapacityAppend must record without host fallback");
    assert!(
        runtime.has_graph_executable().unwrap(),
        "KvCacheCapacityAppend decode did not install a CUDA graph"
    );

    // Replay the SAME captured graph for every remaining logical position,
    // mutating only device-side content (never re-capturing, never changing
    // shapes) between replays.
    for step in 1..CAPACITY {
        let row = current_row_values(step);
        overwrite(&ep, &current, &bytes(&row));
        overwrite(&ep, &position_ids, &bytes(&[step as i64]));

        runtime
            .replay_graph()
            .expect("replay captured KvCacheCapacityAppend decode");

        expect_row(&mut expected, step, &row);
        let observed = floats(&download(&ep, &past_present, TOTAL_ELEMS * 4));
        assert_eq!(
            observed, expected,
            "replayed step {step} diverged from the row-by-row expectation \
             (either wrote the wrong row or disturbed an earlier one)"
        );
        assert_eq!(
            runtime.check_capture_error().unwrap(),
            0,
            "in-capacity replay must never latch the capture-error word (step {step})"
        );
        assert!(
            runtime.has_graph_executable().unwrap(),
            "the graph executable must remain installed across replays (step {step}); a \
             recapture would defeat the point of this test"
        );
    }

    assert!(
        runtime.reset_graph().unwrap(),
        "captured KvCacheCapacityAppend graph was not installed"
    );
    for buffer in [position_ids, current, past_present] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}

/// A `position_ids` value at or beyond `capacity` (or negative) hit **during
/// CUDA graph capture/replay** must latch the capture-error word and leave
/// every row's memory untouched — never write out of bounds and never
/// corrupt an unrelated row. The eager (non-capturing) counterpart of this
/// property is covered separately by
/// [`kv_capacity_append_eager_execute_rejects_out_of_capacity_position_without_writing`],
/// since eager execution uses a completely different mechanism (a
/// synchronous host-side bounds check that hard-errors before the kernel
/// ever launches, rather than a device-side latch polled after the fact) —
/// see the module doc comment on `kv_cache_capacity_append.rs` for why the
/// two paths cannot share one check.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn kv_capacity_append_latches_capture_error_on_out_of_capacity_position_without_corrupting_memory()
{
    let ep = require_cuda();
    let runtime = ep.runtime();
    let kernel = kv_capacity_append_kernel(&ep);

    let past_present = upload(&ep, &bytes(&[SENTINEL; TOTAL_ELEMS]));
    let current = ep.allocate(ROW_ELEMS * 4, 256).unwrap();
    let position_ids = ep.allocate(8, 256).unwrap();

    // Warm + capture at the last in-capacity row.
    let last_row = CAPACITY - 1;
    let row_values = current_row_values(last_row);
    overwrite(&ep, &current, &bytes(&row_values));
    overwrite(&ep, &position_ids, &bytes(&[last_row as i64]));
    execute_step(kernel.as_ref(), &past_present, &current, &position_ids);
    let mut expected = vec![SENTINEL; TOTAL_ELEMS];
    expect_row(&mut expected, last_row, &row_values);

    let kernels: [&dyn Kernel; 1] = [kernel.as_ref()];
    runtime.begin_graph_capture(&kernels).unwrap();
    execute_step(kernel.as_ref(), &past_present, &current, &position_ids);
    runtime.end_graph_capture().unwrap();
    assert_eq!(runtime.check_capture_error().unwrap(), 0);

    // Replay #1: still in-capacity (a fresh value at the same row) — proves
    // ordinary in-bounds replay stays clean before we probe the boundary.
    let refreshed = current_row_values(last_row + 10);
    overwrite(&ep, &current, &bytes(&refreshed));
    runtime.replay_graph().unwrap();
    expect_row(&mut expected, last_row, &refreshed);
    assert_eq!(
        floats(&download(&ep, &past_present, TOTAL_ELEMS * 4)),
        expected
    );
    assert_eq!(runtime.check_capture_error().unwrap(), 0);

    // Replay #2: position == capacity (first invalid, exclusive upper bound).
    let poison = current_row_values(999);
    overwrite(&ep, &current, &bytes(&poison));
    overwrite(&ep, &position_ids, &bytes(&[CAPACITY as i64]));
    runtime.replay_graph().unwrap();
    assert_eq!(
        runtime.check_capture_error().unwrap() & KV_CAPACITY_APPEND_CAPTURE_ERROR_POSITION,
        KV_CAPACITY_APPEND_CAPTURE_ERROR_POSITION,
        "position_ids == capacity must latch the capacity-overflow bit"
    );
    assert_eq!(
        floats(&download(&ep, &past_present, TOTAL_ELEMS * 4)),
        expected,
        "an out-of-capacity replay must not have written anywhere"
    );
    runtime.reset_capture_error().unwrap();
    assert_eq!(runtime.check_capture_error().unwrap(), 0);

    // Replay #3: negative position — the other half of the bounds check.
    overwrite(&ep, &position_ids, &bytes(&[-1_i64]));
    runtime.replay_graph().unwrap();
    assert_eq!(
        runtime.check_capture_error().unwrap() & KV_CAPACITY_APPEND_CAPTURE_ERROR_POSITION,
        KV_CAPACITY_APPEND_CAPTURE_ERROR_POSITION,
        "negative position_ids must also latch the capacity-overflow bit"
    );
    assert_eq!(
        floats(&download(&ep, &past_present, TOTAL_ELEMS * 4)),
        expected,
        "a negative-position replay must not have written anywhere"
    );
    runtime.reset_capture_error().unwrap();

    // Replay #4: back to a valid row proves the latch reset lets the *same*
    // captured graph keep serving correct, clean replays afterward.
    let recovered = current_row_values(last_row + 20);
    overwrite(&ep, &current, &bytes(&recovered));
    overwrite(&ep, &position_ids, &bytes(&[last_row as i64]));
    runtime.replay_graph().unwrap();
    expect_row(&mut expected, last_row, &recovered);
    assert_eq!(
        floats(&download(&ep, &past_present, TOTAL_ELEMS * 4)),
        expected
    );
    assert_eq!(runtime.check_capture_error().unwrap(), 0);

    assert!(runtime.reset_graph().unwrap());
    for buffer in [position_ids, current, past_present] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}

/// The eager (non-capturing) counterpart to the capture-time boundary test
/// above: an out-of-capacity `position_ids` value hit through a genuine
/// `Kernel::execute()` call with no capture in progress must be rejected by
/// a synchronous host-side bounds check with a hard `Err` *before* the
/// kernel ever launches — not silently skipped with `Ok(())` returned and no
/// observable signal (the gap this test exists to close), and not by
/// relying on the device-side capture-error latch, which is a capture-mode-
/// only mechanism.
///
/// That synchronous check is scoped (exactly like `rotary_embedding.rs`'s
/// equivalent guard) to `!capturing && !eager_sync_deferred()`, and
/// `eager_sync_deferred()` defaults to `true` — under the default runtime
/// configuration, ordinary eager calls also latch the device-side error word
/// and rely on the caller's next synchronize/`check_capture_error` poll,
/// exactly as the capturing case does. This test exercises the narrower,
/// explicit-opt-out configuration (`set_defer_eager_sync(false)`, the same
/// "escape hatch for debugging" toggle documented on
/// `CudaRuntime::eager_sync_deferred`) in which the synchronous host check is
/// the only mechanism available, so a silent `Ok(())` would otherwise be
/// truly unobservable until some later, unrelated synchronize call. Checks
/// both the negative and `== capacity` boundary values, and confirms memory
/// is left completely untouched by the rejected call.
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
#[test]
fn kv_capacity_append_eager_execute_rejects_out_of_capacity_position_without_writing() {
    let ep = require_cuda();
    let kernel = kv_capacity_append_kernel(&ep);
    // Force the fully-synchronous eager path so the synchronous host bounds
    // check actually runs instead of being skipped in favor of the deferred
    // device-latch mechanism (which is exercised by the capture-mode
    // counterpart test above and is on by default).
    ep.runtime().set_defer_eager_sync(false);

    let past_present = upload(&ep, &bytes(&[SENTINEL; TOTAL_ELEMS]));
    let current = ep.allocate(ROW_ELEMS * 4, 256).unwrap();
    let position_ids = ep.allocate(8, 256).unwrap();
    let expected = vec![SENTINEL; TOTAL_ELEMS];

    for bad_position in [CAPACITY as i64, -1_i64] {
        let poison = current_row_values(999);
        overwrite(&ep, &current, &bytes(&poison));
        overwrite(&ep, &position_ids, &bytes(&[bad_position]));

        let result = try_execute_step(kernel.as_ref(), &past_present, &current, &position_ids);
        assert!(
            result.is_err(),
            "an eager call with position_ids={bad_position} (capacity={CAPACITY}) must hard-error, \
             not silently succeed"
        );
        assert_eq!(
            floats(&download(&ep, &past_present, TOTAL_ELEMS * 4)),
            expected,
            "a rejected eager call must not have written anywhere for position_ids={bad_position}"
        );
        assert_eq!(
            ep.runtime().check_capture_error().unwrap(),
            0,
            "the eager host-side rejection path must not touch the device capture-error word \
             (that latch is exercised only by the capture-mode counterpart test)"
        );
    }
    ep.runtime().set_defer_eager_sync(true);

    // A subsequent in-bounds eager call on the same kernel/buffers must still
    // succeed cleanly — the rejected calls above must leave no lingering
    // state (e.g. a stale warmed signature or a poisoned lock) behind.
    let good_row = current_row_values(0);
    overwrite(&ep, &current, &bytes(&good_row));
    overwrite(&ep, &position_ids, &bytes(&[0_i64]));
    execute_step(kernel.as_ref(), &past_present, &current, &position_ids);
    let mut recovered_expected = expected;
    expect_row(&mut recovered_expected, 0, &good_row);
    assert_eq!(
        floats(&download(&ep, &past_present, TOTAL_ELEMS * 4)),
        recovered_expected,
        "a valid eager call after rejected ones must still write correctly"
    );

    for buffer in [position_ids, current, past_present] {
        ep.deallocate(buffer).expect("free CUDA test buffer");
    }
}
