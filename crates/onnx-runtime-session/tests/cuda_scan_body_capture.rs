//! CUDA-gated regression for slice 1b: the flag-gated nested `Scan`-body
//! device-graph capture path.
//!
//! Slice 1b hit a STOP obstacle: the device-graph facility is a per-EP singleton
//! segment list shared by every child body (all built from one cloned
//! `Arc<dyn ExecutionProvider>`), so a per-node captured graph cannot be isolated
//! and whole-graph replay silently corrupts output. The capture install/replay is
//! therefore gated behind the explicit unsafe spike flag
//! (`ONNX_GENAI_SCAN_BODY_CAPTURE_UNSAFE`); the production flag
//! (`ONNX_GENAI_SCAN_BODY_CAPTURE`) keeps the single-trip body eager and
//! byte-exact.
//!
//! This test is the non-vacuous guard for that decision:
//!  * flag-ON body-capture output MUST be byte-exact with the flag-OFF loop path
//!    AND with the inline-only (1a) path — so re-enabling a corrupting capture
//!    would make it FAIL;
//!  * the body-capture path MUST actually engage on the single-trip decode
//!    (`scan_body_fallbacks > 0`) — proving it is exercised, not skipped;
//!  * it MUST install ZERO device graphs (`scan_body_captures == 0`), the
//!    correctness invariant of the guard; and
//!  * prefill (`trip_count > 1`) MUST never reach the body-capture path at all.
//!
//! Lives in its own binary so no sibling test races the process-global env flags
//! it toggles at session build.
#![cfg(feature = "cuda")]

use std::sync::{Mutex, OnceLock};

use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_loader::{Model, encode_model};
use onnx_runtime_session::{DevicePreference, InferenceSession, Tensor};

const W: usize = 3;

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn init(graph: &mut Graph, name: &str, dims: &[usize], data: &[f32]) -> ValueId {
    let value =
        graph.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    graph.set_initializer(
        value,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            f32_bytes(data),
        )),
    );
    value
}

fn body_op(body: &mut Graph, op_type: &str, inputs: &[ValueId], name: &str) -> ValueId {
    let out = body.create_named_value(name, DataType::Float32, static_shape([W]));
    body.insert_node(Node::new(
        NodeId(0),
        op_type,
        inputs.iter().copied().map(Some).collect(),
        vec![out],
    ));
    out
}

/// A multi-node `Scan` body threading carried state through two scan inputs:
///   `state_x   = Add(state, x)`
///   `state_out = Mul(state_x, y)`   (next carried state)
///   `scan_out  = Sub(state_out, x)` (stacked on the scan axis)
fn scan_body() -> Graph {
    let mut body = Graph::new();
    body.opset_imports.insert(String::new(), 17);
    let state = body.create_named_value("state", DataType::Float32, static_shape([W]));
    let x = body.create_named_value("x", DataType::Float32, static_shape([W]));
    let y = body.create_named_value("y", DataType::Float32, static_shape([W]));
    body.add_input(state);
    body.add_input(x);
    body.add_input(y);
    let state_x = body_op(&mut body, "Add", &[state, x], "state_x");
    let state_out = body_op(&mut body, "Mul", &[state_x, y], "state_out");
    let scan_out = body_op(&mut body, "Sub", &[state_out, x], "scan_out");
    body.add_output(state_out);
    body.add_output(scan_out);
    body
}

fn scan_model(steps: usize) -> Vec<u8> {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let initial = init(&mut graph, "initial", &[W], &[0.0; W]);
    let x_in = graph.create_named_value("X", DataType::Float32, static_shape([steps, W]));
    let y_in = graph.create_named_value("Y", DataType::Float32, static_shape([steps, W]));
    graph.add_input(x_in);
    graph.add_input(y_in);
    let final_state = graph.create_named_value("final_state", DataType::Float32, static_shape([W]));
    let scan_output =
        graph.create_named_value("scan_output", DataType::Float32, static_shape([steps, W]));
    let body = scan_body();
    let mut scan = Node::new(
        NodeId(0),
        "Scan",
        vec![Some(initial), Some(x_in), Some(y_in)],
        vec![final_state, scan_output],
    );
    scan.attributes
        .insert("num_scan_inputs".into(), Attribute::Int(2));
    scan.attributes
        .insert("body".into(), Attribute::Graph(Box::new(body.clone())));
    let scan_id = graph.insert_node(scan);
    graph.subgraphs.insert((scan_id, "body".into()), body);
    graph.add_output(final_state);
    graph.add_output(scan_output);
    encode_model(&Model::new(&graph)).expect("encode scan model")
}

/// Serializes the env-flag mutation so the CUDA runtime's own session builds
/// (this binary only) never observe a torn flag value.
fn env_lock() -> &'static Mutex<()> {
    static ENV_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENV_LOCK.get_or_init(|| Mutex::new(()))
}

/// Outcome of one CUDA run: raw output bytes plus the observable counters.
struct RunOutcome {
    outputs: Vec<Vec<u8>>,
    inline_count: u64,
    body_captures: u64,
    body_replays: u64,
    body_fallbacks: u64,
}

/// Build a CUDA session with the inline + body-capture flags forced for the
/// duration of the build (they are read once at build), run the feeds, and
/// return the raw outputs plus the slice-1a/1b counters.
fn run_cuda(
    bytes: &[u8],
    feeds: &[(&str, &Tensor)],
    inline: bool,
    body_capture: bool,
) -> RunOutcome {
    let _guard = env_lock().lock().expect("env lock");
    let prev_inline = std::env::var_os("ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP");
    let prev_capture = std::env::var_os("ONNX_GENAI_SCAN_BODY_CAPTURE");
    // SAFETY: all mutations of these vars in this binary are serialized by env_lock.
    unsafe {
        std::env::set_var(
            "ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP",
            if inline { "1" } else { "0" },
        );
        std::env::set_var(
            "ONNX_GENAI_SCAN_BODY_CAPTURE",
            if body_capture { "1" } else { "0" },
        );
    }
    let mut session = InferenceSession::builder()
        .model_bytes(bytes)
        .device(DevicePreference::Gpu { index: Some(0) })
        .build()
        .expect("build CUDA session");
    // SAFETY: serialized by env_lock; restore prior values now the flags are
    // consumed at build.
    unsafe {
        match prev_inline {
            Some(v) => std::env::set_var("ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP", v),
            None => std::env::remove_var("ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP"),
        }
        match prev_capture {
            Some(v) => std::env::set_var("ONNX_GENAI_SCAN_BODY_CAPTURE", v),
            None => std::env::remove_var("ONNX_GENAI_SCAN_BODY_CAPTURE"),
        }
    }
    // Run twice on the SAME session: the child body plan (and its capture slot)
    // is cached across `run` calls, so the first invocation performs one-time
    // host discovery and the second exercises the steady-state body-capture path
    // — mirroring decode, where the body runs once per step over many steps.
    let _warm = session.run(feeds).expect("run CUDA session (warm)");
    let outputs = session.run(feeds).expect("run CUDA session (steady)");
    let out_bytes = outputs.iter().map(|t| t.as_bytes().to_vec()).collect();
    let stats = session.control_flow_stats();
    RunOutcome {
        outputs: out_bytes,
        inline_count: session.scan_inline_single_trip_count(),
        body_captures: stats.scan_body_captures,
        body_replays: stats.scan_body_replays,
        body_fallbacks: stats.scan_body_fallbacks,
    }
}

#[test]
fn cuda_scan_body_capture_flag_is_byte_exact_and_installs_no_shared_graph() {
    // Decode regime: trip_count == 1. Run three configurations over the same
    // multi-node body and single-step feeds:
    //   loop      — both flags OFF (generic exec_scan, the reference),
    //   inline    — inline ON, capture OFF (slice 1a),
    //   captureON — inline ON, capture ON  (slice 1b guarded path).
    let decode = scan_model(1);
    let x1 = Tensor::from_f32(&[1, W], &[1.0, 2.0, 3.0]).unwrap();
    let y1 = Tensor::from_f32(&[1, W], &[2.0, 2.5, 3.0]).unwrap();
    let feeds1 = [("X", &x1), ("Y", &y1)];

    let loop_run = run_cuda(&decode, &feeds1, false, false);
    let inline_run = run_cuda(&decode, &feeds1, true, false);
    let capture_run = run_cuda(&decode, &feeds1, true, true);

    assert_eq!(loop_run.inline_count, 0, "flag OFF must never inline");
    assert_eq!(
        capture_run.inline_count, 2,
        "flag ON at trip_count==1 must engage the inline path once per run (2 runs)"
    );

    // Byte-exactness: the guarded capture path must match BOTH the loop reference
    // and the inline-only path. If a corrupting capture were re-enabled on the
    // production flag, these would diverge and this test would FAIL.
    assert_eq!(
        capture_run.outputs, loop_run.outputs,
        "body-capture flag-ON output must be byte-exact with the exec_scan loop"
    );
    assert_eq!(
        capture_run.outputs, inline_run.outputs,
        "body-capture flag-ON output must be byte-exact with the 1a inline path"
    );

    // Non-vacuity: the body-capture path must actually be REACHED on the
    // single-trip decode (it declines to eager, counted as a fallback), and it
    // must install ZERO device graphs — the correctness invariant of the guard.
    assert!(
        capture_run.body_fallbacks > 0,
        "body-capture path must engage on single-trip decode (fallbacks>0), got {}",
        capture_run.body_fallbacks
    );
    assert_eq!(
        capture_run.body_captures, 0,
        "guarded body-capture must install no shared-EP device graph (captures==0)"
    );
    assert_eq!(
        capture_run.body_replays, 0,
        "guarded body-capture must never replay a shared-EP device graph (replays==0)"
    );

    // The 1a inline path (capture flag OFF) must not touch the body-capture
    // machinery at all.
    assert_eq!(
        inline_run.body_fallbacks, 0,
        "capture flag OFF must not reach the body-capture path"
    );
    assert_eq!(inline_run.body_captures, 0);

    // Prefill regime: trip_count == 3. Even with both flags ON the inline path
    // (and therefore the body-capture path) must NOT engage — runtime-keyed, not
    // a static rewrite.
    let prefill = scan_model(3);
    let x3 = Tensor::from_f32(&[3, W], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let y3 = Tensor::from_f32(&[3, W], &[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]).unwrap();
    let feeds3 = [("X", &x3), ("Y", &y3)];

    let prefill_loop = run_cuda(&prefill, &feeds3, false, false);
    let prefill_capture = run_cuda(&prefill, &feeds3, true, true);
    assert_eq!(
        prefill_capture.inline_count, 0,
        "flag ON must NOT inline a prefill (trip_count>1) Scan — shared-plan tripwire"
    );
    assert_eq!(
        prefill_capture.body_fallbacks, 0,
        "prefill must never reach the body-capture path"
    );
    assert_eq!(prefill_capture.body_captures, 0);
    assert_eq!(
        prefill_capture.outputs, prefill_loop.outputs,
        "prefill output must be identical both-flags-on vs both-flags-off"
    );
}
