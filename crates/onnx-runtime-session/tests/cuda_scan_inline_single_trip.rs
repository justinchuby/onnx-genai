//! CUDA-gated regression for slice 1a: the flag-gated single-trip `Scan` inline
//! dual-path must engage on device at `trip_count == 1`, stay byte-exact with
//! the generic `exec_scan` loop, and never inline a prefill-shaped
//! (`trip_count > 1`) run — the shared-plan tripwire. This lives in its own test
//! binary so no sibling test races the process-global
//! `ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP` env flag it toggles at session build.
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

/// Build a CUDA session with the inline flag forced to `inline` for the duration
/// of the build (it is read once at build), run the feeds, and return the raw
/// output bytes plus the inline-engagement counter.
fn run_cuda(bytes: &[u8], feeds: &[(&str, &Tensor)], inline: bool) -> (Vec<Vec<u8>>, u64) {
    let _guard = env_lock().lock().expect("env lock");
    let prev = std::env::var_os("ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP");
    // SAFETY: all mutations of this var in this binary are serialized by env_lock.
    unsafe {
        std::env::set_var(
            "ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP",
            if inline { "1" } else { "0" },
        );
    }
    let mut session = InferenceSession::builder()
        .model_bytes(bytes)
        .device(DevicePreference::Gpu { index: Some(0) })
        .build()
        .expect("build CUDA session");
    // SAFETY: serialized by env_lock; restore the prior value now that the flag
    // has been consumed at build.
    unsafe {
        match prev {
            Some(v) => std::env::set_var("ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP", v),
            None => std::env::remove_var("ONNX_GENAI_SCAN_INLINE_SINGLE_TRIP"),
        }
    }
    let outputs = session.run(feeds).expect("run CUDA session");
    let out_bytes = outputs.iter().map(|t| t.as_bytes().to_vec()).collect();
    (out_bytes, session.scan_inline_single_trip_count())
}

#[test]
fn cuda_scan_single_trip_inline_is_byte_exact_and_runtime_keyed() {
    // Decode regime: trip_count == 1. Flag ON engages the inline path exactly
    // once and is byte-identical to the flag-OFF loop over both outputs.
    let decode = scan_model(1);
    let x1 = Tensor::from_f32(&[1, W], &[1.0, 2.0, 3.0]).unwrap();
    let y1 = Tensor::from_f32(&[1, W], &[2.0, 2.5, 3.0]).unwrap();
    let feeds1 = [("X", &x1), ("Y", &y1)];

    let (loop_out, loop_count) = run_cuda(&decode, &feeds1, false);
    let (inline_out, inline_count) = run_cuda(&decode, &feeds1, true);
    assert_eq!(loop_count, 0, "flag OFF must never engage the inline path");
    assert_eq!(
        inline_count, 1,
        "flag ON at trip_count==1 must engage the inline path exactly once on CUDA"
    );
    assert_eq!(
        inline_out, loop_out,
        "single-trip inline output must be byte-exact with the loop path on CUDA"
    );

    // Prefill regime: trip_count == 3. Even with the flag ON the inline path must
    // NOT engage (runtime-keyed, not a static rewrite), and output must match.
    let prefill = scan_model(3);
    let x3 = Tensor::from_f32(&[3, W], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
    let y3 = Tensor::from_f32(&[3, W], &[2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]).unwrap();
    let feeds3 = [("X", &x3), ("Y", &y3)];

    let (prefill_loop, prefill_loop_count) = run_cuda(&prefill, &feeds3, false);
    let (prefill_inline, prefill_inline_count) = run_cuda(&prefill, &feeds3, true);
    assert_eq!(prefill_loop_count, 0, "loop path never counts");
    assert_eq!(
        prefill_inline_count, 0,
        "flag ON must NOT inline a prefill (trip_count>1) Scan on CUDA — shared-plan tripwire"
    );
    assert_eq!(
        prefill_inline, prefill_loop,
        "prefill output must be identical flag-on vs flag-off on CUDA"
    );
}
