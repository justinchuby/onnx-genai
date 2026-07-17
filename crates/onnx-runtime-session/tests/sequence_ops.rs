//! End-to-end integration tests for the ONNX **Sequence** ops, exercised
//! through the public [`InferenceSession`] surface. They cover:
//!
//! * `SequenceConstruct` → `SequenceAt` round-trips (element values preserved);
//! * `SequenceInsert` / `SequenceErase` positional semantics (incl. negatives);
//! * `SequenceLength` → int64 scalar;
//! * `SplitToSequence` (default per-slice, keepdims, and explicit `split`);
//! * `ConcatFromSequence` (existing axis and `new_axis=1` stacking);
//! * the **no-copy proof**: a tensor inserted into a sequence and read back with
//!   `SequenceAt`, then consumed by `Identity`, yields the exact same values —
//!   and the round-trip through the sequence never mutates data;
//! * actionable error paths (out-of-bounds index, empty-sequence concat).

use onnx_runtime_ir::{static_shape, Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef};
use onnx_runtime_session::InferenceSession;

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}
fn i64_bytes(data: &[i64]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f32_init(g: &mut Graph, name: &str, dims: &[usize], data: &[f32]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Float32, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(DataType::Float32, dims.to_vec(), f32_bytes(data))),
    );
    vid
}
fn i64_init(g: &mut Graph, name: &str, dims: &[usize], data: &[i64]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Int64, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(DataType::Int64, dims.to_vec(), i64_bytes(data))),
    );
    vid
}

/// Insert an op node whose single output is a fresh anonymous value. `out_dtype`
/// / `out_dims` describe the (tensor) output; Sequence-typed outputs pass a
/// placeholder shape (unused by the executor for sequence values).
fn op(
    g: &mut Graph,
    op_type: &str,
    inputs: &[ValueId],
    out_dtype: DataType,
    out_dims: &[usize],
    attrs: &[(&str, Attribute)],
) -> ValueId {
    g.opset_imports.entry(String::new()).or_insert(17);
    let out = g.create_value(out_dtype, static_shape(out_dims.iter().copied()));
    let mut node = Node::new(
        NodeId(0),
        op_type,
        inputs.iter().map(|&v| Some(v)).collect(),
        vec![out],
    );
    for (k, v) in attrs {
        node.attributes.insert((*k).to_string(), v.clone());
    }
    g.insert_node(node);
    out
}

/// SequenceConstruct(a, b, c) then SequenceAt(seq, idx) returns element `idx`.
#[test]
fn construct_then_at_roundtrips_values() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[2], &[1., 2.]);
    let b = f32_init(&mut g, "b", &[2], &[3., 4.]);
    let c = f32_init(&mut g, "c", &[2], &[5., 6.]);
    let seq = op(&mut g, "SequenceConstruct", &[a, b, c], DataType::Float32, &[], &[]);
    let idx = i64_init(&mut g, "idx", &[], &[1]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[2], &[]);
    g.add_output(at);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![2]);
    assert_eq!(out[0].to_vec_f32(), vec![3., 4.]);
}

/// Negative index counts from the end.
#[test]
fn at_negative_index() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[1], &[10.]);
    let b = f32_init(&mut g, "b", &[1], &[20.]);
    let seq = op(&mut g, "SequenceConstruct", &[a, b], DataType::Float32, &[], &[]);
    let idx = i64_init(&mut g, "idx", &[], &[-1]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[1], &[]);
    g.add_output(at);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_f32(), vec![20.]);
}

/// Insert at a position, then read it back.
#[test]
fn insert_at_position_then_at() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[1], &[1.]);
    let b = f32_init(&mut g, "b", &[1], &[2.]);
    let ins = f32_init(&mut g, "ins", &[1], &[99.]);
    let seq = op(&mut g, "SequenceConstruct", &[a, b], DataType::Float32, &[], &[]);
    let pos = i64_init(&mut g, "pos", &[], &[0]);
    let seq2 = op(&mut g, "SequenceInsert", &[seq, ins, pos], DataType::Float32, &[], &[]);
    // seq2 == [99, 1, 2]; read index 0
    let idx = i64_init(&mut g, "idx", &[], &[0]);
    let at = op(&mut g, "SequenceAt", &[seq2, idx], DataType::Float32, &[1], &[]);
    g.add_output(at);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_f32(), vec![99.]);
}

/// Insert with default (append) position, then length is 3.
#[test]
fn insert_default_append_and_length() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[1], &[1.]);
    let b = f32_init(&mut g, "b", &[1], &[2.]);
    let ins = f32_init(&mut g, "ins", &[1], &[3.]);
    let seq = op(&mut g, "SequenceConstruct", &[a, b], DataType::Float32, &[], &[]);
    let seq2 = op(&mut g, "SequenceInsert", &[seq, ins], DataType::Float32, &[], &[]);
    let len = op(&mut g, "SequenceLength", &[seq2], DataType::Int64, &[], &[]);
    g.add_output(len);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].dtype, DataType::Int64);
    assert_eq!(out[0].to_vec_i64(), vec![3]);
}

/// Exercise the required multi-op path in one graph, including negative
/// insertion and indexing plus scalar int64 length output.
#[test]
fn construct_insert_at_and_length_execute_together() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[1], &[1.]);
    let b = f32_init(&mut g, "b", &[1], &[2.]);
    let inserted = f32_init(&mut g, "inserted", &[1], &[99.]);
    let seq = op(&mut g, "SequenceConstruct", &[a, b], DataType::Float32, &[], &[]);
    let insert_pos = i64_init(&mut g, "insert_pos", &[], &[-1]);
    let seq2 = op(
        &mut g,
        "SequenceInsert",
        &[seq, inserted, insert_pos],
        DataType::Float32,
        &[],
        &[],
    );
    let at_pos = i64_init(&mut g, "at_pos", &[], &[-2]);
    let at = op(&mut g, "SequenceAt", &[seq2, at_pos], DataType::Float32, &[1], &[]);
    let len = op(&mut g, "SequenceLength", &[seq2], DataType::Int64, &[], &[]);
    g.add_output(at);
    g.add_output(len);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_f32(), vec![99.]);
    assert_eq!(out[1].dtype, DataType::Int64);
    assert!(out[1].shape.is_empty());
    assert_eq!(out[1].to_vec_i64(), vec![3]);
}

/// Erase an element (default last), then length drops by one.
#[test]
fn erase_then_length() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[1], &[1.]);
    let b = f32_init(&mut g, "b", &[1], &[2.]);
    let c = f32_init(&mut g, "c", &[1], &[3.]);
    let seq = op(&mut g, "SequenceConstruct", &[a, b, c], DataType::Float32, &[], &[]);
    let erased = op(&mut g, "SequenceErase", &[seq], DataType::Float32, &[], &[]);
    let len = op(&mut g, "SequenceLength", &[erased], DataType::Int64, &[], &[]);
    g.add_output(len);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_i64(), vec![2]);
}

/// The no-copy proof at the executor boundary: a tensor round-trips through a
/// sequence (Construct → Insert front → Erase front → At) and feeds `Identity`
/// (a contiguous-only consumer). Correct values prove the shared element bytes
/// were never corrupted, and — combined with the value-layer `Arc::ptr_eq`
/// unit test — that the round-trip did no deep copy of element data.
#[test]
fn seq_at_feeds_consumer_no_copy_roundtrip() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[3], &[7., 8., 9.]);
    let filler = f32_init(&mut g, "filler", &[3], &[0., 0., 0.]);
    let seq = op(&mut g, "SequenceConstruct", &[a], DataType::Float32, &[], &[]);
    let pos0 = i64_init(&mut g, "pos0", &[], &[0]);
    // Insert filler at front → [filler, a]; a is now index 1.
    let seq2 = op(&mut g, "SequenceInsert", &[seq, filler, pos0], DataType::Float32, &[], &[]);
    // Erase front → [a]; a is now index 0.
    let pos0b = i64_init(&mut g, "pos0b", &[], &[0]);
    let seq3 = op(&mut g, "SequenceErase", &[seq2, pos0b], DataType::Float32, &[], &[]);
    let idx = i64_init(&mut g, "idx", &[], &[0]);
    let at = op(&mut g, "SequenceAt", &[seq3, idx], DataType::Float32, &[3], &[]);
    // Consume the seq-backed tensor with Identity (reads it zero-copy).
    let ident = op(&mut g, "Identity", &[at], DataType::Float32, &[3], &[]);
    g.add_output(ident);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_f32(), vec![7., 8., 9.]);
}

/// SplitToSequence with no `split`: one element per slice along axis 0
/// (keepdims default 1), reassembled with ConcatFromSequence(axis=0) →
/// identity.
#[test]
fn split_then_concat_roundtrips() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[3, 2], &[1., 2., 3., 4., 5., 6.]);
    let seq = op(
        &mut g,
        "SplitToSequence",
        &[data],
        DataType::Float32,
        &[],
        &[("axis", Attribute::Int(0))],
    );
    let cat = op(
        &mut g,
        "ConcatFromSequence",
        &[seq],
        DataType::Float32,
        &[3, 2],
        &[("axis", Attribute::Int(0))],
    );
    g.add_output(cat);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![3, 2]);
    assert_eq!(out[0].to_vec_f32(), vec![1., 2., 3., 4., 5., 6.]);
}

/// SplitToSequence with keepdims=0 squeezes the split axis; each element is a
/// 1-D row. Read element 1 back.
#[test]
fn split_keepdims0_squeezes() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[3, 2], &[1., 2., 3., 4., 5., 6.]);
    let seq = op(
        &mut g,
        "SplitToSequence",
        &[data],
        DataType::Float32,
        &[],
        &[("axis", Attribute::Int(0)), ("keepdims", Attribute::Int(0))],
    );
    let idx = i64_init(&mut g, "idx", &[], &[1]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[2], &[]);
    g.add_output(at);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![2]); // axis 0 squeezed
    assert_eq!(out[0].to_vec_f32(), vec![3., 4.]);
}

/// SplitToSequence with explicit uneven `split` sizes along axis 1.
#[test]
fn split_explicit_sizes() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[2, 3], &[0., 1., 2., 3., 4., 5.]);
    let split = i64_init(&mut g, "split", &[2], &[1, 2]);
    let seq = op(
        &mut g,
        "SplitToSequence",
        &[data, split],
        DataType::Float32,
        &[],
        &[("axis", Attribute::Int(1))],
    );
    let idx = i64_init(&mut g, "idx", &[], &[1]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[2, 2], &[]);
    g.add_output(at);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![2, 2]);
    assert_eq!(out[0].to_vec_f32(), vec![1., 2., 4., 5.]);
}

/// A rank-1 split tensor with one element is a sizes vector, not a scalar
/// chunk-size input, and therefore must cover the whole split axis.
#[test]
fn split_one_element_1d_sizes_must_sum_to_axis() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[4], &[1., 2., 3., 4.]);
    let split = i64_init(&mut g, "split", &[1], &[2]);
    let seq = op(
        &mut g,
        "SplitToSequence",
        &[data, split],
        DataType::Float32,
        &[],
        &[],
    );
    let len = op(&mut g, "SequenceLength", &[seq], DataType::Int64, &[], &[]);
    g.add_output(len);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let err = s.run(&[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("sizes sum to 2"), "msg: {msg}");
    assert!(msg.contains("axis 0 has extent 4"), "msg: {msg}");
}

/// A one-element rank-1 split vector is valid when its size is the full axis.
#[test]
fn split_one_element_1d_sizes_produces_one_chunk() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[1], &[42.]);
    let split = i64_init(&mut g, "split", &[1], &[1]);
    let seq = op(&mut g, "SplitToSequence", &[data, split], DataType::Float32, &[], &[]);
    let idx = i64_init(&mut g, "idx", &[], &[0]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[1], &[]);
    g.add_output(at);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![1]);
    assert_eq!(out[0].to_vec_f32(), vec![42.]);
}

/// A rank-0 split tensor is the scalar chunk size, producing even chunks.
#[test]
fn split_scalar_chunk_size_produces_even_chunks() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[4], &[1., 2., 3., 4.]);
    let split = i64_init(&mut g, "split", &[], &[2]);
    let seq = op(&mut g, "SplitToSequence", &[data, split], DataType::Float32, &[], &[]);
    let len = op(&mut g, "SequenceLength", &[seq], DataType::Int64, &[], &[]);
    g.add_output(len);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_i64(), vec![2]);
}

/// Scalar chunk sizes retain an uneven final chunk when they do not divide axis.
#[test]
fn split_scalar_chunk_size_keeps_uneven_final_chunk() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[4], &[1., 2., 3., 4.]);
    let split = i64_init(&mut g, "split", &[], &[3]);
    let seq = op(&mut g, "SplitToSequence", &[data, split], DataType::Float32, &[], &[]);
    let idx = i64_init(&mut g, "idx", &[], &[1]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[1], &[]);
    g.add_output(at);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![1]);
    assert_eq!(out[0].to_vec_f32(), vec![4.]);
}

/// ConcatFromSequence with new_axis=1 stacks elements along a fresh axis.
#[test]
fn concat_new_axis_stacks() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[2], &[1., 2.]);
    let b = f32_init(&mut g, "b", &[2], &[3., 4.]);
    let seq = op(&mut g, "SequenceConstruct", &[a, b], DataType::Float32, &[], &[]);
    let cat = op(
        &mut g,
        "ConcatFromSequence",
        &[seq],
        DataType::Float32,
        &[2, 2],
        &[("axis", Attribute::Int(0)), ("new_axis", Attribute::Int(1))],
    );
    g.add_output(cat);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![2, 2]);
    assert_eq!(out[0].to_vec_f32(), vec![1., 2., 3., 4.]);
}

/// SequenceEmpty then SequenceInsert builds a one-element sequence.
#[test]
fn empty_then_insert() {
    let mut g = Graph::new();
    let t = f32_init(&mut g, "t", &[1], &[42.]);
    let seq = op(
        &mut g,
        "SequenceEmpty",
        &[],
        DataType::Float32,
        &[],
        &[("dtype", Attribute::Int(1))], // 1 == float32
    );
    let seq2 = op(&mut g, "SequenceInsert", &[seq, t], DataType::Float32, &[], &[]);
    let idx = i64_init(&mut g, "idx", &[], &[0]);
    let at = op(&mut g, "SequenceAt", &[seq2, idx], DataType::Float32, &[1], &[]);
    g.add_output(at);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let out = s.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_f32(), vec![42.]);
}

/// SequenceEmpty's declared dtype must be preserved even before its first insert.
#[test]
fn empty_insert_mismatched_dtype_errors_actionably() {
    let mut g = Graph::new();
    let t = f32_init(&mut g, "t", &[1], &[42.]);
    let seq = op(
        &mut g,
        "SequenceEmpty",
        &[],
        DataType::Int64,
        &[],
        &[("dtype", Attribute::Int(7))], // 7 == int64
    );
    let seq2 = op(&mut g, "SequenceInsert", &[seq, t], DataType::Int64, &[], &[]);
    let len = op(&mut g, "SequenceLength", &[seq2], DataType::Int64, &[], &[]);
    g.add_output(len);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let err = s.run(&[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("does not match"), "msg: {msg}");
    assert!(msg.contains("To fix"), "msg: {msg}");
}

/// Out-of-bounds SequenceAt is an actionable error (states valid range).
#[test]
fn at_out_of_bounds_errors_actionably() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[1], &[1.]);
    let seq = op(&mut g, "SequenceConstruct", &[a], DataType::Float32, &[], &[]);
    let idx = i64_init(&mut g, "idx", &[], &[5]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[1], &[]);
    g.add_output(at);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let err = s.run(&[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("out of bounds"), "msg: {msg}");
    assert!(msg.contains("valid range"), "msg: {msg}");
}

#[test]
fn at_empty_sequence_errors_cleanly() {
    let mut g = Graph::new();
    let seq = op(
        &mut g,
        "SequenceEmpty",
        &[],
        DataType::Float32,
        &[],
        &[("dtype", Attribute::Int(1))],
    );
    let idx = i64_init(&mut g, "idx", &[], &[0]);
    let at = op(&mut g, "SequenceAt", &[seq, idx], DataType::Float32, &[1], &[]);
    g.add_output(at);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let err = s.run(&[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("out of bounds"), "msg: {msg}");
    assert!(msg.contains("length 0"), "msg: {msg}");
}

#[test]
fn construct_non_homogeneous_dtypes_errors_cleanly() {
    let mut g = Graph::new();
    let floats = f32_init(&mut g, "floats", &[1], &[1.]);
    let integers = i64_init(&mut g, "integers", &[1], &[2]);
    let seq = op(
        &mut g,
        "SequenceConstruct",
        &[floats, integers],
        DataType::Float32,
        &[],
        &[],
    );
    let len = op(&mut g, "SequenceLength", &[seq], DataType::Int64, &[], &[]);
    g.add_output(len);

    let mut s = InferenceSession::from_graph(g).expect("build");
    let err = s.run(&[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("SequenceConstruct"), "msg: {msg}");
    assert!(msg.contains("homogeneous"), "msg: {msg}");
    assert!(msg.contains("does not match"), "msg: {msg}");
}

/// A raw Sequence graph output is rejected with an actionable message.
#[test]
fn sequence_graph_output_rejected() {
    let mut g = Graph::new();
    let a = f32_init(&mut g, "a", &[1], &[1.]);
    let seq = op(&mut g, "SequenceConstruct", &[a], DataType::Float32, &[], &[]);
    g.add_output(seq);
    let mut s = InferenceSession::from_graph(g).expect("build");
    let err = s.run(&[]).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("Sequence value"), "msg: {msg}");
    assert!(msg.contains("ConcatFromSequence") || msg.contains("SequenceAt"), "msg: {msg}");
}
