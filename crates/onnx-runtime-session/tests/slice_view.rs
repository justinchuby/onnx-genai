//! Integration tests for zero-copy `Slice` (the layout/movement-op view
//! foundation): a pure sub-view emits a strided [`ViewOutput`] aliasing the
//! input buffer instead of gathering bytes. These tests exercise the executor
//! end-to-end through the public [`InferenceSession`] surface, covering:
//!
//! * a Slice-view as a graph OUTPUT → materialized contiguous at the boundary;
//! * a Slice-view feeding a contiguous-only consumer (`Identity`) → auto
//!   materialized to correct values;
//! * a Slice-of-a-Slice → the second view composes over the first (single hop);
//! * a reversed slice (negative step → negative stride) materialized correctly;
//! * buffer liveness — a source stays valid across a Slice→consumer chain;
//! * the error path for an unsupported slice parameter (step == 0).

use onnx_runtime_ir::{
    static_shape, Attribute, DataType, Graph, Node, NodeId, TensorData, ValueId, WeightRef,
};
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
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            dims.to_vec(),
            f32_bytes(data),
        )),
    );
    vid
}

fn i64_init(g: &mut Graph, name: &str, dims: &[usize], data: &[i64]) -> ValueId {
    let vid = g.create_named_value(name, DataType::Int64, static_shape(dims.iter().copied()));
    g.set_initializer(
        vid,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            dims.to_vec(),
            i64_bytes(data),
        )),
    );
    vid
}

/// Insert an op node producing one output value of the given static shape.
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

/// Insert a `Slice(data, starts, ends, axes, steps)` node whose four parameter
/// inputs are inline initializers, returning the output value id.
fn slice_node(
    g: &mut Graph,
    data: ValueId,
    tag: &str,
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
    out_dims: &[usize],
) -> ValueId {
    let s = i64_init(g, &format!("{tag}_starts"), &[starts.len()], starts);
    let e = i64_init(g, &format!("{tag}_ends"), &[ends.len()], ends);
    let a = i64_init(g, &format!("{tag}_axes"), &[axes.len()], axes);
    let st = i64_init(g, &format!("{tag}_steps"), &[steps.len()], steps);
    op(
        g,
        "Slice",
        &[data, s, e, a, st],
        DataType::Float32,
        out_dims,
        &[],
    )
}

/// (c) A Slice-view that is a graph OUTPUT is materialized contiguous at the
/// executor boundary — external consumers get a normal owned tensor.
#[test]
fn slice_view_as_graph_output_materializes_contiguous() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
    // rows [1:3] → indices 1,2 → [[3,4],[5,6]]
    let sliced = slice_node(&mut g, data, "s", &[1], &[3], &[0], &[1], &[2, 2]);
    g.add_output(sliced);

    let mut session = InferenceSession::from_graph(g).expect("build");
    let out = session.run(&[]).expect("run");
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].shape, vec![2, 2]);
    assert_eq!(out[0].to_vec_f32(), vec![3., 4., 5., 6.]);
}

/// (a)/(d) A Slice-view feeding a strided-capable consumer stays a view (no
/// materialization), and a source buffer remains valid across the whole chain.
/// Slice→Slice: the second Slice composes over the first view (single hop),
/// then the output is materialized at the boundary.
#[test]
fn slice_view_chain_composes_and_keeps_source_alive() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
    // rows [1:3] → indices 1,2 → [[3,4],[5,6]] (a view over `data`)
    let rows = slice_node(&mut g, data, "s0", &[1], &[3], &[0], &[1], &[2, 2]);
    // col [0:1] of that view → [[3],[5]] (a view composed over the first view)
    let col = slice_node(&mut g, rows, "s1", &[0], &[1], &[1], &[1], &[2, 1]);
    g.add_output(col);

    let mut session = InferenceSession::from_graph(g).expect("build");
    let out = session.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![2, 1]);
    // If `data`'s buffer had been freed under the live views, this would be
    // garbage; correct values prove the source outlived every alias.
    assert_eq!(out[0].to_vec_f32(), vec![3., 5.]);
}

/// (b) A Slice-view feeding a contiguous-only consumer (`Identity`, which
/// declares `supports_strided_input == false`) is auto-materialized into a
/// private contiguous temp so the consumer sees correct, dense data.
#[test]
fn slice_view_into_contiguous_only_consumer_is_materialized() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[4, 2], &[1., 2., 3., 4., 5., 6., 7., 8.]);
    // Strided view: rows [0:4:2] → rows 0 and 2 → [[1,2],[5,6]] (stride 2 on ax0)
    let strided = slice_node(&mut g, data, "s", &[0], &[4], &[0], &[2], &[2, 2]);
    let ident = op(&mut g, "Identity", &[strided], DataType::Float32, &[2, 2], &[]);
    g.add_output(ident);

    let mut session = InferenceSession::from_graph(g).expect("build");
    let out = session.run(&[]).expect("run");
    assert_eq!(out[0].shape, vec![2, 2]);
    assert_eq!(out[0].to_vec_f32(), vec![1., 2., 5., 6.]);
}

/// A reversed slice (negative step → negative stride) is a valid view; the
/// boundary gather honors the negative stride and offset.
#[test]
fn slice_view_negative_step_reverses_correctly() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[5], &[1., 2., 3., 4., 5.]);
    // [4:-6:-1] → whole reversed → [5,4,3,2,1]
    let rev = slice_node(&mut g, data, "s", &[4], &[-6], &[0], &[-1], &[5]);
    g.add_output(rev);

    let mut session = InferenceSession::from_graph(g).expect("build");
    let out = session.run(&[]).expect("run");
    assert_eq!(out[0].to_vec_f32(), vec![5., 4., 3., 2., 1.]);
}

/// (e) An unsupported slice parameter (step == 0) surfaces an actionable error
/// (the view fast path declines, the copy path rejects it) rather than silently
/// producing wrong data.
#[test]
fn slice_zero_step_reports_actionable_error() {
    let mut g = Graph::new();
    let data = f32_init(&mut g, "data", &[5], &[1., 2., 3., 4., 5.]);
    let sliced = slice_node(&mut g, data, "s", &[0], &[5], &[0], &[0], &[5]);
    g.add_output(sliced);

    let mut session = InferenceSession::from_graph(g).expect("build");
    let err = session.run(&[]).expect_err("step 0 must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains("step"),
        "error should name the offending parameter, got: {msg}"
    );
}
