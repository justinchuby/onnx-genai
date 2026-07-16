//! Integration tests for the sequential CPU executor (Track D).
//!
//! Each test hand-builds a small [`Graph`] via the IR API, runs it through the
//! public [`InferenceSession`] surface, and asserts the output matches a
//! reference computed here in the test. Nothing below names a model or bakes in
//! a fixed shape path — the executor is exercised as a generic Graph runner.

use onnx_runtime_ir::{
    static_shape, Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, ValueId,
    WeightRef,
};
use onnx_runtime_session::{InferenceSession, SessionError, Tensor, WarmupShape};
use onnx_runtime_shape_inference::{InferenceRegistry, MergePolicy};

// --- graph construction helpers --------------------------------------------

fn f32_bytes(data: &[f32]) -> Vec<u8> {
    data.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// Add an inline f32 initializer, returning its value id.
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

/// Add a named graph input, returning its value id.
fn input(g: &mut Graph, name: &str, dtype: DataType, dims: &[usize]) -> ValueId {
    let vid = g.create_named_value(name, dtype, static_shape(dims.iter().copied()));
    g.add_input(vid);
    vid
}

/// Insert an op node producing a single output value of the given shape/dtype.
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

/// Add a named graph input with an explicit (possibly symbolic) shape.
fn input_shaped(g: &mut Graph, name: &str, dtype: DataType, shape: Shape) -> ValueId {
    let vid = g.create_named_value(name, dtype, shape);
    g.add_input(vid);
    vid
}

fn i32_tensor(shape: &[usize], data: &[i32]) -> Tensor {
    let bytes: Vec<u8> = data.iter().flat_map(|value| value.to_le_bytes()).collect();
    Tensor::from_raw(DataType::Int32, shape.to_vec(), &bytes).unwrap()
}

fn gqa_cache_graph(past_capacity: usize) -> Graph {
    let mut g = Graph::new();
    g.opset_imports.insert(String::new(), 17);
    g.opset_imports.insert("com.microsoft".into(), 1);

    let query = input(&mut g, "query", DataType::Float32, &[1, 1, 8]);
    let key = input(&mut g, "key", DataType::Float32, &[1, 1, 4]);
    let value = input(&mut g, "value", DataType::Float32, &[1, 1, 4]);
    let past_key = input(
        &mut g,
        "past_key",
        DataType::Float32,
        &[1, 2, past_capacity, 2],
    );
    let past_value = input(
        &mut g,
        "past_value",
        DataType::Float32,
        &[1, 2, past_capacity, 2],
    );
    let seqlens = input(&mut g, "seqlens_k", DataType::Int32, &[1]);
    let total = input(&mut g, "total_sequence_length", DataType::Int32, &[]);

    let attention = g.create_value(DataType::Float32, vec![]);
    let present_key = g.create_value(DataType::Float32, vec![]);
    let present_value = g.create_value(DataType::Float32, vec![]);
    let mut node = Node::new(
        NodeId(0),
        "GroupQueryAttention",
        vec![
            Some(query),
            Some(key),
            Some(value),
            Some(past_key),
            Some(past_value),
            Some(seqlens),
            Some(total),
        ],
        vec![attention, present_key, present_value],
    );
    node.domain = "com.microsoft".into();
    node.attributes
        .insert("num_heads".into(), Attribute::Int(4));
    node.attributes
        .insert("kv_num_heads".into(), Attribute::Int(2));
    g.insert_node(node);

    let registry = InferenceRegistry::default_registry();
    let imports = g.opset_imports.clone();
    registry
        .infer_graph(&mut g, &imports, MergePolicy::Permissive)
        .expect("infer GQA output shapes");
    g.add_output(attention);
    g.add_output(present_key);
    g.add_output(present_value);
    g
}

fn run_gqa_decode(past_capacity: usize) -> Vec<Tensor> {
    let mut session =
        InferenceSession::from_graph(gqa_cache_graph(past_capacity)).expect("build GQA session");
    let query = Tensor::from_f32(&[1, 1, 8], &[1.0; 8]).unwrap();
    let key = Tensor::from_f32(&[1, 1, 4], &[0.5; 4]).unwrap();
    let value = Tensor::from_f32(&[1, 1, 4], &[2.0; 4]).unwrap();
    let past_data = vec![0.25; 4 * past_capacity];
    let past_key = Tensor::from_f32(&[1, 2, past_capacity, 2], &past_data).unwrap();
    let past_value = Tensor::from_f32(&[1, 2, past_capacity, 2], &past_data).unwrap();
    let seqlens = i32_tensor(&[1], &[2]);
    let total = i32_tensor(&[], &[3]);
    session
        .run(&[
            ("query", &query),
            ("key", &key),
            ("value", &value),
            ("past_key", &past_key),
            ("past_value", &past_value),
            ("seqlens_k", &seqlens),
            ("total_sequence_length", &total),
        ])
        .expect("GQA decode succeeds")
}

#[test]
fn gqa_decode_fixed_capacity_preserves_present_cache_extent() {
    let outputs = run_gqa_decode(8);
    assert_eq!(outputs[0].shape, vec![1, 1, 8]);
    assert_eq!(outputs[1].shape, vec![1, 2, 8, 2]);
    assert_eq!(outputs[2].shape, vec![1, 2, 8, 2]);
}

#[test]
fn gqa_decode_growing_cache_extends_present_to_logical_total() {
    let outputs = run_gqa_decode(2);
    assert_eq!(outputs[0].shape, vec![1, 1, 8]);
    assert_eq!(outputs[1].shape, vec![1, 2, 3, 2]);
    assert_eq!(outputs[2].shape, vec![1, 2, 3, 2]);
}

/// Insert an op node whose single output carries an explicit (possibly
/// symbolic) shape — mirroring what the loader's shape inference would produce.
fn op_shaped(
    g: &mut Graph,
    op_type: &str,
    inputs: &[ValueId],
    out_dtype: DataType,
    out_shape: Shape,
    attrs: &[(&str, Attribute)],
) -> ValueId {
    g.opset_imports.entry(String::new()).or_insert(17);
    let out = g.create_value(out_dtype, out_shape);
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

#[test]
fn unsupported_op_error_is_actionable() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    let mut node = Node::new(NodeId(0), "Conv", vec![Some(x)], vec![y]);
    node.name = "unsupported_activation".to_string();
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("unsupported operator unexpectedly built"),
    };
    assert!(message.contains("Conv"), "{message}");
    assert!(message.contains("ai.onnx"), "{message}");
    assert!(message.contains("unsupported_activation"), "{message}");
    assert!(message.contains("opset 17"), "{message}");
    assert!(message.contains("cpu_ep"), "{message}");
    assert!(message.contains("To fix:"), "{message}");
}

#[test]
fn unsupported_op_error_formats_unnamed_node_gracefully() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 0);
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Conv",
        vec![Some(x)],
        vec![y],
    ));
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("unsupported operator unexpectedly built"),
    };
    assert!(
        message.contains("node <unnamed node #0>, opset 0"),
        "{message}"
    );
    assert!(!message.contains("node \"\""), "{message}");
}

#[test]
fn from_graph_rejects_missing_opset_import_at_load_time() {
    let mut graph = Graph::new();
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    let mut node = Node::new(NodeId(0), "Sigmoid", vec![Some(x)], vec![y]);
    node.name = "missing_opset_import".to_string();
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("illegal graph unexpectedly built"),
    };
    assert_eq!(
        message,
        "illegal ONNX model: operator ai.onnx::Sigmoid at node \"missing_opset_import\" uses \
         domain 'ai.onnx' but no corresponding opset_import is declared. RULES #1: the model must \
         declare an opset_import for domain 'ai.onnx'; if you built this graph programmatically, \
         add it before loading; if this is a file, the model is malformed/invalid per the ONNX spec"
    );
    assert!(message.contains("Sigmoid"), "{message}");
    assert!(message.contains("ai.onnx"), "{message}");
    assert!(message.contains("RULES #1"), "{message}");
    assert!(!message.contains("18446744073709551615"), "{message}");
}

// --- reference implementations ---------------------------------------------

fn ref_matmul(a: &[f32], m: usize, k: usize, b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

fn ref_add_rowvec(m: &[f32], rows: usize, cols: usize, bias: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] = m[r * cols + c] + bias[c];
        }
    }
    out
}

fn ref_layernorm_last(
    x: &[f32],
    rows: usize,
    cols: usize,
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..r * cols + cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * inv * scale[c] + bias[c];
        }
    }
    out
}

fn ref_relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v.max(0.0)).collect()
}

fn assert_close(got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "length mismatch");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() < 1e-4, "element {i}: got {g}, want {w}");
    }
}

// --- tests ------------------------------------------------------------------

/// MatMul → Add → LayerNormalization → Relu, a realistic multi-node chain.
#[test]
fn matmul_add_layernorm_relu_chain_matches_reference() {
    // Dimensions: X[2,3] · W[3,4] → [2,4], + bias[4], layernorm last axis, relu.
    let x_data = [0.5f32, -1.0, 2.0, 1.5, 0.0, -0.5];
    let w_data = [
        0.1f32, 0.2, -0.3, 0.4, //
        -0.5, 0.6, 0.7, -0.8, //
        0.9, -1.0, 0.2, 0.3,
    ];
    let bias = [0.1f32, -0.2, 0.3, 0.05];
    let scale = [1.2f32, 0.8, 1.0, 0.5];
    let ln_bias = [0.0f32, 0.1, -0.1, 0.2];

    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[2, 3]);
    let w = f32_init(&mut g, "W", &[3, 4], &w_data);
    let m = op(&mut g, "MatMul", &[x, w], DataType::Float32, &[2, 4], &[]);
    let b = f32_init(&mut g, "B", &[4], &bias);
    let a = op(&mut g, "Add", &[m, b], DataType::Float32, &[2, 4], &[]);
    let s = f32_init(&mut g, "Scale", &[4], &scale);
    let bn = f32_init(&mut g, "LnBias", &[4], &ln_bias);
    let l = op(
        &mut g,
        "LayerNormalization",
        &[a, s, bn],
        DataType::Float32,
        &[2, 4],
        &[("axis", Attribute::Int(-1))],
    );
    let y = op(&mut g, "Relu", &[l], DataType::Float32, &[2, 4], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build session");

    let x_tensor = Tensor::from_f32(&[2, 3], &x_data).unwrap();
    let outputs = session.run(&[("X", &x_tensor)]).expect("run");
    assert_eq!(outputs.len(), 1);

    // Reference.
    let m_ref = ref_matmul(&x_data, 2, 3, &w_data, 4);
    let a_ref = ref_add_rowvec(&m_ref, 2, 4, &bias);
    let l_ref = ref_layernorm_last(&a_ref, 2, 4, &scale, &ln_bias, 1e-5);
    let y_ref = ref_relu(&l_ref);

    assert_close(&outputs[0].to_vec_f32(), &y_ref);
    assert_eq!(outputs[0].shape, vec![2, 4]);
}

/// Gather (embedding lookup) → Transpose, exercising an integer-index op and a
/// layout-permuting op in one graph.
#[test]
fn gather_then_transpose_matches_reference() {
    // Embedding table [4,3]; gather rows [2,0,3] → [3,3]; transpose → [3,3]^T.
    let table = [
        0.0f32, 1.0, 2.0, //
        3.0, 4.0, 5.0, //
        6.0, 7.0, 8.0, //
        9.0, 10.0, 11.0,
    ];
    let idx = [2i64, 0, 3];

    let mut g = Graph::new();
    let data = f32_init(&mut g, "Table", &[4, 3], &table);
    let indices = input(&mut g, "Idx", DataType::Int64, &[3]);
    let gathered = op(
        &mut g,
        "Gather",
        &[data, indices],
        DataType::Float32,
        &[3, 3],
        &[("axis", Attribute::Int(0))],
    );
    let transposed = op(
        &mut g,
        "Transpose",
        &[gathered],
        DataType::Float32,
        &[3, 3],
        &[("perm", Attribute::Ints(vec![1, 0]))],
    );
    g.add_output(transposed);

    let mut session = InferenceSession::from_graph(g).expect("build session");
    let idx_tensor = Tensor::from_i64(&[3], &idx).unwrap();
    let outputs = session.run(&[("Idx", &idx_tensor)]).expect("run");

    // Reference: gather rows then transpose 3x3.
    let mut gathered_ref = Vec::new();
    for &i in &idx {
        let base = i as usize * 3;
        gathered_ref.extend_from_slice(&table[base..base + 3]);
    }
    let mut want = vec![0.0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            want[c * 3 + r] = gathered_ref[r * 3 + c];
        }
    }
    assert_close(&outputs[0].to_vec_f32(), &want);
}

/// The shape-keyed kernel cache is populated once and reused on every run: hits
/// grow while the compiled-entry count and miss count stay fixed (§11.1).
#[test]
fn shape_keyed_cache_is_reused_across_runs() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[2, 2]);
    let w = f32_init(&mut g, "W", &[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let m = op(&mut g, "MatMul", &[x, w], DataType::Float32, &[2, 2], &[]);
    let y = op(&mut g, "Relu", &[m], DataType::Float32, &[2, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");

    // After build (compile pass): every node compiled once, no hits.
    let after_build = session.cache_stats();
    assert_eq!(after_build.entries, 2, "two nodes compiled");
    assert_eq!(after_build.misses, 2);
    assert_eq!(after_build.hits, 0);

    let x_tensor = Tensor::from_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]).unwrap();

    let out1 = session.run(&[("X", &x_tensor)]).unwrap();
    let after_run1 = session.cache_stats();
    assert_eq!(after_run1.entries, 2, "no new entries on run");
    assert_eq!(after_run1.misses, 2, "no recompilation");
    assert_eq!(after_run1.hits, 2, "each node served from cache");

    let out2 = session.run(&[("X", &x_tensor)]).unwrap();
    let after_run2 = session.cache_stats();
    assert_eq!(after_run2.entries, 2);
    assert_eq!(after_run2.misses, 2);
    assert_eq!(after_run2.hits, 4, "second run hit the cache again");

    // Identity matmul + relu of [1,2,3,4] → [1,2,3,4].
    assert_close(&out1[0].to_vec_f32(), &[1.0, 2.0, 3.0, 4.0]);
    assert_close(&out2[0].to_vec_f32(), &[1.0, 2.0, 3.0, 4.0]);
}

/// `warmup` names must reference real inputs; a bad name is rejected, a good
/// one keeps the cache warm.
#[test]
fn warmup_validates_input_names() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[1, 2]);
    let y = op(&mut g, "Relu", &[x], DataType::Float32, &[1, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).unwrap();
    assert!(session
        .warmup(&[WarmupShape {
            input_name: "nope".into(),
            shape: vec![1, 2],
        }])
        .is_err());
    assert!(session
        .warmup(&[WarmupShape {
            input_name: "X".into(),
            shape: vec![1, 2],
        }])
        .is_ok());
}

/// A missing required input is reported, not silently defaulted.
#[test]
fn missing_input_is_rejected() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[1, 2]);
    let y = op(&mut g, "Relu", &[x], DataType::Float32, &[1, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).unwrap();
    let err = session.run(&[]).unwrap_err();
    assert!(matches!(
        err,
        onnx_runtime_session::SessionError::InputNotFound { .. }
    ));
}

/// A shape-mismatched input tensor is rejected before dispatch.
#[test]
fn input_shape_mismatch_is_rejected() {
    let mut g = Graph::new();
    let x = input(&mut g, "X", DataType::Float32, &[2, 2]);
    let y = op(&mut g, "Relu", &[x], DataType::Float32, &[2, 2], &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).unwrap();
    let wrong = Tensor::from_f32(&[3, 2], &[0.0; 6]).unwrap();
    let err = session.run(&[("X", &wrong)]).unwrap_err();
    assert!(matches!(
        err,
        onnx_runtime_session::SessionError::ShapeMismatch { .. }
    ));
}

// --- dynamic (symbolic) shape tests ----------------------------------------

/// A graph with a symbolic leading dim (`[batch, 4]` MatMul → Add → Relu) runs
/// correctly for two *different* batch sizes in the same session: shapes resolve
/// from the actual inputs, buffers re-size, and the kernel cache re-resolves for
/// the new shape while reusing the plan for a repeated shape.
#[test]
fn symbolic_batch_matmul_chain_runs_for_multiple_shapes() {
    let w_data = [
        1.0f32, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];
    let bias = [0.5f32, -0.5, 1.0, -1.0];

    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let sym_row = || vec![Dim::Symbolic(batch), Dim::Static(4)];

    let x = input_shaped(&mut g, "X", DataType::Float32, sym_row());
    let w = f32_init(&mut g, "W", &[4, 4], &w_data);
    let m = op_shaped(&mut g, "MatMul", &[x, w], DataType::Float32, sym_row(), &[]);
    let b = f32_init(&mut g, "B", &[4], &bias);
    let a = op_shaped(&mut g, "Add", &[m, b], DataType::Float32, sym_row(), &[]);
    let y = op_shaped(&mut g, "Relu", &[a], DataType::Float32, sym_row(), &[]);
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build symbolic session");

    // A symbolic graph is not compiled at build (no concrete shapes yet).
    let after_build = session.cache_stats();
    assert_eq!(after_build.entries, 0, "no kernels compiled before first run");
    assert_eq!(after_build.misses, 0);

    let run_batch = |session: &mut InferenceSession, rows: usize, fill: f32| -> Vec<f32> {
        let data: Vec<f32> = (0..rows * 4).map(|i| fill + i as f32).collect();
        let x_tensor = Tensor::from_f32(&[rows, 4], &data).unwrap();
        let out = session.run(&[("X", &x_tensor)]).expect("run");
        assert_eq!(out[0].shape, vec![rows, 4]);
        // Reference: identity matmul + row bias + relu.
        let m_ref = ref_matmul(&data, rows, 4, &w_data, 4);
        let a_ref = ref_add_rowvec(&m_ref, rows, 4, &bias);
        let y_ref = ref_relu(&a_ref);
        assert_close(&out[0].to_vec_f32(), &y_ref);
        out[0].to_vec_f32()
    };

    // batch = 2 → first shape: three nodes compiled (misses), no hits.
    run_batch(&mut session, 2, 0.0);
    let s2 = session.cache_stats();
    assert_eq!(s2.entries, 3, "three nodes compiled for batch=2");
    assert_eq!(s2.misses, 3);
    assert_eq!(s2.hits, 0);

    // batch = 3 → new resolved shape: re-resolves + re-plans (3 more entries).
    run_batch(&mut session, 3, 10.0);
    let s3 = session.cache_stats();
    assert_eq!(s3.entries, 6, "batch=3 adds three distinct shape-keyed entries");
    assert_eq!(s3.misses, 6);
    assert_eq!(s3.hits, 0);

    // batch = 2 again → the batch=2 plan is reused (cache hits, no new entries).
    run_batch(&mut session, 2, 100.0);
    let s2b = session.cache_stats();
    assert_eq!(s2b.entries, 6, "no new entries: batch=2 plan reused");
    assert_eq!(s2b.misses, 6);
    assert_eq!(s2b.hits, 3, "each node served from the batch=2 cache");
}

/// Two inputs share a symbol (`batch`); supplying them with *conflicting*
/// concrete sizes is a resolution error, not a silently-wrong run.
#[test]
fn symbol_conflict_across_inputs_is_rejected() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let sym_row = || vec![Dim::Symbolic(batch), Dim::Static(4)];

    let a = input_shaped(&mut g, "A", DataType::Float32, sym_row());
    let b = input_shaped(&mut g, "B", DataType::Float32, sym_row());
    let s = op_shaped(&mut g, "Add", &[a, b], DataType::Float32, sym_row(), &[]);
    g.add_output(s);

    let mut session = InferenceSession::from_graph(g).expect("build");

    let a_t = Tensor::from_f32(&[2, 4], &[0.0; 8]).unwrap();
    let b_t = Tensor::from_f32(&[3, 4], &[0.0; 12]).unwrap();
    let err = session.run(&[("A", &a_t), ("B", &b_t)]).unwrap_err();
    assert!(
        matches!(err, SessionError::SymbolConflict { .. }),
        "expected SymbolConflict, got {err:?}"
    );

    // Agreeing sizes resolve fine.
    let a_ok = Tensor::from_f32(&[2, 4], &[1.0; 8]).unwrap();
    let b_ok = Tensor::from_f32(&[2, 4], &[2.0; 8]).unwrap();
    let out = session.run(&[("A", &a_ok), ("B", &b_ok)]).expect("run");
    assert_close(&out[0].to_vec_f32(), &[3.0; 8]);
    assert_eq!(out[0].shape, vec![2, 4]);
}

/// A value whose shape carries a symbol that no input binds cannot be sized:
/// the session reports it as an uninferred shape naming the producing op,
/// rather than guessing (the loader-shape-inference-gap signal, §5).
#[test]
fn unresolved_symbol_reports_uninferred_shape() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let ghost = g.intern_symbol("ghost"); // never appears on any input

    let x = input_shaped(
        &mut g,
        "X",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
    );
    // Relu output declares an unbindable symbol on its leading dim.
    let y = op_shaped(
        &mut g,
        "Relu",
        &[x],
        DataType::Float32,
        vec![Dim::Symbolic(ghost), Dim::Static(4)],
        &[],
    );
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");
    let x_t = Tensor::from_f32(&[2, 4], &[0.0; 8]).unwrap();
    let err = session.run(&[("X", &x_t)]).unwrap_err();
    assert!(
        matches!(err, SessionError::UnresolvedShape { ref op, .. } if op == "Relu"),
        "expected UnresolvedShape naming the producing op, got {err:?}"
    );
}

/// A symbolic input supplied with the wrong rank is rejected before dispatch.
#[test]
fn symbolic_input_rank_mismatch_is_rejected() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let x = input_shaped(
        &mut g,
        "X",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
    );
    let y = op_shaped(
        &mut g,
        "Relu",
        &[x],
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
        &[],
    );
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");
    // Rank-3 tensor for a rank-2 declared input.
    let wrong = Tensor::from_f32(&[2, 2, 4], &[0.0; 16]).unwrap();
    let err = session.run(&[("X", &wrong)]).unwrap_err();
    assert!(
        matches!(err, SessionError::RankMismatch { .. }),
        "expected RankMismatch, got {err:?}"
    );
}

/// A static dim declared alongside a symbolic one must still match exactly.
#[test]
fn symbolic_input_static_dim_mismatch_is_rejected() {
    let mut g = Graph::new();
    let batch = g.intern_symbol("batch");
    let x = input_shaped(
        &mut g,
        "X",
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
    );
    let y = op_shaped(
        &mut g,
        "Relu",
        &[x],
        DataType::Float32,
        vec![Dim::Symbolic(batch), Dim::Static(4)],
        &[],
    );
    g.add_output(y);

    let mut session = InferenceSession::from_graph(g).expect("build");
    // batch is free, but the trailing static dim (4) is violated (here 5).
    let wrong = Tensor::from_f32(&[2, 5], &[0.0; 10]).unwrap();
    let err = session.run(&[("X", &wrong)]).unwrap_err();
    assert!(
        matches!(err, SessionError::ShapeMismatch { .. }),
        "expected ShapeMismatch on the static dim, got {err:?}"
    );
}

/// A subgraph-bearing op the CPU EP cannot execute (anything other than the
/// implemented `If`/`Loop`/`Scan`) is rejected at session-build time
/// (from_graph path), mirroring the disk loader — we fail fast with a RULES #1
/// message instead of lazily at run time or silently skipping the subgraph.
/// The three implemented control-flow ops are covered by `tests/control_flow.rs`.
#[test]
fn from_graph_rejects_unimplemented_control_flow_subgraph_at_build() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[1]);
    let y = graph.create_named_value("y", DataType::Float32, static_shape([1]));
    // `SequenceMap` is a real ONNX subgraph-bearing op this runtime does not
    // implement — it must still be rejected fast.
    let mut node = Node::new(NodeId(0), "SequenceMap", vec![Some(x)], vec![y]);
    node.name = "control_flow_seqmap".to_string();
    node.attributes
        .insert("body".to_string(), Attribute::Graph(Box::new(Graph::new())));
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("unimplemented control-flow subgraph unexpectedly built"),
    };
    assert!(message.contains("SequenceMap"), "{message}");
    assert!(message.contains("body"), "{message}");
    assert!(message.contains("control-flow"), "{message}");
    assert!(message.contains("RULES #1"), "{message}");
}

/// A node consuming an unsourced tensor is rejected at session-build time.
#[test]
fn from_graph_rejects_dangling_tensor_reference_at_build() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 17);
    let x = input(&mut graph, "x", DataType::Float32, &[2]);
    // `z` is created but never sourced (no input, initializer, or producer).
    let z = graph.create_named_value("z", DataType::Float32, static_shape([2]));
    let y = graph.create_named_value("y", DataType::Float32, static_shape([2]));
    let mut node = Node::new(NodeId(0), "Add", vec![Some(x), Some(z)], vec![y]);
    node.name = "dangling_add".to_string();
    graph.insert_node(node);
    graph.add_output(y);

    let message = match InferenceSession::from_graph(graph) {
        Err(err) => err.to_string(),
        Ok(_) => panic!("dangling reference unexpectedly built"),
    };
    assert!(message.contains("'z'"), "{message}");
    assert!(message.contains("Add"), "{message}");
    assert!(message.contains("RULES #1"), "{message}");
}
