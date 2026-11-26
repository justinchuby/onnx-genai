use onnx_runtime_ir::{
    Attribute, DataType, Dim, Graph, Node, NodeId, TensorData, ValueId, WeightRef, static_shape,
};
use onnx_runtime_session::{InferenceSession, Tensor};

fn graph(opset: u64) -> Graph {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), opset);
    graph
}

fn input(graph: &mut Graph, name: &str, dims: &[usize]) -> ValueId {
    let id = graph.create_named_value(name, DataType::String, static_shape(dims.iter().copied()));
    graph.add_input(id);
    id
}

fn node(
    graph: &mut Graph,
    op: &str,
    inputs: &[ValueId],
    outputs: &[ValueId],
    attrs: &[(&str, Attribute)],
) -> NodeId {
    let mut node = Node::new(
        NodeId(0),
        op,
        inputs.iter().copied().map(Some).collect(),
        outputs.to_vec(),
    );
    node.attributes.extend(
        attrs
            .iter()
            .map(|(key, attr)| ((*key).to_owned(), attr.clone())),
    );
    graph.insert_node(node)
}

fn strings(dims: &[usize], values: &[&str]) -> Tensor {
    Tensor::from_strings(dims.to_vec(), values.iter().map(|s| (*s).into()).collect()).unwrap()
}

fn assert_strings(tensor: &Tensor, dims: &[usize], values: &[&str]) {
    assert_eq!(tensor.dtype, DataType::String);
    assert_eq!(tensor.shape, dims);
    assert_eq!(tensor.as_strings().unwrap(), values);
    assert!(tensor.device().is_host_accessible());
    assert!(tensor.device_ptr().is_null());
}

fn compress_graph(opset: u64, axis: Option<i64>, dtype: DataType, dims: &[usize]) -> Graph {
    let mut graph = graph(opset);
    let x = input(&mut graph, "X", &[2, 3]);
    let condition =
        graph.create_named_value("condition", dtype, static_shape(dims.iter().copied()));
    graph.add_input(condition);
    let selected = Dim::Symbolic(graph.intern_symbol("selected"));
    let output_shape = match axis {
        None => vec![selected],
        Some(0) => vec![selected, Dim::Static(3)],
        Some(-1) => vec![Dim::Static(2), selected],
        _ => unreachable!(),
    };
    let y = graph.create_named_value("Y", DataType::String, output_shape);
    let attrs: Vec<_> = axis
        .into_iter()
        .map(|axis| ("axis", Attribute::Int(axis)))
        .collect();
    node(&mut graph, "Compress", &[x, condition], &[y], &attrs);
    graph.add_output(y);
    graph
}

#[test]
fn string_compress_loads_opsets_19_and_28_and_preserves_values() {
    let x = strings(&[2, 3], &["α", "a\0b", "猫", "", "é", "other"]);
    for opset in [19, 28] {
        for (axis, mask, dims, expected) in [
            (
                None,
                &[0, 1, 0, 1, 0, 0, 1][..],
                &[2][..],
                &["a\0b", ""][..],
            ),
            (
                Some(-1),
                &[1, 1][..],
                &[2, 2][..],
                &["α", "a\0b", "", "é"][..],
            ),
            (
                Some(0),
                &[0, 1, 1][..],
                &[1, 3][..],
                &["", "é", "other"][..],
            ),
            (None, &[][..], &[0][..], &[][..]),
            (Some(-1), &[0, 0, 0, 1][..], &[2, 0][..], &[][..]),
        ] {
            let graph = compress_graph(opset, axis, DataType::Bool, &[mask.len()]);
            let model = onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph))
                .unwrap();
            let mut session = InferenceSession::load_bytes(&model).unwrap();
            let mask = Tensor::from_raw(DataType::Bool, vec![mask.len()], mask).unwrap();
            let outputs = session.run(&[("X", &x), ("condition", &mask)]).unwrap();
            assert_strings(&outputs[0], dims, expected);
        }
    }
}

#[test]
fn string_compress_requires_a_rank_one_bool_condition() {
    let graph = compress_graph(19, None, DataType::Bool, &[1, 2]);
    let model =
        onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
    let error = InferenceSession::load_bytes(&model)
        .err()
        .expect("ONNX loading must reject a declared rank-two condition");
    let detail = error.to_string();
    assert!(
        detail.contains("Compress") && detail.contains("rank 1"),
        "{detail}"
    );

    let x = strings(&[2, 3], &["a", "b", "c", "d", "e", "f"]);
    // In-memory graphs may reach the kernel without the loader's shape inference.
    let rank_two = Tensor::from_raw(DataType::Bool, vec![1, 2], &[0, 1]).unwrap();
    let mut session = InferenceSession::from_graph(graph).unwrap();
    let detail = session
        .run(&[("X", &x), ("condition", &rank_two)])
        .unwrap_err()
        .to_string();
    assert!(
        detail.contains("Compress: condition must be a one-dimensional Bool tensor"),
        "{detail}"
    );

    let bad_condition = Tensor::from_i64(&[2], &[0, 1]).unwrap();
    let detail = match InferenceSession::from_graph(compress_graph(28, None, DataType::Int64, &[2]))
    {
        Ok(mut session) => session
            .run(&[("X", &x), ("condition", &bad_condition)])
            .unwrap_err()
            .to_string(),
        Err(error) => error.to_string(),
    };
    assert!(
        detail.contains("Compress") && detail.to_ascii_lowercase().contains("bool"),
        "{detail}"
    );
}

#[test]
fn string_compress_reads_a_bound_host_bool_condition() {
    let mut session =
        InferenceSession::from_graph(compress_graph(28, Some(-1), DataType::Bool, &[2])).unwrap();
    let mut mask = session
        .allocate_device_binding(
            "condition",
            None::<String>,
            DataType::Bool,
            vec![4],
            vec![2],
        )
        .unwrap();
    mask.write_bytes(0, &[0, 1, 1, 1]).unwrap();
    let x = strings(&[2, 3], &["α", "a\0b", "猫", "", "é", "other"]);
    let outputs = session
        .run_with_device_bindings(&[("X", &x)], std::slice::from_mut(&mut mask))
        .unwrap();
    assert_strings(outputs[0].as_ref().unwrap(), &[2, 1], &["a\0b", "é"]);
}

#[cfg(feature = "cuda")]
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn string_compress_requested_cuda_reports_host_placement() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    let mut session = InferenceSession::from_graph_with_provider(
        compress_graph(28, Some(-1), DataType::Bool, &[2]),
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        Path::new("."),
        Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA device 0")),
        None,
    )
    .unwrap();
    let report = session
        .execution_provider_fallback_report()
        .unwrap()
        .to_string();
    assert!(
        report.contains("Compress") && report.contains("no device representation"),
        "{report}"
    );
    let x = strings(&[2, 3], &["a", "猫", "c", "d", "e\0", "f"]);
    let mask = Tensor::from_raw(DataType::Bool, vec![2], &[0, 1]).unwrap();
    let outputs = session.run(&[("X", &x), ("condition", &mask)]).unwrap();
    assert_strings(&outputs[0], &[2, 1], &["猫", "e\0"]);
}

fn equal_graph(opset: u64, dtype: DataType) -> Graph {
    let mut graph = graph(opset);
    let x = graph.create_named_value("X", dtype, static_shape([3, 1]));
    let y = graph.create_named_value("Y", dtype, static_shape([4]));
    graph.add_input(x);
    graph.add_input(y);
    let z = graph.create_named_value("Z", DataType::Bool, static_shape([3, 4]));
    node(&mut graph, "Equal", &[x, y], &[z], &[]);
    graph.add_output(z);
    graph
}

#[test]
fn equal_string_broadcasts_exact_values_to_bool_at_opset_19_and_later() {
    let lhs = strings(&[3, 1], &["caf\u{e9}", "", "a\0b"]);
    let rhs = strings(&[4], &["cafe\u{301}", "", "a\0b", "caf\u{e9}"]);
    for opset in [19, 20] {
        let mut session = InferenceSession::from_graph(equal_graph(opset, DataType::String))
            .unwrap_or_else(|error| panic!("Equal at opset {opset}: {error}"));
        let output = session.run(&[("X", &lhs), ("Y", &rhs)]).unwrap();
        assert_eq!(output[0].dtype, DataType::Bool);
        assert_eq!(output[0].shape, [3, 4]);
        assert_eq!(output[0].as_bytes(), [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0]);
    }
}

#[test]
fn equal_string_requires_opset_19_without_changing_pre_19_numeric_equal() {
    let error = InferenceSession::from_graph(equal_graph(18, DataType::String))
        .err()
        .expect("opset 18 must not admit string Equal");
    let message = error.to_string();
    assert!(
        message.contains("Equal") && message.contains("opset 18"),
        "{message}"
    );
    assert!(message.contains("opset 19"), "{message}");

    for opset in [18, 19] {
        let mut session = InferenceSession::from_graph(equal_graph(opset, DataType::Float32))
            .unwrap_or_else(|error| panic!("Equal at opset {opset}: {error}"));
        let lhs = Tensor::from_f32(&[3, 1], &[0.0, 1.0, f32::NAN]).unwrap();
        let rhs = Tensor::from_f32(&[4], &[-0.0, 1.0, f32::NAN, 2.0]).unwrap();
        let output = session.run(&[("X", &lhs), ("Y", &rhs)]).unwrap();
        assert_eq!(output[0].dtype, DataType::Bool);
        assert_eq!(output[0].shape, [3, 4]);
        assert_eq!(output[0].as_bytes(), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]);
    }
}

fn regex_graph(opset: u64, dims: &[usize], pattern: &str) -> Graph {
    let mut graph = graph(opset);
    let x = input(&mut graph, "X", dims);
    let y = graph.create_named_value("Y", DataType::Bool, static_shape(dims.iter().copied()));
    node(
        &mut graph,
        "RegexFullMatch",
        &[x],
        &[y],
        &[("pattern", Attribute::String(pattern.as_bytes().to_vec()))],
    );
    graph.add_output(y);
    graph
}

#[test]
fn regex_full_match_dispatch_and_opset() {
    // Pinned ONNX test_regex_full_match_basic, plus prefix/suffix negatives.
    let pattern = r"www\.[\w.-]+\.\bcom\b";
    let values = strings(
        &[5],
        &[
            "www.google.com",
            "www.facebook.com",
            "www.bbc.co.uk",
            "xwww.google.com",
            "www.google.com\n",
        ],
    );
    let mut session = InferenceSession::from_graph(regex_graph(20, &[5], pattern)).unwrap();
    let output = session.run(&[("X", &values)]).unwrap();
    assert_eq!(output[0].dtype, DataType::Bool);
    assert_eq!(output[0].shape, [5]);
    assert_eq!(output[0].as_bytes(), [1, 1, 0, 0, 0]);
    let error = InferenceSession::from_graph(regex_graph(19, &[5], pattern))
        .err()
        .expect("opset 19 must not claim RegexFullMatch");
    assert!(error.to_string().contains("RegexFullMatch"), "{error}");
}

#[test]
fn text_chain_concats_then_splits_with_variable_width() {
    onnx_runtime_session::enable_activation_memory_plan_for_process();
    let mut graph = graph(20);
    let x = input(&mut graph, "X", &[2, 1]);
    let y = input(&mut graph, "Y", &[3]);
    let xy = graph.create_value(DataType::String, static_shape([2, 3]));
    let width = graph.intern_symbol("max_substrings");
    let y_shape = vec![Dim::Static(2), Dim::Static(3), Dim::Symbolic(width)];
    let parts = graph.create_named_value("substrings", DataType::String, y_shape.clone());
    let counts = graph.create_named_value("counts", DataType::Int64, static_shape([2, 3]));
    let matches = graph.create_named_value("matches", DataType::Bool, y_shape);
    node(&mut graph, "StringConcat", &[x, y], &[xy], &[]);
    node(
        &mut graph,
        "StringSplit",
        &[xy],
        &[parts, counts],
        &[
            ("delimiter", Attribute::String("-".as_bytes().to_vec())),
            ("maxsplit", Attribute::Int(2)),
        ],
    );
    node(
        &mut graph,
        "RegexFullMatch",
        &[parts],
        &[matches],
        &[("pattern", Attribute::String("α|猫".as_bytes().to_vec()))],
    );
    graph.add_output(parts);
    graph.add_output(counts);
    graph.add_output(matches);
    let mut session = InferenceSession::from_graph(graph).unwrap();
    let x = strings(&[2, 1], &["α-", ""]);
    let y = strings(&[3], &["β", "-x-", "a\0b"]);
    let output = session.run(&[("X", &x), ("Y", &y)]).unwrap();
    assert_strings(
        &output[0],
        &[2, 3, 3],
        &[
            "α", "β", "", "α", "", "x-", "α", "a\0b", "", "β", "", "", "", "x", "", "a\0b", "", "",
        ],
    );
    assert_eq!(output[1].shape, [2, 3]);
    assert_eq!(output[1].to_vec_i64(), [2, 3, 2, 1, 3, 1]);
    assert_eq!(output[2].shape, [2, 3, 3]);
    assert_eq!(
        output[2].as_bytes(),
        [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    let stats = session.activation_memory_plan_stats().unwrap();
    assert!(!stats.complete);
    assert!(stats.unknown_sizes > 0);

    let x = strings(&[2, 1], &["猫", ""]);
    let y = strings(&[3], &["a", "", "β"]);
    let output = session.run(&[("X", &x), ("Y", &y)]).unwrap();
    assert_strings(&output[0], &[2, 3, 1], &["猫a", "猫", "猫β", "a", "", "β"]);
    assert_eq!(output[1].to_vec_i64(), [1; 6]);
    assert_eq!(output[2].shape, [2, 3, 1]);
    assert_eq!(output[2].as_bytes(), [0, 1, 0, 0, 0, 0]);
}

#[test]
fn raw_sequence_storage_declines_managed_strings() {
    use onnx_runtime_session::sequence::{SeqTensor, SequenceError};

    assert!(matches!(
        SeqTensor::new(strings(&[], &["text"])),
        Err(SequenceError::UnsupportedDtype {
            dtype: DataType::String,
            ..
        })
    ));
}

#[test]
fn control_flow_declines_managed_string_outputs_without_panicking() {
    let mut branch = graph(20);
    let literal = branch.create_named_value("text", DataType::String, static_shape([]));
    branch.set_initializer(
        literal,
        WeightRef::Inline(TensorData {
            name: Some("text".into()),
            dtype: DataType::String,
            dims: vec![],
            data: Vec::new(),
            strings: vec![b"text".to_vec()],
        }),
    );
    let result = branch.create_named_value("result", DataType::String, static_shape([]));
    node(
        &mut branch,
        "StringConcat",
        &[literal, literal],
        &[result],
        &[],
    );
    branch.add_output(result);

    let mut model = graph(20);
    let condition = model.create_named_value("condition", DataType::Bool, static_shape([]));
    model.add_input(condition);
    let output = model.create_named_value("output", DataType::String, static_shape([]));
    let if_node = node(&mut model, "If", &[condition], &[output], &[]);
    model
        .subgraphs
        .insert((if_node, "then_branch".into()), branch.clone());
    model
        .subgraphs
        .insert((if_node, "else_branch".into()), branch);
    model.add_output(output);
    let condition = Tensor::from_raw(DataType::Bool, vec![], &[1]).unwrap();
    let error = match InferenceSession::from_graph(model) {
        Ok(mut session) => session
            .run(&[("condition", &condition)])
            .expect_err("raw control-flow signatures must reject string tensors"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("does not support tensor(string)"),
        "{error}"
    );
}

fn split_graph(dims: &[usize], delimiter: Option<&str>, maxsplit: Option<i64>) -> Graph {
    let mut graph = graph(20);
    let x = input(&mut graph, "X", dims);
    let width = graph.intern_symbol("max_substrings");
    let mut output_shape = static_shape(dims.iter().copied());
    output_shape.push(Dim::Symbolic(width));
    let parts = graph.create_named_value("substrings", DataType::String, output_shape);
    let counts = graph.create_named_value(
        "counts",
        DataType::Int64,
        static_shape(dims.iter().copied()),
    );
    let mut attrs = Vec::new();
    if let Some(delimiter) = delimiter {
        attrs.push((
            "delimiter",
            Attribute::String(delimiter.as_bytes().to_vec()),
        ));
    }
    if let Some(maxsplit) = maxsplit {
        attrs.push(("maxsplit", Attribute::Int(maxsplit)));
    }
    node(&mut graph, "StringSplit", &[x], &[parts, counts], &attrs);
    graph.add_output(parts);
    graph.add_output(counts);
    graph
}

#[test]
fn string_split_whitespace_scalar_and_empty_tensor() {
    let mut session = InferenceSession::from_graph(split_graph(&[], Some(""), Some(1))).unwrap();
    let value = strings(&[], &["\u{2003}é \t猫  "]);
    let output = session.run(&[("X", &value)]).unwrap();
    assert_strings(&output[0], &[2], &["é", "猫  "]);
    assert!(output[1].shape.is_empty());
    assert_eq!(output[1].to_vec_i64(), [2]);

    let mut session = InferenceSession::from_graph(split_graph(&[2, 0], None, None)).unwrap();
    let value = strings(&[2, 0], &[]);
    let output = session.run(&[("X", &value)]).unwrap();
    assert_strings(&output[0], &[2, 0, 0], &[]);
    assert_eq!(output[1].shape, [2, 0]);
    assert!(output[1].as_bytes().is_empty());
}

#[test]
fn text_split_reads_onnx_string_initializer_and_declines_raw_binding() {
    let mut graph = split_graph(&[2], Some("."), None);
    let input = graph.inputs.remove(0);
    graph.value_mut(input).is_graph_input = false;
    graph.set_initializer(
        input,
        WeightRef::Inline(TensorData {
            name: Some("X".into()),
            dtype: DataType::String,
            dims: vec![2],
            data: Vec::new(),
            strings: vec![b"abc.com".to_vec(), b"\0.net".to_vec()],
        }),
    );
    let bytes =
        onnx_runtime_loader::encode_model(&onnx_runtime_loader::Model::new(&graph)).unwrap();
    let mut session = InferenceSession::load_bytes(&bytes).unwrap();
    let output = session.run(&[]).unwrap();
    assert_strings(&output[0], &[2, 2], &["abc", "com", "\0", "net"]);
    assert_eq!(output[1].to_vec_i64(), [2, 2]);
    let error = session
        .allocate_device_output_binding("substrings", DataType::String, vec![2, 2], vec![2, 2])
        .expect_err("raw ABI must decline a string output");
    assert!(error.to_string().contains("tensor(string)"), "{error}");
}

#[cfg(feature = "cuda")]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
#[test]
fn requested_cuda_partitions_text_and_runs_connected_numeric_work() {
    use std::{path::Path, sync::Arc};

    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use onnx_runtime_loader::WeightStore;

    const N: usize = 64;
    let mut graph = graph(20);
    let text = input(&mut graph, "text", &[N, 1]);
    let joined = graph.create_named_value("joined", DataType::String, static_shape([N, 1]));
    graph.insert_node(Node::new(
        NodeId(0),
        "StringConcat",
        vec![Some(text), Some(text)],
        vec![joined],
    ));
    graph.add_output(joined);
    let matches = graph.create_named_value("matches", DataType::Bool, static_shape([N, 1]));
    let mut regex = Node::new(
        NodeId(0),
        "RegexFullMatch",
        vec![Some(joined)],
        vec![matches],
    );
    regex
        .attributes
        .insert("pattern".into(), Attribute::String(b"aa".to_vec()));
    graph.insert_node(regex);
    let mask = graph.create_named_value("mask", DataType::Float32, static_shape([N, 1]));
    let mut cast = Node::new(NodeId(0), "Cast", vec![Some(matches)], vec![mask]);
    cast.attributes
        .insert("to".into(), Attribute::Int(DataType::Float32 as i64));
    graph.insert_node(cast);
    let matrix = graph.create_named_value("matrix", DataType::Float32, static_shape([N, N]));
    graph.add_input(matrix);
    let masked = graph.create_named_value("masked", DataType::Float32, static_shape([N, N]));
    graph.insert_node(Node::new(
        NodeId(0),
        "Mul",
        vec![Some(mask), Some(matrix)],
        vec![masked],
    ));
    let scores = graph.create_named_value("scores", DataType::Float32, static_shape([N, N]));
    graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(masked), Some(matrix)],
        vec![scores],
    ));
    graph.add_output(scores);

    let mut cpu = InferenceSession::from_graph(graph.clone()).unwrap();
    let cuda = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA device 0"));
    let mut mixed = InferenceSession::from_graph_with_provider(
        graph,
        Arc::new(WeightStore::new()),
        Path::new("."),
        cuda.clone(),
        None,
    )
    .expect("build mixed CPU/CUDA text session without opting in");
    let report = mixed.heterogeneous_placement_report().unwrap();
    assert!(report.contains("3 node(s) on cuda_ep"), "{report}");
    assert!(report.contains("2 node(s) on cpu_ep"), "{report}");
    assert!(report.contains("1 cross-provider transfer(s)"), "{report}");
    assert!(mixed.execution_provider_fallback_report().is_none());
    let before = cuda.device_allocation_counts().unwrap().0;
    let text = strings(
        &[N, 1],
        &(0..N)
            .map(|i| if i % 2 == 0 { "a" } else { "b" })
            .collect::<Vec<_>>(),
    );
    let matrix = Tensor::from_f32(&[N, N], &vec![1.0; N * N]).unwrap();
    let inputs = &[("text", &text), ("matrix", &matrix)];
    let expected = cpu.run(inputs).unwrap();
    let actual = mixed.run(inputs).unwrap();
    assert_eq!(
        actual[0].as_strings().unwrap(),
        expected[0].as_strings().unwrap()
    );
    assert_eq!(actual[1].to_vec_f32(), expected[1].to_vec_f32());
    assert_eq!(
        &actual[1].to_vec_f32()[..2 * N],
        &[vec![N as f32; N], vec![0.0; N]].concat()
    );
    assert!(cuda.device_allocation_counts().unwrap().0 > before);
}

#[cfg(feature = "cuda")]
#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn requested_cuda_uses_reported_cpu_placement_for_text() {
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::{path::Path, sync::Arc};

    let cuda = Arc::new(CudaExecutionProvider::initialized(0).expect("initialize CUDA device 0"));
    let mut session = InferenceSession::from_graph_with_provider(
        split_graph(&[2], Some("."), None),
        Arc::new(onnx_runtime_loader::WeightStore::new()),
        Path::new("."),
        cuda,
        None,
    )
    .unwrap();
    let report = session
        .execution_provider_fallback_report()
        .expect("explicit host placement report");
    assert!(report.to_string().contains("StringSplit"));
    assert!(report.to_string().contains("no device representation"));
    let input = strings(&[2], &["abc.com", "def.net"]);
    let output = session.run(&[("X", &input)]).unwrap();
    assert_strings(&output[0], &[2, 2], &["abc", "com", "def", "net"]);
    assert_eq!(output[1].to_vec_i64(), [2, 2]);
}

#[test]
#[ignore = "release-mode benchmark; hold the shared host lock"]
fn text_ops_session_throughput() {
    use std::{hint::black_box, time::Instant};

    const BATCH: usize = 4096;
    const REPEATS: usize = 128;
    let rows = Tensor::from_strings(
        vec![BATCH],
        (0..BATCH)
            .map(|i| {
                if i % 4 == 0 {
                    format!("id:{i:06} status:error sample unexpected response")
                } else {
                    format!("id:{i:06} status:ok sample timed request")
                }
            })
            .collect(),
    )
    .unwrap();
    let suffix = strings(&[], &[" /api"]);
    let mut concat = graph(20);
    let x = input(&mut concat, "X", &[BATCH]);
    let y = input(&mut concat, "Y", &[]);
    let z = concat.create_named_value("Z", DataType::String, static_shape([BATCH]));
    node(&mut concat, "StringConcat", &[x, y], &[z], &[]);
    concat.add_output(z);

    let workloads = [
        (
            "RegexFullMatch",
            regex_graph(20, &[BATCH], r"id:[0-9]{6} status:ok sample .+"),
            vec![("X", &rows)],
        ),
        ("StringConcat", concat, vec![("X", &rows), ("Y", &suffix)]),
        (
            "StringSplit",
            split_graph(&[BATCH], None, None),
            vec![("X", &rows)],
        ),
    ];

    #[cfg(feature = "cuda")]
    let cuda = std::sync::Arc::new(
        onnx_runtime_ep_cuda::CudaExecutionProvider::initialized(0)
            .expect("initialize CUDA device 0"),
    );
    for (op, graph, inputs) in workloads {
        let mut routes = vec![("cpu", InferenceSession::from_graph(graph.clone()).unwrap())];
        #[cfg(not(feature = "cuda"))]
        let _ = &mut routes;
        #[cfg(feature = "cuda")]
        {
            let session = InferenceSession::from_graph_with_provider(
                graph,
                std::sync::Arc::new(onnx_runtime_loader::WeightStore::new()),
                std::path::Path::new("."),
                cuda.clone(),
                None,
            )
            .unwrap();
            assert!(session.execution_provider_fallback_report().is_some());
            routes.push(("cuda-requested/cpu", session));
        }
        #[cfg(not(feature = "cuda"))]
        drop(graph);
        for (_, session) in &mut routes {
            for _ in 0..16 {
                let result = session.run(&inputs).unwrap();
                assert_eq!(result[0].shape[0], BATCH);
                assert!(
                    result
                        .iter()
                        .all(|tensor| tensor.device().is_host_accessible())
                );
            }
        }
        let mut samples = vec![Vec::new(); routes.len()];
        for i in 0..7 {
            for j in 0..routes.len() {
                let index = (i + j) % routes.len();
                let start = Instant::now();
                for _ in 0..REPEATS {
                    black_box(routes[index].1.run(black_box(&inputs)).unwrap());
                }
                samples[index].push((REPEATS * BATCH) as f64 / start.elapsed().as_secs_f64());
            }
        }
        for ((route, _), runs) in routes.into_iter().zip(&mut samples) {
            runs.sort_by(f64::total_cmp);
            eprintln!(
                "{op:<14} {route:<18} {BATCH} rows/batch, {:.2} M rows/s median ({:.2}–{:.2}, n={})",
                runs[runs.len() / 2] / 1e6,
                runs[0] / 1e6,
                runs[runs.len() - 1] / 1e6,
                runs.len()
            );
        }
    }
}
