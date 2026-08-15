// Ported from onnx/onnx onnx/test/parser_test.py and printer_test.py.

use onnx_std::ir::{Attribute, DataType, Dim, Graph, Node, NodeId};
use onnx_std::{Json, Model, ModelMetadata, Text, TextCodec, TextProto, from_text, to_text};
use prost::Message;

const BASIC_MODEL: &str = r#"
<
  ir_version: 7,
  opset_import: [ "" : 10, "com.microsoft": 1]
>
agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N] C)
{
  T = MatMul(X, W)
  S = Add(T, B)
  C = Softmax(S)
}
"#;

const ATTRIBUTE_MODEL: &str = r#"
<
  ir_version: 10,
  opset_import: ["" : 21, "test.domain" : 1]
>
attributes (float[2] X) => (float[2] Y) {
  Y = test.domain.Decorated <alpha = 0.25, axes = [1, -2], label = "hello", labels = ["red", "blue"], mode = 7, scales = [0.5, 2.0], value = <tensor int64[[2]]>>(X)
}
"#;

const INITIALIZER_MODEL: &str = r#"
<
  ir_version: 10,
  opset_import: ["" : 21]
>
initialized (float[2] X) => (float[2] Y) {
  // initializers
  // float[2] W = <inline data omitted>
  Y = Add(X, W)
}
"#;

fn assert_print_stable(source: &str) -> Model {
    let first = from_text(source).expect("parse upstream-derived model");
    let printed = to_text(&first);
    let second = from_text(&printed).expect("reparse printed model");
    assert_eq!(to_text(&second), printed);
    second
}

#[test]
fn parses_upstream_basic_model_golden() {
    let model = from_text(BASIC_MODEL).expect("parse ONNX parser_test.py model");
    assert_eq!(model.metadata.ir_version, 7);
    assert_eq!(model.metadata.graph_name, "agraph");
    assert_eq!(model.graph.opset_imports[""], 10);
    assert_eq!(model.graph.opset_imports["com.microsoft"], 1);

    let ops: Vec<_> = model
        .graph
        .nodes
        .values()
        .map(|node| node.op_type.as_str())
        .collect();
    assert_eq!(ops, ["MatMul", "Add", "Softmax"]);
    assert_eq!(model.graph.inputs.len(), 3);
    assert_eq!(model.graph.outputs.len(), 1);
}

#[test]
fn rejects_upstream_missing_opset_separator() {
    let malformed = BASIC_MODEL.replace(
        r#""" : 10, "com.microsoft": 1"#,
        r#""" : 10   "com.microsoft": 1"#,
    );
    assert!(from_text(&malformed).is_err());
}

#[test]
fn quoted_symbolic_dimension_is_stable() {
    let source = r#"
<
  ir_version: 10,
  opset_import: ["" : 21]
>
agraph (float["M + N"] x) => (float["M + N"] y) {
  y = Identity(x)
}
"#;
    let first = from_text(source).expect("parse quoted symbolic dimension");
    let printed = to_text(&first);
    assert!(printed.contains(r#""M + N""#), "{printed}");
    let second = from_text(&printed).expect("reparse printed model");
    assert_eq!(to_text(&second), printed);
}

#[test]
fn printer_matches_upstream_style_golden() {
    let mut graph = Graph::new();
    graph.opset_imports.insert(String::new(), 10);
    graph.opset_imports.insert("com.microsoft".into(), 1);
    let n = graph.intern_symbol("N");
    let x = graph.create_named_value("X", DataType::Float32, vec![Dim::Symbolic(n), 128.into()]);
    let w = graph.create_named_value("W", DataType::Float32, vec![128.into(), 10.into()]);
    let b = graph.create_named_value("B", DataType::Float32, vec![10.into()]);
    let t = graph.create_named_value("T", DataType::Float32, vec![Dim::Symbolic(n), 10.into()]);
    let s = graph.create_named_value("S", DataType::Float32, vec![Dim::Symbolic(n), 10.into()]);
    let c = graph.create_named_value("C", DataType::Float32, vec![Dim::Symbolic(n)]);
    graph.add_input(x);
    graph.add_input(w);
    graph.add_input(b);
    graph.insert_node(Node::new(
        NodeId(0),
        "MatMul",
        vec![Some(x), Some(w)],
        vec![t],
    ));
    graph.insert_node(Node::new(NodeId(0), "Add", vec![Some(t), Some(b)], vec![s]));
    graph.insert_node(Node::new(NodeId(0), "Softmax", vec![Some(s)], vec![c]));
    graph.add_output(c);
    let metadata = ModelMetadata {
        ir_version: 7,
        graph_name: "agraph".into(),
        ..ModelMetadata::default()
    };

    let expected = r#"<
  ir_version: 7,
  opset_import: ["" : 10, "com.microsoft" : 1]
>
agraph (float[N, 128] X, float[128, 10] W, float[10] B) => (float[N] C) {
  T = MatMul(X, W)
  S = Add(T, B)
  C = Softmax(S)
}
"#;
    assert_eq!(to_text(&Model::with_metadata(graph, metadata)), expected);
}

#[test]
fn parses_upstream_custom_domain_attributes() {
    let source = r#"
<
  ir_version: 9,
  opset_import: ["" : 15, "custom_domain" : 1]
>
agraph (float[N] x) => (float[N] out) {
  out = custom_domain.Selu<alpha=2.0, gamma=3.0>(x)
}
"#;
    let model = from_text(source).expect("parse custom-domain attribute model");
    let node = model.graph.nodes.values().next().expect("Selu node");
    assert_eq!(node.domain, "custom_domain");
    assert_eq!(node.attr("alpha").and_then(Attribute::as_float), Some(2.0));
    assert_eq!(node.attr("gamma").and_then(Attribute::as_float), Some(3.0));
}

#[test]
fn parses_upstream_node_domain_outputs_and_int_attribute() {
    // Port of parser_test.py::test_parse_node, wrapped in a model because
    // onnx-std intentionally exposes model-level text parsing.
    let source = r#"
<
  ir_version: 10,
  opset_import: ["" : 21, "SomeDomain" : 1]
>
node_test (float[2] in1, float[2] in2) => (float[2] out1, float[2] out2) {
  out1, out2 = SomeDomain.SomeOp <attr1 = 1>(in1, in2)
}
"#;
    let model = assert_print_stable(source);
    let node = model.graph.nodes.values().next().expect("SomeOp node");

    assert_eq!(node.domain, "SomeDomain");
    assert_eq!(node.op_type, "SomeOp");
    assert_eq!(node.inputs.len(), 2);
    assert_eq!(node.outputs.len(), 2);
    assert_eq!(node.attr("attr1").and_then(Attribute::as_int), Some(1));
}

#[test]
fn parses_upstream_attribute_value_kinds() {
    let model = assert_print_stable(ATTRIBUTE_MODEL);
    let node = model.graph.nodes.values().next().expect("Decorated node");

    assert_eq!(node.attr("mode").and_then(Attribute::as_int), Some(7));
    assert_eq!(node.attr("alpha").and_then(Attribute::as_float), Some(0.25));
    assert_eq!(
        node.attr("label").and_then(Attribute::as_str),
        Some("hello")
    );
    assert_eq!(
        node.attr("axes").and_then(Attribute::as_ints),
        Some([1, -2].as_slice())
    );
    assert!(
        matches!(node.attr("scales"), Some(Attribute::Floats(values)) if values == &[0.5, 2.0])
    );
    assert!(matches!(
        node.attr("labels"),
        Some(Attribute::Strings(values))
            if values == &[b"red".to_vec(), b"blue".to_vec()]
    ));
    assert!(matches!(
        node.attr("value"),
        Some(Attribute::Tensor(tensor))
            if tensor.dtype == DataType::Int64 && tensor.dims == [2]
    ));
}

#[test]
fn round_trips_initializer_reference() {
    let model = assert_print_stable(INITIALIZER_MODEL);
    let initializer = model
        .graph
        .initializers
        .values()
        .next()
        .expect("W initializer");

    assert_eq!(initializer.dtype(), DataType::Float32);
    assert_eq!(initializer.dims(), &[2]);
    assert_eq!(model.graph.num_nodes(), 1);
}

#[test]
fn parses_supported_upstream_tensor_type_spellings() {
    // Port of parser_test.py::test_parse_graph_types for every concrete tensor
    // element type in the bound ONNX proto.
    let cases = [
        ("bfloat16", DataType::BFloat16),
        ("bool", DataType::Bool),
        ("float64", DataType::Float64),
        ("float16", DataType::Float16),
        ("complex64", DataType::Complex64),
        ("complex128", DataType::Complex128),
        ("float", DataType::Float32),
        ("float8e4m3fn", DataType::Float8E4M3FN),
        ("float8e4m3fnuz", DataType::Float8E4M3FNUZ),
        ("float8e5m2", DataType::Float8E5M2),
        ("float8e5m2fnuz", DataType::Float8E5M2FNUZ),
        ("float8e8m0", DataType::Float8E8M0),
        ("int2", DataType::Int2),
        ("int4", DataType::Int4),
        ("int8", DataType::Int8),
        ("int16", DataType::Int16),
        ("int32", DataType::Int32),
        ("int64", DataType::Int64),
        ("string", DataType::String),
        ("uint4", DataType::Uint4),
        ("uint2", DataType::Uint2),
        ("uint8", DataType::Uint8),
        ("uint16", DataType::Uint16),
        ("uint32", DataType::Uint32),
        ("uint64", DataType::Uint64),
        ("float4e2m1", DataType::Float4E2M1),
    ];

    for (spelling, expected) in cases {
        let source = format!(
            r#"
<
  ir_version: 10,
  opset_import: ["" : 21]
>
type_test (float[1] X) => ({spelling}[1] C) {{
  C = Cast <to = 1>(X)
}}
"#
        );
        let model = assert_print_stable(&source);
        assert_eq!(
            model.graph.value(model.graph.outputs[0]).dtype,
            expected,
            "dtype spelling {spelling}"
        );
    }
}

#[test]
fn round_trips_several_upstream_models_stably() {
    for source in [BASIC_MODEL, ATTRIBUTE_MODEL, INITIALIZER_MODEL] {
        assert_print_stable(source);
    }
}

#[test]
fn json_and_textproto_round_trip_attribute_model() {
    let model = from_text(ATTRIBUTE_MODEL).expect("parse attribute fixture");
    let expected = model.to_proto().unwrap().encode_to_vec();

    let json = Json::serialize(&model, &()).expect("serialize JSON");
    let from_json = Json::deserialize(&json).expect("deserialize JSON");
    assert_eq!(from_json.to_proto().unwrap().encode_to_vec(), expected);

    let textproto = TextProto::serialize(&model, &()).expect("serialize TextProto");
    let from_textproto = TextProto::deserialize(&textproto).expect("deserialize TextProto");
    assert_eq!(from_textproto.to_proto().unwrap().encode_to_vec(), expected);
}

#[test]
fn parses_supported_upstream_special_float_literals() {
    // Upstream covers additional aliases; onnx-std currently supports the
    // canonical spellings emitted by its own printer.
    for (literal, expected) in [
        ("inf", f32::INFINITY),
        ("-inf", f32::NEG_INFINITY),
        ("NaN", f32::NAN),
    ] {
        let source = format!(
            r#"
<
  ir_version: 10,
  opset_import: ["" : 21]
>
float_test () => (float[] Y) {{
  Y = Constant <value_float = {literal}>()
}}
"#
        );
        let model = assert_print_stable(&source);
        let value = model
            .graph
            .nodes
            .values()
            .next()
            .and_then(|node| node.attr("value_float"))
            .and_then(Attribute::as_float)
            .expect("float attribute");
        if expected.is_nan() {
            assert!(value.is_nan());
        } else {
            assert_eq!(value, expected);
        }
    }
}

#[test]
fn rejects_upstream_bracketed_node_invocation() {
    // Direct port of parser_test.py::test_parse_graph_error.
    let malformed = BASIC_MODEL.replace("T = MatMul(X, W)", "T = MatMul[X, W]");
    assert!(from_text(&malformed).is_err());
}

#[test]
fn rejects_unclosed_attribute_block() {
    let malformed = ATTRIBUTE_MODEL.replace(
        "value = <tensor int64[[2]]>>(X)",
        "value = <tensor int64[[2]]>(X)",
    );
    assert!(from_text(&malformed).is_err());
}

#[test]
fn accepts_complex_tensor_types_from_upstream() {
    for spelling in ["complex64", "complex128"] {
        let source = BASIC_MODEL.replace("float[N, 128] X", &format!("{spelling}[N, 128] X"));
        assert!(from_text(&source).is_ok(), "{spelling}");
    }
}

#[test]
fn unified_codecs_share_one_trait_shape() {
    let model = from_text(BASIC_MODEL).expect("parse fixture");

    let text = Text::serialize(&model, &Default::default()).expect("serialize text");
    assert_eq!(Text::deserialize(&text).unwrap().graph.num_nodes(), 3);

    let json = Json::serialize(&model, &()).expect("serialize JSON");
    assert_eq!(Json::deserialize(&json).unwrap().graph.num_nodes(), 3);

    let textproto = TextProto::serialize(&model, &()).expect("serialize TextProto");
    assert_eq!(
        TextProto::deserialize(&textproto)
            .unwrap()
            .graph
            .num_nodes(),
        3
    );
}
