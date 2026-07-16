// Ported from onnx/onnx onnx/test/parser_test.py and printer_test.py.

use onnx_rs::ir::{Attribute, DataType, Dim, Graph, Node, NodeId};
use onnx_rs::{Json, Model, ModelMetadata, Text, TextCodec, TextProto, from_text, to_text};

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
