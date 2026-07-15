//! Tests for model-local function inlining (`function_inline::inline_functions`).
//!
//! Each test hand-builds a `ModelProto` from prost structs and asserts the
//! rewritten proto contains no calls to any declared function, that value names
//! are remapped/freshened correctly, that attributes bind (call-site, default,
//! required-missing), and that recursion is rejected rather than looped.

use onnx_runtime_loader::function_inline::inline_functions;
use onnx_runtime_loader::proto::onnx;
use onnx_runtime_loader::LoaderError;

// --- proto construction helpers --------------------------------------------

fn node(op: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    onnx::NodeProto {
        op_type: op.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        ..Default::default()
    }
}

/// A node that calls a model-local function `(domain, name)`.
fn call(name: &str, domain: &str, inputs: &[&str], outputs: &[&str]) -> onnx::NodeProto {
    let mut n = node(name, inputs, outputs);
    n.domain = domain.to_string();
    n
}

fn float_attr(name: &str, v: f32) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Float as i32,
        f: v,
        ..Default::default()
    }
}

/// A body-node attribute that references a function formal attribute `A`.
fn ref_attr(name: &str, ref_attr_name: &str) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        ref_attr_name: ref_attr_name.to_string(),
        ..Default::default()
    }
}

fn node_with_attrs(
    op: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<onnx::AttributeProto>,
) -> onnx::NodeProto {
    let mut n = node(op, inputs, outputs);
    n.attribute = attrs;
    n
}

fn function(
    name: &str,
    domain: &str,
    inputs: &[&str],
    outputs: &[&str],
    nodes: Vec<onnx::NodeProto>,
) -> onnx::FunctionProto {
    onnx::FunctionProto {
        name: name.to_string(),
        domain: domain.to_string(),
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        node: nodes,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        ..Default::default()
    }
}

fn model(graph: onnx::GraphProto, functions: Vec<onnx::FunctionProto>) -> onnx::ModelProto {
    onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 17,
        }],
        graph: Some(graph),
        functions,
        ..Default::default()
    }
}

/// Every `(domain, op_type)` present in the graph's node list.
fn op_types(gp: &onnx::GraphProto) -> Vec<(String, String)> {
    gp.node
        .iter()
        .map(|n| (n.domain.clone(), n.op_type.clone()))
        .collect()
}

/// All non-empty value names referenced by the graph's nodes.
fn value_names(gp: &onnx::GraphProto) -> std::collections::HashSet<String> {
    let mut s = std::collections::HashSet::new();
    for n in &gp.node {
        for name in n.input.iter().chain(n.output.iter()) {
            if !name.is_empty() {
                s.insert(name.clone());
            }
        }
    }
    s
}

/// Inline, returning an owned `ModelProto` (the returned `Cow` borrows `m`, so
/// tests take ownership to keep the result alive independently).
fn inline(m: onnx::ModelProto) -> onnx::ModelProto {
    inline_functions(&m).unwrap().into_owned()
}

fn inline_err(m: onnx::ModelProto) -> LoaderError {
    inline_functions(&m).unwrap_err()
}

// --- tests -----------------------------------------------------------------

#[test]
fn no_functions_is_a_noop() {
    let graph = onnx::GraphProto {
        node: vec![node("Add", &["a", "b"], &["c"])],
        ..Default::default()
    };
    let m = model(graph, vec![]);
    let out = inline_functions(&m).unwrap();
    // Borrowed, unchanged.
    assert!(matches!(out, std::borrow::Cow::Borrowed(_)));
    assert_eq!(out.functions.len(), 0);
    assert_eq!(op_types(out.graph.as_ref().unwrap()).len(), 1);
}

#[test]
fn simple_function_is_expanded_to_primitives() {
    // MyLinear(X, W, B) = Add(MatMul(X, W), B)
    let body = vec![
        node("MatMul", &["X", "W"], &["xw"]),
        node("Add", &["xw", "B"], &["Y"]),
    ];
    let f = function("MyLinear", "custom.domain", &["X", "W", "B"], &["Y"], body);
    let graph = onnx::GraphProto {
        node: vec![call(
            "MyLinear",
            "custom.domain",
            &["input", "weight", "bias"],
            &["out"],
        )],
        ..Default::default()
    };
    let m = model(graph, vec![f]);
    let out = inline_functions(&m).unwrap();
    let g = out.graph.as_ref().unwrap();

    // No call to the function survives; only primitive ops remain.
    assert_eq!(
        op_types(g),
        vec![
            (String::new(), "MatMul".to_string()),
            (String::new(), "Add".to_string()),
        ]
    );
    // Functions cleared.
    assert!(out.functions.is_empty());

    // Formal I/O rebound to the call-site actuals.
    let names = value_names(g);
    assert!(names.contains("input"));
    assert!(names.contains("weight"));
    assert!(names.contains("bias"));
    assert!(names.contains("out"));
    // The MatMul feeds the Add via an internal (freshened) value.
    let matmul = &g.node[0];
    let add = &g.node[1];
    assert_eq!(matmul.output, vec![add.input[0].clone()]);
    assert!(matmul.output[0].starts_with("__fn"));
}

#[test]
fn attribute_binding_uses_call_site_value() {
    // Scale(X){alpha} = Mul(X, X) but the Mul's 'ignored' attr refs formal alpha.
    let body = vec![node_with_attrs(
        "Mul",
        &["X", "X"],
        &["Y"],
        vec![ref_attr("literal_name", "alpha")],
    )];
    let f = function("Scale", "custom.domain", &["X"], &["Y"], body);
    let mut call_node = call("Scale", "custom.domain", &["x"], &["y"]);
    call_node.attribute = vec![float_attr("alpha", 2.5)];
    let graph = onnx::GraphProto {
        node: vec![call_node],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    let mul = &g.node[0];
    assert_eq!(mul.attribute.len(), 1);
    let a = &mul.attribute[0];
    assert_eq!(a.name, "literal_name"); // body attribute name preserved
    assert!(a.ref_attr_name.is_empty()); // ref cleared
    assert_eq!(a.f, 2.5); // call-site value substituted
}

#[test]
fn attribute_binding_falls_back_to_default() {
    let body = vec![node_with_attrs(
        "Mul",
        &["X", "X"],
        &["Y"],
        vec![ref_attr("literal_name", "alpha")],
    )];
    let mut f = function("Scale", "custom.domain", &["X"], &["Y"], body);
    // alpha has a default (not in the required `attribute` list).
    f.attribute_proto = vec![float_attr("alpha", 9.0)];
    // Call omits alpha.
    let graph = onnx::GraphProto {
        node: vec![call("Scale", "custom.domain", &["x"], &["y"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    assert_eq!(g.node[0].attribute[0].f, 9.0);
}

#[test]
fn missing_required_attribute_is_an_error() {
    let body = vec![node_with_attrs(
        "Mul",
        &["X", "X"],
        &["Y"],
        vec![ref_attr("literal_name", "alpha")],
    )];
    let mut f = function("Scale", "custom.domain", &["X"], &["Y"], body);
    // alpha is required (declared, no default).
    f.attribute = vec!["alpha".to_string()];
    let graph = onnx::GraphProto {
        node: vec![call("Scale", "custom.domain", &["x"], &["y"])],
        ..Default::default()
    };
    let err = inline_err(model(graph, vec![f]));
    assert!(matches!(
        &err,
        LoaderError::MissingRequiredFunctionAttribute { attribute, .. } if attribute == "alpha"
    ));
}

#[test]
fn optional_ref_attribute_without_value_is_dropped() {
    let body = vec![node_with_attrs(
        "Relu",
        &["X"],
        &["Y"],
        vec![ref_attr("maybe", "alpha")], // optional, no default, not supplied
    )];
    let f = function("F", "custom.domain", &["X"], &["Y"], body);
    let graph = onnx::GraphProto {
        node: vec![call("F", "custom.domain", &["x"], &["y"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    // The unresolved optional attribute is omitted.
    assert!(out.graph.as_ref().unwrap().node[0].attribute.is_empty());
}

#[test]
fn nested_functions_expand_fully() {
    // B(X) = Relu(X); A(X) = B(Add(X, X))
    let b = function(
        "B",
        "custom.domain",
        &["X"],
        &["Y"],
        vec![node("Relu", &["X"], &["Y"])],
    );
    let a = function(
        "A",
        "custom.domain",
        &["X"],
        &["Y"],
        vec![
            node("Add", &["X", "X"], &["t"]),
            call("B", "custom.domain", &["t"], &["Y"]),
        ],
    );
    let graph = onnx::GraphProto {
        node: vec![call("A", "custom.domain", &["x"], &["y"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![a, b]));
    let g = out.graph.as_ref().unwrap();
    assert_eq!(
        op_types(g),
        vec![
            (String::new(), "Add".to_string()),
            (String::new(), "Relu".to_string()),
        ]
    );
    // Fully primitive; no residual function domain nodes.
    assert!(g.node.iter().all(|n| n.domain.is_empty()));
}

#[test]
fn direct_recursion_is_rejected() {
    // A(X) = A(X)  — must error, not loop.
    let a = function(
        "A",
        "custom.domain",
        &["X"],
        &["Y"],
        vec![call("A", "custom.domain", &["X"], &["Y"])],
    );
    let graph = onnx::GraphProto {
        node: vec![call("A", "custom.domain", &["x"], &["y"])],
        ..Default::default()
    };
    let err = inline_err(model(graph, vec![a]));
    assert!(matches!(&err, LoaderError::RecursiveFunction { .. }));
}

#[test]
fn mutual_recursion_is_rejected() {
    // A -> B -> A
    let a = function(
        "A",
        "d",
        &["X"],
        &["Y"],
        vec![call("B", "d", &["X"], &["Y"])],
    );
    let b = function(
        "B",
        "d",
        &["X"],
        &["Y"],
        vec![call("A", "d", &["X"], &["Y"])],
    );
    let graph = onnx::GraphProto {
        node: vec![call("A", "d", &["x"], &["y"])],
        ..Default::default()
    };
    let err = inline_err(model(graph, vec![a, b]));
    assert!(matches!(&err, LoaderError::RecursiveFunction { .. }));
}

#[test]
fn two_calls_do_not_collide_internal_names() {
    // MyLinear(X,W,B) = Add(MatMul(X,W), B); call it twice.
    let body = vec![
        node("MatMul", &["X", "W"], &["xw"]),
        node("Add", &["xw", "B"], &["Y"]),
    ];
    let f = function("MyLinear", "d", &["X", "W", "B"], &["Y"], body);
    let graph = onnx::GraphProto {
        node: vec![
            call("MyLinear", "d", &["x", "w1", "b1"], &["h"]),
            call("MyLinear", "d", &["h", "w2", "b2"], &["out"]),
        ],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    // 4 primitive nodes total.
    assert_eq!(g.node.len(), 4);
    // The internal MatMul->Add wire of the two instantiations must differ.
    let wire0 = &g.node[0].output[0];
    let wire1 = &g.node[2].output[0];
    assert_ne!(wire0, wire1, "internal names collided across instantiations");
    // Node names are unique.
    let mut names: Vec<_> = g.node.iter().map(|n| n.name.clone()).collect();
    names.sort();
    let uniq = names.iter().collect::<std::collections::HashSet<_>>();
    assert_eq!(uniq.len(), names.len(), "duplicate node names emitted");
}

#[test]
fn empty_optional_input_is_passed_through() {
    // F(X, opt) = Add(X, opt); call passes "" for the optional second input.
    let body = vec![node("Add", &["X", "opt"], &["Y"])];
    let f = function("F", "d", &["X", "opt"], &["Y"], body);
    let graph = onnx::GraphProto {
        node: vec![call("F", "d", &["x", ""], &["y"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    // The "" formal maps to "" (absent), never freshened.
    assert_eq!(g.node[0].input, vec!["x".to_string(), String::new()]);
}

#[test]
fn arity_mismatch_too_many_inputs_is_rejected() {
    let f = function(
        "F",
        "d",
        &["X"],
        &["Y"],
        vec![node("Relu", &["X"], &["Y"])],
    );
    let graph = onnx::GraphProto {
        node: vec![call("F", "d", &["a", "b"], &["y"])], // 2 > 1
        ..Default::default()
    };
    let err = inline_err(model(graph, vec![f]));
    assert!(matches!(
        &err,
        LoaderError::FunctionArityMismatch { kind, formal: 1, actual: 2, .. } if *kind == "input"
    ));
}

#[test]
fn function_called_inside_control_flow_subgraph_is_inlined() {
    // An If node whose then/else branches each call a function.
    let f = function(
        "F",
        "d",
        &["X"],
        &["Y"],
        vec![node("Relu", &["X"], &["Y"])],
    );

    let then_branch = onnx::GraphProto {
        node: vec![call("F", "d", &["v"], &["tb_out"])],
        output: vec![onnx::ValueInfoProto {
            name: "tb_out".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let else_branch = onnx::GraphProto {
        node: vec![call("F", "d", &["v"], &["eb_out"])],
        output: vec![onnx::ValueInfoProto {
            name: "eb_out".to_string(),
            ..Default::default()
        }],
        ..Default::default()
    };
    let if_node = node_with_attrs(
        "If",
        &["cond"],
        &["r"],
        vec![
            onnx::AttributeProto {
                name: "then_branch".to_string(),
                r#type: onnx::attribute_proto::AttributeType::Graph as i32,
                g: Some(then_branch),
                ..Default::default()
            },
            onnx::AttributeProto {
                name: "else_branch".to_string(),
                r#type: onnx::attribute_proto::AttributeType::Graph as i32,
                g: Some(else_branch),
                ..Default::default()
            },
        ],
    );
    let graph = onnx::GraphProto {
        node: vec![if_node],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    // The If node is preserved...
    assert_eq!(g.node.len(), 1);
    assert_eq!(g.node[0].op_type, "If");
    // ...but both branches now contain only the primitive Relu.
    for attr in &g.node[0].attribute {
        let sub = attr.g.as_ref().unwrap();
        assert_eq!(sub.node.len(), 1);
        assert_eq!(sub.node[0].op_type, "Relu");
        assert!(sub.node[0].domain.is_empty());
    }
}

#[test]
fn overload_disambiguates_matching_names() {
    // Two functions same (domain,name), different overload.
    let mut f_add = function(
        "F",
        "d",
        &["X"],
        &["Y"],
        vec![node("Identity", &["X"], &["Y"])],
    );
    f_add.overload = "add".to_string();
    let mut f_neg = function(
        "F",
        "d",
        &["X"],
        &["Y"],
        vec![node("Neg", &["X"], &["Y"])],
    );
    f_neg.overload = "neg".to_string();

    let mut c = call("F", "d", &["x"], &["y"]);
    c.overload = "neg".to_string();
    let graph = onnx::GraphProto {
        node: vec![c],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f_add, f_neg]));
    let g = out.graph.as_ref().unwrap();
    assert_eq!(g.node.len(), 1);
    assert_eq!(g.node[0].op_type, "Neg");
}

#[test]
fn function_opset_import_is_merged_into_model() {
    let mut f = function(
        "F",
        "d",
        &["X"],
        &["Y"],
        vec![node("Relu", &["X"], &["Y"])],
    );
    // Function relies on a domain the model does not declare.
    f.opset_import = vec![onnx::OperatorSetIdProto {
        domain: "com.microsoft".to_string(),
        version: 1,
    }];
    let graph = onnx::GraphProto {
        node: vec![call("F", "d", &["x"], &["y"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    assert!(out
        .opset_import
        .iter()
        .any(|o| o.domain == "com.microsoft" && o.version == 1));
}
