//! Tests for model-local function inlining (`function_inline::inline_functions`).
//!
//! Each test hand-builds a `ModelProto` from prost structs and asserts the
//! rewritten proto contains no calls to any declared function, that value names
//! are remapped/freshened correctly, that attributes bind (call-site, default,
//! required-missing), and that recursion is rejected rather than looped.

use onnx_runtime_loader::LoaderError;
use onnx_runtime_loader::function_inline::inline_functions;
use onnx_runtime_loader::proto::onnx;

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

/// A `ValueInfoProto` carrying only a name (used for graph inputs/outputs).
fn value_info(name: &str) -> onnx::ValueInfoProto {
    onnx::ValueInfoProto {
        name: name.to_string(),
        ..Default::default()
    }
}

/// A graph-valued attribute (e.g. an `If`/`Loop` branch/body).
fn graph_attr(name: &str, g: onnx::GraphProto) -> onnx::AttributeProto {
    onnx::AttributeProto {
        name: name.to_string(),
        r#type: onnx::attribute_proto::AttributeType::Graph as i32,
        g: Some(g),
        ..Default::default()
    }
}

/// A named initializer tensor (contents irrelevant to name-scoping tests).
fn tensor(name: &str) -> onnx::TensorProto {
    onnx::TensorProto {
        name: name.to_string(),
        ..Default::default()
    }
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
    assert_ne!(
        wire0, wire1,
        "internal names collided across instantiations"
    );
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
    let f = function("F", "d", &["X"], &["Y"], vec![node("Relu", &["X"], &["Y"])]);
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
    let f = function("F", "d", &["X"], &["Y"], vec![node("Relu", &["X"], &["Y"])]);

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
    let mut f_neg = function("F", "d", &["X"], &["Y"], vec![node("Neg", &["X"], &["Y"])]);
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
    let mut f = function("F", "d", &["X"], &["Y"], vec![node("Relu", &["X"], &["Y"])]);
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
    assert!(
        out.opset_import
            .iter()
            .any(|o| o.domain == "com.microsoft" && o.version == 1)
    );
}

// --- regression tests for the four scope/correctness bugs ------------------

#[test]
fn nested_subgraph_ref_attr_is_bound() {
    // BUG 1: a `ref_attr_name` carried by a node *inside* a control-flow
    // subgraph must be bound to the call-site attribute, not left dangling.
    //
    // F(cond, X){alpha} = If(cond){ then: LeakyRelu(X, alpha=@alpha) -> t; out t
    //                               else: LeakyRelu(X, alpha=@alpha) -> e; out e }
    let then_branch = onnx::GraphProto {
        node: vec![node_with_attrs(
            "LeakyRelu",
            &["X"],
            &["t"],
            vec![ref_attr("alpha", "alpha")],
        )],
        output: vec![value_info("t")],
        ..Default::default()
    };
    let else_branch = onnx::GraphProto {
        node: vec![node_with_attrs(
            "LeakyRelu",
            &["X"],
            &["e"],
            vec![ref_attr("alpha", "alpha")],
        )],
        output: vec![value_info("e")],
        ..Default::default()
    };
    let if_node = node_with_attrs(
        "If",
        &["cond"],
        &["Y"],
        vec![
            graph_attr("then_branch", then_branch),
            graph_attr("else_branch", else_branch),
        ],
    );
    let f = function("F", "d", &["cond", "X"], &["Y"], vec![if_node]);
    let mut c = call("F", "d", &["c", "x"], &["y"]);
    c.attribute = vec![float_attr("alpha", 0.2)];
    let graph = onnx::GraphProto {
        node: vec![c],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    let if_n = &g.node[0];
    assert_eq!(if_n.op_type, "If");
    for attr in &if_n.attribute {
        let sub = attr.g.as_ref().unwrap();
        let lr = &sub.node[0];
        assert_eq!(lr.op_type, "LeakyRelu");
        assert_eq!(lr.attribute.len(), 1);
        assert!(
            lr.attribute[0].ref_attr_name.is_empty(),
            "nested ref_attr_name was not resolved"
        );
        assert_eq!(
            lr.attribute[0].f, 0.2,
            "nested attr not bound to call value"
        );
    }
}

#[test]
fn subgraph_output_capturing_function_input_is_remapped() {
    // BUG 2(b): a subgraph `GraphProto.output` that directly names a captured
    // outer value must be rewritten to the call-site actual.
    //
    // F(cond, X) = If(cond){ then: (no nodes) out X ; else: (no nodes) out X }
    let then_branch = onnx::GraphProto {
        output: vec![value_info("X")],
        ..Default::default()
    };
    let else_branch = onnx::GraphProto {
        output: vec![value_info("X")],
        ..Default::default()
    };
    let if_node = node_with_attrs(
        "If",
        &["cond"],
        &["Y"],
        vec![
            graph_attr("then_branch", then_branch),
            graph_attr("else_branch", else_branch),
        ],
    );
    let f = function("F", "d", &["cond", "X"], &["Y"], vec![if_node]);
    let c = call("F", "d", &["c", "a"], &["y"]);
    let graph = onnx::GraphProto {
        node: vec![c],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    for attr in &g.node[0].attribute {
        let sub = attr.g.as_ref().unwrap();
        assert_eq!(
            sub.output[0].name, "a",
            "captured subgraph output not remapped to actual"
        );
    }
}

#[test]
fn subgraph_local_input_shadows_outer_capture() {
    // BUG 2(a): a subgraph's own graph input shadows an outer formal of the
    // same name; inner references must stay local, not follow the outer actual.
    //
    // F(M, cond, X) = Loop body(iter, cond_in, X): Relu(X)->r ; out cond_in, r
    let body = onnx::GraphProto {
        input: vec![value_info("iter"), value_info("cond_in"), value_info("X")],
        node: vec![node("Relu", &["X"], &["r"])],
        output: vec![value_info("cond_in"), value_info("r")],
        ..Default::default()
    };
    let loop_node = node_with_attrs(
        "Loop",
        &["M", "cond", "X"],
        &["Yf"],
        vec![graph_attr("body", body)],
    );
    let f = function("F", "d", &["M", "cond", "X"], &["Yf"], vec![loop_node]);
    let c = call("F", "d", &["m", "c", "a"], &["y"]);
    let graph = onnx::GraphProto {
        node: vec![c],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    let loop_n = &g.node[0];
    // The outer Loop node inputs live in the body scope and are remapped.
    assert_eq!(
        loop_n.input,
        vec!["m".to_string(), "c".to_string(), "a".to_string()]
    );
    let body_sub = loop_n.attribute[0].g.as_ref().unwrap();
    // The inner Relu reads the loop-carried local `X`, never the outer actual.
    assert_eq!(
        body_sub.node[0].input[0], "X",
        "local subgraph input was wrongly remapped to the outer actual"
    );
}

#[test]
fn subgraph_initializer_shadows_but_capture_is_remapped() {
    // BUG 2(a)+(b): a subgraph initializer shadows an outer name (local), while
    // a sibling branch that genuinely captures the same name is remapped.
    //
    // F(cond, X) = If(cond){ then: initializer X; Add(X, X)->tb; out tb
    //                        else: Identity(X)->eb; out eb }
    let then_branch = onnx::GraphProto {
        initializer: vec![tensor("X")],
        node: vec![node("Add", &["X", "X"], &["tb"])],
        output: vec![value_info("tb")],
        ..Default::default()
    };
    let else_branch = onnx::GraphProto {
        node: vec![node("Identity", &["X"], &["eb"])],
        output: vec![value_info("eb")],
        ..Default::default()
    };
    let if_node = node_with_attrs(
        "If",
        &["cond"],
        &["Y"],
        vec![
            graph_attr("then_branch", then_branch),
            graph_attr("else_branch", else_branch),
        ],
    );
    let f = function("F", "d", &["cond", "X"], &["Y"], vec![if_node]);
    let c = call("F", "d", &["c", "a"], &["y"]);
    let graph = onnx::GraphProto {
        node: vec![c],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    let if_n = &g.node[0];
    let then_sub = if_n.attribute[0].g.as_ref().unwrap();
    // `X` is a local initializer here -> Add stays on local `X`.
    assert_eq!(
        then_sub.node[0].input,
        vec!["X".to_string(), "X".to_string()]
    );
    let else_sub = if_n.attribute[1].g.as_ref().unwrap();
    // `X` is a genuine capture here -> remapped to the actual.
    assert_eq!(else_sub.node[0].input[0], "a");
}

#[test]
fn passthrough_output_aliasing_input_is_wired_via_identity() {
    // BUG 3: pass-through F(X) -> X (zero-node body). Call F(a) -> b must
    // actually produce `b` (via Identity from `a`), and downstream reads `b`.
    let f = function("F", "d", &["X"], &["X"], vec![]);
    let graph = onnx::GraphProto {
        node: vec![call("F", "d", &["a"], &["b"]), node("Relu", &["b"], &["c"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    let idn = g
        .node
        .iter()
        .find(|n| n.op_type == "Identity")
        .expect("pass-through must emit an Identity alias");
    assert_eq!(idn.input, vec!["a".to_string()]);
    assert_eq!(idn.output, vec!["b".to_string()]);
    let relu = g.node.iter().find(|n| n.op_type == "Relu").unwrap();
    assert_eq!(relu.input[0], "b", "downstream consumer reads wrong value");
}

#[test]
fn function_output_passing_through_input_alongside_computed() {
    // BUG 3: a function that returns one of its inputs unchanged alongside a
    // computed output. F(X, W) -> (Y, X) with Y = Mul(X, W).
    let f = function(
        "F",
        "d",
        &["X", "W"],
        &["Y", "X"],
        vec![node("Mul", &["X", "W"], &["Y"])],
    );
    let graph = onnx::GraphProto {
        node: vec![call("F", "d", &["a", "w"], &["y", "xp"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    let mul = g.node.iter().find(|n| n.op_type == "Mul").unwrap();
    assert_eq!(mul.input, vec!["a".to_string(), "w".to_string()]);
    assert_eq!(mul.output, vec!["y".to_string()]);
    let idn = g
        .node
        .iter()
        .find(|n| n.op_type == "Identity")
        .expect("pass-through output must emit an Identity alias");
    assert_eq!(idn.input, vec!["a".to_string()]);
    assert_eq!(idn.output, vec!["xp".to_string()]);
}

/// A `sparse_initializer` entry named `name` (indices/values contents are
/// irrelevant to name-scoping tests; only `values.name` matters).
fn sparse_tensor(name: &str) -> onnx::SparseTensorProto {
    onnx::SparseTensorProto {
        values: Some(tensor(name)),
        ..Default::default()
    }
}

#[test]
fn synthesized_identity_gets_default_opset_import() {
    // BUG 3 regression: a pass-through F(X)->X synthesizes a default-domain
    // `Identity`. If the model declared ONLY a custom-domain opset import (a
    // valid model that never used a default-domain op), the inlined model must
    // still gain a default-domain import so loader validation accepts it.
    let mut f = function("F", "custom.domain", &["X"], &["X"], vec![]);
    f.opset_import = vec![onnx::OperatorSetIdProto {
        domain: "custom.domain".to_string(),
        version: 1,
    }];
    let graph = onnx::GraphProto {
        node: vec![call("F", "custom.domain", &["a"], &["b"])],
        ..Default::default()
    };
    // Deliberately declare ONLY the custom domain — no default `""` import.
    let m = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: "custom.domain".to_string(),
            version: 1,
        }],
        graph: Some(graph),
        functions: vec![f],
        ..Default::default()
    };
    let out = inline(m);
    let g = out.graph.as_ref().unwrap();
    assert!(
        g.node.iter().any(|n| n.op_type == "Identity"),
        "pass-through must synthesize an Identity"
    );
    assert!(
        out.opset_import
            .iter()
            .any(|o| (o.domain.is_empty() || o.domain == "ai.onnx") && o.version >= 1),
        "synthesized default-domain Identity must gain a default-domain opset import"
    );
    // The existing custom-domain import is preserved.
    assert!(
        out.opset_import
            .iter()
            .any(|o| o.domain == "custom.domain" && o.version == 1)
    );
}

#[test]
fn existing_default_opset_import_is_not_downgraded() {
    // BUG 3 regression: when the model already declares a default-domain import,
    // synthesizing an Identity must NOT add a duplicate or downgrade it.
    let f = function("F", "custom.domain", &["X"], &["X"], vec![]);
    let graph = onnx::GraphProto {
        node: vec![call("F", "custom.domain", &["a"], &["b"])],
        ..Default::default()
    };
    // `model()` declares default `""` @ 17.
    let out = inline(model(graph, vec![f]));
    let defaults: Vec<_> = out
        .opset_import
        .iter()
        .filter(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .collect();
    assert_eq!(defaults.len(), 1, "no duplicate default-domain import");
    assert_eq!(defaults[0].version, 17, "existing default import untouched");
}

#[test]
fn ai_onnx_spelled_default_import_plus_synthesized_identity_stays_single() {
    // BUG 4 (duck-fn3): the model imports the default domain spelled "ai.onnx",
    // and a custom-domain pass-through function synthesizes a default `""`
    // Identity. `""` and `"ai.onnx"` are the SAME domain, so the inlined model
    // must carry EXACTLY ONE default-domain opset import — kept at the model's
    // spelling ("ai.onnx") and NOT downgraded from v17.
    let mut f = function("F", "custom.domain", &["X"], &["X"], vec![]);
    f.opset_import = vec![onnx::OperatorSetIdProto {
        domain: "custom.domain".to_string(),
        version: 1,
    }];
    let graph = onnx::GraphProto {
        node: vec![call("F", "custom.domain", &["a"], &["b"])],
        ..Default::default()
    };
    // Model declares the default domain under the "ai.onnx" spelling.
    let m = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![
            onnx::OperatorSetIdProto {
                domain: "ai.onnx".to_string(),
                version: 17,
            },
            onnx::OperatorSetIdProto {
                domain: "custom.domain".to_string(),
                version: 1,
            },
        ],
        graph: Some(graph),
        functions: vec![f],
        ..Default::default()
    };
    let out = inline(m);
    let g = out.graph.as_ref().unwrap();
    assert!(
        g.node.iter().any(|n| n.op_type == "Identity"),
        "pass-through must synthesize an Identity"
    );
    let defaults: Vec<_> = out
        .opset_import
        .iter()
        .filter(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .collect();
    assert_eq!(
        defaults.len(),
        1,
        "exactly one default-domain import despite \"\"/\"ai.onnx\" mismatch"
    );
    assert_eq!(
        defaults[0].domain, "ai.onnx",
        "model's original default spelling is preserved"
    );
    assert_eq!(defaults[0].version, 17, "default import not downgraded");
    assert!(
        out.opset_import
            .iter()
            .any(|o| o.domain == "custom.domain" && o.version == 1)
    );
}

#[test]
fn default_domain_merge_prefers_highest_version_across_spellings() {
    // The model imports the default domain as "ai.onnx" @ 18 while a function
    // contributes the default domain spelled "" @ 20. These collapse to ONE
    // default-domain import at the highest version (20), keeping the model's
    // spelling ("ai.onnx").
    let mut f = function("F", "d", &["X"], &["Y"], vec![node("Relu", &["X"], &["Y"])]);
    f.opset_import = vec![
        onnx::OperatorSetIdProto {
            domain: String::new(),
            version: 20,
        },
        onnx::OperatorSetIdProto {
            domain: "d".to_string(),
            version: 1,
        },
    ];
    let graph = onnx::GraphProto {
        node: vec![call("F", "d", &["x"], &["y"])],
        ..Default::default()
    };
    let m = onnx::ModelProto {
        ir_version: 8,
        opset_import: vec![onnx::OperatorSetIdProto {
            domain: "ai.onnx".to_string(),
            version: 18,
        }],
        graph: Some(graph),
        functions: vec![f],
        ..Default::default()
    };
    let out = inline(m);
    let defaults: Vec<_> = out
        .opset_import
        .iter()
        .filter(|o| o.domain.is_empty() || o.domain == "ai.onnx")
        .collect();
    assert_eq!(
        defaults.len(),
        1,
        "single default-domain import across \"\"/\"ai.onnx\" spellings"
    );
    assert_eq!(
        defaults[0].domain, "ai.onnx",
        "model's original default spelling is preserved"
    );
    assert_eq!(
        defaults[0].version, 20,
        "highest version across contributors"
    );
}

#[test]
fn subgraph_sparse_initializer_shadows_outer_capture() {
    // FIX 2 gap: a subgraph `sparse_initializer` is a local binding that shadows
    // an outer capture, exactly like a dense initializer. F(cond, X) = If(cond){
    //   then: sparse_initializer X; Add(X, X)->tb; out tb
    //   else: Identity(X)->eb; out eb }
    let then_branch = onnx::GraphProto {
        sparse_initializer: vec![sparse_tensor("X")],
        node: vec![node("Add", &["X", "X"], &["tb"])],
        output: vec![value_info("tb")],
        ..Default::default()
    };
    let else_branch = onnx::GraphProto {
        node: vec![node("Identity", &["X"], &["eb"])],
        output: vec![value_info("eb")],
        ..Default::default()
    };
    let if_node = node_with_attrs(
        "If",
        &["cond"],
        &["Y"],
        vec![
            graph_attr("then_branch", then_branch),
            graph_attr("else_branch", else_branch),
        ],
    );
    let f = function("F", "d", &["cond", "X"], &["Y"], vec![if_node]);
    let graph = onnx::GraphProto {
        node: vec![call("F", "d", &["c", "a"], &["y"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    let if_n = &g.node[0];
    let then_sub = if_n.attribute[0].g.as_ref().unwrap();
    // `X` is a local sparse initializer here -> Add stays on local `X`.
    assert_eq!(
        then_sub.node[0].input,
        vec!["X".to_string(), "X".to_string()],
        "sparse-initializer local must not be remapped as an outer capture"
    );
    let else_sub = if_n.attribute[1].g.as_ref().unwrap();
    // `X` is a genuine capture here -> remapped to the actual.
    assert_eq!(else_sub.node[0].input[0], "a");
}

#[test]
fn generated_names_avoid_sparse_initializer_collision() {
    // FIX 4 gap: the outer graph declares a `sparse_initializer` named
    // `__fn0_t`, which is exactly the name the first instantiation would naively
    // pick for internal value `t`. The generated name must avoid it.
    let f = function(
        "F",
        "d",
        &["X"],
        &["Y"],
        vec![node("Relu", &["X"], &["t"]), node("Relu", &["t"], &["Y"])],
    );
    let graph = onnx::GraphProto {
        sparse_initializer: vec![sparse_tensor("__fn0_t")],
        node: vec![call("F", "d", &["a"], &["out"])],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    // The function's internal wire took a fresh, non-colliding name.
    let relus: Vec<_> = g.node.iter().filter(|n| n.op_type == "Relu").collect();
    let wire = &relus[0].output[0];
    assert_ne!(
        wire, "__fn0_t",
        "generated name collided with a sparse-initializer name"
    );
    assert!(wire.starts_with("__fn0_t"));
}

#[test]
fn generated_names_avoid_preexisting_collisions() {
    // BUG 4: the outer graph already defines `__fn0_t`, which is exactly the
    // name the first instantiation would naively pick for internal value `t`.
    let f = function(
        "F",
        "d",
        &["X"],
        &["Y"],
        vec![node("Relu", &["X"], &["t"]), node("Relu", &["t"], &["Y"])],
    );
    let graph = onnx::GraphProto {
        node: vec![
            call("F", "d", &["a"], &["out"]),
            node("Identity", &["seed"], &["__fn0_t"]),
        ],
        ..Default::default()
    };
    let out = inline(model(graph, vec![f]));
    let g = out.graph.as_ref().unwrap();
    // The pre-existing `__fn0_t` keeps its single (Identity) producer.
    let producers: Vec<_> = g
        .node
        .iter()
        .filter(|n| n.output.iter().any(|o| o == "__fn0_t"))
        .collect();
    assert_eq!(
        producers.len(),
        1,
        "generated name collided with pre-existing __fn0_t"
    );
    assert_eq!(producers[0].op_type, "Identity");
    // The function's internal wire took a fresh, non-colliding name.
    let relus: Vec<_> = g.node.iter().filter(|n| n.op_type == "Relu").collect();
    let wire = &relus[0].output[0];
    assert_ne!(wire, "__fn0_t");
    assert!(wire.starts_with("__fn0_t"));
}
