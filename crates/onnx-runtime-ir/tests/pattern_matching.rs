use onnx_runtime_ir::pattern::{
    BoundaryPolicy, DimensionValue, MatchBudget, MatchOptions, MatchSession, Pattern, PatternError,
    PatternErrorKind, PatternLibrary,
};
use onnx_runtime_ir::{
    Attribute, DataType, Graph, Node, NodeId, SparseTensorData, TensorData, ValueId, WeightRef,
};

fn input(graph: &mut Graph, name: &str) -> ValueId {
    let value = graph.create_value(DataType::Float32, vec![2.into()]);
    graph.value_mut(value).name = Some(name.into());
    let mut inputs = graph.inputs.clone();
    inputs.push(value);
    graph.set_inputs(inputs);
    value
}

fn op(
    graph: &mut Graph,
    name: &str,
    inputs: &[Option<ValueId>],
    count: usize,
) -> (NodeId, Vec<ValueId>) {
    let outputs = (0..count)
        .map(|_| graph.create_value(DataType::Float32, vec![2.into()]))
        .collect::<Vec<_>>();
    let node = graph.insert_node(Node::new(NodeId(0), name, inputs.to_vec(), outputs.clone()));
    (node, outputs)
}

fn options(boundary: BoundaryPolicy) -> MatchOptions {
    MatchOptions {
        boundary,
        ..Default::default()
    }
}

fn registered(domain: &str, op: &str) -> bool {
    domain.is_empty()
        && [
            "Identity",
            "Sigmoid",
            "Neg",
            "Add",
            "If",
            "Loop",
            "Scan",
            "Clip",
            "Split",
            "Mul",
            "Softmax",
            "LeakyRelu",
            "CastLike",
        ]
        .contains(&op)
}

fn library(source: &str) -> Result<PatternLibrary, PatternError> {
    PatternLibrary::parse_ort(source, &registered)
}

#[test]
fn anonymous_calls_have_independent_attribute_and_dimension_scopes() {
    for body in [
        "return act(x), act(y)",
        "root=act(x);return root,act(y)",
        "_,a=pair(x);_,b=pair(y);return a,b",
    ] {
        let source = format!(
            "pattern act(x:float[N])->(y:float[N]) {{y=LeakyRelu(x,alpha=alpha);return y}}
             pattern pair(x:float[N]) {{a=act(x);b=Identity(a);return a,b}}
             pattern P(x,y)->(left,right) {{{body}}}"
        );
        let pattern = library(&source).unwrap().graph_pattern("P").unwrap();
        let mut graph = Graph::new();
        let x = input(&mut graph, "x");
        let y = input(&mut graph, "y");
        graph.value_mut(y).shape = vec![3.into()];
        let mut returned = Vec::new();
        for (value, alpha) in [(x, 0.1), (y, 0.2)] {
            let (node, values) = op(&mut graph, "LeakyRelu", &[Some(value)], 1);
            graph
                .node_mut(node)
                .attributes
                .insert("alpha".into(), Attribute::Float(alpha));
            graph.value_mut(values[0]).shape = graph.value(value).shape.clone();
            let result = if body.contains("pair(") {
                let (_, output) = op(&mut graph, "Identity", &[Some(values[0])], 1);
                graph.value_mut(output[0]).shape = graph.value(value).shape.clone();
                output[0]
            } else {
                values[0]
            };
            returned.push(result);
        }
        graph.set_outputs(returned);
        assert!(
            MatchSession::new(&graph)
                .unwrap()
                .match_graph(&pattern, Default::default())
                .unwrap()
                .is_some(),
            "{body}"
        );
    }
}

#[test]
fn pinned_interfaces_anonymous_alternatives_and_hygienic_scopes() {
    let fixture = include_str!("fixtures/patterns/interfaces.ortpattern");
    for call in ["Forms", "(first | second)"] {
        let source = format!(
            "{fixture}\npattern P(x,z) {{ a={call}(x); b={call}(z); return Add(a.y,b.y) }}"
        );
        let pattern = library(&source).unwrap().pattern("P").unwrap();
        let mut graph = Graph::new();
        let x = input(&mut graph, "x");
        let z = input(&mut graph, "z");
        graph.value_mut(z).shape = vec![3.into()];
        let (_, a) = op(&mut graph, "Identity", &[Some(x)], 1);
        let (_, b) = op(&mut graph, "Identity", &[Some(z)], 1);
        graph.value_mut(b[0]).shape = vec![3.into()];
        let (root, outputs) = op(&mut graph, "Add", &[Some(a[0]), Some(b[0])], 1);
        graph.set_outputs(outputs);
        let mut attempts = 0;
        MatchSession::new(&graph)
            .unwrap()
            .visit_at(&pattern, root, Default::default(), |matched| {
                attempts += 1;
                assert_eq!(matched.value("a.y")?.unwrap().id(), a[0]);
                assert_eq!(matched.instance("b")?.value("local")?.unwrap().id(), b[0]);
                assert_eq!(matched.dimension("a.N")?, Some(&DimensionValue::Static(2)));
                assert_eq!(matched.dimension("b.N")?, Some(&DimensionValue::Static(3)));
                Ok(true)
            })
            .unwrap();
        assert_eq!(attempts, 4);
    }
}

#[test]
fn pinned_legacy_ports_aliases_and_implicit_results() {
    let library = library(include_str!("fixtures/patterns/legacy_ports.ortpattern")).unwrap();
    assert!(library.pattern("identity").is_err());
    assert!(library.pattern("mixed").is_err());
    for entry in ["entry", "use_identity", "use_implicit", "use_mixed"] {
        let mut graph = Graph::new();
        let x = input(&mut graph, "x");
        let a = if entry == "use_identity" {
            x
        } else {
            op(&mut graph, "Identity", &[Some(x)], 1).1[0]
        };
        let (root, output) = if entry == "use_mixed" {
            op(&mut graph, "Add", &[Some(x), Some(a)], 1)
        } else {
            op(&mut graph, "Identity", &[Some(a)], 1)
        };
        graph.set_outputs(output);
        if entry == "use_identity" {
            let anonymous = Pattern::parse_ort("(output)=>return Identity(output)").unwrap();
            let matched = MatchSession::new(&graph)
                .unwrap()
                .match_at(&anonymous, root, Default::default())
                .unwrap()
                .unwrap();
            assert_eq!(matched.value("output").unwrap().unwrap().id(), x);
        }
        assert!(
            MatchSession::new(&graph)
                .unwrap()
                .match_at(&library.pattern(entry).unwrap(), root, Default::default())
                .unwrap()
                .is_some(),
            "{entry}"
        );
    }
}

#[test]
fn legacy_multi_result_instance_preserves_port_order_and_identity() {
    let source = r"
        interface Pair(x) -> (left,right)
        pattern pair implements Pair { a,b=Split(x);return (right=b,left=a) }
        pattern P(x) { p=pair(x);return Add(p.left,p.right) }
    ";
    let pattern = library(source).unwrap().pattern("P").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (_, values) = op(&mut graph, "Split", &[Some(x)], 2);
    let (root, outputs) = op(&mut graph, "Add", &[Some(values[0]), Some(values[1])], 1);
    graph.set_outputs(outputs);
    let found = MatchSession::new(&graph)
        .unwrap()
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(found.value("p.left").unwrap().unwrap().id(), values[0]);
    assert_eq!(found.value("p.right").unwrap().unwrap().id(), values[1]);
    assert!(found.value("p").is_err());
    // Pinned Parse orders explicit interface mappings by label; an ordinary
    // output signature still consumes returned values positionally.
    let ordinary = source.replace(
        "pattern pair implements Pair",
        "pattern pair(x) -> (left,right)",
    );
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(
                &library(&ordinary).unwrap().pattern("P").unwrap(),
                root,
                Default::default()
            )
            .unwrap()
            .is_none()
    );
    let incorrect = source.replace("Add(p.left,p.right)", "Add(p.left,p.left)");
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(
                &library(&incorrect).unwrap().pattern("P").unwrap(),
                root,
                Default::default()
            )
            .unwrap()
            .is_none()
    );
    for body in [
        "return Identity(p)",
        "p=Identity(x);return Add(p.left,p.right)",
    ] {
        let invalid = source.replace("return Add(p.left,p.right)", body);
        assert!(library(&invalid).is_err(), "{invalid}");
    }
}

#[test]
fn discovered_inputs_and_attributes_are_not_callable_ports() {
    let source = r"
        interface Unary(x:float[N]) -> (y:float[N])
        pattern scale implements Unary: (alpha:float[N],axis:int) -> {
            v=Mul(x,alpha); y=Softmax(v,axis=axis);return y
        }
        pattern P(x,z) { a=scale(x);b=scale(z);return Add(a.y,b.y) }
    ";
    let pattern = library(source).unwrap().pattern("P").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let z = input(&mut graph, "z");
    let alpha = input(&mut graph, "alpha");
    let beta = input(&mut graph, "beta");
    let (_, a) = op(&mut graph, "Mul", &[Some(x), Some(alpha)], 1);
    let (a_node, a) = op(&mut graph, "Softmax", &[Some(a[0])], 1);
    graph
        .node_mut(a_node)
        .attributes
        .insert("axis".into(), Attribute::Int(0));
    let (_, b) = op(&mut graph, "Mul", &[Some(z), Some(beta)], 1);
    let (b_node, b) = op(&mut graph, "Softmax", &[Some(b[0])], 1);
    graph
        .node_mut(b_node)
        .attributes
        .insert("axis".into(), Attribute::Int(1));
    let (root, output) = op(&mut graph, "Add", &[Some(a[0]), Some(b[0])], 1);
    graph.set_outputs(output);
    let found = MatchSession::new(&graph)
        .unwrap()
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(found.value("a.alpha").unwrap().unwrap().id(), alpha);
    assert_eq!(found.value("b.alpha").unwrap().unwrap().id(), beta);
    assert!(matches!(
        found.attribute("a.axis").unwrap(),
        Some(Attribute::Int(0))
    ));
    assert!(matches!(
        found.attribute("b.axis").unwrap(),
        Some(Attribute::Int(1))
    ));
    assert!(library(&source.replace("a=scale(x)", "a=scale(x,x,x)")).is_err());
}

#[test]
fn interface_alias_outputs_and_required_callee_inputs_cannot_bind_absence() {
    let source = r"
        interface I(v:float | None) -> (y)
        pattern P implements I {return (y=v)}
        pattern Alias=P
        pattern Wrapped implements I {p=Alias(v);return (y=p.y)}
        pattern Entry(x,v:float | None) {p=P(v);return Clip(x,p.y)}
        pattern Nested(x,v:float | None) {p=Wrapped(v);return Clip(x,p.y)}
        pattern required(v) {return (y=v)}
        pattern Required(x,v:float | None) {p=required(v);return Clip(x,p.y)}
    ";
    let library = library(source).unwrap();
    for entry in ["Entry", "Nested", "Required"] {
        for scenario in 0..3 {
            let mut graph = Graph::new();
            let x = input(&mut graph, "x");
            let mut inputs = vec![Some(x)];
            if scenario == 1 {
                inputs.push(None);
            }
            if scenario == 2 {
                let min = input(&mut graph, "min");
                graph.value_mut(min).shape.clear();
                inputs.push(Some(min));
            }
            let (root, output) = op(&mut graph, "Clip", &inputs, 1);
            graph.set_outputs(output);
            let found = MatchSession::new(&graph)
                .unwrap()
                .match_at(&library.pattern(entry).unwrap(), root, Default::default())
                .unwrap();
            assert_eq!(found.is_some(), scenario == 2, "{entry}/{scenario}");
        }
    }
}

#[test]
fn library_contract_errors_are_construction_errors_not_nonmatches() {
    for source in [
        "interface I(x)->(y) pattern P(x){a=I(x);return a.y}",
        "pattern P implements Missing {return (y=x)}",
        "interface I(x)->(y) pattern P implements I {a=Identity(x);return (wrong=a)}",
        "interface I(x)->(y) pattern P implements I {a=Identity(x);return (y=a,y=a)}",
        "interface I(x)->(x)",
        "interface I(x:float[x])->(y)",
        "interface I(x)->(y:float|None)",
        "interface I(x,alpha:float)->(y) pattern P implements I {y=LeakyRelu(x,alpha=alpha);return y}",
        "interface I(x)->(y) interface J(x)->(y) pattern P implements I {y=Identity(x);return y} pattern Q implements J {y=Identity(x);return y} pattern Forms=P|Q",
        "pattern P(x){return Identity(x)} pattern Forms=P|P",
        "pattern Pair(x){a,b=Split(x);return a,b} pattern P(x){p:float=Pair(x);return Add(p.a,p.b)}",
        "pattern Pair(x){a,b=Split(x);return a,b} pattern P(x:float[p]){p=Pair(x);return Add(p.a,p.b)}",
        "interface I(x)->(y) pattern P implements I:(unused:float[])->{y=Identity(x);return y}",
        "pattern P(x)->(x){return x}",
        "pattern A(x){y:float[N]=Identity(x);return y} pattern B(x){y:float[M]=Identity(x);return y} pattern Forms=A|B",
    ] {
        assert!(library(source).is_err(), "{source}");
    }
    let error = library("pattern P(x){return Add(x,Identity(x))}").unwrap_err();
    assert_eq!(error.kind, PatternErrorKind::Unsupported);
    assert!(error.message.contains("assign each call"));
    let error = library(
        r"
        pattern A(x)->(y:float[N]){y=Identity(x);return y}
        pattern B(x)->(y:float[M]){y=Identity(x);return y}
        pattern P(x){p=A(x)|B(x);return Identity(p)}
    ",
    )
    .unwrap_err();
    assert!(error.message.contains("incompatible result annotations"));
    assert!(
        library(
            r"
        pattern A(x:float[])->(y:float[]){y=Identity(x);return y}
        pattern B(x:float[...])->(y:float[...]){y=Identity(x);return y}
        pattern Forms=A|B
    "
        )
        .is_ok()
    );
}

#[test]
fn inferred_results_do_not_add_constraints_from_other_alternatives() {
    // Pinned InferredExpressionOutputsAllowUnconstrainedBranches expectations.
    for branches in ["x | Identity(x)", "Identity(x) | x"] {
        let source = format!(
            r"
            pattern Symbolic(x:float[M]) {{ y={branches};return y }}
            pattern Nested(x) {{ y=Symbolic(x);return y }}
            pattern Entry(x) {{ y=Nested(x);return Sigmoid(y) }}
        "
        );
        let pattern = library(&source).unwrap().pattern("Entry").unwrap();
        let mut graph = Graph::new();
        let x = input(&mut graph, "x");
        let anonymous = graph.create_symbol(None);
        graph.value_mut(x).shape = vec![onnx_runtime_ir::Dim::Symbolic(anonymous)];
        let (_, y) = op(&mut graph, "Identity", &[Some(x)], 1);
        graph.value_mut(y[0]).shape = vec![onnx_runtime_ir::Dim::Symbolic(anonymous)];
        let (root, output) = op(&mut graph, "Sigmoid", &[Some(y[0])], 1);
        graph.set_outputs(output);
        let mut count = 0;
        MatchSession::new(&graph)
            .unwrap()
            .visit_at(&pattern, root, Default::default(), |matched| {
                assert_eq!(matched.dimension("y.y.M")?, None);
                count += 1;
                Ok(true)
            })
            .unwrap();
        assert_eq!(count, 2);
    }
}

#[test]
fn annotation_search_order_survives_reordered_and_aliased_calls() {
    // Pinned AnnotationPrioritySurvivesReorderedAndAliasedCalls expectations.
    let source = r"
        pattern F(x:float[M,_]|float[_,M],z:float[_,M]|float[M,_]) {
            y=CastLike(x,z);return y
        }
        pattern Reordered(v,w) {p=F(w,v);return p.y}
        pattern Aliased(v) {p=F(v,v);return p.y}
        pattern Alias=F
        pattern Nested(v,w) {p=Alias(w,v);return p.y}
    ";
    let library = library(source).unwrap();
    for entry in ["F", "Reordered", "Aliased", "Nested"] {
        let mut graph = Graph::new();
        let v = input(&mut graph, "v");
        graph.value_mut(v).shape = vec![2.into(), 3.into()];
        let w = if entry == "Aliased" {
            v
        } else {
            let w = input(&mut graph, "w");
            graph.value_mut(w).shape = vec![2.into(), 3.into()];
            w
        };
        let (root, output) = op(&mut graph, "CastLike", &[Some(w), Some(v)], 1);
        graph.set_outputs(output);
        let matched = MatchSession::new(&graph)
            .unwrap()
            .match_at(&library.pattern(entry).unwrap(), root, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            matched
                .dimension(if entry == "F" { "M" } else { "p.M" })
                .unwrap(),
            Some(&DimensionValue::Static(2)),
            "{entry}"
        );
    }
}

#[test]
fn ort_boundary_keeps_anonymous_call_inputs_outside_the_footprint() {
    let pattern = library(
        r"
        pattern Copy(x){return Identity(x)}
        pattern P(x){a=Identity(x);b=Copy(_);return Add(a,b)}
    ",
    )
    .unwrap()
    .pattern("P")
    .unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (_, a) = op(&mut graph, "Identity", &[Some(x)], 1);
    let (_, b) = op(&mut graph, "Identity", &[Some(a[0])], 1);
    let (root, output) = op(&mut graph, "Add", &[Some(a[0]), Some(b[0])], 1);
    graph.set_outputs(output);
    let session = MatchSession::new(&graph).unwrap();
    assert!(
        session
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        session
            .match_at(&pattern, root, options(BoundaryPolicy::OrtBoundary))
            .unwrap()
            .is_none()
    );
}

#[test]
fn qdq_shared_output_is_structural_not_deletion_safe() {
    let pattern = Pattern::parse_ort(
        "(x,w,s,z : tensor[...] | None)=>{d=DequantizeLinear(w,s,z);return MatMul(x,d)}",
    )
    .unwrap();
    let key = pattern.value_capture("w").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let w = input(&mut graph, "w");
    let s = input(&mut graph, "s");
    let (_, d) = op(&mut graph, "DequantizeLinear", &[Some(w), Some(s)], 1);
    let (root, y) = op(&mut graph, "MatMul", &[Some(x), Some(d[0])], 1);
    op(&mut graph, "Identity", &[Some(d[0])], 1);
    graph.set_outputs(y);
    let session = MatchSession::new(&graph).unwrap();
    let found = session
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(found.captured(&key).unwrap().unwrap().id(), w);
    assert!(found.value("z").unwrap().is_none());
    assert!(
        session
            .match_at(&pattern, root, options(BoundaryPolicy::OrtBoundary))
            .unwrap()
            .is_none()
    );
    let wrong = Pattern::parse_ort("(x)=>return Identity(x)")
        .unwrap()
        .value_capture("x")
        .unwrap();
    assert_eq!(
        found.captured(&wrong).unwrap_err().kind,
        PatternErrorKind::InvalidCapture
    );
}

#[test]
fn absent_slots_present_wildcards_repeated_optional_and_exact_arity() {
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (missing, _) = op(&mut graph, "Clip", &[Some(x), None], 1);
    let (mixed, _) = op(&mut graph, "Clip", &[Some(x), None, Some(x)], 1);
    let (repeated, _) = op(&mut graph, "Clip", &[Some(x), Some(x), Some(x)], 1);
    let absent = Pattern::parse_ort("(x)=>return Clip(x,None,None)").unwrap();
    let repeated_pattern =
        Pattern::parse_ort("(x,v:tensor[...]|None)=>return Clip(x,v,v)").unwrap();
    let present = Pattern::parse_ort("(x)=>return Clip(x,_,_)").unwrap();
    let short = Pattern::parse_ort("(x)=>return Clip(x)").unwrap();
    let session = MatchSession::new(&graph).unwrap();
    assert!(
        session
            .match_at(&absent, missing, Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        session
            .match_at(&absent, mixed, Default::default())
            .unwrap()
            .is_none()
    );
    assert!(
        session
            .match_at(&repeated_pattern, missing, Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        session
            .match_at(&repeated_pattern, mixed, Default::default())
            .unwrap()
            .is_none()
    );
    assert!(
        session
            .match_at(&present, mixed, Default::default())
            .unwrap()
            .is_none()
    );
    assert!(
        session
            .match_at(&present, repeated, Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        session
            .match_at(&short, missing, Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn absent_output_slots_are_distinct_from_present_wildcards() {
    let absent =
        Pattern::parse_ort("(x)=>{y,None=Dropout(x);return y}").expect("absent output pattern");
    let present =
        Pattern::parse_ort("(x)=>{y,_=Dropout(x);return y}").expect("present output pattern");

    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let y = graph.create_value(DataType::Float32, vec![2.into()]);
    let root = graph.insert_node(Node::new_with_output_slots(
        NodeId(0),
        "Dropout",
        vec![Some(x)],
        vec![Some(y), None],
    ));
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&absent, root, Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&present, root, Default::default())
            .unwrap()
            .is_none()
    );

    let mask = graph.create_value(DataType::Bool, vec![2.into()]);
    graph.replace_node(
        root,
        Node::new(NodeId(0), "Dropout", vec![Some(x)], vec![y, mask]),
    );
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&absent, root, Default::default())
            .unwrap()
            .is_none()
    );
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&present, root, Default::default())
            .unwrap()
            .is_some()
    );
}

#[test]
fn literal_ort_alternative_fixture_and_predicate_rollback() {
    // ORT Pattern_Rewriter.md, "Matching alternatives at one root", pinned in
    // docs/architecture/IR_PATTERNS.md. Independently constructed IR fixture.
    let library = library(include_str!("fixtures/patterns/alternatives.ortpattern")).unwrap();
    let pattern = library.pattern("E").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (_, identity) = op(&mut graph, "Identity", &[Some(x)], 1);
    let (root, _) = op(&mut graph, "Sigmoid", &[Some(identity[0])], 1);
    let session = MatchSession::new(&graph).unwrap();
    let mut seen = Vec::new();
    session
        .visit_at(&pattern, root, Default::default(), |m| {
            seen.push((m.selected_alternative(), m.value("x")?.unwrap().id()));
            Ok(true)
        })
        .unwrap();
    assert_eq!(seen, [(0, x), (1, identity[0])]);
    let accepted = session
        .match_at_where(&pattern, root, Default::default(), |m| {
            Ok(m.selected_alternative() == 1)
        })
        .unwrap()
        .unwrap();
    assert_eq!(accepted.value("x").unwrap().unwrap().id(), identity[0]);
    let error = session
        .match_at_where(&pattern, root, Default::default(), |_| {
            Err(PatternError::predicate(
                "test predicate needs metadata; supply its facts",
            ))
        })
        .unwrap_err();
    assert_eq!(error.kind, PatternErrorKind::Predicate);
}

#[test]
fn composition_is_hygienic_and_required_formals_do_not_become_optional() {
    let compiled = library(
        "
        pattern Copy(x){y=Identity(x);return y}
        pattern Pair(x,z){a=Copy(x);b=Copy(z);return Add(a,b)}
    ",
    )
    .unwrap();
    let pattern = compiled.pattern("Pair").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let z = input(&mut graph, "z");
    let (_, a) = op(&mut graph, "Identity", &[Some(x)], 1);
    let (_, b) = op(&mut graph, "Identity", &[Some(z)], 1);
    let (root, _) = op(&mut graph, "Add", &[Some(a[0]), Some(b[0])], 1);
    let session = MatchSession::new(&graph).unwrap();
    let matched = session
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        matched
            .instance("a")
            .unwrap()
            .value("x")
            .unwrap()
            .unwrap()
            .id(),
        x
    );
    assert_eq!(
        matched
            .instance("b")
            .unwrap()
            .value("x")
            .unwrap()
            .unwrap()
            .id(),
        z
    );
    let optional = library(
        "pattern P(x){y=Identity(x);return y} pattern Q(x:tensor[...]|None){y=P(x);return y}",
    )
    .unwrap()
    .pattern("Q")
    .unwrap();
    let mut absent_graph = Graph::new();
    let (root, _) = op(&mut absent_graph, "Identity", &[None], 1);
    assert!(
        MatchSession::new(&absent_graph)
            .unwrap()
            .match_at(&optional, root, Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn multioutput_identity_and_connected_multi_sink_enumeration() {
    let pattern = Pattern::parse_ort("(x)=>{a,b=Split(x,axis=0);return Sub(a,b)}").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (split, outputs) = op(&mut graph, "Split", &[Some(x)], 2);
    graph
        .node_mut(split)
        .attributes
        .insert("axis".into(), Attribute::Int(0));
    let (good, _) = op(&mut graph, "Sub", &[Some(outputs[0]), Some(outputs[1])], 1);
    let (bad, _) = op(&mut graph, "Sub", &[Some(outputs[0]), Some(outputs[0])], 1);
    let session = MatchSession::new(&graph).unwrap();
    assert!(
        session
            .match_at(&pattern, good, Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        session
            .match_at(&pattern, bad, Default::default())
            .unwrap()
            .is_none()
    );
    let sinks =
        Pattern::parse_ort("(x)=>{c=Identity(x);state=Neg(c);y=Abs(c);return y,state}").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (_, c) = op(&mut graph, "Identity", &[Some(x)], 1);
    let (neg1, _) = op(&mut graph, "Neg", &[Some(c[0])], 1);
    let (neg2, _) = op(&mut graph, "Neg", &[Some(c[0])], 1);
    let (root, _) = op(&mut graph, "Abs", &[Some(c[0])], 1);
    let found = MatchSession::new(&graph)
        .unwrap()
        .find_matches(&sinks, Default::default())
        .unwrap();
    assert_eq!(found.len(), 2);
    assert_eq!(found[0].nodes(), &[NodeId(0), neg1, root]);
    assert_eq!(found[1].nodes(), &[NodeId(0), neg2, root]);
}

#[test]
fn distinct_pattern_operators_cannot_alias_one_model_node() {
    let pattern = Pattern::parse_ort("(x)=>{a=Identity(x);b=Identity(x);return Add(a,b)}").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (_, a) = op(&mut graph, "Identity", &[Some(x)], 1);
    let (root, _) = op(&mut graph, "Add", &[Some(a[0]), Some(a[0])], 1);
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn attributes_optional_presence_types_and_repeated_values() {
    let pattern =
        Pattern::parse_ort("(x,p:ints|None)=>{a=Transpose(x,perm=p);return Transpose(a,perm=p)}")
            .unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (first, a) = op(&mut graph, "Transpose", &[Some(x)], 1);
    let (root, _) = op(&mut graph, "Transpose", &[Some(a[0])], 1);
    {
        let session = MatchSession::new(&graph).unwrap();
        let found = session
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .unwrap();
        assert!(found.attribute("p").unwrap().is_none());
    }
    graph
        .node_mut(first)
        .attributes
        .insert("perm".into(), Attribute::Ints(vec![0]));
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
    graph
        .node_mut(root)
        .attributes
        .insert("perm".into(), Attribute::Ints(vec![0]));
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_some()
    );
    graph
        .node_mut(root)
        .attributes
        .insert("perm".into(), Attribute::Int(0));
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn global_type_alternative_search_repeated_unknown_dimensions_and_unknown_rank() {
    let pattern =
        Pattern::parse_ort("(x:float[M]|float[_,M],y:float[M])=>return Add(x,y)").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let y = input(&mut graph, "y");
    graph.value_mut(x).shape = vec![3.into(), 2.into()];
    let (root, _) = op(&mut graph, "Add", &[Some(x), Some(y)], 1);
    {
        let session = MatchSession::new(&graph).unwrap();
        let matched = session
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            matched.dimension("M").unwrap(),
            Some(&DimensionValue::Static(2))
        );
    }
    let symbol = graph.create_symbol(None);
    graph.value_mut(y).shape = vec![symbol.into()];
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
    let singleton = Pattern::parse_ort("(x:float[_,N])=>return Identity(x)").unwrap();
    graph.value_mut(x).shape = vec![2.into(), symbol.into()];
    let (id, _) = op(&mut graph, "Identity", &[Some(x)], 1);
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&singleton, id, Default::default())
            .unwrap()
            .unwrap()
            .dimension("N")
            .unwrap()
            .is_none()
    );
    graph.mark_value_shape_unknown(x);
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&singleton, id, Default::default())
            .unwrap()
            .is_none()
    );
    let any_rank = Pattern::parse_ort("(x:float[])=>return Identity(x)").unwrap();
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&any_rank, id, Default::default())
            .unwrap()
            .is_some()
    );
}

#[test]
fn constants_preserve_dtype_rank_immutability_and_storage_errors() {
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let constant = graph.create_value(DataType::Int64, vec![]);
    graph.set_initializer(
        constant,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Int64,
            vec![],
            3i64.to_le_bytes().to_vec(),
        )),
    );
    let (root, _) = op(&mut graph, "Add", &[Some(x), Some(constant)], 1);
    for (text, expected) in [
        ("(x)=>return Add(x,3)", true),
        ("(x)=>return Add(x,3.0)", false),
        ("(x)=>return Add(x,[3])", false),
    ] {
        let pattern = Pattern::parse_ort(text).unwrap();
        assert_eq!(
            MatchSession::new(&graph)
                .unwrap()
                .match_at(&pattern, root, Default::default())
                .unwrap()
                .is_some(),
            expected
        );
    }
    let pattern = Pattern::parse_ort("(x)=>return Add(x,3)").unwrap();
    graph.set_inputs(vec![x, constant]);
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
    graph.set_inputs(vec![x]);
    graph.set_initializer(
        constant,
        WeightRef::Inline(TensorData::from_raw(DataType::Int64, vec![], vec![1])),
    );
    assert_eq!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap_err()
            .kind,
        PatternErrorKind::InvalidGraph
    );
}

fn branch(name: &str) -> Graph {
    let mut graph = Graph::new();
    let capture = graph.create_value(DataType::Float32, vec![2.into()]);
    graph.value_mut(capture).name = Some(name.into());
    let (_, output) = op(&mut graph, "Identity", &[Some(capture)], 1);
    graph.set_outputs(output);
    graph
}

#[test]
fn graph_attributes_have_independent_scopes_and_explicit_origin_resolution() {
    let library = library(
        "
        pattern Branch(x:float[N]){y=Identity(x);return y}
        pattern Conditional(c){y=If(c,then_branch=Branch,else_branch=Branch);return y}
    ",
    )
    .unwrap();
    let pattern = library.pattern("Conditional").unwrap();
    let mut graph = Graph::new();
    let condition = input(&mut graph, "cond");
    let outer = input(&mut graph, "outer");
    let (root, _) = op(&mut graph, "If", &[Some(condition)], 1);
    graph.node_mut(root).attributes.insert(
        "then_branch".into(),
        Attribute::Graph(Box::new(branch("outer"))),
    );
    graph.node_mut(root).attributes.insert(
        "else_branch".into(),
        Attribute::Graph(Box::new(branch("outer"))),
    );
    let session = MatchSession::new(&graph).unwrap();
    let matched = session
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    let then = matched.instance("y.then_branch").unwrap();
    let otherwise = matched.instance("y.else_branch").unwrap();
    assert_ne!(then.value("x").unwrap(), otherwise.value("x").unwrap());
    assert_eq!(then.resolve_input("x", &graph).unwrap().id(), outer);
    assert_eq!(otherwise.resolve_input("x", &graph).unwrap().id(), outer);
    assert_eq!(
        then.dimension("N").unwrap(),
        Some(&DimensionValue::Static(2))
    );
}

#[test]
fn whole_graph_formals_passthrough_and_dead_work() {
    let library = library("pattern Body(iter,cond,state){return cond,state}").unwrap();
    let body = library.graph_pattern("Body").unwrap();
    assert!(library.pattern("Body").is_err());
    let mut graph = Graph::new();
    input(&mut graph, "iter");
    let cond = input(&mut graph, "cond");
    let state = input(&mut graph, "state");
    graph.set_outputs(vec![cond, state]);
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_graph(&body, Default::default())
            .unwrap()
            .is_some()
    );
    op(&mut graph, "Neg", &[Some(state)], 1);
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_graph(&body, Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn invalid_ir_diagnostics_and_budgets_are_not_nonmatches() {
    let pattern = Pattern::parse_ort("(x)=>return Identity(x)").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (root, _) = op(&mut graph, "Identity", &[Some(x)], 1);
    let session = MatchSession::new(&graph).unwrap();
    let options = MatchOptions {
        budget: MatchBudget {
            steps: 0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert_eq!(
        session.match_at(&pattern, root, options).unwrap_err().kind,
        PatternErrorKind::BudgetExceeded
    );
    assert_eq!(
        session
            .match_at(&pattern, NodeId(999), Default::default())
            .unwrap_err()
            .kind,
        PatternErrorKind::InvalidGraph
    );
    graph.node_mut(root).inputs[0] = Some(ValueId(999));
    assert_eq!(
        MatchSession::new(&graph).err().unwrap().kind,
        PatternErrorKind::InvalidGraph
    );
    for source in [
        "(x)=>return Identity(unknown)",
        "(x)=>{a=Identity(x);a=Neg(a);return a}",
        "(x)=>return Add(x,[1,2.0])",
        "(x)=>return Add(x,1e99)",
    ] {
        let error = Pattern::parse_ort(source).unwrap_err();
        assert!(!error.message.is_empty());
        assert!(error.span.start <= error.span.end);
    }
    assert!(library("pattern A(x){y=B(x);return y} pattern B(x){y=A(x);return y}").is_err());
}

#[test]
fn native_sequences_preserve_order_absence_bounds_and_repeated_identity() {
    let pattern = Pattern::parse_native(
        "(head, xs:slots[0:3], tail)=>{a=Custom(head,...xs,tail);return Custom(a,...xs,tail)}",
    )
    .unwrap();
    assert!(Pattern::parse_ort("(xs:values[0:*])=>return Concat(...xs)").is_err());
    assert!(
        Pattern::parse_native("(xs:slots[0:*],ys:slots[0:*])=>return Add(...xs,...ys)").is_err()
    );
    let mut graph = Graph::new();
    let head = input(&mut graph, "head");
    let tail = input(&mut graph, "tail");
    let (_, a) = op(
        &mut graph,
        "Custom",
        &[Some(head), None, Some(head), Some(tail)],
        1,
    );
    let (root, _) = op(
        &mut graph,
        "Custom",
        &[Some(a[0]), None, Some(head), Some(tail)],
        1,
    );
    {
        let session = MatchSession::new(&graph).unwrap();
        let matched = session
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .unwrap();
        assert_eq!(
            matched
                .sequence("xs")
                .unwrap()
                .iter()
                .map(|value| value.map(|v| v.id()))
                .collect::<Vec<_>>(),
            [None, Some(head)]
        );
        assert!(matched.value("xs").is_err());
        assert!(pattern.value_capture("xs").is_err());
        let present = Pattern::parse_native("(xs:values[0:*])=>return Custom(...xs)").unwrap();
        assert!(
            session
                .match_at(&present, root, Default::default())
                .unwrap()
                .is_none()
        );
    }
    // Rebuild an otherwise identical root with the middle order changed.
    let (bad, _) = op(
        &mut graph,
        "Custom",
        &[Some(a[0]), Some(head), None, Some(tail)],
        1,
    );
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, bad, Default::default())
            .unwrap()
            .is_none()
    );
    let zero = Pattern::parse_native("(head,xs:values[0:0],tail)=>return Custom(head,...xs,tail)")
        .unwrap();
    let (empty, _) = op(&mut graph, "Custom", &[Some(head), Some(tail)], 1);
    let matched = MatchSession::new(&graph)
        .unwrap()
        .match_at(&zero, empty, Default::default())
        .unwrap()
        .unwrap();
    assert!(matched.sequence("xs").unwrap().is_empty());
}

#[test]
fn native_operator_wildcard_and_graph_lists_are_explicit() {
    let pattern = Pattern::parse_native("(xs:values[1:*])=>return *(...xs)").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (node, _) = op(&mut graph, "Anything", &[Some(x)], 1);
    graph.node_mut(node).domain = "vendor".into();
    assert_eq!(
        MatchSession::new(&graph)
            .unwrap()
            .find_matches(&pattern, Default::default())
            .unwrap()
            .len(),
        1
    );
    assert!(Pattern::parse_ort("(x)=>return *(x)").is_err());
    let source = "pattern Branch(x){return x} pattern Owner(x){y=custom.Owner(x,bodies=[Branch,Branch]);return y}";
    let library = PatternLibrary::parse_native(source, &registered).unwrap();
    let pattern = library.pattern("Owner").unwrap();
    let mut body = Graph::new();
    let formal = input(&mut body, "local");
    body.set_outputs(vec![formal]);
    let (root, _) = op(&mut graph, "Owner", &[Some(x)], 1);
    graph.node_mut(root).domain = "custom".into();
    graph
        .node_mut(root)
        .attributes
        .insert("bodies".into(), Attribute::Graphs(vec![body.clone(), body]));
    let matched = MatchSession::new(&graph)
        .unwrap()
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    assert_ne!(
        matched.instance("y.bodies[0]").unwrap().value("x").unwrap(),
        matched.instance("y.bodies[1]").unwrap().value("x").unwrap()
    );
    assert!(PatternLibrary::parse_ort(source, &registered).is_err());
}

#[test]
fn nested_loop_if_formals_captures_and_local_values_do_not_escape() {
    let source = "
            pattern Branch(x){y=Identity(x);return y}
            pattern Body(iter,cond,state){c=Identity(cond);y=If(cond,then_branch=Branch,else_branch=Branch);return c,y}
            pattern Outer(trip:tensor[...]|None,cond:tensor[...]|None,state){y=Loop(trip,cond,state,body=Body);return y}
        ";
    let pattern = library(source).unwrap().pattern("Outer").unwrap();
    let mut graph = Graph::new();
    let cond = input(&mut graph, "cond");
    let state = input(&mut graph, "outer_state");
    let mut body = Graph::new();
    input(&mut body, "iteration");
    let body_cond = input(&mut body, "body_cond");
    let body_state = input(&mut body, "body_state");
    let (_, continue_loop) = op(&mut body, "Identity", &[Some(body_cond)], 1);
    let (conditional, next) = op(&mut body, "If", &[Some(body_cond)], 1);
    body.node_mut(conditional).attributes.insert(
        "then_branch".into(),
        Attribute::Graph(Box::new(branch("body_state"))),
    );
    body.node_mut(conditional).attributes.insert(
        "else_branch".into(),
        Attribute::Graph(Box::new(branch("body_state"))),
    );
    body.set_outputs(vec![continue_loop[0], next[0]]);
    let (root, _) = op(&mut graph, "Loop", &[None, Some(cond), Some(state)], 1);
    graph
        .node_mut(root)
        .attributes
        .insert("body".into(), Attribute::Graph(Box::new(body)));
    let session = MatchSession::new(&graph).unwrap();
    let matched = session
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    let body = matched.instance("y.body").unwrap();
    let nested = body.instance("y.then_branch").unwrap();
    assert_eq!(
        nested.resolve_input("x", body.graph()).unwrap().id(),
        body_state
    );
    assert!(nested.resolve_input("x", &graph).is_err());
    assert!(body.resolve_input("state", &graph).is_err());
    assert_ne!(
        body.value("state").unwrap(),
        matched.value("state").unwrap()
    );
}

#[test]
fn grandparent_capture_can_resolve_through_an_intermediate_proxy() {
    let source = "
            pattern Leaf(x){y=Identity(x);return y}
            pattern Middle(c,x){local=Identity(x);y=If(c,then_branch=Leaf,else_branch=Leaf);return local,y}
            pattern Root(c){a,b=custom.Region(c,body=Middle);return b}
        ";
    let pattern = library(source).unwrap().pattern("Root").unwrap();
    let mut graph = Graph::new();
    let c = input(&mut graph, "cond");
    let outer = input(&mut graph, "outside");
    let mut middle = Graph::new();
    let cond = input(&mut middle, "middle_cond");
    let proxy = middle.create_value(DataType::Float32, vec![2.into()]);
    middle.value_mut(proxy).name = Some("outside".into());
    let (_, local) = op(&mut middle, "Identity", &[Some(proxy)], 1);
    let (conditional, y) = op(&mut middle, "If", &[Some(cond)], 1);
    middle.node_mut(conditional).attributes.insert(
        "then_branch".into(),
        Attribute::Graph(Box::new(branch("outside"))),
    );
    middle.node_mut(conditional).attributes.insert(
        "else_branch".into(),
        Attribute::Graph(Box::new(branch("outside"))),
    );
    middle.set_outputs(vec![local[0], y[0]]);
    let (root, _) = op(&mut graph, "Region", &[Some(c)], 2);
    graph.node_mut(root).domain = "custom".into();
    graph
        .node_mut(root)
        .attributes
        .insert("body".into(), Attribute::Graph(Box::new(middle)));
    let session = MatchSession::new(&graph).unwrap();
    let matched = session
        .match_at(&pattern, root, Default::default())
        .unwrap()
        .unwrap();
    let middle = matched.instance("a.body").unwrap();
    let leaf = middle.instance("y.then_branch").unwrap();
    assert_eq!(leaf.resolve_input("x", middle.graph()).unwrap().id(), proxy);
    assert_eq!(leaf.resolve_input("x", &graph).unwrap().id(), outer);
}

#[test]
fn descendant_capture_is_a_boundary_use_and_capture_dependencies_cannot_cycle() {
    let pattern = Pattern::parse_ort("(x)=>{a=Identity(x);return Neg(a)}").unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (_, a) = op(&mut graph, "Identity", &[Some(x)], 1);
    graph.value_mut(a[0]).name = Some("captured".into());
    let (root, _) = op(&mut graph, "Neg", &[Some(a[0])], 1);
    let (owner, _) = op(&mut graph, "Region", &[], 1);
    graph.node_mut(owner).attributes.insert(
        "body".into(),
        Attribute::Graph(Box::new(branch("captured"))),
    );
    let session = MatchSession::new(&graph).unwrap();
    assert!(
        session
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_some()
    );
    assert!(
        session
            .match_at(&pattern, root, options(BoundaryPolicy::OrtBoundary))
            .unwrap()
            .is_none()
    );
    drop(session);
    // owner -> a -> capture(owner) is an implicit dependency cycle.
    let replacement = Node::new(NodeId(0), "Identity", vec![graph.node(owner).outputs[0]], a);
    graph.replace_node(NodeId(0), replacement);
    let error = MatchSession::new(&graph).err().unwrap();
    assert_eq!(error.kind, PatternErrorKind::InvalidGraph);
    assert!(error.message.contains("cycle"));
}

#[test]
fn unknown_library_calls_and_incompatible_named_alternatives_are_diagnostics() {
    assert!(
        library("pattern Typo(x){y=Identitty(x);return y}")
            .unwrap_err()
            .message
            .contains("unresolved call")
    );
    let source = "pattern A(x:float[]){y=Identity(x);return y} pattern B(x:int64[]){y=Identity(x);return y} pattern Either=A|B";
    assert!(library(source).is_err());
    let valid = library(
        "pattern A(x){y=Identity(x);return y} pattern B(x){y=Neg(x);return y} pattern Either=A|B",
    )
    .unwrap();
    assert!(valid.pattern("Either").is_ok());
    assert!(
        library(
            r"
        interface I(x)->(left,right)
        pattern P implements I:(left:float[])->{
            a=Identity(x);return(left=left,right=a)
        }
    "
        )
        .is_ok()
    );
    assert!(Pattern::parse_ort("(x:float[]|float[...])=>return Identity(x)").is_err());
    assert!(Pattern::parse_ort("(x:tensor<T>[T])=>return Identity(x)").is_err());
}

#[test]
fn required_callee_attribute_cannot_be_weakened_by_optional_caller() {
    let pattern = library(
        "
            pattern P(x,p:int){y=custom.Op(x,axis=p);return y}
            pattern Q(x,p:int|None){y=P(x,p);return y}
        ",
    )
    .unwrap()
    .pattern("Q")
    .unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (root, _) = op(&mut graph, "Op", &[Some(x)], 1);
    graph.node_mut(root).domain = "custom".into();
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn domains_comments_strings_and_float_literal_zero_equality() {
    let pattern = Pattern::parse_ort(
        "# header\n(x)=>{y=ai.onnx::Identity(x);return custom.Op(y,text='a#b',value=-0.0)} # done",
    )
    .unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let (_, a) = op(&mut graph, "Identity", &[Some(x)], 1);
    let (root, _) = op(&mut graph, "Op", &[Some(a[0])], 1);
    graph.node_mut(root).domain = "custom".into();
    graph
        .node_mut(root)
        .attributes
        .insert("text".into(), Attribute::String(b"a#b".to_vec()));
    graph
        .node_mut(root)
        .attributes
        .insert("value".into(), Attribute::Float(0.0));
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_some()
    );
    graph.node_mut(NodeId(0)).domain = "other".into();
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_none()
    );
    let pattern = Pattern::parse_ort("(x)=>return Add(x,0.0)").unwrap();
    let scalar = graph.create_value(DataType::Float32, vec![]);
    graph.set_initializer(
        scalar,
        WeightRef::Inline(TensorData::from_raw(
            DataType::Float32,
            vec![],
            (-0.0f32).to_le_bytes().to_vec(),
        )),
    );
    let (root, _) = op(&mut graph, "Add", &[Some(x), Some(scalar)], 1);
    assert!(
        MatchSession::new(&graph)
            .unwrap()
            .match_at(&pattern, root, Default::default())
            .unwrap()
            .is_some()
    );
}

#[test]
fn repeated_capture_property_matches_reference_identity_for_small_graphs() {
    let pattern = Pattern::parse_ort("(x)=>return Add(x,x)").unwrap();
    for lhs in 0..4 {
        for rhs in 0..4 {
            let mut graph = Graph::new();
            let sources = (0..4)
                .map(|i| input(&mut graph, &format!("x{i}")))
                .collect::<Vec<_>>();
            let (root, _) = op(
                &mut graph,
                "Add",
                &[Some(sources[lhs]), Some(sources[rhs])],
                1,
            );
            let found = MatchSession::new(&graph)
                .unwrap()
                .match_at(&pattern, root, Default::default())
                .unwrap()
                .is_some();
            assert_eq!(found, lhs == rhs);
        }
    }
}

#[test]
fn pinned_upstream_optional_and_union_expectations() {
    let optional =
        Pattern::parse_ort(include_str!("fixtures/patterns/optional.ortpattern")).unwrap();
    for scenario in 0..4 {
        let mut graph = Graph::new();
        let x = input(&mut graph, "x");
        let bound = input(&mut graph, "bound");
        let other = input(&mut graph, "other");
        let (_, a) = op(&mut graph, "Identity", &[Some(x)], 1);
        let low = if scenario == 0 { None } else { Some(bound) };
        let high = if scenario < 2 {
            None
        } else if scenario == 2 {
            Some(bound)
        } else {
            Some(other)
        };
        let (root, _) = op(&mut graph, "Clip", &[Some(a[0]), low, high], 1);
        let matched = MatchSession::new(&graph)
            .unwrap()
            .match_at(&optional, root, Default::default())
            .unwrap()
            .is_some();
        assert_eq!(
            matched,
            scenario == 0 || scenario == 2,
            "upstream optional scenario {scenario}"
        );
    }
    let union =
        Pattern::parse_ort(include_str!("fixtures/patterns/type_union.ortpattern")).unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    graph.value_mut(x).shape = vec![2.into(), 3.into()];
    let y = input(&mut graph, "y");
    graph.value_mut(y).shape = vec![3.into()];
    let (root, _) = op(&mut graph, "CastLike", &[Some(x), Some(y)], 1);
    let matched = MatchSession::new(&graph)
        .unwrap()
        .match_at(&union, root, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(
        matched.dimension("M").unwrap(),
        Some(&DimensionValue::Static(3))
    );
    assert_eq!(matched.element_type("T").unwrap(), None);
    assert_eq!(matched.element_type("U").unwrap(), Some(DataType::Float32));
}

#[test]
fn parser_expansion_and_search_limits_are_explicit() {
    let nested = format!("(x)=>return {}x{}", "(".repeat(70), ")".repeat(70));
    assert_eq!(
        Pattern::parse_ort(&nested).unwrap_err().kind,
        PatternErrorKind::BudgetExceeded
    );
    let mut chain = "(x)=>{".to_string();
    let mut previous = "x".to_string();
    for i in 0..65 {
        let next = format!("v{i}");
        chain.push_str(&format!("{next}=Identity({previous});"));
        previous = next;
    }
    chain.push_str(&format!("return {previous}}}"));
    assert_eq!(
        Pattern::parse_ort(&chain).unwrap_err().kind,
        PatternErrorKind::BudgetExceeded
    );
    let mut definitions = "pattern P0(x){y=Identity(x);return y}".to_string();
    for i in 1..36 {
        definitions.push_str(&format!("pattern P{i}(x){{y=P{}(x);return y}}", i - 1));
    }
    assert_eq!(
        library(&definitions).unwrap_err().kind,
        PatternErrorKind::BudgetExceeded
    );
}

#[test]
fn whole_graph_extra_header_inputs_bind_structural_leaves_like_ort() {
    let pattern = library("pattern Body(x,w){return Add(x,w)}")
        .unwrap()
        .graph_pattern("Body")
        .unwrap();
    let mut graph = Graph::new();
    let x = input(&mut graph, "x");
    let weight = graph.create_value(DataType::Float32, vec![2.into()]);
    graph.set_initializer(
        weight,
        WeightRef::Inline(TensorData::from_raw(DataType::Float32, vec![2], vec![0; 8])),
    );
    let (_, output) = op(&mut graph, "Add", &[Some(x), Some(weight)], 1);
    graph.set_outputs(output);
    let session = MatchSession::new(&graph).unwrap();
    let matched = session
        .match_graph(&pattern, Default::default())
        .unwrap()
        .unwrap();
    assert_eq!(matched.value("w").unwrap().unwrap().id(), weight);
    let unbound = library("pattern Body(x,w,unused){return Add(x,w)}")
        .unwrap()
        .graph_pattern("Body")
        .unwrap();
    assert!(
        session
            .match_graph(&unbound, Default::default())
            .unwrap()
            .is_none()
    );
}

#[test]
fn repeated_sparse_attributes_compare_float_representation_bits() {
    let pattern =
        Pattern::parse_ort("(x,p)=>{a=Identity(x,value=p);return Identity(a,value=p)}").unwrap();
    for left in [0u32, 0x8000_0000, 0x7fc0_0000] {
        for right in [0u32, 0x8000_0000, 0x7fc0_0000] {
            let mut graph = Graph::new();
            let x = input(&mut graph, "x");
            let (first, a) = op(&mut graph, "Identity", &[Some(x)], 1);
            let (root, _) = op(&mut graph, "Identity", &[Some(a[0])], 1);
            for (node, bits) in [(first, left), (root, right)] {
                graph.node_mut(node).attributes.insert(
                    "value".into(),
                    Attribute::SparseTensor(SparseTensorData {
                        dims: vec![1],
                        values: TensorData::from_raw(
                            DataType::Float32,
                            vec![1],
                            bits.to_le_bytes().to_vec(),
                        ),
                        indices: TensorData::from_raw(DataType::Int64, vec![1], vec![0; 8]),
                    }),
                );
            }
            assert_eq!(
                MatchSession::new(&graph)
                    .unwrap()
                    .match_at(&pattern, root, Default::default())
                    .unwrap()
                    .is_some(),
                left == right
            );
        }
    }
}

#[test]
fn repeated_tensor_attributes_charge_string_names_and_sparse_dimensions() {
    let pattern =
        Pattern::parse_ort("(x,p)=>{a=Identity(x,value=p);return Identity(a,value=p)}").unwrap();
    let string = TensorData {
        name: None,
        dtype: DataType::String,
        dims: vec![1],
        data: vec![],
        strings: vec![vec![b'x'; 512]],
    };
    let mut named = TensorData::from_raw(DataType::Float32, vec![0], vec![]);
    named.name = Some("n".repeat(512));
    for attribute in [
        Attribute::Tensor(string.clone()),
        Attribute::Tensors(vec![string]),
        Attribute::Tensor(named),
        Attribute::SparseTensor(SparseTensorData {
            dims: vec![1; 32],
            values: TensorData::from_raw(DataType::Float32, vec![1], vec![0; 4]),
            indices: TensorData::from_raw(DataType::Int64, vec![1], vec![0; 8]),
        }),
    ] {
        let mut graph = Graph::new();
        let x = input(&mut graph, "x");
        let (first, a) = op(&mut graph, "Identity", &[Some(x)], 1);
        let (root, _) = op(&mut graph, "Identity", &[Some(a[0])], 1);
        for node in [first, root] {
            graph
                .node_mut(node)
                .attributes
                .insert("value".into(), attribute.clone());
        }
        let session = MatchSession::new(&graph).unwrap();
        assert!(
            session
                .match_at(&pattern, root, Default::default())
                .unwrap()
                .is_some()
        );
        assert_eq!(
            session
                .match_at(
                    &pattern,
                    root,
                    MatchOptions {
                        budget: MatchBudget {
                            tensor_bytes: 64,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .unwrap_err()
                .kind,
            PatternErrorKind::BudgetExceeded
        );
    }
}

#[test]
fn literal_attribute_comparisons_obey_the_byte_budget() {
    for (literal, attribute) in [
        ("'abc'", Attribute::String(b"abc".to_vec())),
        ("[1,2]", Attribute::Ints(vec![1, 2])),
        ("[1.0,2.0]", Attribute::Floats(vec![1.0, 2.0])),
    ] {
        let pattern =
            Pattern::parse_ort(&format!("(x)=>return Identity(x,value={literal})")).unwrap();
        let mut graph = Graph::new();
        let x = input(&mut graph, "x");
        let (root, _) = op(&mut graph, "Identity", &[Some(x)], 1);
        graph
            .node_mut(root)
            .attributes
            .insert("value".into(), attribute);
        let session = MatchSession::new(&graph).unwrap();
        assert!(
            session
                .match_at(&pattern, root, Default::default())
                .unwrap()
                .is_some()
        );
        assert_eq!(
            session
                .match_at(
                    &pattern,
                    root,
                    MatchOptions {
                        budget: MatchBudget {
                            tensor_bytes: 0,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                )
                .unwrap_err()
                .kind,
            PatternErrorKind::BudgetExceeded
        );
    }
}
