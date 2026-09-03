//! Per-operator unit tests driving each rule through the single-node public API
//! ([`InferenceRegistry::infer_node`]). Covers concrete dims, symbolic dims,
//! broadcasting edge cases, and shape-data propagation.

use std::collections::HashMap;

use onnx_runtime_ir::{Attribute, DataType, Node, NodeId, SymbolId, TensorData, ValueId};
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, MergePolicy, NodeIo, ShapeData, ShapeInferError, SymbolInterner,
    TensorType, TypeInfo, ValueType,
};

// --- construction helpers -------------------------------------------------

fn c(n: i64) -> DimExpr {
    DimExpr::constant(n)
}

fn assert_invalid(error: ShapeInferError, op: &str, expected_detail: &str) {
    assert!(
        matches!(
            &error,
            ShapeInferError::Invalid { op: actual_op, detail }
                if actual_op == op && detail.contains(expected_detail)
        ),
        "expected Invalid {op:?} error containing {expected_detail:?}, got {error:?}"
    );
    assert!(error.to_string().contains(expected_detail));
}

#[test]
fn constant_variants_and_constant_of_shape_static_validation() {
    let int = run(
        &with_attr(node("Constant", 0, 1), "value_int", Attribute::Int(9)),
        vec![],
        13,
    );
    assert_eq!(out_shape(&int), Vec::<DimExpr>::new());
    assert_eq!(int[0].shape_data.as_ref().unwrap().elems, vec![c(9)]);

    let floats = run(
        &with_attr(
            node("Constant", 0, 1),
            "value_floats",
            Attribute::Floats(vec![1.0, 2.0]),
        ),
        vec![],
        13,
    );
    assert_eq!(out_dtype(&floats), DataType::Float32);
    assert_eq!(out_shape(&floats), vec![c(2)]);

    let constant_of_shape = node("ConstantOfShape", 1, 1);
    assert!(
        try_run(
            &constant_of_shape,
            vec![tin(DataType::Int64, vec![c(1), c(1)])],
            13,
        )
        .is_err()
    );
    assert!(try_run(&constant_of_shape, vec![sd_vec(vec![c(-1)])], 13,).is_err());
}

#[test]
fn quantization_static_metadata_rejects_invalid_scalar_block_and_dtype_cases() {
    assert!(
        try_run(
            &node("QuantizeLinear", 2, 1),
            vec![f32in(vec![]), f32in(vec![c(1)])],
            13,
        )
        .is_err()
    );

    let invalid_dtype = with_attr(
        node("QuantizeLinear", 2, 1),
        "output_dtype",
        Attribute::Int(-1),
    );
    assert!(try_run(&invalid_dtype, vec![f32in(vec![c(2)]), f32in(vec![])], 21,).is_err());

    let blocked = with_attr(
        node("QuantizeLinear", 3, 1),
        "block_size",
        Attribute::Int(2),
    );
    assert!(
        try_run(
            &blocked,
            vec![
                f32in(vec![c(2), c(4)]),
                f32in(vec![c(2)]),
                tin(DataType::Uint8, vec![c(2), c(2)]),
            ],
            21,
        )
        .is_err()
    );
    assert!(
        try_run(
            &blocked,
            vec![
                f32in(vec![c(2), c(4)]),
                f32in(vec![c(2), c(2)]),
                tin(DataType::Uint8, vec![c(2)]),
            ],
            21,
        )
        .is_err()
    );
    assert!(
        try_run(
            &blocked,
            vec![
                f32in(vec![c(2), c(4)]),
                f32in(vec![c(2), c(3)]),
                tin(DataType::Uint8, vec![c(2), c(3)]),
            ],
            21,
        )
        .is_err()
    );
}

fn sym(n: u32) -> DimExpr {
    DimExpr::symbol(SymbolId(n))
}

#[test]
fn gather_elementwise_and_nd_validate_rank_and_batch_depth() {
    let elements = with_attr(node("GatherElements", 2, 1), "axis", Attribute::Int(-1));
    let out = run(
        &elements,
        vec![
            f32in(vec![c(2), c(3)]),
            tin(DataType::Int64, vec![c(2), c(4)]),
        ],
        13,
    );
    assert_eq!(out_shape(&out), vec![c(2), c(4)]);
    assert!(
        try_run(
            &elements,
            vec![f32in(vec![c(2), c(3)]), tin(DataType::Int64, vec![c(2)])],
            13,
        )
        .is_err()
    );

    let gather_nd = with_attr(node("GatherND", 2, 1), "batch_dims", Attribute::Int(1));
    let out = run(
        &gather_nd,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2), c(5), c(1)]),
        ],
        13,
    );
    assert_eq!(out_shape(&out), vec![c(2), c(5), c(4)]);
    assert!(
        try_run(
            &gather_nd,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Int64, vec![c(2), c(5), c(2)]),
            ],
            13,
        )
        .is_err()
    );
}

/// A typed input with the given dtype and dims.
fn tin(dt: DataType, dims: Vec<DimExpr>) -> NodeIo {
    NodeIo::typed(TypeInfo::new(dt, dims))
}

/// A float32 input.
fn f32in(dims: Vec<DimExpr>) -> NodeIo {
    tin(DataType::Float32, dims)
}

/// An input carrying a resolved int64 shape-data vector.
fn sd_vec(elems: Vec<DimExpr>) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(DataType::Int64, vec![c(elems.len() as i64)])),
        shape_data: Some(ShapeData::vector(DataType::Int64, elems)),
        value_type: None,
    }
}

/// An input carrying a resolved integer scalar.
fn sd_int_scalar(dtype: DataType, value: DimExpr) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(dtype, vec![])),
        shape_data: Some(ShapeData::scalar(dtype, value)),
        value_type: None,
    }
}

/// A scalar input carrying a resolved floating-point constant.
fn sd_float_scalar(dt: DataType, value: f64) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(dt, vec![])),
        shape_data: Some(ShapeData::float_scalar(dt, value)),
        value_type: None,
    }
}

fn sd_float_vec(values: Vec<f64>) -> NodeIo {
    NodeIo {
        type_info: Some(TypeInfo::new(
            DataType::Float32,
            vec![c(values.len() as i64)],
        )),
        shape_data: Some(ShapeData::float_vector(DataType::Float32, values)),
        value_type: None,
    }
}

fn node(op: &str, n_in: usize, n_out: usize) -> Node {
    Node::new(
        NodeId(0),
        op,
        vec![Some(ValueId(0)); n_in],
        (0..n_out).map(|i| ValueId(i as u32)).collect(),
    )
}

fn with_attr(mut n: Node, name: &str, attr: Attribute) -> Node {
    n.attributes.insert(name.to_string(), attr);
    n
}

fn with_domain(mut n: Node, domain: &str) -> Node {
    n.domain = domain.to_string();
    n
}

fn with_version(mut n: Node, version: i64) -> Node {
    n.version = Some(version);
    n
}

fn run(n: &Node, inputs: Vec<NodeIo>, opset: u64) -> Vec<NodeIo> {
    try_run(n, inputs, opset).unwrap()
}

fn try_run(n: &Node, inputs: Vec<NodeIo>, opset: u64) -> Result<Vec<NodeIo>, ShapeInferError> {
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), opset);
    let mut interner = SymbolInterner::new(0x8000_0000);
    reg.infer_node(n, &imports, inputs, MergePolicy::Permissive, &mut interner)
}

/// The resolved output shape of slot 0.
fn out_shape(outs: &[NodeIo]) -> Vec<DimExpr> {
    outs[0]
        .type_info
        .as_ref()
        .expect("output type resolved")
        .shape
        .clone()
}

fn out_dtype(outs: &[NodeIo]) -> DataType {
    outs[0].type_info.as_ref().unwrap().dtype
}

fn assert_symbolic(dim: &DimExpr) {
    assert!(
        dim.as_const().is_none(),
        "expected symbolic dim, got {dim:?}"
    );
    assert!(
        dim.as_symbol().is_some(),
        "expected fresh symbol, got {dim:?}"
    );
}

#[test]
fn expanded_registry_catalog_count_is_pinned() {
    let registry = InferenceRegistry::default_registry();
    assert_eq!(
        registry.operator_count(),
        221,
        "shape-inference operator count moved: `left` is the live registry, `right` is this \
         pin. If you added or removed a handler, repin to `left` in the same commit and cover \
         the rule with a test (RULES.md §8); if you did not, a registration changed by \
         accident. A stale pin reds `Fast (Linux x86_64)` -- a required check -- on main and \
         on every open PR, so it is worth 30 seconds now."
    );
    assert_eq!(
        registry.entry_count(),
        266,
        "shape-inference entry count moved: `left` is the live registry, `right` is this pin. \
         One operator can carry several opset-versioned entries, so this count does not always \
         move in step with the operator count -- read both numbers off the failure rather than \
         assuming they shift together."
    );
}

fn recurrent_node(op: &str, outputs: usize, direction: &str, hidden_size: i64) -> Node {
    with_attr(
        with_attr(
            node(op, 3, outputs),
            "direction",
            Attribute::String(direction.as_bytes().to_vec()),
        ),
        "hidden_size",
        Attribute::Int(hidden_size),
    )
}

#[test]
fn recurrent_forward_and_bidirectional_shapes_cover_all_three_ops() {
    for op in ["RNN", "GRU", "LSTM"] {
        let output_count = if op == "LSTM" { 3 } else { 2 };
        for (direction, directions) in [("forward", 1), ("bidirectional", 2)] {
            let outputs = run(
                &recurrent_node(op, output_count, direction, 11),
                vec![f32in(vec![c(5), c(3), c(7)])],
                13,
            );
            assert_eq!(
                out_shape(&outputs),
                vec![c(5), c(directions), c(3), c(11)],
                "{op} {direction} Y"
            );
            assert_eq!(
                outputs[1].type_info.as_ref().unwrap().shape,
                vec![c(directions), c(3), c(11)],
                "{op} {direction} Y_h"
            );
            assert_eq!(outputs.len(), output_count, "{op} output arity");
            if op == "LSTM" {
                assert_eq!(
                    outputs[2].type_info.as_ref().unwrap().shape,
                    vec![c(directions), c(3), c(11)],
                    "LSTM {direction} Y_c"
                );
            }
        }
    }
}

#[test]
fn recurrent_symbolic_sequence_and_batch_dims_propagate() {
    let outputs = run(
        &recurrent_node("LSTM", 3, "bidirectional", 13),
        vec![f32in(vec![sym(80), sym(81), c(17)])],
        13,
    );
    assert_eq!(out_shape(&outputs), vec![sym(80), c(2), sym(81), c(13)]);
    assert_eq!(
        outputs[1].type_info.as_ref().unwrap().shape,
        vec![c(2), sym(81), c(13)]
    );
    assert_eq!(
        outputs[2].type_info.as_ref().unwrap().shape,
        vec![c(2), sym(81), c(13)]
    );
}

#[test]
fn recurrent_opset_14_layout_controls_axis_order() {
    for op in ["RNN", "GRU", "LSTM"] {
        let output_count = if op == "LSTM" { 3 } else { 2 };
        let batch_major = with_attr(
            recurrent_node(op, output_count, "bidirectional", 19),
            "layout",
            Attribute::Int(1),
        );
        let outputs = run(&batch_major, vec![f32in(vec![sym(82), sym(83), c(23)])], 14);
        assert_eq!(
            out_shape(&outputs),
            vec![sym(82), sym(83), c(2), c(19)],
            "{op} layout=1 Y"
        );
        assert_eq!(
            outputs[1].type_info.as_ref().unwrap().shape,
            vec![sym(82), c(2), c(19)],
            "{op} layout=1 Y_h"
        );

        let sequence_major = with_attr(
            recurrent_node(op, output_count, "forward", 19),
            "layout",
            Attribute::Int(0),
        );
        let outputs = run(
            &sequence_major,
            vec![f32in(vec![sym(83), sym(82), c(23)])],
            14,
        );
        assert_eq!(
            out_shape(&outputs),
            vec![sym(83), c(1), sym(82), c(19)],
            "{op} layout=0 Y"
        );
    }
}

#[test]
fn recurrent_pre_14_ignores_layout_attribute() {
    let node = with_attr(
        recurrent_node("GRU", 2, "forward", 29),
        "layout",
        Attribute::Int(1),
    );
    let outputs = run(&node, vec![f32in(vec![sym(84), sym(85), c(31)])], 13);
    assert_eq!(out_shape(&outputs), vec![sym(84), c(1), sym(85), c(29)]);
    assert_eq!(
        outputs[1].type_info.as_ref().unwrap().shape,
        vec![c(1), sym(85), c(29)]
    );
}

#[test]
fn recurrent_missing_hidden_size_is_permissive() {
    for op in ["RNN", "GRU", "LSTM"] {
        let outputs = run(&node(op, 3, 3), vec![f32in(vec![c(5), c(3), c(7)])], 14);
        assert!(
            outputs.iter().all(|output| output.type_info.is_none()),
            "{op}"
        );
    }
}

#[test]
fn recurrent_only_sets_declared_outputs() {
    for op in ["RNN", "GRU", "LSTM"] {
        for output_count in 1..=if op == "LSTM" { 3 } else { 2 } {
            let outputs = run(
                &recurrent_node(op, output_count, "forward", 37),
                vec![f32in(vec![c(5), c(3), c(7)])],
                14,
            );
            assert_eq!(outputs.len(), output_count, "{op}");
            assert!(outputs.iter().all(|output| output.type_info.is_some()));
        }
    }

    for op in ["RNN", "GRU"] {
        let outputs = run(
            &recurrent_node(op, 3, "forward", 37),
            vec![f32in(vec![c(5), c(3), c(7)])],
            14,
        );
        assert!(outputs[0].type_info.is_some(), "{op} Y");
        assert!(outputs[1].type_info.is_some(), "{op} Y_h");
        assert!(outputs[2].type_info.is_none(), "{op} has no Y_c");
    }
}

#[test]
fn recurrent_reverse_direction_is_unidirectional() {
    // `reverse` runs a single pass backwards, so num_directions == 1 just like
    // `forward` — only `bidirectional` yields 2.
    for op in ["RNN", "GRU", "LSTM"] {
        let output_count = if op == "LSTM" { 3 } else { 2 };
        let outputs = run(
            &recurrent_node(op, output_count, "reverse", 11),
            vec![f32in(vec![sym(90), sym(91), c(7)])],
            14,
        );
        assert_eq!(
            out_shape(&outputs),
            vec![sym(90), c(1), sym(91), c(11)],
            "{op} reverse Y"
        );
        assert_eq!(
            outputs[1].type_info.as_ref().unwrap().shape,
            vec![c(1), sym(91), c(11)],
            "{op} reverse Y_h"
        );
    }
}

#[test]
fn recurrent_unknown_or_non_rank3_input_stays_permissive() {
    // X absent, unknown-rank (represented by an absent type), or a rank other
    // than 3 cannot pin Y/Y_h/Y_c: the rule must leave every output unresolved
    // rather than fabricate a shape or panic.
    for op in ["RNN", "GRU", "LSTM"] {
        let output_count = if op == "LSTM" { 3 } else { 2 };

        // No input type at all (unknown rank / absent producer).
        let missing = run(
            &recurrent_node(op, output_count, "forward", 11),
            vec![NodeIo::default()],
            14,
        );
        assert!(
            missing.iter().all(|o| o.type_info.is_none()),
            "{op} unknown X"
        );

        // Wrong rank (X must be rank 3: [seq, batch, input_size]).
        for bad in [vec![sym(92), c(7)], vec![sym(92), sym(93), c(7), c(2)]] {
            let outputs = run(
                &recurrent_node(op, output_count, "forward", 11),
                vec![f32in(bad.clone())],
                14,
            );
            assert!(
                outputs.iter().all(|o| o.type_info.is_none()),
                "{op} rank-{} X",
                bad.len()
            );
        }
    }
}

#[test]
fn recurrent_invalid_direction_is_permissive() {
    // An unrecognised `direction` string is not a hard error: leave outputs
    // unresolved rather than guessing num_directions.
    let outputs = run(
        &recurrent_node("LSTM", 3, "sideways", 11),
        vec![f32in(vec![c(5), c(3), c(7)])],
        14,
    );
    assert!(outputs.iter().all(|o| o.type_info.is_none()));
}

#[test]
fn recurrent_optional_inputs_do_not_change_output_shape() {
    // Providing the optional W/R/B/sequence_lens/initial_h(/initial_c) inputs
    // must not perturb the tensor-only output rule, which is driven purely by
    // X's [seq, batch, *] and the hidden_size attr.
    for op in ["RNN", "GRU", "LSTM"] {
        let output_count = if op == "LSTM" { 3 } else { 2 };
        // X plus five trailing optional inputs (contents irrelevant to shape).
        let mut node = recurrent_node(op, output_count, "bidirectional", 13);
        node.inputs = vec![Some(ValueId(0)); 6];
        let mut inputs = vec![f32in(vec![sym(94), sym(95), c(7)])];
        inputs.resize(6, f32in(vec![c(1)]));
        let outputs = run(&node, inputs, 14);
        assert_eq!(
            out_shape(&outputs),
            vec![sym(94), c(2), sym(95), c(13)],
            "{op} Y with optional inputs"
        );
    }
}

#[test]
fn recurrent_layout_attribute_is_inert_before_opset_14() {
    // The `layout` attribute is only defined from opset 14. At the v1 rule
    // boundary (opset 13) it must be ignored even when present, and honoured at
    // exactly opset 14.
    let batch_major = with_attr(
        recurrent_node("LSTM", 3, "forward", 17),
        "layout",
        Attribute::Int(1),
    );
    // Opset 13: sequence-major regardless of layout=1.
    let at_13 = run(&batch_major, vec![f32in(vec![sym(96), sym(97), c(7)])], 13);
    assert_eq!(out_shape(&at_13), vec![sym(96), c(1), sym(97), c(17)]);
    // Opset 14: layout=1 becomes batch-major.
    let at_14 = run(&batch_major, vec![f32in(vec![sym(96), sym(97), c(7)])], 14);
    assert_eq!(out_shape(&at_14), vec![sym(96), sym(97), c(1), c(17)]);
}

#[test]
fn ml_tensor_rules_preserve_symbolic_shapes_and_gate_versions() {
    let ml = |op: &str, inputs: Vec<NodeIo>, version| {
        run(
            &with_version(
                with_domain(node(op, inputs.len(), 1), "ai.onnx.ml"),
                version,
            ),
            inputs,
            25,
        )
    };

    for op in ["Binarizer", "Imputer"] {
        let output = ml(op, vec![tin(DataType::Int32, vec![sym(71), c(4)])], 1);
        assert_eq!(out_dtype(&output), DataType::Int32, "{op}");
        assert_eq!(out_shape(&output), vec![sym(71), c(4)], "{op}");
    }
    for op in ["Normalizer", "Scaler"] {
        let output = ml(op, vec![tin(DataType::Int32, vec![sym(72), c(4)])], 1);
        assert_eq!(out_dtype(&output), DataType::Float32, "{op}");
        assert_eq!(out_shape(&output), vec![sym(72), c(4)], "{op}");
    }

    // StringNormalizer is a DEFAULT-domain (ai.onnx) string op since v10.
    let before = run(
        &node("StringNormalizer", 1, 1),
        vec![tin(DataType::String, vec![c(3)])],
        9,
    );
    assert!(before[0].type_info.is_none());
    let output = run(
        &node("StringNormalizer", 1, 1),
        vec![tin(DataType::String, vec![c(1), sym(73)])],
        10,
    );
    assert_eq!(out_shape(&output)[0], c(1));
    assert_symbolic(&out_shape(&output)[1]);
    // 1-D input: rank preserved, filtered extent is a fresh symbolic dim.
    let one_dim = run(
        &node("StringNormalizer", 1, 1),
        vec![tin(DataType::String, vec![sym(73)])],
        10,
    );
    assert_eq!(out_dtype(&one_dim), DataType::String);
    assert_symbolic(&out_shape(&one_dim)[0]);
    let invalid = node("StringNormalizer", 1, 1);
    assert!(try_run(&invalid, vec![tin(DataType::String, vec![])], 25).is_err());
}

#[test]
fn ml_mapping_and_feature_rules_propagate_shapes_and_types() {
    let category = with_attr(
        with_domain(node("CategoryMapper", 1, 1), "ai.onnx.ml"),
        "default_string",
        Attribute::String(b"unknown".to_vec()),
    );
    let output = run(
        &category,
        vec![tin(DataType::Int64, vec![sym(74), c(2)])],
        25,
    );
    assert_eq!(out_dtype(&output), DataType::String);
    assert_eq!(out_shape(&output), vec![sym(74), c(2)]);

    // LabelEncoder-1 chooses direction (and output dtype) by which default_*
    // attribute is set: default_int64 → int64 output (string→int64 encoding);
    // default_string → string output (int64→string decoding). classes_strings
    // is present in BOTH directions and must NOT drive the output dtype.
    let label_v1_str_to_int = with_attr(
        with_attr(
            with_version(with_domain(node("LabelEncoder", 1, 1), "ai.onnx.ml"), 1),
            "classes_strings",
            Attribute::Strings(vec![b"a".to_vec()]),
        ),
        "default_int64",
        Attribute::Int(-1),
    );
    assert_eq!(
        out_dtype(&run(
            &label_v1_str_to_int,
            vec![tin(DataType::String, vec![sym(75)])],
            25
        )),
        DataType::Int64
    );
    let label_v1_int_to_str = with_attr(
        with_attr(
            with_version(with_domain(node("LabelEncoder", 1, 1), "ai.onnx.ml"), 1),
            "classes_strings",
            Attribute::Strings(vec![b"a".to_vec()]),
        ),
        "default_string",
        Attribute::String(b"?".to_vec()),
    );
    assert_eq!(
        out_dtype(&run(
            &label_v1_int_to_str,
            vec![tin(DataType::Int64, vec![sym(75)])],
            25
        )),
        DataType::String
    );
    let label_v2 = with_attr(
        with_version(with_domain(node("LabelEncoder", 1, 1), "ai.onnx.ml"), 2),
        "values_floats",
        Attribute::Floats(vec![1.0]),
    );
    let output = run(
        &label_v2,
        vec![tin(DataType::String, vec![sym(76), c(2)])],
        25,
    );
    assert_eq!(out_dtype(&output), DataType::Float32);
    assert_eq!(out_shape(&output), vec![sym(76), c(2)]);
    let label_v4 = with_attr(
        with_version(with_domain(node("LabelEncoder", 1, 1), "ai.onnx.ml"), 4),
        "values_tensor",
        Attribute::Tensor(TensorData::from_raw(
            DataType::Int16,
            vec![1],
            9i16.to_le_bytes().to_vec(),
        )),
    );
    assert_eq!(
        out_dtype(&run(
            &label_v4,
            vec![tin(DataType::String, vec![sym(76)])],
            25
        )),
        DataType::Int16
    );

    let extractor = with_domain(node("ArrayFeatureExtractor", 2, 1), "ai.onnx.ml");
    let output = run(
        &extractor,
        vec![
            f32in(vec![sym(77), c(8)]),
            tin(DataType::Int64, vec![sym(78)]),
        ],
        25,
    );
    assert_eq!(out_shape(&output), vec![sym(77), sym(78)]);
    assert!(
        try_run(
            &extractor,
            vec![f32in(vec![c(2)]), tin(DataType::Int64, vec![c(1), c(1)])],
            25,
        )
        .is_err()
    );
}

#[test]
fn tfidf_vectorizer_replaces_sequence_extent_and_gates_opset() {
    // TfIdfVectorizer is a DEFAULT-domain (ai.onnx) op since v9.
    let vectorizer = with_attr(
        node("TfIdfVectorizer", 1, 1),
        "ngram_indexes",
        Attribute::Ints(vec![3, 0, 7]),
    );
    let unresolved = run(&vectorizer, vec![tin(DataType::String, vec![c(5)])], 8);
    assert!(unresolved[0].type_info.is_none());

    let one_dim = run(&vectorizer, vec![tin(DataType::String, vec![sym(79)])], 9);
    assert_eq!(out_dtype(&one_dim), DataType::Float32);
    assert_eq!(out_shape(&one_dim), vec![c(8)]);
    let two_dim = run(
        &vectorizer,
        vec![tin(DataType::Int64, vec![sym(80), c(5)])],
        9,
    );
    assert_eq!(out_shape(&two_dim), vec![sym(80), c(8)]);
    assert!(
        try_run(
            &vectorizer,
            vec![tin(DataType::String, vec![c(1), c(2), c(3)])],
            9,
        )
        .is_err()
    );
}

#[test]
fn kernel_gap_unary_rules_preserve_symbolic_shapes_and_versions() {
    for (op, since_version) in [
        ("Selu", 6),
        ("ThresholdedRelu", 10),
        ("Hardmax", 13),
        ("LpNormalization", 1),
        ("GroupNormalization", 18),
        ("BitwiseNot", 18),
    ] {
        let input = tin(DataType::Float32, vec![sym(1), c(4)]);
        let unresolved = run(&node(op, 1, 1), vec![input.clone()], since_version - 1);
        assert!(unresolved[0].type_info.is_none(), "{op}");
        let output = run(&node(op, 1, 1), vec![input], since_version);
        assert_eq!(out_shape(&output), vec![sym(1), c(4)], "{op}");
    }
}

#[test]
fn bitwise_and_prelu_broadcast_symbolic_dims() {
    for (op, opset, dtype) in [
        ("BitShift", 11, DataType::Uint32),
        ("BitwiseAnd", 18, DataType::Int32),
        ("BitwiseOr", 18, DataType::Int32),
        ("BitwiseXor", 18, DataType::Int32),
        ("PRelu", 16, DataType::Float32),
    ] {
        let output = run(
            &node(op, 2, 1),
            vec![
                tin(dtype, vec![sym(7), c(1), c(8)]),
                tin(dtype, vec![c(3), c(8)]),
            ],
            opset,
        );
        assert_eq!(out_shape(&output), vec![sym(7), c(3), c(8)], "{op}");
        assert!(
            run(
                &node(op, 2, 1),
                vec![tin(dtype, vec![c(2)]), tin(dtype, vec![c(2)])],
                opset - 1,
            )[0]
            .type_info
            .is_none()
        );
    }
}

#[test]
fn predicates_dropout_and_eye_like_resolve_all_outputs() {
    for (op, opset) in [("IsNaN", 9), ("IsInf", 10)] {
        let output = run(&node(op, 1, 1), vec![f32in(vec![sym(3), c(5)])], opset);
        assert_eq!(out_dtype(&output), DataType::Bool);
        assert_eq!(out_shape(&output), vec![sym(3), c(5)]);
    }

    for opset in [13, 22] {
        let output = run(
            &node("Dropout", 1, 2),
            vec![f32in(vec![sym(4), c(7)])],
            opset,
        );
        assert_eq!(out_shape(&output), vec![sym(4), c(7)]);
        assert_eq!(output[1].type_info.as_ref().unwrap().dtype, DataType::Bool);
        assert_eq!(
            output[1].type_info.as_ref().unwrap().shape,
            vec![sym(4), c(7)]
        );
    }
    assert_eq!(
        out_shape(&run(&node("Dropout", 1, 2), vec![f32in(vec![c(2)])], 12)),
        vec![c(2)]
    );

    let eye = with_attr(
        node("EyeLike", 1, 1),
        "dtype",
        Attribute::Int(DataType::Float16.to_onnx() as i64),
    );
    let output = run(&eye, vec![f32in(vec![sym(5), c(9)])], 9);
    assert_eq!(out_dtype(&output), DataType::Float16);
    assert_eq!(out_shape(&output), vec![sym(5), c(9)]);
    assert!(try_run(&eye, vec![f32in(vec![c(2), c(3), c(4)])], 9).is_err());
}

#[test]
fn unique_tracks_axis_and_flattened_inverse_lengths() {
    let flat = run(&node("Unique", 1, 4), vec![f32in(vec![sym(2), c(4)])], 11);
    assert_eq!(flat[0].type_info.as_ref().unwrap().shape.len(), 1);
    assert_eq!(
        flat[2].type_info.as_ref().unwrap().shape,
        vec![sym(2).mul(&c(4))]
    );
    for output in [&flat[0], &flat[1], &flat[3]] {
        let shape = &output.type_info.as_ref().unwrap().shape;
        assert_eq!(shape.len(), 1);
        assert_symbolic(&shape[0]);
    }

    let axis = with_attr(node("Unique", 1, 4), "axis", Attribute::Int(-1));
    let output = run(&axis, vec![f32in(vec![sym(8), c(6)])], 11);
    assert_eq!(output[0].type_info.as_ref().unwrap().shape[0], sym(8));
    assert_eq!(output[0].type_info.as_ref().unwrap().shape.len(), 2);
    assert_eq!(output[2].type_info.as_ref().unwrap().shape, vec![c(6)]);
    assert_symbolic(&output[0].type_info.as_ref().unwrap().shape[1]);
    assert_symbolic(&output[1].type_info.as_ref().unwrap().shape[0]);
    assert_symbolic(&output[3].type_info.as_ref().unwrap().shape[0]);
    assert!(
        try_run(
            &with_attr(node("Unique", 1, 4), "axis", Attribute::Int(2)),
            vec![f32in(vec![c(2), c(3)])],
            11,
        )
        .is_err()
    );
}

// --- MatMul ---------------------------------------------------------------

#[test]
fn matmul_2d() {
    let n = node("MatMul", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
}

#[test]
fn matmul_batched_symbolic() {
    // [N, 8, 64] @ [N, 64, 32] -> [N, 8, 32]
    let n = node("MatMul", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(64)]),
            f32in(vec![sym(0), c(64), c(32)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(32)]);
}

#[test]
fn matmul_broadcast_batch() {
    // [2,1,8,64] @ [64,32] -> [2,1,8,32]
    let n = node("MatMul", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(1), c(8), c(64)]),
            f32in(vec![c(64), c(32)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(1), c(8), c(32)]);
}

#[test]
fn matmul_1d_1d_scalar() {
    let n = node("MatMul", 2, 1);
    let outs = run(&n, vec![f32in(vec![c(5)]), f32in(vec![c(5)])], 13);
    assert_eq!(out_shape(&outs), Vec::<DimExpr>::new());
}

#[test]
fn matmul_contraction_mismatch_errors() {
    let n = node("MatMul", 2, 1);
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), 13u64);
    let mut interner = SymbolInterner::new(0x8000_0000);
    let res = reg.infer_node(
        &n,
        &imports,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(4), c(5)])],
        MergePolicy::Permissive,
        &mut interner,
    );
    assert!(res.is_err());
}

#[test]
fn qlinear_matmul_uses_matmul_shape_and_output_zero_point_dtype() {
    let n = node("QLinearMatMul", 8, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Uint8, vec![c(2), c(3)]),
            f32in(vec![]),
            tin(DataType::Uint8, vec![]),
            tin(DataType::Int8, vec![c(3), c(4)]),
            f32in(vec![c(4)]),
            tin(DataType::Int8, vec![c(4)]),
            f32in(vec![]),
            tin(DataType::Int8, vec![]),
        ],
        10,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int8);
}

#[test]
fn mod_broadcasts_and_preserves_dtype() {
    let n = node("Mod", 2, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Int64, vec![c(3), c(1)]),
            tin(DataType::Int64, vec![c(1), c(4)]),
        ],
        10,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

// --- Quantized matmul ------------------------------------------------------

fn quantized_matmul_node(op: &str, domain: &str, n_in: usize, n: i64) -> Node {
    with_attr(
        with_domain(node(op, n_in, 1), domain),
        "N",
        Attribute::Int(n),
    )
}

fn assert_quantized_matmul_shapes(n: &Node, n_in: usize) {
    let packed_inputs = || (1..n_in).map(|_| NodeIo::default());

    let outs = run(
        n,
        std::iter::once(tin(DataType::Float16, vec![c(1), sym(0), c(896)]))
            .chain(packed_inputs())
            .collect(),
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(1), sym(0), c(4864)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);

    let outs = run(
        n,
        std::iter::once(f32in(vec![sym(1), c(896)]))
            .chain(packed_inputs())
            .collect(),
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(4864)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn block_quantized_matmul_uses_n_and_preserves_leading_dims() {
    let n = quantized_matmul_node("BlockQuantizedMatMul", "pkg.nxrt", 2, 4864);
    assert_quantized_matmul_shapes(&n, 2);
}

#[test]
fn matmul_nbits_uses_n_and_preserves_leading_dims() {
    let n = quantized_matmul_node("MatMulNBits", "com.microsoft", 3, 4864);
    assert_quantized_matmul_shapes(&n, 3);
}

// --- Gemm -----------------------------------------------------------------

#[test]
fn gemm_transb() {
    // A [8, 64], B [32, 64] with transB=1 -> [8, 32]
    let n = with_attr(node("Gemm", 3, 1), "transB", Attribute::Int(1));
    let outs = run(
        &n,
        vec![
            f32in(vec![c(8), c(64)]),
            f32in(vec![c(32), c(64)]),
            f32in(vec![c(32)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

// --- FusedMatMul (com.microsoft) ------------------------------------------

/// A `com.microsoft::FusedMatMul` node with the given int attributes.
fn fused_matmul_node(attrs: &[(&str, i64)]) -> Node {
    let mut n = with_domain(node("FusedMatMul", 2, 1), "com.microsoft");
    for &(name, v) in attrs {
        n = with_attr(n, name, Attribute::Int(v));
    }
    n
}

#[test]
fn fused_matmul_transb() {
    // The exact case Chew cited: A [8,64] · B [32,64]^T -> [8,32]. The plain
    // matmul reuse produced the wrong [8,64]; the dedicated handler is correct.
    let n = fused_matmul_node(&[("transB", 1)]);
    let outs = run(
        &n,
        vec![f32in(vec![c(8), c(64)]), f32in(vec![c(32), c(64)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_transa() {
    // A supplied as [K, M] = [64, 8], transA=1 -> M=8; B [64, 32] -> [8, 32].
    let n = fused_matmul_node(&[("transA", 1)]);
    let outs = run(
        &n,
        vec![f32in(vec![c(64), c(8)]), f32in(vec![c(64), c(32)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_transa_and_transb() {
    // A [K,M]=[64,8] transA, B [N,K]=[32,64] transB -> [8, 32].
    let n = fused_matmul_node(&[("transA", 1), ("transB", 1)]);
    let outs = run(
        &n,
        vec![f32in(vec![c(64), c(8)]), f32in(vec![c(32), c(64)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_batched_transb() {
    // Batched: A [N,8,64] · B [N,32,64]^T -> [N,8,32] (symbolic batch preserved).
    let n = fused_matmul_node(&[("transB", 1)]);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(64)]),
            f32in(vec![sym(0), c(32), c(64)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(32)]);
}

#[test]
fn fused_matmul_plain_matches_matmul() {
    // With no transpose flags, FusedMatMul must equal plain MatMul.
    let n = fused_matmul_node(&[]);
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
}

#[test]
fn fused_matmul_alpha_is_shape_neutral() {
    // `alpha` scales values only; it must not affect the output shape.
    let mut n = fused_matmul_node(&[("transB", 1)]);
    n = with_attr(n, "alpha", Attribute::Float(2.0));
    let outs = run(
        &n,
        vec![f32in(vec![c(8), c(64)]), f32in(vec![c(32), c(64)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(8), c(32)]);
}

#[test]
fn fused_matmul_trans_batch_a_moves_leading_axis() {
    // transBatchA relocates the leading axis into the row (M) slot:
    // A [4, 2, 8] -> effective [2, 4, 8] (batch=2, M=4, K=8);
    // B [2, 8, 16] -> [2, 4, 16].
    let n = fused_matmul_node(&[("transBatchA", 1)]);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(4), c(2), c(8)]),
            f32in(vec![c(2), c(8), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4), c(16)]);
}

#[test]
fn fused_gemm_output_equals_matmul_shape() {
    // com.microsoft::FusedGemm = Relu(MatMul(A, B) + bias); output shape is the
    // plain MatMul shape (bias broadcasts, Relu is elementwise).
    let n = with_domain(node("FusedGemm", 3, 1), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3)]),
            f32in(vec![c(3), c(4)]),
            f32in(vec![c(4)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn fused_gemm_batched_symbolic_shape() {
    // Batched, symbolic leading dim carries through unchanged.
    let n = with_domain(node("FusedGemm", 3, 1), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(1), c(8), c(64)]),
            f32in(vec![c(64), c(32)]),
            f32in(vec![c(32)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(8), c(32)]);
}

#[test]
fn fused_attention_pretransposed_k_concrete() {
    // com.microsoft::FusedAttention with k_transposed=1: K is already
    // [batch, heads, head_dim, seq_k]. Output == MatMul(probs, V) shape =
    // Q's leading dims + [seq_q, head_dim_v].
    // Q [2,4,3,8], K^T [2,4,8,5], V [2,4,5,16] -> out [2,4,3,16].
    let n = with_attr(
        with_domain(node("FusedAttention", 3, 1), "com.microsoft"),
        "k_transposed",
        Attribute::Int(1),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(4), c(3), c(8)]),
            f32in(vec![c(2), c(4), c(8), c(5)]),
            f32in(vec![c(2), c(4), c(5), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4), c(3), c(16)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn fused_attention_internal_transpose_k_concrete() {
    // k_transposed unset/0: K is [batch, heads, seq_k, head_dim] and the rule
    // transposes its last two dims to form Kᵀ before the score MatMul.
    // Q [2,4,3,8], K [2,4,5,8], V [2,4,5,16] -> out [2,4,3,16].
    let n = with_domain(node("FusedAttention", 3, 1), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(4), c(3), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4), c(3), c(16)]);
}

#[test]
fn fused_attention_symbolic_batch_and_mask() {
    // Symbolic batch carries through; the optional 4th (mask) input is
    // shape-neutral. Q [B,4,S,8], K^T [B,4,8,S], V [B,4,S,8] -> out [B,4,S,8].
    let n = with_attr(
        with_domain(node("FusedAttention", 4, 1), "com.microsoft"),
        "k_transposed",
        Attribute::Int(1),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(1), c(4), sym(2), c(8)]),
            f32in(vec![sym(1), c(4), c(8), sym(2)]),
            f32in(vec![sym(1), c(4), sym(2), c(8)]),
            f32in(vec![sym(1), c(1), c(1), sym(2)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(4), sym(2), c(8)]);
}

#[test]
fn attention_4d_all_outputs_with_cache() {
    // Standard ai.onnx::Attention, 4D, with a past KV cache and all 4 outputs.
    // Q [2,4,3,8], K [2,4,5,8], V [2,4,5,16], past_key [2,4,7,8].
    // total_seq = 7 + 5 = 12.
    //   Y            = [2,4,3,16]
    //   present_key  = [2,4,12,8]
    //   present_value= [2,4,12,16]
    //   qk_matmul    = [2,4,3,12]
    let n = node("Attention", 5, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(4), c(3), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(8)]),
            f32in(vec![c(2), c(4), c(5), c(16)]),
            NodeIo::default(),                   // attn_mask (skipped)
            f32in(vec![c(2), c(4), c(7), c(8)]), // past_key
        ],
        23,
    );
    let shape_i = |i: usize| outs[i].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(shape_i(0), vec![c(2), c(4), c(3), c(16)]);
    assert_eq!(shape_i(1), vec![c(2), c(4), c(12), c(8)]);
    assert_eq!(shape_i(2), vec![c(2), c(4), c(12), c(16)]);
    assert_eq!(shape_i(3), vec![c(2), c(4), c(3), c(12)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn attention_3d_reshapes_hidden_by_num_heads() {
    // 3D inputs: Q [2,S,32] with q_num_heads=4 -> head_size=8; V [2,S,32]
    // with kv_num_heads=4 -> v_head_size=8. Y hidden = q_heads*v_head_size = 32.
    let n = with_attr(
        with_attr(node("Attention", 3, 1), "q_num_heads", Attribute::Int(4)),
        "kv_num_heads",
        Attribute::Int(4),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), sym(2), c(32)]),
            f32in(vec![c(2), sym(3), c(32)]),
            f32in(vec![c(2), sym(3), c(32)]),
        ],
        23,
    );
    assert_eq!(out_shape(&outs), vec![c(2), sym(2), c(32)]);
}

#[test]
fn attention_gqa_present_uses_kv_heads() {
    // GQA: q_heads=4, kv_heads=2. present_key/value carry kv_heads, not q_heads.
    // Q [1,4,S,8], K [1,2,S,8], V [1,2,S,8].
    let n = node("Attention", 3, 3);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(1), c(4), sym(2), c(8)]),
            f32in(vec![c(1), c(2), sym(2), c(8)]),
            f32in(vec![c(1), c(2), sym(2), c(8)]),
        ],
        23,
    );
    let shape_i = |i: usize| outs[i].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(shape_i(0), vec![c(1), c(4), sym(2), c(8)]);
    assert_eq!(shape_i(1), vec![c(1), c(2), sym(2), c(8)]);
    assert_eq!(shape_i(2), vec![c(1), c(2), sym(2), c(8)]);
}

#[test]
fn attention_resolves_for_opsets_23_through_26() {
    // The opset-23 rule serves model opsets 24, 25 and 26 too (the registry
    // resolves the highest `min_opset <= version`). Y is sized at every opset.
    let n = node("Attention", 3, 1);
    for opset in [23, 24, 25, 26] {
        let outs = run(
            &n,
            vec![
                f32in(vec![c(1), c(2), c(3), c(8)]),
                f32in(vec![c(1), c(2), c(5), c(8)]),
                f32in(vec![c(1), c(2), c(5), c(16)]),
            ],
            opset,
        );
        assert_eq!(
            out_shape(&outs),
            vec![c(1), c(2), c(3), c(16)],
            "Y shape wrong at opset {opset}"
        );
    }
}

#[test]
fn attention_opset24_nonpad_external_cache_no_past_concat() {
    // opset-24 external-cache path: `nonpad_kv_seqlen` (7th input) with no
    // past_key, so total_seq == kv_seq of K (no concat). All four outputs sized.
    // Q [1,2,3,8], K [1,2,5,8], V [1,2,5,16] -> total_seq = 5.
    let n = node("Attention", 7, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(1), c(2), c(3), c(8)]),
            f32in(vec![c(1), c(2), c(5), c(8)]),
            f32in(vec![c(1), c(2), c(5), c(16)]),
            NodeIo::default(),                // attn_mask (skipped)
            NodeIo::default(),                // past_key (absent)
            NodeIo::default(),                // past_value (absent)
            tin(DataType::Int64, vec![c(1)]), // nonpad_kv_seqlen
        ],
        24,
    );
    let shape_i = |i: usize| outs[i].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(shape_i(0), vec![c(1), c(2), c(3), c(16)]);
    assert_eq!(shape_i(1), vec![c(1), c(2), c(5), c(8)]);
    assert_eq!(shape_i(2), vec![c(1), c(2), c(5), c(16)]);
    assert_eq!(shape_i(3), vec![c(1), c(2), c(3), c(5)]);
}

// --- com.microsoft::Attention ---------------------------------------------

fn msft_attention_node(num_inputs: usize, num_outputs: usize, num_heads: i64) -> Node {
    with_attr(
        with_domain(node("Attention", num_inputs, num_outputs), "com.microsoft"),
        "num_heads",
        Attribute::Int(num_heads),
    )
}

#[test]
fn msft_attention_foundry_whisper_encoder_shape() {
    // Foundry Whisper-tiny encoder Attention: X [B,1500,384], packed projection
    // [384,1152] and 6 heads. The context preserves the model hidden width.
    let n = msft_attention_node(3, 1, 6);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(1), c(1500), c(384)]),
            f32in(vec![c(384), c(1152)]),
            f32in(vec![c(1152)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(1500), c(384)]);
}

#[test]
fn msft_attention_asymmetric_value_and_present_cache_shapes() {
    let n = with_attr(
        msft_attention_node(5, 2, 4),
        "qkv_hidden_sizes",
        Attribute::Ints(vec![32, 32, 64]),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]),
            f32in(vec![c(32), c(128)]),
            f32in(vec![c(128)]),
            NodeIo::default(),
            f32in(vec![c(2), c(2), c(4), c(7), c(8)]),
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![c(2), c(3), c(64)]);
    assert_eq!(shape_at(&outs, 1), vec![c(2), c(2), c(4), c(10), c(8)]);
}

#[test]
fn msft_attention_without_past_sets_precise_present_shape() {
    let mut n = with_attr(
        msft_attention_node(5, 2, 4),
        "qkv_hidden_sizes",
        Attribute::Ints(vec![32, 32, 64]),
    );
    n.inputs[4] = None;
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]),
            f32in(vec![c(32), c(128)]),
            f32in(vec![c(128)]),
            NodeIo::default(),
            NodeIo::default(),
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![c(2), c(3), c(64)]);
    // When `past` is absent the present cache holds exactly the current
    // sequence's keys and values, so `total_sequence == sequence`. This
    // framework's executor allocates the present-cache buffer FROM the inferred
    // shape, so it must be the full precise rank-5 shape
    // `(2, batch, num_heads, sequence, head_size)` -- an empty (scalar) shape
    // would under-allocate the buffer and break the kernel's cache write.
    // batch=2, num_heads=4, sequence=3, head_size = q_hidden(32) / num_heads(4) = 8.
    let present = outs[1]
        .type_info
        .as_ref()
        .expect("present cache dtype must be propagated when past is absent");
    assert_eq!(present.dtype, DataType::Float32);
    assert_eq!(shape_at(&outs, 1), vec![c(2), c(2), c(4), c(3), c(8)]);
}

#[test]
fn msft_attention_growth_with_symbolic_sequence_leaves_present_sequence_dynamic() {
    let n = with_attr(
        msft_attention_node(5, 2, 4),
        "qkv_hidden_sizes",
        Attribute::Ints(vec![32, 32, 64]),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), sym(1), c(32)]),
            f32in(vec![c(32), c(128)]),
            f32in(vec![c(128)]),
            NodeIo::default(),
            f32in(vec![c(2), c(2), c(4), c(7), c(8)]),
        ],
        1,
    );
    let present = shape_at(&outs, 1);
    assert_eq!(present[..3], [c(2), c(2), c(4)]);
    assert!(present[3].as_const().is_none());
    assert_eq!(present[4], c(8));
}

#[test]
fn msft_attention_shared_buffer_present_preserves_past_seq() {
    // With `past_present_share_buffer=1`, present and past alias the same
    // buffer, so ORT keeps the present cache the SAME shape as the past input:
    // dim 3 (the buffer's max_sequence_length) is preserved rather than grown.
    // past [2,2,4,7,8], sequence 3 => present stays [2,2,4,7,8] (not [..,10,..]).
    let n = with_attr(
        with_attr(
            msft_attention_node(5, 2, 4),
            "qkv_hidden_sizes",
            Attribute::Ints(vec![32, 32, 64]),
        ),
        "past_present_share_buffer",
        Attribute::Int(1),
    );
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]),
            f32in(vec![c(32), c(128)]),
            f32in(vec![c(128)]),
            NodeIo::default(),
            f32in(vec![c(2), c(2), c(4), c(7), c(8)]),
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![c(2), c(3), c(64)]);
    assert_eq!(shape_at(&outs, 1), vec![c(2), c(2), c(4), c(7), c(8)]);
}

#[test]
fn msft_attention_missing_num_heads_errors() {
    let n = with_domain(node("Attention", 3, 1), "com.microsoft");
    let err = try_run(
        &n,
        vec![
            f32in(vec![c(1), c(3), c(32)]),
            f32in(vec![c(32), c(96)]),
            f32in(vec![c(96)]),
        ],
        1,
    )
    .expect_err("missing num_heads must error");
    assert_invalid(err, "Attention", "num_heads");
}

// --- com.microsoft::MultiHeadAttention ------------------------------------

fn multi_head_attention_node(num_inputs: usize, num_outputs: usize, num_heads: i64) -> Node {
    with_attr(
        with_domain(
            node("MultiHeadAttention", num_inputs, num_outputs),
            "com.microsoft",
        ),
        "num_heads",
        Attribute::Int(num_heads),
    )
}

fn shape_at(outs: &[NodeIo], i: usize) -> Vec<DimExpr> {
    outs[i]
        .type_info
        .as_ref()
        .expect("output type resolved")
        .shape
        .clone()
}

#[test]
fn multi_head_attention_value_head_size_differs_from_query_key_head_size() {
    // The defining MHA case: value head size (16) != query/key head size (8).
    // num_heads=4. Q [2,3,32] -> qk_head_size = 32/4 = 8. V [2,5,64] ->
    // value_head_size = 64/4 = 16. No past cache, so total_sequence = 5.
    //
    // Cross-validated against onnxruntime 1.26.0 (CPUExecutionProvider): a real
    // MultiHeadAttention session on these exact inputs returns
    //   output        = (2, 3, 64)   # num_heads * value_head_size
    //   present_key   = (2, 4, 5, 8) # query/key head size
    //   present_value = (2, 4, 5, 16)# value head size
    // confirming present_key and present_value carry DIFFERENT head sizes.
    let n = multi_head_attention_node(3, 3, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]), // query
            f32in(vec![c(2), c(5), c(32)]), // key
            f32in(vec![c(2), c(5), c(64)]), // value
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![c(2), c(3), c(64)]);
    assert_eq!(shape_at(&outs, 1), vec![c(2), c(4), c(5), c(8)]);
    assert_eq!(shape_at(&outs, 2), vec![c(2), c(4), c(5), c(16)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn multi_head_attention_past_cache_extends_total_sequence() {
    // past_key/past_value (inputs 6 and 7) are rank 4 with past_sequence = 7.
    // kv_sequence = 5, so present total_sequence = 7 + 5 = 12. num_heads=4,
    // qk_head_size = value_head_size = 8 here (ORT requires them equal on the
    // past-cache path).
    let n = multi_head_attention_node(8, 3, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]),      // query
            f32in(vec![c(2), c(5), c(32)]),      // key
            f32in(vec![c(2), c(5), c(32)]),      // value
            NodeIo::default(),                   // bias
            NodeIo::default(),                   // key_padding_mask
            NodeIo::default(),                   // attention_bias
            f32in(vec![c(2), c(4), c(7), c(8)]), // past_key
            f32in(vec![c(2), c(4), c(7), c(8)]), // past_value
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![c(2), c(3), c(32)]);
    assert_eq!(shape_at(&outs, 1), vec![c(2), c(4), c(12), c(8)]);
    assert_eq!(shape_at(&outs, 2), vec![c(2), c(4), c(12), c(8)]);
}

#[test]
fn multi_head_attention_past_cache_keeps_respective_head_sizes() {
    // Even with a past cache and a differing value head size, present_key is
    // sized from Q/K (head size 8) and present_value from V (head size 16).
    // total_sequence = past(7) + kv(5) = 12.
    let n = multi_head_attention_node(8, 3, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]),       // query -> qk head size 8
            f32in(vec![c(2), c(5), c(32)]),       // key
            f32in(vec![c(2), c(5), c(64)]),       // value -> value head size 16
            NodeIo::default(),                    // bias
            NodeIo::default(),                    // key_padding_mask
            NodeIo::default(),                    // attention_bias
            f32in(vec![c(2), c(4), c(7), c(8)]),  // past_key
            f32in(vec![c(2), c(4), c(7), c(16)]), // past_value
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![c(2), c(3), c(64)]);
    assert_eq!(shape_at(&outs, 1), vec![c(2), c(4), c(12), c(8)]);
    assert_eq!(shape_at(&outs, 2), vec![c(2), c(4), c(12), c(16)]);
}

#[test]
fn multi_head_attention_rank_four_pretransposed_key_value() {
    // Rank-4 (already transposed) K/V: (batch, num_heads, kv_sequence,
    // head_size). Head sizes come straight from the last dim: qk_head_size = 8
    // from key, value_head_size = 16 from value. No past, so total = kv = 5.
    let n = multi_head_attention_node(3, 3, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(1), c(3), c(32)]),       // query
            f32in(vec![c(1), c(4), c(5), c(8)]),  // key  (rank 4)
            f32in(vec![c(1), c(4), c(5), c(16)]), // value (rank 4)
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![c(1), c(3), c(64)]);
    assert_eq!(shape_at(&outs, 1), vec![c(1), c(4), c(5), c(8)]);
    assert_eq!(shape_at(&outs, 2), vec![c(1), c(4), c(5), c(16)]);
}

#[test]
fn multi_head_attention_symbolic_sequences_and_optional_qk_output() {
    // Symbolic batch and sequences survive; the optional 4th output is the QK
    // attention-score tensor (batch, num_heads, query_sequence, total_sequence).
    let n = multi_head_attention_node(3, 4, 4);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(1), sym(2), c(32)]), // query
            f32in(vec![sym(1), sym(3), c(32)]), // key
            f32in(vec![sym(1), sym(3), c(64)]), // value
        ],
        1,
    );
    assert_eq!(shape_at(&outs, 0), vec![sym(1), sym(2), c(64)]);
    assert_eq!(shape_at(&outs, 1), vec![sym(1), c(4), sym(3), c(8)]);
    assert_eq!(shape_at(&outs, 2), vec![sym(1), c(4), sym(3), c(16)]);
    assert_eq!(shape_at(&outs, 3), vec![sym(1), c(4), sym(2), sym(3)]);
}

#[test]
fn multi_head_attention_missing_num_heads_errors() {
    // Without the required `num_heads` attribute the per-head split is
    // undefined; the rule must reject cleanly rather than emit a wrong shape.
    let n = with_domain(node("MultiHeadAttention", 3, 3), "com.microsoft");
    let err = try_run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]),
            f32in(vec![c(2), c(5), c(32)]),
            f32in(vec![c(2), c(5), c(64)]),
        ],
        1,
    )
    .expect_err("missing num_heads must error");
    assert_invalid(err, "MultiHeadAttention", "num_heads");
}

#[test]
fn multi_head_attention_rejects_unsupported_query_rank() {
    // Packed-QKV (rank-5 query) is not supported; reject rather than guess.
    let n = multi_head_attention_node(3, 1, 4);
    let err = try_run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4), c(3), c(8)]), // packed QKV query
            f32in(vec![c(2), c(5), c(32)]),
            f32in(vec![c(2), c(5), c(64)]),
        ],
        1,
    )
    .expect_err("rank-5 query must error");
    assert_invalid(err, "MultiHeadAttention", "query must be rank 3");
}

#[test]
fn multi_head_attention_rejects_key_value_rank_mismatch() {
    // Key and value must share a rank (both 3 or both 4).
    let n = multi_head_attention_node(3, 3, 4);
    let err = try_run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(32)]),
            f32in(vec![c(2), c(5), c(32)]),       // key rank 3
            f32in(vec![c(2), c(4), c(5), c(16)]), // value rank 4
        ],
        1,
    )
    .expect_err("rank mismatch must error");
    assert_invalid(err, "MultiHeadAttention", "key and value must both be rank");
}

#[test]
fn add_broadcast_concrete() {
    let n = node("Add", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(1)]), f32in(vec![c(1), c(4)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(4)]);
}

#[test]
fn add_broadcast_symbolic_batch() {
    // [N, 8, 768] + [768] -> [N, 8, 768]
    let n = node("Add", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8), c(768)]), f32in(vec![c(768)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
}

#[test]
fn add_symbolic_vs_concrete_prefers_concrete() {
    // broadcast(N, 8) -> 8 (the concrete non-1 extent wins)
    let n = node("Add", 2, 1);
    let outs = run(&n, vec![f32in(vec![sym(0)]), f32in(vec![c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![c(8)]);
}

#[test]
fn add_two_distinct_symbols_keeps_named_representative() {
    // Broadcasting a data-dependent/anonymous symbol (high-range id, as minted
    // by inference for an unresolved extent) against a named graph symbol
    // (low-range id) must re-unify onto the *named* one — never a fresh symbol
    // — so the session can bind it. This is the invariant that keeps a
    // `Shape`-driven `Expand`/`Add` chain resolvable end-to-end.
    let named = sym(1);
    let anon = sym(0x8000_0000);
    let n = node("Add", 2, 1);
    // Order-independent: named wins whether it is the left or the right operand.
    let outs = run(
        &n,
        vec![f32in(vec![anon.clone()]), f32in(vec![named.clone()])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![named.clone()]);
    let outs = run(&n, vec![f32in(vec![named.clone()]), f32in(vec![anon])], 13);
    assert_eq!(out_shape(&outs), vec![named]);
}

#[test]
fn div_strict_incompatible_broadcast_errors() {
    let n = node("Div", 2, 1);
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), 13u64);
    let mut interner = SymbolInterner::new(0x8000_0000);
    let res = reg.infer_node(
        &n,
        &imports,
        vec![f32in(vec![c(3)]), f32in(vec![c(4)])],
        MergePolicy::Strict,
        &mut interner,
    );
    assert!(res.is_err());
}

// --- unary ----------------------------------------------------------------

#[test]
fn relu_passthrough() {
    let n = node("Relu", 1, 1);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn microsoft_silu_uses_unary_shape_and_dtype_propagation() {
    let n = with_version(with_domain(node("Silu", 1, 1), "com.microsoft"), 1);

    let static_output = run(
        &n,
        vec![tin(DataType::Float16, vec![c(1), c(10240), c(4)])],
        1,
    );
    assert_eq!(out_shape(&static_output), vec![c(1), c(10240), c(4)]);
    assert_eq!(out_dtype(&static_output), DataType::Float16);

    let symbolic_output = run(
        &n,
        vec![tin(DataType::Float32, vec![sym(0), sym(1), c(128), sym(2)])],
        1,
    );
    assert_eq!(
        out_shape(&symbolic_output),
        vec![sym(0), sym(1), c(128), sym(2)]
    );
    assert_eq!(out_dtype(&symbolic_output), DataType::Float32);

    let unknown_rank_output = run(&n, vec![NodeIo::default()], 1);
    assert!(unknown_rank_output[0].type_info.is_none());
}

#[test]
fn microsoft_silu_is_registered_from_version_one() {
    let registry = InferenceRegistry::default_registry();
    assert!(registry.get("com.microsoft", "Silu", 0).is_none());
    assert!(registry.get("com.microsoft", "Silu", 1).is_some());
}

#[test]
fn round3_math_schemas_have_shape_rules() {
    let binary_inputs = vec![f32in(vec![sym(0), c(1), c(64)]), f32in(vec![c(8), c(64)])];
    for op in ["Sub", "Div", "Mod"] {
        let outs = run(&node(op, 2, 1), binary_inputs.clone(), 14);
        assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(64)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Float32, "{op}");
    }
    for op in ["Neg", "Abs"] {
        let outs = run(
            &node(op, 1, 1),
            vec![tin(DataType::Int32, vec![sym(0), c(64)])],
            13,
        );
        assert_eq!(out_shape(&outs), vec![sym(0), c(64)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Int32, "{op}");
    }
}

#[test]
fn acos_passthrough() {
    let n = node("Acos", 1, 1);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 7);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

// --- Selection ------------------------------------------------------------

#[test]
fn argmax_keepdims_variants_return_int64() {
    let input = f32in(vec![c(2), c(3), c(4)]);
    let keep = with_attr(node("ArgMax", 1, 1), "axis", Attribute::Int(1));
    let outs = run(&keep, vec![input.clone()], 13);
    assert_eq!(out_shape(&outs), vec![c(2), c(1), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);

    let drop = with_attr(
        with_attr(node("ArgMax", 1, 1), "axis", Attribute::Int(1)),
        "keepdims",
        Attribute::Int(0),
    );
    let outs = run(&drop, vec![input], 13);
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

#[test]
fn argmin_returns_int64() {
    let n = with_attr(node("ArgMin", 1, 1), "keepdims", Attribute::Int(0));
    let outs = run(&n, vec![f32in(vec![c(2), c(3)])], 12);
    assert_eq!(out_shape(&outs), vec![c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

#[test]
fn topk_outputs_and_dynamic_k() {
    let n = with_attr(node("TopK", 2, 2), "axis", Attribute::Int(-1));
    let outs = run(&n, vec![f32in(vec![c(2), c(8)]), sd_vec(vec![c(3)])], 11);
    assert_eq!(out_shape(&outs), vec![c(2), c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(3)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);

    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(8)]), tin(DataType::Int64, vec![])],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], c(2));
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);
}

#[test]
fn topk_v1_reads_k_attribute() {
    let n = with_attr(node("TopK", 1, 2), "k", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![c(3), c(8)])], 1);
    assert_eq!(out_shape(&outs), vec![c(3), c(2)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);
}

#[test]
fn topk_replaces_the_selected_middle_axis_and_rejects_invalid_axes() {
    let middle = with_attr(node("TopK", 2, 2), "axis", Attribute::Int(1));
    let outs = run(
        &middle,
        vec![f32in(vec![c(2), c(8), c(4)]), sd_vec(vec![c(3)])],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(2), c(3), c(4)]
    );

    let invalid = with_attr(node("TopK", 2, 2), "axis", Attribute::Int(3));
    assert!(
        try_run(
            &invalid,
            vec![f32in(vec![c(2), c(8), c(4)]), sd_vec(vec![c(3)])],
            11,
        )
        .is_err()
    );
}

#[test]
fn axis_operators_reject_out_of_range_and_duplicate_axes() {
    let input = f32in(vec![c(2), c(3), c(4)]);
    assert!(
        try_run(
            &with_attr(node("ArgMax", 1, 1), "axis", Attribute::Int(-4)),
            vec![input.clone()],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("Transpose", 1, 1),
                "perm",
                Attribute::Ints(vec![0, 1, 3]),
            ),
            vec![input.clone()],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("Transpose", 1, 1),
                "perm",
                Attribute::Ints(vec![0, 1, 1]),
            ),
            vec![input.clone()],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &node("Unsqueeze", 2, 1),
            vec![input.clone(), sd_vec(vec![c(4)])],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &node("Unsqueeze", 2, 1),
            vec![input.clone(), sd_vec(vec![c(0), c(0)])],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(node("Gather", 2, 1), "axis", Attribute::Int(3)),
            vec![input, tin(DataType::Int64, vec![])],
            13,
        )
        .is_err()
    );
}

#[test]
fn tile_static_repeats() {
    let n = node("Tile", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            sd_vec(vec![c(1), c(2), c(3)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(6), c(12)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn tile_unknown_repeats_keeps_rank() {
    // `repeats` has no shape-data (runtime-computed): every extent degrades to a
    // fresh symbol, but the rank stays == rank(input).
    let n = node("Tile", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), tin(DataType::Int64, vec![c(2)])],
        13,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 2);
    assert!(shape[0].as_symbol().is_some());
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn tile_rejects_non_vector_repeats_and_extent_overflow() {
    let n = node("Tile", 2, 1);
    assert!(
        try_run(
            &n,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Int64, vec![c(1), c(2)]),
            ],
            13,
        )
        .is_err()
    );
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(isize::MAX as i64)]), sd_vec(vec![c(2)]),],
            13,
        )
        .is_err()
    );
}

#[test]
fn range_static_and_dynamic() {
    let n = node("Range", 3, 1);
    let scalar = |value| NodeIo {
        type_info: Some(TypeInfo::new(DataType::Int64, vec![])),
        shape_data: Some(ShapeData::scalar(DataType::Int64, c(value))),
        value_type: None,
    };
    let outs = run(&n, vec![scalar(1), scalar(10), scalar(2)], 11);
    assert_eq!(out_shape(&outs), vec![c(5)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);

    let outs = run(
        &n,
        vec![
            tin(DataType::Int64, vec![]),
            tin(DataType::Int64, vec![]),
            tin(DataType::Int64, vec![]),
        ],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 1);
    assert!(shape[0].as_symbol().is_some());
}

#[test]
fn range_float_positive_delta() {
    // start=0.0, limit=1.0, delta=0.3 -> ceil(1.0 / 0.3) = ceil(3.33) = 4
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            sd_float_scalar(DataType::Float32, 0.0),
            sd_float_scalar(DataType::Float32, 1.0),
            sd_float_scalar(DataType::Float32, 0.3),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn range_float32_uses_cpu_kernel_arithmetic() {
    // Keep in sync with `float_range_count`: f32 arithmetic makes this 25,
    // whereas f64 arithmetic on this f32 round-trip yields 26.
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            sd_float_scalar(DataType::Float32, 0.0),
            sd_float_scalar(DataType::Float32, 1.0),
            sd_float_scalar(DataType::Float32, f64::from(0.04_f32)),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(25)]);
}

#[test]
fn range_float_negative_delta() {
    // start=10.0, limit=2.0, delta=-2.5 -> ceil(-8.0 / -2.5) = ceil(3.2) = 4
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            sd_float_scalar(DataType::Float64, 10.0),
            sd_float_scalar(DataType::Float64, 2.0),
            sd_float_scalar(DataType::Float64, -2.5),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float64);
}

#[test]
fn range_float64_rejects_two_to_the_63_length() {
    let n = node("Range", 3, 1);
    let error = try_run(
        &n,
        vec![
            sd_float_scalar(DataType::Float64, 0.0),
            sd_float_scalar(DataType::Float64, 2_f64.powi(63)),
            sd_float_scalar(DataType::Float64, 1.0),
        ],
        11,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

#[test]
fn range_float_dynamic() {
    // Non-constant float operands (typed but no shape-data) -> unknown length.
    let n = node("Range", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float32, vec![]),
            tin(DataType::Float32, vec![]),
            tin(DataType::Float32, vec![]),
        ],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 1);
    assert!(shape[0].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn cumsum_passthrough() {
    let n = node("CumSum", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8)]), tin(DataType::Int64, vec![])],
        14,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn squeeze_and_unsqueeze_static_axes_and_dynamic_axes() {
    let squeeze = node("Squeeze", 2, 1);
    let outs = run(
        &squeeze,
        vec![f32in(vec![c(2), c(1), c(4)]), sd_vec(vec![c(1)])],
        24,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);

    let dynamic_axes = run(
        &squeeze,
        vec![
            f32in(vec![c(2), c(1), c(4)]),
            tin(DataType::Int64, vec![sym(0)]),
        ],
        24,
    );
    assert!(dynamic_axes[0].type_info.is_none());

    let unsqueeze = node("Unsqueeze", 2, 1);
    let outs = run(
        &unsqueeze,
        vec![f32in(vec![c(2), c(4)]), sd_vec(vec![c(1)])],
        24,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(1), c(4)]);
    let dynamic_axes = run(
        &unsqueeze,
        vec![f32in(vec![c(2), c(4)]), tin(DataType::Int64, vec![sym(0)])],
        24,
    );
    assert!(dynamic_axes[0].type_info.is_none());
}

#[test]
fn nonzero_rank_and_dynamic_nnz() {
    let n = node("NonZero", 1, 1);
    let outs = run(&n, vec![f32in(vec![c(2), c(3), c(4)])], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape[0], c(3));
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Int64);
}

#[test]
fn one_hot_inserts_known_depth_at_axis_for_opsets_9_and_11() {
    for opset in [9, 11] {
        for (axis, expected) in [
            (0, vec![c(5), c(2), c(3)]),
            (1, vec![c(2), c(5), c(3)]),
            (-1, vec![c(2), c(3), c(5)]),
            (-2, vec![c(2), c(5), c(3)]),
        ] {
            let n = with_attr(node("OneHot", 3, 1), "axis", Attribute::Int(axis));
            let outs = run(
                &n,
                vec![
                    tin(DataType::Int64, vec![c(2), c(3)]),
                    sd_int_scalar(DataType::Int64, c(5)),
                    tin(DataType::Float16, vec![c(2)]),
                ],
                opset,
            );
            assert_eq!(out_shape(&outs), expected, "opset {opset}, axis {axis}");
            assert_eq!(out_dtype(&outs), DataType::Float16);
        }
    }
}

#[test]
fn one_hot_preserves_symbolic_indices_and_handles_dynamic_depth() {
    let n = node("OneHot", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Int32, vec![sym(0), c(3)]),
            tin(DataType::Int64, vec![]),
            tin(DataType::Int32, vec![c(2)]),
        ],
        11,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 3);
    assert_eq!(shape[0], sym(0));
    assert_eq!(shape[1], c(3));
    assert!(shape[2].as_symbol().is_some());
    assert_eq!(out_dtype(&outs), DataType::Int32);

    let outs = run(
        &n,
        vec![
            tin(DataType::Int32, vec![sym(0)]),
            sd_int_scalar(DataType::Int64, sym(7)),
            tin(DataType::Uint8, vec![c(2)]),
        ],
        11,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), sym(7)]);
    assert_eq!(out_dtype(&outs), DataType::Uint8);
}

#[test]
fn one_hot_rejects_invalid_axis_and_values_length() {
    let inputs = || {
        vec![
            tin(DataType::Int64, vec![c(2)]),
            sd_int_scalar(DataType::Int64, c(4)),
            tin(DataType::Float32, vec![c(2)]),
        ]
    };
    let n = with_attr(node("OneHot", 3, 1), "axis", Attribute::Int(2));
    assert!(try_run(&n, inputs(), 11).is_err());

    let n = node("OneHot", 3, 1);
    let mut bad_values = inputs();
    bad_values[2] = tin(DataType::Float32, vec![c(3)]);
    assert!(try_run(&n, bad_values, 11).is_err());
}

#[test]
fn compress_axis_and_flatten_variants_for_opsets_9_and_11() {
    for opset in [9, 11] {
        let n = with_attr(node("Compress", 2, 1), "axis", Attribute::Int(-2));
        let outs = run(
            &n,
            vec![
                tin(DataType::Float16, vec![sym(0), c(3), c(4)]),
                tin(DataType::Bool, vec![c(3)]),
            ],
            opset,
        );
        let shape = out_shape(&outs);
        assert_eq!(shape.len(), 3);
        assert_eq!(shape[0], sym(0));
        assert!(shape[1].as_symbol().is_some());
        assert_eq!(shape[2], c(4));
        assert_eq!(out_dtype(&outs), DataType::Float16);

        let n = node("Compress", 2, 1);
        let outs = run(
            &n,
            vec![
                tin(DataType::Int32, vec![c(2), c(3), c(4)]),
                tin(DataType::Bool, vec![sym(1)]),
            ],
            opset,
        );
        let shape = out_shape(&outs);
        assert_eq!(shape.len(), 1);
        assert!(shape[0].as_symbol().is_some());
        assert_eq!(out_dtype(&outs), DataType::Int32);
    }
}

#[test]
fn compress_rejects_invalid_axis_and_condition_rank() {
    let n = with_attr(node("Compress", 2, 1), "axis", Attribute::Int(2));
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), tin(DataType::Bool, vec![c(3)])],
            11
        )
        .is_err()
    );

    let n = node("Compress", 2, 1);
    assert!(
        try_run(
            &n,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Bool, vec![c(1), c(3)])
            ],
            11
        )
        .is_err()
    );
}

// --- Reshape (shape-data) -------------------------------------------------

#[test]
fn reshape_from_shape_data_with_minus_one() {
    // input [B, S, 768], target [0, 0, 12, -1] -> [B, S, 12, 64]
    let n = node("Reshape", 2, 1);
    let target = sd_vec(vec![c(0), c(0), c(12), c(-1)]);
    let outs = run(&n, vec![f32in(vec![sym(0), sym(1), c(768)]), target], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), sym(1), c(12), c(64)]);
}

#[test]
fn reshape_zero_copies_input_dim() {
    // input [4, 8, 16], target [0, -1] -> [4, 128]
    let n = node("Reshape", 2, 1);
    let target = sd_vec(vec![c(0), c(-1)]);
    let outs = run(&n, vec![f32in(vec![c(4), c(8), c(16)]), target], 13);
    assert_eq!(out_shape(&outs), vec![c(4), c(128)]);
}

#[test]
fn reshape_rejects_multiple_inferred_dimensions_and_product_mismatches() {
    let n = node("Reshape", 2, 1);

    // ONNX permits at most one -1 in the target shape.
    assert_invalid(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(-1), c(-1)])],
            13,
        )
        .unwrap_err(),
        "Reshape",
        "at most one dimension may be -1",
    );

    // A fully concrete target must preserve the element count.
    assert_invalid(
        try_run(&n, vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(4)])], 13).unwrap_err(),
        "Reshape",
        "input element count 6 does not match target element count 4",
    );
}

#[test]
fn reshape_validates_static_target_values_without_guessing_dynamic_targets() {
    let n = node("Reshape", 2, 1);
    assert_invalid(
        try_run(&n, vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(-2)])], 13).unwrap_err(),
        "Reshape",
        "target dimension -2 is invalid",
    );
    assert_invalid(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(-1), c(4)])],
            13,
        )
        .unwrap_err(),
        "Reshape",
        "input element count is not divisible",
    );
    assert_invalid(
        try_run(&n, vec![f32in(vec![c(2)]), sd_vec(vec![c(0), c(0)])], 13).unwrap_err(),
        "Reshape",
        "0 at target index 1 has no corresponding input dimension",
    );

    let allowzero = with_attr(node("Reshape", 2, 1), "allowzero", Attribute::Int(1));
    assert_invalid(
        try_run(
            &allowzero,
            vec![f32in(vec![c(0)]), sd_vec(vec![c(0), c(-1)])],
            14,
        )
        .unwrap_err(),
        "Reshape",
        "allowzero=1 does not permit 0 and -1",
    );
    let zero = run(&allowzero, vec![f32in(vec![c(0)]), sd_vec(vec![c(0)])], 14);
    assert_eq!(out_shape(&zero), vec![c(0)]);

    let dynamic = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), tin(DataType::Int64, vec![c(2)])],
        13,
    );
    assert!(dynamic[0].type_info.is_none());
}

#[test]
fn reshape_rejects_indeterminate_minus_one_with_zero_product() {
    let error = try_run(
        &node("Reshape", 2, 1),
        vec![f32in(vec![c(0), c(3)]), sd_vec(vec![c(0), c(-1)])],
        13,
    )
    .unwrap_err();
    assert_invalid(
        error,
        "Reshape",
        "cannot infer -1 dimension when the remaining target product is zero",
    );
}

#[test]
fn reshape_symbolic_target_dim() {
    // target carries a symbolic dim (batch read from a Shape op)
    let n = node("Reshape", 2, 1);
    let target = sd_vec(vec![sym(0), c(-1)]);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(16)]), target], 13);
    // -1 = (N*8*16)/N = 128
    assert_eq!(out_shape(&outs), vec![sym(0), c(128)]);
}

#[test]
fn reshape_overflowing_total_degrades_to_symbol() {
    // Regression (Holden): an input whose concrete element count is 2^80
    // overflows i64. The inferred `-1` dim must degrade to a fresh symbol, not
    // panic (debug) and not wrap to a bogus static 0 (release).
    let n = node("Reshape", 2, 1);
    let big = c(1 << 20);
    let target = sd_vec(vec![c(-1)]);
    let outs = run(
        &n,
        vec![
            f32in(vec![big.clone(), big.clone(), big.clone(), big]),
            target,
        ],
        13,
    );
    let out = out_shape(&outs);
    assert_eq!(out.len(), 1);
    // Fresh symbol (anon range), never a concrete 0 or negative dim.
    assert_eq!(out[0].as_const(), None);
    assert!(out[0].as_symbol().is_some());
}

#[test]
fn size_rejects_total_above_isize_max() {
    // A concrete tensor extent that cannot be represented by Rust indexing must
    // be rejected rather than wrapped or lowered to a bogus static dimension.
    let n = node("Size", 1, 1);
    let big = c(1 << 20);
    let error = try_run(
        &n,
        vec![f32in(vec![big.clone(), big.clone(), big.clone(), big])],
        13,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

// --- Transpose ------------------------------------------------------------

#[test]
fn transpose_perm() {
    let n = with_attr(
        node("Transpose", 1, 1),
        "perm",
        Attribute::Ints(vec![0, 2, 1, 3]),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(12), c(64)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(12), c(8), c(64)]);
}

#[test]
fn transpose_default_reverses() {
    let n = node("Transpose", 1, 1);
    let outs = run(&n, vec![f32in(vec![c(2), c(3), c(4)])], 13);
    assert_eq!(out_shape(&outs), vec![c(4), c(3), c(2)]);
}

#[test]
fn trilu_preserves_known_and_symbolic_shape_and_dtype() {
    for upper in [0, 1] {
        let n = with_attr(node("Trilu", 2, 1), "upper", Attribute::Int(upper));
        let outs = run(
            &n,
            vec![
                tin(DataType::Float16, vec![sym(0), c(3), c(4)]),
                sd_int_scalar(DataType::Int64, c(-1)),
            ],
            14,
        );
        assert_eq!(out_shape(&outs), vec![sym(0), c(3), c(4)]);
        assert_eq!(out_dtype(&outs), DataType::Float16);
    }
}

#[test]
fn depth_to_space_known_dims_and_modes_across_schema_versions() {
    let n = with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2));
    let outs = run(
        &n,
        vec![tin(DataType::Uint8, vec![c(2), c(12), c(5), c(7)])],
        1,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(10), c(14)]);
    assert_eq!(out_dtype(&outs), DataType::Uint8);

    for opset in [11, 13] {
        for mode in ["DCR", "CRD"] {
            let n = with_attr(
                with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2)),
                "mode",
                Attribute::String(mode.as_bytes().to_vec()),
            );
            let outs = run(
                &n,
                vec![tin(DataType::Uint8, vec![c(2), c(12), c(5), c(7)])],
                opset,
            );
            assert_eq!(out_shape(&outs), vec![c(2), c(3), c(10), c(14)]);
            assert_eq!(out_dtype(&outs), DataType::Uint8);
        }
    }
}

#[test]
fn depth_to_space_handles_symbolic_dims_without_panicking() {
    let n = with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![sym(0), sym(1), sym(2), c(7)])], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0], sym(0));
    assert!(shape[1].as_symbol().is_some());
    assert_eq!(shape[2].as_const(), None);
    assert_eq!(shape[3], c(14));
}

#[test]
fn depth_to_space_rejects_non_divisible_channels() {
    let n = with_attr(node("DepthToSpace", 1, 1), "blocksize", Attribute::Int(2));
    assert!(try_run(&n, vec![f32in(vec![c(1), c(10), c(4), c(4)])], 13).is_err());
}

#[test]
fn space_to_depth_known_dims_across_schema_versions() {
    for opset in [1, 13] {
        let n = with_attr(node("SpaceToDepth", 1, 1), "blocksize", Attribute::Int(2));
        let outs = run(
            &n,
            vec![tin(DataType::Int32, vec![c(2), c(3), c(10), c(14)])],
            opset,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(12), c(5), c(7)]);
        assert_eq!(out_dtype(&outs), DataType::Int32);
    }
}

#[test]
fn space_to_depth_handles_symbolic_dims_without_panicking() {
    let n = with_attr(node("SpaceToDepth", 1, 1), "blocksize", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![sym(0), sym(1), sym(2), c(14)])], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0], sym(0));
    assert_eq!(shape[1].as_const(), None);
    assert!(shape[2].as_symbol().is_some());
    assert_eq!(shape[3], c(7));
}

#[test]
fn space_to_depth_rejects_non_divisible_spatial_dims() {
    let n = with_attr(node("SpaceToDepth", 1, 1), "blocksize", Attribute::Int(2));
    assert!(try_run(&n, vec![f32in(vec![c(1), c(3), c(5), c(8)])], 13).is_err());
    assert!(try_run(&n, vec![f32in(vec![c(1), c(3), c(8), c(5)])], 13).is_err());
}

#[test]
fn spatial_rearrangements_reject_blocksize_square_overflow() {
    for op in ["DepthToSpace", "SpaceToDepth"] {
        let n = with_attr(node(op, 1, 1), "blocksize", Attribute::Int(i64::MAX));
        assert!(
            try_run(&n, vec![f32in(vec![c(1), c(4), c(4), c(4)])], 13).is_err(),
            "{op}"
        );
    }
}

// --- Gather ---------------------------------------------------------------

#[test]
fn gather_axis0_scalar_index() {
    // data [10, 768], scalar index -> [768]
    let n = node("Gather", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(10), c(768)]), tin(DataType::Int64, vec![])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(768)]);
}

#[test]
fn gather_shape_data_selects_dim() {
    // Shape of [N, 8, 768] gathered at index [0] -> shape-data [N]
    let shape_out = sd_vec(vec![sym(0), c(8), c(768)]);
    let idx = sd_vec(vec![c(0)]);
    let n = with_attr(node("Gather", 2, 1), "axis", Attribute::Int(0));
    let outs = run(&n, vec![shape_out, idx], 13);
    let sd = outs[0].shape_data.as_ref().expect("gather shape-data");
    assert_eq!(sd.elems, vec![sym(0)]);
}

#[test]
fn gather_nd_canonical_shape() {
    // data [2, 3, 4], indices [5, 2] -> [5, 4].
    let n = node("GatherND", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(5), c(2)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(5), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

// --- Scatter --------------------------------------------------------------

#[test]
fn scatter_nd_preserves_data_shape_and_dtype() {
    let n = node("ScatterND", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(5), c(2)]),
            tin(DataType::Float16, vec![c(5), c(4)]),
        ],
        18,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn scatter_elements_non_default_axis_preserves_data_shape() {
    let n = with_attr(node("ScatterElements", 3, 1), "axis", Attribute::Int(-2));
    let outs = run(
        &n,
        vec![
            tin(DataType::Int32, vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2), c(1), c(4)]),
            tin(DataType::Int32, vec![c(2), c(1), c(4)]),
        ],
        16,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Int32);
}

#[test]
fn scatter_deprecated_alias_preserves_data_shape_and_dtype() {
    let n = with_attr(node("Scatter", 3, 1), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![
            tin(DataType::Float64, vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2), c(1), c(4)]),
            tin(DataType::Float64, vec![c(2), c(1), c(4)]),
        ],
        9,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)]);
    assert_eq!(out_dtype(&outs), DataType::Float64);
}

#[test]
fn scatter_unknown_data_shape_leaves_output_unresolved() {
    for (op, opset) in [("Scatter", 9), ("ScatterElements", 16), ("ScatterND", 18)] {
        let n = node(op, 3, 1);
        let outs = run(
            &n,
            vec![
                NodeIo::default(),
                tin(DataType::Int64, vec![c(2), c(1)]),
                f32in(vec![c(2)]),
            ],
            opset,
        );
        assert!(outs[0].type_info.is_none(), "{op}");
    }
}

#[test]
fn scatter_rank_relations_are_validated() {
    let elements = node("ScatterElements", 3, 1);
    assert!(
        try_run(
            &elements,
            vec![
                f32in(vec![c(2), c(3)]),
                tin(DataType::Int64, vec![c(2)]),
                f32in(vec![c(2)]),
            ],
            18,
        )
        .is_err()
    );

    let nd = node("ScatterND", 3, 1);
    assert!(
        try_run(
            &nd,
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                tin(DataType::Int64, vec![c(5), c(2)]),
                f32in(vec![c(5)]),
            ],
            18,
        )
        .is_err()
    );
}

// --- Concat ---------------------------------------------------------------

#[test]
fn concat_sums_axis() {
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(2), c(5)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(8)]);
}

#[test]
fn concat_shape_data_builds_vector() {
    // Concat of scalars/vectors of dims -> a shape vector.
    let a = sd_vec(vec![sym(0)]);
    let b = sd_vec(vec![c(12), c(64)]);
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(0));
    let outs = run(&n, vec![a, b], 13);
    let sd = outs[0].shape_data.as_ref().expect("concat shape-data");
    assert_eq!(sd.elems, vec![sym(0), c(12), c(64)]);
}

#[test]
fn concat_dynamic_axis_is_unresolved_and_other_dims_must_match() {
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(-1));
    let outs = run(
        &n,
        vec![f32in(vec![c(2), sym(0)]), f32in(vec![c(2), c(5)])],
        13,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape[0], c(2));
    assert!(shape[1].as_symbol().is_some());

    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), f32in(vec![c(4), c(5)])],
            13,
        )
        .is_err()
    );
}

#[test]
fn concat_rejects_axis_sum_beyond_isize_max() {
    let n = with_attr(node("Concat", 2, 1), "axis", Attribute::Int(0));
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(isize::MAX as i64)]), f32in(vec![c(1)]),],
            13,
        )
        .is_err()
    );
}

#[test]
fn concat_symbolic_axis_rejects_overflowing_known_partial_and_stays_unresolved_normally() {
    let n = with_attr(node("Concat", 3, 1), "axis", Attribute::Int(0));
    let error = try_run(
        &n,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            f32in(vec![sym(0)]),
            f32in(vec![c(1)]),
        ],
        13,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));

    let outs = run(
        &n,
        vec![f32in(vec![c(2)]), f32in(vec![sym(0)]), f32in(vec![c(3)])],
        13,
    );
    let shape = out_shape(&outs);
    let axis = &shape[0];
    assert!(axis.as_const().is_none());
    assert!(axis.as_symbol().is_some());
}

// --- Shape / Size ---------------------------------------------------------

#[test]
fn shape_emits_dims_as_shape_data() {
    let n = node("Shape", 1, 1);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 13);
    assert_eq!(out_shape(&outs), vec![c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Int64);
    let sd = outs[0].shape_data.as_ref().unwrap();
    assert_eq!(sd.elems, vec![sym(0), c(8), c(768)]);
}

#[test]
fn shape_with_start_end() {
    let n = with_attr(
        with_attr(node("Shape", 1, 1), "start", Attribute::Int(1)),
        "end",
        Attribute::Int(3),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768), c(2)])], 15);
    let sd = outs[0].shape_data.as_ref().unwrap();
    assert_eq!(sd.elems, vec![c(8), c(768)]);
}

// --- Unsqueeze / Squeeze (opset-range dispatch) ---------------------------

#[test]
fn unsqueeze_v1_axes_attr() {
    // opset 11: axes is an attribute.
    let n = with_attr(node("Unsqueeze", 1, 1), "axes", Attribute::Ints(vec![0]));
    let outs = run(&n, vec![f32in(vec![c(8), c(768)])], 11);
    assert_eq!(out_shape(&outs), vec![c(1), c(8), c(768)]);
}

#[test]
fn unsqueeze_v13_axes_input() {
    // opset 13: axes is input 1 (shape-data).
    let n = node("Unsqueeze", 2, 1);
    let outs = run(&n, vec![f32in(vec![c(8), c(768)]), sd_vec(vec![c(0)])], 13);
    assert_eq!(out_shape(&outs), vec![c(1), c(8), c(768)]);
}

#[test]
fn unsqueeze_scalar_shape_data_to_vector() {
    // A scalar dim unsqueezed to a 1-vector keeps its value (shape-chain).
    let scalar = NodeIo {
        type_info: Some(TypeInfo::new(DataType::Int64, vec![])),
        shape_data: Some(ShapeData::scalar(DataType::Int64, sym(0))),
        value_type: None,
    };
    let n = with_attr(node("Unsqueeze", 1, 1), "axes", Attribute::Ints(vec![0]));
    let outs = run(&n, vec![scalar], 11);
    let sd = outs[0].shape_data.as_ref().expect("unsqueeze shape-data");
    assert_eq!(sd.elems, vec![sym(0)]);
    assert!(!sd.is_scalar());
}

#[test]
fn squeeze_v13_axes_input() {
    let n = node("Squeeze", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(1), c(8), c(1)]), sd_vec(vec![c(0), c(2)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);
}

#[test]
fn squeeze_static_axes_reject_invalid_dims_and_leave_dynamic_dims_unresolved() {
    let axes_input = node("Squeeze", 2, 1);
    let err = try_run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), sd_vec(vec![c(2)])],
        13,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 2 is out of range for rank 2")
    );

    let err = try_run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), sd_vec(vec![c(1)])],
        13,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("cannot squeeze axis 1 with non-singleton extent 8")
    );

    let outs = run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), sd_vec(vec![c(0)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);

    let dynamic_extent = run(
        &axes_input,
        vec![f32in(vec![sym(0), c(8)]), sd_vec(vec![c(0)])],
        13,
    );
    assert!(dynamic_extent[0].type_info.is_none());

    let dynamic_axes = run(
        &axes_input,
        vec![f32in(vec![c(1), c(8)]), tin(DataType::Int64, vec![sym(0)])],
        13,
    );
    assert!(dynamic_axes[0].type_info.is_none());
}

#[test]
fn squeeze_static_axes_validate_structure_before_dynamic_extents() {
    let axes_input = node("Squeeze", 2, 1);
    let input = f32in(vec![sym(0), c(1)]);

    let err = try_run(
        &axes_input,
        vec![input.clone(), sd_vec(vec![c(0), c(0)])],
        13,
    )
    .unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 0 is specified more than once")
    );

    let err = try_run(&axes_input, vec![input, sd_vec(vec![c(0), c(2)])], 13).unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 2 is out of range for rank 2")
    );
}

#[test]
fn squeeze_v11_rejects_duplicate_static_axes() {
    let n = with_attr(node("Squeeze", 1, 1), "axes", Attribute::Ints(vec![0, 0]));
    let err = try_run(&n, vec![f32in(vec![c(1), c(8)])], 11).unwrap_err();
    assert!(
        err.to_string()
            .contains("axis 0 is specified more than once")
    );
}

// --- Slice ----------------------------------------------------------------

#[test]
fn slice_concrete_bounds() {
    // data [10, 768], slice axis 0 [2:8] -> [6, 768]
    let n = node("Slice", 5, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(10), c(768)]),
            sd_vec(vec![c(2)]),
            sd_vec(vec![c(8)]),
            sd_vec(vec![c(0)]),
            sd_vec(vec![c(1)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(6), c(768)]);
}

#[test]
fn slice_data_dependent_keeps_rank_symbolic() {
    // Bounds unknown (no shape-data on starts/ends): axis stays symbolic.
    let n = node("Slice", 3, 1);
    let starts = f32in(vec![c(1)]); // present but no shape-data
    let ends = f32in(vec![c(1)]);
    let outs = run(&n, vec![f32in(vec![c(10), c(768)]), starts, ends], 13);
    let shape = out_shape(&outs);
    assert_eq!(shape.len(), 2);
    assert!(shape[0].as_symbol().is_some());
    assert_eq!(shape[1], c(768));
}

#[test]
fn slice_dynamic_bounds_only_clear_selected_axes() {
    let n = node("Slice", 4, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(10), c(20)]),
            tin(DataType::Int64, vec![c(1)]),
            tin(DataType::Int64, vec![c(1)]),
            sd_vec(vec![c(1)]),
        ],
        13,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape[0], c(10));
    assert!(shape[1].as_symbol().is_some());
}

#[test]
fn slice_dynamic_axes_clear_every_extent() {
    let n = node("Slice", 4, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(10), c(20)]),
            sd_vec(vec![c(0)]),
            sd_vec(vec![c(5)]),
            tin(DataType::Int64, vec![c(1)]),
        ],
        13,
    );
    assert!(
        out_shape(&outs)
            .iter()
            .all(|extent| extent.as_symbol().is_some())
    );
}

#[test]
fn slice_negative_step_clamps_extreme_bounds() {
    let n = node("Slice", 5, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(5)]),
            sd_vec(vec![c(i64::MAX)]),
            sd_vec(vec![c(i64::MIN)]),
            sd_vec(vec![c(0)]),
            sd_vec(vec![c(-1)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(5)]);
}

// --- ReduceMean -----------------------------------------------------------

#[test]
fn reduce_mean_keepdims() {
    let n = with_attr(
        with_attr(node("ReduceMean", 1, 1), "axes", Attribute::Ints(vec![-1])),
        "keepdims",
        Attribute::Int(1),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 12);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(1)]);
}

#[test]
fn reduce_mean_no_keepdims() {
    let n = with_attr(
        with_attr(node("ReduceMean", 1, 1), "axes", Attribute::Ints(vec![1])),
        "keepdims",
        Attribute::Int(0),
    );
    let outs = run(&n, vec![f32in(vec![c(2), c(8), c(768)])], 12);
    assert_eq!(out_shape(&outs), vec![c(2), c(768)]);
}

// --- Softmax / LayerNorm --------------------------------------------------

#[test]
fn softmax_passthrough() {
    let n = with_attr(node("Softmax", 1, 1), "axis", Attribute::Int(-1));
    let outs = run(&n, vec![f32in(vec![sym(0), c(12), c(8), c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(12), c(8), c(8)]);
}

#[test]
fn layer_norm_main_and_reduced_outputs() {
    let n = node("LayerNormalization", 3, 3);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(768)]),
            f32in(vec![c(768)]),
            f32in(vec![c(768)]),
        ],
        17,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    // Mean / InvStdDev: last axis collapses to 1.
    let mean = outs[1].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(mean, vec![sym(0), c(8), c(1)]);
}

#[test]
fn skip_layer_norm_emits_x_shaped_skip_bias_sum() {
    // com.microsoft SkipLayerNormalization with all four outputs: output 0 and
    // output 3 (input_skip_bias_sum) are X-shaped; mean/inv_std collapse last.
    let n = with_domain(node("SkipLayerNormalization", 3, 4), "com.microsoft");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(768)]),
            f32in(vec![sym(0), c(8), c(768)]),
            f32in(vec![c(768)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    let mean = outs[1].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(mean, vec![sym(0), c(8), c(1)]);
    let inv = outs[2].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(inv, vec![sym(0), c(8), c(1)]);
    let skip_sum = outs[3].type_info.as_ref().unwrap().shape.clone();
    assert_eq!(skip_sum, vec![sym(0), c(8), c(768)]);
}

fn gqa_inputs(past_capacity: i64, total: Option<i64>) -> Vec<NodeIo> {
    let mut inputs = vec![
        f32in(vec![c(1), c(1), c(8)]),
        f32in(vec![c(1), c(1), c(4)]),
        f32in(vec![c(1), c(1), c(4)]),
        f32in(vec![c(1), c(2), c(past_capacity), c(2)]),
        f32in(vec![c(1), c(2), c(past_capacity), c(2)]),
        tin(DataType::Int32, vec![c(1)]),
        tin(DataType::Int32, vec![]),
    ];
    if let Some(total) = total {
        inputs[6] = sd_int_scalar(DataType::Int32, c(total));
    }
    inputs
}

fn gqa_node() -> Node {
    with_attr(
        with_attr(
            with_domain(node("GroupQueryAttention", 7, 3), "com.microsoft"),
            "num_heads",
            Attribute::Int(4),
        ),
        "kv_num_heads",
        Attribute::Int(2),
    )
}

#[test]
fn group_query_attention_missing_past_shape_still_emits_present_shapes() {
    let mut inputs = gqa_inputs(8, Some(3));
    inputs[3] = NodeIo::default();
    let outs = run(&gqa_node(), inputs, 1);
    let present_key = &outs[1]
        .type_info
        .as_ref()
        .expect("present key shape resolved")
        .shape;
    let present_value = &outs[2]
        .type_info
        .as_ref()
        .expect("present value shape resolved")
        .shape;
    assert_eq!(present_key.len(), 4);
    assert_eq!(present_key[0], c(1));
    assert_eq!(present_key[1], c(2));
    assert!(present_key[2].as_symbol().is_some());
    assert_eq!(present_key[3], c(2));
    assert_eq!(present_key, present_value);
}

#[test]
fn group_query_attention_fixed_capacity_present_uses_max_capacity_total() {
    let outs = run(&gqa_node(), gqa_inputs(8, Some(3)), 1);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
}

#[test]
fn group_query_attention_non_rank_four_past_still_emits_present_shapes() {
    let mut inputs = gqa_inputs(8, Some(3));
    inputs[3] = f32in(vec![c(1), c(8), c(4)]);
    let outs = run(&gqa_node(), inputs, 1);
    let present_key = &outs[1]
        .type_info
        .as_ref()
        .expect("present key shape resolved")
        .shape;
    let present_value = &outs[2]
        .type_info
        .as_ref()
        .expect("present value shape resolved")
        .shape;
    assert_eq!(present_key.len(), 4);
    assert_eq!(present_key[0], c(1));
    assert_eq!(present_key[1], c(2));
    assert!(present_key[2].as_symbol().is_some());
    assert_eq!(present_key[3], c(2));
    assert_eq!(present_key, present_value);
}

#[test]
fn group_query_attention_growing_present_uses_logical_total() {
    let outs = run(&gqa_node(), gqa_inputs(2, Some(3)), 1);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(3), c(2)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(3), c(2)]
    );
}

#[test]
fn group_query_attention_dynamic_total_leaves_present_sequence_symbolic() {
    let outs = run(&gqa_node(), gqa_inputs(8, None), 1);
    let present_key = &outs[1].type_info.as_ref().unwrap().shape;
    let present_value = &outs[2].type_info.as_ref().unwrap().shape;
    assert!(
        present_key[2].as_symbol().is_some(),
        "dynamic max(capacity, total) must remain data-dependent"
    );
    assert_eq!(present_key[2], present_value[2]);
}

#[test]
fn group_query_attention_packed_qkv_splits_output_and_cache_shapes() {
    let mut inputs = gqa_inputs(8, Some(3));
    inputs[0] = f32in(vec![c(1), c(1), c(16)]);
    inputs[1] = NodeIo::default();
    inputs[2] = NodeIo::default();
    let mut packed_node = gqa_node();
    packed_node.inputs[1] = None;
    packed_node.inputs[2] = None;
    let outs = run(&packed_node, inputs, 1);
    assert_eq!(out_shape(&outs), vec![c(1), c(1), c(8)]);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(1), c(2), c(8), c(2)]
    );
}

#[test]
fn moe_and_qmoe_preserve_activation_shape() {
    for op in ["MoE", "QMoE"] {
        let n = with_domain(node(op, 7, 1), "com.microsoft");
        let inputs = vec![
            f32in(vec![sym(0), c(4), c(512)]),
            f32in(vec![c(4), c(8)]),
            tin(DataType::Uint8, vec![c(8), c(1024), c(256)]),
            f32in(vec![c(8), c(1024), c(16)]),
            NodeIo::default(),
            tin(DataType::Uint8, vec![c(8), c(512), c(512)]),
            f32in(vec![c(8), c(512), c(32)]),
        ];
        let outs = run(&n, inputs, 1);
        assert_eq!(out_shape(&outs), vec![sym(0), c(4), c(512)]);
        assert_eq!(out_dtype(&outs), DataType::Float32);
    }

    let n = with_domain(node("BlockQuantizedMoE", 5, 1), "pkg.nxrt");
    let inputs = vec![
        f32in(vec![sym(0), c(4), c(512)]),
        f32in(vec![c(4), c(8)]),
        tin(DataType::Uint8, vec![c(8), c(1024), c(2), c(136)]),
        NodeIo::default(),
        tin(DataType::Uint8, vec![c(8), c(512), c(4), c(136)]),
    ];
    let outs = run(&n, inputs, 1);
    assert_eq!(out_shape(&outs), vec![sym(0), c(4), c(512)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn sparse_kv_gather_emits_selected_kv_shape() {
    let n = with_domain(node("SparseKvGather", 2, 1), "pkg.nxrt");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(2), c(64), c(128)]),
            tin(DataType::Int32, vec![sym(0), c(2), c(3), c(16)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(2), c(3), c(16), c(128)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn index_share_mirrors_query_and_present_kv() {
    // Exporter emission: 6 inputs (q, key, value, past_key, past_value,
    // selected_indices), single attn_output that mirrors the query.
    let n = with_domain(node("IndexShare", 6, 1), "pkg.nxrt");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(4), c(3), c(24)]),
            f32in(vec![sym(0), c(4), c(3), c(24)]),
            f32in(vec![sym(0), c(4), c(3), c(24)]),
            NodeIo::default(),
            NodeIo::default(),
            tin(DataType::Int64, vec![sym(0), c(1), c(3), c(3)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(4), c(3), c(24)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);

    // With 3 outputs, present K/V are `[B, kv, S_past + S_cur, H]`.
    let n = with_domain(node("IndexShare", 6, 3), "pkg.nxrt");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(4), c(1), c(24)]),
            f32in(vec![sym(0), c(2), c(1), c(24)]),
            f32in(vec![sym(0), c(2), c(1), c(24)]),
            f32in(vec![sym(0), c(2), c(7), c(24)]),
            f32in(vec![sym(0), c(2), c(7), c(24)]),
            tin(DataType::Int64, vec![sym(0), c(1), c(1), c(8)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(4), c(1), c(24)]);
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![sym(0), c(2), c(8), c(24)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![sym(0), c(2), c(8), c(24)]
    );
}

#[test]
fn kv_cache_capacity_append_reports_past_capacity_not_a_grown_concat() {
    // The capacity write is in place and `present` aliases `past`, so the
    // output keeps `past`'s shape even though `current` contributes tokens.
    // The `Concat` this op replaces would report `S_past + S_cur` (7 + 1 = 8)
    // for the same inputs, so 7-not-8 is the whole contract.
    let n = with_domain(node("KvCacheCapacityAppend", 3, 1), "pkg.nxrt");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(2), c(7), c(24)]),
            f32in(vec![sym(0), c(2), c(1), c(24)]),
            tin(DataType::Int64, vec![sym(0), c(1)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(2), c(7), c(24)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);

    // Shape and dtype both come from `past`, not from a fixed KV geometry and
    // not from `current`. The dtypes differ deliberately: a capacity-backed
    // cache may store `past` at a lower precision than the freshly computed
    // `current`, and `present` aliases the storage, so f16 is the answer.
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![sym(1), c(5), c(512), c(80)]),
            tin(DataType::Float32, vec![sym(1), c(5), c(4), c(80)]),
            tin(DataType::Int64, vec![sym(1), c(4)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(1), c(5), c(512), c(80)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn varlen_attention_preserves_packed_query_geometry() {
    let n = with_domain(node("VarlenAttention", 5, 1), "pkg.nxrt");
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(8), c(64)]),
            f32in(vec![sym(1), c(2), c(64)]),
            f32in(vec![sym(1), c(2), c(80)]),
            tin(DataType::Int32, vec![sym(2)]),
            tin(DataType::Int32, vec![sym(2)]),
        ],
        1,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(80)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn custom_ops_validate_arity_rank_and_compression_contracts() {
    assert!(
        try_run(
            &with_domain(node("MoE", 1, 2), "com.microsoft"),
            vec![f32in(vec![c(2), c(4)])],
            1,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_domain(node("SparseKvGather", 2, 1), "pkg.nxrt"),
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                tin(DataType::Int32, vec![c(2), c(3), c(4), c(5)]),
            ],
            1,
        )
        .is_err()
    );

    let invalid_ratio = with_attr(
        with_domain(node("CompressedSparseAttention", 10, 3), "pkg.nxrt"),
        "compression_ratio",
        Attribute::Int(8),
    );
    assert!(
        try_run(
            &invalid_ratio,
            vec![f32in(vec![c(1), c(1), c(1), c(64)])],
            1,
        )
        .is_err()
    );

    let mut wrong_ratio128_arity =
        with_domain(node("CompressedSparseAttention", 10, 2), "pkg.nxrt");
    for (name, value) in [
        ("num_heads", 1),
        ("head_dim", 64),
        ("compression_ratio", 128),
    ] {
        wrong_ratio128_arity
            .attributes
            .insert(name.into(), Attribute::Int(value));
    }
    assert!(
        try_run(
            &wrong_ratio128_arity,
            vec![f32in(vec![c(1), c(1), c(1), c(64)])],
            1,
        )
        .is_err()
    );
}

fn csa_ratio4_node(domain: &str) -> Node {
    let mut n = with_domain(node("CompressedSparseAttention", 19, 6), domain);
    for (name, value) in [
        ("num_heads", 8),
        ("head_dim", 512),
        ("qk_rope_head_dim", 64),
        ("compression_ratio", 4),
        ("index_num_heads", 2),
        ("index_head_dim", 128),
        ("index_topk", 512),
    ] {
        n.attributes.insert(name.into(), Attribute::Int(value));
    }
    n.attributes.insert(
        "cache_format".into(),
        Attribute::String("fp8_e4m3_block64".into()),
    );
    n
}

#[test]
fn compressed_sparse_attention_emits_all_ratio4_state_shapes() {
    for domain in ["pkg.nxrt", "com.microsoft"] {
        let mut inputs = vec![NodeIo::default(); 19];
        inputs[0] = f32in(vec![sym(0), c(5), c(8), c(512)]);
        inputs[9] = sd_int_scalar(DataType::Int64, c(12));
        let outs = run(&csa_ratio4_node(domain), inputs, 1);
        let expected = [
            (DataType::Float32, vec![sym(0), c(5), c(8), c(512)]),
            (DataType::Uint8, vec![sym(0), c(3), c(583)]),
            (DataType::Float32, vec![sym(0), c(8), c(2), c(1024)]),
            (DataType::Uint8, vec![sym(0), c(3), c(68)]),
            (DataType::Float32, vec![sym(0), c(8), c(2), c(256)]),
            (DataType::Int32, vec![sym(0), c(2), c(5), c(3)]),
        ];
        for (output, (dtype, shape)) in outs.iter().zip(expected.iter()) {
            let info = output.type_info.as_ref().expect("CSA output resolved");
            assert_eq!(info.dtype, *dtype);
            assert_eq!(info.shape, *shape);
        }
    }
}

#[test]
fn compressed_sparse_attention_dynamic_total_resolves_every_output() {
    let mut inputs = vec![NodeIo::default(); 19];
    inputs[0] = f32in(vec![c(2), sym(0), c(8), c(512)]);
    inputs[9] = tin(DataType::Int64, vec![]);
    let outs = run(&csa_ratio4_node("pkg.nxrt"), inputs, 1);
    assert!(outs.iter().all(|output| output.type_info.is_some()));
    let cache_records = outs[1].type_info.as_ref().unwrap().shape[1].clone();
    let index_records = outs[3].type_info.as_ref().unwrap().shape[1].clone();
    assert!(cache_records.as_symbol().is_some());
    assert_eq!(cache_records, index_records);
    assert!(
        outs[5].type_info.as_ref().unwrap().shape[3]
            .as_symbol()
            .is_some()
    );
}

#[test]
fn compressed_sparse_attention_ratio128_emits_three_outputs() {
    let mut n = with_domain(node("CompressedSparseAttention", 11, 3), "pkg.nxrt");
    for (name, value) in [
        ("num_heads", 8),
        ("head_dim", 512),
        ("qk_rope_head_dim", 64),
        ("compression_ratio", 128),
    ] {
        n.attributes.insert(name.into(), Attribute::Int(value));
    }
    n.attributes.insert(
        "cache_format".into(),
        Attribute::String("fp8_e4m3_block64".into()),
    );
    let mut inputs = vec![NodeIo::default(); 11];
    inputs[0] = f32in(vec![c(2), c(1), c(8), c(512)]);
    inputs[9] = sd_int_scalar(DataType::Int64, c(256));
    let outs = run(&n, inputs, 1);
    assert_eq!(
        outs[0].type_info.as_ref().unwrap().shape,
        vec![c(2), c(1), c(8), c(512)]
    );
    assert_eq!(
        outs[1].type_info.as_ref().unwrap().shape,
        vec![c(2), c(2), c(583)]
    );
    assert_eq!(
        outs[2].type_info.as_ref().unwrap().shape,
        vec![c(2), c(128), c(2), c(512)]
    );
}

#[test]
fn standard_simplified_layer_norm_passthrough() {
    let n = node("SimplifiedLayerNormalization", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8), c(768)]), f32in(vec![c(768)])],
        21,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn rms_norm_passthrough() {
    // Single output equal to X (opset 23).
    let n = node("RMSNormalization", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![sym(0), c(8), c(768)]), f32in(vec![c(768)])],
        23,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn batch_norm_inference_passthrough_opsets_9_14_15() {
    let n = node("BatchNormalization", 5, 1);
    for opset in [9, 14, 15] {
        let outs = run(
            &n,
            vec![
                tin(DataType::Float16, vec![c(2), c(3), c(4), c(5)]),
                tin(DataType::Float16, vec![c(3)]),
                tin(DataType::Float16, vec![c(3)]),
                tin(DataType::Float16, vec![c(3)]),
                tin(DataType::Float16, vec![c(3)]),
            ],
            opset,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4), c(5)]);
        assert_eq!(out_dtype(&outs), DataType::Float16);
    }
}

#[test]
fn batch_norm_training_with_optional_outputs_preserves_y_without_fabricating_statistics() {
    let n = with_attr(
        node("BatchNormalization", 5, 3),
        "training_mode",
        Attribute::Int(1),
    );
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(2), c(3), c(4), c(5)]),
            tin(DataType::Float16, vec![c(3)]),
            tin(DataType::Float16, vec![c(3)]),
            tin(DataType::Float16, vec![c(3)]),
            tin(DataType::Float16, vec![c(3)]),
        ],
        15,
    );

    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4), c(5)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);
    assert!(outs[1].type_info.is_none());
    assert!(outs[2].type_info.is_none());
}

#[test]
fn instance_norm_passthrough_opset_6() {
    let n = node("InstanceNormalization", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(1), c(8), c(16), c(16)]),
            tin(DataType::Float16, vec![c(8)]),
            tin(DataType::Float16, vec![c(8)]),
        ],
        6,
    );
    assert_eq!(out_shape(&outs), vec![c(1), c(8), c(16), c(16)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn normalization_unknown_x_leaves_output_unresolved() {
    for (op, n_in, opset) in [
        ("BatchNormalization", 5, 15),
        ("InstanceNormalization", 3, 6),
    ] {
        let n = node(op, n_in, 1);
        let mut inputs = vec![NodeIo::default()];
        inputs.extend((1..n_in).map(|_| f32in(vec![c(3)])));
        let outs = run(&n, inputs, opset);
        assert!(outs[0].type_info.is_none());
    }
}

#[test]
fn rotary_embedding_passthrough_4d() {
    // Output equals input X (opset 23), 4D [batch, heads, seq, head_size].
    let n = node("RotaryEmbedding", 3, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(12), c(16), c(64)]),
            f32in(vec![c(16), c(32)]),
            f32in(vec![c(16), c(32)]),
        ],
        23,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(12), c(16), c(64)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn swish_passthrough() {
    // Elementwise, same shape/dtype (opset 24).
    let n = with_attr(node("Swish", 1, 1), "alpha", Attribute::Float(1.0));
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 24);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn swish_uses_node_local_opset_when_graph_import_is_older() {
    let n = with_version(node("Swish", 1, 1), 24);
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 21);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn std_gelu_passthrough() {
    // Standard ai.onnx::Gelu (opset 20), same shape/dtype.
    let n = with_attr(
        node("Gelu", 1, 1),
        "approximate",
        Attribute::String(b"tanh".to_vec()),
    );
    let outs = run(&n, vec![f32in(vec![sym(0), c(8), c(768)])], 20);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn std_gelu_is_unregistered_before_opset_20() {
    let registry = InferenceRegistry::default_registry();
    assert!(registry.get("", "Gelu", 19).is_none());
}

// --- Cast -----------------------------------------------------------------

#[test]
fn cast_changes_dtype_keeps_shape_and_shape_data() {
    let input = sd_vec(vec![sym(0), c(8)]);
    // Cast int64 -> int32 (to=6)
    let n = with_attr(node("Cast", 1, 1), "to", Attribute::Int(6));
    let outs = run(&n, vec![input], 13);
    assert_eq!(out_dtype(&outs), DataType::Int32);
    assert_eq!(out_shape(&outs), vec![c(2)]);
    let sd = outs[0].shape_data.as_ref().unwrap();
    assert_eq!(sd.dtype, DataType::Int32);
    assert_eq!(sd.elems, vec![sym(0), c(8)]);
}

#[test]
fn data_ops_propagate_shape_data_and_reject_invalid_metadata() {
    let shape = with_attr(
        with_attr(node("Shape", 1, 1), "start", Attribute::Int(-2)),
        "end",
        Attribute::Int(99),
    );
    let shape_out = run(&shape, vec![f32in(vec![sym(0), c(3), c(4)])], 15);
    assert_eq!(out_shape(&shape_out), vec![c(2)]);
    assert_eq!(
        shape_out[0].shape_data.as_ref().unwrap().elems,
        vec![c(3), c(4)]
    );

    let size_out = run(&node("Size", 1, 1), vec![f32in(vec![c(2), c(3)])], 13);
    assert_eq!(size_out[0].shape_data.as_ref().unwrap().elems, vec![c(6)]);

    let constant = with_attr(
        node("Constant", 0, 1),
        "value_ints",
        Attribute::Ints(vec![2, 5]),
    );
    let constant_out = run(&constant, vec![], 13);
    assert_eq!(out_shape(&constant_out), vec![c(2)]);
    assert_eq!(
        constant_out[0].shape_data.as_ref().unwrap().elems,
        vec![c(2), c(5)]
    );

    let identity = run(&node("Identity", 1, 1), vec![sd_vec(vec![c(7)])], 13);
    assert_eq!(identity[0].shape_data.as_ref().unwrap().elems, vec![c(7)]);

    let cast_like = run(
        &node("CastLike", 2, 1),
        vec![f32in(vec![c(2), c(3)]), tin(DataType::Uint8, vec![c(1)])],
        19,
    );
    assert_eq!(out_dtype(&cast_like), DataType::Uint8);
    assert_eq!(out_shape(&cast_like), vec![c(2), c(3)]);

    assert!(try_run(&node("Cast", 1, 1), vec![f32in(vec![c(1)])], 13).is_err());
    assert!(
        try_run(
            &with_attr(node("Cast", 1, 1), "to", Attribute::Int(-1)),
            vec![f32in(vec![c(1)])],
            13,
        )
        .is_err()
    );
}

// --- ConstantOfShape / Expand --------------------------------------------

#[test]
fn constant_of_shape_uses_shape_data() {
    let n = node("ConstantOfShape", 1, 1);
    let outs = run(&n, vec![sd_vec(vec![sym(0), c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(8)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

#[test]
fn constant_of_shape_dynamic_input_is_unresolved_but_empty_shape_is_scalar() {
    let n = node("ConstantOfShape", 1, 1);
    let dynamic = run(&n, vec![tin(DataType::Int64, vec![c(3)])], 25);
    assert!(dynamic[0].type_info.is_none());

    let scalar = run(&n, vec![sd_vec(vec![])], 25);
    assert_eq!(out_shape(&scalar), Vec::<DimExpr>::new());
    assert_eq!(out_dtype(&scalar), DataType::Float32);
}

#[test]
fn expand_broadcasts_against_target() {
    // input [1, 8, 1], target [N, 8, 768] -> [N, 8, 768]
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![
            f32in(vec![c(1), c(8), c(1)]),
            sd_vec(vec![sym(0), c(8), c(768)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(768)]);
}

#[test]
fn expand_adds_leading_target_dimensions() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(1)]), sd_vec(vec![c(2), c(1), c(6)])],
        8,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3), c(6)]);
}

#[test]
fn expand_target_one_keeps_input_dimension() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(4)]), sd_vec(vec![c(3), c(1)])],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(4)]);
}

#[test]
fn expand_preserves_input_dtype() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(1), c(4)]),
            sd_vec(vec![c(3), c(4)]),
        ],
        13,
    );
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn expand_unknown_shape_tensor_leaves_output_unresolved() {
    let n = node("Expand", 2, 1);
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(1)]), tin(DataType::Int64, vec![c(3)])],
        13,
    );
    assert!(outs[0].type_info.is_none());
}

#[test]
fn expand_rejects_incompatible_concrete_dimensions() {
    let n = node("Expand", 2, 1);
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(2), c(3)]), sd_vec(vec![c(2), c(4)])],
            13,
        )
        .is_err()
    );
}

// --- Where ----------------------------------------------------------------

#[test]
fn where_broadcasts_all_three() {
    let n = node("Where", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Bool, vec![c(1), c(8)]),
            f32in(vec![c(3), c(1)]),
            f32in(vec![c(3), c(8)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(8)]);
    assert_eq!(out_dtype(&outs), DataType::Float32);
}

// --- Flatten / Split ------------------------------------------------------

#[test]
fn flatten_axis() {
    let n = with_attr(node("Flatten", 1, 1), "axis", Attribute::Int(2));
    let outs = run(&n, vec![f32in(vec![c(2), c(3), c(4), c(5)])], 13);
    assert_eq!(out_shape(&outs), vec![c(6), c(20)]);
}

#[test]
fn split_equal() {
    let n = with_attr(node("Split", 1, 2), "axis", Attribute::Int(1));
    let outs = run(&n, vec![f32in(vec![c(2), c(8)])], 13);
    assert_eq!(out_shape(&outs), vec![c(2), c(4)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(4)]);
}

#[test]
fn split_dynamic_sizes_leave_split_axis_unknown() {
    let n = with_attr(node("Split", 2, 2), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![f32in(vec![c(2), c(6)]), tin(DataType::Int64, vec![c(2)])],
        13,
    );
    for output in outs {
        let shape = output.type_info.unwrap().shape;
        assert_eq!(shape[0], c(2));
        assert!(shape[1].as_symbol().is_some());
    }
}

#[test]
fn split_num_outputs_uses_ceil_chunks_and_final_remainder() {
    let n = with_attr(
        with_attr(node("Split", 1, 3), "axis", Attribute::Int(1)),
        "num_outputs",
        Attribute::Int(3),
    );
    let outs = run(&n, vec![f32in(vec![c(2), c(7)])], 18);
    assert_eq!(out_shape(&outs), vec![c(2), c(3)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(3)]);
    assert_eq!(outs[2].type_info.as_ref().unwrap().shape, vec![c(2), c(1)]);
}

#[test]
fn split_num_outputs_zero_size_final_chunk() {
    let n = with_attr(
        with_attr(node("Split", 1, 3), "axis", Attribute::Int(1)),
        "num_outputs",
        Attribute::Int(3),
    );
    let outs = run(&n, vec![f32in(vec![c(2), c(2)])], 18);
    assert_eq!(out_shape(&outs), vec![c(2), c(1)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(2), c(1)]);
    assert_eq!(outs[2].type_info.as_ref().unwrap().shape, vec![c(2), c(0)]);
}

#[test]
fn split_input_sizes_are_uneven_and_must_match_outputs_and_axis_extent() {
    let n = with_attr(node("Split", 2, 2), "axis", Attribute::Int(1));
    let outs = run(
        &n,
        vec![f32in(vec![c(3), c(7)]), sd_vec(vec![c(2), c(5)])],
        18,
    );
    assert_eq!(out_shape(&outs), vec![c(3), c(2)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, vec![c(3), c(5)]);

    assert_invalid(
        try_run(
            &n,
            vec![f32in(vec![c(3), c(7)]), sd_vec(vec![c(2), c(4)])],
            18,
        )
        .unwrap_err(),
        "Split",
        "split sizes sum to 6, but axis extent is 7",
    );

    let both = with_attr(n.clone(), "num_outputs", Attribute::Int(2));
    assert_invalid(
        try_run(
            &both,
            vec![f32in(vec![c(3), c(7)]), sd_vec(vec![c(2), c(5)])],
            18,
        )
        .unwrap_err(),
        "Split",
        "split input and num_outputs cannot both be specified",
    );
}

#[test]
fn split_rejects_non_positive_num_outputs() {
    for num_outputs in [0, -1] {
        let n = with_attr(
            node("Split", 1, 2),
            "num_outputs",
            Attribute::Int(num_outputs),
        );
        let error = try_run(&n, vec![f32in(vec![c(2), c(8)])], 18).unwrap_err();
        assert_invalid(
            error,
            "Split",
            &format!("num_outputs must be positive, got {num_outputs}"),
        );
    }
}

#[test]
fn comparison_ops_broadcast_to_bool() {
    for op in ["Less", "LessOrEqual", "Greater", "GreaterOrEqual", "Equal"] {
        let outs = run(
            &node(op, 2, 1),
            vec![
                tin(DataType::Float32, vec![c(2), c(1), c(4)]),
                tin(DataType::Float32, vec![c(1), c(3), c(1)]),
            ],
            19,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(3), c(4)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Bool, "{op}");
    }
}

#[test]
fn logical_ops_broadcast_to_bool() {
    for op in ["And", "Or", "Xor"] {
        let outs = run(
            &node(op, 2, 1),
            vec![
                tin(DataType::Bool, vec![c(2), c(1)]),
                tin(DataType::Bool, vec![c(1), c(3)]),
            ],
            19,
        );
        assert_eq!(out_shape(&outs), vec![c(2), c(3)], "{op}");
        assert_eq!(out_dtype(&outs), DataType::Bool, "{op}");
    }
}

#[test]
fn elementwise_shape_data_handles_vector_scalar_and_exact_division() {
    let add = run(
        &node("Add", 2, 1),
        vec![
            sd_vec(vec![c(2), c(5)]),
            sd_int_scalar(DataType::Int64, c(3)),
        ],
        13,
    );
    assert_eq!(add[0].shape_data.as_ref().unwrap().elems, vec![c(5), c(8)]);

    let div = run(
        &node("Div", 2, 1),
        vec![
            sd_int_scalar(DataType::Int64, c(12)),
            sd_vec(vec![c(2), c(3)]),
        ],
        13,
    );
    assert_eq!(div[0].shape_data.as_ref().unwrap().elems, vec![c(6), c(4)]);

    let maximum = run(
        &node("Max", 2, 1),
        vec![sd_vec(vec![c(2), c(9)]), sd_vec(vec![c(7), c(3)])],
        13,
    );
    assert_eq!(
        maximum[0].shape_data.as_ref().unwrap().elems,
        vec![c(7), c(9)]
    );
}

#[test]
fn not_preserves_shape_and_outputs_bool() {
    let outs = run(
        &node("Not", 1, 1),
        vec![tin(DataType::Bool, vec![c(2), c(3)])],
        19,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(3)]);
    assert_eq!(out_dtype(&outs), DataType::Bool);
}

// --- Conv / Pool / Pad ----------------------------------------------------

#[test]
fn conv_spatial_formula() {
    // X [N, 3, 224, 224], W [64, 3, 7, 7], stride 2, pad 3 -> [N, 64, 112, 112]
    let n = {
        let n = with_attr(node("Conv", 2, 1), "strides", Attribute::Ints(vec![2, 2]));
        with_attr(n, "pads", Attribute::Ints(vec![3, 3, 3, 3]))
    };
    let outs = run(
        &n,
        vec![
            f32in(vec![sym(0), c(3), c(224), c(224)]),
            f32in(vec![c(64), c(3), c(7), c(7)]),
        ],
        13,
    );
    assert_eq!(out_shape(&outs), vec![sym(0), c(64), c(112), c(112)]);
}

#[test]
fn maxpool_spatial_formula() {
    // X [N, 64, 112, 112], kernel 3, stride 2, pad 1 -> [N, 64, 56, 56]
    let n = {
        let n = with_attr(
            node("MaxPool", 1, 1),
            "kernel_shape",
            Attribute::Ints(vec![3, 3]),
        );
        let n = with_attr(n, "strides", Attribute::Ints(vec![2, 2]));
        with_attr(n, "pads", Attribute::Ints(vec![1, 1, 1, 1]))
    };
    let outs = run(&n, vec![f32in(vec![sym(0), c(64), c(112), c(112)])], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(64), c(56), c(56)]);
}

#[test]
fn pooling_dilation_ceil_mode_and_indices_shape() {
    let mut n = with_attr(
        node("MaxPool", 1, 2),
        "kernel_shape",
        Attribute::Ints(vec![3]),
    );
    n = with_attr(n, "strides", Attribute::Ints(vec![2]));
    n = with_attr(n, "pads", Attribute::Ints(vec![1, 1]));
    n = with_attr(n, "dilations", Attribute::Ints(vec![2]));
    n = with_attr(n, "ceil_mode", Attribute::Int(1));
    let outs = run(&n, vec![f32in(vec![c(1), c(4), c(10)])], 22);
    assert_eq!(out_shape(&outs), vec![c(1), c(4), c(5)]);
    assert_eq!(outs[1].type_info.as_ref().unwrap().shape, out_shape(&outs));
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Int64);
}

#[test]
fn pooling_auto_pad_same_and_valid() {
    let same = with_attr(
        with_attr(
            with_attr(
                node("AveragePool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![3, 3]),
            ),
            "strides",
            Attribute::Ints(vec![2, 2]),
        ),
        "auto_pad",
        Attribute::String("SAME_LOWER".into()),
    );
    assert_eq!(
        out_shape(&run(&same, vec![f32in(vec![c(1), c(3), c(5), c(6)])], 22,)),
        vec![c(1), c(3), c(3), c(3)]
    );

    let valid = with_attr(
        with_attr(
            with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![3]),
            ),
            "strides",
            Attribute::Ints(vec![2]),
        ),
        "auto_pad",
        Attribute::String("VALID".into()),
    );
    assert_eq!(
        out_shape(&run(&valid, vec![f32in(vec![c(1), c(3), c(8)])], 22)),
        vec![c(1), c(3), c(3)]
    );
}

#[test]
fn pooling_validates_kernel_and_rank_and_preserves_dynamic_spatial_rank() {
    assert!(
        try_run(
            &node("AveragePool", 1, 1),
            vec![f32in(vec![c(1), c(2), c(8)])],
            22,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![2, 2]),
            ),
            vec![f32in(vec![c(1), c(2), c(8)])],
            22,
        )
        .is_err()
    );
    assert!(
        try_run(
            &with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![2]),
            ),
            vec![f32in(vec![c(1)])],
            22,
        )
        .is_err()
    );

    let n = with_attr(
        node("AveragePool", 1, 1),
        "kernel_shape",
        Attribute::Ints(vec![3]),
    );
    let outs = run(&n, vec![f32in(vec![c(1), c(2), sym(0)])], 22);
    assert_eq!(out_shape(&outs).len(), 3);
    assert!(out_shape(&outs)[2].as_symbol().is_some());
}

#[test]
fn global_pool_sets_every_spatial_dimension_to_one() {
    for op in ["GlobalAveragePool", "GlobalMaxPool"] {
        let outs = run(
            &node(op, 1, 1),
            vec![f32in(vec![sym(0), c(8), c(7), sym(1)])],
            22,
        );
        assert_eq!(out_shape(&outs), vec![sym(0), c(8), c(1), c(1)]);
    }
}

#[test]
fn pooling_rejects_guaranteed_symbolic_overflow() {
    let n = with_attr(
        with_attr(
            node("MaxPool", 1, 1),
            "kernel_shape",
            Attribute::Ints(vec![1]),
        ),
        "pads",
        Attribute::Ints(vec![isize::MAX as i64, 1]),
    );
    let error = try_run(&n, vec![f32in(vec![c(1), c(1), sym(0)])], 22).unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

#[test]
fn pooling_rejects_cancellation_masked_symbolic_overflow() {
    let n = with_attr(
        with_attr(
            with_attr(
                node("MaxPool", 1, 1),
                "kernel_shape",
                Attribute::Ints(vec![isize::MAX as i64]),
            ),
            "dilations",
            Attribute::Ints(vec![2]),
        ),
        "pads",
        Attribute::Ints(vec![isize::MAX as i64, isize::MAX as i64]),
    );
    let error = try_run(&n, vec![f32in(vec![c(1), c(1), sym(0)])], 22).unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

// --- Resize ---------------------------------------------------------------

#[test]
fn resize_infers_constant_scales_and_sizes() {
    let legacy = run(
        &node("Resize", 2, 1),
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            sd_float_vec(vec![1.0, 1.0, 1.5]),
        ],
        10,
    );
    assert_eq!(out_shape(&legacy), vec![c(2), c(3), c(6)]);

    let mut scales_node = node("Resize", 4, 1);
    scales_node.inputs[1] = None;
    scales_node.inputs[3] = None;
    let scales = run(
        &scales_node,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            NodeIo::default(),
            sd_float_vec(vec![1.0, 1.0, 1.5]),
            NodeIo::default(),
        ],
        19,
    );
    assert_eq!(out_shape(&scales), vec![c(2), c(3), c(6)]);

    let mut sizes_node = with_attr(node("Resize", 4, 1), "axes", Attribute::Ints(vec![-2, -1]));
    sizes_node.inputs[1] = None;
    sizes_node.inputs[2] = None;
    let sizes = run(
        &sizes_node,
        vec![
            f32in(vec![c(2), c(3), c(4), c(5)]),
            NodeIo::default(),
            NodeIo::default(),
            sd_vec(vec![c(8), c(9)]),
        ],
        19,
    );
    assert_eq!(out_shape(&sizes), vec![c(2), c(3), c(8), c(9)]);
}

#[test]
fn resize_dynamic_scales_and_sizes_leave_extents_unresolved() {
    for use_scales in [true, false] {
        let mut n = node("Resize", 4, 1);
        n.inputs[1] = None;
        if use_scales {
            n.inputs[3] = None;
        } else {
            n.inputs[2] = None;
        }
        let vector = if use_scales {
            tin(DataType::Float32, vec![c(3)])
        } else {
            tin(DataType::Int64, vec![c(3)])
        };
        let inputs = if use_scales {
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                NodeIo::default(),
                vector,
                NodeIo::default(),
            ]
        } else {
            vec![
                f32in(vec![c(2), c(3), c(4)]),
                NodeIo::default(),
                NodeIo::default(),
                vector,
            ]
        };
        let shape = out_shape(&run(&n, inputs, 19));
        assert_eq!(shape.len(), 3);
        assert!(
            shape
                .iter()
                .all(|dimension| dimension.as_symbol().is_some())
        );
    }
}

#[test]
fn resize_accepts_ignored_roi_and_absent_extent_inputs() {
    for (roi, has_roi) in [
        (tin(DataType::Float32, vec![c(1), c(4)]), true),
        (NodeIo::default(), false),
    ] {
        let mut resize = with_attr(
            node("Resize", 4, 1),
            "coordinate_transformation_mode",
            Attribute::String("asymmetric".into()),
        );
        if !has_roi {
            resize.inputs[1] = None;
        }
        resize.inputs[3] = None;
        let outputs = run(
            &resize,
            vec![
                f32in(vec![c(2), c(3)]),
                roi,
                sd_float_vec(vec![1.0, 2.0]),
                NodeIo::default(),
            ],
            19,
        );
        assert_eq!(out_shape(&outputs), vec![c(2), c(6)]);
    }

    let mut neither = node("Resize", 4, 1);
    neither.inputs[1] = None;
    neither.inputs[2] = None;
    neither.inputs[3] = None;
    let shape = out_shape(&run(
        &neither,
        vec![
            f32in(vec![c(2), c(3)]),
            NodeIo::default(),
            NodeIo::default(),
            NodeIo::default(),
        ],
        19,
    ));
    assert_eq!(shape.len(), 2);
    assert!(
        shape
            .iter()
            .all(|dimension| dimension.as_symbol().is_some())
    );
}

#[test]
fn resize_rejects_both_scales_and_sizes() {
    let mut both = node("Resize", 4, 1);
    both.inputs[1] = None;
    assert!(
        try_run(
            &both,
            vec![
                f32in(vec![c(2), c(3)]),
                NodeIo::default(),
                sd_float_vec(vec![1.0, 2.0]),
                sd_vec(vec![c(2), c(6)]),
            ],
            19,
        )
        .is_err()
    );
}

#[test]
fn resize_accepts_maximum_extent_with_unit_scale() {
    let mut resize = node("Resize", 4, 1);
    resize.inputs[1] = None;
    resize.inputs[3] = None;
    let outputs = run(
        &resize,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            NodeIo::default(),
            sd_float_vec(vec![1.0]),
            NodeIo::default(),
        ],
        19,
    );
    assert_eq!(out_shape(&outputs), vec![c(isize::MAX as i64)]);

    let mut aspect_resize = with_attr(
        node("Resize", 4, 1),
        "keep_aspect_ratio_policy",
        Attribute::String("not_larger".into()),
    );
    aspect_resize.inputs[1] = None;
    aspect_resize.inputs[2] = None;
    let outputs = run(
        &aspect_resize,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            NodeIo::default(),
            NodeIo::default(),
            sd_vec(vec![c(isize::MAX as i64)]),
        ],
        19,
    );
    assert_eq!(out_shape(&outputs), vec![c(isize::MAX as i64)]);

    let error = try_run(
        &resize,
        vec![
            f32in(vec![c(isize::MAX as i64)]),
            NodeIo::default(),
            sd_float_vec(vec![2.0]),
            NodeIo::default(),
        ],
        19,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));
}

// --- Linear quantization --------------------------------------------------

#[test]
fn quantize_and_dequantize_preserve_shape_and_infer_dtype() {
    let mut quantize = with_attr(node("QuantizeLinear", 3, 1), "axis", Attribute::Int(-1));
    quantize.inputs[2] = None;
    let quantized = run(
        &quantize,
        vec![
            f32in(vec![c(2), c(3)]),
            f32in(vec![c(3)]),
            NodeIo::default(),
        ],
        21,
    );
    assert_eq!(out_shape(&quantized), vec![c(2), c(3)]);
    assert_eq!(out_dtype(&quantized), DataType::Uint8);

    let dequantized = run(
        &node("DequantizeLinear", 3, 1),
        vec![
            tin(DataType::Int4, vec![c(2), c(8)]),
            tin(DataType::Float16, Vec::new()),
            tin(DataType::Int4, Vec::new()),
        ],
        21,
    );
    assert_eq!(out_shape(&dequantized), vec![c(2), c(8)]);
    assert_eq!(out_dtype(&dequantized), DataType::Float16);
}

#[test]
fn quantize_uses_zero_point_dtype_and_validates_blocking() {
    let quantized = run(
        &node("QuantizeLinear", 3, 1),
        vec![
            f32in(vec![c(2), c(3)]),
            f32in(Vec::new()),
            tin(DataType::Int4, Vec::new()),
        ],
        21,
    );
    assert_eq!(out_dtype(&quantized), DataType::Int4);

    let blocked = with_attr(
        with_attr(node("DequantizeLinear", 3, 1), "axis", Attribute::Int(-1)),
        "block_size",
        Attribute::Int(4),
    );
    let outs = run(
        &blocked,
        vec![
            tin(DataType::Uint4, vec![c(2), c(8)]),
            f32in(vec![c(2), c(2)]),
            tin(DataType::Uint4, vec![c(2), c(2)]),
        ],
        21,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(8)]);

    let rank_one_blocked = with_attr(
        with_attr(node("DequantizeLinear", 3, 1), "axis", Attribute::Int(0)),
        "block_size",
        Attribute::Int(4),
    );
    let outs = run(
        &rank_one_blocked,
        vec![
            tin(DataType::Uint4, vec![c(8)]),
            f32in(vec![c(2)]),
            tin(DataType::Uint4, vec![c(2)]),
        ],
        21,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);

    let rank_one_quantize = with_attr(
        with_attr(node("QuantizeLinear", 3, 1), "axis", Attribute::Int(0)),
        "block_size",
        Attribute::Int(4),
    );
    let outs = run(
        &rank_one_quantize,
        vec![
            f32in(vec![c(8)]),
            f32in(vec![c(2)]),
            tin(DataType::Uint4, vec![c(2)]),
        ],
        21,
    );
    assert_eq!(out_shape(&outs), vec![c(8)]);
}

#[test]
fn rank_one_blocked_quantization_validates_axis_range() {
    let inputs = |op: &str| {
        let data = if op == "QuantizeLinear" {
            f32in(vec![c(8)])
        } else {
            tin(DataType::Uint4, vec![c(8)])
        };
        vec![data, f32in(vec![c(2)]), tin(DataType::Uint4, vec![c(2)])]
    };

    for op in ["QuantizeLinear", "DequantizeLinear"] {
        for axis in [0, -1] {
            let blocked = with_attr(
                with_attr(node(op, 3, 1), "axis", Attribute::Int(axis)),
                "block_size",
                Attribute::Int(4),
            );
            let outputs = run(&blocked, inputs(op), 21);
            assert_eq!(out_shape(&outputs), vec![c(8)]);
        }
    }

    for (op, axis) in [
        ("DequantizeLinear", 1),
        ("DequantizeLinear", -2),
        ("QuantizeLinear", 1),
    ] {
        let blocked = with_attr(
            with_attr(node(op, 3, 1), "axis", Attribute::Int(axis)),
            "block_size",
            Attribute::Int(4),
        );
        let error = try_run(&blocked, inputs(op), 21).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(&format!("axis {axis} is out of range for rank 1")),
            "{error}"
        );
    }
}

#[test]
fn quantization_rejects_invalid_axis_and_block_shape() {
    let bad_axis = with_attr(node("QuantizeLinear", 3, 1), "axis", Attribute::Int(-3));
    assert!(
        try_run(
            &bad_axis,
            vec![
                f32in(vec![c(2), c(3)]),
                f32in(vec![c(3)]),
                tin(DataType::Uint8, vec![c(3)]),
            ],
            21,
        )
        .is_err()
    );

    let bad_block = with_attr(
        with_attr(node("DequantizeLinear", 3, 1), "axis", Attribute::Int(1)),
        "block_size",
        Attribute::Int(4),
    );
    assert!(
        try_run(
            &bad_block,
            vec![
                tin(DataType::Uint4, vec![c(2), c(8)]),
                f32in(vec![c(2), c(3)]),
                tin(DataType::Uint4, vec![c(2), c(3)]),
            ],
            21,
        )
        .is_err()
    );
}

#[test]
fn dynamic_quantize_linear_outputs_tensor_and_scalars() {
    let outs = run(
        &node("DynamicQuantizeLinear", 1, 3),
        vec![f32in(vec![sym(0), c(7)])],
        11,
    );
    assert_eq!(
        outs[0].type_info.as_ref().unwrap().shape,
        vec![sym(0), c(7)]
    );
    assert_eq!(outs[0].type_info.as_ref().unwrap().dtype, DataType::Uint8);
    assert!(outs[1].type_info.as_ref().unwrap().shape.is_empty());
    assert_eq!(outs[1].type_info.as_ref().unwrap().dtype, DataType::Float32);
    assert!(outs[2].type_info.as_ref().unwrap().shape.is_empty());
    assert_eq!(outs[2].type_info.as_ref().unwrap().dtype, DataType::Uint8);
}

#[test]
fn pad_grows_dims() {
    let n = node("Pad", 2, 1);
    // pads = [0,0,1,1, 0,0,1,1] over rank 4 -> H,W grow by 2
    let pads = sd_vec(vec![c(0), c(0), c(1), c(1), c(0), c(0), c(1), c(1)]);
    let outs = run(&n, vec![f32in(vec![sym(0), c(3), c(32), c(32)]), pads], 13);
    assert_eq!(out_shape(&outs), vec![sym(0), c(3), c(34), c(34)]);
}

#[test]
fn pad_expanded_attention_axes_shape_and_bytes() {
    let n = node("Pad", 4, 1);
    // Expanded Attention pads a [2,3,4,4] bias on axis -1 by [0,2].
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4), c(4)]),
            sd_vec(vec![c(0), c(2)]),
            sd_float_scalar(DataType::Float32, f64::NEG_INFINITY),
            sd_vec(vec![c(-1)]),
        ],
        23,
    );
    let shape = out_shape(&outs);
    assert_eq!(shape, vec![c(2), c(3), c(4), c(6)]);
    let elements: i64 = shape.iter().map(|dim| dim.as_const().unwrap()).product();
    assert_eq!(elements, 144);
    assert_eq!(elements * DataType::Float32.byte_size() as i64, 576);
}

#[test]
fn pad_dynamic_pads_only_clear_selected_axes() {
    let mut n = node("Pad", 4, 1);
    n.inputs[2] = None;
    let outs = run(
        &n,
        vec![
            f32in(vec![c(2), c(3), c(4)]),
            tin(DataType::Int64, vec![c(2)]),
            NodeIo::default(),
            sd_vec(vec![c(-1)]),
        ],
        25,
    );
    let shape = out_shape(&outs);
    assert_eq!(&shape[..2], &[c(2), c(3)]);
    assert!(shape[2].as_symbol().is_some());
}

#[test]
fn pad_rejects_extent_beyond_isize_max() {
    let n = node("Pad", 2, 1);
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(1)]), sd_vec(vec![c(isize::MAX as i64), c(0)]),],
            25,
        )
        .is_err()
    );
}

#[test]
fn pad_symbolic_extent_rejects_guaranteed_overflow_and_stays_symbolic_normally() {
    let n = node("Pad", 2, 1);
    let error = try_run(
        &n,
        vec![
            f32in(vec![sym(0)]),
            sd_vec(vec![c(isize::MAX as i64), c(1)]),
        ],
        25,
    )
    .unwrap_err();
    assert!(error.to_string().contains("exceeds isize::MAX"));

    let outs = run(&n, vec![f32in(vec![sym(0)]), sd_vec(vec![c(1), c(1)])], 25);
    let shape = out_shape(&outs);
    let extent = &shape[0];
    assert!(extent.as_const().is_none());
    assert!(extent.as_symbol().is_some());
}

// --- unregistered op is permissive ---------------------------------------

#[test]
fn unregistered_op_leaves_output_unresolved() {
    let n = node("SomeExoticOp", 1, 1);
    let outs = run(&n, vec![f32in(vec![c(2), c(3)])], 13);
    assert!(outs[0].type_info.is_none());
}

// --- Issue #75 catalog extension: ConvTranspose / GridSample / activations /
//     normalization / generators / Einsum ---------------------------------

fn ints(values: &[i64]) -> Attribute {
    Attribute::Ints(values.to_vec())
}

#[test]
fn conv_transpose_deconvolution_formula_and_channels() {
    // stride*(input-1) + output_padding + effective_kernel - pads.
    let mut n = with_attr(node("ConvTranspose", 2, 1), "strides", ints(&[2, 2]));
    n = with_attr(n, "kernel_shape", ints(&[3, 3]));
    let out = run(
        &n,
        vec![
            f32in(vec![c(1), c(8), c(4), c(4)]),
            f32in(vec![c(8), c(16), c(3), c(3)]),
        ],
        11,
    );
    // channels = W.shape[1] * group(1) = 16; spatial = 2*(4-1)+3 = 9.
    assert_eq!(out_shape(&out), vec![c(1), c(16), c(9), c(9)]);
}

#[test]
fn conv_transpose_kernel_from_weight_and_output_padding() {
    let n = with_attr(node("ConvTranspose", 2, 1), "output_padding", ints(&[1, 0]));
    let out = run(
        &n,
        vec![
            f32in(vec![c(1), c(4), c(5), c(5)]),
            // kernel_shape omitted: taken from W trailing dims [2, 2].
            f32in(vec![c(4), c(6), c(2), c(2)]),
        ],
        11,
    );
    // effective_kernel = 2; stride defaults to 1.
    // dim0 = 1*(5-1)+1(output_padding)+2 = 7; dim1 = 1*(5-1)+0+2 = 6.
    assert_eq!(out_shape(&out), vec![c(1), c(6), c(7), c(6)]);
}

#[test]
fn conv_transpose_output_shape_attr_and_same_pad_and_symbolic() {
    // Explicit output_shape wins over the formula.
    let explicit = with_attr(
        with_attr(node("ConvTranspose", 2, 1), "strides", ints(&[2, 2])),
        "output_shape",
        ints(&[10, 12]),
    );
    let out = run(
        &explicit,
        vec![
            f32in(vec![c(1), c(3), c(4), c(4)]),
            f32in(vec![c(3), c(3), c(3), c(3)]),
        ],
        11,
    );
    assert_eq!(out_shape(&out), vec![c(1), c(3), c(10), c(12)]);

    // SAME_UPPER fixes each spatial extent at input*stride.
    let same = with_attr(
        with_attr(node("ConvTranspose", 2, 1), "strides", ints(&[2, 3])),
        "auto_pad",
        Attribute::String(b"SAME_UPPER".to_vec()),
    );
    let out = run(
        &same,
        vec![
            f32in(vec![c(1), c(2), c(4), c(5)]),
            f32in(vec![c(2), c(2), c(3), c(3)]),
        ],
        11,
    );
    assert_eq!(out_shape(&out), vec![c(1), c(2), c(8), c(15)]);

    // A symbolic spatial input keeps the rank but yields a fresh symbol there.
    let symbolic = with_attr(node("ConvTranspose", 2, 1), "kernel_shape", ints(&[3, 3]));
    let out = run(
        &symbolic,
        vec![
            f32in(vec![c(1), c(2), sym(1), c(4)]),
            f32in(vec![c(2), c(2), c(3), c(3)]),
        ],
        11,
    );
    let shape = out_shape(&out);
    assert_symbolic(&shape[2]);
    assert_eq!(shape[3], c(6)); // 1*(4-1)+3 = 6
}

#[test]
fn conv_transpose_is_gated_and_rejects_low_rank() {
    let n = node("ConvTranspose", 2, 1);
    assert!(
        run(
            &n,
            vec![
                f32in(vec![c(1), c(2), c(4), c(4)]),
                f32in(vec![c(2), c(2), c(3), c(3)])
            ],
            0,
        )[0]
        .type_info
        .is_none()
    );
    assert!(
        try_run(
            &n,
            vec![f32in(vec![c(1), c(2)]), f32in(vec![c(2), c(2)])],
            11
        )
        .is_err()
    );
}

#[test]
fn grid_sample_takes_spatial_from_grid() {
    let out = run(
        &node("GridSample", 2, 1),
        vec![
            f32in(vec![c(2), c(3), c(8), c(8)]),
            f32in(vec![c(2), c(10), c(12), c(2)]),
        ],
        16,
    );
    assert_eq!(out_shape(&out), vec![c(2), c(3), c(10), c(12)]);

    // Symbolic N/C flow through; unresolved grid leaves fresh spatial dims.
    let out = run(
        &node("GridSample", 2, 1),
        vec![f32in(vec![sym(1), c(3), c(8), c(8)])],
        20,
    );
    let shape = out_shape(&out);
    assert_eq!(shape[0], sym(1));
    assert_eq!(shape[1], c(3));
    assert_symbolic(&shape[2]);
    assert_symbolic(&shape[3]);

    // Gated at opset 16.
    assert!(
        run(
            &node("GridSample", 2, 1),
            vec![
                f32in(vec![c(2), c(3), c(8), c(8)]),
                f32in(vec![c(2), c(4), c(4), c(2)])
            ],
            15,
        )[0]
        .type_info
        .is_none()
    );
}

#[test]
fn added_activations_preserve_shape_and_respect_since_version() {
    for (op, since_version) in [("Shrink", 9), ("Celu", 12), ("HardSwish", 14), ("Mish", 18)] {
        let input = f32in(vec![sym(2), c(6)]);
        assert!(
            run(&node(op, 1, 1), vec![input.clone()], since_version - 1)[0]
                .type_info
                .is_none(),
            "{op} must be gated at {since_version}"
        );
        let out = run(&node(op, 1, 1), vec![input], since_version);
        assert_eq!(out_shape(&out), vec![sym(2), c(6)], "{op}");
        assert_eq!(out_dtype(&out), DataType::Float32, "{op}");
    }
}

#[test]
fn lrn_mvn_reverse_sequence_are_shape_preserving() {
    for (op, since_version) in [
        ("LRN", 1),
        ("MeanVarianceNormalization", 9),
        ("ReverseSequence", 10),
    ] {
        let n = node(op, if op == "ReverseSequence" { 2 } else { 1 }, 1);
        let input = tin(DataType::Float32, vec![c(2), sym(3), c(4)]);
        if since_version > 1 {
            assert!(
                run(&n, vec![input.clone()], since_version - 1)[0]
                    .type_info
                    .is_none(),
                "{op} must be gated"
            );
        }
        let out = run(&n, vec![input], since_version);
        assert_eq!(out_shape(&out), vec![c(2), sym(3), c(4)], "{op}");
    }
}

#[test]
fn random_generators_shape_and_dtype() {
    // RandomNormal/RandomUniform: shape from attr, dtype override.
    let n = with_attr(
        with_attr(node("RandomNormal", 0, 1), "shape", ints(&[2, 3])),
        "dtype",
        Attribute::Int(11),
    );
    let out = run(&n, vec![], 1);
    assert_eq!(out_shape(&out), vec![c(2), c(3)]);
    assert_eq!(out_dtype(&out), DataType::Float64);

    // Default dtype is Float32.
    let default = with_attr(node("RandomUniform", 0, 1), "shape", ints(&[4]));
    assert_eq!(out_dtype(&run(&default, vec![], 1)), DataType::Float32);

    // *Like/Bernoulli: mirror input shape, dtype override else input dtype.
    let like = with_attr(node("RandomNormalLike", 1, 1), "dtype", Attribute::Int(11));
    let out = run(&like, vec![tin(DataType::Int32, vec![c(3), sym(1)])], 1);
    assert_eq!(out_shape(&out), vec![c(3), sym(1)]);
    assert_eq!(out_dtype(&out), DataType::Float64);

    let bernoulli = node("Bernoulli", 1, 1);
    let out = run(&bernoulli, vec![tin(DataType::Float32, vec![c(5)])], 15);
    assert_eq!(out_dtype(&out), DataType::Float32);
    assert!(
        run(&bernoulli, vec![f32in(vec![c(5)])], 14)[0]
            .type_info
            .is_none()
    );
}

#[test]
fn multinomial_batch_by_sample_size() {
    let n = with_attr(node("Multinomial", 1, 1), "sample_size", Attribute::Int(5));
    let out = run(&n, vec![f32in(vec![c(4), c(7)])], 7);
    assert_eq!(out_shape(&out), vec![c(4), c(5)]);
    assert_eq!(out_dtype(&out), DataType::Int32);

    let as_int64 = with_attr(
        with_attr(node("Multinomial", 1, 1), "sample_size", Attribute::Int(2)),
        "dtype",
        Attribute::Int(7),
    );
    let out = run(&as_int64, vec![f32in(vec![sym(1), c(3)])], 7);
    assert_eq!(out_shape(&out), vec![sym(1), c(2)]);
    assert_eq!(out_dtype(&out), DataType::Int64);

    // Rank other than [batch, classes] is rejected.
    assert!(try_run(&node("Multinomial", 1, 1), vec![f32in(vec![c(3)])], 7).is_err());
}

fn einsum_node(equation: &str, n_in: usize) -> Node {
    with_attr(
        node("Einsum", n_in, 1),
        "equation",
        Attribute::String(equation.as_bytes().to_vec()),
    )
}

#[test]
fn einsum_requires_a_decodable_string_equation_attribute() {
    let input = vec![f32in(vec![c(2), c(3)])];

    let missing = try_run(&node("Einsum", 1, 1), input.clone(), 12).unwrap_err();
    assert_eq!(
        missing,
        ShapeInferError::MissingAttribute {
            op: "Einsum".into(),
            attr: "equation".into(),
        }
    );

    let wrong_type = try_run(
        &with_attr(node("Einsum", 1, 1), "equation", Attribute::Int(0)),
        input.clone(),
        12,
    )
    .unwrap_err();
    assert_eq!(
        wrong_type,
        ShapeInferError::MissingAttribute {
            op: "Einsum".into(),
            attr: "equation".into(),
        }
    );

    let invalid_utf8 = try_run(
        &with_attr(
            node("Einsum", 1, 1),
            "equation",
            Attribute::String(vec![b'i', 0xff, b'-', b'>', b'i']),
        ),
        input.clone(),
        12,
    )
    .unwrap_err();
    assert_eq!(
        invalid_utf8,
        ShapeInferError::Invalid {
            op: "Einsum".into(),
            detail: "attribute `equation` is not valid UTF-8: invalid byte sequence of length 1 starts at byte offset 1".into(),
        }
    );

    let valid = run(&einsum_node("ij->ji", 1), input, 12);
    assert_eq!(out_shape(&valid), vec![c(3), c(2)]);
}

#[test]
fn einsum_rejects_invalid_operator_arity() {
    for (inputs, outputs, expected) in [
        (0, 1, "expected at least one input"),
        (1, 0, "requires exactly 1 output"),
        (1, 2, "requires exactly 1 output"),
    ] {
        let n = with_attr(
            node("Einsum", inputs, outputs),
            "equation",
            Attribute::String(b"i->i".to_vec()),
        );
        let input_metadata = (0..inputs).map(|_| f32in(vec![c(2)])).collect();
        let error = try_run(&n, input_metadata, 12).unwrap_err();
        assert_invalid(error, "Einsum", expected);
    }
}

#[test]
fn einsum_matmul_transpose_and_implicit() {
    // Explicit matmul.
    let out = run(
        &einsum_node("ik,kj->ij", 2),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])],
        12,
    );
    assert_eq!(out_shape(&out), vec![c(2), c(4)]);

    // Transpose.
    let out = run(&einsum_node("ij->ji", 1), vec![f32in(vec![c(2), c(3)])], 12);
    assert_eq!(out_shape(&out), vec![c(3), c(2)]);

    // Implicit output: once-only labels in ASCII order (i, k).
    let out = run(
        &einsum_node("ij,jk", 2),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])],
        12,
    );
    assert_eq!(out_shape(&out), vec![c(2), c(4)]);

    // Mixed-case labels are distinct. Implicit output follows byte/ASCII
    // ordering, so `B` precedes `Z` and both precede lower-case labels.
    let out = run(
        &einsum_node("Za,aB", 2),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(3), c(4)])],
        12,
    );
    assert_eq!(out_shape(&out), vec![c(4), c(2)]);
}

#[test]
fn einsum_shape_inference_preserves_supported_float_dtype() {
    for dtype in [DataType::Float16, DataType::Float32] {
        let out = run(
            &einsum_node("ij->ji", 1),
            vec![tin(dtype, vec![c(2), c(3)])],
            12,
        );
        let type_info = out[0].type_info.as_ref().unwrap();
        assert_eq!(type_info.dtype, dtype);
        assert_eq!(type_info.shape, vec![c(3), c(2)]);
    }

    let rejected = try_run(
        &einsum_node("ij->ji", 1),
        vec![tin(DataType::BFloat16, vec![c(2), c(3)])],
        27,
    )
    .unwrap_err();
    assert_invalid(rejected, "Einsum", "not admitted by Einsum-12");

    let accepted = run(
        &einsum_node("ij->ji", 1),
        vec![tin(DataType::BFloat16, vec![c(2), c(3)])],
        28,
    );
    assert_eq!(
        accepted[0].type_info.as_ref().unwrap(),
        &TypeInfo::new(DataType::BFloat16, vec![c(3), c(2)])
    );
}

#[test]
fn einsum_ellipsis_broadcasts_fixed_rank_batch_dims() {
    let out = run(
        &einsum_node("...ij,...jk->...ik", 2),
        vec![f32in(vec![c(5), c(2), c(3)]), f32in(vec![c(5), c(3), c(4)])],
        12,
    );
    assert_eq!(out_shape(&out), vec![c(5), c(2), c(4)]);

    // A leading 1 broadcasts against a concrete batch extent.
    let out = run(
        &einsum_node("...ij,...jk->...ik", 2),
        vec![f32in(vec![c(1), c(2), c(3)]), f32in(vec![c(6), c(3), c(4)])],
        12,
    );
    assert_eq!(out_shape(&out), vec![c(6), c(2), c(4)]);

    // A term without ellipsis contributes no batch dimensions and broadcasts
    // across the fixed-rank ellipsis from the other term.
    let out = run(
        &einsum_node("ij,...jk->...ik", 2),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(6), c(5), c(3), c(4)])],
        12,
    );
    assert_eq!(out_shape(&out), vec![c(6), c(5), c(2), c(4)]);

    // Implicit output retains the ellipsis first and once-only labels in ASCII
    // order, even when another term has no ellipsis.
    let out = run(
        &einsum_node("ij,...jk", 2),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(6), c(5), c(3), c(4)])],
        12,
    );
    assert_eq!(out_shape(&out), vec![c(6), c(5), c(2), c(4)]);
}

#[test]
fn einsum_symbolic_dims_and_gating() {
    let out = run(
        &einsum_node("ik,kj->ij", 2),
        vec![f32in(vec![sym(1), c(3)]), f32in(vec![c(3), c(4)])],
        12,
    );
    let shape = out_shape(&out);
    assert_symbolic(&shape[0]);
    assert_eq!(shape[1], c(4));

    // Gated at opset 12.
    assert!(
        run(&einsum_node("ij->ji", 1), vec![f32in(vec![c(2), c(3)])], 11,)[0]
            .type_info
            .is_none()
    );

    // A provably invalid equation now fails explicitly instead of producing a
    // plan that a native execution provider could misinterpret.
    let error = try_run(
        &einsum_node("ij->ji", 1),
        vec![f32in(vec![c(2), c(3), c(4)])],
        12,
    )
    .unwrap_err();
    assert_invalid(error, "Einsum", "input #0 rank 3");
}

#[test]
fn einsum_scalar_diagonal_reduction_zero_and_whitespace() {
    let scalar = run(&einsum_node(" -> ", 1), vec![f32in(vec![])], 12);
    assert_eq!(out_shape(&scalar), Vec::<DimExpr>::new());

    let scalar_times_vector = run(
        &einsum_node(" , i -> i ", 2),
        vec![f32in(vec![]), f32in(vec![c(7)])],
        12,
    );
    assert_eq!(out_shape(&scalar_times_vector), vec![c(7)]);

    let diagonal = run(
        &einsum_node("...ii -> ...i", 1),
        vec![f32in(vec![c(0), c(5), c(5)])],
        12,
    );
    assert_eq!(out_shape(&diagonal), vec![c(0), c(5)]);

    let reduction = run(&einsum_node("ij->i", 1), vec![f32in(vec![c(2), c(3)])], 12);
    assert_eq!(out_shape(&reduction), vec![c(2)]);
}

#[test]
fn einsum_rejects_unequal_explicit_ellipsis_ranks_and_non_space_whitespace() {
    // ONNX opset 12 requires every explicit ellipsis to cover one fixed number
    // of dimensions. NumPy accepts these by right-aligning unequal ranks.
    for (equation, inputs, detail) in [
        (
            "...ij,j...k->...ik",
            vec![
                f32in(vec![c(5), c(2), c(3)]),
                f32in(vec![c(3), c(6), c(5), c(4)]),
            ],
            "input term #1 explicit ellipsis has expansion rank 2, but input term #0 explicit ellipsis has expansion rank 1",
        ),
        (
            "...ij,...jk->...ik",
            vec![f32in(vec![c(2), c(3)]), f32in(vec![c(6), c(5), c(3), c(4)])],
            "input term #1 explicit ellipsis has expansion rank 2, but input term #0 explicit ellipsis has expansion rank 0",
        ),
    ] {
        let error = try_run(&einsum_node(equation, 2), inputs, 12).unwrap_err();
        assert_invalid(error, "Einsum", detail);
    }

    for invalid in ['\t', '\n', '\u{00a0}', '\u{2003}', '\u{2028}'] {
        let equation = format!("i{invalid}->i");
        let error = try_run(&einsum_node(&equation, 1), vec![f32in(vec![c(7)])], 12).unwrap_err();
        assert_invalid(
            error,
            "Einsum",
            &format!("invalid character `{invalid}` at normalized byte offset 1"),
        );
    }
}

#[test]
fn einsum_invalid_equations_dimensions_and_dtypes_are_actionable() {
    let invalid_cases = [
        (
            &einsum_node("ij->ji", 1),
            vec![f32in(vec![c(2), c(3), c(4)])],
            "input #0 rank 3",
        ),
        (
            &einsum_node("ij->ii", 1),
            vec![f32in(vec![c(2), c(3)])],
            "output label `i` appears more than once",
        ),
        (
            &einsum_node("ij->ik", 1),
            vec![f32in(vec![c(2), c(3)])],
            "output label `k` does not appear",
        ),
        (
            &einsum_node("i$,i->", 2),
            vec![f32in(vec![c(2)]), f32in(vec![c(2)])],
            "invalid character `$`",
        ),
        (
            &einsum_node("ii->i", 1),
            vec![f32in(vec![c(2), c(3)])],
            "label `i` requires equal dimensions",
        ),
        (
            &einsum_node("...i,...i->...i", 2),
            vec![f32in(vec![c(2), c(3)]), f32in(vec![c(5), c(3)])],
            "ellipsis axis #0 cannot broadcast",
        ),
    ];
    for (node, inputs, detail) in invalid_cases {
        let error = try_run(node, inputs, 12).unwrap_err();
        assert_invalid(error, "Einsum", detail);
    }

    let mismatch = try_run(
        &einsum_node("i,i->i", 2),
        vec![f32in(vec![c(2)]), tin(DataType::Int32, vec![c(2)])],
        12,
    )
    .unwrap_err();
    assert_invalid(mismatch, "Einsum", "input #1 has dtype Int32");

    let unsupported = try_run(
        &einsum_node("i->i", 1),
        vec![tin(DataType::Bool, vec![c(2)])],
        12,
    )
    .unwrap_err();
    assert_invalid(
        unsupported,
        "Einsum",
        "dtype Bool, which is not admitted by Einsum-12",
    );

    // Missing type/shape metadata remains best-effort and leaves the output
    // unresolved, preserving the crate-wide permissive inference contract.
    assert!(
        run(&einsum_node("i->i", 1), vec![NodeIo::default()], 12)[0]
            .type_info
            .is_none()
    );
}

#[test]
fn einsum_ellipsis_uses_the_inference_context_lineage_chokepoint() {
    let node = einsum_node("...i,...i->...i", 2);
    let reg = InferenceRegistry::default_registry();
    let mut imports = HashMap::new();
    imports.insert(String::new(), 12);
    let anonymous = SymbolId(0x8000_0001);
    let named = SymbolId(7);
    let mut interner = SymbolInterner::new(0x8000_0000);
    let output = reg
        .infer_node(
            &node,
            &imports,
            vec![
                f32in(vec![DimExpr::symbol(anonymous), c(3)]),
                f32in(vec![DimExpr::symbol(named), c(3)]),
            ],
            MergePolicy::Permissive,
            &mut interner,
        )
        .unwrap();
    assert_eq!(out_shape(&output), vec![DimExpr::symbol(named), c(3)]);
    assert_eq!(interner.unifications(), &[(anonymous, named)]);
}

#[test]
fn det_collapses_trailing_square_matrix_and_gates_since_version() {
    // Batched input: the trailing [M, M] matrix collapses to a scalar per batch.
    let out = run(&node("Det", 1, 1), vec![f32in(vec![c(2), c(3), c(3)])], 11);
    assert_eq!(out_shape(&out), vec![c(2)]);
    assert_eq!(out_dtype(&out), DataType::Float32);

    // A bare square matrix yields a rank-0 output.
    let scalar = run(&node("Det", 1, 1), vec![f32in(vec![c(4), c(4)])], 11);
    assert_eq!(out_shape(&scalar), Vec::<DimExpr>::new());

    // Symbolic batch dims are preserved; only the matrix axes are dropped.
    let symbolic = run(
        &node("Det", 1, 1),
        vec![f32in(vec![sym(1), sym(2), c(5), c(5)])],
        11,
    );
    assert_eq!(out_shape(&symbolic), vec![sym(1), sym(2)]);

    // since_version 11: opset 10 leaves the output unresolved.
    assert!(
        run(&node("Det", 1, 1), vec![f32in(vec![c(3), c(3)])], 10)[0]
            .type_info
            .is_none()
    );

    // A rank-1 input has no matrix to reduce.
    assert!(try_run(&node("Det", 1, 1), vec![f32in(vec![c(3)])], 11).is_err());
}

#[test]
fn lp_pool_and_global_lp_pool_reuse_spatial_rules() {
    // LpPool applies the same windowed spatial formula as AveragePool.
    let lp = with_attr(
        node("LpPool", 1, 1),
        "kernel_shape",
        Attribute::Ints(vec![2]),
    );
    let out = run(&lp, vec![f32in(vec![c(1), c(3), c(8)])], 18);
    assert_eq!(out_shape(&out), vec![c(1), c(3), c(7)]);

    // A symbolic spatial extent degrades to a fresh symbol but keeps the rank.
    let symbolic = run(&lp, vec![f32in(vec![c(1), c(3), sym(5)])], 18);
    let shape = out_shape(&symbolic);
    assert_eq!(shape[..2], [c(1), c(3)]);
    assert_symbolic(&shape[2]);

    // kernel_shape is required.
    assert!(
        try_run(
            &node("LpPool", 1, 1),
            vec![f32in(vec![c(1), c(3), c(8)])],
            18
        )
        .is_err()
    );

    // GlobalLpPool collapses every spatial dim to 1.
    let global = run(
        &node("GlobalLpPool", 1, 1),
        vec![f32in(vec![c(2), c(3), c(7), c(7)])],
        1,
    );
    assert_eq!(out_shape(&global), vec![c(2), c(3), c(1), c(1)]);
}

#[test]
fn max_unpool_uses_transpose_formula_or_explicit_output_shape() {
    // Transpose formula: stride*(input-1) - pads + kernel.
    let unpool = with_attr(
        with_attr(
            node("MaxUnpool", 2, 1),
            "kernel_shape",
            Attribute::Ints(vec![2, 2]),
        ),
        "strides",
        Attribute::Ints(vec![2, 2]),
    );
    let out = run(
        &unpool,
        vec![
            f32in(vec![c(1), c(1), c(2), c(2)]),
            tin(DataType::Int64, vec![c(1), c(1), c(2), c(2)]),
        ],
        9,
    );
    assert_eq!(out_shape(&out), vec![c(1), c(1), c(4), c(4)]);

    // A symbolic spatial extent degrades to a fresh symbol.
    let symbolic = run(
        &unpool,
        vec![
            f32in(vec![c(1), c(1), sym(3), c(2)]),
            tin(DataType::Int64, vec![c(1), c(1), sym(3), c(2)]),
        ],
        9,
    );
    let shape = out_shape(&symbolic);
    assert_eq!(shape[..2], [c(1), c(1)]);
    assert_symbolic(&shape[2]);
    assert_eq!(shape[3], c(4));

    // The optional output_shape input (slot 2) overrides the formula.
    let out = run(
        &node("MaxUnpool", 3, 1),
        vec![
            f32in(vec![c(1), c(1), c(2), c(2)]),
            tin(DataType::Int64, vec![c(1), c(1), c(2), c(2)]),
            sd_vec(vec![c(1), c(1), c(5), c(5)]),
        ],
        11,
    );
    assert_eq!(out_shape(&out), vec![c(1), c(1), c(5), c(5)]);

    // since_version 9: opset 8 leaves the output unresolved.
    assert!(
        run(
            &unpool,
            vec![
                f32in(vec![c(1), c(1), c(2), c(2)]),
                tin(DataType::Int64, vec![c(1), c(1), c(2), c(2)])
            ],
            8
        )[0]
        .type_info
        .is_none()
    );
}

#[test]
fn col2im_folds_columns_back_into_image() {
    // data [N, C*prod(block), L], image_shape [4, 4], block_shape [2, 2]:
    // C = 12 / (2*2) = 3, so output is [1, 3, 4, 4].
    let out = run(
        &node("Col2Im", 3, 1),
        vec![
            f32in(vec![c(1), c(12), c(9)]),
            sd_vec(vec![c(4), c(4)]),
            sd_vec(vec![c(2), c(2)]),
        ],
        18,
    );
    assert_eq!(out_shape(&out), vec![c(1), c(3), c(4), c(4)]);

    // A symbolic image extent stays symbolic in the output.
    let symbolic = run(
        &node("Col2Im", 3, 1),
        vec![
            f32in(vec![c(1), c(12), c(9)]),
            sd_vec(vec![sym(7), c(4)]),
            sd_vec(vec![c(2), c(2)]),
        ],
        18,
    );
    let shape = out_shape(&symbolic);
    assert_eq!(shape[..2], [c(1), c(3)]);
    assert_symbolic(&shape[2]);
    assert_eq!(shape[3], c(4));

    // An unknown block volume leaves the channel count symbolic.
    let unknown_channels = run(
        &node("Col2Im", 3, 1),
        vec![
            f32in(vec![c(1), c(12), c(9)]),
            sd_vec(vec![c(4), c(4)]),
            tin(DataType::Int64, vec![c(2)]),
        ],
        18,
    );
    let shape = out_shape(&unknown_channels);
    assert_eq!(shape[0], c(1));
    assert_symbolic(&shape[1]);
    assert_eq!(shape[2..], [c(4), c(4)]);

    // Rank other than 3 is rejected.
    assert!(
        try_run(
            &node("Col2Im", 3, 1),
            vec![
                f32in(vec![c(1), c(12)]),
                sd_vec(vec![c(4)]),
                sd_vec(vec![c(2)])
            ],
            18,
        )
        .is_err()
    );
}

#[test]
fn center_crop_pad_resizes_selected_axes() {
    // axes [0, 1]: axes 0 and 1 take the target extents, axis 2 is untouched.
    let cropped = with_attr(
        node("CenterCropPad", 2, 1),
        "axes",
        Attribute::Ints(vec![0, 1]),
    );
    let out = run(
        &cropped,
        vec![f32in(vec![c(10), c(8), c(3)]), sd_vec(vec![c(20), c(5)])],
        18,
    );
    assert_eq!(out_shape(&out), vec![c(20), c(5), c(3)]);

    // Omitting axes defaults to every axis.
    let out = run(
        &node("CenterCropPad", 2, 1),
        vec![
            f32in(vec![c(4), c(5), c(6)]),
            sd_vec(vec![c(7), c(2), c(9)]),
        ],
        18,
    );
    assert_eq!(out_shape(&out), vec![c(7), c(2), c(9)]);

    // A symbolic (statically unknown) target extent degrades the targeted axis
    // to a fresh symbol; untargeted axes are copied through unchanged.
    let symbolic = run(
        &cropped,
        vec![f32in(vec![c(10), c(8), c(3)]), sd_vec(vec![sym(9), c(5)])],
        18,
    );
    let shape = out_shape(&symbolic);
    assert_symbolic(&shape[0]);
    assert_eq!(shape[1..], [c(5), c(3)]);

    // since_version 18: opset 17 leaves the output unresolved.
    assert!(
        run(
            &node("CenterCropPad", 2, 1),
            vec![f32in(vec![c(4), c(5)]), sd_vec(vec![c(2), c(2)])],
            17,
        )[0]
        .type_info
        .is_none()
    );
}

// --- signal-processing family --------------------------------------------

/// An int64 scalar shape-data input (e.g. a DFT length or window size).
fn i64_scalar(value: i64) -> NodeIo {
    sd_int_scalar(DataType::Int64, c(value))
}

#[test]
fn dft_v17_real_to_complex_with_length_and_onesided() {
    // Real input [batch, N, 1] with no dft_length: axis (default 1) unchanged,
    // trailing dim coerced to complex 2.
    let plain = run(
        &node("DFT", 1, 1),
        vec![f32in(vec![sym(1), c(16), c(1)])],
        17,
    );
    assert_eq!(out_dtype(&plain), DataType::Float32);
    assert_eq!(out_shape(&plain)[0], sym(1));
    assert_eq!(out_shape(&plain)[1..], [c(16), c(2)]);

    // dft_length overrides the signal axis.
    let with_length = run(
        &node("DFT", 2, 1),
        vec![f32in(vec![c(2), c(16), c(1)]), i64_scalar(10)],
        17,
    );
    assert_eq!(out_shape(&with_length), vec![c(2), c(10), c(2)]);

    // onesided halves the (dft_length) axis to floor(n/2)+1 = 6.
    let onesided = run(
        &with_attr(node("DFT", 2, 1), "onesided", Attribute::Int(1)),
        vec![f32in(vec![c(2), c(16), c(1)]), i64_scalar(10)],
        17,
    );
    assert_eq!(out_shape(&onesided), vec![c(2), c(6), c(2)]);
}

#[test]
fn dft_v17_axis_attribute_negative_and_symbolic_axis_degrades() {
    // Negative axis counts from the back; axis=-2 on rank 4 targets index 2.
    let neg_axis = run(
        &with_attr(node("DFT", 1, 1), "axis", Attribute::Int(-2)),
        vec![f32in(vec![c(2), c(3), c(8), c(2)])],
        17,
    );
    assert_eq!(out_shape(&neg_axis), vec![c(2), c(3), c(8), c(2)]);

    // onesided over a symbolic signal axis cannot be computed -> fresh symbol.
    let symbolic = run(
        &with_attr(node("DFT", 1, 1), "onesided", Attribute::Int(1)),
        vec![f32in(vec![c(2), sym(5), c(1)])],
        17,
    );
    let shape = out_shape(&symbolic);
    assert_eq!(shape[0], c(2));
    assert_symbolic(&shape[1]);
    assert_eq!(shape[2], c(2));
}

#[test]
fn dft_since_version_rank_and_axis_validation() {
    // since_version 17: opset 16 leaves the output unresolved.
    assert!(
        run(&node("DFT", 1, 1), vec![f32in(vec![c(2), c(8), c(1)])], 16)[0]
            .type_info
            .is_none()
    );

    // Rank < 2 is a contract violation.
    assert!(matches!(
        try_run(&node("DFT", 1, 1), vec![f32in(vec![c(8)])], 17),
        Err(ShapeInferError::InvalidRank { .. })
    ));

    // The trailing complex dimension is never a valid axis.
    let bad_axis = with_attr(node("DFT", 1, 1), "axis", Attribute::Int(2));
    assert_invalid(
        try_run(&bad_axis, vec![f32in(vec![c(2), c(8), c(2)])], 17).unwrap_err(),
        "DFT",
        "axis 2 is invalid",
    );
}

#[test]
fn dft_v20_axis_input_default_and_unknown() {
    // Opset 20 default axis is -2 (last signal axis).
    let default_axis = run(&node("DFT", 1, 1), vec![f32in(vec![c(2), c(8), c(1)])], 20);
    assert_eq!(out_shape(&default_axis), vec![c(2), c(8), c(2)]);

    // Axis supplied as a resolved input (dft_length slot omitted).
    let mut axis_only = node("DFT", 3, 1);
    axis_only.inputs[1] = None;
    let axis_input = run(
        &axis_only,
        vec![
            f32in(vec![c(2), c(8), c(4), c(1)]),
            NodeIo::default(),
            i64_scalar(1),
        ],
        20,
    );
    assert_eq!(out_shape(&axis_input), vec![c(2), c(8), c(4), c(2)]);

    // Axis input present but unknown, without onesided/dft_length: the input
    // shape is preserved except the trailing complex 2.
    let mut unknown = node("DFT", 3, 1);
    unknown.inputs[1] = None;
    let unknown_plain = run(
        &unknown,
        vec![
            f32in(vec![c(2), c(8), c(1)]),
            NodeIo::default(),
            NodeIo::default(),
        ],
        20,
    );
    assert_eq!(out_shape(&unknown_plain), vec![c(2), c(8), c(2)]);

    // Axis input unknown *and* onesided: every signal extent is unknowable, so
    // each becomes a fresh symbol, with the trailing 2 pinned.
    let mut unknown_os = with_attr(node("DFT", 3, 1), "onesided", Attribute::Int(1));
    unknown_os.inputs[1] = None;
    let unknown_onesided = run(
        &unknown_os,
        vec![
            f32in(vec![c(2), c(8), c(1)]),
            NodeIo::default(),
            NodeIo::default(),
        ],
        20,
    );
    let shape = out_shape(&unknown_onesided);
    assert_eq!(shape.len(), 3);
    assert_symbolic(&shape[0]);
    assert_symbolic(&shape[1]);
    assert_eq!(shape[2], c(2));
}

#[test]
fn stft_frame_length_window_and_onesided_default() {
    // frame_length = 16, frame_step = 4, signal_length = 64, onesided default 1
    // -> frames = (64-16)/4 + 1 = 13, bins = 16/2 + 1 = 9.
    let mut frame_length_node = node("STFT", 4, 1);
    frame_length_node.inputs[2] = None;
    let with_frame_length = run(
        &frame_length_node,
        vec![
            f32in(vec![c(2), c(64), c(1)]),
            i64_scalar(4),
            NodeIo::default(),
            i64_scalar(16),
        ],
        17,
    );
    assert_eq!(out_shape(&with_frame_length), vec![c(2), c(13), c(9), c(2)]);

    // Two-sided: bins == frame_length.
    let mut two_sided_node = with_attr(node("STFT", 4, 1), "onesided", Attribute::Int(0));
    two_sided_node.inputs[2] = None;
    let two_sided = run(
        &two_sided_node,
        vec![
            f32in(vec![c(2), c(64), c(1)]),
            i64_scalar(4),
            NodeIo::default(),
            i64_scalar(16),
        ],
        17,
    );
    assert_eq!(out_shape(&two_sided), vec![c(2), c(13), c(16), c(2)]);

    // Transform size taken from the window vector length when frame_length is
    // absent (input slot 3 skipped).
    let mut windowed = node("STFT", 4, 1);
    windowed.inputs[3] = None;
    let window_driven = run(
        &windowed,
        vec![
            f32in(vec![c(1), c(32), c(1)]),
            i64_scalar(8),
            f32in(vec![c(8)]),
            NodeIo::default(),
        ],
        17,
    );
    // frames = (32-8)/8 + 1 = 4, bins = 8/2 + 1 = 5.
    assert_eq!(out_shape(&window_driven), vec![c(1), c(4), c(5), c(2)]);
}

#[test]
fn stft_symbolic_signal_degrades_frame_count() {
    // Unknown signal length -> unknown frame count (fresh), but batch/bins/2
    // stay resolved.
    let mut stft = node("STFT", 4, 1);
    stft.inputs[2] = None;
    let symbolic = run(
        &stft,
        vec![
            tin(DataType::Float32, vec![sym(3), sym(4), c(1)]),
            i64_scalar(4),
            NodeIo::default(),
            i64_scalar(16),
        ],
        17,
    );
    let shape = out_shape(&symbolic);
    assert_eq!(shape[0], sym(3));
    assert_symbolic(&shape[1]);
    assert_eq!(shape[2], c(9));
    assert_eq!(shape[3], c(2));

    // since_version boundary: opset 16 is unresolved.
    assert!(
        run(
            &stft,
            vec![
                f32in(vec![c(2), c(64), c(1)]),
                i64_scalar(4),
                NodeIo::default(),
                i64_scalar(16),
            ],
            16,
        )[0]
        .type_info
        .is_none()
    );
}

#[test]
fn stft_rejects_invalid_static_contracts() {
    let mut frame_length_node = node("STFT", 4, 1);
    frame_length_node.inputs[2] = None;

    let zero_step = try_run(
        &frame_length_node,
        vec![
            f32in(vec![c(1), c(8), c(1)]),
            i64_scalar(0),
            NodeIo::default(),
            i64_scalar(4),
        ],
        17,
    )
    .unwrap_err()
    .to_string();
    assert!(zero_step.contains("frame_step must be greater than zero"));

    let short = try_run(
        &frame_length_node,
        vec![
            f32in(vec![c(1), c(3), c(1)]),
            i64_scalar(1),
            NodeIo::default(),
            i64_scalar(4),
        ],
        17,
    )
    .unwrap_err()
    .to_string();
    assert!(short.contains("complete unpadded frames"));

    let complex_onesided = try_run(
        &frame_length_node,
        vec![
            f32in(vec![c(1), c(8), c(2)]),
            i64_scalar(2),
            NodeIo::default(),
            i64_scalar(4),
        ],
        17,
    )
    .unwrap_err()
    .to_string();
    assert!(complex_onesided.contains("onesided=1 requires a real signal"));

    let mismatched = try_run(
        &node("STFT", 4, 1),
        vec![
            f32in(vec![c(1), c(8), c(1)]),
            i64_scalar(2),
            f32in(vec![c(3)]),
            i64_scalar(4),
        ],
        17,
    )
    .unwrap_err()
    .to_string();
    assert!(mismatched.contains("window length 3 must equal frame_length 4"));
}

#[test]
fn mel_weight_matrix_shape_and_output_datatype() {
    // dft_length = 16 -> rows = 16/2 + 1 = 9; num_mel_bins = 4 -> cols = 4.
    let concrete = run(
        &with_attr(
            node("MelWeightMatrix", 5, 1),
            "output_datatype",
            Attribute::Int(11),
        ),
        vec![
            i64_scalar(4),
            i64_scalar(16),
            i64_scalar(16000),
            sd_float_scalar(DataType::Float32, 0.0),
            sd_float_scalar(DataType::Float32, 8000.0),
        ],
        17,
    );
    assert_eq!(out_dtype(&concrete), DataType::Float64);
    assert_eq!(out_shape(&concrete), vec![c(9), c(4)]);

    // Unknown extents degrade to fresh symbols, but the rank-2 shape and the
    // default Float32 dtype are still emitted.
    let symbolic = run(
        &node("MelWeightMatrix", 5, 1),
        vec![
            NodeIo::default(),
            NodeIo::default(),
            NodeIo::default(),
            NodeIo::default(),
            NodeIo::default(),
        ],
        17,
    );
    assert_eq!(out_dtype(&symbolic), DataType::Float32);
    let shape = out_shape(&symbolic);
    assert_eq!(shape.len(), 2);
    assert_symbolic(&shape[0]);
    assert_symbolic(&shape[1]);
}

#[test]
fn window_generators_length_dtype_and_since_version() {
    for op in ["HannWindow", "HammingWindow", "BlackmanWindow"] {
        // Concrete size with an explicit output_datatype (10 = Float16).
        let concrete = run(
            &with_attr(node(op, 1, 1), "output_datatype", Attribute::Int(10)),
            vec![i64_scalar(320)],
            17,
        );
        assert_eq!(out_dtype(&concrete), DataType::Float16, "{op}");
        assert_eq!(out_shape(&concrete), vec![c(320)], "{op}");

        // Unknown size -> rank-1 with a fresh symbol; default Float32 dtype.
        let symbolic = run(&node(op, 1, 1), vec![NodeIo::default()], 17);
        assert_eq!(out_dtype(&symbolic), DataType::Float32, "{op}");
        let shape = out_shape(&symbolic);
        assert_eq!(shape.len(), 1, "{op}");
        assert_symbolic(&shape[0]);

        // since_version 17: opset 16 leaves the output unresolved.
        assert!(
            run(&node(op, 1, 1), vec![i64_scalar(320)], 16)[0]
                .type_info
                .is_none(),
            "{op}"
        );
    }
}

#[test]
fn affine_grid_2d_3d_symbolic_and_since_version() {
    // 2-D: theta (N, 2, 3), size = [N, C, H, W] -> grid [N, H, W, 2].
    let two_d = run(
        &node("AffineGrid", 2, 1),
        vec![
            f32in(vec![c(5), c(2), c(3)]),
            sd_vec(vec![c(5), c(3), c(8), c(9)]),
        ],
        20,
    );
    assert_eq!(out_shape(&two_d), vec![c(5), c(8), c(9), c(2)]);

    // 3-D: size = [N, C, D, H, W] -> grid [N, D, H, W, 3].
    let three_d = run(
        &node("AffineGrid", 2, 1),
        vec![
            f32in(vec![c(5), c(3), c(4)]),
            sd_vec(vec![c(5), c(3), c(6), c(8), c(9)]),
        ],
        20,
    );
    assert_eq!(out_shape(&three_d), vec![c(5), c(6), c(8), c(9), c(3)]);

    // Symbolic extents in the size vector propagate through.
    let symbolic = run(
        &node("AffineGrid", 2, 1),
        vec![
            f32in(vec![sym(1), c(2), c(3)]),
            sd_vec(vec![sym(1), c(3), sym(2), c(9)]),
        ],
        20,
    );
    let shape = out_shape(&symbolic);
    assert_eq!(shape[0], sym(1));
    assert_eq!(shape[1], sym(2));
    assert_eq!(shape[2..], [c(9), c(2)]);

    // since_version 20: opset 19 leaves the output unresolved.
    assert!(
        run(
            &node("AffineGrid", 2, 1),
            vec![
                f32in(vec![c(5), c(2), c(3)]),
                sd_vec(vec![c(5), c(3), c(8), c(9)])
            ],
            19,
        )[0]
        .type_info
        .is_none()
    );
}

// --- loss family ----------------------------------------------------------

#[test]
fn negative_log_likelihood_loss_reduction_shapes() {
    // reduction = none -> (N, d1, ..., dk) from input (N, C, d1, ..., dk).
    let none = run(
        &with_attr(
            node("NegativeLogLikelihoodLoss", 2, 1),
            "reduction",
            Attribute::String(b"none".to_vec()),
        ),
        vec![
            f32in(vec![sym(1), c(7), c(4), c(5)]),
            tin(DataType::Int64, vec![sym(1), c(4), c(5)]),
        ],
        13,
    );
    assert_eq!(out_dtype(&none), DataType::Float32);
    assert_eq!(out_shape(&none)[0], sym(1));
    assert_eq!(out_shape(&none)[1..], [c(4), c(5)]);

    // Default reduction (mean) -> scalar.
    let mean = run(
        &node("NegativeLogLikelihoodLoss", 2, 1),
        vec![f32in(vec![c(3), c(7)]), tin(DataType::Int64, vec![c(3)])],
        13,
    );
    assert_eq!(out_shape(&mean), Vec::<DimExpr>::new());

    // Rank < 2 input is a contract violation.
    assert!(matches!(
        try_run(
            &node("NegativeLogLikelihoodLoss", 2, 1),
            vec![f32in(vec![c(3)]), tin(DataType::Int64, vec![])],
            13,
        ),
        Err(ShapeInferError::InvalidRank { .. })
    ));

    // since_version 12: opset 11 leaves the output unresolved.
    assert!(
        run(
            &node("NegativeLogLikelihoodLoss", 2, 1),
            vec![f32in(vec![c(3), c(7)]), tin(DataType::Int64, vec![c(3)])],
            11,
        )[0]
        .type_info
        .is_none()
    );
}

#[test]
fn softmax_cross_entropy_loss_reduction_and_log_prob() {
    // reduction = none -> loss follows the labels' shape; log_prob mirrors
    // scores.
    let none = run(
        &with_attr(
            node("SoftmaxCrossEntropyLoss", 2, 2),
            "reduction",
            Attribute::String(b"none".to_vec()),
        ),
        vec![
            f32in(vec![sym(1), c(10), c(4)]),
            tin(DataType::Int64, vec![sym(1), c(4)]),
        ],
        13,
    );
    assert_eq!(out_shape(&none)[0], sym(1));
    assert_eq!(out_shape(&none)[1..], [c(4)]);
    let log_prob = &none[1].type_info.as_ref().unwrap();
    assert_eq!(log_prob.dtype, DataType::Float32);
    assert_eq!(log_prob.shape, vec![sym(1), c(10), c(4)]);

    // Default reduction (mean) -> scalar loss; log_prob still mirrors scores.
    let mean = run(
        &node("SoftmaxCrossEntropyLoss", 2, 2),
        vec![f32in(vec![c(3), c(10)]), tin(DataType::Int64, vec![c(3)])],
        13,
    );
    assert_eq!(out_shape(&mean), Vec::<DimExpr>::new());
    assert_eq!(mean[1].type_info.as_ref().unwrap().shape, vec![c(3), c(10)]);

    // since_version 12: opset 11 leaves the output unresolved.
    assert!(
        run(
            &node("SoftmaxCrossEntropyLoss", 2, 1),
            vec![f32in(vec![c(3), c(10)]), tin(DataType::Int64, vec![c(3)])],
            11,
        )[0]
        .type_info
        .is_none()
    );
}

// --- container types: the Sequence subset (#449) ------------------------------

/// The container [`ValueType`] of output slot 0.
fn out_value_type(outs: &[NodeIo]) -> &ValueType {
    outs[0]
        .value_type
        .as_ref()
        .expect("output container type resolved")
}

/// The tensor leaf of a sequence output's element type.
fn seq_element_tensor(outs: &[NodeIo]) -> &TensorType {
    out_value_type(outs)
        .as_sequence_element()
        .expect("output is a sequence")
        .as_tensor()
        .expect("sequence element is a tensor")
}

/// A `NodeIo` carrying a sequence-of-tensor container input.
fn seq_in(dtype: DataType, shape: Option<Vec<DimExpr>>) -> NodeIo {
    let tensor = match shape {
        Some(shape) => TensorType::new(dtype, shape),
        None => TensorType::dtype_only(dtype),
    };
    NodeIo::container(ValueType::sequence(ValueType::Tensor(tensor)))
}

#[test]
fn sequence_empty_element_dtype_follows_attr_and_defaults_to_float32() {
    // Default (no `dtype` attr) is Float32, per the ONNX spec.
    let default = run(&node("SequenceEmpty", 0, 1), vec![], 13);
    let element = seq_element_tensor(&default);
    assert_eq!(element.dtype, DataType::Float32);
    assert!(
        element.shape.is_none(),
        "empty sequence element shape is unknown"
    );

    // Every recognised `dtype` attribute value flows through to the element.
    for dtype in [DataType::Int64, DataType::Float16, DataType::Bool] {
        let out = run(
            &with_attr(
                node("SequenceEmpty", 0, 1),
                "dtype",
                Attribute::Int(dtype.to_onnx() as i64),
            ),
            vec![],
            13,
        );
        assert_eq!(seq_element_tensor(&out).dtype, dtype);
    }
}

#[test]
fn sequence_construct_matching_element_dtypes_yield_common_element() {
    let out = run(
        &node("SequenceConstruct", 2, 1),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(2), c(3)])],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.dtype, DataType::Float32);
    assert_eq!(element.shape.as_deref(), Some([c(2), c(3)].as_slice()));
}

#[test]
fn sequence_construct_mismatched_element_dtypes_error() {
    let error = try_run(
        &node("SequenceConstruct", 2, 1),
        vec![f32in(vec![c(2)]), tin(DataType::Int64, vec![c(2)])],
        13,
    )
    .expect_err("mismatched element dtypes must be rejected");
    assert_invalid(error, "SequenceConstruct", "share a dtype");
}

#[test]
fn sequence_construct_disagreeing_extents_degrade_to_symbol_but_keep_rank() {
    let out = run(
        &node("SequenceConstruct", 2, 1),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(2), c(5)])],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.dtype, DataType::Float32);
    let shape = element.shape.as_ref().expect("rank preserved");
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], c(2), "agreeing extent is preserved");
    assert_symbolic(&shape[1]);
}

#[test]
fn sequence_construct_differing_ranks_yield_unknown_element_shape() {
    let out = run(
        &node("SequenceConstruct", 2, 1),
        vec![f32in(vec![c(2), c(3)]), f32in(vec![c(2)])],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.dtype, DataType::Float32);
    assert!(
        element.shape.is_none(),
        "differing ranks give an unknown shape"
    );
}

#[test]
fn sequence_construct_preserves_symbolic_element_dims() {
    let batch = DimExpr::symbol(SymbolId(7));
    let out = run(
        &node("SequenceConstruct", 2, 1),
        vec![
            tin(DataType::Float32, vec![batch.clone(), c(4)]),
            tin(DataType::Float32, vec![batch.clone(), c(4)]),
        ],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.shape.as_deref(), Some([batch, c(4)].as_slice()));
}

#[test]
fn sequence_length_is_int64_scalar() {
    let out = run(
        &node("SequenceLength", 1, 1),
        vec![seq_in(DataType::Float32, Some(vec![c(2), c(3)]))],
        13,
    );
    assert_eq!(out_dtype(&out), DataType::Int64);
    assert_eq!(out_shape(&out), Vec::<DimExpr>::new());
}

#[test]
fn sequence_at_recovers_the_element_tensor_type() {
    let out = run(
        &node("SequenceAt", 2, 1),
        vec![
            seq_in(DataType::Int64, Some(vec![c(6), c(8)])),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    assert_eq!(out_dtype(&out), DataType::Int64);
    assert_eq!(out_shape(&out), vec![c(6), c(8)]);
}

#[test]
fn sequence_at_preserves_symbolic_element_dims() {
    let seq = DimExpr::symbol(SymbolId(3));
    let out = run(
        &node("SequenceAt", 2, 1),
        vec![
            seq_in(DataType::Float32, Some(vec![seq.clone(), c(16)])),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    assert_eq!(out_shape(&out), vec![seq, c(16)]);
}

#[test]
fn sequence_at_on_dtype_only_element_stays_unresolved() {
    // `SequenceEmpty` -> `SequenceAt`: the element shape is unknown, so the
    // tensor output is left unresolved rather than fabricated.
    let out = run(
        &node("SequenceAt", 2, 1),
        vec![
            seq_in(DataType::Float32, None),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    assert!(out[0].type_info.is_none());
}

#[test]
fn sequence_construct_then_at_round_trips_the_element_type() {
    let constructed = run(
        &node("SequenceConstruct", 3, 1),
        vec![
            f32in(vec![c(2), c(4)]),
            f32in(vec![c(2), c(4)]),
            f32in(vec![c(2), c(4)]),
        ],
        13,
    );
    let recovered = run(
        &node("SequenceAt", 2, 1),
        vec![
            constructed.into_iter().next().unwrap(),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    assert_eq!(out_dtype(&recovered), DataType::Float32);
    assert_eq!(out_shape(&recovered), vec![c(2), c(4)]);
}

// --- container types: increment 2 (mutation + tensor⇔sequence conversion) -----

#[test]
fn sequence_insert_unifies_element_with_inserted_tensor() {
    let out = run(
        &node("SequenceInsert", 2, 1),
        vec![
            seq_in(DataType::Float32, Some(vec![c(2), c(3)])),
            f32in(vec![c(2), c(3)]),
        ],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.dtype, DataType::Float32);
    assert_eq!(element.shape.as_deref(), Some([c(2), c(3)].as_slice()));
}

#[test]
fn sequence_insert_disagreeing_extent_degrades_but_keeps_rank() {
    let out = run(
        &node("SequenceInsert", 3, 1),
        vec![
            seq_in(DataType::Float32, Some(vec![c(2), c(3)])),
            f32in(vec![c(2), c(5)]),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    let element = seq_element_tensor(&out);
    let shape = element.shape.as_ref().expect("rank preserved");
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], c(2), "agreeing extent is preserved");
    assert_symbolic(&shape[1]);
}

#[test]
fn sequence_insert_preserves_symbolic_element_dims() {
    let batch = sym(9);
    let out = run(
        &node("SequenceInsert", 2, 1),
        vec![
            seq_in(DataType::Float32, Some(vec![batch.clone(), c(4)])),
            tin(DataType::Float32, vec![batch.clone(), c(4)]),
        ],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.shape.as_deref(), Some([batch, c(4)].as_slice()));
}

#[test]
fn sequence_insert_mismatched_dtype_errors() {
    let error = try_run(
        &node("SequenceInsert", 2, 1),
        vec![
            seq_in(DataType::Float32, Some(vec![c(2)])),
            tin(DataType::Int64, vec![c(2)]),
        ],
        13,
    )
    .expect_err("mismatched inserted dtype must be rejected");
    assert_invalid(error, "SequenceInsert", "share a dtype");
}

#[test]
fn sequence_insert_into_dtype_only_sequence_keeps_dtype_but_unknown_shape() {
    // `SequenceEmpty` -> `SequenceInsert`: the empty sequence has a dtype-only
    // element, so a concrete inserted shape cannot be confirmed as the element
    // shape; the dtype survives, the shape stays unknown.
    let out = run(
        &node("SequenceInsert", 2, 1),
        vec![seq_in(DataType::Float32, None), f32in(vec![c(2), c(3)])],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.dtype, DataType::Float32);
    assert!(
        element.shape.is_none(),
        "unconfirmed element shape stays unknown"
    );
}

#[test]
fn sequence_insert_into_untyped_sequence_adopts_inserted_tensor() {
    // Input 0 carries no resolved container type: the inserted tensor is the
    // only exemplar, so it becomes the element type.
    let out = run(
        &node("SequenceInsert", 2, 1),
        vec![NodeIo::default(), f32in(vec![c(4), c(5)])],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.dtype, DataType::Float32);
    assert_eq!(element.shape.as_deref(), Some([c(4), c(5)].as_slice()));
}

#[test]
fn sequence_construct_insert_at_round_trips_the_element_type() {
    let constructed = run(
        &node("SequenceConstruct", 2, 1),
        vec![f32in(vec![c(2), c(4)]), f32in(vec![c(2), c(4)])],
        13,
    );
    let inserted = run(
        &node("SequenceInsert", 2, 1),
        vec![
            constructed.into_iter().next().unwrap(),
            f32in(vec![c(2), c(4)]),
        ],
        13,
    );
    let recovered = run(
        &node("SequenceAt", 2, 1),
        vec![
            inserted.into_iter().next().unwrap(),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    assert_eq!(out_dtype(&recovered), DataType::Float32);
    assert_eq!(out_shape(&recovered), vec![c(2), c(4)]);
}

#[test]
fn sequence_erase_preserves_element_type() {
    let seq = sym(3);
    let erased = run(
        &node("SequenceErase", 2, 1),
        vec![
            seq_in(DataType::Int64, Some(vec![seq.clone(), c(16)])),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    let element = seq_element_tensor(&erased);
    assert_eq!(element.dtype, DataType::Int64);
    assert_eq!(
        element.shape.as_deref(),
        Some([seq.clone(), c(16)].as_slice())
    );

    // Erase then read back: the element type survives unchanged.
    let recovered = run(
        &node("SequenceAt", 2, 1),
        vec![
            erased.into_iter().next().unwrap(),
            tin(DataType::Int64, vec![]),
        ],
        13,
    );
    assert_eq!(out_dtype(&recovered), DataType::Int64);
    assert_eq!(out_shape(&recovered), vec![seq, c(16)]);
}

#[test]
fn split_to_sequence_default_keeps_split_axis_at_extent_one() {
    for dtype in [DataType::Float32, DataType::Int64] {
        let out = run(
            &with_attr(node("SplitToSequence", 1, 1), "axis", Attribute::Int(1)),
            vec![tin(dtype, vec![c(2), c(6)])],
            13,
        );
        let element = seq_element_tensor(&out);
        assert_eq!(element.dtype, dtype);
        assert_eq!(element.shape.as_deref(), Some([c(2), c(1)].as_slice()));
    }
}

#[test]
fn split_to_sequence_keepdims_zero_removes_split_axis() {
    let out = run(
        &with_attr(
            with_attr(node("SplitToSequence", 1, 1), "axis", Attribute::Int(1)),
            "keepdims",
            Attribute::Int(0),
        ),
        vec![f32in(vec![c(2), c(6)])],
        13,
    );
    let element = seq_element_tensor(&out);
    assert_eq!(element.shape.as_deref(), Some([c(2)].as_slice()));
}

#[test]
fn split_to_sequence_negative_axis_and_symbolic_dims() {
    let batch = sym(5);
    let out = run(
        &with_attr(node("SplitToSequence", 1, 1), "axis", Attribute::Int(-1)),
        vec![tin(DataType::Float32, vec![batch.clone(), c(8)])],
        13,
    );
    // axis -1 == last axis: it collapses to extent 1, the batch dim is kept.
    let element = seq_element_tensor(&out);
    assert_eq!(element.shape.as_deref(), Some([batch, c(1)].as_slice()));
}

#[test]
fn split_to_sequence_explicit_split_makes_axis_symbolic() {
    let out = run(
        &with_attr(node("SplitToSequence", 2, 1), "axis", Attribute::Int(0)),
        vec![f32in(vec![c(6), c(4)]), tin(DataType::Int64, vec![c(3)])],
        13,
    );
    let element = seq_element_tensor(&out);
    let shape = element.shape.as_ref().expect("rank preserved");
    assert_eq!(shape.len(), 2);
    assert_symbolic(&shape[0]);
    assert_eq!(shape[1], c(4), "non-split axis preserved");
}

#[test]
fn split_to_sequence_scalar_input_errors() {
    let error = try_run(&node("SplitToSequence", 1, 1), vec![f32in(vec![])], 13)
        .expect_err("cannot split a rank-0 tensor");
    assert!(matches!(
        error,
        ShapeInferError::InvalidRank { op, index: 0, rank: 0, .. } if op == "SplitToSequence"
    ));
}

#[test]
fn concat_from_sequence_recovers_tensor_with_symbolic_concat_axis() {
    let out = run(
        &with_attr(node("ConcatFromSequence", 1, 1), "axis", Attribute::Int(1)),
        vec![seq_in(DataType::Float32, Some(vec![c(2), c(3)]))],
        13,
    );
    assert_eq!(out_dtype(&out), DataType::Float32);
    let shape = out_shape(&out);
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], c(2), "non-concat axis preserved");
    assert_symbolic(&shape[1]);
}

#[test]
fn concat_from_sequence_new_axis_inserts_a_symbolic_stack_dim() {
    let out = run(
        &with_attr(
            with_attr(node("ConcatFromSequence", 1, 1), "axis", Attribute::Int(1)),
            "new_axis",
            Attribute::Int(1),
        ),
        vec![seq_in(DataType::Int64, Some(vec![c(2), c(3)]))],
        13,
    );
    assert_eq!(out_dtype(&out), DataType::Int64);
    let shape = out_shape(&out);
    assert_eq!(shape.len(), 3, "new_axis raises the rank by one");
    assert_eq!(shape[0], c(2));
    assert_symbolic(&shape[1]);
    assert_eq!(shape[2], c(3));
}

#[test]
fn concat_from_sequence_preserves_symbolic_non_concat_dims() {
    let feat = sym(4);
    let out = run(
        &with_attr(node("ConcatFromSequence", 1, 1), "axis", Attribute::Int(0)),
        vec![seq_in(DataType::Float32, Some(vec![c(3), feat.clone()]))],
        13,
    );
    let shape = out_shape(&out);
    assert_symbolic(&shape[0]);
    assert_eq!(shape[1], feat, "non-concat symbolic dim survives");
}

#[test]
fn concat_from_sequence_missing_axis_errors() {
    let error = try_run(
        &node("ConcatFromSequence", 1, 1),
        vec![seq_in(DataType::Float32, Some(vec![c(2), c(3)]))],
        13,
    )
    .expect_err("axis is mandatory");
    assert_invalid(error, "ConcatFromSequence", "mandatory 'axis'");
}

#[test]
fn concat_from_sequence_dtype_only_element_stays_unresolved() {
    // The element rank is unknown, so no tensor shape can be fabricated.
    let out = run(
        &with_attr(node("ConcatFromSequence", 1, 1), "axis", Attribute::Int(0)),
        vec![seq_in(DataType::Float32, None)],
        13,
    );
    assert!(out[0].type_info.is_none());
}

#[test]
fn split_to_sequence_then_concat_from_sequence_recovers_a_tensor() {
    // The full tensor -> sequence -> tensor seam: split a rank-2 tensor along
    // axis 1, then concatenate the sequence back along axis 1. The dtype and
    // rank are recovered; the concat axis extent is (honestly) symbolic.
    let split = run(
        &with_attr(node("SplitToSequence", 1, 1), "axis", Attribute::Int(1)),
        vec![f32in(vec![c(2), c(6)])],
        13,
    );
    let concat = run(
        &with_attr(node("ConcatFromSequence", 1, 1), "axis", Attribute::Int(1)),
        vec![split.into_iter().next().unwrap()],
        13,
    );
    assert_eq!(out_dtype(&concat), DataType::Float32);
    let shape = out_shape(&concat);
    assert_eq!(shape.len(), 2);
    assert_eq!(shape[0], c(2), "batch dim recovered");
    assert_symbolic(&shape[1]);
}

#[test]
fn tensor_scatter_keeps_the_cache_shape_and_dtype() {
    // The cache is fixed-capacity: the update writes a window into it, so the
    // present cache has exactly the past cache's shape even though `update` is
    // shorter along the sequence axis.
    let n = node("TensorScatter", 3, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float16, vec![c(2), c(8), c(1024), c(128)]),
            tin(DataType::Float16, vec![c(2), c(8), c(1), c(128)]),
            tin(DataType::Int64, vec![c(2)]),
        ],
        24,
    );
    assert_eq!(out_shape(&outs), vec![c(2), c(8), c(1024), c(128)]);
    assert_eq!(out_dtype(&outs), DataType::Float16);
}

#[test]
fn tensor_scatter_accepts_the_two_input_prefill_form() {
    let n = node("TensorScatter", 2, 1);
    let outs = run(
        &n,
        vec![
            tin(DataType::Float32, vec![c(1), c(4), c(64), c(16)]),
            tin(DataType::Float32, vec![c(1), c(4), c(12), c(16)]),
        ],
        24,
    );
    assert_eq!(out_shape(&outs), vec![c(1), c(4), c(64), c(16)]);
}

#[test]
fn tensor_scatter_rejects_an_update_longer_than_the_cache() {
    let n = node("TensorScatter", 2, 1);
    let error = try_run(
        &n,
        vec![
            tin(DataType::Float32, vec![c(1), c(4), c(64), c(16)]),
            tin(DataType::Float32, vec![c(1), c(4), c(65), c(16)]),
        ],
        24,
    )
    .expect_err("an update that cannot fit must not infer");
    assert_invalid(error, "TensorScatter", "exceeds cache capacity");
}

#[test]
fn tensor_scatter_rejects_a_non_sequence_dimension_mismatch() {
    // Only the sequence axis may differ; a head-count mismatch is a real error
    // rather than something to infer through.
    let n = node("TensorScatter", 2, 1);
    let error = try_run(
        &n,
        vec![
            tin(DataType::Float32, vec![c(1), c(4), c(64), c(16)]),
            tin(DataType::Float32, vec![c(1), c(5), c(8), c(16)]),
        ],
        24,
    )
    .expect_err("a mismatched non-sequence dimension must not infer");
    assert_invalid(error, "TensorScatter", "must match past_cache dimension");
}

#[test]
fn tensor_scatter_rejects_an_axis_that_selects_the_batch_dimension() {
    // `write_indices` is indexed by the batch coordinate, so the sequence axis
    // has to sit after it.
    let n = with_attr(node("TensorScatter", 2, 1), "axis", Attribute::Int(0));
    let error = try_run(
        &n,
        vec![
            tin(DataType::Float32, vec![c(2), c(64), c(16)]),
            tin(DataType::Float32, vec![c(2), c(1), c(16)]),
        ],
        24,
    )
    .expect_err("axis 0 must not infer");
    assert_invalid(
        error,
        "TensorScatter",
        "must not select the batch dimension",
    );
}

/// Every rule the shape-inference registry claims, as `(domain, operator,
/// min_opset)`, sorted.
///
/// This is the catalog pin proper; `expanded_registry_catalog_count_is_pinned`
/// pins its two summary numbers. Those numbers are the cheaper and more
/// readable signal, but each is blind to a change that preserves it:
///
/// - a **rename** -- registering `LinearAttenion` instead of `LinearAttention`
///   -- drops one key and adds another, so both counts hold;
/// - an **opset move** -- changing an existing `reg.register(.., 1, ..)` to
///   `13` -- rewrites an entry in place, so `entry_count` holds too.
///
/// Neither is loud anywhere else. `InferenceRegistry::get` returns `None` both
/// for an unknown key and for a version below every registration, and
/// `infer_node` treats `None` permissively: outputs are left unknown and the
/// model still runs, having quietly lost shape inference. Both mutations were
/// constructed against this crate and the whole suite stayed green -- 281 and
/// 282 passed, 0 failed -- before this pin existed.
const PINNED_CATALOG: &[(&str, &str, u64)] = &[
    ("", "Abs", 1),
    ("", "Acos", 7),
    ("", "Acosh", 9),
    ("", "Add", 1),
    ("", "AffineGrid", 20),
    ("", "And", 1),
    ("", "ArgMax", 1),
    ("", "ArgMax", 11),
    ("", "ArgMax", 12),
    ("", "ArgMax", 13),
    ("", "ArgMin", 1),
    ("", "ArgMin", 11),
    ("", "ArgMin", 12),
    ("", "ArgMin", 13),
    ("", "Asin", 7),
    ("", "Asinh", 9),
    ("", "Atan", 7),
    ("", "Atanh", 9),
    ("", "Attention", 23),
    ("", "AveragePool", 1),
    ("", "BatchNormalization", 9),
    ("", "BatchNormalization", 14),
    ("", "BatchNormalization", 15),
    ("", "Bernoulli", 15),
    ("", "BitShift", 11),
    ("", "BitwiseAnd", 18),
    ("", "BitwiseNot", 18),
    ("", "BitwiseOr", 18),
    ("", "BitwiseXor", 18),
    ("", "BlackmanWindow", 17),
    ("", "Cast", 1),
    ("", "CastLike", 1),
    ("", "CausalConvWithState", 27),
    ("", "Ceil", 1),
    ("", "Celu", 12),
    ("", "CenterCropPad", 18),
    ("", "Clip", 1),
    ("", "Col2Im", 18),
    ("", "Compress", 9),
    ("", "Compress", 11),
    ("", "Concat", 1),
    ("", "ConcatFromSequence", 11),
    ("", "Constant", 1),
    ("", "ConstantOfShape", 1),
    ("", "Conv", 1),
    ("", "ConvTranspose", 1),
    ("", "Cos", 1),
    ("", "Cosh", 9),
    ("", "CumSum", 11),
    ("", "CumSum", 14),
    ("", "DFT", 17),
    ("", "DFT", 20),
    ("", "DepthToSpace", 1),
    ("", "DepthToSpace", 11),
    ("", "DepthToSpace", 13),
    ("", "DequantizeLinear", 10),
    ("", "DequantizeLinear", 13),
    ("", "DequantizeLinear", 19),
    ("", "DequantizeLinear", 21),
    ("", "DequantizeLinear", 23),
    ("", "DequantizeLinear", 25),
    ("", "Det", 11),
    ("", "Div", 1),
    ("", "Dropout", 1),
    ("", "DynamicQuantizeLinear", 11),
    ("", "Einsum", 12),
    ("", "Elu", 1),
    ("", "Equal", 1),
    ("", "Erf", 1),
    ("", "Exp", 1),
    ("", "Expand", 8),
    ("", "EyeLike", 9),
    ("", "Flatten", 1),
    ("", "Floor", 1),
    ("", "GRU", 1),
    ("", "GRU", 14),
    ("", "Gather", 1),
    ("", "GatherElements", 1),
    ("", "GatherND", 11),
    ("", "GatherND", 12),
    ("", "GatherND", 13),
    ("", "Gelu", 20),
    ("", "Gemm", 1),
    ("", "GlobalAveragePool", 1),
    ("", "GlobalLpPool", 1),
    ("", "GlobalMaxPool", 1),
    ("", "Greater", 1),
    ("", "GreaterOrEqual", 12),
    ("", "GridSample", 16),
    ("", "GroupNormalization", 18),
    ("", "GroupNormalization", 21),
    ("", "HammingWindow", 17),
    ("", "HannWindow", 17),
    ("", "HardSigmoid", 1),
    ("", "HardSwish", 14),
    ("", "Hardmax", 13),
    ("", "Identity", 1),
    ("", "InstanceNormalization", 6),
    ("", "IsInf", 10),
    ("", "IsNaN", 9),
    ("", "LRN", 1),
    ("", "LSTM", 1),
    ("", "LSTM", 14),
    ("", "LayerNormalization", 1),
    ("", "LeakyRelu", 1),
    ("", "Less", 1),
    ("", "LessOrEqual", 12),
    ("", "LinearAttention", 1),
    ("", "Log", 1),
    ("", "LogSoftmax", 1),
    ("", "LpNormalization", 1),
    ("", "LpPool", 1),
    ("", "MatMul", 1),
    ("", "Max", 1),
    ("", "MaxPool", 1),
    ("", "MaxUnpool", 9),
    ("", "MaxUnpool", 11),
    ("", "Mean", 1),
    ("", "MeanVarianceNormalization", 9),
    ("", "MelWeightMatrix", 17),
    ("", "Min", 1),
    ("", "Mish", 18),
    ("", "Mod", 10),
    ("", "Mul", 1),
    ("", "Multinomial", 7),
    ("", "Neg", 1),
    ("", "NegativeLogLikelihoodLoss", 12),
    ("", "NonMaxSuppression", 10),
    ("", "NonZero", 9),
    ("", "NonZero", 13),
    ("", "Not", 1),
    ("", "OneHot", 9),
    ("", "OneHot", 11),
    ("", "Or", 1),
    ("", "PRelu", 16),
    ("", "Pad", 1),
    ("", "Pow", 1),
    ("", "QLinearMatMul", 10),
    ("", "QuantizeLinear", 10),
    ("", "QuantizeLinear", 13),
    ("", "QuantizeLinear", 19),
    ("", "QuantizeLinear", 21),
    ("", "QuantizeLinear", 23),
    ("", "QuantizeLinear", 25),
    ("", "RMSNormalization", 23),
    ("", "RNN", 1),
    ("", "RNN", 14),
    ("", "RandomNormal", 1),
    ("", "RandomNormalLike", 1),
    ("", "RandomUniform", 1),
    ("", "RandomUniformLike", 1),
    ("", "Range", 11),
    ("", "Reciprocal", 1),
    ("", "ReduceL1", 1),
    ("", "ReduceL2", 1),
    ("", "ReduceLogSum", 1),
    ("", "ReduceLogSumExp", 1),
    ("", "ReduceMax", 1),
    ("", "ReduceMean", 1),
    ("", "ReduceMin", 1),
    ("", "ReduceProd", 1),
    ("", "ReduceSum", 1),
    ("", "ReduceSumSquare", 1),
    ("", "Relu", 1),
    ("", "Reshape", 1),
    ("", "Resize", 10),
    ("", "Resize", 11),
    ("", "ReverseSequence", 10),
    ("", "RotaryEmbedding", 23),
    ("", "Round", 1),
    ("", "STFT", 17),
    ("", "Scatter", 9),
    ("", "ScatterElements", 11),
    ("", "ScatterElements", 13),
    ("", "ScatterElements", 16),
    ("", "ScatterND", 11),
    ("", "ScatterND", 13),
    ("", "ScatterND", 16),
    ("", "ScatterND", 18),
    ("", "Selu", 6),
    ("", "SequenceAt", 11),
    ("", "SequenceConstruct", 11),
    ("", "SequenceEmpty", 11),
    ("", "SequenceErase", 11),
    ("", "SequenceInsert", 11),
    ("", "SequenceLength", 11),
    ("", "Shape", 1),
    ("", "Shrink", 9),
    ("", "Sigmoid", 1),
    ("", "Sign", 1),
    ("", "SimplifiedLayerNormalization", 1),
    ("", "Sin", 1),
    ("", "Sinh", 9),
    ("", "Size", 1),
    ("", "Slice", 1),
    ("", "Softmax", 1),
    ("", "SoftmaxCrossEntropyLoss", 12),
    ("", "Softplus", 1),
    ("", "Softsign", 1),
    ("", "SpaceToDepth", 1),
    ("", "SpaceToDepth", 13),
    ("", "Split", 1),
    ("", "SplitToSequence", 11),
    ("", "Sqrt", 1),
    ("", "Squeeze", 1),
    ("", "Squeeze", 13),
    ("", "StringNormalizer", 10),
    ("", "Sub", 1),
    ("", "Sum", 1),
    ("", "Swish", 24),
    ("", "Tan", 7),
    ("", "Tanh", 1),
    ("", "TensorScatter", 24),
    ("", "TfIdfVectorizer", 9),
    ("", "ThresholdedRelu", 10),
    ("", "Tile", 6),
    ("", "TopK", 1),
    ("", "TopK", 10),
    ("", "TopK", 11),
    ("", "Transpose", 1),
    ("", "Trilu", 14),
    ("", "Unique", 11),
    ("", "Unsqueeze", 1),
    ("", "Unsqueeze", 13),
    ("", "Where", 1),
    ("", "Xor", 1),
    ("ai.onnx.ml", "ArrayFeatureExtractor", 1),
    ("ai.onnx.ml", "Binarizer", 1),
    ("ai.onnx.ml", "CategoryMapper", 1),
    ("ai.onnx.ml", "Imputer", 1),
    ("ai.onnx.ml", "LabelEncoder", 1),
    ("ai.onnx.ml", "LabelEncoder", 2),
    ("ai.onnx.ml", "LabelEncoder", 4),
    ("ai.onnx.ml", "Normalizer", 1),
    ("ai.onnx.ml", "Scaler", 1),
    ("com.microsoft", "Attention", 1),
    ("com.microsoft", "BiasGelu", 1),
    ("com.microsoft", "CausalConvWithState", 1),
    ("com.microsoft", "CompressedSparseAttention", 1),
    ("com.microsoft", "FastGelu", 1),
    ("com.microsoft", "FusedAttention", 1),
    ("com.microsoft", "FusedGemm", 1),
    ("com.microsoft", "FusedMatMul", 1),
    ("com.microsoft", "FusedMatMulBias", 1),
    ("com.microsoft", "GatherBlockQuantized", 1),
    ("com.microsoft", "Gelu", 1),
    ("com.microsoft", "GroupQueryAttention", 1),
    ("com.microsoft", "LayerNormalization", 1),
    ("com.microsoft", "LinearAttention", 1),
    ("com.microsoft", "MatMulNBits", 1),
    ("com.microsoft", "MoE", 1),
    ("com.microsoft", "MultiHeadAttention", 1),
    ("com.microsoft", "QMoE", 1),
    ("com.microsoft", "QuickGelu", 1),
    ("com.microsoft", "RotaryEmbedding", 1),
    ("com.microsoft", "Silu", 1),
    ("com.microsoft", "SimplifiedLayerNormalization", 1),
    ("com.microsoft", "SkipLayerNormalization", 1),
    ("com.microsoft", "SkipSimplifiedLayerNormalization", 1),
    ("pkg.nxrt", "BlockQuantizedMatMul", 1),
    ("pkg.nxrt", "BlockQuantizedMoE", 1),
    ("pkg.nxrt", "CompressedSparseAttention", 1),
    ("pkg.nxrt", "IndexShare", 1),
    ("pkg.nxrt", "KvCacheCapacityAppend", 1),
    ("pkg.nxrt", "SparseKvGather", 1),
    ("pkg.nxrt", "VarlenAttention", 1),
];

/// `""` is the normalized spelling of the default ONNX domain (see
/// `normalize_domain`); render it as `ai.onnx` so a failure names something a
/// reader can search the model for.
fn catalog_label((domain, op, min_opset): &(&str, &str, u64)) -> String {
    let domain = if domain.is_empty() { "ai.onnx" } else { domain };
    format!("{domain}::{op}@{min_opset}")
}

#[test]
fn expanded_registry_catalog_is_pinned() {
    // The literal has to be sorted and duplicate-free before it can be binary
    // searched below; a mis-sorted pin would report spurious adds and removes
    // forever after, which is a worse failure than no pin at all.
    let mut sorted = PINNED_CATALOG.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    if sorted.as_slice() != PINNED_CATALOG {
        let at = sorted
            .iter()
            .zip(PINNED_CATALOG)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| sorted.len().min(PINNED_CATALOG.len()));
        panic!(
            "PINNED_CATALOG must be sorted and duplicate-free; it first diverges at index {at}, \
             where the corrected form has {} and the literal has {}. Regenerate it from \
             `InferenceRegistry::default_registry().operator_versions()` rather than editing by \
             hand.",
            sorted.get(at).map_or("(end)".to_string(), catalog_label),
            PINNED_CATALOG
                .get(at)
                .map_or("(end)".to_string(), catalog_label),
        );
    }

    let registry = InferenceRegistry::default_registry();
    let live = registry.operator_versions();

    let added: Vec<String> = live
        .iter()
        .filter(|r| PINNED_CATALOG.binary_search(r).is_err())
        .map(catalog_label)
        .collect();
    let removed: Vec<String> = PINNED_CATALOG
        .iter()
        .filter(|r| live.binary_search(r).is_err())
        .map(catalog_label)
        .collect();

    let render = |v: &[String]| {
        if v.is_empty() {
            "(none)".to_string()
        } else {
            v.join(", ")
        }
    };
    assert!(
        added.is_empty() && removed.is_empty(),
        "shape-inference operator catalog moved.\n  registered but not pinned: {}\n  pinned \
         but not registered: {}\nEach entry reads `domain::Op@min_opset`. If you added, removed \
         or re-versioned a handler, regenerate PINNED_CATALOG from `operator_versions()` and \
         update the counts in the same commit, and cover the rule with a test (RULES.md section \
         8). `pinned but not registered` is the direction to read first: nothing else in this \
         suite fails when an operator loses its handler or gains a higher opset floor, because \
         unregistered and under-versioned ops are both inferred permissively rather than \
         rejected.",
        render(&added),
        render(&removed),
    );
}
