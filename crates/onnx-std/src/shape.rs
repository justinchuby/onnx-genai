//! Symbolic shape inference over the shared runtime IR (ONNX_RS §9).
//!
//! This module is a thin standard-library wrapper around
//! [`onnx_runtime_shape_inference`]. It does not duplicate operator rules:
//! [`infer_shapes`] uses the crate's built-in [`InferenceRegistry`], while
//! [`infer_shapes_with_registry`] accepts a caller-supplied registry containing
//! custom rules.

use crate::Model;

pub use onnx_runtime_shape_inference::{
    DimExpr, InferenceContext, InferenceFn, InferenceRegistry, MergePolicy, NodeIo, ShapeData,
    ShapeInferError as ShapeError, TypeInfo, TypedShape,
};

/// Summary of a whole-model shape-inference run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShapeInferenceResult {
    /// Values whose dtype and known-rank shape were resolved.
    pub inferred: usize,
    /// Values whose type or shape could not be resolved.
    pub unknown: usize,
    /// Non-fatal diagnostics. The underlying engine currently reports
    /// unsupported or data-dependent values through `unknown`, not warnings.
    pub warnings: Vec<String>,
}

/// Infer value types and shapes with the built-in ONNX operator registry.
///
/// Results are written directly into `model.graph`. Inference is permissive:
/// unsupported operators leave their outputs unknown rather than failing.
pub fn infer_shapes(model: &mut Model) -> Result<ShapeInferenceResult, ShapeError> {
    let registry = InferenceRegistry::default_registry();
    infer_shapes_with_registry(model, &registry)
}

/// Infer value types and shapes with a caller-supplied registry.
///
/// Start with [`InferenceRegistry::default_registry`] and use
/// [`register_shape_inference`] to add or replace custom operator rules.
pub fn infer_shapes_with_registry(
    model: &mut Model,
    registry: &InferenceRegistry,
) -> Result<ShapeInferenceResult, ShapeError> {
    let opsets = model.graph.opset_imports.clone();
    let report = registry.infer_graph(&mut model.graph, &opsets, MergePolicy::Permissive)?;
    Ok(ShapeInferenceResult {
        inferred: report.num_resolved(),
        unknown: report.num_unresolved(),
        warnings: Vec::new(),
    })
}

/// Register an opset-aware shape-inference function on `registry`.
///
/// The underlying engine uses function pointers receiving an
/// [`InferenceContext`], rather than a global handler trait. A rule registered at
/// `min_opset` applies until superseded by a registration at a higher opset.
pub fn register_shape_inference(
    registry: &mut InferenceRegistry,
    domain: &str,
    op_type: &str,
    min_opset: u64,
    handler: InferenceFn,
) {
    registry.register(domain, op_type, min_opset, handler);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{
        Attribute, DataType, Dim, Graph, Node, NodeId, Shape, TensorData, ValueId, WeightRef,
    };

    fn i64_initializer(graph: &mut Graph, name: &str, values: &[i64]) -> ValueId {
        let value =
            graph.create_named_value(name, DataType::Int64, vec![Dim::Static(values.len())]);
        let mut bytes = Vec::with_capacity(values.len().saturating_mul(size_of::<i64>()));
        for item in values {
            bytes.extend_from_slice(&item.to_le_bytes());
        }
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                vec![values.len()],
                bytes,
            )),
        );
        value
    }

    fn i64_scalar_initializer(graph: &mut Graph, name: &str, value: i64) -> ValueId {
        let id = graph.create_named_value(name, DataType::Int64, Vec::<Dim>::new());
        graph.set_initializer(
            id,
            WeightRef::Inline(TensorData::from_raw(
                DataType::Int64,
                Vec::new(),
                value.to_le_bytes().to_vec(),
            )),
        );
        id
    }

    #[test]
    fn infers_matmul_and_add_output_shapes() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);

        let x =
            graph.create_named_value("x", DataType::Float32, vec![Dim::Static(2), Dim::Static(3)]);
        let weights = graph.create_named_value(
            "weights",
            DataType::Float32,
            vec![Dim::Static(3), Dim::Static(4)],
        );
        let bias = graph.create_named_value("bias", DataType::Float32, vec![Dim::Static(4)]);
        let product = graph.create_named_value("product", DataType::Float32, Shape::new());
        let output = graph.create_named_value("output", DataType::Float32, Shape::new());

        graph.add_input(x);
        graph.add_input(weights);
        graph.add_input(bias);
        graph.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(x), Some(weights)],
            vec![product],
        ));
        graph.insert_node(Node::new(
            NodeId(1),
            "Add",
            vec![Some(product), Some(bias)],
            vec![output],
        ));
        graph.add_output(output);

        let mut model = Model::new(graph);
        let result = infer_shapes(&mut model).unwrap();

        assert_eq!(
            model.graph.value(product).shape,
            vec![Dim::Static(2), Dim::Static(4)]
        );
        assert_eq!(
            model.graph.value(output).shape,
            vec![Dim::Static(2), Dim::Static(4)]
        );
        assert_eq!(result.inferred, 5);
        assert_eq!(result.unknown, 0);
        assert!(result.warnings.is_empty());
    }

    fn infer_custom_identity(ctx: &mut InferenceContext) -> Result<(), ShapeError> {
        if let Some(input) = ctx.input_type(0).cloned() {
            ctx.set_output_type(0, input);
        }
        Ok(())
    }

    #[test]
    fn custom_rule_uses_underlying_registry() {
        let mut graph = Graph::new();
        graph.opset_imports.insert("example".to_string(), 1);
        let input = graph.create_named_value("input", DataType::Float32, vec![Dim::Static(7)]);
        let output = graph.create_named_value("output", DataType::Float32, Shape::new());
        graph.add_input(input);
        let mut node = Node::new(NodeId(0), "CopyShape", vec![Some(input)], vec![output]);
        node.domain = "example".to_string();
        graph.insert_node(node);
        graph.add_output(output);

        let mut registry = InferenceRegistry::default_registry();
        register_shape_inference(
            &mut registry,
            "example",
            "CopyShape",
            1,
            infer_custom_identity,
        );

        let mut model = Model::new(graph);
        let result = infer_shapes_with_registry(&mut model, &registry).unwrap();

        assert_eq!(model.graph.value(output).shape, vec![Dim::Static(7)]);
        assert_eq!(result.inferred, 2);
        assert_eq!(result.unknown, 0);
    }

    #[test]
    fn infers_added_elementwise_expand_where_and_reduction_schemas() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 18);

        let x = graph.create_named_value(
            "x",
            DataType::Float32,
            vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)],
        );
        let exponent = graph.create_named_value("exponent", DataType::Int64, vec![Dim::Static(1)]);
        let condition = graph.create_named_value(
            "condition",
            DataType::Bool,
            vec![Dim::Static(2), Dim::Static(1), Dim::Static(4)],
        );
        let alternative = graph.create_named_value(
            "alternative",
            DataType::Float32,
            vec![Dim::Static(1), Dim::Static(3), Dim::Static(1)],
        );
        let expand_source = graph.create_named_value(
            "expand_source",
            DataType::Float32,
            vec![Dim::Static(3), Dim::Static(1)],
        );
        for value in [x, exponent, condition, alternative, expand_source] {
            graph.add_input(value);
        }

        let expand_shape = i64_initializer(&mut graph, "expand_shape", &[2, 3, 4]);
        let reduce_axis_1 = i64_initializer(&mut graph, "reduce_axis_1", &[1]);
        let reduce_axis_2 = i64_initializer(&mut graph, "reduce_axis_2", &[2]);

        let mut current = x;
        for (id, op) in ["Sigmoid", "Tanh", "Erf", "Sqrt", "Exp", "Log", "Clip"]
            .into_iter()
            .enumerate()
        {
            let output =
                graph.create_named_value(format!("{op}_out"), DataType::Float32, Shape::new());
            let inputs = if op == "Clip" {
                vec![Some(current), None, None]
            } else {
                vec![Some(current)]
            };
            graph.insert_node(Node::new(NodeId(id as u32), op, inputs, vec![output]));
            current = output;
        }

        let pow = graph.create_named_value("pow", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(7),
            "Pow",
            vec![Some(current), Some(exponent)],
            vec![pow],
        ));
        let selected = graph.create_named_value("selected", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(8),
            "Where",
            vec![Some(condition), Some(pow), Some(alternative)],
            vec![selected],
        ));
        let expanded = graph.create_named_value("expanded", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(9),
            "Expand",
            vec![Some(expand_source), Some(expand_shape)],
            vec![expanded],
        ));
        let sum = graph.create_named_value("sum", DataType::Float32, Shape::new());
        let mut reduce_sum = Node::new(
            NodeId(10),
            "ReduceSum",
            vec![Some(selected), Some(reduce_axis_1)],
            vec![sum],
        );
        reduce_sum
            .attributes
            .insert("keepdims".into(), Attribute::Int(0));
        graph.insert_node(reduce_sum);
        let mean = graph.create_named_value("mean", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(11),
            "ReduceMean",
            vec![Some(selected), Some(reduce_axis_2)],
            vec![mean],
        ));
        graph.add_output(expanded);
        graph.add_output(sum);
        graph.add_output(mean);

        let mut model = Model::new(graph);
        let result = infer_shapes(&mut model).unwrap();
        assert_eq!(
            model.graph.value(selected).shape,
            vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)]
        );
        assert_eq!(
            model.graph.value(expanded).shape,
            vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)]
        );
        assert_eq!(
            model.graph.value(sum).shape,
            vec![Dim::Static(2), Dim::Static(4)]
        );
        assert_eq!(
            model.graph.value(mean).shape,
            vec![Dim::Static(2), Dim::Static(3), Dim::Static(1)]
        );
        assert_eq!(result.unknown, 0);
    }

    #[test]
    fn infers_existing_conv_norm_and_movement_schemas() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);

        let image = graph.create_named_value(
            "image",
            DataType::Float32,
            vec![
                Dim::Static(1),
                Dim::Static(3),
                Dim::Static(32),
                Dim::Static(32),
            ],
        );
        let weights = graph.create_named_value(
            "weights",
            DataType::Float32,
            vec![
                Dim::Static(8),
                Dim::Static(3),
                Dim::Static(3),
                Dim::Static(3),
            ],
        );
        let scale = graph.create_named_value("scale", DataType::Float32, vec![Dim::Static(15_360)]);
        let gather_indices = graph.create_named_value(
            "gather_indices",
            DataType::Int64,
            vec![Dim::Static(5), Dim::Static(6)],
        );
        let concat_rhs = graph.create_named_value(
            "concat_rhs",
            DataType::Float32,
            vec![
                Dim::Static(1),
                Dim::Static(32),
                Dim::Static(7),
                Dim::Static(6),
                Dim::Static(32),
            ],
        );
        for value in [image, weights, scale, gather_indices, concat_rhs] {
            graph.add_input(value);
        }

        let starts = i64_initializer(&mut graph, "starts", &[0]);
        let ends = i64_initializer(&mut graph, "ends", &[10]);
        let axes = i64_initializer(&mut graph, "axes", &[2]);
        let steps = i64_initializer(&mut graph, "steps", &[2]);
        let reshape_target = i64_initializer(&mut graph, "reshape_target", &[2, -1]);

        let convolved = graph.create_named_value("convolved", DataType::Float32, Shape::new());
        let mut conv = Node::new(
            NodeId(0),
            "Conv",
            vec![Some(image), Some(weights)],
            vec![convolved],
        );
        conv.attributes
            .insert("strides".into(), Attribute::Ints(vec![2, 2]));
        conv.attributes
            .insert("pads".into(), Attribute::Ints(vec![1, 1, 1, 1]));
        graph.insert_node(conv);

        let gathered = graph.create_named_value("gathered", DataType::Float32, Shape::new());
        let mut gather = Node::new(
            NodeId(1),
            "Gather",
            vec![Some(image), Some(gather_indices)],
            vec![gathered],
        );
        gather.attributes.insert("axis".into(), Attribute::Int(1));
        graph.insert_node(gather);

        let transposed = graph.create_named_value("transposed", DataType::Float32, Shape::new());
        let mut transpose = Node::new(
            NodeId(2),
            "Transpose",
            vec![Some(gathered)],
            vec![transposed],
        );
        transpose
            .attributes
            .insert("perm".into(), Attribute::Ints(vec![0, 3, 1, 2, 4]));
        graph.insert_node(transpose);

        let concatenated =
            graph.create_named_value("concatenated", DataType::Float32, Shape::new());
        let mut concat = Node::new(
            NodeId(3),
            "Concat",
            vec![Some(transposed), Some(concat_rhs)],
            vec![concatenated],
        );
        concat.attributes.insert("axis".into(), Attribute::Int(2));
        graph.insert_node(concat);

        let sliced = graph.create_named_value("sliced", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(4),
            "Slice",
            vec![
                Some(concatenated),
                Some(starts),
                Some(ends),
                Some(axes),
                Some(steps),
            ],
            vec![sliced],
        ));

        let reshaped = graph.create_named_value("reshaped", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(5),
            "Reshape",
            vec![Some(sliced), Some(reshape_target)],
            vec![reshaped],
        ));
        let normalized = graph.create_named_value("normalized", DataType::Float32, Shape::new());
        let mean = graph.create_named_value("norm_mean", DataType::Float32, Shape::new());
        let inv_std = graph.create_named_value("inv_std", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(6),
            "LayerNormalization",
            vec![Some(reshaped), Some(scale), None],
            vec![normalized, mean, inv_std],
        ));
        let identity = graph.create_named_value("identity", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(7),
            "Identity",
            vec![Some(normalized)],
            vec![identity],
        ));
        graph.add_output(convolved);
        graph.add_output(identity);

        let mut model = Model::new(graph);
        infer_shapes(&mut model).unwrap();
        assert_eq!(
            model.graph.value(convolved).shape,
            vec![
                Dim::Static(1),
                Dim::Static(8),
                Dim::Static(16),
                Dim::Static(16)
            ]
        );
        assert_eq!(
            model.graph.value(transposed).shape,
            vec![
                Dim::Static(1),
                Dim::Static(32),
                Dim::Static(5),
                Dim::Static(6),
                Dim::Static(32)
            ]
        );
        assert_eq!(
            model.graph.value(identity).shape,
            vec![Dim::Static(2), Dim::Static(15_360)]
        );
        assert_eq!(
            model.graph.value(mean).shape,
            vec![Dim::Static(2), Dim::Static(1)]
        );
    }

    #[test]
    fn infers_round_four_normalization_reduction_and_arg_shapes() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 24);
        let x = graph.create_named_value(
            "x",
            DataType::Float32,
            vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)],
        );
        let scale = graph.create_named_value("scale", DataType::Float16, vec![Dim::Static(4)]);
        graph.add_input(x);
        graph.add_input(scale);
        let axes = i64_initializer(&mut graph, "axes", &[1]);

        let log_softmax = graph.create_named_value("log_softmax", DataType::Float32, Shape::new());
        graph.insert_node(Node::new(
            NodeId(0),
            "LogSoftmax",
            vec![Some(x)],
            vec![log_softmax],
        ));

        let rms = graph.create_named_value("rms", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(1),
            "RMSNormalization",
            vec![Some(x), Some(scale)],
            vec![rms],
        ));

        let mut reduction_outputs = Vec::new();
        for (offset, op) in [
            "ReduceMax",
            "ReduceMin",
            "ReduceProd",
            "ReduceL1",
            "ReduceL2",
            "ReduceLogSum",
            "ReduceLogSumExp",
            "ReduceSumSquare",
        ]
        .into_iter()
        .enumerate()
        {
            let output =
                graph.create_named_value(format!("{op}_out"), DataType::Float32, Shape::new());
            let mut node = Node::new(
                NodeId(2 + offset as u32),
                op,
                vec![Some(x), Some(axes)],
                vec![output],
            );
            node.attributes.insert("keepdims".into(), Attribute::Int(0));
            graph.insert_node(node);
            reduction_outputs.push(output);
        }

        let arg_max = graph.create_named_value("arg_max", DataType::Int64, Shape::new());
        let mut arg_max_node = Node::new(NodeId(10), "ArgMax", vec![Some(x)], vec![arg_max]);
        arg_max_node
            .attributes
            .insert("axis".into(), Attribute::Int(-1));
        arg_max_node
            .attributes
            .insert("keepdims".into(), Attribute::Int(0));
        graph.insert_node(arg_max_node);

        let arg_min = graph.create_named_value("arg_min", DataType::Int64, Shape::new());
        let mut arg_min_node = Node::new(NodeId(11), "ArgMin", vec![Some(x)], vec![arg_min]);
        arg_min_node
            .attributes
            .insert("axis".into(), Attribute::Int(1));
        graph.insert_node(arg_min_node);

        for output in [log_softmax, rms, arg_max, arg_min]
            .into_iter()
            .chain(reduction_outputs.iter().copied())
        {
            graph.add_output(output);
        }

        let mut model = Model::new(graph);
        let result = infer_shapes(&mut model).unwrap();
        assert_eq!(
            model.graph.value(log_softmax).shape,
            vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)]
        );
        assert_eq!(
            (
                model.graph.value(rms).dtype,
                model.graph.value(rms).shape.clone()
            ),
            (
                DataType::Float16,
                vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)]
            )
        );
        for output in reduction_outputs {
            assert_eq!(
                model.graph.value(output).shape,
                vec![Dim::Static(2), Dim::Static(4)]
            );
        }
        assert_eq!(
            (
                model.graph.value(arg_max).dtype,
                model.graph.value(arg_max).shape.clone()
            ),
            (DataType::Int64, vec![Dim::Static(2), Dim::Static(3)])
        );
        assert_eq!(
            (
                model.graph.value(arg_min).dtype,
                model.graph.value(arg_min).shape.clone()
            ),
            (
                DataType::Int64,
                vec![Dim::Static(2), Dim::Static(1), Dim::Static(4)]
            )
        );
        assert_eq!(result.unknown, 0);
    }

    #[test]
    fn infers_round_five_index_predicate_and_shape_ops() {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 25);

        let data = graph.create_named_value(
            "data",
            DataType::Float32,
            vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)],
        );
        let gather_elements_indices = graph.create_named_value(
            "gather_elements_indices",
            DataType::Int64,
            vec![Dim::Static(2), Dim::Static(5), Dim::Static(4)],
        );
        let gather_nd_indices = graph.create_named_value(
            "gather_nd_indices",
            DataType::Int64,
            vec![Dim::Static(2), Dim::Static(5), Dim::Static(2)],
        );
        let lhs = graph.create_named_value(
            "lhs",
            DataType::Float32,
            vec![Dim::Static(2), Dim::Static(1), Dim::Static(4)],
        );
        let rhs = graph.create_named_value(
            "rhs",
            DataType::Float32,
            vec![Dim::Static(1), Dim::Static(3), Dim::Static(1)],
        );
        let split_input = graph.create_named_value(
            "split_input",
            DataType::Float32,
            vec![Dim::Static(2), Dim::Static(6), Dim::Static(4)],
        );
        for input in [
            data,
            gather_elements_indices,
            gather_nd_indices,
            lhs,
            rhs,
            split_input,
        ] {
            graph.add_input(input);
        }

        let range_start = i64_scalar_initializer(&mut graph, "range_start", 0);
        let range_limit = i64_scalar_initializer(&mut graph, "range_limit", 10);
        let range_delta = i64_scalar_initializer(&mut graph, "range_delta", 2);
        let split_sizes = i64_initializer(&mut graph, "split_sizes", &[2, 4]);

        let gathered_elements =
            graph.create_named_value("gathered_elements", DataType::Undefined, Shape::new());
        let mut gather_elements = Node::new(
            NodeId(0),
            "GatherElements",
            vec![Some(data), Some(gather_elements_indices)],
            vec![gathered_elements],
        );
        gather_elements
            .attributes
            .insert("axis".into(), Attribute::Int(-2));
        graph.insert_node(gather_elements);

        let gathered_nd =
            graph.create_named_value("gathered_nd", DataType::Undefined, Shape::new());
        let mut gather_nd = Node::new(
            NodeId(1),
            "GatherND",
            vec![Some(data), Some(gather_nd_indices)],
            vec![gathered_nd],
        );
        gather_nd
            .attributes
            .insert("batch_dims".into(), Attribute::Int(1));
        graph.insert_node(gather_nd);

        let equal = graph.create_named_value("equal", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(2),
            "Equal",
            vec![Some(lhs), Some(rhs)],
            vec![equal],
        ));
        let greater = graph.create_named_value("greater", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(3),
            "Greater",
            vec![Some(lhs), Some(rhs)],
            vec![greater],
        ));
        let less = graph.create_named_value("less", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(4),
            "Less",
            vec![Some(lhs), Some(rhs)],
            vec![less],
        ));
        let and = graph.create_named_value("and", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(5),
            "And",
            vec![Some(equal), Some(greater)],
            vec![and],
        ));
        let or = graph.create_named_value("or", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(6),
            "Or",
            vec![Some(and), Some(less)],
            vec![or],
        ));
        let not = graph.create_named_value("not", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(NodeId(7), "Not", vec![Some(or)], vec![not]));

        let cast = graph.create_named_value("cast", DataType::Undefined, Shape::new());
        let mut cast_node = Node::new(NodeId(8), "Cast", vec![Some(data)], vec![cast]);
        cast_node.attributes.insert(
            "to".into(),
            Attribute::Int(i64::from(DataType::Int64.to_onnx())),
        );
        graph.insert_node(cast_node);

        let shape = graph.create_named_value("shape", DataType::Undefined, Shape::new());
        let mut shape_node = Node::new(NodeId(9), "Shape", vec![Some(data)], vec![shape]);
        shape_node
            .attributes
            .insert("start".into(), Attribute::Int(1));
        shape_node
            .attributes
            .insert("end".into(), Attribute::Int(i64::MAX));
        graph.insert_node(shape_node);

        let size = graph.create_named_value("size", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(NodeId(10), "Size", vec![Some(data)], vec![size]));
        let non_zero = graph.create_named_value("non_zero", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(11),
            "NonZero",
            vec![Some(data)],
            vec![non_zero],
        ));
        let range = graph.create_named_value("range", DataType::Undefined, Shape::new());
        graph.insert_node(Node::new(
            NodeId(12),
            "Range",
            vec![Some(range_start), Some(range_limit), Some(range_delta)],
            vec![range],
        ));

        let split_first =
            graph.create_named_value("split_first", DataType::Undefined, Shape::new());
        let split_second =
            graph.create_named_value("split_second", DataType::Undefined, Shape::new());
        let mut split = Node::new(
            NodeId(13),
            "Split",
            vec![Some(split_input), Some(split_sizes)],
            vec![split_first, split_second],
        );
        split.attributes.insert("axis".into(), Attribute::Int(-2));
        graph.insert_node(split);

        for output in [
            gathered_elements,
            gathered_nd,
            equal,
            greater,
            less,
            and,
            or,
            not,
            cast,
            shape,
            size,
            non_zero,
            range,
            split_first,
            split_second,
        ] {
            graph.add_output(output);
        }

        let mut model = Model::new(graph);
        let result = infer_shapes(&mut model).unwrap();

        assert_eq!(
            (
                model.graph.value(gathered_elements).dtype,
                model.graph.value(gathered_elements).shape.clone()
            ),
            (
                DataType::Float32,
                vec![Dim::Static(2), Dim::Static(5), Dim::Static(4)]
            )
        );
        assert_eq!(
            model.graph.value(gathered_nd).shape,
            vec![Dim::Static(2), Dim::Static(5)]
        );
        for output in [equal, greater, less, and, or, not] {
            assert_eq!(
                (
                    model.graph.value(output).dtype,
                    model.graph.value(output).shape.clone()
                ),
                (
                    DataType::Bool,
                    vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)]
                )
            );
        }
        assert_eq!(
            (
                model.graph.value(cast).dtype,
                model.graph.value(cast).shape.clone()
            ),
            (
                DataType::Int64,
                vec![Dim::Static(2), Dim::Static(3), Dim::Static(4)]
            )
        );
        assert_eq!(
            (
                model.graph.value(shape).dtype,
                model.graph.value(shape).shape.clone()
            ),
            (DataType::Int64, vec![Dim::Static(2)])
        );
        assert_eq!(
            (
                model.graph.value(size).dtype,
                model.graph.value(size).shape.clone()
            ),
            (DataType::Int64, Vec::<Dim>::new())
        );
        assert!(matches!(
            model.graph.value(non_zero).shape.as_slice(),
            [Dim::Static(3), Dim::Symbolic(_)]
        ));
        assert_eq!(model.graph.value(range).shape, vec![Dim::Static(5)]);
        assert_eq!(
            model.graph.value(split_first).shape,
            vec![Dim::Static(2), Dim::Static(2), Dim::Static(4)]
        );
        assert_eq!(
            model.graph.value(split_second).shape,
            vec![Dim::Static(2), Dim::Static(4), Dim::Static(4)]
        );
        assert_eq!(result.unknown, 0);
    }

    #[test]
    fn round_five_shape_arithmetic_rejects_unrepresentable_values() {
        let mut range_graph = Graph::new();
        range_graph.opset_imports.insert(String::new(), 25);
        let start = i64_scalar_initializer(&mut range_graph, "start", i64::MIN);
        let limit = i64_scalar_initializer(&mut range_graph, "limit", i64::MAX);
        let delta = i64_scalar_initializer(&mut range_graph, "delta", 1);
        let output = range_graph.create_named_value("range", DataType::Undefined, Shape::new());
        range_graph.insert_node(Node::new(
            NodeId(0),
            "Range",
            vec![Some(start), Some(limit), Some(delta)],
            vec![output],
        ));
        range_graph.add_output(output);
        let error = infer_shapes(&mut Model::new(range_graph)).unwrap_err();
        assert!(error.to_string().contains("exceeds isize::MAX"));

        let mut split_graph = Graph::new();
        split_graph.opset_imports.insert(String::new(), 25);
        let input = split_graph.create_named_value(
            "input",
            DataType::Float32,
            vec![Dim::Static(2), Dim::Static(4)],
        );
        let first = split_graph.create_named_value("first", DataType::Undefined, Shape::new());
        let second = split_graph.create_named_value("second", DataType::Undefined, Shape::new());
        split_graph.add_input(input);
        let mut split = Node::new(NodeId(0), "Split", vec![Some(input)], vec![first, second]);
        split
            .attributes
            .insert("axis".into(), Attribute::Int(i64::MIN));
        split_graph.insert_node(split);
        split_graph.add_output(first);
        split_graph.add_output(second);
        let error = infer_shapes(&mut Model::new(split_graph)).unwrap_err();
        assert!(error.to_string().contains("axis"));
    }

    #[test]
    fn if_inference_unions_branch_output_shapes() {
        fn branch(shape: Vec<Dim>) -> Graph {
            let mut graph = Graph::new();
            let output = graph.create_named_value("branch_output", DataType::Float32, shape);
            graph.add_input(output);
            graph.add_output(output);
            graph
        }

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let condition = graph.create_named_value("condition", DataType::Bool, Vec::<Dim>::new());
        graph.add_input(condition);
        let output = graph.create_named_value("output", DataType::Float32, Shape::new());
        let node = graph.insert_node(Node::new(
            NodeId(0),
            "If",
            vec![Some(condition)],
            vec![output],
        ));
        graph.subgraphs.insert(
            (node, "then_branch".into()),
            branch(vec![Dim::Static(2), Dim::Static(4)]),
        );
        graph.subgraphs.insert(
            (node, "else_branch".into()),
            branch(vec![Dim::Static(3), Dim::Static(4)]),
        );
        graph.add_output(output);

        let mut model = Model::new(graph);
        let result = infer_shapes(&mut model).unwrap();
        assert!(matches!(
            model.graph.value(output).shape.as_slice(),
            [Dim::Symbolic(_), Dim::Static(4)]
        ));
        assert_eq!(result.unknown, 0);
    }

    #[test]
    fn if_inference_does_not_invent_unresolved_branch_shapes() {
        fn unresolved_branch() -> Graph {
            let mut graph = Graph::new();
            let input = graph.create_named_value("input", DataType::Float32, vec![Dim::Static(2)]);
            graph.add_input(input);
            let output = graph.create_named_value("output", DataType::Float32, Shape::new());
            graph.mark_value_shape_unknown(output);
            graph.insert_node(Node::new(
                NodeId(0),
                "Unsupported",
                vec![Some(input)],
                vec![output],
            ));
            graph.add_output(output);
            graph
        }

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let condition = graph.create_named_value("condition", DataType::Bool, Vec::<Dim>::new());
        graph.add_input(condition);
        let output = graph.create_named_value("output", DataType::Float32, Shape::new());
        graph.mark_value_shape_unknown(output);
        let node = graph.insert_node(Node::new(
            NodeId(0),
            "If",
            vec![Some(condition)],
            vec![output],
        ));
        graph
            .subgraphs
            .insert((node, "then_branch".into()), unresolved_branch());
        graph
            .subgraphs
            .insert((node, "else_branch".into()), unresolved_branch());
        graph.add_output(output);

        let mut model = Model::new(graph);
        let result = infer_shapes(&mut model).unwrap();
        assert_eq!(result.unknown, 1);
    }

    #[test]
    fn if_inference_preserves_known_scalar_branch_outputs() {
        fn scalar_branch() -> Graph {
            let mut graph = Graph::new();
            let input = graph.create_named_value("input", DataType::Float32, vec![Dim::Static(2)]);
            graph.add_input(input);
            let output = graph.create_named_value("output", DataType::Float32, Shape::new());
            graph.insert_node(Node::new(
                NodeId(0),
                "Unsupported",
                vec![Some(input)],
                vec![output],
            ));
            graph.add_output(output);
            graph
        }

        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let condition = graph.create_named_value("condition", DataType::Bool, Shape::new());
        graph.add_input(condition);
        let output = graph.create_named_value("output", DataType::Float32, Shape::new());
        graph.mark_value_shape_unknown(output);
        let node = graph.insert_node(Node::new(
            NodeId(0),
            "If",
            vec![Some(condition)],
            vec![output],
        ));
        graph
            .subgraphs
            .insert((node, "then_branch".into()), scalar_branch());
        graph
            .subgraphs
            .insert((node, "else_branch".into()), scalar_branch());
        graph.add_output(output);

        let mut model = Model::new(graph);
        let result = infer_shapes(&mut model).unwrap();
        assert!(model.graph.value(output).shape.is_empty());
        assert_eq!(result.unknown, 0);
    }
}
