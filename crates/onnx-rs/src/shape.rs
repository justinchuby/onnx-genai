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
    use crate::ir::{DataType, Dim, Graph, Node, NodeId, Shape};

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
}
