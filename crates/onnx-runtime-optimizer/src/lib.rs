//! # `onnx-runtime-optimizer`
//!
//! Device-independent graph→graph optimization passes for the ORT 2.0 runtime
//! (see `docs/architecture/ORT2.md` §18 "Optimization Passes"). This is the first Phase-2
//! crate: pure, safe Rust graph rewriting over [`onnx_runtime_ir`] — **no**
//! CUDA, no ORT C library, no FFI.
//!
//! ## What lives here
//!
//! | Concept | Type |
//! |---------|------|
//! | Pass contract | [`OptimizationPass`], [`PassContext`], [`run_passes`] |
//! | Dead-code removal | [`DeadNodeElimination`] |
//! | Bounded constant folding | [`ConstantFolding`] |
//! | Reusable provider-scoped fusion machinery | [`OpFusion`], [`FusionPattern`], [`PatternMatch`] |
//! | Errors | [`OptimizerError`], [`Result`] |
//!
//! ## Pipeline
//!
//! [`default_passes`] returns only the device-independent passes implemented
//! here, in pipeline order: `ConstantFolding → DeadNodeElimination`.
//! [`OpFusion`] remains in this crate as reusable machinery, but providers must
//! schedule it themselves so private fused ops are introduced only when the
//! selected provider can run them.
//!
//! ### Deferred (Phase 2b / Phase 3)
//!
//! The full pipeline in `docs/architecture/ORT2.md` §18.1 also lists passes that depend on
//! crates or analyses not yet built. They are intentionally **not** implemented
//! here and are listed in [`default_passes`]'s source in their eventual
//! pipeline position: `ShapeInference` (the loader owns inference for now),
//! `AttentionFusionPass`, `LayoutPropagation`, `PlacementOptimizer`,
//! `TransferInsertion`, `InPlaceDetection`, `MemoryPlanning`,
//! `CudaGraphRegionDetection`, and `OverlapScheduling`.

#![forbid(unsafe_code)]

mod constant_folding;
mod dead_node;
mod error;
mod fusion;
mod pass;

pub use constant_folding::ConstantFolding;
pub use dead_node::DeadNodeElimination;
pub use error::{OptimizerError, Result};
pub use fusion::{CONTRIB_DOMAIN, FusionPattern, OpFusion, PatternMatch, default_fusion_patterns};
pub use pass::{InitializerResolver, OptimizationPass, PassContext, run_passes};

/// The device-independent Phase-1 pass pipeline, in run order.
///
/// ```text
/// ConstantFolding  →  DeadNodeElimination
/// ```
///
/// Constant folding runs first so it can materialize shape-computation
/// constants, then dead-node elimination prunes any node left unreachable, and
/// then dead-node elimination prunes any node left unreachable. Operator fusion
/// is provider-scoped because its replacements change the operator set.
///
/// **Deferred passes** (each in its eventual pipeline slot; see the crate-level
/// docs for why): after `ConstantFolding` would come `ShapeInference`; after
/// provider-scoped fusion would come `AttentionFusionPass`, then `LayoutPropagation`,
/// `PlacementOptimizer`, `TransferInsertion`, `InPlaceDetection`,
/// `MemoryPlanning`, `CudaGraphRegionDetection`, and `OverlapScheduling`.
pub fn default_passes() -> Vec<Box<dyn OptimizationPass>> {
    vec![
        Box::new(ConstantFolding),
        // ShapeInference — deferred (Phase 2b): the loader owns inference.
        Box::new(DeadNodeElimination),
        // AttentionFusionPass, LayoutPropagation, PlacementOptimizer,
        // TransferInsertion, InPlaceDetection, MemoryPlanning,
        // CudaGraphRegionDetection, OverlapScheduling — deferred (Phase 2b/3).
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{DataType, Graph, Node, NodeId, static_shape};

    #[test]
    fn default_passes_lists_two() {
        let passes = default_passes();
        assert_eq!(passes.len(), 2);
        assert_eq!(passes[0].name(), "ConstantFolding");
        assert_eq!(passes[1].name(), "DeadNodeElimination");
    }

    #[test]
    fn run_passes_pipeline_on_matmul_add_with_dead_branch() {
        // MatMul+Add feeding an output, plus a dead Neg branch off `a`.
        let mut g = Graph::new();
        g.opset_imports.insert(String::new(), 17);
        let mk =
            |g: &mut Graph, n: &str| g.create_named_value(n, DataType::Float32, static_shape([4]));
        let a = mk(&mut g, "a");
        let w = mk(&mut g, "w");
        let bias = mk(&mut g, "bias");
        g.add_input(a);
        g.add_input(w);
        g.add_input(bias);
        let m = mk(&mut g, "m");
        g.insert_node(Node::new(
            NodeId(0),
            "MatMul",
            vec![Some(a), Some(w)],
            vec![m],
        ));
        let out = mk(&mut g, "out");
        g.insert_node(Node::new(
            NodeId(0),
            "Add",
            vec![Some(m), Some(bias)],
            vec![out],
        ));
        g.add_output(out);
        // Dead branch.
        let dead = mk(&mut g, "dead");
        g.insert_node(Node::new(NodeId(0), "Neg", vec![Some(a)], vec![dead]));

        run_passes(&mut g, &default_passes(), &PassContext::new()).unwrap();

        // Dead Neg removed by DCE; provider-scoped fusion is not part of the
        // runtime default pipeline, so the standard MatMul/Add pair remains.
        assert_eq!(g.num_nodes(), 2);
        assert_eq!(
            g.nodes
                .values()
                .filter(|node| matches!(node.op_type.as_str(), "MatMul" | "Add"))
                .count(),
            2
        );
        assert!(g.validate().is_ok());
    }

    #[test]
    fn run_passes_is_ok_on_empty_graph() {
        let mut g = Graph::new();
        run_passes(&mut g, &default_passes(), &PassContext::new()).unwrap();
        assert_eq!(g.num_nodes(), 0);
    }
}
