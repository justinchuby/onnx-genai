//! Pass infrastructure: the [`OptimizationPass`] trait, a minimal
//! [`PassContext`], and the [`run_passes`] pipeline runner (see
//! `docs/ORT2.md` §18.1).

use onnx_runtime_ir::Graph;

use crate::error::{OptimizerError, Result};

/// Shared, read-only context threaded through every pass.
///
/// **Phase-1 minimalism.** The design in `docs/ORT2.md` §18.1 gives this struct
/// `cost_model`, `ep_registry`, and `target_devices` fields. Those depend on
/// the `onnx-runtime-cost-model` and EP-registry crates, which do not exist
/// yet, and none of the device-independent Phase-1 passes
/// ([`DeadNodeElimination`](crate::DeadNodeElimination),
/// [`ConstantFolding`](crate::ConstantFolding), [`OpFusion`](crate::OpFusion))
/// need them. The context is intentionally empty for now.
///
/// It is `#[non_exhaustive]` so the Phase-2b cost-model / EP-registry /
/// placement fields can be added without breaking downstream construction.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct PassContext {
    // Phase 2b (deferred): pub cost_model: Arc<CostModel>,
    //                      pub ep_registry: Arc<EpRegistry>,
    //                      pub target_devices: Vec<DeviceId>,
}

impl PassContext {
    /// A context with no device/cost information (the Phase-1 default).
    pub fn new() -> Self {
        Self::default()
    }
}

/// A single graph→graph rewrite (see `docs/ORT2.md` §18.1).
///
/// Passes mutate the [`Graph`] in place and must preserve its structural
/// invariants; [`postconditions`](OptimizationPass::postconditions) is checked
/// after each pass in debug builds by [`run_passes`].
pub trait OptimizationPass: Send + Sync {
    /// A short, stable name for logging and error messages.
    fn name(&self) -> &str;

    /// Apply the rewrite in place.
    fn run(&self, graph: &mut Graph, ctx: &PassContext) -> Result<()>;

    /// Invariants that must hold after this pass. The default requires the
    /// graph to pass full structural validation (`Graph::validate`).
    fn postconditions(&self, graph: &Graph) -> Result<()> {
        graph
            .validate()
            .map_err(|errors| OptimizerError::PostconditionFailed {
                pass: self.name().to_string(),
                errors,
            })
    }
}

/// Run `passes` over `graph` in order.
///
/// Each pass runs to completion, then — in debug builds only — its
/// [`postconditions`](OptimizationPass::postconditions) are checked. In release
/// builds the postcondition check is compiled out for speed, matching the
/// "checked in debug builds" contract of `docs/ORT2.md` §18.1.
pub fn run_passes(
    graph: &mut Graph,
    passes: &[Box<dyn OptimizationPass>],
    ctx: &PassContext,
) -> Result<()> {
    for pass in passes {
        pass.run(graph, ctx)?;
        #[cfg(debug_assertions)]
        pass.postconditions(graph)?;
    }
    Ok(())
}
