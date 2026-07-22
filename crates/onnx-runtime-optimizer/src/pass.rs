//! Pass infrastructure: the [`OptimizationPass`] trait, a minimal
//! [`PassContext`], and the [`run_passes`] pipeline runner (see
//! `docs/ORT2.md` §18.1).

use std::sync::Arc;

use onnx_runtime_ir::{Graph, WeightRef};

use crate::error::{OptimizerError, Result};

/// Resolves graph initializer descriptors to their backing bytes.
pub trait InitializerResolver: Send + Sync {
    /// Resolve an initializer descriptor to its raw little-endian bytes.
    fn bytes<'a>(&'a self, weight: &'a WeightRef) -> Option<&'a [u8]>;
}

/// Shared, read-only context threaded through every pass.
///
/// **Phase-1 minimalism.** The design in `docs/ORT2.md` §18.1 gives this struct
/// `cost_model`, `ep_registry`, and `target_devices` fields. Those depend on
/// crates or analyses that do not exist yet, and none of the device-independent
/// Phase-1 passes
/// ([`DeadNodeElimination`](crate::DeadNodeElimination),
/// [`ConstantFolding`](crate::ConstantFolding), [`OpFusion`](crate::OpFusion))
/// need them. The only current service is an optional initializer resolver for
/// EP-scoped passes that physically rewrite immutable weights.
///
/// It is `#[non_exhaustive]` so the Phase-2b cost-model / EP-registry /
/// placement fields can be added without breaking downstream construction.
#[derive(Clone, Default)]
#[non_exhaustive]
pub struct PassContext {
    initializer_resolver: Option<Arc<dyn InitializerResolver>>,
    // Phase 2b (deferred): pub cost_model: Arc<CostModel>,
    //                      pub ep_registry: Arc<EpRegistry>,
    //                      pub target_devices: Vec<DeviceId>,
}

impl std::fmt::Debug for PassContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PassContext")
            .field(
                "initializer_resolver",
                &self.initializer_resolver.as_ref().map(|_| "<resolver>"),
            )
            .finish()
    }
}

impl PassContext {
    /// A context with no device/cost information (the Phase-1 default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Attach the resolver that exposes inline or externally mapped initializer
    /// bytes to passes that rewrite immutable weights.
    pub fn with_initializer_resolver(mut self, resolver: Arc<dyn InitializerResolver>) -> Self {
        self.initializer_resolver = Some(resolver);
        self
    }

    /// Resolve an initializer's raw bytes. Inline initializers are always
    /// available; external references require an attached resolver.
    pub fn initializer_bytes<'b>(&'b self, weight: &'b WeightRef) -> Option<&'b [u8]> {
        match weight {
            WeightRef::Inline(tensor) => Some(&tensor.data),
            WeightRef::External { .. } => self.initializer_resolver.as_deref()?.bytes(weight),
        }
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
