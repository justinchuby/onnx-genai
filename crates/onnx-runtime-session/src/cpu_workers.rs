//! A loaded-model prerequisite, not a complete governed Engine worker factory.

use std::{path::Path, sync::Arc};

use onnx_runtime_ep_cpu::{CpuExecutionProvider, CpuMemoryResources};
use onnx_runtime_ir::Graph;
use onnx_runtime_loader::{EpContextDumpConfig, ModelMetadata, WeightStore};

use crate::{ExecutionProviderOrder, InferenceSession, SessionError, error::Result};

/// Immutable external-mapped CPU model descriptors and their existing backing.
///
/// Each worker has a fresh provider, executor identity, scratch, kernel caches,
/// bindings, outputs and trace state. Only the already loaded external weight
/// store is shared. No model files or external weights are reopened or reparsed.
/// Inline payloads, Constant kernels, nested graphs/functions, sparse data,
/// EPContext and existing decode-inline/verify siblings are explicitly refused.
///
/// `CpuMemoryResources` governs workspace and derived/packed weight caches. It
/// does NOT newly meter ordinary provider allocations, IR or session buffers.
/// Before enabling Engine/server workers, the caller must separately admit
/// per-worker KV and general buffer residency and retain the Engine's existing
/// fixed-model-weight lease for the entire lifetime of this plan and its workers.
/// An `Arc<WeightStore>` preserves backing lifetime, not its accounting lease.
#[derive(Clone)]
pub struct CpuWorkerPlan {
    graph: Arc<Graph>,
    weights: Arc<WeightStore>,
    model_metadata: ModelMetadata,
}

impl InferenceSession {
    /// Freeze the supported loaded CPU ownership profile without moving or
    /// exposing any mutable state of this executor to a second worker.
    pub fn cpu_worker_plan(&self) -> Result<CpuWorkerPlan> {
        if self.decode_inline_exec.is_some() || self.verify_exec.is_some() {
            return Err(SessionError::UnsupportedCpuWorkerPlan {
                reason: "decode-inline or speculative verify siblings have separate mutable execution state; disable those modes for this initial profile".into(),
            });
        }
        let (graph, weights) = self.exec.cpu_worker_parts()?;
        Ok(CpuWorkerPlan {
            graph: Arc::new(graph),
            weights,
            model_metadata: self.model_metadata.clone(),
        })
    }
}

impl CpuWorkerPlan {
    /// Build on the owning worker thread with explicit model-authority resources.
    ///
    /// No memory budget is copied or created here. Engine integrations must
    /// supply resources over the same model-wide authorities, retain the fixed
    /// weight accounting owner, and admit their ordinary buffers/KV separately.
    /// A governed resource set without a workspace pool is not rejected just
    /// because it is incomplete for some kernels: the CPU provider validates
    /// the pool at the first actual workspace demand and fails closed there.
    /// Zero-workspace graphs can therefore use governed derived-weight resources
    /// without constructing an otherwise unused workspace pool.
    pub fn build_worker(&self, resources: CpuMemoryResources) -> Result<InferenceSession> {
        if !resources.is_governed() {
            return Err(SessionError::UnsupportedCpuWorkerPlan {
                reason: "worker workspace/derived-weight resources are ungoverned; supply explicit resources from the model's shared Host authority and workspace pool".into(),
            });
        }
        let provider = Arc::new(CpuExecutionProvider::initialized_with_resources(resources)?);
        let worker = InferenceSession::from_parts(
            self.graph.as_ref().clone(),
            Arc::clone(&self.weights),
            Path::new("."),
            EpContextDumpConfig::default(),
            self.model_metadata.clone(),
            ExecutionProviderOrder::new(provider),
            None,
        )?;
        // Fresh placement must not silently introduce an inline constant or
        // initializer upload. Failure drops the newly constructed executor and
        // its caches; the source executor and the immutable plan are untouched.
        worker.exec.validate_cpu_worker_ownership()?;
        Ok(worker)
    }
}

#[cfg(test)]
mod tests;
