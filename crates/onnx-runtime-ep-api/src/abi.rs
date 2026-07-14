//! ORT graph ABI bridge for legacy plugin EPs (§3.4, §4.5).
//!
//! Projects our IR through ORT's C graph API so third-party EPs compiled
//! against upstream ORT can inspect and claim subgraphs. This is a **Phase 2**
//! deliverable requiring `unsafe` FFI; only the data model is sketched here.

use onnx_runtime_ir::{Graph, NodeId, ValueId};

use crate::provider::{EpId, ExecutionProvider};

/// A read-only projection of a [`Graph`] exposed through the ORT C graph API.
pub struct OrtGraphView<'a> {
    #[allow(dead_code)]
    graph: &'a Graph,
}

/// An EP's claim over a subgraph it wants to compile and run.
#[derive(Clone, Debug)]
pub struct SubgraphClaim {
    pub ep_id: EpId,
    pub node_ids: Vec<NodeId>,
    pub input_values: Vec<ValueId>,
    pub output_values: Vec<ValueId>,
    pub meta_def: Option<String>,
}

impl<'a> OrtGraphView<'a> {
    /// Wrap a graph for ABI projection.
    pub fn new(graph: &'a Graph) -> Self {
        Self { graph }
    }

    /// Ask an EP which subgraphs it can handle.
    pub fn query_capabilities(&self, ep: &dyn ExecutionProvider) -> Vec<SubgraphClaim> {
        let _ = ep;
        todo!("ort2-ep-api Phase 2: project IR via ORT C graph API and gather EP claims")
    }
}
