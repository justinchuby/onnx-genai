//! Op → kernel-factory registry and the EP registry (§4.3, §4.6).

use std::collections::HashMap;
use std::path::Path;

use onnx_runtime_ir::{Node, Shape, TensorLayout};

use crate::error::Result;
use crate::kernel::{Kernel, KernelMatch};
use crate::provider::{EpConfig, EpId, ExecutionProvider};

/// Registry key: an operator identity plus the opset version it was introduced.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct OpKey {
    pub op_type: String,
    pub domain: String,
    pub since_version: u64,
}

impl OpKey {
    pub fn new(op_type: impl Into<String>, domain: impl Into<String>, since_version: u64) -> Self {
        Self {
            op_type: op_type.into(),
            domain: domain.into(),
            since_version,
        }
    }
}

/// Normalise the default ONNX domain: the empty string and `"ai.onnx"` name the
/// same (standard) domain. Contrib domains (e.g. `"com.microsoft"`) are left
/// untouched. Keeps dispatch keyed on `(op_type, domain)` model-agnostically.
fn norm_domain(domain: &str) -> &str {
    if domain == "ai.onnx" { "" } else { domain }
}

/// Creates kernels for a specific op.
pub trait KernelFactory: Send + Sync {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>>;
}

/// Maps `(op_type, domain, opset)` → kernel factory (§4.3).
#[derive(Default)]
pub struct OpRegistry {
    entries: HashMap<OpKey, Box<dyn KernelFactory>>,
}

impl OpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under `key`.
    pub fn register(&mut self, key: OpKey, factory: Box<dyn KernelFactory>) {
        self.entries.insert(key, factory);
    }

    /// Look up the best matching factory: the highest `since_version` that is
    /// `<= opset` for the given `(op_type, domain)`.
    pub fn lookup(&self, op_type: &str, domain: &str, opset: u64) -> Option<&dyn KernelFactory> {
        let domain = norm_domain(domain);
        self.entries
            .iter()
            .filter(|(k, _)| {
                k.op_type == op_type && k.domain == domain && k.since_version <= opset
            })
            .max_by_key(|(k, _)| k.since_version)
            .map(|(_, f)| f.as_ref())
    }

    /// Whether any factory is registered for `(op_type, domain)`, ignoring the
    /// opset version. Used by providers to answer "is this an op/domain we
    /// support?" without a concrete opset in hand (`supports_op`). Keeps the
    /// support decision keyed on `(op_type, domain)` via the registry rather
    /// than a hardcoded op/domain whitelist.
    pub fn supports(&self, op_type: &str, domain: &str) -> bool {
        let domain = norm_domain(domain);
        self.entries
            .keys()
            .any(|k| k.op_type == op_type && k.domain == domain)
    }

    /// Number of registered entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Ordered set of execution providers with a priority list (§4.6).
#[derive(Default)]
pub struct EpRegistry {
    eps: Vec<Box<dyn ExecutionProvider>>,
    /// Priority order as indices into `eps` (front = highest priority).
    priority: Vec<EpId>,
}

impl EpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an EP, returning its [`EpId`]. Appended to the priority list.
    pub fn register(&mut self, ep: Box<dyn ExecutionProvider>) -> EpId {
        let id = EpId(self.eps.len() as u32);
        self.eps.push(ep);
        self.priority.push(id);
        id
    }

    /// Load a legacy ORT plugin EP from a shared library (Phase 2).
    pub fn load_legacy(&mut self, path: &Path, config: &EpConfig) -> Result<EpId> {
        let _ = (path, config);
        todo!("ort2-ep-api Phase 2: dlopen legacy ORT plugin EP and adapt its vtable")
    }

    /// Override the priority order.
    pub fn set_priority(&mut self, order: Vec<EpId>) {
        self.priority = order;
    }

    /// Borrow an EP by id.
    pub fn get(&self, id: EpId) -> Option<&dyn ExecutionProvider> {
        self.eps.get(id.0 as usize).map(|b| b.as_ref())
    }

    /// The priority order.
    pub fn priority(&self) -> &[EpId] {
        &self.priority
    }

    /// All EPs (in priority order) that can handle `op`, with their match info.
    pub fn candidates_for_op(
        &self,
        op: &Node,
        shapes: &[Shape],
        layouts: &[TensorLayout],
    ) -> Vec<(EpId, KernelMatch)> {
        let mut out = Vec::new();
        for &id in &self.priority {
            if let Some(ep) = self.get(id) {
                let m = ep.supports_op(op, shapes, layouts);
                if m.is_supported() {
                    out.push((id, m));
                }
            }
        }
        out
    }
}
