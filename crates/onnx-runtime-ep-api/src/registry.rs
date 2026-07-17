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
    /// Normalized domain → op type → sorted registered `since_version`s.
    by_op: HashMap<String, HashMap<String, Vec<u64>>>,
}

impl OpRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a factory under `key`.
    pub fn register(&mut self, mut key: OpKey, factory: Box<dyn KernelFactory>) {
        key.domain = norm_domain(&key.domain).to_owned();
        let versions = self
            .by_op
            .entry(key.domain.clone())
            .or_default()
            .entry(key.op_type.clone())
            .or_default();
        if let Err(index) = versions.binary_search(&key.since_version) {
            versions.insert(index, key.since_version);
        }
        self.entries.insert(key, factory);
    }

    /// Look up the best matching factory: the highest `since_version` that is
    /// `<= opset` for the given `(op_type, domain)`.
    pub fn lookup(&self, op_type: &str, domain: &str, opset: u64) -> Option<&dyn KernelFactory> {
        let domain = norm_domain(domain);
        let versions = self.by_op.get(domain)?.get(op_type)?;
        let index = versions.partition_point(|&version| version <= opset);
        let since_version = *versions.get(index.checked_sub(1)?)?;
        self.entries
            .get(&OpKey::new(op_type, domain, since_version))
            .map(Box::as_ref)
    }

    /// Whether a factory is registered for `(op_type, domain)` at or before
    /// `opset`.
    pub fn supports(&self, op_type: &str, domain: &str, opset: u64) -> bool {
        let domain = norm_domain(domain);
        self.by_op
            .get(domain)
            .and_then(|ops| ops.get(op_type))
            .and_then(|versions| versions.first())
            .is_some_and(|&since_version| since_version <= opset)
    }

    /// Earliest registered opset for `(op_type, domain)`, if the EP knows the
    /// operator at any version. Used only to make decline diagnostics actionable.
    pub fn earliest_since_version(&self, op_type: &str, domain: &str) -> Option<u64> {
        let domain = norm_domain(domain);
        self.by_op.get(domain)?.get(op_type)?.first().copied()
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

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyFactory(u64);

    impl KernelFactory for DummyFactory {
        fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
            let _ = self.0;
            unreachable!("registry tests do not create kernels")
        }
    }

    #[test]
    fn indexed_queries_match_linear_reference() {
        let mut registry = OpRegistry::new();
        let mut state = 0x9e37_79b9_u64;
        let ops = ["Add", "Mul", "Gemm", "Attention"];
        let domains = ["", "ai.onnx", "com.microsoft", "pkg.nxrt"];

        for factory_id in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let op_type = ops[(state as usize) % ops.len()];
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let domain = domains[(state as usize) % domains.len()];
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let since_version = state % 25;
            registry.register(
                OpKey::new(op_type, domain, since_version),
                Box::new(DummyFactory(factory_id)),
            );
        }

        for _ in 0..512 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let op_type = ops[(state as usize) % ops.len()];
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let domain = domains[(state as usize) % domains.len()];
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let opset = state % 30;
            let domain = norm_domain(domain);

            let linear_lookup = registry
                .entries
                .iter()
                .filter(|(key, _)| {
                    key.op_type == op_type && key.domain == domain && key.since_version <= opset
                })
                .max_by_key(|(key, _)| key.since_version)
                .map(|(_, factory)| factory.as_ref());
            match (registry.lookup(op_type, domain, opset), linear_lookup) {
                (Some(indexed), Some(linear)) => assert!(std::ptr::eq(indexed, linear)),
                (None, None) => {}
                _ => panic!("indexed lookup differed from linear reference"),
            }

            let linear_supports = registry.entries.keys().any(|key| {
                key.op_type == op_type && key.domain == domain && key.since_version <= opset
            });
            assert_eq!(registry.supports(op_type, domain, opset), linear_supports);

            let linear_earliest = registry
                .entries
                .keys()
                .filter(|key| key.op_type == op_type && key.domain == domain)
                .map(|key| key.since_version)
                .min();
            assert_eq!(
                registry.earliest_since_version(op_type, domain),
                linear_earliest
            );
        }
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
        opset: u64,
        shapes: &[Shape],
        layouts: &[TensorLayout],
    ) -> Vec<(EpId, KernelMatch)> {
        let mut out = Vec::new();
        for &id in &self.priority {
            if let Some(ep) = self.get(id) {
                let m = ep.supports_op(op, opset, shapes, layouts);
                if m.is_supported() {
                    out.push((id, m));
                }
            }
        }
        out
    }
}
