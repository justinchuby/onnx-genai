//! Op → kernel-factory registry and the EP registry (§4.3, §4.6).

use std::collections::HashMap;
use std::path::Path;

use onnx_runtime_ir::{DataType, Node, Shape, TensorLayout};

use crate::abi::LegacyOrtEp;
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

    /// Iterate over every registered key `(op_type, domain, since_version)`.
    ///
    /// Domains are already normalized (`"ai.onnx"` collapsed to `""`) at
    /// registration time, so the returned keys carry the exact domain a plugin
    /// EP should advertise to ORT. Used to derive kernel-registry entries from
    /// the real registry rather than a hand-maintained parallel list.
    pub fn keys(&self) -> impl Iterator<Item = &OpKey> {
        self.entries.keys()
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

    #[test]
    fn schema_resolution_selects_latest_registered_since_version() {
        let mut registry = OpRegistry::new();
        registry.register(
            OpKey::new("DsaIndexSelect", "pkg.nxrt", 1),
            Box::new(DummyFactory(1)),
        );
        registry.register(
            OpKey::new("DsaIndexSelect", "pkg.nxrt", 3),
            Box::new(DummyFactory(3)),
        );

        let resolved_v1 = registry
            .lookup("DsaIndexSelect", "pkg.nxrt", 1)
            .expect("opset 1 resolves schema v1");
        let resolved_v2 = registry
            .lookup("DsaIndexSelect", "pkg.nxrt", 2)
            .expect("opset 2 still resolves schema v1");
        let resolved_v3 = registry
            .lookup("DsaIndexSelect", "pkg.nxrt", 3)
            .expect("opset 3 resolves the newly registered schema");
        let resolved_v4 = registry
            .lookup("DsaIndexSelect", "pkg.nxrt", 4)
            .expect("opset 4 keeps using the newest schema");

        assert!(std::ptr::eq(resolved_v1, resolved_v2));
        assert!(!std::ptr::eq(resolved_v1, resolved_v3));
        assert!(std::ptr::eq(resolved_v3, resolved_v4));
        assert!(registry.lookup("DsaIndexSelect", "pkg.nxrt", 0).is_none());
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

    /// Load a legacy ORT plugin EP from a shared library.
    ///
    /// The loaded provider remains in this registry (and keeps its dynamic
    /// library handle alive). Legacy EPs negotiate graph-level subgraphs through
    /// [`crate::PluginExecutionPlan`], rather than claiming individual nodes.
    pub fn load_legacy(&mut self, path: &Path, config: &EpConfig) -> Result<EpId> {
        Ok(self.register(Box::new(LegacyOrtEp::load(path, config)?)))
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
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> Vec<(EpId, KernelMatch)> {
        let mut out = Vec::new();
        for &id in &self.priority {
            if let Some(ep) = self.get(id) {
                let m = ep.supports_op(op, opset, shapes, input_dtypes, layouts);
                if m.is_supported() {
                    out.push((id, m));
                }
            }
        }
        out
    }
}

#[cfg(all(test, target_os = "linux"))]
mod legacy_loader_tests {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use super::*;

    fn build_fixture(name: &str) -> PathBuf {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let source = manifest.join("tests/fixtures").join(format!("{name}.c"));
        let output_dir = manifest.join("target/legacy-plugin-fixtures");
        std::fs::create_dir_all(&output_dir).expect("create fixture output directory");
        let output = output_dir.join(format!("lib{name}.so"));
        let status = Command::new("cc")
            .args(["-shared", "-fPIC"])
            .arg(&source)
            .args(["-o"])
            .arg(&output)
            .status()
            .expect("invoke C compiler for legacy plugin fixture");
        assert!(status.success(), "compile legacy plugin fixture");
        output
    }

    #[test]
    fn load_legacy_resolves_and_invokes_plugin_factory() {
        let path = build_fixture("legacy_plugin_stub");
        let mut registry = EpRegistry::new();
        let mut config = EpConfig::default();
        config.options.insert(
            "legacy.registration_name".into(),
            "synthetic-registration".into(),
        );

        let id = registry.load_legacy(&path, &config).expect("load plugin");

        assert_eq!(id, EpId(0));
        assert_eq!(
            registry.get(id).map(ExecutionProvider::name),
            Some("synthetic_legacy_ep")
        );
    }

    #[test]
    fn load_legacy_reports_missing_factory_symbol() {
        let path = build_fixture("missing_create_ep_factories");
        let error = EpRegistry::new()
            .load_legacy(&path, &EpConfig::default())
            .expect_err("missing CreateEpFactories must fail cleanly");

        assert!(matches!(
            error,
            crate::EpError::EpLoadFailed { ref reason, .. }
                if reason.contains("CreateEpFactories symbol was not found")
        ));
    }

    #[test]
    fn load_legacy_rejects_an_incompatible_plugin_abi() {
        let path = build_fixture("incompatible_api_plugin");
        let error = EpRegistry::new()
            .load_legacy(&path, &EpConfig::default())
            .expect_err("newer plugin ABI must fail cleanly");

        assert!(matches!(
            error,
            crate::EpError::EpLoadFailed { ref reason, .. }
                if reason.contains("requires ORT API version")
        ));
    }
}
