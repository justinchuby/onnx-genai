//! The domain registry: standard + custom ONNX operator domains and their
//! default opset versions (`docs/EAGER.md` §6).
//!
//! Each domain carries a `default_opset`. Unlike the design's `DomainInfo`
//! (`docs/EAGER.md` §6.4), which also nests a per-domain `KernelRegistry`, the
//! kernels here are **not** owned by the domain registry: they live in the
//! execution provider's [`OpRegistry`](onnx_runtime_ep_api::OpRegistry), keyed
//! on `(op_type, domain, opset)`. Reusing the EP registry keeps a single source
//! of truth for op support and avoids duplicating the kernel abstraction. The
//! per-domain kernel registry and the domain-namespace handle
//! (`ms.dispatch(...)`, §6.2) are DEFERRED.

use std::collections::HashMap;

use crate::opset::{resolve_opset, LATEST_ONNX_OPSET};

/// Metadata for one registered operator domain.
#[derive(Clone, Debug)]
pub struct DomainInfo {
    /// The domain name (`""` == the default ONNX domain).
    pub name: String,
    /// The default opset version for ops in this domain.
    pub default_opset: u64,
}

/// Registry of known operator domains and their default opsets (`docs/EAGER.md`
/// §6.4).
#[derive(Debug)]
pub struct DomainRegistry {
    domains: HashMap<String, DomainInfo>,
}

impl DomainRegistry {
    /// A registry pre-populated with the standard and common contrib domains
    /// (`docs/EAGER.md` §6.1): the default ONNX domain at [`LATEST_ONNX_OPSET`],
    /// `ai.onnx.ml` at 3, and `com.microsoft` at 1.
    pub fn new() -> Self {
        let mut reg = Self {
            domains: HashMap::new(),
        };
        reg.register("", LATEST_ONNX_OPSET);
        reg.register("ai.onnx.ml", 3);
        reg.register("com.microsoft", 1);
        reg
    }

    /// Register (or update) a domain with a default opset (`docs/EAGER.md` §6.1
    /// `register_domain`).
    pub fn register(&mut self, domain: &str, default_opset: u64) {
        self.domains.insert(
            domain.to_string(),
            DomainInfo {
                name: domain.to_string(),
                default_opset,
            },
        );
    }

    /// The registered default opset for `domain`, or [`LATEST_ONNX_OPSET`] for
    /// an unregistered domain (`docs/EAGER.md` §6.4 `resolve_opset`).
    pub fn default_opset(&self, domain: &str) -> u64 {
        self.domains
            .get(domain)
            .map(|d| d.default_opset)
            .unwrap_or(LATEST_ONNX_OPSET)
    }

    /// Resolve the effective opset for a dispatch: an explicit per-call value
    /// wins, otherwise the domain's registered default (`docs/EAGER.md` §5.2).
    pub fn resolve_opset(&self, domain: &str, explicit: Option<u64>) -> u64 {
        resolve_opset(self.default_opset(domain), explicit)
    }

    /// Whether `domain` is registered.
    pub fn contains(&self, domain: &str) -> bool {
        self.domains.contains_key(domain)
    }

    /// All registered domains and their default opsets (`docs/EAGER.md` §6.1
    /// `domains()`).
    pub fn domains(&self) -> Vec<(String, u64)> {
        let mut out: Vec<(String, u64)> = self
            .domains
            .values()
            .map(|d| (d.name.clone(), d.default_opset))
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }
}

impl Default for DomainRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_registered() {
        let reg = DomainRegistry::new();
        assert_eq!(reg.default_opset(""), LATEST_ONNX_OPSET);
        assert_eq!(reg.default_opset("ai.onnx.ml"), 3);
        assert_eq!(reg.default_opset("com.microsoft"), 1);
    }

    #[test]
    fn unregistered_domain_defaults_to_latest() {
        let reg = DomainRegistry::new();
        assert!(!reg.contains("com.acme"));
        assert_eq!(reg.default_opset("com.acme"), LATEST_ONNX_OPSET);
    }

    #[test]
    fn explicit_opset_overrides_domain_default() {
        let reg = DomainRegistry::new();
        assert_eq!(reg.resolve_opset("com.microsoft", Some(2)), 2);
        assert_eq!(reg.resolve_opset("com.microsoft", None), 1);
    }

    #[test]
    fn custom_domain_registration() {
        let mut reg = DomainRegistry::new();
        reg.register("com.acme", 5);
        assert_eq!(reg.default_opset("com.acme"), 5);
        assert!(reg.domains().contains(&("com.acme".to_string(), 5)));
    }
}
