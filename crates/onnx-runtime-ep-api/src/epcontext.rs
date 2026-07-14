//! Runtime `EpContext` form and the `source`-keyed [`EpContextRegistry`]
//! (`docs/ORT2.md` §55.1 / §55.6).
//!
//! ## Two forms of one thing
//!
//! An ORT `com.microsoft::EPContext` node is the **on-disk / interchange** form
//! of a compiled-EP context (§55.1). The loader (`onnx-runtime-loader`) owns the
//! typed *view* over that node (`EpContextNode`) and resolves its blob
//! (`EpContextBlob`). This module owns the **runtime** form: the in-memory
//! [`EpContext`] an EP produces via [`ExecutionProvider::save_context`] and
//! consumes via [`ExecutionProvider::load_context`], plus the registry that maps
//! a node's `source` attribute → the [`EpId`] that owns it.
//!
//! ## Model-agnostic dispatch (hard rule, §55.6)
//!
//! EP selection for an `EPContext` node is **always** by the node's `source`
//! attribute, resolved through [`EpContextRegistry`] — **never** by a hardcoded
//! EP/vendor name. Each EP declares the `source` key(s) it accepts (from its own
//! config/data, not code) via [`ExecutionProvider::context_source_keys`]; the
//! registry maps key → EP. Adding a new compiled EP requires **no change** to
//! loader/session dispatch code.

use std::collections::HashMap;

use crate::error::{EpError, Result};
use crate::provider::{EpId, ExecutionProvider};

/// The **runtime form** of a compiled-EP context (`docs/ORT2.md` §4 / §55.1).
///
/// This is the in-memory representation an EP produces from a freshly compiled
/// subgraph ([`ExecutionProvider::save_context`]) or restores at load
/// ([`ExecutionProvider::load_context`]), skipping the expensive
/// convert+compile step. It is distinct from the loader's on-disk
/// `EpContextNode`/`EpContextBlob` (which is the serialized, in-graph,
/// tool-portable interchange form); the session maps between the two (§55.3).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EpContext {
    /// Name of the EP that produced (and must consume) this context.
    pub ep_name: String,
    /// SDK/toolchain version that generated the context (maps to the node's
    /// `ep_sdk_version` attribute — used for invalidation / diagnostics).
    pub ep_version: String,
    /// Opaque compiled vendor blob (maps to the node's `ep_cache_context`).
    pub data: Vec<u8>,
    /// Graph nodes this context covers (the partition boundary it replaces).
    pub covered_nodes: Vec<onnx_runtime_ir::NodeId>,
    /// Device/target fingerprint the blob was compiled for; filled and
    /// validated by the owning EP to reject a mismatched load.
    pub device_fingerprint: String,
}

impl EpContext {
    /// Construct a runtime context from its parts.
    pub fn new(
        ep_name: impl Into<String>,
        ep_version: impl Into<String>,
        data: Vec<u8>,
        covered_nodes: Vec<onnx_runtime_ir::NodeId>,
        device_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            ep_name: ep_name.into(),
            ep_version: ep_version.into(),
            data,
            covered_nodes,
            device_fingerprint: device_fingerprint.into(),
        }
    }
}

/// Registry mapping an `EPContext` node's `source` key → the [`EpId`] that owns
/// it (`docs/ORT2.md` §55.6).
///
/// EPs register the `source` key(s) they accept; dispatch ([`claim`]) is a pure
/// lookup. There are **no hardcoded vendor names** here — the keys come entirely
/// from each EP's [`ExecutionProvider::context_source_keys`].
///
/// ## Duplicate keys are rejected (reject-duplicate-key)
///
/// [`register`] returns [`EpError::DuplicateContextSource`] if a `source` key is
/// already claimed by another EP, rather than silently overwriting
/// (last-writer-wins) it. Rationale: two EPs claiming the same `source` is a
/// **configuration error** (the model would dispatch ambiguously), and silently
/// dropping one binding would make which EP restores a context depend on
/// registration order — a non-deterministic, hard-to-debug failure at load. A
/// registrant re-declaring the *same* `(key, ep)` binding is idempotent and
/// accepted (it is not a conflict).
///
/// [`register`]: EpContextRegistry::register
/// [`claim`]: EpContextRegistry::claim
#[derive(Clone, Debug, Default)]
pub struct EpContextRegistry {
    by_source: HashMap<String, EpId>,
}

impl EpContextRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declare that `ep` accepts each key in `source_keys` for `EPContext`
    /// dispatch. Keys come from the EP's own config/data (§55.6), never
    /// hardcoded here.
    ///
    /// # Errors
    ///
    /// Returns [`EpError::DuplicateContextSource`] if any key is already
    /// registered to a **different** EP (reject-duplicate-key). Re-registering
    /// the same `(key, ep)` binding is idempotent. On error, keys processed
    /// before the conflict remain registered (the caller should treat a
    /// duplicate as a fatal configuration error).
    pub fn register(&mut self, ep: EpId, source_keys: &[String]) -> Result<()> {
        for key in source_keys {
            match self.by_source.get(key) {
                Some(&existing) if existing == ep => {} // idempotent re-declare
                Some(&existing) => {
                    return Err(EpError::DuplicateContextSource {
                        source_key: key.clone(),
                        existing,
                        new: ep,
                    });
                }
                None => {
                    self.by_source.insert(key.clone(), ep);
                }
            }
        }
        Ok(())
    }

    /// Look up the EP that owns a node's `source` attribute (§55.6).
    ///
    /// A pure lookup: `None` source (attribute absent) or a `source` matching no
    /// registered EP both yield `None` — the node is **unclaimed**. The session
    /// turns an unclaimed node into a clear [`EpError::NoEpForContext`] rather
    /// than guessing.
    pub fn claim(&self, source: Option<&str>) -> Option<EpId> {
        self.by_source.get(source?).copied()
    }

    /// Number of registered `source` keys.
    pub fn len(&self) -> usize {
        self.by_source.len()
    }

    /// Whether no `source` keys are registered.
    pub fn is_empty(&self) -> bool {
        self.by_source.is_empty()
    }
}

/// Build an [`EpContextRegistry`] from a set of registered EPs (§55.6).
///
/// A **pure function over the EP set**: it iterates the `(EpId, &dyn
/// ExecutionProvider)` pairs, asks each EP for its
/// [`ExecutionProvider::context_source_keys`], and populates the registry. This
/// lets the session build the dispatch table with **no hardcoded vendor names**
/// — an EP that returns no keys (the default) simply doesn't participate.
///
/// # Errors
///
/// Propagates [`EpError::DuplicateContextSource`] if two EPs declare the same
/// `source` key (reject-duplicate-key — see [`EpContextRegistry::register`]).
pub fn build_ep_context_registry<'a, I>(eps: I) -> Result<EpContextRegistry>
where
    I: IntoIterator<Item = (EpId, &'a dyn ExecutionProvider)>,
{
    let mut registry = EpContextRegistry::new();
    for (id, ep) in eps {
        let keys = ep.context_source_keys();
        if keys.is_empty() {
            continue;
        }
        registry.register(id, &keys)?;
    }
    Ok(registry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::{Kernel, KernelMatch};
    use crate::provider::{DeviceBuffer, EpConfig, Fence};
    use onnx_runtime_ir::{DeviceId, DeviceType, Node, NodeId, Shape, TensorLayout};

    /// A mock compiled EP that participates in `EPContext` dispatch. It declares
    /// two `source` keys, produces a fixed context blob, and validates the blob
    /// on load. All other `ExecutionProvider` methods are unused stubs.
    struct MockCompiledEp {
        source_keys: Vec<String>,
    }

    impl MockCompiledEp {
        const BLOB: &'static [u8] = b"mock-compiled-context-v1";

        fn new() -> Self {
            // Keys come from "config", not hardcoded in dispatch logic (§55.6).
            Self {
                source_keys: vec!["MOCK".to_string(), "MockExecutionProvider".to_string()],
            }
        }
    }

    impl ExecutionProvider for MockCompiledEp {
        fn name(&self) -> &str {
            "mock_compiled_ep"
        }

        fn device_type(&self) -> DeviceType {
            DeviceType::Custom(0)
        }

        fn device_id(&self) -> DeviceId {
            DeviceId::new(DeviceType::Custom(0), 0)
        }

        fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
            Ok(())
        }

        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }

        fn supports_op(
            &self,
            _op: &Node,
            _shapes: &[Shape],
            _layouts: &[TensorLayout],
        ) -> KernelMatch {
            KernelMatch::Unsupported
        }

        fn get_kernel(
            &self,
            _op: &Node,
            _shapes: &[Vec<usize>],
            _opset: u64,
        ) -> Result<Box<dyn Kernel>> {
            Err(EpError::NoEpForOp {
                op_type: "<mock>".to_string(),
            })
        }

        fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
            Err(EpError::NotInitialized)
        }

        fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
            Ok(())
        }

        fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
            Ok(())
        }

        fn copy_async(
            &self,
            _src: &DeviceBuffer,
            _dst: &mut DeviceBuffer,
            _size: usize,
        ) -> Result<Fence> {
            Ok(Fence::default())
        }

        fn sync(&self) -> Result<()> {
            Ok(())
        }

        // --- EPContext contract (§55) ---

        fn context_source_keys(&self) -> Vec<String> {
            self.source_keys.clone()
        }

        fn save_context(&self) -> Result<EpContext> {
            Ok(EpContext::new(
                self.name(),
                "1.2.3",
                Self::BLOB.to_vec(),
                vec![NodeId(7)],
                "mock-device",
            ))
        }

        fn load_context(&self, ctx: &EpContext) -> Result<()> {
            if ctx.data == Self::BLOB {
                Ok(())
            } else {
                Err(EpError::KernelFailed("mock: unexpected context blob".to_string()))
            }
        }
    }

    /// A plain EP with no `EPContext` override — exercises the trait defaults.
    struct PlainEp;

    impl ExecutionProvider for PlainEp {
        fn name(&self) -> &str {
            "plain_ep"
        }
        fn device_type(&self) -> DeviceType {
            DeviceType::Cpu
        }
        fn device_id(&self) -> DeviceId {
            DeviceId::cpu()
        }
        fn initialize(&mut self, _config: &EpConfig) -> Result<()> {
            Ok(())
        }
        fn shutdown(&mut self) -> Result<()> {
            Ok(())
        }
        fn supports_op(
            &self,
            _op: &Node,
            _shapes: &[Shape],
            _layouts: &[TensorLayout],
        ) -> KernelMatch {
            KernelMatch::Unsupported
        }
        fn get_kernel(
            &self,
            _op: &Node,
            _shapes: &[Vec<usize>],
            _opset: u64,
        ) -> Result<Box<dyn Kernel>> {
            Err(EpError::NoEpForOp {
                op_type: "<plain>".to_string(),
            })
        }
        fn allocate(&self, _size: usize, _alignment: usize) -> Result<DeviceBuffer> {
            Err(EpError::NotInitialized)
        }
        fn deallocate(&self, _buffer: DeviceBuffer) -> Result<()> {
            Ok(())
        }
        fn copy(&self, _src: &DeviceBuffer, _dst: &mut DeviceBuffer, _size: usize) -> Result<()> {
            Ok(())
        }
        fn copy_async(
            &self,
            _src: &DeviceBuffer,
            _dst: &mut DeviceBuffer,
            _size: usize,
        ) -> Result<Fence> {
            Ok(Fence::default())
        }
        fn sync(&self) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn register_and_claim_by_source_key() {
        let mock = MockCompiledEp::new();
        let mut reg = EpContextRegistry::new();
        let ep_id = EpId(3);
        reg.register(ep_id, &mock.context_source_keys()).unwrap();

        // Every declared key resolves to the mock's id.
        assert_eq!(reg.claim(Some("MOCK")), Some(ep_id));
        assert_eq!(reg.claim(Some("MockExecutionProvider")), Some(ep_id));
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn unmatched_and_absent_source_are_unclaimed() {
        let mock = MockCompiledEp::new();
        let mut reg = EpContextRegistry::new();
        reg.register(EpId(0), &mock.context_source_keys()).unwrap();

        // A different vendor's key is not registered → unclaimed (no guessing).
        assert_eq!(reg.claim(Some("QNN")), None);
        // Absent `source` attribute → unclaimed.
        assert_eq!(reg.claim(None), None);
    }

    #[test]
    fn duplicate_source_key_is_rejected() {
        let mut reg = EpContextRegistry::new();
        reg.register(EpId(0), &["MOCK".to_string()]).unwrap();

        // A different EP claiming the same key is a configuration error.
        let err = reg
            .register(EpId(1), &["MOCK".to_string()])
            .expect_err("duplicate source key must be rejected");
        match err {
            EpError::DuplicateContextSource {
                source_key,
                existing,
                new,
            } => {
                assert_eq!(source_key, "MOCK");
                assert_eq!(existing, EpId(0));
                assert_eq!(new, EpId(1));
            }
            other => panic!("expected DuplicateContextSource, got {other:?}"),
        }

        // The original binding is untouched.
        assert_eq!(reg.claim(Some("MOCK")), Some(EpId(0)));
    }

    #[test]
    fn re_registering_same_binding_is_idempotent() {
        let mut reg = EpContextRegistry::new();
        reg.register(EpId(2), &["MOCK".to_string()]).unwrap();
        // Same key, same EP: not a conflict.
        reg.register(EpId(2), &["MOCK".to_string()]).unwrap();
        assert_eq!(reg.claim(Some("MOCK")), Some(EpId(2)));
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn build_registry_from_eps_skips_non_participants() {
        let mock = MockCompiledEp::new();
        let plain = PlainEp;
        let eps: Vec<(EpId, &dyn ExecutionProvider)> =
            vec![(EpId(0), &plain), (EpId(1), &mock)];

        let reg = build_ep_context_registry(eps).unwrap();

        // The plain EP (empty keys) does not appear; the mock's keys map to it.
        assert_eq!(reg.len(), 2);
        assert_eq!(reg.claim(Some("MOCK")), Some(EpId(1)));
        assert_eq!(reg.claim(Some("MockExecutionProvider")), Some(EpId(1)));
    }

    #[test]
    fn build_registry_from_eps_propagates_duplicate_error() {
        // Two mocks declaring the same keys under different ids must conflict.
        let a = MockCompiledEp::new();
        let b = MockCompiledEp::new();
        let eps: Vec<(EpId, &dyn ExecutionProvider)> = vec![(EpId(0), &a), (EpId(1), &b)];

        let err = build_ep_context_registry(eps)
            .expect_err("two EPs on the same source key must conflict");
        assert!(matches!(err, EpError::DuplicateContextSource { .. }));
    }

    #[test]
    fn save_load_round_trip_preserves_bytes() {
        let mock = MockCompiledEp::new();
        let ctx = mock.save_context().unwrap();
        assert_eq!(ctx.ep_name, "mock_compiled_ep");
        assert_eq!(ctx.ep_version, "1.2.3");
        assert_eq!(ctx.data, MockCompiledEp::BLOB);
        assert_eq!(ctx.covered_nodes, vec![NodeId(7)]);
        // The EP accepts its own saved context.
        mock.load_context(&ctx).unwrap();

        // A tampered blob is rejected.
        let mut bad = ctx.clone();
        bad.data.push(0xFF);
        assert!(mock.load_context(&bad).is_err());
    }

    #[test]
    fn plain_ep_defaults_are_empty_and_unsupported() {
        let plain = PlainEp;
        assert!(plain.context_source_keys().is_empty());
        assert!(matches!(
            plain.save_context(),
            Err(EpError::UnsupportedContext { .. })
        ));

        let ctx = EpContext::default();
        assert!(matches!(
            plain.load_context(&ctx),
            Err(EpError::UnsupportedContext { .. })
        ));
    }

    #[test]
    fn no_ep_for_context_error_carries_source() {
        // Illustrates how session dispatch surfaces an unclaimed node (§55.3).
        let reg = EpContextRegistry::new();
        let source = Some("QNN");
        let err = reg
            .claim(source)
            .ok_or_else(|| EpError::NoEpForContext {
                source_key: source.map(str::to_owned),
            })
            .unwrap_err();
        match err {
            EpError::NoEpForContext { source_key } => {
                assert_eq!(source_key.as_deref(), Some("QNN"))
            }
            other => panic!("expected NoEpForContext, got {other:?}"),
        }
    }
}
