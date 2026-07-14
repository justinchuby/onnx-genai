//! # `onnx-runtime-eager`
//!
//! Eager (single-op) execution for the ORT 2.0 runtime (`docs/EAGER.md`).
//! Dispatch individual ONNX ops to execution-provider kernels **without building
//! a graph** — a PyTorch-style experience with ONNX op semantics.
//!
//! This crate is the pure-Rust core of the design. It reuses the existing
//! runtime abstractions rather than inventing parallel ones:
//!
//! * [`ExecutionProvider`](onnx_runtime_ep_api::ExecutionProvider) /
//!   [`Kernel`](onnx_runtime_ep_api::Kernel) /
//!   [`OpRegistry`](onnx_runtime_ep_api::OpRegistry) from `onnx-runtime-ep-api`,
//! * the populated CPU registry from `onnx-runtime-ep-cpu`,
//! * per-op shape/dtype inference from `onnx-runtime-shape-inference`,
//! * the IR vocabulary ([`Node`](onnx_runtime_ir::Node),
//!   [`DataType`](onnx_runtime_ir::DataType),
//!   [`Attribute`](onnx_runtime_ir::Attribute), …) from `onnx-runtime-ir`.
//!
//! ## Phase-1 scope (`docs/EAGER.md`)
//!
//! Implemented: [`EagerContext`], [`EagerContext::dispatch`] (single-op CPU
//! dispatch), the [`DomainRegistry`](domain::DomainRegistry), opset resolution
//! ([`opset`]), and the compiled-[`KernelCache`](cache::KernelCache).
//!
//! DEFERRED (each is flagged at its hook point):
//! * PyO3 / Python bindings (§11), including the `Tensor` single→tensor /
//!   multi→tuple sugar and `ops.*` typed wrappers (§4.1).
//! * Subgraph ops `If`/`Loop`/`Scan` (§7).
//! * The opset context-manager / `nxrt.device()` context (§5.3, §2.2).
//! * GPU/CUDA EP dispatch and implicit cross-device transfer.
//! * Kernel-provided shape-inference fallback (§9.2).
//! * DLPack / numpy interop (§3).

mod cache;
mod dispatch;
mod domain;
mod error;
mod opset;
mod tensor;

use std::sync::{Mutex, OnceLock, RwLock};
use std::sync::Arc;

use onnx_runtime_ep_api::{EpConfig, ExecutionProvider};
use onnx_runtime_ep_cpu::CpuExecutionProvider;
use onnx_runtime_ir::DeviceId;
use onnx_runtime_shape_inference::InferenceRegistry;

pub use cache::{CacheStats, KernelCache, KernelCacheKey};
pub use domain::{DomainInfo, DomainRegistry};
pub use error::{EagerError, Result};
pub use opset::{resolve_opset, LATEST_ONNX_OPSET};
pub use tensor::Tensor;

/// Default compiled-kernel cache capacity (`docs/EAGER.md` §13: per-process LRU,
/// 4096).
const DEFAULT_CACHE_CAPACITY: usize = 4096;

/// The process-wide eager execution context (`docs/EAGER.md` §10.1).
///
/// Manages the available EPs, the domain registry, per-op shape inference, the
/// compiled-kernel cache, and the default device. It is thread-safe via internal
/// locking so it can back the [`global_context`] singleton.
pub struct EagerContext {
    /// Available EPs, auto-detected at initialization. CPU only in Phase-1;
    /// GPU EPs are added here as a *registration*, not a rewrite.
    pub(crate) eps: Vec<Arc<dyn ExecutionProvider>>,
    /// Compiled-kernel cache (keyed by op + shapes + dtypes + device).
    pub(crate) cache: Mutex<KernelCache>,
    /// Domain registry (standard + custom).
    pub(crate) domains: RwLock<DomainRegistry>,
    /// Per-op output shape/dtype inference (§9).
    pub(crate) inference: InferenceRegistry,
    /// Default device for ops with no inputs (`docs/EAGER.md` §4.3 case 3).
    pub(crate) default_device: DeviceId,
}

/// Auto-detect the available execution providers (`docs/EAGER.md` §10.1
/// `detect_available_eps`).
///
/// Phase-1: the always-available CPU EP only. Adding a GPU EP later is a matter
/// of pushing another initialized provider here — the dispatch path selects an
/// EP generically by device, with no hardcoded backend/vendor name.
fn detect_available_eps() -> Result<Vec<Arc<dyn ExecutionProvider>>> {
    // DEFERRED (EAGER.md §4.3 / §2.2): GPU/CUDA EP discovery + auto-transfer.
    let mut cpu = CpuExecutionProvider::new();
    cpu.initialize(&EpConfig::default())?;
    Ok(vec![Arc::new(cpu) as Arc<dyn ExecutionProvider>])
}

impl EagerContext {
    /// Initialize a context with auto-detected devices (`docs/EAGER.md` §10.1).
    pub fn new() -> Result<Self> {
        let eps = detect_available_eps()?;
        let default_device = eps
            .first()
            .map(|ep| ep.device_id())
            .unwrap_or_else(DeviceId::cpu);
        Ok(Self {
            eps,
            cache: Mutex::new(KernelCache::new(DEFAULT_CACHE_CAPACITY)),
            domains: RwLock::new(DomainRegistry::new()),
            inference: InferenceRegistry::default_registry(),
            default_device,
        })
    }

    /// The default device for input-less ops (`docs/EAGER.md` §4.3).
    pub fn default_device(&self) -> DeviceId {
        self.default_device
    }

    /// Register (or update) a custom domain's default opset (`docs/EAGER.md`
    /// §6.1 `register_domain`).
    pub fn register_domain(&self, domain: &str, default_opset: u64) {
        self.domains
            .write()
            .expect("domain registry lock poisoned")
            .register(domain, default_opset);
    }

    /// All registered domains and their default opsets (`docs/EAGER.md` §6.1
    /// `domains()`).
    pub fn domains(&self) -> Vec<(String, u64)> {
        self.domains
            .read()
            .expect("domain registry lock poisoned")
            .domains()
    }

    /// Current kernel-cache statistics (entries / hits / misses).
    pub fn cache_stats(&self) -> CacheStats {
        self.cache.lock().expect("kernel cache lock poisoned").stats()
    }

    /// Resolve the single target device from `inputs`; mixed devices are an
    /// error (`docs/EAGER.md` §1.6 / §4.3, Design Decision).
    pub(crate) fn resolve_device(&self, inputs: &[&Tensor]) -> Result<DeviceId> {
        let devices: Vec<DeviceId> = inputs.iter().map(|t| t.device()).collect();
        resolve_device_from_ids(&devices, self.default_device)
    }

    /// Get the EP (as an owned `Arc` clone) that owns `device`.
    pub(crate) fn ep_for_device(&self, device: DeviceId) -> Result<Arc<dyn ExecutionProvider>> {
        self.eps
            .iter()
            .find(|ep| ep.device_id() == device)
            .cloned()
            .ok_or(EagerError::NoEpForDevice(device))
    }
}

/// Resolve a single device from a list of input devices (`docs/EAGER.md` §4.3):
/// no inputs → `default`; all on one device → that device; otherwise a
/// [`EagerError::MixedDeviceInputs`] error.
///
/// Split out from [`EagerContext::resolve_device`] so the mixed-device policy is
/// unit-testable with fabricated [`DeviceId`]s even on a CPU-only build.
pub(crate) fn resolve_device_from_ids(devices: &[DeviceId], default: DeviceId) -> Result<DeviceId> {
    let mut unique: Vec<DeviceId> = Vec::new();
    for &d in devices {
        if !unique.contains(&d) {
            unique.push(d);
        }
    }
    match unique.len() {
        0 => Ok(default),
        1 => Ok(unique[0]),
        _ => Err(EagerError::MixedDeviceInputs {
            devices: unique,
            hint: "use .to(device) to move all inputs to the same device".to_string(),
        }),
    }
}

static GLOBAL_CONTEXT: OnceLock<EagerContext> = OnceLock::new();

/// Get or lazily initialize the process-global eager context (`docs/EAGER.md`
/// §10.2).
pub fn global_context() -> &'static EagerContext {
    GLOBAL_CONTEXT.get_or_init(|| EagerContext::new().expect("failed to initialize eager context"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn _assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn context_is_send_sync() {
        // Required for the `OnceLock<EagerContext>` global singleton.
        _assert_send_sync::<EagerContext>();
    }

    #[test]
    fn new_context_detects_cpu() {
        let ctx = EagerContext::new().unwrap();
        assert_eq!(ctx.default_device(), DeviceId::cpu());
        assert!(ctx.domains().iter().any(|(d, _)| d.is_empty()));
    }

    #[test]
    fn resolve_device_same_device_ok() {
        let ctx = EagerContext::new().unwrap();
        let a = Tensor::from_raw_in(
            ctx.ep_for_device(DeviceId::cpu()).unwrap(),
            onnx_runtime_ir::DataType::Float32,
            vec![2],
            &1.0f32.to_le_bytes().repeat(2),
        )
        .unwrap();
        let b = a.clone();
        assert_eq!(ctx.resolve_device(&[&a, &b]).unwrap(), DeviceId::cpu());
    }

    #[test]
    fn resolve_device_no_inputs_uses_default() {
        let ctx = EagerContext::new().unwrap();
        assert_eq!(ctx.resolve_device(&[]).unwrap(), ctx.default_device());
    }

    #[test]
    fn resolve_device_mixed_is_error() {
        // Fabricated CPU + CUDA device ids exercise the mixed-device guard even
        // though only a CPU EP exists (a real cross-device tensor is infeasible
        // CPU-only), per docs/EAGER.md §1.6.
        use onnx_runtime_ir::DeviceType;
        let cpu = DeviceId::cpu();
        let cuda = DeviceId::new(DeviceType::Cuda, 0);
        assert!(matches!(
            resolve_device_from_ids(&[cpu, cuda], cpu),
            Err(EagerError::MixedDeviceInputs { .. })
        ));
        // Same-device and empty cases resolve fine.
        assert_eq!(resolve_device_from_ids(&[cpu, cpu], cpu).unwrap(), cpu);
        assert_eq!(resolve_device_from_ids(&[], cpu).unwrap(), cpu);
    }
}
