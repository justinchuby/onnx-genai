//! Native-LoRA runtime manager (design `docs/NATIVE_LORA_DESIGN.md` §D, **P4**).
//!
//! The [`LoraManager`] is the engine-side owner of decoded adapters. It:
//!
//! 1. loads a PEFT adapter directory or ONNX Runtime `.onnx_adapter` file
//!    through the format-detecting loader ([`super::format::load_adapter`]),
//!    which normalizes each `A`/`B` factor into the ONNX-`MatMul` orientation;
//! 2. keeps a small, byte-budgeted **LRU** of decoded adapters (the transposed
//!    `A_t`/`B_t` host bytes), evicting the least-recently-used inactive adapter
//!    when the budget is exceeded — mirroring the KV cache's byte-budget pattern;
//! 3. translates a decoded adapter into the session crate's format-agnostic
//!    [`LoraAdapterSpec`] (the **loaded-adapter → spec bridge**), which is what
//!    the injection pass consumes. This is the single place that couples semantic
//!    module names to the session's injection input, keeping the session crate
//!    free of any on-disk adapter format specifics (the dependency direction is
//!    engine → session).
//!
//! # Phase-1 selection model — single fixed adapter per session
//!
//! Phase 1 applies **one** adapter (or none) to the whole session via the
//! named-input override mechanism, matching the current single-sequence native
//! decode. The manager tracks which adapter is *active*; the injected graph is
//! produced at session/graph **build** time (the adapter's targets must be known
//! then, so the overridable inputs exist), and activation/deactivation is a cheap
//! session-level toggle over the already-injected override buffers
//! ([`InferenceSession::set_lora_active`]). There is **no per-request adapter id**
//! in Phase 1 — that is deferred to P7 behind scheduler isolation.
//!
//! [`LoraAdapterSpec`]: onnx_runtime_session::lora_inject::LoraAdapterSpec
//! [`InferenceSession::set_lora_active`]: onnx_runtime_session::InferenceSession::set_lora_active

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use onnx_runtime_session::lora_inject::{LoraAdapterSpec, LoraModuleSpec};

use super::format::{AdapterLoadError, LoadedAdapter, load_adapter};

/// A stable identifier for a loaded adapter. Derived from the adapter's on-disk
/// name so re-loading the same directory resolves to the same id.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdapterId(String);

impl AdapterId {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for AdapterId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Errors raised by the [`LoraManager`].
#[derive(Debug, thiserror::Error)]
pub enum LoraManagerError {
    #[error(transparent)]
    Load(#[from] AdapterLoadError),
    #[error("no adapter with id {0:?} is loaded")]
    UnknownAdapter(String),
    #[error(
        "adapter {name:?} needs {needed} bytes but the manager budget is only {budget} bytes; \
         raise the adapter cache budget"
    )]
    OverBudget {
        name: String,
        needed: usize,
        budget: usize,
    },
}

/// One decoded adapter held in the LRU, with its precomputed host-byte footprint.
struct CachedAdapter {
    id: AdapterId,
    adapter: LoadedAdapter,
    bytes: usize,
}

/// The engine-side native-LoRA manager (design §D). Owns decoded adapters,
/// enforces a byte-budgeted LRU, and tracks the single active adapter.
pub struct LoraManager {
    /// Byte budget for the decoded-adapter cache. `0` means unbounded.
    budget_bytes: usize,
    /// Current decoded-adapter host-byte footprint.
    used_bytes: usize,
    /// LRU order: front = least-recently-used, back = most-recently-used.
    entries: VecDeque<CachedAdapter>,
    /// The single active adapter (Phase-1 selection model), or `None` for
    /// base-only.
    active: Option<AdapterId>,
}

impl LoraManager {
    /// A manager with the given decoded-adapter cache budget in bytes. Pass `0`
    /// for an unbounded cache (never evicts).
    pub fn with_budget(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            entries: VecDeque::new(),
            active: None,
        }
    }

    /// Load (or re-touch) a supported adapter and cache it. Returns its id.
    /// A single adapter larger than a non-zero budget is rejected rather than
    /// silently blowing the budget.
    pub fn load(&mut self, path: impl AsRef<Path>) -> Result<AdapterId, LoraManagerError> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let adapter = load_adapter(&path)?;
        let id = AdapterId(adapter.name.clone());

        // Already cached: move it to most-recently-used and return.
        if let Some(index) = self.entries.iter().position(|entry| entry.id == id) {
            let entry = self.entries.remove(index).expect("index just found");
            self.entries.push_back(entry);
            return Ok(id);
        }

        let bytes = adapter_bytes(&adapter);
        if self.budget_bytes != 0 && bytes > self.budget_bytes {
            return Err(LoraManagerError::OverBudget {
                name: id.0,
                needed: bytes,
                budget: self.budget_bytes,
            });
        }
        self.evict_until_fits(bytes);
        self.used_bytes += bytes;
        self.entries.push_back(CachedAdapter {
            id: id.clone(),
            adapter,
            bytes,
        });
        Ok(id)
    }

    /// Evict least-recently-used **inactive** adapters until `incoming` bytes fit
    /// within the budget. The active adapter is never evicted.
    fn evict_until_fits(&mut self, incoming: usize) {
        if self.budget_bytes == 0 {
            return;
        }
        while self.used_bytes + incoming > self.budget_bytes {
            let Some(index) = self
                .entries
                .iter()
                .position(|entry| Some(&entry.id) != self.active.as_ref())
            else {
                break; // only the active adapter remains; keep it
            };
            let evicted = self.entries.remove(index).expect("index just found");
            self.used_bytes -= evicted.bytes;
        }
    }

    /// Borrow a cached decoded adapter, touching its LRU recency.
    pub fn get(&mut self, id: &AdapterId) -> Option<&LoadedAdapter> {
        let index = self.entries.iter().position(|entry| &entry.id == id)?;
        let entry = self.entries.remove(index).expect("index just found");
        self.entries.push_back(entry);
        Some(&self.entries.back().expect("just pushed").adapter)
    }

    /// Build the session-crate injection spec for a loaded adapter (the PEFT →
    /// spec bridge). Touches the adapter's LRU recency.
    pub fn spec(&mut self, id: &AdapterId) -> Result<LoraAdapterSpec, LoraManagerError> {
        let adapter = self
            .get(id)
            .ok_or_else(|| LoraManagerError::UnknownAdapter(id.0.clone()))?;
        Ok(adapter_spec_from_loaded(adapter))
    }

    /// Mark an adapter active (Phase-1 single fixed adapter). The adapter must be
    /// loaded. Activation on the running session is a separate, cheap toggle (see
    /// the module docs); this records the manager's intent and recency.
    pub fn activate(&mut self, id: &AdapterId) -> Result<(), LoraManagerError> {
        if self.get(id).is_none() {
            return Err(LoraManagerError::UnknownAdapter(id.0.clone()));
        }
        self.active = Some(id.clone());
        Ok(())
    }

    /// Clear the active adapter (base-only). The decoded adapter stays cached.
    pub fn deactivate(&mut self) {
        self.active = None;
    }

    /// The most-recently-used cached adapter id, if any. In the Phase-1
    /// single-fixed-adapter model this is the session's one adapter, and is used
    /// to re-activate after a [`deactivate`](Self::deactivate).
    pub fn most_recent(&self) -> Option<&AdapterId> {
        self.entries.back().map(|entry| &entry.id)
    }

    /// The currently active adapter id, if any.
    pub fn active(&self) -> Option<&AdapterId> {
        self.active.as_ref()
    }

    /// The number of decoded adapters currently cached.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Current decoded-adapter host-byte footprint.
    pub fn used_bytes(&self) -> usize {
        self.used_bytes
    }
}

/// Translate a decoded [`LoadedAdapter`] (P2a) into the session crate's
/// format-agnostic [`LoraAdapterSpec`] (the input to the injection pass). This is
/// the **loaded-adapter → spec bridge**: it maps each normalized `A_t = [K, r]` / `B_t =
/// [r, N]` factor and its semantic module name onto a [`LoraModuleSpec`], so the
/// session crate never sees a format-specific type. Modules are emitted in the
/// loader's stable (`module_key`-sorted) order.
pub fn adapter_spec_from_loaded(adapter: &LoadedAdapter) -> LoraAdapterSpec {
    LoraAdapterSpec {
        name: adapter.name.clone(),
        modules: adapter
            .modules
            .values()
            .map(|module| LoraModuleSpec {
                module_name: module.module_name.clone(),
                layer_index: module.layer_index,
                rank: module.rank,
                scale: module.scale,
                a_t: module.a_transposed.clone(),
                b_t: module.b_transposed.clone(),
            })
            .collect(),
    }
}

/// Host-byte footprint of a decoded adapter's transposed factors.
fn adapter_bytes(adapter: &LoadedAdapter) -> usize {
    adapter
        .modules
        .values()
        .map(|module| module.a_transposed.data.len() + module.b_transposed.data.len())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::DataType;
    use safetensors::Dtype;
    use safetensors::tensor::{TensorView, serialize_to_file};
    use std::collections::HashMap;
    use std::fs;

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect()
    }

    /// Write a minimal single-module PEFT adapter into a named subdirectory of a
    /// fresh temp root, so the loaded adapter's `name` is the deterministic
    /// `name` argument. Returns the temp root (kept alive by the caller) and the
    /// adapter directory path to load.
    fn write_adapter(name: &str, r: usize, k: usize, n: usize) -> (tempfile::TempDir, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join(name);
        fs::create_dir(&directory).unwrap();
        fs::write(
            directory.join("adapter_config.json"),
            format!(
                r#"{{"r": {r}, "lora_alpha": {alpha}, "target_modules": ["q_proj"], "fan_in_fan_out": false}}"#,
                alpha = 2 * r
            ),
        )
        .unwrap();
        let a = f32_bytes(&vec![0.5_f32; r * k]);
        let b = f32_bytes(&vec![0.25_f32; n * r]);
        let a_view = TensorView::new(Dtype::F32, vec![r, k], &a).unwrap();
        let b_view = TensorView::new(Dtype::F32, vec![n, r], &b).unwrap();
        let mut views = HashMap::new();
        views.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight".to_owned(),
            a_view,
        );
        views.insert(
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight".to_owned(),
            b_view,
        );
        serialize_to_file(&views, None, &directory.join("adapter_model.safetensors")).unwrap();
        (root, directory)
    }

    #[test]
    fn bridge_maps_every_module_to_matmul_ready_spec() {
        let (_root, directory) = write_adapter("adapterA", 2, 4, 6);
        let adapter = load_adapter(&directory).unwrap();
        let spec = adapter_spec_from_loaded(&adapter);
        assert_eq!(spec.name, "adapterA");
        assert_eq!(spec.modules.len(), 1);
        let module = &spec.modules[0];
        assert_eq!(module.module_name, "self_attn.q_proj");
        assert_eq!(module.layer_index, 0);
        assert_eq!(module.rank, 2);
        assert_eq!(module.a_t.dtype, DataType::Float32);
        assert_eq!(module.a_t.dims, vec![4, 2]); // [K, r]
        assert_eq!(module.b_t.dims, vec![2, 6]); // [r, N]
        assert_eq!(module.scale, 4.0 / 2.0);
    }

    #[test]
    fn load_activate_deactivate_tracks_single_adapter() {
        let (_root, directory) = write_adapter("adapterA", 2, 4, 6);
        let mut manager = LoraManager::with_budget(0);
        let id = manager.load(&directory).unwrap();
        assert_eq!(manager.len(), 1);
        assert!(manager.active().is_none());
        manager.activate(&id).unwrap();
        assert_eq!(manager.active(), Some(&id));
        manager.deactivate();
        assert!(manager.active().is_none());
        // The decoded adapter stays cached after deactivation.
        assert_eq!(manager.len(), 1);
    }

    #[test]
    fn reload_is_idempotent_and_touches_recency() {
        let (_root, directory) = write_adapter("adapterA", 1, 2, 2);
        let mut manager = LoraManager::with_budget(0);
        let first = manager.load(&directory).unwrap();
        let bytes_after_first = manager.used_bytes();
        let second = manager.load(&directory).unwrap();
        assert_eq!(first, second);
        assert_eq!(manager.len(), 1);
        assert_eq!(manager.used_bytes(), bytes_after_first);
    }

    #[test]
    fn lru_evicts_inactive_but_never_active() {
        let (_root_a, dir_a) = write_adapter("adapterA", 1, 2, 2);
        let (_root_b, dir_b) = write_adapter("adapterB", 1, 2, 2);
        let (_root_c, dir_c) = write_adapter("adapterC", 1, 2, 2);
        let one = load_adapter(&dir_a).unwrap();
        let per_adapter = adapter_bytes(&one);
        // Budget fits exactly two adapters.
        let mut manager = LoraManager::with_budget(per_adapter * 2);
        let a = manager.load(&dir_a).unwrap();
        manager.activate(&a).unwrap();
        let _b = manager.load(&dir_b).unwrap();
        assert_eq!(manager.len(), 2);
        // Loading a third evicts the LRU inactive (B), keeping active A.
        let _c = manager.load(&dir_c).unwrap();
        assert_eq!(manager.len(), 2);
        assert_eq!(manager.active(), Some(&a));
        assert!(manager.get(&a).is_some());
    }

    #[test]
    fn oversized_adapter_is_rejected() {
        let (_root, directory) = write_adapter("adapterA", 2, 4, 6);
        let mut manager = LoraManager::with_budget(1);
        let error = manager.load(&directory).unwrap_err();
        assert!(matches!(error, LoraManagerError::OverBudget { .. }));
    }
}
