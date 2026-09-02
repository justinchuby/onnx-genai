//! Fixed-capacity, device-resident CSA state allocation.
//!
//! The Phase-B cache contract reserves stable addresses during kernel/runner
//! construction. Claim-time callers use [`CsaBufferLayout::from_claim`] only; it
//! performs the same checked static sizing without touching CUDA memory.

use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::{Node, Shape};

use crate::kernels::csa_state_group::{
    CsaStateGroupBytes, CsaStateGroupCharge, CsaStateGroupLedger,
};
use crate::runtime::CudaRuntime;

const ATTN_WIDTH: usize = 583;
const INDEX_WIDTH: usize = 68;
const DENSE_WIDTH: usize = 583;
const MAX_SEQUENCE_LEN: usize = 1 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CsaBufferLayout {
    pub batch: usize,
    pub max_seq_len: usize,
    pub window: usize,
    pub attention_r4_bytes: usize,
    pub attention_r4_carry_bytes: usize,
    pub attention_r128_bytes: usize,
    pub attention_r128_carry_bytes: usize,
    pub index_r4_bytes: usize,
    pub index_r4_carry_bytes: usize,
    pub dense_ring_bytes: usize,
}

impl CsaBufferLayout {
    pub(crate) fn from_claim(node: &Node, shapes: &[Shape], ratio: usize) -> Result<Option<Self>> {
        let Some(batch) = shapes
            .first()
            .and_then(|shape| shape.first())
            .and_then(|dim| dim.as_static())
        else {
            return Ok(None);
        };
        let sequence = shapes
            .first()
            .and_then(|shape| shape.get(1))
            .and_then(|dim| dim.as_static());
        let past_records = shapes
            .get(6)
            .and_then(|shape| shape.get(1))
            .and_then(|dim| dim.as_static());
        Self::from_values(node, batch, sequence, past_records, ratio).map(Some)
    }

    pub(crate) fn from_runner(
        node: &Node,
        input_shapes: &[Vec<usize>],
        ratio: usize,
    ) -> Result<Self> {
        let query = input_shapes
            .first()
            .ok_or_else(|| error("missing query shape"))?;
        let batch = *query
            .first()
            .ok_or_else(|| error("query batch axis is missing"))?;
        let sequence = *query
            .get(1)
            .ok_or_else(|| error("query sequence axis is missing"))?;
        let past_records = input_shapes.get(6).and_then(|shape| shape.get(1)).copied();
        Self::from_values(node, batch, Some(sequence), past_records, ratio)
    }

    fn from_values(
        node: &Node,
        batch: usize,
        sequence: Option<usize>,
        past_records: Option<usize>,
        ratio: usize,
    ) -> Result<Self> {
        let metadata_max =
            static_attr(node, "max_seq_len").or_else(|| static_attr(node, "max_sequence_length"));
        let inferred = match (past_records, sequence) {
            (Some(records), Some(sequence)) => records
                .checked_mul(ratio)
                .and_then(|v| v.checked_add(sequence)),
            _ => None,
        };
        let max_seq_len = metadata_max.or(inferred).ok_or_else(|| {
            error("max_seq_len metadata or static cache/query capacity is required")
        })?;
        if batch == 0 || max_seq_len == 0 || max_seq_len > MAX_SEQUENCE_LEN {
            return Err(error(format!(
                "batch={batch} and max_seq_len={max_seq_len} must be within supported fixed bounds"
            )));
        }
        let window = static_attr(node, "sliding_window")
            .or_else(|| static_attr(node, "window_size"))
            .unwrap_or(max_seq_len);
        if window == 0 || window > max_seq_len {
            return Err(error(format!(
                "dense window {window} must be in 1..={max_seq_len}"
            )));
        }
        let records4 = ceil_div(max_seq_len, 4)?;
        let records128 = ceil_div(max_seq_len, 128)?;
        Ok(Self {
            batch,
            max_seq_len,
            window,
            attention_r4_bytes: bytes(&[batch, records4, ATTN_WIDTH], 1)?,
            attention_r4_carry_bytes: bytes(&[batch, 8, 2, 1024], 4)?,
            attention_r128_bytes: bytes(&[batch, records128, ATTN_WIDTH], 1)?,
            attention_r128_carry_bytes: bytes(&[batch, 128, 2, 512], 4)?,
            index_r4_bytes: bytes(&[batch, records4, INDEX_WIDTH], 1)?,
            index_r4_carry_bytes: bytes(&[batch, 8, 2, 256], 4)?,
            dense_ring_bytes: bytes(&[batch, window, DENSE_WIDTH], 1)?,
        })
    }
}

/// Stable-address buffers reserved once for a CSA runner. They are intentionally
/// not read by B0: graph-threaded `past_* → present_*` remains authoritative.
pub(crate) struct CsaDeviceBufferManager {
    runtime: Arc<CudaRuntime>,
    pub(crate) layout: CsaBufferLayout,
    /// B6 pooled scratch (index transform / scores / selection / attention
    /// scores). Reserved once at runner init with stable addresses so the
    /// device-only capture path never allocates per call.
    workspaces: Vec<CUdeviceptr>,
    /// RAII accounting charge against the CSA state-group ledger, holding the
    /// governor [`MemoryLease`] for these bytes. Held for the manager's lifetime
    /// so residency is released from the one accounting authority
    /// (`MemoryGovernor::used(Tier::Device)`) exactly when the physical buffers
    /// are freed (`Drop`) or the reservation is rolled back. The ledger is
    /// mandatory: every CSA device-buffer reservation is charged to the shared
    /// governor, so there is no unaccounted path (B6/B6.2). Read only through
    /// its `Drop` in production; `charged_bytes` reads it under test.
    ///
    /// [`MemoryLease`]: onnx_runtime_memory_governor::MemoryLease
    #[allow(dead_code)]
    charge: CsaStateGroupCharge,
}

impl CsaDeviceBufferManager {
    pub(crate) fn reserve(
        runtime: Arc<CudaRuntime>,
        layout: CsaBufferLayout,
        workspace_bytes: &[usize],
        ledger: Arc<CsaStateGroupLedger>,
        charge_key: (u64, u32),
    ) -> Result<Self> {
        // Fail closed on the one accounting authority BEFORE touching CUDA: the
        // charge reserves these bytes from the shared `MemoryGovernor` (B6.2),
        // so an over-budget state group is refused with a typed reason and no
        // physical device memory is reserved — a rejected group can never leak,
        // and the reserved bytes are visible in the same device books as every
        // other holder. The ledger is always present (single accountant, no
        // unaccounted bypass); "disarmed" means the backing governor is
        // unlimited, so it never refuses and adds no device op, keeping the
        // reservation byte-identical. `charge` holds the governor lease in a
        // local so any early return below releases both the lease and the
        // attribution mirror via RAII (transaction rollback).
        let charge = ledger
            .try_charge(
                charge_key,
                CsaStateGroupBytes::workspace_only(workspace_bytes),
            )
            .map_err(EpError::from)?;
        let mut workspaces = Vec::with_capacity(workspace_bytes.len());
        let rollback = |workspaces: &mut Vec<CUdeviceptr>| {
            for ptr in workspaces.drain(..).rev() {
                // SAFETY: each pointer was allocated by this runtime and has not escaped.
                let _ = unsafe { runtime.free_raw(ptr) };
            }
        };
        for &size in workspace_bytes {
            match runtime.alloc_raw(size.max(1)) {
                Ok(ptr) => workspaces.push(ptr),
                Err(error) => {
                    rollback(&mut workspaces);
                    return Err(error);
                }
            }
        }
        Ok(Self {
            runtime,
            layout,
            workspaces,
            charge,
        })
    }

    /// Stable address of pooled workspace `index` (reserved in `reserve`).
    pub(crate) fn workspace(&self, index: usize) -> CUdeviceptr {
        self.workspaces[index]
    }

    /// Bytes this manager currently holds against the CSA state-group ledger.
    /// Lets a test assert `ledger.resident == sum(reserved)` and that teardown
    /// returns to baseline.
    #[cfg(test)]
    pub(crate) fn charged_bytes(&self) -> u64 {
        self.charge.total()
    }
}

impl Drop for CsaDeviceBufferManager {
    fn drop(&mut self) {
        for ptr in self.workspaces.drain(..).rev() {
            // SAFETY: this manager exclusively owns every pointer it reserved.
            let _ = unsafe { self.runtime.free_raw(ptr) };
        }
    }
}

fn static_attr(node: &Node, name: &str) -> Option<usize> {
    node.attr(name)
        .and_then(|attribute| attribute.as_int())
        .and_then(|value| usize::try_from(value).ok())
}
fn ceil_div(value: usize, divisor: usize) -> Result<usize> {
    value
        .checked_add(divisor - 1)
        .map(|v| v / divisor)
        .ok_or_else(|| error("CSA buffer capacity overflow"))
}
fn bytes(shape: &[usize], element_bytes: usize) -> Result<usize> {
    shape
        .iter()
        .try_fold(element_bytes, |n, &d| n.checked_mul(d))
        .ok_or_else(|| error("CSA buffer byte size overflow"))
}
fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!(
        "CompressedSparseAttention fixed-capacity state: {}",
        message.into()
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, Graph, Node, NodeId, static_shape};

    #[test]
    fn layout_uses_static_metadata_without_allocating() {
        let mut graph = Graph::new();
        let query = graph.create_named_value(
            "q",
            onnx_runtime_ir::DataType::Float32,
            static_shape([2, 1, 1, 512]),
        );
        let cache = graph.create_named_value(
            "cache",
            onnx_runtime_ir::DataType::Uint8,
            static_shape([2, 0, 583]),
        );
        let mut node = Node::new(
            NodeId(0),
            "CompressedSparseAttention",
            vec![Some(query), Some(cache)],
            vec![],
        );
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("max_seq_len".into(), Attribute::Int(1024));
        node.attributes
            .insert("sliding_window".into(), Attribute::Int(128));
        let layout = CsaBufferLayout::from_claim(
            &node,
            &[static_shape([2, 1, 1, 512]), static_shape([2, 0, 583])],
            4,
        )
        .unwrap()
        .unwrap();
        assert_eq!(layout.attention_r4_bytes, 2 * 256 * 583);
        assert_eq!(layout.index_r4_bytes, 2 * 256 * 68);
        assert_eq!(layout.dense_ring_bytes, 2 * 128 * 583);
    }

    /// A fixed, non-trivial layout for accounting tests (reuses the static
    /// sizing path, so it never touches CUDA).
    fn sample_layout() -> CsaBufferLayout {
        let mut graph = Graph::new();
        let query = graph.create_named_value(
            "q",
            onnx_runtime_ir::DataType::Float32,
            static_shape([2, 1, 1, 512]),
        );
        let cache = graph.create_named_value(
            "cache",
            onnx_runtime_ir::DataType::Uint8,
            static_shape([2, 0, 583]),
        );
        let mut node = Node::new(
            NodeId(0),
            "CompressedSparseAttention",
            vec![Some(query), Some(cache)],
            vec![],
        );
        node.domain = "pkg.nxrt".into();
        node.attributes
            .insert("max_seq_len".into(), Attribute::Int(1024));
        node.attributes
            .insert("sliding_window".into(), Attribute::Int(128));
        CsaBufferLayout::from_claim(
            &node,
            &[static_shape([2, 1, 1, 512]), static_shape([2, 0, 583])],
            4,
        )
        .unwrap()
        .unwrap()
    }

    /// The op-owned scratch reservation is charged exactly and returns to
    /// baseline on drop. Record/carry state is owned and governed by the native
    /// session bindings, so it is intentionally absent from this ledger.
    #[test]
    fn reserve_charges_ledger_and_releases_on_drop() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping CSA ledger reserve test: CUDA runtime unavailable");
            return;
        };
        let layout = sample_layout();
        let workspaces = [4096usize, 2048, 0];
        let expected = CsaStateGroupBytes::workspace_only(&workspaces).total();
        assert!(expected > 0, "sample layout must reserve some bytes");

        let ledger = Arc::new(CsaStateGroupLedger::default());
        assert_eq!(ledger.resident_bytes(), 0, "baseline before reserve");
        {
            let manager = CsaDeviceBufferManager::reserve(
                runtime.clone(),
                layout,
                &workspaces,
                Arc::clone(&ledger),
                (7, runtime.ordinal()),
            )
            .expect("reserve within unlimited ledger");
            assert_eq!(
                manager.charged_bytes(),
                expected,
                "manager holds the charge"
            );
            assert_eq!(
                ledger.resident_bytes(),
                expected,
                "ledger residency equals reserved bytes"
            );
            assert_eq!(ledger.peak_bytes(), expected);
            assert_eq!(ledger.active_group_count(), 1);
        }
        assert_eq!(ledger.resident_bytes(), 0, "teardown returns to baseline");
        assert_eq!(ledger.active_group_count(), 0);
        assert_eq!(
            ledger.peak_bytes(),
            expected,
            "peak is retained after release"
        );
    }

    /// Over-limit reservation fails closed with a typed OOM *before* any device
    /// allocation, and leaves the ledger untouched (no leaked residency).
    #[test]
    fn reserve_fails_closed_over_limit_without_leaking() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping CSA ledger OOM test: CUDA runtime unavailable");
            return;
        };
        let layout = sample_layout();
        let workspaces = [4096usize];
        let needed = CsaStateGroupBytes::workspace_only(&workspaces).total();

        let ledger = Arc::new(CsaStateGroupLedger::with_device_limit(needed - 1));
        let result = CsaDeviceBufferManager::reserve(
            runtime.clone(),
            layout,
            &workspaces,
            Arc::clone(&ledger),
            (1, runtime.ordinal()),
        )
        .map(|_| ());
        assert!(
            matches!(result, Err(EpError::OutOfMemory { .. })),
            "over-limit reservation must fail closed with OutOfMemory, got {result:?}"
        );
        assert_eq!(ledger.resident_bytes(), 0, "nothing charged on refusal");
        assert_eq!(ledger.active_group_count(), 0, "no group admitted");
        assert_eq!(ledger.charge_failures(), 1, "the refusal is counted");
        assert_eq!(
            ledger.governor_device_used(),
            0,
            "the governor's device books also stay at baseline on refusal"
        );
    }

    /// Two reservations under distinct `(request, device)` keys are isolated:
    /// each carries its own residency and dropping one does not disturb the
    /// other.
    #[test]
    fn reservations_are_isolated_per_request() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping CSA ledger isolation test: CUDA runtime unavailable");
            return;
        };
        let workspaces = [1024usize];
        let per = CsaStateGroupBytes::workspace_only(&workspaces).total();
        let ledger = Arc::new(CsaStateGroupLedger::default());
        let device = runtime.ordinal();

        let first = CsaDeviceBufferManager::reserve(
            runtime.clone(),
            sample_layout(),
            &workspaces,
            Arc::clone(&ledger),
            (1, device),
        )
        .expect("first reserve");
        let second = CsaDeviceBufferManager::reserve(
            runtime.clone(),
            sample_layout(),
            &workspaces,
            Arc::clone(&ledger),
            (2, device),
        )
        .expect("second reserve");

        assert_eq!(ledger.active_group_count(), 2);
        assert_eq!(ledger.resident_for(1, device), per);
        assert_eq!(ledger.resident_for(2, device), per);
        assert_eq!(ledger.resident_bytes(), per * 2);

        drop(first);
        assert_eq!(ledger.resident_for(1, device), 0, "first released");
        assert_eq!(
            ledger.resident_for(2, device),
            per,
            "second unaffected by first's teardown"
        );
        drop(second);
        assert_eq!(ledger.resident_bytes(), 0, "both released to baseline");
    }

    /// Reserved workspace addresses are stable for the manager's lifetime, as
    /// the capture path requires (no per-call reallocation, no VA churn).
    #[test]
    fn reserved_workspace_addresses_are_stable() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping CSA workspace stability test: CUDA runtime unavailable");
            return;
        };
        let workspaces = [2048usize, 4096, 8192];
        let manager = CsaDeviceBufferManager::reserve(
            runtime.clone(),
            sample_layout(),
            &workspaces,
            Arc::new(CsaStateGroupLedger::default()),
            (3, runtime.ordinal()),
        )
        .expect("reserve");
        for i in 0..workspaces.len() {
            assert_eq!(
                manager.workspace(i),
                manager.workspace(i),
                "workspace {i} address must be stable across reads"
            );
        }
        assert_ne!(
            manager.workspace(0),
            manager.workspace(1),
            "distinct workspaces occupy distinct addresses"
        );
    }

    /// An unlimited (default) ledger accounts the reservation but never refuses,
    /// so the reservation is byte-identical to the pre-accounting path — the
    /// ledger is always present (no unaccounted bypass), and "disarmed" is a
    /// limit setting, not a structural escape hatch.
    #[test]
    fn unlimited_ledger_accounts_without_refusing() {
        let Some(runtime) = crate::test_support::maybe_runtime() else {
            eprintln!("skipping CSA unlimited-ledger test: CUDA runtime unavailable");
            return;
        };
        let workspaces = [1024usize];
        let expected = CsaStateGroupBytes::workspace_only(&workspaces).total();
        let ledger = Arc::new(CsaStateGroupLedger::default());
        assert_eq!(
            ledger.device_available_bytes(),
            u64::MAX,
            "default ledger's backing governor is unlimited"
        );
        let manager = CsaDeviceBufferManager::reserve(
            runtime.clone(),
            sample_layout(),
            &workspaces,
            Arc::clone(&ledger),
            (0, runtime.ordinal()),
        )
        .expect("unlimited ledger never refuses");
        assert_eq!(
            manager.charged_bytes(),
            expected,
            "reservation is accounted"
        );
        assert_eq!(
            ledger.governor_device_used(),
            expected,
            "the physical reservation is visible in the one governor's device books"
        );
        assert_eq!(
            ledger.charge_failures(),
            0,
            "no refusal on the disarmed path"
        );
        drop(manager);
        assert_eq!(ledger.resident_bytes(), 0, "teardown returns to baseline");
        assert_eq!(
            ledger.governor_device_used(),
            0,
            "teardown also returns the governor's device books to baseline"
        );
    }
}
