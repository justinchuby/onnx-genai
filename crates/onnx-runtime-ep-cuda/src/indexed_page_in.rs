//! Fused indexed page-in for routed MoE projection banks.
//!
//! FreeToken's relevant semantic observation is narrow: one cache-miss index
//! list applies to every packed projection bank and its auxiliary-scale bank,
//! so the physical copies can be described once and submitted together. This
//! module adopts that scheduling idea without adopting FreeToken's cache,
//! allocator, or accounting implementation.
//!
//! The destination addresses are existing PMM/VMM-owned stable mappings. This
//! module never allocates, maps, evicts, remaps, or publishes residency. It only
//! builds a deterministic descriptor batch and submits it at an already-proven
//! coarse boundary. Payload bytes are invariant: fusion reduces driver
//! submissions, not bytes moved.

use std::collections::{BTreeMap, HashSet};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{EpError, Result};

use crate::runtime::{CudaBatchCopy, CudaRuntime, FailedHtodCompletion};

/// Execution phase retained on every completion record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum IndexedPageInPhase {
    Prefill,
    Decode,
}

/// Existing layer/phase attribution for one page-in request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexedPageInAttribution {
    pub layer: u32,
    pub phase: IndexedPageInPhase,
}

/// One expert-major bank already mapped into CUDA's unified address space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedExpertBank {
    pub source_base: CUdeviceptr,
    pub destination_base: CUdeviceptr,
    pub source_expert_stride: usize,
    pub destination_slot_stride: usize,
    pub bytes_per_expert: usize,
    pub experts: usize,
    pub slots: usize,
}

/// A projection's packed bank and mandatory auxiliary-scale bank.
///
/// Pairing is structural: callers cannot submit a packed planar projection
/// without naming the scale bank consumed by the same projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionBankPair {
    pub projection: u32,
    pub packed: IndexedExpertBank,
    pub auxiliary_scale: IndexedExpertBank,
}

/// One normalized expert-cache lookup.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertSlot {
    pub expert: usize,
    pub slot: usize,
    pub resident: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedBankKind {
    Packed,
    AuxiliaryScale,
}

/// Test/telemetry-visible descriptor identity. CUDA may execute entries in any
/// order; destinations are disjoint, so only descriptor order is meaningful.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedCopyIdentity {
    pub expert: usize,
    pub slot: usize,
    pub projection: u32,
    pub bank: IndexedBankKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PlannedCopy {
    identity: IndexedCopyIdentity,
    copy: CudaBatchCopy,
}

/// Fully validated, immutable submission plan.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedMultiBankPageInPlan {
    attribution: IndexedPageInAttribution,
    selected: usize,
    hits: usize,
    misses: usize,
    payload_bytes: u64,
    copies: Vec<PlannedCopy>,
}

impl IndexedMultiBankPageInPlan {
    pub fn build(
        attribution: IndexedPageInAttribution,
        selections: &[ExpertSlot],
        projections: &[ProjectionBankPair],
    ) -> Result<Self> {
        if selections.is_empty() {
            return Err(EpError::KernelFailed(
                "indexed expert page-in requires a non-empty selected set".into(),
            ));
        }
        if projections.is_empty() {
            return Err(EpError::KernelFailed(
                "indexed expert page-in requires at least one packed/aux-scale pair".into(),
            ));
        }

        let mut projection_ids = HashSet::new();
        for pair in projections {
            if !projection_ids.insert(pair.projection) {
                return Err(EpError::KernelFailed(format!(
                    "indexed expert page-in repeats projection {}",
                    pair.projection
                )));
            }
            validate_pair(pair)?;
        }

        let mut experts = HashSet::new();
        let mut slots = HashSet::new();
        for selection in selections {
            if !experts.insert(selection.expert) {
                return Err(EpError::KernelFailed(format!(
                    "indexed expert page-in repeats expert {}",
                    selection.expert
                )));
            }
            if !slots.insert(selection.slot) {
                return Err(EpError::KernelFailed(format!(
                    "indexed expert page-in repeats destination slot {}",
                    selection.slot
                )));
            }
            for pair in projections {
                if selection.expert >= pair.packed.experts {
                    return Err(EpError::KernelFailed(format!(
                        "expert {} is outside projection {} expert domain {}",
                        selection.expert, pair.projection, pair.packed.experts
                    )));
                }
                if selection.slot >= pair.packed.slots {
                    return Err(EpError::KernelFailed(format!(
                        "slot {} is outside projection {} cache domain {}",
                        selection.slot, pair.projection, pair.packed.slots
                    )));
                }
            }
        }

        let hits = selections
            .iter()
            .filter(|selection| selection.resident)
            .count();
        let misses = selections.len() - hits;
        let mut payload_bytes = 0u64;
        let mut copies = Vec::with_capacity(
            misses
                .checked_mul(projections.len())
                .and_then(|value| value.checked_mul(2))
                .ok_or_else(|| {
                    EpError::KernelFailed("indexed page-in descriptor count overflow".into())
                })?,
        );
        // Deterministic order: selected miss order, projection registration
        // order, then packed immediately followed by its auxiliary scale.
        for selection in selections.iter().filter(|selection| !selection.resident) {
            for pair in projections {
                for (bank_kind, bank) in [
                    (IndexedBankKind::Packed, pair.packed),
                    (IndexedBankKind::AuxiliaryScale, pair.auxiliary_scale),
                ] {
                    let src_offset = selection
                        .expert
                        .checked_mul(bank.source_expert_stride)
                        .ok_or_else(|| {
                            EpError::KernelFailed("indexed source offset overflow".into())
                        })?;
                    let dst_offset = selection
                        .slot
                        .checked_mul(bank.destination_slot_stride)
                        .ok_or_else(|| {
                            EpError::KernelFailed("indexed destination offset overflow".into())
                        })?;
                    let src = bank
                        .source_base
                        .checked_add(src_offset as u64)
                        .ok_or_else(|| {
                            EpError::KernelFailed("indexed source pointer overflow".into())
                        })?;
                    let dst = bank
                        .destination_base
                        .checked_add(dst_offset as u64)
                        .ok_or_else(|| {
                            EpError::KernelFailed("indexed destination pointer overflow".into())
                        })?;
                    payload_bytes = payload_bytes
                        .checked_add(bank.bytes_per_expert as u64)
                        .ok_or_else(|| {
                            EpError::KernelFailed("indexed page-in payload overflow".into())
                        })?;
                    copies.push(PlannedCopy {
                        identity: IndexedCopyIdentity {
                            expert: selection.expert,
                            slot: selection.slot,
                            projection: pair.projection,
                            bank: bank_kind,
                        },
                        copy: CudaBatchCopy {
                            src,
                            dst,
                            bytes: bank.bytes_per_expert,
                        },
                    });
                }
            }
        }

        Ok(Self {
            attribution,
            selected: selections.len(),
            hits,
            misses,
            payload_bytes,
            copies,
        })
    }

    pub fn payload_bytes(&self) -> u64 {
        self.payload_bytes
    }

    pub fn hits(&self) -> usize {
        self.hits
    }

    pub fn misses(&self) -> usize {
        self.misses
    }

    pub fn copy_identities(&self) -> Vec<IndexedCopyIdentity> {
        self.copies.iter().map(|copy| copy.identity).collect()
    }
}

fn validate_bank(projection: u32, name: &str, bank: IndexedExpertBank) -> Result<()> {
    if bank.source_base == 0 || bank.destination_base == 0 {
        return Err(EpError::KernelFailed(format!(
            "projection {projection} {name} bank has a null CUDA address"
        )));
    }
    if bank.bytes_per_expert == 0
        || bank.source_expert_stride < bank.bytes_per_expert
        || bank.destination_slot_stride < bank.bytes_per_expert
        || bank.experts == 0
        || bank.slots == 0
    {
        return Err(EpError::KernelFailed(format!(
            "projection {projection} {name} bank has invalid expert/slot geometry"
        )));
    }
    Ok(())
}

fn validate_pair(pair: &ProjectionBankPair) -> Result<()> {
    validate_bank(pair.projection, "packed", pair.packed)?;
    validate_bank(pair.projection, "auxiliary-scale", pair.auxiliary_scale)?;
    if pair.packed.experts != pair.auxiliary_scale.experts
        || pair.packed.slots != pair.auxiliary_scale.slots
    {
        return Err(EpError::KernelFailed(format!(
            "projection {} packed/aux-scale bank domains disagree",
            pair.projection
        )));
    }
    Ok(())
}

trait BatchSubmitter {
    fn submit(&self, copies: &[CudaBatchCopy]) -> std::result::Result<Duration, String>;
}

impl BatchSubmitter for CudaRuntime {
    fn submit(&self, copies: &[CudaBatchCopy]) -> std::result::Result<Duration, String> {
        // SAFETY: plan validation checked pointer arithmetic and disjoint slot
        // ownership; the caller owns the mapped bank lifetimes for this coarse
        // boundary. The runtime host-synchronizes before returning.
        match unsafe { self.indexed_batch_copy_elapsed_ms(copies) } {
            Ok((elapsed_ms, _completed)) => {
                Ok(Duration::from_secs_f64(f64::from(elapsed_ms) / 1_000.0))
            }
            Err(error) => {
                let (detail, completion) = error.into_parts();
                let suffix = match completion {
                    FailedHtodCompletion::NotSubmitted => "not submitted",
                    FailedHtodCompletion::Completed(_) => "submitted and completion established",
                    FailedHtodCompletion::MayBeInFlight => "submitted with unresolved completion",
                };
                Err(format!("{detail}; {suffix}"))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IndexedPageInAttributionStats {
    pub batches: u64,
    pub selected: u64,
    pub hits: u64,
    pub misses: u64,
    pub copy_entries: u64,
    pub payload_bytes: u64,
    pub failures: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexedPageInReceipt {
    pub attribution: IndexedPageInAttribution,
    pub selected: usize,
    pub hits: usize,
    pub misses: usize,
    pub copy_entries: usize,
    pub payload_bytes: u64,
    pub elapsed: Duration,
}

fn attribution_stats()
-> &'static Mutex<BTreeMap<IndexedPageInAttribution, IndexedPageInAttributionStats>> {
    static STATS: OnceLock<
        Mutex<BTreeMap<IndexedPageInAttribution, IndexedPageInAttributionStats>>,
    > = OnceLock::new();
    STATS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

pub fn indexed_page_in_attribution_stats()
-> Vec<(IndexedPageInAttribution, IndexedPageInAttributionStats)> {
    attribution_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .map(|(key, value)| (*key, *value))
        .collect()
}

pub(crate) fn reset_indexed_page_in_attribution_stats() {
    attribution_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clear();
}

/// Execute one validated fused page-in batch.
///
/// Hits produce no copy and an all-hit request produces no CUDA submission.
/// For misses, physical H2D/page-in accounting is committed only after the
/// batch completion witness is returned. A failed or partially executed batch
/// records only `failures`; it never records payload or successful page-ins.
pub fn execute_indexed_multi_bank_page_in(
    runtime: &CudaRuntime,
    plan: &IndexedMultiBankPageInPlan,
) -> Result<IndexedPageInReceipt> {
    execute_with(runtime, plan)
}

fn execute_with(
    submitter: &dyn BatchSubmitter,
    plan: &IndexedMultiBankPageInPlan,
) -> Result<IndexedPageInReceipt> {
    let copies: Vec<CudaBatchCopy> = plan.copies.iter().map(|entry| entry.copy).collect();
    let elapsed = if copies.is_empty() {
        Duration::ZERO
    } else {
        match submitter.submit(&copies) {
            Ok(elapsed) => elapsed,
            Err(error) => {
                let mut stats = attribution_stats()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                stats.entry(plan.attribution).or_default().failures += 1;
                return Err(EpError::KernelFailed(format!(
                    "indexed multi-bank expert page-in failed: {error}"
                )));
            }
        }
    };

    if !copies.is_empty() {
        crate::weight_paging::record_indexed_page_in_completion(
            plan.payload_bytes,
            copies.len() as u64,
            plan.misses as u64,
            elapsed,
        );
    }
    let mut stats = attribution_stats()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let row = stats.entry(plan.attribution).or_default();
    row.batches += u64::from(!copies.is_empty());
    row.selected += plan.selected as u64;
    row.hits += plan.hits as u64;
    row.misses += plan.misses as u64;
    row.copy_entries += copies.len() as u64;
    row.payload_bytes += plan.payload_bytes;

    Ok(IndexedPageInReceipt {
        attribution: plan.attribution,
        selected: plan.selected,
        hits: plan.hits,
        misses: plan.misses,
        copy_entries: copies.len(),
        payload_bytes: plan.payload_bytes,
        elapsed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn stats_test_lock() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn bank(src: u64, dst: u64, bytes: usize) -> IndexedExpertBank {
        IndexedExpertBank {
            source_base: src,
            destination_base: dst,
            source_expert_stride: bytes,
            destination_slot_stride: bytes,
            bytes_per_expert: bytes,
            experts: 8,
            slots: 8,
        }
    }

    fn fixture() -> IndexedMultiBankPageInPlan {
        IndexedMultiBankPageInPlan::build(
            IndexedPageInAttribution {
                layer: 7,
                phase: IndexedPageInPhase::Decode,
            },
            &[
                ExpertSlot {
                    expert: 4,
                    slot: 1,
                    resident: false,
                },
                ExpertSlot {
                    expert: 2,
                    slot: 5,
                    resident: true,
                },
                ExpertSlot {
                    expert: 6,
                    slot: 3,
                    resident: false,
                },
            ],
            &[
                ProjectionBankPair {
                    projection: 10,
                    packed: bank(0x1000, 0x10_0000, 64),
                    auxiliary_scale: bank(0x2000, 0x20_0000, 16),
                },
                ProjectionBankPair {
                    projection: 20,
                    packed: bank(0x3000, 0x30_0000, 96),
                    auxiliary_scale: bank(0x4000, 0x40_0000, 24),
                },
            ],
        )
        .unwrap()
    }

    #[test]
    fn hit_miss_fixture_pairs_banks_and_preserves_descriptor_order() {
        let plan = fixture();
        assert_eq!((plan.hits(), plan.misses()), (1, 2));
        assert_eq!(plan.payload_bytes(), 2 * (64 + 16 + 96 + 24));
        assert_eq!(
            plan.copy_identities(),
            vec![
                IndexedCopyIdentity {
                    expert: 4,
                    slot: 1,
                    projection: 10,
                    bank: IndexedBankKind::Packed,
                },
                IndexedCopyIdentity {
                    expert: 4,
                    slot: 1,
                    projection: 10,
                    bank: IndexedBankKind::AuxiliaryScale,
                },
                IndexedCopyIdentity {
                    expert: 4,
                    slot: 1,
                    projection: 20,
                    bank: IndexedBankKind::Packed,
                },
                IndexedCopyIdentity {
                    expert: 4,
                    slot: 1,
                    projection: 20,
                    bank: IndexedBankKind::AuxiliaryScale,
                },
                IndexedCopyIdentity {
                    expert: 6,
                    slot: 3,
                    projection: 10,
                    bank: IndexedBankKind::Packed,
                },
                IndexedCopyIdentity {
                    expert: 6,
                    slot: 3,
                    projection: 10,
                    bank: IndexedBankKind::AuxiliaryScale,
                },
                IndexedCopyIdentity {
                    expert: 6,
                    slot: 3,
                    projection: 20,
                    bank: IndexedBankKind::Packed,
                },
                IndexedCopyIdentity {
                    expert: 6,
                    slot: 3,
                    projection: 20,
                    bank: IndexedBankKind::AuxiliaryScale,
                },
            ]
        );
    }

    struct PartialFailure {
        observed: AtomicUsize,
    }

    impl BatchSubmitter for PartialFailure {
        fn submit(&self, copies: &[CudaBatchCopy]) -> std::result::Result<Duration, String> {
            self.observed.store(3.min(copies.len()), Ordering::Relaxed);
            Err("injected failure after three device writes".into())
        }
    }

    #[test]
    fn partial_copy_failure_commits_no_payload_or_success() {
        let _guard = stats_test_lock();
        crate::weight_paging::reset_global_offload_stats();
        reset_indexed_page_in_attribution_stats();
        let before = crate::weight_paging::global_offload_stats();
        let submitter = PartialFailure {
            observed: AtomicUsize::new(0),
        };
        let error = execute_with(&submitter, &fixture()).unwrap_err();
        assert!(error.to_string().contains("injected failure"));
        assert_eq!(submitter.observed.load(Ordering::Relaxed), 3);
        let after = crate::weight_paging::global_offload_stats();
        assert_eq!(after.htod_bytes, before.htod_bytes);
        assert_eq!(
            after.indexed_expert_page_ins,
            before.indexed_expert_page_ins
        );
        let rows = indexed_page_in_attribution_stats();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].1.failures, 1);
        assert_eq!(rows[0].1.payload_bytes, 0);
        assert_eq!(rows[0].1.batches, 0);
    }

    struct Success;

    impl BatchSubmitter for Success {
        fn submit(&self, _copies: &[CudaBatchCopy]) -> std::result::Result<Duration, String> {
            Ok(Duration::from_micros(25))
        }
    }

    #[test]
    fn completion_updates_existing_physical_byte_authority_once() {
        let _guard = stats_test_lock();
        crate::weight_paging::reset_global_offload_stats();
        reset_indexed_page_in_attribution_stats();
        let plan = fixture();
        let receipt = execute_with(&Success, &plan).unwrap();
        let stats = crate::weight_paging::global_offload_stats();
        assert_eq!(receipt.payload_bytes, plan.payload_bytes());
        assert_eq!(stats.htod_bytes, plan.payload_bytes());
        assert_eq!(stats.indexed_page_in_batches, 1);
        assert_eq!(stats.indexed_page_in_entries, 8);
        assert_eq!(stats.indexed_expert_page_ins, 2);
        let rows = indexed_page_in_attribution_stats();
        assert_eq!(rows[0].0, plan.attribution);
        assert_eq!(rows[0].1.payload_bytes, plan.payload_bytes());
    }
}
