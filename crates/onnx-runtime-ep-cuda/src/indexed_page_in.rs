//! Transactional fused indexed page-in for routed MoE projection banks.
//!
//! One host-known miss list applies to every packed projection bank and its
//! mandatory auxiliary-scale bank. Fusion changes only driver submission
//! count: every packed and scale byte still moves.
//!
//! There is intentionally no live kernel caller on current `main`: QMoE routing
//! is device-fused, so the host cannot name the current step's experts before
//! dispatch, and the public #2082 readiness branch still reports no dedicated
//! per-bank VMM reservation. The authority-only API below therefore fails
//! closed unless a caller supplies both an explicit completion boundary and
//! sealed PMM/VMM bank ownership; it must not be presented as production-active
//! until those dependencies land.

use std::any::Any;
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{EpError, Result};

use crate::runtime::{CudaBatchCopy, CudaRuntime, FailedHtodCompletion};
use crate::weight_paging::{
    CudaWeightResidency, IndexedPageInBoundaryWitness, quarantine_indexed_page_in,
};

pub type IndexedBankOwner = Arc<dyn Any + Send + Sync>;

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

/// Raw geometry sealed into an authority-owned expert bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedExpertBankSpec {
    pub source_base: CUdeviceptr,
    pub destination_base: CUdeviceptr,
    pub source_expert_stride: usize,
    pub destination_slot_stride: usize,
    pub bytes_per_expert: usize,
    pub experts: usize,
    pub slots: usize,
}

/// Visibility state for a destination bank.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedBankVisibility {
    Ready,
    Updating,
    Poisoned,
}

#[derive(Debug)]
struct DestinationAuthority {
    visibility: Mutex<IndexedBankVisibility>,
}

/// One expert-major bank whose source and destination ownership is retained by
/// the PMM/VMM residency authority.
#[derive(Clone)]
pub struct IndexedExpertBank {
    spec: IndexedExpertBankSpec,
    authority_id: u64,
    source_owner: IndexedBankOwner,
    destination_owner: IndexedBankOwner,
    destination: Arc<DestinationAuthority>,
}

impl fmt::Debug for IndexedExpertBank {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("IndexedExpertBank")
            .field("spec", &self.spec)
            .field("authority_id", &self.authority_id)
            .field("visibility", &self.visibility())
            .finish_non_exhaustive()
    }
}

impl IndexedExpertBank {
    fn new(
        spec: IndexedExpertBankSpec,
        authority_id: u64,
        source_owner: IndexedBankOwner,
        destination_owner: IndexedBankOwner,
    ) -> Result<Self> {
        validate_bank_spec(spec)?;
        Ok(Self {
            spec,
            authority_id,
            source_owner,
            destination_owner,
            destination: Arc::new(DestinationAuthority {
                visibility: Mutex::new(IndexedBankVisibility::Ready),
            }),
        })
    }

    pub fn visibility(&self) -> IndexedBankVisibility {
        *self
            .destination
            .visibility
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    #[cfg(test)]
    fn test(spec: IndexedExpertBankSpec) -> Self {
        Self::new(spec, 1, Arc::new(()), Arc::new(())).unwrap()
    }
}

impl CudaWeightResidency {
    /// Seal raw bank geometry to exact source/destination lifetime owners and
    /// this residency authority.
    ///
    /// # Safety
    /// `source_owner` must keep every source interval in `spec` live and
    /// immutable. `destination_owner` must keep every destination interval live
    /// and exclusively unpublished while its bank is `Updating`; consumers must
    /// reject `Poisoned`. Both address spaces must belong to this residency's
    /// CUDA context.
    pub unsafe fn seal_indexed_expert_bank(
        self: &Arc<Self>,
        spec: IndexedExpertBankSpec,
        source_owner: IndexedBankOwner,
        destination_owner: IndexedBankOwner,
    ) -> Result<IndexedExpertBank> {
        IndexedExpertBank::new(
            spec,
            self.indexed_authority_id(),
            source_owner,
            destination_owner,
        )
    }
}

/// A projection's packed bank and mandatory auxiliary-scale bank.
#[derive(Clone, Debug)]
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

/// Test/telemetry-visible descriptor identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexedCopyIdentity {
    pub expert: usize,
    pub slot: usize,
    pub projection: u32,
    pub bank: IndexedBankKind,
}

#[derive(Clone, Debug)]
struct PlannedCopy {
    identity: IndexedCopyIdentity,
    copy: CudaBatchCopy,
    bank: IndexedExpertBank,
}

/// Fully validated, immutable submission plan.
#[derive(Clone, Debug)]
pub struct IndexedMultiBankPageInPlan {
    authority_id: u64,
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

        let authority_id = projections[0].packed.authority_id;
        let mut projection_ids = HashSet::new();
        for pair in projections {
            if !projection_ids.insert(pair.projection) {
                return Err(EpError::KernelFailed(format!(
                    "indexed expert page-in repeats projection {}",
                    pair.projection
                )));
            }
            validate_pair(pair, authority_id)?;
        }
        validate_all_destination_intervals(projections)?;

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
                if selection.expert >= pair.packed.spec.experts {
                    return Err(EpError::KernelFailed(format!(
                        "expert {} is outside projection {} expert domain {}",
                        selection.expert, pair.projection, pair.packed.spec.experts
                    )));
                }
                if selection.slot >= pair.packed.spec.slots {
                    return Err(EpError::KernelFailed(format!(
                        "slot {} is outside projection {} cache domain {}",
                        selection.slot, pair.projection, pair.packed.spec.slots
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
        for selection in selections.iter().filter(|selection| !selection.resident) {
            for pair in projections {
                for (bank_kind, bank) in [
                    (IndexedBankKind::Packed, &pair.packed),
                    (IndexedBankKind::AuxiliaryScale, &pair.auxiliary_scale),
                ] {
                    let src = indexed_address(
                        bank.spec.source_base,
                        selection.expert,
                        bank.spec.source_expert_stride,
                        bank.spec.bytes_per_expert,
                        "source",
                    )?;
                    let dst = indexed_address(
                        bank.spec.destination_base,
                        selection.slot,
                        bank.spec.destination_slot_stride,
                        bank.spec.bytes_per_expert,
                        "destination",
                    )?;
                    payload_bytes = payload_bytes
                        .checked_add(u64::try_from(bank.spec.bytes_per_expert).map_err(|_| {
                            EpError::KernelFailed(
                                "indexed page-in payload entry does not fit u64".into(),
                            )
                        })?)
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
                            bytes: bank.spec.bytes_per_expert,
                        },
                        bank: bank.clone(),
                    });
                }
            }
        }

        Ok(Self {
            authority_id,
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

fn validate_bank_spec(bank: IndexedExpertBankSpec) -> Result<()> {
    if bank.source_base == 0 || bank.destination_base == 0 {
        return Err(EpError::KernelFailed(
            "indexed expert bank has a null CUDA address".into(),
        ));
    }
    if bank.bytes_per_expert == 0
        || bank.source_expert_stride < bank.bytes_per_expert
        || bank.destination_slot_stride < bank.bytes_per_expert
        || bank.experts == 0
        || bank.slots == 0
    {
        return Err(EpError::KernelFailed(
            "indexed expert bank has invalid expert/slot geometry".into(),
        ));
    }
    indexed_address(
        bank.source_base,
        bank.experts - 1,
        bank.source_expert_stride,
        bank.bytes_per_expert,
        "source",
    )?;
    indexed_address(
        bank.destination_base,
        bank.slots - 1,
        bank.destination_slot_stride,
        bank.bytes_per_expert,
        "destination",
    )?;
    Ok(())
}

fn indexed_address(
    base: CUdeviceptr,
    index: usize,
    stride: usize,
    bytes: usize,
    kind: &str,
) -> Result<CUdeviceptr> {
    let offset = index
        .checked_mul(stride)
        .ok_or_else(|| EpError::KernelFailed(format!("indexed {kind} offset overflow")))?;
    let offset = u64::try_from(offset)
        .map_err(|_| EpError::KernelFailed(format!("indexed {kind} offset does not fit u64")))?;
    let start = base
        .checked_add(offset)
        .ok_or_else(|| EpError::KernelFailed(format!("indexed {kind} pointer overflow")))?;
    let bytes = u64::try_from(bytes)
        .map_err(|_| EpError::KernelFailed(format!("indexed {kind} length does not fit u64")))?;
    start
        .checked_add(bytes)
        .ok_or_else(|| EpError::KernelFailed(format!("indexed {kind} interval overflow")))?;
    Ok(start)
}

fn validate_pair(pair: &ProjectionBankPair, authority_id: u64) -> Result<()> {
    for (name, bank) in [
        ("packed", &pair.packed),
        ("auxiliary-scale", &pair.auxiliary_scale),
    ] {
        validate_bank_spec(bank.spec)?;
        if bank.authority_id != authority_id {
            return Err(EpError::KernelFailed(format!(
                "projection {} {name} bank belongs to a different residency authority",
                pair.projection
            )));
        }
        if bank.visibility() != IndexedBankVisibility::Ready {
            return Err(EpError::KernelFailed(format!(
                "projection {} {name} destination is not ready",
                pair.projection
            )));
        }
    }
    if pair.packed.spec.experts != pair.auxiliary_scale.spec.experts
        || pair.packed.spec.slots != pair.auxiliary_scale.spec.slots
    {
        return Err(EpError::KernelFailed(format!(
            "projection {} packed/aux-scale bank domains disagree",
            pair.projection
        )));
    }
    Ok(())
}

fn validate_all_destination_intervals(projections: &[ProjectionBankPair]) -> Result<()> {
    let interval_count = projections
        .iter()
        .try_fold(0usize, |count, pair| {
            count
                .checked_add(pair.packed.spec.slots)
                .and_then(|count| count.checked_add(pair.auxiliary_scale.spec.slots))
        })
        .ok_or_else(|| {
            EpError::KernelFailed("indexed destination interval count overflow".into())
        })?;
    let mut intervals = Vec::with_capacity(interval_count);
    for pair in projections {
        for (kind, bank) in [
            (IndexedBankKind::Packed, &pair.packed),
            (IndexedBankKind::AuxiliaryScale, &pair.auxiliary_scale),
        ] {
            for slot in 0..bank.spec.slots {
                let start = indexed_address(
                    bank.spec.destination_base,
                    slot,
                    bank.spec.destination_slot_stride,
                    bank.spec.bytes_per_expert,
                    "destination",
                )?;
                let end = start
                    .checked_add(u64::try_from(bank.spec.bytes_per_expert).map_err(|_| {
                        EpError::KernelFailed("indexed destination length does not fit u64".into())
                    })?)
                    .ok_or_else(|| {
                        EpError::KernelFailed("indexed destination interval overflow".into())
                    })?;
                intervals.push((start, end, pair.projection, kind, slot));
            }
        }
    }
    intervals.sort_unstable_by_key(|interval| interval.0);
    for window in intervals.windows(2) {
        let left = window[0];
        let right = window[1];
        if right.0 < left.1 {
            return Err(EpError::KernelFailed(format!(
                "indexed page-in destination intervals overlap: projection {} {:?} slot {} \
                 [{:#x}, {:#x}) with projection {} {:?} slot {} [{:#x}, {:#x})",
                left.2, left.3, left.4, left.0, left.1, right.2, right.3, right.4, right.0, right.1
            )));
        }
    }
    Ok(())
}

struct IndexedSubmitFailure {
    detail: String,
    completion: FailedHtodCompletion,
}

trait BatchSubmitter {
    fn submit(
        &self,
        copies: &[CudaBatchCopy],
    ) -> std::result::Result<Duration, IndexedSubmitFailure>;
}

impl BatchSubmitter for CudaRuntime {
    fn submit(
        &self,
        copies: &[CudaBatchCopy],
    ) -> std::result::Result<Duration, IndexedSubmitFailure> {
        // SAFETY: the sealed plan retained exact endpoint owners and validated
        // every destination interval as pairwise disjoint.
        match unsafe { self.indexed_batch_copy_elapsed_ms(copies) } {
            Ok((elapsed_ms, _completed)) => {
                Ok(Duration::from_secs_f64(f64::from(elapsed_ms) / 1_000.0))
            }
            Err(error) => {
                let (detail, completion) = error.into_parts();
                Err(IndexedSubmitFailure { detail, completion })
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IndexedPageInFailureDisposition {
    NotSubmitted,
    QuarantinedCompleted,
    QuarantinedMayBeInFlight,
}

#[derive(Debug)]
pub struct IndexedPageInError {
    detail: String,
    completion: FailedHtodCompletion,
    disposition: IndexedPageInFailureDisposition,
}

impl IndexedPageInError {
    fn not_submitted(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
            completion: FailedHtodCompletion::NotSubmitted,
            disposition: IndexedPageInFailureDisposition::NotSubmitted,
        }
    }

    pub fn completion(&self) -> &FailedHtodCompletion {
        &self.completion
    }

    pub fn disposition(&self) -> IndexedPageInFailureDisposition {
        self.disposition
    }
}

impl fmt::Display for IndexedPageInError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "indexed multi-bank expert page-in failed: {} ({:?})",
            self.detail, self.disposition
        )
    }
}

impl std::error::Error for IndexedPageInError {}

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

struct DestinationTransaction {
    destinations: Vec<Arc<DestinationAuthority>>,
}

impl DestinationTransaction {
    fn begin(plan: &IndexedMultiBankPageInPlan) -> std::result::Result<Self, IndexedPageInError> {
        let mut seen = HashSet::new();
        let mut destinations = Vec::new();
        for entry in &plan.copies {
            let identity = Arc::as_ptr(&entry.bank.destination) as usize;
            if seen.insert(identity) {
                destinations.push(Arc::clone(&entry.bank.destination));
            }
        }
        for (index, destination) in destinations.iter().enumerate() {
            let mut visibility = destination
                .visibility
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if *visibility != IndexedBankVisibility::Ready {
                drop(visibility);
                for prior in &destinations[..index] {
                    *prior
                        .visibility
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()) =
                        IndexedBankVisibility::Ready;
                }
                return Err(IndexedPageInError::not_submitted(
                    "a destination bank is already updating or poisoned",
                ));
            }
            *visibility = IndexedBankVisibility::Updating;
        }
        Ok(Self { destinations })
    }

    fn ready(self) {
        for destination in self.destinations {
            *destination
                .visibility
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = IndexedBankVisibility::Ready;
        }
    }

    fn poison(self) {
        for destination in self.destinations {
            *destination
                .visibility
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = IndexedBankVisibility::Poisoned;
        }
    }
}

/// Execute one fused page-in through the residency authority.
pub fn execute_indexed_multi_bank_page_in(
    residency: &CudaWeightResidency,
    boundary: &IndexedPageInBoundaryWitness,
    plan: &IndexedMultiBankPageInPlan,
) -> std::result::Result<IndexedPageInReceipt, IndexedPageInError> {
    execute_authorized_with(
        residency,
        boundary,
        plan,
        residency.runtime_for_indexed_page_in(),
    )
}

fn execute_authorized_with(
    residency: &CudaWeightResidency,
    boundary: &IndexedPageInBoundaryWitness,
    plan: &IndexedMultiBankPageInPlan,
    submitter: &dyn BatchSubmitter,
) -> std::result::Result<IndexedPageInReceipt, IndexedPageInError> {
    if plan.authority_id != residency.indexed_authority_id() {
        return Err(IndexedPageInError::not_submitted(
            "plan and execution residency authorities differ",
        ));
    }
    let _authority = residency
        .begin_indexed_page_in(boundary)
        .map_err(|error| IndexedPageInError::not_submitted(error.to_string()))?;
    let _residency = residency.lock_for_indexed_page_in();
    execute_with(submitter, plan)
}

#[cfg(feature = "gpu-tests")]
struct PartialWriteFaultSubmitter<'a> {
    runtime: &'a CudaRuntime,
    writes: usize,
}

#[cfg(feature = "gpu-tests")]
impl BatchSubmitter for PartialWriteFaultSubmitter<'_> {
    fn submit(
        &self,
        copies: &[CudaBatchCopy],
    ) -> std::result::Result<Duration, IndexedSubmitFailure> {
        let writes = self.writes.min(copies.len());
        if writes > 0 {
            // SAFETY: the production plan already validated and retained these
            // exact intervals. The fault harness deliberately submits only a
            // prefix, then reports ambiguous completion.
            if let Err(error) = unsafe {
                self.runtime
                    .indexed_batch_copy_elapsed_ms(&copies[..writes])
            } {
                let (detail, completion) = error.into_parts();
                return Err(IndexedSubmitFailure { detail, completion });
            }
        }
        Err(IndexedSubmitFailure {
            detail: format!(
                "injected submitted failure after {writes} partial destination write(s)"
            ),
            completion: FailedHtodCompletion::MayBeInFlight,
        })
    }
}

/// GPU-test fault injection: perform a real prefix of destination writes, then
/// force the structured ambiguous-completion quarantine path.
#[cfg(feature = "gpu-tests")]
pub fn execute_indexed_multi_bank_page_in_with_partial_write_fault(
    residency: &CudaWeightResidency,
    boundary: &IndexedPageInBoundaryWitness,
    plan: &IndexedMultiBankPageInPlan,
    writes: usize,
) -> std::result::Result<IndexedPageInReceipt, IndexedPageInError> {
    execute_authorized_with(
        residency,
        boundary,
        plan,
        &PartialWriteFaultSubmitter {
            runtime: residency.runtime_for_indexed_page_in(),
            writes,
        },
    )
}

fn execute_with(
    submitter: &dyn BatchSubmitter,
    plan: &IndexedMultiBankPageInPlan,
) -> std::result::Result<IndexedPageInReceipt, IndexedPageInError> {
    let transaction = DestinationTransaction::begin(plan)?;
    let copies: Vec<CudaBatchCopy> = plan.copies.iter().map(|entry| entry.copy).collect();
    let elapsed = if copies.is_empty() {
        transaction.ready();
        Duration::ZERO
    } else {
        match submitter.submit(&copies) {
            Ok(elapsed) => {
                transaction.ready();
                elapsed
            }
            Err(failure) => {
                let disposition = match &failure.completion {
                    FailedHtodCompletion::NotSubmitted => {
                        transaction.ready();
                        IndexedPageInFailureDisposition::NotSubmitted
                    }
                    FailedHtodCompletion::Completed(_) => {
                        transaction.poison();
                        quarantine_plan_ownership(plan, false);
                        IndexedPageInFailureDisposition::QuarantinedCompleted
                    }
                    FailedHtodCompletion::MayBeInFlight => {
                        transaction.poison();
                        quarantine_plan_ownership(plan, true);
                        IndexedPageInFailureDisposition::QuarantinedMayBeInFlight
                    }
                };
                let mut stats = attribution_stats()
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                stats.entry(plan.attribution).or_default().failures += 1;
                return Err(IndexedPageInError {
                    detail: failure.detail,
                    completion: failure.completion,
                    disposition,
                });
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

fn quarantine_plan_ownership(plan: &IndexedMultiBankPageInPlan, include_sources: bool) {
    let mut seen = HashSet::new();
    let mut retained: Vec<IndexedBankOwner> = Vec::new();
    for entry in &plan.copies {
        for owner in std::iter::once(&entry.bank.destination_owner)
            .chain(include_sources.then_some(&entry.bank.source_owner))
        {
            let identity = Arc::as_ptr(owner) as *const () as usize;
            if seen.insert(identity) {
                retained.push(Arc::clone(owner));
            }
        }
    }
    quarantine_indexed_page_in(Box::new(retained));
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
        IndexedExpertBank::test(IndexedExpertBankSpec {
            source_base: src,
            destination_base: dst,
            source_expert_stride: bytes,
            destination_slot_stride: bytes,
            bytes_per_expert: bytes,
            experts: 8,
            slots: 8,
        })
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
        assert_eq!(plan.copy_identities().len(), 8);
        assert_eq!(plan.copy_identities()[0].bank, IndexedBankKind::Packed);
        assert_eq!(
            plan.copy_identities()[1].bank,
            IndexedBankKind::AuxiliaryScale
        );
    }

    #[test]
    fn rejects_overlapping_banks_and_pointer_overflow() {
        let selections = [ExpertSlot {
            expert: 0,
            slot: 0,
            resident: false,
        }];
        let overlap = IndexedMultiBankPageInPlan::build(
            IndexedPageInAttribution {
                layer: 1,
                phase: IndexedPageInPhase::Decode,
            },
            &selections,
            &[ProjectionBankPair {
                projection: 0,
                packed: bank(0x1000, 0x20_0000, 64),
                auxiliary_scale: bank(0x2000, 0x20_0020, 64),
            }],
        )
        .unwrap_err();
        assert!(overlap.to_string().contains("intervals overlap"));

        let cross_projection_overlap = IndexedMultiBankPageInPlan::build(
            IndexedPageInAttribution {
                layer: 1,
                phase: IndexedPageInPhase::Decode,
            },
            &selections,
            &[
                ProjectionBankPair {
                    projection: 0,
                    packed: bank(0x1000, 0x10_0000, 64),
                    auxiliary_scale: bank(0x2000, 0x20_0000, 64),
                },
                ProjectionBankPair {
                    projection: 1,
                    packed: bank(0x3000, 0x20_0020, 64),
                    auxiliary_scale: bank(0x4000, 0x40_0000, 64),
                },
            ],
        )
        .unwrap_err();
        assert!(
            cross_projection_overlap
                .to_string()
                .contains("intervals overlap")
        );

        let source_overflow = IndexedExpertBank::new(
            IndexedExpertBankSpec {
                source_base: u64::MAX - 31,
                destination_base: 0x1000,
                source_expert_stride: 64,
                destination_slot_stride: 64,
                bytes_per_expert: 64,
                experts: 2,
                slots: 2,
            },
            1,
            Arc::new(()),
            Arc::new(()),
        )
        .unwrap_err();
        assert!(source_overflow.to_string().contains("overflow"));

        let destination_overflow = IndexedExpertBank::new(
            IndexedExpertBankSpec {
                source_base: 0x1000,
                destination_base: u64::MAX - 31,
                source_expert_stride: 64,
                destination_slot_stride: 64,
                bytes_per_expert: 64,
                experts: 2,
                slots: 2,
            },
            1,
            Arc::new(()),
            Arc::new(()),
        )
        .unwrap_err();
        assert!(destination_overflow.to_string().contains("overflow"));

        let multiplication_overflow = IndexedExpertBank::new(
            IndexedExpertBankSpec {
                source_base: 0x1000,
                destination_base: 0x2000,
                source_expert_stride: usize::MAX,
                destination_slot_stride: 64,
                bytes_per_expert: 1,
                experts: 3,
                slots: 2,
            },
            1,
            Arc::new(()),
            Arc::new(()),
        )
        .unwrap_err();
        assert!(
            multiplication_overflow
                .to_string()
                .contains("offset overflow")
        );
    }

    struct PartialFailure {
        observed: AtomicUsize,
    }

    impl BatchSubmitter for PartialFailure {
        fn submit(
            &self,
            copies: &[CudaBatchCopy],
        ) -> std::result::Result<Duration, IndexedSubmitFailure> {
            self.observed.store(3.min(copies.len()), Ordering::Relaxed);
            Err(IndexedSubmitFailure {
                detail: "injected failure after three destination writes".into(),
                completion: FailedHtodCompletion::MayBeInFlight,
            })
        }
    }

    #[test]
    fn partial_copy_failure_preserves_structured_lifetime_and_poisons_every_bank() {
        let _guard = stats_test_lock();
        crate::weight_paging::reset_global_offload_stats();
        reset_indexed_page_in_attribution_stats();
        let plan = fixture();
        let before = crate::weight_paging::global_offload_stats();
        let submitter = PartialFailure {
            observed: AtomicUsize::new(0),
        };
        let error = execute_with(&submitter, &plan).unwrap_err();
        assert!(matches!(
            error.completion(),
            FailedHtodCompletion::MayBeInFlight
        ));
        assert_eq!(
            error.disposition(),
            IndexedPageInFailureDisposition::QuarantinedMayBeInFlight
        );
        assert_eq!(submitter.observed.load(Ordering::Relaxed), 3);
        assert!(
            plan.copies
                .iter()
                .all(|entry| entry.bank.visibility() == IndexedBankVisibility::Poisoned)
        );
        let after = crate::weight_paging::global_offload_stats();
        assert_eq!(after.htod_bytes, before.htod_bytes);
        assert_eq!(
            after.indexed_expert_page_ins,
            before.indexed_expert_page_ins
        );
        let rows = indexed_page_in_attribution_stats();
        assert_eq!(rows[0].1.failures, 1);
        assert_eq!(rows[0].1.payload_bytes, 0);
        assert_eq!(rows[0].1.batches, 0);
    }

    struct Success;

    impl BatchSubmitter for Success {
        fn submit(
            &self,
            _copies: &[CudaBatchCopy],
        ) -> std::result::Result<Duration, IndexedSubmitFailure> {
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
        assert!(
            plan.copies
                .iter()
                .all(|entry| entry.bank.visibility() == IndexedBankVisibility::Ready)
        );
    }
}
