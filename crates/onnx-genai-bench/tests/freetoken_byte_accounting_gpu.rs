#![cfg(feature = "gpu-tests")]

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use cudarc::driver::sys::CUdeviceptr;
use onnx_genai_bench::freetoken_byte_ab::{
    Phase, SemanticProof, SyntheticFixture, WorkloadSpec, run_estimate_comparison,
    synthetic_workload,
};
use onnx_runtime_ep_api::{
    ExecutionProvider, ExecutorArtifactGeneration, ExecutorInstanceId, ExternalMmapRegion,
    LazyWeight, LazyWeightBoundary, MmapRegionSource, ResidencyResizeRequest, ResidentWeight,
    ResizeDirection, WeightHandleError, plan_resize,
};
use onnx_runtime_ep_cuda::CudaExecutionProvider;
use onnx_runtime_ep_cuda::byte_telemetry::{
    OBSERVED_BYTE_SCHEMA, ObservedBoundary, ObservedByteLedger, ObservedCategory, ObservedEvent,
    ObservedLayer, ObservedPhase, ObservedScope, ObservedSnapshot, ObservedStatus,
};
use onnx_runtime_ep_cuda::runtime::CudaRuntime;
use onnx_runtime_ep_cuda::weight_paging::DeviceOffloadPolicy;
use onnx_runtime_ep_cuda::{CsaCheckpointJournal, CsaMetrics};
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::{
    DeviceKey, HolderId, LeaseLedger, LedgerGovernor, MemoryGovernor, Tier,
};
use serde::Serialize;
use sha2::{Digest, Sha256};

const EVENT_CAPACITY: usize = 16_384;
const CONTROL_BYTES: usize = 64 * 1024;

struct SyntheticMmap {
    mapping_id: usize,
    bytes: Arc<[u8]>,
}

impl MmapRegionSource for SyntheticMmap {
    fn region_bytes(&self, region: &ExternalMmapRegion) -> Result<&[u8], WeightHandleError> {
        if region.mapping_id != self.mapping_id {
            return Err(WeightHandleError::DeviceBinding(format!(
                "synthetic mmap identity mismatch: expected {}, got {}",
                self.mapping_id, region.mapping_id
            )));
        }
        self.bytes
            .get(region.offset..region.offset.saturating_add(region.len))
            .ok_or_else(|| {
                WeightHandleError::DeviceBinding(format!(
                    "synthetic mmap range {}..{} exceeds {} bytes",
                    region.offset,
                    region.offset.saturating_add(region.len),
                    self.bytes.len()
                ))
            })
    }
}

struct ObservedProvider {
    provider: CudaExecutionProvider,
    ledger: ObservedByteLedger,
}

impl ObservedProvider {
    fn new(
        executor: u64,
        generation: u64,
        logical_session: u64,
        budget_bytes: u64,
    ) -> Result<Self> {
        let policy = DeviceOffloadPolicy {
            enabled: true,
            device_budget_bytes: Some(budget_bytes),
            ..DeviceOffloadPolicy::default()
        };
        let capacity = budget_bytes
            .checked_add(2_u64 << 30)
            .context("observed provider governor capacity overflow")?;
        let governor: Arc<dyn MemoryGovernor + Send + Sync> = Arc::new(LedgerGovernor::new(
            LeaseLedger::new_for_device(DeviceKey::device(0), capacity, capacity, 0),
        ));
        let provider = CudaExecutionProvider::initialized_with_offload_policy_and_governor(
            0,
            policy,
            Arc::clone(&governor),
        )
        .context("construct initialized governed CUDA provider for observed workload")?;
        provider
            .adopt_memory_governor(governor.as_ref(), Tier::Device, HolderId::new(executor))
            .context("adopt mapped weight allowance for observed workload")?;
        let ledger = provider
            .open_observed_byte_session(
                ExecutorInstanceId::from_raw(executor),
                ExecutorArtifactGeneration::from_raw(generation),
                logical_session,
                EVENT_CAPACITY,
            )
            .context("open exact provider/executor observed-byte session")?;
        Ok(Self { provider, ledger })
    }

    fn runtime(&self) -> &Arc<CudaRuntime> {
        self.provider.runtime()
    }

    fn shutdown(&mut self, label: &str) -> Result<()> {
        self.provider
            .shutdown()
            .with_context(|| format!("shutdown {label} observed provider"))?;
        ensure!(
            self.provider
                .release_queue()
                .wait_until_idle(std::time::Duration::from_secs(30)),
            "{label} observed provider did not drain deferred releases"
        );
        Ok(())
    }

    fn into_snapshot(self, label: &str) -> Result<ObservedSnapshot> {
        let queue = Arc::clone(self.provider.release_queue());
        let Self { provider, ledger } = self;
        drop(provider);
        ensure!(
            queue.wait_until_idle(std::time::Duration::from_secs(30)),
            "{label} provider drop did not drain deferred releases"
        );
        ledger.snapshot().map_err(anyhow::Error::from)
    }
}

fn observed_phase(phase: Phase) -> ObservedPhase {
    match phase {
        Phase::Setup => ObservedPhase::Setup,
        Phase::Prefill => ObservedPhase::Prefill,
        Phase::DirectWarmup => ObservedPhase::DirectWarmup,
        Phase::CaptureSetup => ObservedPhase::CaptureSetup,
        Phase::Replay => ObservedPhase::Replay,
        Phase::DecodeSteady => ObservedPhase::DecodeSteady,
        Phase::Failure => ObservedPhase::Failure,
    }
}

fn reconstructed_phase_bytes(
    snapshot: &ObservedSnapshot,
    phase: ObservedPhase,
    category: ObservedCategory,
) -> Result<u64> {
    let mut useful = 0_u64;
    for status in ObservedStatus::ALL {
        let reconstructed = snapshot
            .events
            .iter()
            .filter(|event| {
                event.phase == phase && event.category == category && event.status == status
            })
            .try_fold(0_u64, |total, event| {
                total
                    .checked_add(event.bytes)
                    .context("reconstructed event byte sum overflow")
            })?;
        ensure!(
            reconstructed == snapshot.phase_bytes(phase, category, status),
            "independent {phase:?}/{category:?}/{status:?} reconstruction {reconstructed} \
             disagrees with ledger total {}",
            snapshot.phase_bytes(phase, category, status)
        );
        if status.is_useful() {
            useful = useful
                .checked_add(reconstructed)
                .context("reconstructed useful byte sum overflow")?;
        }
    }
    Ok(useful)
}

fn fixture_budget(workload: &WorkloadSpec) -> Result<u64> {
    workload.banks.iter().try_fold(0_u64, |total, bank| {
        bank.bytes_per_expert
            .checked_mul(u64::from(bank.cache_slots))
            .and_then(|bytes| total.checked_add(bytes))
            .context("fixture cache budget overflow")
    })
}

fn fixture_sources(workload: &WorkloadSpec) -> Result<Vec<SyntheticMmap>> {
    workload
        .banks
        .iter()
        .enumerate()
        .map(|(bank_index, bank)| {
            let len = usize::try_from(bank.bytes_per_expert)
                .context("fixture expert extent exceeds usize")?;
            Ok(SyntheticMmap {
                mapping_id: bank_index + 1,
                bytes: vec![(bank_index as u8).wrapping_mul(37).wrapping_add(11); len].into(),
            })
        })
        .collect()
}

fn fixture_lazy_weights(
    workload: &WorkloadSpec,
    sources: &[SyntheticMmap],
) -> Result<Vec<Vec<LazyWeight>>> {
    workload
        .banks
        .iter()
        .enumerate()
        .map(|(bank_index, bank)| {
            let source = &sources[bank_index];
            (0..bank.expert_count)
                .map(|_| {
                    let bytes = Arc::clone(&source.bytes);
                    let shape = vec![bytes.len()];
                    LazyWeight::new(
                        LazyWeightBoundary::BlockQuantizedMoe,
                        DataType::Uint8,
                        shape.clone(),
                        vec![ExternalMmapRegion {
                            mapping_id: source.mapping_id,
                            offset: 0,
                            len: bytes.len(),
                        }],
                        move || {
                            ResidentWeight::new(DataType::Uint8, shape.clone(), Arc::clone(&bytes))
                        },
                    )
                    .map_err(anyhow::Error::from)
                })
                .collect()
        })
        .collect()
}

fn lazy_for_source(source: &SyntheticMmap) -> Result<LazyWeight> {
    let bytes = Arc::clone(&source.bytes);
    let shape = vec![bytes.len()];
    LazyWeight::new(
        LazyWeightBoundary::BlockQuantizedMoe,
        DataType::Uint8,
        shape.clone(),
        vec![ExternalMmapRegion {
            mapping_id: source.mapping_id,
            offset: 0,
            len: bytes.len(),
        }],
        move || ResidentWeight::new(DataType::Uint8, shape.clone(), Arc::clone(&bytes)),
    )
    .map_err(anyhow::Error::from)
}

fn verify_first_page(
    runtime: &CudaRuntime,
    page: &onnx_runtime_ep_cuda::weight_paging::CudaWeightPage,
    expected: u8,
) -> Result<()> {
    let len = page.len().min(64);
    let mut bytes = vec![0_u8; len];
    // SAFETY: the page owns at least `len` initialized device bytes.
    unsafe { runtime.dtoh(&mut bytes, page.device_ptr() as usize as CUdeviceptr) }
        .context("read production-paged expert control bytes")?;
    ensure!(
        bytes.iter().all(|&byte| byte == expected),
        "production-paged expert bytes differ from the synthetic mmap source"
    );
    Ok(())
}

fn semantic_digest(workload: &WorkloadSpec) -> Result<String> {
    let bytes = serde_json::to_vec(&workload.routes).context("serialize fixture routes")?;
    Ok(Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn run_baseline(workload: &WorkloadSpec, executor: u64) -> Result<ObservedSnapshot> {
    let budget = fixture_budget(workload)?;
    let mut observed = ObservedProvider::new(executor, 1, executor + 10_000, budget)?;
    let sources = fixture_sources(workload)?;
    observed.ledger.set_phase(ObservedPhase::Setup)?;
    let buffers = workload
        .banks
        .iter()
        .map(|bank| {
            observed
                .runtime()
                .alloc_raw(bank.bytes_per_expert as usize)
                .context("allocate baseline streaming buffer")
        })
        .collect::<Result<Vec<_>>>()?;

    for step in &workload.routes {
        observed.ledger.set_phase(observed_phase(step.phase))?;
        for (bank_index, selections) in step.selections.iter().enumerate() {
            let unique = selections.iter().copied().collect::<BTreeSet<_>>();
            for _expert in unique {
                // SAFETY: each baseline buffer exactly covers its bank's source.
                unsafe {
                    observed
                        .runtime()
                        .htod(&sources[bank_index].bytes, buffers[bank_index])
                }
                .context("complete baseline production H2D stream")?;
            }
        }
    }
    observed.ledger.set_phase(ObservedPhase::Verification)?;
    for (bank_index, &ptr) in buffers.iter().enumerate() {
        let mut control = [0_u8; 64];
        // SAFETY: every fixture expert is at least 64 bytes.
        unsafe { observed.runtime().dtoh(&mut control, ptr) }
            .context("read baseline output control")?;
        ensure!(
            control
                .iter()
                .all(|&byte| byte == sources[bank_index].bytes[0]),
            "baseline stream changed bank {bank_index} bytes"
        );
        // SAFETY: pointer came from this exact runtime and is freed once.
        unsafe { observed.runtime().free_raw(ptr) }.context("free baseline stream buffer")?;
    }
    observed.ledger.set_phase(ObservedPhase::Teardown)?;
    observed.shutdown("baseline")?;
    observed.into_snapshot("baseline")
}

fn run_optimized(workload: &WorkloadSpec, executor: u64) -> Result<ObservedSnapshot> {
    let budget = fixture_budget(workload)?;
    let mut observed = ObservedProvider::new(executor, 1, executor + 10_000, budget)?;
    let sources = fixture_sources(workload)?;
    let lazies = fixture_lazy_weights(workload, &sources)?;
    let residency = Arc::clone(
        observed
            .provider
            .residency()
            .context("optimized provider omitted production weight residency")?,
    );
    let mut verified_bank = vec![false; workload.banks.len()];

    observed.ledger.set_phase(ObservedPhase::Setup)?;
    for step in &workload.routes {
        observed.ledger.set_phase(observed_phase(step.phase))?;
        for (bank_index, selections) in step.selections.iter().enumerate() {
            let unique = selections.iter().copied().collect::<BTreeSet<_>>();
            for expert in unique {
                let key = ((bank_index as u64) << 32) | u64::from(expert);
                let page = residency
                    .resident_mapped(
                        key,
                        &lazies[bank_index][expert as usize],
                        &sources[bank_index],
                    )
                    .with_context(|| {
                        format!(
                            "page production expert {expert} for bank {}",
                            workload.banks[bank_index].bank
                        )
                    })?;
                if !verified_bank[bank_index] {
                    verify_first_page(observed.runtime(), &page, sources[bank_index].bytes[0])?;
                    verified_bank[bank_index] = true;
                }
            }
        }
    }
    ensure!(
        verified_bank.into_iter().all(|verified| verified),
        "optimized workload failed to exercise every expert bank"
    );
    observed.ledger.set_phase(ObservedPhase::Teardown)?;
    let shrink = ResidencyResizeRequest {
        direction: ResizeDirection::Shrink,
        target_bytes: budget,
        priority: 0,
    };
    let outcome = residency.execute_resize(plan_resize(shrink, residency.resize_safe_point(1)), 1);
    ensure!(
        outcome.rejection.is_none(),
        "optimized teardown reclaim was rejected: {:?}",
        outcome.rejection
    );
    ensure!(
        observed
            .provider
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "optimized teardown did not complete VMM page releases"
    );
    drop(residency);
    observed.shutdown("optimized")?;
    observed.into_snapshot("optimized")
}

#[derive(Serialize)]
struct ArmSummary {
    scope: ObservedScope,
    event_count: usize,
    decode_tokens: u64,
    phase_useful_bytes: BTreeMap<ObservedPhase, BTreeMap<ObservedCategory, u64>>,
    decode_useful_bytes: BTreeMap<ObservedCategory, u64>,
    total_vmm_map_committed: u64,
    total_vmm_unmap_reclaimed: u64,
    mapped_bytes_not_reclaimed: u64,
    quarantined_page_in_bytes: u64,
    category_coverage: BTreeMap<ObservedCategory, CategoryCoverage>,
    events: Vec<ObservedEvent>,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum CategoryCoverage {
    Observed,
    Unsupported,
    NotObserved,
}

fn summarize(workload: &WorkloadSpec, snapshot: &ObservedSnapshot) -> Result<ArmSummary> {
    let decode_tokens = workload
        .routes
        .iter()
        .filter(|step| step.phase == Phase::DecodeSteady)
        .try_fold(0_u64, |tokens, _| {
            tokens
                .checked_add(u64::from(workload.batch))
                .context("decode token denominator overflow")
        })?;
    let phase_useful_bytes = ObservedPhase::ALL
        .into_iter()
        .map(|phase| {
            ObservedCategory::ALL
                .into_iter()
                .map(|category| {
                    reconstructed_phase_bytes(snapshot, phase, category)
                        .map(|bytes| (category, bytes))
                })
                .collect::<Result<BTreeMap<_, _>>>()
                .map(|bytes| (phase, bytes))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    let decode_useful_bytes = phase_useful_bytes[&ObservedPhase::DecodeSteady].clone();
    let category_coverage = ObservedCategory::ALL
        .into_iter()
        .map(|category| {
            let unsupported = snapshot.events.iter().any(|event| {
                event.category == category && event.status == ObservedStatus::Unsupported
            });
            let observed = snapshot
                .events
                .iter()
                .any(|event| event.category == category);
            (
                category,
                if unsupported {
                    CategoryCoverage::Unsupported
                } else if observed {
                    CategoryCoverage::Observed
                } else {
                    CategoryCoverage::NotObserved
                },
            )
        })
        .collect();
    let total_vmm_map_committed =
        snapshot.bytes(ObservedCategory::VmmMap, ObservedStatus::Committed);
    let total_vmm_unmap_reclaimed =
        snapshot.bytes(ObservedCategory::VmmUnmap, ObservedStatus::Reclaimed);
    let mapped_bytes_not_reclaimed = total_vmm_map_committed
        .checked_sub(total_vmm_unmap_reclaimed)
        .context("observed VMM unmap bytes exceed committed map bytes")?;
    Ok(ArmSummary {
        scope: snapshot.scope,
        event_count: snapshot.events.len(),
        decode_tokens,
        phase_useful_bytes,
        decode_useful_bytes,
        total_vmm_map_committed,
        total_vmm_unmap_reclaimed,
        mapped_bytes_not_reclaimed,
        quarantined_page_in_bytes: snapshot
            .bytes(ObservedCategory::PageIn, ObservedStatus::Quarantined),
        category_coverage,
        events: snapshot.events.clone(),
    })
}

#[derive(Serialize)]
struct FixtureReport {
    schema: &'static str,
    provenance: &'static str,
    category_layers: BTreeMap<ObservedCategory, ObservedLayer>,
    fixture: String,
    limitation: &'static str,
    semantic_route_digest: String,
    semantic_equivalent: bool,
    baseline_semantics: SemanticProof,
    optimized_semantics: SemanticProof,
    decode_h2d: DecodeH2dComparison,
    baseline: ArmSummary,
    optimized: ArmSummary,
}

#[derive(Serialize)]
struct DecodeH2dComparison {
    baseline_bytes: u64,
    optimized_bytes: u64,
    optimized_minus_baseline: String,
    baseline_bytes_per_token: u64,
    optimized_bytes_per_token: u64,
}

fn write_report_if_requested(fixture: &str, report: &FixtureReport) -> Result<()> {
    let Some(directory) = std::env::var_os("ONNX_GENAI_FREETOKEN_OBSERVED_OUTPUT_DIR") else {
        return Ok(());
    };
    let directory = std::path::PathBuf::from(directory);
    std::fs::create_dir_all(&directory)
        .with_context(|| format!("create observed report directory {}", directory.display()))?;
    let path = directory.join(format!("{fixture}.json"));
    std::fs::write(&path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("write observed report {}", path.display()))
}

#[test]
fn production_runtime_boundaries_record_only_completed_real_operations() -> Result<()> {
    const PAGE_BYTES: usize = 2 << 20;
    const MAIN_STATE_BYTES: usize = 4 << 10;
    const INDEX_STATE_BYTES: usize = 2 << 10;
    const STATE_BYTES: usize = MAIN_STATE_BYTES + INDEX_STATE_BYTES;

    let mut observed = ObservedProvider::new(91, 3, 701, PAGE_BYTES as u64)?;
    observed.ledger.set_phase(ObservedPhase::Setup)?;
    let source = vec![0x5a_u8; CONTROL_BYTES];
    let first = observed.runtime().alloc_raw(CONTROL_BYTES)?;
    let second = observed.runtime().alloc_raw(CONTROL_BYTES)?;
    // SAFETY: both allocations cover CONTROL_BYTES.
    unsafe {
        observed.runtime().memset_zero(first, CONTROL_BYTES)?;
        observed.runtime().htod(&source, first)?;
        observed.runtime().dtod(first, second, CONTROL_BYTES)?;
    }
    let mut output = vec![0_u8; CONTROL_BYTES];
    // SAFETY: second contains CONTROL_BYTES initialized by the completed D2D.
    unsafe { observed.runtime().dtoh(&mut output, second)? };
    ensure!(output == source, "real CUDA boundary control changed bytes");

    observed.ledger.set_phase(ObservedPhase::CaptureSetup)?;
    observed.runtime().begin_graph_capture(&[])?;
    // SAFETY: capture records a same-size D2D over two live fixed addresses.
    unsafe {
        observed
            .runtime()
            .dtod_async(first, second, CONTROL_BYTES)?;
    }
    observed.runtime().end_graph_capture()?;
    observed.ledger.set_phase(ObservedPhase::Replay)?;
    for _ in 0..3 {
        observed.runtime().replay_graph()?;
        let mut replay_output = vec![0_u8; CONTROL_BYTES];
        // SAFETY: D2H orders after the replay and the destination covers it.
        unsafe { observed.runtime().dtoh(&mut replay_output, second)? };
        ensure!(
            replay_output == source,
            "captured production D2D replay changed control bytes"
        );
    }
    observed.ledger.set_phase(ObservedPhase::CaptureSetup)?;
    let main_state = observed.runtime().alloc_raw(MAIN_STATE_BYTES)?;
    let index_state = observed.runtime().alloc_raw(INDEX_STATE_BYTES)?;
    let main_source = vec![0xa5_u8; MAIN_STATE_BYTES];
    let index_source = vec![0x5a_u8; INDEX_STATE_BYTES];
    // SAFETY: both state allocations exactly cover their host sources.
    unsafe {
        observed.runtime().htod(&main_source, main_state)?;
        observed.runtime().htod(&index_source, index_state)?;
    }
    let journal = CsaCheckpointJournal::new(
        Arc::clone(observed.runtime()),
        4,
        MAIN_STATE_BYTES,
        INDEX_STATE_BYTES,
        Arc::new(CsaMetrics::default()),
    )?;
    // SAFETY: carry pointers and byte extents name the live allocations above.
    let checkpoint = unsafe {
        journal.checkpoint(
            main_state,
            index_state,
            MAIN_STATE_BYTES,
            INDEX_STATE_BYTES,
            32,
            3,
        )?
    };
    // SAFETY: zeroing and restore stay within the live carry allocations.
    unsafe {
        observed
            .runtime()
            .memset_zero(main_state, MAIN_STATE_BYTES)?;
        observed
            .runtime()
            .memset_zero(index_state, INDEX_STATE_BYTES)?;
        journal.restore_prefix(&checkpoint, 24, 3, main_state, index_state, None)?;
    }
    let mut main_restored = vec![0_u8; MAIN_STATE_BYTES];
    let mut index_restored = vec![0_u8; INDEX_STATE_BYTES];
    // SAFETY: the restored carry allocations cover both destinations.
    unsafe {
        observed.runtime().dtoh(&mut main_restored, main_state)?;
        observed.runtime().dtoh(&mut index_restored, index_state)?;
    }
    ensure!(
        main_restored == main_source && index_restored == index_source,
        "CSA checkpoint rollback did not restore exact state bytes"
    );

    let mmap = SyntheticMmap {
        mapping_id: 91,
        bytes: vec![0x3c_u8; PAGE_BYTES].into(),
    };
    let first_lazy = lazy_for_source(&mmap)?;
    let second_lazy = lazy_for_source(&mmap)?;
    let residency = Arc::clone(
        observed
            .provider
            .residency()
            .context("governed control provider omitted weight residency")?,
    );
    observed.ledger.set_phase(ObservedPhase::DirectWarmup)?;
    let first_page = residency
        .resident_mapped(1, &first_lazy, &mmap)
        .context("page first real production weight")?;
    verify_first_page(observed.runtime(), &first_page, 0x3c)?;
    drop(first_page);
    let second_page = residency
        .resident_mapped(2, &second_lazy, &mmap)
        .context("page second real production weight")?;
    verify_first_page(observed.runtime(), &second_page, 0x3c)?;
    drop(second_page);
    ensure!(
        observed
            .provider
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "evicted production page did not finish its deferred VMM unmap"
    );

    observed.ledger.set_phase(ObservedPhase::Verification)?;
    // SAFETY: both pointers came from this runtime and are freed once.
    unsafe {
        observed.runtime().free_raw(second)?;
        observed.runtime().free_raw(first)?;
        observed.runtime().free_raw(index_state)?;
        observed.runtime().free_raw(main_state)?;
    }
    drop(journal);
    drop(residency);
    observed.ledger.set_phase(ObservedPhase::Teardown)?;
    observed.shutdown("production boundary")?;
    let snapshot = observed.into_snapshot("production boundary")?;
    for (category, bytes) in [
        (
            ObservedCategory::CudaMemset,
            (CONTROL_BYTES + STATE_BYTES) as u64,
        ),
        (
            ObservedCategory::H2d,
            (CONTROL_BYTES + STATE_BYTES + 2 * PAGE_BYTES) as u64,
        ),
        (
            ObservedCategory::D2d,
            (CONTROL_BYTES + 2 * STATE_BYTES) as u64,
        ),
        (
            ObservedCategory::D2h,
            (4 * CONTROL_BYTES + STATE_BYTES + 2 * 64) as u64,
        ),
    ] {
        ensure!(
            snapshot.bytes(category, ObservedStatus::Completed) == bytes,
            "{category:?} completed bytes did not match the real operation argument"
        );
    }
    ensure!(
        snapshot.bytes(ObservedCategory::SourceRead, ObservedStatus::Completed) == 0,
        "unperformed source reads must remain zero"
    );
    ensure!(
        snapshot.bytes(ObservedCategory::SourceRead, ObservedStatus::Unsupported) == 0
            && snapshot.events.iter().any(|event| {
                event.category == ObservedCategory::SourceRead
                    && event.status == ObservedStatus::Unsupported
                    && event.bytes == 0
            }),
        "synthetic mmap source I/O must be explicitly unsupported with zero bytes"
    );
    ensure!(
        snapshot.bytes(ObservedCategory::MmapPageIn, ObservedStatus::Completed) == 0
            && snapshot.events.iter().any(|event| {
                event.category == ObservedCategory::MmapPageIn
                    && event.status == ObservedStatus::Unsupported
                    && event.bytes == 0
            }),
        "synthetic mmap page-in must be explicitly unsupported with zero bytes"
    );
    ensure!(
        snapshot.bytes(ObservedCategory::HostWrite, ObservedStatus::Completed)
            == 2 * PAGE_BYTES as u64,
        "host writes must come from the exact production staging copies"
    );
    ensure!(
        snapshot.bytes(ObservedCategory::VmmMap, ObservedStatus::Committed)
            == 2 * PAGE_BYTES as u64,
        "VMM map bytes must come from committed production granules"
    );
    ensure!(
        snapshot.bytes(ObservedCategory::VmmUnmap, ObservedStatus::Reclaimed) >= PAGE_BYTES as u64,
        "eviction must observe the production VMM unmap result"
    );
    ensure!(
        snapshot.bytes(ObservedCategory::PageIn, ObservedStatus::Published)
            == 2 * PAGE_BYTES as u64
            && snapshot.bytes(
                ObservedCategory::ExpertPublication,
                ObservedStatus::Published
            ) == 2 * PAGE_BYTES as u64,
        "page-in and expert publication must match the exact produced pages"
    );
    ensure!(
        snapshot.bytes(
            ObservedCategory::StatePublication,
            ObservedStatus::Published
        ) == STATE_BYTES as u64
            && snapshot.bytes(
                ObservedCategory::StatePublication,
                ObservedStatus::RolledBack
            ) == STATE_BYTES as u64,
        "state publication and rollback must match exact checkpoint carry bytes"
    );
    ensure!(
        snapshot.bytes(ObservedCategory::VmmReserve, ObservedStatus::Committed) == 0,
        "provider VMM reservation happened before attachment and must not be inferred per page"
    );
    ensure!(
        snapshot.events.iter().any(|event| {
            event.phase == ObservedPhase::CaptureSetup
                && event.category == ObservedCategory::D2d
                && event.status == ObservedStatus::Unsupported
        }),
        "capture D2D must be explicitly unsupported until replay completion receipts are wired"
    );
    ensure!(
        snapshot.events.iter().all(|event| {
            event.scope == snapshot.scope
                && event.sequence > 0
                && event.submission > 0
                && event.epoch == snapshot.epoch
        }),
        "every observed event must carry exact scope/submission/sequence identity"
    );
    eprintln!(
        "freetoken_boundary_control={}",
        serde_json::json!({
            "schema": OBSERVED_BYTE_SCHEMA,
            "events": snapshot.events.len(),
            "h2d_completed": snapshot.bytes(ObservedCategory::H2d, ObservedStatus::Completed),
            "d2h_completed": snapshot.bytes(ObservedCategory::D2h, ObservedStatus::Completed),
            "d2d_completed": snapshot.bytes(ObservedCategory::D2d, ObservedStatus::Completed),
            "cuda_memset_completed": snapshot.bytes(
                ObservedCategory::CudaMemset,
                ObservedStatus::Completed
            ),
            "host_write_completed": snapshot.bytes(
                ObservedCategory::HostWrite,
                ObservedStatus::Completed
            ),
            "vmm_map_committed": snapshot.bytes(
                ObservedCategory::VmmMap,
                ObservedStatus::Committed
            ),
            "vmm_unmap_reclaimed": snapshot.bytes(
                ObservedCategory::VmmUnmap,
                ObservedStatus::Reclaimed
            ),
            "page_in_published": snapshot.bytes(
                ObservedCategory::PageIn,
                ObservedStatus::Published
            ),
            "state_published": snapshot.bytes(
                ObservedCategory::StatePublication,
                ObservedStatus::Published
            ),
            "state_rolled_back": snapshot.bytes(
                ObservedCategory::StatePublication,
                ObservedStatus::RolledBack
            ),
            "source_read_completed": 0,
            "vmm_reserve": "not_observed_before_attachment"
        })
    );
    Ok(())
}

#[test]
fn default_off_records_nothing_and_same_label_siblings_cannot_bleed() -> Result<()> {
    let mut provider = CudaExecutionProvider::initialized(0)
        .context("construct default-off CUDA provider for telemetry isolation")?;
    let source = vec![0x2d_u8; CONTROL_BYTES];
    let ptr = provider.runtime().alloc_raw(CONTROL_BYTES)?;
    // SAFETY: `ptr` covers the source and is freed once below.
    unsafe { provider.runtime().htod(&source, ptr)? };
    unsafe { provider.runtime().free_raw(ptr)? };

    ensure!(
        provider
            .open_observed_byte_session(
                ExecutorInstanceId::UNSCOPED,
                ExecutorArtifactGeneration::from_raw(9),
                9001,
                16,
            )
            .is_err(),
        "unscoped executor identity must fail before attaching a recorder"
    );
    ensure!(
        provider
            .open_observed_byte_session(
                ExecutorInstanceId::from_raw(501),
                ExecutorArtifactGeneration::from_raw(0),
                9001,
                16,
            )
            .is_err(),
        "zero generation must fail before attaching a recorder"
    );
    ensure!(
        provider
            .open_observed_byte_session(
                ExecutorInstanceId::from_raw(501),
                ExecutorArtifactGeneration::from_raw(9),
                9001,
                0,
            )
            .is_err(),
        "zero event capacity must fail before attaching a recorder"
    );
    let ledger = provider
        .open_observed_byte_session(
            ExecutorInstanceId::from_raw(501),
            ExecutorArtifactGeneration::from_raw(9),
            9001,
            16,
        )
        .context("open ledger after default-off control")?;
    ensure!(
        ledger.snapshot()?.events.is_empty(),
        "operations completed before explicit attachment must not reach a hidden global ledger"
    );
    ensure!(
        provider
            .open_observed_byte_session(
                ExecutorInstanceId::from_raw(501),
                ExecutorArtifactGeneration::from_raw(9),
                9001,
                16,
            )
            .is_err(),
        "one provider runtime must not accept a second recorder"
    );

    let mut sibling = CudaExecutionProvider::initialized(0)
        .context("construct same-label sibling CUDA provider")?;
    let sibling_ledger = sibling.open_observed_byte_session(
        ExecutorInstanceId::from_raw(501),
        ExecutorArtifactGeneration::from_raw(9),
        9001,
        16,
    )?;
    ensure!(
        ledger.scope().provider != sibling_ledger.scope().provider,
        "provider identity must be derived from the exact instance, not public executor labels"
    );
    let sibling_ptr = sibling.runtime().alloc_raw(CONTROL_BYTES)?;
    // SAFETY: sibling pointer is owned by the sibling runtime.
    unsafe { sibling.runtime().free_raw(sibling_ptr)? };
    ensure!(
        ledger.snapshot()?.events.is_empty(),
        "sibling provider events bled into the first ledger"
    );
    ensure!(
        !sibling_ledger.snapshot()?.events.is_empty(),
        "sibling provider did not record its own exact event"
    );
    provider.shutdown()?;
    ensure!(
        provider
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "default-off provider did not drain"
    );
    sibling.shutdown()?;
    ensure!(
        sibling
            .release_queue()
            .wait_until_idle(std::time::Duration::from_secs(30)),
        "sibling provider did not drain"
    );
    Ok(())
}

#[test]
fn deepseek_and_glm_like_eight_token_results_come_from_production_receipts() -> Result<()> {
    for (fixture, label, executor) in [
        (SyntheticFixture::DeepseekLike, "deepseek-like", 101_u64),
        (SyntheticFixture::Glm52Like, "glm52-like", 102_u64),
    ] {
        let workload = synthetic_workload(fixture);
        let semantic = run_estimate_comparison(workload.clone())?;
        ensure!(
            semantic.contract.passed,
            "{label} semantic control failed: {:?}",
            semantic.contract.diagnostics
        );
        ensure!(
            semantic.baseline.semantics == semantic.optimized.semantics,
            "{label} baseline/optimized semantic proofs differ"
        );
        let baseline = run_baseline(&workload, executor)?;
        let optimized = run_optimized(&workload, executor + 1_000)?;
        let baseline_summary = summarize(&workload, &baseline)?;
        let optimized_summary = summarize(&workload, &optimized)?;
        ensure!(
            baseline_summary.decode_tokens == 8 && optimized_summary.decode_tokens == 8,
            "{label} must use the exact eight-token warmed decode denominator"
        );
        ensure!(
            baseline_summary.decode_useful_bytes[&ObservedCategory::H2d] > 0
                && optimized_summary.decode_useful_bytes[&ObservedCategory::H2d] > 0,
            "{label} did not execute a non-empty CUDA H2D path"
        );
        ensure!(
            optimized_summary.decode_useful_bytes[&ObservedCategory::VmmMap] > 0,
            "{label} optimized arm did not execute the production VMM page-in path"
        );
        ensure!(
            optimized_summary.decode_useful_bytes[&ObservedCategory::PageIn] > 0
                && optimized_summary.decode_useful_bytes[&ObservedCategory::ExpertPublication] > 0,
            "{label} optimized arm did not publish production expert pages"
        );
        let baseline_h2d = baseline_summary.decode_useful_bytes[&ObservedCategory::H2d];
        let optimized_h2d = optimized_summary.decode_useful_bytes[&ObservedCategory::H2d];
        let optimized_minus_baseline = if optimized_h2d >= baseline_h2d {
            (optimized_h2d - baseline_h2d).to_string()
        } else {
            format!("-{}", baseline_h2d - optimized_h2d)
        };
        let report = FixtureReport {
            schema: OBSERVED_BYTE_SCHEMA,
            provenance: "observed_production_boundary",
            category_layers: ObservedCategory::ALL
                .into_iter()
                .map(|category| (category, category.layer()))
                .collect(),
            fixture: label.to_string(),
            limitation: "synthetic zero-filled expert payloads through real CUDA/VMM residency; \
                         not a DeepSeek/GLM checkpoint or throughput claim",
            semantic_route_digest: semantic_digest(&workload)?,
            semantic_equivalent: true,
            baseline_semantics: semantic.baseline.semantics,
            optimized_semantics: semantic.optimized.semantics,
            decode_h2d: DecodeH2dComparison {
                baseline_bytes: baseline_h2d,
                optimized_bytes: optimized_h2d,
                optimized_minus_baseline,
                baseline_bytes_per_token: baseline_h2d / baseline_summary.decode_tokens,
                optimized_bytes_per_token: optimized_h2d / optimized_summary.decode_tokens,
            },
            baseline: baseline_summary,
            optimized: optimized_summary,
        };
        write_report_if_requested(label, &report)?;
        eprintln!(
            "freetoken_observed_summary={}",
            serde_json::json!({
                "fixture": report.fixture,
                "decode_h2d": report.decode_h2d,
                "baseline_events": report.baseline.event_count,
                "optimized_events": report.optimized.event_count,
                "semantic_equivalent": report.semantic_equivalent,
            })
        );
    }
    Ok(())
}

#[test]
fn observed_schema_and_boundary_names_are_stable() {
    assert_eq!(
        OBSERVED_BYTE_SCHEMA,
        "onnx-genai.freetoken-observed-bytes.v1"
    );
    assert_eq!(
        serde_json::to_string(&ObservedBoundary::AsyncCompletionUnsupported).unwrap(),
        "\"async_completion_unsupported\""
    );
    assert_eq!(
        serde_json::to_string(&ObservedCategory::MmapPageIn).unwrap(),
        "\"mmap_page_in\""
    );
}
