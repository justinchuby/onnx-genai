//! Deterministic, byte-level FreeToken A/B accounting.
//!
//! The harness is deliberately model-package independent. Synthetic expert
//! banks expose exact byte extents through a chunked source, so large MoE
//! layouts can be represented without allocating a checkpoint-sized buffer.
//! Every byte event names one authoritative boundary and is committed only
//! after that operation completes. Logical work, transport traffic, mapping
//! topology, and persistent residency are separate dimensions and are never
//! summed into a misleading universal "total bytes" number.

use std::collections::{BTreeMap, BTreeSet};

use anyhow::{Context, Result, bail, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const WORKLOAD_SCHEMA: &str = "onnx-genai.freetoken-byte-ab.workload.v2";
pub const RUN_SCHEMA: &str = "onnx-genai.freetoken-byte-ab.run.v2";
pub const COMPARISON_SCHEMA: &str = "onnx-genai.freetoken-byte-ab.comparison.v2";
pub const TAXONOMY_SCHEMA: &str = "onnx-genai.freetoken-byte-taxonomy.v1";

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Arm {
    /// Shipped/default path. It performs the semantic workload but never enters
    /// a FreeToken lookup or feature-specific accounting boundary.
    BaselineAbsent,
    /// Explicit synthetic FreeToken residency/cache path.
    Optimized,
    /// Baseline movement followed by deterministic rollback and quarantine.
    BaselineFailureControl,
    /// Optimized movement followed by the identical failure conditions.
    OptimizedFailureControl,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Setup,
    Prefill,
    DirectWarmup,
    CaptureSetup,
    Replay,
    DecodeSteady,
    Failure,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ByteClass {
    /// Bytes returned by the synthetic source's `read_exact` boundary.
    SourceRead,
    /// Requested payload capacity of a successful host allocation.
    HostAllocation,
    /// Non-zero payload bytes written into host storage.
    HostWrite,
    /// Zero-initialized host bytes. Disjoint from `host_write`.
    ZeroFill,
    /// H2D payload bytes counted only after completion.
    H2d,
    /// D2H payload bytes counted only after completion.
    D2h,
    /// D2D payload bytes counted only after completion.
    D2d,
    /// OS page-in bytes. Synthetic runs report zero unless an authoritative
    /// page-fault receipt is supplied.
    MmapPageIn,
    /// Bytes whose virtual range gained a physical mapping. This is topology,
    /// not transport, and is never added to H2D/D2H/D2D traffic.
    VmmMap,
    /// Bytes whose virtual range lost a physical mapping.
    VmmUnmap,
    /// Logical expert payload consumed by routed computation. This is useful
    /// work, not traffic.
    ExpertMaterialization,
    /// Logical recurrent/attention state payload advanced by the workload.
    StateMaterialization,
    /// Scratch or rollback-journal bytes touched.
    ScratchJournal,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CounterClass {
    Requests,
    Tokens,
    ExpertSelections,
    UniqueExperts,
    Submissions,
    FeatureLookups,
    CacheHits,
    CacheMisses,
    Evictions,
    CaptureSetups,
    Replays,
    StateUpdates,
    Rollbacks,
    Quarantines,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FailureDisposition {
    Failed,
    RolledBack,
    Quarantined,
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
pub struct ScopeIdentity {
    pub provider: u64,
    pub device: u32,
    pub executor: u64,
    pub generation: u64,
    pub logical_session: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct TaxonomyEntry {
    pub unit: String,
    pub boundary: String,
    pub aggregation: String,
}

pub fn byte_taxonomy() -> BTreeMap<ByteClass, TaxonomyEntry> {
    use ByteClass::*;
    [
        (
            SourceRead,
            "bytes returned by the exact source read boundary",
            "traffic",
        ),
        (
            HostAllocation,
            "requested payload capacity after host allocation succeeds",
            "allocation",
        ),
        (
            HostWrite,
            "non-zero payload copied into host storage",
            "traffic",
        ),
        (
            ZeroFill,
            "bytes explicitly initialized to zero; excluded from host_write",
            "traffic",
        ),
        (
            H2d,
            "payload covered by a completed H2D receipt",
            "transport",
        ),
        (
            D2h,
            "payload covered by a completed D2H receipt",
            "transport",
        ),
        (
            D2d,
            "payload covered by a completed D2D receipt",
            "transport",
        ),
        (
            MmapPageIn,
            "bytes confirmed faulted into memory by an external page-fault receipt",
            "os_traffic",
        ),
        (
            VmmMap,
            "virtual bytes mapped to physical backing",
            "mapping_topology",
        ),
        (
            VmmUnmap,
            "virtual bytes unmapped from physical backing",
            "mapping_topology",
        ),
        (
            ExpertMaterialization,
            "logical selected-expert payload consumed by computation",
            "logical_work",
        ),
        (
            StateMaterialization,
            "logical state payload advanced by computation",
            "logical_work",
        ),
        (
            ScratchJournal,
            "scratch/checkpoint/journal payload touched",
            "logical_work",
        ),
    ]
    .into_iter()
    .map(|(class, boundary, aggregation)| {
        (
            class,
            TaxonomyEntry {
                unit: "bytes".to_string(),
                boundary: boundary.to_string(),
                aggregation: aggregation.to_string(),
            },
        )
    })
    .collect()
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ByteBuckets {
    pub useful: BTreeMap<ByteClass, u64>,
    pub failed: BTreeMap<ByteClass, u64>,
    pub rolled_back: BTreeMap<ByteClass, u64>,
    pub quarantined: BTreeMap<ByteClass, u64>,
}

impl ByteBuckets {
    fn bucket_mut(
        &mut self,
        disposition: Option<FailureDisposition>,
    ) -> &mut BTreeMap<ByteClass, u64> {
        match disposition {
            None => &mut self.useful,
            Some(FailureDisposition::Failed) => &mut self.failed,
            Some(FailureDisposition::RolledBack) => &mut self.rolled_back,
            Some(FailureDisposition::Quarantined) => &mut self.quarantined,
        }
    }

    fn checked_add(
        &mut self,
        class: ByteClass,
        bytes: u64,
        disposition: Option<FailureDisposition>,
    ) -> Result<()> {
        let bucket = self.bucket_mut(disposition);
        let current = bucket.get(&class).copied().unwrap_or(0);
        let updated = current.checked_add(bytes).with_context(|| {
            format!(
                "{class:?} counter exhausted at {current} + {bytes}; refusing to wrap or saturate"
            )
        })?;
        bucket.insert(class, updated);
        Ok(())
    }

    pub fn value(&self, class: ByteClass) -> u64 {
        self.useful.get(&class).copied().unwrap_or(0)
    }

    pub fn non_useful_value(&self, class: ByteClass) -> u128 {
        u128::from(self.failed.get(&class).copied().unwrap_or(0))
            + u128::from(self.rolled_back.get(&class).copied().unwrap_or(0))
            + u128::from(self.quarantined.get(&class).copied().unwrap_or(0))
    }

    pub fn checked_merge(&mut self, other: &Self) -> Result<()> {
        for (disposition, source) in [
            (None, &other.useful),
            (Some(FailureDisposition::Failed), &other.failed),
            (Some(FailureDisposition::RolledBack), &other.rolled_back),
            (Some(FailureDisposition::Quarantined), &other.quarantined),
        ] {
            for (&class, &bytes) in source {
                self.checked_add(class, bytes, disposition)?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct PhaseAccounting {
    pub bytes: ByteBuckets,
    /// Subset of `bytes` attributable to the explicitly enabled FreeToken
    /// capability. The absent arm must keep every bucket empty.
    pub feature_bytes: ByteBuckets,
    pub counters: BTreeMap<CounterClass, u64>,
}

impl PhaseAccounting {
    fn checked_add_counter(&mut self, class: CounterClass, amount: u64) -> Result<()> {
        let current = self.counters.get(&class).copied().unwrap_or(0);
        self.counters.insert(
            class,
            current.checked_add(amount).with_context(|| {
                format!("{class:?} counter exhausted at {current} + {amount}; refusing to wrap")
            })?,
        );
        Ok(())
    }

    pub fn counter(&self, class: CounterClass) -> u64 {
        self.counters.get(&class).copied().unwrap_or(0)
    }

    fn checked_merge(&mut self, other: &Self) -> Result<()> {
        self.bytes.checked_merge(&other.bytes)?;
        self.feature_bytes.checked_merge(&other.feature_bytes)?;
        for (&class, &value) in &other.counters {
            self.checked_add_counter(class, value)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct StagedByteEvent {
    class: ByteClass,
    bytes: u64,
    feature_specific: bool,
}

#[derive(Clone, Debug)]
struct PendingSubmission {
    phase: Phase,
    bytes: Vec<StagedByteEvent>,
    counters: Vec<(CounterClass, u64)>,
}

/// Unforgeable-by-safe-code handle for one ledger instance. It is benchmark
/// authority only and is never consumed by production placement or execution.
#[derive(Debug)]
pub struct LedgerAuthority {
    identity: ScopeIdentity,
    nonce: u64,
}

/// Session-owned deterministic ledger. There is no process-global state.
#[derive(Debug)]
pub struct ScopedLedger {
    identity: ScopeIdentity,
    nonce: u64,
    epoch: u64,
    next_submission: u64,
    pending: BTreeMap<u64, PendingSubmission>,
    phases: BTreeMap<Phase, PhaseAccounting>,
}

impl ScopedLedger {
    pub fn new(identity: ScopeIdentity) -> (Self, LedgerAuthority) {
        let nonce = scope_nonce(&identity);
        (
            Self {
                identity: identity.clone(),
                nonce,
                epoch: 0,
                next_submission: 1,
                pending: BTreeMap::new(),
                phases: BTreeMap::new(),
            },
            LedgerAuthority { identity, nonce },
        )
    }

    fn authorize(&self, authority: &LedgerAuthority) -> Result<()> {
        ensure!(
            authority.identity == self.identity && authority.nonce == self.nonce,
            "measurement authority is foreign to provider={}, device={}, executor={}, \
             generation={}, logical_session={}",
            self.identity.provider,
            self.identity.device,
            self.identity.executor,
            self.identity.generation,
            self.identity.logical_session
        );
        Ok(())
    }

    pub fn begin_submission(&mut self, authority: &LedgerAuthority, phase: Phase) -> Result<u64> {
        self.authorize(authority)?;
        let id = self.next_submission;
        self.next_submission = self
            .next_submission
            .checked_add(1)
            .context("submission identity space exhausted; refusing ABA reuse")?;
        let previous = self.pending.insert(
            id,
            PendingSubmission {
                phase,
                bytes: Vec::new(),
                counters: vec![(CounterClass::Submissions, 1)],
            },
        );
        ensure!(previous.is_none(), "submission identity {id} was reused");
        Ok(id)
    }

    pub fn stage_bytes(
        &mut self,
        authority: &LedgerAuthority,
        submission: u64,
        class: ByteClass,
        bytes: u64,
        feature_specific: bool,
    ) -> Result<()> {
        self.authorize(authority)?;
        let pending = self.pending.get_mut(&submission).with_context(|| {
            format!("submission {submission} is not active; it was committed, failed, or foreign")
        })?;
        pending.bytes.push(StagedByteEvent {
            class,
            bytes,
            feature_specific,
        });
        Ok(())
    }

    pub fn stage_counter(
        &mut self,
        authority: &LedgerAuthority,
        submission: u64,
        class: CounterClass,
        amount: u64,
    ) -> Result<()> {
        self.authorize(authority)?;
        self.pending
            .get_mut(&submission)
            .with_context(|| format!("submission {submission} is not active"))?
            .counters
            .push((class, amount));
        Ok(())
    }

    /// Publish a submission after its stream/event completion boundary.
    pub fn commit_submission(
        &mut self,
        authority: &LedgerAuthority,
        submission: u64,
    ) -> Result<()> {
        self.finish_submission(authority, submission, None)
    }

    /// Publish attempted work only into a non-useful bucket.
    pub fn fail_submission(
        &mut self,
        authority: &LedgerAuthority,
        submission: u64,
        disposition: FailureDisposition,
    ) -> Result<()> {
        self.finish_submission(authority, submission, Some(disposition))
    }

    fn finish_submission(
        &mut self,
        authority: &LedgerAuthority,
        submission: u64,
        disposition: Option<FailureDisposition>,
    ) -> Result<()> {
        self.authorize(authority)?;
        let pending = self.pending.remove(&submission).with_context(|| {
            format!("submission {submission} is not active; completion cannot be counted twice")
        })?;
        let phase = self.phases.entry(pending.phase).or_default();
        for event in pending.bytes {
            phase
                .bytes
                .checked_add(event.class, event.bytes, disposition)?;
            if event.feature_specific {
                phase
                    .feature_bytes
                    .checked_add(event.class, event.bytes, disposition)?;
            }
        }
        for (class, amount) in pending.counters {
            phase.checked_add_counter(class, amount)?;
        }
        match disposition {
            Some(FailureDisposition::RolledBack) => {
                phase.checked_add_counter(CounterClass::Rollbacks, 1)?;
            }
            Some(FailureDisposition::Quarantined) => {
                phase.checked_add_counter(CounterClass::Quarantines, 1)?;
            }
            _ => {}
        }
        Ok(())
    }

    pub fn snapshot(&self, authority: &LedgerAuthority) -> Result<Vec<PhaseRecord>> {
        self.authorize(authority)?;
        ensure!(
            self.pending.is_empty(),
            "snapshot refused with {} in-flight submission(s); wait for completion or roll back",
            self.pending.len()
        );
        Ok(self
            .phases
            .iter()
            .map(|(&phase, accounting)| PhaseRecord {
                phase,
                accounting: accounting.clone(),
            })
            .collect())
    }

    pub fn reset(&mut self, authority: &LedgerAuthority) -> Result<()> {
        self.authorize(authority)?;
        ensure!(
            self.pending.is_empty(),
            "reset refused with {} in-flight submission(s); this prevents reset/completion TOCTOU",
            self.pending.len()
        );
        self.epoch = self
            .epoch
            .checked_add(1)
            .context("measurement epoch exhausted; refusing to wrap")?;
        self.phases.clear();
        Ok(())
    }
}

fn scope_nonce(identity: &ScopeIdentity) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(identity.provider.to_le_bytes());
    hasher.update(identity.device.to_le_bytes());
    hasher.update(identity.executor.to_le_bytes());
    hasher.update(identity.generation.to_le_bytes());
    hasher.update(identity.logical_session.to_le_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(digest[..8].try_into().expect("SHA-256 has eight bytes"))
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct PhaseRecord {
    pub phase: Phase,
    pub accounting: PhaseAccounting,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct QuantizationLayout {
    pub storage: String,
    pub block_elements: u32,
    pub payload_bits: u8,
    pub scale_bytes_per_block: u8,
    pub zero_point_bytes_per_block: u8,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ExpertBankSpec {
    pub bank: String,
    pub dimensions: BTreeMap<String, u64>,
    pub expert_count: u32,
    /// Exact synthetic source extent for one expert. It is not inferred from
    /// `dimensions`; dimensions are descriptive typed metadata.
    pub bytes_per_expert: u64,
    pub cache_slots: u32,
    pub expert_groups: u32,
    pub quantization: QuantizationLayout,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct StateGroupSpec {
    pub group: String,
    pub dimensions: BTreeMap<String, u64>,
    pub persistent_bytes: u64,
    pub carry_bytes: u64,
    pub scratch_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RouteStep {
    pub phase: Phase,
    pub sequence_position: u64,
    /// One selected-expert list per bank, in bank order. Each list has exactly
    /// `batch * top_k` entries and may repeat experts across batch rows.
    pub selections: Vec<Vec<u32>>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct WorkloadSpec {
    pub schema: String,
    pub label: String,
    pub fixture_limit: String,
    pub batch: u32,
    pub top_k: u32,
    pub banks: Vec<ExpertBankSpec>,
    pub state_groups: Vec<StateGroupSpec>,
    pub routes: Vec<RouteStep>,
}

impl WorkloadSpec {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == WORKLOAD_SCHEMA,
            "workload schema must be {WORKLOAD_SCHEMA}, got {}",
            self.schema
        );
        ensure!(self.batch > 0, "workload batch must be greater than zero");
        ensure!(self.top_k > 0, "workload top_k must be greater than zero");
        ensure!(
            !self.banks.is_empty(),
            "workload must contain at least one expert bank"
        );
        ensure!(
            !self.state_groups.is_empty(),
            "workload must contain at least one state group"
        );
        ensure!(!self.routes.is_empty(), "workload route census is empty");
        let selected_per_bank = self
            .batch
            .checked_mul(self.top_k)
            .context("batch * top_k overflowed u32")? as usize;
        for bank in &self.banks {
            ensure!(
                bank.expert_count > 0,
                "bank '{}' expert_count must be greater than zero",
                bank.bank
            );
            ensure!(
                bank.bytes_per_expert > 0,
                "bank '{}' bytes_per_expert must be greater than zero",
                bank.bank
            );
            ensure!(
                self.top_k <= bank.expert_count,
                "workload top_k {} exceeds bank '{}' expert_count {}",
                self.top_k,
                bank.bank,
                bank.expert_count
            );
            ensure!(
                bank.cache_slots > 0 && bank.cache_slots <= bank.expert_count,
                "bank '{}' cache_slots {} must be in 1..={}",
                bank.bank,
                bank.cache_slots,
                bank.expert_count
            );
            ensure!(
                bank.expert_groups > 0 && bank.expert_groups <= bank.expert_count,
                "bank '{}' expert_groups {} must be in 1..={}",
                bank.bank,
                bank.expert_groups,
                bank.expert_count
            );
        }
        for state in &self.state_groups {
            ensure!(
                state.persistent_bytes > 0,
                "state group '{}' persistent_bytes must be greater than zero",
                state.group
            );
        }
        let required_phases = [
            Phase::Prefill,
            Phase::DirectWarmup,
            Phase::Replay,
            Phase::DecodeSteady,
        ];
        for required in required_phases {
            ensure!(
                self.routes.iter().any(|step| step.phase == required),
                "workload has no {required:?} route; phase comparisons would be vacuous"
            );
        }
        for (step_index, step) in self.routes.iter().enumerate() {
            ensure!(
                matches!(
                    step.phase,
                    Phase::Prefill | Phase::DirectWarmup | Phase::Replay | Phase::DecodeSteady
                ),
                "route step {step_index} uses non-execution phase {:?}",
                step.phase
            );
            ensure!(
                step.selections.len() == self.banks.len(),
                "route step {step_index} has {} bank selections, expected {}",
                step.selections.len(),
                self.banks.len()
            );
            for (bank_index, selected) in step.selections.iter().enumerate() {
                ensure!(
                    selected.len() == selected_per_bank,
                    "route step {step_index} bank '{}' selected {} experts, expected batch {} * \
                     top_k {} = {selected_per_bank}",
                    self.banks[bank_index].bank,
                    selected.len(),
                    self.batch,
                    self.top_k
                );
                for &expert in selected {
                    ensure!(
                        expert < self.banks[bank_index].expert_count,
                        "route step {step_index} bank '{}' expert {expert} is outside 0..{}",
                        self.banks[bank_index].bank,
                        self.banks[bank_index].expert_count
                    );
                }
            }
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String> {
        let bytes = serde_json::to_vec(self).context("serialize workload for stable digest")?;
        Ok(hex_digest(&bytes))
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum SyntheticFixture {
    DeepseekLike,
    Glm52Like,
}

pub fn synthetic_workload(fixture: SyntheticFixture) -> WorkloadSpec {
    let (label, batch, top_k, banks, state_groups) = match fixture {
        SyntheticFixture::DeepseekLike => (
            "synthetic-many-expert-hybrid-attention",
            2,
            8,
            vec![
                bank(
                    "routed_ffn_primary",
                    256,
                    16 * 1024 * 1024,
                    24,
                    8,
                    "int4-block32",
                ),
                bank(
                    "routed_ffn_auxiliary",
                    256,
                    2 * 1024 * 1024,
                    32,
                    8,
                    "fp8-block64",
                ),
            ],
            vec![
                state_group(
                    "compressed_attention",
                    8 * 1024 * 1024,
                    256 * 1024,
                    512 * 1024,
                ),
                state_group(
                    "dense_attention_ring",
                    16 * 1024 * 1024,
                    128 * 1024,
                    256 * 1024,
                ),
            ],
        ),
        SyntheticFixture::Glm52Like => (
            "synthetic-grouped-many-expert-recurrent",
            2,
            8,
            vec![
                bank(
                    "grouped_ffn_primary",
                    160,
                    12 * 1024 * 1024,
                    16,
                    10,
                    "int4-block32",
                ),
                bank(
                    "grouped_ffn_gate",
                    160,
                    3 * 1024 * 1024,
                    24,
                    10,
                    "fp8-block64",
                ),
                bank("shared_ffn", 32, 1024 * 1024, 8, 4, "int4-block64"),
            ],
            vec![
                state_group("recurrent_carry", 6 * 1024 * 1024, 384 * 1024, 512 * 1024),
                state_group("attention_state", 4 * 1024 * 1024, 192 * 1024, 256 * 1024),
                state_group(
                    "multimodal_temporal",
                    2 * 1024 * 1024,
                    128 * 1024,
                    256 * 1024,
                ),
            ],
        ),
    };
    WorkloadSpec {
        schema: WORKLOAD_SCHEMA.to_string(),
        label: label.to_string(),
        fixture_limit:
            "Synthetic structural fixture only: exact declared byte extents and routes; \
                        not an exported checkpoint, model-quality run, or full-model E2E claim."
                .to_string(),
        batch,
        top_k,
        routes: deterministic_routes(batch, top_k, &banks),
        banks,
        state_groups,
    }
}

fn bank(
    name: &str,
    experts: u32,
    bytes_per_expert: u64,
    cache_slots: u32,
    groups: u32,
    storage: &str,
) -> ExpertBankSpec {
    ExpertBankSpec {
        bank: name.to_string(),
        dimensions: BTreeMap::from([
            ("experts".to_string(), experts as u64),
            ("groups".to_string(), groups as u64),
            ("typed_axis_version".to_string(), 1),
        ]),
        expert_count: experts,
        bytes_per_expert,
        cache_slots,
        expert_groups: groups,
        quantization: QuantizationLayout {
            storage: storage.to_string(),
            block_elements: if storage.contains("64") { 64 } else { 32 },
            payload_bits: if storage.starts_with("fp8") { 8 } else { 4 },
            scale_bytes_per_block: 2,
            zero_point_bytes_per_block: if storage.starts_with("int4") { 1 } else { 0 },
        },
    }
}

fn state_group(name: &str, persistent: u64, carry: u64, scratch: u64) -> StateGroupSpec {
    StateGroupSpec {
        group: name.to_string(),
        dimensions: BTreeMap::from([
            ("typed_state_group_version".to_string(), 1),
            ("sequence_axis".to_string(), 1),
        ]),
        persistent_bytes: persistent,
        carry_bytes: carry,
        scratch_bytes: scratch,
    }
}

fn deterministic_routes(batch: u32, top_k: u32, banks: &[ExpertBankSpec]) -> Vec<RouteStep> {
    let phases = [
        Phase::Prefill,
        Phase::Prefill,
        Phase::DirectWarmup,
        Phase::DirectWarmup,
        Phase::Replay,
        Phase::Replay,
        Phase::Replay,
        Phase::DecodeSteady,
        Phase::DecodeSteady,
        Phase::DecodeSteady,
        Phase::DecodeSteady,
    ];
    phases
        .into_iter()
        .enumerate()
        .map(|(step, phase)| RouteStep {
            phase,
            sequence_position: 128 + step as u64 * batch as u64,
            selections: banks
                .iter()
                .enumerate()
                .map(|(bank_index, bank)| {
                    (0..batch)
                        .flat_map(|row| {
                            (0..top_k).map(move |slot| {
                                if step % 3 == 0 {
                                    // Deliberately repeated/hot experts across rows.
                                    (slot + bank_index as u32) % bank.expert_count
                                } else {
                                    // Deterministic cold growth with grouped-bank wraparound.
                                    ((step as u32 * 17)
                                        + row * 5
                                        + slot * 3
                                        + bank_index as u32 * 11)
                                        % bank.expert_count
                                }
                            })
                        })
                        .collect()
                })
                .collect(),
        })
        .collect()
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ResidencySummary {
    pub host_expert_source_bytes: u64,
    pub final_device_expert_bytes: u64,
    pub peak_device_expert_bytes: u64,
    pub final_host_state_bytes: u64,
    pub final_device_state_bytes: u64,
    pub persistent_scratch_journal_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SemanticProof {
    pub workload_digest: String,
    pub route_digest: String,
    pub state_digest: String,
    pub output_digest: String,
    pub generated_token_ids: Vec<u32>,
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct ContractStatus {
    pub passed: bool,
    pub diagnostics: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct FailureProof {
    pub state_before: String,
    pub state_after: String,
    pub residency_before: ResidencySummary,
    pub residency_after: ResidencySummary,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RunReport {
    pub schema: String,
    pub taxonomy_schema: String,
    pub arm: Arm,
    pub identity: ScopeIdentity,
    pub workload_label: String,
    pub phases: Vec<PhaseRecord>,
    pub totals: PhaseAccounting,
    pub residency: ResidencySummary,
    pub semantics: SemanticProof,
    pub failure: Option<FailureProof>,
    pub contract: ContractStatus,
}

#[derive(Clone, Debug)]
struct CacheEntry {
    bytes: u64,
    last_use: u64,
}

#[derive(Clone, Debug, Default)]
struct SimulationState {
    cache: Vec<BTreeMap<u32, CacheEntry>>,
    device_expert_bytes: u64,
    peak_device_expert_bytes: u64,
    semantic_state: u64,
    output_tokens: Vec<u32>,
}

pub fn run_arm(workload: &WorkloadSpec, arm: Arm, identity: ScopeIdentity) -> Result<RunReport> {
    workload.validate()?;
    let (mut ledger, authority) = ScopedLedger::new(identity.clone());
    let mut state = SimulationState {
        cache: vec![BTreeMap::new(); workload.banks.len()],
        ..SimulationState::default()
    };
    let host_expert_source_bytes = workload.banks.iter().try_fold(0u64, |sum, bank| {
        let bank_bytes = bank
            .bytes_per_expert
            .checked_mul(u64::from(bank.expert_count))
            .with_context(|| format!("bank '{}' source extent overflowed", bank.bank))?;
        sum.checked_add(bank_bytes)
            .context("host expert source census overflowed")
    })?;
    let state_bytes = workload.state_groups.iter().try_fold(0u64, |sum, group| {
        sum.checked_add(group.persistent_bytes)
            .context("persistent state census overflowed")
    })?;
    let scratch_bytes = workload.state_groups.iter().try_fold(0u64, |sum, group| {
        sum.checked_add(group.scratch_bytes)
            .context("scratch/journal census overflowed")
    })?;

    record_setup(workload, arm, &mut ledger, &authority)?;
    let mut tick = 0u64;
    let mut capture_setup_recorded = false;
    for step in &workload.routes {
        if step.phase == Phase::Replay && !capture_setup_recorded {
            record_capture_setup(workload, arm, &mut ledger, &authority)?;
            capture_setup_recorded = true;
        }
        tick = tick.checked_add(1).context("route clock exhausted")?;
        record_step(
            workload,
            arm,
            step,
            tick,
            &mut state,
            &mut ledger,
            &authority,
        )?;
    }
    ensure!(
        capture_setup_recorded,
        "capture setup was never reached because the replay census is empty"
    );

    let successful_semantics = semantic_proof(workload, &state)?;
    let successful_residency = ResidencySummary {
        host_expert_source_bytes,
        final_device_expert_bytes: state.device_expert_bytes,
        peak_device_expert_bytes: state.peak_device_expert_bytes,
        final_host_state_bytes: if !feature_enabled(arm) {
            state_bytes
        } else {
            0
        },
        final_device_state_bytes: if !feature_enabled(arm) {
            0
        } else {
            state_bytes
        },
        persistent_scratch_journal_bytes: scratch_bytes,
    };

    let failure = if matches!(
        arm,
        Arm::BaselineFailureControl | Arm::OptimizedFailureControl
    ) {
        Some(run_failure_probe(
            workload,
            &mut state,
            &mut ledger,
            &authority,
            &successful_semantics,
            &successful_residency,
            feature_enabled(arm),
        )?)
    } else {
        None
    };
    let phases = ledger.snapshot(&authority)?;
    let totals = aggregate_phases(&phases)?;
    let mut report = RunReport {
        schema: RUN_SCHEMA.to_string(),
        taxonomy_schema: TAXONOMY_SCHEMA.to_string(),
        arm,
        identity,
        workload_label: workload.label.clone(),
        phases,
        totals,
        residency: successful_residency,
        semantics: successful_semantics,
        failure,
        contract: ContractStatus::default(),
    };
    report.contract.diagnostics = validate_run(&report);
    report.contract.passed = report.contract.diagnostics.is_empty();
    Ok(report)
}

fn record_setup(
    workload: &WorkloadSpec,
    arm: Arm,
    ledger: &mut ScopedLedger,
    authority: &LedgerAuthority,
) -> Result<()> {
    let feature = feature_enabled(arm);
    for group in &workload.state_groups {
        let submission = ledger.begin_submission(authority, Phase::Setup)?;
        ledger.stage_bytes(
            authority,
            submission,
            ByteClass::HostAllocation,
            group.persistent_bytes,
            feature,
        )?;
        ledger.stage_bytes(
            authority,
            submission,
            ByteClass::ZeroFill,
            group.persistent_bytes,
            feature,
        )?;
        ledger.stage_bytes(
            authority,
            submission,
            ByteClass::ScratchJournal,
            group.scratch_bytes,
            feature,
        )?;
        if feature {
            ledger.stage_bytes(
                authority,
                submission,
                ByteClass::H2d,
                group.persistent_bytes,
                true,
            )?;
            ledger.stage_bytes(
                authority,
                submission,
                ByteClass::VmmMap,
                group
                    .persistent_bytes
                    .checked_add(group.scratch_bytes)
                    .context("state setup mapping bytes overflowed")?,
                true,
            )?;
        }
        ledger.commit_submission(authority, submission)?;
    }
    Ok(())
}

fn record_capture_setup(
    workload: &WorkloadSpec,
    arm: Arm,
    ledger: &mut ScopedLedger,
    authority: &LedgerAuthority,
) -> Result<()> {
    let feature = feature_enabled(arm);
    let submission = ledger.begin_submission(authority, Phase::CaptureSetup)?;
    ledger.stage_counter(authority, submission, CounterClass::CaptureSetups, 1)?;
    if feature {
        let bytes = workload.state_groups.iter().try_fold(0u64, |sum, group| {
            sum.checked_add(group.carry_bytes)
                .context("capture journal bytes overflowed")
        })?;
        ledger.stage_bytes(
            authority,
            submission,
            ByteClass::ScratchJournal,
            bytes,
            true,
        )?;
    }
    ledger.commit_submission(authority, submission)
}

#[allow(clippy::too_many_arguments)]
fn record_step(
    workload: &WorkloadSpec,
    arm: Arm,
    step: &RouteStep,
    tick: u64,
    state: &mut SimulationState,
    ledger: &mut ScopedLedger,
    authority: &LedgerAuthority,
) -> Result<()> {
    let feature = feature_enabled(arm);
    let submission = ledger.begin_submission(authority, step.phase)?;
    ledger.stage_counter(authority, submission, CounterClass::Requests, 1)?;
    ledger.stage_counter(
        authority,
        submission,
        CounterClass::Tokens,
        u64::from(workload.batch),
    )?;
    if step.phase == Phase::Replay {
        ledger.stage_counter(authority, submission, CounterClass::Replays, 1)?;
    }

    for (bank_index, selections) in step.selections.iter().enumerate() {
        let bank = &workload.banks[bank_index];
        ledger.stage_counter(
            authority,
            submission,
            CounterClass::ExpertSelections,
            selections.len() as u64,
        )?;
        if feature {
            ledger.stage_counter(
                authority,
                submission,
                CounterClass::FeatureLookups,
                selections.len() as u64,
            )?;
        }
        let unique: BTreeSet<u32> = selections.iter().copied().collect();
        ledger.stage_counter(
            authority,
            submission,
            CounterClass::UniqueExperts,
            unique.len() as u64,
        )?;
        let logical_bytes = bank
            .bytes_per_expert
            .checked_mul(selections.len() as u64)
            .with_context(|| format!("bank '{}' logical expert bytes overflowed", bank.bank))?;
        ledger.stage_bytes(
            authority,
            submission,
            ByteClass::ExpertMaterialization,
            logical_bytes,
            feature,
        )?;

        match arm {
            Arm::BaselineAbsent | Arm::BaselineFailureControl => {
                for _expert in unique {
                    stage_expert_load(ledger, authority, submission, bank.bytes_per_expert, false)?;
                    ledger.stage_bytes(
                        authority,
                        submission,
                        ByteClass::VmmUnmap,
                        bank.bytes_per_expert,
                        false,
                    )?;
                }
            }
            Arm::Optimized | Arm::OptimizedFailureControl => {
                for expert in unique {
                    if state.cache[bank_index].contains_key(&expert) {
                        ledger.stage_counter(authority, submission, CounterClass::CacheHits, 1)?;
                        state.cache[bank_index]
                            .get_mut(&expert)
                            .expect("entry was observed")
                            .last_use = tick;
                    } else {
                        ledger.stage_counter(
                            authority,
                            submission,
                            CounterClass::CacheMisses,
                            1,
                        )?;
                        if state.cache[bank_index].len() == bank.cache_slots as usize {
                            let victim = state.cache[bank_index]
                                .iter()
                                .min_by_key(|(expert, entry)| (entry.last_use, **expert))
                                .map(|(&expert, _)| expert)
                                .context("cache capacity was reached without a victim")?;
                            let removed = state.cache[bank_index]
                                .remove(&victim)
                                .expect("selected victim is present");
                            state.device_expert_bytes = state
                                .device_expert_bytes
                                .checked_sub(removed.bytes)
                                .context("device expert residency underflowed during eviction")?;
                            ledger.stage_counter(
                                authority,
                                submission,
                                CounterClass::Evictions,
                                1,
                            )?;
                            ledger.stage_bytes(
                                authority,
                                submission,
                                ByteClass::VmmUnmap,
                                removed.bytes,
                                true,
                            )?;
                        }
                        stage_expert_load(
                            ledger,
                            authority,
                            submission,
                            bank.bytes_per_expert,
                            true,
                        )?;
                        state.cache[bank_index].insert(
                            expert,
                            CacheEntry {
                                bytes: bank.bytes_per_expert,
                                last_use: tick,
                            },
                        );
                        state.device_expert_bytes = state
                            .device_expert_bytes
                            .checked_add(bank.bytes_per_expert)
                            .context("device expert residency overflowed")?;
                        state.peak_device_expert_bytes = state
                            .peak_device_expert_bytes
                            .max(state.device_expert_bytes);
                    }
                }
            }
        }
    }

    for group in &workload.state_groups {
        ledger.stage_counter(authority, submission, CounterClass::StateUpdates, 1)?;
        ledger.stage_bytes(
            authority,
            submission,
            ByteClass::StateMaterialization,
            group.persistent_bytes,
            feature,
        )?;
        if !feature {
            ledger.stage_bytes(
                authority,
                submission,
                ByteClass::D2h,
                group.persistent_bytes,
                false,
            )?;
            ledger.stage_bytes(
                authority,
                submission,
                ByteClass::HostWrite,
                group.persistent_bytes,
                false,
            )?;
            ledger.stage_bytes(
                authority,
                submission,
                ByteClass::H2d,
                group.persistent_bytes,
                false,
            )?;
        } else if step.phase == Phase::Replay {
            ledger.stage_bytes(
                authority,
                submission,
                ByteClass::D2d,
                group.carry_bytes,
                true,
            )?;
            ledger.stage_bytes(
                authority,
                submission,
                ByteClass::ScratchJournal,
                group.carry_bytes,
                true,
            )?;
        }
    }

    state.semantic_state = semantic_step(state.semantic_state, step);
    state.output_tokens.extend((0..workload.batch).map(|row| {
        (state.semantic_state as u32)
            .wrapping_add(row)
            .rotate_left((step.sequence_position % 31) as u32)
    }));
    ledger.commit_submission(authority, submission)
}

fn stage_expert_load(
    ledger: &mut ScopedLedger,
    authority: &LedgerAuthority,
    submission: u64,
    bytes: u64,
    feature: bool,
) -> Result<()> {
    for class in [
        ByteClass::SourceRead,
        ByteClass::HostAllocation,
        ByteClass::HostWrite,
        ByteClass::H2d,
        ByteClass::VmmMap,
    ] {
        ledger.stage_bytes(authority, submission, class, bytes, feature)?;
    }
    Ok(())
}

fn run_failure_probe(
    workload: &WorkloadSpec,
    state: &mut SimulationState,
    ledger: &mut ScopedLedger,
    authority: &LedgerAuthority,
    semantics: &SemanticProof,
    residency: &ResidencySummary,
    feature: bool,
) -> Result<FailureProof> {
    let bank = workload
        .banks
        .first()
        .context("failure probe requires an expert bank")?;
    let rolled_back = ledger.begin_submission(authority, Phase::Failure)?;
    stage_expert_load(
        ledger,
        authority,
        rolled_back,
        bank.bytes_per_expert,
        feature,
    )?;
    ledger.fail_submission(authority, rolled_back, FailureDisposition::RolledBack)?;

    let quarantined = ledger.begin_submission(authority, Phase::Failure)?;
    let quarantine_bytes = workload
        .state_groups
        .first()
        .context("failure probe requires a state group")?
        .scratch_bytes;
    ledger.stage_bytes(
        authority,
        quarantined,
        ByteClass::VmmMap,
        quarantine_bytes,
        feature,
    )?;
    ledger.stage_bytes(
        authority,
        quarantined,
        ByteClass::ScratchJournal,
        quarantine_bytes,
        feature,
    )?;
    ledger.fail_submission(authority, quarantined, FailureDisposition::Quarantined)?;

    // The failed operations deliberately do not mutate semantic state or the
    // useful-residency gauges.
    let after = semantic_proof(workload, state)?;
    Ok(FailureProof {
        state_before: semantics.state_digest.clone(),
        state_after: after.state_digest,
        residency_before: residency.clone(),
        residency_after: residency.clone(),
    })
}

fn semantic_step(mut state: u64, step: &RouteStep) -> u64 {
    state ^= step.sequence_position.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    for selected in &step.selections {
        for &expert in selected {
            state = state.rotate_left(9) ^ u64::from(expert).wrapping_mul(0x1000_0000_01b3);
        }
    }
    state
}

fn semantic_proof(workload: &WorkloadSpec, state: &SimulationState) -> Result<SemanticProof> {
    let route_bytes =
        serde_json::to_vec(&workload.routes).context("serialize routes for stable digest")?;
    Ok(SemanticProof {
        workload_digest: workload.digest()?,
        route_digest: hex_digest(&route_bytes),
        state_digest: hex_digest(&state.semantic_state.to_le_bytes()),
        output_digest: hex_digest(
            &state
                .output_tokens
                .iter()
                .flat_map(|token| token.to_le_bytes())
                .collect::<Vec<_>>(),
        ),
        generated_token_ids: state.output_tokens.clone(),
    })
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn aggregate_phases(phases: &[PhaseRecord]) -> Result<PhaseAccounting> {
    let mut total = PhaseAccounting::default();
    for phase in phases {
        total.checked_merge(&phase.accounting)?;
    }
    Ok(total)
}

pub fn validate_run(report: &RunReport) -> Vec<String> {
    let mut errors = Vec::new();
    if report.schema != RUN_SCHEMA {
        errors.push(format!(
            "run schema must be {RUN_SCHEMA}, got {}",
            report.schema
        ));
    }
    if report.taxonomy_schema != TAXONOMY_SCHEMA {
        errors.push(format!(
            "taxonomy schema must be {TAXONOMY_SCHEMA}, got {}",
            report.taxonomy_schema
        ));
    }
    if report.totals.counter(CounterClass::Tokens) == 0 {
        errors.push("token census is empty".to_string());
    }
    if report.totals.counter(CounterClass::ExpertSelections) == 0 {
        errors.push("expert-selection census is empty".to_string());
    }
    if report.semantics.generated_token_ids.is_empty() {
        errors.push("generated token output is empty".to_string());
    }
    for required in [
        Phase::Setup,
        Phase::Prefill,
        Phase::DirectWarmup,
        Phase::CaptureSetup,
        Phase::Replay,
        Phase::DecodeSteady,
    ] {
        if !report.phases.iter().any(|phase| phase.phase == required) {
            errors.push(format!("required phase {required:?} is absent"));
        }
    }
    match report.arm {
        Arm::BaselineAbsent | Arm::BaselineFailureControl => {
            if report.totals.counter(CounterClass::FeatureLookups) != 0 {
                errors.push("absent arm performed FreeToken feature lookups".to_string());
            }
            if report.totals.feature_bytes != ByteBuckets::default() {
                errors.push("absent arm recorded feature-specific bytes".to_string());
            }
        }
        Arm::Optimized | Arm::OptimizedFailureControl => {
            if report.totals.counter(CounterClass::FeatureLookups) == 0 {
                errors.push("optimized arm performed zero FreeToken lookups".to_string());
            }
            if report.totals.feature_bytes.value(ByteClass::H2d) == 0 {
                errors.push("optimized positive control committed zero H2D bytes".to_string());
            }
        }
    }
    if matches!(
        report.arm,
        Arm::BaselineFailureControl | Arm::OptimizedFailureControl
    ) {
        let Some(failure) = &report.failure else {
            errors.push("failure-control arm omitted its rollback proof".to_string());
            return errors;
        };
        if failure.state_before != failure.state_after {
            errors.push("failed work changed semantic state".to_string());
        }
        if failure.residency_before != failure.residency_after {
            errors.push("failed work changed useful residency".to_string());
        }
        let failure_phase = report
            .phases
            .iter()
            .find(|phase| phase.phase == Phase::Failure);
        match failure_phase {
            Some(phase) => {
                if phase.accounting.bytes.value(ByteClass::H2d) != 0 {
                    errors.push("failed H2D was counted as useful committed traffic".to_string());
                }
                if phase.accounting.bytes.non_useful_value(ByteClass::H2d) == 0 {
                    errors.push("failure control recorded zero failed/rolled-back H2D".to_string());
                }
                if phase.accounting.counter(CounterClass::Quarantines) == 0 {
                    errors.push("failure control recorded zero quarantine events".to_string());
                }
            }
            None => errors.push("failure-control phase is absent".to_string()),
        }
    }
    errors
}

fn feature_enabled(arm: Arm) -> bool {
    matches!(arm, Arm::Optimized | Arm::OptimizedFailureControl)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ByteDelta {
    pub phase: Phase,
    pub class: ByteClass,
    pub baseline: u64,
    pub optimized: u64,
    /// Decimal string avoids narrowing an exact u64 difference into i64.
    pub optimized_minus_baseline: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ComparisonReport {
    pub schema: String,
    pub taxonomy_schema: String,
    pub taxonomy: BTreeMap<ByteClass, TaxonomyEntry>,
    pub workload: WorkloadSpec,
    pub baseline: RunReport,
    pub optimized: RunReport,
    pub baseline_failure_control: RunReport,
    pub optimized_failure_control: RunReport,
    pub deterministic_deltas: Vec<ByteDelta>,
    pub contract: ContractStatus,
}

pub fn run_comparison(workload: WorkloadSpec) -> Result<ComparisonReport> {
    workload.validate()?;
    let baseline = run_arm(
        &workload,
        Arm::BaselineAbsent,
        ScopeIdentity {
            provider: 1,
            device: 0,
            executor: 1,
            generation: 1,
            logical_session: 1,
        },
    )?;
    let optimized = run_arm(
        &workload,
        Arm::Optimized,
        ScopeIdentity {
            provider: 2,
            device: 0,
            executor: 2,
            generation: 1,
            logical_session: 2,
        },
    )?;
    let baseline_failure_control = run_arm(
        &workload,
        Arm::BaselineFailureControl,
        ScopeIdentity {
            provider: 3,
            device: 0,
            executor: 3,
            generation: 1,
            logical_session: 3,
        },
    )?;
    let optimized_failure_control = run_arm(
        &workload,
        Arm::OptimizedFailureControl,
        ScopeIdentity {
            provider: 4,
            device: 0,
            executor: 4,
            generation: 1,
            logical_session: 4,
        },
    )?;
    let deterministic_deltas = deltas(&baseline, &optimized);
    let mut report = ComparisonReport {
        schema: COMPARISON_SCHEMA.to_string(),
        taxonomy_schema: TAXONOMY_SCHEMA.to_string(),
        taxonomy: byte_taxonomy(),
        workload,
        baseline,
        optimized,
        baseline_failure_control,
        optimized_failure_control,
        deterministic_deltas,
        contract: ContractStatus::default(),
    };
    report.contract.diagnostics = validate_comparison(&report);
    report.contract.passed = report.contract.diagnostics.is_empty();
    Ok(report)
}

fn deltas(baseline: &RunReport, optimized: &RunReport) -> Vec<ByteDelta> {
    let baseline_phases: BTreeMap<_, _> = baseline
        .phases
        .iter()
        .map(|phase| (phase.phase, &phase.accounting))
        .collect();
    let optimized_phases: BTreeMap<_, _> = optimized
        .phases
        .iter()
        .map(|phase| (phase.phase, &phase.accounting))
        .collect();
    let mut output = Vec::new();
    for phase in [
        Phase::Setup,
        Phase::Prefill,
        Phase::DirectWarmup,
        Phase::CaptureSetup,
        Phase::Replay,
        Phase::DecodeSteady,
    ] {
        for class in [
            ByteClass::SourceRead,
            ByteClass::HostAllocation,
            ByteClass::HostWrite,
            ByteClass::ZeroFill,
            ByteClass::H2d,
            ByteClass::D2h,
            ByteClass::D2d,
            ByteClass::MmapPageIn,
            ByteClass::VmmMap,
            ByteClass::VmmUnmap,
            ByteClass::ExpertMaterialization,
            ByteClass::StateMaterialization,
            ByteClass::ScratchJournal,
        ] {
            let baseline = baseline_phases
                .get(&phase)
                .map_or(0, |accounting| accounting.bytes.value(class));
            let optimized = optimized_phases
                .get(&phase)
                .map_or(0, |accounting| accounting.bytes.value(class));
            output.push(ByteDelta {
                phase,
                class,
                baseline,
                optimized,
                optimized_minus_baseline: if optimized >= baseline {
                    (optimized - baseline).to_string()
                } else {
                    format!("-{}", baseline - optimized)
                },
            });
        }
    }
    output
}

pub fn validate_comparison(report: &ComparisonReport) -> Vec<String> {
    let mut errors = Vec::new();
    if report.schema != COMPARISON_SCHEMA {
        errors.push(format!(
            "comparison schema must be {COMPARISON_SCHEMA}, got {}",
            report.schema
        ));
    }
    for (name, run) in [
        ("baseline", &report.baseline),
        ("optimized", &report.optimized),
        ("baseline_failure_control", &report.baseline_failure_control),
        (
            "optimized_failure_control",
            &report.optimized_failure_control,
        ),
    ] {
        for error in validate_run(run) {
            errors.push(format!("{name}: {error}"));
        }
    }
    for (name, left, right) in [
        ("baseline/optimized", &report.baseline, &report.optimized),
        (
            "baseline/baseline-failure-control",
            &report.baseline,
            &report.baseline_failure_control,
        ),
        (
            "optimized/optimized-failure-control",
            &report.optimized,
            &report.optimized_failure_control,
        ),
    ] {
        if left.semantics.workload_digest != right.semantics.workload_digest {
            errors.push(format!("{name} workload digests differ"));
        }
        if left.semantics.route_digest != right.semantics.route_digest {
            errors.push(format!("{name} route digests differ"));
        }
        if left.semantics.state_digest != right.semantics.state_digest {
            errors.push(format!("{name} final state differs"));
        }
        if left.semantics.generated_token_ids != right.semantics.generated_token_ids {
            errors.push(format!("{name} generated token IDs differ"));
        }
    }
    if phase_record(&report.baseline, Phase::DecodeSteady).is_none()
        || phase_record(&report.optimized, Phase::DecodeSteady).is_none()
    {
        errors.push("steady decode phase is unavailable".to_string());
    }
    if report.baseline.totals.counter(CounterClass::FeatureLookups) != 0 {
        errors.push("default-off baseline feature lookup census is nonzero".to_string());
    }
    if report.optimized.totals.counter(CounterClass::CacheHits) == 0
        || report.optimized.totals.counter(CounterClass::CacheMisses) == 0
    {
        errors.push("optimized workload did not prove both hot and cold experts".to_string());
    }
    let baseline_failure = phase_record(&report.baseline_failure_control, Phase::Failure);
    let optimized_failure = phase_record(&report.optimized_failure_control, Phase::Failure);
    match (baseline_failure, optimized_failure) {
        (Some(baseline), Some(optimized))
            if baseline.accounting.bytes.rolled_back == optimized.accounting.bytes.rolled_back
                && baseline.accounting.bytes.quarantined
                    == optimized.accounting.bytes.quarantined => {}
        (Some(_), Some(_)) => {
            errors.push("baseline/optimized failure conditions differ".to_string())
        }
        _ => errors.push("paired failure controls are unavailable".to_string()),
    }
    errors
}

fn phase_record(report: &RunReport, phase: Phase) -> Option<&PhaseRecord> {
    report.phases.iter().find(|record| record.phase == phase)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct AggregateReport {
    pub schema: String,
    pub sessions: Vec<ScopeIdentity>,
    pub totals: PhaseAccounting,
}

/// Stable aggregation for independently-owned session reports. Sorting by the
/// complete identity makes output independent of thread completion order.
pub fn aggregate_reports(reports: &[RunReport]) -> Result<AggregateReport> {
    let mut ordered = reports.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|report| report.identity.clone());
    let mut totals = PhaseAccounting::default();
    for report in &ordered {
        totals.checked_merge(&report.totals)?;
    }
    Ok(AggregateReport {
        schema: "onnx-genai.freetoken-byte-ab.aggregate.v1".to_string(),
        sessions: ordered
            .into_iter()
            .map(|report| report.identity.clone())
            .collect(),
        totals,
    })
}

pub fn read_workload(path: &std::path::Path) -> Result<WorkloadSpec> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("read FreeToken workload {}", path.display()))?;
    let workload: WorkloadSpec = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse FreeToken workload {}", path.display()))?;
    workload.validate()?;
    Ok(workload)
}

pub fn write_report(path: &std::path::Path, report: &ComparisonReport) -> Result<()> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create FreeToken report directory {}", parent.display()))?;
    }
    std::fs::write(
        path,
        serde_json::to_vec_pretty(report).context("serialize FreeToken comparison report")?,
    )
    .with_context(|| format!("write FreeToken comparison report {}", path.display()))
}

pub fn require_passing(report: &ComparisonReport) -> Result<()> {
    if report.contract.passed {
        Ok(())
    } else {
        bail!(
            "FreeToken deterministic byte contract failed: {}",
            report.contract.diagnostics.join("; ")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_workload() -> WorkloadSpec {
        let banks = vec![bank("bank", 4, 10, 2, 2, "int4-block32")];
        WorkloadSpec {
            schema: WORKLOAD_SCHEMA.to_string(),
            label: "tiny".to_string(),
            fixture_limit: "test".to_string(),
            batch: 1,
            top_k: 1,
            banks,
            state_groups: vec![state_group("state", 4, 2, 2)],
            routes: vec![
                RouteStep {
                    phase: Phase::Prefill,
                    sequence_position: 1,
                    selections: vec![vec![0]],
                },
                RouteStep {
                    phase: Phase::DirectWarmup,
                    sequence_position: 2,
                    selections: vec![vec![1]],
                },
                RouteStep {
                    phase: Phase::Replay,
                    sequence_position: 3,
                    selections: vec![vec![0]],
                },
                RouteStep {
                    phase: Phase::DecodeSteady,
                    sequence_position: 4,
                    selections: vec![vec![2]],
                },
            ],
        }
    }

    #[test]
    fn known_transfer_bytes_are_exact_and_setup_is_separate() {
        let report = run_comparison(tiny_workload()).expect("tiny comparison");
        assert!(report.contract.passed, "{:?}", report.contract.diagnostics);
        // Baseline streams one 10-byte expert in each of four phases.
        assert_eq!(
            report.baseline.totals.bytes.value(ByteClass::SourceRead),
            40
        );
        assert_eq!(report.baseline.totals.bytes.value(ByteClass::H2d), 56);
        // State setup is 4 bytes; decode-state H2D is counted in execution
        // phases, never averaged into setup.
        let setup = report
            .optimized
            .phases
            .iter()
            .find(|phase| phase.phase == Phase::Setup)
            .expect("setup");
        assert_eq!(setup.accounting.bytes.value(ByteClass::H2d), 4);
    }

    #[test]
    fn counter_overflow_fails_closed_without_saturation() {
        let mut buckets = ByteBuckets::default();
        buckets
            .checked_add(ByteClass::H2d, u64::MAX, None)
            .expect("first add");
        let error = buckets
            .checked_add(ByteClass::H2d, 1, None)
            .expect_err("overflow must fail");
        assert!(error.to_string().contains("refusing to wrap"));
        assert_eq!(buckets.value(ByteClass::H2d), u64::MAX);
    }

    #[test]
    fn completion_is_required_and_failure_is_not_useful() {
        let identity = ScopeIdentity {
            provider: 1,
            device: 0,
            executor: 1,
            generation: 1,
            logical_session: 1,
        };
        let (mut ledger, authority) = ScopedLedger::new(identity);
        let submission = ledger
            .begin_submission(&authority, Phase::DecodeSteady)
            .unwrap();
        ledger
            .stage_bytes(&authority, submission, ByteClass::H2d, 64, true)
            .unwrap();
        assert!(ledger.snapshot(&authority).is_err());
        ledger
            .fail_submission(&authority, submission, FailureDisposition::RolledBack)
            .unwrap();
        let phase = &ledger.snapshot(&authority).unwrap()[0].accounting;
        assert_eq!(phase.bytes.value(ByteClass::H2d), 0);
        assert_eq!(phase.bytes.rolled_back[&ByteClass::H2d], 64);
    }

    #[test]
    fn reset_rejects_in_flight_work_and_clears_only_its_scope() {
        let identity = ScopeIdentity {
            provider: 7,
            device: 1,
            executor: 8,
            generation: 9,
            logical_session: 10,
        };
        let (mut ledger, authority) = ScopedLedger::new(identity);
        let submission = ledger
            .begin_submission(&authority, Phase::DecodeSteady)
            .unwrap();
        assert!(ledger.reset(&authority).is_err());
        ledger
            .fail_submission(&authority, submission, FailureDisposition::Failed)
            .unwrap();
        ledger.reset(&authority).unwrap();
        assert!(ledger.snapshot(&authority).unwrap().is_empty());
    }

    #[test]
    fn foreign_authority_cannot_snapshot_or_reset_a_sibling() {
        let (mut first, first_authority) = ScopedLedger::new(ScopeIdentity {
            provider: 1,
            device: 0,
            executor: 1,
            generation: 1,
            logical_session: 1,
        });
        let (_second, second_authority) = ScopedLedger::new(ScopeIdentity {
            provider: 1,
            device: 0,
            executor: 2,
            generation: 1,
            logical_session: 2,
        });
        assert!(
            first
                .snapshot(&second_authority)
                .unwrap_err()
                .to_string()
                .contains("foreign")
        );
        assert!(first.reset(&second_authority).is_err());
        assert!(first.snapshot(&first_authority).unwrap().is_empty());
    }

    #[test]
    fn absent_path_has_zero_feature_bytes_and_lookups() {
        let report = run_comparison(tiny_workload()).unwrap();
        assert_eq!(
            report.baseline.totals.counter(CounterClass::FeatureLookups),
            0
        );
        assert_eq!(report.baseline.totals.feature_bytes, ByteBuckets::default());
    }

    #[test]
    fn empty_or_altered_workload_is_detected() {
        let mut empty = tiny_workload();
        empty.routes.clear();
        assert!(empty.validate().unwrap_err().to_string().contains("empty"));

        let mut report = run_comparison(tiny_workload()).unwrap();
        report.optimized.semantics.generated_token_ids[0] ^= 1;
        let errors = validate_comparison(&report).join("\n");
        assert!(errors.contains("generated token IDs differ"), "{errors}");
    }

    #[test]
    fn failure_controls_preserve_state_and_separate_quarantine() {
        let report = run_comparison(tiny_workload()).unwrap();
        for failure_run in [
            &report.baseline_failure_control,
            &report.optimized_failure_control,
        ] {
            let failure = failure_run.failure.as_ref().expect("proof");
            assert_eq!(failure.state_before, failure.state_after);
            let phase = failure_run
                .phases
                .iter()
                .find(|phase| phase.phase == Phase::Failure)
                .expect("failure phase");
            assert_eq!(phase.accounting.bytes.value(ByteClass::H2d), 0);
            assert_eq!(phase.accounting.bytes.rolled_back[&ByteClass::H2d], 10);
            assert!(phase.accounting.bytes.quarantined[&ByteClass::VmmMap] > 0);
        }
        assert_eq!(
            report.baseline_failure_control.totals.feature_bytes,
            ByteBuckets::default()
        );
    }

    #[test]
    fn phase_and_json_order_are_stable() {
        let first = run_comparison(tiny_workload()).unwrap();
        let second = run_comparison(tiny_workload()).unwrap();
        assert_eq!(
            serde_json::to_vec(&first).unwrap(),
            serde_json::to_vec(&second).unwrap()
        );
    }

    #[test]
    fn sibling_sessions_aggregate_without_bleed_or_thread_order_dependence() {
        let workload = tiny_workload();
        let handles = (0..4)
            .map(|session| {
                let workload = workload.clone();
                std::thread::spawn(move || {
                    run_arm(
                        &workload,
                        Arm::Optimized,
                        ScopeIdentity {
                            provider: 1,
                            device: 0,
                            executor: session + 10,
                            generation: 1,
                            logical_session: session + 20,
                        },
                    )
                    .unwrap()
                })
            })
            .collect::<Vec<_>>();
        let mut reports = handles
            .into_iter()
            .rev()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();
        let reverse = aggregate_reports(&reports).unwrap();
        reports.reverse();
        let forward = aggregate_reports(&reports).unwrap();
        assert_eq!(reverse, forward);
        assert_eq!(reverse.sessions.len(), 4);
        assert_eq!(
            reverse.totals.counter(CounterClass::Tokens),
            reports[0].totals.counter(CounterClass::Tokens) * 4
        );
    }

    #[test]
    fn structurally_truthful_presets_cover_required_workload_shapes() {
        for fixture in [SyntheticFixture::DeepseekLike, SyntheticFixture::Glm52Like] {
            let workload = synthetic_workload(fixture);
            workload.validate().unwrap();
            assert!(workload.banks.len() >= 2);
            assert!(workload.state_groups.len() >= 2);
            assert!(workload.batch > 1);
            assert!(workload.top_k > 1);
            assert!(
                workload
                    .routes
                    .iter()
                    .any(|route| route.phase == Phase::Replay)
            );
        }
    }

    #[test]
    fn property_matrix_preserves_semantics_and_checked_accounting() {
        for experts in [4, 8, 16] {
            for cache_slots in [1, 2, 3] {
                let mut workload = tiny_workload();
                workload.banks[0].expert_count = experts;
                workload.banks[0].cache_slots = cache_slots.min(experts);
                workload.banks[0].expert_groups = 1;
                for (step, route) in workload.routes.iter_mut().enumerate() {
                    route.selections[0][0] = match step {
                        0 | 1 => 0,
                        _ => step as u32 % experts,
                    };
                }
                let report = run_comparison(workload).expect("property matrix comparison");
                assert!(report.contract.passed, "{:?}", report.contract.diagnostics);
                assert_eq!(
                    report.baseline.semantics.generated_token_ids,
                    report.optimized.semantics.generated_token_ids
                );
                assert_eq!(
                    report.baseline_failure_control.semantics.state_digest,
                    report.optimized_failure_control.semantics.state_digest
                );
            }
        }
    }
}
