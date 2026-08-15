//! Continuous batching scheduler.
//!
//! Decides which sequences to run each iteration, managing:
//! - Admission of new requests
//! - Preemption under memory pressure
//! - Priority ordering
//! - Batch formation

pub mod byte_budget;
pub mod governor;
pub mod host_lease;
pub mod policy;
pub mod pressure;

pub use byte_budget::{
    AdmissionCeiling, BudgetSnapshot, ByteBudget, ByteBudgetError, ByteBudgetReservation,
    ReconfigureOutcome as ByteBudgetReconfigureOutcome,
};
pub use governor::{
    CapacityProvider, CapacityProviders, DerivedBudget, EvictionTier, FixedCapacity,
    GovernorReconfigureOutcome, GovernorSnapshot, ModelKvConfig, ResolvedLimits, ResourceError,
    ResourceGovernor, ResourceLimit, ResourceLimits, TierSnapshot, UnknownCapacity, VramBreakdown,
    derive_kv_budget, resolve_limit,
};
pub use policy::FairSharePolicy;
pub use pressure::{
    HostAllocation, HostGovernor, HostGovernorConfig, HostLedgerSnapshot, HostPageRequest,
    HostPriority, NullPressureTraceSink, PressureState, PressureTicket, PressureTraceSink,
    TicketPoll, TimeoutOutcome,
};

use onnx_genai_kv::SequenceId;

/// Request priority level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    /// Background work (batch processing, pre-generation).
    Low = 0,
    /// Normal interactive request.
    Normal = 1,
    /// User is actively waiting (typing indicator visible).
    High = 2,
}

/// A pending request waiting to be scheduled.
#[derive(Debug, Clone)]
pub struct Request {
    pub id: u64,
    pub seq_id: SequenceId,
    pub priority: Priority,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    pub arrived_at: u64,
}

/// A single request admitted by the minimal FCFS drive loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledRequest {
    pub request_id: u64,
    pub seq_id: SequenceId,
    /// The admitted generation ceiling. This can be lower than the requested
    /// ceiling when the shared byte budget can conservatively guarantee a
    /// smaller run but not the caller's larger safety ceiling.
    pub max_tokens: usize,
    /// Details when `max_tokens` was capped below the request's original ceiling.
    pub budget_cap: Option<ScheduledBudgetCap>,
}

/// A scheduler admission cap applied to preserve conservative byte reservation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScheduledBudgetCap {
    pub requested_max_tokens: usize,
    pub admitted_max_tokens: usize,
    pub requested_bytes: u64,
    pub admitted_bytes: u64,
    pub available_bytes: u64,
}

/// Why a queued request could not be admitted.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum SchedulerAdmissionError {
    #[error(
        "scheduler admission failed: running batch is full (running {running} sequences, \
         max_batch_size {max_batch_size}); wait for a running request to finish or raise \
         SchedulerConfig::max_batch_size"
    )]
    BatchFull {
        running: usize,
        max_batch_size: usize,
    },
    #[error(
        "scheduler admission failed: KV byte budget cannot reserve even one generated token for \
         request {request_id} on sequence {seq_id}: requested {requested} B for the full ceiling \
         ({prompt_tokens} prompt + {max_tokens} max_new_tokens at {bytes_per_token} B/token), \
         minimum required {minimum_required} B ({prompt_tokens} prompt + 1 generated token), but \
         only {available} B free (used {used} B of {limit} B limit, shortfall {shortfall} B; \
         running {running}/{max_batch_size} sequences); raise --vram-limit, reduce concurrent \
         requests, shorten the prompt, or lower --max-new-tokens"
    )]
    ByteBudget {
        request_id: u64,
        seq_id: SequenceId,
        prompt_tokens: usize,
        max_tokens: usize,
        bytes_per_token: u64,
        requested: u64,
        minimum_required: u64,
        used: u64,
        limit: u64,
        available: u64,
        shortfall: u64,
        running: usize,
        max_batch_size: usize,
    },
}

/// A sequence currently in the running batch.
#[derive(Debug, Clone)]
pub struct RunningSequence {
    pub seq_id: SequenceId,
    pub request_id: u64,
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub priority: Priority,
    pub arrived_at: u64,
    /// Hot-tier KV bytes reserved for this sequence against the shared
    /// [`ByteBudget`], if byte accounting is enabled. Released when the sequence
    /// completes or is preempted to CPU, re-reserved on swap-in.
    reserved_bytes: u64,
    budget_cap: Option<ScheduledBudgetCap>,
}

/// The scheduler's decision for one iteration.
#[derive(Debug, Default)]
pub struct ScheduleDecision {
    /// New sequences to prefill this iteration.
    pub prefill: Vec<SequenceId>,
    /// Sequences continuing generation.
    pub decode: Vec<SequenceId>,
    /// Sequences to preempt (evict KV to CPU).
    pub preempt: Vec<SequenceId>,
    /// Sequences to swap back in from CPU.
    pub swap_in: Vec<SequenceId>,
}

/// Scheduler configuration.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum sequences in a single batch.
    pub max_batch_size: usize,
    /// Maximum total tokens across all sequences (KV budget).
    pub max_total_tokens: usize,
    /// Policy for ordering waiting requests.
    pub priority_policy: PriorityPolicy,
    /// Policy for interrupting lower-priority running work.
    pub preemption_policy: PreemptionPolicy,
    /// Per-model hot-tier KV cost of one token, in bytes. When set together with
    /// a shared [`ByteBudget`], admission is additionally gated on the global
    /// cross-session byte ceiling (DESIGN.md §26.4/§26.11). `None` disables byte
    /// accounting and preserves the token-only behaviour. Stays model-agnostic
    /// (RULES.md #2): the caller supplies the byte cost from model metadata.
    pub bytes_per_token: Option<u64>,
}

#[derive(Debug, Clone, Copy)]
pub enum PriorityPolicy {
    /// First-come first-served.
    Fcfs,
    /// Higher priority goes first.
    Priority,
    /// Fair share across priority levels.
    FairShare,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreemptionPolicy {
    /// Never preempt running sequences.
    Disabled,
    /// Preserve decode/KV state in place and resume later.
    Swap,
    /// Future policy: drop active KV and recompute when resumed.
    Recompute,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_total_tokens: 65536,
            priority_policy: PriorityPolicy::Fcfs,
            preemption_policy: PreemptionPolicy::Swap,
            bytes_per_token: None,
        }
    }
}

/// The continuous batching scheduler.
pub struct Scheduler {
    config: SchedulerConfig,
    waiting: Vec<Request>,
    running: Vec<RunningSequence>,
    swapped: Vec<RunningSequence>,
    next_request_id: u64,
    clock: u64,
    fair_share: FairSharePolicy,
    /// Shared, cross-session hot-tier KV byte budget. When present (and
    /// `config.bytes_per_token` is set), admission and swap-in additionally
    /// reserve bytes here so no scheduler/session can exceed the global ceiling.
    byte_budget: Option<ByteBudget>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            waiting: Vec::new(),
            running: Vec::new(),
            swapped: Vec::new(),
            next_request_id: 0,
            clock: 0,
            fair_share: FairSharePolicy::new(),
            byte_budget: None,
        }
    }

    /// Configure fair-share weights for `(Low, Normal, High)` priority classes.
    ///
    /// This only affects [`PriorityPolicy::FairShare`]. Each weight must be
    /// non-zero so every continuously backlogged class is guaranteed service.
    pub fn with_fair_share_weights(mut self, low: u32, normal: u32, high: u32) -> Self {
        self.fair_share = FairSharePolicy::with_weights(low, normal, high);
        self
    }

    /// Create a scheduler that shares a global cross-session byte budget.
    ///
    /// Pass the same [`ByteBudget`] handle (via `.clone()`) to every scheduler
    /// serving the same device so their live KV usage is accounted against one
    /// ceiling (DESIGN.md §26.11.3). Byte gating only takes effect when
    /// `config.bytes_per_token` is also set.
    pub fn with_byte_budget(config: SchedulerConfig, byte_budget: ByteBudget) -> Self {
        Self {
            byte_budget: Some(byte_budget),
            ..Self::new(config)
        }
    }

    /// Access the shared byte budget, if any.
    pub fn byte_budget(&self) -> Option<&ByteBudget> {
        self.byte_budget.as_ref()
    }

    /// Access the scheduler's configuration.
    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    /// Submit a new request to the scheduler.
    pub fn add_request(
        &mut self,
        seq_id: SequenceId,
        prompt_tokens: usize,
        max_tokens: usize,
        priority: Priority,
    ) -> u64 {
        let id = self.next_request_id;
        self.next_request_id += 1;
        self.clock += 1;
        self.waiting.push(Request {
            id,
            seq_id,
            priority,
            prompt_tokens,
            max_tokens,
            arrived_at: self.clock,
        });
        id
    }

    /// Enqueue a generation request.
    ///
    /// This is the Phase 2 engine-facing API. It currently aliases `add_request`;
    /// future continuous batching work can expand the request shape without making
    /// engine callers manipulate the lower-level scheduler queue directly.
    pub fn enqueue_generate_request(
        &mut self,
        seq_id: SequenceId,
        prompt_tokens: usize,
        max_tokens: usize,
        priority: Priority,
    ) -> u64 {
        self.add_request(seq_id, prompt_tokens, max_tokens, priority)
    }

    /// Admit one queued request using the current FCFS policy.
    ///
    /// This is intentionally single-request for the first session integration;
    /// full continuous batching will replace this with batch formation.
    pub fn drive_next_fcfs(&mut self) -> Option<ScheduledRequest> {
        self.drive_next_fcfs_result().ok().flatten()
    }

    /// Admit one queued request using FCFS and report why admission failed.
    pub fn drive_next_fcfs_result(
        &mut self,
    ) -> Result<Option<ScheduledRequest>, SchedulerAdmissionError> {
        if self.running.len() >= self.config.max_batch_size {
            return Err(SchedulerAdmissionError::BatchFull {
                running: self.running.len(),
                max_batch_size: self.config.max_batch_size,
            });
        }
        if self.waiting.is_empty() {
            return Ok(None);
        }
        let request = self.waiting.remove(0);
        let (request, bytes, budget_cap) = match self.reserve_or_cap_request(&request) {
            Ok(admitted) => admitted,
            Err(error) => {
                self.waiting.insert(0, request);
                return Err(error);
            }
        };
        self.push_admitted_request(request.clone(), bytes, budget_cap);
        Ok(Some(ScheduledRequest {
            request_id: request.id,
            seq_id: request.seq_id,
            max_tokens: request.max_tokens,
            budget_cap,
        }))
    }

    fn reserve_or_cap_request(
        &self,
        request: &Request,
    ) -> Result<(Request, u64, Option<ScheduledBudgetCap>), SchedulerAdmissionError> {
        let mut admitted = request.clone();
        let requested_bytes = self.estimated_bytes(request.prompt_tokens, request.max_tokens);

        let Some(budget) = &self.byte_budget else {
            return Ok((admitted, requested_bytes, None));
        };
        let Some(bytes_per_token) = self.config.bytes_per_token.filter(|&bytes| bytes > 0) else {
            budget
                .try_reserve(requested_bytes)
                .map_err(|error| self.admission_byte_error(request, requested_bytes, error))?;
            return Ok((admitted, requested_bytes, None));
        };

        let minimum_required = self.estimated_bytes(request.prompt_tokens, 1);
        let reservation =
            match budget.try_reserve_at_most(requested_bytes, minimum_required, bytes_per_token) {
                Ok(reservation) => reservation,
                Err(error) => {
                    return Err(self.admission_byte_error(request, requested_bytes, error));
                }
            };
        if reservation.reserved == requested_bytes {
            return Ok((admitted, requested_bytes, None));
        }

        let Some(admitted_max_tokens) =
            self.max_tokens_for_reserved_bytes(request.prompt_tokens, reservation.reserved)
        else {
            self.release_bytes(reservation.reserved);
            return Err(self.admission_byte_error(
                request,
                requested_bytes,
                ByteBudgetError {
                    requested: requested_bytes,
                    used: budget.used(),
                    limit: budget.limit(),
                    available: reservation.available_before,
                    shortfall: minimum_required.saturating_sub(reservation.available_before),
                },
            ));
        };
        admitted.max_tokens = admitted_max_tokens.min(request.max_tokens);
        Ok((
            admitted,
            reservation.reserved,
            Some(ScheduledBudgetCap {
                requested_max_tokens: request.max_tokens,
                admitted_max_tokens: admitted_max_tokens.min(request.max_tokens),
                requested_bytes,
                admitted_bytes: reservation.reserved,
                available_bytes: reservation.available_before,
            }),
        ))
    }

    fn max_tokens_for_reserved_bytes(&self, prompt_tokens: usize, reserved: u64) -> Option<usize> {
        let bytes_per_token = self.config.bytes_per_token?;
        if bytes_per_token == 0 {
            return None;
        }
        let reserved_tokens = (reserved / bytes_per_token) as usize;
        reserved_tokens
            .checked_sub(prompt_tokens)
            .filter(|&n| n > 0)
    }

    fn admission_byte_error(
        &self,
        request: &Request,
        requested_bytes: u64,
        error: ByteBudgetError,
    ) -> SchedulerAdmissionError {
        let minimum_required = self.estimated_bytes(request.prompt_tokens, 1);
        SchedulerAdmissionError::ByteBudget {
            request_id: request.id,
            seq_id: request.seq_id,
            prompt_tokens: request.prompt_tokens,
            max_tokens: request.max_tokens,
            bytes_per_token: self.config.bytes_per_token.unwrap_or(0),
            requested: requested_bytes,
            minimum_required,
            used: error.used,
            limit: error.limit,
            available: error.available,
            shortfall: minimum_required.saturating_sub(error.available),
            running: self.running.len(),
            max_batch_size: self.config.max_batch_size,
        }
    }

    fn push_admitted_request(
        &mut self,
        request: Request,
        bytes: u64,
        budget_cap: Option<ScheduledBudgetCap>,
    ) -> SequenceId {
        let seq_id = request.seq_id;
        self.running.push(RunningSequence {
            seq_id: request.seq_id,
            request_id: request.id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: 0,
            max_tokens: request.max_tokens,
            priority: request.priority,
            arrived_at: request.arrived_at,
            reserved_bytes: bytes,
            budget_cap,
        });
        seq_id
    }

    /// Called each iteration to decide what to run.
    pub fn schedule(&mut self) -> ScheduleDecision {
        let mut decision = ScheduleDecision::default();

        // Remove completed sequences, releasing their reserved KV bytes.
        let mut still_running = Vec::with_capacity(self.running.len());
        for sequence in std::mem::take(&mut self.running) {
            if sequence.generated_tokens < sequence.max_tokens {
                still_running.push(sequence);
            } else {
                self.release_bytes(sequence.reserved_bytes);
            }
        }
        self.running = still_running;
        let previously_running = self
            .running
            .iter()
            .map(|sequence| sequence.seq_id)
            .collect::<Vec<_>>();

        self.apply_preemption(&mut decision);

        // Admit new sequences if budget allows
        while self.has_capacity_for_candidate() {
            let Some(candidate) = self.pop_next_candidate() else {
                break;
            };
            match candidate {
                Candidate::Waiting(request) => {
                    let (request, bytes, budget_cap) = match self.reserve_or_cap_request(&request) {
                        Ok(admitted) => admitted,
                        Err(_) => {
                            self.waiting.push(request);
                            break;
                        }
                    };
                    let seq_id = request.seq_id;
                    self.push_admitted_request(request, bytes, budget_cap);
                    decision.prefill.push(seq_id);
                }
                Candidate::Swapped(mut sequence) => {
                    let bytes = self.estimated_bytes(sequence.prompt_tokens, sequence.max_tokens);
                    if self.try_reserve_bytes(bytes).is_err() {
                        self.swapped.push(sequence);
                        break;
                    }
                    sequence.reserved_bytes = bytes;
                    decision.swap_in.push(sequence.seq_id);
                    self.running.push(sequence);
                }
            }
        }

        // Decode sequences that were already running and were not just preempted.
        for seq in &self.running {
            if previously_running.contains(&seq.seq_id) {
                decision.decode.push(seq.seq_id);
            }
        }

        decision
    }

    /// Mark a sequence as having generated one more token.
    pub fn advance(&mut self, seq_id: SequenceId) {
        if let Some(seq) = self.running.iter_mut().find(|s| s.seq_id == seq_id) {
            seq.generated_tokens += 1;
        }
    }

    /// Mark a sequence as completed.
    pub fn complete(&mut self, seq_id: SequenceId) {
        if let Some(pos) = self.running.iter().position(|s| s.seq_id == seq_id) {
            let sequence = self.running.remove(pos);
            self.release_bytes(sequence.reserved_bytes);
        }
        // Swapped sequences already released their hot-tier bytes on preemption
        // (reserved_bytes is 0 while swapped), so removing them frees nothing.
        self.swapped.retain(|s| s.seq_id != seq_id);
    }

    /// Cancel a queued or running request and release any hot-tier reservation it
    /// owns. Engine callers use this when admission was attempted for a
    /// synchronous request but the request cannot proceed.
    pub fn cancel_request(&mut self, request_id: u64) {
        self.waiting.retain(|request| request.id != request_id);
        if let Some(pos) = self
            .running
            .iter()
            .position(|sequence| sequence.request_id == request_id)
        {
            let sequence = self.running.remove(pos);
            self.release_bytes(sequence.reserved_bytes);
        }
        self.swapped
            .retain(|sequence| sequence.request_id != request_id);
    }

    /// Number of waiting requests.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Number of running sequences.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    /// Number of preempted requests waiting to resume.
    pub fn swapped_count(&self) -> usize {
        self.swapped.len()
    }

    /// Admitted generation ceiling for a running sequence.
    pub fn running_max_tokens(&self, seq_id: SequenceId) -> Option<usize> {
        self.running
            .iter()
            .find(|sequence| sequence.seq_id == seq_id)
            .map(|sequence| sequence.max_tokens)
    }

    /// Budget cap applied to a running sequence, if admission lowered its
    /// requested generation ceiling.
    pub fn running_budget_cap(&self, seq_id: SequenceId) -> Option<ScheduledBudgetCap> {
        self.running
            .iter()
            .find(|sequence| sequence.seq_id == seq_id)
            .and_then(|sequence| sequence.budget_cap)
    }

    fn apply_preemption(&mut self, decision: &mut ScheduleDecision) {
        if matches!(self.config.preemption_policy, PreemptionPolicy::Disabled)
            || matches!(self.config.priority_policy, PriorityPolicy::FairShare)
            || self.running.is_empty()
            || self.waiting.is_empty()
        {
            return;
        }

        while !self.has_capacity_for_candidate() {
            let Some(best_waiting_idx) = self.best_waiting_index() else {
                break;
            };
            let Some(victim_idx) = self.lowest_priority_running_index() else {
                break;
            };
            if self.waiting[best_waiting_idx].priority <= self.running[victim_idx].priority {
                break;
            }

            let mut victim = self.running.remove(victim_idx);
            self.release_bytes(victim.reserved_bytes);
            victim.reserved_bytes = 0;
            decision.preempt.push(victim.seq_id);
            self.swapped.push(victim);
        }
    }

    /// Estimated hot-tier KV bytes for a sequence's worst-case footprint.
    ///
    /// Reserving the full `prompt + max_tokens` footprint up front makes byte
    /// admission conservative: once admitted, a sequence's KV growth can never
    /// push the shared budget over its ceiling. Returns 0 when byte accounting is
    /// disabled (`config.bytes_per_token` is `None`).
    fn estimated_bytes(&self, prompt_tokens: usize, max_tokens: usize) -> u64 {
        match self.config.bytes_per_token {
            Some(bytes_per_token) => {
                let footprint_tokens = prompt_tokens.saturating_add(max_tokens) as u64;
                footprint_tokens.saturating_mul(bytes_per_token)
            }
            None => 0,
        }
    }

    /// Reserve `bytes` against the shared budget. Returns `Ok(())` when there is
    /// no budget (accounting disabled) or the reservation succeeds.
    fn try_reserve_bytes(&self, bytes: u64) -> Result<(), ByteBudgetError> {
        match &self.byte_budget {
            Some(budget) => budget.try_reserve(bytes),
            None => Ok(()),
        }
    }

    /// Release `bytes` back to the shared budget, if one is present.
    fn release_bytes(&self, bytes: u64) {
        if let Some(budget) = &self.byte_budget {
            budget.release(bytes);
        }
    }
    fn has_capacity_for_candidate(&self) -> bool {
        self.running.len() < self.config.max_batch_size
            && self.running_token_budget() < self.config.max_total_tokens
    }

    fn running_token_budget(&self) -> usize {
        self.running
            .iter()
            .map(|sequence| sequence.prompt_tokens + sequence.generated_tokens)
            .sum()
    }

    fn best_waiting_index(&self) -> Option<usize> {
        self.waiting
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| self.cmp_request(a, b))
            .map(|(idx, _)| idx)
    }

    fn lowest_priority_running_index(&self) -> Option<usize> {
        self.running
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.priority
                    .cmp(&b.priority)
                    .then_with(|| b.arrived_at.cmp(&a.arrived_at))
                    .then_with(|| b.request_id.cmp(&a.request_id))
            })
            .map(|(idx, _)| idx)
    }

    fn pop_next_candidate(&mut self) -> Option<Candidate> {
        if matches!(self.config.priority_policy, PriorityPolicy::FairShare) {
            return self.pop_next_fair_share_candidate();
        }

        let waiting = self.best_waiting_index();
        let swapped = self.best_swapped_index();
        match (waiting, swapped) {
            (None, None) => None,
            (Some(idx), None) => Some(Candidate::Waiting(self.waiting.remove(idx))),
            (None, Some(idx)) => Some(Candidate::Swapped(self.swapped.remove(idx))),
            (Some(waiting_idx), Some(swapped_idx)) => {
                let waiting_key = CandidateKey::from_request(&self.waiting[waiting_idx]);
                let swapped_key = CandidateKey::from_running(&self.swapped[swapped_idx]);
                if self.cmp_candidate_key(waiting_key, swapped_key).is_lt() {
                    Some(Candidate::Waiting(self.waiting.remove(waiting_idx)))
                } else {
                    Some(Candidate::Swapped(self.swapped.remove(swapped_idx)))
                }
            }
        }
    }

    fn pop_next_fair_share_candidate(&mut self) -> Option<Candidate> {
        let selected_priority = self.fair_share.select(
            self.waiting
                .iter()
                .map(|request| request.priority)
                .chain(self.swapped.iter().map(|sequence| sequence.priority)),
        )?;

        let waiting = self
            .waiting
            .iter()
            .enumerate()
            .filter(|(_, request)| request.priority == selected_priority)
            .min_by_key(|(_, request)| (request.arrived_at, request.id))
            .map(|(index, request)| (index, request.arrived_at, request.id));
        let swapped = self
            .swapped
            .iter()
            .enumerate()
            .filter(|(_, sequence)| sequence.priority == selected_priority)
            .min_by_key(|(_, sequence)| (sequence.arrived_at, sequence.request_id))
            .map(|(index, sequence)| (index, sequence.arrived_at, sequence.request_id));

        match (waiting, swapped) {
            (Some((index, _, _)), None) => Some(Candidate::Waiting(self.waiting.remove(index))),
            (None, Some((index, _, _))) => Some(Candidate::Swapped(self.swapped.remove(index))),
            (
                Some((waiting_index, waiting_arrival, waiting_id)),
                Some((swapped_index, swapped_arrival, swapped_id)),
            ) => {
                if (waiting_arrival, waiting_id) < (swapped_arrival, swapped_id) {
                    Some(Candidate::Waiting(self.waiting.remove(waiting_index)))
                } else {
                    Some(Candidate::Swapped(self.swapped.remove(swapped_index)))
                }
            }
            (None, None) => unreachable!("selected fair-share class must have a candidate"),
        }
    }

    fn best_swapped_index(&self) -> Option<usize> {
        self.swapped
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                self.cmp_candidate_key(CandidateKey::from_running(a), CandidateKey::from_running(b))
            })
            .map(|(idx, _)| idx)
    }

    fn cmp_request(&self, a: &Request, b: &Request) -> std::cmp::Ordering {
        self.cmp_candidate_key(CandidateKey::from_request(a), CandidateKey::from_request(b))
    }

    fn cmp_candidate_key(&self, a: CandidateKey, b: CandidateKey) -> std::cmp::Ordering {
        match self.config.priority_policy {
            PriorityPolicy::Fcfs => a.arrived_at.cmp(&b.arrived_at),
            PriorityPolicy::Priority => b
                .priority
                .cmp(&a.priority)
                .then_with(|| a.arrived_at.cmp(&b.arrived_at)),
            PriorityPolicy::FairShare => a.arrived_at.cmp(&b.arrived_at),
        }
        .then_with(|| a.request_id.cmp(&b.request_id))
    }
}

enum Candidate {
    Waiting(Request),
    Swapped(RunningSequence),
}

#[derive(Clone, Copy)]
struct CandidateKey {
    priority: Priority,
    arrived_at: u64,
    request_id: u64,
}

impl CandidateKey {
    fn from_request(request: &Request) -> Self {
        Self {
            priority: request.priority,
            arrived_at: request.arrived_at,
            request_id: request.id,
        }
    }

    fn from_running(sequence: &RunningSequence) -> Self {
        Self {
            priority: sequence.priority,
            arrived_at: sequence.arrived_at,
            request_id: sequence.request_id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SchedulerConfig {
        SchedulerConfig {
            max_batch_size: 1,
            max_total_tokens: 1024,
            priority_policy: PriorityPolicy::Priority,
            preemption_policy: PreemptionPolicy::Swap,
            bytes_per_token: None,
        }
    }

    #[test]
    fn higher_priority_request_runs_before_earlier_lower_priority_request() {
        let mut scheduler = Scheduler::new(config());
        scheduler.enqueue_generate_request(10, 3, 2, Priority::Low);
        scheduler.enqueue_generate_request(20, 3, 2, Priority::High);

        let decision = scheduler.schedule();

        assert_eq!(decision.prefill, vec![20]);
        assert!(decision.decode.is_empty());
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(scheduler.running_count(), 1);
    }

    #[test]
    fn higher_priority_arrival_preempts_lower_priority_running_sequence() {
        let mut scheduler = Scheduler::new(config());
        scheduler.enqueue_generate_request(10, 3, 4, Priority::Low);
        let first = scheduler.schedule();
        assert_eq!(first.prefill, vec![10]);
        scheduler.advance(10);

        scheduler.enqueue_generate_request(20, 3, 2, Priority::High);
        let preempt = scheduler.schedule();

        assert_eq!(preempt.preempt, vec![10]);
        assert_eq!(preempt.prefill, vec![20]);
        assert!(preempt.decode.is_empty());
        assert_eq!(scheduler.running_count(), 1);
        assert_eq!(scheduler.swapped_count(), 1);

        scheduler.advance(20);
        scheduler.advance(20);
        scheduler.complete(20);
        let resume = scheduler.schedule();
        assert_eq!(resume.swap_in, vec![10]);
        assert!(resume.decode.is_empty());
    }

    #[test]
    fn fair_share_policy_is_used_for_scheduler_admission() {
        let mut fair_config = config();
        fair_config.priority_policy = PriorityPolicy::FairShare;
        fair_config.preemption_policy = PreemptionPolicy::Disabled;
        let mut scheduler = Scheduler::new(fair_config).with_fair_share_weights(1, 1, 3);

        for index in 0..100 {
            scheduler.enqueue_generate_request(1_000 + index, 1, 1, Priority::Low);
            scheduler.enqueue_generate_request(2_000 + index, 1, 1, Priority::High);
        }

        let mut low = 0;
        let mut high = 0;
        for _ in 0..40 {
            let decision = scheduler.schedule();
            let selected = decision.prefill[0];
            if selected < 2_000 {
                low += 1;
            } else {
                high += 1;
            }
            scheduler.complete(selected);
        }

        assert_eq!((low, high), (10, 30));
    }

    fn byte_budget_config(bytes_per_token: u64) -> SchedulerConfig {
        SchedulerConfig {
            // Large token/batch limits so the *byte* budget is the binding gate.
            max_batch_size: 32,
            max_total_tokens: 1 << 20,
            priority_policy: PriorityPolicy::Fcfs,
            preemption_policy: PreemptionPolicy::Disabled,
            bytes_per_token: Some(bytes_per_token),
        }
    }

    #[test]
    fn byte_budget_gates_admission_below_token_and_batch_budget() {
        // 10 bytes/token, footprint = (prompt 4 + max 6) * 10 = 100 B each.
        // Budget of 240 B admits only 2 of 3: the remaining 40 B cannot
        // conservatively cover even prompt + one generated token for seq 3.
        let budget = ByteBudget::new(240);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(10), budget.clone());
        scheduler.enqueue_generate_request(1, 4, 6, Priority::Normal);
        scheduler.enqueue_generate_request(2, 4, 6, Priority::Normal);
        scheduler.enqueue_generate_request(3, 4, 6, Priority::Normal);

        let decision = scheduler.schedule();

        assert_eq!(decision.prefill, vec![1, 2]);
        assert_eq!(scheduler.running_count(), 2);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(budget.used(), 200);
    }

    #[test]
    fn completion_releases_bytes_and_admits_waiting_sequence() {
        let budget = ByteBudget::new(240);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(10), budget.clone());
        scheduler.enqueue_generate_request(1, 4, 6, Priority::Normal);
        scheduler.enqueue_generate_request(2, 4, 6, Priority::Normal);
        scheduler.enqueue_generate_request(3, 4, 6, Priority::Normal);
        scheduler.schedule();
        assert_eq!(scheduler.running_count(), 2);

        // Freeing one running sequence returns its 100 B, admitting seq 3.
        scheduler.complete(1);
        assert_eq!(budget.used(), 100);
        let decision = scheduler.schedule();
        assert_eq!(decision.prefill, vec![3]);
        assert_eq!(budget.used(), 200);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn shared_budget_is_accounted_across_two_schedulers() {
        // One device budget shared by two sessions/models (DESIGN §26.11.3).
        let device_budget = ByteBudget::new(240);
        let mut session_a =
            Scheduler::with_byte_budget(byte_budget_config(10), device_budget.clone());
        let mut session_b =
            Scheduler::with_byte_budget(byte_budget_config(10), device_budget.clone());

        session_a.enqueue_generate_request(1, 4, 6, Priority::Normal);
        session_a.enqueue_generate_request(2, 4, 6, Priority::Normal);
        let a_decision = session_a.schedule();
        assert_eq!(a_decision.prefill, vec![1, 2]);
        assert_eq!(device_budget.used(), 200);

        // Only 40 B remain device-wide, so session B cannot admit prompt + one token.
        session_b.enqueue_generate_request(3, 4, 6, Priority::Normal);
        let b_decision = session_b.schedule();
        assert!(b_decision.prefill.is_empty());
        assert_eq!(session_b.waiting_count(), 1);
    }

    #[test]
    fn preemption_releases_hot_bytes_and_swap_in_re_reserves() {
        let budget = ByteBudget::new(150);
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_total_tokens: 1 << 20,
            priority_policy: PriorityPolicy::Priority,
            preemption_policy: PreemptionPolicy::Swap,
            bytes_per_token: Some(10),
        };
        let mut scheduler = Scheduler::with_byte_budget(config, budget.clone());

        // Low-priority sequence admitted first: footprint (2 + 4) * 10 = 60 B.
        scheduler.enqueue_generate_request(10, 2, 4, Priority::Low);
        scheduler.schedule();
        assert_eq!(budget.used(), 60);
        scheduler.advance(10);

        // High-priority arrival preempts it, releasing its hot bytes, then
        // reserves its own footprint (2 + 2) * 10 = 40 B.
        scheduler.enqueue_generate_request(20, 2, 2, Priority::High);
        let preempt = scheduler.schedule();
        assert_eq!(preempt.preempt, vec![10]);
        assert_eq!(preempt.prefill, vec![20]);
        assert_eq!(budget.used(), 40);

        // When the high-priority sequence finishes, the swapped one re-reserves.
        scheduler.advance(20);
        scheduler.advance(20);
        scheduler.complete(20);
        let resume = scheduler.schedule();
        assert_eq!(resume.swap_in, vec![10]);
        assert_eq!(budget.used(), 60);
    }

    #[test]
    fn budget_cap_survives_swap_out_and_swap_in() {
        let budget = ByteBudget::new(640);
        let config = SchedulerConfig {
            max_batch_size: 1,
            max_total_tokens: 1 << 20,
            priority_policy: PriorityPolicy::Priority,
            preemption_policy: PreemptionPolicy::Swap,
            bytes_per_token: Some(1),
        };
        let mut scheduler = Scheduler::with_byte_budget(config, budget.clone());

        scheduler.enqueue_generate_request(10, 512, 3584, Priority::Low);
        let capped = scheduler.schedule();
        assert_eq!(capped.prefill, vec![10]);
        let cap = scheduler.running_budget_cap(10).unwrap();
        assert_eq!(scheduler.running_max_tokens(10), Some(128));
        scheduler.advance(10);

        scheduler.enqueue_generate_request(20, 1, 1, Priority::High);
        let preempt = scheduler.schedule();
        assert_eq!(preempt.preempt, vec![10]);
        assert_eq!(preempt.prefill, vec![20]);

        scheduler.complete(20);
        let resume = scheduler.schedule();
        assert_eq!(resume.swap_in, vec![10]);
        assert_eq!(scheduler.running_max_tokens(10), Some(128));
        assert_eq!(scheduler.running_budget_cap(10), Some(cap));
        assert_eq!(budget.used(), 640);
    }

    #[test]
    fn reconfigure_lower_reports_overage_and_blocks_new_admissions() {
        let budget = ByteBudget::new(300);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(10), budget.clone());
        scheduler.enqueue_generate_request(1, 4, 6, Priority::Normal);
        scheduler.enqueue_generate_request(2, 4, 6, Priority::Normal);
        scheduler.schedule();
        assert_eq!(budget.used(), 200);

        // Governor turns the device budget down (DESIGN §26.11.2).
        let outcome = budget.reconfigure(150);
        assert_eq!(outcome.overage, 50);

        // Already-running sequences keep running, but nothing new is admitted.
        scheduler.enqueue_generate_request(3, 4, 6, Priority::Normal);
        let decision = scheduler.schedule();
        assert!(decision.prefill.is_empty());
        assert_eq!(scheduler.running_count(), 2);
    }

    #[test]
    fn disabled_byte_accounting_preserves_token_only_behaviour() {
        // No bytes_per_token and no budget: byte gate is inert.
        let mut scheduler = Scheduler::new(SchedulerConfig {
            max_batch_size: 4,
            max_total_tokens: 1 << 20,
            priority_policy: PriorityPolicy::Fcfs,
            preemption_policy: PreemptionPolicy::Disabled,
            bytes_per_token: None,
        });
        scheduler.enqueue_generate_request(1, 4, 6, Priority::Normal);
        scheduler.enqueue_generate_request(2, 4, 6, Priority::Normal);
        let decision = scheduler.schedule();
        assert_eq!(decision.prefill, vec![1, 2]);
        assert!(scheduler.byte_budget().is_none());
    }

    #[test]
    fn full_context_ceiling_is_capped_to_budget_preserving_conservative_reservation() {
        let budget = ByteBudget::new(640);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(1), budget.clone());
        scheduler.enqueue_generate_request(42, 512, 3584, Priority::Normal);

        let scheduled = scheduler.drive_next_fcfs_result().unwrap().unwrap();

        assert_eq!(scheduled.seq_id, 42);
        assert_eq!(scheduled.max_tokens, 128);
        assert_eq!(
            scheduled.budget_cap,
            Some(ScheduledBudgetCap {
                requested_max_tokens: 3584,
                admitted_max_tokens: 128,
                requested_bytes: 4096,
                admitted_bytes: 640,
                available_bytes: 640,
            })
        );
        assert_eq!(scheduler.running_max_tokens(42), Some(128));
        assert_eq!(scheduler.running_budget_cap(42), scheduled.budget_cap);
        assert_eq!(budget.used(), 640);
        assert_eq!(scheduler.waiting_count(), 0);
    }

    #[test]
    fn capped_reservation_uses_current_shared_budget_atomically() {
        let budget = ByteBudget::new(640);
        budget.try_reserve(100).unwrap();
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(1), budget.clone());
        scheduler.enqueue_generate_request(42, 512, 3584, Priority::Normal);

        let scheduled = scheduler.drive_next_fcfs_result().unwrap().unwrap();

        assert_eq!(scheduled.max_tokens, 28);
        assert_eq!(
            scheduled.budget_cap,
            Some(ScheduledBudgetCap {
                requested_max_tokens: 3584,
                admitted_max_tokens: 28,
                requested_bytes: 4096,
                admitted_bytes: 540,
                available_bytes: 540,
            })
        );
        assert_eq!(budget.used(), 640);
    }

    #[test]
    fn long_multi_turn_session_is_not_rejected_purely_because_ceiling_grew() {
        let budget = ByteBudget::new(1024);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(1), budget.clone());

        for (prompt_tokens, expected_max_tokens) in [(128, 896), (300, 724), (600, 424)] {
            scheduler.enqueue_generate_request(
                7,
                prompt_tokens,
                4096 - prompt_tokens,
                Priority::Normal,
            );
            let scheduled = scheduler.drive_next_fcfs_result().unwrap().unwrap();
            assert_eq!(scheduled.seq_id, 7);
            assert_eq!(scheduled.max_tokens, expected_max_tokens);
            assert_eq!(budget.used(), 1024);
            scheduler.complete(7);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn repeated_turns_release_capped_reservations_without_leaking() {
        let budget = ByteBudget::new(640);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(1), budget.clone());

        for _ in 0..5 {
            scheduler.enqueue_generate_request(7, 512, 3584, Priority::Normal);
            let scheduled = scheduler.drive_next_fcfs_result().unwrap().unwrap();
            assert_eq!(scheduled.max_tokens, 128);
            assert_eq!(scheduler.running_count(), 1);
            scheduler.complete(7);
            assert_eq!(scheduler.running_count(), 0);
            assert_eq!(scheduler.waiting_count(), 0);
            assert_eq!(budget.used(), 0);
        }
    }

    #[test]
    fn cancel_request_clears_failed_or_running_turn_state() {
        let budget = ByteBudget::new(500);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(1), budget.clone());

        let failed = scheduler.enqueue_generate_request(7, 512, 3584, Priority::Normal);
        assert!(scheduler.drive_next_fcfs_result().is_err());
        assert_eq!(scheduler.waiting_count(), 1);
        scheduler.cancel_request(failed);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(budget.used(), 0);

        budget.reconfigure(640);
        let admitted = scheduler.enqueue_generate_request(8, 512, 3584, Priority::Normal);
        scheduler.drive_next_fcfs_result().unwrap().unwrap();
        assert_eq!(budget.used(), 640);
        scheduler.cancel_request(admitted);
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn mismatched_admission_cleanup_clears_admitted_and_original_requests() {
        let budget = ByteBudget::new(1000);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(1), budget.clone());

        let stale = scheduler.enqueue_generate_request(20, 2, 2, Priority::Normal);
        let original = scheduler.enqueue_generate_request(7, 512, 3584, Priority::Normal);
        let scheduled = scheduler.drive_next_fcfs_result().unwrap().unwrap();
        assert_eq!(scheduled.request_id, stale);
        assert_eq!(scheduler.running_count(), 1);
        assert_eq!(scheduler.waiting_count(), 1);
        assert_eq!(budget.used(), 4);

        scheduler.cancel_request(scheduled.request_id);
        scheduler.cancel_request(original);

        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.waiting_count(), 0);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn admission_failure_names_cause_and_actionable_budget_numbers() {
        let budget = ByteBudget::new(500);
        let mut scheduler = Scheduler::with_byte_budget(byte_budget_config(1), budget);
        scheduler.enqueue_generate_request(9, 512, 3584, Priority::Normal);

        let err = scheduler.drive_next_fcfs_result().unwrap_err();
        let text = err.to_string();

        assert!(text.contains("KV byte budget"), "{text}");
        assert!(text.contains("requested 4096 B"), "{text}");
        assert!(text.contains("minimum required 513 B"), "{text}");
        assert!(text.contains("only 500 B free"), "{text}");
        assert!(text.contains("shortfall 13 B"), "{text}");
        assert!(text.contains("running 0/32 sequences"), "{text}");
        assert!(text.contains("raise --vram-limit"), "{text}");
        assert!(text.contains("lower --max-new-tokens"), "{text}");
    }

    #[test]
    fn batch_full_failure_names_running_count_and_limit() {
        let mut cfg = byte_budget_config(1);
        cfg.max_batch_size = 1;
        let mut scheduler = Scheduler::with_byte_budget(cfg, ByteBudget::new(10_000));
        scheduler.enqueue_generate_request(1, 4, 6, Priority::Normal);
        scheduler.drive_next_fcfs_result().unwrap().unwrap();
        scheduler.enqueue_generate_request(2, 4, 6, Priority::Normal);

        let err = scheduler.drive_next_fcfs_result().unwrap_err();
        let text = err.to_string();

        assert!(text.contains("running batch is full"), "{text}");
        assert!(text.contains("running 1 sequences"), "{text}");
        assert!(text.contains("max_batch_size 1"), "{text}");
    }
}
