//! Continuous batching scheduler.
//!
//! Decides which sequences to run each iteration, managing:
//! - Admission of new requests
//! - Preemption under memory pressure
//! - Priority ordering
//! - Batch formation

pub mod byte_budget;
pub mod governor;
pub mod policy;

pub use byte_budget::{
    BudgetSnapshot, ByteBudget, ByteBudgetError, ReconfigureOutcome as ByteBudgetReconfigureOutcome,
};
pub use governor::{
    derive_kv_budget, resolve_limit, CapacityProvider, CapacityProviders, DerivedBudget,
    EvictionTier, FixedCapacity, GovernorReconfigureOutcome, GovernorSnapshot, ModelKvConfig,
    ResolvedLimits, ResourceError, ResourceGovernor, ResourceLimit, ResourceLimits, VramBreakdown,
};
pub use policy::FairSharePolicy;

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
        if self.running.len() >= self.config.max_batch_size || self.waiting.is_empty() {
            return None;
        }

        let request = self.waiting.remove(0);
        let bytes = self.estimated_bytes(request.prompt_tokens, request.max_tokens);
        if !self.try_reserve_bytes(bytes) {
            self.waiting.insert(0, request);
            return None;
        }
        self.running.push(RunningSequence {
            seq_id: request.seq_id,
            request_id: request.id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: 0,
            max_tokens: request.max_tokens,
            priority: request.priority,
            arrived_at: request.arrived_at,
            reserved_bytes: bytes,
        });
        Some(ScheduledRequest {
            request_id: request.id,
            seq_id: request.seq_id,
        })
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
                    let bytes = self.estimated_bytes(request.prompt_tokens, request.max_tokens);
                    if !self.try_reserve_bytes(bytes) {
                        self.waiting.push(request);
                        break;
                    }
                    decision.prefill.push(request.seq_id);
                    self.running.push(RunningSequence {
                        seq_id: request.seq_id,
                        request_id: request.id,
                        prompt_tokens: request.prompt_tokens,
                        generated_tokens: 0,
                        max_tokens: request.max_tokens,
                        priority: request.priority,
                        arrived_at: request.arrived_at,
                        reserved_bytes: bytes,
                    });
                }
                Candidate::Swapped(mut sequence) => {
                    let bytes = self.estimated_bytes(sequence.prompt_tokens, sequence.max_tokens);
                    if !self.try_reserve_bytes(bytes) {
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

    /// Reserve `bytes` against the shared budget. Returns `true` when there is no
    /// budget (accounting disabled) or the reservation succeeds.
    fn try_reserve_bytes(&self, bytes: u64) -> bool {
        match &self.byte_budget {
            Some(budget) => budget.try_reserve(bytes).is_ok(),
            None => true,
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
        // Budget of 250 B admits only 2 of 3 otherwise-eligible sequences.
        let budget = ByteBudget::new(250);
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
        let budget = ByteBudget::new(250);
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
        let device_budget = ByteBudget::new(250);
        let mut session_a =
            Scheduler::with_byte_budget(byte_budget_config(10), device_budget.clone());
        let mut session_b =
            Scheduler::with_byte_budget(byte_budget_config(10), device_budget.clone());

        session_a.enqueue_generate_request(1, 4, 6, Priority::Normal);
        session_a.enqueue_generate_request(2, 4, 6, Priority::Normal);
        let a_decision = session_a.schedule();
        assert_eq!(a_decision.prefill, vec![1, 2]);
        assert_eq!(device_budget.used(), 200);

        // Only 50 B remain device-wide, so session B cannot admit its 100 B request.
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
}
