//! Continuous batching scheduler.
//!
//! Decides which sequences to run each iteration, managing:
//! - Admission of new requests
//! - Preemption under memory pressure
//! - Priority ordering
//! - Batch formation

pub mod policy;

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
        }
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
        self.running.push(RunningSequence {
            seq_id: request.seq_id,
            request_id: request.id,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: 0,
            max_tokens: request.max_tokens,
            priority: request.priority,
            arrived_at: request.arrived_at,
        });
        Some(ScheduledRequest {
            request_id: request.id,
            seq_id: request.seq_id,
        })
    }

    /// Called each iteration to decide what to run.
    pub fn schedule(&mut self) -> ScheduleDecision {
        let mut decision = ScheduleDecision::default();

        // Remove completed sequences
        self.running.retain(|s| s.generated_tokens < s.max_tokens);
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
                    decision.prefill.push(request.seq_id);
                    self.running.push(RunningSequence {
                        seq_id: request.seq_id,
                        request_id: request.id,
                        prompt_tokens: request.prompt_tokens,
                        generated_tokens: 0,
                        max_tokens: request.max_tokens,
                        priority: request.priority,
                        arrived_at: request.arrived_at,
                    });
                }
                Candidate::Swapped(sequence) => {
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
        self.running.retain(|s| s.seq_id != seq_id);
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

            let victim = self.running.remove(victim_idx);
            decision.preempt.push(victim.seq_id);
            self.swapped.push(victim);
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
            PriorityPolicy::Priority | PriorityPolicy::FairShare => b
                .priority
                .cmp(&a.priority)
                .then_with(|| a.arrived_at.cmp(&b.arrived_at)),
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
}
