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
#[derive(Debug)]
pub struct Request {
    pub id: u64,
    pub seq_id: SequenceId,
    pub priority: Priority,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    pub arrived_at: u64,
}

/// A sequence currently in the running batch.
#[derive(Debug)]
pub struct RunningSequence {
    pub seq_id: SequenceId,
    pub request_id: u64,
    pub generated_tokens: usize,
    pub max_tokens: usize,
    pub priority: Priority,
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

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_total_tokens: 65536,
            priority_policy: PriorityPolicy::Fcfs,
        }
    }
}

/// The continuous batching scheduler.
pub struct Scheduler {
    config: SchedulerConfig,
    waiting: Vec<Request>,
    running: Vec<RunningSequence>,
    next_request_id: u64,
    clock: u64,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            config,
            waiting: Vec::new(),
            running: Vec::new(),
            next_request_id: 0,
            clock: 0,
        }
    }

    /// Submit a new request to the scheduler.
    pub fn add_request(&mut self, seq_id: SequenceId, prompt_tokens: usize, max_tokens: usize, priority: Priority) -> u64 {
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

    /// Called each iteration to decide what to run.
    pub fn schedule(&mut self) -> ScheduleDecision {
        let mut decision = ScheduleDecision::default();

        // Remove completed sequences
        self.running.retain(|s| s.generated_tokens < s.max_tokens);

        // Decode all currently running sequences
        for seq in &self.running {
            decision.decode.push(seq.seq_id);
        }

        // Admit new sequences if budget allows
        while self.running.len() < self.config.max_batch_size && !self.waiting.is_empty() {
            // Sort by policy
            match self.config.priority_policy {
                PriorityPolicy::Fcfs => {} // already in order
                PriorityPolicy::Priority => {
                    self.waiting.sort_by(|a, b| b.priority.cmp(&a.priority));
                }
                PriorityPolicy::FairShare => {} // TODO
            }

            let request = self.waiting.remove(0);
            decision.prefill.push(request.seq_id);
            self.running.push(RunningSequence {
                seq_id: request.seq_id,
                request_id: request.id,
                generated_tokens: 0,
                max_tokens: request.max_tokens,
                priority: request.priority,
            });
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
    }

    /// Number of waiting requests.
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    /// Number of running sequences.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }
}
