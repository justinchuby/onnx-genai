//! Generic independent replay-checker harness.
//!
//! This harness is **test-only infrastructure** (`REFINEMENT.md` § "Independent
//! Replay Checker"). It must never call the implementation's transition
//! functions; a protocol supplies an *independent* abstract reducer via
//! [`AbstractProtocol`], and this driver validates the envelope, resolves
//! exactly one enabled abstract action per event, applies it to an independent
//! state, evaluates invariants after every transition, and rejects leftover
//! active entries unless the trace ends at a declared crash boundary.
//!
//! The harness is protocol-agnostic: the pressure protocol, and later the
//! communicator buffer-ownership / collective-ordering protocols, all plug the
//! same driver by implementing [`AbstractProtocol`].

use std::collections::HashMap;
use std::sync::Mutex;

use crate::event::{ProtocolEvent, ProtocolTraceEvent, ProtocolTraceSink};
use crate::ids::ProtocolSourceId;

/// An independent abstract state machine for one protocol.
///
/// Implementations MUST NOT share the state-transition reducer with the
/// implementation under test; sharing only identity/event *definitions* is
/// allowed (`REFINEMENT.md`).
pub trait AbstractProtocol {
    /// The independent abstract state.
    type State;
    /// A resolved, enabled abstract action ready to apply.
    type Action;

    /// Resolves an event payload to exactly one enabled abstract action for the
    /// current state. Returns [`ActionResolution::None`] for impossible or
    /// reordered events and [`ActionResolution::Ambiguous`] if more than one
    /// abstract action is enabled for the event.
    fn resolve(&self, state: &Self::State, event: &ProtocolEvent)
    -> ActionResolution<Self::Action>;

    /// Applies a resolved action to the independent abstract state using checked
    /// arithmetic. Returns `Err` with a reason on overflow, negative headroom,
    /// duplicate identity, or snapshot mismatch.
    fn apply(&self, state: &mut Self::State, action: &Self::Action) -> Result<(), String>;

    /// Evaluates all protocol invariants after a transition.
    fn check_invariants(&self, state: &Self::State) -> Result<(), String>;

    /// Returns human-readable summaries of leftover *active* (non-terminal)
    /// entries. A clean trace must leave none.
    fn active_entries(&self, state: &Self::State) -> Vec<String>;
}

/// Outcome of resolving an event to an abstract action.
pub enum ActionResolution<A> {
    /// Exactly one enabled abstract action matched.
    Enabled(A),
    /// No enabled abstract action — impossible or reordered event.
    None(String),
    /// More than one enabled abstract action matched — ambiguous.
    Ambiguous(String),
}

/// Whether a trace is expected to end cleanly or at a modeled crash boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TraceEnd {
    /// No leftover active entries are allowed.
    Clean,
    /// Leftover active entries are permitted (modeled crash/shutdown boundary).
    CrashBoundary,
}

/// Why the collector could not deliver a lossless trace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceIntegrityError {
    /// The bounded buffer overflowed; events were lost starting at this index.
    BufferOverflow { at_index: usize },
    /// An event failed to serialize into the trace.
    SerializationFailure { at_index: usize },
}

/// Why a conformance run failed, plus the smallest offending prefix length
/// (1-based count of events consumed up to and including the offending event;
/// `0` for pre-scan integrity failures and end-of-trace leftovers).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceFailure {
    /// Smallest offending trace prefix (number of events consumed).
    pub prefix_len: usize,
    /// The specific reason.
    pub reason: FailureReason,
}

/// The specific reason a conformance run failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailureReason {
    /// The collector reported buffer overflow or serialization failure.
    TraceIntegrity(TraceIntegrityError),
    /// An event carried an unexpected contract revision.
    UnknownContractRevision { found: u32, expected: u32 },
    /// A `source_sequence` repeated within a source.
    DuplicateSourceSequence {
        source: ProtocolSourceId,
        sequence: u64,
    },
    /// A `source_sequence` moved backwards within a source.
    NonMonotonicSourceSequence {
        source: ProtocolSourceId,
        sequence: u64,
        previous: u64,
    },
    /// No enabled abstract action matched (impossible or reordered event).
    NoEnabledAction { detail: String },
    /// More than one enabled abstract action matched.
    AmbiguousAction { detail: String },
    /// Applying the action was rejected (checked-arithmetic / identity error).
    ApplyRejected { detail: String },
    /// An invariant failed after the transition.
    InvariantViolated { detail: String },
    /// The clean trace left active entries behind.
    LeftoverActive { entries: Vec<String> },
}

/// A successful conformance run summary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConformanceReport {
    /// Number of events checked.
    pub events_checked: usize,
}

/// The generic replay-checker driver.
pub struct ReplayChecker<P> {
    protocol: P,
    expected_revision: u32,
}

impl<P: AbstractProtocol> ReplayChecker<P> {
    /// Creates a checker expecting the current [`crate::CONTRACT_REVISION`].
    pub fn new(protocol: P) -> Self {
        Self {
            protocol,
            expected_revision: crate::CONTRACT_REVISION,
        }
    }

    /// Overrides the expected contract revision (used to prove that an old
    /// revision is rejected).
    pub fn with_expected_revision(mut self, revision: u32) -> Self {
        self.expected_revision = revision;
        self
    }

    /// Replays a captured trace against a fresh independent abstract state.
    pub fn run(
        &self,
        mut state: P::State,
        snapshot: &TraceSnapshot,
        end: TraceEnd,
    ) -> Result<ConformanceReport, ConformanceFailure> {
        // The collector reports its own integrity first: a lossy trace can never
        // be conformant, regardless of content.
        if let Err(integrity) = &snapshot.integrity {
            let prefix_len = match integrity {
                TraceIntegrityError::BufferOverflow { at_index }
                | TraceIntegrityError::SerializationFailure { at_index } => *at_index,
            };
            return Err(ConformanceFailure {
                prefix_len,
                reason: FailureReason::TraceIntegrity(integrity.clone()),
            });
        }

        let mut last_seq: HashMap<ProtocolSourceId, u64> = HashMap::new();

        for (index, event) in snapshot.events.iter().enumerate() {
            let prefix_len = index + 1;

            if event.contract_revision != self.expected_revision {
                return Err(ConformanceFailure {
                    prefix_len,
                    reason: FailureReason::UnknownContractRevision {
                        found: event.contract_revision,
                        expected: self.expected_revision,
                    },
                });
            }

            if let Some(prev) = last_seq.get(&event.source).copied() {
                if event.source_sequence == prev {
                    return Err(ConformanceFailure {
                        prefix_len,
                        reason: FailureReason::DuplicateSourceSequence {
                            source: event.source,
                            sequence: event.source_sequence,
                        },
                    });
                }
                if event.source_sequence < prev {
                    return Err(ConformanceFailure {
                        prefix_len,
                        reason: FailureReason::NonMonotonicSourceSequence {
                            source: event.source,
                            sequence: event.source_sequence,
                            previous: prev,
                        },
                    });
                }
            }
            last_seq.insert(event.source, event.source_sequence);

            match self.protocol.resolve(&state, &event.kind) {
                ActionResolution::Enabled(action) => {
                    if let Err(detail) = self.protocol.apply(&mut state, &action) {
                        return Err(ConformanceFailure {
                            prefix_len,
                            reason: FailureReason::ApplyRejected { detail },
                        });
                    }
                    if let Err(detail) = self.protocol.check_invariants(&state) {
                        return Err(ConformanceFailure {
                            prefix_len,
                            reason: FailureReason::InvariantViolated { detail },
                        });
                    }
                }
                ActionResolution::None(detail) => {
                    return Err(ConformanceFailure {
                        prefix_len,
                        reason: FailureReason::NoEnabledAction { detail },
                    });
                }
                ActionResolution::Ambiguous(detail) => {
                    return Err(ConformanceFailure {
                        prefix_len,
                        reason: FailureReason::AmbiguousAction { detail },
                    });
                }
            }
        }

        if end == TraceEnd::Clean {
            let leftover = self.protocol.active_entries(&state);
            if !leftover.is_empty() {
                return Err(ConformanceFailure {
                    prefix_len: 0,
                    reason: FailureReason::LeftoverActive { entries: leftover },
                });
            }
        }

        Ok(ConformanceReport {
            events_checked: snapshot.events.len(),
        })
    }
}

/// An immutable view of a captured trace plus its integrity result.
#[derive(Debug, Clone)]
pub struct TraceSnapshot {
    /// The recorded events in order.
    pub events: Vec<ProtocolTraceEvent>,
    /// `Ok` when the trace was captured losslessly.
    pub integrity: Result<(), TraceIntegrityError>,
}

/// A lossless, bounded trace collector usable as a [`ProtocolTraceSink`].
///
/// A conformance campaign installs this as the governor's sink, drives the
/// campaign, then feeds [`BoundedTraceCollector::snapshot`] into a
/// [`ReplayChecker`]. It records buffer overflow explicitly so the checker can
/// reject a lossy trace.
pub struct BoundedTraceCollector {
    inner: Mutex<CollectorInner>,
}

struct CollectorInner {
    capacity: usize,
    events: Vec<ProtocolTraceEvent>,
    overflow_at: Option<usize>,
}

impl BoundedTraceCollector {
    /// Creates a collector that overflows after `capacity` events.
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(CollectorInner {
                capacity,
                events: Vec::new(),
                overflow_at: None,
            }),
        }
    }

    /// Creates an effectively unbounded collector.
    pub fn unbounded() -> Self {
        Self::new(usize::MAX)
    }

    /// Number of events recorded so far.
    pub fn len(&self) -> usize {
        self.inner.lock().expect("collector poisoned").events.len()
    }

    /// Whether no events have been recorded.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Takes an immutable snapshot of the recorded trace and its integrity.
    pub fn snapshot(&self) -> TraceSnapshot {
        let inner = self.inner.lock().expect("collector poisoned");
        let integrity = match inner.overflow_at {
            Some(at_index) => Err(TraceIntegrityError::BufferOverflow { at_index }),
            None => Ok(()),
        };
        TraceSnapshot {
            events: inner.events.clone(),
            integrity,
        }
    }
}

impl ProtocolTraceSink for BoundedTraceCollector {
    fn record(&self, event: ProtocolTraceEvent) {
        let mut inner = self.inner.lock().expect("collector poisoned");
        if inner.events.len() >= inner.capacity {
            let at = inner.events.len();
            inner.overflow_at.get_or_insert(at);
            return;
        }
        inner.events.push(event);
    }
}
