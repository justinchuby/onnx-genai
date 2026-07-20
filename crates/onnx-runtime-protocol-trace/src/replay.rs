//! Protocol-agnostic replay-checker harness.

use std::collections::{HashMap, HashSet};

use crate::{ActorId, EventEnvelope, EventId, EventSequence, LogicalTimestamp};

/// A protocol-specific reducer used by [`ReplayHarness`].
///
/// Implementations own their abstract state and invariant definitions. This
/// crate only validates generic envelope ordering and drives the reducer.
pub trait ReplayReducer<P> {
    /// Independent abstract protocol state.
    type State;

    /// Applies one event to the abstract state.
    fn apply(
        &self,
        state: &mut Self::State,
        event: &EventEnvelope<P>,
    ) -> Result<(), InvariantViolation>;

    /// Checks protocol-specific invariants after each applied event.
    fn check_invariants(&self, state: &Self::State) -> Result<(), InvariantViolation>;

    /// Performs end-of-stream validation.
    fn finish(&self, _state: &Self::State) -> Result<(), InvariantViolation> {
        Ok(())
    }
}

/// A protocol-defined invariant violation before envelope context is attached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvariantViolation {
    pub code: String,
    pub message: String,
}

impl InvariantViolation {
    /// Creates a structured protocol violation.
    pub fn new(code: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
        }
    }
}

/// A replay failure with the shortest offending prefix.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayViolation {
    /// Number of events consumed, including the offending event.
    pub prefix_len: usize,
    /// Event responsible for the failure, if failure occurred during replay.
    pub event_id: Option<EventId>,
    /// Structured failure reason.
    pub kind: ReplayViolationKind,
}

/// Generic envelope and protocol failure categories.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReplayViolationKind {
    DuplicateEventId {
        event_id: EventId,
    },
    NonMonotonicSequence {
        actor_id: ActorId,
        previous: EventSequence,
        found: EventSequence,
    },
    NonMonotonicLogicalTimestamp {
        actor_id: ActorId,
        previous: LogicalTimestamp,
        found: LogicalTimestamp,
    },
    Reducer(InvariantViolation),
    Invariant(InvariantViolation),
    Finish(InvariantViolation),
}

/// Summary of a successful replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReplayReport {
    pub events_checked: usize,
}

/// Drives an ordered event stream through an independent reducer.
pub struct ReplayHarness<R> {
    reducer: R,
}

impl<R> ReplayHarness<R> {
    /// Creates a replay harness.
    pub const fn new(reducer: R) -> Self {
        Self { reducer }
    }

    /// Replays events in iterator order and returns the shortest violation.
    pub fn check<P, I>(
        &self,
        mut state: R::State,
        events: I,
    ) -> Result<ReplayReport, ReplayViolation>
    where
        R: ReplayReducer<P>,
        I: IntoIterator<Item = EventEnvelope<P>>,
    {
        let mut event_ids = HashSet::new();
        let mut actor_order = HashMap::new();
        let mut events_checked = 0;

        for event in events {
            events_checked += 1;

            if !event_ids.insert(event.event_id) {
                return Err(ReplayViolation {
                    prefix_len: events_checked,
                    event_id: Some(event.event_id),
                    kind: ReplayViolationKind::DuplicateEventId {
                        event_id: event.event_id,
                    },
                });
            }

            if let Some((previous_sequence, previous_timestamp)) =
                actor_order.get(&event.actor_id).copied()
            {
                if event.sequence <= previous_sequence {
                    return Err(ReplayViolation {
                        prefix_len: events_checked,
                        event_id: Some(event.event_id),
                        kind: ReplayViolationKind::NonMonotonicSequence {
                            actor_id: event.actor_id,
                            previous: previous_sequence,
                            found: event.sequence,
                        },
                    });
                }
                if event.logical_timestamp < previous_timestamp {
                    return Err(ReplayViolation {
                        prefix_len: events_checked,
                        event_id: Some(event.event_id),
                        kind: ReplayViolationKind::NonMonotonicLogicalTimestamp {
                            actor_id: event.actor_id,
                            previous: previous_timestamp,
                            found: event.logical_timestamp,
                        },
                    });
                }
            }
            actor_order.insert(event.actor_id, (event.sequence, event.logical_timestamp));

            self.reducer
                .apply(&mut state, &event)
                .map_err(|violation| ReplayViolation {
                    prefix_len: events_checked,
                    event_id: Some(event.event_id),
                    kind: ReplayViolationKind::Reducer(violation),
                })?;
            self.reducer
                .check_invariants(&state)
                .map_err(|violation| ReplayViolation {
                    prefix_len: events_checked,
                    event_id: Some(event.event_id),
                    kind: ReplayViolationKind::Invariant(violation),
                })?;
        }

        self.reducer
            .finish(&state)
            .map_err(|violation| ReplayViolation {
                prefix_len: events_checked,
                event_id: None,
                kind: ReplayViolationKind::Finish(violation),
            })?;

        Ok(ReplayReport { events_checked })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PressurePayload;

    struct NoopReducer;

    impl ReplayReducer<PressurePayload> for NoopReducer {
        type State = ();

        fn apply(
            &self,
            _state: &mut Self::State,
            _event: &EventEnvelope<PressurePayload>,
        ) -> Result<(), InvariantViolation> {
            Ok(())
        }

        fn check_invariants(&self, _state: &Self::State) -> Result<(), InvariantViolation> {
            Ok(())
        }
    }

    #[test]
    fn trivial_reducer_accepts_empty_stream() -> Result<(), ReplayViolation> {
        let report = ReplayHarness::new(NoopReducer).check((), Vec::new())?;

        assert_eq!(report.events_checked, 0);
        Ok(())
    }
}
