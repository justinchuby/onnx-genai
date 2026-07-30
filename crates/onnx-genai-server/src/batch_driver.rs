//! Which decode driver a model's engine actually runs, and why.
//!
//! Continuous batching is a *headline* capability of this server, and it is
//! selected at startup by a fallible probe: if the engine cannot build a
//! continuous-batch manager, the driver silently runs one generation at a time
//! instead. Every metric stays honest across that fallback -- occupancy really
//! is 1 of 1 -- which is exactly the hazard. **Our design protects against
//! fabricated numbers; it does not protect against true numbers describing a
//! different machine than the one being described aloud.**
//!
//! So the selection is recorded as a value, with the refusal reason attached,
//! and published. A capability this central must be legible to whoever is
//! reading the page, not just to whoever reads the startup log.

use std::fmt;

/// The decode driver an engine selected at startup.
///
/// Constructed once, on the thread that still owns the engine, and never
/// changed afterwards: the selection cannot drift from what is running because
/// there is nothing to re-evaluate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum BatchDriver {
    /// Many generations share each decode step. The capability being demoed.
    Continuous {
        /// Rows the batch can hold.
        capacity: usize,
    },
    /// One generation at a time, because continuous batching was refused.
    ///
    /// Carries the engine's own refusal text. The cause is not knowable from
    /// anywhere else: the probe returns a `Result` whose `Err` used to be
    /// dropped by an `.is_ok()`, which left "why" unanswerable from logs,
    /// status, or a debugger attached after the fact.
    PerRequest {
        /// The engine's verbatim reason for refusing.
        reason: String,
    },
    /// A pipeline engine, which has no continuous-batch path at all.
    ///
    /// Distinct from [`Self::PerRequest`]: nothing was refused here, so
    /// presenting a refusal reason would invent a failure that never happened.
    Pipeline,
}

impl BatchDriver {
    /// A stable identifier for clients to branch on.
    ///
    /// Separate from [`Self::explain`] so a UI never has to parse prose to
    /// decide what to render.
    pub(crate) const fn kind(&self) -> &'static str {
        match self {
            Self::Continuous { .. } => "continuous_batch",
            Self::PerRequest { .. } => "per_request",
            Self::Pipeline => "pipeline",
        }
    }

    /// Whether generations actually share decode steps on this engine.
    pub(crate) const fn batches_continuously(&self) -> bool {
        matches!(self, Self::Continuous { .. })
    }

    /// Prose for a human reading the page, always populated.
    ///
    /// The enabled case explains itself too. A reason shown only on failure
    /// teaches a reader that silence means success, and silence is also what a
    /// field that was never wired up looks like.
    pub(crate) fn explain(&self) -> String {
        match self {
            Self::Continuous { capacity } => format!(
                "continuous batching is active: up to {capacity} generations \
                 share each decode step"
            ),
            Self::PerRequest { reason } => format!(
                "continuous batching is INACTIVE; this engine decodes one \
                 generation at a time. The engine refused it: {reason}"
            ),
            Self::Pipeline => "this is a pipeline engine, which decodes one \
                               request at a time by construction; continuous \
                               batching was never attempted"
                .to_string(),
        }
    }
}

impl fmt::Display for BatchDriver {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_refused_continuous_batch_always_carries_the_engines_own_words() {
        let driver = BatchDriver::PerRequest {
            reason: "continuous batching requires a STATIC-CACHE or \
                     shared-buffer past/present model"
                .to_string(),
        };
        assert_eq!(driver.kind(), "per_request");
        assert!(!driver.batches_continuously());
        // The refusal text must survive to the reader verbatim. A summary like
        // "unsupported" would be true and would not tell an operator whether
        // to change a flag, a model, or nothing.
        assert!(
            driver.explain().contains("STATIC-CACHE or shared-buffer"),
            "the engine's reason was dropped on the way to the reader: {}",
            driver.explain()
        );
    }

    #[test]
    fn an_active_batcher_explains_itself_rather_than_staying_silent() {
        let driver = BatchDriver::Continuous { capacity: 4 };
        assert_eq!(driver.kind(), "continuous_batch");
        assert!(driver.batches_continuously());
        assert!(driver.explain().contains('4'));
    }

    #[test]
    fn a_pipeline_is_not_reported_as_a_refusal() {
        // Pipeline and PerRequest both decode serially, and collapsing them
        // would attach a refusal reason to an engine that never asked.
        let pipeline = BatchDriver::Pipeline;
        assert_eq!(pipeline.kind(), "pipeline");
        assert!(!pipeline.batches_continuously());
        assert_ne!(
            pipeline.kind(),
            BatchDriver::PerRequest {
                reason: String::new()
            }
            .kind()
        );
        assert!(pipeline.explain().contains("never attempted"));
    }
}
