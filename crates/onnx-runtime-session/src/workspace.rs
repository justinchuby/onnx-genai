//! Metadata preparation is a report, never permission to launch or capture.
use crate::{SessionError, error::Result};
use onnx_runtime_ep_api::{WorkspaceLifetime, WorkspaceRequirement};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceBounds {
    persistent: WorkspaceRequirement,
    step: WorkspaceRequirement,
}

impl WorkspaceBounds {
    pub const NONE: Self = Self {
        persistent: WorkspaceRequirement {
            lifetime: WorkspaceLifetime::SessionPersistent,
            ..WorkspaceRequirement::NONE
        },
        step: WorkspaceRequirement::NONE,
    };

    pub(crate) fn from_requirements(
        requirements: impl IntoIterator<Item = WorkspaceRequirement>,
    ) -> Self {
        let mut bounds = Self::NONE;
        for requirement in requirements {
            let peak = match requirement.lifetime {
                WorkspaceLifetime::SessionPersistent => &mut bounds.persistent,
                WorkspaceLifetime::StepScoped => &mut bounds.step,
            };
            peak.bytes = peak.bytes.max(requirement.bytes);
            peak.alignment = peak.alignment.max(requirement.alignment);
        }
        bounds
    }

    pub fn persistent(self) -> WorkspaceRequirement {
        self.persistent
    }
    pub fn step(self) -> WorkspaceRequirement {
        self.step
    }
}

/// Known bounds were reserved successfully. Deferred means more demand must be
/// resolved and admitted at actual eager dispatch; it never means zero demand.
/// Neither variant is a capture-readiness proof.
#[must_use = "inspect Complete versus Deferred, or explicitly require a complete metadata bound"]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum WorkspacePreparationReport {
    Complete(WorkspaceBounds),
    Deferred {
        known: WorkspaceBounds,
        pending: Vec<String>,
    },
}

impl WorkspacePreparationReport {
    pub const NONE: Self = Self::Complete(WorkspaceBounds::NONE);

    pub fn known(&self) -> WorkspaceBounds {
        match self {
            Self::Complete(bounds) | Self::Deferred { known: bounds, .. } => *bounds,
        }
    }

    pub fn require_complete(self) -> Result<WorkspaceBounds> {
        match self {
            Self::Complete(bounds) => Ok(bounds),
            Self::Deferred { pending, .. } => Err(SessionError::Internal(format!(
                "metadata workspace preparation is deferred: {}; provide concrete bindings/values or configure provable shape bounds, or use eager execution-time refinement and admission before launching these nodes",
                pending.join("; ")
            ))),
        }
    }

    /// Combine preparation passes (for example prefill and decode) without
    /// treating one completed pass as proof that another deferred pass finished.
    pub fn merge(self, other: Self) -> Self {
        let left = self.known();
        let right = other.known();
        let known = WorkspaceBounds::from_requirements([
            left.persistent,
            left.step,
            right.persistent,
            right.step,
        ]);
        let deferred =
            matches!(self, Self::Deferred { .. }) || matches!(other, Self::Deferred { .. });
        let mut pending = Vec::new();
        for report in [self, other] {
            if let Self::Deferred { pending: nodes, .. } = report {
                pending.extend(nodes);
            }
        }
        if deferred {
            Self::Deferred { known, pending }
        } else {
            Self::Complete(known)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn combined_passes_keep_both_lifetime_bounds_and_never_hide_deferral() {
        let persistent = WorkspaceRequirement {
            bytes: 128,
            alignment: 64,
            lifetime: WorkspaceLifetime::SessionPersistent,
        };
        let step = WorkspaceRequirement {
            bytes: 256,
            alignment: 16,
            ..WorkspaceRequirement::NONE
        };
        let deferred = WorkspacePreparationReport::Deferred {
            known: WorkspaceBounds::from_requirements([persistent]),
            pending: vec!["node #7 needs live output bindings".into()],
        };
        let complete =
            WorkspacePreparationReport::Complete(WorkspaceBounds::from_requirements([step]));
        for (first, second) in [(deferred.clone(), complete.clone()), (complete, deferred)] {
            let report = first.merge(second);
            assert_eq!(report.known().persistent(), persistent);
            assert_eq!(report.known().step(), step);
            assert!(
                report
                    .require_complete()
                    .unwrap_err()
                    .to_string()
                    .contains("node #7")
            );
        }
        let unlabelled = WorkspacePreparationReport::Deferred {
            known: WorkspaceBounds::NONE,
            pending: Vec::new(),
        };
        assert!(matches!(
            unlabelled.merge(WorkspacePreparationReport::NONE),
            WorkspacePreparationReport::Deferred { .. }
        ));
        assert_eq!(
            WorkspacePreparationReport::NONE.require_complete().unwrap(),
            WorkspaceBounds::NONE
        );
    }
}
