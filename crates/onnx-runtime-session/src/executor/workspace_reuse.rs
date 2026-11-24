//! One reclaimable completed workspace per executor lane. This owns no budget.
use super::*;
use onnx_runtime_memory_api::{MemoryError, Tier};
use onnx_runtime_memory_governor::{
    AllocationSettlementStatus, PressureRegistration, PressureResponder,
};
use std::sync::{Mutex, Weak};

#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) enum WorkspaceCompletion {
    #[default]
    Pending,
    Completed,
}

/// Captured graphs may keep either lifetime's workspace address installed.
#[derive(Clone, Copy)]
pub(super) enum WorkspaceReplacement {
    Allowed,
    CapturePinned,
}

impl WorkspaceReplacement {
    pub(super) fn for_captures(captures: &[SlotCaptureState]) -> Self {
        if captures.iter().any(|capture| {
            capture.device_graph_token.is_some()
                || capture.capture_schedule.as_ref().is_some_and(|schedule| {
                    schedule.segments.iter().any(|segment| segment.captured)
                })
        }) {
            Self::CapturePinned
        } else {
            Self::Allowed
        }
    }

    pub(super) fn including(self, enclosing: Option<Self>) -> Self {
        if matches!(self, Self::CapturePinned) || matches!(enclosing, Some(Self::CapturePinned)) {
            Self::CapturePinned
        } else {
            Self::Allowed
        }
    }

    pub(super) fn check(
        self,
        slot: Option<&PreparedWorkspace>,
        requirement: WorkspaceRequirement,
    ) -> Result<()> {
        if matches!(self, Self::CapturePinned)
            && slot.is_some_and(|slot| {
                requirement.bytes > slot.bytes as u64 || requirement.alignment > slot.alignment
            })
        {
            return Err(EpError::KernelFailed(format!(
                "cannot grow {:?} workspace to {} bytes aligned to {} while an installed device graph may reference its current address; reset_device_graph for every installed graph slot, then prepare the larger shape again",
                requirement.lifetime, requirement.bytes, requirement.alignment
            )).into());
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) enum WorkspaceReuse {
    #[default]
    Uninitialized,
    Disabled,
    Enabled {
        idle: Arc<IdleWorkspace>,
        _registration: PressureRegistration,
    },
}

pub(super) struct IdleWorkspace {
    owner: Mutex<Option<PreparedWorkspace>>,
    provider: Weak<dyn ExecutionProvider>,
    tier: Tier,
}

impl IdleWorkspace {
    fn take(&self) -> Option<PreparedWorkspace> {
        self.owner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }
}

impl PressureResponder for IdleWorkspace {
    fn on_pressure(&self, tier: Tier, want: u64) -> std::result::Result<u64, MemoryError> {
        if tier != self.tier || want == 0 {
            return Ok(0);
        }
        let Some(provider) = self.provider.upgrade() else {
            return Ok(0);
        };
        let Some(workspace) = self.take() else {
            return Ok(0);
        };
        // Neither a provider call nor a settlement wait holds the idle lock.
        // CUDA retirement may enqueue release rather than immediately refund.
        let settlement = workspace.buffer.managed_settlement_wait();
        let charged = settlement.as_ref().map_or(0, |wait| wait.charged_bytes());
        provider
            .deallocate_workspace(workspace.buffer)
            .map_err(|source| MemoryError::ReclaimFailed {
                tier: tier.name(),
                requested: want,
                source: Box::new(source),
            })?;
        // This wait is confined to cold pressure, never a fitting warm run.
        // Unmanaged or retained releases cannot report an authority refund.
        Ok(match settlement {
            Some(wait)
                if wait.wait(Duration::from_secs(30)) == AllocationSettlementStatus::Released =>
            {
                charged
            }
            _ => 0,
        })
    }
}

impl WorkspaceReuse {
    fn initialize(&mut self, provider: &Arc<dyn ExecutionProvider>) -> Result<()> {
        if !matches!(self, Self::Uninitialized) {
            return Ok(());
        }
        let Some(registry) = provider.workspace_pressure_registry()? else {
            *self = Self::Disabled;
            return Ok(());
        };
        let idle = Arc::new(IdleWorkspace {
            owner: Mutex::new(None),
            provider: Arc::downgrade(provider),
            tier: registry.tier(),
        });
        let responder: Arc<dyn PressureResponder> = idle.clone();
        let registration = registry.register(&responder).map_err(|error| {
            EpError::KernelFailed(format!(
                "{}: cannot enroll completed workspace in its pressure registry: {error}; keep workspace backing and pressure registration in the same authority",
                provider.name()
            ))
        })?;
        *self = Self::Enabled {
            idle,
            _registration: registration,
        };
        Ok(())
    }

    pub(super) fn take(&self) -> Option<PreparedWorkspace> {
        match self {
            Self::Enabled { idle, .. } => idle.take(),
            _ => None,
        }
    }

    /// Consume one completed owner, preserving its allocation and original charge.
    pub(super) fn recycle(
        &mut self,
        provider: &Arc<dyn ExecutionProvider>,
        workspace: PreparedWorkspace,
    ) -> Result<()> {
        if let Err(error) = self.initialize(provider) {
            let release = provider.deallocate_workspace(workspace.buffer);
            return match release {
                Ok(()) => Err(error),
                Err(release) => Err(EpError::KernelFailed(format!(
                    "completed workspace reuse initialization failed: {error}; retiring its backing also failed: {release}"
                ))
                .into()),
            };
        }
        match self {
            Self::Enabled { idle, .. } => {
                let old = idle
                    .owner
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .replace(workspace);
                // Normally the lane took its idle owner before running. Keep
                // retirement explicit even if preparation supplied another one.
                if let Some(old) = old {
                    provider.deallocate_workspace(old.buffer)?;
                }
                Ok(())
            }
            _ => provider
                .deallocate_workspace(workspace.buffer)
                .map_err(Into::into),
        }
    }

    pub(super) fn retire(&mut self, provider: &dyn ExecutionProvider, safe: bool) {
        let old = std::mem::replace(self, Self::Disabled);
        let workspace = old.take();
        // Withdraw enrollment before retiring storage. A pressure callback
        // already snapshotted can only take the same owner once.
        drop(old);
        if let Some(workspace) = workspace {
            if safe {
                if let Err(error) = provider.deallocate_workspace(workspace.buffer) {
                    eprintln!("completed workspace backing could not retire: {error}");
                }
            } else {
                drop(workspace);
            }
        }
    }
}

/// Keep both dimensions of the physical lane peak when either must grow.
pub(super) fn workspace_high_water(
    requirement: WorkspaceRequirement,
    old: Option<&PreparedWorkspace>,
) -> WorkspaceRequirement {
    match old {
        Some(old) => WorkspaceRequirement {
            bytes: requirement.bytes.max(old.bytes as u64),
            alignment: requirement.alignment.max(old.alignment),
            ..requirement
        },
        None => requirement,
    }
}

impl Executor {
    pub(super) fn workspace_replacement(&self) -> WorkspaceReplacement {
        WorkspaceReplacement::for_captures(&self.slot_capture)
            .including(self.parent_workspace_policy)
    }

    pub(super) fn restore_completed_step_workspace(&mut self) {
        if self.parent_workspace_policy.is_none() && self.step_workspace.is_none() {
            self.step_workspace = self.workspace_reuse.take();
        }
    }
}
