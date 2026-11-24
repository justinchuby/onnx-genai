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
    /// Successful execution deferred only its same-generation validation.
    AwaitingValidation,
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

/// The two independent physical lifetime peaks, moved as one root lane.
#[derive(Default)]
pub(super) struct WorkspacePair {
    pub(super) persistent: Option<PreparedWorkspace>,
    pub(super) step: Option<PreparedWorkspace>,
}

impl WorkspacePair {
    pub(super) fn is_empty(&self) -> bool {
        self.persistent.is_none() && self.step.is_none()
    }

    fn into_owners(self) -> impl Iterator<Item = (WorkspaceLifetime, PreparedWorkspace)> {
        [
            (WorkspaceLifetime::SessionPersistent, self.persistent),
            (WorkspaceLifetime::StepScoped, self.step),
        ]
        .into_iter()
        .filter_map(|(lifetime, owner)| owner.map(|owner| (lifetime, owner)))
    }

    fn take_pressure(
        &mut self,
        want: u64,
        lifetime: Option<WorkspaceLifetime>,
    ) -> Option<(WorkspaceLifetime, PreparedWorkspace)> {
        let lifetime = lifetime.or_else(|| {
            // A cold, non-consuming charge snapshot; no provider callback or
            // settlement wait runs under the idle lock. Logical bytes need not
            // equal physical granule charge, so use the latter when available.
            let capacity = |owner: &PreparedWorkspace| {
                owner
                    .buffer
                    .managed_settlement_wait()
                    .map_or(owner.bytes as u64, |wait| wait.charged_bytes())
            };
            match (
                self.persistent.as_ref().map(capacity),
                self.step.as_ref().map(capacity),
            ) {
                (Some(p), Some(s)) if s >= want && (p < want || s <= p) => {
                    Some(WorkspaceLifetime::StepScoped)
                }
                (Some(p), _) if p >= want => Some(WorkspaceLifetime::SessionPersistent),
                (None, Some(_)) => Some(WorkspaceLifetime::StepScoped),
                (Some(p), Some(s)) if s <= p => Some(WorkspaceLifetime::StepScoped),
                (Some(_), _) => Some(WorkspaceLifetime::SessionPersistent),
                _ => None,
            }
        })?;
        let owner = match lifetime {
            WorkspaceLifetime::SessionPersistent => self.persistent.take(),
            WorkspaceLifetime::StepScoped => self.step.take(),
        }?;
        Some((lifetime, owner))
    }
}

enum WorkspacePlacement {
    Active(WorkspacePair),
    Idle,
}

pub(super) struct WorkspaceLane {
    placement: WorkspacePlacement,
    pub(super) reuse: WorkspaceReuse,
}

impl Default for WorkspaceLane {
    fn default() -> Self {
        Self {
            placement: WorkspacePlacement::Active(WorkspacePair::default()),
            reuse: WorkspaceReuse::default(),
        }
    }
}

impl WorkspaceLane {
    /// Active hits do not inspect or lock the idle cell, even for an empty pair.
    pub(super) fn active_mut(&mut self) -> &mut WorkspacePair {
        if matches!(self.placement, WorkspacePlacement::Idle) {
            self.placement = WorkspacePlacement::Active(self.reuse.take().unwrap_or_default());
        }
        match &mut self.placement {
            WorkspacePlacement::Active(pair) => pair,
            WorkspacePlacement::Idle => unreachable!("idle placement activated immediately above"),
        }
    }

    pub(super) fn active(&self) -> Option<&WorkspacePair> {
        match &self.placement {
            WorkspacePlacement::Active(pair) => Some(pair),
            WorkspacePlacement::Idle => None,
        }
    }

    /// Loan Drop must never touch the pressure cell. A valid loan always keeps
    /// this placement Active; release/capture boundaries refuse child pooling.
    pub(super) fn take_active(&mut self) -> Option<WorkspacePair> {
        match &mut self.placement {
            WorkspacePlacement::Active(pair) => Some(std::mem::take(pair)),
            WorkspacePlacement::Idle => None,
        }
    }

    pub(super) fn recycle_completed(
        &mut self,
        provider: &Arc<dyn ExecutionProvider>,
    ) -> Result<()> {
        if self.active().is_none_or(WorkspacePair::is_empty) {
            return Ok(());
        }
        if let Err(error) = self.reuse.initialize(provider) {
            // Match the original Step retirement on failed enrollment. Persistent
            // remains owned/active; no fallible callback can orphan that owner.
            let step = self.active_mut().step.take();
            return match step
                .map(|owner| provider.deallocate_workspace(owner.buffer))
                .transpose()
            {
                Ok(_) => Err(error),
                Err(retirement) => Err(EpError::Backend {
                    operation: "enroll completed workspace and retire step backing",
                    source: Box::new(WorkspaceEnrollmentRetirement {
                        initialization: error,
                        retirement,
                    }),
                }
                .into()),
            };
        }
        match &self.reuse {
            WorkspaceReuse::Enabled { idle, .. } => {
                let WorkspacePlacement::Active(pair) =
                    std::mem::replace(&mut self.placement, WorkspacePlacement::Idle)
                else {
                    unreachable!("nonempty active pair checked above")
                };
                let old = idle.replace(pair);
                // Root placement is already Idle if retiring an unexpected old
                // owner fails. The new owners remain reachable exactly once.
                if let Some(old) = old {
                    retire_pair(provider.as_ref(), old).map_err(|source| EpError::Backend {
                        operation: "retire superseded completed workspace",
                        source: Box::new(source),
                    })?;
                }
                Ok(())
            }
            _ => {
                // No pressure capability: preserve Persistent's existing warm
                // retention; Step keeps its original provider retirement policy.
                self.retire_step(provider.as_ref())
            }
        }
    }

    pub(super) fn retire_step(&mut self, provider: &dyn ExecutionProvider) -> Result<()> {
        if let WorkspacePlacement::Active(pair) = &mut self.placement
            && let Some(owner) = pair.step.take()
        {
            provider.deallocate_workspace(owner.buffer)?;
        }
        Ok(())
    }

    pub(super) fn retire(&mut self, provider: &dyn ExecutionProvider, safe: bool) {
        self.reuse.retire(provider, safe);
        if let Some(pair) = self.take_active() {
            if safe {
                if let Err(error) = retire_pair(provider, pair) {
                    eprintln!("workspace lane backing could not retire: {error}");
                }
            } else {
                drop(pair);
            }
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "workspace enrollment failed: {initialization}; retiring Step backing also failed: {retirement}"
)]
struct WorkspaceEnrollmentRetirement {
    #[source]
    initialization: SessionError,
    retirement: EpError,
}

#[derive(Debug)]
struct WorkspaceRetirementErrors(Vec<(WorkspaceLifetime, EpError)>);
impl std::fmt::Display for WorkspaceRetirementErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (index, (lifetime, error)) in self.0.iter().enumerate() {
            if index != 0 {
                write!(f, "; ")?;
            }
            write!(f, "{lifetime:?} workspace release failed: {error}")?;
        }
        write!(
            f,
            "; unsettled backing remains charged and must settle through the original provider"
        )
    }
}
impl std::error::Error for WorkspaceRetirementErrors {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.0.first().map(|(_, error)| error as _)
    }
}

fn retire_pair(
    provider: &dyn ExecutionProvider,
    pair: WorkspacePair,
) -> std::result::Result<(), WorkspaceRetirementErrors> {
    let mut errors = Vec::new();
    for (lifetime, owner) in pair.into_owners() {
        if let Err(error) = provider.deallocate_workspace(owner.buffer) {
            errors.push((lifetime, error));
        }
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(WorkspaceRetirementErrors(errors))
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
    owners: Mutex<Option<WorkspacePair>>,
    provider: Weak<dyn ExecutionProvider>,
    tier: Tier,
    #[cfg(test)]
    locks: std::sync::atomic::AtomicUsize,
}

impl IdleWorkspace {
    fn lock(&self) -> std::sync::MutexGuard<'_, Option<WorkspacePair>> {
        #[cfg(test)]
        self.locks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.owners
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
    fn take(&self) -> Option<WorkspacePair> {
        self.lock().take()
    }
    fn replace(&self, pair: WorkspacePair) -> Option<WorkspacePair> {
        self.lock().replace(pair)
    }
    fn take_pressure(
        &self,
        want: u64,
        lifetime: Option<WorkspaceLifetime>,
    ) -> Option<(WorkspaceLifetime, PreparedWorkspace)> {
        let mut owners = self.lock();
        let selected = owners
            .as_mut()
            .and_then(|pair| pair.take_pressure(want, lifetime));
        if owners.as_ref().is_some_and(WorkspacePair::is_empty) {
            owners.take();
        }
        selected
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
        // Select the smallest whole physical owner that can satisfy pressure,
        // leaving untouched ownership in the cell before unlocking. Only actual
        // settled refunds decide whether the other lifetime is needed. Provider
        // calls and settlement waits run outside the idle-cell lock.
        let deadline = std::time::Instant::now() + Duration::from_secs(30);
        let mut released = 0u64;
        let mut selected = self.take_pressure(want, None);
        for attempt in 0..2 {
            let Some((lifetime, workspace)) = selected else {
                break;
            };
            let settlement = workspace.buffer.managed_settlement_wait();
            let charged = settlement.as_ref().map_or(0, |wait| wait.charged_bytes());
            provider
                .deallocate_workspace(workspace.buffer)
                .map_err(|source| MemoryError::ReclaimFailed {
                    tier: tier.name(),
                    requested: want,
                    source: Box::new(source),
                })?;
            if let Some(wait) = settlement
                && wait.wait(deadline.saturating_duration_since(std::time::Instant::now()))
                    == AllocationSettlementStatus::Released
            {
                released = released.saturating_add(charged);
            }
            if released >= want || attempt == 1 {
                break;
            }
            let other = match lifetime {
                WorkspaceLifetime::SessionPersistent => WorkspaceLifetime::StepScoped,
                WorkspaceLifetime::StepScoped => WorkspaceLifetime::SessionPersistent,
            };
            selected = self.take_pressure(want.saturating_sub(released), Some(other));
        }
        Ok(released)
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
            owners: Mutex::new(None),
            provider: Arc::downgrade(provider),
            tier: registry.tier(),
            #[cfg(test)]
            locks: std::sync::atomic::AtomicUsize::new(0),
        });
        let responder: Arc<dyn PressureResponder> = idle.clone();
        let registration = registry
            .register(&responder)
            .map_err(|source| EpError::Backend {
                operation: "enroll completed workspace in its original authority pressure registry",
                source: Box::new(source),
            })?;
        *self = Self::Enabled {
            idle,
            _registration: registration,
        };
        Ok(())
    }
    pub(super) fn take(&self) -> Option<WorkspacePair> {
        match self {
            Self::Enabled { idle, .. } => idle.take(),
            _ => None,
        }
    }
    #[cfg(test)]
    pub(super) fn lock_count(&self) -> usize {
        match self {
            Self::Enabled { idle, .. } => idle.locks.load(std::sync::atomic::Ordering::Relaxed),
            _ => 0,
        }
    }
    pub(super) fn retire(&mut self, provider: &dyn ExecutionProvider, safe: bool) {
        let old = std::mem::replace(self, Self::Disabled);
        let pair = old.take();
        drop(old);
        if let Some(pair) = pair {
            if safe {
                if let Err(error) = retire_pair(provider, pair) {
                    eprintln!("completed workspace backing could not retire: {error}");
                }
            } else {
                drop(pair);
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
    /// Only successful deferred runs can promote from a host-only receipt.
    /// Unknown/error/prepare-only Pending work keeps the provider retirement path.
    pub(super) fn resolve_awaiting_validation(&mut self) -> Result<()> {
        if self.parent_workspace_policy.is_some()
            || self.workspace_completion != WorkspaceCompletion::AwaitingValidation
        {
            return Ok(());
        }
        self.refresh_awaiting_workspace_completion()?;
        match self.workspace_completion {
            WorkspaceCompletion::AwaitingValidation => Err(EpError::KernelFailed(format!("{}: previous deferred workspace validation is still pending; read a bound output or finish its request boundary before preparing/submitting another generation", self.ep.name())).into()),
            WorkspaceCompletion::Pending => self.release_workspaces(),
            WorkspaceCompletion::Completed => Ok(()),
        }
    }

    /// One classification for both release and the next submission. Only the
    /// successful-deferred state may be upgraded by an exact clean receipt.
    pub(super) fn refresh_awaiting_workspace_completion(&mut self) -> Result<()> {
        if self.workspace_completion != WorkspaceCompletion::AwaitingValidation {
            return Ok(());
        }
        let token = self.pending_device_validation.ok_or_else(|| SessionError::Internal("successful deferred workspace is missing its exact validation token; recreate the session instead of reusing unproven backing".into()))?;
        let registration = self.validation_registration.as_ref().ok_or_else(|| SessionError::Internal("successful deferred workspace is missing its validation registration; recreate the session".into()))?;
        match self.ep.poll_device_validation_completion(registration, token)? {
            onnx_runtime_ep_api::DeviceValidationCompletion::Complete { flags: 0 } => self.workspace_completion = WorkspaceCompletion::Completed,
            onnx_runtime_ep_api::DeviceValidationCompletion::Complete { flags } => return Err(EpError::KernelFailed(format!("{}: previous workspace submission failed device validation (flags=0x{flags:x}); finish its failed request boundary before submitting another generation", self.ep.name())).into()),
            onnx_runtime_ep_api::DeviceValidationCompletion::Pending => {}
            onnx_runtime_ep_api::DeviceValidationCompletion::Unavailable => self.workspace_completion = WorkspaceCompletion::Pending,
        }
        Ok(())
    }

    pub(super) fn workspace_replacement(&self) -> WorkspaceReplacement {
        WorkspaceReplacement::for_captures(&self.slot_capture)
            .including(self.parent_workspace_policy)
    }

    pub(super) fn restore_completed_workspaces(&mut self) {
        if self.parent_workspace_policy.is_none() {
            self.workspaces.active_mut();
        }
    }
}
