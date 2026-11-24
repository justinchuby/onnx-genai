//! One retained allocation per ORT compute state, borrowed by serial kernels.

use onnx_runtime_ep_api::kernel::WorkspaceView;
use onnx_runtime_ep_api::provider::WorkspaceAllocation;
use onnx_runtime_ep_api::tensor::DevicePtrMut;
use onnx_runtime_memory_api::MemoryRole;

use crate::ep::EpHandle;

/// Registration proof for ordered workspace use. Constructed only at the unsafe
/// factory registration boundary, not inferred from a device or stream pointer.
#[derive(Clone)]
pub(crate) struct OrderedWorkspace {
    preflight: Option<crate::factory::WorkspaceExecutionGuard>,
}

impl OrderedWorkspace {
    /// All uses, including error paths and helper streams, must be ordered before
    /// the next borrow. Replacement/release must settle work on this provider.
    pub(crate) unsafe fn new(preflight: Option<crate::factory::WorkspaceExecutionGuard>) -> Self {
        Self { preflight }
    }
}

pub(crate) struct WorkspaceLane {
    backing: Option<WorkspaceAllocation>,
    failure: Option<String>,
    ordering: Option<OrderedWorkspace>,
    // Must outlive allocation retirement, including error and unwind paths.
    provider: Option<EpHandle>,
}

impl WorkspaceLane {
    pub(crate) fn new(provider: Option<EpHandle>, ordering: Option<OrderedWorkspace>) -> Self {
        Self {
            backing: None,
            failure: None,
            ordering,
            provider,
        }
    }

    pub(crate) fn retire(&mut self, reason: &str) {
        if self.failure.is_none() {
            self.failure = Some(reason.to_owned());
        }
    }

    pub(crate) fn begin_compute(&self) -> Result<(), String> {
        self.check_active()?;
        if let Some(guard) = self
            .ordering
            .as_ref()
            .and_then(|proof| proof.preflight.as_ref())
        {
            guard().map_err(|error| {
                format!("persistent workspace execution domain rejected submission: {error}")
            })?;
        }
        Ok(())
    }

    pub(crate) fn check_active(&self) -> Result<(), String> {
        if let Some(error) = &self.failure {
            return Err(format!(
                "persistent workspace lane is retired after an earlier failure: {error}; recreate the session"
            ));
        }
        Ok(())
    }

    /// The caller holds this lane exclusively through the entire kernel use.
    pub(crate) fn borrow(
        &mut self,
        bytes: usize,
        alignment: usize,
    ) -> Result<WorkspaceView, String> {
        self.check_active()?;
        if bytes == 0 || !alignment.is_power_of_two() {
            return Err(format!(
                "invalid persistent workspace request: {bytes} bytes, alignment {alignment}; request nonzero bytes with power-of-two alignment"
            ));
        }
        if self.ordering.is_none() {
            return Err("persistent workspace requires registered execution ordering; register synchronous completion or a single ordered execution domain covering all kernel/helper-stream uses".into());
        }
        let provider = self.provider.as_ref().ok_or_else(|| "persistent workspace has no originating provider; compile through an EP before creating compute state".to_string())?;
        let fits = self
            .backing
            .as_ref()
            .is_some_and(|backing| backing.len() >= bytes && backing.alignment() >= alignment);
        if !fits {
            // Take once: replace consumes old ownership even when it fails.
            let capacity = self
                .backing
                .as_ref()
                .map_or(bytes, |old| old.len().max(bytes));
            let alignment = self
                .backing
                .as_ref()
                .map_or(alignment, |old| old.alignment().max(alignment));
            let old = self.backing.take();
            let result = provider.with(|ep| {
                let backing = ep.replace_workspace(old, capacity, alignment, MemoryRole::Workspace { step_scoped: false })?;
                if backing.len() < capacity || backing.alignment() < alignment || !backing.alignment().is_power_of_two()
                    || !(backing.as_ptr() as usize).is_multiple_of(alignment)
                    || backing.device() != ep.device_id()
                {
                    let detail = format!("provider '{}' returned invalid persistent workspace: requested {bytes} bytes aligned to {alignment} on {:?}, received {} bytes aligned to {} on {:?}", ep.name(), ep.device_id(), backing.len(), backing.alignment(), backing.device());
                    let release = ep.deallocate_workspace(backing);
                    return Err(onnx_runtime_ep_api::EpError::KernelFailed(format!("{detail}; release outcome: {release:?}; fix the provider workspace allocator")));
                }
                Ok(backing)
            });
            match result {
                Ok(backing) => self.backing = Some(backing),
                Err(error) => {
                    let error = format!(
                        "persistent workspace replacement ({bytes} bytes, alignment {alignment}) failed: {error}"
                    );
                    self.failure = Some(error.clone());
                    return Err(error);
                }
            }
        }
        let backing = self
            .backing
            .as_mut()
            .expect("successful replacement installs backing");
        Ok(WorkspaceView::new(
            DevicePtrMut(backing.as_mut_ptr()),
            bytes,
        ))
    }
}

impl Drop for WorkspaceLane {
    fn drop(&mut self) {
        if let Some(backing) = self.backing.take() {
            // A backed lane necessarily retained its originating provider.
            let provider = self
                .provider
                .as_ref()
                .expect("workspace backing requires a provider");
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                provider.with(|ep| ep.deallocate_workspace(backing))
            }));
            match result {
                Ok(Ok(())) => {}
                Ok(Err(error)) => eprintln!(
                    "nxrt ep plugin: persistent workspace release failed: {error}; inspect provider settlement/quarantine diagnostics; no refund is assumed"
                ),
                Err(_) => eprintln!(
                    "nxrt ep plugin: persistent workspace release panicked; inspect provider ownership and quarantine diagnostics; no retry or refund is assumed"
                ),
            }
        }
    }
}
