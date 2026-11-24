//! ORT Allocator and MemoryInfo.

use std::sync::Arc;

use crate::thread_affinity::{OwnerThread, ThreadAccess, ThreadAffinity};
use crate::{OrtError, Result, Session};

/// Memory type (CPU or device).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryType {
    /// CPU memory accessible by the device.
    CpuInput,
    /// CPU memory for outputs.
    CpuOutput,
    /// Default device memory (GPU HBM, NPU memory, etc.).
    Default,
}

/// Allocator type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocatorType {
    /// Device allocator (e.g., CUDA allocator).
    Device,
    /// Arena allocator (pooled).
    Arena,
}

mod memory_info;
pub use memory_info::MemoryInfo;
pub(crate) use memory_info::ResolvedMemoryInfo;

/// Where an [`Allocator`] came from, which decides both who releases it and
/// how long the session behind it has to stay alive.
enum AllocatorOrigin<'s> {
    /// ORT's process-wide default CPU allocator. It is owned by ORT, must never
    /// be released here, and the C API documents it as usable from any thread.
    ProcessDefault,
    /// `CreateAllocator` against a borrowed session. ORT frees the allocator's
    /// memory through that session's EP, so the borrow — not a comment — is
    /// what stops the allocator from outliving it.
    SessionBorrowed(&'s Session),
    /// `CreateAllocator` against a co-owned session, for owners that store the
    /// session and the allocator in the same struct and therefore cannot
    /// express the borrow.
    SessionShared(Arc<Session>),
}

impl AllocatorOrigin<'_> {
    fn is_session_owned(&self) -> bool {
        !matches!(self, Self::ProcessDefault)
    }

    fn session(&self) -> Option<&Session> {
        match self {
            Self::ProcessDefault => None,
            Self::SessionBorrowed(session) => Some(session),
            Self::SessionShared(session) => Some(session),
        }
    }
}

/// ORT Allocator.
///
/// # Threading
///
/// A session-derived allocator allocates through its session's execution
/// provider, so it carries the same exclusive-use rule as the binding it feeds
/// (see [`crate::thread_affinity`]); ORT's process-wide default CPU allocator is
/// documented as thread-safe and is guarded as [`ThreadAffinity::Shared`].
pub struct Allocator<'s> {
    ptr: *mut onnx_genai_ort_sys::OrtAllocator,
    memory_info: MemoryInfo,
    affinity: OwnerThread,
    origin: AllocatorOrigin<'s>,
}

impl Allocator<'static> {
    /// Get the default CPU allocator.
    pub fn default_cpu() -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let get = api
            .GetAllocatorWithDefaultOptions
            .ok_or(OrtError::ApiUnavailable("GetAllocatorWithDefaultOptions"))?;
        // SAFETY: `ptr` is a valid out-parameter. ORT returns a process-owned
        // default allocator that must not be released by this wrapper.
        crate::error::check_status(unsafe { get(&mut ptr) })?;
        Ok(Self {
            ptr,
            memory_info: MemoryInfo::cpu()?,
            affinity: OwnerThread::new("Allocator(default CPU)", ThreadAffinity::Shared),
            origin: AllocatorOrigin::ProcessDefault,
        })
    }
}

impl<'s> Allocator<'s> {
    /// The immutable placement description for this allocator. It cannot be
    /// replaced independently of the allocator's actual memory resource.
    pub fn memory_info(&self) -> &MemoryInfo {
        &self.memory_info
    }

    /// Create an allocator for `session` that allocates on the device described
    /// by `memory_info` (e.g. the WebGPU EP's `WebGPU_Buffer` device).
    ///
    /// This wraps the session's internal EP allocator, so tensors created with
    /// it are device-resident. `CreateAllocator` fails if the session has no
    /// allocator matching `memory_info` (for example when the requested EP was
    /// not actually attached and the session fell back to CPU); callers should
    /// treat that error as "device allocation unavailable" and fall back to the
    /// CPU allocator.
    pub(crate) fn for_session_device(
        session: &'s Session,
        memory_info: MemoryInfo,
    ) -> Result<Self> {
        Self::create(
            session.as_mut_ptr(),
            memory_info,
            AllocatorOrigin::SessionBorrowed(session),
        )
    }

    /// Same as [`Allocator::for_session_device`], but the allocator co-owns the
    /// session instead of borrowing it.
    pub(crate) fn for_shared_session_device(
        session: Arc<Session>,
        memory_info: MemoryInfo,
    ) -> Result<Allocator<'static>> {
        let ptr = session.as_mut_ptr();
        Allocator::create(ptr, memory_info, AllocatorOrigin::SessionShared(session))
    }

    fn create(
        session: *mut onnx_genai_ort_sys::OrtSession,
        memory_info: MemoryInfo,
        origin: AllocatorOrigin<'s>,
    ) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateAllocator
            .ok_or(OrtError::ApiUnavailable("CreateAllocator"))?;
        // SAFETY: `session` is a valid ORT session pointer and `memory_info` is a
        // valid handle; `ptr` is an out-parameter. On success ORT transfers an
        // owned allocator that this wrapper releases in Drop.
        crate::error::check_status(unsafe {
            create(session, memory_info.resolve()?.as_ptr(), &mut ptr)
        })?;
        Ok(Self {
            ptr,
            memory_info,
            affinity: OwnerThread::new("Allocator(session device)", ThreadAffinity::Exclusive),
            origin,
        })
    }

    /// The thread-ownership guard protecting this allocator.
    #[must_use]
    pub fn thread_affinity(&self) -> &OwnerThread {
        &self.affinity
    }

    /// The session this allocator allocates through, or `None` for ORT's
    /// process-wide default CPU allocator.
    #[must_use]
    pub fn session(&self) -> Option<&Session> {
        self.origin.session()
    }

    /// Take exclusive use of this allocator for one allocation.
    pub(crate) fn enter(&self, operation: &'static str) -> Result<ThreadAccess<'_>> {
        self.affinity.enter(operation).map_err(Into::into)
    }

    pub(crate) fn as_ptr(&self) -> *mut onnx_genai_ort_sys::OrtAllocator {
        self.ptr
    }
}

impl Drop for Allocator<'_> {
    fn drop(&mut self) {
        if !self.origin.is_session_owned() {
            return;
        }
        // Releasing an allocator another thread is still allocating from is a
        // use-after-free ORT cannot diagnose; `Drop` cannot fail, so name it.
        if let Err(violation) = self.affinity.check("drop") {
            tracing::error!("{violation}");
            debug_assert!(false, "{violation}");
        }
        if !self.ptr.is_null()
            && let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseAllocator
        {
            // SAFETY: Session-derived allocators come from ORT CreateAllocator
            // and are released exactly once here, before the session they were
            // created from: `origin` holds either a borrow the compiler checks
            // or an `Arc` this allocator co-owns. The process-default allocator
            // returned early above and is never released.
            unsafe { release(self.ptr) };
        }
    }
}
