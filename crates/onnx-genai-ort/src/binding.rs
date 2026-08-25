//! ORT IoBinding — pre-bind inputs/outputs to avoid per-run copies.

use std::collections::HashSet;
use std::ffi::CString;
use std::ptr::NonNull;
use std::sync::Arc;

use crate::thread_affinity::{OwnerThread, ThreadAccess, ThreadAffinity};
use crate::{Allocator, MemoryInfo, OrtError, Result, Session, Value};

/// How an [`IoBinding`] keeps the session it was created from alive.
///
/// ORT frees a binding's internal state against the session that created it, so
/// releasing the session first is a use-after-free. Which of these two variants
/// a binding holds is the *proof* that this cannot happen — a comment saying
/// "must not outlive the session" is not one.
enum BindingSession<'s> {
    /// The binding borrows the session, so the compiler rejects any owner that
    /// could outlive it.
    Borrowed(&'s Session),
    /// The binding co-owns the session. Owners that hold the session and the
    /// binding in the same struct need this: struct fields drop in declaration
    /// order, so field order alone would decide the release order, and a shared
    /// owner takes that decision away from whoever edits the struct next.
    Shared(Arc<Session>),
}

impl BindingSession<'_> {
    fn session(&self) -> &Session {
        match self {
            Self::Borrowed(session) => session,
            Self::Shared(session) => session,
        }
    }
}

/// IoBinding allows pre-allocating and binding device tensors.
/// This is critical for KV cache: we keep cache pages on-device and
/// bind them directly without host↔device copies each step.
///
/// # Lifetime
///
/// ORT frees a binding's bound state through the session that created it, so a
/// binding must never outlive its session. That is enforced, not documented:
/// [`IoBinding::new`] borrows the session, and [`IoBinding::for_shared_session`]
/// co-owns it.
///
/// The first block below is a positive control - a `compile_fail` block passes
/// when its snippet fails to build for *any* reason, including a renamed type,
/// so without a matching block that must compile it is evidence of nothing.
///
/// ```
/// use std::sync::Arc;
/// use onnx_genai_ort::{IoBinding, Session};
///
/// // A binding that co-owns its session may outlive the caller's handle.
/// fn owning(session: Arc<Session>) -> IoBinding<'static> {
///     IoBinding::for_shared_session(session).unwrap()
/// }
/// ```
///
/// ```compile_fail
/// use onnx_genai_ort::{IoBinding, Session};
///
/// // A borrowed binding cannot escape its session: this is the drop-order
/// // use-after-free, rejected at compile time.
/// fn escaping(session: Session) -> IoBinding<'static> {
///     IoBinding::new(&session).unwrap()
/// }
/// ```
///
/// # Threading
///
/// ORT's io binding is not thread-safe, and this type is `!Send + !Sync` so the
/// compiler keeps it on one thread. Containers with their own `unsafe impl Send`
/// can still carry one across a handoff; every operation therefore takes the
/// binding's [`OwnerThread`] guard, which permits an idle binding to change
/// threads and refuses two threads inside it at once. See
/// [`crate::thread_affinity`].
pub struct IoBinding<'s> {
    ptr: NonNull<onnx_genai_ort_sys::OrtIoBinding>,
    affinity: OwnerThread,
    session: BindingSession<'s>,
}

impl<'s> IoBinding<'s> {
    /// Create a new IoBinding borrowed from `session`.
    ///
    /// The returned binding cannot outlive `session`. Owners that store the
    /// session next to the binding cannot express that borrow and must use
    /// [`IoBinding::for_shared_session`] instead.
    pub fn new(session: &'s Session) -> Result<Self> {
        Self::create(BindingSession::Borrowed(session))
    }

    /// Create a new IoBinding that co-owns `session`.
    ///
    /// The ORT session outlives the binding by construction, whatever order the
    /// owning struct's fields drop in.
    pub fn for_shared_session(session: Arc<Session>) -> Result<IoBinding<'static>> {
        IoBinding::create(BindingSession::Shared(session))
    }

    fn create(session: BindingSession<'s>) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        let api = crate::error::api()?;
        let create = api
            .CreateIoBinding
            .ok_or(OrtError::ApiUnavailable("CreateIoBinding"))?;
        // SAFETY: `session` is a valid ORT session and `ptr` is an out-param.
        crate::error::check_status(unsafe { create(session.session().as_mut_ptr(), &mut ptr) })?;
        Ok(Self {
            ptr: NonNull::new(ptr).ok_or(OrtError::NullPointer)?,
            affinity: OwnerThread::new("IoBinding", ThreadAffinity::Exclusive),
            session,
        })
    }

    /// The session this binding runs against.
    #[must_use]
    pub fn session(&self) -> &Session {
        self.session.session()
    }

    /// The thread-ownership guard protecting this binding.
    #[must_use]
    pub fn thread_affinity(&self) -> &OwnerThread {
        &self.affinity
    }

    /// Bind a pre-existing tensor to a named input.
    pub fn bind_input(&mut self, name: &str, value: &Value) -> Result<()> {
        let _access = self.affinity.enter("bind_input")?;
        let name = c_name(name)?;
        let api = crate::error::api()?;
        let bind = api.BindInput.ok_or(OrtError::ApiUnavailable("BindInput"))?;
        // SAFETY: binding and value are valid ORT handles; `name` is
        // NUL-terminated and lives for the call.
        crate::error::check_status(unsafe {
            bind(self.ptr.as_ptr(), name.as_ptr(), value.as_ptr())
        })
    }

    /// Bind output to a specific device (ORT allocates on that device).
    pub fn bind_output_to_device(&mut self, name: &str, memory_info: &MemoryInfo) -> Result<()> {
        let _access = self.affinity.enter("bind_output_to_device")?;
        let name = c_name(name)?;
        let api = crate::error::api()?;
        let bind = api
            .BindOutputToDevice
            .ok_or(OrtError::ApiUnavailable("BindOutputToDevice"))?;
        // SAFETY: binding and memory info are valid ORT handles; `name` is
        // NUL-terminated and lives for the call.
        crate::error::check_status(unsafe {
            bind(self.ptr.as_ptr(), name.as_ptr(), memory_info.as_ptr())
        })
    }

    /// Bind a pre-existing tensor to a named output.
    pub fn bind_output(&mut self, name: &str, value: &Value) -> Result<()> {
        let _access = self.affinity.enter("bind_output")?;
        let name = c_name(name)?;
        let api = crate::error::api()?;
        let bind = api
            .BindOutput
            .ok_or(OrtError::ApiUnavailable("BindOutput"))?;
        // SAFETY: binding and value are valid ORT handles; `name` is
        // NUL-terminated and lives for the call.
        crate::error::check_status(unsafe {
            bind(self.ptr.as_ptr(), name.as_ptr(), value.as_ptr())
        })
    }

    /// Clear all bindings (reuse the binding object).
    pub fn clear(&mut self) -> Result<()> {
        let _access = self.affinity.enter("clear")?;
        let api = crate::error::api()?;
        let clear_inputs = api
            .ClearBoundInputs
            .ok_or(OrtError::ApiUnavailable("ClearBoundInputs"))?;
        let clear_outputs = api
            .ClearBoundOutputs
            .ok_or(OrtError::ApiUnavailable("ClearBoundOutputs"))?;
        // SAFETY: binding is valid; ORT clear functions do not return status.
        unsafe {
            clear_inputs(self.ptr.as_ptr());
            clear_outputs(self.ptr.as_ptr());
        }
        Ok(())
    }

    /// Clear only the bound inputs, leaving output bindings in place.
    ///
    /// ORT re-copies bound CPU inputs host->device on each run only for inputs
    /// (re)bound since the previous run; re-binding an already-bound value is a
    /// no-op that skips the copy. Captured-decode replay exploits this: it must
    /// refresh its mutated CPU inputs every step (so it clears inputs and
    /// re-binds them) while keeping the stable device-resident output bindings,
    /// avoiding a full-rebind of every KV/logits output each token.
    pub fn clear_inputs(&mut self) -> Result<()> {
        let _access = self.affinity.enter("clear_inputs")?;
        let api = crate::error::api()?;
        let clear_inputs = api
            .ClearBoundInputs
            .ok_or(OrtError::ApiUnavailable("ClearBoundInputs"))?;
        // SAFETY: binding is valid; ORT clear function does not return status.
        unsafe {
            clear_inputs(self.ptr.as_ptr());
        }
        Ok(())
    }

    pub(crate) fn as_ptr(&self) -> *const onnx_genai_ort_sys::OrtIoBinding {
        self.ptr.as_ptr()
    }

    /// Take exclusive use of this binding for the length of an ORT run.
    pub(crate) fn enter_run(&self, operation: &'static str) -> Result<ThreadAccess<'_>> {
        self.affinity.enter(operation).map_err(Into::into)
    }

    /// Take the OrtValues produced by the most recent `RunWithBinding`.
    ///
    /// Values are returned in the same order outputs were bound.
    pub fn output_values(&self) -> Result<Vec<Value>> {
        self.output_values_or_borrowed(&[])
            .map(|values| values.into_iter().flatten().collect())
    }

    pub(crate) fn output_values_or_borrowed(
        &self,
        borrowed_raw_ptrs: &[usize],
    ) -> Result<Vec<Option<Value>>> {
        let _access = self.affinity.enter("output_values")?;
        let borrowed_raw_ptrs = borrowed_raw_ptrs.iter().copied().collect::<HashSet<_>>();
        let allocator = Allocator::default_cpu()?;
        let api = crate::error::api()?;
        let get_outputs = api
            .GetBoundOutputValues
            .ok_or(OrtError::ApiUnavailable("GetBoundOutputValues"))?;
        let free = api
            .AllocatorFree
            .ok_or(OrtError::ApiUnavailable("AllocatorFree"))?;
        let mut output_ptrs = std::ptr::null_mut();
        let mut output_count = 0usize;
        // SAFETY: binding and allocator are valid; ORT allocates an array of
        // OrtValue pointers and transfers ownership of each OrtValue to us.
        crate::error::check_status(unsafe {
            get_outputs(
                self.ptr.as_ptr(),
                allocator.as_ptr(),
                &mut output_ptrs,
                &mut output_count,
            )
        })?;
        if output_count == 0 {
            return Ok(Vec::new());
        }
        if output_ptrs.is_null() {
            return Err(OrtError::NullPointer);
        }

        // SAFETY: ORT returned `output_count` pointers in an allocator-owned
        // array. We copy the pointers, free the array, and wrap each value.
        let raw_values = unsafe { std::slice::from_raw_parts(output_ptrs, output_count) }.to_vec();
        crate::error::check_status(unsafe { free(allocator.as_ptr(), output_ptrs.cast()) })?;
        raw_values
            .into_iter()
            .map(|ptr| {
                if borrowed_raw_ptrs.contains(&(ptr as usize)) {
                    Ok(None)
                } else {
                    // SAFETY: ORT returned this non-borrowed output OrtValue to
                    // the caller, so this wrapper owns and releases it.
                    unsafe { Value::from_raw(ptr).map(Some) }
                }
            })
            .collect()
    }
}

impl Drop for IoBinding<'_> {
    fn drop(&mut self) {
        // A binding released while another thread is inside it is exactly the
        // use-after-free this guard exists to name, and `Drop` cannot return an
        // error — so report it and release anyway: leaking the handle would not
        // make the other thread's pointer valid either.
        if let Err(violation) = self.affinity.check("drop") {
            tracing::error!("{violation}");
            debug_assert!(false, "{violation}");
        }
        if let Ok(api) = crate::error::api()
            && let Some(release) = api.ReleaseIoBinding
        {
            // SAFETY: `ptr` is owned by this wrapper and released exactly once here.
            unsafe { release(self.ptr.as_ptr()) };
        }
    }
}

fn c_name(name: &str) -> Result<CString> {
    CString::new(name).map_err(|_| OrtError::InvalidArgument(format!("name contains NUL: {name}")))
}
