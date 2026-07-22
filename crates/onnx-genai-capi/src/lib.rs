//! # `onnx-genai-capi`
//!
//! A C ABI over [`onnx_genai_engine`] focused on the **pluggable sampler** seam:
//! external code (C, C++, Python via ctypes/cffi, …) can supply its own token
//! sampler and drive generation with it, without any Rust knowledge.
//!
//! The Rust generation loop still runs the full logit-processor chain
//! (temperature, top-k/top-p, min-p, repetition/frequency/presence penalties,
//! constraints, …) configured on the request; the foreign sampler only replaces
//! the terminal token selection (the step that would otherwise be greedy argmax
//! or categorical sampling). This mirrors how onnxruntime-genai layers top-k/
//! top-p/temperature filtering before the final pick, but lets you own that pick.
//!
//! ## Surface
//!
//! * [`OgeSamplerVTable`] — a `#[repr(C)]` description of a foreign sampler:
//!   a `user_data` pointer, a `sample` callback, an optional `name`, and an
//!   optional `free` destructor for `user_data`.
//! * [`ForeignSampler`] — the Rust adapter that implements
//!   [`onnx_genai_engine::Sampler`] by invoking the vtable. It is `pub` so it can
//!   be unit-tested (and reused by other Rust callers) without a model.
//! * `oge_sampler_new` / `oge_sampler_free` — construct/destroy a boxed sampler.
//! * `oge_engine_load` / `oge_engine_free` — load/free a model engine.
//! * `oge_engine_generate` / `oge_engine_generate_with_sampler` — run generation;
//!   the latter consumes an `OgeSampler*` (taking ownership of it).
//! * `oge_string_free` — free any `char*` returned by this library.
//! * `oge_last_error` — thread-local message for the most recent failure.
//!
//! ## Safety model
//!
//! * Opaque handles ([`OgeEngine`], [`OgeSampler`]) are created with
//!   [`Box::into_raw`] and freed exactly once by the matching `*_free`, which is
//!   null-tolerant so `free(x); x = NULL;` makes double-free unreachable.
//! * Every incoming pointer is null-checked and turned into an error (recorded in
//!   [`oge_last_error`]) rather than dereferenced.
//! * No panic crosses the boundary: every body runs inside
//!   [`std::panic::catch_unwind`]; a panic becomes a null/`0` return plus a
//!   recorded error, never an unwind into C (which is undefined behavior).

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::ptr;

use onnx_genai_engine::{
    Engine, EngineConfig, GenerateOptions, GenerateRequest, ProcessorContext, Sampler, TokenId,
};

// ---------------------------------------------------------------------------
// Thread-local last-error
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(message: impl Into<String>) {
    let message = message.into();
    let cstring = CString::new(message).unwrap_or_else(|_| {
        CString::new("onnx-genai-capi: error message contained an interior NUL").unwrap()
    });
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(cstring));
}

fn clear_last_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

/// Borrow the most recent error message for the calling thread as a
/// NUL-terminated C string, or `NULL` if the last call on this thread succeeded.
///
/// The pointer is owned by the library and valid until the next fallible call on
/// the same thread. Do **not** free it.
///
/// # Safety
/// Trivially safe; returns a borrowed pointer or null.
#[unsafe(no_mangle)]
pub extern "C" fn oge_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| {
        slot.borrow()
            .as_ref()
            .map_or(ptr::null(), |cstring| cstring.as_ptr())
    })
}

/// Run `body`, converting a panic into `on_panic` and recording an error.
fn guard<T>(context: &str, on_panic: T, body: impl FnOnce() -> T) -> T {
    match catch_unwind(AssertUnwindSafe(body)) {
        Ok(value) => value,
        Err(_) => {
            set_last_error(format!(
                "{context}: panicked (recovered at the C ABI boundary)"
            ));
            on_panic
        }
    }
}

// ---------------------------------------------------------------------------
// Foreign sampler vtable + adapter
// ---------------------------------------------------------------------------

/// C callback that selects a token id from processed logits.
///
/// Arguments:
/// * `user_data` — the pointer supplied in [`OgeSamplerVTable::user_data`].
/// * `logits` / `logits_len` — the post-processor logits for the current step
///   (length is the vocabulary size). Filtered-out tokens are `-inf`.
/// * `generated` / `generated_len` — token ids generated so far this request.
/// * `step` — 0-based decode step index.
///
/// Returns the chosen token id. It must be a valid index into `logits`
/// (`0 <= token < logits_len`); out-of-range values select nothing useful and
/// are the caller's responsibility.
pub type OgeSampleFn = unsafe extern "C" fn(
    user_data: *mut c_void,
    logits: *const f32,
    logits_len: usize,
    generated: *const u32,
    generated_len: usize,
    step: usize,
) -> u32;

/// Optional destructor for [`OgeSamplerVTable::user_data`], invoked once when the
/// sampler is dropped (via `oge_sampler_free` or after generation consumes it).
pub type OgeFreeFn = unsafe extern "C" fn(user_data: *mut c_void);

/// `#[repr(C)]` description of a foreign sampler.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct OgeSamplerVTable {
    /// Opaque state passed back to every callback. May be null.
    pub user_data: *mut c_void,
    /// Required token-selection callback.
    pub sample: OgeSampleFn,
    /// Optional NUL-terminated name (borrowed, copied on construction). May be
    /// null, in which case the sampler is named `"foreign"`.
    pub name: *const c_char,
    /// Optional destructor for `user_data`; null means the library never frees it.
    pub free: Option<OgeFreeFn>,
}

/// Rust adapter wrapping an [`OgeSamplerVTable`] as an [`Sampler`].
///
/// This is the bridge between the C ABI and the engine's Rust sampler trait. It
/// is exposed publicly so it can be unit-tested and reused by Rust callers that
/// already hold a vtable.
pub struct ForeignSampler {
    vtable: OgeSamplerVTable,
    name: String,
}

// SAFETY: `Sampler: Send` is required by the engine. The caller of
// `ForeignSampler::from_vtable` promises the `user_data` and callbacks are safe
// to move to (and use from) the generation thread.
unsafe impl Send for ForeignSampler {}

impl ForeignSampler {
    /// Build an adapter from a vtable.
    ///
    /// # Safety
    /// * `vtable.sample` must be a valid function pointer for the sampler's life.
    /// * `vtable.user_data` must remain valid until the sampler is dropped, at
    ///   which point `vtable.free` (if set) is called with it exactly once.
    /// * `vtable.name`, if non-null, must point to a valid NUL-terminated string.
    pub unsafe fn from_vtable(vtable: OgeSamplerVTable) -> Self {
        let name = if vtable.name.is_null() {
            "foreign".to_string()
        } else {
            unsafe { CStr::from_ptr(vtable.name) }
                .to_string_lossy()
                .into_owned()
        };
        Self { vtable, name }
    }
}

impl Drop for ForeignSampler {
    fn drop(&mut self) {
        if let Some(free) = self.vtable.free {
            // SAFETY: the constructor's contract guarantees `free` matches
            // `user_data`; we call it exactly once here on drop.
            unsafe { free(self.vtable.user_data) };
        }
    }
}

impl Sampler for ForeignSampler {
    fn sample(&mut self, logits: &[f32], context: &ProcessorContext) -> TokenId {
        // SAFETY: pointers/lengths describe live slices for the call's duration;
        // the callback validity is guaranteed by `from_vtable`'s contract.
        unsafe {
            (self.vtable.sample)(
                self.vtable.user_data,
                logits.as_ptr(),
                logits.len(),
                context.generated_tokens.as_ptr(),
                context.generated_tokens.len(),
                context.step,
            )
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Opaque handle to a boxed foreign sampler.
pub struct OgeSampler(Box<ForeignSampler>);

/// Create a sampler from a vtable. Returns null on failure (see
/// [`oge_last_error`]). Free with `oge_sampler_free`, or hand it to
/// `oge_engine_generate_with_sampler`, which consumes it.
///
/// # Safety
/// `vtable` must satisfy [`ForeignSampler::from_vtable`]'s contract.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oge_sampler_new(vtable: OgeSamplerVTable) -> *mut OgeSampler {
    clear_last_error();
    guard("oge_sampler_new", ptr::null_mut(), || {
        let sampler = unsafe { ForeignSampler::from_vtable(vtable) };
        Box::into_raw(Box::new(OgeSampler(Box::new(sampler))))
    })
}

/// Free a sampler created by `oge_sampler_new` that was **not** consumed by a
/// generation call. Null-tolerant. Invokes the vtable's `free`, if any.
///
/// # Safety
/// `sampler` must be null or a pointer previously returned by `oge_sampler_new`
/// and not yet freed/consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oge_sampler_free(sampler: *mut OgeSampler) {
    if sampler.is_null() {
        return;
    }
    guard("oge_sampler_free", (), || {
        drop(unsafe { Box::from_raw(sampler) });
    });
}

// ---------------------------------------------------------------------------
// Engine handle
// ---------------------------------------------------------------------------

/// Opaque handle to a loaded generation engine.
pub struct OgeEngine(Engine);

/// Load a model directory into an engine. Returns null on failure (see
/// [`oge_last_error`]). Free with `oge_engine_free`.
///
/// # Safety
/// `model_dir` must be a valid NUL-terminated UTF-8 path.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oge_engine_load(model_dir: *const c_char) -> *mut OgeEngine {
    clear_last_error();
    guard("oge_engine_load", ptr::null_mut(), || {
        if model_dir.is_null() {
            set_last_error("oge_engine_load: model_dir is null");
            return ptr::null_mut();
        }
        let dir = match unsafe { CStr::from_ptr(model_dir) }.to_str() {
            Ok(dir) => dir,
            Err(err) => {
                set_last_error(format!(
                    "oge_engine_load: model_dir is not valid UTF-8: {err}"
                ));
                return ptr::null_mut();
            }
        };
        match Engine::from_dir(Path::new(dir), EngineConfig::default()) {
            Ok(engine) => Box::into_raw(Box::new(OgeEngine(engine))),
            Err(err) => {
                set_last_error(format!("oge_engine_load: {err:#}"));
                ptr::null_mut()
            }
        }
    })
}

/// Free an engine created by `oge_engine_load`. Null-tolerant.
///
/// # Safety
/// `engine` must be null or a pointer previously returned by `oge_engine_load`
/// and not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oge_engine_free(engine: *mut OgeEngine) {
    if engine.is_null() {
        return;
    }
    guard("oge_engine_free", (), || {
        drop(unsafe { Box::from_raw(engine) });
    });
}

fn build_request(prompt: &str, max_new_tokens: usize) -> GenerateRequest {
    let mut request = GenerateRequest::new(prompt.to_string());
    if max_new_tokens > 0 {
        request.options = GenerateOptions {
            max_new_tokens,
            ..request.options
        };
    }
    request
}

fn result_to_cstring(context: &str, result: anyhow::Result<String>) -> *mut c_char {
    match result {
        Ok(text) => match CString::new(text) {
            Ok(cstring) => cstring.into_raw(),
            Err(_) => {
                set_last_error(format!(
                    "{context}: generated text contained an interior NUL"
                ));
                ptr::null_mut()
            }
        },
        Err(err) => {
            set_last_error(format!("{context}: {err:#}"));
            ptr::null_mut()
        }
    }
}

/// Generate text with the engine's default sampler (greedy/categorical per the
/// request options). Returns a heap `char*` (free with `oge_string_free`) or null
/// on failure (see [`oge_last_error`]). `max_new_tokens == 0` keeps the request
/// default.
///
/// # Safety
/// `engine` must be a valid engine pointer; `prompt` a valid NUL-terminated
/// UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oge_engine_generate(
    engine: *mut OgeEngine,
    prompt: *const c_char,
    max_new_tokens: usize,
) -> *mut c_char {
    clear_last_error();
    guard("oge_engine_generate", ptr::null_mut(), || {
        let engine = match unsafe { engine.as_mut() } {
            Some(engine) => engine,
            None => {
                set_last_error("oge_engine_generate: engine is null");
                return ptr::null_mut();
            }
        };
        let prompt = match unsafe { cstr_to_str("oge_engine_generate", prompt) } {
            Some(prompt) => prompt,
            None => return ptr::null_mut(),
        };
        let request = build_request(prompt, max_new_tokens);
        let result = engine.0.generate(request).map(|generated| generated.text);
        result_to_cstring("oge_engine_generate", result)
    })
}

/// Generate text using a caller-supplied sampler. **Consumes `sampler`** (it is
/// freed by this call, success or failure); do not use or free it afterwards.
/// Returns a heap `char*` (free with `oge_string_free`) or null on failure.
///
/// # Safety
/// `engine` must be a valid engine pointer; `prompt` a valid NUL-terminated
/// UTF-8 string; `sampler` a pointer from `oge_sampler_new` not yet freed/consumed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oge_engine_generate_with_sampler(
    engine: *mut OgeEngine,
    prompt: *const c_char,
    max_new_tokens: usize,
    sampler: *mut OgeSampler,
) -> *mut c_char {
    clear_last_error();
    guard("oge_engine_generate_with_sampler", ptr::null_mut(), || {
        if sampler.is_null() {
            set_last_error("oge_engine_generate_with_sampler: sampler is null");
            return ptr::null_mut();
        }
        // Take ownership of the sampler regardless of the outcome below.
        let sampler = unsafe { Box::from_raw(sampler) };
        let engine = match unsafe { engine.as_mut() } {
            Some(engine) => engine,
            None => {
                set_last_error("oge_engine_generate_with_sampler: engine is null");
                return ptr::null_mut();
            }
        };
        let prompt = match unsafe { cstr_to_str("oge_engine_generate_with_sampler", prompt) } {
            Some(prompt) => prompt,
            None => return ptr::null_mut(),
        };
        let request = build_request(prompt, max_new_tokens);
        let boxed: Box<dyn Sampler> = sampler.0;
        let result = engine
            .0
            .generate_with_sampler(request, boxed)
            .map(|generated| generated.text);
        result_to_cstring("oge_engine_generate_with_sampler", result)
    })
}

unsafe fn cstr_to_str<'a>(context: &str, ptr: *const c_char) -> Option<&'a str> {
    if ptr.is_null() {
        set_last_error(format!("{context}: prompt is null"));
        return None;
    }
    match unsafe { CStr::from_ptr(ptr) }.to_str() {
        Ok(text) => Some(text),
        Err(err) => {
            set_last_error(format!("{context}: prompt is not valid UTF-8: {err}"));
            None
        }
    }
}

/// Free a `char*` returned by any function in this library. Null-tolerant.
///
/// # Safety
/// `text` must be null or a pointer previously returned by this library and not
/// yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn oge_string_free(text: *mut c_char) {
    if text.is_null() {
        return;
    }
    guard("oge_string_free", (), || {
        drop(unsafe { CString::from_raw(text) });
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    // A foreign "sampler" that always picks the last token id, exercised through
    // the real extern "C" callback path (no model needed).
    unsafe extern "C" fn last_token_sample(
        _user_data: *mut c_void,
        _logits: *const f32,
        logits_len: usize,
        _generated: *const u32,
        _generated_len: usize,
        _step: usize,
    ) -> u32 {
        (logits_len - 1) as u32
    }

    #[test]
    fn foreign_sampler_invokes_callback() {
        let vtable = OgeSamplerVTable {
            user_data: ptr::null_mut(),
            sample: last_token_sample,
            name: c"last_token".as_ptr(),
            free: None,
        };
        let mut sampler = unsafe { ForeignSampler::from_vtable(vtable) };
        assert_eq!(sampler.name(), "last_token");

        let logits = [0.5_f32, 9.0, 0.1];
        let context = ProcessorContext::default();
        assert_eq!(sampler.sample(&logits, &context), 2);
    }

    static FREE_CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn counting_free(_user_data: *mut c_void) {
        FREE_CALLS.fetch_add(1, Ordering::SeqCst);
    }

    unsafe extern "C" fn zero_sample(
        _user_data: *mut c_void,
        _logits: *const f32,
        _logits_len: usize,
        _generated: *const u32,
        _generated_len: usize,
        _step: usize,
    ) -> u32 {
        0
    }

    #[test]
    fn free_callback_runs_exactly_once_on_drop() {
        FREE_CALLS.store(0, Ordering::SeqCst);
        let vtable = OgeSamplerVTable {
            user_data: ptr::null_mut(),
            sample: zero_sample,
            name: ptr::null(),
            free: Some(counting_free),
        };
        let sampler = unsafe { oge_sampler_new(vtable) };
        assert!(!sampler.is_null());
        unsafe { oge_sampler_free(sampler) };
        assert_eq!(FREE_CALLS.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn null_name_defaults_and_null_free_is_noop() {
        let vtable = OgeSamplerVTable {
            user_data: ptr::null_mut(),
            sample: zero_sample,
            name: ptr::null(),
            free: None,
        };
        let sampler = unsafe { ForeignSampler::from_vtable(vtable) };
        assert_eq!(sampler.name(), "foreign");
        drop(sampler); // no free callback -> must not crash
    }

    #[test]
    fn last_error_roundtrips() {
        clear_last_error();
        assert!(oge_last_error().is_null());
        // A null engine pointer records an error rather than dereferencing.
        let out = unsafe { oge_engine_generate(ptr::null_mut(), c"hi".as_ptr(), 0) };
        assert!(out.is_null());
        let err = oge_last_error();
        assert!(!err.is_null());
        let message = unsafe { CStr::from_ptr(err) }.to_string_lossy();
        assert!(message.contains("engine is null"), "got: {message}");
    }

    #[test]
    fn free_is_null_tolerant() {
        unsafe {
            oge_sampler_free(ptr::null_mut());
            oge_engine_free(ptr::null_mut());
            oge_string_free(ptr::null_mut());
        }
    }
}
