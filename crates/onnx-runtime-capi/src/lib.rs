//! # `onnx-runtime-capi`
//!
//! The C ABI layer for the ORT 2.0 runtime (see `docs/ORT2.md` §21). This is
//! Phase 1, **Tier 1**: a clean, direct `extern "C"` surface (`nxrt_*` names)
//! that lets a C caller load a model, build input tensors, run inference, and
//! read outputs back. It is a thin marshalling layer over
//! [`onnx_runtime_session`] — nothing here is model-, op-, or shape-specific.
//!
//! Phase 1 deliberately does **not** reproduce upstream ORT's `OrtApi` vtable
//! (that is Phase 2's `OrtGetApiBase`, §21.2). There are no backward-compat
//! shims.
//!
//! ## Safety model
//!
//! Every exported function that dereferences a caller pointer is `unsafe`-bodied
//! and documents its preconditions. The rules, enforced uniformly:
//!
//! * **Opaque handles** ([`OrtSession`], [`OrtValue`], [`OrtStatus`]) are
//!   created with [`Box::into_raw`] and freed with [`Box::from_raw`] *exactly
//!   once* by the matching `nxrt_release_*`. After release the caller must drop
//!   its copy of the pointer; reusing it is a use-after-free the API cannot
//!   detect (the standard C ownership contract). `release` is null-tolerant, so
//!   the idiomatic `release(x); x = NULL;` makes double-release unreachable.
//! * **Null checks**: every incoming handle/pointer is null-checked and turned
//!   into an [`OrtErrorCode::InvalidArgument`] status rather than dereferenced.
//! * **No panics cross the boundary**: every body runs inside
//!   [`std::panic::catch_unwind`]; a panic becomes an [`OrtErrorCode::Fail`]
//!   status instead of unwinding into C (which is undefined behavior).
//! * **Status convention**: fallible functions return `*mut OrtStatus` —
//!   `null` on success, an owned status the caller must
//!   [`nxrt_release_status`] on error.

use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;

use onnx_runtime_session::{InferenceSession, SessionBuilder, SessionError, Tensor};

// ---------------------------------------------------------------------------
// Status codes (§22)
// ---------------------------------------------------------------------------

/// ORT-compatible status code for the C API layer (`docs/ORT2.md` §22).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum OrtErrorCode {
    Ok = 0,
    Fail = 1,
    InvalidArgument = 2,
    NoSuchFile = 3,
    NoModel = 4,
    EngineMismatch = 5,
    InvalidProtobuf = 6,
    ModelLoaded = 7,
    NotImplemented = 8,
    InvalidGraph = 10,
    EpFail = 11,
}

/// The runtime version string, NUL-terminated for C consumers.
const VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();

/// Return the runtime version as a C string (`ORT`'s `GetVersionString`).
///
/// # Safety
/// The returned pointer is valid for the lifetime of the process and must not
/// be freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn nxrt_get_version_string() -> *const c_char {
    VERSION.as_ptr() as *const c_char
}

// ---------------------------------------------------------------------------
// OrtStatus — owned (code + message), returned as `*mut` on error, null on ok
// ---------------------------------------------------------------------------

/// An owned error status: a code plus a NUL-terminated message. Returned from
/// fallible entry points as `*mut OrtStatus` (null == success). The caller owns
/// any non-null status and must free it with [`nxrt_release_status`].
pub struct OrtStatus {
    code: OrtErrorCode,
    message: CString,
}

impl OrtStatus {
    /// Box a new status and leak it to the caller as a raw pointer.
    fn boxed(code: OrtErrorCode, message: impl Into<Vec<u8>>) -> *mut OrtStatus {
        // Sanitize interior NULs so the message is always a valid C string.
        let message = CString::new(message).unwrap_or_else(|_| {
            CString::new("status message contained an interior NUL").expect("literal is NUL-free")
        });
        Box::into_raw(Box::new(OrtStatus { code, message }))
    }
}

/// The internal result of an FFI body: `Ok(())` on success, or a
/// `(code, message)` pair to surface as an [`OrtStatus`].
type FfiResult = Result<(), (OrtErrorCode, String)>;

/// Run an FFI body, converting its outcome into the `*mut OrtStatus` contract
/// and catching any panic so it never unwinds into C (undefined behavior).
///
/// The closure is wrapped in [`AssertUnwindSafe`]: FFI bodies work through raw
/// pointers whose validity is the caller's precondition, and on panic we return
/// a status without exposing any partially-mutated state, so unwind-safety is
/// upheld by construction.
fn guard<F: FnOnce() -> FfiResult>(f: F) -> *mut OrtStatus {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => ptr::null_mut(),
        Ok(Err((code, msg))) => OrtStatus::boxed(code, msg),
        Err(_) => OrtStatus::boxed(
            OrtErrorCode::Fail,
            "panic caught at the FFI boundary (this is a runtime bug)",
        ),
    }
}

/// Map a [`SessionError`] to the closest ORT status code (§22).
fn map_session_error(err: &SessionError) -> OrtErrorCode {
    use onnx_runtime_session::SessionError as E;
    match err {
        E::InputNotFound { .. }
        | E::DtypeMismatch { .. }
        | E::ShapeMismatch { .. }
        | E::UnknownOption { .. }
        | E::InvalidOption { .. }
        | E::DynamicShape { .. }
        | E::SymbolConflict { .. }
        | E::RankMismatch { .. } => OrtErrorCode::InvalidArgument,
        E::NoModelSource => OrtErrorCode::NoModel,
        E::UnsupportedOp { .. } => OrtErrorCode::NotImplemented,
        E::Ep(_) => OrtErrorCode::EpFail,
        E::DanglingEpContext { .. }
        | E::Ir(_)
        | E::Graph(_)
        | E::Optimize(_)
        | E::ShapeInfer(_) => OrtErrorCode::InvalidGraph,
        E::Load(load) => map_loader_error(load),
        E::NotInitialized
        | E::Internal(_)
        | E::UnresolvedShape { .. }
        | E::ShapeOverflow { .. }
        | E::OutputShapeCountMismatch { .. }
        | E::SequenceOp { .. } => OrtErrorCode::Fail,
    }
}

/// Map a [`onnx_runtime_loader::LoaderError`] to an ORT status code.
fn map_loader_error(err: &onnx_runtime_loader::LoaderError) -> OrtErrorCode {
    use onnx_runtime_loader::LoaderError as L;
    match err {
        L::Io { source, .. } if source.kind() == std::io::ErrorKind::NotFound => {
            OrtErrorCode::NoSuchFile
        }
        L::ExternalDataNotFound { .. } => OrtErrorCode::NoSuchFile,
        L::ProtobufParse(_) => OrtErrorCode::InvalidProtobuf,
        L::ExternalDataPath { .. }
        | L::EpContextPath { .. }
        | L::Ir(_)
        | L::GraphBuild(_) => OrtErrorCode::InvalidGraph,
        _ => OrtErrorCode::Fail,
    }
}

/// Turn a `SessionError` into the FFI error pair.
fn session_err(err: SessionError) -> (OrtErrorCode, String) {
    (map_session_error(&err), err.to_string())
}

/// Get the status code of a status. Returns [`OrtErrorCode::Ok`] for a null
/// status (which by convention means "no error").
///
/// # Safety
/// `status`, if non-null, must be a pointer returned by this library and not
/// yet released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_get_error_code(status: *const OrtStatus) -> OrtErrorCode {
    if status.is_null() {
        return OrtErrorCode::Ok;
    }
    // SAFETY: non-null and, per the documented precondition, a live status
    // produced by this library. We only take a shared borrow.
    unsafe { (*status).code }
}

/// Get the NUL-terminated error message of a status. Returns an empty string
/// for a null status. The pointer is valid until the status is released.
///
/// # Safety
/// `status`, if non-null, must be a pointer returned by this library and not
/// yet released. The returned pointer must not be freed by the caller and must
/// not be used after [`nxrt_release_status`] frees `status`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_get_error_message(status: *const OrtStatus) -> *const c_char {
    if status.is_null() {
        return c"".as_ptr();
    }
    // SAFETY: non-null and a live status per the precondition; the returned
    // pointer borrows the status's `CString`, valid until release.
    unsafe { (*status).message.as_ptr() }
}

/// Free a status returned by this library. Null-tolerant. Must be called
/// *exactly once* per non-null status.
///
/// # Safety
/// `status`, if non-null, must be a pointer returned by this library that has
/// not already been released. After this call the pointer is dangling and must
/// not be used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_release_status(status: *mut OrtStatus) {
    if status.is_null() {
        return;
    }
    // SAFETY: non-null and, per the precondition, an as-yet-unreleased status
    // created by `OrtStatus::boxed` via `Box::into_raw`. Reconstituting the box
    // frees it exactly once.
    drop(unsafe { Box::from_raw(status) });
}

// ---------------------------------------------------------------------------
// OrtSession — opaque wrapper over InferenceSession
// ---------------------------------------------------------------------------

/// Opaque session handle wrapping an [`InferenceSession`].
pub struct OrtSession {
    inner: InferenceSession,
}

/// Create a session from an on-disk model path, writing an owned session handle
/// to `*out` on success. Wraps [`InferenceSession::load`] (auto CPU device).
///
/// On success returns null and `*out` holds a handle to release with
/// [`nxrt_release_session`]. On error returns a status and leaves `*out` null.
///
/// # Safety
/// * `model_path` must be a non-null pointer to a NUL-terminated C string that
///   is valid UTF-8.
/// * `out` must be a non-null, writable, well-aligned pointer to a
///   `*mut OrtSession` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_create_session(
    model_path: *const c_char,
    out: *mut *mut OrtSession,
) -> *mut OrtStatus {
    guard(|| {
        if out.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "out pointer is null".into()));
        }
        // SAFETY: `out` is non-null and writable per the precondition.
        unsafe { *out = ptr::null_mut() };
        if model_path.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "model_path is null".into()));
        }
        // SAFETY: non-null NUL-terminated C string per the precondition.
        let path = unsafe { CStr::from_ptr(model_path) }
            .to_str()
            .map_err(|_| (OrtErrorCode::InvalidArgument, "model_path is not valid UTF-8".into()))?;

        let inner = InferenceSession::load(path).map_err(session_err)?;
        let handle = Box::into_raw(Box::new(OrtSession { inner }));
        // SAFETY: `out` validated non-null and writable above.
        unsafe { *out = handle };
        Ok(())
    })
}

/// Free a session created by [`nxrt_create_session`]. Null-tolerant. Must be
/// called *exactly once* per non-null session.
///
/// # Safety
/// `session`, if non-null, must be a handle from [`nxrt_create_session`] that
/// has not already been released. After this call the pointer is dangling.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_release_session(session: *mut OrtSession) {
    if session.is_null() {
        return;
    }
    // SAFETY: non-null and an unreleased handle per the precondition.
    drop(unsafe { Box::from_raw(session) });
}

// ---------------------------------------------------------------------------
// OrtSessionOptions — string key/value session configuration (§21.4 / §55.5)
// ---------------------------------------------------------------------------

/// Opaque session-options handle: an ordered bag of string key/value config
/// entries set via [`nxrt_add_session_config_entry`] (ORT's
/// `AddSessionConfigEntry`) and consumed by [`nxrt_create_session_with_options`].
///
/// This is the C-API path by which ORT tooling sets the `ep.context_*` options
/// (§21.4) that drive the EPContext dump: the entries are forwarded verbatim to
/// [`SessionBuilder::option`], so their parsing/validation lives in one place
/// (the session layer) and the C API adds no divergent option logic.
#[derive(Default)]
pub struct OrtSessionOptions {
    entries: Vec<(String, String)>,
}

/// Create an empty session-options handle, writing it to `*out`.
///
/// On success returns null and `*out` holds a handle to release with
/// [`nxrt_release_session_options`].
///
/// # Safety
/// `out` must be a non-null, writable, well-aligned pointer to a
/// `*mut OrtSessionOptions` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_create_session_options(
    out: *mut *mut OrtSessionOptions,
) -> *mut OrtStatus {
    guard(|| {
        if out.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "out pointer is null".into()));
        }
        // SAFETY: `out` is non-null and writable per the precondition.
        unsafe { *out = Box::into_raw(Box::new(OrtSessionOptions::default())) };
        Ok(())
    })
}

/// Free a session-options handle from [`nxrt_create_session_options`].
/// Null-tolerant. Must be called *exactly once* per non-null handle.
///
/// # Safety
/// `options`, if non-null, must be an unreleased handle from
/// [`nxrt_create_session_options`]. After this call the pointer is dangling.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_release_session_options(options: *mut OrtSessionOptions) {
    if options.is_null() {
        return;
    }
    // SAFETY: non-null and an unreleased handle per the precondition.
    drop(unsafe { Box::from_raw(options) });
}

/// Add a string key/value config entry to `options` (ORT's
/// `AddSessionConfigEntry`). The entry is stored verbatim and only validated
/// when a session is built from these options — an unknown key or invalid value
/// surfaces then as `InvalidArgument` (mirroring [`SessionBuilder::build`]).
///
/// The `ep.context_enable` / `ep.context_file_path` / `ep.context_embed_mode`
/// keys (§21.4) reach the session's EPContext dump config this way.
///
/// # Safety
/// * `options` must be a non-null, unreleased handle from
///   [`nxrt_create_session_options`].
/// * `key` and `value` must be non-null pointers to NUL-terminated, valid UTF-8
///   C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_add_session_config_entry(
    options: *mut OrtSessionOptions,
    key: *const c_char,
    value: *const c_char,
) -> *mut OrtStatus {
    guard(|| {
        if options.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "options handle is null".into()));
        }
        if key.is_null() || value.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "key or value pointer is null".into()));
        }
        // SAFETY: non-null handle per the precondition.
        let options = unsafe { &mut *options };
        // SAFETY: non-null NUL-terminated C strings per the precondition.
        let key = unsafe { CStr::from_ptr(key) }
            .to_str()
            .map_err(|_| (OrtErrorCode::InvalidArgument, "key is not valid UTF-8".into()))?;
        let value = unsafe { CStr::from_ptr(value) }
            .to_str()
            .map_err(|_| (OrtErrorCode::InvalidArgument, "value is not valid UTF-8".into()))?;
        options.entries.push((key.to_string(), value.to_string()));
        Ok(())
    })
}

/// Create a session from an on-disk model path, applying the string key/value
/// config entries in `options` (§21.4 / §55.5). Each entry is forwarded to
/// [`SessionBuilder::option`], so option parsing/validation is the session
/// layer's (an unknown key or bad value fails here as `InvalidArgument`).
///
/// `options` may be null, in which case this behaves like
/// [`nxrt_create_session`] (no extra options).
///
/// # Safety
/// * `model_path` must be a non-null pointer to a NUL-terminated, valid UTF-8
///   C string.
/// * `options`, if non-null, must be an unreleased handle from
///   [`nxrt_create_session_options`].
/// * `out` must be a non-null, writable, well-aligned pointer to a
///   `*mut OrtSession` slot.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_create_session_with_options(
    model_path: *const c_char,
    options: *const OrtSessionOptions,
    out: *mut *mut OrtSession,
) -> *mut OrtStatus {
    guard(|| {
        if out.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "out pointer is null".into()));
        }
        // SAFETY: `out` is non-null and writable per the precondition.
        unsafe { *out = ptr::null_mut() };
        if model_path.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "model_path is null".into()));
        }
        // SAFETY: non-null NUL-terminated C string per the precondition.
        let path = unsafe { CStr::from_ptr(model_path) }
            .to_str()
            .map_err(|_| (OrtErrorCode::InvalidArgument, "model_path is not valid UTF-8".into()))?;

        let mut builder = SessionBuilder::new().model(path);
        if !options.is_null() {
            // SAFETY: non-null and an unreleased handle per the precondition.
            for (key, value) in &unsafe { &*options }.entries {
                builder = builder.option(key, value);
            }
        }

        let inner = builder.build().map_err(session_err)?;
        let handle = Box::into_raw(Box::new(OrtSession { inner }));
        // SAFETY: `out` validated non-null and writable above.
        unsafe { *out = handle };
        Ok(())
    })
}


/// Opaque tensor value handle wrapping an owned [`Tensor`].
pub struct OrtValue {
    inner: Tensor,
}

/// Total element count for a caller-provided shape, rejecting negative dims and
/// overflow.
fn shape_numel(shape: &[i64]) -> Result<usize, (OrtErrorCode, String)> {
    let mut numel: usize = 1;
    for (i, &d) in shape.iter().enumerate() {
        if d < 0 {
            return Err((
                OrtErrorCode::InvalidArgument,
                format!("shape dim {i} is negative ({d})"),
            ));
        }
        numel = numel.checked_mul(d as usize).ok_or((
            OrtErrorCode::InvalidArgument,
            "shape element count overflows usize".into(),
        ))?;
    }
    Ok(numel)
}

/// Create an input tensor value from caller-owned bytes, copying them into an
/// owned [`Tensor`]. Validates that `data_len` equals the storage size implied
/// by `dtype` and `shape`.
///
/// `data_type` is the ONNX `TensorProto.DataType` integer (e.g. 1 = FLOAT,
/// 7 = INT64). The bytes are interpreted little-endian, matching the runtime.
///
/// # Safety
/// * `out` must be a non-null, writable, well-aligned `*mut OrtValue` slot.
/// * `shape` must point to `rank` readable, well-aligned `i64` values (may be
///   null iff `rank == 0`, denoting a scalar).
/// * `data` must point to `data_len` readable bytes (may be null iff
///   `data_len == 0`). The bytes are copied; the caller retains ownership.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_create_tensor(
    data: *const c_void,
    data_len: usize,
    shape: *const i64,
    rank: usize,
    data_type: i32,
    out: *mut *mut OrtValue,
) -> *mut OrtStatus {
    guard(|| {
        if out.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "out pointer is null".into()));
        }
        // SAFETY: `out` is non-null and writable per the precondition.
        unsafe { *out = ptr::null_mut() };

        let dtype = onnx_runtime_ir::DataType::from_onnx(data_type).ok_or((
            OrtErrorCode::InvalidArgument,
            format!("unsupported ONNX data_type {data_type}"),
        ))?;

        if shape.is_null() && rank != 0 {
            return Err((
                OrtErrorCode::InvalidArgument,
                "shape is null but rank is non-zero".into(),
            ));
        }
        // SAFETY: `shape` points to `rank` readable i64s per the precondition;
        // when rank == 0 we skip the read entirely.
        let dims: Vec<i64> = if rank == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(shape, rank) }.to_vec()
        };
        let numel = shape_numel(&dims)?;
        let expected = dtype.storage_bytes(numel);
        if data_len != expected {
            return Err((
                OrtErrorCode::InvalidArgument,
                format!(
                    "data_len {data_len} does not match {expected} bytes for dtype {dtype:?} shape {dims:?}"
                ),
            ));
        }
        if data.is_null() && data_len != 0 {
            return Err((
                OrtErrorCode::InvalidArgument,
                "data is null but data_len is non-zero".into(),
            ));
        }
        // SAFETY: `data` points to `data_len` readable bytes per the
        // precondition; when data_len == 0 we produce an empty slice.
        let bytes: &[u8] = if data_len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(data as *const u8, data_len) }
        };

        let shape_usize: Vec<usize> = dims.iter().map(|&d| d as usize).collect();
        let tensor = Tensor::from_raw(dtype, shape_usize, bytes).map_err(session_err)?;
        let handle = Box::into_raw(Box::new(OrtValue { inner: tensor }));
        // SAFETY: `out` validated non-null and writable above.
        unsafe { *out = handle };
        Ok(())
    })
}

/// Free a value created by [`nxrt_create_tensor`] or returned from
/// [`nxrt_run`]. Null-tolerant. Must be called *exactly once* per non-null
/// value.
///
/// # Safety
/// `value`, if non-null, must be a handle produced by this library that has not
/// already been released. After this call the pointer is dangling.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_release_value(value: *mut OrtValue) {
    if value.is_null() {
        return;
    }
    // SAFETY: non-null and an unreleased handle per the precondition.
    drop(unsafe { Box::from_raw(value) });
}

/// Write the ONNX `TensorProto.DataType` integer of a value to `*out`.
///
/// # Safety
/// * `value` must be a non-null, unreleased [`OrtValue`] handle.
/// * `out` must be a non-null, writable, well-aligned `*mut i32`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_get_tensor_dtype(
    value: *const OrtValue,
    out: *mut i32,
) -> *mut OrtStatus {
    guard(|| {
        // SAFETY: precondition — non-null unreleased handle, or null (handled).
        let value = unsafe { value.as_ref() }
            .ok_or((OrtErrorCode::InvalidArgument, "value handle is null".into()))?;
        if out.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "out pointer is null".into()));
        }
        // SAFETY: `out` non-null and writable per the precondition.
        unsafe { *out = value.inner.dtype.to_onnx() };
        Ok(())
    })
}

/// Write the rank (number of dimensions) of a value to `*out`.
///
/// # Safety
/// * `value` must be a non-null, unreleased [`OrtValue`] handle.
/// * `out` must be a non-null, writable, well-aligned `*mut usize`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_get_tensor_rank(
    value: *const OrtValue,
    out: *mut usize,
) -> *mut OrtStatus {
    guard(|| {
        // SAFETY: precondition — non-null unreleased handle, or null (handled).
        let value = unsafe { value.as_ref() }
            .ok_or((OrtErrorCode::InvalidArgument, "value handle is null".into()))?;
        if out.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "out pointer is null".into()));
        }
        // SAFETY: `out` non-null and writable per the precondition.
        unsafe { *out = value.inner.shape.len() };
        Ok(())
    })
}

/// Copy the value's dimensions into the caller's `out_dims` buffer. `rank` must
/// equal the value's rank (query it first with [`nxrt_get_tensor_rank`]); a
/// mismatch is rejected as an error.
///
/// # Safety
/// * `value` must be a non-null, unreleased [`OrtValue`] handle.
/// * `out_dims` must point to `rank` writable, well-aligned `i64` slots (may be
///   null iff `rank == 0`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_get_tensor_shape(
    value: *const OrtValue,
    out_dims: *mut i64,
    rank: usize,
) -> *mut OrtStatus {
    guard(|| {
        // SAFETY: precondition — non-null unreleased handle, or null (handled).
        let value = unsafe { value.as_ref() }
            .ok_or((OrtErrorCode::InvalidArgument, "value handle is null".into()))?;
        let shape = &value.inner.shape;
        if rank != shape.len() {
            return Err((
                OrtErrorCode::InvalidArgument,
                format!("rank {rank} does not match tensor rank {}", shape.len()),
            ));
        }
        if rank == 0 {
            return Ok(());
        }
        if out_dims.is_null() {
            return Err((OrtErrorCode::InvalidArgument, "out_dims is null".into()));
        }
        // SAFETY: `out_dims` points to `rank` writable i64s per the precondition
        // and `rank == shape.len()`, so every write is in bounds.
        let dst = unsafe { std::slice::from_raw_parts_mut(out_dims, rank) };
        for (slot, &d) in dst.iter_mut().zip(shape.iter()) {
            *slot = d as i64;
        }
        Ok(())
    })
}

/// Expose a read-only pointer to a value's raw little-endian element bytes and
/// its byte length. The pointer is valid until the value is released.
///
/// # Safety
/// * `value` must be a non-null, unreleased [`OrtValue`] handle.
/// * `out_data` must be a non-null, writable `*mut *const c_void`.
/// * `out_len` must be a non-null, writable `*mut usize`.
/// * The returned pointer must not be used after the value is released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn nxrt_get_tensor_data(
    value: *const OrtValue,
    out_data: *mut *const c_void,
    out_len: *mut usize,
) -> *mut OrtStatus {
    guard(|| {
        // SAFETY: precondition — non-null unreleased handle, or null (handled).
        let value = unsafe { value.as_ref() }
            .ok_or((OrtErrorCode::InvalidArgument, "value handle is null".into()))?;
        if out_data.is_null() || out_len.is_null() {
            return Err((
                OrtErrorCode::InvalidArgument,
                "out_data or out_len is null".into(),
            ));
        }
        let bytes = value.inner.as_bytes();
        // SAFETY: both out pointers are non-null and writable per the
        // precondition. The data pointer borrows the tensor's buffer, valid
        // until the value is released.
        unsafe {
            *out_data = bytes.as_ptr() as *const c_void;
            *out_len = bytes.len();
        }
        Ok(())
    })
}

// ---------------------------------------------------------------------------
// Run
// ---------------------------------------------------------------------------

/// Run inference. Binds `n_inputs` named input values, drives
/// [`InferenceSession::run`], then writes `n_outputs` owned output value handles
/// (selected by `output_names`) into `out_values`.
///
/// Output order follows `output_names`: each requested name is matched against
/// the model's declared outputs. On any error nothing is written to the
/// caller's slots (they are pre-nulled) and all partially-produced handles are
/// freed, so the caller never owns a leaked or half-initialized value.
///
/// # Safety
/// * `session` must be a non-null, unreleased [`OrtSession`] handle.
/// * `input_names` / `input_values` must each point to `n_inputs` readable
///   elements (may be null iff `n_inputs == 0`). Each name must be a non-null
///   NUL-terminated UTF-8 C string; each value a non-null unreleased
///   [`OrtValue`].
/// * `output_names` must point to `n_outputs` readable non-null NUL-terminated
///   UTF-8 C strings (may be null iff `n_outputs == 0`).
/// * `out_values` must point to `n_outputs` writable `*mut OrtValue` slots (may
///   be null iff `n_outputs == 0`).
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn nxrt_run(
    session: *mut OrtSession,
    input_names: *const *const c_char,
    input_values: *const *const OrtValue,
    n_inputs: usize,
    output_names: *const *const c_char,
    n_outputs: usize,
    out_values: *mut *mut OrtValue,
) -> *mut OrtStatus {
    guard(|| {
        // SAFETY: precondition — non-null unreleased handle, or null (handled).
        let session = unsafe { session.as_mut() }
            .ok_or((OrtErrorCode::InvalidArgument, "session handle is null".into()))?;

        // Pre-null the output slots so an early error leaves no dangling caller
        // pointers, and validate the output arrays.
        if n_outputs != 0 {
            if out_values.is_null() {
                return Err((
                    OrtErrorCode::InvalidArgument,
                    "out_values is null but n_outputs is non-zero".into(),
                ));
            }
            if output_names.is_null() {
                return Err((
                    OrtErrorCode::InvalidArgument,
                    "output_names is null but n_outputs is non-zero".into(),
                ));
            }
            // SAFETY: `out_values` points to `n_outputs` writable slots.
            let slots = unsafe { std::slice::from_raw_parts_mut(out_values, n_outputs) };
            for slot in slots.iter_mut() {
                *slot = ptr::null_mut();
            }
        }

        if n_inputs != 0 && (input_names.is_null() || input_values.is_null()) {
            return Err((
                OrtErrorCode::InvalidArgument,
                "input_names or input_values is null but n_inputs is non-zero".into(),
            ));
        }

        // Marshal inputs into borrowed (name, &Tensor) pairs.
        let mut inputs: Vec<(&str, &Tensor)> = Vec::with_capacity(n_inputs);
        for i in 0..n_inputs {
            // SAFETY: `input_names`/`input_values` each hold `n_inputs` elems.
            let name_ptr = unsafe { *input_names.add(i) };
            let val_ptr = unsafe { *input_values.add(i) };
            if name_ptr.is_null() {
                return Err((
                    OrtErrorCode::InvalidArgument,
                    format!("input name #{i} is null"),
                ));
            }
            // SAFETY: non-null NUL-terminated C string per the precondition.
            let name = unsafe { CStr::from_ptr(name_ptr) }.to_str().map_err(|_| {
                (
                    OrtErrorCode::InvalidArgument,
                    format!("input name #{i} is not valid UTF-8"),
                )
            })?;
            // SAFETY: non-null unreleased handle, or null (handled).
            let value = unsafe { val_ptr.as_ref() }.ok_or((
                OrtErrorCode::InvalidArgument,
                format!("input value #{i} is null"),
            ))?;
            inputs.push((name, &value.inner));
        }

        // Resolve requested output names before running so a bad name fails
        // fast. Outputs come back in the model's declared order.
        let declared: Vec<&str> =
            session.inner.outputs().iter().map(|m| m.name.as_str()).collect();
        let mut want: Vec<usize> = Vec::with_capacity(n_outputs);
        for i in 0..n_outputs {
            // SAFETY: `output_names` holds `n_outputs` elems (validated above).
            let name_ptr = unsafe { *output_names.add(i) };
            if name_ptr.is_null() {
                return Err((
                    OrtErrorCode::InvalidArgument,
                    format!("output name #{i} is null"),
                ));
            }
            // SAFETY: non-null NUL-terminated C string per the precondition.
            let name = unsafe { CStr::from_ptr(name_ptr) }.to_str().map_err(|_| {
                (
                    OrtErrorCode::InvalidArgument,
                    format!("output name #{i} is not valid UTF-8"),
                )
            })?;
            let pos = declared.iter().position(|d| *d == name).ok_or((
                OrtErrorCode::InvalidArgument,
                format!("requested output {name:?} is not a model output"),
            ))?;
            want.push(pos);
        }

        let outputs = session.inner.run(&inputs).map_err(session_err)?;
        // Wrap in `Option` so we can move selected tensors out by index.
        let mut outputs: Vec<Option<Tensor>> = outputs.into_iter().map(Some).collect();

        // Box the selected outputs. On failure, free everything already boxed
        // so no handle leaks.
        let mut produced: Vec<*mut OrtValue> = Vec::with_capacity(n_outputs);
        for &pos in &want {
            let tensor = match outputs.get_mut(pos).and_then(Option::take) {
                Some(t) => t,
                None => {
                    for p in produced {
                        // SAFETY: each `p` was just produced by `Box::into_raw`.
                        drop(unsafe { Box::from_raw(p) });
                    }
                    return Err((
                        OrtErrorCode::Fail,
                        format!("output index {pos} requested more than once"),
                    ));
                }
            };
            produced.push(Box::into_raw(Box::new(OrtValue { inner: tensor })));
        }

        // Commit: write handles into the caller's slots.
        for (i, handle) in produced.into_iter().enumerate() {
            // SAFETY: `out_values` holds `n_outputs == produced.len()` slots.
            unsafe { *out_values.add(i) = handle };
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nul_terminated() {
        assert_eq!(*VERSION.last().unwrap(), 0);
        let ptr = nxrt_get_version_string();
        assert!(!ptr.is_null());
    }

    #[test]
    fn status_codes_match_ort() {
        assert_eq!(OrtErrorCode::Ok as i32, 0);
        assert_eq!(OrtErrorCode::InvalidGraph as i32, 10);
        assert_eq!(OrtErrorCode::EpFail as i32, 11);
    }

    #[test]
    fn null_status_accessors_are_ok() {
        // A null status means "no error" by convention.
        assert_eq!(
            unsafe { nxrt_get_error_code(ptr::null()) },
            OrtErrorCode::Ok
        );
        let msg = unsafe { nxrt_get_error_message(ptr::null()) };
        assert!(!msg.is_null());
        // Releasing null is a no-op (idempotent guard against double-free).
        unsafe { nxrt_release_status(ptr::null_mut()) };
    }

    #[test]
    fn session_error_mapping_covers_structural_and_shape_variants() {
        use onnx_runtime_session::SessionError as E;

        assert_eq!(
            map_session_error(&E::DanglingEpContext {
                source_key: Some("QNN".into()),
                partition_name: Some("encoder".into()),
            }),
            OrtErrorCode::InvalidGraph
        );

        // Input-validation failures surface as INVALID_ARGUMENT, consistent
        // with the existing dtype/shape/rank mismatch mappings.
        assert_eq!(
            map_session_error(&E::SymbolConflict {
                symbol: "N".into(),
                first: 2,
                second: 3,
            }),
            OrtErrorCode::InvalidArgument
        );
        assert_eq!(
            map_session_error(&E::RankMismatch {
                name: "x".into(),
                expected: 2,
                got: 3,
            }),
            OrtErrorCode::InvalidArgument
        );

        // Internal shape-resolution failures are not caused by caller
        // arguments, so they map to FAIL.
        assert_eq!(
            map_session_error(&E::UnresolvedShape {
                value: "y".into(),
                op: "Reshape".into(),
            }),
            OrtErrorCode::Fail
        );
        assert_eq!(
            map_session_error(&E::ShapeOverflow {
                value: "y".into(),
                dims: vec![usize::MAX, 2],
            }),
            OrtErrorCode::Fail
        );
        assert_eq!(
            map_session_error(&E::OutputShapeCountMismatch {
                op: "NonZero".into(),
                expected: 1,
                got: 2,
            }),
            OrtErrorCode::Fail
        );
    }
}
