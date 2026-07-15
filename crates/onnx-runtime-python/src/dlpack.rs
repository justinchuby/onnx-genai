//! Zero-copy DLPack **export** for nxrt output tensors (the Python Array-API
//! producer protocol).
//!
//! [`NxrtValue`] wraps an owned nxrt [`Tensor`] behind an `Arc` and implements
//! `__dlpack__` / `__dlpack_device__`, so consumers such as `torch.from_dlpack`
//! and `numpy.from_dlpack` borrow nxrt's buffer with **no copy**. A `.numpy()`
//! method preserves the old copy-based access, so this type is a superset of the
//! previous behaviour rather than a replacement.
//!
//! ## Why this module (and the crate's "no `unsafe`" note) changed
//!
//! Exporting DLPack requires two things `numpy`'s buffer protocol cannot give
//! us: a raw `DLManagedTensor` whose lifetime we control, and a `PyCapsule`
//! whose destructor implements the DLPack ownership handshake. The struct/ABI
//! and the memory-owning `deleter` are isolated in the dependency-free
//! [`onnx_runtime_dlpack`] crate; the only `unsafe` *here* is the thin PyCapsule
//! FFI (creating the `"dltensor"` capsule and its name-checking destructor),
//! which is inherently tied to `pyo3::ffi` and cannot be pushed further down.
//! The crate docs (`lib.rs`) were updated to say so honestly.
//!
//! ## The capsule ownership handshake (the double-free/leak trap)
//!
//! Per the Array-API standard:
//!
//! * `__dlpack__` returns a `PyCapsule` named `"dltensor"` wrapping a
//!   `*mut DLManagedTensor` we allocated (its `deleter` owns an `Arc<Tensor>`).
//! * When a consumer accepts it, **the consumer** renames the capsule to
//!   `"used_dltensor"` and becomes responsible for calling the `deleter`.
//! * If no consumer ever takes it, Python eventually garbage-collects the
//!   capsule and runs *our* destructor ([`dlpack_capsule_destructor`]), which
//!   must call the `deleter` itself — but only if the capsule is still named
//!   `"dltensor"`. The `"used_dltensor"` name-check is what prevents the classic
//!   double-free.

use std::ffi::{CStr, c_void};

use onnx_runtime_ep_api::DeviceBuffer;
use onnx_runtime_ir::{DataType, DeviceId, DeviceType, TensorLayout};
use onnx_runtime_session::{Tensor, cpu_allocator};
use pyo3::exceptions::{PyBufferError, PyTypeError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::sync::Arc;

/// The capsule name for an unconsumed DLPack tensor.
const DLTENSOR: &CStr = c"dltensor";
/// The capsule name a consumer renames to after taking ownership.
const USED_DLTENSOR: &CStr = c"used_dltensor";
/// The capsule name for an unconsumed *versioned* DLPack tensor.
const DLTENSOR_VERSIONED: &CStr = c"dltensor_versioned";
/// The versioned capsule name a consumer renames to after taking ownership.
const USED_DLTENSOR_VERSIONED: &CStr = c"used_dltensor_versioned";

/// Map an nxrt [`DataType`] to a DLPack [`DLDataType`](onnx_runtime_dlpack::DLDataType).
///
/// Only dtypes with an unambiguous DLPack encoding that consumers can actually
/// import are supported; the rest (sub-byte, float8, string) return an
/// actionable `TypeError` naming the offending dtype rather than exporting a
/// tensor no consumer can read.
fn to_dldatatype(dtype: DataType) -> PyResult<onnx_runtime_dlpack::DLDataType> {
    use onnx_runtime_dlpack as dl;
    let (code, bits) = match dtype {
        DataType::Bool => (dl::DL_BOOL, 8),
        DataType::Int8 => (dl::DL_INT, 8),
        DataType::Int16 => (dl::DL_INT, 16),
        DataType::Int32 => (dl::DL_INT, 32),
        DataType::Int64 => (dl::DL_INT, 64),
        DataType::Uint8 => (dl::DL_UINT, 8),
        DataType::Uint16 => (dl::DL_UINT, 16),
        DataType::Uint32 => (dl::DL_UINT, 32),
        DataType::Uint64 => (dl::DL_UINT, 64),
        DataType::Float16 => (dl::DL_FLOAT, 16),
        DataType::Float32 => (dl::DL_FLOAT, 32),
        DataType::Float64 => (dl::DL_FLOAT, 64),
        DataType::BFloat16 => (dl::DL_BFLOAT, 16),
        other => {
            return Err(PyTypeError::new_err(format!(
                "output tensor has dtype {other:?}, which nxrt cannot yet export \
                 over DLPack (supported: bool, int8/16/32/64, uint8/16/32/64, \
                 float16/32/64, bfloat16). Use `.numpy()` if a copy is \
                 acceptable, or file for DLPack support of this dtype."
            )));
        }
    };
    Ok(onnx_runtime_dlpack::DLDataType { code, bits, lanes: 1 })
}

/// Map a DLPack [`DLDataType`](onnx_runtime_dlpack::DLDataType) back to an nxrt
/// [`DataType`] — the inverse of [`to_dldatatype`], used on the zero-copy import
/// path.
///
/// Only scalar (`lanes == 1`) encodings nxrt can represent losslessly are
/// accepted; anything else (packed SIMD lanes, sub-byte int4/uint4, float8,
/// complex) returns an actionable `TypeError` naming the offending
/// code/bits so the caller can fall back or cast.
fn from_dldatatype(dt: onnx_runtime_dlpack::DLDataType) -> PyResult<DataType> {
    use onnx_runtime_dlpack as dl;
    if dt.lanes != 1 {
        return Err(PyTypeError::new_err(format!(
            "DLPack input has {} SIMD lanes; nxrt only imports scalar (lanes=1) \
             tensors zero-copy. Provide a de-interleaved array or use a copy.",
            dt.lanes
        )));
    }
    let ty = match (dt.code, dt.bits) {
        (dl::DL_BOOL, 8) => DataType::Bool,
        (dl::DL_INT, 8) => DataType::Int8,
        (dl::DL_INT, 16) => DataType::Int16,
        (dl::DL_INT, 32) => DataType::Int32,
        (dl::DL_INT, 64) => DataType::Int64,
        (dl::DL_UINT, 8) => DataType::Uint8,
        (dl::DL_UINT, 16) => DataType::Uint16,
        (dl::DL_UINT, 32) => DataType::Uint32,
        (dl::DL_UINT, 64) => DataType::Uint64,
        (dl::DL_FLOAT, 16) => DataType::Float16,
        (dl::DL_FLOAT, 32) => DataType::Float32,
        (dl::DL_FLOAT, 64) => DataType::Float64,
        (dl::DL_BFLOAT, 16) => DataType::BFloat16,
        (code, bits) => {
            return Err(PyTypeError::new_err(format!(
                "DLPack input has dtype (code={code}, bits={bits}), which nxrt \
                 cannot represent for a zero-copy import (supported: bool, \
                 int8/16/32/64, uint8/16/32/64, float16/32/64, bfloat16). \
                 sub-byte (int4/uint4), float8 and complex types are not \
                 importable; cast the array to a supported dtype."
            )));
        }
    };
    Ok(ty)
}


/// Map an nxrt [`Tensor`]'s device to a DLPack
/// [`DLDevice`](onnx_runtime_dlpack::DLDevice).
///
/// Host-accessible devices (CPU, and Apple MLX unified memory) export as
/// `kDLCPU`. CUDA is intentionally rejected in this pass (see [`NxrtValue`]);
/// the ABI already carries `kDLCUDA`, so wiring it in is additive.
fn to_dldevice(tensor: &Tensor) -> PyResult<onnx_runtime_dlpack::DLDevice> {
    let dev = tensor.device();
    match dev.device_type {
        DeviceType::Cpu | DeviceType::Mlx => Ok(onnx_runtime_dlpack::DLDevice {
            device_type: onnx_runtime_dlpack::DL_CPU,
            device_id: 0,
        }),
        DeviceType::Cuda => Err(PyBufferError::new_err(
            "this nxrt output lives in CUDA device memory; zero-copy DLPack \
             export for CUDA tensors is not implemented yet (the DLPack ABI \
             carries kDLCUDA + stream semantics, but the producer-side stream \
             ordering is not wired up in this build). Move the value to host \
             first, or use `.numpy()`.",
        )),
        other => Err(PyBufferError::new_err(format!(
            "nxrt output lives on device {other:?}, which has no DLPack export \
             path yet. Use `.numpy()` to obtain a host copy."
        ))),
    }
}

/// PyCapsule destructor for an **unconsumed** `"dltensor"` capsule.
///
/// # Safety
///
/// Installed only on capsules created by [`NxrtValue::__dlpack__`], whose stored
/// pointer is a live `*mut DLManagedTensor` from `onnx_runtime_dlpack::export`.
/// Python calls this at most once, when the capsule is collected.
unsafe extern "C" fn dlpack_capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: `capsule` is the PyCapsule being finalized. If a consumer took the
    // tensor it renamed the capsule to "used_dltensor" and owns the deleter, so
    // we must NOT free it here (that is the double-free trap). Only when the
    // capsule is still named "dltensor" do we own it and must run the deleter.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, USED_DLTENSOR.as_ptr()) == 1 {
            return;
        }
        let ptr = ffi::PyCapsule_GetPointer(capsule, DLTENSOR.as_ptr())
            as *mut onnx_runtime_dlpack::DLManagedTensor;
        if ptr.is_null() {
            // GetPointer set an exception; a destructor cannot propagate it.
            ffi::PyErr_WriteUnraisable(capsule);
            return;
        }
        onnx_runtime_dlpack::release(ptr);
    }
}

/// PyCapsule destructor for an **unconsumed** `"dltensor_versioned"` capsule.
///
/// # Safety
///
/// Installed only on versioned capsules created by [`NxrtValue::__dlpack__`],
/// whose stored pointer is a live `*mut DLManagedTensorVersioned`. Same
/// name-check handshake as [`dlpack_capsule_destructor`].
unsafe extern "C" fn dlpack_versioned_capsule_destructor(capsule: *mut ffi::PyObject) {
    // SAFETY: see `dlpack_capsule_destructor`; this is the versioned analogue.
    unsafe {
        if ffi::PyCapsule_IsValid(capsule, USED_DLTENSOR_VERSIONED.as_ptr()) == 1 {
            return;
        }
        let ptr = ffi::PyCapsule_GetPointer(capsule, DLTENSOR_VERSIONED.as_ptr())
            as *mut onnx_runtime_dlpack::DLManagedTensorVersioned;
        if ptr.is_null() {
            ffi::PyErr_WriteUnraisable(capsule);
            return;
        }
        onnx_runtime_dlpack::release_versioned(ptr);
    }
}

/// A zero-copy-capable nxrt output value.
///
/// Returned by [`InferenceSession::run_with_values`](crate::InferenceSession).
/// Implements the DLPack producer protocol (`__dlpack__` /
/// `__dlpack_device__`) so `torch.from_dlpack(v)` / `np.from_dlpack(v)` borrow
/// nxrt's buffer without copying, and keeps a `.numpy()` method for the
/// copy-based path.
///
/// The wrapped tensor is held behind an `Arc`: every `__dlpack__` export moves a
/// clone of that `Arc` into the `DLManagedTensor`'s owner, so the backing buffer
/// outlives this Python object for as long as any consumer holds the exported
/// array (the lifetime-safety guarantee this whole feature turns on).
#[pyclass(module = "nxrt", name = "NxrtValue")]
pub struct NxrtValue {
    tensor: Arc<Tensor>,
    name: String,
}

impl NxrtValue {
    /// Wrap an owned output tensor (moved in from `InferenceSession::run`).
    pub fn new(tensor: Tensor, name: String) -> Self {
        Self { tensor: Arc::new(tensor), name }
    }

    /// Borrow the wrapped tensor (for `.numpy()` reuse in the parent module).
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// The tensor's name (model output name).
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[pymethods]
impl NxrtValue {
    /// `(device_type, device_id)` for this value, per the DLPack protocol.
    ///
    /// Returns the `kDLCPU`/`kDLCUDA` integer and ordinal a consumer uses to
    /// decide how to import (and, for CUDA, which stream to pass to
    /// `__dlpack__`).
    fn __dlpack_device__(&self, py: Python<'_>) -> PyResult<Py<PyTuple>> {
        let dev = to_dldevice(&self.tensor)?;
        Ok(PyTuple::new(py, [dev.device_type as i64, dev.device_id as i64])?.unbind())
    }

    /// Export this value as a DLPack `"dltensor"` PyCapsule (zero-copy).
    ///
    /// Follows the Array-API producer contract. `stream`/`max_version`/
    /// `dl_device`/`copy` are accepted for signature compatibility:
    ///
    /// * `copy=True` is refused — nxrt exports a *borrow*, never a copy, so a
    ///   caller demanding a fresh copy must use `.numpy()` instead.
    /// * `dl_device` may only request this value's own device (no cross-device
    ///   move on export).
    /// * `stream` is only meaningful for CUDA, which this build does not export.
    /// * `max_version` selects the wire form: when the consumer advertises
    ///   DLPack major ≥ 1 (e.g. numpy ≥ 2.1, recent torch) nxrt emits the
    ///   **versioned** `DLManagedTensorVersioned` (`"dltensor_versioned"`) with
    ///   the writable flag, so the borrowed array is mutable in place; otherwise
    ///   it falls back to the universally-consumed unversioned `DLManagedTensor`
    ///   (`"dltensor"`), which some importers surface as read-only.
    ///
    /// The returned view aliases this tensor's storage: consumer writes are
    /// visible in place by design. A DLPack read-only flag could be offered as a
    /// future opt-in, but this export remains writable.
    #[pyo3(signature = (stream=None, max_version=None, dl_device=None, copy=None))]
    fn __dlpack__(
        &self,
        py: Python<'_>,
        stream: Option<Py<PyAny>>,
        max_version: Option<Py<PyAny>>,
        dl_device: Option<Py<PyAny>>,
        copy: Option<bool>,
    ) -> PyResult<Py<PyAny>> {
        let _ = stream;

        if copy == Some(true) {
            return Err(PyBufferError::new_err(
                "__dlpack__(copy=True) is not supported: nxrt exports a \
                 zero-copy borrow of its output buffer. Call `.numpy()` for an \
                 owned copy instead.",
            ));
        }

        // Resolve device/dtype up front so an unsupported case errors before we
        // allocate any C-side state.
        let device = to_dldevice(&self.tensor)?;
        let dtype = to_dldatatype(self.tensor.dtype)?;

        if let Some(dl_device) = dl_device {
            let requested: (i64, i64) = dl_device.extract(py).map_err(|_| {
                PyTypeError::new_err(
                    "__dlpack__(dl_device=...) must be a (device_type, device_id) \
                     tuple of ints",
                )
            })?;
            if requested != (device.device_type as i64, device.device_id as i64) {
                return Err(PyBufferError::new_err(format!(
                    "__dlpack__ cannot move this value to device {requested:?} on \
                     export; it lives on {:?}. Import it on its own device, or \
                     use `.numpy()`.",
                    (device.device_type, device.device_id)
                )));
            }
        }

        // Prefer the versioned protocol when the consumer supports DLPack major
        // ≥ 1 (`max_version` is a `(major, minor)` tuple). It is the only form
        // that carries the writable flag, so an in-place mutable borrow needs
        // it; a `None`/older `max_version` falls back to the unversioned form.
        let use_versioned = match &max_version {
            None => false,
            Some(mv) => mv
                .extract::<(i64, i64)>(py)
                .map(|(major, _)| major >= 1)
                .unwrap_or(false),
        };

        // Base pointer of the backing allocation. The `Arc<Tensor>` clone moved
        // into the exporter below keeps this address valid for the capsule's
        // life, independent of this `NxrtValue`.
        let keep_alive = self.tensor.clone();
        let data = if keep_alive.numel() == 0 {
            std::ptr::null_mut()
        } else {
            keep_alive.as_bytes().as_ptr() as *mut c_void
        };
        let shape: Vec<i64> = keep_alive.shape.iter().map(|&d| d as i64).collect();

        if use_versioned {
            // Row-major contiguous → empty strides → null strides. `read_only =
            // false`: nxrt owns a writable host buffer and hands out a genuine
            // mutable borrow.
            // SAFETY: the Arc moved into `keep_alive` owns `data`'s allocation
            // through the managed tensor's deleter; shape/strides describe it.
            let managed = unsafe {
                onnx_runtime_dlpack::export_versioned(
                    Box::new(keep_alive),
                    data,
                    device,
                    dtype,
                    shape,
                    Vec::new(),
                    0,
                    false,
                )
            };
            // SAFETY: `managed` is a live versioned export pointer; the name is
            // NUL-terminated; the destructor matches the capsule ABI and only
            // runs on capsules of this exact shape. On creation failure Python
            // never took ownership, so we free `managed` ourselves.
            let capsule = unsafe {
                let raw = ffi::PyCapsule_New(
                    managed as *mut c_void,
                    DLTENSOR_VERSIONED.as_ptr(),
                    Some(dlpack_versioned_capsule_destructor),
                );
                if raw.is_null() {
                    onnx_runtime_dlpack::release_versioned(managed);
                    return Err(PyErr::fetch(py));
                }
                Py::<PyAny>::from_owned_ptr(py, raw)
            };
            return Ok(capsule);
        }

        // Unversioned fallback: wrap the raw `DLManagedTensor*` in a `"dltensor"`
        // PyCapsule. The capsule's stored pointer IS the `DLManagedTensor*` (not
        // a pointer to it), as consumers require.
        // SAFETY: the Arc moved into `keep_alive` owns `data`'s allocation
        // through the managed tensor's deleter; shape/strides describe it.
        let managed = unsafe {
            onnx_runtime_dlpack::export(
                Box::new(keep_alive),
                data,
                device,
                dtype,
                shape,
                Vec::new(),
                0,
            )
        };

        // SAFETY: `managed` is a live export pointer we just created; `DLTENSOR`
        // is a valid NUL-terminated name; `dlpack_capsule_destructor` matches
        // PyCapsule's destructor ABI and only ever runs on capsules of this
        // exact shape. On capsule-creation failure we must free `managed`
        // ourselves (Python never took ownership).
        let capsule = unsafe {
            let raw = ffi::PyCapsule_New(
                managed as *mut c_void,
                DLTENSOR.as_ptr(),
                Some(dlpack_capsule_destructor),
            );
            if raw.is_null() {
                onnx_runtime_dlpack::release(managed);
                return Err(PyErr::fetch(py));
            }
            Py::<PyAny>::from_owned_ptr(py, raw)
        };
        Ok(capsule)
    }

    /// Copy this value into a numpy array (the pre-DLPack behaviour).
    ///
    /// Delegates to the parent module's `tensor_to_numpy`, so `.numpy()` matches
    /// exactly what `run()` used to return for this output.
    fn numpy(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let np = py.import("numpy")?;
        crate::tensor_to_numpy(py, &np, &self.name, &self.tensor)
    }

    fn __repr__(&self) -> String {
        format!(
            "NxrtValue(name={:?}, dtype={:?}, shape={:?})",
            self.name, self.tensor.dtype, self.tensor.shape
        )
    }
}

/// Compute the row-major contiguous element strides for `shape`, and test the
/// producer's `strides` against them (dimensions of size ≤ 1 are wildcards, as
/// their stride is unobservable). Returns `true` when the layout is C-contiguous
/// — the only layout nxrt can borrow without a copy.
fn is_row_major_contiguous(shape: &[i64], strides: Option<&[i64]>) -> bool {
    let strides = match strides {
        // DLPack convention: null strides == C-contiguous.
        None => return true,
        Some(s) => s,
    };
    if strides.len() != shape.len() {
        return false;
    }
    let mut expected: i64 = 1;
    for i in (0..shape.len()).rev() {
        let dim = shape[i];
        if dim > 1 && strides[i] != expected {
            return false;
        }
        expected = expected.saturating_mul(dim.max(1));
    }
    true
}

/// Outcome of inspecting a borrowed DLPack view: either borrow it zero-copy
/// (`Commit`) or hand back to the caller's copy path (`Fallback`).
enum ImportPlan {
    Fallback,
    Commit {
        dtype: DataType,
        shape: Vec<usize>,
        /// First-element pointer (`data + byte_offset`).
        data: *mut c_void,
        /// Storage byte length for the whole tensor.
        len: usize,
        /// Alignment (power of two) to record on the borrowed buffer.
        align: usize,
    },
}

/// Validate a borrowed DLPack view for a zero-copy CPU import.
///
/// Returns [`ImportPlan::Fallback`] (defer to the copy path) for non-contiguous
/// or empty tensors, an `Err` for an unsupported dtype, and
/// [`ImportPlan::Commit`] otherwise. Performs **no** ownership transfer — the
/// caller commits (renames the capsule, builds the guard) only for `Commit`.
fn plan_import(view: &onnx_runtime_dlpack::BorrowedDlpack<'_>) -> PyResult<ImportPlan> {
    // Device is re-checked here defensively; the caller already gated on
    // `__dlpack_device__`, but the capsule's own device is authoritative.
    if view.device.device_type != onnx_runtime_dlpack::DL_CPU {
        return Ok(ImportPlan::Fallback);
    }
    // Unsupported dtype → defer to the copy path, which raises a `TypeError`
    // naming the *numpy* dtype (more actionable to users than a DLPack
    // code/bits pair, and it keeps a single error surface for bad dtypes).
    let dtype = match from_dldatatype(view.dtype) {
        Ok(d) => d,
        Err(_) => return Ok(ImportPlan::Fallback),
    };

    // Non-contiguous → copy fallback (arbitrary strides unsupported this pass).
    if !is_row_major_contiguous(view.shape, view.strides) {
        return Ok(ImportPlan::Fallback);
    }

    // Reject negative/degenerate dims defensively.
    let mut shape: Vec<usize> = Vec::with_capacity(view.shape.len());
    for &d in view.shape {
        if d < 0 {
            return Ok(ImportPlan::Fallback);
        }
        shape.push(d as usize);
    }
    // The DLPack shape is FOREIGN/untrusted, so the element count and its
    // byte-length must both be computed with checked arithmetic — a crafted
    // dim product that wraps `usize` would otherwise under-size the borrow and
    // license an out-of-bounds alias. On overflow, fall back to the copy path
    // (which re-materialises safely) rather than panicking.
    let mut numel: usize = 1;
    for &d in &shape {
        match numel.checked_mul(d) {
            Some(n) => numel = n,
            None => return Ok(ImportPlan::Fallback),
        }
    }
    // Empty tensors have a possibly-null data pointer; `from_borrowed_parts`
    // requires non-null, so borrow only non-empty buffers and copy the rest.
    if numel == 0 || view.data.is_null() {
        return Ok(ImportPlan::Fallback);
    }

    let len = match dtype.checked_storage_bytes(numel) {
        Some(l) => l,
        // Element count fits `usize` but count×byte_size wraps → copy fallback.
        None => return Ok(ImportPlan::Fallback),
    };
    // Fold the byte offset into the base pointer so nxrt sees element origin 0.
    // SAFETY: the producer guarantees `data + byte_offset` is the first element
    // of a `len`-byte allocation; we only compute the address (no deref).
    let data = unsafe { (view.data as *mut u8).add(view.byte_offset as usize) as *mut c_void };

    // A truthful, power-of-two alignment we can guarantee: the element size
    // (1/2/4/8 for every supported dtype). Underclaiming is always sound.
    let align = dtype.byte_size().max(1);

    // `DeviceBuffer::from_borrowed_parts` documents an alignment precondition
    // (`data` aligned to at least `align`). numpy ≥ 2.5 can hand out a
    // contiguous but UNALIGNED buffer; borrowing it would violate that contract
    // and any aligned load downstream. If the first-element address is not
    // `align`-aligned, fall back to the copy path (which re-materialises into a
    // freshly-aligned allocation).
    if !(data as usize).is_multiple_of(align) {
        return Ok(ImportPlan::Fallback);
    }

    Ok(ImportPlan::Commit { dtype, shape, data, len, align })
}

/// Try to import `obj` as a zero-copy nxrt [`Tensor`] via the DLPack **consumer**
/// protocol. Returns `Ok(None)` when `obj` cannot (or should not) be borrowed —
/// no `__dlpack__`, a producer that refuses to export, a non-CPU device, an
/// unsupported dtype, empty, or non-contiguous — so the caller falls back to the
/// numpy copy path (which raises its own actionable error for bad dtypes).
///
/// On success the returned tensor **borrows** `obj`'s buffer: it owns the
/// DLPack `deleter` (via an `import_guard`) and frees the foreign allocation
/// exactly once on drop. Mutations to the source are visible in place, and vice
/// versa — that is the whole point of the zero-copy path.
pub(crate) fn tensor_from_dlpack(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
) -> PyResult<Option<Tensor>> {
    // Producers must expose both halves of the protocol.
    if !obj.hasattr("__dlpack__")? || !obj.hasattr("__dlpack_device__")? {
        return Ok(None);
    }

    // Device gate: only CPU is borrowable this pass; anything else copies.
    let (device_type, _device_id): (i64, i64) =
        obj.call_method0("__dlpack_device__")?.extract().map_err(|_| {
            PyTypeError::new_err(
                "__dlpack_device__() must return a (device_type, device_id) tuple of ints",
            )
        })?;
    if device_type != onnx_runtime_dlpack::DL_CPU as i64 {
        // CUDA/other-device import is out of scope; copy instead of borrowing
        // (never silently produce a host tensor aliasing device memory).
        return Ok(None);
    }

    // Request the capsule, preferring the versioned protocol. Older producers
    // reject the `max_version` kwarg, so retry without it. If the producer
    // *refuses* to export DLPack at all (e.g. numpy's `__dlpack__` raises
    // `BufferError` for a dtype it cannot encode, such as bfloat16), fall back
    // to the copy path rather than propagating — the copy path may still handle
    // the value (it moves bytes directly).
    let capsule: Bound<'_, PyAny> = {
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("max_version", (1i64, 0i64))?;
        match obj.call_method("__dlpack__", (), Some(&kwargs)) {
            Ok(c) => c,
            Err(_) => match obj.call_method0("__dlpack__") {
                Ok(c) => c,
                Err(_) => return Ok(None),
            },
        }
    };
    let cap_ptr = capsule.as_ptr();

    // SAFETY: `cap_ptr` is a live PyObject for the duration of `capsule`. All
    // capsule calls below are name-checked; we never take ownership of the
    // managed pointer's deleter until the final `PyCapsule_SetName` rename, so
    // on every early return the capsule's own destructor frees the tensor (no
    // leak, no double free).
    unsafe {
        let versioned = ffi::PyCapsule_IsValid(cap_ptr, DLTENSOR_VERSIONED.as_ptr()) == 1;
        let unversioned =
            !versioned && ffi::PyCapsule_IsValid(cap_ptr, DLTENSOR.as_ptr()) == 1;
        if !versioned && !unversioned {
            // Not a DLPack tensor capsule we understand — defer to copy.
            return Ok(None);
        }

        if versioned {
            let managed = ffi::PyCapsule_GetPointer(cap_ptr, DLTENSOR_VERSIONED.as_ptr())
                as *mut onnx_runtime_dlpack::DLManagedTensorVersioned;
            if managed.is_null() {
                return Err(PyErr::fetch(py));
            }
            let view = onnx_runtime_dlpack::borrowed_view_versioned(managed);
            match plan_import(&view)? {
                ImportPlan::Fallback => Ok(None),
                ImportPlan::Commit { dtype, shape, data, len, align } => {
                    // Commit point: rename so the producer's destructor won't
                    // also free, then become the sole owner of the deleter.
                    if ffi::PyCapsule_SetName(cap_ptr, USED_DLTENSOR_VERSIONED.as_ptr()) != 0 {
                        return Err(PyErr::fetch(py));
                    }
                    let guard: Box<dyn std::any::Any + Send + Sync> =
                        Box::new(GilDlpackVersionedGuard {
                            owner: onnx_runtime_dlpack::ManagedTensorVersionedOwner::new(managed),
                        });
                    let buffer =
                        DeviceBuffer::from_borrowed_parts(data, DeviceId::cpu(), len, align);
                    let tensor = Tensor::from_borrowed_parts_with_guard(
                        cpu_allocator(),
                        dtype,
                        shape,
                        TensorLayout::contiguous(),
                        buffer,
                        guard,
                    );
                    Ok(Some(tensor))
                }
            }
        } else {
            let managed = ffi::PyCapsule_GetPointer(cap_ptr, DLTENSOR.as_ptr())
                as *mut onnx_runtime_dlpack::DLManagedTensor;
            if managed.is_null() {
                return Err(PyErr::fetch(py));
            }
            let view = onnx_runtime_dlpack::borrowed_view(managed);
            match plan_import(&view)? {
                ImportPlan::Fallback => Ok(None),
                ImportPlan::Commit { dtype, shape, data, len, align } => {
                    if ffi::PyCapsule_SetName(cap_ptr, USED_DLTENSOR.as_ptr()) != 0 {
                        return Err(PyErr::fetch(py));
                    }
                    let guard: Box<dyn std::any::Any + Send + Sync> =
                        Box::new(GilDlpackGuard {
                            owner: onnx_runtime_dlpack::ManagedTensorOwner::new(managed),
                        });
                    let buffer =
                        DeviceBuffer::from_borrowed_parts(data, DeviceId::cpu(), len, align);
                    let tensor = Tensor::from_borrowed_parts_with_guard(
                        cpu_allocator(),
                        dtype,
                        shape,
                        TensorLayout::contiguous(),
                        buffer,
                        guard,
                    );
                    Ok(Some(tensor))
                }
            }
        }
    }
}

/// Register `NxrtValue` on the module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<NxrtValue>()?;
    Ok(())
}

/// GIL-acquiring guard stored in an imported [`Tensor`]'s `import_guard`.
///
/// The foreign DLPack `deleter` (numpy/torch) calls into the CPython C-API
/// (`Py_DECREF`, buffer release, …) and therefore **requires the GIL**. The
/// underlying [`ManagedTensorOwner`](onnx_runtime_dlpack::ManagedTensorOwner) is
/// `Send + Sync` and the `Tensor` that owns it is likewise `Send + Sync`, so
/// nothing stops a caller from stashing an imported `Tensor` and dropping it on
/// a background thread that does *not* hold the GIL. This guard makes that case
/// sound **by construction**: its `Drop` calls `Python::with_gil`, which
/// attaches the current thread to the interpreter and acquires the GIL (a no-op
/// re-borrow when the thread already holds it), and only then drives the foreign
/// deleter via `call_deleter`. Because `call_deleter` is idempotent and the
/// owner's own `Drop` becomes a no-op afterwards, the deleter still runs exactly
/// once — now guaranteed under the GIL regardless of which thread drops the
/// tensor.
struct GilDlpackGuard {
    owner: onnx_runtime_dlpack::ManagedTensorOwner,
}

impl Drop for GilDlpackGuard {
    fn drop(&mut self) {
        Python::with_gil(|_py| {
            // SAFETY: `with_gil` guarantees this thread holds the GIL for the
            // duration of the closure, satisfying the foreign deleter's CPython
            // C-API requirement. `call_deleter` runs it exactly once.
            unsafe { self.owner.call_deleter() };
        });
    }
}

/// Versioned analogue of [`GilDlpackGuard`].
struct GilDlpackVersionedGuard {
    owner: onnx_runtime_dlpack::ManagedTensorVersionedOwner,
}

impl Drop for GilDlpackVersionedGuard {
    fn drop(&mut self) {
        Python::with_gil(|_py| {
            // SAFETY: as `GilDlpackGuard::drop`; the GIL is held here.
            unsafe { self.owner.call_deleter() };
        });
    }
}
