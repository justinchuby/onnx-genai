//! # `nxrt` — Python binding for the nxrt ONNX runtime
//!
//! A PyO3 extension module that exposes the pure-Rust
//! [`onnx_runtime_session::InferenceSession`] to Python under the import name
//! `nxrt`, mirroring the shape of `onnxruntime.InferenceSession` so existing
//! conformance suites (e.g. `cbourjau/onnx-tests`) can drive nxrt with a
//! one-line runtime swap.
//!
//! ## Public Python surface
//!
//! ```python
//! import nxrt
//! sess = nxrt.InferenceSession("model.onnx")            # path or raw bytes
//! outs = sess.run(None, {"x": np_array})                # onnxruntime-compatible
//! [i.name for i in sess.get_inputs()]                   # NodeArg-ish metadata
//! nxrt.get_available_providers()                        # -> ["CPUExecutionProvider", ...]
//! nxrt.__version__
//! ```
//!
//! ## Error-quality contract (`RULES.md` §1)
//!
//! Every failure that crosses into Python carries **what failed, why, and how to
//! fix it**: dtype/shape mismatches name the offending input and show
//! expected-vs-got, an unknown provider lists the available ones, and an
//! unsupported operator surfaces the runtime's actionable message rather than a
//! bare `RuntimeError`. Errors are mapped to the most specific Python exception
//! type (`TypeError`/`ValueError`/`KeyError`/`FileNotFoundError`) so callers can
//! branch on them.
//!
//! ## `unsafe`
//!
//! Numpy ⇆ raw-bytes conversion goes through numpy's own
//! `ascontiguousarray`/`tobytes`/`frombuffer` (the buffer protocol), so the
//! input/output *copy* paths contain no hand-rolled pointer arithmetic. The
//! zero-copy DLPack **export** path ([`dlpack`]) does use `unsafe`, but only a
//! thin `pyo3::ffi` PyCapsule shim: the `DLManagedTensor` ABI and its
//! memory-owning `deleter` live in the dependency-free `onnx-runtime-dlpack`
//! crate, so the surface exposed here is limited to creating the `"dltensor"`
//! capsule and its name-checking destructor. bf16/f16 are moved as opaque
//! 2-byte little-endian storage — never reinterpreted through f32.

mod dlpack;

use std::sync::Mutex;

use onnx_runtime_ir::{DataType, Dim, Shape};
use onnx_runtime_session::{InferenceSession as RtSession, IoMeta, Tensor};
use pyo3::exceptions::{
    PyFileNotFoundError, PyKeyError, PyRuntimeError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString, PyTuple};

/// The `nxrt` distribution version, kept in lockstep with the crate version.
const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Execution providers this build can actually service, most-preferred first.
///
/// The CPU provider is always present; `CUDAExecutionProvider` appears only when
/// the crate is compiled with the `cuda` feature (so the list never advertises a
/// provider the wheel cannot honor — `RULES.md` §2/§5).
fn available_providers() -> Vec<&'static str> {
    #[cfg(feature = "cuda")]
    let v = vec!["CUDAExecutionProvider", "CPUExecutionProvider"];
    #[cfg(not(feature = "cuda"))]
    let v = vec!["CPUExecutionProvider"];
    v
}

/// Map a requested provider string to a concrete, buildable EP, or return an
/// actionable error naming what is available.
fn validate_provider(name: &str) -> PyResult<()> {
    let available = available_providers();
    if available.contains(&name) {
        return Ok(());
    }
    // Distinguish "real ORT provider we simply don't build" from an outright
    // typo, so the message can nudge toward the right next step.
    let hint = if name == "CUDAExecutionProvider" {
        " (this wheel was built without CUDA support; install a CUDA-enabled \
         nxrt build or select \"CPUExecutionProvider\")"
    } else {
        ""
    };
    Err(PyValueError::new_err(format!(
        "unknown execution provider {name:?}{hint}. Available providers in this \
         build: {available:?}"
    )))
}

/// numpy dtype canonical `.name` → nxrt [`DataType`].
///
/// Uses numpy's canonical dtype *name* (e.g. `"float32"`, `"int64"`,
/// `"bfloat16"`) rather than fragile kind/itemsize probing, so the mapping is
/// exact and platform-independent. Returns `None` for any dtype nxrt cannot
/// represent; the caller turns that into an actionable `TypeError`.
fn numpy_name_to_dtype(name: &str) -> Option<DataType> {
    Some(match name {
        "bool" => DataType::Bool,
        "int8" => DataType::Int8,
        "int16" => DataType::Int16,
        "int32" => DataType::Int32,
        "int64" => DataType::Int64,
        "uint8" => DataType::Uint8,
        "uint16" => DataType::Uint16,
        "uint32" => DataType::Uint32,
        "uint64" => DataType::Uint64,
        "float16" => DataType::Float16,
        "float32" => DataType::Float32,
        "float64" => DataType::Float64,
        "bfloat16" => DataType::BFloat16,
        _ => return None,
    })
}

/// nxrt [`DataType`] → the numpy dtype string used to reconstruct an output
/// array. `bfloat16` requires the `ml_dtypes` package at runtime (numpy has no
/// native bf16); the caller imports it lazily and errors clearly if absent.
fn dtype_to_numpy_name(dtype: DataType) -> PyResult<&'static str> {
    Ok(match dtype {
        DataType::Bool => "bool",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Uint8 => "uint8",
        DataType::Uint16 => "uint16",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Float16 => "float16",
        DataType::Float32 => "float32",
        DataType::Float64 => "float64",
        DataType::BFloat16 => "bfloat16",
        // Types with no lossless numpy representation in this iteration. Fail
        // with a message that says exactly which output dtype is unsupported.
        other => {
            return Err(PyTypeError::new_err(format!(
                "output tensor has dtype {other:?}, which nxrt's Python binding \
                 cannot yet convert to a numpy array (supported: bool, \
                 int8/16/32/64, uint8/16/32/64, float16/32/64, bfloat16). This \
                 is a binding limitation, not a model error."
            )));
        }
    })
}

/// Human-readable ONNX type string for a [`DataType`], shaped like
/// onnxruntime's NodeArg `.type` (e.g. `"tensor(float)"`).
fn onnx_type_string(dtype: DataType) -> String {
    let inner = match dtype {
        DataType::Float32 => "float",
        DataType::Float64 => "double",
        DataType::Float16 => "float16",
        DataType::BFloat16 => "bfloat16",
        DataType::Int8 => "int8",
        DataType::Int16 => "int16",
        DataType::Int32 => "int32",
        DataType::Int64 => "int64",
        DataType::Uint8 => "uint8",
        DataType::Uint16 => "uint16",
        DataType::Uint32 => "uint32",
        DataType::Uint64 => "uint64",
        DataType::Bool => "bool",
        DataType::String => "string",
        DataType::Float8E4M3FN => "float8e4m3fn",
        DataType::Float8E4M3FNUZ => "float8e4m3fnuz",
        DataType::Float8E5M2 => "float8e5m2",
        DataType::Float8E5M2FNUZ => "float8e5m2fnuz",
        DataType::Uint4 => "uint4",
        DataType::Int4 => "int4",
        DataType::Float4E2M1 => "float4e2m1",
    };
    format!("tensor({inner})")
}

/// Convert an nxrt [`Shape`] into an onnxruntime-style shape list: static dims
/// become ints, symbolic dims become their name (or `None` when anonymous).
fn shape_to_py(py: Python<'_>, shape: &Shape) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for dim in shape {
        match dim {
            Dim::Static(n) => list.append(*n)?,
            Dim::Symbolic(sym) => {
                // No interned name is reachable from `Shape` alone, so expose a
                // stable synthetic label; onnxruntime likewise yields either a
                // string or None for dynamic axes.
                list.append(format!("sym_{}", sym.0))?;
            }
        }
    }
    Ok(list.unbind())
}

/// onnxruntime-style `NodeArg`: the `.name`, `.type`, and `.shape` of a model
/// input or output. Returned by [`InferenceSession::get_inputs`] /
/// [`InferenceSession::get_outputs`].
#[pyclass(module = "nxrt", name = "NodeArg", frozen)]
struct NodeArg {
    #[pyo3(get)]
    name: String,
    #[pyo3(get, name = "type")]
    type_str: String,
    shape: Shape,
}

#[pymethods]
impl NodeArg {
    /// The value's shape: a list mixing `int` (static dims) and `str`/`None`
    /// (symbolic dims), matching `onnxruntime`'s `NodeArg.shape`.
    #[getter]
    fn shape(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        shape_to_py(py, &self.shape)
    }

    fn __repr__(&self) -> String {
        format!("NodeArg(name={:?}, type={:?})", self.name, self.type_str)
    }
}

impl NodeArg {
    fn from_meta(meta: &IoMeta) -> Self {
        Self {
            name: meta.name.clone(),
            type_str: onnx_type_string(meta.dtype),
            shape: meta.shape.clone(),
        }
    }
}

/// A loaded model ready for inference — the Python-facing mirror of
/// `onnxruntime.InferenceSession`.
///
/// The wrapped [`RtSession`] needs `&mut` to `run`, and a free-threaded
/// interpreter can call methods concurrently, so it is guarded by a [`Mutex`];
/// this keeps the type `Sync` and the abi3t (no-GIL) wheel sound without holding
/// the GIL across inference.
#[pyclass(module = "nxrt", name = "InferenceSession")]
struct InferenceSession {
    inner: Mutex<RtSession>,
    inputs: Vec<IoMeta>,
    outputs: Vec<IoMeta>,
    providers: Vec<String>,
}

#[pymethods]
impl InferenceSession {
    /// Load a model from a filesystem path or raw ONNX model bytes.
    ///
    /// * `path_or_bytes` — `str`/`os.PathLike` path, or a `bytes` object holding
    ///   a serialized `ModelProto`.
    /// * `providers` — ordered execution-provider preference list; defaults to
    ///   `["CPUExecutionProvider"]`. Unknown or unbuilt providers raise
    ///   `ValueError` listing what this build supports.
    #[new]
    #[pyo3(signature = (path_or_bytes, providers=None))]
    fn new(
        py: Python<'_>,
        path_or_bytes: &Bound<'_, PyAny>,
        providers: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let providers = providers.unwrap_or_else(|| vec!["CPUExecutionProvider".to_string()]);
        if providers.is_empty() {
            return Err(PyValueError::new_err(
                "providers must be a non-empty list; pass \
                 [\"CPUExecutionProvider\"] or omit the argument for the default",
            ));
        }
        for p in &providers {
            validate_provider(p)?;
        }

        // Accept raw model bytes directly; otherwise treat the argument as a
        // path (covering str and os.PathLike via Python's str()).
        let session = if let Ok(pybytes) = path_or_bytes.downcast::<PyBytes>() {
            let bytes = pybytes.as_bytes();
            RtSession::load_bytes(bytes).map_err(|e| load_error(&format!("<{} bytes>", bytes.len()), e))?
        } else {
            let path: String = path_or_bytes
                .str()
                .map_err(|_| {
                    PyTypeError::new_err(
                        "InferenceSession(path_or_bytes): expected a filesystem \
                         path (str/os.PathLike) or model bytes",
                    )
                })?
                .to_string_lossy()
                .into_owned();
            if !std::path::Path::new(&path).exists() {
                return Err(PyFileNotFoundError::new_err(format!(
                    "model file not found: {path:?}. Pass a path to an existing \
                     .onnx file, or pass the serialized model as bytes."
                )));
            }
            RtSession::load(&path).map_err(|e| load_error(&path, e))?
        };

        let inputs = session.inputs().to_vec();
        let outputs = session.outputs().to_vec();
        let _ = py; // reserved for future GIL-scoped work
        Ok(Self {
            inner: Mutex::new(session),
            inputs,
            outputs,
            providers,
        })
    }

    /// Run inference. Mirrors `onnxruntime.InferenceSession.run`.
    ///
    /// * `output_names` — list of outputs to return, or `None` for all outputs
    ///   in graph order.
    /// * `input_feed` — `{name: numpy.ndarray}` mapping every model input.
    ///
    /// Returns a list of numpy arrays, one per requested output.
    #[pyo3(signature = (output_names, input_feed))]
    fn run(
        &self,
        py: Python<'_>,
        output_names: Option<Vec<String>>,
        input_feed: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyList>> {
        let np = py.import("numpy")?;
        let (names, tensors) = self.run_inner(py, &np, output_names, input_feed)?;
        let list = PyList::empty(py);
        for (name, tensor) in names.iter().zip(tensors.iter()) {
            let arr = tensor_to_numpy(py, &np, name, tensor)?;
            list.append(arr)?;
        }
        Ok(list.unbind())
    }

    /// Run inference and return zero-copy-capable [`NxrtValue`] wrappers instead
    /// of eagerly-copied numpy arrays.
    ///
    /// Same arguments and output selection as [`run`](Self::run), but each
    /// result is an `nxrt.NxrtValue` implementing the DLPack producer protocol
    /// (`__dlpack__` / `__dlpack_device__`), so `torch.from_dlpack(v)` /
    /// `np.from_dlpack(v)` borrow nxrt's output buffer with **no copy**. Each
    /// value also has a `.numpy()` method for the copy-based path.
    ///
    /// `run()` still returns plain numpy arrays; this is the opt-in zero-copy
    /// entry point, so existing onnxruntime-compatible callers are unaffected.
    #[pyo3(signature = (output_names, input_feed))]
    fn run_with_values(
        &self,
        py: Python<'_>,
        output_names: Option<Vec<String>>,
        input_feed: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyList>> {
        let np = py.import("numpy")?;
        let (names, tensors) = self.run_inner(py, &np, output_names, input_feed)?;
        let list = PyList::empty(py);
        for (name, tensor) in names.into_iter().zip(tensors) {
            list.append(Py::new(py, dlpack::NxrtValue::new(tensor, name))?)?;
        }
        Ok(list.unbind())
    }

    /// Model input metadata as onnxruntime-style `NodeArg`s.
    fn get_inputs(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        node_args(py, &self.inputs)
    }

    /// Model output metadata as onnxruntime-style `NodeArg`s.
    fn get_outputs(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        node_args(py, &self.outputs)
    }

    /// The execution providers this session was created with (as requested).
    fn get_providers(&self) -> Vec<String> {
        self.providers.clone()
    }
}

impl InferenceSession {
    /// Shared inference core for [`run`](Self::run) and
    /// [`run_with_values`](Self::run_with_values).
    ///
    /// Validates and builds owned input tensors from `input_feed`, runs the
    /// session, then selects/orders the requested outputs and returns them as
    /// **owned** tensors paired with their names. Ownership is what lets the
    /// zero-copy path move a tensor into an `NxrtValue`/`DLManagedTensor`; the
    /// numpy path just borrows them for the copy.
    fn run_inner(
        &self,
        py: Python<'_>,
        np: &Bound<'_, PyModule>,
        output_names: Option<Vec<String>>,
        input_feed: &Bound<'_, PyDict>,
    ) -> PyResult<(Vec<String>, Vec<Tensor>)> {
        // Build owned tensors from the feed, validating names + dtypes with
        // actionable messages before touching the runtime.
        let mut owned: Vec<(String, Tensor)> = Vec::with_capacity(input_feed.len());
        for (key, value) in input_feed.iter() {
            let name: String = key.extract().map_err(|_| {
                PyTypeError::new_err("input_feed keys must be strings (input names)")
            })?;
            if !self.inputs.iter().any(|m| m.name == name) {
                let known: Vec<&str> = self.inputs.iter().map(|m| m.name.as_str()).collect();
                return Err(PyValueError::new_err(format!(
                    "unknown input {name:?} in input_feed. Model inputs are: {known:?}"
                )));
            }
            let tensor = numpy_to_tensor(py, np, &name, &value)?;
            owned.push((name, tensor));
        }

        // Every model input must be fed (the executor needs them all).
        for meta in &self.inputs {
            if !owned.iter().any(|(n, _)| n == &meta.name) {
                let provided: Vec<&str> = owned.iter().map(|(n, _)| n.as_str()).collect();
                return Err(PyValueError::new_err(format!(
                    "missing required input {:?} of type {} in input_feed \
                     (provided: {provided:?})",
                    meta.name,
                    onnx_type_string(meta.dtype),
                )));
            }
        }

        let feed: Vec<(&str, &Tensor)> =
            owned.iter().map(|(n, t)| (n.as_str(), t)).collect();

        let results = {
            let mut guard = self
                .inner
                .lock()
                .map_err(|_| PyRuntimeError::new_err("nxrt session mutex poisoned"))?;
            guard.run(&feed).map_err(run_error)?
        };

        // Resolve the requested outputs to indices into `results` (all model
        // outputs, in graph order).
        let indices: Vec<usize> = match &output_names {
            None => (0..self.outputs.len()).collect(),
            Some(names) => {
                let mut idxs = Vec::with_capacity(names.len());
                for want in names {
                    let idx = self
                        .outputs
                        .iter()
                        .position(|m| &m.name == want)
                        .ok_or_else(|| {
                            let known: Vec<&str> =
                                self.outputs.iter().map(|m| m.name.as_str()).collect();
                            PyValueError::new_err(format!(
                                "requested output {want:?} is not a model output. \
                                 Model outputs are: {known:?}"
                            ))
                        })?;
                    idxs.push(idx);
                }
                idxs
            }
        };

        // Move the selected tensors out by index. `Option::take` keeps this a
        // move (not a copy) so the zero-copy path really owns the buffer; a
        // duplicate request would take an already-moved slot and errors clearly.
        let mut slots: Vec<Option<Tensor>> = results.into_iter().map(Some).collect();
        let mut names = Vec::with_capacity(indices.len());
        let mut tensors = Vec::with_capacity(indices.len());
        for i in indices {
            let tensor = slots[i].take().ok_or_else(|| {
                PyValueError::new_err(format!(
                    "output {:?} was requested more than once; request each output \
                     at most once",
                    self.outputs[i].name
                ))
            })?;
            names.push(self.outputs[i].name.clone());
            tensors.push(tensor);
        }
        Ok((names, tensors))
    }
}

fn node_args(py: Python<'_>, metas: &[IoMeta]) -> PyResult<Py<PyList>> {
    let list = PyList::empty(py);
    for meta in metas {
        list.append(Py::new(py, NodeArg::from_meta(meta))?)?;
    }
    Ok(list.unbind())
}

/// Convert a numpy array (any feed value) into an nxrt [`Tensor`], with
/// actionable errors for unsupported dtypes or unreadable buffers.
fn numpy_to_tensor(
    py: Python<'_>,
    np: &Bound<'_, PyModule>,
    input_name: &str,
    value: &Bound<'_, PyAny>,
) -> PyResult<Tensor> {
    // Force a C-contiguous array so `tobytes()` yields row-major little-endian
    // storage regardless of the caller's array flags (copies only if needed).
    let arr = np.call_method1("ascontiguousarray", (value,)).map_err(|_| {
        PyTypeError::new_err(format!(
            "input {input_name:?}: expected a numpy.ndarray, got a value numpy \
             could not interpret as an array"
        ))
    })?;

    let dtype_name: String = arr
        .getattr("dtype")?
        .getattr("name")?
        .extract()
        .map_err(|_| PyTypeError::new_err(format!("input {input_name:?}: unreadable dtype")))?;
    let dtype = numpy_name_to_dtype(&dtype_name).ok_or_else(|| {
        PyTypeError::new_err(format!(
            "input {input_name:?} has numpy dtype {dtype_name:?}, which nxrt does \
             not support. Supported dtypes: bool, int8/16/32/64, \
             uint8/16/32/64, float16, float32, float64, bfloat16. Cast the array \
             with `.astype(...)` to a supported dtype."
        ))
    })?;

    let shape: Vec<usize> = arr
        .getattr("shape")?
        .downcast::<PyTuple>()
        .map_err(|_| PyValueError::new_err(format!("input {input_name:?}: unreadable shape")))?
        .extract()?;

    let bytes_obj = arr.call_method0("tobytes")?;
    let bytes: &[u8] = bytes_obj.downcast::<PyBytes>()?.as_bytes();

    let _ = py;
    Tensor::from_raw(dtype, shape.clone(), bytes).map_err(|e| {
        PyValueError::new_err(format!(
            "input {input_name:?}: failed to build a {} tensor of shape {shape:?}: {e}",
            onnx_type_string(dtype)
        ))
    })
}

/// Convert an nxrt output [`Tensor`] back into a numpy array via
/// `numpy.frombuffer(...).reshape(...)`.
///
/// `name` is the model-output name, used only to make dtype errors actionable.
pub(crate) fn tensor_to_numpy(
    py: Python<'_>,
    np: &Bound<'_, PyModule>,
    name: &str,
    tensor: &Tensor,
) -> PyResult<Py<PyAny>> {
    let np_name = dtype_to_numpy_name(tensor.dtype)?;

    // Resolve the numpy dtype object; bfloat16 lives in the optional `ml_dtypes`
    // package, so import it lazily with a message that says how to get it.
    let dtype_obj = if np_name == "bfloat16" {
        let ml = py.import("ml_dtypes").map_err(|_| {
            PyRuntimeError::new_err(format!(
                "output {name:?} is bfloat16, which requires the `ml_dtypes` \
                 package to represent as a numpy array. Install it with `pip \
                 install ml_dtypes`."
            ))
        })?;
        ml.getattr("bfloat16")?.into_any()
    } else {
        np.getattr("dtype")?.call1((np_name,))?.into_any()
    };

    let py_bytes = PyBytes::new(py, tensor.as_bytes());
    // frombuffer over empty buffers is fine; reshape restores the logical shape.
    let flat = np.call_method1("frombuffer", (py_bytes, &dtype_obj))?;
    let shape = PyTuple::new(py, &tensor.shape)?;
    let reshaped = flat.call_method1("reshape", (shape,))?;
    // Own the data (frombuffer aliases the temporary bytes buffer).
    let owned = reshaped.call_method0("copy")?;
    Ok(owned.unbind())
}

/// Map a session load error to the most specific Python exception, preserving
/// the runtime's rich causal message (`RULES.md` §1).
fn load_error(source: &str, err: onnx_runtime_session::SessionError) -> PyErr {
    PyValueError::new_err(format!(
        "failed to load ONNX model from {source}: {err}"
    ))
}

/// Map a `run` error to the most specific Python exception type, keeping the
/// runtime's actionable message intact (unsupported op, dtype/shape mismatch,
/// missing input) rather than collapsing to a generic `RuntimeError`.
fn run_error(err: onnx_runtime_session::SessionError) -> PyErr {
    use onnx_runtime_session::SessionError as E;
    let msg = err.to_string();
    match err {
        E::InputNotFound { .. } => PyKeyError::new_err(msg),
        E::UnsupportedOp { .. } => PyValueError::new_err(format!(
            "{msg}. This operator is not implemented by nxrt's CPU execution \
             provider yet."
        )),
        E::DtypeMismatch { .. } | E::RankMismatch { .. } | E::ShapeMismatch { .. } => {
            PyValueError::new_err(msg)
        }
        _ => PyRuntimeError::new_err(msg),
    }
}

/// Module-level `nxrt.get_available_providers()` — mirrors
/// `onnxruntime.get_available_providers()`.
#[pyfunction]
fn get_available_providers() -> Vec<String> {
    available_providers().iter().map(|s| s.to_string()).collect()
}

/// Return whether CUDA 13 CUPTI can be loaded for GPU tracing.
///
/// Absence or version mismatch is intentionally reported as `false`, not an
/// import error, so CUDA wheels remain importable on driverless machines.
#[cfg(feature = "cuda")]
#[pyfunction]
fn cupti_available() -> bool {
    onnx_runtime_tracer::cupti::CuptiProfiler::new().is_ok_and(|profiler| profiler.available())
}

/// Feed the CUPTI loader the interpreter's real search paths at import time.
///
/// The tracer dlopen's `libcupti` lazily and caches the discovery result, so we
/// must inject before any tracing begins. We hand it the live `sys.path` (which
/// reflects the real venv/user-site even when `VIRTUAL_ENV`/`PYTHONPATH` are
/// unset and `/proc/self/exe` is the base interpreter) plus this extension's own
/// directory and its parent, so pip-installed `nvidia-cuda-cupti-cu13` sitting
/// beside nxrt is found with zero setup. The tracer probes each root for the
/// pip layout `<root>/nvidia/cuda_cupti/lib/libcupti.so*`.
///
/// Best-effort: any lookup failure is ignored (env hints remain as fallback).
#[cfg(feature = "cuda")]
fn inject_cupti_search_paths(m: &Bound<'_, PyModule>) {
    use std::path::PathBuf;

    let mut paths: Vec<PathBuf> = Vec::new();

    // The loaded extension module's own directory (…/site-packages/nxrt/…) and
    // its parent (the site-packages root where `nvidia/` is a sibling).
    if let Ok(file) = m.getattr("__file__")
        && let Ok(file) = file.extract::<String>()
    {
        let path = PathBuf::from(file);
        if let Some(dir) = path.parent() {
            paths.push(dir.to_path_buf());
            if let Some(parent) = dir.parent() {
                paths.push(parent.to_path_buf());
            }
        }
    }

    if let Ok(sys) = m.py().import("sys")
        && let Ok(sys_path) = sys.getattr("path")
        && let Ok(iter) = sys_path.try_iter()
    {
        for entry in iter.flatten() {
            if let Ok(entry) = entry.extract::<String>()
                && !entry.is_empty()
            {
                paths.push(PathBuf::from(entry));
            }
        }
    }

    onnx_runtime_tracer::cupti::set_search_paths(paths);
}

/// The `nxrt` Python module.
#[pymodule]
fn nxrt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Inject the interpreter's real library search paths before anything can
    // trigger CUPTI discovery (its OnceLock caches on first use).
    #[cfg(feature = "cuda")]
    inject_cupti_search_paths(m);

    m.add("__version__", VERSION)?;
    let doc = "nxrt — Python binding for the nxrt ONNX runtime (onnxruntime-compatible InferenceSession).";
    m.add("__doc__", PyString::new(m.py(), doc))?;
    m.add_class::<InferenceSession>()?;
    m.add_class::<NodeArg>()?;
    dlpack::register(m)?;
    m.add_function(wrap_pyfunction!(get_available_providers, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(cupti_available, m)?)?;
    Ok(())
}
