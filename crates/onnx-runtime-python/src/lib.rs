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

use onnx_runtime_ir::{DataType, DeviceId, Dim, Shape};
use onnx_runtime_session::{InferenceSession as RtSession, IoMeta, Tensor};
use pyo3::exceptions::{
    PyAttributeError, PyFileNotFoundError, PyIndexError, PyKeyError, PyRuntimeError, PyTypeError,
    PyValueError,
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

/// Model-level metadata returned by [`InferenceSession::get_modelmeta`], shaped
/// like `onnxruntime.ModelMetadata`.
#[pyclass(module = "nxrt", name = "ModelMetadata", frozen)]
struct ModelMetadata {
    #[pyo3(get)]
    producer_name: String,
    #[pyo3(get)]
    graph_name: String,
    #[pyo3(get)]
    domain: String,
    #[pyo3(get)]
    description: String,
    #[pyo3(get)]
    version: i64,
    custom_metadata_map: Vec<(String, String)>,
}

#[pymethods]
impl ModelMetadata {
    #[getter]
    fn custom_metadata_map(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let metadata = PyDict::new(py);
        for (key, value) in &self.custom_metadata_map {
            metadata.set_item(key, value)?;
        }
        Ok(metadata.unbind())
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

    /// Model-level metadata, matching `onnxruntime.InferenceSession.get_modelmeta`.
    fn get_modelmeta(&self, _py: Python<'_>) -> PyResult<ModelMetadata> {
        let metadata = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("InferenceSession state is unavailable"))?
            .model_metadata()
            .clone();
        Ok(ModelMetadata {
            producer_name: metadata.producer_name,
            graph_name: metadata.graph_name,
            domain: metadata.domain,
            description: metadata.doc_string.unwrap_or_default(),
            version: metadata.model_version,
            custom_metadata_map: metadata.metadata_props,
        })
    }

    /// The execution providers this session was created with (as requested).
    fn get_providers(&self) -> Vec<String> {
        self.providers.clone()
    }

    /// Model input names in graph order (convenience over `get_inputs()`).
    #[getter]
    fn input_names(&self) -> Vec<String> {
        self.inputs.iter().map(|m| m.name.clone()).collect()
    }

    /// Model output names in graph order (convenience over `get_outputs()`).
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.outputs.iter().map(|m| m.name.clone()).collect()
    }

    /// Call the session like a function — the ergonomic alternative to the
    /// onnxruntime-shaped `run(None, {...})`.
    ///
    /// Inputs are resolved into a `{name: array}` feed, then handed to the same
    /// zero-copy core as [`run_with_values`](Self::run_with_values):
    ///
    /// * a single positional `dict`/`Mapping` is used directly as the feed
    ///   (keyword arguments may add more inputs, but must not clash with a key
    ///   already present);
    /// * otherwise positional arguments map to model inputs **by order** and
    ///   keyword arguments fill or set the rest **by name**.
    ///
    /// Every value flows through the DLPack import (numpy/torch/cupy/jax) with a
    /// numpy fallback, so no wrapper type is needed. This base-session call
    /// always returns **all** model outputs in graph order; use
    /// [`bind_outputs`](Self::bind_outputs) to obtain a proxy that returns a
    /// subset. A single output is returned directly as an
    /// [`NxrtValue`](dlpack::NxrtValue); multiple outputs come back in an
    /// [`Outputs`] container.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        self.call_impl(py, args, kwargs, None)
    }

    /// Return a lightweight [`BoundSession`] proxy that selects which outputs
    /// its calls return, without mutating this session.
    ///
    /// ```python
    /// with sess.bind_outputs("logits") as s:
    ///     logits = s(x)             # only "logits" is returned
    /// # or without a `with`:
    /// s = sess.bind_outputs("logits")
    /// logits = s(x)
    /// ```
    ///
    /// The selected names are validated against the model outputs immediately,
    /// with an actionable error, and are fixed for the life of the returned
    /// proxy. Selection is currently a return-value convenience: inference still
    /// computes all graph outputs. Computation pruning may be added as a future
    /// optimization. The proxy owns its own output list, so it is thread-/async-safe:
    /// overlapping calls on distinct proxies (or the base session) never clobber
    /// each other. This session's own `run()`/`run_with_values()`/`__call__`
    /// are entirely unaffected.
    #[pyo3(signature = (*names))]
    fn bind_outputs(slf: Bound<'_, Self>, names: Vec<String>) -> PyResult<BoundSession> {
        {
            let s = slf.borrow();
            if names.is_empty() {
                return Err(PyValueError::new_err(
                    "bind_outputs: pass at least one output name to select",
                ));
            }
            for n in &names {
                if !s.outputs.iter().any(|m| &m.name == n) {
                    let known: Vec<&str> = s.outputs.iter().map(|m| m.name.as_str()).collect();
                    return Err(PyValueError::new_err(format!(
                        "bind_outputs: {n:?} is not a model output. Model outputs \
                         are: {known:?}"
                    )));
                }
            }
        }
        Ok(BoundSession {
            session: slf.unbind(),
            names,
        })
    }
}

/// FIX 3 (honest CPU-session boundary, INPUT side): a device-resident (e.g.
/// CUDA) input imported zero-copy would reach the CPU executor, which reads
/// host bytes and PANICS. Full CUDA session execution — device-resident inputs
/// actually consumed by a CUDA EP — is a **separate epic**; until it lands we
/// refuse a non-host input with an actionable message *before* execution.
///
/// Returns the error message when `dev` is not host-accessible, else `None`.
/// Pure (no Python/GPU) so it is unit-testable on the CPU dev machine.
fn non_host_input_error(name: &str, dev: DeviceId) -> Option<String> {
    if dev.is_host_accessible() {
        return None;
    }
    Some(format!(
        "input {name:?} is on {dev:?} but this session executes on CPU; running \
         device-resident (CUDA) tensors requires CUDA session execution, which \
         is not yet supported — pass a CPU tensor (e.g. tensor.cpu()) or build \
         with CUDA session support."
    ))
}

/// FIX 3 (honest boundary, host-READ side): host-reading a device-resident
/// (CUDA) value into numpy calls `as_bytes()`/`host_bytes()`, whose
/// host-accessibility assert PANICS (surfacing as a `PanicException`). Refuse
/// with an actionable message instead.
///
/// Returns the error message when `dev` is not host-accessible, else `None`.
fn non_host_read_error(name: &str, dev: DeviceId) -> Option<String> {
    if dev.is_host_accessible() {
        return None;
    }
    Some(format!(
        "output {name:?} is on {dev:?}; host-reading a device-resident (CUDA) \
         tensor into a numpy array is not supported — export it zero-copy via \
         __dlpack__ (e.g. torch.from_dlpack) or move it to CPU first."
    ))
}

impl InferenceSession {
    /// Core of the callable API shared by [`__call__`](Self::__call__) and the
    /// [`BoundSession`] proxy. `output_names` is threaded in explicitly as a
    /// per-call parameter (never read from shared session state), so concurrent
    /// callers with different selections cannot clobber each other.
    fn call_impl(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
        output_names: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let np = py.import("numpy")?;
        let feed = self.build_feed(py, args, kwargs)?;
        let (names, tensors) = self.run_inner(py, &np, output_names, &feed)?;

        if tensors.len() == 1 {
            let name = names.into_iter().next().expect("len checked");
            let tensor = tensors.into_iter().next().expect("len checked");
            return Ok(Py::new(py, dlpack::NxrtValue::new(tensor, name))?.into_any());
        }

        let mut values = Vec::with_capacity(tensors.len());
        for (name, tensor) in names.iter().cloned().zip(tensors) {
            values.push(Py::new(py, dlpack::NxrtValue::new(tensor, name))?);
        }
        Ok(Py::new(py, Outputs { names, values })?.into_any())
    }

    /// Resolve `__call__`'s positional/keyword arguments into a validated
    /// `{name: value}` feed dict (values are left as raw Python objects so
    /// `run_inner` can zero-copy import them).
    fn build_feed<'py>(
        &self,
        py: Python<'py>,
        args: &Bound<'py, PyTuple>,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let feed = PyDict::new(py);

        // Case 1: a single positional Mapping (dict, OrderedDict, …) is the feed
        // itself; keyword arguments may add inputs but must not clash.
        if args.len() == 1 {
            let first = args.get_borrowed_item(0)?;
            if is_mapping(&first) {
                let items = first.call_method0("items")?;
                for item in items.try_iter()? {
                    let pair = item?;
                    let key = pair.get_item(0)?;
                    let val = pair.get_item(1)?;
                    feed.set_item(key, val)?;
                }
                if let Some(kw) = kwargs {
                    for (key, val) in kw.iter() {
                        if feed.contains(&key)? {
                            let name: String =
                                key.extract().unwrap_or_else(|_| format!("{key:?}"));
                            return Err(PyValueError::new_err(format!(
                                "input {name:?} was supplied both in the feed \
                                 mapping and as a keyword argument; provide it \
                                 only once"
                            )));
                        }
                        feed.set_item(key, val)?;
                    }
                }
                return Ok(feed);
            }
        }

        // Case 2: positional arguments map to model inputs by order, keyword
        // arguments fill/override the rest by name.
        if args.len() > self.inputs.len() {
            let known: Vec<&str> = self.inputs.iter().map(|m| m.name.as_str()).collect();
            return Err(PyValueError::new_err(format!(
                "too many positional inputs: got {}, but the model has {} \
                 input(s): {known:?}. Pass at most {} positional array(s), or use \
                 keyword arguments by input name.",
                args.len(),
                self.inputs.len(),
                self.inputs.len(),
            )));
        }
        for (i, arg) in args.iter().enumerate() {
            feed.set_item(&self.inputs[i].name, arg)?;
        }
        if let Some(kw) = kwargs {
            for (key, val) in kw.iter() {
                let name: String = key.extract().map_err(|_| {
                    PyTypeError::new_err("keyword-argument input names must be strings")
                })?;
                if !self.inputs.iter().any(|m| m.name == name) {
                    let known: Vec<&str> = self.inputs.iter().map(|m| m.name.as_str()).collect();
                    return Err(PyValueError::new_err(format!(
                        "unknown input {name:?} passed as a keyword argument. Model \
                         inputs are: {known:?}"
                    )));
                }
                if feed.contains(&name)? {
                    return Err(PyValueError::new_err(format!(
                        "input {name:?} was supplied both positionally and as a \
                         keyword argument; provide it only once"
                    )));
                }
                feed.set_item(name, val)?;
            }
        }
        Ok(feed)
    }

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
        // Validate every explicit output request before converting inputs or
        // invoking the executor, so misspellings fail without an inference run.
        let indices = self.output_indices(output_names.as_deref())?;

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
            // Prefer a zero-copy DLPack borrow when the value exposes the
            // producer protocol (torch.Tensor, numpy ≥ 1.23, …) and lives on a
            // CPU, contiguous buffer; otherwise fall back to the numpy copy
            // path. This keeps `run()`/`run_with_values()` transparently
            // zero-copy for supported inputs without changing their contract.
            let tensor = match dlpack::tensor_from_dlpack(py, &value)? {
                Some(t) => t,
                None => numpy_to_tensor(py, np, &name, &value)?,
            };
            // FIX 3: honest CPU-session boundary — refuse device-resident inputs
            // before execution rather than panicking in the CPU executor. Full
            // CUDA session execution is a SEPARATE epic (see `non_host_input_error`).
            if let Some(msg) = non_host_input_error(&name, tensor.device()) {
                return Err(PyRuntimeError::new_err(msg));
            }
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

        // `indices` resolves requested names into `results`, which contains all
        // model outputs in graph order.
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

    /// Resolve an optional output selection to graph-output indices, rejecting
    /// unknown names before execution.
    fn output_indices(&self, output_names: Option<&[String]>) -> PyResult<Vec<usize>> {
        let indices = match output_names {
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
        Ok(indices)
    }
}

/// Whether `v` should be treated as a name→array mapping feed (dict,
/// OrderedDict, or any `collections.abc.Mapping`) rather than a single array
/// argument. numpy/torch arrays have no `keys`/`items`, so this never
/// misclassifies a real tensor.
fn is_mapping(v: &Bound<'_, PyAny>) -> bool {
    v.hasattr("keys").unwrap_or(false) && v.hasattr("items").unwrap_or(false)
}

/// Multi-output result of a callable [`InferenceSession`].
///
/// Ergonomic access to the selected outputs, in graph/selected order:
/// `out[0]`, `out["logits"]`, `out.logits`, `len(out)`, unpacking
/// (`a, b = sess(x)`), `.keys()`/`.values()`/`.items()`.
#[pyclass(module = "nxrt", name = "Outputs")]
struct Outputs {
    names: Vec<String>,
    values: Vec<Py<dlpack::NxrtValue>>,
}

#[pymethods]
impl Outputs {
    fn __len__(&self) -> usize {
        self.values.len()
    }

    /// Index by position (`out[0]`, negatives allowed) or by name (`out["y"]`).
    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<dlpack::NxrtValue>> {
        if let Ok(idx) = key.extract::<isize>() {
            let n = self.values.len() as isize;
            let i = if idx < 0 { idx + n } else { idx };
            if i < 0 || i >= n {
                return Err(PyIndexError::new_err(format!(
                    "output index {idx} out of range for {n} output(s)"
                )));
            }
            return Ok(self.values[i as usize].clone_ref(py));
        }
        if let Ok(name) = key.extract::<String>() {
            if let Some(pos) = self.names.iter().position(|n| n == &name) {
                return Ok(self.values[pos].clone_ref(py));
            }
            return Err(PyKeyError::new_err(format!(
                "no output named {name:?}; outputs are: {:?}",
                self.names
            )));
        }
        Err(PyTypeError::new_err(
            "Outputs indices must be an int (position) or str (output name)",
        ))
    }

    /// Attribute access by output name (`out.logits`).
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<dlpack::NxrtValue>> {
        if let Some(pos) = self.names.iter().position(|n| n == name) {
            return Ok(self.values[pos].clone_ref(py));
        }
        Err(PyAttributeError::new_err(format!(
            "'Outputs' object has no output named {name:?}; outputs are: {:?}",
            self.names
        )))
    }

    fn __contains__(&self, name: &str) -> bool {
        self.names.iter().any(|n| n == name)
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let list = PyList::empty(py);
        for v in &self.values {
            list.append(v.clone_ref(py))?;
        }
        Ok(list.try_iter()?.into_any().unbind())
    }

    /// Output names, in order.
    fn keys(&self) -> Vec<String> {
        self.names.clone()
    }

    /// Output values, in order.
    fn values(&self, py: Python<'_>) -> Vec<Py<dlpack::NxrtValue>> {
        self.values.iter().map(|v| v.clone_ref(py)).collect()
    }

    /// `(name, value)` pairs, in order.
    fn items(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let list = PyList::empty(py);
        for (name, value) in self.names.iter().zip(&self.values) {
            list.append((name, value.clone_ref(py)))?;
        }
        Ok(list.unbind())
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut parts = Vec::with_capacity(self.values.len());
        for (name, value) in self.names.iter().zip(&self.values) {
            let v = value.borrow(py);
            parts.push(format!(
                "{}: {} {:?}",
                name,
                dtype_display_name(v.tensor().dtype),
                v.tensor().shape,
            ));
        }
        Ok(format!("Outputs({})", parts.join(", ")))
    }
}

/// Lightweight proxy returned by [`InferenceSession::bind_outputs`] that applies
/// a fixed subset of returned outputs per call, without mutating the underlying
/// session.
///
/// The proxy holds a reference to its [`InferenceSession`] plus its own
/// immutable `names` list. Because the selection lives on the proxy (not on
/// shared session state), overlapping calls across threads or asyncio tasks —
/// on distinct proxies or on the base session — never clobber each other's
/// output selection. It is itself callable and exposes `run`/`run_with_values`
/// that default their output selection to this subset, and it doubles as a
/// context manager (`__enter__` yields the proxy, `__exit__` is a no-op) so
/// `with session.bind_outputs("y") as s: s(x)` keeps working.
///
/// This is a return-value filter only: inference still computes all graph
/// outputs. Output-subgraph pruning is a future optimization.
#[pyclass(module = "nxrt", name = "BoundSession")]
struct BoundSession {
    session: Py<InferenceSession>,
    names: Vec<String>,
}

#[pymethods]
impl BoundSession {
    /// Call the session with this proxy's fixed output subset.
    #[pyo3(signature = (*args, **kwargs))]
    fn __call__(
        &self,
        py: Python<'_>,
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        let sess = self.session.borrow(py);
        sess.call_impl(py, args, kwargs, Some(self.names.clone()))
    }

    /// Like [`InferenceSession::run`], but defaults the output selection to this
    /// proxy's subset when `output_names` is `None`. An explicit `output_names`
    /// still wins, matching the onnxruntime-shaped signature.
    #[pyo3(signature = (output_names, input_feed))]
    fn run(
        &self,
        py: Python<'_>,
        output_names: Option<Vec<String>>,
        input_feed: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyList>> {
        let sess = self.session.borrow(py);
        let names = output_names.or_else(|| Some(self.names.clone()));
        sess.run(py, names, input_feed)
    }

    /// Like [`InferenceSession::run_with_values`], but defaults the output
    /// selection to this proxy's subset when `output_names` is `None`.
    #[pyo3(signature = (output_names, input_feed))]
    fn run_with_values(
        &self,
        py: Python<'_>,
        output_names: Option<Vec<String>>,
        input_feed: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyList>> {
        let sess = self.session.borrow(py);
        let names = output_names.or_else(|| Some(self.names.clone()));
        sess.run_with_values(py, names, input_feed)
    }

    /// The output names this proxy selects, in order.
    #[getter]
    fn output_names(&self) -> Vec<String> {
        self.names.clone()
    }

    /// The underlying session (the escape hatch back to the full surface).
    #[getter]
    fn session(&self, py: Python<'_>) -> Py<InferenceSession> {
        self.session.clone_ref(py)
    }

    /// Context-manager entry: yields the proxy itself (no session mutation).
    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Context-manager exit: a no-op — there is no shared state to restore.
    #[pyo3(signature = (_exc_type=None, _exc_value=None, _traceback=None))]
    fn __exit__(
        &self,
        _exc_type: Option<Py<PyAny>>,
        _exc_value: Option<Py<PyAny>>,
        _traceback: Option<Py<PyAny>>,
    ) -> bool {
        false
    }

    fn __repr__(&self, py: Python<'_>) -> String {
        let _ = py;
        format!("BoundSession(outputs={:?})", self.names)
    }
}

/// Map a friendly `device` string to the execution-provider list the session
/// constructor understands. `cuda`/`cuda:N` fall back to CPU; `metal` maps to
/// the CoreML provider name (unavailable in the pure-Rust build, so it raises an
/// actionable error listing what is buildable).
fn device_to_providers(device: Option<&str>) -> PyResult<Vec<String>> {
    let dev = device.unwrap_or("cpu");
    let kind = dev.split(':').next().unwrap_or(dev).to_ascii_lowercase();
    Ok(match kind.as_str() {
        "cpu" => vec!["CPUExecutionProvider".to_string()],
        "cuda" | "gpu" => {
            // A `cuda:N` suffix must be a non-negative integer ordinal. The
            // ordinal has no effect yet (CUDA falls back to CPU in this build),
            // but malformed input like `cuda:abc` is rejected up front.
            if let Some((_, ordinal)) = dev.split_once(':')
                && (ordinal.is_empty() || !ordinal.bytes().all(|b| b.is_ascii_digit()))
            {
                return Err(PyValueError::new_err(format!(
                    "invalid CUDA device ordinal in {dev:?}: expected a \
                     non-negative integer after ':', e.g. \"cuda:0\"."
                )));
            }
            vec![
                "CUDAExecutionProvider".to_string(),
                "CPUExecutionProvider".to_string(),
            ]
        }
        "metal" | "coreml" => vec![
            "CoreMLExecutionProvider".to_string(),
            "CPUExecutionProvider".to_string(),
        ],
        _ => {
            return Err(PyValueError::new_err(format!(
                "unknown device {dev:?}. Supported devices: \"cpu\", \"cuda\" \
                 (or \"cuda:N\"), \"metal\". For full control pass \
                 providers=[...] instead."
            )));
        }
    })
}

/// `nxrt.load(path, *, device=None, providers=None)` — friendly loader that
/// returns a callable [`InferenceSession`].
///
/// `device` is sugar over provider selection (`"cpu"` default, `"cuda"`/
/// `"cuda:N"`, `"metal"`). If `providers` is given it wins and `device` is
/// ignored (the advanced escape hatch). This just calls through to the existing
/// `InferenceSession(path, providers=...)` constructor.
#[pyfunction]
#[pyo3(signature = (path, *, device=None, providers=None))]
fn load(
    py: Python<'_>,
    path: &Bound<'_, PyAny>,
    device: Option<String>,
    providers: Option<Vec<String>>,
) -> PyResult<InferenceSession> {
    let providers = match providers {
        Some(p) => p,
        None => device_to_providers(device.as_deref())?,
    };
    InferenceSession::new(py, path, Some(providers))
}

/// The numpy dtype *name* for a [`DataType`] (e.g. `"float32"`), falling back to
/// the `Debug` form for types with no numpy name. Used for human-facing `repr`s.
pub(crate) fn dtype_display_name(dtype: DataType) -> String {
    match dtype_to_numpy_name(dtype) {
        Ok(name) => name.to_string(),
        Err(_) => format!("{dtype:?}"),
    }
}

/// Resolve the numpy dtype *object* for an nxrt [`DataType`] (bfloat16 via the
/// optional `ml_dtypes` package). Shared by `NxrtValue.dtype`.
pub(crate) fn numpy_dtype_object(py: Python<'_>, dtype: DataType) -> PyResult<Py<PyAny>> {
    let np = py.import("numpy")?;
    let np_name = dtype_to_numpy_name(dtype)?;
    let arg = if np_name == "bfloat16" {
        let ml = py.import("ml_dtypes").map_err(|_| {
            PyRuntimeError::new_err(
                "dtype is bfloat16, which requires the `ml_dtypes` package to \
                 represent as a numpy dtype. Install it with `pip install \
                 ml_dtypes`.",
            )
        })?;
        ml.getattr("bfloat16")?.into_any()
    } else {
        PyString::new(py, np_name).into_any()
    };
    Ok(np.getattr("dtype")?.call1((arg,))?.unbind())
}

fn node_args(py: Python<'_>, metas: &[IoMeta]) -> PyResult<Py<PyList>> {    let list = PyList::empty(py);
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

    // FIX 3: honest host-read boundary. `tensor.as_bytes()` below asserts the
    // tensor is host-accessible and PANICS (surfacing as a PanicException) for a
    // CUDA-resident value. Surface a clean, actionable RuntimeError instead so
    // `.numpy()`/`run()` never panic on a device-resident output.
    if let Some(msg) = non_host_read_error(name, tensor.device()) {
        return Err(PyRuntimeError::new_err(msg));
    }

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

/// Test-only: import `obj` via the zero-copy DLPack consumer path and return the
/// borrowed tensor's base data pointer as an integer, or `None` if the value
/// fell back to the copy path (non-borrowable device, non-contiguous, empty,
/// unaligned, overflowing, unsupported dtype).
///
/// This is the only layer that can *prove* a zero-copy borrow: a Python test
/// compares this pointer against the source buffer's own address
/// (`ndarray.ctypes.data` / `torch tensor.data_ptr()`, including CUDA device
/// pointers); equality means no copy happened. Underscore-prefixed and
/// undocumented in the public API.
#[pyfunction]
fn _dlpack_import_data_ptr(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<Option<usize>> {
    match dlpack::tensor_from_dlpack(py, obj)? {
        Some(t) => Ok(Some(t.device_ptr() as usize)),
        None => Ok(None),
    }
}

/// Test-only: import `obj` zero-copy, then drop the resulting borrowed tensor on
/// a spawned OS thread that does **not** hold the GIL, exercising the
/// GIL-acquiring import guard. Returns `True` if a zero-copy borrow occurred
/// (guard exercised), `False` if the value fell back to the copy path.
///
/// If the guard did not reacquire the GIL before running the foreign deleter
/// (numpy/torch call `Py_DECREF`), this would deadlock or corrupt the
/// interpreter; returning cleanly with the producer's deleter flag incremented
/// proves the GIL-at-drop invariant holds off the Python thread.
#[pyfunction]
fn _dlpack_import_drop_on_thread(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    let tensor = match dlpack::tensor_from_dlpack(py, obj)? {
        Some(t) => t,
        None => return Ok(false),
    };
    // Release the GIL on this (Python) thread, then drop the imported tensor on
    // a fresh OS thread. The import guard's `Drop` must reacquire the GIL via
    // `Python::with_gil` to run the foreign deleter — that is what we prove.
    py.allow_threads(move || {
        std::thread::spawn(move || drop(tensor)).join().expect("drop thread panicked");
    });
    Ok(true)
}

/// Lazily create (and cache per device ordinal) a CUDA execution provider to
/// back a zero-copy DLPack-imported CUDA tensor.
///
/// A DLPack-imported tensor carries a **borrowed** buffer, so this EP never
/// frees the aliased device memory (`deallocate` is a no-op for borrowed
/// buffers). Its only role is to report `CUDA:ordinal` so the executor routes
/// the imported input to the GPU. The provider is cached process-wide per
/// ordinal so repeated imports on the same device reuse one context.
///
/// Only compiled with the `cuda` feature; the default (CPU-only) build never
/// reaches the CUDA import path (`plan_import` falls back to a copy there).
#[cfg(feature = "cuda")]
pub(crate) fn cuda_import_allocator(
    index: u32,
) -> PyResult<std::sync::Arc<dyn onnx_runtime_ep_api::ExecutionProvider>> {
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ep_cuda::CudaExecutionProvider;
    use std::sync::{Arc, OnceLock};

    type Cache = Mutex<Vec<(u32, Arc<CudaExecutionProvider>)>>;
    static CACHE: OnceLock<Cache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(Vec::new()));
    let mut guard = cache
        .lock()
        .map_err(|_| PyRuntimeError::new_err("cuda import-allocator cache poisoned"))?;
    if let Some((_, ep)) = guard.iter().find(|(i, _)| *i == index) {
        return Ok(ep.clone());
    }
    let mut ep = CudaExecutionProvider::new(index).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "failed to initialize CUDA device {index} for a zero-copy DLPack \
             import: {e}"
        ))
    })?;
    ep.initialize(&Default::default()).map_err(|e| {
        PyRuntimeError::new_err(format!(
            "failed to bind CUDA device {index} for a zero-copy DLPack import: {e}"
        ))
    })?;
    let ep = Arc::new(ep);
    guard.push((index, ep.clone()));
    Ok(ep)
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
    m.add_class::<ModelMetadata>()?;
    m.add_class::<Outputs>()?;
    m.add_class::<BoundSession>()?;
    dlpack::register(m)?;
    m.add_function(wrap_pyfunction!(load, m)?)?;
    m.add_function(wrap_pyfunction!(get_available_providers, m)?)?;
    m.add_function(wrap_pyfunction!(_dlpack_import_data_ptr, m)?)?;
    m.add_function(wrap_pyfunction!(_dlpack_import_drop_on_thread, m)?)?;
    #[cfg(feature = "cuda")]
    m.add_function(wrap_pyfunction!(cupti_available, m)?)?;
    Ok(())
}

#[cfg(test)]
mod honest_boundary_tests {
    //! FIX 3 unit tests for the honest CPU-session boundary guards. These are
    //! pure (no Python, no GPU): they exercise the device predicate + message
    //! wording that `run_inner` (input side) and `tensor_to_numpy` (host-read
    //! side) rely on, so the guard is covered on the CPU dev machine. The real
    //! end-to-end CUDA path is validated on hardware in `test_dlpack_gpu.py`.
    use super::{non_host_input_error, non_host_read_error};
    use onnx_runtime_ir::{DeviceId, DeviceType};

    #[test]
    fn host_devices_pass_the_input_guard() {
        for dev in [DeviceId::cpu(), DeviceId::new(DeviceType::Mlx, 0)] {
            assert!(non_host_input_error("x", dev).is_none());
        }
    }

    #[test]
    fn cuda_input_is_refused_with_actionable_message() {
        let msg = non_host_input_error("logits", DeviceId::cuda(0))
            .expect("a CUDA input must be refused on a CPU session");
        assert!(msg.contains("logits"), "names the offending input");
        assert!(msg.contains("executes on CPU"), "explains the CPU boundary");
        assert!(msg.contains("tensor.cpu()"), "offers an actionable fix");
    }

    #[test]
    fn host_devices_pass_the_read_guard() {
        for dev in [DeviceId::cpu(), DeviceId::new(DeviceType::Mlx, 0)] {
            assert!(non_host_read_error("y", dev).is_none());
        }
    }

    #[test]
    fn host_reading_cuda_output_is_refused_with_actionable_message() {
        let msg = non_host_read_error("probs", DeviceId::cuda(1))
            .expect("host-reading a CUDA output must be refused");
        assert!(msg.contains("probs"), "names the offending output");
        assert!(msg.contains("__dlpack__"), "points at the zero-copy export path");
    }
}
