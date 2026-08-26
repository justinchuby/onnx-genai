use std::path::Path;

use onnx_genai_engine::{
    Engine as RustEngine, EngineConfig, FinishReason, GenerateOptions, GenerateRequest,
    GenerateResult as RustGenerateResult, GenerateToken, StopSequence,
};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

use crate::thread_owned::{CallError, SpawnError, ThreadOwned};

fn finish_reason_name(reason: &FinishReason) -> String {
    match reason {
        FinishReason::MaxTokens => "max_tokens".to_string(),
        FinishReason::EosToken => "eos_token".to_string(),
        FinishReason::StopSequence { index } => format!("stop_sequence:{index}"),
        FinishReason::Length => "length".to_string(),
    }
}

#[pyclass(module = "nxrt.genai", name = "GenerateResult", frozen)]
struct GenerateResult {
    #[pyo3(get)]
    text: String,
    #[pyo3(get)]
    token_ids: Vec<u32>,
    #[pyo3(get)]
    finish_reason: String,
    #[pyo3(get)]
    prefix_cache_hit_len: usize,
}

impl From<RustGenerateResult> for GenerateResult {
    fn from(result: RustGenerateResult) -> Self {
        Self {
            text: result.text,
            token_ids: result.token_ids,
            finish_reason: finish_reason_name(&result.finish_reason),
            prefix_cache_hit_len: result.prefix_cache_hit_len,
        }
    }
}

#[pymethods]
impl GenerateResult {
    fn __repr__(&self) -> String {
        format!(
            "GenerateResult(text={:?}, token_ids={}, finish_reason={:?}, \
             prefix_cache_hit_len={})",
            self.text,
            self.token_ids.len(),
            self.finish_reason,
            self.prefix_cache_hit_len
        )
    }
}

fn build_options(
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
) -> PyResult<GenerateOptions> {
    if max_tokens == 0 {
        return Err(PyValueError::new_err(
            "max_tokens must be greater than zero; choose the maximum number of new tokens",
        ));
    }
    if !temperature.is_finite() || temperature < 0.0 {
        return Err(PyValueError::new_err(
            "temperature must be finite and non-negative; use 0 for greedy decoding",
        ));
    }
    if !top_p.is_finite() || !(0.0..=1.0).contains(&top_p) {
        return Err(PyValueError::new_err(
            "top_p must be finite and between 0 and 1 inclusive",
        ));
    }
    Ok(GenerateOptions {
        max_new_tokens: max_tokens,
        temperature,
        top_p,
        top_k,
        seed,
        greedy: temperature == 0.0,
        stop_sequences: stop
            .unwrap_or_default()
            .into_iter()
            .map(StopSequence::Text)
            .collect(),
        ..GenerateOptions::default()
    })
}

fn request(
    prompt: &str,
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    top_k: usize,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
) -> PyResult<GenerateRequest> {
    if prompt.is_empty() {
        return Err(PyValueError::new_err(
            "prompt must not be empty; pass text containing at least one model token",
        ));
    }
    Ok(GenerateRequest {
        prompt: prompt.into(),
        options: build_options(max_tokens, temperature, top_p, top_k, seed, stop)?,
    })
}

fn generation_error(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!(
        "text generation failed: {err}. Verify the prompt fits the model context, \
         generation parameters are valid, and the model directory contains matching \
         ONNX graphs, metadata/config, and tokenizer files."
    ))
}

const ENGINE_IN_USE: &str = "engine is in use by another thread — nxrt genai Engine is not \
re-entrant; serialize calls or use one Engine per thread";

const ENGINE_LOST: &str = "nxrt genai Engine state is unavailable because a previous generation \
panicked; create a new Engine instance";

/// Map a failed call on the owning thread onto the two Python errors this
/// module has always raised. The wording is unchanged from when the engine was
/// held in a `Mutex` on the calling thread: the same two situations arise, and
/// the advice for each is the same. The panic message is appended when there is
/// one, because the `Mutex` version unwound into the caller and Python saw it.
fn call_error(err: CallError) -> PyErr {
    match err {
        CallError::InUse => PyRuntimeError::new_err(ENGINE_IN_USE),
        CallError::WorkerLost { panic: None } => PyRuntimeError::new_err(ENGINE_LOST),
        CallError::WorkerLost { panic: Some(panic) } => {
            PyRuntimeError::new_err(format!("{ENGINE_LOST}. The panic was: {panic}"))
        }
    }
}

/// A genai engine, owned by a thread of its own.
///
/// The engine is `!Send` by construction (see [`crate::thread_owned`]), so it
/// is not stored here — only a handle to the thread that holds it. Every method
/// hands a closure to that thread and waits for the answer.
#[pyclass(module = "nxrt.genai", name = "Engine")]
struct Engine {
    inner: ThreadOwned<RustEngine>,
}

#[pymethods]
impl Engine {
    #[staticmethod]
    #[pyo3(signature = (model_dir, *, page_size=None))]
    fn from_dir(model_dir: &Bound<'_, PyAny>, page_size: Option<usize>) -> PyResult<Self> {
        let path = model_dir
            .str()
            .map_err(|_| {
                PyTypeError::new_err(
                    "Engine.from_dir(model_dir): expected a filesystem path (str/os.PathLike)",
                )
            })?
            .to_string_lossy()
            .into_owned();
        let path_ref = Path::new(&path);
        if !path_ref.exists() {
            return Err(PyFileNotFoundError::new_err(format!(
                "genai model directory not found: {path:?}. Pass a directory containing \
                 the model ONNX graph(s), tokenizer.json, and model metadata/config."
            )));
        }
        if !path_ref.is_dir() {
            return Err(PyValueError::new_err(format!(
                "genai model path is not a directory: {path:?}. Engine.from_dir expects \
                 a model directory, not an individual .onnx file."
            )));
        }
        let mut config = EngineConfig::default();
        if let Some(value) = page_size {
            if value == 0 {
                return Err(PyValueError::new_err(
                    "page_size must be greater than zero when provided",
                ));
            }
            config.page_size = value;
        }
        // Built on the worker, because the sessions, allocators and bindings it
        // creates belong to whichever thread creates them.
        let owned_path = path.clone();
        let inner = ThreadOwned::new("nxrt-genai-engine", move || {
            RustEngine::from_dir(Path::new(&owned_path), config).map_err(|err| err.to_string())
        })
        .map_err(|err| match err {
            SpawnError::Build(err) => PyValueError::new_err(format!(
                "failed to load genai model from {path:?}: {err}. Verify the directory \
                 contains compatible ONNX graph(s), tokenizer.json, and \
                 inference_metadata.yaml or genai_config.json."
            )),
            SpawnError::Thread(err) => PyRuntimeError::new_err(format!(
                "failed to start the thread that owns the genai engine: {err}. Each \
                 Engine holds its ONNX Runtime handles on a dedicated thread."
            )),
        })?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (prompt, *, max_tokens=128, temperature=1.0, top_p=1.0, top_k=0, seed=None, stop=None))]
    // The Python API intentionally exposes each generation option as a keyword argument.
    #[allow(clippy::too_many_arguments)]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: Option<u64>,
        stop: Option<Vec<String>>,
    ) -> PyResult<GenerateResult> {
        let request = request(prompt, max_tokens, temperature, top_p, top_k, seed, stop)?;
        py.detach(|| {
            self.inner
                .with(move |engine| {
                    engine
                        .generate(request)
                        .map(GenerateResult::from)
                        .map_err(generation_error)
                })
                .map_err(call_error)?
        })
    }

    #[pyo3(signature = (prompt, callback, *, max_tokens=128, temperature=1.0, top_p=1.0, top_k=0, seed=None, stop=None))]
    // The Python API intentionally exposes each generation option as a keyword argument.
    #[allow(clippy::too_many_arguments)]
    fn generate_stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        callback: Py<PyAny>,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
        top_k: usize,
        seed: Option<u64>,
        stop: Option<Vec<String>>,
    ) -> PyResult<GenerateResult> {
        if !callback.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "callback must be callable and accept (text, token_id, finish_reason)",
            ));
        }
        let request = request(prompt, max_tokens, temperature, top_p, top_k, seed, stop)?;
        py.detach(|| {
            self.inner
                .with(move |engine| {
                    let mut callback_error: Option<PyErr> = None;
                    // Runs on the engine's thread, which holds no GIL, so it
                    // takes one per token. The calling thread has released its
                    // own for the duration of this call.
                    let mut callback_fn = |token: GenerateToken| {
                        let call = Python::attach(|py| {
                            callback.call1(
                                py,
                                (
                                    token.text,
                                    token.token_id,
                                    token.finish_reason.as_ref().map(finish_reason_name),
                                ),
                            )
                        });
                        match call {
                            Ok(_) => Ok(()),
                            Err(err) => {
                                callback_error = Some(err);
                                Err(std::io::Error::other(
                                    "Python streaming callback raised an exception",
                                )
                                .into())
                            }
                        }
                    };
                    let callback_fn: &mut onnx_genai_engine::GenerateTokenCallback<'_> =
                        &mut callback_fn;
                    let result = engine.generate_with_callback(request, Some(callback_fn));
                    if let Some(err) = callback_error {
                        return Err(err);
                    }
                    result.map(GenerateResult::from).map_err(generation_error)
                })
                .map_err(call_error)?
        })
    }

    fn tokenize(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        // Owned because the job outlives this borrow on another thread.
        let text = text.to_owned();
        py.detach(|| {
            self.inner
                .with(move |engine| {
                    engine.tokenize(&text).map_err(|err| {
                        PyValueError::new_err(format!(
                            "failed to tokenize input text: {err}. Verify the model directory \
                             contains a valid tokenizer.json compatible with the loaded model."
                        ))
                    })
                })
                .map_err(call_error)?
        })
    }
}

pub(crate) fn register(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    let module = PyModule::new(py, "genai")?;
    module.add(
        "__doc__",
        "Local text generation using nxrt's Rust genai engine (no webserver).",
    )?;
    module.add_class::<Engine>()?;
    module.add_class::<GenerateResult>()?;
    parent.add_submodule(&module)?;
    py.import("sys")?
        .getattr("modules")?
        .cast_into::<PyDict>()?
        .set_item("nxrt.genai", &module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;
    use std::rc::Rc;

    use super::{ENGINE_IN_USE, ENGINE_LOST, Engine, build_options, call_error};
    use crate::thread_owned::CallError;

    #[test]
    fn generation_options_match_python_arguments() {
        let options = build_options(17, 0.7, 0.9, 12, Some(42), Some(vec!["stop".into()])).unwrap();
        assert_eq!(options.max_new_tokens, 17);
        assert_eq!(options.temperature, 0.7);
        assert_eq!(options.top_p, 0.9);
        assert_eq!(options.top_k, 12);
        assert_eq!(options.seed, Some(42));
        assert!(!options.greedy);
        assert_eq!(options.stop_sequences.len(), 1);
    }

    /// The `#[pyclass]` requirement that the engine crate's `!Send` worker state
    /// used to satisfy only through `unsafe impl Send for Engine` (removed in
    /// #2132). It now holds structurally; see `crate::thread_owned`.
    #[test]
    fn engine_pyclass_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    }

    /// The engine itself must stay `!Send`. This is a tripwire, not a
    /// requirement: if it fails, `onnx_genai_engine::Engine` became `Send`
    /// again, and the first question is whether that happened structurally or
    /// by a reinstated `unsafe impl Send`. Only in the first case is deleting
    /// [`crate::thread_owned`] the right response.
    #[test]
    fn the_underlying_engine_is_still_not_send() {
        // Inherent methods win over trait methods, and the inherent impl only
        // applies when `T: Send` — the stable stand-in for specialization.
        struct Probe<T>(PhantomData<T>);
        trait NotSend {
            fn is_send(&self) -> bool {
                false
            }
        }
        impl<T> NotSend for Probe<T> {}
        impl<T: Send> Probe<T> {
            fn is_send(&self) -> bool {
                true
            }
        }

        assert!(
            Probe::<u32>(PhantomData).is_send(),
            "positive control: u32 is Send, so the probe is not simply always false"
        );
        assert!(
            !Probe::<Rc<()>>(PhantomData).is_send(),
            "negative control: Rc is not Send"
        );
        assert!(
            !Probe::<super::RustEngine>(PhantomData).is_send(),
            "onnx_genai_engine::Engine is Send again — see this test's doc comment"
        );
    }

    #[test]
    fn call_failures_keep_their_python_wording() {
        pyo3::Python::initialize();
        assert_eq!(
            call_error(CallError::InUse).to_string(),
            format!("RuntimeError: {ENGINE_IN_USE}")
        );
        assert_eq!(
            call_error(CallError::WorkerLost { panic: None }).to_string(),
            format!("RuntimeError: {ENGINE_LOST}")
        );
        assert_eq!(
            call_error(CallError::WorkerLost {
                panic: Some("kv cache overflow".to_owned())
            })
            .to_string(),
            format!("RuntimeError: {ENGINE_LOST}. The panic was: kv cache overflow")
        );
    }
}
