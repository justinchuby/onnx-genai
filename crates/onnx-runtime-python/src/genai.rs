use std::path::Path;
use std::sync::{Mutex, MutexGuard, TryLockError};

use onnx_genai_engine::{
    Engine as RustEngine, EngineConfig, FinishReason, GenerateOptions, GenerateRequest,
    GenerateResult as RustGenerateResult, GenerateToken, StopSequence,
};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

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
    let mut options = GenerateOptions::default();
    options.max_new_tokens = max_tokens;
    options.temperature = temperature;
    options.top_p = top_p;
    options.top_k = top_k;
    options.seed = seed;
    options.greedy = temperature == 0.0;
    options.stop_sequences = stop
        .unwrap_or_default()
        .into_iter()
        .map(StopSequence::Text)
        .collect();
    Ok(options)
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

fn try_lock_engine<T>(inner: &Mutex<T>) -> PyResult<MutexGuard<'_, T>> {
    inner.try_lock().map_err(|err| match err {
        TryLockError::WouldBlock => PyRuntimeError::new_err(ENGINE_IN_USE),
        TryLockError::Poisoned(_) => PyRuntimeError::new_err(
            "nxrt genai Engine state is unavailable because a previous generation panicked; \
             create a new Engine instance",
        ),
    })
}

#[pyclass(module = "nxrt.genai", name = "Engine")]
struct Engine {
    inner: Mutex<RustEngine>,
}

#[pymethods]
impl Engine {
    #[staticmethod]
    #[pyo3(signature = (model_dir, *, num_gpu_pages=None, page_size=None))]
    fn from_dir(
        model_dir: &Bound<'_, PyAny>,
        num_gpu_pages: Option<usize>,
        page_size: Option<usize>,
    ) -> PyResult<Self> {
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
        if let Some(value) = num_gpu_pages {
            config.num_gpu_pages = value;
        }
        if let Some(value) = page_size {
            if value == 0 {
                return Err(PyValueError::new_err(
                    "page_size must be greater than zero when provided",
                ));
            }
            config.page_size = value;
        }
        let engine = RustEngine::from_dir(path_ref, config).map_err(|err| {
            PyValueError::new_err(format!(
                "failed to load genai model from {path:?}: {err}. Verify the directory \
                 contains compatible ONNX graph(s), tokenizer.json, and \
                 inference_metadata.yaml or genai_config.json."
            ))
        })?;
        Ok(Self {
            inner: Mutex::new(engine),
        })
    }

    #[pyo3(signature = (prompt, *, max_tokens=128, temperature=1.0, top_p=1.0, top_k=0, seed=None, stop=None))]
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
        py.allow_threads(|| {
            let mut engine = try_lock_engine(&self.inner)?;
            engine
                .generate(request)
                .map(GenerateResult::from)
                .map_err(generation_error)
        })
    }

    #[pyo3(signature = (prompt, callback, *, max_tokens=128, temperature=1.0, top_p=1.0, top_k=0, seed=None, stop=None))]
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
        py.allow_threads(|| {
            let mut callback_error: Option<PyErr> = None;
            let mut callback_fn = |token: GenerateToken| {
                let call = Python::with_gil(|py| {
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
                        Err(
                            std::io::Error::other("Python streaming callback raised an exception")
                                .into(),
                        )
                    }
                }
            };
            // The guard remains held while Rust generates, including callback
            // invocations. Re-entry is safe because every method uses try_lock
            // and therefore fails immediately instead of waiting on this mutex.
            let mut engine = try_lock_engine(&self.inner)?;
            let callback_fn: &mut onnx_genai_engine::GenerateTokenCallback<'_> = &mut callback_fn;
            let result = engine.generate_with_callback(request, Some(callback_fn));
            if let Some(err) = callback_error {
                return Err(err);
            }
            result.map(GenerateResult::from).map_err(generation_error)
        })
    }

    fn tokenize(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        py.allow_threads(|| {
            let engine = try_lock_engine(&self.inner)?;
            engine.tokenize(text).map_err(|err| {
                PyValueError::new_err(format!(
                    "failed to tokenize input text: {err}. Verify the model directory contains \
                     a valid tokenizer.json compatible with the loaded model."
                ))
            })
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
        .downcast_into::<PyDict>()?
        .set_item("nxrt.genai", &module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use super::{ENGINE_IN_USE, Engine, build_options, try_lock_engine};

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

    #[test]
    fn engine_pyclass_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Engine>();
    }

    #[test]
    fn engine_lock_contention_returns_actionable_python_error() {
        pyo3::prepare_freethreaded_python();
        let inner = Arc::new(Mutex::new(()));
        let guard = inner.lock().unwrap();
        let contender = Arc::clone(&inner);
        let error = std::thread::spawn(move || try_lock_engine(&contender).unwrap_err())
            .join()
            .expect("contending thread panicked");
        drop(guard);

        assert_eq!(error.to_string(), format!("RuntimeError: {ENGINE_IN_USE}"));
    }
}
