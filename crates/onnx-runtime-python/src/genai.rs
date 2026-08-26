use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle, ThreadId};

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

const ENGINE_IN_USE: &str = "nxrt genai Engine cannot be re-entered from its generation \
callback; return from the callback before calling the same Engine again";

fn reject_owner_thread(owner_thread: ThreadId) -> PyResult<()> {
    if thread::current().id() == owner_thread {
        return Err(PyRuntimeError::new_err(ENGINE_IN_USE));
    }
    Ok(())
}

type EngineTask = Box<dyn FnOnce(&mut RustEngine) + Send + 'static>;

#[pyclass(module = "nxrt.genai", name = "Engine")]
struct Engine {
    tasks: Option<Sender<EngineTask>>,
    owner_thread: ThreadId,
    callback_active: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Engine {
    fn start(path: String, config: EngineConfig) -> PyResult<Self> {
        let (tasks, task_rx) = mpsc::channel::<EngineTask>();
        let (ready_tx, ready_rx) = mpsc::sync_channel(1);
        let worker = thread::Builder::new()
            .name("nxrt-python-genai".to_string())
            .spawn(move || {
                let owner_thread = thread::current().id();
                let mut engine = match RustEngine::from_dir(Path::new(&path), config) {
                    Ok(engine) => {
                        let _ = ready_tx.send(Ok(owner_thread));
                        engine
                    }
                    Err(error) => {
                        let _ = ready_tx.send(Err(format!(
                            "failed to load genai model from {path:?}: {error}. Verify the \
                             directory contains compatible ONNX graph(s), tokenizer.json, and \
                             inference_metadata.yaml or genai_config.json."
                        )));
                        return;
                    }
                };
                for task in task_rx {
                    task(&mut engine);
                }
            })
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "failed to start the nxrt genai engine owner thread: {error}"
                ))
            })?;

        let owner_thread = ready_rx
            .recv()
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "nxrt genai engine owner thread exited before initialization completed",
                )
            })?
            .map_err(PyValueError::new_err)?;
        Ok(Self {
            tasks: Some(tasks),
            owner_thread,
            callback_active: Arc::new(AtomicBool::new(false)),
            worker: Some(worker),
        })
    }

    fn dispatch<T, F>(&self, py: Python<'_>, operation: F) -> PyResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut RustEngine) -> PyResult<T> + Send + 'static,
    {
        reject_owner_thread(self.owner_thread)?;
        if self.callback_active.load(Ordering::Acquire) {
            return Err(PyRuntimeError::new_err(ENGINE_IN_USE));
        }
        self.dispatch_unchecked(py, operation)
    }

    fn dispatch_stream<T, F>(&self, py: Python<'_>, operation: F) -> PyResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut RustEngine) -> PyResult<T> + Send + 'static,
    {
        reject_owner_thread(self.owner_thread)?;
        self.callback_active
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PyRuntimeError::new_err(ENGINE_IN_USE))?;
        let result = self.dispatch_unchecked(py, operation);
        self.callback_active.store(false, Ordering::Release);
        result
    }

    fn dispatch_unchecked<T, F>(&self, py: Python<'_>, operation: F) -> PyResult<T>
    where
        T: Send + 'static,
        F: FnOnce(&mut RustEngine) -> PyResult<T> + Send + 'static,
    {
        let (result_tx, result_rx) = mpsc::sync_channel(1);
        self.tasks
            .as_ref()
            .ok_or_else(|| {
                PyRuntimeError::new_err(
                    "nxrt genai engine is shutting down; create a new Engine instance",
                )
            })?
            .send(Box::new(move |engine| {
                let _ = result_tx.send(operation(engine));
            }))
            .map_err(|_| {
                PyRuntimeError::new_err(
                    "nxrt genai engine owner thread exited; create a new Engine instance",
                )
            })?;
        py.detach(move || {
            result_rx.recv().map_err(|_| {
                PyRuntimeError::new_err(
                    "nxrt genai engine owner thread exited before returning a result",
                )
            })?
        })
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        self.tasks.take();
        if thread::current().id() == self.owner_thread {
            return;
        }
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
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
        Self::start(path, config)
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
        self.dispatch(py, move |engine| {
            engine
                .generate(request)
                .map(GenerateResult::from)
                .map_err(generation_error)
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
        self.dispatch_stream(py, move |engine| {
            let mut callback_error: Option<PyErr> = None;
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
                        Err(
                            std::io::Error::other("Python streaming callback raised an exception")
                                .into(),
                        )
                    }
                }
            };
            let callback_fn: &mut onnx_genai_engine::GenerateTokenCallback<'_> = &mut callback_fn;
            let result = engine.generate_with_callback(request, Some(callback_fn));
            if let Some(err) = callback_error {
                return Err(err);
            }
            result.map(GenerateResult::from).map_err(generation_error)
        })
    }

    fn tokenize(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        let text = text.to_string();
        self.dispatch(py, move |engine| {
            engine.tokenize(&text).map_err(|err| {
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
        .cast_into::<PyDict>()?
        .set_item("nxrt.genai", &module)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{ENGINE_IN_USE, Engine, build_options, reject_owner_thread};

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
    fn engine_owner_thread_reentry_returns_actionable_python_error() {
        pyo3::Python::initialize();
        let error = reject_owner_thread(std::thread::current().id()).unwrap_err();
        assert_eq!(error.to_string(), format!("RuntimeError: {ENGINE_IN_USE}"));
    }

    #[test]
    fn another_thread_may_dispatch_to_the_owner() {
        let owner = std::thread::current().id();
        std::thread::spawn(move || reject_owner_thread(owner).unwrap())
            .join()
            .expect("client thread panicked");
    }
}
