//! Python extension module `onnx_genai`.
//!
//! Exposes the ONNX Runtime-backed genai `Engine` and `GenerateResult` at the
//! top level of the `onnx_genai` module. The API mirrors `nxrt.genai`, but this
//! module runs on ONNX Runtime (the default `onnx-genai-engine` backend), so it
//! is ONNX Runtime compatible; `nxrt.genai` provides the same API backed by the
//! native nxrt runtime.

use std::path::{Path, PathBuf};
use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
    mpsc as std_mpsc,
};
use std::thread;

use onnx_genai_engine::{
    Engine as RustEngine, EngineConfig, FinishReason, GenerateOptions, GenerateRequest,
    GenerateResult as RustGenerateResult, GenerateToken, SamplingOverrides, StopSequence,
};
use pyo3::exceptions::{PyFileNotFoundError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyModule, PyString};

const VERSION: &str = env!("CARGO_PKG_VERSION");

fn finish_reason_name(reason: &FinishReason) -> String {
    match reason {
        FinishReason::MaxTokens => "max_tokens".to_string(),
        FinishReason::EosToken => "eos_token".to_string(),
        FinishReason::StopSequence { index } => format!("stop_sequence:{index}"),
        FinishReason::Length => "length".to_string(),
        FinishReason::ToolCalls => "tool_calls".to_string(),
    }
}

#[pyclass(module = "onnx_genai", name = "GenerateResult", frozen)]
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

/// The caller's *explicit* sampling selections, for
/// [`GenerateOptions::resolve_sampling_defaults`].
///
/// Each argument is `None` when the Python caller omitted the keyword, leaving
/// that control to the runtime greedy fallback. An explicit `temperature=0`
/// forces greedy; any other explicit sampling control requests sampling.
fn sampling_overrides(
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
) -> SamplingOverrides {
    let forces_greedy = temperature == Some(0.0);
    let requests_sampling = seed.is_some()
        || temperature.is_some_and(|value| value > 0.0)
        || top_p.is_some()
        || top_k.is_some_and(|value| value > 0);
    let greedy = if forces_greedy {
        Some(true)
    } else if requests_sampling {
        Some(false)
    } else {
        None
    };
    SamplingOverrides {
        greedy,
        temperature,
        top_p,
        top_k,
    }
}

fn build_options(
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
) -> PyResult<GenerateOptions> {
    if max_tokens == 0 {
        return Err(PyValueError::new_err(
            "max_tokens must be greater than zero; choose the maximum number of new tokens",
        ));
    }
    if let Some(temperature) = temperature
        && (!temperature.is_finite() || temperature < 0.0)
    {
        return Err(PyValueError::new_err(
            "temperature must be finite and non-negative; use 0 for greedy decoding",
        ));
    }
    if let Some(top_p) = top_p
        && (!top_p.is_finite() || !(0.0..=1.0).contains(&top_p))
    {
        return Err(PyValueError::new_err(
            "top_p must be finite and between 0 and 1 inclusive",
        ));
    }
    // Sampling controls (temperature/top_p/top_k/greedy) are intentionally left
    // at their defaults here and resolved later from the caller's explicit
    // keyword arguments in `resolve_sampling_defaults`.
    Ok(GenerateOptions {
        max_new_tokens: max_tokens,
        seed,
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
    temperature: Option<f32>,
    top_p: Option<f32>,
    top_k: Option<usize>,
    seed: Option<u64>,
    stop: Option<Vec<String>>,
) -> PyResult<(GenerateRequest, SamplingOverrides)> {
    if prompt.is_empty() {
        return Err(PyValueError::new_err(
            "prompt must not be empty; pass text containing at least one model token",
        ));
    }
    let overrides = sampling_overrides(temperature, top_p, top_k, seed);
    Ok((
        GenerateRequest {
            prompt: prompt.into(),
            options: build_options(max_tokens, temperature, top_p, seed, stop)?,
        },
        overrides,
    ))
}

fn generation_error(err: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(format!(
        "text generation failed: {err}. Verify the prompt fits the model context, \
         generation parameters are valid, and the model directory contains matching \
         ONNX graphs, metadata/config, and tokenizer files."
    ))
}

const ENGINE_IN_USE: &str = "engine is in use by another thread — onnx_genai Engine is not \
re-entrant; serialize calls or use one Engine per thread";

fn owner_stopped() -> PyErr {
    PyRuntimeError::new_err(
        "onnx_genai Engine owner thread stopped unexpectedly; create a new Engine instance",
    )
}

type EngineOperation = Box<dyn FnOnce(&mut RustEngine) + Send + 'static>;

enum EngineCommand {
    Run(EngineOperation),
    Shutdown,
}

#[derive(Debug)]
struct EngineCallGuard<'a> {
    in_use: &'a AtomicBool,
}

impl Drop for EngineCallGuard<'_> {
    fn drop(&mut self) {
        self.in_use.store(false, Ordering::Release);
    }
}

struct EngineOwner {
    commands: std_mpsc::Sender<EngineCommand>,
    join: Mutex<Option<thread::JoinHandle<()>>>,
    in_use: AtomicBool,
}

impl EngineOwner {
    fn start(model_dir: PathBuf, config: EngineConfig) -> Result<Self, String> {
        let (commands, rx) = std_mpsc::channel();
        let (ready_tx, ready_rx) = std_mpsc::sync_channel(1);
        let join = thread::Builder::new()
            .name("onnx-genai-python-engine".to_string())
            .spawn(move || match RustEngine::from_dir(&model_dir, config) {
                Ok(mut engine) => {
                    if ready_tx.send(Ok(())).is_err() {
                        return;
                    }
                    while let Ok(command) = rx.recv() {
                        match command {
                            EngineCommand::Run(operation) => operation(&mut engine),
                            EngineCommand::Shutdown => break,
                        }
                    }
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error.to_string()));
                }
            })
            .map_err(|error| format!("failed to spawn Python engine owner: {error}"))?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                commands,
                join: Mutex::new(Some(join)),
                in_use: AtomicBool::new(false),
            }),
            Ok(Err(error)) => {
                let _ = join.join();
                Err(error)
            }
            Err(_) => {
                let panic = join.join().is_err();
                if panic {
                    return Err("Python engine owner panicked during initialization".to_string());
                }
                Err("Python engine owner exited before reporting initialization".to_string())
            }
        }
    }

    fn begin_call(&self) -> PyResult<EngineCallGuard<'_>> {
        self.in_use
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| PyRuntimeError::new_err(ENGINE_IN_USE))?;
        Ok(EngineCallGuard {
            in_use: &self.in_use,
        })
    }

    fn call<R>(
        &self,
        operation: impl FnOnce(&mut RustEngine) -> PyResult<R> + Send + 'static,
    ) -> PyResult<R>
    where
        R: Send + 'static,
    {
        let (reply, response) = std_mpsc::sync_channel(1);
        self.commands
            .send(EngineCommand::Run(Box::new(move |engine| {
                let _ = reply.send(operation(engine));
            })))
            .map_err(|_| owner_stopped())?;
        response.recv().map_err(|_| owner_stopped())?
    }
}

impl Drop for EngineOwner {
    fn drop(&mut self) {
        let _ = self.commands.send(EngineCommand::Shutdown);
        if let Some(join) = self
            .join
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .take()
        {
            let _ = join.join();
        }
    }
}

#[pyclass(module = "onnx_genai", name = "Engine")]
struct Engine {
    owner: EngineOwner,
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
        let owner = EngineOwner::start(path_ref.to_path_buf(), config).map_err(|err| {
            PyValueError::new_err(format!(
                "failed to load genai model from {path:?}: {err}. Verify the directory \
                 contains compatible ONNX graph(s), tokenizer.json, and \
                 inference_metadata.yaml or genai_config.json."
            ))
        })?;
        Ok(Self { owner })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (prompt, *, max_tokens=128, temperature=None, top_p=None, top_k=None, seed=None, stop=None))]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
        stop: Option<Vec<String>>,
    ) -> PyResult<GenerateResult> {
        let (mut request, overrides) =
            request(prompt, max_tokens, temperature, top_p, top_k, seed, stop)?;
        py.detach(|| {
            let _call = self.owner.begin_call()?;
            self.owner.call(move |engine| {
                // Inference metadata no longer carries generation defaults, so
                // only explicit kwargs select sampling.
                request.options.resolve_sampling_defaults(None, &overrides);
                engine
                    .generate(request)
                    .map(GenerateResult::from)
                    .map_err(generation_error)
            })
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (prompt, callback, *, max_tokens=128, temperature=None, top_p=None, top_k=None, seed=None, stop=None))]
    fn generate_stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        callback: Py<PyAny>,
        max_tokens: usize,
        temperature: Option<f32>,
        top_p: Option<f32>,
        top_k: Option<usize>,
        seed: Option<u64>,
        stop: Option<Vec<String>>,
    ) -> PyResult<GenerateResult> {
        if !callback.bind(py).is_callable() {
            return Err(PyTypeError::new_err(
                "callback must be callable and accept (text, token_id, finish_reason)",
            ));
        }
        let (mut request, overrides) =
            request(prompt, max_tokens, temperature, top_p, top_k, seed, stop)?;
        py.detach(|| {
            let _call = self.owner.begin_call()?;
            self.owner.call(move |engine| {
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
                            Err(std::io::Error::other(
                                "Python streaming callback raised an exception",
                            )
                            .into())
                        }
                    }
                };
                request.options.resolve_sampling_defaults(None, &overrides);
                let callback_fn: &mut onnx_genai_engine::GenerateTokenCallback<'_> =
                    &mut callback_fn;
                let result = engine.generate_with_callback(request, Some(callback_fn));
                if let Some(err) = callback_error {
                    return Err(err);
                }
                result.map(GenerateResult::from).map_err(generation_error)
            })
        })
    }

    fn tokenize(&self, py: Python<'_>, text: &str) -> PyResult<Vec<u32>> {
        let text = text.to_string();
        py.detach(|| {
            let _call = self.owner.begin_call()?;
            self.owner.call(move |engine| {
                engine.tokenize(&text).map_err(|err| {
                    PyValueError::new_err(format!(
                        "failed to tokenize input text: {err}. Verify the model directory contains \
                         a valid tokenizer.json compatible with the loaded model."
                    ))
                })
            })
        })
    }
}

#[pymodule]
fn _onnx_genai(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    let doc = "onnx_genai — ONNX Runtime-backed text generation. Same API as nxrt.genai, \
               implemented on ONNX Runtime.";
    m.add("__doc__", PyString::new(m.py(), doc))?;
    m.add_class::<Engine>()?;
    m.add_class::<GenerateResult>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{ENGINE_IN_USE, EngineOwner, build_options, sampling_overrides};
    use onnx_genai_engine::{GenerateOptions, GenerationDefaults};

    #[test]
    fn build_options_carries_non_sampling_arguments() {
        let options = build_options(
            17,
            Some(0.7),
            Some(0.9),
            Some(42),
            Some(vec!["stop".into()]),
        )
        .unwrap();
        assert_eq!(options.max_new_tokens, 17);
        assert_eq!(options.seed, Some(42));
        assert_eq!(options.stop_sequences.len(), 1);
        // Sampling controls are deferred to resolve_sampling_defaults, so the
        // base options keep their defaults here.
        assert_eq!(options.temperature, GenerateOptions::default().temperature);
        assert!(options.greedy);
    }

    #[test]
    fn explicit_kwargs_map_to_sampling_overrides() {
        // Explicit sampling controls request sampling and are carried through.
        let overrides = sampling_overrides(Some(0.7), Some(0.9), Some(12), Some(42));
        assert_eq!(overrides.greedy, Some(false));
        assert_eq!(overrides.temperature, Some(0.7));
        assert_eq!(overrides.top_p, Some(0.9));
        assert_eq!(overrides.top_k, Some(12));

        // An explicit temperature of 0 forces greedy.
        assert_eq!(
            sampling_overrides(Some(0.0), None, None, None).greedy,
            Some(true)
        );

        // A fully-silent call defers the greedy decision to the model.
        assert_eq!(sampling_overrides(None, None, None, None).greedy, None);
    }

    #[test]
    fn silent_call_honors_model_declared_sampling() {
        // A caller that passes no sampling kwargs adopts the model's declared
        // do_sample/temperature instead of the greedy fallback.
        let mut options = build_options(8, None, None, None, None).unwrap();
        assert!(options.greedy, "default before resolution is greedy");
        let declared = GenerationDefaults {
            do_sample: Some(true),
            temperature: Some(0.6),
            top_k: None,
            top_p: None,
            repetition_penalty: None,
            num_beams: None,
            num_return_sequences: None,
            min_length: None,
            max_length: None,
            length_penalty: None,
            no_repeat_ngram_size: None,
            diversity_penalty: None,
            early_stopping: None,
        };
        options.resolve_sampling_defaults(
            Some(&declared),
            &sampling_overrides(None, None, None, None),
        );
        assert!(!options.greedy, "model do_sample=true must disable greedy");
        assert_eq!(options.temperature, 0.6);
    }

    #[test]
    fn engine_lock_contention_returns_actionable_python_error() {
        pyo3::Python::initialize();
        let model_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
        let owner = Arc::new(
            EngineOwner::start(model_dir, onnx_genai_engine::EngineConfig::default()).unwrap(),
        );
        let guard = owner.begin_call().unwrap();
        let contender = Arc::clone(&owner);
        let error = std::thread::spawn(move || contender.begin_call().unwrap_err())
            .join()
            .expect("contending thread panicked");
        drop(guard);

        assert_eq!(error.to_string(), format!("RuntimeError: {ENGINE_IN_USE}"));
    }

    #[test]
    fn engine_owner_is_cross_thread_safe_and_keeps_engine_on_owner_thread() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EngineOwner>();
        pyo3::Python::initialize();
        let caller = std::thread::current().id();
        let model_dir =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-llm");
        let owner = Arc::new(
            EngineOwner::start(model_dir, onnx_genai_engine::EngineConfig::default()).unwrap(),
        );
        let operation_owner = Arc::clone(&owner);
        let (operation_thread, tokens) = std::thread::spawn(move || {
            let _guard = operation_owner.begin_call().unwrap();
            operation_owner
                .call(|engine| {
                    Ok((
                        std::thread::current().id(),
                        engine.tokenize("hello").unwrap(),
                    ))
                })
                .unwrap()
        })
        .join()
        .unwrap();

        assert_ne!(operation_thread, caller);
        assert!(!tokens.is_empty());
    }
}
