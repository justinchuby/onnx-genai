use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use half::{bf16, f16};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{CanonicalTensor, CaseRecord, ConformanceDType, OracleError, infer_output_shape};

const ADAPTER: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/adapters/python_reference.py");

/// Installed Python execution engine.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PythonEngine {
    /// `onnx.reference.ReferenceEvaluator`.
    OnnxReference,
    /// `onnxruntime.InferenceSession` with CPUExecutionProvider.
    OnnxRuntime,
}

/// Adapter availability result.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AdapterStatus {
    /// Engine is available.
    Available,
    /// Dependency or schema lane is unavailable.
    Unavailable,
    /// Adapter execution failed.
    Error,
}

/// Installed adapter versions and latest locally exposed Einsum schema.
#[derive(Clone, Debug, Deserialize)]
pub struct AdapterProbe {
    /// ONNX `ReferenceEvaluator` status.
    pub status: AdapterStatus,
    /// Installed ONNX version.
    pub onnx_version: Option<String>,
    /// Latest Einsum schema exposed by installed ONNX.
    pub latest_einsum_schema: Option<u64>,
    /// ONNX Runtime status.
    pub onnxruntime_status: AdapterStatus,
    /// Installed ONNX Runtime version.
    pub onnxruntime_version: Option<String>,
    /// ONNX `ReferenceEvaluator` unavailability or error reason.
    pub reason: Option<String>,
    /// ONNX Runtime unavailability or error reason.
    pub onnxruntime_reason: Option<String>,
}

/// Subprocess adapter for optional installed Python reference engines.
#[derive(Clone, Debug)]
pub struct PythonReferenceAdapter {
    python: PathBuf,
}

impl Default for PythonReferenceAdapter {
    fn default() -> Self {
        Self::new("python3")
    }
}

impl PythonReferenceAdapter {
    /// Select a Python interpreter.
    pub fn new(python: impl Into<PathBuf>) -> Self {
        Self {
            python: python.into(),
        }
    }

    /// Probe installed ONNX/ORT without treating them as schema authority.
    pub fn probe(&self) -> Result<AdapterProbe, ReferenceAdapterError> {
        let output = Command::new(&self.python)
            .arg(ADAPTER)
            .arg("--probe")
            .output()
            .map_err(|source| ReferenceAdapterError::Spawn {
                program: self.python.clone(),
                source,
            })?;
        decode_probe(&output.stdout, &output.stderr, output.status.success())
    }

    /// Execute a finite floating case in an installed reference engine.
    pub fn run(
        &self,
        engine: PythonEngine,
        case: &CaseRecord,
        inputs: &[CanonicalTensor],
    ) -> Result<CanonicalTensor, ReferenceAdapterError> {
        if !matches!(
            case.dtype,
            ConformanceDType::Float16
                | ConformanceDType::Float32
                | ConformanceDType::Float64
                | ConformanceDType::BFloat16
        ) {
            return Err(ReferenceAdapterError::Unsupported(format!(
                "Python reference adapter does not implement {:?}",
                case.dtype
            )));
        }
        let input_records = inputs
            .iter()
            .enumerate()
            .map(|(index, tensor)| {
                let values = tensor.to_f64_values();
                if values.iter().any(|value| !value.is_finite()) {
                    return Err(ReferenceAdapterError::Unsupported(format!(
                        "input #{index} contains NaN or infinity; JSON adapter lanes are finite-only"
                    )));
                }
                Ok(AdapterTensor {
                    shape: tensor.shape().to_vec(),
                    values,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let request = AdapterRequest {
            engine,
            equation: &case.equation,
            opset: case.opset,
            dtype: dtype_name(case.dtype),
            output_shape: infer_output_shape(&case.equation, &case.input_shapes)?,
            inputs: input_records,
        };
        let mut child = Command::new(&self.python)
            .arg(ADAPTER)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| ReferenceAdapterError::Spawn {
                program: self.python.clone(),
                source,
            })?;
        serde_json::to_writer(child.stdin.as_mut().expect("piped stdin"), &request)?;
        child.stdin.as_mut().expect("piped stdin").flush()?;
        drop(child.stdin.take());
        let output = child.wait_with_output()?;
        let response: AdapterResponse =
            decode_output(&output.stdout, &output.stderr, output.status.success())?;
        match response.status {
            AdapterStatus::Available => {
                let shape = response.shape.ok_or_else(|| {
                    ReferenceAdapterError::Protocol("available response omitted shape".into())
                })?;
                let values = response.values.ok_or_else(|| {
                    ReferenceAdapterError::Protocol("available response omitted values".into())
                })?;
                let bits = values
                    .into_iter()
                    .map(|value| value_bits(case.dtype, value))
                    .collect();
                CanonicalTensor::new(case.dtype, shape, bits).map_err(Into::into)
            }
            AdapterStatus::Unavailable => Err(ReferenceAdapterError::Unavailable(
                response
                    .reason
                    .unwrap_or_else(|| "no reason returned".into()),
            )),
            AdapterStatus::Error => Err(ReferenceAdapterError::Protocol(
                response
                    .reason
                    .unwrap_or_else(|| "no reason returned".into()),
            )),
        }
    }
}

#[derive(Serialize)]
struct AdapterRequest<'a> {
    engine: PythonEngine,
    equation: &'a str,
    opset: u64,
    dtype: &'static str,
    output_shape: Vec<usize>,
    inputs: Vec<AdapterTensor>,
}

#[derive(Serialize)]
struct AdapterTensor {
    shape: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Deserialize)]
struct AdapterResponse {
    status: AdapterStatus,
    shape: Option<Vec<usize>>,
    values: Option<Vec<f64>>,
    reason: Option<String>,
}

fn dtype_name(dtype: ConformanceDType) -> &'static str {
    match dtype {
        ConformanceDType::Float16 => "float16",
        ConformanceDType::Float32 => "float32",
        ConformanceDType::Float64 => "float64",
        ConformanceDType::BFloat16 => "bfloat16",
        _ => "integer",
    }
}

fn value_bits(dtype: ConformanceDType, value: f64) -> u64 {
    match dtype {
        ConformanceDType::Float16 => u64::from(f16::from_f32(value as f32).to_bits()),
        ConformanceDType::BFloat16 => u64::from(bf16::from_f32(value as f32).to_bits()),
        ConformanceDType::Float32 => u64::from((value as f32).to_bits()),
        ConformanceDType::Float64 => value.to_bits(),
        _ => unreachable!("adapter is floating-only"),
    }
}

fn decode_output<T: for<'de> Deserialize<'de>>(
    stdout: &[u8],
    stderr: &[u8],
    success: bool,
) -> Result<T, ReferenceAdapterError> {
    if !success {
        return Err(ReferenceAdapterError::Process {
            stderr: String::from_utf8_lossy(stderr).into_owned(),
            stdout: String::from_utf8_lossy(stdout).into_owned(),
        });
    }
    serde_json::from_slice(stdout).map_err(|source| ReferenceAdapterError::Decode {
        source,
        stdout: String::from_utf8_lossy(stdout).into_owned(),
        stderr: String::from_utf8_lossy(stderr).into_owned(),
    })
}

fn decode_probe(
    stdout: &[u8],
    stderr: &[u8],
    success: bool,
) -> Result<AdapterProbe, ReferenceAdapterError> {
    let probe: AdapterProbe = decode_output(stdout, stderr, success)?;
    validate_probe_status(
        "ONNX ReferenceEvaluator",
        &probe.status,
        probe.reason.as_deref(),
    )?;
    validate_probe_status(
        "ONNX Runtime",
        &probe.onnxruntime_status,
        probe.onnxruntime_reason.as_deref(),
    )?;
    if probe.status == AdapterStatus::Available
        && (probe.onnx_version.is_none() || probe.latest_einsum_schema.is_none())
    {
        return Err(ReferenceAdapterError::Protocol(
            "available ONNX ReferenceEvaluator probe omitted ONNX version or Einsum schema".into(),
        ));
    }
    if probe.onnxruntime_status == AdapterStatus::Available && probe.onnxruntime_version.is_none() {
        return Err(ReferenceAdapterError::Protocol(
            "available ONNX Runtime probe omitted its version".into(),
        ));
    }
    Ok(probe)
}

fn validate_probe_status(
    engine: &str,
    status: &AdapterStatus,
    reason: Option<&str>,
) -> Result<(), ReferenceAdapterError> {
    if *status != AdapterStatus::Available && reason.is_none_or(str::is_empty) {
        return Err(ReferenceAdapterError::Protocol(format!(
            "{engine} probe reported {status:?} without a reason"
        )));
    }
    Ok(())
}

/// Optional Python reference adapter failure.
#[derive(Debug, Error)]
pub enum ReferenceAdapterError {
    /// Interpreter could not be launched.
    #[error("failed to launch Python reference adapter with {program:?}: {source}")]
    Spawn {
        /// Program path.
        program: PathBuf,
        /// OS error.
        source: std::io::Error,
    },
    /// Adapter exited unsuccessfully.
    #[error("Python reference adapter failed; stderr: {stderr}; stdout: {stdout}")]
    Process {
        /// Stderr.
        stderr: String,
        /// Stdout.
        stdout: String,
    },
    /// JSON could not be decoded.
    #[error(
        "Python reference adapter returned invalid JSON: {source}; stderr: {stderr}; stdout: {stdout}"
    )]
    Decode {
        /// JSON error.
        source: serde_json::Error,
        /// Stdout.
        stdout: String,
        /// Stderr.
        stderr: String,
    },
    /// I/O failure after launch.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// JSON encoding failure.
    #[error(transparent)]
    Json(#[from] serde_json::Error),
    /// Direct equation analysis failed.
    #[error(transparent)]
    Equation(#[from] crate::EquationError),
    /// Tensor construction failed.
    #[error(transparent)]
    Oracle(#[from] OracleError),
    /// Engine or lane is intentionally unavailable.
    #[error("Python reference adapter unavailable: {0}")]
    Unavailable(String),
    /// Request is outside the adapter's portable lane.
    #[error("Python reference adapter request unsupported: {0}")]
    Unsupported(String),
    /// Adapter response violated the protocol.
    #[error("Python reference adapter protocol error: {0}")]
    Protocol(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_probe(json: &str) -> AdapterProbe {
        decode_probe(json.as_bytes(), b"", true).unwrap()
    }

    #[test]
    fn missing_python_is_a_spawn_unavailability() {
        let error = PythonReferenceAdapter::new("nxrt-python-does-not-exist-for-test")
            .probe()
            .unwrap_err();
        assert!(matches!(error, ReferenceAdapterError::Spawn { .. }));
    }

    #[test]
    fn probe_parses_python_without_onnx() {
        let probe = parse_probe(
            r#"{
                "status":"unavailable",
                "onnx_version":null,
                "latest_einsum_schema":null,
                "onnxruntime_status":"unavailable",
                "onnxruntime_version":null,
                "reason":"ONNX ReferenceEvaluator unavailable: No module named 'onnx'",
                "onnxruntime_reason":"ONNX Runtime unavailable: No module named 'onnxruntime'"
            }"#,
        );
        assert_eq!(probe.status, AdapterStatus::Unavailable);
        assert_eq!(probe.onnxruntime_status, AdapterStatus::Unavailable);
        assert!(probe.reason.unwrap().contains("No module named 'onnx'"));
    }

    #[test]
    fn probe_keeps_ort_available_without_reference_evaluator() {
        let probe = parse_probe(
            r#"{
                "status":"unavailable",
                "onnx_version":"1.24.0",
                "latest_einsum_schema":28,
                "onnxruntime_status":"available",
                "onnxruntime_version":"1.28.0",
                "reason":"ONNX ReferenceEvaluator unavailable: cannot import name 'ReferenceEvaluator'",
                "onnxruntime_reason":null
            }"#,
        );
        assert_eq!(probe.status, AdapterStatus::Unavailable);
        assert_eq!(probe.onnxruntime_status, AdapterStatus::Available);
        assert_eq!(probe.latest_einsum_schema, Some(28));
    }

    #[test]
    fn probe_parses_available_stale_onnx_schema() {
        let probe = parse_probe(
            r#"{
                "status":"available",
                "onnx_version":"1.22.0",
                "latest_einsum_schema":12,
                "onnxruntime_status":"unavailable",
                "onnxruntime_version":null,
                "reason":null,
                "onnxruntime_reason":"ONNX Runtime unavailable: No module named 'onnxruntime'"
            }"#,
        );
        assert_eq!(probe.status, AdapterStatus::Available);
        assert_eq!(probe.latest_einsum_schema, Some(12));
    }

    #[test]
    fn probe_parses_available_current_onnx_schema() {
        let probe = parse_probe(
            r#"{
                "status":"available",
                "onnx_version":"1.24.0",
                "latest_einsum_schema":28,
                "onnxruntime_status":"available",
                "onnxruntime_version":"1.28.0",
                "reason":null,
                "onnxruntime_reason":null
            }"#,
        );
        assert_eq!(probe.status, AdapterStatus::Available);
        assert_eq!(probe.latest_einsum_schema, Some(28));
        assert_eq!(probe.onnxruntime_status, AdapterStatus::Available);
    }

    #[test]
    fn probe_parses_ort_missing_library_and_version() {
        for reason in [
            "ONNX Runtime unavailable: libonnxruntime.so could not be opened",
            "ONNX Runtime unavailable: imported module has no non-empty __version__",
        ] {
            let json = format!(
                r#"{{
                    "status":"available",
                    "onnx_version":"1.24.0",
                    "latest_einsum_schema":28,
                    "onnxruntime_status":"unavailable",
                    "onnxruntime_version":null,
                    "reason":null,
                    "onnxruntime_reason":{reason:?}
                }}"#
            );
            let probe = parse_probe(&json);
            assert_eq!(probe.onnxruntime_status, AdapterStatus::Unavailable);
            assert_eq!(probe.onnxruntime_reason.as_deref(), Some(reason));
        }
    }

    #[test]
    fn probe_parses_reasoned_error_statuses() {
        let probe = parse_probe(
            r#"{
                "status":"error",
                "onnx_version":null,
                "latest_einsum_schema":null,
                "onnxruntime_status":"error",
                "onnxruntime_version":null,
                "reason":"ONNX probe failed unexpectedly",
                "onnxruntime_reason":"ONNX Runtime probe failed unexpectedly"
            }"#,
        );
        assert_eq!(probe.status, AdapterStatus::Error);
        assert_eq!(probe.onnxruntime_status, AdapterStatus::Error);
    }

    #[test]
    fn probe_rejects_unavailable_status_without_reason() {
        let error = decode_probe(
            br#"{
                "status":"unavailable",
                "onnx_version":null,
                "latest_einsum_schema":null,
                "onnxruntime_status":"unavailable",
                "onnxruntime_version":null,
                "reason":null,
                "onnxruntime_reason":"missing"
            }"#,
            b"",
            true,
        )
        .unwrap_err();
        assert!(matches!(error, ReferenceAdapterError::Protocol(_)));
    }
}
