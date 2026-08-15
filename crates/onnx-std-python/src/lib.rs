//! Python bindings for `onnx-std` model serialization.

use std::path::{Path, PathBuf};

use ::onnx_std as onnx;
use onnx::{Error as OnnxError, Model as OnnxModel};
use onnx_runtime_loader::{LoaderError, ModelMetadata, load_model_bytes_with_weights};
use pyo3::exceptions::{
    PyAttributeError, PyFileNotFoundError, PyIsADirectoryError, PyOSError, PyPermissionError,
    PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// An opaque, owned ONNX model handle.
#[pyclass(module = "onnx_std", name = "Model")]
struct Model {
    inner: OnnxModel,
}

#[pymethods]
impl Model {
    fn __repr__(&self) -> String {
        format!(
            "Model(graph_name={:?}, nodes={})",
            self.inner.metadata.graph_name,
            self.inner.graph.num_nodes()
        )
    }
}

/// Load an ONNX protobuf model from a path or serialized bytes.
#[pyfunction]
#[pyo3(signature = (path_or_bytes))]
fn load_model(path_or_bytes: &Bound<'_, PyAny>) -> PyResult<Model> {
    if let Ok(bytes) = path_or_bytes.cast::<PyBytes>() {
        let bytes = bytes.as_bytes();
        return load_model_bytes(bytes)
            .map(|inner| Model { inner })
            .map_err(|error| {
                map_loader_error(
                    "load ONNX model",
                    &format!("{} bytes", bytes.len()),
                    error,
                    "Pass bytes containing a complete serialized ONNX ModelProto. For models \
                     using external data, pass the model path instead so weight files resolve \
                     relative to it.",
                )
            });
    }

    let path = path_arg(path_or_bytes, "load_model(path_or_bytes)", true)?;

    onnx::load_model(&path)
        .map(|inner| Model { inner })
        .map_err(|error| {
            map_onnx_error(
                "load ONNX model",
                &format!("{path:?}"),
                error,
                "Pass a path to an existing, readable ONNX protobuf model. If it uses external \
                 data, keep those files beside it at the paths recorded in the model.",
            )
        })
}

/// Save an ONNX model as binary protobuf.
#[pyfunction]
#[pyo3(signature = (model, path))]
fn save_model(model: PyRef<'_, Model>, path: &Bound<'_, PyAny>) -> PyResult<()> {
    let path = path_arg(path, "save_model(model, path)", false)?;
    onnx::save_model(&model.inner, &path).map_err(|error| {
        map_onnx_error(
            "save ONNX model",
            &format!("{path:?}"),
            error,
            "Choose a writable file path whose parent directory exists. If the model uses \
             external weights, keep the original model handle alive so its weight data remains \
             available.",
        )
    })
}

/// Serialize a model using the human-readable onnx-std text DSL.
#[pyfunction]
#[pyo3(signature = (model))]
fn to_text(model: PyRef<'_, Model>) -> String {
    onnx::to_text(&model.inner)
}

/// Parse a model from the human-readable onnx-std text DSL.
#[pyfunction]
#[pyo3(signature = (source))]
fn from_text(source: &str) -> PyResult<Model> {
    onnx::from_text(source)
        .map(|inner| Model { inner })
        .map_err(|error| {
            codec_error(
                "parse",
                "onnx-std text",
                error,
                "Pass text produced by onnx_std.to_text(model), and correct the reported line.",
            )
        })
}

/// Serialize a model using canonical ONNX protobuf JSON mapping.
#[pyfunction]
#[pyo3(signature = (model))]
fn to_json(model: PyRef<'_, Model>) -> PyResult<String> {
    onnx::json::to_json(&model.inner).map_err(|error| {
        codec_error(
            "serialize",
            "ONNX JSON",
            error,
            "Reload the model from its original path if external initializer data is no longer \
             available.",
        )
    })
}

/// Parse a model using canonical ONNX protobuf JSON mapping.
#[pyfunction]
#[pyo3(signature = (source))]
fn from_json(source: &str) -> PyResult<Model> {
    onnx::json::from_json(source)
        .map(|inner| Model { inner })
        .map_err(|error| {
            codec_error(
                "parse",
                "ONNX JSON",
                error,
                "Pass a valid ONNX protobuf-JSON document, such as output from \
                 onnx_std.to_json(model).",
            )
        })
}

/// Serialize a model using protobuf TextFormat.
#[pyfunction]
#[pyo3(signature = (model))]
fn to_textproto(model: PyRef<'_, Model>) -> PyResult<String> {
    onnx::textproto::to_textproto(&model.inner).map_err(|error| {
        codec_error(
            "serialize",
            "ONNX TextProto",
            error,
            "Reload the model from its original path if external initializer data is no longer \
             available.",
        )
    })
}

/// Parse a model using protobuf TextFormat.
#[pyfunction]
#[pyo3(signature = (source))]
fn from_textproto(source: &str) -> PyResult<Model> {
    onnx::textproto::from_textproto(source)
        .map(|inner| Model { inner })
        .map_err(|error| {
            codec_error(
                "parse",
                "ONNX TextProto",
                error,
                "Pass valid protobuf TextFormat, such as output from \
                 onnx_std.to_textproto(model).",
            )
        })
}

fn path_arg(
    value: &Bound<'_, PyAny>,
    call: &'static str,
    bytes_allowed: bool,
) -> PyResult<PathBuf> {
    if value.cast::<PyString>().is_err() {
        match value.getattr("__fspath__") {
            Ok(_) => {}
            Err(error) if error.is_instance_of::<PyAttributeError>(value.py()) => {
                let accepted = if bytes_allowed {
                    "a filesystem path (str or os.PathLike), or bytes containing a serialized \
                     ONNX ModelProto"
                } else {
                    "a filesystem path (str or os.PathLike)"
                };
                return Err(PyTypeError::new_err(format!(
                    "{call} expected {accepted}; got {}. Pass one of the accepted types.",
                    value.get_type().name()?
                )));
            }
            Err(error) => return Err(error),
        }
    }

    // PyO3's PathBuf extractor calls PyOS_FSPath and losslessly converts Python's
    // filesystem encoding to OsString, preserving non-UTF-8 Unix paths.
    let path = value.extract::<PathBuf>()?;
    if path.as_os_str().is_empty() {
        return Err(PyValueError::new_err(format!(
            "{call} received an empty path. Pass a non-empty path to an ONNX model file."
        )));
    }
    Ok(path)
}

fn load_model_bytes(bytes: &[u8]) -> Result<OnnxModel, LoaderError> {
    let proto = onnx_runtime_loader::proto::decode_model(bytes)?;
    let metadata = metadata_from_proto(&proto);
    let (graph, weights) = load_model_bytes_with_weights(bytes, Path::new("."))?;
    let mut model = OnnxModel::with_metadata(graph, metadata);
    model.set_weights(weights);
    Ok(model)
}

fn metadata_from_proto(proto: &onnx_runtime_loader::proto::ModelProto) -> ModelMetadata {
    ModelMetadata {
        ir_version: proto.ir_version,
        producer_name: proto.producer_name.clone(),
        producer_version: proto.producer_version.clone(),
        domain: proto.domain.clone(),
        model_version: proto.model_version,
        doc_string: (!proto.doc_string.is_empty()).then(|| proto.doc_string.clone()),
        graph_name: proto
            .graph
            .as_ref()
            .map(|graph| graph.name.clone())
            .unwrap_or_default(),
        metadata_props: proto
            .metadata_props
            .iter()
            .map(|entry| (entry.key.clone(), entry.value.clone()))
            .collect(),
    }
}

fn map_onnx_error(operation: &str, source: &str, error: OnnxError, fix: &str) -> PyErr {
    match error {
        OnnxError::Read {
            path,
            source: io_error,
        } => map_io_error(operation, &format!("{path:?}"), io_error, fix),
        OnnxError::Loader(loader_error) => map_loader_error(operation, source, loader_error, fix),
        parse_error => PyValueError::new_err(format!(
            "failed to {operation} from {source}: {parse_error}. {fix}"
        )),
    }
}

fn map_loader_error(operation: &str, source: &str, error: LoaderError, fix: &str) -> PyErr {
    match error {
        LoaderError::Io {
            path,
            source: io_error,
        } => map_io_error(operation, &format!("{path:?}"), io_error, fix),
        LoaderError::ExternalDataNotFound { path } => PyFileNotFoundError::new_err(format!(
            "failed to {operation} from {source}: external initializer data file {path:?} was not \
             found. Keep the external-data file at the location recorded by the ONNX model, or \
             load the model from its filesystem path so relative references resolve correctly."
        )),
        other => PyValueError::new_err(format!(
            "failed to {operation} from {source}: {other}. {fix}"
        )),
    }
}

fn map_io_error(operation: &str, path: &str, error: std::io::Error, fix: &str) -> PyErr {
    let message = format!("failed to {operation} at {path}: {error}. {fix}");
    match io_exception_kind(error.kind()) {
        IoExceptionKind::FileNotFound => PyFileNotFoundError::new_err(message),
        IoExceptionKind::Permission => PyPermissionError::new_err(message),
        IoExceptionKind::IsADirectory => PyIsADirectoryError::new_err(message),
        IoExceptionKind::Os => PyOSError::new_err(message),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IoExceptionKind {
    FileNotFound,
    Permission,
    IsADirectory,
    Os,
}

fn io_exception_kind(kind: std::io::ErrorKind) -> IoExceptionKind {
    match kind {
        std::io::ErrorKind::NotFound => IoExceptionKind::FileNotFound,
        std::io::ErrorKind::PermissionDenied => IoExceptionKind::Permission,
        std::io::ErrorKind::IsADirectory => IoExceptionKind::IsADirectory,
        _ => IoExceptionKind::Os,
    }
}

fn codec_error(operation: &str, format: &str, error: OnnxError, fix: &str) -> PyErr {
    PyValueError::new_err(format!(
        "failed to {operation} model as {format}: {error}. {fix}"
    ))
}

/// The `onnx_std` Python module.
#[pymodule]
fn onnx_std(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add(
        "__doc__",
        PyString::new(
            m.py(),
            "onnx_std — Python bindings for onnx-std model loading, saving, and string codecs.",
        ),
    )?;
    m.add_class::<Model>()?;
    m.add_function(wrap_pyfunction!(load_model, m)?)?;
    m.add_function(wrap_pyfunction!(save_model, m)?)?;
    m.add_function(wrap_pyfunction!(to_text, m)?)?;
    m.add_function(wrap_pyfunction!(from_text, m)?)?;
    m.add_function(wrap_pyfunction!(to_json, m)?)?;
    m.add_function(wrap_pyfunction!(from_json, m)?)?;
    m.add_function(wrap_pyfunction!(to_textproto, m)?)?;
    m.add_function(wrap_pyfunction!(from_textproto, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx::ir::{DataType, Graph, Node, NodeId, static_shape};

    fn tiny_model() -> OnnxModel {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), 21);
        let x = graph.create_named_value("X", DataType::Float32, static_shape([1]));
        let y = graph.create_named_value("Y", DataType::Float32, static_shape([1]));
        let z = graph.create_named_value("Z", DataType::Float32, static_shape([1]));
        graph.add_input(x);
        graph.add_input(y);
        graph.insert_node(Node::new(NodeId(0), "Add", vec![Some(x), Some(y)], vec![z]));
        graph.add_output(z);
        OnnxModel::new(graph)
    }

    #[test]
    fn all_string_codecs_round_trip_a_tiny_model() {
        let model = tiny_model();

        let text = onnx::to_text(&model);
        let from_text = onnx::from_text(&text).expect("text round-trip");
        assert_eq!(from_text.graph.num_nodes(), 1);

        let json = onnx::json::to_json(&model).expect("JSON serialization");
        let from_json = onnx::json::from_json(&json).expect("JSON round-trip");
        assert_eq!(from_json.graph.num_nodes(), 1);

        let textproto = onnx::textproto::to_textproto(&model).expect("TextProto serialization");
        let from_textproto =
            onnx::textproto::from_textproto(&textproto).expect("TextProto round-trip");
        assert_eq!(from_textproto.graph.num_nodes(), 1);
    }

    #[test]
    fn binary_bytes_load_preserves_model_metadata() {
        let mut model = tiny_model();
        model.metadata.producer_name = "onnx-std-python-test".to_string();
        let path = std::env::current_dir()
            .expect("current directory")
            .join("target")
            .join(format!("onnx_std_python_{}.onnx", std::process::id()));
        std::fs::create_dir_all(path.parent().expect("test output parent"))
            .expect("create test output directory");
        onnx::save_model(&model, &path).expect("save test model");
        let bytes = std::fs::read(&path).expect("read test model");
        let loaded = load_model_bytes(&bytes).expect("load model bytes");
        assert_eq!(loaded.graph.num_nodes(), 1);
        assert_eq!(loaded.metadata.producer_name, "onnx-std-python-test");
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn io_error_kinds_select_specific_python_exceptions() {
        assert_eq!(
            io_exception_kind(std::io::ErrorKind::NotFound),
            IoExceptionKind::FileNotFound
        );
        assert_eq!(
            io_exception_kind(std::io::ErrorKind::PermissionDenied),
            IoExceptionKind::Permission
        );
        assert_eq!(
            io_exception_kind(std::io::ErrorKind::IsADirectory),
            IoExceptionKind::IsADirectory
        );
        assert_eq!(
            io_exception_kind(std::io::ErrorKind::InvalidData),
            IoExceptionKind::Os
        );
    }
}
