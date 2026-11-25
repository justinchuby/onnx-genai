// Preserve the session crate's structured error type at this CLI boundary.
#![allow(clippy::result_large_err)]

use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use onnx_runtime_ir::{DataType, DeviceId, DeviceType};
use onnx_runtime_loader::{proto, read_model_binary};
use onnx_runtime_session::{
    DevicePreference, InferenceSession, SessionError, SessionOutput, Tensor,
};

const MAGIC: &[u8; 8] = b"NXRTCF01";
const REQUIRE_CUDA: &str = "ONNX_GENAI_REQUIRE_CUDA";

fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    let (cuda, model, case_dir) = match args.as_slice() {
        [_, model, case_dir] if model != "--cuda" => (false, model, case_dir),
        [_, flag, model, case_dir] if flag == "--cuda" => (true, model, case_dir),
        _ => {
            eprintln!("usage: conformance_runner [--cuda] MODEL.onnx CASE_DIR");
            std::process::exit(64);
        }
    };

    // The native runtime freezes its environment on first use. Configure the
    // strict policy at process startup, without mutating this process's env.
    if cuda && std::env::var_os(REQUIRE_CUDA).as_deref() != Some(std::ffi::OsStr::new("1")) {
        match std::env::current_exe().and_then(|exe| {
            std::process::Command::new(exe)
                .args(&args[1..])
                .env(REQUIRE_CUDA, "1")
                .status()
        }) {
            Ok(status) => std::process::exit(status.code().unwrap_or(3)),
            Err(err) => {
                println!("ERROR\tcould not start strict CUDA runner: {err}");
                std::process::exit(3);
            }
        }
    }

    match run(Path::new(model), Path::new(case_dir), cuda) {
        Ok(count) if cuda => println!("OK\t{count} outputs on CUDA:0"),
        Ok(count) => println!("OK\t{count} outputs"),
        Err(RunnerError::Session(err @ SessionError::UnsupportedOp { .. })) => {
            println!("UNSUPPORTED_OP\t{err}");
            std::process::exit(2);
        }
        Err(RunnerError::Session(err @ SessionError::HeterogeneousPlacementRequired { .. }))
            if cuda =>
        {
            println!("UNSUPPORTED_OP\t{err}");
            std::process::exit(2);
        }
        Err(err) if cuda && err.is_unsupported_fixture() => {
            println!("UNSUPPORTED_FIXTURE\t{err}");
            std::process::exit(4);
        }
        Err(RunnerError::Skipped(reason)) => {
            println!("SKIP\t{reason}");
            std::process::exit(5);
        }
        Err(err) => {
            println!("ERROR\t{err}");
            std::process::exit(3);
        }
    }
}

fn run(model: &Path, case_dir: &Path, cuda: bool) -> Result<usize, RunnerError> {
    let mut session = if cuda {
        check_fixture_io(model)?;
        InferenceSession::builder()
            .model(model)
            .device(DevicePreference::Explicit {
                device_type: DeviceType::Cuda,
                index: 0,
            })
            .build()
    } else {
        InferenceSession::load(model)
    }
    .map_err(RunnerError::Session)?;
    if cuda
        && (session.device_id() != DeviceId::cuda(0)
            || session.execution_provider_fallback_report().is_some()
            || session.heterogeneous_placement_report().is_some())
    {
        return Err(RunnerError::Placement(session.device_id()));
    }
    if cuda {
        if let Some(node) = session.host_graph_op() {
            return Err(RunnerError::Skipped(format!(
                "node {:?} ({}) uses host graph dispatch, not a CUDA kernel",
                node.name, node.op_type
            )));
        }
        if session.graph().nodes.is_empty() {
            return Err(RunnerError::Skipped(
                "no executable graph nodes to verify on CUDA".into(),
            ));
        }
        for meta in session.outputs() {
            dtype_code(meta.dtype, true).map_err(|err| {
                RunnerError::UnsupportedFixture(format!("output {:?}: {err}", meta.name))
            })?;
        }
    }
    let input_meta = session.inputs().to_vec();
    let mut inputs = Vec::with_capacity(input_meta.len());
    for (index, meta) in input_meta.iter().enumerate() {
        let path = case_dir.join(format!("input_{index}.nxrt"));
        let tensor = read_tensor(&path).map_err(|source| RunnerError::Input {
            name: meta.name.clone(),
            path,
            source,
        })?;
        inputs.push((meta.name.clone(), tensor));
    }
    let bindings: Vec<_> = inputs
        .iter()
        .map(|(name, tensor)| (name.as_str(), tensor))
        .collect();

    let outputs = if cuda {
        session
            .run_outputs(&bindings)
            .map_err(RunnerError::Session)?
    } else {
        session
            .run(&bindings)
            .map_err(RunnerError::Session)?
            .into_iter()
            .map(SessionOutput::Tensor)
            .collect()
    };
    for (index, tensor) in outputs.iter().enumerate() {
        let tensor = tensor.as_tensor().ok_or_else(|| {
            RunnerError::UnsupportedFixture(format!(
                "output #{index} is a sequence; NXRTCF01 requires a dense tensor"
            ))
        })?;
        let path = case_dir.join(format!("output_{index}.nxrt"));
        write_tensor(&path, tensor, cuda).map_err(|source| RunnerError::Output {
            index,
            path,
            source,
        })?;
    }
    Ok(outputs.len())
}

fn check_fixture_io(path: &Path) -> Result<(), RunnerError> {
    use proto::onnx::type_proto::Value;

    let bytes = read_model_binary(path).map_err(SessionError::from);
    let model = bytes
        .and_then(|bytes| proto::decode_model(&bytes).map_err(SessionError::from))
        .map_err(RunnerError::Session)?;
    let graph = model.graph.ok_or_else(|| {
        RunnerError::Session(SessionError::Internal(format!(
            "{} contains no ONNX graph; supply a valid ONNX model",
            path.display()
        )))
    })?;
    for (role, values) in [("input", &graph.input), ("output", &graph.output)] {
        for value in values {
            let kind = value.r#type.as_ref().and_then(|tp| tp.value.as_ref());
            let reason = match kind {
                Some(Value::TensorType(tensor)) => {
                    dtype_from_code(u8::try_from(tensor.elem_type).unwrap_or_default())
                        .err()
                        .map(|err| format!("ONNX dtype {}: {err}", tensor.elem_type))
                }
                _ => Some("NXRTCF01 requires a declared dense tensor type".into()),
            };
            if let Some(reason) = reason {
                return Err(RunnerError::UnsupportedFixture(format!(
                    "{role} {:?}: {reason}",
                    value.name
                )));
            }
        }
    }
    Ok(())
}

#[derive(Debug)]
enum RunnerError {
    Session(SessionError),
    Placement(DeviceId),
    UnsupportedFixture(String),
    Skipped(String),
    Input {
        name: String,
        path: PathBuf,
        source: io::Error,
    },
    Output {
        index: usize,
        path: PathBuf,
        source: io::Error,
    },
}

impl RunnerError {
    fn is_unsupported_fixture(&self) -> bool {
        matches!(self, Self::UnsupportedFixture(_))
            || matches!(self, Self::Input { source, .. } | Self::Output { source, .. }
                if source.kind() == ErrorKind::Unsupported)
    }
}

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session(err) => write!(f, "{err}"),
            Self::UnsupportedFixture(reason) | Self::Skipped(reason) => f.write_str(reason),
            Self::Placement(device) => write!(
                f,
                "strict CUDA:0 requested but session reported {device:?} or mixed/fallback placement; no inference was run"
            ),
            Self::Input { name, path, source } => write!(
                f,
                "failed to read input {name:?} from {}: {source}",
                path.display()
            ),
            Self::Output {
                index,
                path,
                source,
            } => write!(
                f,
                "failed to write output #{index} to {}: {source}",
                path.display()
            ),
        }
    }
}

fn read_tensor(path: &Path) -> io::Result<Tensor> {
    let bytes = fs::read(path)?;
    let mut cursor = 0;
    if take(&bytes, &mut cursor, MAGIC.len())? != MAGIC {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            "invalid tensor magic; expected NXRTCF01",
        ));
    }
    let dtype = dtype_from_code(take(&bytes, &mut cursor, 1)?[0])?;
    let rank = u32::from_le_bytes(
        take(&bytes, &mut cursor, 4)?
            .try_into()
            .expect("fixed length"),
    ) as usize;
    let mut shape = Vec::with_capacity(rank);
    for _ in 0..rank {
        let dim = u64::from_le_bytes(
            take(&bytes, &mut cursor, 8)?
                .try_into()
                .expect("fixed length"),
        );
        shape.push(usize::try_from(dim).map_err(|_| {
            io::Error::new(
                ErrorKind::InvalidData,
                format!("dimension {dim} exceeds usize"),
            )
        })?);
    }
    let payload_len = u64::from_le_bytes(
        take(&bytes, &mut cursor, 8)?
            .try_into()
            .expect("fixed length"),
    );
    let payload_len = usize::try_from(payload_len)
        .map_err(|_| io::Error::new(ErrorKind::InvalidData, "payload length exceeds usize"))?;
    let payload = take(&bytes, &mut cursor, payload_len)?;
    if cursor != bytes.len() {
        return Err(io::Error::new(
            ErrorKind::InvalidData,
            format!(
                "{} trailing bytes after tensor payload",
                bytes.len() - cursor
            ),
        ));
    }
    Tensor::from_raw(dtype, shape, payload)
        .map_err(|err| io::Error::new(ErrorKind::InvalidData, err.to_string()))
}

fn write_tensor(path: &Path, tensor: &Tensor, strict: bool) -> io::Result<()> {
    fs::write(path, encode_tensor(tensor, strict)?)
}

fn encode_tensor(tensor: &Tensor, strict: bool) -> io::Result<Vec<u8>> {
    let dtype = dtype_code(tensor.dtype, strict)?;
    let payload = tensor.as_bytes();
    let mut bytes =
        Vec::with_capacity(MAGIC.len() + 1 + 4 + tensor.shape.len() * 8 + 8 + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.push(dtype);
    bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
    for &dim in &tensor.shape {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;
    use proto::onnx::{GraphProto, ModelProto, TypeProto, ValueInfoProto, type_proto};

    #[test]
    fn declared_string_output_is_an_unsupported_fixture_before_cuda_initializes() {
        let model = ModelProto {
            graph: Some(GraphProto {
                output: vec![ValueInfoProto {
                    name: "text".into(),
                    r#type: Some(TypeProto {
                        value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                            elem_type: DataType::String.to_onnx(),
                            ..Default::default()
                        })),
                        ..Default::default()
                    }),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        };
        let root = std::env::temp_dir();
        let path = root.join(format!(
            "nxrt-conformance-string-{}.onnx",
            std::process::id()
        ));
        fs::write(&path, model.encode_to_vec()).unwrap();
        let error = run(&path, &root, true).unwrap_err();
        fs::remove_file(path).unwrap();
        assert!(error.is_unsupported_fixture(), "{error}");
        assert!(error.to_string().contains("\"text\""));
    }

    #[test]
    fn legacy_cpu_complex_output_keeps_its_wire_encoding() {
        let payload = [0, 0, 128, 63, 0, 0, 0, 64];
        let tensor = Tensor::from_raw(DataType::Complex64, vec![1], &payload).unwrap();
        let mut expected = b"NXRTCF01\x0e".to_vec();
        expected.extend(1u32.to_le_bytes());
        expected.extend(1u64.to_le_bytes());
        expected.extend(8u64.to_le_bytes());
        expected.extend(payload);
        assert_eq!(encode_tensor(&tensor, false).unwrap(), expected);
        assert_eq!(
            encode_tensor(&tensor, true).unwrap_err().kind(),
            ErrorKind::Unsupported
        );
    }
}

fn take<'a>(bytes: &'a [u8], cursor: &mut usize, len: usize) -> io::Result<&'a [u8]> {
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| io::Error::new(ErrorKind::InvalidData, "tensor length overflow"))?;
    let value = bytes.get(*cursor..end).ok_or_else(|| {
        io::Error::new(
            ErrorKind::UnexpectedEof,
            format!(
                "tensor ended at byte {}, needed {len} more bytes",
                bytes.len()
            ),
        )
    })?;
    *cursor = end;
    Ok(value)
}

fn dtype_from_code(code: u8) -> io::Result<DataType> {
    let dtype = DataType::from_onnx(i32::from(code)).ok_or_else(|| {
        io::Error::new(
            ErrorKind::InvalidData,
            format!("invalid NXRTCF01 tensor dtype code {code}"),
        )
    })?;
    dtype_code(dtype, true)?;
    Ok(dtype)
}

fn dtype_code(dtype: DataType, strict: bool) -> io::Result<u8> {
    match dtype {
        // Legacy CPU runner emitted these output codes but did not accept them
        // on input. Keep that asymmetric wire behavior.
        DataType::Complex64 | DataType::Complex128 if !strict => Ok(dtype as u8),
        DataType::String
        | DataType::Complex64
        | DataType::Complex128
        | DataType::Float8E4M3FN
        | DataType::Float8E4M3FNUZ
        | DataType::Float8E5M2
        | DataType::Float8E5M2FNUZ
        | DataType::Uint4
        | DataType::Int4
        | DataType::Float4E2M1
        | DataType::Float8E8M0
        | DataType::Uint2
        | DataType::Int2
        | DataType::Float6E2M3
        | DataType::Float6E3M2 => Err(io::Error::new(
            ErrorKind::Unsupported,
            format!(
                "NXRTCF01 cannot encode {dtype:?}; choose a fixture with supported dense tensor I/O"
            ),
        )),
        other => Ok(other as u8),
    }
}
