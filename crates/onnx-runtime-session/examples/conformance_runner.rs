use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};

use onnx_runtime_ir::DataType;
use onnx_runtime_session::{InferenceSession, SessionError, Tensor};

const MAGIC: &[u8; 8] = b"NXRTCF01";

fn main() {
    let args: Vec<_> = std::env::args_os().collect();
    if args.len() != 3 {
        eprintln!("usage: conformance_runner MODEL.onnx CASE_DIR");
        std::process::exit(64);
    }

    let model = PathBuf::from(&args[1]);
    let case_dir = PathBuf::from(&args[2]);
    match run(&model, &case_dir) {
        Ok(count) => println!("OK\t{count} outputs"),
        Err(RunnerError::Session(err @ SessionError::UnsupportedOp { .. })) => {
            println!("UNSUPPORTED_OP\t{err}");
            std::process::exit(2);
        }
        Err(err) => {
            println!("ERROR\t{err}");
            std::process::exit(3);
        }
    }
}

fn run(model: &Path, case_dir: &Path) -> Result<usize, RunnerError> {
    let mut session = InferenceSession::load(model).map_err(RunnerError::Session)?;
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

    let outputs = session.run(&bindings).map_err(RunnerError::Session)?;
    for (index, tensor) in outputs.iter().enumerate() {
        let path = case_dir.join(format!("output_{index}.nxrt"));
        write_tensor(&path, tensor).map_err(|source| RunnerError::Output {
            index,
            path,
            source,
        })?;
    }
    Ok(outputs.len())
}

#[derive(Debug)]
enum RunnerError {
    Session(SessionError),
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

impl std::fmt::Display for RunnerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Session(err) => write!(f, "{err}"),
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

fn write_tensor(path: &Path, tensor: &Tensor) -> io::Result<()> {
    let payload = tensor.as_bytes();
    let mut bytes =
        Vec::with_capacity(MAGIC.len() + 1 + 4 + tensor.shape.len() * 8 + 8 + payload.len());
    bytes.extend_from_slice(MAGIC);
    bytes.push(dtype_code(tensor.dtype)?);
    bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
    for &dim in &tensor.shape {
        bytes.extend_from_slice(&(dim as u64).to_le_bytes());
    }
    bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    bytes.extend_from_slice(payload);
    fs::write(path, bytes)
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
    match code {
        1 => Ok(DataType::Float32),
        2 => Ok(DataType::Uint8),
        3 => Ok(DataType::Int8),
        4 => Ok(DataType::Uint16),
        5 => Ok(DataType::Int16),
        6 => Ok(DataType::Int32),
        7 => Ok(DataType::Int64),
        9 => Ok(DataType::Bool),
        10 => Ok(DataType::Float16),
        11 => Ok(DataType::Float64),
        12 => Ok(DataType::Uint32),
        13 => Ok(DataType::Uint64),
        16 => Ok(DataType::BFloat16),
        _ => Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("unsupported tensor dtype code {code}"),
        )),
    }
}

fn dtype_code(dtype: DataType) -> io::Result<u8> {
    match dtype {
        DataType::String
        | DataType::Float8E4M3FN
        | DataType::Float8E4M3FNUZ
        | DataType::Float8E5M2
        | DataType::Float8E5M2FNUZ
        | DataType::Uint4
        | DataType::Int4
        | DataType::Float4E2M1 => Err(io::Error::new(
            ErrorKind::InvalidData,
            format!("interchange format does not support {dtype:?}"),
        )),
        other => Ok(other as u8),
    }
}
