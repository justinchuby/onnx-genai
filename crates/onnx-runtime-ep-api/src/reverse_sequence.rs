//! Shared ReverseSequence schema and dense permutation planning.

use onnx_runtime_ir::{DataType, Node, Shape};

use crate::{EpError, Result, TensorMut, TensorView};

pub const VERSIONS: [u64; 2] = [10, 28];
pub const DTYPES: &[DataType] = &[
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::Bool,
    DataType::Complex64,
    DataType::Complex128,
    DataType::BFloat16,
];
pub const INPUT_DTYPES: &[(usize, &[DataType])] = &[(1, &[DataType::Int64])];

fn fail(reason: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("ReverseSequence: {reason}"))
}

fn validate_dtypes(version: u64, data: DataType, lengths: DataType) -> Result<()> {
    if lengths != DataType::Int64 {
        return Err(fail(format!(
            "`sequence_lens` must be Int64, got {lengths:?}"
        )));
    }
    if data == DataType::String {
        return Err(fail(
            "String input requires mixed managed-string/Int64 transport, which native execution does not support",
        ));
    }
    if !DTYPES.contains(&data) {
        return Err(fail(format!(
            "input dtype {data:?} is not supported by native execution (opset {version})"
        )));
    }
    Ok(())
}

pub fn unsupported_reason(
    node: &Node,
    version: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Option<String> {
    if node.op_type != "ReverseSequence" || !node.is_default_domain() {
        return None;
    }
    let check = || -> Result<()> {
        let axes = Axes::from_node(node)?;
        let &[data, lengths] = dtypes else {
            return Err(fail(
                "expected dtype metadata for input and `sequence_lens`",
            ));
        };
        validate_dtypes(version, data, lengths)?;
        if let [data, lengths] = shapes {
            axes.validate_shape(
                data.len(),
                lengths.len(),
                data.get(axes.batch).and_then(|d| d.as_static()),
                lengths.first().and_then(|d| d.as_static()),
            )?;
        }
        Ok(())
    };
    check().err().map(|error| error.to_string())
}

#[derive(Clone, Copy)]
pub struct Axes {
    time: usize,
    batch: usize,
}

impl Axes {
    pub fn from_node(node: &Node) -> Result<Self> {
        if !matches!(node.inputs.as_slice(), [Some(_), Some(_)])
            || !matches!(node.outputs.as_slice(), [Some(_)])
        {
            return Err(fail("expected two present inputs and one present output"));
        }
        let axis = |name, default| match node.attr(name) {
            None => Ok(default),
            Some(attr) => match attr.as_int() {
                Some(0) => Ok(0),
                Some(1) => Ok(1),
                _ => Err(fail(format!(
                    "`{name}` must be integer 0 or 1, got {attr:?}"
                ))),
            },
        };
        let time = axis("time_axis", 0)?;
        let batch = axis("batch_axis", 1)?;
        if time == batch {
            return Err(fail("`time_axis` and `batch_axis` must be distinct"));
        }
        Ok(Self { time, batch })
    }

    fn validate_shape(
        self,
        data_rank: usize,
        lengths_rank: usize,
        batch: Option<usize>,
        lengths: Option<usize>,
    ) -> Result<()> {
        if data_rank < 2 {
            return Err(fail(format!(
                "input rank must be at least 2, got {data_rank}"
            )));
        }
        if lengths_rank != 1 {
            return Err(fail(format!(
                "`sequence_lens` rank must be 1, got {lengths_rank}"
            )));
        }
        if let (Some(batch), Some(lengths)) = (batch, lengths)
            && lengths != batch
        {
            return Err(fail(format!(
                "`sequence_lens` must have one length per batch (expected {batch}, got {lengths})"
            )));
        }
        Ok(())
    }

    pub fn plan(
        self,
        version: u64,
        input: &TensorView<'_>,
        lengths: &TensorView<'_>,
        output: &TensorMut<'_>,
    ) -> Result<Permutation> {
        validate_dtypes(version, input.dtype, lengths.dtype)?;
        self.validate_shape(
            input.shape.len(),
            lengths.shape.len(),
            input.shape.get(self.batch).copied(),
            lengths.shape.first().copied(),
        )?;
        if input.shape != output.shape || input.dtype != output.dtype {
            return Err(fail(format!(
                "output must match input shape {:?} and dtype {:?}, got {:?} {:?}",
                input.shape, input.dtype, output.shape, output.dtype
            )));
        }
        let batch = input.shape[self.batch];
        let time = input.shape[self.time];
        let lengths_bytes = batch
            .checked_mul(size_of::<i64>())
            .ok_or_else(|| fail("`sequence_lens` byte size exceeds the address space"))?;
        let width = input.dtype.byte_size();
        let bytes = if input.shape.contains(&0) {
            0
        } else {
            input
                .shape
                .iter()
                .try_fold(width, |size, dim| size.checked_mul(*dim))
                .ok_or_else(|| fail("input byte size exceeds the address space"))?
        };
        let block_bytes = if bytes == 0 { 0 } else { bytes / batch / time };
        Ok(Permutation {
            batch,
            time,
            batch_first: self.batch == 0,
            block_bytes,
            bytes,
            lengths_bytes,
        })
    }
}

pub struct Permutation {
    pub batch: usize,
    pub time: usize,
    pub batch_first: bool,
    pub block_bytes: usize,
    pub bytes: usize,
    pub lengths_bytes: usize,
}

impl Permutation {
    pub fn validate_lengths(&self, lengths: impl Iterator<Item = i64>) -> Result<()> {
        for (batch, raw) in lengths.enumerate() {
            if !matches!(usize::try_from(raw), Ok(length) if length <= self.time) {
                return Err(fail(format!(
                    "`sequence_lens`[{batch}] = {raw} is outside 0..={}",
                    self.time
                )));
            }
        }
        Ok(())
    }
}
