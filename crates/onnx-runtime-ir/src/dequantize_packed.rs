//! Shared metadata contract for ONNX DequantizePacked-29.

use crate::{Attribute, DataType, Node, Shape};

/// The union of input and output types advertised by plugin EPs.
pub const DTYPES: &[DataType] = &[
    DataType::Uint8,
    DataType::Float32,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float64,
];
pub const INPUT_DTYPES: &[(usize, &[DataType])] = &[(0, &[DataType::Uint8])];
pub const OUTPUT_DTYPES: &[(usize, &[DataType])] = &[(
    0,
    &[
        DataType::Float32,
        DataType::Float16,
        DataType::BFloat16,
        DataType::Float64,
    ],
)];

/// Check the metadata and any statically known facts during native EP placement.
pub fn validate_claim(node: &Node, shapes: &[Shape], dtypes: &[DataType]) -> Result<(), String> {
    if node.inputs.len() != 1
        || node.inputs[0].is_none()
        || node.outputs.len() != 1
        || node.outputs[0].is_none()
    {
        return Err("expected exactly one packed input x and one decoded output y".into());
    }
    let metadata = Metadata::from_node(node)?;
    if let Some(&dtype) = dtypes.first()
        && dtype != DataType::Undefined
    {
        metadata.validate_input_dtype(dtype)?;
    }
    if let Some(shape) = shapes.first() {
        metadata.validate_input_rank(shape.len())?;
        if let Some(bytes) = shape[0].as_static() {
            metadata.validate_packed_bytes(bytes)?;
        }
    }
    Ok(())
}

macro_rules! formats {
    ($($kind:ident = $tag:literal, $name:literal, $width:literal, $bytes:literal;)*) => {
        /// A fixed ONNX packed layout. The discriminant is the GGML tensor ID.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        #[repr(u8)]
        pub enum Format { $($kind = $tag,)* }

        impl Format {
            pub fn parse(name: &str) -> Option<Self> {
                match name { $($name => Some(Self::$kind),)* _ => None }
            }

            pub fn name(self) -> &'static str {
                match self { $(Self::$kind => $name,)* }
            }

            /// (Decoded elements, packed bytes) per block.
            pub fn block(self) -> (usize, usize) {
                match self { $(Self::$kind => ($width, $bytes),)* }
            }
        }
    };
}

formats! {
    Q4_0 = 2, "gguf:q4_0", 32, 18;
    Q4_1 = 3, "gguf:q4_1", 32, 20;
    Q5_0 = 6, "gguf:q5_0", 32, 22;
    Q5_1 = 7, "gguf:q5_1", 32, 24;
    Q8_0 = 8, "gguf:q8_0", 32, 34;
    Q8_1 = 9, "gguf:q8_1", 32, 36;
    Q2K = 10, "gguf:q2_k", 256, 84;
    Q3K = 11, "gguf:q3_k", 256, 110;
    Q4K = 12, "gguf:q4_k", 256, 144;
    Q5K = 13, "gguf:q5_k", 256, 176;
    Q6K = 14, "gguf:q6_k", 256, 210;
    Q8K = 15, "gguf:q8_k", 256, 292;
    Iq2Xxs = 16, "gguf:iq2_xxs", 256, 66;
    Iq2Xs = 17, "gguf:iq2_xs", 256, 74;
    Iq3Xxs = 18, "gguf:iq3_xxs", 256, 98;
    Iq1S = 19, "gguf:iq1_s", 256, 50;
    Iq4Nl = 20, "gguf:iq4_nl", 32, 18;
    Iq3S = 21, "gguf:iq3_s", 256, 110;
    Iq2S = 22, "gguf:iq2_s", 256, 82;
    Iq4Xs = 23, "gguf:iq4_xs", 256, 136;
    Iq1M = 29, "gguf:iq1_m", 256, 56;
    Tq1_0 = 34, "gguf:tq1_0", 256, 54;
    Tq2_0 = 35, "gguf:tq2_0", 256, 66;
    Mxfp4 = 39, "gguf:mxfp4", 32, 17;
    Nvfp4 = 40, "gguf:nvfp4", 64, 36;
    Q1_0 = 41, "gguf:q1_0", 128, 18;
    Q2_0 = 42, "gguf:q2_0", 64, 18;
}

/// Validated shape, dtype, and storage sizes for a DequantizePacked node.
#[derive(Debug)]
pub struct Metadata {
    format: Format,
    shape: Vec<usize>,
    dtype: DataType,
    packed_bytes: usize,
    elements: usize,
    output_bytes: usize,
}

impl Metadata {
    pub fn from_node(node: &Node) -> Result<Self, String> {
        let name = node
            .attr("format")
            .and_then(Attribute::as_str)
            .ok_or("attribute 'format' must be a case-sensitive GGUF profile string")?;
        let format = Format::parse(name).ok_or_else(|| {
            format!("unknown format {name:?}; use a DequantizePacked-29 GGUF profile")
        })?;
        let dims = node
            .attr("shape")
            .and_then(Attribute::as_ints)
            .ok_or("attribute 'shape' must be a nonempty integer list")?;
        let shape: Vec<_> = dims
            .iter()
            .enumerate()
            .map(|(i, &dim)| {
                usize::try_from(dim).map_err(|_| {
                    format!("shape dimension {i} ({dim}) must be nonnegative and fit usize")
                })
            })
            .collect::<Result<_, _>>()?;
        let (width, bytes) = format.block();
        let (&last, _) = shape
            .split_last()
            .ok_or("attribute 'shape' cannot be empty")?;
        if last % width != 0 {
            return Err(format!(
                "innermost shape dimension {last} must be a multiple of {width} for {name}"
            ));
        }
        let dtype = match node.attr("output_dtype") {
            None => DataType::Float32,
            Some(raw) => raw
                .as_int()
                .and_then(|value| i32::try_from(value).ok())
                .and_then(DataType::from_onnx)
                .filter(|dtype| OUTPUT_DTYPES[0].1.contains(dtype))
                .ok_or_else(|| format!("attribute 'output_dtype' {raw:?} must be FLOAT, DOUBLE, FLOAT16, or BFLOAT16"))?,
        };
        let elements = if shape.contains(&0) {
            0
        } else {
            shape
                .iter()
                .try_fold(1usize, |n, &dim| n.checked_mul(dim))
                .ok_or_else(|| format!("logical shape {shape:?} element count overflows usize"))?
        };
        let packed_bytes = (elements / width)
            .checked_mul(bytes)
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| {
                format!("packed byte count for {name} and shape {shape:?} exceeds isize::MAX")
            })?;
        let output_bytes = elements
            .checked_mul(dtype.byte_size())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| {
                format!("output byte count for {shape:?} and {dtype:?} exceeds isize::MAX")
            })?;
        Ok(Self {
            format,
            shape,
            dtype,
            packed_bytes,
            elements,
            output_bytes,
        })
    }

    pub fn format(&self) -> Format {
        self.format
    }
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
    pub fn dtype(&self) -> DataType {
        self.dtype
    }
    pub fn packed_bytes(&self) -> usize {
        self.packed_bytes
    }
    pub fn elements(&self) -> usize {
        self.elements
    }
    pub fn output_bytes(&self) -> usize {
        self.output_bytes
    }

    pub fn validate_input_dtype(&self, dtype: DataType) -> Result<(), String> {
        if dtype != DataType::Uint8 {
            return Err(format!("packed input x must be UINT8, got {dtype:?}"));
        }
        Ok(())
    }

    pub fn validate_input_rank(&self, rank: usize) -> Result<(), String> {
        if rank != 1 {
            return Err(format!(
                "packed input x must have rank one, got rank {rank}"
            ));
        }
        Ok(())
    }

    pub fn validate_packed_bytes(&self, length: usize) -> Result<(), String> {
        if length != self.packed_bytes {
            return Err(format!(
                "packed input x has {length} bytes; expected exactly {} for {} and shape {:?}",
                self.packed_bytes,
                self.format.name(),
                self.shape
            ));
        }
        Ok(())
    }

    pub fn validate_input(&self, dtype: DataType, shape: &[usize]) -> Result<(), String> {
        self.validate_input_dtype(dtype)?;
        self.validate_input_rank(shape.len())?;
        self.validate_packed_bytes(shape[0])
    }

    pub fn validate_output(&self, dtype: DataType, shape: &[usize]) -> Result<(), String> {
        if dtype != self.dtype || shape != self.shape {
            return Err(format!(
                "output y is {dtype:?}{shape:?}; expected {:?}{:?} from node attributes",
                self.dtype, self.shape
            ));
        }
        Ok(())
    }
}
