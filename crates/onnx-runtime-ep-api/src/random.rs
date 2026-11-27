//! Shared ONNX schema and node-local stream for the native random generators.

use std::any::Any;
use std::sync::{Arc, Mutex};

use onnx_runtime_ir::{Attribute, DataType, Node, checked_numel};

use crate::{EpError, Result, TensorMut, TensorView};

pub const OPS: [&str; 4] = [
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
];
pub const VERSIONS: [u64; 2] = [1, 22];
// The standalone executor retains node run state across kernel variants; the
// ORT plugin does not bind that lifecycle.
pub fn is_native_only(op: &str, domain: &str) -> bool {
    onnx_runtime_ir::is_default_domain(domain) && OPS.contains(&op)
}

pub const OUTPUT_TYPES: &[DataType] = &[
    DataType::Float16,
    DataType::Float32,
    DataType::Float64,
    DataType::BFloat16,
];

#[derive(Clone, Copy)]
pub enum Distribution {
    Normal { mean: f64, scale: f64 },
    Uniform { low: f64, high: f64 },
}

impl Distribution {
    pub fn words_per_sample(self) -> u64 {
        match self {
            Self::Normal { .. } => 2,
            Self::Uniform { .. } => 1,
        }
    }

    pub fn sample(self, key: u64, counter: u64) -> f64 {
        match self {
            Self::Uniform { low, high } => {
                let u = (word(key, counter) >> 11) as f64 * (1.0 / (1_u64 << 53) as f64);
                u.mul_add(high - low, low)
            }
            Self::Normal { mean, scale } => {
                let open = |counter| {
                    ((word(key, counter) >> 12) as f64 + 0.5) * (1.0 / (1_u64 << 52) as f64)
                };
                let radius = (-2.0 * open(counter).ln()).sqrt();
                let angle = std::f64::consts::TAU * open(counter + 1);
                (radius * angle.cos()).mul_add(scale, mean)
            }
        }
    }
}

// SplitMix64's counter permutation; the CUDA kernel uses the same mapping.
fn word(key: u64, counter: u64) -> u64 {
    let mut z = key.wrapping_add(counter.wrapping_mul(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[derive(Default)]
struct Stream(Mutex<Option<(u64, u64)>>);

pub struct Generator {
    op: String,
    version: u64,
    // None means the input supplies the shape. A missing dtype is allowed
    // only in that case and also comes from the input.
    shape: Option<Vec<usize>>,
    dtype: Option<DataType>,
    seed: Option<u64>,
    pub distribution: Distribution,
    stream: Arc<Stream>,
}

fn invalid(op: &str, detail: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("{op}: {detail}"))
}

impl Generator {
    pub fn from_node(node: &Node, version: u64) -> Result<Self> {
        let op = node.op_type.as_str();
        let (normal, like) = match op {
            "RandomNormal" => (true, false),
            "RandomNormalLike" => (true, true),
            "RandomUniform" => (false, false),
            "RandomUniformLike" => (false, true),
            _ => return Err(invalid(op, "expected a standard random generator")),
        };
        if node.inputs.len() != usize::from(like)
            || node.inputs.iter().any(Option::is_none)
            || node.outputs.len() != 1
            || node.outputs[0].is_none()
        {
            return Err(invalid(
                op,
                format!(
                    "provide {} input(s) and exactly one output",
                    usize::from(like)
                ),
            ));
        }
        let shape = if like {
            None
        } else {
            let Some(Attribute::Ints(dims)) = node.attr("shape") else {
                return Err(invalid(op, "provide the required INTS `shape` attribute"));
            };
            Some(
                dims.iter()
                    .map(|&dim| {
                        usize::try_from(dim).map_err(|_| {
                            invalid(op, format!("`shape` dimension {dim} must be a nonnegative addressable extent"))
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
            )
        };
        let dtype = match node.attr("dtype") {
            None if like => None,
            None => Some(DataType::Float32),
            Some(Attribute::Int(id)) => Some(
                i32::try_from(*id)
                    .ok()
                    .and_then(DataType::from_onnx)
                    .ok_or_else(|| {
                        invalid(op, format!("`dtype` {id} is not a valid ONNX tensor dtype"))
                    })?,
            ),
            _ => return Err(invalid(op, "`dtype` must be an ONNX tensor type ID (INT)")),
        };
        let read_float = |name, default| match node.attr(name) {
            None => Ok(default),
            Some(Attribute::Float(value)) if value.is_finite() => Ok(f64::from(*value)),
            _ => Err(invalid(op, format!("`{name}` must be a finite FLOAT"))),
        };
        let seed = match node.attr("seed") {
            None => None,
            Some(Attribute::Float(seed)) if seed.is_nan() => None,
            Some(Attribute::Float(seed)) if seed.is_finite() => {
                // Preserve distinct fractional and negative seeds. Signed zero
                // is one numeric seed, and an explicit zero is not auto-seeding.
                Some(if *seed == 0.0 {
                    0
                } else {
                    u64::from(seed.to_bits())
                })
            }
            _ => {
                return Err(invalid(
                    op,
                    "`seed` must be a finite FLOAT or omitted for automatic seeding",
                ));
            }
        };
        let distribution = if normal {
            let mean = read_float("mean", 0.0)?;
            let scale = read_float("scale", 1.0)?;
            if scale < 0.0 {
                return Err(invalid(
                    op,
                    format!("`scale` must be nonnegative, got {scale}"),
                ));
            }
            Distribution::Normal { mean, scale }
        } else {
            let low = read_float("low", 0.0)?;
            let high = read_float("high", 1.0)?;
            if low > high {
                return Err(invalid(
                    op,
                    format!("`low` ({low}) must not exceed `high` ({high})"),
                ));
            }
            Distribution::Uniform { low, high }
        };
        let generator = Self {
            op: op.into(),
            version,
            shape,
            dtype,
            seed,
            distribution,
            stream: Arc::default(),
        };
        generator.resolve_dtype(None)?;
        Ok(generator)
    }

    fn resolve_dtype(&self, input: Option<DataType>) -> Result<Option<DataType>> {
        if self.shape.is_none()
            && let Some(input) = input
            && !matches!(
                input,
                DataType::Float16
                    | DataType::Float32
                    | DataType::Float64
                    | DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::Uint8
                    | DataType::Uint16
                    | DataType::Uint32
                    | DataType::Uint64
                    | DataType::Bool
                    | DataType::String
                    | DataType::Complex64
                    | DataType::Complex128
            )
            && !(input == DataType::BFloat16 && self.version >= 22)
        {
            return Err(invalid(
                &self.op,
                format!(
                    "opset {} cannot take {input:?} as its Like input; use an opset-compatible tensor dtype (BF16 requires opset 22)",
                    self.version
                ),
            ));
        }
        let dtype = self.dtype.or(input);
        if let Some(dtype) = dtype
            && (!OUTPUT_TYPES.contains(&dtype)
                || (dtype == DataType::BFloat16 && self.version < 22))
        {
            return Err(invalid(
                &self.op,
                format!(
                    "opset {} cannot generate {dtype:?}; choose a float `dtype` (FP16, FP32, FP64, or BF16 starting at opset 22)",
                    self.version
                ),
            ));
        }
        Ok(dtype)
    }

    pub fn check_output(&self, inputs: &[TensorView], outputs: &[TensorMut]) -> Result<usize> {
        let like = self.shape.is_none();
        if inputs.len() != usize::from(like) || outputs.len() != 1 {
            return Err(invalid(
                &self.op,
                "bind the schema's required inputs and single output",
            ));
        }
        let shape = self.shape.as_deref().unwrap_or_else(|| inputs[0].shape);
        let dtype = self.resolve_dtype(inputs.first().map(|input| input.dtype))?;
        let out = &outputs[0];
        if Some(out.dtype) != dtype || out.shape != shape {
            return Err(invalid(
                &self.op,
                format!(
                    "expected output {dtype:?} {shape:?}, got {:?} {:?}; bind the inferred shape and dtype",
                    out.dtype, out.shape
                ),
            ));
        }
        self.checked_output_count(shape, out.dtype)
    }

    pub fn host_string_output(&self, shape: &[usize]) -> Result<(DataType, usize)> {
        let dtype = self.resolve_dtype(Some(DataType::String))?.ok_or_else(|| {
            invalid(
                &self.op,
                "a string Like input needs an explicit float `dtype`",
            )
        })?;
        if self.shape.is_some() {
            return Err(invalid(
                &self.op,
                "only Like generators accept a shape input",
            ));
        }
        Ok((dtype, self.checked_output_count(shape, dtype)?))
    }

    fn checked_output_count(&self, shape: &[usize], dtype: DataType) -> Result<usize> {
        checked_numel(shape)
            .filter(|&n| dtype.checked_storage_bytes(n).is_some())
            .ok_or_else(|| {
                invalid(
                    &self.op,
                    format!(
                        "output shape {shape:?} exceeds addressable storage; reduce the extents"
                    ),
                )
            })
    }

    pub fn bind_node_run_state(
        &mut self,
        bind: &mut dyn FnMut(Arc<dyn Any + Send + Sync>) -> Arc<dyn Any + Send + Sync>,
    ) -> Result<()> {
        self.stream = bind(self.stream.clone()).downcast().map_err(|_| {
            invalid(
                &self.op,
                "executor supplied incompatible RNG state; recreate the session",
            )
        })?;
        Ok(())
    }

    /// Reserve a disjoint span of words without tying state to a concrete shape.
    pub fn reserve(&self, samples: usize) -> Result<(u64, u64)> {
        let exhausted = || invalid(&self.op, "RNG stream exhausted; start a new session");
        let words = u64::try_from(samples)
            .ok()
            .and_then(|n| n.checked_mul(self.distribution.words_per_sample()))
            .ok_or_else(exhausted)?;
        let mut state = self.stream.0.lock().map_err(|_| {
            invalid(
                &self.op,
                "RNG state lock was poisoned; recreate the session",
            )
        })?;
        if state.is_none() {
            let key = match self.seed {
                Some(seed) => word(0xa0761d6478bd642f, seed),
                None => getrandom::u64().map_err(|err| {
                    invalid(
                        &self.op,
                        format!(
                            "OS random seed unavailable ({err}); supply a finite `seed` attribute"
                        ),
                    )
                })?,
            };
            *state = Some((key, 0));
        }
        let (key, counter) = state.as_mut().expect("stream was just initialized");
        let start = *counter;
        *counter = start.checked_add(words).ok_or_else(exhausted)?;
        Ok((*key, start))
    }
}

pub fn unsupported_reason(node: &Node, opset: u64, input_dtypes: &[DataType]) -> Option<String> {
    if !is_native_only(&node.op_type, &node.domain) {
        return None;
    }
    Generator::from_node(node, opset)
        .and_then(|spec| spec.resolve_dtype(input_dtypes.first().copied()))
        .err()
        .map(|err| err.to_string())
}
