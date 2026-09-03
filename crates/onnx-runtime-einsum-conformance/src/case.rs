use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::{EquationError, RouteProbe, SchemaAuthority, SchemaAuthorityError, analyze_equation};

/// Maximum bytes in one materialized input or output tensor.
pub const UNIT_TENSOR_BYTES: usize = 8 * 1024 * 1024;
/// Maximum aggregate CPU input/output/working-set budget for one case.
pub const CPU_WORKING_SET_BYTES: usize = 32 * 1024 * 1024;
/// Maximum aggregate GPU case budget for one case.
pub const GPU_CASE_BYTES: usize = 64 * 1024 * 1024;

/// Resource ceilings carried by every corpus record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CaseLimits {
    /// Maximum bytes in any one tensor.
    pub unit_tensor_bytes: usize,
    /// Maximum aggregate CPU working set.
    pub cpu_working_set_bytes: usize,
    /// Maximum aggregate GPU case footprint.
    pub gpu_case_bytes: usize,
    /// Maximum scalar factor/reduction visits in the direct oracle.
    pub oracle_work_items: usize,
}

impl Default for CaseLimits {
    fn default() -> Self {
        Self {
            unit_tensor_bytes: UNIT_TENSOR_BYTES,
            cpu_working_set_bytes: CPU_WORKING_SET_BYTES,
            gpu_case_bytes: GPU_CASE_BYTES,
            oracle_work_items: 2_000_000,
        }
    }
}

/// Homogeneous numeric tensor dtypes admitted by at least one ONNX Einsum schema.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConformanceDType {
    /// Unsigned 8-bit integer.
    Uint8,
    /// Unsigned 16-bit integer.
    Uint16,
    /// Unsigned 32-bit integer.
    Uint32,
    /// Unsigned 64-bit integer.
    Uint64,
    /// Signed 8-bit integer.
    Int8,
    /// Signed 16-bit integer.
    Int16,
    /// Signed 32-bit integer.
    Int32,
    /// Signed 64-bit integer.
    Int64,
    /// IEEE binary16.
    Float16,
    /// IEEE binary32.
    Float32,
    /// IEEE binary64.
    Float64,
    /// Brain floating point, admitted by Einsum-28.
    BFloat16,
}

impl ConformanceDType {
    /// All Einsum-12 dtypes in pinned ONNX source order.
    pub const V12_TYPES: [Self; 11] = [
        Self::Uint8,
        Self::Uint16,
        Self::Uint32,
        Self::Uint64,
        Self::Int8,
        Self::Int16,
        Self::Int32,
        Self::Int64,
        Self::Float16,
        Self::Float32,
        Self::Float64,
    ];

    /// Stored bytes per element.
    pub const fn byte_size(self) -> usize {
        match self {
            Self::Uint8 | Self::Int8 => 1,
            Self::Uint16 | Self::Int16 | Self::Float16 | Self::BFloat16 => 2,
            Self::Uint32 | Self::Int32 | Self::Float32 => 4,
            Self::Uint64 | Self::Int64 | Self::Float64 => 8,
        }
    }

    /// Fixed integer width, or `None` for floating point.
    pub const fn integer_bits(self) -> Option<u32> {
        match self {
            Self::Uint8 | Self::Int8 => Some(8),
            Self::Uint16 | Self::Int16 => Some(16),
            Self::Uint32 | Self::Int32 => Some(32),
            Self::Uint64 | Self::Int64 => Some(64),
            Self::Float16 | Self::Float32 | Self::Float64 | Self::BFloat16 => None,
        }
    }

    /// Whether the dtype is signed integer.
    pub const fn is_signed_integer(self) -> bool {
        matches!(self, Self::Int8 | Self::Int16 | Self::Int32 | Self::Int64)
    }

    /// Whether the dtype is floating point.
    pub const fn is_float(self) -> bool {
        self.integer_bits().is_none()
    }
}

/// Dtypes used by malformed records, including schema-invalid categories.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclaredDType {
    /// A legal numeric dtype.
    Numeric(ConformanceDType),
    /// Boolean tensor.
    Bool,
    /// String tensor.
    String,
    /// Complex64 tensor.
    Complex64,
    /// Complex128 tensor.
    Complex128,
}

/// Deterministic input-value family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ValueProfile {
    /// Small finite signed values with cancellation.
    Finite,
    /// Exact overflow and sign-boundary integer bit patterns.
    IntegerEdges,
    /// IEEE zeros, infinities, NaNs, subnormals, and normal endpoints.
    BinarySpecials,
}

/// Compact recipe used instead of serialized tensor payloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ValueSpec {
    /// Seed mixed independently with each input index.
    pub seed: u64,
    /// Value family.
    pub profile: ValueProfile,
}

/// One deterministic legal conformance case.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CaseRecord {
    /// Stable human-readable identifier.
    pub id: String,
    /// ONNX Einsum equation.
    pub equation: String,
    /// Imported `ai.onnx` opset.
    pub opset: u64,
    /// Homogeneous input/output dtype.
    pub dtype: ConformanceDType,
    /// Concrete row-major input shapes.
    pub input_shapes: Vec<Vec<usize>>,
    /// Deterministic value recipe.
    pub values: ValueSpec,
    /// Per-case resource ceilings.
    pub limits: CaseLimits,
    /// Future backend routes that must consume this same oracle result.
    pub route_probes: Vec<RouteProbe>,
}

impl CaseRecord {
    /// Native input/output bytes for the homogeneous tensor dtype.
    pub fn native_io_bytes(&self) -> Result<usize, CaseValidationError> {
        let analysis = analyze_equation(&self.equation, &self.input_shapes)?;
        let input_elements = self.input_shapes.iter().try_fold(0usize, |total, shape| {
            checked_numel(shape)
                .and_then(|count| total.checked_add(count))
                .ok_or(CaseValidationError::SizeOverflow {
                    tensor: "aggregate inputs".into(),
                })
        })?;
        let output_elements =
            checked_numel(analysis.output_shape()).ok_or(CaseValidationError::SizeOverflow {
                tensor: "output".into(),
            })?;
        input_elements
            .checked_add(output_elements)
            .and_then(|elements| elements.checked_mul(self.dtype.byte_size()))
            .ok_or(CaseValidationError::SizeOverflow {
                tensor: "aggregate native input/output".into(),
            })
    }

    /// Actual direct-oracle buffers: u64 input/output bits plus one f64
    /// condition-scale value per output element.
    pub fn oracle_working_set_bytes(&self) -> Result<usize, CaseValidationError> {
        let analysis = analyze_equation(&self.equation, &self.input_shapes)?;
        let input_elements = self.input_shapes.iter().try_fold(0usize, |total, shape| {
            checked_numel(shape)
                .and_then(|count| total.checked_add(count))
                .ok_or(CaseValidationError::SizeOverflow {
                    tensor: "oracle inputs".into(),
                })
        })?;
        let output_elements =
            checked_numel(analysis.output_shape()).ok_or(CaseValidationError::SizeOverflow {
                tensor: "oracle output".into(),
            })?;
        input_elements
            .checked_mul(std::mem::size_of::<u64>())
            .and_then(|input_bytes| {
                output_elements
                    .checked_mul(std::mem::size_of::<u64>() + std::mem::size_of::<f64>())
                    .and_then(|output_bytes| input_bytes.checked_add(output_bytes))
            })
            .ok_or(CaseValidationError::SizeOverflow {
                tensor: "oracle working set".into(),
            })
    }

    /// Validate schema, equation, shapes, and resource ceilings independently
    /// of the production planner.
    pub fn validate(&self) -> Result<(), CaseValidationError> {
        validate_case_signature(
            &self.equation,
            self.opset,
            &vec![DeclaredDType::Numeric(self.dtype); self.input_shapes.len()],
            &self.input_shapes,
            1,
        )?;
        let analysis = analyze_equation(&self.equation, &self.input_shapes)?;
        let mut io_bytes = 0usize;
        for (input, shape) in self.input_shapes.iter().enumerate() {
            let elements = checked_numel(shape).ok_or(CaseValidationError::SizeOverflow {
                tensor: format!("input #{input}"),
            })?;
            let bytes = elements.checked_mul(self.dtype.byte_size()).ok_or(
                CaseValidationError::SizeOverflow {
                    tensor: format!("input #{input}"),
                },
            )?;
            if bytes > self.limits.unit_tensor_bytes {
                return Err(CaseValidationError::UnitTensor {
                    tensor: format!("input #{input}"),
                    bytes,
                    limit: self.limits.unit_tensor_bytes,
                });
            }
            io_bytes = io_bytes
                .checked_add(bytes)
                .ok_or(CaseValidationError::SizeOverflow {
                    tensor: "aggregate inputs".into(),
                })?;
        }
        let output_elements =
            checked_numel(analysis.output_shape()).ok_or(CaseValidationError::SizeOverflow {
                tensor: "output".into(),
            })?;
        let output_bytes = output_elements.checked_mul(self.dtype.byte_size()).ok_or(
            CaseValidationError::SizeOverflow {
                tensor: "output".into(),
            },
        )?;
        if output_bytes > self.limits.unit_tensor_bytes {
            return Err(CaseValidationError::UnitTensor {
                tensor: "output".into(),
                bytes: output_bytes,
                limit: self.limits.unit_tensor_bytes,
            });
        }
        io_bytes = io_bytes
            .checked_add(output_bytes)
            .ok_or(CaseValidationError::SizeOverflow {
                tensor: "aggregate input/output".into(),
            })?;
        let oracle_bytes = self.oracle_working_set_bytes()?;
        if oracle_bytes > self.limits.cpu_working_set_bytes {
            return Err(CaseValidationError::CpuWorkingSet {
                bytes: oracle_bytes,
                limit: self.limits.cpu_working_set_bytes,
            });
        }
        if io_bytes > self.limits.gpu_case_bytes {
            return Err(CaseValidationError::GpuCase {
                bytes: io_bytes,
                limit: self.limits.gpu_case_bytes,
            });
        }
        if analysis.work_items() > self.limits.oracle_work_items {
            return Err(CaseValidationError::OracleWork {
                work_items: analysis.work_items(),
                limit: self.limits.oracle_work_items,
            });
        }
        Ok(())
    }
}

/// Compact checked-in generator snapshot.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CorpusSnapshot {
    /// Snapshot format.
    pub format_version: u32,
    /// Generator configuration.
    pub generator: crate::GeneratorConfig,
    /// Expected legal case count.
    pub expected_case_count: usize,
    /// Expected malformed case count.
    pub expected_malformed_count: usize,
    /// Expected forced-route probe count across legal cases.
    pub expected_route_probe_count: usize,
    /// SHA-256 of canonical JSON for the generated legal records.
    pub generated_sha256: String,
}

/// Independent malformed-input category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MalformedKind {
    /// Equation grammar or arrow error.
    Syntax,
    /// Duplicate or unknown output label.
    Output,
    /// Multiple ellipses or incompatible fixed ellipsis.
    Ellipsis,
    /// Operator declares zero inputs.
    NoInputs,
    /// Operator declares an invalid number of outputs.
    OutputCount,
    /// Input rank and term rank disagree.
    Rank,
    /// Repeated diagonal dimensions disagree.
    DiagonalDimension,
    /// Named dimensions or ellipsis broadcast dimensions disagree.
    Dimension,
    /// Homogeneous dtype constraint is violated.
    MixedDType,
    /// Dtype is nonnumeric.
    Nonnumeric,
    /// BF16 is used before Einsum-28.
    Bf16Before28,
}

/// One stable malformed corpus record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MalformedCase {
    /// Stable identifier.
    pub id: String,
    /// Equation bytes represented as UTF-8.
    pub equation: String,
    /// Imported opset.
    pub opset: u64,
    /// Declared input dtypes.
    pub input_dtypes: Vec<DeclaredDType>,
    /// Declared input shapes.
    pub input_shapes: Vec<Vec<usize>>,
    /// Declared output count.
    pub output_count: usize,
    /// Failure category.
    pub kind: MalformedKind,
    /// Actionable text fragment expected from validators.
    pub expected_fragment: String,
}

impl MalformedCase {
    /// Run the independent signature validator and return its failure.
    pub fn validate_failure(&self) -> Result<(), CaseValidationError> {
        validate_case_signature(
            &self.equation,
            self.opset,
            &self.input_dtypes,
            &self.input_shapes,
            self.output_count,
        )
    }
}

/// Validate node arity, dtype constraints, schema boundary, and equation/shape
/// semantics without using production Einsum code.
pub fn validate_case_signature(
    equation: &str,
    opset: u64,
    input_dtypes: &[DeclaredDType],
    input_shapes: &[Vec<usize>],
    output_count: usize,
) -> Result<(), CaseValidationError> {
    SchemaAuthority::verify()?;
    if input_dtypes.is_empty() {
        return Err(CaseValidationError::NoInputs);
    }
    if input_dtypes.len() != input_shapes.len() {
        return Err(CaseValidationError::MetadataArity {
            dtypes: input_dtypes.len(),
            shapes: input_shapes.len(),
        });
    }
    if output_count != 1 {
        return Err(CaseValidationError::OutputCount(output_count));
    }
    let first = match input_dtypes[0] {
        DeclaredDType::Numeric(dtype) => dtype,
        other => return Err(CaseValidationError::Nonnumeric(other)),
    };
    for (index, declared) in input_dtypes.iter().copied().enumerate().skip(1) {
        match declared {
            DeclaredDType::Numeric(dtype) if dtype == first => {}
            DeclaredDType::Numeric(dtype) => {
                return Err(CaseValidationError::MixedDtype {
                    input: index,
                    first,
                    found: dtype,
                });
            }
            other => return Err(CaseValidationError::Nonnumeric(other)),
        }
    }
    if !SchemaAuthority::supports(opset, first)? {
        return Err(CaseValidationError::UnsupportedDtype {
            opset,
            dtype: first,
            schema: SchemaAuthority::since_version(opset)?,
        });
    }
    analyze_equation(equation, input_shapes)?;
    Ok(())
}

/// SHA-256 of canonical compact JSON for legal corpus records.
pub fn corpus_digest(cases: &[CaseRecord]) -> String {
    let bytes = serde_json::to_vec(cases).expect("CaseRecord serialization is infallible");
    format!("{:x}", Sha256::digest(bytes))
}

fn checked_numel(shape: &[usize]) -> Option<usize> {
    shape
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
}

/// Independent case validation failure.
#[derive(Debug, Error)]
pub enum CaseValidationError {
    /// Pinned schema authority failure.
    #[error(transparent)]
    Schema(#[from] SchemaAuthorityError),
    /// Equation/shape failure.
    #[error(transparent)]
    Equation(#[from] EquationError),
    /// No input was declared.
    #[error("Einsum expected at least one input, found none")]
    NoInputs,
    /// Input dtype and shape metadata arrays disagree.
    #[error("Einsum metadata declares {dtypes} input dtypes but {shapes} input shapes")]
    MetadataArity {
        /// Dtype count.
        dtypes: usize,
        /// Shape count.
        shapes: usize,
    },
    /// Invalid node output count.
    #[error("Einsum requires exactly 1 output, but the node declares {0} outputs")]
    OutputCount(usize),
    /// A nonnumeric dtype was supplied.
    #[error("Einsum requires numeric tensors, found {0:?}")]
    Nonnumeric(DeclaredDType),
    /// Inputs are not homogeneous.
    #[error(
        "Einsum input #{input} has dtype {found:?}, which does not match input #0 dtype {first:?}"
    )]
    MixedDtype {
        /// Mismatching input.
        input: usize,
        /// First dtype.
        first: ConformanceDType,
        /// Found dtype.
        found: ConformanceDType,
    },
    /// Schema does not admit the dtype.
    #[error(
        "dtype {dtype:?} is not admitted by Einsum-{schema} selected from ai.onnx opset {opset}"
    )]
    UnsupportedDtype {
        /// Imported opset.
        opset: u64,
        /// Rejected dtype.
        dtype: ConformanceDType,
        /// Selected schema.
        schema: u64,
    },
    /// Tensor element or byte count overflowed.
    #[error("Einsum case tensor size overflowed while sizing {tensor}")]
    SizeOverflow {
        /// Tensor label.
        tensor: String,
    },
    /// One tensor exceeds its limit.
    #[error("Einsum case {tensor} needs {bytes} bytes, exceeding the {limit}-byte unit-tensor cap")]
    UnitTensor {
        /// Tensor label.
        tensor: String,
        /// Required bytes.
        bytes: usize,
        /// Limit.
        limit: usize,
    },
    /// Aggregate CPU footprint exceeds its limit.
    #[error("Einsum case needs {bytes} CPU bytes, exceeding the {limit}-byte working-set cap")]
    CpuWorkingSet {
        /// Required bytes.
        bytes: usize,
        /// Limit.
        limit: usize,
    },
    /// Aggregate GPU footprint exceeds its limit.
    #[error("Einsum case needs {bytes} GPU bytes, exceeding the {limit}-byte case cap")]
    GpuCase {
        /// Required bytes.
        bytes: usize,
        /// Limit.
        limit: usize,
    },
    /// Direct evaluator work exceeds its limit.
    #[error(
        "Einsum case needs {work_items} oracle work items, exceeding the configured cap {limit}"
    )]
    OracleWork {
        /// Required visits.
        work_items: usize,
        /// Limit.
        limit: usize,
    },
}
