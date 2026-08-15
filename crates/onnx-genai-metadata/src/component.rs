//! Backend-neutral pipeline component-session interface.
//!
//! A multi-component (pipeline) model runs several ONNX graphs — an embedding
//! stage, a decoder, optional per-step encoders, and so on. Historically the
//! engine only ever instantiated these components through ONNX Runtime `Session`
//! objects, so the pure-Rust native backend could not drive a pipeline at all.
//!
//! [`ComponentSession`] is the backend-neutral seam that lets `PipelineEngine`
//! construct and query every declared component through **either** an ORT
//! `Session` **or** the native `InferenceSession`, without either backend's
//! concrete tensor/session types leaking into the engine's construction path.
//! It lives in the metadata crate — the lowest crate that **both** backend
//! crates already depend on — so `onnx-genai-ort` and the native backend can
//! each implement it without introducing a dependency cycle.
//!
//! The interface exposes exactly what backend-neutral pipeline construction and
//! wiring need:
//!
//! * **Graph I/O metadata** — [`ComponentSession::inputs`] /
//!   [`ComponentSession::outputs`] return each port's name, dtype, and shape
//!   (and therefore rank), so the engine can bind routed inputs and publish
//!   outputs without inspecting a backend-specific session type.
//! * **Named-tensor execution** — [`ComponentSession::run`] takes a map of
//!   named input tensors and returns named output tensors.
//! * **Output names** — [`ComponentSession::output_names`], used to route each
//!   component's outputs into the shared tensor pool.
//!
//! Tensors crossing this seam are host-resident [`ComponentTensor`]s carrying
//! raw little-endian element bytes, so any [`ComponentDataType`] round-trips
//! without a per-dtype host representation and without either backend's tensor
//! type appearing in the interface.

use std::fmt;

/// Scalar element type at a pipeline component's tensor boundary.
///
/// This is the backend-neutral dtype vocabulary the [`ComponentSession`]
/// interface speaks. It mirrors the ONNX tensor element types the pipeline path
/// carries; a backend that observes an element type outside this set surfaces a
/// clear error at the boundary (see [`ComponentError::UnsupportedDataType`])
/// rather than silently coercing it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComponentDataType {
    Float32,
    Float16,
    BFloat16,
    Float8E4M3,
    Float8E5M2,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Bool,
}

impl ComponentDataType {
    /// Size in bytes of one element.
    pub fn size_of(self) -> usize {
        match self {
            ComponentDataType::Float32 | ComponentDataType::Int32 | ComponentDataType::Uint32 => 4,
            ComponentDataType::Float16
            | ComponentDataType::BFloat16
            | ComponentDataType::Int16
            | ComponentDataType::Uint16 => 2,
            ComponentDataType::Float8E4M3
            | ComponentDataType::Float8E5M2
            | ComponentDataType::Int8
            | ComponentDataType::Uint8
            | ComponentDataType::Bool => 1,
            ComponentDataType::Int64 | ComponentDataType::Uint64 => 8,
        }
    }

    /// Stable lower-case name for diagnostics.
    pub fn as_str(self) -> &'static str {
        match self {
            ComponentDataType::Float32 => "float32",
            ComponentDataType::Float16 => "float16",
            ComponentDataType::BFloat16 => "bfloat16",
            ComponentDataType::Float8E4M3 => "float8e4m3",
            ComponentDataType::Float8E5M2 => "float8e5m2",
            ComponentDataType::Int8 => "int8",
            ComponentDataType::Int16 => "int16",
            ComponentDataType::Int32 => "int32",
            ComponentDataType::Int64 => "int64",
            ComponentDataType::Uint8 => "uint8",
            ComponentDataType::Uint16 => "uint16",
            ComponentDataType::Uint32 => "uint32",
            ComponentDataType::Uint64 => "uint64",
            ComponentDataType::Bool => "bool",
        }
    }
}

impl fmt::Display for ComponentDataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A declared graph input or output port.
///
/// `shape` uses the ORT convention where a negative entry denotes a
/// dynamic/symbolic axis, so [`ComponentIo::rank`] is always `shape.len()`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentIo {
    /// Port name as declared by the graph.
    pub name: String,
    /// Declared element type of the port.
    pub dtype: ComponentDataType,
    /// Declared shape; negative entries denote dynamic axes.
    pub shape: Vec<i64>,
}

impl ComponentIo {
    /// The declared rank (number of axes) of the port.
    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

/// An owned, host-resident tensor crossing the component-session boundary.
///
/// `data` holds raw little-endian element bytes in row-major order — the same
/// on-wire representation both the ORT and native backends already use for
/// host tensors — so any [`ComponentDataType`] round-trips through the seam
/// without a per-dtype host container. A concrete tensor's `shape` must be fully
/// static (no negative axes) and its byte length must equal
/// `numel * dtype.size_of()`; both are enforced by [`ComponentTensor::from_raw`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentTensor {
    dtype: ComponentDataType,
    shape: Vec<i64>,
    data: Vec<u8>,
}

impl ComponentTensor {
    /// Build a tensor from raw little-endian element bytes.
    ///
    /// Fails with an actionable error when `shape` carries a dynamic (negative)
    /// axis or when `data.len()` does not match the element count implied by
    /// `shape` and `dtype`.
    pub fn from_raw(
        dtype: ComponentDataType,
        shape: Vec<i64>,
        data: Vec<u8>,
    ) -> Result<Self, ComponentError> {
        let numel = static_numel(&shape)?;
        let expected = numel.saturating_mul(dtype.size_of());
        if data.len() != expected {
            return Err(ComponentError::ByteLengthMismatch {
                dtype,
                shape,
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { dtype, shape, data })
    }

    /// Element type of the tensor.
    pub fn dtype(&self) -> ComponentDataType {
        self.dtype
    }

    /// Fully-static shape of the tensor.
    pub fn shape(&self) -> &[i64] {
        &self.shape
    }

    /// Number of elements (product of the static dimensions).
    pub fn numel(&self) -> usize {
        // Constructed only via `from_raw`, which validates the shape, so the
        // product is well-defined here.
        self.shape.iter().map(|&d| d as usize).product()
    }

    /// Raw little-endian element bytes in row-major order.
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }

    /// Consume the tensor, yielding its raw element bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.data
    }
}

/// Product of a fully-static shape, rejecting dynamic (negative) axes.
fn static_numel(shape: &[i64]) -> Result<usize, ComponentError> {
    let mut numel = 1usize;
    for &dim in shape {
        if dim < 0 {
            return Err(ComponentError::DynamicShape(shape.to_vec()));
        }
        numel =
            numel
                .checked_mul(dim as usize)
                .ok_or_else(|| ComponentError::ByteLengthMismatch {
                    dtype: ComponentDataType::Uint8,
                    shape: shape.to_vec(),
                    expected: usize::MAX,
                    actual: 0,
                })?;
    }
    Ok(numel)
}

/// Failure surface of the backend-neutral component-session interface.
#[derive(Debug, thiserror::Error)]
pub enum ComponentError {
    /// A concrete tensor's byte length did not match its shape and dtype.
    #[error(
        "component tensor byte length {actual} does not match a {dtype} tensor of shape {shape:?} \
         (expected {expected} bytes)"
    )]
    ByteLengthMismatch {
        dtype: ComponentDataType,
        shape: Vec<i64>,
        expected: usize,
        actual: usize,
    },

    /// A concrete tensor was given a dynamic (negative) axis.
    #[error(
        "component tensor shape {0:?} contains a dynamic (negative) dimension; a concrete tensor \
         crossing the component-session boundary must be fully static"
    )]
    DynamicShape(Vec<i64>),

    /// A backend produced or consumed an element type outside the neutral set.
    #[error(
        "unsupported component tensor element type '{observed}'; the pipeline component-session \
         interface supports float{{32,16}}, bfloat16, float8{{e4m3,e5m2}}, \
         int{{8,16,32,64}}, uint{{8,16,32,64}}, and bool"
    )]
    UnsupportedDataType { observed: String },

    /// The underlying backend failed while running the component.
    #[error("component '{component}' failed to run on the {backend} backend: {detail}")]
    Backend {
        component: String,
        backend: &'static str,
        detail: String,
    },
}

/// A single loaded pipeline component, driven through a backend-neutral seam.
///
/// Implementors wrap a concrete backend session (an ORT `Session` or the native
/// `InferenceSession`) and translate the neutral [`ComponentTensor`] boundary
/// to and from that backend's tensor type. The trait is intentionally
/// object-safe so the engine can hold a heterogeneous set of components as
/// `Box<dyn ComponentSession>` after selecting a single backend for the whole
/// pipeline.
pub trait ComponentSession {
    /// Declared graph inputs (name, dtype, shape/rank), in graph order.
    fn inputs(&self) -> &[ComponentIo];

    /// Declared graph outputs (name, dtype, shape/rank), in graph order.
    fn outputs(&self) -> &[ComponentIo];

    /// Names of the declared graph inputs, in graph order.
    fn input_names(&self) -> Vec<&str> {
        self.inputs().iter().map(|io| io.name.as_str()).collect()
    }

    /// Names of the declared graph outputs, in graph order.
    ///
    /// The order matches the tensors returned by [`run`](Self::run).
    fn output_names(&self) -> Vec<&str> {
        self.outputs().iter().map(|io| io.name.as_str()).collect()
    }

    /// Run the component over a set of named input tensors.
    ///
    /// Returns one `(output_name, tensor)` pair per declared graph output, in
    /// [`output_names`](Self::output_names) order.
    fn run(
        &mut self,
        inputs: &[(&str, &ComponentTensor)],
    ) -> Result<Vec<(String, ComponentTensor)>, ComponentError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_raw_accepts_matching_byte_length() {
        let data = vec![0u8; 2 * 3 * 4]; // 6 float32 elements
        let tensor = ComponentTensor::from_raw(ComponentDataType::Float32, vec![2, 3], data)
            .expect("tensor");
        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.shape(), &[2, 3]);
        assert_eq!(tensor.as_bytes().len(), 24);
    }

    #[test]
    fn from_raw_rejects_byte_length_mismatch() {
        let err = ComponentTensor::from_raw(ComponentDataType::Int64, vec![2, 2], vec![0u8; 8])
            .expect_err("mismatch");
        match err {
            ComponentError::ByteLengthMismatch {
                expected, actual, ..
            } => {
                assert_eq!(expected, 32);
                assert_eq!(actual, 8);
            }
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn from_raw_rejects_dynamic_shape() {
        let err = ComponentTensor::from_raw(ComponentDataType::Float32, vec![-1, 4], vec![])
            .expect_err("dynamic");
        assert!(matches!(err, ComponentError::DynamicShape(_)));
    }

    #[test]
    fn io_rank_matches_shape_len() {
        let io = ComponentIo {
            name: "x".to_string(),
            dtype: ComponentDataType::Float16,
            shape: vec![1, -1, 8],
        };
        assert_eq!(io.rank(), 3);
    }

    #[test]
    fn data_type_sizes() {
        assert_eq!(ComponentDataType::Float32.size_of(), 4);
        assert_eq!(ComponentDataType::Int64.size_of(), 8);
        assert_eq!(ComponentDataType::Bool.size_of(), 1);
        assert_eq!(ComponentDataType::BFloat16.size_of(), 2);
    }
}
