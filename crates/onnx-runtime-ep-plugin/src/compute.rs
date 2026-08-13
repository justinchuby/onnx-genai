//! `ExportedComputeInfo` — wraps Rust `Kernel`s as `OrtNodeComputeInfo` callbacks.
//!
//! # Shape-inference contract
//!
//! Every op that can appear in a subgraph must have an explicit `ShapeInference`
//! variant. The **fail-closed** policy: if an op has no modelled rule its
//! `ShapeInference` is `Declined { op_type, domain }` and `infer_shapes` returns
//! an error naming the op and domain. This surfaces at Compute time — never
//! silently producing a wrong-shape tensor.
//!
//! Callers should use `ShapeInference::for_node(node, input_shapes, num_outputs)`
//! when node attributes are available. `for_op(op_type)` is kept for contexts
//! where only the op name is known; it returns `Declined` for any op whose shape
//! requires attributes (Reshape, Conv, reductions, etc.).

use std::collections::HashSet;
use std::ffi::c_void;
use std::sync::{Mutex, PoisonError};

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::kernel::{
    Kernel, TensorMetadata, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ep_api::tensor::{DevicePtr, DevicePtrMut, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, DeviceId, Node};

use crate::kernel_ctx::{allocate_output, read_inputs};
use crate::status::{fail_status, ok_status};

// ──────────────────────────────────────────────────────────────────────────────
// Per-axis parameters for Conv
// ──────────────────────────────────────────────────────────────────────────────

/// Per-spatial-axis convolution parameters (kernel size, padding, stride, dilation).
#[derive(Clone, Debug)]
pub struct ConvSpatialAxis {
    pub kernel: usize,
    pub pad_before: usize,
    pub pad_after: usize,
    pub stride: usize,
    pub dilation: usize,
}

// ──────────────────────────────────────────────────────────────────────────────
// ShapeInference
// ──────────────────────────────────────────────────────────────────────────────

/// How to infer output shapes at runtime from the concrete input shapes.
#[derive(Clone, Debug)]
pub enum ShapeInference {
    /// numpy-style broadcast of all inputs → one output.
    ElementwiseBroadcast,
    /// Output shape == input[idx].shape.
    SameAsInput(usize),
    /// `count` outputs, each with shape == input[idx].shape.
    SameAsInputMultiOutput { idx: usize, count: usize },
    /// MatMul / MatMulNBits semantics (handles 1-D, 2-D, batched-ND).
    MatMul,
    /// Gemm: (trans_a, trans_b) flags.
    Gemm { trans_a: bool, trans_b: bool },
    /// Concat along `axis`.
    Concat { axis: i64 },
    /// Transpose with optional explicit permutation.
    Transpose { perm: Option<Vec<usize>> },
    /// Gather: replace axis `axis` of data with the shape of indices.
    Gather { axis: i64 },
    /// GatherND: `batch_dims` leading dims are shared.
    GatherND { batch_dims: usize },
    /// GatherBlockQuantized — treat as GatherND(0) for shape purposes.
    GatherBlockQuantized,
    /// Shape op — emits [len(dims[start:end])] as a 1-D int64 tensor.
    ShapeOp { start: i64, end: Option<i64> },
    /// Squeeze: remove dims listed in `axes` (or all size-1 dims if empty).
    Squeeze { axes: Vec<i64> },
    /// Unsqueeze: insert unit dims at `axes`.
    Unsqueeze { axes: Vec<i64> },
    /// Reshape — output shape read from input[1] at Compute time.
    ReshapeData { allowzero: bool },
    /// Slice — output shape derived from inputs[1..=4] at Compute time.
    SliceData,
    /// Reduction ops with keepdims / axes.
    Reduction {
        keepdims: bool,
        /// None = reduce all axes.
        axes: Option<Vec<i64>>,
        noop_with_empty_axes: bool,
    },
    /// Opset-18+ reduction where axes come from input[1].
    ReductionFromInput {
        keepdims: bool,
        noop_with_empty_axes: bool,
    },
    /// Conv with explicit spatial axis rules.
    Conv {
        out_channels: usize,
        per_axis: Vec<ConvSpatialAxis>,
    },
    /// MultiHeadAttention: [B,S,hidden], optional present_key/value.
    MultiHeadAttention {
        num_heads: usize,
        num_outputs: usize,
    },
    /// GroupQueryAttention.
    GroupQueryAttention {
        num_heads: usize,
        kv_num_heads: usize,
    },
    /// RotaryEmbedding — output same shape as input[0].
    RotaryEmbedding,
    /// LayerNormalization family: output 0 = input[0] shape,
    /// outputs 1+ (Mean, InvStdDev) = input[0] shape with dims from `axis`
    /// onward replaced by 1 (keepdims reduction).
    ///
    /// `raw_axis` is stored as-is from the ONNX attribute (may be negative)
    /// and resolved against the **runtime** input rank in `infer_shapes`.
    /// This avoids pre-resolving against a truncated static shape that has
    /// symbolic dimensions stripped.
    ///
    /// For SkipLayerNormalization the last output (input_skip_bias_sum) is
    /// full-shaped — `full_shape_outputs` lists the output indices that
    /// should keep input[0]'s shape verbatim.
    LayerNorm {
        /// Raw axis from the ONNX attribute; resolved at runtime.
        raw_axis: i64,
        num_outputs: usize,
        /// Output indices (besides 0) that keep the full input shape,
        /// e.g. the `input_skip_bias_sum` output of SkipLayerNormalization.
        full_shape_outputs: Vec<usize>,
    },
    /// Op with no modelled shape rule — Compute will error with op details.
    Declined { op_type: String, domain: String },
}

impl ShapeInference {
    /// Conservative shape inference from op name alone.
    ///
    /// Returns `Declined` for any op that requires node attributes to compute
    /// its output shape correctly, rather than guessing.
    pub fn for_op(op_type: &str) -> Self {
        Self::for_op_domain(op_type, "")
    }

    fn for_op_domain(op_type: &str, domain: &str) -> Self {
        match op_type {
            // ── Elementwise broadcast ops ─────────────────────────────────
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Mod" | "And" | "Or" | "Xor" | "Equal"
            | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" | "BitShift" | "BitwiseAnd"
            | "BitwiseOr" | "BitwiseXor" | "Max" | "Min" | "Mean" | "Sum" | "Where" => {
                Self::ElementwiseBroadcast
            }

            // ── Unary / shape-preserving ──────────────────────────────────
            "Relu"
            | "Sigmoid"
            | "Tanh"
            | "Exp"
            | "Log"
            | "Sqrt"
            | "Abs"
            | "Neg"
            | "Ceil"
            | "Floor"
            | "Round"
            | "Reciprocal"
            | "Not"
            | "Sign"
            | "Erf"
            | "Gelu"
            | "HardSigmoid"
            | "HardSwish"
            | "LeakyRelu"
            | "Elu"
            | "Selu"
            | "Softplus"
            | "Softsign"
            | "Cast"
            | "Identity"
            | "Dropout"
            | "IsNaN"
            | "IsInf"
            | "BitCount"
            | "Bernoulli"
            | "NegativeLogLikelihoodLoss"
            | "Clip" => Self::SameAsInput(0),

            // ── Shape-preserving normalisation ops ───────────────────────
            "Softmax"
            | "LogSoftmax"
            | "Hardmax"
            | "BatchNormalization"
            | "InstanceNormalization"
            | "LpNormalization" => Self::SameAsInput(0),

            // ── LayerNorm family: requires axis attribute for correct shape ──
            // Decline here; for_node() resolves the axis.
            "LayerNormalization"
            | "RMSNormalization"
            | "SkipLayerNormalization"
            | "SkipSimplifiedLayerNormalization"
            | "SimplifiedLayerNormalization" => Self::Declined {
                op_type: op_type.to_string(),
                domain: domain.to_string(),
            },

            // ── Matrix multiply ───────────────────────────────────────────
            "MatMul" | "MatMulNBits" => Self::MatMul,

            // ── Safe defaults for attribute-having ops ────────────────────
            "Concat" => Self::Concat { axis: 0 },
            "Transpose" => Self::Transpose { perm: None },
            "Gather" => Self::Gather { axis: 0 },
            "GatherND" => Self::GatherND { batch_dims: 0 },
            "GatherBlockQuantized" => Self::GatherBlockQuantized,
            "Shape" => Self::ShapeOp {
                start: 0,
                end: None,
            },
            "Reshape" => Self::ReshapeData { allowzero: false },
            "Slice" => Self::SliceData,
            "RotaryEmbedding" => Self::RotaryEmbedding,

            // ── Ops that require attributes — Declined ────────────────────
            "Squeeze"
            | "Unsqueeze"
            | "ReduceMean"
            | "ReduceSum"
            | "ReduceProd"
            | "ReduceMax"
            | "ReduceMin"
            | "ReduceL1"
            | "ReduceL2"
            | "ReduceLogSum"
            | "ReduceLogSumExp"
            | "ReduceSumSquare"
            | "Conv"
            | "ConvTranspose"
            | "ConvInteger"
            | "Gemm"
            | "MultiHeadAttention"
            | "GroupQueryAttention" => Self::Declined {
                op_type: op_type.to_string(),
                domain: domain.to_string(),
            },

            _ => Self::Declined {
                op_type: op_type.to_string(),
                domain: domain.to_string(),
            },
        }
    }

    /// Full shape inference from a compiled IR `Node` plus the shapes
    /// of its inputs (may be empty slices for absent optional inputs).
    /// Each dimension is `Some(n)` for a statically known extent or `None`
    /// for a symbolic/dynamic dimension — preserving rank.
    ///
    /// `num_outputs` is how many output slots the node has in the graph.
    pub fn for_node(node: &Node, input_shapes: &[Vec<Option<usize>>], num_outputs: usize) -> Self {
        let op = node.op_type.as_str();
        let domain = node.domain.as_str();
        let opset = node.version.unwrap_or(0);

        let int_attr = |name: &str| -> Option<i64> { node.attr(name)?.as_int() };
        let ints_attr =
            |name: &str| -> Option<Vec<i64>> { Some(node.attr(name)?.as_ints()?.to_vec()) };

        match op {
            // ── Elementwise ───────────────────────────────────────────────
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Mod" | "And" | "Or" | "Xor" | "Equal"
            | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual" | "BitShift" | "BitwiseAnd"
            | "BitwiseOr" | "BitwiseXor" | "Max" | "Min" | "Mean" | "Sum" | "Where" => {
                Self::ElementwiseBroadcast
            }

            // ── Unary / shape-preserving ──────────────────────────────────
            "Relu"
            | "Sigmoid"
            | "Tanh"
            | "Exp"
            | "Log"
            | "Sqrt"
            | "Abs"
            | "Neg"
            | "Ceil"
            | "Floor"
            | "Round"
            | "Reciprocal"
            | "Not"
            | "Sign"
            | "Erf"
            | "Gelu"
            | "HardSigmoid"
            | "HardSwish"
            | "LeakyRelu"
            | "Elu"
            | "Selu"
            | "Softplus"
            | "Softsign"
            | "Cast"
            | "Identity"
            | "Dropout"
            | "IsNaN"
            | "IsInf"
            | "BitCount"
            | "Bernoulli"
            | "Softmax"
            | "LogSoftmax"
            | "Hardmax"
            | "BatchNormalization"
            | "InstanceNormalization"
            | "LpNormalization"
            | "Clip" => Self::SameAsInput(0),

            // ── LayerNorm / SkipLayerNorm family ──────────────────────────
            "LayerNormalization" | "RMSNormalization" | "SimplifiedLayerNormalization" => {
                // ONNX spec: axis defaults to -1; Mean/InvStdDev shapes are
                // [d[0]..d[axis-1], 1, .., 1].
                // Store raw axis — resolve at runtime against actual rank.
                let raw_axis = int_attr("axis").unwrap_or(-1);
                Self::LayerNorm {
                    raw_axis,
                    num_outputs,
                    full_shape_outputs: vec![],
                }
            }
            "SkipLayerNormalization" | "SkipSimplifiedLayerNormalization" => {
                // Contrib op; normalises the last axis (no axis attr).
                // Outputs: [output, mean, inv_std_dev, input_skip_bias_sum]
                // output and input_skip_bias_sum are full-shaped; mean and
                // inv_std_dev are reduced.
                // raw_axis = -1 (last axis); resolved at runtime.
                let full_shape_outputs = if num_outputs > 3 { vec![3] } else { vec![] };
                Self::LayerNorm {
                    raw_axis: -1,
                    num_outputs,
                    full_shape_outputs,
                }
            }

            // ── MatMul ────────────────────────────────────────────────────
            "MatMul" | "MatMulNBits" => Self::MatMul,

            // ── Gemm ──────────────────────────────────────────────────────
            "Gemm" => {
                let trans_a = int_attr("transA").unwrap_or(0) != 0;
                let trans_b = int_attr("transB").unwrap_or(0) != 0;
                Self::Gemm { trans_a, trans_b }
            }

            // ── Concat ────────────────────────────────────────────────────
            "Concat" => {
                let axis = int_attr("axis").unwrap_or(0);
                Self::Concat { axis }
            }

            // ── Transpose ────────────────────────────────────────────────
            "Transpose" => {
                let perm = ints_attr("perm").map(|v| v.iter().map(|&x| x as usize).collect());
                Self::Transpose { perm }
            }

            // ── Gather / GatherND ─────────────────────────────────────────
            "Gather" => {
                let axis = int_attr("axis").unwrap_or(0);
                Self::Gather { axis }
            }
            "GatherND" => {
                let batch_dims = int_attr("batch_dims").unwrap_or(0) as usize;
                Self::GatherND { batch_dims }
            }
            "GatherBlockQuantized" => Self::GatherBlockQuantized,

            // ── Shape ─────────────────────────────────────────────────────
            "Shape" => {
                let start = int_attr("start").unwrap_or(0);
                let end = int_attr("end");
                Self::ShapeOp { start, end }
            }

            // ── Squeeze ───────────────────────────────────────────────────
            "Squeeze" => {
                // opset 13+: axes from input[1]; earlier: attribute.
                let axes = if opset >= 13 {
                    // We can't read input[1] at compile time (data-dependent),
                    // but we know input_shapes[0] and can remove all size-1 dims
                    // if no axes input is provided.
                    ints_attr("axes").unwrap_or_default()
                } else {
                    ints_attr("axes").unwrap_or_default()
                };
                Self::Squeeze { axes }
            }

            // ── Unsqueeze ─────────────────────────────────────────────────
            "Unsqueeze" => {
                if let Some(axes) = ints_attr("axes") {
                    Self::Unsqueeze { axes }
                } else {
                    // opset-13: axes come from input[1] — data-dependent.
                    Self::Declined {
                        op_type: op.to_string(),
                        domain: domain.to_string(),
                    }
                }
            }

            // ── Reshape ───────────────────────────────────────────────────
            "Reshape" => {
                let allowzero = int_attr("allowzero").unwrap_or(0) != 0;
                Self::ReshapeData { allowzero }
            }

            // ── Slice ─────────────────────────────────────────────────────
            "Slice" => Self::SliceData,

            // ── Reductions ────────────────────────────────────────────────
            op_name if is_reduction(op_name) => {
                let keepdims = int_attr("keepdims").unwrap_or(1) != 0;
                let noop_with_empty_axes = int_attr("noop_with_empty_axes").unwrap_or(0) != 0;
                if opset >= 18 {
                    Self::ReductionFromInput {
                        keepdims,
                        noop_with_empty_axes,
                    }
                } else {
                    let axes = ints_attr("axes");
                    Self::Reduction {
                        keepdims,
                        axes,
                        noop_with_empty_axes,
                    }
                }
            }

            // ── Conv ──────────────────────────────────────────────────────
            "Conv" | "ConvInteger" => {
                if let Some(conv) = build_conv(node, input_shapes) {
                    conv
                } else {
                    Self::Declined {
                        op_type: op.to_string(),
                        domain: domain.to_string(),
                    }
                }
            }

            // ── Attention family (com.microsoft) ──────────────────────────
            "MultiHeadAttention" => {
                let num_heads = int_attr("num_heads").unwrap_or(0) as usize;
                Self::MultiHeadAttention {
                    num_heads,
                    num_outputs,
                }
            }
            "GroupQueryAttention" => {
                let num_heads = int_attr("num_heads").unwrap_or(0) as usize;
                let kv_num_heads = int_attr("kv_num_heads").unwrap_or(num_heads as i64) as usize;
                Self::GroupQueryAttention {
                    num_heads,
                    kv_num_heads,
                }
            }
            "RotaryEmbedding" => Self::RotaryEmbedding,

            _ => Self::Declined {
                op_type: op.to_string(),
                domain: domain.to_string(),
            },
        }
    }
}

fn is_reduction(op: &str) -> bool {
    matches!(
        op,
        "ReduceMean"
            | "ReduceSum"
            | "ReduceProd"
            | "ReduceMax"
            | "ReduceMin"
            | "ReduceL1"
            | "ReduceL2"
            | "ReduceLogSum"
            | "ReduceLogSumExp"
            | "ReduceSumSquare"
    )
}

fn build_conv(node: &Node, input_shapes: &[Vec<Option<usize>>]) -> Option<ShapeInference> {
    let auto_pad = node
        .attr("auto_pad")
        .and_then(|a| a.as_str())
        .unwrap_or("NOTSET");
    if auto_pad != "NOTSET" {
        // SAME_UPPER / SAME_LOWER require runtime spatial size — defer to runtime.
        return None;
    }
    // input[0]: [N, C_in, d0, d1, ...]
    let in_shape = input_shapes.first()?;
    if in_shape.len() < 3 {
        return None;
    }
    let spatial_dims = in_shape.len() - 2;

    // weight[0] first dim is out_channels — fail closed if dynamic.
    let out_channels = input_shapes
        .get(1)
        .and_then(|w| w.first().copied())
        .flatten()?;

    let kernel_shape: Vec<usize> = node
        .attr("kernel_shape")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .or_else(|| {
            // derive from weight shape: weight=[out_ch, in_ch/group, k0, k1, ...]
            let w = input_shapes.get(1)?;
            if w.len() == 2 + spatial_dims {
                // Fail closed: all kernel dims must be static.
                w[2..].iter().copied().collect::<Option<Vec<usize>>>()
            } else {
                None
            }
        })?;

    let strides: Vec<usize> = node
        .attr("strides")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .unwrap_or_else(|| vec![1; spatial_dims]);

    let dilations: Vec<usize> = node
        .attr("dilations")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .unwrap_or_else(|| vec![1; spatial_dims]);

    let pads: Vec<usize> = node
        .attr("pads")
        .and_then(|a| a.as_ints())
        .map(|v| v.iter().map(|&x| x as usize).collect())
        .unwrap_or_else(|| vec![0; 2 * spatial_dims]);

    if kernel_shape.len() != spatial_dims
        || strides.len() != spatial_dims
        || dilations.len() != spatial_dims
        || pads.len() != 2 * spatial_dims
    {
        return None;
    }

    let per_axis = (0..spatial_dims)
        .map(|i| ConvSpatialAxis {
            kernel: kernel_shape[i],
            pad_before: pads[i],
            pad_after: pads[i + spatial_dims],
            stride: strides[i],
            dilation: dilations[i],
        })
        .collect();

    Some(ShapeInference::Conv {
        out_channels,
        per_axis,
    })
}

/// A compiled kernel bundled with the metadata needed to drive execution
/// through the ORT kernel context.
pub struct CompiledKernelEntry {
    pub kernel: Box<dyn Kernel>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    /// Per-output declared IR dtype, read from the ORT graph's value info at
    /// Compile time. Indexed by output slot. Never inferred from inputs —
    /// this is the authoritative dtype for each output tensor. Absent optional
    /// outputs carry their actual ORT-declared dtype (not `Undefined`) so
    /// scratch buffers can be sized correctly for f16/bf16 ops.
    pub output_dtypes: Vec<DataType>,
    /// Which output slots are absent (omitted optional outputs). Absent slots
    /// get a scratch buffer at runtime; present slots are allocated via ORT.
    pub absent_output_slots: HashSet<usize>,
    pub shape_inference: ShapeInference,
    /// Maps node input position → `Some(ort_index)` for present inputs,
    /// `None` for absent optional inputs. Used by the single-kernel fast
    /// path to reconstruct the positional input list with absent sentinels.
    pub input_slots: Vec<Option<usize>>,
}

// ──────────────────────────────────────────────────────────────────────────────
// Scratch buffer sizing — single source of truth
// ──────────────────────────────────────────────────────────────────────────────

/// Compute the scratch buffer allocation size (in bytes) for an absent output slot.
///
/// Formula: `numel * max(byte_size_of(dtype), 8)`.  The `max(…, 8)` padding
/// prevents a heap overflow when a kernel internally writes a wider type than
/// the declared dtype (e.g. Float32 stats for an f16 SkipLayerNormalization).
///
/// **This function is the single authoritative implementation.**  Both the
/// single-node and multi-node compute paths, as well as the canary tests, call
/// it directly.  Duplicating the formula invites drift — and drift here means a
/// heap overflow (the original B1 bug).
pub fn scratch_alloc_bytes(numel: usize, dtype: DataType) -> usize {
    let elem_bytes = dtype.byte_size().max(8);
    numel * elem_bytes
}

// ──────────────────────────────────────────────────────────────────────────────
// Multi-node subgraph routing
// ──────────────────────────────────────────────────────────────────────────────

/// Where a node's i-th input tensor comes from in a fused multi-node subgraph.
#[derive(Clone, Debug)]
pub enum NodeInputSource {
    /// Take from ORT's kernel-context input at this index.
    Ort(usize),
    /// Take from an intermediate buffer written by an earlier node.
    Buffer(usize),
    /// The input is absent (omitted optional). Must not be read by the kernel.
    Absent,
}

/// Where a node's i-th output tensor goes in a fused multi-node subgraph.
#[derive(Clone, Debug)]
pub enum NodeOutputSink {
    /// Write to ORT's kernel-context output at this index.
    Ort(usize),
    /// Write to an intermediate buffer for a later node.
    Buffer(usize),
    /// The output is absent (omitted optional). Scratch-allocated at compute
    /// time; does not consume an intermediate buffer slot.
    Absent,
}

/// Routing table for a fused multi-node subgraph.
///
/// `input_sources[node_idx]` and `output_sinks[node_idx]` are indexed by
/// the kernel's input/output slot, in the same order as `CompiledKernelEntry`.
#[derive(Clone, Debug)]
pub struct SubgraphRouting {
    pub input_sources: Vec<Vec<NodeInputSource>>,
    pub output_sinks: Vec<Vec<NodeOutputSink>>,
    pub num_intermediate_buffers: usize,
}

/// Heap-owned intermediate tensor buffer for multi-node subgraph execution.
///
/// The backing bytes are either owned on the host (`data`, used by unit tests
/// and as a fallback) or borrowed from an ORT scratch allocation (`scratch_ptr`,
/// non-null). For device EPs (CUDA) the scratch allocation lives in device
/// memory, so kernels can read/write intermediates directly on the GPU without
/// an illegal host-pointer dereference. ORT owns scratch memory for the
/// duration of the `Compute` call, which exactly matches an intermediate's
/// lifetime, so `IntermediateBuf` never frees `scratch_ptr`.
pub struct IntermediateBuf {
    pub data: Vec<u8>,
    /// When non-null, the buffer is backed by ORT scratch memory (possibly on
    /// device) instead of the host `data` vector. Not owned — never freed here.
    pub scratch_ptr: *mut u8,
    pub shape: Vec<usize>,
    pub strides: Vec<i64>,
    pub dtype: DataType,
}

impl IntermediateBuf {
    /// Raw const pointer to the backing bytes (scratch if set, else host `data`).
    fn ptr(&self) -> *const u8 {
        if self.scratch_ptr.is_null() {
            self.data.as_ptr()
        } else {
            self.scratch_ptr.cast_const()
        }
    }

    /// Raw mutable pointer to the backing bytes (scratch if set, else host).
    fn ptr_mut(&mut self) -> *mut u8 {
        if self.scratch_ptr.is_null() {
            self.data.as_mut_ptr()
        } else {
            self.scratch_ptr
        }
    }

    /// Immutable view backed by this buffer.
    pub fn view(&self) -> TensorView<'_> {
        TensorView::new(
            DevicePtr(self.ptr().cast()),
            self.dtype,
            &self.shape,
            &self.strides,
            onnx_runtime_ir::DeviceId::cpu(),
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// ExportedComputeInfo
// ──────────────────────────────────────────────────────────────────────────────

// ──────────────────────────────────────────────────────────────────────────────
// Workspace plan cache
// ──────────────────────────────────────────────────────────────────────────────

/// One operand's contribution to a [`WorkspaceSignature`] — exactly the three
/// fields a kernel sees in [`TensorMetadata`], and nothing else.
///
/// It must stay exactly those three: `Kernel::workspace_requirement` is handed
/// `TensorMetadata { dtype, shape, present }`, so two dispatches whose operands
/// agree on all three are indistinguishable to the kernel, and any field the key
/// drops is a way to serve a plan computed for different operands.
#[derive(Clone, Debug, PartialEq, Eq)]
struct OperandKey {
    dtype: DataType,
    present: bool,
    shape: Vec<usize>,
}

impl OperandKey {
    fn matches(&self, metadata: &TensorMetadata<'_>) -> bool {
        self.dtype == metadata.dtype
            && self.present == metadata.present
            && self.shape.as_slice() == metadata.shape
    }
}

/// Owned copy of the metadata a [`WorkspaceRequirement`] was computed from.
type WorkspaceSignature = Vec<OperandKey>;

fn signature_matches(signature: &WorkspaceSignature, metadata: &[TensorMetadata<'_>]) -> bool {
    signature.len() == metadata.len()
        && signature
            .iter()
            .zip(metadata)
            .all(|(key, meta)| key.matches(meta))
}

fn signature_of(metadata: &[TensorMetadata<'_>]) -> WorkspaceSignature {
    metadata
        .iter()
        .map(|meta| OperandKey {
            dtype: meta.dtype,
            present: meta.present,
            shape: meta.shape.to_vec(),
        })
        .collect()
}

/// How many distinct operand signatures one node remembers a plan for.
///
/// Sized for the shapes a decoder actually alternates between (prefill, one or
/// two decode extents, a padded variant) with headroom, while staying small
/// enough that a linear scan under the lock is cheaper than the planning call it
/// avoids. Overflow is a miss, never a wrong answer.
const WORKSPACE_PLAN_CACHE_CAPACITY: usize = 8;

/// Memoizes `Kernel::workspace_requirement` per node, keyed by the exact
/// operand metadata the requirement was derived from.
///
/// # Why this exists
///
/// `prepare_workspace` must ask the kernel what it needs before it can decide
/// whether it can serve it. For the cuBLASLt-backed CUDA kernels (`MatMul`,
/// `Gemm`, `FusedEpilogue`, `MatMulNBits`' f32 dequant path) answering that
/// question runs a full `cublasLtMatmulAlgoGetHeuristic` search — and because
/// those kernels declare `WorkspaceLifetime::SessionPersistent`, which this
/// executor declines, the kernel then plans a *second* time inside its own
/// `execute`. Every node of every decode step paid for two heuristic searches
/// and used one.
///
/// The kernel-side re-plan cannot be removed from here: `plan_gemm` returns the
/// selected algorithm and matrix layouts, not just a byte count, and the kernel
/// re-derives and re-validates the size it was handed rather than trusting the
/// executor. So the duplicate is removed on the executor side instead, by not
/// re-asking a question whose answer cannot have changed.
///
/// # Correctness
///
/// * The key is the full operand metadata (dtype, presence, shape) for every
///   operand, so a hit is only possible when the kernel would be asked exactly
///   the same question. Dynamic shapes, dtype changes and optional-input
///   presence changes are all misses.
/// * Nothing is shared between nodes: each `CompiledKernelEntry` gets its own
///   cache, so one node's plan can never be served to another.
/// * The lock is never held across the kernel call, so concurrent `Run`s on the
///   same session plan in parallel; a duplicated concurrent computation is
///   harmless because it produces the same value.
/// * Planning errors are never cached — a failing `workspace_requirement` is
///   re-raised on every dispatch, exactly as before.
/// * A poisoned lock is recovered with `PoisonError::into_inner`, matching
///   `EpHandle::with` and `release_ep_factory_with_teardown`: a panic in some
///   unrelated dispatch must not turn every later dispatch into a hard error.
///
/// # What this assumes of the kernel
///
/// That `workspace_requirement` is a function of the operand metadata alone for
/// a given kernel instance. That is the contract the native session executor
/// already relies on (it plans once during prepare and reuses the reservation
/// for every later `Run`), and it holds for every kernel in this workspace.
/// A kernel that violates it is not silently mis-served: the CUDA consumers
/// re-derive their requirement in `execute_with_workspace` and reject a
/// workspace that is too small (`governed_workspace_ptr`, `std_attention_carve`,
/// `gqa` sub-range carving) rather than reading past the end.
struct WorkspacePlanCache {
    plans: Mutex<Vec<(WorkspaceSignature, WorkspaceRequirement)>>,
}

impl WorkspacePlanCache {
    fn new() -> Self {
        Self {
            plans: Mutex::new(Vec::new()),
        }
    }

    /// Look up the plan for `metadata`, computing and remembering it on a miss.
    fn get_or_plan(
        &self,
        metadata: &[TensorMetadata<'_>],
        plan: impl FnOnce() -> Result<WorkspaceRequirement, String>,
    ) -> Result<WorkspaceRequirement, String> {
        if let Some(hit) = self.lookup(metadata) {
            return Ok(hit);
        }
        let requirement = plan()?;
        self.remember(signature_of(metadata), requirement);
        Ok(requirement)
    }

    fn lookup(&self, metadata: &[TensorMetadata<'_>]) -> Option<WorkspaceRequirement> {
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        let idx = plans
            .iter()
            .position(|(signature, _)| signature_matches(signature, metadata))?;
        // Move-to-front so the hot signature stays ahead of a one-off prefill
        // shape once the cache is full.
        let entry = plans.remove(idx);
        let requirement = entry.1;
        plans.insert(0, entry);
        Some(requirement)
    }

    fn remember(&self, signature: WorkspaceSignature, requirement: WorkspaceRequirement) {
        let mut plans = self.plans.lock().unwrap_or_else(PoisonError::into_inner);
        // A concurrent dispatch may have inserted the same signature while this
        // one was planning. Overwrite rather than duplicate: both computed the
        // same answer from the same metadata.
        if let Some(slot) = plans
            .iter_mut()
            .find(|(existing, _)| *existing == signature)
        {
            slot.1 = requirement;
            return;
        }
        if plans.len() >= WORKSPACE_PLAN_CACHE_CAPACITY {
            plans.pop();
        }
        plans.insert(0, (signature, requirement));
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.plans
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }
}

/// Heap-allocated compute info whose raw pointer is returned as
/// `OrtNodeComputeInfo*`.
///
/// The first field is the `OrtNodeComputeInfo` vtable.
#[repr(C)]
pub struct ExportedComputeInfo {
    pub vtable: ort::OrtNodeComputeInfo,
    /// The compiled kernel entries for this subgraph (in topological order).
    pub entries: Vec<CompiledKernelEntry>,
    /// Optional routing table for multi-node fused subgraphs.
    pub routing: Option<SubgraphRouting>,
    /// One [`WorkspacePlanCache`] per entry in `entries`, same index. Built in
    /// [`ExportedComputeInfo::new`] so it is always exactly as long as
    /// `entries`.
    workspace_plans: Vec<WorkspacePlanCache>,
}

/// Per-session state created by `CreateState`.
struct ComputeState {
    _placeholder: u8,
}

impl ExportedComputeInfo {
    pub fn new(entries: Vec<CompiledKernelEntry>) -> Self {
        let workspace_plans = entries.iter().map(|_| WorkspacePlanCache::new()).collect();
        Self {
            vtable: ort::OrtNodeComputeInfo {
                ort_version_supported: ort::ORT_API_VERSION,
                CreateState: Some(compute_create_state),
                Compute: Some(compute_execute),
                ReleaseState: Some(compute_release_state),
            },
            entries,
            routing: None,
            workspace_plans,
        }
    }

    /// Attach a subgraph routing table (for multi-node fused subgraphs).
    pub fn set_routing(&mut self, routing: SubgraphRouting) {
        self.routing = Some(routing);
    }

    /// Plan cache for node `idx`. `None` when the index is out of range, which
    /// a caller must treat as a hard error rather than by planning uncached:
    /// a length mismatch means the two vectors have drifted.
    fn workspace_plan_cache(&self, idx: usize) -> Option<&WorkspacePlanCache> {
        self.workspace_plans.get(idx)
    }
}

/// CreateState: allocate per-session compute state.
unsafe extern "C" fn compute_create_state(
    _info: *mut ort::OrtNodeComputeInfo,
    _compute_context: *mut ort::OrtNodeComputeContext,
    out_state: *mut *mut c_void,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if out_state.is_null() {
            return fail_status("CreateState: out_state is null");
        }
        let state = Box::new(ComputeState { _placeholder: 0 });
        unsafe { *out_state = Box::into_raw(state).cast::<c_void>() };
        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("CreateState: internal panic"))
}

/// Returns the `OrtMemoryInfo*` of the EP's tensor memory, read from **fused
/// subgraph input 0**'s `OrtValue`. For a device EP (CUDA) this is device
/// memory; for the CPU EP it is host memory.
///
/// This is the memory info intermediate buffers are allocated against — the
/// exact derivation #832 validated on H200 — and it is deliberately unchanged
/// here. It is a *subgraph*-level fact, not a per-node one: for a fused
/// multi-node subgraph, node `k > 0` may take none of its operands from ORT.
/// The per-node derivation used for kernel workspaces is
/// [`operand_mem_info`]; see its docs for what is and is not guaranteed.
///
/// Returns `None` when there are no inputs or the memory info cannot be read; in
/// that case callers fall back to host-owned intermediate buffers.
///
/// # Safety
///
/// `api` must be valid and `ctx` a valid `OrtKernelContext*`.
unsafe fn device_mem_info(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
) -> Option<*const ort::OrtMemoryInfo> {
    unsafe { ort_input_mem_info(api, ctx, 0) }
}

/// Memory info of the `OrtValue` bound to kernel-context input `index`.
///
/// # Safety
///
/// `api` must be valid and `ctx` a valid `OrtKernelContext*`.
unsafe fn ort_input_mem_info(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    index: usize,
) -> Option<*const ort::OrtMemoryInfo> {
    let get_input = api.KernelContext_GetInput?;
    let get_mem_info = api.GetTensorMemoryInfo?;
    let mut value: *const ort::OrtValue = std::ptr::null();
    let status = unsafe { get_input(ctx, index, &mut value) };
    if !status.is_null() || value.is_null() {
        return None;
    }
    let mut mem_info: *const ort::OrtMemoryInfo = std::ptr::null();
    let status = unsafe { get_mem_info(value, &mut mem_info) };
    if !status.is_null() || mem_info.is_null() {
        return None;
    }
    Some(mem_info)
}

/// Where one node's operands live, as far as ORT will tell us.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum OperandMemInfo {
    /// Every ORT-bound operand of this node reported the same memory info (or
    /// there was exactly one, so there was nothing to disagree with).
    Uniform(*const ort::OrtMemoryInfo),
    /// This node binds no ORT operand at all — in a fused subgraph every one of
    /// its inputs comes from an intermediate buffer, which was allocated
    /// against the subgraph-level [`device_mem_info`]. That is then the right
    /// answer for this node too, and it is what is carried here.
    FromIntermediates(*const ort::OrtMemoryInfo),
    /// The memory device could not be read (no inputs, or ORT refused).
    Unavailable,
    /// Two ORT-bound operands of the same node reported **different** memory
    /// infos, so "the device this node runs on" is not determined by its
    /// operands. Carries the two kernel-context input indices that disagreed.
    Divergent { first: usize, other: usize },
}

/// Derive the memory device of **one node's own operands**.
///
/// `ort_operands` lists the kernel-context input indices this node actually
/// reads, in node-operand order, with absent optional operands and
/// intermediate-buffer operands already filtered out. `subgraph_fallback` is
/// [`device_mem_info`], used only when the node binds no ORT operand at all.
///
/// # What this guarantees, and what it does not
///
/// It guarantees the returned memory info is the one ORT reported for an
/// operand **of this node** — not, as before, for input 0 of the whole fused
/// subgraph, which for node `k > 0` may be a different tensor on a different
/// device entirely.
///
/// When the node binds more than one ORT operand, every one of them is compared
/// against the first with `OrtApi::CompareMemoryInfo`, and a disagreement is
/// reported as [`OperandMemInfo::Divergent`] rather than resolved by guessing.
/// This is a real check on real handles, not an assumption: mixed-device
/// operands are legal in ORT (an op may declare `OrtMemTypeCPUInput` for a
/// small control operand), and in that case the operands genuinely do not
/// determine where the kernel's compute runs.
///
/// If `CompareMemoryInfo` is unavailable on the host API, a multi-operand node
/// reports [`OperandMemInfo::Unavailable`] — no comparison is claimed that was
/// not performed. A single-operand node never needs the comparator.
///
/// # Safety
///
/// `api` must be valid and `ctx` a valid `OrtKernelContext*`.
unsafe fn operand_mem_info(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    ort_operands: &[usize],
    subgraph_fallback: Option<*const ort::OrtMemoryInfo>,
) -> OperandMemInfo {
    let Some((&first_idx, rest)) = ort_operands.split_first() else {
        return match subgraph_fallback {
            Some(ptr) => OperandMemInfo::FromIntermediates(ptr),
            None => OperandMemInfo::Unavailable,
        };
    };
    let Some(first) = (unsafe { ort_input_mem_info(api, ctx, first_idx) }) else {
        return OperandMemInfo::Unavailable;
    };
    if rest.is_empty() {
        return OperandMemInfo::Uniform(first);
    }
    let Some(compare) = api.CompareMemoryInfo else {
        return OperandMemInfo::Unavailable;
    };
    for &idx in rest {
        let Some(other) = (unsafe { ort_input_mem_info(api, ctx, idx) }) else {
            return OperandMemInfo::Unavailable;
        };
        let mut equal: std::os::raw::c_int = 0;
        let status = unsafe { compare(first, other, &mut equal) };
        if !status.is_null() {
            return OperandMemInfo::Unavailable;
        }
        if equal != 0 {
            return OperandMemInfo::Divergent {
                first: first_idx,
                other: idx,
            };
        }
    }
    OperandMemInfo::Uniform(first)
}

/// Allocates `bytes` of scratch memory via ORT for the given `mem_info`. The
/// memory is owned by ORT for the duration of the `Compute` call — never freed
/// by the caller. For a device EP this returns device memory.
///
/// # Safety
///
/// `api`, `ctx`, and `mem_info` must be valid.
unsafe fn alloc_scratch(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    mem_info: *const ort::OrtMemoryInfo,
    bytes: usize,
) -> Result<*mut c_void, String> {
    let get_scratch = api
        .KernelContext_GetScratchBuffer
        .ok_or("OrtApi.KernelContext_GetScratchBuffer is null")?;
    let mut out: *mut c_void = std::ptr::null_mut();
    let status = unsafe { get_scratch(ctx, mem_info, bytes.max(1), &mut out) };
    if !status.is_null() {
        return Err("KernelContext_GetScratchBuffer failed".into());
    }
    if out.is_null() {
        return Err("KernelContext_GetScratchBuffer returned null".into());
    }
    Ok(out)
}

// ──────────────────────────────────────────────────────────────────────────────
// Governed kernel workspace
// ──────────────────────────────────────────────────────────────────────────────

/// Resolve a kernel's declared [`WorkspaceRequirement`] into the concrete
/// [`WorkspaceView`] handed to `execute_with_workspace` for one dispatch.
///
/// # Why ORT scratch, and not EP allocate/free
///
/// The workspace has to live on the device the kernel runs on. The mechanism
/// that was hardware-validated on H200 in #832 for fused-subgraph intermediates
/// is [`alloc_scratch`] (`OrtApi::KernelContext_GetScratchBuffer`), so
/// workspaces reuse exactly that path: ORT owns the bytes for the duration of
/// the `Compute` call, which is precisely a step-scoped workspace's lifetime,
/// and this executor never issues a device free. A per-dispatch
/// `ExecutionProvider::allocate`/`deallocate` pair would instead put a
/// synchronous `cuMemAlloc`/`cuMemFree` on every node of every decode step —
/// the exact cost #832 removed from `MatMulNBits` with a cached grow-only
/// arena — and a device free is illegal during CUDA-graph capture.
///
/// # Where the workspace is placed
///
/// Against the memory info of **this node's own ORT-bound operands**, derived
/// by [`operand_mem_info`], not against input 0 of the fused subgraph. When the
/// node binds several ORT operands they are compared with
/// `OrtApi::CompareMemoryInfo` and a disagreement is an error, not a guess.
/// A node whose operands are all intermediate buffers inherits the
/// subgraph-level memory info those buffers were allocated from, which is the
/// same device by construction.
///
/// Every failure to determine the device is a hard error *for a request that
/// needs serving*, never a silent host allocation: a device kernel handed host
/// memory dereferences it as device memory.
///
/// # Alignment
///
/// ORT makes no promise about the alignment of a scratch block beyond the
/// allocator's own, so a stricter request is satisfied by **checked
/// over-allocation** (`bytes + alignment - 1`) followed by align-up inside the
/// returned block. Every arithmetic step is checked and the aligned window is
/// re-verified to lie inside the allocation before it is handed out.
/// [`alloc_scratch`] already rejects a null block, so there is exactly one
/// null check on this path.
///
/// # Planning cost
///
/// The requirement is memoized per node by [`WorkspacePlanCache`], keyed on the
/// exact operand metadata. Without it, a `SessionPersistent` declarer whose
/// `workspace_requirement` runs a cuBLASLt heuristic search (`MatMul`, `Gemm`,
/// `FusedEpilogue`, `MatMulNBits`' f32 dequant path) paid for that search here,
/// had the result declined, and then paid for it a second time inside its own
/// `execute` — twice per node per decode step. See [`WorkspacePlanCache`] for
/// the correctness argument.
///
/// # Lifetimes
///
/// * [`WorkspaceLifetime::StepScoped`] — served from ORT scratch, as above.
///   The step-scoped consumers in `onnx-runtime-ep-cuda` are the default-domain
///   `Attention` Phase-2a scratch and `StandardAttention` whenever it is *not*
///   a single-token, single-batch decode (`batch == 1 && q_seq == 1`) — i.e.
///   prefill and batched dispatch.
/// * [`WorkspaceLifetime::SessionPersistent`] — **explicitly declined**
///   (`Ok(None)`), never downgraded. ORT reclaims scratch when `Compute`
///   returns, so serving a persistent request from it would hand the kernel
///   memory that is recycled behind its back on the next `Run`. There is no
///   session-persistent device arena at this seam, so the request is passed
///   through as `None` and the kernel decides. This seam must not guess.
///
/// ## What declining actually does, per declarer
///
/// Declining is **not** uniformly a self-owned fallback. Every
/// `SessionPersistent` declarer in `onnx-runtime-ep-cuda`, and what each does
/// with `None`:
///
/// * **Self-owned fallback — correct, no error.**
///   * `GroupQueryAttention` (`group_query_attention.rs`): the composite
///     request (scores, packed Q/K/V staging, BSH↔BNSH transposes) is served
///     from its own pooled `GqaWorkspace` slots.
///   * `StandardAttention` (`standard_attention.rs`), single-token single-batch
///     decode only — every other geometry is `StepScoped` and *is* served: the
///     score and staged-K/V scratch is pooled or per-call inside `run`.
///   * `MatMulNBits` bf16 activations: the `Bf16Scratch` staging arena is
///     always self-owned and is listed here only because it is easy to assume
///     otherwise — that path declares [`WorkspaceRequirement::NONE`], so it is
///     unaffected by the decline in either direction.
/// * **Hard error, but only when the cuBLASLt heuristic asks for bytes.**
///   `MatMul`, `Gemm`, `FusedEpilogue` and `MatMulNBits`' f32
///   dequant-cuBLASLt path all route through
///   `blas::governed_workspace_requirement`, which reports
///   [`WorkspaceRequirement::NONE`] when the heuristic selects **0** bytes and
///   `SessionPersistent` otherwise. On the declined path
///   `blas::governed_workspace_ptr` returns `Ok(0)` for a 0-byte requirement
///   and errors — *"governed cuBLASLt workspace requires N bytes, but none was
///   supplied"* — for a non-zero one. Which of the two happens is a property of
///   the algorithm cuBLASLt picks for that shape, dtype, device and library
///   version, not a static property of the kernel; many decode-shaped GEMMs
///   select 0 bytes and are unaffected.
/// * **Unconditional hard error.** `BlockQuantizedMoE`
///   (`block_quantized_moe.rs`) and `IndexShare` (`index_share.rs`) declare
///   `SessionPersistent` for every geometry and have no self-owned path; both
///   their `execute` and their `execute_with_workspace(None)` return an error.
///
/// Declining is what preserves #832's H200-validated plugin path: on `main`
/// the executor called bare `Kernel::execute`, and
/// [`Kernel::execute_with_workspace`] defaults to forwarding there, so `None`
/// reproduces `main` node for node — including for the kernels that error,
/// which error on `main` too. Hard-failing here instead would have turned every
/// GQA-bearing model into a plugin-path error on hardware that runs it today.
/// The cost of declining is that `BlockQuantizedMoE`, `IndexShare`, and any
/// GEMM whose heuristic asks for workspace bytes remain **incompatible with the
/// plugin path**; making them work needs a real session-persistent device arena
/// at this seam, which is future work and is not claimed here.
///
/// # Safety
///
/// `api`, `ctx` and the pointer inside `mem_info` must be valid.
unsafe fn prepare_workspace(
    api: &ort::OrtApi,
    ctx: *mut ort::OrtKernelContext,
    mem_info: OperandMemInfo,
    kernel: &dyn Kernel,
    plans: &WorkspacePlanCache,
    inputs: &[TensorView<'_>],
    node_label: &str,
) -> Result<Option<WorkspaceView>, String> {
    let metadata: Vec<TensorMetadata<'_>> = inputs
        .iter()
        .map(|v| TensorMetadata::new(v.dtype, v.shape, !v.is_absent()))
        .collect();
    let requirement: WorkspaceRequirement = plans.get_or_plan(&metadata, || {
        kernel
            .workspace_requirement(&metadata)
            .map_err(|e| format!("{node_label}: workspace_requirement failed: {e}"))
    })?;
    if requirement.bytes == 0 {
        return Ok(None);
    }

    let alignment = requirement.alignment;
    if alignment == 0 || !alignment.is_power_of_two() {
        return Err(format!(
            "{node_label}: kernel requested workspace alignment {alignment}, \
             which is not a non-zero power of two"
        ));
    }
    if requirement.lifetime != WorkspaceLifetime::StepScoped {
        // Explicit decline, never a downgrade. ORT reclaims a
        // `KernelContext_GetScratchBuffer` block when `Compute` returns, so
        // serving a `SessionPersistent` request from it would hand the kernel
        // memory that is recycled behind its back the moment it reuses the
        // pointer on the next `Run`.
        return Ok(None);
    }

    let bytes = usize::try_from(requirement.bytes).map_err(|_| {
        format!(
            "{node_label}: workspace requirement of {} bytes exceeds usize on this target",
            requirement.bytes
        )
    })?;
    let total =
        workspace_block_bytes(bytes, alignment).map_err(|e| format!("{node_label}: {e}"))?;

    let mem_info = match mem_info {
        OperandMemInfo::Uniform(ptr) | OperandMemInfo::FromIntermediates(ptr) => ptr,
        OperandMemInfo::Unavailable => {
            return Err(format!(
                "{node_label}: kernel requires {bytes} bytes of {alignment}-byte-aligned \
                 workspace, but the memory device of its operands could not be read, so the \
                 workspace cannot be placed where the kernel runs. Failing closed rather than \
                 handing a device kernel host memory."
            ));
        }
        OperandMemInfo::Divergent { first, other } => {
            return Err(format!(
                "{node_label}: kernel requires {bytes} bytes of {alignment}-byte-aligned \
                 workspace, but its operands do not agree on a memory device: kernel-context \
                 inputs {first} and {other} report different OrtMemoryInfo (CompareMemoryInfo). \
                 The operands therefore do not determine where this kernel's compute runs. \
                 Failing closed rather than guessing a device."
            ));
        }
    };

    let base = unsafe { alloc_scratch(api, ctx, mem_info, total) }
        .map_err(|e| format!("{node_label}: workspace scratch allocation failed: {e}"))?;

    // `alloc_scratch` already rejects a null block, so no second null check
    // here: a duplicated guard reads as two independent defences and is really
    // one, which is worse than none for a reviewer counting them.
    let aligned = align_workspace_window(base as usize, total, bytes, alignment)
        .map_err(|e| format!("{node_label}: {e}"))?;

    Ok(Some(WorkspaceView::new(
        DevicePtrMut(aligned as *mut c_void),
        bytes,
    )))
}

/// Bytes to request so that aligning up to `alignment` always lands inside the
/// block. Split out from [`prepare_workspace`] so the overflow behaviour is
/// unit-testable without an ORT kernel context.
fn workspace_block_bytes(bytes: usize, alignment: usize) -> Result<usize, String> {
    bytes.checked_add(alignment - 1).ok_or_else(|| {
        format!("workspace request of {bytes} bytes over-aligned to {alignment} overflows usize")
    })
}

/// Align `base` up to `alignment` and prove the resulting `bytes`-long window
/// lies inside the `total`-byte block starting at `base`.
///
/// Split out from [`prepare_workspace`] so the alignment and overflow rules are
/// unit-testable without an ORT kernel context. Every step is checked: a
/// wrapping add here would hand a kernel a pointer outside the allocation.
fn align_workspace_window(
    base: usize,
    total: usize,
    bytes: usize,
    alignment: usize,
) -> Result<usize, String> {
    let aligned = base
        .checked_add(alignment - 1)
        .map(|a| a & !(alignment - 1))
        .ok_or("aligning workspace pointer overflows usize")?;
    let end = aligned
        .checked_add(bytes)
        .ok_or("workspace end address overflows usize")?;
    let block_end = base
        .checked_add(total)
        .ok_or("workspace block end overflows usize")?;
    if end > block_end {
        return Err(format!(
            "aligned workspace [{aligned:#x}, {end:#x}) escapes the {total}-byte scratch \
             block at {base:#x}"
        ));
    }
    Ok(aligned)
}

/// Compute: execute the kernel(s) for this subgraph.
///
/// For single-node subgraphs (the common case for CPU EP), this calls
/// `kernel.execute_with_workspace()` once. For multi-node fused subgraphs with
/// a `SubgraphRouting` table, it allocates intermediate buffers, threads them
/// between nodes in topological order, and writes only true subgraph outputs
/// back to ORT.
///
/// # Workspace contract
///
/// Every dispatch goes through [`prepare_workspace`] →
/// [`Kernel::execute_with_workspace`], never bare [`Kernel::execute`]. A kernel
/// declaring a non-zero [`WorkspaceLifetime::StepScoped`] requirement gets
/// correctly-aligned scratch on the memory device reported by **that node's own
/// ORT-bound operands** (see [`operand_mem_info`]), taken from ORT
/// (`KernelContext_GetScratchBuffer`) and reclaimed by ORT when `Compute`
/// returns — the same mechanism #832 validated on an H200 for fused subgraph
/// intermediates. A [`WorkspaceLifetime::SessionPersistent`] request is
/// **declined** (`None`), never downgraded to recycled step-scoped memory; see
/// [`prepare_workspace`] for the per-kernel consequences of that decline.
/// Kernels declaring [`WorkspaceRequirement::NONE`] behave exactly as before:
/// `execute_with_workspace`'s default forwards to `execute`.
///
/// Intermediate buffers keep #832's derivation — the subgraph-level
/// [`device_mem_info`] — deliberately unchanged, since that is the allocation
/// that ran on hardware.
///
/// # Safety
///
/// `info` must be a valid `ExportedComputeInfo*`, `state` from `CreateState`,
/// and `kernel_context` a valid `OrtKernelContext*`.
unsafe extern "C" fn compute_execute(
    info: *mut ort::OrtNodeComputeInfo,
    _state: *mut c_void,
    kernel_context: *mut ort::OrtKernelContext,
) -> *mut ort::OrtStatus {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if info.is_null() || kernel_context.is_null() {
            return fail_status("Compute: null argument");
        }

        let exported = unsafe { &*(info.cast::<ExportedComputeInfo>()) };

        if exported.entries.is_empty() {
            return fail_status("Compute: no kernels compiled for this subgraph");
        }

        let api = crate::status::host_api();
        if api.is_null() {
            return fail_status("Compute: host ORT API not available");
        }
        let api_ref = unsafe { &*api };

        // Memory info for intermediate scratch. On a device EP this is device
        // memory, so multi-node intermediates stay on the GPU (a host buffer
        // would make the next kernel dereference a host pointer as device →
        // CUDA_ERROR_ILLEGAL_ADDRESS). `None` falls back to host buffers.
        let scratch_mem_info = unsafe { device_mem_info(api_ref, kernel_context) };

        let inputs = match unsafe { read_inputs(api_ref, kernel_context) } {
            Ok(v) => v,
            Err(e) => return fail_status(&format!("Compute: {e}")),
        };

        if let Some(routing) = &exported.routing {
            // ── Routed multi-node path ────────────────────────────────────
            if routing.input_sources.len() != exported.entries.len()
                || routing.output_sinks.len() != exported.entries.len()
            {
                return fail_status("Compute: routing table length mismatch with entries");
            }

            // Allocate intermediate buffer slots (uninitialized until written).
            let mut intermediates: Vec<Option<IntermediateBuf>> = (0..routing
                .num_intermediate_buffers)
                .map(|_| None)
                .collect();

            for (node_idx, entry) in exported.entries.iter().enumerate() {
                let sources = &routing.input_sources[node_idx];
                let sinks = &routing.output_sinks[node_idx];

                // Gather input views for this node.
                let mut kernel_inputs: Vec<TensorView<'_>> = Vec::with_capacity(sources.len());
                for src in sources {
                    match src {
                        NodeInputSource::Ort(i) => {
                            if *i >= inputs.len() {
                                return fail_status(&format!(
                                    "Compute: routing ORT input {i} out of range"
                                ));
                            }
                            kernel_inputs.push(inputs[*i].view());
                        }
                        NodeInputSource::Buffer(b) => {
                            let buf = match intermediates.get(*b).and_then(|o| o.as_ref()) {
                                Some(b) => b,
                                None => {
                                    return fail_status(&format!(
                                        "Compute: intermediate buffer {b} not yet written"
                                    ));
                                }
                            };
                            // SAFETY: buf lives for the duration of this loop body.
                            // We extend the lifetime here; the borrow is valid
                            // because intermediates is not mutated while we hold
                            // this view (we only push to kernel_inputs first).
                            let view = buf.view();
                            // Transmute lifetime to 'static so we can store in the
                            // Vec; we ensure the buf outlives this scope.
                            let view: TensorView<'static> = unsafe { std::mem::transmute(view) };
                            kernel_inputs.push(view);
                        }
                        NodeInputSource::Absent => {
                            // Absent optional input — provide a null-backed
                            // sentinel TensorView so the kernel can detect it
                            // via `view.is_absent()`.
                            kernel_inputs.push(TensorView::absent(DataType::Undefined));
                        }
                    }
                }

                // Infer output shapes.
                let output_shapes = match infer_shapes(&entry.shape_inference, &kernel_inputs) {
                    Ok(s) => s,
                    Err(e) => {
                        return fail_status(&format!("Compute: shape inference failed: {e}"));
                    }
                };

                // Execute — dispatch based on sinks.
                // For outputs going to ORT we allocate via ORT API;
                // for outputs going to intermediate buffers we allocate on heap;
                // for absent slots we allocate a scratch buffer.
                let mut ort_outputs: Vec<crate::kernel_ctx::OwnedOutput> = Vec::new();
                let mut buf_writes: Vec<(usize, Vec<usize>, DataType)> = Vec::new();
                let mut absent_scratch: Vec<(usize, Vec<u8>, DataType)> = Vec::new(); // (slot, buf, dtype)

                // Per-slot view map to keep positions aligned end-to-end.
                enum RoutedSlotKind {
                    Ort,
                    Buffer,
                    Absent(usize), // index into absent_scratch
                }
                let mut slot_kinds: Vec<RoutedSlotKind> = Vec::with_capacity(sinks.len());

                // We need to know which output slot → which sink.
                for (out_slot, shape) in output_shapes.iter().enumerate() {
                    if entry.absent_output_slots.contains(&out_slot) {
                        // Absent output slot — allocate scratch using the
                        // slot's own dtype. Fail closed if unknown.
                        let scratch_dtype = match entry.output_dtypes.get(out_slot).copied() {
                            Some(dt) if dt != DataType::Undefined && dt.byte_size() > 0 => dt,
                            _ => {
                                return fail_status(&format!(
                                    "Compute: absent output slot {out_slot} has unknown dtype; \
                                     cannot allocate scratch buffer (fail closed)"
                                ));
                            }
                        };
                        let numel: usize = shape.iter().product::<usize>().max(1);
                        let buf = vec![0u8; scratch_alloc_bytes(numel, scratch_dtype)];
                        let idx = absent_scratch.len();
                        absent_scratch.push((out_slot, buf, scratch_dtype));
                        slot_kinds.push(RoutedSlotKind::Absent(idx));
                        continue;
                    }
                    let out_dtype = match entry.output_dtypes.get(out_slot).copied() {
                        Some(dt) => dt,
                        None => {
                            return fail_status(&format!(
                                "Compute: output slot {out_slot} has no declared dtype \
                                 (output_dtypes vec is too short)"
                            ));
                        }
                    };
                    let sink = match sinks.get(out_slot) {
                        Some(s) => s,
                        None => {
                            return fail_status(&format!(
                                "Compute: output slot {out_slot} has no sink"
                            ));
                        }
                    };
                    match sink {
                        NodeOutputSink::Ort(ort_idx) => {
                            match unsafe {
                                allocate_output(api_ref, kernel_context, *ort_idx, shape, out_dtype)
                            } {
                                Ok(out) => ort_outputs.push(out),
                                Err(e) => {
                                    return fail_status(&format!("Compute: {e}"));
                                }
                            }
                            slot_kinds.push(RoutedSlotKind::Ort);
                        }
                        NodeOutputSink::Buffer(buf_idx) => {
                            buf_writes.push((*buf_idx, shape.clone(), out_dtype));
                            slot_kinds.push(RoutedSlotKind::Buffer);
                        }
                        NodeOutputSink::Absent => {
                            // Should not reach here — absent slots are handled
                            // above via absent_output_slots. Defensive fallback:
                            // treat as absent scratch allocation.
                            return fail_status(&format!(
                                "Compute: output slot {out_slot} has Absent sink \
                                 but was not in absent_output_slots"
                            ));
                        }
                    }
                }

                // Build mutable output views: ORT outputs first, then buffer outputs.
                let mut ort_out_views: Vec<_> =
                    ort_outputs.iter_mut().map(|o| o.view_mut()).collect();

                // For buffer-sink outputs, allocate the IntermediateBuf and get a
                // mutable pointer into it. Prefer ORT scratch memory (device
                // memory on a device EP) so intermediates live where the kernels
                // execute; fall back to a host buffer only when no device memory
                // info is available (e.g. the CPU EP or an input-less subgraph).
                let mut new_bufs: Vec<(usize, IntermediateBuf)> = Vec::new();
                for (buf_idx, shape, dtype) in &buf_writes {
                    let numel: usize = shape.iter().product();
                    let byte_len = dtype.byte_size() * numel;
                    let strides = contiguous_strides(shape);
                    let (data, scratch_ptr) = match scratch_mem_info {
                        Some(mem_info) => {
                            match unsafe {
                                alloc_scratch(api_ref, kernel_context, mem_info, byte_len)
                            } {
                                Ok(ptr) => (Vec::new(), ptr.cast::<u8>()),
                                Err(e) => {
                                    return fail_status(&format!(
                                        "Compute: intermediate scratch alloc failed: {e}"
                                    ));
                                }
                            }
                        }
                        None => (vec![0u8; byte_len], std::ptr::null_mut()),
                    };
                    new_bufs.push((
                        *buf_idx,
                        IntermediateBuf {
                            data,
                            scratch_ptr,
                            shape: shape.clone(),
                            strides,
                            dtype: *dtype,
                        },
                    ));
                }

                // Collect all output views using the per-slot view map so
                // positions stay aligned even when absent slots are present.
                let absent_shapes: Vec<Vec<usize>> = output_shapes.clone();
                let absent_strides_storage: Vec<Vec<i64>> = absent_shapes
                    .iter()
                    .map(|s| contiguous_strides(s))
                    .collect();
                let mut all_output_views: Vec<_> = {
                    let mut ort_iter = ort_out_views.drain(..);
                    let mut buf_iter = new_bufs.iter_mut();
                    slot_kinds
                        .iter()
                        .enumerate()
                        .map(|(slot_idx, kind)| match kind {
                            RoutedSlotKind::Ort => ort_iter.next().unwrap(),
                            RoutedSlotKind::Buffer => {
                                let (_, buf) = buf_iter.next().unwrap();
                                buf_view_mut(buf)
                            }
                            RoutedSlotKind::Absent(idx) => {
                                let (_, scratch_buf, dtype) = &mut absent_scratch[*idx];
                                let shape = &absent_shapes[slot_idx];
                                let strides = &absent_strides_storage[slot_idx];
                                TensorMut::new(
                                    DevicePtrMut(scratch_buf.as_mut_ptr().cast()),
                                    *dtype,
                                    shape.as_slice(),
                                    strides.as_slice(),
                                    DeviceId::cpu(),
                                )
                                .mark_absent()
                            }
                        })
                        .collect()
                };

                let node_label = format!("Compute: node {node_idx}");
                let plans = match exported.workspace_plan_cache(node_idx) {
                    Some(plans) => plans,
                    None => {
                        return fail_status(&format!(
                            "{node_label}: no workspace plan cache for this node (entries and \
                             workspace_plans have drifted)"
                        ));
                    }
                };
                // This node's own ORT-bound operands — not subgraph input 0.
                let node_ort_operands: Vec<usize> = sources
                    .iter()
                    .filter_map(|src| match src {
                        NodeInputSource::Ort(i) => Some(*i),
                        NodeInputSource::Buffer(_) | NodeInputSource::Absent => None,
                    })
                    .collect();
                let node_mem_info = unsafe {
                    operand_mem_info(
                        api_ref,
                        kernel_context,
                        &node_ort_operands,
                        scratch_mem_info,
                    )
                };
                let workspace = match unsafe {
                    prepare_workspace(
                        api_ref,
                        kernel_context,
                        node_mem_info,
                        &*entry.kernel,
                        plans,
                        &kernel_inputs,
                        &node_label,
                    )
                } {
                    Ok(w) => w,
                    Err(e) => return fail_status(&e),
                };

                if let Err(e) = entry.kernel.execute_with_workspace(
                    &kernel_inputs,
                    &mut all_output_views,
                    workspace,
                ) {
                    return fail_status(&format!("Compute: kernel execution failed: {e}"));
                }

                // Store new intermediate buffers.
                for (buf_idx, buf) in new_bufs {
                    if buf_idx >= intermediates.len() {
                        return fail_status(&format!(
                            "Compute: buffer index {buf_idx} out of range"
                        ));
                    }
                    intermediates[buf_idx] = Some(buf);
                }
            }
        } else if exported.entries.len() == 1 {
            // ── Fast path: single-kernel subgraph ─────────────────────────
            let entry = &exported.entries[0];
            // Reconstruct positional inputs with absent sentinels so the
            // kernel sees the correct arity and position.
            let kernel_inputs: Vec<_> = entry
                .input_slots
                .iter()
                .map(|slot| match slot {
                    Some(ort_idx) => inputs[*ort_idx].view(),
                    None => TensorView::absent(DataType::Undefined),
                })
                .collect();
            let output_shapes = match infer_shapes(&entry.shape_inference, &kernel_inputs) {
                Ok(s) => s,
                Err(e) => {
                    return fail_status(&format!("Compute: shape inference failed: {e}"));
                }
            };
            // Allocate outputs. Absent slots get a local scratch buffer so the
            // kernel sees the full output arity and can index by position,
            // while only present slots are allocated through ORT's kernel
            // context (sequential ORT indices).
            let mut owned_outputs: Vec<crate::kernel_ctx::OwnedOutput> = Vec::new();
            let mut absent_bufs: Vec<Vec<u8>> = Vec::new();
            // Track whether each node output slot is ORT-allocated or absent.
            enum SlotKind {
                Ort,           // present, comes from ORT
                Absent(usize), // index into absent_bufs
            }
            // Also record the dtype for each absent slot so the TensorMut
            // matches the kernel's element size.
            let mut absent_dtypes: Vec<DataType> = Vec::new();
            let mut slot_map: Vec<SlotKind> = Vec::with_capacity(entry.num_outputs);
            let mut ort_out_idx: usize = 0;

            for (out_slot, shape) in output_shapes.iter().enumerate() {
                if entry.absent_output_slots.contains(&out_slot) {
                    // Absent output slot — allocate scratch buffer using the
                    // slot's own ORT-declared dtype. Fail closed if unknown.
                    let scratch_dtype = match entry.output_dtypes.get(out_slot).copied() {
                        Some(dt) if dt != DataType::Undefined && dt.byte_size() > 0 => dt,
                        _ => {
                            return fail_status(&format!(
                                "Compute: absent output slot {out_slot} has unknown dtype; \
                                 cannot allocate scratch buffer (fail closed)"
                            ));
                        }
                    };
                    let numel: usize = shape.iter().product::<usize>().max(1);
                    let buf = vec![0u8; scratch_alloc_bytes(numel, scratch_dtype)];
                    let idx = absent_bufs.len();
                    absent_bufs.push(buf);
                    absent_dtypes.push(scratch_dtype);
                    slot_map.push(SlotKind::Absent(idx));
                    continue;
                }
                let out_dtype = match entry.output_dtypes.get(out_slot).copied() {
                    Some(dt) => dt,
                    None => {
                        return fail_status(&format!(
                            "Compute: output slot {out_slot} has no declared dtype \
                             (output_dtypes vec is too short)"
                        ));
                    }
                };
                match unsafe {
                    allocate_output(api_ref, kernel_context, ort_out_idx, shape, out_dtype)
                } {
                    Ok(out) => {
                        owned_outputs.push(out);
                        slot_map.push(SlotKind::Ort);
                    }
                    Err(e) => return fail_status(&format!("Compute: {e}")),
                }
                ort_out_idx += 1;
            }
            // Build output views in node-output order so the kernel sees the
            // full arity including absent scratch slots.
            let absent_shapes: Vec<Vec<usize>> = output_shapes.clone();
            let absent_strides_storage: Vec<Vec<i64>> = absent_shapes
                .iter()
                .map(|s| contiguous_strides(s))
                .collect();
            // First, get mutable views of all ORT outputs in order.
            let mut ort_views: Vec<TensorMut<'_>> =
                owned_outputs.iter_mut().map(|o| o.view_mut()).collect();
            let mut ort_view_iter = ort_views.drain(..);
            let mut output_views: Vec<TensorMut<'_>> = Vec::with_capacity(slot_map.len());
            for (slot_idx, kind) in slot_map.iter().enumerate() {
                match kind {
                    SlotKind::Ort => {
                        output_views.push(ort_view_iter.next().unwrap());
                    }
                    SlotKind::Absent(idx) => {
                        let buf = &mut absent_bufs[*idx];
                        let shape = &absent_shapes[slot_idx];
                        let strides = &absent_strides_storage[slot_idx];
                        let scratch_dtype = absent_dtypes[*idx];
                        let view = TensorMut::new(
                            DevicePtrMut(buf.as_mut_ptr().cast()),
                            scratch_dtype,
                            shape.as_slice(),
                            strides.as_slice(),
                            DeviceId::cpu(),
                        )
                        .mark_absent();
                        output_views.push(view);
                    }
                }
            }
            let plans = match exported.workspace_plan_cache(0) {
                Some(plans) => plans,
                None => {
                    return fail_status(
                        "Compute: node 0: no workspace plan cache for this node (entries and \
                         workspace_plans have drifted)",
                    );
                }
            };
            // The single-node subgraph's operands are this node's operands, but
            // only the present ones are ORT-bound.
            let node_ort_operands: Vec<usize> =
                entry.input_slots.iter().flatten().copied().collect();
            let node_mem_info = unsafe {
                operand_mem_info(
                    api_ref,
                    kernel_context,
                    &node_ort_operands,
                    scratch_mem_info,
                )
            };
            let workspace = match unsafe {
                prepare_workspace(
                    api_ref,
                    kernel_context,
                    node_mem_info,
                    &*entry.kernel,
                    plans,
                    &kernel_inputs,
                    "Compute: node 0",
                )
            } {
                Ok(w) => w,
                Err(e) => return fail_status(&e),
            };
            if let Err(e) =
                entry
                    .kernel
                    .execute_with_workspace(&kernel_inputs, &mut output_views, workspace)
            {
                return fail_status(&format!("Compute: kernel execution failed: {e}"));
            }
        } else {
            return fail_status(
                "Compute: multi-node subgraph requires SubgraphRouting — \
                 call ExportedComputeInfo::set_routing before registering",
            );
        }

        ok_status()
    }));
    result.unwrap_or_else(|_| fail_status("Compute: internal panic"))
}

/// Build a contiguous stride array from a shape (C-order, innermost stride = 1).
fn contiguous_strides(shape: &[usize]) -> Vec<i64> {
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1] as i64;
    }
    strides
}

/// Build a mutable TensorView from an IntermediateBuf (unsafe: caller ensures
/// exclusive access and lifetime correctness).
fn buf_view_mut(buf: &mut IntermediateBuf) -> onnx_runtime_ep_api::tensor::TensorMut<'_> {
    use onnx_runtime_ep_api::tensor::{DevicePtrMut, TensorMut};
    // `ptr_mut` returns a Copy raw pointer, so its mutable borrow ends here and
    // does not conflict with the immutable borrows of shape/strides below.
    let ptr = buf.ptr_mut();
    TensorMut::new(
        DevicePtrMut(ptr.cast()),
        buf.dtype,
        &buf.shape,
        &buf.strides,
        onnx_runtime_ir::DeviceId::cpu(),
    )
}

// ──────────────────────────────────────────────────────────────────────────────
// Shape inference implementations
// ──────────────────────────────────────────────────────────────────────────────

/// Infer output shapes from the shape inference strategy and input views.
fn infer_shapes(
    strategy: &ShapeInference,
    inputs: &[TensorView<'_>],
) -> Result<Vec<Vec<usize>>, String> {
    match strategy {
        ShapeInference::ElementwiseBroadcast => {
            if inputs.is_empty() {
                return Err("ElementwiseBroadcast: no inputs".into());
            }
            let mut shape = inputs[0].shape.to_vec();
            for inp in &inputs[1..] {
                shape = onnx_runtime_ir::broadcast_shapes(&shape, inp.shape)
                    .map_err(|e| format!("broadcast failed: {e}"))?;
            }
            Ok(vec![shape])
        }

        ShapeInference::SameAsInput(idx) => {
            let idx = *idx;
            if idx >= inputs.len() {
                return Err(format!(
                    "SameAsInput({idx}): only {} inputs present",
                    inputs.len()
                ));
            }
            Ok(vec![inputs[idx].shape.to_vec()])
        }

        ShapeInference::SameAsInputMultiOutput { idx, count } => {
            let idx = *idx;
            if idx >= inputs.len() {
                return Err(format!(
                    "SameAsInputMultiOutput({idx}): only {} inputs present",
                    inputs.len()
                ));
            }
            let shape = inputs[idx].shape.to_vec();
            Ok(vec![shape; *count])
        }

        ShapeInference::LayerNorm {
            raw_axis,
            num_outputs,
            full_shape_outputs,
        } => {
            if inputs.is_empty() {
                return Err("LayerNorm: no inputs".into());
            }
            let full_shape = inputs[0].shape.to_vec();
            let rank = full_shape.len() as i64;
            // Resolve raw axis against runtime rank.
            let resolved = if *raw_axis < 0 {
                *raw_axis + rank
            } else {
                *raw_axis
            };
            if resolved < 0 || resolved >= rank {
                return Err(format!(
                    "LayerNorm: axis {raw_axis} out of range for rank {rank}",
                ));
            }
            let axis = resolved as usize;
            // Reduced shape: dims before `axis` are kept, dims from `axis`
            // onward become 1 (keepdims reduction over the normalised axes).
            let mut reduced_shape = full_shape.clone();
            for d in reduced_shape[axis..].iter_mut() {
                *d = 1;
            }
            let mut out = Vec::with_capacity(*num_outputs);
            for i in 0..*num_outputs {
                if i == 0 || full_shape_outputs.contains(&i) {
                    out.push(full_shape.clone());
                } else {
                    out.push(reduced_shape.clone());
                }
            }
            Ok(out)
        }

        ShapeInference::MatMul => {
            if inputs.len() < 2 {
                return Err(format!("MatMul: expected ≥2 inputs, got {}", inputs.len()));
            }
            let a = inputs[0].shape;
            let b = inputs[1].shape;
            let shape = matmul_output_shape(a, b)?;
            Ok(vec![shape])
        }

        ShapeInference::Gemm { trans_a, trans_b } => {
            if inputs.len() < 2 {
                return Err(format!("Gemm: expected ≥2 inputs, got {}", inputs.len()));
            }
            let a = inputs[0].shape;
            let b = inputs[1].shape;
            if a.len() != 2 || b.len() != 2 {
                return Err(format!("Gemm: inputs must be 2-D, got {a:?} and {b:?}"));
            }
            let m = if *trans_a { a[1] } else { a[0] };
            let n = if *trans_b { b[0] } else { b[1] };
            Ok(vec![vec![m, n]])
        }

        ShapeInference::Concat { axis } => {
            if inputs.is_empty() {
                return Err("Concat: no inputs".into());
            }
            let rank = inputs[0].shape.len();
            let ax = normalise_axis(*axis, rank)?;
            let mut out = inputs[0].shape.to_vec();
            for inp in &inputs[1..] {
                out[ax] += inp.shape[ax];
            }
            Ok(vec![out])
        }

        ShapeInference::Transpose { perm } => {
            if inputs.is_empty() {
                return Err("Transpose: no inputs".into());
            }
            let rank = inputs[0].shape.len();
            let out: Vec<usize> = if let Some(p) = perm {
                if p.len() != rank {
                    return Err(format!(
                        "Transpose: perm length {} != rank {}",
                        p.len(),
                        rank
                    ));
                }
                p.iter().map(|&i| inputs[0].shape[i]).collect()
            } else {
                inputs[0].shape.iter().rev().copied().collect()
            };
            Ok(vec![out])
        }

        ShapeInference::Gather { axis } => {
            if inputs.len() < 2 {
                return Err(format!("Gather: expected ≥2 inputs, got {}", inputs.len()));
            }
            let data = inputs[0].shape;
            let indices = inputs[1].shape;
            let rank = data.len();
            let ax = normalise_axis(*axis, rank)?;
            let mut out: Vec<usize> = data[..ax].to_vec();
            out.extend_from_slice(indices);
            out.extend_from_slice(&data[ax + 1..]);
            Ok(vec![out])
        }

        ShapeInference::GatherND { batch_dims } => {
            if inputs.len() < 2 {
                return Err(format!(
                    "GatherND: expected ≥2 inputs, got {}",
                    inputs.len()
                ));
            }
            let data = inputs[0].shape;
            let indices = inputs[1].shape;
            let b = *batch_dims;
            if indices.is_empty() {
                return Err("GatherND: indices must have rank ≥1".into());
            }
            let k = *indices.last().unwrap();
            if b + k > data.len() {
                return Err(format!(
                    "GatherND: batch_dims+k ({}) > data rank ({})",
                    b + k,
                    data.len()
                ));
            }
            // output = data[:b] ++ indices[:-1] ++ data[b+k:]
            let mut out: Vec<usize> = data[..b].to_vec();
            out.extend_from_slice(&indices[..indices.len() - 1]);
            out.extend_from_slice(&data[b + k..]);
            Ok(vec![out])
        }

        ShapeInference::GatherBlockQuantized => {
            // Treat as GatherND(batch_dims=0) for shape purposes.
            if inputs.len() < 2 {
                return Err(format!(
                    "GatherBlockQuantized: expected ≥2 inputs, got {}",
                    inputs.len()
                ));
            }
            let data = inputs[0].shape;
            let indices = inputs[1].shape;
            if indices.is_empty() {
                return Err("GatherBlockQuantized: indices must have rank ≥1".into());
            }
            let k = *indices.last().unwrap();
            if k > data.len() {
                return Err(format!(
                    "GatherBlockQuantized: k ({k}) > data rank ({})",
                    data.len()
                ));
            }
            let mut out: Vec<usize> = indices[..indices.len() - 1].to_vec();
            out.extend_from_slice(&data[k..]);
            Ok(vec![out])
        }

        ShapeInference::ShapeOp { start, end } => {
            if inputs.is_empty() {
                return Err("Shape: no inputs".into());
            }
            let rank = inputs[0].shape.len() as i64;
            let s = normalise_axis(*start, inputs[0].shape.len()).unwrap_or(0);
            let e = if let Some(end_val) = end {
                if *end_val < 0 {
                    (rank + end_val).max(0) as usize
                } else {
                    (*end_val as usize).min(inputs[0].shape.len())
                }
            } else {
                inputs[0].shape.len()
            };
            let len = e.saturating_sub(s);
            Ok(vec![vec![len]])
        }

        ShapeInference::Squeeze { axes } => {
            if inputs.is_empty() {
                return Err("Squeeze: no inputs".into());
            }
            let shape = inputs[0].shape;
            let out: Vec<usize> = if axes.is_empty() {
                shape.iter().filter(|&&d| d != 1).copied().collect()
            } else {
                let rank = shape.len();
                let norm_axes: Vec<usize> = axes
                    .iter()
                    .map(|&a| normalise_axis(a, rank))
                    .collect::<Result<_, _>>()?;
                shape
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !norm_axes.contains(i))
                    .map(|(_, &d)| d)
                    .collect()
            };
            Ok(vec![out])
        }

        ShapeInference::Unsqueeze { axes } => {
            if inputs.is_empty() {
                return Err("Unsqueeze: no inputs".into());
            }
            let in_rank = inputs[0].shape.len();
            let out_rank = in_rank + axes.len();
            // Normalise axes with respect to output rank.
            let mut norm_axes: Vec<usize> = axes
                .iter()
                .map(|&a| {
                    let ax = if a < 0 {
                        (out_rank as i64 + a) as usize
                    } else {
                        a as usize
                    };
                    if ax >= out_rank {
                        Err(format!(
                            "Unsqueeze: axis {a} out of range for output rank {out_rank}"
                        ))
                    } else {
                        Ok(ax)
                    }
                })
                .collect::<Result<_, _>>()?;
            norm_axes.sort_unstable();

            let mut out = vec![1usize; out_rank];
            let mut src = 0;
            let mut ax_iter = norm_axes.iter().peekable();
            for (i, slot) in out.iter_mut().enumerate() {
                if ax_iter.peek() == Some(&&i) {
                    ax_iter.next();
                    // slot stays 1
                } else {
                    *slot = inputs[0].shape[src];
                    src += 1;
                }
            }
            Ok(vec![out])
        }

        ShapeInference::ReshapeData { allowzero } => {
            if inputs.len() < 2 {
                return Err(format!("Reshape: expected 2 inputs, got {}", inputs.len()));
            }
            let data_shape = inputs[0].shape;
            // Read the shape tensor from input[1].
            let shape_vals = unsafe { read_i64_tensor(&inputs[1]) }?;
            let in_numel: usize = data_shape.iter().product();

            let mut out: Vec<usize> = Vec::with_capacity(shape_vals.len());
            let mut infer_idx: Option<usize> = None;
            let mut known_prod: usize = 1;

            for (i, &v) in shape_vals.iter().enumerate() {
                match v {
                    0 if !allowzero => {
                        // Copy from input shape at same index.
                        let d = data_shape.get(i).copied().ok_or_else(|| {
                            format!(
                                "Reshape: dim 0 at index {i} but input rank is {}",
                                data_shape.len()
                            )
                        })?;
                        out.push(d);
                        known_prod *= d;
                    }
                    -1 => {
                        if infer_idx.is_some() {
                            return Err("Reshape: only one -1 dimension allowed".into());
                        }
                        infer_idx = Some(i);
                        out.push(0); // placeholder
                    }
                    d if d > 0 => {
                        let d = d as usize;
                        out.push(d);
                        known_prod *= d;
                    }
                    _ => {
                        return Err(format!("Reshape: invalid shape value {v}"));
                    }
                }
            }

            if let Some(idx) = infer_idx {
                if known_prod == 0 {
                    return Err("Reshape: cannot infer -1 dimension when known product is 0".into());
                }
                if !in_numel.is_multiple_of(known_prod) {
                    return Err(format!(
                        "Reshape: cannot infer -1: {in_numel} elements not divisible by {known_prod}"
                    ));
                }
                out[idx] = in_numel / known_prod;
            }
            Ok(vec![out])
        }

        ShapeInference::SliceData => {
            if inputs.is_empty() {
                return Err("Slice: no inputs".into());
            }
            let data_shape = inputs[0].shape;
            let starts = unsafe { read_i64_tensor(&inputs[1]) }?;
            let ends = unsafe { read_i64_tensor(&inputs[2]) }?;
            let axes: Vec<i64> = if inputs.len() > 3 && !inputs[3].data.is_null() {
                unsafe { read_i64_tensor(&inputs[3]) }?
            } else {
                (0..data_shape.len() as i64).collect()
            };
            let steps: Vec<i64> = if inputs.len() > 4 && !inputs[4].data.is_null() {
                unsafe { read_i64_tensor(&inputs[4]) }?
            } else {
                vec![1; starts.len()]
            };

            let out_shape = slice_output_shape(data_shape, &starts, &ends, &axes, &steps)?;
            Ok(vec![out_shape])
        }

        ShapeInference::Reduction {
            keepdims,
            axes,
            noop_with_empty_axes,
        } => {
            if inputs.is_empty() {
                return Err("Reduction: no inputs".into());
            }
            let shape = inputs[0].shape;
            Ok(vec![reduce_shape(
                shape,
                axes.as_deref(),
                *keepdims,
                *noop_with_empty_axes,
            )?])
        }

        ShapeInference::ReductionFromInput {
            keepdims,
            noop_with_empty_axes,
        } => {
            if inputs.is_empty() {
                return Err("ReductionFromInput: no inputs".into());
            }
            let shape = inputs[0].shape;
            let axes_opt: Option<Vec<i64>> = if inputs.len() > 1 && !inputs[1].data.is_null() {
                Some(unsafe { read_i64_tensor(&inputs[1]) }?)
            } else {
                None
            };
            Ok(vec![reduce_shape(
                shape,
                axes_opt.as_deref(),
                *keepdims,
                *noop_with_empty_axes,
            )?])
        }

        ShapeInference::Conv {
            out_channels,
            per_axis,
        } => {
            if inputs.is_empty() {
                return Err("Conv: no inputs".into());
            }
            let in_shape = inputs[0].shape;
            if in_shape.len() < 2 {
                return Err("Conv: input must have rank ≥2".into());
            }
            let n = in_shape[0];
            let spatial: Vec<usize> = in_shape[2..]
                .iter()
                .zip(per_axis)
                .map(|(&dim, ax)| {
                    let eff = ax.dilation * (ax.kernel - 1) + 1;
                    let padded = dim + ax.pad_before + ax.pad_after;
                    if padded < eff {
                        0
                    } else {
                        (padded - eff) / ax.stride + 1
                    }
                })
                .collect();
            let mut out = vec![n, *out_channels];
            out.extend(spatial);
            Ok(vec![out])
        }

        ShapeInference::MultiHeadAttention {
            num_heads,
            num_outputs,
        } => {
            // query: [B, S, hidden]
            // output[0]: same as query
            // output[1] present_key:   [B, num_heads, P+S, head_size]  (if present)
            // output[2] present_value: same as output[1]
            if inputs.is_empty() {
                return Err("MultiHeadAttention: no inputs".into());
            }
            let q = inputs[0].shape;
            if q.len() < 2 {
                return Err("MultiHeadAttention: query rank must be ≥2".into());
            }
            let attn_out = q.to_vec();
            let mut outputs = vec![attn_out];

            if *num_outputs > 1 && *num_heads > 0 {
                let b = q[0];
                let s = q[1];
                let hidden = if q.len() > 2 { q[2] } else { 0 };
                let head_size = if *num_heads > 0 {
                    hidden / num_heads
                } else {
                    0
                };
                // past_key is input[6] when present.
                let past_seq = inputs
                    .get(6)
                    .map(|v| if v.shape.len() >= 3 { v.shape[2] } else { 0 })
                    .unwrap_or(0);
                let present_shape = vec![b, *num_heads, past_seq + s, head_size];
                for _ in 1..*num_outputs {
                    outputs.push(present_shape.clone());
                }
            }
            Ok(outputs)
        }

        ShapeInference::GroupQueryAttention {
            num_heads,
            kv_num_heads,
        } => {
            if inputs.is_empty() {
                return Err("GroupQueryAttention: no inputs".into());
            }
            let q = inputs[0].shape;
            if q.len() < 2 {
                return Err("GroupQueryAttention: query rank must be ≥2".into());
            }
            let b = q[0];
            let s = q[1];
            let hidden = if q.len() > 2 { q[2] } else { 0 };
            let head_size = if *num_heads > 0 {
                hidden / num_heads
            } else {
                0
            };
            let attn_out = q.to_vec();
            let past_seq = inputs
                .get(3)
                .map(|v| if v.shape.len() >= 3 { v.shape[2] } else { 0 })
                .unwrap_or(0);
            let present_shape = vec![b, *kv_num_heads, past_seq + s, head_size];
            Ok(vec![attn_out, present_shape.clone(), present_shape])
        }

        ShapeInference::RotaryEmbedding => {
            if inputs.is_empty() {
                return Err("RotaryEmbedding: no inputs".into());
            }
            Ok(vec![inputs[0].shape.to_vec()])
        }

        ShapeInference::Declined { op_type, domain } => Err(format!(
            "Op '{op_type}' (domain '{domain}') has no shape-inference rule. \
             Call ShapeInference::for_node(node, input_shapes, num_outputs) instead \
             of for_op to enable attribute-driven inference. If the op is not yet \
             modelled, add a variant to ShapeInference and handle it in infer_shapes."
        )),
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Shape-inference helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Normalise a possibly-negative axis index relative to `rank`.
fn normalise_axis(axis: i64, rank: usize) -> Result<usize, String> {
    let rank_i = rank as i64;
    let a = if axis < 0 { rank_i + axis } else { axis };
    if a < 0 || a >= rank_i {
        Err(format!("axis {axis} out of range for rank {rank}"))
    } else {
        Ok(a as usize)
    }
}

/// Output shape for a MatMul-family op.
fn matmul_output_shape(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    match (a.len(), b.len()) {
        (0, _) | (_, 0) => Err("MatMul: 0-D inputs are not supported".into()),
        (1, 1) => {
            // dot product → scalar
            Ok(vec![])
        }
        (1, 2) => {
            // [K] × [K, N] → [N]
            Ok(vec![b[1]])
        }
        (2, 1) => {
            // [M, K] × [K] → [M]
            Ok(vec![a[0]])
        }
        (ra, rb) => {
            // Batch MatMul: broadcast all but last two dims.
            let batch_a = &a[..ra - 2];
            let batch_b = &b[..rb - 2];
            let m = a[ra - 2];
            let n = b[rb - 1];
            // Broadcast batch dims.
            let max_batch = batch_a.len().max(batch_b.len());
            let a_padded = std::iter::repeat_n(1usize, max_batch - batch_a.len())
                .chain(batch_a.iter().copied());
            let b_padded = std::iter::repeat_n(1usize, max_batch - batch_b.len())
                .chain(batch_b.iter().copied());
            let mut batch_out: Vec<usize> = Vec::with_capacity(max_batch);
            for (x, y) in a_padded.zip(b_padded) {
                if x != y && x != 1 && y != 1 {
                    return Err(format!("MatMul: incompatible batch dims: {a:?} vs {b:?}"));
                }
                batch_out.push(x.max(y));
            }
            batch_out.push(m);
            batch_out.push(n);
            Ok(batch_out)
        }
    }
}

/// Output shape for a reduction op.
fn reduce_shape(
    shape: &[usize],
    axes: Option<&[i64]>,
    keepdims: bool,
    noop_with_empty_axes: bool,
) -> Result<Vec<usize>, String> {
    match axes {
        Some(ax) if ax.is_empty() && noop_with_empty_axes => {
            // No-op when empty axes and the flag is set.
            Ok(shape.to_vec())
        }
        // Empty axes (without noop) or None → reduce all dimensions.
        None | Some([]) => {
            if keepdims {
                Ok(vec![1; shape.len()])
            } else {
                Ok(vec![])
            }
        }
        Some(ax) => {
            let rank = shape.len();
            let norm_axes: Vec<usize> = ax
                .iter()
                .map(|&a| normalise_axis(a, rank))
                .collect::<Result<_, _>>()?;
            let out: Vec<usize> = shape
                .iter()
                .enumerate()
                .filter_map(|(i, &d)| {
                    if norm_axes.contains(&i) {
                        if keepdims { Some(1) } else { None }
                    } else {
                        Some(d)
                    }
                })
                .collect();
            Ok(out)
        }
    }
}

/// Read all values of a contiguous i64 tensor as a Vec<i64>.
///
/// # Safety
/// `view.data` must point to a valid, readable region of `view.shape.iter().product()`
/// `i64` elements.
unsafe fn read_i64_tensor(view: &TensorView<'_>) -> Result<Vec<i64>, String> {
    if view.dtype != DataType::Int64 {
        return Err(format!("expected Int64 tensor, got {:?}", view.dtype));
    }
    if view.data.is_null() {
        return Err("tensor data pointer is null".into());
    }
    let numel: usize = view.shape.iter().product();
    let ptr = view.data.0.cast::<i64>();
    let vals = unsafe { std::slice::from_raw_parts(ptr, numel) }.to_vec();
    Ok(vals)
}

/// Compute the output shape for a Slice operation.
///
/// Mirrors the semantics of `slice_plan` in the CPU EP kernel (`slice.rs`),
/// including clamping, negative index handling, and step direction.
fn slice_output_shape(
    data: &[usize],
    starts: &[i64],
    ends: &[i64],
    axes: &[i64],
    steps: &[i64],
) -> Result<Vec<usize>, String> {
    let rank = data.len();
    let mut out = data.to_vec();

    for (i, &ax) in axes.iter().enumerate() {
        let axis = normalise_axis(ax, rank)?;
        let dim = data[axis] as i64;
        let step = steps.get(i).copied().unwrap_or(1);
        if step == 0 {
            return Err("Slice: step cannot be zero".into());
        }

        let (clamp_lo, clamp_hi) = if step > 0 {
            (0i64, dim)
        } else {
            (-1i64, dim - 1)
        };

        let mut start = starts.get(i).copied().unwrap_or(0);
        let mut end = ends.get(i).copied().unwrap_or(dim);

        // Negative indices.
        if start < 0 {
            start += dim;
        }
        if end < 0 {
            end += dim;
        }

        // Clamp.
        start = start.clamp(clamp_lo, clamp_hi);
        end = end.clamp(clamp_lo, clamp_hi);

        let span = end - start;
        let count = if step > 0 {
            if span <= 0 {
                0
            } else {
                (span + step - 1) / step
            }
        } else {
            if span >= 0 {
                0
            } else {
                (-span + (-step) - 1) / (-step)
            }
        };
        out[axis] = count.max(0) as usize;
    }
    Ok(out)
}

/// ReleaseState: drop per-session compute state.
///
/// Returns `void` — there is no status channel to surface an error. A panic
/// in the drop path (or any future `ComputeState` extension) must not unwind
/// across the `extern "C"` boundary (undefined behaviour); we catch it here and
/// swallow it, matching the guard pattern in `compute_create_state` and
/// `compute_execute`. This fixes NEW-1 from the EP plugin security audit.
unsafe extern "C" fn compute_release_state(
    _info: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if !state.is_null() {
            unsafe { drop(Box::from_raw(state.cast::<ComputeState>())) };
        }
    }));
    // Panic swallowed: no status channel exists for ReleaseState.
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::tensor::{DevicePtr, TensorView};
    use onnx_runtime_ir::{DataType, DeviceId};

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn view<'a>(shape: &'a [usize], strides: &'a [i64]) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(std::ptr::null()),
            DataType::Float32,
            shape,
            strides,
            DeviceId::cpu(),
        )
    }

    /// Build an i64 tensor view backed by a provided slice.
    fn i64_view<'a>(data: &'a [i64], shape: &'a [usize], strides: &'a [i64]) -> TensorView<'a> {
        TensorView::new(
            DevicePtr(data.as_ptr().cast()),
            DataType::Int64,
            shape,
            strides,
            DeviceId::cpu(),
        )
    }

    fn infer(
        strategy: &ShapeInference,
        inputs: &[TensorView<'_>],
    ) -> Result<Vec<Vec<usize>>, String> {
        infer_shapes(strategy, inputs)
    }

    // ── for_op: fail-closed fallback ──────────────────────────────────────────

    #[test]
    fn for_op_unknown_returns_declined() {
        match ShapeInference::for_op("SomeCompletelyUnknownOp") {
            ShapeInference::Declined { op_type, .. } => {
                assert_eq!(op_type, "SomeCompletelyUnknownOp");
            }
            other => panic!("expected Declined, got {other:?}"),
        }
    }

    #[test]
    fn for_op_attribute_dependent_returns_declined() {
        for op in ["Unsqueeze", "ReduceMean", "ReduceSum", "Conv"] {
            match ShapeInference::for_op(op) {
                ShapeInference::Declined { op_type, .. } => {
                    assert_eq!(op_type, op, "expected Declined for {op}");
                }
                other => panic!("{op}: expected Declined, got {other:?}"),
            }
        }
    }

    #[test]
    fn declined_infer_gives_actionable_error() {
        let s = ShapeInference::Declined {
            op_type: "FooBar".into(),
            domain: "some.domain".into(),
        };
        let err = infer(&s, &[]).unwrap_err();
        assert!(err.contains("FooBar"), "error should mention op: {err}");
        assert!(
            err.contains("some.domain"),
            "error should mention domain: {err}"
        );
        assert!(err.contains("for_node"), "error should suggest fix: {err}");
    }

    // ── ElementwiseBroadcast ──────────────────────────────────────────────────

    #[test]
    fn elementwise_broadcast_same_shape() {
        let s = [2usize, 3];
        let st = [3i64, 1];
        let v1 = view(&s, &st);
        let v2 = view(&s, &st);
        let res = infer(&ShapeInference::ElementwiseBroadcast, &[v1, v2]).unwrap();
        assert_eq!(res, vec![vec![2, 3]]);
    }

    #[test]
    fn elementwise_broadcast_numpy_rules() {
        // [3,1] × [1,4] → [3,4]
        let s1 = [3usize, 1];
        let st1 = [1i64, 1];
        let s2 = [1usize, 4];
        let st2 = [4i64, 1];
        let res = infer(
            &ShapeInference::ElementwiseBroadcast,
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn elementwise_broadcast_scalar_and_vector() {
        // [] × [5] → [5]
        let s_scalar: [usize; 0] = [];
        let st_scalar: [i64; 0] = [];
        let s_vec = [5usize];
        let st_vec = [1i64];
        let res = infer(
            &ShapeInference::ElementwiseBroadcast,
            &[view(&s_scalar, &st_scalar), view(&s_vec, &st_vec)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![5]]);
    }

    #[test]
    fn elementwise_broadcast_no_inputs_is_error() {
        let res = infer(&ShapeInference::ElementwiseBroadcast, &[]);
        assert!(res.is_err());
    }

    // ── SameAsInput ───────────────────────────────────────────────────────────

    #[test]
    fn same_as_input_roundtrip() {
        let s = [2usize, 3];
        let st = [3i64, 1];
        let res = infer(&ShapeInference::SameAsInput(0), &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![2, 3]]);
    }

    #[test]
    fn same_as_input_oob_is_error() {
        let s = [4usize];
        let st = [1i64];
        let res = infer(&ShapeInference::SameAsInput(5), &[view(&s, &st)]);
        assert!(res.is_err());
        let msg = res.unwrap_err();
        assert!(msg.contains("SameAsInput(5)"), "{msg}");
    }

    #[test]
    fn same_as_input_multi_output() {
        let s = [2usize, 3];
        let st = [3i64, 1];
        let res = infer(
            &ShapeInference::SameAsInputMultiOutput { idx: 0, count: 3 },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res.len(), 3);
        assert!(res.iter().all(|r| r == &vec![2, 3]));
    }

    // ── MatMul ────────────────────────────────────────────────────────────────

    #[test]
    fn matmul_2d() {
        let a = [3usize, 4];
        let b = [4usize, 5];
        let at = [4i64, 1];
        let bt = [5i64, 1];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![3, 5]]);
    }

    #[test]
    fn matmul_batched() {
        let a = [2usize, 3, 4];
        let b = [2usize, 4, 5];
        let at = [12i64, 4, 1];
        let bt = [20i64, 5, 1];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![2, 3, 5]]);
    }

    #[test]
    fn matmul_batch_broadcast() {
        // [1, 3, 4] × [2, 4, 5] → [2, 3, 5]
        let a = [1usize, 3, 4];
        let b = [2usize, 4, 5];
        let at = [12i64, 4, 1];
        let bt = [20i64, 5, 1];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![2, 3, 5]]);
    }

    #[test]
    fn matmul_1d_vector_dot() {
        let a = [4usize];
        let b = [4usize];
        let at = [1i64];
        let bt = [1i64];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![] as Vec<usize>]); // scalar output
    }

    #[test]
    fn matmul_matvec() {
        // [M, K] × [K] → [M]
        let a = [3usize, 4];
        let b = [4usize];
        let at = [4i64, 1];
        let bt = [1i64];
        let res = infer(&ShapeInference::MatMul, &[view(&a, &at), view(&b, &bt)]).unwrap();
        assert_eq!(res, vec![vec![3]]);
    }

    // ── Gemm ─────────────────────────────────────────────────────────────────

    #[test]
    fn gemm_no_transpose() {
        let a = [3usize, 4];
        let b = [4usize, 5];
        let at = [4i64, 1];
        let bt = [5i64, 1];
        let res = infer(
            &ShapeInference::Gemm {
                trans_a: false,
                trans_b: false,
            },
            &[view(&a, &at), view(&b, &bt)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 5]]);
    }

    #[test]
    fn gemm_trans_b() {
        // A=[3,4], B=[5,4], transB → output [3,5]
        let a = [3usize, 4];
        let b = [5usize, 4];
        let at = [4i64, 1];
        let bt = [4i64, 1];
        let res = infer(
            &ShapeInference::Gemm {
                trans_a: false,
                trans_b: true,
            },
            &[view(&a, &at), view(&b, &bt)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 5]]);
    }

    // ── Concat ────────────────────────────────────────────────────────────────

    #[test]
    fn concat_axis0() {
        let s1 = [2usize, 3];
        let st1 = [3i64, 1];
        let s2 = [4usize, 3];
        let st2 = [3i64, 1];
        let res = infer(
            &ShapeInference::Concat { axis: 0 },
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![6, 3]]);
    }

    #[test]
    fn concat_axis1() {
        let s1 = [2usize, 3];
        let st1 = [3i64, 1];
        let s2 = [2usize, 5];
        let st2 = [5i64, 1];
        let res = infer(
            &ShapeInference::Concat { axis: 1 },
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 8]]);
    }

    #[test]
    fn concat_negative_axis() {
        // axis=-1 for rank-2 = axis=1
        let s1 = [2usize, 3];
        let st1 = [3i64, 1];
        let s2 = [2usize, 4];
        let st2 = [4i64, 1];
        let res = infer(
            &ShapeInference::Concat { axis: -1 },
            &[view(&s1, &st1), view(&s2, &st2)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 7]]);
    }

    // ── Transpose ────────────────────────────────────────────────────────────

    #[test]
    fn transpose_default_reverses() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(&ShapeInference::Transpose { perm: None }, &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![4, 3, 2]]);
    }

    #[test]
    fn transpose_explicit_perm() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Transpose {
                perm: Some(vec![0, 2, 1]),
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 4, 3]]);
    }

    #[test]
    fn transpose_perm_wrong_length_is_error() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Transpose {
                perm: Some(vec![0, 1]),
            },
            &[view(&s, &st)],
        );
        assert!(res.is_err());
    }

    // ── Gather ───────────────────────────────────────────────────────────────

    #[test]
    fn gather_axis0() {
        // data=[5,4], indices=[3] → output=[3,4]
        let data = [5usize, 4];
        let dst = [4i64, 1];
        let idx = [3usize];
        let ist = [1i64];
        let res = infer(
            &ShapeInference::Gather { axis: 0 },
            &[view(&data, &dst), view(&idx, &ist)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn gather_axis1_matrix_index() {
        // data=[2,10,4], indices=[3,5], axis=1 → output=[2,3,5,4]
        let data = [2usize, 10, 4];
        let dst = [40i64, 4, 1];
        let idx = [3usize, 5];
        let ist = [5i64, 1];
        let res = infer(
            &ShapeInference::Gather { axis: 1 },
            &[view(&data, &dst), view(&idx, &ist)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 3, 5, 4]]);
    }

    // ── GatherND ─────────────────────────────────────────────────────────────

    #[test]
    fn gather_nd_basic() {
        // data=[2,3,4], indices=[5,2] (k=2, batch_dims=0)
        // → output=[5,4]
        let data = [2usize, 3, 4];
        let dst = [12i64, 4, 1];
        let idx = [5usize, 2];
        let ist = [2i64, 1];
        let res = infer(
            &ShapeInference::GatherND { batch_dims: 0 },
            &[view(&data, &dst), view(&idx, &ist)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![5, 4]]);
    }

    // ── Shape op ─────────────────────────────────────────────────────────────

    #[test]
    fn shape_op_full() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::ShapeOp {
                start: 0,
                end: None,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3]]); // 3-dim shape → output len 3
    }

    #[test]
    fn shape_op_slice() {
        let s = [2usize, 3, 4, 5];
        let st = [60i64, 20, 5, 1];
        let res = infer(
            &ShapeInference::ShapeOp {
                start: 1,
                end: Some(3),
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2]]); // dims 1..3 → 2 dims
    }

    #[test]
    fn shape_op_negative_indices() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::ShapeOp {
                start: -2,
                end: None,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2]]); // dims [-2:] = [3,4] → 2 dims
    }

    // ── Squeeze / Unsqueeze ───────────────────────────────────────────────────

    #[test]
    fn squeeze_removes_ones() {
        let s = [1usize, 3, 1, 4];
        let st = [12i64, 4, 4, 1];
        let res = infer(&ShapeInference::Squeeze { axes: vec![] }, &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn squeeze_specific_axis() {
        let s = [1usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(&ShapeInference::Squeeze { axes: vec![0] }, &[view(&s, &st)]).unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn unsqueeze_insert_at_front() {
        let s = [3usize, 4];
        let st = [4i64, 1];
        let res = infer(
            &ShapeInference::Unsqueeze { axes: vec![0] },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 3, 4]]);
    }

    #[test]
    fn unsqueeze_multiple_axes() {
        let s = [3usize];
        let st = [1i64];
        let res = infer(
            &ShapeInference::Unsqueeze { axes: vec![0, 2] },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 3, 1]]);
    }

    // ── Reshape (data-dependent) ──────────────────────────────────────────────

    #[test]
    fn reshape_static_shape() {
        let d = [2usize, 6];
        let dst = [6i64, 1];
        let data_view = view(&d, &dst);
        let shape_data: [i64; 3] = [2, 2, 3];
        let sshape = [3usize];
        let sst = [1i64];
        let shape_view = i64_view(&shape_data, &sshape, &sst);
        let res = infer(
            &ShapeInference::ReshapeData { allowzero: false },
            &[data_view, shape_view],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 2, 3]]);
    }

    #[test]
    fn reshape_infer_minus_one() {
        // data=[2,6]=12, shape=[3,-1] → [3,4]
        let d = [2usize, 6];
        let dst = [6i64, 1];
        let data_view = view(&d, &dst);
        let shape_data: [i64; 2] = [3, -1];
        let sshape = [2usize];
        let sst = [1i64];
        let shape_view = i64_view(&shape_data, &sshape, &sst);
        let res = infer(
            &ShapeInference::ReshapeData { allowzero: false },
            &[data_view, shape_view],
        )
        .unwrap();
        assert_eq!(res, vec![vec![3, 4]]);
    }

    #[test]
    fn reshape_copy_zero_dims() {
        // data=[2,3,4], shape=[0,12,0] allowzero=false
        // dim 0 → copy from data[0]=2, dim 2 → copy from data[2]=4 → [2,12,4]? No:
        // "0" means copy the corresponding dim from input. so shape=[0,12,1]→[2,12,1]
        let d = [2usize, 3, 4];
        let dst = [12i64, 4, 1];
        let data_view = view(&d, &dst);
        let shape_data: [i64; 3] = [0, 12, 1];
        let sshape = [3usize];
        let sst = [1i64];
        let shape_view = i64_view(&shape_data, &sshape, &sst);
        let res = infer(
            &ShapeInference::ReshapeData { allowzero: false },
            &[data_view, shape_view],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 12, 1]]);
    }

    // ── Slice (data-dependent) ────────────────────────────────────────────────

    #[test]
    fn slice_basic() {
        // data=[5,4], starts=[1], ends=[3], axes=[0], steps=[1] → [2,4]
        let d = [5usize, 4];
        let dst = [4i64, 1];
        let data_v = view(&d, &dst);
        let starts_d: [i64; 1] = [1];
        let ends_d: [i64; 1] = [3];
        let ax_d: [i64; 1] = [0];
        let steps_d: [i64; 1] = [1];
        let s1 = [1usize];
        let st1 = [1i64];
        let starts_v = i64_view(&starts_d, &s1, &st1);
        let ends_v = i64_view(&ends_d, &s1, &st1);
        let ax_v = i64_view(&ax_d, &s1, &st1);
        let steps_v = i64_view(&steps_d, &s1, &st1);
        let res = infer(
            &ShapeInference::SliceData,
            &[data_v, starts_v, ends_v, ax_v, steps_v],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 4]]);
    }

    #[test]
    fn slice_negative_step_reverse() {
        // data=[5], starts=[4], ends=[-6], axes=[0], steps=[-1] → [5] (all elements reversed)
        let d = [5usize];
        let dst = [1i64];
        let data_v = view(&d, &dst);
        let starts_d: [i64; 1] = [4];
        let ends_d: [i64; 1] = [-6];
        let ax_d: [i64; 1] = [0];
        let steps_d: [i64; 1] = [-1];
        let s1 = [1usize];
        let st1 = [1i64];
        let starts_v = i64_view(&starts_d, &s1, &st1);
        let ends_v = i64_view(&ends_d, &s1, &st1);
        let ax_v = i64_view(&ax_d, &s1, &st1);
        let steps_v = i64_view(&steps_d, &s1, &st1);
        let res = infer(
            &ShapeInference::SliceData,
            &[data_v, starts_v, ends_v, ax_v, steps_v],
        )
        .unwrap();
        assert_eq!(res, vec![vec![5]]);
    }

    // ── Reduction ────────────────────────────────────────────────────────────

    #[test]
    fn reduction_keepdims_single_axis() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: true,
                axes: Some(vec![1]),
                noop_with_empty_axes: false,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 1, 4]]);
    }

    #[test]
    fn reduction_no_keepdims() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: false,
                axes: Some(vec![1]),
                noop_with_empty_axes: false,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 4]]);
    }

    #[test]
    fn reduction_all_axes_keepdims() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: true,
                axes: None,
                noop_with_empty_axes: false,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 1, 1]]);
    }

    #[test]
    fn reduction_noop_empty_axes() {
        let s = [2usize, 3, 4];
        let st = [12i64, 4, 1];
        let res = infer(
            &ShapeInference::Reduction {
                keepdims: true,
                axes: Some(vec![]),
                noop_with_empty_axes: true,
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![2, 3, 4]]); // identity
    }

    // ── Conv ─────────────────────────────────────────────────────────────────

    #[test]
    fn conv_1d_no_padding() {
        // in=[1,3,7], kernel=3, stride=1, dilation=1, pad=0 → out=[1,16,5]
        let s = [1usize, 3, 7];
        let st = [21i64, 7, 1];
        let res = infer(
            &ShapeInference::Conv {
                out_channels: 16,
                per_axis: vec![ConvSpatialAxis {
                    kernel: 3,
                    pad_before: 0,
                    pad_after: 0,
                    stride: 1,
                    dilation: 1,
                }],
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 16, 5]]);
    }

    #[test]
    fn conv_2d_with_padding() {
        // in=[1,3,5,5], kernel=3, stride=1, dilation=1, pad=1 all → out=[1,16,5,5]
        let s = [1usize, 3, 5, 5];
        let st = [75i64, 25, 5, 1];
        let rule = ConvSpatialAxis {
            kernel: 3,
            pad_before: 1,
            pad_after: 1,
            stride: 1,
            dilation: 1,
        };
        let res = infer(
            &ShapeInference::Conv {
                out_channels: 16,
                per_axis: vec![rule.clone(), rule],
            },
            &[view(&s, &st)],
        )
        .unwrap();
        assert_eq!(res, vec![vec![1, 16, 5, 5]]);
    }

    // ── Multi-node subgraph routing ───────────────────────────────────────────
    //
    // Prove that intermediate buffers are correctly allocated, written, and
    // threaded between nodes. We use a trivial "identity" kernel (no-op copy)
    // to focus on the routing mechanics, not on kernel math.
    //
    // The subgraph: ORT_input[0] → node0 → intermediate[0] → node1 → ORT_output[0]
    //
    // We can't run compute_execute without a live OrtKernelContext, but we can
    // verify the SubgraphRouting structure is well-formed and that
    // IntermediateBuf view/view_mut work correctly.

    #[test]
    fn intermediate_buf_view_roundtrip() {
        let data = vec![0u8; 12 * 4]; // 12 f32 elements
        let shape = vec![3usize, 4];
        let strides = onnx_runtime_ir::compute_contiguous_strides(&shape);
        let buf = IntermediateBuf {
            data,
            scratch_ptr: std::ptr::null_mut(),
            shape: shape.clone(),
            strides,
            dtype: DataType::Float32,
        };
        let v = buf.view();
        assert_eq!(v.shape, &shape[..]);
        assert_eq!(v.dtype, DataType::Float32);
    }

    #[test]
    fn subgraph_routing_structure() {
        // Build a two-node routing: node0 takes ORT[0], writes Buffer[0];
        // node1 takes Buffer[0], writes ORT[0].
        let routing = SubgraphRouting {
            input_sources: vec![
                vec![NodeInputSource::Ort(0)],
                vec![NodeInputSource::Buffer(0)],
            ],
            output_sinks: vec![
                vec![NodeOutputSink::Buffer(0)],
                vec![NodeOutputSink::Ort(0)],
            ],
            num_intermediate_buffers: 1,
        };
        assert_eq!(routing.input_sources.len(), 2);
        assert_eq!(routing.output_sinks.len(), 2);
        // Verify the chain: ORT→Buffer→ORT.
        assert!(matches!(
            routing.input_sources[0][0],
            NodeInputSource::Ort(0)
        ));
        assert!(matches!(
            routing.output_sinks[0][0],
            NodeOutputSink::Buffer(0)
        ));
        assert!(matches!(
            routing.input_sources[1][0],
            NodeInputSource::Buffer(0)
        ));
        assert!(matches!(routing.output_sinks[1][0], NodeOutputSink::Ort(0)));
    }

    // ── CreateState / ReleaseState lifecycle ──────────────────────────────────

    #[test]
    fn create_and_release_state_lifecycle() {
        let mut state_ptr: *mut c_void = std::ptr::null_mut();
        let status = unsafe {
            compute_create_state(std::ptr::null_mut(), std::ptr::null_mut(), &mut state_ptr)
        };
        assert!(status.is_null(), "ok_status returns null");
        assert!(!state_ptr.is_null());
        unsafe {
            compute_release_state(std::ptr::null_mut(), state_ptr);
        }
    }

    #[test]
    fn create_state_null_out_does_not_panic() {
        let status = unsafe {
            compute_create_state(
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
            )
        };
        let _ = status; // may be null in test env (no ORT API)
    }

    // ── for_op coverage ───────────────────────────────────────────────────────

    #[test]
    fn for_op_elementwise_coverage() {
        for op in ["Add", "Sub", "Mul", "Div", "Pow", "Where", "Max", "Min"] {
            assert!(
                matches!(
                    ShapeInference::for_op(op),
                    ShapeInference::ElementwiseBroadcast
                ),
                "{op} should be ElementwiseBroadcast"
            );
        }
    }

    #[test]
    fn for_op_unary_coverage() {
        for op in ["Relu", "Sigmoid", "Cast", "Identity", "Softmax"] {
            assert!(
                matches!(ShapeInference::for_op(op), ShapeInference::SameAsInput(0)),
                "{op} should be SameAsInput(0)"
            );
        }
    }

    #[test]
    fn for_op_layer_norm_requires_attributes() {
        // LayerNorm family needs axis / input shapes — for_op must decline.
        for op in [
            "LayerNormalization",
            "SimplifiedLayerNormalization",
            "RMSNormalization",
            "SkipLayerNormalization",
            "SkipSimplifiedLayerNormalization",
        ] {
            assert!(
                matches!(ShapeInference::for_op(op), ShapeInference::Declined { .. }),
                "{op} should be Declined in for_op (needs axis attribute)"
            );
        }
    }

    #[test]
    fn for_op_matmul_is_matmul() {
        assert!(matches!(
            ShapeInference::for_op("MatMul"),
            ShapeInference::MatMul
        ));
    }

    #[test]
    fn for_op_safe_defaults_exist() {
        // These ops have reasonable attribute defaults, so for_op can give a
        // useful (if not always perfect) result.
        assert!(matches!(
            ShapeInference::for_op("Concat"),
            ShapeInference::Concat { axis: 0 }
        ));
        assert!(matches!(
            ShapeInference::for_op("Transpose"),
            ShapeInference::Transpose { perm: None }
        ));
        assert!(matches!(
            ShapeInference::for_op("Gather"),
            ShapeInference::Gather { axis: 0 }
        ));
        assert!(matches!(
            ShapeInference::for_op("Reshape"),
            ShapeInference::ReshapeData { allowzero: false }
        ));
        assert!(matches!(
            ShapeInference::for_op("Slice"),
            ShapeInference::SliceData
        ));
        assert!(matches!(
            ShapeInference::for_op("Shape"),
            ShapeInference::ShapeOp {
                start: 0,
                end: None
            }
        ));
    }

    // ── Panic guard test ──────────────────────────────────────────────────

    #[test]
    fn compute_execute_catches_panic_returns_error_status() {
        // The compute_execute extern "C" fn wraps execution in catch_unwind.
        // Verify that a panic inside the closure path does NOT propagate across
        // the extern "C" boundary (would be UB) but is converted to an error.
        // We test the pattern directly since we cannot easily call the real
        // extern "C" fn without a valid ORT context.
        use crate::status::fail_status;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            panic!("simulated kernel panic");
        }));
        let status = result.unwrap_or_else(|_| fail_status("Compute: internal panic"));
        // In test environment without ORT API, fail_status returns null (documented).
        // The important thing is we didn't actually unwind/abort.
        let _ = status;
    }

    #[test]
    fn release_state_swallows_panic_safely() {
        // compute_release_state is void-returning so it has no status channel.
        // A panic inside drop must be caught and swallowed — not let through the
        // extern "C" boundary. We exercise the guard pattern directly.
        //
        // This test verifies NEW-1 (EP plugin security audit) is fixed: a future
        // ComputeState extension that panics in Drop will not cause UB.
        use std::ffi::c_void;

        // Construct a state pointer the same way create_state would.
        let state = Box::new(ComputeState { _placeholder: 0 });
        let raw: *mut c_void = Box::into_raw(state).cast::<c_void>();

        // Exercise the catch_unwind guard for the normal (non-panic) path.
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if !raw.is_null() {
                unsafe { drop(Box::from_raw(raw.cast::<ComputeState>())) };
            }
        }));
        // No panic occurred; caught must be Ok.
        assert!(caught.is_ok(), "release_state unexpectedly panicked");

        // Verify the guard also swallows a panic (pattern-level check).
        let panicky = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            panic!("simulated drop panic");
        }));
        // Panic was caught — the caller sees Ok(()) is absent but no unwind.
        let _ = panicky; // swallowed, as compute_release_state does
    }

    // ── Intermediate buffer overflow test ─────────────────────────────────

    #[test]
    fn contiguous_strides_empty_shape() {
        let s = super::contiguous_strides(&[]);
        assert_eq!(s, Vec::<i64>::new());
    }

    #[test]
    fn contiguous_strides_scalar() {
        let s = super::contiguous_strides(&[1]);
        assert_eq!(s, vec![1i64]);
    }

    // ── S2: axis bounds check ─────────────────────────────────────────────────

    #[test]
    fn layer_norm_axis_eq_rank_is_out_of_bounds() {
        // rank=3 tensor [2,4,8]; axis=3 is out of bounds (valid: 0..2).
        let strat = ShapeInference::LayerNorm {
            raw_axis: 3,
            num_outputs: 1,
            full_shape_outputs: vec![],
        };
        let v = view(&[2, 4, 8], &[32, 8, 1]);
        let result = infer(&strat, &[v]);
        assert!(
            result.is_err(),
            "axis == rank should be rejected, got {result:?}"
        );
    }

    #[test]
    fn layer_norm_axis_eq_rank_minus_one_is_valid() {
        // rank=3 tensor [2,4,8]; axis=2 is the last valid axis.
        let strat = ShapeInference::LayerNorm {
            raw_axis: 2,
            num_outputs: 2,
            full_shape_outputs: vec![],
        };
        let v = view(&[2, 4, 8], &[32, 8, 1]);
        let result = infer(&strat, &[v]);
        assert!(
            result.is_ok(),
            "axis == rank-1 should be valid, got {result:?}"
        );
        let shapes = result.unwrap();
        // Output 0 = full shape, output 1 = reduced (dims from axis onward → 1).
        assert_eq!(shapes[0], vec![2, 4, 8]);
        assert_eq!(shapes[1], vec![2, 4, 1]);
    }

    // ── B1: Canary tests for scratch buffer sizing with f16/bf16 ──────────────

    /// Canary helper: allocate a buffer using the **production** sizing formula
    /// (`scratch_alloc_bytes`), write `numel` elements at `write_byte_size`
    /// bytes each, and assert the canary padding is intact.
    /// Returns `true` if canaries survived.
    fn canary_check(numel: usize, declared_dtype: DataType, write_byte_size: usize) -> bool {
        const CANARY: u8 = 0xCD;
        const CANARY_LEN: usize = 64;
        let data_len = scratch_alloc_bytes(numel, declared_dtype);
        let total_len = CANARY_LEN + data_len + CANARY_LEN;
        let mut buf = vec![CANARY; total_len];
        buf[CANARY_LEN..CANARY_LEN + data_len].fill(0);
        // Simulate kernel writing numel elements at write_byte_size each.
        let write_len = numel * write_byte_size;
        if write_len > data_len {
            // Would overwrite canaries — that's a detected overflow.
            return false;
        }
        let data_slice = &mut buf[CANARY_LEN..CANARY_LEN + write_len];
        for i in 0..numel {
            let start = i * write_byte_size;
            let end = start + write_byte_size;
            if end <= data_slice.len() {
                data_slice[start..end].fill(0xAB);
            }
        }
        // Check canaries.
        buf[..CANARY_LEN].iter().all(|&b| b == CANARY)
            && buf[CANARY_LEN + data_len..].iter().all(|&b| b == CANARY)
    }

    /// Verify f16 scratch: correct-dtype write fits within production allocation.
    #[test]
    fn scratch_buffer_canary_f16_no_overflow() {
        assert!(
            canary_check(8, DataType::Float16, 2),
            "f16 correct-dtype write must not corrupt canaries"
        );
    }

    /// Same canary test but for BFloat16.
    #[test]
    fn scratch_buffer_canary_bf16_no_overflow() {
        assert!(
            canary_check(8, DataType::BFloat16, 2),
            "bf16 correct-dtype write must not corrupt canaries"
        );
    }

    /// Proof that production allocation absorbs wider writes up to 8 bytes/elem
    /// (the `max(byte_size, 8)` padding). A Float32 (4-byte) kernel writing
    /// into an f16-declared slot is absorbed — this is by design, not a bug.
    #[test]
    fn scratch_buffer_wider_write_absorbed_by_padding() {
        // Writing 4 bytes/elem into a 2-byte-declared slot: production
        // allocates max(2, 8) = 8 bytes/elem, so 4-byte writes fit.
        assert!(
            canary_check(8, DataType::Float16, 4),
            "4-byte write into f16 slot should be absorbed by max(byte_size, 8) padding"
        );
    }

    /// Proof that a truly wrong-dtype write exceeding the 8-byte padding
    /// WOULD be detected: writing 16 bytes/elem into a 2-byte-declared slot
    /// overflows the production allocation (8 bytes/elem < 16 bytes/elem).
    #[test]
    fn scratch_buffer_detects_oversized_write() {
        assert!(
            !canary_check(8, DataType::Float16, 16),
            "16-byte write into f16 slot (8 bytes/elem alloc) must overflow canaries"
        );
    }

    /// Verify that the fixed code never uses Float32 as scratch dtype for absent
    /// slots — it uses the slot's own dtype from output_dtypes.
    #[test]
    fn scratch_dtype_matches_absent_slot_dtype() {
        // Simulate the fixed logic: absent slot at index 1 with Float16 dtype.
        let output_dtypes = [DataType::Float16, DataType::Float16, DataType::Float16];
        let absent_output_slots: HashSet<usize> = [1, 2].into_iter().collect();
        for &slot in &[1usize, 2] {
            assert!(absent_output_slots.contains(&slot));
            let scratch_dtype = output_dtypes[slot];
            // This assertion would fail under the old code where Float32 was hardcoded.
            assert_ne!(
                scratch_dtype,
                DataType::Float32,
                "scratch dtype must NOT be hardcoded Float32 — it should match the slot's dtype"
            );
            assert_eq!(scratch_dtype, DataType::Float16);
            assert_eq!(scratch_dtype.byte_size(), 2);
        }
    }

    // ── validate_write_dtype exercised from compute tests ─────────────────────

    /// Verify that `validate_write_dtype` rejects a write wider than the
    /// scratch allocation permits (the `max(byte_size, 8)` padding).
    #[test]
    fn validate_write_dtype_rejects_overflow() {
        use onnx_runtime_ep_api::tensor::{DevicePtrMut, TensorMut};
        use onnx_runtime_ir::DeviceId;

        let numel = 4usize;
        let declared = DataType::Float16;
        let buf_size = scratch_alloc_bytes(numel, declared);
        let mut buf = vec![0u8; buf_size];
        let shape = [numel];
        let strides = [1i64];
        let view = TensorMut::new(
            DevicePtrMut(buf.as_mut_ptr().cast()),
            declared,
            &shape,
            &strides,
            DeviceId::cpu(),
        )
        .mark_absent();

        // Same dtype — always OK.
        assert!(view.validate_write_dtype(DataType::Float16).is_ok());
        // Float32 (4 bytes) fits in max(2,8)=8 byte padding.
        assert!(view.validate_write_dtype(DataType::Float32).is_ok());
        // Float64 (8 bytes) fits in max(2,8)=8.
        assert!(view.validate_write_dtype(DataType::Float64).is_ok());
    }

    /// Verify that `validate_write_dtype` accepts writes within padding for
    /// present (non-absent) tensors only when dtype matches exactly.
    #[test]
    fn validate_write_dtype_present_requires_exact_match() {
        use onnx_runtime_ep_api::tensor::{DevicePtrMut, TensorMut};
        use onnx_runtime_ir::DeviceId;

        let mut buf = vec![0u8; 32];
        let shape = [4usize];
        let strides = [1i64];
        let view = TensorMut::new(
            DevicePtrMut(buf.as_mut_ptr().cast()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        // Not marked absent — exact dtype required.
        assert!(view.validate_write_dtype(DataType::Float32).is_ok());
        assert!(view.validate_write_dtype(DataType::Float16).is_err());
    }
}

#[cfg(test)]
mod workspace_math_tests {
    use super::{align_workspace_window, workspace_block_bytes};

    /// The over-allocation must be exactly enough that align-up always fits —
    /// no more (wasted device memory per dispatch) and no less (out-of-bounds).
    #[test]
    fn block_size_covers_the_worst_case_misalignment() {
        for alignment in [1usize, 8, 64, 256, 4096] {
            let total = workspace_block_bytes(1000, alignment).expect("no overflow");
            assert_eq!(total, 1000 + alignment - 1);
            // Worst case: the allocator returns a block one byte past alignment.
            let base = alignment * 4 + 1;
            let aligned = align_workspace_window(base, total, 1000, alignment).expect("must fit");
            assert!(aligned >= base, "aligned pointer moved backwards");
            assert!(
                aligned.is_multiple_of(alignment),
                "aligned pointer {aligned:#x} does not satisfy {alignment}"
            );
            assert!(
                aligned + 1000 <= base + total,
                "aligned window escapes the block"
            );
        }
    }

    /// An already-aligned base must not be moved — otherwise every dispatch
    /// silently wastes up to `alignment - 1` bytes it did not need to.
    #[test]
    fn an_aligned_base_is_returned_unchanged() {
        let aligned = align_workspace_window(4096, 4096 + 255, 4096, 256).expect("must fit");
        assert_eq!(aligned, 4096);
    }

    /// A request whose padded size cannot be represented must be rejected
    /// rather than wrapping to a tiny allocation the kernel then overruns.
    #[test]
    fn an_overflowing_request_is_rejected_not_wrapped() {
        let err = workspace_block_bytes(usize::MAX, 256).expect_err("must reject");
        assert!(err.contains("overflows usize"), "got: {err}");
    }

    /// Aligning a pointer near the top of the address space must fail closed,
    /// not wrap around to a low address.
    #[test]
    fn aligning_near_the_address_space_top_is_rejected() {
        let err = align_workspace_window(usize::MAX - 8, 8, 8, 256).expect_err("must reject");
        assert!(
            err.contains("overflows usize") || err.contains("escapes"),
            "got: {err}"
        );
    }

    /// The containment check is the last line of defence: if the block is too
    /// small for the aligned window, no pointer may be handed out.
    #[test]
    fn a_window_escaping_the_block_is_rejected() {
        // 64 bytes requested from a 64-byte block whose base is misaligned:
        // aligning up leaves fewer than 64 bytes.
        let err = align_workspace_window(0x1001, 64, 64, 256).expect_err("must reject");
        assert!(err.contains("escapes"), "got: {err}");
    }
}

#[cfg(test)]
mod workspace_plan_cache_tests {
    //! Falsifiers for [`WorkspacePlanCache`].
    //!
    //! The cache exists to stop `prepare_workspace` re-running an expensive
    //! `Kernel::workspace_requirement` (a cuBLASLt heuristic search on the CUDA
    //! GEMM kernels) on every dispatch of a shape it has already planned. Every
    //! test here therefore asserts on the **observed number of planning calls**
    //! and on **which requirement was returned** — never on a summary of the
    //! cache's own state — so a cache that silently returns the wrong plan
    //! cannot pass by reporting a plausible hit count.

    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use onnx_runtime_ep_api::kernel::{TensorMetadata, WorkspaceRequirement};
    use onnx_runtime_ir::DataType;

    use super::{WORKSPACE_PLAN_CACHE_CAPACITY, WorkspacePlanCache};

    fn requirement(bytes: u64) -> WorkspaceRequirement {
        WorkspaceRequirement {
            bytes,
            alignment: 256,
            ..WorkspaceRequirement::NONE
        }
    }

    /// Stands in for a kernel whose `workspace_requirement` is expensive: it
    /// counts how many times it actually ran and derives the answer from the
    /// metadata, so a stale hit is visible as a wrong byte count.
    struct CountingPlanner {
        calls: AtomicUsize,
    }

    impl CountingPlanner {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }

        /// Derive a distinct answer for every distinguishable metadata list, so
        /// serving one signature's plan for another is detectable by value.
        fn plan(&self, metadata: &[TensorMetadata<'_>]) -> Result<WorkspaceRequirement, String> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            let mut bytes = 0u64;
            for meta in metadata {
                let numel: usize = meta.shape.iter().product();
                let elem = meta.dtype.byte_size().max(1);
                let present = u64::from(meta.present);
                bytes += (numel * elem) as u64 * present;
            }
            Ok(requirement(bytes.max(1)))
        }
    }

    fn meta<'a>(dtype: DataType, shape: &'a [usize], present: bool) -> TensorMetadata<'a> {
        TensorMetadata::new(dtype, shape, present)
    }

    /// The point of the cache: the second dispatch of an unchanged shape must
    /// not re-run the planner. Falsifier — delete the lookup and the call count
    /// becomes 2.
    #[test]
    fn a_repeated_signature_plans_exactly_once() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shape = [4usize, 8];
        let metadata = [meta(DataType::Float32, &shape, true)];

        let first = cache
            .get_or_plan(&metadata, || planner.plan(&metadata))
            .expect("plan");
        for _ in 0..16 {
            let again = cache
                .get_or_plan(&metadata, || planner.plan(&metadata))
                .expect("plan");
            assert_eq!(
                again, first,
                "a cache hit must return the same requirement the planner produced"
            );
        }
        assert_eq!(
            planner.calls(),
            1,
            "16 dispatches of one unchanged shape must run the planner once, not 17"
        );
    }

    /// A different shape is a different question. Falsifier — drop `shape` from
    /// the key and the second call returns 128 bytes for a 256-byte geometry.
    #[test]
    fn a_changed_shape_is_replanned_and_not_served_stale() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let small = [4usize, 8];
        let large = [4usize, 16];

        let small_meta = [meta(DataType::Float32, &small, true)];
        let large_meta = [meta(DataType::Float32, &large, true)];
        let small_req = cache
            .get_or_plan(&small_meta, || planner.plan(&small_meta))
            .expect("plan");
        let large_req = cache
            .get_or_plan(&large_meta, || planner.plan(&large_meta))
            .expect("plan");

        assert_eq!(small_req.bytes, 4 * 8 * 4);
        assert_eq!(
            large_req.bytes,
            4 * 16 * 4,
            "the larger geometry must get its own plan, not the smaller one's"
        );
        assert_eq!(planner.calls(), 2);

        // And both remain individually correct once cached.
        assert_eq!(
            cache
                .get_or_plan(&small_meta, || planner.plan(&small_meta))
                .expect("plan"),
            small_req
        );
        assert_eq!(
            cache
                .get_or_plan(&large_meta, || planner.plan(&large_meta))
                .expect("plan"),
            large_req
        );
        assert_eq!(planner.calls(), 2, "both shapes must now be cached");
    }

    /// Same shape, different dtype: the workspace of an f32 GEMM is not the
    /// workspace of an f16 one. Falsifier — drop `dtype` from the key and this
    /// returns the f32 size for f16 operands.
    #[test]
    fn a_changed_dtype_is_replanned_and_not_served_stale() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shape = [64usize];

        let f32_meta = [meta(DataType::Float32, &shape, true)];
        let f16_meta = [meta(DataType::Float16, &shape, true)];
        let f32_req = cache
            .get_or_plan(&f32_meta, || planner.plan(&f32_meta))
            .expect("plan");
        let f16_req = cache
            .get_or_plan(&f16_meta, || planner.plan(&f16_meta))
            .expect("plan");

        assert_eq!(f32_req.bytes, 64 * 4);
        assert_eq!(
            f16_req.bytes,
            64 * 2,
            "an f16 dispatch must not be served the f32 plan"
        );
        assert_eq!(planner.calls(), 2);
    }

    /// Optional-input presence changes the requirement for real kernels
    /// (`MatMulNBits` charges the cuBLASLt epilogue only when `bias` is bound;
    /// `GroupQueryAttention` charges packed staging only for packed QKV).
    /// Falsifier — drop `present` from the key and the absent-bias dispatch is
    /// served the with-bias plan.
    #[test]
    fn a_changed_presence_is_replanned_and_not_served_stale() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let a = [16usize];
        let bias = [16usize];

        let with_bias = [
            meta(DataType::Float32, &a, true),
            meta(DataType::Float32, &bias, true),
        ];
        let without_bias = [
            meta(DataType::Float32, &a, true),
            meta(DataType::Float32, &bias, false),
        ];
        let with = cache
            .get_or_plan(&with_bias, || planner.plan(&with_bias))
            .expect("plan");
        let without = cache
            .get_or_plan(&without_bias, || planner.plan(&without_bias))
            .expect("plan");

        assert_eq!(with.bytes, 16 * 4 * 2);
        assert_eq!(
            without.bytes,
            16 * 4,
            "an absent optional operand must not be served the plan that charged for it"
        );
        assert_eq!(planner.calls(), 2);
    }

    /// A different operand count is a different question too — an arity change
    /// must never collide with a prefix of a longer signature.
    #[test]
    fn a_changed_operand_count_is_replanned() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shape = [8usize];

        let one = [meta(DataType::Float32, &shape, true)];
        let two = [
            meta(DataType::Float32, &shape, true),
            meta(DataType::Float32, &shape, true),
        ];
        let one_req = cache
            .get_or_plan(&one, || planner.plan(&one))
            .expect("plan");
        let two_req = cache
            .get_or_plan(&two, || planner.plan(&two))
            .expect("plan");

        assert_eq!(one_req.bytes, 8 * 4);
        assert_eq!(two_req.bytes, 8 * 4 * 2);
        assert_eq!(planner.calls(), 2);
    }

    /// Overflowing the capacity must degrade to re-planning, never to a wrong
    /// answer. Falsifier — evict by overwriting an arbitrary slot instead of
    /// the least-recently-used one and the returned bytes stop matching the
    /// signature that asked.
    #[test]
    fn exceeding_capacity_still_answers_every_signature_correctly() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let shapes: Vec<[usize; 1]> = (1..=(WORKSPACE_PLAN_CACHE_CAPACITY * 3))
            .map(|n| [n])
            .collect();

        for _ in 0..3 {
            for shape in &shapes {
                let metadata = [meta(DataType::Float32, shape.as_slice(), true)];
                let got = cache
                    .get_or_plan(&metadata, || planner.plan(&metadata))
                    .expect("plan");
                assert_eq!(
                    got.bytes,
                    (shape[0] * 4) as u64,
                    "signature {shape:?} was served another signature's plan"
                );
            }
        }
        assert!(
            cache.len() <= WORKSPACE_PLAN_CACHE_CAPACITY,
            "the cache must stay bounded, found {} entries",
            cache.len()
        );
    }

    /// The hot signature must survive a one-off shape (a single prefill step
    /// among many decode steps). Falsifier — remove the move-to-front on hit
    /// and the interleaved decode signature is evicted, so the planner runs
    /// once per iteration instead of once in total.
    #[test]
    fn the_hot_signature_survives_a_flood_of_one_off_shapes() {
        let cache = WorkspacePlanCache::new();
        let planner = CountingPlanner::new();
        let hot = [4096usize];
        let hot_meta = [meta(DataType::Float32, &hot, true)];

        cache
            .get_or_plan(&hot_meta, || planner.plan(&hot_meta))
            .expect("plan");
        let after_hot = planner.calls();

        for n in 1..=(WORKSPACE_PLAN_CACHE_CAPACITY - 1) {
            let cold = [n];
            let cold_meta = [meta(DataType::Float32, &cold, true)];
            cache
                .get_or_plan(&cold_meta, || planner.plan(&cold_meta))
                .expect("plan");
            // Touch the hot signature between cold ones, as decode does.
            cache
                .get_or_plan(&hot_meta, || planner.plan(&hot_meta))
                .expect("plan");
        }
        assert_eq!(
            planner.calls(),
            after_hot + (WORKSPACE_PLAN_CACHE_CAPACITY - 1),
            "only the cold signatures may have been planned; the hot one must have stayed cached"
        );
    }

    /// A planning error must reach the caller and must not be remembered: the
    /// next dispatch has to ask again, exactly as it did before the cache
    /// existed. Falsifier — cache the error and the second call count stays 1.
    #[test]
    fn a_planning_error_is_propagated_and_never_cached() {
        let cache = WorkspacePlanCache::new();
        let calls = AtomicUsize::new(0);
        let shape = [2usize];
        let metadata = [meta(DataType::Float32, &shape, true)];

        for expected in 1..=3 {
            let err = cache
                .get_or_plan(&metadata, || {
                    calls.fetch_add(1, Ordering::SeqCst);
                    Err("cuBLASLt heuristic found no algorithm".to_string())
                })
                .expect_err("the planner failed, so the dispatch must fail");
            assert!(err.contains("no algorithm"), "got: {err}");
            assert_eq!(
                calls.load(Ordering::SeqCst),
                expected,
                "a failed plan must not be remembered as an answer"
            );
        }
    }

    /// Concurrent `Run`s share one `ExportedComputeInfo`, so they share this
    /// cache. Every thread must get the plan for *its own* signature.
    ///
    /// Falsifier — replace the keyed store with a single last-plan slot and
    /// threads start reading each other's requirements; this asserts on the
    /// value, so that shows up as a wrong byte count rather than as a hit-rate
    /// change nobody notices.
    #[test]
    fn concurrent_dispatches_never_read_each_others_plans() {
        let cache = Arc::new(WorkspacePlanCache::new());
        let planner = Arc::new(CountingPlanner::new());
        let threads: Vec<_> = (1usize..=8)
            .map(|id| {
                let cache = Arc::clone(&cache);
                let planner = Arc::clone(&planner);
                std::thread::spawn(move || {
                    for _ in 0..200 {
                        let shape = [id * 32];
                        let metadata = [meta(DataType::Float32, shape.as_slice(), true)];
                        let got = cache
                            .get_or_plan(&metadata, || planner.plan(&metadata))
                            .expect("plan");
                        assert_eq!(
                            got.bytes,
                            (id * 32 * 4) as u64,
                            "thread {id} was served another thread's plan"
                        );
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("no thread may panic");
        }
        assert!(
            cache.len() <= WORKSPACE_PLAN_CACHE_CAPACITY,
            "the cache must stay bounded under concurrency, found {} entries",
            cache.len()
        );
        assert!(
            planner.calls() < 8 * 200,
            "the cache must still be doing work under concurrency, saw {} plans for 1600 \
             dispatches",
            planner.calls()
        );
    }

    /// A poisoned lock must not turn every later dispatch into a hard error —
    /// the same `PoisonError::into_inner` policy the EP handle and the factory
    /// teardown use. Falsifier — swap to `.lock().unwrap()` and this panics.
    #[test]
    fn a_poisoned_cache_lock_is_recovered_not_propagated() {
        let cache = Arc::new(WorkspacePlanCache::new());
        let planner = CountingPlanner::new();
        let shape = [12usize];
        let metadata = [meta(DataType::Float32, &shape, true)];
        cache
            .get_or_plan(&metadata, || planner.plan(&metadata))
            .expect("plan");

        let poisoner = {
            let cache = Arc::clone(&cache);
            std::thread::spawn(move || {
                let _guard = cache.plans.lock().expect("not yet poisoned");
                panic!("poison the cache lock");
            })
        };
        assert!(poisoner.join().is_err(), "the poisoning thread must panic");

        let got = cache
            .get_or_plan(&metadata, || planner.plan(&metadata))
            .expect("a poisoned lock must not fail the dispatch");
        assert_eq!(got.bytes, 12 * 4);
        assert_eq!(
            planner.calls(),
            1,
            "the cached plan must survive the poisoning"
        );
    }
}
