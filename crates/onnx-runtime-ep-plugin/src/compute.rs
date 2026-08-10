//! `ExportedComputeInfo` — wraps Rust `Kernel`s as `OrtNodeComputeInfo` callbacks.
//!
//! Each compiled subgraph gets one `ExportedComputeInfo` with a vector of
//! compiled kernel entries. The `Compute` callback dispatches to the
//! appropriate kernel, reads inputs from ORT, infers output shapes, allocates
//! output tensors, and executes the kernel.

use std::ffi::c_void;

use onnx_genai_ort_sys as ort;
use onnx_runtime_ep_api::kernel::Kernel;
use onnx_runtime_ir::DataType;

use crate::kernel_ctx::{allocate_output, read_inputs};
use crate::status::{fail_status, ok_status};

/// How to infer output shapes at runtime from the concrete input shapes.
#[derive(Clone, Debug)]
pub enum ShapeInference {
    /// Output shape = numpy-style broadcast of all input shapes.
    /// Used for elementwise ops (Add, Mul, Sub, Div, etc.).
    ElementwiseBroadcast,
    /// Output shape is identical to input at the given index.
    SameAsInput(usize),
}

impl ShapeInference {
    /// Select the appropriate shape inference strategy for a given ONNX op_type.
    ///
    /// Ops not listed here get `SameAsInput(0)` as a conservative default.
    /// If that is wrong for a particular op, Compute will fail with a shape
    /// mismatch rather than silently producing wrong results.
    pub fn for_op(op_type: &str) -> Self {
        match op_type {
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Mod" | "And" | "Or" | "Xor"
            | "Equal" | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual"
            | "BitShift" | "BitwiseAnd" | "BitwiseOr" | "BitwiseXor"
            | "Max" | "Min" | "Mean" | "Sum" | "Where" => Self::ElementwiseBroadcast,
            // Unary / shape-preserving ops.
            "Relu" | "Sigmoid" | "Tanh" | "Exp" | "Log" | "Sqrt" | "Abs" | "Neg"
            | "Ceil" | "Floor" | "Round" | "Reciprocal" | "Not" | "Sign"
            | "Erf" | "Gelu" | "HardSigmoid" | "LeakyRelu" | "Elu" | "Selu"
            | "Softplus" | "Softsign" | "Cast" | "Identity" | "Dropout"
            | "LayerNormalization" => Self::SameAsInput(0),
            _ => Self::SameAsInput(0),
        }
    }
}

/// A compiled kernel bundled with the metadata needed to drive execution
/// through the ORT kernel context.
pub struct CompiledKernelEntry {
    pub kernel: Box<dyn Kernel>,
    pub num_inputs: usize,
    pub num_outputs: usize,
    /// Output dtype (all outputs share this for elementwise ops).
    pub output_dtype: DataType,
    pub shape_inference: ShapeInference,
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
}

/// Per-session state created by `CreateState`.
struct ComputeState {
    _placeholder: u8,
}

impl ExportedComputeInfo {
    pub fn new(entries: Vec<CompiledKernelEntry>) -> Self {
        Self {
            vtable: ort::OrtNodeComputeInfo {
                ort_version_supported: ort::ORT_API_VERSION,
                CreateState: Some(compute_create_state),
                Compute: Some(compute_execute),
                ReleaseState: Some(compute_release_state),
            },
            entries,
        }
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

/// Compute: execute the kernel(s) for this subgraph.
///
/// For single-node subgraphs (the common case for CPU EP), this calls
/// `kernel.execute()` once. For multi-node fused subgraphs, it iterates
/// in topological order.
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
    if info.is_null() || kernel_context.is_null() {
        return fail_status("Compute: null argument");
    }

    let exported = unsafe { &*(info.cast::<ExportedComputeInfo>()) };

    if exported.entries.is_empty() {
        return fail_status("Compute: no kernels compiled for this subgraph");
    }

    // Get the host OrtApi for kernel context operations.
    let api = unsafe { crate::status::host_api() };
    if api.is_null() {
        return fail_status("Compute: host ORT API not available");
    }
    let api_ref = unsafe { &*api };

    // Read all inputs from the ORT kernel context.
    let inputs = match unsafe { read_inputs(api_ref, kernel_context) } {
        Ok(inputs) => inputs,
        Err(e) => return fail_status(&format!("Compute: {e}")),
    };

    // For single-kernel subgraphs (the common path), execute directly.
    // For multi-kernel subgraphs, iterate in order (topologically sorted
    // at Compile time).
    let mut input_offset = 0;
    let mut output_offset = 0;

    for entry in &exported.entries {
        // Gather this kernel's input views.
        let kernel_inputs: Vec<_> = inputs[input_offset..input_offset + entry.num_inputs]
            .iter()
            .map(|inp| inp.view())
            .collect();

        // Infer output shapes.
        let output_shapes = match infer_shapes(&entry.shape_inference, &kernel_inputs) {
            Ok(shapes) => shapes,
            Err(e) => {
                return fail_status(&format!(
                    "Compute: shape inference failed: {e}"
                ));
            }
        };

        // Allocate outputs from ORT.
        let mut owned_outputs = Vec::with_capacity(entry.num_outputs);
        for (out_idx, shape) in output_shapes.iter().enumerate() {
            match unsafe {
                allocate_output(
                    api_ref,
                    kernel_context,
                    output_offset + out_idx,
                    shape,
                    entry.output_dtype,
                )
            } {
                Ok(out) => owned_outputs.push(out),
                Err(e) => {
                    return fail_status(&format!("Compute: {e}"));
                }
            }
        }

        // Create mutable views and execute.
        let mut output_views: Vec<_> =
            owned_outputs.iter_mut().map(|o| o.view_mut()).collect();

        if let Err(e) = entry.kernel.execute(&kernel_inputs, &mut output_views) {
            return fail_status(&format!("Compute: kernel execution failed: {e}"));
        }

        input_offset += entry.num_inputs;
        output_offset += entry.num_outputs;
    }

    ok_status()
}

/// Infer output shapes from the shape inference strategy and input views.
fn infer_shapes(
    strategy: &ShapeInference,
    inputs: &[onnx_runtime_ep_api::tensor::TensorView<'_>],
) -> Result<Vec<Vec<usize>>, String> {
    match strategy {
        ShapeInference::ElementwiseBroadcast => {
            if inputs.is_empty() {
                return Err("no inputs for broadcast shape inference".into());
            }
            let mut result_shape = inputs[0].shape.to_vec();
            for input in &inputs[1..] {
                result_shape =
                    onnx_runtime_ir::broadcast_shapes(&result_shape, input.shape)
                        .map_err(|e| format!("broadcast failed: {e}"))?;
            }
            // Elementwise ops produce exactly one output.
            Ok(vec![result_shape])
        }
        ShapeInference::SameAsInput(idx) => {
            if *idx >= inputs.len() {
                return Err(format!(
                    "SameAsInput({idx}) but only {} inputs",
                    inputs.len()
                ));
            }
            Ok(vec![inputs[*idx].shape.to_vec()])
        }
    }
}

/// ReleaseState: drop per-session compute state.
unsafe extern "C" fn compute_release_state(
    _info: *mut ort::OrtNodeComputeInfo,
    state: *mut c_void,
) {
    if !state.is_null() {
        unsafe { drop(Box::from_raw(state.cast::<ComputeState>())) };
    }
}
