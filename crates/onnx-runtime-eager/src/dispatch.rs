//! The core single-op dispatch flow (`docs/EAGER.md` §10.1), reconciled to the
//! real runtime APIs.
//!
//! ## Design-vs-real-API reconciliation
//!
//! The design pseudocode (`docs/EAGER.md` §10.1) calls
//! `registry.lookup(op_type, domain, opset)` then `factory.create(attrs)`. The
//! real APIs are:
//!
//! * [`OpRegistry::lookup`](onnx_runtime_ep_api::OpRegistry::lookup) /
//!   [`KernelFactory::create`](onnx_runtime_ep_api::KernelFactory) take a
//!   [`Node`] and `input_shapes`, and the CPU EP already wraps both behind
//!   [`ExecutionProvider::get_kernel`](onnx_runtime_ep_api::ExecutionProvider::get_kernel)
//!   `(node, shapes, opset)`. Eager dispatch therefore builds an **ephemeral
//!   single [`Node`]** (op_type, domain, attributes, and placeholder
//!   input/output value slots) and calls `get_kernel`, mapping the EP's
//!   `NoEpForOp` into [`EagerError::NoKernel`]. This mirrors how
//!   `onnx-runtime-session/src/executor.rs` drives kernels.
//! * Output shapes come from
//!   [`InferenceRegistry::infer_node`](onnx_runtime_shape_inference::InferenceRegistry)
//!   (§9), fed the same ephemeral node plus per-input [`NodeIo`] built from the
//!   concrete input shapes/dtypes. Because eager inputs have fully static
//!   shapes, the inferred [`DimExpr`]s resolve to constants; a symbolic or
//!   missing output is a [`EagerError::ShapeInference`] error (the kernel-
//!   provided fallback of §9.2 is DEFERRED).

use std::collections::HashMap;
use std::ffi::c_void;

use onnx_runtime_ep_api::{DevicePtr, DevicePtrMut, EpError, TensorMut, TensorView};
use onnx_runtime_ir::{compute_contiguous_strides, Attribute, DataType, Node, NodeId, ValueId};
use onnx_runtime_shape_inference::{DimExpr, MergePolicy, NodeIo, SymbolInterner, TypeInfo};

use crate::cache::KernelCacheKey;
use crate::error::{EagerError, Result};
use crate::tensor::Tensor;
use crate::EagerContext;

/// Build the ephemeral [`Node`] that feeds both `get_kernel` and shape
/// inference. Input/output value ids are positional placeholders — no real
/// graph exists in eager mode.
///
/// DEFERRED (EAGER.md §9): multi-output arity. Phase-1 dispatch materialises a
/// single output slot, which covers every Phase-1 op. Multi-output ops (e.g.
/// `TopK`, `Split`) need the true output count here (from an op schema or an
/// explicit argument) before their extra outputs are inferred/allocated.
fn ephemeral_node(
    op_type: &str,
    domain: &str,
    attrs: &HashMap<String, Attribute>,
    num_inputs: usize,
    num_outputs: usize,
) -> Node {
    let inputs: Vec<Option<ValueId>> = (0..num_inputs).map(|i| Some(ValueId(i as u32))).collect();
    let outputs: Vec<ValueId> = (0..num_outputs)
        .map(|i| ValueId((num_inputs + i) as u32))
        .collect();
    let mut node = Node::new(NodeId(0), op_type, inputs, outputs);
    node.domain = domain.to_string();
    node.attributes = attrs.clone();
    node
}

impl EagerContext {
    /// Dispatch a single ONNX op to a kernel and return its outputs
    /// (`docs/EAGER.md` §10.1). The 7-step flow, reconciled to the real APIs:
    ///
    /// 1. resolve the effective opset (explicit per-call value > domain default),
    /// 2. resolve the target device from the inputs (mixed devices = error),
    /// 3. build the compiled-kernel cache key,
    /// 4. get-or-compile the kernel via the device's EP (missing kernel =
    ///    [`EagerError::NoKernel`]),
    /// 5. infer output shapes/dtypes for the single op,
    /// 6. allocate the output tensors on the device,
    /// 7. build zero-copy views and execute.
    pub fn dispatch(
        &self,
        op_type: &str,
        domain: &str,
        inputs: &[&Tensor],
        attrs: &HashMap<String, Attribute>,
        explicit_opset: Option<u64>,
    ) -> Result<Vec<Tensor>> {
        // 1. Effective opset for this domain.
        let opset = self
            .domains
            .read()
            .expect("domain registry lock poisoned")
            .resolve_opset(domain, explicit_opset);

        // 2. Target device (mixed-device inputs are rejected, §1.6).
        let device = self.resolve_device(inputs)?;
        let ep = self.ep_for_device(device)?;

        // Concrete input shapes, reused for the kernel, cache key, and views.
        let input_shapes: Vec<Vec<usize>> = inputs.iter().map(|t| t.shape().to_vec()).collect();

        // DEFERRED (EAGER.md §9): single output slot only (see `ephemeral_node`).
        let node = ephemeral_node(op_type, domain, attrs, inputs.len(), 1);

        // 3 + 4. Cache key, then get-or-compile the kernel through the EP.
        let cache_key = KernelCacheKey {
            op_type: op_type.to_string(),
            domain: domain.to_string(),
            opset,
            input_shapes: input_shapes.clone(),
            input_dtypes: inputs.iter().map(|t| t.dtype()).collect(),
            device,
        };
        let kernel = self
            .cache
            .lock()
            .expect("kernel cache lock poisoned")
            .get_or_create(cache_key, || -> Result<Box<dyn onnx_runtime_ep_api::Kernel>> {
                ep.get_kernel(&node, &input_shapes, opset).map_err(|e| match e {
                    EpError::NoEpForOp { .. } => EagerError::NoKernel {
                        op_type: op_type.to_string(),
                        domain: domain.to_string(),
                        device,
                    },
                    other => EagerError::Kernel(other),
                })
            })?;

        // 5. Infer output shapes/dtypes for the single op.
        let output_meta = self.infer_output_meta(&node, op_type, domain, opset, inputs)?;

        // 6. Allocate output tensors on the target device.
        let mut outputs: Vec<Tensor> = Vec::with_capacity(output_meta.len());
        for (dtype, shape) in &output_meta {
            outputs.push(Tensor::zeros_in(ep.clone(), *dtype, shape.clone())?);
        }

        // 7. Build zero-copy views over the raw device buffers and execute.
        // Stride/shape holders must outlive the views that borrow them.
        let in_strides: Vec<Vec<i64>> = input_shapes
            .iter()
            .map(|s| compute_contiguous_strides(s))
            .collect();
        let out_shapes: Vec<Vec<usize>> = output_meta.iter().map(|(_, s)| s.clone()).collect();
        let out_strides: Vec<Vec<i64>> = out_shapes
            .iter()
            .map(|s| compute_contiguous_strides(s))
            .collect();

        let in_ptrs: Vec<*const c_void> = inputs.iter().map(|t| t.device_ptr()).collect();
        let out_ptrs: Vec<*mut c_void> =
            outputs.iter_mut().map(|t| t.device_ptr_mut()).collect();

        let input_views: Vec<TensorView> = (0..inputs.len())
            .map(|i| {
                TensorView::new(
                    DevicePtr(in_ptrs[i]),
                    inputs[i].dtype(),
                    &input_shapes[i],
                    &in_strides[i],
                    device,
                )
            })
            .collect();
        let mut output_views: Vec<TensorMut> = (0..outputs.len())
            .map(|i| {
                TensorMut::new(
                    DevicePtrMut(out_ptrs[i]),
                    output_meta[i].0,
                    &out_shapes[i],
                    &out_strides[i],
                    device,
                )
            })
            .collect();

        {
            let guard = kernel.lock().expect("cached kernel mutex poisoned");
            guard.execute(&input_views, &mut output_views)?;
        }
        // Drop the views (they borrow the ptr/stride holders) before the owned
        // output tensors leave the function.
        drop(output_views);
        drop(input_views);

        Ok(outputs)
    }

    /// Per-op output shape/dtype inference (`docs/EAGER.md` §9), driven by the
    /// shared [`InferenceRegistry`](onnx_runtime_shape_inference::InferenceRegistry).
    /// Returns one `(dtype, shape)` per output slot.
    fn infer_output_meta(
        &self,
        node: &Node,
        op_type: &str,
        domain: &str,
        opset: u64,
        inputs: &[&Tensor],
    ) -> Result<Vec<(DataType, Vec<usize>)>> {
        let input_ios: Vec<NodeIo> = inputs
            .iter()
            .map(|t| {
                let shape: Vec<DimExpr> =
                    t.shape().iter().map(|&d| DimExpr::constant(d as i64)).collect();
                NodeIo::typed(TypeInfo::new(t.dtype(), shape))
            })
            .collect();

        // The registry keys inference on `(domain, op)` and reads the opset for
        // the op's domain from `opset_imports`; supply exactly that entry.
        let mut opset_imports: HashMap<String, u64> = HashMap::new();
        opset_imports.insert(domain.to_string(), opset);

        let mut interner = SymbolInterner::new(0);
        let out_ios = self.inference.infer_node(
            node,
            &opset_imports,
            input_ios,
            MergePolicy::Permissive,
            &mut interner,
        )?;

        let mut meta = Vec::with_capacity(out_ios.len());
        for (i, io) in out_ios.iter().enumerate() {
            let type_info = io.type_info.as_ref().ok_or_else(|| EagerError::ShapeInference {
                op_type: op_type.to_string(),
                domain: domain.to_string(),
                reason: format!(
                    "no shape-inference rule resolved output {i} \
                     (kernel-provided fallback is DEFERRED, EAGER.md §9.2)"
                ),
            })?;
            let mut dims = Vec::with_capacity(type_info.shape.len());
            for (axis, d) in type_info.shape.iter().enumerate() {
                let c = d.as_const().ok_or_else(|| EagerError::ShapeInference {
                    op_type: op_type.to_string(),
                    domain: domain.to_string(),
                    reason: format!("output {i} axis {axis} is symbolic, not allocatable"),
                })?;
                if c < 0 {
                    return Err(EagerError::ShapeInference {
                        op_type: op_type.to_string(),
                        domain: domain.to_string(),
                        reason: format!("output {i} axis {axis} has a negative extent {c}"),
                    });
                }
                dims.push(c as usize);
            }
            meta.push((type_info.dtype, dims));
        }
        Ok(meta)
    }
}
