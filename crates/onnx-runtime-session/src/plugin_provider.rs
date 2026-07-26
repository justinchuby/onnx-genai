use std::collections::HashSet;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use onnx_runtime_ep_api::{
    CaptureRegionShapeStatus, DeviceBuffer, EpConfig, EpError, ExecutionProvider, Fence,
    KernelMatch, PluginCompiledKernel, PluginExecutionPlan, StructuralCaptureDecline,
};
use onnx_runtime_ir::{
    Attribute, DataType, DeviceId, DeviceType, Graph, Node, NodeId, Shape, TensorLayout,
};
use onnx_runtime_optimizer::{OptimizationPass, OptimizerError, PassContext};

/// Synthetic op type used after a plugin EP compiles and replaces a claimed subgraph.
pub(crate) const PLUGIN_FUSED_OP: &str = "NativePluginFused";
pub(crate) const PLUGIN_FUSED_DOMAIN: &str = "com.microsoft";
const FUSION_ID_ATTR: &str = "native_plugin_fusion_id";

/// A native execution provider that dispatches plugin-claimed fused subgraphs
/// through ORT's plugin-EP ABI and delegates all unclaimed nodes to the CPU EP.
pub struct PluginExecutionProvider {
    library: PathBuf,
    registration_name: Option<String>,
    provider_name: String,
    device_label: String,
    cpu: onnx_runtime_ep_cpu::CpuExecutionProvider,
    compiled: Arc<Mutex<Option<PluginExecutionPlan>>>,
}

impl PluginExecutionProvider {
    /// Create a plugin bridge. Names come from `ep_compat` metadata; the bridge
    /// itself does not special-case vendor/provider spellings.
    pub fn new(
        library: PathBuf,
        registration_name: Option<String>,
        provider_name: impl Into<String>,
        device_label: impl Into<String>,
    ) -> onnx_runtime_ep_api::Result<Self> {
        let mut cpu = onnx_runtime_ep_cpu::CpuExecutionProvider::new();
        cpu.initialize(&EpConfig::default()).map_err(|err| {
            EpError::KernelFailed(format!(
                "initialize CPU fallback for plugin EP failed: {err}; fix by enabling the native CPU EP"
            ))
        })?;
        Ok(Self {
            library,
            registration_name,
            provider_name: provider_name.into(),
            device_label: device_label.into(),
            cpu,
            compiled: Arc::new(Mutex::new(None)),
        })
    }
}

impl ExecutionProvider for PluginExecutionProvider {
    fn name(&self) -> &str {
        &self.provider_name
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Custom(0)
    }

    fn device_id(&self) -> DeviceId {
        DeviceId::cpu()
    }

    fn initialize(&mut self, _config: &EpConfig) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn shutdown(&mut self) -> onnx_runtime_ep_api::Result<()> {
        Ok(())
    }

    fn supports_op(
        &self,
        op: &Node,
        opset: u64,
        shapes: &[Shape],
        input_dtypes: &[DataType],
        layouts: &[TensorLayout],
    ) -> KernelMatch {
        if is_plugin_fused(op) {
            return KernelMatch::Supported {
                cost: onnx_runtime_ep_api::Cost::ZERO,
                required_input_layouts: None,
                output_layouts: vec![TensorLayout::contiguous(); op.outputs.len()],
            };
        }
        self.cpu
            .supports_op(op, opset, shapes, input_dtypes, layouts)
    }

    fn get_kernel(
        &self,
        op: &Node,
        shapes: &[Vec<usize>],
        opset: u64,
    ) -> onnx_runtime_ep_api::Result<Box<dyn onnx_runtime_ep_api::Kernel>> {
        if is_plugin_fused(op) {
            let id = fusion_id(op)?;
            let guard = self.compiled.lock().map_err(|_| {
                EpError::KernelFailed(
                    "plugin compiled-plan lock was poisoned; recreate the session".into(),
                )
            })?;
            let plan = guard.as_ref().ok_or_else(|| {
                EpError::KernelFailed(
                    "plugin fused node has no compiled OrtNodeComputeInfo; fix by running the plugin fusion pass before execution".into(),
                )
            })?;
            let kernel: PluginCompiledKernel = plan.kernel(id).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "plugin fused node references missing fusion id {id}; fix the plugin fusion pass"
                ))
            })?;
            return Ok(Box::new(kernel));
        }
        self.cpu.get_kernel(op, shapes, opset)
    }

    fn plan_capture_region(
        &self,
        node: &Node,
        shape_status: CaptureRegionShapeStatus,
    ) -> Option<StructuralCaptureDecline> {
        self.cpu.plan_capture_region(node, shape_status)
    }

    fn allocate(&self, size: usize, alignment: usize) -> onnx_runtime_ep_api::Result<DeviceBuffer> {
        self.cpu.allocate(size, alignment)
    }

    fn deallocate(&self, buffer: DeviceBuffer) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.deallocate(buffer)
    }

    fn copy(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.copy(src, dst, size)
    }

    fn copy_async(
        &self,
        src: &DeviceBuffer,
        dst: &mut DeviceBuffer,
        size: usize,
    ) -> onnx_runtime_ep_api::Result<Fence> {
        self.cpu.copy_async(src, dst, size)
    }

    fn sync(&self) -> onnx_runtime_ep_api::Result<()> {
        self.cpu.sync()
    }

    fn custom_passes(&self) -> Vec<Box<dyn OptimizationPass>> {
        vec![Box::new(PluginFusionPass {
            library: self.library.clone(),
            registration_name: self.registration_name.clone(),
            device_label: self.device_label.clone(),
            compiled: Arc::clone(&self.compiled),
        })]
    }
}

struct PluginFusionPass {
    library: PathBuf,
    registration_name: Option<String>,
    device_label: String,
    compiled: Arc<Mutex<Option<PluginExecutionPlan>>>,
}

impl OptimizationPass for PluginFusionPass {
    fn name(&self) -> &str {
        "plugin_ep_fusion"
    }

    fn run(&self, graph: &mut Graph, _ctx: &PassContext) -> onnx_runtime_optimizer::Result<()> {
        let registration = self
            .registration_name
            .as_ref()
            .map(|name| {
                CString::new(name.as_str()).map_err(|_| {
                    OptimizerError::Fusion(
                        "plugin registration name contains an interior NUL byte".into(),
                    )
                })
            })
            .transpose()?;
        let (claims, plan) = PluginExecutionPlan::compile_with_device_label(
            graph,
            &self.library,
            registration.as_deref(),
            self.device_label.clone(),
        )
        .map_err(|err| OptimizerError::Fusion(format!("{err}")))?;

        let mut groups = Vec::new();
        for (index, claim) in claims.iter().enumerate() {
            if claim.node_ids.is_empty() {
                continue;
            }
            let mut node = Node::new(
                NodeId(0),
                PLUGIN_FUSED_OP,
                claim.input_values.iter().copied().map(Some).collect(),
                claim.output_values.clone(),
            );
            node.domain = PLUGIN_FUSED_DOMAIN.to_string();
            node.name = format!("native_plugin_fused_{index}");
            node.attributes
                .insert(FUSION_ID_ATTR.to_string(), Attribute::Int(index as i64));
            groups.push((claim.node_ids.clone(), node));
        }
        if groups.is_empty() {
            return Err(OptimizerError::Fusion(
                "plugin compiled successfully but produced no non-empty fused groups".into(),
            ));
        }
        let graph_outputs = graph.outputs.iter().copied().collect::<HashSet<_>>();
        graph.replace_node_groups(groups, &graph_outputs);
        let mut guard = self.compiled.lock().map_err(|_| {
            OptimizerError::Fusion(
                "plugin compiled-plan lock was poisoned while installing fused groups".into(),
            )
        })?;
        *guard = Some(plan);
        Ok(())
    }
}

fn is_plugin_fused(node: &Node) -> bool {
    node.domain == PLUGIN_FUSED_DOMAIN && node.op_type == PLUGIN_FUSED_OP
}

fn fusion_id(node: &Node) -> onnx_runtime_ep_api::Result<usize> {
    node.attr(FUSION_ID_ATTR)
        .and_then(Attribute::as_int)
        .and_then(|id| usize::try_from(id).ok())
        .ok_or_else(|| {
            EpError::KernelFailed(
                "plugin fused node is missing a valid native_plugin_fusion_id attribute; fix the plugin fusion pass".into(),
            )
        })
}
