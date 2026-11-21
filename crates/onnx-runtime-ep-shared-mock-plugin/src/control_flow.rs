//! Structural control-flow conformance kernel. Production providers supply
//! their own graph compiler; this test provider proves transport and dispatch.
use onnx_runtime_ep_api::{
    EpError, GraphKernelRequest, Kernel, KernelSizedOutput, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{
    DataType, Graph, NodeId,
    pattern::{MatchBudgetScope, MatchOptions, MatchSession, Pattern},
};

pub(super) static EXECUTIONS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

fn pattern(axis: i64) -> Pattern {
    let mut then = Pattern::builder();
    let x = then.input("x");
    let y = then.op("Identity", [x]);
    let then = then.finish_body([y]).unwrap();
    let mut otherwise = Pattern::builder();
    let x = otherwise.input("x");
    let y = otherwise.node("Concat", [x, x]).attr("axis", axis).output();
    let otherwise = otherwise.finish_body([y]).unwrap();
    let mut root = Pattern::builder();
    let cond = root.input("condition");
    let y = root
        .node("If", [cond])
        .attr("then_branch", then)
        .attr("else_branch", otherwise)
        .output();
    root.capture("out", y);
    root.finish(y).unwrap()
}

pub(super) fn claim(graph: &Graph) -> Result<Vec<NodeId>> {
    let session =
        MatchSession::new(graph).map_err(|error| EpError::KernelFailed(error.to_string()))?;
    let mut claims = Vec::new();
    for axis in [0, -1] {
        let matches = session
            .find_matches(
                &pattern(axis),
                MatchOptions {
                    budget_scope: MatchBudgetScope::Root,
                    ..Default::default()
                },
            )
            .map_err(|error| EpError::KernelFailed(error.to_string()))?;
        claims.extend(matches.into_iter().map(|matched| {
            matched
                .value("out")
                .unwrap()
                .unwrap()
                .value()
                .producer
                .unwrap()
        }));
    }
    Ok(claims)
}

pub(super) fn compile(request: GraphKernelRequest<'_>) -> Result<Box<dyn Kernel>> {
    if !claim(request.graph)?.contains(&request.node.id) {
        return Err(EpError::KernelFailed(
            "shared mock control-flow compiler received an unclaimed graph node".into(),
        ));
    }
    if request.inputs.len() != 2 {
        return Err(EpError::KernelFailed(format!(
            "shared mock If pattern requires condition and one lexical capture, got {} inputs",
            request.inputs.len()
        )));
    }
    Ok(Box::new(IfKernel))
}
struct IfKernel;
impl Kernel for IfKernel {
    fn execute(&self, _: &[TensorView], _: &mut [TensorMut]) -> Result<()> {
        Err(EpError::KernelFailed(
            "shared mock If requires Compute-time output allocation".into(),
        ))
    }
    fn has_kernel_sized_outputs(&self) -> bool {
        true
    }
    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
    fn execute_kernel_sized(
        &self,
        inputs: &[TensorView],
        requested: &[bool],
    ) -> Result<Vec<Option<KernelSizedOutput>>> {
        if inputs.len() != 2 || requested != [true] {
            return Err(EpError::KernelFailed(
                "shared mock If has invalid boundary arity".into(),
            ));
        }
        for input in inputs {
            input.validate()?;
            if !input.device.is_host_accessible() || !input.is_contiguous() {
                return Err(EpError::KernelFailed(
                    "shared mock If requires contiguous host inputs".into(),
                ));
            }
        }
        if inputs[0].dtype != DataType::Bool
            || !inputs[0].shape.is_empty()
            || inputs[1].dtype != DataType::Float32
            || inputs[1].shape.len() != 1
        {
            return Err(EpError::KernelFailed(
                "shared mock If requires a scalar bool and a rank-one float32 capture".into(),
            ));
        }
        let condition = unsafe { *inputs[0].data_ptr::<u8>() } != 0;
        let bytes = if inputs[1].numel() == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(inputs[1].data_ptr::<u8>(), inputs[1].byte_size()) }
        };
        let mut result = bytes.to_vec();
        if !condition {
            result.extend_from_slice(bytes);
        }
        EXECUTIONS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Ok(vec![Some(KernelSizedOutput {
            shape: vec![inputs[1].numel() * if condition { 1 } else { 2 }],
            dtype: DataType::Float32,
            bytes: result,
        })])
    }
}
