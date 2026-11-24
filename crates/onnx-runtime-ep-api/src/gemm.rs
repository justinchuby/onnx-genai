//! Native Gemm shape capability shared by CPU and CUDA.

use onnx_runtime_ir::{
    Node, Shape,
    gemm::{GemmAxes, ND_GEMM_OPSET, bias_broadcasts},
};

pub fn unsupported_reason(node: &Node, opset: u64, shapes: &[Shape]) -> Option<String> {
    if !node.is_default_domain() || node.op_type != "Gemm" {
        return None;
    }
    let [a, b, ..] = shapes else { return None };
    let nd = opset >= ND_GEMM_OPSET;
    let axes = match GemmAxes::new(
        a.len(),
        b.len(),
        node.attr("transA").and_then(|a| a.as_int()).unwrap_or(0) != 0,
        node.attr("transB").and_then(|a| a.as_int()).unwrap_or(0) != 0,
        nd,
    ) {
        Ok(axes) => axes,
        Err(reason) => return Some(reason),
    };
    if let (Some(k), Some(kb)) = (a[axes.a_k].as_static(), b[axes.b_k].as_static())
        && k != kb
    {
        return Some(format!(
            "Gemm contraction dimensions disagree for A {a:?} and B {b:?} ({k} vs {kb}); make them equal after transposition"
        ));
    }
    if let Some(c) = shapes.get(2) {
        // An omitted C is passed as an empty shape, which is also valid for a scalar.
        let y: Shape = axes.output_dims(a, b).copied().collect();
        if !bias_broadcasts(c, &y, |d| d.as_static()) {
            return Some(format!(
                "Gemm bias C {c:?} does not broadcast to output {y:?}; use equal or singleton right-aligned dimensions"
            ));
        }
    }
    None
}
