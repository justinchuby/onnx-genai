//! Native Gemm shape capability shared by CPU and CUDA.

use onnx_runtime_ir::{
    DataType, Node, Shape,
    gemm::{GemmAxes, ND_GEMM_OPSET, bias_broadcasts},
};

pub fn validate_dtypes(
    allowed: &[DataType],
    mut dtypes: impl Iterator<Item = DataType>,
) -> Result<(), String> {
    if let Some(dtype) = dtypes.next() {
        if !allowed.contains(&dtype) {
            return Err(format!(
                "Gemm dtype {dtype:?} is unsupported on this provider; use one of {allowed:?} or select a different provider"
            ));
        }
        if let Some(other) = dtypes.find(|&other| other != dtype) {
            return Err(format!(
                "Gemm requires A, B, optional C, and Y to share one dtype, got {dtype:?} and {other:?}; cast all operands to the same supported dtype"
            ));
        }
    }
    Ok(())
}

pub fn unsupported_reason(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
    allowed: Option<&[DataType]>,
) -> Option<String> {
    if !node.is_default_domain() || node.op_type != "Gemm" {
        return None;
    }
    if let Some(allowed) = allowed
        && let Err(reason) = validate_dtypes(
            allowed,
            dtypes
                .iter()
                .copied()
                .filter(|&dtype| dtype != DataType::Undefined),
        )
    {
        return Some(reason);
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

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{NodeId, static_shape};

    #[test]
    fn gemm_29_claim_requires_homogeneous_supported_dtypes() {
        use DataType::{Float16, Float32, Int32, Undefined};
        let node = Node::new(NodeId(0), "Gemm", vec![], vec![]);
        let shapes = [static_shape([2, 3, 5]), static_shape([5, 7]), vec![]];
        let cpu = &[Float16, Float32];
        let cuda = &[Float32];
        let claim = |dtypes: &[DataType], allowed: &[DataType]| {
            unsupported_reason(&node, 29, &shapes, dtypes, Some(allowed))
        };
        assert_eq!(claim(&[Float32, Float32, Undefined], cuda), None);
        assert_eq!(claim(&[Float16, Float16, Float16], cpu), None);
        assert!(
            claim(&[Float16, Float16], cuda)
                .unwrap()
                .contains("unsupported")
        );
        assert!(claim(&[Int32, Int32], cpu).unwrap().contains("unsupported"));
        assert!(
            claim(&[Float32, Float32, Float16], cpu)
                .unwrap()
                .contains("share one dtype")
        );
    }
}
