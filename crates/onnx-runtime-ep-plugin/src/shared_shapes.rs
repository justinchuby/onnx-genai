//! Adapter from concrete plugin inputs to the native symbolic shape registry.

use std::collections::HashMap;
use std::sync::OnceLock;

use onnx_runtime_ep_api::TensorView;
use onnx_runtime_ir::{Node, normalize_domain};
use onnx_runtime_shape_inference::{
    DimExpr, InferenceRegistry, MAX_SHAPE_DATA_ELEMS, MergePolicy, NodeIo, ShapeData,
    SymbolInterner, TypeInfo,
};

/// The deliberately small first set of plugin rules delegated to the native
/// shape registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SharedNativeShapeRule {
    ConstantOfShape,
    Expand,
    Stft,
    Tile,
}

impl SharedNativeShapeRule {
    const ALL: [Self; 4] = [Self::ConstantOfShape, Self::Expand, Self::Stft, Self::Tile];

    /// Every rule currently routed through the shared adapter.
    #[cfg(feature = "testutil")]
    pub(crate) const fn all() -> &'static [Self] {
        &Self::ALL
    }

    /// The ONNX operator name for this rule.
    pub(crate) const fn op_type(self) -> &'static str {
        match self {
            Self::ConstantOfShape => "ConstantOfShape",
            Self::Expand => "Expand",
            Self::Stft => "STFT",
            Self::Tile => "Tile",
        }
    }

    pub(crate) fn for_node(node: &Node) -> Option<Self> {
        if !node.is_default_domain() {
            return None;
        }
        Self::ALL
            .iter()
            .copied()
            .find(|rule| rule.op_type() == node.op_type)
    }
}

/// Result of asking the native shape registry to resolve one plugin node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum SharedShapeResult {
    /// Every output rank and extent was resolved concretely.
    Resolved(Vec<Vec<usize>>),
    /// The rule left an output unknown or symbolic.
    SymbolicOrUnknown,
    /// The native rule rejected malformed input.
    Rejected(String),
}

/// Infer one node through the native registry using concrete plugin metadata.
///
/// Host-accessible rank-0/rank-1 values are supplied as [`ShapeData`], bounded
/// by [`MAX_SHAPE_DATA_ELEMS`]. Device values are deliberately not read: the
/// native rule remains symbolic and the caller can use its existing
/// Compute-time fallback.
///
/// The plugin graph reader stores ORT's `Node_GetSinceVersion` result in
/// [`Node::version`]. That is the selected kernel schema's `since_version`, not
/// necessarily the model's graph-level opset. This adapter therefore dispatches
/// with that exact local version. If it is absent or invalid, it deliberately
/// uses opset 1 rather than guessing the latest version. Future migrations with
/// version-dependent semantics must account for this contract explicitly.
pub(crate) fn infer_shared_node(node: &Node, inputs: &[TensorView<'_>]) -> SharedShapeResult {
    let input_ios = match inputs.iter().map(node_io).collect::<Result<Vec<_>, _>>() {
        Ok(inputs) => inputs,
        Err(reason) => return SharedShapeResult::Rejected(reason),
    };
    let opset = node.local_opset().unwrap_or(1);
    let mut imports = HashMap::new();
    imports.insert(normalize_domain(&node.domain).to_string(), opset);
    let mut interner = SymbolInterner::new(0x8000_0000);

    static REGISTRY: OnceLock<InferenceRegistry> = OnceLock::new();
    let outputs = match REGISTRY
        .get_or_init(InferenceRegistry::default_registry)
        .infer_node(
            node,
            &imports,
            input_ios,
            MergePolicy::Permissive,
            &mut interner,
        ) {
        Ok(outputs) => outputs,
        Err(error) => return SharedShapeResult::Rejected(error.to_string()),
    };

    let Some(shapes) = outputs
        .iter()
        .map(|output| {
            output
                .type_info
                .as_ref()?
                .shape
                .iter()
                .map(|dim| dim.as_const().and_then(|n| usize::try_from(n).ok()))
                .collect::<Option<Vec<_>>>()
        })
        .collect::<Option<Vec<_>>>()
    else {
        return SharedShapeResult::SymbolicOrUnknown;
    };

    if shapes.is_empty() {
        SharedShapeResult::SymbolicOrUnknown
    } else {
        SharedShapeResult::Resolved(shapes)
    }
}

fn node_io(input: &TensorView<'_>) -> Result<NodeIo, String> {
    if input.is_absent() {
        return Ok(NodeIo::default());
    }
    let shape = input
        .shape
        .iter()
        .map(|&extent| {
            i64::try_from(extent)
                .map(DimExpr::constant)
                .map_err(|_| format!("input extent {extent} exceeds i64::MAX"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let type_info = TypeInfo::new(input.dtype, shape);
    Ok(NodeIo {
        type_info: Some(type_info),
        shape_data: tensor_shape_data(input),
        value_type: None,
    })
}

fn tensor_shape_data(input: &TensorView<'_>) -> Option<ShapeData> {
    if !input.device.is_host_accessible() || input.shape.len() > 1 {
        return None;
    }
    let numel = input.shape.first().copied().unwrap_or(1);
    if numel > MAX_SHAPE_DATA_ELEMS {
        return None;
    }
    let element_size = input.dtype.byte_size();
    if element_size == 0 || (numel != 0 && input.data.is_null()) {
        return None;
    }

    let mut bytes = Vec::with_capacity(numel.checked_mul(element_size)?);
    let stride = isize::try_from(input.strides.first().copied().unwrap_or(1)).ok()?;
    let element_size_isize = isize::try_from(element_size).ok()?;
    let origin = input.data.as_ptr::<u8>();
    for index in 0..numel {
        let element_offset = isize::try_from(index).ok()?.checked_mul(stride)?;
        let byte_offset = element_offset.checked_mul(element_size_isize)?;
        // SAFETY: TensorView's contract makes each logical element address
        // readable, including negative strides; only the bounded scalar/vector
        // shape-data subset is copied.
        let element = unsafe { origin.add(input.byte_offset).offset(byte_offset) };
        // SAFETY: `element` addresses one complete element under that contract.
        bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(element, element_size) });
    }
    ShapeData::from_tensor(input.dtype, input.shape, &bytes)
}

#[cfg(test)]
mod tests {
    use onnx_runtime_ep_api::DevicePtr;
    use onnx_runtime_ir::{DataType, DeviceId, NodeId, ValueId};

    use super::*;

    fn node_at(op: &str, input_count: usize, version: Option<i64>) -> Node {
        let mut node = Node::new(
            NodeId(0),
            op,
            (0..input_count)
                .map(|index| Some(ValueId(index as u32)))
                .collect(),
            vec![ValueId(100)],
        );
        node.version = version;
        node
    }

    fn node(op: &str, input_count: usize) -> Node {
        node_at(op, input_count, Some(13))
    }

    fn view<'a>(
        data: *const u8,
        dtype: DataType,
        shape: &'a [usize],
        strides: &'a [i64],
        device: DeviceId,
    ) -> TensorView<'a> {
        TensorView::new(DevicePtr(data.cast()), dtype, shape, strides, device)
    }

    #[test]
    fn device_value_leaves_the_native_answer_symbolic() {
        let data = [0.0f32; 3];
        let target = [1i64, 4];
        let inputs = [
            view(
                data.as_ptr().cast(),
                DataType::Float32,
                &[3, 1],
                &[1, 1],
                DeviceId::cpu(),
            ),
            view(
                target.as_ptr().cast(),
                DataType::Int64,
                &[2],
                &[1],
                DeviceId::cuda(0),
            ),
        ];

        assert_eq!(
            infer_shared_node(&node("Expand", 2), &inputs),
            SharedShapeResult::SymbolicOrUnknown
        );
    }

    #[test]
    fn expand_dispatches_with_the_exact_node_version_not_latest() {
        let data = [0.0f32; 3];
        let target = [1i64, 4];
        let inputs = [
            view(
                data.as_ptr().cast(),
                DataType::Float32,
                &[3, 1],
                &[1, 1],
                DeviceId::cpu(),
            ),
            view(
                target.as_ptr().cast(),
                DataType::Int64,
                &[2],
                &[1],
                DeviceId::cpu(),
            ),
        ];

        assert_eq!(
            infer_shared_node(&node_at("Expand", 2, Some(7)), &inputs),
            SharedShapeResult::SymbolicOrUnknown,
            "Expand was introduced at opset 8"
        );
        assert_eq!(
            infer_shared_node(&node_at("Expand", 2, Some(8)), &inputs),
            SharedShapeResult::Resolved(vec![vec![3, 4]])
        );
        assert_eq!(
            infer_shared_node(&node_at("Expand", 2, None), &inputs),
            SharedShapeResult::SymbolicOrUnknown,
            "an absent version must use opset 1, not silently select latest"
        );
    }

    #[test]
    fn shared_rule_selection_requires_the_default_domain() {
        let mut foreign = node("Expand", 2);
        foreign.domain = "example.foreign".into();
        assert_eq!(SharedNativeShapeRule::for_node(&foreign), None);
        assert_eq!(
            SharedNativeShapeRule::for_node(&node("Expand", 2)),
            Some(SharedNativeShapeRule::Expand)
        );
    }

    #[test]
    fn unregistered_operator_is_symbolic_not_rejected() {
        assert_eq!(
            infer_shared_node(&node("NotRegisteredAnywhere", 0), &[]),
            SharedShapeResult::SymbolicOrUnknown
        );
    }

    #[test]
    fn malformed_input_is_a_rejection_not_an_unknown_shape() {
        let data = [0.0f32; 6];
        let repeats = [3i64];
        let inputs = [
            view(
                data.as_ptr().cast(),
                DataType::Float32,
                &[2, 3],
                &[3, 1],
                DeviceId::cpu(),
            ),
            view(
                repeats.as_ptr().cast(),
                DataType::Int64,
                &[1],
                &[1],
                DeviceId::cpu(),
            ),
        ];

        assert!(matches!(
            infer_shared_node(&node("Tile", 2), &inputs),
            SharedShapeResult::Rejected(reason) if reason.contains("input rank is 2")
        ));
    }

    #[test]
    fn strided_shape_data_is_copied_in_logical_order() {
        let data = [0.0f32; 3];
        let target = [1i64, 99, 4];
        let inputs = [
            view(
                data.as_ptr().cast(),
                DataType::Float32,
                &[3, 1],
                &[1, 1],
                DeviceId::cpu(),
            ),
            view(
                target.as_ptr().cast(),
                DataType::Int64,
                &[2],
                &[2],
                DeviceId::cpu(),
            ),
        ];

        assert_eq!(
            infer_shared_node(&node("Expand", 2), &inputs),
            SharedShapeResult::Resolved(vec![vec![3, 4]])
        );
    }
}
