use super::*;
use onnx_runtime_ep_cpu::{LayerWeightRegions, PlacementPlan, plan_placement};
use onnx_runtime_ir::{Graph, Node, WeightRef};
use onnx_runtime_loader::{
    ExpertQuantization, ExpertStorageOrder, ExpertTensorLayout, WeightRegionCatalog,
};
use onnx_runtime_session::InferenceSession;

impl From<PlacementPlan> for WeightPlacementReport {
    fn from(plan: PlacementPlan) -> Self {
        Self {
            coordinated_weight_budget_bytes: plan.coordinated_weight_budget_bytes,
            effective_budget_bytes: plan.effective_budget_bytes,
            device_bytes: plan.device_bytes,
            host_bytes: plan.host_bytes,
            explanation: plan.explanation,
        }
    }
}

/// Compute a load-time static placement plan from the same engine configuration
/// users set.
///
/// This is advisory today: the native executor still owns one EP per session,
/// so enforcing host/device partitions would be executor surgery. Computing the
/// plan here still removes the "right code that nothing reaches" failure mode:
/// invalid policies fail during load, and `--profile` can show the exact plan
/// that the next enforcement change must consume.
pub(crate) fn plan_static_weight_placement(
    session: &InferenceSession,
    policy: DevicePolicy,
    coordinated_weight_budget_bytes: u64,
) -> anyhow::Result<Option<WeightPlacementReport>> {
    let layers = qmoe_layers(session.graph())?;
    if layers.is_empty() {
        return Ok(None);
    }
    let (budget, gpu_layers) = match policy {
        DevicePolicy::Auto => (coordinated_weight_budget_bytes, None),
        DevicePolicy::Cpu => (0, None),
        DevicePolicy::GpuLayers(layers) => (coordinated_weight_budget_bytes, Some(layers)),
        DevicePolicy::DeviceBytes(bytes) => (coordinated_weight_budget_bytes.min(bytes), None),
    };
    let borrowed: Vec<_> = layers
        .iter()
        .map(|layer| LayerWeightRegions {
            layer_index: layer.layer_index,
            name: layer.name.as_str(),
            regions: layer.regions.as_slice(),
        })
        .collect();
    let plan = plan_placement(&borrowed, budget, gpu_layers)
        .context("planning static weight placement from device_policy")?;
    Ok(Some(plan.into()))
}

#[derive(Debug)]
struct OwnedLayerWeightRegions {
    layer_index: usize,
    name: String,
    regions: Vec<WeightRegionCatalog>,
}

fn qmoe_layers(graph: &Graph) -> anyhow::Result<Vec<OwnedLayerWeightRegions>> {
    let mut layers = Vec::new();
    for (_, node) in graph.nodes.iter() {
        if !is_qmoe(node) {
            continue;
        }
        if let Some(layer) = qmoe_layer(graph, node, layers.len())? {
            layers.push(layer);
        }
    }
    Ok(layers)
}

fn is_qmoe(node: &Node) -> bool {
    node.op_type == "QMoE" && (node.domain.is_empty() || node.domain == "com.microsoft")
}

fn qmoe_layer(
    graph: &Graph,
    node: &Node,
    layer_index: usize,
) -> anyhow::Result<Option<OwnedLayerWeightRegions>> {
    let bits = int_attr(node, "expert_weight_bits", 4)?;
    if !matches!(bits, 1 | 2 | 4 | 8) {
        return Ok(None);
    }
    let block_size = usize::try_from(int_attr(node, "block_size", 0)?)
        .context("QMoE block_size does not fit usize")?;
    if block_size == 0 {
        return Ok(None);
    }
    let quantization = |blocks_per_row| ExpertQuantization {
        bits: bits as usize,
        block_size,
        blocks_per_row,
    };

    let mut regions = Vec::new();
    push_qmoe_pair(graph, node, 2, 3, 11, quantization, &mut regions)?;
    push_qmoe_pair(graph, node, 5, 6, 12, quantization, &mut regions)?;
    if node.inputs.get(8).and_then(|slot| *slot).is_some() {
        push_qmoe_pair(graph, node, 8, 9, 13, quantization, &mut regions)?;
    }
    if regions.is_empty() {
        return Ok(None);
    }

    Ok(Some(OwnedLayerWeightRegions {
        layer_index,
        name: if node.name.is_empty() {
            format!("node#{}", node.id.0)
        } else {
            node.name.clone()
        },
        regions,
    }))
}

fn push_qmoe_pair(
    graph: &Graph,
    node: &Node,
    packed_index: usize,
    scales_index: usize,
    zero_points_index: usize,
    quantization: impl Fn(usize) -> ExpertQuantization,
    regions: &mut Vec<WeightRegionCatalog>,
) -> anyhow::Result<()> {
    let packed = required_initializer(graph, node, packed_index)?;
    let scales = required_initializer(graph, node, scales_index)?;
    let packed_dims = qmoe_dims(packed, packed_index)?;
    let scale_dims = qmoe_dims(scales, scales_index)?;
    let blocks_per_row = scale_dims[2];
    regions.push(WeightRegionCatalog::classify(
        packed,
        ExpertTensorLayout {
            version: 1,
            experts: packed_dims[0],
            rows_per_expert: packed_dims[1],
            storage_elements_per_row: packed_dims[2],
            order: ExpertStorageOrder::ExpertMajor,
            quantization: Some(quantization(blocks_per_row)),
        },
    ));
    regions.push(WeightRegionCatalog::classify(
        scales,
        ExpertTensorLayout {
            version: 1,
            experts: scale_dims[0],
            rows_per_expert: scale_dims[1],
            storage_elements_per_row: scale_dims[2],
            order: ExpertStorageOrder::ExpertMajor,
            quantization: Some(quantization(blocks_per_row)),
        },
    ));
    if let Some(zero_points) = optional_initializer(graph, node, zero_points_index)? {
        let zero_dims = qmoe_dims(zero_points, zero_points_index)?;
        regions.push(WeightRegionCatalog::classify(
            zero_points,
            ExpertTensorLayout {
                version: 1,
                experts: zero_dims[0],
                rows_per_expert: zero_dims[1],
                storage_elements_per_row: zero_dims[2],
                order: ExpertStorageOrder::ExpertMajor,
                quantization: Some(quantization(blocks_per_row)),
            },
        ));
    }
    Ok(())
}

fn required_initializer<'a>(
    graph: &'a Graph,
    node: &Node,
    input_index: usize,
) -> anyhow::Result<&'a WeightRef> {
    let value = node
        .inputs
        .get(input_index)
        .and_then(|slot| *slot)
        .with_context(|| format!("QMoE input {input_index} is absent"))?;
    initializer_or_initializer_cast(graph, value)
        .with_context(|| format!("QMoE input {input_index} is not a graph initializer"))
}

fn optional_initializer<'a>(
    graph: &'a Graph,
    node: &Node,
    input_index: usize,
) -> anyhow::Result<Option<&'a WeightRef>> {
    let Some(value) = node.inputs.get(input_index).and_then(|slot| *slot) else {
        return Ok(None);
    };
    Ok(Some(
        initializer_or_initializer_cast(graph, value).with_context(|| {
            format!("QMoE input {input_index} is present but is not a graph initializer")
        })?,
    ))
}

fn initializer_or_initializer_cast(
    graph: &Graph,
    value: onnx_runtime_ir::ValueId,
) -> Option<&WeightRef> {
    if let Some(weight) = graph.initializers.get(&value) {
        return Some(weight);
    }
    let producer = graph.values.get(value)?.producer?;
    let node = graph.nodes.get(producer)?;
    if node.domain.is_empty() && node.op_type == "Cast" {
        let source = node.inputs.first().and_then(|slot| *slot)?;
        return graph.initializers.get(&source);
    }
    None
}

fn qmoe_dims(weight: &WeightRef, input_index: usize) -> anyhow::Result<&[usize]> {
    let dims = weight.dims();
    anyhow::ensure!(
        dims.len() == 3,
        "QMoE input {input_index} must be rank-3 expert-major weight data, got {dims:?}"
    );
    Ok(dims)
}

fn int_attr(node: &Node, name: &'static str, default: i64) -> anyhow::Result<i64> {
    match node.attr(name) {
        Some(attr) => attr
            .as_int()
            .with_context(|| format!("QMoE attribute {name} must be an integer")),
        None => Ok(default),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, DataType, NodeId, TensorData, static_shape};

    #[test]
    fn qmoe_static_placement_accepts_cast_backed_scale_initializers() {
        let mut graph = Graph::new();
        let activation = graph.create_value(DataType::Float32, static_shape([1, 4]));
        graph.add_input(activation);

        let fc1_packed = initializer(&mut graph, DataType::Int4, vec![2, 3, 2]);
        let fc1_scales = cast_initializer(
            &mut graph,
            DataType::Float16,
            DataType::Float32,
            vec![2, 3, 1],
        );
        let fc2_packed = initializer(&mut graph, DataType::Int4, vec![2, 4, 2]);
        let fc2_scales = cast_initializer(
            &mut graph,
            DataType::Float16,
            DataType::Float32,
            vec![2, 4, 1],
        );
        let output = graph.create_value(DataType::Float32, static_shape([1, 4]));

        let mut node = Node::new(
            NodeId(0),
            "QMoE",
            vec![
                Some(activation),
                None,
                Some(fc1_packed),
                Some(fc1_scales),
                None,
                Some(fc2_packed),
                Some(fc2_scales),
            ],
            vec![output],
        );
        node.domain = "com.microsoft".to_owned();
        node.attributes
            .insert("block_size".to_owned(), Attribute::Int(32));
        graph.insert_node(node);

        let layers = qmoe_layers(&graph).expect("QMoE placement should accept Cast(initializer)");

        assert_eq!(layers.len(), 1);
        assert_eq!(layers[0].regions.len(), 4);
    }

    fn initializer(
        graph: &mut Graph,
        dtype: DataType,
        dims: Vec<usize>,
    ) -> onnx_runtime_ir::ValueId {
        let value = graph.create_value(dtype, static_shape(dims.iter().copied()));
        graph.set_initializer(
            value,
            WeightRef::Inline(TensorData::from_raw(dtype, dims, Vec::new())),
        );
        value
    }

    fn cast_initializer(
        graph: &mut Graph,
        source_dtype: DataType,
        cast_dtype: DataType,
        dims: Vec<usize>,
    ) -> onnx_runtime_ir::ValueId {
        let source = initializer(graph, source_dtype, dims.clone());
        let cast_output = graph.create_value(cast_dtype, static_shape(dims.iter().copied()));
        graph.insert_node(Node::new(
            NodeId(0),
            "Cast",
            vec![Some(source)],
            vec![cast_output],
        ));
        cast_output
    }
}
