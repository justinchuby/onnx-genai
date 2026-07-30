use super::*;

/// Recompute the output shape of standard elementwise broadcasting ops from
/// their concrete runtime inputs. Loader inference is only a prior: a
/// data-dependent upstream value may acquire a different live shape.
pub(super) fn runtime_elementwise_output_shape(
    node: &Node,
    input_shapes: &[Vec<usize>],
) -> Option<std::result::Result<Vec<usize>, onnx_runtime_ir::IrError>> {
    if !node.is_default_domain() {
        return None;
    }

    let input_count = match node.op_type.as_str() {
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Mod" | "BitShift" | "Less" | "Greater"
        | "Equal" | "And" | "Or" | "Xor" | "LessOrEqual" | "GreaterOrEqual" => 2,
        "Where" => 3,
        "Min" | "Max" | "Sum" | "Mean" => input_shapes.len(),
        _ => return None,
    };
    if input_count == 0 || input_shapes.len() < input_count {
        return None;
    }

    let mut shape = input_shapes[0].clone();
    for input in &input_shapes[1..input_count] {
        shape = match broadcast_shapes(&shape, input) {
            Ok(shape) => shape,
            Err(error) => return Some(Err(error)),
        };
    }
    Some(Ok(shape))
}

/// Compute concrete output shapes from already-resolved input shapes and the
/// runtime *values* of integer inputs. This is the executor's fallback for the
/// rare value whose shape the loader's static (symbolic) inference could not pin
/// down — e.g. a `Slice` whose `ends` is produced by a runtime
/// `Shape → Min → Cast` chain, followed by movement/broadcast nodes.
///
/// Model-agnostic: it dispatches on the op type alone. Returns `None` for ops
/// this executor cannot resolve dynamically, which surfaces as
/// [`SessionError::UnresolvedShape`] exactly as before.
pub(super) fn dynamic_output_shapes(
    node: &Node,
    input_shapes: &[Vec<usize>],
    input_dtypes: &[DataType],
    input_values: &[Option<Vec<i64>>],
    input_float_values: &[Option<Vec<f64>>],
    opset: u64,
) -> Option<Vec<Vec<usize>>> {
    // A plugin-fused node stands for a whole claimed subgraph, so no per-op
    // rule can describe its outputs. It carries a subgraph the executor
    // replays instead, which is exact rather than reconstructed -- a wrong KV
    // shape here would corrupt the cache while still producing plausible text.
    if node.domain == crate::plugin_provider::PLUGIN_FUSED_DOMAIN
        && node.op_type == crate::plugin_provider::PLUGIN_FUSED_OP
    {
        return plugin_fused_output_shapes(
            node,
            input_shapes,
            input_dtypes,
            input_values,
            input_float_values,
        );
    }
    match node.op_type.as_str() {
        "Resize" if node.is_default_domain() => {
            let input = input_shapes.first()?;
            let rank = input.len();
            let axes = if let Some(raw) = node.attr("axes").and_then(Attribute::as_ints) {
                let mut axes = Vec::with_capacity(raw.len());
                for &axis in raw {
                    let axis = if axis < 0 { axis + rank as i64 } else { axis };
                    let axis = usize::try_from(axis).ok()?;
                    if axis >= rank || axes.contains(&axis) {
                        return None;
                    }
                    axes.push(axis);
                }
                if axes.is_empty() {
                    (0..rank).collect()
                } else {
                    axes
                }
            } else {
                (0..rank).collect()
            };
            let scales_index = if opset == 10 { 1 } else { 2 };
            let scales = input_float_values
                .get(scales_index)
                .and_then(|values| values.as_deref())
                .filter(|values| !values.is_empty());
            let sizes = (opset >= 11)
                .then(|| input_values.get(3).and_then(|values| values.as_deref()))
                .flatten()
                .filter(|values| !values.is_empty());
            if scales.is_some() == sizes.is_some() {
                return None;
            }
            let mut output = input.clone();
            if let Some(scales) = scales {
                if scales.len() != axes.len()
                    || node
                        .attr("keep_aspect_ratio_policy")
                        .and_then(Attribute::as_str)
                        .is_some_and(|policy| policy != "stretch")
                {
                    return None;
                }
                for (&axis, &scale) in axes.iter().zip(scales) {
                    if !scale.is_finite() || scale <= 0.0 {
                        return None;
                    }
                    let extent = input[axis] as f64 * scale;
                    if extent > usize::MAX as f64 {
                        return None;
                    }
                    output[axis] = extent.floor() as usize;
                }
            } else {
                let sizes = sizes?;
                if sizes.len() != axes.len() {
                    return None;
                }
                let requested = sizes
                    .iter()
                    .map(|&size| usize::try_from(size).ok().filter(|&size| size > 0))
                    .collect::<Option<Vec<_>>>()?;
                match node
                    .attr("keep_aspect_ratio_policy")
                    .and_then(Attribute::as_str)
                    .unwrap_or("stretch")
                {
                    "stretch" => {
                        for (&axis, &size) in axes.iter().zip(&requested) {
                            output[axis] = size;
                        }
                    }
                    policy @ ("not_larger" | "not_smaller") => {
                        if axes.iter().any(|&axis| input[axis] == 0) {
                            return None;
                        }
                        let (numerator, denominator) = axes
                            .iter()
                            .zip(&requested)
                            .map(|(&axis, &size)| (size, input[axis]))
                            .reduce(|left, right| {
                                let order = (left.0 as u128 * right.1 as u128)
                                    .cmp(&(right.0 as u128 * left.1 as u128));
                                if (policy == "not_larger" && order.is_le())
                                    || (policy == "not_smaller" && order.is_ge())
                                {
                                    left
                                } else {
                                    right
                                }
                            })?;
                        if denominator == 0 {
                            return None;
                        }
                        for &axis in &axes {
                            let product = (input[axis] as u128).checked_mul(numerator as u128)?;
                            output[axis] = usize::try_from(
                                (product + denominator as u128 / 2) / denominator as u128,
                            )
                            .ok()?;
                        }
                    }
                    _ => return None,
                }
            }
            Some(vec![output])
        }
        // Opset-10+ `Slice`: data, starts, ends, [axes], [steps] as inputs. The
        // per-axis element count mirrors the `Slice` kernel's clamp semantics
        // exactly (ONNX reference), so the buffer we size here matches what the
        // kernel writes.
        "Slice" if node.is_default_domain() => {
            let data_shape = input_shapes.first()?;
            let starts = input_values.get(1)?.as_ref()?;
            let ends = input_values.get(2)?.as_ref()?;
            let (axes, steps) = onnx_runtime_ep_cpu::slice_axes_steps(
                starts.len(),
                input_values.get(3).and_then(|v| v.as_deref()),
                input_values.get(4).and_then(|v| v.as_deref()),
            );
            // Reuse the exact kernel geometry helper so the buffer we size here
            // always matches what the Slice kernel writes. Any error (length
            // mismatch, out-of-range axis, zero step) means "cannot resolve".
            let plan =
                onnx_runtime_ep_cpu::slice_plan(data_shape, starts, ends, &axes, &steps).ok()?;
            let count: Vec<usize> = plan.iter().map(|p| p.count).collect();
            Some(vec![count])
        }
        "NonMaxSuppression" if node.is_default_domain() => {
            let boxes_shape = input_shapes.first()?;
            let scores_shape = input_shapes.get(1)?;
            let boxes = input_float_values.first()?.as_ref()?;
            let scores = input_float_values.get(1)?.as_ref()?;
            let max_output_boxes_per_class = input_values
                .get(2)
                .and_then(|value| value.as_ref())
                .filter(|value| value.len() == 1)
                .map(|value| value[0])
                .unwrap_or(0);
            let iou_threshold = input_float_values
                .get(3)
                .and_then(|value| value.as_ref())
                .filter(|value| value.len() == 1)
                .map(|value| value[0] as f32)
                .unwrap_or(0.0);
            let score_threshold = input_float_values
                .get(4)
                .and_then(|value| value.as_ref())
                .filter(|value| value.len() == 1)
                .map(|value| value[0] as f32)
                .unwrap_or(f32::NEG_INFINITY);
            let center_point_box = node
                .attr("center_point_box")
                .and_then(Attribute::as_int)
                .unwrap_or(0);
            let boxes = boxes.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let scores = scores.iter().map(|&value| value as f32).collect::<Vec<_>>();
            let selected = onnx_runtime_ep_cpu::non_max_suppression(
                &boxes,
                boxes_shape,
                &scores,
                scores_shape,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                center_point_box,
            )
            .ok()?;
            Some(vec![vec![selected.len(), 3]])
        }
        "GroupQueryAttention" if node.domain == "com.microsoft" => {
            let query = input_shapes.first()?;
            let past_key = input_shapes.get(3)?;
            if query.len() != 3 || past_key.len() != 4 {
                return None;
            }
            let num_heads = usize::try_from(node.attr("num_heads")?.as_int()?).ok()?;
            let kv_heads = usize::try_from(node.attr("kv_num_heads")?.as_int()?).ok()?;
            if num_heads == 0 || kv_heads == 0 {
                return None;
            }
            let (output, head_dim) = if node.inputs.get(1).and_then(|input| *input).is_some() {
                let key = input_shapes.get(1)?;
                if key.len() != 3 || !key[2].is_multiple_of(kv_heads) {
                    return None;
                }
                (query.clone(), key[2] / kv_heads)
            } else {
                let packed_heads = num_heads.checked_add(kv_heads.checked_mul(2)?)?;
                if !query[2].is_multiple_of(packed_heads) {
                    return None;
                }
                let head_dim = query[2] / packed_heads;
                (
                    vec![query[0], query[1], head_dim.checked_mul(num_heads)?],
                    head_dim,
                )
            };
            let total_sequence_values = input_values.get(6)?.as_ref()?;
            if total_sequence_values.len() != 1 {
                return None;
            }
            let total_sequence = usize::try_from(total_sequence_values[0]).ok()?;
            let present_sequence = past_key[2].max(total_sequence);
            let present = vec![query[0], kv_heads, present_sequence, head_dim];
            let mut shapes = vec![output];
            if node.outputs.len() >= 2 {
                shapes.push(present.clone());
            }
            if node.outputs.len() >= 3 {
                shapes.push(present);
            }
            Some(shapes)
        }
        _ => {
            // Re-run the standard, opset-aware shape rule with the concrete
            // runtime input shapes and any small integer input values now
            // available. This covers shape-preserving movement and broadcasting
            // ops after a data-dependent node without duplicating their ONNX
            // semantics here (notably Unsqueeze axis normalization).
            let inputs = node
                .inputs
                .iter()
                .enumerate()
                .map(|(i, input)| {
                    if input.is_none() {
                        return Some(NodeIo::default());
                    }
                    let shape = input_shapes
                        .get(i)?
                        .iter()
                        .map(|&dim| i64::try_from(dim).ok().map(DimExpr::constant))
                        .collect::<Option<Vec<_>>>()?;
                    let dtype = *input_dtypes.get(i)?;
                    let shape_data = input_values.get(i)?.as_ref().and_then(|values| {
                        let elems = values
                            .iter()
                            .copied()
                            .map(DimExpr::constant)
                            .collect::<Vec<_>>();
                        match input_shapes[i].as_slice() {
                            [] if elems.len() == 1 => {
                                Some(ShapeData::scalar(dtype, elems[0].clone()))
                            }
                            [len] if *len == elems.len() => Some(ShapeData::vector(dtype, elems)),
                            _ => None,
                        }
                    });
                    Some(NodeIo {
                        type_info: Some(TypeInfo::new(dtype, shape)),
                        shape_data,
                        value_type: None,
                    })
                })
                .collect::<Option<Vec<_>>>()?;
            let mut imports = HashMap::new();
            imports.insert(node.domain.clone(), opset);
            let mut interner = SymbolInterner::new(0x8000_0000);
            static REGISTRY: std::sync::OnceLock<InferenceRegistry> = std::sync::OnceLock::new();
            REGISTRY
                .get_or_init(InferenceRegistry::default_registry)
                .infer_node(node, &imports, inputs, MergePolicy::Strict, &mut interner)
                .ok()?
                .into_iter()
                .map(|output| {
                    output
                        .type_info?
                        .shape
                        .into_iter()
                        .map(|dim| usize::try_from(dim.as_const()?).ok())
                        .collect()
                })
                .collect()
        }
    }
}

pub(super) fn plugin_fused_output_shapes(
    node: &Node,
    input_shapes: &[Vec<usize>],
    input_dtypes: &[DataType],
    input_values: &[Option<Vec<i64>>],
    input_float_values: &[Option<Vec<f64>>],
) -> Option<Vec<Vec<usize>>> {
    let shape_graph = match node.attr(crate::plugin_provider::FUSION_SHAPE_GRAPH_ATTR)? {
        Attribute::Graph(graph) => graph.as_ref(),
        _ => return None,
    };
    if shape_graph.inputs.len() != input_shapes.len() {
        return None;
    }

    let mut resolved: HashMap<ValueId, Vec<usize>> = HashMap::new();
    let mut shape_data: HashMap<ValueId, ShapeData> = HashMap::new();
    let mut bindings: HashMap<SymbolId, usize> = HashMap::new();
    for (index, &value) in shape_graph.inputs.iter().enumerate() {
        let shape = input_shapes.get(index)?.clone();
        bind_shape_symbols(
            shape_graph.value(value).shape.as_slice(),
            &shape,
            &mut bindings,
        )?;
        resolved.insert(value, shape.clone());
        if let Some(data) = shape_data_from_runtime_input(
            input_dtypes[index],
            &shape,
            input_values.get(index).and_then(|v| v.as_ref()),
            input_float_values.get(index).and_then(|v| v.as_ref()),
        ) {
            shape_data.insert(value, data);
        }
    }

    for (&value, weight) in &shape_graph.initializers {
        let dims = weight.dims().to_vec();
        bind_shape_symbols(
            shape_graph.value(value).shape.as_slice(),
            &dims,
            &mut bindings,
        )?;
        resolved.insert(value, dims);
        if let WeightRef::Inline(tensor) = weight
            && let Some(data) = ShapeData::from_tensor(tensor.dtype, &tensor.dims, &tensor.data)
        {
            shape_data.insert(value, data);
        }
    }

    let registry = InferenceRegistry::default_registry();
    let order = shape_graph.topological_order().ok()?;
    for node_id in order {
        let inner = shape_graph.node(node_id);
        let mut node_input_shapes = Vec::with_capacity(inner.inputs.len());
        let mut node_input_dtypes = Vec::with_capacity(inner.inputs.len());
        let mut node_input_values = Vec::with_capacity(inner.inputs.len());
        let mut node_input_float_values = Vec::with_capacity(inner.inputs.len());
        let inputs = inner
            .inputs
            .iter()
            .map(|slot| {
                let Some(value) = slot else {
                    node_input_shapes.push(Vec::new());
                    node_input_dtypes.push(DataType::Float32);
                    node_input_values.push(None);
                    node_input_float_values.push(None);
                    return Some(NodeIo::default());
                };
                let value_meta = shape_graph.value(*value);
                let shape = resolved
                    .get(value)
                    .cloned()
                    .or_else(|| substitute(&value_meta.shape, &bindings))?;
                node_input_shapes.push(shape.clone());
                node_input_dtypes.push(value_meta.dtype);
                let data = shape_data.get(value).cloned();
                node_input_values.push(data.as_ref().and_then(shape_data_as_i64));
                node_input_float_values.push(data.as_ref().and_then(shape_data_as_f64));
                Some(NodeIo {
                    type_info: Some(TypeInfo::new(
                        value_meta.dtype,
                        shape
                            .iter()
                            .map(|&dim| i64::try_from(dim).ok().map(DimExpr::constant))
                            .collect::<Option<Vec<_>>>()?,
                    )),
                    shape_data: data,
                    value_type: None,
                })
            })
            .collect::<Option<Vec<_>>>()?;

        let mut interner = SymbolInterner::new(0x8000_0000);
        if let Ok(outputs) = registry.infer_node(
            inner,
            &shape_graph.opset_imports,
            inputs,
            MergePolicy::Strict,
            &mut interner,
        ) {
            for (&output, io) in inner.outputs.iter().zip(outputs) {
                if let Some(type_info) = io.type_info {
                    let concrete = type_info
                        .shape
                        .iter()
                        .map(|dim| usize::try_from(dim.as_const()?).ok())
                        .collect::<Option<Vec<_>>>();
                    if let Some(shape) = concrete {
                        bind_shape_symbols(
                            shape_graph.value(output).shape.as_slice(),
                            &shape,
                            &mut bindings,
                        )?;
                        resolved.insert(output, shape);
                    }
                }
                if let Some(data) = io.shape_data
                    && data.within_bounds()
                {
                    shape_data.insert(output, data);
                }
            }
        }

        if inner
            .outputs
            .iter()
            .any(|output| !resolved.contains_key(output))
        {
            let inner_opset = effective_opset(shape_graph, inner);
            let out_shapes = dynamic_output_shapes(
                inner,
                &node_input_shapes,
                &node_input_dtypes,
                &node_input_values,
                &node_input_float_values,
                inner_opset,
            )?;
            if out_shapes.len() != inner.outputs.len() {
                return None;
            }
            for (&output, shape) in inner.outputs.iter().zip(out_shapes) {
                bind_shape_symbols(
                    shape_graph.value(output).shape.as_slice(),
                    &shape,
                    &mut bindings,
                )?;
                resolved.insert(output, shape);
            }
        }
    }

    shape_graph
        .outputs
        .iter()
        .map(|output| {
            resolved
                .get(output)
                .cloned()
                .or_else(|| substitute(&shape_graph.value(*output).shape, &bindings))
        })
        .collect()
}

pub(super) fn bind_shape_symbols(
    declared: &[Dim],
    concrete: &[usize],
    bindings: &mut HashMap<SymbolId, usize>,
) -> Option<()> {
    if declared.len() != concrete.len() {
        return None;
    }
    for (dim, &actual) in declared.iter().zip(concrete) {
        match dim {
            Dim::Static(expected) if *expected != actual => return None,
            Dim::Static(_) => {}
            Dim::Symbolic(symbol) => match bindings.get(symbol) {
                Some(&previous) if previous != actual => return None,
                Some(_) => {}
                None => {
                    bindings.insert(*symbol, actual);
                }
            },
        }
    }
    Some(())
}

fn shape_data_from_runtime_input(
    dtype: DataType,
    shape: &[usize],
    ints: Option<&Vec<i64>>,
    floats: Option<&Vec<f64>>,
) -> Option<ShapeData> {
    if let Some(values) = ints {
        if shape.is_empty() && values.len() == 1 {
            return Some(ShapeData::scalar(dtype, DimExpr::constant(values[0])));
        }
        if matches!(shape, [len] if *len == values.len()) {
            return Some(ShapeData::vector(
                dtype,
                values.iter().copied().map(DimExpr::constant).collect(),
            ));
        }
    }
    if let Some(values) = floats {
        if shape.is_empty() && values.len() == 1 {
            return Some(ShapeData::float_scalar(dtype, values[0]));
        }
        if matches!(shape, [len] if *len == values.len()) {
            return Some(ShapeData::float_vector(dtype, values.clone()));
        }
    }
    None
}

fn shape_data_as_i64(data: &ShapeData) -> Option<Vec<i64>> {
    data.elems.iter().map(DimExpr::as_const).collect()
}

fn shape_data_as_f64(data: &ShapeData) -> Option<Vec<f64>> {
    data.float_elems.clone()
}
