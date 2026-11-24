//! Address-free geometry and one checked caller-funded QMoE scratch peak.
use super::*;
use onnx_runtime_ep_api::{
    OutputMetadata, TensorMetadata, WorkspaceLayout, WorkspaceLifetime, WorkspaceRequirement,
};

pub(super) const REGIONS: usize = 19;

#[derive(Clone, Copy, Debug)]
pub(super) struct ExpertGeometry {
    pub(super) out_features: usize,
    pub(super) in_features: usize,
    pub(super) packed_in: usize,
    pub(super) blocks: usize,
    pub(super) zero_point_bytes: usize,
}

#[allow(clippy::too_many_arguments)]
fn expert_geometry(
    name: &str,
    packed: TensorMetadata<'_>,
    scales: TensorMetadata<'_>,
    zero_points: Option<TensorMetadata<'_>>,
    bias: Option<TensorMetadata<'_>>,
    experts: usize,
    out_features: usize,
    in_features: usize,
    bits: usize,
    block_size: usize,
) -> Result<ExpertGeometry> {
    require_dtype(
        &format!("{name}_experts_weights"),
        packed.dtype,
        DataType::Uint8,
    )?;
    float_widen_entry(&format!("{name}_scales"), scales.dtype)?;
    let pack_size = 8 / bits;
    if !in_features.is_multiple_of(pack_size) {
        return Err(error(format!(
            "{name} input features {in_features} must be divisible by pack_size {pack_size}"
        )));
    }
    if !in_features.is_multiple_of(block_size) {
        return Err(error(format!(
            "{name} input features {in_features} must be divisible by block_size {block_size}"
        )));
    }
    let packed_in = in_features / pack_size;
    let blocks = in_features / block_size;
    let zero_point_bytes = checked_div_ceil(
        blocks,
        pack_size,
        &format!("{name} zero-point row byte count"),
    )?;
    require_shape(
        &format!("{name}_experts_weights"),
        packed.shape,
        &[experts, out_features, packed_in],
    )?;
    require_shape(
        &format!("{name}_scales"),
        scales.shape,
        &[experts, out_features, blocks],
    )?;
    if let Some(zero_points) = zero_points {
        require_dtype(
            &format!("{name}_zero_points"),
            zero_points.dtype,
            DataType::Uint8,
        )?;
        require_shape(
            &format!("{name}_zero_points"),
            zero_points.shape,
            &[experts, out_features, zero_point_bytes],
        )?;
    }
    if let Some(bias) = bias {
        float_widen_entry(&format!("{name}_experts_bias"), bias.dtype)?;
        require_shape(
            &format!("{name}_experts_bias"),
            bias.shape,
            &[experts, out_features],
        )?;
    }

    for (label, tensor) in [
        (format!("{name}_experts_weights"), Some(packed)),
        (format!("{name}_scales"), Some(scales)),
        (format!("{name}_zero_points"), zero_points),
        (format!("{name}_experts_bias"), bias),
    ] {
        if let Some(tensor) = tensor {
            checked_tensor_layout(&label, tensor.shape, tensor.dtype)?;
        }
    }
    Ok(ExpertGeometry {
        out_features,
        in_features,
        packed_in,
        blocks,
        zero_point_bytes,
    })
}

#[derive(Debug)]
pub(super) struct QMoEGeometry {
    pub(super) rows: usize,
    pub(super) hidden: usize,
    pub(super) experts: usize,
    pub(super) inter: usize,
    pub(super) fc1: ExpertGeometry,
    pub(super) fc2: ExpertGeometry,
    pub(super) fc3: Option<ExpertGeometry>,
    pub(super) fused_gate_up_decode: bool,
    pub(super) layout: WorkspaceLayout<REGIONS>,
}

impl QMoEGeometry {
    pub(super) fn requirement(&self) -> WorkspaceRequirement {
        if self.layout.bytes() == 0 {
            WorkspaceRequirement::NONE
        } else {
            WorkspaceRequirement {
                bytes: self.layout.bytes() as u64,
                alignment: self.layout.alignment(),
                lifetime: WorkspaceLifetime::SessionPersistent,
            }
        }
    }
}

fn optional<'a>(inputs: &[TensorMetadata<'a>], index: usize) -> Option<TensorMetadata<'a>> {
    inputs.get(index).filter(|input| input.present).copied()
}

pub(super) fn geometry(
    attributes: &MoeAttributes,
    bits: usize,
    block_size: usize,
    inputs: &[TensorMetadata<'_>],
    outputs: &[OutputMetadata<'_>],
) -> Result<QMoEGeometry> {
    if !(7..=21).contains(&inputs.len()) || outputs.len() != 1 {
        return Err(error(format!(
            "expected 7 to 21 inputs and exactly 1 output fact, got {} and {}",
            inputs.len(),
            outputs.len()
        )));
    }
    for (index, name) in [
        (0, "input"),
        (1, "router_probs"),
        (2, "fc1_experts_weights"),
        (3, "fc1_scales"),
        (5, "fc2_experts_weights"),
        (6, "fc2_scales"),
    ] {
        if !inputs[index].present {
            return Err(error(format!(
                "required input {index} ('{name}') is absent"
            )));
        }
    }
    if let Some(index) = inputs
        .iter()
        .enumerate()
        .skip(15)
        .find_map(|(i, x)| x.present.then_some(i))
    {
        return Err(error(format!(
            "input {index} is only used by deferred FP4/FP8 QMoE modes"
        )));
    }
    FloatDtype::from_input(inputs[0].dtype)?;
    float_widen_entry("router_probs", inputs[1].dtype)?;
    let shape = inputs[0].shape;
    if !matches!(shape.len(), 2 | 3) {
        return Err(error(format!(
            "input must be [rows, hidden] or [batch, sequence, hidden], got {shape:?}"
        )));
    }
    match &outputs[0] {
        OutputMetadata::Absent => return Err(error("required output 0 is absent")),
        OutputMetadata::Unresolved { dtype }
            if *dtype == inputs[0].dtype || *dtype == DataType::Undefined => {}
        OutputMetadata::Tensor { dtype, shape: out }
            if *dtype == inputs[0].dtype && out.as_ref() == shape => {}
        fact => {
            return Err(error(format!(
                "output fact {fact:?} must match input dtype {:?} and shape {shape:?}",
                inputs[0].dtype
            )));
        }
    }
    let hidden = shape[shape.len() - 1];
    let rows = checked_product(&shape[..shape.len() - 1], "flattened input row count")?;
    let experts = router_probs_experts(inputs[1].shape, rows)?;
    if attributes.k > experts {
        return Err(error(format!(
            "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
            attributes.k
        )));
    }
    if !hidden.is_multiple_of(block_size) {
        return Err(error(format!(
            "hidden_size {hidden} must be divisible by block_size {block_size}"
        )));
    }
    require_rank("fc2_experts_weights", inputs[5].shape, 3)?;
    if inputs[5].shape[0] != experts || inputs[5].shape[1] != hidden {
        return Err(error(format!(
            "fc2_experts_weights must start with [{experts}, {hidden}], got {:?}",
            inputs[5].shape
        )));
    }
    let inter = inputs[5].shape[2]
        .checked_mul(8 / bits)
        .ok_or_else(|| error("fc2 inter_size exceeds usize limits"))?;
    if inter == 0 || !inter.is_multiple_of(block_size) {
        return Err(error(format!(
            "inter_size {inter} must be nonzero and divisible by block_size {block_size}"
        )));
    }
    let fc1_size = attributes.fc1_size(inter)?;
    let fc1 = expert_geometry(
        "fc1",
        inputs[2],
        inputs[3],
        optional(inputs, 11),
        optional(inputs, 4),
        experts,
        fc1_size,
        hidden,
        bits,
        block_size,
    )?;
    let fc2 = expert_geometry(
        "fc2",
        inputs[5],
        inputs[6],
        optional(inputs, 12),
        optional(inputs, 7),
        experts,
        hidden,
        inter,
        bits,
        block_size,
    )?;
    let uses_separate_gate = attributes.uses_separate_gate(optional(inputs, 8).is_some());
    let fc3 = if uses_separate_gate {
        Some(expert_geometry(
            "fc3",
            optional(inputs, 8)
                .ok_or_else(|| error("unfused swiglu requires fc3_experts_weights"))?,
            optional(inputs, 9).ok_or_else(|| error("fc3_experts_weights requires fc3_scales"))?,
            optional(inputs, 13),
            optional(inputs, 10),
            experts,
            inter,
            hidden,
            bits,
            block_size,
        )?)
    } else {
        for index in [8, 9, 10, 13] {
            if optional(inputs, index).is_some() {
                return Err(error(format!(
                    "input {index} is only valid for a separate gated-GLU FC3"
                )));
            }
        }
        None
    };
    if let Some(weights) = optional(inputs, 14) {
        float_widen_entry("router_weights", weights.dtype)?;
        require_shape("router_weights", weights.shape, &[rows, experts])?;
    }
    for index in [0, 1, 14] {
        if let Some(input) = optional(inputs, index) {
            checked_tensor_layout("QMoE input", input.shape, input.dtype)?;
        }
    }
    let fused_gate_up_decode = rows == 1
        && attributes.k <= LINEAR_ONE_TASK_PER_BLOCK_MAX_ROUTES
        && ((fc3.is_some()
            && matches!(attributes.activation, Activation::Silu | Activation::Swiglu))
            || (fc3.is_none()
                && attributes.activation == Activation::Swiglu
                && attributes.swiglu_fusion != 0));
    let mut sizes = [0usize; REGIONS];
    if rows != 0 && hidden != 0 {
        let routes = checked_product(&[rows, attributes.k], "route count")?;
        sizes[0] = checked_bytes(routes, 4, "route indices")?;
        sizes[1] = checked_bytes(routes, 4, "route weights")?;
        let activation = checked_bytes(
            checked_product(&[routes, inter], "activation elements")?,
            4,
            "activation scratch",
        )?;
        if !fused_gate_up_decode {
            sizes[2] = checked_bytes(
                checked_product(&[routes, fc1_size], "FC1 elements")?,
                4,
                "FC1 scratch",
            )?;
            if fc3.is_some() {
                sizes[3] = activation;
            }
        }
        sizes[4] = activation;
        sizes[5] = checked_bytes(
            checked_product(&[routes, hidden], "route output elements")?,
            4,
            "route output scratch",
        )?;
        if rows > 1 {
            sizes[6] = checked_bytes(experts, 8, "expert counts")?;
            sizes[7] = checked_bytes(
                experts
                    .checked_add(1)
                    .ok_or_else(|| error("expert offset entry count exceeds usize limits"))?,
                8,
                "expert offsets",
            )?;
            sizes[8] = sizes[6];
            sizes[9] = checked_bytes(routes, 8, "grouped route indices")?;
            sizes[10] = checked_bytes(
                checked_product(&[routes, hidden.max(inter)], "grouped activation elements")?,
                4,
                "grouped activation scratch",
            )?;
        }
        for (input_index, region, name) in [
            (1, 11, "router_probs"),
            (14, 12, "router_weights"),
            (3, 13, "fc1_scales"),
            (6, 14, "fc2_scales"),
            (9, 15, "fc3_scales"),
            (4, 16, "fc1_bias"),
            (7, 17, "fc2_bias"),
            (10, 18, "fc3_bias"),
        ] {
            if let Some(input) = optional(inputs, input_index)
                && float_widen_entry(name, input.dtype)?.is_some()
            {
                sizes[region] =
                    checked_bytes(checked_product(input.shape, "widened elements")?, 4, name)?;
            }
        }
    }
    let mut regions = [std::alloc::Layout::new::<()>(); REGIONS];
    for (index, bytes) in sizes.into_iter().enumerate() {
        regions[index] = std::alloc::Layout::from_size_align(bytes, 256).map_err(|_| {
            error(format!(
                "QMoE workspace region {index} exceeds addressable capacity; reduce the input shape"
            ))
        })?;
    }
    Ok(QMoEGeometry {
        rows,
        hidden,
        experts,
        inter,
        fc1,
        fc2,
        fc3,
        fused_gate_up_decode,
        layout: WorkspaceLayout::new(regions)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::{Attribute, NodeId};
    fn attrs(activation: &str, fusion: i64) -> MoeAttributes {
        let mut node = Node::new(NodeId(0), "QMoE", vec![], vec![]);
        node.attributes.insert(
            "activation_type".into(),
            Attribute::String(activation.into()),
        );
        node.attributes
            .insert("swiglu_fusion".into(), Attribute::Int(fusion));
        MoeAttributes::from_node(&node).unwrap()
    }
    fn inputs(rows: usize, dtype: DataType) -> Vec<TensorMetadata<'static>> {
        let shape: &'static [usize] = match rows {
            0 => &[0, 16],
            1 => &[1, 16],
            _ => &[3, 16],
        };
        let router: &'static [usize] = match rows {
            0 => &[0, 2],
            1 => &[1, 2],
            _ => &[3, 2],
        };
        let mut values = vec![TensorMetadata::new(DataType::Undefined, &[], false); 15];
        values[0] = TensorMetadata::new(dtype, shape, true);
        values[1] = TensorMetadata::new(dtype, router, true);
        values[2] = TensorMetadata::new(DataType::Uint8, &[2, 32, 8], true);
        values[3] = TensorMetadata::new(dtype, &[2, 32, 1], true);
        values[5] = TensorMetadata::new(DataType::Uint8, &[2, 16, 16], true);
        values[6] = TensorMetadata::new(dtype, &[2, 16, 2], true);
        values
    }
    fn plan(activation: &str, fusion: i64, values: &[TensorMetadata<'_>]) -> QMoEGeometry {
        geometry(
            &attrs(activation, fusion),
            4,
            16,
            values,
            &[OutputMetadata::tensor(values[0].dtype, values[0].shape)],
        )
        .unwrap()
    }
    fn sizes(plan: &QMoEGeometry) -> [usize; REGIONS] {
        plan.layout
            .bind(Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(0x10000usize as *mut _),
                plan.layout.bytes(),
            )))
            .unwrap()
            .map(|view| view.bytes())
    }
    #[test]
    fn decode_and_grouped_plans_fund_one_composite_without_unused_widening() {
        let decode = plan("relu", 0, &inputs(1, DataType::Float32));
        let decode_sizes = sizes(&decode);
        assert_eq!(&decode_sizes[..6], &[4, 4, 128, 0, 128, 64]);
        assert!(decode_sizes[6..].iter().all(|&bytes| bytes == 0));
        let grouped = plan("relu", 0, &inputs(3, DataType::Float32));
        let group_sizes = sizes(&grouped);
        assert_eq!(&group_sizes[6..11], &[16, 24, 16, 24, 384]);
        assert!(group_sizes[11..].iter().all(|&bytes| bytes == 0));
        assert_eq!(
            grouped.requirement().lifetime,
            WorkspaceLifetime::SessionPersistent
        );
        assert_eq!(grouped.requirement().bytes, grouped.layout.bytes() as u64);
    }
    #[test]
    fn each_parameter_dtype_independently_selects_its_conversion_region() {
        let mut values = inputs(3, DataType::BFloat16);
        values[3].dtype = DataType::Float32;
        values[4] = TensorMetadata::new(DataType::Float16, &[2, 32], true);
        values[7] = TensorMetadata::new(DataType::Float32, &[2, 16], true);
        values[14] = TensorMetadata::new(DataType::Float16, &[3, 2], true);
        let bytes = sizes(&plan("relu", 0, &values));
        assert_eq!(&bytes[11..], &[24, 24, 0, 256, 0, 256, 0, 0]);
    }
    #[test]
    fn fused_decode_and_separate_gate_reserve_only_their_reachable_intermediates() {
        let mut fused = inputs(1, DataType::Float32);
        fused[2].shape = &[2, 64, 8];
        fused[3].shape = &[2, 64, 1];
        let p = plan("swiglu", 1, &fused);
        assert!(p.fused_gate_up_decode);
        assert_eq!(&sizes(&p)[2..4], &[0, 0]);
        let mut separate = inputs(1, DataType::Float32);
        separate[8] = TensorMetadata::new(DataType::Uint8, &[2, 32, 8], true);
        separate[9] = TensorMetadata::new(DataType::Float16, &[2, 32, 1], true);
        let p = plan("silu", 0, &separate);
        assert!(p.fc3.is_some());
        assert!(p.fused_gate_up_decode);
        assert_eq!(sizes(&p)[15], 256);
        separate[0].shape = &[3, 16];
        separate[1].shape = &[3, 2];
        let p = plan("silu", 0, &separate);
        assert!(!p.fused_gate_up_decode);
        assert_eq!(&sizes(&p)[2..4], &[384, 384]);
    }
    #[test]
    fn complete_binding_rejects_short_unaligned_and_missing_capacity_without_device_access() {
        let p = plan("relu", 0, &inputs(3, DataType::Float16));
        let bytes = p.layout.bytes();
        for view in [
            None,
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(0x10000usize as *mut _),
                bytes - 1,
            )),
            Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(0x10001usize as *mut _),
                bytes,
            )),
        ] {
            assert!(
                p.layout
                    .bind(view)
                    .unwrap_err()
                    .to_string()
                    .contains("prepare and pass")
            );
        }
        let regions = p
            .layout
            .bind(Some(WorkspaceView::new(
                onnx_runtime_ep_api::DevicePtrMut(0x10000usize as *mut _),
                bytes,
            )))
            .unwrap();
        let mut end = 0;
        for view in regions.into_iter().filter(|view| view.bytes() != 0) {
            let start = view.ptr().0 as usize;
            assert!(start >= end);
            assert!(start.is_multiple_of(256));
            end = start + view.bytes();
        }
    }
    #[test]
    fn unresolved_output_is_inferred_but_wrong_known_facts_and_holes_fail() {
        let values = inputs(1, DataType::Float32);
        let a = attrs("relu", 0);
        let inferred = geometry(
            &a,
            4,
            16,
            &values,
            &[OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            }],
        )
        .unwrap()
        .requirement();
        assert_eq!(inferred, plan("relu", 0, &values).requirement());
        assert_eq!(
            geometry(
                &a,
                4,
                16,
                &values,
                &[OutputMetadata::Unresolved {
                    dtype: DataType::Undefined
                }]
            )
            .unwrap()
            .requirement(),
            inferred
        );
        for output in [
            OutputMetadata::Absent,
            OutputMetadata::tensor(DataType::Float16, &[1, 16]),
            OutputMetadata::tensor(DataType::Float32, &[]),
        ] {
            assert!(geometry(&a, 4, 16, &values, &[output]).is_err());
        }
        let mut missing = values.clone();
        missing[3].present = false;
        assert!(
            geometry(
                &a,
                4,
                16,
                &missing,
                &[OutputMetadata::Unresolved {
                    dtype: DataType::Float32
                }]
            )
            .unwrap_err()
            .to_string()
            .contains("fc1_scales")
        );
    }
    #[test]
    fn empty_rows_require_no_storage_and_large_geometry_fails_checked() {
        let empty = plan("relu", 0, &inputs(0, DataType::Float16));
        assert_eq!(empty.requirement(), WorkspaceRequirement::NONE);
        empty.layout.bind(None).unwrap();
        let mut huge = inputs(1, DataType::Float32);
        huge[0].shape = &[usize::MAX, 16];
        huge[1].shape = &[usize::MAX, 2];
        let error = geometry(
            &attrs("relu", 0),
            4,
            16,
            &huge,
            &[OutputMetadata::Unresolved {
                dtype: DataType::Float32,
            }],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("exceeds"), "{error}");
    }
}
