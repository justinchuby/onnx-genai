//! Standard channel-wise normalization kernels and `PRelu`.
//!
//! All kernels accept f32/f64/f16/bf16 storage. f64 computes and accumulates in
//! f64, while f32/f16/bf16 compute and accumulate in f32. GroupNormalization
//! keeps the schema-version distinction: opset 18 affine parameters are
//! per-group, while opset 21 parameters are per-channel.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::add::{broadcast_apply, require_same_dtype};
use super::check_arity;
use crate::dispatch_float;
use crate::dtype::{ComputeDomain, NumericElem, to_dense, write_dense};
use crate::strided::numel;

trait NormAcc: ComputeDomain + PartialOrd {
    fn from_f32(value: f32) -> Self;
    fn sqrt(self) -> Self;
}

impl NormAcc for f32 {
    fn from_f32(value: f32) -> Self {
        value
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

impl NormAcc for f64 {
    fn from_f32(value: f32) -> Self {
        value as f64
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }
}

fn parameter<T: NumericElem>(op: &str, input: &TensorView) -> Result<Vec<T::Acc>> {
    require_same_dtype(op, input, T::DTYPE)?;
    Ok(to_dense::<T>(input)?.into_iter().map(T::to_acc).collect())
}

fn write_output<T: NumericElem>(out: &mut TensorMut, values: Vec<T::Acc>) -> Result<()> {
    let values = values.into_iter().map(T::from_acc).collect::<Vec<_>>();
    write_dense::<T>(out, &values)
}

fn require_x_output_shape(op: &str, input: &TensorView, output: &TensorMut) -> Result<()> {
    if input.shape != output.shape {
        return Err(EpError::KernelFailed(format!(
            "{op}: output shape {:?} must match X shape {:?}",
            output.shape, input.shape
        )));
    }
    Ok(())
}

fn require_channel_vector(op: &str, name: &str, shape: &[usize], len: usize) -> Result<()> {
    if shape != [len] {
        return Err(EpError::KernelFailed(format!(
            "{op}: {name} must have shape [{len}], got {shape:?}"
        )));
    }
    Ok(())
}

pub struct BatchNormFactory;

impl KernelFactory for BatchNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let training_mode = node
            .attr("training_mode")
            .and_then(|a| a.as_int())
            .unwrap_or(0);
        if training_mode != 0 {
            return Err(EpError::KernelFailed(
                "BatchNormalization: training_mode=1 is unsupported by the inference-only CPU EP"
                    .into(),
            ));
        }
        Ok(Box::new(BatchNormKernel {
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
        }))
    }
}

pub struct BatchNormKernel {
    epsilon: f32,
}

impl Kernel for BatchNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("BatchNormalization", inputs, outputs, 5, 5, 1)?;
        if outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "BatchNormalization: inference mode produces exactly 1 output, got {}",
                outputs.len()
            )));
        }
        dispatch_float!(inputs[0].dtype, "BatchNormalization", T =>
            batch_norm_typed::<T>(inputs, outputs, self.epsilon)
        )
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn batch_norm_typed<T: NumericElem>(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    epsilon: f32,
) -> Result<()>
where
    T::Acc: NormAcc,
{
    let shape = inputs[0].shape;
    if shape.len() < 2 {
        return Err(EpError::KernelFailed(
            "BatchNormalization: X must have rank at least 2 (N,C,...)".into(),
        ));
    }
    require_x_output_shape("BatchNormalization", &inputs[0], &outputs[0])?;
    let channels = shape[1];
    let scale = parameter::<T>("BatchNormalization", &inputs[1])?;
    let bias = parameter::<T>("BatchNormalization", &inputs[2])?;
    let mean = parameter::<T>("BatchNormalization", &inputs[3])?;
    let variance = parameter::<T>("BatchNormalization", &inputs[4])?;
    for (name, input, values) in [
        ("scale", &inputs[1], &scale),
        ("B", &inputs[2], &bias),
        ("input_mean", &inputs[3], &mean),
        ("input_var", &inputs[4], &variance),
    ] {
        require_channel_vector("BatchNormalization", name, input.shape, channels)?;
        debug_assert_eq!(values.len(), channels);
    }
    let spatial: usize = shape[2..].iter().product();
    if channels == 0 || spatial == 0 {
        return Err(EpError::KernelFailed(
            "BatchNormalization: channel and spatial dimensions must be non-empty".into(),
        ));
    }
    let x = parameter::<T>("BatchNormalization", &inputs[0])?;
    let epsilon = <T::Acc as NormAcc>::from_f32(epsilon);
    let y = x
        .iter()
        .enumerate()
        .map(|(i, &value)| {
            let channel = (i / spatial) % channels;
            value
                .c_sub(mean[channel])
                .c_div(variance[channel].c_add(epsilon).sqrt())
                .c_mul(scale[channel])
                .c_add(bias[channel])
        })
        .collect();
    write_output::<T>(&mut outputs[0], y)
}

pub struct InstanceNormFactory;

impl KernelFactory for InstanceNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(InstanceNormKernel {
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
        }))
    }
}

pub struct InstanceNormKernel {
    epsilon: f32,
}

impl Kernel for InstanceNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("InstanceNormalization", inputs, outputs, 3, 3, 1)?;
        dispatch_float!(inputs[0].dtype, "InstanceNormalization", T =>
            instance_norm_typed::<T>(inputs, outputs, self.epsilon)
        )
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn instance_norm_typed<T: NumericElem>(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    epsilon: f32,
) -> Result<()>
where
    T::Acc: NormAcc,
{
    let shape = inputs[0].shape;
    if shape.len() < 3 {
        return Err(EpError::KernelFailed(
            "InstanceNormalization: X must have rank at least 3 (N,C,spatial...)".into(),
        ));
    }
    require_x_output_shape("InstanceNormalization", &inputs[0], &outputs[0])?;
    let channels = shape[1];
    let spatial: usize = shape[2..].iter().product();
    if channels == 0 || spatial == 0 {
        return Err(EpError::KernelFailed(
            "InstanceNormalization: channel and spatial dimensions must be non-empty".into(),
        ));
    }
    let scale = parameter::<T>("InstanceNormalization", &inputs[1])?;
    let bias = parameter::<T>("InstanceNormalization", &inputs[2])?;
    require_channel_vector("InstanceNormalization", "scale", inputs[1].shape, channels)?;
    require_channel_vector("InstanceNormalization", "B", inputs[2].shape, channels)?;
    let x = parameter::<T>("InstanceNormalization", &inputs[0])?;
    let epsilon = <T::Acc as NormAcc>::from_f32(epsilon);
    let one = <T::Acc as NormAcc>::from_f32(1.0);
    let mut y = vec![T::Acc::default(); x.len()];
    for (instance_channel, slice) in x.chunks_exact(spatial).enumerate() {
        let channel = instance_channel % channels;
        let mean = slice
            .iter()
            .copied()
            .fold(T::Acc::default(), ComputeDomain::c_add)
            .c_div_usize(spatial);
        let variance = slice
            .iter()
            .map(|&value| {
                let centered = value.c_sub(mean);
                centered.c_mul(centered)
            })
            .fold(T::Acc::default(), ComputeDomain::c_add)
            .c_div_usize(spatial);
        let inv_std = one.c_div(variance.c_add(epsilon).sqrt());
        let base = instance_channel * spatial;
        for (offset, &value) in slice.iter().enumerate() {
            y[base + offset] = value
                .c_sub(mean)
                .c_mul(inv_std)
                .c_mul(scale[channel])
                .c_add(bias[channel]);
        }
    }
    write_output::<T>(&mut outputs[0], y)
}

#[derive(Clone, Copy)]
enum GroupAffine {
    PerGroup,
    PerChannel,
}

pub struct GroupNormFactory {
    pub since_version: u64,
}

impl KernelFactory for GroupNormFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let stash_type = node
            .attr("stash_type")
            .and_then(|a| a.as_int())
            .unwrap_or(1);
        if self.since_version >= 21 && stash_type != 1 {
            return Err(EpError::KernelFailed(format!(
                "GroupNormalization: stash_type {stash_type} unsupported (only 1 = float)"
            )));
        }
        let num_groups = node
            .attr("num_groups")
            .and_then(|a| a.as_int())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "GroupNormalization: required num_groups attribute is missing".into(),
                )
            })?;
        if num_groups <= 0 {
            return Err(EpError::KernelFailed(format!(
                "GroupNormalization: num_groups must be positive, got {num_groups}"
            )));
        }
        Ok(Box::new(GroupNormKernel {
            num_groups: num_groups as usize,
            epsilon: node
                .attr("epsilon")
                .and_then(|a| a.as_float())
                .unwrap_or(1e-5),
            affine: if self.since_version >= 21 {
                GroupAffine::PerChannel
            } else {
                GroupAffine::PerGroup
            },
        }))
    }
}

pub struct GroupNormKernel {
    num_groups: usize,
    epsilon: f32,
    affine: GroupAffine,
}

impl Kernel for GroupNormKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("GroupNormalization", inputs, outputs, 3, 3, 1)?;
        dispatch_float!(inputs[0].dtype, "GroupNormalization", T =>
            group_norm_typed::<T>(inputs, outputs, self.num_groups, self.epsilon, self.affine)
        )
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn group_norm_typed<T: NumericElem>(
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
    num_groups: usize,
    epsilon: f32,
    affine: GroupAffine,
) -> Result<()>
where
    T::Acc: NormAcc,
{
    let shape = inputs[0].shape;
    if shape.len() < 3 {
        return Err(EpError::KernelFailed(
            "GroupNormalization: X must have rank at least 3 (N,C,spatial...)".into(),
        ));
    }
    require_x_output_shape("GroupNormalization", &inputs[0], &outputs[0])?;
    let channels = shape[1];
    if channels == 0 || channels % num_groups != 0 {
        return Err(EpError::KernelFailed(format!(
            "GroupNormalization: channel count {channels} must be non-zero and divisible by num_groups {num_groups}"
        )));
    }
    let spatial: usize = shape[2..].iter().product();
    if spatial == 0 {
        return Err(EpError::KernelFailed(
            "GroupNormalization: spatial dimensions must be non-empty".into(),
        ));
    }
    let affine_len = match affine {
        GroupAffine::PerGroup => num_groups,
        GroupAffine::PerChannel => channels,
    };
    let scale = parameter::<T>("GroupNormalization", &inputs[1])?;
    let bias = parameter::<T>("GroupNormalization", &inputs[2])?;
    require_channel_vector("GroupNormalization", "scale", inputs[1].shape, affine_len)?;
    require_channel_vector("GroupNormalization", "bias", inputs[2].shape, affine_len)?;

    let x = parameter::<T>("GroupNormalization", &inputs[0])?;
    let channels_per_group = channels / num_groups;
    let group_size = channels_per_group * spatial;
    let groups_per_instance = num_groups;
    let epsilon = <T::Acc as NormAcc>::from_f32(epsilon);
    let one = <T::Acc as NormAcc>::from_f32(1.0);
    let mut y = vec![T::Acc::default(); x.len()];
    for (flat_group, slice) in x.chunks_exact(group_size).enumerate() {
        let group = flat_group % groups_per_instance;
        let mean = slice
            .iter()
            .copied()
            .fold(T::Acc::default(), ComputeDomain::c_add)
            .c_div_usize(group_size);
        let variance = slice
            .iter()
            .map(|&value| {
                let centered = value.c_sub(mean);
                centered.c_mul(centered)
            })
            .fold(T::Acc::default(), ComputeDomain::c_add)
            .c_div_usize(group_size);
        let inv_std = one.c_div(variance.c_add(epsilon).sqrt());
        let base = flat_group * group_size;
        for (offset, &value) in slice.iter().enumerate() {
            let channel_in_group = offset / spatial;
            let channel = group * channels_per_group + channel_in_group;
            let affine_index = match affine {
                GroupAffine::PerGroup => group,
                GroupAffine::PerChannel => channel,
            };
            y[base + offset] = value
                .c_sub(mean)
                .c_mul(inv_std)
                .c_mul(scale[affine_index])
                .c_add(bias[affine_index]);
        }
    }
    write_output::<T>(&mut outputs[0], y)
}

pub struct PReluFactory;

impl KernelFactory for PReluFactory {
    fn create(&self, _node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(PReluKernel))
    }
}

pub struct PReluKernel;

impl Kernel for PReluKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("PRelu", inputs, outputs, 2, 2, 1)?;
        dispatch_float!(inputs[0].dtype, "PRelu", T => prelu_typed::<T>(inputs, outputs))
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn prelu_typed<T: NumericElem>(inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()>
where
    T::Acc: NormAcc,
{
    require_x_output_shape("PRelu", &inputs[0], &outputs[0])?;
    if inputs[1].shape.len() > inputs[0].shape.len() {
        return Err(EpError::KernelFailed(format!(
            "PRelu: slope rank {} exceeds X rank {}",
            inputs[1].shape.len(),
            inputs[0].shape.len()
        )));
    }
    let x = parameter::<T>("PRelu", &inputs[0])?;
    let slope = parameter::<T>("PRelu", &inputs[1])?;
    let zero = <T::Acc as NormAcc>::from_f32(0.0);
    let mut slopes = vec![T::Acc::default(); numel(inputs[0].shape)];
    broadcast_apply(&slope, inputs[1].shape, inputs[0].shape, |i, value| {
        slopes[i] = value
    })
    .map_err(|_| {
        EpError::KernelFailed(format!(
            "PRelu: slope shape {:?} is not unidirectionally broadcastable to X shape {:?}",
            inputs[1].shape, inputs[0].shape
        ))
    })?;
    let y = x
        .into_iter()
        .zip(slopes)
        .map(|(value, slope)| {
            if value >= zero {
                value
            } else {
                value.c_mul(slope)
            }
        })
        .collect();
    write_output::<T>(&mut outputs[0], y)
}

#[cfg(test)]
mod tests {
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, DataType, Graph, Node, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    fn model_node(
        op: &str,
        opset: u64,
        input_shapes: &[&[usize]],
        output_shape: &[usize],
        attrs: &[(&str, Attribute)],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert(String::new(), opset);
        let inputs = input_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                let value = graph.create_named_value(
                    format!("input_{i}"),
                    DataType::Float32,
                    static_shape(shape.iter().copied()),
                );
                graph.add_input(value);
                Some(value)
            })
            .collect();
        let output = graph.create_named_value(
            "Y",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), op, inputs, vec![output]);
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "index {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn batch_normalization_inference_matches_reference() {
        let (graph, node) = model_node(
            "BatchNormalization",
            15,
            &[&[1, 2, 2], &[2], &[2], &[2], &[2]],
            &[1, 2, 2],
            &[("epsilon", Attribute::Float(0.0))],
        );
        let model = Model::new(&graph);
        let x = Owned::f32(&[1, 2, 2], &[1.0, 3.0, 10.0, 14.0]);
        let scale = Owned::f32(&[2], &[2.0, 0.5]);
        let bias = Owned::f32(&[2], &[1.0, -1.0]);
        let mean = Owned::f32(&[2], &[2.0, 12.0]);
        let variance = Owned::f32(&[2], &[1.0, 4.0]);
        let mut y = Owned::zeros_f32(&[1, 2, 2]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 15)
            .unwrap()
            .execute(
                &[
                    x.view(),
                    scale.view(),
                    bias.view(),
                    mean.view(),
                    variance.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(&y.to_f32(), &[-1.0, 3.0, -1.5, -0.5]);
    }

    #[test]
    fn batch_normalization_preserves_f64_precision() {
        let (graph, node) = model_node(
            "BatchNormalization",
            15,
            &[&[1, 1, 1], &[1], &[1], &[1], &[1]],
            &[1, 1, 1],
            &[("epsilon", Attribute::Float(0.0))],
        );
        let model = Model::new(&graph);
        let x = Owned::f64(&[1, 1, 1], &[16_777_217.0]);
        let scale = Owned::f64(&[1], &[1.0]);
        let bias = Owned::f64(&[1], &[0.0]);
        let mean = Owned::f64(&[1], &[0.0]);
        let variance = Owned::f64(&[1], &[1.0]);
        let mut y = Owned::zeros(DataType::Float64, &[1, 1, 1]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 15)
            .unwrap()
            .execute(
                &[
                    x.view(),
                    scale.view(),
                    bias.view(),
                    mean.view(),
                    variance.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_eq!(y.to_f64(), vec![16_777_217.0]);
    }

    #[test]
    fn instance_normalization_matches_reference() {
        let (graph, node) = model_node(
            "InstanceNormalization",
            6,
            &[&[1, 2, 2], &[2], &[2]],
            &[1, 2, 2],
            &[("epsilon", Attribute::Float(0.0))],
        );
        let model = Model::new(&graph);
        let x = Owned::f32(&[1, 2, 2], &[1.0, 3.0, 2.0, 6.0]);
        let scale = Owned::f32(&[2], &[2.0, 0.5]);
        let bias = Owned::f32(&[2], &[1.0, -1.0]);
        let mut y = Owned::zeros_f32(&[1, 2, 2]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 6)
            .unwrap()
            .execute(&[x.view(), scale.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &[-1.0, 3.0, -1.5, -0.5]);
    }

    #[test]
    fn group_normalization_opset18_per_group_matches_reference() {
        let (graph, node) = model_node(
            "GroupNormalization",
            18,
            &[&[1, 4, 1], &[2], &[2]],
            &[1, 4, 1],
            &[
                ("num_groups", Attribute::Int(2)),
                ("epsilon", Attribute::Float(0.0)),
            ],
        );
        let model = Model::new(&graph);
        let x = Owned::f32(&[1, 4, 1], &[1.0, 3.0, 10.0, 14.0]);
        let scale = Owned::f32(&[2], &[2.0, 0.5]);
        let bias = Owned::f32(&[2], &[1.0, -1.0]);
        let mut y = Owned::zeros_f32(&[1, 4, 1]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 18)
            .unwrap()
            .execute(&[x.view(), scale.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &[-1.0, 3.0, -1.5, -0.5]);
    }

    #[test]
    fn group_normalization_opset21_uses_per_channel_affine() {
        let (graph, node) = model_node(
            "GroupNormalization",
            21,
            &[&[1, 4, 1], &[4], &[4]],
            &[1, 4, 1],
            &[
                ("num_groups", Attribute::Int(2)),
                ("epsilon", Attribute::Float(0.0)),
            ],
        );
        let model = Model::new(&graph);
        let x = Owned::f32(&[1, 4, 1], &[1.0, 3.0, 10.0, 14.0]);
        let scale = Owned::f32(&[4], &[1.0, 2.0, 3.0, 4.0]);
        let bias = Owned::f32(&[4], &[0.0, 10.0, 20.0, 30.0]);
        let mut y = Owned::zeros_f32(&[1, 4, 1]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 21)
            .unwrap()
            .execute(&[x.view(), scale.view(), bias.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &[-1.0, 12.0, 17.0, 34.0]);
    }

    #[test]
    fn batch_normalization_rejects_training_and_extra_outputs() {
        let (mut graph, node) = model_node(
            "BatchNormalization",
            15,
            &[&[1, 1, 2], &[1], &[1], &[1], &[1]],
            &[1, 1, 2],
            &[("training_mode", Attribute::Int(1))],
        );
        let ep = CpuExecutionProvider::new();
        let err = match ep.get_kernel(graph.node(node), &[], 15) {
            Ok(_) => panic!("training BatchNormalization must be rejected"),
            Err(err) => err,
        };
        assert!(err.to_string().contains("training_mode=1"));

        graph
            .node_mut(node)
            .attributes
            .insert("training_mode".into(), Attribute::Int(0));
        let x = Owned::f32(&[1, 1, 2], &[1.0, 2.0]);
        let param = Owned::f32(&[1], &[1.0]);
        let mut y = Owned::zeros_f32(&[1, 1, 2]);
        let mut extra = Owned::zeros_f32(&[1]);
        let err = ep
            .get_kernel(graph.node(node), &[], 15)
            .unwrap()
            .execute(
                &[
                    x.view(),
                    param.view(),
                    param.view(),
                    param.view(),
                    param.view(),
                ],
                &mut [y.view_mut(), extra.view_mut()],
            )
            .unwrap_err();
        assert!(err.to_string().contains("exactly 1 output"));
    }

    #[test]
    fn prelu_broadcasts_per_channel_slope() {
        let (graph, node) = model_node(
            "PRelu",
            16,
            &[&[1, 2, 2, 2], &[2, 1, 1]],
            &[1, 2, 2, 2],
            &[],
        );
        let model = Model::new(&graph);
        let x = Owned::f32(&[1, 2, 2, 2], &[-1.0, 2.0, -3.0, 4.0, -1.0, 2.0, -3.0, 4.0]);
        let slope = Owned::f32(&[2, 1, 1], &[0.1, 0.5]);
        let mut y = Owned::zeros_f32(&[1, 2, 2, 2]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 16)
            .unwrap()
            .execute(&[x.view(), slope.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &[-0.1, 2.0, -0.3, 4.0, -0.5, 2.0, -1.5, 4.0]);
    }

    #[test]
    fn prelu_broadcasts_scalar_slope() {
        let (graph, node) = model_node("PRelu", 16, &[&[2, 2], &[]], &[2, 2], &[]);
        let model = Model::new(&graph);
        let x = Owned::f32(&[2, 2], &[-2.0, -1.0, 0.0, 3.0]);
        let slope = Owned::f32(&[], &[0.25]);
        let mut y = Owned::zeros_f32(&[2, 2]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 16)
            .unwrap()
            .execute(&[x.view(), slope.view()], &mut [y.view_mut()])
            .unwrap();
        assert_close(&y.to_f32(), &[-0.5, -0.25, 0.0, 3.0]);
    }

    #[test]
    fn prelu_preserves_f64_precision() {
        let (graph, node) = model_node("PRelu", 16, &[&[1], &[]], &[1], &[]);
        let model = Model::new(&graph);
        let x = Owned::f64(&[1], &[-16_777_217.0]);
        let slope = Owned::f64(&[], &[1.0]);
        let mut y = Owned::zeros(DataType::Float64, &[1]);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 16)
            .unwrap()
            .execute(&[x.view(), slope.view()], &mut [y.view_mut()])
            .unwrap();
        assert_eq!(y.to_f64(), vec![-16_777_217.0]);
    }
}
