//! Float32 reference kernel for ORT 1.27 `com.microsoft::MoE`.
//!
//! The positional inputs are:
//! `input`, `router_probs`, `fc1_experts_weights`, `fc1_experts_bias?`,
//! `fc2_experts_weights`, `fc2_experts_bias?`, `fc3_experts_weights?`,
//! `fc3_experts_bias?`. Weights use ORT's expert-major canonical layout.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::gelu::exact_gelu;
use super::{check_arity, to_dense_f32, write_dense_f32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Activation {
    Relu,
    Gelu,
    Silu,
    Swiglu,
    Identity,
}

/// Factory for the ORT contrib `MoE` operator.
pub struct MoEFactory;

/// Correctness-first dense, per-row float MoE kernel.
///
/// Phase 2 can replace the row loop with batch-union expert grouping so each
/// selected expert's weights are consumed once for all routed rows.
pub struct MoEKernel {
    k: usize,
    activation: Activation,
    normalize_routing_weights: bool,
    swiglu_fusion: usize,
    activation_alpha: f32,
    activation_beta: f32,
    swiglu_limit: f32,
}

impl KernelFactory for MoEFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let k = required_int(node, "k")?;
        if k <= 0 {
            return Err(error(format!("k must be > 0, got {k}")));
        }
        let activation_name = node
            .attr("activation_type")
            .and_then(|a| a.as_str())
            .ok_or_else(|| error("required string attribute activation_type is missing"))?;
        let activation = match activation_name {
            "relu" => Activation::Relu,
            "gelu" => Activation::Gelu,
            "silu" => Activation::Silu,
            "swiglu" => Activation::Swiglu,
            "identity" => Activation::Identity,
            other => {
                return Err(error(format!(
                    "unsupported activation_type '{other}' (supported: relu, gelu, silu, swiglu, identity)"
                )));
            }
        };
        let normalize = bool_attr(node, "normalize_routing_weights", false)?;
        if bool_attr(node, "use_sparse_mixer", false)? {
            return Err(error(
                "use_sparse_mixer=1 is unsupported by the Phase-1 CPU reference kernel",
            ));
        }
        let swiglu_fusion = int_attr(node, "swiglu_fusion", 0)?;
        if !(0..=2).contains(&swiglu_fusion) {
            return Err(error(format!(
                "swiglu_fusion must be 0, 1, or 2, got {swiglu_fusion}"
            )));
        }
        if activation != Activation::Swiglu && swiglu_fusion != 0 {
            return Err(error(
                "swiglu_fusion is only valid when activation_type='swiglu'",
            ));
        }
        if int_attr(node, "block_size", 0)? != 0 {
            return Err(error(
                "block_size is a QMoE quantization attribute and is unsupported by float MoE",
            ));
        }
        Ok(Box::new(MoEKernel {
            k: k as usize,
            activation,
            normalize_routing_weights: normalize,
            swiglu_fusion: swiglu_fusion as usize,
            activation_alpha: float_attr(node, "activation_alpha", 1.0)?,
            activation_beta: float_attr(node, "activation_beta", 0.0)?,
            swiglu_limit: float_attr(node, "swiglu_limit", f32::INFINITY)?,
        }))
    }
}

impl Kernel for MoEKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("MoE", inputs, outputs, 5, 8, 1)?;
        if outputs.len() != 1 {
            return Err(error(format!(
                "expected exactly 1 output, got {}",
                outputs.len()
            )));
        }
        for (index, name) in [
            (0, "input"),
            (1, "router_probs"),
            (2, "fc1_experts_weights"),
            (4, "fc2_experts_weights"),
        ] {
            if inputs[index].is_absent() {
                return Err(error(format!(
                    "required input {index} ('{name}') is absent"
                )));
            }
        }
        for (index, input) in inputs.iter().enumerate().filter(|(_, v)| !v.is_absent()) {
            if input.dtype != DataType::Float32 {
                return Err(error(format!(
                    "input {index} requires Float32 in Phase 1, got {:?}",
                    input.dtype
                )));
            }
        }
        if outputs[0].dtype != DataType::Float32 {
            return Err(error(format!(
                "output requires Float32 in Phase 1, got {:?}",
                outputs[0].dtype
            )));
        }

        let x_shape = inputs[0].shape;
        if !matches!(x_shape.len(), 2 | 3) {
            return Err(error(format!(
                "input must be 2-D [rows, hidden] or 3-D [batch, sequence, hidden], got {x_shape:?}"
            )));
        }
        if outputs[0].shape != x_shape {
            return Err(error(format!(
                "output shape {:?} must equal input shape {x_shape:?}",
                outputs[0].shape
            )));
        }
        let hidden = *x_shape.last().unwrap();
        let rows = x_shape[..x_shape.len() - 1].iter().product::<usize>();
        require_shape("router_probs", inputs[1].shape, 2)?;
        if inputs[1].shape[0] != rows {
            return Err(error(format!(
                "router_probs rows {} must equal flattened input rows {rows}",
                inputs[1].shape[0]
            )));
        }
        let experts = inputs[1].shape[1];
        if self.k > experts {
            return Err(error(format!(
                "requires 0 < k <= num_experts, got k={} and num_experts={experts}",
                self.k
            )));
        }

        require_shape("fc1_experts_weights", inputs[2].shape, 3)?;
        require_shape("fc2_experts_weights", inputs[4].shape, 3)?;
        if inputs[2].shape[0] != experts || inputs[4].shape[0] != experts {
            return Err(error(format!(
                "expert weight counts must equal router num_experts {experts}"
            )));
        }
        if inputs[2].shape[2] != hidden {
            return Err(error(format!(
                "fc1_experts_weights must have canonical shape [experts, fc1_size, hidden={hidden}], got {:?}",
                inputs[2].shape
            )));
        }
        if inputs[4].shape[1] != hidden {
            return Err(error(format!(
                "fc2_experts_weights must have canonical shape [experts, hidden={hidden}, inter_size], got {:?}",
                inputs[4].shape
            )));
        }
        let inter = inputs[4].shape[2];
        let expected_fc1 = if self.activation == Activation::Swiglu && self.swiglu_fusion != 0 {
            2 * inter
        } else {
            inter
        };
        if inputs[2].shape[1] != expected_fc1 {
            return Err(error(format!(
                "fc1_experts_weights dimension 1 must be {expected_fc1}, got {}",
                inputs[2].shape[1]
            )));
        }

        let fc1_bias = optional_dense(inputs, 3)?;
        let fc2_bias = optional_dense(inputs, 5)?;
        validate_bias("fc1_experts_bias", inputs, 3, experts, expected_fc1)?;
        validate_bias("fc2_experts_bias", inputs, 5, experts, hidden)?;

        let has_fc3 = inputs.get(6).is_some_and(|v| !v.is_absent());
        let uses_separate_gate = (self.activation == Activation::Swiglu && self.swiglu_fusion == 0)
            || (self.activation == Activation::Silu && has_fc3);
        let (fc3, fc3_bias) = if uses_separate_gate {
            let view = inputs
                .get(6)
                .filter(|v| !v.is_absent())
                .ok_or_else(|| error("unfused swiglu requires input 6 fc3_experts_weights"))?;
            require_exact_shape("fc3_experts_weights", view.shape, &[experts, inter, hidden])?;
            validate_bias("fc3_experts_bias", inputs, 7, experts, inter)?;
            (Some(to_dense_f32(view)?), optional_dense(inputs, 7)?)
        } else {
            if has_fc3 {
                return Err(error(
                    "fc3_experts_weights is only valid for unfused swiglu or silu gated-GLU",
                ));
            }
            if inputs.get(7).is_some_and(|v| !v.is_absent()) {
                return Err(error(
                    "fc3_experts_bias requires fc3_experts_weights in unfused swiglu or silu gated-GLU",
                ));
            }
            (None, None)
        };

        let x = to_dense_f32(&inputs[0])?;
        let router = to_dense_f32(&inputs[1])?;
        let fc1 = to_dense_f32(&inputs[2])?;
        let fc2 = to_dense_f32(&inputs[4])?;
        let mut output = vec![0.0f32; rows * hidden];

        for row in 0..rows {
            let route = routing_weights(
                &router[row * experts..(row + 1) * experts],
                self.k,
                self.normalize_routing_weights,
            );
            let input_row = &x[row * hidden..(row + 1) * hidden];
            for (expert, route_weight) in route {
                let mut fc1_out = linear(
                    input_row,
                    &fc1[expert * expected_fc1 * hidden..(expert + 1) * expected_fc1 * hidden],
                    fc1_bias
                        .as_deref()
                        .map(|b| &b[expert * expected_fc1..(expert + 1) * expected_fc1]),
                    expected_fc1,
                    hidden,
                );
                let activated = match self.activation {
                    Activation::Swiglu => {
                        let linear_part;
                        let gate_part;
                        if self.swiglu_fusion == 0 {
                            gate_part = fc1_out;
                            let weights = fc3.as_ref().unwrap();
                            linear_part = linear(
                                input_row,
                                &weights[expert * inter * hidden..(expert + 1) * inter * hidden],
                                fc3_bias
                                    .as_deref()
                                    .map(|b| &b[expert * inter..(expert + 1) * inter]),
                                inter,
                                hidden,
                            );
                        } else if self.swiglu_fusion == 1 {
                            let mut gate = Vec::with_capacity(inter);
                            let mut linear = Vec::with_capacity(inter);
                            for pair in fc1_out.chunks_exact(2) {
                                gate.push(pair[0]);
                                linear.push(pair[1]);
                            }
                            gate_part = gate;
                            linear_part = linear;
                        } else {
                            linear_part = fc1_out.split_off(inter);
                            gate_part = fc1_out;
                        }
                        gate_part
                            .into_iter()
                            .zip(linear_part)
                            .map(|(g, l)| self.swiglu(g, l))
                            .collect()
                    }
                    Activation::Silu if fc3.is_some() => {
                        let weights = fc3.as_ref().unwrap();
                        let linear_part = linear(
                            input_row,
                            &weights[expert * inter * hidden..(expert + 1) * inter * hidden],
                            fc3_bias
                                .as_deref()
                                .map(|b| &b[expert * inter..(expert + 1) * inter]),
                            inter,
                            hidden,
                        );
                        fc1_out
                            .into_iter()
                            .zip(linear_part)
                            .map(|(g, l)| self.swiglu(g, l))
                            .collect()
                    }
                    activation => {
                        for value in &mut fc1_out {
                            *value = activate(activation, *value);
                        }
                        fc1_out
                    }
                };
                let expert_out = linear(
                    &activated,
                    &fc2[expert * hidden * inter..(expert + 1) * hidden * inter],
                    fc2_bias
                        .as_deref()
                        .map(|b| &b[expert * hidden..(expert + 1) * hidden]),
                    hidden,
                    inter,
                );
                for h in 0..hidden {
                    output[row * hidden + h] += route_weight * expert_out[h];
                }
            }
        }
        write_dense_f32(&mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

impl MoEKernel {
    fn swiglu(&self, gate: f32, linear: f32) -> f32 {
        let g = gate.min(self.swiglu_limit);
        let l = linear.clamp(-self.swiglu_limit, self.swiglu_limit);
        g * sigmoid(self.activation_alpha * g) * (l + self.activation_beta)
    }
}

fn linear(
    input: &[f32],
    weights: &[f32],
    bias: Option<&[f32]>,
    out_features: usize,
    in_features: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; out_features];
    for o in 0..out_features {
        let mut sum = bias.map_or(0.0, |b| b[o]);
        for i in 0..in_features {
            sum += input[i] * weights[o * in_features + i];
        }
        output[o] = sum;
    }
    output
}

fn routing_weights(logits: &[f32], k: usize, normalize: bool) -> Vec<(usize, f32)> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exponentials: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let all_sum: f32 = exponentials.iter().sum();
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    indices.truncate(k);
    let denominator = if normalize {
        indices.iter().map(|&i| exponentials[i]).sum()
    } else {
        all_sum
    };
    indices
        .into_iter()
        .map(|i| (i, exponentials[i] / denominator))
        .collect()
}

fn activate(activation: Activation, value: f32) -> f32 {
    match activation {
        Activation::Relu => value.max(0.0),
        Activation::Gelu => exact_gelu(value),
        Activation::Silu => value * sigmoid(value),
        Activation::Identity => value,
        Activation::Swiglu => unreachable!(),
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn required_int(node: &Node, name: &str) -> Result<i64> {
    node.attr(name)
        .and_then(|a| a.as_int())
        .ok_or_else(|| error(format!("required integer attribute {name} is missing")))
}

fn int_attr(node: &Node, name: &str, default: i64) -> Result<i64> {
    match node.attr(name) {
        Some(attr) => attr
            .as_int()
            .ok_or_else(|| error(format!("attribute {name} must be an integer"))),
        None => Ok(default),
    }
}

fn bool_attr(node: &Node, name: &str, default: bool) -> Result<bool> {
    match int_attr(node, name, i64::from(default))? {
        0 => Ok(false),
        1 => Ok(true),
        value => Err(error(format!(
            "attribute {name} must be 0 or 1, got {value}"
        ))),
    }
}

fn float_attr(node: &Node, name: &str, default: f32) -> Result<f32> {
    match node.attr(name) {
        Some(attr) => attr
            .as_float()
            .ok_or_else(|| error(format!("attribute {name} must be a float"))),
        None => Ok(default),
    }
}

fn optional_dense(inputs: &[TensorView], index: usize) -> Result<Option<Vec<f32>>> {
    inputs
        .get(index)
        .filter(|v| !v.is_absent())
        .map(to_dense_f32)
        .transpose()
}

fn validate_bias(
    name: &str,
    inputs: &[TensorView],
    index: usize,
    experts: usize,
    width: usize,
) -> Result<()> {
    if let Some(view) = inputs.get(index).filter(|v| !v.is_absent()) {
        require_exact_shape(name, view.shape, &[experts, width])?;
    }
    Ok(())
}

fn require_shape(name: &str, shape: &[usize], rank: usize) -> Result<()> {
    if shape.len() != rank {
        return Err(error(format!(
            "{name} must be {rank}-D, got shape {shape:?}"
        )));
    }
    Ok(())
}

fn require_exact_shape(name: &str, got: &[usize], expected: &[usize]) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have shape {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("MoE: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuExecutionProvider;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ep_api::ExecutionProvider;
    use onnx_runtime_ir::{Attribute, Graph, NodeId, static_shape};
    use onnx_runtime_loader::Model;

    fn model_node(
        input_shapes: &[Option<&[usize]>],
        output_shape: &[usize],
        attrs: &[(&str, Attribute)],
    ) -> (Graph, NodeId) {
        let mut graph = Graph::new();
        graph.opset_imports.insert("com.microsoft".into(), 1);
        let inputs = input_shapes
            .iter()
            .enumerate()
            .map(|(i, shape)| {
                shape.map(|shape| {
                    let value = graph.create_named_value(
                        format!("input_{i}"),
                        DataType::Float32,
                        static_shape(shape.iter().copied()),
                    );
                    graph.add_input(value);
                    value
                })
            })
            .collect();
        let output = graph.create_named_value(
            "output",
            DataType::Float32,
            static_shape(output_shape.iter().copied()),
        );
        let mut node = Node::new(NodeId(0), "MoE", inputs, vec![output]);
        node.domain = "com.microsoft".into();
        for (name, value) in attrs {
            node.attributes.insert((*name).into(), value.clone());
        }
        let node = graph.insert_node(node);
        graph.add_output(output);
        (graph, node)
    }

    fn kernel(graph: &Graph, node: NodeId) -> Box<dyn Kernel> {
        let model = Model::new(graph);
        CpuExecutionProvider::new()
            .get_kernel(model.graph.node(node), &[], 1)
            .unwrap()
    }

    fn assert_close(got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len());
        for (i, (&got, &want)) in got.iter().zip(want).enumerate() {
            assert!(
                (got - want).abs() <= 1e-5,
                "index {i}: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn moe_gelu_top1_biases_selects_different_experts_per_row() {
        let shapes = [
            Some(&[2, 2][..]),
            Some(&[2, 2]),
            Some(&[2, 2, 2]),
            Some(&[2, 2]),
            Some(&[2, 2, 2]),
            Some(&[2, 2]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[2, 2],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"gelu".to_vec())),
                ("normalize_routing_weights", Attribute::Int(0)),
            ],
        );
        let x = Owned::f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let router = Owned::f32(&[2, 2], &[4.0, 0.0, 0.0, 4.0]);
        let fc1 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let fc1_bias = Owned::f32(&[2, 2], &[0.5, -0.5, 1.0, 1.0]);
        let fc2 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]);
        let fc2_bias = Owned::f32(&[2, 2], &[0.25, -0.25, 0.5, 0.5]);
        let mut y = Owned::zeros_f32(&[2, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    fc1_bias.view(),
                    fc2.view(),
                    fc2_bias.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let p = 4.0f32.exp() / (4.0f32.exp() + 1.0);
        assert_close(
            &y.to_f32(),
            &[
                p * (exact_gelu(1.5) + 0.25),
                p * (exact_gelu(1.5) - 0.25),
                p * (exact_gelu(7.0) + 0.5),
                p * (exact_gelu(9.0) + 0.5),
            ],
        );
    }

    #[test]
    fn moe_silu_top2_normalized_without_biases_preserves_3d_shape() {
        let shapes = [
            Some(&[1, 2, 2][..]),
            Some(&[2, 2]),
            Some(&[2, 2, 2]),
            None,
            Some(&[2, 2, 2]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 2, 2],
            &[
                ("k", Attribute::Int(2)),
                ("activation_type", Attribute::String(b"silu".to_vec())),
                ("normalize_routing_weights", Attribute::Int(1)),
            ],
        );
        let x = Owned::f32(&[1, 2, 2], &[1.0, -1.0, 2.0, 1.0]);
        let router = Owned::f32(&[2, 2], &[0.0, 0.0, 2.0, 1.0]);
        let fc1 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0]);
        let fc2 = Owned::f32(&[2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5]);
        let mut y = Owned::zeros_f32(&[1, 2, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let p0 = 2.0f32.exp() / (2.0f32.exp() + 1.0f32.exp());
        let p1 = 1.0 - p0;
        assert_close(
            &y.to_f32(),
            &[
                0.5 * activate(Activation::Silu, 1.0) + 0.5 * 0.5 * activate(Activation::Silu, 2.0),
                0.5 * activate(Activation::Silu, -1.0)
                    + 0.5 * 0.5 * activate(Activation::Silu, -2.0),
                p0 * activate(Activation::Silu, 2.0) + p1 * 0.5 * activate(Activation::Silu, 4.0),
                p0 * activate(Activation::Silu, 1.0) + p1 * 0.5 * activate(Activation::Silu, 2.0),
            ],
        );
    }

    #[test]
    fn moe_swiglu_unfused_fc3_with_biases() {
        let shapes = [
            Some(&[1, 2][..]),
            Some(&[1, 2]),
            Some(&[2, 1, 2]),
            Some(&[2, 1]),
            Some(&[2, 2, 1]),
            None,
            Some(&[2, 1, 2]),
            Some(&[2, 1]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 2],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"swiglu".to_vec())),
                ("swiglu_fusion", Attribute::Int(0)),
                ("normalize_routing_weights", Attribute::Int(1)),
            ],
        );
        let x = Owned::f32(&[1, 2], &[2.0, 1.0]);
        let router = Owned::f32(&[1, 2], &[0.0, 3.0]);
        let fc1 = Owned::f32(&[2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
        let fc1_bias = Owned::f32(&[2, 1], &[0.0, 1.0]);
        let fc2 = Owned::f32(&[2, 2, 1], &[1.0, 2.0, 3.0, 4.0]);
        let fc3 = Owned::f32(&[2, 1, 2], &[0.0, 1.0, 1.0, 1.0]);
        let fc3_bias = Owned::f32(&[2, 1], &[0.0, 0.5]);
        let mut y = Owned::zeros_f32(&[1, 2]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    fc1_bias.view(),
                    fc2.view(),
                    TensorView::absent(DataType::Float32),
                    fc3.view(),
                    fc3_bias.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let expert = 2.0 * sigmoid(2.0) * 3.5;
        assert_close(&y.to_f32(), &[3.0 * expert, 4.0 * expert]);
    }

    #[test]
    fn moe_swiglu_fused_interleaved() {
        let shapes = [
            Some(&[1, 1][..]),
            Some(&[1, 1]),
            Some(&[1, 4, 1]),
            None,
            Some(&[1, 1, 2]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 1],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"swiglu".to_vec())),
                ("swiglu_fusion", Attribute::Int(1)),
            ],
        );
        let x = Owned::f32(&[1, 1], &[2.0]);
        let router = Owned::f32(&[1, 1], &[0.0]);
        let fc1 = Owned::f32(&[1, 4, 1], &[1.0, 3.0, 2.0, 4.0]);
        let fc2 = Owned::f32(&[1, 1, 2], &[1.0, 0.5]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        let expected = 2.0 * sigmoid(2.0) * 6.0 + 0.5 * (4.0 * sigmoid(4.0) * 8.0);
        assert_close(&y.to_f32(), &[expected]);
    }

    #[test]
    fn moe_silu_with_fc3_uses_ort_mixtral_gated_form() {
        let shapes = [
            Some(&[1, 1][..]),
            Some(&[1, 1]),
            Some(&[1, 1, 1]),
            None,
            Some(&[1, 1, 1]),
            None,
            Some(&[1, 1, 1]),
        ];
        let (graph, node) = model_node(
            &shapes,
            &[1, 1],
            &[
                ("k", Attribute::Int(1)),
                ("activation_type", Attribute::String(b"silu".to_vec())),
            ],
        );
        let x = Owned::f32(&[1, 1], &[2.0]);
        let router = Owned::f32(&[1, 1], &[0.0]);
        let fc1 = Owned::f32(&[1, 1, 1], &[3.0]);
        let fc2 = Owned::f32(&[1, 1, 1], &[0.5]);
        let fc3 = Owned::f32(&[1, 1, 1], &[4.0]);
        let mut y = Owned::zeros_f32(&[1, 1]);
        kernel(&graph, node)
            .execute(
                &[
                    x.view(),
                    router.view(),
                    fc1.view(),
                    TensorView::absent(DataType::Float32),
                    fc2.view(),
                    TensorView::absent(DataType::Float32),
                    fc3.view(),
                ],
                &mut [y.view_mut()],
            )
            .unwrap();
        assert_close(&y.to_f32(), &[0.5 * (6.0 * sigmoid(6.0) * 8.0)]);
    }
}
