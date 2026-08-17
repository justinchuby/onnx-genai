//! CPU reference kernel for `com.microsoft::PackedMultiHeadAttention` v1.
//!
//! This adapter implements the separate-Q/K/V, bias-free packed-sequence
//! subset. Q/K/V are `[total_tokens, hidden]`; cumulative sequence lengths
//! delimit independent non-causal attention problems, and output rows remain
//! in packed token order. `token_offset` describes positions in the padded
//! source tensor but does not reorder this operator's output.

use std::borrow::Cow;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::check_arity;
use super::sdpa::ScaleMode;
use super::varlen_attention::{
    PackedAttentionSpec, compute_packed_attention, offsets as parse_offsets,
};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

const OP: &str = "PackedMultiHeadAttention";
const DOMAIN: &str = "com.microsoft";

pub struct PackedMultiHeadAttentionFactory;

struct PackedMultiHeadAttentionKernel {
    num_heads: usize,
    scale: Option<f32>,
}

impl KernelFactory for PackedMultiHeadAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = node
            .attr("num_heads")
            .and_then(|attribute| attribute.as_int())
            .ok_or_else(|| error("missing required int attribute 'num_heads'"))?;
        let num_heads = usize::try_from(num_heads)
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| error(format!("num_heads must be positive, got {num_heads}")))?;
        let scale = node
            .attr("scale")
            .map(|attribute| {
                attribute
                    .as_float()
                    .ok_or_else(|| error("scale attribute must be a float"))
            })
            .transpose()?;
        if scale.is_some_and(|value| !value.is_finite()) {
            return Err(error("scale must be finite"));
        }
        // Same rule as the other four attention kernels: an explicit
        // non-positive scale means "use 1/sqrt(head_size)", not "multiply every
        // score by zero". This op became reachable at the same time as they
        // did, so it gets the same guard rather than a silently different one.
        let scale = scale.filter(|value| *value > 0.0);
        Ok(Box::new(PackedMultiHeadAttentionKernel {
            num_heads,
            scale,
        }))
    }
}

/// Deny unsupported optional forms and layouts before graph partitioning.
pub fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let fail = |detail: String| Some(Cow::Owned(format!("{OP}: {detail}")));
    if !(6..=7).contains(&node.inputs.len()) || node.outputs.len() != 1 {
        return fail(format!(
            "expected 6 or 7 inputs and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    if node.inputs.get(3).is_some_and(Option::is_some) {
        return fail("input 3 Q/K/V bias is unsupported".into());
    }
    if node.inputs.get(6).is_some_and(Option::is_some) {
        return fail("input 6 attention_bias is unsupported".into());
    }
    if [0, 1, 2, 4, 5]
        .into_iter()
        .any(|index| node.inputs.get(index).is_none_or(Option::is_none))
    {
        return fail("Q, K, V, token_offset, and cumulative_sequence_length are required".into());
    }
    if shapes.len() < 6 || input_dtypes.len() < 6 {
        return fail("missing input shape or dtype metadata".into());
    }

    let dtype = input_dtypes[0];
    if !matches!(
        dtype,
        DataType::Float32 | DataType::Float16 | DataType::BFloat16
    ) {
        return fail(format!(
            "Q/K/V must be Float32, Float16, or BFloat16, got {dtype:?}"
        ));
    }
    for index in 0..3 {
        if input_dtypes[index] != dtype {
            return fail("Q, K, and V must use the same floating dtype".into());
        }
        if shapes[index].len() != 2 {
            return fail(format!("input {index} must be rank 2 [tokens,hidden]"));
        }
    }
    if input_dtypes[4] != DataType::Int32 || shapes[4].len() != 2 {
        return fail("token_offset must be rank-2 Int32".into());
    }
    if input_dtypes[5] != DataType::Int32 || shapes[5].len() != 1 {
        return fail("cumulative_sequence_length must be rank-1 Int32".into());
    }

    let num_heads = match node
        .attr("num_heads")
        .and_then(|attribute| attribute.as_int())
    {
        Some(value) if value > 0 => value as usize,
        Some(value) => return fail(format!("num_heads must be positive, got {value}")),
        None => return fail("missing required int attribute 'num_heads'".into()),
    };
    for (index, label) in [(0, "query"), (1, "key"), (2, "value")] {
        let Some(hidden) = shapes[index][1].as_static() else {
            return fail(format!("{label} hidden dimension must be static"));
        };
        if hidden == 0 || !hidden.is_multiple_of(num_heads) {
            return fail(format!(
                "{label} hidden dimension {hidden} must be positive and divisible by num_heads {num_heads}"
            ));
        }
    }
    let query_hidden = shapes[0][1].as_static().unwrap();
    let key_hidden = shapes[1][1].as_static().unwrap();
    if query_hidden != key_hidden {
        return fail(format!(
            "query hidden dimension {query_hidden} must equal key hidden dimension {key_hidden}"
        ));
    }
    for index in [1, 2] {
        if let (Some(query_tokens), Some(tokens)) =
            (shapes[0][0].as_static(), shapes[index][0].as_static())
            && query_tokens != tokens
        {
            return fail(format!(
                "Q/K/V token dimensions must match, got {query_tokens} and {tokens}"
            ));
        }
    }
    if let (Some(batch), Some(cumulative_length)) =
        (shapes[4][0].as_static(), shapes[5][0].as_static())
        && cumulative_length != batch + 1
    {
        return fail(format!(
            "cumulative_sequence_length length {cumulative_length} must equal token_offset batch {batch} + 1"
        ));
    }
    if let Some(attribute) = node.attr("scale") {
        let Some(scale) = attribute.as_float() else {
            return fail("scale attribute must be a float".into());
        };
        if !scale.is_finite() {
            return fail("scale must be finite".into());
        }
    }
    None
}

impl Kernel for PackedMultiHeadAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 6, 7, 1)?;
        if !inputs[3].is_absent() {
            return Err(error("input 3 Q/K/V bias is unsupported"));
        }
        if inputs.len() == 7 && !inputs[6].is_absent() {
            return Err(error("input 6 attention_bias is unsupported"));
        }

        let dtype = inputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(error(format!(
                "Q/K/V dtype {dtype:?} is unsupported (expected Float32, Float16, or BFloat16)"
            )));
        }
        if inputs[1].dtype != dtype || inputs[2].dtype != dtype {
            return Err(error("Q, K, and V must use the same floating dtype"));
        }
        for (index, name) in [(0, "query"), (1, "key"), (2, "value")] {
            if inputs[index].shape.len() != 2 {
                return Err(error(format!(
                    "{name} must be rank 2 [tokens,hidden], got {:?}",
                    inputs[index].shape
                )));
            }
        }
        let token_count = inputs[0].shape[0];
        if inputs[1].shape[0] != token_count || inputs[2].shape[0] != token_count {
            return Err(error(format!(
                "Q/K/V token counts must match, got {}, {}, and {}",
                token_count, inputs[1].shape[0], inputs[2].shape[0]
            )));
        }
        let query_hidden = inputs[0].shape[1];
        let key_hidden = inputs[1].shape[1];
        let value_hidden = inputs[2].shape[1];
        if query_hidden != key_hidden {
            return Err(error(format!(
                "query hidden size {query_hidden} must equal key hidden size {key_hidden}"
            )));
        }
        if query_hidden == 0
            || value_hidden == 0
            || !query_hidden.is_multiple_of(self.num_heads)
            || !value_hidden.is_multiple_of(self.num_heads)
        {
            return Err(error(format!(
                "query/key hidden size {query_hidden} and value hidden size {value_hidden} must be positive and divisible by num_heads {}",
                self.num_heads
            )));
        }
        if inputs[4].dtype != DataType::Int32 || inputs[4].shape.len() != 2 {
            return Err(error("token_offset must be rank-2 Int32"));
        }
        if inputs[5].dtype != DataType::Int32 || inputs[5].shape.len() != 1 {
            return Err(error("cumulative_sequence_length must be rank-1 Int32"));
        }
        let batch = inputs[4].shape[0];
        if inputs[5].shape[0] != batch + 1 {
            return Err(error(format!(
                "cumulative_sequence_length length {} must equal token_offset batch {} + 1",
                inputs[5].shape[0], batch
            )));
        }
        let padded_tokens = batch
            .checked_mul(inputs[4].shape[1])
            .ok_or_else(|| error("token_offset shape element count overflow"))?;
        if padded_tokens < token_count {
            return Err(error(format!(
                "token_offset covers {padded_tokens} padded positions, fewer than {token_count} packed tokens"
            )));
        }
        if outputs[0].dtype != dtype || outputs[0].shape != [token_count, value_hidden] {
            return Err(error(format!(
                "output must use dtype {dtype:?} and shape [{token_count}, {value_hidden}], got {:?} {:?}",
                outputs[0].dtype, outputs[0].shape
            )));
        }

        let cumulative = parse_offsets(&inputs[5], "cumulative_sequence_length", token_count)?;
        let query = to_dense_f32_widen(OP, &inputs[0])?;
        let key = to_dense_f32_widen(OP, &inputs[1])?;
        let value = to_dense_f32_widen(OP, &inputs[2])?;
        let head_size = query_hidden / self.num_heads;
        let value_head_size = value_hidden / self.num_heads;
        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let score_elements = cumulative.windows(2).fold(0u64, |sum, span| {
                let length = (span[1] - span[0]) as u64;
                sum.saturating_add(
                    (self.num_heads as u64)
                        .saturating_mul(length)
                        .saturating_mul(length),
                )
            });
            score_elements
                .saturating_mul((head_size + value_head_size) as u64)
                .saturating_mul(2)
        });
        let output = compute_packed_attention(PackedAttentionSpec {
            query: &query,
            key: &key,
            value: &value,
            cu_seqlens_q: &cumulative,
            cu_seqlens_kv: &cumulative,
            nonpad_kv_seqlen: None,
            num_heads: self.num_heads,
            kv_num_heads: self.num_heads,
            head_size,
            value_head_size,
            scale: ScaleMode::PostDot(scale),
            is_causal: false,
            softcap: None,
            fast_path: true,
        })?;
        write_dense_f32_narrow(OP, &mut outputs[0], &output)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

fn error(detail: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{DOMAIN}::{OP}: {}", detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;
    use onnx_runtime_ir::{Attribute, NodeId, ValueId, static_shape};

    fn run(dtype: DataType) -> Vec<f32> {
        // Three independent sequences of lengths 2, 1, and 3. Zero Q/K makes
        // every attention row the per-sequence mean of V.
        let query_values = vec![0.0; 6 * 4];
        let key_values = query_values.clone();
        let value_values = [
            1.0, 10.0, 3.0, 30.0, // sequence 0, token 0
            5.0, 50.0, 7.0, 70.0, // sequence 0, token 1
            100.0, 200.0, 300.0, 400.0, // sequence 1
            2.0, 20.0, 4.0, 40.0, // sequence 2, token 0
            6.0, 60.0, 8.0, 80.0, // sequence 2, token 1
            10.0, 100.0, 12.0, 120.0, // sequence 2, token 2
        ];
        let make = |values: &[f32]| match dtype {
            DataType::Float32 => Owned::f32(&[6, 4], values),
            DataType::BFloat16 => Owned::bf16(&[6, 4], values),
            _ => unreachable!(),
        };
        let query = make(&query_values);
        let key = make(&key_values);
        let value = make(&value_values);
        let token_offset = Owned::i32(&[3, 3], &[0, 1, 3, 2, 4, 5, 6, 7, 8]);
        let cumulative = Owned::i32(&[4], &[0, 2, 3, 6]);
        let mut output = Owned::zeros(dtype, &[6, 4]);
        PackedMultiHeadAttentionKernel {
            num_heads: 2,
            scale: Some(0.25),
        }
        .execute(
            &[
                query.view(),
                key.view(),
                value.view(),
                TensorView::absent(dtype),
                token_offset.view(),
                cumulative.view(),
            ],
            &mut [output.view_mut()],
        )
        .unwrap();
        match dtype {
            DataType::Float32 => output.to_f32(),
            DataType::BFloat16 => output.to_bf16_as_f32(),
            _ => unreachable!(),
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "output {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn variable_length_sequences_stay_independent_and_packed() {
        let expected = [
            3.0, 30.0, 5.0, 50.0, 3.0, 30.0, 5.0, 50.0, // length 2
            100.0, 200.0, 300.0, 400.0, // length 1
            6.0, 60.0, 8.0, 80.0, 6.0, 60.0, 8.0, 80.0, 6.0, 60.0, 8.0, 80.0,
        ];
        assert_close(&run(DataType::Float32), &expected, 1e-5);
    }

    #[test]
    fn bf16_tracks_f32_reference_after_output_rounding() {
        assert_close(&run(DataType::BFloat16), &run(DataType::Float32), 0.25);
    }

    #[test]
    fn rejects_present_optional_biases() {
        let query = Owned::f32(&[1, 2], &[0.0, 0.0]);
        let token_offset = Owned::i32(&[1, 1], &[0]);
        let cumulative = Owned::i32(&[2], &[0, 1]);
        let present_bias = Owned::f32(&[6], &[0.0; 6]);
        let mut output = Owned::zeros_f32(&[1, 2]);
        let kernel = PackedMultiHeadAttentionKernel {
            num_heads: 1,
            scale: None,
        };
        let error = kernel
            .execute(
                &[
                    query.view(),
                    query.view(),
                    query.view(),
                    present_bias.view(),
                    token_offset.view(),
                    cumulative.view(),
                ],
                &mut [output.view_mut()],
            )
            .unwrap_err();
        assert!(error.to_string().contains("input 3"));

        let attention_bias = Owned::f32(&[1, 1, 1, 1], &[0.0]);
        let error = kernel
            .execute(
                &[
                    query.view(),
                    query.view(),
                    query.view(),
                    TensorView::absent(DataType::Float32),
                    token_offset.view(),
                    cumulative.view(),
                    attention_bias.view(),
                ],
                &mut [output.view_mut()],
            )
            .unwrap_err();
        assert!(error.to_string().contains("input 6"));
    }

    #[test]
    fn claim_accepts_only_bias_free_separate_qkv_contract() {
        let mut node = Node::new(
            NodeId(0),
            OP,
            vec![
                Some(ValueId(0)),
                Some(ValueId(1)),
                Some(ValueId(2)),
                None,
                Some(ValueId(4)),
                Some(ValueId(5)),
                None,
            ],
            vec![ValueId(6)],
        );
        node.domain = DOMAIN.into();
        node.attributes
            .insert("num_heads".into(), Attribute::Int(2));
        let shapes = vec![
            static_shape([6, 4]),
            static_shape([6, 4]),
            static_shape([6, 4]),
            static_shape([]),
            static_shape([3, 3]),
            static_shape([4]),
            static_shape([]),
        ];
        let dtypes = vec![
            DataType::BFloat16,
            DataType::BFloat16,
            DataType::BFloat16,
            DataType::BFloat16,
            DataType::Int32,
            DataType::Int32,
            DataType::BFloat16,
        ];
        assert!(unsupported_reason(&node, &shapes, &dtypes).is_none());

        node.inputs[6] = Some(ValueId(7));
        let reason = unsupported_reason(&node, &shapes, &dtypes).unwrap();
        assert!(reason.contains("attention_bias"));
        node.inputs[6] = None;
        node.inputs[3] = Some(ValueId(3));
        let reason = unsupported_reason(&node, &shapes, &dtypes).unwrap();
        assert!(reason.contains("Q/K/V bias"));
    }
}
