//! CPU reference kernel for `pkg.nxrt::PackedVarlenAttention` v1.

use std::borrow::Cow;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::check_arity;
use super::sdpa::ScaleMode;
use super::varlen_attention::{
    PackedAttentionSpec, compute_packed_attention, offsets as parse_offsets,
};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

const OP: &str = "PackedVarlenAttention";
const DOMAIN: &str = onnx_runtime_ir::RUNTIME_DOMAIN;

pub struct PackedVarlenAttentionFactory;

struct PackedVarlenAttentionKernel {
    num_heads: usize,
    kv_num_heads: Option<usize>,
    scale: Option<f32>,
    is_causal: bool,
    softcap: Option<f32>,
}

impl KernelFactory for PackedVarlenAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = positive_attribute(node, "num_heads")?;
        let kv_num_heads = node
            .attr("kv_num_heads")
            .and_then(|attribute| attribute.as_int())
            .map(|value| {
                usize::try_from(value)
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| error(format!("kv_num_heads must be positive, got {value}")))
            })
            .transpose()?;
        let softcap = node
            .attr("softcap")
            .and_then(|attribute| attribute.as_float())
            .unwrap_or(0.0);
        Ok(Box::new(PackedVarlenAttentionKernel {
            num_heads,
            kv_num_heads,
            scale: node
                .attr("scale")
                .and_then(|attribute| attribute.as_float()),
            is_causal: node
                .attr("is_causal")
                .and_then(|attribute| attribute.as_int())
                .unwrap_or(0)
                != 0,
            softcap: (softcap != 0.0).then_some(softcap),
        }))
    }
}

fn positive_attribute(node: &Node, name: &str) -> Result<usize> {
    let value = node
        .attr(name)
        .and_then(|attribute| attribute.as_int())
        .ok_or_else(|| error(format!("missing required int attribute '{name}'")))?;
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| error(format!("{name} must be positive, got {value}")))
}

pub fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let fail = |detail: String| Some(Cow::Owned(format!("{OP}: {detail}")));
    if node.inputs.len() != 5 || node.outputs.len() != 1 {
        return fail(format!(
            "expected 5 inputs and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    if shapes.len() < 5 || input_dtypes.len() < 5 {
        return fail("missing input shape or dtype metadata".into());
    }
    if node
        .attr("num_heads")
        .and_then(|attribute| attribute.as_int())
        .is_none()
    {
        return fail("missing required int attribute 'num_heads'".into());
    }
    for index in 0..3 {
        if !matches!(
            input_dtypes[index],
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return fail(format!(
                "input {index} must be Float32, Float16, or BFloat16, got {:?}",
                input_dtypes[index]
            ));
        }
        if input_dtypes[index] != input_dtypes[0] {
            return fail("Q, K, and V must use the same floating dtype".into());
        }
        if !matches!(shapes[index].len(), 2 | 3) {
            return fail(format!("input {index} must be rank 2 or 3"));
        }
    }
    for index in 3..5 {
        if input_dtypes[index] != DataType::Int32 {
            return fail(format!(
                "cu_seqlens input {index} must be Int32, got {:?}",
                input_dtypes[index]
            ));
        }
        if shapes[index].len() != 1 {
            return fail(format!("cu_seqlens input {index} must be rank 1"));
        }
    }
    None
}

struct PackedDimensions {
    tokens: usize,
    dimension: usize,
    rank: usize,
}

fn resolve_packed(input: &TensorView, name: &str, heads: usize) -> Result<PackedDimensions> {
    match input.shape {
        [tokens, input_heads, dimension] => {
            if *input_heads != heads {
                return Err(error(format!(
                    "{name} rank-3 head dimension {input_heads} must equal head count {heads}"
                )));
            }
            Ok(PackedDimensions {
                tokens: *tokens,
                dimension: *dimension,
                rank: 3,
            })
        }
        [tokens, hidden] => {
            if !hidden.is_multiple_of(heads) {
                return Err(error(format!(
                    "{name} rank-2 hidden size {hidden} is not divisible by head count {heads}"
                )));
            }
            Ok(PackedDimensions {
                tokens: *tokens,
                dimension: hidden / heads,
                rank: 2,
            })
        }
        _ => Err(error(format!(
            "{name} must be rank 2 or 3, got shape {:?}",
            input.shape
        ))),
    }
}

impl Kernel for PackedVarlenAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 5, 5, 1)?;
        let dtype = inputs[0].dtype;
        if !matches!(
            dtype,
            DataType::Float32 | DataType::Float16 | DataType::BFloat16
        ) {
            return Err(error(format!(
                "Q/K/V dtype {dtype:?} not supported (expected f32, f16, or bf16)"
            )));
        }
        if inputs[1].dtype != dtype || inputs[2].dtype != dtype {
            return Err(error("Q, K, and V must use the same floating dtype"));
        }
        if inputs[3].dtype != DataType::Int32 || inputs[4].dtype != DataType::Int32 {
            return Err(error("cu_seqlens_q and cu_seqlens_kv must be int32"));
        }

        let kv_num_heads = self.kv_num_heads.unwrap_or(self.num_heads);
        if !self.num_heads.is_multiple_of(kv_num_heads) {
            return Err(error(format!(
                "num_heads {} must be a positive multiple of kv_num_heads {kv_num_heads}",
                self.num_heads
            )));
        }
        let query_dimensions = resolve_packed(&inputs[0], "query", self.num_heads)?;
        let key_dimensions = resolve_packed(&inputs[1], "key", kv_num_heads)?;
        let value_dimensions = resolve_packed(&inputs[2], "value", kv_num_heads)?;
        if query_dimensions.dimension != key_dimensions.dimension {
            return Err(error(format!(
                "query head_size {} != key head_size {}",
                query_dimensions.dimension, key_dimensions.dimension
            )));
        }
        if key_dimensions.tokens != value_dimensions.tokens {
            return Err(error(format!(
                "key token count {} != value token count {}",
                key_dimensions.tokens, value_dimensions.tokens
            )));
        }
        let cu_seqlens_q = parse_offsets(&inputs[3], "cu_seqlens_q", query_dimensions.tokens)?;
        let cu_seqlens_kv = parse_offsets(&inputs[4], "cu_seqlens_kv", key_dimensions.tokens)?;
        if cu_seqlens_q.len() != cu_seqlens_kv.len() {
            return Err(error(format!(
                "cu_seqlens_q length {} != cu_seqlens_kv length {}",
                cu_seqlens_q.len(),
                cu_seqlens_kv.len()
            )));
        }

        let expected_shape = if query_dimensions.rank == 3 {
            vec![
                query_dimensions.tokens,
                self.num_heads,
                value_dimensions.dimension,
            ]
        } else {
            vec![
                query_dimensions.tokens,
                self.num_heads * value_dimensions.dimension,
            ]
        };
        if outputs[0].dtype != dtype || outputs[0].shape != expected_shape {
            return Err(error(format!(
                "output must use dtype {dtype:?} and shape {expected_shape:?}, got {:?} {:?}",
                outputs[0].dtype, outputs[0].shape
            )));
        }

        let query = to_dense_f32_widen(OP, &inputs[0])?;
        let key = to_dense_f32_widen(OP, &inputs[1])?;
        let value = to_dense_f32_widen(OP, &inputs[2])?;
        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (query_dimensions.dimension as f32).sqrt());
        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let pairs = cu_seqlens_q.windows(2).zip(cu_seqlens_kv.windows(2)).fold(
                0u64,
                |sum, (query, key_value)| {
                    sum.saturating_add(
                        ((query[1] - query[0]) as u64)
                            .saturating_mul((key_value[1] - key_value[0]) as u64),
                    )
                },
            );
            pairs
                .saturating_mul(self.num_heads as u64)
                .saturating_mul((query_dimensions.dimension + value_dimensions.dimension) as u64)
                .saturating_mul(2)
        });
        let output = compute_packed_attention(PackedAttentionSpec {
            query: &query,
            key: &key,
            value: &value,
            cu_seqlens_q: &cu_seqlens_q,
            cu_seqlens_kv: &cu_seqlens_kv,
            nonpad_kv_seqlen: None,
            num_heads: self.num_heads,
            kv_num_heads,
            head_size: query_dimensions.dimension,
            value_head_size: value_dimensions.dimension,
            scale: ScaleMode::SplitSqrt(scale),
            is_causal: self.is_causal,
            softcap: self.softcap,
            fast_path: false,
        })?;
        write_dense_f32_narrow(OP, &mut outputs[0], &output)
    }
}

fn error(detail: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{DOMAIN}::{OP}: {}", detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::kernels::testutil::Owned;
    use crate::kernels::varlen_attention::VarlenAttentionKernel;

    fn data(elements: usize, phase: f32) -> Vec<f32> {
        (0..elements)
            .map(|index| ((index as f32 + phase) * 0.173).sin())
            .collect()
    }

    fn cumulative(lengths: &[usize]) -> Vec<i32> {
        let mut total = 0i32;
        std::iter::once(0)
            .chain(lengths.iter().map(|&length| {
                total += length as i32;
                total
            }))
            .collect()
    }

    fn run_packed(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        lengths: &[usize],
        num_heads: usize,
        kv_num_heads: usize,
        dimension: usize,
        is_causal: bool,
    ) -> Vec<f32> {
        let total: usize = lengths.iter().sum();
        let offsets = cumulative(lengths);
        let query = Owned::f32(&[total, num_heads, dimension], query);
        let key = Owned::f32(&[total, kv_num_heads, dimension], key);
        let value = Owned::f32(&[total, kv_num_heads, dimension], value);
        let offsets = Owned::i32(&[offsets.len()], &offsets);
        let mut output = Owned::zeros_f32(&[total, num_heads, dimension]);
        PackedVarlenAttentionKernel {
            num_heads,
            kv_num_heads: Some(kv_num_heads),
            scale: Some(0.5),
            is_causal,
            softcap: None,
        }
        .execute(
            &[
                query.view(),
                key.view(),
                value.view(),
                offsets.view(),
                offsets.view(),
            ],
            &mut [output.view_mut()],
        )
        .unwrap();
        output.to_f32()
    }

    fn run_padded_varlen(
        query: &[f32],
        key: &[f32],
        value: &[f32],
        lengths: &[usize],
        num_heads: usize,
        kv_num_heads: usize,
        dimension: usize,
    ) -> Vec<f32> {
        let batch = lengths.len();
        let maximum_length = lengths.iter().copied().max().unwrap();
        let allocated_rows = batch * maximum_length;
        let mut padded_query = vec![0.0; allocated_rows * num_heads * dimension];
        let mut padded_key = vec![0.0; allocated_rows * kv_num_heads * dimension];
        let mut padded_value = padded_key.clone();
        let mut packed_row = 0;
        for (sequence, &length) in lengths.iter().enumerate() {
            for token in 0..length {
                let padded_row = sequence * maximum_length + token;
                let query_width = num_heads * dimension;
                let kv_width = kv_num_heads * dimension;
                padded_query[padded_row * query_width..(padded_row + 1) * query_width]
                    .copy_from_slice(
                        &query[packed_row * query_width..(packed_row + 1) * query_width],
                    );
                padded_key[padded_row * kv_width..(padded_row + 1) * kv_width]
                    .copy_from_slice(&key[packed_row * kv_width..(packed_row + 1) * kv_width]);
                padded_value[padded_row * kv_width..(padded_row + 1) * kv_width]
                    .copy_from_slice(&value[packed_row * kv_width..(packed_row + 1) * kv_width]);
                packed_row += 1;
            }
        }
        let offsets = (0..=batch)
            .map(|index| (index * maximum_length) as i64)
            .collect::<Vec<_>>();
        let nonpad = lengths
            .iter()
            .map(|&length| length as i64)
            .collect::<Vec<_>>();
        let query = Owned::f32(&[allocated_rows, num_heads, dimension], &padded_query);
        let key = Owned::f32(&[allocated_rows, kv_num_heads, dimension], &padded_key);
        let value = Owned::f32(&[allocated_rows, kv_num_heads, dimension], &padded_value);
        let offsets = Owned::i64(&[offsets.len()], &offsets);
        let nonpad = Owned::i64(&[nonpad.len()], &nonpad);
        let mut output = Owned::zeros_f32(&[allocated_rows, num_heads, dimension]);
        VarlenAttentionKernel {
            scale: Some(0.5),
            is_causal: false,
            softcap: None,
        }
        .execute(
            &[
                query.view(),
                key.view(),
                value.view(),
                offsets.view(),
                offsets.view(),
                nonpad.view(),
            ],
            &mut [output.view_mut()],
        )
        .unwrap();
        let padded_output = output.to_f32();
        let mut packed_output =
            Vec::with_capacity(lengths.iter().sum::<usize>() * num_heads * dimension);
        for (sequence, &length) in lengths.iter().enumerate() {
            let width = num_heads * dimension;
            let start = sequence * maximum_length * width;
            packed_output.extend_from_slice(&padded_output[start..start + length * width]);
        }
        packed_output
    }

    fn assert_close(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 1e-6,
                "output {index}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn packed_varlen_matches_padded_nonpad_and_saves_rows() {
        for (lengths, num_heads, kv_num_heads) in
            [(vec![3, 1, 4], 4, 2), (vec![5], 2, 2), (vec![1, 2], 2, 1)]
        {
            let dimension = 2;
            let packed_rows: usize = lengths.iter().sum();
            let padded_rows = lengths.len() * lengths.iter().copied().max().unwrap();
            let query = data(packed_rows * num_heads * dimension, 1.0);
            let key = data(packed_rows * kv_num_heads * dimension, 7.0);
            let value = data(packed_rows * kv_num_heads * dimension, 13.0);
            let packed = run_packed(
                &query,
                &key,
                &value,
                &lengths,
                num_heads,
                kv_num_heads,
                dimension,
                false,
            );
            let padded = run_padded_varlen(
                &query,
                &key,
                &value,
                &lengths,
                num_heads,
                kv_num_heads,
                dimension,
            );
            assert_close(&packed, &padded);
            assert_eq!(packed.len(), packed_rows * num_heads * dimension);
            assert_eq!(
                cumulative(&lengths).last().copied().unwrap() as usize,
                packed_rows
            );
            assert!(packed_rows <= padded_rows);
            if lengths.len() > 1 && lengths.iter().any(|length| *length != lengths[0]) {
                assert!(packed_rows < padded_rows);
            }
        }
    }

    #[test]
    fn packed_varlen_causal_mask_stays_within_segment_boundaries() {
        let lengths = [3, 1, 4];
        let total: usize = lengths.iter().sum();
        let query = vec![0.0; total];
        let key = vec![0.0; total];
        let value = [1.0, 3.0, 8.0, 100.0, 10.0, 20.0, 40.0, 80.0];
        let output = run_packed(&query, &key, &value, &lengths, 1, 1, 1, true);
        assert_close(
            &output,
            &[1.0, 2.0, 4.0, 100.0, 10.0, 15.0, 70.0 / 3.0, 37.5],
        );
    }

    #[test]
    fn packed_varlen_rank_two_uses_tail_aligned_causal_mask() {
        let query = Owned::f32(&[1, 1], &[0.0]);
        let key = Owned::f32(&[3, 1], &[0.0, 0.0, 0.0]);
        let value = Owned::f32(&[3, 1], &[1.0, 3.0, 5.0]);
        let cu_query = Owned::i32(&[2], &[0, 1]);
        let cu_key_value = Owned::i32(&[2], &[0, 3]);
        let mut output = Owned::zeros_f32(&[1, 1]);
        PackedVarlenAttentionKernel {
            num_heads: 1,
            kv_num_heads: None,
            scale: None,
            is_causal: true,
            softcap: None,
        }
        .execute(
            &[
                query.view(),
                key.view(),
                value.view(),
                cu_query.view(),
                cu_key_value.view(),
            ],
            &mut [output.view_mut()],
        )
        .unwrap();
        assert_close(&output.to_f32(), &[3.0]);
    }
}
