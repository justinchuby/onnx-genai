//! CPU reference kernel for `pkg.nxrt::VarlenAttention` v1.
//!
//! Q/K/V use token-major packed layouts (`[total_tokens, heads, dim]`).
//! `cu_seqlens_q` and `cu_seqlens_kv` delimit each sequence, so the shared SDPA
//! core runs only over real tokens instead of a padded batch rectangle.

use std::borrow::Cow;

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node, Shape};

use super::sdpa::{
    AttnBias, NoBias, NoMask, ScaleMode, SdpaConfig, SdpaTensors, sdpa_f32, sdpa_f32_scalar,
};
use super::{check_arity, to_dense_i64};
use crate::dtype::{to_dense_f32_widen, write_dense_f32_narrow};

const OP: &str = "VarlenAttention";

pub struct VarlenAttentionFactory;

pub struct VarlenAttentionKernel {
    pub(super) scale: Option<f32>,
    pub(super) is_causal: bool,
    pub(super) softcap: Option<f32>,
}

pub(super) struct PackedAttentionSpec<'a> {
    pub query: &'a [f32],
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub cu_seqlens_q: &'a [usize],
    pub cu_seqlens_kv: &'a [usize],
    pub nonpad_kv_seqlen: Option<&'a [i64]>,
    pub num_heads: usize,
    pub kv_num_heads: usize,
    pub head_size: usize,
    pub value_head_size: usize,
    pub scale: ScaleMode,
    pub is_causal: bool,
    pub softcap: Option<f32>,
    pub fast_path: bool,
}

impl KernelFactory for VarlenAttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let softcap = node
            .attr("softcap")
            .and_then(|attribute| attribute.as_float())
            .unwrap_or(0.0);
        if softcap < 0.0 {
            return Err(error("softcap must be non-negative"));
        }
        Ok(Box::new(VarlenAttentionKernel {
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

/// Claim-time validation keeps unsupported dtypes/layouts from being assigned
/// to the f32 CPU oracle.
pub fn unsupported_reason(
    node: &Node,
    shapes: &[Shape],
    input_dtypes: &[DataType],
) -> Option<Cow<'static, str>> {
    let fail = |detail: String| Some(Cow::Owned(format!("{OP}: {detail}")));
    if !(5..=6).contains(&node.inputs.len()) || node.outputs.len() != 1 {
        return fail(format!(
            "expected 5 or 6 inputs and 1 output, got {} inputs and {} outputs",
            node.inputs.len(),
            node.outputs.len()
        ));
    }
    if shapes.len() < 5 || input_dtypes.len() < 5 {
        return fail("missing input shape or dtype metadata".into());
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
        if shapes[index].len() != 3 {
            return fail(format!(
                "input {index} must be rank 3 [tokens,heads,dim], got rank {}",
                shapes[index].len()
            ));
        }
    }
    for index in 3..node.inputs.len() {
        if node.inputs[index].is_none() {
            continue;
        }
        if !matches!(input_dtypes[index], DataType::Int32 | DataType::Int64) {
            return fail(format!(
                "input {index} must be Int32 or Int64, got {:?}",
                input_dtypes[index]
            ));
        }
        if shapes[index].len() != 1 {
            return fail(format!("input {index} must be rank 1"));
        }
    }
    None
}

struct CausalOffset {
    offset: i64,
}

impl AttnBias for CausalOffset {
    #[inline]
    fn at(&self, _batch: usize, _head: usize, query: usize, key: usize) -> f32 {
        if key as i64 > query as i64 + self.offset {
            f32::NEG_INFINITY
        } else {
            0.0
        }
    }
}

impl Kernel for VarlenAttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 5, 6, 1)?;
        for (index, input) in inputs.iter().take(3).enumerate() {
            if !matches!(
                input.dtype,
                DataType::Float32 | DataType::Float16 | DataType::BFloat16
            ) {
                return Err(error(format!(
                    "input {index} must be Float32, Float16, or BFloat16, got {:?}",
                    input.dtype
                )));
            }
            if input.dtype != inputs[0].dtype {
                return Err(error("Q, K, and V must use the same floating dtype"));
            }
            if input.shape.len() != 3 {
                return Err(error(format!(
                    "input {index} must be rank 3 [tokens,heads,dim], got {:?}",
                    input.shape
                )));
            }
        }

        let (total_q, q_heads, head_size) =
            (inputs[0].shape[0], inputs[0].shape[1], inputs[0].shape[2]);
        let (total_kv, kv_heads, key_dim) =
            (inputs[1].shape[0], inputs[1].shape[1], inputs[1].shape[2]);
        let (value_tokens, value_heads, value_dim) =
            (inputs[2].shape[0], inputs[2].shape[1], inputs[2].shape[2]);
        if total_kv != value_tokens || kv_heads != value_heads {
            return Err(error(format!(
                "key/value token and head dimensions must match, got K {:?}, V {:?}",
                inputs[1].shape, inputs[2].shape
            )));
        }
        if head_size != key_dim {
            return Err(error(format!(
                "query/key head dimensions must match, got {head_size} and {key_dim}"
            )));
        }
        if q_heads == 0 || kv_heads == 0 || q_heads % kv_heads != 0 {
            return Err(error(format!(
                "query heads {q_heads} must be a positive multiple of KV heads {kv_heads}"
            )));
        }
        let expected_output = [total_q, q_heads, value_dim];
        if outputs[0].shape != expected_output {
            return Err(error(format!(
                "output shape {:?} must be {:?}",
                outputs[0].shape, expected_output
            )));
        }

        let cu_q = offsets(&inputs[3], "cu_seqlens_q", total_q)?;
        let cu_kv = offsets(&inputs[4], "cu_seqlens_kv", total_kv)?;
        if cu_q.len() != cu_kv.len() {
            return Err(error(format!(
                "cu_seqlens_q and cu_seqlens_kv lengths differ: {} vs {}",
                cu_q.len(),
                cu_kv.len()
            )));
        }
        let batch = cu_q.len() - 1;
        let nonpad = if inputs.len() > 5 && !inputs[5].is_absent() {
            let values = to_dense_i64(&inputs[5])?;
            if values.len() != batch {
                return Err(error(format!(
                    "nonpad_kv_seqlen length {} must equal batch {batch}",
                    values.len()
                )));
            }
            Some(values)
        } else {
            None
        };

        let q = to_dense_f32_widen(OP, &inputs[0])?;
        let k = to_dense_f32_widen(OP, &inputs[1])?;
        let v = to_dense_f32_widen(OP, &inputs[2])?;
        let scale = self
            .scale
            .unwrap_or_else(|| 1.0 / (head_size as f32).sqrt());

        crate::trace::record_kernel_metrics(inputs, outputs, || {
            let score_elements = (0..batch).fold(0u64, |sum, sequence| {
                let q_len = cu_q[sequence + 1] - cu_q[sequence];
                let allocated_kv = cu_kv[sequence + 1] - cu_kv[sequence];
                let kv_len = nonpad
                    .as_ref()
                    .map_or(allocated_kv, |lengths| lengths[sequence].max(0) as usize);
                sum.saturating_add(
                    (q_heads as u64)
                        .saturating_mul(q_len as u64)
                        .saturating_mul(kv_len.min(allocated_kv) as u64),
                )
            });
            score_elements
                .saturating_mul((head_size + value_dim) as u64)
                .saturating_mul(2)
                .saturating_add(score_elements.saturating_mul(4))
        });

        let output = compute_packed_attention(PackedAttentionSpec {
            query: &q,
            key: &k,
            value: &v,
            cu_seqlens_q: &cu_q,
            cu_seqlens_kv: &cu_kv,
            nonpad_kv_seqlen: nonpad.as_deref(),
            num_heads: q_heads,
            kv_num_heads: kv_heads,
            head_size,
            value_head_size: value_dim,
            scale: ScaleMode::SplitSqrt(scale),
            is_causal: self.is_causal,
            softcap: self.softcap,
            fast_path: false,
        })?;
        write_dense_f32_narrow(OP, &mut outputs[0], &output)
    }
}

pub(super) fn compute_packed_attention(spec: PackedAttentionSpec<'_>) -> Result<Vec<f32>> {
    let total_q = spec.cu_seqlens_q.last().copied().unwrap_or(0);
    let batch = spec.cu_seqlens_q.len() - 1;
    let mut output = vec![0.0; total_q * spec.num_heads * spec.value_head_size];
    for sequence in 0..batch {
        let (q_start, q_end) = (spec.cu_seqlens_q[sequence], spec.cu_seqlens_q[sequence + 1]);
        let (kv_start, kv_end) = (
            spec.cu_seqlens_kv[sequence],
            spec.cu_seqlens_kv[sequence + 1],
        );
        let query_length = q_end - q_start;
        let allocated_kv_length = kv_end - kv_start;
        let kv_length = match spec.nonpad_kv_seqlen {
            Some(lengths) => usize::try_from(lengths[sequence]).map_err(|_| {
                error(format!(
                    "nonpad_kv_seqlen[{sequence}] must be non-negative, got {}",
                    lengths[sequence]
                ))
            })?,
            None => allocated_kv_length,
        };
        if kv_length > allocated_kv_length {
            return Err(error(format!(
                "nonpad_kv_seqlen[{sequence}]={kv_length} exceeds packed KV span {allocated_kv_length}"
            )));
        }

        let query_bhsd = token_major_to_bhsd(
            spec.query,
            q_start,
            query_length,
            spec.num_heads,
            spec.head_size,
        );
        let key_bhsd = token_major_to_bhsd(
            spec.key,
            kv_start,
            kv_length,
            spec.kv_num_heads,
            spec.head_size,
        );
        let value_bhsd = token_major_to_bhsd(
            spec.value,
            kv_start,
            kv_length,
            spec.kv_num_heads,
            spec.value_head_size,
        );
        let tensors = SdpaTensors {
            q: &query_bhsd,
            k: &key_bhsd,
            v: &value_bhsd,
            batch: 1,
            num_heads: spec.num_heads,
            num_kv_heads: spec.kv_num_heads,
            q_seq: query_length,
            kv_seq: kv_length,
            head_size: spec.head_size,
            v_head_size: spec.value_head_size,
        };
        let config = SdpaConfig {
            scale: spec.scale,
            softcap: spec.softcap,
            causal: false,
            past_seq: 0,
            causal_fill: f32::NEG_INFINITY,
        };
        let causal = CausalOffset {
            offset: kv_length as i64 - query_length as i64,
        };
        let bias: &dyn AttnBias = if spec.is_causal { &causal } else { &NoBias };
        let mut output_bhsd = vec![0.0; spec.num_heads * query_length * spec.value_head_size];
        if spec.fast_path {
            sdpa_f32(&tensors, &config, bias, &NoMask, &mut output_bhsd, None);
        } else {
            sdpa_f32_scalar(&tensors, &config, bias, &NoMask, &mut output_bhsd, None);
        }
        bhsd_to_token_major(
            &output_bhsd,
            &mut output,
            q_start,
            query_length,
            spec.num_heads,
            spec.value_head_size,
        );
    }
    Ok(output)
}

pub(super) fn offsets(view: &TensorView, name: &str, total: usize) -> Result<Vec<usize>> {
    if view.shape.len() != 1 {
        return Err(error(format!(
            "{name} must be rank 1, got {:?}",
            view.shape
        )));
    }
    let raw = to_dense_i64(view)?;
    if raw.len() < 2 {
        return Err(error(format!("{name} must contain at least [0, total]")));
    }
    let mut parsed = Vec::with_capacity(raw.len());
    for (index, value) in raw.into_iter().enumerate() {
        let value = usize::try_from(value)
            .map_err(|_| error(format!("{name}[{index}] must be non-negative")))?;
        if index == 0 && value != 0 {
            return Err(error(format!("{name}[0] must be 0, got {value}")));
        }
        if parsed.last().is_some_and(|previous| value < *previous) {
            return Err(error(format!("{name} must be non-decreasing")));
        }
        parsed.push(value);
    }
    if parsed.last().copied() != Some(total) {
        return Err(error(format!(
            "{name} last offset must equal packed token count {total}, got {:?}",
            parsed.last()
        )));
    }
    Ok(parsed)
}

fn token_major_to_bhsd(
    source: &[f32],
    token_start: usize,
    sequence: usize,
    heads: usize,
    dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; heads * sequence * dim];
    for token in 0..sequence {
        for head in 0..heads {
            let source_start = ((token_start + token) * heads + head) * dim;
            let output_start = (head * sequence + token) * dim;
            output[output_start..output_start + dim]
                .copy_from_slice(&source[source_start..source_start + dim]);
        }
    }
    output
}

fn bhsd_to_token_major(
    source: &[f32],
    output: &mut [f32],
    token_start: usize,
    sequence: usize,
    heads: usize,
    dim: usize,
) {
    for token in 0..sequence {
        for head in 0..heads {
            let source_start = (head * sequence + token) * dim;
            let output_start = ((token_start + token) * heads + head) * dim;
            output[output_start..output_start + dim]
                .copy_from_slice(&source[source_start..source_start + dim]);
        }
    }
}

fn error(detail: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::kernels::testutil::Owned;

    struct RaggedPaddedBias<'a> {
        q_lengths: &'a [usize],
        kv_lengths: &'a [usize],
        causal: bool,
    }

    impl AttnBias for RaggedPaddedBias<'_> {
        fn at(&self, batch: usize, _head: usize, query: usize, key: usize) -> f32 {
            if query >= self.q_lengths[batch] || key >= self.kv_lengths[batch] {
                return f32::NEG_INFINITY;
            }
            let offset = self.kv_lengths[batch] as i64 - self.q_lengths[batch] as i64;
            if self.causal && key as i64 > query as i64 + offset {
                f32::NEG_INFINITY
            } else {
                0.0
            }
        }
    }

    fn kernel(causal: bool) -> VarlenAttentionKernel {
        VarlenAttentionKernel {
            scale: Some(0.5),
            is_causal: causal,
            softcap: None,
        }
    }

    fn run(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        cu_q: &[i64],
        cu_kv: &[i64],
        nonpad: Option<&[i64]>,
        q_heads: usize,
        kv_heads: usize,
        dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let total_q = *cu_q.last().unwrap() as usize;
        let total_kv = *cu_kv.last().unwrap() as usize;
        let q_owned = Owned::f32(&[total_q, q_heads, dim], q);
        let k_owned = Owned::f32(&[total_kv, kv_heads, dim], k);
        let v_owned = Owned::f32(&[total_kv, kv_heads, dim], v);
        let cu_q_owned = Owned::i64(&[cu_q.len()], cu_q);
        let cu_kv_owned = Owned::i64(&[cu_kv.len()], cu_kv);
        let nonpad_owned = nonpad.map(|values| Owned::i64(&[values.len()], values));
        let mut output = Owned::zeros_f32(&[total_q, q_heads, dim]);
        let mut inputs = vec![
            q_owned.view(),
            k_owned.view(),
            v_owned.view(),
            cu_q_owned.view(),
            cu_kv_owned.view(),
        ];
        if let Some(nonpad) = nonpad_owned.as_ref() {
            inputs.push(nonpad.view());
        }
        kernel(causal)
            .execute(&inputs, &mut [output.view_mut()])
            .unwrap();
        output.to_f32()
    }

    #[allow(clippy::too_many_arguments)]
    fn padded_reference(
        q: &[f32],
        k: &[f32],
        v: &[f32],
        q_lengths: &[usize],
        allocated_kv_lengths: &[usize],
        kv_lengths: &[usize],
        q_heads: usize,
        kv_heads: usize,
        dim: usize,
        causal: bool,
    ) -> Vec<f32> {
        let batch = q_lengths.len();
        let max_q = q_lengths.iter().copied().max().unwrap_or(0);
        let max_kv = kv_lengths.iter().copied().max().unwrap_or(0);
        let mut padded_q = vec![0.0; batch * q_heads * max_q * dim];
        let mut padded_k = vec![0.0; batch * kv_heads * max_kv * dim];
        let mut padded_v = padded_k.clone();
        let mut q_base = 0;
        let mut kv_base = 0;
        for batch_index in 0..batch {
            for token in 0..q_lengths[batch_index] {
                for head in 0..q_heads {
                    for channel in 0..dim {
                        padded_q
                            [((batch_index * q_heads + head) * max_q + token) * dim + channel] =
                            q[((q_base + token) * q_heads + head) * dim + channel];
                    }
                }
            }
            for token in 0..kv_lengths[batch_index] {
                for head in 0..kv_heads {
                    for channel in 0..dim {
                        let source = ((kv_base + token) * kv_heads + head) * dim + channel;
                        let target =
                            ((batch_index * kv_heads + head) * max_kv + token) * dim + channel;
                        padded_k[target] = k[source];
                        padded_v[target] = v[source];
                    }
                }
            }
            q_base += q_lengths[batch_index];
            kv_base += allocated_kv_lengths[batch_index];
        }
        let tensors = SdpaTensors {
            q: &padded_q,
            k: &padded_k,
            v: &padded_v,
            batch,
            num_heads: q_heads,
            num_kv_heads: kv_heads,
            q_seq: max_q,
            kv_seq: max_kv,
            head_size: dim,
            v_head_size: dim,
        };
        let config = SdpaConfig {
            scale: ScaleMode::SplitSqrt(0.5),
            softcap: None,
            causal: false,
            past_seq: 0,
            causal_fill: f32::NEG_INFINITY,
        };
        let bias = RaggedPaddedBias {
            q_lengths,
            kv_lengths,
            causal,
        };
        let mut padded_output = vec![0.0; batch * q_heads * max_q * dim];
        sdpa_f32_scalar(&tensors, &config, &bias, &NoMask, &mut padded_output, None);
        let mut packed = Vec::with_capacity(q_lengths.iter().sum::<usize>() * q_heads * dim);
        for (batch_index, &q_len) in q_lengths.iter().enumerate() {
            for token in 0..q_len {
                for head in 0..q_heads {
                    let start = ((batch_index * q_heads + head) * max_q + token) * dim;
                    packed.extend_from_slice(&padded_output[start..start + dim]);
                }
            }
        }
        packed
    }

    fn data(tokens: usize, heads: usize, dim: usize, phase: f32) -> Vec<f32> {
        (0..tokens * heads * dim)
            .map(|index| ((index as f32 + phase) * 0.173).sin())
            .collect()
    }

    fn assert_bits_eq(actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "bit mismatch at output {index}: {actual:?} vs {expected:?}"
            );
        }
    }

    #[test]
    fn packed_varlen_is_bit_identical_to_padded_masked_reference() {
        for (q_lengths, kv_allocated, kv_lengths, q_heads, kv_heads) in [
            (vec![3], vec![3], vec![3], 2, 2),
            (vec![3, 1, 5], vec![3, 1, 5], vec![3, 1, 5], 4, 2),
            (vec![2, 2, 2], vec![2, 2, 2], vec![2, 2, 2], 2, 1),
            (vec![3, 1, 5], vec![4, 2, 5], vec![3, 1, 5], 4, 2),
        ] {
            let dim = 2;
            let total_q = q_lengths.iter().sum();
            let total_kv = kv_allocated.iter().sum();
            let q = data(total_q, q_heads, dim, 1.0);
            let mut k = data(total_kv, kv_heads, dim, 7.0);
            let mut v = data(total_kv, kv_heads, dim, 13.0);
            // Poison allocated-but-nonpad tails so ignoring nonpad_kv_seqlen
            // observably changes the result.
            let mut base = 0;
            for (&allocated, &valid) in kv_allocated.iter().zip(&kv_lengths) {
                for token in valid..allocated {
                    for item in (base + token) * kv_heads * dim..(base + token + 1) * kv_heads * dim
                    {
                        k[item] = 50.0 + item as f32;
                        v[item] = -70.0 - item as f32;
                    }
                }
                base += allocated;
            }
            let cumulative = |lengths: &[usize]| {
                let mut total = 0i64;
                std::iter::once(0)
                    .chain(lengths.iter().map(|&length| {
                        total += length as i64;
                        total
                    }))
                    .collect::<Vec<_>>()
            };
            let cu_q = cumulative(&q_lengths);
            let cu_kv = cumulative(&kv_allocated);
            let nonpad = kv_lengths
                .iter()
                .map(|&length| length as i64)
                .collect::<Vec<_>>();
            let packed = run(
                &q,
                &k,
                &v,
                &cu_q,
                &cu_kv,
                Some(&nonpad),
                q_heads,
                kv_heads,
                dim,
                true,
            );
            let padded = padded_reference(
                &q,
                &k,
                &v,
                &q_lengths,
                &kv_allocated,
                &kv_lengths,
                q_heads,
                kv_heads,
                dim,
                true,
            );
            assert_bits_eq(&packed, &padded);

            if kv_allocated != kv_lengths {
                let ignoring_nonpad = run(
                    &q, &k, &v, &cu_q, &cu_kv, None, q_heads, kv_heads, dim, true,
                );
                assert_ne!(
                    ignoring_nonpad
                        .iter()
                        .map(|x| x.to_bits())
                        .collect::<Vec<_>>(),
                    padded.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                    "oracle must fail if nonpad_kv_seqlen is ignored"
                );
            }
        }
    }

    #[test]
    fn packed_boundaries_prevent_cross_sequence_attention() {
        let lengths = [3usize, 1, 5];
        let cu = [0i64, 3, 4, 9];
        let q = data(9, 1, 2, 2.0);
        let k = data(9, 1, 2, 5.0);
        let mut v = data(9, 1, 2, 9.0);
        v[3 * 2] = 10_000.0;
        v[3 * 2 + 1] = -10_000.0;
        let correct = run(&q, &k, &v, &cu, &cu, None, 1, 1, 2, false);
        let reference = padded_reference(&q, &k, &v, &lengths, &lengths, &lengths, 1, 1, 2, false);
        assert_bits_eq(&correct, &reference);

        // Mutation oracle: move the first boundary across sequence 1's token.
        // Sequence 0 then attends to a neighbor and must no longer match.
        let leaked_cu = [0i64, 4, 4, 9];
        let leaked = run(&q, &k, &v, &cu, &leaked_cu, None, 1, 1, 2, false);
        assert_ne!(
            leaked[..lengths[0] * 2]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            reference[..lengths[0] * 2]
                .iter()
                .map(|x| x.to_bits())
                .collect::<Vec<_>>(),
            "cross-sequence boundary mutation must be detected"
        );
    }
}
