//! Checked reference implementation of `pkg.nxrt::SparseKvGather` v1.
//!
//! The reusable helper is also the indexing seam used by the Phase-1
//! `CompressedSparseAttention` reference kernel. The public operator is strict:
//! negative and out-of-range indices are errors. The fused attention path uses
//! the same helper with the official `-1` invalid-entry sentinel enabled.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use super::{check_arity, write_dense_f32};
use crate::strided::{elem_offset, next_index};

pub const CSA_WINDOW_SIZE: usize = 128;
pub const CSA_RATIO_4: usize = 4;
pub const CSA_RATIO_128: usize = 128;

const OP: &str = "SparseKvGather";
const INDEX_LAYOUT_VERSION: i64 = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsaIndexPlan {
    pub queries: usize,
    pub selections: usize,
    pub indices: Vec<i64>,
}

#[derive(Clone, Copy)]
enum InvalidIndexPolicy {
    Error,
    MinusOneMask,
}

pub(crate) struct GatheredKv {
    pub values: Vec<f32>,
    pub valid: Vec<bool>,
}

pub struct SparseKvGatherFactory;

struct SparseKvGatherKernel;

impl KernelFactory for SparseKvGatherFactory {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let version = node
            .attr("index_layout_version")
            .and_then(|attribute| attribute.as_int())
            .unwrap_or(INDEX_LAYOUT_VERSION);
        if version != INDEX_LAYOUT_VERSION {
            return Err(error(format!(
                "index_layout_version must be {INDEX_LAYOUT_VERSION}, got {version}"
            )));
        }
        let out_of_range = match node.attr("out_of_range") {
            Some(attribute) => attribute
                .as_str()
                .ok_or_else(|| error("attribute out_of_range must be a UTF-8 string"))?,
            None => "error",
        };
        if out_of_range != "error" {
            return Err(unsupported(format!(
                "out_of_range='{out_of_range}' is unsupported; v1 requires 'error'"
            )));
        }
        if input_shapes.len() >= 2 {
            infer_output_shape(&input_shapes[0], &input_shapes[1])?;
        }
        Ok(Box::new(SparseKvGatherKernel))
    }
}

impl Kernel for SparseKvGatherKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity(OP, inputs, outputs, 2, 3, 1)?;
        require_dtype("cache", inputs[0].dtype, DataType::Float32)?;
        require_index_dtype("indices", inputs[1].dtype)?;
        require_dtype("selected", outputs[0].dtype, DataType::Float32)?;

        let cache_shape = shape4("cache", inputs[0].shape)?;
        let indices_shape = shape4("indices", inputs[1].shape)?;
        let expected_output = infer_output_shape(inputs[0].shape, inputs[1].shape)?;
        if outputs[0].shape != expected_output {
            return Err(error(format!(
                "selected must have shape {expected_output:?}, got {:?}",
                outputs[0].shape
            )));
        }

        let cache = read_dense_f32(&inputs[0], "cache")?;
        let indices = read_dense_indices(&inputs[1], "indices")?;
        let valid_lengths = inputs
            .get(2)
            .filter(|input| !input.is_absent())
            .map(|input| read_valid_lengths(input, cache_shape[0], cache_shape[2]))
            .transpose()?;
        let gathered = gather_f32(
            &cache,
            cache_shape,
            &indices,
            indices_shape,
            valid_lengths.as_deref(),
            InvalidIndexPolicy::Error,
        )?;
        write_dense_f32(&mut outputs[0], &gathered.values)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        true
    }
}

/// Infer `[B,G,Q,K,D]` from cache `[B,G,C,D]` and indices `[B,G,Q,K]`.
pub fn infer_output_shape(cache: &[usize], indices: &[usize]) -> Result<Vec<usize>> {
    let cache = shape4("cache", cache)?;
    let indices = shape4("indices", indices)?;
    if cache[0] != indices[0] || cache[1] != indices[1] {
        return Err(error(format!(
            "cache and indices batch/group dimensions must match, got {:?} and {:?}",
            &cache[..2],
            &indices[..2]
        )));
    }
    let output = vec![cache[0], cache[1], indices[2], indices[3], cache[3]];
    checked_layout(&output, std::mem::size_of::<f32>(), "selected")?;
    Ok(output)
}

/// Strict standalone gather preserving index order and duplicates.
pub fn sparse_kv_gather_f32(
    cache: &[f32],
    cache_shape: [usize; 4],
    indices: &[i64],
    indices_shape: [usize; 4],
    valid_lengths: Option<&[usize]>,
) -> Result<Vec<f32>> {
    Ok(gather_f32(
        cache,
        cache_shape,
        indices,
        indices_shape,
        valid_lengths,
        InvalidIndexPolicy::Error,
    )?
    .values)
}

pub(crate) fn sparse_kv_gather_masked_f32(
    cache: &[f32],
    cache_shape: [usize; 4],
    indices: &[i64],
    indices_shape: [usize; 4],
    valid_lengths: Option<&[usize]>,
) -> Result<GatheredKv> {
    gather_f32(
        cache,
        cache_shape,
        indices,
        indices_shape,
        valid_lengths,
        InvalidIndexPolicy::MinusOneMask,
    )
}

fn gather_f32(
    cache: &[f32],
    cache_shape: [usize; 4],
    indices: &[i64],
    indices_shape: [usize; 4],
    valid_lengths: Option<&[usize]>,
    invalid_policy: InvalidIndexPolicy,
) -> Result<GatheredKv> {
    let [batch, groups, cache_len, dim] = cache_shape;
    let [index_batch, index_groups, queries, selections] = indices_shape;
    if (batch, groups) != (index_batch, index_groups) {
        return Err(error(format!(
            "cache and indices batch/group dimensions must match, got [{batch},{groups}] and [{index_batch},{index_groups}]"
        )));
    }
    let cache_elements = checked_product(&cache_shape, "cache element count")?;
    let index_elements = checked_product(&indices_shape, "indices element count")?;
    if cache.len() != cache_elements {
        return Err(error(format!(
            "cache contains {} elements, expected {cache_elements}",
            cache.len()
        )));
    }
    if indices.len() != index_elements {
        return Err(error(format!(
            "indices contains {} elements, expected {index_elements}",
            indices.len()
        )));
    }
    if let Some(lengths) = valid_lengths {
        if lengths.len() != batch {
            return Err(error(format!(
                "valid_lengths must contain {batch} entries, got {}",
                lengths.len()
            )));
        }
        if let Some((b, &length)) = lengths
            .iter()
            .enumerate()
            .find(|(_, length)| **length > cache_len)
        {
            return Err(error(format!(
                "valid_lengths[{b}]={length} exceeds cache length {cache_len}"
            )));
        }
    }

    let output_shape = [batch, groups, queries, selections, dim];
    let output_elements = checked_layout(&output_shape, std::mem::size_of::<f32>(), "selected")?;
    let selected_records = checked_product(
        &[batch, groups, queries, selections],
        "selected record count",
    )?;
    let mut values = fallible_filled(output_elements, 0.0f32, "selected values")?;
    let mut valid = fallible_filled(selected_records, false, "selected validity")?;

    for b in 0..batch {
        let valid_length = valid_lengths.map_or(cache_len, |lengths| lengths[b]);
        for g in 0..groups {
            for q in 0..queries {
                for k in 0..selections {
                    let record = checked_flat4(
                        [b, g, q, k],
                        [batch, groups, queries, selections],
                        "indices",
                    )?;
                    let raw = indices[record];
                    if raw == -1 && matches!(invalid_policy, InvalidIndexPolicy::MinusOneMask) {
                        continue;
                    }
                    let index = usize::try_from(raw).map_err(|_| {
                        error(format!(
                            "negative index {raw} at [batch={b}, group={g}, query={q}, index={k}]"
                        ))
                    })?;
                    if index >= valid_length {
                        return Err(error(format!(
                            "index {raw} at [batch={b}, group={g}, query={q}, index={k}] is out of range for valid length {valid_length}"
                        )));
                    }
                    let source_record =
                        checked_flat3([b, g, index], [batch, groups, cache_len], "cache record")?;
                    let source = source_record
                        .checked_mul(dim)
                        .ok_or_else(|| error("cache source offset overflow"))?;
                    let destination = record
                        .checked_mul(dim)
                        .ok_or_else(|| error("selected destination offset overflow"))?;
                    let source_end = source
                        .checked_add(dim)
                        .ok_or_else(|| error("cache source end overflow"))?;
                    let destination_end = destination
                        .checked_add(dim)
                        .ok_or_else(|| error("selected destination end overflow"))?;
                    values[destination..destination_end]
                        .copy_from_slice(&cache[source..source_end]);
                    valid[record] = true;
                }
            }
        }
    }
    Ok(GatheredKv { values, valid })
}

/// Build the official prefill candidate order:
/// `[left-padded ascending sliding window, compressed history]`.
pub fn prefill_csa_indices(
    sequence_length: usize,
    compression_ratio: usize,
    ratio4_topk: Option<(&[i64], usize)>,
) -> Result<CsaIndexPlan> {
    require_ratio(compression_ratio)?;
    let compressed_width = if compression_ratio == CSA_RATIO_4 {
        let (indices, width) = ratio4_topk.ok_or_else(|| {
            error("ratio-4 prefill index construction requires learned top-k indices")
        })?;
        let expected = sequence_length
            .checked_mul(width)
            .ok_or_else(|| error("ratio-4 top-k element count overflow"))?;
        if indices.len() != expected {
            return Err(error(format!(
                "ratio-4 top-k contains {} elements, expected {expected}",
                indices.len()
            )));
        }
        width
    } else {
        if ratio4_topk.is_some() {
            return Err(error(
                "ratio-128 index construction does not accept top-k indices",
            ));
        }
        sequence_length / CSA_RATIO_128
    };
    let selections = CSA_WINDOW_SIZE
        .checked_add(compressed_width)
        .ok_or_else(|| error("candidate width overflow"))?;
    let total = sequence_length
        .checked_mul(selections)
        .ok_or_else(|| error("candidate index count overflow"))?;
    let mut result = fallible_filled(total, -1i64, "prefill candidate indices")?;

    for query in 0..sequence_length {
        let token_count = query
            .checked_add(1)
            .ok_or_else(|| error("prefill query position overflow"))?;
        let window_start = token_count.saturating_sub(CSA_WINDOW_SIZE);
        let window_count = query
            .checked_sub(window_start)
            .and_then(|count| count.checked_add(1))
            .ok_or_else(|| error("prefill window length overflow"))?;
        let left_padding = CSA_WINDOW_SIZE - window_count;
        let row = query
            .checked_mul(selections)
            .ok_or_else(|| error("prefill candidate row offset overflow"))?;
        for (slot, index) in (window_start..=query).enumerate() {
            result[row + left_padding + slot] =
                i64::try_from(index).map_err(|_| error("prefill window index exceeds i64::MAX"))?;
        }

        let available = token_count / compression_ratio;
        if compression_ratio == CSA_RATIO_4 {
            let (topk, width) = ratio4_topk.expect("ratio-4 top-k was validated above");
            for slot in 0..width {
                let compressed = topk[query * width + slot];
                if let Ok(compressed) = usize::try_from(compressed)
                    && compressed < available
                {
                    let offset = sequence_length
                        .checked_add(compressed)
                        .ok_or_else(|| error("prefill compressed index offset overflow"))?;
                    result[row + CSA_WINDOW_SIZE + slot] = i64::try_from(offset)
                        .map_err(|_| error("prefill compressed index exceeds i64::MAX"))?;
                }
            }
        } else {
            for compressed in 0..available {
                let offset = sequence_length
                    .checked_add(compressed)
                    .ok_or_else(|| error("prefill compressed index offset overflow"))?;
                result[row + CSA_WINDOW_SIZE + compressed] = i64::try_from(offset)
                    .map_err(|_| error("prefill compressed index exceeds i64::MAX"))?;
            }
        }
    }
    Ok(CsaIndexPlan {
        queries: sequence_length,
        selections,
        indices: result,
    })
}

/// Build one decode candidate row. `absolute_position` is the query token's
/// zero-based absolute position and dense-ring indices are returned oldest first.
pub fn decode_csa_indices(
    absolute_position: usize,
    compression_ratio: usize,
    ratio4_topk: Option<&[i64]>,
) -> Result<CsaIndexPlan> {
    require_ratio(compression_ratio)?;
    let token_count = absolute_position
        .checked_add(1)
        .ok_or_else(|| error("absolute position overflow"))?;
    let compressed_width = if compression_ratio == CSA_RATIO_4 {
        ratio4_topk
            .ok_or_else(|| error("ratio-4 decode index construction requires learned top-k"))?
            .len()
    } else {
        if ratio4_topk.is_some() {
            return Err(error(
                "ratio-128 index construction does not accept top-k indices",
            ));
        }
        token_count / CSA_RATIO_128
    };
    let selections = CSA_WINDOW_SIZE
        .checked_add(compressed_width)
        .ok_or_else(|| error("decode candidate width overflow"))?;
    let mut result = fallible_filled(selections, -1i64, "decode candidate indices")?;

    if token_count <= CSA_WINDOW_SIZE {
        let left_padding = CSA_WINDOW_SIZE - token_count;
        for physical in 0..token_count {
            result[left_padding + physical] = i64::try_from(physical)
                .map_err(|_| error("decode window index exceeds i64::MAX"))?;
        }
    } else {
        let newest = absolute_position % CSA_WINDOW_SIZE;
        let mut slot = 0;
        for physical in newest + 1..CSA_WINDOW_SIZE {
            result[slot] =
                i64::try_from(physical).map_err(|_| error("decode ring index exceeds i64::MAX"))?;
            slot += 1;
        }
        for physical in 0..=newest {
            result[slot] =
                i64::try_from(physical).map_err(|_| error("decode ring index exceeds i64::MAX"))?;
            slot += 1;
        }
    }

    if compression_ratio == CSA_RATIO_4 {
        let available = token_count / CSA_RATIO_4;
        for (slot, &compressed) in ratio4_topk
            .expect("ratio-4 top-k was validated above")
            .iter()
            .enumerate()
        {
            if let Ok(compressed) = usize::try_from(compressed)
                && compressed < available
            {
                let offset = CSA_WINDOW_SIZE
                    .checked_add(compressed)
                    .ok_or_else(|| error("decode compressed index offset overflow"))?;
                result[CSA_WINDOW_SIZE + slot] = i64::try_from(offset)
                    .map_err(|_| error("decode compressed index exceeds i64::MAX"))?;
            }
        }
    } else {
        for compressed in 0..compressed_width {
            let offset = CSA_WINDOW_SIZE
                .checked_add(compressed)
                .ok_or_else(|| error("decode compressed index offset overflow"))?;
            result[CSA_WINDOW_SIZE + compressed] = i64::try_from(offset)
                .map_err(|_| error("decode compressed index exceeds i64::MAX"))?;
        }
    }
    Ok(CsaIndexPlan {
        queries: 1,
        selections,
        indices: result,
    })
}

pub(crate) fn checked_product(shape: &[usize], what: &str) -> Result<usize> {
    shape.iter().try_fold(1usize, |count, &dimension| {
        count
            .checked_mul(dimension)
            .ok_or_else(|| error(format!("{what} overflow for shape {shape:?}")))
    })
}

pub(crate) fn checked_layout(shape: &[usize], element_size: usize, what: &str) -> Result<usize> {
    let elements = checked_product(shape, &format!("{what} element count"))?;
    elements
        .checked_mul(element_size)
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| {
            error(format!(
                "{what} byte count overflow or exceeds isize::MAX for shape {shape:?}"
            ))
        })?;
    Ok(elements)
}

pub(crate) fn read_dense_f32(view: &TensorView, name: &str) -> Result<Vec<f32>> {
    require_dtype(name, view.dtype, DataType::Float32)?;
    read_dense(view, name, |pointer, offset| {
        // SAFETY: `view.validate` plus the caller-owned backing bounds check make
        // each in-shape strided offset readable as one f32.
        unsafe { *pointer.cast::<f32>().offset(offset) }
    })
}

pub(crate) fn read_dense_indices(view: &TensorView, name: &str) -> Result<Vec<i64>> {
    require_index_dtype(name, view.dtype)?;
    match view.dtype {
        DataType::Int32 => read_dense(view, name, |pointer, offset| {
            // SAFETY: same argument as `read_dense_f32`, for i32 storage.
            unsafe { *pointer.cast::<i32>().offset(offset) as i64 }
        }),
        DataType::Int64 => read_dense(view, name, |pointer, offset| {
            // SAFETY: same argument as `read_dense_f32`, for i64 storage.
            unsafe { *pointer.cast::<i64>().offset(offset) }
        }),
        _ => unreachable!("index dtype was validated"),
    }
}

fn read_dense<T>(
    view: &TensorView,
    name: &str,
    read: impl Fn(*const u8, isize) -> T,
) -> Result<Vec<T>> {
    view.validate()?;
    let elements = checked_layout(view.shape, view.dtype.byte_size(), name)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| error(format!("failed to allocate {elements} elements for {name}")))?;
    if elements == 0 {
        return Ok(output);
    }
    let mut index = fallible_filled(view.shape.len(), 0usize, "strided read index")?;
    let origin = view.data_ptr::<u8>();
    loop {
        output.push(read(origin, elem_offset(view.strides, &index)));
        if !next_index(view.shape, &mut index) {
            break;
        }
    }
    Ok(output)
}

fn read_valid_lengths(view: &TensorView, batch: usize, cache_len: usize) -> Result<Vec<usize>> {
    if view.shape != [batch] {
        return Err(error(format!(
            "valid_lengths must have shape [{batch}], got {:?}",
            view.shape
        )));
    }
    let raw = read_dense_indices(view, "valid_lengths")?;
    raw.into_iter()
        .enumerate()
        .map(|(b, length)| {
            let length = usize::try_from(length)
                .map_err(|_| error(format!("valid_lengths[{b}] must be non-negative")))?;
            if length > cache_len {
                return Err(error(format!(
                    "valid_lengths[{b}]={length} exceeds cache length {cache_len}"
                )));
            }
            Ok(length)
        })
        .collect()
}

pub(crate) fn fallible_filled<T: Clone>(elements: usize, value: T, what: &str) -> Result<Vec<T>> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(elements)
        .map_err(|_| error(format!("failed to allocate {elements} elements for {what}")))?;
    output.resize(elements, value);
    Ok(output)
}

fn checked_flat3(index: [usize; 3], shape: [usize; 3], what: &str) -> Result<usize> {
    index[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(index[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(index[2]))
        .ok_or_else(|| error(format!("{what} offset overflow")))
}

fn checked_flat4(index: [usize; 4], shape: [usize; 4], what: &str) -> Result<usize> {
    index[0]
        .checked_mul(shape[1])
        .and_then(|value| value.checked_add(index[1]))
        .and_then(|value| value.checked_mul(shape[2]))
        .and_then(|value| value.checked_add(index[2]))
        .and_then(|value| value.checked_mul(shape[3]))
        .and_then(|value| value.checked_add(index[3]))
        .ok_or_else(|| error(format!("{what} offset overflow")))
}

fn shape4(name: &str, shape: &[usize]) -> Result<[usize; 4]> {
    shape
        .try_into()
        .map_err(|_| error(format!("{name} must be rank 4, got shape {shape:?}")))
}

fn require_ratio(ratio: usize) -> Result<()> {
    if !matches!(ratio, CSA_RATIO_4 | CSA_RATIO_128) {
        return Err(error(format!(
            "compression_ratio must be 4 or 128, got {ratio}"
        )));
    }
    Ok(())
}

fn require_dtype(name: &str, got: DataType, expected: DataType) -> Result<()> {
    if got != expected {
        return Err(error(format!(
            "{name} must have dtype {expected:?}, got {got:?}"
        )));
    }
    Ok(())
}

fn require_index_dtype(name: &str, dtype: DataType) -> Result<()> {
    if !matches!(dtype, DataType::Int32 | DataType::Int64) {
        return Err(error(format!(
            "{name} must have dtype Int32 or Int64, got {dtype:?}"
        )));
    }
    Ok(())
}

fn unsupported(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: Unsupported: {}", message.into()))
}

fn error(message: impl Into<String>) -> EpError {
    EpError::KernelFailed(format!("{OP}: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testutil::Owned;

    #[test]
    fn strict_gather_preserves_order_and_duplicates() {
        let cache = [
            0.0, 1.0, // record 0
            10.0, 11.0, // record 1
            20.0, 21.0, // record 2
            30.0, 31.0, // record 3
        ];
        let selected = sparse_kv_gather_f32(
            &cache,
            [1, 1, 4, 2],
            &[2, 0, 2, 3],
            [1, 1, 1, 4],
            Some(&[4]),
        )
        .unwrap();
        assert_eq!(selected, [20.0, 21.0, 0.0, 1.0, 20.0, 21.0, 30.0, 31.0]);
    }

    #[test]
    fn registered_kernel_executes_declared_shape() {
        let kernel = SparseKvGatherKernel;
        let cache = Owned::f32(&[1, 1, 3, 2], &[0.0, 1.0, 10.0, 11.0, 20.0, 21.0]);
        let indices = Owned::i32(&[1, 1, 1, 3], &[2, 0, 2]);
        let mut selected = Owned::zeros_f32(&[1, 1, 1, 3, 2]);
        kernel
            .execute(&[cache.view(), indices.view()], &mut [selected.view_mut()])
            .unwrap();
        assert_eq!(selected.to_f32(), [20.0, 21.0, 0.0, 1.0, 20.0, 21.0]);
    }

    #[test]
    fn gather_reports_exact_offending_coordinate() {
        let error = sparse_kv_gather_f32(
            &[0.0; 8],
            [1, 1, 4, 2],
            &[0, 1, 3],
            [1, 1, 1, 3],
            Some(&[3]),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("[batch=0, group=0, query=0, index=2]"));
        assert!(error.contains("valid length 3"));
    }

    #[test]
    fn masked_gather_zeroes_minus_one_without_reordering() {
        let gathered = sparse_kv_gather_masked_f32(
            &[1.0, 2.0, 3.0, 4.0],
            [1, 1, 2, 2],
            &[1, -1, 0],
            [1, 1, 1, 3],
            None,
        )
        .unwrap();
        assert_eq!(gathered.values, [3.0, 4.0, 0.0, 0.0, 1.0, 2.0]);
        assert_eq!(gathered.valid, [true, false, true]);
    }

    #[test]
    fn empty_selection_is_a_valid_contiguous_output() {
        let selected =
            sparse_kv_gather_f32(&[1.0, 2.0], [1, 1, 1, 2], &[], [1, 1, 3, 0], None).unwrap();
        assert!(selected.is_empty());
    }

    #[test]
    fn ratio4_prefill_indices_match_frozen_formula() {
        let topk = [
            0, 1, // s=0: no completed records
            0, 1, // s=1
            0, 1, // s=2
            0, 1, // s=3: c0 becomes causal
            1, 0, // s=4
            1, 0, // s=5
            1, 0, // s=6
            1, 0, // s=7: c0,c1 causal
        ];
        let plan = prefill_csa_indices(8, CSA_RATIO_4, Some((&topk, 2))).unwrap();
        assert_eq!(plan.selections, 130);
        assert_eq!(
            &plan.indices[0..128],
            &[{
                let mut row = [-1i64; 128];
                row[127] = 0;
                row
            }][0]
        );
        assert_eq!(&plan.indices[128..130], &[-1, -1]);

        let s3 = 3 * plan.selections;
        assert_eq!(&plan.indices[s3 + 124..s3 + 128], &[0, 1, 2, 3]);
        assert_eq!(&plan.indices[s3 + 128..s3 + 130], &[8, -1]);

        let s7 = 7 * plan.selections;
        assert_eq!(&plan.indices[s7 + 120..s7 + 128], &[0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(&plan.indices[s7 + 128..s7 + 130], &[9, 8]);
    }

    #[test]
    fn ratio128_prefill_indices_match_frozen_formula() {
        let plan = prefill_csa_indices(257, CSA_RATIO_128, None).unwrap();
        assert_eq!(plan.selections, 130);

        let s126 = 126 * plan.selections;
        assert_eq!(&plan.indices[s126 + 128..s126 + 130], &[-1, -1]);
        let s127 = 127 * plan.selections;
        assert_eq!(
            &plan.indices[s127..s127 + 128],
            &(0..128).map(i64::from).collect::<Vec<_>>()
        );
        assert_eq!(&plan.indices[s127 + 128..s127 + 130], &[257, -1]);
        let s255 = 255 * plan.selections;
        assert_eq!(&plan.indices[s255..s255 + 4], &[128, 129, 130, 131]);
        assert_eq!(&plan.indices[s255 + 124..s255 + 128], &[252, 253, 254, 255]);
        assert_eq!(&plan.indices[s255 + 128..s255 + 130], &[257, 258]);
    }

    #[test]
    fn decode_ring_and_compressed_offsets_match_frozen_formula() {
        let ratio4 = decode_csa_indices(130, CSA_RATIO_4, Some(&[31, 0, 32])).unwrap();
        assert_eq!(&ratio4.indices[..5], &[3, 4, 5, 6, 7]);
        assert_eq!(&ratio4.indices[125..128], &[0, 1, 2]);
        assert_eq!(&ratio4.indices[128..], &[159, 128, -1]);

        let ratio128 = decode_csa_indices(255, CSA_RATIO_128, None).unwrap();
        assert_eq!(&ratio128.indices[..4], &[0, 1, 2, 3]);
        assert_eq!(&ratio128.indices[124..128], &[124, 125, 126, 127]);
        assert_eq!(&ratio128.indices[128..], &[128, 129]);
    }
}
