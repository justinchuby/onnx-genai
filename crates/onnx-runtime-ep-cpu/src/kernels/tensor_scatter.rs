//! `TensorScatter` (ai.onnx opset 24): the standardized KV-cache update.
//!
//! Models an in-place write into a fixed-capacity cache buffer as a functional
//! op. `past_cache` and `present_cache` share the shape
//! `(batch, D1, ..., max_sequence_length, ..., Dn)`; `update` differs only in
//! the sequence dimension named by `axis`. Each batch row writes its slice at
//! `write_indices[batch]`, so prefill (index 0) and decode (index = current
//! valid length) are the same operation.
//!
//! Spec note: the specification prose states "the write index is modulo
//! max_sequence_length" (and the `mode` attribute doc repeats it), while its
//! illustrative pseudocode applies `np.mod` to the whole index tuple. Those
//! disagree whenever a prefix coordinate (batch, head, ...) is at least
//! `max_sequence_length`.
//!
//! Measured, the ONNX reference implementation follows the *pseudocode*: with
//! `batch = 5` and `max_sequence_length = 4`, batch 4 wraps onto batch 0, so
//! one sample's write silently overwrites another sample's cache and the last
//! sample keeps stale contents. We implement the prose instead — only the
//! sequence coordinate wraps — because cross-sample cache corruption is not a
//! behavior any caller can want, and both prose statements agree with us.
//! Reported upstream as onnx/onnx#8353; once that lands the two agree again.

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{Attribute, DataType, Node};

use super::{check_arity, to_dense_i64};
use crate::dispatch_arith;
use crate::dtype::{NumericElem, to_dense, write_dense};

/// Row-major strides for `shape`.
fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for dimension in (0..shape.len().saturating_sub(1)).rev() {
        strides[dimension] = strides[dimension + 1] * shape[dimension + 1];
    }
    strides
}

/// How an out-of-capacity write index is resolved.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum WriteMode {
    /// Writes must land inside the buffer; anything past the end is an error.
    Linear,
    /// The sequence coordinate wraps modulo `max_sequence_length`.
    Circular,
}

pub struct TensorScatterKernel {
    axis: i64,
    mode: WriteMode,
}

pub struct TensorScatterFactory;

impl KernelFactory for TensorScatterFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let mode = match node.attr("mode").and_then(Attribute::as_str) {
            None | Some("linear") => WriteMode::Linear,
            Some("circular") => WriteMode::Circular,
            Some(value) => {
                return Err(EpError::KernelFailed(format!(
                    "TensorScatter: unsupported mode {value:?} (expected \"linear\" or \"circular\")"
                )));
            }
        };
        Ok(Box::new(TensorScatterKernel {
            axis: node.attr("axis").and_then(Attribute::as_int).unwrap_or(-2),
            mode,
        }))
    }
}

/// Claim-time rejection reasons, so an unsupported shape declines rather than
/// failing mid-execution.
pub(crate) fn tensor_scatter_unsupported_reason(input_dtypes: &[DataType]) -> Option<String> {
    if input_dtypes.is_empty() {
        return None;
    }
    if input_dtypes.len() < 2 || input_dtypes.len() > 3 {
        return Some(format!(
            "TensorScatter requires 2 or 3 inputs, got {}",
            input_dtypes.len()
        ));
    }
    // The ONNX type constraint `T` is wider than `dispatch_arith!` (it also
    // admits bool and the float8 variants). Declining here keeps the claim
    // honest: a dtype this kernel cannot dispatch must not be accepted and then
    // fail at execute, which is a hard session failure rather than a fallback.
    if !ARITH_DISPATCHABLE.contains(&input_dtypes[0]) {
        return Some(format!(
            "TensorScatter: cache dtype {:?} is not implemented by the CPU kernel",
            input_dtypes[0]
        ));
    }
    if input_dtypes[1] != input_dtypes[0] {
        return Some(format!(
            "TensorScatter: update dtype {:?} must match past_cache dtype {:?}",
            input_dtypes[1], input_dtypes[0]
        ));
    }
    if input_dtypes.len() == 3 && input_dtypes[2] != DataType::Int64 {
        return Some(format!(
            "TensorScatter: write_indices must have Int64 dtype, got {:?}",
            input_dtypes[2]
        ));
    }
    None
}

/// The dtypes `dispatch_arith!` can actually route, mirrored here so the claim
/// gate and the execute path cannot drift apart.
const ARITH_DISPATCHABLE: &[DataType] = &[
    DataType::Float32,
    DataType::Float16,
    DataType::BFloat16,
    DataType::Float64,
    DataType::Int8,
    DataType::Int16,
    DataType::Int32,
    DataType::Int64,
    DataType::Uint8,
    DataType::Uint16,
    DataType::Uint32,
    DataType::Uint64,
];

impl Kernel for TensorScatterKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        check_arity("TensorScatter", inputs, outputs, 2, 3, 1)?;
        dispatch_arith!(inputs[0].dtype, "TensorScatter", T => {
            tensor_scatter_typed::<T>(self, inputs, outputs)
        })
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        true
    }
}

fn tensor_scatter_typed<T: NumericElem>(
    kernel: &TensorScatterKernel,
    inputs: &[TensorView],
    outputs: &mut [TensorMut],
) -> Result<()> {
    let past = &inputs[0];
    let update = &inputs[1];

    if update.dtype != T::DTYPE || outputs[0].dtype != T::DTYPE {
        return Err(EpError::KernelFailed(
            "TensorScatter: past_cache, update, and present_cache must share a dtype".into(),
        ));
    }
    if outputs[0].shape != past.shape {
        return Err(EpError::KernelFailed(
            "TensorScatter: present_cache shape must match past_cache".into(),
        ));
    }
    let rank = past.shape.len();
    if update.shape.len() != rank {
        return Err(EpError::KernelFailed(format!(
            "TensorScatter: update rank {} must match past_cache rank {rank}",
            update.shape.len()
        )));
    }

    let axis = normalize_sequence_axis(kernel.axis, rank)?;
    // `batch_idx` is the first prefix coordinate, so the sequence axis has to
    // sit strictly after the batch axis for the per-sample write index to mean
    // anything.
    if axis == 0 {
        return Err(EpError::KernelFailed(
            "TensorScatter: axis must not select the batch dimension".into(),
        ));
    }
    for dimension in 0..rank {
        if dimension != axis && update.shape[dimension] != past.shape[dimension] {
            return Err(EpError::KernelFailed(format!(
                "TensorScatter: update dimension {} at index {dimension} must match past_cache \
                 dimension {} (only the sequence axis {axis} may differ)",
                update.shape[dimension], past.shape[dimension]
            )));
        }
    }

    let max_sequence_length = past.shape[axis];
    let sequence_length = update.shape[axis];
    if sequence_length > max_sequence_length {
        return Err(EpError::KernelFailed(format!(
            "TensorScatter: update sequence length {sequence_length} exceeds cache capacity \
             {max_sequence_length}"
        )));
    }
    if max_sequence_length == 0 {
        return Err(EpError::KernelFailed(
            "TensorScatter: cache capacity must be non-zero".into(),
        ));
    }

    let batch = past.shape[0];
    let write_indices = match inputs.get(2) {
        None => vec![0i64; batch],
        Some(view) => {
            if view.dtype != DataType::Int64 {
                return Err(EpError::KernelFailed(
                    "TensorScatter: write_indices must be Int64".into(),
                ));
            }
            let values = to_dense_i64(view)?;
            if values.len() != batch {
                return Err(EpError::KernelFailed(format!(
                    "TensorScatter: write_indices has {} entries but the batch is {batch}",
                    values.len()
                )));
            }
            values
        }
    };

    // The output starts as the untouched cache; only the written slice differs.
    let mut present = to_dense::<T>(past)?;
    let update_values = to_dense::<T>(update)?;
    let update_strides = contiguous(update.shape);
    let cache_strides = contiguous(past.shape);

    for (linear, value) in update_values.into_iter().enumerate() {
        let mut remainder = linear;
        let mut cache_offset = 0usize;
        let mut batch_index = 0usize;
        for dimension in 0..rank {
            let coordinate = remainder / update_strides[dimension];
            remainder %= update_strides[dimension];
            if dimension == 0 {
                batch_index = coordinate;
            }
            let coordinate = if dimension == axis {
                resolve_write_position(
                    write_indices[batch_index],
                    coordinate,
                    max_sequence_length,
                    kernel.mode,
                )?
            } else {
                coordinate
            };
            cache_offset += coordinate * cache_strides[dimension];
        }
        present[cache_offset] = value;
    }

    write_dense::<T>(&mut outputs[0], &present)
}

/// Resolve `write_indices[batch] + sequence_idx` into a cache coordinate.
fn resolve_write_position(
    write_index: i64,
    sequence_index: usize,
    max_sequence_length: usize,
    mode: WriteMode,
) -> Result<usize> {
    if write_index < 0 {
        return Err(EpError::KernelFailed(format!(
            "TensorScatter: write index {write_index} must not be negative"
        )));
    }
    let position = (write_index as u128) + (sequence_index as u128);
    match mode {
        WriteMode::Circular => Ok((position % max_sequence_length as u128) as usize),
        WriteMode::Linear => {
            if position >= max_sequence_length as u128 {
                return Err(EpError::KernelFailed(format!(
                    "TensorScatter: linear write position {position} exceeds cache capacity \
                     {max_sequence_length}; use mode=\"circular\" for a ring buffer"
                )));
            }
            Ok(position as usize)
        }
    }
}

/// Normalize a possibly-negative `axis` against `rank`. The attribute defaults
/// to -2, so negative values are the common case rather than an edge case.
fn normalize_sequence_axis(axis: i64, rank: usize) -> Result<usize> {
    let rank_i64 = rank as i64;
    let normalized = if axis < 0 { axis + rank_i64 } else { axis };
    if normalized < 0 || normalized >= rank_i64 {
        return Err(EpError::KernelFailed(format!(
            "TensorScatter: axis {axis} is out of range for rank {rank}"
        )));
    }
    Ok(normalized as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_negative_axis_resolves_from_the_back() {
        assert_eq!(normalize_sequence_axis(-2, 4).expect("in range"), 2);
        assert_eq!(normalize_sequence_axis(2, 4).expect("in range"), 2);
        assert!(normalize_sequence_axis(4, 4).is_err());
        assert!(normalize_sequence_axis(-5, 4).is_err());
    }

    #[test]
    fn linear_mode_refuses_to_write_past_the_end_instead_of_wrapping() {
        // The distinction matters: silently wrapping in linear mode would
        // overwrite the oldest tokens of the same sample.
        assert_eq!(
            resolve_write_position(2, 1, 8, WriteMode::Linear).expect("inside capacity"),
            3
        );
        assert!(resolve_write_position(7, 1, 8, WriteMode::Linear).is_err());
    }

    #[test]
    fn circular_mode_wraps_the_sequence_coordinate() {
        assert_eq!(
            resolve_write_position(7, 1, 8, WriteMode::Circular).expect("wraps"),
            0
        );
        assert_eq!(
            resolve_write_position(6, 3, 8, WriteMode::Circular).expect("wraps"),
            1
        );
    }

    #[test]
    fn a_large_write_index_cannot_overflow_the_position_sum() {
        // usize addition of two near-max values would wrap and silently select
        // a valid-looking slot; the u128 widening keeps it detectable.
        assert!(resolve_write_position(i64::MAX, 1, 8, WriteMode::Linear).is_err());
        assert_eq!(
            resolve_write_position(i64::MAX, 1, 8, WriteMode::Circular).expect("wraps"),
            ((i64::MAX as u128 + 1) % 8) as usize
        );
    }

    #[test]
    fn a_negative_write_index_is_rejected() {
        assert!(resolve_write_position(-1, 0, 8, WriteMode::Linear).is_err());
        assert!(resolve_write_position(-1, 0, 8, WriteMode::Circular).is_err());
    }

    #[test]
    fn a_cache_dtype_the_kernel_cannot_dispatch_declines_at_claim_time() {
        // ONNX's `T` constraint is wider than `dispatch_arith!` (it also admits
        // bool and the float8 variants). Claiming one of those and then failing
        // inside `execute` is a hard session failure rather than a fallback, so
        // the claim gate has to reject it.
        let reason =
            tensor_scatter_unsupported_reason(&[DataType::Bool, DataType::Bool, DataType::Int64])
                .expect("a non-dispatchable cache dtype must decline");
        assert!(reason.contains("not implemented"), "{reason}");
    }

    #[test]
    fn an_ordinary_kv_cache_node_is_claimed() {
        // f16 cache with Int64 write_indices is the realistic decode-phase node
        // and must not decline.
        assert!(
            tensor_scatter_unsupported_reason(&[
                DataType::Float16,
                DataType::Float16,
                DataType::Int64,
            ])
            .is_none()
        );
        // Prefill omits write_indices entirely.
        assert!(
            tensor_scatter_unsupported_reason(&[DataType::Float16, DataType::Float16]).is_none()
        );
    }
}
