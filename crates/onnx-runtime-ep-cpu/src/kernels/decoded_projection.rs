//! Row-decoded f32 projections shared by compressed weight representations.

use onnx_runtime_ep_api::{EpError, Result};
use rayon::prelude::*;

/// Decode each weight row once per projection and reuse it across all routed
/// tokens. The factory makes decoder scratch once per worker shard. Shards do
/// no nested Rayon work, so suspended tasks cannot retain additional rows.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_decoded_grouped<D>(
    context: &str,
    input: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    bias: Option<&[f32]>,
    make_decoder: impl Fn() -> D + Sync,
) -> Result<Vec<f32>>
where
    D: FnMut(usize, &mut [f32]) -> Result<()>,
{
    let invalid = |reason: &str| {
        EpError::KernelFailed(format!(
            "{context} row-decoded projection [{rows},{in_features}] -> [{rows},{out_features}]: {reason}; supply addressable dimensions and matching input/bias storage"
        ))
    };
    if rows.checked_mul(in_features) != Some(input.len())
        || bias.is_some_and(|bias| bias.len() != out_features)
    {
        return Err(invalid(
            "input or bias element count does not match geometry",
        ));
    }
    let elements = rows
        .checked_mul(out_features)
        .ok_or_else(|| invalid("output element count overflow"))?;
    if elements == 0 {
        return Ok(Vec::new());
    }
    let columns_per_shard = out_features.div_ceil(rayon::current_num_threads().max(1));
    let mut column_major = zeroed_f32(context, "column-major output", elements)?;
    column_major
        .par_chunks_mut(columns_per_shard * rows)
        .enumerate()
        .try_for_each(|(shard, columns)| -> Result<()> {
            let mut weight_row = zeroed_f32(context, "decoded weight row", in_features)?;
            let mut decode = make_decoder();
            for (column, output) in columns.chunks_exact_mut(rows).enumerate() {
                let feature = shard * columns_per_shard + column;
                decode(feature, &mut weight_row)?;
                super::matmul::gemm_bt_serial(input, &weight_row, output, rows, in_features, 1)?;
                if let Some(bias) = bias {
                    for value in output {
                        *value += bias[feature];
                    }
                }
            }
            Ok(())
        })?;
    let mut output = zeroed_f32(context, "row-major output", elements)?;
    for (row, target) in output.chunks_exact_mut(out_features).enumerate() {
        for (column, value) in target.iter_mut().enumerate() {
            *value = column_major[column * rows + row];
        }
    }
    Ok(output)
}

fn zeroed_f32(context: &str, buffer: &str, elements: usize) -> Result<Vec<f32>> {
    let bytes = elements.checked_mul(std::mem::size_of::<f32>()).ok_or_else(|| EpError::KernelFailed(format!(
        "{context} {buffer} f32 byte count overflows for {elements} elements; lower the projection or batch dimensions"
    )))?;
    let mut values = Vec::new();
    values.try_reserve_exact(elements).map_err(|source| EpError::Memory(onnx_runtime_memory_api::MemoryError::AllocationFailed {
        tier: "host", requested: bytes as u64,
        reason: format!("{context} could not allocate {buffer}: {source}; lower concurrent projection demand or free Host memory"),
    }))?;
    values.resize(elements, 0.0);
    Ok(values)
}
