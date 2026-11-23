//! Row-decoded f32 projections shared by compressed weight representations.

use onnx_runtime_ep_api::{EpError, Result};
use rayon::prelude::*;

/// Decode a bounded tile of weight rows once and reuse it across routed tokens.
/// FEATURE_TILE is a compute tile, not a model dimension. Each worker owns at
/// most FEATURE_TILE decoded rows and one decoder state, reused across tiles.
/// The serial projection callback overwrites a complete column-major tile.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_decoded_grouped<const FEATURE_TILE: usize, D>(
    context: &str,
    input: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    bias: Option<&[f32]>,
    make_decoder: impl Fn() -> D + Sync,
    project: impl Fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()> + Sync,
    dispatch: impl FnOnce(&mut [f32], usize, &DecodedShard<'_>) -> Result<()>,
) -> Result<Vec<f32>>
where
    D: FnMut(usize, &mut [f32]) -> Result<()>,
{
    const {
        assert!(
            FEATURE_TILE > 0,
            "decoded projection tile must contain an output"
        );
    }
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
    let mut column_major = zeroed_f32(context, "column-major output", elements)?;
    let compute = |column_start: usize, columns: &mut [f32]| -> Result<()> {
        if !columns.len().is_multiple_of(rows) {
            return Err(invalid(
                "decoder shard must contain complete output columns",
            ));
        }
        let tile_features = FEATURE_TILE.min(columns.len() / rows);
        if tile_features == 0 {
            return Ok(());
        }
        let decoded_elements = tile_features
            .checked_mul(in_features)
            .ok_or_else(|| invalid("decoded weight tile element count overflow"))?;
        let mut weight_rows = zeroed_f32(context, "decoded weight tile", decoded_elements)?;
        let mut decode = make_decoder();
        for (tile, outputs) in columns.chunks_mut(rows * tile_features).enumerate() {
            let count = outputs.len() / rows;
            let first_feature = column_start + tile * tile_features;
            for column in 0..count {
                decode(
                    first_feature + column,
                    &mut weight_rows[column * in_features..(column + 1) * in_features],
                )?;
            }
            project(
                input,
                &weight_rows[..count * in_features],
                outputs,
                rows,
                in_features,
                count,
            )?;
            if let Some(bias) = bias {
                for (column, output) in outputs.chunks_exact_mut(rows).enumerate() {
                    for value in output {
                        *value += bias[first_feature + column];
                    }
                }
            }
        }
        Ok(())
    };
    dispatch(&mut column_major, rows, &compute)?;
    // A unit axis makes column-major and row-major storage identical.
    if rows == 1 || out_features == 1 {
        return Ok(column_major);
    }
    let mut output = zeroed_f32(context, "row-major output", elements)?;
    for (row, target) in output.chunks_exact_mut(out_features).enumerate() {
        for (column, value) in target.iter_mut().enumerate() {
            *value = column_major[column * rows + row];
        }
    }
    Ok(output)
}

/// A contiguous set of complete decoded output columns. `start` counts
/// features, not f32 elements. Callbacks must not enter a nested scheduler.
pub(super) type DecodedShard<'a> = dyn Fn(usize, &mut [f32]) -> Result<()> + Sync + 'a;

/// Default scheduling on the caller's already selected Rayon pool.
pub(super) fn rayon_decoded_columns(
    columns: &mut [f32],
    rows: usize,
    compute: &DecodedShard<'_>,
) -> Result<()> {
    if columns.is_empty() {
        return Ok(());
    }
    if rows == 0 || !columns.len().is_multiple_of(rows) {
        return Err(EpError::KernelFailed("decoded projection dispatch requires complete nonempty columns; supply a positive token count dividing the output buffer".into()));
    }
    let features = columns.len() / rows;
    let per_shard = features.div_ceil(rayon::current_num_threads().max(1));
    columns
        .par_chunks_mut(per_shard * rows)
        .enumerate()
        .try_for_each(|(shard, output)| compute(shard * per_shard, output))
}

pub(super) fn zeroed_f32(context: &str, buffer: &str, elements: usize) -> Result<Vec<f32>> {
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
