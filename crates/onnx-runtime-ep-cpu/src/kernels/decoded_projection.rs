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
        project_shard::<FEATURE_TILE, _>(
            input,
            rows,
            in_features,
            bias,
            column_start,
            columns,
            &mut weight_rows,
            &make_decoder,
            &project,
        )?;
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

#[allow(clippy::too_many_arguments)]
fn project_shard<const FEATURE_TILE: usize, D>(
    input: &[f32],
    rows: usize,
    in_features: usize,
    bias: Option<&[f32]>,
    column_start: usize,
    columns: &mut [f32],
    weight_rows: &mut [f32],
    make_decoder: &(impl Fn() -> D + Sync),
    project: &(impl Fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()> + Sync),
) -> Result<()>
where
    D: FnMut(usize, &mut [f32]) -> Result<()>,
{
    let tile_features = FEATURE_TILE.min(columns.len() / rows);
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
}

/// A fixed logical concurrency bound. Tasks carry their exclusive scratch slice;
/// the Rayon worker that happens to execute a task is irrelevant to ownership.
#[derive(Clone, Copy, Debug)]
pub(super) struct DecodedProjectionPlan {
    shards: std::num::NonZeroUsize,
    tile: std::num::NonZeroUsize,
}

impl DecodedProjectionPlan {
    pub(super) fn new(shards: std::num::NonZeroUsize, tile: std::num::NonZeroUsize) -> Self {
        Self { shards, tile }
    }

    pub(super) fn for_current_pool(tile: std::num::NonZeroUsize) -> Self {
        Self::new(
            std::num::NonZeroUsize::new(rayon::current_num_threads().max(1)).unwrap(),
            tile,
        )
    }

    fn geometry(self, rows: usize, k: usize, n: usize) -> Result<(usize, usize, usize)> {
        if rows == 0 || n == 0 {
            return Ok((0, 0, 0));
        }
        let per_shard = n.div_ceil(self.shards.get());
        let shards = n.div_ceil(per_shard);
        let tile_elements = self.tile.get().min(per_shard).checked_mul(k)
            .ok_or_else(|| EpError::KernelFailed(format!("decoded projection tile overflows for K={k}, N={n}; reduce projection dimensions")))?;
        Ok((per_shard, shards, tile_elements))
    }

    pub(super) fn layout(
        self,
        rows: usize,
        k: usize,
        n: usize,
    ) -> Result<onnx_runtime_ep_api::WorkspaceLayout<2>> {
        let (_, shards, tile_elements) = self.geometry(rows, k, n)?;
        let invalid = || {
            EpError::KernelFailed(format!(
                "decoded projection workspace overflows for rows={rows}, K={k}, N={n}; reduce batch or projection dimensions"
            ))
        };
        let columns = if rows > 1 && n > 1 {
            rows.checked_mul(n).ok_or_else(invalid)?
        } else {
            0
        };
        // Last shard may be shorter; uniform slots permit allocation-free zip
        // scheduling and remain bounded by the stored logical shard plan.
        let decoded = shards.checked_mul(tile_elements).ok_or_else(invalid)?;
        onnx_runtime_ep_api::WorkspaceLayout::new([
            std::alloc::Layout::array::<f32>(columns).map_err(|_| invalid())?,
            std::alloc::Layout::array::<f32>(decoded).map_err(|_| invalid())?,
        ])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn project<const FEATURE_TILE: usize, D>(
        self,
        input: &[f32],
        rows: usize,
        k: usize,
        n: usize,
        bias: Option<&[f32]>,
        output: &mut [f32],
        column_major: &mut [f32],
        decoded: &mut [f32],
        make_decoder: impl Fn() -> D + Sync,
        project: impl Fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()> + Sync,
    ) -> Result<()>
    where
        D: FnMut(usize, &mut [f32]) -> Result<()>,
    {
        const {
            assert!(FEATURE_TILE > 0);
        }
        let (per_shard, shards, tile_elements) = self.geometry(rows, k, n)?;
        let layout = self.layout(rows, k, n)?;
        if self.tile.get() != FEATURE_TILE
            || rows.checked_mul(k) != Some(input.len())
            || rows.checked_mul(n) != Some(output.len())
            || bias.is_some_and(|bias| bias.len() != n)
            || column_major.len() != if rows > 1 && n > 1 { output.len() } else { 0 }
            || decoded.len() != shards * tile_elements
        {
            return Err(EpError::KernelFailed(format!(
                "decoded projection buffers or tile disagree with plan for rows={rows}, K={k}, N={n} (workspace {} bytes); bind the matching plan and complete input/output storage",
                layout.bytes()
            )));
        }
        if output.is_empty() {
            return Ok(());
        }
        let transpose = rows > 1 && n > 1;
        let columns = if transpose {
            &mut *column_major
        } else {
            &mut *output
        };
        if tile_elements == 0 {
            // K=0 needs no decoded storage, but still writes output and bias.
            columns
                .par_chunks_mut(per_shard * rows)
                .enumerate()
                .try_for_each(|(shard, values)| {
                    project_shard::<FEATURE_TILE, _>(
                        input,
                        rows,
                        k,
                        bias,
                        shard * per_shard,
                        values,
                        &mut [],
                        &make_decoder,
                        &project,
                    )
                })?;
        } else {
            columns
                .par_chunks_mut(per_shard * rows)
                .zip(decoded.par_chunks_mut(tile_elements))
                .enumerate()
                .try_for_each(|(shard, (values, weights))| {
                    project_shard::<FEATURE_TILE, _>(
                        input,
                        rows,
                        k,
                        bias,
                        shard * per_shard,
                        values,
                        weights,
                        &make_decoder,
                        &project,
                    )
                })?;
        }
        if transpose {
            for (row, target) in output.chunks_exact_mut(n).enumerate() {
                for (column, value) in target.iter_mut().enumerate() {
                    *value = column_major[column * rows + row];
                }
            }
        }
        Ok(())
    }
}
