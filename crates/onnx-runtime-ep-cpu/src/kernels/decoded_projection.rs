//! Row-decoded f32 projections shared by compressed weight representations.

use onnx_runtime_ep_api::{EpError, Result, WorkspaceLayout, WorkspaceView};
use rayon::prelude::*;
use std::mem::MaybeUninit;

/// One dispatch's exclusive f32 scratch lane. Binding does not initialize or
/// touch backing: dense-cache hits pay no streamed-scratch clearing cost.
/// Only the first streamed projection initializes the entire admitted lane;
/// subsequent projections borrow initialized values without re-clearing them.
pub(super) struct ProjectionScratch<'a> {
    storage: &'a mut [MaybeUninit<f32>],
    initialized: bool,
}

impl<'a> ProjectionScratch<'a> {
    /// # Safety
    /// Nonempty workspace must be valid, Host-accessible, writable storage,
    /// exclusively lent for `'a`. It must outlive this dispatch and every
    /// short-lived initialized slice. No references may escape the dispatch.
    pub(super) unsafe fn bind(
        operation: &'static str,
        bytes: usize,
        workspace: Option<WorkspaceView>,
    ) -> Result<Self> {
        if !bytes.is_multiple_of(std::mem::size_of::<f32>()) {
            return Err(EpError::KernelFailed(format!(
                "{operation}: {bytes} scratch bytes do not form complete f32 values; use the projection's prepared layout"
            )));
        }
        let elements = bytes / std::mem::size_of::<f32>();
        let region = std::alloc::Layout::array::<f32>(elements).map_err(|_| {
            EpError::KernelFailed(format!("{operation}: {bytes} scratch bytes exceed addressable capacity; reduce batch or projection dimensions"))
        })?;
        let [region] = WorkspaceLayout::new([region])?
            .bind(workspace)
            .map_err(|source| EpError::Backend {
                operation,
                source: Box::new(source),
            })?;
        let storage = if elements == 0 {
            &mut []
        } else {
            // SAFETY: the checked layout proves non-null/aligned capacity;
            // the caller guarantees Host accessibility, exclusivity and lifetime.
            unsafe { std::slice::from_raw_parts_mut(region.ptr().0.cast(), elements) }
        };
        Ok(Self {
            storage,
            initialized: false,
        })
    }

    pub(super) fn initialized(&mut self) -> &mut [f32] {
        if !self.initialized {
            // SAFETY: binding exclusively lends every element. All-zero bytes
            // initialize valid f32 values before any f32 references are made.
            unsafe { self.storage.as_mut_ptr().write_bytes(0, self.storage.len()) };
            self.initialized = true;
        }
        // SAFETY: first use initialized the entire lane; the only mutable views
        // subsequently exposed are initialized f32 values with this short borrow.
        unsafe {
            std::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast(), self.storage.len())
        }
    }
}

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

    fn workspace_elements(self, rows: usize, k: usize, n: usize) -> Result<[usize; 2]> {
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
        Ok([columns, decoded])
    }

    pub(super) fn layout(self, rows: usize, k: usize, n: usize) -> Result<WorkspaceLayout<2>> {
        let [columns, decoded] = self.workspace_elements(rows, k, n)?;
        let invalid = || {
            EpError::KernelFailed(format!(
                "decoded projection workspace overflows for rows={rows}, K={k}, N={n}; reduce batch or projection dimensions"
            ))
        };
        WorkspaceLayout::new([
            std::alloc::Layout::array::<f32>(columns).map_err(|_| invalid())?,
            std::alloc::Layout::array::<f32>(decoded).map_err(|_| invalid())?,
        ])
    }

    /// Partition only this projection's prefix, leaving the rest of a larger
    /// shared lane untouched. Safe slice borrowing enforces non-overlap.
    pub(super) fn split_scratch(
        self,
        rows: usize,
        k: usize,
        n: usize,
        scratch: &mut [f32],
    ) -> Result<(&mut [f32], &mut [f32])> {
        let [columns, decoded] = self.workspace_elements(rows, k, n)?;
        let elements = columns.checked_add(decoded).ok_or_else(|| EpError::KernelFailed(format!(
            "decoded projection workspace element count overflows for rows={rows}, K={k}, N={n}; reduce projection dimensions"
        )))?;
        let available = scratch.len();
        let scratch = scratch.get_mut(..elements).ok_or_else(|| EpError::KernelFailed(format!(
            "decoded projection needs {elements} initialized scratch f32 values for rows={rows}, K={k}, N={n}, received {available}; prepare and pass the matching projection workspace"
        )))?;
        Ok(scratch.split_at_mut(columns))
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

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtrMut;
    use std::num::NonZeroUsize;

    #[test]
    fn projection_scratch_is_lazy_once_per_dispatch_and_only_borrows_the_needed_prefix() {
        let mut backing = [MaybeUninit::new(37.0f32); 32];
        let view = WorkspaceView::new(
            DevicePtrMut(backing.as_mut_ptr().cast()),
            std::mem::size_of_val(&backing),
        );
        {
            // SAFETY: backing is exclusive Host storage, aligned and live until
            // the owner and all short projection borrows below are dropped.
            let mut scratch =
                unsafe { ProjectionScratch::bind("test projection binding", 16 * 4, Some(view)) }
                    .unwrap();
            assert!(!scratch.initialized);
            assert_eq!(scratch.storage.len(), 16);
            // SAFETY: the test populated every MaybeUninit with 37.0.
            assert!(
                scratch
                    .storage
                    .iter()
                    .all(|value| unsafe { value.assume_init() } == 37.0),
                "binding must not clear dense-hit scratch"
            );
            assert!(
                scratch
                    .initialized()
                    .iter()
                    .all(|&value| value.to_bits() == 0)
            );
            scratch.initialized().fill(19.0);
            assert!(
                scratch.initialized().iter().all(|&value| value == 19.0),
                "a later FC must reuse initialized storage without resetting it"
            );
        }
        // SAFETY: all backing values were initialized by the test or owner.
        assert!(
            backing[16..]
                .iter()
                .all(|value| unsafe { value.assume_init() } == 37.0),
            "a larger pool slot's unused suffix must remain untouched"
        );
        {
            // A new dispatch must not inherit an initialization proof from an
            // unrelated kernel that may have used the pooled bytes in between.
            let mut scratch =
                unsafe { ProjectionScratch::bind("test projection binding", 16 * 4, Some(view)) }
                    .unwrap();
            assert!(
                scratch
                    .initialized()
                    .iter()
                    .all(|&value| value.to_bits() == 0)
            );
        }
    }

    #[test]
    fn projection_scratch_rejects_invalid_layouts_and_accepts_empty_without_backing() {
        let mut backing = [0u64; 4];
        let ptr = backing.as_mut_ptr().cast();
        for (bytes, workspace) in [
            (3, None),
            (16, None),
            (16, Some(WorkspaceView::new(DevicePtrMut(ptr), 8))),
            (
                16,
                Some(WorkspaceView::new(
                    DevicePtrMut(ptr.cast::<u8>().wrapping_add(1).cast()),
                    16,
                )),
            ),
            (
                16,
                Some(WorkspaceView::new(
                    DevicePtrMut((usize::MAX - 7) as *mut _),
                    16,
                )),
            ),
        ] {
            // SAFETY: invalid views are rejected before dereferencing them.
            let error =
                unsafe { ProjectionScratch::bind("test projection binding", bytes, workspace) }
                    .err()
                    .expect("invalid workspace must be rejected");
            assert!(
                error.to_string().contains("test projection binding"),
                "{error}"
            );
        }
        // SAFETY: an empty owner dereferences no backing.
        let mut scratch = unsafe { ProjectionScratch::bind("empty", 0, None) }.unwrap();
        assert!(scratch.initialized().is_empty());
    }

    #[test]
    fn projection_scratch_partitions_each_projection_prefix_without_aliasing() {
        for shards in [1, 4] {
            for tile in [1, 4] {
                let plan = DecodedProjectionPlan::new(
                    NonZeroUsize::new(shards).unwrap(),
                    NonZeroUsize::new(tile).unwrap(),
                );
                for rows in [0, 1, 3] {
                    for k in [0, 32] {
                        for n in [0, 1, 17] {
                            let layout = plan.layout(rows, k, n).unwrap();
                            let elements = layout.bytes() / 4;
                            let mut storage = vec![13.0f32; elements + 7];
                            let (columns, decoded) =
                                plan.split_scratch(rows, k, n, &mut storage).unwrap();
                            assert_eq!(columns.len(), if rows > 1 && n > 1 { rows * n } else { 0 });
                            assert_eq!(columns.len() + decoded.len(), elements);
                            columns.fill(2.0);
                            decoded.fill(3.0);
                            assert!(columns.iter().all(|&value| value == 2.0));
                            assert!(storage[elements..].iter().all(|&value| value == 13.0));
                            if elements > 0 {
                                let error = plan
                                    .split_scratch(rows, k, n, &mut storage[..elements - 1])
                                    .unwrap_err();
                                assert!(
                                    error.to_string().contains("initialized scratch f32 values"),
                                    "{error}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
