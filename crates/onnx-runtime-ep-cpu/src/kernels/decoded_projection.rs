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
        // SAFETY: the caller lends the complete Host workspace exclusively.
        let [scratch] =
            unsafe { Self::bind_layout(operation, WorkspaceLayout::new([region])?, workspace)? };
        Ok(scratch)
    }

    /// Bind one checked composite layout, validating f32 regions before any
    /// typed references are formed. No region is initialized by binding.
    ///
    /// # Safety
    /// Nonempty backing must be live, Host-accessible and exclusively lent for
    /// `'a`, including every region and all short borrows made from its owners.
    pub(super) unsafe fn bind_layout<const N: usize>(
        operation: &'static str,
        layout: WorkspaceLayout<N>,
        workspace: Option<WorkspaceView>,
    ) -> Result<[Self; N]> {
        let regions = layout.bind(workspace).map_err(|source| EpError::Backend {
            operation,
            source: Box::new(source),
        })?;
        for (index, region) in regions.iter().enumerate() {
            if !region.bytes().is_multiple_of(std::mem::size_of::<f32>())
                || (region.bytes() != 0
                    && !(region.ptr().0 as usize).is_multiple_of(std::mem::align_of::<f32>()))
            {
                return Err(EpError::KernelFailed(format!(
                    "{operation}: region {index} has {} bytes at {:#x}, which is not aligned complete f32 storage; declare f32 workspace layouts",
                    region.bytes(),
                    region.ptr().0 as usize,
                )));
            }
        }
        Ok(regions.map(|region| {
            let storage = if region.bytes() == 0 {
                &mut []
            } else {
                // SAFETY: complete layout validation proves capacity, address
                // range and disjointness; the checks above prove f32 geometry.
                // The caller guarantees Host accessibility and exclusive lifetime.
                unsafe {
                    std::slice::from_raw_parts_mut(
                        region.ptr().0.cast(),
                        region.bytes() / std::mem::size_of::<f32>(),
                    )
                }
            };
            Self {
                storage,
                initialized: false,
            }
        }))
    }

    /// Gather only the requested rows, initializing every returned value by
    /// copying it. Unused capacity is untouched; no whole-lane clear is needed.
    pub(super) fn gather_rows(
        &mut self,
        input: &[f32],
        width: usize,
        rows: impl Iterator<Item = usize>,
    ) -> Result<&mut [f32]> {
        if width == 0 {
            return Ok(&mut []);
        }
        let mut written = 0usize;
        for row in rows {
            let start = row.checked_mul(width).ok_or_else(|| {
                EpError::KernelFailed(format!(
                    "projection input row {row} times width {width} overflows addressable storage"
                ))
            })?;
            let source = input.get(start..).and_then(|rest| rest.get(..width)).ok_or_else(|| EpError::KernelFailed(
                format!("projection input row {row} of width {width} exceeds {} input values; validate routed row indices", input.len())
            ))?;
            let end = written.checked_add(width).ok_or_else(|| {
                EpError::KernelFailed(
                    "projection input gather size overflow; reduce routed batch dimensions".into(),
                )
            })?;
            let capacity = self.storage.len();
            let target = self.storage.get_mut(written..end).ok_or_else(|| EpError::KernelFailed(
                format!("projection input gather needs {end} values, workspace has {capacity}; prepare the complete grouped-input bound")
            ))?;
            for (target, &value) in target.iter_mut().zip(source) {
                target.write(value);
            }
            written = end;
        }
        // SAFETY: each complete target row was initialized above, and written
        // counts actual copied values, without trusting an iterator size hint.
        // The returned prefix is exclusive and cannot outlive this short borrow.
        Ok(unsafe { std::slice::from_raw_parts_mut(self.storage.as_mut_ptr().cast(), written) })
    }

    /// Gather in caller-selected whole-row tasks, preserving its threading
    /// policy. Each returned element is written once; unused capacity is untouched.
    pub(super) fn gather_rows_parallel(
        &mut self,
        input: &[f32],
        width: usize,
        rows: &[u32],
        rows_per_task: std::num::NonZeroUsize,
    ) -> Result<&mut [f32]> {
        if width == 0 || rows.is_empty() {
            return Ok(&mut []);
        }
        let count = rows.len().checked_mul(width).ok_or_else(|| {
            EpError::KernelFailed(
                "projection parallel gather size overflow; reduce routed batch dimensions".into(),
            )
        })?;
        let capacity = self.storage.len();
        let target = self.storage.get_mut(..count).ok_or_else(|| EpError::KernelFailed(
            format!("projection parallel gather needs {count} values, workspace has {capacity}; prepare the complete grouped-input bound")
        ))?;
        // count above proves this clamped chunk multiplication addressable.
        let chunk_rows = rows_per_task.get().min(rows.len());
        target.par_chunks_mut(chunk_rows * width).zip(rows.par_chunks(chunk_rows)).try_for_each(|(target, rows)| -> Result<()> {
            for (target, &row) in target.chunks_mut(width).zip(rows) {
                let start = (row as usize).checked_mul(width).ok_or_else(|| EpError::KernelFailed(
                    format!("projection input row {row} times width {width} overflows addressable storage")
                ))?;
                let source = input.get(start..).and_then(|rest| rest.get(..width)).ok_or_else(|| EpError::KernelFailed(
                    format!("projection input row {row} of width {width} exceeds {} input values; validate routed row indices", input.len())
                ))?;
                for (target, &value) in target.iter_mut().zip(source) { target.write(value); }
            }
            Ok(())
        })?;
        // SAFETY: disjoint complete rows were initialized above. A failed
        // gather never creates an f32 slice over partially initialized backing.
        Ok(unsafe { std::slice::from_raw_parts_mut(target.as_mut_ptr().cast(), count) })
    }

    /// Initialize only the requested prefix for a fresh projection destination.
    /// A partial prefix does not establish initialization of the whole bank.
    pub(super) fn zeroed_prefix(&mut self, elements: usize) -> Result<&mut [f32]> {
        let capacity = self.storage.len();
        let target = self.storage.get_mut(..elements).ok_or_else(|| EpError::KernelFailed(
            format!("projection destination needs {elements} values, workspace has {capacity}; prepare the complete projection bound")
        ))?;
        // SAFETY: the short borrow is exclusive; zero bytes initialize f32.
        unsafe {
            target.as_mut_ptr().write_bytes(0, elements);
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(target.as_mut_ptr().cast(), elements) })
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
/// The initialized destination is caller-owned staging: geometry failures leave
/// it untouched, while a later decoder/dispatch failure may modify it. Callers
/// publish actual tensor outputs only after successful computation.
#[allow(clippy::too_many_arguments)]
pub(super) fn linear_decoded_grouped_into<const FEATURE_TILE: usize, D>(
    context: &str,
    input: &[f32],
    rows: usize,
    in_features: usize,
    out_features: usize,
    bias: Option<&[f32]>,
    output: &mut [f32],
    make_decoder: impl Fn() -> D + Sync,
    project: impl Fn(&[f32], &[f32], &mut [f32], usize, usize, usize) -> Result<()> + Sync,
    dispatch: impl FnOnce(&mut [f32], usize, &DecodedShard<'_>) -> Result<()>,
) -> Result<()>
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
            "{context} row-decoded projection [{rows},{in_features}] -> [{rows},{out_features}]: {reason}; supply addressable dimensions and matching input/output/bias storage"
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
    if output.len() != elements {
        return Err(invalid(&format!(
            "destination has {} elements, expected {elements}",
            output.len()
        )));
    }
    if elements == 0 {
        return Ok(());
    }
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
    // A unit axis makes column-major and row-major storage identical.
    // The caller's staging is already initialized and receives complete writes.
    if rows == 1 || out_features == 1 {
        return dispatch(output, rows, &compute);
    }
    let mut column_major = zeroed_f32(context, "column-major output", elements)?;
    dispatch(&mut column_major, rows, &compute)?;
    for (row, target) in output.chunks_exact_mut(out_features).enumerate() {
        for (column, value) in target.iter_mut().enumerate() {
            *value = column_major[column * rows + row];
        }
    }
    Ok(())
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

    /// Serial projections reuse one complete layout. Keep each projection's
    /// internal offsets intact rather than taking separate region maxima.
    pub(super) fn maximum_layout(
        self,
        rows: usize,
        projections: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<std::alloc::Layout> {
        let mut bytes = 0;
        let mut alignment = 1;
        for (k, n) in projections {
            let layout = self.layout(rows, k, n)?;
            bytes = bytes.max(layout.bytes());
            alignment = alignment.max(layout.alignment());
        }
        std::alloc::Layout::from_size_align(bytes, alignment).map_err(|_| {
            EpError::KernelFailed("serial decoded projection workspace exceeds addressable capacity; reduce batch or projection dimensions".into())
        })
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
    fn serial_projection_layout_uses_maximum_complete_phase() {
        let plan = DecodedProjectionPlan::new(
            NonZeroUsize::new(2).unwrap(),
            NonZeroUsize::new(1).unwrap(),
        );
        let fc1 = plan.layout(3, 16, 32).unwrap();
        let fc2 = plan.layout(3, 32, 16).unwrap();
        assert_eq!(fc1.bytes(), (96 + 32) * 4);
        assert_eq!(fc2.bytes(), (48 + 64) * 4);
        let maximum = plan.maximum_layout(3, [(16, 32), (32, 16)]).unwrap();
        assert_eq!(maximum.size(), 128 * 4);
        assert_eq!(maximum.align(), 4);
        assert_eq!(plan.maximum_layout(3, []).unwrap().size(), 0);
        assert_eq!(
            plan.maximum_layout(0, [(16, 32), (32, 16)]).unwrap().size(),
            0
        );
    }

    #[test]
    fn composite_scratch_binds_disjoint_f32_regions_and_rejects_untyped_geometry() {
        let mut backing = [MaybeUninit::new(29.0f32); 16];
        let view = WorkspaceView::new(DevicePtrMut(backing.as_mut_ptr().cast()), 16 * 4);
        let layout = WorkspaceLayout::new([
            std::alloc::Layout::array::<f32>(4).unwrap(),
            std::alloc::Layout::array::<f32>(3).unwrap(),
        ])
        .unwrap();
        {
            // SAFETY: live aligned Host backing is exclusively lent to both regions.
            let [mut projection, mut gathered] =
                unsafe { ProjectionScratch::bind_layout("composite test", layout, Some(view)) }
                    .unwrap();
            assert_eq!(
                gathered
                    .gather_rows(&[1., 2., 3.], 3, [0].into_iter())
                    .unwrap(),
                &[1., 2., 3.]
            );
            projection.initialized().fill(7.);
            assert_eq!(
                gathered.initialized(),
                &[0.; 3],
                "a prefix copy does not initialize the entire owner"
            );
            assert_eq!(projection.initialized(), &[7.; 4]);
        }
        // SAFETY: this fixture initialized the entire backing before binding.
        assert!(
            backing[7..]
                .iter()
                .all(|x| unsafe { x.assume_init() } == 29.)
        );
        let odd = WorkspaceLayout::new([std::alloc::Layout::array::<u8>(3).unwrap()]).unwrap();
        // SAFETY: backing accessibility/lifetime are valid; f32 geometry is checked by bind_layout.
        let error = unsafe { ProjectionScratch::bind_layout("odd region", odd, Some(view)) }
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("region 0") && error.contains("complete f32"),
            "{error}"
        );
        let byte_layout =
            WorkspaceLayout::new([std::alloc::Layout::array::<u8>(4).unwrap()]).unwrap();
        let misaligned = WorkspaceView::new(
            DevicePtrMut(view.ptr().0.cast::<u8>().wrapping_add(1).cast()),
            4,
        );
        // SAFETY: the live byte region is addressable; bind_layout must reject its f32 alignment.
        let error = unsafe {
            ProjectionScratch::bind_layout("misaligned region", byte_layout, Some(misaligned))
        }
        .err()
        .unwrap()
        .to_string();
        assert!(
            error.contains("region 0") && error.contains("aligned"),
            "{error}"
        );
    }

    #[test]
    fn gathered_projection_input_initializes_only_actual_rows_and_reuses_capacity() {
        let mut backing = [MaybeUninit::new(29.0f32); 16];
        let view = WorkspaceView::new(DevicePtrMut(backing.as_mut_ptr().cast()), 16 * 4);
        // SAFETY: the test exclusively lends live aligned Host storage.
        let mut scratch =
            unsafe { ProjectionScratch::bind("gather test", 16 * 4, Some(view)) }.unwrap();
        let input = [1., 2., 3., 4., 5., 6.];
        assert_eq!(
            scratch
                .gather_rows(&input, 2, [2, 0, 2].into_iter())
                .unwrap(),
            &[5., 6., 1., 2., 5., 6.]
        );
        assert!(
            !scratch.initialized,
            "a valid prefix is not a whole-lane initialization proof"
        );
        assert_eq!(
            scratch.gather_rows(&input, 2, [1].into_iter()).unwrap(),
            &[3., 4.]
        );
        // SAFETY: the test initially filled every value; gathering only copies f32.
        let values: Vec<_> = scratch
            .storage
            .iter()
            .map(|x| unsafe { x.assume_init() })
            .collect();
        assert_eq!(&values[2..6], &[1., 2., 5., 6.]);
        assert!(values[6..].iter().all(|&x| x == 29.));
        assert!(
            scratch
                .gather_rows(&input, 2, [].into_iter())
                .unwrap()
                .is_empty()
        );
        assert!(
            scratch
                .gather_rows(&input, 0, [0].into_iter())
                .unwrap()
                .is_empty()
        );
        assert!(
            scratch
                .gather_rows(&input, 2, [3].into_iter())
                .unwrap_err()
                .to_string()
                .contains("row 3")
        );
        assert!(
            scratch
                .gather_rows(&input, 2, [usize::MAX].into_iter())
                .unwrap_err()
                .to_string()
                .contains("overflows")
        );
        assert!(
            scratch
                .gather_rows(&input, 2, [0; 9].into_iter())
                .unwrap_err()
                .to_string()
                .contains("workspace has 16")
        );
        // A partially written failure never grants a whole-lane proof. The
        // ordinary initialized path must still establish every remaining value.
        assert!(!scratch.initialized);
        assert!(scratch.initialized().iter().all(|&x| x == 0.));
    }

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
