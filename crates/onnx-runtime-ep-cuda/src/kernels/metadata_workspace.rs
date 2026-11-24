//! Checked rank-sized metadata borrowed from the root caller lane.
//!
//! This helper owns only a layout. It never allocates, retains or releases device
//! backing. Synchronous uploads preserve the existing host-data lifetime contract.
use std::alloc::Layout;

use cudarc::driver::sys::CUdeviceptr;
use onnx_runtime_ep_api::{
    EpError, OutputMetadata, Result, WorkspaceLayout, WorkspaceLifetime, WorkspaceRequirement,
    WorkspaceView,
};

use crate::runtime::{CudaRuntime, cuptr};

pub(super) struct MetadataWorkspace {
    op: &'static str,
    layout: WorkspaceLayout<1>,
}

impl MetadataWorkspace {
    pub(super) fn for_rank(
        op: &'static str,
        rank: usize,
        words_per_axis: usize,
        output: &OutputMetadata<'_>,
    ) -> Result<Self> {
        let empty = match output {
            OutputMetadata::Absent => {
                return Err(EpError::KernelFailed(format!(
                    "cuda_ep {op}: required output 0 is absent; provide the output metadata"
                )));
            }
            OutputMetadata::Tensor { shape, .. } => shape.contains(&0),
            OutputMetadata::Unresolved { .. } => false,
        };
        let bytes = if empty {
            0
        } else {
            rank.checked_mul(words_per_axis).and_then(|words| words.checked_mul(8))
                .ok_or_else(|| EpError::KernelFailed(format!(
                    "cuda_ep {op}: rank {rank} with {words_per_axis} metadata words per axis overflows; reduce the tensor rank")))?
        };
        let region = Layout::from_size_align(bytes, std::mem::align_of::<u64>())
            .map_err(|_| EpError::KernelFailed(format!(
                "cuda_ep {op}: {bytes} metadata bytes exceed addressable capacity; reduce the tensor rank")))?;
        Ok(Self {
            op,
            layout: WorkspaceLayout::new([region])?,
        })
    }

    pub(super) fn requirement(&self) -> WorkspaceRequirement {
        if self.layout.bytes() == 0 {
            WorkspaceRequirement::NONE
        } else {
            WorkspaceRequirement {
                bytes: self.layout.bytes() as u64,
                alignment: self.layout.alignment(),
                lifetime: WorkspaceLifetime::SessionPersistent,
            }
        }
    }

    fn bind(&self, workspace: Option<WorkspaceView>, words: usize) -> Result<WorkspaceView> {
        if words.checked_mul(8) != Some(self.layout.bytes()) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {}: payload has {words} metadata words but the prepared layout contains {} bytes; prepare the complete current-rank workspace",
                self.op,
                self.layout.bytes()
            )));
        }
        self.layout
            .bind(workspace)
            .map(|[region]| region)
            .map_err(|source| EpError::Backend {
                operation: self.op,
                source: Box::new(source),
            })
    }

    pub(super) fn upload(
        &self,
        runtime: &CudaRuntime,
        workspace: Option<WorkspaceView>,
        values: &[u64],
    ) -> Result<CUdeviceptr> {
        let region = self.bind(workspace, values.len())?;
        if values.is_empty() {
            return Ok(0);
        }
        // SAFETY: u64 contains no padding; the borrowed slice stays live until
        // synchronous htod completes. The complete region was validated above.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), region.bytes()) };
        let pointer = cuptr(region.ptr().0.cast_const());
        unsafe { runtime.htod(bytes, pointer) }?;
        Ok(pointer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtrMut;
    use onnx_runtime_ir::DataType;
    #[test]
    fn rank_metadata_sizes_are_one_persistent_requirement() {
        for (op, rank, words, bytes) in [
            ("Mod", 3, 3, 72),
            ("LpPool", 2, 6, 96),
            ("NonZero", 4, 1, 32),
        ] {
            let plan = MetadataWorkspace::for_rank(
                op,
                rank,
                words,
                &OutputMetadata::Unresolved {
                    dtype: DataType::Undefined,
                },
            )
            .unwrap();
            assert_eq!(plan.requirement().bytes, bytes);
            assert_eq!(plan.requirement().alignment, 8);
            assert_eq!(
                plan.requirement().lifetime,
                WorkspaceLifetime::SessionPersistent
            );
        }
    }
    #[test]
    fn scalar_is_not_empty_but_needs_no_rank_metadata_and_empty_output_skips_overflow() {
        let scalar = OutputMetadata::tensor(DataType::Float32, &[]);
        let plan = MetadataWorkspace::for_rank("Mod", 0, 3, &scalar).unwrap();
        assert_eq!(plan.requirement(), WorkspaceRequirement::NONE);
        plan.bind(None, 0).unwrap();
        let empty = MetadataWorkspace::for_rank(
            "LpPool",
            usize::MAX,
            6,
            &OutputMetadata::tensor(DataType::Float32, &[1, 0, 3]),
        )
        .unwrap();
        assert_eq!(empty.requirement(), WorkspaceRequirement::NONE);
        assert!(MetadataWorkspace::for_rank("Mod", usize::MAX, 3, &scalar).is_err());
        assert!(
            MetadataWorkspace::for_rank("Mod", 1, 3, &OutputMetadata::Absent)
                .err()
                .unwrap()
                .to_string()
                .contains("required output 0")
        );
    }
    #[test]
    fn complete_binding_rejects_missing_short_unaligned_and_stale_payload() {
        let plan = MetadataWorkspace::for_rank(
            "Mod",
            2,
            3,
            &OutputMetadata::tensor(DataType::Int32, &[2, 3]),
        )
        .unwrap();
        let view = |address, bytes| WorkspaceView::new(DevicePtrMut(address as *mut _), bytes);
        for backing in [
            None,
            Some(view(0x10000usize, 47)),
            Some(view(0x10001usize, 48)),
        ] {
            let error = plan.bind(backing, 6).unwrap_err().to_string();
            assert!(error.contains("Mod"), "{error}");
            assert!(error.contains("prepare and pass"), "{error}");
        }
        assert_eq!(
            plan.bind(Some(view(0x10000usize, 64)), 6).unwrap().bytes(),
            48
        );
        assert!(
            plan.bind(Some(view(0x10000usize, 64)), 5)
                .unwrap_err()
                .to_string()
                .contains("payload has 5")
        );
    }
}
