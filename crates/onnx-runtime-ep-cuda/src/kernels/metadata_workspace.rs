//! Checked record-sized metadata borrowed from the root caller lane.
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

fn output_is_empty(op: &str, output: &OutputMetadata<'_>) -> Result<bool> {
    match output {
        OutputMetadata::Absent => Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: required output 0 is absent; provide the output metadata"
        ))),
        OutputMetadata::Tensor { shape, .. } => Ok(shape.contains(&0)),
        OutputMetadata::Unresolved { .. } => Ok(false),
    }
}

impl MetadataWorkspace {
    pub(super) fn for_records(
        op: &'static str,
        records: usize,
        words_per_record: usize,
        output: &OutputMetadata<'_>,
    ) -> Result<Self> {
        let empty = output_is_empty(op, output)?;
        let bytes = if empty {
            0
        } else {
            records.checked_mul(words_per_record).and_then(|words| words.checked_mul(8))
                .ok_or_else(|| EpError::KernelFailed(format!(
                    "cuda_ep {op}: {records} metadata records with {words_per_record} words per record overflow; reduce the tensor shape")))?
        };
        Self::from_bytes(op, bytes)
    }

    pub(super) fn for_bytes(
        op: &'static str,
        bytes: usize,
        output: &OutputMetadata<'_>,
    ) -> Result<Self> {
        Self::from_bytes(
            op,
            if output_is_empty(op, output)? {
                0
            } else {
                bytes
            },
        )
    }

    fn from_bytes(op: &'static str, bytes: usize) -> Result<Self> {
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

    pub(super) fn bind(&self, workspace: Option<WorkspaceView>) -> Result<BoundMetadataUpload> {
        let [region] = self
            .layout
            .bind(workspace)
            .map_err(|source| EpError::Backend {
                operation: self.op,
                source: Box::new(source),
            })?;
        Ok(BoundMetadataUpload {
            op: self.op,
            region,
        })
    }
}

/// Proof that the full current descriptor fits. This is only a borrowed view;
/// the caller remains responsible for ownership, completion and retirement.
#[derive(Debug)]
pub(super) struct BoundMetadataUpload {
    op: &'static str,
    region: WorkspaceView,
}

impl BoundMetadataUpload {
    fn validate_bytes(&self, bytes: usize) -> Result<()> {
        if bytes != self.region.bytes() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep {}: payload has {bytes} metadata bytes but the prepared layout contains {}; prepare the complete current-layout workspace",
                self.op,
                self.region.bytes()
            )));
        }
        Ok(())
    }
    pub(super) fn upload(self, runtime: &CudaRuntime, values: &[u64]) -> Result<CUdeviceptr> {
        let bytes = values.len().checked_mul(8).ok_or_else(|| EpError::KernelFailed(format!(
            "cuda_ep {}: metadata word payload exceeds addressable byte count; reduce the descriptor", self.op)))?;
        // SAFETY: u64 has no padding and the slice remains live through the
        // synchronous byte upload. upload_bytes validates the complete span.
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr().cast(), bytes) };
        self.upload_bytes(runtime, bytes)
    }
    pub(super) fn upload_bytes(self, runtime: &CudaRuntime, bytes: &[u8]) -> Result<CUdeviceptr> {
        self.validate_bytes(bytes.len())?;
        if bytes.is_empty() {
            return Ok(0);
        }
        let pointer = cuptr(self.region.ptr().0.cast_const());
        // SAFETY: the proof covers exactly bytes.len(); synchronous htod keeps
        // the borrowed host payload live through completion.
        unsafe { runtime.htod(bytes, pointer) }?;
        Ok(pointer)
    }
}

/// Dense byte spans and strides must fit pointer arithmetic before control
/// readback, offset generation or a device launch can consume their geometry.
pub(super) fn checked_tensor_bytes(
    op: &str,
    name: &str,
    dtype: onnx_runtime_ir::DataType,
    shape: &[usize],
) -> Result<usize> {
    let elements = if shape.contains(&0) {
        Some(0)
    } else {
        onnx_runtime_ir::checked_numel(shape)
    };
    let bytes = elements.and_then(|elements| dtype.checked_storage_bytes(elements))
        .filter(|&bytes| bytes <= isize::MAX as usize)
        .ok_or_else(|| EpError::KernelFailed(format!(
            "cuda_ep {op}: {name} shape {shape:?} with dtype {dtype:?} exceeds addressable storage; reduce the tensor shape")))?;
    Ok(bytes)
}

pub(super) fn checked_metadata_strides(op: &str, name: &str, shape: &[usize]) -> Result<Vec<i64>> {
    if shape.contains(&0) {
        return Ok(vec![0; shape.len()]);
    }
    checked_tensor_bytes(op, name, onnx_runtime_ir::DataType::Uint8, shape)?;
    Ok(onnx_runtime_ir::compute_contiguous_strides(shape))
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ep_api::DevicePtrMut;
    use onnx_runtime_ir::DataType;
    #[test]
    fn mixed_byte_payloads_keep_exact_tail_size_and_empty_output_skips_backing() {
        for bytes in [1, 65, 96] {
            let plan = MetadataWorkspace::for_bytes(
                "Pad",
                bytes,
                &OutputMetadata::tensor(DataType::Uint8, &[]),
            )
            .unwrap();
            assert_eq!(plan.requirement().bytes, bytes as u64);
            let bound = plan
                .bind(Some(WorkspaceView::new(
                    DevicePtrMut(0x10000usize as *mut _),
                    bytes,
                )))
                .unwrap();
            bound.validate_bytes(bytes).unwrap();
            assert!(
                bound
                    .validate_bytes(bytes - 1)
                    .unwrap_err()
                    .to_string()
                    .contains("payload")
            );
        }
        let empty = MetadataWorkspace::for_bytes(
            "Resize",
            usize::MAX,
            &OutputMetadata::tensor(DataType::Float32, &[0]),
        )
        .unwrap();
        assert_eq!(empty.requirement(), WorkspaceRequirement::NONE);
        empty.bind(None).unwrap().validate_bytes(0).unwrap();
    }

    #[test]
    fn checked_dense_spans_reject_overflow_and_empty_strides_need_no_address_math() {
        assert_eq!(
            checked_metadata_strides("QLinearMatMul", "batches", &[2, 3, 4]).unwrap(),
            vec![12, 4, 1]
        );
        assert_eq!(
            checked_metadata_strides("CenterCropPad", "input", &[usize::MAX, usize::MAX, 0])
                .unwrap(),
            vec![0; 3]
        );
        assert_eq!(
            checked_tensor_bytes("Col2Im", "input", DataType::Float32, &[usize::MAX, 0]).unwrap(),
            0
        );
        for shape in [&[usize::MAX, 2][..], &[isize::MAX as usize][..]] {
            assert!(
                checked_tensor_bytes("Col2Im", "image", DataType::Float32, shape)
                    .unwrap_err()
                    .to_string()
                    .contains("addressable storage")
            );
        }
    }

    #[test]
    fn rank_metadata_sizes_are_one_persistent_requirement() {
        for (op, rank, words, bytes) in [
            ("Mod", 3, 3, 72),
            ("LpPool", 2, 6, 96),
            ("NonZero", 4, 1, 32),
        ] {
            let plan = MetadataWorkspace::for_records(
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
        let plan = MetadataWorkspace::for_records("Mod", 0, 3, &scalar).unwrap();
        assert_eq!(plan.requirement(), WorkspaceRequirement::NONE);
        plan.bind(None).unwrap().validate_bytes(0).unwrap();
        let empty = MetadataWorkspace::for_records(
            "LpPool",
            usize::MAX,
            6,
            &OutputMetadata::tensor(DataType::Float32, &[1, 0, 3]),
        )
        .unwrap();
        assert_eq!(empty.requirement(), WorkspaceRequirement::NONE);
        assert!(MetadataWorkspace::for_records("Mod", usize::MAX, 3, &scalar).is_err());
        assert!(
            MetadataWorkspace::for_records("Mod", 1, 3, &OutputMetadata::Absent)
                .err()
                .unwrap()
                .to_string()
                .contains("required output 0")
        );
    }
    #[test]
    fn complete_binding_rejects_missing_short_unaligned_and_stale_payload() {
        let plan = MetadataWorkspace::for_records(
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
            let error = plan.bind(backing).unwrap_err().to_string();
            assert!(error.contains("Mod"), "{error}");
            assert!(error.contains("prepare and pass"), "{error}");
        }
        assert_eq!(
            plan.bind(Some(view(0x10000usize, 64)))
                .unwrap()
                .region
                .bytes(),
            48
        );
        assert!(
            plan.bind(Some(view(0x10000usize, 64)))
                .unwrap()
                .validate_bytes(40)
                .unwrap_err()
                .to_string()
                .contains("payload has 40")
        );
    }
}
