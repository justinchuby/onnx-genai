use std::sync::Arc;

use onnx_runtime_ir::{DataType, TensorLayout, compute_contiguous_strides};

use crate::Tensor;
use crate::tensor::{SharedTensorBuffer, host_bytes};

use super::{
    ConcatCopyStats, SequenceError, SequenceResult, checked_add, checked_mul, checked_product,
    gather_strided, overflow, validate_tensor_bytes, validate_view_bounds, zeroed_bytes,
};

/// An immutable tensor view used as one sequence element.
///
/// Cloning this type only bumps the backing allocation's `Arc` count. Shape,
/// strides, and byte offset are metadata, so a split slice can share the source
/// allocation even when it is not contiguous.
#[derive(Clone, Debug)]
pub struct SeqTensor {
    pub(super) storage: Arc<SharedTensorBuffer>,
    pub dtype: DataType,
    pub shape: Vec<usize>,
    pub layout: TensorLayout,
    pub(super) byte_offset: usize,
}

impl SeqTensor {
    /// Wrap an existing session tensor in an immutable shared handle.
    pub fn new(tensor: Tensor) -> Self {
        let (storage, dtype, shape, layout) = tensor.into_shared_parts();
        Self {
            storage,
            dtype,
            shape,
            layout,
            byte_offset: 0,
        }
    }

    /// Build a host tensor from raw element bytes and share it as a sequence item.
    pub fn from_raw(dtype: DataType, shape: Vec<usize>, bytes: &[u8]) -> SequenceResult<Self> {
        validate_tensor_bytes("SequenceTensor", bytes, dtype, &shape)?;
        let mut storage = SharedTensorBuffer::allocate_cpu(bytes.len()).map_err(|source| {
            SequenceError::TensorCreation {
                op: "SequenceTensor",
                source,
            }
        })?;
        let allocator = Arc::clone(storage.allocator());
        allocator
            .copy_from_host(
                bytes,
                Arc::get_mut(&mut storage)
                    .expect("fresh sequence storage is uniquely owned")
                    .buffer_mut(),
            )
            .map_err(|source| SequenceError::TensorCreation {
                op: "SequenceTensor",
                source: source.into(),
            })?;
        Ok(Self {
            storage,
            dtype,
            shape,
            layout: TensorLayout::contiguous(),
            byte_offset: 0,
        })
    }

    pub(crate) fn from_shared(
        storage: Arc<SharedTensorBuffer>,
        dtype: DataType,
        shape: Vec<usize>,
        layout: TensorLayout,
        byte_offset: usize,
    ) -> SequenceResult<Self> {
        let strides = layout.resolved_strides(&shape);
        validate_view_bounds(
            "SequenceTensor",
            &shape,
            &strides,
            byte_offset,
            dtype,
            storage.buffer().len(),
        )?;
        Ok(Self {
            storage,
            dtype,
            shape,
            layout,
            byte_offset,
        })
    }

    /// Whether two handles share the same underlying device allocation.
    pub fn shares_storage_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.storage, &other.storage)
    }

    /// Number of live handles to the shared allocation.
    pub fn storage_strong_count(&self) -> usize {
        Arc::strong_count(&self.storage)
    }

    pub(crate) fn storage(&self) -> &Arc<SharedTensorBuffer> {
        &self.storage
    }

    /// Base address of the shared tensor allocation.
    pub fn as_ptr(&self) -> *const std::ffi::c_void {
        self.storage.buffer().as_ptr()
    }

    /// Byte offset of this view's logical origin from [`Self::as_ptr`].
    pub fn byte_offset(&self) -> usize {
        self.byte_offset
    }

    pub fn device(&self) -> onnx_runtime_ir::DeviceId {
        self.storage.buffer().device()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub(crate) fn root_len(&self) -> usize {
        self.storage.buffer().len()
    }

    /// Materialize this logical tensor as contiguous host bytes.
    pub fn contiguous_bytes(&self) -> SequenceResult<Vec<u8>> {
        const OP: &str = "SequenceTensor";
        let esize = self.dtype.byte_size();
        if esize == 0 {
            return Err(SequenceError::UnsupportedDtype {
                op: OP,
                dtype: self.dtype,
            });
        }
        let mut copied;
        let root = if self.device().is_host_accessible() {
            host_bytes(self.storage.buffer())
        } else {
            copied = zeroed_bytes(OP, "device tensor download", self.root_len(), &self.shape)?;
            self.storage
                .allocator()
                .copy_to_host(self.storage.buffer(), &mut copied)
                .map_err(|source| SequenceError::TensorCreation {
                    op: OP,
                    source: source.into(),
                })?;
            &copied
        };
        let strides = self.layout.resolved_strides(&self.shape);
        if onnx_runtime_ir::is_contiguous(&self.shape, &strides) {
            let bytes = self
                .dtype
                .checked_storage_bytes(self.numel())
                .ok_or_else(|| overflow(OP, "tensor byte count", &self.shape))?;
            let end = checked_add(
                OP,
                "tensor byte range",
                self.byte_offset,
                bytes,
                &self.shape,
            )?;
            return Ok(root[self.byte_offset..end].to_vec());
        }
        gather_strided(
            root,
            &self.shape,
            &strides,
            self.byte_offset,
            self.dtype,
            esize,
        )
    }

    pub(super) fn write_contiguous_range<F>(
        &self,
        logical_offset: usize,
        bytes: usize,
        destination_offset: usize,
        scratch: &mut [u8],
        write: &mut F,
        stats: &mut ConcatCopyStats,
    ) -> crate::Result<()>
    where
        F: FnMut(usize, &[u8]) -> crate::Result<()>,
    {
        const OP: &str = "ConcatFromSequence";
        let esize = self.dtype.byte_size();
        if esize == 0 {
            return Err(SequenceError::UnsupportedDtype {
                op: OP,
                dtype: self.dtype,
            }
            .into());
        }
        let logical_bytes = self
            .dtype
            .checked_storage_bytes(checked_product(OP, "source element count", &self.shape)?)
            .ok_or_else(|| overflow(OP, "source byte count", &self.shape))?;
        let logical_end = checked_add(
            OP,
            "source logical byte range",
            logical_offset,
            bytes,
            &self.shape,
        )?;
        if logical_end > logical_bytes
            || !logical_offset.is_multiple_of(esize)
            || !bytes.is_multiple_of(esize)
        {
            return Err(SequenceError::ByteLengthMismatch {
                op: OP,
                dtype: self.dtype,
                shape: self.shape.clone(),
                expected: logical_bytes,
                actual: logical_end,
            }
            .into());
        }
        if bytes == 0 {
            return Ok(());
        }

        let root = if self.device().is_host_accessible() {
            host_bytes(self.storage.buffer())
        } else {
            let destination =
                scratch
                    .get_mut(..self.root_len())
                    .ok_or_else(|| SequenceError::Allocation {
                        op: OP,
                        context: "device source materialization",
                        bytes: self.root_len(),
                    })?;
            self.storage
                .allocator()
                .copy_to_host(self.storage.buffer(), destination)
                .map_err(|source| SequenceError::TensorCreation {
                    op: OP,
                    source: source.into(),
                })?;
            stats.source_materializations += 1;
            destination
        };
        let strides = self.layout.resolved_strides(&self.shape);
        validate_view_bounds(
            OP,
            &self.shape,
            &strides,
            self.byte_offset,
            self.dtype,
            root.len(),
        )?;
        if onnx_runtime_ir::is_contiguous(&self.shape, &strides) {
            let source_offset = checked_add(
                OP,
                "contiguous source byte offset",
                self.byte_offset,
                logical_offset,
                &self.shape,
            )?;
            let source_end = checked_add(
                OP,
                "contiguous source byte range",
                source_offset,
                bytes,
                &self.shape,
            )?;
            let source = root
                .get(source_offset..source_end)
                .ok_or_else(|| overflow(OP, "contiguous source byte range", &self.shape))?;
            write(destination_offset, source)?;
            stats.destination_writes += 1;
            return Ok(());
        }

        let logical_strides = compute_contiguous_strides(&self.shape);
        let start_element = logical_offset / esize;
        let elements = bytes / esize;
        for element_offset in 0..elements {
            let linear = checked_add(
                OP,
                "strided source logical index",
                start_element,
                element_offset,
                &self.shape,
            )?;
            let mut remainder = linear;
            let mut source_element = 0i128;
            for dimension in 0..self.shape.len() {
                let coordinate = if self.shape[dimension] == 0 {
                    0
                } else {
                    remainder / logical_strides[dimension] as usize
                };
                if self.shape[dimension] != 0 {
                    remainder %= logical_strides[dimension] as usize;
                }
                source_element += coordinate as i128 * strides[dimension] as i128;
            }
            let source_offset = (self.byte_offset as i128)
                .checked_add(
                    source_element
                        .checked_mul(esize as i128)
                        .ok_or_else(|| overflow(OP, "strided source byte offset", &self.shape))?,
                )
                .and_then(|offset| usize::try_from(offset).ok())
                .ok_or_else(|| overflow(OP, "strided source byte offset", &self.shape))?;
            let source_end = checked_add(
                OP,
                "strided source byte range",
                source_offset,
                esize,
                &self.shape,
            )?;
            let destination = checked_add(
                OP,
                "strided destination byte offset",
                destination_offset,
                checked_mul(
                    OP,
                    "strided destination element offset",
                    element_offset,
                    esize,
                    &self.shape,
                )?,
                &self.shape,
            )?;
            let source = root
                .get(source_offset..source_end)
                .ok_or_else(|| overflow(OP, "strided source byte range", &self.shape))?;
            write(destination, source)?;
            stats.destination_writes += 1;
        }
        Ok(())
    }

    /// Borrow bytes directly when this is a contiguous host view.
    pub fn as_bytes(&self) -> SequenceResult<&[u8]> {
        const OP: &str = "SequenceTensor";
        if !self.device().is_host_accessible() {
            return Err(SequenceError::ByteBorrowUnavailable {
                shape: self.shape.clone(),
                device: format!("{:?}", self.device()),
                reason: "the storage is not host-accessible",
            });
        }
        let strides = self.layout.resolved_strides(&self.shape);
        if !onnx_runtime_ir::is_contiguous(&self.shape, &strides) {
            return Err(SequenceError::ByteBorrowUnavailable {
                shape: self.shape.clone(),
                device: format!("{:?}", self.device()),
                reason: "the view is strided",
            });
        }
        validate_view_bounds(
            OP,
            &self.shape,
            &strides,
            self.byte_offset,
            self.dtype,
            self.root_len(),
        )?;
        let bytes = self
            .dtype
            .checked_storage_bytes(checked_product(OP, "tensor element count", &self.shape)?)
            .ok_or_else(|| overflow(OP, "tensor byte count", &self.shape))?;
        let end = checked_add(
            OP,
            "tensor byte range",
            self.byte_offset,
            bytes,
            &self.shape,
        )?;
        host_bytes(self.storage.buffer())
            .get(self.byte_offset..end)
            .ok_or_else(|| overflow(OP, "tensor byte range", &self.shape))
    }
}
