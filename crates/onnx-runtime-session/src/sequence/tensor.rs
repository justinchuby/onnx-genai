use std::sync::Arc;

use onnx_runtime_ir::{DataType, TensorLayout, compute_contiguous_strides};

use crate::Tensor;
use crate::tensor::{SharedTensorBuffer, host_bytes};

use super::{ConcatCopyStats, SequenceError, SequenceResult};

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

pub(super) fn normalize_axis(
    op: &'static str,
    axis: i64,
    rank: usize,
    new_axis: bool,
) -> SequenceResult<usize> {
    let rank_i64 =
        i64::try_from(rank).map_err(|_| SequenceError::LengthOverflow { op, len: rank })?;
    let normalized = if axis < 0 {
        rank_i64.checked_add(axis)
    } else {
        Some(axis)
    };
    match normalized {
        Some(axis) if axis >= 0 && axis < rank_i64 => Ok(axis as usize),
        _ => Err(SequenceError::InvalidAxis {
            op,
            axis,
            rank: rank - usize::from(new_axis),
            new_axis,
        }),
    }
}

pub(super) fn overflow(op: &'static str, context: &'static str, shape: &[usize]) -> SequenceError {
    SequenceError::ShapeOverflow {
        op,
        context,
        shape: shape.to_vec(),
    }
}

/// Multiply dimensions while still detecting overflow hidden by a zero extent.
pub(super) fn checked_product(
    op: &'static str,
    context: &'static str,
    shape: &[usize],
) -> SequenceResult<usize> {
    let mut product = 1usize;
    let mut has_zero = false;
    for &dimension in shape {
        if dimension == 0 {
            has_zero = true;
        } else {
            product = product
                .checked_mul(dimension)
                .ok_or_else(|| overflow(op, context, shape))?;
        }
    }
    Ok(if has_zero { 0 } else { product })
}

pub(super) fn checked_mul(
    op: &'static str,
    context: &'static str,
    lhs: usize,
    rhs: usize,
    shape: &[usize],
) -> SequenceResult<usize> {
    lhs.checked_mul(rhs)
        .ok_or_else(|| overflow(op, context, shape))
}

pub(super) fn checked_add(
    op: &'static str,
    context: &'static str,
    lhs: usize,
    rhs: usize,
    shape: &[usize],
) -> SequenceResult<usize> {
    lhs.checked_add(rhs)
        .ok_or_else(|| overflow(op, context, shape))
}

pub(super) fn addressable(
    op: &'static str,
    context: &'static str,
    bytes: usize,
    shape: &[usize],
) -> SequenceResult<usize> {
    if bytes > isize::MAX as usize {
        return Err(overflow(op, context, shape));
    }
    Ok(bytes)
}

pub(super) fn zeroed_bytes(
    op: &'static str,
    context: &'static str,
    bytes: usize,
    shape: &[usize],
) -> SequenceResult<Vec<u8>> {
    addressable(op, context, bytes, shape)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(bytes)
        .map_err(|_| SequenceError::Allocation { op, context, bytes })?;
    output.resize(bytes, 0);
    Ok(output)
}

pub(crate) fn clone_shape(op: &'static str, shape: &[usize]) -> SequenceResult<Vec<usize>> {
    let bytes = shape
        .len()
        .checked_mul(std::mem::size_of::<usize>())
        .ok_or_else(|| overflow(op, "shape allocation", shape))?;
    let mut cloned = Vec::new();
    cloned
        .try_reserve_exact(shape.len())
        .map_err(|_| SequenceError::Allocation {
            op,
            context: "shape",
            bytes,
        })?;
    cloned.extend_from_slice(shape);
    Ok(cloned)
}

pub(super) fn validate_tensor_bytes(
    op: &'static str,
    data: &[u8],
    dtype: DataType,
    shape: &[usize],
) -> SequenceResult<()> {
    if dtype.byte_size() == 0 {
        return Err(SequenceError::UnsupportedDtype { op, dtype });
    }
    let numel = checked_product(op, "tensor element count", shape)?;
    let expected = dtype
        .checked_storage_bytes(numel)
        .ok_or_else(|| overflow(op, "tensor byte count", shape))?;
    addressable(op, "tensor byte count", expected, shape)?;
    if data.len() != expected {
        return Err(SequenceError::ByteLengthMismatch {
            op,
            dtype,
            shape: clone_shape(op, shape)?,
            expected,
            actual: data.len(),
        });
    }
    Ok(())
}

pub(super) fn validate_view_bounds(
    op: &'static str,
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    dtype: DataType,
    root_len: usize,
) -> SequenceResult<()> {
    if shape.len() != strides.len() {
        return Err(SequenceError::InvalidSplit {
            op,
            reason: format!(
                "view rank mismatch: shape has {} dims but strides has {}",
                shape.len(),
                strides.len()
            ),
        });
    }
    let esize = dtype.byte_size();
    if esize == 0 {
        return Err(SequenceError::UnsupportedDtype { op, dtype });
    }
    if shape.contains(&0) {
        return Ok(());
    }
    let mut min_element = 0i128;
    let mut max_element = 0i128;
    for (&dim, &stride) in shape.iter().zip(strides) {
        let span = (dim.saturating_sub(1) as i128)
            .checked_mul(stride as i128)
            .ok_or_else(|| overflow(op, "view stride span", shape))?;
        if span < 0 {
            min_element = min_element
                .checked_add(span)
                .ok_or_else(|| overflow(op, "view minimum offset", shape))?;
        } else {
            max_element = max_element
                .checked_add(span)
                .ok_or_else(|| overflow(op, "view maximum offset", shape))?;
        }
    }
    let origin = byte_offset as i128;
    let min_byte = origin
        .checked_add(
            min_element
                .checked_mul(esize as i128)
                .ok_or_else(|| overflow(op, "view minimum byte offset", shape))?,
        )
        .ok_or_else(|| overflow(op, "view minimum byte offset", shape))?;
    let end_byte = origin
        .checked_add(
            max_element
                .checked_mul(esize as i128)
                .ok_or_else(|| overflow(op, "view maximum byte offset", shape))?,
        )
        .and_then(|offset| offset.checked_add(esize as i128))
        .ok_or_else(|| overflow(op, "view byte range", shape))?;
    if min_byte < 0 || end_byte > root_len as i128 {
        return Err(SequenceError::InvalidSplit {
            op,
            reason: format!(
                "view byte range [{min_byte}, {end_byte}) exceeds backing allocation of {root_len} bytes"
            ),
        });
    }
    Ok(())
}

fn gather_strided(
    root: &[u8],
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
    dtype: DataType,
    esize: usize,
) -> SequenceResult<Vec<u8>> {
    const OP: &str = "SequenceTensor";
    validate_view_bounds(OP, shape, strides, byte_offset, dtype, root.len())?;
    let numel = checked_product(OP, "view element count", shape)?;
    let bytes = checked_mul(OP, "view byte count", numel, esize, shape)?;
    let mut output = zeroed_bytes(OP, "strided tensor materialization", bytes, shape)?;
    if numel == 0 {
        return Ok(output);
    }
    let logical_strides = compute_contiguous_strides(shape);
    for linear in 0..numel {
        let mut remainder = linear;
        let mut source_element = 0i128;
        for dimension in 0..shape.len() {
            let coordinate = if shape[dimension] == 0 {
                0
            } else {
                remainder / logical_strides[dimension] as usize
            };
            if shape[dimension] != 0 {
                remainder %= logical_strides[dimension] as usize;
            }
            source_element += coordinate as i128 * strides[dimension] as i128;
        }
        let source = (byte_offset as i128 + source_element * esize as i128) as usize;
        output[linear * esize..(linear + 1) * esize].copy_from_slice(&root[source..source + esize]);
    }
    Ok(output)
}
