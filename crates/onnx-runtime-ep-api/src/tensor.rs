//! Zero-copy device tensor views (§5.4) and their DLPack alignment (§5.3).
//!
//! These are thin, non-owning views over device-resident memory used at the
//! kernel boundary. They intentionally use raw device pointers; the owning
//! session/EP guarantees the backing memory outlives the view.
//!
//! ## DLPack correspondence (§5.3)
//!
//! A [`TensorView`] carries exactly the fields a DLPack `DLTensor` needs to be
//! reconstructed, so import/export is a field-wise mapping rather than a copy:
//!
//! | `DLTensor`            | `TensorView`                |
//! |----------------------|-----------------------------|
//! | `data`               | [`TensorView::data`]        |
//! | `byte_offset`        | [`TensorView::byte_offset`] |
//! | `shape` / `ndim`     | [`TensorView::shape`]       |
//! | `strides` (elements) | [`TensorView::strides`]     |
//! | `device`             | [`TensorView::device`]      |
//! | `dtype`              | [`TensorView::dtype`]       |
//!
//! DLPack keeps `byte_offset` *separate* from `data` (the base pointer is the
//! allocation start; `byte_offset` selects the element origin) and permits
//! negative strides. Both are representable here: strides are `i64` and the
//! offset is applied lazily by [`TensorView::data_ptr`]. Use
//! [`TensorView::validate`] to check the invariants of an imported view before
//! handing it to a kernel.

use onnx_runtime_ir::DataType;
use onnx_runtime_ir::DeviceId;
use std::marker::PhantomData;

use crate::error::{EpError, Result};

/// An opaque immutable device pointer (a host pointer for CPU tensors).
#[derive(Clone, Copy, Debug)]
pub struct DevicePtr(pub *const std::ffi::c_void);

/// An opaque mutable device pointer.
#[derive(Clone, Copy, Debug)]
pub struct DevicePtrMut(pub *mut std::ffi::c_void);

impl DevicePtr {
    /// Reinterpret as a typed const pointer. Caller ensures the element type.
    pub fn as_ptr<T>(self) -> *const T {
        self.0 as *const T
    }

    /// Whether the underlying address is null.
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

impl DevicePtrMut {
    /// Reinterpret as a typed mutable pointer. Caller ensures the element type.
    pub fn as_ptr<T>(self) -> *mut T {
        self.0 as *mut T
    }

    /// Whether the underlying address is null.
    pub fn is_null(self) -> bool {
        self.0.is_null()
    }
}

/// Shared invariant check for both view kinds (§5.3 DLPack import path).
///
/// Verifies the properties a kernel relies on: matching rank between `shape`
/// and `strides`, a raw-view-representable dtype, and a `byte_offset` that keeps
/// the element origin aligned to the element size. Storage bounds cannot be
/// checked here — the view does not know its backing allocation size — so that
/// remains the owning EP's responsibility.
fn validate_view(
    data_is_null: bool,
    dtype: DataType,
    shape: &[usize],
    strides: &[i64],
    byte_offset: usize,
) -> Result<()> {
    if data_is_null {
        return Err(EpError::InvalidTensorView {
            reason: "data pointer is null".into(),
        });
    }
    if shape.len() != strides.len() {
        return Err(EpError::InvalidTensorView {
            reason: format!(
                "rank mismatch: shape has {} dims but strides has {}",
                shape.len(),
                strides.len()
            ),
        });
    }
    if dtype == DataType::String {
        return Err(EpError::InvalidTensorView {
            reason: "String dtype has no fixed-width raw layout".into(),
        });
    }
    // For fixed-width (non-sub-byte) types the element origin must land on an
    // element boundary; sub-byte packed types are addressed per byte.
    let esize = dtype.byte_size();
    if esize > 1 && !byte_offset.is_multiple_of(esize) {
        return Err(EpError::InvalidTensorView {
            reason: format!("byte_offset {byte_offset} is not a multiple of element size {esize}"),
        });
    }
    Ok(())
}

/// Immutable, non-owning view of a tensor on any device.
pub struct TensorView<'a> {
    pub data: DevicePtr,
    pub dtype: DataType,
    pub shape: &'a [usize],
    /// Strides in **elements** (may be negative, matching DLPack).
    pub strides: &'a [i64],
    /// Offset in **bytes** of the element origin from `data` (DLPack semantics).
    pub byte_offset: usize,
    pub device: DeviceId,
    _marker: PhantomData<&'a ()>,
}

impl<'a> TensorView<'a> {
    /// Construct a zero-offset view. `data` must remain valid for `'a`.
    pub fn new(
        data: DevicePtr,
        dtype: DataType,
        shape: &'a [usize],
        strides: &'a [i64],
        device: DeviceId,
    ) -> Self {
        Self {
            data,
            dtype,
            shape,
            strides,
            byte_offset: 0,
            device,
            _marker: PhantomData,
        }
    }

    /// Construct an **absent** view: the positional placeholder an executor
    /// passes for an *omitted optional input* (an ONNX empty-string input name).
    /// It carries a null pointer and an empty shape so it can never be read as a
    /// real tensor — kernels test [`TensorView::is_absent`] before touching an
    /// optional slot. This preserves positional arity so a later present input
    /// (e.g. `Slice`'s `steps` when `axes` is omitted) is not misread as the
    /// omitted one.
    pub fn absent(dtype: DataType) -> Self {
        Self {
            data: DevicePtr(std::ptr::null()),
            dtype,
            shape: &[],
            strides: &[],
            byte_offset: 0,
            device: DeviceId::cpu(),
            _marker: PhantomData,
        }
    }

    /// Whether this view is the [`TensorView::absent`] placeholder for an
    /// omitted optional input (null backing pointer). Kernels with optional
    /// inputs check this to distinguish "input not supplied" from a real,
    /// possibly-empty, tensor.
    pub fn is_absent(&self) -> bool {
        self.data.is_null()
    }

    /// Set the DLPack-style byte offset of the element origin.
    pub fn with_byte_offset(mut self, byte_offset: usize) -> Self {
        self.byte_offset = byte_offset;
        self
    }

    /// Check the view's invariants (rank, dtype, offset alignment). See
    /// [`validate_view`]; call this on any view imported from DLPack before use.
    pub fn validate(&self) -> Result<()> {
        validate_view(
            self.data.is_null(),
            self.dtype,
            self.shape,
            self.strides,
            self.byte_offset,
        )
    }

    /// Whether the view is contiguous row-major.
    pub fn is_contiguous(&self) -> bool {
        onnx_runtime_ir::is_contiguous(self.shape, self.strides)
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Logical element byte size (dense; ignores stride gaps). Uses
    /// `storage_bytes` so sub-byte (int4/uint4) types are counted correctly.
    pub fn byte_size(&self) -> usize {
        self.dtype.storage_bytes(self.numel())
    }

    /// Typed const pointer to the element origin, applying `byte_offset`.
    /// Computed with wrapping arithmetic (no deref) — safe to call.
    pub fn data_ptr<T>(&self) -> *const T {
        (self.data.0 as *const u8).wrapping_add(self.byte_offset) as *const T
    }
}

/// Mutable, non-owning view of a tensor on any device.
pub struct TensorMut<'a> {
    pub data: DevicePtrMut,
    pub dtype: DataType,
    pub shape: &'a [usize],
    /// Strides in **elements** (may be negative, matching DLPack).
    pub strides: &'a [i64],
    /// Offset in **bytes** of the element origin from `data` (DLPack semantics).
    pub byte_offset: usize,
    pub device: DeviceId,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a> TensorMut<'a> {
    /// Construct a zero-offset mutable view. `data` must remain valid and
    /// exclusively borrowed for `'a`.
    pub fn new(
        data: DevicePtrMut,
        dtype: DataType,
        shape: &'a [usize],
        strides: &'a [i64],
        device: DeviceId,
    ) -> Self {
        Self {
            data,
            dtype,
            shape,
            strides,
            byte_offset: 0,
            device,
            _marker: PhantomData,
        }
    }

    /// Set the DLPack-style byte offset of the element origin.
    pub fn with_byte_offset(mut self, byte_offset: usize) -> Self {
        self.byte_offset = byte_offset;
        self
    }

    /// Check the view's invariants (rank, dtype, offset alignment).
    pub fn validate(&self) -> Result<()> {
        validate_view(
            self.data.is_null(),
            self.dtype,
            self.shape,
            self.strides,
            self.byte_offset,
        )
    }

    /// Whether the view is contiguous row-major.
    pub fn is_contiguous(&self) -> bool {
        onnx_runtime_ir::is_contiguous(self.shape, self.strides)
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Logical element byte size (dense; ignores stride gaps). Uses
    /// `storage_bytes` so sub-byte (int4/uint4) types are counted correctly.
    pub fn byte_size(&self) -> usize {
        self.dtype.storage_bytes(self.numel())
    }

    /// Typed mutable pointer to the element origin, applying `byte_offset`.
    /// Computed with wrapping arithmetic (no deref) — safe to call.
    pub fn data_ptr_mut<T>(&mut self) -> *mut T {
        (self.data.0 as *mut u8).wrapping_add(self.byte_offset) as *mut T
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_ir::compute_contiguous_strides;

    fn ptr(buf: &[u8]) -> DevicePtr {
        DevicePtr(buf.as_ptr() as *const std::ffi::c_void)
    }

    #[test]
    fn contiguous_view_roundtrips_invariants() {
        let buf = vec![0u8; 6 * 4];
        let shape = [2usize, 3];
        let strides = compute_contiguous_strides(&shape);
        let v = TensorView::new(ptr(&buf), DataType::Float32, &shape, &strides, DeviceId::cpu());
        v.validate().unwrap();
        assert!(v.is_contiguous());
        assert_eq!(v.numel(), 6);
        assert_eq!(v.byte_size(), 24);
        assert_eq!(v.byte_offset, 0);
    }

    #[test]
    fn strided_noncontiguous_view_is_representable() {
        // A transposed [3,2] view over a [2,3] contiguous buffer: strides [1,3].
        let buf = vec![0u8; 6 * 4];
        let shape = [3usize, 2];
        let strides = [1i64, 3];
        let v = TensorView::new(ptr(&buf), DataType::Float32, &shape, &strides, DeviceId::cpu())
            .with_byte_offset(4);
        v.validate().unwrap();
        assert!(!v.is_contiguous());
        assert_eq!(v.shape, &[3, 2]);
        assert_eq!(v.strides, &[1, 3]);
        assert_eq!(v.byte_offset, 4);
        // data_ptr applies the byte offset (one f32 in).
        let base = buf.as_ptr() as usize;
        assert_eq!(v.data_ptr::<f32>() as usize, base + 4);
    }

    #[test]
    fn negative_strides_are_representable() {
        // DLPack permits negative strides (reverse iteration).
        let buf = vec![0u8; 4 * 4];
        let shape = [4usize];
        let strides = [-1i64];
        let v = TensorView::new(ptr(&buf), DataType::Float32, &shape, &strides, DeviceId::cpu());
        v.validate().unwrap();
        assert!(!v.is_contiguous());
    }

    #[test]
    fn validate_rejects_rank_mismatch() {
        let buf = vec![0u8; 8];
        let shape = [2usize, 2];
        let strides = [1i64]; // wrong rank
        let v = TensorView::new(ptr(&buf), DataType::Float32, &shape, &strides, DeviceId::cpu());
        assert!(v.validate().is_err());
    }

    #[test]
    fn validate_rejects_misaligned_offset_and_string_and_null() {
        let buf = vec![0u8; 16];
        let shape = [2usize];
        let strides = [1i64];
        // byte_offset not a multiple of f32 element size.
        let bad_off =
            TensorView::new(ptr(&buf), DataType::Float32, &shape, &strides, DeviceId::cpu())
                .with_byte_offset(1);
        assert!(bad_off.validate().is_err());
        // String has no fixed-width raw layout.
        let bad_dt =
            TensorView::new(ptr(&buf), DataType::String, &shape, &strides, DeviceId::cpu());
        assert!(bad_dt.validate().is_err());
        // Null data pointer.
        let bad_null = TensorView::new(
            DevicePtr(std::ptr::null()),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        );
        assert!(bad_null.validate().is_err());
    }

    #[test]
    fn mut_view_offset_pointer() {
        let mut buf = vec![0u8; 4 * 4];
        let base = buf.as_ptr() as usize;
        let shape = [2usize];
        let strides = [1i64];
        let mut v = TensorMut::new(
            DevicePtrMut(buf.as_mut_ptr() as *mut std::ffi::c_void),
            DataType::Float32,
            &shape,
            &strides,
            DeviceId::cpu(),
        )
        .with_byte_offset(8);
        v.validate().unwrap();
        assert_eq!(v.data_ptr_mut::<f32>() as usize, base + 8);
    }

    #[test]
    fn sub_byte_byte_size_uses_packing() {
        let buf = vec![0u8; 4];
        let shape = [5usize];
        let strides = [1i64];
        let v = TensorView::new(ptr(&buf), DataType::Int4, &shape, &strides, DeviceId::cpu());
        // 5 int4 elements pack into 3 bytes.
        assert_eq!(v.byte_size(), 3);
        v.validate().unwrap();
    }
}
