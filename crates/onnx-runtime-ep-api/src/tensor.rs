//! Zero-copy device tensor views (§5.4).
//!
//! These are thin, non-owning views over device-resident memory used at the
//! kernel boundary. They intentionally use raw device pointers; the owning
//! session/EP guarantees the backing memory outlives the view.

use onnx_runtime_ir::DataType;
use onnx_runtime_ir::DeviceId;
use std::marker::PhantomData;

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
}

impl DevicePtrMut {
    /// Reinterpret as a typed mutable pointer. Caller ensures the element type.
    pub fn as_ptr<T>(self) -> *mut T {
        self.0 as *mut T
    }
}

/// Immutable, non-owning view of a tensor on any device.
pub struct TensorView<'a> {
    pub data: DevicePtr,
    pub dtype: DataType,
    pub shape: &'a [usize],
    /// Strides in **elements**.
    pub strides: &'a [i64],
    pub device: DeviceId,
    _marker: PhantomData<&'a ()>,
}

impl<'a> TensorView<'a> {
    /// Construct a view. `data` must remain valid for `'a`.
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
            device,
            _marker: PhantomData,
        }
    }

    /// Whether the view is contiguous row-major.
    pub fn is_contiguous(&self) -> bool {
        onnx_runtime_ir::is_contiguous(self.shape, self.strides)
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total element byte size (dense; ignores stride gaps).
    pub fn byte_size(&self) -> usize {
        self.dtype.storage_bytes(self.numel())
    }

    /// Typed const data pointer.
    pub fn data_ptr<T>(&self) -> *const T {
        self.data.as_ptr::<T>()
    }
}

/// Mutable, non-owning view of a tensor on any device.
pub struct TensorMut<'a> {
    pub data: DevicePtrMut,
    pub dtype: DataType,
    pub shape: &'a [usize],
    /// Strides in **elements**.
    pub strides: &'a [i64],
    pub device: DeviceId,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a> TensorMut<'a> {
    /// Construct a mutable view. `data` must remain valid and exclusively
    /// borrowed for `'a`.
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
            device,
            _marker: PhantomData,
        }
    }

    /// Whether the view is contiguous row-major.
    pub fn is_contiguous(&self) -> bool {
        onnx_runtime_ir::is_contiguous(self.shape, self.strides)
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Typed mutable data pointer.
    pub fn data_ptr_mut<T>(&mut self) -> *mut T {
        self.data.as_ptr::<T>()
    }
}
