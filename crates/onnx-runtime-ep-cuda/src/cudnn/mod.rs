//! cuDNN handle and tensor-descriptor foundation.
//!
//! The handle is created lazily on first cuDNN use, reuses the CUDA EP's
//! existing stream/context, and is serialized because cuDNN handles are not
//! safe for concurrent host-thread use. cudarc owns all native resources, so
//! handles and descriptors are destroyed through its RAII wrappers.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use cudarc::cudnn::{Cudnn, CudnnDataType, TensorDescriptor, sys};
use cudarc::driver::CudaStream;
use half::{bf16, f16};
use onnx_runtime_ep_api::{EpError, Result};
use onnx_runtime_ir::DataType;

use crate::error::{cudnn_err, cudnn_unavailable, driver_err};

/// cuDNN element types supported by the CUDA EP's library-backed kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudnnTensorType {
    F32,
    F16,
    Bf16,
}

impl CudnnTensorType {
    /// Convert an ONNX tensor dtype into the corresponding cuDNN dtype.
    pub fn from_onnx(dtype: DataType) -> Result<Self> {
        match dtype {
            DataType::Float32 => Ok(Self::F32),
            DataType::Float16 => Ok(Self::F16),
            DataType::BFloat16 => Ok(Self::Bf16),
            other => Err(EpError::KernelFailed(format!(
                "cuda_ep: cuDNN tensor descriptors support f32, f16, and bf16; got {other:?}"
            ))),
        }
    }

    /// The raw cuDNN datatype value used by descriptor creation.
    pub fn as_raw(self) -> sys::cudnnDataType_t {
        match self {
            Self::F32 => <f32 as CudnnDataType>::DATA_TYPE,
            Self::F16 => <f16 as CudnnDataType>::DATA_TYPE,
            Self::Bf16 => <bf16 as CudnnDataType>::DATA_TYPE,
        }
    }
}

/// Validated, cuDNN-ready tensor descriptor dimensions and strides.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorDescriptorSpec {
    dtype: CudnnTensorType,
    dims: Vec<i32>,
    strides: Vec<i32>,
}

impl TensorDescriptorSpec {
    /// Validate ONNX dimensions/element-strides and pad ranks below four as
    /// required by `cudnnSetTensorNdDescriptor`.
    pub fn new(dtype: DataType, dims: &[usize], strides: &[usize]) -> Result<Self> {
        if dims.len() != strides.len() {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep: cuDNN tensor descriptor has {} dims but {} strides",
                dims.len(),
                strides.len()
            )));
        }
        if dims.is_empty() {
            return Err(EpError::KernelFailed(
                "cuda_ep: cuDNN tensor descriptor requires rank >= 1".into(),
            ));
        }
        if dims.contains(&0) {
            return Err(EpError::KernelFailed(
                "cuda_ep: cuDNN tensor descriptors cannot represent zero-sized dimensions; \
                 empty tensors must return before cuDNN dispatch"
                    .into(),
            ));
        }
        if strides.contains(&0) {
            return Err(EpError::KernelFailed(
                "cuda_ep: cuDNN tensor descriptor strides must be positive".into(),
            ));
        }

        let dtype = CudnnTensorType::from_onnx(dtype)?;
        let mut padded_dims = Vec::with_capacity(dims.len().max(4));
        let mut padded_strides = Vec::with_capacity(strides.len().max(4));
        let leading_stride = dims[0].checked_mul(strides[0]).ok_or_else(|| {
            EpError::KernelFailed(
                "cuda_ep: cuDNN tensor descriptor leading stride overflowed usize".into(),
            )
        })?;
        for _ in dims.len()..4 {
            padded_dims.push(1);
            padded_strides.push(i32_value("leading stride", leading_stride)?);
        }
        for (&dim, &stride) in dims.iter().zip(strides) {
            padded_dims.push(i32_value("dimension", dim)?);
            padded_strides.push(i32_value("stride", stride)?);
        }

        Ok(Self {
            dtype,
            dims: padded_dims,
            strides: padded_strides,
        })
    }

    pub fn dtype(&self) -> CudnnTensorType {
        self.dtype
    }

    pub fn dims(&self) -> &[i32] {
        &self.dims
    }

    pub fn strides(&self) -> &[i32] {
        &self.strides
    }
}

fn i32_value(name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        EpError::KernelFailed(format!(
            "cuda_ep: cuDNN tensor descriptor {name} {value} exceeds i32"
        ))
    })
}

/// An owned cudarc tensor descriptor for one of the supported ONNX dtypes.
///
/// Its lifetime prevents native resources from escaping
/// [`CudnnBackend::with_handle`]'s serialization lock.
#[derive(Debug)]
pub struct CudnnTensorDescriptor<'handle> {
    inner: TensorDescriptorInner,
    _handle: PhantomData<&'handle CudnnHandle<'handle>>,
}

#[derive(Debug)]
enum TensorDescriptorInner {
    F32(TensorDescriptor<f32>),
    F16(TensorDescriptor<f16>),
    Bf16(TensorDescriptor<bf16>),
}

impl CudnnTensorDescriptor<'_> {
    pub fn dtype(&self) -> CudnnTensorType {
        match self.inner {
            TensorDescriptorInner::F32(_) => CudnnTensorType::F32,
            TensorDescriptorInner::F16(_) => CudnnTensorType::F16,
            TensorDescriptorInner::Bf16(_) => CudnnTensorType::Bf16,
        }
    }

    pub fn as_f32(&self) -> Option<&TensorDescriptor<f32>> {
        match &self.inner {
            TensorDescriptorInner::F32(descriptor) => Some(descriptor),
            _ => None,
        }
    }

    pub fn as_f16(&self) -> Option<&TensorDescriptor<f16>> {
        match &self.inner {
            TensorDescriptorInner::F16(descriptor) => Some(descriptor),
            _ => None,
        }
    }

    pub fn as_bf16(&self) -> Option<&TensorDescriptor<bf16>> {
        match &self.inner {
            TensorDescriptorInner::Bf16(descriptor) => Some(descriptor),
            _ => None,
        }
    }
}

/// Exclusive, lifetime-bound access to the cuDNN handle.
pub struct CudnnHandle<'handle> {
    handle: &'handle Arc<Cudnn>,
}

impl CudnnHandle<'_> {
    /// Build an owned tensor descriptor using cudarc's RAII wrapper.
    pub fn tensor_descriptor<'handle>(
        &'handle self,
        spec: &TensorDescriptorSpec,
    ) -> Result<CudnnTensorDescriptor<'handle>> {
        let inner = match spec.dtype {
            CudnnTensorType::F32 => self
                .handle
                .create_nd_tensor::<f32>(&spec.dims, &spec.strides)
                .map(TensorDescriptorInner::F32),
            CudnnTensorType::F16 => self
                .handle
                .create_nd_tensor::<f16>(&spec.dims, &spec.strides)
                .map(TensorDescriptorInner::F16),
            CudnnTensorType::Bf16 => self
                .handle
                .create_nd_tensor::<bf16>(&spec.dims, &spec.strides)
                .map(TensorDescriptorInner::Bf16),
        }
        .map_err(|e| cudnn_err("creating tensor descriptor", e))?;
        Ok(CudnnTensorDescriptor {
            inner,
            _handle: PhantomData,
        })
    }
}

/// Serialized access to a lazily created cuDNN handle.
pub struct CudnnBackend {
    stream: Arc<CudaStream>,
    handle: Mutex<Option<Arc<Cudnn>>>,
}

impl std::fmt::Debug for CudnnBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudnnBackend").finish_non_exhaustive()
    }
}

// SAFETY: cudarc deliberately keeps `Cudnn` !Send/!Sync because a handle must
// not be used concurrently. Every access here is serialized by `handle`, and
// `with_handle` binds the owning CUDA context to the calling thread first.
unsafe impl Send for CudnnBackend {}
unsafe impl Sync for CudnnBackend {}

impl CudnnBackend {
    /// Create an uninitialized backend for the EP's existing compute stream.
    pub fn new(stream: Arc<CudaStream>) -> Self {
        Self {
            stream,
            handle: Mutex::new(None),
        }
    }

    /// Run one cuDNN operation with exclusive access to the stream-bound handle.
    ///
    /// Later op implementations should create all descriptors and submit the
    /// cuDNN call inside this closure.
    pub fn with_handle<T>(
        &self,
        operation: impl for<'handle> FnOnce(CudnnHandle<'handle>) -> Result<T>,
    ) -> Result<T> {
        self.stream
            .context()
            .bind_to_thread()
            .map_err(|e| driver_err("binding context for cuDNN", e))?;
        ensure_cudnn_available(cudnn_library_present)?;

        let mut handle = self.handle.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep: cuDNN handle mutex was poisoned".into())
        })?;
        if handle.is_none() {
            *handle = Some(initialize_cudnn(|| Cudnn::new(self.stream.clone()))?);
        }
        let handle = handle.as_ref().ok_or_else(|| {
            EpError::KernelFailed("cuda_ep: cuDNN handle initialization produced no handle".into())
        })?;
        operation(CudnnHandle { handle })
    }
}

impl Drop for CudnnBackend {
    fn drop(&mut self) {
        let handle = self
            .handle
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if handle.is_some() {
            let _ = self.stream.context().bind_to_thread();
            handle.take();
        }
    }
}

fn cudnn_library_present() -> bool {
    // SAFETY: this only asks cudarc to probe the platform loader for cuDNN and
    // immediately unloads the probe handle. No cuDNN function is called.
    unsafe { sys::is_culib_present() }
}

fn ensure_cudnn_available(probe: impl FnOnce() -> bool) -> Result<()> {
    if probe() {
        Ok(())
    } else {
        Err(cudnn_unavailable())
    }
}

fn initialize_cudnn<T>(
    initialize: impl FnOnce() -> std::result::Result<T, cudarc::cudnn::CudnnError>,
) -> Result<T> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(initialize))
        .map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep: cuDNN handle initialization failed while loading the cuDNN runtime \
                 or required symbols; install a compatible cuDNN 9 runtime with \
                 'pip install nvidia-cudnn-cu13'"
                    .into(),
            )
        })?
        .map_err(|e| cudnn_err("cudnnCreate / cudnnSetStream", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_supported_onnx_dtypes() {
        assert_eq!(
            CudnnTensorType::from_onnx(DataType::Float32).unwrap(),
            CudnnTensorType::F32
        );
        assert_eq!(
            CudnnTensorType::from_onnx(DataType::Float16).unwrap(),
            CudnnTensorType::F16
        );
        assert_eq!(
            CudnnTensorType::from_onnx(DataType::BFloat16).unwrap(),
            CudnnTensorType::Bf16
        );
        assert_eq!(
            CudnnTensorType::F32.as_raw(),
            sys::cudnnDataType_t::CUDNN_DATA_FLOAT
        );
        assert_eq!(
            CudnnTensorType::F16.as_raw(),
            sys::cudnnDataType_t::CUDNN_DATA_HALF
        );
        assert_eq!(
            CudnnTensorType::Bf16.as_raw(),
            sys::cudnnDataType_t::CUDNN_DATA_BFLOAT16
        );
    }

    #[test]
    fn rejects_unsupported_onnx_dtype() {
        let error = CudnnTensorType::from_onnx(DataType::Int32).unwrap_err();
        assert!(error.to_string().contains("f32, f16, and bf16"));
    }

    #[test]
    fn descriptor_spec_preserves_dims_and_strides() {
        let spec =
            TensorDescriptorSpec::new(DataType::Float16, &[2, 3, 5, 7], &[105, 35, 7, 1]).unwrap();
        assert_eq!(spec.dtype(), CudnnTensorType::F16);
        assert_eq!(spec.dims(), &[2, 3, 5, 7]);
        assert_eq!(spec.strides(), &[105, 35, 7, 1]);
    }

    #[test]
    fn descriptor_spec_pads_low_rank_tensors() {
        let spec = TensorDescriptorSpec::new(DataType::BFloat16, &[2, 3], &[3, 1]).unwrap();
        assert_eq!(spec.dims(), &[1, 1, 2, 3]);
        assert_eq!(spec.strides(), &[6, 6, 3, 1]);
    }

    #[test]
    fn descriptor_spec_rejects_invalid_layouts() {
        assert!(TensorDescriptorSpec::new(DataType::Float32, &[2, 3], &[3]).is_err());
        assert!(TensorDescriptorSpec::new(DataType::Float32, &[2, 0], &[1, 1]).is_err());
        assert!(TensorDescriptorSpec::new(DataType::Float32, &[2, 3], &[0, 1]).is_err());
        assert!(
            TensorDescriptorSpec::new(DataType::Float32, &[i32::MAX as usize + 1], &[1]).is_err()
        );
    }

    #[test]
    fn missing_cudnn_is_an_actionable_runtime_error() {
        let error = ensure_cudnn_available(|| false).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("libcudnn.so.9"));
        assert!(message.contains("pip install nvidia-cudnn-cu13"));
    }

    #[test]
    fn handle_creation_failure_is_an_error() {
        let error = initialize_cudnn(|| {
            Err::<(), _>(cudarc::cudnn::CudnnError(
                sys::cudnnStatus_t::CUDNN_STATUS_NOT_INITIALIZED,
            ))
        })
        .unwrap_err();
        assert!(error.to_string().contains("cudnnCreate / cudnnSetStream"));
    }
}
