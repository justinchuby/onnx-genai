//! cuDNN handle and tensor-descriptor foundation.
//!
//! The handle is created lazily on first cuDNN use, reuses the CUDA EP's
//! existing stream/context, and is serialized because cuDNN handles are not
//! safe for concurrent host-thread use. cudarc owns all native resources, so
//! handles and descriptors are destroyed through its RAII wrappers.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use cudarc::cudnn::{
    Cudnn, CudnnDataType, NoIndices, ReduceTensor, ReductionDescriptor, SoftmaxForward,
    TensorDescriptor, sys,
};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
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

/// ONNX softmax layouts mapped to cuDNN's two supported reduction modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudnnSoftmaxMode {
    /// Legacy ONNX Softmax: one flattened trailing instance per leading row.
    Instance,
    /// Opset-13 Softmax: reduce the channel dimension at each outer/inner point.
    Channel,
}

impl CudnnSoftmaxMode {
    fn as_raw(self) -> sys::cudnnSoftmaxMode_t {
        match self {
            Self::Instance => sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
            Self::Channel => sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL,
        }
    }
}

/// cuDNN reductions used by the library-first CUDA kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudnnReduceOp {
    Add,
    Average,
}

impl CudnnReduceOp {
    fn as_raw(self) -> sys::cudnnReduceTensorOp_t {
        match self {
            Self::Add => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            Self::Average => sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AVG,
        }
    }
}

/// Raw EP buffers and element counts for one cuDNN operation.
#[derive(Clone, Copy, Debug)]
pub struct CudnnBufferPair {
    pub input: CUdeviceptr,
    pub output: CUdeviceptr,
    pub input_numel: usize,
    pub output_numel: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReductionScratchSizes {
    workspace_bytes: usize,
    indices_bytes: usize,
}

impl ReductionScratchSizes {
    fn no_indices(workspace_bytes: usize) -> Self {
        Self {
            workspace_bytes,
            indices_bytes: 0,
        }
    }

    fn workspace_allocation_bytes(self) -> usize {
        self.workspace_bytes.max(1)
    }
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
    stream: &'handle Arc<CudaStream>,
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

    /// Execute numerically-stable cuDNN softmax on raw EP device buffers.
    pub fn softmax(
        &self,
        spec: &TensorDescriptorSpec,
        mode: CudnnSoftmaxMode,
        buffers: CudnnBufferPair,
    ) -> Result<()> {
        let descriptor = self.tensor_descriptor(spec)?;
        match &descriptor.inner {
            TensorDescriptorInner::F32(desc) => {
                self.softmax_t(desc, mode, buffers, (1.0f32, 0.0f32))
            }
            TensorDescriptorInner::F16(desc) => self.softmax_t(
                desc,
                mode,
                buffers,
                (f16::from_f32(1.0), f16::from_f32(0.0)),
            ),
            TensorDescriptorInner::Bf16(desc) => self.softmax_t(
                desc,
                mode,
                buffers,
                (bf16::from_f32(1.0), bf16::from_f32(0.0)),
            ),
        }
    }

    fn softmax_t<T: CudnnDataType + Copy>(
        &self,
        descriptor: &TensorDescriptor<T>,
        mode: CudnnSoftmaxMode,
        buffers: CudnnBufferPair,
        scaling: (T, T),
    ) -> Result<()> {
        let softmax = self
            .handle
            .create_softmax::<T>(mode.as_raw())
            .map_err(|e| cudnn_err("creating softmax operation", e))?;
        let op = SoftmaxForward {
            softmax: &softmax,
            x: descriptor,
            y: descriptor,
        };
        let input = RawDevice::<T>::new(buffers.input, buffers.input_numel, self.stream.clone());
        let mut output =
            RawDevice::<T>::new(buffers.output, buffers.output_numel, self.stream.clone());
        // SAFETY: the descriptor dtype/layout matches both raw buffers, which
        // are live EP allocations containing `numel` elements.
        unsafe {
            op.launch(
                scaling,
                sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE,
                &input,
                &mut output,
            )
        }
        .map_err(|e| cudnn_err("cudnnSoftmaxForward", e))
    }

    /// Query scratch space and execute a no-indices cuDNN tensor reduction.
    pub fn reduce(
        &self,
        input_spec: &TensorDescriptorSpec,
        output_spec: &TensorDescriptorSpec,
        op: CudnnReduceOp,
        buffers: CudnnBufferPair,
    ) -> Result<()> {
        let input = self.tensor_descriptor(input_spec)?;
        let output = self.tensor_descriptor(output_spec)?;
        match (&input.inner, &output.inner) {
            (TensorDescriptorInner::F32(a), TensorDescriptorInner::F32(c)) => {
                self.reduce_t(a, c, op, buffers, (1.0f32, 0.0f32))
            }
            (TensorDescriptorInner::F16(a), TensorDescriptorInner::F16(c)) => {
                self.reduce_t(a, c, op, buffers, (f16::from_f32(1.0), f16::from_f32(0.0)))
            }
            (TensorDescriptorInner::Bf16(a), TensorDescriptorInner::Bf16(c)) => self.reduce_t(
                a,
                c,
                op,
                buffers,
                (bf16::from_f32(1.0), bf16::from_f32(0.0)),
            ),
            _ => Err(EpError::KernelFailed(
                "cuda_ep: cuDNN reduction input/output descriptor dtypes differ".into(),
            )),
        }
    }

    fn reduce_t<T: CudnnDataType + Copy>(
        &self,
        a: &TensorDescriptor<T>,
        c: &TensorDescriptor<T>,
        reduce_op: CudnnReduceOp,
        buffers: CudnnBufferPair,
        scaling: (T, T),
    ) -> Result<()> {
        let descriptor: ReductionDescriptor<T, NoIndices> = self
            .handle
            .create_reduction_no_indices::<T>(
                reduce_op.as_raw(),
                sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            )
            .map_err(|e| cudnn_err("creating reduction descriptor", e))?;
        let op = ReduceTensor {
            reduce: &descriptor,
            a,
            c,
        };
        let scratch = ReductionScratchSizes::no_indices(
            op.get_workspace_size()
                .map_err(|e| cudnn_err("cudnnGetReductionWorkspaceSize", e))?,
        );
        debug_assert_eq!(scratch.indices_bytes, 0);
        let mut workspace = self
            .stream
            .alloc_zeros::<u8>(scratch.workspace_allocation_bytes())
            .map_err(|e| driver_err("allocating cuDNN reduction workspace", e))?;
        let input = RawDevice::<T>::new(buffers.input, buffers.input_numel, self.stream.clone());
        let mut output =
            RawDevice::<T>::new(buffers.output, buffers.output_numel, self.stream.clone());
        // SAFETY: descriptors and raw buffers have matching dtypes/layouts;
        // workspace is at least the size returned by cuDNN and indices are off.
        unsafe { op.launch(&mut workspace, scaling, &input, &mut output) }
            .map_err(|e| cudnn_err("cudnnReduceTensor", e))
    }
}

struct RawDevice<T> {
    ptr: CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
    _type: PhantomData<T>,
}

impl<T> RawDevice<T> {
    fn new(ptr: CUdeviceptr, len: usize, stream: Arc<CudaStream>) -> Self {
        Self {
            ptr,
            len,
            stream,
            _type: PhantomData,
        }
    }
}

impl<T> DeviceSlice<T> for RawDevice<T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<T> DevicePtr<T> for RawDevice<T> {
    fn device_ptr<'a>(&'a self, _stream: &'a CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
    }
}

impl<T> DevicePtrMut<T> for RawDevice<T> {
    fn device_ptr_mut<'a>(&'a mut self, _stream: &'a CudaStream) -> (CUdeviceptr, SyncOnDrop<'a>) {
        (self.ptr, SyncOnDrop::Record(None))
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
        operation(CudnnHandle {
            handle,
            stream: &self.stream,
        })
    }

    /// Cheap loader probe used to select an existing non-cuDNN fallback.
    pub fn is_available(&self) -> bool {
        cudnn_library_present()
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
    fn maps_softmax_modes_and_reduce_ops() {
        assert_eq!(
            CudnnSoftmaxMode::Instance.as_raw(),
            sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE
        );
        assert_eq!(
            CudnnSoftmaxMode::Channel.as_raw(),
            sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL
        );
        assert_eq!(
            CudnnReduceOp::Add.as_raw(),
            sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD
        );
        assert_eq!(
            CudnnReduceOp::Average.as_raw(),
            sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_AVG
        );
    }

    #[test]
    fn reduction_workspace_query_result_is_raii_allocatable() {
        let zero = ReductionScratchSizes::no_indices(0);
        assert_eq!(zero.workspace_allocation_bytes(), 1);
        assert_eq!(zero.indices_bytes, 0);
        let nonzero = ReductionScratchSizes::no_indices(4096);
        assert_eq!(nonzero.workspace_allocation_bytes(), 4096);
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
