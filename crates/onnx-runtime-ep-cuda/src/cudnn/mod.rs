//! cuDNN handle and tensor-descriptor foundation.
//!
//! The handle is created lazily on first cuDNN use, reuses the CUDA EP's
//! existing stream/context, and is serialized because cuDNN handles are not
//! safe for concurrent host-thread use. cudarc owns all native resources, so
//! handles and descriptors are destroyed through its RAII wrappers.

use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use cudarc::cudnn::{
    ConvBiasActivationForward, ConvForward, Cudnn, CudnnDataType, PoolingForward, SoftmaxForward,
    TensorDescriptor, result, sys,
};
use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{CudaStream, DevicePtr, DevicePtrMut, DeviceSlice, SyncOnDrop};
use half::{bf16, f16};
use onnx_runtime_ep_api::{
    EpError, Result, WorkspaceLifetime, WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::DataType;
use onnx_runtime_memory_governor::MemoryRole;

use crate::dynamic_library::{CudaLibrary, is_available};
use crate::error::{cudnn_err, cudnn_unavailable, driver_err};

pub const WORKSPACE_ALIGNMENT: usize = 256;

pub const fn governed_workspace_requirement(bytes: usize) -> WorkspaceRequirement {
    if bytes == 0 {
        WorkspaceRequirement::NONE
    } else {
        WorkspaceRequirement {
            bytes: bytes as u64,
            alignment: WORKSPACE_ALIGNMENT,
            lifetime: WorkspaceLifetime::SessionPersistent,
            role: MemoryRole::Workspace { step_scoped: false },
        }
    }
}

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

/// Validated 2-D NCHW convolution geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudnnConvSpec {
    pub dtype: CudnnTensorType,
    pub input_dims: [i32; 4],
    pub input_strides: [i32; 4],
    pub filter_dims: [i32; 4],
    pub output_dims: [i32; 4],
    pub output_strides: [i32; 4],
    pub pads: [i32; 2],
    pub strides: [i32; 2],
    pub dilations: [i32; 2],
    pub groups: i32,
}

/// Raw EP buffers for one cuDNN convolution.
#[derive(Clone, Copy, Debug)]
pub struct CudnnConvBuffers {
    pub input: CUdeviceptr,
    pub filter: CUdeviceptr,
    pub bias: Option<CUdeviceptr>,
    pub output: CUdeviceptr,
    pub input_numel: usize,
    pub filter_numel: usize,
    pub bias_numel: usize,
    pub output_numel: usize,
}

/// cuDNN pooling mode selected from the ONNX operator and its attributes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CudnnPoolingMode {
    Max,
    AverageIncludePadding,
    AverageExcludePadding,
}

impl CudnnPoolingMode {
    fn as_raw(self) -> sys::cudnnPoolingMode_t {
        match self {
            Self::Max => sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            Self::AverageIncludePadding => {
                sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
            }
            Self::AverageExcludePadding => {
                sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
            }
        }
    }
}

/// Validated 2-D NCHW pooling geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudnnPoolingSpec {
    pub dtype: CudnnTensorType,
    pub input_dims: [i32; 4],
    pub input_strides: [i32; 4],
    pub output_dims: [i32; 4],
    pub output_strides: [i32; 4],
    pub window: [i32; 2],
    pub pads: [i32; 2],
    pub strides: [i32; 2],
    pub mode: CudnnPoolingMode,
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

fn governed_workspace_ptr(
    workspace: Option<WorkspaceView>,
    required: usize,
    op: &str,
) -> Result<CUdeviceptr> {
    if required == 0 {
        return Ok(0);
    }
    let workspace = workspace.ok_or_else(|| {
        EpError::KernelFailed(format!(
            "cuda_ep {op}: prepared cuDNN workspace requires {required} bytes, but none was supplied"
        ))
    })?;
    if workspace.bytes() < required {
        return Err(EpError::KernelFailed(format!(
            "cuda_ep {op}: prepared cuDNN workspace requires {required} bytes, supplied {}",
            workspace.bytes()
        )));
    }
    Ok(workspace.ptr().0 as CUdeviceptr)
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
    /// Raw cuDNN handle, stream-bound, used only by the f32-compute-type reduce
    /// path (see [`CudnnHandle::reduce_with_workspace`]). It aliases the same CUDA
    /// stream as `handle` and is serialized by the backend's handle mutex.
    reduce_handle: sys::cudnnHandle_t,
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

    /// Build an independent cuDNN reduce plan for one signature.
    ///
    /// The executor owns the actual device workspace through the provider's
    /// governed allocator. Callers stage this host-only plan until the matching
    /// execution succeeds, then publish it as part of their immutable warmed
    /// snapshot.
    pub fn prepare_reduce(
        &self,
        input_spec: &TensorDescriptorSpec,
        output_spec: &TensorDescriptorSpec,
        op: CudnnReduceOp,
    ) -> Result<CudnnReduceCache> {
        if input_spec.dtype() != output_spec.dtype() {
            return Err(EpError::KernelFailed(
                "cuda_ep: cuDNN reduction input/output descriptor dtypes differ".into(),
            ));
        }
        if input_spec.dtype() == CudnnTensorType::Bf16 {
            return Err(EpError::KernelFailed(
                "cuda_ep: cuDNN cannot reduce bf16 tensors; route bf16 reductions \
                 to the NVRTC block-reduction kernel"
                    .into(),
            ));
        }

        let key = CudnnReduceKey {
            op,
            input: input_spec.clone(),
            output: output_spec.clone(),
        };
        let a_desc = RawTensorDescriptor::new(input_spec)?;
        let c_desc = RawTensorDescriptor::new(output_spec)?;
        let reduce = RawReductionDescriptor::new_f32_comp(op)?;

        // SAFETY: the handle is live and stream-bound; the tensor and reduction
        // descriptors were just created and are still alive.
        let workspace_bytes = unsafe {
            result::get_reduction_workspace_size(self.reduce_handle, reduce.0, a_desc.0, c_desc.0)
        }
        .map_err(|e| cudnn_err("cudnnGetReductionWorkspaceSize", e))?;

        Ok(CudnnReduceCache {
            key: Some(key),
            input_desc: Some(a_desc),
            output_desc: Some(c_desc),
            reduce_desc: Some(reduce),
            workspace_bytes,
        })
    }

    /// Return the exact workspace bytes from an already-prepared immutable
    /// reduction plan, rejecting any attempt to pair it with another signature.
    pub fn reduce_workspace_bytes(
        &self,
        cache: &CudnnReduceCache,
        input_spec: &TensorDescriptorSpec,
        output_spec: &TensorDescriptorSpec,
        op: CudnnReduceOp,
    ) -> Result<usize> {
        if !cache.matches(input_spec, output_spec, op) {
            return Err(EpError::KernelFailed(
                "cuda_ep: prepared cuDNN reduction plan does not match the requested signature; \
                 prepare the exact reduction before execution"
                    .into(),
            ));
        }
        Ok(cache.workspace_bytes)
    }

    /// CUDA-graph-capture-eligible cuDNN reduce that reuses cached descriptors
    /// and an executor-prepared device workspace across calls with the same
    /// signature.
    ///
    /// A per-call `cudnnGetReductionWorkspaceSize` plus a device allocation and
    /// the reduce kernel's trailing `synchronize()` are what made the old path
    /// non-capturable. The queried workspace size now flows through
    /// `Kernel::workspace_requirement*` to the executor's prepared persistent
    /// workspace, so this hot path only verifies the warmed signature and
    /// enqueues `cudnnReduceTensor`.
    ///
    /// This drives the raw cuDNN FFI with an **f32 compute type and f32
    /// alpha/beta** for both f32 and f16 I/O. For f16 that is the ONNX
    /// accumulate-in-f32 semantics (`cudnnReduceTensor` rejects a half
    /// `reduceTensorCompType` — `CUDNN_STATUS_NOT_SUPPORTED` — and requires
    /// `CUDNN_DATA_FLOAT` accumulation for half I/O, which is exactly ONNX's
    /// accumulate-in-f32-then-cast-back rule); for f32 it is byte-identical to
    /// the type-coupled safe reduce (f32 comp type, `alpha = 1`, `beta = 0`,
    /// same descriptors). bf16 is rejected here — cuDNN cannot reduce it — and
    /// is routed to the NVRTC kernel by the caller.
    ///
    /// The caller must **not** synchronize after this returns while a capture is
    /// recording; in eager mode it keeps its usual trailing sync.
    #[allow(clippy::too_many_arguments)]
    pub fn reduce_with_workspace(
        &self,
        cache: &CudnnReduceCache,
        input_spec: &TensorDescriptorSpec,
        output_spec: &TensorDescriptorSpec,
        op: CudnnReduceOp,
        buffers: CudnnBufferPair,
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let workspace_bytes = self.reduce_workspace_bytes(cache, input_spec, output_spec, op)?;

        let reduce = cache
            .reduce_desc
            .as_ref()
            .expect("reduce descriptor cached");
        let a_desc = cache.input_desc.as_ref().expect("input descriptor cached");
        let c_desc = cache
            .output_desc
            .as_ref()
            .expect("output descriptor cached");

        let alpha = 1.0f32;
        let beta = 0.0f32;
        let workspace = governed_workspace_ptr(workspace, workspace_bytes, "Reduce")?;
        // SAFETY: the cached tensor descriptors describe the f16/f32 device
        // buffers, the cached reduction descriptor uses an f32 comp type (so
        // alpha/beta are f32), and the executor-prepared workspace is at least
        // the queried size for this signature with indices disabled.
        unsafe {
            // cuDNN allocates and synchronizes internally; gate the whole
            // invocation. See `onnx_runtime_cuda_memory::capture_gate`.
            let _section = onnx_runtime_cuda_memory::capture_gate::synchronizing_section();
            result::reduce_tensor(
                self.reduce_handle,
                reduce.0,
                std::ptr::null_mut(),
                0,
                workspace as *mut std::ffi::c_void,
                workspace_bytes,
                (&alpha as *const f32).cast::<std::ffi::c_void>(),
                a_desc.0,
                buffers.input as *const std::ffi::c_void,
                (&beta as *const f32).cast::<std::ffi::c_void>(),
                c_desc.0,
                buffers.output as *mut std::ffi::c_void,
            )
        }
        .map_err(|e| cudnn_err("cudnnReduceTensor", e))
    }

    /// Build an independent cuDNN convolution plan for one signature.
    pub fn prepare_conv(&self, spec: &CudnnConvSpec, has_bias: bool) -> Result<CudnnConvPlanCache> {
        let key = CudnnConvKey {
            spec: spec.clone(),
            has_bias,
        };
        let (algo, workspace_bytes) = match spec.dtype {
            CudnnTensorType::F32 => self.conv_plan_t::<f32>(spec, has_bias)?,
            CudnnTensorType::F16 => self.conv_plan_t::<f16>(spec, has_bias)?,
            CudnnTensorType::Bf16 => self.conv_plan_t::<bf16>(spec, has_bias)?,
        };
        Ok(CudnnConvPlanCache {
            key: Some(key),
            algo: Some(algo),
            workspace_bytes,
        })
    }

    /// Return the workspace bytes from an immutable matching convolution plan.
    pub fn conv_workspace_bytes(
        &self,
        cache: &CudnnConvPlanCache,
        spec: &CudnnConvSpec,
        has_bias: bool,
    ) -> Result<usize> {
        if !cache.matches(spec, has_bias) {
            return Err(EpError::KernelFailed(
                "cuda_ep: prepared cuDNN convolution plan does not match the requested signature; \
                 prepare the exact convolution before execution"
                    .into(),
            ));
        }
        Ok(cache.workspace_bytes)
    }

    /// Execute a 2-D NCHW convolution with an optional fused channel bias,
    /// consuming an executor-prepared workspace selected through
    /// [`CudnnHandle::conv_workspace_bytes`].
    pub fn conv2d(
        &self,
        cache: &CudnnConvPlanCache,
        spec: &CudnnConvSpec,
        buffers: CudnnConvBuffers,
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let workspace_bytes = self.conv_workspace_bytes(cache, spec, buffers.bias.is_some())?;
        let algo = cache.algo.ok_or_else(|| {
            EpError::KernelFailed("cuda_ep: missing cuDNN convolution plan".into())
        })?;
        match spec.dtype {
            CudnnTensorType::F32 => {
                self.conv2d_t::<f32>(spec, buffers, workspace, workspace_bytes, algo, (1.0, 0.0))
            }
            CudnnTensorType::F16 => self.conv2d_t::<f16>(
                spec,
                buffers,
                workspace,
                workspace_bytes,
                algo,
                (f16::from_f32(1.0), f16::from_f32(0.0)),
            ),
            CudnnTensorType::Bf16 => self.conv2d_t::<bf16>(
                spec,
                buffers,
                workspace,
                workspace_bytes,
                algo,
                (bf16::from_f32(1.0), bf16::from_f32(0.0)),
            ),
        }
    }

    fn conv_plan_t<T: CudnnDataType + Copy>(
        &self,
        spec: &CudnnConvSpec,
        has_bias: bool,
    ) -> Result<(sys::cudnnConvolutionFwdAlgo_t, usize)> {
        let x_desc = self
            .handle
            .create_4d_tensor_ex::<T>(spec.input_dims, spec.input_strides)
            .map_err(|e| cudnn_err("creating convolution input descriptor", e))?;
        let w_desc = self
            .handle
            .create_4d_filter::<T>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                spec.filter_dims,
            )
            .map_err(|e| cudnn_err("creating convolution filter descriptor", e))?;
        let y_desc = self
            .handle
            .create_4d_tensor_ex::<T>(spec.output_dims, spec.output_strides)
            .map_err(|e| cudnn_err("creating convolution output descriptor", e))?;
        let mut conv_desc = self
            .handle
            .create_conv2d::<f32>(
                spec.pads,
                spec.strides,
                spec.dilations,
                sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .map_err(|e| cudnn_err("creating convolution descriptor", e))?;
        conv_desc
            .set_group_count(spec.groups)
            .map_err(|e| cudnn_err("cudnnSetConvolutionGroupCount", e))?;
        conv_desc
            .set_math_type(match spec.dtype {
                CudnnTensorType::F32 => sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
                CudnnTensorType::F16 | CudnnTensorType::Bf16 => {
                    sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH
                }
            })
            .map_err(|e| cudnn_err("cudnnSetConvolutionMathType", e))?;
        let op = ConvForward {
            conv: &conv_desc,
            x: &x_desc,
            w: &w_desc,
            y: &y_desc,
        };
        let algo = if has_bias {
            sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
        } else {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| op.pick_algorithm()))
                .map_err(|_| {
                    EpError::KernelFailed(
                        "cuda_ep Conv: cuDNN forward algorithm selection failed or returned no \
                         usable algorithm"
                            .into(),
                    )
                })?
                .map_err(|e| cudnn_err("cudnnGetConvolutionForwardAlgorithm_v7", e))?
        };
        let workspace_bytes = op
            .get_workspace_size(algo)
            .map_err(|e| cudnn_err("cudnnGetConvolutionForwardWorkspaceSize", e))?;
        Ok((algo, workspace_bytes))
    }

    fn conv2d_t<T: CudnnDataType + Copy>(
        &self,
        spec: &CudnnConvSpec,
        buffers: CudnnConvBuffers,
        workspace: Option<WorkspaceView>,
        workspace_bytes: usize,
        algo: sys::cudnnConvolutionFwdAlgo_t,
        scaling: (T, T),
    ) -> Result<()> {
        let x_desc = self
            .handle
            .create_4d_tensor_ex::<T>(spec.input_dims, spec.input_strides)
            .map_err(|e| cudnn_err("creating convolution input descriptor", e))?;
        let w_desc = self
            .handle
            .create_4d_filter::<T>(
                sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                spec.filter_dims,
            )
            .map_err(|e| cudnn_err("creating convolution filter descriptor", e))?;
        let y_desc = self
            .handle
            .create_4d_tensor_ex::<T>(spec.output_dims, spec.output_strides)
            .map_err(|e| cudnn_err("creating convolution output descriptor", e))?;
        // cuDNN recommends fp32 accumulation for fp16/bf16 storage. Keep f32 in
        // default math mode so the kernel does not silently opt into TF32.
        let mut conv_desc = self
            .handle
            .create_conv2d::<f32>(
                spec.pads,
                spec.strides,
                spec.dilations,
                sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .map_err(|e| cudnn_err("creating convolution descriptor", e))?;
        conv_desc
            .set_group_count(spec.groups)
            .map_err(|e| cudnn_err("cudnnSetConvolutionGroupCount", e))?;
        conv_desc
            .set_math_type(match spec.dtype {
                CudnnTensorType::F32 => sys::cudnnMathType_t::CUDNN_DEFAULT_MATH,
                CudnnTensorType::F16 | CudnnTensorType::Bf16 => {
                    sys::cudnnMathType_t::CUDNN_TENSOR_OP_MATH
                }
            })
            .map_err(|e| cudnn_err("cudnnSetConvolutionMathType", e))?;

        let op = ConvForward {
            conv: &conv_desc,
            x: &x_desc,
            w: &w_desc,
            y: &y_desc,
        };
        let workspace_ptr = governed_workspace_ptr(workspace, workspace_bytes, "Conv")?;
        let mut workspace = (workspace_bytes != 0)
            .then(|| RawDevice::<u8>::new(workspace_ptr, workspace_bytes, self.stream.clone()));
        let input = RawDevice::<T>::new(buffers.input, buffers.input_numel, self.stream.clone());
        let filter = RawDevice::<T>::new(buffers.filter, buffers.filter_numel, self.stream.clone());
        let mut output =
            RawDevice::<T>::new(buffers.output, buffers.output_numel, self.stream.clone());

        if let Some(bias_ptr) = buffers.bias {
            let bias_desc = self
                .handle
                .create_4d_tensor::<T>(
                    sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    [1, spec.output_dims[1], 1, 1],
                )
                .map_err(|e| cudnn_err("creating convolution bias descriptor", e))?;
            let activation = self
                .handle
                .create_activation::<T>(
                    sys::cudnnActivationMode_t::CUDNN_ACTIVATION_IDENTITY,
                    sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                    0.0,
                )
                .map_err(|e| cudnn_err("creating convolution identity activation", e))?;
            let fused = ConvBiasActivationForward {
                conv: &conv_desc,
                act: &activation,
                x: &x_desc,
                w: &w_desc,
                z: &y_desc,
                bias: &bias_desc,
                y: &y_desc,
            };
            let z = RawDevice::<T>::new(buffers.output, buffers.output_numel, self.stream.clone());
            let bias = RawDevice::<T>::new(bias_ptr, buffers.bias_numel, self.stream.clone());
            // SAFETY: all descriptors match live EP allocations. `z` aliases
            // `y`, but alpha2 is zero, so the fused residual term is disabled.
            unsafe {
                fused.launch(
                    algo,
                    workspace.as_mut(),
                    scaling,
                    &input,
                    &filter,
                    &z,
                    &bias,
                    &mut output,
                )
            }
            .map_err(|e| cudnn_err("cudnnConvolutionBiasActivationForward", e))
        } else {
            // SAFETY: descriptors and raw buffers have matching dtypes/layouts;
            // workspace is at least the size returned by cuDNN.
            unsafe {
                op.launch(
                    algo,
                    workspace.as_mut(),
                    scaling,
                    &input,
                    &filter,
                    &mut output,
                )
            }
            .map_err(|e| cudnn_err("cudnnConvolutionForward", e))
        }
    }

    /// Execute 2-D NCHW pooling using cuDNN's descriptor-based forward API.
    pub fn pool2d(&self, spec: &CudnnPoolingSpec, buffers: CudnnBufferPair) -> Result<()> {
        match spec.dtype {
            CudnnTensorType::F32 => self.pool2d_t::<f32>(spec, buffers, (1.0f32, 0.0f32)),
            CudnnTensorType::F16 => {
                self.pool2d_t::<f16>(spec, buffers, (f16::from_f32(1.0), f16::from_f32(0.0)))
            }
            CudnnTensorType::Bf16 => {
                self.pool2d_t::<bf16>(spec, buffers, (bf16::from_f32(1.0), bf16::from_f32(0.0)))
            }
        }
    }

    fn pool2d_t<T: CudnnDataType + Copy>(
        &self,
        spec: &CudnnPoolingSpec,
        buffers: CudnnBufferPair,
        scaling: (T, T),
    ) -> Result<()> {
        let input = self
            .handle
            .create_4d_tensor_ex::<T>(spec.input_dims, spec.input_strides)
            .map_err(|e| cudnn_err("creating pooling input descriptor", e))?;
        let output = self
            .handle
            .create_4d_tensor_ex::<T>(spec.output_dims, spec.output_strides)
            .map_err(|e| cudnn_err("creating pooling output descriptor", e))?;
        let pooling = self
            .handle
            .create_poolingnd::<T>(
                &spec.window,
                &spec.pads,
                &spec.strides,
                spec.mode.as_raw(),
                sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            )
            .map_err(|e| {
                cudnn_err(
                    "cudnnCreatePoolingDescriptor / cudnnSetPoolingNdDescriptor",
                    e,
                )
            })?;
        let op = PoolingForward {
            pooling: &pooling,
            x: &input,
            y: &output,
        };
        let input = RawDevice::<T>::new(buffers.input, buffers.input_numel, self.stream.clone());
        let mut output =
            RawDevice::<T>::new(buffers.output, buffers.output_numel, self.stream.clone());
        // SAFETY: the descriptors exactly describe the live EP allocations.
        unsafe { op.launch(scaling, &input, &mut output) }
            .map_err(|e| cudnn_err("cudnnPoolingForward", e))
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
    /// Lazily created raw handle for the half/bf16 reduce path. Created once and
    /// reused (not per call) so the f32-comp reduce adds no per-op handle cost.
    reduce_handle: Mutex<Option<RawReduceHandle>>,
}

/// An owned raw cuDNN handle, destroyed on drop.
///
/// Used only by the half/bf16 reduce path, which needs an f32 compute type that
/// cudarc's type-coupled safe reduce API cannot express while keeping half I/O.
struct RawReduceHandle(sys::cudnnHandle_t);

// SAFETY: like `Cudnn`, a raw cuDNN handle is not safe for concurrent use, but
// every access is serialized by `CudnnBackend`'s handle mutexes and runs on the
// thread bound to the owning CUDA context.
unsafe impl Send for RawReduceHandle {}

impl Drop for RawReduceHandle {
    fn drop(&mut self) {
        // SAFETY: the handle was created by `result::create_handle` and is
        // destroyed exactly once here.
        unsafe {
            let _ = result::destroy_handle(self.0);
        }
    }
}

/// An owned raw cuDNN tensor descriptor, destroyed on drop.
struct RawTensorDescriptor(sys::cudnnTensorDescriptor_t);

impl std::fmt::Debug for RawTensorDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawTensorDescriptor")
            .finish_non_exhaustive()
    }
}

impl RawTensorDescriptor {
    fn new(spec: &TensorDescriptorSpec) -> Result<Self> {
        let desc = result::create_tensor_descriptor()
            .map_err(|e| cudnn_err("creating tensor descriptor", e))?;
        let guard = Self(desc);
        // SAFETY: `desc` was just created; `dims`/`strides` are validated i32s
        // of equal, rank-padded length.
        unsafe {
            result::set_tensornd_descriptor(
                desc,
                spec.dtype().as_raw(),
                spec.dims().len() as std::ffi::c_int,
                spec.dims().as_ptr(),
                spec.strides().as_ptr(),
            )
        }
        .map_err(|e| cudnn_err("setting tensor descriptor", e))?;
        Ok(guard)
    }
}

impl Drop for RawTensorDescriptor {
    fn drop(&mut self) {
        // SAFETY: the descriptor is live and destroyed exactly once here.
        unsafe {
            let _ = result::destroy_tensor_descriptor(self.0);
        }
    }
}

/// An owned raw cuDNN reduction descriptor with an f32 compute type.
struct RawReductionDescriptor(sys::cudnnReduceTensorDescriptor_t);

impl std::fmt::Debug for RawReductionDescriptor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawReductionDescriptor")
            .finish_non_exhaustive()
    }
}

impl RawReductionDescriptor {
    fn new_f32_comp(op: CudnnReduceOp) -> Result<Self> {
        let desc = result::create_reduce_tensor_descriptor()
            .map_err(|e| cudnn_err("creating reduction descriptor", e))?;
        let guard = Self(desc);
        // SAFETY: `desc` was just created; the f32 comp type and no-indices mode
        // are valid for any half/bf16 reduction.
        unsafe {
            result::set_reduce_tensor_descriptor(
                desc,
                op.as_raw(),
                sys::cudnnDataType_t::CUDNN_DATA_FLOAT,
                sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                sys::cudnnReduceTensorIndices_t::CUDNN_REDUCE_TENSOR_NO_INDICES,
                sys::cudnnIndicesType_t::CUDNN_32BIT_INDICES,
            )
        }
        .map_err(|e| cudnn_err("setting reduction descriptor", e))?;
        Ok(guard)
    }
}

impl Drop for RawReductionDescriptor {
    fn drop(&mut self) {
        // SAFETY: the descriptor is live and destroyed exactly once here.
        unsafe {
            let _ = result::destroy_reduce_tensor_descriptor(self.0);
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct CudnnConvKey {
    spec: CudnnConvSpec,
    has_bias: bool,
}

#[derive(Clone, Debug, Default)]
pub struct CudnnConvPlanCache {
    key: Option<CudnnConvKey>,
    algo: Option<sys::cudnnConvolutionFwdAlgo_t>,
    workspace_bytes: usize,
}

impl CudnnConvPlanCache {
    pub(crate) fn matches(&self, spec: &CudnnConvSpec, has_bias: bool) -> bool {
        self.key.as_ref()
            == Some(&CudnnConvKey {
                spec: spec.clone(),
                has_bias,
            })
    }

    pub(crate) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

/// Signature identifying a cached cuDNN reduce: the op plus the input and output
/// descriptor specs (dtype, padded dims, strides). Axes/`keepdims` are fully
/// captured by the output spec, so a change in any of them is a cache miss.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CudnnReduceKey {
    op: CudnnReduceOp,
    input: TensorDescriptorSpec,
    output: TensorDescriptorSpec,
}

/// Per-reduce-kernel cache of cuDNN descriptors and the exact workspace bytes
/// for the warmed signature. The executor owns the actual device workspace.
/// One instance is owned per `ReduceKernel` and serialized behind that kernel's
/// mutex.
#[derive(Debug)]
pub struct CudnnReduceCache {
    key: Option<CudnnReduceKey>,
    input_desc: Option<RawTensorDescriptor>,
    output_desc: Option<RawTensorDescriptor>,
    reduce_desc: Option<RawReductionDescriptor>,
    workspace_bytes: usize,
}

// SAFETY: cuDNN descriptors are not safe for concurrent use, but every access
// is serialized behind the owning kernel's mutex and runs on the thread bound
// to the owning CUDA context.
unsafe impl Send for CudnnReduceCache {}

impl CudnnReduceCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            key: None,
            input_desc: None,
            output_desc: None,
            reduce_desc: None,
            workspace_bytes: 0,
        }
    }

    fn matches(
        &self,
        input_spec: &TensorDescriptorSpec,
        output_spec: &TensorDescriptorSpec,
        op: CudnnReduceOp,
    ) -> bool {
        self.key.as_ref()
            == Some(&CudnnReduceKey {
                op,
                input: input_spec.clone(),
                output: output_spec.clone(),
            })
    }

    /// Exact executor-owned workspace size queried for this plan.
    pub(crate) fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

impl Default for CudnnReduceCache {
    fn default() -> Self {
        Self::new()
    }
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
            reduce_handle: Mutex::new(None),
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

        let mut reduce_handle = self.reduce_handle.lock().map_err(|_| {
            EpError::KernelFailed("cuda_ep: cuDNN reduce handle mutex was poisoned".into())
        })?;
        if reduce_handle.is_none() {
            *reduce_handle = Some(self.create_reduce_handle()?);
        }
        let reduce_handle = reduce_handle
            .as_ref()
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep: cuDNN reduce handle initialization produced no handle".into(),
                )
            })?
            .0;

        operation(CudnnHandle {
            handle,
            stream: &self.stream,
            reduce_handle,
        })
    }

    /// Create a raw cuDNN handle bound to the EP's compute stream, mirroring
    /// what cudarc's `Cudnn::new` does for the safe handle.
    fn create_reduce_handle(&self) -> Result<RawReduceHandle> {
        let handle = initialize_cudnn(result::create_handle)?;
        let guard = RawReduceHandle(handle);
        // SAFETY: `handle` was just created; the stream outlives the backend and
        // shares the context already bound to this thread.
        unsafe { result::set_stream(handle, self.stream.cu_stream() as sys::cudaStream_t) }
            .map_err(|e| cudnn_err("cudnnSetStream", e))?;
        Ok(guard)
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
        let reduce_handle = self
            .reduce_handle
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if handle.is_some() || reduce_handle.is_some() {
            let _ = self.stream.context().bind_to_thread();
            handle.take();
            reduce_handle.take();
        }
    }
}

fn cudnn_library_present() -> bool {
    is_available(CudaLibrary::Cudnn)
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
        assert_eq!(
            CudnnPoolingMode::Max.as_raw(),
            sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX
        );
        assert_eq!(
            CudnnPoolingMode::AverageIncludePadding.as_raw(),
            sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
        );
        assert_eq!(
            CudnnPoolingMode::AverageExcludePadding.as_raw(),
            sys::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
        );
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
