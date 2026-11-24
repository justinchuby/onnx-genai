use std::{
    ffi::c_void,
    sync::{Arc, OnceLock},
};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::packed_routed::{
    AffineGrouped, fail, group_layout, packed_group_layout, validate,
};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMetadata, TensorMut, TensorView,
    WorkspacePreparation, WorkspaceView,
};
use onnx_runtime_ir::{DataType, Node, dequantize_packed::Metadata, packed_routed::Op};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = concat!(
    include_str!("packed_kernel_prelude.cuh"),
    include_str!("grouped_matmul.cu")
);
const PACKED_ENTRY_I32: &str = "packed_grouped_i32";
const PACKED_ENTRY_I64: &str = "packed_grouped_i64";
const PACKED_SOURCE: &str = r#"
template<typename I>
__device__ void packed_grouped(
    const B* x, const B* packed, const B* indices, B* y, unsigned* status,
    U routes, U lanes, U groups, U selections, U k, U n,
    int width, int bytes, int kind)
{
    const U blocks = k / (U)width;
    GRID(i, n == 0 ? routes : lanes) {
        const U route = n == 0 ? i : i / n;
        const I group = read<I>(indices, route);
        if (group < 0 || (U)group >= groups) {
            atomicOr(status, 1u);
            continue;
        }
        if (n == 0) continue;
        const U token = route / selections;
        const U column = i % n;
        const U row = (U)group * n + column;
        float sum = 0.0f;
        for (U j = 0; j < k; ++j) {
            const U block = j / (U)width;
            const B* data = packed + (row * blocks + block) * (U)bytes;
            sum += read<float>(x, token * k + j)
                 * decode_onnx_gguf(data, kind, (int)(j % (U)width));
        }
        write<float>(y, i, sum);
    }
}

#define PACKED_GROUP_ENTRY(I, NAME) \
extern "C" __global__ void NAME( \
    const B* x, const B* packed, const B* indices, B* y, unsigned* status, \
    U routes, U lanes, U groups, U selections, U k, U n, \
    int width, int bytes, int kind) \
{ \
    packed_grouped<I>(x, packed, indices, y, status, routes, lanes, groups, \
                      selections, k, n, width, bytes, kind); \
}
PACKED_GROUP_ENTRY(int, packed_grouped_i32)
PACKED_GROUP_ENTRY(long long, packed_grouped_i64)
"#;

fn packed_source() -> &'static str {
    static SOURCE: OnceLock<String> = OnceLock::new();
    SOURCE.get_or_init(|| {
        let mut source = super::dequantize_packed::decoder_source().to_owned();
        source.push_str(include_str!("packed_kernel_prelude.cuh"));
        source.push_str(PACKED_SOURCE);
        source
    })
}

pub(super) struct Factory(pub Arc<CudaRuntime>);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native {
            op: Op::from_node(node).map_err(EpError::KernelFailed)?,
            runtime: Arc::clone(&self.0),
        }))
    }
}

struct Native {
    op: Op,
    runtime: Arc<CudaRuntime>,
}

pub(super) struct AffineFactory(pub Arc<CudaRuntime>);

impl KernelFactory for AffineFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Affine {
            config: AffineGrouped::from_node(node)?,
            runtime: self.0.clone(),
        }))
    }
}

struct Affine {
    config: AffineGrouped,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for Affine {
    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "grouped projection reads the scale and route validation status on the host",
        )
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Ok(WorkspacePreparation::Ready(
            super::packed_validation::requirement(super::packed_validation::routing_needed(
                inputs.get(2).map(|input| input.shape),
                Some(inputs.get(3).map_or(&[][..], |input| input.shape)),
            )),
        ))
    }

    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(x, y, None)
    }

    fn execute_with_workspace(
        &self,
        x: &[TensorView],
        y: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let l = self.config.layout(x, y)?;
        let op = Op::GroupedMatMul { transposed: true };
        if x.iter().any(|t| !t.is_contiguous()) || !y[0].is_contiguous() {
            return Err(fail(
                op,
                "quantized CUDA projection requires contiguous tensors",
            ));
        }
        // layout() checked the decoded bank element count and its scale shape.
        let scale_count = x[3].numel();
        if !super::packed_validation::routing_needed(Some(x[2].shape), Some(x[3].shape)) {
            return Ok(());
        }
        let status = super::packed_validation::status(workspace, "AffineQuantizedGroupedMatMul")?;
        // SAFETY: the checked caller-owned status region covers four bytes.
        unsafe { self.runtime.htod(&[0; 4], status)? };
        let ptr = |t: &TensorView| cuptr(t.data_ptr::<u8>() as *const c_void);
        let (a, w, idx, scale, zero_points) = (
            ptr(&x[0]),
            ptr(&x[1]),
            ptr(&x[2]),
            ptr(&x[3]),
            x.get(4).map_or(0, ptr),
        );
        let dst = cuptr(y[0].data_ptr_mut::<u8>() as *const c_void);
        let (routes, lanes, groups, selections, k, n, block) = (
            l.routes as u64,
            l.lanes as u64,
            l.groups as u64,
            l.selections as u64,
            l.k as u64,
            l.n as u64,
            self.config.block as u64,
        );
        let zero = i32::from(self.config.zero);
        let entry = if x[2].dtype == DataType::Int32 {
            "affine_grouped_i32"
        } else {
            "affine_grouped_i64"
        };
        let func = self
            .runtime
            .nvrtc_function("grouped_matmul", SOURCE, entry)?;
        let stream = self.runtime.stream();
        let mut launch = stream.launch_builder(&func);
        launch
            .arg(&a)
            .arg(&w)
            .arg(&idx)
            .arg(&scale)
            .arg(&zero_points)
            .arg(&dst)
            .arg(&status)
            .arg(&routes)
            .arg(&lanes)
            .arg(&groups)
            .arg(&selections)
            .arg(&k)
            .arg(&n)
            .arg(&block)
            .arg(&zero);
        // SAFETY: validated extents and ABI, with live allocations through synchronization.
        unsafe { launch.launch(launch_config(l.lanes.max(l.routes).max(scale_count))) }
            .map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        let mut bytes = [0; 4];
        // SAFETY: the live flag covers exactly four bytes.
        unsafe { self.runtime.dtoh(&mut bytes, status)? };
        let status = u32::from_ne_bytes(bytes);
        if status & 2 != 0 {
            return Err(fail(
                op,
                "DequantizeLinear scales must be positive and finite",
            ));
        }
        if status & 1 != 0 {
            return Err(fail(
                op,
                format!(
                    "group_indices contains a value outside [0, {}); provide valid expert IDs",
                    l.groups
                ),
            ));
        }
        Ok(())
    }
}

pub(super) struct PackedFactory(pub Arc<CudaRuntime>);

impl KernelFactory for PackedFactory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Packed {
            metadata: Metadata::from_node(node).map_err(|detail| {
                EpError::KernelFailed(format!("cuda_ep PackedGroupedMatMul: {detail}"))
            })?,
            runtime: self.0.clone(),
        }))
    }
}

struct Packed {
    metadata: Metadata,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for Packed {
    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "packed grouped projection reads route validation status on the host",
        )
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Ok(WorkspacePreparation::Ready(
            super::packed_validation::requirement(super::packed_validation::routing_needed(
                inputs.get(2).map(|input| input.shape),
                None,
            )),
        ))
    }

    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(x, y, None)
    }

    fn execute_with_workspace(
        &self,
        x: &[TensorView],
        y: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let l = packed_group_layout(&self.metadata, x, y)?;
        if x.iter().any(|tensor| !tensor.is_contiguous()) || !y[0].is_contiguous() {
            return Err(EpError::KernelFailed(
                "cuda_ep PackedGroupedMatMul: inputs and output must use contiguous storage".into(),
            ));
        }
        if !super::packed_validation::routing_needed(Some(x[2].shape), None) {
            return Ok(());
        }
        let status = super::packed_validation::status(workspace, "PackedGroupedMatMul")?;
        // SAFETY: the checked caller-owned status region covers four bytes.
        unsafe { self.runtime.htod(&[0; 4], status)? };
        let ptr = |tensor: &TensorView| cuptr(tensor.data_ptr::<u8>() as *const c_void);
        let activation = ptr(&x[0]);
        let packed = ptr(&x[1]);
        let indices = ptr(&x[2]);
        let output = cuptr(y[0].data_ptr_mut::<u8>() as *const c_void);
        let (routes, lanes, groups, selections, k, n) = (
            l.routes as u64,
            l.lanes as u64,
            l.groups as u64,
            l.selections as u64,
            l.k as u64,
            l.n as u64,
        );
        let (width, bytes) = self.metadata.format().block();
        let width = i32::try_from(width).map_err(|_| {
            EpError::KernelFailed("cuda_ep PackedGroupedMatMul: block width exceeds i32".into())
        })?;
        let bytes = i32::try_from(bytes).map_err(|_| {
            EpError::KernelFailed(
                "cuda_ep PackedGroupedMatMul: block byte count exceeds i32".into(),
            )
        })?;
        let kind = super::dequantize_packed::decoder_kind(self.metadata.format());
        let entry = if x[2].dtype == DataType::Int32 {
            PACKED_ENTRY_I32
        } else {
            PACKED_ENTRY_I64
        };
        let function = self
            .runtime
            .nvrtc_function("packed_grouped", packed_source(), entry)?;
        let stream = self.runtime.stream();
        let mut launch = stream.launch_builder(&function);
        launch
            .arg(&activation)
            .arg(&packed)
            .arg(&indices)
            .arg(&output)
            .arg(&status)
            .arg(&routes)
            .arg(&lanes)
            .arg(&groups)
            .arg(&selections)
            .arg(&k)
            .arg(&n)
            .arg(&width)
            .arg(&bytes)
            .arg(&kind);
        // SAFETY: validated contiguous tensors cover all indexed lanes and ABI matches.
        unsafe { launch.launch(launch_config(l.lanes.max(l.routes))) }
            .map_err(|error| driver_err(&format!("launch {entry}"), error))?;
        let mut bytes = [0; 4];
        // SAFETY: the live flag covers exactly four bytes.
        unsafe { self.runtime.dtoh(&mut bytes, status)? };
        if u32::from_ne_bytes(bytes) != 0 {
            return Err(fail(
                Op::GroupedMatMul { transposed: true },
                format!(
                    "group_indices contains a value outside [0, {}); provide valid expert IDs",
                    l.groups
                ),
            ));
        }
        Ok(())
    }
}

fn launch_config(lanes: usize) -> LaunchConfig {
    const BLOCK: u32 = 256;
    LaunchConfig {
        grid_dim: (lanes.div_ceil(BLOCK as usize).clamp(1, 65_535) as u32, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    }
}

impl Kernel for Native {
    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "opset 29 packing and routing currently reads validation status from the GPU; use eager execution",
        )
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _outputs: &[onnx_runtime_ep_api::OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        Ok(WorkspacePreparation::Ready(
            super::packed_validation::requirement(super::packed_validation::routing_needed(
                inputs.get(2).map(|input| input.shape),
                None,
            )),
        ))
    }

    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(x, y, None)
    }

    fn execute_with_workspace(
        &self,
        x: &[TensorView],
        y: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let op = self.op;
        validate(op, x, y)?;
        if x.iter()
            .any(|t| t.dtype != DataType::Undefined && !t.is_contiguous())
            || !y[0].is_contiguous()
        {
            return Err(fail(
                op,
                "CUDA requires contiguous operands; materialize strided tensors first",
            ));
        }
        let ptr = |t: &TensorView| cuptr(t.data_ptr::<u8>() as *const c_void);
        let input = ptr(&x[0]);
        let output = cuptr(y[0].data_ptr_mut::<u8>() as *const c_void);
        let stream = self.runtime.stream();
        let l = group_layout(op, x, &y[0])?;
        if !super::packed_validation::routing_needed(Some(x[2].shape), None) {
            return Ok(());
        }
        // Keep tensor data on the device; only the four-byte validation result
        // is read back. This kernel is eager-only until it uses the capture latch.
        let status = super::packed_validation::status(workspace, op.name())?;
        // SAFETY: the checked caller-owned status region covers four bytes.
        unsafe { self.runtime.htod(&[0; 4], status)? };

        let ty = match x[0].dtype {
            DataType::Float32 => "f32",
            DataType::Float64 => "f64",
            DataType::Float16 | DataType::BFloat16 => {
                self.runtime.require_nvrtc_half_headers(op.name())?;
                if x[0].dtype == DataType::Float16 {
                    "f16"
                } else {
                    "bf16"
                }
            }
            _ => unreachable!("validated floating input"),
        };
        let index_ty = if x[2].dtype == DataType::Int32 {
            "i32"
        } else {
            "i64"
        };
        let entry = format!("grouped_{ty}_{index_ty}");
        let func = self
            .runtime
            .nvrtc_function("grouped_matmul", SOURCE, &entry)?;
        let (weights, indices) = (ptr(&x[1]), ptr(&x[2]));
        let bias = if l.bias { ptr(&x[3]) } else { 0 };
        let (routes, lanes, groups, selections, k, n) = (
            l.routes as u64,
            l.lanes as u64,
            l.groups as u64,
            l.selections as u64,
            l.k as u64,
            l.n as u64,
        );
        let trans_b = i32::from(l.transposed);
        let mut launch = stream.launch_builder(&func);
        launch
            .arg(&input)
            .arg(&weights)
            .arg(&indices)
            .arg(&bias)
            .arg(&output)
            .arg(&status)
            .arg(&routes)
            .arg(&lanes)
            .arg(&groups)
            .arg(&selections)
            .arg(&k)
            .arg(&n)
            .arg(&trans_b);
        // SAFETY: tensor types and extents were validated, and arguments match GROUP_ENTRY.
        unsafe { launch.launch(launch_config(l.lanes.max(l.routes))) }
            .map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        let mut bytes = [0; 4];
        // SAFETY: the live flag covers four bytes; dtoh drains the compute stream.
        unsafe { self.runtime.dtoh(&mut bytes, status)? };
        let bits = u32::from_ne_bytes(bytes);
        if bits == 0 {
            return Ok(());
        }
        Err(fail(
            op,
            format!(
                "group_indices contains a value outside [0, {}); provide valid expert IDs",
                l.groups
            ),
        ))
    }
}
