use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::packed_routed::{AffineGrouped, fail, group_layout, validate};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node, packed_routed::Op};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, GraphDeviceAllocation, cuptr};

const SOURCE: &str = concat!(
    include_str!("packed_kernel_prelude.cuh"),
    include_str!("grouped_matmul.cu")
);

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

    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
        let l = self.config.layout(x, y)?;
        let op = Op::GroupedMatMul { transposed: true };
        if x.iter().any(|t| !t.is_contiguous()) || !y[0].is_contiguous() {
            return Err(fail(
                op,
                "quantized CUDA projection requires contiguous tensors",
            ));
        }
        let scale_count = l.groups * l.n * (l.k / self.config.block);
        if l.routes == 0 && scale_count == 0 {
            return Ok(());
        }
        let flag = GraphDeviceAllocation::allocate(&self.runtime, 4)?;
        let status = flag.ptr();
        // SAFETY: fresh live four-byte result flag.
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

    fn execute(&self, x: &[TensorView], y: &mut [TensorMut]) -> Result<()> {
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
        if l.routes == 0 {
            return Ok(());
        }
        // Keep tensor data on the device; only the four-byte validation result
        // is read back. This kernel is eager-only until it uses the capture latch.
        let flag = GraphDeviceAllocation::allocate(&self.runtime, 4)?;
        let status = flag.ptr();
        // SAFETY: this is a fresh, live four-byte allocation for this invocation.
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
