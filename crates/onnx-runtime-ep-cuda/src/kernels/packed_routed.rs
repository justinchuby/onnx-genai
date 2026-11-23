use std::{ffi::c_void, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::packed_routed::{fail, group_layout, pack_layout, validate};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::{DataType, Node, packed_routed::Op};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, GraphDeviceAllocation, cuptr};

const SOURCE: &str = include_str!("packed_routed.cu");

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
        let has_group = matches!(op, Op::GroupedMatMul { .. });
        let (layout, route) = if has_group {
            let l = group_layout(op, x, &y[0])?;
            if l.routes == 0 {
                return Ok(());
            }
            (None, Some(l))
        } else {
            let l = pack_layout(op, &x[0], &y[0], None)?;
            if l.lanes == 0 && matches!(op, Op::Pack(..)) {
                return Ok(());
            }
            (Some(l), None)
        };
        // Keep tensor data on the device; only the four-byte validation result
        // is read back. This kernel is eager-only until it uses the capture latch.
        let flag = GraphDeviceAllocation::allocate(&self.runtime, 4)?;
        let status = flag.ptr();
        // SAFETY: this is a fresh, live four-byte allocation for this invocation.
        unsafe { self.runtime.htod(&[0; 4], status)? };

        if let Some(l) = layout {
            let unpack = matches!(op, Op::Unpack(_));
            let entry = match (unpack, l.word_bits) {
                (true, 8) => "unpack_u8",
                (true, _) => "unpack_u32",
                (false, 8) => "pack_u8",
                (false, _) => "pack_u32",
            };
            let func = self
                .runtime
                .nvrtc_function("packed_routed", SOURCE, entry)?;
            let count_input = if unpack { ptr(&x[1]) } else { 0 };
            let (lanes, count, words, inner) = (
                l.lanes as u64,
                l.count as u64,
                l.words as u64,
                l.inner as u64,
            );
            let p = l.packing;
            let (group, bits) = (p.group as u64, p.bits as u64);
            let mut launch = stream.launch_builder(&func);
            launch
                .arg(&input)
                .arg(&output)
                .arg(&count_input)
                .arg(&status)
                .arg(&lanes)
                .arg(&count)
                .arg(&words)
                .arg(&inner)
                .arg(&p.radix)
                .arg(&group)
                .arg(&bits)
                .arg(&p.limit);
            // SAFETY: tensor types and extents were validated, and arguments match PACK_ENTRY.
            unsafe { launch.launch(launch_config(l.lanes)) }
                .map_err(|e| driver_err(&format!("launch {entry}"), e))?;
        }
        if let Some(l) = route {
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
                .nvrtc_function("packed_routed", SOURCE, &entry)?;
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
        }
        let mut bytes = [0; 4];
        // SAFETY: the live flag covers four bytes; dtoh drains the compute stream.
        unsafe { self.runtime.dtoh(&mut bytes, status)? };
        let bits = u32::from_ne_bytes(bytes);
        if bits == 0 {
            return Ok(());
        }
        Err(fail(
            op,
            match op {
                Op::Pack(p, _) => format!(
                    "input contains a code outside [0, {}); fix X or the packing attributes",
                    p.radix
                ),
                Op::Unpack(_) if bits == 2 => format!(
                    "count must be nonnegative and equal the output axis extent {}",
                    layout.unwrap().count
                ),
                Op::Unpack(p) => format!(
                    "payload contains a group index outside [0, {}); check the radix and group_size or regenerate X",
                    p.limit
                ),
                Op::GroupedMatMul { .. } => format!(
                    "group_indices contains a value outside [0, {}); provide valid expert IDs",
                    route.unwrap().groups
                ),
            },
        ))
    }
}
