use std::sync::Arc;

use cudarc::driver::{DeviceRepr, LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::deform_conv::{Attributes, validate_types};
use onnx_runtime_ir::{DataType, Node, Shape};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

const SOURCE: &str = include_str!("deform_conv.cu");
const MAX_RANK: usize = 32;

#[derive(Clone, Copy, Default)]
#[repr(C)]
struct Axis {
    input: u64,
    output: u64,
    kernel: u64,
    stride: u64,
    dilation: u64,
    pad: u64,
}

#[repr(C)]
struct Params {
    rank: u64,
    batches: u64,
    channels: u64,
    cg: u64,
    mg: u64,
    co: u64,
    offset_cg: u64,
    mask_c: u64,
    kv: u64,
    in_size: u64,
    out_size: u64,
    axis: [Axis; MAX_RANK],
}

// SAFETY: identical to the CUDA parameter struct: consecutive 64-bit integers,
// no padding or references. The driver copies it by value at launch.
unsafe impl DeviceRepr for Params {}

pub struct Factory {
    pub runtime: Arc<CudaRuntime>,
    pub version: u64,
}

struct DeformConv {
    runtime: Arc<CudaRuntime>,
    attrs: Attributes,
    version: u64,
}

fn fail(detail: String) -> EpError {
    EpError::KernelFailed(format!("CUDA DeformConv: {detail}"))
}

fn check_rank(rank: usize) -> std::result::Result<(), String> {
    if rank > MAX_RANK {
        return Err(format!(
            "CUDA supports at most {MAX_RANK} spatial axes, got {rank}"
        ));
    }
    Ok(())
}

pub(crate) fn unsupported_reason(
    node: &Node,
    opset: u64,
    shapes: &[Shape],
    dtypes: &[DataType],
) -> Option<String> {
    onnx_runtime_ir::deform_conv::validate_claim(node, opset, shapes, dtypes)
        .and_then(|_| {
            shapes
                .first()
                .map_or(Ok(()), |x| check_rank(x.len().saturating_sub(2)))
        })
        .err()
        .map(|r| format!("DeformConv: {r}"))
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        if let Some(x) = shapes.first() {
            check_rank(x.len().saturating_sub(2)).map_err(fail)?;
        }
        Ok(Box::new(DeformConv {
            attrs: Attributes::parse(node).map_err(fail)?,
            runtime: self.runtime.clone(),
            version: self.version,
        }))
    }
}

impl Kernel for DeformConv {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(3..=5).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(fail(
                "expected X, W, offset, optional B and mask, and Y".into(),
            ));
        }
        let slots = std::array::from_fn(|i| inputs.get(i).filter(|v| !v.is_absent()));
        let y = &mut outputs[0];
        validate_types(
            self.version,
            slots.map(|s| s.map(|t| t.dtype)),
            Some(y.dtype),
        )
        .map_err(fail)?;
        let g = self
            .attrs
            .resolve(slots.map(|s| s.map(|t| t.shape)))
            .map_err(fail)?;
        check_rank(g.axes.len()).map_err(fail)?;
        if y.shape != g.output {
            return Err(fail(format!(
                "Y shape {:?}, expected {:?}",
                y.shape, g.output
            )));
        }
        if slots.iter().flatten().any(|v| !v.is_contiguous()) || !y.is_contiguous() {
            return Err(fail(
                "requires contiguous X, W, offset, B, mask and Y".into(),
            ));
        }
        if g.output.contains(&0) {
            return Ok(());
        }
        let (tag, half) = match y.dtype {
            DataType::Float32 => ("f32", false),
            DataType::Float64 => ("f64", false),
            DataType::Float16 => ("f16", true),
            DataType::BFloat16 => ("bf16", true),
            other => return Err(fail(format!("unsupported dtype {other:?}"))),
        };
        let (suffix, tile) = match (g.axes.len(), g.output_per_group) {
            (2, 1) => ("2d_single", 1),
            (2, _) => ("2d", 8),
            _ => ("nd", 4),
        };
        let entry = format!("deform_{tag}_{suffix}");
        let function = if half {
            self.runtime.require_nvrtc_half_headers("DeformConv")?;
            self.runtime.nvrtc_function(
                "deform_conv_half",
                concat!(
                    "#define DEFORM_CONV_HALF_TYPES\n",
                    include_str!("deform_conv.cu")
                ),
                &entry,
            )?
        } else {
            self.runtime
                .nvrtc_function("deform_conv_float", SOURCE, &entry)?
        };
        let mut p = Params {
            rank: g.axes.len() as u64,
            batches: g.output[0] as u64,
            channels: g.input_channels as u64,
            cg: g.input_per_group as u64,
            mg: g.output_per_group as u64,
            co: g.output[1] as u64,
            offset_cg: g.input_per_offset_group as u64,
            mask_c: g.mask_channels as u64,
            kv: g.kernel_spatial as u64,
            in_size: g.input_spatial as u64,
            out_size: g.output_spatial as u64,
            axis: [Axis::default(); MAX_RANK],
        };
        for (dst, a) in p.axis.iter_mut().zip(g.axes) {
            *dst = Axis {
                input: a.input as u64,
                output: a.output as u64,
                kernel: a.kernel as u64,
                stride: a.stride as u64,
                dilation: a.dilation as u64,
                pad: a.pad as u64,
            };
        }
        let [x, w, offset, b, mask] =
            slots.map(|s| s.map_or(0, |v| cuptr(v.data_ptr::<u8>().cast())));
        let y = cuptr(y.data_ptr_mut::<u8>().cast());
        let block = 64;
        let blocks = p.batches * (p.co / p.mg) * p.mg.div_ceil(tile) * p.out_size.div_ceil(block);
        let mut builder = self.runtime.stream().launch_builder(&function);
        builder
            .arg(&x)
            .arg(&w)
            .arg(&offset)
            .arg(&b)
            .arg(&mask)
            .arg(&y)
            .arg(&p);
        // SAFETY: validated shapes and layouts match the indexing and parameter ABI above.
        unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (blocks.min(65_535) as u32, 1, 1),
                block_dim: (block as u32, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| driver_err("launch DeformConv", e))?;
        if self.runtime.is_capturing()? {
            return Ok(());
        }
        self.runtime.synchronize()
    }
}
