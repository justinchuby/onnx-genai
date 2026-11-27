use std::sync::Arc;

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, Result, TensorMut, TensorView,
};
use onnx_runtime_ir::max_roi_pool::{Attributes, validate_types};
use onnx_runtime_ir::{DataType, DeviceId, Node};

use crate::error::driver_err;
use crate::runtime::{CudaRuntime, cuptr};

pub struct Factory {
    pub runtime: Arc<CudaRuntime>,
    pub version: u64,
}

struct MaxRoiPool {
    runtime: Arc<CudaRuntime>,
    attrs: Attributes,
    version: u64,
}

fn fail(detail: String) -> EpError {
    EpError::KernelFailed(format!("CUDA MaxRoiPool: {detail}"))
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(MaxRoiPool {
            runtime: self.runtime.clone(),
            attrs: Attributes::parse(node).map_err(fail)?,
            version: self.version,
        }))
    }
}

impl Kernel for MaxRoiPool {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let ([x, rois], [y]) = (inputs, outputs) else {
            return Err(fail(
                "expected two inputs X and rois and one output Y".into(),
            ));
        };
        validate_types(
            self.version,
            [Some(x.dtype), Some(rois.dtype), Some(y.dtype)],
        )
        .map_err(fail)?;
        let shape = self.attrs.output_shape(x.shape, rois.shape).map_err(fail)?;
        if y.shape != shape {
            return Err(fail(format!("Y shape {:?}, expected {shape:?}", y.shape)));
        }
        if !x.is_contiguous() || !rois.is_contiguous() || !y.is_contiguous() {
            return Err(fail("requires contiguous X, rois and Y".into()));
        }
        let device = DeviceId::cuda(self.runtime.ordinal());
        for (name, actual) in [("X", x.device), ("rois", rois.device), ("Y", y.device)] {
            if actual != device {
                return Err(fail(format!(
                    "{name} is on {actual:?}; place it on {device:?} before executing MaxRoiPool"
                )));
            }
        }
        if shape.contains(&0) {
            return Ok(());
        }
        if self.runtime.is_capturing()? {
            return Err(fail(
                "cannot capture because ROI validation requires a status readback".into(),
            ));
        }
        let entry = match x.dtype {
            DataType::Float32 => "max_roi_pool_f32",
            DataType::Float64 => "max_roi_pool_f64",
            DataType::Float16 => "max_roi_pool_f16",
            DataType::BFloat16 => "max_roi_pool_bf16",
            other => return Err(fail(format!("unsupported X dtype {other:?}"))),
        };
        let function = if matches!(x.dtype, DataType::Float16 | DataType::BFloat16) {
            self.runtime.require_nvrtc_half_headers("MaxRoiPool")?;
            self.runtime.nvrtc_function(
                "max_roi_pool_half",
                concat!(
                    "#define MAX_ROI_POOL_HALF_TYPES\n",
                    include_str!("max_roi_pool.cu")
                ),
                entry,
            )?
        } else {
            self.runtime
                .nvrtc_function("max_roi_pool", include_str!("max_roi_pool.cu"), entry)?
        };
        let [batches, channels, height, width] =
            [x.shape[0], x.shape[1], x.shape[2], x.shape[3]].map(|d| d as u64);
        // `output_shape` has checked that both dimensions fit the CUDA ABI.
        let [ph, pw] = self.attrs.pooled.map(|d| d as i32);
        let total = y.numel() as u64;
        let x = cuptr(x.data_ptr::<u8>().cast());
        let rois = cuptr(rois.data_ptr::<u8>().cast());
        let y = cuptr(y.data_ptr_mut::<u8>().cast());
        let error = self.runtime.alloc_raw(4)?;
        let launch = || -> Result<()> {
            let mut status = [0; 4];
            // SAFETY: error is an owned four-byte allocation for this call.
            unsafe { self.runtime.htod(&status, error) }?;
            let mut builder = self.runtime.stream().launch_builder(&function);
            builder
                .arg(&x)
                .arg(&rois)
                .arg(&y)
                .arg(&total)
                .arg(&batches)
                .arg(&channels)
                .arg(&height)
                .arg(&width)
                .arg(&ph)
                .arg(&pw)
                .arg(&self.attrs.spatial_scale)
                .arg(&error);
            const BLOCK: u32 = 128;
            // SAFETY: validated layouts match the CUDA indexing and parameter ABI.
            // Each ROI is validated before any X access.
            unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (total.div_ceil(u64::from(BLOCK)).min(65_535) as u32, 1, 1),
                    block_dim: (BLOCK, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .map_err(|e| driver_err("launch MaxRoiPool", e))?;
            // SAFETY: the four-byte device status remains live through readback.
            unsafe { self.runtime.dtoh(&mut status, error) }?;
            let status = u32::from_ne_bytes(status);
            if status & 1 != 0 {
                return Err(fail(format!(
                    "rois batch_id must be an integer in [0,{batches})"
                )));
            }
            if status & 2 != 0 {
                return Err(fail("rois x1, y1, x2 and y2 must have finite scaled and rounded coordinates in the signed 32-bit range".into()));
            }
            Ok(())
        };
        let result = launch();
        // SAFETY: error is the allocation owned by this call, freed exactly once.
        let released = unsafe { self.runtime.free_raw(error) };
        result.and(released)
    }

    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported("MaxRoiPool validates ROI values with a device status readback")
    }

    fn supports_strided_input(&self, _: usize) -> bool {
        false
    }
}
