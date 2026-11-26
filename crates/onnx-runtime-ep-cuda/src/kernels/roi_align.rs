use std::{alloc::Layout, ffi::c_void, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata,
    TensorMut, TensorView, WorkspaceLayout, WorkspaceLifetime, WorkspacePreparation,
    WorkspaceRequirement, WorkspaceView,
};
use onnx_runtime_ir::{
    DataType, Node,
    roi_align::{Pool, RoiAlign},
};

use crate::{
    error::driver_err,
    runtime::{CudaRuntime, cuptr},
};

const SOURCE: &str = include_str!("roi_align.cu");
const HALF_SOURCE: &str = concat!(
    "#include <cuda_fp16.h>\n#include <cuda_bf16.h>\n#define ROI_ALIGN_HALF\n",
    include_str!("roi_align.cu")
);
const BLOCK: u32 = 128;
const STATUS: Layout = Layout::new::<u32>();

pub(super) struct Factory(pub Arc<CudaRuntime>, pub u64);

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native {
            runtime: Arc::clone(&self.0),
            params: RoiAlign::parse(node, self.1).map_err(EpError::KernelFailed)?,
        }))
    }
}

struct Native {
    runtime: Arc<CudaRuntime>,
    params: RoiAlign,
}

impl Kernel for Native {
    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "CUDA RoiAlign reads ROI validation status; use eager execution",
        )
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let empty = inputs
            .get(1)
            .is_some_and(|input| input.shape.first() == Some(&0));
        Ok(WorkspacePreparation::Ready(if empty {
            WorkspaceRequirement::NONE
        } else {
            WorkspaceRequirement {
                bytes: STATUS.size() as u64,
                alignment: STATUS.align(),
                lifetime: WorkspaceLifetime::SessionPersistent,
            }
        }))
    }

    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.execute_with_workspace(inputs, outputs, None)
    }

    fn execute_with_workspace(
        &self,
        inputs: &[TensorView],
        outputs: &mut [TensorMut],
        workspace: Option<WorkspaceView>,
    ) -> Result<()> {
        let [x, rois, indices] = inputs else {
            return Err(EpError::KernelFailed(
                "CUDA RoiAlign requires X, rois and batch_indices".into(),
            ));
        };
        let [y] = outputs else {
            return Err(EpError::KernelFailed(
                "CUDA RoiAlign requires one output".into(),
            ));
        };
        let params = self.params;
        params
            .validate_dtypes(Some(x.dtype), Some(rois.dtype), Some(indices.dtype))
            .map_err(EpError::KernelFailed)?;
        RoiAlign::validate_shapes(Some(x.shape), Some(rois.shape), Some(indices.shape), |&n| {
            Some(n)
        })
        .map_err(EpError::KernelFailed)?;
        let [n, channels, height, width] = <[usize; 4]>::try_from(x.shape).expect("validated rank");
        let expected = [
            rois.shape[0],
            channels,
            params.output_height,
            params.output_width,
        ];
        if y.shape != expected || y.dtype != x.dtype {
            return Err(EpError::KernelFailed(format!(
                "CUDA RoiAlign Y has shape {:?} and dtype {:?}; expected {expected:?} and {:?}",
                y.shape, y.dtype, x.dtype
            )));
        }
        if rois.shape[0] == 0 {
            return Ok(());
        }
        if !inputs.iter().all(TensorView::is_contiguous) || !y.is_contiguous() {
            return Err(EpError::KernelFailed(
                "CUDA RoiAlign requires contiguous tensors; materialize strided operands first"
                    .into(),
            ));
        }
        let dtype = match x.dtype {
            DataType::Float32 => 0i32,
            DataType::Float64 => 1,
            DataType::Float16 => 2,
            DataType::BFloat16 => 3,
            _ => unreachable!("validated floating dtype"),
        };
        let (key, source) = if dtype >= 2 {
            self.runtime.require_nvrtc_half_headers("RoiAlign")?;
            ("roi_align_half", HALF_SOURCE)
        } else {
            ("roi_align_float", SOURCE)
        };
        let f = self.runtime.nvrtc_function(key, source, "roi_align")?;
        let [flag] = WorkspaceLayout::new([STATUS])?
            .bind(workspace)
            .map_err(|error| {
                EpError::KernelFailed(format!(
                    "CUDA RoiAlign needs four bytes of validation workspace: {error}"
                ))
            })?;
        let status = cuptr(flag.ptr().0.cast_const());
        // SAFETY: the bound workspace provides an aligned four-byte status word.
        unsafe { self.runtime.htod(&[0; 4], status)? };
        let out_count = y.numel() as u64;
        let work = out_count.max(rois.shape[0] as u64);
        let ptr = |t: &TensorView| cuptr(t.data_ptr::<c_void>());
        let (x, rois, indices, output) = (
            ptr(x),
            ptr(rois),
            ptr(indices),
            cuptr(y.data_ptr_mut::<c_void>()),
        );
        let (n, channels, height, width) = (n as u64, channels as u64, height as u64, width as u64);
        let (oh, ow) = (params.output_height as u64, params.output_width as u64);
        let scale = f64::from(params.spatial_scale);
        let half_pixel = i32::from(params.half_pixel);
        let max_pool = i32::from(params.pool == Pool::Max);
        let stream = self.runtime.stream();
        let mut launch = stream.launch_builder(&f);
        launch
            .arg(&x)
            .arg(&rois)
            .arg(&indices)
            .arg(&output)
            .arg(&status)
            .arg(&work)
            .arg(&out_count)
            .arg(&n)
            .arg(&channels)
            .arg(&height)
            .arg(&width)
            .arg(&oh)
            .arg(&ow)
            .arg(&params.sampling_ratio)
            .arg(&scale)
            .arg(&dtype)
            .arg(&half_pixel)
            .arg(&max_pool);
        // SAFETY: dtypes, shapes and contiguous tensor layout match the CUDA entry point.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (work.div_ceil(BLOCK as u64).clamp(1, 65_535) as u32, 1, 1),
                block_dim: (BLOCK, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|error| driver_err("launch RoiAlign", error))?;
        let mut bytes = [0; 4];
        // SAFETY: the workspace remains live and dtoh drains the compute stream.
        unsafe { self.runtime.dtoh(&mut bytes, status)? };
        let status = u32::from_ne_bytes(bytes);
        if status & 1 != 0 {
            return Err(EpError::KernelFailed(format!(
                "CUDA RoiAlign batch_indices contains a value outside [0, {n}); supply one in-range index per ROI"
            )));
        }
        if status & 2 != 0 {
            return Err(EpError::KernelFailed(
                "CUDA RoiAlign has non-finite scaled coordinates or an unrepresentable adaptive sampling grid; check rois and spatial_scale".into()
            ));
        }
        Ok(())
    }
}
