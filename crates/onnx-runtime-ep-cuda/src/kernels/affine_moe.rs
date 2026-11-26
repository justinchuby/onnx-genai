use std::{alloc::Layout, ffi::c_void, sync::Arc};

use cudarc::driver::{LaunchConfig, PushKernelArg};
use onnx_runtime_ep_api::{
    CaptureSupport, EpError, Kernel, KernelFactory, OutputMetadata, Result, TensorMetadata,
    TensorMut, TensorView, WorkspaceLayout, WorkspacePreparation, WorkspaceRequirement,
    WorkspaceView,
};
use onnx_runtime_ir::{Attribute, DataType, Node, Shape, as_static_shape};
use onnx_runtime_optimizer::affine_moe::softmax_first::{MAX_ROUTES_PER_TOKEN, OP, route_extent};

use crate::{
    error::driver_err,
    runtime::{CudaRuntime, cuptr},
};

const SOURCE: &str = concat!(
    include_str!("packed_kernel_prelude.cuh"),
    include_str!("topk_order.cuh"),
    include_str!("affine_grouped_dot.cuh"),
    include_str!("affine_moe.cu"),
);

pub(super) struct Factory(pub Arc<CudaRuntime>);

#[derive(Clone, Copy)]
struct Bank {
    block: usize,
    zero: i32,
}

struct KernelConfig {
    k: usize,
    first: Bank,
    down: Bank,
}

struct Prepared {
    tokens: usize,
    experts: usize,
    hidden: usize,
    inter: usize,
    routes: usize,
    workspace: WorkspaceLayout<5>,
}

impl KernelConfig {
    fn parse(node: &Node) -> Result<Self> {
        let attr = |name| node.attr(name).and_then(Attribute::as_int);
        let positive = |name| {
            attr(name)
                .and_then(|v| usize::try_from(v).ok())
                .filter(|&v| v > 0)
                .ok_or_else(|| error(format!("{name} must be a positive INT64")))
        };
        let bank = |block, zero| -> Result<_> {
            Ok(Bank {
                block: positive(block)?,
                zero: attr(zero)
                    .filter(|v| (0..=15).contains(v))
                    .map(|v| v as i32)
                    .ok_or_else(|| error(format!("{zero} must be a UINT4 zero point")))?,
            })
        };
        if node.attributes.len() != 5 {
            return Err(error("expected only k and the two affine block/zero pairs"));
        }
        Ok(Self {
            k: positive("k")?,
            first: bank("first_block", "first_zero")?,
            down: bank("down_block", "down_zero")?,
        })
    }

    fn prepare(&self, x: &[TensorMetadata<'_>]) -> Result<Prepared> {
        use DataType::{Float32, Uint8};
        let [input, router, w1, s1, z1, w2, s2, z2] = x else {
            return Err(error(
                "expected activation, router and two affine banks with optional zero-point slots",
            ));
        };
        let (&[tokens, hidden], &[router_tokens, experts], &[_, _, packed_inter]) =
            (input.shape, router.shape, w2.shape)
        else {
            return Err(error(
                "expected rank-2 activation/router and rank-3 down codes",
            ));
        };
        let inter = packed_inter
            .checked_mul(2)
            .ok_or_else(|| error("packed intermediate width overflows"))?;
        let routes = route_extent(tokens, self.k, experts, hidden, inter).ok_or_else(|| {
            error("invalid shape or workspace size; reduce tokens, experts, or widths")
        })?;
        if router_tokens != tokens
            || [input, router]
                .iter()
                .any(|t| !t.present || t.dtype != Float32)
        {
            return Err(error(
                "activation and router must be FLOAT with the same token count",
            ));
        }
        for (bank, codes, scales, zeros, width, depth) in [
            (self.first, w1, s1, z1, inter * 2, hidden),
            (self.down, w2, s2, z2, hidden, inter),
        ] {
            let blocks = depth / bank.block;
            if !depth.is_multiple_of(2)
                || !depth.is_multiple_of(bank.block)
                || !codes.present
                || codes.dtype != Uint8
                || codes.shape != [experts, width, depth / 2]
                || !scales.present
                || scales.dtype != Float32
                || scales.shape != [experts, width, blocks]
                || (zeros.present
                    && (zeros.dtype != Uint8
                        || zeros.shape != [experts, width, blocks.div_ceil(2)]))
            {
                return Err(error(format!(
                    "expected UINT8 packed codes and zeros and FLOAT scales for [{experts}, {width}, {depth}] at block size {}; received {:?}, {:?}, {:?}",
                    bank.block, codes, scales, zeros
                )));
            }
        }
        let layout_error = |_| error("affine MoE workspace extent exceeds layout capacity");
        let workspace = WorkspaceLayout::new([
            Layout::array::<i64>(routes).map_err(layout_error)?,
            Layout::array::<f32>(routes).map_err(layout_error)?,
            Layout::array::<f32>(routes * inter).map_err(layout_error)?,
            Layout::new::<u32>(),
            Layout::array::<f32>(tokens * hidden).map_err(layout_error)?,
        ])?;
        Ok(Prepared {
            tokens,
            experts,
            hidden,
            inter,
            routes,
            workspace,
        })
    }
}

impl KernelFactory for Factory {
    fn create(&self, node: &Node, _: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        Ok(Box::new(Native {
            config: KernelConfig::parse(node)?,
            runtime: Arc::clone(&self.0),
        }))
    }

    fn unsupported_reason(
        &self,
        node: &Node,
        _: u64,
        shapes: &[Shape],
        dtypes: &[DataType],
    ) -> Option<String> {
        let check = || -> Result<()> {
            let config = KernelConfig::parse(node)?;
            let mut dimensions = Vec::with_capacity(node.inputs.len());
            for (index, input) in node.inputs.iter().enumerate() {
                let shape = if input.is_some() {
                    shapes
                        .get(index)
                        .and_then(|shape| as_static_shape(shape))
                        .ok_or_else(|| error(format!("input {index} must have a concrete shape")))?
                } else {
                    Vec::new()
                };
                dimensions.push(shape);
            }
            let metadata: Vec<_> = node
                .inputs
                .iter()
                .zip(&dimensions)
                .enumerate()
                .map(|(index, (slot, shape))| {
                    TensorMetadata::new(
                        dtypes.get(index).copied().unwrap_or(DataType::Undefined),
                        shape,
                        slot.is_some(),
                    )
                })
                .collect();
            config.prepare(&metadata)?;
            Ok(())
        };
        check().err().map(|err| err.to_string())
    }
}

struct Native {
    config: KernelConfig,
    runtime: Arc<CudaRuntime>,
}

impl Kernel for Native {
    fn capture_support(&self) -> CaptureSupport {
        CaptureSupport::unsupported(
            "affine MoE checks DequantizeLinear scale validation on the host",
        )
    }

    fn workspace_requirement(
        &self,
        inputs: &[TensorMetadata<'_>],
        _: &[OutputMetadata<'_>],
    ) -> Result<WorkspacePreparation> {
        let p = self.config.prepare(inputs)?;
        Ok(WorkspacePreparation::Ready(WorkspaceRequirement {
            bytes: p.workspace.bytes() as u64,
            alignment: p.workspace.alignment(),
            ..WorkspaceRequirement::NONE
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
        let metadata: Vec<_> = inputs
            .iter()
            .map(|t| TensorMetadata::new(t.dtype, t.shape, !t.is_absent()))
            .collect();
        let p = self.config.prepare(&metadata)?;
        let [out] = outputs else {
            return Err(error("expected one FLOAT output"));
        };
        if out.is_absent()
            || out.dtype != DataType::Float32
            || out.shape != [p.tokens, p.hidden]
            || !out.is_contiguous()
            || inputs.iter().any(|t| !t.is_absent() && !t.is_contiguous())
        {
            return Err(error(
                "expected contiguous inputs and a contiguous FLOAT [tokens, hidden] output",
            ));
        }
        let [ids, weights, activated, status, staged] = p.workspace.bind(workspace)?;
        let output_bytes = staged.bytes();
        let [ids, weights, activated, status, staged] =
            [ids, weights, activated, status, staged].map(|w| cuptr(w.ptr().0.cast_const()));
        let ptr = |t: &TensorView| {
            if t.is_absent() {
                0
            } else {
                cuptr(t.data_ptr::<u8>() as *const c_void)
            }
        };
        let [x, probabilities, w1, s1, z1, w2, s2, z2] = std::array::from_fn(|i| ptr(&inputs[i]));
        let dst = cuptr(out.data_ptr_mut::<u8>() as *const c_void);
        let (tokens, experts, hidden, inter, routes, k) = (
            p.tokens as u64,
            p.experts as u64,
            p.hidden as u64,
            p.inter as u64,
            p.routes as u64,
            self.config.k as u64,
        );
        let caps = self.runtime.capabilities();
        let warp = MAX_ROUTES_PER_TOKEN as u32;
        if caps.max_threads_per_block() < warp || caps.max_grid_dim_x() == 0 {
            return Err(error(
                "device does not support a full warp for the combine kernel",
            ));
        }
        let threads = caps.max_threads_per_block().min(128) / warp * warp;
        let launch = |lanes: u64| LaunchConfig {
            grid_dim: (
                lanes
                    .div_ceil(u64::from(threads))
                    .min(u64::from(caps.max_grid_dim_x())) as u32,
                1,
                1,
            ),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let stream = self.runtime.stream();
        let function = |entry| {
            self.runtime
                .nvrtc_function("affine_softmax_first_moe", SOURCE, entry)
        };
        let select = function("select_experts")?;
        let first = function("project_activate")?;
        let down = function("project_combine")?;
        let (block1, block2) = (
            self.config.first.block as u64,
            self.config.down.block as u64,
        );
        let (count1, count2) = (
            experts * (2 * inter) * (hidden / block1),
            experts * hidden * (inter / block2),
        );
        // SAFETY: the caller keeps device tensors and disjoint workspace live
        // through completion on this stream. No kernel writes to the final
        // output, which may alias an input.
        unsafe {
            self.runtime.htod(&[0; 4], status)?;
            stream
                .launch_builder(&select)
                .arg(&probabilities)
                .arg(&ids)
                .arg(&weights)
                .arg(&tokens)
                .arg(&experts)
                .arg(&k)
                .launch(launch(tokens))
                .map_err(|e| driver_err("launch affine MoE TopK", e))?;
            stream
                .launch_builder(&first)
                .arg(&x)
                .arg(&w1)
                .arg(&s1)
                .arg(&z1)
                .arg(&ids)
                .arg(&activated)
                .arg(&status)
                .arg(&count1)
                .arg(&routes)
                .arg(&k)
                .arg(&hidden)
                .arg(&inter)
                .arg(&block1)
                .arg(&self.config.first.zero)
                .launch(launch((routes * inter).max(count1)))
                .map_err(|e| driver_err("launch affine MoE first projection", e))?;
            stream
                .launch_builder(&down)
                .arg(&activated)
                .arg(&w2)
                .arg(&s2)
                .arg(&z2)
                .arg(&ids)
                .arg(&weights)
                .arg(&staged)
                .arg(&status)
                .arg(&count2)
                .arg(&tokens)
                .arg(&k)
                .arg(&hidden)
                .arg(&inter)
                .arg(&block2)
                .arg(&self.config.down.zero)
                .launch(LaunchConfig {
                    // Output lanes are warps, while validation lanes are threads.
                    grid_dim: (
                        (tokens * hidden)
                            .div_ceil(u64::from(threads / warp))
                            .max(count2.div_ceil(u64::from(threads)))
                            .min(u64::from(caps.max_grid_dim_x())) as u32,
                        1,
                        1,
                    ),
                    ..launch(1)
                })
                .map_err(|e| driver_err("launch affine MoE down projection", e))?;
            let mut bytes = [0; 4];
            self.runtime.dtoh(&mut bytes, status)?;
            if u32::from_ne_bytes(bytes) != 0 {
                return Err(error(
                    "DequantizeLinear scales in both banks must be positive and finite",
                ));
            }
            // Readback drained both projections. Publish only after validation,
            // when no projection can still read output-aliased input bytes.
            self.runtime.dtod_async(staged, dst, output_bytes)?;
        }
        Ok(())
    }
}

fn error(message: impl std::fmt::Display) -> EpError {
    EpError::KernelFailed(format!("cuda_ep {OP}: {message}"))
}
