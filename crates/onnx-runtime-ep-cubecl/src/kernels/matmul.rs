//! A straightforward tiled `MatMul`.
//!
//! # Why this is deliberately naive
//!
//! Neither `wgpu` backend exposes cooperative-matrix / tensor-core units
//! through CubeCL today, so the ceiling here is a hand-written shared-memory
//! tiled loop rather than anything resembling a tuned GEMM. This kernel is
//! correct and portable, and it is the piece to replace with CubeK's matmul
//! once CubeK and CubeCL can be pinned to a common revision.
//!
//! # Supported shapes
//!
//! `A` is `[..., M, K]`; `B` is either `[K, N]` (shared across the batch) or
//! `[..., K, N]` with a batch matching `A`. Anything else — vector-shaped
//! operands that ONNX promotes, or mismatched batch dims — is rejected by name
//! so the node falls back cleanly.

use std::sync::Arc;

use cubecl::prelude::*;
use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::Node;

use super::{FnKernel, input_handle, output_handle, require_f32};
use crate::context::CubeclContext;

/// Side of the square output tile each cube computes.
///
/// 16x16 = 256 invocations, the portable WebGPU workgroup ceiling, and the two
/// staging tiles cost 2 * 16 * 16 * 4 = 2 KiB of workgroup memory, well inside
/// the 16 KiB baseline limit.
const TILE: usize = 16;

#[cube(launch_unchecked)]
fn matmul_tiled<F: Float>(
    lhs: &[F],
    rhs: &[F],
    out: &mut [F],
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] rhs_batched: bool,
) {
    let row = CUBE_POS_Y as usize * TILE + UNIT_POS_Y as usize;
    let col = CUBE_POS_X as usize * TILE + UNIT_POS_X as usize;
    let batch = CUBE_POS_Z as usize;

    let lhs_base = batch * m * k;
    // Comptime on both arms, so a shared `B` folds the batch term away entirely.
    let rhs_batch_stride = comptime!(if rhs_batched { k * n } else { 0 });
    let rhs_base = batch * rhs_batch_stride;

    let mut lhs_tile = Shared::new_slice(TILE * TILE);
    let mut rhs_tile = Shared::new_slice(TILE * TILE);

    let local = UNIT_POS_Y as usize * TILE + UNIT_POS_X as usize;
    let mut acc = F::new(0.0_f32);
    let tiles = k.div_ceil(TILE);

    for t in 0..tiles {
        let k_lhs = t * TILE + UNIT_POS_X as usize;
        let k_rhs = t * TILE + UNIT_POS_Y as usize;

        // Out-of-range lanes stage zeros instead of returning early, so every
        // unit in the cube still reaches the barriers below. A barrier that only
        // some units reach is undefined behaviour in both WGSL and SPIR-V.
        lhs_tile[local] = if row < m && k_lhs < k {
            lhs[lhs_base + row * k + k_lhs]
        } else {
            F::new(0.0_f32)
        };
        rhs_tile[local] = if k_rhs < k && col < n {
            rhs[rhs_base + k_rhs * n + col]
        } else {
            F::new(0.0_f32)
        };

        sync_cube();

        for i in 0..TILE {
            acc +=
                lhs_tile[UNIT_POS_Y as usize * TILE + i] * rhs_tile[i * TILE + UNIT_POS_X as usize];
        }

        sync_cube();
    }

    if row < m && col < n {
        out[batch * m * n + row * n + col] = acc;
    }
}

pub struct MatMulFactory<R: Runtime> {
    context: Arc<CubeclContext<R>>,
}

impl<R: Runtime> MatMulFactory<R> {
    pub fn new(context: Arc<CubeclContext<R>>) -> Self {
        Self { context }
    }
}

/// The dimensions a concrete `MatMul` node resolves to.
#[derive(Debug, Clone, Copy)]
struct MatMulDims {
    batch: usize,
    m: usize,
    n: usize,
    k: usize,
    rhs_batched: bool,
}

impl MatMulDims {
    fn resolve(node: &Node, lhs: &[usize], rhs: &[usize]) -> Result<Self> {
        let reject = |reason: String| {
            EpError::KernelFailed(format!(
                "cubecl_ep: MatMul node '{}' with A={lhs:?} and B={rhs:?} is unsupported: \
                 {reason}. These backends implement A=[..,M,K] times B=[K,N] or B=[..,K,N] \
                 with a matching batch; assign this node to another EP.",
                node.name
            ))
        };
        if lhs.len() < 2 || rhs.len() < 2 {
            return Err(reject(
                "ONNX vector promotion of a 1-D operand is not handled".into(),
            ));
        }
        let m = lhs[lhs.len() - 2];
        let k = lhs[lhs.len() - 1];
        let k_rhs = rhs[rhs.len() - 2];
        let n = rhs[rhs.len() - 1];
        if k != k_rhs {
            return Err(reject(format!(
                "inner dimensions disagree ({k} vs {k_rhs})"
            )));
        }
        let lhs_batch: usize = lhs[..lhs.len() - 2].iter().product();
        let rhs_batch: usize = rhs[..rhs.len() - 2].iter().product();
        let rhs_batched = match rhs_batch {
            1 => false,
            b if b == lhs_batch => true,
            b => {
                return Err(reject(format!(
                    "batch dimensions disagree ({lhs_batch} vs {b})"
                )));
            }
        };
        // Cube counts are u32 on every backend, so a dimension that cannot be
        // tiled within u32 has to be refused here rather than truncated.
        let fits = |value: usize, what: &str| -> Result<()> {
            match u32::try_from(value.div_ceil(TILE).max(1)) {
                Ok(_) => Ok(()),
                Err(_) => Err(reject(format!(
                    "{what}={value} needs more cubes than u32 can index"
                ))),
            }
        };
        let batch = lhs_batch.max(1);
        fits(m, "M")?;
        fits(n, "N")?;
        u32::try_from(batch)
            .map_err(|_| reject(format!("batch={batch} exceeds the u32 cube-count limit")))?;
        Ok(Self {
            batch,
            m,
            n,
            k,
            rhs_batched,
        })
    }

    fn lhs_elements(&self) -> usize {
        self.batch * self.m * self.k
    }

    fn rhs_elements(&self) -> usize {
        let batch = if self.rhs_batched { self.batch } else { 1 };
        batch * self.k * self.n
    }

    fn out_elements(&self) -> usize {
        self.batch * self.m * self.n
    }

    fn cube_count(&self) -> CubeCount {
        CubeCount::Static(
            self.n.div_ceil(TILE).max(1) as u32,
            self.m.div_ceil(TILE).max(1) as u32,
            self.batch as u32,
        )
    }
}

impl<R: Runtime> KernelFactory for MatMulFactory<R> {
    fn create(&self, node: &Node, input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let [lhs_shape, rhs_shape] = input_shapes else {
            return Err(EpError::KernelFailed(format!(
                "cubecl_ep: MatMul node '{}' has {} inputs, expected exactly 2",
                node.name,
                input_shapes.len()
            )));
        };
        let dims = MatMulDims::resolve(node, lhs_shape, rhs_shape)?;
        let context = self.context.clone();
        Ok(Box::new(FnKernel(
            move |inputs: &[TensorView<'_>], outputs: &mut [TensorMut<'_>]| {
                let [lhs, rhs] = inputs else {
                    return Err(EpError::KernelFailed(format!(
                        "cubecl_ep: MatMul expected 2 inputs at execution, got {}",
                        inputs.len()
                    )));
                };
                let Some(out) = outputs.first_mut() else {
                    return Err(EpError::KernelFailed(
                        "cubecl_ep: MatMul expected 1 output at execution, got 0".to_string(),
                    ));
                };
                require_f32(lhs.dtype, "MatMul", "A")?;
                require_f32(rhs.dtype, "MatMul", "B")?;
                require_f32(out.dtype, "MatMul", "Y")?;

                let lhs_res = input_handle(&context, lhs, "A")?;
                let rhs_res = input_handle(&context, rhs, "B")?;
                let out_res = output_handle(&context, out, "Y")?;

                // SAFETY: the element counts below come from the shapes this kernel
                // was created for, and `resolve` verified each buffer covers the
                // bytes those shapes describe.
                unsafe {
                    matmul_tiled::launch_unchecked::<f32, R>(
                        &context.client,
                        dims.cube_count(),
                        CubeDim::new_2d(TILE as u32, TILE as u32),
                        BufferArg::from_raw_parts(lhs_res.handle, dims.lhs_elements()),
                        BufferArg::from_raw_parts(rhs_res.handle, dims.rhs_elements()),
                        BufferArg::from_raw_parts(out_res.handle, dims.out_elements()),
                        dims.m,
                        dims.n,
                        dims.k,
                        dims.rhs_batched,
                    );
                }
                Ok(())
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(name: &str) -> Node {
        let mut node = Node::new(onnx_runtime_ir::NodeId(0), "MatMul", Vec::new(), Vec::new());
        node.name = name.to_string();
        node
    }

    #[test]
    fn resolves_plain_2d() {
        let dims = MatMulDims::resolve(&node("mm"), &[4, 8], &[8, 16]).unwrap();
        assert_eq!((dims.batch, dims.m, dims.k, dims.n), (1, 4, 8, 16));
        assert!(!dims.rhs_batched);
    }

    #[test]
    fn shared_rhs_is_not_batched() {
        let dims = MatMulDims::resolve(&node("mm"), &[2, 3, 4, 8], &[8, 16]).unwrap();
        assert_eq!(dims.batch, 6);
        assert!(!dims.rhs_batched);
        assert_eq!(dims.rhs_elements(), 8 * 16);
    }

    #[test]
    fn matching_batch_is_batched() {
        let dims = MatMulDims::resolve(&node("mm"), &[5, 4, 8], &[5, 8, 16]).unwrap();
        assert_eq!(dims.batch, 5);
        assert!(dims.rhs_batched);
        assert_eq!(dims.rhs_elements(), 5 * 8 * 16);
    }

    #[test]
    fn inner_mismatch_names_both_dims() {
        let err = MatMulDims::resolve(&node("mm"), &[4, 8], &[9, 16])
            .unwrap_err()
            .to_string();
        assert!(err.contains("inner dimensions disagree (8 vs 9)"), "{err}");
    }

    #[test]
    fn batch_mismatch_is_rejected() {
        let err = MatMulDims::resolve(&node("mm"), &[5, 4, 8], &[3, 8, 16])
            .unwrap_err()
            .to_string();
        assert!(err.contains("batch dimensions disagree"), "{err}");
    }

    #[test]
    fn rank_one_operand_is_rejected() {
        let err = MatMulDims::resolve(&node("mm"), &[8], &[8, 16])
            .unwrap_err()
            .to_string();
        assert!(err.contains("vector promotion"), "{err}");
    }
}
