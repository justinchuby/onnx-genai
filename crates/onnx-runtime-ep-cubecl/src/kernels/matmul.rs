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

use super::{ElementKind, FnKernel, input_handle, launch_kind, output_handle};
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
    // Accumulate in f32 even when `F` is f16. A K-long f16 dot product loses
    // roughly log2(K) bits to rounding, which for a transformer's K in the
    // thousands is visible in the output; the staging tiles stay f16 so the
    // bandwidth win — the entire reason to run f16 — is preserved.
    let mut acc = f32::new(0.0_f32);
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
            let a = f32::cast_from(lhs_tile[UNIT_POS_Y as usize * TILE + i]);
            let b = f32::cast_from(rhs_tile[i * TILE + UNIT_POS_X as usize]);
            acc += a * b;
        }

        sync_cube();
    }

    if row < m && col < n {
        out[batch * m * n + row * n + col] = F::cast_from(acc);
    }
}

/// Output block a cube computes, and the per-thread sub-block inside it.
///
/// `matmul_tiled` gives every thread exactly one output element, so its inner
/// loop is two shared-memory reads per FMA — arithmetic intensity 0.5, and the
/// kernel is shared-memory bound rather than ALU bound. Measured against the
/// official WebGPU EP on a workload large enough to leave the per-Run fixed
/// cost behind (see `docs/benchmarks/`), that cost us ~3.6x.
///
/// Register tiling fixes the ratio: each thread keeps a `TM x TN` accumulator
/// block in registers, so one pass of the inner loop reads `TM + TN` values and
/// issues `TM * TN` FMAs. At 4x4 that is 16 FMAs per 8 reads — intensity 2.0,
/// 4x better.
///
/// The sizes are constrained by WebGPU's portable baseline:
/// - workgroup is `(BN / TN) x (BM / TM)` = 16x16 = 256 invocations, the
///   portable ceiling.
/// - staging is `BM * BK + BK * BN` = 2048 elements = 8 KiB at f32, inside the
///   16 KiB baseline limit (f16 halves it).
/// - `BM * BK` and `BK * BN` are both 1024 = 4 * 256, so staging divides evenly
///   across the workgroup and needs no bounds check on the staging loop itself.
const BM: usize = 128;
const BN: usize = 128;
const BK: usize = 8;
const TM: usize = 8;
const TN: usize = 8;
const WG_X: usize = BN / TN;
const WG_Y: usize = BM / TM;
const WG_SIZE: usize = WG_X * WG_Y;

/// Smallest problem worth handing to `matmul_regtiled`.
///
/// The register-tiled kernel claims a 64x64 output block per cube. When `M` is
/// tiny — a decode-step GEMV has `M = 1` — 63 of every 64 rows in the block are
/// padding, and the cube count collapses far below what it takes to fill the
/// GPU. `matmul_tiled`'s 16x16 block wastes 4x less and launches 4x more cubes.
///
/// The threshold is deliberately on `M` alone: `N` and `K` are large in every
/// shape we care about, and `M` is the dimension that actually collapses.
const REGTILE_MIN_M: usize = 128;

#[cube(launch_unchecked)]
fn matmul_regtiled<F: Float>(
    lhs: &[F],
    rhs: &[F],
    out: &mut [F],
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] rhs_batched: bool,
) {
    let batch = CUBE_POS_Z as usize;
    let lhs_base = batch * m * k;
    let rhs_batch_stride = comptime!(if rhs_batched { k * n } else { 0 });
    let rhs_base = batch * rhs_batch_stride;

    let block_row = CUBE_POS_Y as usize * BM;
    let block_col = CUBE_POS_X as usize * BN;

    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let tid = ty * WG_X + tx;

    let mut lhs_tile = Shared::new_slice(BM * BK);
    let mut rhs_tile = Shared::new_slice(BK * BN);

    // f32 accumulation even when `F` is f16, for the same reason as
    // `matmul_tiled`: a K-long f16 running sum stalls once the total outgrows
    // the ulp of the addend.
    let mut acc = Array::<f32>::new(TM * TN);
    #[unroll]
    for i in 0..TM * TN {
        acc[i] = f32::new(0.0_f32);
    }
    let mut a_reg = Array::<f32>::new(TM);
    let mut b_reg = Array::<f32>::new(TN);

    let tiles = k.div_ceil(BK);
    for t in 0..tiles {
        let k_base = t * BK;

        // Out-of-range lanes stage zeros rather than returning, so every unit
        // reaches the barriers. A partially-attended barrier is UB in both
        // WGSL and SPIR-V.
        // `lhs_tile` is staged transposed, as [BK][BM] rather than [BM][BK].
        //
        // The inner loop reads TM values down a column of A for a fixed k. In
        // [BM][BK] layout those are BK apart, so the reads are strided and land
        // on a fraction of the shared-memory banks; transposed they are
        // adjacent, matching what the `rhs_tile` reads already do.
        //
        // The cost is moved to the staging write, which becomes strided — but
        // each element is written once and read TM times per k-step, so the
        // trade favours the read. The global read stays contiguous either way,
        // which matters more than either: `idx / BK` keeps consecutive lanes on
        // consecutive addresses of A.
        #[unroll]
        for s in 0..(BM * BK / WG_SIZE) {
            let idx = s * WG_SIZE + tid;
            let r = idx / BK;
            let c = idx % BK;
            let gr = block_row + r;
            let gc = k_base + c;
            lhs_tile[c * BM + r] = if gr < m && gc < k {
                lhs[lhs_base + gr * k + gc]
            } else {
                F::new(0.0_f32)
            };
        }
        #[unroll]
        for s in 0..(BK * BN / WG_SIZE) {
            let idx = s * WG_SIZE + tid;
            let gr = k_base + idx / BN;
            let gc = block_col + idx % BN;
            rhs_tile[idx] = if gr < k && gc < n {
                rhs[rhs_base + gr * n + gc]
            } else {
                F::new(0.0_f32)
            };
        }

        sync_cube();

        #[unroll]
        for kk in 0..BK {
            #[unroll]
            for i in 0..TM {
                a_reg[i] = f32::cast_from(lhs_tile[kk * BM + ty * TM + i]);
            }
            #[unroll]
            for j in 0..TN {
                b_reg[j] = f32::cast_from(rhs_tile[kk * BN + tx * TN + j]);
            }
            #[unroll]
            for i in 0..TM {
                #[unroll]
                for j in 0..TN {
                    acc[i * TN + j] += a_reg[i] * b_reg[j];
                }
            }
        }

        sync_cube();
    }

    let out_base = batch * m * n;
    #[unroll]
    for i in 0..TM {
        let gr = block_row + ty * TM + i;
        #[unroll]
        for j in 0..TN {
            let gc = block_col + tx * TN + j;
            if gr < m && gc < n {
                out[out_base + gr * n + gc] = F::cast_from(acc[i * TN + j]);
            }
        }
    }
}

#[cube(launch_unchecked)]
fn matmul_regtiled_vec4<F: Float>(
    lhs: &[Vector<F, Const<4>>],
    rhs: &[Vector<F, Const<4>>],
    out: &mut [Vector<F, Const<4>>],
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] rhs_batched: bool,
) {
    let batch = CUBE_POS_Z as usize;
    let lhs_base = batch * m * k;
    let rhs_batch_stride = comptime!(if rhs_batched { k * n } else { 0 });
    let rhs_base = batch * rhs_batch_stride;

    let block_row = CUBE_POS_Y as usize * BM;
    let block_col = CUBE_POS_X as usize * BN;

    let tx = UNIT_POS_X as usize;
    let ty = UNIT_POS_Y as usize;
    let tid = ty * WG_X + tx;

    let mut lhs_tile = Shared::new_slice(BM * BK);
    let mut rhs_tile = Shared::new_slice(BK * BN);

    let mut acc = Array::<f32>::new(TM * TN);
    #[unroll]
    for i in 0..TM * TN {
        acc[i] = f32::new(0.0_f32);
    }
    let mut a_reg = Array::<f32>::new(TM);
    let mut b_reg = Array::<f32>::new(TN);

    let tiles = k.div_ceil(BK);
    for t in 0..tiles {
        let k_base = t * BK;

        #[unroll]
        for s in 0..(BM * BK / 4 / WG_SIZE) {
            let idx = s * WG_SIZE + tid;
            let r = idx / (BK / 4);
            let c4 = idx % (BK / 4);
            let gr = block_row + r;
            let gc = k_base + c4 * 4;
            let pack = if gr < m && gc + 3 < k {
                lhs[(lhs_base + gr * k + gc) / 4]
            } else {
                Vector::<F, Const<4>>::new(F::new(0.0_f32))
            };
            #[unroll]
            for q in 0..4 {
                lhs_tile[(c4 * 4 + q) * BM + r] = pack.extract(q);
            }
        }
        #[unroll]
        for s in 0..(BK * BN / 4 / WG_SIZE) {
            let idx = s * WG_SIZE + tid;
            let r = idx / (BN / 4);
            let c4 = idx % (BN / 4);
            let gr = k_base + r;
            let gc = block_col + c4 * 4;
            let pack = if gr < k && gc + 3 < n {
                rhs[(rhs_base + gr * n + gc) / 4]
            } else {
                Vector::<F, Const<4>>::new(F::new(0.0_f32))
            };
            #[unroll]
            for q in 0..4 {
                rhs_tile[r * BN + c4 * 4 + q] = pack.extract(q);
            }
        }

        sync_cube();

        #[unroll]
        for kk in 0..BK {
            #[unroll]
            for i in 0..TM {
                a_reg[i] = f32::cast_from(lhs_tile[kk * BM + ty * TM + i]);
            }
            #[unroll]
            for j in 0..TN {
                b_reg[j] = f32::cast_from(rhs_tile[kk * BN + tx * TN + j]);
            }
            #[unroll]
            for i in 0..TM {
                #[unroll]
                for j in 0..TN {
                    acc[i * TN + j] += a_reg[i] * b_reg[j];
                }
            }
        }

        sync_cube();
    }

    let out_base = batch * m * n;
    #[unroll]
    for i in 0..TM {
        let gr = block_row + ty * TM + i;
        #[unroll]
        for j4 in 0..(TN / 4) {
            let gc = block_col + tx * TN + j4 * 4;
            if gr < m && gc + 3 < n {
                let mut pack = Vector::<F, Const<4>>::empty();
                #[unroll]
                for q in 0..4 {
                    pack.insert(q, F::cast_from(acc[i * TN + j4 * 4 + q]));
                }
                out[(out_base + gr * n + gc) / 4] = pack;
            }
        }
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

    /// Whether this shape is big enough for the register-tiled kernel to pay
    /// off. See `REGTILE_MIN_M`.
    fn use_regtile(&self) -> bool {
        self.m >= REGTILE_MIN_M && self.n >= BN && self.k >= BK
    }

    fn use_regtile_vec4(&self) -> bool {
        self.use_regtile() && self.k.is_multiple_of(4) && self.n.is_multiple_of(4)
    }

    fn regtile_cube_count(&self) -> CubeCount {
        CubeCount::Static(
            self.n.div_ceil(BN).max(1) as u32,
            self.m.div_ceil(BM).max(1) as u32,
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
                let kind = launch_kind(
                    &[(lhs.dtype, "A"), (rhs.dtype, "B"), (out.dtype, "Y")],
                    context.f16,
                    "MatMul",
                )?;

                let lhs_res = input_handle(&context, lhs, "A")?;
                let rhs_res = input_handle(&context, rhs, "B")?;
                let out_res = output_handle(&context, out, "Y")?;

                macro_rules! dispatch {
                    ($float:ty) => {
                        // SAFETY: the element counts below come from the shapes this
                        // kernel was created for, and `resolve` verified each buffer
                        // covers the bytes those shapes describe.
                        unsafe {
                            if dims.use_regtile_vec4() {
                                matmul_regtiled_vec4::launch_unchecked::<$float, R>(
                                    &context.client,
                                    dims.regtile_cube_count(),
                                    CubeDim::new_2d(WG_X as u32, WG_Y as u32),
                                    BufferArg::from_raw_parts(
                                        lhs_res.handle.clone(),
                                        dims.lhs_elements() / 4,
                                    ),
                                    BufferArg::from_raw_parts(
                                        rhs_res.handle.clone(),
                                        dims.rhs_elements() / 4,
                                    ),
                                    BufferArg::from_raw_parts(
                                        out_res.handle.clone(),
                                        dims.out_elements() / 4,
                                    ),
                                    dims.m,
                                    dims.n,
                                    dims.k,
                                    dims.rhs_batched,
                                );
                            } else if dims.use_regtile() {
                                matmul_regtiled::launch_unchecked::<$float, R>(
                                    &context.client,
                                    dims.regtile_cube_count(),
                                    CubeDim::new_2d(WG_X as u32, WG_Y as u32),
                                    BufferArg::from_raw_parts(lhs_res.handle, dims.lhs_elements()),
                                    BufferArg::from_raw_parts(rhs_res.handle, dims.rhs_elements()),
                                    BufferArg::from_raw_parts(out_res.handle, dims.out_elements()),
                                    dims.m,
                                    dims.n,
                                    dims.k,
                                    dims.rhs_batched,
                                );
                            } else {
                                matmul_tiled::launch_unchecked::<$float, R>(
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
                        }
                    };
                }
                match kind {
                    ElementKind::F32 => dispatch!(f32),
                    ElementKind::F16 => dispatch!(half::f16),
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

    /// Pins the shapes the GPU tests use to the register-tiled path.
    ///
    /// `regtiled_matmul_*` in `tests/provider_gpu.rs` check numbers, not which
    /// kernel produced them — both kernels compute the same function, so those
    /// tests would pass unchanged if the dispatch silently fell back. This test
    /// is the other half: it fixes *which* path those shapes take, so the pair
    /// together is evidence about `matmul_regtiled` specifically.
    #[test]
    fn gpu_test_shapes_select_the_register_tiled_path() {
        let regtiled = MatMulDims::resolve(&node("mm"), &[200, 43], &[43, 150]).unwrap();
        assert!(
            regtiled.use_regtile(),
            "regtiled_matmul_matches_a_host_reference must exercise the new kernel"
        );
        assert!(
            !regtiled.use_regtile_vec4(),
            "regtiled_matmul_matches_a_host_reference must stay on the scalar register path"
        );
        let batched = MatMulDims::resolve(&node("mm"), &[3, 128, 32], &[32, 128]).unwrap();
        assert!(
            batched.use_regtile(),
            "regtiled_matmul_handles_batches must exercise the new kernel"
        );
        assert!(
            batched.use_regtile_vec4(),
            "regtiled_matmul_handles_batches must exercise the vec4 register path"
        );

        let vec4 = MatMulDims::resolve(&node("mm"), &[200, 44], &[44, 148]).unwrap();
        assert!(
            vec4.use_regtile_vec4(),
            "vec4_regtiled_matmul_matches_a_host_reference must exercise the vec4 kernel"
        );

        // And the small-shape tests must keep exercising the old one.
        let small = MatMulDims::resolve(&node("mm"), &[37, 23], &[23, 19]).unwrap();
        assert!(
            !small.use_regtile(),
            "matmul_matches_a_host_reference must stay on the simple kernel"
        );
    }

    /// Each of the three conditions must be able to veto on its own, otherwise
    /// the threshold is not doing what its doc comment claims.
    #[test]
    fn each_dimension_can_veto_the_register_tiled_path() {
        let ok = MatMulDims::resolve(&node("mm"), &[128, 8], &[8, 128]).unwrap();
        assert!(ok.use_regtile());

        let m_too_small = MatMulDims::resolve(&node("mm"), &[127, 8], &[8, 128]).unwrap();
        assert!(
            !m_too_small.use_regtile(),
            "M below REGTILE_MIN_M must veto"
        );

        let n_too_small = MatMulDims::resolve(&node("mm"), &[128, 8], &[8, 127]).unwrap();
        assert!(!n_too_small.use_regtile(), "N below BN must veto");

        let k_too_small = MatMulDims::resolve(&node("mm"), &[128, 7], &[7, 128]).unwrap();
        assert!(!k_too_small.use_regtile(), "K below BK must veto");
    }

    #[test]
    fn vec4_alignment_can_veto_the_vec4_register_tiled_path() {
        let ok = MatMulDims::resolve(&node("mm"), &[128, 12], &[12, 132]).unwrap();
        assert!(ok.use_regtile_vec4());

        let k_not_vec4 = MatMulDims::resolve(&node("mm"), &[128, 10], &[10, 132]).unwrap();
        assert!(k_not_vec4.use_regtile());
        assert!(!k_not_vec4.use_regtile_vec4(), "K % 4 must veto vec4");

        let n_not_vec4 = MatMulDims::resolve(&node("mm"), &[128, 12], &[12, 130]).unwrap();
        assert!(n_not_vec4.use_regtile());
        assert!(!n_not_vec4.use_regtile_vec4(), "N % 4 must veto vec4");
    }
}
