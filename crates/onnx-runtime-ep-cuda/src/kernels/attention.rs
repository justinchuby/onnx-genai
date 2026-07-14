//! Phase-2a scaled-dot-product / grouped-query **attention** on the GPU
//! (`docs/ORT2.md` §13 + §15.5).
//!
//! This is the **clean baseline** the roadmap calls for ("Phase 2a cuDNN fused
//! SDPA → Phase 2b FlashAttention-3"): a correct, mergeable attention kernel
//! that establishes the [`Kernel`] wiring — the exact §13.3 binding shape
//! (`causal`, `num_heads`, `head_dim`, `scale`, plus `num_kv_heads` for GQA) —
//! so a cuDNN-fused SDPA or an FA3 shim can drop in behind the same interface
//! later without touching callers. It is deliberately **not** a FlashAttention
//! kernel.
//!
//! ## What it computes
//!
//! ```text
//! O = softmax( scale · Q·Kᵀ  [+ mask] , axis = keys ) · V
//! ```
//!
//! with `Q : [B, num_heads, Sq, D]`, `K,V : [B, num_kv_heads, Sk, D]`, and
//! `O : [B, num_heads, Sq, D]`, all row-major f32.
//!
//! ## Design — two batched cuBLAS GEMMs around one NVRTC softmax
//!
//! 1. **Scores** `S = scale·Q·Kᵀ` via [`blas::gemm_ex`]. cuBLAS is
//!    column-major; a row-major `X[r,c]` (ld=c) is byte-identically the
//!    column-major `Xᵀ[c,r]`. We want the row-major bytes of `S[Sq,Sk]`, i.e.
//!    the column-major `Sᵀ = K·Qᵀ`, so we ask cuBLAS for
//!    `C[m=Sk, n=Sq] = opᵀ(K) · op(Q)` with `k = D` (`transa = T`,
//!    `transb = N`). The softmax `scale` folds into the GEMM `alpha` for free.
//! 2. **Softmax** over the last (keys) axis of `S`, fused with the `scale`
//!    (already applied), the optional additive `mask`, and the `causal`
//!    upper-triangular mask, in a single NVRTC-compiled kernel. Numerically
//!    stable (subtract row max). Runs in place, turning `S` into the
//!    probabilities `P`.
//! 3. **Output** `O = P·V` via [`blas::gemm_ex`]. Row-major `O[Sq,D]` bytes are
//!    the column-major `Oᵀ = Vᵀ·Pᵀ`, i.e. `C[m=D, n=Sq] = op(V) · op(P)` with
//!    `k = Sk` (`transa = N`, `transb = N`).
//!
//! All three steps submit onto the EP's single stream, so their ordering is
//! implicit — no host sync between stages, one sync at the end.
//!
//! ## GQA / MQA
//!
//! `num_kv_heads` may be smaller than `num_heads`; each KV head is shared by a
//! contiguous group of `num_heads / num_kv_heads` query heads. The baseline
//! iterates `(batch, query-head)` and points the QKᵀ / PV GEMMs at the KV head
//! `h / group`, so the KV broadcast costs no extra memory (no materialised
//! expansion). Per-`(b,h)` GEMMs keep the GQA pointer mapping trivially correct;
//! collapsing them into a single strided-batch call (KV stride 0 within a group)
//! is a Phase-2b throughput optimisation.
//!
//! ## Phase-2a limits (all actionable errors, never panics)
//!
//! * dtype other than f32 → deferred (fp16 SDPA lands with cuDNN in Phase 2b).
//! * ranks other than the explicit 4-D `[B, H, S, D]` layout → deferred.
//! * non-contiguous (strided) Q/K/V/O or mask → actionable "materialise" error.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys::CUdeviceptr;
use cudarc::driver::{LaunchConfig, PushKernelArg};

use onnx_runtime_ep_api::{EpError, Kernel, KernelFactory, Result, TensorMut, TensorView};
use onnx_runtime_ir::{DataType, Node};

use crate::blas::{gemm_ex, GemmDtype, GemmEx, WORKSPACE_BYTES};
use crate::error::{driver_err, not_implemented};
use crate::runtime::{cuptr, CudaRuntime};

/// NVRTC source for the fused, numerically-stable softmax over the last (keys)
/// axis of the score matrix, with `causal` + optional additive `mask` folded in.
///
/// One thread block per score row (`= B · num_heads · Sq` rows); the block
/// cooperatively reduces the row max then the row sum in shared memory. The
/// `scale` is already baked into the scores by the QKᵀ GEMM `alpha`.
const SOFTMAX_SRC: &str = r#"
extern "C" __global__ void attn_softmax_f32(
    float*       scores,       // [nrows, sk] row-major, in/out
    const float* mask,         // additive mask planes, or null when mask_planes==0
    const int    nrows,        // B * heads * sq
    const int    sk,           // key length (softmax axis)
    const int    sq,           // query length
    const int    heads,        // num query heads
    const int    causal,       // 0/1
    const int    mask_planes,  // 0 (none), 1, batch, or batch*heads
    const int    batch)
{
    // NVRTC has no <math.h>: build +inf from its bit pattern.
    const float INF = __int_as_float(0x7f800000);

    const int row = blockIdx.x;
    if (row >= nrows) return;

    // row = ((b*heads) + h)*sq + i
    const int i  = row % sq;
    const int bh = row / sq;
    const int b  = bh / heads;

    float* s = scores + (size_t)row * sk;

    // Causal alignment: query i (absolute position sk-sq+i for cached decode)
    // attends to keys j <= sk-sq+i. Reduces to lower-triangular when sq==sk.
    const int causal_max = sk - sq + i;

    const float* mrow = 0;
    if (mask_planes > 0) {
        int plane = 0;
        if (mask_planes == batch)            plane = b;
        else if (mask_planes == batch*heads) plane = bh;
        // else mask_planes == 1 -> plane 0 (shared [sq,sk])
        mrow = mask + ((size_t)plane * sq + i) * sk;
    }

    extern __shared__ float red[];
    const int tid = threadIdx.x;
    const int nt  = blockDim.x;

    // Pass 1: apply masks, find the row max.
    float local_max = -INF;
    for (int j = tid; j < sk; j += nt) {
        float v;
        if (causal && j > causal_max) {
            v = -INF;
        } else {
            v = s[j];
            if (mrow) v += mrow[j];
        }
        s[j] = v;
        local_max = fmaxf(local_max, v);
    }
    red[tid] = local_max;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] = fmaxf(red[tid], red[tid + off]);
        __syncthreads();
    }
    const float row_max = red[0];
    __syncthreads();

    // Pass 2: exponentiate (stable) and sum. A fully-masked row (max == -inf)
    // yields all-zero exponentials.
    float local_sum = 0.0f;
    for (int j = tid; j < sk; j += nt) {
        const float v = s[j];
        const float e = (v == -INF) ? 0.0f : expf(v - row_max);
        s[j] = e;
        local_sum += e;
    }
    red[tid] = local_sum;
    __syncthreads();
    for (int off = nt >> 1; off > 0; off >>= 1) {
        if (tid < off) red[tid] += red[tid + off];
        __syncthreads();
    }
    const float row_sum = red[0];
    __syncthreads();

    // Pass 3: normalise (guard the degenerate fully-masked row).
    const float inv = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int j = tid; j < sk; j += nt) {
        s[j] *= inv;
    }
}
"#;

/// Stable module + entry-point names for the NVRTC softmax (see
/// [`CudaRuntime::nvrtc_function`]).
const SOFTMAX_MODULE: &str = "attn_softmax_f32";
const SOFTMAX_ENTRY: &str = "attn_softmax_f32";

/// Threads per block for the softmax reduction (a power of two, so the tree
/// reduction is exact); rows longer than this are handled by the strided loop.
const SOFTMAX_BLOCK: u32 = 256;

/// Factory for [`AttentionKernel`]; reads the §13.3 binding attributes.
///
/// Attributes (model-agnostic — all runtime data, RULES.md #2):
/// * `num_heads` (int, **required**) — number of query heads.
/// * `kv_num_heads` (int, optional; default `num_heads`) — GQA/MQA KV heads.
/// * `causal` (int 0/1, optional; default 0) — causal masking.
/// * `scale` (float, optional; default `1/sqrt(head_dim)`).
pub struct AttentionFactory {
    pub runtime: Arc<CudaRuntime>,
}

impl KernelFactory for AttentionFactory {
    fn create(&self, node: &Node, _input_shapes: &[Vec<usize>]) -> Result<Box<dyn Kernel>> {
        let num_heads = node
            .attr("num_heads")
            .and_then(|a| a.as_int())
            .ok_or_else(|| {
                EpError::KernelFailed(
                    "cuda_ep Attention: missing required int `num_heads` attribute".into(),
                )
            })?;
        if num_heads <= 0 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: `num_heads` must be positive, got {num_heads}"
            )));
        }
        let num_kv_heads = node
            .attr("kv_num_heads")
            .and_then(|a| a.as_int())
            .unwrap_or(num_heads);
        let causal = node.attr("causal").and_then(|a| a.as_int()).unwrap_or(0) != 0;
        let scale = node.attr("scale").and_then(|a| a.as_float());

        AttentionKernel::new(
            self.runtime.clone(),
            causal,
            num_heads as usize,
            num_kv_heads as usize,
            scale,
        )
        .map(|k| Box::new(k) as Box<dyn Kernel>)
    }
}

/// Phase-2a SDPA/GQA attention kernel (cuBLAS batched GEMM + NVRTC softmax).
#[derive(Debug)]
pub struct AttentionKernel {
    runtime: Arc<CudaRuntime>,
    causal: bool,
    num_heads: usize,
    num_kv_heads: usize,
    /// Softmax scale; `None` means the default `1/sqrt(head_dim)`, resolved once
    /// `head_dim` is known from the Q shape at execute time.
    scale: Option<f32>,
}

impl AttentionKernel {
    /// Direct constructor (the testable §13.3-style entry point, independent of
    /// the not-yet-wired fusion pass). `num_kv_heads` must divide `num_heads`.
    pub fn new(
        runtime: Arc<CudaRuntime>,
        causal: bool,
        num_heads: usize,
        num_kv_heads: usize,
        scale: Option<f32>,
    ) -> Result<Self> {
        if num_heads == 0 || num_kv_heads == 0 {
            return Err(EpError::KernelFailed(
                "cuda_ep Attention: num_heads and num_kv_heads must be non-zero".into(),
            ));
        }
        if !num_heads.is_multiple_of(num_kv_heads) {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: num_heads ({num_heads}) must be a multiple of \
                 num_kv_heads ({num_kv_heads}) for grouped-query attention"
            )));
        }
        Ok(Self {
            runtime,
            causal,
            num_heads,
            num_kv_heads,
            scale,
        })
    }

    fn run(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        if !(3..=4).contains(&inputs.len()) || outputs.len() != 1 {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: expected 3 inputs (Q,K,V) or 4 (Q,K,V,mask) \
                 and 1 output, got {} inputs and {} outputs",
                inputs.len(),
                outputs.len()
            )));
        }
        let q = &inputs[0];
        let k = &inputs[1];
        let v = &inputs[2];
        let mask = inputs.get(3);

        // Phase-2a is f32-only (fp16 SDPA arrives with cuDNN in Phase 2b).
        for (name, dt) in [("Q", q.dtype), ("K", k.dtype), ("V", v.dtype), ("O", outputs[0].dtype)]
        {
            if dt != DataType::Float32 {
                return Err(not_implemented(format!(
                    "Attention with {name} dtype {dt:?} (Phase-2a baseline is f32-only)"
                )));
            }
        }

        // Explicit 4-D [B, heads, seq, head_dim] layout only.
        for (name, t) in [("Q", q.shape), ("K", k.shape), ("V", v.shape)] {
            if t.len() != 4 {
                return Err(not_implemented(format!(
                    "Attention with {name} rank {} (Phase-2a expects 4-D \
                     [batch, heads, seq, head_dim]); reshape/transpose upstream",
                    t.len()
                )));
            }
        }

        let (batch, hq, sq, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
        let (bk, hk, sk, dk) = (k.shape[0], k.shape[1], k.shape[2], k.shape[3]);

        if hq != self.num_heads || hk != self.num_kv_heads {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: Q heads {hq} / K heads {hk} disagree with \
                 num_heads {} / num_kv_heads {}",
                self.num_heads, self.num_kv_heads
            )));
        }
        if bk != batch || dk != d {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: Q {:?} and K {:?} must share batch and head_dim",
                q.shape, k.shape
            )));
        }
        if v.shape != [batch, self.num_kv_heads, sk, d] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: V shape {:?} must be [batch {batch}, kv_heads {}, \
                 seq_k {sk}, head_dim {d}]",
                v.shape, self.num_kv_heads
            )));
        }
        if outputs[0].shape != [batch, hq, sq, d] {
            return Err(EpError::KernelFailed(format!(
                "cuda_ep Attention: output shape {:?} must be [batch {batch}, \
                 heads {hq}, seq_q {sq}, head_dim {d}]",
                outputs[0].shape
            )));
        }

        // The baseline addresses per-head slices with plain pointer arithmetic,
        // so it requires dense row-major buffers.
        for (name, contiguous) in [
            ("Q", q.is_contiguous()),
            ("K", k.is_contiguous()),
            ("V", v.is_contiguous()),
            ("O", outputs[0].is_contiguous()),
        ] {
            if !contiguous {
                return Err(not_implemented(format!(
                    "Attention with a non-contiguous (strided) {name}; \
                     materialise it (insert a copy) before the attention op"
                )));
            }
        }

        let group = self.num_heads / self.num_kv_heads;
        let scale = self.scale.unwrap_or_else(|| 1.0 / (d as f32).sqrt());

        // Optional additive mask: f32, contiguous, element count a whole number
        // of [sq,sk] planes broadcasting over {1, batch, batch*heads}.
        let (mask_ptr, mask_planes) = match mask {
            None => (0u64, 0i32),
            Some(m) => {
                if m.dtype != DataType::Float32 {
                    return Err(not_implemented(format!(
                        "Attention additive mask dtype {:?} (Phase-2a is f32-only)",
                        m.dtype
                    )));
                }
                if !m.is_contiguous() {
                    return Err(not_implemented(
                        "Attention with a non-contiguous (strided) mask; materialise it first",
                    ));
                }
                let plane = sq * sk;
                let n = m.numel();
                if plane == 0 || !n.is_multiple_of(plane) {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Attention: mask has {n} elements, not a whole number of \
                         [seq_q {sq}, seq_k {sk}] planes"
                    )));
                }
                let planes = n / plane;
                if planes != 1 && planes != batch && planes != batch * self.num_heads {
                    return Err(EpError::KernelFailed(format!(
                        "cuda_ep Attention: mask has {planes} [seq_q,seq_k] planes; expected a \
                         broadcastable 1, batch ({batch}), or batch*heads ({})",
                        batch * self.num_heads
                    )));
                }
                (
                    cuptr(m.data_ptr::<u8>() as *const c_void),
                    planes as i32,
                )
            }
        };

        let q_base = cuptr(q.data_ptr::<u8>() as *const c_void);
        let k_base = cuptr(k.data_ptr::<u8>() as *const c_void);
        let v_base = cuptr(v.data_ptr::<u8>() as *const c_void);
        let o_base = cuptr(outputs[0].data_ptr_mut::<u8>() as *const c_void);

        const F32: u64 = std::mem::size_of::<f32>() as u64;

        // Scratch scores/probabilities buffer S = [B, heads, Sq, Sk] (f32) and a
        // single reused cuBLASLt workspace.
        let scores_elems = batch * self.num_heads * sq * sk;
        let scores_buf = self.runtime.alloc_raw(scores_elems * F32 as usize)?;
        let workspace = self.runtime.alloc_raw(WORKSPACE_BYTES)?;

        let result = self.run_stages(
            batch, sq, sk, d, group, scale, q_base, k_base, v_base, o_base, scores_buf, workspace,
            mask_ptr, mask_planes,
        );

        // Always release scratch + workspace, even on failure.
        // SAFETY: both pointers came from the `alloc_raw` calls above and are
        // each freed exactly once here.
        let free_scores = unsafe { self.runtime.free_raw(scores_buf) };
        let free_ws = unsafe { self.runtime.free_raw(workspace) };
        result.and(free_scores).and(free_ws)
    }

    /// The three stream-ordered stages (QKᵀ, softmax, PV) plus a final sync.
    #[allow(clippy::too_many_arguments)]
    fn run_stages(
        &self,
        batch: usize,
        sq: usize,
        sk: usize,
        d: usize,
        group: usize,
        scale: f32,
        q_base: CUdeviceptr,
        k_base: CUdeviceptr,
        v_base: CUdeviceptr,
        o_base: CUdeviceptr,
        scores_buf: CUdeviceptr,
        workspace: CUdeviceptr,
        mask_ptr: CUdeviceptr,
        mask_planes: i32,
    ) -> Result<()> {
        const F32: u64 = std::mem::size_of::<f32>() as u64;
        let blas = self.runtime.blas();
        let stream = self.runtime.stream_ptr();

        // Stage 1: per-head S = scale · Q·Kᵀ.  Column-major C[Sk,Sq] = Kᵀ·Q.
        for b in 0..batch {
            for h in 0..self.num_heads {
                let kv = h / group;
                let q_head = q_base + ((b * self.num_heads + h) * sq * d) as u64 * F32;
                let k_head = k_base + ((b * self.num_kv_heads + kv) * sk * d) as u64 * F32;
                let s_head = scores_buf + ((b * self.num_heads + h) * sq * sk) as u64 * F32;

                let p = GemmEx {
                    dtype: GemmDtype::F32,
                    transa: true,  // op(A=K) = Kᵀ  -> [Sk, D]
                    transb: false, // op(B=Q) = Q   -> [D, Sq] (col-major view)
                    m: sk,
                    n: sq,
                    k: d,
                    alpha: scale,
                    beta: 0.0,
                    a: k_head,
                    lda: d,
                    b: q_head,
                    ldb: d,
                    c: s_head,
                    ldc: sk,
                };
                // SAFETY: per-head pointers lie inside the validated dense Q/K
                // and freshly-allocated scores buffers; `workspace` is live;
                // `s_head` (output) aliases neither operand.
                unsafe { gemm_ex(blas, stream, &p, workspace, WORKSPACE_BYTES) }?;
            }
        }

        // Stage 2: fused softmax over the keys axis (scale already applied).
        let nrows = batch * self.num_heads * sq;
        let func = self
            .runtime
            .nvrtc_function(SOFTMAX_MODULE, SOFTMAX_SRC, SOFTMAX_ENTRY)?;
        let cfg = LaunchConfig {
            grid_dim: (nrows as u32, 1, 1),
            block_dim: (SOFTMAX_BLOCK, 1, 1),
            shared_mem_bytes: SOFTMAX_BLOCK * F32 as u32,
        };
        let nrows_i = i32::try_from(nrows).map_err(|_| {
            EpError::KernelFailed(format!("cuda_ep Attention: {nrows} score rows exceed i32"))
        })?;
        let (sk_i, sq_i, heads_i, batch_i) = (
            sk as i32,
            sq as i32,
            self.num_heads as i32,
            batch as i32,
        );
        let causal_i: i32 = self.causal.into();
        let stream_ref = self.runtime.stream();
        // Device pointers are passed by value (as u64) — a CUDA pointer kernel
        // parameter is ABI-identical to a 64-bit scalar argument.
        let mut builder = stream_ref.launch_builder(&func);
        builder
            .arg(&scores_buf)
            .arg(&mask_ptr)
            .arg(&nrows_i)
            .arg(&sk_i)
            .arg(&sq_i)
            .arg(&heads_i)
            .arg(&causal_i)
            .arg(&mask_planes)
            .arg(&batch_i);
        // SAFETY: `func` is the compiled softmax entry; the argument list and
        // its ABI match the kernel signature; `scores_buf`/`mask_ptr` are live
        // device allocations sized for [nrows, sk] / the mask planes.
        unsafe { builder.launch(cfg) }
            .map_err(|e| driver_err("launch attn_softmax_f32", e))?;

        // Stage 3: per-head O = P·V.  Column-major C[D,Sq] = Vᵀ·Pᵀ.
        for b in 0..batch {
            for h in 0..self.num_heads {
                let kv = h / group;
                let s_head = scores_buf + ((b * self.num_heads + h) * sq * sk) as u64 * F32;
                let v_head = v_base + ((b * self.num_kv_heads + kv) * sk * d) as u64 * F32;
                let o_head = o_base + ((b * self.num_heads + h) * sq * d) as u64 * F32;

                let p = GemmEx {
                    dtype: GemmDtype::F32,
                    transa: false, // op(A=V) = V  -> [D, Sk] (col-major view)
                    transb: false, // op(B=P) = P  -> [Sk, Sq] (col-major view)
                    m: d,
                    n: sq,
                    k: sk,
                    alpha: 1.0,
                    beta: 0.0,
                    a: v_head,
                    lda: d,
                    b: s_head,
                    ldb: sk,
                    c: o_head,
                    ldc: d,
                };
                // SAFETY: per-head pointers lie inside the validated dense V and
                // the softmaxed scores buffer and the dense output; `workspace`
                // is live; `o_head` aliases neither operand.
                unsafe { gemm_ex(blas, stream, &p, workspace, WORKSPACE_BYTES) }?;
            }
        }

        self.runtime.synchronize()
    }
}

impl Kernel for AttentionKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        self.run(inputs, outputs)
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        // §13.3 binding: the interface advertises strided support so the FA3 /
        // cuDNN drop-in (Phase 2b) needs no signature change. The Phase-2a
        // baseline validates contiguity and returns an actionable error for a
        // strided view rather than silently mis-reading it.
        true
    }

    fn cuda_graph_compatible(&self) -> bool {
        // §13.3 binding value. NOTE (Phase 2b): real capture additionally needs
        // the per-call scratch/workspace `alloc_raw`/`free_raw` replaced by a
        // pooled, stream-ordered allocator (same follow-up as the MatMul path).
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rt() -> Option<Arc<CudaRuntime>> {
        // `CudaRuntime::new` may *panic* (not just `Err`) when a CUDA library is
        // absent: cudarc's dynamic loader `expect()`s the shared object on first
        // use. Catch that so these GPU-gated tests skip cleanly on a host without
        // libcuda/libcublasLt, instead of failing the CPU-only suite.
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let runtime = std::panic::catch_unwind(|| CudaRuntime::new(0).ok().map(Arc::new))
            .ok()
            .flatten();
        std::panic::set_hook(prev);
        runtime
    }

    #[test]
    fn new_rejects_indivisible_gqa_groups() {
        let Some(runtime) = rt() else {
            eprintln!("skip: no CUDA GPU");
            return;
        };
        let e = AttentionKernel::new(runtime, false, 8, 3, None).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("multiple of"), "{msg}");
    }

    #[test]
    fn new_accepts_mha_and_gqa_and_mqa() {
        let Some(runtime) = rt() else {
            eprintln!("skip: no CUDA GPU");
            return;
        };
        // MHA (8/8), GQA (8/2), MQA (8/1) all divide cleanly.
        for kv in [8usize, 2, 1] {
            AttentionKernel::new(runtime.clone(), true, 8, kv, Some(0.5)).unwrap();
        }
    }
}
