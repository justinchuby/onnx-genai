# CUDA Phase-2b fused attention

## Backend decision

The implementation uses a custom NVRTC CUDA-C kernel.

`cudarc 0.19.8` exposes cuDNN's raw backend descriptor symbols, including SDPA
descriptor enums, but its safe `cudnn` API only wraps legacy
softmax/reduce/conv/pool operations. It does not provide the cuDNN frontend graph
builder, execution-plan selection, variant-pack lifecycle, or SDPA plan cache
needed for a maintainable fused-attention path. Building that raw descriptor
stack locally would carry more API/version risk than the kernel itself.

NVRTC preserves the crate's toolkit-free build: CUDA headers and libraries are
needed only when the runtime compiles/launches the kernel.

## Algorithm

The fused path is behind the existing `AttentionKernel` interface.

- One block owns a tile of query rows from one `(batch, query-head)` plane.
- K/V are read in 16-token tiles and shared by the query rows.
- QK scores exist only in shared memory.
- Each tile computes a stable local maximum and exponential sum.
- Running maximum `m`, denominator `l`, and output numerator `o` are merged with:

  ```text
  m' = max(m, mt)
  alpha = exp(m - m')
  beta  = exp(mt - m')
  l' = alpha*l + beta*lt
  o' = alpha*o + beta*ot
  ```

- The output is divided by `l` and rounded once to its storage dtype.

The f16 specialization uses WMMA tensor cores for 16x16 QK and probability-V
tiles. The general f32/f16/bf16 kernel widens storage to f32 and uses an
8-query-row tile. Causal alignment is `j <= Sk - Sq + i`; GQA maps
`kv_head = q_head / (num_heads / num_kv_heads)`.

## Coverage and fallback

| Case | Fused support | Automatic selection |
|---|---|---|
| f32, f16, bf16; contiguous `[B,H,S,D]` | `Sq > 1`, `D <= 128` | all dtypes through sequence span 128 |
| f16 tensor-core path | `D % 16 == 0`, SM70+ | through sequence span 512 |
| MHA / GQA / MQA | yes | as above |
| causal / non-causal | yes | as above |
| additive mask | shared, per-batch, or per-`(batch,head)` planes | as above |
| decode (`Sq == 1`) | Phase-2a fallback | always |
| `D > 128`, unsupported dtype/layout/mask | Phase-2a fallback | always |
| longer prefill | fused can be forced for memory-bounded use | Phase-2a is faster today |

`AttentionKernel::new` applies the measured heuristic.
`AttentionKernel::new_fused` and `new_phase2a` are parity/benchmark controls.
The registered operator uses `new`, so unsupported or slower shapes retain the
existing implementation without changing callers.

The standard-domain `Attention` and the separate
`com.microsoft::GroupQueryAttention` kernel keep their existing implementations;
the fused `com.microsoft::Attention` path itself covers grouped KV heads.

## Numeric validation

H200 parity tests execute fused and Phase-2a on identical inputs:

- f32: `atol=1e-4`, `rtol=1e-5`;
- f16: `atol=rtol=5e-3` (several half-rounding boundaries);
- bf16: `atol=rtol=3e-2` (7-bit mantissa).

Coverage includes small MHA, `S=129,D=128` causal GQA with an additive mask,
all three dtypes, and large-magnitude f32 scores that overflow a naive
`exp(score)` implementation. Observed maximum fused-vs-baseline error for the
non-trivial GQA case was `1.49e-7` f32, `4.88e-4` f16, and `3.91e-3` bf16.

## H200 benchmark

Device: NVIDIA H200, driver 580.105.08. Shape: causal f16
`B=1,H=32,D=128`; median of 10 runs at S=512 and 3 runs at S=2048 after warmup.
Timing includes the kernel's production allocation/synchronization behavior but
not input upload or output download.

| Sequence | Fused | Phase-2a | Result | Temporary global scratch |
|---:|---:|---:|---:|---:|
| 512 | 0.572 ms | 0.878 ms | **1.53x faster** | 0 vs 48 MiB |
| 2048 | 8.329 ms | 2.159 ms | 0.26x | 0 vs 288 MiB |

Phase-2a scratch is the f16 `[B,H,S,S]` score tensor plus the 32 MiB cuBLASLt
workspace. The fused kernel allocates neither. Automatic dispatch therefore
uses fused f16 through S=512 and retains Phase-2a at S=2048; callers prioritizing
memory over latency can explicitly select the fused path.

The next performance step is a larger multi-query tile / asynchronous K/V
pipeline, or a maintained cuDNN frontend SDPA wrapper.
