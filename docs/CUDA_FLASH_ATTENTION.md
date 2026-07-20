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

The fused path is shared by the existing `AttentionKernel` interface and the
multi-token prefill path of `com.microsoft::GroupQueryAttention`.

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

### GroupQueryAttention

GQA reuses its existing preparation pipeline: packed-QKV splitting, BSH→BNSH
transpose, cache append/build, and RoPE all run before attention. The fused kernel
therefore consumes the same post-RoPE query and post-append present K/V buffers as
the baseline. It uses the physical cache capacity only as the BNSH stride and
iterates to `max(seqlens_k + 1)`. Cache construction and implicit RoPE positions
use the physical append origin `total_length - key_sequence_length`; causal
attention instead uses the query origin
`total_length - query_sequence_length`, so query row `i` can attend through
`total_length - query_sequence_length + i`. These origins are equal for ordinary
cached prefill but deliberately differ when `Sq != Sk`.

| GQA case | Fused prefill | Fallback / existing behavior |
|---|---|---|
| f32, f16, bf16; packed or unpacked QKV | `Sq > 1`, `D <= 128` | decode and `D > 128` use baseline |
| fresh prefill / prefill with past | yes | same cache append/build and RoPE preparation |
| MHA / GQA / MQA head sharing | yes | same `q_head / group` mapping |
| ragged `seqlens_k`, fixed-capacity cache | yes | per-batch logical total/query-origin pointers |
| local/sliding window and softcap | yes | identical mask order to baseline |
| f32/bf16 automatic selection | logical total length through 128 | longer shapes use baseline |
| f16 tensor-core automatic selection | logical total length through 512, `D % 16 == 0`, SM70+ | longer shapes use baseline |
| forced fused parity/benchmark mode | all supported prefill shapes | unsupported shapes fall back |
| attention bias, head sink, smooth-softmax, QK capture, quantized KV | not implemented by the current CUDA GQA baseline | existing actionable rejection is unchanged |

The public `GroupQueryAttentionBackend` control mirrors `AttentionKernel`'s test
controls: production factory-created kernels use `Auto`; parity and benchmark
tests select `Fused` or `Phase2a` directly.

## Numeric validation

H200 parity tests execute forced fused and forced Phase-2a on identical inputs:

- f32: `atol=3e-6`, `rtol=2e-6`;
- f16: `atol=rtol=7e-4`;
- bf16: `atol=rtol=5e-3`.

The tolerances closely bracket the observed storage-rounding boundaries:
`2.52e-6` f32 (the large-magnitude stress case), `4.88e-4` f16, and
`3.91e-3` bf16.

The table-driven GroupQueryAttention matrix covers every dtype across MHA, GQA,
and MQA for fresh-uniform, cached-uniform, and cached-ragged inputs. It also
forces fused-vs-baseline parity with RoPE, local/sliding window plus softcap, a
non-WMMA-multiple f16 head dimension (`D=72`, generic kernel), and
large-magnitude online softmax. A dedicated `Sq=2,Sk=4` regression covers both
fresh and cached-ragged batches in all three dtypes; it verifies causal origin is
`total-Sq`, while present-cache append remains `total-Sk`. Present K/V buffers
match exactly in every case.

Separate Auto tests assert and execute Phase-2a fallback for decode (`Sq=1`) and
the measured-slower cached shape `Sq=512,total=1024`; both outputs and caches
match forced Phase-2a exactly.

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

### GroupQueryAttention prefill

Device: NVIDIA H200, driver 580.105.08. Causal f16
`B=1,H=32,KVH=8,D=128`; median of 10 runs at `Q=512` and 3 runs at `Q=2048`
after warmup. Timing includes production GQA transpose/cache preparation,
allocation, and synchronization, but not input upload or output download.

| Query | Past | Fused | Baseline | Result | Baseline attention scratch |
|---:|---:|---:|---:|---:|---:|
| 512 | 0 | 1.014 ms | 1.324 ms | **1.31x faster** | 48 MiB |
| 512 | 512 | 1.581 ms | 1.370 ms | 0.87x | 64 MiB |
| 2048 | 0 | 8.783 ms | 2.429 ms | 0.28x | 288 MiB |
| 2048 | 2048 | 17.788 ms | 4.134 ms | 0.23x | 544 MiB |

The fused path uses zero global attention scratch in all four cases. Automatic
dispatch selects the 512-token fresh-prefill win and transparently retains the
baseline for the cached 1024-total-token and both 2048-query cases.

The next performance step is a larger multi-query tile / asynchronous K/V
pipeline, or a maintained cuDNN frontend SDPA wrapper.
