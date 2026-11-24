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


### Prepared fallback workspace

`com.microsoft::Attention` obtains its Phase-2a scores and cuBLASLt scratch
through the executor workspace contract. Session/plugin and direct callers
obtain the requirement from input metadata and pass an exclusive prepared view
to `execute_with_workspace`. Planning and binding share `WorkspaceLayout`, with
checked shape multiplication and alignment. Short, null, misaligned, wrapping,
or unaddressable prepared workspace is rejected with preparation guidance.

A direct `execute` call is valid when the selected fused route needs no global
scratch. Phase-2a rejects a missing workspace rather than allocating outside
preparation. Eager registration currently exposes only CPU execution; future
asynchronous registration needs an explicit completion/retirement contract.

GroupQueryAttention's general half/bf16 fallback also borrows the complete
Phase-2a score/32 MiB BLAS region from its existing composite workspace. One
route classifier drives KV-read support, execution and scratch planning. The
input-derived `past_capacity + current_key_length` bound covers every legal
valid length without inspecting device length values during preparation. Auto
planning can reserve fallback scratch even when a shorter individual invocation
selects fused; guaranteed-fused and dedicated decode shapes reserve neither
matrix nor BLAS scratch. Missing or invalid fallback backing is an error.

GQA now requires its composite for all large temporary buffers: packed staging,
Q/output transposes, scores/BLAS, and hidden KV outputs. There is no private
allocation fallback for these regions. A single visible K or V output fixes the
physical capacity of its hidden sibling; when both are omitted, both temporary
caches use the input-derived logical bound. Planning receives actual output facts
and rejects unresolved visible cache geometry when a hidden sibling needs that
capacity before admission. Complete backing is checked before metadata
preparation or attention launches.

The three sequence metadata arrays are now a checked final region of that same
composite. GQA preparation owns no allocation or pool and returns no private
graph resource.
During capture, the complete used view must be covered by one registered, real
workspace pin. Unregistered capture, arbitrary tokens, and insufficient spans
are refused before metadata preparation. Manual callers pass the workspace
resource to capture and retire the graph before returning its backing.

Capacity remains SessionPersistent: a deferred device-binding decode can return
before completion, so step retirement could reintroduce admission on every token.
Metadata contents remain temporary. This retention policy adds no per-token sync
or issuer operation. Even an empty metadata group funds one backing byte for its
legal empty windows rather than relying on every downstream launch being skipped.
Dedicated f32/f16/bf16 split-K decode now borrows partial states from the same
composite. Its checked bound is `min(rows, row_cap) * max_splits * (head_dim + 2)`
f32s. Only rows after the cap execute direct; earlier rows still use partials.
All split columns remain funded even for a short/direct invocation, so changing
the direct-output test/configuration or a device-resident valid length cannot
invalidate preparation. Host and NVRTC row/split limits come from one definition.
The three fixed module-global arrays are removed. Partial and merge launchers
receive the same borrowed pointer; their reduction order and launch geometry are
unchanged. Actual CUDA parity, NVRTC compilation and throughput require a device
runner and are not established by Rust compilation.

`com.microsoft::MultiHeadAttention` also borrows one SessionPersistent composite:
Q/output BNSH staging, optional temporary K/V, additive mask, padding upload,
and the complete Phase-2a scores/cuBLASLt region. Compile-time optional slots
and supplied output metadata are checked before omitting K/V regions. Planning
and execution share geometry and output validation; regions receive no separate
grants. Retained capacity permits fitting device-binding runs to avoid repeated
admission even when the existing synchronization policy defers completion.

MHA no longer allocates or frees device scratch. Its stream-use guard preserves
the existing synchronization policy and both execution/policy errors; a deferred
sync returning Ok is not a completion witness. Provider fences and settlement
still own physical retirement. A failed legacy-default-stream padding upload
latches that stream into subsequent release fences, so compute/copy completion
alone cannot release an unsettled destination. The host padding mask remains an
owned Vec; this is not a complete process-memory bound. Capture stays refused.

All Phase-2a callers now pass typed, checked caller scratch. Its raw-allocation
fallback and runtime-private shared BLAS allocation are removed. Borrowed score
capacity is validated against the actual invocation before launches. This is a
source/Host-tested contract, with CUDA execution and performance still pending.

The fallback retains the existing runtime synchronization policy and capture
exclusion. The parity tests use the public provider workspace preparation path.
The benchmark prepares and reuses its admitted backing outside timing, then
explicitly waits for device completion before recording each wall-clock sample;
its samples include dispatch through completion, rather than enqueue latency.
