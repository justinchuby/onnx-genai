# Contiguous-VA-per-sequence KV: can the kernel be bounded so the tail is never committed?

Scope: the owner's directive for #750 — give each sequence one stable, flat,
contiguous virtual address for its whole context so the flat GQA attention
kernel keeps reading a plain `[heads, seq, head_dim]` buffer while KV is
physically scattered and grown on demand, paging invisibly underneath. If that
holds, multi-request batching is just N sequences each with its own stable
contiguous VA and the existing flat kernel works unchanged, replacing the
ragged/paged-attention route.

This note reports what is achievable **with evidence on this hardware**, and
confronts the known blocker (#721 stage 4) first.

## TL;DR

- **The crux — "can the decode kernel be bounded to the live length so the
  reservation tail is never committed?" — is answered by an isolating GPU test**
  (`crates/onnx-runtime-cuda-memory/tests/vmm_kv_contiguous_tail_gpu.rs`, 3
  tests, pass on device):
  - A read bounded to the committed live length — including a **captured graph
    replay**, the decode hot path — succeeds while the reservation tail stays
    physically **uncommitted**. The flat contiguous VA is real and the tail is
    physically absent.
  - Reading **one byte past** the live length into the uncommitted tail faults
    with `CUDA_ERROR_INVALID_VALUE`. This is #721 stage 4 reproduced in
    isolation: **the read pattern, not the reservation, forces the commit.** A
    kernel that reads the full padded shape dereferences the tail, so the tail
    can never be decommitted. The error is synchronous and non-sticky.
  - Even a perfectly length-bounded read cannot make a **fixed full-context
    stride** cheap: with the KV's per-binding/head-major layout each head's
    valid prefix lands in its own granule, so the commit floor is
    `objects x granule`, independent of content. Reproduces the documented
    qwen2.5-0.5b floor on hardware: **96 head-stripes x 2 MiB granule = 192 MiB
    committed for ~12 KiB of live content.**

- **Committed-bytes verdict (the number that decides adoption):**
  - *Fixed full-context stride* (the literal "one flat VA, never re-strided, so
    growth never re-captures"): pays the head-major granule floor —
    **192 MiB near-empty for qwen2.5-0.5b, 1.5 GB at 32K per #721 stage 4, a 32x
    regression over bucket growth's 48 MB.** Does **not** beat bucket growth.
  - *Bucketed stride under a stable VA* (already landed, `kv_commits_on_demand`
    in `native_decode/cuda.rs`, #682/#740/#748): commits the **current bucket**,
    not the full context, so committed bytes are **at parity with bucket
    growth**, with the added benefit that the virtual address is stable across
    growth (verified: `device_ptrs` unchanged, commit delta == bucket delta).
    But it **still re-strides and re-captures on growth**, because growing the
    head-major seq axis changes every head's stride.

- **Therefore the two goals the directive wants together — (1) low committed
  bytes and (2) no re-stride / no re-capture on growth — are in direct conflict
  under the head-major KV layout, and neither the fixed-stride nor the
  bucket-stride variant delivers both.** The kernel's read bound is *necessary*
  (proven: bounded reads leave the tail uncommitted) but *not sufficient*; the
  grow-axis stride sets the floor.

## Ground truth this builds on (verified, not re-derived)

- `#727` proved a captured CUDA graph replays correctly after
  `cuMemUnmap`/`cuMemCreate`/`cuMemMap` at the **same VA**. Growth is issued
  outside the captured segment; `cuMemMap` during capture is not proven
  replayable.
- `#740` provides the authority-scoped physical handle pool in
  `CudaVmmAllocator`; `#748` the transactional `MappedGrowthGrant`; `#745`
  committed-granule admission. The landed KV path draws from these — no second
  allocator, no per-sequence physical reservation.
- Measured granule on this device: **2,097,152 B (2 MiB)** — from both the
  driver directly and the live VMM arena log (`granules ... of 2097152 B`).
- qwen2.5-0.5b geometry (`genai_config.json` / `model.onnx`): 24 layers,
  key+value ⇒ **48 KV bindings**, each `[1, 2 kv_heads, 32768, 64]` **f16**;
  per-head stripe = `32768*64*2 = 4 MiB = 2 granules`; **96 head-stripes** total.
- The decode attention op is **`GroupQueryAttention`** (24 nodes). Its inputs
  include **`seqlens_k`** and **`total_sequence_length`** (the model's
  `v_model.Cast_8` / `v_model.Cast_11`), so its *compute* is length-bounded. The
  granule floor above is set by the *memory layout stride*, not by the compute
  bound — which is exactly why a bounded kernel is not enough on its own.

## Why the landed path is the right realization, and its residual cost

`native_decode/cuda.rs`'s `kv_commits_on_demand` branch (verified on real
qwen2.5-0.5b via
`native_cuda_vmm_kv_grows_in_place_and_commits_more_granules`):

- reserves the **full-context VA per binding** (free — address space only),
- commits only the **initial bucket** at load (`< full_context/2`, asserted),
- on growth: **keeps the same `device_ptrs`** (contiguous VA preserved),
  commits exactly the **next bucket delta**, repacks the valid prefix in place,
  then re-captures.

So it already delivers "contiguous VA + physically scattered + grown on demand"
at **bucket-stride** committed bytes. What it does **not** remove is the
re-stride/re-capture on growth, because the head-major seq axis means a larger
bucket changes every head's stride. Removing that would require a **fixed**
stride, which — per the granule-floor test — reintroduces the 32x floor.

## What would actually unlock the directive (next experiments, not done here)

1. **Commit below the bucket.** Since GQA is length-bounded via
   `seqlens_k`/`total_sequence_length`, in principle only
   `ceil(logical_len / granule_tokens)` granules per head need backing, not the
   whole bucket. The open question is whether the onnxruntime GQA **CUDA kernel**
   (and graph capture) ever touches the bucket tail `[logical_len, bucket)`
   (present-buffer initialization, flash-attention tiles). The isolating test
   shows that if it does, it faults on uncommitted sub-bucket granules; if it
   does not, sub-bucket commit is safe and beats bucket growth on committed
   bytes. This needs an engine change (commit to `logical_len`, not bucket) plus
   a byte-identical decode run — high value, deferred.
2. **Seq-major (token-major) KV layout** so the grow axis is outermost and
   appends land contiguously at the tail: one granule at the end per growth, no
   re-stride, no re-capture, and no per-head granule floor. This is the only
   layout that gives *both* low commit *and* no re-capture — but the past/present
   KV shape is baked into the exported model and the onnxruntime GQA kernel, so
   it requires re-exported models or a custom attention kernel.

## Does `onnx-genai-kv`'s page machinery still matter, and who owns paging?

Direct answer to the owner's question:

- **The native CUDA decode path has no production consumer of
  `onnx-genai-kv`'s `PageTable`/`PagedKvCache`.** `native_decode/cuda.rs`
  contains no `PageTable`/`PagedKvCache` reference; `paged_gqa.rs` is a batch-1
  **CPU** primitive with no CUDA consumer. Per `docs/MEMORY_ARCHITECTURE.md` the
  KV page store is **host-only** and "never touched device memory."
- **Under the VMM contiguous-VA design, device-side KV paging is owned by the
  CUDA VMM layer** — `CudaVmmAllocator` (#740 pool) mapping granules under a
  stable per-binding reservation, with growth/attribution through the governor
  (`MappedGrowthGrant` #748, committed-granule admission #745). The attention
  kernel sees a flat contiguous buffer and **never learns page tables** — paging
  is `cuMemMap` under a fixed VA, exactly as directed.
- **Implication:** for the CUDA batch path, `onnx-genai-kv`'s page machinery is
  **not needed** as the device KV mechanism; a batch is N sequences each with
  its own stable contiguous VMM reservation, and the flat GQA kernel is
  unchanged. `onnx-genai-kv` remains relevant for the **CPU** paged-GQA path and
  for host-side concerns (KV page store, prefix sharing / CoW, quantization
  layout) — i.e. as a host KV store and layout library, not the device pager.

## Correctness / measurement bar status

- **Correctness:** the isolating tests assert byte-level read correctness of the
  committed prefix (captured-graph replay fills the exact live length). The
  landed engine path's decode byte-identity (capture on/off) is covered by the
  existing native-CUDA smoke/verify tests, which pass on qwen2.5-0.5b.
- **Deterministic counters (contiguous-VA / VMM path, real qwen2.5-0.5b):**
  granule 2 MiB; load commits only the initial bucket (`< full_context/2`);
  growth keeps `device_ptrs` stable and commits exactly the bucket delta. These
  are the byte/event counters the throughput-variance-prone box cannot obscure.
