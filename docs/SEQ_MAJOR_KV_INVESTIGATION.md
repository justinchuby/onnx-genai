# Seq-major (BSNH) KV layout: investigation & measurement

Status: investigation + measurement + narrow prototype. **No production kernel
conversion in this round.** References #750 (multi-request batching), #759 (VMM
dummy-page / unified KV strategy), **#776 (granularity probe — granule lever
dead)**, and **#777 (page-level prefix sharing)**; builds on #522 and #726
(per-backend KV ownership), and on #727 / #740 / #745
(VMM multi-map proof, physical-handle pool, granule-ref accounting).

**Reordered per owner priority (#777):** page-level prefix sharing is now the
**co-primary justification** for seq-major, alongside the ORT BSNH compatibility
crux — ahead of the granule-floor saving that originally motivated it. Rationale:
the floor is a constant that amortizes with context length, whereas prefix
sharing scales with **concurrency**, the axis multi-request serving cares about.
See §3 (promoted, with capacity consequences and a mechanism check against the
real allocator/ledger).

Context: KV is stored head-major (BNSH `[batch, kv_heads, seq, head_dim]`).
Under a full-context virtual reservation, each head's live prefix lands in its
own 2 MiB granule (head stripes are `max_seq * head_dim * dtype` apart), so a
one-token qwen2.5-0.5b prefill commits **96 stripes × 2 MiB = 192 MiB for
~12 KiB of live content** (measured, draft #772
`crates/onnx-runtime-cuda-memory/tests/vmm_kv_contiguous_tail_gpu.rs`). The
model-size-independent crossover where content fills those granules is
`granule / (head_dim × sizeof(dtype))` ≈ **16,384 tokens** (head_dim 64/fp16)
and **8,192 tokens** (head_dim 128/fp16) at a 2 MiB granule.

Seq-major (BSNH `[batch, seq, kv_heads, head_dim]`) **shrinks that floor by a
factor of `kv_heads`** — it does *not* remove it (correction folded in, verified
below). Each layer's K and V remain **separate tensors/bindings**, so the number
of separately-strided regions that each floor to their own granule goes from
`layers × 2 × kv_heads` (head-major: every head is its own stripe) to
`layers × 2` (seq-major: all heads of a layer-side are contiguous, so the whole
buffer shares one stripe). The token stride within a buffer becomes
`kv_heads × head_dim`, independent of sequence length, so growth never moves an
offset and never invalidates a captured CUDA graph.

**Corrected floor + crossover table** (verified numerically; matches owner):

| layout | floor granules | qwen14b (48L, kv8, hd128) | qwen0.5b (24L, kv2, hd64) | crossover formula |
|---|---|---|---|---|
| head-major (today) | `layers×2×kv_heads` | 768 → 1.50 GiB | 96 → 192 MiB | `granule/(hd·dtype)` |
| **seq-major** | `layers×2` | 96 → 192 MiB | 48 → 96 MiB | `granule/(kv_heads·hd·dtype)` |
| token-major (all layers) | `1` | 1 → 2 MiB | 1 → 2 MiB | — |

The seq-major crossover is `granule / (kv_heads × head_dim × sizeof(dtype))`:
- qwen14b: 2 MiB / (8·128·2) = **1,024 tokens** (vs 8,192 head-major → **8×**).
- qwen0.5b: 2 MiB / (2·64·2) = **8,192 tokens** (vs 16,384 head-major → **2×**).

**Important consequence of the correction: the seq-major crossover is NOT
model-size independent** — it scales with `kv_heads`. So seq-major looks much
better on wide-GQA models (qwen14b, 8 kv-heads → 8×) than on narrow ones
(qwen0.5b, 2 kv-heads → 2×). The head-major crossover *is* model-independent
(`layers`/`kv_heads` cancel, hardware-verified by #776); seq-major trades that
independence for a `kv_heads`-proportional improvement.

**Token-major across all layers** (one reservation, all layers' K/V interleaved
by token) would take the floor to a **single granule** (2 MiB, both models). It
is *not prototyped* here per scope, but evaluated in §2 against the measured
strided-read numbers: because those reads are cheap, token-major is worth a
future experiment — with the caveat that its read stride is ~`layers×2` larger
than the seq-major stride actually measured, so its TLB/locality cost is the open
question a targeted microbench would have to settle before committing.

**The granule lever is dead on this hardware (#776 / draft #776,
`vmm_granularity_gpu.rs`), independently re-confirmed by this round's
`remap_burst` bench (`granule MINIMUM == RECOMMENDED == 2 MiB`).** The parallel
#759 probe queried both granularities for the exact production allocation
properties and found `CU_MEM_ALLOC_GRANULARITY_MINIMUM ==
CU_MEM_ALLOC_GRANULARITY_RECOMMENDED == 2 MiB` — **there is no 64 KiB device
granule.** So the alternative fix for the `objects × granule` floor — shrink the
granule and collapse the crossover ~32× — **cannot be pulled.** The head-major
crossover stays at 8,192 tokens (head_dim 128/fp16) and 16,384 (head_dim 64/fp16).
The same probe hardware-verified the head-major derivation across a 480×
model-size range plus both real models: the floor unit is genuinely
per-head-stripe, `layers` and `kv_heads` cancel, and `crossover_headmajor =
granule / (head_dim × sizeof(dtype))` holds.

**Quantized KV makes the head-major floor worse, not better.** fp8/int8 halves
bytes per token, which *doubles* the crossover in tokens (fewer live bytes fill
each granule more slowly). Since we are independently moving toward quantized KV,
the two optimizations fight each other under head-major — but not under
seq-major, which cuts the crossover by `kv_heads` regardless. And §2 shows the
strided read stays free even at quantized (32 B) run sizes, so seq-major is
compatible with the quantized-KV direction while head-major is antagonistic to it.

**Consequence:** with the granule lever dead, seq-major is the only *layout* lever
that lowers the floor without a finer granule, and it does so by exactly
`kv_heads` (14B: 8×; 0.5b: 2×). It does **not** remove the floor concept —
`layers × 2` granules remain because K and V stay separate bindings — but it
brings the crossover down to a reachable context length (1,024 tokens on 14B).
Combined with #777 prefix sharing (practical only under seq-major, §3) and the
measured remap-burst reduction (§2.1), this is the convergence point for three
goals: lower committed bytes, no CUDA-graph re-capture on growth, and
cross-request prefix sharing. This investigation is therefore **decision-critical**,
not exploratory.

---

## 1. THE CRUX — does ORT GroupQueryAttention support BSNH past/present KV?

**Definitive answer: NO.** ORT GQA past/present KV is **BNSH-only**, unconditionally,
on every dispatch path. BSNH past/present is *not reachable* through the public
op contract on current ORT `main`.

Inspected: `microsoft/onnxruntime` `main` (verified against
`raw.githubusercontent.com/.../main/...`; research agent inspected commit
`e415ef9afd877f74068a7caf167982abce30e470`, 2026-08-11).

Primary-source evidence:

- **Hardcoded format.**
  `onnxruntime/contrib_ops/cpu/bert/group_query_attention_helper.h`, in
  `CheckInputs`:
  ```cpp
  AttentionQkvFormat qkv_format = Q_K_V_BSNH;      // query
  AttentionQkvFormat past_kv_format = Q_K_V_BNSH;  // past/present KV — always BNSH
  ...
  output_parameters->past_kv_format = past_kv_format;  // never anything but BNSH
  ```
  `past_kv_format` is a local, assigned once, never reassigned to BSNH.

- **Input validation is BNSH.** `CheckPast` (same file) requires
  `past_key_dims[1] == kv_num_heads` (dim 1 = heads ⇒ BNSH) and the mismatch
  error string literally reads `"BNSH Input 'past_key' and 'past_value'..."`.
  A BSNH tensor `[B, S, N_kv, H]` would put `S` at dim 1 and fail validation.
  *(Verified directly against upstream raw source.)*

- **Present output shape is BNSH.**
  `contrib_ops/cuda/bert/group_query_attention.cc`:
  `present_dims = {batch_size, kv_num_heads, seqlen_present_kv_cache, head_size}`.

- **`is_past_bsnh_` exists but is dead for GQA.** Declared in
  `group_query_attention.h`, set `false` in the constructor, no `GetAttr`
  toggles it. There is no attribute, input shape, env var, or build flag that
  selects BSNH present for GQA.

- **Per-backend gating (all require BNSH):**
  - **cuDNN SDPA**: `use_cudnn_sdpa` guard includes
    `parameters.past_kv_format == AttentionQkvFormat::Q_K_V_BNSH` explicitly.
  - **FlashAttention**: writes into the BNSH `present_dims` buffer.
  - **XQA / TensorRT**: `LaunchCopyKvCacheWindow` is documented "BNSH KV caches"
    and requires `past_present_share_buffer`.
  - **Memory-efficient**: `LaunchUngroup`'s `is_bsnh` flag refers to the
    head-expansion *scratch* buffer, not the past/present cache.

- **`AttentionQkvFormat`** (`contrib_ops/cpu/bert/attention_common.h`) does
  define `Q_K_V_BSNH`, and `LAYOUT_BSNH=false/LAYOUT_BNSH=true` constants exist —
  but GQA's cache path never selects them.

**Consequence (the branch we are on): KV layout is a per-backend capability, not
a global constant.** The native backend can use seq-major; the ORT backend stays
BNSH; **each backend owns its own KV buffers**. This is exactly the direction
already recorded: #522 ("the two backends share no KV buffer ownership
contract") and #726 (the `KvPageStore` contract). The contract change required:

> The KV storage contract must expose layout as a per-backend property
> (e.g. `KvLayout::{HeadMajorBnsh, SeqMajorBsnh}`) carried by the KV store /
> page store, and the engine must stop assuming a single physical KV ordering
> shared across backends. Capacity-growth (`KvCapacityGrowthBackend`) and
> ownership (`KvPageStore`) already separate cleanly (#522); layout attaches to
> ownership, not to the op contract.

**Explicitly rejected:** a per-step transpose to bridge native↔ORT layouts.
Decode reads the *entire* KV every step, so a per-step transpose costs a full
extra KV pass on the hot path — disqualifying. Do not propose it. Divergent
per-backend buffers are correct precisely because KV never crosses the backend
boundary on the native path (see §4).

Bonus finding relevant to §3: ORT GQA *does* support a shared-KV mode
(`kv_sequence_length == 0`, past buffer already holds the full shared cache, no
append) — so ORT has a notion of prefix sharing even though its layout is BNSH.
This is a logical shared-cache input, not physical page-level sharing across
distinct sequences' VA ranges; the #777 prize (§3) still requires seq-major.

---

## 2. The one thing that can falsify seq-major: strided decode reads — MEASURED

Seq-major makes the per-head read strided: `K_h[0..L]` moves from a contiguous
`L × head_dim` run to a strided matrix with leading dimension
`kv_heads × head_dim`. Decode is bandwidth-bound and reads all KV every step, so
this lands on the hot path.

**Result: no measurable penalty**, including the predicted danger case.

Microbenchmark: `bench-seqmajor/` (this branch). One warp per (layer, kv-head),
fused online-softmax single pass reading K and V exactly once each — faithful to
the decode KV traffic and to the warp-shuffle reduction in `gqa_decode_fp16`.
The two kernels differ **only** in the token address stride:
`head-major k[(head*L+t)*head_dim]` vs `seq-major k[t*heads*head_dim +
head*head_dim]`; total bytes moved is identical. Runs interleaved
(H,S,H,S,…), 40 reps after 8 warmups, min+median reported to fight this box's
extreme variance. Device: sm_89 (Ada), CUDA-13 NVRTC compiled straight to cubin
(driver rejects the CUDA-13 PTX ISA — same cubin fallback as `runtime.rs`).

Median seq/head-major time ratio (>1.0 = seq-major slower):

| config (kv_heads, head_dim, layers) | contig run | L=512 | L=2048 | L=8192 | L=32768 |
|---|---|---|---|---|---|
| **q-run 32 B — int4 KV** (8, hd=16, 40) | 32 B | 1.003 | 0.833 | 0.988 | 1.001 |
| **head_dim=32 danger** (8, 32, 40) | 64 B | 1.002 | 0.949 | 0.998 | 0.999 |
| qwen0.5b (2, 64, 24) | 128 B | 0.995 | 1.003 | 0.999 | 0.998 |
| kv8 (8, 64, 40) | 128 B | 0.810 | 0.980 | 0.988 | 0.993 |
| qwen14b (8, 128, 48) | 256 B | 1.014 | 1.016 | 1.006 | 1.006 |

Seq-major is within ±2% of head-major everywhere (and *faster* in several
cases). **The predicted `head_dim × dtype < 128 B` cache-line-halving does not
occur — not even for a 32 B contiguous run** (int4 KV at head_dim 64, or fp8 at
head_dim 32), which is a quarter of a 128 B cache line. Reason: heads are
contiguous in the seq-major layout, so a cache line holding head *h*'s partial
run also holds head *h+1*'s run, and all heads of a layer are read by concurrent
warps — L2 recovers the neighbor. The contiguous run being smaller than a cache
line is harmless as long as the adjacent head is co-resident and co-read, which
it always is.

**Quantized KV covered.** The bench's `head_dim` is the count of contiguous
fp16 elements per token run, so a `B`-byte run is modelled by `head_dim = B/2`;
the 32 B and 64 B rows above stand in for int4/fp8 KV whose runs shrink below
64 B. This matters because we are independently moving toward quantized KV
(§ intro), and the result says the strided read stays free there too.

**Decode-step cost, not just synthetic bandwidth.** The reported ratio is the
end-to-end kernel time of a faithful fused decode step (K+V read once, softmax,
weighted V), interleaved H/S — i.e. the decode-step cost difference is what is
tabled, and it is ≤2%.

Caveat: absolute GB/s here (10–225 GB/s) is below peak DRAM because the
single-warp-per-head fused kernel is latency/occupancy-bound at these block
counts — but *both layouts are equally bound*, so the relative delta (what we
care about) is valid and near-zero. A production seq-major kernel would use the
same split-K / multi-warp strategy as today's head-major kernel; the strided LD
only changes the base-offset arithmetic, not the coalescing within a token row.

**Bearing on token-major (owner item 1).** Token-major-across-all-layers would
push the read stride to `layers × 2 × kv_heads × head_dim` — roughly `layers×2`
larger than the seq-major stride actually measured above — in exchange for a
1-granule floor. The measured result says the *stride magnitude itself* is not
what costs bandwidth (seq-major already strides by `kv_heads × head_dim` with
zero penalty because each token still contributes a contiguous `head_dim × dtype`
run). So token-major is **worth a future experiment** on the strength of these
numbers — but not conclusively free, because the untested risk at that stride is
TLB/page-locality (each per-token run now lands in a different 2 MiB granule for
the *same* layer), not cache-line coalescing. Verdict: promising, gated on a
targeted stride-sweep microbench; not dead, not proven. Do not prototype this
round.

---

## 2.1 Remap cost is a lockstep BURST, not a per-token tax — MEASURED

Owner correction folded in: remap frequency is a non-issue (a granule holds many
tokens of headroom — ~8,192 tokens per head-major stripe, ~1,024 per seq-major
buffer on qwen14b — so remaps are naturally batched and amortize to <0.2% of a
decode step). **The real cost is that all `layers × 2 [× kv_heads]` buffers cross
their granule boundary on the *same* decode step**, because they grow in lockstep
with sequence length, so the cost lands as a single-token latency spike.

Measured directly on this hardware with a new real-VMM microbench
(`bench-seqmajor/src/bin/remap_burst.rs`, driver `cuMemCreate` + `cuMemMap` +
`cuMemSetAccess` into a pre-reserved range — the exact production commit path):

| scenario (buffers crossing together) | N granules | measured burst |
|---|---|---|
| qwen0.5b seq-major (`layers×2`) | 48 | **2.8 ms** |
| qwen0.5b head-major (`layers×2×kv`) | 96 | **4.9 ms** |
| qwen14b seq-major (`layers×2`) | 96 | **4.5 ms** |
| qwen14b head-major (`layers×2×kv`) | 768 | **49.5 ms** |

Per-granule commit measured **~48–64 µs** (median of 32), *lower* than the #776
probe's ~150 µs/granule estimate — likely because that figure folded in
first-touch page-in; the pure reserve→map→set-access path is cheaper. Granule
independently re-confirmed **MINIMUM == RECOMMENDED == 2 MiB**.

**Reading the numbers honestly:** against a ~12 ms/token decode step, qwen14b
head-major's **49.5 ms** burst is a ~5× step blow-up — a very visible stutter in
streaming inter-token latency. Seq-major cuts it to **4.5 ms** (≈`1/kv_heads`),
about a third of a step — noticeable but far milder. So the seq-major win here is
the *same* `kv_heads` factor as the floor, now on the growth-spike axis. This is
a second, independent reason to prefer seq-major that is orthogonal to committed
bytes.

**Mitigation 1 — commit ahead of the write frontier (evaluated, recommended).**
Because `cuMemMap` can run on any step, a high-water-mark-plus-slack policy
commits the next granule(s) *before* the write frontier reaches them, moving the
remap off the critical decode step entirely. This is functionally what today's
bucket-growth realloc already does — **the VMM advantage is that the virtual
address does not change**, so pre-committing does **not** invalidate a captured
CUDA graph (bucket growth reallocates and forces re-capture). That is the
concrete, stateable win: same pre-growth policy, zero graph re-capture.

**Mitigation 2 — stagger the boundaries (evaluated; achievable, not inherent).**
The lockstep spike is an artifact of every buffer being granule-*aligned* at the
same phase, so they all cross together. Giving each buffer a different sub-granule
starting offset (phase) — e.g. pre-committing `i mod k` extra tokens of slack on
buffer `i`, or reserving each buffer at a staggered base — spreads the crossings
across up to `crossover` distinct steps, flattening the `N × per_granule` spike
to ~`per_granule` per step. This is correctness-neutral: each (layer, side)
buffer is independent, nothing requires their boundaries to coincide, and the
only cost is a bounded amount of extra committed slack. **So lockstep is NOT
inherent** — it is the default that staggering removes. Combined with
Mitigation 1, the growth spike is fully removable; even unmitigated, seq-major
already shrinks it by `kv_heads`.

---

## 3. CO-PRIMARY: page-level prefix sharing (#777) — quantified prize, capacity consequence, mechanism check

Under seq-major a shared prefix `[0..P]` across all heads is a **single
contiguous VA range** per (layer, side), so cross-request sharing is "map the
same physical granules into several sequences' address spaces" — free prefix
caching with granule-granularity copy-on-write. Under head-major the same prefix
is `layers × 2 × kv_heads` scattered fragments. This is the #777 prize and it is
now considered the primary reason to adopt seq-major (§ header).

### 3.1 Byte saving — checked, not trusted

`bytes_per_token = layers × 2 × kv_heads × head_dim × sizeof(dtype)`
(qwen2.5-0.5b = 12 KiB; qwen2.5-14b = 196,608 B = 192 KiB — matches #777).
Duplicated KV removed by sharing a `P`-token prefix across `R` requests is
`(R-1) × P × bytes_per_token` (ideal). But page sharing only banks **whole
granules**, and the shareable unit is a per-(layer,side) stripe of
`P × kv_heads × head_dim × dtype` bytes, so the granule-rounded saving is
`(R-1) × floor(P × kv_heads × head_dim × dtype / granule) × granule × layers × 2`.

| model | P (tok) | R | ideal removed | **granule-rounded removed** |
|---|---|---|---|---|
| qwen2.5-14b | 512  | 8  | 0.70 GiB  | **0** (sub-granule stripe) |
| qwen2.5-14b | 2048 | 4  | 1.12 GiB  | **1.12 GiB** |
| qwen2.5-14b | 2048 | 8  | 2.62 GiB  | **2.62 GiB** |
| qwen2.5-14b | 2048 | 16 | 5.62 GiB  | **5.62 GiB** |
| qwen2.5-14b | 8192 | 8  | 10.50 GiB | **10.50 GiB** |
| qwen2.5-0.5b | 2048 | 8  | 0.16 GiB  | **0** (sub-granule stripe) |
| qwen2.5-0.5b | 8192 | 8  | 0.66 GiB  | **0.66 GiB** |

**#777's headline confirmed:** 2048-token prefix × 8 concurrent on qwen14b
removes **2.62 GiB** (owner's "≈2.6 GB" is correct; granule-rounded is *exactly*
equal here because 2048 tok × 2048 B/tok/stripe = 4 MiB = 2 whole granules).

**New rigor finding (not in #777):** the saving has a *granule floor of its own*.
Page sharing banks a granule only when the per-(layer,side) stripe prefix
`P × kv_heads × head_dim × dtype ≥ granule`, i.e.
`P ≥ granule / (kv_heads × head_dim × dtype)`:

- qwen2.5-14b (kv=8, hd=128): **P ≥ 1024 tokens**.
- qwen2.5-0.5b (kv=2, hd=64): **P ≥ 8192 tokens**.

Below that the shared prefix is sub-granule per stripe and page sharing saves
**nothing** — the same granule-floor problem, now on the sharing side. So the
prize is large for **big models with prefixes ≥ ~1–2K tokens** and evaporates for
small models / short prefixes unless the arena packs several stripes into shared
granules (a layout-packing question, out of scope here). The headline #750
scenario (14B, multi-K-token system prompt) is solidly in the win zone.

### 3.2 Capacity consequence — the number that matters

On a fixed KV budget, sharing the prefix once (charged once) lets the Nth request
cost only its *private* bytes. Granule-rounded (shared prefix in whole granules;
private tail rounded up per stripe), same-prompt requests of `ctx` total tokens:

| model | KV budget | ctx | prefix | no-share | **shared** | gain |
|---|---|---|---|---|---|---|
| qwen2.5-14b | 2 GiB | 2560 | 2048 | 3 | **8** | 2.7× |
| qwen2.5-14b | 4 GiB | 2560 | 2048 | 7 | **19** | 2.7× |
| qwen2.5-14b | 6 GiB | 2560 | 2048 | 10 | **30** | 3.0× |
| qwen2.5-14b | 4 GiB | 4096 | 2048 | 5 | **9** | 1.8× |
| qwen2.5-14b | 2 GiB | 2560 | 512  | 3 | **5** | 1.7× |
| qwen2.5-0.5b | any | any | ≤2048 | — | — | 1.0× (sub-granule) |

**#777's "2 vs 8" claim confirmed and honest:** qwen14b, 2 GiB KV budget,
2560-token context with a 2048 shared prefix, goes from **3 → 8** concurrent
requests (≈2.7×); at 6 GiB, **10 → 30**. The gain grows with the shared fraction
of the context and is a whole *multiple*, not a percentage — which is why it
outranks percentage-level optimizations elsewhere. (qwen0.5b shows 1.0× because
both its shared prefix and private tail are sub-granule at these lengths — the
granule floor dominates the small model regardless.)

### 3.3 Mechanism — confirmed against the REAL allocator/ledger, not a mock

Verified in `crates/onnx-runtime-cuda-memory/src/{virtual_memory.rs,vmm_allocator.rs}`:

- **Multi-map primitive is proven and available.** #727 already demonstrated on
  this hardware that **one physical handle mapped at two live VAs works**
  (captured graph, a DtoD copy from alias A saw bytes written through alias B; no
  recycling confound because both aliases were simultaneously live). This
  generalizes to N: `cuMemMap` of one handle into N ranges is the same primitive
  repeated, and CUDA imposes no N-limit beyond address space. The wrappers exist
  (`virtual_memory.rs`: `cuMemMap` @1111-1123, `cuMemSetAccess` @1126-1143,
  `cuMemAddressReserve` @1053-1060).
- **Charge-once / keep-alive-until-last-unmap is the real ledger's shape.**
  `Spans.granule_refs: Vec<u32>` (`vmm_allocator.rs:318-337`) is refcounted:
  first claim sets a granule to 1 and charges (`853-856`); a shared claim
  increments (`812-814`); release is **refcount-gated** — only
  `Some(0) => backing.release(...)` frees, otherwise just decrements
  (`907-924`). Pooled physical handles (#740) are charged to the authority as
  `Backing { authority }` once, **not per mapping** (`1043-1050`,
  `physical_memory_accounting`), so a shared granule is already accounted once
  regardless of map count. This is exactly the "charge the Nth request only its
  private bytes; keep the shared granule alive until the last sharer unmaps"
  contract #777 asks for.
- **The one real gap (bookkeeping, not capability).** The allocator's per-block
  record `CudaReservation.blocks: Vec<(offset, len, handle)>` currently has each
  commit create its own handle; there is **no `handle → Vec<VA>` table**, and the
  KV arena is a *single* `cuMemAddressReserve` range with allocations carved as
  offsets (`vmm_allocator.rs:709-733`). Prefix sharing therefore needs a
  bookkeeping extension: allow the same pooled handle to appear at multiple
  offsets/VAs with a handle-level refcount so release is gated across sharers.
  The CUDA capability (#727) and the accounting *shape* (granule_refs, pooled
  authority charge) are already present; what's missing is the 1:N handle map.
  **This is an implementation task, explicitly out of scope for this round.**
- **Write-protection composes with #759.** `cuMemSetAccess` is called per mapped
  subrange (`virtual_memory.rs:1126-1143`), currently only with
  `CU_MEM_ACCESS_FLAGS_PROT_READWRITE`. Setting a shared prefix range to
  `PROT_READ` (so any errant write to another request's KV faults loudly instead
  of silently corrupting it, per #777 point 2) and the #759 uncommitted-fallback
  range to `PROT_NONE`/read-only dummy page are **independent per-subrange access
  descriptors on the same reservation** — they do not conflict; both merely need
  the production path to start emitting non-RW flags, which it does not yet do.
  No architectural collision between the #777 read-only shared prefix and the
  #759 dummy-page probe was found.

### 3.4 Head-major impossibility — confirmed, and quantified

Not just repeated from #777: under BNSH each head's prefix is a `P × head_dim ×
dtype` run separated by the full `capacity × head_dim × dtype` reservation
stride, and the granule holding a head's prefix *tail* also holds that same
head's **private future tokens** `[P..capacity]`, so it cannot be shared
read-only. Shareable whole granules per head-stripe is
`floor(P × head_dim × dtype / granule)`, requiring
`P ≥ granule / (head_dim × dtype)` = **8192 tokens (hd128)** / **16384 (hd64)**
to share even one granule. At the headline P=2048 head-major shares **zero**.
Seq-major packs all `kv_heads` into one stripe, dividing that threshold by
`kv_heads` (14B: 8192 → **1024**). So seq-major's sharing advantage is precisely
a factor-`kv_heads` reduction in the minimum shareable prefix length — there is
no head-major grouping that recovers it, because the private-tail-in-granule
problem is structural. **The impossibility claim holds.**

---

## 4. Full change inventory (so the cost is visible)

**(a) Every attention kernel's KV indexing** (all encode BNSH
`(… kv_heads + kv_head) × capacity × head_dim`, would need a seq-major base
offset `t × kv_heads × head_dim + kv_head × head_dim`):

- `gqa_decode.rs:161-205` — `kv_plane = (batch*kv_heads + kv_head)*capacity*head_size`
- `gqa_decode_fp16.rs:180-205` — same `kv_plane`
- `flash_attention.rs:85-98` — `kv_base = (b*kv_heads + kvh)*kv_capacity*dim`
- `varlen_attention.rs:152,225` — `(gk*kv_heads + kvh)*head_size`
- `packed_varlen_attention.rs` — varlen indexing (same family)
- `standard_attention.rs:203,212` — `((b*heads+h)*past_cap + t)*dim + d`
- `group_query_attention.rs:192,572` — present append `((b*heads+h)*present_capacity + target_s)*dim + d`
- `attention.rs:744` — `k_base + ((b*num_kv_heads+kv)*kv_capacity*d)`
- `compressed_sparse_attention.rs` — CSA KV gather (own indexing, later increment)
- `rotary_embedding.rs` / RoPE-on-append paths write into the same cache layout.

This is a large surface — hence **not** converted in this round. The append path
(`group_query_attention.rs` present write) and the decode read path
(`gqa_decode*.rs`) are the minimal pair for a first real seq-major kernel.

**(b) The byte-exact native-vs-ORT KV comparison** used for validation: **it does
not exist as a byte comparison.** The existing parity oracles already compare
**outputs**, not KV bytes:
- `native_decode/tests.rs:1723-1777` ("bit-identical … no KV corruption") compares
  decoded outputs via `to_bits()`.
- `native_decode/paged_gqa.rs:309-361` compares attention **outputs** via
  `a.to_bits() == b.to_bits()`.
So the concern "validation would have to compare outputs rather than KV bytes if
layouts diverge" is **already satisfied**: current tests are output-level and
survive a native-only layout change. Any *new* byte-level cross-backend KV assert
must not be added; assert on outputs.

**(c) Declared ONNX shapes of `present_key`/`present_value`.** On the native path
KV is **device-resident and consumed only by our own CUDA kernels**; it leaves
the device only through explicit host-mirror/seed utilities
(`native_decode/mod.rs:368-415` `host_present_kv()` / `seed_growable_kv()` /
`seed_device_kv()`, layout note `[1, num_kv_heads, seq, head_dim]`;
`cuda.rs:2202-2299` mirror/seed). No ORT/CPU/engine consumer reads native KV
bytes during decode. Therefore the **physical** layout may differ from the
**logical** declared `present_key`/`present_value` shape *iff only our kernels
touch it* — which holds on the native path. The declared ONNX shape can remain
BNSH (logical contract) while physical storage is seq-major, provided the
host-mirror/seed helpers transpose at those explicit, rare boundaries (prefill
seed and cross-tier spill), not per step. The `KvPageStore` / `backing_store.rs`
spill payload (`backing_store.rs:125-170`, flat per-layer rows keyed by
`num_tokens,num_layers,num_kv_heads,head_dim`) is layout-aware only by metadata —
it needs a layout tag, not a structural change.

---

## 5. Recommendation and smallest next increment

**GO / NO-GO: GO** for seq-major on the native backend, as a per-backend
capability. Both falsifiers that could have blocked it are cleared, and with the
granule lever dead (#776, 2 MiB min == recommended) seq-major is the only
*layout* lever that lowers the granule floor without a finer granule — it shrinks
the floor and crossover by exactly `kv_heads` (14B: 8×, crossover 8,192 → 1,024
tokens; 0.5b: 2×). It is a convergence point for three goals (lower committed
bytes, no CUDA-graph re-capture on growth, cross-request prefix sharing), not an
optional optimization. **Correction folded in:** seq-major does *not* remove the
floor — `layers × 2` granules remain because K and V stay separate per-layer
bindings — and the crossover is no longer model-size independent (it scales with
`kv_heads`), so the win is large on wide-GQA models and modest on narrow ones.

The two falsification risks are cleared:
1. ORT compatibility is preserved *by divergence*, not by a bridge — ORT stays
   BNSH, native goes seq-major, neither reads the other's KV bytes (§1, §4c). No
   unacceptable contract break; only a layout tag on the (already per-backend) KV
   ownership contract (#522/#726).
2. The strided-read cost is within ±2% and the head_dim<128 B danger case does
   not materialize for any target model (§2).

And the upside is large and concrete, now led by prefix sharing (#777):
- **Page-level prefix sharing (co-primary, §3):** removes **2.62 GiB** of
  duplicated KV and lifts concurrent capacity **≈2.7–3.0×** on qwen14b with a
  2K-token shared system prompt — a whole multiple, and it scales with
  concurrency. The mechanism (multi-map #727, charge-once granule-ref ledger
  #740/#745) is real; only a 1:N handle-bookkeeping extension is missing.
  Impossible under head-major at realistic prefix lengths (§3.4).
- Shrinks the granule floor and crossover by `kv_heads` (14B: 768→96 granules,
  crossover 8,192→1,024 tok; §intro) and keeps the token stride
  sequence-independent (no CUDA-graph re-capture on growth).
- **Cuts the lockstep growth-spike by `kv_heads` (§2.1, measured):** qwen14b
  boundary-crossing burst 49.5 ms → 4.5 ms; fully removable via commit-ahead
  and/or boundary staggering, both of which the fixed VA makes graph-safe.

Caveat surfaced by this round: page sharing has its own **granule floor** — it
saves nothing when the per-(layer,side) shared stripe is sub-granule
(P < ~1K tok on 14B, < 8K tok on 0.5b), so the prize is a large-model /
long-prefix win, not universal (§3.1). And the layout win itself is
`kv_heads`-proportional (§intro), so it is far more compelling on wide-GQA models
(qwen14b, 8 kv-heads) than narrow ones (qwen0.5b, 2 kv-heads).

**Relationship to the #759 fixed-stride + dummy-tail design.** Because the
granule lever is dead (#776), the #759 fixed-stride + dummy-page design is a
**long-context-only** optimization under head-major — it beats bucket growth only
*above* the 8K/16K crossover, where content finally fills the reserved granules.
Seq-major lowers that crossover by `kv_heads` (to 1,024 tok on 14B), making the
benefit reachable at far shorter contexts. The two coexist and are complementary:
the #759 dummy page remains valuable independent of layout for fault safety (a
read-only tail page faults loudly instead of reading stale KV) and for enabling
under-commitment; seq-major shrinks the floor while #759 hardens the tail.
Nothing here supersedes #759 — it reframes it as a long-context hardening layer
rather than the primary floor fix.

**Smallest next increment** (one reviewable PR, not a fleet-wide conversion):
1. Add a `KvLayout` tag to the native KV store / `KvPageStore` (default
   `HeadMajorBnsh`; introduce `SeqMajorBsnh`).
2. Convert exactly the **decode pair** — the `group_query_attention.rs` present
   *append* and the `gqa_decode_fp16.rs` decode *read* — to honor the tag, behind
   the seq-major layout. Leave prefill/flash/CSA/standard/varlen on head-major
   for now (they gate out or fall back).
3. Reuse the existing **output-level** parity oracle
   (`paged_gqa.rs` / `tests.rs`) to prove bit-identical decode outputs
   head-major vs seq-major on qwen2.5-0.5b.
4. Then bank the #777 prize: add the 1:N handle-map bookkeeping to the allocator
   (§3.3) and wire read-only (`PROT_READ`) shared-prefix multi-map into the #726
   store, charged once via the existing granule-ref ledger.
5. Only after that: extend the layout to flash/prefill and the remaining kernels.

A negative result was in scope; this is a **positive** one, but deliberately
staged: the numbers justify the first decode-pair increment, not a big-bang
kernel rewrite.

---

### Reproduction

```
# the bench is its own workspace; run it from inside its directory
cd bench-seqmajor
cargo run --release                    # prints the §2 strided-read table
cargo run --release --bin remap_burst  # prints the §2.1 remap-burst table
```
The §2 strided-read and §2.1 remap-burst numbers come from `bench-seqmajor/`
(GPU; `remap_burst` uses the real driver VMM path `cuMemCreate`/`cuMemMap`/
`cuMemSetAccess`). The §3.1/§3.2
prefix-sharing and capacity tables are closed-form from
`bytes_per_token = layers·2·kv_heads·head_dim·dtype` with a 2 MiB granule and
per-(layer,side) granule rounding (formulas inline in §3). The §3.3 mechanism
facts are cited to `crates/onnx-runtime-cuda-memory/src/{virtual_memory.rs,
vmm_allocator.rs}` line numbers and to #727/#740/#745.
CUDA PATH (this box): prepend
`$sp\onnxruntime\capi;$sp\nvidia\cu13\bin\x86_64;$sp\nvidia\cublas\bin;$sp\nvidia\cudnn\bin`
where `$sp = C:\Users\justinchu\AppData\Local\anaconda3\Lib\site-packages`.
The bench is a standalone crate (`bench-seqmajor/`, excluded from the workspace)
that only depends on `cudarc` — it does not perturb the main build.
