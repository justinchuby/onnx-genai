# Seq-major (BSNH) KV layout: investigation & measurement

Status: investigation + measurement + narrow prototype. **No production kernel
conversion in this round.** References #750 (multi-request batching), #759 (VMM
dummy-page / unified KV strategy), and **#777 (page-level prefix sharing)**;
builds on #522 and #726 (per-backend KV ownership), and on #727 / #740 / #745
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

Seq-major (BSNH `[batch, seq, kv_heads, head_dim]`) removes that floor: the live
prefix of all heads is one contiguous region, commit is
`ceil(live_bytes / granule)`, and the token stride (`kv_heads × head_dim`) is
independent of sequence length, so growth never moves an offset and never
invalidates a captured CUDA graph.

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
| **head_dim=32 danger** (8, 32, 40) | 64 B | 0.991 | 0.965 | 0.994 | 0.996 |
| qwen0.5b (2, 64, 24) | 128 B | 0.995 | 1.003 | 0.999 | 0.998 |
| kv8 (8, 64, 40) | 128 B | 0.810 | 0.980 | 0.988 | 0.993 |
| qwen14b (8, 128, 48) | 256 B | 1.003 | 1.019 | 1.008 | 1.005 |

Seq-major is within ±2% of head-major everywhere, and *faster* in the kv8/hd64
cases. **The predicted `head_dim × dtype < 128 B` cache-line-halving does not
occur.** Reason: heads are contiguous in the seq-major layout, so a 128 B cache
line that holds head *h*'s 64 B token run also holds head *h+1*'s 64 B run, and
all heads of a layer are read by concurrent warps — L2 recovers the neighbor
half. The contiguous run being smaller than a cache line is harmless as long as
the adjacent head is co-resident and co-read, which it always is.

**Danger-case reach:** the flagged case needs `head_dim × sizeof(dtype) < 128 B`,
i.e. head_dim < 32 at fp16 (head_dim 32 = exactly 64 B, already tested clean).
No model we target has head_dim < 64: Qwen2.5 uses 64/128, Llama/Mistral 128.
head_dim 32 fp16 is already covered above with zero penalty. **Not a real risk.**

Caveat: absolute GB/s here (10–225 GB/s) is below peak DRAM because the
single-warp-per-head fused kernel is latency/occupancy-bound at these block
counts — but *both layouts are equally bound*, so the relative delta (what we
care about) is valid and near-zero. A production seq-major kernel would use the
same split-K / multi-warp strategy as today's head-major kernel; the strided LD
only changes the base-offset arithmetic, not the coalescing within a token row.

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

**Recommendation: pursue seq-major on the native backend, as a per-backend
capability.** The two falsification risks are cleared:
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
- Eliminates the model-size-independent granule floor and keeps the token stride
  sequence-independent (no CUDA-graph re-capture on growth).

Caveat surfaced by this round: page sharing has its own **granule floor** — it
saves nothing when the per-(layer,side) shared stripe is sub-granule
(P < ~1K tok on 14B, < 8K tok on 0.5b), so the prize is a large-model /
long-prefix win, not universal (§3.1).

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
# from this worktree, with the CUDA PATH prepended (see below)
cargo run --release -p bench-seqmajor   # prints the §2 strided-read table
```
The §2 strided-read numbers come from `bench-seqmajor/` (GPU). The §3.1/§3.2
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
