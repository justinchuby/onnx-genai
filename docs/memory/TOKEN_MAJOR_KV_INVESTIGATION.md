# Token-major-across-all-layers KV: viability probe & measurement

Status: **measurement probe with one open risk settled.** No production kernel
converted this round (microbenchmark + one isolating GPU floor test only, per
scope). References **#750** (multi-request batching), **#777** (page-level prefix
sharing), **#783** (KV layout as a stride descriptor); builds directly on the
merged **#772** head-major floor
(`crates/onnx-runtime-cuda-memory/tests/vmm_kv_contiguous_tail_gpu.rs`), **#776**
granularity probe (`vmm_granularity_gpu.rs`), and the landed **#782** seq-major
decode pair and the #778 seq-major investigation.

Device under test: **NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, Ada), 8 GiB**,
driver 591.55, CUDA 13 NVRTC compiled straight to cubin (the driver rejects the
CUDA-13 PTX ISA — same cubin fallback as `runtime.rs`).

## TL;DR — the recommendation

**Token-major is worth implementing.** The one risk that could have killed it —
TLB pressure from the ~192 KB per-token read stride — **does not materialize on
sm_89**: at a 6 GiB working set the 192 KB stride reads at **207.1 GB/s, ratio
1.000** against the ~2 KB BSNH per-layer stride. 2 MiB device pages cover the
stride; the token-major read is *free*. And the payoff is measured through the
real #740 pool: the whole model's live KV for a fresh sequence commits **one
2 MiB granule** token-major versus **1.5 GiB** head-major — a **768×** floor
reduction for identical live content.

And on memory *granularity* token-major is **equivalent-or-finer than vLLM
paging** (§6): vLLM's real per-sequence allocation quantum is `block_size × all
layers` = **~3 MiB** on qwen14b (16 tokens), versus token-major's **2 MiB**
granule (~10.7 tokens) — same order, and token-major keeps flat kernels (no page
walk) and returns physical memory on demand. Quantization is a *split* result
(§6, both verified): it auto-scales tokens-per-granule up (fp8 21.3, int4 42.7)
with no tuning, but token-major's per-sequence **byte** floor stays a fixed 2 MiB
at any dtype, so under aggressive quantization a token-denominated paged design
wastes *fewer* bytes than we do. The claim that survives intact is the important
one: **head-major makes quantization and residency antagonistic** (its crossover
doubles as bytes halve), **while token-major makes them independent** (no
crossover — the floor is already one granule per sequence).

The smallest next increment is **not** a kernel rewrite: it is a `KvLayout`
stride-descriptor (§4, siding with #783 over a third enum variant) plus binding
the per-layer KV views as sub-ranges of one reservation, validated on the
existing output-level parity oracle. Kernel conversion follows only after that.

---

## The floor progression this settles (qwen14b: 48 layers, 8 kv_heads, hd128, fp16)

| Layout | Floor unit | Floor | Measured here |
|---|---|---|---|
| BNSH head-major | `layers × 2 × kv_heads` = 768 granules | ~1.5 GiB | 1536 MiB (GPU test) |
| BSNH seq-major (landed #782) | `layers × 2` = 96 granules | ~192 MiB | — (#778) |
| **token-major across all layers** | **1 granule per sequence** | **~2 MiB** | **2 MiB (GPU test)** |

"Token-major" = one virtual reservation for the whole KV cache, all layers' K and
V interleaved by token, so the live prefix of every layer and every head forms a
single contiguous run. Windows fill densely from the start; only the last is
partial. At ~2 MiB/sequence (≈16 MiB at 8 concurrent sequences) the floor stops
being a design constraint, while the flat contiguous VA — and therefore kernels
that never walk a page table — is preserved.

---

## 1. The open risk — TLB pressure — QUANTIFIED (owner item 1)

Under token-major, reading one layer's K strides by the **full per-token KV size**
(~192 KB for qwen14b) instead of ~2 KB. Each head's run is still a
`head_dim × sizeof(dtype)` = 256-byte contiguous segment — a length #778 already
measured as fine — so the risk is **not coalescing, it is TLB pressure**: each
256-byte run lands on a different page.

Microbenchmark: `bench-tokenmajor/` (this branch). One warp per (plane, head)
streams its `L × head_dim` KV run once with a fused query-dot reduction (every
load live — faithful decode traffic and the warp-shuffle reduction of
`gqa_decode_fp16`). The buffer holds the **identical bytes** for every layout;
only the token stride changes, set by `planes_per_group` (G), with `P = layers×2`
planes:

- `G = 1` → stride `kv_heads×head_dim` (**2 KB**, BSNH per-layer)
- `G = P/4`, `P/1` → intermediate and **192 KB** (token-major across all layers)

Total bytes moved and the contiguous per-token run (`head_dim×dtype`) are
identical across G; only the page spread differs, so any bandwidth delta *is* TLB
reach, isolated. Runs interleaved across G, 30 reps after 6 warmups, min+median
reported to fight this box's extreme variance.

**Result (median GB/s, `ratio` = this stride ÷ the 2 KB stride at the same L):**

qwen14b (kv8, hd128, 48L; P=96 planes, 768 warps):

| L | working set | G=1 (2 KB) | G=4 (8 KB) | G=24 (48 KB) | G=96 (**192 KB**) | ratio @192 KB |
|---|---|---|---|---|---|---|
| 512   | 96 MiB   | 210.4 | 202.0 | 202.3 | 201.6 | 1.036 |
| 2048  | 384 MiB  | 203.2 | 203.8 | 203.6 | 208.3 | 0.998 |
| 8192  | 1.5 GiB  | 206.3 | 206.0 | 206.1 | 206.1 | 0.999 |
| 32768 | **6.0 GiB** | 207.1 | 207.4 | 206.9 | **207.1** | **1.000** |

qwen0.5b (kv2, hd64, 24L): flat to within ±2.5% across all G and L (e.g.
L=32768, 384 MiB working set, stride 384 B→ratio 1.000).

**Reading the numbers honestly.** At the realistic working sets — where TLB reach
would actually matter — the 192 KB token-major stride is **dead-flat, ratio 1.000
at 6 GiB**. The only >2% blips are at L=512 (96 MiB), where the kernel is
launch/occupancy-bound at 768 tiny warps and run-to-run variance dominates, not
TLB. The reads are DRAM-bandwidth-bound (~207 GB/s ≈ 80% of the card's ~256 GB/s
peak with a single-warp-per-head fused kernel), and that bound is **independent of
stride**. TLB pressure is not the bottleneck at any tested stride or working set.

TLB *hardware counters* (`ncu` `tlb__*` / `l2tex` metrics) would have corroborated
this directly, but **Nsight Compute is not installed on this box** (`ncu` not on
PATH, none under the CUDA site-packages). Per the owner's guidance that
deterministic counters lead over wall-clock, the deterministic signal here is the
**bandwidth-vs-stride curve**, which is a valid TLB proxy precisely because every
other variable (bytes moved, contiguous run, occupancy, buffer) is held constant
across G — a stride-only sweep. A future `ncu` pass on a workstation GPU can add
the raw counters, but it cannot change a ratio that is already 1.000.

## 2. Do 2 MiB device pages rescue it? — YES, decisively (owner item 2)

CUDA device memory is 2 MiB-page backed (the granule is 2 MiB — MINIMUM ==
RECOMMENDED, hardware-verified by #776 and re-confirmed by #778's `remap_burst`).
The question is whether the sm_89 TLB *reach* covers a 192 KB stride at
multi-GB working sets, or degrades.

**It covers it.** The G=96 (192 KB stride) column above holds 207 GB/s from a
96 MiB working set all the way to a **6 GiB** working set — 3,072 distinct 2 MiB
pages walked per head, one 256-byte touch each — with **zero** bandwidth loss
versus the 2 KB stride. If the L2 TLB were thrashing at this stride/footprint,
bandwidth would fall off as the working set grew past TLB reach; it does not. On
sm_89 the 2 MiB page backing makes the token-major stride **effectively free on
the read path** — the single most valuable thing this probe establishes, because
it removes the only measured objection to token-major.

Caveat, stated: this is measured on one Ada laptop part. The result is expected to
hold on larger Ada/Hopper parts (same 2 MiB granule, larger TLB and higher
bandwidth), but the *specific* 6 GiB-at-207 GB/s number is this device's. The
qualitative conclusion — stride-independent, DRAM-bound reads — is what transfers.

## 3. The floor claim — VERIFIED ON HARDWARE (owner item 3)

`crates/onnx-runtime-cuda-memory/tests/vmm_kv_token_major_floor_gpu.rs` extends
the merged #772 pattern, but through the **real #740 authority-scoped
physical-handle pool via `carve()` suballocation** (`CudaVmmAllocator`), not raw
driver calls — no second allocator, no per-sequence physical reservation — and
reports committed **physical** bytes (`committed_and_reserved().0`), never nominal
content bytes.

- **`token_major_reservation_commits_one_granule_per_sequence`**: one reservation
  for the whole KV (768 MiB VA for a 4,096-token context — free), commit the live
  prefix of a fresh sequence. **Result: 2 MiB committed (one granule)** for one
  token's whole-model KV (196,608 B across all 768 objects), and it *stays* one
  granule up to 10 tokens (`floor(2 MiB / 192 KiB)`), proving the density claim.
  On release, committed physical bytes return to 0.
- **`head_major_fixed_stride_commits_one_granule_per_object`**: the same live
  content laid out head-major with a fixed full-context stride commits **one
  granule per head-stripe**. Proven on a 16-object subset (32 MiB), then the whole
  model's closed form asserted: **768 objects × 2 MiB = 1536 MiB**.

> `head-major … 768 objects × 2 MiB granule = 1536 MiB committed for 196608 B of
> live content … Token-major commits 2 MiB for the identical content — a 768×
> reduction.` (test stdout)

The payoff claim is therefore **measured, not asserted**: 768× less committed
physical memory for a fresh sequence, one granule per sequence.

## 4. Implementation cost — stated honestly (owner item 4)

What must change to make token-major a production layout:

1. **`KvLayout` becomes a stride descriptor, not a third enum variant — side with
   #783.** Today `KvLayout::{HeadMajorBnsh, SeqMajorBsnh}`
   (`crates/onnx-genai-metadata/src/schema/model_io.rs`) is a two-value enum that
   maps to a `kv_layout` GQA attribute (`0`/`1`). A third `TokenMajorAllLayers`
   variant would work but is brittle: token-major is not "another BNSH/BSNH
   permutation", it collapses the per-(layer,side) buffer boundary that both
   existing variants keep. **Recommendation: promote `KvLayout` to a small stride
   descriptor** — `{ token_stride, plane_stride, head_stride }` in elements, plus
   a `single_reservation: bool` — of which head-major, seq-major, and token-major
   are three parameterizations. This is #783's direction, and the microbench here
   is already written against exactly such a descriptor (the `G`/plane-group
   arithmetic), so the kernel-side indexing generalizes with no new branch per
   layout — only different stride constants. A bare third enum variant would force
   a third hand-written index path into every kernel in the inventory below.
2. **KV bindings become sub-ranges of one reservation.** Today each layer's K and
   V are separate allocations. `crates/onnx-genai-engine/src/engine/model_io.rs`
   declares `key_cache_*` / `value_cache_*` as **per-layer positionally-paired
   vectors**, and downstream code (host-mirror/seed, spill, kernel bindings)
   indexes them per layer. **Per-layer bindings must survive as *views into* the
   single token-major reservation** — same `Vec<binding>` shape, but each entry is
   `base + layer_plane_offset` with the token-major stride, not an independent
   `cuMemAlloc`. This is the real structural work and it is where correctness risk
   lives (off-by-one in the plane offset corrupts a neighbour's KV).
3. **Every attention kernel's KV indexing changes** — the same surface #778
   enumerated for seq-major (`gqa_decode*.rs`, `flash_attention.rs`,
   `varlen_attention.rs`, `standard_attention.rs`, `group_query_attention.rs`
   append, `attention.rs`, CSA, RoPE-on-append). Token-major only changes the
   base-offset arithmetic (add the `token_stride`/`plane_stride` from the
   descriptor); the intra-token `head_dim` run is unchanged, which is exactly why
   the read stayed coalesced and TLB-flat in §1. As with #782, convert the
   **decode pair** (append + `gqa_decode_fp16` read) first, gate the rest.
4. **Physical backing stays inside the #740 contract.** Granules come from the
   authority-scoped pool via `carve()` with 0→1 / 1→0 granule-ref attribution (the
   floor test already exercises this path). No second allocator; no per-sequence
   physical reservation — that is what made #733 net-negative. Growth pre-commits
   ahead of the write frontier on the *same* fixed VA (never map/grow inside a
   captured region; never unmap while a replay may be in flight, per #727/#727).

Honest cost summary: token-major is **strictly more invasive than seq-major on the
binding layer** (it dissolves the per-layer allocation boundary) but **no more
invasive on the kernel layer** (same base-offset edit), and if `KvLayout` is
first refactored into a stride descriptor (#783) the kernel edit is *shared* with
seq-major rather than duplicated.

## 5. Interaction with #777 prefix sharing (owner item 5)

Under token-major a shared token prefix `[0..P]` is **one contiguous byte range
covering all layers and both sides at once**, so cross-request sharing is a single
multi-map of that range, versus `layers × 2` separate multi-maps under seq-major
(and `layers × 2 × kv_heads` scattered fragments under head-major). The #727
multi-map primitive and the #740/#745 charge-once granule-ref ledger already
support N-way mapping of a pooled handle; token-major just makes the shared unit
*one range* instead of many.

**Saving for a 2,000-token shared system prompt, 8 concurrent requests, qwen14b**
(196,608 B/token, the number in the prompt):

- Duplicated KV removed by sharing = `(R−1) × P × bytes_per_token`
  = `7 × 2000 × 196,608` = **2.563 GiB** of committed physical memory not
  re-committed for requests 2–8.
- Granule-rounding: token-major shares whole 2 MiB granules of *one* contiguous
  range, so the shared prefix rounds at `ceil(2000 × 196,608 / 2 MiB) = 188`
  granules = 376 MiB shared once; the rounding waste is at most one granule for
  the *whole* shared prefix (versus one per (layer,side) stripe under seq-major).
  So token-major realizes essentially the full **≈2.56 GiB** saving, whereas
  seq-major's per-stripe granule floor can erase it entirely when the per-stripe
  prefix is sub-granule (#778 §3.1: 14B needs P ≥ 1024 tokens *per stripe* to bank
  a granule; token-major needs P ≥ ~11 tokens for the *whole model*).

So token-major makes #777 both **cleaner** (one multi-map) and **more robust to
short prefixes** (the sharing granule floor is per-sequence, not per-stripe). This
is an independent second reason to prefer it, orthogonal to the read-path result.

## 6. Comparison with vLLM paging — token-major is equivalent-or-finer on granularity

A correction folded in (and independently re-verified below): vLLM's block is
often described as "small and dense (~32 KB)", but that is the **per-layer**
block. The unit that actually costs memory is the **per-sequence allocation
quantum**: growing a sequence by one 16-token block requires giving *every layer*
a block, so the quantum is `block_size × kv_bytes_per_token`. For qwen14b that is
**~3 MiB**, the same order as our 2 MiB granule — not orders smaller. Token-major
+ VMM is therefore not merely "approaching paged behaviour"; on memory
granularity it is **equivalent or slightly finer**, while keeping flat kernels and
on-demand physical release.

Verified arithmetic (`kv_bytes_per_token = layers × 2 × kv_heads × head_dim ×
dtype`; granule = 2 MiB):

| model | per-layer K+V/tok | kv_bytes_per_token (all layers) | **token-major tokens/granule** = `granule / kv_bytes_per_token` | **vLLM tokens/quantum** = `block_size` | vLLM per-seq quantum = `block_size × kv_bytes_per_token` |
|---|---|---|---|---|---|
| qwen14b (48L, kv8, hd128, fp16)   | 4,096 B | 196,608 B (192 KiB) | **10.67** | 16 | **3.00 MiB** |
| qwen2.5-0.5b (24L, kv2, hd64, fp16) | 512 B | 12,288 B (12 KiB)  | **170.67** | 16 | **0.188 MiB** |

General formulas (reproducible for any model):
- **token-major:** `tokens_per_granule = granule / kv_bytes_per_token`.
- **vLLM-equivalent:** `tokens_per_quantum = block_size` (default 16), at a
  per-sequence physical quantum of `block_size × kv_bytes_per_token`.

| | Per-sequence quantum | Kernel walks a page table? | Physical memory returnable? |
|---|---|---|---|
| vLLM paging | `block_size × kv_bytes_per_token` (**~3 MiB** qwen14b) | **yes** | no — pool pre-committed |
| **token-major + VMM** | **1 granule = 2 MiB** (~**10.7** tokens qwen14b) | **no** | **yes** |

### Internal fragmentation per sequence — same units, both designs
Stated so the two are directly comparable (bytes wasted per sequence, worst case):

- **token-major:** at most **one partial granule** = **≤ 2 MiB**,
  *model- and dtype-independent* (the last granule holds the write frontier;
  everything before it is dense). The 2 MiB bound is **hardware-imposed** — it is
  the CUDA device granule, measured by #776 as `MINIMUM == RECOMMENDED == 2 MiB`,
  with no finer granule available on this device.
- **paged:** at most **one partial block** = `≤ block_size × kv_bytes_per_token`.
  The `block_size` (16 tokens) is a **design choice**, so this bound is
  token-denominated and *scales with the model and the dtype* — 3 MiB on qwen14b
  fp16, 0.19 MiB on qwen0.5b fp16.

**Honest reading, both directions.** On qwen14b fp16 token-major is *slightly
finer* (2 MiB vs 3 MiB bound; 10.7 vs 16 tokens/quantum). But the bound cuts the
other way on small models: on qwen0.5b the 2 MiB granule holds 170 tokens, so
token-major's ≤ 2 MiB waste is *coarser* than paged's ≤ 0.19 MiB. The absolute
waste is one 2 MiB granule per sequence regardless (≈16 MiB across 8 concurrent
sequences), so this is a nuance, not a threat.

### Quantized KV — corrected, two separate claims
My earlier "quantization is a uniform win for token-major" was half right. Split
honestly (both tables verified on both models at fp16/fp8/int4):

**Claim 1 (true, and good): tokens-per-granule auto-scales with quantization,
with no tuning,** because our quantum is *byte-denominated* (the hardware 2 MiB
granule):

| qwen14b | kv_bytes_per_token | token-major tokens/granule |
|---|---|---|
| fp16 | 196,608 B | **10.7** |
| fp8  |  98,304 B | **21.3** |
| int4 |  49,152 B | **42.7** |

(qwen0.5b: 170.7 → 341.3 → 682.7 for fp16/fp8/int4.)

**Claim 2 (also true, and it cuts against us): the per-sequence byte floor does
NOT shrink with quantization — it stays one granule, 2 MiB, always.** A paged
design's quantum is *token-denominated* (`block_size` tokens), so its byte waste
*does* shrink as the model is quantized. Worst-case wasted bytes per sequence:

| | Quantum type | fp16 | fp8 | int4 |
|---|---|---|---|---|
| Paged (token-denominated, block=16) | scales | ~3 MiB | ~1.5 MiB | ~0.75 MiB |
| **token-major + VMM** (byte-denominated, 2 MiB granule) | fixed | **2 MiB** | **2 MiB** | **2 MiB** |

So at fp16 token-major wastes less; **under aggressive KV quantization a
token-denominated (paged) design wastes fewer bytes per sequence than we do.**
Stated plainly rather than hidden — though the absolute numbers are small (≤ one
partial granule, ~16 MiB at 8 concurrent), so it is a nuance, not a threat.

**The claim that survives intact — and is the one worth emphasising:** under
**head-major**, KV quantization and memory efficiency *fight each other*. The
head-major granule crossover `granule / (head_dim × sizeof(dtype))` *doubles* in
tokens as bytes halve (qwen14b hd128: **8,192 → 16,384 → 32,768** for
fp16/fp8/int4), so a quantized model needs an even *longer* context before a fixed
stride stops losing to bucket growth. Under **token-major there is no crossover at
all**, because the floor is already one granule per sequence at any dtype. The
correct statement is therefore: **head-major makes quantization and residency
antagonistic; token-major makes them independent** (and, per Claim 1, throws in
free extra tokens-per-granule as a bonus).

### Hardware-imposed vs design choice — so the comparison is fair
- **Hardware-imposed:** the **2 MiB granule** (#776: `MINIMUM == RECOMMENDED`, no
  finer granule on this device) — this fixes token-major's per-sequence byte
  floor and is not tunable without a driver patch we do not take (the vAttention
  64 KiB UVM patch, rejected in `vmm_allocator.rs` rationale).
- **Design choices:** vLLM's **`block_size`** (16 tokens) — freely tunable, which
  is exactly why paged waste is token-denominated and shrinks under quantization;
  and token-major's decision to keep **one flat contiguous VA** (so kernels never
  walk a page table), which is what forces the byte-denominated 2 MiB quantum in
  the first place. The trade is explicit: token-major spends a fixed ≤ 2 MiB/seq
  of internal fragmentation to buy page-table-free kernels and on-demand release.

### Historical note — layout is the whole story
The "≈10.7 tokens per granule" figure was computed early in this work, applied to
**head-major**, found wrong, and publicly retracted — head-major scatters one
token's live bytes across 96 stripes (768 objects), each landing in its own
granule, so a single token costs 768 granules and the tokens-per-granule
arithmetic does not hold. Under **token-major** that same arithmetic is *exactly*
correct, because the token's bytes are one contiguous 192 KiB run. Same granule,
same model, same VMM — **only the layout differs.** That is the entire thesis of
this line of work, restated as a single number that is false one way and true the
other. (Verified: `python` check of both models at fp16/fp8/int4, reproduced in
the commit.)

---

## Recommendation & smallest next increment

**GO.** The TLB risk is cleared (ratio 1.000 at 6 GiB, §1–2), the floor payoff is
measured (768×, §3), and #777 sharing is strictly better under this layout (§5).
The read path is free; the cost is on the binding layer (§4).

Smallest reviewable increment, in order:
1. Refactor `KvLayout` into a **stride descriptor** (#783) — head/seq/token-major
   as three parameterizations, no new per-layout kernel branch.
2. Make the native KV store hand out **per-layer views into one reservation** (the
   `model_io.rs` positional vectors become offset views), still head-major by
   default. Prove no output change on the existing **output-level** parity oracle
   (`native_decode/paged_gqa.rs`, `tests.rs`) — do not add a byte-level KV assert.
3. Only then flip the decode pair (`group_query_attention` append +
   `gqa_decode_fp16` read) to the token-major stride behind the descriptor, reuse
   the same oracle.
4. Bank #777: multi-map the single shared prefix range read-only (`PROT_READ`),
   charged once via the granule-ref ledger.

A truthful negative was in scope; this is a **positive** result, deliberately
staged: the numbers justify the descriptor + binding-view increment, not a
big-bang kernel rewrite.

### Reproduction

```
# TLB stride sweep (§1–2); its own workspace, run from inside it with CUDA on PATH
cd bench-tokenmajor
cargo run --release            # prints the stride×L bandwidth table

# floor proof (§3), through the real #740 pool
cargo test -p onnx-runtime-cuda-memory --features gpu-tests \
  --test vmm_kv_token_major_floor_gpu -- --nocapture
```
The §5 prefix-sharing numbers are closed-form from
`bytes_per_token = layers·2·kv_heads·head_dim·dtype` = 196,608 B with a 2 MiB
granule. Nsight Compute was unavailable, so §1's TLB conclusion rests on the
stride-only bandwidth sweep (deterministic; every other variable held constant),
not on raw `tlb__*` counters.
