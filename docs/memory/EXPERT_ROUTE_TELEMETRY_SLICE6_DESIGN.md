# Slice 6 — device-side expert-route telemetry for QMoE/BlockQuantizedMoE residency

Issue #1810 (composable sub-weight VMM: mixed device/host-NUMA expert granules
under one stable VA). Author: Deckard (Systems/Perf). Status: **design +
inert proof harness only**. No production residency/lifecycle wiring. This
slice is independent of PR #1854 (Slice 5 coarse-boundary plan application) and
touches none of its files.

> **What this slice answers.** Slices 1–5 built the *mechanism* to make some
> experts device-resident and others host-backed under one stable VA, and to
> apply a per-expert residency plan at a coarse safe boundary. They did **not**
> answer *which* experts a running request actually routes to — the input a
> residency **policy** needs. The host cannot see that today (§1). This slice
> specifies the smallest GPU-resident telemetry that observes the real routed
> set on-device, with zero steady-state host sync and full CUDA-graph safety,
> adapting FreeToken's device-side route-observation idea (§5) **without**
> importing its slot-cache/second-allocator authority (§4).

---

## 1. Where expert IDs exist on device today, and why the host is blind

### 1.1 The routed set is computed inside the fused op, on device, from runtime activations

`com.microsoft::QMoE` fuses routing **and** expert GEMV into one dispatch. The
route kernel is the *only* place the routed set is materialised:

- `crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs:115` — `qmoe_route(const float* router_probs, …, int* selected_experts, float* selected_weights, rows, experts, top_k, normalize)`.
  One CUDA block cooperatively top-k-selects one row (grid-strided over rows);
  it writes `selected_experts + row*top_k` (`qmoe.rs:133`), i.e. an
  **`int32[rows, top_k]`** buffer, flattened to `routes = rows * top_k`.
- The consumers read that buffer **on device**:
  `qmoe.rs:378` (`const int expert = selected_experts[route];` in the grouped/
  ungrouped linear kernels) and `qmoe.rs:614` (per-route GEMV). The contiguous
  expert bank is then indexed `packed[expert*out_features*packed_in + …]`
  (`qmoe.rs` linear kernels; see §1.3).
- `BlockQuantizedMoE` mirrors this exactly:
  `crates/onnx-runtime-ep-cuda/src/kernels/block_quantized_moe.rs:79`
  (`bqmoe_route(… int* selected_experts …)`, `int32[rows, top_k]`) consumed at
  `block_quantized_moe.rs:183` (`const int expert = selected_experts[route];`)
  and in `bqmoe_combine`.

`router_probs` themselves are produced **on device** by the preceding gate
`MatMul`/`MatMulNBits` from the *current* hidden state. The routed set is a
data-dependent function of a runtime activation that never leaves the GPU. There
is no point in the host program where the exact routes for a token are known
before the kernel that consumes them launches.

### 1.2 Buffer shape and lifetime (prefill vs decode)

`route_indices` is scratch slot 0 of `QMoEKernel`'s persistent `ScratchPool`:

- Allocated at `qmoe.rs:1723`, `route_index_bytes = routes * sizeof(i32)` where
  `routes = rows * k` (`qmoe.rs:1671`).
- **Prefill:** `rows = prompt length`, so `routes = prompt_len * top_k`.
- **Decode:** `rows = 1`, so `routes = top_k` (6 for DeepSeek-V4-Flash, 8 for
  GLM-5.2, 2 for the tiny DeepSeek-V4 fixture).
- The slot is **persistent and size-classed** with a **stable device pointer**
  reused across calls (`ScratchPool::ensure`, `qmoe.rs:2793`). It is overwritten
  every call; nothing accumulates.
- **Capture safety is already load-bearing here:** under capture,
  `ScratchPool::ensure` **refuses to grow** and returns `Err` if the warmed
  capacity is too small (`qmoe.rs:2810-2814`). Warm-up fixes every scratch
  pointer before capture; capture records launches against those fixed pointers;
  replay re-runs them. This is exactly the contract a telemetry buffer must obey.

### 1.3 The contiguous expert-bank pointer ABI (must be preserved)

The executor hands each expert-weight tensor to the kernel as **one contiguous
base pointer**, and the kernel derives every expert's rows by arithmetic on it
(`packed[expert*out_features*packed_in + …]`). This is documented in
`docs/benchmarks/2026-08-18-moe-per-expert-dispatch-seam-design.md` (§Q3) and
re-confirmed by the zero-copy spike's contract note
(`crates/onnx-runtime-ep-cuda/tests/qmoe_zero_copy_cold_expert_spike_gpu.rs`
header: "one whole `LazyWeight` → one contiguous device pointer"). Slices 1–5
keep the **virtual** address contiguous and change only which granules are
device- vs host-backed. **Any telemetry design that rewrites expert ids to slot
ids (as FreeToken does, §5) breaks this ABI and is rejected.**

### 1.4 The only existing host-visible route readback is a debug sync (what we must not do)

`QMoEKernel::dump_route_selection` (`qmoe.rs:2035`) copies `selected_experts` and
`router_probs` to the host with blocking `dtoh` (`qmoe.rs:2053`, `qmoe.rs:2064`).
It is:

- **off by default** (`ONNX_GENAI_QMOE_ROUTE_DUMP`, `qmoe.rs:1874`),
- **guarded `!capturing`** — a `dtoh` during capture is illegal, and
- a **full host sync** on the decode critical path.

The steady-state decode path issues no route readback at all
(`qmoe.rs:1999-2004`: "no host-visible read of their output happens in this
function"), and under CUDA graphs the host only issues `cuGraphLaunch`. **A
residency policy therefore has no route signal today without adding either a
per-step sync (kills capture and decode latency) or a device-side observer
(this slice).**

---

## 2. The telemetry contract (smallest fixed-capacity GPU-resident observer)

One **fixed-capacity, GPU-resident, per-(MoE-layer)** telemetry record, allocated
once at warm-up from static config (`num_moe_layers`, `num_experts`, `top_k`) and
held for the session at a **stable VA** (same discipline as the scratch slot,
§1.2). Two representations are specified; the harness (§7) measures both and the
cost model (§6) picks between them per model.

### 2.1 Representation A — route bitmap (recommended default)

```
per layer: u32 word[ ceil(num_experts / 32) ]   // bit e == expert e routed this window
```

- Written with `atomicOr(word[e>>5], 1u << (e&31))` for each `e = selected_experts[route]`.
- **Cannot overflow**: capacity is `num_experts` bits by construction, so a
  correct route can never exceed it. The only failure is an *out-of-range* id
  (`e < 0 || e >= num_experts`), which sets a dedicated **poison bit** in the
  header and fails the record closed (§3).
- Loses per-expert counts (presence only). That is sufficient for a *residency*
  policy: which experts to keep mapped is a set question, not a histogram
  question. (A frequency variant, `u32 count[num_experts]` via `atomicAdd`, is a
  drop-in if the policy later wants recency-weighting; it is `num_experts*4`
  B/layer and still bounded.)

### 2.2 Representation B — bounded deduplicated route/miss queue

```
per layer: header + i32 slot[capacity]          // distinct routed (or missing) expert ids
           + u32 seen[ ceil(num_experts/32) ]   // dedup filter (scratch)
```

- For each `e = selected_experts[route]`: if `atomicOr(seen…)` shows `e` unseen,
  `pos = atomicAdd(&header.count, 1)`; if `pos < capacity` write `slot[pos]=e`,
  else set `header.overflow = 1` and stop appending.
- Captures order/recency and (with a `miss` predicate against the current
  resident set) exactly the *misses* a policy would act on — the FreeToken
  `src_indices`/`num_indices` shape (§5), but as **observation only**, never
  driving a copy.
- **Overflow is the fail-closed signal**: if distinct routed experts exceed
  `capacity`, `overflow=1` and the consumer treats the record as untrustworthy
  (§3). `capacity` is a static GO/NO-GO knob (§6).

### 2.3 Header, identity, epoch (both representations)

```
struct TelemetryHeader {         // 32 B, device-resident, one per (layer, buffer)
    u64 epoch;        // generation stamp; bumped by the producer at each safe boundary
    u64 request_id;   // owning request (sequence) identity
    u32 device_id;    // owning CUDA device ordinal
    u32 overflow;      // queue overflow (rep. B) — sticky
    u32 poison;        // out-of-range id / identity conflict — sticky, fail-closed
    u32 count;         // distinct count (rep. B); unused (0) for rep. A
}
```

- **Zero steady-state host sync.** The producer writes only via atomics on the
  EP stream; there is **no `dtoh` and no `synchronize` per step**. This is the
  FreeToken `lru_stats` property (`offload_cache.py:193-203`,
  `record_decode_stats` is a no-op `offload_cache.py:819`): the accumulation is
  *inside the launch* and is captured into the decode graph.
- **Graph capture/replay safe.** Fixed shape, stable VA, atomic accumulation —
  the `atomicOr`/`atomicAdd` is recorded into the graph and re-executes on each
  replay against that replay's **real** routing. The consumer reads the record
  once, at a boundary, not per replay.
- **Contiguous expert-bank ABI preserved.** The telemetry buffer is a **separate
  side buffer**. The route kernel still writes `selected_experts` and the GEMV
  still indexes `packed[expert*out+…]` unchanged (§1.3). Telemetry is an
  additional side-write, not an id remap.

Cost of the record itself (per MoE layer): rep. A = `ceil(E/32)*4` B; rep. B =
`capacity*4 + 32 + ceil(E/32)*4` B. At GLM-5.2/DeepSeek-V4 `E=256`, rep. A is
**32 B/layer** (§6).

---

## 3. State machine (coarse safe boundary)

```
                 ┌─────────────────────────────────────────────────────────────┐
   warm-up  ──▶  │ ALLOCATE fixed-capacity records, stable VA, zeroed,          │
                 │ epoch:=0, request_id/device_id stamped for the owning stream  │
                 └─────────────────────────────────────────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
   PRODUCE ──▶│ during graph/eager launch, on the in-order EP stream:│
              │  route/telemetry kernel atomically OR/append routes. │
              │  NO host sync. Capture-safe (fixed shape, stable VA).│
              └─────────────────────────┬──────────────────────────┘
                                        │  (decode step completes)
              ┌─────────────────────────▼──────────────────────────┐
   CONSUME ──▶│ ONLY after the existing stream-completion authority  │
              │ (the sync/event that already gates output/KV read).  │
              │ Read records once (single dtoh of a few KB).         │
              └─────────────────────────┬──────────────────────────┘
                                        │
              ┌─────────────────────────▼──────────────────────────┐
   VALIDATE ─▶│ epoch==expected ∧ request_id==self ∧ device_id==self │
              │ ∧ overflow==0 ∧ poison==0                             │
              └───────────┬─────────────────────────┬───────────────┘
                     pass │                     fail │
              ┌───────────▼──────────┐   ┌───────────▼───────────────────────────┐
              │ hand desired-set to  │   │ FAIL CLOSED → whole-bank proof:        │
              │ PMM/VMM policy (§4)  │   │ desired-set := ALL experts resident    │
              │ at coarse boundary   │   │ (pre-#1810 behavior). Never partial-    │
              └──────────────────────┘   │ trust a stale/overflowed/foreign buffer.│
                                         └────────────────────────────────────────┘
```

Fail-closed triggers (all → whole-bank proof, never a partial residency change):

- **Overflow** (rep. B queue full), **poison** (out-of-range id),
- **Stale epoch** (record older than the boundary being decided),
- **Multi-request** contention (`request_id != self`): two sequences sharing a
  record — reject; each request owns its own record or the shared record is
  poisoned on identity mismatch,
- **Multi-device** (`device_id != self`): a record produced on another device.

**Explicitly forbidden anywhere in this machine:** `cuMemMap` / `cuMemSetAccess`
during capture or replay. The composable-VMM spike proved the driver returns
`CUDA_SUCCESS` for a remap issued mid-capture rather than refusing it
(`.squad/decisions/inbox/deckard-1810-composable-vmm-spike-results.md`, "Critical
finding"). Every mapping change is owned by PMM/VMM and routed through
`capture_gate::synchronizing_section()` at a coarse safe boundary — telemetry
**observes**; it never maps.

---

## 4. One-authority rule

- **Policy (the telemetry consumer)** decides *desired residency*: a per-expert
  hot/cold desired-set derived from the validated record. It computes intent
  only. It owns no memory, no mapping, no eviction.
- **PMM/VMM** (`PhysicalHandlePool`, `CudaVirtualBacking`, and the Slice 4/5
  coarse-boundary plan application) **exclusively** owns physical mapping,
  byte accounting, quarantine, and rollback. It already fails closed and rolls
  back (deckard-1810 spike: fault-injection leaves 0 residual granules, 0
  underflow).
- **No second LRU / cache / allocator authority is introduced.** There is one
  question ("which experts should be mapped") and one mechanism that answers the
  mapping side of it (PMM/VMM). This is the `design-discipline` rule: two
  mechanisms answering the same question is duplicated state, decided by whichever
  runs first. Telemetry adds an *input* to the existing authority, not a rival to
  it.

This is the sharpest line between this design and FreeToken (§5), which **does**
run a second authority: a slot cache with its own admission/eviction and its own
`copy_missing` DMA mover.

---

## 5. FreeToken comparison — copied concepts vs rejected assumptions

Authoritative source: `github.com/FlashML-org/FreeToken`,
`python/freetoken/moe/offload_cache.py` (the `OffloadMoeCache` device-side expert
cache) and its `flashlib.kernels.slot_cache` / `freetoken.moe.offload_kernels`
`ensure_experts` kernel. Paper: arXiv:2608.16157.

### 5.1 What FreeToken does (exact mechanism)

- A **fixed-size GPU slot cache** of `cache_size` expert slots
  (`offload_cache.py:88+`), with a forward map `slot_for_id[num_layers,
  num_experts]` and reverse map `id_of_slot[cache_size]` over a **flat id space**
  `id = layer_id*num_experts + expert` (`offload_cache.py:150-165`), LRU `usage`
  and a device `step` clock.
- `ensure_experts(layer_id, expert_ids)` (`offload_cache.py:761`) runs a
  device kernel that, in **one launch**: computes misses vs the slot cache
  (`active_mask`, `num_indices`, `src_indices`, `evict_slots`), performs LRU
  admission/eviction, **rewrites `expert_ids` in place from expert ids to slot
  ids**, and accumulates telemetry into `lru_stats[num_layers, N_STATS]`.
- **Device-side, graph-safe telemetry with no per-step host sync**:
  `lru_stats` is accumulated inside `ensure_experts`' own launch; the previous
  eight-torch-op-per-layer version was removed and `record_decode_stats` is now a
  **no-op** (`offload_cache.py:819`). It is captured into the decode graph and
  "re-executes with each replay's REAL routing" (`offload_cache.py:193-203`).
  Read once via `decode_miss_stats` / `decode_miss_stats_per_layer`
  (`offload_cache.py:843`, `:866`) with "no per-step host sync".
- Actual weight movement is `copy_missing` (`offload_cache.py:926`), a fused
  `cudaMemcpyBatchAsync` H2D that fills the missed slots.
- Elastic VRAM via `rebuild(cache_size)` (`offload_cache.py:400+`): resizes the
  slot cache in place (`torch.cuda.synchronize` + `empty_cache`, slots cold-start).

### 5.2 Copied concepts (adapted faithfully)

| FreeToken concept | Exact source | How Slice 6 adapts it |
|---|---|---|
| Device-side route observation, **zero per-step host sync**, accumulated *inside the launch* | `lru_stats` `offload_cache.py:193-203`; `record_decode_stats` no-op `:819` | §2.3 header + atomic OR/append written on the EP stream; consumed once at a boundary |
| **Graph-safe** accumulation that re-runs on replay with the real routing | `offload_cache.py:196-199` | §2.3 / §3 PRODUCE; harness `capture_replay` test (§7) |
| **Flat id space** `layer*num_experts + expert` | `offload_cache.py:150-158` | §2 per-(layer) records keyed the same way |
| **Fixed-capacity** buffer sized once, never grown live | `cache_size`, `validate_rebuild` `:377` | §2 fixed capacity from static config; capture forbids growth (§1.2) |
| Device **epoch/step** clock to reason about freshness | `self.step` `offload_cache.py:163` | §2.3 `epoch`; §3 VALIDATE |
| **Bounded** overflow discipline | `num_indices`/`cache_size` cap | §2.2 `overflow` bit + §3 fail-closed |

### 5.3 Rejected assumptions (and why they do not fit onnx-genai)

| FreeToken assumption | Why rejected here |
|---|---|
| **Fixed slot table + id→slot rewrite** (`slot_for_id`/`id_of_slot`, `ensure_experts` rewriting `expert_ids` in place) | Breaks the contiguous expert-bank pointer ABI (§1.3): our kernels index `packed[expert*out+…]` off one base; a slot indirection is a real ABI change. It is also a **second residency authority** (owns admission/eviction), violating §4. We keep the stable VA and change only granule backing via PMM/VMM. |
| **`copy_missing` H2D mover as the cache-fill authority** | A second byte-mover with its own accounting. PMM/VMM already owns mapping/accounting/rollback (§4); a rival mover is exactly the duplicated-state anti-pattern. |
| **Per-token / per-step remap** (slots churn every `ensure_experts`) | The measured intra-layer prediction window on this codebase is ≈0 (`docs/benchmarks/2026-08-18-moe-per-expert-dispatch-seam-design.md` §Q2, citing `-moe-prediction-window.md`), so demand paging cannot hide latency. Residency changes only at **coarse safe boundaries** (Slices 4/5), and remap-during-capture is forbidden (§3). |
| **`rebuild` as an elastic second allocator** | Elastic device/host granule composition is already PMM/VMM's job under one stable VA (Slice 1–5). |
| **FTW fast-weight format / O_DIRECT host loading** | onnx-genai has its own loader + VMM granules; the host weight format is orthogonal to route observation. |
| **`decode_freq` per-expert host scatter** (`offload_cache.py:764-767`) | FreeToken itself flags it "only accurate with CUDA graphs disabled" — the host `scatter_add_` is not replayed. We take the graph-safe device path (`lru_stats` analog) and never the host-scatter path. |
| **CPU/hybrid `q★` co-execution** (`decode_target`, `hybrid_max_fetch`) | A larger execution-placement decision; out of scope for a route *observer*. |

---

## 6. Quantitative cost model and falsification (GO/NO-GO) gates

### 6.1 Authoritative shapes (cited configs, no invented numbers)

| Model | experts `E` | top_k | hidden | moe_inter | source |
|---|---|---|---|---|---|
| tiny DeepSeek-V4 QMoE (repo fixture) | **4** | **2** | (tiny) | (tiny) | `tests/fixtures/tiny-deepseek-v4-qmoe/model.onnx.textproto` (`fc1_experts_weights dims[0]=4`, QMoE `k=2`, `block_size=16`, `expert_weight_bits=4`) |
| DeepSeek-V4-Flash (full) | **256** | **6** | 4096 | 2048 | `config.json@60d8d70` (`n_routed_experts=256`, `num_experts_per_tok=6`, `num_hidden_layers=43`, `expert_dtype=fp4`, `norm_topk_prob=true`) |
| GLM-5.2 (full) | **256** | **8** | 6144 | 2048 | `crates/onnx-runtime-ep-cuda/tests/qmoe_gpu.rs:1741` (`GLM_5_2_MOE`, cited `zai-org/GLM-5.2/config.json`) |
| DeepSeek-V2-Lite (tiny stand-in) | **64** | **6** | 2048 | 1408 | `qmoe_gpu.rs:1727` (`DEEPSEEK_V2_LITE_MOE`, cited `DeepSeek-V2-Lite/config.json`) |

### 6.2 Telemetry footprint and per-step work

- **Bitmap (rep. A)** bytes/layer = `ceil(E/32)*4`:
  - `E=4` → 4 B; `E=64` → 8 B; `E=256` → **32 B**.
  - Whole model (worst case all layers MoE): DeepSeek-V4-Flash `43 * 32 B ≈ 1.4 KiB`;
    GLM-5.2 similarly ≈ 1.4 KiB. This is the entire per-session telemetry footprint.
- **Dedup queue (rep. B)** bytes/layer = `capacity*4 + 32 + ceil(E/32)*4`. With
  `capacity = E` it is strictly larger than the bitmap and cannot beat it on
  footprint; its only reason to exist is order/recency. At decode the distinct
  routed count per layer per step is `≤ top_k` (≤ 8), so a per-step queue needs
  only `capacity ≥ top_k`; a window of `W` steps needs `capacity ≥ min(E, W*top_k)`.
- **Per-step producer work** = `rows*top_k` atomics per MoE layer. At decode
  (`rows=1`) that is **6–8 `atomicOr`s per layer** — far below the route kernel's
  own top-k reduction and orders below the expert GEMV. The record read on the
  host is one `dtoh` of ≈1.4 KiB per decision, at a boundary, not per step.

### 6.3 Cacheability headroom (the number that decides whether a policy can win)

A route observer is only worth wiring if the routed working set is skewed enough
that pinning a subset removes real traffic. Cited, measured, prior:

- Experts are **67.6% (f32) / 77.2% (fp16)** of the weight bank, ≥3.5× the
  read-every-token dense floor (`2026-08-18-moe-per-expert-dispatch-seam-design.md`).
- Routing skew is **mild**: top 12.5% of expert keys → **27.1%** of traffic
  (`docs/benchmarks/2026-08-18-moe-per-expert-paging-churn.md`); the offline
  simulator (`scripts/moe_expert_cache_sim.py` over `scripts/moe_expert_trace.json`,
  granite 32-expert/top-8 real router) is the reusable falsifier for a policy.
- **Unknown carried forward:** whether granite-scale skew generalises to a
  256-expert/top-6..8 router (`moe_expert_cache_sim.py` header, labelled
  INFERRED). The telemetry this slice specifies is exactly what would *measure*
  it on the real model instead of inferring it.

### 6.4 GO / NO-GO gates (falsifiable, no speedup quoted)

Telemetry graduates from observer to policy input **only if all** hold; any
failure is a NO-GO for wiring (the observer stays inert / gets redesigned):

| Gate | GO threshold | NO-GO signal |
|---|---|---|
| **G1 kernel overhead** | telemetry GPU-event time ≤ **2 µs** at decode **and** ≤ **1%** of the QMoE route+GEMV step (measured, §7 microbench, ramped A100) | telemetry time measurable against the step |
| **G2 host sync** | **0** added `dtoh`/`synchronize` per decode step; record read only at a boundary | any per-step host sync |
| **G3 capture safety** | captured graph replays and the record reflects **each replay's real routes**, buffer VA unchanged, `fallbacks==0`, tokens byte-identical | capture declines, or replay needs a sync, or VA moves |
| **G4 bounded/fail-closed** | rep. A never overflows; rep. B `overflow=1` on capacity-exceed and consumer falls to whole-bank; poison/stale/foreign all fail closed | any partial-trust of a poisoned record |
| **G5 queue sizing (rep. B only)** | `capacity ≥ observed distinct/window` with `overflow==0` across the corpus | overflow occurs in normal decode → queue undersized |
| **G6 headroom** | measured `byte_hit_rate` of the pinned hot-set > the dense floor on the **real** model (via this telemetry) | miss_rate high / flat routing → policy cannot beat whole-bank |

`byte_hit_rate` (not count `hit_rate`) is the cost metric per
`measurement-discipline` §4. Wall-clock is corroboration only; no tok/s or
speedup is claimed here.

---

## 7. Inert proof harness (this slice's code)

`crates/onnx-runtime-ep-cuda/tests/expert_route_telemetry_probe_gpu.rs` — a
**new, test-only** integration test. It is a separate compilation unit: it needs
**no change to `src/lib.rs`, `weight_paging.rs`, `coarse_residency.rs`,
`vmm_allocator.rs`, or `ep-api`** (all PR #1854 files) and wires **no production
residency/lifecycle** path. It allocates its own device buffers and compiles its
own NVRTC kernels through the public `CudaRuntime` API only.

Contents:

- **CPU oracle** — pure-Rust reference bitmap and deduplicated queue from a route
  array; the ground truth every GPU test diffs against (runs with no GPU).
- **CUDA kernels** (NVRTC, in-harness): `route_bitmap` (atomicOr + poison on
  out-of-range), `route_dedup_queue` (dedup via seen-bitmap, atomicAdd, overflow
  bit), both stamping epoch/request/device identity.
- **CUDA tests** (`#[ignore]`, idle-GPU): bitmap == oracle; dedup set == oracle;
  overflow sets the bit and clamps count (fail-closed); epoch advances and a
  stale epoch is detectable; identity mismatch (foreign request/device) fails
  closed; **capture/replay** re-accumulates each replay's real routes into a
  stable-VA buffer with no host sync during capture; multi-request/multi-device
  isolation fails closed.
- **Microbenchmark** — GPU-event kernel time (batched launches enclosed by
  events) and host enqueue time reported **separately** (cuda-perf-measurement
  Trap 4), a captured-graph replay timing, device ramped and re-checked idle
  (Traps 5/6). No wall-clock speedup claim.

Run (solo, idle GPU):

```bash
CUDA_VISIBLE_DEVICES=<idle> cargo test -p onnx-runtime-ep-cuda \
  --features cuda-13000,gpu-tests --release \
  --test expert_route_telemetry_probe_gpu -- --ignored --nocapture --test-threads=1
```

### 7.1 Measured results (idle A100-SXM4-80GB, GPU 5, CUDA 13.0, driver 580.105.08)

All 8 tests pass (1 CPU-only + 7 GPU `#[ignore]`), `--test-threads=1`, GPU 5
verified idle (210 MHz, 4 MiB) before the run:

| Test | Result |
|---|---|
| `cpu_oracle_and_validator_self_consistent` | OK — bitmap set == dedup set; validator accepts clean, fails closed on poison/overflow/req/dev/stale |
| `cuda_route_bitmap_matches_cpu_oracle` | OK — E=4/64/256, rows=1 and 37; device bitmap byte-equal to oracle (E=256 rows=37 → 150 routed) |
| `cuda_dedup_queue_matches_cpu_oracle_set` | OK — 199 distinct, `overflow=0`, device set == oracle |
| `cuda_dedup_overflow_fails_closed` | OK — distinct=226 > capacity=8 → `overflow=1`, consumer → `WholeBank` |
| `cuda_poison_out_of_range_fails_closed` | OK — out-of-range id → `poison=1` → `WholeBank` |
| `cuda_identity_isolation_fails_closed` | OK — foreign request and foreign device both → `WholeBank`; owner accepts |
| `cuda_capture_replay_reaccumulates_real_routes` | OK — 3 replays re-accumulated each replay's real routes, buffer VA stable, epoch 1→2→3, stale (epoch 3 vs boundary 4) fails closed |
| `microbench_telemetry_overhead_gpu_event_and_host_enqueue` | decode (routes=8, E=256): GPU/launch **median 2.48 µs**, host-enqueue/launch **median 2.47 µs**; prefill (routes=4096, E=256): GPU **2.99 µs**, host-enqueue **2.61 µs** |

**Reading the microbench (measurement-discipline):** the standalone telemetry
kernel is *launch-bound* — its GPU-event time (2.48 µs) essentially equals the
host enqueue time (2.47 µs), i.e. the trivial `atomicOr` bitmap cannot run faster
than the host feeds a separate launch. This is the pessimistic upper bound for a
**separate** launch. Production integration (§8) folds the atomics into the
existing `qmoe_route` kernel, so the marginal cost is a few `atomicOr` per route
with **zero** extra launches — well inside gate **G1**. No speedup is claimed;
these are kernel/host overheads reported separately per Trap 4.

**Gate status from this harness (observer-only evidence):** G2 (0 added per-step
sync — record read only at boundary), G3 (capture replays reflect each replay's
real routes, VA stable), and G4 (bounded + fail-closed on overflow/poison/stale/
foreign) are **demonstrated inert**. G1 is bounded but must be re-measured *fused*
in Slice 7. G5 (queue sizing) and G6 (real-model byte-hit headroom) require the
real routing corpus and remain open (§8).

---

## 8. Exact next implementation slice (not in scope here)

Slice 7 (production wiring, gated on §6 GO): add telemetry as **scratch slot(s)**
of `QMoEKernel`/`BlockQuantizedMoEKernel` (new slots in `ScratchPool`,
`qmoe.rs`), write it from `qmoe_route`/`bqmoe_route` (or one appended
telemetry kernel on the same stream), expose a boundary-time consumer that
produces a per-expert desired-set, and feed that set to the **existing** Slice
4/5 coarse-boundary plan application (`coarse_residency.rs`) as its policy input —
with **no** new allocator, **no** id→slot rewrite, and every mapping change still
owned by PMM/VMM through `capture_gate::synchronizing_section()`. New types:
`ExpertRouteTelemetry` (buffer handle + header), `RouteObserverPolicy` (record →
desired-set). New tests: an end-to-end capture/replay that asserts the observed
set matches a `dump_route_selection` control, and a residency-decision test that
asserts fail-closed → whole-bank on a poisoned record.
