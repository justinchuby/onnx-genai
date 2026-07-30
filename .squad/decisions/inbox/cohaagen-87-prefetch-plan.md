# #87 — Compute–transfer overlap / async weight prefetch (Phase 4): ASSESS + PLAN

**Author:** Cohaagen (CUDA/systems) · **Date:** 2026-07-30 · **Status:** assessment / plan — NOT yet implemented (awaiting Justin's green-light)
**Depends on:** #63 live GPU weight offload (MERGED, PR #444) · **Relates to:** `docs/WEIGHT_OFFLOAD.md` §4, `.squad/decisions/inbox/cohaagen-63-offload-gaps.md`

---

## TL;DR

Most of #87's *mechanism* is **already shipped and GPU-tested**. What is missing is the **live wiring**: the device residency page-in path still uses a **synchronous** `cuMemcpyHtoD`, and the dispatch loop pages each lazy weight **right before** the kernel consumes it, so there is **zero compute/transfer overlap** in the offload hot-path. The prerequisite the existing code called out — "Phase-3b live device weight binding" — is exactly what #444 (my merged increment) delivered, so the wiring is now **unblocked**.

The low-risk first increment is small and self-contained: **switch the residency page-in to the existing async copy-stream + fence machinery** (so a page-in overlaps the *current* kernel and the compute stream waits only on the transfer's event), gated behind the same `ONNX_GENAI_WEIGHT_OFFLOAD` policy. The full double-buffered look-ahead schedule (`plan_double_buffer`/`drive_double_buffer`) is a second increment layered on top.

---

## 1. What already exists (do NOT rebuild)

Grep-confirmed in this tree (fresh `origin/main`, incl. #444):

**EP mechanism — landed & GPU-tested** (`crates/onnx-runtime-ep-cuda/src/runtime.rs`, `provider.rs`):
- Dedicated **transfer/copy stream**: `CudaRuntime::copy_stream()` (runtime.rs:402), constructed as a non-default stream (runtime.rs:324).
- Real **async H2D**: `htod_async` (runtime.rs:944, `cuMemcpyHtoDAsync` on the copy stream), `dtod_async_on_copy_stream` (runtime.rs:968).
- Genuine **completion events**: `record_copy_fence` (runtime.rs:990), `compute_wait_fence` (runtime.rs:1023, `cuStreamWaitEvent` — non-host-blocking), `copy_wait_fence` (runtime.rs:1031, WAR direction), `sync_copy_stream` (runtime.rs:1058).
- **Pinned host staging** allocation (runtime.rs:~1065).
- EP-API surface implemented on the CUDA EP: `copy_async` (provider.rs:474) → real `Fence`, `wait_fence` (provider.rs:518), `copy_wait_fence` (provider.rs:533). GPU test: `copy_async_fence_orders_h2d_prefetch_through_ep_api` (provider.rs:693) — proves the fence actually orders a consumer after the transfer (reads poison without the fence).
- `Fence { id }` with id `0` = already-signalled (`onnx-runtime-ep-api/src/provider.rs:288`); synchronous EPs are correct no-ops.

**Executor strategy — landed & unit/GPU-tested** (`crates/onnx-runtime-session/src/executor/prefetch.rs`):
- `plan_double_buffer(num_experts)` (prefetch.rs:86): pure, inspectable schedule of `PrefetchStep::{Prefetch, Await, Compute}` that keeps the copy stream one expert ahead of compute; 2 device slots, alternating `n % 2`.
- `drive_double_buffer(...)` (prefetch.rs:136): EP-agnostic driver over `&dyn ExecutionProvider` (`copy_async` + `wait_fence` + WAR `copy_wait_fence`). Degrades to a correct sequential run on a synchronous EP.
- WAR reuse safety enforced by the driver itself; GPU regression `drive_double_buffer_war_safe_across_waves` corrupts if the WAR fence is removed.

**Host-side prefetch precedent** (`crates/onnx-runtime-ep-cpu/src/kernels/qmoe.rs`): route-first mmap→dequant prefetch pipeline with byte-identical + stat-identical A/B guards (`prefetch_pipeline_is_byte_and_stat_identical_to_serial`, `prefetch_matches_serial_across_routing_and_budget_sweep`, `prefetch_ab_benchmark`). This is the **host↔host** cache prefetch; #87 is the **host↔device** DMA overlap. Its A/B discipline is the template to reuse.

## 2. What is missing (the actual #87 gap)

The **live device residency page-in is fully synchronous and serializes with compute**:

- `CudaWeightPage::upload` (`weight_paging.rs:164`) and the reload path (`weight_paging.rs:270`) use the **synchronous** `runtime.htod(...)` (`cuMemcpyHtoD_sync`), not `htod_async` on the copy stream.
- `CudaWeightResidency::admit` (`weight_paging.rs:388`) does a full `runtime.synchronize()` before eviction (correct for safety, but host-blocking).
- The dispatch loop (`crates/onnx-runtime-session/src/executor/dispatch.rs:844`) calls `ep.page_lazy_weight(vid, lazy)` **inline, immediately before** binding the device view the kernel consumes. There is **no look-ahead**: the transfer of weight *N* cannot overlap the compute of weight *N-1* because it is not issued until weight *N* is already needed.

Net: at a constrained VRAM budget the offload path pays the H2D cost on the critical path every page-in. Measured on my #444 validation (Qwen3-0.6B int4 native CUDA, 2 MiB device budget — an *extreme* budget forcing page-in on essentially every matmul): **~1.21× decode wall-time** vs. offload-off, with page_ins≈12.5k. That 1.21× is the (roughly) fully-serialized H2D tax; it is the **theoretical ceiling** of what perfect overlap could hide at that budget (≈17% of decode time). Realistic budgets page far less often, so the absolute win shrinks — overlap matters most exactly when the working set barely fits.

## 3. Overlap strategy

**Use a dedicated copy stream + CUDA events (already built). `cp.async` is NOT applicable** — `cp.async` is a Hopper/Ampere *shared-memory* (gmem→smem) register-prefetch instruction inside a kernel; #87 is *host↔device* DMA, which is a copy-engine/stream concern, not an in-kernel one.

Two layered increments:

### Increment 1 (LOW RISK, proposed first) — async page-in on the copy stream
Make the residency page-in **non-blocking on the compute stream**:
1. Add `CudaWeightPage::upload_async` that allocates the page and issues `htod_async` on `copy_stream`, returning `(page, Fence)` instead of synchronously copying.
2. `resident_materialized` (and `page_lazy_weight`) return a page carrying its readiness `Fence`; dispatch calls `wait_fence` on the **compute stream** right before binding (RAW order) — a `cuStreamWaitEvent`, not a host sync.
3. Eviction safety unchanged: `admit` still records/awaits the compute fence before freeing a slot (WAR), reusing `record_compute_fence`/`copy_wait_fence` instead of the current full host `synchronize()` where possible.

Even with **no look-ahead**, this lets the *next* page-in's transfer begin overlapping the *current* kernel (the copy stream runs ahead of the compute stream by one dispatch), and removes host-thread blocking on every copy. It is a strict superset of the shipped `copy_async` path already GPU-proven by `copy_async_fence_orders_h2d_prefetch_through_ep_api`.

### Increment 2 (MEDIUM, after Increment 1 proves out) — double-buffered look-ahead
Drive the live dispatch loop with the shipped `plan_double_buffer`/`drive_double_buffer` schedule so weight *N+1* is prefetched into the alternate device slot **while** weight *N* computes. This requires the dispatch/executor to know the *upcoming* lazy-weight order for a node/wave (MoE expert order, or the per-layer MatMulNBits sequence) and to hold **two** residency slots per pageable weight class. This is where the real 2× headroom is, but also where the risk lives (see §5).

## 4. Where it hooks into the code

| Concern | File · symbol | Change |
|---|---|---|
| Async page-in mechanism | `ep-cuda/src/weight_paging.rs` · `CudaWeightPage::upload` (164), reload (270) | add `upload_async` using `runtime.htod_async` on `copy_stream`; return readiness `Fence` |
| Residency return type | `ep-cuda/src/weight_paging.rs` · `resident_materialized` (354), `admit` (388) | carry `Fence`; use `record_compute_fence`/`copy_wait_fence` for WAR instead of blanket `synchronize()` |
| EP paging surface | `ep-cuda/src/provider.rs` · `page_lazy_weight` (193) | thread the `Fence` out via `PagedWeight` (or a sibling `page_lazy_weight_async`) |
| PagedWeight readiness | `onnx-runtime-ep-api/src/weight.rs` · `PagedWeight` (320) | optional `fence: Fence` field (default signalled → back-compat for sync EPs) |
| Dispatch consume order | `onnx-runtime-session/src/executor/dispatch.rs:844` | `wait_fence` on compute stream before binding; (Increment 2) issue next-weight `copy_async` before current compute |
| Look-ahead schedule | `onnx-runtime-session/src/executor/prefetch.rs` · `plan_double_buffer`/`drive_double_buffer` (already exist) | wire into the live loop (Increment 2) |
| Policy gate | `ep-cpu` `WEIGHT_OFFLOAD_ENV` / `ep-cuda` `DeviceOffloadPolicy`; add `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH` (mirror `ep-cpu`'s existing `WEIGHT_OFFLOAD_PREFETCH_ENV`) | prefetch defaults ON when offload on, but A/B-toggleable |

**Interaction with PR #446 (offload⊄capture):** offload is already mutually exclusive with CUDA graph capture, so prefetch never runs under capture — the copy-stream events are always legal. No new capture interaction.

## 5. Risk register

1. **Prior SW-prefetch regression precedent (HIGH-attention).** We have recorded evidence that a *naive software prefetch* REGRESSED issue-bound kernels (gate-up GEMV SW-prefetch decisions). **That risk does not directly transfer** — that was *in-kernel register/`cp.async` prefetch* competing for issue slots/registers on a compute-bound GEMV. This is *host↔device DMA on a separate copy engine*, hiding *bandwidth/latency*, touching no kernel registers. **But the lesson stands:** only a win when the path is transfer-bound; on a compute-bound / resident-fits budget it can only add scheduling overhead. → **Gate + default-safe + A/B before claiming any win.**
2. **Extra VRAM for double-buffering (Increment 2).** Two live slots per pageable weight raises the effective residency floor; on a *tight* budget that can *increase* eviction churn and net-lose. → Increment 2 must make double-buffering budget-aware (fall back to single-buffer async when the budget can't hold 2 slots).
3. **Eviction/WAR correctness under async.** Freeing a page whose transfer or consuming kernel is still in flight = use-after-free. → Reuse the *already GPU-tested* fence discipline (`compute_wait_fence`/`copy_wait_fence`); never free without the consumer's compute fence resolved. Add a forced-tiny-budget async eviction GPU test mirroring `residency_never_evicts_a_referenced_page`.
4. **Token divergence.** Any overlap bug shows as wrong tokens. → Reuse #444's token-parity harness: offload-off baseline vs. offload+prefetch, **token-exact**, on a trusted model (qwen2.5-7b-instruct int4 MatMulNBits), plus counters (`page_ins`/`evictions` > 0) so the test can't pass with paging disabled.
5. **PTX/driver copy-engine contention with KV D2H.** The per-step logits/argmax D2H already uses stream sync; a busy copy stream could serialize with it. → measure end-to-end tok/s, not microbench copy time.

## 6. A/B methodology (real numbers only)

- Model: `qwen2.5-7b-instruct-cuda-gpu-4/v4` (int4 MatMulNBits) — non-offloaded native output already trusted.
- Pin: `CUDA_VISIBLE_DEVICES=0 taskset -c 0`; `source .cudaenv.sh`. Device 0 only (Mary=4, Lori=5).
- Matrix, ≥3 runs each, report median tok/s + token IDs:
  1. offload OFF (baseline, trusted token IDs).
  2. offload ON, **prefetch OFF** (`ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH=0`) — the synchronous #444 path.
  3. offload ON, **prefetch ON** (Increment 1 async).
  4. (later) offload ON, prefetch ON, double-buffer (Increment 2).
- Sweep `ONNX_GENAI_WEIGHT_OFFLOAD_DEVICE_BYTES` across a tight (forces churn), a mid, and a loose budget to find where overlap actually helps and where it's neutral/negative.
- **Accept only if:** token IDs identical across all rows AND (3)/(4) tok/s ≥ (2) at the transfer-bound budgets AND not materially worse at loose budgets. If a budget regresses, keep prefetch **opt-in** there.

## 7. Recommended first increment (for green-light)

**Ship Increment 1 only**: async page-in on the copy stream + fence-ordered consume in dispatch, gated by `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH` (default ON when offload on). It:
- reuses 100% already-GPU-tested primitives (`htod_async`, `record_copy_fence`, `compute_wait_fence`),
- keeps the non-offload path byte-identical and the offload path token-exact,
- delivers the "next transfer overlaps current compute" win with **no** extra VRAM (single buffer),
- is independently A/B-able and reversible via the env flag.

Defer the double-buffered look-ahead (Increment 2) to a follow-up once Increment 1's A/B numbers justify the extra VRAM and scheduling complexity.

**Awaiting green-light before implementing.**

---

## 8. Increment 1 — IMPLEMENTED + A/B RESULT (2026-07-30)

**Status:** GREEN-LIT and shipped (branch `squad/87-async-prefetch-inc1`, stacks on #446). Env flag `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH=1` (device async path is **opt-in**; unset = synchronous #444 path, byte-identical).

### What shipped
- `CudaWeightResidency::admit_async()` (weight_paging.rs): stages canonical bytes into a **single reusable pinned buffer**, issues `htod_async` on the copy stream, `record_copy_fence()`, then `compute_wait_fence(fence)` so the single compute stream (which runs every kernel) waits on the copy before the consuming kernel reads it. All under the residency lock (page-ins serialized — fine for single-session decode).
- **Eviction/WAR/UAF safety:** before evict/alloc/pinned-refill, `synchronize()` (compute) + `sync_copy_stream()` (transfer) drain any prior in-flight copy. The drains happen BEFORE issuing the *current* copy, so a page-in never drains itself → stays overlappable. No async H2D can be in flight into a page that is freed/reused.
- Global observability counter `async_page_ins` (⊆ `page_ins`) so an opaque e2e decode can prove the async path actually ran (vs silent sync fallback).

### Validation (real numbers, Qwen3-0.6B int4, device 0, taskset-pinned)
- **Token parity (the correctness proof):** baseline (offload OFF) == sync offload (prefetch OFF) == async offload (prefetch ON), **byte-identical** token IDs `[12095, 11, 323, 279, 6722, 315, 15344, 374, ...]`. The copy fence orders H2D before the kernel correctly.
- **Paging exercised:** budget 64 MiB → `page_ins=12544`, `evictions=12441`; prefetch ON → `async_page_ins=12544` (prefetch OFF → `async_page_ins=0`, proving no silent fallback).
- **Perf A/B (median of 3 + warmup, 64 MiB realistic budget):** prefetch OFF = **2.62 tok/s** (12.209 s); prefetch ON = **2.61 tok/s** (12.243 s); **speedup 0.997× — a WASH** (within run-to-run noise).

### Honest verdict: WASH, not a win (as expected for single-buffer)
Increment 1 is **correctness/safety infrastructure**, not a perf win by itself. At this budget every weight is re-paged each step (`hits=0`), and **single-buffer** async issues each weight's copy *immediately before its own kernel*, then the fence forces that same kernel to wait on that same copy — there is no *other* weight's compute to overlap it with. Latency hiding requires **look-ahead**: prefetch layer N+1's weights on the copy stream *while layer N computes*. That is **Increment 2 (double-buffered look-ahead)** — now clearly justified: the async machinery is proven live + safe + token-exact + non-regressing (0.997×), so Increment 2 only needs to add the look-ahead schedule + budget-aware second slot.

**Recommendation:** merge Increment 1 as the safe, reversible, opt-in foundation (no regression); green-light Increment 2 for the actual overlap win. Keep prefetch opt-in until Increment 2 demonstrates tok/s ≥ sync at realistic budgets.

---

## 9. Increment 2 — IMPLEMENTED + A/B RESULT (2026-07-30)

**Status:** GREEN-LIT and implemented (branch `squad/87-async-prefetch-inc2`, stacks on #447). Same env flag `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH=1`; look-ahead depth via `ONNX_GENAI_WEIGHT_OFFLOAD_PREFETCH_DEPTH` (default 2 when prefetch on; `0` = Increment-1 single-buffer). Default/unset = synchronous #444 path, byte-identical.

### What was implemented
- **Self-learned access order:** the residency records the autoregressive weight-access sequence (decode replays the same order every token). On the first repeat it freezes a `successors` map (next `depth` unique keys per key) and caches cheap `LazyWeight` clones by key (`lazy_by_key`). Zero dispatch/EP-trait changes — the residency prefetches itself.
- **Look-ahead page-in (`prefetch_admit`):** after serving a weight, issues async H2D copies for its learned successors on the copy stream into a **double-buffered pinned ring** (`depth+1` slots), recording a *deferred* copy fence per successor. On the later consume-HIT, `get_hit` does `compute_wait_fence(deferred)` so the consuming kernel waits only on the (already-overlapped) copy. The copy overlaps the intervening compute — the actual overlap the single-buffer path structurally can't do.
- **Budget→depth policy:** a successor is prefetched only if the *protected* working set (pages a live kernel still references + in-flight prefetched pages) leaves room (`protected + len <= budget`); otherwise skipped → graceful fallback to single-buffer under tight budgets (no thrash). Eviction (`evict_cold_to_fit`) reclaims only cold, unprotected pages, never a sibling in-flight prefetch.
- **Multi-fence eviction safety (the real hazard):** making room may evict a cold page, so `prefetch_admit` takes the same drains as the miss path — `synchronize()` (compute) + `sync_copy_stream()` (retires **every** outstanding look-ahead copy) — before freeing. So an evicted page never has an in-flight DMA writing it, even with multiple prefetch copies simultaneously in flight. WAR on the pinned ring is enforced per slot (host-wait the slot's prior copy fence before refilling). Two fences per copy (one WAR, one RAW) because fence events are single-use.

### Validation (real numbers, Qwen3-0.6B int4, device 0, taskset-pinned, median of 3 + warmup)
Token IDs **identical** across all three (baseline offload-OFF == single-buffer == depth-2) at every budget: `[12095, 11, 323, 279, 6722, 315, 15344, 374, 21718, 13, 576, ...]`. Look-ahead genuinely ran: `look_ahead_prefetches=12153` of `async_page_ins=12546` (proves the double-buffer path fired, not a silent fallback).

| Budget | single-buffer tok/s | depth-2 tok/s | speedup | evictions (sb→d2) |
|-------:|--------------------:|--------------:|--------:|-------------------|
| 32 MiB | 2.72 (11.75 s)      | 2.68 (11.95 s)| **0.983×** | 12493 → 12495 |
| 64 MiB | 2.73 (11.71 s)      | 2.65 (12.10 s)| **0.968×** | 12441 → 12441 |
|128 MiB | 2.73 (11.73 s)      | 2.59 (12.35 s)| **0.950×** | 12333 → 12333 |

### Honest verdict: REGRESSION (0.95–0.98×) — bankable negative, DO NOT MERGE the #87 stack
Look-ahead double-buffer is token-exact and safe, but **consistently ~2–5% slower** than single-buffer across every budget tried. Root cause is structural: safe cold-page eviction under a full budget requires a host `synchronize()` per look-ahead `prefetch_admit` (no stream-ordered async allocator is in use). At steady state depth-2 issues ~`depth` prefetch admits per consumed weight vs the single-buffer path's ~1 miss-admit — roughly **doubling the host synchronizes**. The overlap the copies buy does not recover that added serialization: **offload decode on this graph at these budgets is not H2D-latency-bound** — it is issue/compute + sync-overhead bound. This matches our prior speculative-perf negatives (o_proj / gate-up GEMV SW-prefetch).

**Consequence (per speculative-perf discipline):** keep #447 (Increment 1) and #448 (Increment 2) as **draft, unmerged**. Increment 1 remains the correct, safe, non-regressing async substrate; Increment 2's look-ahead does not turn it into a win, so the stack stays off `main`.

### What a real win would require (future Increment 3, not in scope)
Eliminate the per-eviction host `synchronize()` by adopting a **stream-ordered allocator** (`cuMemAllocAsync`/`cuMemFreeAsync` on the copy/compute streams) so cold pages free in stream order without a host block, letting the look-ahead copies overlap with *zero* added synchronization. That was assessed and deliberately deferred (cross-stream memory-pool reuse correctness is subtle/unverified on this driver — high blast radius). Only worth pursuing if a model/budget is shown to be genuinely H2D-bandwidth-bound (e.g. 7B/27B with larger per-weight bytes); 7B A/B is currently blocked on the genai_config→inference_metadata metadata workaround (the e2e harness requires `inference_metadata.yaml`; the Foundry 7B export ships only `genai_config.json`).
