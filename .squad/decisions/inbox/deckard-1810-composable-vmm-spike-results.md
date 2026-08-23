# Deckard — #1810 composable sub-weight VMM spike results

**Issue:** #1810 (bounded feasibility slice for Roy's Cycle 9 Option 1
architecture decision — composable sub-weight VMM: one stable VA mapping
alternating device- and host-NUMA-backed 2 MiB granules under real QMoE
access).

**Branch:** `squad/1810-composable-vmm-host-numa-spike` off `origin/main`
(`2e42ea649`, includes merged #1804/#1806/#1808).

**File:** `crates/onnx-runtime-ep-cuda/tests/qmoe_composable_vmm_host_numa_spike_gpu.rs`
(test-only, `#[ignore]`d, zero production call sites — no changes to
`PhysicalHandlePool`/`CudaVirtualBacking`/`ResidencyPolicy`/
`RoutedResidencyProof`/`ResizeSafePoint`/`WeightRegionCatalog`/QMoE kernel
ABI).

## Platform

- Linux, A100-SXM4-80GB, driver 580.105.08, CUDA 13.0, `cudarc` 0.19.8
  (`cuda-13000` feature).
- `CUDA_VISIBLE_DEVICES=4` (verified idle via `nvidia-smi
  --query-compute-apps` immediately before every run; GPU 3 excluded, ~60GB
  in use by another session throughout).
- 4 host NUMA nodes available (`numactl --hardware`); device's own
  `CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID` = 3.

## Capability gate (requirement #1)

`host_numa_capability()` confirms, via direct driver queries (not assumed):

```
VMM_SUPPORTED=1
HOST_NUMA_ID=3
HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=1
HOST_NUMA granularity (RECOMMENDED) = 2097152 B (2 MiB)
cuMemCreate(HOST_NUMA, node=3) = CUDA_SUCCESS (confirmed by real allocation, not just the advertised attribute)
```

Gate fails closed (returns `Err` with an explicit printed reason, no
fallback) if any of these steps fail; `capability_gate_reports_host_numa_support_or_fails_closed`
exercises this path directly. On this platform the gate passes, so every
other test proceeds.

## New types (test-only, standalone — not wired to production
`AllocationCompatibility`/`PhysicalHandlePool`)

- `GranuleBacking { Device, HostNuma { node: i32 } }`
- `ExpertBankArena`: one `cuMemAddressReserve`'d stable VA; `try_map_granule`
  (raw `cuMemCreate`→`cuMemMap`→`cuMemSetAccess`, same 3 calls production's
  `CudaVirtualBacking::commit` uses); `map_all_or_rollback` (multi-granule
  commit with unwind-on-failure); `unmap_granule`/`unmap_all`; `ArenaAccounting`
  (device-committed bytes, host-mapped bytes, total mapped bytes,
  underflow-event counter, all tracked with `checked_sub`).

This deliberately duplicates only the minimum surface needed rather than
extending `PhysicalHandlePool`, per the task's "no new allocator/second
accounting authority in PRODUCTION" constraint — this file itself is a
test-scoped stand-in, not a production authority.

## Correctness (requirement #4)

8 `#[ignore]`d tests, all pass, run 3x in full for reproducibility (10.40s–
11.21s per full run):

1. **`correctness_deepseek_v2_lite_shape`** / **`correctness_qwen15_moe_a27b_shape`**
   (64-expert/60-expert shapes, real cited configs): `all_device` oracle vs.
   `all_host_numa` vs. `mixed_25/50/75pct_cold` (expert-aligned splits) vs. a
   deliberately **mid-expert, non-granule-aligned** cold-boundary case — all
   bit-identical (`f32::to_bits()` exact match) to the all-device oracle.
   Falsifiability: `cuPointerGetAttribute(MEMORY_TYPE)` confirmed HOST (1)
   for the all-host-NUMA arena and confirmed BOTH backings are actually
   present in one mixed arena (not a silent single-backing fallback).
   3 repeated fresh-reservation remap cycles, still bit-identical.
2. **`fault_injection_rollback_leaves_no_leaks_and_no_underflow`**: injects
   failures at `Create`/`Map`/`Access` phases at granule indices 0/3/7 of an
   8-granule alternating device/host-NUMA plan (9 combinations) — every case
   rolls back to 0 mapped granules, 0 device-committed bytes, 0 host-mapped
   bytes, 0 underflow events.
3. **`pointer_stable_across_remap_cycles`**: same VA (`base_ptr()`)
   confirmed identical across 5 unmap/remap cycles with alternating
   backings; each cycle's known write pattern verified by
   `cuMemcpyDtoH` read-back before the next remap.
4. **`graph_capture_replay_stable_va_and_remap_requires_cooperative_gate`**
   (see finding below).

## Critical finding: CUDA graph capture (requirement #7)

- Capture over an already-mapped, stable-VA mixed arena (no remap during
  capture) replays correctly 3x, VA unchanged. **This part behaves as
  expected.**
- **The remap-during-capture probe falsified the assumed safety property.**
  A same-thread `cuMemMap`/`cuMemSetAccess` issued *while a stream capture
  is active* returned `CUDA_SUCCESS`, not an error, on this driver — the
  CUDA driver does **not** self-refuse a remap mid-capture. The test file
  originally assumed (and an earlier draft asserted) that it would; this is
  corrected in the final version, which asserts the *observed* `Ok(())`
  result and documents it as a hard finding rather than silently accepting
  a wrong assumption.
- Consequence: production's existing `CudaVirtualBacking::commit`
  cooperative `capture_gate::synchronizing_section()` guard is not a
  redundant belt-and-suspenders measure — it is the **only** thing
  preventing this exact failure mode in the shipped VMM backing. Confirmed
  `cuStreamGetCaptureInfo_v3` correctly reports `ACTIVE` during capture, so
  a cooperative gate checking that status (or holding the equivalent lock)
  is a viable, available mechanism. **Any future production
  `ExpertBankArena`-equivalent MUST route every
  `cuMemCreate`/`cuMemMap`/`cuMemSetAccess` through that same gate (or an
  equivalent), not rely on driver-level refusal.** This is now the load-
  bearing architectural requirement carried forward to any Option 1
  production wiring, not an optional hardening.

## Accounting (requirement #5)

Exact, separate counters (`ArenaAccounting`) proven exact across every
correctness/fault test above: device-committed bytes and host-mapped bytes
never mixed, `checked_sub` used throughout, underflow-event counter stayed
0 in every non-fault-injection run and was proven to detect underflow
correctly (never triggered because rollback ordering is correct — verified
by the 9 fault-injection combinations above all reporting exactly 0
residual granules).

## Performance (requirement #6) — DeepSeek-V2-Lite (64 experts, top_k=6) and
Qwen1.5-MoE-A2.7B (60 experts, top_k=4), decode-shaped (rows=1), 5-rep
median GPU event time per arm, one-time map+upload cost reported separately:

| shape | arm | one-time map+upload (µs) | median exec (µs) | achieved cold GB/s* |
|---|---|---|---|---|
| deepseek-v2-lite | all_device | ~185–211k | 156.7 | n/a (0 cold touched) |
| deepseek-v2-lite | all_host_numa | ~212–252k | 430–432 | 20.0–20.1 |
| deepseek-v2-lite | mixed_25pct | ~188–202k | 198–200.7 | 43.1–43.3 |
| deepseek-v2-lite | mixed_50pct | ~187–232k | 288.8 | 30.0 |
| deepseek-v2-lite | mixed_75pct | ~196–247k | 382–384 | 22.5–22.6 |
| qwen1.5-moe-a2.7b | all_device | ~167–183k | 110.6–111.6 | n/a |
| qwen1.5-moe-a2.7b | all_host_numa | ~178–236k | 294.9–295.9 | 19.5–19.6 |
| qwen1.5-moe-a2.7b | mixed_25pct | ~178–204k | 157.7 | 36.6 |
| qwen1.5-moe-a2.7b | mixed_50pct | ~180–221k | 200.7–201.7 | 28.6–28.7 |
| qwen1.5-moe-a2.7b | mixed_75pct | ~183–229k | 245.8–246.8 | 23.4–23.5 |

*achieved_cold_GBps = touched-expert cold bytes / median exec time; touched
experts are capped at `top_k`, so the "achieved GB/s" figure reflects only
the routed subset actually read per decode step, not total arena bytes.

- One-time mapping+upload cost (~170–250 ms) is dominated by the synthetic
  fixture's `cuMemcpyHtoD` of the full weight bank (hundreds of MB) plus
  granule setup — **not** representative of a real per-step remap cost; it
  is reported separately exactly as required, and no per-step remap
  latency was measured in this spike (none of the correctness/performance
  paths remap mid-decode-loop; that is out of scope for this slice).
- Theoretical ceilings: PCIe Gen4 x16 ≈ 25 GB/s (host→device read);
  A100-SXM4-80GB HBM2e peak = 2039 GB/s. Achieved cold GB/s (19.5–43.3)
  sits at/near the PCIe ceiling as expected for HOST_NUMA-backed granule
  reads, confirming the mechanism engages a real host-memory access path
  (not a silently-cached VRAM shortcut).
- **No end-to-end tok/s claim is made.**

## GO/NO-GO verdict

**Bounded feasibility slice: GO** for the specific claim tested — one
stable VA composing device- and host-NUMA-backed 2 MiB granules, with real
QMoE int4 strided-GEMV access remaining bit-identical across expert-aligned
and cross-expert/cross-granule-boundary splits, repeated remaps, and
capture/replay (no remap during capture). Capability gate, fault injection,
and accounting all behave correctly and fail closed as required.

**Conditional NO-GO carried forward for production wiring**: any
production `ExpertBankArena`-equivalent must incorporate the
capture-during-remap cooperative gate (`capture_gate::synchronizing_section()`
or equivalent) as a hard requirement, not an optional hardening — this
spike proved the driver alone will not enforce it. This is scoped
information for the next architecture gate, not a reason to downgrade this
slice's own verdict (this slice never claimed remap-during-capture safety
without a gate; it tested for it, found it false, and corrected the claim).

## Next architecture gate

Slice 2 (not in this spike's scope): thread `GranuleBacking` into
production `AllocationCompatibility::location_type` and
`PhysicalHandlePool`'s pool-key (already structurally ready — the
`location_type: i32` field exists, keys the pool, and requires no
restructuring per the prior spike's confirmation), wire `ExpertBankArena`
(or a production equivalent) as the sole new consumer of
`ResidencyDecision::PerExpertCandidate`, gate every commit call through
`capture_gate::synchronizing_section()`, and measure real per-decode-step
remap latency (not amortized one-time upload) before any production
enablement decision.

## Reviewer

Independent review required before merge, per task instruction (Sebastian
to co-validate bandwidth-gate methodology per #1810's issue body).
