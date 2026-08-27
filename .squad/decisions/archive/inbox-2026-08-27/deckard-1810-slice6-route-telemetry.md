# Deckard — #1810 Slice 6: device-side expert-route telemetry (design + inert harness)

**Issue:** #1810 (composable sub-weight VMM for MoE experts). Slice 6 =
adapt FreeToken's device-side expert-route observation/admission to
onnx-genai's QMoE/BlockQuantizedMoE **without** violating existing PMM/VMM and
CUDA-graph authorities. Scope stopped at a bounded design + inert proof harness;
**no** production residency/lifecycle wiring.

**Independence from PR #1854 (Slice 5, rejected — Deckard locked out):** this
work touches **none** of #1854's files (`vmm_allocator.rs`, `ep-api/{lib,weight}.rs`,
`ep-cuda/{coarse_residency,lib,weight_paging}.rs`, `coarse_residency_plan{,_gpu}.rs`).
The harness is a **new** integration test = its own compilation unit, so it needs
no `src/` edit and shares no code with #1854.

**Branch:** `squad/1810-slice6-route-observation-harness` (worktree at
`/home/justinchu/dev/onnx-genai-slice6-route-harness`, off `origin/main`
`5827df5a1`).

**Files (both new, additive):**
- `docs/memory/EXPERT_ROUTE_TELEMETRY_SLICE6_DESIGN.md` — the design (8 sections,
  all requirements, source-cited).
- `crates/onnx-runtime-ep-cuda/tests/expert_route_telemetry_probe_gpu.rs` —
  inert, test-only proof harness (CPU oracle + CUDA bitmap/dedup/overflow/
  poison/epoch/capture-replay/isolation + microbench). No production wiring.

## Decisions carried to Slice 7 (production wiring — GO-gated)

1. **Telemetry is an observer, not an authority.** Policy chooses a desired
   residency set from the record; **PMM/VMM alone** owns mapping, accounting,
   quarantine, rollback. **No** second LRU/cache/allocator, **no** id→slot
   rewrite (this is the FreeToken mechanism we reject — it breaks the contiguous
   expert-bank pointer ABI and adds a second authority).

2. **Contract:** fixed-capacity, GPU-resident. Default = **route bitmap**
   (`ceil(E/32)` u32 words, `atomicOr`); optional **bounded dedup queue** for
   miss-style policies. Fixed `u32[6]` header `{EPOCH, REQUEST, DEVICE, OVERFLOW,
   POISON, COUNT}`. Zero steady-state host sync; produced during launch;
   consumed **only** after the existing stream-completion authority at a coarse
   boundary. Preserves the contiguous expert-bank pointer ABI.

3. **Fail closed, always.** Overflow / poison (out-of-range id) / stale epoch /
   foreign request / foreign device ⇒ the consumer discards the record and falls
   back to the **whole-bank** proof. Proven inert in the harness.

4. **Capture/replay safety.** Producer captures into the decode graph and
   re-accumulates **each replay's real routes** into a stable-VA buffer with no
   host sync inside capture (mirrors FreeToken `lru_stats`,
   `offload_cache.py:193-203`). **Forbidden during capture/replay:**
   `cuMemMap`/`cuMemSetAccess` (the driver returns `CUDA_SUCCESS` mid-capture and
   does **not** self-refuse — the composable-VMM spike proved this; only
   `capture_gate::synchronizing_section()` guards it).

## Validation (idle A100-SXM4-80GB, GPU 5, CUDA 13.0, driver 580.105.08, cudarc 0.19.8)

All 8 tests pass (`--test-threads=1`, GPU verified idle before run):

- bitmap == CPU oracle (E=4/64/256, rows 1 & 37);
- dedup set == oracle (199 distinct, `overflow=0`);
- overflow fails closed (distinct=226 > cap=8 → `overflow=1` → `WholeBank`);
- poison fails closed; foreign request/device fail closed; owner accepts;
- capture/replay: 3 replays re-accumulated real routes, buffer VA stable,
  epoch 1→2→3, stale (epoch 3 vs boundary 4) fails closed;
- microbench (Trap-4 separated, ramped): decode telemetry launch GPU **2.48 µs**
  ≈ host-enqueue **2.47 µs** (launch-bound — pessimistic separate-launch upper
  bound; Slice 7 folds atomics into `qmoe_route` for ~0 extra launches). **No
  speedup claimed** (measurement-discipline; wall clock cannot resolve this).

**Independent review:** `code-review` agent (not Deckard) — **no high-confidence
bugs**; confirmed race-correct dedup atomics, genuine CPU/GPU cross-check (HashSet
vs seen-bitmap), fail-closed validator, correct capture region (no sync inside),
Trap-4 microbench separation, alignment-safe byte casts, and no
use-after-free (`DeviceBuffer` has no `Drop`; buffers outlive device use).

## GO/NO-GO status

- **Demonstrated inert:** G2 (0 per-step sync), G3 (capture replays real routes,
  VA stable), G4 (bounded + fail-closed).
- **Bounded, re-measure fused in Slice 7:** G1 (kernel overhead).
- **Open — need real routing corpus:** G5 (queue sizing), G6 (real-model
  byte-hit headroom). Whole-bank remains the safe default until G6 passes.

## Exact next slice (Slice 7, not started — GO-gated on §6)

Add telemetry as new **`ScratchPool` slot(s)** of `QMoEKernel`/
`BlockQuantizedMoEKernel` (`qmoe.rs`), write from `qmoe_route`/`bqmoe_route` (or
one appended telemetry kernel on the same stream), add a boundary-time consumer
→ per-expert desired-set, feed that set to the **existing** coarse-boundary plan
as policy input — **no** new allocator, **no** id→slot rewrite, every mapping
change still owned by PMM/VMM via `capture_gate::synchronizing_section()`. New
types: `ExpertRouteTelemetry`, `RouteObserverPolicy`. New tests: e2e
capture/replay observed-set vs `dump_route_selection` control; residency-decision
fail-closed → whole-bank on a poisoned record.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
