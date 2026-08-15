# Decision drop — PDL (Programmatic Dependent Launch) feasibility spike — **NO-GO**

**Author:** Gaff (CUDA performance engineer, decode-perf)
**Branch:** `squad/pdl-launch-overlap` (off `origin/main` 6dfc30cd, post-#1029)
**Date:** 2026-08-15T16:07:00Z
**Status:** DIAGNOSIS-FIRST SPIKE → **NO-GO. No PDL machinery built.** Evidence-backed stop.
**Model:** glm-4-9b-int4-cuda (`/home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda`), H200 (GPU 1 pinned), base non-speculative M=1 greedy decode.
**Refs:** profiling SKILL (`--cuda-graph-trace=node` mandatory); TP-feasibility launch-chain estimate (~2568 nodes, ~8.2us/node); certified ORT graph-on ~250 tok/s.

## Why (the lever we were told to test)
Our incremental GEMV/attention compute levers hit an ncu-proven floor. The
remaining dominant axis was hypothesized to be the **launch/latency chain**:
decode is a long serial chain of many tiny per-layer kernels, so per-kernel
launch/scheduling latency (not compute or bandwidth) is the critical path.
PDL (sm_90+ Hopper: `cudaGridDependencySynchronize()` +
`cudaTriggerProgrammaticLaunchCompletion()` + the
`cudaLaunchAttributeProgrammaticStreamSerialization` attribute) lets the NEXT
kernel begin its prologue (grid setup, weight prefetch) while the CURRENT kernel
drains its tail — overlapping the otherwise-serialized launch latency.

The gate (binding, from the task): build PDL **only if** inter-kernel gaps are
`> ~5%` of decode **AND** that gap is not already recovered by CUDA graph
capture. A NO-GO with evidence is a valid, valuable result.

## What I measured (Phase 1 — quantify before building)
Build: `cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native` (clean).
nsys: `nsys profile --cuda-graph-trace=node` on steady decode; per-node gap
analysis over the CUPTI kernel table (start/end per node on the single decode
stream). Medians; shared 8×H200 host → absolute tok/s carries host variance.

### Critical interaction: decode runs under CUDA graph capture BY DEFAULT
`crates/onnx-genai-engine/src/native_decode/load.rs` auto-enables graph capture
whenever the topology is structurally graph-safe (CUDA device + owned,
device-resident KV). Confirmed at runtime:
`cuda_graph: enabled=true captures=… replays=… fallbacks=0`. This is the
certified base-decode path we benchmark against ORT.

### The gap decomposition (captured = default path)
Within captured decode graphs (graphIds 2/5/8, ~80k kernel instances):

| component | % of decode | what it is | PDL can help? |
|---|---|---|---|
| kernel **active** | **78.3%** | real compute | n/a |
| **per-node launch gap** | **4.30%** (median **416 ns**/node) | GPU grid-dispatch between consecutive graph nodes | **This is PDL's entire domain** |
| **cross-replay gap** | **17.4%** (46 gaps, all at the single replay-boundary node 2666→2651, ~100–1000 us each) | host token-feedback between graph replays (argmax readback → next-token → relaunch) | **No** — not a kernel→kernel boundary |

The distribution is **bimodal**: tiny ~416 ns per-node gaps, or huge >100 us
replay-boundary gaps. There is essentially **nothing in 5–100 us**.

### The counterfactual proves capture already wins PDL's axis
Same workload with `ONNX_GENAI_CUDA_GRAPH=0` (eager, capture off):

| path | decode ms/tok | tok/s (48-tok run) | active % | per-node launch gap |
|---|---|---|---|---|
| **eager (capture off)** | 9.44 | 108.8 | ~48% | **~30%** of decode (median 1280 ns/node) |
| **captured (default)** | 5.43 | 184.7 | 78.3% | **4.30%** (median 416 ns/node) |

CUDA graph capture collapses the per-node launch chain from **~30% → 4.3%**
(+70% tok/s, ~4 ms/token recovered). That launch latency **is exactly what PDL
targets** — capture already occupies PDL's domain.

## Verdict — NO-GO (fails the gate on both clauses)
1. **Gap is below threshold.** Under the default captured path the
   PDL-addressable per-node gap is **4.30% < ~5%**, with a median of **416 ns**/node —
   near the GPU grid-dispatch floor. PDL overlaps only the *prologue* fraction of
   that residual, so realistic upside is **~1–2% of decode at best**, well inside
   shared-host timing noise.
2. **Already recovered by capture.** The launch/latency chain is real and large
   (~30% eager) but CUDA graph capture — on by default, `fallbacks=0` — already
   recovers ~85–90% of it. PDL is **redundant with capture** on this path.
3. **The real remaining idle is out of PDL's reach.** The dominant non-compute
   cost (17.4%) is **cross-replay host token-feedback latency** (data-dependent
   CPU control flow between graph replays), not a kernel→kernel boundary. Levers
   there are **on-device sampling / CUDA-graph conditional nodes**, not PDL.
4. PDL would only help the **eager fallback** (capture declined: weight-offload
   without stable-VA, non-owned/non-CUDA KV) — a degraded path we already avoid.
   Not worth the intrinsics + correctness/parity risk to speed a path we don't ship.

**Decision: do NOT build the PDL machinery.** No producer/consumer intrinsics, no
launch-attribute plumbing, no opt-in flag. This matches the task's own stated
hypothesis ("captured graphs already elide per-node launch latency… capture may
already win what PDL would win") — measured and confirmed.

## Arch applicability (recorded for future readers)
PDL is **Hopper + Blackwell, sm_90+** (`__CUDA_ARCH__ >= 900`). **Not** available
on Ampere/Ada (RTX 30/40, A100, L40). Even where available, the finding above
holds wherever decode runs under graph capture: **capture subsumes PDL's benefit.**

## Where to look next (higher-leverage than PDL)
- The 17.4% cross-replay host-feedback gap: on-device greedy/sampling to avoid the
  argmax→host→relaunch round-trip, or CUDA-graph conditional nodes to fold the
  token-feedback into the captured graph (keeps the win inside one launch).
- Compute floor stays the GEMV story (interleave-dequant/fp16 levers already merged).

## Files touched
- **None** (source). Diagnosis-only spike.
- This decision drop.
- Scratch nsys artifacts were generated under the worktree and removed; not committed.
