---
title: "MoE offload — the prediction-window measurement (dynamic prefetch feasibility)"
date: 2026-08-18
hardware: "Intel i7-13800H (14C/20T), RTX 4060 Laptop 8 GB, WDDM, driver 591.55, CUDA 13.1"
model: "granite-3.0-1b-a400m-instruct (32 experts, top-8, 24 layers), onnxruntime 1.27.0 CPU EP, batch 1"
status: measured (topology, CPU wall-clock) + inferred (page-in cross-ref, GPU scaling) — separated
---

# MoE offload — the prediction-window measurement

The residency studies established that **static hot-pin** is the cheap, effective
policy. The remaining question is whether a *dynamic* prefetch could beat it: is
the interval between **"router output for layer L is known"** and **"layer L's
expert weights are needed"** longer than the time to page an expert in? If the
window is shorter than the page-in, predictive prefetch cannot hide the latency —
**regardless of prediction quality** — and static residency is the ceiling for what
can be kept warm. Both this workstream and the mechanism agent's independently
identified this as *the* external number gating the storage-to-device family.

Measured on **granite-3.0-1b-a400m-instruct** (32 experts, top-8, 24 layers, real
IBM router), **onnxruntime 1.27.0 CPU EP**, this box **i7-13800H (14C/20T), RTX 4060
8 GB, WDDM**. Reproduce: `python scripts/moe_prediction_window_probe.py`.

---

## Two windows, measured separately

### Intra-layer window ≈ 0 — structural, EP-independent (MEASURED topology)

Graph topology: each layer's router `TopK` selected-expert indices feed **32 `Equal`
ops** (a per-expert one-hot mask, in this dense-f16 build) whose immediate
successors are the expert MatMuls. **The expert computation is the direct consumer
of routing, with no intervening compute.** So the window between "experts chosen"
and "expert weights read" is **~0 on any hardware** — a data dependency, not a
timing artefact. *Demand-prefetch of a layer's own experts can hide nothing.* This
is the robust, hardware-independent conclusion.

(Aside, consistent with earlier findings: the dense-f16 build computes **all 32**
experts and masks them, so it has no per-expert *load* to exercise — granite is a
routing-*selection* fixture, not a paging fixture. The intra-layer-window conclusion
comes from graph structure, which is the same for a sparse/offloaded build.)

### Cross-layer window — exists, but gated on prediction, and small on the target regime

The only window with lead time is **cross-layer**: prefetch layer L's experts during
the compute of layers < L. Measured upper bound = one layer's compute:

| quantity | measured (CPU EP, this box) |
|---|---|
| decode step (1 token) | **188 ms** median (20 steps, min 177 / max 208) |
| per layer (÷24) | **7.83 ms** |

Naively 7.83 ms exceeds a 1.5 ms PCIe page-in — but **this number is not the target
window**, for two reasons:

1. **It requires prediction that does not exist.** Layer L's experts are unknown
   until L's router runs, which needs L's attention output, which needs L−1 complete.
   Using the cross-layer window means *predicting* L's routing before L runs.
   MoE routing is data-dependent, and cross-layer correlation was measured to be
   **weak / second-order** (see the router-skew record). The window is real; the
   prediction to exploit it is not.
2. **CPU dense-f16 wildly overstates it.** This 7.83 ms/layer computes all 32
   experts on CPU. On the **GPU sparse-offload target regime** the whole 188 ms/token
   collapses to a few ms, putting the per-layer window **below a single page-in**
   (PCIe 1.5 ms or NVMe 3.7 ms per 16 MiB expert — mechanism agent, measured on this
   box). The CPU cross-layer figure must not be read as the deployable window.

---

## Verdict (MEASURED structure + cross-referenced page-in)

**Dynamic demand-prefetch cannot beat the established static hot-pin.** The only
experts that are prefetchable with both certainty and lead time are the **always-on
core** — predictable with zero lookahead — and the static pin **already keeps those
resident.** For everything else:

- a layer's **own** experts have a **~0 window** (router feeds compute directly), so
  no mechanism, however fast, can prefetch them in time; and
- **future-layer** experts need **prediction the routing does not support** (weak
  cross-layer correlation), and even the generous CPU window shrinks below a single
  page-in on GPU.

This also settles the cold tier during decode: a **cold-NVMe expert (~3.7 ms)**
exceeds any realistic per-layer decode window, so **cold-tier fetches are not viable
in the decode critical path** — they must be hidden by residency (VRAM/DRAM), not by
prefetch. This matches the mechanism agent's independent conclusion that the SSD is a
cold-miss backstop, not a hot-path source.

**Implication for the family:** the lever ordering stands — quantise (if f16) →
keep resident (static pin + DRAM tier) → batch → schedule. **Dynamic prefetch is not
a lever here**; the window forecloses it. The one thing that *would* change this is a
genuinely predictable router (strong cross-layer or cross-token correlation), which
this model does not exhibit and which our trace already measured as second-order.

---

## What this establishes / does not
- **Measured:** the router→expert graph topology (intra-layer window ≈ 0) and the
  CPU per-token / per-layer decode wall-clock on this box.
- **Cross-referenced (measured elsewhere):** the page-in costs (PCIe 1.5 ms, NVMe
  3.7 ms per 16 MiB expert — mechanism agent, same box).
- **Inferred:** the GPU-regime window (the CPU dense-f16 per-layer time overstates
  it; the direction — "below a page-in on GPU" — is reasoned, not measured here),
  and generalisation to a 256-expert router.
- **Not established:** a native-CUDA sparse-decode per-layer time (would need the
  native EP with per-op timing on a runnable MoE; granite's dense-f16 CPU build is
  not that). The intra-layer ≈ 0 conclusion does not depend on it; the cross-layer
  magnitude claim is where a native measurement would tighten the bound.
