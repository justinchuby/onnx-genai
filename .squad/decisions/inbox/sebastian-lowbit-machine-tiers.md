### 2026-08-13: Lower-bit quant is DEVICE-DEPENDENT — H200 NO-GO, but a real speed+fit lever on consumer/edge

**By:** Sebastian (Performance/Systems). Extends `docs/research/lowbit-quant-feasibility.md` (new §6 "Machine-class sensitivity"). Branch `squad/lowbit-machine-tiers`.

**One-line verdict:** lower-bit quant is 🟥 NO-GO **only on the H200/datacenter tier** (measured latency-bound); on bandwidth-starved consumer/edge GPUs it is 🟢/🟡 a real **speed** lever (below the ~0.7 TB/s crossover) **and** the only way to **fit** a 30B model at all (int4 ≈ 15 GB won't load ≤12 GB) — so KEEP IT ON THE ROADMAP for that tier.

**Why the earlier blanket NO-GO was H200-specific.** The byte-fold probe (§5) measured *this box* (H200, ~4.8 TB/s). Weight reads there are almost entirely *hidden* behind the serial ~2568-node launch-latency chain (~8.2 µs/node × 2568 ≈ 21 ms/token), so cutting bytes buys ~+3% max. That is a property of H200's huge bandwidth, not a universal truth.

**Two-component model (per token):** `T_latency` (dispatch/launch chain, ~bandwidth-independent, ~21 ms on H200) and `T_weightread = 15.3 GB / B_device`. Per-token ≈ `max(...)` in the overlapped limit. On H200 `T_weightread` = 3.19 ms is hidden under `T_latency` → latency-bound → byte cuts ~free-ride. As `B_device` drops, `T_weightread` grows and eventually dominates → byte-fold slope steepens → lowbit pays off. **Crossover ≈ 15.3 GB / 21 ms ≈ 0.73 TB/s** (EXTRAPOLATION from one device — model, not measurement; no fabricated cross-device tok/s).

**Device tiers (spec-sheet ranges, not benchmarks I ran):**
| tier | example | mem BW | VRAM | fits 15.3 GB int4? | regime | lowbit value |
|---|---|---|---|---|---|---|
| Datacenter | H200/H100 | 3.3–4.8 TB/s | 80–141 GB | ✅ | latency-bound (MEASURED) | 🟥 useless for speed |
| High-end consumer | RTX 4090/5090 | 1.0–1.8 TB/s | 24–32 GB | ✅ | near crossover | 🟡 modest speed |
| Mid consumer | RTX 4060/4070 | 270–500 GB/s | 8–12 GB | ⚠️ often won't fit | bandwidth-bound | 🟢 speed + fit |
| Laptop/iGPU/Jetson/edge | Orin, iGPU | 100–270 GB/s | ≤8 GB | ❌ | strongly BW-bound | 🟢 required to run at all |

**Two independent values of lowbit (do not conflate):** (1) **speed** — only in the bandwidth-bound regime (mid-consumer and below); (2) **fit-ability** — 30B int4 ≈ 15 GB won't load ≤12 GB; int3 (~11.5 GB)/int2 (~7.7 GB) makes it *run*. Fit is a portability win independent of the speed roofline and may be the stronger motivation ("runs vs doesn't run").

**Device-conditioned recommendation:**
- **H200/datacenter:** 🟥 NO-GO for speed → the lever is the **decode megakernel / node-collapse** (latency-bound; still the true datacenter lever).
- **Consumer/edge:** 🟢/🟡 KEEP ON ROADMAP. **Concrete gate before investing:** we only have an H200 and CANNOT measure this regime here — the next validation step is to run the SAME `ONNX_GENAI_WEIGHT_FOLD` byte-fold probe on a representative consumer GPU (RTX 4070 ~500 GB/s, and a ≤8 GB laptop dGPU). Steep slope there ⇒ GO for that tier.
- **Accuracy path is device-independent** (Fact Checker): int3 / ~3.5 bpw imatrix / SpQR 🟢; int2 needs codebook/trellis 🟡; scalar int2 🔴; **all** require re-quant from the fp16 source (not staged) + new sub-4-bit kernels (int3 non-byte-aligned M–L; int2 clean S–M).

No code change. Docs only.
