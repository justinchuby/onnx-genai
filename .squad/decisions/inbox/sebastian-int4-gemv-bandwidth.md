### 2026-08-14 — int4 decode GEMV bandwidth rewrite: split-K & cp.async NO-GO, fold-scale +2.7% opt-in (pending Chew)

**By:** Sebastian (Performance/Systems), measured on H200 (`CUDA_VISIBLE_DEVICES` pinned to a verified-idle high index, GPU 7). Numerics gate: Chew.

**Context / reframe:** The prior "47 tok/s = launch-amortized floor" arc (#898/#900/#903/#916) attacked
launch/dispatch overhead, which graph replay already amortizes. That framing was **incomplete**: batch-1
decode is HBM-bandwidth-bound (~15.37 GB int4 ÷ 4.8 TB/s ≈ 3.2 ms/tok ⇒ ~300 tok/s roofline; ~100–180
realistic). A measurement turn ran `ncu` on the dominant int4 GEMV and found it sustains only **~29% of
peak DRAM** — a kernel-efficiency floor, not a hardware floor. GO to rewrite. This is the rewrite result.

**What:** Three phased levers, each BUILT & MEASURED on Muse-Glimmer-30B (dense int4, 52L, H=6656):
- **Phase A — higher-way split-K (2→4→8):** NO-GO. K=2 47.3 / K=4 47.8 (+1%, noise, DRAM *fell* 29.4→27.75,
  dominant kernel *slower* 56→59.4 µs) / K=8 45.4 (regression). Occupancy already ~91%, Not-Selected 18.6%
  — more warps don't reduce per-warp load latency.
- **Phase B — cp.async double-buffered weight loads (SM80+ guarded, sync fallback):** NO-GO. K=2 41.2
  (−13%), K=4 41.0. The 4 B/lane cp.async granularity is too small; commit/wait + shared round-trip cost
  more than the latency hidden. A real win needs 16 B/`.cg` async over a Marlin-style tiled relayout
  (from-scratch kernel, out of scope).
- **Phase C — fold per-block scale into the LOP3 dequant** (drop 4 `__hmul2` per 8 weights;
  `fma(code,scale,-zp·scale)`): **the only kernel-level win, +2.7%.** The dequant was *already*
  Marlin/LOP3, which is why the ALU pipe is at 65%.

**Numerics (Chew's gate):** Fold-scale on the **real model** greedy 128-token stream is **byte-identical**
to baseline. BUT it is not byte-exact per element (fused fma sums two fp16-rounded terms vs plain exact
`(code-zp)` then one multiply) and **fails** the existing synthetic asymmetric-zp parity guard
`fp16_gemv_matches_dequant_reference_phi_int4_zp_dims`: **worst rel 0.104 vs 5e-2 bound** (max-abs
1.19e-2 was *within* the 2.55e-2 abs bound — only near-zero output columns fail the relative check).
Plain split-K passes the guard.

**Measured (H200 GPU 7, steady, warmups=2 runs=5, `ONNX_GENAI_CUDA_GRAPH=1`):** baseline ~47.6–47.7;
fold-scale K=2 **48.9–49.0 (+2.7%)**; split-K K4=47.8 / K8=45.4; cp.async K2=41.2 / K4=41.0. ncu on
fold-scale: dominant kernel 56→53.4 µs (−5%), DRAM 29.4→30.9%, Long-Scoreboard relatively rose (ALU
relief shifts balance back to loads — confirms the mechanism).

**Kernel↔token reconciliation (Roper's honesty check):** kernel −4.6% (56→53.4 µs) over the ~61%
GEMV fraction predicts `1/(0.39+0.61×0.954)=+2.9%` end-to-end; **measured +2.7%** — the token gain is
**fully Amdahl-explained, no hidden serial-dispatch floor** (graph replay already amortized launches).
This is the *benign inverse* of the `lowbit-quant-feasibility.md` byte-fold probe (−75% bytes → +2.8%,
because bytes weren't the binding): here the kernel genuinely improved and the token moved by exactly the
predicted amount — real but small. The hope that cp.async would beat +2.8% by hiding the 40%
Long-Scoreboard did NOT hold (cp.async −13%, split-K flat); the current weight layout defeats those levers.

**Why:** The GEMV is **co-bound** (40.7% Long-Scoreboard load-latency AND 64.8% dequant-ALU), not purely
latency-bound, so the textbook latency-hiding levers (split-K, small-granularity cp.async) can't raise
achieved DRAM. The hoped 1.3–1.6× (60–75 tok/s) did **not** materialize — the existing GEMV is already
near its efficient design point. ~39% of decode is non-GEMV (Amdahl-caps end-to-end gain regardless).

**Verdict:** Ship fold-scale **opt-in, default OFF** (`ONNX_GENAI_GEMV_FOLDSCALE=1`). Production + CI stay
on the exact plain split-K path (parity guard green). **SHIP-pending-Chew** on whether the +2.7% justifies
flipping the default given the per-element accuracy cost. Split-K & cp.async removed (dead-code-free
minimal diff). Bigger single-GPU wins need a from-scratch Marlin int4 kernel (multi-week); higher-leverage
next steps are speculative decoding / tensor-parallel, which stack on top.

**Validation:** `cargo fmt --all -- --check` clean; clippy clean on the changed crate; ep-cuda
`matmul_nbits` suite 24/24 pass with default OFF (the previously-failing parity guard is GREEN); CUDA EP
cdylib does NOT link libonnxruntime (ldd-confirmed); fold-scale template is arch-agnostic (no cp.async) so
no portability regression. Do NOT self-merge — Chew gates numerics.
