# Decision: GEMV latency lever — symmetric int8 MatMulNBits split-K grid-fill

- **Agent:** Gaff (CUDA-kernel specialist)
- **Branch:** `squad/gemv-bias-fusion` (PR: symmetric int8 GEMV split-K)
- **Date:** 2026-08-19
- **Verdict:** ✅ **Clean win, shipped behind an opt-out.** +9.8 tok/s (254.0 → 263.8),
  both golden token-ID locks PASS (0.8b + 2b), ncu confirms the grid-starved
  int8 GEMVs' occupancy roughly doubled and per-call ns dropped 20–30%.

## Sub-lever 1 (bias-epilogue fusion): NO-OP here — reported honestly

Deckard's attribution said "ORT fuses bias into the GEMV epilogue; native runs a
SEPARATE add kernel." Ground-truth check on the actual exports:

- Bias-epilogue fusion **already exists** on origin/main: the
  `CudaMatMulNBitsBiasFusion` optimizer pass folds a standalone `Add` into the
  MatMulNBits bias slot, and the GEMV kernels already carry a fused `fold_bias_f16`
  epilogue (native and post-round modes).
- **These Qwen3.5 exports are bias-free**: `0 / 187` MatMulNBits carry a bias
  input, and **no `Add` node consumes a MatMulNBits output** on qwen3.5-0.8b,
  qwen3.5-2b, or qwen3.5-0.8b-text-cuda. Qwen3 projections have no bias.

So there was no separate add kernel to fuse — the sub-lever is a no-op for this
model and already-solved for models that do have bias. No code shipped for it.

## Sub-lever 2 (occupancy/split-K): the real, buildable win

### Ground truth (ncu `--graph-profiling node`, H200, qwen3.5-0.8b fp16io)

The model is **hybrid: 144 int8 + 43 int4 MatMulNBits, all symmetric** (no
zero-points). Per-shape decode GEMV profile:

| kernel / shape | grid | ns/call | occ% | note |
|---|---|---|---|---|
| int8 N=1024 (down/o, 36/step) | 128 | **11424** | **11.6** | #1 offender, grid-starved |
| int8 N=2048 (18/step) | 256 | 6065 | 22.2 | grid-starved |
| int8 N=3584 (36/step) | 448 | 6318 | 38.4 | moderately occupied |
| int8 N=6144 (18/step) | 768 | 6971 | 62.8 | occupied |
| int8 N=16 (36/step) | 8 | 5079 | 3.1 | tiny, 64-thread path |
| int4 N=248320 (LM head, 1/step) | 31040 | 85600 | 87.1 | already near-roofline |

The int4 path already had the full treatment (symmetric split-K
`use_f16_symmetric_splitk`, occupancy-gated pipeline #1501, wide/multicol). But
the **int8 split-K was gated on `zero_points.is_some()`** — symmetric int8 was
deliberately pinned to the single-warp kernel to stay byte-identical. That left
the dominant int8 projections grid-starved.

### The change

Relax the int8 GEMV split-K gate: symmetric int8 now takes the (pre-existing,
symmetric-safe) `matmul_nbits_gemv_int8_f16_splitk` kernel when the shape is
grid-starved, reusing the **same** `use_f16_symmetric_splitk` predicate the int4
path uses (fire when `N < mp_count * 16`; keep the `K % 256 == 0` and non-small
constraints). Device-derived, no hardcoded dims. Opt-out:
`ONNX_GENAI_CUDA_DISABLE_INT8_SYMMETRIC_SPLITK=1`.

### After (ncu, same setup)

| shape | before | after |
|---|---|---|
| int8 N=1024 | grid 128, 11424 ns, 11.6% occ | **splitk grid 256, 8049 ns, 23.4% occ** (−30%) |
| int8 N=2048 | grid 256, 6065 ns, 22.2% occ | **splitk grid 512, 4877 ns, 45.4% occ** (−20%) |
| int8 N=3584 | 6318 ns, 38.4% | unchanged ✓ (excluded, occupied) |
| int8 N=6144 | 6971 ns, 62.8% | unchanged ✓ |
| int8 N=16 | 5079 ns | unchanged ✓ (small 64-thread path) |

Est. GEMV time saved ≈ 36×3.4µs + 18×1.2µs ≈ **~145µs/step**.

### End-to-end (profile_native, 128 tokens, paired interleaved A/B, GPU-pinned)

| pair | ON (tok/s) | OFF (tok/s) |
|---|---|---|
| 1 | 258.5 | 253.8 |
| 2 | 263.8 | 254.0 |
| 3 | 264.2 | 254.1 |
| **median** | **263.8** | **254.0** |

**+9.8 tok/s (+3.9%)**. The OFF arm spread is 0.3 tok/s (quiet box), so the ~10
tok/s gap is far outside noise. Matches Deckard's +8–12 est. Native 254 → ~264
narrows the ORT-eager (279) gap from 1.099× to ~1.057×.

## Correctness

- **Golden token-ID locks PASS on both models** (qwen3.5-0.8b @
  `/home/justinchu/qwen35-0.8b-text-cuda`, qwen3.5-2b): greedy stream unchanged.
  The lock asserts the greedy Vec<u32> argmax sequence, not bit-exact bytes.
- The split-K kernel reassociates the per-column fp32 partial sums (K_SPLIT=2),
  so output is **near-equal, not byte-identical** — a ULP-level shift under
  preserved fp32 accumulation, argmax-stable. This is the identical trade
  already shipped for symmetric int4 split-K.
- New unit test `symmetric_int8_splitk_gate_targets_grid_starved_only` +
  existing 36 matmul_nbits routing/parity tests pass.

## Generality / capture-safety

- Keys only on `K`, `N`, `bits`, zero-point presence, and live SM count — no
  qwen-specific dims. Asymmetric int8, the small 64-thread path, GQA/dual-head
  geometry, and small consumer GPUs are all untouched.
- Launch-time constant gate (stable across CUDA-graph replays); the split-K
  kernel uses only static shared memory + warp shuffles, no host sync — already
  proven capture-safe on the asymmetric route.

## Recommendation to coordinator

Ship after independent lock re-validation. Remaining native-vs-ORT decode gap is
now dominated by the still-grid-starved N=16 gates (structurally hard: only 16
columns) and the moderately-occupied N=3584 int8 — both lower-yield. Future
GEMV levers should target K_SPLIT=4 on the very-starved N=1024 (8049 ns still at
23% occ) if a deeper split stays greedy-stable.
