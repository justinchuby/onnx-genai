# Review: perf/phi-standalone-int8-splitk (HEAD ff1ac51)

**Reviewer:** Holden (worker) — read-only critique, reviewer lockout (did NOT modify Deckard's code)
**Date:** 2026-07-23
**Change:** within-block split-K on the STANDALONE int8-zp decode GEMV to fix grid-starvation on Phi's down-proj.

## VERDICT: 🟢 — correct by construction (separate gated entry), tolerance relaxation is legitimate fp32 reassociation, gate 192/0 + clippy clean.

## 1. Routing correctness / byte-identity by construction
- Split-K is a **separate NVRTC entry** `matmul_nbits_gemv_int8_f16_splitk`, selected at runtime in `launch_int8_f16_gemv` — only reachable for **bits==8, block_size==32, M==1 decode** (M>1 → tiled GEMM; int4 → different dispatch entirely).
- Gate: `use_splitk = zero_points.is_some() && k % 256 == 0 && !(n <= 1152 && k <= 1152)`.
- Non-routed cases (symmetric int8 / no zp, small shapes, K not %256) fall through to the **original single-warp `matmul_nbits_gemv_int8_f16`, unchanged** → byte-identical **by construction** (separate entry, no shared codegen touched).
- **Which models route to `_splitk`:** only int8-zp, block-32, M==1 decode GEMVs with K%256==0 and non-small shape. In the current model set that is **Phi** (down-proj K=8192, and its other large int8-zp projections e.g. K=3072). The gate is shape-general, but no other current model has int8-zp weights that reach it.
- **Qwen/DeepSeek/GLM:** these use int4 (block-128) → never enter the int8 f16 GEMV dispatch (`self.bits == 8` guard) → byte-identical. Deckard's Qwen 7B 294.22 vs 293.97 is genuine noise: **Qwen has no int8 path routing here at all.** Confirmed.

## 2. Tolerance relaxation — SCRUTINIZED: LEGITIMATE, not masking
- Relaxation is scoped to exactly `bits == 8 && explicit_zp && m == 1`; every other case keeps the strict `assert_eq!` bit-identity check.
- Real cause: the three-op reference's standalone GEMV (K=3072, N=5120, zp, K%256==0) now genuinely routes to split-K, which reassociates the fp32 block-sum across 2 cooperating warps, while the fused following kernel keeps the single-warp association. That is real fp reassociation → near-equal, not a masked bug.
- New bound `max(max_abs * 2e-3, 1e-3)` = 0.2% relative + 1e-3 abs floor. Tight enough: fp16 output ULP ~5e-4; a genuine K-partition bug (dropped/double-counted block, 1 of 96) yields ~10% column error — two orders of magnitude above the bound → would be caught. Also asserts `is_finite`.
- Independent corroboration: split-K is validated against an **f64 dequant oracle** at `max_rel <= 1e-2` via the new `(8192, 3072, true, false, true)` case in `int8_fp16_gemv_matches_dequant_reference_phi_dims` — passed. So correctness is anchored to a true reference, not only to the single-warp kernel.

## 3. Correctness math
- K partition: each warp handles 8 blocks/iter (`lane>>2` ∈ 0..7); `block_base` starts at `ks*8`, stride `K_SPLIT*8 = 16`. ks=0 → block groups [0,8),[16,24)…; ks=1 → [8,16),[24,32)… → **complete cover, no gaps/overlaps** for K%256==0.
- Within a block, `quarter*8` (quarter 0..3) covers all 32 elements; 4-lane partial reduced via width-4 `__shfl_down_sync`.
- Per-block contribution accumulated only by quarter-0 lanes into `value`, then `warp_sum` over 32 lanes; `partials[col_local][ks]` written by lane 0 of each warp into **disjoint slots**; a **single `__syncthreads`** separates all writes from the `ks==0 && lane==0` read → **race-free**.
- Fast path `depth + 8 <= k` is always taken (K%256==0 ⇒ every block fully in-range); tail branch is dead for routed shapes. Launch geometry consistent: split-K always on the 256-thread large path (8 warps), `columns_per_block = 4`, grid 2× — Rust and kernel agree.

## 4. Independent gate run (worktree off ff1ac51, CUDA_VISIBLE_DEVICES=7)
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **192 passed; 0 failed** (incl. `fused_skip_rmsnorm_int8_asymmetric_zp_...` and `int8_fp16_gemv_matches_dequant_reference_phi_dims`).
- `cargo clippy ... --lib -- -D warnings` → **clean, 0 warnings**.
- profile_native Phi coherence spot-check: skipped — no local Phi int8 ONNX model available; the f64-oracle parity test provides equivalent numerical-coherence assurance.

## Nit (non-blocking, no fix required under lockout)
- "Only Phi" is true for the current roster but the gate is shape-general; any future int8-zp block-32 model with K%256==0 large shapes will also route here. That's fine (split-K is numerically validated), just worth noting so the routing comment isn't read as a hard Phi-only guarantee.
