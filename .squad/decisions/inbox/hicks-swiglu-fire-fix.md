### 2026-07-26: Make the SwiGLU Silu+Mul→SiluMul fusion actually fire on the native CPU decode path (PR #171)
**By:** Hicks

**OPTIMIZATION-LEVEL VERDICT (read this first):** The native CPU decode path runs
**NO device-independent session-level fusions by default.** `OptimizationLevel`
defaults to `None` (`crates/onnx-runtime-session/src/lib.rs:383-395`), and the
native decode session builders
(`crates/onnx-genai-engine/src/native_decode.rs:623`, used by
`Engine::from_dir` → `load_with_weight_offload_host_cache`) never set the
`"optimization"` option. So the session-level `OpFusion` pass — which owns the
`Silu+Mul→SiluMul` pattern (`fusion.rs:1382`) and every other fusion
(LayerNorm/GELU/attention/MatMul+Bias) — is a pure no-op on decode. Roy's PR #171
was correct and Hudson-approved but delivered **zero** gain because the pattern
was registered in a pass set that decode never runs. This is a whole-strategy
finding: any future perf fusion added only to the session-level `OpFusion` will
also silently never fire on native decode.

**What actually runs on every native decode build (regardless of `"optimization"`):**
`crates/onnx-runtime-session/src/executor.rs:2255-2257`
1. `fuse_silu_patterns` — normalizes `x*Sigmoid(x)` → `com.microsoft::Silu`.
2. `run_ep_scoped_passes` → the CPU EP's `cpu_optimization_passes()`
   (`crates/onnx-runtime-ep-cpu/src/optimizer.rs`): ProjectionFusion (env-gated),
   MatMulNBitsBiasFusion, ConvBN, NCHWc. These are **always-on, EP-scoped**.

**Fix (minimal, sound, CPU-scoped, token-exact):** register the `Silu+Mul→SiluMul`
fusion as an always-on pass in `cpu_optimization_passes()` (after ProjectionFusion,
which anchors on the `Silu` node, and after `fuse_silu_patterns` produces the
`Silu`). It fuses **only** Silu+Mul (bit-identical, single-consumer guarded) — it
does **not** enable the broad session-level `All`, so no other numerics-changing
fusion is pulled in. **CUDA is untouched**: CUDA has its own EP-level SwiGLU fusion
(`crates/onnx-runtime-ep-cuda/src/optimizer.rs`) and no `com.microsoft::SiluMul`
kernel, so gating this to the CPU EP avoids emitting an op CUDA can't dispatch.

**Why not just set `optimization="all"` on the decode builder?** It would (a) also
run LayerNorm/GELU/attention/MatMul+Bias fusions (broader numeric risk than this PR
should take), and (b) apply to the CUDA decode path too, which would emit a
CPU-domain `SiluMul` the CUDA EP cannot run. The EP-scoped pass is strictly safer.

**Verification (Qwen3-0.6B generic-cpu-4/v4, prompt "Hello", 64 tokens, CPU, greedy):**
- Before (main behavior, fusion off): `ONNX_GENAI_PROFILE_OPS=1` shows separate
  `Silu` + `Mul` per layer, no `SiluMul`. Decode ≈ 23.795 ms/token.
- After: profiler shows `SiluMul` present; **standalone `Silu` = 0, `Mul` = 0**.
  Decode ≈ 23.899 ms/token.
- **Token IDs identical** between the two builds (first ids
  `[1479, 271, 40, 1184, 311, 1855, 264, 2025, 429, 646, 387, 1483, …]`) — the
  fusion is token-exact, confirming Hudson's bit-identical review end-to-end.
- Rough delta is within single-run noise (≈ ±0.1 ms/token, uncontended host). The
  fusion now FIRES; the rigorous median A/B is the coordinator's to run.

**Tests updated:** three `crates/onnx-runtime-session/tests/projection_fusion.rs`
cases asserted node/`Silu` counts on SwiGLU fixtures; they now account for the
always-on `SiluMul` fold (Silu→0, SiluMul→1, node count −1). Projection-fusion
invariants (MatMulNBits/Split) are unchanged.

**Green:** `onnx-runtime-ep-cpu` (--features mlas, 934+ tests),
`onnx-runtime-session` (--features mlas), `onnx-runtime-optimizer`. Native clippy
`-D warnings` clean (ep-cpu + session); ARM aarch64 cross clippy (no mlas) clean.

**Follow-up (not this PR):** consider wiring the native decode path to opt into the
device-independent fusion pipeline (or promote other decode-critical fusions to the
EP-scoped set) so future session-level fusions aren't silently dead on decode.
