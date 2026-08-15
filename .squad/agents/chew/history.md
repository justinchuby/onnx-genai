# Chew — History (compacted 2026-08-12)

**Role:** Numerics/precision reviewer. Require reference-backed coherent outputs rather than mere execution success, and guard dtype/layout symmetry, silent coercions, opset semantics, broadcast behavior, stable reductions/softmax, and realistic parity tests.

## Durable lessons
- Review standard: coherent reference-backed outputs beat successful execution; dtype/layout symmetry, opset semantics, broadcast, stable reductions/softmax, and realistic parity tests are mandatory.
- Connector/KV work must preserve cache separation, byte-layout symmetry, prefix-dependent hashing, fetch/recompute boundaries, per-layer heterogeneous geometry, and graceful recompute fallback.
- Original contrib FusedMatMul shape rule ignored transpose attributes; Chew rejected it and Deckard's corrected rule is canonical.
- ONNX dtype decoding must fail closed; never silently fall back to Float32.
- Fusion tolerances are distinct from conformance tolerances and must not be loosened; LayerNorm needs axis-as-input, epsilon-type, and operand-order decline guards.
- EPContext cannot fall through to CPU execution; payloads remain byte-exact, FFI is null/UTF-8/panic guarded, and disabled export must be side-effect-free.
- CSA B5's five-output ratio-4 dispatch bug was a real misroute to ratio-128; Roy's ratio-keyed fix/regression is canonical.
- CPU reduction axes semantics distinguish omitted axes from present-empty axes; Deckard's fix after Chew rejection is canonical.
- Fused QMoE must not clobber `_group_topk_selection`; grouped routing requires original signature and group-mask behavior.
- CUDA graph capture reviews require real replay coverage, exact signatures, detect-before-consume poisoning, and guard-break proofs, not just smoke success.
- WP-B loader-IR shape authority rejection directly informed Sapper's final WP-B3 v3 fix.
- PR #227 SiLU polynomial measured ~28 ULP, not ~1 ULP; the docstring claim was wrong though numerics were acceptable for inference. NEON SDPA dispatch had zero path coverage until follow-up.
- PR #334 formatting failures are review-blocking even when numerics are sound; Iran was the revision agent after rejection.
- BNNS grouped/depthwise convolution via deprecated API is genuinely broken for groups > 1; guard is justified, but im2col is only a correct intermediate step and direct NEON depthwise should target 2–3× ORT.
- Documentation rationales are correctness artifacts: wrong L1-cache premises and derived-looking fitted constants must be corrected before merge.
- A reviewer's "SAFE" is not proof; verify the load-bearing claim. Sebastian cleared n%8≠0 as safe on #31988; the coordinator's reading found n=12 would have been newly accepted, changing routing.
- An occupancy change must not become a routing change. Exhaustive pinning test (`SelectColsPerBlock_OnlyMod8Accepted`, n∈[1..256] × 5 SM counts) prevents accepted-shape-set drift.
- Never validate with `--compile_no_warning_as_error`; it masks the exact class of failure that blocks upstream CI.
- Leak scans must cover committed C++ source comments, not just `.squad/` paths — two agent names sat in public upstream PR #31973 comments, forcing a third history rewrite.

_Pre-2026-08-11 dated entries archived to `history-archive.md`. 2026-08-11 detailed entries also archived there._

## 2026-08-11 — PR #31988 routing guard + template cost docs

- **Finding:** SelectColsPerBlock would expand accepted shapes (n%4, n%2) beyond upstream's n%8 requirement. Shape routing DOES change without a guard.
- **Fix:** Added `n % kColsPerThreadBlock != 0 → return false` before M=1 path. Option (a) — no routing change.
- **Template cost:** Documented 24→72 instantiations, ~38 KB binary, all reachable.
- **Tests:** Added `SelectColsPerBlock_OnlyMod8Accepted` and `SelectColsPerBlock_RoutingInvariance_NMod8Required`.
- **Commit:** a4aa076657, pushed to `nxrt/cuda-matmulnbits-sm-cols`. PR stays draft.
- **No perf numbers, no leaks confirmed.**

## 2026-08-12 — PR #31988 routing guard (lockout revision)

- **Finding:** Sebastian cleared `n % 8 != 0` as SAFE — incorrect. `SelectColsPerBlock` returning 4 for n=12 (12%4==0) would have been newly accepted, changing shape routing beyond occupancy.
- **Fix:** Added `n % kColsPerThreadBlock != 0 → return false` before M=1 GEMV call. Accepted-shape set now identical to upstream.
- **Tests:** `SelectColsPerBlock_OnlyMod8Accepted` (exhaustive n∈[1..256] × 5 SM counts) and `SelectColsPerBlock_RoutingInvariance_NMod8Required`.
- **Template cost documented:** 24 → 72 instantiations, ~38 KB, all reachable.
- **Commit:** a4aa076657. Chew barred Sebastian from revising his own rejected finding.

### 2026-08-12 — PR #31974 coverage strengthening

- **Task:** Add PrePack and generic-broadcast BF16 test coverage; remove internal labels; fix comments.
- **Tests added:** `LayerNorm17_PrePack_ScaleBiasInitializers`, `SkipLayerNorm_PrePack_GammaBetaInitializers`, `LayerNorm17_GenericBroadcast`.
- **Coverage:** 17 → 20 BF16 tests, 103 → 106 total LayerNorm suite. All pass.
- **Internal labels removed:** 2× "B5" in test comments.
- **SrcDispatcher comment fixed:** now accurately describes `if constexpr` preventing `ComputeImpl<NarrowType, NarrowType>` instantiation.
- **Tolerance comments fixed:** aligned with actual checker semantics (`tolerance = absolute + relative * |expected|`, numpy.isclose style).
- **Commit:** `a12c7ddde3` on `nxrt/mlas-bf16-layernorm`. PR stays draft.

## 2026-08-12 — PR #31974 Opus v2 coverage wave

- **Task:** Add PrePack and generic-broadcast BF16 test coverage; remove internal labels; fix comments.
- **Tests added:** `LayerNorm17_PrePack_ScaleBiasInitializers` (`is_initializer=true`, lines 717-718), `SkipLayerNorm_PrePack_GammaBetaInitializers` (`is_initializer=true`, lines 752-753), `LayerNorm17_GenericBroadcast` (X={2,2,2}, scale={2,2} → `use_generic_broadcast=true`).
- **Coverage:** 17 → 20 BF16 tests, 103 → 106 total LayerNorm suite. All pass.
- **Hygiene:** Removed 2× "B5" internal labels; fixed SrcDispatcher comment; aligned tolerance comments with checker reality (`absolute + relative × |expected|`, numpy.isclose).
- **Commit:** `a12c7ddde3`. Delta review by Holden: no blockers. PR remains draft (vcpkg bootstrap TLS infra flake in CI).

## 2026-08-12/13 — CUDA capture arc: bf16 GQA kernel numerics accepted (shared: 11.4 → 23.13 tok/s)
Reviewed the accuracy gate on Sebastian's `gqa_decode_bf16` (**#855**, `1022b912`):
parity vs an f64-accumulated softmax oracle fed bf16-rounded inputs, fp32 accumulation
preserved (bf16 only at load/store), measured max_abs=1.953e-3 / max_rel=3.888e-3 within
justified bounds (abs<2e-2, rel<1e-1). Byte-exact greedy parity. Part of the 5-blocker
chain that took Muse-Glimmer native decode **11.4 → 23.13 tok/s** (capture fully engaged).
Reinforced rule: bf16 kernels accumulate in fp32, oracle-gate against f64.

## 2026-08-12/13 — PR #860 numerics gate 🟢: parallel reduction is *more* accurate (CUDA goal MET)
Gated Sebastian's RMSNorm cast-fold + parallel bf16 tree reduction. Verified fp32
accumulation airtight, op-swap execution-identical (same `RmsNormFactory→RmsNormKernel`),
independent f64 oracle 4/4 (≤1 bf16 ulp). **Key finding:** tree reduction is **~807× MORE
accurate** than the old serial order (tree_err 2.07e-8 vs serial 1.67e-5 vs f64 truth). The
~37-token drift is downstream int4-quant greedy sensitivity, not a norm regression. Part of
the arc taking native CUDA decode **11.4 → 40.21 tok/s** — goal MET (matches ORT ~40 tok/s).
Rule reinforced: a parallel tree reduction may replace a serial order when the f64 oracle
shows it is at least as accurate.

## 2026-08-13 — PR #871 numerics gate 🟢: bf16 decomposed SiLU is 0-ulp byte-exact
Gated Sebastian's bf16 decomposed SiLU/SiLU-Mul kernels (#871). Verified fp32 accumulation
airtight, bf16 only at load/store; **byte-exact, 0 ulp bit-identical** vs the unfused two-op
graph and an f64 oracle; 5/5 silu tests on H200. Confirmed it fixes a hard-crash portability
defect (bf16 decomposed SiLU previously errored `"requires float16"`). No further numerics-gated
fusion pursued: the fusion arc concluded that **native int4 decode of Muse-Glimmer-30B is
weight-bandwidth/compute-floor bound at ~47.25 tok/s (H200)** (the architectural ceiling), so
node/launch fusion — cheap or expensive — cannot help. Rule reinforced: bf16 accumulate in fp32,
oracle-gate against f64.

## 2026-08-14 — PR #960 Marlin int4 M>1 GEMM numerics review 🟡 APPROVE-WITH-NOTES
Reviewed Deckard's Marlin fp16×int4 tensor-core GEMM (numerics only; Gaff owns quality). Verified by close
read of the mma B-fragment lane mapping + repack (host `repack_int4_weights` == device repack, byte-identical,
all group sizes), the **scale-after-accumulate** factoring (exact: scale/zp constant across K within a group),
asymmetric nibble zp indexing, padding-column guard, split-K disjoint fp32 partials (deterministic fixed-order
reduce ⇒ capture-stable), and the lm_head cached dense-GEMM plan (cuBLASLt heuristic deterministic for fixed
shape ⇒ cached algo == dynamic algo). Ran the in-crate GPU parity suite on H200: **11/11 pass** incl.
`marlin_parity_vs_f64_oracle`, `marlin_splitk_parity_vs_f64_oracle`, `marlin_splitk_is_deterministic`,
`repack_roundtrip_all_group_sizes`. Confirmed Pris's #961 f64 tolerance is engineering-justified (~8× fp16-ULP
headroom), not a rubber stamp. **No correctness bug — ship.** Three non-blocking notes: (N1) "byte-identical
greedy tokens" is a soft argmax-stability guarantee (empirical over 2 models × 24 tokens), NOT a numeric
invariant — a near-tie on an untested prompt could flip; keep the flag opt-in. (N2) hard Marlin launch errors
are silently swallowed into the tiled fallback (`Err(_) => fall through`) — numerically safe but log/count it
so a real kernel fault can't hide behind the slow path. (N3) Marlin assumes nibble-packed int zero-points
(matches every existing int4 tiled kernel — not a regression; guard dispatch if float-zp int4 is ever added).
Rule reinforced: a relayout that reorders partial sums cannot be diffed bit-for-bit; the only defensible
reference is a high-precision f64 oracle with a justified tolerance.
