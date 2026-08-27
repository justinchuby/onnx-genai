# Deckard — History (compacted 2026-08-18T20-40-00Z)

**Role:** Systems developer on the Rust runtime, ORT2 loader/shape/IR/EPContext tracks, and CPU/CUDA execution performance. Preserve model-agnostic dispatch, fail-closed claims, checked arithmetic, byte-exact serialization, precision-sensitive tests, and reviewer-lockout ownership transfers.

## Durable lessons
- Repeated invariants: model-agnostic dispatch, fail closed at claim time, checked arithmetic, byte-exact serialization, and precision-sensitive tests.
- Parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership and must be recorded.
- **ORT plugin-EP ABI:** `OrtMemoryInfo` passed to `EpDevice_AddAllocatorInfo` must outlive the `OrtEpDevice` — ORT stores the raw pointer; do not call `ReleaseMemoryInfo` on success. Use `CreateMemoryInfo_V2`. Release only on failure.
- **Shape inference fail-closed:** `Declined` is the correct return for any unmodelled op; never fall back to `SameAsInput(0)`.
- **`validate_dims` must be wired** in the actual read path, not just implemented.
- `OrtGraph*` / `OrtNode*` handles must NOT be stored beyond callback return.
- **Selective cache hints:** profile each caller of a shared helper separately before applying `__ldg` or occupancy hints uniformly; blanket application can regress one caller while helping another.

## Historical context

Pre-2026-08-10 entries in `history-archive.md` (shape inference overhaul, EP device lifetime UAF, clippy lint cleanup).

2026-08-10 ep-plugin-export wave archived in `history-archive.md` under "Archive batch 2026-08-10".

2026-08-11/12 ep-plugin-parity-cuda wave and upstream-CI correction chronicle archived in `history-archive.md` under "Archive batch 2026-08-14 (Scribe decode-levers)".

**2026-08-12 — 2026-08-15 archive batch (Scribe 2026-08-18T20-40-00Z):** CUDA capture arc COMPLETE (11.4→23.13 tok/s, #848); native speculative decode benchmarked/root-caused/KILLED (#932/#935); Lever B (capture-stable M=K verify) CLOSED as NO-GO, Marlin promoted primary (#948/#949, #957); Marlin fp16×int4 tensor-core GEMM LANDED & MERGED (#960, 7774ec5b, 41-seg→single-graph, B\*=8.76→2.16×, byte-identical); glm int4 decode second act (wide-load GEMV target, cp.async NO-GO).

## 2026-08-18T01:35Z — Assigned V2-Lite graph-capture unlock

- New implementation target: topology-gated capacity-policy fix for V2-Lite's additive attention-mask builder so CUDA graph capture can engage on MoE without regressing GLM-5.2 logical-width masks.
- Expected surface is executor geometry/build tests plus capture eligibility, not CUDA kernels; Rachael review and Wallace byte-identity/perf validation are required.

## 2026-08-18T03:15Z — V2-Lite classifier lockout episode recorded

- Authored the first V2-Lite additive-mask capacity classifier for CUDA graph capture; the target topology and GLM-5.2 exclusion were directionally correct.
- Rachael rejected the artifact for two safety blockers: non-capacity Attention could be blessed, and a root graph-output mask escape was not rejected. Strict reviewer lockout was honored, so Deckard did not revise the rejected artifact.
- Wallace owned the revision, tightened both cases with negative tests, and Rachael approved; #1171 merged as `bc1e97ff`.

## 2026-08-18T05:25Z — Gate-3 B* post-Marlin re-probe NO-GO

Luv's current-main (`923dc592`) re-probe found captured verify B* still far above the ≤~2 GO gate after Marlin: qwen2.5-14b-zp **17.5×/18.4×/20.0×** and qwen2.5-7b **14.9×/15.7×/17.4×** at K=2/4/8. Spec-decode family work remains shelved. The blocker shifted from the old #957 cheap-seam hypothesis to M>1 `MatMulNBits` launching `matmul_nbits_gemm_f16` eagerly; rerun only after the MatMulNBits/Marlin M>1 graph-safe path is actually selected.

## 2026-08-18T20:40Z — QMoE decode campaign MERGED (#1317 + #1323)

- **PR #1317** (`squad/qmoe-expert-gemv`): block-parallel router TopK for decode shapes; 81.7→9.73 µs/layer (8.4×); V2-Lite capture decode 101.17→125.68 tok/s (+24.2%); byte-identical.
- **PR #1323**: `qmoe_gate_up_activate_f32` occupancy default-ON + selective `__ldg` via `ReadOnly` template param (gate/up only); gate_up 42.68→34.24 µs (−19.8%); +2.23% V2-Lite capture decode; byte-identical. Key lesson: blanket `__ldg` on shared helper regressed fc2 +14.6%; selective scoping nets the full win.
## 2026-08-19T11:56Z — Qwen3.5 family trio complete

Scribe recorded the family-trio closure: Deckard's qwen3.5-0.8b text-only export via graph surgery merged in PR #1456 (`169febb1f`), establishing the hybrid graph-block moat at the small end. Sebastian's `CudaDropIdentityCast` PR #1459 (`792958ecf`) is a general byte-identical cleanup win (+3.0% short ctx) for hybrid graphs. Wallace completed the fair numeric 0.8B row, closing the {0.8B, 2B, 9B} qwen3.5 evidence set.


## 2026-08-19T16:40Z — fp16 fair race and per-op profiling (deckard-8)

- **fp16 export root-cause:** "Native fp16 slower than fp32" was entirely a `keep_io_types=True` export artifact. fp16io (keep_io_types=False) captures cleanly at 236 tok/s (+10% over fp32). fp16act (keep_io_types=True) declines CUDA-graph capture → eager penalty ~3× (72 tok/s). No native kernel regression.
- **Fair fp16io per-op profile (nsys captured-graph):** Native 4.769 ms/tok, 209.7 tok/s. #1 inefficiency: SSM-reduce fragmentation 20.7% (cuDNN fp32 ReduceSum + casts + LinearAttention). GQA 2.0%, argmax 0.17% — cheap and NOT the problem. Fix hypothesis (fuse f16 ReduceSum) confirmed by Gaff (+7.4%). ORT captured baseline unverifiable on-box.
- **gpt-oss-20b moat retracted:** fp32 export artifact; honest native lead ~1.5× GPU-vs-GPU. File upstream exporter change (bf16 activations), not an ORT kernel fix.
- Read-only turn; no source code changes, no PRs.

## 2026-08-19T18:20Z — Fair ORT fp16io baseline + kernel attribution complete (deckard-8)

- **Fair baseline:** ORT eager ~284–296 tok/s on qwen3.5-0.8b-hybrid fp16io (1019 CUDA / 50 CPU nodes; no structural kernel gap). Native post-#1486 = 233.6 tok/s → **~1.24× ORT-ahead**. 2.1× / 496 tok/s retired; now.md corrected in PR #1493.
- **Attribution:** Gap is entirely GPU-busy kernel time (+1269 µs/step). #1 lever: native's decomposed LinearAttention subsystem (~700–900 µs/step; fp32 cuDNN reduce + data-shuffle + gating chain). #4 lever: int4 GEMV latency-bound (+193 µs; ORT kernel 33%/call faster on M=1). Gap is fusion-closable; more cuda-graph coverage will not help. Banked in decisions.md.

## 2026-08-19T21:15Z — Post-#1503 re-attribution delivered; data-shuffle lever corrected

Delivered fresh post-#1503 gap attribution: native captured 253.9 tok/s vs ORT eager 279.1 tok/s (1.099x ORT-ahead, ~356 us/step), initially ranking data-shuffle +289 us/step as #1. Gaff's follow-up refuted that lever for captured decode: the count was an eager standalone-kernel artifact; captured replay amortizes movement kernels and future attribution must use graph-node profiling.


## 2026-08-20T00:15Z — Captured-replay attribution corrected; honest ceiling set

Corrected the post-#1503 native-vs-ORT attribution using captured-replay-aware profiling only. Native vs ORT had near-identical kernel counts (**795 vs 793**) but native carried **+617 µs/step GPU-busy**; every hot kernel was occupancy/latency-bound, not roofline-bound. Data-shuffle is retired as a dead lever because elision fragments capture. The one remaining buildable lever was int4/int8 GEMV occupancy tuning; Gaff landed it in #1516. Honest practical ceiling on this arch is **~260–270 tok/s**, with the remaining qwen3.5-0.8b batch=1 gap being launch/latency floor.

## 2026-08-20T05:50Z — VMM reservation task stood down as redundant

Stood down the device-aware VMM reservation task because `origin/main` already contains the reservation ladder via #1517. Closed redundant PR #1547, deleted branch `squad/vmm-device-aware`, reverted the memory note, and committed nothing. The large diff was branch drift from being three commits behind `origin/main`; `runtime.rs -273` came from upstream #1542 scratch-pool, not Deckard.
## 2026-08-20T05:50:19+00:00 — Phase-4 Gated-DeltaNet L2-normalize glue fusion merged

Scribe recorded Deckard's #1562 after merge to `origin/main`: `CudaL2NormalizeFusion` collapses Q/K L2-normalize chains in Qwen3.8 Gated-DeltaNet from ReduceSumSquare→Sqrt→Div into a byte-faithful fused route, reducing roughly **288→96 launches/token**. Sebastian's integrated validation measured the stacked #1561+#1562 result at q38 **61.32 tok/s** (+12.4% over the #1557 base) and mary **60.59 tok/s** (+3.0%), with mary byte-identical. Standing lesson: SSM glue fusion is useful but secondary; q38 is still forward int4 M=1 GEMV latency/occupancy-bound.

## 2026-08-20T13:46Z — #1569 merged; next GDN megakernel lever active

Scribe recorded PR #1569 after Sebastian's independent re-validation and merge to `origin/main` (`b693f2bb2`): q38 improved **61.27 → 62.76 tok/s (+2.43%)** under the relaxed dtype-tolerance bar; mary control stayed byte-identical and q38 clear prompts were coherent/byte-identical. The unsupported q38 determinism claim should be dropped because split-K GEMV nondeterminism persists. Deckard's next assigned lever is the GDN recurrence megakernel: fold β-sigmoid + softplus/dt_bias + conv1d/state into the fused recurrence.


## 2026-08-26 — #1896 rejected initial revision

Deckard's initial #1896 revision was rejected for classifying the problem away from the production defect, using a non-equivalent mutation, and unsafe unwind behavior. Durable lesson: CUDA ordering proofs must preserve the production operation and must never permit panic/unwind across teardown or FFI-sensitive paths.
<!-- Full pre-compaction hot-history snapshot archived by Scribe on 2026-08-27; original hot history above is preserved subject to checkout line-ending normalization. -->
