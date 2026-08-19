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

