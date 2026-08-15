# Chew — History Archive

## Archived 2026-07-29 (full pre-compaction snapshot)

# Chew — History

## Role and review principles
Numerics/precision reviewer. Require reference-backed coherent outputs, not merely successful execution. Watch dtype/layout symmetry, silent coercions, opset semantics, broadcast behavior, stable reductions/softmax, and realistic parity tests.

## Summary through 2026-07-14T20:05:00Z

### KV and speculative decoding
Verified connector cache separation, byte-layout symmetry, prefix-dependent hashing, fetch/recompute boundaries, per-layer heterogeneous KV geometry, and Gemma4 shared-KV correctness. Flagged the configurable CPU-load estimate bug (fixed by Zhora), multi-layer fixture coverage, graceful recompute fallback on import failure, and heterogeneous connector payload support. Gemma4 acceptance correction raised acceptance from 25% to 70.6% while preserving token identity.

### ORT2 CPU and session numerics
Reviewed CPU kernels, session executor/dynamic shapes, and Phase-1 hardening. Confirmed GEMM, LayerNorm, softmax stability, broadcast, Erf/Gemm, allocation bounds, and dynamic-shape behavior. Key follow-ups included legacy Softmax semantics, Min/Max NaN propagation, saturating casts, and cache-key dtype completeness; hardening subsequently closed the numeric defects.

### Shape inference and dtype safety
Rejected the original contrib FusedMatMul shape rule because transpose attributes were ignored; Deckard's corrected rule passed re-review. Approved loader/session shape-inference wiring and symbolic representative behavior. Independently verified ONNX dtype discriminants and supported fail-closed decoding rather than silent Float32 fallback.

### Optimizer and fusion reviews
Approved opt-in session optimization and the DAG-aware LayerNorm, FusedAttention, and related fused paths after parity/adversarial checks. Earlier LayerNorm review identified axis-as-input, epsilon-type, and operand-order decline guards; later work closed the sign-flip over-match. Fusion tolerances remained distinct from conformance tolerances and were not loosened.

### EPContext and C API
Approved the model-agnostic EP API contract and reviewed consume, option parsing, export, and FFI paths. Confirmed EPContext nodes cannot fall through to CPU execution, binary payloads remain byte-exact, FFI entry points are null/UTF-8/panic guarded, and disabled export is side-effect-free. Fixed explicit `DanglingEpContext` C API error mapping.

### Recent binding follow-up
At 2026-07-14T19:05:00Z, fixed clippy findings and corrected Python pytest counts in merged commit `878559f`.

## 2026-07-15T01:52:00Z — Session update

- Delivered DLPack zero-copy export (`6fdccc8`): C ABI plus Python NxrtValue `__dlpack__`/`__dlpack_device__`.

## 2026-07-15T00:00:00Z — Cross-agent session update

- Delivered contrib fused transformer kernels; follow-up review fixes for SkipLayerNormalization/SimplifiedLayerNormalization merged in the opset coverage wave.

## 2026-07-16T17:00:38+0000 — DeepSeek-V4-Flash MTP and CSA export
- Updated Mobius PR #405 (`7e26e6e`) with the 0/4/128 CSA schedule, sparse-index/compression tensors, attention sinks, dense fallback, and an MTP sidecar.
- Native sparse KV-cache/index operations and iterative MTP orchestration remain required runtime work.

## 2026-07-16T23:58:29+0000 — Comparison/logical Bool inference

- Delivered `d06d1e7`: comparison/logical shape inference now produces `tensor(bool)` while preserving broadcast and unary shapes; Leon 🟢 cleared 115 tests.
- Expanded-Attention now reaches unsupported `Mod` at node 50; `mod-op-support` is next.


## 2026-07-17T00:58:13Z — Logical execution and Expand inference

- Merged `557ca87`: CPU `And`/`Or`/`Xor`/`Not` kernels use Bool truth semantics, broadcasting, and canonical output bytes; Bryant 🟢 cleared 436 CPU tests.
- Merged `14b5136`: opset-8+ `Expand` shape inference performs bidirectional broadcasting with dtype passthrough and known-rank fallback; Bryant 🟢 cleared 120 shape-inference tests. Expanded-Attention now advances past node 58.

## 2026-07-17T07:19:39Z — WEIGHT_OFFLOAD repair

- Repaired all four Phase-1 findings in `a77eed0`: bounded dequant residency, unaligned mmap provenance, endpoint-overflow rejection, and sum-of-distinct mapped-byte metrics.
- Nabil 🟢 approved; 691 tests passed.
- 2026-07-19: Reviewed BQMoE through three cycles and approved final zero-allocation claim gate (`67abdb5`).
- 2026-07-19T07:55:00Z: Approved IndexShare v1 with a coverage nit that was addressed, and approved CSA B0 after full FP8/FP4 quantization and meaningful oracle tests.


## 2026-07-19T07:42:20Z — CSA B2 review

- Reviewed B2 as 🟡 APPROVE-WITH-NITS, then approved Batty’s `2067504` nit fix 🟢; 14/14 GPU parity tests remained bit-exact on H200.

## 2026-07-19T07:42:20Z — CSA Phase B B3/B4 reviews

- Approved Sapper’s B3 (`3ae3244`) and Roy’s B4 (`77a44a4`) after numerics review. B3 passed 15/15 and B4 17/17 H200 GPU parity tests bit-exact with no blocking findings.

## 2026-07-19T07:42:20Z — CSA B5 review and re-review

- Rejected B5’s five-output ratio-4 misrouting to the ratio-128 kernel, then approved Roy’s ratio-keyed fix and regression test in `1ddf01b`; 19/19 H200 parity tests were bit-exact.

## 2026-07-19T07:42:20Z — CSA B5 review and re-review

- Rejected B5’s five-output ratio-4 misrouting to the ratio-128 kernel, then approved Roy’s ratio-keyed fix and regression test in `1ddf01b`; 19/19 H200 parity tests were bit-exact.

## 2026-07-19T07:42:20Z — CSA Phase B B6 review

- Approved B6 with non-blocking nits: top-k workspace bound assertion, eager warmup documentation, and multi-step cursor/geometry advancement for B7.
- Confirmed 20/20 CSA parity/capture tests and the full ep-cuda suite green on H200; capture/replay was byte-identical to eager and the CPU oracle.

## 2026-07-19T07:42:20Z — CSA Phase B B7 review

- Approved B7 with non-blocking nits: add a completed-compression-block rollback boundary test and correct the five-output ratio-4 host metrics mode label. Verified 24 CSA tests plus 1 ignored MTP smoke and the full ep-cuda suite green on H200.
## 2026-07-19T14:10Z — Scan/window review
- 🟡 Approved Bryant's `5816d23`: CumSum/CumProd semantics, dtype coverage, and Hann/Hamming/Blackman formulas were correct. Nits: scalar-axis strictness and non-f32/default window coverage.


- **2026-07-19T16:15:00Z — CPU-EP review:** Rejected the initial reduction axes semantics, then approved Deckard’s omitted-axes fix (`6e97ee6`).


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Rejected Bryant’s initial pooling/layout implementation, then approved Deckard’s SpaceToDepth and ceil-mode fixes (`014cf02`).

- 2026-07-19: Rejected fused QMoE integration after it clobbered `_group_topk_selection` and broke grouped routing. Approved after Deckard restored the original signature and group-mask implementation.


### 2026-07-20 — Vendored MLAS CPU-GEMM parity

Recorded the integration rejection and subsequent approval after dependency metadata and SIMD guard fixes (`85087ac`).


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Reviewed both CUDA flash paths: standard Attention 🟡 approved with coverage notes; initial GQA fusion 🔴 rejected, then Rachael’s 40-scenario causal-origin correction 🟢 approved.

## 2026-07-21T03:15:00Z — CUDA graph M4 validated
- Drove multi-round capture-safety review to final green: required real replay coverage, exact elementwise signatures, GQA detect-before-consume poisoning, and corrected Qwen smoke assertions before the zero-fallback M4 track landed.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.
- 2026-07-21T23:55Z — WP2 rejection/lockout and ScatterElements review were recorded; final WP2 acceptance came through Pris after Sapper revision.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Chew's rejection of loader-IR shape authority directly informed the final Sapper WP-B3 v3 fix.

2026-07-22T22:15:00Z — Approved Zhora’s `f8848c9` MatMulNBits f16/bf16 and topology tuning with non-blocking parity/regression follow-ups.

## 2026-07-26T20:00:00Z — Scribe update

- 2026-07-26T20:00:01Z — Approved PR #201 after independently reproducing the `dtod` revert failure (`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`) and verifying persistent workspace staging preserved #193 copy-back safety.
## 2026-07-26T22:38:02+00:00 — PR #208 RoPE capture review landed

- Independent APPROVE for PR #208 landed with the merge commit `5eb0d8db`, closing #88. Guard proof remains the key review evidence: removing `!capturing` at `rotary_embedding.rs:495` made the new test fail with `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, then restoring it passed.

## 2026-07-27T01:30:00-07:00 — PR #227 CPU EP NEON numerics review

- **APPROVE with concerns** for Iran's 4-commit CPU EP optimization branch (`squad/mac-cpu-ep-roofline`): NEON SiLU, SDPA, GEMV, Accelerate sgemm, dtype fast path.
- SiLU polynomial: measured ~28 ULP in practical range (claimed ~1 ULP — docstring incorrect). Acceptable for inference.
- Swish→SiLU canonicalization: exact f32 equality correct, no silent misroute.
- SDPA NEON: numerics sound (softmax max-subtraction stability inherited), but zero test coverage for the NEON dispatch path — all tests call scalar reference directly.
- GEMV: transpose correct, tail handling correct, f32 accumulation throughout. Guard-break test passed.
- dtype.rs f32 memcpy: contiguity guard is sound.
- matmul_nbits.rs: visibility change only, safe.
- All NEON intrinsics are ARMv8 baseline. No hardcoded cache/thread counts.
- 7 dead code items from removed Accelerate sgemv path.
- Filed to `.squad/decisions/inbox/chew-pr227-numerics-review.md`.

## 2026-07-27T02:00:00-07:00 — PR #227 FP16 Path Review (Second Pass)

**Scope:** Commits `75311827` (FP16 storage GEMV + NEON bulk f16↔f32) and `3a88ba8c` (SPMD pool for FP32 GEMV + cleanup).

**Verdict: APPROVE** — numerics are sound.

### Key findings
- **Inline asm `fcvtl`:** Constraints, clobbers, and options are correct. Bit-exact against scalar `half::f16::to_f32()` across all edge cases (denorm, inf, NaN, ±0). Using asm to avoid nightly `f16` type is justified today. Recommend TODO for intrinsic replacement.
- **FP32 accumulation verified:** Measured max relative error 2.38e-7 vs f64 reference across model-scale shapes (gate/down/q/kv projections). FP16 vs F32 GEMV same-weight discrepancy: 1.73e-6 — confirms accumulation is genuinely f32.
- **Bulk conversion:** `fcvtn` narrow matches `half::f16::from_f32()` bit-for-bit. Round-to-nearest-even confirmed. Overflow → inf. Denormal/NaN preserved. Asm annotations correct (`nostack` only for write path, `readonly,pure` for read path).
- **Tail handling:** K=67, N=9 correct. K=1/N=1 correct.
- **Transpose cache:** `OnceLock` provides thread-safe lazy init. Rayon `par_chunks_mut` writes to disjoint slices.
- **SPMD pool:** `perf_cores.saturating_sub(1).max(1).min(available)` guarantees ≥1 worker. `None` fallback on Intel/VM is correct.
- **Tests:** 922 passing (906 lib + extras). 3 new FP16 GEMV tests + 1 updated cache test.
- **Non-blocking concerns:** C1 = add TODO for intrinsic migration; C2 = tighten test error thresholds (2% → 1e-4).
- Filed to `.squad/decisions/inbox/chew-pr227-fp16-review.md`.


## 2026-07-28T00:40:00-07:00 — PR #334 Grouped/Depthwise Conv Review

- **REJECT** (formatting) for Deckard's depthwise conv im2col+GEMM PR.
- `cargo fmt --all -- --check` fails with 3 violations — same class as #324.
- **Numerics: SOUND.** Grouped im2col indexing is correct across all 8 parity tests (true depthwise, grouped-not-depthwise, channel multiplier, non-SIMD-width channels, stride>1, dilation, asymmetric padding). Guard-break test detects off-by-one immediately.
- **BNNS claim independently verified:** Probed `BNNSFilterCreateLayerConvolution` directly via FFI. With `groups > 1`, BNNS either returns NULL (oc_per_group in descriptor) or accepts but only writes group 0's output (full oc mode). The deprecated API is genuinely broken for groups > 1. Guard is justified.
- **Fall-through:** No #275 pattern. Both paths produce fully-populated output vectors.
- **Non-grouped path untouched** (byte-identical except defensive n==0 guard).
- **Reachability:** Counter `CONV_IM2COL_GEMM_TEST_HITS` covers both branches, manifest claim present, test genuinely forces grouped path.
- **12× gap judgement:** im2col is structurally wrong for depthwise (memory-bound, K=9, M=1). Direct NEON kernel would be 4–8× faster (eliminates im2col buffer entirely). This PR is a correct intermediate step. Schedule NEON depthwise follow-up targeting 2–3× ORT.
- **Revision agent:** Iran.
- Filed to `.squad/decisions/inbox/chew-pr334-review.md`.
- 2026-07-28: Reviews of PR #347 and #349 approved after verifying numerical bounds and real decode firing. Documentation rationales are reviewable correctness artifacts: wrong L1-cache premises and derived-looking fitted constants must be corrected before merge.

## Archived from live history (Scribe compaction 2026-08-12T00:15:00Z)

### 2026-07-27T01:30:00-07:00 — PR #227 CPU EP NEON numerics review
APPROVE with concerns. SiLU polynomial ~28 ULP (not ~1 ULP as claimed). SDPA zero dispatch coverage. GEMV transpose/tail/f32-accumulation correct. Filed chew-pr227-numerics-review.md.

### 2026-07-27T02:00:00-07:00 — PR #227 FP16 Path Review (Second Pass)
APPROVE. fcvtl asm bit-exact, FP32 accumulation max-rel-error 2.38e-7, bulk conversion bit-for-bit, tail handling correct, SPMD pool guaranteed ≥1 worker.

### 2026-07-28T00:40:00-07:00 — PR #334 Grouped/Depthwise Conv Review
REJECT (formatting). 3 cargo fmt violations. Numerics sound. BNNS genuinely broken for groups>1 verified by FFI probe. im2col correct intermediate step; NEON depthwise follow-up targeting 2-3× ORT.

### 2026-08-11 — AVX2 LayerNorm/RMSNorm precision audit + test updates (PRs #31973, #31974)
Two-pass E[x²]-mean² suffers catastrophic cancellation at high base. Welford SIMD recommended. BF16 widen-accumulate-narrow ≤1 ULP above floor. Leak scrub (2 agent names) under lockout.

### 2026-08-11 — PR #31973 clarity pass
Added explanatory comment: two-pass safe in fp64; kept as oracle for independence. Removed dead (void)0; 40/40 tests pass.

### 2026-08-11 — #762 Test Repair (Corrective Wave, B1-B4)
All four blockers resolved. 154+20+32+10+6 tests passed. Clippy/fmt clean.

### 2026-08-11 — PR #762: LayerNorm test hardening
Removed #[ignore], added shape assertions, neg-axis test, RMSNorm test. 23 passed.

### 2026-08-11 — PR #31974 CI fix: unused functions
Removed BF16Ulp and ReportErrors dead functions. 45/45 BF16 tests pass.

### 2026-08-11 — Leak scrubs on PR #31973 (×2)
Both interactive-rebase rewrites; force-pushed. 42/42 tests green.
