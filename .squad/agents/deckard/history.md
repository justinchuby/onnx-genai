# deckard — History

## Condensed history through 2026-07-18

- Systems developer on onnx-genai Rust runtime and ORT2 tracks. Delivered and reviewed loader, shape-inference, IR dtype, EPContext, encoder, external-data safety, and CPU/CUDA execution work.
- Repeated review practice: preserve model-agnostic dispatch, fail closed at claim time, use checked arithmetic, maintain byte-exact serialization, and require precision-sensitive tests.
- Owned revisions after reviewer lockouts for shape inference, IR dtype, EPContext writer, and the 2026-07-19 CPU reduction and activation dtype waves.
- Shared lesson: parallel commit-producing work requires separate worktrees; reviewer rejection transfers ownership and must be recorded.

## 2026-07-19T07:42:20Z — CSA B2 landing

- Delivered device ratio-128 compression plus device-resident FP8 cache/carry in `2f5f5e9`; Chew’s review was 🟡 APPROVE-WITH-NITS and the change landed to `main`.

## 2026-07-19T07:42:20Z — CSA B5 review and landing

- Authored the B5 ratio-4 fused candidate assembly. Chew rejected the initial slice for the five-output ratio-4 dispatch bug; Roy corrected the routing and landed `1ddf01b`, with 19/19 H200 parity tests approved.

## 2026-07-19T07:42:20Z — CSA B5 review and landing

- Authored the B5 ratio-4 fused candidate assembly. Chew rejected the initial slice for the five-output ratio-4 dispatch bug; Roy corrected the routing and landed `1ddf01b`, with 19/19 H200 parity tests approved.

- 2026-07-19T12:40Z: Root-caused CUDA token-index-10 drift to SkipSimplifiedLayerNorm RMS FMA contraction; fix already landed in de3c556 and verified at ccf994c. Logged cudarc cuda-12060/cuda-13000 feature-unification build conflict as backlog.

## 2026-07-19T13:10Z — cudarc CUDA-version unification
Fixed the cudarc CUDA-version-feature conflict blocking `onnx-genai-engine --features cuda,native-backend`: ORT keeps CUDA 12.6 as a weak default, while engine disables ORT defaults and selects CUDA 13.0 to align with `onnx-runtime-ep-cuda`. Landed to main as `db3f733`; builds passed and native CUDA Qwen decode parity was revalidated for 64 tokens.
## 2026-07-19T14:10Z — Bitwise/Hardmax lockout revision
- Revised Pris's rejected artifact: fp16/bf16 Hardmax plus stronger bitwise broadcast/rejection and invalid-axis tests. Luv 🟢 approved `7fe8961`; landed as `0b38d59`.


- **2026-07-19T16:15:00Z — CPU-EP fixes:** Corrected omitted-vs-present-empty reduction axes semantics (`6e97ee6`) after Chew’s rejection; also widened Selu/ThresholdedRelu dtype paths, with Sapper subsequently correcting f64 precision (`39edb76`).


## 2026-07-19T18:20:00Z — CPU-EP op coverage 936→975

- Corrected SpaceToDepth DCR ordering and pooling ceil-mode sizing (`014cf02`); also authored AffineGrid/Col2Im/CenterCropPad (`8e49948`).


## 2026-07-19T20:10Z — CPU-EP op coverage Batch 4

- Fixed Pris's rejected EyeLike artifact with checked diagonal arithmetic and checked dtype conversion (`114180e`); Luv approved.
- Authored GridSample 2-D/3-D coverage (`1f63750`); Gaff rejected opset-16 rank-5 acceptance, locking Deckard out before Sapper's approved correction.

## 2026-07-19T18:05Z — DeepSeek-V2 tiny E2E

- Validated the shared MLA + MoE path with a tiny fp32 export: prefill plus eight decode tokens completed without runtime changes.
- Added gated engine coverage (`0caaf32`) and the Mobius export helper (`2b629cc`); Gaff approved both.
- DeepSeek-V4 remains blocked upstream by the missing usable reference configuration/export artifact.

- 2026-07-19: ConvTranspose CPU kernel landed as 7219025 with 11 conformance tests; Gaff approved. Restored DeepSeek grouped top-k routing after QMoE regression (cd782dd), enabling Chew approval. Unique String attempt exposed runtime-layer UB and was superseded by safe removal. MLAS-style SIMD GEMM port remains in progress on `deckard/mlas-gemm`.


### 2026-07-20 — Vendored MLAS CPU-GEMM parity

Recorded the MLAS vendoring spike (`556b0d8`) and multi-threaded Rayon hook (`8764b3d`); provenance was corrected in `ee7a6cd`.

## 2026-07-20T05:20:00Z — MLAS int4 and PackedB milestones

- Landed MLAS PackedB reuse and MLAS SQNBitGemm wiring for CPU MatMul/MatMulNBits (`3eed80a`); f32 direct-output and feature reachability were completed in the same milestone batch. Gaff-51 reviewed the SQNBit change 🟢; int4 decode improved ~1.9× and prefill up to 9.5×.


## 2026-07-20T07:15Z — M-based MLAS int4 routing

- Landed `4bb98be`: M-based `NXRT_SQNBIT_PREFILL_MIN` routing keeps hand int4 for M=1 decode and MLAS for prefill; gaff-52 reviewed 🟢.


## 2026-07-20T13:35:00Z — Multistream performance and issue #40

- Investigated coarser CPU decode fork-join granularity, measured 7–8% regressions, reverted the prototype, and established GQA (20.6 ms) as larger than MatMulNBits (15.5 ms) post-residency.

## 2026-07-21T03:15:00Z — CUDA graph M4 validated
- Fixed CUDA graph handle ownership, persisted GQA decode scratch, hardened replay metadata bounds, and replaced elementwise boolean capture gates with exact warmed signatures (`5470c01`, `dcb4f1b`, `82c249d`, `85b6f4e`).

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.

## 2026-07-21T05:40:00Z — fp16 decode and cross-platform reconciliation

- Landed structured CUDA-decline/whole-session CPU fallback reporting and strict `ONNX_GENAI_REQUIRE_CUDA` enforcement (`3a8eebe`); Batty approved. Also propagated optional CPU tracing to native consumers in `61f4d2c`.


## 2026-07-21 — Wave-2 and CI milestone
CI now covers all 27 offline crates with warnings-as-errors and native Windows ARM64. Capture-safe native fp16 CUDA decode wave 2 stacked GQA prep fusion, warp-shuffle RMSNorm, and specialized down-projection GEMV on wave 1, reaching 663–672 tok/s on H200 versus ORT GenAI at 657, with zero fallbacks. All CUDA EP kernel work must remain correct and fast across supported SM architectures, not only sm_90.

## 2026-07-21T13:15:00Z — Replay binding cache dropped
- Evaluated capture-generation binding metadata caching; tests passed but paired runs gained only +0.23%. The raw-address correctness-sensitive hot-path change was not merged and is recorded as a dead end not to re-attempt without stronger evidence.
- 2026-07-21T23:55Z — DS-1 generic Slice→Unsqueeze shape propagation landed after Holden bounded materialization and Pris approved; ScatterElements dtype expansion also landed.
## 2026-07-22T12:00:00Z — Luv Phase 0 review
- Independently reviewed Luv's partial-CUDA-graph capture path-kind change at `3c94a57`; approved 🟢 GREEN after confirming additive behavior, correct structural seam mapping, model-agnostic dispatch, exhaustive matches, and clean fmt/clippy/tests.

### 2026-07-22T14:59:36+0000 — WP-B landed
WP-B landed: Deckard's intermediate WP-B3 revision fixed raw membership/default classification but was superseded by Sapper's v3 raw signature fix.

- 2026-07-23: Authored Phi int8 fused norm work (`c644b0f`, +13% Phi) and root-caused/fixed the real Qwen int4 fused-GEMV regression with `12efc92` (`HasZp` specialization), merged to main.

## 2026-07-23T14:55:00Z — Qwen-7B decode roofline no-go

- Completed Qwen-7B column-split, true K-slice split-K, and vectorized-load/roofline investigations. All were no-go; the remaining int4 GEMVs are shared-memory/weight-read-efficiency bound, so main stayed clean.
- Standalone Phi int8-zp split-K remains a validated +2.1% branch result with clean CUDA tests/clippy.

## 2026-07-23T18:30:00Z — On-device LongRoPE select landed

- `97c1a56` landed the shared `CudaOnDeviceConstantSelect` capability and capture-safe scalar `Where`: pure constant-branch `If` nodes can now lower to a live device predicate with conservative unequal-table zero-padding guards.
- Phi LongRoPE decode collapsed from two captured regions to one. Interleaved idle-GPU performance improved 203.50→322.15 tok/s (+58.3%), with byte-identical 160-token and 4,200-token boundary generation and a 201/0 CUDA gate.

## 2025-06-14T00:00:00Z — QMoE vectorized unpack merged

- Vectorized int4 unpack plus compile-time quant-layout specialization landed on `origin/main` as `53f9df6`.
- QMoE linear time fell 6.36→2.51 ms/token (-60.5%); real DeepSeek E2E rose +11.9% (block-32) and +13.8% (block-128), with dense behavior unchanged.
## 2026-07-24T05:11:20+0000 — Whole-step DeepSeek CUDA-graph capture

- DeepSeek-V2-Lite int4 decode reached one captured segment and **0 eager seams** (727→0) on main after Leon's Reshape fold (`661618b8`) and Rachael's mask-island closure (`3dc0843b`).
- Integration retained deterministic coherent output (` Paris.\nThe currency of France is the Euro.`); CUDA library gate: 205/0.

## 2026-07-24T05:48:20+0000 — Same-harness native/ORT backend flag landed

- `profile_native --backend native|ort|auto` landed as `d03261c7` after the reviewer-required restoration of the byte-identical native header and invalid-value parser coverage.
- The flag enabled the definitive foundry-local CUDA A/B: native whole-step capture reached 902 vs ORT 584 tok/s on Qwen2.5-0.5B (1.55×) and 322 vs 238 on Phi-4-mini (1.35×), with exact token parity. ORT capture currently fails (`ort_value must contain a constructed tensor`), so ORT was eager.
