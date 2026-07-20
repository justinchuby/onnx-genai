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
