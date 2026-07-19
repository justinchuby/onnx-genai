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
