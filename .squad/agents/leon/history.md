# Leon — History

## Role and invariants
Engine/KV/runtime-buffer implementer. Runtime owns KV; model geometry comes from `inference_metadata.yaml`, not ORT-GenAI configuration. Preserve device-buffer ownership, past/present aliasing contracts, exact real-model comparison settings, and reviewer lockouts.

## Summary through 2026-07-14T20:05:00Z

### Engine and KV
Implemented attention-sink SWA, SharedKv generalization, connector engine wiring, and real KV byte materialization. Prefix lookup initially remained metric-only until K4 added symmetric f32 payload extraction/injection. Prefix-dependent hashing now proves equal keys imply equal prefixes. Follow-ups remain for multi-layer fixtures, graceful recompute fallback, and heterogeneous connector payloads.

### Gemma4 speculative execution
Migrated engine paths to per-layer KV geometry and helped deliver real heterogeneous Gemma4 E2B execution. Corrected proposer inputs to `embed(last_token) + last_hidden`, raising acceptance from 25% to 70.6% with token identity preserved. Performance remains below greedy and is a separate tuning concern.

### Loader dtype and fusion hardening
Closed silent Float32 fallbacks with `UnsupportedDataType` and fail-closed decoding across all real dtype sites; Holden approved. Added strict LayerNorm operand-order guards and adversarial coverage; Gaff approved.

### EPContext and encoder
Rejected encoder v1 for generic-layer EPContext literals violating the model-agnostic rule; Deckard's v2 passed. Revised EPContext writer sidecar naming after Batty's rejection, but introduced an over-broad duplicate-identity rejection; Leon is locked out of that writer artifact and Gaff's v3 is final. Later unified external-path guarding and explicit C API mapping were approved.

### Product/API and packaging
Renamed the full C ABI from `ort2_*` to `nxrt_*` with no compatibility aliases. Broke the shape-inference/loader publication cycle by making the loader dev-dependency path-only in the packaged manifest; Roy approved.

### Recent validation
Loader opset-import validation for file, from-parts, and nested-subgraph paths merged in `00cda89`; the executor's sentinel failure path is now an unreachable invariant. Holden's final review was green.

- 2026-07-15 — Added Windows oneDNN wheel bundling in `ef89a95`; CI verification is pending.

- 2026-07-16T00:00:01Z — Re-profiled native CPU decode after MatMulNBits threading and landed allocation-free, same-shape contiguous-f32 `Mul` (`347060f`). The guarded non-aliased fast path reduced Mul 3.12→0.25 ms and decode 40.5→44.2 tok/s; Holden 🟡 approved (independent +6.35%).

- 2026-07-16T00:00:00Z — Streamlined M=1 GQA decode to write contiguous f32 attention and present K/V outputs directly (`1fdd1ec`), preserving the generic prefill/strided/non-f32 path. GQA fell 0.865→0.690 ms/step and decode rose 54.38→58.44 tok/s (+7.5%) with exact eight-token output; Sebastian cleared the change and 413 CPU EP tests pass.

- 2026-07-16T00:00:00Z — Repaired the rejected CUDA executor control-flow paths in `5c0f05f` under Deckard's lockout. Non-host SequenceAt values now synchronously upload to correctly stamped CUDA buffers; Scan retains host staging and relies on child-executor H2D. Added CUDA SequenceAt/Scan versus CPU parity coverage; Holden cleared the repair, with exact Qwen tokens, session 112/112, and CUDA EP 117/117.

## 2026-07-16T15:39:27Z — Scribe session update

- 🟢 Reviewed BlockQuantizedMatMul: hand-verified MXFP4 0xD7→12.0/-6.0 and IQ4_NL decoding; unsupported IQ formats fail closed and 420 CPU tests pass.

## 2026-07-16T18:11:48+0000 — IQ-family CPU decode reviews

- 🟢 Cleared Bryant's IQ2_XS/IQ2_S/IQ3_XXS and IQ1_S/IQ1_M implementations after upstream llama.cpp grid, layout, fingerprint, and hand-trace audits.
- CPU `BlockQuantizedMatMul` now covers the complete supported IQ family.

## 2026-07-16T19:05:18+0000 — BlockQuantizedMatMul prefill review

- 🟢 Cleared Joi's `5010261`: all ten formats matched scalar decode bits, selected MXFP4/IQ4_NL/IQ4_XS AVX2 paths were independently checked, and generic GEMM retained K accumulation order.
- Default and oneDNN CPU EP suites each passed 430 tests; M=64 generic matmul gains measured 32–35×.

## 2026-07-16T19:27:57+0000 — CUDA IQ super-block GEMV wave

- 🟢 Cleared Roy's shared `onnx-runtime-quantization` extraction: all seven moved grids/sign tables are byte-identical (IQ1S FNV-1a `0x6703ed863501ae2e`); CPU decode and Joi's AVX2 paths are unchanged, and the standalone crate builds/tests cleanly.

## 2026-07-16T19-27-57+0000 — Scribe session update

- 🟢 Cleared Sapper's `67c1e3b` quantized-matmul shape rules: domains, `N`, symbolic dimensions, dtype preservation, error handling, and 2D/3D coverage are correct (93 unit tests + one doc-test).

## 2026-07-16T23:30:00+0000 — GAFF loader foundation review

- 🟢 Cleared Sapper's `2a9e5b1`: formal subgraph I/O is ordered and typed, recursive scopes isolate inline initializers, and UNDEFINED graph attributes retain populated graph fields.
- The Loop load regression and all 101 loader tests passed; existing validation already permits If/Loop/Scan.

## 2026-07-16T23:58:29+0000 — Comparison/logical inference review

- 🟢 Cleared Chew's `d06d1e7`: all comparison/logical output dtypes are Bool, broadcast/unary shapes hold, and bitwise operators were untouched; 115 tests passed.


## 2026-07-17T00:58:13Z — GAFF Loop remediation

- Under Sapper's lockout, repaired the rejected Loop design: removed the untrusted eager scan reservation and validated every loop-carried output against its initial dtype and full shape.
- Holden 🟢 re-approved the huge-`M` early-exit and second-iteration shape-change regressions; 121 session tests passed and final commit `f6e8ba6` merged. `Scan` is now the remaining control-flow work.

## 2026-07-14T00:00:00Z — Scan hardening and normalization inference

- Repaired Scan stack-shape arithmetic against zero-masked overflow (Holden 🟢) and added BatchNormalization/InstanceNormalization shape inference (Bryant 🟢).

## 2026-07-17T07:19:39Z — WEIGHT_OFFLOAD Phase 1 landed

- Delivered `f601cad`: `WeightRegionCatalog`, route-first mmap QMoE expert selection, and opt-in `ONNX_GENAI_WEIGHT_OFFLOAD=1`; default behavior remains unchanged.
- Chew's corrective `a77eed0` and Nabil 🟢 approval closed the landing; large-model exact-logit/throughput validation remains deferred.

- 2026-07-18 Scribe: Initial Reshape/Split validation and coverage work was superseded by the reviewed correction.

## 2026-07-18T01:20:34Z — CUDA SparseKvGather D==0 fix landed
- Fixed validation ordering in `c2180c9`; three D==0 parity tests passed 12/12, Gorman re-approved, and CUDA SparseKvGather landed.

- 2026-07-18: CUDA CSA claim gate was corrected to mirror CPU ratio-specific contracts and parity tests, then superseded by Deckard's shared attention_bias validation; final CSA approval landed.
- 2026-07-18T05:55:00Z — Added CPU CSA `supports_op` claim validation via unified factory dry-run on lockout reassignment (`2a08ef9`); Deckard approved.
- 2026-07-19: Fixed PR #30 device sampling parity/safety; continuing PR #32 rebase, build, and review-comment fixes.
- 2026-07-19T07:55:00Z: PR #32's EP-capabilities refactor merged at `9683a08` after the rebase and three review fixes.

## 2026-07-19T07:42:20Z — Mobius-head E2E harness landed

- Landed `3d47ea9`: pinned GLM-5.2 and DeepSeek-V4-Flash manifest plus ignored, environment-gated real-engine E2E smoke. Gaff approved; absent artifacts skip cleanly and no download path was added.

- 2026-07-21: Scribe reconciled the perf campaign inbox; key decisions are now consolidated in `.squad/decisions.md` under the 2026-07-21 perf campaign section.
