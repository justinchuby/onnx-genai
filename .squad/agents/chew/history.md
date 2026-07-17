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
