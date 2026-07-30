# mary — History

## 2026-07-30T04:10:00Z — Reduction and shape-aware CUDA claim-gate work

- Authored PR #420 to widen extended reductions to f16/bf16 with f32 accumulation; merged as `6610f86f`, clearing the native 27B FP16 `ReduceSumSquare` CUDA fallback.
- Revised PR #424 at `93d9e7b8` with `require_input_rank`, making CUDA claim gates shape-aware so deferred ranks retain CPU fallback instead of being treated as unsupported static shapes.
