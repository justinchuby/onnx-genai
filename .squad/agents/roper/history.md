# roper — History

## Role
Research and feasibility specialist. Produces read-only technical scoping, competitive analysis, and go/no-go recommendations.

## 2026-08-11T03:25:00Z — Megakernel feasibility scoping delivered

- Delivered a read-only scoping study concluding a whole-step/persistent megakernel is the only remaining lever that attacks batch-1 decode GPU-side bubbles after CUDA graph capture.
- Found vLLM full CUDA graph and llama.cpp are capture/per-op systems, not true megakernels; Mirage MPK is blueprint-only for this Rust/ONNX/int4-QMoE stack.
- Recommended Phase 0 only first: persistent single-op QMoE decode, preserving fp32 accumulation order, gated on oracle margin 0.09375 and >=3% model wall-clock improvement.
