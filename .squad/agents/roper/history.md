# roper — History

## Role
Research and feasibility specialist. Produces read-only technical scoping, competitive analysis, and go/no-go recommendations.

## 2026-08-11T03:25:00Z — Megakernel feasibility scoping delivered

- Delivered a read-only scoping study concluding a whole-step/persistent megakernel is the only remaining lever that attacks batch-1 decode GPU-side bubbles after CUDA graph capture.
- Found vLLM full CUDA graph and llama.cpp are capture/per-op systems, not true megakernels; Mirage MPK is blueprint-only for this Rust/ONNX/int4-QMoE stack.
- Recommended Phase 0 only first: persistent single-op QMoE decode, preserving fp32 accumulation order, gated on oracle margin 0.09375 and >=3% model wall-clock improvement.

## 2026-08-14 — Tensor-parallelism feasibility (#933, docs on main)
Design-only scoping of Megatron 1-D TP for native CUDA decode. **tok/s NO-GO as the next lever:** decode
is not bandwidth-bound (47 tok/s = ~15% of the 4.8 TB/s roofline; byte-fold −75% bytes → +2.8%); TP adds
104 all-reduces/token → net −3% to −7%. **Precondition to flip GO:** single-GPU GEMV >~55% peak — Sebastian's
~29%-DRAM measurement (#928) keeps it NO-GO. **🟢 GO for fit/capacity:** weights 15.3→7.65 GB/GPU @ N=2 +
KV sharding runs models that don't fit one H200 (may be the stronger reason to build TP). kv_heads=2 splits
clean only at N=2 (N≥4 needs KV replication); `onnx-runtime-comm` has the trait but no NCCL backend
(multi-week to wire). Recommend the S-sized Phase-0 2-GPU all-reduce microbench first.
