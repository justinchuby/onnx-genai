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

## 2026-08-14 — Decode remaining big-build levers: build B first (#938, doc on main)
Feasibility scoping of the two multi-week levers left after the cheap ones closed. **Recommend Lever B
first** — a capture-stable padded M=K verify graph that REPLAYS (attacks the dispatch binding; floor
≈1.0×, ceiling ~2–3×; one build unlocks prompt-lookup + EAGLE-3/MTP), gated on a cheap Phase-0
capture-stability probe. Keep **Lever A** (Marlin int4 weight relayout, unconditional ~1.3–1.6×) funded
as fallback/parallel; A becomes primary only if B's Phase-0 fails. B 🟢 GO to Phase-0; A 🟡 CONDITIONAL
(full GO iff an M=1 Marlin GEMV microbench lifts achieved DRAM 29% → ≥~55%). Next step = throwaway
`#[ignore]` Phase-0 microbench before committing eng-weeks. Doc `docs/research/decode-remaining-levers-feasibility.md`.
