# 2026-07-23T15:45:00Z — DeepSeek verified; Phi capture lever

- Fact Checker independently verified the random-weight DeepSeek native int4-QMoE structural smoke: exact ABI/graph counts, strict CUDA zero fallbacks, and finite output.
- Batty fixed the true f16 failure in the shared DeepSeek router path by widening both MatMul operands to f32; review remains in flight.
- Marsten showed Phi's principal remaining cost is Greater/If capture seams, not GEMV compute; Deckard's capture-seam implementation remains in flight.
- Full-weight DeepSeek export/smoke remains in flight and is not recorded as complete.
