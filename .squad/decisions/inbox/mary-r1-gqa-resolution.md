### 2026-07-30: DeepSeek-R1 GQA decode resolution

**By:** Mary

**What:** Native R1 decode is correct: native CPU, native CUDA, and ORT CPU select token 374 with the f32 oracle margin. ORT CUDA's fp16 MatMulNBits near-tie flips generated token 7 to 315 and leads to repetition. CI regression coverage now exercises 12:2 grouped-query attention with non-interleaved rotary and multi-step KV decode at head widths 64 and 128.

**Why:** The width-128 case mirrors the deployed R1 graph, while width 64 retains coverage for the originally reported geometry. Both guard rotary position advancement, 6:1 head grouping, causal attention, and chained past/present KV correctness.
