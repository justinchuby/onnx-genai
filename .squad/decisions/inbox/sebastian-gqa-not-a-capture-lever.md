### 2026-08-13: GroupQueryAttention is NOT a viable perf lever under CUDA-graph capture — decode is node-dispatch bound (2568 nodes/token)

**By:** Sebastian (Performance/Systems)

**What:**
Investigated the coordinator-assigned GQA optimization (GQA showed ~41% of *eager* decode). Conclusion: **there is no beneficial GQA code change** at this workload. The eager 41% is almost entirely per-launch overhead that CUDA-graph capture already amortizes. The real remaining lever to go beyond 47 tok/s is **reducing the captured graph node count (kernel/op fusion)**, which is decode-graph-structure work (Batty's domain), not GQA kernel work. **No GQA PR opened** (a "GQA optimization" here would be a no-op that risks parity for zero gain).

**Evidence (all H200, CUDA_VISIBLE_DEVICES=0, ONNX_GENAI_CUDA_GRAPH=1, staged Muse-Glimmer, --pipeline --backend native --ep cuda, capture 1seg/0seams, first-16 greedy ids match reference):**
- **Baseline:** 47.20 tok/s median (47.44/47.17/47.20), 20.98 ms/token.
- **Captured node count (instrumented `cuGraphGetNodes` on the decode segment): 2568 nodes/token.** → 20.98 ms / 2568 = **8.17 µs/node.** Graph replay removed the CPU launch-API overhead, but the GPU still serially issues 2568 grid launches/token on the single capture stream; ~8µs/node of fixed launch latency + minimal-execution dominates.
- **Diagnostic: cap GQA decode key-loop to 1 key** (removes ~700 keys of KV read/softmax work, identical launch geometry → capture-safe): **47.52 tok/s** (flat/noise). GQA sequence-length work is <~0.5 ms/token = <2% of decode. Optimizing it 2× would gain <1% tok/s.
- **Diagnostic: split_fill TARGET_WAVES 2→8** (9→16 splits): **47.27 tok/s** (flat) → decode is NOT split-parallelism starved.
- **Diagnostic: force split_fill=1** (removes 52 merge nodes but serializes 700 keys/CTA): **46.31 tok/s (WORSE)** → existing 9-split config is near-optimal; the 52-node reduction did not pay because it serialized the loop-carried flash-softmax dependency.
- **Diagnostic: cheapen general f16 GEMV depth-loops** (qkv/o-proj): **47.13 tok/s** (flat).
- **Diagnostic: cheapen MLP int4 GEMV loops** (down + swiglu block_base): **47.17 tok/s** (flat).
- **Signature finding:** cheapening *any single kernel's inner loop* — GQA seq-loop, general GEMV, MLP GEMV — leaves tok/s unchanged. That is the fingerprint of **dispatch-bound execution**: inner-loop compute is a small fraction of the ~8µs fixed per-node cost.

**Kernel state (for the record):** `gqa_decode_bf16` (from #855) is already bf16-native I/O (Q/K/V/out `__nv_bfloat16`, all softmax stats + value accumulators fp32), already `__nv_bfloat162`-vectorized, flash online-softmax with split-K (direct-output fast path when active_splits==1). No residual bf16↔f32 Cast round-trip, no obvious structural inefficiency. The known GQA-broadcast DRAM re-read (group_size=16 → 16× KV amplification) is immaterial at the benchmark's ~700 seq (KV traffic ≈ 0.6 GB/token ≈ 0.9% of decode); it would only matter at 4k–8k context, which is not in the benchmark.

**Why (root cause & recommended redirect):**
Under capture the decode is bound by **node count, not per-kernel compute or bandwidth.** 2568 tiny serial nodes × ~8µs = the 21ms/token floor. The node population is dominated by small elementwise/norm ops that are pure per-node overhead and prime fusion targets (per-token approx from the eager profile): MatMulNBits ~208, SimplifiedLayerNormalization ~156, Add ~155, Mul ~105, GQA 52, plus GQA split/merge and the mask/position elementwise chain.

**Recommendation to coordinator:** do not pursue GQA further. Redirect the next lever to **captured-graph node-count reduction via fusion** — e.g. fuse residual-Add + SimplifiedLayerNormalization (skip-norm), fold the SwiGLU `Mul` into the gate/up GEMV epilogue, and collapse the attention mask/position elementwise chain. That is decode-graph-structure work → **bring in Batty**, with me (Sebastian) available for the kernel-side epilogue fusion. Any fusion touching accumulation/softmax → **flag Chew** (numerics). Keep byte-exact greedy parity (ref ids `[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, 328, 2885, 262]`).

**Note on profiling:** hardware profilers remain blocked in this sandbox (ncu not installed; nsys fails "Creating threads in this process is forbidden by design"; RmProfilingAdminOnly=1). All numbers above are from the built-in op timer, the `cuGraphGetNodes` instrumentation, and capture-safe end-to-end A/B loop-cap diagnostics.
