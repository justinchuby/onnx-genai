### 2026-08-15: Combined base-decode stack measurement (glm-4-9b-int4) — where we stand vs ORT base

**By:** Deckard (Systems Dev, CUDA/decode-perf)
**Branch:** `squad/combined-stack-measurement` (throwaway; off main `3f826936` which already has #991 lm_head cuBLASLt) + merged #996 fp16 GEMV + #999 flash-decoding. Both merges clean (disjoint files: matmul_nbits.rs vs gqa_decode*.rs). No source changes of my own — pure build+profile.
**Purpose:** Produce the definitive honest base-vs-ORT decode number with ALL landed/pending base wins stacked, graph-ON, and name the residual dominator.

#### Stack under test
- **#991 lm_head cuBLASLt** — opt-in `ONNX_GENAI_LMHEAD_CUBLASLT=1` (in main).
- **#996 fp16 int4 GEMV** — opt-in `ONNX_GENAI_GEMV_FP16=1` (accuracy-gated, ORT-matching fp16 half2 accumulate; NOT byte-identical to fp32).
- **#999 GQA flash-decoding retune** — **default-ON** on that branch (`ONNX_GENAI_CUDA_GQA_SPLITS` is only an A/B rollback knob, not an enable flag).

#### Combined glm base greedy decode, graph-ON (GPU6, verified 0 MiB/0% idle, release, median of runs)
| Context | flags-OFF* | COMBINED (all 3) | Δ from flags |
|---|---|---|---|
| short (`--tokens 160 --decode-skip 40`, runs=5) | 202.05 tok/s | **232.10 tok/s** | +14.9% |
| KV≈2048 (`--tokens 2160 --decode-skip 2000`, runs=3) | 168.17 tok/s | **182.11 tok/s** | +8.3% |

\* flags-OFF still includes #999 flash-decoding (default-on); it isolates only the two opt-in flags (fp16 GEMV + lm_head cuBLASLt). The pre-everything main baseline is ~199 tok/s short (#992). The flags help less at KV2048 because attention is KV-dependent and dilutes the GEMV/lm_head share as context grows.

CUDA graph: `captures` intact, **`fallbacks=0`** in every combined run (short and KV2048). The three configs coexist cleanly — no interaction bug, capture preserved. This is the config we would ship (fp16 GEMV + lm_head cuBLASLt opt-in, flash-decode default).

#### Honest native-vs-ORT (base, non-speculative, equal conditions)
| Comparison | native | ORT | gap |
|---|---|---|---|
| **graph-on vs graph-on (north-star)** | 232.10 | ~250-252 (certified) | **1.08× behind (92.8% of ORT)** |
| graph-on native vs graph-off ORT | 232.10 | 197.07 (fresh #999, same GPU class) | **1.18× AHEAD** |

**Verdict: not caught yet on the north-star (graph-on vs graph-on), but close — gap closed from 1.30× (192 tok/s, pre-program) to 1.08×.** We beat ORT graph-off comfortably, but ORT's own best is graph-on ~250, and equal-conditions requires graph-on vs graph-on.

##### ORT harness note (independently re-confirmed this session)
I attempted a fresh ORT graph-on/off run on the same GPU and could NOT load the glm model in any locally-installed ORT: 1.27.0 and 1.28.0 (and the 1.25 baseline Release) all reject `GroupQueryAttention` with `Unrecognized attribute: rotary_embedding_dim` — the glm export targets an ORT newer than any build present here. (Also hit the cu12 vs cu13 provider-lib split: ORT 1.28's `libonnxruntime_providers_cuda.so` needs `libcublasLt.so.12`.) This corroborates the documented "ORT graph-on fails in this harness"; the certified ~250 remains the comparator, and #999's fresh 197.07 graph-off (same GPU class, 2h earlier) is the reproducible ORT floor.

#### Residual per-op dominator (ONNX_GENAI_PROFILE_OPS=1, combined stack, final decode pass)
Eager per-op mode (graph-off, fixed launch overhead compresses KV-scaling), relative shares at ~KV520 and ~KV2080 — consistent:
| op_type | ~KV512 | ~KV2048 | note |
|---|---|---|---|
| **GroupQueryAttention** | **39.3%** | **38.6%** | #1 residual; KV-dependent, grows with context in real graph-on timing (#999: attn core 442→1315 µs KV512→2048) |
| **MatMulNBits (int4 GEMV)** | 34.8% | 35.0% | #2; proven at its occupancy wall (three convergent negatives: depth-4, cp.async #1000, PORT-1 #1002) |
| SkipSimplifiedLayerNormalization | 12.2% | 12.1% | |
| Mul | 5.2% | 5.4% | |
| Split | 2.7% | 2.9% | |
| **MatMul (lm_head)** | **1.0%** | **1.0%** | #991 cuBLASLt shrank it from a ~1.5× hotspot to noise ✅ |

**Names the NEXT lever:** **attention (GQA), especially long-context.** It is the #1 residual op, it is the term that drops us from 232 (short) to 182 (KV2048), and #999 already improved it but explicitly flagged remaining headroom ("still latency/grid-limited rather than bandwidth-bound; future work should reduce/combine the per-layer merge/prep launches"). The int4 GEMV (#2) is at its measured occupancy wall — no cheap further win there. lm_head is done.

#### Bottom line for Justin's north-star
- Combined base decode graph-on: **232 tok/s short / 182 tok/s KV2048**, fallbacks=0, capture-safe.
- **Gap to ORT base (graph-on) closed 1.30× → 1.08×; not yet caught.**
- The remaining ~8% is **attention-bound, not GEMV-bound** — the next investment should be long-context GQA decode, not more int4-GEMV micro-opt.

#### Repro
```
git worktree add .worktrees/deckard-combined -b squad/combined-stack-measurement origin/main
cd .worktrees/deckard-combined
git merge origin/squad/int4-gemv-fp16-mixed origin/squad/attn-decode-flashdecoding   # clean
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native
source /home/justinchu/onnx-genai/.cudaenv.sh
CUDA_VISIBLE_DEVICES=<idle> ONNX_GENAI_GEMV_FP16=1 ONNX_GENAI_LMHEAD_CUBLASLT=1 \
  target/release/profile_native --model /home/justinchu/glm-e2e-artifacts/glm-4-9b-int4-cuda \
  --ep cuda --steady --tokens 160 --decode-skip 40 --runs 5 --warmups 1
```
