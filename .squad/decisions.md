# Decisions — live standing directives
Last consolidated: 2026-08-20T05:50:19+00:00 (Scribe Phase-4 kernel optimization batch: 5 inbox drops processed; live narrative compacted to .squad/decisions-archive/2026-08.md; pre-check live size 33411 bytes.)


Standing governance rules and active directives. Full narrative is archived; keep this file to current decisions plus durable rules.

## Ledger health rule

Archive by SIZE, not age. Age-only archiving can silently no-op during high-volume campaigns because most entries are recent. When the live ledger crosses the spawn-budget gate, preserve full history in an archive and keep `decisions.md` to standing directives, active decisions, and pointers. Assemble from inbox drops, dedupe, then delete merged drops; leave `decisions/inbox/README.md`.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and `.squad/decisions/archive/`. The detailed 2026-08 decode-vs-ORT, graph-capture, Qwen3.8 conversion, and Phase-4 kernel optimization narratives are preserved in `.squad/decisions-archive/2026-08.md` (latest compaction/merge: `2026-08-20T05:50:19+00:00`).

## Current decode campaign standing

Native int4 decode leads stock ORT CUDA EP in production because native owns full-decode CUDA-graph capture and device-resident sampling on dynamic-KV int4 paths that ORT cannot capture. Equalizable eager-vs-eager dense-model results show ORT kernels are comparable or sometimes faster; do not frame the dense wins as intrinsic per-kernel superiority.

For DeepSeek-V2-Lite int4 QMoE the finding is stronger and different: stock ORT CUDA EP cannot place `com.microsoft::QMoE` on GPU, so its run falls back through CPU EP for 26 MoE layers. Report this as a GPU-vs-CPU-fallback capability gap, not a per-kernel multiplier.

Batch-1 byte-identical single-kernel/fusion work is mined out for now. Further wins should come from structural capabilities (capture, device token loop, higher arithmetic intensity, model support) or explicitly default-off experimental levers.

## Native-vs-ORT fairness rule

Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output. If one engine crashes, rejects the graph, runs CPU, disables CUDA graphs, or uses a different weight file/config, report a capability/config gap rather than a throughput multiplier. For ORT-genai decode, verify CUDA provider and share-buffer/cuda-graph fast path are active before quoting tok/s.

## Benchmark and profiling discipline

Separate measured, estimated, and projected. Same-run PR-vs-base deltas beat absolute numbers under shared-host load. For CUDA-graph decode, `ONNX_GENAI_PROFILE_OPS=1` is a host/eager dispatch view and can mis-rank kernels; use `nsys --cuda-graph-trace=node` for kernel mix and `ncu --graph-profiling node --set full` for stall mechanism. A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.

## Numerics and portability discipline

Default-on CUDA decode optimizations must be portable or explicitly arch-gated with byte-identical fallback. Token byte-identity is an argmax stability claim, not a numeric invariant; numeric changes need oracle/tolerance justification. Preserve Rule 11: unsupported devices must fall back without behavior loss. Env knobs used for A/B must be documented, deterministic under capture, and not hide default regressions.

For int4 GEMV/QMoE reductions, CPU bit-identity is not an oracle when accumulation order differs. Correctness is bounded agreement with an independent higher-precision reference plus deterministic backend output and explicit golden rationale.

## Testing and CI standing directives

- `cargo test --workspace` silently truncates on failure; use `--no-fail-fast` for full-suite evidence.
- Run new tests in isolation before trusting full-suite green. Assert on what code did, not summaries.
- An agent self-report is not evidence; verify with code, command output, and tests.
- Reviewer lockout is enforced: authors do not revise their own rejected artifacts.
- CI is asynchronous; required local targeted tests/builds/hardware probes remain blocking, but do not idle solely waiting for CI.
- Never commit `.squad/` files to external repos; if that happens, purge history rather than only deleting in a follow-up commit.

## CUDA availability directive

The primary Windows development box has a working RTX 4060 CUDA path even though `nvcc`, `CUDA_PATH`, and default PATH probes may fail. A complete CUDA 13 runtime is available under anaconda site-packages; agents must distinguish absent from misconfigured before claiming CUDA is unavailable. On that box, add the `cu13` and `cudnn` bin directories to PATH and build with `--features native-cuda`.

---

### 2026-08-20: Phase-4 Qwen3.8-27B int4 decode optimization summary — #1557/#1561/#1562 merged

**By:** Batty, Deckard, Sebastian; consolidated by Scribe  
**Timestamp:** 2026-08-20T05:50:19+00:00

Phase-4 corrected the Qwen3.8-27B int4 decode thesis and banked three merged wins on `origin/main`:

- **#1557 bf16 device-argmax (Batty):** q38 moved from **52.6 → 54.6 tok/s** (~+3.8%) by supporting bf16 logits on the device greedy path. This removed the host-argmax crash/serialization, but proved device token-loop/argmax was only a modest prerequisite, not the dominant lever.
- **#1561 asymmetric int4 block-32 split-K occupancy gate (Batty):** removed the large-N zero-point split-K mis-route; q38 standalone reached ~**59.5 tok/s** (+9% on the #1557 base) with mary unchanged.
- **#1562 Gated-DeltaNet L2-normalize glue fusion (Deckard):** rewrote 96 Q/K L2-normalize chains/token from ReduceSumSquare→Sqrt→Div into byte-faithful fused LpNormalization routing, cutting roughly **288 → 96 launches/token**. Standalone q38 gain was ~+2.4%; mary was byte-faithful.
- **Sebastian integration lock:** clean current-main A/B of #1561+#1562 on top of #1557 measured q38 **54.56 → 61.32 tok/s (+12.4%)** and mary **58.81 → 60.59 tok/s (+3.0%)**. mary remained byte-identical. q38 stream diffs are intrinsic razor-thin argmax tie flips from split-K GEMV accumulation, not a correctness regression.

Honest correction: Qwen3.8 decode is **forward int4 M=1 GEMV latency/launch-bound at ~26% of HBM roofline**, not host-argmax bound. The device token-loop is now a proven non-lever for the 150 tok/s target. At **61.3 tok/s**, q38 remains **~2.45× short** of 150 tok/s. Standing next lever: make int4 M=1 GEMV move toward bandwidth-bound execution (occupancy/arithmetic intensity/dequant-in-register). Unresolved blocker: split-K GEMV nondeterminism still prevents a stable q38 golden token oracle.

### 2026-08-20: Qwen3.8-27B conversion/status pointer

Sapper's GGUF→ONNX int4 conversion produced a coherent runnable artifact at `/home/justinchu/qwen38-27b-int4-cuda`; full conversion details and native-EP gaps were archived to `.squad/decisions-archive/2026-08.md`. Active follow-ups remain: native MatMulNBits small-N (`N=48`) bounds handling to remove the dense workaround, and continued CUDA-EP decode optimization focused on int4 M=1 GEMV rather than N=48 dequant traffic.

### 2026-08-20: Phase-4 inbox details archived

### 2026-08-19: gpt-oss-20b QMoE moat RETRACTED — export artifact identical to DeepSeek; honest native lead ~1.5×

**By:** Deckard

**What:** "446× moat / ORT can't run gpt-oss-20b QMoE on GPU" RETRACTED. Root cause: shipped Foundry fp32-activation export → ORT CUDA QMoE/GQA registered only for fp16/bf16 → 48 nodes fall to CPU EP (24 QMoE + 24 GQA) + 48 MemcpyFromHost. Identical to DeepSeek-V2-Lite precedent. Fair A/B (native fp32 CUDA-graph vs ORT-CUDA fp16 re-export eager, H200 medians-of-5): native **218/210 tok/s** vs ORT **144/140 tok/s** (short/mid) = **honest lead ~1.5×**, GPU-vs-GPU. Residual gap = native CUDA-graph coverage + on-GPU argmax (ORT backend has no CUDA-graph path). Fix: exporter emit bf16 activations (no ORT kernel change needed — CUDA QMoE already works for fp16/bf16). Note: native's standing #1 lever remains fp16/bf16-activation QMoE kernel for a single shared CUDA-fair export.

---

### 2026-08-19: fuse SSM/linear-attention f16 ReduceSum — GO (PR #1486, merged)

**By:** Gaff

**What:** Routed f16 ReduceSum/ReduceMean on CUDA EP to the existing NVRTC block reduction (fp16 IO, fp32 register accumulation) instead of cuDNN, eliminating the fp32 round-trip on the hybrid SSM decode path. One-line change in `reduce.rs`: cuDNN gate `Float32 | Float16` → `Float32`. Branch `squad/ssm-reduce-fuse`, PR #1486. General: shape/axis/dim agnostic, no model-shape assumptions; bf16 was already on NVRTC path; fp32 unchanged on cuDNN. Capture-safe: NVRTC reduction still captured (reduce_capture_gpu assertion preserved).

**Result (H200 GPU5, qwen3.5-0.8b-hybrid fp16io, medians):** 4.664 → **4.318 ms/tok** (214.4 → **231.6 tok/s, +7.4%**). Byte-identical greedy tokens. captures=4 preserved. GPU tests pass.

---

### 2026-08-19: glue-tail CudaRsqrtFusion — GO (PR #1486, merged)

**By:** Gaff

**What:** Landed `CudaRsqrtFusion` collapsing `Sqrt+Reciprocal` → fused `rsqrt_{dtype}` kernel (post-SSM-reduce-fusion profile ranked this as highest-value clean fusion: Sqrt 1.2% + Reciprocal 1.2%, 36/step). Other candidates rejected: Cast/op_tensor (already eliminated by SSM-reduce fix), RMSNorm+residual (already fused), Transpose/Split (layout-invasive), Mul (scattered graph-structure-dependent). Fusion is bit-identical (reproduces two-kernel rounding), capture-safe, general, gated on single-consumer predicate. PR #1486.

**Result (H200 GPU5, incremental on SSM-reduce):** 4.318 → **4.282 ms/tok** (231.6 → **233.6 tok/s, +0.8%**). sqrt_f16 + reciprocal_f16 GONE in nsys; replaced by rsqrt_f16. Byte-identical.

**Combined PR #1486 (both fusions):** 4.664 → 4.282 ms/tok (**214.4 → 233.6 tok/s, +8.9%**), byte-identical, captures=4 preserved.

---

### 2026-08-19: qwen3.5-hybrid moat is FAIR (GPU-vs-GPU); root-caused to ORT fp32 GQA kernel registration gap

**By:** Wallace

**What:** Confirmed the qwen3.5 moat is NOT native-GPU-vs-ORT-CPU. Root cause of 25-Memcpy graph-block: ORT CUDA EP has NO `float` registration for `GroupQueryAttention` (only MLFloat16/BFloat16); all 6 GQA nodes forced to CPU → 25 MemcpyFromHost/ToHost → CUDA-graph hard-throw. `LinearAttention` and `CausalConvWithState` stay on CUDA (they do register `float`). Fair A/B: native captured **196–206 tok/s** (ctx-flat, pstdev ~0.1) vs ORT-CUDA eager **81→13 tok/s** (decays ~1/ctx, noisy). The fp32 GQA registration gap is ORT-fixable (kernel supports fp32 via mem-efficient path per source comment; rotary path needs float branch too; sub-optimal vs fp16 flash). Upstream issue drafted at `.scratch/ORT_ISSUE_gqa_fp32_cuda.md` (not filed; user has ORT write access).

---

### 2026-08-19: qwen3.5-hybrid fp16 same-graph race — moat does NOT survive; ORT fp16 graph 1.6–1.9× faster

**By:** Wallace

**What:** Fair fp16 A/B (qwen3.5-2b-text, H200 GPU0/1, medians-of-5, native graph vs ORT-genai 0.14.1 `enable_cuda_graph=1`). fp16 export: keep_io_types=False, linear-attn subgraph (756 nodes) blocked at fp32 (ORT LinearAttention CUDA kernel requires float decay input). GQA flips to CUDA, 25 Memcpy drop to 0, ORT captures. Result at ctx 16/1024/4000: ORT fp16 graph **353/352/348 tok/s** (context-flat) vs native fp16 **219/214/182 tok/s** → **ORT 1.6–1.9× faster at all depths**. Native fp16 only +18% over fp32; ORT fp16 +67% over fp32-short. **Conclusion: the qwen3.5 context-scaling moat was a fp32-export artifact — do NOT cite it without "fp32-export-only" qualifier.** Real open work: native's fp16 decode gap (kernel fusion/integration, not a single-kernel ORT win). Fp16 exports at `/home/justinchu/qwen35-{0.8b,2b}-text-fp16/` (scratch).

---

### 2026-08-19: eager crossover generality (dense models) — MIXED result; production capture unaffected

**By:** Wallace

**What:** Dense eager A/B with forced `ONNX_GENAI_CUDA_GRAPH=0` (qwen2.5-0.5b, 1.5b, Phi-4-mini; H200; medians-of-5). Verdict MIXED: **qwen2.5-1.5b 1.19–1.29× native wins; Phi-4-mini ~1.03–1.04× parity; qwen2.5-0.5b 0.82–0.90× native LOSES** (small-model host-dispatch overhead dominates). #1383 defer-eager-sync lever generalizes (+1.5–5.5% on dense eager) but does not flip 0.5b. Production default (auto-capture) unchanged: captured native 1046 vs ORT-eager 666 tok/s (1.57×) on 0.5b. **Do NOT re-open eager-chasing for 0.5b gap.** Methodology note: native auto-captures graph-safe topologies by default; force `ONNX_GENAI_CUDA_GRAPH=0` for true eager measurement.

---

### 2026-08-19: #1474 memory-stack real-hardware verification (8×H200) — 6 deferred-release tests fail; acceptance gate NOT met

**By:** Bryant

**What:** Verified Phase 1–7 memory-stack reconstruction (#1474 checklist) on 8×H200 (driver 580.105.08 / CUDA 13.3 / cuDNN 9.19). tip=`31f3a2dde` (#1468), Phase6=`21370921e` (#1462), base=`1d5ef758e`. CPU suites pass (116/0 memory-host/abi/testplugin; 464/1/458 cuda-memory+ep-cuda; sole failure = pre-existing matmul_nbits fp16 numerics). GPU: tip 886/16/21 vs base 836/27/21. **6 stack-owned GPU tests fail deterministically on real hardware** (were `#[ignore]`'d on dev box, first real execution): all rooted in Phase4/7 async deferred-release queue — `an_injected_external_eager_allocator_replaces_the_built_in_arena`, `a_zero_byte_allocation_is_freed_*`, `a_rolled_back_decommit_*`, `weight_paging::vmm_retained_weight_key*` (×2), `provider::tests::public_constructor_*`. Checks #3 (release path) / #6 (one pool test) / #7 (VMM isolation) all ⚠️. Criterion-10 decode: Phase6 11.28 → Phase7 **11.52 tok/s (+2.1%, no regression)**, VMM ledger clean. Acceptance gate (CUDA suite real-hardware all-green) NOT met; author must fix test harness or async release timing. Comment posted to issue #1474. No PR merged/closed.

---

### 2026-08-19: Fair ORT fp16io baseline — 2.1× downgraded to ~1.2–1.4× ORT-ahead (qwen3.5-0.8b-hybrid)

**By:** Deckard (deckard-8)

Correct device-resident ORT decode harness on `/home/justinchu/qwen35-0.8b-fp16io` (onnxruntime 1.28.0, H200 GPU5, IOBinding, growing-KV, greedy-validated). **ORT eager = ~284–296 tok/s (flat, depth 16→1040).** Node placement: 1019 CUDA / 50 CPU — the 50 CPU nodes are trivial mrope/mask index ops (Gather/Mul/Concat/Cast) placed by cost heuristic, NOT missing kernels. ALL heavy ops (LinearAttention, CausalConvWithState, GQA, MatMulNBits) are on CUDA. **No structural ORT kernel gap** for the qwen3.5-hybrid recurrent arch.

Honest gap: **native 209.8 tok/s (pre-#1486) ÷ ORT 290 = 0.72× (~1.38× ORT-ahead); native 233.6 (post-#1486) ÷ ORT 290 = 0.81× (~1.24× ORT-ahead).** The **496 tok/s and 2.1× figures are RETIRED** — 496 requires a share-buffer genai re-export + cuda-graph, which this fp16io export cannot produce correctly (GQA in-place share causes wrong tokens). ORT cuda-graph on this export is incorrect AND slower (226 tok/s). Cite **~1.2–1.4× ORT-ahead** for qwen3.5-0.8b-hybrid fp16io.

---

### 2026-08-19: Kernel-level attribution of ~24% native-vs-ORT decode gap (qwen3.5-0.8b-hybrid fp16io)

**By:** Deckard (deckard-8)

nsys per-kernel diff, matched steady-state decode. **Gap is GPU-busy kernel time (+1269 µs/step), NOT launch overhead** — more cuda-graph coverage cannot close it. Top attribution:

| bucket | Δ nat−ORT (µs/step) | verdict |
|---|---|---|
| Gated-delta LinearAttention subsystem (core + fp32 cuDNN reduce + data-shuffle + gating chain) | ~700–900 | **#1 lever — fuse** |
| elementwise glue (unfused gating) | +333 | part of #1 |
| data-shuffle (transpose/split/concat) | +300 | part of #1 |
| fp32 cuDNN reduce_tensor | +297 | part of #1 |
| int4 MatMulNBits GEMV | +193 | #4 lever — latency-bound, ORT kernel 33%/call faster |
| LinearAttention core | +185 | folded in #1 |

**Recommendation:** Fuse gated-delta LinearAttention decode path into one fp16 kernel (mirror ORT `LinearAttentionDecodeColKernel`): eliminate fp32 cuDNN ReduceSum, fold transpose/split/concat, fuse gating chain. Est. recovery ~650–900 µs/step → ~250–256 tok/s, closing most of the gap. This is a native fusion project; ORT has no deficiency here. Gaff assigned to implement.

---

### 2026-08-19: LinearAttention fp32 ReduceSum routed off cuDNN → NVRTC (PR #1495)

**By:** Gaff (gaff-14)

Routed the fp32 cuDNN ReduceSum in the gated-delta LinearAttention decode path onto the NVRTC block reduction — fp32 analogue of #1486. Result: 200.95 → 209.71 tok/s (+4.36%), byte-identical, golden lock PASS. Admin-merged (squash, 2026-08-19). Sub-levers #2 (transpose/split/concat layout fold) and #3 (gating chain fusion) deferred as higher-risk graph-level work; a fresh Gaff has been dispatched to attempt them.

---

### 2026-08-19: CudaLinearAttentionGatingFusion — structural pass folds gating chain into kernel epilogue (PR #1496)

**By:** Gaff (gaff-15)

Structural graph-fusion pass `CudaLinearAttentionGatingFusion` folds both standalone gate chains feeding `com.microsoft::LinearAttention` into the kernel epilogue: **beta = Sigmoid(x)** (delta-rule mixing gate; beta slot rewired to pre-Sigmoid value, kernel applies inline) and **decay = exp(neg_exp_A · Softplus(a + dt_bias))** (per-head decay gate; `a`/`dt_bias`/`neg_exp_A` rewired as trailing kernel inputs, kernel recomputes chain). Match is purely structural — op type, single-consumer/no-escape topology, initializer identity, optional identity Cast tail. No shape baked in; fires for any head/layer count. Byte-identity: kernel reproduces each folded op's device function bit-for-bit and rounds through storage dtype at exact standalone-kernel boundaries. Drops ~5 elementwise nodes/layer × 18 layers from the captured decode graph.

**Result (H200, qwen3.5-0.8b fp16io, medians):** 229.4 → **237.4 tok/s (+3.4%)**, byte-identical, golden lock PASS (coordinator re-validated, 32.62 s). Admin-merged (squash, 2026-08-19). Opt-out: `ONNX_GENAI_CUDA_DISABLE_LINATTN_GATING_FUSION=1`. 3 new pass unit tests.

**Sub-lever #2 (layout / L2-norm addressing) — DEFERRED:** Folding transpose/split/concat + L2-norm into kernel addressing requires reproducing ORT `ReduceSum` exact summation order inside the per-thread LA loop (byte-divergence risk) and strided-input plumbing (`supports_strided_input=false`). Deferred pending a dedicated byte-lock harness; should be tackled in isolation.

---

### 2026-08-19: LinearAttention warp-cooperative kernel rewrite — spill eliminated (PR #1503)

**By:** Gaff (gaff-16)

Rewrote the gated-delta `LinearAttention` decode kernel (`linear_attention.rs`) to the warp-cooperative layout: one warp owns each state column with d_k rows distributed across lane registers, replacing the `sc[MAX_D_K=256]` local-memory array (which spilled to local-mem at 56 regs/thread). Two d_k dot-products (`r = Sᵀk`, `o = qᵀS`) became `__shfl_xor` warp reductions, replacing 128-iteration serial loops. ncu: local-load sectors 163,840 → 0, local-store 98,304 → 0, kernel 21,664 → 12,510 ns (−42%), SM util 1.31% → 13.3%, occupancy 12.0% → 21.2%. Admin-merged (squash, 2026-08-19). Opt-out: `ONNX_GENAI_CUDA_DISABLE_LINATTN_WARP_COOP=1`.

**Result (H200, qwen3.5-0.8b fp16io, end-to-end):** +7.4–10.6 tok/s (~240 → ~249 tok/s, +3.2% at median). Native ~249 vs ORT ~290 (~1.16× ORT-ahead, narrowed from ~1.19×).

**CRITICAL FINDING — lock tests assert greedy token-IDs, NOT bit-exact bytes:** The qwen35_*_text_decode_lock tests assert `Vec<u32>` argmax (greedy token-ID sequences), not byte-identical output. Warp-shuffle reduction reorder is ULP-divergent but argmax-stable — qwen3.5-0.8b and qwen3.5-2b lock PASS, GPU oracle-parity suite PASS across GQA/inverse-GQA/key-sharing/per-key-decay/shared-beta/all update rules + new d_k=128/96/130 configs. **De-risk implication:** future ULP-divergent CUDA kernel rewrites (warp reductions, reduction-order changes, etc.) are not barred by the lock gate provided argmax stability is preserved. Validate ≥2 models; do not assume bit-identity.

---

### 2026-08-19: int4 MatMulNBits GEMV occupancy-gated pipe-vs-plain entry (PR #1501)

**By:** Gaff (gaff-16)

Occupancy gate for int4 MatMulNBits GEMV entry selection. Routing rule: if `ceil(N / cols_per_cta) >= mp_count * 32` (well-occupied, e.g. LM head N=248320) → plain/low-register entry (−14% latency: 98.8→85.0 µs). Otherwise → prefetch-pipelined entry (grid-starved projections already at byte-identical floor ~4.3–4.7 µs/call, under ORT's 4733 ns). Keys on N, launch-width, and SM count only — capture-safe. Byte-identical golden lock PASS. Coordinator independently re-validated on GPU0 (33.06 s). Admin-merged (squash, 2026-08-19). Opt-out: `ONNX_GENAI_GEMV_PIPE_WELLOCC=0`.

**Result (H200, qwen3.5-0.8b fp16io, end-to-end medians):** +1.3–1.6 tok/s (~242→243.5 tok/s). Grid-starved projections at byte-identical floor; split-K would raise occupancy but reorders fp32 reduction (not byte-identical — barred under the golden-lock rule).

---

### 2026-08-19: data-shuffle lever investigation — attribution refuted by captured-graph honest floor

**By:** Deckard (deckard-8) + Gaff (gaff-16)

**Attribution:** After PR #1503 on current `origin/main` (`4bbd9152c`), Deckard re-measured qwen3.5-0.8b-hybrid fp16io on H200: native captured decode **253.9 tok/s** (3.939 ms/tok, 8 captures, 1008 replays, fallbacks=0) vs ORT eager **279.1 tok/s** (3.583 ms/tok), leaving **1.099x ORT-ahead / ~356 us per step**. The eager kernel-count attribution ranked `data_shuffle` as apparent #1: native 341.2 us/step vs ORT 52.3 us/step (**+288.9 us/step**) across transpose/split/gather/concat/where copy-like kernels; int4 GEMV was #2 (+176.1 us/step), and linear attention residual #3 (+58.7 us/step).

**Attempt:** Gaff tested the data-shuffle fusion lever on branch `squad/shuffle-fusion` by using CUDA `TransposeKernel::view_outputs()` for seq_len=1 no-op transposes, gated during the experiment by `ONNX_GENAI_CUDA_DISABLE_SHUFFLE_FUSION=1`. The elision path fired correctly, but zero-copy view installation mutates device-buffer lifetimes and is illegal during CUDA graph capture.

**Correction / decision:** Deckard's **+289 us/step data-shuffle lever is refuted for the captured decode path**. The standalone eager kernel count is real, but its cost attribution is an eager-mode/nsys artifact: inside captured replay, the 60 no-op transposes are ~2 KB copies and cost roughly **~40 ns/step** (~0.001% of a ~4 ms step). Installing views mid-capture aborts graph recording, quarantines Transpose, and fragments the decode graph from **16 captured segments / 15 eager seams** to **70 captured segments / 69 eager seams** — a regression, not a buildable win.

**Outcome:** No code shipped. `movement.rs` was reverted, origin/main behavior is unchanged, and docs-only PR **#1511** was closed by the coordinator without merge. Do **not** treat data-shuffle fusion as the top buildable lever. Future attribution for captured decode must use captured-replay profiling (`ncu --graph-profiling node` or `nsys --cuda-graph-trace=node`) rather than eager standalone kernel counts.


---

### 2026-08-20: Corrected captured-replay attribution — honest native ceiling is the batch=1 launch/latency floor

**By:** Gaff

**What:** Re-ran qwen3.5-0.8b-hybrid fp16io native-vs-ORT decode attribution using captured-replay-aware tooling only (`nsys --cuda-graph-trace=node`, `cuda_gpu_kern_sum`, and `ncu --graph-profiling node`) after Gaff refuted the data-shuffle hypothesis. Native captured replay measured ~249.6 tok/s (run-to-run 249–254; later campaign baseline ~254) vs ORT eager 279.1 tok/s, with GPU-busy time **native 2516 µs/step vs ORT 1899 µs/step (+617 µs)** and near-identical kernel counts (**795 vs 793**). Every hot native kernel is occupancy/latency-bound rather than bandwidth/compute-roofline-bound: main int8/int4 GEMV ~22% occupancy, `transpose_bytes` ~12% occupancy and near-zero bytes moved, and post-#1503 `linear_attention_f16_coop` ~21.5% occupancy.

**Why:** The previous data-shuffle attribution was real GPU timeline cost but not a buildable lever. The transposes are tiny seq_len=1 floor kernels; eliding them mid-capture fragments CUDA graph replay and regresses. Deckard retired data-shuffle and int64-index chasing, identified **int4/int8 GEMV occupancy tuning** as the single remaining credible lever, and set the honest practical ceiling for this arch at roughly **260–270 tok/s**: ORT parity minus the small-batch launch/latency floor tax.

---

### 2026-08-20: Symmetric int8 MatMulNBits split-K grid-fill landed — sixth clean token-stable win (PR #1516, merged)

**By:** Gaff

**What:** PR **#1516** landed the final buildable GEMV lever on `origin/main` (`ec695d897`): symmetric int8 `MatMulNBits` GEMV now uses the existing split-K/grid-fill path for grid-starved decode projections, keyed only on `K`, `N`, bits, zero-point presence, and live SM count. The bias-fusion sub-lever was a no-op: native already has `CudaMatMulNBitsBiasFusion`, and all Qwen3.5 exports checked here are bias-free. The shipped change is general, capture-safe, and opt-out guarded by `ONNX_GENAI_CUDA_DISABLE_INT8_SYMMETRIC_SPLITK=1`.

**Result:** H200 qwen3.5-0.8b-hybrid fp16io paired A/B improved **254.0 → 263.8 tok/s (+3.9%, +9.8 tok/s)**. ncu confirmed the targeted grid-starved int8 projections improved: N=1024 **11424 → 8049 ns** with occupancy **11.6% → 23.4%**; N=2048 **6065 → 4877 ns** with occupancy **22.2% → 45.4%**. Golden token-ID locks passed on qwen3.5-0.8b and qwen3.5-2b, and the coordinator independently re-validated both as `ok`.

**Campaign state:** The decode campaign now has **six clean token-identical/token-stable wins**: #1486, #1495, #1496, #1501, #1503, and #1516. Native is now ~**263.8 tok/s** vs ORT **279.1 tok/s** (**~1.057× ORT-ahead**), narrowed from ~1.24× at session start. Per Deckard's corrected attribution, the int4/int8 GEMV lever was the **last genuine buildable lever**; the remaining gap is the batch=1 0.8b launch/latency floor, not an unmined roofline or data-shuffle opportunity.
**What:** Routed f16 ReduceSum/ReduceMean on CUDA EP to the existing NVRTC block reduction (fp16 IO, fp32 register accumulation) instead of cuDNN, eliminating the fp32 round-trip on the hybrid SSM decode path. One-line change in `reduce.rs`: cuDNN gate `Float32 | Float16` → `Float32`. Branch `squad/ssm-reduce-fuse`, PR #1486. General: shape/axis/dim agnostic, no model-shape assumptions; bf16 was already on NVRTC path; fp32 unchanged on cuDNN. Capture-safe: NVRTC reduction still captured (reduce_capture_gpu assertion preserved).

**Result (H200 GPU5, qwen3.5-0.8b-hybrid fp16io, medians):** 4.664 → **4.318 ms/tok** (214.4 → **231.6 tok/s, +7.4%**). Byte-identical greedy tokens. captures=4 preserved. GPU tests pass.

---

### 2026-08-19: glue-tail CudaRsqrtFusion — GO (PR #1486, merged)

**By:** Gaff

**What:** Landed `CudaRsqrtFusion` collapsing `Sqrt+Reciprocal` → fused `rsqrt_{dtype}` kernel (post-SSM-reduce-fusion profile ranked this as highest-value clean fusion: Sqrt 1.2% + Reciprocal 1.2%, 36/step). Other candidates rejected: Cast/op_tensor (already eliminated by SSM-reduce fix), RMSNorm+residual (already fused), Transpose/Split (layout-invasive), Mul (scattered graph-structure-dependent). Fusion is bit-identical (reproduces two-kernel rounding), capture-safe, general, gated on single-consumer predicate. PR #1486.

**Result (H200 GPU5, incremental on SSM-reduce):** 4.318 → **4.282 ms/tok** (231.6 → **233.6 tok/s, +0.8%**). sqrt_f16 + reciprocal_f16 GONE in nsys; replaced by rsqrt_f16. Byte-identical.

**Combined PR #1486 (both fusions):** 4.664 → 4.282 ms/tok (**214.4 → 233.6 tok/s, +8.9%**), byte-identical, captures=4 preserved.

---

### 2026-08-19: qwen3.5-hybrid moat is FAIR (GPU-vs-GPU); root-caused to ORT fp32 GQA kernel registration gap

**By:** Wallace

**What:** Confirmed the qwen3.5 moat is NOT native-GPU-vs-ORT-CPU. Root cause of 25-Memcpy graph-block: ORT CUDA EP has NO `float` registration for `GroupQueryAttention` (only MLFloat16/BFloat16); all 6 GQA nodes forced to CPU → 25 MemcpyFromHost/ToHost → CUDA-graph hard-throw. `LinearAttention` and `CausalConvWithState` stay on CUDA (they do register `float`). Fair A/B: native captured **196–206 tok/s** (ctx-flat, pstdev ~0.1) vs ORT-CUDA eager **81→13 tok/s** (decays ~1/ctx, noisy). The fp32 GQA registration gap is ORT-fixable (kernel supports fp32 via mem-efficient path per source comment; rotary path needs float branch too; sub-optimal vs fp16 flash). Upstream issue drafted at `.scratch/ORT_ISSUE_gqa_fp32_cuda.md` (not filed; user has ORT write access).

---

### 2026-08-19: qwen3.5-hybrid fp16 same-graph race — moat does NOT survive; ORT fp16 graph 1.6–1.9× faster

**By:** Wallace

**What:** Fair fp16 A/B (qwen3.5-2b-text, H200 GPU0/1, medians-of-5, native graph vs ORT-genai 0.14.1 `enable_cuda_graph=1`). fp16 export: keep_io_types=False, linear-attn subgraph (756 nodes) blocked at fp32 (ORT LinearAttention CUDA kernel requires float decay input). GQA flips to CUDA, 25 Memcpy drop to 0, ORT captures. Result at ctx 16/1024/4000: ORT fp16 graph **353/352/348 tok/s** (context-flat) vs native fp16 **219/214/182 tok/s** → **ORT 1.6–1.9× faster at all depths**. Native fp16 only +18% over fp32; ORT fp16 +67% over fp32-short. **Conclusion: the qwen3.5 context-scaling moat was a fp32-export artifact — do NOT cite it without "fp32-export-only" qualifier.** Real open work: native's fp16 decode gap (kernel fusion/integration, not a single-kernel ORT win). Fp16 exports at `/home/justinchu/qwen35-{0.8b,2b}-text-fp16/` (scratch).

---

### 2026-08-19: eager crossover generality (dense models) — MIXED result; production capture unaffected

**By:** Wallace

**What:** Dense eager A/B with forced `ONNX_GENAI_CUDA_GRAPH=0` (qwen2.5-0.5b, 1.5b, Phi-4-mini; H200; medians-of-5). Verdict MIXED: **qwen2.5-1.5b 1.19–1.29× native wins; Phi-4-mini ~1.03–1.04× parity; qwen2.5-0.5b 0.82–0.90× native LOSES** (small-model host-dispatch overhead dominates). #1383 defer-eager-sync lever generalizes (+1.5–5.5% on dense eager) but does not flip 0.5b. Production default (auto-capture) unchanged: captured native 1046 vs ORT-eager 666 tok/s (1.57×) on 0.5b. **Do NOT re-open eager-chasing for 0.5b gap.** Methodology note: native auto-captures graph-safe topologies by default; force `ONNX_GENAI_CUDA_GRAPH=0` for true eager measurement.

---

### 2026-08-19: #1474 memory-stack real-hardware verification (8×H200) — 6 deferred-release tests fail; acceptance gate NOT met

**By:** Bryant

**What:** Verified Phase 1–7 memory-stack reconstruction (#1474 checklist) on 8×H200 (driver 580.105.08 / CUDA 13.3 / cuDNN 9.19). tip=`31f3a2dde` (#1468), Phase6=`21370921e` (#1462), base=`1d5ef758e`. CPU suites pass (116/0 memory-host/abi/testplugin; 464/1/458 cuda-memory+ep-cuda; sole failure = pre-existing matmul_nbits fp16 numerics). GPU: tip 886/16/21 vs base 836/27/21. **6 stack-owned GPU tests fail deterministically on real hardware** (were `#[ignore]`'d on dev box, first real execution): all rooted in Phase4/7 async deferred-release queue — `an_injected_external_eager_allocator_replaces_the_built_in_arena`, `a_zero_byte_allocation_is_freed_*`, `a_rolled_back_decommit_*`, `weight_paging::vmm_retained_weight_key*` (×2), `provider::tests::public_constructor_*`. Checks #3 (release path) / #6 (one pool test) / #7 (VMM isolation) all ⚠️. Criterion-10 decode: Phase6 11.28 → Phase7 **11.52 tok/s (+2.1%, no regression)**, VMM ledger clean. Acceptance gate (CUDA suite real-hardware all-green) NOT met; author must fix test harness or async release timing. Comment posted to issue #1474. No PR merged/closed.

---

### 2026-08-19: Fair ORT fp16io baseline — 2.1× downgraded to ~1.2–1.4× ORT-ahead (qwen3.5-0.8b-hybrid)

**By:** Deckard (deckard-8)

Correct device-resident ORT decode harness on `/home/justinchu/qwen35-0.8b-fp16io` (onnxruntime 1.28.0, H200 GPU5, IOBinding, growing-KV, greedy-validated). **ORT eager = ~284–296 tok/s (flat, depth 16→1040).** Node placement: 1019 CUDA / 50 CPU — the 50 CPU nodes are trivial mrope/mask index ops (Gather/Mul/Concat/Cast) placed by cost heuristic, NOT missing kernels. ALL heavy ops (LinearAttention, CausalConvWithState, GQA, MatMulNBits) are on CUDA. **No structural ORT kernel gap** for the qwen3.5-hybrid recurrent arch.

Honest gap: **native 209.8 tok/s (pre-#1486) ÷ ORT 290 = 0.72× (~1.38× ORT-ahead); native 233.6 (post-#1486) ÷ ORT 290 = 0.81× (~1.24× ORT-ahead).** The **496 tok/s and 2.1× figures are RETIRED** — 496 requires a share-buffer genai re-export + cuda-graph, which this fp16io export cannot produce correctly (GQA in-place share causes wrong tokens). ORT cuda-graph on this export is incorrect AND slower (226 tok/s). Cite **~1.2–1.4× ORT-ahead** for qwen3.5-0.8b-hybrid fp16io.

---

### 2026-08-19: Kernel-level attribution of ~24% native-vs-ORT decode gap (qwen3.5-0.8b-hybrid fp16io)

**By:** Deckard (deckard-8)

nsys per-kernel diff, matched steady-state decode. **Gap is GPU-busy kernel time (+1269 µs/step), NOT launch overhead** — more cuda-graph coverage cannot close it. Top attribution:

| bucket | Δ nat−ORT (µs/step) | verdict |
|---|---|---|
| Gated-delta LinearAttention subsystem (core + fp32 cuDNN reduce + data-shuffle + gating chain) | ~700–900 | **#1 lever — fuse** |
| elementwise glue (unfused gating) | +333 | part of #1 |
| data-shuffle (transpose/split/concat) | +300 | part of #1 |
| fp32 cuDNN reduce_tensor | +297 | part of #1 |
| int4 MatMulNBits GEMV | +193 | #4 lever — latency-bound, ORT kernel 33%/call faster |
| LinearAttention core | +185 | folded in #1 |

**Recommendation:** Fuse gated-delta LinearAttention decode path into one fp16 kernel (mirror ORT `LinearAttentionDecodeColKernel`): eliminate fp32 cuDNN ReduceSum, fold transpose/split/concat, fuse gating chain. Est. recovery ~650–900 µs/step → ~250–256 tok/s, closing most of the gap. This is a native fusion project; ORT has no deficiency here. Gaff assigned to implement.

---

### 2026-08-19: LinearAttention fp32 ReduceSum routed off cuDNN → NVRTC (PR #1495)

**By:** Gaff (gaff-14)

Routed the fp32 cuDNN ReduceSum in the gated-delta LinearAttention decode path onto the NVRTC block reduction — fp32 analogue of #1486. Result: 200.95 → 209.71 tok/s (+4.36%), byte-identical, golden lock PASS. Admin-merged (squash, 2026-08-19). Sub-levers #2 (transpose/split/concat layout fold) and #3 (gating chain fusion) deferred as higher-risk graph-level work; a fresh Gaff has been dispatched to attempt them.

---

### 2026-08-19: CudaLinearAttentionGatingFusion — structural pass folds gating chain into kernel epilogue (PR #1496)

**By:** Gaff (gaff-15)

Structural graph-fusion pass `CudaLinearAttentionGatingFusion` folds both standalone gate chains feeding `com.microsoft::LinearAttention` into the kernel epilogue: **beta = Sigmoid(x)** (delta-rule mixing gate; beta slot rewired to pre-Sigmoid value, kernel applies inline) and **decay = exp(neg_exp_A · Softplus(a + dt_bias))** (per-head decay gate; `a`/`dt_bias`/`neg_exp_A` rewired as trailing kernel inputs, kernel recomputes chain). Match is purely structural — op type, single-consumer/no-escape topology, initializer identity, optional identity Cast tail. No shape baked in; fires for any head/layer count. Byte-identity: kernel reproduces each folded op's device function bit-for-bit and rounds through storage dtype at exact standalone-kernel boundaries. Drops ~5 elementwise nodes/layer × 18 layers from the captured decode graph.

**Result (H200, qwen3.5-0.8b fp16io, medians):** 229.4 → **237.4 tok/s (+3.4%)**, byte-identical, golden lock PASS (coordinator re-validated, 32.62 s). Admin-merged (squash, 2026-08-19). Opt-out: `ONNX_GENAI_CUDA_DISABLE_LINATTN_GATING_FUSION=1`. 3 new pass unit tests.

**Sub-lever #2 (layout / L2-norm addressing) — DEFERRED:** Folding transpose/split/concat + L2-norm into kernel addressing requires reproducing ORT `ReduceSum` exact summation order inside the per-thread LA loop (byte-divergence risk) and strided-input plumbing (`supports_strided_input=false`). Deferred pending a dedicated byte-lock harness; should be tackled in isolation.

---

### 2026-08-19: int4 MatMulNBits GEMV occupancy-gated pipe-vs-plain entry (PR #1501)

**By:** Gaff (gaff-16)

Occupancy gate for int4 MatMulNBits GEMV entry selection. Routing rule: if `ceil(N / cols_per_cta) >= mp_count * 32` (well-occupied, e.g. LM head N=248320) → plain/low-register entry (−14% latency: 98.8→85.0 µs). Otherwise → prefetch-pipelined entry (grid-starved projections already at byte-identical floor ~4.3–4.7 µs/call, under ORT's 4733 ns). Keys on N, launch-width, and SM count only — capture-safe. Byte-identical golden lock PASS. Coordinator independently re-validated on GPU0 (33.06 s). Admin-merged (squash, 2026-08-19). Opt-out: `ONNX_GENAI_GEMV_PIPE_WELLOCC=0`.

**Result (H200, qwen3.5-0.8b fp16io, end-to-end medians):** +1.3–1.6 tok/s (~242→243.5 tok/s). Grid-starved projections at byte-identical floor; split-K would raise occupancy but reorders fp32 reduction (not byte-identical — barred under the golden-lock rule).
Scribe merged and archived detailed inbox drops for `sebastian-q38-benchmark.md`, `batty-bf16-device-argmax.md`, `batty-gemv27b.md`, `deckard-ssm-glue-fusion.md`, and `sebastian-p4-integration-revalidation.md`. The live ledger keeps the durable summary above; full per-agent narratives are in `.squad/decisions-archive/2026-08.md` under `Decision inbox details merged @ 2026-08-20T05:50:19+00:00`.
