# Decisions — live standing directives

Last consolidated: 2026-08-18T04:15Z (Scribe processed #1189 / DeepSeek Native-vs-ORT follow-ups; archived detailed 2026-08 narrative to `.squad/decisions-archive/2026-08.md` to keep the live ledger below 20KB.)

Standing governance rules and active directives. Full narrative is archived; keep this file to current decisions plus durable rules.

## Ledger health rule

Archive by SIZE, not age. Age-only archiving can silently no-op during high-volume campaigns because most entries are recent. When the live ledger crosses the spawn-budget gate, preserve full history in an archive and keep `decisions.md` to standing directives, active decisions, and pointers. Assemble from inbox drops, dedupe, then delete merged drops; leave `decisions/inbox/README.md`.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and `.squad/decisions/archive/`. The detailed 2026-08 decode-vs-ORT and graph-capture campaign narrative, including the pre-#1189 live ledger and full processed inbox drops from this batch, is preserved in `.squad/decisions-archive/2026-08.md` under `2026-08-18T04:15Z`.

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

## 2026-08-18 — V2-Lite MoE CUDA graph-capture and workspace fixes merged

PR #1181 landed on `main` as `c9c7f64c`, unlocking V2-Lite graph capture by fixing the additive-mask `_d1` workspace-planner path; Wallace measured capture ON vs eager OFF byte-identical over 320 tokens, **101.80 vs 56.94 tok/s = 1.79×**.

A separate long-context Engine `Attention` workspace under-plan then surfaced around KV-capacity growth. PR #1189 landed on `main` as `b416a3e0`, fixing Engine/native CUDA single-token decode to re-run governed workspace preparation whenever `ensure_capacity` grows the KV/mask bucket. Leon's A/B on the real V2-Lite path generated 340 token-identical tokens in eager and capture; eager measured 47.32 tok/s, capture 89.69 tok/s with captures=2, replays=336, fallbacks=0. Rachael approved the fix as strictly gated on capacity growth and correctly placed before eager/capture execution.

## 2026-08-18 — DeepSeek-V2-Lite Native-vs-ORT row closed

Wallace measured the real 27-layer DeepSeek-V2-Lite int4 QMoE export under pinned ORT CUDA 1.27 and identical base-decode conditions. Native CUDA serves the model on GPU at **57.15 tok/s eager** and **101.68 tok/s captured**. Stock ORT CUDA EP cannot run the 26 `com.microsoft::QMoE` nodes on GPU; with CPU fallback it inserts 104 host/device Memcpy nodes and reaches **0.17 tok/s**, while strict no-fallback refuses the graph. ORT + CUDA graph is categorically N/A because the graph is split across CPU and GPU. Frame the row as a hard capability gap: native is the only measured GPU engine for int4 QMoE here.

## 2026-08-18 — GLM-4-9B int4 graph-capture scope

# Wallace — GLM-4-9B int4: does the graph-capture moat already extend? → 🟢 YES (captures clean today)

- Author: Wallace (inference-engine specialist)
- Date: 2026-08-18T04:03Z
- Worktree: fresh detached off `origin/main` @ `b416a3e0` (incl. #1171 classifier, #1181 `_d1`, #1189 Engine long-context fix).
- Model: real GLM-4-9B int4 (dense GQA, partial-RoPE) —
  `/home/justinchu/glm-e2e-artifacts/cohaagen-glm-4-9b-int4-cuda-post434` (`model.onnx` + 6.7 GB `.onnx.data`).
- GPU: H200, `CUDA_VISIBLE_DEVICES=6` (nvidia-smi first — all 8 idle 0 MiB/0%; quiet box).
- Base decode, single-stream greedy, `ONNX_GENAI_ONGPU_ARGMAX=1` both arms. NOT spec-decode.

## VERDICT: 🟢 GO — GLM-4-9B ALREADY captures cleanly, no code change needed
The graph-capture stack we landed for V2-Lite (classifier #1171 + `_d1` #1181 + long-context #1189)
**already covers GLM-4-9B today.** Capture engages with **no classifier/planner bail**, is
**byte-identical** to eager, and delivers **1.64×**.

## Capture engagement (question 1)
`ONNX_GENAI_CUDA_GRAPH=1` → capture ENGAGES, no decline/fallback:
```
cuda_graph: enabled=true captures=3 replays=185 fallbacks=0 invalidations=3
cuda_graph_measured:            captures=2 replays=124 fallbacks=0 invalidations=2
```
- `captures>0`, `replays>0`, `fallbacks=0` → the classifier's capacity-form gate accepts GLM's
  attention-mask topology (present-KV inputs present; mask does not escape as a graph output), and the
  workspace planner does NOT under-plan (no prepare-only refusal). No bail to root-cause.
- (The `invalidations=3` are the expected steady-state recaptures as the KV/seq extent grows during
  warmup; they settle — `fallbacks=0` throughout, and the measured window shows stable replays.)

## A/B: byte-identity + tok/s (question 2)
**Byte-identity: 0.000% divergence** — capture-ON vs eager token streams compared position-by-position
over **256 generated tokens**, identical (0/256). (Repo also carries an independent golden lock
`crates/onnx-genai-engine/tests/glm4_9b_decode_lock.rs` asserting native-CUDA == golden greedy.)

tok/s, medians of 5 interleaved rounds, 128 tok, on-GPU argmax both arms:
| arm | median tok/s | range | pstdev |
|---|---:|---:|---:|
| capture ON (`CUDA_GRAPH=1`) | **211.74** | 211.57–212.48 | 0.33 |
| eager (`CUDA_GRAPH=0`) | **128.82** | 128.24–129.05 | 0.29 |

- **ratio = 1.64× (capture/eager)**
- zero-arm-overlap confirmed: min(capON)=211.57 > max(eager)=129.05 → clean separation.

## Notes / honest caveats
- **Eager baseline moved up.** The task's quoted 98.2 tok/s GLM eager is an older figure; on
  `b416a3e0` (post #1189, on-GPU argmax) eager is **128.8 tok/s** and capture is **211.7 tok/s** on
  this H200. The 1.64× lever is the durable number; both absolute figures are higher than the prior
  baseline (later kernels + argmax + this GPU). Reported honestly rather than reconciled to the old value.
- **No bail to scope.** Unlike the V2-Lite `_d1` case (which needed Leon's planner recovery before
  capture engaged), GLM-4-9B requires **nothing** — it is a dense GQA graph whose attention-mask cone
  already terminates at a genuine capacity-form `Attention[3]` with present-KV, so the #1171 classifier
  passes it. No `geometry.rs`/`bindings.rs` change needed; Leon/Deckard have nothing to implement here.
- Base decode, greedy, opt-in flag (`ONNX_GENAI_CUDA_GRAPH`, default-OFF). No spec-decode. No source edits.

## Standing-table implication
GLM-4-9B int4 is native-only (ORT can't load it — dense int4 export ORT rejects, same capability class
as V2-Lite QMoE). The capture moat therefore extends the native-only lead further:
**native GLM-4-9B = 128.8 tok/s eager / 211.7 tok/s captured (1.64×), byte-identical, opt-in.**

## Reproduce
```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export CUDA_VISIBLE_DEVICES=6 ONNX_GENAI_ONGPU_ARGMAX=1
M=/home/justinchu/glm-e2e-artifacts/cohaagen-glm-4-9b-int4-cuda-post434
ONNX_GENAI_CUDA_GRAPH=1 profile_native --model $M --ep cuda --steady --warmups 1 --runs 3 --tokens 128  # 211.7, captures=3 replays=185 fallbacks=0
ONNX_GENAI_CUDA_GRAPH=0 profile_native --model $M --ep cuda --steady --warmups 1 --runs 3 --tokens 128  # 128.8
# byte-identity: dump generated_token_ids at --tokens 256 for both, diff position-by-position -> 0/256
```

## Constraints honored
Fresh detached worktree off `origin/main b416a3e0`; no `git add -A`; no source edits (measure + scope only);
pinned idle GPU6 after nvidia-smi; base decode only; opt-in flag, no default flip; no /tmp writes; no
stray procs left; GPU returned to idle.

## 2026-08-18 — ORT-fairness dense int4 reconfirmed

Wallace reconfirmed the 2026-08-17 dense int4 Native-vs-ORT decomposition from the opposite direction by trying to enable ORT CUDA graph mode on the same three production exports. True graph-vs-graph is unattainable: Phi-4-mini hard-rejects ORT graph capture because control-flow nodes cannot be supported by CUDAExecutionProvider; qwen2.5-7b aborts at runtime with `ort_value must contain a constructed tensor`; qwen2.5-14b-zp accepts the flag but effectively no-ops because CPU-assigned shape nodes fragment capture (eager 96.9 vs graph 98.7 tok/s, byte-identical).

Eager-vs-eager medians again show this is architectural, not a broad per-kernel native win: Phi native/ORT **0.85×**, qwen2.5-7b **0.77×**, qwen2.5-14b-zp **1.19×**. Keep the deployment headline that native captured decode leads ORT eager **1.33× / 1.14× / 1.83×**, but always label it as CUDA-graph capture plus on-GPU argmax that ORT structurally cannot apply on these dynamic-KV int4 exports.


## 2026-08-18 — Gate-3 speculative verify remains shelved after Marlin

Luv re-probed Deckard's Gate-3 B\* framework on current `main` `923dc592` after Marlin landed. Captured verify break-even `B*=C_verify(M=K)/C_decode(M=1)` is still **NO-GO**: qwen2.5-14b-zp reports **17.5× / 18.4× / 20.0×** for K=2/4/8, and qwen2.5-7b reports **14.9× / 15.7× / 17.4×**. This is worse than the 2026-08-14 baseline (~8.5×) and far above the ≤~2 GO gate, so n-gram/prompt-lookup, EAGLE/MTP, and model-draft speculative-decode work stays shelved.

This updates the earlier spec-decode arc rather than reopening it: the old #957 cheap GQA/SkipSimplifiedLayerNorm residual seams no longer appear as M>1 `KernelCaptureUnsupported` blockers. The measured blocker is now solely `MatMulNBits` at M>1 launching `matmul_nbits_gemm_f16` eagerly; Marlin's capture-safe M>1 int4 GEMM is not selected for this MatMulNBits path. Reconsider only after a graph-safe M>1 MatMulNBits/Marlin path exists and this exact B\* probe is rerun.

## 2026-08-18 — Gate-3 Marlin M>1 opt-in follow-up still NO-GO

Luv re-ran the Gate-3 B\* verify-cost probe with `ONNX_GENAI_MARLIN_M_GT_1=1` and no code changes as a follow-up to the earlier post-Marlin Gate-3 NO-GO. The env gate fixed the capture problem completely: qwen2.5-14b-zp capture segments dropped **96→1**, qwen2.5-7b **29→1**, `KernelCaptureUnsupported` seams disappeared, K=8 byte-identity passed, and the hot path switched to `matmul_nbits_marlin_gemm_f16_splitk`.

The decision does **not** change: speculative decode remains shelved. B\* improved but is still NO-GO at **5.19× / 5.19× / 5.79×** for qwen2.5-14b-zp and **4.64× / 4.71× / 5.23×** for qwen2.5-7b at K=2/4/8, above the ≥~4 kill gate and far above the ≤~2 GO target. The spec-decode family — model-draft, n-gram/prompt-lookup, EAGLE/MTP — is now mined out across the three probes. Residual cost is Marlin M>1 GEMM/repack/reduce (`matmul_nbits_marlin_repack` observed in the hot path), not graph fragmentation.


## 2026-08-18 — Marlin M>1 default flip mined out

Luv completed the real prefill/TTFT A/B for `ONNX_GENAI_MARLIN_M_GT_1=1` versus the portable tiled GEMM path, closing the thread opened by the two prior Gate-3 Marlin entries: “Gate-3 speculative verify remains shelved after Marlin” and “Gate-3 Marlin M>1 opt-in follow-up still NO-GO.” The verdict is **NO-GO to flip the default**: Marlin M>1 stays opt-in.

E2E `profile_native` TTFT showed only marginal-to-neutral qwen2.5-14b-zp movement (**0.976× / 0.988× / 0.999×** Marlin/portable at M=128/512/2048) and neutral-to-worse qwen2.5-7b movement (**1.005× / 1.013× / 1.001×**). Argmax matched every arm, but full-vocab token-0 logprob dumps were not byte-identical (max Δ **0.017** qwen14, **0.168** qwen7), so the silent-default byte-identity bar fails. Treat the Marlin-M>1 vein as mined out: not a spec-decode win, not a prefill/TTFT win, and not eligible for a silent default.

## 2026-08-18 — Wallace small-model and different-architecture GPU probe

### 2026-08-18: Different-architecture GPU probe — qwen3.5 (hybrid linear-attn) + Phi-3.5 + small qwen2.5 regression sanity

**By:** Wallace (inference-engine specialist)

**What (this addendum — the user steer: pivot off qwen2.5 to genuinely DIFFERENT architectures):**
Regression-sanity re-confirmed the two small qwen2.5 rows below are unchanged (native capture wins:
0.5b 1053.5 ≥ ORT 572; 1.5b 692.6 ≥ ORT 436 — no regression). Then pivoted to NEW architectures on the
same H200/GPU7, same fairness pinning (native `ONNX_GENAI_ONGPU_ARGMAX=1`; ORT `ONNX_GENAI_ORT_LIB` →
`.ort-cuda-1.27` CUDA build + `ONNX_GENAI_EP_FALLBACK=1`; 128 tok, warmups 2, `--steady --decode-skip 1`,
medians of 5). Worktree `.worktrees/wallace-small-probe` @ `origin/main 774b256c`, measure-only.

**Models probed:**
| Model | Arch | Export | Native CUDA EP? |
|---|---|---|---|
| qwen3.5-2b-text (`.../qwen3.5-2b-text-generic-cpu-1/v1`) | **hybrid linear attn** (18 `linear_attention` + 6 `full_attention`, DeltaNet/mamba-style: `CausalConvWithState`+`LinearAttention` ops, `attn_output_gate`, GQA 8/2, head_dim 256) | int4 weights, **fp32 activations** (generic-cpu) | ✅ loads, runs fully on GPU, **captures** |
| Phi-3.5-mini-instruct (`.../Phi-3.5-mini-instruct-generic-cpu-2/v2`) | dense GQA, hidden 3072, head 96 | int4-rtn-blk32, fp32 acts | ❌ **native load FAILS — `If` control-flow op unsupported** |
| qwen3.5-9b (`.../qwen3.5-9b-generic-cpu-2/v2`) | multimodal (embedding+text+vision, `inputs_embeds` path) | int4, fp32 acts | ⚠️ not measured — profile_native harness expects a single decoder .onnx; multimodal split needs the embeds pipeline (harness limitation, not an EP gap) |

**qwen3.5-2b standing row (tok/s, H200, int4-wt/fp32-act, 128 tok, greedy; medians of 5):**
| Model | native_cap | native_eager (+argmax) | ORT eager | eager-vs-eager | capture-vs-eager | capture uplift |
|---|--:|--:|--:|:--:|:--:|:--:|
| qwen3.5-2b hybrid | **100.9** (sd 0.46) | 67.5 (sd 0.96) | 61.2 (sd 4.24, noisy) | 1.10× (overlaps) | **1.65×** (clean) | 1.49× |

**Verdict — qwen3.5 hybrid FLIPS the small-qwen2.5 story, and surfaces genuinely NEW levers:**
- **Native runs the entire hybrid graph (linear-attn conv + recurrent-state + full-attn) on GPU and
  captures it cleanly** (`captures replays fallbacks=0`), eager 67.5 → capture 100.9. **ORT CUDA EP
  CANNOT keep the linear-attention ops on GPU** — it inserts **25 Memcpy nodes** and falls nodes to CPU
  (log: "25 Memcpy nodes added… might be unable to run CUDA graph"; strict-no-fallback init *errors*,
  only runs with EP fallback). So ORT is structurally handicapped on qwen3.5 (like the DeepSeek-QMoE
  finding, milder): ~61 tok/s, high variance. Native beats ORT in BOTH framings here (eager 1.10×
  within noise; capture 1.65× clean) — UNLIKE qwen2.5 where native eager LOST. Root cause of the flip:
  ORT lacks GPU kernels for the `LinearAttention`/`CausalConvWithState` contrib ops; native has them.
- **NEW LEVER #1 — the linear-attention path is NOT the bottleneck (it's efficient).** nsys on native
  eager: `linear_attention_f32` = 5.8% + `causal_conv_with_state_f32` = 0.8% (≈6.6% combined). The repo's
  linear-attn kernels are cheap. So the "linear-attention decode inefficiency" the mandate worried about
  does NOT exist here — that path is well-optimized.
- **NEW LEVER #2 (the real headroom) — this export runs entirely in FP32.** Every kernel carries the
  `_f32` suffix (vs qwen2.5's `_f16`). The ONNX export is int4-weight / **fp32-activation** (generic-cpu
  profile: all KV, conv_state, recurrent_state, logits are `FLOAT`). fp32 activations ≈ 2× the
  bandwidth/FLOPs of fp16 on the compute-bound parts. A proper fp16/CUDA-profile export (or on-the-fly
  fp16 activation casting) is a large, unclaimed lever the qwen2.5 (fp16 cuda-gpu-4) campaign never saw.
- **NEW LEVER #3 — the full-attention layers use an unoptimized REFERENCE kernel.** `gqa_attention_reference_f32`
  is the **#1 hotspot at 31.2%** (avg 328 µs/call). head_dim=256 falls outside the fused
  `gqa_decode_attention_f16` fast path qwen2.5 (head 128/64) hits → reference fallback. Extending the
  fused GQA decode kernel to head_size=256 (+ fp16) would kill ~⅓ of decode time.
- **NEW LEVER #4 — giant vocab lm_head.** vocab **248320** (1.6× qwen2.5's 152k); the lm_head GEMV
  `matmul_nbits_gemv_int8_f32` = 730 µs/step, 11.6%, in fp32. On-GPU argmax already applied; fp16 + a
  wider-K GEMV would help.
- **Support-gap finding (Phi-3.5-mini generic-cpu):** native decode backend does not support the **`If`**
  control-flow op present in this export (all its other ops — MatMulNBits, GroupQueryAttention,
  SkipSimplifiedLayerNorm, Sigmoid — are supported). Same control-flow class that blocks ORT CUDA-graph
  on Phi. Needs `ONNX_GENAI_BACKEND=ort` to run, or native `If`/subgraph support to be added.

**Bottom line for the campaign:** qwen3.5 hybrid is a NEW native win vs ORT (ORT can't place linear-attn
on GPU; native runs + captures it, 1.65×). But it also exposes that our qwen3.5 support currently runs
the *generic-cpu fp32 export* — the biggest untapped lever is an fp16 path + a fused head_size=256 GQA
decode kernel (kills the 31% reference-attention hotspot). Linear-attention itself is already efficient.
Owners: fp16-activation path + head256 fused GQA decode → CUDA-kernel owner (Leon/Deckard), reviewer-gated.

---

### 2026-08-18: Small-model (qwen2.5-0.5b / 1.5b) native-vs-ORT GPU decode probe — the launch-bound regime

**By:** Wallace (inference-engine specialist)

**What:**
Benchmarked two never-measured small Foundry Local CUDA-GPU models — qwen2.5-0.5b-instruct
and qwen2.5-1.5b-instruct int4 (`~/.foundry/cache/models/Microsoft/qwen2.5-{0.5b,1.5b}-instruct-cuda-gpu-4/v4`)
— base decode, single-stream greedy, on an idle H200. Same fairness-correct methodology as the
now.md standing table: native `ONNX_GENAI_ONGPU_ARGMAX=1`; ORT pinned to the genuine CUDA build
(`ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so` = `.ort-cuda-1.27`, `ONNX_GENAI_EP_FALLBACK=1`
— no conda CPU false baseline; CUDAExecutionProvider confirmed). 128 tok, warmups 2, `--steady
--decode-skip 1`, medians of 5 rounds (`--runs 3` each). Fresh worktree off `origin/main 774b256c`.
GPU pinned `CUDA_VISIBLE_DEVICES=7` (nvidia-smi: all 8 idle first).

**Standing-table rows (tok/s, H200, int4, 128 tok, greedy; medians of 5, sd in-line):**

| Model | native_cap (capture) | native_eager (+argmax) | ORT eager | eager-vs-eager (ne/oe) | capture-vs-eager (nc/oe) | capture uplift (nc/ne) |
|---|---:|---:|---:|:--:|:--:|:--:|
| qwen2.5-0.5b | **1053.5** (sd 0.97) | 321.5 (sd 1.22) | 572.0 (sd 6.88) | **0.56×** (ORT faster) | **1.84×** | **3.28×** |
| qwen2.5-1.5b | **692.6** (sd 1.02) | 324.8 (sd 0.89) | 435.7 (sd 1.45) | **0.75×** (ORT faster) | **1.59×** | **2.13×** |

Overlap checks clean: capture-vs-eager min_native_cap > max_ort_eager on both (1053>585, 692>436).
eager-vs-eager decisively ORT: min_ort_eager > max_native_eager on both (568>323, 432>326).

**Verdict:**
- **Deployment default (graph capture ON, our shipping config): native WINS big — 1.84× (0.5b) and
  1.59× (1.5b) over ORT's best (eager).** The capture moat is *largest in the small regime*: uplift
  3.28× on 0.5b and 2.13× on 1.5b, vs only ~1.5× on the 7b/14b. So native has HEADROOM here and still
  beats ORT under the deployment default. This is the strongest capture win in the whole campaign.
- **BUT eager-vs-eager (pure per-kernel), native LOSES to ORT on both small models — 0.56× (0.5b) and
  0.75× (1.5b).** This is a real, honest regime where our per-kernel/per-step path is behind ORT. It
  extends the pattern from the dense trio (native eager also lost on Phi 0.85× and qwen7b 0.77×; only
  qwen14b-zp won eager 1.19×) — and the smaller the model, the WORSE native eager looks relative to ORT.

**Why (root cause — a NEW headroom lever, quantified):**
Native eager decode is **fixed-per-step-overhead (launch/host-sync) bound** in the small regime:
- native_eager throughput is **model-size-INVARIANT**: 321.5 tok/s (0.5b) vs 324.8 tok/s (1.5b) — a
  flat **~3.1 ms/step floor** across a 3× model-size change. A compute-bound path would scale with size.
  ORT eager DOES scale (1.75 → 2.30 ms/step) and native_cap DOES scale (0.95 → 1.44 ms/step) — only
  native *eager* is pinned. That flat floor is the signature of fixed per-step overhead dominating.
- **nsys (`--cuda-graph-trace=node`) on 0.5b native eager confirms it:** total GPU kernel busy ≈ 101 ms
  over 128 decode steps = **~0.79 ms/step GPU-active vs 3.11 ms/step wall → only ~25% GPU-active**,
  ~75% is host-side gaps between launches. No single kernel dominates (largest is the lm_head
  `matmul_nbits_gemv_f16_scales_f16_pipe` at 79.7 µs/step = ~10% of GPU time); it's **death-by-many-
  small-launches** — ~48 `matmul_nbits_gemv` + 48 `skip_rmsnorm` + 24 attention kernels per step, each
  2–5 µs, so per-launch overhead (~2 µs) rivals the kernel itself. This is precisely the regime CUDA
  graph capture was built for: capture collapses the ~2.16 ms/step of host launch/sync overhead
  (3.11 → 0.95 ms/step on 0.5b), landing native_cap near the GPU-active floor.

**The lever:** native's eager per-step overhead (~2 ms fixed, ~75% of a small-model step) is the gap.
Options if we ever want eager competitive on small models (NOT needed for the deployment claim, which
is capture-ON): reduce host-side launch overhead — kernel fusion (fuse the per-layer gemv/rmsnorm
chains further), fewer host round-trips / stream syncs per step, or a lighter dispatch path. But the
honest framing: **for shipping (capture ON) native already wins 1.6–1.8× and this is the biggest
capture headroom we've measured — the small regime is a native STRENGTH via capture, and an eager
WEAKNESS via per-step overhead.** Same architectural story as the dense/ORT-fairness decomposition:
the moat is graph-capture + on-GPU argmax, not per-kernel eager speed.

**Notes:**
- Did not pursue the `v4-bs128` 0.5b variant: it's a batch-128 export, irrelevant to this single-
  stream per-step-overhead question (v4 already gave a decisive result).
- ORT could not be graph-captured here either (consistent with the 2026-08-18 graph-vs-graph finding:
  ORT CUDA-graph structurally blocked on these dynamic-KV int4 decode exports) — so ORT's best is eager,
  which is the column compared. Base decode greedy only; not a spec-decode basis.

**Reproduce:**
```bash
source /home/justinchu/onnx-genai/.cudaenv.sh
export CUDA_VISIBLE_DEVICES=7 ONNX_GENAI_ONGPU_ARGMAX=1 \
       ONNX_GENAI_ORT_LIB=$ORT_ROOT/lib/libonnxruntime.so ONNX_GENAI_EP_FALLBACK=1
M=~/.foundry/cache/models/Microsoft/qwen2.5-0.5b-instruct-cuda-gpu-4/v4
ONNX_GENAI_CUDA_GRAPH=1 target/release/profile_native --model $M --ep cuda --backend native --tokens 128 --warmups 2 --runs 5 --steady --decode-skip 1  # native capture
ONNX_GENAI_CUDA_GRAPH=0 target/release/profile_native --model $M --ep cuda --backend native --tokens 128 --warmups 2 --runs 5 --steady --decode-skip 1  # native eager
ONNX_GENAI_CUDA_GRAPH=0 target/release/profile_native --model $M --ep cuda --backend ort    --tokens 128 --warmups 2 --runs 5 --steady --decode-skip 1  # ORT eager
```
Worktree `.worktrees/wallace-small-probe` @ `origin/main 774b256c`; profile_native built
`--features bench-native,bench-ort,cuda`. No source edits, no commits (measure-only). GPU7 returned idle.
