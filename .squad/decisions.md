# Decisions — live standing directives
Last consolidated: 2026-08-20T13-46-00+0000 (Scribe q38 #1569 merge/next-round batch: 5 inbox drops processed; detailed narrative archived to .squad/decisions-archive/2026-08.md; pre-check live size 38009 bytes.)


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

Detailed pre-existing 2026-08-19 narrative remains in `.squad/decisions-archive/2026-08.md` after size compaction; live ledger keeps standing directives and current summaries.

---

### 2026-08-20: PR #1569 merged after relaxed golden-lock re-validation

**By:** Sebastian; consolidated by Scribe
**What:** Sebastian independently validated PR #1569 integrated on origin/main and reported PASS under the relaxed dtype-tolerance bar. q38 improved **61.27 → 62.76 tok/s (+2.43%)** on idle H200 GPU7, batch=1 greedy, 128-token steady window, warmup 1, median N=5, q38 int4 block-32 asymmetric bf16, `bench-native,cuda`. mary control stayed byte-identical because the fold refuses on f16; q38 clear prompts (Japan/Italy/Water) were byte-identical main==PR. PR #1569 merged; `origin/main` is now `b693f2bb2`.
**Why:** Confirms Deckard's DeltaNet `Neg(Exp(A_log))` decay-chain fusion is coherent and above the >=2% bar. Drop the unsupported determinism claim: q38 split-K GEMV nondeterminism persists and is non-blocking.

---

### 2026-08-20: External decode survey prioritizes inter-kernel fusion for q38 int4 GDN decode

**By:** Holden; consolidated by Scribe
**What:** Holden surveyed vLLM/FLA/llama.cpp/Marlin for int4 GDN batch=1 decode. Core finding: launch-bound M=1 decode wins come from **far fewer kernels**, not faster individual kernels. Ranked next levers: (1) GDN recurrence megakernel folding β-sigmoid + softplus/dt_bias + conv1d/state into the fused recurrence, (2) adjacent projection fusion with inline SwiGLU, (3) Marlin-style prepack/streaming, then smaller attention/state/cache packaging levers.
**Why:** Sets next-round routing: Deckard owns recurrence megakernel lever #1; Batty owns projection/GLU GEMV fusion lever #2. Full survey detail is archived in `.squad/decisions-archive/2026-08.md`.

---

### 2026-08-20: Batty GEMV latency-hiding pass hit floor; pivot GEMV-side work to inter-kernel fusion

**By:** Batty; consolidated by Scribe
**What:** Batty swept the block-32 asymmetric int4 M=1 GEMV prefetch depth and found shipped PF=2 is already the optimum: q38 ~61.2 tok/s at PF=2 vs 57.4 PF=1, 60.9 PF=3, 60.3 PF=4, 56.9 PF=6, ~30 PF=8. Wider 128-bit loads do not apply to block-32 without breaking layout/identity; redundant scale/zp shuffle is judged L2-cheap and inner-loop-hostile. No PR.
**Why:** The 74% roofline gap is not closable by more per-warp load-latency hiding inside the current GEMV. Future GEMV-side work should raise graph-wide occupancy by fusing adjacent projections / GLU rather than deepening prefetch.

---

### 2026-08-20: Relaxed golden-lock bar for perf work

**By:** Justin via Coordinator; consolidated by Scribe
**What:** Performance optimizations no longer require bit/byte identity to the pre-change baseline. Acceptance requires coherent decode behavior that matches a higher-precision reference on clear prompts and numerical differences within a reasonable tolerance for the operating dtype (bf16/f16/f32). mary byte-identity is useful as a low-noise control but not mandatory.
**Why:** Strict byte-faithfulness over-constrained kernel fusions and could reject more accurate implementations that only flip razor-thin ties. Honest ceilings, generality, and no coherence regression remain mandatory.

---

### 2026-08-20: Deckard DeltaNet decay-chain fusion validated and merged

**By:** Deckard; consolidated by Scribe
**What:** `CudaLinearAttentionGatingFusion` now absorbs exported `Neg(Exp(A_log))` decay chains for Qwen3.8 Gated-DeltaNet by passing raw `A_log` and computing `-round_store<T>(expf(A_log))` inline when dtype/topology guards pass. The fold fires for q38 bf16 and refuses when the graph runs decay in f32/f16, preserving correctness.
**Why:** Recovers tiny decode-step launch fragments across the 48-layer stack and landed in PR #1569 after independent validation.
