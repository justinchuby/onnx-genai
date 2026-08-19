# Decisions — live standing directives

Last consolidated: 2026-08-19T11:56Z (Scribe qwen3.5 family-trio consolidation: 19 inbox notes processed; archive skipped because pre-check live ledger was 14,881 bytes < 20,480-byte requested gate.)

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

### 2026-08-18: adaptive split-KV sizing for attention_row decode — GO, default-ON (PR #1350, merged)

**By:** Deckard

**What:** Sizes `num_splits` adaptively from fixed KV `cap` + fixed row count to target ~one occupancy wave (grid ≈ 512), floored at MIN_CHUNK=128. Replaces fixed `chunk=256` from #1340. `ONNX_GENAI_ATTN_SPLIT_CHUNK` still pins for A/B. Capture-safe (geometry from cap+rows only). Merged origin/main @ 32ee82cb5.

**Why:** Deep-ctx re-profile (~2600 tok) revealed split-KV win is +63% over monolithic (not the +10% at shallow ctx). Fixed-256 under-fills at depth (grid 256, waves/SM 0.48, warps 11%). Sweep proved optimum at grid ≈ 512 (one full wave, ~4 CTAs/SM), not grid ≈ SM count. Adaptive targets one wave automatically across head counts.

**Key result (H200, eager, medians-of-5):** deep fixed-256 46.37 → adaptive **48.32 tok/s (+4.2%; +70% vs monolithic)**; shallow **60.82 (+4.7%; +17.7% vs mono)**; short capture neutral. Occupancy A/B: monolithic grid16/waves0.02/warps9.8%/DRAM0.8% → fixed-256 grid256/0.48/11%/34% → **adaptive grid512/0.97/24%/48%**. Byte-identical short; f64-tol wide.

---

### 2026-08-19: attention_split launch_bounds / register pass — NO-GO (latency-bound, not occupancy-bound)

**By:** Deckard

**What:** Tested `__launch_bounds__(256, minblocks)` pass on `attention_split` to raise resident CTAs/SM. NO change shipped; branch `squad/attention-split-launchbounds` carries no code change.

**Why NO-GO:** lb worked at register level (56→40 reg/thread, occupancy limit 4→6 blk/SM, no spill). But produced no E2E win: at grid 512 all blocks already fit in one wave at 4 blk/SM, so freed occupancy headroom places no new blocks. Denser grid (chunk 96 → grid 688) did raise warps-active 24.2%→33.0%, but deep E2E stayed flat (DRAM actually dropped 47.8%→42.0%). Roofline verdict: `attention_split` at M=1 is bound by per-warp **dependent-load latency chain** (serial Q·K + P·V accumulation), not occupancy. Grid-512 from #1350 already sits at the effective knee. Consistent with QMoE GEMV roofline NO-GOs — moat decode kernels are at the M=1 latency roofline.

---

### 2026-08-19: Parallelize `derive_len` mask-frontier scan — GO (PR #1357, merged 354be8fc7)

**By:** Deckard

**What:** Replaced single-thread O(key_len) serial mask scan with block-parallel 256-thread min-index reduction. Launch stays grid(1,1,1)/block(256,1,1) → capture-safe. One file: `standard_attention.rs`.

**Why:** Deep-ctx re-rank surfaced `derive_len` as 2nd-largest decode kernel at ~2600 tok (143.6µs/call, 5.7% GPU time, fired per-layer per-step, O(context) cost). Root cause: grid(1,1,1)/block(1,1,1) single-thread serial scan. Block-min reduction is byte-identical by construction (integer index, no fp accumulation).

**Key result (H200, medians-of-5, DeepSeek-V2-Lite int4):** derive_len kernel 143.6µs → **4.86µs (29.5×)**; E2E deep ctx 46.55 → **56.43 tok/s (+21.2%)**; short ctx +2.7%. Deep win reproduced across two 5-run sets. Win matches roofline prediction (removing ~3.9ms/tok serial scan from 21.5ms/tok decode). Perp-step hoist (b) evaluated and deferred: after (a) the residual 27×4.86µs is ~0.7% of deep-ctx decode — cross-cutting refactor risk not justified.

---

### 2026-08-19: Block-parallel int64 cumsum for low-lane decode — GO (PR #1366, merged e19697fb0)

**By:** Deckard

**What:** Added `cumsum_block_i64` — block-per-lane cooperative prefix scan (intra-tile Hillis-Steele, tile=blockDim=256, running per-block base) for int64 low-lane calls (`lanes ≤ sm_count`). Launcher routes these to it; fp32 and high-lane (prefill) calls keep the existing serial-per-lane kernel. Two files: `cumsum.rs`, `indexing_gpu.rs`. Byte-identical (int64 only — integer addition is associative; fp32 explicitly excluded to preserve byte-identity). Grid=lanes, block=256 fixed → capture-safe.

**Why:** Post-#1357 deep-ctx re-rank surfaced `cumsum_i64` as the standout: 341µs/call at ~2600 tok (confirmed O(context): 45.8µs at ~640 → 341µs at ~2600). One call per decode step. Derives-len anti-pattern again: batch-1 sequence cumsum (`position_ids = cumsum(attention_mask)`) has lanes==1, collapsing to one thread scanning seqlen serially.

**Key result (H200, medians-of-5, DeepSeek-V2-Lite int4, CUDA graphs ON):** cumsum kernel 341.5µs → **18.08µs (18.9×)**; E2E deep ctx CUDA graphs ON: 109.5 → **113.6 tok/s (+3.8%)**, short +1.1% (flat); eager deep +1.7% (launch-bound path, cumsum hid under gaps). CUDA-graph path is the production metric — the ~330µs/token saving lands directly on the critical path.

---

### 2026-08-19: GLM-4-9B fair same-export A/B, block-32 prize-bounding, splitk_wide deep-profile

**By:** Wallace (benchmarking / perf-fairness) — read-only measurement; no PR, no kernel edits.

**What (three findings):**

1. **Fair same-export A/B (identical `model.onnx.data` inode):** H200, CUDA-graph both engines, greedy, medians-of-5: native 213.39 tok/s vs ORT-genai **251.37 tok/s = ORT 1.18× faster**. The prior 97.62 tok/s native number (Sapper) does NOT reproduce across 2 binaries × 2 exports × 2 configs — retracted as an unrepresentable contention artifact. GLM is a **~18% native optimization target**, not a moat.

2. **Block-32 re-export prize-bounding = ≤0.** Re-quanting to block-32 DOES engage the fused fast-path (`select_f16_gemv_variant` returns DownProjection only when block_size==32), confirmed by nsys. But native is **SLOWER** at block-32: −3.4% short (211.64→204.44), −4.2% deep (194.96→186.84). Gain from fusing RMSNorm more than cancelled by 4× scale-metadata DRAM traffic. ORT hurts less (−1.8%/−1.9%). Gap WIDENS at block-32 (1.18→1.20× short, 1.25→1.28× deep). **Do NOT spawn a block-128→32 kernel agent on the fusion rationale.**

3. **splitk_wide is latency-bound at M=1 (DRAM 16.6%, No-Eligible 30.8%).** Root cause: single-column-per-warp accumulation exposes load latency; DRAM idle 83% despite healthy occupancy (59.8%). Native's own `wide_multicol` reaches 37.4% DRAM (1.79 TB/s) on the same GPU at LOWER occupancy — by register-blocking 4 columns/warp. Fix: multicol × split-K hybrid for down_proj/qkv shapes (register-block 2–4 cols AND keep K_SPLIT for grid fill). Predicted native 243–274 tok/s (closing or passing ORT's 249.83). Sebastian implementing.

**Depth-scaling finding:** block-128 native loses −7.9% short→deep-2600 vs ORT −2.7%; gap widens 1.18×→1.25×. GQA decode-attention path scales worse than ORT's. Second lever required for deep-ctx parity (post GEMV fix).

---

### 2026-08-19: CLAIM RETRACTION — GLM-4-9B is NOT a moat; it is an ~18% native deficit

**By:** Wallace / Scribe (claim-integrity)

**What:** The "2.56×" native-over-ORT claim for GLM-4-9B is **RETRACTED**. Sapper's 97.62 tok/s native baseline was an un-reproducible contention artifact (VRAM-limit/weight-streaming in that run). Fair same-export CUDA-graph A/B (identical `model.onnx.data` inode 239679076): **native 213.39 vs ORT-genai 251.37 tok/s = ORT 1.18× faster**. GLM is not a "only-we-can-run" moat — ORT loads and runs it faster on the identical export. Do NOT cite 97.62 or 2.56× anywhere. The native GLM decode gap is real, isolated, and tractable: root cause is `matmul_nbits_gemv_f16_general_bs_splitk_wide` (41% decode, 16.6% DRAM peak, latency-bound at M=1 from single-column accumulation); block-32 re-export was prize-bounded at ≤0. Real lever: register-block the split-K GEMV to match `wide_multicol`'s 37.4% DRAM (Sebastian implementing).

---

### 2026-08-19: CLAIM RETRACTION — DeepSeek int4 QMoE is NOT an "ORT-can't-run-it" moat; ORT is FASTER on a fair CUDA export

**By:** Sapper / Scribe (claim-integrity)

**What:** The "598×/0.17 tok/s ORT" claim for DeepSeek-V2-Lite int4 QMoE is **RETRACTED**. Root cause: ORT's CUDA QMoE kernel requires fp16/bf16 activations; our default export emits fp32 → ORT partitioner drops all 26 QMoE nodes to CPU EP silently. The 0.17 tok/s was a CPU-fallback artifact, never an ORT-CUDA-QMoE measurement. ORT 1.27 DOES ship a CUDA QMoE kernel (int4/block128/swiglu/k=6). Fair CUDA-vs-CUDA A/B on an fp16-activation export: **ORT places all 26 QMoE on GPU (0 Memcpy) at 86.78 tok/s vs native fp32-activation 55.15 tok/s = ORT 1.57× faster** at M=1 short decode. Do NOT cite 598× or "ORT can't run QMoE." The export-compat gap (native EP requires fp32 QMoE activations, ORT CUDA QMoE requires fp16/bf16) means no single export currently runs QMoE-on-CUDA in both engines — this is a mutual constraint, not an ORT deficiency. Deep-ctx (~2600) native currently aborts (Attention workspace not resized for large KV — under investigation). Native's #1 lever: fp16-activation QMoE kernel.

---

### 2026-08-19: Decode per-kernel fp32 shaving is near the floor

**By:** Deckard / Scribe (claim-integrity)

**What:** The two O(context) single-thread anti-patterns in the decode path are now eliminated: `derive_len` (#1357, 29.5×) and `cumsum_i64` (#1366, 18.9×). The `attention_split` kernel is roofline-NO-GO'd for further per-kernel work (latency-bound at M=1, not occupancy-bound; launch_bounds pass was neutral). QMoE GEMVs are latency-roofline'd and were NO-GO'd twice. **Future decode gains come from structural improvements: fp16-activation QMoE kernel (M=1 tensor-core path), register-blocking the GLM split-K GEMV (multicol×split-K hybrid), and wider CUDA-graph coverage to close the eager CPU-launch gap (eager decode ~17ms/tok wall vs ~9ms/tok graphs).** Per-kernel fp32 shaving of the remaining small kernels (scatter_f32_i64 ~4.4µs, skip_rmsnorm ~2.7µs, build_kv ~1.8µs) is not a productive next lever.

## 2026-08-19T11:56Z — Qwen3.5 family-trio arc and adjacent batching/kernel inbox consolidation

**By:** Scribe

**Pre-check:** live `decisions.md` was 14,881 bytes, below the 20,480-byte hard gate requested for this pass, so no archive was created. Inbox contained 20 markdown files including `README.md`; 19 drops were processed.

### Qwen3.5 hybrid family decisions

- **Deckard — qwen3.5-0.8b text export (PR #1456, merged @169febb1f): GO.** Graph surgery composed the text-only 0.8B export; native loads/captures the hybrid block and decodes byte-identical to ORT. This proves the hybrid graph-block moat also holds at the small end and locks the {0.8B, 2B, 9B} family coverage.
- **Deckard — qwen3.5-9b text export (PR #1449): GO.** Graph surgery produced a clean 9B text export; native captures the full hybrid path with byte-identical decode. The moat scales up as a family property, though 9B is memory-ceiling limited.
- **Sebastian — qwen3.5-2b hybrid kernel ladder: moat confirmed; `lm_head` int8 GEMV NO-GO; op-soup declined.** Profiling showed the dominant win is context-flat graph/capability coverage for hybrid recurrent ops, not a single dense kernel. The realistic lever became cast identity cleanup rather than a bespoke `lm_head` kernel.
- **Sebastian — qwen3.5 identity-Cast elimination (PR #1459, merged @792958ecf): GO, general rule.** `CudaDropIdentityCast` removes redundant CUDA-bridge `Cast` nodes when source/target element types match; qwen3.5 casts dropped 270→0 with byte-identical outputs and +3.0% short-context throughput. Apply as RULES §2-style graph cleanup, not a model-specific hack.
- **Wallace — qwen3.5 2B golden lock (PR #1418): GO.** The 2B hybrid moat has a regression-proof golden lock; byte identity and the native-only graph/capture path are now guarded.
- **Wallace — qwen3.5 family moat: GO.** The hybrid graph-block moat is architectural across the family, originally quantified for {0.8B, 2B} and then completed with 9B. The 2B curve is strongest at depth (1.00×@16 → 4.03×@1729 → 9.08×@5000, no asymptote before native memory ceiling).
- **Wallace — qwen3.5-9B deep curve: GO with caveat.** 9B confirms the family property but does not strengthen with size; it has a higher floor, gentler slope, and ~1500-context memory ceiling, capping around 2.7×.
- **Wallace — qwen3.5-0.8B fixed-depth fair A/B: profile-only complete.** Native vs ORT reaches 3.45×@16 and 7.50×@1729, with ORT runnable only via raw onnxruntime CPU-fallback; the numeric family trio is complete.
- **Wallace — Foundry-Local moat sweep: GO for qwen3.5 hybrid as a new context-scaling graph-block moat.** The recurrent/linear-attention operators form a capability moat distinct from ordinary dense decode kernels.

### Adjacent model/kernel decisions

- **Deckard — gemma4-e2b dual head_dim 256+512 e2e: partial GO.** Text export was composed and head_dim=512 eager golden lock landed (PR #1438/#1442 context); remaining blocker is CUDA-graph capture for the real dual-head-size path.
- **Deckard — Nemotron streaming ASR triage: NO-GO for LM-decode moat.** The 0.6B Nemotron streaming models are RNN-T ASR/transducer systems, not autoregressive LM decode targets; loader decline is correct, not a bug.
- **Sebastian — GLM deep-context re-rank: next general lever is block-128 RMSNorm-fold gap, not attention.** Post-#1435 profiling shifted attention away from attention kernels; deep gap decomposition kept the RMSNorm-fold gate capability-driven and near floor after PR #1445 work.
- **Sebastian — CPU budget semantics: physical cores, not logical CPUs.** `ONNX_GENAI_CPU_DECODE_THREADS=N` should constrain the process to `N` physical cores for stable CPU decode interpretation.
- **Sebastian — paired A/B harness caution.** The paired native/ORT harness depresses the native arm via co-residency/measurement interference; use it for relative controlled checks, not absolute native ceilings.

### Batch and placement decisions from copilot drops

- **M≥2 batch-decode resident cliff: fused-SwiGLU capture segmentation, not sampling or launch count.** A byte-identical fix exists but was not merged because it needs coordination with related CUDA capture segmentation work; model-size dependence must be reported explicitly.
- **Batch device-logits router (#1155): killed for the 8GB large-model speedup claim.** Measurement showed it is not the large-model speedup on the 8GB box; keep any benefit narrowly framed and data-backed.
- **Batch-N large-model scaling: 1/N HtoD amortization is real, but wall-clock is capped by VMM churn plus an M≥2 decode-GEMM cliff.** Structural CUDA dispatch changes remain held for owner sign-off.
- **Multi-row decode GEMV ceiling probe: NO-GO.** The resident-model prize is ≤1 ms/step, too small to justify building a multi-row decode GEMV path now.
- **Placement/native capture compatibility: viable with strict placement constraints.** Native CUDA capture compatibility must preserve fail-closed placement behavior and respect heterogeneous fallback boundaries.

