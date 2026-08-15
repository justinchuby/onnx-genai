# Decisions — live standing directives

Last consolidated: 2026-08-15T03:10:00Z (Scribe decode-vs-ORT arc; local state; merged 12 requested inbox drops and deleted them, keeping inbox/README.md. HARD-GATE size compaction: live decisions.md was 42,104 B before this batch and would exceed the 20,480 B prompt budget after merging; the complete pre-compaction live file was archived to `.squad/decisions/archive/2026-08-15T03-10-00Z-decisions-pre-decode-arc.md`. Older detailed narrative remains in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and `.squad/decisions/archive/`.)

Standing governance rules and active directives. Full narrative is archived; keep this file to current decisions plus durable rules.

## Ledger health rule

Archive by SIZE, not age. Age-only archiving can silently no-op during high-volume campaigns because most entries are recent. When the live ledger crosses the spawn-budget gate, preserve full history in an archive and keep `decisions.md` to standing directives, active decisions, and pointers. Assemble from inbox drops, dedupe, then delete merged drops; leave `decisions/inbox/README.md`.

## 2026-08-14/15 — glm-4-9b-int4 decode-vs-ORT program

### Same-weights ground truth vs stock ORT (Sebastian)

On identical int4 weights and fair ORT CUDA fastcfg (`past_present_share_buffer=true`, CUDA graph enabled), native default M=1 decode initially lost to stock ORT:

| model | native | ORT | gap |
|---|---:|---:|---:|
| glm-4-9b-int4 (block-128) | 97.5 tok/s | 250.3 tok/s | ORT 2.57× |
| qwen2.5-14b-int4 (block-32) | 127.0 tok/s | 184.5 tok/s | ORT 1.45× |

Fairness: glm ORT artifact strips only the redundant GQA `rotary_embedding_dim` while hardlinking the same `model.onnx.data`; native and ORT greedy tokens match byte-identically for the certified glm prompt. qwen shares the same weight file; its early divergence is the known near-tie cascade, not a weight mismatch. Eager op profiling overweights launch-heavy GQA; CUDA-graph kernel tracing is the authority for decode mix.

### PR #978 — int4 M=1 decode GEMV split-K (Deckard; Chew/Gaff approved)

**MERGED `532ef6bc`.** glm block-128 routes through generic `matmul_nbits_gemv_f16_general_bs`, not the already-tuned block-32 path. Deckard added default-on portable LOP3 dequant plus grid-fill split-K (`ONNX_GENAI_GENERAL_SPLITK=0|1` A/B). Result: glm **97.5→112.4 tok/s** (+15.3% vs original main; Deckard local 98.09→112.88), ORT gap **2.57×→2.24×**; qwen block-32 untouched by the `block_size != 32` gate. Greedy tokens byte-identical on glm; f64 oracle passes.

Chew numerics review: 🟢 APPROVE. LOP3 dequant is bit-identical for full 8-chunks; split-K is fp32, race-free, complete K partition, near-equal only by add reassociation and inside f64 oracle tolerance. Gaff quality review: 🟢 APPROVE. Rule 11 pass (SM53+ portable instructions already used by existing kernels), capture-safe static routing, fmt clean, no new clippy warnings in touched code. Default-on is accepted.

### PR #980 / cp.async GEMV spike — measured NO-GO (Deckard)

**CLOSED, not merged.** Deckard fully built SM80-gated cp.async variants behind `ONNX_GENAI_GEMV_CPASYNC=1`; correctness was fine (byte-identical tokens, f64 oracle), but every measured config regressed glm on H200: default direct **112.2 tok/s**, cp.async stages 3/5 **95.9/95.4** (≈−15%), batched **89.7** (≈−20%), single-warp direct→cp.async **102.6→88.3**. ncu showed occupancy unchanged (~79.4→79.9%) but duration +43%; M=1 GEMV has only ~8 FMAs per loaded word, so cp.async commit/wait/address/shared-load overhead has no compute to overlap. Standing verdict: **do not default-enable or retry cp.async for M=1 int4 GEMV**; occupancy/split-K was the right first lever.

### PR #981 — multi-warp block SkipSimplifiedLayerNorm for M=1 decode (Deckard; Chew approved)

**MERGED `b24e961e`.** Deckard replaced the one-warp, one-SM decode SkipRMSNorm path with a portable block-parallel half4 kernel for `num_groups <= 8`; prefill keeps the warp path. Env A/B: `ONNX_GENAI_CUDA_DISABLE_SKIP_RMSNORM_BLOCK=1`. Result on H200: glm **112.3→137.8 tok/s** (+22.7%), qwen **125.6→148.85 tok/s** (+18% range), greedy tokens byte-identical. Kernel-level RMSNorm fell ~28µs→4µs and occupancy 1.56%→43.8%.

Chew numerics review: 🟡 APPROVE-WITH-NOTES. Residual/skip writes are byte-identical to the warp path; only fp32 sum-of-squares reduction order changes by ULPs, with numerics tests green in isolation. Notes: a pre-existing capture assertion failure in `skip_simplified_layer_norm_gpu.rs` is Sebastian/capture-owner territory, not caused by #981; a test-only tolerance relaxation for fused-norm-prologue parity is intentional. `gaff-pr981-quality.md` was an empty drop, so no quality verdict text was available to merge.

### Cumulative base-decode result and corrected ORT-gap mechanism

Cumulative glm base decode this arc: **97.5→112.4 (#978)→137.8 (#981) = +41%**. ORT gap narrowed **2.57×→1.82×** against ~250 tok/s stock ORT. All shipped changes are byte-identical at token level, portable, capture-safe, and default-on.

Deckard's immediate post-#981 re-profile first concluded base decode was near its floor: GEMV ~77% of decode at high occupancy, RMSNorm down to ~3.4%, GQA ~7.5%, lm_head ~5.7%; base-only realistic ceiling was estimated ~160–175 tok/s. That floor verdict is **OVERTURNED** by the later ORT head-to-head GEMV diagnosis.

ORT's dominant glm kernel `MatMulFloatInt4Kernel<__half,128,0>` uses the same numeric class and same gate_up geometry but streams weights much better:

| gate_up int4 GEMV | duration | DRAM | achieved BW | SM | occ |
|---|---:|---:|---:|---:|---:|
| ORT | 24.9µs | 50.5% | 2.42 TB/s | 61.7% | 80.9% |
| native | 65.3µs | 19.1% | 0.92 TB/s | 78.8% | 84.6% |

The gap is **narrow-load / memory-level-parallelism**, not algorithm, occupancy, or irreducible dequant math. A byte-identical synchronous 128-bit wide-load GEMV with independent in-flight loads is now the primary base-decode lever. Expected payoff: partial **180–200 tok/s**, full ORT-like streaming **~236 tok/s** (near ORT base parity), with no numeric cost because it reorders loads but preserves per-lane ascending-K accumulation. **In flight:** Deckard on branch `squad/int4-gemv-wideload` (GPU6).

### Speculative decode / captured verify status (Sebastian)

Marlin M>1 whole-graph capture remains the multiplicative lever. Sebastian closed the captured-vs-eager 2×2 parity cell with a test-only harness: glm and qwen M=8 captured verify are **byte-identical to eager logits and argmax**, segments=1, with refreshed B*: glm **~2.16–2.19×** (practical GO), qwen **~4.45–4.7×** (denominator/drafting-depth story, not a GEMM bug). **In flight:** Sebastian on branch `squad/spec-decode-e2e` (GPU7), captured M=8 verify and selective KV commit path. This stacks with the base wide-load GEMV and is the route to passing ORT.

### Standing directive for this arc

Do not revive the base-decode “floor” claim without first comparing against ORT's streaming efficiency. Prioritize: (1) Deckard's 128-bit wide-load int4 GEMV, default-on if byte-identical and portable; (2) Sebastian's speculative-decode e2e path; (3) only then revisit smaller base kernels such as GQA or lm_head. Keep cp.async M=1 GEMV as a recorded NO-GO.

## Native-vs-ORT fairness rule

Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output. If one engine crashes, rejects the graph, runs CPU, disables CUDA graphs, or uses a different weight file/config, report a capability/config gap rather than a throughput multiplier. For ORT-genai decode, verify CUDA provider and share-buffer/cuda-graph fast path are active before quoting tok/s.

## Benchmark and profiling discipline

Separate measured, estimated, and projected. Same-run PR-vs-base deltas beat absolute numbers under shared-host load. For CUDA-graph decode, `ONNX_GENAI_PROFILE_OPS=1` is a host/eager dispatch view and can mis-rank kernels; use `nsys --cuda-graph-trace=node` for kernel mix and `ncu --graph-profiling node --set full` for stall mechanism. A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.

## Numerics and portability discipline

Default-on CUDA decode optimizations must be portable or explicitly arch-gated with byte-identical fallback. Token byte-identity is an argmax stability claim, not a numeric invariant; numeric changes need oracle/tolerance justification. Preserve Rule 11: unsupported devices must fall back without behavior loss. Env knobs used for A/B must be documented, deterministic under capture, and not hide default regressions.

## Testing and CI standing directives

- `cargo test --workspace` silently truncates on failure; use `--no-fail-fast` for full-suite evidence.
- Run new tests in isolation before trusting full-suite green. Assert on what code did, not summaries.
- An agent self-report is not evidence; verify with code, command output, and tests.
- Reviewer lockout is enforced: authors do not revise their own rejected artifacts.
- CI is asynchronous; required local targeted tests/builds/hardware probes remain blocking, but do not idle solely waiting for CI.
- Never commit `.squad/` files to external repos; if that happens, purge history rather than only deleting in a follow-up commit.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and `.squad/decisions/archive/`. The complete live ledger immediately before this decode-arc compaction is `.squad/decisions/archive/2026-08-15T03-10-00Z-decisions-pre-decode-arc.md`.
