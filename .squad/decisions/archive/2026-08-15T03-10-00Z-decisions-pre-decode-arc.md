# Archived live decisions snapshot — 2026-08-15T03-10-00Z decode arc compaction

Archived by Scribe before merging the glm-4-9b-int4 decode-vs-ORT arc. This preserves the complete pre-compaction live `.squad/decisions.md` (42104 bytes).

---

# Decisions — live standing directives

Last consolidated: 2026-08-14T21:43:12Z (Scribe marlin-int4-gemm-landed batch, local state; merged 5 inbox drops — deckard-marlin-kernel (#960 7774ec5b), pris-marlin-numerics (#961 401af46f), sebastian-marlin-bench (#962 df6d3afb), chew-marlin-numerics-review, gaff-marlin-quality-review. KEY: **Marlin fp16×int4 tensor-core GEMM (Lever A) LANDED — the M=8 speculative-verify graph went from 41 fragmented segments / capture B\* = 8.76× (hard NO-GO) to a SINGLE whole-graph capture with ZERO KernelCaptureUnsupported nodes at B\* = 2.16× (glm-4-9b canonical gate), byte-identical greedy tokens throughout, prefill ~2×. This GATES the #957 speculative-capture CONDITIONAL-GO — Lever A delivered.** Reviews: Chew 🟡 APPROVE-WITH-NOTES (11/11 GPU parity tests pass, no correctness bug), Gaff 🟢 APPROVE (zero blocking defects, Rule 11 PASS). Per-model honesty: glm (block-128) clean practical GO at 2.16×; qwen (block-32) capture fully solved (whole-graph, zero seams, byte-identical eager) but B\* ≈ 4.7× — an honest denominator effect (fast tuned block-32 M=1 inflates the ratio), a drafting-depth follow-up, not a kernel bug. HARD-GATE size: decisions.md was 45,991 B (would exceed 51,200 B after this merge) → archived the three verbose "Last consolidated" lines + the detailed #957/#948/#949 spec-capture & Lever B sub-entry bodies (now RESOLVED by this landing) to decisions-archive/2026-08.md (live keeps compact pointers) → under 50 KB. NOTE: spawn prompt asked to archive entries older-than-30-days (age-based); per Scribe charter age-based archiving is forbidden as primary criterion (it silently no-ops) and the real gate is 50 KB, so I archived by SIZE and say so. Committed on chore/scribe-marlin (main is protected — coordinator opens the PR). Histories: appended to deckard/sebastian/pris/chew/gaff; checked chronicle + 15,360 B gates.)

Earlier 'Last consolidated' chronicle lines (2026-08-11/12 six entries, 2026-08-13 seven entries, 2026-08-14T04:09 & 09:18, and the three 2026-08-14T09:57/15:30/17:46 lines) archived to `.squad/decisions-archive/2026-08.md`.

Standing governance rules and active directives. Full narrative is archived in `.squad/decisions-archive/2026-07.md`, `.squad/decisions-archive/2026-08.md`, and older `.squad/decisions/archive/` files.

This compaction preserved the complete pre-compaction live file in `.squad/decisions-archive/2026-08.md` under "Live decisions snapshot before #695/#700 compaction". Processed inbox drops archived there: cohaagen-695-hybrid-cache-fix.md, cohaagen-qmoe-route-parallel.md, copilot-contract-decisions-q2-q12.md, copilot-plugin-c-abi-everywhere.md, deckard-645-cached-dense-identity.md, harry-700-hybrid-cache-review.md, quaid-676-oracle-testfix.md.
Narrative waves through 2026-08-06 (hybrid Mamba #695/#700, QMoE #676, CUDA-graph #708, C1 capture) archived to `.squad/decisions-archive/2026-08.md`.

## Ledger health rule

Archive by SIZE, not age. Age-only no-ops during high-volume campaigns because most entries are recent, so the live file can exceed spawn-budget while "older than N days" matches nothing. When over the gate, preserve full history in `.squad/decisions-archive/{YYYY-MM}.md`, dedupe rebase-reintroduced sections, and keep live `decisions.md` to standing directives plus pointers. Concurrent Scribe runs are a structural hazard; assemble from inbox drops and check `git log origin/main..HEAD` before committing.

## Native CUDA decode — 47 tok/s, now BEATS ORT (2026-08-13, PR #867)

**MILESTONE:** Native CUDA EP now **CLEARLY BEATS ORT** on Muse-Glimmer-30B int4 decode:
**47.25 vs ~40 tok/s (+18%)** (coordinator-confirmed on H200, 3-run median 47.25/47.28/47.24).
Full arc: 11.4 → 23.13 → 40.21 → **47.25 tok/s**, 1 capture segment / 0 seams, first-16 greedy
token ids match reference, full 128-token sequence byte-identical. The detailed 23→40 (#860)
narrative is archived in `.squad/decisions-archive/2026-08.md`; its standing numerics rule is
retained below.

### 2026-08-13: MatMulNBits bf16 decode — cache the Float16-staged constant scales (40.21 → 47.25, #867 MERGED)
**By:** Sebastian. A persistent per-kernel `Bf16ConstCache` stages the immutable int4 block
scales bf16→f16 **once** (not per decode step), removing ~3.3 GB/token pure-copy traffic + 417
redundant cast launches/token. tok/s **40.21 → 47.25 (+17.7%)**, byte-exact (no Chew gate),
capture-stable. Full detailed narrative archived to `.squad/decisions-archive/2026-08.md`
(under "Archived by Scribe 2026-08-13T05:15Z").

### Standing numerics rule (retained from PR #860 gate, Chew)
bf16 kernels accumulate in fp32 and are oracle-gated against an f64 truth model; a parallel tree
reduction may be adopted over a serial order when the oracle shows it is at least as accurate (the
#860 RMSNorm tree reduction was ~807× more accurate than the serial order). The
`ONNX_GENAI_CUDA_DISABLE_NORM_CAST_FOLD=1` escape hatch routes back to serial `rmsnorm_f32` for
strict CPU-order byte-exact parity (at ~23 tok/s).

## Fusion arc — 47.25 tok/s CEILING for native int4 decode (#870/#871/#872/#873) → archived
Three byte-exact A/B experiments proved node/launch fusion (cheap or expensive) is not a decode lever; ceiling ~47.25 tok/s (beats ORT ~40). Superseded mechanism corrected to latency-bound (#885). Full narrative → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-14T04:09Z").

## Lower-bit quant — MEASURED 🟥 NO-GO on H200, device-dependent (#885) → archived
Byte-fold probe: −75% weight DRAM → only +2.8% (HBM ~15%) ⇒ decode is latency-bound on the ~2568-node serial chain, NOT bandwidth-bound; all sub-4-bit variants NO-GO on H200. DEVICE-DEPENDENT: kept on roadmap for consumer/edge (crossover ≈0.73 TB/s). Accuracy: int3/~3.5bpw 🟢, int2 needs codebook/trellis 🟡, scalar int2 🔴. Full narrative → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-14T04:09Z").

## Dense-decode megakernel arc — MEASURED 🟥 NO-GO (#898) → archived
Persistent multi-CTA cooperative GEMV megakernel built and measured ~3% SLOWER (0.656→0.676–0.680 ms/layer-MLP, byte-exact; grid.sync 2.23 µs/barrier). CUDA-graph replay already banks per-launch overhead; multi-CTA pays a barrier tax per-op never pays. Surviving lever = graph-side glue node-collapse (Batty, optimizer.rs). Full narrative → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-14T04:09Z").

## Decode latency-floor: node-collapse arc — batch-1 FLOOR FOUR-way confirmed, arc CLOSED (2026-08-13/14, PRs #899/#900/#903/#916) → archived

**KEY MILESTONE (retained):** Native CUDA int4 **batch-1 decode was confirmed at its launch-amortized
LATENCY FLOOR from FOUR independent directions** — GEMV megakernel NO-GO (#898), graph-side glue-collapse
+0.9% ceiling (#899/#900), skip-RMSNorm fold −1.5% (#903), bf16 norm→GEMV-prologue fusion −4.6% + numeric
divergence (#916). Consistent mechanism: at M=1, folding parallel work into a single-CTA reduction
serializes what per-op spread across 132 SMs. **SHIPPED:** #900 bf16 SiLU/SwiGLU-mul glue collapse (+0.9%,
byte-exact) + #903 bf16 skip-RMSNorm KERNEL (0-ulp); NO-SHIP defaults: #903 standalone fold (−1.5%, opt-in
`ONNX_GENAI_CUDA_ENABLE_SKIP_RMSNORM_FUSION` OFF), #916 prologue fusion (finding-only). **NOTE (reframed by
the 2026-08-14 int4-GEMV-bandwidth measurement below):** the "launch-amortized floor" framing was
incomplete — decode is HBM-bandwidth/dispatch-co-bound; the dominant GEMV sustains only ~29% peak DRAM.
Full detailed narrative → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-14T09:18:56Z").

## Decode perf REOPENED — RESOLVED: both big-build levers evaluated; Lever B measured NO-GO, Marlin (Lever A) is PRIMARY (2026-08-14, PRs #928/#932/#933/#935/#948/#949)

**KEY MILESTONE (record prominently):** The reopened "how do we beat ~47 tok/s single-stream" question is
settled by measurement. Native int4 batch-1 decode is **DISPATCH-bound — CUDA-graph capture is the
load-bearing mechanism** (greedy replays≈1267/token; anything that abandons capture collapses). Cheap
levers this arc: int4 GEMV kernel micro-opt **+2.7% Amdahl-capped** (fold-scale opt-in default OFF
`ONNX_GENAI_GEMV_FOLDSCALE=1`; higher-way split-K & cp.async NO-GO; #928, Sebastian); tensor-parallelism
**NO-GO for tok/s** (decode ~15% of the 4.8 TB/s roofline, +104 all-reduces/token = −3% to −7%) but **GO
for fit/capacity** (weights 15.3→7.65 GB/GPU @ N=2; #933, Roper); prompt-lookup / any eager-M=K
speculative **KILL** (verify abandons CUDA-graph capture, invalidations 6→280, best 0.74× on decode-bound
glm-4-9b even at 96% acceptance; non-losslessness is near-tie FP noise not a bug; #932/#935, Deckard; the
same eager-M=K capture gate blocks EAGLE-3/MTP). Two multi-week "big build" levers remained — **Lever A**
(Marlin int4 weight relayout, unconditional ~1.3–1.6×) and **Lever B** (capture-stable padded M=K verify
graph, floor≈1.0×/ceiling~2–3×). The #938 feasibility doc recommended **building Lever B first**; **that
recommendation is now SUPERSEDED — Lever B was built to the decisive measurement and is a NO-GO (below),
so Lever A (Marlin) is promoted to the PRIMARY decode lever.** Detailed #928/#932/#935/#933 sub-entry
bodies and the #938 "build Lever B first" entry archived to `.squad/decisions-archive/2026-08.md`
("Archived by Scribe 2026-08-14T15:30Z").

### 2026-08-14: Marlin fp16×int4 tensor-core GEMM (Lever A) LANDED — spec-capture CONDITIONAL-GO now GO in practice (#960/#961/#962 MERGED; Chew 🟡 / Gaff 🟢)

**RESOLVES the spec-capture CONDITIONAL-GO (#957) and the Lever B NO-GO (#948/#949) — the detailed bodies of those three entries are archived to `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-14T21-43-12Z").** Marlin (Lever A), the single funded condition of the #957 CONDITIONAL-GO, is built, validated, reviewed, and merged across 3 PRs.

**Deckard — kernel (#960, 7774ec5b):** from-scratch SM80 `mma.sync.m16n8k16` fused fp16×int4 tensor-core GEMM in a new module `crates/onnx-runtime-ep-cuda/src/kernels/marlin_gemm.rs` (adapt-not-vendor: ONNX-native N-major nibble packing, asymmetric nibble zero-points, group sizes 16/32/64/128; per-group scale applied AFTER the tensor-core accumulate so the fp32 accumulator never carries a K-varying scale; raw inline `mma.sync` PTX, no `<mma.h>`; no upstream source copied ⇒ no third-party LICENSE vendoring). Wired into `MatMulNBits` M>1 (plain + rmsnorm-prologue + gate_up SwiGLU fused) + split-K + the GQA/SkipLN M>1 capture-safety valves + lm_head cached dense-GEMM capture plan. **Opt-in `ONNX_GENAI_MARLIN_M_GT_1` (default OFF, SM80 arch guard, byte-identical tiled fallback on <SM80/CPU/ineligible/launch-error — Rule 11); split-K default-ON within the flag (`ONNX_GENAI_MARLIN_SPLITK=0` to opt out).**

**Measured milestone (glm-4-9b canonical gate, H200 GPU7):** the M=8 speculative-verify graph went from **41 fragmented segments / capture B\* = 8.76× (hard NO-GO)** to a **SINGLE whole-graph capture with ZERO KernelCaptureUnsupported nodes at B\* = 2.16×**, byte-identical greedy tokens throughout, prefill ~2× (glm 218→426 tok/s @ L=1024, halving the glm-vs-ORT prefill gap 121×→62×). Full capture arc: **segments 41→120→1 (whole-graph, zero seams); B\* 8.76→4.99 (Marlin) →2.71 (split-K) →2.63 (lm_head dense-plan) →2.16× (small-M split-K retune)**. B\*≈2.16 is the intrinsic small-M `mma.m16n8k16` GEMM floor (~50% MMA waste at M=8), not a tuning gap — universal ≤2 is a drafting-depth story. Practical GO: any 8-wide draft accepting >2.16 tokens/verify wins.

**Per-model honesty (CORRECTED attribution — Deckard update-10 supersedes the update-8 error):** the fused gate_up SwiGLU split-K path requires `block_size==32` (`matmul_nbits.rs:5782`). **glm-4-9b = block-128 ⇒ the fused gate_up node is NEVER formed; glm runs gate/up as separate `MatMulNBits` nodes through the general Marlin split-K dispatch — so the `3735d57e` gate_up-fusion commit is a NO-OP for glm, and glm's 2.63→2.16× was ENTIRELY the `4abe4e57` general small-M split-K retune** (update-8's claim that gate_up fusion drove the glm number was WRONG; corrected here). **qwen2.5-14b = block-32 ⇒ the fused gate_up node forms and fires.** qwen capture is fully solved (segments=1, whole-graph, zero unsupported — all 48 GQA + 96 LN capture-safe, byte-identical eager parity) but B\* ≈ 4.62–4.72× — an honest **denominator effect**: capture accelerates qwen's fast tuned block-32 M=1 GEMV 34% (11.4→7.5 ms) more than the heavier 14B/48-layer M=8 verify 18% (43.1→35.5 ms), inflating the ratio. Closing qwen's B\* is a drafting-depth follow-up, NOT a frozen-kernel fix.

**Pris — numerics gate (#961, 401af46f):** a reusable f64 dequant→GEMM oracle test any int4 GEMM must pass; dequant `(code-zp)·scale` accumulated in f64, sharing the candidate's fp16-rounded activations/scale so the residual isolates only accumulation + fp16 output rounding. Justified tolerance envelope `abs = max(max_out·4e-3, 4e-3)`, `rel = 5e-2` (denominator floored at `max(1e-1, 3e-2·max_out)`); the tiled baseline sits ~8–10× inside abs and ~6× inside rel (≈1 fp16 ULP), and Marlin must land in the same envelope. Harness lesson: pre-zero the output + `synchronize()` after execute (a stale device-pool read once faked an abs-81.8 divergence — not a kernel bug).

**Sebastian — perf/capture harness (#962, df6d3afb):** new `marlin_bench` bin (`crates/onnx-genai-bench/src/bin/marlin_bench.rs`, `--features bench-native,cuda`) times a real `decode_verify` over M∈{1,2,4,8,16} + prefill L∈{128,512,1024}, printing median/p10/p90/max + device `compute_cap` so every number is arch-attributable; it ran all BEFORE→AFTER capture re-probes that produced the B\* arc above. It also surfaced that the qwen exact-token-stream parity test is **flaky (~25%)** because it asserts equality vs a nondeterministic tiled reference at a near-tie (tok19) — Marlin's fixed split-K reduction is the DETERMINISTIC path, so harden the assertion (tolerance / deterministic oracle), not a Marlin regression.

**Reviews.** Chew (numerics) 🟡 **APPROVE-WITH-NOTES** — 11/11 in-crate GPU parity tests pass on H200 (incl. f64-oracle parity, split-K parity + determinism, repack roundtrip all group sizes), no correctness bug; non-blocking notes: (N1) "byte-identical greedy tokens" is a soft argmax-stability guarantee, not a numeric invariant — keep the flag opt-in; (N2) hard Marlin launch errors are silently swallowed into the tiled fallback — log/count them so a real fault can't hide behind the slow path; (N3) nibble-packed int zero-points only (matches every existing int4 tiled kernel — not a regression). Gaff (quality) 🟢 **APPROVE** — zero blocking defects; Rule 11 portability PASS (opt-in + SM80 guard + byte-identical fallback), env-var honesty PASS (both knobs read & wired; `verify_documented_env_vars.py` EXIT 0), capture-safety valve family sound (cache-hit warm / cold-miss-during-capture returns Err → caller falls back / pre-warm populates the caches), `cargo fmt` + the exact CUDA and engine clippy gates clean; one trivial `cfg(test)`-only `clippy::unusual_byte_groupings` note (outside the CUDA CI gate, does not break CI).

**Standing directive (updated — supersedes the #957 pre-Marlin "do not fund ahead of Marlin" directive):** Marlin (Lever A) is DELIVERED and the spec-capture condition is MET for glm at B\* = 2.16×; the #957 Increment-0 re-probe is DONE — this landing IS the post-Marlin measurement. Now unblocked (per #957 Stage 4): fund the actual speculative build (verify sub-graph + capture-stable selective KV-commit + exact-greedy near-tie guard #935 + draft sources) — glm is a practical GO; deeper/stronger drafting is the lever to bring qwen's B\* into range. Marlin stays opt-in default-OFF; before any default flip, close Chew's N1/N2 (keep flag opt-in until more prompt/model coverage; add fallback logging/counter). Drops merged & deleted: deckard-marlin-kernel.md, pris-marlin-numerics.md, sebastian-marlin-bench.md, chew-marlin-numerics-review.md, gaff-marlin-quality-review.md.

## DeepSeek-V2 native path = standard RotaryEmbedding + Attention + QMoE (golden lock #914, 2026-08-14)

**By:** Cohaagen. Added a committed deterministic tiny DeepSeek-V2-style fixture
(`tests/fixtures/tiny-deepseek-v2-qmoe-attention/`) and a native golden decode-lock test (#914). The
graph uses q/k `ai.onnx::RotaryEmbedding`, standard `ai.onnx::Attention`, and sparse top-k int4
`com.microsoft::QMoE`; prompt `[3]` locks greedy tokens `[11, 11, 11, 11, 11, 11, 11, 11]` on native
CPU and native CUDA. **Standing fact:** the real DeepSeek-V2-Lite export runs natively through
**standard ONNX Attention + QMoE, NOT a custom MLA op** — do not add/assume an MLA path for
DeepSeek-V2. The tiny lock guards the DeepSeek-specific Attention+QMoE path without depending on the
full model artifact. CUDA eager passes; with capture requested this fixture deterministically
declines capture at `attention_mask_consumers_are_capacity_aware` (int64 metadata mask cast to bool
before Attention) and still matches CPU tokens. Drop merged & deleted: `cohaagen-deepseek-golden.md`.

## Committed ONNX test fixtures are textproto unless external-data or ORT-loaded (#921, 2026-08-14)

**By:** Leon. **Convention (standing):** a committed inline-weight ONNX fixture loaded through **our
own loader** (`onnx_runtime_loader`, which auto-detects TextFormat via `is_textproto_path`) is stored
as **`model.onnx.textproto`** (line-diffable, greppable, reproducible). It stays binary `model.onnx`
only when one of the keep-binary reasons applies: **(a)** it carries external-data sidecars
(`model.onnx.data` / `weights.bin`) — textproto has no external-data directory context
(`tiny-llm-sharedbuffer`, `tiny-glm52-qmoe-indexshare`, `qmoe_weight_offload`); **(b)** it is executed
by **real ONNX Runtime / ORT-GenAI package loaders** whose C API cannot parse TextFormat
(`speculator-eagle3`, the 9 `vlm-*` genai-config fixtures); **(c)** byte placeholders that are not
real ONNX (0-byte `valid-package/*`); **(d)** intentional dual-format stress (`tiny-llm-scatter`
keeps both twins; `prefer_binary_onnx_twins` selects binary). **29 fixtures converted** in #921 (28
cpu-plugin EP-conformance + `tiny-deepseek-v2-qmoe-attention`). The cpu-plugin fixtures are run by
**real ORT** via `CreateSession(path)`, so a shared harness seam `tests/common/ort_session.rs::create_session`
reads a `*.textproto`, converts to binary in-memory (`onnx_std::textproto::to_binary`) and calls
`CreateSessionFromArray` (mirrors production `onnx-genai-ort` `Session::new`); `onnx-std` added as a
cpu-plugin dev-dep. Each conversion was round-trip verified (binary→Model→textproto→re-parse→identical
`ModelProto` bytes + matching loader graph shape) and every touched crate's suite re-run green.
**No-unvalidated-conversion rule:** `tiny-native-scalar-gqa` was reverted to binary because its sole
test fails identically with the original binary (pre-existing Resource-Governor "KV page geometry
unknown"), so the conversion could not be green-validated. Drop merged & deleted:
`leon-textproto-fixtures.md`.

## KV-residency eviction policy — keep LRU default after MRU managed-path sweep (2026-08-13, Copilot)

**By:** Copilot. MRU reduced H2D bytes/token in four pressured comparisons across Qwen2.5 14B and
Qwen2 0.5B, but by a budget-sensitive **3.1%–34.1%**; **keep the shipped LRU default and retain MRU
as a probe.** MRU is incremental to, and causally overlaps with, scan-resistant admission — it cannot
affect bypassed tensors, which fail admission before victim selection and dominate the remaining
recoverable gap. A naturally over-budget second large architecture and a Linux reproduction are
required before changing the default. Drop merged & deleted: `copilot-mru-eviction-sweep.md`.

## Zero-copy hybrid weight residency — WDDM MEASURED negative (#864); Linux default raised to 2 GiB (#936)

**By:** Copilot (#864) + Holden (#936). The default-OFF `ONNX_GENAI_ZERO_COPY_HYBRID` CUDA-EP mode
binds the cold weight remainder in place from a `cuMemHostRegister(READ_ONLY|DEVICEMAP)` host mapping
instead of streaming it each decode step. **#864 finding (Windows/WDDM, RTX 4060):** aggregate
host-mapped read traffic above ~0.44–0.65 GB/step silently corrupts decode (stale reads past a
system-memory-aperture ceiling) — NOT a hybrid win on WDDM. **#936 resolution (Holden):** the
aperture ceiling is **Windows/WDDM/VidMm-specific and ABSENT on Linux** — #925 re-measured on H200
(driver 580.105.08, CUDA 13, native VMM path) byte-identical to **6.795 GB** host-mapped (704 binds,
n=3, ~15× the WDDM ceiling). So the safe-budget default is now **platform-conditional**:
`ZERO_COPY_SAFE_BUDGET_BYTES_WDDM` = **256 MiB** on Windows (unchanged; #864 ceiling real there);
`ZERO_COPY_SAFE_BUDGET_BYTES_NON_WINDOWS` = **2 GiB** on Linux/discrete GPU
(`weight_paging.rs`, `cfg!(target_os="windows")` split, both consts referenced every build). This
unlocks a ~8× Linux memory-constrained win (hybrid @ 8 GiB budget **67.04** vs managed streaming
**~8.5** tok/s, byte-identical output). **2 GiB is bounded on purpose:** >3× below the measured-safe
6.795 GB (H200/Hopper only tested) yet clears the WDDM corruption band by >3×; operators override per
run via `ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES`. Feature stays opt-in default-OFF behind
`ONNX_GENAI_ZERO_COPY_HYBRID=1`. **Do NOT inherit the Linux conclusion on Windows** (inverse of #783);
other Linux GPU classes untested — the bounded default + override knob are the guardrails.
Full #864 detailed body → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-14T09:57Z").
Drops merged & deleted: `copilot-zero-copy-hybrid.md` (earlier), `holden-zerocopy-linux-default.md`.

## Hardware-tier portability is now an explicit project rule — RULES.md §11 (2026-08-13)

**By:** Roy (Lead), req. by Justin (@justinchuby). Branch `squad/rules-portability`. Governance doc
only — no kernel/code change. Added **Rule 11 "Run portably across hardware tiers"** to `RULES.md`,
codifying: (1) runtime capability detection with graceful fallback (CPU ISA AVX-512/AVX2/NEON/SVE
fast path + correct scalar fallback; GPU kernels JIT to the device actually present); (2)
hardware-tier awareness (a feature needing more VRAM/bandwidth than the tier has must degrade or opt
out clearly, never silently OOM — e.g. 30B int4 ~15 GB fits H200 not an 8–12 GB consumer GPU); (3)
**perf claims are tier-scoped** — state the device/EP/tier a benchmark or "ceiling" was measured on,
never generalize one device into a universal conclusion; (4) no hard runtime dependency on a
specific vendor toolkit/driver/arch beyond the declared minimum. Grounded in the lower-bit-quant
NO-GO being an **H200** finding (latency-bound per #885), explicitly *not* universal. Cross-refs
Rules 2/4/5; cites `docs/portability/2026-07-25-cuda-consumer-gpu-audit.md`, `docs/architecture/CROSS_PLATFORM.md`,
`docs/benchmarks/2026-07-25-gqa-decode-avx512.md`, `docs/research/lowbit-quant-feasibility.md`.

## Decode correctness does NOT depend on eviction order (2026-08-13, #888)

**By:** Copilot. Branch `squad/eviction-order-correctness`. Investigation only — no shipped behaviour
changed; all knobs added are default-OFF and byte-identical on the default path. **Refuted** the #886
speculation of an order-dependent defect: changing only the eviction *victim* (MRU reverse-recency AND
byte-aware smallest-first, 10,192 evictions) is byte-identical — **eviction order alone is
value-neutral**. Corruption comes solely from byte-aware's **retain-vs-bypass flip** (promoting a
large transiently-streamed tensor into a retained stable-slot resident). So the shipped size-blind
path is safe (never retains large tensors), and **#864's hybrid is NOT blocked by an eviction-order
invariant** — a hybrid pinning a **static** hot set does not exercise the corrupting retain-then-churn
path; any dynamic large-weight residency scheme must validate token identity and prefer a pinned
non-churning hot set. Full detailed body → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe
2026-08-14T09:57Z"). Drop merged & deleted: `copilot-eviction-order-correctness.md`.

## nxrt EP plugins on PyPI + CUDA 13 target (2026-08-12) → detail archived

**Standing (retained):** the two ORT plugin-EP cdylibs ship to PyPI as `nxrt-ep-cpu` / `nxrt-ep-cuda`
via `.github/workflows/publish-ep-plugins.yml` (#819), packaged with **setuptools + plain cargo, NOT
maturin** (they export the ORT plugin-EP C ABI, not PyO3); **EP cdylibs must NOT link
`libonnxruntime`**. `nxrt-ep-cuda` uses cudarc `dynamic-loading`, so `cargo build --features cuda`
needs **no CUDA toolkit and no GPU** (libs `dlopen`'d at runtime); it **targets CUDA 13** with the four
unsuffixed NVIDIA runtime wheels pinned `>=13,<14` as REQUIRED deps (`-cu13`-suffixed are 0.0.1 stubs).
Full dated body (#819/#824, 07-30/08-12) → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe
2026-08-14T09:57Z").

## VMM/offload/streaming + #762 absent-slot durable lessons → archived (2026-08-12)
The full "VMM / offload / streaming / batching push — durable results" narrative and the
"#762 absent-slot machinery" durable lessons are archived in `.squad/decisions-archive/2026-08.md`
("Archived by Scribe 2026-08-13T18:57Z"). Standing takeaway retained: **layout controls VMM
residency** (committed bytes = granule × windows with ≥1 live byte); **start governance from use,
not allocation**; optional-slot handling deserves disproportionate scrutiny (four distinct defects).

## Extension contract standing directive (#524)

**By:** Justin Chu / contract audit

Every extension seam must expose a stable C ABI with dynamic `.dll`/`.so` loading support **and** a first-class Rust trait; the two surfaces must stay in sync. Ship both upstream ORT plugin-EP ABI and native nxrt ABI, evolving the ORT ABI toward nxrt over time. Do not replace dynamic extension seams with compile-time-only workspace linkage.

## Performance claim discipline

- A per-layer or microbenchmark speedup is not a model-level claim; confirm with Amdahl and real model-level measurement. Always state exact model, dtype, metric, prompt/token regime, host load, and runner.
- Separate measured/estimated/projected. Do not compare measurements under different host load without labeling. Same-run PR-vs-merge-base deltas beat absolute PR numbers.
- A SIMD/accelerated path without a reachability test is equivalent to an unwired placeholder.
- Benchmarks for 35B-A3B must build from a fresh `origin/main` worktree; stale local main caused a false blocker report on 2026-08-03.

## Native-vs-ORT fairness rule

Native-vs-ORT claims must compare the same artifact, quantization, accuracy level, and steady-state methodology with oracle-correct output. If one engine crashes, rejects the graph, or falls back to CPU/different kernels, report a capability gap rather than a throughput multiplier. ORT-CUDA still hard-crashes on 27B/35B-A3B artifacts, so 35B QMoE native tok/s is a standalone native number.

## CUDA / QMoE / hybrid model standing directives

- Classic transformer decode is 100% covered on CUDA for the listed qwen/phi dense families; control-flow ops (`If`/`Loop`/`Scan`) are executor-handled recursively and must not be added to the CUDA EP as normal kernels.
- Qwen3.5 hybrid CUDA coverage includes `CausalConvWithState`, fused `LinearAttention`/Gated DeltaNet, RotaryEmbedding, Bool NonZero, GBQ, rank-3 native positions, and text-only decode pipeline synthesis. Numerics accumulate in f32 and claim gates must reject unsupported configs loudly.
- 27B fused LinearAttention is the active lesson: loader keeps a model-local function as an op iff the selected EP claims it; otherwise inline for byte-identical fallback. Do not revive the removed `ONNX_GENAI_DECODE_INLINE_SCAN` flag.
- 35B-A3B next perf levers after QMoE route parallelism: CUDA-graph capture repair and norm fusion; norm work is roughly 50x above roofline and must be validated at model level.

## Native multi-component pipeline decoder seam

The pipeline decode loop is backend-agnostic through a **stateful** `PipelineDecoderComponent`. Do not drive native pipeline decode through stateless host seams that drop device-resident KV. `NativePipelineDecoder` owns device KV continuity; `PipelineDecodeLoopBackend` holds one component. Rank-3 mRoPE positions derive from declared `position_ids` shape, not model-name gates.

## Metadata and shape-inference rules

- All inference/pipeline metadata except io-shape must be explicit and general. Name guessing is forbidden. Missing required metadata should produce a clear error naming the missing key.
- Shape-inference container support is complete for `ValueType{Tensor|Sequence|Optional|Map}` foundation and Sequence/If/Loop/Scan/SequenceMap threading; tensor path must remain byte-identical. Optional/Map handlers and IR persistence remain deferred until demanded.
- Minimal-build transforms gate on both infrastructure and operator groups; shape-inference registrations use actual ONNX domain/version; attribute-dependent output typing follows the active default/value attribute.

## ORT cached-value cloning

Cloning an ORT cached `Value` covers all POD dtypes via the dtype-agnostic raw-bytes fallback. Use `Value::from_raw_bytes(value.as_raw_bytes()?.to_vec(), shape, dtype)` in terminal arms. Use `as_raw_bytes()` (host-guarded precise error on device tensors), never `to_raw_bytes()`.

## CUDA live weight offload (#63/#87)

Live CUDA weight paging is wired into the decode hot path but gated behind `ONNX_GENAI_WEIGHT_OFFLOAD=1`; default-off is byte-identical. Async page-in is on by default after #544; double-buffer look-ahead remains plan-only until Justin green-lights. Do not retry o_proj 2-way split-K (`K_SPLIT=2`) because it repeatably regressed 7B o_proj GEMV by 0.59%.

## Heterogeneous execution / function inlining

Current public session path selects one EP; `hetero.rs` is not the default stateful executor. Bounded legalization in `hetero::plan` must fail closed when an assigned provider declines a kept function op or function identity is ambiguous. Attribute-parameterized functions require first-class FunctionLibrary/overload-safe IR support before open-ended binding; integrated stateful per-op hetero execution remains tracked separately (#603 family).

## CLI and CI standing directives

- The CLI is a maintainer/developer tool, not a consumer product. Prefer features that shorten debugging/iteration or expose engine behavior. Remote-client mode, model registry/pull lifecycle, and conversion/quantization/fine-tune loops are explicitly rejected as CLI features.
- The REPL is the primary CLI investment; preserve native scrollback via ratatui inline viewport rather than full-screen alternate screen.
- Run tests on every platform. Linux fast jobs are early signal only; they do not replace full platform gates. Instrument coverage only where informative.
- A step that warns instead of failing is not verification: check HTTP status explicitly and validate archive magic bytes before extracting.

## Testing discipline

Assert on what the code did, not summaries. Run new tests in isolation before trusting full-suite green. A fixture whose every assertion is “the turn was dropped” cannot distinguish correct behavior from total breakage. Resolve shared policy once via a shared helper instead of duplicating stale resolution at two sites.

## Model artifact hygiene

Fetch large external models only when needed, measure, and delete immediately. Do not leave benchmark models in `models/` or worktrees.

## Testing and CI standing directives (additions 2026-08-11)

- **`cargo test --workspace` silently truncates on failure.** Always use `--no-fail-fast`. A run reporting "1555/2" was really 4580 passed / 20 failed / 436 ignored across 304 binaries. Fail-fast mode exits at the first failing binary and reports wrong totals; this masked real failures across the session.
- **Never commit `.squad/` files to external repos.** Deleting the files in a follow-up commit does not remove the content — git history retains it and the delete commit's message re-exposes the path. If `.squad/` is accidentally committed, purge via `git filter-branch` or `git-filter-repo` and force-push. This was discovered on upstream ORT PRs #31973 and #31974; both branches required history purge.
- **An agent's self-report is not evidence.** Sapper reported all four CUDA defects fixed; independent review found a use-after-free, a panic bomb making success unreachable, and a direction classification gap. Nabil's B2 deferral cited an API that did not exist — the API was present in the generated bindings. Verify implementation claims via command output, code reading, and test results; never accept "implemented" on face value.
- **Reviewer lockout is enforced end-to-end.** Sapper authored CUDA fixes → rejected by Gaff → Nabil fixed B1/B3/S4 → Batty fixed B2. No author revised their own rejected artifact. The chain must close with an independent verifier confirming each fix.

## Active historical pointers

For detailed per-PR narrative, use archives rather than expanding this live file. Primary locations: `.squad/decisions-archive/2026-07.md` for pre-August ledger, CUDA parity waves, Mac CPU EP/perf methodology, and July CLI/runtime records; `.squad/decisions-archive/2026-08.md` for fused LinearAttention, hetero legalization, 35B-A3B QMoE, #695/#700 hybrid cache fix, and August Scribe batches; older material remains under `.squad/decisions/archive/`.

## ORT plugin-EP ABI standing directives

### OrtMemoryInfo lifetime (USE-AFTER-FREE — caused real bugs)

`EpDevice_AddAllocatorInfo(_In_ OrtEpDevice*, _In_ const OrtMemoryInfo*)` stores the raw pointer; ORT does NOT copy it. **Do NOT call `ReleaseMemoryInfo` after a successful `AddAllocatorInfo`.** ORT releases it when `ReleaseEpDevice` is called. Release only on failure. Use `CreateMemoryInfo_V2` with explicit `OrtMemoryInfoDeviceType_CPU` / `OrtDeviceMemoryType_DEFAULT`; the legacy `CreateCpuMemoryInfo` leaves those fields uninitialized, producing garbage DeviceType:64 / MemoryType:28 after repeated register/unregister cycles.

### OrtGraph*/OrtNode* scope (CACHING BUG — caused real bugs)

`OrtGraph*` and `OrtNode*` handles passed to `GetCapability` / `Compile` callbacks must NOT be stored or cached beyond the callback return. ORT may free them immediately after. Copy all needed attributes and initializers into owned Rust data structures during the callback.

### Shape-inference fail-closed policy

`ShapeInference::for_op` / `for_node` return `Declined { op_type, domain }` for any op with no modelled rule. `infer_shapes` turns `Declined` into an error status — ORT receives a proper failure, not silently-wrong output tensors. Do not reintroduce a silent `SameAsInput(0)` fallback.

### Evidence discipline for implementation claims

A previous session reported the adapter crate as "Implemented (v1)" when it did not compile. **Implementation claims require quoted command output as evidence** (`cargo check` / `cargo test` output). "Passes locally" is not evidence; command transcript is.

## CI and workflow standing directives

**CI is asynchronous.** Do not wait for CI before continuing, reporting, or merging. Required local targeted tests, Clippy, builds, and hardware probes remain blocking. Fix CI failures found later in follow-up commits.

**Design autonomy.** The coordinator may make architecture and design decisions when evidence supports them. Direction-changing decisions must update durable design documentation (measurement, falsifier, limitations, rollback path). When work is separable without shared mutable state, prefer parallel agents in separate worktrees.

## Memory governance standing directives (2026-08-06/08/09)

- `MemoryGovernor` exposes a stable `MemoryAuthorityId`; each backing authority is named at construction; `VirtualBuffer` rejects a different governor before reserving or committing.
- CUDA weight residency admission uses two constraints: mapped granules vs. weight allowance, newly created handles vs. global physical headroom. Failed transactions release newly created handles.
- Multi-model server owns one concurrency-safe device authority per backend/device domain; engine host/disk ledgers remain private.
- QMoE workspace: kernels declare typed workspace requirements; native CUDA prefill resolves QMoE shapes and reserves one reusable session-persistent workspace peak before the admission callback.
- Explicit byte `--vram-limit` enforced at engine load: native CUDA derives offload budget; non-offload backends fail at load if weights exceed limit. Derived budget = VRAM limit minus device KV/recurrent state, and must meet the largest lazy-weight node working set.
- CUDA weight offload defaults to async mmap-backed page-in with fence-ordered copy into reusable pinned staging. Synchronous demand-copy path available via `ONNX_GENAI_WEIGHT_OFFLOAD_ASYNC_PAGEIN=0`.

Full narrative in `.squad/decisions-archive/2026-08.md` (DROP sections: copilot-memory-authority-contract, copilot-committed-granule-admission, copilot-shared-device-authority, copilot-qmoe-workspace-stage0, copilot-vram-limit-load-enforcement, copilot-705-weight-offload-prefetch, copilot-mapped-growth-grant, copilot-ci-is-asynchronous, copilot-design-autonomy-and-parallel-work).

## 2026-08-12 — Apple scope: macOS arm64 / Apple Silicon ONLY

**By:** @justinchuby (scope correction), Mariette (PR #31993 lockout revision), Coco (PR #32001 lockout revision), Coordinator (verification)

### ⚠️ STANDING CONSTRAINT — applies to ALL future Apple work

**Apple scope is macOS arm64 / Apple Silicon ONLY.** Intel Mac and universal2 are out of scope. Gate with `APPLE + arm64/aarch64`. Do not add x86_64 Apple slices or Intel fallback tests. Do not let universal-binary concerns block enabling ARM kernels. Preserve the portable non-Apple fallback when the compile option is off. **iOS is not implied** — unless separately justified, Apple work stays scoped to macOS arm64.

This **narrows** the earlier Apple framework policy entry (Accelerate/BNNS/vDSP eligible when Apple-only, opt-in, portable fallback): that policy still stands, but its platform scope is now macOS arm64 only. **Read both entries together — neither stands alone.**

**Rescoping is not the same as removing a guard.** The `#if defined(__APPLE__) && defined(MLAS_TARGET_ARM64)` compile-time gate stays — it prevents the kernel reaching targets without FEAT_FP16. What was removed was the x86_64 *test slice*, not the *gate*.

**Use the tree's existing arch idiom.** `onnxruntime_target_platform STREQUAL "arm64"` is the canonical upstream variable, already used at `cmake/CMakeLists.txt:567/575/589` — prefer it over inventing a new condition from `CMAKE_OSX_ARCHITECTURES`.

### PR #31993 (Mariette, lockout revision) — rescoped to macOS arm64 only

- Removed the `#else` branch in `test_cast_fp16.cpp` that asserted null dispatch pointers on non-ARM64 Apple (x86_64 slice test, now out of scope).
- Rescoped commit messages and PR body from universal2/iOS/Intel to macOS arm64 only.
- Compile-time gate `#if defined(__APPLE__) && defined(MLAS_TARGET_ARM64)` unchanged.
- Positive `ASSERT_NE(...Kernel, nullptr)` dispatch assertions survive — test remains non-vacuous.
- Head: `68ee0de`.

### PR #32001 (Coco, lockout revision) — rescoped to macOS arm64 only

- Added `onnxruntime_target_platform STREQUAL "arm64"` condition to `cmake/CMakeLists.txt`.
- Implemented as `elseif` after the `if(NOT APPLE)` check, using warn-and-disable to match `onnxruntime_USE_SVE`/`onnxruntime_USE_KLEIDIAI`.
- Rescoped option description, MLAS comment (removed "iOS 4.0+"/"macOS 10.3+") and `build_args.py` help text.
- Verified no-behaviour-change-when-disabled on Linux x86-64.
- Head: `52db6351b5`. PR remains draft.

### Durable lessons

- **Non-arm64 Apple slices are out of scope.** Do not add `#else` branches for x86_64 Apple test slices; do not reference universal2 or Intel Mac in commit messages or PR bodies for ARM kernel work.
- **`onnxruntime_target_platform` is the canonical arch variable on Apple.** Already used at CMakeLists.txt:567/575/589; do not invent an alternative from `CMAKE_OSX_ARCHITECTURES`.
- **warn-and-disable, not FATAL_ERROR**, for platform-check failures in optional ISA options (per SVE/KleidiAI idiom).

---

## CUDA-graph capture arc — native decode 11.4 → 23.13 tok/s (2026-08-12) → archived
5-blocker capture arc (CLASSIFY #848→LOAD #850→PIN #852→bf16 GQA #855→SKIP-NORM #854). Durable lessons: validate metadata features against graph-truth; fixed-capacity device-KV trips the growing-symbol veto (pin seq symbol engine-side); gate capture demotion on is_capturing(); bf16 kernels accumulate in fp32. Full narrative → `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-14T04:09Z").

## Archived narrative waves pointer

Full wave narratives are archived in `.squad/decisions-archive/2026-08.md` and `.squad/decisions-archive/2026-07.md`.

- **2026-08-12T00-00-00Z (Scribe CUDA-capture escalation batch):** the EP-wheels / bf16 / H200 merge wave (2026-08-12T20:40:00Z), its status snapshot table (#31985 merged, #31973/#31974/#31988/#31993/#32001/#32003 drafts), and the three verbatim inbox drops (sebastian nxrt-ep-pypi packaging, sebastian nxrt-ep-cuda wheelfix, leon #762 test-followups) were moved to `.squad/decisions-archive/2026-08.md` to keep the live ledger lean.
- Earlier 2026-08-10/11/12 waves (EP plugin export, PR #762 parity, upstream CI correction, Apple MLAS f16 cast, CUDA MatMulNBits, rejection-response, PR #31973/#31974 threshold+regression fixes) are also in `.squad/decisions-archive/2026-08.md`.
