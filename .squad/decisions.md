# Decisions — live standing directives

Last consolidated: 2026-08-13T17:07:00Z (Scribe megakernel-P2-NO-GO batch, local state; merged 6 inbox drops — sebastian-megakernel-feasibility + sebastian-megakernel-p15 + sebastian-megakernel-multicta (folded into one "Dense-decode megakernel arc" section), sebastian-lowbit-machine-tiers, roy-portability-rule (RULES.md §11), copilot-eviction-order-correctness (#888). KEY SEMANTIC CORRECTION: the persistent multi-CTA cooperative GEMV megakernel — earlier reopened as "the true remaining latency lever" — was BUILT AND MEASURED a 🟥 NO-GO (~3% SLOWER, 0.656→0.676–0.680 ms/layer-MLP, byte-exact 0-ulp; grid.sync 2.23 µs/barrier; PR #898 @ main 0790849c). Annotated all three prior "megakernel is the reopened lever" claims (#885 correction block, lowbit section, lowbit KEY-CONCLUSION) to reflect the NO-GO; the surviving datacenter lever is graph-side glue node-collapse (Batty, optimizer.rs). Lower-bit quant remains H200 NO-GO but is device-dependent (kept on roadmap for consumer/edge). Size gate: after merge decisions.md would exceed 50 KB → archived the 2026-08-12 Profiling gotchas + Parallel-work decisions sections to decisions-archive/2026-08.md → under gate. Histories: appended multi-CTA NO-GO to sebastian, graph-side node-collapse-is-primary-lever to batty; checked chronicle + 15,360 B gates.) 

Last consolidated: 2026-08-13T14:45:00Z (Scribe lowbit-nogo-probe batch @ main 26bd410f; merged 2 inbox drops — sebastian-lowbit-feasibility + fact-checker-lowbit-accuracy — into new "Lower-bit quant — MEASURED no-go; ceiling is latency-bound not bandwidth-bound" section [PR #885]. KEY: byte-fold probe (−75% weight DRAM → +2.8%, HBM util ~15%) REFUTES the earlier "weight-bandwidth-bound" attribution; decode is LATENCY-bound on the ~2568-node serial chain (~8.2 µs/node). Appended a Correction note to the #870/#872/#873 fusion-arc entry (ceiling VALUE + "marginal fusion not a lever" stand; mechanism + lower-bit future-lever were wrong). Also corrected the mechanism wording in docs/PROGRESS.md (#875 lines) + bumped HEAD → 26bd410f. Megakernel/node-collapse REOPENED as true lever. Size gate: after merge decisions.md hit 53,745 B (>50 KB) → archived the detailed #870/#871/#872/#873 fusion-arc sub-entries to decisions-archive/2026-08.md (milestone conclusion + correction kept live) → 49,381 B, under the 50 KB gate. Histories: appended probe+NO-GO to sebastian/fact-checker history.md; checked chronicle + 15,360 B gates.)

Last consolidated: 2026-08-13T05:15:00Z (Scribe fusion-arc batch @ main 887e3742; merged 4 inbox drops — chew-pr871-numerics, batty-fusion-contract, batty-qkv-contract, sebastian-bf16-swiglu-fusion-contract, sebastian-gqa-not-a-capture-lever — into the "Fusion arc — 47.25 tok/s is the architectural ceiling" milestone section [PRs #870/#871/#872/#873]. KEY CONCLUSION: three byte-exact A/B experiments prove native int4 decode of Muse-Glimmer-30B is weight-bandwidth/compute-floor bound at ~47.25 tok/s, NOT dispatch-bound — node/launch fusion (cheap OR expensive) does not help; #873 QKV fusion retained opt-in behind ONNX_GENAI_CUDA_ENABLE_QKV_FUSION=1. Size gate: after merge decisions.md exceeded 50 KB → archived the detailed #867 MatMulNBits narrative to decisions-archive/2026-08.md, kept milestone + standing numerics rule live. Histories: appended #871/#872/#873 + ceiling to sebastian/chew/batty history.md; all < 15,360 B chronicle gate, none summarized.)
Last consolidated: 2026-08-13T04:10:00Z (Scribe CUDA-47tok/s-beats-ORT batch @ main 1002e360; merged sebastian-cuda-matmulnbits-gemv (PR #867, MatMulNBits bf16 constant-scale cache, native decode 40.21 → 47.25 tok/s — native now clearly beats ORT ~40, +18%). Size gate: 49,418 B + drop would exceed 50 KB → archived the detailed 23→40 (#860) narrative to decisions-archive/2026-08.md, kept its standing numerics rule live → decisions.md now 48,855 B, under charter 50 KB gate. Histories: appended PR #867 milestone to sebastian/history.md; checked all histories against the chronicle + 15,360 B gates — none summarized.)
Last consolidated: 2026-08-13T03:03:13Z (Scribe CUDA-40tok/s milestone batch; merged sebastian-cuda-cast-elimination (PR #860) + recorded Chew's PR #860 numerics sign-off — Chew's inbox drop file was absent, decision reconstructed from spawn manifest. NO archive: decisions.md 44,755 B, below charter 50 KB gate. NOTE: the spawn prompt's "archive entries older than 30 days at ≥20,480 B" is an age-based gate the charter forbids — it no-ops since all live entries are 2026-07/08, and 20 KB is below the standing-directive floor. Histories: sebastian 3,948 B / chew 6,951 B, both below the chronicle + 15,360 B gates, none summarized.)
Earlier 'Last consolidated' chronicle lines (2026-08-11/12, six entries) archived to `.squad/decisions-archive/2026-08.md`.

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

## Fusion arc — 47.25 tok/s is the architectural CEILING for native CUDA int4 decode (2026-08-13, PRs #870/#871/#872/#873)

**KEY CONCLUSION (record prominently):** Three independent, byte-exact A/B experiments —
**#870** (GQA / inner-loop cheapening), **#872** (cheap constant-`Add` fold, −208 nodes/token),
and **#873** (QKV projection fusion, −104 **expensive** GEMV launches/token) — conclusively prove
native CUDA int4 decode of **Muse-Glimmer-30B is weight-bandwidth / compute-floor bound at
~47.25 tok/s (H200)**, NOT launch-dispatch bound. Graph node/launch fusion (cheap OR expensive)
does not help because at M=1 decode each disjoint int4 weight is read exactly once — a
DRAM-bandwidth roofline. **To beat 47 you must cut weight BYTES/token** (lower-bit quant,
sparsity) or move to a **different kernel family (megakernel)** — NOT node fusion. Native
decisively beats ORT (**47.25 vs ~40, +18%**). **The perf arc is concluded at the ceiling;
no code perf change shipped, one opt-in pass retained.**

> **Correction (2026-08-13, per probe #885):** the mechanism named above ("weight-bandwidth /
> compute-floor bound") is **wrong**. A controlled weight-DRAM byte-fold probe (#885) measured
> that cutting weight bytes read to 25% raises tok/s only 47.29 → 48.62 (**+2.8%**, HBM util
> ~15%) — decode is **latency-bound on the ~2568-node serial dependency chain (~8.2 µs/node)**,
> NOT weight-bandwidth-bound. The ceiling VALUE (~47.25) and the "marginal node/launch fusion is
> not a lever" conclusion still stand; only the WHY changes. Consequently the "cut weight
> BYTES/token (lower-bit quant, sparsity)" future-lever above is a **MEASURED no-go** (see the
> Lower-bit quant section below). The correction reopened a **decode megakernel** as a
> candidate lever — but that has since been **BUILT AND MEASURED: a persistent multi-CTA
> cooperative GEMV megakernel is a 🟥 NO-GO** (~3% slower, #898; see "Dense-decode megakernel
> arc" below). **The remaining recoverable-overhead lever is graph-side glue node-collapse
> (Batty, `optimizer.rs`), NOT a GEMV megakernel.**

### Detailed #870/#871/#872/#873 sub-entries → archived
The four detailed per-PR narratives (bf16 SiLU #871, GQA-not-a-lever #870, cheap-Add regression #872, QKV-fusion-flat #873) are in `.squad/decisions-archive/2026-08.md` ("Archived by Scribe 2026-08-13T14:45Z"). The milestone conclusion + latency-bound correction above are the live record; #873's `CudaQkvProjectionFusion` stays opt-in via `ONNX_GENAI_CUDA_ENABLE_QKV_FUSION=1`.

**Profiling note (all four investigations):** hardware profilers remain blocked in-sandbox (ncu
absent; nsys "Creating threads in this process is forbidden by design"; RmProfilingAdminOnly=1).
All numbers from the built-in op timer + `ONNX_GENAI_PROFILE_OPS`/`cuGraphGetNodes` node counts +
capture-safe env-gated A/B on a single release binary.

## Lower-bit quant — MEASURED no-go; ceiling is latency-bound not bandwidth-bound (2026-08-13, PR #885)

**KEY CONCLUSION (record prominently):** Research asked whether lower-bit quant (int3/int2/mixed/
2:4/NF4) could beat ~47 tok/s on Muse-Glimmer-30B native CUDA decode. **NET RESULT: a MEASURED
🟥 NO-GO** — and the probe that settled it also **corrected the earlier bandwidth-bound
mis-attribution.** Decode is **latency-bound on the serial ~2568-node dependency chain
(~8.2 µs/node × 2568 ≈ 21 ms/token)**, NOT weight-bandwidth-bound. Reconciles both prior
negatives: not bandwidth (byte-fold flat) AND not marginal-node-sensitive (#872/#873 flat/worse).
**Megakernel note (SUPERSEDED — see "Dense-decode megakernel arc" below):** the correction
originally reopened a decode megakernel / per-layer node-collapse as "the true lever." That
GEMV megakernel has since been built and measured a 🟥 NO-GO (#898, ~3% slower); the surviving
lever is graph-side glue node-collapse (Batty, `optimizer.rs`).

### 2026-08-13: Bandwidth probe (#885 MERGED, docs-only) — the mechanism is latency, not bytes
**By:** Sebastian (Perf). Full brief: `docs/research/lowbit-quant-feasibility.md`.
**Byte budget** (measured from the real ONNX, 417 MatMulNBits, bits=4/bs=32/asymmetric/bf16-scales):
packed weights 13 254 MB + bf16 scales 1 657 MB + int4 zero-points 414 MB = **15 325 MB/token**
= at 47 tok/s only **724 GB/s = ~15% of the H200's 4.8 TB/s HBM roofline** (if bandwidth-bound
we'd be ~313 tok/s). The bf16 scale floor does NOT shrink with fewer bits, so int2-everywhere is
only 0.554× bytes, not 0.5×.
**Controlled probe** (throwaway/reverted, `ONNX_GENAI_WEIGHT_FOLD=D` folds the int4 GEMV weight-read
column so DRAM footprint → 1/D with loop-trip/instruction/launch/node-count byte-identical; H200,
CUDA_GRAPH=1, --pipeline, 3×128-tok median):
| weight DRAM | tok/s | Δ |
|---|---|---|
| full (D=1) | **47.29** | — |
| half (D=2) | 47.98 | +1.5% |
| quarter (D=4) | 48.62 | +2.8% |

−75% weight DRAM → **+2.8%** ⇒ weight-DRAM-bound fraction ≈ **3–4%**. int2-everywhere (−45% bytes)
projects to **≈+1.6% (~48 tok/s)**, not the naive +14%/+80%. **Lower-bit quant (all variants) =
MEASURED 🟥 NO-GO** as the next lever.

### 2026-08-13: Accuracy reality-check (Fact Checker) — every sub-4-bit path also needs a re-quant
**By:** Fact Checker (independent, accuracy lens only; read-only, no kernels touched). Full brief:
merged from `fact-checker-lowbit-accuracy.md`.
- **int3 weight-only (imatrix/AWQ-class, ~3.5 bpw / SpQR mixed):** 🟢 credible — small real quality
  tax, least-risky sub-4-bit lever.
- **int2 scalar / Q2_K:** 🔴 accuracy-prohibitive (cliff) — do not ship for quality-sensitive output.
- **int2 via codebook/trellis (QuIP#/AQLM/QTIP):** 🟡 SOTA-for-2-bit but still a visible FP16 gap at
  30B, AND replaces scalar dequant with LUT/trellis decode that **spends the bandwidth win back** —
  accuracy win and bandwidth win are coupled, not independent.
- **Mixed-precision (SpQR/LLM-MQ, ~2.5–3.5 avg bit):** 🟢 accuracy / 🟡 kernel+tooling (irregular
  layout + outlier FP16 sidecar).
- **2:4 structured sparsity:** 🟡 needs a fine-tune for quality; no M=1 tensor-core benefit anyway.
- **Load-bearing blockers:** (1) we only HAVE int4 — EVERY sub-4-bit method must re-quantize/calibrate
  from the **fp16/bf16 SOURCE** checkpoint (re-squeezing the existing int4 compounds error → collapse);
  (2) ORT-stack tooling for sub-4-bit >7B is immature (GGUF/imatrix ships it but off-stack; Olive not
  demonstrated). Chew is the numerics gate if any sub-4-bit path is ever funded.

**Disposition:** no code/quant change made or planned. Lower-bit quant was replaced on the roadmap
by a one-layer decode-megakernel prototype — **but that prototype has since been built and measured
a 🟥 NO-GO** (persistent multi-CTA GEMV megakernel ~3% slower, #898; see "Dense-decode megakernel
arc" below). The surviving datacenter lever is **graph-side glue node-collapse (Batty,
`optimizer.rs`)**, not a GEMV megakernel. Decision drops merged & deleted:
`sebastian-lowbit-feasibility.md`, `fact-checker-lowbit-accuracy.md`.

### 2026-08-13: Lower-bit quant is DEVICE-DEPENDENT — H200 NO-GO, but a real lever on consumer/edge
**By:** Sebastian (Perf). Extends `docs/research/lowbit-quant-feasibility.md` §6 "Machine-class
sensitivity". Branch `squad/lowbit-machine-tiers`. Docs-only, no code change.
The measured 🟥 NO-GO above is **H200-specific**: the byte-fold probe ran on this ~4.8 TB/s box,
where weight reads are hidden behind the serial ~2568-node launch-latency chain (~21 ms/token), so
cutting bytes buys ~+3% max. Two-component model: `T_latency` (~21 ms, ~bandwidth-independent) vs
`T_weightread = 15.3 GB / B_device`; per-token ≈ `max(...)` overlapped. **Crossover ≈
15.3 GB / 21 ms ≈ 0.73 TB/s** (extrapolation from one device — a model, not a measurement). Below
that (mid-consumer RTX 4060/4070 ~270–500 GB/s and edge/Jetson) lower-bit is 🟢 a real **speed**
lever; near it (RTX 4090/5090 ~1–1.8 TB/s) 🟡 modest. Independently, lower-bit is the only way to
**fit** 30B on ≤12 GB VRAM (int4 ~15 GB won't load; int3 ~11.5 GB / int2 ~7.7 GB makes it *run*) —
a portability win separate from the speed roofline. **Recommendation:** H200/datacenter 🟥 NO-GO
for speed (lever is the node-collapse path); **keep lower-bit ON THE ROADMAP for consumer/edge**,
gated on running the SAME `ONNX_GENAI_WEIGHT_FOLD` byte-fold probe on a representative consumer GPU
(we only have an H200 and cannot measure that regime here). Accuracy path is device-independent
(Fact Checker: int3/~3.5 bpw imatrix/SpQR 🟢; int2 needs codebook/trellis 🟡; scalar int2 🔴; all
require re-quant from the fp16 source). Ties to Roy's RULES.md §11 portability rule (below).

## Dense-decode megakernel arc — MEASURED 🟥 NO-GO on the whole-layer GEMV megakernel (2026-08-13, PR #898)

**KEY CONCLUSION (record prominently):** After the #885 correction reopened a decode megakernel as
the candidate datacenter lever, Sebastian **built and measured** the persistent multi-CTA
cooperative GEMV megakernel end-to-end. **FINAL VERDICT: NO-GO.** The megakernel is **~3% SLOWER**
per layer than an identical-math per-op baseline. Projected whole-model gain from a GEMV megakernel
≈ **0% (decode stays ~47 tok/s)**. **The only remaining recoverable-overhead lever is graph-side
glue node-collapse (Batty, `optimizer.rs`)** — no cooperative kernel, no grid.sync tax, no numerics
reorder. Merged as PR #898 (main @ 0790849c). Full brief:
`docs/research/dense-decode-megakernel-feasibility.md` §7 (§5/§6 marked superseded).

**Staged arc (all H200, throwaway `#[ignore]` GPU probes, never pipeline-wired):**
- **Phase A/B feasibility (GO-to-prototype, now SUPERSEDED by the P2 measurement):** headroom gate
  passed (captured 21.4 ms/token; recoverable overhead ~85% of the token); a glue-only micro-bench
  fused 22 glue ops into 1 launch recovering 85.6% of the *glue* chain. This projected large upside
  **but did NOT build the int4 GEMV path** — that per-layer number was the real P2 gate.
- **P1.5 (architecture pinned):** single-CTA fused int4 MLP = **926× SLOWER** (one SM ≈ 1/132 of
  device weight-read bandwidth) → residency-only fusion is dead; the megakernel MUST be multi-CTA.
  Confirmed **`grid.sync` / cooperative launch IS capturable** under CUDA-graph capture on this
  H200/driver (keep a runtime capability check + graph-break fallback for older drivers).
- **P2 (multi-CTA cooperative, the deciding measurement — NO-GO):** built the pinned persistent
  multi-CTA megakernel for the MLP triple-GEMV block (1056 co-resident CTAs = 8/SM × 132 SMs,
  grid.sync seams, L2-resident global scratch) with production int4 GEMV math, measured vs
  identical-math per-op baseline. **Per-op baseline 0.656 ms/layer-MLP → megakernel 0.676–0.680 ms
  = recovered fraction −3.2% (−2.9%…−3.5%), ~3% SLOWER, byte-exact 0-ulp.** grid.sync =
  2.23 µs/barrier (full 1056-CTA grid); a full layer would pay ~0.7–0.9 ms/token of barrier tax
  across 52 layers.

**Why the megakernel loses (mechanism, kernel-speed-independent):** (1) CUDA-graph replay already
removes the per-launch overhead the megakernel targets (eager 27.6 → captured 21.4 ms, ~6.1 ms
already banked); (2) the multi-CTA design must PAY a grid.sync tax the per-op path never pays, which
roughly cancels/exceeds the savings; (3) GEMVs are genuine full-device weight-read work already
fanned across all 132 SMs per-op — a megakernel does the *same* reads and cannot accelerate them,
and removed activation round-trips are already L2-resident (~80 KB, ~nothing saved).

**Ownership / redirect:** graph-side glue node-collapse (the live lever) = **Batty**
(`optimizer.rs`); fused kernel epilogues that enable node deletion (#867 SwiGLU-mul, #854
skip-RMSNorm) = **Sebastian** (already landed); numerics gate for any future fused reduction reorder
= **Chew**. One un-excluded future path (scope only if node-collapse + GQA tuning are exhausted): a
software-pipelined design overlapping next-layer int4 weight prefetch with current compute
(Hazy-style) attacks the GEMV time itself, not launch overhead. Decision drops merged & deleted:
`sebastian-megakernel-feasibility.md`, `sebastian-megakernel-p15.md`,
`sebastian-megakernel-multicta.md`.

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
Rules 2/4/5; cites `docs/portability/2026-07-25-cuda-consumer-gpu-audit.md`, `docs/CROSS_PLATFORM.md`,
`docs/benchmarks/2026-07-25-gqa-decode-avx512.md`, `docs/research/lowbit-quant-feasibility.md`.

## Decode correctness does NOT depend on eviction order (2026-08-13, #888)

**By:** Copilot. Branch `squad/eviction-order-correctness`. Investigation only — no shipped behaviour
changed, all knobs added are default-OFF and byte-identical on the default path.
#886 rejected a byte-aware residency policy that corrupts decode (16→3 tokens) when weight offload
engages, speculating a latent *order-dependent* defect in the shipped offload path (which would block
#864's hybrid). **Refuted:** a default-OFF probe (`ONNX_GENAI_WEIGHT_OFFLOAD_EVICT_ORDER`) changing
only the eviction *victim* on the size-blind path shows two independent still-correct orders — MRU
reverse-recency AND byte-aware's exact smallest-first victim under 10,192 evictions — are
**byte-identical** with clean ledgers. **Changing eviction order alone is value-neutral.** The
corruption comes solely from byte-aware's *other* change: the **retain-vs-bypass flip** (promoting a
large transiently-streamed tensor into a retained stable-slot resident served as a hit across steps).
Mechanism is NOT captured-VA baking (graph-OFF corrupts too) and NOT a copy/compute fence hazard
(full drain before every page-in fill doesn't fix it); one secondary consistency bug found
(`stable_slot=true` key never rejoins `pages`; `ONNX_GENAI_WEIGHT_OFFLOAD_RETAIN_SLOTTED=1` closes it
but does NOT stop corruption). **Consequence:** the shipped size-blind path is **safe** (never
retains large tensors → buggy path unreachable), so **#864's hybrid is NOT blocked by an
eviction-order invariant** — a hybrid pinning a **static** hot set (retain once, never
evict/re-admit) does not exercise the corrupting retain-then-churn path. Any dynamic scheme moving
large weight residents in/out (byte-aware, possibly #866/#750 if they churn large pages) must
validate token identity and prefer a pinned non-churning hot set. Drop merged & deleted:
`copilot-eviction-order-correctness.md`.

## nxrt EP plugins on PyPI + CUDA 13 target (2026-08-12)

### 2026-08-12: EP plugin cdylibs published to PyPI as `nxrt-ep-cpu` / `nxrt-ep-cuda`
**By:** Squad (Coordinator), req. by Justin (@justinchuby)
**What:** The two ORT plugin-EP cdylibs are packaged and published to PyPI via
`.github/workflows/publish-ep-plugins.yml` (PR #819) with `python/nxrt-ep-cpu/*` and
`python/nxrt-ep-cuda/*`. Packaging uses **setuptools + plain cargo, NOT maturin** — the
plugins are cdylibs exporting the ORT plugin-EP C ABI, not PyO3 modules. EP cdylibs must
**NOT** link `libonnxruntime`. `nxrt-ep-cpu` 0.1.0.dev5 is LIVE (manylinux_2_28 +
macosx_arm64 + win_amd64). CUDA wheel build (PR #824) switched the cuda job from
`nvidia/cuda:13.0.0-devel-ubi9` to standard `quay.io/pypa/manylinux_2_28_x86_64`.
**Why:** Ship the EP plugins as installable wheels consistent with the extension-contract
directive (#524: stable C ABI + dynamic loading).

### 2026-08-12: `nxrt-ep-cuda` needs no CUDA toolkit/GPU to build; NVIDIA runtime wheels are required deps
**By:** Squad (Coordinator)
**What:** `onnx-runtime-ep-cuda` uses cudarc `dynamic-loading`, so `cargo build --features
cuda` needs **NO CUDA toolkit and NO GPU** — CUDA libs are `dlopen`'d at runtime
(`readelf -d` confirmed the `.so` links zero CUDA libs). The four NVIDIA runtime wheels are
**REQUIRED** deps pinned `>=13,<14` (unsuffixed names are the real CUDA 13 wheels;
`-cu13`-suffixed are 0.0.1 stubs).
**Why:** Removes the toolchain/GPU requirement from the CUDA wheel CI job and pins the EP
wheel to CUDA 13 at runtime.

### 2026-07-30: nxrt-ep-cuda wheel targets CUDA 13
**By:** Squad (Coordinator), req. by Justin (@justinchuby)
**What:** The `nxrt-ep-cuda` PyPI package must build against / target CUDA 13. Runtime NVIDIA
deps use the CUDA 13 wheels (nvidia-cuda-runtime>=13, nvidia-cublas>=13,
nvidia-cuda-nvrtc>=13, nvidia-cuda-cupti>=13), matching the existing `nxrt[cuda]` extra in
crates/onnx-runtime-python/pyproject.toml.
**Why:** User directive "记得用cuda 13"; keeps EP wheel consistent with the main nxrt CUDA
wheel toolchain.

## VMM / offload / streaming / batching push — durable results (2026-08-12)

**By:** Copilot (coordinator). Every claim is backed by a merged, executable test;
refutations are recorded alongside confirmations.

**Governing rule (#772/#776/#787):** `cuMemMap` maps whole granule-aligned windows onto
whole physical granules, so `committed bytes = granule × (windows containing ≥1 live byte)`.
**Layout controls residency** — the allocator cannot compact what layout scattered.
`CU_MEM_ALLOC_GRANULARITY_MINIMUM == RECOMMENDED == 2 MiB` here, so the floor is fixable
only by layout, not by shrinking the granule. Minimum mapping granularity spans ~500× across
platforms (Level Zero/Vulkan ~64 KiB, CPU mmap 4 KiB) → layout must be a queried per-EP,
per-platform capability (#783), not a constant.

**Confirmed:**
- Floor is layout-determined: 768 granules (~1.5 GiB) head-major → 96 (~192 MiB) seq-major →
  1/seq (~2 MiB) token-major = **768× reduction** (#787).
- Strided reads are not the obstacle: seq/head bandwidth ratio 0.80–1.02; 192 KB token-major
  stride measured 1.000 at 6 GiB working set — reads are DRAM-bound independent of stride
  (device memory already 2 MiB-page backed) (#778/#787).
- Offload and capture no longer mutually exclusive (#796): weights page under a stable VA;
  page-in remaps physical granules instead of returning a new pointer. Unblocked #755.
- Managed no-spill VMM is default, auto weight-streaming when a model exceeds budget (#798);
  a fitting model does not page (`FullResident`, offload off, 0 page-ins).
- Prefix sharing is sound (#793/#803): one handle maps into N=8 sequences under captured
  replay; ledger charges once, alive until last sharer, additional sharer costs 0 bytes.

**Refuted (and why it mattered):**
- "seq-major landed ⇒ 8× floor realised" — false: #794 measured head-major and seq-major
  committing identical bytes (bindings didn't consume the layout descriptor); fixed #797.
- "decoder structurally declines capture" — false (#804): `captures=0` came from a cached
  `ONNX_GENAI_CUDA_GRAPH=0` in a long-lived test process. #794/#801 misattributed it.
- "fixed KV stride removes growth-triggered re-capture" — true in mechanism, irrelevant:
  engine invalidates the graph unconditionally on growth (#805).
- "tokens per granule" KV cost model — wrong for head-major (retracted), exactly right for
  token-major. Layout is the whole story.

**#736 audit recurring finding (six slices):** 4/5 completed slices found **over-reservation**
(bytes charged on a path that never uses them), not ungoverned allocation — #751 IndexShare,
#795 GQA WS_SCORES (~128 MiB f32-only), #799 cuBLASLt GEMM (32 MiB heuristic ceiling, measured
0–96 B), #802 default-domain Attention scores (genuinely needed), #806 GQA QKV staging. Guidance
in `MEMORY_ARCHITECTURE.md`: **start from use, not from allocation** — governing a bypass without
sizing it to use converts invisible waste into charged waste (tightens #745 admission, reduces
concurrency).

**Method notes:** order-dependent test state cost two wrong conclusions this week
(process-frozen `RuntimeConfig` #804; CUDA context warmed by alphabetically-earlier sibling
#797) — #807 added a debug-only freeze guard, single-stream helper, and an inventory. Negative
results delivered as first-class outcomes. Never extrapolate an unmeasured number (`qwen14b-zp`
lacks `inference_metadata.yaml`, not native-loadable #384 — reported as not measured).

## Durable lessons — #762 absent-slot machinery (2026-08-12)

- **The absent-slot machinery has now produced four distinct defects:** compacted output slots, absent inputs aliased to input 0, a forgeable name-based sentinel, and a 2× heap buffer overflow. Any change touching optional-slot handling deserves disproportionate scrutiny.
- **Allocate and interpret with the same dtype.** Sizing a buffer from one dtype while handing the consumer a different one is a memory-safety bug. Derive both from one source and fail closed when it is unknown.
- **A canary test must mirror production allocation exactly.** Canaries allocating at `byte_size` while production used `max(byte_size, 8)` could not detect wrong-dtype writes — the padding absorbed them. A test that passes for a reason unrelated to its claim is the most-repeated defect on this PR.
- **Verify a fail-loud gate by actually creating the failure condition.** Renaming one `ort-prebuilt` directory was a false negative; only renaming all 16 proved the gate fires.
- **Third false "API does not exist" deferral.** `MemoryDevice_GetDeviceId` and `Session_GetEpGraphAssignmentInfo` (twice) were all claimed unavailable and all existed — the latter already in use in our own tree. Check the generated bindings before deferring.
- **Merging upstream `main` into a long-lived branch:** resolve append-only archives and `.gitignore` as **unions**, never by taking one side, or user work is silently lost.


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


## CUDA-graph capture arc — Muse-Glimmer-30B native decode 11.4 → 23.13 tok/s (2026-08-12)

Full 5-blocker narrative (CLASSIFY #848 → LOAD #850 → PIN #852 → bf16 GQA kernel #855 →
SKIP-NORM #854; capture 54 seg/53 seams → 1 seg/0 seams; 11.4 → 23.13 tok/s) archived to
`.squad/decisions-archive/2026-08.md` (under "Archived by Scribe 2026-08-13T05:15Z"). The arc
then continued 23 → 40.21 (#860) → 47.25 (#867) → ceiling (#870/#872/#873, above).

**Durable lessons (retained):** (a) a metadata-declared feature (sliding_window) must be
validated against **graph-truth**, not trusted blind — a vestigial window silently forced the
non-capturable path. (b) The capture classifier's growing-symbol veto is a **false positive for
fixed-capacity device-KV**; pin the seq symbol engine-side, keeping the kernel
`capture_support()` gate as an independent backstop. (c) A capture-safety flag sampled right
after a warm-time arena grow reads false at the worst moment — gate the demotion on
`is_capturing()`. (d) bf16 kernels accumulate in fp32; bf16 only at load/store boundaries,
oracle-gated against f64 softmax.

## Archived narrative waves pointer

Full wave narratives are archived in `.squad/decisions-archive/2026-08.md` and `.squad/decisions-archive/2026-07.md`.

- **2026-08-12T00-00-00Z (Scribe CUDA-capture escalation batch):** the EP-wheels / bf16 / H200 merge wave (2026-08-12T20:40:00Z), its status snapshot table (#31985 merged, #31973/#31974/#31988/#31993/#32001/#32003 drafts), and the three verbatim inbox drops (sebastian nxrt-ep-pypi packaging, sebastian nxrt-ep-cuda wheelfix, leon #762 test-followups) were moved to `.squad/decisions-archive/2026-08.md` to keep the live ledger lean.
- Earlier 2026-08-10/11/12 waves (EP plugin export, PR #762 parity, upstream CI correction, Apple MLAS f16 cast, CUDA MatMulNBits, rejection-response, PR #31973/#31974 threshold+regression fixes) are also in `.squad/decisions-archive/2026-08.md`.
