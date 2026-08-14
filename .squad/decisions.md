# Decisions — live standing directives

Last consolidated: 2026-08-14T04:09:13Z (Scribe decode-floor+textproto batch, local state; merged 5 inbox drops — sebastian-norm-gemv-prologue (#916), cohaagen-deepseek-golden (#914), leon-textproto-fixtures (#921), copilot-mru-eviction-sweep, copilot-zero-copy-hybrid (#864). KEY: decode latency floor now FOUR-way confirmed — norm→GEMV-prologue fusion measured −4.6% AND diverges ≈token 38 under replay (#916), the 4th independent NO-GO after megakernel (#898), glue-under-replay ceiling (#899/#900), standalone skip-fold −1.5% (#903); arc CLOSED, native ~47.6–48 tok/s beats ORT ~40. Added standing facts: DeepSeek-V2 native path = standard RotaryEmbedding+Attention+QMoE (NOT MLA), golden lock #914; textproto fixture convention (#921, 29 converted, keep-binary = external-data or ORT-loaded). Merged external copilot drops: LRU-default-after-MRU-sweep + zero-copy-hybrid #864 negative. HARD-GATE size: decisions.md was 49,833 B (at 50 KB gate) → archived FOUR closed/older narrative arcs (Fusion-arc CEILING #870-873, Lower-bit-quant #885, Dense-megakernel #898, CUDA-graph-capture 08-12) to decisions-archive/2026-08.md with one-line pointers → 38,333 B before merge, comfortably under 45 KB. NOTE: spawn prompt asked to archive by AGE (before 2026-07-14) and push to main; per Scribe charter I archived by SIZE and committed on a chore/scribe branch instead. Histories: appended merged-PR lines to sebastian/leon/cohaagen; checked chronicle + 15,360 B gates.)

Last consolidated: 2026-08-13T18:57:00Z (Scribe decode-latency-floor batch, local state; merged 3 inbox drops — sebastian-glue-replay-gate (#899) + batty-glue-node-collapse (#900) + sebastian-bf16-skip-rmsnorm (#903) — into ONE "Decode latency-floor: node-collapse arc" section, deduped against the megakernel/glue/lowbit sections. KEY MILESTONE: native CUDA int4 batch-1 decode is confirmed at its launch-amortized LATENCY FLOOR from THREE independent directions — megakernel NO-GO (#898), glue-collapse +0.9% ceiling (#899/#900), skip-RMSNorm fold −1.5% regression (#903); consistent mechanism = at M=1 folding parallel work into a single-CTA reduction serializes what per-op spread across 132 SMs. Native ~47.6–47.8 tok/s (beats ORT ~40). SHIPPED: #900 bf16 SiLU/SwiGLU-mul glue collapse (+0.9%, byte-exact, Rule 11 portability fix) + #903 bf16 skip-RMSNorm KERNEL (0-ulp byte-exact); NO-SHIP as default: #903 standalone fold (−1.5%, opt-in behind ONNX_GENAI_CUDA_ENABLE_SKIP_RMSNORM_FUSION default OFF). Only speculative remaining lever = norm-into-GEMV-prologue fusion (NOT funded). Size gate: adding the section would exceed 50 KB → archived the 2026-08-12 VMM/offload/streaming + #762 absent-slot durable-lessons narratives to decisions-archive/2026-08.md → decisions.md now ~48.4 KB, under gate. Histories: appended #903 to sebastian, #900 to batty; checked chronicle + 15,360 B gates.)


Last consolidated: 2026-08-13T14:45:00Z (Scribe lowbit-nogo-probe batch @ main 26bd410f; merged 2 inbox drops — sebastian-lowbit-feasibility + fact-checker-lowbit-accuracy — into new "Lower-bit quant — MEASURED no-go; ceiling is latency-bound not bandwidth-bound" section [PR #885]. KEY: byte-fold probe (−75% weight DRAM → +2.8%, HBM util ~15%) REFUTES the earlier "weight-bandwidth-bound" attribution; decode is LATENCY-bound on the ~2568-node serial chain (~8.2 µs/node). Appended a Correction note to the #870/#872/#873 fusion-arc entry (ceiling VALUE + "marginal fusion not a lever" stand; mechanism + lower-bit future-lever were wrong). Also corrected the mechanism wording in docs/status/PROGRESS.md (#875 lines) + bumped HEAD → 26bd410f. Megakernel/node-collapse REOPENED as true lever. Size gate: after merge decisions.md hit 53,745 B (>50 KB) → archived the detailed #870/#871/#872/#873 fusion-arc sub-entries to decisions-archive/2026-08.md (milestone conclusion + correction kept live) → 49,381 B, under the 50 KB gate. Histories: appended probe+NO-GO to sebastian/fact-checker history.md; checked chronicle + 15,360 B gates.)

Earlier 'Last consolidated' chronicle lines (2026-08-11/12 six entries, plus 2026-08-13T03:03–17:07 five entries) archived to `.squad/decisions-archive/2026-08.md`.

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

## Decode latency-floor: node-collapse arc — glue collapse ships +0.9%, skip-RMSNorm fold is a no-ship; batch-1 latency FLOOR FOUR-way confirmed, arc CLOSED (2026-08-13/14, PRs #899/#900/#903/#916)

**KEY MILESTONE (record prominently):** Native CUDA int4 **batch-1 decode is confirmed at its
launch-amortized LATENCY FLOOR from FOUR independent directions** — GEMV megakernel NO-GO (#898),
graph-side glue-collapse +0.9% ceiling (#899/#900), skip-RMSNorm fold −1.5% regression (#903), and
now **bf16 norm-into-GEMV-prologue fusion −4.6% regression + numeric divergence (#916, kill-gate
CLOSED)**. The consistent mechanism across all four: **at M=1, folding parallel work into a
single-CTA (or single-warp-serial) reduction serializes what per-op spread across 132 SMs.**
CUDA-graph replay already amortizes per-launch overhead (~0.9 µs/node dispatch floor survives
replay), so any scheme that trades whole-GPU multi-CTA parallelism for a fused single-CTA launch is
strictly heavier. Native decode sits at **~47.6–48 tok/s** (beats the ORT backend ~40). **The
speculative norm→GEMV-prologue lever is now itself a MEASURED NO-GO (#916); the arc is CLOSED** —
do not pursue norm→GEMV-prologue on H200 without first building a genuine multi-CTA
cooperative-reduction prologue kernel. Docs: `docs/research/dense-decode-megakernel-feasibility.md`
§8/§8.5/§8.6/§8.7.

### 2026-08-14: bf16 norm-into-GEMV-prologue fusion — measured NO-GO (kill-gate CLOSED, #916, Sebastian)
Prototyped the bf16 analogue of `CudaSkipRmsNormMatMulFusion` — a throwaway bf16-only optimizer pass
folding `x → SimplifiedLayerNormalization(x,γ) → MatMulNBits` (Gemma3 pre-norm seams) into the
following int4 GEMV's in-kernel RMS prologue, so the RMS reduction rides the GEMV's full-device
(132-SM) occupancy instead of a standalone launch. Intended as the distributed-reduction fix for
#903's −1.5% standalone-skip regression. Kernel needed NO change (bf16 GEMV stages through fp16 in
`run_bf16`, reusing the existing fp16 prologue kernel); 104 foldable pre-norms/model (2/layer × 52).
**NOT byte-exact:** greedy 128-token stream diverges at ≈token 38 (`…2963,38,9520…` →
`…2963,38,8323,2481,9520…`) — the fp16 `warp_half4` prologue reduction tree differs from
`rmsnorm_bf16`'s block-tree by ~1 fp32 ULP and skips the intermediate bf16 rounding.
**Measured (H200, CUDA_GRAPH=1, --pipeline, interleaved A/B): baseline 47.9 → fused ON 45.7 tok/s =
−4.6% REGRESSION** — folding saved only ~1.8 ms standalone norm but added +180 ms to the GEMVs
(368.91→548.81 ms, +48%). **Why:** inverse of the fp16 skip case — bf16 standalone norm
(`rmsnorm_bf16`, block-tree parallel) is already cheap; the available fp16 prologue reduction
(`skip_rmsnorm_f16_warp_half4`) is single-warp-serial over H=6656, re-run once per fan-out follower
GEMV (≤4), on the critical GEMV path. The `fusion_benefit_is_positive` gate was calibrated for the
fp16 assumption (standalone skip = ~24% of decode), FALSE for bf16. **VERDICT: NO-SHIP / NO-GO —
prototype NOT landed (finding-only: docs §8.7 + this drop, no `src/`/`crates/` change, no flag, no
dead code).** This is the **fourth** independent confirmation of the decode latency floor. Drop
merged & deleted: `sebastian-norm-gemv-prologue.md`.

### 2026-08-13: Glue node-collapse is a GO — survives graph replay (#899, Sebastian)
Before staffing a multi-week `optimizer.rs` pass, Sebastian measured whether glue collapse recovers
anything **under graph replay** (the real production decode path), since #898's own mechanism was
"replay already removes per-launch overhead." Per-op glue chain (22 nodes) vs fused (1 node), each
captured into a CUDA graph and timed under replay: eager recovers 84–85%, **under replay 74.0–75.5%
recovered — a ~0.9 µs/node residual dispatch cost SURVIVES replay** (22 trivial graph nodes vs 1),
byte-exact 0-ulp. Whole-model projection was **+5.3% ceiling** (46.7 → ~49.2 tok/s) — a projection,
since glue nodes interleave with the dominant GEMVs. **GO** — the opposite of the megakernel because
glue ops are tiny L2-resident dispatch-bound work (not irreducible GEMV work) and collapse uses an
ordinary fused launch (no grid.sync tax, no reduction reorder, no Chew gate).

### 2026-08-13: Glue collapse REALIZED — bf16 SiLU/SwiGLU-mul, +0.9% byte-exact (#900, Batty)
Converted the #899 ceiling into a measured number on the production Muse-Glimmer-30B decode graph.
**Root cause:** `CudaSiluFusion` (`crates/onnx-runtime-ep-cuda/src/optimizer.rs`) was gated to
**Float16 only** and never fired on the bf16 stream — extended to accept **BFloat16** (a portability
fix under Rule 11). The standalone `Sigmoid`+`Mul`+`Mul` glue then collapses through
`CudaSiluFusion`→`CudaSwiGluFusion` into the tagged `Mul[_cuda_silu_mul]`, lowered to the landed
`decomposed_silu_mul_bf16` epilogue (#867). **Measured (H200, CUDA_GRAPH=1, interleaved A/B):
47.20 → 47.63 tok/s = +0.9%**, byte-exact (24-token stream bit-identical), node count 22→20
glue/layer (−104 total: `Sigmoid` 104→52, `Mul` 210→158). `CudaGateUpSwiGluFusion` needs an fp16
activation so stays dormant → **int4 GEMVs untouched**. **SHIP** (small-but-real). Honest bound vs
the +5.3% ceiling: only the 2 SiLU/SwiGLU-mul nodes/layer are byte-exactly collapsible; bigger levers
blocked — (1) **6 norms/layer** need a byte-exact bf16 skip-RMSNorm kernel (see #903); (2) **~208
`gamma+1` Adds/layer** already MEASURED −2.8% (#872) — do not re-attempt; (3) **4 reshapes/layer** are
GQA head-split metadata coupled to the attention kernel, not free deletions.

### 2026-08-13: bf16 skip-RMSNorm — KERNEL ships byte-exact, standalone FOLD is a no-ship (#903, Sebastian)
Closed the #900 blocker by building the missing **byte-exact bf16 skip-RMSNorm kernel** for
Gemma3 sandwich-norm (6 norms/layer × 52 = 312 nodes, seam `Add(residual, sublayer)→
SimplifiedLayerNormalization`). The `skip_rmsnorm_bf16` NVRTC kernel
(`crates/onnx-runtime-ep-cuda/src/kernels/normalization.rs`) computes
`sum = __float2bfloat16_rn(f32(residual)+f32(x))` (bit-for-bit a standalone bf16 `Add`), then the
identical `rmsnorm_bf16` block-tree reduction. **Numeric fidelity (Chew gate): BYTE-EXACT, 0-ulp** —
GPU unit tests bit-identical vs standalone `Add`→`rmsnorm_bf16` at H=6656; real-model greedy stream
fold OFF vs ON bit-identical (128/128 tokens). **KERNEL: SHIP.**
**But the standalone fold REGRESSES:** perf A/B (H200, CUDA_GRAPH=1, interleaved) — fold OFF (default)
**47.77 tok/s**, fold ON **47.06 tok/s = −1.5%** (104 seams folded). **Why:** at M=1 the RMS reduction
is single-CTA (all H=6656 in one block); folding the residual add into it **serializes** work the
standalone `Add` spread across all 132 SMs. Under replay the launch saving is already banked, so the
fused single-CTA skip kernel is strictly heavier than `multi-CTA Add + single-CTA norm` — same
structural reason as the megakernel NO-GO. **FOLD: NO-SHIP as default** — retained opt-in behind
**`ONNX_GENAI_CUDA_ENABLE_SKIP_RMSNORM_FUSION` (default OFF)**; default binary unchanged (47.77 tok/s
== baseline). The kernel is the prerequisite for the only bf16 path that could win: folding the norm
into the neighbouring **multi-CTA int4 GEMV prologue/epilogue** (keeps the reduction distributed) —
a larger GEMV-kernel job, NOT this fold, and NOT yet funded. **Do NOT self-merge** — Chew gates
numerics. Drops merged & deleted: `sebastian-glue-replay-gate.md`, `batty-glue-node-collapse.md`,
`sebastian-bf16-skip-rmsnorm.md`.

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

## Zero-copy hybrid weight residency is a MEASURED negative on RTX 4060 / WDDM (#864, Copilot)

**By:** Copilot (branch `squad/zero-copy-hybrid`). Built a default-OFF `ONNX_GENAI_ZERO_COPY_HYBRID`
CUDA-EP mode: keep the size-blind `StableResident` hot set in VRAM and bind the cold remainder in
place from a `cuMemHostRegister(READ_ONLY | DEVICEMAP)` host mapping instead of streaming it each
decode step; the bypass decision is intercepted **before** any eviction so the hot set never
evicts/re-admits a large stable slot (avoids the #886 corruption pattern). **Finding (negative, the
point of the work):** a *single* zero-copy host-mapped read is bit-identical (verified at 1/8/16/32
cold weights/step), but **aggregate host-mapped read traffic above ~0.44–0.65 GB/step silently
corrupts decode** (32 cold ≈0.44 GB = correct; 48 ≈0.65 GB = generation collapsed 16→3 tokens) — same
signature as #886 but the mechanism is **stale host-mapped reads past a system-memory-aperture
ceiling**, not eviction/re-admission (an A/B that defers exactly as zero-copy would but performs the
real copy was byte-identical). `cuMemHostRegister` of the full 16.65 GB mapping **only succeeds with
READ_ONLY** (DEVICEMAP-only fails OOM), so READ_ONLY cannot be dropped; CPU pre-faulting did not fix
it (not a demand-paging race); pointers are 256-byte aligned (not alignment). **Decision:** the
hybrid does **not** beat WDDM on this hardware (WDDM keeps ~7.7 GB resident, moves ~0.6 GB/step via
the driver's own paging; our managed budget caps ~6.1 GB and zero-copy can only *safely* cover
~0.44 GB/step — both levers worse than the OS here). **Ship default-OFF with a conservative 256 MiB
zero-copy budget** so the opt-in knob is always byte-identical (never exercises the corruption
ceiling); retained as instrumented, reviewable infrastructure for other hardware (datacenter GPUs
with resizable BAR / larger host apertures may not hit the ceiling), **not** a Windows win. **Do NOT
build a churning dynamic hot set** — unnecessary and unsafe (#886). Safety gates verified (token IDs
byte-identical, `captures>0`/`fallbacks==0`, `oversubscribed_bytes==0`, all underflow/unaccounted
counters 0, `mobius_seqmajor_growth_parity_native_cuda` passed solo). Drop merged & deleted:
`copilot-zero-copy-hybrid.md`.

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
