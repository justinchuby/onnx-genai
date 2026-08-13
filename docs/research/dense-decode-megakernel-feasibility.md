# Dense-decode megakernel feasibility (Muse-Glimmer-30B, native CUDA, M=1)

**Author:** Sebastian (Performance/Systems). **Date:** 2026-08-13. **Branch:** `squad/dense-decode-megakernel`.
**Status:** Phase A gate **PASSED** → Phase B glue micro-benchmark → P1.5 fused-int4-GEMV + grid.sync-capture probes → **P2-prototype multi-CTA cooperative megakernel MEASURED → NO-GO for the GEMV megakernel** (see §7) → **§8 glue node-collapse validated UNDER graph replay → GO** (the real lever, Batty's `optimizer.rs`, +5.3% ceiling, byte-exact).

> **Headline.** Native decode of Muse-Glimmer-30B is confirmed **latency-bound on a
> ~2568-node serial launch chain** (~21.4 ms/token, 46.7 tok/s on H200 at
> `ONNX_GENAI_CUDA_GRAPH=1`), **not** weight-bandwidth-bound (prior byte-fold probe:
> −75% weight bytes → +2.8%). Phase A shows the megakernel-recoverable overhead is
> **large** — pure per-launch floor alone is ~20% of the token, and the essential
> weight+KV DRAM floor is only ~3.2 ms (a **~6.7× theoretical ceiling**, ~313 tok/s).
> Phase B directly measured the fusion mechanism: collapsing the elementwise/norm
> "glue" chain (70% of the node count, ~44% of eager decode) into one resident
> kernel **recovers 85.6% of that chain's GPU time**, byte-exact. **Verdict: GO** —
> a dense one-layer megakernel is a real lever worth prototyping (P2), projected
> **~62 tok/s conservative (+33%) to ~100+ tok/s (2×+)**. This is a *different* op
> chain than the #769 P0 QMoE-MoE megakernel (Amdahl-capped ~13%, NO-SHIP); that
> result does not bind the dense path.
>
> **P1.5 update (2026-08-13).** Two decisive follow-up probes now bound the P2
> *architecture*: (1) a **single-CTA** fused int4 MLP that holds the 19968-wide
> intermediate resident in shared memory is **926× SLOWER** than the per-op int4
> GEMV baseline (byte-exact, 0 ulp) — residency alone is a dead end because one SM
> is ~1/132 of the device's weight-read bandwidth; the megakernel **must be
> multi-CTA**. (2) The feared blocker for a multi-CTA design — whether a
> **cooperative launch (`cuLaunchCooperativeKernel`, the only launch path for
> `grid.sync`) can be captured into the decode CUDA graph** — is empirically
> **CLEARED**: on this H200/driver the cooperative launch records into an active
> capture and instantiates a graph. **Net: P2 stays GO, now scoped concretely as a
> persistent multi-CTA cooperative megakernel** (grid.sync barriers between the
> resident sub-GEMVs), not a single giant CTA. See §6.
>
> **P2-prototype update (2026-08-13) — the decisive negative.** The multi-CTA
> cooperative megakernel P1.5 pinned was actually **built and measured** (MLP
> triple-GEMV block: gate/up → SiLU·Mul → down, grid sized to occupancy = 1056
> co-resident CTAs, `grid.sync` seams, activations in L2-resident global scratch).
> Result vs the identical-math per-op baseline: **−3.2% (megakernel is ~3% SLOWER),
> reproducible, byte-exact (0 ulp).** The launch-collapse win the whole thesis
> rested on is **already captured by CUDA-graph replay** for the GEMV-dominated
> path, and the multi-CTA design must *pay* a `grid.sync` tax (**2.23 µs/barrier**
> on the full 1056-CTA grid). The GEMVs are genuine full-device weight-read work a
> megakernel cannot accelerate — it only reorganizes them and adds barriers.
> **Revised verdict: NO-GO on the whole-layer GEMV megakernel (P2).** The only
> component with recoverable overhead is the tiny elementwise/norm **glue**, and
> that is better attacked **graph-side** (Batty's node-count collapse to shrink the
> captured graph's replay overhead) — cheaper, lower-risk, and no numerics gate.
> See §7.
>
> **§8 update (2026-08-13) — the §7.4 redirect is now MEASURED, not assumed.** The
> obvious objection to §7.4 was self-referential: §7's own mechanism (1) is "graph
> replay already removes launch overhead," and glue node-collapse targets that same
> overhead — so does it also collapse to ~0 under replay? **Measured directly:** the
> ~22-op glue chain captured as 22 graph nodes vs 1 fused node, **timed under graph
> replay**, still recovers **74–75%** (byte-exact), because a **~0.90 µs/node dispatch
> floor SURVIVES replay** (replay cuts eager ~2.3 µs/op to ~0.9 µs/node, not to zero).
> Unlike the megakernel, collapse pays **no grid.sync tax** and reorders no reductions.
> **Verdict: GO on graph-side glue node-collapse** (Batty, `optimizer.rs`), bounded
> upside **~+5% decode (46.7 → ~49 tok/s)** — cheap, low-risk, no Chew gate. See §8.

---

## 1. Baseline & op mix (measured, this H200, `CUDA_VISIBLE_DEVICES=0`)

`profile_native --model <muse-glimmer> --pipeline --ep cuda --backend native --steady --warmups 1 --runs 3 --tokens 128`

| config | ms/token | tok/s | note |
|---|---:|---:|---|
| captured (`ONNX_GENAI_CUDA_GRAPH=1`) | **21.40** | **46.72** | baseline; 1 capture segment / 0 seams |
| eager (`ONNX_GENAI_CUDA_GRAPH=0`) | 27.65 | 36.17 | capture already recovered ~6.1 ms/token of **CPU** launch overhead |

Eager per-op decode mix (`ONNX_GENAI_PROFILE_OPS=1`, one forward, 27.64 ms total):

| op | total ms | % eager | calls | per-call µs |
|---|---:|---:|---:|---:|
| MatMulNBits | 7.80 | 28.2 | 417 | 18.7 |
| GroupQueryAttention | 7.26 | 26.3 | 52 | 139.5 |
| Mul | 6.12 | 22.1 | 210 | 29.1 |
| Add | 2.44 | 8.8 | 311 | 7.9 |
| SimplifiedLayerNormalization | 1.33 | 4.8 | 312 | 4.3 |
| Sigmoid | 1.32 | 4.8 | 104 | 12.7 |
| Reshape | 0.95 | 3.4 | 208 | 4.5 |
| (tail: ReduceSum/Skip/Gather/Cast/…) | <0.2 | <1 | ~11 | — |

Executor plan ≈ **1626 op-nodes/token**, which the capture pass expands to **2568
device-graph nodes/token** (memsets, broadcast expansions, view materializations).
The **elementwise/norm "glue"** — `Mul`+`Add`+`Norm`+`Sigmoid`+`Reshape` = **1145
op-nodes (70% of the count), ~44% of eager time**. Each is a tiny launch that reads
and rewrites the 6656-element bf16 hidden vector (~13 KiB) through global memory —
exactly the inter-op round-trip a megakernel keeps in registers/shared. The 469
"heavy" nodes (417 GEMV + 52 GQA, 54% of eager time) carry the real weight/KV DRAM
+ MAC work.

---

## 2. Phase A — headroom gate (the three megakernel-recoverable costs)

Decomposing the **21.4 ms captured floor** (CPU launch already ≈0 under graph replay,
so this is the GPU-side serial floor). Real per-node cost = 21.4 ms / 2568 = **8.33 µs/node**.

**(a) Per-node launch/schedule floor.** Measured directly (Phase B micro-bench, trivial
1-thread kernel, 2568 sequential launches on one stream): **~1.5–2.0 µs/launch**. So the
pure launch floor is 2568 × ~1.7 µs ≈ **4.4 ms ≈ 20% of the token** — recoverable by
node collapse alone, and this is only the *floor* (excludes the kernel's own work).

**(b) Inter-op activation DRAM round-trips.** The hidden vector bounces to global memory
between ~1145 glue ops (read+write ~26 KiB each ≈ 30 MB/token — trivial *bandwidth*, but
each round-trip is **serialized**: op N+1 cannot start its global load until op N's store
retires). The cost is the *latency* of that read-after-write chain, folded into the 8.33 µs.

**(c) Occupancy fill/drain.** Every M=1 decode kernel launches ~1 block on a 132-SM H200
(<1% of the machine) and finishes before steady state — the global-load latency
(Long-Scoreboard) is never hidden. A resident, pipelined megakernel overlaps the next
layer's weight loads with the current layer's compute.

**Essential floor a megakernel must still pay.** Weight + KV DRAM = 15.325 GB/token; at
H200 HBM3 4.8 TB/s that is **3.19 ms/token = ~313 tok/s** if perfectly bandwidth-bound.
The byte-fold probe (#885) showed only ~3–4% of that (~0.7 ms) is *currently exposed* —
i.e. today the weights are latency-hidden behind the launch chain, not throughput-bound.

**Headroom = 21.4 ms − 3.2 ms essential ≈ 18.2 ms ≈ 85% of the token is recoverable
overhead** (launch + round-trip latency + fill/drain). **This is ≫ the 25% gate.**

> **GATE: PASS.** Recoverable overhead ~85% of the token; theoretical ceiling ~6.7×
> (313 tok/s). Proceed to Phase B.

---

## 3. Phase B — fusion-mechanism micro-benchmark (measured)

A **throwaway** CUDA-event micro-bench (`crates/onnx-runtime-ep-cuda/tests/megakernel_headroom_gpu.rs`,
`#[ignore]`, **not wired into any pipeline** — run with `--ignored --nocapture`) isolates
the recoverable costs on a faithful model of one layer's glue chain: G=22 glue ops per
layer, each reading+rewriting the H=6656 bf16 vector, vs one fused kernel that loads H
once, applies all 22 ops in registers (fp32), stores once. Median of 200 iters, H200:

| measurement | result |
|---|---|
| per-launch GPU floor (trivial kernel × 2568) | **1.5–2.0 µs/launch** |
| realistic glue op (H=6656 bf16 round-trip) | **~2.1 µs/op** |
| **22 unfused glue launches** | **0.044–0.046 ms** |
| **1 fused launch (22 ops in registers)** | **0.006–0.007 ms** |
| **glue-chain time recovered by fusion** | **85.6%** (2 runs: 85.5 / 85.7) |
| numerics: fused (fp32-chained) vs unfused (bf16 per-op) | **max_abs = 0.0** (byte-identical here) |

Two facts fall out: (1) a glue op costs ~2.1 µs ≈ the ~1.7 µs launch floor + ~0.4 µs
memory — glue ops are **launch/latency-bound, not compute-bound**, so collapsing them
recovers essentially all their time (85.6% measured). (2) The fused fp32 chain matched the
per-op bf16 round-trip to 0 ulp on this input (silu with stable inputs) — **but this does
NOT clear the numerics gate for the real megakernel**, whose RMSNorm tree-reduction and
GQA softmax reorder fp32 reductions; those must keep per-thread fp32 accumulation and go
through **Chew** (as in #855/#860).

---

## 4. Whole-model projection (measured-anchored)

- **Conservative (glue-only fusion, GEMV/GQA left as separate launches).** The glue is
  ~30% of the *captured* 21.4 ms (less than the 44% eager share, since capture already
  removed the CPU-launch half) ≈ 6.4 ms. Recover 85.6% → save ~5.5 ms → **15.9 ms/token ≈
  63 tok/s (+35%)**. This is the *floor* of the win and needs only glue fusion.
- **Optimistic (full one-layer persistent megakernel).** Collapse each layer's ~49
  device nodes to ~2–3 launches (a GQA softmax dependency forces one chunk boundary,
  Hazy-style), keep activations resident, and pipeline the next layer's int4 weight loads
  to hide Long-Scoreboard latency. Recovering the launch floor (~4.4 ms) plus most of the
  round-trip/fill-drain latency plausibly reaches **2–3× → ~95–140 tok/s**.
- **Absolute ceiling** (weight+KV bandwidth-bound, perfectly pipelined): **~313 tok/s**.

Projected range: **~63 tok/s conservative → ~100+ tok/s realistic**, ceiling ~313.
All above 47.25; the lever is real.

---

## 5. Go/no-go for P2 (whole-step integration)

> **⚠️ SUPERSEDED by §7 (2026-08-13).** This section's **GO** was based on Phase A/B
> headroom + the *projection* that a megakernel recovers ~85% of the chain. §7 then
> **built and measured** the multi-CTA cooperative megakernel and found **no
> recovery on the GEMV-dominated path (−3.2%)**. The verdict below is retained for
> provenance; **the operative verdict is §7's NO-GO for the GEMV megakernel.**

**GO — but staged, because P2 is a multi-week capture-safe kernel effort, not a quick win.**

- **Why the #769 P0 no-go does not apply:** P0 was a per-op persistent **QMoE** kernel on
  the 35B-A3B **MoE** path (Amdahl-capped ~13% of decode, regressed). The dense
  Muse-Glimmer path is a completely different op chain (417 dense int4 GEMV + 52 GQA +
  1145 glue) that P0 never touched, and Phase A/B show its recoverable overhead is ~85%.
- **P1.5 (done, §6):** the real fused int4 GEMV per-layer number and the
  grid.sync-capture gate — this brief's Phase B measured only the **glue** recovery
  (70%-of-nodes component) and the launch floor; §6 adds the GEMV/residency and
  capture-architecture findings that actually scope the P2 build.
- **P2 risks to budget:** (1) capture-safety — the megakernel must do no alloc/free/sync
  internally (all scratch pre-staged, per the #854/#867 capture rules); (2) numerics —
  fused RMSNorm/softmax reductions reorder fp32 sums → **Chew gate** + f64 oracle
  mandatory; (3) it needs decode-loop/graph-structure changes to emit one fused op instead
  of ~49 nodes/layer → coordinate with **Batty** (graph/optimizer side). Keep byte-exact
  greedy parity target (first-16 ref ids `[24,372,1045,10016,328,2885,262,5091,8811,511,917,4921,768,328,2885,262]`).

---

## 6. P1.5 — real fused int4 GEMV + grid.sync-under-capture (measured, this H200)

Two throwaway `#[ignore]` probes were added to the Phase B harness
(`tests/megakernel_headroom_gpu.rs`; `megakernel_int4_mlp_probe`,
`grid_sync_capture_gate_probe`). Both use faithful block-32 int4 dequant (nibble
unpack, symmetric zp=8, fp32 accumulate + `block_sum`, identical math/order to the
production `matmul_nbits_gemv_f32` reference). Real Muse-Glimmer MLP shapes:
gate/up `6656→19968`, down `19968→6656`, block_size 32.

### 6.1 Single-CTA fused int4 MLP vs per-op baseline

The fused kernel is one block that keeps the hidden input **and** the 19968-wide
SiLU·Mul intermediate resident in 104 KiB of opt-in dynamic shared memory across
the whole MLP — **zero activation DRAM round-trips**, only packed weights stream
from DRAM. Baseline = the current 4 separate launches (gate GEMV, up GEMV, SiLU·Mul,
down GEMV), each with `grid = N` columns so weight reads fan out across all SMs.

| Variant | Launches | GPU time / layer-MLP | vs baseline |
|---|---|---|---|
| Per-op baseline (full-device parallel) | 4 | **0.664 ms** | 1× |
| Fused **single-CTA** (intermediate resident) | 1 | **615.3 ms** | **926× SLOWER** |

Numerics: **max_abs = 0, max_ulp = 0 — byte-exact.** Identical dequant + identical
`block_sum` reduction order means residency fusion does *not* reorder any sum;
byte-exact greedy parity is preserved (no Chew gate needed for pure structural
residency; it *would* be needed for any reduction-reordering GQA-softmax/RMSNorm
tree-collapse in the full kernel).

> **Absolute-number caveat (be honest):** the baseline here is my *reference* f32
> int4 GEMV, which is slower than the production f16-staged split-K dp4a kernel and
> is not the captured path, so **0.664 ms/MLP-layer is NOT the decode budget**
> (whole-token wall is ~21.4 ms / 52 layers ≈ 0.41 ms/layer for *everything*). The
> finding is the **926× ratio and the byte-exactness**, not the absolute time.

**Interpretation.** Residency-only fusion (the trick that won 85.6% on the tiny
6656-wide *glue* vectors in Phase B) is **catastrophic on the weight-heavy GEMVs**:
one SM pulls the ~199 MB of MLP weights alone and serializes all 26 624 output
columns, losing the ~132× full-device weight-read parallelism the per-op kernels
get for free. **Conclusion: a dense-layer megakernel cannot be single-CTA** — it
must be multi-CTA so weight reads stay spread across the whole GPU, which forces a
grid-wide barrier (`grid.sync`) to keep activations resident across the sub-GEMV
dependency boundaries. That makes §6.2 the gating question.

### 6.2 grid.sync / cooperative-launch under CUDA-graph capture — the P2 gate

A persistent multi-CTA megakernel with grid-wide producer/consumer sync can only be
launched with `cuLaunchCooperativeKernel`. The decisive P2 question: **can that
launch be captured into the decode CUDA graph** (decode replays a captured graph)?
Probe: compile a trivial co-resident cooperative kernel to CUBIN (sm_90), (A) launch
it cooperatively outside capture, then (B) begin thread-local stream capture, attempt
the cooperative launch, read capture status, and end capture.

| Step | Result |
|---|---|
| (A) cooperative launch outside capture | **OK** — device supports cooperative launch |
| (B) cooperative launch **during** capture | launch = **Ok**, capture status = **ACTIVE** (not invalidated), `end_capture` → **graph instantiated** |

**Verdict: grid.sync IS capturable on this H200/driver.** The most-feared P2
architecture blocker — "a cooperative megakernel can't live inside the captured
decode graph, forcing an eager seam" — **does not fire here.** Caveats to budget,
not fabricate around: (i) this is driver/CTK-version-dependent — older drivers
historically returned `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED` for cooperative launch
under capture, so P2 must keep a graph-break fallback path and gate on a runtime
capability check; (ii) a real `grid.sync` requires the whole grid **co-resident**, so
the grid must be sized to `occupancy × SM count` (via
`cuOccupancyMaxActiveBlocksPerMultiprocessor`), not to the problem size — a hard
design constraint but not a capture blocker; (iii) capture-safety still forbids any
internal alloc/free and dynamic parallelism inside the kernel (#854/#867 rules).

### 6.3 What P1.5 changes for P2

- **Design is now concrete:** P2 = a **persistent multi-CTA cooperative megakernel**,
  grid sized to occupancy, `grid.sync` barriers between the resident sub-GEMVs, small
  activation vectors + norm/RoPE/SiLU state held in shared/registers, weights + KV
  streamed from DRAM with full-device parallelism. The single-CTA residency shortcut
  is ruled out.
- **The biggest risk retired:** capture-compatibility of the cooperative launch is
  measured-OK, so P2 is not forced onto an eager (uncaptured) seam on this platform.
- **Remaining P2 risks (unchanged from §5 + refined):** occupancy/co-residency grid
  sizing at H=6656 register/smem pressure; hosting GEMVs of different N (19968 vs 6656)
  under one fixed cooperative grid; numerics (any fused RMSNorm/GQA-softmax reduction
  reorder → **Chew** gate + f64 oracle); driver-version portability of captured
  cooperative launch (keep a graph-break fallback).
- **Recovery expectation:** Phase B already banked 85.6% recovery on the glue (70% of
  nodes). P1.5 shows the GEMV portion can't be *sped up* by fusion (it's real
  compute/weight-read work done in parallel), but it *can* be folded into the same
  cooperative kernel to erase its per-node launch/schedule overhead and its activation
  round-trips — which is where the §4 whole-model **~62 → ~100+ tok/s** projection
  comes from. The multi-CTA megakernel is the vehicle that captures both.

**P2 go/no-go: GO (staged prototype).** Capture blocker cleared; architecture pinned
to persistent multi-CTA cooperative. Fund the P2 prototype behind an env flag with a
graph-break fallback and the Chew numerics gate.

> **Superseded by §7:** §6's "GO (staged prototype)" was correct on *architecture*
> (multi-CTA, capturable) but was still a projection on the *payoff*. §7 built that
> exact architecture and measured the payoff — see the NO-GO below.

---

## 7. P2-prototype — multi-CTA cooperative megakernel BUILT & MEASURED (the go/no-go)

P1.5 pinned the only viable architecture: a **persistent multi-CTA cooperative
kernel** (grid sized to occupancy so every sub-GEMV reads its int4 weights across the
full device, `grid.sync` barriers between sub-GEMVs, activations passed through
L2-resident global scratch — not one CTA's shared memory). §7 **builds that kernel**
for the MLP triple-GEMV block (`gate/up → SiLU·Mul → down`, the largest self-contained
GEMV chain in a decoder layer) and measures per-MLP GPU time vs the current per-op
launch sequence on **identical tensors and identical int4 dequant math** (block-32,
fp32 accumulate, same `block_sum` order on both sides — so the recovered *fraction*,
the `grid.sync` cost, and the achieved occupancy are apples-to-apples).

### 7.1 Measured result (H200, median of 200 iters, 3 repeats)

| Variant | Launches | GPU time / layer-MLP | vs baseline |
|---|---|---|---|
| Per-op baseline (grid = N, full-device) | 4 | **0.656 ms** | 1× |
| **Multi-CTA cooperative megakernel** (1 coop launch, `grid.sync` seams) | 1 | **0.676–0.680 ms** | **1.03× — ~3% SLOWER** |

- **Recovered fraction: −3.2%** (reproducible: −3.5%, −3.5%, −2.9%). The megakernel is
  *slightly slower*, not faster.
- **Occupancy:** mega kernel = 8 blocks/SM → **1056 co-resident CTAs** across 132 SMs
  (cooperative launch supported, `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH=1`).
- **`grid.sync` barrier cost = 2.23 µs/barrier** on the full 1056-CTA grid; the MLP
  megakernel pays 2 (a full layer would pay ~6–8 at the QKV/RoPE/GQA/O/norm/SiLU seams
  → ~13–18 µs/layer × 52 ≈ **0.7–0.9 ms/token of pure barrier tax**).
- **Numerics: max_abs = 0, max_ulp = 0 — byte-exact.** Multi-CTA + grid.sync does not
  reorder any reduction (no Chew gate needed for this structure).

### 7.2 Why the megakernel does not win (the mechanism)

The whole megakernel thesis was: collapse ~49 launches/layer → 1, recovering per-launch
overhead + activation DRAM round-trips. Two measured facts kill that for the
GEMV-dominated path:

1. **CUDA-graph replay already removes the launch overhead.** Phase A measured eager
   27.6 ms/token → captured 21.4 ms — capture already recovered ~6.1 ms of CPU-side
   launch cost. Under graph replay the per-node residual cost is small; there is little
   left for a megakernel to recover on the fat GEMV launches (each ~0.16 ms of real
   work here).
2. **The multi-CTA design must pay a `grid.sync` tax** (2.23 µs × barriers) that the
   per-op baseline never pays — and this tax scales with the number of fused seams. It
   roughly cancels (here, slightly exceeds) the launch/round-trip savings.

The GEMVs themselves are genuine **full-device weight-read work** — the per-op kernels
already fan weight reads across all 132 SMs (that's why P1.5's single-CTA was 926×
worse). A megakernel does the *same* reads; it cannot make them faster, it only
reorganizes them and adds barriers. The activation round-trips it removes (gate/up/act
scratch, ~80 KB each) are already L2-resident, so removing them saves ~nothing.

### 7.3 Caveats (stated, not papered over)

- **Representative kernel, not the production f16 dp4a split-K GEMV.** Absolute ms is
  ~10× the production kernel's, so the *fraction* transfers but the absolute per-layer
  budget does not (same caveat as §6.1). A faster production GEMV makes fixed launch
  overhead a *larger* relative share — but it makes the `grid.sync` tax a larger share
  too, and graph replay still removes the launch overhead regardless. The mechanism in
  §7.2 is kernel-speed-independent.
- **Eager-timed baseline.** The real decode replays a captured graph, where the
  baseline's per-launch overhead is *even lower* → the megakernel's (already negative)
  edge only gets worse.
- **MLP subset, not the full layer.** The MLP is the largest GEMV block; attention adds
  QKV/O GEMVs + GQA-decode + more norm/RoPE seams (more barriers), which pushes the
  megakernel further behind, not ahead.

### 7.4 Revised P2 verdict — **NO-GO on the GEMV megakernel; redirect to graph-side collapse**

- **NO-GO:** funding the multi-week whole-layer cooperative GEMV megakernel + its
  capture-safety/numerics gating is **not justified** — the measured per-layer recovery
  is **negative** on the GEMV-dominated path. The architecture is sound and capturable;
  the *payoff isn't there* because graph replay already banked the launch win and the
  GEMVs are irreducible full-device work.
- **The real remaining lever is graph-side, and it's Batty's, not a kernel megakernel.**
  The only decode component with recoverable overhead is the elementwise/norm **glue**
  (Phase B: 85.6% of the *glue* GPU time is fusible). Attack it by **collapsing glue
  nodes in the graph/optimizer** (`optimizer.rs`, Batty) to shrink the captured graph's
  replay overhead — this needs no cooperative kernel, no `grid.sync` tax, and no numerics
  reorder. Sebastian's kernel-side contribution is limited to the already-landed fused
  epilogues (#867 SwiGLU-mul, #854 skip-RMSNorm) that let Batty delete standalone nodes.
- **Projected tok/s from a GEMV megakernel: ~0% (decode stays ~47 tok/s).** The §4
  "~62 → ~100+ tok/s" projection assumed ~85% chain recovery; §7 shows that recovery
  does **not** apply to the GEMVs, so the projection does not hold for a megakernel.
  Realistic upside now lives entirely in glue node-count reduction (bounded, graph-side)
  plus any GQA-decode kernel improvement — both cheaper than a megakernel.

### 7.5 If anyone still wants to chase a megakernel later

The one scenario not excluded here: a design that **overlaps the next layer's int4
weight prefetch with the current layer's compute** (software-pipelined, Hazy-style) to
hide Long-Scoreboard global-load latency — that attacks the *GEMV* time itself, which
§7 shows is the actual floor, rather than launch overhead (already gone). That is a
fundamentally harder kernel than node-collapse and should only be scoped if graph-side
glue collapse + GQA tuning are exhausted and still short of target. It would still face
the `grid.sync`/occupancy/register-pressure constraints from §6.

---

## 8. Glue node-collapse UNDER graph replay — the kill-gate for §7.4's redirect (measured, this H200)

§7.4 redirected the lever to **graph-side glue node-collapse** (Batty, `optimizer.rs`).
But §7's own NO-GO mechanism (1) was *"CUDA-graph replay already removes the per-launch
overhead"* — and glue node-collapse targets that **same** launch overhead. The Phase B
85.6% glue recovery (§3) was measured **eager**, where every launch pays its full cost.
Under the production decode path every op is a **node in a captured graph replayed each
token**, so the launch cost is largely pre-paid. **If replay amortizes per-node dispatch
to ~0, glue collapse dies exactly like the megakernel did.** This section measures that
directly — the gate before staffing a multi-week `optimizer.rs` pass.

**Method.** Build both (a) the per-op glue sequence (G = 22 in-place SiLU round-trips on
the H = 6656 bf16 vector) and (b) the fused single launch (same 22 ops chained in
registers, one launch), **capture EACH into its own CUDA graph**, and time **graph
replay** (median ≥ 200 iters, CUDA events). Plus a launch-floor pair — a graph of 22
`trivial` nodes vs a graph of 1 — to isolate any residual **per-node** replay cost
independent of memory traffic. Test: `glue_collapse_replay_gate_probe`.

### 8.1 Measured result (H200, median of 200 iters, 4 repeats)

| Path | per-op (22 nodes) | fused (1 node) | recovered |
|---|---|---|---|
| **Eager** (reference; reproduces Phase B) | 0.048–0.051 ms | 0.0077–0.0078 ms | **84–85%** |
| **Under CUDA-graph replay** (production path) | **0.0280 ms** | **0.0069–0.0073 ms** | **74.0–75.5%** |

- **Launch-floor under replay:** 22 trivial nodes 0.0219 ms vs 1 node 0.0030 ms →
  **~0.90 µs/node residual dispatch cost that SURVIVES replay.**
- **Numerics:** per-op-graph vs fused-graph `max_abs = 0.000e0` — **byte-exact (0 ulp)**.
- **Whole-model projection (52 layers, anchored to the 21.4 ms/token baseline):** glue
  ~1.46 ms/token → ~0.37 ms/token, **saves ~1.08–1.10 ms/token → 46.7 → ~49.2 tok/s
  (+5.3–5.4%)**. *(Projection — see caveats.)*

### 8.2 Interpretation — the gate PASSES; glue collapse is a real lever, unlike the megakernel

**Graph replay does NOT make per-node dispatch free.** It cuts the eager per-op cost
(~2.3 µs/op) down to a **~0.90 µs/node floor — but that floor persists under replay**.
Collapsing 22 nodes → 1 removes 21 × ~0.90 µs ≈ **19 µs of the 28 µs** per-op replay time.
That is why the recovery stays high (**74–75%**) even under replay, where the megakernel's
recovery went **negative** (−3.2%, §7). The two results are consistent, not contradictory:

| | #898 GEMV megakernel | §8 glue node-collapse |
|---|---|---|
| What it collapses | irreducible full-device **GEMV** weight-read work | tiny **L2-resident** elementwise/norm ops = pure dispatch overhead |
| Recoverable cost under replay | ~none (GEMVs are real work; launch already amortized) | **~0.9 µs/node dispatch floor that survives replay** |
| Extra cost paid | **grid.sync tax** (2.23 µs/barrier × ~6–8/layer) | **none** — ordinary fused launch, no cooperative kernel |
| Numerics | byte-exact but adds barriers | **byte-exact, no reduction reorder** |
| Verdict | **NO-GO** (−3.2%) | **GO** (+5.3% ceiling, cheap, graph-side) |

The §7 concern is therefore **empirically disproven for glue**: replay amortizes per-node
dispatch by ~2.5×, not to zero, and node-collapse recovers the residual with no grid.sync
tax and no numerics gate.

### 8.3 Caveats (stated, not papered over)

- **Modeled glue chain, not the exact production op set.** The probe uses an in-place
  SiLU round-trip on the H-vector as a representative dispatch-bound glue op; the real
  chain (RoPE, RMSNorm, residual add, cast, SiLU-mul over the 19968-wide intermediate)
  differs in width and count. The **~0.9 µs/node dispatch floor** is the transferable
  quantity; absolute ms is representative, not the production kernels' absolute ms.
- **The +5.3% is a CEILING, not a promise.** It assumes the full per-op glue replay time
  is on the critical path and fully removable. In the real graph, glue nodes **interleave
  with the GEMV nodes that remain the dominant serial cost**, so realized decode gain ≤
  this ceiling. Marked as a **projection**.
- **~0.9 µs/node is this device/driver's replay dispatch floor**; it will differ on other
  GPUs/driver versions. The *direction* (dispatch floor survives replay) is the robust finding.

### 8.4 Verdict — **GO on graph-side glue node-collapse (bounded, cheap, Batty)**

- **GO, gate PASSED.** Under the production graph-replay path, collapsing the ~22 glue
  nodes/layer recovers **74–75%** of their replay time (~1.08 ms/token, ~5% of the 21.4 ms
  token), **byte-exact**, with **no grid.sync tax** — the opposite outcome to the §7 GEMV
  megakernel. This is the lever §7.4 pointed at, now **measured under the real path**, not
  assumed.
- **Costed plan for Batty (`optimizer.rs`):** fuse/collapse the elementwise+norm chain in
  the captured graph (target ~49 nodes/layer → a handful), leaning on the already-landed
  fused epilogues (#867 SwiGLU-mul, #854 skip-RMSNorm) so standalone nodes can be deleted.
  No cooperative kernel, no `grid.sync`, no numerics-reorder → **no Chew gate**. Bounded
  upside **~+5% decode** (46.7 → ~49 tok/s); low risk, small surface.
- **Next validation step:** measure node-collapse on the **real** captured decode graph
  (not the modeled chain) to convert the +5.3% ceiling into a realized number, once Batty
  has a candidate collapse in `optimizer.rs`.

---

- Env: `source /home/justinchu/onnx-genai/.cudaenv.sh`; `CUDA_VISIBLE_DEVICES=0 ONNX_GENAI_CUDA_DEVICE=0`.
- Baseline/eager/op-mix: `profile_native` as in §1 (`ONNX_GENAI_CUDA_GRAPH=0/1`, `ONNX_GENAI_PROFILE_OPS=1`).
- Phase B / P1.5 / P2-prototype micro-bench (throwaway, `#[ignore]`, never shipped):
  `cargo test --release -p onnx-runtime-ep-cuda --features cuda --test megakernel_headroom_gpu -- --ignored --nocapture`.
  Tests: `megakernel_headroom_probe` (glue), `megakernel_int4_mlp_probe` (fused single-CTA int4 MLP vs per-op),
  `grid_sync_capture_gate_probe` (cooperative-launch-under-capture gate),
  `megakernel_multicta_mlp_probe` (§7: persistent multi-CTA cooperative MLP vs per-op + grid.sync barrier cost),
  `glue_collapse_replay_gate_probe` (§8: glue per-op vs fused, both captured into a CUDA graph and timed under replay + per-node replay floor).
  Knobs: `NXRT_MK_HIDDEN` (6656), `NXRT_MK_INTER` (19968), `NXRT_MK_GLUE_OPS` (22),
  `NXRT_MK_ITERS` (200), `NXRT_MK_MLP_ITERS` (50), `NXRT_MK_MC_ITERS` (200), `NXRT_MK_LAYERS` (52).
- Model dir: `.../olive-recipes/meta-models-Muse-Glimmer-30B/cuda/int4/models` (`--pipeline`; never write through the symlink).
- Profilers (ncu/nsys) are blocked in-sandbox (`RmProfilingAdminOnly=1`); all numbers are built-in op timer + CUDA-event + wall-clock, per the profiling skill.
