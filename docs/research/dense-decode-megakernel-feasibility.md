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

### 8.5 REALIZED on the production graph — bf16 SiLU/SwiGLU-mul collapse (measured, this H200)

The §8.4 ceiling (+5.3%) is a bound on collapsing *all* ~22 glue nodes/layer. Auditing the
**real** Muse-Glimmer-30B decoder graph against the landed fused epilogues shows most of that
glue is **not** byte-exactly collapsible here — so the realized number is bounded well below
the ceiling. What the real graph actually contains, per 52-layer decoder (runtime op histogram,
`ONNX_GENAI_PROFILE_OPS=1`, eager, one forward):

| op_type | count | /layer | collapsible byte-exact? |
|---|---|---|---|
| `MatMulNBits` | 417 | 8 | no — irreducible int4 GEMV (§7, #898); out of scope |
| `GroupQueryAttention` | 52 | 1 | no — GQA-decode; out of scope |
| `SimplifiedLayerNormalization` | 312 | 6 | **no for bf16** — see below |
| `Add` | 311 | 6 | ~208 are constant `gamma+1` folds (#872 `CudaFoldConstantAdd` REGRESSED −2.8%, not shipped); ~104 are residual adds |
| `Mul` | 210 | 4 | 52 are the SwiGLU multiply → **collapsed here** |
| `Reshape` | 208 | 4 | structural/kernel-coupled (GQA head-split); not a standalone deletion |
| `Sigmoid` | 104 | 2 | 52 are the MLP SiLU gate → **collapsed here**; 52 are the attention-output gate `sigmoid(gate)*attn` (not self-gated SiLU) |
| `Cast` | 2 | ~0 | already folded 834 → 2 by `CudaDropNormalizationCasts` + `CudaFoldConstantCast` |

**Why the elementwise/norm fusions were dormant on this model:** every SiLU/SwiGLU fusion was
gated to `Float16`, but Muse-Glimmer's activation stream is **bf16**, so `CudaSiluFusion` never
fired. The architecture is Gemma3-style *sandwich* norm — the residual `Add` comes **after** the
post-norm (`x + norm(y)`), so #854's `SkipSimplifiedLayerNormalization` (which computes
`norm(x + skip)`) does apply across the layer seam, **but only f32/f16 skip kernels exist**: the
f16 skip kernel rounds the residual sum to f16 *before* the RMS reduction (byte-exact), whereas
the f32-template path (the only one bf16 could reach) accumulates over the **unrounded** fp32 sum.
That changes the reduction vs the standalone bf16 `Add`→norm → **not byte-exact → flagged for
Chew/Sebastian (needs a bf16 skip kernel that rounds the sum), NOT implemented here.**

**What shipped (byte-exact, reuses #867):** extend `CudaSiluFusion` to accept `BFloat16`. The
standalone `Sigmoid(x)` + `Mul(x, sigmoid)` + `Mul(silu, up)` chain then collapses through
`CudaSiluFusion` → `CudaSwiGluFusion` into the tagged decomposed `Mul[_cuda_silu_mul]`, which the
runtime lowers to the already-landed **`decomposed_silu_mul_bf16`** kernel. That kernel is
documented (and here confirmed) to reproduce the standalone per-op bf16 rounding exactly — sigmoid
and the products each rounded via `__float2bfloat16_rn`. `CudaGateUpSwiGluFusion` requires an fp16
activation, so it stays dormant on bf16: **the int4 GEMVs are left untouched** (per §7 / #898), and
only the glue nodes collapse.

**Measured (H200, `CUDA_VISIBLE_DEVICES=0`, `ONNX_GENAI_CUDA_GRAPH=1`, `--steady --warmups 1
--runs 3 --tokens 128`, 3 interleaved A/B rounds on the same release binary):**

| metric | before | after |
|---|---|---|
| decode | 21.19 ms/token | 20.99 ms/token |
| throughput | **47.20 tok/s** (mean; 47.18/47.24/47.19) | **47.63 tok/s** (mean; 47.55/47.74/47.60) |
| glue nodes/layer | ~22 | ~20 |
| `Sigmoid` (per forward) | 104 | **52** |
| `Mul` (per forward) | 210 | **158** |
| total decoder nodes | 2458 | **2354** (−104; −2/layer) |
| greedy tokens (24-tok, deterministic) | `[24, 372, 1045, 10016, …, 1740, 2885]` | **identical (byte/token-exact)** |

**Realized decode delta: 47.20 → 47.63 tok/s = +0.9% (−0.19 ms/token), byte-exact.**

**Interpretation — a small-but-real SHIP, far below the +5.3% ceiling, honestly bounded by what
is collapsible.** The +5.3% assumed all ~22 glue nodes/layer collapse; on the real bf16 graph only
the **2** SiLU/SwiGLU-mul nodes/layer are byte-exactly collapsible with a landed kernel. The
larger glue populations are each blocked for a concrete reason: the 6 norms/layer have no
byte-exact bf16 skip kernel (reduction-rounding mismatch → Chew), the 4 constant `gamma+1`
adds/layer are a **measured regression** (#872), and the 4 reshapes/layer are GQA head-split
metadata coupled to the attention kernel, not free-standing deletions. Removing 104 dispatch-bound
nodes recovers ~0.19 ms/token — consistent with the §8.1 ~0.9 µs/node replay floor plus the small
real memory traffic of the 52 removed `Sigmoid` launches over the 19968-wide intermediate. The
glue that remains interleaves with the dominant GEMV serial cost (§8.3 caveat), so realized ≤
ceiling as predicted. **Verdict: SHIP** — zero-risk, byte-exact, no GEMV touch, no numerics
reorder, no Chew gate; it simply activates the landed #867 bf16 kernel that the `Float16`-only pass
gate was hiding. Further glue collapse on this model requires a **bf16 skip-RMSNorm kernel that
rounds the residual sum** (Sebastian/Chew) before #854's fold can fire byte-exactly on bf16.

### 8.6 REALIZED — byte-exact bf16 skip-RMSNorm kernel + fold (measured, this H200)

This closes the §8.5 blocker: the missing **byte-exact bf16 skip kernel**. Muse-Glimmer-30B
(Gemma3 sandwich-norm) has 6 `SimplifiedLayerNormalization`/layer × 52 = 312 norm nodes, each
residual seam being `Add(residual, sublayer_out) → SimplifiedLayerNormalization`. #854's
`SkipSimplifiedLayerNormalization` fold applies across the seam, but until now no bf16 skip kernel
existed that is **bit-identical** to the standalone `Add(bf16)` + `rmsnorm_bf16` pair, so Batty
(#900) could not collapse the norms.

**What was built (kept, proven byte-exact):**
- A new **`skip_rmsnorm_bf16`** NVRTC kernel (`kernels/normalization.rs`): computes
  `sum = __float2bfloat16_rn(f32(residual) + f32(x))` (bit-for-bit what a standalone bf16 `Add`
  writes), stores that bf16-rounded sum as the next layer's residual, then runs the **identical**
  `rmsnorm_bf16` block-tree reduction (fp32 accumulate over the *rounded* sum, same `NORM_BLOCK=256`
  reduction config). So `y` and the residual `sum` are bit-identical to running the two ops
  separately. Guarded native dispatch with a graceful `run_bf16_via_f32` fallback for non-dense /
  bias / header-unavailable cases (Rule 11 portability). **GPU-verified 0-ulp** vs standalone
  `Add(bf16)`→`rmsnorm_bf16` at H=6656 for both bf16 and f32 gamma
  (`bf16_native_skip_rmsnorm_is_byte_exact_with_{bf16,f32}_gamma`).
- An optimizer fold **`CudaSkipRmsNormFusion`** that collapses the `Add → SimplifiedLayerNormalization`
  seam into one `com.microsoft::SkipSimplifiedLayerNormalization`, deleting the standalone `Add` +
  norm launches (bf16/f32-gamma-over-bf16 only; fp16 is left to `CudaSkipRmsNormMatMulFusion`).

**Numeric fidelity (Chew gate): byte-exact.** Real-model greedy stream, fold OFF vs ON, is
**bit-identical** (48/48 tokens; 128-token run also identical):
`[24, 372, 1045, 10016, 328, 2885, 262, 5091, 8811, 511, 917, 4921, 768, …]`. The fold is byte-exact
by construction (the kernel rounds the residual sum before the reduction, and the reduction order is
unchanged), so no reduction is reordered — **no numerics divergence.**

**Measured decode A/B (H200, `ONNX_GENAI_CUDA_GRAPH=1`, `--steady --warmups 1 --runs 3 --tokens 128`,
interleaved on one release binary):**

| metric | fold OFF (default) | fold ON |
|---|---|---|
| `SkipSimplifiedLayerNormalization` nodes | 4 | **104** (2 seams/layer × 52) |
| standalone `Add` + norm seams removed | — | 104 `Add` + 104 norm folded away |
| decode | 20.93 ms/token | 21.25 ms/token |
| throughput | **47.77 tok/s** | **47.06 tok/s (−1.5%)** |
| greedy tokens | reference | **byte-identical** |

**Verdict — kernel SHIP, fold NO-SHIP (opt-in, default OFF).** The fold is byte-exact but a measured
**~1.5% regression** under CUDA-graph replay, and the per-op eager timer *confirms the mechanism*: in
eager the fused glue is faster (Add+norm+skip summed 355.6 → 349.0 ms — the launch saving is real),
but under **graph replay those launches are already amortized** (§8.1 ~0.9 µs/node floor), so the only
thing left is that one heavier single-CTA `skip` kernel replaces a fast **multi-CTA** `Add` (whole-GPU)
plus a single-CTA norm. At M=1 the norm reduction is single-CTA (all H=6656 in one block), and folding
the residual add *into* it **serializes** work that the standalone `Add` had spread across all 132 SMs —
the exact structural reason the multi-CTA GEMV megakernel was NO-GO (§7) and glue collapse only paid
+0.9% (§8.5). Decode is confirmed at its launch-amortized latency floor; collapsing these two nodes
recovers nothing and the serialized add costs net-negative.

The **kernel is kept** (proven byte-exact, portability-gated) because it is the prerequisite for the
*only* path that could win on bf16: folding the norm into the neighbouring **multi-CTA int4 GEMV**
prologue/epilogue (the bf16 analogue of `CudaSkipRmsNormMatMulFusion`, which keeps the reduction
distributed across the GEMV's CTAs instead of a standalone single-CTA launch) — a larger GEMV-kernel
job, not this fold. The fold itself is **retained behind `ONNX_GENAI_CUDA_ENABLE_SKIP_RMSNORM_FUSION`
(default OFF)** purely for A/B and for a future bandwidth-bound device where the multi-CTA-Add-vs-single-CTA-skip
trade may flip. **No production regression: the default binary is unchanged (47.77 tok/s == baseline).**

---

### 8.7 PROTOTYPED — bf16 norm-into-GEMV-prologue fusion (measured **NO-GO**, this H200)

> **Finding-only record.** The prototype pass measured below was a throwaway probe; because the result
> is a regression **and** numerically divergent (not byte-exact), the code was **NOT landed** — no
> production `src/` change, no opt-in flag, no kernel change ships from this work. Only this NO-GO
> finding and its mechanism are retained. Reproduce from the design notes here if a future device or a
> new cooperative-reduction prologue kernel (see the mechanism below) revisits it.

§8.6 closed by naming the *only* bf16 path that could still win: fold the standalone bf16 RMSNorm
into the **prologue of the following multi-CTA int4 GEMV** — the bf16 analogue of the fp16
`CudaSkipRmsNormMatMulFusion` — so the RMS reduction rides the GEMV's full-device (132-SM) occupancy
instead of a standalone single-CTA launch. Pattern: `x → SimplifiedLayerNormalization(x, γ) → MatMulNBits`
collapses to one fused GEMV that reads `x`, computes RMS(x) in its prologue, normalizes, then does the
int4 GEMV. This is the distributed-reduction fix that §8.6's standalone fold (−1.5%, #903) lacked.
The prototype was a dedicated bf16-only optimizer pass (redirect the follower GEMV's activation to the
norm input `x`, bind γ at slot 6/5, set the prologue attr+epsilon, delete the norm) — measured, then
discarded per the finding-only note above.

**Structure (Gemma3 sandwich-norm, from the real graph — 312 `SimplifiedLayerNormalization`,
417 `MatMulNBits`):** only the two **pre-norms** per layer (`input_layernorm` → Q/K/V/gate GEMVs;
`pre_feedforward_layernorm` → gate/up GEMVs) are `norm → GEMV` seams (`x` is the residual stream, not a
GEMV output → no residual/preceding-GEMV fold). 2 foldable pre-norms/layer × 52 = **104 foldable**.
The qk-norms (→ Mul/Reshape), post-attention and post-feedforward norms (→ Add) do not feed a GEMV and
are left standalone.

**Kernel:** no changes needed. The bf16 GEMV stages through fp16 in `MatMulNBitsKernel::run_bf16`
(casts bf16→fp16 — mantissa-lossless — runs the tuned fp16 int4 GEMV that honors the prologue attr,
casts fp16→bf16), so a bf16 GEMV carrying the prologue attr reuses the existing fp16 prologue kernel.

**Measured (H200, `ONNX_GENAI_CUDA_GRAPH=1`, interleaved A/B, `--pipeline`):**

| Configuration | tok/s | Δ |
|---|---|---|
| Baseline (default, pass OFF) | **47.9** | — |
| Norm→GEMV-prologue fused ON | **45.7** | **−4.6 % REGRESSION** |

Greedy 128-token stream **DIVERGES at ≈token 38** (`…2963, 38, 9520…` → `…2963, 38, 8323, 2481, 9520…`)
→ **not byte-exact** (would need Chew even if perf were neutral).

**Mechanism (op timer, eager decomposition):** folding removed only **~1.8 ms** of standalone norm
(106.55 → 104.78 ms — the bf16 standalone norm `rmsnorm_bf16` is a block-tree parallel reduction and is
*already cheap*), but added **+180 ms** to the GEMVs (368.91 → 548.81 ms, **+48 %**). The fp16 prologue
reduction is `skip_rmsnorm_f16_warp_half4` — a **single-warp serial** sweep of the whole H=6656 row while
the block's other warps idle at `__syncthreads`, re-executed **once per following GEMV** (input_layernorm
fans out to ≤4 GEMVs). So the fold trades ~1.8 ms of a cheap parallel reduction for ~180 ms of serial
warp reduction placed **on the critical GEMV path, redundantly per fan-out follower**.

This is the **exact inverse** of the fp16 skip case that motivated `CudaSkipRmsNormMatMulFusion`: there
the standalone `skip_rmsnorm` was ~24 % of decode and the prologue absorbed it for free; the
`fusion_benefit_is_positive` gate was calibrated on that fp16 assumption, which is **FALSE for bf16**
because the bf16 block-tree norm is already efficient. The kernel-level *distributed-reduction* idea is
sound in principle, but the prologue reduction that actually exists is **single-warp-serial**, not a
genuine multi-CTA cooperative reduction across the GEMV grid — a real win would require a **new
cooperative-reduction prologue kernel**, out of scope for this bounded gate.

**Verdict — NO-GO / NO-SHIP.** The prototype pass was **not landed** (finding-only; no `src/` change,
no opt-in flag, no dead code) — the default binary is unchanged (**48.15 tok/s == baseline,
byte-identical stream**). Kill-gate CLOSED: **decode remains at its launch-amortized latency floor
(~47.6–48 tok/s)**; do not pursue norm→GEMV-prologue on H200 without first writing a true multi-CTA
cooperative-reduction prologue. This is the **fourth** independent confirmation of the launch-amortized
latency floor (megakernel §7, glue-under-replay §8, standalone skip-fold §8.6, and now this).

---

## 9. The "47 = launch-amortized floor" framing was INCOMPLETE — the int4 GEMV was a KERNEL-EFFICIENCY floor at 29% peak DRAM (measured, H200, `ncu`)

§7–§8.7 concluded four times that decode is at a **launch-amortized latency floor** (~47.6 tok/s).
That conclusion is correct about *launch/dispatch* overhead — graph replay already amortizes it, so
every launch-collapse lever (megakernel, glue node-collapse, norm-into-GEMV) recovers ≈0. **But
launch overhead is not the hardware limit.** The batch-1 decode roofline is HBM **bandwidth**:
~15.37 GB of int4 weights ÷ 4.8 TB/s (H200 HBM3e) ≈ 3.2 ms/token ⇒ ~300 tok/s roofline; a well-tuned
single-stream engine reaches ~100–180 tok/s. At 47 we are ~6× off roofline. That gap lives inside the
int4 GEMV **kernel efficiency (achieved HBM bandwidth)** — a number we had *never* measured; every
prior §7–§8.7 number was a *relative* timing between fusion variants.

### 9.1 The money measurement (ncu, `--graph-profiling node`, dominant decode kernel)

Dominant decode kernel = `matmul_nbits_gemv_f16_scales_f16_zp_splitk` (the per-layer Q/K/V/O/gate/up/down
int4 GEMVs + the vocab-202048 lm_head). ncu on the H200 (GPU pinned idle, `CUDA_VISIBLE_DEVICES`):

| Metric | Measured | Reading |
| --- | --- | --- |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | **~29%** | **NOT bandwidth-saturated** — headroom exists |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | ~73% | SM pipes are the co-bottleneck |
| achieved occupancy | ~91% | occupancy is NOT the problem |
| stall: **Long Scoreboard** (global-load latency) | **~40.7%** (#1 stall) | classic latency-bound signature |
| dequant-ALU pipe utilization | ~64.8% | **ALU co-bound** — dequant math competes with loads |
| non-GEMV fraction of decode | ~39% | **Amdahl ceiling**: even a perfect GEMV caps end-to-end gain |

**Interpretation:** the GEMV is **co-bound** — 40% Long-Scoreboard (load latency) AND 65% dequant-ALU.
It is *not* purely latency-bound, so the textbook latency-hiding levers (more warps, deeper pipelines)
were expected to help only partially, and the 39% non-GEMV tail Amdahl-caps end-to-end wins. Naïve
roofline (close to 60–70% peak BW ⇒ ~1.3–1.6× ⇒ 60–75 tok/s) is an **upper** bound that ignores the
ALU co-bound and the Amdahl tail.

### 9.2 Three phased levers — BUILT & MEASURED (H200, steady, GPU-idle-pinned, `ONNX_GENAI_CUDA_GRAPH=1`)

Baseline this pass: **~47.6–47.7 tok/s** (plain `..._zp_splitk`, K_SPLIT=2).

**Phase A — higher-way split-K (2→4→8).** More cooperating warps per output column ⇒ more concurrent
global loads to hide Long Scoreboard. Env-parametrized `ONNX_GENAI_GEMV_KSPLIT`.

| K_SPLIT | tok/s | DRAM% (ncu) | dominant-kernel µs |
| --- | --- | --- | --- |
| 2 (baseline) | 47.3 | 29.4 | 56.0 |
| 4 | 47.8 (+1%, noise) | 27.75 | 59.4 (slower) |
| 8 | 45.4 (regression) | — | — |

**Phase A = NO-GO.** Occupancy is already ~91% and the scheduler already has eligible warps
(Not-Selected 18.6%), so adding warps does not reduce per-warp load latency; K=4 did *not* raise DRAM
(it fell) and made the dominant kernel *slower*; K=8 regresses (too little K-work/warp ⇒ reduction+sync
dominate).

**Phase B — cp.async double-buffered weight loads (attacks the 40% Long Scoreboard directly).** Separate
2-stage `cp.async.ca.shared.global` (4 B/lane) template, arch-guarded `#if __CUDA_ARCH__>=800` with a
synchronous fallback, env `ONNX_GENAI_GEMV_CPASYNC`.

| variant | tok/s |
| --- | --- |
| baseline (sync) K=2 | 47.6 |
| cp.async K=2 | **41.2 (−13%)** |
| cp.async K=4 | 41.0 |

**Phase B = NO-GO.** The 4-byte-per-lane cp.async granularity is too small: per-group commit/wait +
the extra shared round-trip cost more than the load latency they hide. A profitable cp.async needs a
16 B/`.cg` async copy over a **Marlin-style tiled weight relayout** so each async transaction moves a
full 128-bit sector — that is a from-scratch kernel, out of scope for this bounded pass.

**Phase C — dequant-ALU relief.** The dequant (`int4x8_to_half2x4_sub`) **already** uses Marlin-style
LOP3 (4×`lop3.b32` + debias) — that is *why* the ALU pipe is at 65%. The one remaining ALU lever is to
**fold the per-block scale into the dequant's zp-subtract**: replace the plain `q=(code-zp)` then a
separate `__hmul2(q, scale)` with a single `fma(code, scale, -zp*scale)`, dropping **4 `__hmul2` per 8
weights** (~20% fewer fp16 ALU ops in the MAC). Env `ONNX_GENAI_GEMV_FOLDSCALE`.

| variant | tok/s | Δ | dominant-kernel µs | DRAM% |
| --- | --- | --- | --- | --- |
| baseline | 47.7 | — | 56.0 | 29.4 |
| **fold-scale K=2** | **48.9–49.0** | **+2.7%** | 53.4 (−5%) | 30.9 |
| fold-scale K=4 | 48.7 | (no stacking) | — | — |

**Phase C fold-scale = the only kernel-level win (+2.7%), but NOT byte-exact.** ncu confirms the
mechanism: dominant kernel 56→53.4 µs, DRAM 29.4→30.9%, Long-Scoreboard *relatively* rose (the ALU
relief shifts the balance back toward loads) — i.e. genuine ALU relief. On the **real** Muse-Glimmer-30B
the greedy 128-token stream is **byte-identical** to baseline. **However** the fused `fma(code,scale,
-zp·scale)` sums two fp16-rounded terms (`code·scale` + a pre-rounded `-zp·scale`) versus the plain
path's exact integer `(code-zp)` then a single rounded multiply — so it is strictly less accurate per
element. It **fails** the existing synthetic parity guard
`fp16_gemv_matches_dequant_reference_phi_int4_zp_dims` (K=3072 N=3072, asymmetric zp): **worst rel
0.104 vs the 5e-2 bound** (max-abs 1.19e-2 was *within* the 2.55e-2 abs bound — only the near-zero-column
relative check fails). Plain split-K passes that guard.

### 9.3 Verdict — the hoped 1.3–1.6× did NOT materialize; the GEMV is near its efficient design point

- The "47 = launch floor" framing was **incomplete**: the real limit at 47 was a **kernel-efficiency
  floor within the existing GEMV design** (29% peak DRAM), NOT hardware and NOT (only) launch overhead.
- BUT the optimistic roofline (60–75 tok/s via split-K + cp.async) **over-estimated**: the kernel is
  **ALU-co-bound** (65% dequant pipe), not purely latency-bound, so split-K (NO-GO) and small-granularity
  cp.async (NO-GO, −13%) do not raise achieved DRAM. The dequant is **already** LOP3/Marlin-style.
- **The only realized kernel-level lever is fold-scale (+2.7%)**, and it carries a per-element accuracy
  cost (fails the asymmetric-zp parity guard at rel 0.104). It is shipped **opt-in, default OFF**
  (`ONNX_GENAI_GEMV_FOLDSCALE=1`); production + CI stay on the exact plain split-K path. **Chew gates**
  whether the +2.7% is worth flipping the default given the accuracy trade.
- **Bigger single-GPU wins require a from-scratch Marlin-style int4 kernel** (16 B/`.cg` cp.async over a
  tiled weight relayout, so the async pipeline actually amortizes) — a multi-week rewrite, not a bounded
  lever. Beyond the kernel, the ~39% non-GEMV tail Amdahl-caps decode; the large multipliers
  (speculative decoding, tensor-parallel) stack on top and are the higher-leverage next steps.

Repro: `matmul_nbits.rs` fold-scale entries `..._foldscale_splitk` / `..._foldscale_zp_splitk`, gated by
`gemv_foldscale_enabled()`; A/B with `ONNX_GENAI_GEMV_FOLDSCALE=1` vs unset on
`profile_native --model <dir> --pipeline --ep cuda --backend native --steady --warmups 2 --runs 5 --tokens 128`.

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
