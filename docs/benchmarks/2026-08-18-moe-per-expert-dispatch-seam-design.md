# Design scope: the executor→pager per-expert dispatch seam (#82/#87)

**Status:** design only — no implementation. Requested by the owner after Path 1
(fused int4 QMoE emission from Mobius) shipped and validated, to decide whether to
fund the runtime work that makes the budget-sweep **knee** appear.

**Hardware / model on every number below:** RTX 4060 8 GB (driver 591.55, CUDA 13.1,
WDDM), i7‑13800H, 63.8 GB DDR5. Model under discussion: `qwen15-moe-qmoe-mobius`
(Qwen1.5‑MoE‑A2.7B‑Chat GPTQ‑int4, fused `com.microsoft::QMoE`, 60 experts, top‑8,
24 layers, ~4.3 MiB per expert per fc1/fc2). Trace facts are **read from source**
(files/functions named); performance numbers are **cited** from the merged
`2026-08-18-moe-*` docs and marked *(measured, prior)*. **No new sweep was run and no
sweep numbers are invented here.**

---

## TL;DR (answering the owner's five questions)

1. **Where the seam lives:** three named points, all already threaded for whole-bank
   paging — `build_weight_handles` (`crates/onnx-runtime-session/src/executor/build.rs`
   ~L371–410) constructs the `LazyWeight` with **one** region and **one** key;
   `page_lazy_weight` dispatch (`crates/onnx-runtime-session/src/executor/dispatch.rs`
   ~L920–930) calls the pager **once per initializer** keyed by `vid.0 as u64`;
   `bind_block_quantized_moe` (`crates/onnx-runtime-ep-cuda/src/weight_paging.rs`
   ~L1668) streams **all** regions into **one** contiguous page. Per-expert geometry
   *already exists* in `onnx_runtime_loader` (`WeightRegionCatalog` /
   `ExpertTensorLayout` / `ExpertWeightRegion`, `crates/onnx-runtime-loader/src/weights.rs`
   L89–260) but is **not wired into the CUDA paging path** (that path has zero `expert`
   references today).

2. **What the executor must tell the pager, and when:** with the intra-layer
   prediction window measured at **≈0** (`2026-08-18-moe-prediction-window.md`,
   *measured, prior*), demand-prefetch cannot hide latency, so the design must be
   **static / frequency-based pinning decided once at load**, not per-token demand
   paging. That collapses the executor→pager contract to a **one-time** "here are the
   per-expert regions and here is the resident set" message — no per-token routing
   feedback loop, far less plumbing than a demand-driven seam.

3. **Is per-expert keying enough, or must the QMoE kernel change? — the decisive
   question. Answer: keying alone is NOT enough. The kernel must change, OR routing
   must be lifted out of the fused op.** The `com.microsoft::QMoE` kernel
   (`crates/onnx-runtime-ep-cuda/src/kernels/qmoe.rs`) fuses routing **and** expert
   GEMV into one dispatch and indexes `fc1_experts_weights` as **one contiguous base
   pointer** (`packed[expert_row * packed_in + …]`, `expert_row = expert*out+…`). The
   executor hands it a single `base_ptr` (dispatch.rs L940) and the kernel decides
   *inside the launch* which experts to read. So the executor cannot know the routed
   set before the kernel runs, and the kernel cannot tolerate an expert's bytes being
   absent. **This is why the churn doc says it "cannot be implemented as an allocator
   tweak."** Concrete options in §Q3.

4. **Smallest change that produces a knee:** the **VMM stable-VA arena + static hot
   set** variant (§Q4). Keep one contiguous virtual address for the bank (so the
   kernel's base+arithmetic is unchanged), but back only a **pinned hot subset** of
   experts with physical granules and map the **cold** experts to a **zero/backstop**
   granule that is refilled on demand. Requires (a) per-expert regions in `LazyWeight`,
   (b) a residency policy input, and (c) a **guard that guarantees a routed cold expert
   is mapped before the kernel reads it** — which, because routing is in-op, forces
   either a pre-pass router or a kernel-level residency check. Estimated **medium**
   (see §size).

5. **Cost/risk:** per-expert *keying* is **neutral** (arm B, ≤ measured overhead,
   `2026-08-18-moe-per-expert-paging-churn.md`, #1308, *measured, prior*); per-expert
   *paging* churn is bandwidth-dominated but **wins** under skew (top‑8 streams 25% of
   whole-bank, skew removes a further ~46% of page-ins, *measured, prior*). The
   oversubscription cliff was **refuted as span-count-driven** — it is WDDM fault-in
   beyond usable VRAM (`copilot-1295-vmm-oversubscription-cliff.md`), and this design
   keeps the resident set **within** budget by construction, so per-expert keying does
   **not** move the cliff. Confirmed below, not assumed.

**Headroom (measured, added after the owner's node-count observation):** experts are
**67.6%** of the bank (f32 model) / **77.2%** (fp16 deploy) — the max fraction per-expert
paging can remove, **3.5–5× the read-every-token dense weights**. Well above the "worth
it" line. But the predicted curve has a **dense-dominated linear Region A** below the
~1.8 GiB (f32) / ~1.2 GiB (fp16) dense working set — so the knee only appears if the
sweep **pins dense weights first** and ranges the budget into the expert regime. See
§"Byte-split headroom and predicted sweep curve."

---

## Byte-split headroom and predicted sweep curve (measured on this box)

Before the mechanism, the ceiling — the same discipline that killed the multi-row GEMV.
Node counts and per-tensor byte totals were **read directly** from the two built models
(initializer dims × dtype width; int4 experts stored as packed `UINT8`). Tensors were
classified by their **transitive terminal compute op** (following `Reshape`/`Transpose`/
`Cast` pass-throughs), because ~half the expert tensors reach `QMoE` via a `Reshape` and
are miscounted by immediate-consumer alone.

**`qwen15-moe-qmoe-f32` (9.12 GiB total, the model the owner inspected):**

| Class | Bytes | Share | Read pattern | Pageable? |
|---|---|---|---|---|
| **Expert banks (QMoE)** — weights 5940 MiB + scales 371 MiB | **6311 MiB** | **67.6%** | routing-sparse (top-8/60) | yes (`QMoe` boundary) |
| Dense read-every-token — lm_head 1187 + attn/o/shared 625 + gate 11 | **1823 MiB** | **19.5%** | every token, full | yes (`MatMul`/`MatMulNBits`) |
| Embeddings + RoPE caches | 1203 MiB | 12.9% | row-gather (~1 row/token) | **no** (`Gather` is not a paging boundary) |
| norms/bias/const | 0.9 MiB | 0.0% | — | — |

**`qwen15-moe-qmoe-mobius` (fp16 deploy build, 7.75 GiB):** expert **77.2%**, dense
read-every-token **15.2%**, embed-gather **7.5%**. (fp16 halves the fp32 scales, lm_head
and embeddings, so the expert share *rises* — the int4 expert weights dominate more.)

**Headroom verdict:** experts are **67.6% (f32) / 77.2% (fp16)** of the bank — the
maximum fraction of streamed bytes that per-expert paging can ever remove. That is **well
above** the owner's "clearly worth it" line (85%… actually the *relevant* comparison is
to the dense floor: experts outweigh the read-every-token dense weights **3.5:1** in f32
and **5:1** in fp16). The prize is real; the seam is not a ≤1 ms curiosity like the GEMV
ceiling was.

**Predicted curve (state before running — this is the falsifiable prediction).**
Per-token H2D bytes vs VRAM weight budget `B`, under per-expert paging with a static
hot-set and a **dense-first** pin policy (see below). Three regions:

- **Region A — `B` below the dense working set** (≈1.8 GiB f32 / ≈1.2 GiB fp16): budget
  cannot even hold the read-every-token dense weights, so they stream: htod ≈
  `(dense − B) + full expert traffic`, **linear and steep**. **The original 256–2304 MiB
  sweep window sits almost entirely here** — which predicts that a naive sweep in that
  window shows a *linear* curve *even after* per-expert paging lands, because the streamed
  bytes are dense-dominated, not expert-dominated, at those budgets. That would be a
  correct-but-misleading "still linear" result. **To see the expert knee the sweep must
  pin dense first and range `B` from ~dense-size upward** (≈1.2 → 7.7 GiB fp16).
- **Region B — dense pinned, remaining budget buys hot experts:** htod falls from ~841
  MiB/token (8 routed cold experts × ~105 MiB f32; ~102 MiB fp16) toward ~0 as the
  hottest experts become resident. Because measured skew is **mild** (top 12.5% of expert
  keys → 27.1% of traffic, `2026-08-18-moe-per-expert-paging-churn.md`), the knee is
  **concave but gentle**, not a sharp cliff: the first pinned (hottest) experts give the
  steepest drop, later ones flatten.
- **Region C — `B` ≥ full bank** (needs > 8 GB; unreachable on the 4060): htod → 0.

**So the predicted shape is: a steep dense-dominated linear segment, then a gentle concave
expert knee once budget exceeds the dense working set.** Knee *height* ≤ expert share
(67.6%/77.2%); residual floor once dense is pinned = **0** htod for dense (they are
resident), so there is **no residual linear floor in Region B** — the floor the owner
worried about only exists in Region A, where budget is too small to pin dense at all. If a
sweep with dense-first pinning and `B` in the expert regime is **still linear**, that
falsifies this model and means a cold expert is still forcing whole-bank refill (the Q3
kernel-mapping issue) — a real finding.

**Dense-first is the correct policy and the seam must not foreclose it.** lm_head +
attention + shared expert are read **every** token; experts only when routed. Under a
fixed budget the optimal residency is: pin all read-every-token dense weights first, then
spend the remainder on the highest-frequency experts. Embeddings are `Gather`-fed and
**not** a paging boundary, so they are resident-or-host regardless — they consume budget
(1.2 GiB f32 / 0.6 GiB fp16) but contribute ~0 htod (one row/token). The design's policy
input (§Q2) must therefore rank **dense > hot experts > cold experts**, not experts-only.

**Note on the 8 GB box:** the fp16 model is 7.75 GiB and *almost fits entirely* in 8 GB,
so the offload regime is only reached at deliberately constrained budgets. That is fine
(it is the regime we want) but it means the knee is a **small-budget** phenomenon here;
on a larger MoE whose bank far exceeds VRAM the same mechanism gives a deeper knee.

---

## The path a QMoE weight travels today (read from source)

1. **Candidate classification.** `lazy_weight_candidates(graph)`
   (`crates/onnx-runtime-ep-api/src/weight.rs` ~L170) marks an initializer pageable iff
   it is consumed only by a `LazyWeightBoundary` op. That enum includes `QMoe` and
   `BlockQuantizedMoe`, so `fc1_experts_weights` / `fc2_experts_weights` of a
   `com.microsoft::QMoE` node **are** eligible.

2. **Handle construction (ONE region, ONE key).**
   `build_weight_handles` (build.rs L371–410) takes `external_mmap_provenance(weight)`
   → a single `(mapping_id, offset, len)` for the **whole** `[60,2816,1024]` tensor,
   wraps it as **one** `ExternalMmapRegion`, and builds
   `LazyWeight::new(boundary, dtype, shape, vec![region], …)`. The handle is stored in
   `HashMap<ValueId, WeightHandle>` keyed by the tensor's `ValueId`. **This is the
   single-key origin.**

3. **Per-node paging drive (ONE call per tensor).** During decode,
   dispatch.rs L920–930 finds the lazy handle for the input `vid`, then calls
   `self.ep.page_lazy_weight(vid.0 as u64, lazy, source)`. The `key` doc on the trait
   (`crates/onnx-runtime-ep-api/src/provider.rs` L436) is explicit: *"`key` is a stable
   per-weight identity (the executor passes the initializer's value id)."* One value id
   ⇒ one key ⇒ whole bank.

4. **Binding (streams ALL regions into ONE contiguous page).**
   `bind_block_quantized_moe` (weight_paging.rs L1668) does
   `alloc_raw(region_bytes_len())` then a `for region in &weight.regions` loop that
   H2D-copies each region into a **running contiguous offset**. Note it *already*
   iterates a `Vec<region>` — but it copies **all** of them and has **no selection**;
   residency is keyed and cached at whole-page granularity in `CudaWeightResidency`
   (`resident_mapped`, weight_paging.rs ~L2552), VMM stable-VA arena, 2 MiB granules.

5. **Kernel launch (ONE base pointer, in-op routing).** dispatch.rs L940 substitutes
   `paged.device_ptr()` as the input `base_ptr`. `qmoe.rs` then runs `qmoe_route` →
   `qmoe_activate` → `qmoe_linear_*` **inside a single kernel dispatch**, indexing the
   bank by arithmetic from that one base pointer. Header (qmoe.rs L1–7): *"Expert
   tensors remain resident on one GPU… Weight paging, asynchronous prefetch… are
   intentionally deferred."*

**Consequence (already in the decision record):** `htod = bank − budget`, perfectly
linear, no knee — because the paging key is the **bank**, not the **expert**, and
fusion does not change the key.

---

## Q1 — Where the seam has to live (narrowest points, named)

Per-expert residency needs a change at **each** of the three whole-bank points, plus a
policy source. None is a refactor; each is a widening of an existing single-item path
to a per-expert list:

| Seam point | File / function | Change |
|---|---|---|
| Region geometry | `onnx_runtime_loader::WeightRegionCatalog::classify` (weights.rs L165) → feed into `build_weight_handles` (build.rs L385–408) | Replace the single `vec![region]` with the **N per-expert** `ExpertWeightRegion`s the catalog already computes for an ExpertMajor QMoE tensor. `LazyWeight.regions` is already a `Vec`. |
| Keying | `page_lazy_weight` call site (dispatch.rs L926) + `CudaWeightResidency::resident_mapped` (weight_paging.rs L2552) | Key residency by **(value_id, expert_index)** rather than `value_id` alone, so an expert can be resident/evicted independently. Measured **neutral** (arm B, #1308). |
| Binding / residency | `bind_block_quantized_moe` (weight_paging.rs L1668) + VMM arena | Map only the **resident** experts' granules; leave cold experts unmapped within the stable VA (§Q3/§Q4). |
| Policy input | new, at load: `plan_placement`-adjacent (`crates/onnx-runtime-ep-cpu/src/weight_offload/placement.rs`) already models per-expert `RegionPlacement` | Supply the **static hot set** (see §Q2). |

The per-expert **representation already exists and is validated** in the loader
(`ExpertTensorLayout { experts, rows_per_expert, storage_elements_per_row, order:
ExpertMajor, quantization }`, weights.rs L98; `ExpertWeightRegion { expert, offset,
len }`, L138). The gap is purely that the **CUDA paging path never consumes it** —
`weight_paging.rs` has **zero** `expert` references today.

## Q2 — What the executor tells the pager, and when

**Measured input:** intra-layer prediction window ≈ **0**
(`2026-08-18-moe-prediction-window.md`, *measured, prior*). Demand-paging at expert
granularity therefore **cannot** hide H2D latency behind compute — by the time the
in-op router picks experts, the kernel needs them **now**.

**Therefore: static / frequency-based pinning, decided once at load.** The
executor→pager contract collapses to a **one-time** message:

- the per-expert region list (from Q1), and
- a **resident hot set** = the top-frequency experts (plus always-on layers 1–2,
  `2026-08-18-moe-router-skew-granite.md`, *measured, prior*) that fit the budget.

No per-token routing feedback, no prefetch queue, no prediction-window plumbing. This
is materially **less** plumbing than a demand-driven seam — it is the reason the design
is medium and not large. `prefetch_lazy_weight` (provider.rs L455, currently a stub
`Ok(false)`) can **stay a stub**; the static path does not need it.

## Q3 — Per-expert keying is NOT sufficient; the kernel must change (answer first)

The fused `com.microsoft::QMoE` kernel makes two hard demands that keying alone cannot
satisfy:

1. **One contiguous base pointer.** `qmoe.rs` indexes `packed[expert_row*packed_in+…]`
   from a single `base_ptr`. Splitting *keys* does not give the kernel N pointers — it
   still receives one `device_ptr` (dispatch.rs L940). So the resident experts must
   share **one contiguous virtual address range**, i.e. a **VMM stable-VA arena** where
   per-expert granules are mapped/unmapped under a fixed VA (the machinery
   `resident_mapped`/#716/#776 already provides). Keying without a stable VA would hand
   the kernel a pointer to a partial page → out-of-bounds reads.

2. **Routing is fused in-op, so residency must be guaranteed before launch.** The
   executor cannot learn the routed set before the kernel runs (router is
   `qmoe_route`, the first kernel entry *inside* the same dispatch). If a routed expert
   is a **cold** (unmapped) granule, the kernel faults. There is no way around this
   with paging keys alone. The three ways out, smallest first:

   - **(a) Static hot set + guaranteed-mapped cold backstop (§Q4).** Pin the hot set;
     map cold experts' VA to a **backstop granule** so a read never faults, and refill
     it. Produces a knee (hot experts are free; cold experts cost H2D) **without**
     changing the kernel's ABI — but needs an arena residency guard and accepts that a
     cold routed expert is served from a just-filled granule (a correctness-preserving
     stall, not a fault). **Medium.** *This is the recommended minimum.*
   - **(b) Split routing out of the fused op** into a standalone `Router` node so the
     executor sees the routed set, then page only routed experts before an
     expert-GEMV op. Changes the emitted graph **and** the kernel decomposition; large;
     also loses the fusion the whole task just added. **Not recommended.**
   - **(c) Kernel ABI change** to take per-expert device pointers (array of N pointers)
     so residency is per-expert-explicit. Cleanest long-term, but is exactly *"reworking
     how the QMoE kernel binds its weights"* — **large**, and the owner should make
     that call.

**Plain statement the owner asked for:** the *minimum* (a) does **not** require
reworking the kernel's math, only adding a residency guard around its single-pointer
arena. The *general* solution (c) **does** rework how QMoE binds weights and is a large
commitment.

## Q4 — Smallest change that produces a knee

**VMM stable-VA arena + static hot set (option a).** Concretely:

1. Wire per-expert `ExpertWeightRegion`s into `LazyWeight.regions` for QMoE fc1/fc2
   (Q1, build.rs). *(no kernel change)*
2. Extend `CudaWeightResidency` to map a **pinned hot subset** of experts to physical
   granules within the bank's stable VA, and back the **cold** experts with a shared
   backstop granule that `bind_block_quantized_moe` refills on demand. *(weight_paging.rs
   L1668 + arena)*
3. Feed the hot set from a **load-time frequency policy** (Q2), reusing the per-expert
   `RegionPlacement` planner that already exists in `weight_offload/placement.rs`. **The
   policy must rank dense (read-every-token: lm_head + attention + shared expert) *above*
   hot experts** — see the byte-split section: pinning a routing-sparse expert ahead of a
   read-every-token dense tensor is strictly worse under a fixed budget.

This is the minimum that lets the pager **keep a hot subset resident and stream the
rest**, so the sweep's `htod` stops tracking `bank − budget` and bends at the point
where the budget covers the hot set — **the knee** — and *which* experts you pin starts
to matter. It deliberately does **not** build demand prefetch, expert-parallel
sharding, or a kernel ABI change.

**Falsifiable acceptance:** re-run the device-tier VRAM budget sweep
(`2026-08-18-moe-*` method) on `qwen15-moe-qmoe-f32` with this path. Knee ⇒ success;
still-linear ⇒ a cold expert is still forcing a whole-bank refill (a real finding, not
a failure).

## Q5 — Cost and risk

- **Keying overhead:** neutral — arm B (all experts resident, per-expert keys) measured
  ≤ prior overhead, #1308 (*measured, prior*). More, smaller keys is **not** the cost.
- **Paging bandwidth:** the win. Per-expert@top‑8 streams **25%** of whole-bank; skew
  removes a further **~46%** of page-ins (`2026-08-18-moe-per-expert-paging-churn.md`,
  *measured, prior*). This is the knee's source.
- **Granule floor:** Qwen1.5‑MoE int4 experts ≈ **4.3 MiB** > the **2 MiB** VMM granule,
  so experts are **individually pageable** (unlike granite's 0.75 MiB, granule-blocked).
  This model is a suitable target; granite is not.
- **Oversubscription cliff:** refuted as span-count-driven — the cliff is WDDM fault-in
  **beyond usable VRAM**, not region count (`copilot-1295-vmm-oversubscription-cliff.md`,
  *measured, prior*). This design keeps the **resident set within budget by
  construction** (static hot set sized to fit), so it does **not** push past usable VRAM
  and does **not** move the cliff. The reasoning holds for this case: the risk was
  "more spans," the measured driver was "over-budget residency," and we stay
  under-budget.
- **Correctness risk (option a):** a routed **cold** expert served from a just-refilled
  backstop granule adds a **stall**, not a wrong answer — the bytes are correct once
  copied. The only true failure mode is reading before refill completes, which the
  residency guard must serialize. This is the one piece that needs careful
  implementation and a test.

---

## Size estimate

| Option | Kernel change | Scope | Estimate |
|---|---|---|---|
| (a) stable-VA arena + static hot set + residency guard | **No** (math unchanged; add guard) | build.rs regions, weight_paging.rs residency/arena, load-time policy, one guard + test | **Medium** — the recommended minimum to get a knee |
| (b) split routing out of the fused op | Yes (decompose) | emitter + kernel split; **loses fusion** | Large — not recommended |
| (c) per-expert-pointer kernel ABI | **Yes** (rebind) | qmoe.rs ABI + all call sites | Large — owner's call |

**Recommendation:** fund **(a)**. It is the smallest change that can produce the knee,
reuses the per-expert region machinery that already exists in the loader and the CPU
placement planner, needs no kernel-math change, and its one real risk (cold-expert
refill serialization) is testable against the existing `qwen15-moe-qmoe-f32` /
`qwen15-moe-dense-f32` oracle pair. If the owner wants the general solution, that is
**(c)** and it is explicitly a *"rework how QMoE binds its weights"* commitment.

**Honest blocker statement:** even option (a) is **not** an allocator tweak — it
requires a residency guard that interacts with an in-op router the executor cannot
observe before launch. That is the precise reason the churn doc flagged this seam, and
it is why this is a scoping deliverable rather than a same-session implementation.
