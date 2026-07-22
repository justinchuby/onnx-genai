# Partial CUDA-Graph Capture within EP Subgraph Claiming

**Author:** Keaton (architecture) — via Copilot
**Date:** 2026-07-22
**Status:** Design (no implementation)
**Scope:** pure-Rust CUDA EP (`onnx-runtime-ep-cuda`) + session executor (`onnx-runtime-session`)
**Related:** [`RULES.md` §2 / §2.1](../RULES.md), [`docs/CUDA_GRAPH_CAPTURE.md`](CUDA_GRAPH_CAPTURE.md),
[`docs/EP_CLAIM_DIAGNOSTICS.md`](EP_CLAIM_DIAGNOSTICS.md), [`docs/NATIVE_CUDA_DECODE.md`](NATIVE_CUDA_DECODE.md),
[`docs/ORT2.md` §15.1 / §55.6](ORT2.md)

---

## 1. Problem statement & motivation

### 1.1 Node-level accept/reject is not the whole story

The EP's claim surface today is a per-node boolean with a reason:
`ExecutionProvider::supports_op` returns [`KernelMatch::Supported { .. }`] or
[`KernelMatch::Unsupported { reason }`] (`crates/onnx-runtime-ep-api/src/kernel.rs`).
That answers *"can this EP run this node at all?"* but it deliberately does **not**
answer the follow-on questions that actually decide decode throughput:

- **Which kernel implementation** will run for this claimed node? A single claimed
  op (`MatMulNBits`, `GroupQueryAttention`) already picks materially different
  implementations for the same shape family. The codebase already models this as a
  *separate* decision — [`KernelVariantSelection`] — precisely because "node-level
  claims alone do not explain what executed."
- **On which execution path** will the node run inside a claimed subgraph:
  recorded into a CUDA graph and replayed, or dispatched eagerly on the stream each
  step? This is the [`CaptureSupport`] decision, and it is *also* separate from the
  claim.

So there are already **three distinct kernel-path decisions** the EP makes, only
one of which (`KernelMatch`) is currently framed as "the claim." The claim is not a
boolean; it is the root of a small decision tree. This document makes that tree
explicit and, crucially, closes the gap for **partial** capture.

### 1.2 Measured evidence (2026-07-22, H200, Qwen2.5-0.5B)

Gaff's native-CUDA decode profile
(`.squad/decisions/inbox/gaff-decode-profile.md`, ORT 1.27) establishes the
size and source of the opportunity:

- With CUDA graph capture off, steady decode measured **435.34 tok/s median**
  (**2.297 ms/token**) and **226 kernels/token**. GPU activity occupied only
  **49.2%** of wall time: **46.1% kernel execution + 3.0% memcpy + 50.8% CPU
  dispatch/launch/idle gaps**, with an effective gap of about **5.2 us per
  launch**. KV traffic was **0 H2D / 0 D2H bytes**; the KV cache was already
  device-resident.
- The largest per-token kernel groups were skip RMSNorm (48 launches,
  0.253 ms), int4 GEMV (49, 0.213 ms), down-projection GEMV (24, 0.175 ms),
  gate/up SwiGLU GEMV (24, 0.154 ms), and GQA (24, 0.151 ms).
- Whole-step capture already succeeds on current `main` for this structurally
  eligible decode path. `ONNX_GENAI_CUDA_GRAPH=1` measured **825.51 tok/s
  median** (**1.211 ms/token, +89.6%, 1.90x**), with token-exact output,
  `enabled=true`, one capture, 30 replays, and **zero fallbacks**. That is
  **93.2%** of the approximately **886 tok/s** gap-free ceiling.

This workload is decisively **launch/CPU-dispatch-bound, not
HBM-bandwidth-bound**. The architectural priority is therefore:

1. when a decode step has device-resident owned KV, fixed addresses, static
   per-step topology, and no dynamic seam, capture the **whole step**;
2. use the partial-capture design below only when structural or declared
   metadata identifies dynamic seams that make whole-step capture decline.

Both decisions are metadata/structure-driven. The successful fixture's model
name is evidence, never a dispatch key; whole-step and partial eligibility must
not depend on model or architecture identity.

### 1.3 The partial-capture opportunity for decode

The single most valuable capture target is the steady-state one-token decode
subgraph: a fixed `[1,1]` token/position, a fixed-capacity attention mask, a
fixed-address shared KV cache, and a persistent `[1,1,vocab]` logits output
(`docs/CUDA_GRAPH_CAPTURE.md`). Capturing it collapses hundreds of per-step kernel
launches into one `cudaGraphLaunch`, which is where the H200 roofline gap lives on
the decode hot path.

Whole-step capture is the simplest and highest-value case, and §1.2 shows that it
already works when the structural invariants hold. The remaining blocker is for
decode graphs that contain **shape-massaging / dynamic** nodes — data-dependent
`Gather`/`Slice`, host-visible sequence-length math, control-flow, mask-delta
construction, or ops whose output shape is only known at runtime — interleaved
with a **static-shape core** of GEMMs, attention, norms, and elementwise ops that
are capture-safe. Under an all-or-nothing gate, one such node poisons the whole
step and forces eager execution.

The whole point of this design: **claim the whole subgraph, capture the static core,
and run the dynamic/host-massaging seams eagerly** — instead of surrendering capture
because the subgraph is not uniformly capturable.

### 1.4 Where we already are (do not reinvent)

The session executor already contains a first-generation implementation of exactly
this idea — **segmented capture** (`crates/onnx-runtime-session/src/executor.rs`):

- `RunMode::{Eager, Capture, Replay}` drive the device-graph lifecycle.
- `plan_capture_segments` partitions the plan into maximal contiguous **captured
  segments** separated by non-capturable **eager seams**, and *never* declines a run
  merely because some node is non-capturable — it only hard-declines on a
  graph-level precondition (outputs must be persistent device bindings) or when
  *nothing* is capturable.
- `node_capture_reason` classifies each node structurally (control-flow/sequence,
  unresolved data-dependent shapes, un-warmed kernel, or a kernel's own
  `CaptureSupport::Unsupported` reason).
- The EP trait already exposes segmented replay:
  `replay_device_graph_segment(index)` and its doc comment says outright *"Segmented
  capture claims a whole subgraph even when only parts are device-graph
  capturable."*

**This design does not throw that away.** It (a) formalizes the *claim-as-kernel-path*
model the code half-expresses, (b) moves the capture-eligibility *policy* to where
RULES wants fusion/dispatch policy — **inside the EP**, using the IR crate for
structural matching — while the executor keeps owning the mechanical
capture/replay/seam lifecycle, and (c) hardens the seam boundary contract (device
residency, KV stability, annotation-id uniqueness) that partial capture depends on.

---

## 2. EP claim as a kernel-path decision

### 2.1 The decision tree

Reframe a claim as a rooted decision, not a boolean. For a claimed subgraph the EP
resolves, per region, one of a small set of **kernel paths**:

```
supports_op (node claim)            ── "can the EP own this node?"  → KernelMatch
   └── (claimed node) kernel variant ── "which impl runs?"          → KernelVariantSelection
         └── capture path            ── "how is it executed in a captured run?"
                ├── CaptureRegion     ── recorded into a device graph, replayed
                ├── EagerDeviceSeam   ── dispatched on-stream each step (device-resident)
                └── HostSeam          ── runs on host/CPU (or forces a host round-trip)
```

Define the path kinds explicitly (proposed enum, EP-internal):

| Path kind | Meaning | Executed by | Per step |
|---|---|---|---|
| `CaptureRegion` | Maximal contiguous run of capture-safe device kernels with static device signatures | recorded once, `cudaGraphLaunch` thereafter | replay only; scalar device inputs updated before launch |
| `EagerDeviceSeam` | Device-resident node that is *not* capture-safe (allocates/frees, lazily compiles, D2H-validates, syncs, or has a data-dependent shape) | normal eager stream dispatch | re-run every step |
| `HostSeam` | Host/CPU-side work: control-flow, sequence ops, shape math, mask-delta construction, host-visible scalars | host, with explicit device transfers at the boundary | re-run every step; **CPU inputs re-bound every step** |

A claimed subgraph is thus a **totally-ordered partition**
`[R0, R1, R2, …]` alternating capture regions and seams. This is the same shape as
today's `CaptureSchedule { segments, boundaries }`, generalized so a seam carries
*why* it is a seam (device-eager vs host) rather than only "non-capturable."

### 2.2 Claim-time vs run-time split

Two-phase, matching the existing warm-then-capture flow:

1. **Claim / plan time (static):** the EP inspects graph structure + declared
   metadata and produces a *capture plan sketch* — for each node, its intended path
   kind and the structural reason. This is pure and side-effect-free; no shapes are
   concretized. It answers "what would we capture if shapes resolve as expected?"
2. **Warm / capture time (concrete):** on the first eligible step, once shapes are
   resolved and kernels warmed, the executor realizes the sketch into a concrete
   `CaptureSchedule`, records each `CaptureRegion` into its own device graph, and
   runs seams eagerly. This is where `plan_capture_segments` already lives.

The claim itself (`supports_op`) is unchanged in spirit — the EP still claims each
node it can run. What we add is a **capture-plan contract** the EP exposes so the
executor's segmentation is driven by EP *policy* instead of a generic per-kernel
boolean audit.

### 2.3 Why the EP owns the partition, not the executor

RULES §2.1: *"Fusion happens inside the EP (as part of its claim/compile), not in
generic graph code."* Capture-region selection is the same class of decision as
fusion — it is a structural, capability-gated rewrite of *how* a region executes.
Today the policy leaks into the session (`node_capture_reason` hardcodes
control-flow/sequence/shape predicates). The design moves the **policy** into the EP
(which knows its kernels' capture contracts) while leaving the **mechanism**
(capture/replay/seam interleaving, guards, lifecycle) in the executor, which owns the
stream and buffers.

---

## 3. Structural detection algorithm (generic, no model identity)

The EP decides each node's path kind purely from **graph structure + explicit
metadata**. No model names, architecture keys, layer counts, or magic dimensions
(RULES §2 / §2.1). A node is `CaptureRegion`-eligible **iff all** of the following
structural predicates hold; otherwise it is a seam (device-eager or host) with the
first failing predicate as its reason.

### 3.1 Capture-eligibility predicates (all structural)

1. **Static device signature.** Every input/output has a shape that is
   statically/symbolically resolved before capture (no data-dependent output shape).
   Mirrors the existing `node_capture_reason` "data-dependent shape was unresolved"
   check. Detected from shape inference + resolved bindings, not from op identity.
2. **No host-side data-dependent control.** The node is not control-flow (`If`,
   `Loop`, `Scan`) and not a sequence op — these branch on runtime values and cannot
   be baked into a static graph. Detected by op *category*, via the IR crate's op
   classification, never by a model hint.
3. **Capture-safe kernel only.** The warmed kernel's [`CaptureSupport`] is
   `Supported`: it must not, on its steady path, allocate/free device memory, compile
   a module lazily, perform device→host validation, or synchronize the stream
   (`docs/CUDA_GRAPH_CAPTURE.md`). This is the kernel's *own* declared contract,
   colocated with the kernel (same invariant as `KernelMatch`/`CaptureSupport`
   reasons: the reason travels with the decision).
4. **Stable device residency at the boundary.** All tensors the node reads/writes
   are device-resident and live at fixed addresses across replay (persistent
   buffers, initializers, shared KV, or another capture region's output). A node that
   must consume a freshly host-produced value each step is a boundary, not interior.
5. **Fixed launch configuration.** The node's grid/block/workspace derive only from
   the (now fixed) shapes and dtypes, not from per-step host scalars that change the
   *launch*, only ones that change *data* (scalar device inputs updated in place
   before replay are fine; a launch-shape that changes is not).

### 3.2 Pattern matching lives in the IR crate

Predicates (1)–(2) are **structural graph queries** — exactly the sanctioned use of
the IR crate's pattern-matcher/rewriter (RULES §2.1: *"The IR crate is the sanctioned
home for a reusable pattern-matcher + rewriter"*). The EP expresses region detection
as IR pattern queries ("op-category ∈ {compute, elementwise, norm, attention} ∧
all-shapes-static ∧ no-host-escape") rather than ad-hoc string matching on op names.
Predicates (3)–(5) are kernel-capability facts the EP already owns.

### 3.3 Maximal-region formation

Given per-node path kinds, form **maximal contiguous runs** of `CaptureRegion` nodes
in plan (topological) order, split by any seam node — identical to the existing
`plan_capture_segments` single pass. Two refinements:

- **Seam sub-classification.** A seam records whether it is `EagerDeviceSeam` or
  `HostSeam`, because their boundary contracts differ (§4). Today all seams are
  lumped as "non-capturable."
- **Region viability floor.** A capture region of trivial size (e.g. a single cheap
  elementwise op surrounded by seams) may cost more in graph bookkeeping than it
  saves; the EP may, via **explicit declared policy** (a threshold in metadata/config,
  §6), demote a sub-floor region to `EagerDeviceSeam`. The floor is a declared number,
  never a per-model constant.

### 3.4 Explicitly *not* used

No model name, architecture family, vendor string, tensor magic-dimension, or head
count enters the decision. If a decode subgraph and a vision subgraph both present a
static GEMM→norm→attention core flanked by dynamic gather seams, they partition
identically. That is the genericity contract.

---

## 4. Region ↔ seam boundary contract

Partial capture is only correct if data crosses region/seam boundaries deterministically.
The boundary rules:

### 4.1 Device residency & buffer stability

- **Interior of a capture region** references only fixed device addresses. A captured
  graph *bakes in pointers*; any buffer it reads/writes must not move or be
  reallocated for the life of the executable. This is already an invariant of the
  installed executable (`docs/CUDA_GRAPH_CAPTURE.md`: *"destroyed before its
  referenced buffers move or are released"*). Partial capture extends it: a region's
  **inputs produced by a preceding seam** and **outputs consumed by a following seam**
  must live in **persistent** buffers, not transient scratch, so the boundary address
  is stable across replay.
- **Seam → region hand-off.** An `EagerDeviceSeam` writes its outputs into the
  persistent buffers the next region reads. Because the seam runs eagerly every step,
  it may legitimately produce *new values* into the *same addresses* — the region's
  replay picks them up automatically (graphs capture addresses, not values).
- **Region → seam hand-off.** A region's outputs land in persistent device buffers a
  following seam reads. No copy is needed if the seam is device-resident.

### 4.2 CPU inputs must be re-bound every step

Prior lesson (`crates/onnx-genai-ort/src/cuda_rt.rs`, `docs/CUDA_GRAPH_CAPTURE.md`):
host-produced scalar inputs (token id, position, past-sequence-length, mask delta)
change every step and must be **re-written into their device buffers before each
replay** — the capture froze the *address*, so the value must be refreshed in place.
`HostSeam` outputs are exactly these: the boundary contract requires the executor to
copy host results into the region's persistent input buffers on every step
(`copy_from_host` into the fixed buffer), then replay. The existing capture-pass
already does `copy_from_host` for inputs before executing; partial capture makes this
a *per-step* obligation for every host→region boundary, not just the graph inputs.

### 4.3 Seam output shape and valid-extent invariant

A seam may perform internally data-dependent work, but any output consumed by a
following `CaptureRegion` must still have a **statically resolved physical
shape, fixed allocation extent, and stable address** for the lifetime of that
capture. If the seam produces a variable logical length inside a fixed-capacity
buffer, that valid length must be supplied as refreshed device data (for
example, a scalar or mask), and every captured consumer must provably bound its
reads to that logical extent.

A dynamic seam output whose physical shape changes, whose extent changes a
consumer's launch geometry, or whose stale tail could be observed by the
captured consumer cannot terminate at that boundary. The seam must extend
through the dependent nodes (or the remainder must stay eager) until a static,
fixed-extent boundary is reached.

### 4.4 KV cache across replay

The shared KV cache being fixed-address and device-resident is necessary but
not sufficient. A captured append must **not bake a growing sequence-length
offset into host launch parameters**: replaying such a graph would overwrite
the same slot and silently corrupt the cache.

KV append remains capture-interior only under the fixed-topology invariant:
physical KV capacity and pointers stay fixed; the graph has a fixed query
topology (the current M=1 decode case, or the generalized padded **M=maxK**
design); the append destination and attention read bound come from a
device-resident logical-length/index value refreshed in place; and rewind
changes only logical length/mask contents, not graph topology or bindings.
Under that invariant, a captured kernel may write a different logical slot on
each replay without changing its launch geometry.

Gaff's profile measured this successful whole-step case: device-resident KV,
token-exact output, one capture / 30 replays, and zero fallbacks. That validates
keeping KV append interior for graphs that satisfy the invariant; it is not a
model-name exception.

KV append is instead an `EagerDeviceSeam` when the offset is a host scalar baked
into kernel parameters, when sequence length changes grid/block/workspace, when
KV storage or output extent can move, or when rewind requires a different
topology. The classification is structural and metadata-driven in every case.

### 4.5 Process-unique annotation ids

Prior lesson from the ORT-backed path (`crates/onnx-genai-ort/src/session.rs`:
`gpu_graph_id` / `graph_annotation_id`): each captured graph must carry a
**process-unique annotation id** so replays never collide across sessions or across
distinct captured regions. For the pure-Rust EP, each `CaptureRegion` executable is
assigned a monotonically-allocated id (an `AtomicU64`, cf. `profile.rs`'s `NEXT`
counter) at capture time; segmented replay launches by that id in capture order. Two
regions in the same subgraph, or the same region re-captured after invalidation, must
never reuse an id, so a stale replay can be detected rather than launching a wrong
graph. This is a hardening of today's `graph_index` (which is only unique within one
schedule).

### 4.6 Boundary summary invariant

> Every value that crosses a region/seam boundary lives in a persistent, fixed-address
> device buffer; host-produced boundary values are re-copied into those buffers every
> step before replay; a seam's captured consumer sees a static physical shape and
> fixed extent, with any changing logical extent supplied as bounded device data;
> captured executables address KV and boundary buffers by stable pointer and are
> keyed by a process-unique id.

---

## 5. Interaction with existing fusion and the IR crate

### 5.1 Ordering relative to fusion passes

`CudaGateUpSwiGluFusion` (`crates/onnx-runtime-ep-cuda/src/optimizer.rs` ~L348–508)
runs as part of the EP's claim/compile and *reduces node count* by fusing paired
gate/up `MatMulNBits` + `Mul(Silu)` into one node. Capture-region detection must run
**after** all such structural fusions, because:

- Fusion changes the node set the partitioner sees; a fused SwiGLU node is a single
  capture-eligible unit whose kernel already declares its own `CaptureSupport`.
- The fusion's own eligibility checks already assert the *"capture-safe with a fixed
  device signature"* form (fp16 in/out, persistent initializer weights/scales) — note
  the existing comment in `eligible_projection`: *"the only form that is capture-safe
  with a fixed device signature."* Fusion and capture-eligibility share this vocabulary;
  the design makes it a shared, reusable predicate rather than a per-pass restatement.

So the EP pipeline is: **IR fusions/rewrites → warm kernels → capture-region
partition**. The partitioner consumes the post-fusion graph.

### 5.2 Shared pattern-matcher

Both fusion and region detection are structural graph queries. The design proposes
they share the IR crate's pattern-matcher/rewriter:

- Fusion = match a pattern, **rewrite** it to a fused node.
- Region detection = match a **region predicate** (op-category + static-signature +
  no-host-escape), **annotate** each node with its path kind (no rewrite; the graph
  topology is preserved, only metadata is attached).

This keeps a single sanctioned matching infrastructure (RULES §2.1) instead of the EP
growing a second ad-hoc classifier. `node_capture_reason`'s current predicates move
into EP-side IR queries.

---

## 6. Metadata contract additions

All assumptions must be **declared explicit metadata**, never inferred from model
identity (RULES §2). The additions are minimal — capture eligibility is mostly
derivable from structure, so metadata only covers what structure cannot express:

1. **Per-kernel capture contract (already exists).** `Kernel::capture_support() ->
   CaptureSupport`. No change; this is the authoritative per-kernel fact. Every new
   capturable kernel must implement it honestly.
2. **Region viability floor (new, declared).** An optional, declared threshold
   (config/metadata, e.g. `cuda.capture.min_region_nodes`) controlling §3.3's demotion
   of trivial regions. A plain number with a documented default; never a per-model
   constant baked in code.
3. **Boundary-persistence hints (new, declared *only if needed*).** If a value's
   residence/persistence cannot be inferred from the graph (e.g. an externally-owned
   binding), it is surfaced as explicit binding metadata (`ExternalBindings` already
   carries persistent-output info the capture path consumes). Prefer inference from
   the graph; fall back to declared metadata, fail clearly when a required property is
   missing rather than assuming it (RULES §2: *"Missing metadata fails clearly"*).
4. **Capture master switch (already exists).** `ONNX_GENAI_CUDA_GRAPH=1` /
   `NativeDecodeCudaOptions::graph_capture` (`docs/CUDA_GRAPH_CAPTURE.md`). Partial
   capture is gated behind the same switch; default remains eager.

Explicitly **not** added: any `model_type`, architecture family, or op-name allowlist
keyed to a specific network. The absence of such keys is part of the contract.

---

## 7. Correctness & fallback

Graceful degradation is layered, from most to least aggressive, and every level stays
on the CUDA EP (placement never changes):

1. **Per-region fallback.** A region that fails to *record* (a kernel declines
   mid-capture) is not fatal: the `SegmentCaptureGuard` already aborts the in-progress
   capture cleanly (ending stream capture so `reset_device_graph` is not wedged). That
   region is demoted to an `EagerDeviceSeam` and the rest of the schedule proceeds.
   *(Today a mid-record failure fails the whole capture pass and falls back fully
   eager; per-region demotion is a design refinement — see Phase 3.)*
2. **Whole-subgraph eager fallback.** If *nothing* is capturable, or a graph-level
   precondition fails (outputs not in persistent device bindings), the run executes
   fully eager on the CUDA EP — `plan_capture_segments` already returns a
   `CaptureDeclineReport` in these cases and the caller runs `run_plan_eager`. Tokens
   are byte-identical to the eager path.
3. **Invalidation.** Reset, rewind, prefill/multi-token shape changes, binding
   address/shape changes, and session drop invalidate captured executables
   (`docs/CUDA_GRAPH_CAPTURE.md`). A later step re-warms and re-captures a fresh
   schedule; a live executable is never reused across incompatible bindings. Partial
   capture inherits this wholesale and additionally re-assigns fresh process-unique ids
   (§4.5) on re-capture.
4. **Diagnostics.** Every seam boundary keeps its structured reason
   (`CaptureDecline` with node id, op type, domain, reason), surfaced via
   `ONNX_GENAI_LOG_CAPTURE_SEGMENTS` (already implemented in
   `log_capture_segmentation`). This is the §1 principle applied to capture: the
   *why* travels with the decision. Correctness contract: **captured + eager runs must
   produce identical tokens**; a conformance test asserts this (Phase 4).

The safety property: partial capture can only ever *improve* on today's all-or-nothing
gate — worst case it degrades to the exact eager path already shipping.

---

## 8. Cross-platform note

- The partition/policy types (`CaptureRegion`, seam kinds, path-kind enum) are pure
  Rust with no CUDA dependency and live behind the same feature gating as the rest of
  the CUDA EP. On targets without CUDA (notably **Windows arm64**, which has no CUDA —
  see [`docs/CROSS_PLATFORM.md`](CROSS_PLATFORM.md)), the CUDA EP is not compiled and
  the design contributes **no** new code to the non-CUDA build. The session executor's
  device-graph lifecycle already no-ops through the EP trait's default
  `Err("... not supported")` implementations for non-graph EPs, so the CPU/eager path
  is unaffected.
- **No SM assumption.** Capture eligibility is a function of kernel contracts and graph
  structure, not compute capability. The design must not gate a region on a specific
  SM/arch; a kernel that needs an SM floor expresses that in its own
  `KernelMatch`/`CaptureSupport` reason, keeping the partitioner arch-agnostic
  (consistent with `docs/CROSS_PLATFORM.md` PTX/arch handling).
- The ORT-backed path (`onnx-genai-ort`, `gpu_graph_id`) is **out of scope** here but
  shares the annotation-id lesson (§4.5); its `graph_capture` capability gating is
  unchanged.

---

## 9. Phased implementation plan

Work items are sized for follow-up agents. Each phase is independently
buildable/reviewable (RULES §9) and does not touch `docs/PROGRESS.md` here.

### Phase 0 — Formalize the path-kind model (types + docs only) (**DONE 2026-07-22**)
- Introduce an EP-internal `CapturePathKind { CaptureRegion, EagerDeviceSeam, HostSeam }`
  and a `SeamReason` that sub-classifies today's `CaptureDecline`.
- Wire seam sub-classification through `CaptureSchedule` (extend, don't rewrite).
- **Acceptance:** builds on CUDA and non-CUDA targets; existing capture tests pass
  unchanged; `ONNX_GENAI_LOG_CAPTURE_SEGMENTS` now prints seam *kind* alongside reason.

### Phase 0.5 — Measurement spike (**DONE 2026-07-22**)
- Gaff reused the existing capture-segment diagnostics and native decode
  instrumentation on the real H200 fixture before implementation investment.
- The spike quantified the launch-bound gap (226 kernels/token; 50.8% dispatch,
  launch, and idle gaps) and directly measured whole-step capture at 1.90x eager
  throughput with token-exact output and zero fallbacks (§1.2).
- For this fixture the result is stronger than a partial-region histogram: the
  structural plan admits one whole-step capture with no dynamic seam. Partial
  capture is therefore gated on separate fixtures whose structural diagnostics
  report actual seam positions and sufficiently large maximal static regions.
- **Acceptance: complete.** Evidence is recorded in
  `.squad/decisions/inbox/gaff-decode-profile.md`; whole-step capture is the
  first-line path, while partial capture remains justified for measured
  dynamic-seam cases.

### Phase 1 — Move region-detection policy into the EP
- Add an EP hook (e.g. `ExecutionProvider::plan_capture_regions(graph, resolved,
  bindings) -> CapturePlan`) with a default impl reproducing today's
  `node_capture_reason` predicates, so behavior is byte-identical at first.
- Executor's `plan_capture_segments` calls the EP hook instead of hardcoding the
  predicates; generic session code no longer owns capture *policy*.
- **Acceptance:** identical segmentation to `main` on the Qwen int4 decode fixture
  (same segment boundaries, same tokens); a unit test asserts the EP-produced plan
  equals the previous session-side plan for a representative graph.

### Phase 2 — Structural detection via the IR pattern-matcher
- Express predicates §3.1(1)–(2) as IR pattern queries (op-category + static-signature
  + no-host-escape); reuse the same matcher used by `CudaGateUpSwiGluFusion`.
- Run region detection strictly **after** fusion passes; add the shared
  "capture-safe fixed device signature" predicate used by both.
- **Acceptance:** region partition is provably independent of model identity — a test
  with two synthetic graphs (different fake "architectures", identical topology)
  yields identical partitions; clippy/Miri clean.

### Phase 3 — Per-region record fallback (no whole-pass abort)
- On a mid-record decline, demote just that region to `EagerDeviceSeam` via the
  existing `SegmentCaptureGuard` abort path and continue; only hard graph-level
  preconditions fall fully eager.
- Assign **process-unique** capture ids (`AtomicU64`) per region; segmented replay
  launches by id.
- **Acceptance:** an injected "declines on 2nd kernel" region demotes locally while
  other regions still capture; tokens identical to eager; no `STREAM_CAPTURE_INVALIDATED`
  after a demotion (stream lifecycle test).

### Phase 4 — Boundary hardening + conformance
- Enforce §4: persistent fixed-address boundary buffers, per-step host→region re-copy,
  KV-append seam/interior classification.
- Add a capture-vs-eager **token-equivalence** conformance test on the decode fixture
  and a boundary-stability test (buffers do not move across N replays).
- **Acceptance:** on a real graph that diagnostics show contains dynamic seams, the
  static core captures into ≥1 viable region with those seams eager; measured
  tokens/sec improves vs the fully-eager fallback; captured and eager tokens are
  byte-identical. Whole-step-eligible graphs continue to use one whole-step capture.

### Phase 5 — Metadata & policy knobs
- Add declared `cuda.capture.min_region_nodes` (viability floor) and any needed
  boundary-persistence metadata, with documented defaults and clear "missing metadata
  fails" errors.
- **Acceptance:** floor demotes trivial regions as configured; omitting an *inferable*
  hint changes nothing; omitting a *required* one yields an actionable error (RULES §1).

---

## 10. Open questions & risks

1. **Where exactly does the EP↔executor policy seam sit?** Phase 1 proposes an EP hook
   returning a plan the executor realizes. Alternative: the EP returns per-node path
   annotations and the executor forms regions. The hook keeps *policy* in the EP and
   *mechanism* in the executor; confirm this split with reviewers before Phase 1.
2. **KV-append prevalence.** §4.4 resolves the correctness classification:
   fixed-topology, device-indexed append may be interior; host-baked or
   launch-shaping offsets force a seam. The remaining audit is empirical: how
   often do dynamic-seam graphs satisfy the interior case?
3. **Seam count vs benefit.** The completed Phase 0.5 measurement proves the
   whole-step launch-gap benefit, but not the region-size distribution for
   graphs that actually contain seams. Many small regions may not beat one
   eager pass. The viability floor (§3.3/§6) mitigates this; each target graph
   class must expose its region histogram before Phase 4 investment.
4. **Re-capture churn.** Frequent invalidation (variable prompt handling, rewind-heavy
   workloads) could re-warm/re-capture often. Need a policy for when to *stop* trying to
   capture and stay eager after N invalidations — declared, not heuristic.
5. **Interaction with future multi-stream / graph-of-graphs.** Segmented replay is
   currently in-order on one stream. If regions become independent, could they replay
   concurrently? Out of scope now; the path-kind model should not preclude it.
6. **Trivially-small static cores.** Some architectures may present *no* capture region
   large enough to clear the floor. Design must degrade silently to eager there (it
   does), but we should measure how common that is before over-investing.

---

## Appendix A — Illustrative pseudocode (non-normative)

EP-side region planning (structural, no model identity):

```rust
// EP-internal. Runs AFTER fusion passes, on the post-fusion graph.
fn plan_capture_regions(
    graph: &Graph,
    resolved: &ResolvedShapes,
    bindings: &ExternalBindings,
) -> CapturePlan {
    let mut path = Vec::with_capacity(graph.plan.len());
    for node in graph.plan_order() {
        let kind = if is_control_flow_or_sequence(node) {
            // op-CATEGORY, via IR classifier — never op name / model id.
            CapturePathKind::HostSeam(SeamReason::HostControl)
        } else if !static_device_signature(node, resolved) {
            CapturePathKind::HostSeam(SeamReason::DataDependentShape)
        } else if let Some(reason) = kernel_of(node).capture_support().reason() {
            // Kernel's own declared contract (alloc/free, lazy compile, D2H, sync).
            CapturePathKind::EagerDeviceSeam(SeamReason::Kernel(reason.into()))
        } else if !boundary_buffers_persistent(node, bindings) {
            CapturePathKind::EagerDeviceSeam(SeamReason::UnstableBoundary)
        } else {
            CapturePathKind::CaptureRegion
        };
        path.push((node.id, kind));
    }
    // Maximal contiguous CaptureRegion runs, split by seams; demote sub-floor regions.
    CapturePlan::from_path(path, declared_min_region_nodes())
}
```

Executor realization (mechanism, unchanged in spirit from `plan_capture_segments`):

```rust
match mode {
    RunMode::Capture => for region in plan.regions() {
        match region.kind {
            CaptureRegion => {
                let _guard = SegmentCaptureGuard::arm(ep);   // aborts on ? early-return
                ep.begin_device_graph_capture(region.kernels())?;
                run_region_eager_recording(region)?;         // records into the graph
                ep.end_device_graph_capture()?;               // installs, assigns unique id
            }
            EagerDeviceSeam | HostSeam => run_seam_eager(region)?, // re-run every step
        }
    },
    RunMode::Replay => for region in schedule.regions() {
        match region.kind {
            CaptureRegion => { rebind_host_boundary_inputs(region); ep.replay_device_graph_segment(region.id)?; }
            EagerDeviceSeam | HostSeam => run_seam_eager(region)?,
        }
    },
    RunMode::Eager => run_plan_eager()?,
}
```

[`KernelMatch::Supported { .. }`]: ../crates/onnx-runtime-ep-api/src/kernel.rs
[`KernelMatch::Unsupported { reason }`]: ../crates/onnx-runtime-ep-api/src/kernel.rs
[`KernelVariantSelection`]: ../crates/onnx-runtime-ep-api/src/kernel.rs
[`CaptureSupport`]: ../crates/onnx-runtime-ep-api/src/kernel.rs
