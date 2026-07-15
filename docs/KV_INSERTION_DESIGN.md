# Runtime-Managed KV Insertion vs. Operator-Managed KV Update

**Status:** Architecture evaluation (decision-ready). **Revision 2** — reframed after the
"Mobius is our exporter" degree of freedom (see §8). No code changes.
**Author:** Tyrell (architect)
**Date:** 2026-07-15 (rev 2)
**Related:** `.squad/decisions/inbox/fact-checker-kv-insertion-da.md` (external-claim verification),
`.squad/decisions/inbox/tyrell-kv-mobius-exporter.md` (rev-2 decision note),
docs/DESIGN.md §40 (SWA/sinks), tiered-memory epic (paged/spillable KV), in-flight zero-copy-view work.

> **Revision 2 in one line.** The original memo treated exported GQA as a *fixed*
> constraint ("Path A is forced because we can't rewrite the baked-in op"). That is only
> true for **third-party pre-exported models**. **Mobius is our exporter**, so for the
> Mobius-controlled pipeline the attention-op contract is a **design lever, not a
> constraint** — and we should pull it toward the runtime-managed / functional path
> (Path B) as the strategic default. §8 is the authoritative recommendation; §1–§7 are
> the original rev-1 analysis, still valid but now *scoped to the third-party case* where
> noted. Where §6 and §8 disagree, **§8 wins.**

---

## 1. The proposal

> Move KV-cache **write** responsibility out of the attention operator and into the
> runtime / KV-manager. The attention op becomes **pure/functional**: it *reads* K/V
> (possibly paged / non-contiguous) bound at the right locations and *writes* its
> outputs to bound locations, with no in-place mutation of a cache buffer it doesn't
> logically own. The runtime inserts new K/V into the cache and *binds* tensors to
> attention I/O so they are read/written in the correct locations. Rationale: the
> operator stops doing "confusing" in-place ops (like GQA) that are "not really within
> spec."

This memo evaluates the claim, maps it onto vLLM / HF precedent, grounds feasibility in
**our** code, and gives a concrete phased recommendation.

---

## 2. Is the "not within spec" claim correct? — *Partly. Precise version below.*

**What is genuinely non-portable.** ONNX IR defines an inference model as a stateless,
side-effect-free function with SSA outputs. Nothing in the standard lets a *portable*
consumer assume an input buffer and an output buffer **physically alias**. So "the
operator mutates a cache buffer in place" is **not** a portable ONNX semantic — a
conforming runtime is free to allocate `present.*` fresh. That part of the colleague's
instinct is right: *physical in-place aliasing is a runtime/provider contract, not an
ONNX-graph guarantee.*

**What is sanctioned, and therefore NOT a smell.** `com.microsoft::GroupQueryAttention`
(and MHA) explicitly declare optional `past_key`/`past_value` inputs and
`present_key`/`present_value` outputs, and the contrib schema explicitly allows a past
tensor to *be the same tensor* as its present tensor (both sized to
`max_sequence_length`). The runtime knob that selects this same-storage mode is what we
(and ORT-genai) call `past_present_share_buffer`. So the in-place-into-a-shared-buffer
behavior of GQA is a **deliberate, documented provider contract** — calling *that
specific behavior* "not within spec" is inaccurate. It is in-spec *for the contrib
domain*; it is simply **not portable ONNX** and it **couples the op to buffer
allocation/lifetime**.

**Where the "confusing in-place" smell is real.** The coupling is the real cost, not a
spec violation:
- The op's correctness depends on the runtime having pre-sized and pre-bound a
  max-length buffer and on stable addresses across steps.
- It bakes cache-management policy (append position, sliding window, sinks, rollback)
  into an opaque fused node the runtime cannot introspect or reorder.
- It is a `com.microsoft` domain op: portability stops at ORT.

> Nuance (see fact-checker note §1): recent **standard** `ai.onnx::Attention` (opset
> 23/24 on ONNX main) *does* now specify logical KV-cache update — either the op
> concatenates incoming K/V to past, or an **external** update (e.g. `TensorScatter`)
> supplies the whole cache. So "logical cache update belongs outside the op" is becoming
> a standardized *option*. This still does **not** standardize ORT-style same-storage
> aliasing. **Flag for Fact Checker: already verified in the inbox note.**

**Verdict on Q1:** The claim is directionally right about *portability/coupling* but
wrong if read as "GQA's in-place update violates spec." Precise statement:
**physical in-place aliasing is non-portable and couples op↔allocator; ORT GQA adds a
sanctioned provider contract for it, so it is not a bug — it is a portability/coupling
trade-off.**

---

## 3. How the precedents actually do it (map the proposal onto them)

*(External facts — best technical understanding; corroborated by the fact-checker note
in `.squad/decisions/inbox/fact-checker-kv-insertion-da.md`, which cites vLLM v0.6.3 and
HF `cache_utils.py`.)*

**vLLM PagedAttention — a *split*, not "no write":**
1. `reshape_and_cache` (a **write kernel**) scatters newly projected K/V into paged
   physical blocks addressed by `slot_mapping`/block table
   (`block_idx = slot / block_size`, `offset = slot % block_size`).
2. A separate **attention kernel** *gathers* K/V from those blocks via `block_tables`.

The write is still a kernel — but it is an **explicit, separate step keyed by a block
table**, not an opaque mutation buried inside the fused attention op. This is *exactly*
the colleague's "runtime handles insertion; op reads bound locations," with the caveat
that the insertion is a real kernel launch, not free binding.

**HF Transformers `DynamicCache`/`StaticCache`:**
- The cache object **owns storage + update policy**; `DynamicLayer.update()`
  concatenates, `StaticLayer.update()` does in-place `index_copy_`.
- Important correction (fact-checker §3): `LlamaAttention.forward` calls
  `past_key_values.update(...)` **inside the attention module**, immediately before the
  math backend. So the accurate statement is: *the cache object owns storage; the
  mathematical attention backend consumes the already-updated K/V returned from it.* It
  is **not** literally "the framework updates it outside the attention module."

**Mapping to the proposal:** Both precedents separate **cache-state ownership** (manager)
from **the attention math** (functional kernel). Neither eliminates the write; both make
it an explicit, manager-owned step. The colleague's proposal is the vLLM decomposition:
**manager-owned scatter-write + functional block-table-aware gather-attention.**

---

## 4. Feasibility in OUR stack (grounded)

### 4.1 What our KV manager already owns

Our paged cache **already owns insertion** — this is not hypothetical:
- `PagedKvCache::write_token_kv` / `append_token_kv` allocate pages, perform page-level
  Copy-on-Write, and scatter a token's per-layer K/V into page storage
  (`crates/onnx-genai-kv/src/paged_cache.rs:157-246`).
- `PageTable` maps sequences → physical pages with allocate/free/retain and HOT/COLD
  tiering (`crates/onnx-genai-kv/src/page_table.rs:530-800`) — this is our **block table**
  analog.
- It supports sliding-window + attention-sink retention page-granularly
  (`paged_cache.rs:349-491`), rewind/fork/checkpoint (`lib.rs:72-102`), and tiered spill
  (`local_tiered.rs`, `connector.rs`).

**Critical limitation:** storage is **host `f32`**, and consumption is via
`materialize_sequence`, which **copies pages into a contiguous `[heads, seq, head_dim]`
buffer** (`paged_cache.rs:248-320`). It is not yet a device-resident, quantized, paged
GQA cache. So today the manager owns insertion *logically* but only as a **host mirror**.

### 4.2 How attention/KV is executed today (which path is live for decode?)

**The live decode path runs attention via ORT `com.microsoft::GroupQueryAttention`
baked into the exported (Mobius/ORT-genai) model — NOT our EP kernels.** Evidence:
- The engine holds a `Box<Session>` and drives decode through ORT sessions
  (`crates/onnx-genai-engine/src/engine.rs:12-27, 64, 142`).
- `crates/onnx-genai-ort/src/decode.rs` selects between two KV strategies
  (`decode.rs:39-50`):
  - **`SharedBuffer`** = the sanctioned `past_present_share_buffer` mode: one max-length
    `OrtValue` bound as *both* past input and present output (in-place GQA).
  - **`ZeroCopyRebind`** = ORT allocates `present.*`, rebound as next-step `past.*`
    (this is the **in-flight zero-copy-view work**). Engine selects per metadata
    (`crates/onnx-genai-engine/src/decode.rs:850-880`).
- Our paged cache is populated **after the fact** by copying ORT's `present.*` outputs to
  host f32 and appending them into pages:
  `mirror_present_kv_to_pages` (`crates/onnx-genai-engine/src/kv_bridge.rs:190-255`).
  **This is a mirror of operator-managed production, not device-page insertion**, and it
  is precisely the "extra copy" the proposal wants to avoid — except today we *add* a
  copy on top of GQA rather than replacing it.

**Our native EP attention kernels are already functional/pure — but not live for decode:**
- CUDA `AttentionKernel` takes `(Q,K,V[,mask]) → 1 output`, contiguous 4-D only, f32-only,
  no `past`/`present`, **no block table** (`crates/onnx-runtime-ep-cuda/src/kernels/attention.rs:219-340`).
  It is "Phase-2a," and the fusion pass that would wire it into a decode graph is
  explicitly **not yet wired** (`attention.rs:232-234`; CUDA EP is MatMul-only Phase-2a
  per `provider.rs:2-3`). CUDA registers `com.microsoft::Attention` but **not**
  `GroupQueryAttention` (`crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:70, 138`).
- CPU `FusedAttention` is likewise pure SDPA `Softmax(QKᵀ·scale[+mask])·V`, `[Q,K,V,mask]`
  in, one out, no cache (`crates/onnx-runtime-ep-cpu/src/kernels/fused_attention.rs:1-30`).

**So we already have the two halves the proposal wants** — a manager that owns paged
insertion, and functional attention kernels — but they are **not connected**, and the
live path is ORT GQA.

### 4.3 The crux constraint

"Runtime writes new KV into the cache and binds tensors to attention I/O" requires one of:

- **(i) Block-table-aware attention op** that accepts paged/non-contiguous K/V. Stock
  `com.microsoft::GQA` cannot do this — it consumes contiguous past/present. This is a
  **custom op**, viable **only on EPs we control** (`onnx-runtime-ep-*`), never on an
  exported ORT-genai GQA model.
- **(ii) IoBinding pre-placement**: runtime writes new K/V into the `present` buffer
  *before* the op runs, then binds. We already have `IoBinding` for zero-copy KV
  (`crates/onnx-genai-ort/src/lib.rs:4, binding.rs`; used in eagle3/gemma4 runners). But
  for stock GQA this **is** `past_present_share_buffer` — i.e. the op *still* logically
  owns the write into the shared buffer; binding alone cannot make GQA functional, and it
  cannot make the cache *paged* (GQA still needs contiguous max-length storage).

**Conclusion:** For **stock ORT-genai GQA exports**, the proposal is **not achievable by
binding**; making the op functional/paged requires **re-exporting** the model. For **our
native EP path**, a **block-table-aware functional attention custom op is genuinely on
the table** because we own the EPs and already own the page table.

---

## 5. Trade-offs

| Axis | Operator-managed (GQA `share_buffer`) | Runtime-managed insertion + functional paged attention |
|---|---|---|
| **Spec portability** | ORT contrib only; non-portable aliasing but sanctioned | Custom op → our-EP only; not portable either, but *we* define it |
| **EP portability** | Baked into export; CPU/CUDA/Metal/WebGPU must all honor GQA | New paged kernel per EP (CUDA/CPU/Metal/WebGPU) — real N-way cost |
| **Decode perf (M=1)** | 1 fused launch, contiguous coalesced loads, stable addr → CUDA-graph friendly | Extra write launch + block-table gather; risk of launch-bound / less-coalesced regressions at M=1 |
| **Long-context / batch** | Contiguous max-length buffer wastes VRAM; no prefix sharing | Paging → no fragmentation, prefix sharing, spill (our tiered epic) |
| **Complexity / correctness** | One well-understood node | Device page allocator + slot mapping + COW/rollback/window/sink correctness across streams |
| **Feature parity** | fp8/int8, RoPE, Q/K-norm, per-layer geometry already in the exported op | Must re-implement all of the above in the paged kernel + on-device quant cache |
| **Interaction: tiered-memory epic** | Fights it (contiguous, non-spillable) | Natural fit (paged/spillable is the whole point) |
| **Interaction: zero-copy-view work** | `ZeroCopyRebind` already avoids the past→present copy without paging | Complementary; device-resident insertion would remove today's host-f32 mirror copy |

---

## 6. Recommendation (concrete, phased)

**Adopt a dual-path model — do NOT try to unify.** The exported-GQA path and a native
paged path are different ABIs; pretending they are interchangeable is the main risk.

### Path A — Exported ORT-genai / Mobius GQA models: **keep operator-managed.**
- Keep `com.microsoft::GQA` with `past_present_share_buffer` (`decode.rs` `SharedBuffer`)
  as the **default compatibility + M=1 latency path**. We cannot change that op without
  re-export, and it is a sanctioned, launch-efficient contract.
- Continue the **`ZeroCopyRebind`** zero-copy-view work to eliminate the past→present
  copy where `share_buffer` isn't declared. This already captures most of the "avoid
  redundant copy" benefit the proposal wants, *without* paging.
- **Fix the real waste here:** the current `mirror_present_kv_to_pages` host-f32 copy
  (`kv_bridge.rs:190-255`) is pure overhead when pages are only a mirror. Only mirror
  into pages for sequences that actually need paging features (fork/rewind/spill/prefix),
  not on every decode step.

### Path B — Our native EP decode (`onnx-runtime-ep-*`): **adopt runtime-managed
insertion + a block-table-aware functional attention op.**
This is the vLLM decomposition and it is *where our architecture already points*:
1. **Phase B0 (design/ABI):** Define an internal paged-attention ABI: a
   `KVCacheWrite`/scatter primitive keyed by our `PageTable` block table, plus a
   functional `PagedAttention` op that gathers K/V via the block table and writes O to a
   bound output. No in-place mutation of inputs → clean SSA. Reuse the existing pure
   `AttentionKernel`/`FusedAttention` math as the gather-attention core.
2. **Phase B1 (device page table):** Promote `PageTable`/`PagedKvCache` from host-f32 to
   a **device-resident** allocator + slot mapping (CUDA first), so insertion happens on
   device and the host-f32 mirror disappears. Preserve COW/rewind/window/sink invariants.
3. **Phase B2 (kernel):** Implement the paged scatter-write + block-table gather
   attention kernel on CUDA; CPU keeps a materialize-then-contiguous fallback initially;
   Metal/WebGPU follow or fall back.
4. **Phase B3 (gate):** Ship behind a flag. **Do not make it default** until an **M=1
   p50/p99 decode latency gate** proves it does not regress interactive decode vs
   shared-buffer GQA, and correctness traces pass for append/rewind/COW/window/sink/
   batch-compaction/cancellation.

### Net
- **What changes where:** nothing changes for exported GQA models except trimming the
  unconditional page-mirror copy. The new functional paged op lives **only** in our EPs
  and only for models we run natively.
- This gives paging/prefix-sharing/spill (the tiered epic) where it pays off, keeps the
  portable/latency path intact, and makes the attention op functional **on the path where
  we actually control the op** — without the false promise of retrofitting stock GQA.

---

## 7. Devil's advocate — where the proposal is wrong or costly

1. **"Just bind and the op reads/writes the right place" is false for stock GQA.**
   Exported ORT-genai models bake the past/present contract in; they cannot become
   functional/paged by binding — only by **re-export**. Binding *is* `share_buffer`, and
   that is still operator-owned in-place. (kv_bridge/decode evidence above.)
2. **Separating write from read is not free.** Fused in-place GQA does one launch with
   contiguous coalesced loads and stable addresses (CUDA-graph friendly). A separate
   scatter-write + block-table gather adds a launch and indirection that can **regress
   M=1 decode latency** even while improving long-context throughput. vLLM earns the
   split via scheduler-level paging/prefix-sharing and heavily tuned kernels — it is not
   a latency proof for a generic M=1 decode shape.
3. **Paged/non-contiguous attention kernels are harder and often slower** than contiguous
   ones, and must be re-tuned per dtype / GQA ratio / RoPE / mask / window / sink / EP.
   Our current kernels are contiguous-only and explicitly tell callers to materialize
   strided data (`attention.rs`), so this is net-new, N-way EP work.
4. **Today's page path would *erase* the benefit:** insertion currently goes through a
   **host-f32 mirror** (`materialize_sequence` + `mirror_present_kv_to_pages`). Without a
   device-resident page table, "runtime-managed insertion" adds copies rather than
   removing them.
5. **Correctness surface explodes:** a wrong block table yields plausible-but-wrong
   tokens; speculative rollback, COW, eviction, sinks, and batch compaction must all keep
   the slot mapping exact across streams.
6. **"Portable ONNX" is not achieved** by adding a runtime-private paged cache ABI — it
   just moves the non-portability from a contrib op to our runtime. The only genuinely
   portable direction is the emerging standard `ai.onnx::Attention` external-cache form,
   which is a separate, longer bet.

**Bottom line:** the proposal is architecturally sound **and already half-built** in our
native EP direction, but it is **not** a transparent change to exported GQA models and
**not** guaranteed to win at M=1. Pursue it as an **opt-in, benchmarked, native-EP paged
path**, keep operator-managed GQA for exports, and delete the unconditional page-mirror
copy that currently adds cost with no functional change.

---

## 8. Revision 2 — Mobius is our exporter: the attention-op contract is a design lever

Rev-1 §6 split the world into **Path A (operator-managed GQA `share_buffer`)** and **Path
B (runtime-managed insertion + functional paged attention)** and recommended A as the
*default* for all exported models because "we cannot change that op without re-export."
That premise is now relaxed: **Mobius is our exporter, so re-export is a knob we own.**
This section supersedes §6 for anything Mobius produces.

### 8.1 Reframing Path A vs Path B

- **Path A is now the *compatibility* path, scoped to third-party / pre-existing GQA
  exports only.** For a model we did not produce, the baked-in `com.microsoft::GQA` +
  declared `past_present_share_buffer` metadata is a fixed contract; §1–§7's analysis
  applies unchanged, and `SharedBuffer` mode (`crates/onnx-genai-ort/src/decode.rs:45-47,
  242-243`) stays the correct, launch-efficient way to serve it.
- **For the Mobius-controlled pipeline, Path B (runtime-managed / functional attention)
  becomes the strategic default.** We choose what Mobius emits, so we can emit the
  contract that composes with paging, spilling, and our own EP kernels instead of the one
  that fights them.

The rev-1 "do NOT unify" guidance still holds — but the split is now **by model origin
detected at load time** (§8.4), not "operator-managed for everyone."

### 8.2 The central new question: what attention-op contract SHOULD Mobius emit?

Three candidates, graded against what our code already supports vs. what is net-new.

**Option 1 — Keep emitting `com.microsoft::GQA`, but drop the `past_present_share_buffer`
reliance (runtime binds functional in/out; op concats logically). — *Available today,
zero net-new kernel work.***
This is the key rev-2 finding and it is **already implemented and live**:
- Share-buffer is **optional, metadata-gated, not required**. `DecodeSession::new` sets
  `share_buffer = options.past_present_share_buffer.unwrap_or(session
  .past_present_share_buffer_supported())` (`decode.rs:239-241`). When the model does
  **not** declare it, the session runs in **`ZeroCopyRebind`** mode
  (`decode.rs:242-245`): **ORT allocates `present.*` fresh each step and rebinds it as
  next step's `past.*`** — i.e. the GQA kernel runs **functionally, with no in-place
  aliasing**, and the runtime owns the buffer lifetime (`decode.rs:546-560` rewind logic
  operates purely on runtime-held `current_kv`).
- The gate is a single metadata key: `past_present_share_buffer` /
  `past.present.share_buffer` (`crates/onnx-genai-ort/src/session.rs:474-484`). **If
  Mobius simply does not stamp that flag, the identical GQA graph runs functionally** —
  no re-export of the op itself, no new kernel, no schema fight. The contrib schema
  permits this cleanly (past and present are distinct optional tensors; shared storage is
  the *opt-in* case, not the default), and ORT's GQA kernel demonstrably runs in this mode
  — it is the path the in-flight zero-copy-view work already exercises.
- **What Option 1 does *not* give us:** it is functional but **still contiguous**. GQA
  consumes a contiguous past/present; it is not block-table-addressable, so it does **not**
  yet compose with paged/spillable KV. It removes the *aliasing* coupling but not the
  *contiguity* coupling. It is the cheap, safe first step, not the endpoint.

**Option 2 — Emit standard `ai.onnx::Attention` (opset 23/24) with logical KV-cache
update (concat-past or external `TensorScatter`). — *Net-new EP work; portable but not
covered today.***
- Our EPs do **not** register standard-domain `Attention`. CUDA registers only
  `OpKey::new("Attention", "com.microsoft", 1)` (`crates/onnx-runtime-ep-cuda/src/kernels/mod.rs:138`;
  supported-op list at `mod.rs:69-70`) — the com.microsoft SDPA/GQA baseline, not
  `ai.onnx::Attention` v23/24. CPU registers `com.microsoft::FusedAttention`
  (`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:208`), also contrib-domain SDPA.
- `TensorScatter` is **not implemented anywhere** in the KV crate or EPs (no
  `slot_mapping` / `block_table` / `scatter` primitive exists —
  `crates/onnx-genai-kv/src/*.rs` has none). So the external-update form is entirely
  net-new.
- Upside: it is the only genuinely *portable* direction (standard domain). Downside: it is
  the largest lift (new op registration on every EP + a scatter op) and standard
  `Attention` v23/24 is still stabilizing. **Good long-horizon bet, wrong first move.**

**Option 3 — Emit a decomposed form: runtime-managed KV-insertion step
(scatter / `reshape_and_cache`-like) + a functional block-table-aware paged-attention op
we own (the vLLM split). — *Net-new kernels, but the tiered-friendly endpoint.***
- This is exactly rev-1 §4.3(i) + §6 Path B. We already own **half**: a page table with
  allocate/free/COW/HOT-COLD tiering (`crates/onnx-genai-kv/src/page_table.rs`) and pure,
  cache-free attention math on both EPs (CUDA `AttentionKernel`,
  `crates/onnx-runtime-ep-cuda/src/kernels/attention.rs:219-234`, explicitly "not-yet-wired
  fusion"; CPU `fused_attention.rs`). We own **neither** the device-resident scatter-write
  nor the block-table gather kernel — those are net-new (rev-1 §4.1 limitation: storage is
  host-`f32`, consumed via `materialize_sequence` contiguous copy,
  `crates/onnx-genai-kv/src/paged_cache.rs:249`).
- This is the only option that is **block-table-addressed end to end**, i.e. the only one
  that composes with spill/tiering (§8.5).

### 8.3 Recommendation for Mobius — primary + fallback

**Primary direction: a two-stage migration that ends at Option 3, using Option 1 as the
immediately-shippable bridge.**

- **Stop stamping `past_present_share_buffer` in Mobius exports (Option 1) now.** This
  flips Mobius models onto the functional `ZeroCopyRebind` path with **zero net-new kernel
  work** — it is already live and tested. It removes the aliasing coupling and the
  fixed-max-length contiguous buffer that `SharedBuffer` forces (`decode.rs:260-266`
  requires `max_length` and pre-allocates it), which is the first thing that has to go for
  tiering. Gate it on §8.6.
- **Then build Option 3 behind our EPs (the vLLM split):** a device-resident, block-table
  keyed scatter-write + a functional paged-attention gather that reuses the existing pure
  `AttentionKernel` / `FusedAttention` math. Mobius emits the decomposed contract for the
  native-EP target. This is rev-1 Path B phases B0–B3, now promoted from "opt-in someday"
  to "the Mobius default target."

**Fallback:** if the paged kernels miss the §8.6 M=1 gate or slip, **Mobius stays on
Option 1 (functional contiguous GQA) as a stable, shippable default** — strictly better
than `SharedBuffer` for tiering-readiness and requiring no new kernels — while Option 3
matures behind a flag. **Do not** make Option 2 (standard `ai.onnx::Attention`) the
near-term target; treat it as a *separate portability track* to converge on once v23/24 is
stable and we need cross-runtime portability, not as the mechanism for the tiered-memory
epic.

> Net: **Mobius should emit functional (non-share-buffer) GQA immediately, and a
> decomposed scatter-write + paged-attention contract as the flagged strategic target.**

### 8.4 Compat / migration — detecting origin and routing at load time

We do **not** need to change how third-party models are served. Route by contract detected
at load:
- **Third-party pre-exported GQA** declaring `past_present_share_buffer` → **Path A /
  `SharedBuffer`**, unchanged. Detection already exists:
  `Session::past_present_share_buffer_supported()` (`session.rs:474-484`) reads that exact
  metadata key. Absence of the flag already selects the functional path (`decode.rs:242-245`).
- **Mobius-emitted functional GQA** (no share-buffer flag) → **`ZeroCopyRebind`**, today.
- **Mobius-emitted decomposed/paged contract** → our EP paged path, once it exists.
  Detect via an explicit Mobius provenance/producer tag in model metadata (add a
  `mobius.kv_contract = {shared_buffer|functional|paged}` custom-metadata key at export
  and branch on it in `DecodeSession::new`, alongside the existing share-buffer probe).

**Does this cost a permanent dual ABI?** Yes — and the Fact Checker flagged permanent dual
ABI as the #2 risk. Rev-2 accepts it, with scope control:
- The dual ABI is **not** GQA-shared-buffer vs. paged as *two live in-house code paths we
  must co-evolve forever*. It is **compatibility-shim (Path A) vs. strategic default (Path
  B)**. Path A becomes **frozen**: we keep the `SharedBuffer` code that already exists
  (`decode.rs`) to *consume* third-party GQA, but we stop *producing* it and stop investing
  in it. A frozen read-only compat path is a bounded, acceptable cost — the same shape as
  keeping an old file-format reader.
- The alternative (drop share-buffer support) breaks every existing third-party ORT-genai /
  Foundry GQA model, which is unacceptable. So the dual ABI is the price of not breaking
  the ecosystem; we minimize it by freezing, not by unifying.

### 8.5 Tiered-memory alignment — the strategic argument

This is the decisive reason to move the Mobius pipeline to Path B. The user's top-priority
epic is **run any-size model even without enough VRAM**, which requires a KV cache that can
**page and spill**:
- The spillable machinery already exists and is **block/page addressed**: `PageTable`
  hot(`Device::Gpu(0)`)/cold(`Device::Cpu`) tiering with transparent LRU offload/promote
  (`crates/onnx-genai-kv/src/tiered.rs:1-12`, `local_tiered.rs`), quantized int8/fp8 page
  storage (`tiered.rs`), paged COW/rewind/window/sink (`paged_cache.rs`).
- **Operator-managed `share_buffer` cannot compose with this.** It requires a *single
  contiguous max-length buffer* pre-allocated up front (`decode.rs:260-266`,
  `allocate_shared_buffers`) with stable addresses across steps — the exact opposite of
  block-table-addressed pages that can live partly on GPU, partly on CPU, partly on disk.
  A contiguous fixed buffer is non-spillable by construction; you cannot evict "the middle
  third" of a `share_buffer`.
- **Functional GQA (Option 1)** is spill-*compatible in spirit* (runtime owns the buffer)
  but still contiguous, so it does not yet page. **Only the decomposed block-table form
  (Option 3)** is the shape the tiered stack already speaks — its scatter-write is the
  device analog of `PagedKvCache::write_token_kv`, and its gather is block-table-addressed
  like `PageTable`.

**Therefore:** the tiered-memory epic *is* the business case for pushing Mobius to Option
3. Operator-managed share_buffer is architecturally incompatible with the #1 priority;
functional GQA unblocks the transition; block-table paged attention is the destination.

### 8.6 The M=1 gate (unchanged, still binding)

Whatever Mobius emits must **not regress single-token decode**. This gate governs both the
Option 1 flip and the Option 3 promotion:
- **Where it lives:** the existing `profile_decode` harness —
  `crates/onnx-genai-bench/src/bin/profile_decode.rs` (built with
  `--features bench-ort[,onnx-genai-ort/cuda] --bin profile_decode`, env-gated by
  `ONNX_GENAI_PROFILE`; see `docs/benchmarks/2026-07-13-foundry-local-analysis.md:36-59`
  for the interleaved-run methodology already in use).
- **Bar:** M=1 p50 **and** p99 decode latency for the functional/paged path must be within
  noise of the `SharedBuffer` GQA baseline on the same model, **plus** passing correctness
  traces for append / rewind / COW / sliding-window / sink / batch-compaction /
  cancellation. Only after both pass does the new contract become the Mobius **default**;
  until then it ships flag-gated.
- Rationale unchanged from rev-1 §7.2: a separate scatter-write + block-table gather adds a
  launch and indirection that can hurt M=1 even while helping long-context throughput. The
  CUDA-graph capture path (`decode.rs:20-37`, per-session capture ids) is share-buffer /
  stable-address friendly; the paged path must prove it stays capture-friendly or that the
  loss is within budget.

### 8.7 Revised bottom line (supersedes §6)

1. **Mobius: stop emitting `past_present_share_buffer` now** → functional GQA via the
   already-live `ZeroCopyRebind` path, zero new kernels, gated on §8.6.
2. **Mobius strategic target: the decomposed scatter-write + block-table paged-attention
   contract** on our EPs, reusing the existing pure attention kernels and page table;
   flag-gated until it clears the M=1 gate, then default.
3. **Third-party GQA: frozen Path A / `SharedBuffer` compat**, detected via existing
   metadata probe. Accept the bounded dual ABI as a frozen read-only shim.
4. **Standard `ai.onnx::Attention` v23/24: a separate, later portability track**, not the
   tiering mechanism.
5. This is the only sequencing that unblocks the tiered-memory epic without breaking the
   existing ecosystem or gambling on an unproven M=1 latency profile.
