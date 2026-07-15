# Runtime-Managed KV Insertion vs. Operator-Managed KV Update

**Status:** Architecture evaluation (decision-ready). No code changes.
**Author:** Tyrell (architect)
**Date:** 2026-07-15
**Related:** `.squad/decisions/inbox/fact-checker-kv-insertion-da.md` (external-claim verification),
docs/DESIGN.md §40 (SWA/sinks), tiered-memory epic (paged/spillable KV), in-flight zero-copy-view work.

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
