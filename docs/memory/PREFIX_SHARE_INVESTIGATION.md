# VMM page-level prefix sharing: viability probe & design (#777)

Status: **isolating GPU probe + written design + the allocator primitive landed.
No KV integration, no engine or production-kernel change this round** (per
scope). The primitive is proven sound at the CUDA driver level and against the
real #740 allocator/ledger, and the cross-reservation multi-map entry point
(`commit_shared_prefix`) is now **implemented and tested** on that allocator.
This document records the measured answers, the design decisions that do **not**
get implemented yet, what the landed entry point does, and the smallest next
increment.

References **#777** (this issue), **#727** (one handle at two VAs, captured-graph
safe), **#740/#745** (authority-scoped physical-handle pool, `carve()`
suballocation, 0->1 / 1->0 granule-ref attribution), **#759** (dummy-page
tail), **#776** (2 MiB granule), **#787** (token-major floor), **#750**
(multi-request serving), **#783** (KV as a stride descriptor). Background — the
authoritative design line — lives in
[`MEMORY_ARCHITECTURE.md` → "KV layout and residency" → "Prefix sharing is the
concurrency-scaled prize"](MEMORY_ARCHITECTURE.md); this note does not restate
it, per the #781 anti-staleness rule.

Device under test: **NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, Ada), 8 GiB**,
WDDM. Granule `CU_MEM_ALLOC_GRANULARITY_MINIMUM == RECOMMENDED == 2 MiB` (#776).

## TL;DR — the recommendation

**The primitive is sound enough to build KV integration on.** The
production-shaped probe clarifies that its protection is fail-stop, not a
recoverable kernel-error boundary:

1. **N-way sharing works**, including under captured-graph replay. One physical
   granule mapped into 8 sequences' reservations is read identically by all 8,
   and a captured device-to-device copy (the decode hot path) reading each
   sequence's shared-prefix VA replays correctly. Physical cost is **1 granule,
   not N** (a write through any alias is visible at every other alias — one page,
   not N copies).
2. **The real ledger charges it once and keeps it alive correctly.** Against the
   production `CudaVmmAllocator` + `LedgerGovernor` (no mock), 8 tenants of one
   prefix granule are charged **one** granule; each additional sharer needs **0**
   incremental owned bytes; the granule stays mapped until the **last** sharer
   leaves; teardown returns the ledger to zero. Admission arithmetic: **admitting
   the Nth request costs only its private bytes.**
3. **Kernel write protection is fail-stop.** Copy-engine and memset writes into a
   `PROT_READ` shared prefix return `CUDA_ERROR_INVALID_VALUE` without poisoning
   the context, but a real kernel `st.global` returns
   `CUDA_ERROR_ILLEGAL_ADDRESS` at synchronization and leaves the context
   unusable. This is normal CUDA illegal-address behavior: protection prevents
   silent cross-request KV corruption, but it cannot recover a broken kernel.
4. **It composes with the #759 dummy page.** A read-only shared prefix head, a
   read/write private live granule, and a read-only shared dummy tail coexist in
   **one** reservation with three distinct `cuMemSetAccess` postures. Reads
   succeed everywhere; copy-engine writes are rejected without corruption. A
   kernel store still has the sticky behavior above.
5. **Copy-on-write at the boundary is a one-time burst, not a per-token tax.**
   Diverging one sequence (copy 2 MiB + remap) measured **~5.5 ms pooled**
   (production #740 retained handle, no `cuMemCreate`) and **~9.8 ms cold** on
   this WDDM box — roughly **22% / 39%** of an assumed 25 ms decode step, paid
   **once** when a sequence first writes past the shared prefix.

**Smallest next increment (now landed):** not a kernel change and not COW. It
was an **explicit pinned-prefix API on the token-major reservation model**
(§ Design) — a `commit_shared_prefix(handle, range)` entry point on the arena
that maps an existing physical handle into a new sequence's reservation at
offset 0 and takes a granule-ref on it, charged on the owned axis. This entry
point is now **implemented and GPU-tested** (§ Landed). Everything downstream
(the attention kernel, the flat VA) is unchanged. Detection by hashing and COW
come
later; the explicit case is unambiguous and covers the highest-value scenario
(a pinned system prompt / RAG document).

The tests are self-contained GPU probes, one question per binary so a poisoned
context or a process-global counter cannot contaminate another:

| Test binary | Question |
|---|---|
| `vmm_prefix_share_gpu.rs` | Q1 (N-way + captured replay), Q5 (COW cost) |
| `vmm_prefix_share_ledger_gpu.rs` | Q2 (real allocator/ledger charge-once) |
| `vmm_prefix_share_write_protect_gpu.rs` | Q3 (write-protect soundness + stickiness) |
| `vmm_prefix_share_compose_gpu.rs` | Q4 (compose with #759 dummy page) |
| `vmm_commit_shared_prefix_gpu.rs` | The landed entry point (cross-reservation multi-map on the real allocator) |

---

## The five questions, measured

### Q1 — does N-way sharing work, including under captured-graph replay?

`vmm_prefix_share_gpu.rs::one_prefix_granule_shared_across_n_sequences_reads_identically_under_replay`.
#727 proved 1 handle at 2 VAs. This generalizes it to **N = 8**: one
`cuMemCreate`d prefix granule mapped at offset 0 of eight separate reservations,
each also holding a private tail granule. Written **once** through sequence 0's
alias, the prefix reads back identically at all eight VAs; each private tail is
its own. A captured `cuMemcpyDtoDAsync_v2` per sequence — reading its shared
prefix VA into a private sink, i.e. the decode read in miniature — replays
without fault and reproduces the shared bytes. A rewrite through sequence 7's
alias is immediately visible at sequence 0: **one physical page, not eight
copies**. Physical accounting is by created handles (each `cuMemCreate` = one
2 MiB granule): the shared prefix is **1 granule** versus **8** for per-sequence
copies.

### Q2 — does the real ledger charge once and keep it alive correctly?

`vmm_prefix_share_ledger_gpu.rs::shared_prefix_granule_charged_once_alive_until_last_and_nth_sharer_is_free`.
Against the production `CudaVmmAllocator` + `LedgerGovernor` (no mock), eight
sub-granule tenants of one 2 MiB prefix granule drive the #740/#745 granule
refcount 0->1->…->8 and back. Measured:

- The first tenant charges the governor **exactly one granule**; `mapped_bytes`
  equals that charge (one physical handle).
- Every additional sharer's `incremental_owned_bytes_for_span` is **0**, and the
  governor charge **does not rise** — the admission cost of the Nth request is
  only its private bytes.
- Dropping seven of eight sharers keeps the granule charged and mapped
  throughout (**alive until last**). Dropping the last drops `mapped_bytes` to 0
  (mapping refcount 1->0); the physical handle is then **retained by the pool**
  for reuse (`pooled_unmapped_bytes == granule`), not leaked. Teardown releases
  it and the ledger returns to zero with `creates == releases`.

Honest caveat: the current allocator expresses "N tenants of one granule" as
sub-granule allocations packed into a granule, which is the exact 0->1 / 1->0
attribution a cross-reservation prefix multi-map will use, but it is **not** the
cross-reservation multi-map itself — that API is the next increment. Q1 proves
the physical multi-map at the driver level; Q2 proves the **attribution** the
allocator entry point must reuse.

### Q3 — is write protection sound and non-sticky? (potential kill finding)

`vmm_prefix_share_write_protect_gpu.rs::shared_prefix_kernel_write_fault_is_sticky`,
in its own binary. One prefix handle is mapped into a `victim` VA (downgraded to
`PROT_READ`) and an `other` VA (a second sharer). Both a synchronous
`cuMemcpyHtoD_v2` and an asynchronous `cuMemsetD8Async` into the read-only prefix
**fault** (`CUDA_ERROR_INVALID_VALUE`; async surfaces at the stream sync). After
**each** fault, a `cuCtxSynchronize` plus a fresh independent
allocate/write/read succeeds — the fault is **non-sticky**, the context is not
poisoned — and the `other` sharer still reads the original prefix, so the
rejected copy-engine write **did not corrupt** the shared page.

That result does not generalize to production kernels. A driver-JITed probe
first reads both aliases with `ld.global`, then issues `st.global` into the
read-only victim. The launch succeeds, synchronization returns
`CUDA_ERROR_ILLEGAL_ADDRESS`, and a fresh-context-health operation returns the
same sticky error. This establishes the production contract: correct kernels
read the shared prefix; an invalid store fails loudly and requires context
recreation, just like any other CUDA illegal address.

### Q4 — does it compose with the #759 dummy page?

`vmm_prefix_share_compose_gpu.rs::shared_prefix_and_dummy_tail_coexist_in_one_reservation`,
own binary. One reservation carries three regions with three access postures:
read-only shared prefix head (aliased to a second reservation), read/write
private live granule, and a read-only shared **dummy** tail (one dummy handle at
every uncommitted tail granule, the #759 primitive). Measured: a full-padded
read across the whole reservation succeeds (prefix + live + dummy tail, no #772
tail fault); the private region accepts writes while the shared regions stay
protected; writes into **either** read-only region fault non-stickily and leave
both shared pages intact. The two mechanisms that both want `cuMemSetAccess` on
the same address space **coexist** — no conflict on this hardware.

### Q5 — copy-on-write at the boundary

`vmm_prefix_share_gpu.rs::copy_on_write_at_the_shared_boundary_granule_cost_and_isolation`.
Sharing ends at a granule boundary, so a diverging sequence must obtain a private
copy of the boundary granule (copy 2 MiB, remap) before writing. Measured in two
regimes:

| Regime | Per divergence | % of assumed 25 ms decode step |
|---|---:|---:|
| Cold (`cuMemCreate` each time) | ~9.8 ms | ~39% |
| Pooled (#740 retained handle, no `cuMemCreate`) | ~5.5 ms | ~22% |

`cuMemCreate` on WDDM is the dominant term, which is exactly why the #740 pool
exists — the **pooled** number is the production cost. Isolation is verified: the
diverging sequence mutates only its private copy; the still-shared original is
unchanged for the other sharers. These millisecond figures are WDDM-dominated
and vary run-to-run; the durable conclusion is that COW is a **one-time** burst
paid at first divergence, not a per-token cost, and it never happens for the
pinned-prefix scenario where the prefix is read-only for the request's whole
life.

Constraints honoured (#727/#777): the divergence remap synchronises the stream
before unmapping (never unmap under a replay in flight); captures only **read**
already-mapped stable VAs (never map or grow inside a captured region); no
assertion runs inside any `Drop`; physical bytes are reported from handle counts
/ the governor's owned axis, never nominal content bytes.

---

## Concurrency-scaled saving

KV geometry (fp16, 2 MiB granule): **qwen14b** = 48 layers × 2 × 8 KV heads ×
128 head-dim × 2 B = **196,608 B/token** (192 KiB, 10.67 tokens/granule);
**qwen2.5-0.5b** = 24 × 2 × 2 × 64 × 2 = **12,288 B/token** (12 KiB, 170.67
tokens/granule).

Under **token-major** the whole-model prefix is one contiguous range, so a shared
prefix costs `ceil(P × bytes_per_token / granule)` granules **once**, and sharing
it across `C` concurrent requests removes `C − 1` duplicate copies. Physical bytes
removed = `(C − 1) × ceil(P × bytes/token / granule) × granule` (reported on the
committed **physical** axis, granule-rounded — never nominal content bytes).

Per-copy prefix residency (granule-rounded):

| Prefix P | qwen14b | qwen2.5-0.5b |
|---:|---:|---:|
| 512 tok | 96 MiB (48 granules) | 6 MiB (3) |
| 2,048 tok | 384 MiB (192) | 24 MiB (12) |
| 8,192 tok | 1,536 MiB (768) | 96 MiB (48) |

**Physical KV removed by sharing** = `(C − 1) ×` the per-copy figure:

qwen14b:

| Prefix \ C | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|
| 512 tok | 96 MiB | 288 MiB | 672 MiB | 1,440 MiB |
| 2,048 tok | 384 MiB | 1,152 MiB | **2,688 MiB (≈2.6 GiB)** | 5,760 MiB |
| 8,192 tok | 1,536 MiB | 4,608 MiB | 10,752 MiB | 23,040 MiB |

qwen2.5-0.5b:

| Prefix \ C | 2 | 4 | 8 | 16 |
|---:|---:|---:|---:|---:|
| 512 tok | 6 MiB | 18 MiB | 42 MiB | 90 MiB |
| 2,048 tok | 24 MiB | 72 MiB | 168 MiB | 360 MiB |
| 8,192 tok | 96 MiB | 288 MiB | 672 MiB | 1,440 MiB |

The 2,048-token / 8-way qwen14b cell is **2,688 MiB (≈2.6 GiB)**, matching the
`MEMORY_ARCHITECTURE.md` figure of ~2.56 GiB for a 2,000-token prompt (the small
difference is 2,048 vs 2,000 tokens and granule rounding). The saving **scales
with concurrency** (the `C` axis serving cares about), whereas the residency
floor is a per-sequence constant that only amortizes with context length.

### Capacity consequence — the number that matters

On a **fixed** KV budget `B`, without sharing each request costs
`prefix + private`; with sharing the prefix is paid **once** and every further
request costs only its **private** bytes. So the admissible concurrency rises
from

```
N_no_sharing = floor( B / (prefix + private) )
```

to

```
N_sharing    = 1 + floor( (B − prefix) / private )
```

Worked example (qwen14b, 2,048-token shared prefix = 384 MiB, each request adds a
private 2,048-token continuation = a further 384 MiB, so `prefix + private` =
768 MiB, `private` = 384 MiB):

| KV budget `B` | N without sharing | N with sharing | Extra requests |
|---:|---:|---:|---:|
| 1.5 GiB | 2 | 4 | +2 |
| 3.0 GiB | 4 | 8 | +4 |
| 6.0 GiB | 8 | 16 | +8 |

Sharing roughly **doubles** admissible concurrency for a 50/50 prefix/private
split, and the multiple grows as the shared prefix grows relative to the private
continuation — a longer system prompt or RAG document makes sharing **more**
valuable, not less. This reproduces the issue's "difference between 2 and 8
concurrent requests" illustration: at the small-budget end of an 8–16 GiB card, a
large shared prefix is precisely what converts a 2-request ceiling into an
8-request one.

---

## Design answers (not implemented this round)

### Detection — recommend the explicit pinned-prefix case first

Two options identify a shared prefix: (a) exact token-prefix match via a rolling
hash of the prompt tokens, keyed to a shared-prefix registry; (b) an explicit API
for a pinned system prompt / tool schema / RAG document. **Recommend starting
with the explicit case.** It is unambiguous (no hash-collision or
partial-match edge cases, no per-token bookkeeping on the admission path) and it
already covers the highest-value scenario — multi-tenant serving of one pinned
system prompt to many concurrent requests. Hash-based automatic detection is a
later layer on the same primitive once the explicit path is proven end-to-end.

### Lifetime — the union, not any single request

The shared physical handle must outlive **every** sharer, so its lifetime is the
**union** of all sharers' lifetimes, not tied to any one request. This is exactly
what the #740 pool's granule refcount already expresses (Q2): the handle is
released only on the 1->0 transition. A pinned prefix additionally has an
owner-held reference so it survives even when momentarily zero sharers are live
(e.g. between requests), which the explicit API models naturally as a registry
entry holding one reference.

### Eviction interaction — one action, all sharers

Because the granules are physically shared, **evicting or migrating a shared
prefix affects every sharer at once.** The eviction policy must treat a shared
prefix as a single unit whose "temperature" is the aggregate of all sharers, and
must never migrate it while any sharer may issue a decode read against it (the
#727 constraint: no unmap under a replay in flight). Migration would be a
coordinated remap of the shared handle at every sharer's VA — cheap in bytes
(one physical copy) but requiring a barrier across all sharers' streams. For the
first increment, a pinned prefix is simply **non-evictable** for its pinned
lifetime, sidestepping this entirely.

### Layout dependency — impractical head-major, cleanest token-major

Sharing requires the prefix to be **contiguous** so it maps as whole granules:

- **Head-major (BNSH):** the prefix is `layers × 2 × kv_heads` scattered
  fragments (768 on qwen14b), each far under a granule — sharing would need 768
  sub-granule multi-maps per sequence and cannot pool sub-granule live bytes.
  **Impractical.**
- **Seq-major (BSNH):** the prefix is `layers × 2` contiguous layer-side ranges
  (96 on qwen14b) — **workable** as 96 multi-maps per sequence.
- **Token-major across all layers:** the prefix is **one** contiguous range
  covering every layer — a **single** physical-handle multi-map per sequence.
  **Cleanest.**

This is a further, independent argument for the token-major direction in
#783/#787: the concurrency-scaled prize is only cleanly reachable there. On this
2 MiB-granule CUDA device, prefix sharing should be built on the token-major
reservation model.

---

## Landed this round — the `commit_shared_prefix` entry point (#777)

The cross-reservation multi-map that Q1/Q2 said "remains" is now implemented on
the real allocator, as an allocator primitive only (no KV path, no engine, no
detection, no COW):

1. **`CudaVmmAllocator::create_shared_prefix(bytes)`** reserves a private window
   and acquires `granule_count` physical handles from the existing #740
   authority-scoped pool, maps them read/write for the owner to fill, and charges
   them **once** on the owned axis (`governor.used(Tier::Device)`). It returns a
   `SharedPrefix` handle whose lifetime holds one owner-reference — the
   registry-entry model from § Lifetime, so the prefix survives momentarily-zero
   sharers.
2. **`CudaVmmAllocator::commit_shared_prefix(&prefix, ptr, allocation_bytes,
   byte_offset)`** maps those already-owned physical handles into a *different*
   sequence's reservation at `byte_offset`, **`PROT_READ`**, and takes a
   pool-level shared-map ref. It goes
   through the existing #740 pool and `carve()` suballocation on the existing
   global granule-ref **0->1 / 1->0** attribution — **no second allocator, no
   per-sequence physical reservation** (which is what made #733 net-negative).
   Two refcounts compose: the arena's per-granule allocation refcount and the
   pool's cross-reservation `shared[handle]` count; the physical handle returns
   to the pool only when the **last** mapping (owner or sharer) unmaps — lifetime
   is the **union** of sharers, structurally (§ Lifetime).
3. **`incremental_owned_bytes_for_shared_prefix(&prefix)` returns 0** — the Nth
   sharer bills only its private bytes, Q2's admission arithmetic made explicit
   on the entry point (#745).
4. Failure modes error rather than mis-map: a non-granule-aligned target VA, a
   granularity mismatch, a prefix that overflows or exceeds the target
   allocation, a pointer outside the arena or not a live allocation, a target
   granule that is **already committed** (never overlay live KV), and — crucially
   — **any attempt to map while a graph capture is open** (`enter_graph_capture`
   raises a depth counter that `commit_shared_prefix` refuses under; this makes
   "never `cuMemMap` inside a captured region" an *enforced* rule, not a comment).
   Every failure path rolls back cleanly (unmap + pool return), leaving the
   region uncommitted.
5. Reported bytes are **physical** (`committed_physical_bytes`, `mapped_bytes`),
   never nominal content bytes; no `assert!` runs inside any `Drop`.
6. A pooled CUDA allocator exposes the optional `SharedMapping` capability;
   detached allocators without the physical-handle pool report capability
   absence. The capability's protection contract is fail-stop as established by
   Q3.

`vmm_commit_shared_prefix_gpu.rs` proves this against the **real**
`CudaVmmAllocator` + `LedgerGovernor` (no mock), one concern per test:

- **N sequences share one pinned prefix, charged once, alive until the last** —
  8 sequences map one 2 MiB prefix; owned bytes are 18 MiB (1 prefix + 8 private
  tails), not 32 MiB; freeing 7 sharers keeps the survivor reading the shared
  page; teardown returns `total_owned_bytes`, `governor.used`, and
  `creates - releases` all to zero.
- **Admitting a sharer costs only private bytes** — asserted against the real
  ledger: mapping the shared prefix moves the governor by 0; only committing the
  sequence's own private granule charges it.
- **Copy-engine writes fault without corruption** — synchronous and asynchronous
  copy paths are non-sticky. The isolated Q3 kernel-store probe separately
  records the sticky kill finding that keeps the production capability disabled.
- **Unsupported requests error rather than mis-map** — misaligned offset,
  over-long prefix, mapping over a committed granule, and mapping under an open
  capture all return `Err` and leave the region clean.

All device operations in the new suite route through synchronous copies or a
single created stream (the #797 harness hazard — a legacy-default-stream readback
racing a non-blocking-stream memset with no CUDA error — is avoided), and each
test constructs its own allocator so no alphabetically-earlier sibling warms the
context.

**What is now possible vs. what still needs KV integration.** The allocator can
now physically share a pinned prefix across independent sequence reservations,
charged once, read-only, with the correct union lifetime and admission
arithmetic — the capacity mechanism (roughly doubled admissible concurrency for a
large shared system prompt) exists at the allocator layer. What it does **not**
yet do: nothing constructs a sequence's KV as `shared prefix + private tail` in
the engine, nothing detects that two requests share a prefix, and nothing handles
divergence past the shared region. Those are the increments below.

## Smallest next increment

1. **KV-path integration (first production consumer — landed for seq-major,
   #777).** A pooled `CudaVmmAllocator` exposes the independent `SharedMapping`
   capability from the selected `DeviceAllocator`; detached allocators report
   capability absence. The **seq-major** fused fp16 GQA decode kernel is the
   end-to-end consumer: the test pins a token prefix through the capability,
   maps it into a second sequence, and runs the real kernel. In
   `crates/onnx-runtime-ep-cuda/tests/gqa_shared_prefix_parity_gpu.rs`, two
   sequences sharing one pinned seq-major prefix (`layers × 2` contiguous ranges)
   produce **byte-identical** output to two independent sequences. Measured
   (KV_HEADS=8, HEAD_DIM=128, f16, 1024-token prefix + 1024-token private tail):
   independent = 8 granules (16,777,216 B), shared = 6 granules (12,582,912 B);
   prefix charged **once** (`incremental_owned_bytes_for_shared_prefix` = 0),
   second sharer's admission = **private bytes only** (4,194,304 B), saving
   `(C−1)×(K_prefix+V_prefix)` = 2 granules. Shared-prefix mappings are read-only;
   an invalid kernel store is a fail-stop CUDA error rather than a recoverable
   operation.

   Seq-major accepts `layers × 2` multi-maps per sequence rather than the single
   multi-map that only token-major (#783/#787) achieves; token-major is measured
   but not built. **Structural blocker for auto engine use:** the engine
   generation loop cannot call this seam yet because
   `persistent_state_shapes` in `native_decode/cuda.rs` builds a hard-coded BNSH
   physical KV shape and no model declares a seq-major end-to-end fixed-stride
   physical shape (#794 showed seq-major changed only kernel indexing, not commit
   geometry). A BNSH/seq-major fixed-stride physical-shape build is the
   prerequisite for wiring the primitive into the automatic decode path.
2. Then: hash-based automatic detection (layer 2) — a rolling-hash prefix
   registry keyed to the same handle, once the explicit path is proven
   end-to-end.
3. Then: boundary copy-on-write for divergence past a shared prefix (layer 3, Q5
   cost already characterized at ~5.5 ms pooled / ~9.8 ms cold, once at
   divergence).

Nothing in the landed primitive touches the attention kernel or the flat
per-sequence VA — the whole point is that the kernel is unchanged and learns
nothing.

---

## Generalising shareability: the arithmetic predicate (#777, this round)

The framing above (and my earlier writing) said prefix sharing "requires
seq-major" and is "impractical under head-major". That is a **granule-relative**
claim stated absolutely, and it is wrong. The owner's correction is right:

> 前缀共享应该是一个通用 所有情况都能用的功能 ... ort backend我也想做前缀共享呀

Whether a shared prefix can be physically shared is **arithmetic**, not a
property of a named layout. Sharing maps whole granules, so a granule that holds
both shared prefix and private continuation cannot be shared:

```
fragment_bytes                  = prefix_len × (contiguous bytes per fragment in that layout)
shareable                       = fragment_bytes ≥ granule
shareable_granules_per_fragment = floor(fragment_bytes / granule)
multi_map_ops                   = fragments × shareable_granules_per_fragment
wasted_boundary_bytes/sequence  = fragments × (fragment_bytes mod granule)   [the straddling
                                  granule is private per sequence]
```

Layout sets `fragment_bytes` and the **cost** (fragments and multi-map ops), not
the **possibility**. The genuine requirements are (a) the KV buffer is
**VMM-backed** and (b) `fragment_bytes ≥ granule` for the layout on the platform.
Neither says "seq-major".

This is now a real, tested function rather than a prose rule:
`onnx_runtime_memory_governor::shareability::evaluate_prefix_shareability`
(`crates/onnx-runtime-memory-governor/src/shareability.rs`), over a
`ModelKvGeometry` and a `KvFragmentation` descriptor (`head_major_bnsh`,
`seq_major_bsnh`, `token_major`, or an arbitrary stride arrangement per #783).
Its `PrefixShareability` result carries `shareable`, `fragments`,
`fragment_bytes`, `shareable_granules_per_fragment`, `multi_map_ops`,
`wasted_boundary_bytes_per_sequence`, and a `refusal_reason()` a KV path uses to
**refuse with a reason** instead of silently making N private copies. It is the
authority that replaces any "is this seq-major" check;
`gqa_shared_prefix_parity_gpu.rs` now consults it to admit the share before
mapping.

### A correction to the issue's worked table

The 2 MiB column of the qwen14b table in the issue thread reads **"seq-major: 2
granules each — shareable"**. That is `ceil` (granules the prefix *touches* /
residency), not `floor` (whole granules that fall entirely inside the shared
prefix). The shareable count is `floor(4_096_000 / 2_097_152) = 1`, giving
`96 × 1 = 96` multi-map ops — which is exactly the "96 multi-maps per sequence"
the design section states for seq-major. So the correct cell is **1 granule per
fragment, 96 ops**, and the rest of the table's *possibility* verdicts are
unchanged. This is a slip in one cell (ceil vs floor), not a missing term in the
model; the predicate uses `floor`, which is the one that counts shareable
granules. Every other verified cell (head-major not shareable at 2 MiB / ~7 at
64 KiB; token-major ~187 at 2 MiB) matches.

### Shareability across layout × granule × prefix length

`share, N ops` means shareable with `N` total multi-map operations per sequence;
`no` means `fragment_bytes < granule` (not shareable at that granule). Verified by
`evaluate_prefix_shareability` and the module tests.

#### qwen14b (48 layers, 8 kv_heads, head_dim 128, fp16)

| Layout | Granule | 512 tok | 2,048 tok | 8,192 tok |
|---|---|---|---|---|
| head-major BNSH | 2 MiB | no | no | share, 768 ops |
| head-major BNSH | 64 KiB | share, 1,536 ops | share, 6,144 ops | share, 24,576 ops |
| head-major BNSH | 4 KiB | share, 24,576 ops | share, 98,304 ops | share, 393,216 ops |
| seq-major BSNH | 2 MiB | no | share, 192 ops | share, 768 ops |
| seq-major BSNH | 64 KiB | share, 1,536 ops | share, 6,144 ops | share, 24,576 ops |
| seq-major BSNH | 4 KiB | share, 24,576 ops | share, 98,304 ops | share, 393,216 ops |
| token-major | 2 MiB | share, 48 ops | share, 192 ops | share, 768 ops |
| token-major | 64 KiB | share, 1,536 ops | share, 6,144 ops | share, 24,576 ops |
| token-major | 4 KiB | share, 24,576 ops | share, 98,304 ops | share, 393,216 ops |

The **head-major @ 2 MiB @ 8,192 tokens** cell is the headline: head-major *does*
become shareable when the arithmetic says so. The threshold is exactly
`granule / (head_dim × dtype) = 2_097_152 / 256 = 8,192` tokens — realistic for
RAG and long system prompts, not hypothetical.

#### qwen2.5-0.5b (24 layers, 2 kv_heads, head_dim 64, fp16)

| Layout | Granule | 512 tok | 2,048 tok | 8,192 tok |
|---|---|---|---|---|
| head-major BNSH | 2 MiB | no | no | no |
| head-major BNSH | 64 KiB | share, 96 ops | share, 384 ops | share, 1,536 ops |
| head-major BNSH | 4 KiB | share, 1,536 ops | share, 6,144 ops | share, 24,576 ops |
| seq-major BSNH | 2 MiB | no | no | share, 48 ops |
| seq-major BSNH | 64 KiB | share, 96 ops | share, 384 ops | share, 1,536 ops |
| seq-major BSNH | 4 KiB | share, 1,536 ops | share, 6,144 ops | share, 24,576 ops |
| token-major | 2 MiB | share, 3 ops | share, 12 ops | share, 48 ops |
| token-major | 64 KiB | share, 96 ops | share, 384 ops | share, 1,536 ops |
| token-major | 4 KiB | share, 1,536 ops | share, 6,144 ops | share, 24,576 ops |

On the small model the per-fragment bytes are `kv_heads`/`head_dim` smaller, so at
2 MiB only token-major shares below 8,192 tokens — again the arithmetic, not a
layout preference, saying so. At 64 KiB and 4 KiB every layout shares.

### Cross-checking whether a term is missing

Asked to verify the arithmetic independently: the invariant
`fragments × contiguous_bytes_per_token = layers × 2 × kv_heads × head_dim ×
dtype` (the whole-model per-token byte count) holds for all three layouts, so the
fragment descriptors partition the same bytes — a term-conservation check the
module test `total_bytes_per_token_is_layout_invariant` encodes. The only
divergence from the issue's numbers is the ceil-vs-floor slip noted above. One
honest caveat the predicate makes explicit but does not resolve: it counts
*whole* shareable granules and treats the straddling boundary granule as private;
it does **not** model sharing at a sub-granule unit (which VMM cannot do) nor the
CPU/`mmap` case where sharing at page rather than granule granularity could share
the boundary page too. Those are platform capabilities the caller supplies as
`granule`, not terms missing from the formula.

## The ORT backend — definitive answer (deliverable #2)

The question is **not** "can ORT share prefixes" — the `commit_shared_prefix`
primitive (#803) and the `DeviceAllocator` seam (#809) are layout-agnostic and
apply to any VMM-backed buffer. The question is **"does ORT's KV allocation route
through an allocator we can back with the VMM arena?"** Answer, with citations
into `crates/onnx-genai-ort/`:

### Where ORT KV comes from today — two paths, both bypass a VMM arena

1. **Dynamic decode (`DecodeKvMode::ZeroCopyRebind`,
   `src/decode/mod.rs:66`).** ORT allocates the `present.*` outputs itself and we
   rebind them as next step's `past_key_values.*` — "No Rust-side KV copy is
   performed" (`src/decode/mod.rs:67`). These outputs are allocated by the EP's
   own output allocator (its BFC arena); no allocator parameter of ours is
   involved at all. **This path cannot be routed through a provided allocator.**

2. **Shared-buffer decode (`DecodeKvMode::SharedBuffer`,
   `src/decode/shared_batch.rs`, `src/decode/dynamic.rs:413`).** Here we *do*
   pre-allocate one max-length KV `OrtValue` per tensor and bind it as both past
   input and present output. But it is created with
   `Value::empty_in(&shape, dtype, device_allocator)`
   (`src/decode/shared_batch.rs:360`), which calls `CreateTensorAsOrtValue`
   (`src/value.rs:141`) with an allocator obtained from
   `Session::device_kv_allocator()` (`src/session/mod.rs:607`). That allocator is
   built by `Allocator::for_session_device` → **`CreateAllocator`**
   (`src/allocator.rs:239`), which the doc comment states plainly **"wraps the
   session's internal EP allocator"** (`src/allocator.rs:232`). So `CreateAllocator`
   lets us pick the *device* the tensor lands on, **not the allocation
   implementation** — the bytes still come from ORT's internal CUDA allocator, not
   a VMM arena we control.

The one seam that lets us supply the *implementation* is **`RegisterAllocator`**
(`src/governed_allocator.rs:738`, `register_governed_allocator`), which installs
our `OrtAllocator` vtable on the environment for sessions created with
`session.use_env_allocators` (`src/session/options.rs:63`). But this confirms the
`allocator.rs:65` hint rather than defeating it: ORT **reserves the arena kind
for its own internal arenas** — a registered allocator must describe itself as a
non-arena `OrtDeviceAllocator` (`src/governed_allocator.rs:746`), and ORT then
fronts it with its **own BFC arena**, calling our `Alloc` only for coarse chunks
and sub-allocating each KV tensor *inside* a chunk. So even under
`RegisterAllocator`, an individual `past_key_values`/`present` tensor is a
byte-range inside an arena chunk at an arena-chosen offset — **not a
granule-aligned VMM reservation whose base and granules we own**. `commit_shared_prefix`
needs the latter (it maps physical handles into a specific reservation at a
granule-aligned `byte_offset`, `PROT_READ`, without disturbing neighbours), so it
cannot apply to an ORT-arena sub-allocation.

**Definitive negative for the automatic paths:** as wired today, neither the
dynamic nor the shared-buffer KV path yields a KV buffer we can back with the VMM
arena. `CreateAllocator` selects device, not implementation; `RegisterAllocator`
gives us coarse chunks that ORT re-carves. `commit_shared_prefix` has nothing to
attach to.

### The one affirmative route, and its exact scope

There **is** a concrete path, and it is worth stating precisely so it is not
mistaken for "ORT just works". ORT exposes `CreateTensorWithDataAsOrtValue`
(`src/value.rs:1103`, wrapped by `create_tensor_with_data_in`), whose doc says it
exists "so a caller can hand ORT memory it allocated itself" (`src/value.rs:1099`)
with an explicit device `MemoryInfo`. So a KV buffer allocated on **our VMM
arena** can be presented to ORT as an `OrtValue` over external device memory and
bound through the existing `IoBinding` (`src/decode/binding.rs`). This only works
in the **shared-buffer / `past_present_share_buffer`** mode, because that mode
already uses one fixed max-length buffer bound as both past and present
(`src/decode/mod.rs:70`) — a stable reservation whose prefix region can be pinned
read-only for the sharers' lifetime. The dynamic mode cannot, because ORT owns
the `present.*` allocation.

Two consequences to record before anyone builds it:

- **ORT's KV is head-major BNSH.** Per `schema/inference_metadata.schema.json`
  and the ORT GQA contract, past/present is `[batch, kv_heads, seq, head_dim]` on
  every ORT dispatch path (Flash, cuDNN SDPA, memory-efficient, XQA). By the
  predicate above, head-major at a 2 MiB CUDA granule is shareable only for
  prefixes **≥ 8,192 tokens** (or on a finer-granule EP). So an ORT prefix-share
  is real but bounded: it pays for long pinned system prompts / RAG documents, and
  the seam correctly **refuses** (with reason) for shorter prefixes rather than
  mis-mapping.
- **It requires bypassing ORT's arena for the KV OrtValue**, i.e. constructing
  the shared-buffer KV via `CreateTensorWithDataAsOrtValue` over VMM memory
  instead of `Value::empty_in`/`CreateTensorAsOrtValue`, and keeping that VMM
  reservation alive for the OrtValue's whole life (ORT will not free external
  memory). This is a real change to `shared_batch.rs`'s allocation call, scoped
  but not trivial, and explicitly **out of scope for this PR**.

**Bottom line:** ORT prefix sharing is not "free via the existing allocator". The
`CreateAllocator` KV path selects device, not implementation, and the dynamic path
bypasses our allocators entirely; `commit_shared_prefix` cannot attach to an
ORT-arena sub-allocation. The viable route is to allocate the shared-buffer KV on
the VMM arena and bind it as external device memory via
`CreateTensorWithDataAsOrtValue` + `IoBinding`, at which point the same
`create_shared_prefix`/`commit_shared_prefix`/`DeviceAllocator` machinery applies
unchanged — bounded by the head-major arithmetic (≥ 8,192-token prefixes at
2 MiB). That is the scoped follow-up, not this PR.
