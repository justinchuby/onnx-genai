# VMM page-level prefix sharing: viability probe & design (#777)

Status: **isolating GPU probe + written design. No KV integration, no engine or
production-kernel change this round** (per scope). The primitive is proven sound
at the CUDA driver level and against the real #740 allocator/ledger; this
document records the measured answers, the design decisions that do **not** get
implemented yet, and the smallest next increment.

References **#777** (this issue), **#727** (one handle at two VAs, captured-graph
safe), **#740/#745** (authority-scoped physical-handle pool, `carve()`
suballocation, 0->1 / 1->0 granule-ref attribution), **#759** (dummy-page
tail), **#776** (2 MiB granule), **#787** (token-major floor), **#750**
(multi-request serving), **#783** (KV as a stride descriptor). Background — the
authoritative design line — lives in
[`MEMORY_ARCHITECTURE.md` → "KV layout and residency" → "Prefix sharing is the
concurrency-scaled prize"](./MEMORY_ARCHITECTURE.md); this note does not restate
it, per the #781 anti-staleness rule.

Device under test: **NVIDIA GeForce RTX 4060 Laptop GPU (sm_89, Ada), 8 GiB**,
WDDM. Granule `CU_MEM_ALLOC_GRANULARITY_MINIMUM == RECOMMENDED == 2 MiB` (#776).

## TL;DR — the recommendation

**The primitive is sound enough to build KV integration on.** All five
questions cleared with no kill finding:

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
3. **Write protection is sound and — critically — non-sticky.** A synchronous
   and an asynchronous write into a `PROT_READ` shared prefix both fault
   (`CUDA_ERROR_INVALID_VALUE`); the context **survives both** and later work
   succeeds; the other sharer's copy is **uncorrupted**. This was the potential
   kill finding — it did **not** fire.
4. **It composes with the #759 dummy page.** A read-only shared prefix head, a
   read/write private live granule, and a read-only shared dummy tail coexist in
   **one** reservation with three distinct `cuMemSetAccess` postures. Reads
   succeed everywhere; only the private region accepts writes; both shared pages
   survive rejected writes.
5. **Copy-on-write at the boundary is a one-time burst, not a per-token tax.**
   Diverging one sequence (copy 2 MiB + remap) measured **~5.5 ms pooled**
   (production #740 retained handle, no `cuMemCreate`) and **~9.8 ms cold** on
   this WDDM box — roughly **22% / 39%** of an assumed 25 ms decode step, paid
   **once** when a sequence first writes past the shared prefix.

**Smallest next increment:** not a kernel change and not COW. It is an
**explicit pinned-prefix API on the token-major reservation model** (§ Design)
— a `commit_shared_prefix(handle, range)` entry point on the arena that maps an
existing physical handle into a new sequence's reservation at offset 0 and takes
a granule-ref on it, charged on the owned axis. Everything downstream (the
attention kernel, the flat VA) is unchanged. Detection by hashing and COW come
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

`vmm_prefix_share_write_protect_gpu.rs::shared_prefix_write_fault_is_loud_non_sticky_and_non_corrupting`,
in its own binary. One prefix handle is mapped into a `victim` VA (downgraded to
`PROT_READ`) and an `other` VA (a second sharer). Both a synchronous
`cuMemcpyHtoD_v2` and an asynchronous `cuMemsetD8Async` into the read-only prefix
**fault** (`CUDA_ERROR_INVALID_VALUE`; async surfaces at the stream sync). After
**each** fault, a `cuCtxSynchronize` plus a fresh independent
allocate/write/read succeeds — the fault is **non-sticky**, the context is not
poisoned — and the `other` sharer still reads the original prefix, so the
rejected write **did not corrupt** the shared page. A sticky fault here would
have made write-protection worse than the corruption it prevents; it **did not
fire**.

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

## Smallest next increment

1. Add an explicit `commit_shared_prefix(handle, offset, len)` entry point to the
   token-major arena that maps an existing pooled physical handle into a new
   sequence's reservation at offset 0 and takes a granule-ref on it (charged on
   the owned axis, so the Nth sharer bills 0 — Q2's arithmetic). No detection, no
   COW, no eviction: a **pinned** prefix, read-only for its lifetime.
2. Validate it end-to-end on the existing output-level parity oracle: two
   concurrent sequences sharing a pinned prefix must produce identical logits to
   two independent sequences, at the measured physical saving.
3. Only after that: hash-based automatic detection (layer 2) and boundary COW
   for divergence past a shared prefix (layer 3, Q5 cost already characterized).

Nothing here touches the attention kernel or the flat per-sequence VA — the whole
point of the primitive is that the kernel is unchanged and learns nothing.
