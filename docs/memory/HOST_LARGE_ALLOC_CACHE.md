# Host large-allocation cache

**Status:** implemented, default-on, `crates/onnx-runtime-memory-governor/src/large_alloc_cache.rs`

## The failure this removes

`HostAllocator` is a thin wrapper over `std::alloc`, and its own doc comment
records an earlier measured result: an arena layered over the system allocator
was *slower* than the system allocator. That result is correct **for small
allocations** — glibc serves those from per-thread caches, so a pool on top adds
a lock and removes nothing.

Large allocations are a different mechanism, and the earlier conclusion does not
transfer to them. Above `M_MMAP_THRESHOLD` (128 KiB by default) `malloc` calls
`mmap` and `free` calls `munmap`. A fresh mapping is **demand-zeroed by the
kernel**: the first store to each page traps, and the kernel zeroes the page
before the store retires. That cost is proportional to the buffer and it is paid
again on every run.

The native CPU EP hits this on every inference. Graph outputs are handed to the
caller by *moving* the produced buffer out of the executor
(`try_move_host_output` in `crates/onnx-runtime-session/src/executor/control_flow.rs`),
which is zero-copy but deliberately drops that value from `buffer_shapes`, so
the next run cannot take the reuse fast path in `ensure_output_backings` and
allocates the output afresh.

Measured with `strace -c` and `/usr/bin/time -v` on a Whisper cross-attention
softmax graph (30000x1500 f32, 180 MB in and out), same binary, `--native-threads 1`:

| | runs=5 | runs=45 | per run |
|---|--:|--:|--:|
| `munmap` calls, cache off | 24 | 64 | **1** |
| `munmap` calls, cache on | 16 | 17 | ~0 |
| minor page faults, cache off | 12,082 | 32,521 | **511** |
| minor page faults, cache on | 9,016 | 9,950 | **23** |

ONNX Runtime pays neither, because its CPU allocator reuses memory it already
owns. That is a structural difference in the allocator, not in any kernel.

## What it does

`LargeAllocCache<A: DeviceAllocator>` wraps an inner allocator. On `deallocate`,
a block whose size falls in `[MIN_CACHED_BYTES, MAX_CACHED_BYTES]`
(256 KiB..1 GiB) is pushed onto a free list keyed by its **exact
`(bytes, align)`** pair instead of being returned to the system; on `allocate`,
an exact match is popped. Anything outside the band is delegated to the inner
allocator untouched, so the small-allocation path stays exactly as it was.

The floor sits above `M_MMAP_THRESHOLD` on purpose: below it, glibc already
recycles without a syscall or a page fault, which is the case the earlier arena
result covers.

### Why exact match rather than size classes

A size-class pool answers a request for `n` bytes with a block of `m >= n`, so
the block's true `Layout` stops matching the caller's view of it — and Rust
requires the *originating* layout at `dealloc`. Exact match keeps
`(bytes, align)` an invariant of the block for its whole life, so a cached block
is indistinguishable from a fresh one and the eventual real free uses the layout
the real allocation used. It costs nothing here: decode is a loop over a small
set of stable shapes, and the second iteration asks for exactly what the first
released.

### Why one cache per EP

The executor allocates a moved-out output through its own EP and hands that same
EP to the resulting tensor, so the matching free returns to the EP that made the
allocation. Scoping the cache to the EP therefore loses no reuse, and it bounds
retention by the session's life rather than the process's. `Drop` frees
everything retained.

## Safety

`DeviceAllocator` requires that "a region becomes reusable only once its matching
`deallocate` has been called". A block enters this cache *from* `deallocate` —
the moment its sole owner relinquished it — and leaves into exactly one
`allocate`, under a shard lock. A cached block is therefore never live twice,
which is the property the trait asks for; recycling is compliant rather than an
exception.

Two consequences stated plainly:

* **Recycled memory is not zeroed.** Neither is `std::alloc::alloc`'s, so the
  contract is unchanged — but a fresh `mmap` *happens* to be zero, and a kernel
  that only partially wrote its output would have been silently masked by that
  accident. Under `debug_assertions` every recycled block is poisoned with `0xA5`
  on the way out, so such a kernel fails loudly in tests instead of passing by
  luck.
* **Cached bytes are retained, not leaked.** They are live process memory under a
  hard cap enforced *before* insertion, so retention cannot exceed the budget
  even transiently.

Falsifiers in `large_alloc_cache.rs`: a released block is recycled; a cached
block is never served to a different layout in either direction; sub-floor and
above-ceiling blocks are never retained; the budget holds; a zero budget makes
the wrapper transparent; concurrently live blocks never overlap; an 8-thread
alloc/write/free stress never hands one block to two threads; `drain` releases
everything.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES` | `2147483648` (2 GiB) | Cap on total retained bytes per CPU EP. `0` disables retention entirely and restores the previous `HostAllocator` behaviour; that is also the control arm for A/B measurement. An unparseable value falls back to the default. |

## Measured effect against ONNX Runtime

Same binary, both arms in one interleaved `scripts/ort_ab/ab.py` invocation,
5 trials x 9 runs x 3 warmups, medians of per-trial p50 native/ORT ratios. Lower
is better; `nocache` is `ONNX_GENAI_HOST_ALLOC_CACHE_BYTES=0`.

| Model | t | nocache | cache | change |
|---|--:|--:|--:|--:|
| kvcat_llama3_p1023 | 1 | 4.835 | **1.841** | -61.9% |
| kvcat_llama3_p1023 | 8 | 14.244 | **5.553** | -61.0% |
| kvcat_llama3_p2047 | 1 | 2.594 | **1.753** | -32.4% |
| kvcat_llama3_p2047 | 8 | 8.546 | **5.264** | -38.4% |
| kvcat_llama3_p4095 | 8 | 5.655 | **4.663** | -17.5% |
| kvcat_llama3_p8191 | 1 | 3.204 | **1.965** | -38.7% |
| kvcat_llama3_p8191 | 8 | 7.087 | **4.435** | -37.4% |
| kvcat_llama3_p8191 | 16 | 8.415 | **5.056** | -39.9% |
| kvcat_llama3_b8_p2047 | 1 | 3.063 | **1.877** | -38.7% |
| kvcat_llama3_b8_p2047 | 8 | 7.088 | **4.254** | -40.0% |
| kvcat_llama3_b8_p2047 | 16 | 7.651 | **4.489** | -41.3% |
| sm_prefill_h32_s512 | 1 | 2.476 | **1.755** | -29.1% |
| sm_prefill_h32_s512 | 16 | 5.979 | **4.708** | -21.3% |
| sm_whisper_cross | 1 | 2.209 | **1.732** | -21.6% |

The pattern is exactly what the mechanism predicts: the win scales with output
size and is largest where the operator is pure data movement (`Concat` over a
growing KV history), because there the page faults are most of the work. Cells
whose outputs sit below the 256 KiB floor — the decode-shaped softmaxes — move
within noise, which is the intended outcome, not a shortfall.

The full 57-cell matrix, including the cells that moved the wrong way, is in
`docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md`.
