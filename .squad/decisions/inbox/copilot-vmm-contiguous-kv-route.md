### 2026-08-10: KV batching uses VMM contiguous virtual addresses, not paged attention
**By:** Copilot (coordinator), at the owner's direction
**What:** Multi-request batching (#750) gives each sequence its own contiguous
device virtual address, with physical granules mapped underneath on demand. The
attention kernel keeps seeing a flat contiguous KV buffer and never learns about
pages. `#721` stage 3 (`CudaPageStore`, a device-resident paged KV consumer) is
superseded unless this route fails.
**Why:** Owner directive — "有没有可以把kernel抽象出去 用vmm解决的方法？".
Paged attention requires teaching every CUDA kernel to walk a page table, which
is a permanent complexity tax; VMM does the same job once, in the allocator.
Dense batch>1 was rejected because it pads to the longest sequence in the batch
and re-triggers CUDA graph capture on every membership change.

**The result this must beat:** #721 stage 4 measured a stable full-context VA
committing **1.5 GB** where bucket growth commits **48 MB** (32x worse). The
cause was the decode kernel reading the full padded shape and relying on
masking, which forbids decommitting the tail. The route therefore reduces to one
question: can the decode kernel be bounded to the live sequence length instead
of reading the padded shape? If not, this decision must be revisited.

**Constraints carried forward:** granules come from the #740 authority-scoped
shared pool through existing `carve()` suballocation (per-sequence physical
reservations are what made #733 net-negative); commit floors are per-object, not
per-token (qwen0.5b committed 192 MiB for ~3 MiB of content); `cuMemMap` during
capture is not proven replayable; never unmap while a replay may be in flight.
