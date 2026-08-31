### 2026-08-25: CUDA MHA scratch is RAII-owned and geometry is checked twice
**By:** Deckard
**What:** `MultiHeadAttention` now owns per-call device scratch through a runtime-backed RAII guard, synchronizes only after stream submission, and validates statically-known claim geometry plus all execute-time products, bytes, offsets, grids, and CUDA integer arguments before allocation or launch.
**Why:** Manual tail cleanup leaked on later allocation/H2D failures, while lossy casts and overflowing products could undersize unsafe CUDA work. Present tensors remain caller-owned; cuBLASLt workspace remains runtime-owned.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
