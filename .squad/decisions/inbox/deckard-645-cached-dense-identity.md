### 2026-08-05: Kernel-local identity for bounded cached-dense weights
**By:** Deckard
**What:** Block-quantized CPU kernels memoize immutable constant-slot identity per kernel, use mmap mapping metadata when available, hash each packed matrix/expert slice once, and keep dense cache builds under the kernel-local mutex for single-flight expansion.
**Why:** This restores near-OnceLock hit overhead without trusting a reusable address as global identity, avoids full MoE tensor hashing, and prevents concurrent duplicate dense allocations. The byte budget remains per kernel instance rather than model-wide.
