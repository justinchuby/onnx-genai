### 2026-08-18T04-28-32: Add final VMM-only CUDA phase
**By:** copilot-coordinator
**What:** Add final VMM-only CUDA phase
**References:** #1186, PR #1192, wiki/memory/Memory Management for Beginners.md
**Why:** The owner decided that issue #1186 will gain a final Phase 7 after the existing plugin ABI phase. Phase 7 removes the built-in CUDA eager (`cuMemAlloc`) allocator only after VMM, capability identity, provider/context pinning, deferred release, ProcessMemoryManager bindings, and plugin ABI work demonstrate that VMM fully subsumes it. The ordinary `DeviceAllocator` capability remains for CPU, injected/custom mechanisms, and integration boundaries. The deletion PR must use the `timemachine` label and preserve explicit historical/migration evidence so the removed architecture can be referenced later.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
