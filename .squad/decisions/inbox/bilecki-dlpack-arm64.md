### 2026-07-24: Use per-test counters for DLPack import deleter tests
**By:** Bilecki
**What:** Store an `Arc<AtomicUsize>` in each fake producer context and have its foreign deleter increment that test-local counter; remove the shared import counters and serialization lock.
**Why:** The shared static counter allowed unrelated deferred deleters to contaminate another test's assertion, observed as a Windows ARM64-only failure. Per-test ownership makes the deleter assertions hermetic while leaving production idempotency behavior unchanged.
