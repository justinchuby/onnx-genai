### 2026-08-26: Fence mutation proofs must isolate the post-removal CUDA operation
**By:** Roy
**What:** For CUDA fence ordering tests, select an explicit post-registry-removal operation (`Enqueue` or test-only `Omit`) while keeping event lookup/removal, ownership, stream submissions, messages, gates, drains, and cleanup on one path. Host release order is a predetermined arm schedule driven by READY/DONE, never by whether the event remains registered. A separate pre-wait READY kernel prevents circular progress.
**Why:** PR #2235 revision 3 mutated `wait_fence` before event removal, so a resolver's missing-event observation—not `cuStreamWaitEvent`—selected the branch and made host scheduling load-bearing. Revision 4 isolates the exact operation and makes the one-line production selector mutation fail deterministically with POISON.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
