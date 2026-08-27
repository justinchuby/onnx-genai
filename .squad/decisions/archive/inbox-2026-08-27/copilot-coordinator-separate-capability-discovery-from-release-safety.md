### 2026-08-18T14-35-37: Separate capability discovery from release safety
**By:** copilot-coordinator
**What:** Separate capability discovery from release safety
**References:** #1186, #1247, #1233, #1237
**Why:** After reverting unsafe combined Phase 1/2 in #1247, issue #1186 was clarified. Phase 2 owns only explicit VirtualBacking/SharedMapping capability discovery and dispatch through the already-selected allocator reference, with an honest trusted-implementor coherence contract for unsafe third-party mechanisms. It must not introduce self-reported MechanismId/pointer/TypeId proofs, retryable pointer release, allocation generations, RAII, or partial-release completeness claims. Phase 3 owns manager-issued binding/mechanism identity and opaque allocation generation/cookies. Phase 4 owns consuming owning handles, stream-ordered deferred release, ABA protection, structured complete/quarantined/failure outcomes, mapped-vs-physical accounting reconciliation, and allocator-owned quarantine for partial CUDA unmap/handle-release or rollback failures.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
