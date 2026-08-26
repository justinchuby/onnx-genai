### 2026-08-25 — Provider artifacts finalize per executor after resolved compilation

**By:** Deckard (Systems Dev)

**What:** The session executor now assigns a process-unique executor identity and
uses one idempotent provider-artifact finalization transition after kernel
producers are compiled. Static build and first resolved symbolic compilation use
the same transition. CUDA QMoE telemetry, install outcomes, retained artifacts,
boundary consumption, and teardown are scoped by that identity.

**Why:** EP-global install state let symbolic builds permanently latch an empty
telemetry registry and let sibling/MTP executors sharing one EP overwrite or
drain each other. Readiness absence is now non-latching; structural outcomes
latch once; dynamic specializations reuse a stable producer; teardown drains
only the owning executor.
