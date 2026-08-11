# Decision: EP Assignment Proof via `disable_cpu_ep_fallback`

**Date:** 2026-08-11
**Author:** Freysa (Perf & Testing)
**PR:** #762

## Context

Conformance tests in `plugin_ort_e2e.rs` previously proved only that ORT *ran* the model successfully — not that **our EP** executed the nodes. ORT silently falls back to its built-in CPU EP when a plugin EP declines nodes, making the tests vacuous.

## Decision

Set `session.disable_cpu_ep_fallback=1` via `AddSessionConfigEntry` in `conformance_setup()`. This forces ORT to error if any node falls back, proving our EP claimed all nodes.

The `conformance_mixed_partition` test is exempted (uses `disable_fallback: false`) because it intentionally tests a model containing ops our EP does not support — the test exercises the partition path.

## Proof of non-vacuity

With `disable_cpu_ep_fallback=1` applied universally (before exempting mixed_partition), `conformance_mixed_partition` correctly **FAILED** with:
```
STAGE [CreateSession] FAILED: This session contains graph nodes that are assigned to the
default CPU EP, but fallback to CPU EP has been explicitly disabled by the user.
```
This proves the flag is enforced and would catch any test where our EP silently declines nodes.

## Profiling assertion

Not added. ORT 1.27's plugin-EP API does not expose a per-node "which provider ran this" query post-session. The `disable_cpu_ep_fallback` mechanism is the canonical ORT-endorsed approach for proving assignment. The device-lookup assertion in `conformance_setup` additionally confirms our EP is registered and appended.
