# Decision: CUDA EP is implementation-blocked, not hardware-blocked

**Author:** Roy (Lead)
**Date:** 2026-08-11
**Context:** PR #762 rubber-duck review rejection, B1–B4 corrective wave

## Decision

The CUDA EP plugin is **implementation-blocked**. Four defects in the plugin code
prevent correct operation on any host, including hosts with a GPU. We must stop
describing CUDA as "hardware-validation-blocked" or "merely needing a GPU."

The plugin now fails closed: `CreateEpFactories` returns zero factories in both
feature configurations. This is the correct behaviour until all four defects are
resolved.

## Rationale

The rubber-duck review found that the CUDA plugin was **failing open**: advertising
a GPU EP while `CreateDataTransfer` returned NULL, `GetHandle` returned a NULL
stream, the EP/allocator/stream each built separate CUDA runtimes, and `Free`
passed `size=0`. Even with a GPU, this code would produce silent corruption.

We had been describing this as "hardware-blocked" for several sessions — that
framing was wrong and we propagated it through multiple docs. This decision
records the correction.

## Impact

- `docs/CUDA_EP_STATUS.md` — rewritten to reflect implementation-blocked status
- `docs/EP_PLUGIN_EXPORT_PR.md` — rejection recorded with B1–B4 details
- `docs/NXRT_ABI.md` — new ABI contracts: inline buffer rule, `c_char` portability
- `docs/EP_PLUGIN_EXPORT_INVENTORY.md` — CUDA status changed from SCAFFOLDED to IMPLEMENTATION-BLOCKED
- Issue #768 (GPU validation) remains necessary but is no longer sufficient

## What is genuinely working

- CPU EP plugin: 189 passing tests (1 ignored for LayerNorm Mean-shape bug)
- nxrt native ABI: 32/32 ABI + 10/10 host round-trip
- Workspace compiles cleanly
