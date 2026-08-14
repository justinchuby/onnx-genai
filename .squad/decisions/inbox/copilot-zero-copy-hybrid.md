# Decision: zero-copy hybrid weight residency is a measured negative on RTX 4060 / WDDM (#864)

**Date:** 2025
**Author:** Copilot (agent)
**Branch/PR:** `squad/zero-copy-hybrid`

## What was built
A default-OFF `ONNX_GENAI_ZERO_COPY_HYBRID` mode in the CUDA EP: keep the
size-blind `StableResident` hot set copied into VRAM, and bind the cold
remainder in place from a `cuMemHostRegister(READ_ONLY | DEVICEMAP)` host
mapping instead of streaming it transiently each decode step. The bypass
decision is intercepted **before** any eviction, so the hot set never
evicts/re-admits a large stable slot (the #886 corruption pattern). The whole
mmap is registered once per `mapping_id` so each weight's device pointer is
contiguous over its full length.

## The finding (negative, and the point of the work)
- A **single** zero-copy host-mapped weight read is **bit-identical** — verified
  correct at 1, 8, 16, 32 cold weights per step (confirms #877/#880 for one op).
- **Aggregate** distinct host-mapped read traffic above ~0.44–0.65 GB/step
  **silently corrupts** decode: 32 cold weights (~0.44 GB) = correct; 48
  (~0.65 GB) = generation collapsed 16 → 3 tokens. Same signature as #886, but
  the mechanism here is **stale host-mapped reads past a system-memory-aperture
  ceiling**, not eviction/re-admission. An A/B isolation (defer exactly as the
  zero-copy path would, but perform the real copy) produced byte-identical
  output, proving the deferral/admission flow is correct and the fault is the
  host-mapped READ itself.
- `cuMemHostRegister` of the full 16.65 GB mapping **only succeeds with
  READ_ONLY**; DEVICEMAP-only fails `CUDA_ERROR_OUT_OF_MEMORY`. So we cannot
  drop READ_ONLY to work around it.
- CPU pre-faulting every page before registration did **not** fix it (not a
  simple demand-paging race). Device pointers are 256-byte aligned (not an
  alignment fault).

## Throughput (medians, `qwen14b-zp`, `--tokens 16 --steady`, graph ON)
| Arm | tok/s | htod_bytes/token | zero_copy_bytes/token | byte_hit_rate | tokens | captures/fallbacks |
|-----|-------|------------------|-----------------------|---------------|--------|--------------------|
| WDDM (`LEGACY_ALLOCATOR=1`) | **7.37** (5.61–8.32, n=3) | 0 (driver-managed) | 0 | n/a | ✅ ref | 4/0 |
| Managed streaming | 0.14 | 2.35 GB | 0 | 70.2% | ✅ ref | 2/0 |
| Hybrid (safe 256 MiB budget) | 0.04 | 1.71 GB | 0.26 GB | 77.5% | ✅ ref | 2/0 |
| Hybrid (full cold set, ~200 zc/step) | — | — | ~0.87 GB | — | ❌ 16→3 collapse | — |

Reference tokens (all ✅ rows byte-identical):
`[96347, 3375, 724, 11, 358, 2776, 14589, 311, 6723, 429, 498, 3003, 2581, 6617, 315, 752]`.

## Decision
- **The hybrid does not beat WDDM on this hardware.** WDDM keeps ~7.7 GB
  resident and moves only ~0.6 GB/step through the driver's own paging; our
  managed budget caps at ~6.1 GB and zero-copy can only *safely* cover
  ~0.44 GB/step. Both levers are worse than the OS here.
- **Ship it default-OFF with a conservative 256 MiB zero-copy budget** so the
  opt-in knob is always byte-identical (never exercises the corruption ceiling).
  It is retained as instrumented, reviewable infrastructure for other hardware
  (e.g. datacenter GPUs with resizable BAR / larger host apertures may not hit
  the ceiling), not as a Windows win.
- **Do not build a churning dynamic hot set** — unnecessary and unsafe (#886).

## Safety / gates verified
Token IDs byte-identical; `captures>0`/`fallbacks==0`; `oversubscribed_bytes==0`;
`ref_underflows`/`byte_underflows`/`unaccounted_committed_bytes` all 0; MANDATORY
`mobius_seqmajor_growth_parity_native_cuda` passed solo (2/2, no ILLEGAL_ADDRESS).
