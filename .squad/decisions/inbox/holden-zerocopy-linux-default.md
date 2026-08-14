# Holden: raise zero-copy hybrid budget default on Linux — the WDDM aperture ceiling is Windows-specific

Date: 2026-08-14
Author: Holden (measurement + fix; H200 box, Squad team)
Requested by: Justin (@justinchuby)
Evidence: #925 comment https://github.com/justinchuby/onnx-genai/issues/925#issuecomment-5291022718
Code: `crates/onnx-runtime-ep-cuda/src/weight_paging.rs` (`ZERO_COPY_SAFE_BUDGET_BYTES*`)

## What

Made the zero-copy hybrid default safety budget (`ZERO_COPY_SAFE_BUDGET_BYTES`)
platform-conditional instead of a single global 256 MiB constant:

- **Windows (WDDM):** unchanged at **256 MiB** (`ZERO_COPY_SAFE_BUDGET_BYTES_WDDM`).
  The #864 aperture ceiling is real there and this value stays strictly under it.
- **Non-Windows (Linux / discrete GPU):** raised to **2 GiB**
  (`ZERO_COPY_SAFE_BUDGET_BYTES_NON_WINDOWS`).

Split is `const ZERO_COPY_SAFE_BUDGET_BYTES = if cfg!(target_os = "windows") {WDDM} else {NON_WINDOWS}`,
so both named consts are referenced in every build (greppable, testable, no dead_code).

## Why

**The WDDM "aperture ceiling" is Windows/VidMm-specific and absent on Linux.** #925
re-measured the #864 hybrid on this H200 box (driver 580.105.08, CUDA 13.0, kernel
6.6.141.1-1.azl3, Azure Linux 3.0, native VMM decode path). The finding was decisive:

- Generation stayed **byte-identical** to the Step-0 baseline with `cuda_graph`
  `fallbacks=0` up to **6.795 GB** of distinct host-mapped weights bound and re-read
  in place every decode step (704 `cuMemHostRegister` binds), **n=3, all runs
  byte-identical** — ~15× the WDDM ~0.44 GB corruption onset.
- On WDDM (#864/#912, RTX 4060 Laptop) the same mechanism silently returned stale
  data above ~0.44–0.65 GB/step (48 cold weights collapsed generation 16 → 3 tokens).
  On Linux `cuMemHostRegister(READ_ONLY|DEVICEMAP)` pins pages in the driver with no
  VidMm layer above it, so that ceiling does not exist.
- The shipped 256 MiB default left an **~8× decode win unused** in the over-budget
  regime on Linux: hybrid @ 8 GiB budget **67.04 tok/s** vs managed streaming
  **~8.5 tok/s** median (both byte-identical output). The safe 256 MiB arm ran ~3.62.

**Why 2 GiB and not higher / unbounded:** bounded on purpose. It sits **>3× below**
the #925 measured-safe 6.795 GB (margin, since only Hopper/H200 was tested), yet
clears the entire WDDM 0.44–0.65 GB corruption band by >3× and covers the observed
~0.85 GB per-step cold working set plus a realistic deferred fraction with headroom —
which is what unlocks the win. Staying bounded avoids over-committing pinned host RAM
and avoids extrapolating past tested hardware (Rule 11 / portability). Operators can
still override per-run via `ONNX_GENAI_ZERO_COPY_HYBRID_BUDGET_BYTES`.

**Blast radius is low:** the whole feature is opt-in (defaults OFF behind
`ONNX_GENAI_ZERO_COPY_HYBRID=1`; the `zero_copy_hybrid_env_defaults_off_and_opts_in`
test is intact), so this default only affects users who already turned the hybrid on.

## Scope / non-goals

- Windows behaviour is deliberately **unchanged** — do not inherit the Linux
  conclusion there (the inverse of #783's lesson).
- Measurement only proved the ceiling absent on H200/Hopper. Other Linux GPU classes
  are untested; the bounded 2 GiB default and the override knob are the guardrails.
