# Vendored MLAS subset — provenance

**FEASIBILITY SPIKE.** This directory contains a **partial copy** of ONNX
Runtime's MLAS math library, vendored to evaluate building MLAS's real f32
SGEMM standalone and calling it over FFI. See `docs/MLAS_SYS_SPIKE.md`.

## Source

- Upstream: https://github.com/microsoft/onnxruntime
- Commit: `da9049437190fa3552d1b31eacb164c3ec48d8b4`
- Copied paths (from the upstream repo root):
  - `onnxruntime/core/mlas/inc/`  → `mlas/onnxruntime/core/mlas/inc/`
  - `onnxruntime/core/mlas/lib/`  → `mlas/onnxruntime/core/mlas/lib/`
    (only the x86-64 + generic subset; ARM/POWER/WASM/RISC-V/LoongArch/s390x
    kernel sources were dropped, along with the Windows MASM `amd64/` tree and
    the two uncompiled TUs `cast.cpp`/`convolve.cpp` whose external includes
    — GSL `narrow`, `SafeInt.hpp` — we do not satisfy standalone)
  - `onnxruntime/core/platform/env_var.h` → `mlas/onnxruntime/core/platform/env_var.h`
    (self-contained; needed by `qkv_quant_kernel_avx512vnni.cpp`)

The `x86_64/` directory holds the GAS/Linux (`.S`) assembly kernels. Windows
would use the MASM `.asm` variants (not vendored here) — see the spike doc.

## License

MLAS is MIT-licensed. ONNX Runtime's `LICENSE` is preserved verbatim in
`mlas/LICENSE`. Individual source files retain their original Microsoft/Intel
copyright headers. This repository is also MIT, so the two are compatible.

## Local additions (NOT from upstream)

These small files were written for the spike and are **not** MLAS source:

- `shim.cpp` — `extern "C"` wrappers over `MlasGemmBatch` / `MlasGemmPackB`.
- `probe.cpp` — reports which f32 GEMM microkernel MLAS's runtime dispatch
  selected (used to prove the AVX-512 kernel is active).
- `compat/core/common/common.h` — empty stand-in. `q4common.h` `#include`s
  `core/common/common.h` but uses nothing from it on the path we compile;
  the stub avoids pulling ORT's `core/common` tree.

## Local modifications

The following vendored MLAS files contain small, in-source-marked patches:

- `onnxruntime/core/mlas/lib/mlasi.h`
- `onnxruntime/core/mlas/lib/threading.cpp`

Each patch is marked with the exact `nxrt-mlas-mt` comment tag so it can be
located with `grep`. The patches wire MLAS's standalone
`BUILD_MLAS_NO_ONNXRUNTIME` parallel-for primitives to the pluggable host
thread-pool hooks in `shim.cpp` (`MlasStandaloneParallelFor` and
`MlasStandaloneMaxThreads`), so `MlasGemmBatch` runs multi-threaded on our
Rayon pool.

After re-vendoring from upstream, these two patches **MUST** be re-applied:
grep the old tree for `nxrt-mlas-mt` first to locate them, and keep this
section in sync with the patched files.

## How it is built

`build.rs` compiles the subset with the `cc` crate (no cmake), grouping sources
by ISA exactly as `cmake/onnxruntime_mlas.cmake` does for the `X86_64` branch,
with `-DBUILD_MLAS_NO_ONNXRUNTIME` (MLAS's standalone CPUID/threading shim).
The vendored MLAS sources are otherwise unmodified from upstream; the few TUs
that rely on headers ORT supplies transitively are handled with compiler
`-include` flags in `build.rs`.
