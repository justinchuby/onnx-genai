# Vendored MLAS — provenance

This directory contains ONNX Runtime's MLAS math library plus the KleidiAI
micro-kernels needed by the ARM64 QNBit/SQNBit path. It is built directly by
`crates/mlas-sys/build.rs` without CMake.

## Source

- Upstream: https://github.com/microsoft/onnxruntime
- Commit: `da9049437190fa3552d1b31eacb164c3ec48d8b4`
- Copied paths (from the upstream repo root):
  - `onnxruntime/core/mlas/inc/`  → `mlas/onnxruntime/core/mlas/inc/`
  - `onnxruntime/core/mlas/lib/`  → `mlas/onnxruntime/core/mlas/lib/`
  - `onnxruntime/core/platform/env_var.h` → `mlas/onnxruntime/core/platform/env_var.h`
    (self-contained; needed by `qkv_quant_kernel_avx512vnni.cpp`)
- KleidiAI: https://github.com/ARM-software/kleidiai
- KleidiAI version: `v1.20.0` (the version pinned by ORT `cmake/deps.txt`)
- Copied paths:
  - `kai/` → `mlas/kleidiai/kai/`
  - `LICENSES/`, `README.md`, `CHANGELOG.md` → `mlas/kleidiai/`

The `x86_64/` directory holds the GAS/Linux (`.S`) assembly kernels. The
`arm64/` directory holds the Windows ARM64 (`.asm`) kernels; `build.rs`
preprocesses them with `cl.exe /P` before invoking `armasm64.exe`, mirroring
upstream CMake. KleidiAI's qsi4 QNBit `.S` micro-kernels are preprocessed with
clang's assembler preprocessor and assembled with `armasm64.exe` on Windows
ARM64.

## License

MLAS is MIT-licensed. ONNX Runtime's `LICENSE` is preserved verbatim in
`mlas/LICENSE`. KleidiAI is Apache-2.0 and its license files are preserved under
`mlas/kleidiai/LICENSES/`. Individual source files retain their original
copyright headers.

## Local additions (NOT from upstream)

These small files were written for the spike and are **not** MLAS source:

- `shim.cpp` — `extern "C"` wrappers over `MlasGemmBatch` / `MlasGemmPackB`.
- `probe.cpp` — reports which f32 GEMM microkernel MLAS's runtime dispatch
  selected (used to prove the AVX-512 kernel is active).
- `kai_qnbit_interface.cpp` — a minimal KleidiAI registry for the NEON
  DotProd/I8MM qsi4 QNBit micro-kernels used by MLAS's ARM64 SQNBit dispatch.
- `compat/core/common/common.h` — empty stand-in. `q4common.h` `#include`s
  `core/common/common.h` but uses nothing from it on the path we compile;
  the stub avoids pulling ORT's `core/common` tree.

## Local modifications

The following vendored MLAS files contain small, in-source-marked patches:

- `onnxruntime/core/mlas/lib/mlasi.h`
- `onnxruntime/core/mlas/lib/threading.cpp`
- `onnxruntime/core/mlas/lib/reorder.cpp`
- `onnxruntime/core/mlas/lib/snchwc.cpp`

Each patch is marked with the exact `nxrt-mlas-mt` comment tag so it can be
located with `grep`. The patches wire MLAS's standalone
`BUILD_MLAS_NO_ONNXRUNTIME` parallel-for primitives to the pluggable host
thread-pool hooks in `shim.cpp` (`MlasStandaloneParallelFor` and
`MlasStandaloneMaxThreads`), so `MlasGemmBatch` runs multi-threaded on our
persistent Rust work-stealing pool.

After re-vendoring from upstream, these patches **MUST** be re-applied:
grep the old tree for `nxrt-mlas-mt` first to locate them, and keep this
section in sync with the patched files.

## How it is built

`build.rs` compiles the needed MLAS source groups with the `cc` crate (no
cmake), grouping sources by ISA from `cmake/onnxruntime_mlas.cmake` for x86-64
and ARM64, with `-DBUILD_MLAS_NO_ONNXRUNTIME` (MLAS's standalone
CPUID/threading shim). The ARM64 QNBit build compiles the ORT NEON
`sqnbitgemm*` sources plus KleidiAI's qsi4 DotProd/I8MM pack/matmul
micro-kernels so `SQNBIT_CompInt8` and `SQNBIT_CompFp32` are both available.
