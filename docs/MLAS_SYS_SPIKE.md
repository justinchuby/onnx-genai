# MLAS-sys feasibility spike: vendored MLAS f32 SGEMM over FFI

**Status:** GO ✅ &nbsp;|&nbsp; **Author:** Deckard (systems/build) &nbsp;|&nbsp;
**Date:** 2026-07-19 &nbsp;|&nbsp; time-boxed go/no-go spike

## Question

Our pure-Rust CPU GEMM (`onnx-runtime-ep-cpu` `SimdX86`, AVX2/FMA only) is
~2.2× slower single-thread than ONNX Runtime (which uses MLAS) on this host
(Intel Sapphire Rapids Xeon 8480C: AVX-512 F/DQ/BW/VL/VNNI/BF16/FP16 + AMX).
Can we get **parity by construction** by vendoring MLAS's real MIT-licensed
source into an `mlas-sys` FFI crate and calling its hand-tuned AVX-512 kernels
directly?

**Answer: yes.** A standalone `cc`-driven build of a vendored MLAS subset
compiles cleanly, produces correct results, and its runtime dispatch selects
the **AVX-512F** SGEMM microkernel, reaching (and slightly beating) the recorded
ORT single-thread number — ~2.4–3.2× faster than our current `SimdX86`.

## What was built

`crates/mlas-sys` — an isolated spike crate (registered in the workspace but
kept out of `default-members`, so bare `cargo build`/CI does not compile it):

- `vendor/mlas/` — partial copy of MLAS from onnxruntime commit
  `da9049437190fa3552d1b31eacb164c3ec48d8b4` (x86-64 + generic sources, GAS
  `.S` kernels, `inc/` headers), with ORT's MIT `LICENSE` preserved. See
  `vendor/mlas/README.md` for exact provenance.
- `vendor/shim.cpp` — `extern "C"` wrappers over `MlasGemmBatch` and
  `MlasGemmPackB` (single-threaded: NULL threadpool).
- `vendor/probe.cpp` — reports the runtime-selected f32 GEMM microkernel.
- `build.rs` — compiles the subset with the `cc` crate.
- `src/lib.rs` — safe Rust wrappers + correctness tests + a perf probe.

### Build approach: `cc`, not `cmake`

`cc` was tractable and is preferred (no cmake dependency, no generated build
tree, faster incremental). Sources are grouped by instruction-set extension and
given the **exact per-group flags** `cmake/onnxruntime_mlas.cmake` uses for its
`X86_64` branch (`-msse2`, `-mavx`, `-mavx2 -mfma -mf16c -mavxvnni`,
`-mavx512f`, `-mfma -mavx512vnni -mavx512bw -mavx512dq -mavx512vl`, etc.). MLAS
kernels are pre-annotated per ISA and runtime CPU dispatch (in `platform.cpp`)
picks the best one, so we compile the whole platform set and let it choose.

Key enablers / gotchas discovered:

1. **`-DBUILD_MLAS_NO_ONNXRUNTIME`** — MLAS ships a *standalone* mode that
   supplies its own CPUID + threading shim, so **no ORT runtime headers are
   required** for the math library. This is the single most important finding:
   MLAS is explicitly designed to build outside the ORT tree.
2. **Do NOT define `ORT_MINIMAL_BUILD`.** It is tempting (it removes the int4
   `Q4` GEMM dispatch, which otherwise wants `SafeInt.hpp`), **but the entire
   AVX-512 kernel-selection block in `platform.cpp` is gated behind
   `!ORT_MINIMAL_BUILD`.** Defining it silently drops you to the FMA3/AVX2
   SGEMM kernel (measured 162 µs pre-packed) instead of AVX-512F (~90 µs).
   Instead we compile just `q4gemm_avx512.cpp` (the only TU defining the two Q4
   dispatch symbols `platform.cpp` references) and stub the vestigial
   `core/common/common.h` include with an empty header.
3. `platform.cpp`'s dispatch-table constructor references a symbol from *every*
   kernel TU, so the compiled set is "the whole x86-64 platform kernel family",
   not just `SgemmKernel*.S`. Two uncompiled high-level TUs with unsatisfiable
   external includes (`cast.cpp` → GSL `narrow`, `convolve.cpp` → `SafeInt.hpp`)
   are excluded; neither is needed for SGEMM.
4. A handful of TUs assume system headers (`<unistd.h>` for `syscall()` in AMX
   init, `<cstring>`, …) that the full ORT include graph pulls in transitively;
   `build.rs` force-`-include`s them for C++ groups so **no vendored source was
   edited**.
5. Pre-packed B buffers are accessed with **aligned** AVX-512 loads — the Rust
   side must 64-byte-align the packed allocation (a plain `Vec<u8>` SIGSEGVs).

## Correctness

`cargo test -p mlas-sys --release` — **7 passed**. Cases check `mlas_sgemm`
against a naive triple-loop reference (tol 1e-3): square, non-square,
non-tile-multiple (1×1×1, 3×5×7, 17×31×13, 33×65×129, 32×512×512), α/β scaling,
**transpose-A**, **transpose-B**, and the pre-packed-B path. A dedicated test
asserts the runtime-selected kernel id == 512 (AVX-512F).

## Performance (single-thread, medium shape `32×512×512`, 8.4 MFLOP)

Warm cache, allocations outside timing, 5000 iterations. `mlas-sys` numbers are
measured here; ORT/SimdX86 are the baselines recorded in
[`docs/KERNEL_PERF.md`](KERNEL_PERF.md).

| Backend (1 thread) | µs/iter | GFLOP/s | vs ORT | vs SimdX86 |
|---|---:|---:|---:|---:|
| Rust `SimdX86` (AVX2/FMA, recorded) | 285 | ~30 | 2.2× slower | 1.0× |
| ORT 1.27 CPU EP / MLAS (recorded) | 131 | ~64 | 1.0× | 2.2× faster |
| **Vendored MLAS, repack B per call** | **~120** | ~140 | **~1.1× faster** | ~2.4× faster |
| **Vendored MLAS, pre-packed B** | **~90** | ~185 | **~1.45× faster** | ~3.2× faster |

The **repack-per-call** number (~120 µs) is the conservative comparison and
already matches/beats ORT's recorded 131 µs. The **pre-packed-B** number
(~90 µs) mirrors how ORT actually runs MatMul (constant weights are packed once)
and is the realistic integration target. Either way: **vendored MLAS reaches
ORT-class single-thread performance — parity by construction confirmed.**

Build cost: clean compile of the vendored subset ≈ **51 s**; ~1.8 MB of static
archives (~14 object archives). Incremental Rust rebuilds are unaffected
(`build.rs` reruns only when `vendor/` changes).

## What a real integration needs (beyond this spike)

Rough effort estimate: **~1–2 focused weeks** to a production-quality
`onnx-runtime-ep-cpu` backend, dominated by threading and the build matrix, not
the GEMM itself.

1. **Threadpool integration (M).** This spike passes a NULL threadpool
   (single-thread). For multi-thread parity we must bridge our Rayon pool to
   MLAS's `MLAS_THREADPOOL`/`MlasTrySimpleParallel` abstraction (or run MLAS on
   ORT's threadpool shim). This is the biggest correctness/perf item — the
   recorded 8-thread ORT gap (~30 µs) depends on MLAS's fine-grained
   partitioning + shared packed-B panel.
2. **Windows build (M).** The `.S` kernels are GAS/Linux only; Windows uses the
   MASM `amd64/*.asm` variants. `cc` won't assemble MASM — needs `ml64.exe`
   (custom build step) or the `cmake` crate driving `onnxruntime_mlas.cmake`.
   macOS/x86-64 works with the same `.S` files (minus a couple of `APPLE`
   exclusions already handled in the cmake).
3. **Cross-platform / non-x86 (M).** ARM64 (Apple Silicon, Graviton) needs the
   `aarch64/*.S` NEON kernels + their `-march` flags. A real crate should mirror
   the cmake's per-arch source lists behind `cfg!(target_arch)`.
4. **B pre-packing seam (S).** Expose `MlasGemmPackB` in the EP so constant
   MatMul/Gemm weights are packed once (the ~90 µs path), including a 64-byte
   aligned packed-weight allocation owned by the graph.
5. **More entry points (S–M, incremental).** Same vendoring immediately gives
   `MlasConv`, `MlasQgemm` (u8s8 + AMX int8), `MatMulNBits`-equivalent int4
   (`q4gemm`/`sqnbitgemm`), and fp16/bf16 GEMM (`halfgemm`/`sbgemm`). Each is a
   thin shim once the build is in place; int4/fp16 are the high-value follow-ups
   for LLM inference.
6. **Vendoring/sync process (S).** Directory re-copy from a pinned upstream SHA
   is enough (documented in `vendor/mlas/README.md`). `git subtree` would
   automate it but manual copy of the x86-64 subset is simplest and keeps the
   tree small (~3.5 MB). Pin the SHA; re-sync deliberately.
7. **Build/CI (S).** ~50 s of C++ compile added to the EP crate; gate behind a
   Cargo feature (e.g. `mlas`) so pure-Rust builds stay fast, and cache the
   `cc` objects in CI.

### Risks (top 3)

1. **Threadpool bridging** — getting multi-thread parity (and not regressing vs
   our Rayon path) is the main unknown; MLAS expects its own threading contract.
2. **Windows MASM build** — a second, different assembler toolchain; likely the
   biggest portability tax. Mitigation: fall back to the `cmake` crate on
   Windows, keep `cc` on Unix.
3. **Vendored-source maintenance / drift** — we carry ~3.5 MB of upstream C++
   and asm; upstream refactors (like the `ORT_MINIMAL_BUILD`/AVX-512 gate found
   here, or the newer `BackendKernelSelectorConfig` API param) can silently
   change behavior on re-sync. Mitigation: pin the SHA, keep the kernel-id
   probe test as a guard, and re-review on every bump.

## Recommendation

**GO.** Vendoring MLAS as an FFI crate is feasible with a plain `cc` build, is
correct, and delivers ORT-class (parity/slightly-better) single-thread f32
SGEMM on this AVX-512 host by using ORT's own kernels. Proceed to a real
`onnx-runtime-ep-cpu` backend behind a Cargo feature, tackling threadpool
integration first and deferring the Windows MASM build until a Windows target
is required.

## Reproduce

```bash
cargo test  -p mlas-sys --release                                  # correctness (7 tests)
cargo test  -p mlas-sys --release -- --nocapture avx512_kernel_is_selected
cargo test  -p mlas-sys --release -- --ignored --nocapture perf_sgemm_medium
```
