# Cross-platform audit

Audit date: 2026-07-14. Scope: the Cargo workspace under `crates/`, Python
packaging, and GitHub Actions. This is a first-pass, read-only audit; it does not
claim that any target has passed a native build yet.

## Support matrix

| Operating system | CPU EP | CUDA EP | Metal |
|---|---:|---:|---:|
| Linux (x86_64; arm64 where dependencies permit) | Required | Required on NVIDIA systems | N/A |
| Windows (x86_64 initially) | Required | Required on NVIDIA systems | N/A |
| macOS (x86_64 and arm64) | Required | Unsupported; CUDA absence must never break import or CPU use | Future/optional; MLX plugin currently macOS arm64 only |

The default `nxrt` wheel must import and run CPU inference on every row. A CUDA
build must probe the driver and user-space libraries at runtime and report
availability accurately. Missing CUDA must produce either `available == false`
or an actionable error that says what was missing, why it is needed, and how to
install/select CPU instead; it must not panic or silently execute on CPU.

Linux, macOS, and Windows `nxrt` wheels bundle a static oneDNN build and enable
the oneDNN CPU backend by default, following the PyTorch-style wheel model. The
Cargo/crates.io surface remains pure Rust by default because `onednn` stays a
non-default feature. Windows initially uses oneDNN's sequential runtime for
linker robustness; OMP/TBB is a tracked follow-up optimization.

## Findings

| Severity | Area | File:line | Issue | Recommended fix |
|---|---|---|---|---|
| 🔴 | CI | `.github/workflows/ci.yml:13-36` | All formatting, clippy, build, and test coverage runs only on `ubuntu-latest`. Windows and macOS compile failures are invisible. | Under `xplat-ci-wheel-matrix`, add Ubuntu, Windows, macOS x86_64, and macOS arm64 coverage. Run the portable CPU surface everywhere; compile-check CUDA on Linux/Windows and assert it is disabled on macOS. |
| 🔴 | Wheel release | `crates/onnx-runtime-python/pyproject.toml:6-42`; `.github/workflows/publish.yml:12-114` | Maturin metadata exists, but there is no cibuildwheel configuration or Python wheel publishing job. The current publish workflow is Linux-only and publishes Rust crates only. | Add cibuildwheel jobs for manylinux, Windows, macOS x86_64, and macOS arm64, with separate standard `cp310-abi3` and free-threaded `abi3t` lanes as specified by `docs/PIPELINE.md:940-968`; use Trusted Publishing and install/import smoke tests. |
| 🔴 | CUPTI loading | `crates/onnx-runtime-tracer/src/cupti.rs:92-100` | Every hardcoded CUPTI candidate is a Linux soname: `libcupti.so`, `.so.13`, `.so.12`, `.so.11`. Windows candidates such as `cupti64_*.dll` are absent. | Under `xplat-dlopen-oses`, use target-specific candidate lists. CUDA 13 Windows candidates must include the actual NVIDIA wheel/toolkit names (for example `cupti64_2025.1.0.dll`, while allowing compatible CUDA-13 variants). macOS should return unavailable without attempting CUDA names. |
| 🔴 | CUPTI wheel discovery | `crates/onnx-runtime-tracer/src/cupti.rs:158-166` | Loading tries bare filenames through the process loader only. It never searches pip-installed `nvidia/cuda_cupti/lib` on Linux or `nvidia/cuda_cupti/bin` on Windows, so the promised zero-setup CUDA wheel will not find CUPTI without loader-path configuration. | Discover Python `site-packages` roots, then try absolute paths under both OS layouts before system candidates. Preserve graceful `available == false`; optionally retain the failed candidates for a RULES.md #1 diagnostic. |
| 🔴 | CUDA library discovery | `crates/onnx-runtime-ep-cuda/Cargo.toml:16-26`; `Cargo.lock:584-590` | Resolved cudarc 0.19.8 dynamically loads base names `cuda`/`nvcuda`, `cublasLt`, and `nvrtc`. Its generated OS-aware candidates include Linux `libcuda.so.1`, `libcublasLt.so.13`, `libnvrtc.so.13` and Windows `nvcuda.dll`, `cublasLt64_13.dll`, `nvrtc64_130_0.dll`, but it searches only normal loader paths—not pip wheel `nvidia/*/lib` or `nvidia/*/bin` directories. | Add an nxrt-owned discovery/preload seam or an upstream cudarc extension that accepts absolute candidate paths. Search `nvidia/cublas`, `nvidia/cuda_nvrtc`, and any runtime dependency in `lib` (Linux) or `bin` (Windows); the NVIDIA driver library remains a system prerequisite. |
| 🔴 | CUDA graceful absence | `crates/onnx-runtime-ep-cuda/src/runtime.rs:38-47`; `crates/onnx-runtime-ep-cuda/src/blas.rs:99-103`; `crates/onnx-runtime-ep-cuda/src/error.rs:1-20` | The local API promises `CudaRuntime::new` returns an error when libraries are absent, but cudarc's dynamic loader panics when a required library cannot be loaded. The local `map_err` runs only after symbol loading succeeds, so missing CUDA can unwind instead of becoming an actionable `EpError`. | Replace/patchextend the loader with fallible library acquisition before calling cudarc symbols. Never use `catch_unwind` as the primary design. Report the missing component, searched paths/names, required driver/PyPI package, and CPU fallback option. |
| 🔴 | Python CUDA availability | `crates/onnx-runtime-python/src/lib.rs:51-82,224-299`; `crates/onnx-runtime-python/Cargo.toml:22-39` | A CUDA-feature build advertises `CUDAExecutionProvider` solely at compile time. Session construction validates the string but always creates the same CPU `RtSession`; requested CUDA is stored and reported without wiring a CUDA EP or probing its libraries/device. This is a silent correctness failure and can misreport CUDA on macOS/driverless hosts. | Wire the requested EP into session creation. Compute availability from a fallible runtime probe, advertise CUDA only on supported OSes with a usable stack, and return a what/why/how-to-fix error if explicitly requested but unavailable. Default remains CPU. |
| 🔴 | ORT Windows bootstrap | `crates/onnx-genai-ort/ort-sys/build.rs:204-267` | Automatic Windows setup downloads a zip and invokes an external `unzip` executable. `unzip` is not a Windows platform guarantee, so a clean native build can fail before compilation. Existing panic text does not explain how to install the tool or use `ORT_ROOT`. | Extract with a Rust zip crate or a guaranteed platform facility. If an external command remains, include the command, archive path, underlying error, and concrete `ORT_ROOT` workaround per RULES.md #1. |
| 🟢 | oneDNN Windows feature | `crates/onnx-runtime-ep-cpu/build.rs:72-89,131-169`; `.github/workflows/wheels.yml:28-51`; `crates/onnx-runtime-python/pyproject.toml:67-76` | **RESOLVED:** `build.rs` is MSVC-aware: it emits no `stdc++`/`gomp` for MSVC, relies on automatically linked `msvcprt`/`vcomp`, and supports `ONEDNN_OMP_LIB` for Intel OpenMP. The Windows CPU wheel now builds with `--features onednn`, initially using the SEQ runtime. | Verify the native MSVC oneDNN CMake wheel build in CI. Track Windows OMP/TBB as a follow-up optimization after the sequential configuration is proven. |
| 🟡 | ORT target coverage | `crates/onnx-genai-ort/ort-sys/build.rs:274-287` | Automatic ORT download supports Linux x64, macOS x86_64/arm64, and Windows x64 only. Other Linux/Windows architectures hit an unsupported or wrong-target path. | Make the supported target triples explicit in documentation and errors; add archives/checksums for required arm64 targets before claiming those targets. |
| 🟡 | ORT integrity | `crates/onnx-genai-ort/ort-sys/build.rs:22-35,277-282,290-300` | macOS x86_64 is selectable but has no pinned checksum, so the build warns and continues without archive verification. | Pin the official macOS x86_64 digest and make a missing checksum a hard, actionable build error for every release target. |
| 🟡 | Python paths | `crates/onnx-runtime-python/src/lib.rs:266-288` | Python path-like values are converted through `str` into a Rust UTF-8 `String`. This is fragile for Windows paths that are not representable as Unicode and does not use Python's filesystem-path protocol directly. | Use `os.fspath`/PyO3 path extraction and retain an `OsString`/`PathBuf` through the Rust boundary where possible. Include the rejected path representation in errors. |
| 🟡 | Developer-specific test path | `crates/onnx-runtime-loader/tests/loader.rs:755-769` | A test falls back to `/home/justinchu/...`. It skips when absent, so it does not fail other OSes, but it hides coverage everywhere except one developer machine. | Require `BERT_TOY_MODEL` for an explicitly ignored real-model test, or use a repository fixture. Do not encode a user home directory. |
| 🟡 | Native CPU dependency | `crates/onnx-runtime-cpuinfo/build.rs:4-31`; `crates/onnx-runtime-cpuinfo/src/lib.rs:115-146` | cpuinfo is built unconditionally with CMake and bindings are generated with libclang. The C library is cross-platform, but no Windows/macOS CI proves the toolchain, static library names, or generated bindings. | Cover it in the OS matrix. Prefer checked-in/version-pinned bindings if practical, and make missing CMake/libclang errors identify the dependency and installation/feature workaround. |
| 🟡 | Benchmark host metadata | `crates/onnx-genai-bench/src/bin/compare.rs:637-648` | Core/OS collection assumes `sysctl`, `getconf`, and `uname`. Windows runs continue, but reports lose useful machine metadata as `unknown`. | Use `std::thread::available_parallelism` and `std::env::consts`; add a Windows-specific version query only when needed. |
| 🟢 | Loader mmap | `crates/onnx-runtime-loader/src/weights.rs:9-13,56-69`; `crates/onnx-runtime-loader/src/epcontext.rs:241-250` | Weight and EPContext mapping use `memmap2::Mmap`, not raw `libc::mmap`; the production loader contains no Unix-only mmap/page-size calls. | Keep `memmap2`; add native Windows/macOS loader tests to prove file locking, mapping lifetime, and deletion semantics. |
| 🟢 | Path traversal handling | `crates/onnx-runtime-loader/src/pathsafe.rs:1-20,40-55` | External-data paths use `Path` components and explicitly test Unix absolute paths and Windows rooted/prefixed paths. | Retain this pattern and add a Windows drive/UNC regression case in the Windows CI lane. |
| 🟢 | Temporary files | `crates/onnx-runtime-capi/tests/capi.rs:122-125`; `crates/onnx-runtime-loader/tests/writer.rs:1-9,44`; `crates/onnx-genai-server/src/models_config.rs:272-335` | Audited temporary outputs use `CARGO_TARGET_TMPDIR` or `tempfile`; no production code writes to a literal `/tmp`. | Keep using `tempfile`/`std::env::temp_dir` and path APIs. |
| 🟢 | Generated filenames | `crates/onnx-runtime-loader/src/writer.rs:340-379`; `crates/onnx-runtime-loader/tests/writer.rs:428-442` | EP sidecar components replace unsafe characters and tests reject both Unix and Windows separators. A repository path scan found no case-fold collisions or Windows-invalid tracked names. | Extend the test with Windows reserved basenames (`CON`, `NUL`, `COM1`) if user-controlled names can ever become the complete filename stem. |
| 🟢 | CRLF handling | `crates/onnx-genai-ort/ort-sys/build.rs:343-349,396-410`; `crates/onnx-runtime-tracer/tests/collectors.rs:125-131` | Text parsing uses `str::lines()` or semantic parsers; no production parser was found that requires LF-only input. Protocol output intentionally uses the protocol-required newline form. | Keep semantic parsing; add CRLF fixtures when introducing line-oriented configuration formats. |
| 🟢 | Endianness | `crates/onnx-runtime-loader/src/weights.rs:35-39,192-232`; `crates/onnx-runtime-eager/src/tensor.rs:174-183,244-253`; `crates/onnx-runtime-session/src/tensor.rs:165-215` | Tensor storage is explicitly little-endian. This is correct for ONNX data and all currently intended Windows/macOS/Linux hardware, but would need conversion on a big-endian target. | Document little-endian as the supported host contract or add a target-endian conversion layer before adding big-endian targets. |
| 🟢 | macOS CPU default | `crates/onnx-runtime-python/Cargo.toml:26-39`; `crates/onnx-runtime-python/src/lib.rs:53-60,255-264` | CUDA is optional and the default Python provider is CPU, so a normal macOS CPU build does not import CUDA code. | Preserve feature separation; fix runtime probing/advertising before shipping any CUDA-enabled wheel. |

## Prioritized fix checklist

### P0 — `xplat-dlopen-oses`

- [ ] Centralize target-specific CUDA library candidates and absolute search
  paths. Cover driver (`libcuda.so.1` / `nvcuda.dll`), cuBLASLt
  (`libcublasLt.so.13` / `cublasLt64_13.dll`), NVRTC
  (`libnvrtc.so.13` / `nvrtc64_130_0.dll`), and CUPTI
  (`libcupti.so.13` / CUDA-13 `cupti64_*.dll`).
- [ ] Search pip NVIDIA wheel directories: `nvidia/<component>/lib` on Linux and
  `nvidia/<component>/bin` on Windows. Do not expect CUDA on macOS.
- [ ] Make every loader fallible. Missing libraries, missing symbols, driver
  mismatch, or no device must never panic or break Python import.
- [ ] Make CUDA availability a runtime fact, not a Cargo-feature fact. Wire the
  selected CUDA EP into Python sessions; never report CUDA while executing CPU.
- [ ] Add RULES.md #1 diagnostics listing the missing component, searched
  names/paths, the matching `nvidia-*-cu13` package where applicable, the NVIDIA
  driver prerequisite, and the CPU fallback.

### P0/P1 — `xplat-ci-wheel-matrix`

- [ ] Convert Rust CI to an OS/architecture matrix. At minimum: Ubuntu x86_64,
  Windows x86_64 MSVC, macOS x86_64, and macOS arm64.
- [ ] Exercise the default CPU workspace everywhere. Add focused oneDNN Windows
  coverage and CUDA compile/discovery tests on Linux/Windows without requiring a
  GPU; assert CPU-only behavior on macOS.
- [ ] Configure cibuildwheel for manylinux, Windows, macOS x86_64, and macOS
  arm64. Build `cp310-abi3` plus the separately specified `abi3t` artifacts.
- [ ] Produce CPU wheels for all three OSes and CUDA wheels/extras only for
  Linux/Windows. Declare the complete CUDA-13 NVIDIA wheel dependency set with
  platform markers and the correct `lib`/`bin` discovery behavior.
- [ ] Smoke-test each wheel in a clean environment: import, CPU inference,
  `get_available_providers`, explicit unavailable-CUDA error, and CUDA discovery
  where a GPU runner is available.
- [ ] Replace Windows `unzip` dependence and pin every ORT archive checksum.
- [ ] Publish wheels with PyPI Trusted Publishing; retain the stable ABI policy
  in `docs/PIPELINE.md` rather than multiplying wheels per CPython minor.

### P2 — portability hardening

- [ ] Replace the developer-specific real-model fallback with an opt-in
  environment variable or repository fixture.
- [ ] Preserve non-Unicode filesystem paths across the Python/Rust boundary.
- [ ] Improve Windows benchmark host metadata.
- [ ] Add Windows UNC/drive/reserved-name tests, macOS case-insensitive
  filesystem tests, CRLF fixtures, and mmap lifecycle tests.
