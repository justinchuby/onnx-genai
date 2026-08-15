# Cross-platform audit (Windows / macOS / Linux)

Audit date: 2026-07-20. Scope: all 35 crate directories under `crates/`
(822 tracked crate files), Python packaging, and GitHub Actions at
`e33a0f5`. This was a read-only source audit; it does not claim native execution
on any platform.

## Intended support

| Platform | CPU | CUDA | Metal |
|---|---:|---:|---:|
| Linux x86_64 | Required | Required with NVIDIA driver | N/A |
| Windows x86_64 | Required | Required with NVIDIA driver | N/A |
| macOS x86_64 / arm64 | Required | Unsupported; CPU import must remain usable | Optional plugin; currently arm64-oriented |

Priorities mean:

- **P0:** can silently select the wrong backend or makes a supported CUDA wheel unusable.
- **P1:** blocks a supported build/release path or leaves it materially untested.
- **P2:** portability hardening, developer tooling, or divergent edge behavior.

## Findings

### P0

| File:line | Issue and affected OS | Recommended fix |
|---|---|---|
| `crates/onnx-runtime-tracer/src/cupti.rs:93-95,245-317` | CUPTI discovery is Linux-only: candidates are only `libcupti.so.13` / `libcupti.so`, and the pip layout is hardcoded to `nvidia/cuda_cupti/lib`. A Windows CUDA wheel installs DLLs under a `bin` layout and needs `cupti64_*.dll`, so tracing always reports unavailable there. macOS also tries irrelevant Linux names instead of immediately reporting unsupported. | Implement target-specific names and layouts under **`xplat-dlopen-oses`**; use `lib` on Linux, `bin` plus safe DLL-directory handling on Windows, and an unsupported result on macOS. Wheel bundling/discovery belongs to **`tracer-cupti-wheel-bundle`**; do not duplicate it here. |
| `crates/onnx-genai-ort/src/cuda_rt.rs:58-68` | The local CUDA runtime loader only names CUDA 12 (`cudart64_12.dll`, `cudart64_120.dll`, `libcudart.so.12`) while Python declares CUDA 13 dependencies (`crates/onnx-runtime-python/pyproject.toml:32-40`). A clean CUDA-13-only Linux or Windows environment cannot load this path. | In **`xplat-dlopen-oses`**, derive candidates from the selected CUDA ABI and include CUDA-13 Linux/Windows names. Share one resolver with the EP rather than maintaining a second stale list. |
| `crates/onnx-runtime-ep-cuda/src/runtime.rs:31-56`; `crates/onnx-runtime-python/src/lib.rs:1349-1390` | CUDA header/library discovery is not shared. NVRTC headers inspect `CUDA_HOME`, `CUDA_PATH`, `/usr/local/cuda/include`, and Linux `LD_LIBRARY_PATH`; Python injects real `sys.path` only into CUPTI. Linux/Windows `nvidia-*` wheels and Conda installs can therefore provide CUDA 13 successfully but remain undiscoverable to NVRTC/cuBLAS/runtime components, especially Windows `nvidia/<component>/bin`. | Under **`xplat-dlopen-oses`**, centralize NVIDIA component discovery from live Python paths, Conda, CUDA roots, and platform loader paths. Support Linux `lib`, Windows `bin`/`Library\\bin`, and actionable per-component errors. |
| `crates/onnx-runtime-python/src/lib.rs:62-72,270-345,993-1017` | A CUDA-feature build advertises `CUDAExecutionProvider`, accepts it, then constructs the same CPU-only `onnx_runtime_session::InferenceSession`; `device_to_providers` explicitly says CUDA falls back to CPU. Linux/Windows callers can silently benchmark or serve CPU while reporting CUDA. A mistakenly CUDA-enabled macOS build would advertise an impossible provider. | Make provider availability a fallible runtime fact and wire the chosen CUDA EP into session creation. Explicit CUDA requests must either execute CUDA or fail with the missing driver/library/device and CPU fallback instructions; never silently run CPU. |

### P1

| File:line | Issue and affected OS | Recommended fix |
|---|---|---|
| `.github/workflows/ci.yml:61-100,150-178`; `.github/workflows/wheels.yml:22-115` | CI now has Linux/Windows/macOS portable lanes and four CPU wheel targets, but portable tests cover only six runtime crates. The GenAI engine/server/ORT/C APIs, Python CPU behavior, `mlas-sys`, and native ORT bootstrap do not run on Windows/macOS. CUDA is compile/import-only on Linux/Windows with no GPU/provider assertion. | Complete **`xplat-ci-wheel-matrix`**: add representative native ORT + GenAI tests on every supported OS, Windows/macOS C API smoke tests, opt-in MLAS target checks, clean-wheel CPU inference, unavailable-CUDA behavior, and GPU-backed CUDA promotion tests. |
| `crates/onnx-genai-ort/ort-sys/build.rs:22-35,364-377` | Automatic ORT download maps every Linux target to `linux-x64` and every Windows target to `win-x64`; Linux/Windows arm64 therefore downloads the wrong ABI. macOS x86_64 is selected but has no pinned checksum, so the release wheel lane continues after a warning. | Match full target triples, reject unsupported architectures before download, pin the macOS x86_64 digest, and make missing checksums fatal for release targets. |
| `crates/onnx-genai-ort/ort-sys/build.rs:441-470,473-529` | The build mutates the resolved ORT installation: it creates Unix symlinks in `lib/`, rewrites macOS install names, and re-codesigns dylibs. `ORT_ROOT`/`ORT_LIB_DIR` may point to read-only or centrally managed installations, causing Linux/macOS builds to fail or unexpectedly alter vendor artifacts. The commands also use target OS to select tools that execute on the host, impeding cross-compilation to macOS. | Copy/stage required runtime files in `OUT_DIR`, create links and adjust install names only in that private staging directory, and distinguish host tools from target selection. Never mutate user-supplied ORT installations. |
| `crates/mlas-sys/build.rs:1-21,147-180`; `crates/onnx-runtime-ep-cpu/Cargo.toml:11-20` | The opt-in `mlas` feature is described as x86-64 Linux and compiles GAS `.S` files plus Unix-oriented headers/flags. Enabling it on Windows MSVC or macOS is not target-gated and will fail before the Rust fallback can run. | Gate the current implementation to its proven target with a clear compile error, then add upstream Windows MASM and macOS assembly/source groups before advertising the feature there. Add feature-specific CI. |
| `crates/onnx-genai-ort/ort-sys/build.rs:204-238` | Native ORT bootstrap shells out to `curl` on all OSes and `tar` on Linux/macOS. GitHub runners provide them, but clean developer machines and hermetic builders are not guaranteed to do so; failure occurs during dependency build. | Use a Rust HTTP/archive implementation, or document and validate the tools up front with an `ORT_ROOT` workaround. Keep the already-portable Rust ZIP extraction used on Windows. |
| `crates/onnx-runtime-python/src/lib.rs:1366-1390` | Python module and `sys.path` entries are extracted as UTF-8 `String`. Windows supports filesystem paths that Python can represent but Rust UTF-8 conversion may reject or alter; those environments then lose CUDA wheel discovery. | Extract via Python's filesystem-path protocol into `PathBuf`/`OsString` without lossy conversion. Apply the same path-preserving boundary to model loading. |

### P2

| File:line | Issue and affected OS | Recommended fix |
|---|---|---|
| `crates/onnx-runtime-loader/src/writer.rs:335-389` | EP source/partition components are sanitized, but the original model/output stem is copied verbatim into generated context and sidecar names. A file created on Linux with `:`, trailing dot/space, or a Windows-reserved stem (`CON`, `NUL`, `COM1`) cannot be recreated or unpacked on Windows. | Sanitize generated stems against the Windows superset, including reserved basenames and trailing dots/spaces; preserve the requested directory separately. |
| `crates/onnx-genai-server/src/models_config.rs:108-155` | Model discovery probes exact lowercase marker names. Case-insensitive default Windows/macOS filesystems may accept `MODEL.ONNX`, while case-sensitive Linux does not, producing deployment-dependent discovery. | Define canonical filenames explicitly and enforce them uniformly by enumerating entries, or intentionally perform ASCII case-insensitive matching on every OS with collision detection. |
| `crates/onnx-runtime-loader/tests/loader.rs:759-770`; `crates/onnx-genai-engine/src/native_decode.rs:1552` | Tests contain `/home/justinchu/...` fallbacks. They skip when absent, so other developers and all CI platforms silently lose coverage. | Make real-model tests explicitly ignored/opt-in via environment variables, with no personal fallback, or add a repository fixture. |
| `crates/onnx-runtime-session/tests/projection_fusion.rs:52-58` | The test invokes `python3`, which is normally `python.exe`/`py` on Windows even when Python is installed. The portable CI installs Python but this test can still fail when included. | Use a configured interpreter (`PYTHON`, `pyo3-build-config`, or CI-provided path) and fall back by platform with an actionable skip/error. |
| `crates/onnx-genai-bench/src/bin/compare.rs:623-635,845-880`; `crates/onnx-genai-ort/examples/long_context_bench.rs:328-341` | Benchmark metadata/RSS collection assumes Unix/macOS commands (`hostname -s`, `sysctl`, `date +...`, `pmset`, `ps`). Windows reports partial `unknown`/`NaN` metadata, reducing result comparability. | Use Rust APIs for date, hostname, CPU count, OS, and memory; isolate optional OS probes behind `cfg` modules. |
| `crates/onnx-genai-ort/ort-sys/build.rs:70-79` | Bindgen calls `header_path.to_str().unwrap()`. A non-Unicode checkout/build path panics; Windows path representations make this especially visible. | Pass an OS-native path if supported by bindgen, or return an actionable error rather than unwrapping. |
| `crates/onnx-genai-ort/ort-sys/build.rs:297-360` | ZIP extraction writes symlink entries as ordinary files and ignores Unix permission-setting failures. The current Windows ORT archive does not depend on these semantics, but reusing the helper for Unix/vendor archives would be incorrect. | Explicitly reject symlink entries or use a tested safe extractor; propagate permission failures when permissions matter. |

## Audited categories with no current blocker

- **Hardcoded separators:** fixture joins such as `Path::join("../../tests/fixtures/...")`
  use Rust `Path` parsing; `/` is accepted as a separator on Windows. URL
  construction uses `/` intentionally. The material exception is the NVIDIA
  package layout called out above.
- **Temporary paths:** production/tests use `std::env::temp_dir`, `tempfile`, or
  Cargo target directories. Literal `/tmp` occurrences found in
  `onnx-runtime-session` are test values, not writes.
- **Filename timestamps:** no production filename generation from RFC3339 or
  `%H:%M:%S` was found. The `.squad` log naming policy already replaces `:` with
  `-`.
- **CRLF/text mode:** line-oriented parsing uses `str::lines()` or semantic
  JSON/TOML/YAML parsers; Rust file writes are byte-oriented. No LF-only
  production parser was found.
- **Signals/TTY:** no production Unix signal API, raw TTY/ioctl, or
  `std::os::unix` terminal dependency was found under `crates/`.
- **Application-data homes:** production code does not invent storage beneath
  `HOME`, `USERPROFILE`, XDG, or `%APPDATA%`; paths are caller/config supplied.
- **Path traversal:** `crates/onnx-runtime-loader/src/pathsafe.rs:3-20,40-55`
  uses `Path::components` and has separate Unix and Windows rooted-path tests.
- **Dynamic names already handled:** cuDNN diagnostics name both
  `libcudnn.so.9` and `cudnn64_9.dll`
  (`crates/onnx-runtime-ep-cuda/src/error.rs:25-37`), and ORT library selection
  distinguishes `.so`, `.dylib`, and `.dll`
  (`crates/onnx-genai-ort/ort-sys/build.rs:532-557`).

## Recommended order

1. **P0:** finish `xplat-dlopen-oses`, including the stale CUDART list and shared
   pip/Conda resolver; finish `tracer-cupti-wheel-bundle`; make Python CUDA
   selection truthful.
2. **P1:** finish `xplat-ci-wheel-matrix`, then harden ORT target/checksum/staging
   behavior and explicitly target-gate MLAS.
3. **P2:** make generated filenames Windows-safe, remove personal test paths,
   preserve non-Unicode paths, and replace shell-based benchmark metadata.
