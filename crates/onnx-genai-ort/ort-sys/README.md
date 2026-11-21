# ONNX Runtime bootstrap

The default native runtime is **ONNX Runtime 1.32.0**, the PyPI stable release
verified on 2026-11-20. The build downloads a pinned official wheel directly from
`files.pythonhosted.org`, verifies its SHA-256 before extraction, and copies only
native libraries from `onnxruntime/capi` into the Cargo build cache. It does not
install or import the Python package. The `cp311` wheel tag is the artifact's
packaging tag; using its standalone C library does not require Python 3.11.

Matching, unmodified C API 32 headers are bundled in
[`third_party/onnxruntime`](third_party/onnxruntime/README.md), with Microsoft's
license, source-package provenance and digests. This makes automatic builds
independent of the upstream GitHub repository's availability.

Automatic targets:

| OS | Architectures | Artifact baseline |
|---|---|---|
| Linux | x86_64, aarch64 | manylinux_2_28 |
| macOS | aarch64 | macOS 14 |
| Windows | x86_64, aarch64 | official CPU wheel |

Unsupported target pairs fail with an explicit `ORT_ROOT` remedy; architectures
are never substituted. Artifact URLs and hashes are in `build_support.rs` and
come from `https://pypi.org/pypi/onnxruntime/1.32.0/json`.

`ORT_ROOT` or `ORT_LIB_DIR` can select an existing native installation containing
`lib/` and matching `include/` headers. The header must declare C API 32. On macOS,
an explicitly selected Metal plugin's linked ORT installation takes precedence,
as before. Python `onnx-genai` packages still discover the installed Python ORT
wheel at runtime; their CPU and GPU dependency floors are updated together.

Validation commands:

```sh
cargo test -p onnx-genai-ort-sys
cargo test -p onnx-genai-ort --test runtime_upgrade
cargo clippy -p onnx-genai-ort-sys --all-targets -- -D warnings
cargo check --workspace --all-targets
cargo check -p onnx-runtime-ep-cuda --all-targets --features gpu-tests
```

The bootstrap tests cover target selection, tampered downloads, native-only
extraction, missing libraries, invalid ZIP data and unsafe paths. Runtime tests
load the actual library, round-trip a C API status, and run a CPU Identity model.
These tests do not establish CUDA device execution or performance.
