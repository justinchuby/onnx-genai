# Reserving the `onnx-runtime-*` Crates

All twelve crates use version `0.1.0-dev.0`. Their internal dependencies are
exact-pinned to that version, so they must be published in dependency order:

1. `onnx-runtime-ir`
2. `onnx-runtime-cpuinfo`
3. `onnx-runtime-tracer`
4. `onnx-runtime-shape-inference`
5. `onnx-runtime-loader`
6. `onnx-runtime-optimizer`
7. `onnx-runtime-ep-api`
8. `onnx-runtime-ep-cpu`
9. `onnx-runtime-ep-cuda`
10. `onnx-runtime-eager`
11. `onnx-runtime-session`
12. `onnx-runtime-capi`

`onnx-runtime-cpuinfo` and `onnx-runtime-tracer` are **foundational** crates
like `onnx-runtime-ir`: they have **no internal (`onnx-runtime-*`)
dependencies**, so they can be published early, before anything depends on
them.

Authenticate with either `cargo login <token>` or by setting
`CARGO_REGISTRY_TOKEN`. Then run:

```sh
cargo publish -p onnx-runtime-ir
cargo publish -p onnx-runtime-cpuinfo
cargo publish -p onnx-runtime-tracer
cargo publish -p onnx-runtime-shape-inference
cargo publish -p onnx-runtime-loader
cargo publish -p onnx-runtime-optimizer
cargo publish -p onnx-runtime-ep-api
cargo publish -p onnx-runtime-ep-cpu
cargo publish -p onnx-runtime-ep-cuda
cargo publish -p onnx-runtime-eager
cargo publish -p onnx-runtime-session
cargo publish -p onnx-runtime-capi
```

Wait for each version to appear in the crates.io index before publishing the
next crate that depends on it; propagation can take a short time.

> **`onnx-runtime-python`** is intentionally **not** in this list. It is a
> `cdylib` PyO3 extension (`publish = false`) distributed as the `nxrt` **wheel
> on PyPI** via `maturin`, not as a crate on crates.io — see
> `crates/onnx-runtime-python/README.md`.

`onnx-runtime-shape-inference` has a test-only dependency on
`onnx-runtime-loader` that is intentionally path-only, with no version. Cargo
omits that dev-dependency from the published manifest, breaking the otherwise
cyclic publish requirement while preserving local tests. This is why shape
inference can be published before the loader in the order above.

## `onnx-runtime-ep-cuda` — publish & build considerations

`onnx-runtime-ep-cuda` (the CUDA execution provider, `docs/ORT2.md` §15) only
depends on `onnx-runtime-ir` and `onnx-runtime-ep-api` among the internal
crates, so it publishes any time after `onnx-runtime-ep-api` (it is placed right
after `onnx-runtime-ep-cpu` above, mirroring the CPU EP). Its one notable
external dependency is **`cudarc`** (pinned to the `0.19` line), used with
`default-features = false` and the features
`["std", "driver", "cublaslt", "f16", "cuda-13000", "dynamic-loading"]`:

- **`dynamic-loading`** means the crate **builds with no CUDA toolkit present** —
  the CUDA driver and cuBLASLt are `dlopen`'d at *runtime*, not linked at build
  time. `cargo publish`/`cargo build` therefore work on a plain host (e.g.
  docs.rs, CI without a GPU). Consumers only need the CUDA runtime libraries
  (`libcuda`, `libcublasLt`) on the loader path (`LD_LIBRARY_PATH`) to actually
  execute on a GPU; without them the crate still compiles and its GPU tests
  self-skip.
- **`cuda-13000`** pins the cuBLASLt/driver API surface to the CUDA 13.0 headers
  (matching the target hosts' `libcublasLt.so.13`). Bumping the target CUDA
  version is a one-line feature change; it does **not** require a toolkit because
  of `dynamic-loading`. (Do **not** switch to `cuda-version-from-build-system`,
  which would reintroduce a build-time CUDA dependency.)
- There is **no `build.rs` and no `nvcc` step** in this crate — the Phase-2a
  slice is cuBLASLt GEMM only. When Phase-2b custom `.cu`/CuTe kernels land they
  will introduce a compile step and a real build-host CUDA requirement; that
  will change the publish story and must be revisited here.

## `onnx-runtime-ep-cpu` — `onednn` feature & publish considerations

`onnx-runtime-ep-cpu` publishes as a **pure-Rust, offline** crate by default: its
default GEMM backend is the Generic blocked/register-tiled, rayon-parallelized
kernel (`docs/ORT2.md` §25.2), so `cargo publish`/`cargo build` need no external
library, C++ toolchain, or network beyond crates.io.

The non-default **`onednn`** feature links oneDNN statically for the x86 / ARM-server
CPU GEMM path. It is intentionally **excluded from the default published surface**:

- It requires the `third_party/onednn` git submodule (pinned to `v3.9.2`), which
  crates.io cannot fetch — published `.crate` tarballs do not carry submodules.
  Consumers who want the feature build from a git/workspace checkout with the
  submodule initialized (`git submodule update --init --depth 1 third_party/onednn`).
- Enabling it activates the optional `cmake`/`bindgen` build-dependencies and a
  `build.rs` source build of oneDNN (static, CPU-only). With the feature off,
  those build-dependencies are never compiled, keeping the default build offline.
- Do **not** make `onednn` a default feature; doing so would break `cargo publish`
  and any consumer that installs ep-cpu from crates.io.

For a real release, replace each crate's explicit `0.1.0-dev.0` version with
the chosen stable version, update all twelve workspace dependency pins to the
same exact version, rebuild, and publish again in this order. Published
prerelease versions are immutable and remain on crates.io.
