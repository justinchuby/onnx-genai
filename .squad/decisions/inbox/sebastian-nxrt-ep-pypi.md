### 2026-07-30: nxrt-ep-cpu / nxrt-ep-cuda PyPI packaging + publish pipeline

**By:** Sebastian

**What:**
- Added two PyPI packages under `python/`:
  - `python/nxrt-ep-cpu/` → dist `nxrt-ep-cpu`, import `nxrt_ep_cpu`, bundles
    `libonnx_runtime_ep_cpu_plugin.{so,dll,dylib}`.
  - `python/nxrt-ep-cuda/` → dist `nxrt-ep-cuda`, import `nxrt_ep_cuda`, bundles
    the CUDA plugin cdylib built WITH `--features cuda` (CUDA 13). Marked
    experimental/pre-release (EP not hardware-validated, #768).
- Both expose `get_library_path() -> str` (absolute path to the bundled cdylib)
  and `register(session_options=None)` (thin wrapper over onnxruntime's
  `register_execution_provider_library`; module-level API confirmed present in
  onnxruntime 1.28.0, SessionOptions-scoped path also supported).
- Added workflow `.github/workflows/publish-ep-plugins.yml` (environment `pypi`).

**Build backend chosen: setuptools + plain `cargo` (route b), NOT maturin.**
Why: the plugin crates are C-ABI cdylibs (`CreateEpFactories`/`ReleaseEpFactory`),
not PyO3 extensions. A maturin dual-purpose module would require forcing the
`#[no_mangle]` factory symbols to survive linking alongside `PyInit_*` — fragile.
Instead `setup.py`'s `build_py` runs `cargo build --release -p <crate>` (locating
the workspace root by walking up for `[workspace]`), copies the cdylib into the
package, and a custom `bdist_wheel` tags it `py3-none-<platform>` (the cdylib has
no CPython ABI dependency, so one wheel per platform serves all Pythons). A thin
pure-Python `__init__` provides the API. auditwheel (Linux) / delvewheel (Windows)
repair the wheels. Validated locally: both wheels build, install into a fresh
venv, and `get_library_path()` returns an existing file; `register()` succeeded
against onnxruntime 1.28.0. CUDA plugin compiled locally with `--features cuda`.

**Package layout:** standalone `python/<pkg>/` dirs (src layout). The plugin
crates are already workspace members, so this does NOT churn the Cargo workspace.

**Workflow gating (independent CPU/CUDA):** the `cpu` and `cuda` build jobs and
their `publish-cpu`/`publish-cuda` jobs are fully independent (no cross-`needs:`).
PyPI won't allow two pending trusted publishers from one repo at once, so
`nxrt-ep-cpu` publishes first; `publish-cuda` never runs unless `publish_cuda=true`
(or an `nxrt-ep-v*` tag), letting the user register the cuda publisher later.
Trusted Publishing (OIDC, `id-token: write`, `environment: pypi`,
`pypa/gh-action-pypi-publish`, no tokens). `workflow_dispatch` inputs
`publish_cpu`/`publish_cuda`/`testpypi` (TestPyPI dry-run route) + `push` tags
`nxrt-ep-v*`. CUDA built inside `nvidia/cuda:13.0.0-devel-ubi9` manylinux image.

**Why:** ship the ORT plugin EPs as installable wheels that expose the cdylib
path for `RegisterExecutionProviderLibrary`, with a publish pipeline matching the
exact filename/environment the user is registering PyPI trusted publishers against.
