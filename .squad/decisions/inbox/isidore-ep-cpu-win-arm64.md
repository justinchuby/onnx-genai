# Decision: publish a Windows ARM64 wheel for nxrt-ep-cpu

**Author:** Isidore (Mobile & Bindings Engineer)
**Date:** 2026-08-12
**Branch:** squad/ep-cpu-win-arm64

## Context
Justin: "我们的ep也要发 windows arm 64的wheel" — publish a Windows ARM64
(`win_arm64`) wheel for the EP plugin. Only `nxrt-ep-cpu` is in scope; the CUDA
EP is not (CUDA is unavailable on Windows ARM64), so the `cuda` job is untouched.

## Change
`.github/workflows/publish-ep-plugins.yml` — added one matrix row to the **cpu**
job:

```yaml
- name: Windows ARM64
  os: windows-11-arm      # GitHub-hosted Windows 11 ARM64 runner (GA)
  archs: ARM64
  cibw_build: cp311-*
  python_version: "3.11"
```

Supporting wiring:
- New job `env`: `CIBW_BUILD: ${{ matrix.cibw_build || 'cp310-*' }}`. The three
  existing rows have no `cibw_build`, so they keep building `cp310-*`; the ARM64
  row builds `cp311-*`.
- `actions/setup-python` now uses `python-version: ${{ matrix.python_version || '3.10' }}`
  so only the ARM64 row bumps to 3.11 (the other rows stay on 3.10).

## cp310 → cp311 rationale (the critical caveat)
Official CPython has **no `win_arm64` build for 3.10** — ARM64 Windows CPython
starts at **3.11**. The global `build = "cp310-*"` selector in
`python/nxrt-ep-cpu/pyproject.toml` would therefore match **nothing** on the
ARM64 runner and the job would produce no wheel / fail. Because the wheel is
ABI-less (`setup.py` tags it `py3-none-win_arm64`; it bundles a plain C-ABI
cdylib `onnx_runtime_ep_cpu_plugin.dll` with no CPython ABI dependency), any
single CPython ≥3.11 is sufficient to drive the build. So we override
`CIBW_BUILD` to `cp311-*` for the ARM64 row only, and likewise pin setup-python
to 3.11 there (3.10 has no arm64 build to install). cibuildwheel then provisions
its own CPython for the actual build; setup-python only runs cibuildwheel itself.

## Other verifications
- **Runner label** `windows-11-arm` confirmed against cibuildwheel docs (GA,
  used in their example matrix alongside `ubuntu-24.04-arm`).
- **Native build:** `rustup toolchain install stable` on `windows-11-arm`
  installs the native `aarch64-pc-windows-msvc` host toolchain → native aarch64
  build, no cross-compilation. rustup ships preinstalled on the runner.
- **delvewheel repair** kept as-is; `--exclude ext-ms-win-dxcore-l1-1-0.dll` is a
  Windows API set (fine on ARM64).
- **Artifact plumbing:** upload artifact name `nxrt-ep-cpu-windows-11-arm-ARM64`
  is unique; `publish-cpu` downloads `pattern: nxrt-ep-cpu-*` with
  `merge-multiple: true`, so it is picked up automatically.
- Updated the `build = "cp310-*"` comment in `pyproject.toml` and the Platforms
  section of `python/nxrt-ep-cpu/README.md` (now lists Windows ARM64).

## Open risk for the first CI run
- Confirm `windows-11-arm` is available in this repo's runner allotment and that
  rustup is preinstalled (add an explicit rustup bootstrap step if not).
- Verify cibuildwheel v4.1.0 actually provisions a `win_arm64` CPython for
  `cp311-*` (rather than only host-arch); if it can't, pin to `cp312-*`.
- Validate via a build-only dispatch (`publish_cpu=false`) before publishing.
