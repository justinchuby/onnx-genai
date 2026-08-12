### 2026-08-12: Fix nxrt-ep-cuda wheel CI — manylinux image, no CUDA toolkit needed

**By:** Sebastian

**What:**
- `.github/workflows/publish-ep-plugins.yml` `cuda` job:
  - `CIBW_MANYLINUX_X86_64_IMAGE`: `nvidia/cuda:13.0.0-devel-ubi9` →
    `quay.io/pypa/manylinux_2_28_x86_64:2026.07.25-1` (same image as the cpu job).
  - Removed `CIBW_MANYLINUX_X86_64_MOUNT_HOST_DXCORE` (irrelevant).
  - Kept the `auditwheel repair --exclude libcud*` command as a defensive no-op
    (nothing is actually linked; see below) and updated comments accordingly.
  - Renamed the image-pull step; updated build-step comment.
- `python/nxrt-ep-cuda/pyproject.toml` `[tool.cibuildwheel.linux].before-all`:
  UNCHANGED (`clang-devel` + Rust, same as cpu). Only the comment was updated to
  explain no CUDA toolkit is required.
- CPU job/package untouched. `cuda`/`publish-cuda` stay independent from cpu.

**Why the nvidia base image failed:** `nvidia/cuda:13.0.0-devel-ubi9` is NOT a
manylinux image — it lacks cibuildwheel's `/opt/python/*` CPython toolchains, so
cibuildwheel aborted with "'/opt/python/cp310-cp310/bin/python' executable
doesn't exist in image". The standard PyPA manylinux image has those builds.

**Why NO CUDA toolkit is installed (deviation from the proposed before-all
toolkit install):** `onnx-runtime-ep-cuda` binds CUDA through cudarc's
`dynamic-loading` feature and `onnx-genai-cuda-version-guard` has no build.rs —
so `cargo build -p onnx-runtime-ep-cuda-plugin --features cuda` needs no nvcc,
no CUDA headers, and no GPU; libcuda/cuBLASLt/nvrtc/cupti are dlopen'd at
runtime. Installing `cuda-toolkit-13-0` from the NVIDIA rhel8 repo would add a
multi-GB download and a repo/package-name fragility point (a risk the task
itself flagged) for zero build benefit. Verified empirically (see below), so the
before-all matches the cpu job (clang + rust only).

**Validation (containerized, on this host via `sudo docker`):** built the plugin
and the wheel inside `quay.io/pypa/manylinux_2_28_x86_64:2026.07.25-1` with NO
CUDA toolkit / no nvcc present:
- `cargo build --release -p onnx-runtime-ep-cuda-plugin --features cuda` → OK,
  exports `CreateEpFactories` / `ReleaseEpFactory`.
- `python -m build --wheel` → `nxrt_ep_cuda-...-py3-none-linux_x86_64.whl`.
- `readelf -d` on the bundled `.so`: NEEDED = libdl, libgcc_s, libpthread, libm,
  libc, ld-linux — **no CUDA libraries linked** (confirms dynamic-loading and
  that the auditwheel excludes are no-ops).
- `auditwheel repair` → retagged `py3-none-manylinux_2_28_x86_64`.
- `pip install` + `python -c "import nxrt_ep_cuda; get_library_path()"` → OK.

**Risks to watch when CI runs:** none related to a CUDA repo (we don't touch it).
The wheel is CUDA-runtime-free by construction; the nvidia-*-cu13 pip deps
satisfy the runtime libs. Hardware validation of the EP itself remains open
(#768) — the CI build is compile+import only, no GPU session.

**NVIDIA runtime deps (Justin: "cuda要包含需要的nvidia pypi包当dependency"):**
Kept the four NVIDIA CUDA-13 runtime libs as REQUIRED `[project].dependencies`
(NOT an optional extra), so `pip install nxrt-ep-cuda` pulls them automatically —
they provide the libs the cdylib dlopen's (libcudart / libcublas(+Lt) / libnvrtc
/ libcupti) which auditwheel excludes from the wheel. Tightened `>=13` → `>=13,<14`
to lock the CUDA 13 major so a future CUDA 14 under the same rolling package name
can't be pulled against our CUDA-13-built cdylib. Uses the UNSUFFIXED names,
which (verified on PyPI 2026-08-12) ARE the real CUDA 13 wheels
(nvidia-cuda-runtime 13.3.29, nvidia-cublas 13.6.1.10, nvidia-cuda-nvrtc 13.3.33,
nvidia-cuda-cupti 13.3.75); the `-cu13`-suffixed names are only 0.0.1 stubs and
are deliberately avoided. Final list:
`nvidia-cuda-runtime>=13,<14`, `nvidia-cublas>=13,<14`,
`nvidia-cuda-nvrtc>=13,<14`, `nvidia-cuda-cupti>=13,<14`.
