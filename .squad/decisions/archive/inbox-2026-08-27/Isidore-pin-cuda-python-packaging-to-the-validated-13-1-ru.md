### 2026-08-26T05-05-57: Pin CUDA Python packaging to the validated 13.1 runtime line
**By:** Isidore
**What:** Pin CUDA Python packaging to the validated 13.1 runtime line
**References:** PR #2181, requirements-cuda-dev.txt, python/nxrt-ep-cuda/pyproject.toml
**Why:** The nxrt-ep-cuda wheel now pins the same CUDA 13.1 releases as requirements-cuda-dev.txt instead of accepting any rolling CUDA 13 package. This prevents newer NVRTC/nvJitLink output from outrunning the validated driver line, includes landed cuDNN runtime coverage, and keeps builds toolkit/nvcc-free through dynamic loading.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
