### 2026-08-25: Production GPU tests get compile/inventory CI, not false execution
**By:** Gaff
**What:** The Linux CUDA lane now compiles and inventories `onnx-runtime-ep-cuda-plugin/cuda_unique_ort_e2e` with `cuda`, and inventories `onnx-runtime-session/hetero_cuda_gpu` both with and without `gpu-tests`; the base hetero test remains present and ignored. GitHub-hosted runners do not execute either GPU-enabled path. The required fast lane explicitly runs a textproto fixture census with a >200 floor, six cross-tree sentinels, and zero binary `.onnx` files.
**Why:** Hosted CI has no GPU, so compilation/listing is the strongest honest enforcement available there. Running and silently skipping hardware paths would overstate coverage; sentinels plus a floor prevent a broken fixture root/pathspec from passing vacuously.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
