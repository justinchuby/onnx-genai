# ARM64 Debug CI Failure on PR #31973

**Date:** 2026-08-11
**Author:** Batty (Systems Engineer)
**PR:** microsoft/onnxruntime#31973 (`nxrt/mlas-avx2-layernorm`)

## Verdict: STILL UNKNOWN — strong OOM circumstantial evidence, but not conclusively proven

### Root Cause Analysis

The single failing job is **Build Linux arm64 Debug / build_test_pipeline** (job ID `93705050460`, run `31468051681`).

**What the log shows:**

1. The CMake generation step (`--update`) completed successfully:
   ```
   2026-08-11T07:14:53.9748Z  build [INFO] - Build complete
   2026-08-11T07:14:56.3498Z  Docker command executed successfully.
   ```

2. The BUILD step (`--build`) starts at 07:14:56 and ninja compiles from `[1/1458]` to `[1452/1458]` with **zero compiler errors**.

3. At `[1452/1458]` (07:18:45), the last completed target is `Linking CXX executable onnxruntime_mlas_test`. Then **29 seconds of silence** followed by `Post job cleanup` (07:19:14).

4. **Missing completion markers:** No `Docker command executed successfully` for the BUILD step. No `Process completed with exit code`. No `FAILED:`. No `ninja: build stopped`. No `Killed`.

5. **The 6 uncompleted targets (1453–1458)** are the large test executable link steps — `onnxruntime_test_all`, `onnxruntime_provider_test`, `onnxruntime_autoep_test`, `onnxruntime_ep_graph_test`, and 2 others. These are the largest link targets in the build.

**Why OOM is the leading hypothesis:**

- VM SKU is `Standard_D8pds_v5` (8 vCPUs, 32 GB RAM)
- CCache missed (`Cache not found for input keys: ccache`) — all targets compiled fresh
- The 6 remaining targets are all large Debug executable link steps that run in parallel
- Debug mode (`-O0`, full DWARF info) produces object files 3–5× larger than Release
- Silent process death (no error, no exit code) is the hallmark of OOM-kill (SIGKILL)
- The Docker container ran with `--rm`; when OOM-killed, no output is captured

**Why not conclusively proven:**

- No explicit OOM signal (no "Killed", no exit code 137, no dmesg access)
- Other unrelated PRs (#31972, #31971, #31970) pass arm64 Debug on the same runner pool
- Cannot rule out a runner-specific transient issue (e.g., residual memory pressure from prior jobs)
- Only one run observed (previous pushes have no Linux_CI runs in the API)

### Code Review: No Bug Found

Verified all ARM64/x86 guards:

- **`mlasi.h:1415`** — `MlasLayerNormKernelAvx2` declaration is inside `#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_IX86)` ✓
- **`platform.cpp:515`** — `LayerNormF32Kernel = &MlasLayerNormKernelAvx2` is inside the AMD64 CPUID detection block ✓
- **`cmake/onnxruntime_mlas.cmake:863`** — `layernorm_kernel_avx2.cpp` is in the x86_64 `mlas_platform_srcs_avx2` list ✓
- **`cmake/onnxruntime_mlas.cmake:258`** — same for MSVC path ✓
- No ARM64 code references AVX2 layernorm symbols

### What Changed, If Anything

**No code changes made.** No fix is warranted — there is no demonstrated code bug.

### Recommendations

1. **Re-trigger the CI** (push an empty commit or use `gh workflow run`) to test flakiness. If it passes on retry, this is a transient OOM.
2. If it consistently fails, consider requesting the onnxruntime team increase the runner SKU for arm64 Debug, or add `--parallel 4` to reduce concurrent link parallelism.
3. An alternative workaround: add `-DCMAKE_JOB_POOLS:STRING="link_pool=2"` and `-DCMAKE_JOB_POOL_LINK=link_pool` to limit parallel link steps.

### What Remains Unverifiable

- Cannot run ARM64 binaries on this x86-64 host
- Cannot access runner dmesg/kernel logs to confirm OOM-kill
- Cannot re-run the CI job from a fork PR (`gh run rerun` is blocked)
- Only one run available; previous push runs are not in the API
