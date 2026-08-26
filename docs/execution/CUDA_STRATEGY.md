# CUDA EP Strategy — Current Library-First Architecture

- **Crate:** `onnx-runtime-ep-cuda`
- **Build selector:** cudarc `cuda-13000`, dynamic loading
- **Supported development runtime:** the CUDA 13.1 package set pinned in
`requirements-cuda-dev.txt`

`docs/execution/CUDA_COVERAGE.md` is the operator-level source of truth. This
document records the backend policy, what has landed, and the remaining work.

## 1. Policy

Use vendor libraries for architecture-sensitive primitives and NVRTC only when
there is no suitable runtime library or fusion removes meaningful memory
traffic:

| Work | Current backend | Policy |
|------|-----------------|--------|
| Dense and batched GEMM, fused GEMM epilogues | cuBLASLt | Vendor library. |
| DFT / STFT | cuFFT plus NVRTC packing | Vendor transform with custom layout glue. |
| 2-D Conv, MaxPool, AveragePool | cuDNN | Vendor library; **landed**, not future work. |
| Softmax, ReduceSum, ReduceMean | cuDNN with bounded fallback where documented | Vendor library. |
| ConvTranspose, extended/global pooling and normalization subsets | NVRTC | Current implemented subsets; move only when a library route preserves ONNX semantics and wins. |
| Pointwise, comparison, logical, cast, indexing, movement | NVRTC / D2D copy | No useful dlopen-able library primitive; JIT targets the running device. |
| Fused norm/residual, RoPE, recurrent attention/state | NVRTC | Retained fusion or no-library case. |
| Attention | tiled NVRTC/cuBLASLt cores | Current implementation; further SDPA tuning remains. |

The old plan described adding cuDNN and implementing Conv/pooling in the future.
That work has landed: cudarc enables `cudnn`, `CudaRuntime` owns a lazy
stream-bound handle, and the coverage matrix records the supported Conv,
pooling, softmax, and reduction surfaces.

## 2. Current versus future work

### Landed

- cuBLASLt `MatMul`, `Gemm`, `FusedMatMulBias`, and `FusedGemm`.
- cuDNN 2-D `Conv`, `MaxPool`, `AveragePool`, standalone `Softmax`,
  `ReduceSum`, and `ReduceMean`.
- cuFFT `DFT` and `STFT`.
- Fused CUDA attention cores for `Attention`, plus implemented
  `MultiHeadAttention`, `GroupQueryAttention`, varlen, sparse, and recurrent
  variants documented in `CUDA_COVERAGE.md`.
- Runtime wheel discovery across system, Python-wheel, and Conda layouts.
- Toolkit-free compilation through cudarc `dynamic-loading`.

### Remaining

1. Register/lower `com.microsoft::FusedAttention` onto the implemented attention
   core, then tune long-context and device-tier routing with measured gates.
2. Move another NVRTC primitive to cuDNN only when ONNX attribute/NaN/layout
   parity is proven and the vendor route wins on the target tiers.
3. Improve pointwise-chain and producer/consumer fusion where it removes
   launches or HBM round-trips.
4. Keep runtime-library diagnostics actionable and the Python dependencies
   synchronized with the libraries actually dlopened.

`com.microsoft::MoE` and `PackedMultiHeadAttention` are documented non-gaps, not
kernel work items. See `CUDA_COVERAGE.md` for their evidence and the condition
that would reopen either decision.

## 3. Build and runtime acquisition

The Rust build has no `.cu` compilation, `nvcc`, toolkit headers, or CUDA
`build.rs` step. cudarc resolves the driver and vendor libraries dynamically;
NVRTC compiles embedded device sources at runtime.

The reproducible developer line is:

```powershell
python -m pip install -r requirements-cuda-dev.txt
$env:NXRT_CUDA_WHEEL_ROOTS = "<site-packages directory containing nvidia>"
```

The pinned CUDA 13.1 set includes:

| Runtime library | Package |
|-----------------|---------|
| cudart + NVRTC headers | `nvidia-cuda-runtime==13.1.80` |
| cuBLAS / cuBLASLt | `nvidia-cublas==13.1.1.3` |
| cuFFT | `nvidia-cufft==12.1.0.78` |
| nvJitLink | `nvidia-nvjitlink==13.1.115` |
| NVRTC | `nvidia-cuda-nvrtc==13.1.115` |
| CUPTI tracing | `nvidia-cuda-cupti==13.1.115` |
| cuDNN | `nvidia-cudnn-cu13==9.24.0.43` |

The NVIDIA driver remains a host prerequisite. The `nxrt-ep-cuda` wheel pins
its runtime dependencies to this validated line; it does not follow whatever
newer rolling CUDA 13 packages happen to be latest.

### Compatibility discovery is not the development line

The loader also probes older sonames/DLL names and per-component CUDA 12 wheel
layouts. That is intentional compatibility discovery for deployments that
already provide compatible libraries. It is not a recommendation to install
CUDA 12 packages, and it does not change the CUDA 13.1 build/development line.

## 4. Library discovery

For each component, discovery tries:

1. the system loader name;
2. explicit roots from `NXRT_CUDA_WHEEL_ROOTS`;
3. Python `site-packages` NVIDIA layouts;
4. Conda/virtual-environment library directories.

On Windows, CUDA 13 wheels place their shared runtime under
`nvidia\cu13\bin\x86_64` and headers under `nvidia\cu13\include`; cuDNN remains
under its component directory. On Linux, discovery covers the corresponding
wheel `lib` directories and system sonames.

Missing-library errors must name the component, attempted paths, and the exact
CUDA 13 package that supplies it. Optional compatibility probes may fail
quietly; executing an advertised kernel may not.

## 5. Backend decision rules

- **cuBLASLt/cuDNN/cuFFT first** for heavy primitives they implement with the
  required ONNX semantics.
- **Keep NVRTC** for generic pointwise/indexing/movement work and proven fused
  kernels. NVRTC JITs to the actual device and does not require `nvcc`.
- **Fail closed** on unsupported dtype, rank, attribute, layout, bound, or
  capture mode. Never claim an op and silently change semantics.
- **Measure before moving a route.** Same-artifact, same-dtype parity and
  device-tier performance evidence are required; Hopper-only wins are not a
  universal policy.
- **Keep docs derived.** Operator registration and coverage counts come from
  the registry/tests described in `CUDA_COVERAGE.md`, not from copied totals.
