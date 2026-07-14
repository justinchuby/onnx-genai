# CUDA EP Strategy — Library-First, PyTorch-Style Zero-Setup

**Crate:** `onnx-runtime-ep-cuda` · **Backend stack:** `cudarc` 0.19
(dynamic-loading: driver + cuBLASLt + NVRTC today; **add cuDNN + cuRAND**).

**Status:** design / migration plan. This document is the **authoritative target**
for the CUDA EP's op→backend mapping and its runtime-library acquisition model.
`docs/CUDA_COVERAGE.md` records what is *implemented today*; it will be reconciled
**to this target** in a later pass. **Do not edit `CUDA_COVERAGE.md` from this
branch** (Wallace's wave-3 has it in flight — editing it here causes a merge
conflict).

Governing decisions this doc implements:
`.squad/decisions/inbox/coordinator-cuda-library-first-pytorch.md`,
`coordinator-cuda-kernel-strategy.md`, `coordinator-cuda-zero-setup-deps.md`,
`coordinator-cupti-wheel-bundling.md`.

---

## 1. Principle — library-first is mandatory; NVRTC is the justified exception

The user directive is explicit: the CUDA EP is **too hand-written**, which is a
**device-compatibility risk**. Mirror PyTorch: **use vendor libraries for both
max performance AND max device compatibility.**

**Why libraries win on compatibility, not just speed.** NVIDIA re-tunes
cuBLAS(Lt) and cuDNN *per SM architecture* (SM70 Volta → SM80/86 Ampere →
SM89 Ada → SM90 Hopper → SM100 Blackwell). A single hand-written NVRTC tile /
block size that is good on H100 (SM90) can be badly mis-tuned on an A10 (SM86)
or an L4 (SM89): wrong occupancy, wrong shared-memory budget, wrong tensor-core
path. The vendor library ships **many arch-specialized code paths and picks the
right one at runtime** — that is precisely the portability the user is asking
for. Hand-tuned kernels do *not* adapt across `SM70 → SM90+`; libraries do.

**The rule:**

1. **Library-first is MANDATORY for heavy, arch-sensitive ops** — GEMM, conv,
   pooling, softmax, activations, normalization, LRN/batchnorm, attention,
   reductions. STOP hand-writing these. Route them to cuBLASLt / cuDNN /
   CUTLASS.
2. **NVRTC-custom is allowed ONLY when:**
   - **(a) No library op exists** — generic elementwise / pointwise unary,
     binary, comparison, logical, cast. *This is PyTorch-consistent:* PyTorch
     itself JIT-compiles elementwise via NVRTC / nvFuser rather than calling a
     library, and a JIT-compiled elementwise kernel is **arch-portable by
     construction** (NVRTC targets the running device's compute capability).
     Elementwise is memory-bound and arch-insensitive, so this carries no
     compatibility risk.
   - **(b) A measurable fusion win** over calling the library op-by-op —
     fused norm+residual, RoPE, fused GEMM epilogues (bias/activation), a fused
     elementwise chain. Each such kernel must **justify itself** in the coverage
     doc with the fusion it buys (HBM round-trips removed).
3. **Every runtime-library failure is an actionable `EpError`** (RULES.md #1):
   name the missing lib **and the exact pip/conda package** that provides it.

---

## 2. Op → backend target matrix (authoritative)

Backend legend: **cuBLASLt** (GEMM) · **cuDNN** (conv/pool/softmax/act/norm/
reduce) · **CUTLASS** (NVRTC-compiled device templates for fusions cuDNN/cuBLASLt
can't express — flash attention) · **cub/thrust** (device primitives — see the
⚠️ availability note below) · **NVRTC-custom** (pointwise + justified fusions) ·
**memcpy/view** (data movement, no arithmetic).

> ⚠️ **cub/thrust reality check (verified against cudarc 0.19.8).** `cudarc`
> exposes safe bindings for `cublas`, `cublaslt`, `cudnn`, `curand`, `cufft`,
> `cusolver`, `cusparse`, `cutensor`, `nvrtc`, `cupti` — **but NOT cub or
> thrust.** cub and thrust are **header-only C++ device-template libraries**;
> they are *not* shippable/dlopen-able runtime `.so`s like cuBLAS/cuDNN. So
> "route reductions to cub" cannot mean "dlopen libcub". The realistic
> library path for reductions is **cuDNN `cudnnReduceTensor`** (arch-tuned,
> dlopen-able, already in cudarc's safe API). True cub/thrust use (sort, topk,
> scan) would require **NVRTC-compiling cub device templates** (needs the CCCL
> headers, shipped by the `nvidia-cuda-cccl-cuXX` wheel) or CUTLASS — a heavier
> lift tracked as a stretch item, not the default.

### GEMM family
| Op | Target backend | Notes |
|----|----------------|-------|
| `MatMul` | **cuBLASLt** | landed. 2-D + batched, f32/f16/bf16. |
| `Gemm` | **cuBLASLt** (+ epilogue) | landed; move the NVRTC β·C bias into a `CUBLASLT_EPILOGUE_BIAS` epilogue. |
| `FusedMatMulBias` (`com.microsoft`) | **cuBLASLt** `EPILOGUE_BIAS` | bias fused in-GEMM. |
| `FusedGemm` (`com.microsoft`) | **cuBLASLt** `EPILOGUE_{RELU,GELU}_BIAS` | activation+bias fused in-GEMM. |

### Elementwise — unary / activation / binary / comparison / logical
| Op group | Target backend | Notes |
|----------|----------------|-------|
| `Relu` `Sqrt` `Erf` `Tanh` `Sigmoid` `Gelu` … | **NVRTC-custom** | KEEP. No library op; kept NVRTC precisely so they can fuse into GEMM epilogues / elementwise chains. Arch-portable (JIT). |
| `Add` `Sub` `Mul` `Div` `Pow` `Min` `Max` | **NVRTC-custom** | KEEP. Broadcasting + chain fusion belong here, not cuDNN OpTensor. |
| Pointwise unary / comparison / logical (Wallace wave-3: `Neg` `Abs` `Not` `Equal` `Greater` `Less` `And` `Or` `Xor` …) | **NVRTC-custom** | **KEEP — confirmed.** No library op exists for these; cuDNN OpTensor covers only a few binary arithmetic ops and is less fusable. |

### Convolution / pooling (future coverage — currently absent)
| Op | Target backend | Notes |
|----|----------------|-------|
| `Conv` `ConvTranspose` | **cuDNN** `cudnnConvolutionForward` | vendor-tuned, the #1 reason to add cuDNN for CNN models. |
| `MaxPool` `AveragePool` `GlobalAveragePool` | **cuDNN** `cudnnPoolingForward` | |
| `LRN` | **cuDNN** LRN | |
| `BatchNormalization` `InstanceNormalization` | **cuDNN** `cudnnNormalizationForward` | |

### Normalization & softmax
| Op | Target backend | Decision | Notes |
|----|----------------|----------|-------|
| `Softmax` (v1 & v13) | **cuDNN** `cudnnSoftmaxForward` | **MOVE → cuDNN** | standalone Softmax is a pure library op; cuDNN is arch-tuned. Keep an NVRTC softmax **only** embedded inside the fused attention kernel. |
| `LayerNormalization` | **NVRTC-custom (fused)** | **KEEP** | real fusion win: mean/var + normalize + affine in one HBM read. cuDNN would be reduce + separate pointwise passes. Justified. |
| `SkipLayerNormalization` (`com.microsoft`) | **NVRTC-custom (fused)** | **KEEP** | residual add fused into the norm → saves a full tensor round-trip. Clear fusion win. |
| `RMSNormalization` / `SimplifiedLayerNormalization` | **NVRTC-custom (fused)** | **KEEP** | LLaMA-family fused RMS scale; no cuDNN equivalent that fuses the affine. |

### Reductions
| Op | Target backend | Decision | Notes |
|----|----------------|----------|-------|
| `ReduceSum` `ReduceMean` `ReduceMax` `ReduceMin` | **cuDNN** `cudnnReduceTensor` | **MOVE → cuDNN** | the dlopen-able, arch-tuned library reduce (cub is header-only, not a runtime lib). NaN-propagation semantics must be validated to match the CPU EP; if cuDNN's reduce can't match ONNX NaN semantics for max/min, KEEP those two as NVRTC and move only Sum/Mean. |
| `ReduceProd` `ReduceL1/L2` `ReduceLogSum*` | **cuDNN** `cudnnReduceTensor` | target | cuDNN exposes these reduce ops directly. |
| `ArgMax` `ArgMin` `TopK` `CumSum` `Sort` | **cub/thrust via NVRTC** or NVRTC-custom | **KEEP NVRTC for now** | no dlopen-able library; cub templates need CCCL headers at NVRTC compile. Stretch item. |

### Attention
| Op | Target backend | Decision | Notes |
|----|----------------|----------|-------|
| `Attention` (`com.microsoft`) | **cuBLASLt GEMM + fused softmax** → **CUTLASS/cuDNN flash** | **MOVE (fuse)** | today: cuBLAS GEMM + NVRTC softmax, materialises the full `[B,H,Sq,Sk]` score matrix (O(S²) HBM). Target: flash-attention (scores stay in SRAM). |
| `FusedAttention` / SDPA / GQA | **CUTLASS FlashAttention-3** (NVRTC device templates) or **cuDNN SDPA frontend** (needs a thin shim — see §5) | **top perf item** | cudarc's cuDNN **safe API does not expose the MHA/graph SDPA frontend**, so this is CUTLASS-via-NVRTC unless we add a cuDNN-frontend bindings shim. |

### Shape / data-movement / misc
| Op | Target backend | Notes |
|----|----------------|-------|
| `Identity` `Reshape` `Unsqueeze` `Squeeze` `Flatten` | **view rewrite / memcpy** | metadata-only when contiguous; D2D copy otherwise. |
| `Cast` `CastLike` | **NVRTC-custom** | KEEP — dtype conversion, no library op. |
| `Transpose` | **NVRTC-custom** / fold into consumer GEMM `op` | tiled transpose. |
| `Gather` `Slice` `Expand` `Concat` `Constant` `Shape` | **NVRTC-custom / memcpy / host** | indexed / strided copies. |

---

## 3. Migration plan (concrete, prioritized)

Landed hand-written kernels are **not ripped out immediately** — they are
re-routed to a library where the library is the right call. MOVE/KEEP per op:

| # | Currently hand-written | Decision | Justification | Affected wave |
|---|------------------------|----------|---------------|---------------|
| 1 | **`Softmax`** (`softmax.rs`) | **MOVE → cuDNN** `cudnnSoftmaxForward` | standalone softmax is a pure arch-sensitive library op; cuDNN adapts SM70→SM90+. Keep the NVRTC softmax *only* inline in attention. | Joshi wave-2 |
| 2 | **`ReduceSum/Mean`** (`reduce.rs`) | **MOVE → cuDNN** `cudnnReduceTensor` | dlopen-able arch-tuned reduce; cub is header-only and can't be a runtime lib. | Joshi wave-2 |
| 3 | **`ReduceMax/Min`** (`reduce.rs`) | **MOVE → cuDNN, gated on NaN parity** | move only if cuDNN reduce matches ONNX/CPU NaN-propagation; else KEEP NVRTC. | Joshi wave-2 |
| 4 | **`Attention`** (`attention.rs`) | **MOVE → CUTLASS flash** (or cuDNN SDPA shim) | biggest latency/throughput win; current path is O(S²) HBM. | wave-2 baseline |
| 5 | **`LayerNorm` / `SkipLayerNorm` / `RMSNorm`** (`normalization.rs`) | **KEEP NVRTC (fused)** | genuine fusion win: mean/var+normalize+affine (+residual) in one HBM read beats cuDNN reduce + pointwise passes. | Joshi wave-2 |
| 6 | **Elementwise unary/binary** (`elementwise.rs`) | **KEEP NVRTC** | no library op; kept fusable (chain/epilogue fusion). PyTorch-consistent (nvFuser). | Joshi wave-1 |
| 7 | **Pointwise unary/comparison/logical** (`pointwise.rs`, Wallace) | **KEEP NVRTC — confirmed** | no library op for `Neg/Abs/Not/Equal/Greater/Less/And/Or/Xor`; cuDNN OpTensor doesn't cover them and is less fusable. | Wallace wave-3 |
| 8 | **`Cast` / `CastLike`** (`cast.rs`) | **KEEP NVRTC** | dtype conversion, no library op. | Joshi wave-2 |
| 9 | **`Gemm` NVRTC β·C bias** (`gemm.rs`) | **MOVE → cuBLASLt epilogue** | fuse bias into the GEMM (`EPILOGUE_BIAS`) instead of a separate NVRTC pass. | wave-1 |

**Priority order for execution:** (1) add the cuDNN backend + shared resolver →
(2) Softmax→cuDNN → (3) Reduce{Sum,Mean}→cuDNN → (4) Attention→flash → (5) Gemm
epilogue fusion → (6) Reduce{Max,Min} NaN-parity gate. Norms and all
elementwise/pointwise/cast **stay NVRTC** (they are the justified exceptions).

---

## 4. Runtime-library auto-acquisition (PyTorch-style zero-setup) — THE key ask

Goal: `pip install nxrt[cuda]` (or a `nxrt-cuda` wheel) gives a working CUDA EP
with **no manual CUDA toolkit install** — exactly like `pip install torch`,
which pulls `nvidia-*` wheels and dlopens them from `site-packages`.

### 4.1 Python deps — the `nvidia-*-cu13` wheel set

Declare the NVIDIA-published redistributable wheels under the `cuda` extra
(match the `cudarc` `cuda-13000` pin → `cu13`, cuDNN v9). These are exactly the
wheels torch depends on:

| Runtime lib (Linux soname) | PyPI wheel | Why we need it |
|----------------------------|-----------|----------------|
| `libcudart.so.13` | `nvidia-cuda-runtime-cu13` | CUDA runtime (cudarc `runtime`). |
| `libcublas.so.13`, `libcublasLt.so.13` | `nvidia-cublas-cu13` | GEMM (landed). |
| `libcudnn*.so.9` | `nvidia-cudnn-cu13` | softmax/reduce/conv/pool/act/norm (new). |
| `libcurand.so.10` | `nvidia-curand-cu13` | RNG (Dropout/RandomNormal, future). |
| `libnvrtc.so.13` | `nvidia-cuda-nvrtc-cu13` | runtime kernel compile (elementwise/norm/cast). |
| `libcupti.so.13` | `nvidia-cuda-cupti-cu13` | GPU tracing (already declared). |
| (CCCL headers) | `nvidia-cuda-cccl-cu13` | *only if* we NVRTC-compile cub templates (stretch). |
| **`libcuda.so.1` (driver)** | **NOT on PyPI** | comes from the user's NVIDIA driver — the one documented prerequisite, same as torch. |

**pyproject change (describe for a follow-up agent — keep additive/minimal):**
extend the existing `cuda` extra. Current:

```toml
cuda = ["nvidia-cuda-cupti-cu13"]
```

Target:

```toml
cuda = [
    "nvidia-cuda-runtime-cu13",
    "nvidia-cublas-cu13",
    "nvidia-cudnn-cu13",
    "nvidia-curand-cu13",
    "nvidia-cuda-nvrtc-cu13",
    "nvidia-cuda-cupti-cu13",
]
```

**Version pinning approach:** pin to the `cu13` major line and floor each wheel
to the minimum version whose soname major matches what cudarc dlopens
(`>=x,<y+1` compatible-release style, e.g. `nvidia-cudnn-cu13>=9,<10`). Do **not**
hard-pin exact patch versions — that fights pip resolution when a user also has
torch installed (torch pins its own `nvidia-*`; overlapping compatible ranges
let one shared copy satisfy both). Verify the actually-dlopen'd sonames at
build/audit time so the list matches reality (don't over/under-declare).

### 4.2 Runtime discovery — generalize Leon's CUPTI resolver into a shared nvidia-lib resolver

Leon's `cupti::set_search_paths(Vec<PathBuf>)` + `collect_libcupti_candidates`
(`crates/onnx-runtime-tracer/src/cupti.rs`) is the template. Generalize it into
one shared resolver (new module, e.g. `onnx-runtime-nvidia-loader` or
`onnx-runtime-ep-cuda::nvlibs`) that every dlopen'd NVIDIA lib routes through.

**Search order (per lib), reusing Leon's proven logic:**
1. **System loader path** — bare soname (`libcublasLt.so.13`, then `.so`), so a
   system/toolkit install or `LD_LIBRARY_PATH` wins first.
2. **pip `site-packages`** — probe each injected `sys.path` root (the live
   interpreter path — this is Leon's fix for the **unactivated venv / user-site**
   case where `VIRTUAL_ENV` is unset and `/proc/self/exe` is the base
   interpreter) plus the extension's own dir + parent, for the pip layout
   `<root>/nvidia/<component>/lib/<soname>`.
3. **Conda** — `$CONDA_PREFIX/lib` (Linux/macOS) and `$CONDA_PREFIX/Library/bin`
   (Windows), plus `VIRTUAL_ENV`/`PYTHONHOME` prefixes as today.

**Per-component pip subdir map** (mirrors `nvidia/cuda_cupti/lib`):

| Lib | pip subdir |
|-----|-----------|
| cuBLAS/cuBLASLt | `nvidia/cublas/lib` |
| cuDNN | `nvidia/cudnn/lib` |
| cuRAND | `nvidia/curand/lib` |
| NVRTC | `nvidia/cuda_nvrtc/lib` |
| cudart | `nvidia/cuda_runtime/lib` |
| CUPTI | `nvidia/cuda_cupti/lib` (existing) |

**Injection point:** extend the existing PyO3 `inject_cupti_search_paths`
(`crates/onnx-runtime-python/src/lib.rs:556`) into a generic
`inject_nvidia_search_paths` that feeds the same live `sys.path` + module-dir
roots to **both** the tracer's CUPTI resolver **and** the new shared resolver,
once at module init (before any dlopen, since discovery caches in a `OnceLock`).

**Cross-platform lib naming (the resolver's soname table):**

| Component | Linux | Windows | macOS |
|-----------|-------|---------|-------|
| cuBLASLt | `libcublasLt.so.13`, `libcublasLt.so` | `cublasLt64_13.dll` | n/a |
| cuBLAS | `libcublas.so.13` | `cublas64_13.dll` | n/a |
| cuDNN | `libcudnn.so.9` | `cudnn64_9.dll` | n/a |
| cuRAND | `libcurand.so.10` | `curand64_10.dll` | n/a |
| NVRTC | `libnvrtc.so.13` | `nvrtc64_130_0.dll` | n/a |
| cudart | `libcudart.so.13` | `cudart64_13.dll` | n/a |

### 4.3 Absent-lib UX

- **Actionable RULES#1 error** when a required lib can't be dlopen'd: name the
  lib, the paths attempted (Leon's `attempted` vec already does this for CUPTI),
  **and the exact fix** — e.g.
  `cuda_ep: Softmax needs cuDNN (libcudnn.so.9) which was not found. Install it
  with 'pip install nvidia-cudnn-cu13' or 'conda install -c nvidia cudnn', or
  add it to LD_LIBRARY_PATH. Paths tried: [...]`.
- **Optional opt-in auto-pip fallback** (describe, don't require): behind an
  explicit env flag (e.g. `NXRT_AUTO_INSTALL_CUDA=1`), on first missing-lib the
  runtime could shell `python -m pip install <wheel>` then retry discovery.
  Off by default (never auto-mutate the user's environment silently).

### 4.4 Cross-platform

- **Linux:** SONAME dlopen via `libloading` (as CUPTI does today).
- **Windows:** `.dll` names above; call `AddDllDirectory` / `SetDefaultDllDirectories`
  for each discovered `nvidia/<component>/bin` (pip puts Windows DLLs under
  `bin`, not `lib`) and Conda `Library\bin`, so dependent-DLL resolution works.
- **macOS:** CUDA is unavailable — the whole resolver + EP is feature-gated to a
  no-op (build compiles, runtime reports the EP as unavailable, never panics).

---

## 5. cuDNN integration note

- **cudarc's `cudnn` feature suffices for the standalone ops.** Verified in
  cudarc 0.19.8: the `cudnn` safe API exposes `cudnnSoftmaxForward`,
  `cudnnReduceTensor`, `cudnnActivationForward`, `cudnnPoolingForward`,
  `cudnnConvolutionForward`, and (sys-level) `cudnnNormalizationForward` — i.e.
  everything §2 routes to cuDNN. Enable it by adding `"cudnn"` to the cudarc
  feature list; it pins cuDNN v9 (`cudnn-09021`) and honours `dynamic-loading`.
- **A thin bindings shim IS needed for fused flash attention.** cudarc's cuDNN
  safe API does **not** expose the cuDNN **graph / SDPA (MHA) frontend**. So
  `FusedAttention` via cuDNN would require a small `cudnn_frontend` bindings shim
  — OR we implement flash attention with **CUTLASS device templates compiled via
  NVRTC** (no extra dependency, stays toolkit-free). Recommendation: ship the
  CUTLASS-via-NVRTC flash path first; add a cuDNN-frontend shim only if it beats
  it on the target arch.
- **Build stays toolkit-free.** Adding cuDNN/cuRAND changes nothing about the
  build: `dynamic-loading` means **no `nvcc`, no toolkit, no `build.rs`** — the
  libs are dlopen'd at runtime exactly like the driver, cuBLASLt, and NVRTC
  today. `cargo build -p onnx-runtime-ep-cuda` still builds on a host with only
  `libcuda`.

---

## 6. Prioritized work-item list (each → a follow-up agent task)

1. **`add-cudnn-backend`** — enable cudarc `cudnn` + `curand` features; add a
   `cudnn` handle to `CudaRuntime` alongside cuBLASLt; confirm offline build.
   *(unblocks 2–4, 7)*
2. **`nvidia-lib-resolver`** — generalize `cupti::set_search_paths` /
   `collect_libcupti_candidates` into a shared resolver (system → pip
   site-packages → conda; per-component subdir map; cross-platform soname
   table; actionable missing-lib errors). Extend the PyO3 injector to feed it.
3. **`pyproject-cuda-extra-nvidia-wheels`** — expand the `cuda` extra to the full
   `nvidia-*-cu13` set (§4.1) with compatible-release pins; audit dlopen'd
   sonames match.
4. **`softmax-to-cudnn`** — move standalone `Softmax` to `cudnnSoftmaxForward`;
   keep NVRTC softmax only inside attention.
5. **`reduce-to-cudnn`** — move `ReduceSum`/`ReduceMean` to `cudnnReduceTensor`;
   gate `ReduceMax`/`ReduceMin` on ONNX NaN-parity (else KEEP NVRTC).
6. **`attention-flash`** — replace the O(S²) attention with CUTLASS-via-NVRTC
   flash attention (or a cuDNN-frontend SDPA shim); benchmark vs baseline.
7. **`gemm-epilogue-fusion`** — fold `Gemm` NVRTC β·C bias + `FusedGemm`/
   `FusedMatMulBias` into cuBLASLt `EPILOGUE_{BIAS,RELU_BIAS,GELU_BIAS}`.
8. **`conv-pool-cudnn`** *(coverage expansion)* — add `Conv`/`Pool`/`LRN`/
   `BatchNorm`/`InstanceNorm` via cuDNN (new op coverage for CNN models).
9. **`reconcile-cuda-coverage-doc`** — after Wallace's wave-3 lands, reconcile
   `docs/CUDA_COVERAGE.md` to this target matrix.

**KEEP (no work item — confirmed exceptions):** LayerNorm/SkipLayerNorm/RMSNorm
(fused), all elementwise unary/binary, pointwise unary/comparison/logical
(Wallace), Cast/CastLike — all NVRTC, justified by fusion or absence of a
library op.

---

## 7. Compatibility rationale summary (one paragraph for reviewers)

Hand-written NVRTC kernels bake in one arch's tile/occupancy assumptions and
silently mis-tune across the SM70→SM100 range the user must support; cuBLAS and
cuDNN carry NVIDIA's per-arch tuning and select the right path at runtime, giving
both peak performance **and** device portability. So we make libraries mandatory
for the heavy arch-sensitive ops (GEMM, softmax, reductions, conv/pool/norm-
where-not-fused, attention) and reserve NVRTC for the two cases where it's
strictly better: ops with no library (elementwise/pointwise/cast — the same
choice PyTorch's nvFuser makes) and measurable fusions (fused norms, RoPE, GEMM
epilogues, flash attention). Zero-setup is achieved the PyTorch way: depend on
the `nvidia-*-cu13` PyPI wheels and dlopen them from `site-packages`/Conda via a
generalized version of Leon's CUPTI resolver.
