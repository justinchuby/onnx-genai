# Decisions

Canonical, append-only record of accepted team decisions. Only the Coordinator (via Scribe merge) writes here. Agents drop proposals in `decisions/inbox/`.

---

---

### 2026-07-14: Wire official ONNX backend node tests to nxrt
**By:** Sebastian
**What:** Added `NxrtBackend`/`NxrtBackendRep` using `nxrt.InferenceSession` with `CPUExecutionProvider`, plus a pytest runner exposing only `OnnxBackendNodeModelTest` (offline; model-download groups excluded). The baseline on ONNX 1.22.0 and nxrt commit `f2dd92d` collected 3,530 cases: 130 passed, 1,635 failed, and 1,765 skipped. CPU-only coverage is 130/1,765 passed with 1,635 failed; all 1,765 CUDA variants skip because the adapter is CPU-only. Exact statuses are committed in `crates/onnx-runtime-python/conformance/onnx_backend_node_results.txt` on commit `e738135` / branch `squad/onnx-backend-test`.
**Why:** The official `onnx.backend.test` node suite adds broad standardized single-op coverage beyond the existing cbourjau/onnx-tests integration without hiding current kernel/dtype gaps behind blanket xfails. Largest failing families are Attention, reductions, CastLike, SoftmaxCrossEntropyLoss, Cast, Resize, LayerNormalization, RMSNormalization, and NegativeLogLikelihoodLoss.
**Re-run:** `export PATH=/home/justinchu/.conda/envs/onnx/bin:$PATH; cd crates/onnx-runtime-python; maturin build --release; python -m pip install --force-reinstall ../../target/wheels/nxrt-*cp310-abi3*.whl; mkdir -p ../../target/onnx-backend-test; python -m pytest tests/test_onnx_backend.py -q --junitxml=../../target/onnx-backend-test/junit.xml`. A nonzero pytest exit is expected while coverage gaps remain.

#### Source: `taffey-cuda-op-coverage.md`

### 2026-07-14: `onnx-runtime-ep-cuda` — library-first coverage matrix + first batch of library-backed kernels
**By:** Taffey (CUDA/perf)
**Branch:** `squad/cuda-op-coverage` @ `d7aa5a1` (pushed) | **Reviewed:** ⏳ pending non-author review (do NOT merge to main until reviewed)
**Governing directive:** `.squad/decisions/inbox/coordinator-cuda-kernel-strategy.md` (library-first, PyTorch-class fast, full coverage), RULES.md #1/#2/#4.

**What (library-mapping decisions):**
Authored `docs/CUDA_COVERAGE.md` — the model-agnostic op→backend roadmap, keyed to the CPU EP registry as the coverage reference (31 unique op types). Backend choices:
- **GEMM family** (`MatMul`, `Gemm`, `FusedMatMulBias`, `FusedGemm`) → **cuBLASLt** (+ epilogue fusions for the Fused* variants).
- **Elementwise unary/binary + activations** (`Relu/Sqrt/Erf/Tanh/Sigmoid/Gelu`, `Add/Sub/Mul/Div/Pow/Min/Max`) → **NVRTC-custom** f32 pointwise. Deliberately *not* cuDNN: keeping them as our own kernels is what later enables fusing an activation/add into a GEMM epilogue or an elementwise chain (RULES.md #4 fusion-win rule).
- **Softmax** → cuDNN or the existing NVRTC softmax (extract from attention.rs). **ReduceMean** → cub `DeviceReduce`. **LayerNorm/RMSNorm** → NVRTC-custom fused (mean+var+affine one-pass; fusion win). **Cast/Identity/Reshape/Transpose/Gather/Shape/Unsqueeze/Expand/Slice/Constant** → NVRTC-custom / memcpy / view-rewrite / host as tabulated.
- **Attention** baseline stays cuBLAS-GEMM+NVRTC-softmax; **FusedAttention** → cuDNN SDPA / FlashAttention-3 (top perf item).

**What (implemented this slice):**
- `kernels/gemm.rs` — `Gemm` ("" domain) on cuBLASLt via `blas::gemm_ex` (reuses the proven row-major↔col-major mapping), transA/transB/alpha/beta + a fused NVRTC `beta·C` broadcast-bias epilogue (scalar / per-row / per-col / full [M,N]). f32.
- `kernels/elementwise.rs` — NVRTC f32 pointwise unary (`Relu, Sqrt, Erf, Tanh, Sigmoid, Gelu[com.microsoft]`) and equal-shape binary (`Add, Sub, Mul, Div, Pow, Min, Max`). Broadcasting deferred with an actionable error.
- Registered all in `build_cuda_registry`; renamed `CUDA_PHASE2A_OPS`→`CUDA_COVERED_OPS`.
- `not_implemented` error rewritten to point at `docs/CUDA_COVERAGE.md` + CPU fallback (RULES.md #1). Fixed a real grid-size truncation bug (clamp-before-cast). Hardened attention `rt()` test helper with `catch_unwind` so GPU-gated tests skip (not panic) on a lib-less host.

**Coverage:** CUDA ops **2 → 16** (`MatMul, Gemm, Relu, Sqrt, Erf, Tanh, Sigmoid, Gelu, Add, Sub, Mul, Div, Pow, Min, Max, Attention`).

**Build / verification status (HONEST):**
- `cargo build -p onnx-runtime-ep-cuda` — **clean offline** (cudarc dynamic-loading; no CUDA toolkit needed).
- `cargo clippy -p onnx-runtime-ep-cuda` — **clean**.
- `cargo test -p onnx-runtime-ep-cuda --lib` — **20/20 pass** (all new tests are pure logic: GEMM plan/transpose mapping, bias-broadcast strides, NVRTC entry-point presence, dtype/contiguity guards, grid sizing).
- **NOT runtime-verified.** This host has `libcuda` only — **no `libcublasLt` / `libcudnn`** on the loader path (`ldconfig -p` confirms), and `nvcc` absent. So no kernel actually executed and **no perf benchmark vs PyTorch was run.** Numeric correctness of the new kernels rests on code review + the already-GPU-proven `gemm_ex` mapping. **Runtime + perf verification must happen on the H200 with cuBLASLt/cuDNN installed** (8× H200 are present on the box, but the libs are not).

**Prioritised custom-kernel candidate list (for the next agent):**
1. **FlashAttention-3 / cuDNN SDPA** behind the existing §13.3 `AttentionKernel` binding — baseline materialises the full O(S²) score matrix; biggest latency/throughput win.
2. **Fused LayerNorm / RMSNorm** (mean+var+affine one pass; add residual for residual+norm) — removes intermediate HBM traffic vs a library reduction+pointwise chain.
3. **`FusedGemm`/`FusedMatMulBias`** via `CUBLASLT_EPILOGUE_{GELU,RELU}_BIAS` — library fusion, folds our current Gemm+activation into one call.
4. **Elementwise-chain fusion** (why activations are NVRTC-custom, not cuDNN).
5. **RoPE** — no library op; small in-place fused kernel.
6. **Broadcasting elementwise** (shared index math with `Expand`) — lifts the equal-shape restriction.

**Follow-ups flagged:** add cudarc `cudnn` feature + a `cudnn` handle on `CudaRuntime` for the Softmax/Norm rows (still builds offline via dynamic-loading); pool the per-call cuBLASLt workspace to make MatMul/Gemm CUDA-graph-capturable; extend elementwise/Gemm to f16/bf16.

**Do NOT** touch `.squad/decisions.md` directly (Scribe merges this); coordinator cherry-picks the branch after a non-author review.

#### Source: `tyrell-cuda-strategy.md`

### 2026-07-14: CUDA EP library-first strategy + PyTorch-style zero-setup lib acquisition (Tyrell)

**By:** Tyrell (CUDA architecture lead), requested by Justin Chu (@justinchuby)
**Deliverable:** `docs/CUDA_STRATEGY.md` on branch `squad/cuda-strategy` (pushed, not merged).
**Scope:** DESIGN + migration plan only — no kernel rewrites. Did NOT edit `docs/CUDA_COVERAGE.md` (Wallace wave-3 in flight).

**Principle:** Library-first is MANDATORY for heavy arch-sensitive ops (GEMM, softmax, reductions, conv/pool/norm-when-not-fused, attention) — NVIDIA re-tunes cuBLAS/cuDNN per SM arch (SM70→SM100), hand-tuned NVRTC does not adapt → compatibility risk. NVRTC-custom allowed ONLY for (a) no-library elementwise/pointwise/cast (PyTorch-consistent: nvFuser JIT-compiles elementwise; arch-portable) or (b) measurable fusion win (fused norms, RoPE, GEMM epilogues, flash attention).

**Key finding (cudarc 0.19.8, verified):** cudarc exposes safe bindings for `cudnn` (v9: softmax/reduce/activation/pooling/conv + sys-level normalization), `curand`, cublaslt, nvrtc, cupti — **but NOT cub or thrust** (header-only C++ device templates, not dlopen-able runtime .so). So "reduce→cub" is not literally possible; the dlopen-able arch-tuned library reduce is **cuDNN `cudnnReduceTensor`**. True cub (sort/topk/scan) needs NVRTC-compiled CCCL templates (`nvidia-cuda-cccl-cu13`) — stretch item. cuDNN safe API also lacks the MHA/SDPA graph frontend → flash attention is CUTLASS-via-NVRTC (or a thin cudnn_frontend shim).

## Op → backend target matrix (authoritative; CUDA_COVERAGE.md reconciled to it later)

| Op / group | Target backend | Decision |
|---|---|---|
| MatMul, Gemm, FusedGemm, FusedMatMulBias | cuBLASLt (+ EPILOGUE_{BIAS,RELU_BIAS,GELU_BIAS}) | landed / fuse bias into epilogue |
| Relu/Sqrt/Erf/Tanh/Sigmoid/Gelu + Add/Sub/Mul/Div/Pow/Min/Max | NVRTC-custom | KEEP (fusable, no lib) |
| Pointwise unary/comparison/logical (Neg/Abs/Not/Equal/Greater/Less/And/Or/Xor) — Wallace | NVRTC-custom | KEEP — confirmed (no library op) |
| Softmax (v1/v13) | cuDNN cudnnSoftmaxForward | MOVE → cuDNN (keep NVRTC softmax only inline in attention) |
| LayerNorm / SkipLayerNorm / RMSNorm | NVRTC-custom (fused) | KEEP (mean/var+normalize+affine+residual in 1 HBM read) |
| ReduceSum / ReduceMean | cuDNN cudnnReduceTensor | MOVE → cuDNN |
| ReduceMax / ReduceMin | cuDNN, gated on NaN-parity | MOVE if cuDNN matches ONNX NaN-propagation, else KEEP NVRTC |
| ArgMax/ArgMin/TopK/CumSum/Sort | cub-via-NVRTC / NVRTC-custom | KEEP NVRTC for now (stretch) |
| Attention / FusedAttention (SDPA/GQA) | CUTLASS FlashAttention-3 (NVRTC) or cuDNN SDPA shim | MOVE (fuse) — top perf item; current path is O(S²) HBM |
| Conv/ConvTranspose/Pool/LRN/BatchNorm/InstanceNorm | cuDNN | coverage expansion (currently absent) |
| Cast/CastLike | NVRTC-custom | KEEP (dtype conv, no lib) |
| Identity/Reshape/Unsqueeze/Squeeze/Transpose/Gather/Slice/Expand/Concat | view-rewrite / memcpy / NVRTC | data movement |

## Migration MOVE/KEEP per op (justification)
- Softmax `softmax.rs` (Joshi w2): **MOVE→cuDNN** — pure arch-sensitive library op.
- ReduceSum/Mean `reduce.rs` (Joshi w2): **MOVE→cuDNN** — cub not dlopen-able.
- ReduceMax/Min `reduce.rs` (Joshi w2): **MOVE→cuDNN gated on NaN parity**, else KEEP.
- Attention `attention.rs` (w2): **MOVE→CUTLASS flash** — O(S²)→SRAM, biggest win.
- LayerNorm/SkipLayerNorm/RMSNorm `normalization.rs` (Joshi w2): **KEEP NVRTC (fused)** — real fusion win.
- Elementwise `elementwise.rs` (Joshi w1): **KEEP NVRTC** — no lib, fusable.
- Pointwise unary/comparison/logical `pointwise.rs` (Wallace w3): **KEEP NVRTC — confirmed**.
- Cast/CastLike `cast.rs` (Joshi w2): **KEEP NVRTC**.
- Gemm NVRTC β·C bias `gemm.rs` (w1): **MOVE→cuBLASLt epilogue**.

## Runtime-lib auto-acquisition (PyTorch-style, THE key ask)
**nvidia-*-cu13 PyPI wheels** (extend the existing `cuda` extra, compatible-release pins `>=x,<x+1`, don't hard-pin patch so it coexists with torch's nvidia-* pins):
- nvidia-cuda-runtime-cu13 (libcudart.so.13)
- nvidia-cublas-cu13 (libcublas/.libcublasLt.so.13)
- nvidia-cudnn-cu13 (libcudnn*.so.9)  ← NEW
- nvidia-curand-cu13 (libcurand.so.10)  ← NEW
- nvidia-cuda-nvrtc-cu13 (libnvrtc.so.13)
- nvidia-cuda-cupti-cu13 (libcupti.so.13, already declared)
- (nvidia-cuda-cccl-cu13 only if we NVRTC-compile cub templates — stretch)
- libcuda.so.1 (driver) = NOT on PyPI, user's NVIDIA driver, the one documented prereq (same as torch).

Current `cuda = ["nvidia-cuda-cupti-cu13"]` → expand to the 6-wheel set above. (Did NOT edit pyproject to avoid conflict; exact diff is in the strategy doc §4.1 for a follow-up agent.)

**Runtime discovery:** generalize Leon's `cupti::set_search_paths(Vec<PathBuf>)` + `collect_libcupti_candidates` (`crates/onnx-runtime-tracer/src/cupti.rs`) into ONE shared nvidia-lib resolver used by every dlopen'd lib. Search order per lib: (1) system loader/bare soname, (2) pip site-packages via live sys.path (Leon's unactivated-venv/user-site fix) at `<root>/nvidia/<component>/lib/<soname>`, (3) Conda `$CONDA_PREFIX/lib` + Windows `Library\bin`. Per-component subdirs: cublas→nvidia/cublas/lib, cudnn→nvidia/cudnn/lib, curand→nvidia/curand/lib, nvrtc→nvidia/cuda_nvrtc/lib, cudart→nvidia/cuda_runtime/lib, cupti→nvidia/cuda_cupti/lib. Extend PyO3 `inject_cupti_search_paths` (`crates/onnx-runtime-python/src/lib.rs:556`) into a generic injector feeding both resolvers at module init.

**Cross-platform soname table:** Linux libcublasLt.so.13 / Win cublasLt64_13.dll; cudnn libcudnn.so.9 / cudnn64_9.dll; curand libcurand.so.10 / curand64_10.dll; nvrtc libnvrtc.so.13 / nvrtc64_130_0.dll; cudart libcudart.so.13 / cudart64_13.dll. Windows: AddDllDirectory over nvidia/<component>/bin + Conda Library\bin. macOS: CUDA n/a → feature-gated noop.

**Absent-lib UX:** actionable RULES#1 EpError naming the missing lib + exact `pip install nvidia-*-cu13` / `conda install -c nvidia ...` fix + paths tried (Leon's `attempted` vec). Optional opt-in auto-pip fallback behind `NXRT_AUTO_INSTALL_CUDA=1` (describe, off by default).

## cuDNN integration note
Enable cudarc `cudnn` (v9, cudnn-09021) + `curand` features — dlopen'd, build stays toolkit-free (no nvcc/build.rs). cudarc's cuDNN safe API covers softmax/reduce/activation/pool/conv/norm. Fused flash attention needs a thin cudnn_frontend shim OR CUTLASS-via-NVRTC (recommend CUTLASS-via-NVRTC first — no extra dep).

## Prioritized work-item list (each → a follow-up agent task)
1. add-cudnn-backend — enable cudarc cudnn+curand; add cuDNN handle to CudaRuntime; confirm offline build. (unblocks 2-4,7)
2. nvidia-lib-resolver — generalize cupti discovery into shared resolver; extend PyO3 injector.
3. pyproject-cuda-extra-nvidia-wheels — expand `cuda` extra to the 6 nvidia-*-cu13 wheels, compatible-release pins.
4. softmax-to-cudnn.
5. reduce-to-cudnn (Sum/Mean; Max/Min NaN-parity gate).
6. attention-flash (CUTLASS-via-NVRTC or cuDNN SDPA shim).
7. gemm-epilogue-fusion (fold Gemm/FusedGemm/FusedMatMulBias into cuBLASLt epilogues).
8. conv-pool-cudnn (coverage expansion for CNN models).
9. reconcile-cuda-coverage-doc (after Wallace wave-3 lands).

**KEEP (no work item):** LayerNorm/SkipLayerNorm/RMSNorm (fused), all elementwise unary/binary, pointwise unary/comparison/logical (Wallace), Cast/CastLike — all NVRTC, justified.

**References:** docs/CUDA_STRATEGY.md (this branch), docs/CUDA_COVERAGE.md (do-not-edit), crates/onnx-runtime-tracer/src/cupti.rs (resolver template), crates/onnx-runtime-python/src/lib.rs:556 (injector), crates/onnx-runtime-ep-cuda/{runtime.rs,error.rs,Cargo.toml}, cudarc 0.19.8 features (cudnn/curand available; no cub/thrust). Governing: coordinator-cuda-library-first-pytorch.md, coordinator-cuda-kernel-strategy.md, coordinator-cuda-zero-setup-deps.md.

#### Source: `tyrell-review-joshi.md`

# Review Note — CUDA Wave 2 (Joshi) — Reviewer: Tyrell

**Verdict: 🟢 SHIP** (2 🟡 follow-ups, 0 🔴 blockers)
**Commit:** 2535eb6 (base origin/main a16e261) · **Scope:** ep-cuda + docs/CUDA_COVERAGE.md.
**Constraint:** host has libcuda only (no NVRTC/cuBLASLt/cuDNN/nvcc) → kernels NOT executed. Correctness rests on static review + element-for-element formula match to the CPU EP, which I verified. Runtime/perf on H200 is a known follow-up, not a blocker.

## Build gate (reproduced, offline)
- `cargo build  -p onnx-runtime-ep-cuda` → **clean** (7.6s).
- `cargo clippy -p onnx-runtime-ep-cuda` → **clean** (no warnings).
- `cargo test  -p onnx-runtime-ep-cuda --lib` → **43 passed / 0 failed**.
- Confirms Joshi's report. Note: these are GPU-free host tests (plan/view/axis/dtype-gating/registration). No NVRTC compile or kernel launch is exercised → the kernel *source strings* are validated by review only, never compiled. Inherent to the host; flagged, not charged against the wave.

## Per-op numeric findings

### Softmax — `softmax.rs` ✅ correct
- Row-max subtracted BEFORE exp (l.74–89), normalize l.102–105. Numerically stable.
- Arbitrary axis: `[outer, axis_dim, inner]` view; base = `o*axis_dim*inner + i` (l.67), stride = `a*inner` (l.76/89/104). **Verified on paper** for axis!=last (shape [2,3,4] axis 1: group (o,i) walks the middle dim at stride inner=4 — correct).
- Tree reduce assumes blockDim power-of-two = 256 (l.114) ✓; threads past axis_dim seed NEG_INF/0 so inert ✓. `row_sum>0` guard l.103 safe (exp>0 always).
- Legacy coerce-2D vs per-axis view math correct (softmax_view l.162–180, unit-tested).

### LayerNormalization — `normalization.rs` ✅ correct
- Population variance (÷N, l.98), **epsilon inside sqrt** (l.99), `y=(x-mean)*invstd*scale+bias` (l.107–111). Matches CPU `layernorm.rs`. Optional Mean/InvStdDev lengths validated (=num_groups).

### SkipLayerNormalization — `normalization.rs` ✅ correct
- Residual order `input+skip+bias` (l.188–191) matches the com.microsoft contract; bias per-channel len norm_size ✓. `__syncthreads()` at l.194 before the mean pass prevents the cross-thread read/write race on the stashed sum (correct). Inputs input/skip/gamma/[beta]/[bias], normalizes last dim. `input_skip_bias_sum` optional output len = input.numel() ✓.

### RMSNorm / SimplifiedLayerNorm — `normalization.rs` ✅ correct
- `rms = sqrt(mean(x^2)+eps)`, **no mean-subtract**, `y = x*invstd*scale` (l.137–154). Correct LLaMA-family norm.

### Reductions — `reduce.rs` ✅ correct (highest-scrutiny area)
- **base/delta offset split verified on paper for a 3D middle-axis reduce** (shape [2,3,4] reduce axis 1): strides [12,4,1]; base over kept axes {0,2} = {0,1,2,3,12,13,14,15}, delta over axis 1 = {0,4,8}; output element (i0,i2) → input `i0*12+i2` summed over `+{0,4,8}`. **Exact.** Valid because row-major strides are axis-independent (enumerate_offsets l.205–220).
- Output element order (base row-major over kept dims ascending) matches keepdims out_shape ordering ✓.
- keepdims shape (l.184–193); noop_with_empty_axes (empty+noop=1 → identity; empty/absent+noop=0 → reduce-all, l.291–314) — correct + unit-tested.
- NaN propagation Max/Min (l.83–84, 92–93) matches CPU `reduce_ops.rs:63–73` (`acc.is_nan()||x.is_nan()`) — **verified against CPU**. Inert threads seed ±INF/0, not NaN, so no spurious poisoning ✓. Mean divides sum by reduce_count (l.100) ✓.
- Offset tables alloc/free per-call → `cuda_graph_compatible()=false` (honest, documented).

### Cast / CastLike — `cast.rs` ✅ correct
- float→int truncates toward zero + saturates (`f_to_ll_sat` l.57–62), NaN→0; int→int 2's-complement wrap; →bool is `x!=0`; float↔float round-nearest. Matches ONNX/CPU `cast.rs`.
- Half (f16/bf16) isolated in a separate NVRTC module w/ fp16 headers so the common path is header-free (l.263–277) — good design; only half casts error if headers absent.
- dtype `switch` tags = raw ONNX discriminants, asserted in tests (l.327–334).

## 🟡 Follow-ups (NON-blocking)
1. **Softmax opset-13 default `axis`** (softmax.rs:129): defaults to **1**, but the ONNX opset-13 spec default is **-1**. Deliberate mirror of the CPU EP (verified: ep-cpu `softmax.rs:46` also defaults 1) so CPU/CUDA agree, but **both deviate from spec** when a model omits the `axis` attr. Real transformer exports set axis=-1 explicitly (low practical risk). Revisit the shared default project-wide (ep-cpu out of Joshi's scope — correctly untouched). Track as a cross-EP conformance item.
2. **H200 runtime/perf verification** — no NVRTC/GPU on host; kernel source strings never compiled. Must compile + numerically diff every kernel vs the CPU EP on an H200 before production trust. (Known, expected.)
   - Minor sub-notes, no action now: u64>2^63 through the signed lane and i64/u64>2^53 through the double lane lose precision (both documented in cast.rs); float→i64 saturation hi bound rounds to 2^63 (classic edge, harmless — ONNX leaves out-of-range float→int implementation-defined).

## Bottom line
Formulas, stability, axis/stride index math, dtype/axis gating (RULES#1 actionable errors), and additive registration are all correct and model-agnostic (RULES#2). Numerics genuinely mirror the CPU EP where cross-checked (reduce NaN, softmax axis default, layernorm variance). No 🔴 blockers → **ship**, with H200 numerical validation tracked as the gating follow-up before production use.

#### Source: `wallace-cuda-wave3.md`

# Decision — CUDA Wave 3: pointwise math / logical / comparison ops

**Author:** Wallace (CUDA kernel engineer) · **Branch:** `squad/cuda-wave3` · **Date:** 2026-07-14

## Summary

Extended CUDA EP pointwise coverage **additively** via NVRTC-compiled `extern "C"`
kernels, following the existing `elementwise.rs` pattern. New file:
`crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs`. Registration appended to
`kernels/mod.rs`. **CUDA op count: 27 → 48 (+21).**

Library-first strategy honored: pointwise activations/comparisons/logical ops
have **no NVIDIA library op** and are the endorsed "custom NVRTC" case
(RULES.md #4) — kept as our own kernels so they can later fuse into a
producer→activation→add chain or a GEMM epilogue.

## Ops added (21) with CPU-formula citation

### Unary math — f32→f32 (12), formulas matched **exactly** to CPU `unary_math.rs`
| Op | Kernel | CPU formula (`unary_math.rs:apply`) |
|----|--------|-------------------------------------|
| Abs | `fabsf(x)` | `x.abs()` |
| Neg | `-x` | `-x` |
| Reciprocal | `1.0f/x` | `1.0 / x` |
| Exp | `expf(x)` | `x.exp()` |
| Log | `logf(x)` | `x.ln()` (natural log) |
| Sign | `(v!=v)?v:(v>0?1:(v<0?-1:0))` | `sign()` — NaN→NaN, sign(0)=0 (`unary_math.rs:86`) |
| Floor | `floorf(x)` | `x.floor()` |
| Ceil | `ceilf(x)` | `x.ceil()` |
| Round | `rintf(x)` | `x.round_ties_even()` — round-half-to-**even** (NOT `roundf`) |
| Sin | `sinf(x)` | `x.sin()` |
| Cos | `cosf(x)` | `x.cos()` |
| Softplus | `fmaxf(x,0)+log1pf(expf(-fabsf(x)))` | `x.max(0.0)+(-x.abs()).exp().ln_1p()` (`unary_math.rs:softplus`) |

### Logical — bool (4)
| Op | Kernel | Reference |
|----|--------|-----------|
| Not | `(x==0)?1:0` | CPU `logical.rs` (`u8::from(b==0)`), non-zero byte = true, canonical 1/0 out |
| And | `((a!=0)&&(b!=0))?1:0` | ONNX bool semantics (non-zero = true, matches CPU `Not` byte convention) |
| Or | `((a!=0)\|\|(b!=0))?1:0` | ONNX bool semantics |
| Xor | `((a!=0)!=(b!=0))?1:0` | ONNX bool semantics |

### Comparison — f32→bool (5)
| Op | Kernel | Reference |
|----|--------|-----------|
| Equal | `(a==b)?1:0` | ONNX comparison spec |
| Greater | `(a>b)?1:0` | ONNX comparison spec |
| Less | `(a<b)?1:0` | ONNX comparison spec |
| GreaterOrEqual | `(a>=b)?1:0` | ONNX comparison spec |
| LessOrEqual | `(a<=b)?1:0` | ONNX comparison spec |

**Note on comparison/logical (And/Or/Xor + all comparisons):** these ops are
**not registered in the CPU EP registry** today, so there is no CPU
implementation to match against. Their ONNX semantics are canonical and trivial
(`a==b`, `a>b`, boolean `&&`/`||`/`!=`); kernels follow the ONNX spec directly.
This means the CUDA EP now covers *more* pointwise ops than the CPU EP (safe:
heterogeneous routing can send these to CUDA). `Not` **is** in the CPU registry
and is matched to it exactly.

## dtype coverage

- Unary math + comparison: **f32** (comparison output **bool**).
- Logical + `Not`: **bool** (1 byte/elem, non-zero = true).
- **f16/bf16 deferred** — identical to the existing `elementwise.rs` slice, which
  is also f32-only pending the dtype-templated NVRTC source. No dtype-traits
  pattern exists in `elementwise.rs` yet to follow; deferring keeps parity.
  Non-supported dtypes return an actionable `not_implemented` error naming the
  op + dtype (RULES.md #1).

## Broadcasting

Binary comparison/logical ops require **equal-shape** operands, **matching the
existing `elementwise.rs` binary kernels exactly** — NumPy broadcasting is
deferred crate-wide. A shape mismatch returns the same actionable
"broadcast/materialise upstream" error. No new broadcasting math invented
(per instruction: reuse what Add/Sub use).

## Ops deferred (follow-up list)

Target-set items I did **not** add, and why:
- **Activations:** `LeakyRelu`, `Elu`, `HardSigmoid`, `Clip`, `Softsign`, `Selu`
  — **not in the CPU EP registry**, so no CPU formula to match against under the
  correctness gate (host has libcuda only; no GPU runs). These need attribute
  parsing (alpha/beta, min/max) + a CPU reference to validate against. Deferred
  until a CPU reference lands or an owner signs off on ONNX-spec-only kernels.
  All are straightforward NVRTC pointwise once greenlit.
- **f16/bf16** for the ops added here — deferred with the crate-wide dtype-
  templating effort (also pending for `elementwise.rs`).
- **NumPy broadcasting** for the binary comparison/logical ops — deferred with
  the crate-wide broadcast index-math effort (shared with `Expand`; already
  listed as candidate #6 in `docs/CUDA_COVERAGE.md`).

## Test summary

New GPU-free unit tests (everything testable without a GPU per the correctness
gate): entry-point presence in NVRTC source, distinct entry points, dtype
rejection (actionable), strided rejection, `Round` uses `rintf` (ties-to-even,
not `roundf`), `Sign` NaN guard, `BinaryKind`→operand-dtype mapping, grid
coverage, i32 overflow guard, and coverage-list registration (mod.rs).

## Build gate (offline, per-crate)

```
cargo build  -p onnx-runtime-ep-cuda   → Finished, clean
cargo clippy -p onnx-runtime-ep-cuda   → Finished, no warnings
cargo test   -p onnx-runtime-ep-cuda --lib → ok. 55 passed; 0 failed
```
(43 baseline lib tests + 12 new = 55.)

## Runtime caveat

Host has **libcuda only** (no NVRTC/GPU runtime) — kernels compile + pass
GPU-free unit tests but were **not executed/benchmarked**. Numerical correctness
rests on the CPU-matched formulas cited above. Runtime + perf validation must
happen on an H200 (same caveat as Wave 2).

## Files touched (localized)

- **new:** `crates/onnx-runtime-ep-cuda/src/kernels/pointwise.rs`
- **edit:** `crates/onnx-runtime-ep-cuda/src/kernels/mod.rs` (module decl,
  `CUDA_COVERED_OPS`, registration block, coverage test)
- **edit:** `docs/CUDA_COVERAGE.md` (new op rows + Wave-3 score)

Stayed out of `executor.rs`, `onnx-runtime-session`, tracer, CI, python
bindings, build scripts (per scope).

#### Source: `zhora-concat-efficient.md`

### 2026-07-14: Concat writes directly into its single executor-provided output
**By:** Zhora
**What:** Before this change, Concat materialized every input with `to_dense_bytes`, built a second complete output `Vec<u8>`, then copied that buffer into the executor-provided output. Commit `9ebb1a7` removes all data-sized intermediate allocations. The kernel now validates and computes the output shape once, then writes directly into the already allocated contiguous output view.
**Why:** Concat should be memory-efficient and dtype-agnostic while preserving correctness for strided views. For contiguous inputs, each `outer` row copies one `axis_len * inner_bytes` slab with `ptr::copy_nonoverlapping` into its final output slice. Non-contiguous inputs use a correct stride-aware element gather directly into final output positions, without materializing a dense temporary.

**Race safety:** The kernel remains single-threaded. Each input/outer slab maps to a disjoint output range, and every destination element is written exactly once. The executor's SSA/output-allocation contract keeps source and output allocations disjoint, satisfying `copy_nonoverlapping`.

**Validation:** `cargo build -p onnx-runtime-ep-cpu` passed. `cargo test -p onnx-runtime-ep-cpu --lib` passed: 196 tests, 0 failures. Concat coverage includes axis 0, middle, last/negative axis, f32, i64, u8, a transposed non-contiguous input, axis out-of-bounds, and mismatched non-concat dimensions.


---

### 2026-07-14: If/Loop/Scan efficiency pass
**By:** Batty
**What:** Prepared the selected subgraph once per control-flow invocation; reused cached child executors, Loop scalar tensors, and Scan slice tensors; removed per-iteration capture deep-copies and shape-signature allocations; and replaced retained per-step scan tensors plus a final stacking pass with one preallocated byte accumulator. Added deterministic build/run counters proving a 1,000-iteration Loop and Scan each build their body once. If still builds only the selected branch. Deliberately left cross-executor buffer aliasing and a full liveness planner as future work.
**Why:** Removes repeated graph/signature/capture work, allocator churn, and scan-output double buffering from the iteration hot path while preserving lexical capture semantics and the conservative view-source lifetime rule required for very efficient control flow.

---

### Review: central runtime environment configuration

**Reviewer:** Bryant (non-author reviewer)  
**Commit:** `875bcc1` (`squad/central-env-config`)  
**Verdict:** 🟢 Approved

## Evidence

- `cargo build --offline -p onnx-genai-runtime-config`: passed.
- `cargo test --offline -p onnx-genai-runtime-config`: passed (6 tests).
- `cargo clippy --offline -p onnx-genai-runtime-config --lib -- -D warnings`: passed.
- The pure-Rust crate is a workspace member and is wired through workspace
  dependencies into `onnx-genai-ort`, `onnx-genai-engine`, and
  `onnx-genai-bench`.

## Fidelity spot checks

| Flag | Previous behavior | Registry/migration behavior |
|---|---|---|
| `ONNX_GENAI_EP` | UTF-8 read; trim + lowercase; CPU/WebGPU/CUDA/Metal/CoreML aliases; unsupported warns then CPU | Same UTF-8 handling, normalization, aliases, and unsupported value preserved for the existing warning |
| `ONNX_GENAI_CUDA_DEVICE` | Trimmed non-negative `i32`; default `0`; invalid raw value warns then uses `0` | `CudaDevice::{Id, Invalid}` preserves the same parse/default/warning behavior |
| `ONNX_GENAI_WEBGPU_VALIDATION` | `1`/`true`/`yes`/`on`, case-insensitive and trimmed, retains validation; otherwise disables it | Same truthy parser, with the existing negation at the call site |
| `ONNX_GENAI_PROFILE` | Only exact `1`, `true`, or `yes`; default false | Same intentionally narrow, case-sensitive parser |
| `ONNX_GENAI_METAL_EP_LIB` | `var_os`, ignores an empty path, preserving non-UTF-8 paths | Same `var_os`/`PathBuf` path handling and empty-path filter |
| `ONNX_GENAI_SPEC_ALLOW_SLOW` | Presence-only `var_os` flag, including empty values | Same presence-only lookup |

The requested straggler command produced no output: no remaining matching
`std::env::var` calls outside `runtime-config`, build scripts, or server code.

---

### 2026-07-14: Bundle oneDNN in Linux and macOS Python wheels
**By:** Coco
**What:** Enable the non-default `onednn` Cargo feature for Linux and macOS `nxrt` CPU wheels, while leaving Windows and crates.io defaults on the pure-Rust CPU backend. Linux uses oneDNN's OpenMP runtime and auditwheel repair; macOS uses `ONEDNN_CPU_RUNTIME=SEQ`.
**Why:** oneDNN has no convenient standalone PyPI runtime package, so static wheel bundling provides out-of-box CPU acceleration. Sequential oneDNN on macOS avoids fragile Homebrew/libomp discovery while retaining SIMD kernels and higher-level Rayon parallelism.

### 2026-07-15: Enable oneDNN in the Windows CPU wheel with SEQ
**By:** Leon
**What:** Enable `--features onednn` for the Windows AMD64 CPU wheel and set `ONEDNN_CPU_RUNTIME=SEQ` in cibuildwheel's Windows environment. Keep Linux on OMP/libgomp and macOS on SEQ. Leave Windows OMP/TBB as a follow-up optimization.
**Why:** The current build script is already MSVC-aware and emits no GNU C++/OpenMP libraries on MSVC. SEQ avoids first-pass OpenMP runtime variability; `windows-latest` already provides Visual Studio 2022 and CMake on PATH, while the explicit CMake install remains correctly scoped to the manylinux container.

---

### 2026-07-14T23-31-07: Centralize all runtime env-var flags into one typed registry — no scattered std::env::var reads
**By:** coordinator
**What:** Centralize all runtime env-var flags into one typed registry — no scattered std::env::var reads
**References:** crates/onnx-genai-ort, crates/onnx-genai-engine, crates/onnx-genai-server, crates/onnx-runtime-ep-cuda
**Why:** User directive (Justin Chu): "我们runtime还有各个地方有很多envvar的flag，要统一管理，不要到处都是" — the runtime reads ~35+ environment-variable flags (ONNX_GENAI_*, ORT_*, NXRT_*) scattered across crates via ad-hoc std::env::var calls (13 in onnx-genai-ort, plus server/engine/tracer/ep-cpu/bench). Going forward these MUST be centrally managed: a single typed config registry that declares every runtime flag (name, type, default, doc), parses once (OnceLock), and is the ONLY place runtime env vars are read. New features add their flag to the central registry instead of calling std::env::var inline. Build-time-only vars in build.rs (ORT_ROOT, ORT_LIB_DIR) are out of scope. Benefits: one discoverable list of all knobs, consistent naming/parsing, testability, single doc source.

---

### 2026-07-14T23-50-37: oneDNN CPU accel: ship it enabled in Linux+macOS Python wheels (PyTorch-style), keep crates.io source pure-Rust
**By:** coordinator
**What:** oneDNN CPU accel: ship it enabled in Linux+macOS Python wheels (PyTorch-style), keep crates.io source pure-Rust
**References:** .github/workflows/wheels.yml, crates/onnx-runtime-python/Cargo.toml, crates/onnx-runtime-python/pyproject.toml, docs/CROSS_PLATFORM.md
**Why:** User chose Option A (with "do what PyTorch does"): oneDNN has no convenient standalone PyPI libdnnl (unlike CUDA's nvidia-*-cu13 wheels), so we follow PyTorch's model — statically build oneDNN from the third_party/onednn submodule and bundle it INTO our Python wheels, enabled by default there, while the crates.io source surface stays pure-Rust/offline (onednn remains a non-default cargo feature). Platform scope: enable --features onednn in the CPU wheels for Linux (manylinux, OMP/libgomp) and macOS x86_64+arm64 (OpenMP via libomp or SEQ fallback); DO NOT enable on Windows — the MSVC oneDNN toolchain path is not wired yet (CROSS_PLATFORM.md:33, tracked as follow-up). Requires: add an `onednn` passthrough feature to the nxrt (onnx-runtime-python) crate forwarding to onnx-runtime-ep-cpu/onednn; provision cmake in the manylinux before-all; per-platform CIBW_CONFIG_SETTINGS build-args. Verified locally: cargo test -p onnx-runtime-ep-cpu --features onednn → 203 tests pass on Linux.

---

### 2026-07-14: Concat/Slice memory-efficiency pass
**By:** Deckard
**What:** Confirmed Concat already writes contiguous input slabs directly into the executor-provided output allocation, with no intermediate per-input buffers. Slice now collapses a full, unit-stride trailing suffix and copies each suffix with one `copy_from_slice` run; it no longer performs per-element output copies or creates per-element index state.
**Why:** This preserves Concat's single output-allocation path and reduces Slice's common leading-axis slice path from one copy per element to one bulk copy per contiguous output region, satisfying the memory-efficiency requirement without executor changes.

---

### Review: squad/publish-vendor-cpuinfo (b6b2aad)

**Reviewer:** Gaff (build-systems / native-linking) — non-author
**Author:** Mariette (locked out)
**Verdict:** 🟢 APPROVE

Vendoring cpuinfo in-crate + CARGO_MANIFEST_DIR-relative path + exclude list
correctly unblocks `cargo publish` for `onnx-runtime-cpuinfo`. Verified on this
host (96-core x86_64, cmake + libclang present) in a detached worktree at b6b2aad.

## Evidence (verified myself, not from author's report)

**Build gate**
```
cargo build -p onnx-runtime-cpuinfo  -> Finished dev in 5.78s (clean)
cargo test  -p onnx-runtime-cpuinfo --lib -> ok. 0 tests, links cleanly
example calling CpuInfo::detect() -> "cores=96 l2=2097152 avx2=true"  (runtime OK)
```

**Publish gate (the whole point)**
```
cargo publish --dry-run -p onnx-runtime-cpuinfo
  Packaged 133 files, 1.0MiB (185.5KiB compressed)   # << 10MB limit
  Verifying ... Compiling ... Finished  -> verification compile from tarball SUCCEEDED
```
Tarball contains CMakeLists.txt, cmake/, src/**, include/**, deps/clog/**; the
excluded test/bench/tools dirs are absent. Nothing the build needs was dropped.

## High-scrutiny questions A–E

**A. clog removal — SAFE.** The pinned cpuinfo commit (b1a5d63) references clog
NOWHERE in src/, include/, or CMakeLists.txt (grep = 0 hits). `nm` on the built
libcpuinfo.a shows zero `clog_*` symbols, defined or undefined. The old
`rustc-link-lib=static=clog` was stale; removing it is correct. Downstream
example linked & ran with no unresolved symbols.

**B. wrap_static_fns + cc wrapper — CORRECT & REQUIRED (not scope creep).**
cpuinfo.h has 145 `static inline` functions, incl. `cpuinfo_has_x86_avx2`,
which the crate calls via ffi. Without generated+cc-compiled wrappers these have
no linkable symbols → undefined refs. The wrappers compiled, linked, and returned
correct values at runtime (avx2=true on this AVX2 host). Justified by the move.

**C. bindgen 0.70->0.71 + rust_target(1.85)/edition2024 — OK.** Bindings
regenerated; crate's src/lib.rs (cpuinfo_* usages) compiles unchanged; example
runs. No API breakage.

**D. exclude adequacy — OK.** 185.5KiB compressed, far under 10MB, and the
verification compile from the packaged tarball succeeded — proving all cmake/src/
include/clog inputs survive. (Nit, non-blocking: exclude globs are top-level
`vendor/cpuinfo/test/**` etc., so `deps/clog/test/clog.cc` remains in the tarball;
negligible size, clog isn't built. Optional future tidy: `vendor/cpuinfo/**/test/**`.)

**E. CMAKE_INSTALL_LIBDIR=lib — OK.** Install produced `.../out/lib/libcpuinfo.a`
(not lib64); `rustc-link-search=native={}/lib` points at an existing dir.

## Cargo.lock (-49 net) — benign
Pure dedup: dropping bindgen 0.70 removes its unique transitive deps
(itertools 0.13, rustc-hash 1.1.0); repo unifies on bindgen 0.71. thiserror 2.0.18
is genuinely used (CpuInfoError derives thiserror::Error). No needed dep removed.

## Notes for coordinator
- 🟢 Approve & merge. Do NOT gate on onnx-genai-ort / -engine / -ort-sys (those
  don't build offline; out of scope for this change).
- Optional non-blocking nit (D) can be a follow-up; not required for merge.

---

### 2026-07-14: cuDNN backend foundation (handle + descriptors)
**By:** Holden
**What:** Added `crates/onnx-runtime-ep-cuda/src/cudnn/mod.rs` with a lazy, stream-bound, serialized `CudnnBackend`, cudarc-owned RAII tensor descriptors, validated dim/stride planning, and ONNX f32/f16/bf16 → cuDNN dtype mapping. Enabled cudarc 0.19.8's `cudnn` feature while retaining `dynamic-loading`.
**Why:** Implements the CUDA_STRATEGY library-first foundation and unblocks the Softmax, Reduce, Conv, and Pool cuDNN ports without creating a second CUDA context or introducing a link-time libcudnn dependency.

---

### 2026-07-14: Centralize ONNX GenAI runtime environment configuration
**By:** Joshi
**What:** Added the pure-Rust `onnx-genai-runtime-config` crate as the single typed registry for library-internal `ONNX_GENAI_*` runtime flags. It snapshots the environment once through `OnceLock`; ORT, engine test harnesses, and bench now consume typed fields rather than reading environment variables inline. Server clap configuration, ORT build scripts, and nxrt tracer search paths remain independently managed because they are different configuration layers.
**Why:** A process-wide registry makes runtime knobs discoverable and documented, prevents parsing/default semantics from drifting across call sites, and gives new flags one required home with closure-driven unit tests.

---

### 2026-07-15: Tiered-memory implementation audit
**By:** Luv
**What:** Verdict: **Partially implemented, but the arbitrary-size-model goal is not met.** The design explicitly promises GPU→RAM→disk memory with weight streaming, activation spill, KV paging, and “never OOM” (`docs/ORT2.md:3351-3367,3371-3398,3401-3424,3436-3484,3566-3576`; `docs/DESIGN.md:105-171`). Today, only a limited host-side KV tier/cache and external-weight mmap exist. There is no real device→host→disk unified allocator, weight placement/streaming, activation spill, disk-backed KV, resource governor, or device-OOM retry/offload.

**Why:** The pure-Rust executor is CPU-only and allocates/copies every initializer into its own host buffer, then allocates buffers per graph value (`crates/onnx-runtime-session/src/executor.rs:169-178,516-544,673-689,742-763`). External ONNX data is mmap-backed in the loader (`crates/onnx-runtime-loader/src/weights.rs:19-68,113-153`), but the model protobuf is read fully and executor construction copies mapped weights into allocations (`crates/onnx-runtime-loader/src/lib.rs:202-212`; executor citations above), so mmap is not on-demand execution/weight streaming. The designed `onnx-runtime-memory` crate is absent from the workspace (`Cargo.toml:17-31,52-64`); `MemoryPlanner::plan()` remains design pseudocode with `todo!()` (`docs/ORT2.md:1195-1260`).

The shipped KV tiering is narrower than its names: both “GPU” and CPU page tiers are host RAM bookkeeping (`crates/onnx-genai-kv/src/tiered.rs:1-8`), and promotion/demotion only changes `Page.device` (`crates/onnx-genai-kv/src/page_table.rs:877-925`). `LocalTieredConnector` retains authoritative host KV bytes, explicitly says real disk spill is not implemented, and never reports `LocalDisk` (`crates/onnx-genai-kv/src/local_tiered.rs:53-59,107-123`). Engine injection is limited to f32 `ZeroCopyRebind`; fixed device/shared/static KV cannot be handed off (`crates/onnx-genai-engine/src/connector_bridge.rs:9-28`; `crates/onnx-genai-engine/src/decode.rs:297-307`).

The ORT-backed production session may retry the *whole session* on CPU when non-strict accelerator session creation fails (`crates/onnx-genai-ort/src/session.rs:176-226`), which can let a VRAM-oversized model run if it fits RAM. That is not mixed offload or streaming, and there is no runtime CUDA-OOM→host migration: CUDA allocation errors are wrapped as `KernelFailed` and returned (`crates/onnx-runtime-ep-cuda/src/runtime.rs:139-147`; `crates/onnx-runtime-ep-cuda/src/error.rs:13-20`). Device KV is experimental/opt-in and otherwise uses CPU buffers (`crates/onnx-genai-ort/src/session.rs:545-582,804-818`; `crates/onnx-genai-ort/src/decode.rs:811-859`).

**Gap / recommended order:**
1. Implement `onnx-runtime-memory`: liveness/interference planning, per-tier arenas, accounting, eviction, and the §26.11 governor.
2. Preserve mmap-backed weights through execution; add per-layer placement, pinned staging, async prefetch, and CUDA/EP host↔device copies instead of eager copying.
3. Integrate placement with scheduling and non-copy views; add dynamic activation spill/reload.
4. Make paged KV storage physically device/host backed, bind pages to attention, and support shared/static-cache snapshot/restore.
5. Implement disk-backed KV/activation/weight pages with bounded mmap/direct-I/O, eviction policy, and end-to-end OOM fault/retry semantics.
6. Validate with models exceeding VRAM and then RAM, asserting bounded residency and correct output.

---

### 2026-07-15: Review of Deckard Concat/Slice memory-efficiency
**By:** Pris
**Verdict:** 🟢 APPROVE
**What:** Reviewed commit `69a1633`: only `slice.rs` and Deckard's decision record changed. Concat, executor, and session code are untouched. The new Slice test exercises a large contiguous retained suffix.
**Why:** The memcpy suffix is entered only after every included trailing axis has `start == 0`, `step == 1`, and `count == input_dim`; therefore neither step>1 nor negative-step axes can enter it. The dense source layout makes such a suffix contiguous. Existing tests cover step>1, negative steps, omitted axes with supplied steps, negative/clamped indices, and multi-axis slicing; 203 CPU-EP unit tests passed. Build and `clippy -D warnings` passed (the test build emitted only the pre-existing unused `bf16_bits` warning).

---

### Pris review 3: CPU Range allocation fix

**Verdict: 🟡 approve with a test-coverage follow-up.**

Reviewed `squad/cpu-coverage-wave` at `7911064` in a detached reviewer worktree.

## REJECT#2 resolved

Range supports exactly `Float32` and `Int64`; both paths invoke the same allocation guard before allocating or producing output:

```rust
let bytes = count.checked_mul(elem);
if bytes.is_none_or(|b| b > isize::MAX as usize) {
    return Err(EpError::KernelFailed(...));
}
let mut out = Vec::new();
out.try_reserve(count).map_err(|_| EpError::KernelFailed(...))?;
```

For `Range(0_i64, i64::MAX, 1_i64)`, `count * 8` exceeds `isize::MAX`; the guard therefore returns `KernelFailed` before `Vec` allocation. The focused overflow test passes. `Int64` values are calculated in `i128` then fallibly converted, so no integer overflow panic remains in the generation loop.

## REJECT#1 remains resolved

Float32 uses `float_range_count` and indexed construction (`start + i as f32 * delta`), not a `value += delta` termination loop. The no-progress test and descending fractional-delta test pass.

## Validation

- `cargo build -p onnx-runtime-ep-cpu`: passed.
- `cargo test -p onnx-runtime-ep-cpu --lib`: 217 passed, 0 failed.
- Focused Range tests: 5 passed, 0 failed.
- `cargo clippy -p onnx-runtime-ep-cpu --lib -- -D warnings`: passed.

## Follow-up

Tests cover a negative delta and the unaddressable `Int64` guard, but do not explicitly cover an empty range or a non-even negative fractional interval. The count logic handles both (`0` for direction-mismatched bounds; ceiling division for non-even intervals); add those regression tests in subsequent coverage work.

---

### 2026-07-15: Softmax + Reduce ported to cuDNN
**By:** Roy
**What:** Standalone Softmax now uses `cudnnSoftmaxForward` (`ACCURATE`, INSTANCE for legacy and CHANNEL for opset 13); ReduceSum/ReduceMean use `cudnnReduceTensor` (ADD/AVG) with RAII workspace and no indices. f32 retains the existing NVRTC fallback when cuDNN is unavailable; f16/bf16 return the actionable cuDNN-unavailable error because no handwritten path exists. Added modular softmax/reduce dispatch, raw EP-buffer adapters, mode/op mapping, and scratch-size helpers in `cudnn/mod.rs`.
**Why:** Implements the library-first mapping in `docs/CUDA_STRATEGY.md`: cuDNN supplies NVIDIA's per-architecture compatibility and tuning while preserving toolkit-free dynamic loading and a correct f32 fallback.

---

### Decision: Range kernel guards against unaddressable output sizes

- **Author:** Sapper (Rust correctness)
- **Branch:** `squad/cpu-coverage-wave`
- **Fix commit:** `7911064` (on top of `89c5027`, no rebase/reset)
- **Reviewer:** Pris 🔴 (re-review pending; not self-merged)

## Problem
`RangeKernel` (crates/onnx-runtime-ep-cpu/src/kernels/sequence.rs) could panic on
user-controlled inputs. For very large Int64 ranges the element `count` fits
`usize` on 64-bit, but `count * size_of::<i64>()` exceeds `isize::MAX`, so
`Vec` capacity/index math panicked ("capacity overflow") instead of returning an
error. ONNX kernels must never panic on user inputs.

## Fix
- Added helper `alloc_range_output::<T>(count)` that:
  1. Checks `count.checked_mul(size_of::<T>())` and rejects if it overflows or
     exceeds `isize::MAX` — returns `EpError::KernelFailed` with message
     `"Range output too large: N elements (M bytes) exceeds addressable limit"`.
  2. Uses `Vec::try_reserve(count)` so an over-large-but-technically-representable
     request still fails cleanly as a kernel error rather than aborting.
- Both cited sites reuse this guard **before** allocating:
  - **Int64 path (~L118-126):** `alloc_range_output::<i64>(count)?` then a
    push loop keeping Deckard's i128 overflow-checked value math.
  - **Float32 path (~L102-104):** same guard via `alloc_range_output::<f32>(count)?`
    then `extend`, since the float path had the same exposure.
- Count math (i128 up-front ceil/max, `start + i*delta`) is unchanged; this only
  ADDS the addressability guard.

## Error type reused
`onnx_runtime_ep_api::EpError::KernelFailed(String)` — the standard variant used
throughout this file (Tile, CumSum, Range).

## Test
`kernels::sequence::tests::range_int64_overflow_returns_error` —
`Range(start=0, limit=i64::MAX, delta=1)` asserts `Err(EpError::KernelFailed(_))`.
The guard triggers before allocation, so no exabytes are ever requested.

## Gate results (worktree)
- `cargo build -p onnx-runtime-ep-cpu` — Finished, clean.
- `cargo test -p onnx-runtime-ep-cpu --lib` — **217 passed; 0 failed** (was 216 + new test), no panic/abort.
- `cargo clippy -p onnx-runtime-ep-cpu --lib -- -D warnings` — clean.
- `cargo fmt -p onnx-runtime-ep-cpu -- --check` — clean.

---

### 2026-07-15: Review of Batty If/Loop/Scan efficiency

**By:** Sapper
**Verdict:** 🟡 SHIP-WITH-NOTES
**Commit:** `e2da515` on `squad/control-flow-efficiency`

**What:** Batty rebuilt the control-flow executor path (`executor.rs`) so Loop/Scan
build the child subgraph executor once per stable input-shape signature, prepare
closed-over captures once (owned snapshots), reuse iter/cond scalars and Scan
slice tensors across iterations via a new `Tensor::overwrite_bytes`, and stream
scan outputs into a single-allocation `TensorStackAccumulator` (no retained
per-step tensors, no second stacking pass). If prepares/runs only the selected
branch. New `ControlFlowStats { subgraph_builds, subgraph_runs }` makes reuse
deterministically testable. Build + tests + clippy all green.

**Why (per check):**

1. **Shape-change rebuild (top risk) — CORRECT, but UNTESTED.**
   `run_subgraph` rebuilds when `built_shapes.len() != externals.len()` or any
   external's current `shape` differs from the built shape, over
   formals **and** captures (executor.rs ~L396-416). On mismatch it builds a
   fresh child `Executor` and replaces the cache entry, so a stale-shaped plan
   is never reused → no OOB / wrong results. The dropped name-signature check is
   safe: for a fixed `(NodeId, attr_key)` the body graph is invariant, so
   formal/capture names never change. **Gap:** no test exercises a shape-varying
   body, so the rebuild path is proven only by inspection, not by a test. This
   is the sole reason for 🟡.

2. **Loop-carried aliasing — SAFE, no clobber.** `run_scoped` copies every input
   into child buffers (`write_host`, L965) and collects outputs into fully
   **owned** contiguous tensors (`from_raw_in`, L1013/1037). `carried`/`state`
   hold those owned outputs; each iteration `formal` only *borrows* them
   read-only, `drop(formal)` precedes `carried.clear()/extend`. Reused parent
   buffers (`iter_tensor`, `cond_tensor`, `scan_slices`) are overwritten only
   *after* the prior step fully copied them into the child. No output buffer is
   read as the next input while live.

3. **Scan single-buffer accumulation — CORRECT.** `TensorStackAccumulator::push`
   appends element bytes in iteration order; `finish` prepends `len` as the new
   leading axis → byte-identical layout to the old `stack_new_leading_axis`. No
   off-by-one on the scan axis. Zero-trip returns `(Float32, [0], [])`, matching
   prior behavior. Per-step shape/dtype mismatch errors loudly rather than
   corrupting. Verified by `scan_forward_axis0_...` and `loop_..._scan` tests.

4. **Captures once — SAFE.** Captures are owned snapshots
   (`value_tensor`→`contiguous_bytes`→`to_vec`), re-copied into the child each
   run from the immutable `prepared.captures`. In-place body mutation cannot
   leak across iterations — semantically identical to the old per-run clone.

5. **View-liveness (my rule) — PRESERVED.** Every capture, Scan slice, and body
   output crosses the boundary as owned contiguous bytes; the child clears its
   own run-scoped `views`/`pinned` each `run_scoped`. No view alias outlives its
   source under the reuse scheme: parent-side reused buffers are overwritten only
   after their bytes are materialized into the child, and the source's live
   interval subsumes every use. No free/overwrite of a source while a view of it
   is live.

6. **If — CORRECT.** `attr_key` selects then/else; only that branch is prepared
   and run (untaken branch = zero build, zero run, confirmed by the stats
   assertions). Taken-branch outputs are stored as owned copies.

**Build gate (offline, ran):**
- `cargo build -p onnx-runtime-session` ✅
- `cargo test -p onnx-runtime-session` ✅ (all suites; 5/5 control_flow, incl.
  `loop_many_iterations_accumulates_correctly` proving **1 build / 1000 runs**)
- `cargo build -p onnx-runtime-ep-cpu` ✅ / `cargo test --lib` ✅ (202 passed)
- `cargo clippy -p onnx-runtime-session -- -D warnings` ✅ (clean, no warnings)

**Test-claim audit:** 1-build/1000-run ✅ proven; Scan value+layout ✅ proven;
If selected-branch-only ✅ proven. **Missing:** a shape-varying Loop/Scan body
test to prove the rebuild trigger fires and produces correct results.

**Note to Batty (address before merge, non-blocking):** add a control-flow test
with a data-dependent / shape-varying body across iterations asserting
`subgraph_builds > 1` **and** correct outputs, to lock the #1 rebuild path.

---

### 2026-07-14: Review of Holden cuDNN backend foundation
**By:** Tyrell
**Verdict:** 🟢 APPROVE
**What:** Independent architectural review (I authored `docs/CUDA_STRATEGY.md`; I am not the author of this code) of `squad/cuda-cudnn-backend` @ `d70a6d9`: the shared, lazy, stream-bound `CudnnBackend` (handle + validated tensor-descriptor foundation) that the library-first Softmax/Reduce/Conv/Pool ports will build on. No op is wired through it yet — correct scope discipline.
**Why (per-check):**
- **dynamic-loading (CRITICAL):** ✅ `cudarc` `dynamic-loading` retained; only the `cudnn` feature was added alongside the existing `cublaslt`/`nvrtc`/`f16`. No `build.rs`, no `#[link]`, no link flag added. `cargo build -p onnx-runtime-ep-cuda` succeeds offline with NO libcudnn present. libcudnn is only dlopen'd at first cuDNN op invocation. macOS/no-CUDA path intact — the EP is an optional feature and the module carries no new cfg breakage; `cargo check -p onnx-runtime-python --features .../cuda` passes.
- **handle lifecycle:** ✅ Created lazily via `Mutex<Option<Arc<Cudnn>>>`; reuses the EP's existing `CudaStream`/`CudaContext` (`Cudnn::new(self.stream.clone())`) — no second context. `with_handle` binds the context to the calling thread before use. RAII: cudarc owns the native handle; `Drop` binds context then drops. No leak/double-free. Thread-safety posture (`unsafe impl Send/Sync` + serializing Mutex) is stricter than, and consistent with, the existing `CublasLt`.
- **dtype mapping:** ✅ f32/f16/bf16 → `CUDNN_DATA_FLOAT`/`HALF`/`BFLOAT16` via cudarc's `CudnnDataType::DATA_TYPE` (verified by unit test); unsupported dtypes rejected with an actionable error. Descriptors thinly wrap cudarc's `TensorDescriptor<T>`/`create_nd_tensor` — no redundant re-FFI. Dim/stride validation is sound: rank≥1, matching dims/strides, rejects zero dims, rejects non-positive strides, i32 overflow-checked, low-rank padding to 4D with correct leading strides (test-covered).
- **error path:** ✅ `ensure_cudnn_available` probes `sys::is_culib_present()` and returns `cudnn_unavailable()` (new `EpError` variant path) with install guidance; `initialize_cudnn` wraps creation in `catch_unwind`. No `unwrap`/panic on the load path; mutex-poison handled.
- **conventions & modularity:** ✅ Mirrors `blas.rs`/error-helper/runtime-wiring patterns; not a monolith; naming consistent. `runtime.rs` exposes `cudnn()` only — no op dispatch yet.
- **tests:** ✅ 62 lib tests pass in 0.50s with no GPU; the new tests exercise pure logic (dtype/descriptor/dim-stride/error-path) via injected probes/closures and do not require a device. Adequate for a foundation.
**Build gate:** `cargo build` ✅ · `cargo test --lib` (62 passed) ✅ · `cargo clippy -- -D warnings` clean ✅ · Python `cuda` feature `cargo check` ✅. "No GPU execution here" is expected and not held against the code.
**Reviser if rejected:** N/A (approved).

---

### 2026-07-15: Review of Roy Softmax+Reduce cuDNN port

**By:** Tyrell
**Verdict:** 🟢 APPROVE

**What:** Independent, non-author review of Roy's first library-port (commit `d070076`,
branch `squad/cuda-softmax-reduce-cudnn`): Softmax → `cudnnSoftmaxForward` (ACCURATE) and
ReduceSum/ReduceMean → `cudnnReduceTensor` (ADD/AVG), built on the cuDNN backend. Reviewed
the FFI/setup logic against ONNX semantics (GPU math is untested on this host — expected).
Build gate run offline: `cargo build -p onnx-runtime-ep-cuda` ✓, `cargo test -p
onnx-runtime-ep-cuda --lib` → **67 passed** ✓, `cargo clippy -p onnx-runtime-ep-cuda -D
warnings` clean ✓, `cargo check -p onnx-runtime-python --features .../cuda` ✓.

**Why (per-check):**

1. **Softmax mode/axis (CRITICAL) — CORRECT for BOTH v13 and legacy.**
   `cudnn_softmax_spec` builds a 4-D view `[outer, axis_dim, inner, 1]` with contiguous
   strides `[axis_dim*inner, inner, 1, 1]`.
   - Opset-13 → `CHANNEL` mode: cuDNN normalizes over C=`axis_dim` at each `(N=outer,
     H=inner, W=1)` → softmax over exactly the ONNX `axis`, independently per (outer,inner). ✓
   - Legacy (coerce_2d) → `softmax_view` yields `inner==1`, giving `[outer, axis_dim, 1, 1]`
     with `INSTANCE` mode → cuDNN normalizes over C·H·W = `axis_dim` per `N=outer` → matches
     the 2-D coerced `[outer, inner=axis_dim]` softmax. ✓
   - Negative axis resolved by `resolve_axis` before feeding `softmax_view`; test
     `cudnn_layout_selects_onnx_semantics` locks dims/strides/mode for both paths.

2. **Reduce shape + workspace — CORRECT.** ONNX output shape (`reduced_output_shape`,
   keepdims-aware, squeeze on keepdims=0) is validated against `outputs[0].shape`. The cuDNN
   output descriptor (`cudnn_reduce_specs`) keeps reduced axes as size-1 over the same
   contiguous storage; `TensorDescriptorSpec::new` leading-pads both input and output to 4-D
   consistently, so cuDNN's input-vs-output extent comparison reduces exactly the masked
   axes. AVG=mean, ADD=sum. Multi/negative axes + noop_with_empty_axes handled by
   `resolve_reduce_mask`; the no-reduction / rank-0 identity path does a `dtod` copy (correct
   for any dtype). Workspace sized by `get_workspace_size` (min 1), `alloc_zeros` RAII
   `CudaSlice` freed on scope exit — no leak, no under-size; `NoIndices` (indices_bytes=0).

3. **Fallback selection — SOUND, no gap.** Both kernels branch `if
   cudnn().is_available()` FIRST for ALL dtypes → **f32 genuinely prefers cuDNN when present**
   (NVRTC only when the loader probe fails); the port is not accidentally NVRTC-always.
   `is_available()` and `with_handle`'s guard share the same `cudnn_library_present()` probe,
   so no true/false skew. When cuDNN is absent: f32 → NVRTC fallback (identical op
   semantics); f16/bf16 → `with_handle(|_| Ok(()))` returns the backend's actionable
   `cudnn_unavailable()` error before the closure runs — no panic/unwrap. Max/Min map to
   `None`, skip cuDNN, stay f32 NVRTC.

4. **Backend usage — correct.** New `softmax`/`reduce` helpers are modular, dtype-dispatched
   via the existing descriptor enum. Handle is `Cudnn::new(self.stream)`-bound and
   `RawDevice` uses `SyncOnDrop::Record(None)` (no implicit sync) — safe because each kernel
   calls `self.runtime.synchronize()` after `with_handle`, and the cuDNN stream == the EP
   stream, so ordering against other kernels is preserved.

5. **Coverage doc — updated accurately:** Softmax/ReduceSum/ReduceMean now show cuDNN backend
   with the f32 NVRTC fallback noted; Max/Min remain NVRTC. No op coverage dropped.

6. **No regression:** 67 lib tests pass; they meaningfully exercise the pure setup logic
   (mode selection, axis→descriptor dims/strides, reduce-op mapping, size-one reduced axes,
   workspace-alloc sizing). Clippy clean, python cuda feature checks.

**Non-blocking notes (no action required):** GPU numerics remain unverified by design (no
GPU/cuDNN on host) — first real-hardware run should spot-check a v13 softmax over a
non-trailing axis and a multi-axis ReduceMean against the CPU EP to confirm cuDNN's
CHANNEL/AVG results match element-for-element.

---

## Archived decision index

- 2026-07: Historical decision archive — [`2026-07.md`](decisions-archive/2026-07.md).

---

### 2026-07-15: Zero-copy mmap-backed initializer borrowing
**By:** Rachael; soundness follow-up by Zhora
**Status:** ✅ merged (`3df84d0`, `e0c9669`)
**What:** `DeviceBuffer` now distinguishes owned and borrowed storage and has a `from_borrowed_parts` constructor, allowing the CPU executor to borrow aligned mmap-backed external initializers instead of copying them into RAM. Borrowed buffers are never deallocated by providers. The executor only borrows producer-less initializers, and the loader rejects an initializer that is also a node output with `LoaderError::InitializerHasProducer`.
**Why:** This preserves the mmap owner lifetime while avoiding the resident-RAM duplicate for eligible weights. Restricting the borrow path to producer-less initializers closes the write-through/read-only-mmap soundness gap that the initial review missed.
**Validation:** Regression coverage proves aligned borrowing, unaligned fallback-to-copy, no-op deallocation, and rejection of initializer/output reuse.

#### Sources: `deckard-review-weight-streaming.md`, `zhora-weight-streaming-fix.md`

---

### 2026-07-15: DLPack zero-copy export with explicit FFI ownership contract
**By:** Chew; reviewed by Hassan; hardened by Ana
**Status:** ✅ merged (`6fdccc8`, `e38eaee`)
**What:** Added `onnx-runtime-dlpack` and Python `run_with_values()` returning `NxrtValue` with `__dlpack__` and `__dlpack_device__`. CPU tensors export by borrowing the tensor allocation while a managed owner preserves its lifetime. The public raw-pointer export constructors are `unsafe`, validate dimensions and stride lengths, reject big-endian builds, and emit null `DLTensor.data` for zero-element tensors. Writable aliasing is documented.
**Why:** Consumers such as NumPy and PyTorch can consume runtime output without an additional host copy while retaining a clear one-owner DLPack capsule/deleter contract.
**Validation:** Non-author review approved the capsule handshake, ABI layout, Send/GIL behavior, owner lifetime, and compatibility; Rust, build/clippy, and Python DLPack tests passed.

#### Sources: `chew-dlpack-export.md`, `hassan-review-dlpack-export.md`, `ana-dlpack-hardening.md`

---

### 2026-07-15: KV insertion architecture — Mobius functional GQA first, paged attention later
**By:** Tyrell; fact check by Fact Checker; revised by coordinator directive
**Status:** ✅ decision-ready (`41c2fff`, `6e62902`)
**What:** For third-party `com.microsoft::GroupQueryAttention` exports, retain the sanctioned `past_present_share_buffer` compatibility path. For Mobius exports we control, stop stamping `past_present_share_buffer` as Phase 1: the same GQA graph then uses functional `ZeroCopyRebind`, requiring no new kernel and removing the fixed shared-buffer constraint. Phase 2 is a flag-gated vLLM-style device insertion plus block-table paged-attention contract.
**Why:** The exporter is a design lever for our models, while existing third-party exports require compatibility. ORT GQA shared-buffer behavior is sanctioned, standard `ai.onnx::Attention` opset 23/24 has cache semantics, and Hugging Face calls `cache.update()` inside the attention module; none of those facts make existing GQA transparently interchangeable with a paged ABI.
**Gate:** Functional GQA and any paged path require correctness traces plus M=1 p50/p99 decode latency within noise of the shared-buffer baseline. The paged path remains non-default until that gate passes; old GQA remains a frozen compatibility shim.

#### Sources: `tyrell-kv-insertion-eval.md`, `fact-checker-kv-insertion-da.md`, `tyrell-kv-mobius-exporter.md`, `coordinator-we-control-the-mobius-exporter-so-attention-op-cho.md`

---

### 2026-07-15: Keep the unsupported-op diagnostic probe unsupported
**By:** Taffey
**Status:** ✅ merged (`31e8a0e`)
**What:** `test_unsupported_op_is_actionable` now uses ONNX `Det` rather than `Sinh`.
**Why:** `Sinh` is implemented by the CPU coverage wave; `Det` remains unregistered, so the test continues to exercise the actionable unsupported-operation error path.

#### Source: `taffey-unsupported-op-test.md`


---

## 2026-07-15 — Ergo API

**What:** The PyO3 `nxrt` extension now exposes `nxrt.load(...)`, callable `InferenceSession`, ordered/keyword/mapping feed dispatch, `Outputs`, tensor-like `NxrtValue`, and input/output-name helpers while preserving ORT-compatible `run()` APIs. Explicit output selections are validated before execution. `bind_outputs` is an immutable, lock-free `BoundSession` proxy whose output subset is passed per call; it never mutates session state, so concurrent threads and async use cannot clobber one another.

**Why:** This adds a convenient, single-artifact API without duplicating feed/DLPack logic or changing compatibility behavior. Up-front output-name errors are actionable, and a per-call proxy is the safe composable alternative to global mutable selection. (Joi, Luv, Gaff; merged `5a8a269`, `5b2d565`, `fdc11b1`.)

## 2026-07-15 — DLPack / GPU interop

**What:** CPU DLPack import borrows contiguous row-major producer buffers with ownership guards, final-commit capsule consumption, checked shape/alignment arithmetic, pointer-identity coverage, and GIL-safe exactly-once foreign deleters; unsupported inputs retain copy fallback. CUDA `kDLCUDA` import/export preserves device ordinal and pointer identity, with stream synchronization on export. Commit validation compares the advertised and capsule **raw** `(device_type, device_id)` before normalization. The runtime explicitly rejects device-resident input or host reads at the current CPU-executor boundary rather than panicking or implying CUDA execution.

**Why:** Zero-copy interop must be safe across Python lifetimes and foreign/untrusted capsules, and a normalized device comparison could silently accept mismatches. CUDA transport is useful now, but full CUDA-session execution remains a separate epic; execution-dependent GPU tests are retained as xfail specifications. (Freysa, Zhora, Zuben, Mariette, Roy; merged `cc50ca1`, `50030fa`, `57843ee`.)

## 2026-07-15 — Loader validation and model legality

**What:** Raw-model legality validation now rejects ambiguous duplicate SSA producers, executable unresolved `ref_attr_name`, invalid low IR versions, empty IR>=3 imports, invalid scope shadows, and unsourced graph outputs, while deliberately excluding lint-only checks. There is no upper IR-version ceiling: future IR versions load unless a concrete unsupported feature requires a gate. Structural graph validation runs only after initializers attach; `build_graph` is crate-private. Model-local `FunctionProto` calls are inlined with recursive attribute binding, lexical scope/capture renaming, globally fresh names, alias identities, sparse-initializer handling, recursion/arity checks, and canonical default-domain opset merging. Custom-only non-empty opset imports are valid; synthesized default-domain identities add one appropriate default import.

**Why:** Validate semantics before IR information is coalesced, but validate the fully assembled graph to avoid rejecting legal initializer-backed values. ONNX evolution is broadly backward compatible, and default-domain spellings `""` and `ai.onnx` must not create duplicate imports. (Hodge, Deckard, Rachael, Sebastian, Wallace, Sapper, Howie; merged `98c6c00`–`051e0a5`.)

## 2026-07-15 — Standard LLM operators

**What:** Standard-domain CPU kernels and aligned shape rules landed for `Gelu` (v20), `RMSNormalization` (v23), `RotaryEmbedding` (v23), and `Swish` (v24). They are f32 reference kernels with exact/tanh GELU, NumPy-style unidirectional RMS scale broadcast, axis validation, RoPE 3D/4D and cache layouts, checked cache-offset arithmetic, position validation, and stable Swish. `Gelu` shape registration begins at v20 rather than v1.

**Why:** Transformer exports need standard ai.onnx primitives with kernel/shape membership kept in lockstep. The follow-up review corrected scale broadcasting, invalid axes, empty/overflowing RoPE accesses, and version registration without changing verified core math. (Joshi, Deckard, Sapper; merged `923c8bd`, `0549e1f`, `9fe94d4`.)

## 2026-07-15 — Opset 17–26 coverage

**What:** CPU coverage added `Split`, `Pad`, `ConstantOfShape`, `Sum`, `Mean`, `LogSoftmax`, `CastLike`, com.microsoft `BiasGelu`, `FastGelu`, `QuickGelu`, `SkipLayerNormalization`, and `SimplifiedLayerNormalization`; fused normalization honors independent absent beta/bias, requested output arity, axis, and inverse-std output. C1 shape rules now cover ArgMax/ArgMin, TopK, Tile, Range, CumSum, GatherND, NonZero, and relevant trig/activation unaries. Float `ShapeData` enables Range inference, with f32 arithmetic intentionally matching the CPU kernel. `Attention` v23/v24 coverage was hardened for overflow-safe scaling, scalar masks, softcap, v24 mode semantics, and `nonpad_kv_seqlen` causal **and** unconditional padding masks. The activation-memory planner also landed as a deterministic, executor-unwired IR planning crate.

**Why:** These C1/C2 dispatch and inference gaps blocked a large conformance wave; shape contracts must match actual kernel/version behavior. The static registration-vs-schema audit alone over-reports stale operators because old kernels can implement later migrations, so kernel↔shape comm-diff is the real gap signal. Numerical and optional-slot fixes preserve spec behavior rather than broadening unsupported types. (Wallace, Zhora, Cotton, Chew, Nandez, Freysa, Deckard, Sapper, Pris, Kaiser; merged through `7c06c39`.)

---

## 2026-07-14 — Wave 2 conformance coverage and workspace release

### CPU conformance fixes and coverage
**What:** Wave 2 added CPU `QuantizeLinear`/`DequantizeLinear` (opsets 10/13/19/21/23/25) and `DynamicQuantizeLinear` (11), including scalar, per-axis, and blocked parameters, ties-to-even rounding, and saturation; float8 and packed int4/uint4 remain unsupported. It also added generic N-D AveragePool, MaxPool, GlobalAveragePool, and GlobalMaxPool with f16/bf16/f32/f64 widening paths, aligned pool geometry, optional MaxPool indices, and storage-order support. `Split` shape inference now permits a valid zero final chunk (for example `dim=2, num_outputs=3` yields `[1,1,0]`) and CPU `Equal` supports broadcast Bool and fixed-width numeric tensors.

**Corrective reviews:** A non-author review found MaxPool `Indices` omitted the `nc * spatial_size` global batch/channel offset. Deckard fixed it with two-channel tests for both storage orders; re-review approved. A non-author review likewise found `Split` inference rejected its valid zero remainder; Chew aligned it with the CPU kernel and re-review approved. Quantization was independently approved; its only noted hardening opportunity is an extreme-value saturating add.

**Attention:** Bryant corrected CPU Attention's optional `qk_matmul_output` mode semantics: modes 1 and 2 are not swapped in opset 24. The Shape kernel now implements opset-15 `start`/`end` slicing, preventing executor allocation/write byte-count mismatches. This reduced direct Attention conformance failures from 11 to 2 (the remaining f32-only-policy fp16 cases); expanded Attention still depends on primitive coverage (`Equal`, `Mod`, `And`, `Trilu`) and a Pad byte-count fix. Non-author review approved both fixes and confirmed tests and clippy gates.

**Why:** Kernel and shape-inference sizing and ONNX indexing contracts must agree exactly, particularly for allocation and downstream MaxUnpool use. The remaining gaps are intentionally scoped rather than hidden by broad compatibility behavior.

### ONNX_RS and scheduling sequencing
**Decision:** Start ONNX_RS in the monorepo using existing `onnx-runtime-ir`, first with self-contained, round-trip-testable `crates/onnx-text` (ONNX text parser/printer) and `crates/onnx-json` (IR/JSON serialization). Defer proto/schema/checker/version-converter/umbrella/PyO3 phases until their dependencies are ready. Defer scheduling work until paged-memory and session-lifecycle substrates exist.

**Why:** ONNX_RS is a sound independent pure-Rust concern; text and JSON offer the lowest-risk initial slices. Scheduling is coupled to not-yet-implemented hibernation, page-out, PagerSchedulerAPI, and EP negotiation.

### Workspace version and tracer publication
**What:** All `onnx-runtime-*` crates use `version.workspace = true`; internal workspace dependency pins are unified at `0.1.0`; and `onnx-runtime-tracer` was added to the publish workflow. This landed as `554cbfc` following 🟢 non-author review by duck-ver.

**Why:** The workspace had diverged between runtime `0.1.0-dev.1` and GenAI `0.1.0`. Moving all crates forward to `0.1.0` is monotonic and avoids a crates.io downgrade; publishing tracer closes its missing release-workflow coverage.

**Sources:** `bryant-attention-conformance.md`, `duck-attn-review.md`, `kaiser-quantize-linear.md`, `duck-quant-review.md`, `howie-pooling.md`, `duck-pool-review.md`, `deckard-maxpool-indices-fix.md`, `duck-pool2-review.md`, `mariette-split-equal.md`, `duck-spliteq-review.md`, `chew-split-zerochunk-fix.md`, `duck-spliteq2-review.md`, `coordinator-onnx-rs-scheduling-design-review-phased-implementa.md`, `tycho-version-unify.md`.
