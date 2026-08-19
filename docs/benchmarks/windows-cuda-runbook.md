# Windows CUDA runbook (native EP, pip-wheel CUDA)

A copy-pasteable procedure to run the onnx-genai **native CUDA execution
provider** on a **Windows** development box where there is **no CUDA toolkit
installed** — every CUDA library comes from pip wheels under an Anaconda
`site-packages\nvidia` tree. This is the Windows counterpart to
[`H200-CUDA-runbook.md`](H200-CUDA-runbook.md) (Linux/H200). It is a **setup
doc, not a benchmark**: it contains no performance numbers, and any sample
output is illustrative only.

> **Status when written (2026-08-19):** every command below was executed on the
> primary Windows dev box (see §0) and the results are recorded as
> **passed / failed / ignored**. Where the author's field notes did not match
> what actually happened on this machine, the discrepancy is called out inline
> under **Correction**. The validated smoke command (§2) produces coherent text
> on the GPU.

---

## 0. This machine (measured)

| Property | Value (measured) |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU, 8 GiB, `sm_89` (WDDM) |
| Driver | 591.55, reports **CUDA 13.1** (`nvidia-smi`) |
| CUDA toolkit | **none installed** — no `nvcc` on `PATH`, `CUDA_PATH` unset, nothing under `C:\Program Files\NVIDIA GPU Computing Toolkit` |
| CUDA libs | pip wheels under `…\anaconda3\Lib\site-packages\nvidia\` |
| Python | Anaconda base (`python` on `PATH`) |

Confirm the "no toolkit" reality before assuming this runbook applies to you:

```powershell
Get-Command nvcc -ErrorAction SilentlyContinue      # -> nothing
"CUDA_PATH=$env:CUDA_PATH"                           # -> CUDA_PATH=
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit"   # -> False
nvidia-smi                                           # GPU + driver CUDA version
```

The native CUDA EP does **not** need a toolkit: it JIT-compiles its kernels with
NVRTC (shipped in the `nvidia-cuda-nvrtc` wheel) and dlopens cuBLASLt / cuDNN /
the CUDA runtime from the wheel `bin` directories. It only needs those wheel
DLLs and the CUDA runtime **headers** discoverable at run time.

### Which wheel generation works

Both cu12 and cu13 wheel generations are installed side by side, e.g.:

```
nvidia-cublas            13.6.0.2     nvidia-cublas-cu12         12.9.2.10
nvidia-cuda-runtime      13.3.29      nvidia-cuda-runtime-cu12   12.9.79
nvidia-cuda-nvrtc        13.3.33      nvidia-cuda-nvrtc-cu12     12.9.86
nvidia-cudnn-cu13         9.25.0.15   nvidia-cudnn-cu12          9.24.0.43
```

The two generations use **different on-disk layouts**:

- **cu13** puts everything under one root: `nvidia\cu13\bin\x86_64\` (DLLs:
  `cublasLt64_13.dll`, `cudart64_13.dll`, `nvrtc64_130_0.dll`, …) **and**
  `nvidia\cu13\include\` (`cuda_fp16.h`, `cuda_bf16.h`, …).
- **cu12** uses per-package dirs: `nvidia\cublas\bin\cublasLt64_12.dll`,
  `nvidia\cuda_runtime\include\cuda_fp16.h`, etc.
- **cuDNN is separate in both**: `nvidia\cudnn\bin\cudnn64_9.dll`.

**Validated combination (this doc):** the **cu13** cuBLAS/runtime/NVRTC root
(`cu13\bin\x86_64` + `cu13\include`) plus the shared `nvidia\cudnn\bin`. That is
what ran. The cu12 layout was **not** validated here.

---

## 1. Environment setup (copy-paste)

Env vars **do not persist** between shells here, so this block must be run in the
**same shell invocation** as the run.

```powershell
# Concrete example for this box. See "General discovery" below to adapt it.
$sp  = 'C:\Users\justinchu\AppData\Local\anaconda3\Lib\site-packages\nvidia'
$cu  = "$sp\cu13"
$env:CUDA_PATH = $cu          # so NVRTC finds cuda_fp16.h / cuda_bf16.h under $cu\include
$env:CUDA_HOME = $cu
$env:PATH = "$cu\bin\x86_64;$sp\cudnn\bin;" + $env:PATH   # cuBLASLt + CUDA runtime + cuDNN
```

What each piece fixes maps one-to-one to the three failures in §3:
`$cu\bin\x86_64` → cuBLASLt; `CUDA_PATH`/`CUDA_HOME` → the NVRTC headers;
`$sp\cudnn\bin` → cuDNN.

### General discovery (other machines)

Find the wheel root, then locate the three things by name. This is verified to
work on this box:

```powershell
$nv = python -c "import nvidia, os; print(os.path.dirname(nvidia.__file__))"
# cuBLASLt DLL directory (prefer cu13 if both are present):
Get-ChildItem -Recurse $nv -Filter "cublasLt64_*.dll" | Select-Object FullName
# CUDA runtime headers (the directory that contains cuda_fp16.h -> set CUDA_PATH one level up if needed):
Get-ChildItem -Recurse $nv -Filter "cuda_fp16.h" | ForEach-Object { $_.Directory.FullName }
# cuDNN runtime:
Get-ChildItem -Recurse $nv -Filter "cudnn64_9.dll" | Select-Object FullName
```

For cu13, `CUDA_PATH` points at the `cu13` root because its `include\` sits
directly under it. For cu12 wheels the headers live under
`nvidia\cuda_runtime\include`, so `CUDA_PATH` must point at `nvidia\cuda_runtime`
(the parent of `include`).

---

## 2. Known-good smoke command (validated)

Build the native profiler once (the `cuda` feature enables the CUDA EP; the
default `cuda-13000` API-version feature is already on):

```powershell
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native
```

Then, in the **same shell** as the §1 block, run one short decode. Validated
against the **`models\qwen2.5-0.5b`** package (fp16 GQA decoder; ~1.98 GB of
weights — fits the 8 GiB card):

```powershell
& .\target\release\profile_native.exe `
    --model models\qwen2.5-0.5b --ep cuda --steady `
    --tokens 32 --warmups 1 --runs 1 --prompt "The capital of France is"
```

**Result: passed.** Illustrative (non-timing) output — the point is *coherent
GPU text*, not the numbers:

```
profile_native: model=models\qwen2.5-0.5b ep=Cuda backend=native
generated_text: " Paris. It is the largest city in Europe and the third largest city in the world. ..."
steady_median: backend=native prefill=… ms decode=… ms/token throughput=… tok/s
```

> **Do not read timings from a cold run.** A cold NVRTC kernel cache inflates the
> first run by orders of magnitude. This is a setup gate; benchmark separately
> once the cache is warm and the box is quiet.

**Ignored:** `models\qwen2.5-0.5b-q4` was *not* usable here — the native engine
rejects it before any CUDA work with
`model.io.position_ids_input declares port 'position_ids', but the graph exposes
[…]` (a model-packaging mismatch, unrelated to the CUDA setup). Use the fp16
`qwen2.5-0.5b` package for the smoke test.

---

## 3. The three failures you get without §1

Each of these **looks like a missing engine capability but is only a missing DLL
or header.** They appear in this order as you add pieces of the env. All three
are **loud, actionable errors from the native EP** — which is the good outcome
(contrast §4).

**Failure 1 — at EP init (nothing on `PATH`).** *Reproduced verbatim here:*

```
cuda_ep: CUDA CublasLt library not found; tried
["cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt.dll"];
CPU execution remains available
```
Fix: put `…\cu13\bin\x86_64` on `PATH`.

**Failure 2 — at the first f16/bf16 kernel (cuBLASLt on `PATH`, no `CUDA_PATH`).**
The EP's per-op pre-check emits:

```
cuda_ep <Op>: f16/bf16 NVRTC kernels require cuda_fp16.h and cuda_bf16.h.
Install the CUDA runtime headers (for pip CUDA 13: `pip install nvidia-cuda-runtime`;
alternatively set CUDA_HOME/CUDA_PATH).
```
Fix: set `CUDA_PATH` / `CUDA_HOME` to the wheel root whose `include\` holds
`cuda_fp16.h`.

> **Correction (verified):** with the `qwen2.5-0.5b` model the *first* half kernel
> is an `RMSNormalization`, not a `Mul`, and its code path does **not** run the
> friendly pre-check above. What actually printed here was the **raw NVRTC
> compiler error**:
> ```
> compiling NVRTC CUBIN module 'rmsnorm_bf16_v4' failed (NVRTC_ERROR_COMPILATION);
> rmsnorm_bf16_v4(2): catastrophic error: could not open source file "cuda_fp16.h"
> ```
> Same root cause, same fix (`CUDA_PATH`). The friendly `cuda_ep <Op>:` wording
> does exist in the code (`require_nvrtc_half_headers`) and fires for the ops
> that call it — just not for the RMSNorm that this model hits first.

**Failure 3 — at the first cuDNN-backed op (headers set, cuDNN not on `PATH`).**
The EP would emit:

```
cuda_ep: cuDNN (libcudnn.so.9 / cudnn64_9.dll) was not found at runtime.
Install it with 'pip install nvidia-cudnn-cu13' or 'conda install -c nvidia cudnn',
or add the cuDNN library directory to the platform library search path.
```
Fix: put `nvidia\cudnn\bin` on `PATH`.

> **Correction (verified):** this failure did **not** occur for the
> `qwen2.5-0.5b` native path. With `CUDA_PATH` set and cuBLASLt on `PATH` but
> **cuDNN deliberately left off `PATH`**, the run still **succeeded** with
> coherent output — the native GQA decode compiles its own NVRTC softmax/attention
> kernels and never invokes a cuDNN op. The `cudnn64_9.dll` error is real code
> and can fire for a model whose graph reaches a cuDNN-backed op, so keeping
> `nvidia\cudnn\bin` on `PATH` (as in §1) is still recommended — but it was not
> the blocker for this model.

### Why "loud" matters

These native-EP errors name the exact missing file. Contrast the trap already
recorded in this repo: the **ORT** CUDA provider **silently falls back to CPU**
when `cublasLt64_13.dll` is missing, which once produced a false "ORT has no
Attention-24 kernel" conclusion. A loud failure that names the DLL is strictly
better than a silent one that inverts a benchmark.

---

## 4. `profile_native` gotcha (verified)

`--tokens` must be **strictly greater than** `--decode-skip` (default **8**),
because the steady window is timed from the token just before the first measured
one. `--tokens 8` with the default skip exits immediately:

```
Error: --tokens must be greater than --decode-skip
```
Use `--tokens 32` (or any value > the skip) for a smoke run. Verified both the
guard (`--tokens 8` fails) and a passing value (`--tokens 32`).

---

## 5. ORT-CUDA path (partly verified — read before using)

The steps above run the **native** EP. Running the **ORT** CUDA EP
(`--backend ort` / `ONNX_GENAI_BACKEND=ort` with `--ep cuda`) is different and
needs more than a `PATH` reorder:

- **Verified:** the ONNX Runtime that `ort-sys` downloads at build time is
  **CPU-only**. The only DLLs under
  `target\…\build\onnx-genai-ort-sys-*\out\ort-prebuilt\lib\` are
  `onnxruntime.dll` and `onnxruntime_providers_shared.dll` — there is **no**
  `onnxruntime_providers_cuda.dll` anywhere in `target\`.
- **Verified:** `ort-sys` loads ONNX Runtime by **explicit path** via
  `libloading::Library::new(&candidate.path)` (see
  `crates\onnx-genai-ort\ort-sys\src\lib.rs`), driven by `ONNX_GENAI_ORT_LIB` /
  `ONNX_GENAI_ORT_LIB_DIR`. Because the library is chosen by absolute path
  *before the first ORT API call*, **reordering `PATH` alone cannot change which
  `onnxruntime.dll` is used** — you must point `ONNX_GENAI_ORT_LIB_DIR` at a
  CUDA-enabled ORT install.
- **Verified:** `ONNX_GENAI_EP_FALLBACK` and `ONNX_GENAI_ORT_LIB_DIR` are real,
  documented runtime-config env vars (`crates\onnx-genai-runtime-config`), and
  the EP-selection code tells you to set `ONNX_GENAI_EP_FALLBACK=1` to opt into a
  visible CPU retry.

So, to run ORT-CUDA you need: a CUDA-enabled `onnxruntime.dll` (with the CUDA
provider DLLs beside it) reachable via `ONNX_GENAI_ORT_LIB_DIR`, those provider
DLLs + the CUDA/cuDNN wheels on `PATH`, and `ONNX_GENAI_EP_FALLBACK=1` if you
want a visible CPU retry instead of a hard failure.

> **Unverified:** an end-to-end ORT-CUDA run was **not** performed here (no
> CUDA-enabled ORT is installed on this box). The claims above are code- and
> artifact-verified; the actual ORT-CUDA execution is not.

---

## 6. Shared-box hygiene (do this every time)

This box is shared with build/measure agents. A single leaked `profile_native`
once spun a core for hours. **`Get-Process -Id <pid>` gives false negatives
here** — check by name plus a CPU-time delta, and confirm with CIM:

```powershell
Get-Process -Name profile_native -ErrorAction SilentlyContinue |
    Select-Object Id, Name, CPU        # empty output == clean
# If anything lingers, confirm it is truly gone by PID:
Get-CimInstance Win32_Process -Filter "ProcessId = <pid>"
```

Every run in this doc was confirmed exited (`Get-Process -Name profile_native`
returned nothing after each). If a run hangs, stop it by its specific PID
(`Stop-Process -Id <pid>`), then re-confirm with the two checks above.

---

## 7. Measured vs. inferred

| Claim | Status |
|---|---|
| No CUDA toolkit; libs come from `site-packages\nvidia` wheels | **measured** (passed) |
| Failure 1 (`CublasLt … not found`) verbatim without cuBLASLt on `PATH` | **measured** (reproduced) |
| Failure 2 (missing `cuda_fp16.h`) — surfaced as a raw NVRTC error for RMSNorm | **measured** (reproduced; wording corrected) |
| Failure 3 (cuDNN) for `qwen2.5-0.5b` native path | **measured: did NOT fire** (correction) |
| cu13 cuBLAS/runtime/NVRTC + `nvidia\cudnn\bin` combination runs | **measured** (passed) |
| cu12 layout works | **not tested** (inferred to also expose the same DLLs) |
| Smoke run produces coherent GPU text on `qwen2.5-0.5b` | **measured** (passed) |
| `qwen2.5-0.5b-q4` usable | **measured: no** (unrelated `position_ids` IO mismatch) |
| `--tokens` must exceed `--decode-skip` | **measured** (both fail and pass cases) |
| Discovery snippet (`import nvidia …`) locates the DLLs/headers | **measured** (passed) |
| `ort-prebuilt` ORT is CPU-only; `ort-sys` loads by absolute path | **measured** (artifact + code) |
| End-to-end ORT-CUDA run | **not verified** (no CUDA-enabled ORT installed) |
