### 2026-08-17: This development box has a working GPU; probe it properly

**By:** Coordinator

**What:** The primary Windows development box (RTX 4060, 8 GB, driver 591.55, CUDA 13.1, WDDM) **can run CUDA**. Agents must not report "no CUDA runtime on this machine" — that conclusion is false and has been verified false.

There is no CUDA Toolkit install (`nvcc` absent, `CUDA_PATH` empty) and no CUDA DLLs on the default PATH, which is why shallow probes conclude the GPU is unusable. A complete CUDA 13 runtime ships under anaconda's site-packages, and because this repository loads CUDA libraries dynamically by DLL name, putting those directories on PATH is sufficient:

```powershell
$nv = "$env:LOCALAPPDATA\anaconda3\Lib\site-packages\nvidia"
$env:PATH = "$nv\cu13\bin\x86_64;$nv\cudnn\bin;$env:PATH"
```

`$nv\cu13\bin\x86_64` provides `cudart64_13.dll`, `cublas64_13.dll`, `cublasLt64_13.dll`, `nvrtc64_130_0.dll`; `nvcuda.dll` comes from the display driver. Build the CLI with `--features native-cuda`.

**Why:** The belief that this box had no CUDA was repeated across many sessions and used to justify accepting GPU performance and correctness claims as "unverified, reported by the author". It degraded review quality on every CUDA PR.

Two defects kept the belief alive and hid each other:

1. `onnx-genai-ort::cuda_rt` listed only CUDA 12 `cudart` names while the CUDA EP's own loader already listed CUDA 13. On a CUDA 13 host every candidate failed and the error named the *last* one tried — `libcudart.so`, a Linux name failing on Windows — which reads as "this machine has no CUDA" rather than "this list is stale". Fixed in #1178; the duplicate table is #1180.
2. Native CUDA decode then fails at `::Attention` with an unprepared `SessionPersistent` workspace (#1179).

Together they made native CUDA genuinely unrunnable here, and that unrunnability was reported upward as absence of hardware.

**The general lesson, which outlives these two bugs:** when a capability appears to be missing, distinguish *absent* from *misconfigured* before building conclusions on top of it. A negative result that excuses you from verification deserves more scrutiny than a positive one, not less.
