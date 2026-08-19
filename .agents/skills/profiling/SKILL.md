---
name: profiling
description: How to profile the native CUDA/CPU EP with Nsight (ncu/nsys) and the built-in per-op timer. Read before profiling decode kernels.
---

# Profiling the native EP

Source the repo's CUDA env script first (it puts the ORT libs and the nvidia
wheel libs on `LD_LIBRARY_PATH` so cuBLAS/cuDNN resolve, and CUDA on `PATH`).
`ncu`/`nsys`/`nvcc` ship in the CUDA toolkit `bin` directory.

> The env script above is **Linux-oriented**. On a **Windows** dev box with no
> CUDA toolkit (CUDA libs from pip wheels under `site-packages\nvidia`), see
> [`docs/benchmarks/windows-cuda-runbook.md`](../../../docs/benchmarks/windows-cuda-runbook.md)
> for the equivalent PowerShell env block that puts cuBLASLt/cuDNN on `PATH` and
> `CUDA_PATH` at the NVRTC headers, plus a known-good smoke command.

## The workload: profile_native

Build once, then run the steady-state decode loop:

```bash
cargo build --release -p onnx-genai-bench --features bench-native,cuda --bin profile_native
profile_native --model <model-dir> --ep cuda --steady --warmups 1 --runs 3 --tokens 128
```

## Nsight Compute (ncu) — per-kernel counters

If the driver has `RmProfilingAdminOnly=1` (check
`/proc/driver/nvidia/params`), ncu needs elevated permissions. Run it with the
env forwarded so the loader still finds the CUDA libs — a bare `sudo ncu`
drops `PATH`/`LD_LIBRARY_PATH` and fails to load cuBLAS:

```bash
sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  ncu --graph-profiling node --set full -k regex:<kernel> \
  --launch-count N -o report <profile_native cmd...>
```

- **`--graph-profiling node` is mandatory** whenever the model runs a captured
  CUDA graph (most decode paths) — without it every captured kernel is hidden.
- Isolate a target kernel with `-k regex:<kernel-name>` +
  `--launch-skip`/`--launch-count` (a decode step launches each layer's kernel
  many times).
- Typical decode signal: M=1 GEMVs are memory-latency/issue-bound (low DRAM
  utilization), dominant stall = **Long Scoreboard** (global-load latency), not
  bandwidth. Raising occupancy alone rarely helps.

## Nsight Systems (nsys) — timeline / kernel mix

```bash
sudo -E env PATH="$PATH" LD_LIBRARY_PATH="$LD_LIBRARY_PATH" \
  nsys profile --cuda-graph-trace=node -o timeline <profile_native cmd...>
nsys stats --report cuda_gpu_kern_sum timeline.nsys-rep   # per-kernel % of decode
```

`--cuda-graph-trace=node` is the nsys equivalent of `--graph-profiling node`;
without it captured kernels collapse into one opaque graph node. Use the
kernel-sum report to find the dominant kernel (the % to attack).

## Per-op timing (no Nsight, CPU or CUDA)

```bash
ONNX_GENAI_PROFILE_OPS=1 profile_native --model <dir> --ep <cpu|cuda> --steady --runs 3 --tokens 128
```

`executor.rs` prints per-op-type total_ms/percent/calls per forward pass to
stderr — a fast way to find which op-type dominates before reaching for Nsight.

## Notes

- If other jobs share the host, timing has variance — report medians and
  caveat it; pin a free GPU with `CUDA_VISIBLE_DEVICES`.
- Verify byte/near-identity after any kernel change; split-K reorders fp32
  partials (near-equal, not bit-exact) — validate with tolerance tests.

## Making a measurement trustworthy on a shared box

This machine is shared with build agents, and the failure mode is not "numbers
are a bit off" — it is publishing a conclusion that reverses when the box is
quiet. Three habits, in increasing order of strength:

**1. Measure CPU time, not wall clock.** `TotalProcessorTime` on the process is
nearly contention-immune for a fixed thread count. Measured on this box: three
identical runs of one configuration gave 39.3 / 25.8 / 16.1 s wall, while CPU
time reproduced to ~2%. Peak RSS is per-process and reliable regardless.

```powershell
$p = Start-Process -FilePath $exe -ArgumentList @(...) -NoNewWindow -PassThru `
     -RedirectStandardOutput out.txt -RedirectStandardError err.txt
$peak = 0
while (-not $p.HasExited) {
  $ws = (Get-Process -Id $p.Id -ErrorAction SilentlyContinue).WorkingSet64
  if ($ws -gt $peak) { $peak = $ws }
  Start-Sleep -Milliseconds 200
}
$p.WaitForExit()
"cpu=$($p.TotalProcessorTime.TotalSeconds)s peak=$([math]::Round($peak/1MB))MB"
```

Note `PeakWorkingSet64` reads 0 after exit, so poll while it runs, by PID (not
by process name — that catches the wrong instance when several are alive).

**2. Compare an effect against the spread of its own arm.** If the A/B
difference is smaller than the gap between two runs of the *same* configuration,
it is unmeasured — say so rather than reporting it. A 14B model showed ~20% wall
spread within one arm; anything under ~1.3x there is noise.

**3. Best: include a control arm the change provably cannot touch.** A harness
that also measures a shape or path the toggle does not reach turns "the machine
was busy" from an excuse into a measurement. If the control moves, the run is
unusable and you know it *before* drawing a conclusion.

This caught a real case: an M=1-only GEMV toggle was being evaluated, and the
M=128 rows in the same harness — which that route cannot affect — moved 1.05x →
0.70x and 0.72x → 0.92x between runs. That set a noise floor of ~1.5x and made
an apparent single-shape regression unadjudicable. Without the control it would
have been argued about; with it, the answer was simply "re-run when quiet".

Prefer a control arm to extra repetitions: it costs one more shape in a harness
you already have, and it is stronger evidence than a distribution.

## Separating fixed cost from per-unit cost

Differencing removes an intercept you cannot otherwise isolate:

- **Decode per token:** run N and M tokens, take `(T(M) - T(N)) / (M - N)`. This
  cancels model load and prefill entirely.
- **Prefill per token:** run the *same* token count with two prompt lengths and
  take the slope. This cancels model load, which otherwise dominates a short run.

A single long run cannot separate these, and a single short run is almost all
fixed cost. Both mistakes have been made here.
