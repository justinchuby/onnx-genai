# Continuous-batching device-logits router: does wiring a real producer pay off? — a model-size test

**Date:** 2026-08-18
**Host:** Intel i7-13800H (14C/20T), **RTX 4060 Laptop 8 GB** (driver 591.55,
CUDA 13.1). Single process; `profile_native` built
`--release --features "bench-native,cuda"`. GPU verified idle (0 MiB used)
before each run via `nvidia-smi`.
**Backend:** native CUDA decode (`onnx-genai-engine/native-backend` + `cuda`),
captured device-argmax fast path armed (`captures>0, fallbacks=0`).
**Reference baselines:** every A/B is same-binary, paired, byte-identical
(greedy token streams compared position-by-position; `divergence=0.000%`).

> **What this measures.** PR #1155 (`808dadef`) merged a per-row device-sampling
> router for continuous batching (`BatchStepLogits::Device` + `DeviceRowLogits`).
> On `main` @ `2ecd8fef` the router is still **inert**: the only
> `impl DeviceRowLogits` and the only `BatchStepLogits::Device(..)` constructor
> in the whole tree are the unit-test mock (`batched.rs:1792` / `1850`). Every
> real producer — the native seam `decode_greedy_batch_ragged_logits` and both
> ORT batched sessions — returns `HostRows`/`Ort`, so `device_available` is
> always false and every row plans `Host`. Wiring a real CUDA producer would let
> the router replace the per-step full-logits device→host copy with on-device
> sampling for greedy rows. This doc asks, with data, **whether that change is
> worth building — and whether the answer depends on model size.**

## The instrument

The cost a real `Device` producer would remove is exactly the difference the
existing on-GPU-argmax A/B already isolates: `greedy_decode_host` reads the full
`[1,vocab]` logits to the host and reduces on the CPU (a full-logits D2H + host
argmax + the sync it forces), while `greedy_decode_ongpu` reduces on the device
and copies back only the 4-byte token id. Their ratio is the batch-1 ceiling of
the device-sampling win for a fully-greedy row.

```
profile_native --ep cuda --ongpu-argmax-bench \
  --prompt "Explain the theory of relativity in detail." --tokens N --warmups 1 --runs 5
```

## Result 1 — batch-1: the win collapses as the model grows

| model | quant / vocab | host-argmax (full-logits D2H) | on-GPU argmax (no D2H) | ratio | byte-identical |
|---|---|--:|--:|--:|:--:|
| qwen05b-q4 (0.5B, 24L) | int4 / 151 936 | 292.81 tok/s | 389.42 tok/s | **1.33×** | PASS |
| qwen14b-zp (14B, 48L)  | int4 / 151 936 |   6.29 tok/s |   6.58 tok/s | **1.046×** | PASS |

The full-logits D2H the router eliminates is **~25 %** of a 0.5B decode step but
only **~4.6 %** of a 14B step. Same vocabulary (same bytes moved), but the 14B
step is ~47× longer, so the fixed D2H cost is diluted to near-noise.

The 0.5B arm proves the instrument is live (it *can* show a large delta on the
same binary), so the small 14B delta is a real result, not a broken harness.

## Result 2 — batch-N D2H accounting (`--mid-flight-via-manager`, batch 4)

The native host-logits seam reports its honest D2H cost per step. Reading the
`..._d2h:` line from the mid-flight continuous-batch manager (12 prompts,
`--mid-flight-batch 4 --tokens 48`, 35 steps):

| model | per-step logits D2H | bytes/step | ≈ step time (batch-1 device path) | D2H as % of step |
|---|--:|--:|--:|--:|
| qwen05b-q4 | 1.319 ms | 1 187 KB (4 rows × f16) | ~2.6 ms | **~25–30 %** |
| qwen14b-zp | 0.506 ms | 1 188 KB (4 rows × f16) | ~152 ms | **~0.3 %** |

Note the effective D2H rate (1 187 KB / 1.319 ms ≈ 0.9 GB/s) is far below PCIe:
`read_bytes()` cost is dominated by the host allocation + forced sync, **not**
link bandwidth. That is *good news for the router* — it removes the whole
round-trip for device-sampled rows, not just the bytes — but it does not change
the model-size verdict.

## Verdict — the leading hypothesis is KILLED for the large-model target on this box

**Measured:** on the RTX 4060 8 GB, wiring a real `BatchStepLogits::Device`
producer would recover ≤ ~4.6 % (batch-1) / ~0.3 % (batch-4) of a 14B decode
step, versus 25–33 % on a 0.5B model. For the directive's **large** models it
moves essentially no number here.

**Inferred (labelled):** the 14B int4 weights (~7–8 GB) do not fit alongside KV
and activations in 8 GB, so the 14B decode step is **weight-streaming (HtoD)
bound** at ~152 ms/step ≈ 6.5 tok/s. The bottleneck for large-model batch decode
on this box is weight offload/streaming — *not* the logits D2H, and not anything
in the multi-request-batching code. This matches the standing decode-campaign
finding that native's large-model moat is weight offload + CUDA-graph capture.

**Where the router *would* pay off (not reproducible here):** on a GPU with
enough VRAM to *fit* a large model (e.g. the H200 runs in `.squad/decisions.md`
where GLM-4-9B decodes at ~128 tok/s ≈ 7.8 ms/step), the per-step logits D2H
(~0.5 ms × batch) is a meaningful fraction again, and it *grows with batch size*
while weight cost amortises 1/N across the fused forward. There the device-router
is worth building. On an 8 GB box it is not, because the large model never fits.

**Recommendation:** do **not** build the device-logits producer as a
large-model speedup on this hardware — the measurement does not support it. Keep
the merged router as staged infrastructure (its bytes-moved test harness —
`rows_host_copied` / `rows_device_sampled` / `bytes == rows·vocab·4` — is ready
for whoever wires a producer on a fits-in-VRAM deployment). The 3 native-vs-ORT
`if decode_backend == Native` sites in `batched.rs` (739/833/849) are backend
*constructor* selection (`batching_capability`, `continuous_batch_manager`), not
decode-loop asymmetry: the loop itself is shared through the
`BatchedDecodeSession` trait, so there is no DRY seam to unify here.

## Reproduce

```powershell
$nv = "$env:LOCALAPPDATA\anaconda3\Lib\site-packages\nvidia"
$env:PATH = "$nv\cu13\bin\x86_64;$nv\cudnn\bin;$env:PATH"
cargo build --release -p onnx-genai-bench --bin profile_native --features "bench-native,cuda"

# Result 1 (batch-1 ceiling)
.\target\release\profile_native.exe --model <model> --ep cuda --ongpu-argmax-bench `
  --prompt "Explain the theory of relativity in detail." --tokens 128 --warmups 1 --runs 5

# Result 2 (batch-N D2H accounting)
$p = "The capital of France is||Once upon a time,||2 + 2 equals||The quick brown fox||In the beginning||Water boils at||The sun rises in||My favorite color is||Roses are red and||A journey of a thousand||To be or not||The meaning of life"
.\target\release\profile_native.exe --model <model> --ep cuda `
  --mid-flight-solo-equivalence-prompts $p --mid-flight-via-manager --mid-flight-batch 4 --tokens 48
```
