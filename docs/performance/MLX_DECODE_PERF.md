# MLX EP decode performance — levers for onnx-genai

> Design note for the runtime team. Captures the onnx-genai-side work needed to
> close the decode-throughput gap for the MLX execution provider. The EP-side
> lever (`mlx_compile` compiled-decode) is being implemented in the
> `onnxruntime-mlx` repo separately; this doc covers what belongs in **this**
> repo (KV path, boundary, pipelining, sampling, speculative).

## 1. Measured starting point (2026-07-14, M1 Max, qwen2.5-0.5b q4)

| Runtime | decode tok/s | µs/token |
|---|---|---|
| CPU EP (ORT, int8 MatMulNBits SDOT) | **185** | 5395 |
| MLX EP (current) | 125 | 8022 |
| native `mlx-lm` (reference) | 300–800 | — |
| weight-bandwidth floor (387 MB ÷ ~400 GB/s) | ~1000 | ~970 |

Per-token split of the MLX EP's 8.0 ms (from the EP's Perfetto tracer,
`mlx.subgraph` = whole Compute, `mlx.eval` = GPU):

- **GPU eval ≈ 76 %** (~6.1 ms) — ~6× the bandwidth floor.
- **Host boundary ≈ 24 %** (~1.9 ms) — per-token graph rebuild + KV/logits
  crossing the ORT boundary.

**Takeaways:** on a 0.5 B model, decode is weight-bandwidth-bound and the CPU
int8 path is already excellent — MLX's real wins are **prefill** (already
1.76–2.77×) and **larger models** (7 B+, GPU-compute-bound). Still, both the GPU
eval and the host boundary have large headroom.

## 2. EP-side lever (tracked in onnxruntime-mlx, not here)

**`mlx_compile` compiled-decode.** The EP currently rebuilds + dispatches all
~393 graph kernels every token. Compiling the S==1 decode graph once (shapeless)
and re-applying it removes the per-token rebuild and fuses element-wise
dispatches. The old C++ EP saw +17–30 %; combined with dispatch fusion the win
should be larger. *Status: in progress in `onnxruntime-mlx`.*

Once compiled-decode lands, most of the **24 % host graph-build** disappears,
which also reduces the value of the async pipelining below (§3.3).

## 3. onnx-genai-side levers (this repo — for the runtime team)

### 3.1 Confirm the O(1) shared-buffer KV path is active with the MLX EP
`mlx-lm` pre-allocates the KV cache in `step = 256`-token chunks and writes each
new token **in place** (`keys[..., prev:offset, :] = k`), returning a slice —
no per-token realloc or growing `concatenate`. Our `DecodeKvMode::SharedBuffer`
(O(1)/token) is the equivalent. **Action:** verify a decode run through the MLX
EP takes the `SharedBuffer` path (not the growing `ZeroCopyRebind` path); the
per-token EP time should be flat across context (the EP tracer's `mlx.eval`
span already shows this for q4). Regression-test that `past_present_share_buffer`
is honored for the MLX EP the same way it is for CPU/CUDA.

### 3.2 Reduce the per-token boundary tax
Each token, logits and KV cross the ORT boundary host-side. Options to shrink it:
- **Keep KV device-resident** across tokens so the EP does not re-import/copy the
  past-KV every step (unified memory makes the copy a memcpy, but it is still on
  the critical path). Coordinate with the EP's `SharedBuffer` binding so the
  present-KV is written in place and re-used as next step's past-KV without a
  round-trip.
- **Avoid the full logits copy** when only `argmax`/top-k is needed — a
  device-side sampler (below) removes the [1, vocab] host copy per token.

### 3.3 Async pipelining — is ORT `RunAsync` relevant?
**Partially, and largely subsumed by `mlx_compile`.** `mlx-lm` uses
`mx.async_eval` to overlap host-side work (next-token graph build + sampling)
with the current token's GPU compute, exploiting MLX's lazy/async execution.

- ORT **`RunAsync`** runs the whole `Run()` on an ORT thread-pool thread and
  invokes a callback with the outputs. It frees the *caller* thread but the
  `Run` itself (including the EP's synchronous `mlx_eval`) still blocks the
  worker thread, and the **autoregressive dependency** (token N+1 needs token
  N's sampled id) prevents overlapping consecutive-token *compute*.
- What it *can* overlap: host-side **sampling + detokenization + scheduling** of
  token N with the EP compute — but that orchestration is already ~1 % of
  per-token time, so the win is small.
- The larger overlap (host graph-build ∥ GPU compute) is exactly what
  `mlx_compile` eliminates by building once — so after compiled-decode lands,
  `RunAsync` has little left to hide.

**Recommendation:** do **not** prioritize `RunAsync` for single-stream decode.
It is worth revisiting only for **batched / multi-request serving**, where
`RunAsync` lets independent sequences progress concurrently on the thread pool.

### 3.4 Quantized KV cache with a start threshold (long context)
`mlx-lm` keeps KV in fp16 early and switches to an **int8 KV cache**
(`group_size = 64`) after `quantized_kv_start ≈ 5000` tokens, halving KV
bandwidth once the cache is large. `onnx-genai-kv` already has quantized KV
pages (per-(component, head, token) scales). **Action:** wire a
`quantized_kv_start` threshold so decode uses fp16 KV early (accuracy) and
switches to int8 KV for long context (bandwidth), and confirm the MLX EP
consumes the quantized KV correctly.

### 3.5 Device-side sampling
`mlx-lm` samples on-device (`mx.argmax` / top-k over logits that never leave the
GPU) and only `.tolist()` at the yield boundary. For us, a device-side sampler
in the MLX EP (or a fused sampling op) would remove the per-token [1, vocab]
logits host copy. Requires an EP↔runtime contract for "return the sampled id,
not the full logits" (opt-in; the runtime still needs logits when
`logits_processors`/penalties are active).

### 3.6 Speculative decoding
Already have the infra (Gemma4 assistant proposer, MTP/EAGLE §27/§28). `mlx-lm`
drafts N tokens with a small model and verifies in one `_step` (N+1 positions),
accepting the longest matching prefix. **Action:** confirm speculative decode
runs end-to-end **through the MLX EP** (the draft + target both on MLX), which is
the biggest lever for larger models where the target forward dominates.

### 3.7 Chunked prefill + buffer-cache clearing
`mlx-lm` processes the prompt in `prefill_step_size` chunks, `mx.eval`s the cache
between chunks, and calls `mx.clear_cache()` to bound peak memory. Confirm our
prefill path chunks large prompts and releases MLX's buffer cache between chunks
so long-prompt prefill does not OOM. (MLX already wins prefill 1.76–2.77×; this
is about memory safety on long prompts, not throughput.)

## 4. Priority for the runtime team

1. **§3.1** verify SharedBuffer O(1) KV is active with the MLX EP (cheap, high value).
2. **§3.6** speculative decode through the MLX EP (biggest lever for larger models).
3. **§3.2 / §3.5** shrink the per-token boundary (device-resident KV + device-side sampling).
4. **§3.4** quantized KV start-threshold (long context).
5. **§3.7** chunked prefill memory safety.
6. **§3.3** `RunAsync` — deprioritized for single-stream; revisit for batched serving.

The EP-side `mlx_compile` (onnxruntime-mlx) lands independently and removes most
of the host graph-build overhead; these onnx-genai items address the KV/boundary/
sampling/speculative side that the EP cannot own.
