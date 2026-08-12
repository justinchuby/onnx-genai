### 2026-08-12: Native CUDA decode is dispatch-bound, not GEMV-bound; fix VRAM-capacity portability bug + fold constant norm-weight casts

**By:** Sebastian

**What:**
Investigated why the native CUDA EP decodes Muse-Glimmer-30B (dense int4, 52
layers, hidden 6656, GQA kv_heads=2, vocab 202048, ~15.3 GB weights) at ~10
tok/s while ORT-as-backend hits 40+ tok/s. Two shipped changes plus a firm
diagnosis:

1. **Portability fix — real CUDA VRAM capacity detection (governor).**
   `resolve_vram_limit_bytes` resolved `EngineConfig::default()`'s
   `ResourceLimit::Fraction(0.90)` against `fallback_capacity_providers`, which
   reports a *provisional* 8 GiB (`PROVISIONAL_VRAM_CAPACITY_BYTES`). On ANY
   machine — including a 143 GiB H200 — that caps device leases at ~7.2 GiB, so
   a 15.3 GB model fails to load resident ("growing physical handle pool lease
   failed"). Fixed properly: when the native decode targets a CUDA device we
   now query the driver (`cudaMemGetInfo` via
   `onnx_genai_ort::cuda_rt::device_memory_info`) for the true total, so a
   fraction resolves to ~0.9×143 GiB. The 8 GiB constant survives only as a
   last-resort fallback (CPU-only builds / query failure). Threaded a
   `cuda_device_index: Option<u32>` through the governor constructors and
   `resolve_vram_limit_bytes`; native paths pass the real ordinal (via new
   `NativeDecodeDevice::cuda_index()`), ORT paths pass `None`. Also honor
   `ONNX_GENAI_VRAM_LIMIT` in `profile_native` as a convenience. Verified by a
   CUDA-gated unit test that a 0.90 fraction on device 0 resolves to ~0.9× the
   real device total (>> the 8 GiB provisional cap).

2. **Perf — `CudaFoldConstantCast` EP pass.** bf16 decoders that compute their
   RMS/layer norms in fp32 export a `Cast(bf16→f32)` of every *constant* norm
   weight (gamma). Muse-Glimmer has 208 such constant-weight casts per token (4
   norms × 52 layers) that recompute an identical fp32 constant every step —
   pure launch overhead. The new CUDA-scoped pass materializes each producer-
   less, whole-byte float constant `Cast` into a pre-converted initializer at
   session build, byte-identical to the runtime kernel (widening exact;
   narrowing uses `half` RNE = `__float2bfloat16_rn`). Removes 208 launches per
   token. Measured (muse_decode harness, H200, 128 tok, 5 runs, median):
   **10.21 → 11.41 tok/s (+11.8%, −10.3 ms/token)**, generated text byte-
   identical. Toggle: `ONNX_GENAI_CUDA_DISABLE_CONST_CAST_FOLD=1`.

**Why (the diagnosis that redirects the task):**
Measurement (not assumption) drives this. Per-op enqueue profiling
(`ONNX_GENAI_PROFILE_OPS=1`) shows the decode step is dominated by **Cast
(~42%)** and **GroupQueryAttention (~36%)**, with **MatMulNBits only ~11%** —
i.e. the int4 GEMV is NOT the bottleneck. GPU utilization sampling during steady
decode reads **~0–1%** (127 W of ~700 W): M=1 decode is entirely **CPU
dispatch / kernel-launch-overhead bound** (~1600 tiny launches/token), not
compute bound. Raising occupancy or optimizing the GEMV cannot move a workload
that leaves the GPU 99% idle.

ORT reaches 40 tok/s because its genai_config enables `enable_cuda_graph=1` +
`past_present_share_buffer=true`: a static KV buffer makes the decode step a
**captured CUDA graph**, so replay amortizes all per-launch CPU cost. That —
CUDA-graph capture over a static/paged KV buffer — is the only lever that closes
the gap to 40+ tok/s. Eager op-count reduction (like the cast fold) is real but
bounded: even removing all ~624 norm casts would only reach ~15–17 tok/s while
still leaving the GPU idle.

**Ceiling / what remains (next levers, in priority order):**
- Route this model through the engine's **capturable static-KV decode path**.
  Blocked today: the model's genai_config resolves to Multimodal (vision +
  embedding + decoder), so the single-decoder `io` block isn't built; the
  decoder is embeds-driven (`inputs_embeds`, no `input_ids`), which trips
  `profile_native --model`; and `--pipeline` runs the embedding on ORT where
  bf16 `Where(16)` is unsupported. `muse_decode` runs eager with growing KV, so
  it can't reuse a captured graph (only ~11 tok/s with capture on). A static-KV
  `muse_decode` variant (or unblocking the engine multimodal→single-decoder KV
  path) is the highest-value next step.
- **Fold the ~624 activation-wrapping norm casts** (Cast(bf16→f32) → RMSNorm(f32)
  → Cast(f32→bf16)). Attempted by extending `CudaDropNormalizationCasts` to
  bf16 `RMSNormalization`, but reverted: the ONNX-spec `rms_norm` shape-inference
  rule sets output dtype = *scale* type (V = f32), and it re-runs after EP passes
  (`run_ep_scoped_passes`), clobbering the bf16 retype; the CUDA kernel then
  rejects bf16-X/f32-Y. Doing this correctly needs either a bf16-in/f32-out
  rmsnorm kernel variant (strip only the input casts, keep Y=f32 inference-
  consistent) or narrowing the computed f32 scale to bf16. Left as scoped
  follow-up — it only matters once decode is capture-based or if we stay eager.

**Profiling caveats (this box):** nsys is blocked ("Creating threads in this
process is forbidden by design"); ncu not installed. The per-op profiler times
enqueue latency without per-node sync, so its percentages reflect CPU launch
cost — which is exactly the right signal for a dispatch-bound workload. Pin the
device with `CUDA_VISIBLE_DEVICES=0`.
