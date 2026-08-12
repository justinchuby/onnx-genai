# Decision: CUDA EP validated on physical H200 hardware (Muse-Glimmer-30B)

**Author:** Sebastian (Performance Engineer)
**Date:** 2026-08-12
**Branch:** `squad/muse-glimmer-cuda-validate`
**Context:** `python/nxrt-ep-cuda/README.md` claimed the CUDA EP had "not been
validated on physical CUDA hardware yet (issue #768)." We have 8× H200s, so this
was corrected by actually validating on hardware.

## Result: VALIDATED ✅

- **GPU:** NVIDIA H200 (143 GB), driver 580.105.08, CUDA 13.0 (no toolkit; cudarc
  dynamic-loads libcuda).
- **Model:** Muse-Glimmer-30B int4 (VLM), decoder-only text path (52 layers,
  hidden 6656, GQA 32/2 heads, head_size 128, vocab 202048), bf16 activations +
  KV cache + scales throughout.
- **Native EP run:** embedding + decoder run 100% on the CUDA EP with
  `ONNX_GENAI_REQUIRE_CUDA=1` and **zero CPU fallbacks**. Coherent generation:
  _"The capital of France is Paris. It is the most populous city in France and
  the most populous city in the European Union…"_
- **Throughput (eager):** median **11.08 tok/s** (min 10.98 / max 11.09),
  decode ~90 ms/token, prefill ~300 ms (2 warmups + 3×128 tokens, GPU-pinned).
  Note: `InferenceSession::run` is always eager; CUDA-graph capture is an
  engine-pipeline path and was not measurable here (downloaded ORT prebuilt is
  CPU-only, so the embedding prologue can't run under the pipeline).
- **Plugin cdylib in ORT:** `libonnx_runtime_ep_cuda_plugin.so` (built
  `--features cuda`) registers via `register_execution_provider_library`, ORT
  discovers 8× `cuda_ep` (vendor `nxrt`) H200 devices, and a session selects
  `cuda_ep` and **executes both a single-node and a multi-node fused graph on
  the H200 with correct results** (validated by `scripts/validate_plugin_ep_ort.py`).

## Ops fixed (all were runtime dtype rejections despite 100% placement)

Placement (`supports_op`) claimed every op, but several kernels rejected the
model's bf16 dtypes at *runtime*. `REQUIRE_CUDA=1` only catches placement-time
declines, so an execution harness (`muse_decode`) was required to find these.

1. **Clip (int64)** — integer Clip was rejected; generalized the claim/kernel.
2. **MatMulNBits (bf16 activations)** — the only core decoder kernel with no
   bf16 support (417 nodes/token). Added a bf16 path that stages activations/
   scales bf16→f16 (reusing the tuned f16 GEMV/GEMM), keeps int4 weights, casts
   f16→bf16 output. Added a cached per-kernel grow-only `Bf16Scratch` arena to
   avoid per-call `cuMemAlloc`/`cuMemFree` (both sync the device) — this lifted
   throughput 5.87 → 11.08 tok/s. Capture-safe.
3. **GroupQueryAttention (bf16 cos/sin cache)** — extended `gqa_load_cache` to a
   tri-state (f32/f16/bf16) and accept bf16 rotary caches.
4. **SkipSimplifiedLayerNormalization (bf16)** — final norm rejected bf16; added
   a bf16→f32 staging wrapper (1 node/token, negligible cost).

## Plugin-EP graph execution enabled (ORT plugin path)

The CUDA plugin uses a *shared EP* factory. `CreateEp` was previously disabled
("shared EP is owned by the factory") so a standalone ORT session silently fell
back to CPU. Enabled it correctly:

- `ExportedEp` now holds an `EpHandle` (`Owned` for CPU, `Shared(Arc<Mutex<…>>)`
  for CUDA), so `CreateEp` reuses the *same* shared EP that backs the factory's
  allocator/stream/data-transfer. Release only shuts down owned EPs.
- Fixed a latent device bug: multi-node fused subgraphs allocated **host**
  intermediate buffers (`vec![0u8]`), so the next kernel dereferenced a host
  pointer as device → `CUDA_ERROR_ILLEGAL_ADDRESS`. Intermediates are now
  allocated via `KernelContext_GetScratchBuffer` against the input's memory info
  (device memory on the CUDA EP, host on the CPU EP — uniform). CPU plugin tests
  (incl. `add_skip_layer_norm_mul_routed`) still pass.

## README correction (proposed, already applied on branch)

The ⚠️ block in `python/nxrt-ep-cuda/README.md` (and the matching docstring in
`__init__.py`) was changed from "not validated on physical CUDA hardware yet
(issue #768)" to a truthful H200-validation statement, while keeping the
pre-release framing.

## Follow-ups / notes

- Captured/CUDA-graph decode throughput is higher than eager but needs a
  CUDA-enabled ORT build for the embedding prologue; not measurable in this env.
- `onnxruntime-genai` (stock) cannot load `muse_glimmer` (new model type), as
  expected — native throughput reported instead.
- Tools added: `crates/onnx-genai-bench/src/bin/{cuda_place,muse_decode}.rs`
  (placement probe + end-to-end native decode harness) and
  `scripts/validate_plugin_ep_ort.py` (plugin-EP-in-ORT smoke test).
