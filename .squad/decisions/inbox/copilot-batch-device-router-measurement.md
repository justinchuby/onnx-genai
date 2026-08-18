### 2026-08-18: Batch device-logits router is not a large-model speedup on the 8 GB box — killed with data
**By:** Copilot (multi-request batching slice)
**What:** Measured whether wiring a real producer for the merged-but-inert #1155
per-row device-sampling router (`BatchStepLogits::Device` / `DeviceRowLogits`;
sole impl + sole constructor are still the test mock on `main` @ `2ecd8fef`)
is worth building. On RTX 4060 8 GB, native CUDA, byte-identical greedy:
- batch-1 on-GPU-argmax A/B (the full-logits D2H the router removes):
  qwen05b-q4 292.81→389.42 tok/s = **1.33×**; qwen14b-zp 6.29→6.58 = **1.046×**.
- batch-4 mid-flight D2H: 0.5B 1.319 ms/step (~25–30 % of step); 14B 0.506 ms/step (~0.3 %).
The D2H is 25–33 % of a small-model step but ~0.3–4.6 % of a large-model step.
Full doc: `docs/benchmarks/2026-08-18-batch-device-router-vs-model-size.md`.
**Why:** The directive targets **large** models. Inferred (labelled): 14B int4
(~7–8 GB) does not fit 8 GB, so its decode is weight-streaming (HtoD) bound at
~152 ms/step; the logits D2H is noise there. The batching bottleneck for large
models on this box is weight offload, not the batching code. The device-router
would only pay off where a large model *fits in VRAM* and batch-N is used
(e.g. H200), which is not reproducible here. Recommendation: keep the router as
staged infra (its `rows_host_copied`/`rows_device_sampled`/`bytes==rows·vocab·4`
harness is ready), do **not** build the producer as a large-model win on 8 GB.
The 3 `if decode_backend == Native` sites in `batched.rs` (739/833/849) are
backend constructor selection, not decode-loop asymmetry (the loop is shared via
the `BatchedDecodeSession` trait) — no DRY seam to unify.
