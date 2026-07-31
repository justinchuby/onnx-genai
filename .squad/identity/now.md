# Team Focus — now

**Current focus:** Native multi-component pipeline CUDA decode is the active frontier. As of
#533 (Mary, Lori APPROVED), native CUDA decode now BEATS ORT — 1.38x ORT-CUDA on real qwen3-0.6b
via the default-off `ONNX_GENAI_NATIVE_DECODER_CAPTURE_STEP_INPUTS` captured step-input binding
(mask frozen, KV device-resident), byte-identical tokens. CUDA op-coverage of the Qwen3.5 hybrid
(Mamba + linear-attention) family is complete; #529 (Cohaagen) placed qwen3.5-0.8b 100% on CUDA
(1289 nodes, 0 declines). Shape-inference container types are COMPLETE and issue #449 is CLOSED
(#531, Harry).

**IN FLIGHT:**
- **mary-2:** real-model capture-engagement validation — does the real qwen3-0.6b pipeline ENGAGE
  the captured fast path & beat ORT, or DECLINE via Concat-KV — plus a default-on recommendation.
- **cohaagen-4:** loader-unblock — admit the text-only split hybrid package for decode and flip
  the `qwen35_0_8b_hybrid_native_cuda_e2e` parity harness from graceful-skip to active.
- **harry-5:** generalize ORT `clone_value`/`clone_owned` to all POD dtypes (unblocks Bool /
  gemma-3n audio mask).

**HELD:** #534 (Harry, server contracts #481/#482, Melina APPROVED) targets Justin's active
branch `feat/genai-demo-dashboard` (PR #476); that code is not on main.

**OFF-LIMITS:** #54 model-package and #299 LoRA belong to another team; #106 is under Justin's
study. Do not touch resting other-squad open PRs (#314, #315, #317, #318, #291, #99).

**Updated:** 2026-07-31T00:25:00Z
