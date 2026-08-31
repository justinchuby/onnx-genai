### 2026-08-26T07-08-59: Mobius owns explicit batching declarations; runtime admission remains fail-closed
**By:** Sapper
**What:** Mobius owns explicit batching declarations; runtime admission remains fail-closed
**References:** onnx-genai PR #2009, onnx-genai PR #2010, onnx-genai PR #2137, Mobius PR #636
**Why:** ### 2026-08-26: Keep component batching permission producer-authored
**By:** Sapper
**What:** Mobius workflow builders explicitly emit `batch_capacity` for audited row-independent decoder, image-diffusion, video, TTS, and adapter components, while unproven legacy encoders continue to omit it. Fixed internal row multiplication is represented as `request_expanded`, not mistaken for request batching. The onnx-genai runtime's rejection of undeclared multi-request invocations remains unchanged.
**Why:** Tensor shapes prove where rows live but cannot prove that co-batching preserves each row's result. Relaxing admission or deriving semantic permission would defeat the load-bearing fail-closed contract. Schema v1.1 is stamped only when capacity is authored; non-request symbolic policy dimensions are derived as uniform constraints from producer-owned typed dataflow.
**Implementation:** onnxruntime/mobius PR #649 (`fix/batching-capacity-signal`). No onnx-genai runtime change or PR was needed.
<!-- Archived from the durable decision inbox by Scribe on 2026-08-27; original inbox content above is unchanged. -->
