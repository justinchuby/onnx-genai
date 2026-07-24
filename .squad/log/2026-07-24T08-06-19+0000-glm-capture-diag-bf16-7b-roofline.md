# GLM capture diagnostic, bf16 kernels, and 7B roofline — 2026-07-24T08:06:19+0000

- GLM-5.2 DSA whole-step capture remains blocked by logical packed past-KV/indexer bindings; `8437b059` adds only safe decline diagnostics.
- bf16 CUDA RoPE and normalization support merged in `668a8b77` after a green review and 210/210 CUDA tests.
- Qwen2.5-7B profiling points to gate/up weight-read latency; symmetric prefetch is the approved next experiment, still in progress.
