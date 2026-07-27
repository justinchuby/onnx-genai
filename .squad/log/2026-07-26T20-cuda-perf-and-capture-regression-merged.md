# CUDA perf and capture regression merged

- Merged PR #201: default-domain Attention staged-KV CUDA graph capture regression coverage, preserving #193 stream-ordered copy-back behavior.
- Merged PR #203: portable symmetric int4 GEMV split-K perf path after coverage was repaired to exercise the split-K kernel.
- Fixed pre-existing main CI red from unformatted `decode_spmd.rs` via direct rustfmt commit `1bf119af`.
- Main is green; GLM/DeepSeek mobius export remains in Justin's PR queue (#404/#423/#430).
