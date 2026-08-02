### 2026-08-02: Qwen2.5-1.5B benchmark-prompt divergence adjudicated native-correct
**By:** Cohaagen
**What:** Adjudicated the Foundry sweep divergence for `Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4` with prompt `Hello` (token `[9707]`). Release `profile_native` reproduced the default greedy split deterministically: native and ORT agree for generated indices `0..25`, then first diverge at 0-based generated index `26`; native emits token `1909` (`" top"`) and ORT-CUDA emits token `821` (`" data"`).

**Evidence:** Built an accuracy-level-1 oracle directory with `python3 scripts/qwen_q4_f32_oracle.py --case qwen2.5-1.5b --rewrite-acc1-dir target/qwen15b-bench-acc1` (141 `MatMulNBits` nodes rewritten to fp32 activation accumulation). At index 26, with `top_logprobs` enabled for logit/logprob inspection:

| path | selected | logprob(1909) | logprob(821) | margin 1909-821 |
|---|---:|---:|---:|---:|
| ORT CPU acc-level-1 fp32 oracle | 1909 | -2.7209473 | -2.7365723 | +0.015625 |
| ORT CUDA acc-4 logits/host argmax | 1909 | -2.7209473 | -2.7365723 | +0.015625 |
| native CUDA logits | 1909 | -2.70755 | -2.754425 | +0.046875 |

The default release ORT-CUDA greedy path (without `top_logprobs`) is the path that emitted `821`; requesting logprobs bypasses that device greedy fast path and exposes logits/host argmax that agree with the fp32 oracle. Native default greedy and native logits both stay on token `1909`.

**Verdict:** Case (a): native matches the fp32 oracle; the benchmark divergence is an ORT-CUDA acc-4/default-greedy near-tie on the lower-precision side, not a native bug. No product-code fix is needed. Added a real-model ignored regression lock in `crates/onnx-genai-engine/tests/qwen2_5_1_5b_divergence.rs` for the benchmark prompt, alongside the existing France-prompt lock.

**Validation:**
- `cargo fmt --all`
- `CUDA_VISIBLE_DEVICES=1 ONNX_GENAI_QWEN15B_CUDA_DIR=$HOME/.foundry/cache/models/Microsoft/qwen2.5-1.5b-instruct-cuda-gpu-4/v4 ONNX_GENAI_QWEN15B_ACC1_DIR=/home/justinchu/wt-cohaagen-1p5b/target/qwen15b-bench-acc1 cargo test -p onnx-genai-engine --features native-backend,cuda --test qwen2_5_1_5b_divergence -- --ignored --nocapture` (2 passed)
