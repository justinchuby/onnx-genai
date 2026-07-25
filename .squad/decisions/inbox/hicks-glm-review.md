# Hicks GLM native-validation review — 2026-07-25

**Verdict: 🟡 (mergeable after this review's documentation correction).**

The two ORT non-load claims were reproduced against the CUDA-enabled linked
ORT: GLM-4 fails load at
`GroupQueryAttention_node_19` with unrecognized
`rotary_embedding_dim`; GLM-5.2 QMoE fails load because
`pkg.nxrt:IndexShare(-1)` is unregistered. The q4 ORT number is explicitly
qualified and GPU-residency evidence is recorded, so it is not presented as a
CPU-fallback comparison. A 16-token native CUDA greedy GLM-4 run from
`Hello` produced `", I am a 3rd year student at the University of Waterloo. I"`.

## q4 triage

**Severity: high correctness regression; effort: small-to-medium, localized
native CUDA decode binding fix (roughly 1–3 days including regressions).**

Reproduction on CUDA with prompt `123` fails on decode token 1:

```text
model/layers.0/self_attn/indexer/Add_node_70:
[[1, 1, 2], [1, 1, 4096]] are not broadcast-compatible
```

The `[1,2,3,4]` control fails at the same node with `[1,1,5]` versus
`[1,1,4096]`; CPU completes eight tokens, and ORT CUDA completes decode.
This excludes token corruption, a generic operator/kernel gap, and an export
artifact.

Despite the `indexer` name, q4 contains **zero** `pkg.nxrt::IndexShare` nodes
and two standard `ai.onnx::Attention` nodes. `Add_node_70` combines the
logical-width indexer score (`ReduceSum_67`) with a cast/squeezed
`attention_mask`. On CUDA, `DecodeCudaState::extend_mask`
(`crates/onnx-genai-engine/src/native_decode.rs`) intentionally exposes the
single-token mask at `max_len=4096`; that capacity leaks through the indexer
mask branch while the score remains logical width. This is category **(c)**:
a decode-time mask/capacity binding bug, not the IndexShare DSA kernel.

Dispatch to **native CUDA decode/engine owner** (the agent responsible for
`onnx-genai-engine` fixed-capacity KV and `onnx-runtime-session` device
bindings), not the IndexShare-kernel owner. Fix the physical-mask exposure
policy only for proven-safe topology; preserve logical mask shape when
non-Attention mask consumers reach prefix-sensitive indexer arithmetic. Add
CUDA regressions for prompts `[123]` and `[1,2,3,4]` across at least two
generated tokens.
