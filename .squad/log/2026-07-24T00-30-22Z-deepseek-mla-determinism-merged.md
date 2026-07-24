# DeepSeek MLA determinism merged — 2026-07-24T00-30-22Z

DeepSeek-V2-Lite MLA decode is now deterministic and coherent on `origin/main`.

- `24531c4`: stream-ordered async `copy_reshape` for Reshape/Squeeze; reviewed by Gaff.
- `621936f`: default-domain Attention aliased KV-growth race fixed via disjoint scratch staging + copy-back; reviewed by Holden.
- Coverage reported across DeepSeek-V2-Lite MoE, DeepSeek-Coder-1.3B, GLM-4-9b, and R1-Distill-Qwen-1.5B exact native/ORT parity.
- Remaining work: MLA graph-capture enablement on `perf/deepseek-mla-capture-enable`.
