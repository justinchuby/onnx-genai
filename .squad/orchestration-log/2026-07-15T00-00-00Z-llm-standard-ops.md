# Standard LLM operators consolidation

- **Agents:** Joshi, Deckard, Sapper
- **Merged:** `923c8bd`, `0549e1f`, `9fe94d4`
- **What/why:** Added ai.onnx Gelu, RMSNormalization, RotaryEmbedding, and Swish CPU kernels plus matching shape rules for modern transformer graphs. Follow-up fixes made broadcasting, opset registration, and bounds behavior spec-safe.
- **Review:** Deckard’s blocking revision was re-reviewed before merge.
