# Loader validation consolidation

- **Agents:** Hodge, Deckard, Rachael, Sebastian, Wallace, Sapper, Howie
- **Merged:** `98c6c00`–`051e0a5`
- **What/why:** Added fail-fast ONNX legality checks, post-initializer graph validation, no future-IR ceiling, valid non-empty custom-only imports, and robust model-local-function inlining. This rejects ambiguous execution while accepting legal forward-compatible models.
- **Review:** Multiple blocking function-inliner and validation findings were revised and merged.
