### 2026-07-14: Reserve the `onnx-runtime-*` crate names with a prerelease
**By:** Deckard
**What:** Set the eight nxrt crates to `0.1.0-dev.0`, exact-pin their workspace dependencies to `=0.1.0-dev.0`, and publish in this order: ir, shape-inference, loader, optimizer, ep-api, ep-cpu, session, capi. Actual upload remains blocked until a crates.io token is provided.
**Why:** A prerelease reserves the intended crate names without changing the workspace-wide version or the existing `onnx-genai-*` crates, while exact pins make prerelease dependency resolution unambiguous.
