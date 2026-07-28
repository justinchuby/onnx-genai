### 2026-07-28: Land the model-package MVP as a reusable leaf crate
**By:** Chance
**What:** Add `onnx-model-package` for ORT 1.x directory manifests, selection, resolution, and validation; adapt `ModelDirectory` so existing engine entry points auto-detect packages while retaining flat-directory behavior.
**Why:** This keeps package policy independent of ORT/session code and passes resolved model, metadata, configuration, and tokenizer paths into existing loaders. Advanced authoring/inspection CLI, archives, registries, hashes, multi-component pipelines, and compiled-EP compatibility ranking remain deferred.
