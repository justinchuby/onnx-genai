### 2026-08-28: Workflow metadata is the sole speculative authority
**By:** Rachael
**What:** S8 admits `onnx-genai.speculative@1` contracts only through workflow-native metadata. Native and ORT MTP resolution use the declared proposer artifact, ports, target output, immutable initializer owners, state behavior, and rollback bound; legacy `config.json` speculator content is migration-only and a legacy-only package fails before loading.
**Why:** Proposal-side sidecars and runtime configuration previously supplied overlapping semantic facts. Keeping enablement/width runtime-owned while resolving every executable MTP fact from one versioned declaration makes admission fail closed and prevents a loader from silently selecting a different proposer.
