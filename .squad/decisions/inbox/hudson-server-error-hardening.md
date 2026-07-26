### 2026-07-26
**By:** Hudson

**What:** Hardened `onnx-genai-server` public error paths: debug config/KV endpoints now return the existing `ApiError` when no model is loaded; multimodal generation now returns 400 errors if a vision or audio contract disappears after admission; and `ModelRegistry` centralizes `RwLock` access through fallible read/write helpers.

**Why:** An unloaded default model and a poisoned registry lock are operational failures, not process-abort invariants. Registry poisoning deliberately fails only the affected request with a 500 `ApiError` rather than recovering with `into_inner()`: recovery could expose state interrupted during a write, while an explicit error preserves the lock's safety signal and keeps the server process available for unrelated work.
