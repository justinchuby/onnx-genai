### 2026-08-28: Tool protocols own forced output behavior
**By:** Rachael
**What:** Resolved protocol adapters now own forced-tool constraints and prompt rendering; ATEM explicitly disables the JSON grammar. Buffered and SSE output use the same bounded parser state and fail closed for incomplete or malformed declared envelopes.
**Why:** A tagged-JSON grammar on declared ATEM output made valid XML unreachable and malformed protocol output could otherwise be returned as assistant text.
