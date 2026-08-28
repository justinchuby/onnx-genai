### 2026-08-28: Declare portable tool protocols as exact package facts
**By:** Zhora
**What:** Tool-capable packages declare `package.tool_protocol.identity` and `.version` (introduced in metadata schema v1.3); absence means tools are unsupported.
**Why:** Protocol selection must be deterministic and portable. Server adapters resolve only the exact declaration and fail closed for unsupported pairs, replacing parser trial order and model-family inference.
