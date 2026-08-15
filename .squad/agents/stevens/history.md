# Stevens — History

## 2026-07-29T06:55:00+0000 — Llama and Mistral tool calls merged

- PR #390 (`0e62150e`) extended `parse_tool_calls` to Llama `<|python_tag|>` values and Mistral `[TOOL_CALLS]` arrays while retaining Qwen coverage.
- Consecutive Llama values use a serde byte-offset prefix scanner, safely handling semicolons in JSON strings and malformed/partial input.
- Shared conversion prefers `arguments`, falls back to `parameters`, and creates sequential OpenAI call IDs.
