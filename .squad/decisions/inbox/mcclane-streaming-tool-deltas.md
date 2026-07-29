### 2026-07-29: stream parsed tool calls as OpenAI deltas
**By:** McClane
**What:** Emit one metadata delta followed by an arguments delta for every parsed tool call, then finish with `tool_calls`.
**Why:** Clients can assemble tool invocations incrementally without receiving a monolithic completed tool-call object, while retaining full-output parsing for Qwen, Llama, and Mistral safety.
