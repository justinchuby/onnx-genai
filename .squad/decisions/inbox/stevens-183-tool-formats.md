### 2026-07-29: Support three assistant tool-call output formats
**By:** Stevens
**What:** Server assistant-output parsing recognizes Qwen/Hermes `<tool_call>` objects, Llama 3.1+ `<|python_tag|>` objects, and Mistral `[TOOL_CALLS]` arrays. All formats share one OpenAI call converter, which prefers `arguments`, falls back to Llama's `parameters`, and assigns sequential IDs.
**Why:** Non-streaming and streaming completions already share this parser, so format-specific extraction with a shared converter extends both paths without duplicating argument mapping. Llama values are scanned as complete JSON values with optional top-level semicolon separators, including semicolons inside JSON strings; scanning stops safely at malformed JSON or model terminator tokens.
