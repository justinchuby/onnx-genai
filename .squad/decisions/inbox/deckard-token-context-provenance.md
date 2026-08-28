### 2026-08-28: Token context requires structural token identity
**By:** Deckard
**What:** `onnx-genai.token-context@1` is admitted at schema v1.4 only when its token input is structurally derived from `prompt_tokens` or a component output declared `token_ids`. The exact built-in capability inventory remains a non-normative runtime catalogue.
**Why:** Integer dtype and geometry cannot distinguish tokens from position IDs. A version and capability gate makes older readers fail closed while keeping model-specific computation inside ordinary ONNX graphs.
