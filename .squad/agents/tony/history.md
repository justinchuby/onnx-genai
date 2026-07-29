# Tony — History

## 2026-07-29T05:55:00+0000 — JSON Schema response format merged

- PR #388 (`804ba860`) exposes OpenAI Structured Outputs `json_schema` through chat completions and maps it to `GenerateConstraint::JsonSchema`.
- Replaced json-object-specific handling with constrained-JSON streaming/retry behavior shared by both JSON formats.
- Added type-driven HTTP 400 validation and tests for malformed schema requests.
