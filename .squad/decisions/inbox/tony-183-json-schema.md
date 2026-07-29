### 2026-07-29: Expose structured JSON schema response formats
**By:** Tony
**What:** Chat completions now accept `response_format.type: "json_schema"` with `json_schema: { name, schema, strict? }`, passing the schema object to `GenerateConstraint::JsonSchema`. JSON-object and JSON-schema formats share constrained-JSON streaming and incomplete-output handling.
**Why:** The engine already enforces JSON Schema through llguidance; this makes that capability available through the OpenAI-compatible HTTP API while preserving existing `json_object` behavior.
