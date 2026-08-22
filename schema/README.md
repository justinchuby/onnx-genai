# Inference metadata JSON Schema

`inference_metadata.schema.json` is generated from the Rust types in
`crates/onnx-genai-metadata/src/schema/` with `schemars`. Do not edit the JSON
file by hand.

The normative contract these types encode is
[docs/genai/INFERENCE_METADATA_DECISIONS.md](../docs/genai/INFERENCE_METADATA_DECISIONS.md).
JSON Schema is useful for authoring, but it does not express the semantic
invariants — row-scope derivability, effect/speculation bounds, state lifetimes,
cache dependencies. `validate_metadata` is authoritative:

```sh
cargo run -p onnx-genai-metadata --bin validate_metadata -- <package-or-file>
```

Regenerate it from the repository root:

```sh
cargo run -p onnx-genai-metadata --bin gen_schema
```

`cargo test -p onnx-genai-metadata` includes a sync test that fails when the
committed schema differs from the Rust source.
