# Inference metadata JSON Schema

`inference_metadata.schema.json` is generated from the Rust types in
`crates/onnx-genai-metadata/src/schema.rs` with `schemars`. Do not edit the JSON
file by hand.

Regenerate it from the repository root:

```sh
cargo run -p onnx-genai-metadata --bin gen_schema
```

`cargo test -p onnx-genai-metadata` includes a sync test that fails when the
committed schema differs from the Rust source.
