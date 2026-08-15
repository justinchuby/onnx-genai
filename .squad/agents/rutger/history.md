# Rutger — History

## 2026-07-15T17:52:18Z — Wave 2 `onnx-rs` op-schema system
- Merged `8adee51`: embedded-YAML owned schemas, opset-aware registry, built-ins, and `schema.node_conforms`.
- Holden blocked the initial version on required If else-branch and variadic minimum arity; Deckard corrected both under lockout. `onnx-rs` passes 29 tests.

## 2026-07-15T18:24:41Z — Review of Deckard ONNX JSON commit f072554
- Verdict: 🟢. No significant correctness issues found in the scoped `crates/onnx-rs` changes.
- Verified protobuf JSON conventions: emitted int64/uint64 values are strings and parser accepts strings or numbers; enums write symbolic names (numeric fallback) and parse symbolic/numeric forms; bytes/raw tensor data use base64; lowerCamelCase output with snake_case input aliases; Graph/Graphs recurse through serializer/parser.
- Round-trip coverage includes simple model metadata, raw-data initializers and typed attributes, If then/else subgraphs, typed tensor JSON input, and malformed JSON returning `Err`.
- `cargo test -p onnx-rs 2>&1 | tail -15` output:
```
test schema::tests::yaml_schema_round_trips_every_public_field ... ok
test text::printer::tests::renders_inline_attribute ... ok
test text::printer::tests::prints_single_and_list_subgraph_bodies ... ok
test check::rules::tests::common_builtin_arity_boundaries_pass ... ok

test result: ok. 39 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s

   Doc-tests onnx_rs

running 1 test
test crates/onnx-rs/src/lib.rs - (line 39) - compile ... ok

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.00s

all doctests ran in 0.16s; merged doctests compilation took 0.16s
```

## 2026-07-15T20:03:48Z — Wave 7 checker expansion
- Merged `475cea1`: ONNX-RS §8 checker expanded to nine rules, including I/O declaration, unconnected-node, type-constraint, initializer-declaration, and lower-bound IR-version validation.
- Brion corrected unknown/placeholder type handling, dangling output detection, and static initializer-dimension mismatch coverage. 67 unit tests plus 1 doctest pass; Deckard reviewed.
- `cargo fmt -p onnx-rs -- --check 2>&1 | head -15` exited 0 with no output.

### 2026-07-22T14:59:36+0000 — WP-B landed
Clippy cleanup landed at `6f217a4`, clearing `-D warnings` for `onnx-genai`, `onnx-runtime-capi`, and `onnx-runtime-python`.
