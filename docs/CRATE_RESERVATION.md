# Reserving the `onnx-runtime-*` Crates

All ten crates use version `0.1.0-dev.0`. Their internal dependencies are
exact-pinned to that version, so they must be published in dependency order:

1. `onnx-runtime-ir`
2. `onnx-runtime-cpuinfo`
3. `onnx-runtime-tracer`
4. `onnx-runtime-shape-inference`
5. `onnx-runtime-loader`
6. `onnx-runtime-optimizer`
7. `onnx-runtime-ep-api`
8. `onnx-runtime-ep-cpu`
9. `onnx-runtime-session`
10. `onnx-runtime-capi`

`onnx-runtime-cpuinfo` and `onnx-runtime-tracer` are **foundational** crates
like `onnx-runtime-ir`: they have **no internal (`onnx-runtime-*`)
dependencies**, so they can be published early, before anything depends on
them.

Authenticate with either `cargo login <token>` or by setting
`CARGO_REGISTRY_TOKEN`. Then run:

```sh
cargo publish -p onnx-runtime-ir
cargo publish -p onnx-runtime-cpuinfo
cargo publish -p onnx-runtime-tracer
cargo publish -p onnx-runtime-shape-inference
cargo publish -p onnx-runtime-loader
cargo publish -p onnx-runtime-optimizer
cargo publish -p onnx-runtime-ep-api
cargo publish -p onnx-runtime-ep-cpu
cargo publish -p onnx-runtime-session
cargo publish -p onnx-runtime-capi
```

Wait for each version to appear in the crates.io index before publishing the
next crate that depends on it; propagation can take a short time.

`onnx-runtime-shape-inference` has a test-only dependency on
`onnx-runtime-loader` that is intentionally path-only, with no version. Cargo
omits that dev-dependency from the published manifest, breaking the otherwise
cyclic publish requirement while preserving local tests. This is why shape
inference can be published before the loader in the order above.

For a real release, replace each crate's explicit `0.1.0-dev.0` version with
the chosen stable version, update all ten workspace dependency pins to the
same exact version, rebuild, and publish again in this order. Published
prerelease versions are immutable and remain on crates.io.
