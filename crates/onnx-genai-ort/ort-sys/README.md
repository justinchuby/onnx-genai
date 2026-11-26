# ONNX Runtime bindings and explicit reference runtime

Builds generate bindings from repository-owned ONNX Runtime 1.32.0 C API 32
headers. Their provenance, hashes and upstream license are in
[`third_party/onnxruntime`](third_party/onnxruntime/README.md). Binding generation
requires libclang; it does not need, locate, download, patch or link an ORT shared
library. This applies to Native and reference-feature builds alike.

`ORT_ROOT`, `ORT_LIB_DIR`, pkg-config and Metal plugin dependency inspection no
longer select build headers or provision a runtime. All builds use the same
pinned headers. No runtime library search path or rpath is embedded. The old
Cargo `target/**/ort-prebuilt` runtime cache discovery is removed.

Native execution needs no ORT binary. Reference execution retains explicit
`ONNX_GENAI_ORT_LIB` (full library path), `ONNX_GENAI_ORT_LIB_DIR`, conda/virtualenv
and platform dynamic-loader discovery, plus strict C API compatibility checks.
A Metal EP host must explicitly select the same compatible ORT library used by
its plugin. Runtime paths do not change the selected decode backend.

To provision the pinned reference runtime deliberately:

```sh
cargo run --locked -p onnx-genai-ort-sys --example provision_runtime -- /new/reference-ort
export ONNX_GENAI_ORT_LIB_DIR=/new/reference-ort/lib
cargo test --locked -p onnx-genai-ort-sys --test runtime
cargo test --locked -p onnx-genai-ort --test runtime_upgrade
```

The destination must be new and its parent must exist. The opt-in utility uses
curl, verifies pinned official PyPI wheel SHA-256, extracts only native libraries
and creates standard Unix loader-name symlinks; it never imports Python or
modifies library install names/signatures. A failed provisioning leaves its
chosen directory for inspection; retry into a new directory. The pinned CPU
wheels cover Linux x86_64/aarch64, macOS aarch64 (14+) and Windows x86_64/aarch64.
Other reference targets require an explicit compatible local runtime. This does
not restrict Native build targets. The artifact table and extraction/checksum
logic remain tested in `tests/bootstrap.rs` and used by the opt-in example.

CI reference lanes invoke this utility and set the runtime directory explicitly
for both the runtime and the existing testkit-specific `NXRT_ORT_LIB_DIR`.
Native compilation, lint, packaged Native launchers and offline lanes need no
reference runtime provisioning. This does not prove CUDA/device execution.

## Fresh Native build without an ORT installation

Run this expensive recipe under the shared workload lock. Fetch the locked Rust
dependency graph first. Use a genuinely empty target, not a warmed target whose
bindings predate this change. The deliberately nonexistent runtime paths make
accidental reference execution fail; the subprocess test must still generate.

```sh
cargo fetch --locked
scratch=$(mktemp -d)
mkdir "$scratch/deny-tools"
for tool in curl pkg-config otool install_name_tool codesign; do
  cat > "$scratch/deny-tools/$tool" <<'SH'
#!/bin/sh
echo "unexpected ORT bootstrap tool: $0" >&2
exit 97
SH
  chmod +x "$scratch/deny-tools/$tool"
done
PATH="$scratch/deny-tools:$PATH" \
CARGO_TARGET_DIR="$scratch/target" \
ORT_ROOT="$scratch/absent-install" ORT_LIB_DIR="$scratch/absent-libraries" \
ONNX_GENAI_ORT_LIB="$scratch/absent-reference-library" \
cargo test --locked --offline -p onnx-genai-server \
  --no-default-features --features metrics,native-backend,cuda-13000 \
  --test http native_http_generation_does_not_load_ort -- --exact --nocapture
# Require exactly one passed test; zero selected tests are not evidence.
# No runtime download directory may appear in this fresh build tree.
test -z "$(find "$scratch/target" -type d -name ort-prebuilt -print -quit)"
```

This recipe requires an available pinned Rust toolchain, cached Cargo sources,
libclang and the usual Native compiler tools. It proves this host/feature/fixture
configuration, not every platform. Keep the Cargo log and actual revision for
review, then delete the scratch directory once evidence has been collected.
