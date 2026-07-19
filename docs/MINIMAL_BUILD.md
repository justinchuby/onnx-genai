# Model-Driven Minimal Builds

> **Status: strategy only; not yet implemented.** The commands, features, and
> generated files below define the intended interface. Default builds must
> continue to include the full operator set until each phase lands.

## 1. Goal and scope

onnx-genai should support building a CPU runtime containing only the operators
required by one model or a known group of models. The purpose is a smaller
deployable binary, especially for CPU and edge deployments where binary size is
a product metric. CUDA should use the same selection model later.

This is compile-time selection, not a change to ONNX semantics or execution
provider placement. Binary size becomes a tracked metric alongside the runtime
metrics in [DESIGN §33.6](DESIGN.md#336-tracking-metrics), but optimization is
explicitly deferred: this document records the design and implementation
sequence.

## 2. Current state

- [DESIGN §25.4](DESIGN.md#254-feature-flags-compile-time-selection) describes
  coarse Cargo features for KV-cache implementations, constraint engines,
  samplers, and model formats. It does **not** define per-operator selection.
- The native CPU EP declares its kernel modules and builds one
  [`OpRegistry`](../crates/onnx-runtime-ep-cpu/src/kernels/mod.rs) in
  `build_cpu_registry()`. It currently covers approximately 113 operators (112
  unique names and 114 domain/name keys). The kernel modules and registration
  statements are compiled unconditionally today.
- The native CUDA EP follows the same registry-oriented architecture, although
  its current coverage differs. Therefore native-EP operator stripping is under
  this project's control; it no longer inherently requires an upstream ORT
  change.
- `onnx-rs` already provides most analyzer building blocks:
  [`SchemaRegistry`](../crates/onnx-rs/src/schema/mod.rs) resolves
  `(domain, op_type, opset)` schemas; the
  [checker](../crates/onnx-rs/src/check/mod.rs) walks models and its schema rule
  recursively walks graph nodes and subgraphs
  ([`check_graph_schemas`](../crates/onnx-rs/src/check/rules.rs)); and
  [`shape::infer_shapes`](../crates/onnx-rs/src/shape.rs) validates/informs the
  graph pipeline. A required-op collector is therefore a small addition, not a
  new model parser.

There is no supported minimal-build workflow or per-op gating today.

## 3. Design requirements

1. **Safe default:** a normal build includes all implemented operators.
2. **Opt-in minimality:** stripping occurs only when a user supplies an explicit
   model set or operator profile.
3. **ONNX-aware identity:** selection keys include normalized domain, operator
   type, and required opset compatibility—not only the operator name.
4. **Recursive analysis:** include nodes in nested graphs and model-local
   functions, plus any kernels introduced by the optimizer configuration used
   by the final build.
5. **Deterministic output:** the analyzer emits a sorted, reviewable manifest
   containing model hashes, opset imports, required operator keys, selected
   groups, tool version, and optimizer profile.
6. **Actionable failure:** loading a model that needs a stripped operator fails
   before execution and explains exactly how to rebuild.

## 4. Candidate mechanisms

### 4.1 Cargo features per operator or operator group

The direct mechanism is to gate module declarations and registry registrations:

```rust,ignore
#[cfg(any(feature = "full", feature = "ops-transformer"))]
pub mod attention;

#[cfg(any(feature = "full", feature = "ops-transformer"))]
reg.register(/* ai.onnx::Attention factories */);
```

Possible public features include `op-conv` and `op-attention`, but exposing and
testing roughly 113 independent Cargo features would be difficult to maintain.
Prefer stable capability groups such as:

- `ops-core`: constants, identity, shape, casts, basic movement;
- `ops-elementwise`: unary, binary, logical, and selection;
- `ops-reduction`: reductions and softmax;
- `ops-transformer`: attention, normalization, RoPE, GEMM, and relevant fusions;
- `ops-cnn`: convolution, pooling, and spatial normalization;
- `ops-quantized`: quantize/dequantize and quantized compute;
- `ops-sequence`: sequence/control-flow families as coverage grows;
- `full`: all groups, enabled by default.

**Advantages:** idiomatic Cargo, understandable offline builds, simple Phase 1,
and easy preset testing. **Costs:** groups over-include kernels, group membership
requires maintenance, feature combinations multiply, and one-feature-per-op is
poor developer experience. Shared helpers must remain available when any
dependent group is enabled.

### 4.2 Build-time code generation from models

This is the intended user-facing answer for a truly minimal build. A planned
`cargo xtask minimal-build` command accepts one or more `.onnx` files:

1. Load each model through `onnx-rs`.
2. Run the standard checker and resolve every node against `SchemaRegistry`,
   recursively including subgraphs and functions.
3. Collect normalized `(domain, op_type, opset)` requirements.
4. Apply the selected optimizer profile and include the possible fused operator
   outputs, or analyze a saved post-optimization graph. The manifest must record
   which method was used.
5. Union the sets for multiple models.
6. Compare the set with a single native-EP operator catalog that maps each
   operator key to its factory, source module, dependencies, and capability
   group. Unknown or unimplemented operators fail analysis.
7. Generate a deterministic selection manifest and registry root containing
   only selected factories, then invoke Cargo with `full` disabled. Module/group
   gates and release dead-code elimination ensure unreferenced kernels do not
   remain in the artifact.
8. Embed the manifest (or its hash and operator list) in the binary for startup
   validation and diagnostics.

Illustrative generated manifest:

```toml
format_version = 1
target_ep = "cpu"
optimizer_profile = "release-default"
models = [
  { path = "models/model.onnx", sha256 = "..." },
]
operators = [
  { domain = "ai.onnx", op_type = "MatMul", opset = 21 },
  { domain = "ai.onnx", op_type = "Softmax", opset = 21 },
]
```

**Advantages:** smallest model-specific root set, reproducible group builds,
good automation, and no public surface of 113 Cargo flags. **Costs:** more
tooling, generated-source reproducibility, optimizer/fusion closure, custom-op
handling, and careful Cargo rebuild invalidation.

**Recommendation:** expose model-driven codegen as the primary minimal-build
workflow, built on a small set of Cargo operator groups and a default `full`
feature. Groups provide a maintainable Phase 1 and human-authored presets;
codegen provides exact model/group selection in Phase 2. Do not expose one
per-op Cargo feature as the main interface.

### 4.3 Runtime registry trimming

The runtime could build the full registry and remove entries not used by the
loaded model. This may reduce registry allocation, lookup work, and startup
memory. It does **not** remove referenced kernel code from the binary and
therefore does not solve the binary-size goal. It may be a complementary runtime
optimization, not the minimal-build mechanism.

## 5. Intended user workflows

These commands describe the planned interface and do not work yet.

### 5.1 One model

```console
cargo xtask minimal-build \
  --ep cpu \
  --model models/phi.onnx \
  --output target/minimal/phi
```

The command validates the model, writes
`target/minimal/phi/operator-selection.toml`, builds with `full` disabled, and
places the release artifact beside the manifest. The output reports selected
operator count and artifact size.

### 5.2 A group of models

```console
cargo xtask minimal-build \
  --ep cpu \
  --model models/embed.onnx \
  --model models/reranker.onnx \
  --model models/generator.onnx \
  --output target/minimal/service
```

The analyzer takes the union of all required operator keys and emits one
reproducible service profile. A checked-in manifest may later be rebuilt with:

```console
cargo xtask minimal-build --manifest deploy/service-ops.toml
```

Paths in checked-in manifests should be accompanied by content hashes so model
replacement cannot silently reuse a stale operator set.

### 5.3 Loading a model outside the selected set

Session creation must compare the model's requirements with the embedded build
manifest before kernel compilation. A missing operator uses the existing
actionable session-error style and names the domain, operator, node, and opset:

```text
unsupported operator ai.onnx::Conv: this CPU binary was built without its
kernel; node "encoder/conv1", opset 21; build profile "service-ops".
To fix: rebuild with this model:
  cargo xtask minimal-build --manifest deploy/service-ops.toml \
    --model models/new-model.onnx
or use the default full build.
```

There is no silent fallback to code that is absent. Another configured EP may
run the node only if its kernel is present; otherwise model loading fails before
inference. Custom domains require an explicit plugin/custom-op declaration in
the manifest or fail analysis.

## 6. Binary-size measurement

Keep tracking lightweight and reproducible:

1. Add one `cargo xtask binary-size` (or Make) target that builds the same
   release CPU artifact twice: default `full`, then a stable minimal fixture
   manifest.
2. Use the same target triple, Cargo profile, linker, strip setting, and LTO
   setting for both builds.
3. Record raw bytes, stripped bytes where applicable, absolute delta, percentage
   delta, selected operator count, commit, and toolchain version.
4. CI runs the target on changes to EP kernels, feature/catalog metadata, build
   profiles, or the minimal-build tool. Initially report results without a hard
   regression gate; add a budget only after several stable baselines.
5. Store the latest agreed baseline in this document (or a small generated table
   linked from it), while CI retains per-run artifacts.

| Profile | Selected ops | Raw bytes | Stripped bytes | Baseline commit |
|---|---:|---:|---:|---|
| CPU `full` default | TBD | TBD | TBD | Not measured |
| CPU minimal fixture | TBD | TBD | TBD | Not measured |

This extends [DESIGN §33.6](DESIGN.md#336-tracking-metrics) from ORT deployment
size to native-EP full-versus-minimal artifacts.

## 7. Developer experience and maintainability

- `full` remains the default, so existing users, tests, and downstream crates do
  not break.
- Minimal builds are explicit and reproducible; no model inspection occurs
  implicitly during an ordinary `cargo build`.
- Operator groups—not 113 public flags—are the supported manual interface.
- Maintain one operator catalog as the source of truth for registry generation,
  feature groups, analyzer validation, and coverage tests. CI should verify that
  every registry entry is catalogued and every selected key resolves to a
  factory.
- Group tests compile representative profiles, not the power set. Model-driven
  tests build a small fixture and verify both successful loading and the exact
  stripped-op diagnostic.
- Shared factories, versioned registrations, contrib-domain aliases, optimizer
  fusions, and helper dependencies are catalogued explicitly to avoid accidental
  omission.

## 8. Phasing

All phases are **NOT YET IMPLEMENTED**.

### Phase 1 — operator-group Cargo features (small)

- Add the group features above and default `full`.
- Gate both `pub mod` declarations and matching `build_cpu_registry()`
  registrations in `crates/onnx-runtime-ep-cpu/src/kernels/mod.rs`.
- Introduce the operator catalog and test `full`, `ops-transformer`, and
  `ops-cnn` representative combinations.
- Preserve current default registry coverage and error behavior.

### Phase 2 — model-driven analyzer and codegen

- Add recursive required-op collection to `onnx-rs`, reusing its schema/checker
  graph traversal and shape pipeline.
- Add the deterministic selection manifest and `cargo xtask minimal-build`.
- Generate the registry root, compute optimizer/fusion closure, embed the build
  profile, and implement the actionable stripped-op load error.
- Support one-model and union-of-models workflows.

### Phase 3 — measurement and CUDA parity

- Add the lightweight full-versus-minimal CPU size target and CI reporting;
  populate the baseline table.
- Apply the shared catalog/manifest design to the native CUDA EP, accounting for
  library dependencies and architecture-specific generated kernels.
- Establish regression budgets only after measurements are stable.
