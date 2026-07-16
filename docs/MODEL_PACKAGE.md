# Model Package Support — Design Proposal

> **Status:** Proposal for review; no implementation is implied by this document.
>
> **Scope:** A package format and loading/tooling design for the pure-Rust ONNX
> runtime (`nxrt`) and the generation engine. The compatibility target is the
> ONNX Runtime (ORT) model-package schema as it exists on 2026-07-16.

## 1. Motivation and goals

Today a deployable generative model is a loose set of files: one or more ONNX
graphs, external weights, tokenizer assets, inference metadata or
`genai_config.json`, optional adapters, and potentially compiled execution
provider (EP) contexts. The files have implicit relative-path relationships but
no common identity, compatibility declaration, integrity inventory, or
deterministic variant-selection rule.

A model package should provide:

1. **Portable distribution.** Move, cache, mirror, and install one logical
   artifact without reconstructing an undocumented directory convention.
2. **Reproducible deployment.** Record package, graph, weight, tokenizer,
   compiler, EP, and device identities so the runtime can explain exactly what
   it selected.
3. **Compiled-EP startup.** Reuse `EPContext` models and external context
   binaries instead of repeating convert-and-compile work at deployment time.
4. **Variant selection.** Publish CPU, CUDA, OpenVINO, QNN, Metal, or other
   hardware-specific builds together and select the best compatible variant.
5. **Zero-copy weights.** Preserve the current external-weight `mmap` path after
   package resolution; packaging must not force weights through an archive
   decompressor on every load.
6. **GenAI completeness.** Carry `genai_config.json`, native
   `inference_metadata.{yaml,json}`, tokenizer/processor files, adapters, draft
   models, and pipeline components.
7. **Offline operation.** Loading and generation must not require a network,
   model hub, compiler SDK, or mutable global state.
8. **Minimal-build composition.** A package must be analyzable into the
   deterministic operator-selection manifest proposed in
   [MINIMAL_BUILD.md](MINIMAL_BUILD.md), including every runnable fallback
   variant.
9. **Inspectable tooling.** Users must be able to pack, unpack, validate, hash,
   and explain variant selection before creating a session.

This supports the project's north star: llama.cpp-like ease of model
distribution with a runtime architecture capable of vLLM-class serving. A
package is the deployment boundary joining model semantics, optimized runtime
artifacts, and generation assets without hardcoding a model family.

### Non-goals

- Define a new graph IR or replace ONNX.
- Interpret vendor EPContext payload bytes.
- Make raw CUDA graph handles portable.
- Fetch missing files at load time.
- Silently run a package whose declared hashes or compatibility constraints do
  not match.
- Require packages for ordinary `.onnx` files or existing flat model
  directories.

## 2. Background and source study

### 2.1 ORT standalone `model_package` library

The package format is owned by ORT's standalone
[`model_package/`](https://github.com/microsoft/onnxruntime/tree/main/model_package)
C library, not by the session integration itself. Its source contains
`include/model_package.h`, `include/model_package_api.h`, and implementations
such as `manifest_parser.cc`, `path_resolver.cc`, `asset_hasher.cc`,
`authoring.cc`, and `commit_prune_validate.cc` [ORT-LIB].

The library defines a **directory**, not a single archive:

```text
package_root/
├── manifest.json
├── decoder/
│   ├── component.json
│   └── cuda/
│       ├── model.onnx
│       └── ort_info.json
└── shared_assets/
    └── sha256-<64-lowercase-hex>/
        ├── tokenizer.json
        └── chat_template.jinja
```

Its schema has:

- top-level `schema_version`, optional `package_name`, `package_version`,
  `description`, `layout`, `additional_metadata`, required `components`, and
  optional `shared_assets`;
- inline component objects or external component paths; external directory
  components use a fixed `component.json`;
- a `variants` map whose entries have `variant_directory`, `ep`, `device`,
  `compatibility_string`, `executor_info`, and `additional_metadata`;
- consumer-owned `executor_info` namespaces. The standalone library deliberately
  knows nothing about ONNX, sessions, EPs, or the JSON under
  `executor_info["ort"]`;
- `portable` layout confinement by default, while `installed` may use absolute
  paths and `..`;
- content-addressed shared-asset directories referenced as
  `sha256:<digest>[/confined/tail]`.

The canonical directory hash includes both names and bytes: hash every file,
sort lines of `<file-sha256>  <relative-posix-path>\n`, then SHA-256 that
manifest. Symlinks are rejected. The default location is
`shared_assets/sha256-<digest>/`; `manifest.shared_assets` only overrides that
location.

The concrete C surface includes `ModelPackage_Open`, `ModelPackage_New`,
`ModelPackage_Info`, component/variant lookup and JSON getters,
`ModelPackage_ResolveStringRef`, `ModelPackage_ComputeDirectoryHash`,
`ModelPackage_SetComponent*`, `ModelPackage_SetVariant*`,
`ModelPackage_SetVariantExecutorInfo*`, `ModelPackage_AddSharedAsset`,
`ModelPackage_Commit`, `ModelPackage_Prune`, and `ModelPackage_Validate`.
`Commit` writes a directory in preserve or dense-manifest mode; it is not an
archive packer [ORT-LIB].

`schema_version` is `"<major>.<minor>"`. Major gates the data contract and an
unsupported major is rejected. Minor changes are additive; a reader accepts a
newer minor and relaxes unknown-field strictness so fields can be preserved.

### 2.2 ORT session integration

The requested ORT directory
[`onnxruntime/core/session/model_package/`](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/session/model_package)
currently contains:

- `README.md`
- `model_package_context.h/.cc`
- `model_package_options.h/.cc`
- `model_package_variant_selector.h/.cc`

The C entry points live separately in
`onnxruntime/core/session/model_package_api.cc` [ORT-CORE].

ORT adds three policies over the standalone library:

1. It owns the `executor_info["ort"]` payload:

   ```json
   {
     "model_file": "model.onnx",
     "session_options": {
       "session.model_external_initializers_file_folder_path": "weights",
       "ep.context_file_path": "contexts/model.ctx"
     },
     "provider_options": { "device_id": "0" }
   }
   ```

   The payload may be inline or an external JSON file such as `ort_info.json`.
   `model_file` is required to create a session. ORT resolves the model file and
   the two currently recognized path-valued session options through
   `ModelPackage_ResolveStringRef`.

2. `ModelPackageOptions` snapshots EP intent from session options.
   `VariantSelector` currently considers only the first configured EP, filters
   variants by exact EP name and optional `cpu`/`gpu`/`npu` device, then passes
   the opaque `compatibility_string` to
   `OrtEpFactory::ValidateCompiledModelCompatibilityInfo`. Scores are:
   `EP_SUPPORTED_OPTIMAL=100`,
   `EP_SUPPORTED_PREFER_RECOMPILATION=50`,
   `EP_NOT_APPLICABLE=0`, and `EP_UNSUPPORTED` rejects the candidate.
   Manifest order breaks ties. Ranking across the full EP list is still a TODO.

3. The experimental, version-28 API creates package/options/component contexts,
   enumerates components and variants, selects a component, resolves paths, and
   creates a session. With no caller session options, ORT applies the package's
   session/provider options. With caller options, package options are generally
   not merged, except missing path-valued session entries are retained so
   external weights and EP context files remain loadable.

At the ORT core level a package is therefore a **variant-selection and path
resolution container**. Compiled models are represented by a selected model
file—typically an ONNX graph containing `com.microsoft::EPContext` nodes—plus
any external context binaries and external initializers it references. The
package does not replace the EPContext graph format.

### 2.3 onnxruntime-genai package documentation

[`docs/model_package.md`](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/model_package.md)
defines a GenAI convention over the ORT schema [GENAI-DOC]:

- one inline component conventionally named `model`;
- each variant directory contains a complete `genai_config.json`, ONNX graph
  files, external weights, and any variant-specific custom-op libraries,
  adapters, or other assets;
- tokenizer assets may be shared through a
  `shared_assets/sha256-<digest>/` directory and referenced by
  `model.tokenizer_dir: "sha256:<digest>"`;
- processor-specific
  `model.vision.config_filename`/`model.speech.config_filename` remain
  per-variant in the documented convention;
- `OgaCreateModel(path)` / `og.Model(path)` auto-detect an EP only when the
  package is unambiguous; multi-EP packages use
  `OgaCreateConfigFromPackageEp` / `og.Config.from_package_ep`;
- a `sha256:` reference outside a package is invalid;
- flat model directories continue to load unchanged;
- model packages cannot currently be combined with `OgaRuntimeSettings`.

The current example uses `schema_version: "1.0"` and variant directories
directly below the package root. Despite the common `.ortpackage` suffix, it is
still a directory.

### 2.4 onnxruntime-genai PR #2255

Merged PR
[`microsoft/onnxruntime-genai#2255`](https://github.com/microsoft/onnxruntime-genai/pull/2255),
**“Resolve model package paths and path-valued session options through ONNX
Runtime”**, is the most current reference [GENAI-2255]. It changed:

- `docs/model_package.md`
- `src/config.cpp`, `src/config.h`
- `src/models/model.cpp`
- `src/models/model_package.cpp`, `src/models/model_package.h`
- `src/models/onnxruntime_api.h`, `src/models/onnxruntime_inline.h`
- `test/model_package_test.cpp`

The important decisions are concrete:

1. It replaced the earlier `package:<relative-path>` convention with ORT-owned
   `sha256:<digest>[/tail]` resolution and changed authoring examples to inline
   components with schema `"1.0"`.
2. `Config` now stores a `package_resolver` closure that captures the
   `OrtModelPackageContext`, keeping it alive for later `genai_config.json` path
   resolution.
3. `Config::ResolvePath` delegates package paths to ORT's
   `ModelPackage_ResolveStringRef`; flat directories explicitly reject
   `sha256:` references.
4. `session.model_external_initializers_file_folder_path` and
   `ep.context_file_path` are resolved before being applied to ORT session
   options.
5. For memory-loaded models, the external-initializer folder defaults to the
   config directory only if the config did not already set it.
6. GenAI resolves the experimental API functions locally instead of vendoring
   ORT's experimental C++ header.
7. Tests cover tokenizer shared assets, shared external initializers, relative
   path resolution, EP-context paths, and rejection of `sha256:` in flat
   directories.

PR #2255 adds no package archive, `pack` CLI, or new user-facing pack API. It
strengthens path ownership and lifetime semantics on top of the package-loading
surface added earlier.

### 2.5 Adopt versus diverge

| Topic | Decision |
|---|---|
| Manifest/component/variant schema | **Adopt ORT schema 1.x**, rather than inventing an incompatible manifest. |
| Canonical installed form | **Adopt ORT directory layout** for interoperability and `mmap`. |
| Shared assets | **Adopt `sha256:` URIs and canonical directory hashing.** |
| Consumer extension | Add `executor_info["nxrt"]`; also read compatible fields from `"ort"` where safe. |
| GenAI convention | Adopt complete per-variant `genai_config.json` and shared tokenizer assets, but prefer native `inference_metadata` when both exist. |
| Single-file distribution | **Diverge additively:** define an optional `.nxpkg` transport archive which extracts to the canonical directory form before loading. |
| Variant policy | Start ORT-compatible, but rank across the user's full ordered EP preference list rather than only its first entry. |
| Compiled artifacts | Preserve `EPContext` as the executable interchange; package metadata inventories and validates it but does not invent a second compiled-graph format. |

## 3. Proposed package format

### 3.1 Two representations

1. **Canonical package directory** — ORT-compatible, directly loadable, and the
   only representation from which weights/context binaries are memory-mapped.
   Recommended suffix: `.ortpackage/` for cross-runtime packages or
   `.nxpackage/` for nxrt-specific packages.
2. **Optional `.nxpkg` transport archive** — one deterministic ZIP64 file for
   upload/download and registry distribution. It contains the canonical package
   directory contents at archive root. nxrt validates and extracts it into a
   content-addressed cache, then loads the extracted directory.

The archive is an envelope, not a second manifest schema. Direct execution from
compressed members is forbidden because it defeats stable paths and zero-copy
weight `mmap`. Archive authoring uses sorted POSIX paths, normalized permissions,
zeroed timestamps, no symlinks, and stored (uncompressed) entries for already
compressed or mmap-critical large weight/context files. Extraction rejects
absolute paths, `..`, duplicate normalized paths, special files, and configured
size/entry-count limits.

### 3.2 Example directory

```text
phi4-mini.ortpackage/
├── manifest.json
├── generic-cpu/
│   ├── nxrt_info.json
│   ├── genai_config.json
│   ├── inference_metadata.yaml
│   └── decoder.onnx
├── cuda-sm90/
│   ├── nxrt_info.json
│   ├── genai_config.json
│   ├── inference_metadata.yaml
│   ├── decoder_ctx.onnx
│   └── artifacts/
│       └── epcontext/
│           └── CUDAExecutionProvider/
│               └── sha256-8b6e….bin
├── openvino-npu/
│   ├── nxrt_info.json
│   ├── genai_config.json
│   ├── decoder_ctx.onnx
│   └── model.onnx.data
└── shared_assets/
    ├── sha256-a14d…/
    │   ├── tokenizer.json
    │   ├── tokenizer_config.json
    │   └── chat_template.jinja
    └── sha256-f39c…/
        └── adapters/
            └── support.lora
```

Each variant is independently runnable. A compiled variant's
`decoder_ctx.onnx` contains ordinary `com.microsoft::EPContext` nodes. For
`embed_mode=0`, `ep_cache_context` names a relative file under
`artifacts/epcontext/...`; for `embed_mode=1`, the bytes remain in the ONNX
attribute. External ONNX weights remain relative to the selected model file or
use a resolved shared-asset folder.

### 3.3 Example `manifest.json`

```json
{
  "schema_version": "1.0",
  "package_name": "phi4-mini",
  "package_version": "4.0.0+nxrt.1",
  "description": "Portable CPU plus compiled CUDA/OpenVINO variants",
  "layout": "portable",
  "components": {
    "model": {
      "component_name": "model",
      "variants": {
        "cuda-sm90": {
          "variant_directory": "cuda-sm90",
          "ep": "CUDAExecutionProvider",
          "device": "gpu",
          "compatibility_string": "nxrt-cuda:v1;sm=90;cuda=13",
          "executor_info": {
            "nxrt": "nxrt_info.json"
          }
        },
        "openvino-npu": {
          "variant_directory": "openvino-npu",
          "ep": "OpenVINOExecutionProvider",
          "device": "npu",
          "compatibility_string": "<opaque-OpenVINO-token>",
          "executor_info": {
            "nxrt": "nxrt_info.json"
          }
        },
        "generic-cpu": {
          "variant_directory": "generic-cpu",
          "ep": "CPUExecutionProvider",
          "device": "cpu",
          "executor_info": {
            "nxrt": "nxrt_info.json"
          }
        }
      }
    }
  },
  "additional_metadata": {
    "publisher": "example",
    "license": "see LICENSE",
    "package_content_sha256": "<canonical-content-digest>"
  }
}
```

`compatibility_string` remains EP-owned and opaque to the package layer. The
illustrative CUDA syntax is not imposed on third-party EPs.

### 3.4 Example `nxrt_info.json`

```json
{
  "schema_version": "1.0",
  "model_file": "decoder_ctx.onnx",
  "genai_config": "genai_config.json",
  "inference_metadata": "inference_metadata.yaml",
  "tokenizer": "sha256:a14d…",
  "session_options": {
    "optimization": "none",
    "ep.context_file_path": "artifacts/epcontext/CUDAExecutionProvider/sha256-8b6e….bin"
  },
  "artifacts": [
    {
      "kind": "ep_context",
      "source": "CUDAExecutionProvider",
      "path": "artifacts/epcontext/CUDAExecutionProvider/sha256-8b6e….bin",
      "sha256": "8b6e…",
      "model_graph_sha256": "2f14…",
      "ep_version": "nxrt-cuda-0.3",
      "runtime_abi": "nxrt-ep-context-v1",
      "device_fingerprint": "cuda:sm90",
      "compiler_options_sha256": "11ca…",
      "shape_profile_sha256": "6d91…"
    }
  ],
  "minimal_build_profile": "operator-selection.json",
  "additional_metadata": {}
}
```

Rules:

- `model_file` is required for an executable nxrt variant.
- All path-shaped values are resolved by the package resolver against the
  variant directory. They may be confined relative paths or `sha256:` URIs.
- Native `inference_metadata` is authoritative. `genai_config.json` remains a
  compatibility fallback, matching current
  `onnx-genai-genai-config` behavior.
- `artifacts` is an integrity/diagnostic inventory. The ONNX EPContext nodes
  remain the executable source of partition boundaries, source keys, and
  payload references.
- A generic fallback is a separate variant, not a hidden alternate graph inside
  a compiled variant. This keeps selection and inspection deterministic.
- The `nxrt` payload schema versions independently from the ORT package schema.
  Unknown additive fields are ignored and preserved where possible.

### 3.5 Multi-component and GenAI packages

ORT's component map naturally represents pipeline models:

```text
components:
  decoder: variants [...]
  vision_encoder: variants [...]
  audio_encoder: variants [...]
  draft_model: variants [...]
  embedding: variants [...]
```

The simple onnxruntime-genai convention remains one component named `model`.
Our engine accepts both:

- a single `model` component containing a complete `genai_config.json`; or
- multiple named components connected by native `inference_metadata.pipeline`.

Tokenizer and processor assets may be component-local or shared by digest.
Adapters are ordinary variant files or shared assets; a config must name them
explicitly. No loader scans arbitrary files and guesses their purpose.

### 3.6 Versioning and compatibility

There are three separate contracts:

1. **ORT package schema** — `manifest.schema_version`; adopt ORT major/minor
   semantics.
2. **nxrt executor payload** — `executor_info["nxrt"].schema_version`.
3. **Compiled artifact ABI** — `runtime_abi`, EP version, device fingerprint,
   graph hash, compiler options hash, and shape profile.

Readers:

- reject unsupported major versions;
- accept newer minor versions and ignore unknown optional fields;
- reject malformed known fields and path escapes;
- never treat `package_version` as a compatibility key;
- may read `executor_info["ort"]` when it contains the common
  `model_file`/`session_options`/`provider_options` shape;
- must not claim that an `"ort"` compiled artifact is nxrt-loadable until its
  `EPContext.source` is owned by a registered EP.

## 4. Integration with this repository

### 4.1 Current-state constraints

The repository already has important pieces:

- `onnx-runtime-loader` parses ONNX, resolves external initializer files with a
  path-confinement guard, memory-maps them into `WeightStore`, and exposes typed
  EPContext views/resolution.
- `onnx-runtime-ep-api` has runtime `EpContext`, `context_source_keys()`,
  `save_context()`, `load_context()`, and a reject-duplicate
  `EpContextRegistry`.
- `onnx-runtime-session` consumes EPContext nodes before executor construction
  and can dump `*_ctx.onnx` from `ep.context_*` session options. The current
  concrete executor auto-detects only the CPU EP, so real compiled-EP reuse
  awaits a compiled EP.
- `onnx-rs::load_model`/`save_model` share the runtime IR and retain the live
  `WeightStore`.
- `onnx-genai-engine::Engine::from_dir` currently resolves a flat
  `ModelDirectory`, loads an ORT-backed session, prefers native inference
  metadata, falls back to converted `genai_config.json`, and loads
  `tokenizer.json`.
- `onnx-genai-genai-config` intentionally models only the GenAI fields needed
  for the metadata compatibility path.

One design/code mismatch must be explicit: `docs/ONNX_RS.md` specifies
`ModelFormat` and `FormatRegistry`, but the current `onnx-rs` crate does not yet
implement or export them. Package work should land the generic registry seam
before registering `ModelPackageFormat`; it must not create a package-only
parallel dispatcher.

### 4.2 Package crates and ownership

Proposed ownership:

| Crate | Responsibility |
|---|---|
| `onnx-model-package` (new, leaf) | Pure-Rust ORT 1.x manifest parsing/authoring, path resolver, hashing, validation, archive envelope, typed package/component/variant views. No session dependency. |
| `onnx-rs` | Register `ModelPackageFormat`; graph-only inspection; save/pack entry points built on shared IR. |
| `onnx-runtime-loader` | Load the selected model path, external weights, and EPContext blobs through already-resolved package paths; preserve `mmap`. |
| `onnx-runtime-ep-api` | EP compatibility validation contract and compiled-context save/load. |
| `onnx-runtime-session` | Device intent, variant ranking, EPContext validation/reuse, generic fallback selection, and session lifetime ownership of the package handle/cache lease. |
| `onnx-genai-genai-config` | Parse a resolved `genai_config.json`; add APIs accepting an explicit resolved path/base rather than assuming a flat directory. |
| `onnx-genai-engine` | Resolve package component(s), native metadata, tokenizer, adapters, and draft/pipeline models before constructing sessions. |
| Python/CLI (`nxrt`) | `pack`, `unpack`, `inspect`, `validate`, and package-aware `load`. |

The package crate should mirror ORT's semantics, not bind ORT's C library. That
keeps the runtime pure Rust and avoids a second C ABI. Conformance fixtures
should be authored/read by both implementations.

### 4.3 `ModelFormat` and `FormatRegistry`

`ModelPackageFormat` is a natural built-in format, but a package can contain
multiple components/variants while the proposed `ModelFormat::load` returns one
`Model`. The registry therefore needs a selection-aware load context rather
than silently choosing.

API sketch:

```rust,ignore
pub struct LoadOptions {
    pub component: Option<String>,
    pub variant: VariantPreference,
    pub graph_only: bool,
    pub verify_hashes: VerifyMode,
}

pub enum VariantPreference {
    Auto { devices: Vec<DevicePreference> },
    Named(String),
    Ep(String),
}

pub trait ModelFormat: Send + Sync {
    fn id(&self) -> &str;
    fn extensions(&self) -> &[&str];
    fn can_load(&self, path: &Path) -> bool;
    fn load(&self, path: &Path, opts: &LoadOptions)
        -> Result<Model, LoadError>;
    fn save(&self, model: &Model, path: &Path, opts: &SaveOptions)
        -> Result<(), SaveError>;
    fn load_graph_only(&self, path: &Path, opts: &LoadOptions)
        -> Result<Model, LoadError>;
}

pub struct ModelPackageFormat;

impl ModelFormat for ModelPackageFormat {
    fn id(&self) -> &str { "ort-model-package" }
    fn extensions(&self) -> &[&str] { &["ortpackage", "nxpackage", "nxpkg"] }
    // A directory probes by manifest.json; .nxpkg probes by archive magic
    // plus a root manifest.json entry.
}
```

For full package inspection and multi-component loading, expose a separate
typed handle instead of overloading `Model`:

```rust,ignore
pub struct ModelPackage {
    root: PackageRoot,
    manifest: Manifest,
}

impl ModelPackage {
    pub fn open(path: impl AsRef<Path>, opts: &PackageOpenOptions)
        -> Result<Self, PackageError>;
    pub fn components(&self) -> impl Iterator<Item = ComponentRef<'_>>;
    pub fn select(&self, component: &str, request: &SelectionRequest)
        -> Result<SelectedVariant<'_>, SelectionError>;
    pub fn resolve(&self, base: &Path, reference: &str, must_exist: bool)
        -> Result<PathBuf, ResolveError>;
}
```

`FormatRegistry::with_builtins()` registers `ModelPackageFormat` after
protobuf. Directory probing must be explicit (`manifest.json`), not based only
on a suffix.

### 4.4 Session loading

```rust,ignore
let session = InferenceSession::builder()
    .model("phi4-mini.ortpackage")
    .component("model")
    .device(DevicePreference::Auto)
    .package_verify(VerifyMode::ManifestAndSelectedVariant)
    .build()?;
```

Proposed pipeline:

```text
probe input
  → open/extract package
  → parse + validate schema/paths
  → enumerate local EP/device capabilities
  → rank compatible variants
  → resolve nxrt/ort executor payload
  → verify selected graph/weight/context hashes
  → load graph + mmap external weights
  → dispatch EPContext nodes to source-keyed EP registry
  → if valid, skip convert/compile
  → otherwise try next compatible variant (normally generic)
  → optimize/compile/allocate
```

The session retains an `Arc<ModelPackage>` (and archive-cache lease, if any) for
at least as long as `WeightStore` and all external EPContext mappings. This is
the Rust equivalent of PR #2255 capturing `OrtModelPackageContext` in a resolver
closure.

The package resolver runs before loader path joins. Resolved files are then
passed to the existing loader as canonical paths. `WeightStore` continues to
map the actual extracted/installed file; no weight bytes are copied into a
package-owned buffer.

### 4.5 Generation-engine loading

Add package-aware entry points without breaking `Engine::from_dir`:

```rust,ignore
impl Engine {
    pub fn from_model(
        source: impl AsRef<Path>,
        config: EngineConfig,
    ) -> anyhow::Result<Self>;

    pub fn from_package(
        source: impl AsRef<Path>,
        selection: PackageSelection,
        config: EngineConfig,
    ) -> anyhow::Result<Self>;
}
```

`from_model` auto-detects a flat directory, ONNX file, package directory, or
`.nxpkg`. After selection it builds a resolved `ModelAssets`:

```rust,ignore
pub struct ModelAssets {
    pub model: PathBuf,
    pub inference_metadata: Option<PathBuf>,
    pub genai_config: Option<PathBuf>,
    pub tokenizer_dir: PathBuf,
    pub adapters: Vec<PathBuf>,
    pub component_models: BTreeMap<String, PathBuf>,
}
```

Metadata precedence stays unchanged:

1. resolved native `inference_metadata`;
2. resolved `genai_config.json` converted by
   `onnx-genai-genai-config`;
3. defaults with a warning.

Tokenizer discovery must use the resolved tokenizer directory, including
`sha256:` shared assets, rather than requiring
`variant_dir/tokenizer.json`. Draft, vision, audio, and embedding components
are selected with the same device request, but each component may choose a
different best variant if the manifest permits it. The selected component set
is recorded in engine diagnostics.

### 4.6 Minimal-build composition

The minimal-build analyzer accepts package inputs:

```console
cargo xtask minimal-build \
  --package phi4-mini.ortpackage \
  --variant-policy deploy-fallbacks \
  --ep cpu,cuda
```

Policies:

- `selected`: analyze only explicitly named variants;
- `deploy-fallbacks` (recommended): analyze every variant the deployed binary
  may select for the requested EP/device set;
- `all`: union every component and variant.

For ordinary ONNX variants, recursively collect `(domain, op_type, opset)` as
specified in [MINIMAL_BUILD.md](MINIMAL_BUILD.md). For EPContext nodes:

- include the loader/session support for `com.microsoft::EPContext`;
- do not pretend the opaque compiled partition requires native kernels;
- include all operators from any generic fallback variant;
- record package digest, component/variant names, selected graph hashes,
  optimizer profile, EP plugin requirements, and `source` keys in the generated
  operator-selection manifest.

At runtime, the binary compares its embedded profile to the selected variant.
A package must not select a generic fallback requiring kernels stripped from
the binary; fail before session compilation with the existing actionable
minimal-build diagnostic.

## 5. Compiled EP / EPContext design

### 5.1 One executable representation

The package does not create a parallel compiled-model abstraction.
`com.microsoft::EPContext` remains the interchange:

- variadic graph boundary inputs/outputs;
- `source` dispatch key;
- `main_context`;
- `ep_cache_context`;
- `embed_mode`;
- `ep_sdk_version`;
- `hardware_architecture`;
- `partition_name`;
- optional notes/size metadata.

Our existing loader/session/EP API already models this flow. The package adds
selection, identity, integrity, and lifetime around it.

### 5.2 Artifact identity

A compiled candidate is keyed by:

```text
(
  component,
  source_graph_sha256,
  ep_source_key,
  ep_implementation_version,
  runtime_ep_context_abi,
  device_fingerprint,
  compiler_options_sha256,
  shape_profile_sha256
)
```

The content blob has its own SHA-256. `compatibility_string` remains the EP's
opaque fast-selection token; the structured tuple is for nxrt validation,
cache keys, inspection, and reproducibility. An EP may strengthen validation
but may not weaken hash or graph-identity checks.

### 5.3 Selection and validation

For each component:

1. Walk the user's ordered device/EP preferences.
2. Require exact declared EP name when present.
3. Require the declared coarse device class when present.
4. Ask the EP to validate its opaque `compatibility_string`.
5. Rank `Optimal` above `PreferRecompilation` above `NotApplicable`; reject
   `Unsupported`.
6. Prefer a hash-verified precompiled variant over recompilation only when the
   EP says it is safe. `PreferRecompilation` may be selected under an explicit
   startup-latency policy; otherwise continue to a generic variant.
7. Break ties by manifest order for ORT compatibility and report the reason.

After graph load, every EPContext node is dispatched only by its `source`
through `EpContextRegistry`; package `ep` is not a substitute for the node's
source key. Duplicate source registrations remain fatal.

Validation before `load_context()`:

- graph and external blob exist and remain inside the resolved package/asset;
- declared content hashes match;
- graph hash, EP ABI, EP version policy, compiler options, device fingerprint,
  and shape profile are compatible;
- `main_context=0` references resolve to a loaded matching primary context;
- the owning EP accepts the node's `source`.

### 5.4 Fallback behavior

Fallback is explicit and observable:

| Condition | Behavior |
|---|---|
| No compiled artifact for the device | Select the next compatible variant, normally generic ONNX. |
| Hash/path/schema failure | Reject that variant; do not compile from tampered content. |
| EP says `Unsupported` | Reject candidate and continue. |
| EP says `PreferRecompilation` | Apply policy: prefer generic source variant by default; allow compiled candidate only with an explicit latency-first policy. |
| `EPContext.source` has no registered EP | Reject candidate and continue; error if no fallback exists. |
| Generic fallback selected | Run the ordinary optimize/compile path and optionally emit a new package variant only through explicit tooling. |

The runtime never mutates a signed/read-only package during fallback. Locally
compiled contexts go to a separate content-addressed runtime cache. A later
`nxrt package add-variant` command may deliberately promote a cache entry into
a new package.

### 5.5 Packaging generated contexts

The existing session options:

- `ep.context_enable`
- `ep.context_file_path`
- `ep.context_embed_mode`

continue to produce an EPContext ONNX model and optional external blobs.
`nxrt package pack --compile-for ...` orchestrates that existing writer, then:

1. hashes source graph, generated graph, and blobs;
2. moves/copies them into the variant;
3. rewrites only confined relative references where necessary;
4. records executor metadata and compatibility data;
5. validates by reopening the package and loading with compilation disabled.

Default package authoring should use external blobs for large contexts to keep
the graph inspectable and mmap-friendly. Embedded contexts remain supported for
small artifacts and maximum portability.

### 5.6 CUDA graph capture

A CUDA graph capture is not equivalent to an EP compiled context: it depends on
stable device addresses, allocation plans, stream state, shapes, driver/runtime
versions, and process-local handles. Phase 1 therefore packages **capture
recipes**, not raw captures:

```json
{
  "kind": "cuda_graph_recipe",
  "shape_profile_sha256": "…",
  "buffer_plan_sha256": "…",
  "required_stable_inputs": ["input_ids", "position_ids"],
  "warmup_runs": 3
}
```

The runtime may build a local captured graph after load and cache it outside the
package under a key including package/variant digest, device UUID, driver,
runtime, shape profile, and buffer plan. A future relocatable serialization is
allowed only if CUDA exposes a documented, safe representation and the same
fail-closed validation rules can be met.

## 6. CLI and Python tooling

### 6.1 CLI

```console
# Create an ORT-compatible package directory.
nxrt package pack model-dir/ --output phi4.ortpackage/

# Add/compile variants.
nxrt package pack model-dir/ \
  --variant generic-cpu --ep cpu \
  --variant cuda-sm90 --ep cuda --device gpu \
  --compile-context \
  --output phi4.ortpackage/

# Create deterministic single-file transport.
nxrt package archive phi4.ortpackage/ --output phi4.nxpkg

# Extract without loading.
nxrt package unpack phi4.nxpkg --output phi4.ortpackage/

# Inspect structure and explain local selection.
nxrt package inspect phi4.ortpackage/
nxrt package inspect phi4.nxpkg --device gpu:0 --json

# Structural, path, hash, and optional load validation.
nxrt package validate phi4.ortpackage/ --rehash-assets --load-selected

# Promote a locally compiled cache artifact deliberately.
nxrt package add-variant phi4.ortpackage/ \
  --from-cache <cache-key> --name cuda-sm90
```

`pack` preserves a directory output; `archive` is intentionally named
separately so users do not confuse ORT's package directory with our transport
envelope. `inspect` reports:

- package and executor schema versions;
- components, variants, EP/device/compatibility strings;
- resolved model/config/tokenizer/weight/context paths;
- hashes and verification status;
- local EP compatibility score and selection reason;
- fallback chain;
- minimal-build profile compatibility.

### 6.2 Rust

```rust,ignore
let package = ModelPackage::open("phi4.nxpkg", &PackageOpenOptions::default())?;
let report = package.inspect(&LocalCapabilities::detect()?)?;
let selected = package.select("model", &SelectionRequest::auto())?;
let session = InferenceSession::builder()
    .selected_variant(selected)
    .build()?;
```

### 6.3 Python (`nxrt`)

```python
import nxrt

# Same zero-config entry point for .onnx, package directories, and .nxpkg.
session = nxrt.load("phi4.ortpackage")

# Explicit component/device/variant.
session = nxrt.load(
    "phi4.nxpkg",
    component="model",
    device="gpu:0",
    variant="cuda-sm90",
)

pkg = nxrt.ModelPackage.open("phi4.nxpkg")
print(pkg.inspect(device="gpu:0"))
pkg.validate(rehash_assets=True)

nxrt.package.pack("model-dir", output="phi4.ortpackage")
nxrt.package.archive("phi4.ortpackage", output="phi4.nxpkg")
nxrt.package.unpack("phi4.nxpkg", output="phi4.ortpackage")
```

For GenAI, the high-level engine should also accept the package directly:

```python
engine = nxrt.genai.Engine.from_model("phi4.nxpkg", device="auto")
```

## 7. Security and reliability requirements

Packages are untrusted input. Before session creation:

- enforce portable path confinement and reject traversal;
- reject archive path collisions after normalization, symlinks, hardlinks,
  devices, FIFOs, and extraction outside the cache root;
- cap manifest/config size, archive entry count, expanded bytes, and compression
  ratio;
- validate JSON types and duplicate names;
- verify selected executable artifacts before mapping/executing them;
- never auto-load a custom-op or EP dynamic library merely because it exists in
  a variant; plugin loading requires explicit policy/signature trust;
- keep extracted archive directories immutable and content-addressed;
- use atomic extraction/rename and a lease while sessions hold mmaps;
- do not garbage-collect a cache directory with live leases;
- make `inspect` and structural validation possible without loading plugins or
  executing model code.

Package signatures are deferred, but the manifest must leave room for a
detached signature over the canonical content digest. Hashes provide integrity,
not publisher authenticity.

## 8. Decisions

| ID | Decision | Rationale |
|---|---|---|
| MP-1 | ORT package schema 1.x is the base format. | Maximum interoperability; avoids a competing manifest. |
| MP-2 | Canonical executable form is a directory. | ORT compatibility, stable relative paths, and weight/context `mmap`. |
| MP-3 | `.nxpkg` is an optional deterministic transport archive extracted before load. | Meets single-file distribution without sacrificing mmap or pretending ORT accepts archives. |
| MP-4 | Use `executor_info["nxrt"]`, with safe `"ort"` compatibility reads. | The ORT schema explicitly delegates consumer payloads by namespace. |
| MP-5 | EPContext remains the only compiled-graph interchange. | Existing ORT ecosystem and existing repository implementation; no duplicate compiled format. |
| MP-6 | Generic fallback is a separate variant. | Deterministic selection, inspection, minimal-build analysis, and no hidden graph switch. |
| MP-7 | `sha256:` resolution is owned by one package resolver kept alive with the session. | Follows ORT and PR #2255; prevents inconsistent ad hoc joins/lifetime bugs. |
| MP-8 | Native inference metadata wins over `genai_config.json`. | Preserves current engine semantics while supporting OGA packages. |
| MP-9 | Raw CUDA graph captures are not portable package artifacts in MVP. | Their handles/addresses are process/device specific; package recipes and local cache instead. |
| MP-10 | Package support lands through `ModelFormat`/`FormatRegistry`, after that planned seam is implemented. | Avoids a package-only format dispatcher and follows ONNX_RS §3. |

## 9. Phased rollout

### Phase 0 — conformance design

- Finalize `executor_info["nxrt"]` JSON schema and fixtures.
- Cross-read fixtures with ORT's standalone library.
- Decide `.nxpkg` archive limits and canonicalization.
- Resolve the open questions below.

### Phase 1 — directory MVP

- Pure-Rust parse/inspect/validate for ORT package directories.
- Portable path resolver and `sha256:` shared assets.
- Implement the planned `ModelFormat`/`FormatRegistry` seam and register
  `ModelPackageFormat`.
- Select one component/variant for CPU and explicit named variants.
- Load external weights with existing `WeightStore` mmap.
- Package-aware native metadata, `genai_config.json`, and tokenizer resolution.
- `nxrt package inspect|validate|pack|unpack-directory`.
- Consume existing EPContext packages; generic fallback variant.

### Phase 2 — compiled variants and authoring

- Full ordered EP/device ranking and compatibility callback.
- Package the output of the existing EPContext dump path.
- Artifact hash inventory and local compiled-context cache.
- Minimal-build package analyzer.
- Multi-component pipeline selection.
- GenAI adapters/draft/component assets.

### Phase 3 — transport and distribution

- Deterministic `.nxpkg` ZIP64 envelope.
- Atomic content-addressed extraction cache and leases.
- Python authoring/inspection APIs.
- Optional detached signatures and registry integration.

### Phase 4 — performance artifacts

- Capture recipes and local CUDA graph cache.
- Evaluate whether any future CUDA/Metal graph representation is safely
  relocatable.
- Shared-asset installation/cache deduplication across packages.

## 10. Open questions for review

1. **Archive scope:** Is `.nxpkg` required in the first implementation, or is an
   ORT-compatible package directory sufficient for MVP?
2. **Schema ownership:** Should `executor_info["nxrt"]` be published as a
   standalone JSON Schema immediately, or remain experimental until one real
   CUDA/Metal compiled package exists?
3. **ORT payload:** Should nxrt prefer `"nxrt"` and only fall back to `"ort"`, or
   should authoring write both payloads when their common fields are identical?
4. **Selection policy:** Should `PreferRecompilation` default to the stale
   compiled artifact for lowest startup latency, or to a generic variant for
   best steady-state performance? This proposal chooses generic/recompile.
5. **Fallback closure:** Must every published compiled variant have a generic
   fallback, or may appliance packages intentionally be hardware-exclusive?
6. **Shared weights:** Upstream path-valued session options can point an
   external-initializer folder at a `sha256:` asset. Should our MVP support that
   immediately, or first require weights inside each variant?
7. **Package signatures:** Which trust model is needed before package-contained
   custom-op/EP libraries can ever be enabled?
8. **GenAI components:** Should the first engine MVP support only the upstream
   single `model` component convention, deferring native multi-component
   pipelines?
9. **Extension naming:** Use `.nxpkg`, `.ortpackage.zip`, or no standard archive
   suffix until registry requirements are known?
10. **`ModelFormat` return type:** Extend `LoadOptions` as proposed, or introduce
    a general `ModelArtifact` enum so multi-component formats are first-class?

## 11. References

- **[ORT-LIB]** Microsoft, ONNX Runtime,
  [`model_package/README.md`](https://github.com/microsoft/onnxruntime/blob/main/model_package/README.md),
  [`include/model_package.h`](https://github.com/microsoft/onnxruntime/blob/main/model_package/include/model_package.h),
  and
  [`model_package/` sources](https://github.com/microsoft/onnxruntime/tree/main/model_package),
  accessed 2026-07-16.
- **[ORT-CORE]** Microsoft, ONNX Runtime,
  [`onnxruntime/core/session/model_package/README.md`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/model_package/README.md),
  [`model_package_context.h/.cc`](https://github.com/microsoft/onnxruntime/tree/main/onnxruntime/core/session/model_package),
  and
  [`model_package_variant_selector.cc`](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/model_package/model_package_variant_selector.cc),
  accessed 2026-07-16.
- **[GENAI-DOC]** Microsoft, onnxruntime-genai,
  [`docs/model_package.md`](https://github.com/microsoft/onnxruntime-genai/blob/main/docs/model_package.md),
  accessed 2026-07-16.
- **[GENAI-2255]** Microsoft, onnxruntime-genai,
  [PR #2255](https://github.com/microsoft/onnxruntime-genai/pull/2255),
  merged 2026-07-13; files and diff reviewed 2026-07-16.
- Repository-local context:
  [ONNX_RS.md](ONNX_RS.md), [ORT2.md](ORT2.md),
  [DESIGN.md](DESIGN.md), and [MINIMAL_BUILD.md](MINIMAL_BUILD.md).
