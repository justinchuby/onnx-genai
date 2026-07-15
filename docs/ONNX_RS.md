# onnx-rs — ONNX in Rust

> A pure-Rust replacement for the `onnx` Python package and C++ reference implementation.
> High-performance model I/O, validation, transformation, and tooling — no protobuf
> compiler, no C++ build, no numpy dependency.

**Scope:** The ONNX standard library (load/save/validate/transform models). NOT a runtime.
nxrt consumes onnx-rs as a library; they are separate concerns.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Architecture Overview](#2-architecture-overview)
3. [Protobuf Serde](#3-protobuf-serde)
4. [IR Layer](#4-ir-layer)
5. [Textual Representation](#5-textual-representation)
6. [JSON & TextProto Support](#6-json--textproto-support)
7. [Op Schema System](#7-op-schema-system)
8. [Checker / Validator](#8-checker--validator)
9. [Shape Inference](#9-shape-inference)
10. [Version Converter](#10-version-converter)
11. [Custom Op Registration](#11-custom-op-registration)
12. [Python Bindings](#12-python-bindings)
13. [Crate Structure](#13-crate-structure)
14. [Relationship to nxrt](#14-relationship-to-nxrt)
15. [Phased Roadmap](#15-phased-roadmap)
16. [Open Questions](#16-open-questions)

---

## 1. Motivation

### Why rewrite?

| Problem | Current (Python/C++) | Rust |
|---------|---------------------|------|
| Import time | `import onnx` → ~2s (protobuf + numpy) | Near-zero cold start |
| Build complexity | C++ protobuf compiler + cmake + platform deps | `cargo build`, zero system deps |
| Large model I/O | Full deserialization into Python objects | mmap + zero-copy, stream processing |
| Validation speed | Python checker + C++ shape inference (FFI hop) | Single-process, no FFI |
| Extensibility | Monkey-patching Python, C++ plugin API limited | Trait-based, compile-time checked |
| Distribution | Platform-specific wheels with C extensions | Pure Rust, cross-compile to anything |
| Memory safety | C++ onnx.proto manipulation is manual | Ownership-checked by compiler |

### What this is NOT

- NOT a runtime (that's nxrt)
- NOT a model optimizer (that's `onnx-runtime-optimizer`)
- NOT onnxscript/torchscript (graph construction DSLs are out of scope)

This is the **standard library** for working with ONNX models as data.

---

## 2. Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                     Python Bindings (PyO3)                      │
│                     `pip install onnx-rs`                       │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Checker  │  │ Version  │  │ Shape    │  │ Custom Op    │  │
│  │          │  │ Converter│  │ Inference│  │ Registry     │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│       │              │              │               │          │
│  ┌────▼──────────────▼──────────────▼───────────────▼──────┐  │
│  │                    Op Schema Registry                    │  │
│  │                    (YAML-generated)                      │  │
│  └─────────────────────────┬───────────────────────────────┘  │
│                            │                                   │
│  ┌─────────────────────────▼───────────────────────────────┐  │
│  │                         IR Layer                         │  │
│  │               (Graph, Node, Value, Tensor)               │  │
│  └─────────────────────────┬───────────────────────────────┘  │
│                            │                                   │
│  ┌────────────┐  ┌─────────▼──────┐  ┌─────────────────────┐ │
│  │ Text       │  │ Protobuf Serde │  │ JSON / TextProto    │ │
│  │ Format     │  │ (.onnx/.pb)    │  │                     │ │
│  │ (.onnxtxt) │  │                │  │                     │ │
│  └────────────┘  └────────────────┘  └─────────────────────┘ │
│                          │                                     │
│              ┌───────────┼───────────┐                        │
│              ▼           ▼           ▼                        │
│         .onnx        external     safetensors                 │
│         (binary)     data files   (.safetensors)              │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Protobuf Serde

### 3.1 Design Goals

- **Zero-copy weight access** via mmap — never load 70GB of weights into heap
- **Streaming parse** for models that don't fit in memory
- **External data** as first-class concept (ORT's `external_data_helper` is an afterthought)
- **Safetensors** as an alternative weight format (popular, safer, faster)
- **Round-trip fidelity** — load → save → load produces identical bytes

### 3.2 Extensible Format System

Model serialization is trait-based. Built-in formats (protobuf, safetensors) are just
default implementations. Users can register custom formats for proprietary or emerging
standards.

```rust
/// A format that can load/save ONNX models.
/// Implement this trait to add support for custom serialization formats.
pub trait ModelFormat: Send + Sync {
    /// Format identifier (e.g., "onnx-pb", "safetensors", "gguf").
    fn id(&self) -> &str;

    /// File extensions this format handles (e.g., ["onnx", "pb"]).
    fn extensions(&self) -> &[&str];

    /// Probe: can this format load the given file? (magic bytes / extension check)
    fn can_load(&self, path: &Path) -> bool;

    /// Load a model from this format.
    fn load(&self, path: &Path, opts: &LoadOptions) -> Result<Model, LoadError>;

    /// Save a model in this format. Return Err(Unsupported) if save not implemented.
    fn save(&self, model: &Model, path: &Path, opts: &SaveOptions) -> Result<(), SaveError>;

    /// Load only the graph structure (no weights). Useful for inspection.
    fn load_graph_only(&self, path: &Path) -> Result<Model, LoadError> {
        // Default: load full model. Formats can override for efficiency.
        self.load(path, &LoadOptions::default())
    }
}

/// A weight backend that can load/save tensor data.
/// Separate from ModelFormat because a single model can mix weight backends
/// (e.g., graph in .onnx, weights in .safetensors).
pub trait WeightBackend: Send + Sync {
    /// Backend identifier (e.g., "inline", "external-data", "safetensors", "gguf").
    fn id(&self) -> &str;

    /// Look up a tensor by name.
    fn get(&self, tensor_name: &str) -> Result<Option<WeightRef>, WeightError>;

    /// Store a tensor. Used during save.
    fn put(&mut self, tensor_name: &str, data: &[u8], dtype: DataType, shape: &[i64]) -> Result<(), WeightError>;

    /// List all available tensor names.
    fn tensor_names(&self) -> Vec<&str>;

    /// Flush/finalize (e.g., write safetensors header after all puts).
    fn finalize(&mut self) -> Result<(), WeightError> { Ok(()) }
}
```

### 3.3 Format Registry

```rust
/// Global registry of model formats and weight backends.
pub struct FormatRegistry {
    formats: Vec<Box<dyn ModelFormat>>,
    weight_backends: HashMap<String, Box<dyn WeightBackend>>,
}

impl FormatRegistry {
    /// Create with built-in formats.
    pub fn with_builtins() -> Self {
        let mut registry = Self::empty();
        registry.register_format(OnnxProtobufFormat::new());   // .onnx / .pb
        registry.register_format(SafetensorsFormat::new());     // .safetensors
        registry
    }

    /// Register a custom format.
    pub fn register_format<F: ModelFormat + 'static>(&mut self, format: F);

    /// Register a custom weight backend.
    pub fn register_weight_backend<W: WeightBackend + 'static>(&mut self, backend: W);

    /// Auto-detect format and load.
    pub fn load(&self, path: &Path) -> Result<Model, LoadError> {
        // 1. Try extension match
        // 2. Fall back to probing (magic bytes)
        // 3. Err if no format matches
        for fmt in &self.formats {
            if fmt.can_load(path) {
                return fmt.load(path, &LoadOptions::default());
            }
        }
        Err(LoadError::UnknownFormat(path.to_path_buf()))
    }

    /// Auto-detect format and save.
    pub fn save(&self, model: &Model, path: &Path, opts: &SaveOptions) -> Result<(), SaveError>;
}
```

### 3.4 Built-in: Protobuf Format

```rust
/// Standard ONNX protobuf format (.onnx, .pb).
pub struct OnnxProtobufFormat;

impl ModelFormat for OnnxProtobufFormat {
    fn id(&self) -> &str { "onnx-pb" }
    fn extensions(&self) -> &[&str] { &["onnx", "pb"] }
    fn can_load(&self, path: &Path) -> bool {
        // Check extension or protobuf magic bytes
        todo!()
    }
    fn load(&self, path: &Path, opts: &LoadOptions) -> Result<Model, LoadError> {
        // 1. mmap the file
        // 2. Decode protobuf header + graph structure via prost
        // 3. For each initializer:
        //    - Inline: reference the mmap slice (zero-copy)
        //    - External data: register in mmap registry (lazy load)
        // 4. Return Model with lazy WeightRef handles
        todo!()
    }
    fn save(&self, model: &Model, path: &Path, opts: &SaveOptions) -> Result<(), SaveError> {
        todo!()
    }
}
```

### 3.5 Built-in: Safetensors Weight Backend

```rust
/// Safetensors as a weight backend.
pub struct SafetensorsBackend {
    files: HashMap<PathBuf, Arc<SafetensorsFile>>,
}

impl WeightBackend for SafetensorsBackend {
    fn id(&self) -> &str { "safetensors" }
    fn get(&self, tensor_name: &str) -> Result<Option<WeightRef>, WeightError> {
        // Look up across all registered .safetensors files
        todo!()
    }
    fn put(&mut self, tensor_name: &str, data: &[u8], dtype: DataType, shape: &[i64]) -> Result<(), WeightError> {
        todo!()
    }
    fn tensor_names(&self) -> Vec<&str> { todo!() }
}
```

### 3.6 Example: Custom GGUF Format (User-Provided)

```rust
/// User registers GGUF support — onnx-rs doesn't need to know about it.
struct GgufFormat;

impl ModelFormat for GgufFormat {
    fn id(&self) -> &str { "gguf" }
    fn extensions(&self) -> &[&str] { &["gguf"] }
    fn can_load(&self, path: &Path) -> bool {
        // Check GGUF magic bytes: 0x46475547
        todo!()
    }
    fn load(&self, path: &Path, opts: &LoadOptions) -> Result<Model, LoadError> {
        // Parse GGUF header → construct ONNX IR graph
        // Map GGUF tensor metadata → WeightRef
        todo!()
    }
    fn save(&self, _model: &Model, _path: &Path, _opts: &SaveOptions) -> Result<(), SaveError> {
        Err(SaveError::Unsupported("GGUF save not implemented".into()))
    }
}

// Registration:
registry.register_format(GgufFormat);
```

### 3.7 Composite Loading (Multiple Weight Sources)

```rust
impl ModelLoader {
    /// Load model graph from one source, weights from another.
    /// e.g., graph from .onnx, weights from .safetensors
    pub fn load_composite(
        graph_path: &Path,
        weight_backends: Vec<Box<dyn WeightBackend>>,
    ) -> Result<Model, LoadError>;
}

/// How to persist weights on save.
pub struct SaveOptions {
    /// Weight storage strategy.
    pub weights: WeightStrategy,
    /// Protobuf format version (default: proto3).
    pub proto_version: ProtoVersion,
    /// Maximum size for inline weights (bytes). Larger → external file.
    pub inline_threshold: usize,  // default: 1024 (1KB)
}

#[derive(Clone, Debug)]
pub enum WeightStrategy {
    /// All weights inline in .onnx (original behavior).
    Inline,
    /// Weights in separate external data files (ONNX standard external data).
    ExternalData {
        /// Directory for external data files.
        data_dir: PathBuf,
        /// Split into multiple files if > threshold.
        max_file_size: Option<usize>,
    },
    /// Weights in safetensors format.
    Safetensors {
        /// Output .safetensors path.
        path: PathBuf,
    },
    /// Mixed: small weights inline, large weights external.
    Mixed {
        threshold: usize,
        external: Box<WeightStrategy>,
    },
}
```

### 3.8 Weight Reference (Zero-Copy)

```rust
/// A reference to weight data. Never copies until explicitly materialized.
pub enum WeightRef {
    /// Points into the mmap'd .onnx file.
    Inline { mmap: Arc<Mmap>, offset: usize, len: usize },
    /// Points to a mmap'd external data file.
    External { mmap: Arc<Mmap>, offset: usize, len: usize, path: PathBuf },
    /// Points into a safetensors file (via safetensors crate).
    Safetensors { file: Arc<SafetensorsFile>, tensor_name: String },
    /// Owned bytes (created programmatically, not from file).
    Owned(Vec<u8>),
}

impl WeightRef {
    /// Get a byte slice without copying. Works for all variants.
    pub fn as_bytes(&self) -> &[u8];

    /// Get typed slice (e.g., &[f32]). Checks alignment.
    pub fn as_slice<T: Pod>(&self) -> Result<&[T], AlignmentError>;

    /// Materialize into owned Vec<u8> (only copies if needed).
    pub fn to_owned(&self) -> Vec<u8>;

    /// Size in bytes.
    pub fn len(&self) -> usize;

    /// Data type (from associated TensorProto metadata).
    pub fn dtype(&self) -> DataType;

    /// Shape.
    pub fn shape(&self) -> &[i64];
}
```

### 3.9 Weight Conversion Utilities

```rust
/// Convert between weight formats.
pub struct WeightConverter;

impl WeightConverter {
    /// Convert model weights: any backend → safetensors.
    pub fn to_safetensors(model: &Model, output: &Path) -> Result<(), ExportError>;

    /// Convert safetensors → ONNX external data.
    pub fn to_external_data(st_path: &Path, output_dir: &Path) -> Result<Vec<ExternalDataRef>, ImportError>;

    /// Repack weights: consolidate multiple external files into one.
    pub fn repack(model: &mut Model, strategy: WeightStrategy) -> Result<(), SaveError>;
}
```

---

## 4. IR Layer

Reuse the design from `onnx-runtime-ir` (which was based on Justin's onnx-ir-py) but
with a key difference: **this IR is for the ONNX standard library** (validate, transform,
convert), not for runtime execution.

### 4.1 What's Shared vs. What's Different

| Concept | onnx-rs IR | onnx-runtime-ir (nxrt) |
|---------|-----------|------------------------|
| Graph, Node, Value | ✅ Same core types | ✅ Same |
| TensorLayout / strides | ❌ Not needed | ✅ Runtime concern |
| DeviceId placement | ❌ Not needed | ✅ Runtime concern |
| Subgraph (If/Loop/Scan) | ✅ Full support | ✅ Full support |
| Op domain + version | ✅ Rich (version converter needs it) | ✅ Basic |
| Function definitions | ✅ First-class (for expanding/inlining) | ❌ Inlined at load time |
| Metadata / doc_string | ✅ Preserved | ❌ Stripped |
| Training info | ✅ Preserved (for training tools) | ❌ Inference only |

### 4.2 Design Choice: Shared Crate or Separate?

**Option A: onnx-rs has its own IR, nxrt converts on load.**
- Pro: Each IR is optimal for its purpose
- Con: Conversion cost, two representations to maintain

**Option B: Shared `onnx-ir` crate, both depend on it.**
- Pro: One IR, no conversion
- Con: IR carries fields one side doesn't need (layout, device, doc_string)

**Recommendation: Option B with feature flags.** Shared `onnx-ir` crate. Fields like
`TensorLayout` and `DeviceId` are behind a `runtime` feature. Fields like `doc_string`
and `training_info` are behind a `standard` feature. Both sides see the same `Graph`/`Node`/`Value`.

---

## 5. Textual Representation

A human-readable text format for ONNX models. For debugging, diffing, code review, and
version control.

### 5.1 Why

- Binary protobuf is unreadable → can't `git diff` model changes
- Existing `onnx.printer` and `onnx.parser` are reference.

### 5.2 Format Design

Follow the exact design in the onnx repo.

### 5.3 Design Principles

- **Round-trip:** parse(print(model)) == model (modulo whitespace)
- **Weights are references, not inline.** Text format never embeds binary data
- **SSA-like syntax** — matches how people think about dataflow graphs
- **Attribute syntax:** `{ key: value }` after op, compact
- **Type syntax:** `dtype[shape]` (e.g., `f32[batch, 512]`)
- **Comments:** `//` line comments
- **Subgraphs:** Nested `graph` blocks for If/Loop/Scan body

### 5.4 Implementation referece

```rust
pub struct OnnxTextFormat;

impl OnnxTextFormat {
    /// Parse .onnxtxt into Model.
    pub fn parse(text: &str) -> Result<Model, ParseError>;

    /// Parse from file.
    pub fn load(path: &Path) -> Result<Model, ParseError>;

    /// Print Model to text.
    pub fn print(model: &Model) -> String;

    /// Print with options (indentation, weight detail level, etc.).
    pub fn print_with(model: &Model, opts: &PrintOptions) -> String;

    /// Save to .onnxtxt file.
    pub fn save(model: &Model, path: &Path) -> Result<(), IoError>;
}

pub struct PrintOptions {
    /// Indentation string (default: "  ").
    pub indent: String,
    /// Show weight shapes but not data (default: true).
    pub weight_shapes_only: bool,
    /// Show doc_strings (default: false).
    pub doc_strings: bool,
    /// Max line width for wrapping (default: 100).
    pub max_width: usize,
}
```

---

## 6. JSON & TextProto Support

### 6.1 JSON

Round-trip JSON serialization for web tooling, APIs, and interchange with non-protobuf
ecosystems.

```rust
pub struct OnnxJson;

impl OnnxJson {
    /// Serialize model to JSON. Weights are base64-encoded or referenced.
    pub fn to_json(model: &Model, opts: &JsonOptions) -> Result<String, JsonError>;

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Model, JsonError>;

    /// Load from .json file.
    pub fn load(path: &Path) -> Result<Model, JsonError>;

    /// Save to .json file.
    pub fn save(model: &Model, path: &Path, opts: &JsonOptions) -> Result<(), JsonError>;
}

pub struct JsonOptions {
    /// How to encode weights in JSON.
    pub weight_encoding: JsonWeightEncoding,
    /// Pretty-print with indentation.
    pub pretty: bool,
}

pub enum JsonWeightEncoding {
    /// Base64 inline (small models only).
    Base64Inline,
    /// Reference to external file (path string).
    ExternalRef,
    /// Omit weights entirely (graph structure only).
    OmitWeights,
}
```

### 6.2 TextProto

For compatibility with protobuf text format tooling.

```rust
let text = onnx_rs::textproto::to_textproto(&model)?;
let model = onnx_rs::textproto::from_textproto(&text)?;
```

**Implemented.** The functions live in `onnx_rs::textproto`, alongside the
existing `onnx_rs::json` module, rather than behind a zero-sized
`OnnxTextProto` type. Free functions are more idiomatic for stateless format
conversion and make the two protobuf interchange APIs predictable.

The implementation emits standard protobuf TextFormat field names and syntax:
quoted/escaped strings and bytes, named ONNX enums, nested `{ ... }` messages,
and one line per repeated value. `raw_data` is emitted as an escaped byte
string. Parsing and printing share the generated `ModelProto` conversion path
with binary protobuf and JSON. This gives Rust users a dependency-light,
inspectable interchange format for golden fixtures, diffs, debugging, and
interop with `google.protobuf.text_format` / `onnx.text_format`, without
confusing it with the human-oriented ONNX DSL in §5. Populated proto fields
that the shared IR cannot retain produce explicit errors rather than being
silently discarded.

**§11 implementation note (Wave 8, `1abcf8c`).** TextProto serde is implemented
through this shared protobuf conversion path, so TextFormat parsing and printing
retain the same explicit unsupported-field behavior as binary and JSON I/O.

### 6.3 Unified I/O

Delegates to `FormatRegistry` (§3.3) — auto-detects format from extension or magic bytes.

```rust
/// Auto-detect format and load. Uses the global FormatRegistry.
pub fn load(path: &Path) -> Result<Model, LoadError> {
    FormatRegistry::global().load(path)
}

/// Auto-detect format and save.
pub fn save(model: &Model, path: &Path, opts: &SaveOptions) -> Result<(), SaveError> {
    FormatRegistry::global().save(model, path, opts)
}

/// Register a custom format globally.
pub fn register_format<F: ModelFormat + 'static>(format: F) {
    FormatRegistry::global_mut().register_format(format);
}
```

---

## 7. Op Schema System

### 7.1 YAML-Defined Schemas

Instead of maintaining 700+ op schemas in C++ (onnx/defs.cc) or Python, define them
declaratively in YAML files and code-gen Rust structs.

```yaml
# schemas/standard/matmul.yaml
domain: ""
name: MatMul
since_version: 13
doc: "Matrix multiplication of two tensors."

inputs:
  - name: A
    type_str: "T"
    doc: "First input tensor"
  - name: B
    type_str: "T"
    doc: "Second input tensor"

outputs:
  - name: Y
    type_str: "T"
    doc: "Output tensor"

type_constraints:
  - type_param: "T"
    allowed:
      - float16
      - float32
      - float64
      - int32
      - int64
      - uint32
      - uint64
      - bfloat16

shape_inference: matmul  # references a named shape rule
```

```yaml
# schemas/standard/gemm.yaml
domain: ""
name: Gemm
since_version: 13
doc: "General Matrix multiplication."

attributes:
  - name: alpha
    type: float
    default: 1.0
  - name: beta
    type: float
    default: 1.0
  - name: transA
    type: int
    default: 0
  - name: transB
    type: int
    default: 0

inputs:
  - name: A
    type_str: "T"
  - name: B
    type_str: "T"
  - name: C
    type_str: "T"
    optional: true

outputs:
  - name: Y
    type_str: "T"

type_constraints:
  - type_param: "T"
    allowed: [float16, float32, float64, bfloat16, uint32, uint64, int32, int64]

shape_inference: gemm
```

### 7.2 Code Generation

```
schemas/
  standard/           # ONNX standard ops (ai.onnx domain)
    matmul.yaml
    gemm.yaml
    conv.yaml
    ...
  contrib/             # Contrib ops (com.microsoft, etc.)
    fused_attention.yaml
    rotary_embedding.yaml
    ...
  training/            # Training ops (ai.onnx.training)
    ...

build.rs → codegen → src/generated/
  op_schemas.rs        # All OpSchema structs
  op_registry.rs       # HashMap<(domain, name, version), OpSchema>
```

```rust
/// Generated op schema (one per op version).
pub struct OpSchema {
    pub domain: &'static str,
    pub name: &'static str,
    pub since_version: u32,
    pub doc: &'static str,
    pub inputs: &'static [InputSpec],
    pub outputs: &'static [OutputSpec],
    pub attributes: &'static [AttributeSpec],
    pub type_constraints: &'static [TypeConstraint],
    pub shape_inference_fn: Option<ShapeInferenceFn>,
    pub deprecated: bool,
}

pub struct InputSpec {
    pub name: &'static str,
    pub type_str: &'static str,
    pub doc: &'static str,
    pub optional: bool,
    pub variadic: bool,
    pub differentiable: Differentiability,
}

pub struct AttributeSpec {
    pub name: &'static str,
    pub attr_type: AttributeType,
    pub required: bool,
    pub default_value: Option<AttributeDefault>,
    pub doc: &'static str,
}

pub struct TypeConstraint {
    pub type_param: &'static str,
    pub allowed_types: &'static [DataType],
}
```

### 7.3 Why YAML Instead of .cc or .py

| Approach | Pros | Cons |
|----------|------|------|
| C++ (current onnx) | Fast at runtime | Build nightmare, hard to parse/modify |
| Python decorators | Easy to write | Slow, not usable from Rust |
| YAML + codegen | Human-readable, diffable, language-agnostic, CI-lintable | One-time codegen setup |
| Protobuf schema | Already exists (onnx.in.proto) | Not expressive enough for constraints |

### 7.4 Bootstrap: Generate YAML from Existing onnx

One-time Python script to extract all op schemas from `onnx.defs` → YAML:

```python
# scripts/export_schemas.py
import onnx
from onnx import defs
for schema in defs.get_all_schemas_with_history():
    write_yaml(schema)  # domain, name, since_version, inputs, outputs, attrs, type_constraints
```

After bootstrap, YAML files are the source of truth. New ops added as YAML.

---

## 8. Checker / Validator

### 8.1 Extensible Validation Architecture

```rust
/// A validation rule that checks one aspect of a model.
pub trait ValidationRule: Send + Sync {
    /// Unique identifier for this rule (for enable/disable).
    fn id(&self) -> &str;

    /// Severity level.
    fn severity(&self) -> Severity;

    /// Run the check. Return violations found.
    fn check(&self, model: &Model, ctx: &ValidationContext) -> Vec<Violation>;
}

pub enum Severity {
    Error,    // Model is invalid per ONNX spec
    Warning,  // Likely mistake but technically valid
    Info,     // Suggestion / best practice
}

pub struct Violation {
    pub rule_id: String,
    pub severity: Severity,
    pub message: String,
    pub location: ViolationLocation,
}

pub enum ViolationLocation {
    Model,
    Graph { graph_name: String },
    Node { graph_name: String, node_name: String, node_index: usize },
    Value { graph_name: String, value_name: String },
    Initializer { name: String },
}
```

### 8.2 Built-in Rules

```rust
/// The standard checker — all ONNX spec-mandated checks.
pub struct OnnxChecker {
    rules: Vec<Box<dyn ValidationRule>>,
    disabled: HashSet<String>,
}

impl OnnxChecker {
    pub fn new() -> Self {
        let mut checker = Self::empty();
        // Structural rules
        checker.add_rule(GraphAcyclicRule);
        checker.add_rule(UniqueValueNamesRule);
        checker.add_rule(InputOutputDeclaredRule);
        checker.add_rule(NoUnconnectedNodesRule);

        // Op-level rules (driven by OpSchema registry)
        checker.add_rule(OpExistsInDomainRule);
        checker.add_rule(InputCountMatchesSchemaRule);
        checker.add_rule(OutputCountMatchesSchemaRule);
        checker.add_rule(AttributeTypeMatchesSchemaRule);
        checker.add_rule(RequiredAttributePresentRule);
        checker.add_rule(TypeConstraintSatisfiedRule);

        // Type rules
        checker.add_rule(InitializerTypeMatchesDeclaredRule);
        checker.add_rule(DataTypeValidRule);

        // IR rules
        checker.add_rule(IrVersionSupportedRule);
        checker.add_rule(OpsetImportPresentRule);

        checker
    }

    /// Run all enabled rules.
    pub fn check(&self, model: &Model) -> ValidationResult;

    /// Disable a specific rule by id.
    pub fn disable_rule(&mut self, rule_id: &str);

    /// Add a custom rule.
    pub fn add_rule<R: ValidationRule + 'static>(&mut self, rule: R);
}

pub struct ValidationResult {
    pub violations: Vec<Violation>,
    pub errors: usize,
    pub warnings: usize,
}

impl ValidationResult {
    pub fn is_valid(&self) -> bool { self.errors == 0 }
}
```

Implementation status: the checker now has **9 rules**, including the expanded
structural `InputOutputDeclaredRule` and `NoUnconnectedNodesRule`, schema-driven
`TypeConstraintSatisfiedRule`, `InitializerTypeMatchesDeclaredRule`, and the
lower-bound-only `IrVersionSupportedRule`. The IR rule deliberately has **no
artificial upper-version ceiling**. Operator existence, arity, and attribute
checks remain grouped in `SchemaNodeConformsRule`.

### 8.3 Extensibility

Users register custom rules for custom ops or organizational standards:

```rust
// Example: custom rule for a company's model conventions
struct MaxGraphDepthRule { max_depth: usize }

impl ValidationRule for MaxGraphDepthRule {
    fn id(&self) -> &str { "custom.max_graph_depth" }
    fn severity(&self) -> Severity { Severity::Warning }
    fn check(&self, model: &Model, _ctx: &ValidationContext) -> Vec<Violation> {
        // Check nesting depth of subgraphs
        todo!()
    }
}

// Register it
checker.add_rule(MaxGraphDepthRule { max_depth: 5 });
```

---

## 9. Shape Inference ✅ Implemented

### 9.1 Delegates to Existing Crate

Shape inference reuses `onnx-runtime-shape-inference` (already being built for nxrt).
onnx-rs wraps it with a higher-level API:

```rust
/// Run shape inference on a model, populating all value type/shape info.
pub fn infer_shapes(model: &mut Model) -> Result<ShapeInferenceResult, ShapeError> {
    let registry = InferenceRegistry::default_registry();
    infer_shapes_with_registry(model, &registry)
}

pub struct ShapeInferenceResult {
    /// Values whose shapes were inferred.
    pub inferred: usize,
    /// Values whose shapes could not be determined (dynamic, unknown op, etc.).
    pub unknown: usize,
    /// Non-fatal diagnostics (currently empty; unresolved values count as unknown).
    pub warnings: Vec<String>,
}
```

### 9.2 Custom Op Shape Inference

Custom ops register the underlying crate's opset-aware `InferenceFn` on an
`InferenceRegistry`, then call `infer_shapes_with_registry`:

```rust
/// Register a shape inference handler for a custom op.
pub fn register_shape_inference(
    registry: &mut InferenceRegistry,
    domain: &str,
    op_type: &str,
    min_opset: u64,
    handler: InferenceFn,
);
```

---

## 10. Version Converter

**Status: ✅ Implemented (initial framework and built-ins).** `onnx-rs::version` provides
transactional recursive conversion, custom adapter registration, Reshape v5/v13→v14
`allowzero` rewriting, and schema-backed compatible bumps only where an explicit schema
upper bound proves compatibility. Unsupported downgrades and unproven targets reject without
mutating the model.

### 10.1 Extensible Conversion Framework

Converts models between opset versions. Each conversion is a standalone adapter.

```rust
/// Converts a model from one opset version to another.
pub struct VersionConverter {
    adapters: HashMap<(String, u32, u32), Box<dyn OpAdapter>>,
}

/// Converts one op from version A to version B.
pub trait OpAdapter: Send + Sync {
    /// Source (domain, op, version).
    fn source(&self) -> (&str, &str, u32);
    /// Target version.
    fn target_version(&self) -> u32;
    /// Convert: may rewrite the node, add helper nodes, or change attributes.
    fn adapt(&self, node: &Node, graph: &mut Graph) -> Result<AdaptResult, ConvertError>;
}

pub enum AdaptResult {
    /// Node is compatible as-is (just bump version).
    Compatible,
    /// Node was rewritten in-place.
    Rewritten,
    /// Node was replaced by a subgraph (decomposed).
    Decomposed { replacement_nodes: Vec<Node> },
    /// Cannot convert — op semantics changed in incompatible way.
    Incompatible { reason: String },
}

impl VersionConverter {
    pub fn new() -> Self {
        let mut converter = Self::empty();
        // Register built-in adapters
        converter.register(adapters::resize_v10_to_v11());
        converter.register(adapters::reshape_v5_to_v14());
        converter.register(adapters::batch_norm_v9_to_v15());
        // ... etc
        converter
    }

    /// Convert model to target opset version transactionally.
    pub fn convert(&self, model: &mut Model, target_opset: u32) -> Result<ConvertReport, ConvertError> {
        // Conversion operates on a clone, recursively including nested subgraphs.
        // The default-domain opset import changes only after every node is accepted.
        /* implementation */
    }

    /// Register a custom adapter (e.g., for custom domain ops).
    pub fn register<A: OpAdapter + 'static>(&mut self, adapter: A);

    /// List available conversions for an op.
    pub fn available_conversions(&self, domain: &str, op: &str) -> Vec<(u32, u32)>;
}

pub struct ConvertReport {
    pub ops_converted: usize,
    pub ops_unchanged: usize,
    pub ops_incompatible: Vec<IncompatibleOp>,
    pub source_opset: u32,
    pub target_opset: u32,
}
```

### 10.2 Adapter Registration

Adapters are standalone and composable. To go from opset 11 → 21, the converter
chains: 11→13 → 13→15 → 15→17 → 17→19 → 19→21. Each step is a small adapter.

**Implemented.** The initial built-in rewrite targets Reshape opset 5/13 → 14,
materializing `allowzero = 0`. This is a small, semantics-preserving attribute
rewrite that demonstrates mutable adapters without introducing a parallel IR.
Operators whose source and target resolve to the same schema `since_version`
use the registry's data-driven compatible-bump path.

Users can register adapters for custom domains:

```rust
// Custom domain version migration
converter.register(MyDomainResizeV1ToV2 { ... });
```

---

## 11. Custom Op Registration

### 11.1 Registry

```rust
/// Global registry for custom op definitions.
pub struct OpRegistry {
    /// Standard ops (populated from codegen'd YAML schemas).
    standard: HashMap<(String, String, u32), OpSchema>,
    /// User-registered custom ops.
    custom: HashMap<(String, String, u32), CustomOpDef>,
}

pub struct CustomOpDef {
    pub schema: OpSchema,
    pub shape_inference: Option<Box<dyn ShapeInferenceHandler>>,
    pub version_adapters: Vec<Box<dyn OpAdapter>>,
    pub checker_rules: Vec<Box<dyn ValidationRule>>,
}

impl OpRegistry {
    /// Register a custom op with full metadata.
    pub fn register_custom_op(&mut self, def: CustomOpDef);

    /// Register from YAML file (same format as standard ops).
    pub fn register_from_yaml(&mut self, yaml_path: &Path) -> Result<(), ParseError>;

    /// Register an entire custom domain from a directory of YAML files.
    pub fn register_domain(&mut self, domain: &str, yaml_dir: &Path) -> Result<(), ParseError>;

    /// Look up schema for an op.
    pub fn get_schema(&self, domain: &str, op_type: &str, opset: u32) -> Option<&OpSchema>;
}
```

### 11.2 Custom Op YAML

Same format as standard ops — just drop a YAML file:

```yaml
# schemas/custom/my_domain/my_op.yaml
domain: "com.mycompany"
name: MyCustomOp
since_version: 1
doc: "My custom operation."

inputs:
  - name: X
    type_str: "T"

outputs:
  - name: Y
    type_str: "T"

attributes:
  - name: mode
    type: string
    required: true

type_constraints:
  - type_param: "T"
    allowed: [float32, float16]

shape_inference: identity  # output shape = input shape
```

---

## 12. Python Bindings

### 12.1 ABI3 Stable ABI

All Python bindings use **PyO3 abi3 mode** targeting the stable limited API. This means:
- One `.so`/`.pyd` per platform, works across Python 3.9+
- No per-Python-version wheel builds
- Future Python versions work without recompilation

```toml
# onnx-rs-python/Cargo.toml
[dependencies]
pyo3 = { version = "0.23", features = ["abi3-py39", "extension-module"] }
```

### 12.2 Extensible Traits — Python Interop

All extensible traits (`ModelFormat`, `WeightBackend`, `ValidationRule`, `OpAdapter`,
`ShapeInferenceHandler`) are implementable from Python. Users can subclass in Python
and register with the Rust core — the Rust side calls back into Python via PyO3.

#### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Python user code                       │
│   class GgufFormat(onnx_rs.ModelFormat): ...             │
│   onnx_rs.register_format(GgufFormat())                  │
├─────────────────────────────────────────────────────────┤
│                   PyO3 bridge layer                       │
│   PyModelFormat wraps Python object, implements          │
│   Rust ModelFormat trait, calls back via GIL             │
├─────────────────────────────────────────────────────────┤
│                   Rust core (FormatRegistry)              │
│   Stores Box<dyn ModelFormat> — doesn't know if it's     │
│   Rust-native or Python-backed                           │
└─────────────────────────────────────────────────────────┘
```

#### Bridge Pattern (ModelFormat example)

```rust
// Rust side: bridge that wraps a Python object implementing ModelFormat
#[pyclass]
struct PyModelFormat {
    inner: PyObject,  // Python object with load/save/can_load methods
}

impl ModelFormat for PyModelFormat {
    fn id(&self) -> &str {
        // Cache id string to avoid GIL on every call
        &self.cached_id
    }

    fn extensions(&self) -> &[&str] {
        &self.cached_extensions
    }

    fn can_load(&self, path: &Path) -> bool {
        Python::with_gil(|py| {
            self.inner
                .call_method1(py, "can_load", (path.to_str().unwrap(),))
                .and_then(|r| r.extract::<bool>(py))
                .unwrap_or(false)
        })
    }

    fn load(&self, path: &Path, opts: &LoadOptions) -> Result<Model, LoadError> {
        Python::with_gil(|py| {
            let py_model = self.inner
                .call_method1(py, "load", (path.to_str().unwrap(),))?;
            // Convert Python Model repr → Rust Model
            extract_model(py, py_model)
        })
    }

    fn save(&self, model: &Model, path: &Path, opts: &SaveOptions) -> Result<(), SaveError> {
        Python::with_gil(|py| {
            let py_model = model_to_py(py, model)?;
            self.inner.call_method1(py, "save", (py_model, path.to_str().unwrap()))?;
            Ok(())
        })
    }
}
```

#### Python User API

```python
import onnx_rs

# --- Custom model format ---
class GgufFormat(onnx_rs.ModelFormat):
    def id(self) -> str:
        return "gguf"

    def extensions(self) -> list[str]:
        return ["gguf"]

    def can_load(self, path: str) -> bool:
        with open(path, "rb") as f:
            return f.read(4) == b"GGUF"

    def load(self, path: str) -> onnx_rs.Model:
        # Parse GGUF → construct onnx_rs.Model
        ...

    def save(self, model: onnx_rs.Model, path: str):
        raise NotImplementedError("GGUF save not supported")

onnx_rs.register_format(GgufFormat())

# --- Custom validation rule ---
class MaxNodeCount(onnx_rs.ValidationRule):
    def __init__(self, max_nodes: int = 10000):
        self.max_nodes = max_nodes

    def id(self) -> str:
        return "custom.max_node_count"

    def severity(self) -> str:
        return "warning"  # "error" | "warning" | "info"

    def check(self, model: onnx_rs.Model) -> list[onnx_rs.Violation]:
        count = model.graph.node_count()
        if count > self.max_nodes:
            return [onnx_rs.Violation(
                rule_id=self.id(),
                severity=self.severity(),
                message=f"Graph has {count} nodes, exceeds max {self.max_nodes}",
                location="graph",
            )]
        return []

onnx_rs.checker.add_rule(MaxNodeCount(5000))

# --- Custom weight backend ---
class RedisWeightBackend(onnx_rs.WeightBackend):
    def __init__(self, redis_url: str):
        self.client = redis.Redis.from_url(redis_url)

    def id(self) -> str:
        return "redis"

    def get(self, tensor_name: str) -> bytes | None:
        return self.client.get(f"weight:{tensor_name}")

    def put(self, tensor_name: str, data: bytes, dtype: str, shape: list[int]):
        self.client.set(f"weight:{tensor_name}", data)

    def tensor_names(self) -> list[str]:
        return [k.decode().removeprefix("weight:")
                for k in self.client.keys("weight:*")]

onnx_rs.register_weight_backend(RedisWeightBackend("redis://localhost:6379"))

# --- Custom version converter adapter ---
class MyOpV1ToV2(onnx_rs.OpAdapter):
    def source(self) -> tuple[str, str, int]:
        return ("com.mycompany", "MyOp", 1)

    def target_version(self) -> int:
        return 2

    def adapt(self, node: onnx_rs.Node, graph: onnx_rs.Graph) -> str:
        # Rewrite node attributes for v2 semantics
        node.set_attr("new_attr", node.get_attr("old_attr", 0))
        node.remove_attr("old_attr")
        return "rewritten"  # "compatible" | "rewritten" | "incompatible"

onnx_rs.version_converter.register(MyOpV1ToV2())

# --- Custom shape inference ---
class MyOpShapeInference(onnx_rs.ShapeInferenceHandler):
    def infer(self, node: onnx_rs.Node, input_types: list) -> list:
        # Output shape = input shape
        return [input_types[0]] if input_types else []

onnx_rs.register_shape_inference("com.mycompany", "MyOp", MyOpShapeInference())
```

#### Bridge Traits Summary

Every extensible Rust trait has a corresponding Python base class and PyO3 bridge:

| Rust Trait | Python Base Class | PyO3 Bridge Struct | GIL Strategy |
|-----------|-------------------|-------------------|---------------|
| `ModelFormat` | `onnx_rs.ModelFormat` | `PyModelFormat` | Cache id/extensions; GIL on load/save |
| `WeightBackend` | `onnx_rs.WeightBackend` | `PyWeightBackend` | GIL on get/put |
| `ValidationRule` | `onnx_rs.ValidationRule` | `PyValidationRule` | Cache id/severity; GIL on check |
| `OpAdapter` | `onnx_rs.OpAdapter` | `PyOpAdapter` | Cache source/target; GIL on adapt |
| `ShapeInferenceHandler` | `onnx_rs.ShapeInferenceHandler` | `PyShapeInferenceHandler` | GIL on infer |

#### Performance Considerations

- **Hot path (load/inference):** Rust-native implementations avoid GIL entirely
- **Cold path (registration):** Python callbacks cached where possible (id, extensions, severity)
- **GIL strategy:** Acquire only when calling into Python. Release GIL during Rust-internal work
- **Buffer protocol:** Weight data passed as `bytes` / `memoryview` — zero-copy when possible
- **abi3 constraint:** No `PyList::get_item_unchecked` or version-specific optimizations.
  All PyO3 calls use stable limited API only

#### Thread Safety

Python-backed trait objects are `Send + Sync` by wrapping in `PyObject` (which is
`Send`). Thread safety is guaranteed by GIL acquisition on every callback. This means:
- Python-backed formats work correctly in multi-threaded Rust code
- Performance degrades under contention (GIL bottleneck)
- For high-throughput paths, users should prefer Rust-native implementations

### 12.3 Drop-in Compatibility Goal

```python
import onnx_rs as onnx  # Drop-in for common use cases

# Load / save (including custom-registered formats)
model = onnx.load("model.onnx")
model = onnx.load("model.gguf")  # works if GgufFormat registered
onnx.save(model, "output.onnx")

# Check (including custom rules)
onnx.checker.check_model(model)

# Shape inference
model = onnx.shape_inference.infer_shapes(model)

# Text format
text = onnx.printer.to_text(model)
model = onnx.parser.from_text(text)

# Version convert
model = onnx.version_converter.convert_version(model, target_version=21)

# Custom ops (YAML registration)
onnx.registry.register_from_yaml("my_ops/")
```

### 12.4 Not a Full Drop-in

Things intentionally NOT replicated:
- `onnx.helper.make_*()` — use onnxscript or direct IR construction instead
- `onnx.numpy_helper` — use `onnx_rs.tensor_to_numpy()` / DLPack
- `onnx.compose` — if needed, add as separate feature
- TensorProto direct manipulation — use IR API

---

## 13. Crate Structure

```
onnx-rs/                          # Workspace root
├── onnx-ir/                      # Core IR (shared with nxrt via feature flags)
│   ├── src/
│   │   ├── graph.rs
│   │   ├── node.rs
│   │   ├── value.rs
│   │   ├── tensor.rs
│   │   └── ...
│   └── Cargo.toml
├── onnx-proto/                   # Protobuf serde (prost-generated + weight layer)
│   ├── build.rs                  # prost codegen from onnx.proto3
│   ├── src/
│   │   ├── loader.rs
│   │   ├── saver.rs
│   │   ├── weight_ref.rs
│   │   ├── safetensors.rs
│   │   └── external_data.rs
│   └── Cargo.toml
├── onnx-schema/                  # Op schema registry (YAML + codegen)
│   ├── schemas/
│   │   ├── standard/             # One YAML per op
│   │   ├── contrib/
│   │   └── training/
│   ├── build.rs                  # YAML → Rust codegen
│   ├── src/
│   │   ├── generated/            # codegen output
│   │   ├── registry.rs
│   │   └── custom.rs
│   └── Cargo.toml
├── onnx-checker/                 # Validation / checker
│   ├── src/
│   │   ├── rules/                # Built-in validation rules
│   │   ├── checker.rs
│   │   └── result.rs
│   └── Cargo.toml
├── onnx-text/                    # Textual representation (.onnxtxt)
│   ├── src/
│   │   ├── parser.rs
│   │   ├── printer.rs
│   │   └── format.rs
│   └── Cargo.toml
├── onnx-json/                    # JSON / TextProto support
│   └── ...
├── onnx-version-converter/       # Version conversion
│   ├── src/
│   │   ├── adapters/             # Per-op version adapters
│   │   ├── converter.rs
│   │   └── chain.rs
│   └── Cargo.toml
├── onnx-rs/                      # Umbrella crate (re-exports everything)
│   ├── src/lib.rs
│   └── Cargo.toml
├── onnx-rs-python/               # Python bindings (PyO3)
│   ├── src/lib.rs
│   └── pyproject.toml
└── onnx-cli/                     # CLI tool
    ├── src/main.rs               # onnx check/convert/print/info/diff
    └── Cargo.toml
```

---

## 14. Relationship to nxrt

```
                   ┌───────────────────────┐
                   │       onnx-rs         │
                   │  (standard library)   │
                   │                       │
                   │  load/save/check/     │
                   │  convert/print        │
                   └───────────┬───────────┘
                               │
                       depends on (IR)
                               │
                   ┌───────────▼───────────┐
                   │       onnx-ir         │
                   │  (shared IR crate)    │
                   │                       │
                   │  Graph/Node/Value     │
                   └───────────┬───────────┘
                               │
                       depends on (IR)
                               │
                   ┌───────────▼───────────┐
                   │        nxrt           │
                   │  (inference runtime)  │
                   │                       │
                   │  execute/optimize/    │
                   │  schedule/serve       │
                   └───────────────────────┘
```

- `onnx-ir` is the shared dependency — both onnx-rs and nxrt use the same IR types
- nxrt's loader can delegate to `onnx-proto` for parsing
- nxrt's shape inference IS `onnx-runtime-shape-inference`, which onnx-rs wraps
- nxrt adds runtime-specific layers (layout, device, memory) on top of the shared IR
- onnx-rs never depends on nxrt; nxrt may optionally depend on onnx-rs components

---

## 15. Phased Roadmap

### Phase 1: Core I/O + IR (Foundation)

- [ ] `onnx-proto`: prost codegen from onnx.proto3
- [ ] `onnx-proto`: ModelLoader with mmap + external data support
- [ ] `onnx-proto`: Safetensors weight backend
- [ ] `onnx-proto`: ModelSaver with WeightStrategy
- [ ] `onnx-ir`: Factor out shared IR from `onnx-runtime-ir` (or decide to keep one crate)
- [ ] `onnx-rs`: Umbrella crate, `onnx_rs::load()` / `onnx_rs::save()`
- [ ] Basic round-trip test: load .onnx → save .onnx → binary identical

### Phase 2: Schema + Checker

- [ ] `onnx-schema`: Bootstrap YAML from `onnx.defs` (Python export script)
- [ ] `onnx-schema`: codegen pipeline (build.rs)
- [ ] `onnx-schema`: OpRegistry with lookup
- [ ] `onnx-checker`: Core structural rules
- [ ] `onnx-checker`: Op-level validation (driven by schema)
- [ ] `onnx-checker`: Custom rule registration
- [ ] Pass: `onnx.checker.check_model()` equivalent on test suite

### Phase 3: Text Format + JSON

- [ ] `onnx-text`: Printer (Model → .onnxtxt)
- [ ] `onnx-text`: Parser (.onnxtxt → Model)
- [ ] `onnx-text`: Round-trip tests
- [ ] `onnx-json`: JSON serialization
- [x] `onnx-rs::textproto`: protobuf TextFormat support
- [ ] `onnx-cli`: `onnx check`, `onnx print`, `onnx info` commands

### Phase 4: Version Converter + Polish

- [x] `onnx-version-converter`: Framework + adapter registration
- [x] `onnx-version-converter`: Initial core adapter (Reshape v5/v13 → v14) + schema-compatible bumps
- [x] Custom op registration API
- [ ] Shape inference integration (wrap `onnx-runtime-shape-inference`)
- [ ] `onnx-rs-python`: PyO3 bindings
- [ ] `onnx-cli`: `onnx convert`, `onnx diff` commands
- [ ] Benchmark: load time vs Python `onnx.load()` on large models
- [ ] crates.io publish

---

## 16. Open Questions

1. **Repo location:** Separate repo (`onnx-rs/onnx-rs`) or inside `onnx-genai`? Separate
   is cleaner (it's a standard library, not a runtime), but mono-repo is easier during
   early development.

2. **`onnx-ir` ownership:** Does `onnx-rs` own the IR crate, or does `nxrt`? Or is it a
   standalone crate both depend on? Affects publish order and semver coordination.

3. **YAML schema maintenance:** Who keeps YAML files in sync with upstream ONNX spec
   releases? Automated CI job that diffs `onnx.defs` against YAML and flags drift?

4. **Compatibility with `onnx` Python package:** Can `onnx-rs-python` load models saved
   by `onnx` and vice versa? What's the protobuf wire compatibility story?

5. **ONNX spec conformance tests:** Does ONNX have an official checker test suite? If so,
   run it against `onnx-checker` for compliance.

6. **Training info support:** How much training-specific ONNX spec to support? Training
   graphs, gradient ops, etc. Out of scope for v1?

7. **Text format standardization:** Should .onnxtxt be proposed as an ONNX standard
   format? Or keep it as an onnx-rs extension?

8. **Performance target:** What's the target for `onnx_rs.load()` on a 70B model? Current
   Python `onnx.load()` takes 30-60s and peaks at 2x model size in memory. Target: <5s,
   constant memory (mmap).
