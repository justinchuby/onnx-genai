# Extensible Quantization Type System

**Status:** design draft  
**Authors:** Justin Chu  
**Date:** 2025-07-22

## 1. Motivation

Current ONNX represents quantization via QuantizeLinear/DequantizeLinear (QDQ) operators
with a fixed set of recognized data types. Every new quantized format — MXFP4, IQ2_S,
ternary 1.58-bit, vendor-specific NF4 variants — requires explicit addition to the ONNX
spec and new QDQ op versions.

Meanwhile, runtimes like llama.cpp support 20+ quantization formats through a simple
codec pattern: each type is a struct defining block layout + dequant function. Adding a
new type requires zero spec changes.

This design introduces a **pluggable type system** that:
1. Lets models declare custom quantized types without spec amendments
2. Lets EPs provide native kernels for types they optimize
3. Guarantees every model can still run (fallback through dequantization)
4. Supports both weight quantization (static, in-model) and activation quantization (dynamic/static)

## 2. Design Overview

```
┌────────────────────────────────────────────────────────────────┐
│  Model File (ONNX)                                             │
│                                                                │
│  QuantTypeDecl[] — structural type declarations                │
│  TensorProto.quant_type_uri — "this tensor uses type X"        │
│  QuantizedEdge[] — activation quant policies (static only)     │
└────────────────────────────────┬───────────────────────────────┘
                                 │ load
                                 ▼
┌────────────────────────────────────────────────────────────────┐
│  Runtime (onnx-genai)                                          │
│                                                                │
│  Codec Registry                                                │
│    ├── Built-in: int4, int8, fp8, mxfp4, ...                  │
│    ├── EP-provided: vendor types registered during EP init     │
│    └── User plugins: installed crates / .so / WASM             │
│                                                                │
│  Auto-Codec Generator                                          │
│    └── Derives naive codec from QuantTypeDecl when no plugin   │
│                                                                │
│  Dispatch Chain                                                │
│    └── EP Native → EP Dequant → Streaming Fallback Dequant     │
└────────────────────────────────────────────────────────────────┘
```

## 3. Type Declaration (Model-Side)

### 3.1 QuantTypeDecl Schema

```protobuf
message QuantTypeDecl {
  // Unique identifier. Namespace rules: "onnx:" reserved for spec-blessed types,
  // "vendor:<name>:" for vendor-specific, anything else is community.
  string type_uri = 1;

  // === Structural descriptor (required) ===
  int32 block_size = 2;          // logical elements per block
  int32 bytes_per_block = 3;     // storage bytes per block

  // === Encoding descriptor (required) ===
  EncodingDescriptor encoding = 4;

  // === Composition (optional) ===
  ScalarType scale_type = 5;     // type of per-group scale values
  ScalarType zero_point_type = 6;
  int32 group_size = 7;          // elements sharing one scale (0 = per-tensor)
  PaddingMode padding_mode = 8;  // behavior when tensor dim % group_size != 0

  // === Dequant specification ===
  DequantFormula formula = 9;    // canonical formula with explicit cast points
  bytes test_vector_input = 10;  // reference: packed block bytes
  bytes test_vector_output = 11; // reference: expected f32 values (IEEE 754)
  int32 test_vector_count = 12;  // number of elements in test vector

  // === Metadata ===
  string description = 13;
  string version = 14;           // semver, must bump on semantic change
}

enum EncodingFamily {
  AFFINE = 0;           // (q - zp) * scale
  SYMMETRIC = 1;        // q * scale
  LOOKUP_TABLE = 2;     // codebook[index] * scale
  PACKED_INTEGER = 3;   // base-N packing (ternary, quinary, etc.)
  LOGARITHMIC = 4;      // sign * scale * base^exponent
  CUSTOM = 15;          // requires codec plugin, no auto-generation
}

message EncodingDescriptor {
  EncodingFamily family = 1;

  // Family-specific fields
  int32 packing_base = 2;       // for PACKED_INTEGER: base of the encoding (3 for ternary)
  int32 packing_radix = 3;      // elements packed per storage unit
  BitOrder bit_order = 4;       // LSB_FIRST or MSB_FIRST
  repeated float codebook = 5;  // for LOOKUP_TABLE: the fixed codebook values
  float value_offset = 6;       // additive offset applied after decode (e.g., -1 for {0,1,2}→{-1,0,1})
}

message DequantFormula {
  // Canonical formula expressed as ordered steps with explicit intermediate types.
  // Example for ternary: unpack(base3) → add(-1.0) → cast(f16) → multiply(scale)
  repeated DequantStep steps = 1;
}

message DequantStep {
  enum Op {
    UNPACK = 0;    // decode packed representation to logical integers
    ADD = 1;       // add constant (value_offset)
    MULTIPLY = 2;  // multiply by scale
    CAST = 3;      // cast to target type
    LOOKUP = 4;    // index into codebook
    SUBTRACT = 5;  // subtract zero_point
  }
  Op op = 1;
  ScalarType cast_to = 2;       // for CAST
  float constant = 3;           // for ADD/MULTIPLY with constant
  string operand = 4;           // "scale" | "zero_point" | "codebook"
}

enum PaddingMode {
  ERROR = 0;          // reject if dimension not divisible
  ZERO_PAD = 1;       // pad partial group with zeros
  REPEAT_LAST = 2;    // repeat last value to fill
}
```

### 3.2 TensorProto Extension

```protobuf
message TensorProto {
  // ... existing fields ...

  // If set, raw_data contains packed quantized bytes interpreted by the
  // referenced QuantTypeDecl. data_type field is set to UNDEFINED.
  string quant_type_uri = 20;
}
```

### 3.3 Model IR Version Gating

Models using `quant_type_uri` MUST set `ir_version >= N` (TBD). Runtimes not supporting
extensible types will reject the model with a clear error rather than misinterpret data.

## 4. Type Implementation (Runtime-Side)

### 4.1 Codec Trait

```rust
/// A codec that can dequantize (and optionally quantize) a custom type.
pub trait QuantCodec: Send + Sync + 'static {
    /// Unique type URI this codec handles.
    fn type_uri(&self) -> &str;

    /// Dequantize one block of packed bytes into f16 values.
    /// `src` has exactly `bytes_per_block` bytes.
    /// `dst` has exactly `block_size` elements.
    fn dequantize_block(&self, src: &[u8], scale: f32, zero_point: f32, dst: &mut [f16]);

    /// Optional: quantize f16 values into packed bytes.
    fn quantize_block(&self, src: &[f16], dst: &mut [u8]) -> Option<(f32, f32)> {
        None // default: quantization not supported
    }

    /// Validate codec against declaration's test vectors.
    /// Runtime calls this once at registration time.
    fn validate(&self, decl: &QuantTypeDecl) -> Result<(), CodecValidationError>;
}
```

### 4.2 Codec Registry

```rust
pub struct CodecRegistry {
    codecs: HashMap<String, Arc<dyn QuantCodec>>,
    auto_generator: AutoCodecGenerator,
}

impl CodecRegistry {
    /// Register a codec. Validates against known declarations.
    pub fn register(&mut self, codec: Arc<dyn QuantCodec>) -> Result<()>;

    /// Resolve codec for a type_uri. Falls back to auto-generation if:
    /// 1. No registered codec exists
    /// 2. Declaration's encoding family is not CUSTOM
    /// 3. Declaration has valid test vectors for verification
    pub fn resolve(&self, decl: &QuantTypeDecl) -> Result<Arc<dyn QuantCodec>>;
}
```

### 4.3 Auto-Codec Generation

For types with `encoding.family != CUSTOM`, the runtime can derive a correct (but
potentially slow) codec from the structural declaration + DequantFormula:

```rust
impl AutoCodecGenerator {
    pub fn generate(&self, decl: &QuantTypeDecl) -> Result<Arc<dyn QuantCodec>> {
        // 1. Parse DequantFormula steps into an interpreter
        // 2. Build a generic block decoder
        // 3. Validate against test vectors (MUST pass or reject)
        let codec = InterpretedCodec::from_formula(&decl.formula)?;

        // Verify correctness
        let output = codec.dequantize_block(&decl.test_vector_input, ...);
        assert_vectors_match(&output, &decl.test_vector_output, TOLERANCE)?;

        Ok(Arc::new(codec))
    }
}
```

**Constraints on auto-generated codecs:**
- Constant-time execution (no data-dependent branching) for production safety
- Streaming dequant (per-block, not materializing full tensor) to avoid OOM
- Clearly marked as "fallback" in profiling/logging

## 5. EP Negotiation

### 5.1 EP Interface

```rust
pub enum KernelMatch {
    /// EP has a native kernel. Pass raw packed bytes directly.
    Native,
    /// EP can compute if runtime dequantizes to the specified type first.
    Dequant { target: ScalarType },
    /// EP cannot handle this type at all.
    Unsupported,
}

pub trait ExecutionProvider {
    /// Per-tensor negotiation: can this EP handle the given quantized type?
    fn supports_quant_type(
        &self,
        type_uri: &str,
        layout: &LayoutDescriptor,
    ) -> KernelMatch;

    /// Subgraph-level claim: EP can execute this entire subgraph in quantized
    /// domain without intermediate dequant.
    fn claim_quantized_subgraph(
        &self,
        subgraph: &SubgraphView,
        types: &[&QuantTypeDecl],
    ) -> Option<FusedKernelHandle>;

    /// Register EP-provided codecs into the runtime registry.
    fn register_codecs(&self, registry: &mut CodecRegistry);
}
```

### 5.2 Priority Resolution

When multiple EPs claim support for the same type:

1. **User-specified preference** — explicit EP priority list in runtime config
2. **Subgraph claim > per-tensor** — fused execution preferred over individual ops
3. **Native > Dequant** — avoid unnecessary conversion
4. **Registration order** as final tiebreaker

### 5.3 Correctness Contract

An EP claiming `KernelMatch::Native` MUST produce results within documented numerical
tolerance of the canonical path (dequant → f32/f16 → compute). Runtime MAY verify this
with test inputs during EP registration.

## 6. Dispatch Chain

```
Model loads → for each quantized tensor:
  1. Resolve QuantTypeDecl from model
  2. Check CodecRegistry for matching codec
     → Found: use it
     → Not found + family != CUSTOM: auto-generate, validate against test vectors
     → Not found + family == CUSTOM: error("install codec plugin: {uri}")
  3. Query EPs via supports_quant_type()
     → Native: pass raw bytes to EP kernel
     → Dequant: runtime streams dequant per-block → EP computes on target type
     → All Unsupported: streaming fallback dequant → default EP on f16

Memory safety: fallback dequant is STREAMING (per-block on demand).
Never materialize full dequantized tensor unless EP explicitly requests it.
```

## 7. Activation Quantization

### 7.1 Dynamic (No Model Annotation)

EP internally decides to quantize activations at runtime. The model and runtime
framework are not involved — this is a pure EP optimization.

```rust
// Inside EP's matmul implementation:
fn execute_matmul(&self, input: &Tensor, weight: &QuantTensor) -> Tensor {
    // EP's choice: dynamic int8 quantization of input
    let (input_q, scale) = self.dynamic_quantize_per_token(input);
    self.int8_matmul_kernel(input_q, scale, weight)
}
```

### 7.2 Static (Model Annotation)

Pre-calibrated scale/zero-point stored as edge metadata:

```protobuf
message ActivationQuantPolicy {
  string type_uri = 1;              // quantization type for this edge
  Granularity granularity = 2;      // PER_TENSOR | PER_CHANNEL | PER_TOKEN

  // Pre-calibrated parameters
  TensorProto scale = 3;
  TensorProto zero_point = 4;

  // Which edge this applies to
  string producer_node = 5;
  string producer_output = 6;
  string consumer_node = 7;
  string consumer_input = 8;
}

enum Granularity {
  PER_TENSOR = 0;
  PER_CHANNEL = 1;
  PER_TOKEN = 2;
}
```

Runtime applies quant/dequant around the annotated edge, or passes the policy to
an EP that can execute the quantized subgraph natively.

### 7.3 Quantized Subgraph

EP claims an entire subgraph via `claim_quantized_subgraph()`. All tensors within
the claimed region remain in quantized domain. Only boundary edges (inputs/outputs
of the subgraph) go through quant/dequant.

### 7.4 Per-Token Dynamic Quant with Dynamic Shapes

For `granularity = PER_TOKEN` with dynamic batch/sequence dimensions:
- Scale tensor shape is determined at runtime (one scale per token)
- EP MUST handle variable-length scale tensors
- Runtime hints `dynamic_quant_overhead: {low | medium | high}` in metadata

## 8. Interaction with Graph Optimization

### 8.1 Quantized Tensors in Optimizers

Graph optimization passes see quantized tensors as opaque:
- Constant folding: MUST NOT fold through quantized weights (would require dequant)
- Fusion: fusible only if EP claims the fused pattern via `claim_quantized_subgraph`
- Shape inference: uses `block_size` and tensor dims, ignores packed layout

### 8.2 QDQ Compatibility Layer

For mixed models containing both legacy QDQ nodes and new QuantTypeDecl tensors:
- Both representations are valid in the same model
- Optimizer pass `ConvertQDQToExtensible` can lower QDQ patterns to QuantTypeDecl
  (opt-in, for models that want to migrate)
- No forced migration: QDQ remains valid indefinitely

## 9. Type URI Governance

### 9.1 Namespace Rules

| Prefix | Owner | Registration |
|--------|-------|-------------|
| `onnx:` | ONNX SIG | Requires spec PR |
| `ms:` | Microsoft | Internal |
| `vendor:<name>:` | Named vendor | Self-serve with IANA-style registry |
| (no prefix) | Community | First-come-first-serve, no guarantees |

### 9.2 Versioning

- `type_uri` includes version: `"onnx:mxfp4-block32/v1"`
- Semantic change (different decode behavior) MUST bump version
- Old version URIs remain valid forever (append-only registry)
- Runtime resolves exact version match; no implicit upgrades

## 10. Security

### 10.1 Plugin Loading

- Codec plugins are **opt-in allowlisted** in runtime config
- Type URIs do NOT reference file paths or URLs directly
- Plugins require explicit user installation (`cargo add onnx-codec-ternary`)
- Signed plugin verification (optional, for enterprise deployments)

### 10.2 Auto-Generated Codecs

- Limited to known encoding families (no arbitrary code execution)
- Use constant-time arithmetic (no data-dependent branching)
- Cannot perform I/O, allocation beyond block buffers, or syscalls

### 10.3 Model Trust

- A model's `QuantTypeDecl` is pure data (no executable content)
- Worst case of a malicious declaration: runtime generates wrong results
  → mitigated by test vector validation
- Runtime SHOULD log a warning for unrecognized community-namespace types

## 11. Migration Path

### 11.1 For Existing ONNX Models (QDQ)

No change required. QDQ continues to work as-is.

### 11.2 For Model Converters

Tools like `onnxruntime_genai` model builder and llama.cpp GGUF converters can
emit QuantTypeDecl-based models. Conversion:

```
GGUF model → for each tensor:
  1. Map GGUF type_id to type_uri (e.g., "onnx-community:iq2_xs/v1")
  2. Copy raw packed bytes into TensorProto.raw_data
  3. Emit QuantTypeDecl with encoding descriptor + test vectors from GGUF spec
```

### 11.3 For Runtime Implementers

Minimum viable implementation:
1. Parse QuantTypeDecl from model
2. Implement auto-codec generator for families {AFFINE, SYMMETRIC, PACKED_INTEGER}
3. Dequant all weights to f16 at load time (simple, slow, correct)

Advanced implementation:
- Streaming dequant, EP negotiation, native kernels, plugin registry

## 12. Relationship to Existing `SUB4BIT_QUANT.md`

This design subsumes and generalizes the approach in `SUB4BIT_QUANT.md`. The IQ/MXFP4
types documented there become concrete instances:

| SUB4BIT_QUANT type | Extensible type_uri | Encoding family |
|---|---|---|
| IQ1_S | `onnx-community:iq1_s/v1` | PACKED_INTEGER (base-3 + grid) |
| IQ2_XS | `onnx-community:iq2_xs/v1` | LOOKUP_TABLE |
| IQ4_NL | `onnx-community:iq4_nl/v1` | LOOKUP_TABLE (16-entry codebook) |
| MXFP4 | `onnx:mxfp4-block32/v1` | AFFINE (microscaling) |

The `MatMulNBits`/`BlockQuantizedMatMul` ops remain valid as concrete accelerated paths.
EPs that recognize these ops continue to use them; the extensible type system provides
the *fallback* and *extensibility* story.

## 13. Open Questions

1. **Should test vectors be mandatory?** Currently proposed as required for auto-codec
   validation. Alternative: optional but auto-codec refuses without them.
2. **WASM codec format?** For portable plugins, WASM provides sandboxing. Worth standardizing?
3. **Calibration tooling:** Static activation quant needs calibration workflow. Out of scope
   for this doc but needs a companion design.
4. **Maximum block_size?** Should we cap it to prevent pathological declarations?

## References

- [llama.cpp ggml-quants.h](https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.h)
- [ONNX QuantizeLinear spec](https://onnx.ai/onnx/operators/onnx__QuantizeLinear.html)
- [MX (Microscaling) spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- This project's `SUB4BIT_QUANT.md` for IQ format details
