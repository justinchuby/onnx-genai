# `onnx-rs` ONNX IR v13 specification coverage

This audit binds `onnx-rs` to **ONNX v1.20.0 / IR version 13**. The vendored
schema declares IR 13 and the `UINT2`/`INT2` additions
([`onnx.proto3:115-125`](../crates/onnx-runtime-loader/proto/onnx.proto3#L115-L125)).
The authoritative comparison inputs are:

- `onnx/onnx.proto` and `onnx/onnx.proto3` at tag `v1.20.0`;
- `onnx/onnx-ml.proto` and `onnx/onnx-ml.proto3` at tag `v1.20.0`;
- the normative multi-device section of `docs/IR.md` at tag `v1.20.0`.

The vendored proto has the same standard-ONNX fields and numbers as v1.20.0
(with proto3 presence semantics and updated IR-13 publication comments).
The bound schema includes the ONNX-ML type delta:
`TypeProto.Opaque { domain = 1; name = 2; }` and `opaque_type = 7`.

## Legend and implementation basis

- ✅: covered and round-tripped.
- ⚠️: losslessly preserved, but only through the retained source proto or the
  readable Text extension; validation/editing/runtime projection is incomplete.
- ❌: absent or not validated.
- N/A: the specification defines no additional validation rule.

All standard-ONNX wire, TextProto, and JSON ✅ entries below come from one
generated descriptor
([`proto_serde.rs:11-33`](../crates/onnx-rs/src/proto_serde.rs#L11-L33),
[`json/mod.rs:11-24`](../crates/onnx-rs/src/json/mod.rs#L11-L24),
[`textproto/mod.rs:10-23`](../crates/onnx-rs/src/textproto/mod.rs#L10-L23)).
`Model` retains the exact decoded proto and returns it unchanged
([`model.rs:58-62`](../crates/onnx-rs/src/model.rs#L58-L62),
[`model.rs:99-131`](../crates/onnx-rs/src/model.rs#L99-L131)).

Readable Text natively owns the model IR/opset header and graph syntax, then
appends a protobuf-TextFormat residual for fields outside that DSL
([`text/ser.rs:54-79`](../crates/onnx-rs/src/text/ser.rs#L54-L79),
[`text/extensions.rs:16-66`](../crates/onnx-rs/src/text/extensions.rs#L16-L66)).
Therefore “⚠️ extension” is still lossless, but is not a first-class DSL form.

The descriptor inventory, every bound dtype, all textual codecs, binary model
I/O, and dedicated multi-device paths are exercised in
[`full_spec_serde.rs:483-895`](../crates/onnx-rs/tests/full_spec_serde.rs#L483-L895).

## Priority backlog

| Priority | Gap | Impact |
|---|---|---|
| Closed | ✅ **ONNX-ML `TypeProto.Opaque` is bound** (`Opaque.domain = 1`, `Opaque.name = 2`, `opaque_type = 7`). | Binary, JSON, TextProto, and readable-Text extension round-trips are lossless. |
| P0 | ⚠️ **The checker is substantially expanded but is not yet the full ONNX checker.** Round 3 added function declaration, topology/SSA, attribute-reference, import, recursion, IR-version, and packed-padding checks. As in official ONNX v1.20, local-function call-site consistency remains unenforced. | Most malformed inference-model protobuf structures are rejected; training validation is explicitly out of scope. Local-function call-site consistency, graph-wide parity, and the full schema catalog remain. |
| P0 | ⚠️ **The operator schema catalog now contains 63 high-value operators / 70 versioned entries.** Round 6 added `Tile` (13), `Pad` (25), `ScatterND` (18), `ScatterElements` (18), and `ConstantOfShape` (25); `Slice` (13), `Concat` (13), and `Expand` (13) were already registered. | Common transformer/CNN/indexing/shape-computation graphs validate, but this is not yet the complete standard or ONNX-ML schema catalog. |
| P1 | ⚠️ **Full-schema programmatic mutation is proto-first only.** The execution IR does not own training info, local functions, sparse initializers, quantization annotations, metadata on every object, or distributed annotations. `make_graph_authoritative` drops such fields ([`model.rs:134-141`](../crates/onnx-rs/src/model.rs#L134-L141)). | Loaded models are lossless, but graph rewrites cannot preserve every construct unless callers edit/rebuild a `ModelProto`. |
| P1 | ⚠️ **Readable Text is not a native full-spec grammar.** Many fields, including all multi-device messages, are carried in the extension block. | Round-trip is correct, but direct human editing is split between the graph DSL and TextFormat. |
| P1 | ⚠️ **Every operator currently present in the 63-op schema catalog has shape inference.** Round 6 adds checked Slice clamping, Concat dimension merging, Tile multiplication, bidirectional Expand, input-driven Pad, Scatter rank validation, and ConstantOfShape shape-data handling. Dynamic value inputs leave affected extents or the whole output unresolved rather than fabricating concrete shapes. Unsupported operators outside the schema catalog remain unknown ([`shape.rs:28-50`](../crates/onnx-rs/src/shape.rs#L28-L50)). | The catalog-local gap is closed; full ONNX shape-inference parity still depends on completing the operator catalog and adding sequence/optional/control-flow rules beyond `If`. |
| Closed | ✅ **Dense and sparse payload structural validation is implemented.** | Checked arithmetic covers element/byte counts, sub-byte packing and zero padding bits, segments, external offsets/lengths, sparse NNZ/index shape/order/uniqueness/bounds, and storage-field/dtype compatibility. |
| P2 | ⚠️ **Version conversion is not full ONNX conversion.** Built-ins cover `Reshape` v5/v13→v14 and the official `Softmax`/`LogSoftmax` v12→v13 last-axis rewrite or Shape/Flatten/op/Reshape decomposition. | Most opset transitions and all downgrades remain unsupported even when the wire schema is supported. |

## Complete protobuf message and field inventory

The field-number source is the vendored v1.20-compatible schema:
attributes/value/node/multi-device
([`onnx.proto3:134-320`](../crates/onnx-runtime-loader/proto/onnx.proto3#L134-L320)),
model/training/graph
([`onnx.proto3:341-599`](../crates/onnx-runtime-loader/proto/onnx.proto3#L341-L599)),
tensors/types/functions
([`onnx.proto3:604-996`](../crates/onnx-runtime-loader/proto/onnx.proto3#L604-L996)).

| Message | Every field (number) | Wire model | Checker | Text DSL | TextProto | JSON |
|---|---|---:|---:|---:|---:|---:|
| `AttributeProto` | `name`(1), `f`(2), `i`(3), `s`(4), `t`(5), `g`(6), `floats`(7), `ints`(8), `strings`(9), `tensors`(10), `graphs`(11), `doc_string`(13), `tp`(14), `type_protos`(15), `type`(20), `ref_attr_name`(21), `sparse_tensor`(22), `sparse_tensors`(23) | ✅ | ✅ name uniqueness, discriminator/payload conflicts, message payloads, reference rules; schema types also checked | ⚠️ scalar/list/graph forms native; payload details extension | ✅ | ✅ |
| `ValueInfoProto` | `name`(1), `type`(2), `doc_string`(3), `metadata_props`(4) | ✅ | ✅ names, top-level required types, recursive type validity, metadata uniqueness | ⚠️ name plus tensor dtype/shape native; remainder extension | ✅ | ✅ |
| `NodeProto` | `input`(1), `output`(2), `name`(3), `op_type`(4), `attribute`(5), `doc_string`(6), `domain`(7), `overload`(8), `metadata_props`(9), `device_configurations`(10) | ✅ | ⚠️ connectivity/domain/schema, attribute uniqueness, metadata, function-body topology, local-function call arity/attributes, and multi-device checked; local-function value types remain | ⚠️ edges/op/domain/attributes/subgraphs native; remainder extension | ✅ | ✅ |
| `IntIntListEntryProto` | `key`(1), `value`(2) | ✅ | ✅ group keys/members unique, referenced, non-empty, and in range | ⚠️ extension | ✅ | ✅ |
| `NodeDeviceConfigurationProto` | `configuration_id`(1), `sharding_spec`(2), `pipeline_stage`(3) | ✅ | ✅ configuration reference; ⚠️ sharding checks; `pipeline_stage` N/A | ⚠️ extension | ✅ | ✅ |
| `ShardingSpecProto` | `tensor_name`(1), `device`(2), `index_to_device_group_map`(3), `sharded_dim`(4) | ✅ | ✅ tensor-name, device indices, and group-map semantics | ⚠️ extension | ✅ | ✅ |
| `ShardedDimProto` | `axis`(1), `simple_sharding`(2) | ✅ | ✅ axis when rank is known; ⚠️ unknown-rank tensors cannot be range-checked | ⚠️ extension | ✅ | ✅ |
| `SimpleShardedDimProto` | oneof `dim_value`(1)/`dim_param`(2), `num_shards`(3) | ✅ | ✅ optional dimension oneof and required positive shard count | ⚠️ extension | ✅ | ✅ |
| `TrainingInfoProto` | `initialization`(1), `algorithm`(2), `initialization_binding`(3), `update_binding`(4) | ✅ | ❌ training graph combination/binding semantics | ⚠️ extension | ✅ | ✅ |
| `ModelProto` | `ir_version`(1), `producer_name`(2), `producer_version`(3), `domain`(4), `model_version`(5), `doc_string`(6), `graph`(7), `opset_import`(8), `metadata_props`(14), `training_info`(20), `functions`(25), `configuration`(26) | ✅ | ⚠️ IR 1–13 ceiling, version-gated fields, opsets, graph, functions including call arity/attributes, metadata, and multi-device checked; remaining graph/type invariants incomplete | ⚠️ IR/opset/graph native; other fields extension | ✅ | ✅ |
| `DeviceConfigurationProto` | `name`(1), `num_devices`(2), `device`(3) | ✅ | ✅ required name, positive count, optional name-list cardinality, unique IDs | ⚠️ extension | ✅ | ✅ |
| `StringStringEntryProto` | `key`(1), `value`(2) | ✅ | ✅ distinct metadata/external/group-map keys in checked inference scopes | ⚠️ extension | ✅ | ✅ |
| `TensorAnnotation` | `tensor_name`(1), `quant_parameter_tensor_names`(2) | ✅ | N/A in official v1.20 `checker.cc`; duplicate map keys are checked as metadata hygiene | ⚠️ extension | ✅ | ✅ |
| `GraphProto` | `node`(1), `name`(2), `initializer`(5), `doc_string`(10), `input`(11), `output`(12), `value_info`(13), `quantization_annotation`(14), `sparse_initializer`(15), `metadata_props`(16) | ✅ | ⚠️ acyclicity/SSA-like names/I/O/connectivity/initializer type; remaining graph rules ❌ | ⚠️ nodes/name/dense initializer refs/I/O native; other fields extension | ✅ | ✅ |
| `TensorProto.Segment` | `begin`(1), `end`(2) | ✅ | ✅ non-negative ordered bounds within the full tensor extent | ⚠️ extension | ✅ | ✅ |
| `TensorProto` | `dims`(1), `data_type`(2), `segment`(3), `float_data`(4), `int32_data`(5), `string_data`(6), `int64_data`(7), `name`(8), `raw_data`(9), `double_data`(10), `uint64_data`(11), `doc_string`(12), `external_data`(13), `data_location`(14), `metadata_props`(16) | ✅ | ✅ dtype, dimensions, checked element/byte counts, typed/raw exclusivity, sub-byte packing and zero padding bits, segments, metadata, and external location/offset/length structure; `string_data` is protobuf `bytes` and official v1.20 performs no UTF-8 check | ⚠️ dtype/shape/name references native; bytes/metadata extension | ✅ | ✅ |
| `SparseTensorProto` | `values`(1), `indices`(2), `dims`(3) | ✅ | ✅ values/indices presence, NNZ, INT64 index dtype, index shape/order/uniqueness/bounds | ⚠️ extension/attribute placeholder | ✅ | ✅ |
| `TensorShapeProto.Dimension` | oneof `dim_value`(1)/`dim_param`(2), `denotation`(3) | ✅ | ✅ negative concrete dimensions rejected; denotation has no additional structural rule | ⚠️ value native; denotation extension | ✅ | ✅ |
| `TensorShapeProto` | `dim`(1) | ✅ | ⚠️ rank is represented; all 63 registered operators have rules, while the incomplete catalog limits full shape coverage | ✅ tensor/sparse signatures | ✅ | ✅ |
| `TypeProto.Tensor` | `elem_type`(1), `shape`(2) | ✅ | ✅ required defined dtype and legal concrete dimensions | ✅ dtype/shape, ⚠️ denotations | ✅ | ✅ |
| `TypeProto.Sequence` | `elem_type`(1) | ✅ | ✅ required recursively-valid element type | ⚠️ extension (runtime attribute representation exists) | ✅ | ✅ |
| `TypeProto.Map` | `key_type`(1), `value_type`(2) | ✅ | ✅ integral/string key restriction and required recursive value | ⚠️ extension | ✅ | ✅ |
| `TypeProto.Optional` | `elem_type`(1) | ✅ | ✅ required tensor/sequence/map element | ⚠️ extension | ✅ | ✅ |
| `TypeProto.SparseTensor` | `elem_type`(1), `shape`(2) | ✅ | ✅ required defined dtype and legal concrete dimensions | ⚠️ dtype/shape can project, details extension | ✅ | ✅ |
| `TypeProto` | oneof `tensor_type`(1), `sequence_type`(4), `map_type`(5), `denotation`(6), `opaque_type`(7), `sparse_tensor_type`(8), `optional_type`(9) | ✅ | ✅ selected variant and recursive semantics | ⚠️ tensor/sparse signatures native, containers/opaque/denotation extension | ✅ | ✅ |
| **ONNX-ML only:** `TypeProto.Opaque` | `domain`(1), `name`(2); `TypeProto.opaque_type`(7) | ✅ | ✅ (both payload fields are optional by specification) | ⚠️ extension | ✅ | ✅ |
| `OperatorSetIdProto` | `domain`(1), `version`(2) | ✅ | ✅ used-domain import and function/model schema-revision compatibility checked; repeated-domain map semantics match v1.20 checker behavior | ✅ header | ✅ | ✅ |
| `FunctionProto` | `name`(1), `input`(4), `output`(5), `attribute`(6), `node`(7), `doc_string`(8), `opset_import`(9), `domain`(10), `attribute_proto`(11), `value_info`(12), `overload`(13), `metadata_props`(14) | ✅ | ✅ structural validation: required identifiers, signature/default uniqueness, unique `(domain,name,overload)`, topology/SSA, nested graphs, attribute references, imports/compatibility, recursion, and call arity/attributes; ⚠️ call value types remain | ⚠️ extension | ✅ | ✅ |

Reserved/deprecated schema members are intentionally not public fields:
`AttributeProto` numbers 12 and 16–19/name `v`; `GraphProto` numbers 3, 4,
6–9; `FunctionProto.since_version`(2) and `status`(3).

### Remaining protobuf enums

| Enum | Values | Wire/TextProto/JSON | Checker/Text |
|---|---|---:|---:|
| `Version` | `_START_VERSION`(0), IR versions 1–12, `IR_VERSION`(13) | ✅ | ✅ checker requires `1..=13` and validates inference-field introduction gates; readable Text prints the integer |
| `TensorProto.DataLocation` | `DEFAULT`(0), `EXTERNAL`(1) | ✅ | ❌ complete external-data legality; ⚠️ Text extension |
| `OperatorStatus` | `EXPERIMENTAL`(0), `STABLE`(1) | ✅ descriptor enum | N/A: corresponding `FunctionProto.status` field is reserved since IR 8 |

## `TensorProto.DataType` inventory

The IR enum and ONNX integer conversions cover all standard v1.20/IR-13 values
([`dtype.rs:9-37`](../crates/onnx-runtime-ir/src/dtype.rs#L9-L37),
[`dtype.rs:151-190`](../crates/onnx-runtime-ir/src/dtype.rs#L151-L190)).
Readable Text has a spelling for every value
([`text/ser.rs:345-375`](../crates/onnx-rs/src/text/ser.rs#L345-L375)).

| Value | Number | Wire | Checker | Text | TextProto | JSON |
|---|---:|---:|---:|---:|---:|---:|
| `UNDEFINED` | 0 | ✅ preserved | ✅ rejected where a concrete tensor dtype is required | ✅ spelling | ✅ | ✅ |
| `FLOAT` | 1 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `UINT8` | 2 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `INT8` | 3 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `UINT16` | 4 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `INT16` | 5 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `INT32` | 6 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `INT64` | 7 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `STRING` | 8 | ✅ | ✅ payload count checked; official v1.20 permits arbitrary bytes | ✅ | ✅ | ✅ |
| `BOOL` | 9 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FLOAT16` | 10 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `DOUBLE` | 11 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `UINT32` | 12 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `UINT64` | 13 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `COMPLEX64` | 14 | ✅ | ✅ interleaved count | ✅ | ✅ | ✅ |
| `COMPLEX128` | 15 | ✅ | ✅ interleaved count | ✅ | ✅ | ✅ |
| `BFLOAT16` | 16 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FLOAT8E4M3FN` | 17 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FLOAT8E4M3FNUZ` | 18 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FLOAT8E5M2` | 19 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `FLOAT8E5M2FNUZ` | 20 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `UINT4` | 21 | ✅ | ✅ packed length and unused high padding bits checked | ✅ | ✅ | ✅ |
| `INT4` | 22 | ✅ | ✅ packed length and unused high padding bits checked | ✅ | ✅ | ✅ |
| `FLOAT4E2M1` | 23 | ✅ | ✅ packed length and unused high padding bits checked | ✅ | ✅ | ✅ |
| `FLOAT8E8M0` | 24 | ✅ | ✅ | ✅ | ✅ | ✅ |
| `UINT2` | 25 | ✅ | ✅ packed length and unused high padding bits checked | ✅ | ✅ | ✅ |
| `INT2` | 26 | ✅ | ✅ packed length and unused high padding bits checked | ✅ | ✅ | ✅ |

Unused high padding bits in the last packed sub-byte element are required to be
zero. Official v1.20 does not apply UTF-8 validation to `string_data`, which is
a protobuf `bytes` field.

## `AttributeProto.AttributeType` inventory

All concrete attribute kinds are represented in the shared IR
([`node.rs:81-107`](../crates/onnx-runtime-ir/src/node.rs#L81-L107)) and matched
by schema checking
([`check/rules.rs:1171-1188`](../crates/onnx-rs/src/check/rules.rs#L1171-L1188)).
The protobuf checker also validates names, discriminators, populated-union
conflicts, required message payloads, reference attributes, and duplicate names.

| Value | Number | Wire | Checker | Text DSL | TextProto | JSON |
|---|---:|---:|---:|---:|---:|---:|
| `UNDEFINED` | 0 | ✅ | ✅ rejected for IR-13 attributes | ⚠️ extension/inferred projection | ✅ | ✅ |
| `FLOAT` | 1 | ✅ | ✅ union + schema type | ✅ | ✅ | ✅ |
| `INT` | 2 | ✅ | ✅ union + schema type | ✅ | ✅ | ✅ |
| `STRING` | 3 | ✅ | ✅ union + schema type | ✅ UTF-8; ⚠️ opaque bytes extension | ✅ | ✅ |
| `TENSOR` | 4 | ✅ | ✅ union + schema + payload | ⚠️ reference plus extension payload | ✅ | ✅ |
| `GRAPH` | 5 | ✅ | ✅ union + schema + recursive graph checks | ✅ body; ⚠️ graph metadata extension | ✅ | ✅ |
| `FLOATS` | 6 | ✅ | ✅ union + schema type | ✅ | ✅ | ✅ |
| `INTS` | 7 | ✅ | ✅ union + schema type | ✅ | ✅ | ✅ |
| `STRINGS` | 8 | ✅ | ✅ union + schema type | ✅ UTF-8; ⚠️ opaque bytes extension | ✅ | ✅ |
| `TENSORS` | 9 | ✅ | ✅ union + schema + payload | ⚠️ cardinality placeholder plus extension | ✅ | ✅ |
| `GRAPHS` | 10 | ✅ | ✅ union + schema + recursive graph checks | ✅ bodies; ⚠️ graph metadata extension | ✅ | ✅ |
| `SPARSE_TENSOR` | 11 | ✅ | ✅ union + schema + sparse validity | ⚠️ placeholder plus extension | ✅ | ✅ |
| `SPARSE_TENSORS` | 12 | ✅ | ✅ union + schema + sparse validity | ⚠️ cardinality placeholder plus extension | ✅ | ✅ |
| `TYPE_PROTO` | 13 | ✅ | ✅ union + schema + recursive TypeProto validity | ⚠️ placeholder plus extension | ✅ | ✅ |
| `TYPE_PROTOS` | 14 | ✅ | ✅ union + schema + recursive TypeProto validity | ⚠️ cardinality placeholder plus extension | ✅ | ✅ |

## Type variants and execution projection

The shared IR owns standard `Tensor`, `Sequence`, `Optional`, `Map`, and
`SparseTensor` variants
([`tensor.rs:61-70`](../crates/onnx-runtime-ir/src/tensor.rs#L61-L70)).
The loader converts all five for attribute values, but graph values remain
tensor-centric: sequence/map/optional `ValueInfoProto` values become unknown
execution placeholders while their exact proto remains retained. This is
serialization-complete but runtime-partial. ONNX-ML `Opaque` is retained and
validated, and likewise projects to an unknown execution placeholder.

## Multi-device implementation status

The public `onnx-rs` API re-exports the exact generated IR-13 types
([`model.rs:33-41`](../crates/onnx-rs/src/model.rs#L33-L41)):

- `DeviceConfigurationProto`
- `IntIntListEntryProto`
- `NodeDeviceConfigurationProto`
- `ShardingSpecProto`
- `ShardedDimProto`
- `SimpleShardedDimProto` and its `dim_value`/`dim_param` oneof

The exact field wiring is:

| Path | Field numbers |
|---|---|
| `ModelProto.configuration` | 26 |
| `NodeProto.device_configurations` | 10 |
| `DeviceConfigurationProto` | `name` 1, `num_devices` 2, `device` 3 |
| `NodeDeviceConfigurationProto` | `configuration_id` 1, `sharding_spec` 2, `pipeline_stage` 3 |
| `ShardingSpecProto` | `tensor_name` 1, `device` 2, `index_to_device_group_map` 3, `sharded_dim` 4 |
| `IntIntListEntryProto` | `key` 1, `value` 2 |
| `ShardedDimProto` | `axis` 1, `simple_sharding` 2 |
| `SimpleShardedDimProto` | `dim_value` 1 / `dim_param` 2, `num_shards` 3 |

`MultiDeviceConfigurationRule` checks the IR-version gate, model configuration
names/counts, device name cardinality, node configuration references,
tensor-name references, known-rank axis ranges, optional dimension-oneof
handling, and required positive shard counts
([`check/rules.rs:667-992`](../crates/onnx-rs/src/check/rules.rs#L667-L992)).
It recursively checks main/nested/training graphs and function nodes.

Deferred multi-device checker items:

- axis validation when the referenced tensor rank is not declared;
- backend availability/support (the IR explicitly permits backends to ignore
  unsupported annotations);
- pipeline-stage policy, because v1.20 defines it as an optional identifier and
  does not constrain its numeric range.

## Top-level constructs beyond field serialization

| Construct | Proto/TextProto/JSON | Readable Text | Checker | Shape/runtime |
|---|---:|---:|---:|---:|
| Inference graph | ✅ | ✅ core graph syntax, ⚠️ residual metadata | ⚠️ core structural rules | ⚠️ operator-dependent |
| Nested graph attributes | ✅ | ✅ body, ⚠️ residual metadata | ⚠️ recursive structural rules | ⚠️ operator-dependent |
| Sparse initializers | ✅ | ⚠️ extension | ✅ structural/payload/index semantics | ❌ sparse execution storage |
| Quantization annotations | ✅ | ⚠️ extension | N/A in official v1.20 checker; duplicate parameter-map keys checked | ❌ annotation-driven behavior |
| Local functions | ✅ | ⚠️ extension | ✅ declaration/body; call-site arity/attribute consistency intentionally remains unenforced like official v1.20 | ⚠️ loader may inline; declarations retained only in source proto |
| Training info | ✅ | ⚠️ extension | ❌ training semantics | ❌ training execution |
| Metadata properties on model/graph/node/value/tensor/function | ✅ | ⚠️ extension | ✅ distinct-key checks | N/A |
| Multi-device/sharding | ✅ | ⚠️ extension | ✅/⚠️ only unknown-rank axes deferred | N/A hints; no distributed executor |
| ONNX-ML `Opaque` type | ✅ | ⚠️ extension | ✅ optional payload semantics | ❌ opaque execution values |
| Standard operator schemas through opset 25 | ✅ node serialization | ✅ node syntax | ⚠️ 63 high-value operators / 70 versioned entries | ✅ for the registered catalog; ⚠️ catalog incomplete |
| ONNX-ML operator schemas | ✅ generic node serialization | ✅ generic node syntax | ❌ | ⚠️ only generic/custom registration paths |

## Test evidence

- Descriptor inventory: every message/enum in the bound standard schema plus
  ONNX-ML `TypeProto.Opaque`; descriptor assertions pin `domain` = 1, `name` = 2,
  and `opaque_type` = 7.
- Binary wire: a synthetic IR-13 model containing model configuration, node
  device configuration, sharding specs, group-map entries, numeric and symbolic
  sharded dimensions is encoded, decoded, and re-encoded byte-identically.
- Text: readable Text, protobuf TextFormat, and canonical protobuf JSON each
  round-trip the same multi-device model.
- Checker: metadata, attribute-union, recursive type/container, dense payload,
  sparse COO, function, IR-gating, packed-padding, and distributed-annotation
  rules have positive and negative tests.
  Sharding specs that omit the optional dimension oneof pass; invalid group
  maps, configuration IDs, tensor names, axes, and shard counts are rejected.
  Local-function calls test exact input/output arity, required attributes,
  omitted defaults, undeclared attributes, and calls from function bodies.
- Schemas: each newly added operator has a registry test pinning its ONNX
  version and arity, plus detail tests for attributes, optional inputs, concrete
  inputs, defaults, and type constraints.
- Shape inference: representative graphs cover every family in the registered
  catalog, including Round-6 static, dynamic-input, negative-index/step,
  broadcasting, rank-relation, and checked-overflow cases for Slice, Concat,
  Tile, Expand, Pad, ScatterND, ScatterElements, and ConstantOfShape. `If`
  recursively infers both branches, requires matching output
  counts and element types, preserves agreeing dimensions, and replaces
  conflicting branch dimensions with a fresh symbolic dimension.
- The fixture is synthetic. The ONNX v1.20 repository contains the schema,
  normative IR text, and proposal material, but no checked-in binary multi-device
  model fixture was found.

  ## Explicitly out of scope

  Training execution and `TrainingInfoProto` semantic validation are intentionally
  unsupported per project scope. Training fields remain losslessly serialized,
  but no training-related checker, schema, shape-inference, or runtime work is
  required for full inference-spec support.

  ## Round-2 official-source verification

  The round-2 schema and inference details were checked against tag `v1.20.0`:

  - `onnx/defs/math/defs.cc`: `Sigmoid`/`Tanh`/`Erf`/`Sqrt`/`Exp`/`Log`
    revision 13, `Pow` revision 15, `Clip` revision 13, and `Expand` revision 13;
  - `onnx/defs/tensor/defs.cc`: `Where` revision 16;
  - `onnx/defs/reduction/defs.cc` and `onnx/defs/reduction/utils.cc`:
    `ReduceSum` revision 13, `ReduceMean` revision 18, dynamic optional axes,
    `keepdims = 1`, `noop_with_empty_axes = 0`, and the reduction type set;
  - `onnx/defs/schema.cc`: exact `all_float_types_ir4`,
    `all_numeric_types_ir4`, `all_tensor_types_ir4`, and
    `numeric_types_for_math_reduction_ir4` expansions;
  - `onnx/defs/controlflow/utils.cc` and `onnx/defs/shape_inference.cc`:
    recursive branch inference, output-count/type checks, and `UnionTypeInfo`
    shape merging for `If`;
  - `onnx/checker.cc`: no quantization-annotation validation and no UTF-8
    validation for `TensorProto.string_data`/attribute byte payloads.

  ## Round-3 official-source verification

  Round-3 details were checked against ONNX tag `v1.20.0`:

  - `onnx/checker.cc`: IR versions above `IR_VERSION` are rejected; IR `< 3`
    forbids `opset_import`, IR `>= 3` requires it; IR `<= 3` initializers must
    also be graph inputs; functions require non-empty name/domain, unique
    inputs/outputs/attributes, topologically ordered nodes, SSA outputs,
    function-level imports, and schema-revision-compatible model imports.
  - `docs/IR.md` and `onnx/onnx.proto3`: `BFLOAT16` starts at IR 4;
    `FunctionProto.attribute_proto` starts at IR 9;
    overload/value-info/metadata and `UINT4`/`INT4` start at IR 10;
    multi-device and `FLOAT4E2M1` at IR 11; `FLOAT8E8M0` at IR 12; and
    `UINT2`/`INT2` at IR 13. Training gates remain intentionally out of scope.
  - `onnx/defs/math/defs.cc`: `Sub` and `Div` revision 14 with
    `all_numeric_types_ir4`; `Neg` revision 13 with the eight signed numeric
    types; `Abs` revision 13 with `all_numeric_types_ir4`; `Mod` revision 13
    with `fmod` integer attribute default `0` and `all_numeric_types_ir4`.
  - `onnx/numpy_helper.py` and `onnx/helper.py`: 4-bit values are packed two per
    byte, 2-bit values four per byte, low lanes first, with zero-filled trailing
    lanes in the final byte.

  ## Round-4 official-source verification

  Round-4 details were checked against ONNX tag `v1.20.0` and its opset-25
  schema registry:

  - `onnx/defs/reduction/defs.cc` and `onnx/defs/reduction/utils.cc`:
    `ReduceMax`/`ReduceMin` revision 20, the six revision-18 math reductions,
    optional dynamic `axes`, `keepdims = 1`, `noop_with_empty_axes = 0`, and
    exact reduction type sets; `ArgMax`/`ArgMin` revision 13 and `int64` output.
  - `onnx/defs/nn/defs.cc` and the registered schemas: `LogSoftmax` revision 13;
    `RMSNormalization` revision 23 with output type `V`, `axis = -1`,
    `epsilon = 1e-5`, and `stash_type = 1`.
  - `onnx/checker.cc`: local-function nodes intentionally bypass schema
    verification and the upstream call-site consistency check remains a TODO.
    The local checker likewise accepts call-site arity, required-attribute, and
    undeclared-attribute mismatches to preserve official acceptance semantics.

  ## Round-5 official-source verification

  Round-5 details were checked against ONNX tag `v1.20.0` and its opset-25
  generated operator catalog:

  - `docs/Operators.md` and the corresponding math/tensor definitions:
    `GatherElements`/`GatherND` revision 13; `Equal` revision 19;
    `Greater`/`Less` revision 13; `And`/`Or` revision 7; `Not` revision 1;
    `Cast`/`Shape`/`Size` revision 25; `NonZero` revision 13; `Range` revision
    11; and `Split` revision 18, with their exact attributes, arities, and type
    constraints.
  - `onnx/version_converter/adapters/softmax_12_13.h`: both `Softmax` and
    `LogSoftmax` rewrite a last-axis case to `axis = -1`; other valid axes use
    Shape→Flatten→op→Reshape to preserve revision-12 flattening semantics.
  - Shape inference uses checked rank/capacity/count arithmetic, non-clamping
    axis validation for newly covered indexing/split paths, and rejects concrete
    element/range extents beyond `isize::MAX`.

  ## Round-6 official-source verification

  Round-6 details were checked against ONNX tag `v1.20.0` and its opset-25
  schema registry:

  - `onnx/defs/tensor/defs.cc`, `onnx/defs/tensor/utils.cc`,
    `onnx/defs/math/defs.cc`, and `onnx/defs/generator/defs.cc`: canonical
    schemas are `Slice` 13, `Concat` 13, `Tile` 13, `Expand` 13, `Pad` 25,
    `ScatterND` 18, `ScatterElements` 18, and `ConstantOfShape` 25.
  - `Slice`, `Concat`, and `Expand` were already in the catalog. Round 6 adds
    schemas for `Tile`, `Pad`, both scatter operators, and `ConstantOfShape`,
    including exact optional inputs, defaults, reduction attributes, and v1.20
    type sets.
  - Shape inference uses checked axis normalization and `isize::MAX` extent
    bounds. Runtime-computed Slice/Pad/Tile inputs preserve known rank without
    inventing affected extents; dynamic Expand and ConstantOfShape shape inputs
    leave the output unresolved. An empty ConstantOfShape shape vector still
    produces a known rank-0 scalar.

  ## Remaining non-training gaps

  - Complete standard and ONNX-ML operator-schema catalogs (63 high-value standard
    operators / 70 versioned entries are currently registered). The largest
    standard-library gaps remain pooling/resize, recurrent, quantization,
    sequence/optional, and remaining control-flow/data-movement operators.
  - Local-function call-site consistency, matching the TODO in official
    `onnx/checker.cc`.
  - Full-schema programmatic mutation, native readable-Text grammar, complete
    shape inference outside the registered catalog, and complete opset version
    conversion.
