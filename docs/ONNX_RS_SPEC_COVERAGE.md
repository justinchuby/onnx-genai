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
ONNX-ML adds one schema delta not present in the vendored standard schema:
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
| P0 | ❌ **ONNX-ML `TypeProto.Opaque` is absent** (`Opaque.domain = 1`, `Opaque.name = 2`, `opaque_type = 7`). | Binary protobuf decoding cannot retain the unknown field; JSON and TextProto reject it. |
| P0 | ❌ **The checker is not the full ONNX checker.** It lacks most protobuf-level invariants: required fields, unique metadata keys, attribute union/discriminator consistency, tensor payload rules, sparse tensor rules, function/training rules, and most IR-version gates. | A malformed model can round-trip successfully without being rejected. |
| P0 | ❌ **The operator schema catalog contains only eight operators** (`MatMul`, `Gemm`, `Add`, `Relu`, `Conv`, `Mul`, `Identity`, `If`) ([`schema/mod.rs:345-356`](../crates/onnx-rs/src/schema/mod.rs#L345-L356)). | Schema checking rejects or cannot validate most standard and ONNX-ML operators/opsets. |
| P1 | ⚠️ **Full-schema programmatic mutation is proto-first only.** The execution IR does not own training info, local functions, sparse initializers, quantization annotations, metadata on every object, or distributed annotations. `make_graph_authoritative` drops such fields ([`model.rs:134-141`](../crates/onnx-rs/src/model.rs#L134-L141)). | Loaded models are lossless, but graph rewrites cannot preserve every construct unless callers edit/rebuild a `ModelProto`. |
| P1 | ⚠️ **Readable Text is not a native full-spec grammar.** Many fields, including all multi-device messages, are carried in the extension block. | Round-trip is correct, but direct human editing is split between the graph DSL and TextFormat. |
| P1 | ⚠️ **Shape inference is permissive and incomplete.** Unsupported operators remain unknown ([`shape.rs:28-50`](../crates/onnx-rs/src/shape.rs#L28-L50)). | Full ONNX shape-inference parity is not yet achieved. |
| P1 | ❌ **Tensor/sparse payload semantic validation is missing.** | Incorrect element counts, invalid storage field/dtype combinations, sparse index ordering, and external-data constraints are not comprehensively checked by `onnx-rs`. |
| P2 | ⚠️ **Version conversion is not full ONNX conversion.** Only a small adapter set is registered. | Most opset transitions cannot be performed even when the wire schema is supported. |

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
| `AttributeProto` | `name`(1), `f`(2), `i`(3), `s`(4), `t`(5), `g`(6), `floats`(7), `ints`(8), `strings`(9), `tensors`(10), `graphs`(11), `doc_string`(13), `tp`(14), `type_protos`(15), `type`(20), `ref_attr_name`(21), `sparse_tensor`(22), `sparse_tensors`(23) | ✅ | ⚠️ schema type checks only; union/ref rules ❌ | ⚠️ scalar/list/graph forms native; payload details extension | ✅ | ✅ |
| `ValueInfoProto` | `name`(1), `type`(2), `doc_string`(3), `metadata_props`(4) | ✅ | ⚠️ top-level I/O name/liveness checked; required type/shape and metadata uniqueness ❌ | ⚠️ name plus tensor dtype/shape native; remainder extension | ✅ | ✅ |
| `NodeProto` | `input`(1), `output`(2), `name`(3), `op_type`(4), `attribute`(5), `doc_string`(6), `domain`(7), `overload`(8), `metadata_props`(9), `device_configurations`(10) | ✅ | ⚠️ connectivity/domain/schema and multi-device checked; overload/metadata/doc rules ❌ | ⚠️ edges/op/domain/attributes/subgraphs native; remainder extension | ✅ | ✅ |
| `IntIntListEntryProto` | `key`(1), `value`(2) | ✅ | ❌ group-map semantics/uniqueness not checked | ⚠️ extension | ✅ | ✅ |
| `NodeDeviceConfigurationProto` | `configuration_id`(1), `sharding_spec`(2), `pipeline_stage`(3) | ✅ | ✅ configuration reference; ⚠️ sharding checks; `pipeline_stage` N/A | ⚠️ extension | ✅ | ✅ |
| `ShardingSpecProto` | `tensor_name`(1), `device`(2), `index_to_device_group_map`(3), `sharded_dim`(4) | ✅ | ✅ tensor-name match; ❌ device/group-map semantics | ⚠️ extension | ✅ | ✅ |
| `ShardedDimProto` | `axis`(1), `simple_sharding`(2) | ✅ | ✅ axis when rank is known; ⚠️ unknown-rank tensors cannot be range-checked | ⚠️ extension | ✅ | ✅ |
| `SimpleShardedDimProto` | oneof `dim_value`(1)/`dim_param`(2), `num_shards`(3) | ✅ | ✅ optional dimension oneof and required positive shard count | ⚠️ extension | ✅ | ✅ |
| `TrainingInfoProto` | `initialization`(1), `algorithm`(2), `initialization_binding`(3), `update_binding`(4) | ✅ | ❌ training graph combination/binding semantics | ⚠️ extension | ✅ | ✅ |
| `ModelProto` | `ir_version`(1), `producer_name`(2), `producer_version`(3), `domain`(4), `model_version`(5), `doc_string`(6), `graph`(7), `opset_import`(8), `metadata_props`(14), `training_info`(20), `functions`(25), `configuration`(26) | ✅ | ⚠️ IR-present/opset/graph/multi-device checks; remaining model invariants ❌ | ⚠️ IR/opset/graph native; other fields extension | ✅ | ✅ |
| `DeviceConfigurationProto` | `name`(1), `num_devices`(2), `device`(3) | ✅ | ✅ required name, positive count, optional name-list cardinality, unique IDs | ⚠️ extension | ✅ | ✅ |
| `StringStringEntryProto` | `key`(1), `value`(2) | ✅ | ❌ distinct-key requirements generally unchecked | ⚠️ extension | ✅ | ✅ |
| `TensorAnnotation` | `tensor_name`(1), `quant_parameter_tensor_names`(2) | ✅ | ❌ annotation target/key semantics | ⚠️ extension | ✅ | ✅ |
| `GraphProto` | `node`(1), `name`(2), `initializer`(5), `doc_string`(10), `input`(11), `output`(12), `value_info`(13), `quantization_annotation`(14), `sparse_initializer`(15), `metadata_props`(16) | ✅ | ⚠️ acyclicity/SSA-like names/I/O/connectivity/initializer type; remaining graph rules ❌ | ⚠️ nodes/name/dense initializer refs/I/O native; other fields extension | ✅ | ✅ |
| `TensorProto.Segment` | `begin`(1), `end`(2) | ✅ | ❌ segment bounds/legality | ⚠️ extension | ✅ | ✅ |
| `TensorProto` | `dims`(1), `data_type`(2), `segment`(3), `float_data`(4), `int32_data`(5), `string_data`(6), `int64_data`(7), `name`(8), `raw_data`(9), `double_data`(10), `uint64_data`(11), `doc_string`(12), `external_data`(13), `data_location`(14), `metadata_props`(16) | ✅ | ⚠️ initializer dtype/shape only; payload/storage/external-data rules ❌ | ⚠️ dtype/shape/name references native; bytes/metadata extension | ✅ | ✅ |
| `SparseTensorProto` | `values`(1), `indices`(2), `dims`(3) | ✅ | ❌ NNZ, index dtype/shape/order/uniqueness/bounds | ⚠️ extension/attribute placeholder | ✅ | ✅ |
| `TensorShapeProto.Dimension` | oneof `dim_value`(1)/`dim_param`(2), `denotation`(3) | ✅ | ⚠️ represented, but denotation and most shape legality rules unchecked | ⚠️ value native; denotation extension | ✅ | ✅ |
| `TensorShapeProto` | `dim`(1) | ✅ | ⚠️ rank is represented; full shape rules depend on operators | ✅ tensor/sparse signatures | ✅ | ✅ |
| `TypeProto.Tensor` | `elem_type`(1), `shape`(2) | ✅ | ⚠️ runtime representation; required/non-`UNDEFINED` rules incomplete | ✅ dtype/shape, ⚠️ denotations | ✅ | ✅ |
| `TypeProto.Sequence` | `elem_type`(1) | ✅ | ❌ required element/container semantics | ⚠️ extension (runtime attribute representation exists) | ✅ | ✅ |
| `TypeProto.Map` | `key_type`(1), `value_type`(2) | ✅ | ❌ integral/string key restriction and required value | ⚠️ extension | ✅ | ✅ |
| `TypeProto.Optional` | `elem_type`(1) | ✅ | ❌ required element/container semantics | ⚠️ extension | ✅ | ✅ |
| `TypeProto.SparseTensor` | `elem_type`(1), `shape`(2) | ✅ | ⚠️ represented; required/non-`UNDEFINED` rules incomplete | ⚠️ dtype/shape can project, details extension | ✅ | ✅ |
| `TypeProto` | oneof `tensor_type`(1), `sequence_type`(4), `map_type`(5), `denotation`(6), `sparse_tensor_type`(8), `optional_type`(9) | ✅ | ⚠️ variants represented; oneof-specific semantic validation ❌ | ⚠️ tensor/sparse signatures native, containers/denotation extension | ✅ | ✅ |
| **ONNX-ML only:** `TypeProto.Opaque` | `domain`(1), `name`(2); `TypeProto.opaque_type`(7) | ❌ | ❌ | ❌ | ❌ | ❌ |
| `OperatorSetIdProto` | `domain`(1), `version`(2) | ✅ | ⚠️ used-domain import checked; duplicate domains/version compatibility incomplete | ✅ header | ✅ | ✅ |
| `FunctionProto` | `name`(1), `input`(4), `output`(5), `attribute`(6), `node`(7), `doc_string`(8), `opset_import`(9), `domain`(10), `attribute_proto`(11), `value_info`(12), `overload`(13), `metadata_props`(14) | ✅ | ❌ signature uniqueness, attribute/default rules, recursion, topology, imports, identifier uniqueness | ⚠️ extension | ✅ | ✅ |

Reserved/deprecated schema members are intentionally not public fields:
`AttributeProto` numbers 12 and 16–19/name `v`; `GraphProto` numbers 3, 4,
6–9; `FunctionProto.since_version`(2) and `status`(3).

### Remaining protobuf enums

| Enum | Values | Wire/TextProto/JSON | Checker/Text |
|---|---|---:|---:|
| `Version` | `_START_VERSION`(0), IR versions 1–12, `IR_VERSION`(13) | ✅ | ⚠️ checker requires only `>= 1`; readable Text prints the integer |
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
| `UNDEFINED` | 0 | ✅ preserved | ⚠️ rejected by runtime tensor projection, no complete proto rule | ✅ spelling | ✅ | ✅ |
| `FLOAT` | 1 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `UINT8` | 2 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `INT8` | 3 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `UINT16` | 4 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `INT16` | 5 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `INT32` | 6 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `INT64` | 7 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `STRING` | 8 | ✅ | ⚠️ payload count/UTF-8 tensor rules incomplete | ✅ | ✅ | ✅ |
| `BOOL` | 9 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `FLOAT16` | 10 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `DOUBLE` | 11 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `UINT32` | 12 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `UINT64` | 13 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `COMPLEX64` | 14 | ✅ | ⚠️ interleaved-payload rules incomplete | ✅ | ✅ | ✅ |
| `COMPLEX128` | 15 | ✅ | ⚠️ interleaved-payload rules incomplete | ✅ | ✅ | ✅ |
| `BFLOAT16` | 16 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `FLOAT8E4M3FN` | 17 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `FLOAT8E4M3FNUZ` | 18 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `FLOAT8E5M2` | 19 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `FLOAT8E5M2FNUZ` | 20 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `UINT4` | 21 | ✅ | ⚠️ packing/padding rules incomplete | ✅ | ✅ | ✅ |
| `INT4` | 22 | ✅ | ⚠️ packing/padding rules incomplete | ✅ | ✅ | ✅ |
| `FLOAT4E2M1` | 23 | ✅ | ⚠️ packing/padding rules incomplete | ✅ | ✅ | ✅ |
| `FLOAT8E8M0` | 24 | ✅ | ⚠️ | ✅ | ✅ | ✅ |
| `UINT2` | 25 | ✅ | ⚠️ packing/padding rules incomplete | ✅ | ✅ | ✅ |
| `INT2` | 26 | ✅ | ⚠️ packing/padding rules incomplete | ✅ | ✅ | ✅ |

“⚠️ checker” means the dtype is represented and initializer declaration
dtype/shape consistency can be checked, but the checker does not yet implement
the complete `TensorProto` storage-field and element-count rules.

## `AttributeProto.AttributeType` inventory

All concrete attribute kinds are represented in the shared IR
([`node.rs:81-107`](../crates/onnx-runtime-ir/src/node.rs#L81-L107)) and matched
by schema checking
([`check/rules.rs:1171-1188`](../crates/onnx-rs/src/check/rules.rs#L1171-L1188)).
The missing part is protobuf-level union/discriminator and context validation.

| Value | Number | Wire | Checker | Text DSL | TextProto | JSON |
|---|---:|---:|---:|---:|---:|---:|
| `UNDEFINED` | 0 | ✅ | ❌ discriminator/content inference is not a full validity check | ⚠️ extension/inferred projection | ✅ | ✅ |
| `FLOAT` | 1 | ✅ | ⚠️ schema type only | ✅ | ✅ | ✅ |
| `INT` | 2 | ✅ | ⚠️ schema type only | ✅ | ✅ | ✅ |
| `STRING` | 3 | ✅ | ⚠️ schema type only | ✅ UTF-8; ⚠️ opaque bytes extension | ✅ | ✅ |
| `TENSOR` | 4 | ✅ | ⚠️ schema type only | ⚠️ reference plus extension payload | ✅ | ✅ |
| `GRAPH` | 5 | ✅ | ⚠️ schema type/recursive graph checks | ✅ body; ⚠️ graph metadata extension | ✅ | ✅ |
| `FLOATS` | 6 | ✅ | ⚠️ schema type only | ✅ | ✅ | ✅ |
| `INTS` | 7 | ✅ | ⚠️ schema type only | ✅ | ✅ | ✅ |
| `STRINGS` | 8 | ✅ | ⚠️ schema type only | ✅ UTF-8; ⚠️ opaque bytes extension | ✅ | ✅ |
| `TENSORS` | 9 | ✅ | ⚠️ schema type only | ⚠️ cardinality placeholder plus extension | ✅ | ✅ |
| `GRAPHS` | 10 | ✅ | ⚠️ schema type/recursive graph checks | ✅ bodies; ⚠️ graph metadata extension | ✅ | ✅ |
| `SPARSE_TENSOR` | 11 | ✅ | ⚠️ schema type; sparse validity ❌ | ⚠️ placeholder plus extension | ✅ | ✅ |
| `SPARSE_TENSORS` | 12 | ✅ | ⚠️ schema type; sparse validity ❌ | ⚠️ cardinality placeholder plus extension | ✅ | ✅ |
| `TYPE_PROTO` | 13 | ✅ | ⚠️ schema type; TypeProto validity ❌ | ⚠️ placeholder plus extension | ✅ | ✅ |
| `TYPE_PROTOS` | 14 | ✅ | ⚠️ schema type; TypeProto validity ❌ | ⚠️ cardinality placeholder plus extension | ✅ | ✅ |

## Type variants and execution projection

The shared IR owns standard `Tensor`, `Sequence`, `Optional`, `Map`, and
`SparseTensor` variants
([`tensor.rs:61-70`](../crates/onnx-runtime-ir/src/tensor.rs#L61-L70)).
The loader converts all five for attribute values, but graph values remain
tensor-centric: sequence/map/optional `ValueInfoProto` values become unknown
execution placeholders while their exact proto remains retained. This is
serialization-complete but runtime/checker-partial. ONNX-ML `Opaque` is absent.

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

- semantic validation of `device` and `index_to_device_group_map`;
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
| Sparse initializers | ✅ | ⚠️ extension | ❌ semantic checks | ❌ sparse execution storage |
| Quantization annotations | ✅ | ⚠️ extension | ❌ | ❌ annotation-driven behavior |
| Local functions | ✅ | ⚠️ extension | ❌ full function checker | ⚠️ loader may inline; declarations retained only in source proto |
| Training info | ✅ | ⚠️ extension | ❌ training semantics | ❌ training execution |
| Metadata properties on model/graph/node/value/tensor/function | ✅ | ⚠️ extension | ❌ distinct-key checks | N/A |
| Multi-device/sharding | ✅ | ⚠️ extension | ✅/⚠️ as detailed above | N/A hints; no distributed executor |
| ONNX-ML `Opaque` type | ❌ | ❌ | ❌ | ❌ |
| Standard operator schemas through opset 24 | ✅ node serialization | ✅ node syntax | ❌ only eight schemas | ⚠️ partial inference registry |
| ONNX-ML operator schemas | ✅ generic node serialization | ✅ generic node syntax | ❌ | ⚠️ only generic/custom registration paths |

## Test evidence

- Descriptor inventory: every message/enum in the bound standard schema.
- Binary wire: a synthetic IR-13 model containing model configuration, node
  device configuration, sharding specs, group-map entries, numeric and symbolic
  sharded dimensions is encoded, decoded, and re-encoded byte-identically.
- Text: readable Text, protobuf TextFormat, and canonical protobuf JSON each
  round-trip the same multi-device model.
- Checker: valid distributed annotations, including sharding specs that omit the
  optional dimension oneof, pass; invalid device-name cardinality, missing
  configuration IDs, unknown tensor names, out-of-range axes, and zero shard
  counts are rejected.
- The fixture is synthetic. The ONNX v1.20 repository contains the schema,
  normative IR text, and proposal material, but no checked-in binary multi-device
  model fixture was found.
