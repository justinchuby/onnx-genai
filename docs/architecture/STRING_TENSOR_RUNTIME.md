# `tensor(string)` Runtime Support

Status: design only. Implementation is intentionally deferred to a later slice so it does not
collide with the concurrent Sequence zero-copy tensor-layer work on `wt-seq-zerocopy`.

## Decision summary

Recommend a **separate, host-only string storage variant containing `Vec<String>`**, alongside the
existing byte/device-buffer storage:

```rust
// Illustrative only.
enum TensorStorage {
    Raw(RawTensorStorage), // existing DeviceBuffer + allocator/foreign guard path
    Strings(Vec<String>),  // one initialized Rust String per logical element
}
```

Do not encode `String` objects in the byte-backed path, and do not cast a byte pointer to
`*const String`. Kernel-facing string views must borrow `&[String]` / `&mut [String]` and expose
`&str` accessors. Keep the public view/accessor contract representation-neutral so a packed
UTF-8 arena can be added later if profiling justifies it.

The largest safety risk is treating `storage_bytes == 0` as valid backing for a non-empty string
tensor and then reinterpreting that allocation as initialized Rust `String` values. Such a pointer
has neither the required allocation extent nor initialized `String` validity, alignment, or
provenance; reading it is undefined behavior and dropping values reached through it could corrupt
memory. The storage enum eliminates this state: raw views reject `String`, while string access is
created only from an actual `Vec<String>`.

## 1. Current implementation

There is no shared `onnx-runtime-tensor` crate today. The relevant types are split across IR,
session, eager, and EP API:

| Concern | Current location | Current behavior |
|---|---|---|
| ONNX dtype | `crates/onnx-runtime-ir/src/dtype.rs:9-37` | `DataType::String` is represented. |
| Element/storage sizing | `crates/onnx-runtime-ir/src/dtype.rs:39-67`, `119-149` | `String::byte_size()` is `0`; `checked_storage_bytes` consequently returns `Some(0)` for any string element count. |
| Constant payload | `crates/onnx-runtime-ir/src/tensor.rs:8-24` | `TensorData` already separates numeric `data: Vec<u8>` from `strings: Vec<String>`. |
| Session-owned tensor | `crates/onnx-runtime-session/src/tensor.rs:83-113` | Every runtime tensor owns a `DeviceBuffer`; no string variant exists. |
| Session construction/access | `crates/onnx-runtime-session/src/tensor.rs:279-315`, `388-475` | `from_raw`, `as_bytes`, cloning, and typed access all assume byte storage. |
| Eager-owned tensor | `crates/onnx-runtime-eager/src/tensor.rs:96-110`, `112-168`, `233-255` | Duplicates the same byte-only design. |
| Kernel views | `crates/onnx-runtime-ep-api/src/tensor.rs:86-128`, `130-238`, `240-312` | Views are raw device pointers plus byte offsets; `validate` explicitly rejects `String`. |
| Executor storage | `crates/onnx-runtime-session/src/executor.rs:271-355` | Runtime values are primarily `HashMap<ValueId, DeviceBuffer>` plus special maps for views/sequences. |
| Initializer materialization | `crates/onnx-runtime-session/src/executor.rs:1040-1100` | All initializers are requested as bytes and placed in `DeviceBuffer`s. |
| Runtime I/O | `crates/onnx-runtime-session/src/executor.rs:1798-1806`, `1883-1918` | Inputs are copied from `Tensor::as_bytes`; outputs are reconstructed with `Tensor::from_raw`. |

The IR/protobuf layer is ahead of execution:

- ONNX specifies `string_data` elements as UTF-8, without NUL termination or BOM
  (`crates/onnx-runtime-loader/proto/onnx.proto3:711-716`), and forbids storing strings in
  `raw_data` (`crates/onnx-runtime-loader/proto/onnx.proto3:728-733`).
- Loader import fills `TensorData::strings`, but currently uses lossy UTF-8 conversion
  (`crates/onnx-runtime-loader/src/weights.rs:715-733`).
- Encoder export correctly selects `string_data` instead of `raw_data`
  (`crates/onnx-runtime-loader/src/encoder.rs:423-438`).
- `WeightStore::bytes` returns only `TensorData::data`
  (`crates/onnx-runtime-loader/src/weights.rs:552-571`), so a string initializer reaches the
  executor as an empty byte slice and its actual payload is not represented there.

The rejection is visible at kernels too:

- The raw EP view validator rejects strings because they have no fixed-width raw layout
  (`crates/onnx-runtime-ep-api/src/tensor.rs:86-127`).
- CPU byte movers reject any dtype with zero byte width
  (`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:1031-1042`).
- `Identity` explicitly refuses strings to avoid silent data loss
  (`crates/onnx-runtime-ep-cpu/src/kernels/identity.rs:30-43`).
- `Cast` explicitly refuses string sources and targets
  (`crates/onnx-runtime-ep-cpu/src/kernels/cast.rs:142-157`).

## 2. Representation options

### Option A: separate `Vec<String>` storage variant — recommended

`Vec<String>` guarantees that every element is a valid, initialized Rust string. It supports:

- direct safe access as `&str`;
- efficient element replacement for output kernels;
- straightforward shape/count validation (`values.len() == numel`);
- zero-copy borrowed logical views over the same owner;
- strict UTF-8 import without another decoding step;
- uncomplicated destruction and panic safety.

Costs:

- one `String` header per element and usually one heap allocation per non-empty element;
- worse locality than a packed arena;
- cloning the owned tensor deep-copies strings.

Those costs are acceptable for the first runtime slice. Sequence storage already shares immutable
whole tensors behind `Arc<Tensor>` (`crates/onnx-runtime-session/src/sequence.rs:163-207`), so
sequence insertion/selection does not require adding another `Arc` inside `TensorStorage`.
String movement views can likewise borrow the existing owner without copying.

Use `Vec<String>`, rather than `Arc<[String]>`, initially because runtime outputs are mutable while
kernels fill them. An inner `Arc<[String]>` would require copy-on-write or a second builder type for
every output. Whole-tensor sharing should remain at the owning/value layer.

### Option B: UTF-8 arena plus offsets

An arena representation would hold contiguous UTF-8 bytes plus `numel + 1` validated offsets:

```text
bytes:   [ all UTF-8 payload bytes ]
offsets: [ 0, end(element 0), end(element 1), ... ]
```

Advantages:

- compact metadata and payload;
- good cache locality;
- cheap immutable cloning with `Arc<[u8]>` / `Arc<[u32 or u64]>`;
- contiguous element-range slices can share the arena.

Disadvantages:

- every construction must validate monotonic offsets, terminal length, UTF-8 boundaries, and
  offset-width overflow;
- random element replacement changes subsequent offsets and usually rebuilds the arena;
- `Where`, scatter-like ops, `Cast` to strings, and tokenizer outputs need a separate builder;
- arbitrary strided views require indirection or repeated offset lookups;
- mutable `&str` views are impossible because string lengths may change.

The arena is a reasonable future internal variant for large immutable label tables or tokenizer
outputs, but adopting it now adds a second variable-length builder protocol before correctness and
interop are established. Keep accessors representation-neutral so `StringStorage::Arena` can be
introduced later without changing kernels.

## 3. Recommended storage and view model

### 3.1 Owned storage

Introduce a tagged storage representation at both the public owned-tensor boundary and the
executor value boundary:

```rust
// Illustrative only.
enum TensorStorage {
    Raw {
        buffer: DeviceBuffer,
        allocator: Arc<dyn ExecutionProvider>,
        import_guard: Option<Box<dyn Any + Send + Sync>>,
    },
    Strings(Vec<String>),
}
```

Required invariants:

1. `dtype == String` if and only if storage is `Strings`.
2. `Strings` is host-only (`DeviceId::cpu()` in the first implementation).
3. `Strings.len() == checked_numel(shape)`.
4. `Raw` rejects `String` and preserves all current fixed-width/sub-byte rules.
5. A scalar shape `[]` has one element; any shape containing a zero dimension has zero elements.
6. Shape multiplication is checked before allocation.
7. External/device borrowing constructors reject `String`; a foreign raw pointer is never accepted
   as string storage.

The executor should replace the assumption that every tensor value has a `DeviceBuffer` with a
storage enum (or an equivalent exhaustively matched pair of maps). Prefer the enum: it prevents a
dtype from being recorded separately from an incompatible storage class.

Do not combine this slice with a broad crate-hoist. The session and eager tensor implementations
are currently duplicated (`crates/onnx-runtime-eager/src/tensor.rs:1-10`). First establish the
storage/view contract with surgical changes; a later mechanical move to a shared
`onnx-runtime-tensor` crate can remove duplication.

### 3.2 Borrowed kernel views

Keep the current raw `TensorView` / `TensorMut` as byte/device views and continue rejecting
`String`. Add safe string-specific views:

```rust
// Illustrative only.
struct StringTensorView<'a> {
    values: &'a [String],
    shape: &'a [usize],
    strides: &'a [i64],      // in elements
    element_offset: usize,   // never a byte offset
}

struct StringTensorMut<'a> {
    values: &'a mut [String],
    shape: &'a [usize],
    strides: &'a [i64],
    element_offset: usize,
}
```

The read view exposes safe methods such as:

- `get(&[usize]) -> Result<&str>`;
- `get_flat(usize) -> Result<&str>`;
- `iter() -> impl Iterator<Item = &str>`;
- `is_contiguous()` and `numel()`.

The mutable view exposes `set(index, String)` / `clone_from(index, &str)`, not a `*mut String`.
Initially require writable string outputs to be contiguous and non-overlapping. Read-only string
views may be strided, including negative strides, after validation computes their reachable
minimum/maximum element indices using checked signed arithmetic.

At the kernel trait boundary, add a storage-aware wrapper while retaining the current raw-kernel
entry point as a default adapter:

```rust
// Illustrative only.
enum TensorValueView<'a> {
    Raw(TensorView<'a>),
    Strings(StringTensorView<'a>),
}

enum TensorValueMut<'a> {
    Raw(TensorMut<'a>),
    Strings(StringTensorMut<'a>),
}
```

The executor calls the storage-aware entry point. Its default implementation accepts only `Raw`
variants and delegates to the existing `Kernel::execute`, so existing numeric CPU/CUDA kernels do
not need an all-at-once rewrite. A string-aware kernel overrides the new entry point and obtains
only safe `&str`/owned-`String` access.

This should compose with the existing `KernelInput::{Tensor, Weight}` extension seam
(`crates/onnx-runtime-ep-api/src/kernel.rs:166-227`): broaden its tensor case to the
storage-aware view rather than creating a parallel dispatch stack.

### 3.3 Zero-copy views

Raw tensors retain DLPack-style byte offsets. String tensors use **element offsets**, never byte
offsets. Extend view-output metadata with a distinct string form rather than overloading
`ViewOutput::byte_offset` (`crates/onnx-runtime-ep-api/src/kernel.rs:143-164`).

For a string slice/reshape/transpose:

- keep the root `Vec<String>` owner alive and pinned for all consumers;
- record shape, element strides, and element offset;
- provide consumers a `StringTensorView`;
- materialize a contiguous `Vec<String>` only at an API boundary that requires ownership.

This supplies zero-copy views without exposing `String` object layout or pretending strings are
device bytes.

## 4. Dtype and sizing API changes

Add explicit predicates:

- `DataType::is_string()`;
- `DataType::has_raw_storage()` (or `is_raw_layout()`), true for fixed-width and packed sub-byte
  dtypes, false for `String`/`Undefined`;
- optionally `DataType::is_host_only()` for placement diagnostics.

Make byte sizing unambiguously raw-storage-only:

- add `checked_raw_storage_bytes(count) -> Option<usize>`;
- return `None` for `String` and `Undefined`;
- have `storage_bytes` panic or return an error for non-raw dtypes instead of returning `0`;
- migrate allocation and bounds callers to the checked form.

Do not use a string's UTF-8 payload byte total as its tensor element storage size. That number is
useful for serialization/metrics, but it does not define element addressing and must have a
different name such as `encoded_utf8_bytes()`.

General validation becomes:

```text
checked_numel(shape)
  Raw storage    => expected raw bytes == actual logical bytes
  String storage => expected elements  == strings.len()
```

`TensorView::byte_size()` must not report `0` for a non-empty string tensor. Keep it raw-only
(`Result`/`Option`) and add storage-neutral `numel()`.

## 5. Owned tensor API

### Constructors

- `Tensor::from_raw(...)`: explicitly reject `DataType::String`.
- `Tensor::from_strings(shape, Vec<String>)`.
- Convenience `Tensor::from_strs(shape, &[impl AsRef<str>])`, if desired.
- Internal `Tensor::empty_strings(shape)` for pre-sized kernel outputs
  (`vec![String::new(); numel]`).
- `from_borrowed_parts_with_guard`, device bindings, and raw C constructors reject string dtype.

All constructors validate checked element count before allocation and return an indexed diagnostic
for count mismatch.

### Accessors

- `as_strings() -> Result<StringTensorView<'_>>` or `Result<&[String]>` for contiguous tensors.
- `get_str(index) -> Result<&str>`.
- `to_vec_string() -> Result<Vec<String>>`.
- Internal `as_strings_mut()` only while exclusive ownership is held.
- `as_bytes`, `device_ptr`, raw typed-pointer access, and overwrite-bytes return a clear
  non-raw-storage error for strings.

Avoid panic-only dtype checks in new APIs. Errors should name the requested accessor, actual dtype,
and corrective accessor.

### Clone/drop/debug

- Clone `Raw` using the existing allocation/copy path.
- Clone `Strings` with ordinary safe Rust cloning.
- Drop `Raw` through the allocator/foreign guard ordering already documented in
  `crates/onnx-runtime-session/src/tensor.rs:504-519`.
- Drop `Strings` normally; no allocator or device pointer participates.
- Debug output may report element count and encoded UTF-8 bytes, but must not dump sensitive
  string payloads by default.

## 6. Proto import/export

The IR representation is already structurally suitable. Complete the path as follows:

1. Decode each `string_data` element with strict `String::from_utf8`.
2. On invalid UTF-8, fail model load with tensor name and element index. Do not use
   `from_utf8_lossy`; it makes load-save non-byte-exact and silently changes model semantics.
3. Validate `string_data.len() == checked_numel(dims)`, including scalar and empty tensors.
4. Materialize inline string initializers directly into `TensorStorage::Strings`; bypass
   `WeightStore::bytes`.
5. Export only through `TensorProto.string_data`, one UTF-8 byte vector per element.
6. Continue rejecting external string initializer encoding unless a spec-conforming external
   representation is designed; the current encoder already errors clearly
   (`crates/onnx-runtime-loader/src/encoder.rs:449-483`).

Add round-trip tests containing:

- empty strings and non-ASCII Unicode;
- embedded NUL bytes inside valid UTF-8 strings;
- scalar, zero-element, and multidimensional tensors;
- invalid UTF-8 rejection;
- element-count mismatch;
- load -> execute `Identity` -> save preservation.

## 7. DLPack and other interop

Strings are **not DLPack-representable**. Do not export a pointer to `String`, an offset table, or a
`uint8` surrogate: each would claim a DLPack tensor contract that consumers cannot interpret.

The Python DLPack exporter already rejects unsupported dtypes through `to_dldatatype`
(`crates/onnx-runtime-python/src/dlpack.rs:55-87`). Preserve and strengthen this behavior:

- error text should explicitly say `tensor(string)` has no DLPack dtype;
- `__dlpack_device__` must not imply that a string tensor is exportable;
- DLPack import can never construct `TensorStorage::Strings`;
- `device_ptr()` is unavailable for strings.

Other guards required:

- `DeviceIoBinding` creation rejects `String` before allocation
  (`crates/onnx-runtime-session/src/tensor.rs:139-171` is currently byte-sized).
- CUDA/other non-host EP placement rejects string values with an actionable host-only reason.
- CUDA graph capture rejects any graph region containing string storage.
- The raw C API `nxrt_create_tensor` rejects `String` and points callers to a future
  length-aware string constructor; it currently derives a zero expected byte count
  (`crates/onnx-runtime-capi/src/lib.rs:455-527`).
- Python copy input/output paths gain explicit string-list/object-array conversion or return a
  binding-specific error; they must never call `as_bytes`.
- Eager raw constructors/zero allocation reject `String` until their storage variant is wired.

## 8. Downstream operations unblocked

Do not implement these in the storage slice. Once storage-aware views exist, the following become
possible:

### Baseline tensor operations

- `Constant` and `Identity`;
- reshape/view family: `Reshape`, `Flatten`, `Squeeze`, `Unsqueeze`, `Transpose`, `Slice`;
- indexing/movement where allowed by the applicable ONNX schema: `Gather`, `GatherElements`,
  `GatherND`, `Concat`, `Split`, `Expand`, `Tile`, `Compress`;
- selection/comparison: string `Equal` and `Where`;
- `Cast` / `CastLike` string conversions with ONNX-defined formatting/parsing rules.

Register standard kernels in the CPU registry at
`crates/onnx-runtime-ep-cpu/src/kernels/mod.rs:220-270` and their later registration blocks.
Reuse existing shape-inference registrations where dtype-generic:

- data ops: `crates/onnx-runtime-shape-inference/src/handlers/data_ops.rs:233-237`;
- movement: `crates/onnx-runtime-shape-inference/src/handlers/movement.rs:1726-1733`;
- `Where`: `crates/onnx-runtime-shape-inference/src/handlers/elementwise.rs:214-216`.

Each kernel still must check its exact ONNX opset/type constraints; storage support alone does not
broaden an operator schema.

### String-semantic and tokenizer-adjacent operations

- standard-domain `StringNormalizer` and current string operators such as `StringConcat`,
  `StringSplit`, and `RegexFullMatch` where the selected ONNX opset defines them;
- `ai.onnx.ml::CategoryMapper` and `ai.onnx.ml::LabelEncoder` for label maps;
- `com.microsoft` tokenizer/string contrib operators used by exported text pipelines.

Place their CPU factories in a new focused `kernels/string.rs` (or per-op modules) and register by
their exact domain/opset in `build_cpu_registry`. Add corresponding handlers in a focused
shape-inference `handlers/string.rs`. Do not register string kernels on CUDA until a deliberate
device representation exists.

## 9. Phased implementation plan

### Phase 1 — storage variant, validation, accessors, and tests

1. Coordinate with and rebase onto the completed `wt-seq-zerocopy` tensor-layer work.
2. Add `DataType` raw-layout/string predicates and raw-only checked sizing.
3. Add `TensorStorage::{Raw, Strings}` to the session-owned tensor; mirror the invariant in eager
   or make eager reject strings explicitly until migrated.
4. Make executor values storage-aware, including input binding, output collection, cloning,
   control-flow transfers, and sequence element handling.
5. Add `StringTensorView`/`StringTensorMut` and a storage-aware kernel adapter.
6. Add constructors/accessors and strict shape/element-count validation.
7. Test scalar/empty/multidimensional construction, clone/drop, wrong counts, raw-access
   rejection, safe views, and default-kernel rejection.

Exit criterion: a host `Tensor::from_strings` can enter and leave a session storage boundary
without byte casts, even before semantic string kernels are registered.

### Phase 2 — ONNX proto initializer import/export

1. Replace lossy UTF-8 import with indexed strict errors.
2. Validate string initializer element counts.
3. Route inline string initializers directly to string runtime storage.
4. Preserve `string_data` on export; retain the external-string error.
5. Add load/save and load/run/save round-trip tests.

Exit criterion: valid UTF-8 string initializers survive model round trips exactly and invalid
payloads fail at load.

### Phase 3 — DLPack and other clear-error guards

1. Make DLPack export/import errors explicitly identify the missing string representation.
2. Reject strings in device bindings, raw borrowed-memory constructors, CUDA placement/capture,
   raw C tensor construction, and unsupported eager/Python byte paths.
3. Audit every `storage_bytes`, `as_bytes`, `device_ptr`, and typed-pointer caller for exhaustive
   storage matching.
4. Add negative tests proving no path can create a raw `String` view.

Exit criterion: every unsupported interop path fails before allocation/pointer exposure with a
clear message; no non-empty string tensor can be mistaken for zero bytes.

### Phase 4 — first string operations

1. Land `Constant`/`Identity` and basic view/movement operations first.
2. Add `Equal` and `Where`.
3. Add ONNX-conformant `Cast`/`CastLike` string conversions.
4. Add label-map operations (`CategoryMapper`, `LabelEncoder`), then normalization/string ops.
5. Add tokenizer contrib operators last, with model fixtures and end-to-end registration tests.

Exit criterion: representative tokenizer-adjacent and label-map models run with safe string I/O,
while unsupported GPU/DLPack routes remain explicit errors.

## 10. Sequence zero-copy dependency and shared files

Implementation must not begin by independently editing the tensor layer while
`wt-seq-zerocopy` is active. Both changes alter ownership, view lifetime, materialization, and
executor value classification.

Shared/high-conflict files are:

- `crates/onnx-runtime-session/src/tensor.rs`;
- `crates/onnx-runtime-session/src/executor.rs`;
- `crates/onnx-runtime-session/src/sequence.rs`;
- `crates/onnx-runtime-ep-api/src/tensor.rs`;
- `crates/onnx-runtime-ep-api/src/kernel.rs`;
- likely `crates/onnx-runtime-eager/src/tensor.rs` if tensor APIs are synchronized.

Merge/rebase the Sequence slice first, then adapt its `SeqTensor` and zero-copy element-view paths
to match exhaustively on `TensorStorage`. Preserve its `Arc<Tensor>` sharing model; do not copy
strings merely because sequence elements are extracted. Resolve view metadata deliberately:
Sequence tensor-element aliases, raw byte views, and string element-offset views must remain
distinct storage cases.

## 11. Safety gate

The implementation is not complete unless a repository search confirms there is no
`data_ptr::<String>`, `as_ptr() as *const String`, `from_raw_parts::<String>`, or equivalent cast
from byte/device storage. The only sources of `&str` must be initialized `String` values owned by
the string storage variant or strictly validated UTF-8 serialization bytes during import.
