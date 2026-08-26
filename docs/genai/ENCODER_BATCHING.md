# Generalized encoder batching metadata

Status: **metadata and bounded grouped-runtime primitives implemented; automatic
serving-queue batching remains open**.

This is the design of record for batching independent encoder/media work through
workflow components. The v1.1 schema, parser migrations, semantic validation,
grouped image/frame preprocessing, symbol-keyed admission, packed ownership
composition/splitting, and ORT component invocation are implemented. The
runtime does not yet automatically collect unrelated live requests into encoder
queues or decode encoded video containers.

The normative schema rules also appear in
[`INFERENCE_METADATA_DECISIONS.md`](INFERENCE_METADATA_DECISIONS.md), especially
§8 and §10. [`PIPELINE.md`](PIPELINE.md) summarizes the runtime boundary.

## 1. Compact conceptual model

One declaration answers one question:

> May this component execute several independent contributions in one call
> without changing any contribution's result, and what shapes may that call
> materialize?

The answer has four independent parts:

1. **Permission and bounds — `WorkflowComponent.batch_capacity`.**
   Absence means per-item/per-request execution. Presence is the component
   author's assertion that every group satisfying the declared contracts is
   semantically equivalent to executing the contributions separately.
2. **Dense row ownership — `request_aligned` or `request_expanded`.**
   A tensor carries one row, or a fixed number of physical rows, per request.
3. **Ragged extent — `TensorContract.padding`.**
   A dense rectangle may contain right padding. A named `valid_lengths`
   companion is the single contract-level truth for the valid prefix.
4. **Ragged ownership — `token_packed`.**
   Variable counts are flattened onto one physical axis. One or two ownership
   levels map positions → items → request rows.

These are modality-neutral. Image patches, video frames, audio frames, and text
tokens differ only in semantic names and preprocessing operations.

### One physical packed axis

`token_packed` always packs axis 0. Nesting adds bookkeeping levels, not tensor
axes:

```text
packed positions --level 0--> items --level 1--> request rows
frames                         clips              requests
frames                         windows            requests
tokens                         segments           requests
```

A one-level chain maps positions or items directly to rows. A two-level chain is
the deepest currently accepted form.

### Three kinds of raggedness

| Question | Representation |
| --- | --- |
| How many contributions does a request own? | Outermost ownership level |
| How large is one contribution along a dense dimension? | Right padding plus `valid_lengths`, or packing that dimension |
| Does a contribution contain variable-count children? | Additional ownership level over the same packed axis |

No field combines these questions, and no modality-specific batching flag is
needed.

## 2. What is authored, derived, and runtime policy

### Authored package facts

- `batch_capacity` presence;
- `uniform_dimensions`, keyed by shape symbol;
- materialized-footprint `budgets`, also keyed by shape symbol;
- each tensor's `batch_layout`;
- each padded dimension and its `valid_lengths` value;
- each packed ownership pair;
- `extent: preserved | produced` on every packed component output level;
- optional-input presence and whether an encoder result is
  `externally_suppliable`.

### Derived structural facts

- a component is eligible for grouping only when `batch_capacity` is present;
- two contributions are compatible when every authored uniform symbol agrees,
  every differing dimension has one declared reconciliation, and all budgets
  fit;
- request-local spans follow by composing ownership offsets;
- a packed or padded workflow output must publish the companions needed to read
  it;
- an owner map is runtime-internal, while offsets and lengths can be sliced or
  rebased into request-local views;
- the runtime's decoder batching report is derived from the resolved graph/state
  ABI and backend, not copied from metadata.

### Runtime policy

- whether to group eligible work at all;
- how long to wait and what target group size to choose;
- device-memory limits, placement, and execution provider;
- backend-readiness evidence;
- latency/throughput trade-offs, queueing, cancellation, and fairness.

Static artifact limits belong in metadata. Measured memory availability and
throughput preferences do not.

### Retired duplicate concepts

This design has one authored grouping claim: `batch_capacity`.

- `profiles.*.batch_invariance` is retired. It was profile-wide, could conflict
  with a component declaration, and allowed the nonsensical state
  "`padding_sensitive` but batchable." If co-batching changes the answer, omit
  `batch_capacity`.
- `model.runtime_configurable.continuous_batching` is retired. Structural
  feasibility is derived; enablement and width are deployment policy.
- the built-in `continuous_batching` capability identifier is retired.
  Continuous batching is an optimization, not behavior required for correct
  single-request execution.

The parser recognizes all three old spellings and returns migration errors.
There is no replacement flag and no new `encoder_batching` capability.

## 3. Canonical declarations

### 3.1 Per-item encoder

Omit `batch_capacity`:

```yaml
components:
  encoder:
    implementation: { kind: onnx, artifact: encoder.onnx }
    ports:
      inputs:
        input:
          dtype: float32
          rank: 3
          shape: [batch, sequence, hidden]
          batch_layout: { kind: request_aligned, axis: 0 }
```

`request_aligned` still describes the tensor. It does not grant permission to
coalesce independent work. The safe interpretation is one request contribution
per invocation.

### 3.2 Dense batch with right padding

```yaml
components:
  image_encoder:
    implementation: { kind: onnx, artifact: encoder.onnx }
    batch_capacity:
      uniform_dimensions: [height, width]
      budgets:
        - { dimensions: [batch], max_total: 8 }
        - { dimensions: [batch, max_tiles], max_total: 64 }
    ports:
      inputs:
        pixels:
          dtype: float32
          rank: 4
          shape: [batch, max_tiles, height, width]
          batch_layout: { kind: request_aligned, axis: 0 }
          padding:
            - { dimension: max_tiles, valid_lengths: tile_lengths }
        tile_lengths:
          dtype: int64
          rank: 1
          shape: [batch]
          batch_layout: { kind: shared }
```

The `batch` symbol roots both budgets in the assembled group. The second budget
charges the materialized rectangle `batch × max_tiles`, not the sum of valid
tile counts.

`valid_lengths` is contract provenance, not necessarily the graph's mask. If the
graph consumes a bool or additive mask, that mask is an ordinary typed port
produced by preprocessing. The runtime does not infer lengths from the mask or
the mask from sentinel values.

### 3.3 Packed nested video

```yaml
components:
  video_encoder:
    implementation: { kind: onnx, artifact: encoder.onnx }
    batch_capacity:
      uniform_dimensions: [features]
      budgets:
        - { dimensions: [clips], max_total: 4 }
        - { dimensions: [frames], max_total: 64 }
        - { dimensions: [frames, patches], max_total: 65536 }
    ports:
      inputs:
        pixel_values:
          dtype: float32
          rank: 3
          shape: [frames, patches, features]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: frame_offsets, owner: frame_owner }
              - { offsets: clip_offsets, owner: clip_owner }
          padding:
            - { dimension: patches, valid_lengths: patch_lengths }
        patch_lengths:
          dtype: int64
          rank: 1
          shape: [frames]
          batch_layout: { kind: shared }
        frame_offsets:
          dtype: int64
          rank: 1
          shape: [clips_plus_one]
          batch_layout: { kind: shared }
        frame_owner:
          dtype: int64
          rank: 1
          shape: [frames]
          batch_layout: { kind: shared }
        clip_offsets:
          dtype: int64
          rank: 1
          shape: [rows_plus_one]
          batch_layout: { kind: shared }
        clip_owner:
          dtype: int64
          rank: 1
          shape: [clips]
          batch_layout: { kind: shared }
```

Frames are physically contiguous on axis 0. Level 0 maps frames to clips and
level 1 maps clips to rows. Patch count remains a separate padded dimension.

### 3.4 Output extents

Every packed **component output** states the origin of each level's extent:

```yaml
outputs:
  frame_features:
    dtype: float32
    rank: 2
    shape: [frames, hidden]
    batch_layout:
      kind: token_packed
      axis: 0
      levels:
        - { offsets: frame_offsets, owner: frame_owner, extent: preserved }
        - { offsets: clip_offsets, owner: clip_owner, extent: preserved }

  media_tokens:
    dtype: float32
    rank: 2
    shape: [media_tokens, hidden]
    batch_layout:
      kind: token_packed
      axis: 0
      levels:
        - { offsets: token_offsets, owner: token_owner, extent: produced }
        - { offsets: clip_offsets, owner: clip_owner, extent: preserved }
```

- `preserved` reuses an input ownership pair one-for-one.
- `produced` requires the component to output the pair because the graph chose
  the count.

The declaration is per level because a token-merging encoder commonly produces
the inner extent and preserves the outer request ownership.

## 4. Validation shipped on `main`

The metadata crate currently rejects:

- a batching field used below schema v1.1;
- the retired flat `token_packed` `{ offsets, owner }` spelling;
- unknown or duplicated capacity symbols and footprint paths;
- a budget whose first symbol is not a group count;
- an unbudgeted input ownership level;
- a uniform symbol that is itself a packed or ownership count;
- a free input dimension with neither packing, padding, nor uniformity;
- an invalid packed axis or more than two ownership levels;
- missing, aliased, mis-typed, mis-ranked, or inconsistently shaped companions;
- a CTC profile without the canonical `outputs.logits` role, or missing or
  contradictory CTC frame lengths when logits pad the decoded time axis (the
  decoder role must resolve to that padding entry's exact companion);
- a packed axis that is also padded;
- a padded dimension with two validity truths;
- a packed output level with missing or contradictory `extent`;
- a serving output that withholds a packed or padded companion;
- a declared companion output that no step emits;
- a padded output also trimmed with `Emit.valid_length`;
- an externally supplied owner map.

Load-time validation checks declarations. Invocation-time runtime validation now
checks monotonic offsets, terminal extents, owner lengths/ranges/order, logical
request cardinality, and materialized capacity budgets before backend enqueue.
The grouped preprocessor emits one valid length per physical image/frame and
refuses unresolved empty-tensor dimensions rather than inventing extents.

## 5. Current implementation status

| Layer | Current on `main` |
| --- | --- |
| Schema and JSON schema | Implemented: v1.1 `batch_capacity`, `padding`, ownership levels, `extent`, and vision/audio companion roles |
| Parser and migrations | Implemented: version gate, flat packed migration, and retired batching-hint errors |
| Semantic validator | Implemented: structural rules listed above, including exact padded-CTC length binding |
| Metadata tests | Implemented for dense padding, one/two-level packing, video declarations, budgets, extents, serving companions, padded and unpadded CTC decoding, and modality-neutral audio/text renamings |
| Image preprocessing schema | Implemented |
| Video preprocessing schema | Implemented with the shared vision program and `sample_frames`/`pad_frames` vocabulary |
| Audio preprocessing schema | Implemented |
| Text preprocessing schema | No dedicated program; typed application/runtime tensors can use the same contracts |
| Runtime image preprocessing | Workflow adapter accepts one encoded image; `GroupedVisionPreprocessor` accepts ordered multi-image request rows and emits padding/ownership companions |
| Runtime audio preprocessing | Existing adapter accepts one encoded audio item per workflow invocation |
| Runtime video preprocessing | Ordered pre-extracted frame images are decoded individually by the grouped vision program, producing tensors with frames → clips → requests ownership; encoded containers/temporal sampling return typed `UnsupportedExecution`, including an ordered image-then-video mixed request |
| Packed/padded group assembly and split | Implemented for axis-0 ownership depth 1–2; companions are rebuilt, request views rebase metadata, and payload views alias without copying |
| Encoder scheduling across requests | `group_workflow_component_inputs` deterministically groups request-local contributions under uniform dimensions and materialized budgets; automatic serving queues/materialization remain open |
| Backend readiness | ORT grouped invocation is covered by deterministic image/video fixtures; packed/padded native workflow execution fails closed with `UnsupportedNativeEncoderBatch` |

Declaring `batch_capacity` is enforced at the component boundary. It is not an
instruction to queue automatically: a caller/backend packer still chooses when
to materialize an admitted group.

## 6. Modality acceptance matrix

“Represented” means the schema and validator can faithfully state the case.
“Runtime slice” means the bounded preprocessing/admission/ORT/split APIs are
implemented; it does not mean automatic serving-queue collection.

| Modality | Dense/padded form | Packed form | Nested form | Represented | Runtime slice |
| --- | --- | --- | --- | --- | --- |
| Image | request rows, padded tiles/patches | images → rows | normally unnecessary | yes | yes, encoded images through grouped API + ORT |
| Video | request rows, padded frames/patches | clips → rows or frames → rows | frames → clips → rows | yes | yes for ordered encoded frame images; no encoded-container decoder |
| Audio | request rows, padded samples/frames | windows/frames → rows | frames → windows → rows | yes | contracts/admission only |
| Text encoder/reranker | request rows, padded tokens | segments/tokens → rows | tokens → segments → rows | yes | contracts/admission only |

Per-item execution is represented for every modality by omitting
`batch_capacity`.

## 7. Required edge-case behavior

The current bounded runtime behavior is:

| Case | Metadata representation | Current runtime behavior |
| --- | --- | --- |
| Empty media for one request | repeated outer offsets and no owner entries for that span | preserved exactly, including empty clips inside a nonempty video request |
| Every request has zero new media | packed extent and owner vectors are empty | empty ranked tensors execute on the ORT fixture and return empty output; no prior payload/companions are reused |
| Mixed media requests | separate image/video/audio/text components and queues | no ordered mixed-media grouping surface yet; encoded video and temporal transforms return typed rejection rather than image reinterpretation |
| Decode with zero new media | optional-input presence or an empty span; cached encoder state may be externally supplied | a zero-new-media turn produces fresh empty inputs/outputs and cannot observe a prior media-bearing turn |
| Cancellation/compaction | request ownership chain plus runtime row selection | lift selection through ownership, rebuild companions, and prevent cross-request leakage |
| Packed serving output | output plus every offsets/owner companion | `PackedTensor::request_view` returns a contiguous no-copy payload alias with offsets/owners rebased |
| Padded serving output | output plus every `valid_lengths` companion | `PackedOwnership::slice_lengths` returns the request-local magnitude slice without rebasing |

An owner map is not request data. The validator forbids
`externally_suppliable` owner maps because their positions exist only after the
runtime forms a group.

## 8. Budgets and uniformity

`uniform_dimensions` names per-item properties that must agree within one call.
Examples are channels, hidden width, mel bins, or a fixed frame count.

Budgets name materialized extents:

- a request-aligned row count;
- a packed position count;
- an ownership-level item count;
- a product rooted in one of those counts, such as
  `[batch, max_tiles]`, `[frames, patches]`, or
  `[clips, frames, patches]`.

A pinned dimension may appear inside a composed budget. Pinning means equal
within one group, not globally constant. A singleton budget on a per-item
dimension is rejected because it does not bound the assembled invocation.

Budgets are artifact correctness limits. Device bytes, allocator pressure, and
preferred occupancy remain runtime measurements.

## 9. Fail-closed backend readiness

`batch_capacity` says the artifact's contract permits grouping. It does not
prove that every backend/provider implementation is ready.

The ORT workflow path has fixture-level evidence for grouped axis-0 image and
nested-video encoder calls. Native workflow execution deliberately rejects a
multi-request component carrying packed or padded contracts before its native
session runs. It returns `UnsupportedNativeEncoderBatch`; it never silently
routes the request through ORT.

Backend readiness is intentionally not metadata:

- it changes with runtime version, provider, and hardware;
- it is evidence about an implementation, not a model-package semantic;
- declining to group preserves correctness.

The existing engine `BatchingCapability` remains decoder-focused. Generalized
encoder grouping currently exposes admission and an explicit native refusal,
not a provider-wide readiness registry.

## 10. Runtime work intentionally left open

The following remain open:

1. multi-item workflow adapter inputs (the grouped image/frame API is currently
   called before the workflow boundary);
2. encoded video-container decode, frame sampling, and temporal padding;
3. automatic non-decoder serving queues and backend payload materialization;
4. ordered mixed-media orchestration across image/video/audio components;
5. cancellation and compaction lifting through ownership levels;
6. native/provider readiness evidence sufficient to remove the fail-closed
   packed/padded native guard;
7. generalized audio/text preprocessing and ORT E2E fixtures;
8. production-model and production-size performance evidence.

The current benchmark is intentionally fixture-level only; see
[`../benchmarks/2026-08-25-media-batching-fixture.md`](../benchmarks/2026-08-25-media-batching-fixture.md).
These remaining changes do not require another metadata capability or policy
field.

## 11. Definition of done for runtime branches

For each of image, video, audio, and text, runtime work is accepted only when:

- a batchable fixture and an otherwise identical per-item fixture produce equal
  request-local results;
- incompatible uniform dimensions split into separate groups;
- dense padding is charged by materialized footprint;
- one- and two-level packed companions are validated and round-trip;
- empty, mixed-media, and zero-new-media cases do not fabricate work;
- preserved and produced output levels split at the declared boundaries;
- serving returns request-local offsets/lengths;
- an unproven backend remains on the per-item path;
- no payload incurs a hidden device → host → device round trip merely to group
  or split.
