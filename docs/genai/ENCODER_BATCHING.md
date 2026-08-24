# Generic component batching

Status: **proposed** — this document is the design of record for batching
independent work items through one workflow component. Nothing here is
implemented yet; [§2](#2-what-exists-today-and-what-is-missing) states exactly
what the repository does today, and every proposed field is marked as such
until it lands with the validator rules and tests in
[§8](#8-execution-phases-and-pr-dag).

The motivating workloads are **image and video encoders**: a request may carry
zero, one, or many images or video clips, items from different requests arrive
independently, and running each item as its own session call leaves an
encoder-shaped GPU almost idle. Video sharpens every part of the problem —
a clip is itself a variable-length group of frames, clips differ in frame count,
frames differ in resolution, and one clip is large enough that the group size a
runtime can afford is small and therefore has to be chosen from a declared bound
rather than guessed.

**Nothing in this design is about vision.** An audio encoder (windows of frames),
a text encoder or reranker (segments of tokens), a safety classifier, a codec
analyzer, and a diffusion conditioner all pose the identical question — *may the
runtime put several independent items into one invocation of this component, and
if so under what structural conditions?* — so the answer is one declarative fact
on a component, not a modality feature.

The split this document holds to throughout: **contracts are modality-agnostic,
and modality vocabularies only produce semantic values.** An axis is an axis to
the interpreter and the scheduler; whether it counts frames, patches, mel bins,
or tokens is something only a preprocessing program's content roles say, and
nothing in the batching path reads them.

Read [`RULES.md`](../../RULES.md) first. Three rules shape every decision below:
**Rule 2** (no model, vendor, or EP identity conditionals — behavior is driven by
metadata and declared capability), **Rule 4** (explicit, inspectable behavior;
unsupported configurations fail clearly instead of silently changing semantics),
and **Rule 10** (reduce entropy; two ways to say the same thing are duplicated
state).

The normative contract this extends is
[`INFERENCE_METADATA_DECISIONS.md`](INFERENCE_METADATA_DECISIONS.md) —
[§8](INFERENCE_METADATA_DECISIONS.md#8-batching-varlen-and-paged-attention)
(batch layouts and row semantics),
[§10](INFERENCE_METADATA_DECISIONS.md#10-multimodal-encoders) (encoders), and
[§15](INFERENCE_METADATA_DECISIONS.md#15-preprocessing-and-generated-inputs)
(preprocessing programs).

---

## 1. The question a package must be able to answer

A runtime that wants to co-batch two independent items needs five facts, and it
can obtain none of them today:

1. **May this component see more than one item per invocation at all?** Some
   graphs are written for exactly one item; some carry per-invocation state; some
   are shape-pinned by an exported constant.
2. **Along which axis do items stack, and how many fit?** The bound is a
   *correctness* property of the artifact, not a tuning knob. For video it is
   also the difference between a group that fits in memory and one that does not.
3. **Which axes must agree before two items may share an invocation?** A channel
   count must match; a patch count usually must not have to; a frame rate or a
   resolution may or may not, and only the package knows which.
4. **When extents differ on a free axis, how is the difference expressed** — by
   padding to a rectangle with a mask that says which entries are real, or by
   packing the items end to end with offsets and an owner map? A video item needs
   this answered on more than one axis at once: frames per clip *and* patches per
   frame.
5. **When an item is itself a group, who owns what?** A clip owns frames, an
   audio window owns spectrogram frames, a text segment owns tokens. Splitting a
   grouped result back to requests needs the frames-to-clip map as well as the
   clips-to-row map.

Answer (1)–(5) structurally and grouping becomes a generic runtime service, in
the same way continuous batching is generic over decoders. Answer them with a
model-family branch and the runtime has grown exactly the identity conditional
Rule 2 forbids.

---

## 2. What exists today, and what is missing

### 2.1 The vocabulary exists

`preprocessing.image` and `preprocessing.audio` are complete declarative
programs: an ordered, parameterized transform list plus named outputs with
content roles and full tensor contracts
(`crates/onnx-genai-metadata/src/schema/pipeline.rs:42-273`, vocabularies at
`crates/onnx-genai-metadata/src/schema/mod.rs:274-303`). A program already emits
per-item geometry — `grid_dimensions`, `patch_coordinates`, `original_size`,
`validity_mask` — as ordinary typed tensors, and the image preprocessor library
already accepts **many images in one call**
(`ImagePreprocessor::preprocess_encoded` /
`preprocess`, `crates/onnx-genai-preprocess/src/image/program.rs:423,447`,
bounded by `MAX_IMAGE_COUNT`).

`BatchLayout::TokenPacked { offsets, owner, axis }` exists in the schema
(`crates/onnx-genai-metadata/src/schema/ir.rs:33-59`) and is documented as the
encoder layout in
[§10.1](INFERENCE_METADATA_DECISIONS.md#101-independent-encoder-batching).

### 2.2 Component-level batching capability does not exist

`WorkflowComponent` (`crates/onnx-genai-metadata/src/schema/ir.rs:697-724`)
declares `implementation`, `ports`, `contract`, `application_overridable`,
`effects`, `row_scope`, and `cache_affects_state`. **There is no field that says
whether the component tolerates more than one item per invocation**, no upper
bound, and no statement of which axes must agree between co-batched items. There
is no `supports_batching` anywhere in the metadata crate either — which is
correct, because a boolean would be the wrong shape (see
[§7](#7-what-this-deliberately-does-not-add)) — but the consequence is that a
runtime has nothing to consult.

The closest existing construct, `BatchingCapability`
(`crates/onnx-genai-engine/src/batched.rs:120-171`), is an engine-side *decode
path* capability derived from a decoder's KV ABI. It says nothing about
encoders, is not a per-component metadata fact, and `ContinuousBatchManager`
only admits token prompts — multi-row prompts are refused outright
(`batched.rs:330-342`), and the file contains no reference to images, vision, or
encoders at all.

### 2.3 `token_packed` has no runtime

Outside the metadata crate and its own tests, **nothing consumes `token_packed`**.
The engine, scheduler, server, and ORT crates contain no reference to it. The
layouts the interpreter actually honors are `request_aligned` and
`request_expanded`, through `BatchLayout::request_axis()`, at three call sites:
row-wise emit dispatch, symbolic-shape validation, and inactive-row merging
(`crates/onnx-genai-engine/src/pipeline/workflow.rs:2152,4181,4846`).
`request_axis()` deliberately returns `None` for `TokenPacked`
(`ir.rs:67-72`), so a packed value is invisible to every one of them. It is a
declared shape with no reader.

### 2.4 The image adapter is single-item by construction

`run_image_preprocess_adapter`
(`crates/onnx-genai-engine/src/pipeline/workflow.rs:3020-3116`) validates one
`uint8` rank-1 `encoded` input and calls `preprocess_encoded([encoded_bytes])`
with a one-element array (`workflow.rs:3084`). The multi-image capability of the
preprocessor underneath is unreachable from a workflow. Each `Invoke` runs one
session call over whatever is bound to its ports
(`workflow.rs:1713-1927`, `invoke_onnx_component` at `workflow.rs:2535`), so an
encoder invocation carries exactly one request's payload.

### 2.5 The scheduler groups decode rows only

`crates/onnx-genai-scheduler/src/lib.rs` admits requests, orders them, preempts
under memory pressure, and forms decode batches; `Scheduler::schedule()` returns
`{ prefill, decode, preempt, swap_in }` (`lib.rs:450-508`). There is no notion of
a pending non-decoder work item, so nothing in the system is even in a position
to decide that two encoder items should run together.

### 2.6 Validation of packed values is close to absent

The only rule in `crates/onnx-genai-metadata/src/validation.rs` that mentions
`token_packed` is the serving-emit rule (`validation.rs:3306-3320`), which
requires a per-request emitted value to declare *either* `request_aligned` *or*
`token_packed` and never distinguishes them. Unchecked today: that `offsets` and
`owner` name declared values at all; their dtype, rank, and layout; that `axis`
is within the packed value's rank; and that two packed values sharing one
`offsets` agree about the packed extent.

The cost of that gap is already visible. The schema doc comment calls `offsets`
a "request-aligned value" (`ir.rs:52-53`), while the canonical fixture declares
`image_offsets` as `shared`
(`crates/onnx-genai-metadata/tests/redesign_invariants.rs:158-169`) and
[§10.1](INFERENCE_METADATA_DECISIONS.md#101-independent-encoder-batching)
describes it as "cu_seqlens-style prefix offsets", which is a length-`rows + 1`
vector. Two of those three cannot be true at once, and nothing rejects either
spelling. [§4](#4-strict-token_packed-validation) resolves it.

### 2.7 Video is expressible on the output side and absent on the encoder side

Video *generation* already works and is tested. `WorkflowOutputRole::Video` is a
distinct output role precisely because "a consumer has to know the value carries
a temporal axis and may be published incrementally"
(`crates/onnx-genai-metadata/src/schema/ir.rs:636-643`); `Emit.axis` exists so
incremental publication grows the right axis, and its own doc names video frames
in `[batch, channels, frames, height, width]` as the reason
(`ir.rs:905-915`); there is a canonical `video` workflow fixture
(`tests/fixtures/onnx_genai_workflows/video`) and a conformance test that
publishes causal temporal chunks and runs it at batch 2
(`crates/onnx-genai-engine/tests/onnx_genai_workflow_conformance.rs:900-960`).

Video *encoding* has none of that:

- **No frame-sequence producer.** There is no `preprocessing.video` program.
  The image program's temporal parameter is degenerate by definition —
  `temporal_patch_size` is "Number of identical temporal frames packed into each
  spatial patch" (`crates/onnx-genai-metadata/src/schema/pipeline.rs:180-183`),
  and the preprocessor implements exactly that, replicating one still image
  across the temporal extent of a patch
  (`crates/onnx-genai-preprocess/src/image/packed.rs:261,603`).
  `ImagePreprocessor::preprocess` takes `&[DynamicImage]`
  (`crates/onnx-genai-preprocess/src/image/program.rs:447`), so a real frame
  sequence cannot be handed to it at all.
- **No temporal validity vocabulary on the image side.** The audio vocabulary
  has `valid_frames`, `frame_lengths`, and `validity_mask`
  (`crates/onnx-genai-metadata/src/schema/mod.rs:349-364`); the image vocabulary
  has a single spatial `validity_mask` (`mod.rs:293-303`) and nothing temporal.
- **No nesting.** `BatchLayout::TokenPacked` gives one `offsets`/`owner` pair per
  packed axis, and nothing composes two levels, so "frames belong to clips belong
  to rows" has no declared spelling even though both halves of it are ordinary
  packing.

So the video gap is not a missing modality feature. It is the *same* three
missing generic facts — a batching capacity, a per-axis padding truth, and a
per-level ownership map — observed on a workload that cannot avoid needing all
three at once.

**Summary of the gap:** the tensor vocabulary is there, the packed *shape* is
there, and neither a component-level batching capability nor a `token_packed`
runtime exists. Video additionally has no encoder-side producer for the values
either would consume. This design adds the first, specifies the second, and
states the minimum a modality vocabulary must produce for the third.

---

## 3. The proposed metadata surface

Three additions. Each is a structural fact about an artifact, each is absent by
default, and each is meaningless to a runtime that chooses not to group.

They exist because grouping faces **three independent kinds of raggedness**, and
conflating them is how a batching layer gets subtly wrong answers:

- **Item count** — how many items each request owns (zero, one, or many images;
  zero, one, or many clips). Answered by `token_packed`'s `offsets` and `owner`
  along the item axis.
- **Item extent** — how far two items differ along a shape axis: patches per
  image, frames per clip, spectrogram frames per window, tokens per segment.
  Answered by a padded value plus a mask on that axis, or by packing that axis
  too.
- **Item nesting** — an item that is itself a group, whose parts must be
  attributable after the group is split: frames within a clip, windows within an
  utterance. Answered by a second `offsets`/`owner` pair on the inner axis.

A single video encoder input needs all three at once — clips from three requests,
each clip with a different frame count, each frame with a different patch count —
so each is declared on its **own axis** of the same contract and no two ever
compete for one. An image encoder is the same declaration with the temporal axis
absent; an audio or text encoder is the same declaration with the axes meaning
frames or tokens. The contract never says which.

### 3.1 `WorkflowComponent.batch_capacity`

The example below is a **video** encoder, because it is the case that exercises
every field; [§3.4](#34-worked-cases-images-video-audio-text) reduces it to the
image, audio, and text shapes by deleting axes.

```yaml
components:
  clip_encoder:
    implementation: { kind: onnx, artifact: encoder.onnx }
    batch_capacity:
      axis: 0                # axis along which independent items (clips) stack
      max_rows: 4            # correctness bound of the artifact
      uniform_axes: [3]      # feature width must agree; frames and patches need not
    ports:
      inputs:
        pixel_values:
          dtype: float32
          rank: 4                        # [items, frames, patches, features]
          shape: [items, frames, patches, features]
          batch_layout: { kind: token_packed, offsets: item_offsets,
                          owner: item_owner, axis: 0 }
          pad_mask:
            - { value: frame_mask, axes: [1] }   # clips differ in frame count
            - { value: patch_mask, axes: [2] }   # frames differ in patch count
        frame_mask:
          dtype: bool
          rank: 2
          shape: [items, frames]
          batch_layout: { kind: token_packed, offsets: item_offsets,
                          owner: item_owner, axis: 0 }
        patch_mask:
          dtype: bool
          rank: 3
          shape: [items, frames, patches]
          batch_layout: { kind: token_packed, offsets: item_offsets,
                          owner: item_owner, axis: 0 }
        item_offsets:
          dtype: int64
          rank: 1
          shape: [rows_plus_one]
          batch_layout: { kind: shared }
        item_owner:
          dtype: int64
          rank: 1
          shape: [items]
          batch_layout: { kind: shared }
      outputs:
        media_features:
          dtype: float32
          rank: 3                        # [items, tokens, hidden]
          shape: [items, tokens, hidden]
          batch_layout: { kind: token_packed, offsets: item_offsets,
                          owner: item_owner, axis: 0 }
```

A component whose every request contributes at most one item declares
`request_aligned` on the same axis instead, and needs no `offsets`/`owner` pair
at all — item position *is* row position there.

Semantics:

- **Absent means one request row per invocation.** That is today's behavior
  exactly, so every existing package is unaffected and no existing document has
  to be rewritten. Absence is a statement, not a silence: the runtime **MUST NOT**
  group a component that has not declared a capacity.
- **`axis`** is the axis along which independent items stack. It **MUST** be
  within the rank of every non-`shared` port of the component, and it **MUST**
  equal the request axis of every `request_aligned` port and the packed axis of
  every `token_packed` port. `shared` ports do not participate; they are
  broadcast unchanged to the whole group.
- **`max_rows`** is the largest number of items the artifact tolerates in one
  invocation. It is *static geometry* — a fixed position table, an exported
  constant, a kernel bound — never a measured throughput sweet spot. Metadata
  never carries benchmark-derived cost models
  ([§1.2 non-goal 4](INFERENCE_METADATA_DECISIONS.md#12-non-goals)). It is an
  upper bound and never an obligation: a runtime **MAY** group fewer, including
  one.
- **`uniform_axes`** lists the axes whose extent **MUST** be equal across every
  co-batched item. A runtime **MUST NOT** place two items in one invocation
  unless they agree on every listed axis. Anything not listed is free to vary,
  and a free axis **MUST** be reconciled either by padding declared through
  `pad_mask` ([§3.2](#32-tensorcontractpad_mask)) or by packing declared through
  `token_packed` ([§4](#4-strict-token_packed-validation)).

Validation rules (all fail-closed, all naming the offending component, field,
axis, and the two facts that disagree, per Rule 1):

1. `max_rows` **MUST** be at least 2. A capacity of one is spelled by omitting
   the field; two spellings of one fact is duplicated state (Rule 10).
2. `axis` **MUST NOT** appear in `uniform_axes`, and `uniform_axes` entries
   **MUST** be distinct and within every participating port's rank.
3. If both `row_scope` and `batch_capacity` are declared, `row_scope.axis`
   **MUST** equal `batch_capacity.axis`. A component whose per-request state
   lives on a different axis from its batched items cannot be compacted
   consistently
   ([§8.5](INFERENCE_METADATA_DECISIONS.md#85-compaction-derivability)).
4. Every axis that is neither `axis` nor a member of `uniform_axes` **MUST** be
   symbolic in every participating port contract. A fixed literal extent on a
   free axis is a contradiction: the package has claimed items may differ there
   while pinning the shape that would have to change.
5. A component declaring `batch_capacity` **MUST** provide, for every free axis
   of every participating port, either a `pad_mask` on that axis or a packed
   layout on that axis. Declaring batchability without declaring how raggedness
   is expressed is the promise a runtime cannot honor, so it is rejected at load
   rather than discovered at run.

### 3.2 `TensorContract.pad_mask`

```yaml
pixel_values:
  dtype: float32
  rank: 4
  shape: [items, frames, patches, features]
  batch_layout: { kind: token_packed, offsets: item_offsets, owner: item_owner, axis: 0 }
  pad_mask:
    - { value: frame_mask, axes: [1] }          # temporal validity
    - { value: patch_mask, axes: [2] }          # spatial validity
```

`pad_mask` links a value that may be padded to the values that say which entries
are real. Absent, the value carries no padding and a runtime **MUST NOT**
introduce any.

- It is a **list of entries**, because a value can be padded on more than one
  axis at once. A video clip padded to the group's frame count *and* to the
  group's patch count needs a temporal truth and a spatial truth, and one mask
  cannot be both.
- **`value`** resolves in the namespace of the contract's owner: a sibling port
  name for a component port, an SSA value name for a workflow input or output.
  An unresolved name is a load error.
- **`axes`** are the padded axes *of the owning value* that this entry covers, in
  order. One entry usually covers one axis. An entry **MAY** cover several, which
  is how a single `[items, frames, patches]` bool tensor states joint
  spatiotemporal validity — the case where a clip's later frames are shorter than
  its earlier ones, which two independent 1-D masks cannot express.
- **Axes are covered exactly once.** Entries **MUST NOT** overlap, and every free
  axis of the value **MUST** be covered by exactly one entry or packed by its own
  layout. Two masks claiming one axis is two truths about one fact (Rule 10);
  zero masks on a padded axis is a fabricated value with no recorded truth.
- The referenced mask **MUST** be `bool`, **MUST** declare the same batch layout
  and item axis as the owning value, and its trailing axes **MUST** carry the
  same symbolic extents, in the same order, as the entry's `axes`. `true` means
  the entry is real. One dtype and one polarity, because a mask with two
  conventions is a silent-wrong-answer class.
- A value **MUST NOT** declare a `pad_mask` entry on an axis that its own layout
  packs. Padding and packing are two answers to the same question; a package
  picks one *per axis*. Padding a frame axis while packing the item axis is the
  normal case, not a conflict.
- Padding is a *runtime* act. The package declares that padding is expressible
  and where its truth is recorded; it never states a batch width, a padded
  extent, or a fill value schedule.

`pad_mask` is a structural fact and does **not** by itself make a component
padding-invariant. Whether a row's values change when it is padded to the group
width remains the profile-level declaration `batch_invariance`
(`crates/onnx-genai-metadata/src/schema/package.rs:158-174`,
`row_independent` / `padding_sensitive`). The two compose: a mask is what lets
an implementation *be* row-independent; `batch_invariance` is the package
asserting that it *is*. A `padding_sensitive` component **MUST NOT** be grouped
by padding, even where a mask exists, because the group would change the answer.
Temporal padding makes this concrete: a video encoder that pools over its frame
axis without consulting a temporal mask is `padding_sensitive`, and grouping
clips of different lengths would silently dilute every short clip's embedding.

### 3.3 Ownership values a preprocessing program must be able to produce

`BatchLayout::TokenPacked` names two values it does not have any way to obtain:
`offsets` and `owner`. No declared preprocessing program can produce them today
— the `ImageOutputContent` vocabulary is
`pixels`, `patch_coordinates`, `grid_dimensions`, `original_size`,
`transformed_size`, `validity_mask`
(`crates/onnx-genai-metadata/src/schema/mod.rs:293-303`) — so a package that
declares a packed encoder value must obtain its packing metadata out of band,
which is the model-family guessing this schema refuses everywhere else.

Four content roles close the loop, at two nesting levels:

| Content role | Contract | Meaning |
| --- | --- | --- |
| `item_offsets` | `int64`, rank 1, extent `rows + 1`, `batch_layout: { kind: shared }` | Exclusive prefix offsets of each row's items along the item axis. `offsets[0] == 0`, non-decreasing, `offsets[rows]` equals the packed extent. |
| `item_owner` | `int64`, rank 1, extent equal to the item-axis extent, `batch_layout: { kind: shared }` | For each packed item, the **position** of its owning row within the current invocation. |
| `subitem_offsets` | `int64`, rank 1, extent `items + 1`, `batch_layout: { kind: shared }` | The same fact one level down: exclusive prefix offsets of each item's parts along an inner packed axis. |
| `subitem_owner` | `int64`, rank 1, extent equal to the inner packed extent, `batch_layout: { kind: shared }` | For each packed part, the **position** of its owning item. |

The names say *level*, not modality. For a video encoder the two levels read as
clips-to-rows and frames-to-clips; for audio, windows-to-rows and
frames-to-windows; for a text reranker, segments-to-rows and tokens-to-segments.
The schema contains no `clip`, no `frame`, and no `window`, and a runtime
composing the two maps never learns which it is holding. That is the whole of the
"clip and frame offsets and owners" requirement: two ordinary packed levels, not
a video feature.

```yaml
preprocessing:
  video:                      # a video program; the image program is the same shape
    transforms: [...]
    outputs:
      - { source: patches, name: media.pixel_values,     content: pixels,           dtype: float32 }
      - { source: fmask,   name: media.frame_mask,       content: temporal_validity_mask, dtype: bool }
      - { source: pmask,   name: media.patch_mask,       content: validity_mask,    dtype: bool }
      - { source: coff,    name: media.item_offsets,     content: item_offsets,     dtype: int64 }
      - { source: cown,    name: media.item_owner,       content: item_owner,       dtype: int64 }
      - { source: foff,    name: media.subitem_offsets,  content: subitem_offsets,  dtype: int64 }
      - { source: fown,    name: media.subitem_owner,    content: subitem_owner,    dtype: int64 }
```

`item_owner` and `subitem_owner` carry a **position, never a request identity**
([§8.3](INFERENCE_METADATA_DECISIONS.md#83-no-row-identity)). They are exactly as
persistable as a `row_selection` value: not at all. A runtime that permutes rows
remaps owner values through the same permutation it applies to every
request-aligned value; it does not carry an owner across invocations.

**What batching needs from a modality vocabulary, and what it does not decide.**
The four ownership roles above are generic and shared. What each modality
contributes is only the *semantic* values a grouped encoder consumes: for video,
a temporal validity mask, per-item frame counts, and whatever frame-selection
facts the program applies (sampling stride, target frame count, per-frame
geometry). Those belong to a `preprocessing.video` program — which does not exist
today ([§2.7](#27-video-is-expressible-on-the-output-side-and-absent-on-the-encoder-side))
and whose full transform vocabulary is deliberately **out of scope here**. This
design states only the interface: a program that emits a temporal validity mask
and the two ownership levels is sufficient for grouping, whatever its transform
list turns out to be. The same holds for the audio vocabulary, which already has
`valid_frames` and `frame_lengths` (`mod.rs:349-364`) and needs only the
ownership roles.

### 3.4 Worked cases: images, video, audio, text

Every row below is the *same* component declaration with axes added or removed.
Nothing in the runtime path distinguishes them.

| Encoder | Item | Axes | Free axes and how they are reconciled | Typical `uniform_axes` |
| --- | --- | --- | --- | --- |
| Image | one image | `[items, patches, features]` | patches vary per image → spatial mask, or pack the patch axis | feature width |
| Video | one clip | `[items, frames, patches, features]` | frames vary per clip → temporal mask; patches vary per frame → spatial mask | feature width, and resolution when the encoder is resolution-pinned |
| Audio | one window | `[items, frames, mels]` | frames vary per window → temporal mask (`valid_frames` already exists) | mel-bin count |
| Text | one segment | `[items, tokens]` | tokens vary per segment → token mask, or pack the token axis | none |

**Variable clips, frames, and resolutions.** These are three separate questions
and the package answers them separately:

- *A request carries a different number of clips than another.* Item-count
  raggedness. `item_offsets`/`item_owner`. Grouping is unaffected: a group is
  formed from items, not from requests, and a request contributing zero items
  simply contributes an empty span.
- *Two clips have different frame counts.* Item-extent raggedness on the temporal
  axis. Either pad to the group's frame count and record the truth in a temporal
  mask, or pack the frame axis and carry `subitem_offsets`/`subitem_owner`. A
  runtime **MUST NOT** reconcile the difference by trimming or resampling frames
  — dropping a frame to fit a group changes what the caller asked for, and
  silently, which [§6](#6-fail-closed-behavior-and-backward-compatible-defaults)
  forbids.
- *Two clips have different resolutions.* This one is genuinely the package's
  call, and `uniform_axes` is where it is stated. A resolution-agnostic encoder
  (patchified input, position from coordinates) leaves the patch axis free and
  lets mixed-resolution items group under a spatial mask. A resolution-pinned
  encoder lists the geometry axis in `uniform_axes`, and the scheduler then forms
  groups only from items that agree — a compatibility test on declared extents,
  never on a model name or a resolution table.

**Batch compatibility is therefore a derived predicate, not a policy.** Two items
may share an invocation exactly when they belong to the same component, agree on
every `uniform_axes` extent, and every axis on which they disagree is either
masked or packed. The scheduler evaluates that predicate over declared contracts;
it contains no per-modality branch and no notion of what a frame is.

---

## 4. Strict `token_packed` validation

A packed layout is only usable if its two companion values are exactly what the
consumer assumes. Today none of that is checked
([§2.6](#26-validation-of-packed-values-is-close-to-absent)). The proposed rules,
all load-time and all fail-closed:

1. **Names resolve.** `offsets` and `owner` **MUST** name values declared in the
   same scope as the packed value. A dangling name is rejected, naming both the
   packed value and the missing name.
2. **`offsets` is `shared`, `int64`, rank 1, extent `rows + 1`.** This resolves
   the contradiction in [§2.6](#26-validation-of-packed-values-is-close-to-absent)
   in favor of the fixture and of
   [§10.1](INFERENCE_METADATA_DECISIONS.md#101-independent-encoder-batching), and
   against the `ir.rs` doc comment, for a structural reason: an exclusive prefix
   sum is **not permutation-followable**. Permuting rows does not permute a
   prefix-offset vector, it invalidates it. Labeling it `request_aligned` would
   invite a runtime to gather it during compaction and silently produce
   nonsense. `offsets` describes the *whole* current grouping; a runtime that
   changes the grouping **MUST** recompute it. The doc comment is corrected when
   the rule lands.
3. **`owner` is `shared`, `int64`, rank 1**, with extent equal to the packed
   value's extent along `axis`.
4. **`axis` is within the packed value's rank.**
5. **Shared companions agree.** Two packed values naming the same `offsets`
   **MUST** declare the same packed-extent symbol; otherwise one of them is
   packed against a grouping that does not describe it.
6. **No double spelling on one axis.** A packed value **MUST NOT** declare a
   `pad_mask` entry on an axis its layout packs
   ([§3.2](#32-tensorcontractpad_mask)). A `pad_mask` entry on a *different* axis
   is the normal composition of the three kinds of raggedness
   ([§3](#3-the-proposed-metadata-surface)).
7. **A packed emit publishes its companions.** If a workflow emits a
   `token_packed` output, `offsets` and `owner` **MUST** also be declared outputs.
   The existing serving-emit rule (`validation.rs:3306-3320`) accepts a packed
   emit today; without this rule the consumer receives a ragged buffer it cannot
   split.
8. **Nested levels compose, and their bounds agree.** A value packed on two axes
   declares one `offsets`/`owner` pair per axis. The inner pair's `offsets`
   extent **MUST** be the outer packed extent plus one, and its `owner` values
   **MUST** address positions within the outer extent. A package that packs an
   inner axis without declaring an inner pair is rejected: the group would be
   unsplittable at that level, which is exactly the frames-inside-clips case
   ([§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce)).
9. **Execution-time invariants are checked, not assumed.** Before a packed value
   is consumed the runtime verifies `offsets[0] == 0`, monotonic
   non-decreasing offsets, `offsets[rows] == packed_extent`, `owner[i]` in
   `[0, rows)`, and `owner` consistent with `offsets` (items of a row are
   contiguous) — **at every level**, so a frame owner that points outside its
   clip fails the same way a clip owner that points outside its row does. A
   violation is an error that names the value, the level, the index, and the two
   facts that disagree — never a truncation, a clamp, or a best-effort split.

Rules 1–8 are validator rules with negative fixtures; rule 9 is a runtime
precondition with a test that corrupts each field at each level in turn and
asserts the exact message.

---

## 5. Ownership split

One fact, one owner. Nothing below is a new layer; each row is the existing
owner of that layer doing its existing job over a new declared fact.

| Layer | Owns | Explicitly does not own |
| --- | --- | --- |
| **Metadata** | That a component may be grouped, on which axis, up to what bound, which axes must agree, how raggedness is expressed at every level. | Whether to group, how many to group, when, or how fast it will be. |
| **Modality vocabulary** | The *semantic* values a program produces — pixels, temporal and spatial validity masks, frame counts, grid geometry — and the transforms that produce them. | Anything about grouping. A content role never says an axis is batchable; it says what the numbers mean. |
| **Preprocessor** | Producing per-item tensors and, when a program declares them, the ownership values (`item_*`, `subitem_*`) for the items it was handed. | Deciding which items are handed to it; any request identity; any cross-invocation state. |
| **Scheduler** | Deciding which pending work items co-batch, by evaluating the declared compatibility predicate ([§3.4](#34-worked-cases-images-video-audio-text)) under `max_rows`; admission, fairness, latency-versus-throughput, deadlines, backpressure. | Model identity; modality; tensor construction; padding; any knowledge of what the component computes. |
| **Interpreter** | Building the grouped invocation: broadcasting `shared` inputs, padding only where a `pad_mask` entry exists, concatenating only where a packed layout exists, invoking once, and splitting outputs back to rows through every declared ownership level. | Group composition policy; backend selection; storage layout. |
| **Backends (ORT and native)** | Executing the grouped invocation through the one component-execution seam, with identical results. | Any batching decision of their own. A backend never has a private grouping path. |

The seam that makes the last row true is `WorkflowComponentBackend`
([`../architecture/NATIVE_WORKFLOW_BACKEND.md` §3](../architecture/NATIVE_WORKFLOW_BACKEND.md)):
grouping is built above it in the interpreter, so ORT and native receive the
same named tensors and there is exactly one grouping implementation. If a
backend cannot execute a grouped invocation it fails with a diagnostic naming
the component and the bound — it does not quietly run the items one at a time,
because a silent fallback makes a performance regression indistinguishable from
correct behavior (Rule 4).

The scheduler's new work-item queue is generic over components. It reads
`batch_capacity` and the participating port contracts; it never reads a
component name, a modality, or an artifact filename
([`SCHEDULING.md`](SCHEDULING.md) owns the admission and preemption policy that
this queue plugs into).

---

## 6. Fail-closed behavior and backward-compatible defaults

**Defaults.** `batch_capacity` absent, `pad_mask` absent, and no packed layout is
the whole of today's behavior: one item per invocation, no padding, no packing.
Every package in `tests/fixtures/onnx_genai_workflows/` keeps byte-identical
execution, which is the first row of the acceptance matrix
([§9](#9-e2e-acceptance-matrix)) rather than an assertion.

**Silence never grants behavior.** A runtime that has not implemented grouping
reads `batch_capacity` and ignores it, and it is still correct, because capacity
is an upper bound on a permitted optimization and not a semantic requirement.
That is precisely why grouping does **not** introduce a new
`required_capabilities` identifier
([§4.3a](INFERENCE_METADATA_DECISIONS.md#43a-capability-admission-and-complete-built-in-catalogue)):
a capability identifier is a load-time promise that *correct execution requires*
a behavior, and no package's correctness requires that its encoder be batched.
Adding one would make an old runtime refuse a package it can execute perfectly.

**Fail-closed is about the padding and packing, not the grouping.** The
asymmetry matters:

- Declining to group is always safe and is never an error.
- Grouping *without honoring* a declared `pad_mask` or packed layout is a
  wrong-answer bug, so it is forbidden: if the runtime cannot construct the mask
  or the offsets for some participating port, it **MUST NOT** group that
  invocation.
- A runtime **MUST NOT** fabricate padding for a value that has no `pad_mask`
  entry on that axis, **MUST NOT** invent an `offsets`/`owner` pair a package did
  not declare at any level, and **MUST NOT** group items that disagree on a
  `uniform_axes` extent.
- A runtime **MUST NOT** make two items compatible by changing them. Trimming
  frames, resampling a clip to a common frame count, downscaling to a common
  resolution, or truncating a token segment are all semantic changes to what the
  caller asked for. Padding with a recorded mask is the only sanctioned way to
  reconcile a free axis, and packing is the only sanctioned way to avoid padding.
- A package that *declares* batchability it has not made expressible is rejected
  at load ([§3.1](#31-workflowcomponentbatch_capacity) rule 5), not at the first
  unlucky group.

**Row semantics survive.** Grouping introduces no new identity. Items are
positional inside an invocation, exactly as rows are positional inside a batch
([§8.3](INFERENCE_METADATA_DECISIONS.md#83-no-row-identity)). A grouped
component that holds per-request state is row-scoped and therefore already
implements `compact`/`release`
([§8.6](INFERENCE_METADATA_DECISIONS.md#86-mandatory-row-abi)); rule 3 of
[§3.1](#31-workflowcomponentbatch_capacity) is what guarantees the axes agree.

---

## 7. What this deliberately does not add

**No model-specific booleans.** Not `is_vision`, not `is_video`, not
`qwen_vl_packing`, not `clip_style_encoder`, not `supports_image_batching`, not
`supports_video_batching`. A batching decision must be derivable from geometry
and declared bounds alone. If a new architecture needs a runtime branch to be
batched, the correct outcome is a new *generic* structural fact — or that the
architecture is simply not batchable here — never a name test (Rule 2).

**No modality in the batching path.** The contracts carry axes, bounds, masks,
and ownership levels; only a preprocessing program's content roles say that an
axis counts frames rather than tokens. The interpreter and the scheduler
therefore have exactly one implementation of grouping, and adding a modality adds
a vocabulary and a fixture, not a branch. A "video batching" code path would be
the same defect as a "video attention" kernel.

**No `supports_batching` flag.** A boolean answers "may I?" but not "how many",
"along which axis", or "which axes must agree", so a runtime holding a `true`
still cannot build a group. `batch_capacity` carries the answer and its absence
carries the negative, which is why there is no separate boolean to keep in sync.

**No duplicated derived capability.** This is the general direction, stated once
and applied narrowly here: **a fact the workflow structure already determines
should not also be serialized as a flag.** A grouped invocation is fully
described by `batch_capacity` plus the port contracts, so nothing about it is
restated in `required_capabilities` or in
`pipeline.workflow.manifest.capabilities`. Serialized capability strings earn
their place only for facts the structure cannot recover — an implementation
requirement, a legacy admission, an external ABI.

The existing catalogue contains at least one entry that appears to restate
structure rather than add to it: `packed_image_outputs`
([§4.3a](INFERENCE_METADATA_DECISIONS.md#43a-capability-admission-and-complete-built-in-catalogue))
describes packing metadata that a declared image program's outputs plus a
`token_packed` layout already state. **Re-litigating that entry is out of scope
for this work.** Removing a serialized capability is a compatibility decision
for readers that already require it, it has nothing to do with encoder batching,
and bundling the two would mean this design could not land without also
relitigating capability admission. The direction is recorded here so the new
surface does not add to the pile; the audit is separate.

**No cost model.** `max_rows` is a correctness bound. Metadata never carries
measured throughput, an admission prediction, or a tuned heuristic
([§1.2](INFERENCE_METADATA_DECISIONS.md#12-non-goals)). Choosing a group size at
or below the bound is a scheduler decision informed by runtime measurement.

---

## 8. Execution phases and PR DAG

Each phase lands green on its own, with tests, and no phase requires the next
one to be useful.

```
P0 docs (this change)
     │
     ▼
P1 schema surface ──────────────┬──────────────► P3 preprocessor produces
  batch_capacity, pad_mask,     │                   ownership + validity values
  ownership content roles       │                          │
     │                          │                          │
     ▼                          │                          │
P2 validation                   │                          │
  §3.1 rules 1-5, §4 rules 1-8  │                          │
     └──────────────┬───────────┘                          │
                    └──────────────────┬───────────────────┘
                                       ▼
                          P4 interpreter grouped invocation
                            group build · broadcast · pad ·
                            pack · invoke once · split
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
              P5 backend parity            P6 scheduler grouping
                ORT ≡ native                 work-item queue,
                                             max_rows / uniform_axes
                          └────────────┬────────────┘
                                       ▼
                             P7 E2E + performance
                               acceptance matrix §9
```

- **P0 — design of record (this change).** Docs only. No code, no schema.
- **P1 — schema surface.** Add `WorkflowComponent.batch_capacity`,
  `TensorContract.pad_mask`, and the ownership content roles
  ([§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce)).
  Regenerate `schema/inference_metadata.schema.json` with `gen_schema`. Positive
  fixtures only; nothing reads the fields yet. Guard:
  `cargo test -p onnx-genai-metadata` including the committed-schema comparison.
- **P2 — validation.** Implement [§3.1](#31-workflowcomponentbatch_capacity)
  rules 1–5 and [§4](#4-strict-token_packed-validation) rules 1–8, each with a
  negative fixture asserting the exact message. Depends on P1.
- **P3 — preprocessor.** Two independent pieces, in this order.
  **P3a (items):** let the image adapter accept N encoded items and emit
  `item_offsets` / `item_owner`; unit tests over the offset and owner arithmetic
  on the CPU, with zero-item and single-item rows covered.
  **P3b (nesting):** a frame-sequence producer — the input side that
  [§2.7](#27-video-is-expressible-on-the-output-side-and-absent-on-the-encoder-side)
  shows is missing — whose interface takes an ordered frame sequence per clip and
  emits `subitem_offsets` / `subitem_owner` plus a temporal validity mask. P3b
  needs no new *batching* concept; it is the modality vocabulary that makes the
  nested level producible. Depends on P1, not on P2.
- **P4 — interpreter.** Grouped invocation in the one interpreter: broadcast
  `shared`, pad where a `pad_mask` entry says to, pack where `token_packed` says
  to, invoke once, split results back through every declared ownership level,
  plus the [§4 rule 9](#4-strict-token_packed-validation) runtime precondition
  checks. The default path stays one item per invocation. Key test: grouped
  output equals sequential output row by row. Depends on P2 and P3a; the nested
  split is exercised by P3b.
- **P5 — backend parity.** Run P4's grouped invocation under ORT and under the
  native backend through `WorkflowComponentBackend`; assert identical results and
  identical rejection messages. Depends on P4.
- **P6 — scheduler grouping.** A generic pending-work-item queue that forms
  groups under `max_rows` and the `uniform_axes` compatibility predicate
  ([§3.4](#34-worked-cases-images-video-audio-text)), with cancellation and
  compaction while a group is in flight. Depends on P4; independent of P5.
- **P7 — E2E and performance.** The full matrix in [§9](#9-e2e-acceptance-matrix)
  for both an image encoder and a video encoder, including grouped-versus-
  sequential throughput on fixed hardware. Depends on P5 and P6.

If P4 proves large — the split-by-owner path is the likeliest — it splits into
"pad and broadcast" and "pack and split", in that order, since padding needs no
companion values.

---

## 9. E2E acceptance matrix

Every row is an end-to-end test, not a unit assertion. "Solo" means the same
item executed alone through the same package; per-row equality against solo is
the correctness definition for every batching row. Rows 1–9 are modality-neutral
and are run against both an image encoder and a video encoder; rows 10–15 pin the
modality-specific geometry that motivated the design.

| # | Scenario | What it must prove | Gate |
| --- | --- | --- | --- |
| 1 | **Non-batchable component.** Package declares no `batch_capacity`. | Execution is byte-identical to the pre-change baseline; the runtime never groups. | P4 |
| 2 | **Padded and masked group.** Items differ on a free axis with a declared `pad_mask` entry. | Each row's output equals its solo output; padded positions never influence a real row. | P4 |
| 3 | **Variable item size.** Items with genuinely different geometry grouped via `token_packed`. | Packed result splits back to per-row results equal to solo; no reliance on a common width. | P4 |
| 4 | **Variable item count per request.** Requests carrying 0, 1, and many items in one group. | A zero-item request never gets a fabricated placeholder item; offsets stay consistent with an empty span. | P4 |
| 5 | **Packed ownership.** `offsets`/`owner` round-trip, plus a deliberately corrupted `offsets` and a corrupted `owner`. | Correct split on the good case; on each corrupted case a loud error naming value, index, and the two disagreeing facts. | P4 |
| 6 | **Multi-request grouping.** Items from N concurrent requests in one invocation. | Per-row equality against solo, and no value from one request observable in another's row. | P6 |
| 7 | **ORT and native parity.** Rows 2, 3, 6, 11, and 12 under both backends. | Identical outputs and identical rejection messages; neither backend has a private grouping path. | P5 |
| 8 | **Concurrency.** Arrival, cancellation, and compaction while a group is in flight. | Row positions and owner mappings stay consistent through one permutation; a cancelled request's items never reach another row. | P6 |
| 9 | **Incompatible group refused.** Items disagreeing on a `uniform_axes` extent — for a resolution-pinned encoder, two different resolutions — offered to the same group. | The scheduler forms two groups (or executes solo) and never reconciles by resizing; results equal solo. | P6 |
| 10 | **Variable frames per clip.** Clips of differing frame counts padded on the temporal axis with a temporal `pad_mask` entry. | Each clip's output equals its solo output; no frame is trimmed or resampled to fit; a temporally pooled output is unaffected by pad frames. | P4 |
| 11 | **Variable clips per request.** Requests carrying 0, 1, and several clips, clips themselves carrying differing frame counts. | Both ownership levels round-trip: clips split back to requests and frames split back to clips; per-row equality against solo. | P4 |
| 12 | **Mixed spatial and temporal raggedness.** One group where items differ in frame count *and* in per-frame token count. | Spatial and temporal masks compose; the disjointness rule holds; each row equals solo. | P4 |
| 13 | **Nested ownership corruption.** `subitem_offsets` and `subitem_owner` corrupted in turn: an owner outside its clip, a non-monotonic inner offset, an inner total disagreeing with the outer extent. | Each case is a loud error naming value, level, index, and the two disagreeing facts — never a clamp or a partial split. | P4 |
| 14 | **Mixed-modality serving.** Image items and video clips in flight for the same engine. | Grouping is per component, not per request; an image group and a video group are formed by the same code with no modality branch; per-row equality against solo. | P6 |
| 15 | **A third modality reuses the path.** An audio (or text-segment) encoder declaring `batch_capacity` plus windows-in-rows ownership. | It batches with no new interpreter or scheduler code — the acceptance is that the diff is a fixture and a vocabulary, not a branch. | P6 |
| 16 | **Performance versus sequential direct execution.** Same hardware, same items, grouped versus one-at-a-time, reported separately for image and for video. | Images/s, frames/s, clips/s, and per-request latency for both modes, plus the group sizes actually formed. Per-row outputs identical. A regression at any reachable group size is reported, not hidden behind an average. | P7 |

Row 16 follows the measurement protocol already used for batched decode
([`NATIVE_BATCH_DECODE_2B_IMPL_SCOPING.md` §6](NATIVE_BATCH_DECODE_2B_IMPL_SCOPING.md)):
report the mechanism-level counters and the achieved group sizes, never a bare
wall-clock headline, and verify the device is idle before each run. Video is
reported in frames/s *and* clips/s because a group of few long clips and a group
of many short clips are different operating points that a single items/s number
would blur.

---

## 10. Open questions

1. **Group-size choice.** The scheduler needs a policy for trading first-item
   latency against group occupancy. That is runtime policy and stays out of
   metadata, but P6 should expose the chosen size so row 16 can attribute a
   result. Video sharpens it: one long clip can exceed the useful work of a whole
   image group, so the bound that matters may be total rows *and* total frames.
2. **Cross-invocation reuse.** An encoder result declared
   `externally_suppliable`
   ([§10.2](INFERENCE_METADATA_DECISIONS.md#102-externally-suppliable-results))
   may already be cached. Grouping and reuse interact — a cached item should not
   occupy a group slot — but the cache key derivation is unchanged
   ([§11](INFERENCE_METADATA_DECISIONS.md#11-cache-correctness-dependencies)).
3. **Which modalities need packing versus padding.**
   [§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce)
   makes the ownership roles available to any vocabulary. Whether a given
   modality's first package needs packing, or whether padding plus a `pad_mask`
   entry suffices — plausible for fixed-window speech encoders and fixed-frame
   video encoders, implausible for native-resolution image encoders — is decided
   per package, from its geometry, not decided here.
4. **Naming of the nested level.** `subitem_offsets` / `subitem_owner` are
   level-named on purpose so that clips-in-requests and frames-in-clips share one
   rule. If a third nesting level ever appears, the naming should become
   indexed rather than gaining a third bespoke prefix.
5. **Where the frame-sequence producer lands.** P3b needs an input side that
   accepts an ordered frame sequence per clip; today the image path takes
   independent images and `temporal_patch_size` only replicates a frame
   ([§2.7](#27-video-is-expressible-on-the-output-side-and-absent-on-the-encoder-side)).
   Whether that is a new `preprocessing.video` program or a sequence-aware mode
   of the existing image program is a preprocessing decision, and it does not
   change any contract in this document.
