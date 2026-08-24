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
2. **How much work fits in one invocation?** The bound is a *correctness*
   property of the artifact, not a tuning knob, and it is not one number: an item
   count alone cannot bound a video group, and a frame count alone cannot either.
3. **Which dimensions must agree before two items may share an invocation?** A
   feature width must match; a patch count usually must not have to; a frame rate
   or a resolution may or may not, and only the package knows which.
4. **When extents differ on a free dimension, how is the difference expressed** —
   by padding to a rectangle and recording how much of each item is real, or by
   packing the items end to end with offsets and an owner map? A video item needs
   this answered on more than one dimension at once: frames per clip *and*
   patches per frame.
5. **When an item is itself a group, who owns what?** A clip owns frames, an
   audio window owns spectrogram frames, a text segment owns tokens. Splitting a
   grouped result back to requests needs the frames-to-clip map as well as the
   clips-to-row map — over one flattened buffer, not two.

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
- **No nesting.** `BatchLayout::TokenPacked` gives exactly one `offsets`/`owner`
  pair for the whole value (`crates/onnx-genai-metadata/src/schema/ir.rs:51-58`),
  and nothing composes two levels, so "frames belong to clips belong to rows" has
  no declared spelling even though both halves of it are ordinary packing.

So the video gap is not a missing modality feature. It is the *same* three
missing generic facts — a batching capacity, a per-axis padding truth, and a
per-level ownership map — observed on a workload that cannot avoid needing all
three at once.

### 2.8 The metadata structs are closed, and the version field is not a gate

Adding a field is not a free, forward-compatible act here. `InferenceMetadata` is
`#[serde(deny_unknown_fields)]` (`crates/onnx-genai-metadata/src/schema/mod.rs:36-38`)
and `schema/ir.rs` carries 45 more, including every struct this design would
extend. An older runtime therefore **rejects the whole document** — with a serde
error naming an unknown field — rather than ignoring a field it does not know.
The `schema_version` doc comment claims the opposite ("rely on the
forward-compatible 'ignore unknown fields' rule", `schema/mod.rs:47-53`), and
nothing validates `schema_version` at all: `crates/onnx-genai-metadata/src/validation.rs`
never mentions it. So the *mechanism* for shipping a new metadata surface is
itself missing, and [§6.1](#61-schema-evolution-what-actually-happens-to-an-old-runtime)
specifies it before any field is proposed for emission.

**Summary of the gap:** the tensor vocabulary is there, the packed *shape* is
there, and neither a component-level batching capability nor a `token_packed`
runtime exists. Video additionally has no encoder-side producer for the values
either would consume, and the schema has no version gate through which any of it
can ship. This design adds the first, specifies the second, states the minimum a
modality vocabulary must produce for the third, and defines the fourth.

---

## 3. The proposed metadata surface

Three additions. Each is a structural fact about an artifact, each is absent by
default, and each is meaningless to a runtime that chooses not to group.

They exist because grouping faces **three independent kinds of raggedness**, and
conflating them is how a batching layer gets subtly wrong answers:

- **Item count** — how many items each request owns (zero, one, or many images;
  zero, one, or many clips). Answered by an **ownership level** whose `offsets`
  and `owner` map items to request rows.
- **Item extent** — how far two items differ along a shape dimension: patches per
  image, spectrogram bins per window, tokens per segment. Answered by padding
  that dimension and declaring where its validity truth is recorded
  ([§3.2](#32-tensorcontractpadding)), or by folding that dimension into the
  packed axis.
- **Item nesting** — an item that is itself a group, whose parts must be
  attributable after the group is split: frames within a clip, windows within an
  utterance. Answered by an **additional ownership level over the same physical
  packed axis**.

**A value has exactly one physically packed axis.** Nesting adds levels of
ownership over that one axis; it never adds a second packed axis. Frames are
flattened across every clip of every request into one leading axis, and the
frame→clip and clip→row maps are two levels of bookkeeping over that single
flattened axis. An earlier revision of this document described nested video
packing as two packed axes and showed a `[items, frames, ...]` contract; that
spelling is **withdrawn** — it made the same geometry expressible two ways, and
only one of the two can be split without a strided gather
([§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce),
[§4](#4-strict-token_packed-validation) rule 2).

A single video encoder input needs all three kinds at once — clips from three
requests, each clip with a different frame count, each frame with a different
patch count — and each is declared on its own dimension of the same contract, so
no two ever compete for one. An image encoder is the same declaration with the
frame level absent; an audio or text encoder is the same declaration with the
dimensions meaning spectrogram frames or tokens. The contract never says which.

### 3.1 `WorkflowComponent.batch_capacity`

`batch_capacity` is declared **by shape symbol**, never by axis index. Ports of
one component routinely differ in rank — a rank-3 payload, a rank-1 companion, a
rank-2 pooled output — so an axis index is only meaningful relative to one port
and a component-global integer axis cannot be interpreted coherently across them.
A `TensorDimension` is already either a fixed extent or a **runtime shape
symbol** (`crates/onnx-genai-metadata/src/schema/decoder_abi.rs:266-273`), and the
interpreter already binds those symbols to concrete extents per invocation
(`crates/onnx-genai-engine/src/pipeline/workflow.rs:1115-1145`), so a symbol names
the same quantity on every port that mentions it, whatever its rank.

The example below is a **video** encoder, because it is the case that exercises
every field; [§3.4](#34-worked-cases-images-video-audio-text) reduces it to the
image, audio, and text shapes by deleting levels.

```yaml
components:
  clip_encoder:
    implementation: { kind: onnx, artifact: encoder.onnx }
    batch_capacity:
      uniform_dimensions: [features]          # must agree across co-batched items
      budgets:                                 # materialized footprint bounds
        - { dimensions: [clips],  max_total: 4 }
        - { dimensions: [frames], max_total: 64 }
        - { dimensions: [frames, patches], max_total: 65536 }
```

Semantics:

- **Absent means one request row per invocation.** That is today's behavior
  exactly, so every existing package is unaffected. Absence is a statement, not a
  silence: the runtime **MUST NOT** group a component that has not declared a
  capacity.
- **`uniform_dimensions`** lists the shape symbols whose extent **MUST** be equal
  across every co-batched item — **temporal and spatial alike**. A
  resolution-pinned encoder lists its geometry symbols, a frame-count-pinned
  encoder lists its temporal symbol, and an encoder pinned both ways lists both.
  A runtime **MUST NOT** place two items in one invocation unless they agree on
  every listed symbol. A symbol not listed is free to vary, and a free symbol
  **MUST** be reconciled either by padding
  ([§3.2](#32-tensorcontractpadding)) or by packing
  ([§4](#4-strict-token_packed-validation)).
- **`budgets`** bound the **materialized footprint** of the assembled group,
  keyed by shape symbol. Each entry names one symbol, or an ordered list of
  symbols whose materialized extents multiply, and a `max_total` the group
  **MUST NOT** exceed. They are *static geometry* — a fixed position table, an
  exported constant, a kernel bound — never a measured throughput sweet spot.
  Metadata never carries benchmark-derived cost models
  ([§1.2 non-goal 4](INFERENCE_METADATA_DECISIONS.md#12-non-goals)). Every budget
  is an upper bound and never an obligation: a runtime **MAY** group fewer,
  including one.
- **There is no `max_items` integer.** The item bound is the budget on the item
  level's own symbol — `clips` above, `images` for a still-image encoder. One
  spelling, not two (Rule 10), and it composes with the other budgets instead of
  standing outside them.
- **`request_expanded` participates without companions.** A port declared
  `BatchLayout::RequestExpanded { axis, factor }` gives every request a
  fixed-size contiguous group of `factor` entries on `axis`
  (`crates/onnx-genai-metadata/src/schema/ir.rs:43-50`), so ownership there is
  arithmetic — entry `i` belongs to row `i / factor` — and no `offsets`/`owner`
  pair is declared or permitted for it. Its materialized footprint on that
  dimension is `rows × factor`. A port **MUST NOT** be both request-expanded and
  packed on the same axis: the two state the same ownership with different
  degrees of freedom, and a fixed factor is the stronger claim.

Validation rules (all fail-closed, all naming the offending component, field,
symbol, and the two facts that disagree, per Rule 1):

1. **Symbols resolve.** Every symbol in `uniform_dimensions` and in every
   `budgets` entry **MUST** appear in the declared `shape` of at least one port
   of the component — payload or companion, since a level's unit count is often
   named only by its `owner` companion's extent. An unknown symbol is a load
   error naming the symbol and the component; it is otherwise a typo that
   silently binds nothing.
2. **A symbol is pinned or budgeted, never both.** `uniform_dimensions` and
   `budgets` **MUST NOT** name the same symbol. A pinned dimension has one extent
   across the group, so its footprint is already bounded by the item budget times
   that extent; budgeting it too is duplicated state (Rule 10). Entries within
   each list **MUST** be distinct.
3. **Every input-side ownership level is budgeted.** For each level declared by
   a participating packed **input** port, the symbol that counts that level's
   units — the extent of its `owner` companion — **MUST** carry a budget.
   Otherwise the group is unbounded in exactly the quantity the scheduler is
   choosing. Output levels are not budgeted: an extent the graph decides
   ([§4](#4-strict-token_packed-validation) rule 5) cannot be a precondition on
   forming the group, and bounding it would be a claim about the artifact's
   output the package cannot check either.
4. **Every free symbol is symbolic everywhere.** A symbol that is neither pinned
   by `uniform_dimensions` nor fixed by the artifact **MUST** be declared as a
   `TensorDimension::Symbol`, not a literal, in every participating port that has
   that dimension. A fixed literal on a free dimension is a contradiction: the
   package has claimed items may differ there while pinning the shape that would
   have to change.
5. **Raggedness is expressible.** For every free dimension of every participating
   port, the component **MUST** declare either a `padding` entry on that
   dimension or a packed layout that consumes it. Declaring batchability without
   declaring how raggedness is expressed is a promise a runtime cannot honor, so
   it is rejected at load rather than discovered at run.
6. **Budgets are large enough to serve one item.** Each `max_total` **MUST** be
   at least the largest single item's footprint on that dimension, or the budget
   forbids an invocation the component is otherwise required to serve.
7. **Row scope is about rows.** If both `row_scope` and `batch_capacity` are
   declared, `row_scope.axis` (`ir.rs:729-737`) **MUST** be the axis of the
   component's **request rows** in its row-scoped ports, and it **MUST NOT** be
   the packed item axis of any port. A packed axis counts items, and items are
   not rows ([§5.2](#52-compaction-lifts-a-row-selection-through-the-ownership-chain)).
   An earlier revision required `row_scope.axis` to equal the item axis; that
   rule is **withdrawn** — it conflated the two counts and would have compacted
   per-request state with an item-indexed selection.

### 3.2 `TensorContract.padding`

```yaml
pixel_values:
  dtype: float32
  rank: 3
  shape: [frames, patches, features]
  batch_layout: { kind: token_packed, axis: 0, levels: [...] }
  padding:
    - { dimension: patches, valid_lengths: patch_lengths }
```

`padding` links a value that may be padded to the companion that says how much of
each entry is real. Absent, the value carries no padding and a runtime **MUST
NOT** introduce any.

**Lengths, not a mask.** An earlier revision spelled this `pad_mask` with a
`bool` tensor per padded axis. That spelling is **withdrawn** in favor of a
concise per-item length, for three reasons that are properties of the design
rather than preferences. Padding is right-padding — real entries **MUST** form a
prefix — so a boolean mask is a run of `true` followed by a run of `false` and
carries exactly one number of information per item; storing it as a mask is
`O(items × extent)` state for an `O(items)` fact (Rule 10). The runtime must read
those numbers to build and to split a group, and a length vector is a few
kilobytes on the host while a mask is a payload-sized tensor that may be
device-resident — reading it back would be precisely the hidden host round-trip
[§5.1](#51-grouped-buffers-aliasing-residency-and-what-padding-costs) forbids.
And a length cannot express an interior hole, so the invariant is enforced by the
representation instead of by a check that must be written, tested, and kept
correct. A component whose *graph* consumes a materialized boolean mask declares
that mask as an ordinary input port and its preprocessing program produces it;
the runtime neither derives nor interprets it, and the lengths stay the single
truth.

- It is a **list of entries**, because a value can be padded on more than one
  dimension at once.
- **`dimension`** is the shape symbol of the padded dimension of the owning
  value. It **MUST** appear in the owning value's declared `shape`, and it
  **MUST NOT** be the packed axis's own symbol ([§4](#4-strict-token_packed-validation)
  rule 7) or a member of `uniform_dimensions`.
- **`valid_lengths`** resolves in the namespace of the contract's owner: a
  sibling port name for a component port, an SSA value name for a workflow input
  or output. An unresolved name is a load error.
- **Exact companion shape.** The `valid_lengths` value **MUST** be `int64`,
  **MUST** declare `batch_layout: { kind: shared }`, and **MUST** have rank equal
  to the number of axes **outer** to the padded axis in the owning value, with
  those axes' symbols as its shape, in order. For `pixel_values` of shape
  `[frames, patches, features]` padded on `patches` (axis 1), the companion is
  `[frames]`: one length per packed frame. For a hypothetical
  `[windows, bins, features]` padded on `bins`, it is `[windows]`. Axes *inner*
  to the padded one are not indexed — a length applies to the whole slice.
  Entry `k` gives the count of real positions along the padded dimension for
  outer index `k`; positions `[0, valid)` are real and
  `[valid, padded_extent)` are padding.
- **One truth per dimension.** Entries **MUST NOT** name the same dimension
  twice, and every free dimension of the value **MUST** be either covered by
  exactly one entry or consumed by the packed axis. Two entries on one dimension
  is two truths about one fact (Rule 10); zero on a padded dimension is a
  fabricated value with no recorded truth.
- **Padding is appended, never prepended and never interleaved.** This is the
  right-padding rule, and it is a rule rather than a convention for three
  reasons. Interior holes would force every consumer that slices, reduces, or
  pools to handle gaps rather than a length. Left padding would make position
  indices — temporal position embeddings above all — disagree between a solo item
  and the same item in a group, which is a silent-wrong-answer class and not a
  performance question. And prefix validity is what makes a packed level and a
  padded dimension describe the same geometry: an `offsets` vector is a prefix
  sum, so contiguity at one level and prefix validity at the next are the same
  statement made twice ([§4](#4-strict-token_packed-validation)).
- **Multi-dimension padding is per dimension.** A value padded on two dimensions
  declares two entries, each with its own companion. A clip whose later frames
  carry fewer patches than its earlier ones is expressed by a *per-frame* patch
  length — which the `[frames]`-shaped companion above states exactly — not by a
  two-dimensional mask.
- Padding is a *runtime* act. The package declares that padding is expressible
  and where its truth is recorded; it never states a batch width, a padded
  extent, or a fill value schedule.

`padding` is a structural fact and does **not** by itself make a component
padding-invariant. Whether a row's values change when it is padded to the group
width remains the profile-level declaration `batch_invariance`
(`crates/onnx-genai-metadata/src/schema/package.rs:158-174`,
`row_independent` / `padding_sensitive`). The two compose: a validity truth is
what lets an implementation *be* row-independent; `batch_invariance` is the
package asserting that it *is*. A `padding_sensitive` component **MUST NOT** be
grouped by padding, even where lengths exist, because the group would change the
answer. A video encoder that pools over a padded dimension without consulting the
lengths is `padding_sensitive`, and grouping would silently dilute every short
item's embedding.

### 3.3 Ownership values a preprocessing program must be able to produce

`BatchLayout::TokenPacked` names values it has no way to obtain: no declared
preprocessing program can produce offsets or owners today — the
`ImageOutputContent` vocabulary is `pixels`, `patch_coordinates`,
`grid_dimensions`, `original_size`, `transformed_size`, `validity_mask`
(`crates/onnx-genai-metadata/src/schema/mod.rs:293-303`) — so a package that
declares a packed encoder value must obtain its packing metadata out of band,
which is the model-family guessing this schema refuses everywhere else.

**One packed axis, an ordered chain of ownership levels.** The proposed shape of
`TokenPacked` is one `axis`, which **MUST** be `0`, plus `levels`: an ordered
list of one or two `{ offsets, owner }` pairs, **innermost first**. Level 0 maps
physically packed positions to their immediate parent unit; the last level maps
the outermost unit to request rows. There is exactly one packed axis no matter
how many levels there are. A packed **output** additionally declares
`packed_extent` — `preserved` when its levels correspond one-to-one with an
input's, `produced` when the graph decides the extent and the companions are the
component's own outputs ([§4](#4-strict-token_packed-validation) rule 5).

```yaml
batch_layout:
  kind: token_packed
  axis: 0
  levels:
    - { offsets: frame_offsets, owner: frame_owner }   # frames -> clips
    - { offsets: clip_offsets,  owner: clip_owner }    # clips  -> rows
```

Let `P` be the packed extent (the axis-0 extent), `U` the number of units at the
outer level (clips), and `R` the number of request rows in the invocation. Then:

| Value | Contract | Extent | Meaning |
| --- | --- | --- | --- |
| `levels[0].offsets` | `int64`, rank 1, `shared` | `U + 1` | Exclusive prefix offsets of each unit's packed positions. `[0] == 0`, non-decreasing, `[U] == P`. |
| `levels[0].owner` | `int64`, rank 1, `shared` | `P` | For each packed position, the **position** of its owning unit in `[0, U)`. |
| `levels[1].offsets` | `int64`, rank 1, `shared` | `R + 1` | Exclusive prefix offsets of each row's units. `[0] == 0`, non-decreasing, `[R] == U`. |
| `levels[1].owner` | `int64`, rank 1, `shared` | `U` | For each unit, the **position** of its owning row in `[0, R)`. |

Generally, level `k`'s `offsets` has extent *(count of level `k + 1` units)* `+ 1`
— with the last level's parent count being `R` — and level `k`'s `owner` has
extent *(count of level `k` units)*, which for level 0 is the packed extent `P`.
A single-level declaration is the ordinary flat case: `offsets` is `R + 1` long
and `owner` is `P` long.

The levels say *level*, not modality. For a video encoder they read as
frames-to-clips and clips-to-rows; for audio, frames-to-windows and
windows-to-rows; for a text reranker, tokens-to-segments and segments-to-rows.
The schema contains no `clip`, no `frame`, and no `window`, and a runtime
composing the chain never learns which it is holding.

**Composition is what makes a row addressable.** Because units of a row are
contiguous and positions of a unit are contiguous, row `r` owns the half-open
packed range

```
[ levels[0].offsets[ levels[1].offsets[r] ] ,
  levels[0].offsets[ levels[1].offsets[r + 1] ] )
```

which is a single contiguous span, hence sliceable as an alias
([§5.1](#51-grouped-buffers-aliasing-residency-and-what-padding-costs)). The same
composition with `r` replaced by a unit index gives that unit's span directly
from `levels[0].offsets`.

#### Complete example, with numbers

Two requests. Row 0 owns two clips of 3 and 1 frames; row 1 owns one clip of
2 frames. So `R = 2`, `U = 3`, `P = 6`, and frames are padded to 64 patches:

```yaml
components:
  clip_encoder:
    implementation: { kind: onnx, artifact: encoder.onnx }
    batch_capacity:
      uniform_dimensions: [features]
      budgets:
        - { dimensions: [clips],  max_total: 4 }        # U <= 4
        - { dimensions: [frames], max_total: 64 }       # P <= 64
        - { dimensions: [frames, patches], max_total: 65536 }
    ports:
      inputs:
        pixel_values:                                   # [6, 64, 128] here
          dtype: float32
          rank: 3
          shape: [frames, patches, features]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: frame_offsets, owner: frame_owner }
              - { offsets: clip_offsets,  owner: clip_owner }
          padding:
            - { dimension: patches, valid_lengths: patch_lengths }
        patch_lengths:                                  # [6]
          dtype: int64
          rank: 1
          shape: [frames]
          batch_layout: { kind: shared }
        frame_offsets:                                  # [4] = clips + 1
          dtype: int64
          rank: 1
          shape: [frame_offsets_len]
          batch_layout: { kind: shared }
        frame_owner:                                    # [6] = frames
          dtype: int64
          rank: 1
          shape: [frames]
          batch_layout: { kind: shared }
        clip_offsets:                                   # [3] = rows + 1
          dtype: int64
          rank: 1
          shape: [clip_offsets_len]
          batch_layout: { kind: shared }
        clip_owner:                                     # [3] = clips
          dtype: int64
          rank: 1
          shape: [clips]
          batch_layout: { kind: shared }
      outputs:
        frame_features:                                 # [6, 16, 512] here
          dtype: float32
          rank: 3
          shape: [frames, tokens, hidden]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: frame_offsets, owner: frame_owner }
              - { offsets: clip_offsets,  owner: clip_owner }
          packed_extent: preserved                      # see rule 5 in §4
        clip_embeddings:                                # [3, 512] here
          dtype: float32
          rank: 2
          shape: [clips, hidden]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: clip_offsets, owner: clip_owner }
          packed_extent: preserved
        media_tokens:                                   # [T, 512], T decided by the graph
          dtype: float32
          rank: 2
          shape: [media_tokens_total, hidden]
          batch_layout:
            kind: token_packed
            axis: 0
            levels:
              - { offsets: media_token_offsets, owner: media_token_owner }
              - { offsets: clip_offsets,        owner: clip_owner }
          packed_extent: produced                       # companions are outputs
        media_token_offsets:                            # [4] = clips + 1
          dtype: int64
          rank: 1
          shape: [media_token_offsets_len]
          batch_layout: { kind: shared }
        media_token_owner:                              # [T]
          dtype: int64
          rank: 1
          shape: [media_tokens_total]
          batch_layout: { kind: shared }
```

Concrete companion values for the grouping above:

```
clip_offsets  = [0, 2, 3]        # rows + 1 = 3 entries; row 0 owns clips [0,2), row 1 owns [2,3)
clip_owner    = [0, 0, 1]        # clips = 3 entries
frame_offsets = [0, 3, 4, 6]     # clips + 1 = 4 entries; clip 0 owns frames [0,3), clip 1 [3,4), clip 2 [4,6)
frame_owner   = [0, 0, 0, 1, 2, 2]  # frames = 6 entries
patch_lengths = [64, 64, 64, 16, 36, 36]   # per frame, <= the padded extent 64

pixel_values   : [6, 64, 128]    # frames, padded patches, features
frame_features : [6, 16, 512]
clip_embeddings: [3, 512]
```

Row 1's frames are `[ frame_offsets[clip_offsets[1]], frame_offsets[clip_offsets[2]] ) = [ frame_offsets[2], frame_offsets[3] ) = [4, 6)` —
one contiguous span of two frames, aliased, not copied. Row 1's clips are
`[2, 3)` of `clip_embeddings`, likewise contiguous.

The three outputs show the three cases rule 5 of
[§4](#4-strict-token_packed-validation) distinguishes: `frame_features` preserves
the input's packed extent, so it reuses the input companions unchanged;
`clip_embeddings` preserves the *outer* level's extent and therefore reuses only
that level's pair, dropping the inner level it has consumed; `media_tokens` has
an extent the graph decides, so its inner level's companions are the component's
**own outputs** while its outer level, whose clip→row mapping the component did
not change, reuses the input pair.

```yaml
preprocessing:
  video:                      # a video program; the image program is the same shape
    transforms: [...]
    outputs:
      - { source: patches, name: media.pixel_values,   content: pixels,        dtype: float32 }
      - { source: plen,    name: media.patch_lengths,  content: valid_lengths, dtype: int64 }
      - { source: foff,    name: media.frame_offsets,  content: pack_offsets,  dtype: int64 }
      - { source: fown,    name: media.frame_owner,    content: pack_owner,    dtype: int64 }
      - { source: coff,    name: media.clip_offsets,   content: pack_offsets,  dtype: int64 }
      - { source: cown,    name: media.clip_owner,     content: pack_owner,    dtype: int64 }
```

Two content roles, not four: `pack_offsets` and `pack_owner` are level-agnostic,
and which level a value serves is stated by the `levels` list that references it,
not by its role. A third role, `valid_lengths`, already exists on the audio side
(`crates/onnx-genai-metadata/src/schema/mod.rs:349-364`) and is generalized rather
than duplicated.

`pack_owner` values carry a **position, never a request identity**
([§8.3](INFERENCE_METADATA_DECISIONS.md#83-no-row-identity)). They are exactly as
persistable as a `row_selection` value: not at all. A runtime that permutes rows
recomputes the chain ([§5.2](#52-compaction-lifts-a-row-selection-through-the-ownership-chain));
it does not carry owners across invocations.

**What batching needs from a modality vocabulary, and what it does not decide.**
The roles above are generic and shared. What each modality contributes is only
the *semantic* values a grouped encoder consumes: for video, per-item frame
counts and whatever frame-selection facts the program applies (sampling stride,
target frame count, per-frame geometry). Those belong to a `preprocessing.video`
program — which does not exist today
([§2.7](#27-video-is-expressible-on-the-output-side-and-absent-on-the-encoder-side))
and whose full transform vocabulary is deliberately **out of scope here**. This
design states only the interface: a program that emits per-item lengths and the
ownership levels is sufficient for grouping, whatever its transform list turns
out to be.

### 3.4 Worked cases: images, video, audio, text

Every row below is the *same* component declaration with levels added or removed
and symbols renamed. Nothing in the runtime path distinguishes them.

| Encoder | Packed position | Levels (innermost → outermost) | Free dimensions and how they are reconciled | Typical `uniform_dimensions` |
| --- | --- | --- | --- | --- |
| Image | one image | images → rows | patches vary per image → pad `patches` with per-image lengths | `features` |
| Video | one frame | frames → clips, clips → rows | frame count varies per clip → the frame level; patches vary per frame → pad `patches` with per-frame lengths | `features`, plus `patches` when the encoder is resolution-pinned |
| Audio | one spectrogram frame | frames → windows, windows → rows | frames vary per window → the frame level; bins are fixed | `bins` |
| Text | one token | tokens → segments, segments → rows | tokens vary per segment → the token level | none |

**Variable clips, frames, and resolutions.** Three separate questions, answered
separately:

- *A request carries a different number of clips than another.* Item-count
  raggedness, handled by the clips→rows level. A group is formed from items, not
  from requests, and a request contributing zero items contributes an empty span
  ([§4](#4-strict-token_packed-validation) rule 9).
- *Two clips have different frame counts.* Item-nesting raggedness, handled by
  the frames→clips level — no padding at all, since frames are packed. If the
  artifact genuinely requires a fixed frame count, the package instead pins the
  frame count per clip through `uniform_dimensions` and only equal-length clips
  co-batch. A runtime **MUST NOT** reconcile the difference by trimming or
  resampling frames — dropping or interpolating a frame to fit a group changes
  what the caller asked for, and silently, which
  [§6](#6-fail-closed-behavior-compatibility-and-defaults) forbids.
- *Two clips have different resolutions.* Genuinely the package's call, stated in
  `uniform_dimensions`. A resolution-agnostic encoder (patchified input, position
  from coordinates) leaves `patches` free and lets mixed-resolution items group
  under per-frame lengths. A resolution-pinned encoder lists `patches`, and the
  scheduler then forms groups only from items that agree — a compatibility test
  on declared extents, never on a model name or a resolution table.

**Batch compatibility is therefore a derived predicate, not a policy.** Two items
may share an invocation exactly when they belong to the same component, agree on
every `uniform_dimensions` extent, and every dimension on which they disagree is
either padded with declared lengths or consumed by the packed axis — and the
resulting group must still satisfy every `budgets` entry
([§3.5](#35-budgets-bind-materialized-footprint)). The scheduler evaluates that
predicate over declared contracts; it contains no per-modality branch and no
notion of what a frame is.

### 3.5 Budgets bind materialized footprint

Four quantities are routinely conflated, and video makes the conflation
expensive. They are distinct, they are owned by different layers, and only one of
them is metadata:

| Quantity | What it counts | Bound by | Owner |
| --- | --- | --- | --- |
| **Request rows** | Concurrent requests in a decode batch | decode-side `BatchingCapability` and scheduler config | Engine, unchanged by this design |
| **Items** | Units at an ownership level — images, clips, windows, segments | a `budgets` entry on that level's symbol | Metadata declares the bound; scheduler picks a size at or below it |
| **Packed positions** | Entries on the physically packed axis — frames, tokens | a `budgets` entry on the packed symbol | Same |
| **Memory** | Bytes the group will occupy on the device | runtime measurement and allocator state | Runtime only; **never** metadata |

Request rows and items are independent in both directions. One request can
contribute eight clips, so items can exceed rows; seven requests can contribute
one image between them, so rows can exceed items. A grouping layer that reuses
the decode row bound as an item bound is wrong in both directions, and the decode
path's own `ContinuousBatchManager` already refuses multi-row prompts outright
(`crates/onnx-genai-engine/src/batched.rs:330-342`), so there is no existing
number to borrow even if borrowing were correct.

**A budget bounds what the invocation materializes, not what is nominally
valid.** The distinction only matters for padded dimensions, and there it matters
a great deal:

- **A packed dimension's footprint is the sum of the participating items' valid
  extents** — which is exactly the packed extent, since packing stores no
  padding. For the example above, `frames` has footprint `P = 6`.
- **A padded dimension's footprint is the enclosing count times the padded
  extent** — the rectangle the runtime actually allocates and the kernel actually
  reads, *not* the sum of the valid lengths. For the example above, `patches`
  materializes `6 × 64 = 384` patch slots while only `280` are valid; the budget
  binds `384`. Budgeting the valid sum would let a group of one long and fifteen
  short items pass a budget it then blows through by an order of magnitude, which
  is the exact failure padding causes and the reason the two are distinguished.
- **A composed entry multiplies the footprints of its symbols, in order.**
  `{ dimensions: [frames, patches], max_total: 65536 }` binds
  `6 × 64 = 384` here. This is how a package bounds an activation-shaped
  quantity — total padded patch slots — without naming bytes.
- **`request_expanded` contributes `rows × factor`** on its axis, by the same
  rule: a fixed factor is a padded extent that never varies.

A group **MUST** satisfy every entry. Items and packed positions bind
independently: four one-frame clips and four sixty-frame clips both satisfy a
`clips` budget of 4 and differ by more than an order of magnitude in work, while
one hundred one-frame clips is within any frame budget and beyond most position
tables.

Memory stays out. Budgets are *correctness* bounds — what the artifact tolerates
— and a runtime that can afford less than the bound groups less, using its own
measurement. Metadata carrying a memory number would be a cost model, which
[§1.2](INFERENCE_METADATA_DECISIONS.md#12-non-goals) forbids, and would be wrong
on the first machine that differs from the one it was measured on.

---

## 4. Strict `token_packed` validation

A packed layout is only usable if its companion values are exactly what the
consumer assumes. Today none of that is checked
([§2.6](#26-validation-of-packed-values-is-close-to-absent)). Rules 1–8 are
load-time validator rules with negative fixtures; rule 9 states the
invocation-time preconditions, which are a different set with a different cost
profile and are listed separately for that reason.

1. **Names resolve.** Every level's `offsets` and `owner` **MUST** name values
   declared in the same scope as the packed value. A dangling name is rejected,
   naming the packed value, the level, and the missing name.
2. **The packed axis is axis 0.** For any packed value consumed or produced by a
   component that declares `batch_capacity`, `axis` **MUST** be `0`. The reason is
   mechanical, not stylistic: a no-copy view is a contiguous element window over
   the owner's allocation, and the aliasing API says so outright — "a slice along
   an inner axis is not a contiguous range, and asking for one here would silently
   return the wrong elements. Callers that need one must copy"
   (`crates/onnx-genai-ort/src/value.rs:1521-1537`). With `axis: 0`, every row's
   and every unit's span is a contiguous range, so splitting is
   `alias_with_offset` and costs nothing. With an inner packed axis, every split
   is a strided gather — a full copy of the payload on the device, per row, on
   every invocation — which is the cost grouping exists to avoid. A package that
   needs an inner packed axis is rejected at load with that explanation, rather
   than silently paying it.
3. **Levels are ordered, non-empty, and at most two.** `levels` **MUST** contain
   one or two entries, ordered innermost first
   ([§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce)).
   Two levels — parts in items, items in rows — is what every known workload needs
   (frame → clip → request, frame → window → request, token → segment → request),
   and each additional level multiplies the validation surface, the split
   implementation, and the corruption cases that must be tested. The bound is
   stated rather than left implicit so that a third level is a deliberate schema
   change with its own design, not something a package can assert into existence.
   A three-level declaration is rejected at load, naming the value and the levels.
4. **Companion contracts are exact.** Every `offsets` and `owner` **MUST** be
   `int64`, rank 1, and `batch_layout: { kind: shared }`. Their declared extent
   symbols **MUST** follow [§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce):
   level `k`'s `owner` carries the count of level `k` units (the packed extent
   symbol at level 0), and level `k`'s `offsets` carries a distinct symbol that
   resolves at invocation time to the parent count plus one. `offsets` is
   `shared` rather than `request_aligned` for a structural reason: an exclusive
   prefix sum is **not permutation-followable**. Permuting rows does not permute a
   prefix-offset vector, it invalidates it. Labeling it `request_aligned` would
   invite a runtime to gather it during compaction and silently produce nonsense.
   This resolves the contradiction in
   [§2.6](#26-validation-of-packed-values-is-close-to-absent) in favor of the
   fixture and of [§10.1](INFERENCE_METADATA_DECISIONS.md#101-independent-encoder-batching);
   the `ir.rs` doc comment is corrected when the rule lands.
5. **Output-side raggedness has a declared producer.** A packed **output** may
   reuse an input's companions at a level only when the mapping at that level is
   **extent-preserving** — one output unit per input unit, in the same order. The
   package says which by declaring `packed_extent` on the output's layout:
   - `preserved` — the level's units correspond one-to-one with the input level's
     units. The validator checks that the referenced companions belong to an input
     port of the same component and that the two declare the same extent symbol at
     that level. A component may *drop* inner levels it has consumed (pooling
     frames into clips) and keep the outer ones; it may not invent correspondence.
   - `produced` — the graph decides the extent, so the level's `offsets` and
     `owner` **MUST** themselves be declared **outputs** of the same component.
     Reusing an input companion here is rejected at load, naming both values and
     the level, because the input's offsets describe an extent the output does not
     have. This is the case for token-merging encoders, variable-length poolers,
     and any graph whose output length is data-dependent.
   Mixed chains are normal and are declared per level: `media_tokens` in
   [§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce)
   produces its inner level and preserves its outer one. An output that declares
   neither is rejected: the runtime would have to guess whether the input's
   offsets still describe the result, and guessing wrong splits the payload at
   the wrong boundaries with no error at all.
6. **Shared companions agree.** Two packed values naming the same `offsets` at the
   same level **MUST** declare the same extent symbols; otherwise one of them is
   packed against a grouping that does not describe it.
7. **No double spelling on one dimension.** A packed value **MUST NOT** declare a
   `padding` entry on the dimension its layout packs
   ([§3.2](#32-tensorcontractpadding)). A `padding` entry on a *different*
   dimension is the normal composition of the three kinds of raggedness
   ([§3](#3-the-proposed-metadata-surface)). Likewise a port **MUST NOT** be both
   `request_expanded` and packed on the same axis
   ([§3.1](#31-workflowcomponentbatch_capacity)).
8. **A packed emit publishes its companions, and serving admits them.** If a
   workflow emits a `token_packed` output, every level's `offsets` and `owner`
   **MUST** also be declared outputs; without them the consumer receives a ragged
   buffer it cannot split. That collides with an existing serving rule: a serving
   workflow rejects any emitted value of rank > 0 that declares `shared`, on the
   grounds that a per-request result must declare `request_aligned` or
   `token_packed` (`crates/onnx-genai-metadata/src/validation.rs:3313-3321`), and
   companions are `shared` by rule 4. The **minimal coherent resolution** is a
   carve-out rather than a new layout: the serving rule admits a `shared` emitted
   value **iff** it is `int64`, rank 1, and named as an `offsets` or `owner` of
   some other emitted value's layout **in the same workflow**. No new
   `BatchLayout` variant, no new companion kind, and the admission condition is
   decidable from the declared outputs alone. Anything else `shared` and rank > 0
   is still rejected with the existing message. The runtime side of the carve-out:
   a companion is **never compacted and never split like a payload**. When results
   are delivered per request, the runtime hands each request its own payload span
   plus **rebased** offsets for that span — level offsets recomputed relative to
   the request's own start, so a single-request consumer sees a well-formed chain
   beginning at zero — and it **MUST NOT** deliver invocation-global `owner`
   values, which are positions within a grouping the caller cannot see
   ([§8.3](INFERENCE_METADATA_DECISIONS.md#83-no-row-identity)). Per-request owner
   values, where a consumer wants them, are derived from the rebased offsets.
9. **Invocation-time preconditions.** Load-time rules cannot check values, only
   contracts, so the following are checked per invocation, before the payload is
   consumed, and each failure is an error naming the value, the level, the index,
   and the two facts that disagree — never a truncation, a clamp, or a
   best-effort split:
   - extents resolve consistently: every level's `offsets` extent equals its
     parent count plus one, every level's `owner` extent equals its own unit
     count, and level 0's `owner` extent equals the packed extent;
   - offsets are well-formed: `offsets[0] == 0`, non-decreasing, and the last
     entry equals the child count at that level;
   - owners are consistent with offsets — `owner[i]` lies in range and the units
     of a parent are contiguous — **at every level**, so a frame owner that points
     outside its clip fails the same way a clip owner that points outside its row
     does;
   - every `padding` entry's `valid_lengths` has the exact companion shape from
     [§3.2](#32-tensorcontractpadding), and every entry is in
     `[0, padded_extent]`;
   - every `budgets` entry is satisfied by the assembled group's materialized
     footprint ([§3.5](#35-budgets-bind-materialized-footprint)), and every
     `uniform_dimensions` symbol has one extent across the group;
   - **zero items is a valid group state, not an error.** A request whose span is
     empty (`offsets[r] == offsets[r + 1]`) receives an empty span — rank
     preserved, extent 0 on the packed axis — and the runtime **MUST NOT**
     fabricate a placeholder item, and **MUST NOT** treat the absence as a
     failure. A zero-element alias is already a supported value
     (`crates/onnx-genai-ort/src/value.rs:1567-1573`). A group whose total packed
     extent is zero is **not invoked at all**; every participating request
     receives its empty span directly.
   **These checks never cause a host transfer.** Companions the runtime itself
   constructed are host-resident by construction, so the arithmetic above is free.
   Companions a component *produced* (`packed_extent: produced`) may be
   device-resident, and the runtime **MUST NOT** copy them to the host merely to
   validate them; for those it checks dtype, rank, and resolved extents — facts
   available without reading data — and performs the value-level arithmetic only
   on the transfer it must already make to split the result. That transfer moves
   the companion vectors only, never the payload, and it is counted and
   attributed like any other copy
   ([§5.1](#51-grouped-buffers-aliasing-residency-and-what-padding-costs)).

---

## 5. Ownership split

One fact, one owner. Nothing below is a new layer; each row is the existing
owner of that layer doing its existing job over a new declared fact.

| Layer | Owns | Explicitly does not own |
| --- | --- | --- |
| **Metadata** | That a component may be grouped, which dimensions must agree, what footprint it tolerates, how raggedness is expressed at every level, and who produces each companion. | Whether to group, how many to group, when, or how fast it will be. |
| **Modality vocabulary** | The *semantic* values a program produces — pixels, per-item lengths, grid geometry — and the transforms that produce them. | Anything about grouping. A content role never says a dimension is batchable; it says what the numbers mean. |
| **Preprocessor** | Producing per-item tensors and, when a program declares them, the ownership companions for the items it was handed. | Deciding which items are handed to it; any request identity; any cross-invocation state. |
| **Scheduler** | Deciding which pending work items co-batch, by evaluating the declared compatibility predicate ([§3.4](#34-worked-cases-images-video-audio-text)) under `budgets`; admission, fairness, latency-versus-throughput, deadlines, backpressure. | Model identity; modality; tensor construction; padding; any knowledge of what the component computes. |
| **Interpreter** | Building the grouped invocation: broadcasting `shared` inputs, padding only where a `padding` entry exists, packing only where a packed layout exists, invoking once, and splitting outputs back to rows through every declared level — without a host round-trip for values already device-resident. | Group composition policy; backend selection; storage layout. |
| **Backends (ORT and native)** | Executing the grouped invocation through the one component-execution seam, with identical results, and *declaring whether they can*. | Any batching decision of their own. A backend never has a private grouping path. |

The seam that makes the last row true is `WorkflowComponentBackend`
([`../architecture/NATIVE_WORKFLOW_BACKEND.md` §3](../architecture/NATIVE_WORKFLOW_BACKEND.md)):
grouping is built above it in the interpreter, so ORT and native receive the
same named tensors and there is exactly one grouping implementation.

Backend grouped-execution support is a **runtime** fact, asked *before* a group
is formed, never a metadata field and never discovered by attempting one. The
interpreter asks the seam whether the backend can execute this component grouped;
a backend that answers no simply never receives a group, and declining to group
is always safe ([§6](#6-fail-closed-behavior-compatibility-and-defaults)).
What is forbidden is the other order: forming a group, handing it to a backend
that has not proven it, and discovering the answer as a wrong result or a crash.
Equally forbidden is degrading *after* a group is formed — a backend that accepted
a group and then ran the items one at a time makes a performance regression
indistinguishable from correct behavior (Rule 4). A decline **MUST** be
observable — counted and attributable — because "grouping silently never
happened" and "grouping happened and did not help" are different bugs with the
same throughput number.

The scheduler's new work-item queue is generic over components. It reads
`batch_capacity` and the participating port contracts; it never reads a
component name, a modality, or an artifact filename
([`SCHEDULING.md`](SCHEDULING.md) owns the admission and preemption policy that
this queue plugs into).

### 5.1 Grouped buffers: aliasing, residency, and what padding costs

Grouping exists to save time. A grouping layer that assembles its input by
copying every item through host memory can consume the whole win before the
kernel starts, so buffer handling is part of the design rather than an
optimization to be added later.

The currency is unchanged: `onnx_genai_ort::Value` is already a device-capable
handle exposing `is_host_resident`, `device_id`, `try_alias_clone`,
`alias_with_offset`, and `from_external_memory`, so a device allocation can be
exposed as a device-resident value **without a host round-trip**
([`../architecture/NATIVE_WORKFLOW_BACKEND.md` §3–§4](../architecture/NATIVE_WORKFLOW_BACKEND.md)).
Grouping therefore inherits residency rather than inventing it, and metadata
gains no residency field — transport, cache identity, and residency stay
runtime-owned
([§10.2](INFERENCE_METADATA_DECISIONS.md#102-externally-suppliable-results)).

Three rules follow:

- **A device-resident item stays device-resident when it joins a group.** The
  interpreter **MUST NOT** move an already-resident value to the host and back in
  order to concatenate it. If it cannot assemble the group on the device — mixed
  devices, an allocator that cannot give a contiguous span — it declines to
  group, which is safe, rather than paying a round-trip that grouping was
  supposed to avoid.
- **Splitting a packed output is an aliasing operation.** This is the practical
  reason [§4](#4-strict-token_packed-validation) rule 2 pins the packed axis to 0
  and the ownership rules demand contiguity: a row's span is then a contiguous
  element range, so it is handed back as an alias over the same allocation
  (`alias_with_offset`, `crates/onnx-genai-ort/src/value.rs:1538-1546`), not a
  per-row copy.
- **Any copy that does happen is attributable.** Where an alias is impossible —
  and for `packed_extent: produced` companions, which must reach the host to be
  read at all — the copy is permitted but **MUST** be counted and reported
  alongside the throughput numbers ([§9](#9-e2e-acceptance-matrix) row 19). An
  unmeasured host-device transfer inside the grouping path is how a batching
  change ships as a regression while its own benchmark says it helped.

This also settles a question that looks like a preference and is not: **padding
always materializes, packing can alias.** Padding an item to the group's extent
writes a new, larger buffer for every participating item; packing concatenates
items end to end and leaves each item's span addressable in place. Both are
correct, and a package may only have one of them available, but where a package
can express either, packing is the cheaper spelling on both edges of the
invocation and padding's cost should be a deliberate choice rather than a
surprise. It is also why a padded dimension's budget is charged the whole
rectangle ([§3.5](#35-budgets-bind-materialized-footprint)): for video, padding
sixteen clips' frames to the longest frame's patch count can multiply the input
buffer several-fold, and every one of those bytes is a device write that produces
nothing.

### 5.2 Compaction lifts a row selection through the ownership chain

Request rows and packed items are different counts, indexed differently, and the
row ABI is defined on rows: `compact(selection)` means "destination `i` holds the
state of `selection[i]`", sources may repeat — beam and speculative cloning do
exactly that — and any position absent from the selection is dropped
(`crates/onnx-genai-engine/src/pipeline/row_state.rs:20-49`). A packed value has
no request axis at all: `BatchLayout::request_axis()` returns `None` for
`TokenPacked` (`crates/onnx-genai-metadata/src/schema/ir.rs:67-72`) while
`is_row_scoped()` still reports true, which is precisely the case a naive
implementation would gather along the wrong axis.

So a row `selection` is never applied to a packed axis directly. It is **lifted**
through the chain, outermost level inward:

1. For each destination row `d`, take source row `s = selection[d]` and its unit
   range `[outer_offsets[s], outer_offsets[s + 1])`.
2. Concatenating those ranges in destination order gives a **unit permutation**:
   the list of source unit positions the new grouping contains, in the new order.
   A repeated source row repeats its units; a dropped row contributes none.
3. Apply the same step one level in, using the unit permutation and the inner
   offsets, to obtain the **packed-position permutation**.
4. Recompute every level's `offsets` as a prefix sum over the new order, and
   every level's `owner` from the new offsets. Companions are **recomputed, never
   gathered** — a prefix sum is not permutation-followable
   ([§4](#4-strict-token_packed-validation) rule 4).
5. Apply the packed-position permutation to the payload. Where the permutation is
   order-preserving and drops only whole trailing spans, the result is a
   contiguous prefix and the "permutation" is a free reshape; otherwise it is one
   gather along axis 0.

Two consequences are contractual. **`row_scope.axis` is never the packed item
axis** ([§3.1](#31-workflowcomponentbatch_capacity) rule 7): a component's
row-scoped state is compacted with the row selection, exactly as it is today,
while its packed payload is compacted with the lifted permutation, and conflating
them would index one with the other's positions. And **a repeated source row
duplicates its items**, which is well-defined here precisely because owners are
positions rather than identities — the duplicate is a second, independent set of
positions, not two references to one.

---

## 6. Fail-closed behavior, compatibility, and defaults

**Defaults.** `batch_capacity` absent, `padding` absent, and no packed layout is
the whole of today's behavior: one item per invocation, no padding, no packing.
Every package in `tests/fixtures/onnx_genai_workflows/` keeps byte-identical
execution, which is the first row of the acceptance matrix
([§9](#9-e2e-acceptance-matrix)) rather than an assertion.

### 6.1 Schema evolution: what actually happens to an old runtime

The metadata structs are **closed**. `InferenceMetadata` itself is
`#[serde(deny_unknown_fields)]`
(`crates/onnx-genai-metadata/src/schema/mod.rs:36-38`), and the IR module carries
45 more of them — including `TensorContract` (`ir.rs:5-7`), `BatchLayout`
(`ir.rs:35-36`), `ComponentPorts` (`ir.rs:90-91`), and `WorkflowComponent`
(`ir.rs:695-697`), which are exactly the structs this design extends. So the
consequence of adding `batch_capacity`, `padding`, or `levels` is not that an old
runtime ignores them. **An old runtime fails to parse the document at all**, with
a serde error naming an unknown field, and refuses the entire package — including
the parts it fully supports.

The doc comment on `schema_version` currently says the opposite: that additive
fields "keep the same major version and rely on the forward-compatible 'ignore
unknown fields' rule" (`crates/onnx-genai-metadata/src/schema/mod.rs:47-53`).
Given `deny_unknown_fields` on the same struct three lines above, that rule does
not exist in this codebase. The comment is wrong today, independently of this
design, and correcting it is part of P1.

Nor is the version field a gate today. `SCHEMA_VERSION` is `"v1"`
(`crates/onnx-genai-metadata/src/lib.rs:8`), `schema_version` is
`Option<String>` defaulting to v1 (`schema/mod.rs:54-56`), writers stamp it
(`crates/onnx-genai-genai-config/src/compatibility.rs:243`,
`crates/onnx-genai-comfyui-config/src/lower.rs:101`) — and **nothing validates
it**: there is no reference to `schema_version` anywhere in
`crates/onnx-genai-metadata/src/validation.rs`. A package declaring a future
version is accepted, then rejected field by field, which is the least
actionable order in which to discover the problem.

The migration mechanism this design requires, therefore, is not a compatibility
claim; it is a mechanism, and it lands in P1 **before** any new field is emitted:

- **A version gate that runs before struct deserialization.** Parse the document
  generically, read `schema_version`, and reject a document whose version this
  runtime does not support with one actionable error — the document's version,
  the highest supported version, and the fact that an upgrade is required — in
  place of a field-level serde error that names an internal struct field. The
  precedent is in this repository: `onnx-model-package` parses
  `<major>.<minor>`, rejects a major it does not know with exactly that shape of
  message, and only then reads the rest
  (`crates/onnx-model-package/src/lib.rs:563-579`).
- **A minor bump for this surface.** `batch_capacity`, `padding`, `levels`, and
  the ownership content roles are additive to a v1 document, so they raise the
  **minor** version. A runtime supporting `v1.0` refuses a `v1.1` document with
  the gate's message rather than a serde message; a runtime supporting `v1.1`
  reads both. This is a real minimum-runtime requirement, stated once, and it is
  the reason the gate must precede the parse rather than accompany it.
- **Conditional emission.** A writer **MUST NOT** stamp `v1.1` or emit any of
  these fields for a package that does not use grouping. Packages that do not
  declare `batch_capacity` continue to serialize as v1.0 documents and continue
  to load on every existing runtime, byte for byte. The new minimum applies to
  the packages that actually need the new surface, and to no others.
- **No capability identifier, and not for the reason an earlier revision gave.**
  This design adds nothing to `required_capabilities`
  ([§4.3a](INFERENCE_METADATA_DECISIONS.md#43a-capability-admission-and-complete-built-in-catalogue)).
  The earlier rationale — that an old runtime would otherwise refuse a package it
  could execute — was **wrong**, since `deny_unknown_fields` makes it refuse the
  package anyway. The correct rationale is narrower and survives: a capability
  identifier states that *correct execution requires* a behavior, and no
  package's correctness requires that its encoder be batched. Version gating
  states "this document uses a newer vocabulary"; a capability states "you must
  do this or be wrong". Grouping is the first, never the second. A runtime that
  parses `v1.1` and chooses not to group is correct.

**Fail-closed is about the padding and packing, not the grouping.** The
asymmetry matters:

- Declining to group is always safe and is never an error.
- Grouping *without honoring* a declared `padding` entry or packed layout is a
  wrong-answer bug, so it is forbidden: if the runtime cannot construct the
  lengths or the offsets for some participating port, it **MUST NOT** group that
  invocation.
- A runtime **MUST NOT** fabricate padding for a value that has no `padding`
  entry on that dimension, **MUST NOT** invent an `offsets`/`owner` pair a
  package did not declare at any level, and **MUST NOT** group items that
  disagree on a `uniform_dimensions` extent.
- A runtime **MUST NOT** make two items compatible by changing them. Trimming
  frames, resampling a clip to a common frame count, downscaling to a common
  resolution, or truncating a token segment are all semantic changes to what the
  caller asked for. Padding with recorded lengths is the only sanctioned way to
  reconcile a free dimension, and packing is the only sanctioned way to avoid
  padding.
- A package that *declares* batchability it has not made expressible is rejected
  at load ([§3.1](#31-workflowcomponentbatch_capacity) rule 5), not at the first
  unlucky group.
- A runtime **MUST NOT** exceed any `budgets` entry, and **MUST NOT** treat a
  decode-side row bound as an item bound
  ([§3.5](#35-budgets-bind-materialized-footprint)).

### 6.2 An unproven backend is a backend that does not group

This is the concrete case today, not a hypothetical. The native path is reported
to crash above batch one for vision models: the batch benchmark limits native to
batch 1 and says so in two places — "the native runtime may segfault on batch>1
(a known bug)" and "batch>1 is known to segfault for vision models"
(`crates/onnx-genai-bench/src/bin/batch_vision.rs:122-123,243,250`), and its
`probe_native_max_batch` deliberately never attempts batch 2 so the process
survives. The regression test that would catch it,
`crates/onnx-runtime-session/tests/batch_vision_crash.rs`, asserts exactly the
right property — batch 2 must not crash *and* must equal the two batch-1 runs
element-wise — but it returns early when its model file is missing, so it is not
a standing guard.

**Readiness is per component implementation, operator class, and execution
provider — not one global flip.** The crash is reported for *vision* models,
which is a statement about the operator classes a convolutional encoder uses and
about how a provider implements them at batch > 1, not about the workflow
interpreter. So the answer the seam returns is a function of the specific
component's artifact, the operator classes it contains, and the provider it will
run on; a backend that has proven a transformer encoder at batch > 1 on one
provider has proven nothing about a CNN encoder on another. A single boolean
"native supports grouping" would assert exactly the thing the evidence does not
support, and would flip the CNN case to "supported" the moment any other case
passed.

Therefore, until parity is proven for a given (component, operator class,
provider) combination:

- The native backend **MUST** report grouped execution as unsupported for that
  combination, so the interpreter never forms a group for it
  ([§5.1](#51-grouped-buffers-aliasing-residency-and-what-padding-costs)).
  Reporting "no" is not a failure mode; it is the safe answer, and it costs only
  the optimization.
- The runtime **MUST NOT** discover the answer by attempting a grouped
  invocation. A crash is not a diagnostic, and a segfault in a serving process
  takes every unrelated in-flight request with it.
- **The real-model crash guard is retained, not replaced.**
  `crates/onnx-runtime-session/tests/batch_vision_crash.rs` runs a genuine CNN
  (MobileNetV2) and is the only test that exercises the operator classes the
  crash was reported against. It **MUST** keep running against a real
  convolutional model, and its skip-when-absent path **MUST** be closed so that a
  missing model file fails the job instead of quietly passing. A synthetic
  fixture is added *alongside* it — deterministic, tiny, checked in, and able to
  cover the shape variety the matrix needs — because the synthetic fixture proves
  the grouping machinery while the CNN proves the operator classes. Neither
  substitutes for the other, and dropping the CNN guard in favor of a synthetic
  graph would retire the only evidence about the actual defect.
- Flipping a combination's answer to "yes" is gated on P5
  ([§8](#8-execution-phases-and-pr-dag)): grouped-versus-solo equality under the
  native backend, on both the synthetic fixture and the CNN guard, recorded as a
  readiness entry for that (component, operator class, provider) triple. The gate
  is a test that runs in CI, not a judgement that the bug is probably fixed.
- ORT grouping is not blocked by native's answer. The two backends answer
  independently, per combination, which is the point of asking before forming
  rather than after.

**Fixtures are synthetic, tiny, and checked in.** Every acceptance row runs on a
generated fixture in `tests/fixtures/onnx_genai_workflows/`, built by a
`scripts/build_tiny_*.py` generator like every other fixture in the repository
(there are already ~40, including `build_tiny_vlm.py` and the `video` generation
workflow). Video grouping needs a new one: a **synthetic video-encoder fixture**
with a handful of frames at a small resolution, deterministic outputs so that
solo-versus-group equality is exact rather than approximate, and enough shape
variety to cover the matrix — clips of differing frame counts, requests carrying
zero, one, and several clips, both a resolution-pinned and a resolution-agnostic
variant, and one component whose output companions are `produced` rather than
`preserved`. No downloaded weights, no sample media files, no network access in
the test path; a fixture that cannot be regenerated from a script in this
repository is not a fixture, it is a dependency.

**Row semantics survive.** Grouping introduces no new identity. Items are
positional inside an invocation, exactly as rows are positional inside a batch
([§8.3](INFERENCE_METADATA_DECISIONS.md#83-no-row-identity)). A grouped
component that holds per-request state is row-scoped and therefore already
implements `compact`/`release`
([§8.6](INFERENCE_METADATA_DECISIONS.md#86-mandatory-row-abi)); the lifting rules
in [§5.2](#52-compaction-lifts-a-row-selection-through-the-ownership-chain) are
what keep the row selection and the packed positions from being confused for one
another.

---

## 7. What this deliberately does not add

**No model-specific booleans.** Not `is_vision`, not `is_video`, not
`qwen_vl_packing`, not `clip_style_encoder`, not `supports_image_batching`, not
`supports_video_batching`. A batching decision must be derivable from geometry
and declared bounds alone. If a new architecture needs a runtime branch to be
batched, the correct outcome is a new *generic* structural fact — or that the
architecture is simply not batchable here — never a name test (Rule 2).

**No modality in the batching path.** The contracts carry shape symbols, bounds,
lengths, and ownership levels; only a preprocessing program's content roles say
that a dimension counts frames rather than tokens. The interpreter and the scheduler
therefore have exactly one implementation of grouping, and adding a modality adds
a vocabulary and a fixture, not a branch. A "video batching" code path would be
the same defect as a "video attention" kernel.

**No `supports_batching` flag.** A boolean answers "may I?" but not "how much",
"on which dimensions", or "which dimensions must agree", so a runtime holding a
`true` still cannot build a group. `batch_capacity` carries the answer and its
absence carries the negative, which is why there is no separate boolean to keep
in sync.

**No component-global axis integers.** `batch_capacity` names shape symbols, not
axis indices, because one component's ports differ in rank and an index is only
meaningful against one of them
([§3.1](#31-workflowcomponentbatch_capacity)). The axis a value is packed on is a
property of that value's own layout, stated once, on that value.

**No second physical packed axis.** Nesting is levels of ownership over one
flattened axis. A second packed axis would make one geometry expressible two ways
and would put every per-row split on a strided gather
([§4](#4-strict-token_packed-validation) rule 2).

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

**No cost model.** `budgets` are correctness bounds. Metadata never carries
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
P1 schema surface + version gate ┬─────────────► P3a/P3b preprocessor produces
  batch_capacity, padding,       │                  ownership + length values
  levels, content roles,         │                          │
  pre-parse schema_version gate  │                 P3c synthetic fixtures
     │                           │                   image + video encoder
     ▼                           │                          │
P2 validation                    │                          │
  §3.1 rules 1-7, §4 rules 1-8   │                          │
     └──────────────┬────────────┘                          │
                    └──────────────────┬───────────────────┘
                                       ▼
                          P4 interpreter grouped invocation
                            group build · broadcast · pad ·
                            pack · invoke once · split ·
                            alias, never host round-trip
                                       │
                          ┌────────────┴────────────┐
                          ▼                         ▼
              P5 backend parity            P6 scheduler grouping
                ORT ≡ native, per-            work-item queue,
                triple readiness,             budgets by symbol
                CNN guard retained            / uniform_dimensions
                          └────────────┬────────────┘
                                       ▼
                             P7 E2E + performance
                               acceptance matrix §9
```

- **P0 — design of record (this change).** Docs only. No code, no schema.
- **P1 — schema surface and the version gate.** The gate lands *first within the
  phase*: read `schema_version` from a generic parse and reject an unsupported
  version with one actionable message before any struct is deserialized, matching
  `onnx-model-package`'s precedent
  (`crates/onnx-model-package/src/lib.rs:563-579`), and correct the false
  forward-compatibility comment at `schema/mod.rs:47-53`
  ([§6.1](#61-schema-evolution-what-actually-happens-to-an-old-runtime)). Then add
  `WorkflowComponent.batch_capacity`, `TensorContract.padding`,
  `BatchLayout::TokenPacked.levels`, and the `pack_offsets` / `pack_owner` /
  `valid_lengths` content roles
  ([§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce)),
  emitted only for packages that use them. Regenerate
  `schema/inference_metadata.schema.json` with `gen_schema`. Positive fixtures
  only; nothing reads the fields yet. Guards: `cargo test -p onnx-genai-metadata`
  including the committed-schema comparison, plus a test that a `v1.1` document
  is refused by the gate's message and that every existing fixture still
  serializes as `v1.0`.
- **P2 — validation.** Implement [§3.1](#31-workflowcomponentbatch_capacity)
  rules 1–7 and [§4](#4-strict-token_packed-validation) rules 1–8, each with a
  negative fixture asserting the exact message — including a three-level packing
  declaration, an inner packed axis, a `produced` output that reuses an input
  companion, and a `shared` rank-1 emit that is *not* a referenced companion and
  therefore stays rejected. Depends on P1.
- **P3 — preprocessor.** Two independent pieces, in this order.
  **P3a (one level):** let the image adapter accept N encoded items and emit one
  level's `pack_offsets` / `pack_owner` plus per-item `valid_lengths`; unit tests
  over the offset and owner arithmetic on the CPU, with zero-item and single-item
  rows covered.
  **P3b (nesting):** a frame-sequence producer — the input side that
  [§2.7](#27-video-is-expressible-on-the-output-side-and-absent-on-the-encoder-side)
  shows is missing — whose interface takes an ordered frame sequence per clip and
  emits the inner level's `pack_offsets` / `pack_owner` over the *same* flattened
  frame axis. P3b needs no new *batching* concept; it is the modality vocabulary
  that makes the nested level producible. Depends on P1, not on P2.
- **P3c (fixture):** a synthetic video-encoder workflow fixture generated by a
  `scripts/build_tiny_*.py` script and checked in under
  `tests/fixtures/onnx_genai_workflows/`, with tiny frames, deterministic
  outputs, variable frame counts, variable clip counts, both a resolution-pinned
  and a resolution-agnostic variant, and one `packed_extent: produced` output
  ([§6.2](#62-an-unproven-backend-is-a-backend-that-does-not-group)). It gates
  P4's video tests and every video row of the matrix. It does **not** replace
  `crates/onnx-runtime-session/tests/batch_vision_crash.rs`, whose real
  convolutional model is the only evidence about the reported crash's operator
  classes; closing that test's skip-when-absent path is part of P5.
- **P4 — interpreter.** Grouped invocation in the one interpreter: broadcast
  `shared`, pad where a `padding` entry says to, pack where `token_packed` says
  to, invoke once, split results back through every declared ownership level,
  recompute companions when a row selection is compacted
  ([§5.2](#52-compaction-lifts-a-row-selection-through-the-ownership-chain)), plus
  the [§4 rule 9](#4-strict-token_packed-validation) invocation-time precondition
  checks. Group assembly and splitting follow the aliasing and residency rules in
  [§5.1](#51-grouped-buffers-aliasing-residency-and-what-padding-costs): no host
  round-trip for a device-resident item, splits as aliases where the span is
  contiguous, and any unavoidable copy counted. The default path stays one item
  per invocation. Key test: grouped output equals sequential output row by row.
  Depends on P2, P3a, and P3c; the nested split is exercised by P3b.
- **P5 — backend parity, and the native readiness gate.** Run P4's grouped
  invocation under ORT and under the native backend through
  `WorkflowComponentBackend`; assert identical results and identical rejection
  messages. Readiness is recorded per (component implementation, operator class,
  execution provider) triple and never as one global flip
  ([§6.2](#62-an-unproven-backend-is-a-backend-that-does-not-group)): a triple
  that has not passed reports grouped execution as unsupported and therefore
  never receives a group, because the reported batch>1 vision crash
  (`crates/onnx-genai-bench/src/bin/batch_vision.rs:122-123,243,250`) is not
  something a serving path may discover at runtime. Passing requires **both** the
  P3c synthetic fixture and the retained real-CNN guard in
  `crates/onnx-runtime-session/tests/batch_vision_crash.rs`, whose
  skip-when-absent path is closed here so a missing model fails the job instead
  of passing quietly. Depends on P4 and P3c.
- **P6 — scheduler grouping.** A generic pending-work-item queue that forms
  groups under every `budgets` entry and the `uniform_dimensions` compatibility
  predicate ([§3.4](#34-worked-cases-images-video-audio-text)), with cancellation
  and compaction while a group is in flight
  ([§5.2](#52-compaction-lifts-a-row-selection-through-the-ownership-chain)), and
  with declined groupings counted so a missing win is distinguishable from an
  ineffective one. Depends on P4; independent of P5.
- **P7 — E2E and performance.** The full matrix in [§9](#9-e2e-acceptance-matrix)
  for both an image encoder and a video encoder, including grouped-versus-
  sequential throughput on fixed hardware and the host-transfer counters from
  [§5.1](#51-grouped-buffers-aliasing-residency-and-what-padding-costs). Depends
  on P5 and P6.

If P4 proves large — the split-by-owner path is the likeliest — it splits into
"pad and broadcast" and "pack and split", in that order, since padding needs no
companion values.

---

## 9. E2E acceptance matrix

Every row is an end-to-end test, not a unit assertion. "Solo" means the same
item executed alone through the same package; per-row equality against solo is
the correctness definition for every batching row. Rows 1–9 are modality-neutral
and are run against both an image encoder and a video encoder; rows 10–15 pin the
modality-specific geometry that motivated the design; rows 16–26 cover the
structural bounds, the memory path, compatibility, and backend readiness. Every
row runs on checked-in fixtures — the synthetic ones from P3c for grouping
behavior, plus the retained real-CNN guard for row 20's operator classes — with
no downloaded weights, no sample media, and no network in the test path.

| # | Scenario | What it must prove | Gate |
| --- | --- | --- | --- |
| 1 | **Non-batchable component.** Package declares no `batch_capacity`. | Execution is byte-identical to the pre-change baseline; the runtime never groups. | P4 |
| 2 | **Padded group.** Items differ on a free dimension with a declared `padding` entry and per-item `valid_lengths`. | Each row's output equals its solo output; padded positions never influence a real row; the companion's shape is exactly the outer axes of the padded value. | P4 |
| 3 | **Variable item size.** Items with genuinely different geometry grouped via `token_packed`. | Packed result splits back to per-row results equal to solo; no reliance on a common width. | P4 |
| 4 | **Variable item count per request, including zero.** Requests carrying 0, 1, and many items in one group, and a group in which *every* request carries zero. | A zero-item request receives an empty span with rank preserved and never a fabricated placeholder; an all-zero group is not invoked at all; offsets stay consistent with empty spans. | P4 |
| 5 | **Packed ownership.** `offsets`/`owner` round-trip, plus a deliberately corrupted `offsets` and a corrupted `owner`. | Correct split on the good case; on each corrupted case a loud error naming value, level, index, and the two disagreeing facts. | P4 |
| 6 | **Multi-request grouping.** Items from N concurrent requests in one invocation. | Per-row equality against solo, and no value from one request observable in another's row. | P6 |
| 7 | **ORT and native parity.** Rows 2, 3, 6, 11, and 12 under both backends. | Identical outputs and identical rejection messages; neither backend has a private grouping path. | P5 |
| 8 | **Concurrency and compaction.** Arrival, cancellation, and `compact(selection)` while a group is in flight, including a selection that repeats a source row and one that drops rows. | The selection is lifted through the ownership chain into an item permutation; companions are recomputed, never gathered; a repeated row duplicates its items; a cancelled request's items never reach another row. | P6 |
| 9 | **Incompatible group refused.** Items disagreeing on a `uniform_dimensions` extent — for a resolution-pinned encoder, two different resolutions — offered to the same group. | The scheduler forms two groups (or executes solo) and never reconciles by resizing; results equal solo. | P6 |
| 10 | **Variable frames per clip.** Clips of differing frame counts, flattened onto one packed axis with a frames→clips level. | Each clip's output equals its solo output; no frame is trimmed or resampled to fit; no temporal padding is materialized at all. | P4 |
| 11 | **Variable clips per request.** Requests carrying 0, 1, and several clips, clips themselves carrying differing frame counts. | Both levels round-trip over one flattened axis: composing `frame_offsets` through `clip_offsets` yields each row's contiguous frame span; per-row equality against solo. | P4 |
| 12 | **Mixed spatial and temporal raggedness.** One group where items differ in frame count *and* in per-frame patch count. | The frame level and the per-frame `valid_lengths` compose; the one-truth-per-dimension rule holds; each row equals solo. | P4 |
| 13 | **Nested ownership corruption.** The inner level's `offsets` and `owner` corrupted in turn: an owner outside its clip, a non-monotonic inner offset, an inner total disagreeing with the outer level's unit count. | Each case is a loud error naming value, level, index, and the two disagreeing facts — never a clamp or a partial split. | P4 |
| 14 | **Mixed-modality serving.** Image items and video clips in flight for the same engine. | Grouping is per component, not per request; an image group and a video group are formed by the same code with no modality branch; per-row equality against solo. | P6 |
| 15 | **A third modality reuses the path.** An audio (or text-segment) encoder declaring `batch_capacity` plus windows-in-rows ownership. | It batches with no new interpreter or scheduler code — the acceptance is that the diff is a fixture and a vocabulary, not a branch. | P6 |
| 16 | **Structural bounds at load.** A three-level `levels` chain; a packed value on an inner axis; a `batch_capacity` naming an unknown symbol; a symbol both pinned and budgeted; a free dimension with neither padding nor packing. | Each rejected at load with its own message naming the component, the value, and the symbol or level at fault; the valid two-level, axis-0 path is unaffected. | P2 |
| 17 | **Right-padding enforced.** A `valid_lengths` entry exceeding the padded extent, and a producer that left-pads. | Both rejected before the invocation, naming the item, the dimension, and the offending value; length arithmetic is done on host-resident companions with no device read. | P4 |
| 18 | **Budgets bind materialized footprint.** Groups that hit the item budget with few frames; groups that hit the packed budget with few items (one long clip); and a group whose *valid* padded extents would fit but whose materialized rectangle does not. | Each bound is enforced separately, none inferred from another; the padded dimension is charged `count × padded_extent`, not the sum of valid lengths; a composed entry multiplies its symbols; a decode-side row bound is never used as an item bound. | P6 |
| 19 | **No hidden host round-trip.** A group assembled from device-resident items, and a packed output split back to rows. | Transfer counters show no device→host→device traffic for already-resident values; per-row splits are aliases over the packed allocation where spans are contiguous; any unavoidable copy is counted and reported. | P7 |
| 20 | **Unproven backend declines, per triple, and says so.** Native reports grouped execution unsupported for the CNN encoder while ORT groups the same workload; a second component with a different operator class is evaluated independently. | Outputs are identical between the two paths; native never receives a group for an unproven triple; one triple passing never flips another; the decline is counted and attributable, not silent; no attempted grouped invocation on the unproven path. The real-model guard in `batch_vision_crash.rs` runs and fails the job if its model is missing. | P5 |
| 21 | **Version gate, not a serde error.** A `v1.1` document offered to a runtime that supports only `v1.0`, and a `v1.0` document offered to a `v1.1` runtime. | The first is refused before struct deserialization with one message naming the document version, the highest supported version, and the required upgrade — never an unknown-field error. The second loads and executes unchanged. | P1 |
| 22 | **Conditional emission.** Every existing fixture round-tripped through the writer. | No `batch_capacity`, `padding`, or `levels` field is emitted, the document still declares `v1.0`, and the bytes are unchanged, so no existing runtime's minimum moves. | P1 |
| 23 | **Output companions have a declared producer.** A `packed_extent: produced` output whose companions are component outputs; a token-merging graph whose output length differs from its input's; and a negative case declaring `preserved` while reusing an input companion of a different extent. | The first two split at the graph's own boundaries; the third is rejected at load naming both values and the level; no path ever splits a produced output with input offsets. | P2, P4 |
| 24 | **Serving admits companions, and only companions.** A serving workflow emitting a packed value with its `shared` rank-1 `offsets` and `owner`; and one emitting an unrelated `shared` rank-1 value. | The first validates and each request receives its own span with rebased, zero-based offsets and no invocation-global owner values; the second is still rejected with the existing message. | P2, P6 |
| 25 | **Companion validation causes no hidden transfer.** A group whose companions the runtime built, and a group whose companions a component produced on device. | Runtime-built companions are validated on the host at no transfer cost; produced companions are checked for dtype, rank, and extent without a device read, and the single companion-only transfer needed to split is counted and attributed — the payload never moves. | P4, P7 |
| 26 | **`request_expanded` participates.** A component with a `request_expanded` port at factor > 1 grouped alongside packed ports. | Ownership is arithmetic — entry `i` belongs to row `i / factor` — no companions are declared or required, footprint is charged as `rows × factor`, and a declaration that is both request-expanded and packed on one axis is rejected at load. | P2, P4 |
| 27 | **Performance versus sequential direct execution.** Same hardware, same items, grouped versus one-at-a-time, reported separately for image and for video. | Images/s, frames/s, clips/s, and per-request latency for both modes, plus the group sizes actually formed and the padding overhead paid (padded elements as a fraction of real ones). Per-row outputs identical. A regression at any reachable group size is reported, not hidden behind an average. | P7 |

Row 27 follows the measurement protocol already used for batched decode
([`NATIVE_BATCH_DECODE_2B_IMPL_SCOPING.md` §6](NATIVE_BATCH_DECODE_2B_IMPL_SCOPING.md)):
report the mechanism-level counters and the achieved group sizes, never a bare
wall-clock headline, and verify the device is idle before each run. Video is
reported in frames/s *and* clips/s because a group of few long clips and a group
of many short clips are different operating points that a single items/s number
would blur, and padding overhead is reported next to them because a padded group
can do several times the arithmetic of the packed group that produced the same
answer.

---

## 10. Open questions

1. **Group-size choice.** The scheduler needs a policy for trading first-item
   latency against group occupancy. That is runtime policy and stays out of
   metadata, but P6 should expose the chosen size so row 27 can attribute a
   result. Video sharpens it: one long clip can exceed the useful work of a whole
   image group, which is why `budgets` bound packed positions as well as items
   ([§3.5](#35-budgets-bind-materialized-footprint)) —
   what remains open is which of the two binds first under a given arrival
   pattern, and that is measurement, not schema.
2. **Cross-invocation reuse.** An encoder result declared
   `externally_suppliable`
   ([§10.2](INFERENCE_METADATA_DECISIONS.md#102-externally-suppliable-results))
   may already be cached. Grouping and reuse interact — a cached item should not
   occupy a group slot — but the cache key derivation is unchanged
   ([§11](INFERENCE_METADATA_DECISIONS.md#11-cache-correctness-dependencies)).
3. **Which modalities need packing versus padding.**
   [§3.3](#33-ownership-values-a-preprocessing-program-must-be-able-to-produce)
   makes the ownership roles available to any vocabulary. Whether a given
   modality's first package needs packing, or whether padding plus a
   `valid_lengths` companion suffices — plausible for fixed-window speech
   encoders and fixed-frame video encoders, implausible for native-resolution
   image encoders — is decided per package, from its geometry, not decided here.
4. **The depth cap, and what a third level would take.** `levels` is an ordered
   list, so a third level is a data change rather than a new field name, but it is
   capped at two ([§4 rule 3](#4-strict-token_packed-validation)) because the
   validation surface, the split implementation, and the compaction lifting in
   [§5.2](#52-compaction-lifts-a-row-selection-through-the-ownership-chain) each
   grow with depth. If a workload ever genuinely needs a third, lifting the cap is
   a deliberate schema change with its own evidence — not something a package can
   assert into existence.
5. **Where the frame-sequence producer lands.** P3b needs an input side that
   accepts an ordered frame sequence per clip; today the image path takes
   independent images and `temporal_patch_size` only replicates a frame
   ([§2.7](#27-video-is-expressible-on-the-output-side-and-absent-on-the-encoder-side)).
   Whether that is a new `preprocessing.video` program or a sequence-aware mode
   of the existing image program is a preprocessing decision, and it does not
   change any contract in this document.
6. **How coarse the native readiness key should be.** The gate is per
   (component implementation, operator class, execution provider)
   ([§6.2](#62-an-unproven-backend-is-a-backend-that-does-not-group)), which is
   deliberately conservative because the reported batch>1 vision crash may be one
   defect or several. Whether "operator class" is best keyed by op-set coverage,
   by a named set of operators, or by the artifact's own hash is a P5 question,
   and it is answerable from what the parity runs actually reveal. Until a triple
   passes, native declines and ORT groups, and the system is correct either way —
   which is the property that makes the unknown affordable.
7. **Whether a minor version is enough.** [§6.1](#61-schema-evolution-what-actually-happens-to-an-old-runtime)
   proposes `v1.1` for an additive surface behind a pre-parse gate, on the
   argument that a *gated* additive change is not a breaking change for any
   document that does not use it. The alternative reading — that under
   `deny_unknown_fields` every additive field is breaking and deserves a major
   bump — would move every existing package's minimum runtime for a feature it
   does not use. The proposal picks the narrower blast radius; the decision is
   worth confirming when P1 lands, since it sets the precedent for every later
   additive surface.
