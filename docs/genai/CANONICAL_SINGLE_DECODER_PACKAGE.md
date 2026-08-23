# The canonical single-decoder package

A single decoder is **not a special package shape**. It is a
`pipeline.workflow` with one ONNX component, declared with exactly the
constructs a multi-component workflow uses. There is no `model.io`, no
runtime lowering, and no compatibility path: a package that does not declare a
workflow does not load.

This page is the authoring reference for producing such a package — including
for republishing packages already on a model hub.

## The shape, in one picture

```yaml
schema_version: v1
model:
  max_sequence_length: 4096      # optional; model facts, never port names
pipeline:
  workflow:
    manifest:
      capabilities: [workflow_ssa, linear_effects, typed_emit, streaming_emit,
                     nested_control_flow, loop_induction_values,
                     serving_service_contract]

    inputs:
      request.input_ids:          # the prompt
        contract: {dtype: int64, rank: 2, shape: [batch, sequence],
                   batch_layout: {kind: request_aligned, axis: 0}}
        role: {kind: runtime, version: '1.0', role: prompt_tokens}
        source: {kind: request}
        required: true
      request.max_iterations:     # generation bound
        contract: {dtype: int64, rank: 1, shape: [1]}
        role: {kind: runtime, version: '1.0', role: max_iterations}
        source: {kind: request}
        required: false
        default: 4096             # an optional input MUST carry a default

    outputs:
      tokens:
        contract: {dtype: int64, rank: 2, shape: [batch, sequence],
                   batch_layout: {kind: request_aligned, axis: 0}}
        role: tokens
        stage: pre_adapter

    components:
      decoder:                    # the name is yours; recognition is structural
        implementation: {kind: onnx, artifact: model.onnx}
        ports:
          inputs:  {input_ids: {...}, attention_mask: {...}, position_ids: {...},
                    past_key_values.0.key: {...}, past_key_values.0.value: {...}}
          outputs: {logits: {...}, present.0.key: {...}, present.0.value: {...}}
          roles:                  # what each port MEANS
            input_ids: token_ids
            attention_mask: attention_mask
            position_ids: position_ids
            logits: logits
      token_policy:               # implemented by the runtime, not by a graph
        implementation: {kind: binding}
        contract: {id: onnx-genai.token-policy, version: '1.0'}
        ports:
          inputs:  {logits: {...}}
          outputs: {token: {...}, active: {...}, done: {...}, accepted_len: {...}}

    state: {...}                  # one cell per graph state tensor
    serving:
      active: active
      done: done
      accepted_len: accepted_len
      state_service:
        groups:
          decoder_kv:             # the cache, and the aliases that reach it
            kind: full_attention
            sequence_axis: 2
            layout: head_major_bnsh
            aliasing: permitted   # or forbidden / required
            ports:
              decoder:
                decoder_kv.0000: {input: past_key_values.0.key,
                                  output: present.0.key, role: key, layer: 0}
                decoder_kv.0001: {input: past_key_values.0.value,
                                  output: present.0.value, role: value, layer: 0}

    steps:
      - kind: loop
        termination: generation_eos
        continue_when: active
        max_iterations: request.max_iterations
        steps:
          - {kind: invoke, component: decoder, inputs: {...}, outputs: {...}}
          - {kind: invoke, component: token_policy, inputs: {...}, outputs: {...}}
          - {kind: emit, value: step.token, output: tokens, mode: append, when: active}
        carried: [...]
```

Do not hand-write this. See [Converting a package](#converting-a-package).

## The five rules

1. **Roles name ports; the graph owns their shapes.** A runtime never guesses a
   port from its spelling. `ports.roles` maps a port to what it *means*
   (`token_ids`, `inputs_embeds`, `attention_mask`, `position_ids`, `logits`,
   `hidden_states`, `encoder_hidden_states`, `audio_features`). A package may
   name its ports anything — `tests/fixtures/tiny-llm-explicit-io/` exists to
   prove exactly that.

2. **State lives in a `state_service` group, once.** The group's per-component
   aliases carry `input` (the `past` port), `output` (the `present` port),
   `role: key|value` and `layer`. The `layer` index is what preserves the
   graph's own layer order — port names live in a map whose key order is
   lexicographic, which would otherwise place layer 10 between layers 1 and 2.

3. **A fixed-capacity cache is declared once, as a fixed-capacity cache.** Use
   `update: {kind: indexed_scatter, write_indices_ports, kv_length_ports,
   capacity}`. Do **not** also describe its buffers as growing past/present
   pairs — the runtime reads the absence of growing pairs as "this cache does
   not grow", and a package claiming both gets a paged cache addressed with the
   wrong discipline.

4. **The token policy is a `binding`.** Sampling, stopping, logprobs and KV
   commit are the runtime's, and the workflow says so by declaring a component
   with the contract `onnx-genai.token-policy` and no artifact. This is what
   lets a single decoder keep the rich Rust sampler, paged KV, sessions and
   speculative decode — none of which has an in-graph representation.

5. **Declare the capabilities you use.** Validation computes the capabilities
   your structure requires and rejects a manifest that omits one, so the list
   is checkable rather than decorative.

## End-of-generation tokens

A model may end a turn with one token and a message with another. Both stop it,
so the declaration is a **set**:

```yaml
    inputs:
      package.eos_token_ids:
        contract: {dtype: int64, rank: 1, shape: [eos_count]}
        role: {kind: runtime, version: '1.0', role: eos_token_ids}
        source: {kind: literal}
        required: false
        externally_suppliable: true
        default: [200002, 200012]     # e.g. <|eot|> and <|eom|>
```

The axis is symbolic (`eos_count`) and the element list states its extent, so
adding a third end token is a one-line change with no other edit.

Three things follow from declaring it here:

* **The package states its own stop condition.** A package that ships no
  `generation_config.json` or `tokenizer_config.json` still stops correctly,
  because the workflow says how.
* **Every declared id terminates.** Not just the first. Keeping only one means
  generation runs past its end and emits control tokens as ordinary text — a
  silent failure that reads like the model rambling.
* **A request cannot disarm the others.** `GenerateOptions::eos_token_id` names
  which id a finished result *reports* and adds to the set; it does not narrow
  it. A model's end tokens are facts about the model, and emitting one as text
  is never what a caller meant.

The engine merges the package's declaration with the tokenizer's ids (package
first, neither able to drop the other) into `GenerateOptions::eos_token_ids`,
and `GenerateOptions::terminates` is the only thing that reads it — so the
single-row loop, the batched loop, the speculative verifier and a constrained
decode cannot disagree about whether a model has finished.

## What "single decoder" means to the runtime

A package is a single decoder when its workflow has **exactly one ONNX
component** (components the runtime implements — `binding` — do not count) and
that component is structurally recognizable as a decoder: it consumes the
autoregressive sequence and either produces logits or owns attention state.

This is narrower than "has a decoder". A vision-language package has a
recognizable decoder among its encoder, projector and decoder components, and
is **not** a single decoder — it is a composite package driven by the generic
interpreter. Both execute a declared `pipeline.workflow`; what differs is which
executor implements the decode step, which is a backend choice beneath one
representation.

### One classification, two layers

`onnx_genai_metadata::classify_workflow` is the only place that question is
answered, for every caller. It reads the workflow once and reports two nested
layers:

| Layer | Accessor | Question | Evidence | Asked by |
|---|---|---|---|---|
| 1 | `is_single_decoder()` | does this workflow execute exactly one ONNX graph, and is that graph a decoder? | declared port `roles` | metadata `--shape`, server, CLI |
| 2 | `contracted_single_decoder()` | …*and* does that graph name the step this runtime registered an executor for? | layer 1 **plus** `onnx-genai.autoregressive-decode` | the engine loader |

Layer 2 is *defined* as layer 1 plus the contract, so "the loader chose the
fused decode core for a package the metadata layer calls composite" is not a
state the code can reach. Requiring layer 1 is not an extra safety gate: the
fused executor is driven by the resolved `DecoderAbi`, which is derived from
the declared roles and from nothing else, so a component naming the decode step
without them has no ABI to drive it.

`is_single_decoder_workflow` and `sole_decoder_component` remain as named views
onto the same classification. `decoder_recognizer_agreement.rs` pins all of it:
an exhaustive matrix over every fixture and catalogue example, a coverage guard
so a new fixture cannot skip the matrix, and the adversarial shapes no fixture
has — extra policy bindings, two decoders, a contract with no roles, a
composite whose text head is decoder-shaped, and a 187-component package.

## Converting a package

```bash
# Rewrite a retired `model.io` block as the canonical workflow, in place.
cargo run -p onnx-genai-engine --bin migrate_model_io -- <package-dir>

# Report what would change, write nothing.
cargo run -p onnx-genai-engine --bin migrate_model_io -- --check <package-dir>

# A package whose ports were previously guessed from the graph: state them once.
cargo run -p onnx-genai-engine --bin migrate_model_io -- --abi ports.yaml <package-dir>
```

The tool reads the package's ONNX graph for its real port dtypes and ranks
rather than assuming them — a state tensor's rank differs by cache discipline,
and a contract that disagrees with the graph is rejected at load. It validates
the result before writing, so it never produces a package the runtime would
refuse.

It is deliberately **offline**. A runtime that repaired packages in memory would
mean the package on disk said one thing and the runtime executed another, which
is the second authoritative answer this design exists to prevent.

## Validating a package

```bash
# Full check: the document is valid AND every artifact it names is present.
cargo run -p onnx-genai-metadata --bin validate_metadata -- <package-dir>

# Document only — for checking metadata without downloading weights.
cargo run -p onnx-genai-metadata --bin validate_metadata -- \
    --metadata-only --shape <metadata.yaml>
```

`--metadata-only` is what a publisher needs: a hub package can be hundreds of
gigabytes, and requiring its weights before its metadata could be checked would
mean nobody checks metadata before uploading it. `--shape` reports both layers
of the shared classification — `single-decoder workflow`,
`single-decoder workflow, no decode contract`,
`composite workflow (N ONNX components)`, or `no workflow` — which is the
triage question when migrating a fleet.

## Republishing a hub package

1. `validate_metadata --metadata-only --shape` the *current* published
   `inference_metadata.yaml`. It answers three of the four questions at once:
   does it still parse, does it still validate, and what shape is it.
   - `no workflow` or a `model.io` rejection → needs conversion.
   - `single-decoder workflow` / `composite workflow (N ONNX components)` →
     already canonical; confirm the shape is the one you expect.
2. If it needs conversion, run `migrate_model_io` against a local checkout of
   the package. The tool needs the ONNX artifact present to read port contracts,
   but not the weights of every variant — a single decoder's graph is enough.
3. Re-run `validate_metadata --metadata-only` on the result, then upload the
   metadata file as a new revision. Nothing else in the package changes:
   conversion rewrites how the graph ABI is *stated*, not the graph.
4. Pin the old revision in your release notes. The retired form does not load,
   so consumers on the old revision need a reason to move.

## Reference material

- `docs/genai/INFERENCE_METADATA_DECISIONS.md` §18.1 — why the block was
  removed and what replaced each key.
- `examples/inference_metadata/catalogue/` — 20 worked examples covering text
  decoders, VLMs, encoder-decoder, diffusion, hybrid state and speculative
  decoding.
- `tests/fixtures/tiny-llm/inference_metadata.yaml` — the smallest complete
  converted single decoder.
- `tests/fixtures/tiny-llm-scatter/` — the fixed-capacity variant.
- `crates/onnx-genai-metadata/tests/decoder_workflow_roundtrip.rs` — the
  property every converted package must satisfy.

## Published-package audit

Run against every package this repository references, on 2026-08-23. Reproduce
with:

```bash
curl -sSL "https://huggingface.co/api/models/<repo>" |
  python3 -c "import json,sys; d=json.load(sys.stdin); print(d['sha']); \
print([s['rfilename'] for s in d['siblings'] if 'inference_metadata' in s['rfilename'] \
or s['rfilename'].endswith('genai_config.json')])"
curl -sS "https://huggingface.co/<repo>/raw/<rev>/inference_metadata.yaml" -o meta.yaml
cargo run -p onnx-genai-metadata --bin validate_metadata -- --metadata-only --shape meta.yaml
```

| Repo | Revision audited | Metadata | Status |
|---|---|---|---|
| `justinchuby/sensenova-u1.5-8b-mot-onnx-canonical` | `541afaea12e85222766b694cccc30153ea6dd3c1` | `inference_metadata.yaml` | **Already canonical.** Declares `pipeline.workflow`, no `model.io`. Validates against this branch. Classified `composite workflow` (186 ONNX components). **No action.** |
| `justinchuby/gemma-4-e2b-it-onnx` | `9bcf2cb1c2878b1c68a5f94db037272dfb278384` | 13 × `genai_config.json` (NF4, Q4_K_M, bf16, f16, openvino × cpu/cuda/webgpu) | **No action.** `genai_config.json` is a *foreign* producer's format, not `model.io`. The importer converts it in memory to a canonical workflow, so these load unchanged. |

**No published package requires metadata replacement.** The retired `model.io`
block was only ever emitted into in-repo fixtures, all 14 of which are converted
in this branch.

Third-party models this repository *consumes* (`Qwen/*`, `deepseek-ai/*`,
`moonshotai/*`, `zai-org/*`, `Microsoft/*` Foundry packages) are not ours to
republish. They ship `genai_config.json` or HuggingFace checkpoints, neither of
which is affected.

### What this audit caught

Auditing the sensenova package surfaced a real defect, not just a clean bill of
health. Its 187-component workflow has exactly one component carrying
`token_ids`/`logits` roles — its text head — so a predicate that asked "does
this package have a recognizable decoder?" answered *yes* and would have routed
a 186-graph package to the fused single-graph executor, which cannot run the
other 185. Every vision-language package classified the same way.

The predicate now asks the question the phrase actually means: does this
workflow execute **one** ONNX graph? That reading lives in
`classify_workflow`, which every caller — metadata validation, the CLI, the
server and the engine loader — reads instead of scanning the components itself.
It is pinned by `decoder_workflow_roundtrip::a_multi_component_package_is_not_a_single_decoder`
and, exhaustively, by `decoder_recognizer_agreement.rs`, whose
`a_187_component_package_with_one_text_head_is_composite` reconstructs this
package's shape so the defect is covered in-repo. A published package was the
only place it would otherwise have shown up — no in-repo fixture has that shape.
