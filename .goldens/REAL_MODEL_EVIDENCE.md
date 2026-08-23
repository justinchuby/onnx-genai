# Real-model evidence: baseline versus head, on pinned published packages

Every number here was produced by running the same probe against two builds of
this repository on the same machine, against the same bytes on disk. Nothing is
carried over from an earlier capture, and nothing is a projection.

## What was compared

| | revision | tree |
|---|---|---|
| baseline | `origin/main` @ `cb81745b0ba671b8cb0e35bb52c8843bdcf78384` | `8a18bcec117d52a0b694ed4a5920369078666e0a` |
| head | `feat/native-workflow-backend` @ `38fecc90bebdd4e8eec3762f64cf446bf97ee62b` | `e86e47e7adf49818a55c87b0720bcbe261813fa0` |

The baseline is a **clean worktree of `origin/main`**, built in its own target
directory. It is not this branch with commits reverted, and it is not an earlier
commit of this branch: a branch-internal baseline would compare the work against
itself and could not detect a regression the whole branch shares.

## Packages, pinned by revision

| label | repository | revision | on-disk size |
|---|---|---|---|
| `qwen3_0_6b` | `justinchuby/qwen3-0.6b-onnx-genai` | `a89c02d343fc2b3c49d61b33d61f686698be1e4d` | 512 MiB |
| `gemma4_e2b` | `justinchuby/onnx-genai-example-gemma4-e2b` | `b1189a9b16b745327a38ccf156f2ed9817440459` | 9.6 GiB |
| `gemma4_e2b_speculative` | `justinchuby/onnx-genai-example-gemma4-e2b-speculative` | `3e5d3b6aa6a5f3520ff4a3e1d9e448c01fdd0ca3` | 10 GiB |

`sha256(inference_metadata.yaml)` prefixes, so a rerun can prove it read the same
declaration: `qwen3_0_6b` `235170321029a66d`, `gemma4_e2b` `4f744916ffdc80cc`,
`gemma4_e2b_speculative` `90eb04c08f11a8dc`.

Machine: NVIDIA H200 (driver 580.105.08), ONNX Runtime 1.29.0 from
`~/.ort129/onnxruntime/capi/libonnxruntime.so.1.29.0`.

## How to reproduce

    cargo build --release -p onnx-genai-engine --example evidence_probe
    bash .goldens/capture_pinned.sh .goldens/pinned_head.tsv \
        target/release/examples/evidence_probe
    diff .goldens/pinned_baseline.tsv .goldens/pinned_head.tsv

`capture_pinned.sh` prints `COVERED=` and refuses to be silently empty: a missing
package directory writes a `MISSING_DIR` line and a failing probe writes its
error. There is no allow-empty escape hatch, because a corpus that quietly
emptied would diff clean and prove nothing — which is the failure this file
exists to make impossible.

`evidence_probe.rs` is byte-identical in both worktrees. It uses only public API
that exists on both revisions, because a probe needing a symbol one side lacks
would have had to be two different programs.

## Results

### 1. Greedy decode is byte-identical on the one package both revisions run

`qwen3-0.6b-onnx-genai`, prompt `"Hello"`, 16 tokens, greedy:

    3988,0,358,2776,773,12035,311,387,1588,0,358,2776,773,12035,311,387

Identical on baseline and head, and identical across two runs within each. This
is the load-bearing result: **the same weights, decoded through a completely
different execution path, produce the same tokens.**

The paths really are different. The baseline log reads
`decode_path: past-present kv=zero-copy-rebind` — `main` loaded this package from
its `genai_config.json` and ran it on the fused decode core, *ignoring the
`pipeline.workflow` it ships*. Head reads the workflow the package declares (11
ONNX components, an in-graph sampler and an in-graph termination predicate) and
executes it through the interpreter. Sixteen tokens agree exactly.

### 2. Two packages the baseline cannot run at all, head loads

| package | baseline | head |
|---|---|---|
| `gemma4_e2b` | `ERROR cannot resolve decoder state from tensor shapes (outputs: ["hidden_states.34"])` — fails before generating, and `create_session` fails too | `load ok`, `session.open ok`, `session.close ok` |
| `gemma4_e2b_speculative` | `ERROR declared kv_outputs port 'present.4.key' is not exposed by the graph` — **fails to load** | `load ok`, sessions open and close |

The baseline was forcing both into the decode core's shape-inferred ABI. Head
reads what the packages declare and gets past load. This is a strict improvement,
and it is also why the composite paths have no baseline to compare against:
there is nothing on `main` that runs them.

### 3. Two behaviour changes on `qwen3_0_6b`

Neither is a defect in the interpreter; both follow from head running the
package's own document where `main` ran its `genai_config.json`. They are
recorded here rather than summarised away, because they are what an operator
would notice.

**Seeded sampling differs.**

    baseline  15846,0,358,2776,264,1602,22208,1697,11,323,358,1366,311,3920,1246,311
    head       3988,0,2585,525,498,3351,1939,1986,374,279,11657,1616,311,40786,4325,323

Both are reproducible within their revision (`seeded.tokens` == `seeded.repeat`).
They differ because they are different samplers: the baseline used the runtime's
Rust sampler, head uses the `onnx-genai.token-sampler` component this package
ships, steered through its declared `sampling_*` roles. A package that declares a
sampler and then has the runtime's substituted for it is a package whose
declaration is decorative — so this is the intended direction. It is still a
difference, and a caller pinning seeded output will see it.

**Multi-turn continuation is lost for this package.**

    baseline  turn1 prefix=1   turn2 1052,... prefix=17   session.tokens=34
    head      turn1 prefix=0   turn2 3988,... prefix=0    session.tokens=32

On `main` the decode core carried KV across turns, so turn 2 continued turn 1.
On head the package is executed as declared, and its workflow declares no
`scope: session` state — so each turn is independent, and `prefix_cache_hit_len`
is honestly `0` rather than a number from a cache that did not participate.

**This is a real regression for this package** and is called out as such. The fix
is on the package side (declare session-scoped state for the conversation) or on
the runtime side (prefer the decode core when a package's workflow is
decoder-shaped despite shipping several components). Neither is done here.

### 4. What the probe could not exercise, and why

`gemma4_e2b` and `gemma4_e2b_speculative` declare `request.active`,
`request.done` and `request.accepted_len` as **required application inputs with
no default**. Those are the serving cells the runtime's token policy writes, so a
caller has to seed them; `Engine::generate` with a bare prompt cannot, and says
exactly that. Composite speculative generation at widths 1/2/4 against these two
packages therefore has **no measurement in this file**.

That is a gap, stated plainly rather than filled in with the hermetic
`gemma4_chained` fixture — a different artifact, which would not be evidence
about these packages. Closing it needs either a harness that seeds the declared
serving inputs, or the packages re-published with literal defaults for them.

Composite speculative execution *is* covered on hermetic fixtures, including on
CUDA: `native_workflow_parity::chained_speculative_proposal_parity_native_cuda`
runs propose/verify/accept/reject/rollback on an H200 and asserts non-zero
rejections and rolled-back cells, token-identical to the ORT backend and to plain
greedy decode. That is a different claim from "these two published packages
generate", and this file does not conflate them.

## Fixed while producing this

Two defects were found by running the pinned packages, and are fixed on head:

* `generate_in_session` on a package with no decode core reached
  `require_tokenizer()` and failed, because only the `_with_callbacks` wrapper
  had been routed to the interpreted session path. Every variant now routes
  through the same place.
* A text prompt against a package declaring a `prompt_tokens` input was refused
  with "use a tokenizer adapter for text" even when the package shipped
  `tokenizer.json`. The runtime now encodes with the package's own tokenizer,
  which is what made result 1 measurable at all.

## Pre-existing failure, not from this branch

`onnx-genai-ort`'s `loader::tests::selected_non_dense_candidate_fails_explicitly`
fails identically on `origin/main@cb81745b0`, verified in the clean baseline
worktree.
