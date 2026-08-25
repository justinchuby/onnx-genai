# Real-package chained speculative evidence

A hermetic fixture proves the interpreter's chained-proposal *semantics*. It
cannot prove that a published package decodes, and the gap between the two is
where every failure in this area has actually lived. This page records the
run that closes it, and how to reproduce it.

The gate is
`crates/onnx-genai-engine/tests/chained_speculative_real_evidence.rs`, driven by
the package-agnostic harness in `tests/common/real_workflow.rs`. Nothing in
either knows a model: geometry is read from the package's declared contracts and
its graphs' own static dimensions, never written down.

## The claim

A composite package whose `speculative.proposal_execution` is `chained`
decodes **exactly** what its standalone target decodes, at every proposal width,
and gets there by proposing, accepting and rejecting.

That is checked, not asserted:

| Requirement | How the test fails without it |
| --- | --- |
| Same target model as the standalone package | compares the full token stream at each width |
| Proposals actually made | `proposed > 0` and `proposer_invocations > 0` per width |
| The chain is worth running | the target confirms a first draft in at least one round in four, per width |
| Non-zero acceptance | `accepted_drafts > 0` per width |
| Rejection and rollback exercised | `rejections > 0` and `rolled_back_cells > 0` across widths |
| A block fully accepted at least once | `full_accepts > 0` across widths |
| Real embedding gather, not a fixture bypass | resolves the declared table in the real artifact, requires a non-zero first row of the declared width |
| Real folded carry seed | requires the declared seed value to exist in the verification pass with a per-position shape over the context |
| Table read once per residency | `embedding_table_loads <= 2` |
| Device residency (#1861) preserved | `host_staging == 0`, readback `<=` one token id per proposer invocation |
| No vacuous run | a missing package, a non-directory path, or fewer widths covered than declared all fail under `ONNX_GENAI_REQUIRE_SPECULATIVE_EVIDENCE=1` |

### What the token stream does and does not prove

A verified block commits the *target's* tokens whatever the proposer said, so an
identical stream is evidence that the composite package's target and the
standalone target package are the same model. That is necessary and it is not
sufficient: a proposal chain conditioned on entirely the wrong tensors emits the
same stream, more slowly. The claim that the chain *works* rests on the
acceptance statistics, which a broken chain cannot fake — before the borrowed-
cache and embedding-normalizer fixes below the target confirmed a first draft in
**0 of 12** rounds; after them, in **4 of 10**, at every width.

## Running it

```bash
export ORT=<onnxruntime-linux-x64-gpu_cuda13-1.29.0>
export ONNX_GENAI_ORT_LIB_DIR=$ORT/lib
export LD_LIBRARY_PATH=$ORT/lib:<dir with libcudnn.so and libcublas*.so>:$LD_LIBRARY_PATH
export ONNX_GENAI_EP=cuda
export ONNX_GENAI_EP_FALLBACK=1     # the exports keep a few shape ops on CPU

export ONNX_GENAI_REQUIRE_SPECULATIVE_EVIDENCE=1
export ONNX_GENAI_CHAINED_WORKFLOW_PACKAGE=<composite package dir>
export ONNX_GENAI_SPECULATIVE_TARGET_PACKAGE=<target-only package dir>

cargo test -p onnx-genai-engine --features native-cuda \
  --test chained_speculative_real_evidence -- --nocapture
```

Without `ONNX_GENAI_REQUIRE_SPECULATIVE_EVIDENCE` the case skips when a package
is unset, so a developer machine without 20 GB of weights still runs the suite.
With it, skipping is a failure.

The packages are ordinary Hugging Face snapshots:

```bash
hf download justinchuby/onnx-genai-example-gemma4-e2b-speculative \
  --revision 77a8161bc2a2c9de478dae50307f60e2a0c6beff --local-dir <composite>
hf download justinchuby/onnx-genai-example-gemma4-e2b \
  --revision 79ca25afe326719e4daab79430c90195dfd28f3b --local-dir <target>
```

## The recorded run

| | |
| --- | --- |
| Composite package | `justinchuby/onnx-genai-example-gemma4-e2b-speculative` @ `77a8161bc2a2c9de478dae50307f60e2a0c6beff` |
| Target package | `justinchuby/onnx-genai-example-gemma4-e2b` @ `79ca25afe326719e4daab79430c90195dfd28f3b` |
| ONNX Runtime | 1.29.0, `gpu_cuda13` build, `CUDAExecutionProvider` |
| Hardware | NVIDIA H200 (143 GB), driver 580.105.08 |
| Prompt | `"Once upon a time, in a small village near the mountains, there lived"`, plus the package's declared `<bos>` |
| Sampling | greedy (`temperature = 0`, `stop_on_eos = false`), 24 tokens |
| Contract | proposer `assistant`, target `target`, `max_proposal_width` 6, 30 rollback cells |
| Embedding | `target::model.embed_tokens.weight`, `[262144, 1536]`, `Float16` |
| Folded carry seed | `target::hidden_states.34`, `[1, 16, 1536]`, `Float16` |
| Symbols resolved from the graphs | `full_head_dim=512`, `full_kv_heads=1`, `fused_hidden=3072`, `sliding_head_dim=256`, `sliding_kv_heads=1` |

Every width decoded the identical 24 tokens, and they are the target package's
own greedy stream:

```
[496, 3184, 3953, 7489, 2876, 2032, 236761, 2876, 2032, 1053, 6114, 506,
 2258, 529, 506, 5312, 7217, 532, 496, 7304, 618, 7804, 618, 496]
```

| Drafts / proposal | Chain width | Rounds | Rounds w/ an accepted draft | Proposed | Accepted | Rejected | Rejections | Full accepts | Proposer invocations | Rolled-back cells |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 2 | 10 | 4 | 10 | 4 | 6 | 6 | 4 | 20 | 180 |
| 2 | 3 | 10 | 4 | 20 | 4 | 16 | 10 | 0 | 30 | 300 |
| 4 | 5 | 10 | 4 | 40 | 4 | 36 | 10 | 0 | 50 | 300 |
| **total** | | **30** | **12** | **70** | **12** | **58** | **26** | **4** | **100** | **780** |

Residency: `host_staging = 0` across the whole matrix — the device-residency
property #1861 established holds under a real package. `readback_bytes = 0`
because under the ORT CUDA execution provider this package's values are
published host-resident (the run reports `host_resident=true` for the folded
carry seed), so there is no device read to make; the assertion is an upper
bound of one token id per proposer invocation, which covers both regimes.

A chain's first step is a bootstrap that reproduces the token the target already
handed over, so *d* drafted tokens cost *d + 1* proposer invocations. That is
why the table carries both columns.

## What had to be fixed to get here

Five defects, each of which a hermetic fixture passes and a real package fails.
Four are runtime or metadata-schema changes in this repository; the fifth is
package-only.

**1. The serving contract's own controls were demanded of the caller.**
`serving.{active, done, accepted_len}` names values the runtime's token policy
writes. The published packages declare them as required application inputs with
no default, so admission refused every request — asking an application whether
the row it just submitted is already `done`. Package admission now seeds a
workflow input that a serving control resolves to from that control's role
(`active = true`, `done = false`, `accepted_len = 0`), exactly as it already
does for a runtime-managed state seed. A caller-supplied value still wins, a
package-declared default still wins over the derived one, and an input the
serving contract does not name is untouched.

**2. The chained proposal path was float32-only.**
`token_embedding.table` was required to be float32 and the fused proposer input
was allocated float32. A real fp16 export has an fp16 table and an fp16 fused
input, so the chain refused to start. The chain now takes its arithmetic
currency from the element type of the value the workflow bound to that declared
port, and requires the declared table to agree, naming both sides when they do
not.

**3. A borrowed cache was read from the seed, not from its owner.**
`serving.state_service.groups[*].ports.<proposer>` declares `access: read_only`
aliases: the port *is* the cell another component owns. The chain bound those
ports from the value the pass started with — for a pass over a fresh context, an
empty cache — so the drafter conditioned on nothing. The fixture's target writes
zeros into its cache, so the fixture cannot tell the difference. The chain now
binds a read-only alias from the value the owning component's declared `output`
port produced.

**4. The declared embedding was the unscaled table.**
The target's graph multiplies a gathered row by `sqrt(hidden_size)` before its
backbone reads it, so the initializer's raw rows are not the tensor a proposer's
fused input takes. `token_embedding` gained an optional `scale` for exactly this:
nothing in a `[vocab, hidden]` initializer says whether a normalizer was applied,
and guessing one would be a per-architecture heuristic in the runtime. The
factor is applied once to the table, never per gathered row.

The fourth also needed the packages republished, along with the fifth:

**5. Every sequence-like axis was called `sequence`.**
The graphs distinguish the query length, the cache a pass starts from, the cache
it ends with, and (for the drafter) the borrowed caches it reads. The metadata
collapsed all of them, which made any pass whose cache is not exactly as long as
its prompt unbindable —

```
workflow value 'request.past_key_values.0.key' axis 2 binds symbol
'sequence' to 0, but it was already 6
```

— and narrowed the drafter's borrowed cache to a single position, because a
chained proposal narrows every proposer port that shares the fused input's
position symbol. `scripts/name_sequence_axes.py` performs that rename on a
published `inference_metadata.yaml` without disturbing its authored comments.
