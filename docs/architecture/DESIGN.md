# ONNX GenAI design

## Principles

- Metadata is a universal declarative IR; model math lives in ONNX.
- Runtime behavior is selected by typed features and capabilities, never model names.
- Invalid or unsupported packages fail at load and document input cannot panic.
- Side effects, state, RNG, placement, and package boundaries are explicit.

## Execution architecture

Every package contains one `pipeline.workflow`, whose manifest, typed package
boundary, component registry, state declarations, serving contract, and recursive
control-flow graph are the only orchestration source of truth — and, through
component ports and state groups, the only serialized graph ABI. A bare
single-graph decoder is not an exception: the optimized single-model decode path
recognizes that same workflow, which is the only place a graph ABI is stated,
which survives as an import-only legacy input.

The workflow interpreter executes SSA nodes (`invoke`, `sequence`, `loop`, `branch`,
`transfer`, `emit`). ONNX components implement sampling, predicates, solvers,
speculative acceptance, codecs, and state transforms. Versioned adapters cover only
stable tokenizer/media/request-binding ABIs, preferably implemented as ONNX custom
ops.

## Types and validation

Tensor contracts declare dtype, rank, and symbolic shape. ONNX shape inference owns
intra-component typing; metadata unifies cross-component symbols and checks bounded
state growth. Branch phi contracts must unify. Loop induction values are lexical.
Linear effect tokens order state and output effects.

The manifest pins IR, opset, adapter ABI, and capability versions. Required
capabilities are deterministically derived and checked before sessions are created.
Policy graphs use only standard ONNX operators. Wide last-axis `ArgMax`/`ArgMin`
nodes run directly when the loaded CUDA runtime includes the upstream parallel
reduction capability; older or unknown runtimes receive an equivalent tiled
standard-ONNX lowering. Native execution-provider sampling remains independent.

## Runtime services

Continuous batching, admission, compaction, KV paging/slot allocation, memory
resources, device placement, and concurrency are generic services. Workflow values
declare active/done/accepted-length/slot data. Transfers are explicit and host code
does not implement hidden reducers, scatter, append, scheduler, or sampler math.

## Representative lowerings

Autoregressive decoding, VLM, diffusion, masked diffusion, speculative decoding,
nested speech/codec generation, and world-model rollout all lower to the same nodes,
state cells, effects, and ONNX policy components. Adding a workload requires metadata
and artifacts; a runtime code change is justified only by a new generic capability.

See [MODEL_METADATA.md](../genai/MODEL_METADATA.md) and
[WORKFLOW_POLICY_COMPONENTS.md](../WORKFLOW_POLICY_COMPONENTS.md).
