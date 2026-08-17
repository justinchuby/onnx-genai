# An EP must be fast on its own

**Principle:** an execution provider's performance must not depend on the host
having set up something the EP does not own. If lifting `onnx-runtime-ep-cpu` out
of this repo and driving it from a third-party runtime produces the slow path,
the EP is not substitutable, whatever its API surface says.

This is a contract requirement, not a tuning preference. The stated reason for
having two backends is that they are interchangeable and that each component can
be lifted out, used, or replaced independently. A component that is only fast
inside one embedder has not been lifted out; it has been *copied* out.

## The failure we found (#1138)

`onnx-runtime-ep-cpu` selects its fast dispatch from `IN_SPMD_SCOPE`, a
**thread-local the caller sets**. `parallel_output_rows`
(`kernels/matmul_nbits.rs`) dispatches onto resident SPMD workers when that flag
is set, and otherwise does a fresh Rayon fork-join on every call.

Only `onnx-genai-engine` sets it, and only for single-token forwards
(`native_decode/cpu.rs`, gate `token_ids.len() == 1`). So:

| host | enters the scope? | consequence |
|---|---|---|
| engine, decode | yes | fast path |
| engine, prefill | **no** | fork-join per call; CPU time grows 88% from 4 to 20 threads on a 14B, wasting ~61% of CPU at 20 |
| ORT plugin | **cannot** — ORT owns the graph, schedule and threads | inherits only the pool's costs; 0.376 ms with the pool vs 0.092 ms without, against ORT's own 0.097 ms |
| third-party runtime | **no**, and no way to know | slow path, silently |

The plugin became competitive only by calling `disable_persistent_decode_pool()`
— switching *off* the mechanism the engine switches *on*. Two hosts, opposite
configurations, one EP.

Worse than the speed: the pool imposed a **data-layout** decision on hosts that
could not use it. `MatMulNBits` weights were pre-partitioned into one MLAS shard
per persistent decode worker, which then capped an unscoped GEMV at that worker
count no matter how many threads the host had. A choice made for one execution
model silently penalised the other.

## The pattern that is correct, already in the tree

The CUDA EP does not read a host thread-local to find its memory authority. The
host **passes one in**, explicitly, as part of the load contract:

```rust
NativeDecodeLoadOptions {
    cuda_memory_governor: Arc<dyn onnx_runtime_memory_governor::MemoryGovernor + Send + Sync>,
    ...
}
```

That is the shape to copy. A host that has a better executor, allocator or
governor can supply it through a documented parameter; a host that has none gets
a working default the EP owns. Neither host has to know a private flag exists,
and a third-party EP can implement the same trait to be substitutable.

## Rules

1. **The EP owns its own decomposition and its own defaults.** It must reach a
   good parallel schedule with no host cooperation at all.
2. **Optional host capabilities are explicit parameters, not ambient state.**
   "Here is my thread pool" is an API. A thread-local read behind the caller's
   back is not, because it cannot be discovered, documented, or implemented by a
   third party.
3. **A facility one host cannot use must not cost the hosts that cannot use it.**
   That includes indirect costs like data layouts chosen for a specific worker
   count.
4. **Measure both paths on the same kernel in CI.** The plugin's opt-out exists
   because a change that helped the engine and hurt the plugin landed once with
   nothing to catch it.

## How to check

The question to ask of any EP change is not "is it faster in our engine" but
**"would a third party linking this crate and calling it from their own runtime
get this speed without being told anything?"** If the answer needs a caveat, the
speed lives in the host, not in the EP.
