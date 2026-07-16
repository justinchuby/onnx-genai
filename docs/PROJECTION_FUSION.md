# Native Decode Projection Fusion Design
**Status:** design for review; no fusion is implemented by this document<br>
**Scope:** native CPU decode, `com.microsoft::MatMulNBits`, block-quantized int4 projections<br>
**Primary model:** Qwen2.5-0.5B int4<br>
**Date:** 2026-07-16
## 1. Executive summary
Native Qwen2.5-0.5B decode is now dominated by `MatMulNBits` even after the symmetric block-32 direct-int4 VNNI GEMV landed. The paired profile reports 14.154 ms per step and **82.53%** of node time in `MatMulNBits` ([decode profile, lines 187-209](benchmarks/2026-07-16-decode-profile2.md#L187-L209)). The kernel already partitions output columns (`N`) with a thread-count-aware Rayon policy, so combining projections that read the same activation gives it a larger scheduling domain and removes repeated activation preparation, kernel dispatch, and Rayon synchronization.

The recommended architecture is a **conservative load-time graph rewrite** in `Executor::build`, next to the existing SiLU rewrite and before topological planning. It will:
1. find compatible sibling `MatMulNBits` projections with the same activation;
2. concatenate their immutable packed weights, scales, optional zero points, and optional biases in output-column order;
3. replace the siblings with one larger-`N` `MatMulNBits`;
4. either feed an eligible packed-QKV `GroupQueryAttention` directly, or split the fused output into the original SSA values; and
5. preserve the original projection order and all downstream semantics.
The rewrite should not make the executor gather siblings dynamically at every run. The graph and weights are immutable, the compatibility proof is structural, and concatenation is session-start work. Runtime gathering would repeat graph inspection, complicate buffer lifetime and profiling, and leave the graph plan claiming that nodes execute independently when they do not.

For the general split form, the first implementation should add a zero-copy `Split::view_outputs` path for last-axis slices. At decode shape `M=1`, each Q/K/V or gate/up slice is a contiguous region of the fused result. The executor already supports kernel-declared output views and skips allocation and compute for them ([EP API, lines 95-149](../crates/onnx-runtime-ep-api/src/kernel.rs#L95-L149), [executor, lines 1413-1477](../crates/onnx-runtime-session/src/executor.rs#L1413-L1477)). Using the current copying `Split` unchanged would add avoidable materialization: it densifies the full input, allocates one `Vec` per output, and copies each slice ([split kernel, lines 169-188](../crates/onnx-runtime-ep-cpu/src/kernels/split.rs#L169-L188)).
## 2. Measured opportunity and inspected graph reality
### 2.1 Current bottleneck
The direct-int4 profile used 24 Rayon workers on a dual-socket Xeon 8480C and reproduced the same four greedy tokens in every run. After direct-int4 VNNI, `MatMulNBits` still accounts for 14.154 ms/step, or 82.53% of node time ([decode profile, lines 193-207](benchmarks/2026-07-16-decode-profile2.md#L193-L207)). The earlier phase breakdown also found repeated Rayon barriers and 121 separate projection calls among the remaining scaling costs ([decode profile, lines 136-159](benchmarks/2026-07-16-decode-profile2.md#L136-L159)).

The direct-int4 kernel does the following once per `MatMulNBits` invocation:
- densifies the f32 activation;
- allocates a result `Vec`;
- quantizes the activation row to signed int8;
- partitions `N` into Rayon chunks; and
- writes the result back to the executor output.
Those boundaries are visible in the kernel:
- activation/result preparation: [matmul_nbits.rs, lines 149-156](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L149-L156);
- direct-int4 eligibility: [lines 157-192](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L157-L192);
- activation quantization and `N`-parallel execution: [lines 430-475](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L430-L475).
When Q, K, and V are separate, all three repeat this setup over the same normalized hidden state. Gate and up likewise repeat it over the same post-attention normalized hidden state. The weight bytes still have to be read, but activation conversion, scheduling, dispatch, and synchronization can be shared.
### 2.2 Important fact-check: the local benchmark artifact already packs QKV
Inspection of `/home/justinchu/qwen2.5-0.5b-int4-onnx/model.onnx` on 2026-07-16 found:
- opsets: `ai.onnx` 21 and `com.microsoft` 1;
- 299 nodes and 318 initializers;
- 121 `MatMulNBits` nodes and 24 `GroupQueryAttention` nodes;
- one already-fused QKV projection per layer, named like `/model/layers.0/attn/qkv_proj/MatMul_Q4`, with `K=896`, `N=1152`, `block_size=32`, and `accuracy_level=4`;
- separate gate and up projections per layer, each with `K=896`, `N=4864`; and
- no `Split` node in the artifact.
The packed-QKV fact is consistent with the project progress record, which says native genai-builder compatibility includes packed-QKV GQA exports ([PROGRESS.md, lines 294-303](PROGRESS.md#L294-L303)). The CPU GQA kernel also explicitly supports both unpacked Q/K/V and packed QKV ([group_query_attention.rs, lines 1-6](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L1-L6)).

Therefore:
- **gate/up fusion is directly applicable to the measured artifact;**
- **QKV fusion applies to decoder exports that still contain three sibling projections;** and
- the pass must be idempotent and leave the already-packed QKV node untouched.
This distinction matters for expected speedup. The 82.53% profile is real, but its 121 calls already include only five layer-local projections per layer: packed QKV, attention output, gate, up, and down, plus the LM head. It is not a measurement of a 169-call separate-QKV graph.
## 3. Existing implementation constraints
### 3.1 `MatMulNBits` storage contract
The CPU kernel expects standard ORT layout:
```text
B      : uint8  [N, ceil(K / block_size), block_size / 2]
scales : float  [N, ceil(K / block_size)] or flat equivalent
zp     : uint8  [N, ceil(ceil(K / block_size) / 2)] or flat equivalent
bias   : float  [N]
```
The source documents `B` as `[N, K_blocks, block_size/2]`, with the earlier K element in the low nibble ([matmul_nbits.rs, lines 1-10](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L1-L10)). Runtime shape checks enforce the weight, scale, zero-point, and bias contracts ([lines 117-147](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L117-L147)).

The direct-int4 path caches packed bytes and scales without changing their layout ([lines 164-184](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L164-L184)). For output column `n`, it reads one contiguous packed row and one contiguous scale row ([lines 441-460](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L441-L460)). This output-major layout is what makes concatenation along `N` byte-exact.
### 3.2 Symmetric zero point
When the optional zero-point tensor is absent, the generic prepack and dequantization paths use zero point 8 ([matmul_nbits.rs, lines 290-316](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L290-L316), [lines 350-380](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L350-L380)). The direct-int4 VNNI routines also subtract 8 from each unpacked nibble ([lines 527-558](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L527-L558), [lines 561-591](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L561-L591)).

The measured Qwen graph omits zero-point inputs, so the fused node must also omit them. It must not synthesize a zp initializer merely to make concatenation uniform; doing so would disable the current direct-int4 eligibility check, which requires `zero_points.is_none()` ([lines 157-163](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L157-L163)).
### 3.3 Constant prepacking and initializer lifetime
A `MatMulNBits` kernel prepackages only when B/scales and any optional zp/g_idx inputs are graph constants ([matmul_nbits.rs, lines 149-153](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L149-L153)). The executor derives this flag from `Graph::initializers` when compiling each node ([executor.rs, lines 970-1003](../crates/onnx-runtime-session/src/executor.rs#L970-L1003)).

Large model weights are normally `WeightRef::External`; `WeightStore::bytes` returns a slice over the live mmap ([weights.rs, lines 48-67](../crates/onnx-runtime-loader/src/weights.rs#L48-L67)). A newly concatenated initializer can be represented as `WeightRef::Inline(TensorData)`; `TensorData::from_raw` owns raw little-endian bytes and explicit dimensions ([tensor.rs, lines 8-47](../crates/onnx-runtime-ir/src/tensor.rs#L8-L47), [lines 72-106](../crates/onnx-runtime-ir/src/tensor.rs#L72-L106)).

This is simple and safe, but it creates one additional owned copy of each fused weight group. The rewrite must then remove the old initializers when they are orphaned, or the graph will retain both the old references and new owned copies. The underlying whole-file mmap remains live because the executor retains the `WeightStore`; virtual address space does not shrink, but old projection pages need not be touched during execution.

A later memory optimization may create a composite/segmented `WeightRef`, but that would not provide one contiguous packed buffer to the current kernel and is not required for the first implementation.
## 4. Recommended fusion location
### 4.1 Exact seam
Add an internal pass with a shape such as:
```rust
fn fuse_matmul_nbits_projections(
    graph: &mut Graph,
    weights: &WeightStore,
) -> Result<ProjectionFusionStats>;
```
Call it in `Executor::build` immediately after `fuse_silu_patterns` and before `graph.topological_order()`:
```text
Executor::build
  -> fuse_silu_patterns(&mut graph)
  -> fuse_matmul_nbits_projections(&mut graph, &weights)
  -> graph.validate()                 // recommended new post-pass assertion
  -> graph.topological_order()
  -> initializer buffer population
  -> structural NodePlan construction
  -> shape-keyed kernel compilation
```
The exact existing seam is [executor.rs, lines 666-675](../crates/onnx-runtime-session/src/executor.rs#L666-L675). It is the right location because:
1. both the mutable `Graph` and live `WeightStore` are available;
2. no initializer device buffers have been allocated yet;
3. no topological order or `NodePlan` has been frozen;
4. no per-node kernel cache entries exist;
5. the graph retained for EPContext export is explicitly the post-optimize graph ([executor.rs, lines 1015-1025](../crates/onnx-runtime-session/src/executor.rs#L1015-L1025)); and
6. there is already a tested load-time rewrite precedent at this exact point.
The graph IR is deliberately mutable during optimization. Its mutation API maintains producer/consumer edges and supports `insert_node`, `remove_node`, `replace_node`, and use replacement ([graph.rs, lines 14-18](../crates/onnx-runtime-ir/src/graph.rs#L14-L18), [lines 281-378](../crates/onnx-runtime-ir/src/graph.rs#L281-L378)).
### 4.2 Why not put the pass in the loader?
The loader constructs IR before attaching top-level weight sources, then loads and validates initializers in later stages ([graph_builder.rs, lines 28-42](../crates/onnx-runtime-loader/src/graph_builder.rs#L28-L42)). A weight-concatenating pass needs resolved bytes from inline or external initializers. Placing it in graph construction would couple protobuf decoding to CPU-specific optimization and run before external bytes are conveniently available.

A future generic optimizer pipeline could move the pass out of `executor.rs`, but the first implementation should follow the existing SiLU precedent rather than inventing a new pass manager.
### 4.3 Why not runtime sibling gathering?
A runtime gather would leave three or two nodes in the plan, then make the executor discover that they share input 0 and invoke a hidden group operation. That has several problems:
- sibling compatibility is invariant and should not be reproved per token;
- kernel-cache identity is currently one node plus concrete input shapes ([executor.rs, lines 112-205](../crates/onnx-runtime-session/src/executor.rs#L112-L205));
- output allocation and liveness are currently per `NodePlan`;
- per-op profiling would double-count or require special grouped accounting;
- error handling becomes ambiguous if only part of a sibling group compiles;
- it does not naturally create one immutable prepacked weight object; and
- it pushes model-specific graph semantics into the sequential dispatch loop.
Runtime gathering is therefore rejected as the primary design.
## 5. Conservative pattern matcher
### 5.1 Base sibling compatibility
A candidate group must satisfy all of the following:
1. each node is `com.microsoft::MatMulNBits`;
2. each has exactly one output;
3. input 0 is the same `ValueId`;
4. B and scales are present initializers;
5. `K`, `bits`, `block_size`, `accuracy_level`, and `weight_prepacked` match;
6. `bits == 4` and `weight_prepacked` is absent or zero;
7. B and scale dtypes and dimensions pass the current kernel contract;
8. optional zero-point presence matches across all siblings;
9. optional `g_idx` is absent, or all siblings reference the same initializer with byte-identical contents;
10. optional bias presence matches and every bias is constant;
11. output dtype and all non-last output dimensions match;
12. none of the old projection outputs is a graph output; and
13. the group matches an approved semantic pattern, not merely “same input.”
Matching `accuracy_level` is mandatory. A fused node cannot choose one sibling's numerical mode for another sibling. The factory reads it once into the kernel ([matmul_nbits.rs, lines 70-84](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L70-L84)).
### 5.2 QKV semantic pattern
Do not fuse arbitrary triples. Accept QKV only when one of these is proven:
- **Unpacked GQA pattern:** the three outputs feed query, key, and value slots 0, 1, and 2 of the same `GroupQueryAttention`, possibly through only approved shape-preserving bias/Add or view nodes; or
- **Explicit attention pattern:** names and downstream structure prove the three branches are Q, K, and V and preserve the order `Q | K | V` through RoPE and attention.
The first implementation should support the direct unpacked-GQA case and a single constant bias `Add` on each branch. It should reject unfamiliar transpose/reshape/RoPE decompositions until each is covered by tests.

When the common consumer is this repository's GQA and the three projections have no other consumers, prefer the **packed-consumer lowering**:
```text
Before:
  x -> MM(q) -> [optional Add] -> GQA input 0
  x -> MM(k) -> [optional Add] -> GQA input 1
  x -> MM(v) -> [optional Add] -> GQA input 2

After:
  x -> MM(q|k|v) -> [optional Add(q|k|v)] -> GQA input 0
                                             GQA inputs 1,2 absent
```
GQA defines packed width `(num_heads + 2*kv_num_heads) * head_dim` and slices it in Q, K, V order ([group_query_attention.rs, lines 173-217](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L173-L217)). Its execute path selects packed form exactly when key and value inputs are absent ([group_query_attention.rs, lines 318-354](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L318-L354)). This lowering removes the split completely and matches the already-packed local artifact.

When downstream consumers require separate values, use the general split form:
```text
x -> MM(q|k|v) -> Split(axis=-1, sizes=[Nq,Nk,Nv]) -> original q,k,v ValueIds
```
The original output `ValueId`s should become outputs of the inserted `Split`, so RoPE, GQA, and any other existing consumers remain unchanged.
### 5.3 Gate/up semantic pattern
Accept a pair only when downstream structure proves SwiGLU gate/up roles:
```text
x -> MM(gate) -> Silu -> Mul input 0
x -> MM(up)   ------------> Mul input 1
```
The repository already rewrites exact `x * Sigmoid(x)` to `Silu`, so the matcher should recognize either the pre-rewrite decomposition or the post-rewrite `Silu` form. Because projection fusion runs after `fuse_silu_patterns`, the first implementation can match the post-rewrite form ([executor.rs, lines 602-664](../crates/onnx-runtime-session/src/executor.rs#L602-L664)).

Lower to:
```text
x -> MM(gate|up) -> Split(axis=-1, sizes=[Ngate,Nup]) -> original gate,up ValueIds
```
Order is semantically significant. Do not sort siblings by `NodeId` or name and assume the result. Derive segment order from the recognized consumer roles.
## 6. Weight and metadata concatenation
### 6.1 Shape math
For siblings `i = 0..P-1`:
```text
K                 = common K
B                  = common block_size
KB                = ceil(K / B)
packed_block_bytes = B / 2
Ni                 = sibling output width
Ntotal             = sum(Ni)

Bi shape           = [Ni, KB, packed_block_bytes]
Si shape           = [Ni, KB]
Bfused shape       = [Ntotal, KB, packed_block_bytes]
Sfused shape       = [Ntotal, KB]
Yfused shape       = A.shape[..rank-1] + [Ntotal]
```
The fused node copies all original attributes except `N`, which becomes `Ntotal`. `K`, `bits`, `block_size`, `accuracy_level`, and `weight_prepacked` remain unchanged.
### 6.2 Packed int4 bytes
For each sibling in semantic output order:
```text
expected_B_bytes_i = Ni * KB * (B / 2)
Bfused_bytes.extend_from_slice(Bi_bytes)
```
No nibble repacking is needed.

Why this is exact:
- the innermost nibbles encode adjacent K values within one block;
- a complete output-column row contains `KB * B/2` bytes;
- the next output column starts on a byte boundary;
- concatenation is along the outermost `N` dimension; and
- the kernel computes a row address as `output_index * KB * packed_block_bytes`.
Therefore the first `N0` rows in the fused buffer are byte-for-byte projection 0, the next `N1` rows are projection 1, and so on. Different Q/K/V widths do not change K-block packing.

The pass must check every byte count with checked multiplication and reject a candidate rather than truncate or panic.
### 6.3 Scale bytes
Scales are f32 and output-major:
```text
expected_scale_count_i = Ni * KB
expected_scale_bytes_i = Ni * KB * 4
Sfused_bytes            = S0_bytes || S1_bytes || ...
```
Again, no conversion is needed on little-endian supported hosts. The kernel selects scale row `output_index * KB .. (output_index + 1) * KB` ([matmul_nbits.rs, lines 452-460](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L452-L460)).

The rewrite should preserve the canonical matrix shape `[Ntotal, KB]`, even though the kernel accepts a flat equivalent, because it is clearer for graph inspection and EPContext export.
### 6.4 Zero points
There are two supported cases:
1. **All siblings omit zp.** The fused node omits zp and preserves implicit symmetric `zp=8`. This is the Qwen fast path.
2. **All siblings provide constant zp with compatible shapes.** Concatenate complete zp rows along `N`:
```text
zp_row_bytes = ceil(KB / 2)
Zfused       = Z0_rows || Z1_rows || ...
shape        = [Ntotal, zp_row_bytes]
```
The two nibbles in a zp byte encode adjacent K-block zero points for one output column. Because concatenation occurs after complete output rows, it never joins a nibble from one projection with a nibble from another.

However, explicit zp currently prevents the direct-int4 path. It falls through to the int8 accuracy-4 path. The pass may still fuse it for correctness, but performance expectations must be measured separately.

Mixed absent/present zp groups are rejected, even if the explicit tensor happens to contain only 8s. Canonicalizing that case requires a separate proof and would alter direct-int4 eligibility.
### 6.5 `g_idx`
`g_idx` maps K positions to scale groups and is not indexed by output column ([matmul_nbits.rs, lines 129-139](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L129-L139)). A fused node has only one `g_idx` input. Fusion is legal only if all siblings omit it or share byte-identical constant indices. The first implementation should conservatively require all siblings to omit it, matching the direct-int4 Qwen path.
### 6.6 Bias
`MatMulNBits` supports optional bias at input 5 and requires shape `[N]` ([matmul_nbits.rs, lines 141-147](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L141-L147)). If biases are fused inside the op, concatenate their f32 arrays along `N`.

The inspected Qwen artifact instead uses an `Add` after packed QKV and has no MatMulNBits bias input. For three separate `Add` nodes with constant bias, the packed-consumer form may concatenate biases and replace them with one `Add` over `Ntotal`. Gate/up in the inspected artifact have no biases.

Mixed bias/no-bias groups should initially be rejected rather than synthesize zero bias. Zero synthesis is correct but adds policy and memory for little gain.
### 6.7 Initializer ownership and naming
Create deterministic names such as:
```text
__nxrt_fused_projection_<first-node-id>_weight
__nxrt_fused_projection_<first-node-id>_scales
__nxrt_fused_projection_<first-node-id>_zero_points
__nxrt_fused_projection_<first-node-id>_bias
```
Create new graph values with exact dtype/shape, attach `WeightRef::Inline(TensorData::from_raw(...))`, and wire them to the fused node. After replacing the old nodes, remove old initializers only if their values have no remaining consumers and are not graph inputs/outputs. Do not assume every initializer is projection-private.

Call `graph.validate()` after the pass. Graph validation checks live edges, producer/consumer consistency, duplicate outputs, graph sources, cycles, and subgraphs ([graph.rs, lines 380-475](../crates/onnx-runtime-ir/src/graph.rs#L380-L475)).
## 7. Output splitting and downstream contracts
### 7.1 General split representation
Use an internal `ai.onnx::Split` node with:
```text
axis = -1
split sizes = [N0, N1, ...]
outputs = the original sibling MatMulNBits output ValueIds
```
Opset 21 represents split sizes as an optional int64 input, but the in-tree kernel and shape inference also accept a `split` attribute. For an internal post-load rewrite, using the attribute avoids another initializer and is already supported by:
- the CPU Split factory and kernel ([split.rs, lines 18-42](../crates/onnx-runtime-ep-cpu/src/kernels/split.rs#L18-L42)); and
- shape inference ([movement.rs, lines 448-512](../crates/onnx-runtime-shape-inference/src/handlers/movement.rs#L448-L512)).
If EPContext export must serialize strict opset-21 ONNX rather than internal IR, the exporter should lower the attribute to a constant input. This is an open serialization detail, not a runtime correctness issue.
### 7.2 Decode zero-copy split
Extend `SplitKernel::view_outputs` for fixed-width dtypes when:
- all split sizes are compile-time known;
- the split axis is the last axis;
- outputs exactly cover that axis; and
- byte offsets and strides pass checked arithmetic.
For input shape `[d0, ..., d(r-2), Ntotal]`, output segment `i` has:
```text
shape       = [d0, ..., d(r-2), Ni]
byte_offset = prefix_i * element_size
strides     = input strides, with the same last-axis stride
```
At decode `[1,1,Ntotal]`, choose canonical contiguous output strides `[Ni, Ni, 1]`; extent-1 leading dimensions make these equivalent to the source geometry and allow contiguous-only consumers to avoid executor materialization. For `M>1`, the segment is a regular strided view with row stride `Ntotal`. Consumers that do not support strided input will be materialized by the existing executor gate, preserving correctness.

This avoids changing RoPE or GQA. Their input `ValueId`s, shapes, dtypes, and logical values remain the same. GQA's unpacked path requires rank-3 Q, K, and V with the expected head products ([group_query_attention.rs, lines 121-155](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L121-L155)).
### 7.3 Packed GQA bypass
When legal, packed GQA is better than split-and-recombine. The GQA kernel already interprets Q, K, and V segments in the required order and applies RoPE to the resulting Q and K representations. Rewiring to packed form therefore preserves attention semantics while removing split overhead.

This optimization must be guarded by exact consumer analysis. If Q, K, or V is also a graph output, feeds diagnostics, or has any non-GQA consumer, retain the general split form.
### 7.4 Prefill behavior
The same session graph serves prefill and decode, so the rewrite cannot assume `M=1` for correctness. For `M>1`:
- the fused MatMulNBits remains mathematically valid;
- the current accuracy-4 path quantizes each activation row ([matmul_nbits.rs, lines 619-668](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L619-L668));
- packed GQA remains valid; and
- general split views may be strided and may trigger downstream materialization.
The rollout must benchmark TTFT/prefill as a non-regression gate. If gate/up fusion materially regresses prefill, expose a session option to disable the rewrite until a direct multi-output kernel can write separate contiguous buffers without an intermediate fused result.
## 8. Qwen2.5-0.5B shape math
`genai_config.json` reports:
- `head_size = 64`;
- `hidden_size = 896`;
- `num_attention_heads = 14`;
- `num_key_value_heads = 2`; and
- 24 layers (`/home/justinchu/qwen2.5-0.5b-int4-onnx/genai_config.json`, lines 10-26).
Therefore:
```text
K = hidden_size = 896
block_size = 32
K_blocks = 896 / 32 = 28
packed bytes per output column = 28 * 16 = 448
scale values per output column = 28

Q heads = 14
K heads = 2
V heads = 2
head_dim = 64

Nq = 14 * 64 = 896
Nk =  2 * 64 = 128
Nv =  2 * 64 = 128
Nqkv = 896 + 128 + 128 = 1152
```
QKV fused tensors are:
```text
Bq   [896, 28, 16]   = 401,408 bytes
Bk   [128, 28, 16]   =  57,344 bytes
Bv   [128, 28, 16]   =  57,344 bytes
Bqkv [1152,28, 16]   = 516,096 bytes

Sq   [896, 28]       = 25,088 f32 = 100,352 bytes
Sk   [128, 28]       =  3,584 f32 =  14,336 bytes
Sv   [128, 28]       =  3,584 f32 =  14,336 bytes
Sqkv [1152,28]       = 32,256 f32 = 129,024 bytes

Yqkv decode shape    = [1, 1, 1152]
split sizes          = [896, 128, 128]
```
The inspected model's existing packed QKV initializer is exactly `[1152,28,16]` with external length 516,096 bytes, confirming the formula.

For the MLP, inspection found `intermediate_size = 4864` through each node's `N` attribute and initializer dimensions:
```text
Ngate = 4864
Nup   = 4864
Nfused = 9728

Bgate/Bup each [4864,28,16] = 2,179,072 bytes
Bfused          [9728,28,16] = 4,358,144 bytes

Sgate/Sup each [4864,28] = 136,192 f32 = 544,768 bytes
Sfused          [9728,28] = 272,384 f32 = 1,089,536 bytes

Yfused decode shape = [1,1,9728]
split sizes         = [4864,4864]
```
GQA asymmetry is not a packing problem: it only makes segment lengths unequal. Every segment shares `K=896`, `block_size=32`, and 28 complete K blocks per output column.
## 9. Interaction with current performance paths
### 9.1 Thread-count-aware `N` partition
`output_chunk_len` gates parallel work by `N*K`, current Rayon thread count, minimum work per thread, and task count ([matmul_nbits.rs, lines 811-842](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L811-L842)). Fusion increases `N` while preserving `K`:
```text
QKV:     896 + 128 + 128 -> 1152
Gate/up: 4864 + 4864     -> 9728
```
Expected effects:
- small K/V projections no longer fall below parallel thresholds independently;
- the scheduler can distribute QKV's total 1152 columns across one task set;
- gate/up removes one separate task graph and barrier;
- task chunking is computed once for the larger domain; and
- workers can move directly from one output-column segment into the next.
Gate/up columns are already large enough to parallelize well at 24 workers, so its upside is primarily fewer launches/barriers and shared activation setup, not more theoretical parallelism. QKV benefits more from combining asymmetric small K/V widths.
### 9.2 Direct-int4 VNNI
The fused node remains eligible when:
- `accuracy_level == 4`;
- decode has `M == 1`;
- `block_size == 32`;
- zero points are absent;
- `g_idx` is absent; and
- AVX-VNNI or AVX512-VNNI/AVX512VL is available.
Those are exactly the current checks ([matmul_nbits.rs, lines 157-163](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L157-L163)). Concatenation does not alter packed row layout, so the fused node uses the same `int4_dot_row` routines and implicit zp=8.

One fused invocation quantizes the activation once. Today separate gate and up calls each invoke `quantize_activation_signed`; a separate Q/K/V graph invokes it three times. Fusion changes neither rounding nor scale for a given shared activation, because all siblings already see byte-identical f32 input. It only reuses the one computed quantized row and activation scale.
### 9.3 `accuracy_level`
The pass must preserve `accuracy_level` exactly and reject mixed groups. Level 4 uses quantized activation paths; default level 0 uses f32 dequantized weights for `M=1`; other conditions may use int8 or GEMM fallbacks ([matmul_nbits.rs, lines 193-257](../crates/onnx-runtime-ep-cpu/src/kernels/matmul_nbits.rs#L193-L257)).

Fusion is algebraically valid for every path if attributes match. Performance claims in this document are specifically for symmetric block-32, `accuracy_level=4`, direct-int4 decode.
### 9.4 RoPE and GQA
The fused projection must not alter:
- Q/K/V segment order;
- head-major interpretation inside each segment;
- Q width versus K/V width;
- RoPE configuration or position inputs;
- KV cache shape/layout; or
- attention scale.
The CPU GQA packed form derives `head_dim` from `packed_hidden / (num_heads + 2*kv_num_heads)` and slices Q, K, V in that order ([group_query_attention.rs, lines 173-217](../crates/onnx-runtime-ep-cpu/src/kernels/group_query_attention.rs#L173-L217)). For unpacked consumers, the Split outputs retain the original `ValueId`s and shapes. Thus downstream RoPE/GQA sees unchanged tensors.
## 10. Correctness and validation plan
### 10.1 Unit tests: concatenation
Add table-driven tests for a helper that concatenates projection constants:
1. block size 32, K divisible by 32, unequal `N=[7,2,2]`;
2. K not divisible by block size, proving padded final block bytes remain intact;
3. equal `N=[5,5]` gate/up;
4. absent symmetric zp;
5. explicit packed zp with odd `K_blocks`;
6. concatenated bias;
7. byte-identical shared `g_idx` if supported;
8. mismatched attributes rejected;
9. malformed byte length rejected; and
10. checked-overflow rejection.
For every accepted case:
- compare fused packed bytes with literal row concatenation;
- compare fused scales with literal row concatenation;
- dequantize each old tensor and the fused tensor independently;
- assert fused dequantized rows equal the concatenation of old rows exactly; and
- run old GEMVs and fused GEMV on the same activation, then compare each output segment.
### 10.2 Unit tests: graph rewrite
Build small IR graphs and assert:
- valid QKV siblings fuse in semantic Q/K/V order;
- valid gate/up siblings fuse in gate/up order;
- already-packed QKV is unchanged;
- arbitrary same-input MatMulNBits siblings do not fuse;
- different K/block/bits/accuracy/zp/g_idx/bias contracts do not fuse;
- graph-output projections do not fuse;
- multi-consumer QKV uses Split rather than packed-GQA bypass;
- sole-consumer unpacked GQA becomes packed form;
- old orphan initializers disappear;
- shared initializers remain;
- output `ValueId`s and consumer edges remain correct; and
- `graph.validate()` and topological ordering pass after every rewrite.
### 10.3 Kernel and split tests
Add tests proving:
- fused direct-int4 output equals concatenated separate direct-int4 outputs;
- unequal segment widths and N tails work;
- `Split::view_outputs` yields correct offsets and values;
- decode `[1,1,N]` outputs advertise contiguous strides;
- prefill `[B,S,N]` outputs use correct strided geometry;
- unsupported Split cases fall back to copying; and
- downstream contiguous-only kernels receive correct auto-materialized values.
### 10.4 End-to-end Qwen validation
Run the native decode benchmark with fusion disabled and enabled:
```text
model: /home/justinchu/qwen2.5-0.5b-int4-onnx
RAYON_NUM_THREADS=24
accuracy_level=4
four greedy decode tokens
```
The known-good token IDs are:
```text
[11576, 42740, 11, 358]
```
They are recorded for the direct-int4 comparison ([decode profile, lines 187-196](benchmarks/2026-07-16-decode-profile2.md#L187-L196)). The fused run must reproduce them exactly.

Also capture logits for each step and compare fused versus unfused:
- no NaN/Inf differences;
- same argmax token at every step;
- elementwise `atol`/`rtol` no looser than the existing accuracy-level-4 kernel tests; and
- preferably bit-identical segments, because output columns execute the same dot routine in the same K-block order.
Bit identity is expected for the direct-int4 path: fusion changes scheduling across output rows but does not change reduction order within an output row. The test should nevertheless state tolerance rather than assume all fallback paths are bitwise stable.
### 10.5 Performance validation
Measure separately:
1. projection-fusion pass load time;
2. steady-state first token after warmup;
3. decode tok/s and ms/step;
4. `MatMulNBits` total time and call count;
5. `Split` time/call count;
6. activation quantization time;
7. Rayon worker counts 1, 24, 48, and 96;
8. prefill/TTFT at representative prompt lengths; and
9. resident memory after all fused initializers are touched.
For the inspected packed-QKV model, expected structural counts are:
```text
before: 121 MatMulNBits calls/step
fuse 24 gate/up pairs
 after:  97 MatMulNBits calls/step + 24 logical splits
```
For an otherwise equivalent separate-QKV graph:
```text
before: 169 MatMulNBits calls/step
fuse 24 QKV triples and 24 gate/up pairs
 after:  97 MatMulNBits calls/step
```
Packed-GQA bypass removes QKV Split calls. Gate/up zero-copy views should make logical Split profiler time approximately zero at decode.
## 11. Expected upside
Projection fusion does **not** reduce int4 weight traffic or dot-product count. It attacks the overhead around those dot products:
- one activation densification instead of two or three;
- one activation quantization instead of two or three;
- one result setup and writeback boundary;
- one kernel dispatch;
- one Rayon task graph/barrier;
- larger `N` scheduling domain; and
- better cache residency of the small activation vector while adjacent projection rows execute.
The earlier phase breakdown put activation quantization at about 3.2% of MatMulNBits time before direct-int4 and identified repeated barriers as a scaling issue ([decode profile, lines 141-159](benchmarks/2026-07-16-decode-profile2.md#L141-L159)). Only the sibling subset of quantization is removable, and direct-int4 GEMV still dominates. Therefore expectations should be bounded:
- **measured artifact, gate/up only:** target **3-8% lower MatMulNBits time** and roughly **2-7% higher end-to-end decode throughput**;
- **separate-QKV plus gate/up export:** target **5-12% lower MatMulNBits time** and roughly **4-10% higher end-to-end decode throughput**; and
- larger gains are possible at high thread counts where small independent projections cross the current parallelism cliff, but 24 workers remains the primary absolute-throughput configuration.
These are design targets, not claimed results. The upper bound is constrained by unchanged packed-weight bandwidth and unchanged per-output arithmetic.
## 12. Risks and cases where fusion may not help
### 12.1 GQA asymmetry
Q has 896 outputs while K and V have only 128 each. Fusion improves the small K/V scheduling domain, but Q already dominates QKV work. A poor chunk boundary could also split work unevenly across segments. The kernel partitions the total flat `N`, so segment boundaries must not become task boundaries.
### 12.2 Gate/up already parallelize well
Each gate/up projection has `N=4864`; at 24 workers each is already large. Fusion removes a barrier and repeated activation setup but does not make weight reads cheaper. The benefit may be modest.
### 12.3 Split overhead
The current copying Split may erase much of the win, especially for gate/up's 9728 f32 outputs. Zero-copy decode views or packed-GQA bypass are part of the recommended design, not an optional follow-up.
### 12.4 Prefill materialization
For `M>1`, last-axis segments are strided across rows. Consumers that require contiguous inputs will materialize them. Prefill may therefore gain less or regress even when decode improves.
### 12.5 Session-load memory and latency
Concatenated inline initializers duplicate fused projection bytes relative to the mmap. Per Qwen layer, gate/up fused storage is about 5.20 MiB including weights and scales; across 24 layers this is about 124.7 MiB of owned session memory. QKV adds about 0.62 MiB per layer when starting from separate weights. This is materially smaller than fp32 expansion, but not free.

A production implementation should report fused owned bytes and consider a contiguous anonymous mmap or composite prepack cache if allocator fragmentation appears.
### 12.6 Over-broad matching
“Same activation and compatible dimensions” is insufficient semantic proof. Models may intentionally compute independent heads, adapters, experts, or auxiliary outputs from the same tensor. Conservative consumer-pattern matching is mandatory.
### 12.7 Optional-input differences
Explicit zp, `g_idx`, bias, or different `accuracy_level` values can make a mathematically tempting group illegal or change fast-path eligibility. Reject rather than normalize in the first implementation.
### 12.8 Profiling observability
A fused node hides per-projection timings. Give generated nodes deterministic names and record segment labels in a debug-only attribute or fusion stats so profiles can still identify QKV versus gate/up.
## 13. Generalization
The mechanism is model-independent, but pattern recognition must be model-conservative.
### 13.1 Models likely to benefit
- Qwen/Llama/Mistral-style dense decoders with separate Q/K/V and gate/up;
- Gemma variants with compatible sibling int4 projections;
- models exported with unpacked GQA inputs;
- any decoder whose quantized projection rows share K/block/accuracy contracts.
### 13.2 Models requiring special handling
- models with QKV interleaved by head rather than concatenated Q|K|V;
- models applying distinct pre-projection transforms;
- Q/K normalization between projection and attention;
- separate RoPE graphs with nontrivial reshape/transpose chains;
- fused bias/activation custom ops;
- MoE expert projections, where grouping can explode memory and scheduling semantics differ; and
- adapters or LoRA branches sharing the same hidden activation.
The pass should use structural roles and attributes, not model-name checks. Add new patterns one at a time with a real exported graph fixture.
## 14. Alternatives considered
### 14.1 Runtime sibling gather
**Idea:** retain separate graph nodes, detect ready siblings in the executor, and invoke a grouped kernel.

**Pros:** no graph mutation; could choose grouping from runtime shapes. **Cons:** repeated discovery, complex kernel-cache identity, hidden scheduling, awkward output allocation, harder profiling, and no natural immutable fused initializer. **Decision:** reject.
### 14.2 One Rayon parallel operation over separate GEMVs
**Idea:** keep separate weights and outputs, but create one outer Rayon `parallel_for` whose flat task space covers `(projection, output_column)`.

**Pros:** cheapest implementation; no weight copy; no Split; preserves separate contiguous outputs; may remove sibling barriers and improve K/V utilization. **Cons:** activation densification/quantization is still duplicated unless explicitly hoisted; the kernel API currently receives one node at a time; weights remain multiple `OnceLock` objects; dispatch/executor grouping is still required; activation and scale access crosses several objects. **Decision:** retain as the principal fallback experiment. It may deliver a partial benefit with much lower session-memory cost and is especially attractive if inline concatenation's ~125 MiB gate/up copy is unacceptable.

A serious version of this alternative would introduce a grouped internal kernel object at load time, not runtime sibling discovery. It would hold references to separate packed weights, quantize activation once, and execute one flat Rayon space while writing separate output buffers. This is effectively multi-output fusion without physical weight concatenation.
### 14.3 Custom multi-output `FusedMatMulNBits`
**Idea:** rewrite siblings to an internal custom op with multiple outputs and one grouped kernel, writing directly to separate output buffers.

**Pros:** no intermediate fused result; no Split; prefill outputs remain contiguous; can quantize once. **Cons:** duplicates or refactors the MatMulNBits kernel contract; adds an internal op to CPU registration and EPContext serialization; other EPs cannot consume it without lowering; standard tooling will not understand it. **Decision:** phase-2 option if Split views/materialization limit the standard rewrite.
### 14.4 Export-time fusion only
**Idea:** require model builders to emit packed QKV and gate/up.

**Pros:** no runtime rewrite; model can encode exact semantic intent; packed QKV already exists in the inspected artifact. **Cons:** does not help existing models; exporters differ; gate/up packed output still needs consumer support; native runtime should optimize safe standard patterns. **Decision:** encourage exporter fusion, but do not rely on it exclusively.
### 14.5 Activation-quantization cache without projection fusion
**Idea:** cache quantized activations by input `ValueId` and run identity, then reuse them across separate MatMulNBits kernels.

**Pros:** removes repeated quantization with little graph change. **Cons:** quantization was a small fraction of total time; leaves all launches, barriers, and small-N scheduling; cache lifetime and stale-activation safety must be exact. **Decision:** complementary, lower expected return by itself.
## 15. Proposed implementation phases
### Phase A: pass and byte-exact helpers
- implement conservative gate/up matching;
- implement separate-QKV direct-GQA matching;
- concatenate B/scales with absent zp/g_idx only;
- preserve `accuracy_level=4` and all attributes;
- add graph validation and fusion statistics;
- add unit tests without enabling by default.
### Phase B: output strategy
- use packed-GQA bypass where legal;
- add last-axis zero-copy Split views for decode;
- retain copy/materialization fallback for general shapes;
- test prefill geometry and downstream auto-materialization.
### Phase C: guarded rollout
- add an opt-out environment/session option during benchmarking;
- enable for CPU only;
- record fused group count and owned fused bytes;
- validate Qwen tokens and logits;
- benchmark 1/24/48/96 workers and TTFT.
### Phase D: generalization
- explicit zp support;
- compatible shared `g_idx` support;
- fused bias/Add patterns;
- additional Gemma/Llama graph fixtures; and
- consider grouped multi-output kernel if memory or prefill results demand it.
## 16. Decisions
| ID | Decision | Rationale |
|---|---|---|
| PF-1 | Fuse at load time in `Executor::build`, after SiLU fusion and before topological planning. | Mutable graph and live weight bytes are available before buffers/plans/kernels exist. |
| PF-2 | Physically concatenate packed B and scales along outer `N` for the primary design. | Current output-major layout makes concatenation byte-exact and gives one prepacked kernel object. |
| PF-3 | Preserve implicit symmetric zp=8 by keeping zp absent. | Synthesizing zp would disable direct-int4 eligibility. |
| PF-4 | Require matching K, block size, bits, accuracy level, optional-input contracts, and output prefix shape. | One fused node must have one coherent kernel contract. |
| PF-5 | Pattern-match semantic QKV and gate/up roles; never fuse arbitrary same-input siblings. | Prevents silently combining unrelated projections. |
| PF-6 | Use Q|K|V and gate|up segment order derived from consumers. | Downstream head interpretation and SwiGLU roles are order-sensitive. |
| PF-7 | Prefer packed-GQA bypass when Q/K/V have no other consumers. | Existing GQA supports packed input and this removes Split entirely. |
| PF-8 | Otherwise preserve original output ValueIds through last-axis Split. | Existing RoPE/GQA/MLP consumers remain logically unchanged. |
| PF-9 | Add zero-copy Split views for decode; do not judge fusion using the current copying Split alone. | Current Split materializes enough data to obscure the intended saving. |
| PF-10 | Keep direct-int4 code unchanged; fusion only changes N and constant storage. | Minimizes kernel risk and preserves established numerics. |
| PF-11 | Make the pass idempotent and skip already-packed QKV. | The measured Qwen artifact already contains packed QKV. |
| PF-12 | Treat owned fused-initializer bytes and prefill TTFT as release gates. | Fusion can trade decode speed for memory/load/prefill regressions. |
## 17. Open questions for owner review
1. **Memory budget:** Is approximately 125 MiB of owned gate/up fused bytes for Qwen2.5-0.5B acceptable, or should the first prototype use a grouped kernel over separate weight buffers?
2. **Internal Split encoding:** May the post-load IR use the supported legacy `split` attribute under opset 21, or must EPContext export always materialize an int64 split initializer?
3. **Rollout scope:** Should fusion be enabled by default after Qwen validation, or initially guarded by `ONNX_GENAI_PROJECTION_FUSION=1`?
4. **Prefill policy:** What TTFT regression threshold is acceptable in exchange for decode gain? Proposed gate: no more than 2% at representative prompt lengths.
5. **Packed GQA:** Should the first implementation include packed-GQA rewiring, or start with gate/up only because the benchmark artifact's QKV is already packed?
6. **Bias pattern:** Should separate post-MatMul bias Adds be fused in phase A, or deferred until no-bias gate/up is measured?
7. **Explicit zp:** Is support needed before rollout, or can phase A target the symmetric absent-zp direct-int4 path only?
8. **Optimization ownership:** Keep the pass beside `fuse_silu_patterns` for now, or create a dedicated `onnx-runtime-session::optimizers` module first?
9. **Profiler naming:** Should fused segments be reported as one `MatMulNBits[fused_gate_up]` category or retain synthetic per-segment metadata?
10. **Fallback experiment:** If physical concatenation wins less than 3% decode or exceeds the memory budget, should grouped separate-buffer execution become the preferred architecture?
## 18. Acceptance criteria
The design is ready to implement only when the owner accepts PF-1 through PF-12 or records replacements. Implementation is complete only when:
- the pass fuses all 24 gate/up pairs in the inspected model;
- separate-QKV fixtures fuse or packed-QKV fixtures are correctly skipped;
- packed bytes/scales are proven row-concatenations;
- output segments match unfused numerics;
- Qwen greedy tokens remain `[11576, 42740, 11, 358]`;
- direct-int4 eligibility remains active;
- graph validation and all targeted tests pass;
- decode improves at 24 workers;
- prefill stays within the agreed regression budget; and
- owned fused bytes and pass time are reported, not hidden.
