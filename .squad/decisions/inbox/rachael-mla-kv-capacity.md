# DeepSeek-V2-Lite MLA decode nondeterminism — root cause & fix

**Author:** Rachael (worker)
**Date:** 2026-07-23
**Branch:** `fix/deepseek-mla-kv-capacity` @ `897e53f` (off clean origin/main `1fe314f`) — **pushed, NOT merged**
**Priority:** CORRECTNESS (Justin's "token divergence must be fixed" mandate). Capture/perf enablement is a deferred follow-up (see bottom).

## Root cause (PINNED, with evidence)

The default-domain `Attention` decode graph for DeepSeek-V2-Lite (MLA) binds the
`present_key` / `present_value` **outputs** onto the **same device buffers** as
the `past_key` / `past_value` **inputs** — i.e. in-place KV-cache growth.
Verified at runtime: `present_key_ptr == past_key_ptr` (and value), `alias_k=true, alias_v=true`.

`build_kv` (standard_attention.rs) rebuilds the whole cache with a **wider
per-head stride**: it reads `past` at head-stride `past_seq` and writes
`present` at head-stride `total_seq = past_seq + cur_seq` in the *same* buffer.
For head `h`, the current-token store lands at offset `(h·total_seq + past_seq)·dim`,
which **collides with head `h+1`'s past load** at `((h+1)·past_seq)·dim` (e.g.
head0 cur-write == head1 past-read at offset `past_seq`). Across unordered CUDA
threads this is a read-after-write hazard → **every head beyond head 0 is
nondeterministic; head 0 alone stays correct** (its write region doesn't overlap
earlier-written data).

**Evidence chain:** instrumented the attention op — at decode step 0, layer 0,
`q`/`k_cur`/`v_cur`/head-0 `past_key` inputs were **bit-identical across
processes but attention output Y differed**; the dense all-head `past_key` hash
**differed run-to-run** → past_key heads 1..15 were being corrupted. Confirmed
`present==past` aliasing. QMoE router weights already differed at the *first*
decode-step MoE, i.e. divergence is upstream of routing, in the attention KV.

**Why it evaded detection:** (1) synthetic-weight validation only checked
finiteness; (2) prefill has `has_past=false` so build_kv is a non-overlapping
copy → deterministic; (3) the perturbation is small (~1 logit) so text stays
coherent and argmax only flips at ~token idx 4; (4) `compute-sanitizer
racecheck` did not flag the global RAW hazard (coverage limitation). Ruled OUT:
uninitialized KV padding (zero-init did nothing), attention math, QMoE/TopK,
launch races, GEMV determinism.

## Fix (general, not DeepSeek-special-cased)

`crates/onnx-runtime-ep-cuda/src/kernels/standard_attention.rs`, in
`StandardAttentionKernel::execute`: when `present` K/V aliases `past` K/V,
build the grown cache into a **disjoint scratch buffer** (reads the pristine
past), run attention against the scratch, then `dtod`-copy the fully-formed
dense cache **back** into the aliased buffer. Source and destination are
disjoint → the copy is race-free. Applies to any model whose default-domain
Attention grows an aliased KV cache. GroupQueryAttention (Qwen/Phi) is
unaffected — it appends only the new token at a fixed physical slot and never
restrides existing KV.

`dtod` (runtime.rs:731) does a `synchronize()` + synchronous `cuMemcpyDtoD`;
this keeps the standard-Attention path **eager** (its `capture_support` is
already `Unsupported`), which is consistent with the deferred capture phase.
The earlier dtod race-fix (`dtod_waits_for_pending_stream_writes`) still passes.

## Determinism proof (real weights, GPU 3, committed build)

blk32 `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4-blk32/`, 2 runs:
```
[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]   " Paris.\nThe currency of France is the Euro.\n"
[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]   (identical)
```
blk128 `/home/justinchu/ds-e2e-artifacts/deepseek-v2-lite-real-int4/`, 2 runs:
```
[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]   " Paris.\nThe currency of France is the Euro.\n"
[8913, 13, 185, 549, 19305, 280, 7239, 317, 254, 28071, 13, 185]   (identical)
```
pos0 = **8913 ' Paris'** ✓ (matches ORT/HF reference). Prior session confirmed 3× identical on both exports.

## Dense non-regression
Qwen2.5-0.5b (`/home/justinchu/qwen2.5-0.5b-int4-onnx-native/`): still
`cuda_graph captures=2 replays=18 fallbacks=0`, coherent
(" Paris. It is the largest city in the country and the"). No regression.

## Regression test
`standard_attention.rs` → `#[cfg(test)] mod alias_tests`:
`decode_kv_growth_alias_matches_reference_and_is_deterministic`. Constructs a
multi-head (heads=4), asymmetric-head (k=6, v=4) KV cache grown by one token
with `present` aliasing `past` in one device buffer; asserts the aliased result
(a) equals a non-aliased reference and (b) is bit-identical across 4 repeated
runs. Skips gracefully with no CUDA. General (guards the aliased-KV-growth class,
not DeepSeek).

## Gate (source .cudaenv.sh; CUDA_VISIBLE_DEVICES=3; worktree off 1fe314f)
- `cargo test -p onnx-runtime-ep-cuda --features cuda --lib` → **203 passed / 0 failed** (was 202/0 baseline + 1 new).
- `cargo clippy -p onnx-runtime-ep-cuda --features cuda --lib -- -D warnings` → **clean**.
  (Pre-existing `--tests` clippy nits in matmul_nbits.rs / normalization.rs are not from this change and outside the `--lib` gate.)

## Capture status (deferred follow-up — correctness delivered independently)
DeepSeek MLA decode still runs **eager** (`captures=0 fallbacks=0`) — expected.
Capture enablement is the separate perf lever and is a larger rework:
`kernel_input_uses_physical_capacity` (executor.rs:1122) recognizes only
`com.microsoft::GroupQueryAttention` inputs 3/4, so the default-domain Attention
exposes a *dense growing* KV (hence the per-step O(seq) restride this fix stages
safely). To capture, the KV path must become truly **fixed-capacity** (physical
max_len capacity + a device valid-length scalar so the kernel reads exactly the
valid region), `executor.rs:1122` must recognize default-domain Attention KV/mask
as fixed-capacity (mirroring GQA), and `StandardAttentionKernel::capture_support`
must return `Supported` after its per-step D2H reads (`nonpad_kv_seqlen` + htod
control arrays) are replaced with capture-safe device-side handling. Recommend
landing this correctness fix alone and sequencing capture as its own reviewable
change.

## Recommended owner / next
- This fix: engine / standard_attention.rs (Rachael authored). Ready for a fresh reviewer.
- Capture-enablement follow-up: engine (executor.rs binding recognition +
  StandardAttentionKernel capture path). Needs the fixed-capacity KV redesign above.
