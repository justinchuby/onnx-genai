### 2026-07-23: Adversarial review of `perf/matmul-nbits-block128` (Deckard) — 🔴 REJECT (do not merge to main)

**Reviewer:** Holden
**Branch:** `origin/perf/matmul-nbits-block128` @ `5821162`
**Baseline:** `origin/main` @ `569507c`
**GPU:** device 0 (idle), `.cudaenv.sh` sourced.

## Verdict: 🔴 REJECT — superseded / stale, would regress main

The kernel code Deckard wrote is, on its own, **correct** — but the branch **must not be merged to main** because main *already* implements block-size-general fp16 int4 (including block-128), the branch is **~70 commits behind main**, and merging it would revert large amounts of landed work and re-introduce 2 test failures that main has already fixed.

**Deckard is locked out on this reject.** No re-implementation is needed — the capability already exists on main. Recommended next owner: **coordinator** to close/abandon this branch as *superseded*; optionally **gaff** (owner of the landed block-size-general work, `c04a622` / `0fa57b0`) to confirm main's block-128 path covers any remaining export shapes. Do NOT reassign the "add block-128" task — it is already done on main.

## The gate: ACTUAL numbers

Identical command on both, `CUDA_VISIBLE_DEVICES=0 cargo test -p onnx-runtime-ep-cuda --features cuda --lib`:

| Ref | Result |
|-----|--------|
| `origin/main` @ 569507c | **201 passed / 0 failed** (incl. `fp16_gemv_matches_dequant_reference_block128` PASS) |
| `origin/perf/matmul-nbits-block128` @ 5821162 | **158 passed / 2 failed** (160 total) |

### Explaining Deckard's "158 / 2 pre-existing failures"
His "158 passed, 2 unrelated pre-existing failures" is measured **against his own stale base (parent `689102c`), not against main**. The true main baseline for the same command is **201/0**. The branch's lib test binary contains only 160 tests vs main's 201 — it is missing ~41 tests that main added.

### The "2 pre-existing failures" are NOT pre-existing vs main — they are regressions of the stale base
Both tests exist on main *and* branch. On main they **PASS**; on the branch they **FAIL**:
- `kernels::tests::covered_ops_have_no_duplicates` — `mod.rs:714` `left: 88, right: 87` (op-coverage count drift).
- `kernels::group_query_attention::tests::sequence_lengths_shape_rejects_noncanonical_singleton_layouts_actionably` — `group_query_attention.rs:2433` error-message assertion.

Neither touches MatMulNBits. They were fixed in main's ~70 unmerged commits (e.g. library-expectation refreshes). Characterizing them as "unrelated pre-existing" is misleading: they are only "pre-existing" relative to Deckard's outdated fork.

## Topology (root cause of the reject)

`git merge-base origin/main origin/perf/...block128` = **`f0af865`** (`docs: complete ORT CUDA comparison`) — a stale point. Main has **~70 commits not on the branch**, including the ones that make this whole branch redundant:
- **`0fa57b0 feat(cuda): model-agnostic MatMulNBits block_size for fp16 path`**
- **`c04a622 Address review nits for MatMulNBits block-size-general support`**

Main implements block-size-general fp16 int4 via a `matmul_nbits_gemv_f16_general_bs` kernel selected for **any** `block_size != 32`, with a passing parity test `fp16_gemv_matches_dequant_reference_block128`. Main's factory/run gate errors only for `bits == 8 && block_size != 32` — fp16 **int4** block-128 is fully accepted. This is a *different, already-reviewed* solution than Deckard's `matmul_nbits_gemv_f16_flex_block`.

Merging Deckard's branch to main would either conflict massively or roll back ~70 commits (Phi capture-seam elimination, QMoE graph capture, split-K GEMVs, domain refactor, etc.). That is the regression risk.

## Real-model proof the capability already exists on main
`profile_native --model .../deepseek-v2-lite-real-int4/ --ep cuda --tokens 8` run **on main (569507c)**:
```
model=.../model.onnx ep=Cuda layers=27 prompt_tokens=[17464] tokens=8
throughput: 12.62 tok/s, 79.238 ms/step
generated_token_ids: [11, 304, 608, 245, 207, 16, 22, 1012]
generated_text: ", I am a 17 year"
```
q_proj does **not** error; tokens are finite and coherent. The real block-128 DeepSeek export already works on main. Deckard's premise ("previously only supported block-32 and hard-errored at q_proj") was true of his stale base, but is **false against current main**.

## Code-level assessment of the branch (for the record — the code is fine, the base is not)
To be fair to Deckard, I audited the actual kernel changes and found them numerically correct:
1. **Flex GEMV dequant** (`matmul_nbits_gemv_f16_flex_block`): scale/zp indexed by `depth / block_size`; low-nibble-first unpack (`(within & 1) ? byte>>4 : byte&15`); `blob_size = block_size*bits/8`; ragged-K safe (`for depth=lane; depth<k; depth+=32`). ABI is byte-identical to the General GEMV (verified against `matmul_nbits_gemv_f16` signature and the shared launch arg-binding block), which is required since both share the same launch path. Alloc/sync-free ⇒ capture-safe; the parity test asserts `last_call_capture_safe == true`.
2. **Generalized tiled prefill GEMM**: tiles K in 32-wide chunks (`k_tiles=(k+31)/32`) while indexing the quant block independently via `block = depth/block_size`, `block_within = depth - block*block_size`. For `block_size==32` this reduces exactly to the old `block==tile`, `block_within==within` layout ⇒ **block-32 path unchanged**. Packed offset `(col*k_blocks+block)*blob_size + block_within/2` is contiguous and correct for block-128.
3. **Block-32 fast-path gating preserved**: `select_f16_gemv_variant` returns `FlexibleBlock` only when `block_size != GEMV_F16_DOWN_BLOCK_SIZE`, and `GEMV_F16_DOWN_BLOCK_SIZE == 32` (verified). So block-32 still reaches DownProjection/General unchanged; the flex path is strictly additive.
4. **Parity test is legitimate, not tautological**: `fp16_block128_gemv_matches_dequant_reference` builds an **independent f64 dequant-then-matmul oracle** (`q-8` implicit zp, `*scale`, over the same fp16-rounded activations), runs the real kernel, and asserts finite + abs/rel bounds. Good test.

None of this changes the verdict: correct code on the wrong base is still un-mergeable when main already ships the feature.

## Recommendation
- **Abandon/close** `perf/matmul-nbits-block128` as superseded by `0fa57b0` + `c04a622` on main.
- If there is any doubt that main covers every real export shape, have **gaff** run the DeepSeek/GPTQ block-128 smoke against main (I already confirmed DeepSeek-V2-Lite works) rather than resurrecting this branch.
- If, in future, block-128 work is needed, branch **from current main**, not from `f0af865`.
