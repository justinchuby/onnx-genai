### 2026-08-12: mobius_seqmajor parity gate is flaky under shared-GPU contention — isolate before treating a failure as a regression

**By:** Copilot (pinned-staging-pool, #837 item 2)

**What:**
The mandatory memory-governance gate
`cargo test -p onnx-genai-engine --features cuda,native-backend --test
mobius_seqmajor_growth_parity_native_cuda -- --ignored --test-threads=1`
is **flaky when other agents are using the single shared GPU**. In one session it
failed three times with *different* signatures — twice `CUDA_ERROR_ILLEGAL_ADDRESS`
during the forward pass / VMM arena teardown, once the unrelated subtest
`head_major_growth_that_moves_addresses_must_invalidate_not_keep`, and once a
`no inferred shape for value present.19.key` on a *clean* checkout — yet passed a
clean **2/2** on retry in a solo window. Crucially the failures reproduced on the
**untouched base commit `6c714cb7`** as well as on the feature branch, and the
bit-identical parity subtest itself (`mobius_seq_major_growth_is_bit_identical_to_head_major`)
never produced a data mismatch — only whole-process GPU-context crashes under
contention.

**Why / how to act:**
When this gate fails, do **not** immediately conclude your change regressed it.
Isolate first:
1. Check `nvidia-smi --query-compute-apps` and coordinate a solo window with the
   sibling GPU agents (`vmm-churn`, `native-batch-decode`, `kv-floor-longprefix`).
2. `git stash` + `git checkout <base>` and run the same gate on the clean base. If
   the base flakes identically, the failure is environmental, not your change.
3. Re-run in a confirmed solo window; require a clean 2/2. A random *different*
   subtest failing across runs is the signature of contention, whereas a real
   weight-corruption regression would fail the bit-identical parity subtest
   *deterministically*.
The distinction matters because this path (async H2D weight paging) has a genuine
fence hazard, so a flaky red is easy to misread as the hazard firing.
