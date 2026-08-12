### 2026-08-12: mobius_seqmajor parity gate — CONFIRMED reliable solo (5/5 on clean base); contended runs are INVALID and must be re-run solo

**By:** Copilot (pinned-staging-pool, #837 item 2)

**Update (decisive, replaces the earlier "is it flaky?" question):**
Ran the mandatory gate **5x back-to-back on clean `origin/main` dccb40e8** (per
coordinator request), full per-run stderr captured. **Result: 5/5 PASS**, zero
ILLEGAL_ADDRESS / shape-inference / "1 failed" markers. Two runs fully solo,
three under mild concurrent load — all passed. Contention only inflated wall
time (125s solo → 421s overlapped). So the gate is **NOT intrinsically
nondeterministic**; a single **solo** green run is meaningful. The earlier reds
(illegal-address, shape-inference, wrong-subtest) all occurred under **heavy**
GPU saturation — multiple concurrent full qwen14b mobius runs on one 8 GB card —
which is an OOM-family artifact, not a regression signal.

**Operational rule (write this down):**
- A **red** run of `mobius_seqmajor_growth_parity_native_cuda` under GPU
  contention is **INVALID** — re-run it solo before drawing any conclusion.
- Before treating any red as a regression: check `nvidia-smi
  --query-compute-apps`; if any other compute PID is present, discard and re-run
  in a verified-solo window (0 compute apps for a sustained period).
- A **real** return-to-pool-before-fence corruption would fail the bit-identical
  parity subtest *deterministically and solo* — not intermittently under load.

**Why / how to act (original guidance retained):**
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
