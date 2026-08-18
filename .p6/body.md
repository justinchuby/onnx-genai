## Zero-copy host input binding

`Executor::prepare_run_buffers` ended, for every graph input, in
`self.ep.copy_from_host(tensor.as_bytes(), buf)`. That copy is not a detail:
on the attention/MoE benchmark graphs it is **40–76% of the entire run**.

`--phase-profile`, release, 200 runs, 20 warmups, `rope_llama3_s1`
(1×1×4096 f32 in, 4096×64 f32 cos/sin caches):

```
[nxrt-phase] run_scoped.setup_total.top   65.74 us/call
[nxrt-phase] run_scoped.bind_inputs       62.96 us/call   <-- 66% of a 96 us run
[nxrt-phase] exec_kernel.compute           1.88 us/call   <-- the actual RoPE
[nxrt-phase] bind_inputs.host_bytes       2.016 MB/call
```

`rope_llama3_s512` at one thread: `bind_inputs` 425.96 µs of a 1.40 ms run.

ORT pays nothing equivalent — an `OrtValue` can be constructed over the
caller's allocation and the CPU EP reads that memory directly — so this copy
was a fixed, permanent component of every native/ORT ratio in
`docs/benchmarks/2026-08-15-cpu-ep-vs-ort-attention-moe.md`. §21.2 of that
document identified it in Phase 3 and declined to fix it ("a lifetime and
aliasing change to the EP ABI … not attempted here"). Under the current policy
— *never hand a slow range to ORT, make it faster instead* — declining is not
available, and it turns out the ABI change is not needed.

## What changed

The EP ABI already carries the required piece: `DeviceBuffer::from_borrowed_parts`,
used since the loader to alias mmap'd **initializer** bytes zero-copy. A graph
input is producer-less exactly like an initializer, and `inputs: &[(&str, &Tensor)]`
is borrowed for the whole of `run_scoped_mode`, so the same handle is sound for
the duration of one run.

* `prepare_run_buffers` installs a borrowed handle over the caller's tensor and
  parks the owned buffer in a new `parked_input_buffers`.
* `unbind_borrowed_inputs` restores it before `run_scoped_mode` returns —
  through a new `execute_and_collect` split so the error path is covered too —
  and again at the top of `reset_run_state`, so the invariant does not depend
  on any single exit path. If something replaced the alias mid-run (a sequence
  op re-roots storage), the replacement is kept and the parked allocation is
  deallocated, so neither handle leaks.

### Preconditions (all required)

| guard | why |
|---|---|
| `buf.device().is_host_accessible()` and `== self.ep.device_id()` | a heterogeneous session allocates some values through another provider; the handle must name the device whose `deallocate` will see it |
| `bytes.len() == buf.len()` | no partial or oversized aliasing |
| pointer is `TensorLayout::contiguous().alignment`-aligned | kernels may assume the EP's allocation alignment |
| not in `graph.outputs` | `try_move_host_output` can hand a graph output's buffer to the caller; that must never be foreign memory |
| not in `shared_buffers` | sequence storage is re-rooted and written |
| `stage2_excluded.is_none()` | the Stage-2 decode view memo asserts buffer-pointer stability across runs, which a caller-owned address does not have |
| `!bytes.is_empty()` | a zero-length input has a 1-byte placeholder buffer |

### One behaviour trade

A borrowed buffer must never be written, so compute-in-place aliasing now skips
borrowed inputs (`ctx.buffers.get(&vid).is_some_and(|b| !b.is_borrowed())`).
`compute_in_place_chain_is_byte_identical_and_fires` goes from 2 aliases to 1,
and the test now states why. This is the better trade: the allocation in-place
saved is reused across runs for static shapes, while the copy was paid on
**every** run. Byte-identical output is still asserted.

## Evidence

`scripts/ort_ab/ab.py`, interleaved, same binary pair (`base` = this branch's
merge-base build, `new` = this branch), 5 trials × 7 runs × 3 warmups,
ratio = **ours/ORT p50, lower is better**, thread-matched
(`--native-threads N` + `--ort-intra-threads N`). **All 12 cells**, none omitted:

| model | t | before | after | closer by |
|---|--:|--:|--:|--:|
| rope_llama3_s1 | 1 | 7.54 | **1.40** | 5.4× |
| rope_llama3_s1 | 8 | 9.61 | **1.34** | 7.2× |
| rope_llama3_s1 | 16 | 7.22 | **1.40** | 5.2× |
| rope_llama3_b8_s1 | 1 | 6.49 | **1.24** | 5.2× |
| rope_llama3_b8_s1 | 8 | 6.11 | **1.35** | 4.5× |
| rope_llama3_b8_s1 | 16 | 4.32 | **1.24** | 3.5× |
| rope_llama3_s128 | 1 | 2.84 | **1.73** | 1.6× |
| rope_llama3_s128 | 8 | 12.01 | **4.51** | 2.7× |
| rope_llama3_s128 | 16 | 17.23 | **7.71** | 2.2× |
| rope_llama3_s512 | 1 | 2.07 | **1.25** | 1.7× |
| rope_llama3_s512 | 8 | 4.75 | **4.17** | 1.14× |
| rope_llama3_s512 | 16 | 7.77 | **5.83** | 1.3× |

60/60 trials `parity=PASS`. Native p50 on the decode shape falls
0.077 ms → **0.009 ms**.

## Limitations — stated plainly

* **Every cell still loses to ORT.** The decode shapes are now 1.24–1.40×
  behind rather than 4.3–9.6×; `rope_llama3_s128/s512` at 8 and 16 threads are
  still 4.2–7.7× behind, and that residue is the rotation kernel's parallel
  scaling, not binding. It is the next target — not a decline.
* Ratios come from one contended 32-vCPU host and are only comparable *within*
  this driver invocation, per `scripts/ort_ab/README.md`.
* Synthetic data: dimensions from public Llama-3-8B config, contents from the
  harness's deterministic pattern, fed identically to both runtimes. No trained
  weights.
* Runs that restore a Stage-2 decode view memo still copy. Extending the memo's
  signature to cover borrowed input addresses is deliberately left out of this PR.
* This is a session-executor change, so it moves **every** operator's numbers,
  not only RoPE. The rest of the benchmark matrix is re-measured in a follow-up
  before the document is updated.

## Validation

* `cargo test -p onnx-runtime-session` — all green except `projection_fusion`,
  which fails identically on the merge base (`ModuleNotFoundError: onnxscript`,
  environment-only).
* `cargo fmt --all --check`, `cargo clippy -p onnx-runtime-session --all-targets` clean.
* 60/60 A/B trials numerically parity-checked against a real ORT CPU session.

---

## Round 2 — independent Opus review (BLOCKER found and fixed)

An independent reviewer found a **use-after-free introduced by the first
commit** and reproduced it end-to-end. Recorded here in full because the class
matters more than the instance.

**The bug.** A graph input consumed by a sequence op reaches
`read_seq_element` (`sequence_ops.rs:305`), which *moves* the value's buffer
into a `SharedTensorBuffer` held in `shared_buffers` — state that outlives the
run, because `restore_shared_buffers` reinstates it at the top of the **next**
one. With zero-copy binding, the promoted handle was the borrowed alias over
the caller's tensor. The next run therefore reinstalled a `DeviceBuffer`
pointing at memory the caller had already dropped, *and* deallocated the
genuine owned allocation.

The reviewer demonstrated both halves on the plain CPU EP:

```
after run 2: input buffer borrowed=true ptr=0x7d6f90020780 (t1_ptr=0x7d6f90020780)
panicked … SOUNDNESS: input buffer is a borrowed alias to freed caller memory
```

and, with a run-2 input aligned to 4 but not 64 (forcing the `copy_from_host`
fallback so the dangling handle is *written*):

```
malloc(): unaligned fastbin chunk detected
… (signal: 6, SIGABRT: process abort signal)
```

**Why my guard missed it.** `unbind_borrowed_inputs`'s second arm was written
for exactly this ("a sequence op re-roots a value's storage") and never ran:
the handle the promotion installs, `storage.alias()`, is *itself* borrowed, so
the first arm fired and the escaped copy inside `shared_buffers` was never
examined. The lesson is that guarding the *slot* is not enough when a handle
can be moved out of it.

**The fix** (`37a6f7f43`) is the reviewer's suggested localized one: sequence
storage must own its bytes, so `read_seq_element` copies a borrowed buffer into
a fresh owned allocation before wrapping it. The run's alias stays installed
and is unbound normally. This also covers the view-source case, since the copy
is keyed on the resolved `root`.

**Regression test.** `sequence_promotion_never_retains_a_borrowed_input_alias`
builds `input -> SequenceConstruct -> SequenceAt -> output`, runs it twice with
two different caller tensors, and asserts after each run that the input buffer
is neither borrowed nor equal to the dropped tensor's address, and that
`parked_input_buffers` is empty. **Falsified**: changing the `is_borrowed`
branch condition to `false` fails it on run 2 —

```
panicked at crates/onnx-runtime-session/src/executor/tests.rs:6285:5:
assertion failed: !executor.buffers[&vid].is_borrowed()
```

### Items the reviewer checked and found sound
* compute-in-place `!b.is_borrowed()` guard — correct and necessary; the 2→1
  alias-count change documents a genuine trade, not a hidden regression.
* `try_move_host_output` — double-guarded (`producer.is_none()` already rejects
  producer-less graph inputs, plus `!buf.is_borrowed()`).
* `ensure_buffer` cross-run consistency — `buffer_shapes` tracks the owned
  buffer and the owned buffer is always restored before the next sizing pass,
  so a borrowed handle can never be deallocated there.
* The `stage2_excluded.is_none()` gate — sufficient; `DecodeViewPlan` stores
  `ValueId`s and compares raw pointers for equality without dereferencing them,
  and replay steps never borrow.
* Aliasing — two inputs bound to the same tensor, or an input equal to a
  previously-returned output, are read-only aliases; child control-flow
  executors are separate `Executor`s reading materialized captures.

`cargo test -p onnx-runtime-session --lib` — **169 passed**, 0 failed
(one more than before: the new falsifier). `fmt`/`clippy` clean.
