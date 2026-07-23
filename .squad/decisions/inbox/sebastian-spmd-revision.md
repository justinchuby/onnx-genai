# Persistent SPMD revision after Chew/Gaff review

**Author:** Sebastian (independent revision; Pris locked out)  
**Branch:** `perf/decode-barrier`  
**Date:** 2026-07-22  
**Base artifact:** `eb19730`

## Review fixes

- **B1 — real parity regression:** added a subprocess-isolated ON/OFF test that
  runs six sequential packed-int4 M=1 `MatMulNBits` kernels. Both children use
  31 workers; the ON child removes `ONNX_GENAI_CPU_DECODE_AFFINITY`, sets
  `ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1`, asserts `pools().is_some()`, and
  asserts all six ops entered SPMD dispatch. The full per-op f32 byte stream is
  identical to the flag-OFF child.
- **B2 — inspectable precedence:** `numa-split` remains the intended winner when
  both flags are set and its layout builds. The runtime logs the mutually
  exclusive choice once. If `numa-split` is unavailable, it logs that persistent
  SPMD is active instead; if neither can build, that is also reported. Module and
  NUMA-plan documentation now state this precedence.
- **NB1 — panic-safe barrier:** each worker creates an unwind-only completion
  guard. Normal completion performs the same `fetch_sub` as before and
  `mem::forget`s the guard. On unwind, `Drop` first records the poisoned worker,
  then decrements the pending counter, so the dispatcher reaches the barrier,
  reports an actionable panic, and rejects later dispatches instead of hanging.
  A regression test deliberately panics worker 2 and verifies the diagnostic.
- **NB2 — accurate fallback wording:** single-node/non-NUMA/no-pinning operation
  is documented as a single unpinned SPMD group, not the bounded Rayon pool.
- **NB3 — redundant transmute:** replaced the trait-object/lifetime transmute
  with a thin erased data pointer plus monomorphized call trampoline. No
  `transmute` remains.

## Performance

Qwen2.5-Coder-7B int4, 32 workers, `profile_native --steady --decode-skip 8
--tokens 96`, five interleaved runs, prompt `Hello`.

Panic-safety before/after, persistent SPMD in both binaries:

| Round | `eb19730` | revised |
| --- | ---: | ---: |
| 1 | 16.96 | 18.58 |
| 2 | 16.75 | 17.99 |
| 3 | 16.90 | 17.51 |
| 4 | 17.92 | 17.77 |
| 5 | 15.49 | 18.75 |
| **median** | **16.90** | **17.99** |

No hot-loop regression was measurable; the revised median was 6.4% higher on
the noisy shared host. The normal guard path adds no atomic operation beyond the
pre-existing completion decrement.

Re-confirmed revised strategy A/B:

| Round | `numa-split` | persistent SPMD |
| --- | ---: | ---: |
| 1 | 16.21 | 18.02 |
| 2 | 15.18 | 17.16 |
| 3 | 14.67 | 16.91 |
| 4 | 17.32 | 17.89 |
| 5 | 16.65 | 17.28 |
| **median** | **16.21** | **17.28** |

Persistent won 5/5: +6.6% by median (approximately 7%) and +9.0% by mean.
Generated token IDs were byte-identical in every run, beginning
`[48298, 271, 9707, 0, 2585, 646, 358, 7789, 498, 3351, 30, ...]`.

## Validation

- CPU EP suite: 698 unit tests + 10 numeric regressions passed.
- Clippy with `--features mlas -- -D warnings`: clean.
- SPMD stress: 30/30 runs passed; 11 SPMD tests per run, no hang/failure.
- Both-flags runtime probe emitted the documented once-only `numa-split is
  active because it has precedence` diagnostic.
