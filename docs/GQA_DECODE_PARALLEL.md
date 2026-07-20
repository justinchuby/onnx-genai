# CPU GQA decode parallelism

## Scheme

`GroupQueryAttention` flattens the independent `(batch, query_head,
query_sequence)` rows in BHSD order and uses one Rayon `par_chunks_mut` region.
During M=1 CPU decode this runs on the resident bounded decode pool; prefill uses
the normal Rayon pool. Each task keeps the original fixed-order score, softmax,
and value reductions.

A 163,840 row × key × head-dimension work threshold keeps small calls serial.
On Qwen2.5-0.5B (14 query heads, head size 64), this delays decode parallelism
until about 183 cached tokens. The threshold also naturally enables parallel
prefill when `query_sequence` supplies enough independent rows.

## Soundness and numerical parity

`par_chunks_mut(head_size)` partitions `y_bhsd` into non-overlapping output
rows. A task receives exactly one mutable row, derives its `(b, qh, qs)` index,
and never writes outside that slice. `q`, `present_k`, `present_v`, and sequence
metadata are shared read-only. Every task allocates its own `scores: Vec<f32>`;
there is no shared scratch.

The serial BHSD-to-BSH copy remains after the join. Within each row, dot
products, max, f64 `exp` followed by f32 rounding, softmax sum, and value
accumulation execute in the same order as the serial oracle. Only independent
rows are reordered.

The 512-token baseline and parallel runs produced identical greedy token ID
lists. The first generated token remained `12095` (`Paris`). The new
`large_prefill_parallel_path_matches_reference` test forces the parallel path
and compares it with the existing independent GQA reference.

## Benchmarks

Host: Intel Xeon Platinum 8480C. Model:
`/home/justinchu/qwen2.5-0.5b-int4-onnx`. Prompt:
`The capital of France is`. Default resident decode pool: 8 workers.

Baseline and candidate release binaries were run in counterbalanced order:

| Decode length | Runs | Serial | Parallel | Delta |
|---|---:|---:|---:|---:|
| 128 tokens | 6 paired | 22.97 tok/s, 43.57 ms/token | 23.60 tok/s, 42.41 ms/token | +2.8% |
| 512 tokens | 4 paired | 18.03 tok/s, 55.48 ms/token | 19.58 tok/s, 51.16 ms/token | +8.6% |

Operator profiling over 320 generated tokens reduced aggregate GQA time from
26.991 to 23.235 ms/token (-13.9%). At tokens 257–320, where the parallel path
is continuously active, GQA fell from 31.144 to 24.077 ms/token (-22.7%).
The short 16-token profile remained effectively unchanged: 20.808 to
20.462 ms/token.

For a 225-token prompt, paired prefill profiles reduced GQA from a mean
1444.3 ms to 168.6 ms (-88.3%) and total node execution from 5014.8 ms to
3740.6 ms (-25.4%). This confirms that M>1 also benefits.

`ONNX_GENAI_CPU_DECODE_THREADS=0` remains functional: Rayon uses its global
pool, and the 225-token prompt completed successfully with generated token
`576`.
