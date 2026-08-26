# Generalized media batching fixture benchmark

Scope: deterministic tiny ONNX fixtures only. These numbers are evidence that
one grouped ORT invocation reduces dispatch overhead for the tested shapes; they
are not production-model or hardware-general performance claims.

## Method

- Host: Intel Xeon Platinum 8480C, Linux 6.6, x86-64.
- Backend: ONNX Runtime 1.29 CPU, one intra-op thread.
- Build: `cargo test` debug profile.
- Warmup: 8 samples; measurement: 31 samples; reported statistic: p50.
- Preprocessing and tensor materialization occur before timing.
- Grouping/admission dispatch is timed separately from ORT invocation.
- Per-item mode performs one ORT `Run` per physical image/frame. Grouped mode
  performs one `Run` over the packed axis.
- No pass/fail latency threshold is asserted.

Command:

```bash
cargo test -p onnx-genai-engine --test media_batching_e2e \
  fixture_benchmark_reports_grouping_dispatch_and_ort_compute_separately \
  -- --nocapture
```

## Measurements

| Fixture | Items | Grouping dispatch p50 | Per-item ORT p50 | Grouped ORT p50 | Per-item throughput | Grouped throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Image `[3, 8, 3]` | 3 | 216.847 µs | 34.808 µs | 12.486 µs | 86,187.1 items/s | 240,269.1 items/s |
| Nested video `[5, 8, 3]`, 4 clips | 5 | 506.264 µs | 64.431 µs | 13.430 µs | 77,602.4 frames/s | 372,300.8 frames/s |

The benchmark test also runs the same nontrivial arithmetic graph used by the
E2E correctness fixtures. Variance across hosts and builds is expected.
