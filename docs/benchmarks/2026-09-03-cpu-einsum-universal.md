# Universal CPU Einsum benchmark validation

## Conditions

- Commit: `217a98a8f55cdedc186c792fae5b4f49160fc9c2`
- Host: Intel Xeon Platinum 8480C
- Affinity: four physical cores (`0-3`)
- CPU thread budget: 4
- Rust: 1.98.0
- Host lock: box-wide configured lock, owner `resch`, strict stale-lock refusal
- Harness: 26 selectors covering optimized and forced GenericNative arms for
  bilinear, trilinear, and eight-operand paths, plus zero-copy and equivalent
  MatMul controls

The governed selector census passed `26/26`. Every validation arm used the same
materialized inputs and independent oracle, and observed the intended native
route before timing.

## Result

**Unmeasured.** The full evidence sweep was rejected by the harness during the
MatMul A/A null control because `foreign_pct` rose to 21.1% after the first
clean arm. The run held the box-wide lock, so this identifies activity outside
the lock protocol rather than a competing declared benchmark.

No latency, throughput, or speedup from that rejected window is reported.
