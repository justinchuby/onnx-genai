---
title: Performance Engineering Playbook
aliases:
  - Performance Measurement Discipline
  - 性能工程手册
tags:
  - performance
  - benchmarking
  - profiling
  - correctness
status: maintained
created: 2026-08-17
updated: 2026-08-17
---

# Performance Engineering Playbook

> [!summary] Question answered
> How do we tell whether a performance number describes the runtime, the
> experiment, or a measurement mistake?

A benchmark is not just a number. It is a claim about a particular model,
artifact, configuration, machine and workload. Before interpreting the result,
we must establish the conditions that produced it.

## The evidence ladder

Use evidence in this order:

1. **Correctness output** — token IDs and required runtime invariants.
2. **Deterministic counters** — bytes, pages, captures, fallbacks and accounting.
3. **Synchronized timings** — measurements whose start and stop boundaries are
   understood.
4. **Profiler attribution** — operator, kernel and device-level evidence.
5. **Wall-clock throughput** — repeated under controlled conditions.

Higher levels are easier to interpret. A throughput gain cannot compensate for
wrong tokens, disabled graph capture or unaccounted memory.

## Start with the exact claim

Write the claim before running the experiment:

```text
On <platform>, for <model artifact>, at <context/batch/budget>,
change X should reduce counter Y without changing invariants Z.
```

This exposes missing controls. “CUDA is faster” is not testable; “fused GEMV
reduces issued instructions for batch-1 decode while preserving token IDs and
capture” is.

Every reported result should include:

| Condition | Examples |
|---|---|
| Artifact | model revision, quantization, external-data files |
| Runtime | commit, feature flags, provider path |
| Workload | prompt, batch, context, warmup, generated tokens |
| Hardware | device, OS/driver mode, available memory |
| Isolation | solo run, device idle before every sample |
| Statistics | sample count, median and full range |

## Check whether the experiment forced the result

A configured knob may directly determine the metric later interpreted as a
runtime discovery. For example, setting a minimum KV bucket equal to maximum
capacity makes `committed_len == capacity` true even if the normal runtime grows
on demand.

Ask:

> If the system did nothing interesting, what value would this knob alone
> produce?

The selected constructor or entry point can also be a hidden knob. A conservative
default such as “stable address unavailable” may mean the harness never supplied
the production configuration—not that the hardware rejected the feature.

Always print the inputs to a decline predicate, not only the final decline
reason. Build at least one control that could falsify the claim.

## Check what the instrument actually brackets

GPU operations are commonly asynchronous:

```text
host starts timer
host enqueues copy
enqueue returns
host stops timer
GPU finishes copy later
```

That timer measures submission latency, not transfer completion. Divide bytes by
the measured time: an impossible bandwidth is an immediate warning that the
counter is mislabeled or unsynchronized.

For every timing counter, document:

- the exact start and stop points;
- whether work is asynchronous;
- which event, fence or synchronization makes completion observable;
- whether profiler overhead changes the path being measured.

See [[observability/Tracing and Profiling]] for the tracing architecture and
available collectors.

## Require the arithmetic to close

Compute a bound the result must obey. If measured traffic falls below the
theoretical minimum required to read the weights, suspect the accounting before
claiming a new optimization.

Likewise, avoid rates over populations whose members have very different costs.
A cache can report many small hits while repeatedly missing multi-megabyte
weights:

```text
count hit rate: looks healthy
byte hit rate: shows the real cost
```

Choose the unit that matches what the machine pays: bytes, pages, instructions
or synchronized time.

## Control contention and unstable wall-clock

Shared GPU activity can turn the same configuration into both false regressions
and false wins. Verify the device is idle **before every sample**, not once
before a long loop.

On systems where the OS can page managed GPU memory, throughput may vary widely
without a code change. Lead with process-local deterministic counters and treat
wall-clock as supporting evidence:

- use at least three samples;
- report the median and full range;
- keep the counters beside the throughput;
- never rerun selectively until the preferred result appears.

## Measure the real workload

A convenient proxy is useful only when its differences are explicit. Sequential
device copies do not reproduce strided int4 GEMV access; a small tensor may fit
in cache while the real working set does not.

State what the proxy omits, sweep sizes when caching can change the outcome, and
leave the gap unknown until the real path is measured. For backend-specific
examples, see [[execution/CPU Execution Provider]] and
[[execution/CUDA Execution Provider]].

## Prove that the optimized path ran

An accelerated implementation without a reachability test can remain unwired
while every unit test passes. A complete performance change needs:

1. a correctness test for the implementation;
2. a test or counter proving dispatch reaches it;
3. a fallback test for unsupported devices or shapes;
4. an end-to-end measurement on the intended workload.

Tests must construct their prerequisites. Do not depend on an allocator
happening to reuse an address or a machine happening to expose a feature; assert
non-vacuously that the required condition held.

## Correctness constraints outrank speed

At minimum, relevant changes should preserve:

- byte-identical token IDs for the same prompt and configuration;
- graph capture expectations such as `captures > 0` and `fallbacks == 0`;
- memory bounds and zero oversubscription;
- zero reference, byte and accounting underflows;
- safe destruction—especially no panic/assertion inside `Drop`.

A faster run that silently generates fewer or different tokens is a correctness
failure, not a performance win.

## Reporting a result

Report:

1. the claim and conditions;
2. correctness and deterministic counters;
3. timing methodology and samples;
4. the mechanism supported by profiler evidence;
5. the ceiling on the possible gain;
6. negative results and unresolved uncertainty.

Truthful negative results prevent repeated work. A small favourable result from
one budget should not automatically become a global policy. Correct superseded
numbers where they originally live so later readers do not quote them as fact.

## Related notes

- [[observability/Tracing and Profiling]]
- [[execution/Execution Backends]]
- [[execution/CPU Execution Provider]]
- [[execution/CUDA Execution Provider]]
- [[contracts/Runtime Contracts]]

## Formal sources

- [Measurement discipline skill](../../.github/skills/measurement-discipline/SKILL.md)
- [Kernel performance guide](../../docs/performance/KERNEL_PERF.md)
- [Benchmark artifacts](../../docs/benchmarks/)
- [Research notes](../../docs/research/)
