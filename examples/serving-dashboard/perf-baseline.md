# AC33 — Clean-Tree Decode Throughput Baseline

**Author:** QA Tester (@fc8b5d97)
**Date:** 2026-07-29 22:46 – 23:07 PDT (2026-07-30 05:46 – 06:07 UTC)
**Status:** ✅ COMPLETE — baseline captured, server stopped, tree unmodified by me
**Purpose:** Reference measurement for the acceptance criterion *"telemetry adds <2% decode overhead."*
This measurement is **unrecoverable** once instrumentation lands.

---

## 0. READ THIS FIRST — the headline finding

> **The acceptance criterion "<2% decode overhead" is NOT testable with a naive
> before/after comparison at the originally-planned workload.**

I measured the noise floor empirically. Two runs of the **identical, unmodified binary**,
back-to-back, at 128-token generations, differed by:

| | rep 1 (n=15) | rep 2 (n=15) | delta |
|---|---|---|---|
| median decode | 33.925 tok/s | 33.452 tok/s | **−1.40 %** |
| mean decode | 33.547 tok/s | 32.882 tok/s | **−1.98 %** |

Welch t = 1.108 → statistically indistinguishable, exactly as it must be for the same
binary. **But a −1.98 % apparent "regression" appeared from pure noise.** Anyone who runs
128-token generations before and after the telemetry PR and compares means has a coin-flip
chance of manufacturing a phantom 2 % regression — or of hiding a real one.

**Fix (validated, §5):** measure with **512-token** generations. This cuts CV from 4.95 %
to 1.98 % and drops the sample count needed to resolve 2 % from **97 per arm to 13 per arm**.
The primary baseline below is captured at 512 tokens for exactly this reason.

**A negative result worth knowing:** I also tested an interleaved paired A/B design (the
usual remedy for drift). **It does not help here** — the noise is high-frequency
per-request jitter, not slow drift, so pairing adjacent samples roughly doubles the
variance of the difference. Null A/B with 12 pairs gave mean delta −1.23 % with a 95 % CI
of [−5.68 %, +3.23 %]. Do **not** waste time on pairing; increase generation length instead.

---

## 1. PRIMARY BASELINE (reference numbers — see §5.0 before comparing against these)

> ⚠️ **Do not use these as the comparison baseline for the acceptance test.** The clean
> binary has been preserved; §5.1 requires a back-to-back A/B against it. I measured the
> byte-identical binary reading **9.8 % lower** 75 minutes later purely from machine load.
> These numbers are a sanity reference, not the gate.

Workload: `max_tokens=512`, `temperature=0`, greedy, streaming SSE, single model instance.
Decode rate excludes prefill (see §3).

### 1.1 Single-request decode throughput — **THE headline number**

| statistic | value (tok/s) |
|---|---|
| **median** | **33.415** |
| mean | 33.576 |
| min | 32.233 |
| p25 | 33.113 |
| p75 | 34.102 |
| max | 34.651 |
| stdev | 0.665 |
| **CV** | **1.98 %** |
| IQR | 0.989 |
| n | 15 (+1 warmup discarded) |

Raw (tok/s), in order:
```
34.177 32.889 33.415 33.161 33.405 32.233 33.004 33.887
33.243 34.196 33.900 33.071 34.393 34.651 34.031
```

### 1.2 Four-concurrent decode throughput (max_batch=4)

| statistic | aggregate decode (tok/s) | wall-clock throughput (tok/s) |
|---|---|---|
| **median** | **82.130** | **52.506** |
| mean | 82.847 | 52.814 |
| min | 80.997 | 51.951 |
| max | 86.131 | 54.292 |
| stdev | 2.425 | 1.020 |
| CV | 2.93 % | **1.93 %** |
| n | 4 rounds (+1 warmup discarded) | 4 rounds |

Per-round raw:

| round | aggregate tok/s | per-stream tok/s | wall s | wall tput tok/s |
|---|---|---|---|---|
| 1 | 81.03 | 20.26 | 29.9 | 51.95 |
| 2 | 81.00 | 20.25 | 29.6 | 52.55 |
| 3 | 83.23 | 20.81 | 29.6 | 52.46 |
| 4 | 86.13 | 21.53 | 28.6 | 54.29 |

> ### ⛔ WITHDRAWN VERDICT — THE RATIO BELOW IS NOT CITABLE. THE SAMPLES ABOVE ARE.
>
> Commit `2d6b36ac` withdrew the throughput ratio from every **shipping** document
> (`PR-DESCRIPTION.md`, `QA-PLAN.md`, `README.md`, `check-perf-claims.test.js`).
> **It did not touch this file, which is where the ratio is derived.** So the
> withdrawal was applied to every document that *quoted* the number and not to the
> one that *produces* it — and a reviewer checking the figure travels toward the
> origin, so the more carefully they read, the more certainly they landed here.
> Found while attempting to source a `2.59×` attributed to QA that **does not
> exist**: `2.59` is the *upper confidence bound* of this ratio as restated in
> `REVIEWER-BRIEF.md:580`, read back as a point estimate.
>
> **Live sites of the withdrawn ratio in this file: 2** — the two sentences
> immediately below this notice. The count is stated so this notice is falsifiable
> by `grep`, not merely trusted — a notice that asserts its own completeness is a
> status column, and mine has already been wrong once tonight
> (`prefix-cache-verification.md`, reported as one site by @732c7548, actually four).
> ⚠️ **This notice cannot verify itself by quoting anything, and I proved that the
> hard way — three times.** First I cited line numbers; they were stale within one
> edit. Then I quoted the digits; the notice counted itself. Then I quoted the
> verdict's own wording as a safer anchor; **the notice counted itself again.**
> **A verification instruction written *inside* the document it verifies becomes an
> instance of the thing it counts. There is no string I can print here that is
> exempt from my own search.**
>
> **So the anchor is structural, not textual: the withdrawn verdict is the single
> paragraph beginning at the only line matching `^\*\*Batching speedup`.**
> Everything quoted in this notice is blockquoted — every line carries a `>` prefix
> — so it is **structurally incapable** of matching a start-of-line anchor. The
> check excludes the checker by construction rather than by my remembering to.
> ✅ `grep -cE '^\*\*Batching speedup'` returns **1**.
>
> The count is stated so this notice is falsifiable rather than trusted: a notice
> that asserts its own completeness is a status column, and mine was already wrong
> once tonight (`prefix-cache-verification.md` — reported as one site by
> @732c7548, actually four). ⚠️ **Also note the raw per-round table contains a
> wall-throughput sample whose digits collide as a substring with the ratio. It is
> a measurement, not a citation, and must NOT be struck.**
>
> **Why re-running cannot fix it:** the withdrawal reason is the *model*, not the
> harness, the load, or the box. That artifact was assembled by accident from two
> builds seventeen days apart and its inference metadata was edited fifty-four
> minutes after the build — **inside the measurement window**. The figure is not
> merely unreproducible by a reader; we cannot show it is internally consistent
> with itself. **A fresh run with a perfect harness on a silent box reproduces a
> number that is still unvalidatable — and would be the most persuasive possible
> form of laundering it back into the record.**
>
> **What survives:** every raw sample above, and the *mechanism* — the startup line
> `continuous batch driver enabled max_batch=4` and the independently observed
> 4-concurrent batch occupancy. **A count needs no throughput arithmetic, so it
> does not depend on the model's provenance at all.** Ship the mechanism.

**Batching speedup: 82.13 / 33.41 = 2.46×** aggregate decode at batch 4.
Per-stream decode degrades to ~20.7 tok/s (0.62× of solo), i.e. batching trades
per-stream latency for 2.46× aggregate throughput. Continuous batching is genuinely
engaging — confirmed by the startup log line
`continuous batch driver enabled max_batch=4`.

*(The two sentences above are the withdrawn verdict, retained verbatim under the
notice rather than deleted: striking the words that were actually published would
destroy the evidence that the claim was made, which is the failure mode this
document exists to record. The raw tables are untouched and remain citable.)*

> **Recommended acceptance metric:** `wall_throughput_tps` for the concurrent case
> (CV 1.93 %) and single-request median decode (CV 1.98 %). Avoid
> `aggregate_decode_tps` as the gate — it is the noisiest of the three (CV 2.93 %).

---

## 2. SECONDARY BASELINE (128-token workload, for cross-validation only)

Captured first, before I discovered the variance problem. Retained because it
cross-validates the primary numbers and documents the noise floor.

| metric | median | mean | stdev | CV | n |
|---|---|---|---|---|---|
| single decode tok/s (rep 1) | 33.925 | 33.547 | 1.643 | 4.90 % | 15 |
| single decode tok/s (rep 2) | 33.452 | 32.882 | 1.647 | 5.01 % | 15 |
| single decode, pooled | 33.788 | 33.214 | 1.651 | 4.97 % | 30 |
| single TTFT ms | 2141.3 | 2139.6 | 137.5 | 6.42 % | 15 |
| 4-conc aggregate tok/s | 83.534 | 81.834 | 7.121 | 8.70 % | 8 |
| 4-conc wall tput tok/s | 33.139 | 33.237 | 0.825 | 2.48 % | 8 |

**Cross-validation:** single-request decode is 33.788 tok/s @128 vs 33.415 tok/s @512
(1.1 % apart); 4-concurrent aggregate is 83.53 @128 vs 82.13 @512 (1.7 % apart). The two
independently-captured workloads agree, which is good evidence the measurement is sound
and not an artifact of generation length.

**Prefill (TTFT) baseline:** median **2141 ms**, mean 2140 ms, stdev 137 ms (n=15) for the
standard prompt. Note prefill is ~2.1 s vs ~3.8 s of decode for 128 tokens — prefill is a
large fraction of e2e at short lengths, which is precisely why it must be excluded from a
decode-overhead measurement.

---

## 3. Methodology

### What "decode throughput" means here
The server is driven over **streaming SSE** and every content delta is timestamped at
arrival. Then:

```
decode_tps = (n_tokens - 1) / (t_last_token - t_first_token)
```

The interval starts at the **first** token, so the prefill/TTFT phase is excluded by
construction. This matters: `total_tokens / e2e_time` would blend prefill into the number
and dilute any decode-path regression by roughly 35 % at 128 tokens, making the 2 % gate
even harder to trip. **A later re-run must use this same decode-only definition** or the
comparison is meaningless.

### Controls applied
- `temperature = 0` (greedy) → deterministic token stream, identical work every iteration.
- `max_tokens` fixed and **every single iteration returned `finish_reason: "length"` with
  exactly the requested token count** (128/128 or 512/512, verified on all 60+ requests).
  No early EOS, so every iteration did exactly equal work. This is a hard guarantee, not
  an assumption.
- Warmup iterations executed and **discarded** (3 for the 128 phases, 1 for each 512 phase).
- One server process for the entire session — no reload/JIT/page-cache confounds mid-run.
- Concurrent phase uses **distinct prompts per stream** (`variant 0..3`) so nothing can be
  served from a shared cache.
- ORT thread pool is derived from P-core count (`ThreadPoolRecommendation::from_topology`,
  `crates/onnx-runtime-cpuinfo/src/lib.rs:445`) and is therefore **fixed at 8 on this M1 Max
  with no env override needed** — one less source of run-to-run variation.

### Statistics
Median and IQR are reported as primary (robust to the outliers this noisy machine
produces — e.g. the 128-token concurrent round 2 at 64.6 vs ~83 tok/s). Mean/stdev/CV are
reported for power analysis. Power computed as
`n = 2(z_{0.975}+z_{0.80})^2 σ² / δ²` (two-sample, 80 % power, α=0.05).

---

## 4. EXACT REPRODUCTION RECIPE

### Provenance of this baseline

| item | value |
|---|---|
| Worktree | `/Users/justinc/Documents/GitHub/onnx-genai-demo` |
| Branch | `feat/genai-demo-dashboard` |
| Git SHA at start | `f55e459b7dd6862deab8407c5c81eee1796cb92a` |
| Git SHA at end | `9f3f5d9419dae2810fff6d3adaba8147bd5cfbfb` |
| **Rust/Cargo files changed between them** | **NONE — verified, see §6** |
| Server binary SHA-256 | `d49d3c8fe1b8a98e1a06720870e30524a8ac970192e3b08e99661a40e1c31ec7` |
| Server binary mtime | 2026-07-29 22:30 (predates all crew commits in the window) |
| Binary size | 29,033,360 bytes |
| rustc | 1.97.0 (2d8144b78 2026-07-07) |
| ONNX Runtime | 1.27.0 (API 27), from conda miniconda base |
| Model | `/Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-scatter-v2` |
| `model.onnx` SHA-256 | `d3de664691405f4b636579de5c49f119d2baa1f4fee5488ab3a56b072cd0d8c4` |
| EP | `ONNX_GENAI_EP=cpu` |
| Machine | Apple M1 Max, 10 cores (8P/2E), 32 GiB |
| OS | macOS (Darwin), uptime 3 d 16 h |
| Batch driver | `continuous batch driver enabled max_batch=4` (hardcoded `DEFAULT_MAX_BATCH`, `src/state.rs:25`) |

### Deployment shape — SINGLE-SERVER (explicit, do not assume)

This baseline was captured against **one server, one model, nothing else resident**. Stated
explicitly because it is otherwise an invisible assumption that would silently break the
later comparison:

| condition | value |
|---|---|
| server processes | **exactly 1** |
| models loaded | **exactly 1** (`qwen-scatter`) |
| model dir | `/Users/justinc/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b-scatter-v2` |
| bind address | `127.0.0.1:8123` |
| second server / router / dynamic-KV server | **none running** |
| dashboard | **not running** for §1/§2; running at 4 Hz for the §6b.1 arm only |
| debug endpoints | enabled (`--enable-debug-endpoints`) |
| admin endpoints | disabled |

**Why this matters now:** the demo was re-scoped to a **two-server** shape (a scatter server
for continuous batching plus a dynamic server for paged KV + prefix caching, side by side)
*after* this baseline was captured. See §4b.

### 4b. The two-server demo change — does NOT invalidate this baseline

**Assessment: the baseline stands, and the acceptance measurement should stay single-server.**

1. **The back-to-back protocol absorbs it.** Because the clean binary is preserved (§5.0),
   both arms run in the *same session under the same ambient conditions*. Whatever else is
   resident — a second server, other agents' builds, Time Machine — perturbs both arms
   equally and cancels in the delta. This is exactly the property preserving the binary
   bought us, and it is why the re-scope is not a problem.

2. **The AC measurement should isolate ONE server.** AC33 asks whether *telemetry* adds
   <2 % decode overhead. Measuring the full two-server demo would conflate telemetry cost
   with **deployment resource contention** — a second server competing for the same CPU is
   a large effect (§5.0 shows ambient load alone moves decode by 9.8 %) and would swamp the
   2 % signal, almost certainly yielding INCONCLUSIVE under §5.3.

   > **Recommendation:** run the acceptance A/B **single-server**, apples-to-apples with
   > this baseline, with telemetry active and a dashboard polling that one server at 4 Hz.
   > If a deployment-cost number for the two-server demo is wanted, measure it as a
   > **separate** exercise and do not let it contaminate the <2 % gate.

3. **If the AC is ever re-scoped to the two-server shape**, this baseline is not the right
   BEFORE arm — but the fix is cheap and available: re-run the preserved clean binary in
   the two-server configuration to generate a matched two-server BEFORE. That option exists
   *only* because the binary was preserved.

---

### 4c. Independently verifiable git state at measurement time

Requested by @d7cf9b84 so a reviewer can audit the A/B rather than take it on assertion.
Verbatim, as captured:

**At baseline start (22:45, before any measurement):**
```
$ git rev-parse HEAD
f55e459b7dd6862deab8407c5c81eee1796cb92a
$ git branch --show-current
feat/genai-demo-dashboard
$ git status --short
(no output — tree completely clean)
```

**At baseline end (23:05):**
```
$ git rev-parse HEAD
9f3f5d9419dae2810fff6d3adaba8147bd5cfbfb
$ git status --short
 M examples/serving-dashboard/CONTRACT.md
 M examples/serving-dashboard/telemetry-field.js
 M examples/serving-dashboard/telemetry-store.js
 M examples/serving-dashboard/telemetry-store.test.js
?? examples/serving-dashboard/app.js
?? examples/serving-dashboard/css/
?? examples/serving-dashboard/dashboard/
?? examples/serving-dashboard/index.html
?? examples/serving-dashboard/ui/
```

**Audit of that drift — why it does not invalidate the numbers:**
```
$ git diff --name-only f55e459b HEAD -- '*.rs' 'Cargo.toml' 'Cargo.lock'
(no output — zero Rust or Cargo files changed)
$ git status --porcelain -- '*.rs'
(no output — zero uncommitted Rust)
```
All drift was frontend JS/CSS/HTML/MD. The measured binary (`d49d3c8f…`, mtime 22:30)
predates every one of those commits, and is now preserved (§5.0) so this is checkable
forever rather than asserted.

> **⚠️ Standing hazard for anyone re-running this.** Builds happen **from the shared
> worktree**. If in-progress Rust is sitting in `crates/` when someone rebuilds, it lands in
> the supposedly-"clean" binary **with no cargo command from its author at all**, and
> telemetry overhead would measure ~0 % for entirely the wrong reason — a **false PASS**.
> @d7cf9b84 is correctly staging their Rust as drafts *outside* the worktree; anyone
> touching Rust before sign-off must do the same.
>
> **Fully mitigated for the BEFORE arm:** use the preserved binary (§5.0) and do not rebuild
> it. That is precisely why it was preserved.

### 4d. Commands (copy-paste)

> To reproduce the **exact** BEFORE arm, skip `cargo build` and run the preserved binary
> `clean-binary/onnx-genai-server-clean-d49d3c8` instead — see §5.0. Building from source
> later will *not* reproduce this binary once the telemetry changes land.

```bash
cd /Users/justinc/Documents/GitHub/onnx-genai-demo
export CARGO_TARGET_DIR=/Users/justinc/Documents/GitHub/onnx-genai/target

cargo build --release -p onnx-genai-server

ONNX_GENAI_EP=cpu $CARGO_TARGET_DIR/release/onnx-genai-server \
  --model ../onnx-genai/models/qwen2.5-0.5b-scatter-v2 --model-id qwen-scatter \
  --addr 127.0.0.1:8123 --enable-debug-endpoints
```

Confirm this line appears in the log before measuring — **if it is absent the run is
invalid**, because the non-batched per-request path is a different code path entirely:
```
INFO onnx_genai_server::driver: continuous batch driver enabled max_batch=4
```

Then run the harness (archived at `harness/qa_baseline_harness.py`, next to this file):
```bash
python3 harness/qa_baseline_harness.py /tmp/rerun.json
```

### Fixed workload parameters — a re-run MUST match all of these

| parameter | value |
|---|---|
| endpoint | `POST /v1/chat/completions`, `stream: true` |
| model id | `qwen-scatter` |
| prompt | `Write a detailed explanation of how a hash table works, including collision resolution.` |
| concurrent prompts | same prompt + `" (variant N)"`, N = 0..3 |
| `max_tokens` | **512** (primary) / 128 (secondary) |
| `temperature` | `0.0` |
| concurrency | 1 (single) and 4 (concurrent) |
| warmup | ≥1 at 512, discarded |
| iterations | ≥15 single, ≥4 concurrent rounds |
| decode formula | `(n_tokens - 1) / (t_last - t_first)` |

---

## 5. WHAT A LATER RE-RUN MUST MATCH — the acceptance test

### 5.0 ⛔ MANDATORY PROTOCOL CHANGE — do NOT compare against the numbers in §1

**The clean-tree binary has been preserved.** This changes the correct protocol entirely,
and supersedes any instruction to "match conditions" against the numbers in §1.

```
clean-binary/onnx-genai-server-clean-d49d3c8
SHA-256 d49d3c8fe1b8a98e1a06720870e30524a8ac970192e3b08e99661a40e1c31ec7
```
Verified: byte-identical to the binary that produced §1, boots correctly, logs
`continuous batch driver enabled max_batch=4`, and produces coherent 512/512-token output.

**Why comparing against §1's numbers is invalid — measured, not theorised.** I re-ran the
*byte-identical preserved binary* ~75 minutes after the baseline, same machine, same model,
same prompt, same workload. Only the background load differed (crew now active: load avg
9.70, `mediaanalysisd` at 91 %, Time Machine copying):

| | median decode tok/s | machine load (1-min) |
|---|---|---|
| §1 baseline (22:46–23:07) | 33.415 | ~5–8 |
| identical binary re-run (23:22) | **30.151** | 9.70 |
| **difference** | **−9.8 %** | — |

**A −9.8 % swing from an identical binary — five times the entire 2 % acceptance
threshold.** Any protocol that compares a later "after" run against §1's absolute numbers
will be dominated by machine state and will produce a meaningless verdict.

### 5.1 THE REQUIRED PROTOCOL — back-to-back matched A/B

At telemetry sign-off, run **both binaries in the same session, on the same machine, within
the same few minutes**, alternating in blocks:

1. **BEFORE arm** — `clean-binary/onnx-genai-server-clean-d49d3c8`
2. **AFTER arm** — the telemetry build
3. Alternate in **blocks of 5** (`A A A A A / B B B B B / A A A A A / …`) until ≥15 per arm.
   Blocks, not per-request alternation — see the §0 null-A/B result.
4. **Both arms** run with the dashboard polling at **4 Hz with a scenario running**
   (the amended AC33 condition). Both arms — not just the after arm. See §5.2.
5. Report **median, spread (IQR/CV), n, and the 95 % CI of the delta** per arm, plus raw
   per-run numbers.

This design cancels machine state, thermal drift, and background load, because both arms
experience them equally. It is the only design that can resolve 2 % on this hardware.

> The numbers in §1/§2 remain valuable as a **sanity reference** — if the BEFORE arm at
> sign-off reads wildly different from 33.4 tok/s, the environment has changed in some way
> that needs explaining before anyone trusts the comparison. They are a cross-check, **not
> the comparison baseline**.

### 5.2 ⚠️ Arm-matching trap in the amended criterion

The amendment says *"same workload in both halves"* but specifies 4 Hz polling only for the
**after** run. Taken literally that is not a matched A/B: it would compare
`no-polling BEFORE` against `polling AFTER`, folding the polling load into the measured
"telemetry overhead."

**Both arms must poll at 4 Hz.** The clean-tree polling arm in §6b.1 exists for exactly
this reason: if a back-to-back run is somehow impossible, the correct BEFORE reference is
the **POLL-ON** figure (**median 32.759 tok/s**, n=8), *not* the no-polling 33.415 in §1.

Good news from §6b.1: polling costs no measurable decode throughput on the clean tree
(+1.45 % median, SEM ±1.53 %), so this correction is small — but getting it backwards
biases the result in the *permissive* direction, which is the dangerous direction.

### 5.3 Pass / fail thresholds and required sample size

**Gate:** the AFTER arm's median decode throughput must be **≥ 98 %** of the BEFORE arm's
median, both measured in the same back-to-back session under identical polling.

**Required n — this is not optional.**

| workload | measured CV | n per arm to resolve 2 % @80 % power |
|---|---|---|
| 512-token, no polling, quiet machine | 1.98 % | **13** |
| 512-token, 4 Hz polling, contended machine | 3.38 % | **45** |
| 128-token | 4.95 % | 97 ❌ |

Use **512-token generations**. Budget **≥15 per arm** on a quiesced machine, or **~45 per
arm** if the machine is busy. Quiescing the machine is far cheaper than collecting 45
samples per arm — see §6 P1.

**Inconclusive-result rule (binding, per amended AC33).** If the 95 % CI of the delta spans
±2 %, the result is **INCONCLUSIVE — NOT a pass**. Report it that way in plain language and
collect more samples or quiesce the machine. Do not round a noisy result down to "fine."
§0 is the reason this clause exists: two runs of the *same* binary differed by 1.98 %, and
§5.0 shows load alone moves the number 9.8 %. A bare point estimate near the threshold is
never a pass.

**Reporting requirement.** Publish median, IQR/CV, n, 95 % CI of the delta, raw per-run
numbers, and the full condition block from §4 — for **both** arms. Never a bare percentage.

---

---

## 6. THREATS TO VALIDITY — read before trusting a comparison

Ranked by how much they could distort a re-run.

**P1 — The machine was NOT quiescent.** This is the dominant error source.
- Load average rose from 5.22 at start to 8.04 (1-min) / 16.67 (5-min) by the end.
- A Time Machine backup was running at 68 % CPU with 79,572 changed items when I started.
  I stopped it with `tmutil stopbackup` (reversible; TM reschedules itself — I did **not**
  disable Time Machine). **It restarted before the end of the session** (`Running = 1` at
  23:05), so later phases ran with it active again.
- `mediaanalysisd` was observed at 99.5 % CPU, `mds_stores` at 17.6 %.
- Other crew agents were active throughout (VS Code helpers at 65 % and 52 % CPU).
- **Mitigation for the re-run:** before measuring, confirm `tmutil status` shows
  `Running = 0` and 1-min load average is comparable to this run. Record both. This is a
  CPU execution provider — every competing core directly steals decode throughput.

**P1 — Other agents were working during the measurement window** despite the freeze.
Git HEAD advanced `f55e459 → 9f3f5d9` mid-session (8 files, +2515 lines).
**Impact on validity: none for the binary, real for the timing.**
- Verified `git diff --name-only f55e459 HEAD -- '*.rs' 'Cargo.toml' 'Cargo.lock'` → **empty**.
- Verified `git status --porcelain -- '*.rs'` → **empty**. Zero uncommitted Rust.
- All changes were `examples/serving-dashboard/` JS/CSS/HTML/MD.
- The binary under test (mtime 22:30, SHA `d49d3c8…`) was built **before** any of it.
- **So the numbers are a valid clean-Rust baseline**, but they were taken under CPU
  contention and are therefore likely a mild *under*-estimate of quiet-machine throughput.
  Since the gate is a *relative* comparison, this is acceptable **provided the re-run is
  performed under similarly loaded conditions or, better, both are re-run quiet.**

**P2 — Thermal state.** M1 Max under sustained CPU inference will throttle. The session ran
~20 minutes continuously; later phases (the 512-token ones, i.e. the primary baseline) ran
warmer than early ones. A re-run starting from a cold machine may read slightly *high*.
Mitigation: run the warmup, and prefer comparing runs of similar duration and ordering.

**P2 — `DEFAULT_MAX_BATCH` is hardcoded** at `src/state.rs:25`; there is no CLI flag. If
anyone adds `--max-batch` (the architect flags this as likely, recon §4), the concurrent
baseline is invalidated unless the value is still 4. **Re-run must confirm `max_batch=4`
in the startup log.**

**P3 — Prefix cache is inert on this path (but NOT because of this reading).** `/v1/debug/kv`
after the full session read `prefix_cache_hits: 0, prefix_cache_lookups: 135`.
⚠️ **DO NOT CITE THIS RATIO AS EVIDENCE — BOTH NUMBERS ARE COMPILE-TIME CONSTANTS.**
`prefix_cache_hit_len` is a hardcoded literal `0` (`batched.rs:262`, `:486` — first arg of
`DecodeLoopState::with_rng`, per `decode_loop.rs:39-43`), so the `> 0` test at `metrics.rs:135`
can never be true; and `prefix_cache_lookups` increments unconditionally on every completed
generation (`metrics.rs:130-132`), so the `135` is **135 finished generations** and would read
135 with the prefix cache deleted from the source tree. **`0/135` is a literal divided by a
mislabelled counter; it can neither confirm nor deny anything about the cache.**
The conclusion is nonetheless correct — prefix reuse genuinely is bypassed on the
continuous-batch path — but it is established **by reading the source**
(`ContinuousBatchManager`, `batched.rs:101-110`, holds no `kv_cache` and no page table),
**not by this ratio.** *For this baseline the conclusion is actually
helpful* — it means repeated identical prompts got no caching advantage, so iterations are
comparable. But if someone *fixes* prefix caching before the telemetry PR, these numbers
become invalid and the baseline must be recaptured.

---

## 6b. ADDENDUM — 4 Hz dashboard-polling arm (added after AC33 was tightened)

The tightened AC33 requires the *after* run to have the dashboard polling at **4 Hz with a
scenario running**. My primary baseline was captured with **zero polling**, which would have
made the later delta equal *(instrumentation cost + polling load)* — potentially a **false
FAIL** from polling alone. Because the clean tree is unrecoverable, I captured the missing
arm immediately.

### 6b.1 Does 4 Hz polling cost decode throughput? — **No, not measurably**

Design: alternating blocks of 4 iterations, POLL-OFF / POLL-ON / POLL-OFF / POLL-ON, one
session, `max_tokens=512`. Poller hit `/metrics`, `/v1/status`, `/v1/debug/kv`,
`/v1/resources` at 4 Hz. Blocking (not per-request alternation) per the §0 finding.

| arm | median tok/s | mean | stdev | CV | n |
|---|---|---|---|---|---|
| polling OFF | 32.292 | 32.526 | 0.859 | 2.64 % | 8 |
| polling ON (4 Hz) | 32.759 | 32.677 | 1.104 | 3.38 % | 8 |

**Delta: +1.45 % median / +0.46 % mean, with SEM of the difference = ±1.53 %.**

The delta is **statistically indistinguishable from zero**, and its *positive* sign is
physically meaningless — polling cannot accelerate decode — confirming it is pure noise.

> **Conclusion:** 4 Hz polling of the existing endpoints imposes **no measurable decode
> cost**. The tightened acceptance rule is therefore safe: the polling requirement will not
> by itself consume the 2 % budget or manufacture a false FAIL.
>
> **Caveat:** this holds because today's endpoints are cheap (`/v1/status` and
> `/v1/debug/kv` return stubbed constants). Once telemetry makes them do *real* work, the
> polling cost becomes real — which is exactly what the after-run must measure.

### 6b.2 🚨 P0 — `/metrics` and `/v1/resources` stall for the ENTIRE generation

Found while validating the poller: it logged 12 errors in 52 requests. Not a harness bug —
a real server defect. Measured with one **independent thread per endpoint** so requests
could not queue behind one another, during a 384-token generation:

| endpoint | idle median | during generation (median) | during generation (max) | polls completed |
|---|---|---|---|---|
| `/metrics` | 0.8 ms | 2.5 ms | **14,784 ms** | **5** |
| `/v1/resources` | 0.8 ms | 4.7 ms | **14,785 ms** | **5** |
| `/v1/status` | 0.9 ms | 1.8 ms | 18.1 ms | 61 |
| `/v1/debug/kv` | 0.8 ms | 1.9 ms | 21.2 ms | 61 |
| `/health` | 1.6 ms | 1.8 ms | 25.5 ms | 61 |

`/metrics` and `/v1/resources` complete **5 polls where the others complete 61** — they
block for the full generation duration (~14.8 s ≈ the whole 384-token run). All endpoints
are fast (<2 ms) when the server is idle, so this only manifests under load — i.e. exactly
in the demo scenario.

**Severity: P1 for the server, P0 for the demo** — the dashboard is planned to poll
`/metrics`, and it would freeze on every generation.

**Root cause (traced through the code, not guessed):**
1. `routes/admin.rs:396` — `prometheus_metrics` awaits `state.…engine.resource_snapshot()`.
2. `driver.rs:383-386` — `resource_snapshot()` sends `DriverCommand::ResourceSnapshot` over
   a channel and awaits a `oneshot` reply.
3. The driver loop is occupied running the decode loop for the whole generation, so the
   command waits in the queue until generation completes.
4. `/v1/resources` uses the identical path, hence identical behaviour.

`crate::metrics::encode_prometheus()` is an atomic-registry read and is *not* the problem —
it is the driver round-trip that blocks. `/v1/status`, `/v1/debug/kv` and `/health` are fast
**precisely because they never touch the driver** — which is also why they return stubbed
zeros (§7).

**Reproduction:**
```bash
# terminal 1: start server (see §4), then
python3 harness/qa_baseline_harness.py   # or any 384-token streaming request
# terminal 2, during the generation:
time curl -s localhost:8123/metrics > /dev/null   # ~15 s
time curl -s localhost:8123/v1/status > /dev/null # ~2 ms
```

### 6b.3 ⚠️ This invalidates the planned telemetry design — read before implementing

Recon §8 item 2 proposes adding a `DriverCommand::KvSnapshot` and wiring it into
`/v1/status` and `/v1/debug/kv` to replace the stubbed zeros.

**Those two endpoints are currently the only fast ones.** Routing them through a
`DriverCommand` would give them the same driver-queue dependency as `/metrics` and convert
them from 1.8 ms into ~15 s stalls under load. The dashboard would freeze during every
generation — breaking the demo in precisely the scenario it exists to showcase.

**Recommendation:** do not put KV/batch telemetry behind a driver round-trip. Have the
decode loop **push** state into shared atomics or an `ArcSwap` snapshot that HTTP handlers
read lock-free — the pattern `metrics.rs` already uses, and the reason
`encode_prometheus()` costs 0.8 ms. Write-side push, not read-side request.

A fix for `/metrics` itself follows the same shape: serve `resource_snapshot` from a
periodically-refreshed cached snapshot rather than round-tripping the driver.

---

## 7. Incidental findings (not blocking, filed for the team)

- **P2 — `/v1/status` reports zeros for everything interesting.** After 135 requests and
  28,659 completion tokens: `kv_usage: 0.0, kv_pages_used: 0, kv_pages_total: 0,
  tokens_per_second: 0.0, batch_utilization: 0.0`. Confirms recon §7 — these are stubs, not
  a runtime bug. Note that `tokens_per_second: 0.0` means **the dashboard cannot currently
  source tokens/sec from the server**; it must derive it client-side, or the stub must be
  filled.
- **P2 — `/v1/debug/kv` returns the literal string** `"unavailable: engine does not yet
  expose KV page statistics"`. Also a known stub.
- **P3 — Prefix cache is bypassed on the continuous-batch path.** See §6 P3. ⚠️ Established
  from source, **not** from the `0/135` reading — both sides of that ratio are compile-time
  constants and it would read identically with the cache deleted. Worth a real investigation;
  note the panel itself still ships per the 🔒 ruling, rendering `not-applicable`.
- **Working correctly:** `/metrics` emits real data — `onnx_genai_completion_tokens_total
  28659`, `onnx_genai_tokens_generated_total 34879`,
  `onnx_genai_time_to_first_token_seconds_{sum 681.69, count 135}`. TTFT histogram is live
  and trustworthy.
- **Output quality is good.** The model produced coherent text
  (`"A hash table is a data structure that allows for efficient storage and retrieval…"`),
  not gibberish. `qwen2.5-0.5b-scatter-v2` is confirmed usable for the demo.

---

## 8. Raw data

Archived next to this file:

| file | contents |
|---|---|
| `raw/qa-baseline-primary.json` | **primary** — single-512 n=15, concurrent-512 n=4 |
| `raw/qa-baseline-raw.json` | 128-token phase: single n=15 + concurrent n=8, full per-iteration records |
| `raw/qa-baseline-rep2.json` | independent replicate of the 128-token single phase |
| `raw/qa-baseline-long512.json` | first 512-token variance probe (n=8) |
| `raw/qa-baseline-nullab.json` | null interleaved A/B (12 pairs, same binary) |
| `raw/qa-baseline-polling.json` | **4 Hz polling arm** — POLL-OFF vs POLL-ON blocks (§6b.1) |
| `raw/qa-endpoint-latency.json` | endpoint latency, idle vs during generation (§6b.2) |
| `raw/qa-endpoint-latency-indep.json` | **per-endpoint independent pollers** — the P0 evidence (§6b.2) |
| `raw/qa-preserved-validation.json` | preserved-binary validation run — the −9.8 % load-drift evidence (§5.0) |
| **`clean-binary/onnx-genai-server-clean-d49d3c8`** | **the preserved clean-tree server binary — the BEFORE arm (§5.0)** |
| `raw/qa-baseline-server.log` | server startup log incl. the `max_batch=4` line |
| `harness/qa_baseline_harness.py` | the measurement harness (stdlib only) |

---

## 9. Sign-off

- ✅ Baseline captured at two independent generation lengths that cross-validate to 1.1 %.
- ✅ Server stopped; `curl localhost:8123/health` refused; zero stray
  `onnx-genai-server` processes.
- ✅ **I modified no Rust source and no repository file.** Harness and raw data live in
  my artifact directory only. `git status -- '*.rs'` is clean.
- ⚠️ The acceptance criterion as originally worded is not testable without the protocol in
  §5. **Whoever runs the post-telemetry comparison must read §5 and §6 first.**
- 🚨 **§6b.3 is a blocking design warning for the telemetry implementation.** The currently
  planned `DriverCommand::KvSnapshot` approach would turn the dashboard's only two fast
  endpoints into 15-second stalls. Read it before writing code.
- ✅ All four tightened-AC33 requirements satisfied: ≥3 runs (15 single / 4+8 concurrent),
  raw per-run numbers published (§1, §2, §8), conditions recorded as a first-class
  deliverable (§4), and the inconclusive-result rule plus 4 Hz polling condition folded
  into the acceptance spec (§5, §6b).
- 🔒 **The clean-tree binary is preserved and validated** (§5.0). This largely defuses the
  "unrecoverable measurement" premise of this task: the BEFORE arm can now be re-run on
  demand, at any time, against the telemetry build in the same session. That is a strictly
  better comparison than any recorded number, because it cancels machine state — which I
  measured moving the result by **9.8 %**, five times the acceptance threshold.
- ⚠️ **§5.2 corrects an arm-matching trap in the amendment:** it specifies 4 Hz polling for
  the after run only. Both arms must poll, or the polling load is charged to telemetry.
- 🔀 **Two-server demo re-scope assessed (§4b): baseline stands.** The back-to-back protocol
  absorbs ambient conditions, so a second resident server does not invalidate it. The
  acceptance A/B should stay **single-server** to isolate telemetry cost from deployment
  contention; measure two-server deployment cost separately. Deployment shape is now
  recorded explicitly in §4 rather than left as an unstated assumption.

---

## 6c. 🔴 PROTOCOL DEFECT — THE `<2%` CRITERION CANNOT BE SETTLED BY ONE RUN PER ARM

**Raised by @c0de4c2e; investigated and CONFIRMED, but the proposed remedy does not fix it.**

The challenge: HEAD advanced mid-window (8 files, +2515 lines), so the baseline arm carried CPU
contention that a post-telemetry arm in a quiet tree would not. Contention present in one arm and
absent in the other is a **systematic offset in the mean, and no sample count fixes bias.** Correct
concern. I had cleared it on the grounds that zero Rust changed and the binary predates the commits
— that argument is airtight **for the binary** and irrelevant **for the timing environment.**

I had the per-sample data to settle it, so I did.

### 6c.1 Is there drift within the headline arm? — YES, and it exceeds the criterion

`single512`, n=15, in temporal order:

| statistic | value |
|---|---|
| first-half mean (samples 1–7) | 33.182 tok/s |
| second-half mean (samples 9–15) | 33.926 tok/s |
| **drift, second vs first** | **+2.24%** |
| Spearman ρ vs sample order | +0.475 |
| permutation p (N=200 000) | 0.076 |

p = 0.076 is not conventionally significant, **and that does not rescue the arm.** For a
threats-to-validity judgment the relevant quantity is the *magnitude* of a plausible bias, not
whether it clears p<0.05: a **+2.24% within-arm trend is larger than the entire ±2% acceptance
band.** Demanding significance before acknowledging a bias of criterion-exceeding size is exactly
backwards — it is the failure mode where an underpowered test licenses the thing it failed to detect.

### 6c.2 Is `loadavg` a usable proxy for "quiet"? — NO. This kills remedy option 1

The 128-token arm captured `loadavg` **per sample**, so this is directly testable:

| test | result |
|---|---|
| Spearman ρ(loadavg, decode_tps) | **+0.079** |
| permutation p | **0.785** |

**Load average has no relationship to our throughput at all** — and the sign is *positive*, i.e. if
anything the machine read busier while running faster. This is expected on reflection: `loadavg` is
a 1-minute exponentially-weighted count of runnable threads machine-wide, far too slow and too
coarse to track contention on the specific cores in our decode path.

⇒ **"Re-baseline in a quiet window" is not executable, because we have no instrument that certifies
a window was quiet.** We would simply be relabelling an unverified condition as a controlled one.
That is worse than the status quo: it converts an acknowledged threat into an invisible assumption.

### 6c.3 Is the drift a consistent direction? — NO. It is low-frequency wander, not a fixed bias

| run (identical binary) | first-half → second-half |
|---|---|
| 512-token primary | **+2.24%** |
| 128-token arm | **+3.07%** |
| 512-token replicate 2 | **−5.05%** |

The direction **flips**. So this is not "the machine steadily got faster as work drained" — it is
**autocorrelated wander on the timescale of a run (~8 min)**. Each arm therefore carries a random
offset of a few percent that **more samples within that arm cannot reduce**, because the samples are
not independent with respect to it.

### 6c.4 The decisive number: two runs of a BYTE-IDENTICAL binary differ by more than the criterion

| run (same preserved clean binary, same protocol) | mean | within-run CV |
|---|---|---|
| primary (n=15) | 33.576 tok/s | 1.98% |
| replicate 2 (n=15) | 32.882 tok/s | 5.01% |
| **between-run delta** | **−2.07%** | — |

**A binary compared against itself fails the `<2%` criterion.** The within-run CV of 1.98% that I
reported as evidence the protocol was sound measures **dispersion of samples inside one run**; the
quantity the criterion actually depends on is **dispersion of run means**, which is at least the
same order and is completely unconstrained by n-per-run.

**This is my own §5 acceptance protocol failing on its own data, and it supersedes my earlier
"n=15 at 512 tokens is sufficient" conclusion. That conclusion was right about the wrong variance
component.**

### 6c.5 Corrected protocol — interleave at the RUN level, not the request level

**Unit of analysis is the run mean, not the sample.**

1. Alternate binaries **run by run**: A B A B A B A B A B — **≥5 runs per arm**, 15×512-token
   samples each.
2. Compute one mean per run ⇒ 5 numbers per arm. Compare arms with a test on those 10 run means
   (Welch or exact permutation on run labels).
3. Report the CI **of the run-mean difference**. Report n_runs, not just n_samples.
4. Non-overlapping CI required to claim a regression *or* to claim `<2%`.
5. **Do not gate on "quiet machine"** — §6c.2 shows we cannot verify it. Run-level interleaving
   makes wander a shared nuisance across both arms instead of a confound, which is the point.

**Cost: ~80 min unattended for the full matrix.** Cheap for the project's only quantitative
acceptance criterion, and it is *unattended* — no code freeze, no coordination, since the clean
binary is preserved.

### 6c.6 Relation to my earlier NEGATIVE result — these are consistent, not contradictory

I previously tested and **rejected** interleaved paired A/B, because pairing *doubled* the variance
of the delta (paired stdev **7.01 pp**, SEM 2.02 pp on n=12 — see `raw/qa-baseline-nullab.json`).
That result stands. The reconciliation is the **timescale of the noise versus the granularity of
the interleaving**:

- **Request-level** interleaving attacks *drift* but is defeated by **high-frequency per-request
  jitter** — pairing two adjacent noisy requests sums two independent jitters into every delta.
- **Run-level** interleaving averages that jitter away *within* each run (n=15), then attacks the
  **low-frequency wander** that is the actual confound between arms.

**Rule: interleave at the granularity that matches the noise you are fighting, and average below
it.** Interleaving finer than the noise timescale imports variance without removing bias.

### 6c.7 Status of the existing baseline

The headline **33.415 tok/s median (n=15, CV 1.98%)** stands as a **descriptive** figure and the
raw data is unaffected. What does **not** stand is using it as a **single-run reference arm for a
±2% verdict**. The preserved clean binary means the corrected matrix can be run at any time; the
baseline does not need re-capturing so much as **re-framing plus 4 more runs per arm.**

---

## 6d. NULL TEST EXECUTED — the `<2%` criterion is an EQUIVALENCE claim, and my §6c.5 rule was wrong

**10 consecutive runs of the preserved clean binary against ITSELF**, sham-labelled A/B/A/B…,
15×512-token samples per run, single server, no restart, no treatment of any kind.
**Any difference found is by construction 100% noise.** 55 min. Raw: `raw/qa-runlevel-null.json`,
conditions in `raw/qa-runlevel-null-conditions.txt`, harness `harness/qa_runlevel_null.py`.

Conditions were *not* quiet — another agent's `verify_model.sh` ran concurrently and loadavg
ranged 11.6 → 43.7. That is the realistic shared-machine condition, and it is the condition the
telemetry A/B will actually run in.

### 6d.1 Headline: a binary compared against itself produced a **+6.23%** delta

| quantity | value |
|---|---|
| run means (tok/s) | 30.048, 28.644, 28.589, 28.982, 29.480, 30.428, 29.023, 25.887, **18.246**, 29.880 |
| mean within-run CV | 14.21% |
| **between-run CV** | **12.98%** |
| **sham A vs sham B (naive run means)** | **+6.23%** |
| max pairwise run-to-run delta | **+66.77%** |

**A naive single-number report would have declared a 6.23% regression — 3× the acceptance band —
from a binary that was not changed.** This is the concrete demonstration that the original
one-run-per-arm procedure cannot support a ±2% verdict.

### 6d.2 The decision rule WORKS — the point estimate does not

Running the §6c.5 procedure properly on the same data:

| procedure | result |
|---|---|
| naive point estimate (run means) | **+6.23%** ❌ misleading |
| robust point estimate (run **medians**) | **+1.27%** ✅ much better |
| **exact permutation test on run labels** (all C(10,5)=252 assignments) | **p = 0.643** ✅ correctly *indistinguishable* |
| bootstrap 95% CI of the delta | **[−5.55%, +28.08%]**, width 33.6 pp |

**The run-level permutation test refused to call a regression that a point estimate would have
reported.** That is the protocol validating exactly as designed. Two required amendments follow:

1. **Use run MEDIANS, not run means.** One pathological run (run 8: mean 18.246, CV 37.95%,
   individual samples down to 9.6 tok/s under load 33) dragged the mean-based estimate to +6.23%.
   Medians gave +1.27% on identical data. **Never report a bare delta of run means.**
2. Excluding that single pathological run, between-run CV is still **4.58%** and the sham delta
   **−1.78%** — i.e. **noise alone consumes essentially the entire ±2% budget.**

### 6d.3 🔴 MY §6c.5 RULE WAS WRONG — "CI straddles 0" DOES NOT MEAN "passes `<2%`"

§6c.5 said *"non-overlapping CI required to claim a regression **or** to claim `<2%`."* The second
half is a category error and I am correcting it.

**`<2%` is an EQUIVALENCE claim, not a difference claim.** A non-significant difference test
(p = 0.643 above) means *"we could not detect a difference"* — it is **not** evidence that the
difference is small. With a CI of **[−5.55%, +28.08%]** we manifestly have **not** shown overhead
is under 2%; a +20% regression sits comfortably inside that interval. Reading p > 0.05 as "passes"
is the single most likely way this criterion gets falsely certified.

**Correct rule (TOST / equivalence):**
> To claim `<2%`, the **entire 95% CI of the run-mean-difference must lie inside ±2%.**
> If the CI is wider than the band, the verdict is **`UNRESOLVED` — re-run with more runs.**
> `UNRESOLVED` is **not** a pass and must never be reported as one.

Three-state outcome, never two: **REGRESSION** / **EQUIVALENT (`<2%`)** / **UNRESOLVED**.

### 6d.4 This resolves the "you can't certify a quiet machine" objection

§6c.2 showed we have no instrument that certifies a window as quiet (ρ(loadavg, tok/s) = +0.079).
That looked like a dead end. It is not, because **the CI width adjudicates it after the fact:**

- Do **not** gate on "quiet" in advance — unverifiable, and declaring it invites a false pass.
- **Let the observed CI width certify the run retrospectively.** Contention inflates between-run
  variance ⇒ inflates the CI ⇒ the CI fails to fit inside ±2% ⇒ verdict `UNRESOLVED` ⇒ re-run.
  **A noisy window can no longer produce a confident answer.** The procedure is self-certifying.

### 6d.5 How many runs are actually needed

For the CI half-width to fit the ±2% band, we need SEM ≲ 2/1.96 ≈ 1.02%, so
n_runs ≈ (between-run CV / 1.02)².

| observed between-run CV | condition | runs needed **per arm** | wall time/arm |
|---|---|---|---|
| 12.98% | this test, incl. pathological run | ~162 | infeasible |
| 4.58% | this test, pathological run dropped | **~20** | ~7 h |
| ~2% | quiet machine (plausible) | **~4** | ~25 min |

⇒ **The ±2% criterion is cheap on a quiet machine and effectively unaffordable on a busy one.**
The practical recommendation is unchanged and now quantified: run the A/B when the crew is idle,
**and let the CI decide whether the window was good enough.** Discard-and-rerun on `UNRESOLVED`
is legitimate; discarding an *individual* run for looking bad is not — the decision must be made
on CI width, never by eyeballing which runs to keep.

### 6d.6 Status

- §6c.5's run-level interleaving + permutation test: **VALIDATED** — it correctly returned
  "indistinguishable" on a null where a point estimate reported +6.23%.
- §6c.5's acceptance wording: **CORRECTED** to a TOST equivalence rule with a third
  `UNRESOLVED` state (§6d.3).
- Aggregate run **medians**, not means (§6d.2).

---

## 6e. THE RATIFIED SIGN-OFF PROCEDURE, TESTED ON NULL DATA — AND WHY AC33 IS THE WRONG SHAPE

The 🔒 ratified procedure is *"both binaries back-to-back in one session, **alternating blocks of
5**, to **≥15 samples per arm**, both polling at 4 Hz, 512-token generations."* Blocking is a real
improvement over one-run-per-arm and it removes the perishable-baseline problem entirely.

**But I can now test the instrument instead of reasoning about it.** The null run produced **150
samples of the same binary in temporal order**. Simulating the ratified procedure across every
start offset — assigning alternating blocks of 5 to two arms that are *the same binary*, so the
true difference is exactly **0** — gives its false-verdict rate directly.

### 6e.1 The ratified procedure reports a ≥2% difference between a binary and itself ~2 times in 3

| procedure | n/arm | median \|delta\| | p90 \|delta\| | **false ≥2% verdict** |
|---|---|---|---|---|
| **blocks of 5 (ratified)** | **15** | **3.27%** | 11.70% | **64.5%** |
| blocks of 5 | 30 | 1.33% | 5.40% | 38.5% |
| blocks of 5 | 60 | 0.98% | 4.44% | 25.8% |
| blocks of 10 | 30 | 1.49% | 2.60% | 23.9% |

Restricting to the quieter runs (within-run CV < 11%) **does not help** — 73.7% at n=15 — because
selecting on *within*-run calm does not remove *between*-block wander. **Even at n=60/arm (≈4×
the ratified cost) the false-verdict rate is still 26%.**

⇒ **A ±2% gate cannot be resolved by this instrument at any practical sample count on this machine.**
Blocking fixed the *bias* (the perishable baseline). It cannot fix the *variance*.

### 6e.2 The reframe: we are trying to detect a sub-0.01% effect with a ±3% instrument

The telemetry design (reviewed in detail — atomics only, no allocation, no lock, no syscall on the
decode path) costs a handful of **relaxed atomic stores per decode step**. Against a measured decode
step of **29.94 ms**:

| telemetry work per decode step | cost | share of a decode step |
|---|---|---|
| ~10 relaxed stores (realistic) | 10 ns | **0.000033%** |
| 100 stores (10× pessimistic) | 100 ns | 0.000334% |
| 1000 stores (100× pessimistic) | 1 µs | 0.003340% |

| | |
|---|---|
| AC33 budget | **2%** |
| predicted effect | **~0.00003%** |
| **budget ÷ effect** | **≈ 60,000×** |
| instrument resolution (measured) | **~3.27%** |

**The thing AC33 measures is ~60,000× smaller than the budget it is measured against, and ~100,000×
smaller than the noise floor of the instrument.** No sample count closes a gap of that order. **A
"pass" would be an artifact of rounding, and a "fail" would be someone else's `cargo build`.**

### 6e.3 Recommendation — verify the MECHANISM, bound the magnitude, stop chasing the delta

This is the same move that settled the prefix cache in §9 of `prefix-cache-verification.md`:
**compute what the effect must be, then compare it to what the instrument can see.** There, the
predicted effect (−94%) was far *larger* than the noise, so a null result was decisive. Here the
predicted effect is far *smaller* than the noise, so **a throughput A/B is decisive in neither
direction — it can only manufacture a verdict.**

**Proposed AC33, restated so it is answerable:**
1. **Mechanism check (primary, and it is a hard gate).** On the decode path the instrumentation
   must perform **no allocation, no lock, no syscall, no channel send, and no unbounded work** —
   verified by inspection and by the absence of those constructs. ✅ **Already verified in my
   telemetry design review:** `note_block` is one bounds-checked relaxed store, gauges are relaxed
   stores, the `Vec` allocation is on the HTTP thread. This is the criterion that actually protects
   the decode loop, and it **cannot be satisfied by luck.**
2. **Magnitude bound (secondary).** State the arithmetic above: O(10) relaxed stores against a
   ~30 ms step is bounded well under 0.01% even at 100× pessimism.
3. **Gross-regression guard (tertiary, cheap).** Keep the blocked A/B, but gate it at a threshold
   the instrument can actually resolve — **±10%, not ±2%** — with the verdict rendered as
   **EQUIVALENT / REGRESSION / UNRESOLVED** per §6d.3. Its job is to catch a *blunder* (a lock, an
   allocation, a snapshot in the loop), which would be tens of percent and trivially visible — not
   to certify 2%.
4. **DELETE the absolute thresholds (≥32.75 / ≥51.46 tok/s).** They fail the clean binary **10/10**
   (§6e.4). Comparisons must be relative and same-session.

**This is stricter than the current AC33, not weaker.** A 2% gate that fires at random is not a
safety net — it is a coin toss that will eventually be resolved by whoever argues hardest. A
mechanism gate is binary, cheap, reviewable, and cannot be passed by a quiet machine.

### 6e.4 The absolute thresholds fail the clean binary 10 times out of 10

AC33 lists `≥32.75 tok/s single`. The preserved **clean** binary, unmodified, across the null run:

| runs | 30.05, 28.64, 28.59, 28.98, 29.48, 30.43, 29.02, 25.89, 18.25, 29.88 |
|---|---|
| **passing ≥32.75** | **0 / 10** |

**The baseline binary cannot pass the threshold derived from itself.** An absolute threshold silently
encodes the machine conditions of the ten minutes in which the baseline was captured. Had telemetry
been A/B'd against it tonight, the PR would have been charged with a ~12% regression it did not cause.

---

## 6f. THE −9.8% SWING HAS A NAME. I AM RETRACTING IT AS EVIDENCE — AND THE CONCLUSION SURVIVES ON BETTER EVIDENCE

@1cb42f0e disclosed two CPU-heavy Mobius ONNX exports at **~23:21** and **~23:25**. I reconstructed
each run's window from the embedded server-log timestamps (logged in UTC; local = UTC−7) rather than
from file mtimes, which only record when I archived the artifact:

| run | server start (local) | archived | verdict |
|---|---|---|---|
| primary baseline | **22:46:13** | 23:08:04 | ✅ **CLEAN** — ends 13 min before the first export |
| 4 Hz polling arm | **23:10:32** | 23:20:28 | ✅ **CLEAN** — ends ~30–90 s before the first export |
| **preserved-binary re-run (30.151 tok/s)** | **23:22:30** | 23:26:44 | 🔴 **TAINTED** |
| run-level null test | 00:32 | 01:11 | ✅ **CLEAN** — 67 min after |

The preserved-binary re-run loaded its model for ~55 s and generated from **≈23:23:25 to 23:26:44**.
That window **straddles the ~23:25 export outright** and sits immediately downstream of the ~23:21 one.

### 6f.1 What I retract

**I cited 33.415 → 30.151 tok/s (−9.8%) on a byte-identical binary as proof that the perturbing
variable is unattributable ambient load. That specific claim is withdrawn.** It was attributable, it
has now been attributed, and I attributed it to `mediaanalysisd` and Time Machine on the strength of
a `loadavg` reading — which is precisely the inference I had *already demonstrated to be worthless*
(§6c: `loadavg` vs throughput, ρ = +0.079, p = 0.785). **I used a proxy I had personally disproven,
because it pointed the way I already believed.** The number was real; the story I attached to it was
not evidence.

### 6f.2 What does not change, and why it is now on firmer ground

The conclusion — *a cross-session before/after comparison cannot resolve 2%* — never depended on that
run, and the **run-level null test is the load-bearing evidence.** It ran **00:32–01:11**, more than
an hour clear of any disclosed heavy job, on one binary against itself:

| between-run CV | **12.98%** |
|---|---|
| phantom delta of run means | +6.23% |
| max pairwise run difference | 66.77% |
| ratified procedure's false ≥2% verdict rate | **64.5%** |

**None of that is explained by the exports.** Had the −9.8% been the only evidence, this disclosure
would have overturned the finding. It is not, and it does not.

### 6f.3 The sharper version of the argument

The naive reading is *"so it wasn't mysterious after all — announce heavy jobs and cross-session
comparison works."* That fails on two counts:

1. **The disclosure was voluntary, retrospective, and human.** It arrived ~2 h later because an agent
   remembered. No lock table, DAG edge, or `loadavg` sample surfaced it — I sampled `loadavg`
   *during* the contaminated run and it told me a story about Spotlight.
2. **Attribution is not reinstatement.** To compare a future after-run against 23:26:44's number I
   would have to re-run two multi-GB exports at the same phase offset. **Knowing the cause explains
   the damage; it does not restore comparability.** This is exactly what the back-to-back
   both-arms protocol makes moot, because the contamination lands on both arms.

**And the general lesson is the one that keeps recurring: a named cause for an anomaly is not the
same as a validated instrument.** I had a real number, a plausible culprit, and a confirming proxy
reading, and the culprit was wrong. Only the null test — same binary, no hypothesis, measured
dispersion — could settle it.

# §7 — P0 VERIFIED FIXED (`/v1/resources` and `/metrics` under sustained load)

Fix-verification run. I opened this P0; I am closing it on a measurement.

**Build provenance.** Branch `feat/genai-demo-dashboard` (asserted, not assumed),
`git status --porcelain -uall -- crates/` = 0 entries, HEAD **`1d9a5515`**.
Binary rebuilt from that tree; 0 `.rs` files newer than the binary.
Behavioural fingerprint: `--max-batch` ✅, `--max-queue-depth` ✅,
`--demo-assets-dir` ✅, `--cors-allow-origin` **absent** ✅ (post-fix shape).
Launched via `run-demo.sh` on non-default ports 9231/9232 — never a bare binary.

**Identity check, by behaviour rather than path.** `/v1/models` on the new build
carries `loaded`, `is_default`, `path` — the post-fix key set — and `created` is
**stable across two curls 2 s apart**, which is the discriminator that matters
(the pre-fix binary's `created` moves, because it was `now_unix()`).

> ⚠️ **And the trap the lead warned of cannot fire, for a reason worth recording:
> I ran the discriminator on ALL FOUR listening servers — 9231, 9232, 8123, 8124 —
> and every one is post-fix (7-key payload, `created` stable). There is no
> preserved-baseline binary listening anywhere. The warning was correct in its
> reasoning and inapplicable in fact, and I could only establish that by probing
> every port rather than the two I launched.**

## 7.1 The result

Method unchanged from §6b.2: concurrent probe threads against a sustained 1024-token
generation. Worst-case reported, never the mean.

    /v1/resources           BEFORE (HEAD~)      AFTER (1d9a5515)
    probes completed        1  (blocked)        195
    median                  --                  2.2 ms
    WORST                   51,010 ms           69.5 ms
                                                -> 734x worst-case improvement

    /metrics                51,010 ms worst     47.3 ms worst    n=195, median 2.3 ms
    /v1/debug/kv                                80.9 ms worst    n=195, median 2.3 ms
    /health                                     80.2 ms worst    n=195, median 2.2 ms
    generation              53.0 s

**CONTROL ARM — the batched server, which was never broken:**

    :9231 scatter, 51.6 s generation
    /metrics       n=190  median 2.1 ms  worst 63.4 ms
    /v1/resources  n=190  median 2.2 ms  worst 58.1 ms

**The two servers are now statistically indistinguishable under load.** That is
the correct post-fix shape: the defect was that the non-batched path answered
`ResourceSnapshot` only between commands while `Generate` decoded inline, so the
endpoint was readable *only when there was nothing to report*. Both paths now
answer mid-generation.

## 7.2 Why load cannot corrupt this verdict

Load average was **10.5 → 39.6** across the dynamic arm and **34.2 → 27.8**
across the control — genuinely bad conditions, and I would refuse to publish a
throughput number from them. **This verdict is immune for a reason I should
state rather than assume: the effect is ~734×, and the largest load-induced
swing ever observed on this box is 9.8%.** Contention cannot manufacture three
orders of magnitude. Equally decisive and not a timing quantity at all: **the
probe COUNT went from 1 to 195.** Under the old build the endpoint could not be
sampled during a generation at all; now it can be sampled 195 times. That is a
counting result, and counting results do not have a coefficient of variation.

**Verdict: 🟢 P0 CLOSED. The fix is real, it is present on both profiles, and
the previously-broken path now matches the control.**

## 7.3 A LIVE BEFORE-ARM — and a correction to my own §7 claim

**First, the correction.** In §7 I wrote that all four listening servers were
"post-fix", on the strength of the `/v1/models` key set and a stable `created`.
**That claim was too strong and @1cb42f0e was right to narrow it.** Their probe
is sharper than mine:

    onnxgenai_resource_governor_available in /metrics
      :9231  3 occurrences   NEW image (1d9a5515-era)
      :9232  3 occurrences   NEW image
      :8123  0 occurrences   OLDER image
      :8124  0 occurrences   OLDER image

`created`-stability tests **one** fix; it cannot date a binary in general. My
statement was true of the property I measured and false as the general claim I
made from it — **the exact "wrong noun" failure I have flagged in others twice
tonight.** The narrow version stands: no *preserved AC33 baseline* is listening.

**Second, and it is a gift: the older image turns :8124 into a genuine BEFORE
arm, running concurrently, on the same machine, under the same kernel.**

    DYNAMIC PROFILE          :8124 OLD image        :9232 NEW image
    generation                37.5 s                 53.0 s
    /metrics       n              1                    195
                   worst    35,456.6 ms              47.3 ms
    /v1/resources  n              1                    195
                   worst    35,456.2 ms              69.5 ms
    ---- within-run controls, same harness, same instant ----
    /health        n            136                    195
                   median       2.4 ms                2.2 ms
    /v1/debug/kv   n            136                    195
                   median       2.5 ms                2.3 ms
    loadavg              52.93 -> 74.90        10.54 -> 39.61

**This is the strongest evidence class available for a fix, and it is stronger
than the sequential before/after I published in §7.1.**

1. **The before and after are simultaneous.** No rebuild, no reboot, no elapsed
   hours between arms — the two binaries were serving at the same time.
2. **The old arm ran under HEAVIER load** (52.9→74.9 vs 10.5→39.6) — so if
   contention were the explanation, it would push the *old* arm's `/health` up
   too.
3. **It doesn't.** `/health` and `/v1/debug/kv` on the OLD server returned **136
   probes at 2.4 ms median** during the same generation in which `/metrics`
   returned **one probe after 35 seconds**. The machine was responsive, the
   harness was working, the process was answering — **only the two endpoints
   that call `resource_snapshot()` stalled.**

**That within-run control is what makes this dispositive.** A confound that
explains a 35-second stall on `/metrics` while leaving `/health` at 2.4 ms on the
same process, in the same second, does not exist. **The fix caused the change.**

🟢 **P0 CLOSED, on a same-machine, same-instant, load-adversarial before/after
with an internal control.** No timing claim here depends on the absolute
numbers — it depends on 1-vs-195 and on a within-run control, both of which are
counts.

---

# §8 — AC33 AFTER-ARM: **INCONCLUSIVE**, AND THE INCONCLUSIVENESS IS MEASURED

**Author:** QA Tester (@fc8b5d97) · 02:18–02:30 PDT
**Verdict:** the after arm **cannot be run tonight**. This section does not
assert that — it **measures** it, and reports the number that forbids the run.

## 8.1 The null A/B — an experiment whose correct answer is known in advance

I ran a strictly interleaved A/B in which **both arms are the same server, the
same binary, and the same prompt.** Arms were assigned alternately (A,B,A,B…).

> **The true delta is exactly ZERO. Any delta this experiment reports is, by
> construction, entirely noise. That is the whole point: an experiment whose
> answer I already know is the only kind that can measure my instrument rather
> than my subject.**

    port 8151 · preserved clean binary d49d3c8fe1b8… · max_tokens=128 · 6 pairs

    pair 1   A 21.92   B 33.38  tok/s     delta  +52.30 %
    pair 2   A 38.68   B 27.11              -29.91 %
    pair 3   A 37.18   B 36.20               -2.62 %
    pair 4   A 23.64   B 23.91               +1.13 %
    pair 5   A 22.40   B 13.40              -40.17 %
    pair 6   A 18.70   B 22.00              +17.61 %

    arm A  median 23.02 tok/s   CV 28.88 %
    arm B  median 25.51 tok/s   CV 28.90 %

**🔴 A change that does not exist measured as high as +52.30 % and as low as
−40.17 %. The AC33 acceptance band is ±2 %. A single pair, run in good faith
and reported honestly, could have certified a 26× overhead regression or a 20×
speedup — from identical bytes.**

## 8.2 The trap: the summary statistic passes while every observation fails

    mean paired delta     −0.28 %     <- INSIDE the +/-2 % band
    median paired delta   −0.74 %     <- INSIDE the +/-2 % band
    95 % CI of the mean   −35.28 % .. +34.73 %   (width 70.01 points)

> **⚠️ THE HEADLINE NUMBER PASSES. Had I reported only "mean delta −0.28 %,
> within tolerance", I would have published a PASS that is indistinguishable
> from a real one — and it would have been arithmetically correct and completely
> worthless. The CI is 17.5× wider than the entire acceptance band.**
>
> **Averaging does not remove noise. It hides it, and it hides it behind a
> number that looks MORE precise than the data underneath — two decimal places
> resting on observations that disagree by 92 percentage points.** This is why
> §5 requires the CI and not the mean, and this run is the first time that
> requirement has actually caught something. **A tolerance test that reports
> only a central tendency cannot fail for the right reason.**

## 8.3 What a valid after-arm would now cost

    measured paired-delta stdev            33.35 %
    pairs to resolve 2 % @ 80 % power       2184
    at the observed ~11 s/pair             ~6.7 HOURS of continuous running

    clean-tree CV committed in §4           1.98 %
    CV measured now                        28.90 %      ->  14.6x WORSE

**Per the binding inconclusive-result rule in §5 — *if the 95 % CI of the delta
spans ±2 %, report INCONCLUSIVE* — this is not a judgement call. The CI spans it
by a factor of 17.5. The rule fires on its own terms.**

## 8.4 A SECOND disqualifier that has nothing to do with load

Even at load 0, tonight's after-arm would not be comparable, because **§4 pins
the deployment shape as `server processes: exactly 1`** and states it explicitly
precisely so it could not become an invisible assumption. Right now:

    resident onnx-genai-server processes: 9

**§4's most pedantic-looking line is the one that saved the comparison.** I
wrote it as boilerplate; it turned out to be the load-bearing clause. *An
assumption written down is a disqualifier you can check. The same assumption
left unwritten is a result you publish.*

## 8.5 Internal validity check — and a correction to my own earlier claim

**Validity:** arm A CV **28.88 %** vs arm B CV **28.90 %** — near-identical, as a
null A/B must be. Both arms sampled the same noise distribution, which is the
evidence that the interleaving worked and neither arm got a privileged slot.

**🔻 Correction.** I reported earlier, from a `created`-timestamp discriminator,
that **no preserved baseline binary was still listening.** That was wrong. PID
73902 on `:8151` is the preserved clean baseline, and its SHA-256 and byte size
match §4 exactly — `d49d3c8fe1b8a98e1a067208…`, 29,033,360 bytes. **It was
listening the entire time I said it was not, and it is the server that produced
every number in this section.** My discriminator tested one fix's presence and I
read it as binary identity; the binary's own hash was available the whole time
and settles it in one command. *A weak discriminator that happens to be
convenient will be trusted past its evidence — I did exactly that.*

## 8.6 The deliverable

**AC33 after arm: INCONCLUSIVE — with a measured floor rather than an excuse.**

I would rather ship this than a PASS. The null A/B is reproducible in twelve
minutes on any machine, it needs no privileged access to a quiet window, and it
converts *"the box is too busy to measure"* — an assertion a reader must take on
trust — into **±52 % on a known-zero effect against a ±2 % gate**, which a
reader can check. **The strongest artefact I produced tonight is a measurement
of my own inability to measure.**

---

# §9 — REBUILD, ARM IDENTITY, AND A CORRECTION TO THE IDENTITY RULE ITSELF

**Author:** QA Tester (@fc8b5d97) · 02:27–02:35 PDT

## 9.1 The build has a name, because drift was made impossible rather than detected

The working tree carried **one uncommitted Rust change** at build time. A build
started there yields a binary matching **no commit**. I did not stash it —
stashing silently appropriates another agent's in-progress work — and I did not
build around it.

Instead I built from a **detached worktree pinned to an explicit SHA**:

    pin           5d77ec14   (detached, unshared, dirty=0 at BOTH ends)
    binary sha256 ab175ea1cd44158520310a845a427c9d
    size          29,509,024 bytes      built 02:29:18 in 43.09 s
    flags         --max-batch ✅  --demo-assets-dir ✅  --cors-allow-origin absent ✅

> **@d7cf9b84's bracketing rule says: capture SHA before and after, and rebuild
> if the ends disagree. A pinned detached worktree is strictly stronger — THE
> ENDS CANNOT DISAGREE, BECAUSE NOTHING CAN WRITE TO IT. Detection is a rule you
> must remember to apply; impossibility is a property of the checkout. Prefer
> the construction that removes the failure over the check that catches it.**

## 9.2 🔴 THE RATIFIED IDENTITY CHECK IS NECESSARY BUT **NOT SUFFICIENT**, AND IT CLEARS THREE PRE-FIX SERVERS

The instruction was: *the models endpoint must show the three post-fix fields.*
Applied to every listening arm, alongside the governor family:

    port   loaded is_default path | governor | verdict
    9241     1       1        1   |    3     | ✅ POST-FIX  (this build)
    9242     1       1        1   |    3     | ✅ POST-FIX  (this build)
    9231/2   1       1        1   |    3     | ✅ POST-FIX
    8133/4   1       1        1   |    3     | ✅ POST-FIX
    8123     1       1        1   |    0     | ⛔ PRE-FIX — PASSES THE RATIFIED CHECK
    8124     1       1        1   |    0     | ⛔ PRE-FIX — PASSES THE RATIFIED CHECK
    8152     1       1        1   |    0     | ⛔ PRE-FIX — PASSES THE RATIFIED CHECK
    8151     0       0        0   |    0     | ⛔ PRE-FIX (fails both — trivially caught)

**⚠️ THE WARNING NAMED `:8151` AS THE DANGEROUS ARM. IT IS THE SAFEST ONE — IT
FAILS EVERY CHECK WE OWN. THE ACTUAL TRAP IS `:8123`, `:8124` AND `:8152`, WHICH
PASS THE RATIFIED CHECK COMPLETELY AND ARE PRE-FIX.** Had I applied the
instruction literally and stopped, I would have certified three pre-fix servers
as valid arms and attributed AC33 to code they do not contain.

**🔑 ROOT CAUSE, AND IT GENERALISES TO EVERY IDENTITY CHECK WE WILL EVER WRITE:**

    models-endpoint fields  <- dates an EARLY server commit
    governor family         <- dates 2a104dcc, 02:01:36

> **A BINARY IS NOT "OLD" OR "NEW". IT SITS AT A POINT ON A COMMIT SEQUENCE, AND
> A SINGLE-COMMIT PROBE ONLY TELLS YOU WHICH SIDE OF *ITS OWN* COMMIT YOU ARE
> ON. Any binary built between two fixes passes the earlier probe and fails the
> later one, and both readings are correct.**
>
> **THE RULE: AN IDENTITY CHECK MUST PROBE THE *LATEST* BEHAVIOUR-CHANGING
> COMMIT YOUR MEASUREMENT DEPENDS ON — NOT ANY POST-FIX COMMIT. "Post-fix" is
> not a property of a binary; it is a property of a binary *relative to a named
> commit*, and the name must be stated.**

I made the inverse of this error twice tonight — reading a `created`-timestamp
probe as binary identity, then reporting these two probes as *contradicting*
each other when they were dating different commits. **Twice I treated a one-bit
probe as a date.**

## 9.3 🟢 THE P0 RE-VERIFIES ON THE PINNED BINARY — AT THE WORST LOAD OF THE SESSION

    arm :9242 (dynamic, paged KV) · binary ab175ea1 · pin 5d77ec14
    LOADAVG AT RUN START: 109.58   <- the highest reading all night, 11x cores

    endpoint          n   median    worst
    /health           6    2.2 ms   4.8 ms
    /metrics          6    3.2 ms   6.3 ms
    /v1/resources     6    2.7 ms   4.4 ms   <- pre-fix: 1 probe, 51,010 ms
    /v1/debug/kv      6    2.2 ms   4.6 ms

**Every probe answered. Nothing blocked.**

> **THIS IS THE ONE MEASUREMENT TONIGHT THAT LOAD MAKES *STRONGER*, AND THAT IS
> WHY IT IS ADMISSIBLE WHEN AC33 IS NOT. Contention can only push an endpoint
> TOWARD blocking. A pass at load 109.58 is therefore a pass at every load below
> it — the noise pushes against the conclusion instead of toward it. Choose
> instruments whose error direction opposes your claim, and a hostile machine
> becomes a free stress test rather than a reason to wait.**

## 9.4 ⛔ AC33 TIMING: REFUSED, ON MEASURED GROUNDS

    load during this window: 41.56 -> 15.03 -> 48.40 -> 109.58   (10 cores)

Per §8, a **known-zero** effect on this box measured between **−40.17 %** and
**+52.30 %** against a **±2 %** band, at loads *below* these. I am not taking a
throughput number here, and per the standing order I am saying so and stopping
rather than producing one.

**The refusal is not a lack of data. It is a result: §8 states the floor, and
the floor exceeds the signal by more than an order of magnitude.**

---

# §10 — AC33 ACCEPTANCE A/B: **EXECUTED IN FULL. VERDICT INCONCLUSIVE.**

**Author:** QA Tester (@fc8b5d97) · 02:41–02:52 PDT
**Protocol:** §5.1, run as written. **30/30 completions. Nothing was skipped.**

## 10.1 Both arms, both binaries, one session — and identity checked *in the same invocation*

    BEFORE  :8151  preserved clean binary d49d3c8…  governor_family=0  expected 0  OK
    AFTER   :9242  pinned telemetry build ab175ea1  governor_family=3  expected 3  OK

Both serve the **same model** (`qwen2.5-0.5b`), same machine, same prompt, same
`max_tokens=512`, alternating in blocks of 5, n=15 per arm.

**The identity check runs inside the measurement script and aborts before the
first generation if either arm mismatches** — @1cb42f0e's rule. A separate
identity check is a check of a *different moment*; only an inline one describes
the binary that produced the numbers.

## 10.2 The headline, with the denominator beside it

    BEFORE   completions 15/15   median 29.09 tok/s   WORST 14.37   CV 20.56%
    AFTER    completions 15/15   median 30.35 tok/s   WORST 14.38   CV 23.19%

    delta (medians)   +4.32%
    95% CI of delta   -11.40% .. +20.03%      half-width 15.71 points
    ACCEPTANCE BAND   +/- 2%
    VERDICT           INCONCLUSIVE — the CI spans the band (§5 binding rule)

**`15/15` is reported beside every number per @1cb42f0e: a `0` next to `0/4`
and a `0` next to `4/4` are different universes, and a failed experiment is
otherwise indistinguishable from a real null result.** Both arms completed
every attempt, so this is a genuine measurement that could not resolve the
question — not a broken one.

## 10.3 🔴 THE DECISIVE NUMBER IS NOT THE CI. IT IS THAT ONE RUN CONTAINS THREE INCOMPATIBLE VERDICTS

    block 1   BEFORE 31.95   AFTER 30.35   delta   -5.01%
    block 2   BEFORE 28.32   AFTER 17.37   delta  -38.67%
    block 3   BEFORE 29.61   AFTER 33.01   delta  +11.48%

    THREE BLOCKS SPAN 50.1 PERCENTAGE POINTS AGAINST A +/-2% BAND
    loadavg during the run: 9.78 -> 121.02   (a 12x swing INSIDE one experiment)

> **Any one of these blocks, run alone and reported honestly, is a publishable
> result. Block 1 says telemetry costs 5%. Block 3 says it *pays* 11%. Block 2
> says it costs 39%. THEY ARE THE SAME BINARIES, THE SAME PROMPT AND THE SAME
> TWELVE MINUTES. The reason to run three blocks is not statistical power — it
> is that a single block cannot tell you it is lying.**

## 10.4 🔻 THIS FALSIFIES MY OWN §5.1 MANDATE — BLOCKS ARE THE WRONG DESIGN HERE

§5.1 requires *"blocks of 5, **not** per-request alternation."* **That
instruction is mine and this run shows it is wrong under the conditions we
actually have.**

**Blocks defend against per-request artefacts (warm-up, cache state). Per-request
alternation defends against *drift*. Those are different threats and you must
pick the one that dominates. Drift dominates here by an order of magnitude:** a
block occupies minutes, so a load excursion lands **entirely inside one arm** —
block 2's AFTER arm ran through a spike that its BEFORE arm never saw, which is
the whole of that −38.67%.

The evidence that alternation is the better instrument is already in this file:
**§8's per-request null A/B centred at −0.28 % on a known-zero truth**, while
this blocked design produced −38.67 % in a single block on an unknown one.

> **⚠️ AMENDMENT TO §5.1: alternate PER REQUEST, discard the first generation per
> arm as warm-up, and keep the block structure only as a reporting unit. The
> original rationale was real but it optimised against the smaller threat.**
>
> **THE GENERAL FORM: AN EXPERIMENTAL DESIGN IS A CHOICE ABOUT *WHICH*
> CONFOUNDER YOU CANCEL, NEVER A CHOICE TO CANCEL CONFOUNDING. I wrote "blocks,
> not alternation" as though it were a quality ranking. It is a trade, and I
> took the wrong side of it because I picked before I had measured which
> confounder was larger.**

## 10.5 What ships

**AC33 acceptance verdict: INCONCLUSIVE, protocol fully executed, 30/30
completions, at a measured noise floor that exceeds the acceptance band by
roughly an order of magnitude.**

**There is no evidence that telemetry costs anything** — the AFTER arm was
nominally *faster* — but "no evidence of a cost" is not "proof of <2 %", and
this file will not blur those. **The one thing that did work exactly as
predicted is §2's claim that 512-token generations suppress variance: CV fell
from 28.90 % at 128 tokens to 20.56 %/23.19 % at 512.** The protocol's
statistical advice was sound; its scheduling advice was not.

---

## 11. 🔴 THE DEMO'S TWO PANES DIFFER BY WHETHER CONTINUOUS BATCHING RUNS AT ALL — AND THE PRE-FIX SERVER ADVERTISED A WIDTH IT WAS NOT USING

Found while smoke-testing `d08d44b8` ("publish the batch width the driver will
run at, not the ceiling it was configured with"), which landed on the exact field
my AC189 P1 rests on. The server change is **correct**; what it exposed is not.

### 11.1 The measured matrix — counts only, and why only counts

Four arms, `n=4` concurrent completions each, `max_tokens=160`, identical prompt.
`batch_in_flight` sampled every 500 ms via `/v1/status` for the whole generation.

| binary | model / `--model-id` | `batch_capacity` | **peak `batch_in_flight`** | honest? |
|---|---|---|---|---|
| pre-`d08d44b8` | `qwen2.5-0.5b-scatter-v2` / `qwen-scatter` | 4 | **4** | ✅ |
| pre-`d08d44b8` | `qwen2.5-0.5b` / `qwen-dynamic` | **4** | **1** | ❌ **advertises 4, runs 1** |
| post-`d08d44b8` | `qwen2.5-0.5b` / `qwen-dynamic` | **1** | **1** | ✅ |
| post-`d08d44b8` | `qwen2.5-0.5b-scatter-v2` / `qwen-scatter` | 4 | **4** | ✅ **positive control** |

> **⚠️ THE FOURTH ROW IS THE ONLY REASON THE THIRD ROW MEANS ANYTHING.** A fix that
> simply always published `1` would produce row 3 identically. Without an arm where
> the corrected field is *required to come back large*, "it now reports 1" is
> indistinguishable from "it now reports 1 always." **A correction needs a case it
> could have failed, or it is not a test — it is a coincidence with good manners.**

**No timing ratio is stated here and that is deliberate.** Load ranged **27.00 to
60.82** across these arms — the wall-clock figures spanned 9.2 s to 42.9 s for
byte-identical work and are worthless. **The claim is a COUNT: how many rows the
driver actually ran. A count needs no arithmetic, no denominator and no quiet box,
which is exactly why the crew resolved to ship the mechanism rather than the ratio
(`2d6b36ac`), and it is why this section survives conditions in which §1 would not.**

### 11.2 What this means for the demo, which is the part that matters

`run-demo.sh:62-63` launches both panes from **different model artifacts**:

```
SCATTER_MODEL  qwen2.5-0.5b-scatter-v2   -> continuous batching ENGAGES (4 rows)
DYNAMIC_MODEL  qwen2.5-0.5b              -> continuous batching DOES NOT ENGAGE (1 row)
```

> **🔴 THE TWO PANES DO NOT DIFFER ONLY IN THE VARIABLE THE DEMO NAMES. One pane
> batches and the other does not. Any comparison drawn across the panes — throughput,
> latency, occupancy, scheduling behaviour — is confounded by the presence or absence
> of the headline feature itself.** The pane labelled **dynamic** is the one where
> dynamic batching does not happen.

And pre-`d08d44b8`, that pane served `batch_capacity: 4`, so the scheduling panel
would render **"of 4 max"** on a driver running one row wide. **A capability claim
about the product, rendered confidently, in the product's own UI, on the pane named
after the capability.** This is the same failure shape as the AC189 P1 in
`browser-render-verification.md` §7 — **a wrong number invites doubt; a wrong claim
about the system closes the question** — except here the false statement originated
in the server, not the dashboard.

### 11.3 Status and what is NOT claimed

- ✅ `d08d44b8` is **verified correct and discriminating**, by positive control.
- ✅ The false `batch_capacity: 4` on the dynamic pane is **fixed in HEAD**.
- ❗ **NOT fixed:** the two panes still differ by whether batching runs. That is a
  demo-composition question, not a server bug, and it is not mine to decide.
- 🚫 **NOT claimed:** *why* `qwen2.5-0.5b` lacks a continuous-batch manager. I
  measured that it does not engage; I did not establish the cause, and I am not
  going to infer one from a name.
- 🚫 **NOT claimed:** any throughput ratio between these arms. See the load range above.

**Reproduce:**
```
python3 qa-batch-width.py 8133 qwen-scatter    # expect peak in_flight 4
python3 qa-batch-width.py 8134 qwen-dynamic    # expect peak in_flight 1

```
