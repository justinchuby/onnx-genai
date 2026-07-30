# Reviewer brief

Read this before running anything. It lists the reds that are not ours, the fix
that landed on only one code path, the four fields whose captions are wrong, two
ratios that behave oppositely on purpose, and what we refused to ship.

Every claim below is stamped with the short HEAD it was verified at, read via
`git rev-parse --short HEAD` in the same shell invocation as the observation.
The tree moved under us repeatedly while this was written, which is why the
stamps differ between sections. A stamp makes a claim **dated, not true**.

Verify in the worktree whose `git rev-parse --show-toplevel` ends in
`onnx-genai-demo`. Several sibling checkouts of this repo exist on the build
box and do not contain `examples/serving-dashboard` at all; a check run in one
of those reports every file in this demo as absent.

---

## 1. Pre-existing reds that are not ours

**`cargo test --workspace` does not fail tests on arm64 macOS — it fails to
build** (`a721f033`). Two vendored C/C++ build scripts abort before any test
runs:

```
error: failed to run custom build command for `onnx-runtime-cpuinfo`
error: failed to run custom build command for `mlas-sys`
  cc-rs: command did not execute successfully:
  "c++" ... "--target=arm64-apple-macosx" ... qgemm_kernel_avx2.cpp
```

`mlas-sys` compiles an x86-64 AVX2 kernel with `--target=arm64-apple-macosx`.
`cmake` is installed (`/opt/homebrew/bin/cmake`), so this is not a missing
tool. Nothing in this demo touches either crate.

**Test the two crates this work actually changes; both build and run:**

```
cargo test -p onnx-genai-kv        125 passed, 0 failed        (a721f033)  GREEN
cargo test -p onnx-genai-server     13 failed                  (d6e57c63)
```

`onnx-genai-kv` — where paged-KV and all KV telemetry live — is **fully
green**. Every failure is in `onnx-genai-server`, and none is telemetry.

### The 13 server failures, by cause

**(a) Eleven need `*.onnx` fixture files that cannot exist in a clone**
(`d6e57c63`). Representative panic:

```
Failed to discover pipeline directory: IO error:
model.vision.filename file not found:
  .../onnx-genai-genai-config/tests/fixtures/vlm-executable/vision.onnx
```

plus nine `load fixture: failed to load model 'tiny-packed-vlm' from
target/vlm-image-bundle-tests/...`, which is under the gitignored `target/`.

> **Precise form of this claim, because the obvious version of it is false.**
> `.gitignore:3` is `*.onnx`, but **git ignore rules do not apply to files
> already tracked**, and 15 `.onnx` files *are* tracked (`50d412be`) — so
> "no `.onnx` file can exist in a clone" is wrong and one `git ls-files
> '*.onnx'` disproves it.
>
> The true statement is narrower: **`vlm-executable/` is tracked but its model
> file is not.** Four files are tracked in that directory — `config.json`,
> `genai_config.json`, `processor_config.json`, `tokenizer.json` — and
> `vision.onnx` is absent (`d6e57c63`; control: `vlm-complete/` → 7 tracked
> files, so the pathspec resolves). The fixture directory is half-committed.
> These tests cannot pass in any fresh clone and cannot be fixed from here.

**(b) Two are a real assertion failure and are NOT a fixture problem**
(`d6e57c63`) — `chat_completions_response_format_json_object_returns_valid_json`
and its `streaming_` twin, `tests/http.rs:860` and `:895`, both `left: 400`:

```
request admission exceeded the model context limit. Why: final prefill
length (3) after placeholder expansion plus max_tokens (14) is 17, above 16.
```

The tiny fixture model has a 16-token context and the test asks for 17. This
is a genuine red. It is unrelated to this demo, but do not let (a) absorb it —
it is a different defect with a different fix.

**(c) Skipped, not failed** (`a721f033`): `audio_endpoints_route_through_tiny_whisper_pipeline`,
`vision_request_routes_through_tiny_vlm_pipeline`, and
`qwen_real_model_tool_use_chain_end_to_end` are `#[ignore]`d and need `models/`,
which is gitignored (`.gitignore:2`) and therefore absent from this branch.
A skip here is expected, not a regression.

---

## 2. `/v1/resources`: fixed on one path only

The command-hang fix landed on the static/scatter path and **not** on the paged
path (`d6e57c63`, symbols re-resolved at `2582f5fb`).

> **Do not trust the line numbers below, including these.** Every driver.rs
> citation in the first version of this file went stale within thirty minutes —
> the file shifted about a hundred lines while this brief sat in the tree.
> Resolve them yourself; the command is the citation:
>
> ```
> grep -n 'continuous_batch_supported\|fn run_pipeline_driver\|fn run_fallback_engine_driver\|fn run_static_engine_driver' crates/onnx-genai-server/src/driver.rs
> ```

```
driver.rs:491  let continuous_batch_supported = engine.continuous_batch_manager(max_batch).is_ok();
driver.rs:500  kv_telemetry.set_applicable(!continuous_batch_supported);
driver.rs:502  if continuous_batch_supported { ... }

run_pipeline_driver          driver.rs:511
run_fallback_engine_driver   driver.rs:580   <- still stalls behind &mut Engine
run_static_engine_driver     driver.rs:586   <- fixed
```

> **The origin that now responds fast is the one whose prefix numbers are
> structurally zero. The origin that still stalls is the one hosting paged KV
> and the block table.** The fix and the value are on opposite servers.

Note `driver.rs:464`: KV telemetry is marked applicable when continuous batch
is **not** supported — the negation is deliberate, not a typo.

Consequence for review: panels for the paged-KV scenarios must bind to the
`KvTelemetry` atomics, not to `/v1/resources`. That is structurally safe —
every accessor takes `&self`, including the writer (`d6e57c63`):

```
kv/telemetry.rs:306  pub fn snapshot(&self) -> KvTelemetrySnapshot
kv/telemetry.rs:226  pub fn block_window(&self, ..) -> Vec<BlockState>
kv/telemetry.rs:149  pub fn set_applicable(&self, ..)
```

There is no `&mut` anywhere in that read path, so it cannot be blocked by the
exclusive borrow that stalls `run_fallback_engine_driver`.

---

## 3. Four fields whose captions are wrong

Each is measured, fresh, correctly typed, and moves when exercised. Our
five-state field machinery cannot flag any of them, because the value is fine
and **the label is wrong**. Provenance answers *where did this come from*; it
never answers *what is this*.

| Field | Caption implies | What it actually counts |
|---|---|---|
| `prefix_cache_hit_rate` | hits ÷ cache lookups | hits ÷ **completed generations** |
| `batch_size_current` | engine batch occupancy | **HTTP requests in flight** |
| `ttft` | time to first token from arrival | from **admission** — queue wait excluded |
| `vram.used` | device memory in use | **KV byte accounting** only |

**A fifth field is worse than mislabelled — it cannot ever arrive** (`d5c16fde`).
`dashboard/kv-memory.js` renders two eviction rows on adjacent lines:

```js
kv-memory.js:146  metricRow('hot evictions',    field('kv.hot_evictions'))
kv-memory.js:147  metricRow('prefix evictions', field('kv.prefix_evictions'))
```

`hot_evictions` is real and on the wire — `routes/mod.rs:733-737` declares it
and its own doc comment calls it *"the real pool-is-full signal."*
`prefix_evictions` is emitted by nothing:

```
grep -rc prefix_evictions crates/onnx-genai-server/src/  ->  0
grep -rc hot_evictions    crates/onnx-genai-server/src/  ->  5   (control)
```

Its only definition in the repo is `crates/onnx-genai-cli/src/profile.rs:484`,
in the **offline CLI profiler** — a different binary that never serves HTTP.
The row renders green in CI because `panels.test.js:164` supplies the value as
a fixture; live, it can only ever be blank. A test that supplies the data is
not testing whether the data exists.

That row should be `not-applicable` with the citation, or removed. It must not
be `pending`: `pending` means *not yet*, and this one is *not from here, ever*.

**`ttft` is the one to weigh most** (`metrics.rs`, verified `f45d7228`):
```
metrics.rs:113  pub(crate) fn start() -> Self {
metrics.rs:114      decrement(&REGISTRY.pending);        // leaves the queue
metrics.rs:117      started: Instant::now(),             // clock starts here
metrics.rs:124      REGISTRY.ttft.observe(self.started.elapsed());
metrics.rs:155      REGISTRY.e2e.observe(self.started.elapsed());   // in Drop
```

The clock starts on the same call that removes the request from the queue, so
**queue wait is structurally invisible to both `ttft` and `e2e`**. The error is
zero at one concurrent request and grows with concurrency — it is largest in
the 4-concurrent regime that carries our headline number, and it flatters us.
`request_started()` already exists at `metrics.rs:171` if you want the fix.

**The 2.46× headline is not affected by this.** It is an aggregate decode
*throughput* ratio (82.130 / 33.415 = 2.458, n=15, CV 1.98%), not a latency
measurement. Its receipt file is missing, which is a documentation gap, not a
measurement defect — see §6.

---

## 4. Two ratios that behave oppositely, on purpose

Do not "fix" the inconsistency; it is the point.

**`batch_utilization` clamps** (`d6e57c63`):

```
routes/admin.rs:76  pub(crate) fn batch_utilization(in_flight: u64, capacity: usize) -> f32
routes/admin.rs:80      (in_flight as f32 / capacity as f32).min(1.0)
routes/admin.rs:173     // The raw numerator, unclamped, so the client never has to invert a ...
```

It clamps because `in_flight` sums across drivers and can legitimately exceed
one driver's capacity; the raw numerator ships alongside so nothing is lost.

**KV utilisation must not clamp**, because `pages_in_use` can legitimately
exceed `hot_capacity`. The pool demotes an LRU page to a cold tier and grows
rather than refusing an allocation. Consequences a reviewer will misread:
`allocation_failures` is **structurally pinned at zero** and is not a health
signal; `hot_evictions` is the real pressure signal.

> **Correction to the evidence for this rule, since it will otherwise be cited
> wrongly.** The support for it is `kv/telemetry.rs:116`, which is a **doc
> comment** — *"May exceed `hot_capacity`: eviction demotes a page to the cold
> tier"* — not a test. `kv/telemetry.rs:264-271` is also prose (the
> `note_ref_count_change` docstring), and **no test anywhere asserts
> `pages_in_use > hot_capacity`** (`d6e57c63`; control: `pages_in_use` appears
> 24 times in that file, so the search resolves). This is stated *intent*,
> correctly labelled as such. Treat it as a design decision that is documented
> and unverified, not as a behaviour that is proven.

---

## 5. Two servers is a design decision, not a workaround

The demo runs **two server processes on two ports**, one per model. This looks
like something we failed to clean up. It is load-bearing.

All metrics live in one process-global struct with **no per-model dimension**
(`5650597c`):

```
crates/onnx-genai-server/src/metrics.rs   static REGISTRY: Registry { ... }
  13 fields: requests, prompt_tokens, completion_tokens, ttft, e2e,
  active_sessions, pending, batch_size, prefix_cache_hits,
  prefix_cache_hit_tokens, prefix_cache_lookups, rejections, trace_ids

grep -cE 'with_label_values|const_label|LabelPair'  ->  0
  (control: 'fn ' -> 24 in the same file, so the zero is a real absence)
```

Under one server serving both models, **every one of those 13 fields sums the
two populations**. A latency histogram fed by a batching model and a dynamic
model is not a noisy measurement of one thing; it is a blend of two
distributions with no physical referent.

Two of those fields — `ttft` and `e2e` — are the inputs to the headline
number, and that baseline was taken single-server, single-model, with nothing
else resident. One server would not have made it noisier, it would have
invalidated the conditions it was measured under, **from inside the
instrument**, where a back-to-back A/B protocol cannot detect it.

> Two processes give all 13 fields a model dimension enforced by the operating
> system, instead of by thirteen fields' worth of reviewer vigilance. We chose
> the topology that makes the bug unrepresentable over the one that makes it
> forbidden.

Note the interaction with the request-path `model:` badge: the badge is
accurate — the request really did go to that model — but `/metrics` and the
global registry are **not on the request path**. Under one server the badge
would be a true label beside a blended number, which is worse than no label,
because checking it succeeds and ends the investigation.

---

## 6. What we refused to ship, and why

**The prefix-cache hit rate is withheld.** Both of its terms are broken: the
numerator counts a hit that saved no work, and the denominator is completed
generations. We ship `prefix_tokens_reused` instead — an absolute count with
no denominator to be wrong about (present in `routes/admin.rs`,
`routes/mod.rs`, `d6e57c63`). The panel states the finding in words instead of
rendering a number, and no percentage appears in any form.

The underlying finding is a **reachability** result about this configuration,
not a claim that prefix caching is broken: the branch our models take computes
a token overlap and never writes the variable holding restored prefix length
(`runtime.rs:1028`), and any model with a decode runner takes that branch
(`decode/state.rs:206`).

**Scenario C ships pressure-driven degradation only.** The VRAM-limit route is
not shipped because `set_vram_limit` concedes in its own comment that the
computed eviction order is never carried out (`87a80c0c`). Four functions share
that name; the one that confesses is in the **engine** crate, not the server:

```
crates/onnx-genai-engine/src/engine/governor.rs:164  pub fn set_vram_limit(
:172   // TODO(§26.11.2): execute the returned priority/offload/eviction order
:173   // across live engine sessions when the outcome reports an overage.
:174   Ok(self.inner.set_vram_limit(limit)?)
```

The call returns `Ok` with a computed outcome that nothing acts on. A related
concession sits at `engine/governor.rs:128`. This is the strongest artifact we
have in the overclaim class — it is the source refusing to overclaim about
itself, and it is why the scenario ships the half we can demonstrate.

---

## 7. Current state of the JavaScript suite

```
cd examples/serving-dashboard && node --test
pass 479   fail 1        (2582f5fb)
```

**The single failure is a new check catching real defects, not a regression.**
Do not make it pass by weakening it:

```
check-source-citations.test.js:357
  "a cited line still sits beside the symbol the prose names"
  3 citation(s) name a symbol that is no longer at the cited line:
    README cites driver.rs:717 for handle_or_defer_during_batch -> now :734, :752
    README cites driver.rs:794 for handle_driver_command        -> now :581, :626, :829
    README cites driver.rs:816 for run_fallback_generation      -> now :851, :935
```

An older version of this check verified only that the cited line number fit
inside the file, so it caught a citation going *short* and never one going
*stale*. This one resolves the symbol. It went red on its first run and found
three. Fix the three citations in `README.md`; the assertion is correct.

For context, the field-state suite was `pass 471 / fail 0` at `43eff6fd`
(run twice, same tree state). The `ok` → `measured` migration is fully landed:
`FIELD_STATES.MEASURED` evaluates to `'measured'`, `FIELD_STATES.OK` is
`undefined`, and `styles/shell.css:163` is `[data-state='measured']`. Any
message or document telling you otherwise predates the migration — including
`demo-spec.md`, see below.

Two notes for anyone re-running:

- Node prints `ℹ pass` / `ℹ fail`, not `# pass`. A summary grep anchored on
  `^#` matches nothing and produces an empty summary next to a non-zero exit,
  which is indistinguishable from a catastrophic failure.
- The suite is hermetic but prints production-voiced warnings naming
  `http://127.0.0.1:8123`. `telemetry-store.test.js:74` injects a `fakeFetch`
  at six call sites. Those alarming lines are tests **passing**.

**Two gaps that were open when this was first committed. One has since closed:**

- ~~`perf-baseline.md` does not exist in any branch.~~ **CORRECTED — it landed
  in `87a80c0c` and is tracked** (`7d528de7`). This claim was true when written
  and expired about ten minutes later. The file carries what the measurement
  needed: `n=15`, `CV 1.98%`, per-repetition tables with stdev, and the
  derivation `82.13 / 33.41 = 2.46×` at `perf-baseline.md:93`. `demo-spec.md`
  landed in the same commit and is also tracked.
- The dashboard has been verified served (`/demo/` and all eight JS modules
  return 200 over HTTP) only **at rest**, with no generation in flight. Three
  known defects — the occupancy gauge, `ttft`, and the block grid — are all
  invisible in that state by construction. A page checked at rest is checked
  in the one state that hides them. Treat the served-page evidence as
  incomplete until the panels have been watched during an active generation.

### A caution about `demo-spec.md`, now that it is in the tree

`demo-spec.md` is normative and it currently contains a false claim about this
codebase, stated three times in escalating emphasis (`7d528de7`):

```
demo-spec.md:1245  styles/shell.css:163   [data-state='ok']  { … }
demo-spec.md:1254  the [data-state='ok'] selector change in one commit
demo-spec.md:1303  "AND FOR THE THIRD TIME: ... styles/shell.css:163 is [data-state='ok']"
```

The actual line (`sed -n '163p' styles/shell.css`, `7d528de7`) is:

```css
[data-state='measured'] {
```

The migration completed; the spec indicts a defect that was repaired. Do not
act on those three passages, and do not treat the repetition as corroboration —
all three are one unrefreshed observation. **Where the spec and the source
disagree about the source, the source wins.**

**Request the URL with the trailing slash: `/demo/`.** Without it the module
imports resolve against `/`, every `<script type="module">` 404s, and the page
renders blank with only a console error to show for it.

The server binary takes **`--addr 127.0.0.1:PORT`**; it has no `--port` flag.
`scripts/verify_model.sh` does take `--port` and translates it internally, so
the same token is correct for the script and rejected by the binary.
Pass `--demo-assets-dir` as an **absolute** path — its default is relative, so
a bare launch serves a healthy API with a dead `/demo` from any other
directory.
