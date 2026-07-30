# Reviewer brief

Read this before running anything. It lists the reds that are not ours, the fix
that landed on only one code path, the four fields whose captions are wrong, two
ratios that behave oppositely on purpose, and what we refused to ship.

Every claim below is stamped with the short HEAD it was verified at, read via
`git rev-parse --short HEAD` in the same shell invocation as the observation.
The tree moved under us repeatedly while this was written, which is why the
stamps differ between sections. A stamp makes a claim **dated, not true**.

---

## 0. Stand in the right worktree, or every answer below is wrong

Everything in this document refers to one checkout:

```
/Users/justinc/Documents/GitHub/onnx-genai-demo      branch: feat/genai-demo-dashboard
```

**More than one checkout of this repository exists on the build box.** They are
registered git worktrees on unrelated branches, each with its own
`crates/onnx-genai-server/src/lib.rs` at different contents. One of them is
parked on an older commit and does not contain `examples/serving-dashboard`
at all.

This matters more than it sounds. A command run in the wrong checkout does not
error — it answers, confidently, in the correct format, about a different
universe. A parked checkout produced false negatives for five separate people
in one evening, and each command looked clean and well-formed. Some of those
answers were false *positives*: the parked tree still contains files that were
deleted here.

Put this at the top of any command block before you trust its output:

```bash
git rev-parse --abbrev-ref HEAD    # must print: feat/genai-demo-dashboard
```

Two more command shapes that silently re-anchor to wherever you are standing:

- `git diff <sha>..HEAD` — in a checkout parked at `<sha>` this diffs a commit
  against itself and returns a clean, confident, reassuring **empty**. Name both
  endpoints instead: `git diff <sha> <branch> -- <paths>`.
- Existence checks. `ls` and bare `grep` answer about a directory. `git ls-files
  <path>` answers about the repository, and survives being run in the wrong one.

## 0a. Cite symbols, not line numbers

Every instruction passed around this session that carried a **line number**
regenerated — several of them four to eight times, each honest when written.
Every instruction that carried a **symbol name** landed once and stayed.

This document is the longest-lived thing we are producing, so it is where
position-addressing rots fastest. Where a line number survives below, treat the
**symbol as authoritative and the number as a hint**. If they disagree, the
number is stale. Section 2 prints the `grep -n` that regenerates its own
citations for exactly this reason.

The same rule applies to counts. Do not quote the length of the specification,
the number of acceptance criteria, or a test tally you read in a message. The
spec is a file; read the file. Thirteen different acceptance-criteria counts
circulated as fact in one evening, every one of them honest when taken.

The specification is **append-only**, so identifiers are stable even though
ranges are not: cite `AC52`, never "the last ten ACs."

## 0b. A sha does not identify what you tested

Recording the sha is necessary and not sufficient. Six consecutive runs of the
JavaScript suite at one fixed HEAD, minutes apart, produced three different
results — not flakiness, but uncommitted work appearing and disappearing in a
shared working tree.

Quote `git status --porcelain` alongside the sha, or you have dated a claim
without identifying its subject.

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

**(d) The build-script suite's model-fidelity evidence: runs here, will NOT run
for you unless you have a models directory.** This one is different from (c) in
a way that matters, so please do not read it as more of the same.

`scripts/build_qwen_test.sh` reports **81 tests, 0 failed, 0 skipped** on a
machine that has a checkout containing `models/qwen2.5-0.5b-scatter-v2`. It
finds one via `scripts/lib/models_dir.sh`, which searches `<repo>/models` and
then the sibling `../onnx-genai/models`, accepting a candidate only if it
actually **contains** `model.onnx` — an empty-but-present `models/` holding
only `.hf_cache` and `.scratch` is the normal state of a fresh worktree, and a
plain directory-existence test selects it and defeats the fallback.

**On a machine with no models anywhere — a fresh clone, or CI — three checks
skip, and they are the strongest three in the file:**

| Check | What is not verified when it skips |
|---|---|
| `generator reproduces the 24-layer scatter model io block` | that the generator matches a real 24-layer export rather than only the 1-layer fixture |
| `cache ports are ordered numerically by layer, not lexically` | that `key_cache.10` sorts after `key_cache.2` |
| `generator rejects a dynamic-cache model` | that a model with no scatter ABI is refused instead of given a bogus declaration |

The middle one is the load-bearing one: lexical ordering silently mis-pairs
every buffer past the ninth layer, and **the 1-layer fixture cannot detect it,
by construction.** All three are mutation-proven — changing the generator's
`sorted(...)` to a lexical sort turns them red.

You do not have to remember any of this. **A run that skips them prints a
banner naming the lost evidence and the single command that restores it**
(`MODELS_DIR=/path/to/onnx-genai/models scripts/build_qwen_test.sh`). The
banner exists because relying on a reviewer noticing three `skip` lines
scrolled past seventy `ok` lines is not a control — it is a hope, and it is the
same "visible if you look" that let `panels.css` survive eight reports.

> **The residual gap, stated plainly: `0 skipped` on this branch is a claim
> about the machine that ran it, not about the branch.** If your run says
> `3 skipped`, the suite is telling you the truth and the model-fidelity
> evidence is genuinely absent from your run — not weakened, absent.

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

**The headline is not affected by this.** It is an aggregate decode *throughput*
ratio, not a latency measurement.

⚠️ **But do not quote it as a clean win, and do not quote it from this file.**
The aggregate gain ships **only** alongside the per-stream cost: **per-stream
throughput falls to about 0.62× of solo** (~20.7 tok/s). Batching makes no
single request faster; it trades per-stream latency for total throughput.
`demo-ux.md` §29.1 ratified that both halves appear together, everywhere — a
tradeoff presented as a pure win is a lie told with true numbers.

**The receipt now exists: `perf-baseline.md` is tracked** (landed in `87a80c0c`),
and it is the only place the figure should be read from. It gives the ratio as
**≈2.5×, 95 % CI [2.35, 2.59]** from raw per-run samples. This section
previously printed `2.46×` and `82.130 / 33.415 = 2.458` — **two different
values for one quantity, neither carrying an interval**, which is exactly the
hand-maintained duplication that produced the drift. Cite the file, never the
number.

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

The suite is a command, not a number. Run it, and state the command **and your
working directory** alongside any result you report:

```
cd examples/serving-dashboard && node --test
```

That invocation is the only total the release gate accepts. Anything narrower —
a single file, a subdirectory — is a subset and must say so when reported. The
pass count has moved repeatedly in a single evening while the tree changed
underneath it, which is why no number appears here.

Two cautions about running it. Node's `--test` recurses, so the count of files
it collects is larger than the top-level directory suggests; and consecutive
runs at one fixed sha can disagree if uncommitted work is moving in the shared
tree. Quote `git status --porcelain` with the sha.

**If a check named below is red, do not make it pass by weakening it.**
`check-source-citations.test.js` was written to catch citations that go *stale*
rather than merely out of range: it resolves the symbol the prose names and
confirms the cited line still sits beside it. It went red on its first run and
found three real defects in `README.md` — citations for
`handle_or_defer_during_batch`, `handle_driver_command` and
`run_fallback_generation` whose symbols had moved. An older version of that
check verified only that the cited line fit inside the file, so it caught a
citation going short and never one going stale. Fix the citations; the
assertion is correct.

The `ok` → `measured` migration is fully landed:
`FIELD_STATES.MEASURED` evaluates to `'measured'`, `FIELD_STATES.OK` is
`undefined`, and `styles/shell.css` selects `[data-state='measured']`. Any
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
  derivation at `perf-baseline.md:93` (≈2.5×, 95 % CI [2.35, 2.59]). `demo-spec.md`
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

---

## 8. Six things we learned building this, in the order they will bite you

### 8.1 The commit log will tell you the opposite of the truth in at least four places

**Read diffstats. Never read subjects.** Commit subjects misdescribed their own
contents repeatedly in this session, and the reviewers most likely to be misled
are the careful ones reconstructing history from `git log --oneline`.

The clearest specimen, reproducible today:

```
$ git log -1 --format='%h %s' 54d8ba5a
54d8ba5a docs(demo): state the five field states plainly, dropping the hedge

$ git show --stat 54d8ba5a
 crates/onnx-genai-server/src/cors.rs | 212 -----------------------------
 examples/serving-dashboard/README.md |   7 +-
 2 files changed, 5 insertions(+), 214 deletions(-)
```

A commit announcing itself as a documentation wording change deleted a
212-line router-wired Rust module and its tests. Another commit describing a
docstring fix carried several hundred insertions of Rust KV telemetry across
two crates.

The mechanism was `git add -A` in a worktree shared by several people: it
sweeps whatever is in the index, so **the commit message and the diff can come
from two different authors**. Nobody mislabelled anything; the label was applied
by someone who never saw the change.

Two consequences for you. Searching the log for a feature by name will fail
even when the change is present — a deleted file's history needs
`git log --diff-filter=D -- <path>`, which you would only run if you already
suspected the answer. And a bisect will land you on a docs commit.

### 8.2 Run the thing; do not test for the artefact

The single highest-yield habit here. One module import settled in about a second
a question that filesystem checks got wrong in *both* directions across an hour:

```bash
node -e "import('./telemetry-field.js').then(m => console.log(m.FIELD_STATES))"
```

Grep was wrong in both directions because the file contains comments *about* an
old bug, quoting the very strings being searched for. Executing the module reads
what the program reads.

The general failure: an existence check answers a question next to the one you
asked. A file can exist and never be requested by the page. A stylesheet can be
present and unlinked. A health probe can return success from **someone else's
server on the same port** while yours is dead.

### 8.3 Prove the mutation landed

A check that has never been seen red is indistinguishable from a check that
cannot fail — both produce a green line. So break it deliberately, watch it
fail, restore, and **state the mutation you applied**.

Extend this to the checker itself. An audit that silently under-matches returns
a clean bill of health, which is the one failure mode an audit must not have and
the one that looks exactly like success. Run a positive control before believing
any zero: search for something you know is present, and confirm the tool reaches
the files at all.

### 8.4 Nothing here enforces prose

Our tests cover code, field names and wire values. They do not cover design
specs, READMEs, meta tags, doc comments, commit subjects, or approvals given in
conversation — and prose is where the last of the false claims were found,
because it names features in plain English rather than as identifiers.

Doc comments are the sharpest case: they sit inside source files and inherit the
authority of code while being unable to fail. One comment reading *"the engine
does not yet expose KV page statistics"* was false when written and caused work
to be **skipped** rather than merely misread. Nobody audits the absence of code.

When you check a corrected fact, grep the **shipping copy** — the page, the
README, the strings a visitor sees — and the tests that quote it. A correction
that lands where the argument happened, rather than where the text ships, leaves
the only sentence a human reads still wrong.

### 8.5 Fabricated doubt is as serious as fabricated confidence

Every honesty mechanism in this project points one way: it guards against
claiming a capability we lack. Nothing guarded against overclaiming that
something is **absent**, and a strong false negative walked into the lead of our
own honesty document unchallenged.

Nobody argues with the person claiming less, so conservative errors receive a
fraction of the scrutiny — and they cost the same credibility. An expert who
greps a symbol, finds it wired with a test asserting its counter rises, and then
reads "so this never happens" concludes we did not read our own code, in the one
sentence whose whole job is proving that we did.

Correcting an overclaim by installing the opposite underclaim is the same error
with the sign flipped, and it is harder to spot because it sounds modest.

### 8.6 A measurement is a claim about a binary, and binaries do not expire loudly

Two people independently measured a working feature, correctly, from a binary
built during the four minutes that feature existed in the tree. The results were
real. The subject was gone.

So when you report a runtime result, **cite the commit you built, not the time
you built it**. This has one consequence worth stating plainly, because it is the
place it would do the most damage: a performance comparison whose "after" arm was
built before the instrumentation landed returns a genuine, arithmetically
correct, beautifully tight **0% overhead** — the answer everyone is hoping for,
which is why nobody would question it.

The same applies to observations of a running system. A dashboard checked at rest
is checked in the one state where several defects are invisible; peak-zero — zero
at maximum load, not zero at idle — is the observation that finds them.

### 8.7 Verify your artefact is inside a repository at all

We ratified "verify your commits exist." There is a rung below it that cost us
more.

A failed commit leaves a dirty file, and `git status` will show it to you. A file
written **outside** the repository produces a perfectly clean `git status` —
byte-identical to the output of work that committed successfully. No index entry,
no diff, nothing to notice. The proudest field in our provenance stamp,
`--porcelain 0`, is also the exact signature of work that was never in the
repository.

A census near the end of this session found thirteen documents over 2 KB, roughly
830 KB in total, written by eleven contributors over eight hours, that `git
ls-files` could not find. Two of them had been announced as delivered minutes
earlier. One was a complete review deliverable; one was the evidence for a
release-gate item.

So the family of claims a reviewer should keep separate is longer than it looks.
**Exists · is inside a repository · is committed · is wired · is reachable from a
branch · is in the right checkout.** Every one of those looks identical when it
passes, and this session lost time to all six.

The only instrument that distinguishes them is `git show HEAD:<path>` followed by
counting something you already know the answer to. `git ls-files` proves a *path*
is tracked; it cannot prove the bytes at that path are yours.

### 8.8 A check satisfiable only by a defect is a false-red generator

This one is ours, found in the release gate itself and not in the product.

One gate criterion read: *a live 4-concurrent generation on the **dynamic**
origin returning a non-zero batch size*. The dynamic origin is the per-request
engine path — it returns a batch size of exactly 1, by construction, and that is
the arm demonstrating batching does **not** occur. One is non-zero. So the
criterion was satisfied precisely by the configuration that proves the feature
absent, and would have been violated by a working build.

It never fired, because the item closed on other evidence. That is the part worth
carrying: **a defective check that gets routed around is indistinguishable from a
correct one.** It has no failures to its name and no successes either. Nothing in
a green run, or in a closed checklist, marks the difference.

The mirror of this is the pattern documented throughout §8: a check that cannot
fail, and a check that can only fail wrongly, are the same defect measured from
opposite ends.

### 8.9 Two instruments that share an input do not corroborate — they echo

Several confirmations in this record are weaker than they appear, and it is worth
knowing which kind you are reading.

Our citation checker and its repair script both `readFileSync` the **working
tree**. They agree with each other by construction, and can only disagree with
the tree you will clone. Separately, one guard reports different results in a
clean checkout and a dirty one — it audits whatever is on disk, including files
that are not in the repository and will never ship. It reddened on an orphaned
document and would have gone green if someone deleted that file without fixing
anything.

The same applies to people. Two reviewers using different methods at the same
commit are independent in *method* and correlated in *time*; two reports at one
sha are one report wearing two names. The check is
`git merge-base --is-ancestor <their-sha> <my-sha>` — if one confirmation's
commit is an ancestor of the other's, it is not a second data point.

And the corollary that surprised us: recency is evidence about *staleness*, not
evidence about *correctness*. Late in this session the freshest reading of a file
was the only wrong one, and it nearly retired a live blocker.

For a runtime result the equivalent of "retrieve it from HEAD" is: **identify the
binary by its behaviour in the same invocation as the measurement.** `ps` cannot
tell you which code is executing — two server pairs here shared a binary path
while the older processes held the older inode, and only a payload field
distinguished them.
