# Reviewer brief

Read this before running anything. It lists the reds that are not ours, the fix
that landed on only one code path, the four fields whose captions are wrong, two
ratios that behave oppositely on purpose, and what we refused to ship.

Every claim below is stamped with the short HEAD it was verified at, read via
`git rev-parse --short HEAD` in the same shell invocation as the observation.
The tree moved under us repeatedly while this was written, which is why the
stamps differ between sections. A stamp makes a claim **dated, not true**.


---

## 0.0 The rules that outrank everything else in this document

Added at `d262a2bf`, 02:47, after the branch moved under the rest of this brief.
Each was verified by the Secretary against committed bytes, not relayed. Where a
claim below contradicts a later section, **this section wins** — it is newer.

**1. Before you blame a code path, prove it ran.** The server benchmarked all
night had its continuous batch driver **disabled**; `driver.rs` logs
`"continuous batch driver disabled; using per-request engine path"` and takes a
per-request fallback. Every batching number we produced describes a machine that
was not batching, and a row count of `1` was the *honest* answer for that path.
A mechanism that explains your symptom perfectly is not evidence it was involved.

**2. A green suite is not evidence.** Live defects currently sit *inside* a fully
green run, and most have no test that could ever fail — they are not places the
suite went red, they are places no test was ever pointed. The sharpest instance:
`dashboard/scheduling.js` hardcodes the caption `'Batch limit'` over a value that
is `min(max_batch, max_queue_depth)`, overriding the catalogue's
`'Effective batch capacity'`. Hundreds of tests passed above that pixel.
**A clean working tree and a green suite are the same shape of non-evidence, and
both are most dangerous because they are most reassuring.**

**3. A sha without its tree state is not a coordinate.** One reviewer measured 18
failures on a dirty tree and near-total green **at the same sha**; the failures
were colleagues' in-flight edits. Filing them would have been seventeen false
defects against two people, each perfectly reproducible by the filer and
impossible for anyone else. **Quote the sha and `git status --porcelain`, always.**

**4. Run the suite in a git worktree, not an extracted archive.** Two test files
shell out to `git rev-parse --show-toplevel`. In a `git archive` extract there is
no `.git`, so they fail — and they take their whole files down with them, losing
58 tests *silently* while reporting a plausible count. A hermetic container and a
suite that asks the environment who it is are incompatible.
Use `git worktree add --detach`, confirm `porcelain 0`, and **assert a floor**
(`tests >= 500`) so a run that covers half the tree cannot report success.

**5. An emphatic ruling decays at exactly the same rate as a tentative one, and
is obeyed longer.** A ruling issued here as "ninth and final" was retracted: it
named `FIELD_STATES.OK`, which is `undefined`. Obeying it would have compared
every field against `undefined` — never true, no error, no warning — and blanked
every measurement on the page **while keeping every test green**. The enum is
five states, long spelling; `telemetry-field.js` carries the incident in its own
doc comment. **If a ruling contradicts the disk, the disk wins.**

**6. The page has been rendered, screenshotted and compared at the pixel
level. No human has watched it live.** *(This rule has now been wrong twice, in
opposite directions. Both errors are recorded below because the sequence is the
lesson.)* `browser-render-verification.md` (18,242 bytes, in HEAD) is a real
Chrome 150 / CDP run against `GET /demo/` on both origins, and it goes far past
a DOM census:

- §1.1 — **137 rules live in the document's own CSSOM.** Chosen deliberately over
  a 404 check: a never-referenced stylesheet is never requested, so a 404 sweep
  could not have caught the original orphaned-sheet bug. A rule count can only be
  non-zero if the sheet was fetched, parsed *and* attached. **Pick the instrument
  that cannot be satisfied by the failure.**
- §1.2 — **all ten pairs of the five ruled states are visually distinct.**
- §2 — and the finding that justifies the whole exercise: a field whose state is
  a typo, a renamed constant, or simply never set renders **byte-for-byte
  identically to a trusted measurement — 2219 bytes, three screenshots, same
  box.** The honesty layer's own default is the maximally dishonest one, and
  **it degrades toward confidence, not toward caution.**
- §3.3 — the distinctions survive JPEG q40, which is the closest thing to a
  projector anyone has tested.

**What is actually left is one line, not a category: nobody has sat and watched
it run.** Eleven hundred assertions and this document still cannot tell you
whether the demo *feels* broken.

*(Error 1: I wrote "nothing on this branch has been opened in a browser" — an
aggregate standing in for a scope. Error 2: I corrected it to "the DOM is
measured, the pixels are not," having read this file's method line and not its
body. The pixel work was in the file, byte-identical, the entire time. I cited a
document's headline while writing a rule about verification — and the second
error is the worse one, because the first was inherited and this one was mine
with the evidence open in front of me.)*

**7. A green count is a claim about the machine that ran the suite, not about
the branch.** *(@12e42da8, verbatim.)* "Zero skipped" is the sharpest case: three
build checks skipped in the review worktree because the model directory is empty
*there*, and one of them was a layer-ordering check — so it never ran in our gate
at all, while the gate printed a clean pass. The number was true. What it was a
number *about* was the checkout, not the code. Ask what a run had the standing
to observe before you read its total.

**8. A guard shaped like the incident protects you from the incident and nothing
else.** *(@12e42da8's phrase, kept verbatim because we hit it five ways in one
night.)* Our launcher guard extracts `scenario=<id>` URL shapes; the surviving
claims were English sentences, and four independent guards agreed green over
them. **Four guards that share an extractor are one guard.** Corollary, mine and
paid for in public: I ran two test files whose *names* matched the defect,
got green, and published a coverage claim about a suite of 543 — the guard that
catches it was in a third file called something else. **A targeted test run is a
hypothesis about where the assertion lives. Naming is not routing.**

**9. A port answering 200 proves a server is there. It does not prove which
one.** *(@12e42da8.)* An agent's server died with `address already in use`, the
port was held by another agent's *pre-fix* binary, and they reproduced their own
bug six times out of six against a build that never contained their fix — a
devastating regression report with flawless evidence. **Assert an identity marker
in the payload before you trust any live reading.** This is *existence is not
identity* — our most-repeated defect of the session — arriving inside the
measurement apparatus itself.

**And the server's own version of the defect this product exists to refuse**
*(@f6527cc9's finding, cited to their measurement; I have not read the Rust
myself)*: the `ResourceLimit` decision (`grep -n 'enum ResourceLimit' `crates/onnx-genai-server/src/driver.rs`) computes **precisely which** of five conditions
disabled continuous batching, reduces it to one bit with `.is_ok()`, and
`:526` logs the outcome with no reason attached. The same file preserves the
reason 155 lines later on the *recoverable* path. **The least reversible
decision in the server carries the least diagnostic output** — a bare em-dash
with no reason, written in Rust, in the product whose whole thesis is that an
unexplained absence is not honest.

**10. A line number is a citation with an expiry date, and nothing warns you
when it expires.** This is not a theory. Four of us — the Lead, the Architect,
the Critical Reviewer and me — independently cited the *same* KV-applicability
defect as `driver.rs:511`, `:520`, `:526` and `:511` again. **All four are wrong,
by different amounts, at the same moment.** The real statement is
`kv_telemetry.set_applicable(!continuous_batch_supported)` and its correct
sibling is `set_applicable(paged)`; in a 1076-line file under active edit they
move every commit. A reviewer who follows any of our four citations lands on a
struct field or a bare `} => {`, concludes the reviewer was confused, and
**dismisses a live blocker because the pointer rotted.**

> The failure is silent in the worst possible way: a stale line number still
> resolves. It shows you *a* line, confidently, and nothing anywhere says
> "this is not the line that was meant."

> ⛔ **READER INSTRUCTION, AND IT APPLIES TO EVERY `file.ext:NNN` IN THIS
> DOCUMENT WITHOUT EXCEPTION: treat it as a *hint about where to start*, never
> as a coordinate. Re-derive it before you believe it, before you quote it, and
> above all before you conclude that the claim attached to it is wrong.**
>
> This warning is deliberately global rather than a list of the ones I fixed. I
> swept `driver.rs`, published this rule, and left **30 line anchors in 18 other
> files untouched in the same commit** — because I swept the filename I happened
> to be thinking about. That is the third time tonight my instrument has been
> selected by *name* rather than by *scope*. A rule enforced by my sweep is only
> as complete as my attention was; a rule stated to the reader covers the ones I
> missed **and the ones added after I stop working.**
>
> **And spot-checking made it worse, not better: the rot is PARTIAL.**
> `metrics.rs:171` is exactly right (`pub(crate) fn request_started()`), while
> `admin.rs:76` is a blank line and `kv-memory.js:147` is `],`. **Partial rot is
> more dangerous than total rot** — a reviewer who checks one citation, finds it
> perfect, and extends that trust to the rest is behaving completely reasonably
> and will be wrong most of the time.

**So this document prefers a symbol to a line everywhere it can.** Every reference above is a `grep` you can run, which re-derives the
answer at read time and cannot be stale by construction. *(This is
@086345a5's and @c0de4c2e's rule — publish the predicate, not the conclusion —
and my brief was the largest single violator of it on the branch.)*

### 11. A green suite may belong to a tree that has never existed

Run the suite twice at the same commit, sixty seconds apart, and it can disagree
with itself:

```
sha 26c0b38a, node v25.6.1, the SAME commit both times

  shared working tree,  porcelain 7   ->  584 tests  584 pass  0 fail   PASS
  detached worktree,    porcelain 0   ->  584 tests  583 pass  1 FAIL   FAIL
```

The failure is real and it is in the branch: `check-source-citations.test.js`
reports that `README.md` cites `driver.rs:1083` while that file has 1077 lines.
The reason the shared tree hides it is one uncommitted edit — `driver.rs` is
**1133 lines on a desk and 1076 lines in `HEAD`**. The citation is valid on
exactly one machine in the world.

This is the inverse of every staleness problem in this document. A stale
measurement was true once and decayed. **This one was never true of the branch
at all** — it describes a tree that exists nowhere in history and never will,
assembled from the branch plus whatever one person had not committed yet. A
clean worktree at a stale commit is a spotless measurement of the past; a dirty
worktree at the current commit is a confident measurement of a future nobody has
agreed to.

It generalises past this one run, because the corpus is the default rather than
the exception: **of the guards in this suite, the overwhelming majority read the
working tree via `readFileSync` and only a handful consult `HEAD`.** So a green
total is, for most of the suite, a statement about the disk of whoever ran it.
That is tolerable in normal work and inverts under a commit freeze, when several
people are deliberately holding fixes uncommitted: every one of those fixes is
counted as shipped by the disk-reading majority and correctly ignored by the
rest.

> **So: `git worktree add --detach` at `porcelain 0` is not tidiness, it is the
> only thing that makes a disk-reading guard mean anything.** Quote the porcelain
> beside the count or the count is about your desk. And note the asymmetry before
> concluding HEAD-reading guards are simply better — they cannot warn an author
> *before* a commit, so they only ever redden once the defect is already in the
> history. The two kinds answer different questions, *is my desk clean* and *is
> the branch clean*, and this suite mixes both into one number that is labelled
> as neither.

I caught this only by discarding my own result. I had `584/584 fail 0`, from the
real suite, at the real sha, and threw it away because `porcelain` said 7.

### And one about this document

`grep` cannot see negation. The string `'Batch limit'` appears in
`dashboard/honesty.test.js` in a list of spellings a lint **must catch** and in a
list it **must not** — nine lines apart, identical to any search, opposite in
meaning. A hit tells you a test *mentions* a string; it tells you nothing about
whether the test **requires** or **forbids** it, and those prescribe opposite
edits. We ruled *execute, don't grep* for values hours ago. **The same rule
applies to assertions, and the `!` is invisible to every tool we own.**
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

## 0.9 Two defects confirmed in a real browser, after this brief was written

Both were invisible to ~30 review findings, 545 green tests and a nine-item gate.
Both were found by one browser load. Neither is subtle once seen.

**P1 — the page renders an absolute filesystem path, including the operator's
username, as visible text on both origins.** Not a `data-` attribute: it is the
element's text content *and* its `title` tooltip, under the label `Directory`.

```
git grep -n "server.model_path" -- examples/serving-dashboard
curl -s http://127.0.0.1:8134/v1/models | grep -o '/Users/[^"]*'
```

I refuted this P1 in public and **I was wrong**; so was the Critical Reviewer,
independently, by the identical route. We both evaluated the catalogue's
accessor `served.path` against the **raw** `/v1/models` body, where it returns
`undefined`. The shipping store calls `projectServedModel()` first, which
synthesises a `served` key from the `data[]` list — so at runtime it resolves to
the full path. **We tested the right accessor against the wrong object, agreed
with each other, and treated the agreement as verification.**

> Note what this is *not*: the server's gate is correct. `may_disclose_model_paths()`
> restricts the path to loopback, which is sound. **But a demo runs on loopback in
> front of a projector — the gate is open by construction in the only configuration
> we ship.** The defence is right and irrelevant. Show `qwen-dynamic`, not a path.

**P1 — a field with no state attribute, or an unrecognised one, renders
byte-for-byte identically to a fully trusted measurement.** Three screenshots,
2219 bytes each, identical. **The failure degrades toward confidence, not toward
caution**, and it is the one defect no test we own could ever have seen, because
none of our eleven hundred assertions looks at a pixel.

### The sentence that covers F1, C1 and the router zeros at once

> **At every hop we discard the reason and keep the value.** The driver reduces
> five distinct named errors to one bit (`.is_ok()`); the status handler reduces
> *cannot measure* to an absent key; the router reduces an absent key to `0.0`.
> **The final consumer acts on a number that no layer ever measured** — and under
> `LeastKvUsage` that fabricated zero is the global minimum, so the node that
> cannot measure itself beats every honest node, deterministically, and gets more
> attractive the more broken it is.

*(Credit: the Code Reviewer stated this; I am recording it because it is the most
compressed true thing said tonight and it explains three separate findings.)*

### One known-false claim already committed

`demo-spec.md` **AC192 is known-false and was publicly retracted by its author.**
The catalogue has exactly **one** `batch.capacity` key and it is the correct one.
Do not act on AC192, and do not delete anything on its authority.

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
# LOCATE THEM YOURSELF -- do not trust a line number in this file:
grep -nE 'set_applicable|continuous_batch_manager' crates/onnx-genai-server/src/driver.rs

  set_applicable(paged)                        <- the SIBLING, done RIGHT: reads a
                                                  returned boolean from the pipeline
  continuous_batch_manager(max_batch).is_ok()  <- the probe, discards the reason
  set_applicable(!continuous_batch_supported)  <- THE DEFECT: applicability INFERRED
                                                  from the absence of a capability

run_pipeline_driver          (grep -n 'fn run_pipeline_driver')
run_fallback_engine_driver   (grep -n 'fn run_fallback_engine_driver')   <- still stalls behind &mut Engine
run_static_engine_driver     (grep -n 'fn run_static_engine_driver')     <- fixed
```

> **The origin that now responds fast is the one whose prefix numbers are
> structurally zero. The origin that still stalls is the one hosting paged KV
> and the block table.** The fix and the value are on opposite servers.

Note (`grep -n 'set_applicable' crates/onnx-genai-server/src/driver.rs`): KV telemetry is marked applicable when continuous batch
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
throughput falls** (read `perf-baseline.md` for the figure and its interval;
this brief deliberately no longer prints the operands). Batching makes no
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
cd examples/serving-dashboard && node --test    # bare = recurses; an explicit
# glob does NOT. Print `node -v` beside the result, and treat any total below
# 500 as a FAILED RUN, not a small one. `tests 0, exit 0` is a real outcome here.
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

## 8. Thirteen things we learned building this, in the order they will bite you

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

**The repaired criterion, stated here because deleting a bad check is not the
same as having a good one.** It must name the **static/scatter** origin and
require an active batch of **at least two**. Both halves are load-bearing and
neither is obvious from the failure: naming the origin is what stops the
per-request arm from answering, and `>= 2` is what stops `1` — the value that
*proves batching absent* — from reading as success. Measured on both arms in one
minute, same binary, same flags, only the model differing: static reached an
active batch of **4**, dynamic reached **1**, with **4 of 4 completions on both**.
The denominator is not decoration; an earlier run of the same probe read zero on
both origins, which looks exactly like *batching is broken* and was in fact *all
four requests failed*.

The reason this section is not simply deleted along with the check: a retired
criterion that vanishes leaves nothing to stop the next person deriving the same
one from the same reasoning. **A wrong check and no check are indistinguishable
in a green run — but a wrong check that has been replaced in writing is the only
one of the three that cannot come back.**

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

### 8.10 A 200 is not evidence that a file exists; a 404 is self-certifying

We verified a served page four times in ten minutes and got the wrong answer
three of those times. Every error was the same error wearing a different
costume: the instrument was healthy, its output was accurate, and it was aimed
somewhere we did not intend.

```
probed four ports nobody was listening on   -> 000 000 000 000
`lsof | head -5`                            -> hid four of nine listeners
probed the agent harness's own UI           -> 200 200 200 200   <- the dangerous one
probed the right port at the wrong prefix   -> 404
```

The third one nearly closed a release gate. The origin was a single-page app
with a catch-all route, so it answered `200` and returned its own `index.html`
for *every* path — including `styles/shell.css`, which is not a stylesheet it
has ever had. `curl -o /dev/null -w '%{http_code}'` discards the response body,
which is the only evidence that distinguishes the file you asked for from the
fallback you were given.

So the rule is narrower and sharper than "don't trust curl":

> **A `200` needs a hash. A `404` needs only a sibling `200`.**

A catch-all origin *cannot produce a 404*. So if an origin ever returns one, it
has no fallback, and its other answers can be trusted. This is why the
nine-way confirmation that `prefix-cache.js` is not served is sound: it was
always `404` for the missing file and `200` for a sibling in the same directory
in the same second. Compare bytes when the answer is *present*; a sibling
control is sufficient when the answer is *absent*.

Note which way each error cut. The three false negatives cost one re-run each,
because a "not found" makes you keep looking. The single false positive would
have shipped: it agreed with what we wanted, so it terminated the search.
**The error that survives review is always the one that agrees with you.**

And when you compare, compare against `git show HEAD:<path>` rather than the
file on disk. Our working tree was dirty in five files while we were measuring;
"served matches disk" and "served matches what ships" are different claims, and
only the second one is about the reviewer's clone.

### 8.11 A hand-reconstructed status presented as a query is a guess in costume

The task graph that was this project's mandated source of truth never returned
output to its reader, once, in the entire session. Every status in this document
was assembled from message traffic and re-verified against `git` and the
worktree.

That was disclosed rather than papered over, and the disclosure is the reusable
part. The graph was not merely silent — it was wrong in both directions at once.
It replayed at least seven superseded orders as live ones, carrying full
authority and no timestamp; it marked a node complete without its author's
signal; it attached a new upstream dependency to an already-complete node, so it
simultaneously asserted that work was finished and that its input was mid-repair.
Adding an edge does not un-complete a node, and no reader could see any of it,
because the read side had never worked.

Two consequences worth carrying:

- **A replayed order is indistinguishable from a fresh one.** It has your lead's
  name on it, it has no timestamp, and it reads as more urgent than whatever you
  are actually doing. Prefer the message body, which is written in the moment,
  over the title, which is assembled from history. An instruction whose premise
  has changed is void — check the premise, not the phrasing.
- **A disclosed manual ledger beats an undisclosed one, and both lose to a
  verified one.** The prohibition on redundant checklists exists to stop an
  authoritative-looking document going stale in silence. A ledger that is
  re-derived from disk every sweep, and says openly that it is, is the opposite
  of that failure — but only because of the re-derivation, not the disclosure.

Stated generally, and it is the same defect this product exists to refuse, one
level up from the code: **presenting a reconstruction in the visual grammar of a
measurement is the process version of rendering an unplumbed field as a
confident zero.** Be visibly manual rather than invisibly manual.

### 8.12 Add "should this exist at all?" to the claims family

Earlier sections list the ways a claim about an artefact can be true or false
independently: it exists, it is inside a repository, it is committed, it is
wired, it is reachable from a branch, it is in the right checkout. Every one of
those looks identical when it passes.

There is one more, and we found it the expensive way: **it should be there at
all.** An end-to-end verification here certified the build by fetching a
scenario route and asserting `200`. The route was healthy. The check was correct
for its purpose. It was also the route to the one feature we had proved absent
and ruled unshippable — so the URL used to certify the build was the URL of the
thing that should not have been in the build.

A route can be perfectly healthy and still be a route to something that should
not exist. No status code will ever tell you that, because health and
desirability are unrelated properties and only one of them is on the wire.

The same distinction decides a question that looks inconsistent from outside:
a panel bound to zero fields is honest, while a tab advertising a cut scenario
is not. **Panels display values; tabs advertise capabilities.** A cut field in a
panel is a wrong reading. A cut scenario in a switcher is a wrong product — a
clickable promise made before the visitor sees a single number.

Finally, a gate is a measurement at a sha, not a state of the world. This one
was scored green at a commit that was one behind the branch by the time the
result was written down, on a branch moving at roughly one commit every two
minutes. A hand-off therefore needs a *frozen sha*, not a green light. Without
one, three reviewers read three different trees and every finding they file is
unreproducible — which is the same reason a commit must travel as hash *and*
subject phrase: the hash does not survive a cherry-pick, and the question "is
this in the release?" is unanswerable as usually asked.

### 8.13 Nobody here fabricated an observation, and the log will suggest otherwise

Read this section before you read our commit log or our message history, because
without it you will reach a conclusion about this team that is false.

Repeatedly tonight, two people reported opposite facts about the same file, both
having genuinely looked. The clearest case: a CORS module. One person found it
present and router-wired and said so. Another, later, found it absent and said
so. A third confirmed the absence independently. **All three readings were
correct.** The file existed, and then it was deleted — inside a commit whose
message describes a documentation wording change (§8.1). Nobody was careless and
nobody guessed.

> **A disagreement between two careful readers of a hot file is evidence of time
> passing, not evidence of error.**

This is the dominant failure mode of the whole session, and it is worth stating
precisely because its shape is so unhelpful:

**A claim can be true when its author reads the disk and false by the time they
send it.** The interval is seconds. The author did nothing wrong, the reading was
accurate, and the message is now incorrect.

What makes this genuinely hard rather than merely annoying is that **re-checking
does not detect it.** Read the file again and you get the same bytes you just
read — your second observation confirms your first, and both are about the
present, while the claim you are evaluating was about a moment that has passed.
Two reads by one person at one time are one observation, not two. The trap is
identical in shape to §8.9: the confirmation shares an input with the thing it is
confirming, so it echoes rather than corroborates.

The mitigations we arrived at, in order of usefulness:

- **Timestamp and sha every observation, in the same invocation that produced
  it** (§0, §0b). A claim without a sha is not falsifiable, and one with a sha is
  a historical statement anyone can re-run.
- **Prefer `git show HEAD:<path>` to reading the working tree.** Committed bytes
  do not move under you; a shared working tree does, continuously.
- **When two reports conflict, order them before you adjudicate them.** The
  question is almost never "who is wrong" but "which is later, and did something
  land in between." `git log` between the two shas usually answers it in one
  command.
- **Do not ask an author to re-confirm.** Ask what sha they were at. The
  re-confirmation will agree with them and will tell you nothing.

Two consequences for you as a reviewer. First, our record contains many
retractions — an unusual number, and several where someone withdrew a finding
that was correct when filed. Those are not sloppiness; almost every one is this
mechanism, and a team that publishes them is easier to audit than one that
quietly drops them. Second, if you find a claim in these documents that does not
match what you see, the most likely explanation by a wide margin is that the tree
moved. Check the sha attached to the claim before you conclude anything about the
person who made it.

The same courtesy is owed to our tests and our prose. A comment that narrates a
repair is evidence somebody *intended* one (§8.1); a design document written in
the present tense is a claim, not a record, and it keeps asserting itself
correctly about a moment that has passed. Records need dates. Claims need
checkers.
