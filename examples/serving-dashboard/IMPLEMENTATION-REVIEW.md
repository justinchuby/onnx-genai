# Implementation Review — `feat/genai-demo-dashboard`

Reviewer: Code Reviewer @73e77d95 (lane: correctness, readability, patterns, test coverage, code quality)
Originally reviewed at: `24d831a2`
**FINAL RE-VERIFICATION AT: `6c979fa2`** — see "Re-verification #6" immediately below.
Prior: `9e31a7c7` (#5), `d6e57c63` (#3 full-suite baseline).
Scope: 119 files, +35,869 / −143

## ⚠️ Two limits on this review, stated up front

1. **This review cannot discharge AC52.** @c0de4c2e is correct that `demo-spec.md:407` makes
   *"I verified it in a real browser"* the acceptance standard for every visitor-facing panel,
   and that reading code satisfies no criterion in §5. My `review-code` node was released over a
   `qa` node that is BLOCKED and has never run. **Nothing on this branch has been opened in a
   browser.** Everything below is static analysis plus two automated suites. Treat it as
   necessary-but-not-sufficient; it does not substitute for QA.
2. The branch moves ~1 commit/minute. Every finding below is pinned to a SHA and was re-checked
   against `f45d7228` immediately before filing.

## 📐 Citation form (adopted 02:00, per @12e42da8's symbol rule)

**Every finding below names its symbol and quotes its text. Line numbers are a convenience and are
never the identity.** They were correct when written and this branch moves ~1 commit/minute; if a
number has rotted, the symbol and the quoted text still locate the defect.

I adopt the rule, with one correction to its stated premise. *"Not one symbol name has decayed all
night"* is falsifiable and I have the counterexample — **`formatFieldText` was named in a work order
twice tonight and had already been deleted** (verified absent at `852f7789`). The rule survives for
a better reason:

| | decays when | failure when stale |
|---|---|---|
| line number | the file **changes** (constantly) | **SILENT** — points at different code that looks plausible |
| symbol name | the symbol is **deleted/renamed** (rare) | **LOUD** — points at nothing; unexecutable |

`lib.rs:26` decayed into `mod demo_assets;` and executing that order would have killed `/demo`.
`formatFieldText` decayed into nothing and simply stopped the work. **Symbols are not durable, they
are fail-closed** — the same doctrine behind `renderStateOf`'s throw and `modesFor`'s `default:`
branch. Operational form: **name the symbol, quote the text, the line may accompany and never
substitute.**

## 🔻 Re-verification #6 — TWO OF MY OWN FINDINGS DECAYED (`6c979fa2`)

Clean detached worktree, `--porcelain = 0`, 01:56:58. **`dashboard/prefix-cache.js` was deleted at
`45c1103d`**, and it was load-bearing for two of my findings. Recording the decay before anyone
else finds it.

### F4 downgraded MAJOR → MINOR: the live trigger is gone

`prefix-cache.js:77` shipping `staleCeilingMs: null` was the one caller that made the `??` bug
observable. Every remaining caller passes an explicit positive integer:

```
system.js:53 30000 · throughput.js:51 3000 · requests.js:61 10000
kv-memory.js:56 15000 · scheduling.js:60 3000
callers passing null:  ZERO
```

With no `null` caller, `??` and `=== undefined` behave identically. **The bug is latent.** Still
worth fixing — the JSDoc at `panel-kit.js:449` types it `{staleCeilingMs?: number}`, which does not
admit `null`, so the contract and the code still disagree about a value the code silently absorbs —
**but it is not a merge blocker and I was wrong to keep listing it as one.**

### F11 count corrected: FIVE panels, not six

```js
export const PANELS = Object.freeze([throughput, scheduling, kvMemory, requests, system]...)
```

The defect is unchanged and still live (`index.js:188` passes the non-existent `panel.title`); only
my count was stale. @c7a654ed retracted `PANELS.length === 6` as a proxy that outlived the thing it
proxied — I was still quoting the proxied number.

**F11's landing-order caveat is WITHDRAWN.** `panel-kit.js:1049` gates the throw on
`IS_DEVELOPMENT` and otherwise `console.error`s and keeps building, with the reason in the comment
(*"dropping a panel in front of a room is worse than an unnamed one"*). Loud where a developer sees
it, survivable where a visitor does. **The label-derivation fix can now land alone, in any order.**
This is better than the sequencing constraint @086345a5 and I agreed on: a gate is a property,
ordering is only a promise.

### `case 'paged-kv'` is dead — keep it, but it is untested

Confirmed zero producers (`requires: null` ×4, `'continuous-batch'` ×1). **Deleting the case would
convert "returns the correct answer if used" into "throws at mount"**, because `modesFor`'s
`default:` branch correctly refuses to guess. Keep it; it completes the capability vocabulary.

But no test exercises it, and the `paged-kv` hits in the suite are a **different vocabulary**:

```
scenario-origins.test.js       planScenario('paged-kv', ...)      <- a SCENARIO ID
check-launch-command.test.js:444  'paged-kv-block-table'          <- URL KEYS
registry.test.js:142           modesFor({requires:'paged-kv-v2'}) <- tests the THROW, not the case
```

Anyone grepping `paged-kv` in tests concludes the branch is covered. It is not. **Third instance on
this branch of one string carrying several vocabularies** (after `'ok'` = field state vs HTTP health
payload, and `RENDER_STATES`/`FIELD_STATES`), and each time the grep that should have caught it
confirmed the wrong answer instead.

### F13 — MINOR (new, DRY): the `requires` vocabulary is defined twice

```js
dashboard/honesty.test.js:257   const VALID = new Set(['continuous-batch', 'paged-kv', null]);
dashboard/index.js:85-98        switch (module.meta.requires) { case ... }
```

A hardcoded allowlist duplicating the switch it polices. Add a case and the test still rejects it;
delete one and the test still blesses it. Export the vocabulary from `index.js` and derive the Set.
**The file whose docblock spends twenty lines teaching derive-don't-duplicate has a hand-maintained
copy of its own vocabulary sitting in its test.**

## 🔑 Re-verification #5 — F3 PROVEN BEHAVIOURALLY; F12 FILED; TWO ORDERS INTERCEPTED

Measured in clean detached worktrees (`git worktree add --detach`, `--porcelain = 0` each time),
at `59eff0ce`, `d5c16fde`, `bbba34ac`, `9e31a7c7`, between 01:44 and 01:53.

### The single red, NAMED (nobody had named it; three different counts were circulating)

```
59eff0ce   482 tests · 481 pass · 1 fail
  check-source-citations.test.js › "a cited line still sits beside the symbol the prose names"
  5 README.md citations name a symbol no longer at the cited line.
```

It is **not** the enum. Four enum tests pass **by name** in the same run:
`ok 21 - the README states the CURRENT wire value of the measured state`,
`ok 74 - honesty lint — the measured state is never compared as a literal`,
`ok 115`, `ok 140`. The circulating instruction to "revert the assertion to `'ok'` and the suite
goes green" would not touch the actual red **and** would break `ok 21` — one failure becomes two.
(@086345a5 independently applied the edit in an isolated worktree and measured exactly that:
`481/1 → 480/2`.)

Note also: the consumer audit ordered by the Lead **already exists as `ok 74` and is green.**

### F12 — MAJOR (new): `repair-citations.mjs` computes from the dirty tree, and has no tests

Commit `bb649cbc` — *"compute citation line numbers from source instead of maintaining them by
hand"* — took the suite from **1 red to 2**, breaking a previously-green test:

```
not ok 60 - every line the README cites still exists in the file
  "README.md cites driver.rs:956, but that file has only 912 lines."
```

Root cause, proven arithmetically:

```
repair-citations.mjs:111   readFileSync(join(REPO, file))          <- the WORKING tree

git status --porcelain crates/onnx-genai-server/src/driver.rs  ->  " M "  DIRTY
  committed (git show HEAD:)   911 lines   run_fallback_generation at :861
  working tree (readFileSync) 1006 lines   run_fallback_generation at :956   <- README says 956
```

**The number is not wrong; it is correct for a tree that was never committed and exists only in
one agent's uncommitted edit.** The checker reads committed state, the repairer reads the dirty
tree — producer and consumer read different trees, so they cannot agree while anyone is mid-edit.

Fix: read via `git show HEAD:<file>`, **and** refuse to run when the target path is dirty.
Refusing is the more important half — a repair tool that runs against a dirty tree cannot produce
a correct answer, and this tool already declines loudly on ambiguous paths while never applying
that standard to itself.

**Test coverage (my lane):** `git ls-files | grep repair-citation` returns the `.mjs` only. It is
referenced by `PR-DESCRIPTION.md` and itself — not wired into any test, script, or CI. **194 lines
of new logic whose output is committed documentation, with zero tests, whose first and only
production run regressed a green test and shipped a past-EOF citation into the README.** The commit
trailer reads *"My suite: 59/59, deterministic over five consecutive runs"* — true, and not
evidence, because none of the 59 point at this tool.

To be fair to the author, the tool is **well designed**: it declines loudly rather than skipping
silently, never invents a file, only rewrites on exactly one definition-shaped match, and carries a
regex specifically guarding a struct-field false positive 342 lines from the real definition.
Its header states our doctrine better than we have — *"A line number in prose is a copy of …
COMPUTED FROM THE ARTIFACT OR DELETED."* **The doctrine is right; the artifact selection is wrong.**
This is the crew's signature error — measuring one tree and writing into another — automated, and
handed the authority of the word "computed."

### F3 proven from both ends by execution (was: argued about for ~90 minutes)

Seven shipping producers emit the raw string `'ok'`; **zero** emit `'measured'`:

```
dashboard/store-adapter.js:226 232 338 370 381 518
dashboard/testing/fake-store.js:26                  <- a test fixture teaching the retired spelling
CONTROL — raw state:'measured' in shipping JS:  0
```

The two consumers disagree about that exact value:

```
DASHBOARD  renderStateOf({state:'ok'})  -> 'measured'   ACCEPTS, renders the number
ROOT       formatField({state:'ok'})    -> em-dash, hasValue:false, console warning — REFUSES
```

**Same value, opposite outcomes, both modules individually correct and individually tested.** Route
any store-adapter field through `format.js` and every one blanks — silently, because refusal renders
as an em-dash, a legitimate absence glyph. This is why no experiment settled the argument: the
system was built to answer "fine" to both hypotheses.

The seven literals are a **latent hazard, not a live defect** — normalisation at the render boundary
absorbs them. The claim that `state-vocabulary.test.js` "would reject every one" is falsified by the
suite itself: the literals are in the tree *and* the suite is green, therefore no test rejects them.
Proof by contradiction; no extra run needed. Minimal fix remains @086345a5's: rename the **key**
`OK → MEASURED` in `field-state.js:53` (2 call sites, wire-neutral).

### Two destructive orders intercepted

**(a) `shell.css:163`.** Ordered changed on the premise that `[data-state='measured']` "has matched
nothing all session." The DOM writes the enum **value**:

```
ui/model-card.js:91         element.dataset.state = field.state;
dashboard/panel-kit.js:277  attrs: { 'data-state': state, ... }   (state = renderStateOf(field))
renderStateOf({state:'ok'}) -> 'measured'    <- BOTH stacks converge on 'measured'
```

The selector is **live on every path that exists**. Changing it to `'ok'` would *manufacture* the
dead selector, and would redden two currently-green guards —
`state-channel.test.js:249` (`assert(!shellCss.includes("[data-state='ok']"))`) and
`state-treatments.test.js` (`shell.css does not style a state that cannot occur`).
I deliberately tried to break my own finding here: had the dashboard stack written its raw `'ok'`
straight through, the selector *would* have been dead for six panels and the ruling right by a route
nobody had identified. It normalises first. **There is no stack for which the edit is correct.**

**(b) The bidirectional selector-vs-enum guard ordered built already exists and passes:**

```
node --test state-treatments.test.js state-channel.test.js  ->  13 tests, 13 pass, 0 fail
  ✔ every field state has a visual treatment in shell.css   (enum -> CSS)
  ✔ shell.css does not style a state that cannot occur      (CSS -> enum)
```

Also note `shell.css:158-162` — the comment directly above the line — already describes the ordered
edit's failure mode ("silently fell back to inherited styling … only a cross-language check finds
it"), and `state-channel.test.js:224` records the migration direction `[data-state='ok'] ->
[data-state='measured']`. The migration already ran; the order asks us to run it backwards.

### A rule this session earned

A destructive instruction that names a **symbol** is still not idempotent when it is a
**substitution**: `measured -> ok` run twice, or run after the migration already ran, silently
reverses a completed change. The idempotent form of a substitution is a **predicate** — *"shell.css
must select exactly the values the enum emits"* — which is a test, and that test exists and is green.
**Ship the predicate, not the edit.**

## 🔑 Re-verification #4 — F3 IS THE ROOT CAUSE OF THE ENUM SAGA

Verified in a **clean detached worktree** at `bd31e32f`, `git status --porcelain` = **0 lines**
(instrument switched to `git worktree add --detach` on @086345a5's correction — `git archive`
has no `.git` and breaks every self-inspecting test in this suite). Control:
`grep -c export dashboard/index.js` → 6, so the root resolves.

**Executed, both stacks, side by side:**

```
RENDER_STATES  (dashboard/field-state.js)  = {"OK":"measured", "PENDING":"pending", …}
FIELD_STATES   (telemetry-field.js)        = {"MEASURED":"measured", "PENDING":"pending", …}
                                                ^^^^^^^^          ^^^^^^^^^^
                                              KEYS DIFFER        VALUES IDENTICAL
```

**Both stacks emit `'measured'`. They disagree only on the KEY NAME.** One calls the symbol
`OK`, the other calls it `MEASURED`, and both hold the string `'measured'`.

### This explains nine rulings, four agents, and two reversals — with nobody stale

| who read | what they saw | what they reported | correct? |
| --- | --- | --- | --- |
| dashboard stack | `RENDER_STATES.OK` | "there is an `OK` key" → *the wire value is `'ok'`* | **key: yes. value: no.** |
| root stack | `FIELD_STATES.MEASURED`, no `OK` key | *the wire value is `'measured'`* | **yes** |

@12e42da8's tie-break sentence — *"`FIELD_STATES.MEASURED` IS THE SYMBOL; `'ok'` IS THE WIRE
VALUE"* — is an exact description of `RENDER_STATES.OK` **with its two halves swapped**. There
the symbol is `OK` and the value is `'measured'`. It is the most understandable error of the
night and it was **unavoidable given F3**: the branch ships two vocabularies for one concept,
so "read the enum" has two correct answers.

⚠️ **The diagnosis everyone reached for was the clock, and the clock was never the explanation.**
@12e42da8's rule — *"a disagreement between two honest readers of a hot file is evidence of time
passing"* — is right, and it was applied to a case where the readers were in **different files**.
Adding to it, from this instance: **when two honest readers disagree, the first hypothesis is
the clock, and the second is that they are not reading the same file.** No amount of re-reading
at a fresh sha would ever have converged these two, because both were current.

### The authority was quoted backwards

```
dashboard/state-vocabulary.test.js:28
const RULED_STATES = Object.freeze(['measured', 'pending', 'stale', 'unavailable', 'not-applicable']);
```

The ruling cited this file as freezing `['ok', …]`. **It freezes `'measured'`.** The test passes
**12/12** at `bd31e32f`. So the ruling's *method* was exactly right — @12e42da8's own new binding
rule, *"if a test freezes the current value, the current value is a decision, not a defect"* —
applied to a value that was misread. **The rule works; it was fed the wrong string.**

### A false statement inside that test's own failure message

```
state-vocabulary.test.js:39   "field-state.js no longer bridges spellings, so …"

executed:  normaliseState('ok')       -> 'measured'
           normaliseState('measured') -> 'measured'
```

**The bridge is still live.** The claim that it was deleted survives only in the error message of
the test that would print it — a message a reader sees *precisely when they are already
confused about spellings*. This is my F3 `STATE_ALIASES` finding, confirmed by execution, and it
is the third artifact in this saga that says the opposite of what the code does.

### The comment the ruling ordered deleted does not exist

`grep -rn "see :90|emits ok|named MEASURED"` over the clean tree → **no match anywhere.**
The deletion assigned to @c8d9a40e is a no-op. **The replacement text is the hazard:** a new
comment reading *"`'ok'` is deliberate"* would be **false**, would sit in the canonical file, and
would manufacture report #5 — the exact failure mode the ruling was written to prevent.

### ✅ The safe conclusion: NO ENUM EDIT IS NEEDED, BY ANYONE

Every value comparison on the branch is correct today; both stacks already agree on `'measured'`;
the suite is green because the agreement is real. **The defect is entirely in the SYMBOL names
and in the rulings written about them, not in behaviour.** Per @12e42da8's own standing rule —
*seven mechanisms guard against fabricated additions and zero against removing something
correct* — the highest-risk action available right now is an edit, not the absence of one.

**This is what F3 costs.** I filed it as "two parallel render stacks disagree," which reads like
a tidiness complaint. It is not. **It consumed nine rulings, four agents, two reversals, and a
stand-down order, and the crew's best diagnostic instrument — re-read at a fresh sha — cannot
resolve it, because both readings are current.** Collapsing the stacks is the fix; until then,
any statement about "the enum" must name the file.


**Headline: the JS suite is GREEN and five of my findings are still live. Those two facts are
not in tension, and the gap between them is the most useful thing in this report.**

### The suite number, and why I am quoting two of them

| tree | tests | pass | fail |
| --- | --- | --- | --- |
| **working tree** at `d6e57c63`, `git status --porcelain` = **9 files** | 430 | 412 | **18** |
| **committed** `d6e57c63`, `git archive HEAD` full extract, no pathspec | 485 | **484** | 1 |

The single failure in the committed run is `check-source-citations.test.js`, which shells out to
`git rev-parse --show-toplevel` and therefore **cannot pass in an extract that has no `.git`**.
Confirmed by running it alone: `fatal: not a git repository`. **Discount it: the committed branch
is 484/484 green.**

⚠️ **The two rows differ by 55 tests and 17 failures at ONE sha.** I am publishing both because
either alone is misleading. The dirty run is other agents' in-flight work (the capability-probe
and provenance-column edits) and filing those 17 as branch defects would have been 17 false
reports. The committed run is the branch. **@c7a654ed's practice of quoting the tree state
beside the sha is the only reason I noticed — a sha does not identify what you tested, and this
is the sharpest example of it anyone has produced tonight: same commit, 18 failures or 1.**

### The five findings — RE-VERIFIED AT `b54437df`, not left where they were typed

> **Every status in this table was re-derived by execution at `b54437df`.** The
> `d6e57c63` column below is the *original* observation and is retained as history;
> it is no longer a claim about the tree. A status without a SHA is not a status.
>
> **Search method:** `git grep -nF` (fixed-string). `-E` with `\b` returns a false
> zero in `git grep` — it silently matches nothing and exits 0.

| Finding | Evidence as first observed at `d6e57c63` | Status re-derived at `b54437df` |
| --- | --- | --- |
| **F1** BLOCKER | `driver.rs:464` `kv_telemetry.set_applicable(!continuous_batch_supported);` — and `:450` is the correct sibling `set_applicable(paged)`. `runtime.rs:263` `pub fn attach_kv_telemetry(&mut self, …)` still returns nothing. | **STRUCK @ `459c40c2`** — replaced by `classify_kv_applicability()`. The one surviving textual hit is `tests.rs`, in the doc comment on `fn paged_kv_applicability_is_a_conjunction_of_two_independent_facts` — a comment *describing* the dead bug. **A grep cannot see tense: the hit and the proof-of-fix are byte-identical.** |
| **F3** BLOCKER | `format.js` and `dashboard/field-state.js` both still ship. | **LIVE — VERIFIED-AT `b54437df`** (both paths present in `git ls-tree`) |
| **F4** MAJOR | `panel-kit.js:273` and `:454` both `?? DEFAULT_STALE_CEILING_MS`; `prefix-cache.js:77` `staleCeilingMs: null`. | **LIVE (as F4-REVISED) — VERIFIED-AT `b54437df`.** ⚠️ Citation drifted: the declaration is now `prefix-cache.js:88`, not `:77`. |
| **F5** MAJOR | `audit_citation_targets.py:25` `ROOT="/Users/justinc/Documents/GitHub/onnx-genai-demo"`. | **STRUCK @ `1b4d76c6`** — `ROOT` now derives from `tree_context.repo_root()`; exits `CANNOT_RUN` with no worktree and exits 1 conditionally. Zero hits for the hardcoded path; positive control `repo_root` fires in 7 files. |
| **F11** MAJOR | `index.js:179` `createRovingGroup(root, { label: panel.title })`; `panel.title` is not a key of the frozen registry entry. | **LIVE — VERIFIED-AT `b54437df`.** ⚠️ Citation drifted: now `dashboard/index.js:143`, not `:179`. See also the F11 count retraction below (five panels, not six) — that retraction is about the COUNT, not the defect. |
| F2 | retracted — see #2 above. | **FIXED, and so is its guard — I withdraw the claim that stood here.** I wrote that `check-field-states.test.js:69`'s `FIELD_STATES.MEASURED ?? FIELD_STATES.OK` made the guard unfailable. **Mutation-proven false at `1bca52a8`, against the real module and the real README, no file written:** point the README sentence at the retired spelling `'ok'` → **fires**; delete the sentence entirely → **fires**; unmutated control → **silent**. It is a working, non-vacuous drift guard that reads the wire value from the constant instead of hardcoding a spelling. **The only true residue is cosmetic and I should have filed it as such: `FIELD_STATES.OK` is `undefined`, so the `??` arm is dead code and `:73`'s failure message names a key the enum no longer defines.** Not a defect in the guard — a retired spelling surviving in the guard's own prose. |

**Reader instruction, applying to this whole document:** every `file:NNN` here is a
**hint, not an address**. Two of the five rows above drifted line numbers while their
substance was unchanged, and this branch took ~1.4 commits/minute during the review.
**Locate findings by symbol (`git grep -nF '<symbol>'`), never by line number.**

**A green suite did not catch any of the five.** F1 has a red Rust test, so it is caught by the
*other* suite; F3, F4, F5 and F11 have no test that could fail. That is the finding behind the
findings: **484 green JS tests coexist with an unnamed `role="group"` on every panel, a silently
ignored null ceiling, a linter that cannot fail, and two render stacks that disagree.**
> **TENSE STAMP (re-measured `1bca52a8`): this paragraph is a snapshot and two of its clauses have
> since expired.** `484` was the JS count at the time of writing; the canonical runner reported
> **632 / 95 suites / 0 fail** hours later. **"A linter that cannot fail" is F5, and F5 was STRUCK
> at `1b4d76c6`** (ancestor of HEAD, checked). The clauses that still hold at re-measurement are the
> unnamed `role="group"`, the null ceiling (F4-revised), and the two render stacks (F3). **Do not
> quote this sentence as a current count or a current finding — it is neither.**

### The enum: the ordered audit comes back CLEAN, and the premise it rested on is stale

@12e42da8 ruled: *"the one genuine, severe bug is `FIELD_STATES.MEASURED` holding the value
`'ok'` … `'ok'` IS THE CORRECT WIRE VALUE … audit that every consumer compares against the
CONSTANT and never against a string literal. Three consumers use it."*

**Executed at `d6e57c63`, not grepped:**

```
FIELD_STATES.MEASURED === 'measured'   ->  true
FIELD_STATES.OK                        ->  undefined
```

**So the premise is inverted** — the constant and its value already agree, and `FIELD_STATES.OK`
does not exist. **But I ran the ordered audit anyway, because the ACTION is correct whichever
spelling wins**, and it is the half of the ruling that survives:

```
literal comparisons against a field state in shipping (non-test) JS:  ZERO
control: literal 'measured' in non-test JS -> 14 hits, so the search resolves
```

✅ **No consumer is broken. Nobody needs to be assigned this.** The only two hits are doc
comments. Reporting a clean audit as loudly as a dirty one, because "we checked and there is
nothing to do" is a result, and an unanswered audit request gets re-issued.

### The stale comment IS real — @12e42da8 predicted it exactly, in the opposite direction

The ruling's closing point — *"there is a comment documenting the bug that survives the fix and
re-opens it for the next reader; a stale comment about a fixed bug is a bug generator"* — is
**correct, and it lands on a file nobody has fixed.** Both flagged independently by @376a0297:

```
state-treatments.test.js:7   "while `FIELD_STATES.MEASURED` is the string `'ok'`"
                             ^^ PRESENT TENSE, FALSE at d6e57c63
```

Compare its sibling, which is **exemplary and should be the model for the fix** —
`telemetry-field.js:133-152` says the same thing in the **past tense** (*"the constant and its
value disagreed once — `MEASURED: 'ok'`"*), then explains the silent-failure mode, then warns
against the global replace, then names the test that enforces the atomic pair. **One is a
historical record; the other is a false statement about current behaviour.** The fix is a tense
change on one line, and it belongs in the one file a maintainer opens to learn what the states
mean.


Method per the crew's standing rules: SHA from `git rev-parse` in the same invocation,
`git rev-parse --show-toplevel` confirmed to end in `onnx-genai-demo`,
`git status --porcelain` on every cited path **empty** (so committed == working), evidence read
via `git show HEAD:<path>` (the artifact that ships) — and for the enum, **executed rather than
grepped**.

| Finding | Status at `a721f033` |
| --- | --- |
| F1 `driver.rs:464` | **STRUCK @ `459c40c2`** — re-verified at `b54437df`. Now `classify_kv_applicability()`; the negation is gone from shipped Rust |
| **F2 `'ok'`→`'measured'`** | ✅ **RESOLVED — I RETRACT IT.** `format.test.js` swept; 18/18 green |
| F3 dual render stacks | STILL LIVE |
| F4 `null` ceiling | **STILL LIVE.** `panel-kit.js:273,454` both still `?? DEFAULT_STALE_CEILING_MS` |
| F5 `audit_citation_targets.py:25` | **STRUCK @ `1b4d76c6`** — re-verified at `b54437df`. `ROOT` derives from `tree_context.repo_root()`; exits `CANNOT_RUN` with no worktree, exits 1 conditionally |
| F11 unnamed `role="group"` | **STILL LIVE.** `index.js:179` still `{ label: panel.title }` |

### 🔻 F2 RETRACTED — and the retraction matters more than the finding did

F2 is fixed. `format.test.js` is 18/18 green. I am striking it, loudly, because a stale red that
nobody re-runs is exactly the failure mode @12e42da8 named ("a red you did not re-run is a
rumour") and I do not get an exemption for being the reviewer.

### ⚠️ But the enum ruling in circulation is inverted, and it is dangerous in the write direction

@12e42da8 broadcast, "FOR THE NINTH AND ABSOLUTELY FINAL TIME: THE WIRE VALUE IS `'ok'`.
FIELD_STATES.OK, MEASURED alias DELETED in f19fdb63 (D160)." That was true at `f19fdb63` and was
**reverted at `24d831a2`** ("land the ratified 'measured' rename as one atomic pair"). Executed
at `a721f033`, not grepped:

```
FIELD_STATES = {"MEASURED":"measured","PENDING":"pending","STALE":"stale",
                "UNAVAILABLE":"unavailable","NOT_APPLICABLE":"not-applicable"}
FIELD_STATES.MEASURED === 'ok'        -> false
FIELD_STATES.MEASURED === 'measured'  -> true
FIELD_STATES.OK                       -> undefined      <-- THE RULING NAMES A CONSTANT THAT DOES NOT EXIST
```

Both halves of the atomic pair are consistent on disk: `telemetry-field.js:153`
`MEASURED: 'measured'` and `shell.css:163` `[data-state='measured']` (control: 11 `data-state`
hits in that file, so the grep resolves).

**Why this is a live hazard and not a debating point.** `FIELD_STATES.OK` is `undefined`.
Anyone who writes to the circulating ruling produces `field.state === FIELD_STATES.OK`, which
compares against `undefined` and is **never true** — no error, no warning, no test failure,
and every genuine measurement silently falls through to the unknown-state path. That is
precisely the defect @732c7548 was praised for documenting ("a comparison that is NEVER TRUE,
with no error and no warning") — the ruling now recreates it in the opposite direction.

@376a0297, @c7a654ed and @0837fdf9 have independently reported `'measured'`; the disk agrees
with them. This needs one correction from @12e42da8 so nobody writes `FIELD_STATES.OK`.
I hold no view on which spelling *should* win — only on what the shipping artifact does.

## Re-verification #1 at `f45d7228` — HISTORICAL SNAPSHOT, SUPERSEDED BY #2 ABOVE

> **Every row below is a dated reading, not a claim about the tree.** It is retained as the
> record of what was true at `f45d7228` and is deliberately not updated. **If you arrived here by
> `grep`, you are reading history: the words "still live" below describe `f45d7228` and nothing
> else.** Each cell restates its own tense so it cannot be quoted into a present-tense claim.
> For current status, read Re-verification #2 above — F1 and F5 are both **STRUCK**.

| Finding | Status **as at `f45d7228` only** |
| --- | --- |
| F1 `driver.rs` applicability inference | **Was live at `f45d7228`.** `set_applicable(!continuous_batch_supported)`; `runtime.rs` still returned `()`. **Since STRUCK at `459c40c2` — the negation is gone from shipped Rust.** |
| F2 `'ok'`→`'measured'` half-sweep | **Was red at `f45d7228`.** `format.test.js` fed `'measured'` as the *unknown* state and still enumerated `'ok'` |
| F3 dual render stacks | **Was unchanged at `f45d7228`.** `format.js`, `telemetry-field.js`, `dashboard/field-state.js` all shipped; `ui/model-card.js` imported the root stack; `field-state.js` aliased `ok` |
| F4 `staleCeilingMs: null` | **Changed shape at `f45d7228`** — see F4-revised. Test was rewritten in `45c1103d`; the implementation was byte-identical |
| F5 `audit_citation_targets.py` hardcoded ROOT | **Was live at `f45d7228`. Since STRUCK at `1b4d76c6`.** |

Suite at `f45d7228`: **462 pass / 4 fail** (was 460/6). Of the 4, one
(`check-source-citations.test.js`) is an artifact of my `git archive` extract having no `.git`;
**3 are real**, two of them F2.

Correctly discounted per @c0de4c2e: `sidecar_free_compatibility_package` is a permanent red
(needs a `.gitignore`-excluded `*.onnx` fixture) — it was never in my findings. `cors.rs` does
not exist and is not referenced anywhere in this review.

## Method

The branch worktree `/Users/justinc/Documents/GitHub/onnx-genai-demo` is dirty — four agents
are mid-edit and the Rust crate **does not currently compile** there
(`admin.rs:146`, missing field `batch_in_flight` in `NodeStatus`). To review the *shipping*
state rather than in-flight churn I extracted `git archive HEAD` to a clean tree and ran
everything there.

## VERDICT: REQUEST CHANGES — the branch is red at HEAD

| Suite | Result |
| --- | --- |
| `node --test '**/*.test.js'` (dashboard) | **460 pass, 6 fail** (+1 false failure, see note) |
| `cargo test -p onnx-genai-server -p onnx-genai-kv` | **162 pass, 2 fail** |

Note: `check-source-citations.test.js` fails only in my extracted tree because it shells out to
`git rev-parse --show-toplevel`. Not a branch defect — but see F7, it fails in the worktree too.

---

## F1 — ~~BLOCKER~~ **STRUCK @ `459c40c2`** (re-verified at `b54437df`): applicability is inferred from an unrelated capability

> **THIS DEFECT IS FIXED. Do not act on this section.** `459c40c2` replaced the negation with
> `classify_kv_applicability(paged, continuous_batch_supported)`, a total function over two bools.
> The analysis below is retained because it explains *why* the fix is shaped the way it is.
> **`git grep` for the defect string still hits `tests.rs`, on the doc comment for `fn paged_kv_applicability_is_a_conjunction_of_two_independent_facts` — that is a comment quoting
> the dead bug, not the bug.**

`crates/onnx-genai-server/src/driver.rs:464`

```rust
let continuous_batch_supported = engine.continuous_batch_manager(max_batch).is_ok();
engine.attach_kv_telemetry(Arc::clone(&kv_telemetry));
kv_telemetry.set_applicable(!continuous_batch_supported);
```

"This engine does not support continuous batching" does **not** imply "this engine's decoder
pages through the KV page table." It only implies the fallback per-request path was taken.
The branch's own test proves the inference is wrong:

```
tests::kv_blocks_renders_no_numbers_until_paged_kv_is_known_to_be_in_use
  crates/onnx-genai-server/src/tests.rs:4058
  assertion `left == right` failed
    left: Bool(true)     <- body["applicable"]
   right: false
```

So `/v1/debug/kv/blocks` serves live `page_size` / `pages_in_use` / `hot_capacity` for the
`tiny-llm` fixture — numbers describing a pool the decoder never consults. That is precisely
the failure mode this entire feature was built to prevent, and the doc comment three lines
above it (`admin.rs:587-591`) describes the danger correctly while the code walks into it.

**This is a design bug, not just a wrong boolean.** `set_applicable` is being *handed an
inference* when it could be *handed a fact*. The sibling code path already does it right:

- `pipeline/mod.rs:868` — `attach_kv_telemetry(...) -> bool`, returns whether a paged cache
  actually exists; `driver.rs:450` uses that return value. Correct.
- `engine/runtime.rs:263` — `attach_kv_telemetry(...)` returns `()`. The caller has nothing to
  go on, so it guesses.

**Fix:** make `Engine::attach_kv_telemetry` return `bool` (whether the decode path it will
actually run routes through `kv_cache.page_table`), mirroring the pipeline API, then:

```rust
let paged = engine.attach_kv_telemetry(Arc::clone(&kv_telemetry));
kv_telemetry.set_applicable(paged && !continuous_batch_supported);
```

Two functions with the same name on two engine types should not have different signatures and
different honesty guarantees. Unifying them also removes the guess.

---

## F2 — BLOCKER: the `'ok'` → `'measured'` rename shipped half-swept

Commit `24d831a2` is titled *"land the ratified 'measured' rename **as one atomic pair**"*. It
is not atomic. `telemetry-field.js:122` now reads `MEASURED: 'measured'`, and
`format.js` refuses any state not in `FIELD_STATES` — but `format.test.js` was never swept:

- `format.test.js:198` — *"an unknown state is refused"* feeds `state: 'measured'` and asserts
  it is **rejected**. It is now the ratified value. Test fails: `'999 requests' !== '—'`.
- `format.test.js:269` — *"all five states render distinct text"* enumerates
  `['ok', 'pending', 'stale', 'unavailable', 'not-applicable']`. `'ok'` is no longer a state,
  so it falls through the unknown-state branch to `—` and collides with `unavailable`.
  Test fails: `4 !== 5`.

The CSS half **was** swept correctly (`styles/shell.css:163` is `[data-state='measured']`), and
`state-channel.test.js:223-250` documents the atomic pair properly. Credit where due — the
sweep was 90% disciplined. But a commit that names itself atomic and leaves its own test file
asserting the opposite of the ruling is the exact drift the tripwires exist to catch.

Also stale: `design/skeleton.html:391,394` still emit `data-state="ok"`, which no longer
matches any selector in `shell.css`.

---

## F3 — BLOCKER: two parallel render stacks disagree about the same field

Both ship on the same page (`app.js:28` → `ui/model-card.js` → root `format.js`; `app.js:18` →
`dashboard/index.js` → `dashboard/panel-kit.js` → `dashboard/field-state.js`).

`dashboard/field-state.js:8-22` says so itself and asks to be deleted. It is not merely
duplication — the two stacks give **different answers to the same input**:

| Input | root `format.js` | `dashboard/` stack |
| --- | --- | --- |
| `state: 'ok'` (old wire spelling) | unknown → `—`, page blanks | accepted via `STATE_ALIASES` (`field-state.js:101`) |
| stale field with no `observedAtMs` | renders `41 reqs · age unknown` (value **shown**) | `isPastStaleCeiling` → `ageMs === null` is **past the ceiling** → value **withheld** |

Both behaviours are defended in careful comments in their own file. They cannot both be right.
Two vocabularies for "what is a measurement" on one honesty-focused dashboard is the highest
drift risk on the branch. Pick one; `dashboard/field-state.js` is the more defensive of the two.

---

## F4-REVISED — MAJOR (was blocker): the null ceiling is now live, and silently becomes 10s

**Status changed at `f45d7228`.** Commit `45c1103d` rewrote `staleness.test.js` rather than the
code. The old rule (*`cadence === 0` ⇒ ceiling MUST be `null`*) was replaced by a rule that
allows any number and permits `null` only if the panel's source binds no telemetry. That
resolves the test-vs-`requests.js` contradiction, and the new test is better in one real way:
it asserts `'staleCeilingMs' in meta` (no silent inheritance) and verifies `null` against module
**source** rather than trusting `meta`. Good instinct.

**But the implementation defect it was masking is untouched, and is now completely unguarded:**

- `dashboard/prefix-cache.js:77` declares `staleCeilingMs: null` — this is **live in a shipping
  panel**, not hypothetical.
- `dashboard/panel-kit.js:452` — `panelMeta?.staleCeilingMs ?? DEFAULT_STALE_CEILING_MS`
- `dashboard/panel-kit.js:271` — `options.staleCeilingMs ?? DEFAULT_STALE_CEILING_MS`
- `DEFAULT_STALE_CEILING_MS = 10_000` (`field-state.js:122`)

So a panel declaring *"I have no freshness contract"* is silently given a **10-second** one.
`??` does not fire on `null`… it fires on `null` **and** `undefined` — which is exactly the
problem: the contract needs to distinguish "unset" (use default) from "explicitly no ceiling",
and `??` collapses them.

Impact today is nil **only** because `prefix-cache.js` binds no fields after the counter cut.
That is a landmine, not a fix: the first person to add a field to that panel gets a 10s expiry
they never asked for, with no test covering it. It is a one-line fix while it is still free.

Also: the JSDoc at `panel-kit.js:448` types the parameter `{staleCeilingMs?: number}` — which
does not admit `null` at all, while a shipping caller passes exactly that. The declared type
and the shipping call site disagree.

**Fix:**
```js
const staleCeilingMs =
  panelMeta?.staleCeilingMs === null ? Infinity : (panelMeta?.staleCeilingMs ?? DEFAULT_STALE_CEILING_MS);
```
…in both places, widen the JSDoc to `{staleCeilingMs?: number|null}`, and add the test that
`45c1103d` did not add: **a `null`-ceiling panel still renders a 60-second-old field**. Right
now nothing anywhere exercises the `null` path through `bindPanel`.

---

## F5 — ~~MAJOR~~ **STRUCK @ `1b4d76c6`** (re-verified at `b54437df`): `audit_citation_targets.py` is hardcoded to one machine and cannot fail

> **THIS DEFECT IS FIXED. Do not act on this section.** `1b4d76c6` ("make
> `audit_citation_targets.py` capable of failing") derives `ROOT` from
> `tree_context.repo_root()`, exits `CANNOT_RUN` when there is no worktree, and exits 1
> conditionally. Verified by @732c7548 and independently by @e00032a4 by execution.

`scripts/audit_citation_targets.py:25`

```python
ROOT="/Users/justinc/Documents/GitHub/onnx-genai-demo"
```

Absolute path, one developer's home dir, one worktree name. On `main`, on CI, or in any other
worktree it walks the wrong tree or none. There is also no `main()`/`sys.exit()` — the script
prints a report and **always exits 0**, so it can never gate anything and cannot be told apart
from a vacuous pass. For a tool whose stated purpose is hunting false authority in citations,
that is an uncomfortable irony.

**Fix:** derive `ROOT` from `__file__` (or `git rev-parse --show-toplevel`, as the sibling
checkers do) and return non-zero when any citation resolves to COMMENT/BLANK/DELIM/unresolved.

---

## F6 — MAJOR: in-flight poll can publish after teardown

`examples/serving-dashboard/telemetry-store.js:311-328` — `stop()` clears the interval but does
not cancel the fetch set already awaiting in `runPollCycle()`. That cycle still resolves and
still publishes to subscribers after the store was stopped, so a stale payload can render onto
an unmounted dashboard. **Fix:** an `AbortController` per cycle, plus a re-check of the running
flag before `publish()`.

---

## F7 — MAJOR: the citation checker verifies line NUMBERS, not line CONTENT

@c0de4c2e mutation-tested this and their finding is stronger than mine, so I am recording
theirs: `check-source-citations.test.js` confirms a cited line *number* fits the file but never
checks what is *on* it — they replaced an entire cited function with garbage at identical line
count and **the test stayed green**. A checker that passes on garbage is worse than no checker,
because it is cited as evidence. Fix is a content assertion (anchor on the symbol text).

My adjacent observation stands: the test also fails **spuriously**, going red in the live
worktree right now purely because other agents shifted line numbers in cited files. So it fails
when nothing is wrong and passes when everything is wrong — both directions broken.
`check_doc_citations.py --fix` exists to repair drift, which is the right idea, but it rewrites
`docs/ARCHITECTURE.md` in place (`check_doc_citations.py:95-99`), so an interrupted run leaves a
half-written doc. **Fix:** anchor on symbols rather than line numbers, assert content, and make
`--fix` atomic (temp file + `os.replace`).

---

## F8 — MINOR: non-atomic artifact writes in the model build path

Given this branch already fixed *"verify_model.sh must not certify a model it never loaded"*
and *"promotion step could silently no-op and certify the old model"*, the remaining in-place
writes are the same class of hazard:

- `scripts/lib/write_static_cache_metadata.py:202` — `inference_metadata.yaml` written in
  place; an interrupt leaves a half-written metadata file that later steps will read as valid.
- `scripts/lib/write_tokenizer_assets.py:87` — companion files copied one-by-one with no
  staging; a mid-run failure leaves a partially promoted tokenizer set.

**Fix:** write to a temp file in the same directory, then `os.replace()`.

## F9 — MINOR: two brittle string probes

- `examples/serving-dashboard/run-demo.sh:173` — `grep -q 'static_cache'` matches the substring
  anywhere in the file, including a comment. Parse the actual `model.io.static_cache` key.
- `prometheus-parse.js:183-208` — histogram sub-samples only fold into a family if the
  `# TYPE` line was seen first. A payload that omits or reorders `# TYPE` silently loses the
  histogram, which this dashboard would render as "not measured."
- `scripts/verify_model.sh:111` — `port_is_busy()` probes `/` rather than `/health`.

## F11 — MAJOR: all six panels ship an unnamed `role="group"` (AC29)

Handed to me by @086345a5 while tracing a typedef; verified at `68726d17` and reproduced
empirically.

`dashboard/index.js:179` — `createRovingGroup(root, { label: panel.title })`

`panel.title` does not exist. `PANELS` is built at `index.js:115-118` as
`Object.freeze({ id: module.meta.id, module, modes: modesFor(module) })` — `id` is hoisted out
of `meta`, `title` is not, and the object is frozen so nothing can add it later. The title lives
at `panel.module.meta.title`, which `dashboard/registry.test.js:147` uses correctly.

Reproduced against the real registry:

```
throughput     panel.title=undefined   module.meta.title='Throughput & latency'
scheduling     panel.title=undefined   module.meta.title='Scheduling & batching'
kv-memory      panel.title=undefined   module.meta.title='KV memory'
prefix-cache   panel.title=undefined   module.meta.title='Prefix cache'
requests       panel.title=undefined   module.meta.title='Requests'
system         panel.title=undefined   module.meta.title='System'
=> 6/6 panels pass label:undefined ;  own keys: ['id','module','modes']
```

**Severity is bounded but real.** `panel-kit.js:1034` guards with `if (options.label)`, so no
literal `"undefined"` is announced — good defensive coding, and it is why this is not critical.
But `panel-kit.js:1033` sets `role="group"` **unconditionally**. The page therefore ships six
anonymous groups: a screen-reader user hears a group boundary on every panel with nothing
identifying which panel they entered. An unnamed `role="group"` is worse than no role, because
it adds navigational structure that carries no information.

### The part that matters most to my lane: the tests cannot catch this

`accessibility.test.js` calls `createRovingGroup` seven times (lines 264, 275, 298, 309, 340,
350, 361) and **every single call hardcodes `{ label: 'KV cache' }`**. The tests supply the one
input that is broken in production, so they can never observe the real caller. The function is
well covered; the integration has zero coverage. This is the textbook version of "would this
test fail if the code broke?" — no, and it didn't.

### On the fix — I disagree with the proposed one, respectfully

@086345a5 recommends hoisting `title` alongside `id` so the flattening is consistent. That
removes the trap but **contradicts this file's own ratified lesson**. `index.js:58-80` is a long
comment explaining that `modes` used to be hand-declared separately from each panel's
`meta.requires`, the two drifted, and both `kv-memory` and `prefix-cache` silently vanished from
the batching server as a result. The resolution recorded there was to **derive rather than
duplicate** — "a panel now answers 'where do I belong' exactly once." Hoisting `title` adds a
second copy of exactly the kind that already burned this file once, and invites a third field
next month.

**Recommended instead, in priority order:**

1. **Fix the call site** — `{ label: panel.module.meta.title }`. One line, no new duplication.
2. **Make the silence impossible** — `createRovingGroup` should refuse to set `role="group"`
   without an accessible name, or throw in development. This codebase already uses exactly that
   pattern: `modesFor` at `index.js:151` throws rather than guess, with a message explaining
   the consequence. An accessibility primitive that silently degrades is the same class of
   problem, and fixing it here protects every future caller rather than this one call site.
3. **Add the missing integration test** — mount the real dashboard and assert every
   `[role="group"]` has a non-empty `aria-label`. Reading from real `PANELS`, not a literal.

If the team prefers consistent flattening anyway, the coherent version is the *other*
direction: drop `id` from the registry too and read everything through `module.meta`, so there
is exactly one home for panel metadata. I would not block on which of these is chosen — but I
would block on (2), because it is what turns this bug class from silent into loud.

## F10 — MINOR: `size_blocks` silently keeps the old mirror on re-attach

`crates/onnx-genai-kv/src/telemetry.rs:192` — `let _ = self.blocks.set(...)` on a `OnceCell`.
Documented as "called once, on attach," but a second attach with a *larger* pool silently keeps
the old, smaller mirror; pages past the old cap then read as absent. Either assert single
attach or handle resize explicitly rather than discarding the `Result`.

---

## What is genuinely well done

- **`pack_block` / `block_window`** (`telemetry.rs:95`, `:225`) — the bit layout is correct,
  saturating rather than wrapping (a >65535 refcount is a bug you want visible, not wrapped to
  0), and the present-flag-at-bit-40 scheme means an unwritten page reads as *absent* instead
  of as a wall of zero blocks. The comment explaining that mirror order is *inherent* rather
  than sorted — "a block table that reshuffles shows motion that did not happen" — is exactly
  the kind of WHY comment this codebase should have more of.
- **`BlockTableResponse::pending` vs `::not_applicable`** — separating "we have not decided
  yet" from "the mechanism is not in play" is a real distinction that most codebases collapse,
  and `pending_and_not_applicable_are_distinct_codes` locks it in. The docstring on
  `kv_blocks_renders_no_numbers_until_paged_kv_is_known_to_be_in_use` explaining why it asserts
  the *invariant* rather than one specific code — because the first version "defended the bug
  instead of catching it" — is the single best piece of test reasoning on this branch.
- **`formatField`'s unit-doubling fix** (`format.js:127-131`) — the `appendUnit` rule, with the
  comment recording that "32,768 tokens tokens" shipped to a live page while every unit test
  passed, is a good fix *and* a good explanation of why the obvious API was wrong.
- **`dashboard/package.json`** — using the `description` field to explain that the file is not
  an npm package and exists only so `node --test` sees ES modules is a genuinely clever place
  to put that warning, since it is the file someone would otherwise "helpfully" add deps to.
- The `state-channel.test.js:234` warning to **never** global-replace `'ok'` because three
  unrelated vocabularies share the string is the kind of institutional knowledge that normally
  gets lost. It should have been heeded in `format.test.js` (F2), but recording it was right.
- **`renderStateOf` in `dashboard/field-state.js` is the best-built function on this branch.**
  It fails **safe** on `undefined` (→ `unavailable`, never a fabricated measurement) and fails
  **loud** on garbage, and its error message does what almost none of ours do — it names the full
  ruled vocabulary, then tells you which of two possible causes you are looking at and what to do
  about each: *"if this is a new ruled state, add it to `STATE_ALIASES`; if a producer has drifted,
  fix the producer."* That is a diagnostic, not an assertion. It anticipates the reader's next
  question and answers both branches. **This is the standard for error text on this branch.**
  (Its tolerance for `'ok'` is also the reason the enum argument was unresolvable by experiment —
  a design cost worth naming, but the function itself is exemplary.)
- **`repair-citations.mjs` is well designed despite F12** — it declines loudly rather than skipping
  silently, never invents a file, only rewrites on exactly one definition-shaped match, and carries
  a regex specifically guarding a struct-field false positive 342 lines from the real definition.
  Someone hit that, understood it, and wrote the guard. Its single defect is the artifact it reads,
  not its logic.
- **`check-source-citations.test.js` went red on its first live run with zero false positives**,
  and its error text — *"The code moved: X appears at :A, :B. Update the citation"*, listing every
  occurrence rather than confidently naming one — made a five-item report actionable in a single
  read. Its line *"A citation that resolves is not a citation that is correct; this is the
  difference"* is the best sentence produced on this branch, and it generalises well past prose.
- **`.gitignore`'s `node_modules/` entry** landed with the comment *"agents here commit with
  unscoped adds"* — closing a real hazard by construction rather than by asking people to remember.

---

## Required before merge — FINAL, at committed `9e31a7c7`

1. **F1 (blocker)** — derive paged applicability from the engine, not from
   `!continuous_batch_supported`. Make `runtime.rs:263 attach_kv_telemetry` return `bool` the
   way its pipeline sibling already does, then `set_applicable(paged && !continuous_batch_supported)`.
2. **F3 (blocker)** — collapse the two render stacks, or document and test which one wins per
   surface. `dashboard/field-state.js` says in its own header that it should be deleted.
   Minimal wire-neutral step available today: rename the **key** `OK → MEASURED` in
   `RENDER_STATES` (2 call sites, both `store-adapter.js`). Then sweep the seven raw `'ok'`
   producers to the constant. See Re-verification #5 for the behavioural proof, and **F14 for the
   one line where the two vocabularies physically touch the DOM.**
2b. **F14 (minor, but fix it with F3)** — `ui/model-card.js` / `renderCardField` writes
   `element.dataset.state = field.state` raw. It is the only one of nine `data-state` writes that
   skips normalisation, and the only place the text and style channels can disagree. Latent today
   (the root store emits no `'ok'`), untested at any value. One-line fix.
3. **F15 (major)** — `dashboard/scheduling.js` binds `scheduler.max_batch`, `scheduler.running`,
   `scheduler.waiting` and `kv.allocation_failures`; **no client module publishes any of them.**
   The server emits the denominator as `batch_capacity`, which nothing client-side reads. Fix the
   binding, then add the registry test that fails on `"No field named"` for any panel-requested key.
4. **F11 (major)** — `index.js:188` passes `panel.title`, which does not exist; all **five** panels
   ship an unnamed `role="group"` plus a console error at every mount. Fix by deriving the label the
   way `id` is already derived, not by hoisting a second copy of the title into the registry — that
   would contradict the derive-don't-duplicate lesson this same file documents at `:58-80`. And fix
   `accessibility.test.js`, which hardcodes `{ label: 'KV cache' }` in all 7 calls and therefore
   **cannot** catch this class of defect. **No landing-order constraint** — the `createRovingGroup`
   throw at `panel-kit.js:1049` is gated on `IS_DEVELOPMENT`, so this fix can land alone.
4. **F12 (major)** — `repair-citations.mjs:111` computes citation lines from the **dirty working
   tree** and writes them into committed documentation. Read via `git show HEAD:<file>` and refuse
   to run when the target is dirty. **It has no tests.** `driver.rs` is still dirty, so **re-running
   the tool reproduces the defect** — do not reach for it to clear the red.
5. **Five README citations** must be re-anchored to symbols rather than line numbers. `driver.rs`
   has moved ~100 lines three times tonight and is under active development; no hand-maintained
   or dirty-tree-computed number can win that race.
6. **F4 (MINOR, downgraded)** — `panel-kit.js:272,453` use `??`, so `null` cannot mean "no ceiling".
   **No live caller passes `null` any more** (`prefix-cache.js` was deleted), so this is latent. Fix
   is still right — `=== undefined`, and widen the JSDoc at `:449` which does not admit `null`.
7. **F13 (MINOR)** — `honesty.test.js:257` hardcodes the `requires` vocabulary as a second source of
   truth alongside `modesFor`'s switch. Derive it.
8. **F5 (major)** — `audit_citation_targets.py:25`. **@e00032a4 reports this FIXED and
   mutation-proved; not personally re-verified.**
9. **Run QA in a browser.** Per AC52 this review cannot close the gate on its own. @0837fdf9's
   served-bytes run and @c7a654ed's 17/17 module check are the first real browser evidence tonight,
   and neither is mine. **This remains the largest open risk on the branch.**

**Not required — verified already done and clean:**
- The consumer audit @12e42da8 ordered. Zero shipping consumers compare against a string literal,
  and it is now permanently guarded by `ok 74 - honesty lint — the measured state is never
  compared as a literal`.
- The bidirectional selector-vs-enum guard ordered built. Exists as `state-treatments.test.js`;
  13/13 green covering enum→CSS and CSS→enum.
- Any edit to `shell.css:163`, the field-state enum values, or `formatFieldText` (which no longer
  exists). See Re-verification #5.
- Item 6 of the previous list (`state-treatments.test.js:7` tense) — that file's assertions are
  green at `9e31a7c7`; the stale-tense hazard has since been reported in `demo-spec.md:1245/1254/1303`
  instead, which is @376a0297's to correct in place.

## A pattern worth naming

Twice on this branch a red test was resolved by editing the test rather than the code:
`45c1103d` rewrote the AC45(c) ceiling rule around `requests.js` (F4-revised), and `24d831a2`
landed a rename that left `format.test.js` asserting the opposite of the ruling (F2). In both
cases the rewrite was *defensible on its own terms* — the new AC45(c) test genuinely is better
in one respect. But the underlying implementation defect survived both times, and in the F4
case it is now entirely unguarded. When a tripwire this branch built goes red, the default
should be to suspect the code.

**And its sequel, which is the note I would most like to leave the branch on.** The JS suite
finished the night **484/484 green**, and five defects are live inside that green — including
an accessibility bug on every panel and a linter that cannot fail. The crew spent tonight
learning that *a clean `git status` is not evidence*, that *a 200 is not a working page*, and
that *a fresh binary is not the binary you ran*. **A green suite belongs on that list.** Each of
those is the same shape: an output that reads identically whether the thing is right or was
never checked. The five findings above are not places the tests failed. They are places **no
test was ever pointed**, and a green bar cannot tell you the difference.

> **TENSE STAMP (re-measured `1bca52a8`) — and the passage above earned its own lesson the hard way.**
> `484/484` was the count when written; the canonical runner later reported **632 / 95 suites / 0 fail**.
> **"A linter that cannot fail" is F5, STRUCK at `1b4d76c6`.** The paragraph's *argument* stands and I
> am not withdrawing it — but it is a closing note, the part a reader quotes, and it carried two
> expired facts. **That is the whole thesis of this document happening to the sentence that states
> the thesis.** A rhetorical passage is exactly where staleness hides, because nobody re-measures
> a conclusion.


---

## A second pattern worth naming: **a fixture that supplies the data is not testing that the data exists**

This one has enough instances on this branch to be a class rather than four separate findings, and
it is the sharpest test-quality defect here. In each case a panel or a field renders **perfectly
green in CI and can only ever be blank, or false, against the live wire** — because the test hands
it a value the server cannot produce.

| field / behaviour | what the fixture supplies | what the wire can deliver |
|---|---|---|
| `kv.prefix_evictions` | `measured(...)` in `panels.test.js` | **nothing** — the symbol exists only in the offline CLI profiler |
| `kv.hot_evictions` | `measured(12, ...)` | real, but the fixture proves nothing about that |
| resource-snapshot deferral | fixtures generate in <1ms | the regression needs a request *inside* the batch window |
| chat prefix-cache hits | fixture hit counts | `+24 hit_tokens/request` with `loaded_prompt_prefix` never set — **arithmetically correct, materially false** |

The common shape: **the test asserts that the rendering code handles a value correctly, and is read
as evidence that the value is real.** Those are different claims, and the suite cannot distinguish
them — a fixture-fed panel is green whether the producer works, is broken, or was never built.

The resource-deferral case is the purest specimen, because its author *wrote the limitation down in
the test itself*: the helper-level test "structurally cannot see" the regression and **"passes
whether or not the bug is present."** That is not weak coverage. **It is a green claim about
nothing, and on the summary line it is indistinguishable from the strongest test in the suite.**

The fix in every case is the same and it is not a better assertion — it is a **different
observation point**. Assert against the payload the server actually emits (or against a recorded
capture of it), not against a hand-authored object that asserts the schema you hoped for.

**This is why F1 matters more than its size suggests.** It is the server-side twin: a field whose
*applicability* is inferred rather than observed, so the page can confidently render "not
applicable" for a subsystem that is running, and no fixture-fed test can ever notice.

---

## F14 — MINOR (latent, unguarded, untested) — `ui/model-card.js` / `renderCardField` is the only `data-state` write that bypasses normalisation

Verified in a clean detached worktree at `2a104dcc`, zero dirty files.

```js
// ui/model-card.js, in renderCardField
element.textContent = rendered.text;      // from formatField  — normalised, refuses unknown
element.dataset.state = field.state;      // RAW — not normalised
```

There are nine `data-state` writes in shipping JS. **Eight are safe**: `panel-kit.js` writes
`{'data-state': state}` where `state` is the return of `renderStateOf`; the others write frozen
enum members (`RENDER_STATES.UNAVAILABLE`, `RENDER_STATES.NOT_APPLICABLE`), fixed literals, or the
separate connection vocabulary in `app.js`. **`renderCardField` is the only one that writes a
field's own state through.** Its imports show the mechanism: it takes `formatField` and
`FIELD_STATES` from the **root** stack and never imports `renderStateOf` or `normaliseState`.

**Why it matters:** this is the one place on the page where the *text* channel and the *style*
channel are derived from different sources and can therefore disagree. `formatField` strips the
value honestly (`text: '—'`, `hasValue: false`, console warning); the raw state lands in the DOM and
matches none of the five selectors, so it renders at measured contrast. Everywhere else both
channels descend from a single normalised value and **cannot** diverge.

**This is F3's seam.** F3 is not a naming complaint — two vocabularies coexist, and this is the one
line where they physically touch the DOM.

**Severity is MINOR and I am not inflating it.** `model-card` is fed by `createTelemetryStore` from
the root store, which emits `FIELD_STATES.*` and contains **zero** `state:'ok'` producers. All seven
raw `'ok'` emitters live in `store-adapter.js` / `testing/fake-store.js`, which feed the dashboard
stack. **The two do not meet today.** Same call, same reasoning, as the `'ok'` literals.

**Untested.** No test asserts `model-card`'s `data-state` at any value. The only mention of
model-card in any test file is a comment in `format.test.js`: *"Found in a browser, not in a test"* —
about a different model-card defect. That file has now marked the same coverage boundary twice.

**Fix:** route the write through the normaliser `panel-kit` already uses. One line, no wire change.

### On the proposed CSS catch-all (@376a0297)

The CSS observation is **confirmed** — no bare `[data-state]` rule, no `:not()` fallback, five exact
matches only, and `[data-state='measured']` sets only the default body colour. But the stated
consequence does not follow: **the value is never rendered.** `formatField` returns an em-dash with
`hasValue: false` for any unrecognised state, and `renderStateOf` returns `'unavailable'` in the
browser and throws under `node --test`. An unknown state produces *an honest em-dash in measured
styling* — a **degraded** signal, not a **false** one.

The compensating control is documented directly above the selectors, in `shell.css`:

> *"a panel that forgets to set data-state renders at measured contrast, **which is why the store
> never hands out a value for a field that has none.**"*

Three layers cover this: the store invariant, both formatters, and the strict-mode throw. **Add the
catch-all as defence-in-depth — it is cheap and converts a silent degradation into a loud one — but
fix `model-card` first.** Styling around a divergence is not the same as removing it.

### Bonus: the `shell.css` saga has a documented answer

The same comment block records, in the past tense, the exact state the repeatedly-ordered edit would
have re-created:

> *"They drifted once: the constant was spelled MEASURED while its value stayed 'ok', so this
> selector said 'measured' and therefore matched NOTHING. Every measured value silently fell back to
> inherited styling, which looks close enough to correct that only a cross-language check finds it."*

Every stop-order on this axis tonight reduces to the same cause: **the file contains both the
defect's history and its guard, and the order was a correct reading of a true sentence about a state
that no longer exists.** Third instance tonight of a fix sitting in a comment above the code it
protects.

### Method note against myself

My first probe returned `renderStateOf({state:'ok'}) -> 'unavailable'` and I had a regression report
half-written. **It was wrong.** The probe object had no `value`, and the function's docblock states
that it downgrades a valueless field and warns that passing a Series *"reports every live series as
unavailable."* Re-probed with `value: 42`: returns `'measured'`, as it always has. **The docblock is
the only reason that took sixty seconds instead of a broadcast.**

---

## F15 — MAJOR — the scheduling panel binds four keys no client module publishes

Verified in a clean detached worktree at `c6aa54e1`, `feat/genai-demo-dashboard`, zero dirty files.
Two independent methods, with a control.

`dashboard/scheduling.js` reads eight keys through `field()`. Against a payload shaped like
`admin.rs`, four of them are **structurally absent** — not merely unpopulated:

```
  *** scheduler.running        unavailable  "No field named ... is published by this server build"
  *** scheduler.waiting        unavailable  "No field named ..."
  *** scheduler.max_batch      unavailable  "No field named ..."
  *** kv.allocation_failures   unavailable  "No field named ..."
      admission.slots_available  pending    <- registered, awaiting poll
      batch.active_size          pending    <- registered
      queue.depth                pending    <- registered
      admission.rejections       pending    <- registered
```

**The discriminator is poll-independent, which is what makes this solid.** `pending` means *the
store knows this key and is waiting for data*; `"No field named"` means *this key is not registered
anywhere*. Whether a poll ever completes is irrelevant. Second method, agreeing exactly: grepping
the provenance catalogue returns **0** for those four keys and **≥1** for the other four.

### The wire contract is mismatched at the name

```
batch_capacity      client-side, all non-test modules -> 1 occurrence, inside a PROSE STRING
scheduler.max_batch client-side                        -> 1 occurrence, the READER. Zero producers.
```

The server emits `batch_capacity` deliberately and correctly (`admin.rs`, with a doc comment
explaining why it is not named `max_batch`). **The client never asks for it.** The server publishes
the right number under one name; the client requests it under another; nothing connects them.

Every ruling about the batch denominator — how to render it, whether a hardcoded `4` is forbidden,
the fifteen `3 of 4` comments — concerns **a value that cannot reach the panel.**

### Why the degradation is invisible

It is **honest**. `renderOccupancy` falls back to the bare in-flight count ("6 in flight", never
"6 of 4 max"), exactly as its own comment promises, and the field renders an em-dash with a reason
string. The page tells the truth — permanently, about a feature we believe we shipped. Good design
is precisely what conceals this.

### Why 480+ green tests cannot see it

```js
dashboard/scheduling.test.js  'scheduler.running':      measured(6, { label: 'Running' })
                              'scheduler.max_batch':    measured(4, { unit: 'sequences' })
                              'kv.allocation_failures': measured(0, { unit: 'count' })
```

`createFakeStore(spec)` lets the test supply whatever keys the panel asks for. The test proves the
panel *renders* a denominator; it is read as proving the panel *has* one. **Different claims, and
the suite cannot distinguish them.** Note the fixture supplies the literal `4` — the number the
hardcode ban exists to forbid.

`telemetry-store.test.js`'s `statusBody()`, whose docstring says it is shaped like the real
server's, **omits `batch_capacity` entirely**. The one payload field added for this purpose appears
in no fixture anywhere.

**This is the sharpest instance of the fixture pattern above: it costs an entire panel.**

### Fix — the test matters more than the code

1. Bind `batch_capacity` (or map it onto a registered key). Decide for `scheduler.running`,
   `scheduler.waiting` and `kv.allocation_failures` whether each is real or should be dropped.
2. **Add the registry test.** Statically extract every key any panel passes to `field()` — 53
   distinct today — resolve each against a real store, and fail on `"No field named"`.
   *Mutation:* point any panel at a typo'd key. The panel still renders, every visual check passes,
   every fake-store test stays green, **and this suite goes red.** One line of coverage for a class
   that currently has none.

### Method note — two false alarms caught in one investigation

I probed the raw `telemetryStore` first; panels receive the **adapter**. I then probed with an empty
payload and got *"31 of 53 keys dead"* — an artifact, because `field()` reads `snapshot.fields[key]`
and that is payload-driven. **I had a `34 of 53` broadcast half-written.** The
`pending`-vs-`"No field named"` discriminator survives both errors and is the only reason the final
number is trustworthy. I verified the wrong noun twice before verifying the right one.

### Incidental: the panel count already passes

`import('./dashboard/index.js')` prints `panels: 5` — `throughput, scheduling, kv-memory, requests,
system`. `dashboard/panels/prefix-cache.js` does not exist. The ordered "five deletions" has nothing
to delete; its own acceptance criterion passes at `c6aa54e1`.

---

## Late session: one retraction, one new finding, one governance note

All three below were published as broadcasts before they were written here. That ordering is itself a
defect — see the closing note.

### F11 — RETRACTED

I filed F11 claiming `panel.title` was undefined on registered panels. It is defined on all five.
Verified by execution, not inference:

```
throughput  panel.title="Throughput & latency"     kv-memory  panel.title="KV memory"
scheduling  panel.title="Scheduling & batching"    requests   panel.title="Requests"
system      panel.title="System"
```

`PANELS` lifts `title` out of `module.meta` deliberately. The Readability Reviewer filed the identical
finding independently and withdrew it first. **Both of us verified the call site and *inferred* the
registry.** Two reviewers, one defect, one mechanism — which is why it is worth recording rather than
quietly deleting. Merge-checklist item 4 is struck.

### F16 (~~MAJOR~~ → **LARGELY CLOSED, re-measured at `1bca52a8`**) — a guard whose scope, not construction, was too narrow

> **READ THIS BEFORE THE TABLE BELOW. The table is a photograph of an older tree and its
> headline conclusion no longer holds.** Re-censused at `1bca52a8` with the denominator published
> first: **7** tracked `.md` files carry the withdrawn figure, and the guard's corpus now reaches
> **5** of them. Only `demo-spec.md` and `design/demo-ux.md` remain exempt carriers, both as
> *documented promises with named paragraphs* — and `check-perf-claims.test.js:240` now computes
> `staleDeferrals`, failing when an exemption outlives its reason, on the stated ground that
> *"an exemption that outlives its reason is indistinguishable from a suppression."* @fc8b5d97
> retired one exemption outright and recorded the 0-occurrence measurement that justified it.
> **My original conclusion — "every instance is unreachable except the one that does not carry the
> defect" — was true when taken and is false now.** The finding that survives is much smaller: two
> exempt carriers, both tracked by a mechanism that did not exist when I filed this.
>
> **And the row for F16 in Re-verification #3 named the wrong file entirely** — it described
> `check-field-states.test.js`, which belongs to F2, not to F16. Two findings fused into one row.

`check-perf-claims.test.js` contains a test named `no document presents a withdrawn prefix timing
figure without its noise floor`. It is green. Its reach versus where the withdrawn figure actually
lives:

| file carrying the withdrawn figure | in scope? |
|---|---|
| `design/demo-ux.md` | ❌ explicitly exempted |
| `demo-spec.md` | ❌ explicitly exempted |
| `telemetry-provenance.js` | ❌ wrong file type |
| `prefix-counters-forbidden.test.js` | ❌ wrong file type — **assertion message** |
| `dashboard/honesty.test.js` | ❌ wrong file type — **assertion message** |
| `dashboard/registry.test.js` | ❌ wrong file type |
| `check-perf-claims.test.js` (the guard itself) | ❌ wrong file type |
| `READABILITY-REVIEW.md` | ✅ in scope — carries no withdrawn timing figure |

The corpus is `git ls-files '*.md'` minus two named files minus `design/`. **Of the files carrying the
figure, every one is unreachable except the one that does not carry the defect.** The guard's own
exemption comment names the offending paragraphs and then skips them.

This is not sloppiness — the test is otherwise the best-constructed guard on the branch. It carries an
explicit anti-vacuity assertion (`docs.length > 0, 'this check would pass vacuously'`), it scopes by
paragraph and requires prefix-discussion and a percentage delta to co-occur, and it argues its own
false-positive reasoning in a comment. **It fails on *scope*, not on construction — a rarer defect: a
guard can be built perfectly and still be pointed at a set that excludes every instance.**

**Fix.** Do not simply widen the glob to `*.js`: the guard's own comment and
`telemetry-provenance.js` state the figure *in order to retire it*, so a widening reddens the
explanation. Assert instead on **assertion messages**, which are mechanically distinguishable from
comments:

> No string literal passed as an `assert` message may state a withdrawn figure.
> Mutation: restore the figure to any assertion message → RED. Empty-input mutation: if zero assertion
> messages are extracted, FAIL rather than pass.

**Why that surface first.** A test's failure message is read at the moment a developer is debugging and
trusts the repo most; it is invisible until something breaks; and it is the only prose that arrives
pre-framed as *the reason you are wrong*. Two sites currently state a withdrawn measurement in exactly
that voice.

*Do not sweep `check-readme-claims.test.js` with the others — it asserts the **disagreement** between
the two arms rather than either number, so it survives the withdrawal and is the template the others
should become.*

### Governance note: when editing a red test is the correct move

`dashboard/registry.test.js` pins `PANELS.length === 5`, `existsSync('./prefix-cache.js') === false`,
and `PANELS.some(id === 'prefix-cache') === false`. Any ruling that restores the panel turns these red.

The standing rule is that nobody makes a red test green by touching the test. That rule and this
situation are not in conflict, and the line is worth stating precisely:

> **Editing a test because it is inconvenient is the banned move. Editing a test because the ruling it
> encodes was superseded is the required move. The difference is whether *the requirement* changed or
> *the evidence* changed — and only the ruling's author can say which.**

The test wrote that procedure into its own failure message: *"the panel was cut by ruling; re-adding it
needs a new ruling, not a merge."* It stopped a silent merge and escalated to the party entitled to
decide. **Best-designed test on the branch.** If a restoration is authorised, invert the assertions in
the *same commit* — assert the panel exists and binds an empty field set — rather than deleting them.
Deleting discards the ruling along with the assertion; inverting keeps the ruling and re-points it.

### Closing method note, against myself

Every finding in this section reached the crew as a broadcast before it reached a file, and this
document reached the repository last of all. A file written outside a repository produces a perfectly
clean `git status` — byte-identical to work that committed successfully. **We hardened our evidence
stamp five times (sha → branch → porcelain → node version → toplevel) and none of those fields can
distinguish *saved* from *never present*.** The instrument that sees it is `git ls-files <path>`
returning nothing.

---

## F17 (MAJOR, new) — the field-key guard cannot see dynamically-built keys, and is green because of it

Measured in a clean detached worktree at `13d9214b`, `git status --porcelain` = 0.

`dashboard/field-keys.test.js:105` extracts keys with a matcher that requires a
single-quoted string literal:

```js
source.matchAll(/\.(?:field|series)\(\s*'([a-z0-9_.]+)'/g)
```

`dashboard/throughput.js:274` does not supply one — it builds the key from a
template literal, inside a loop over five definitions (`throughput.js:253-258`)
and three percentiles:

```js
const field = telemetryStore.field(`${definition.prefix}_${percentile}`);
```

Five prefixes x three percentiles = **15 keys the extractor never sees**.
Resolved against both sources of truth:

| bucket | count |
| --- | --- |
| listed in `NOT_YET_PUBLISHED` | 2 |
| registered in `telemetry-provenance.js` | 0 |
| **known to neither** | **13** |

The 13 are every `p95` and every `max` in the latency table, plus all of
`latency.itl_client_*`, `latency.tpot_client_*` and `latency.e2e_server_*`.

The guard's own test name is `has no unexplained key — an unlisted one is almost
certainly a typo`. It passes. It passes because it extracted zero keys from that
function, and zero keys produce zero violations.

**Fix — do not write a cleverer regex.** Static evaluation of template literals
is the wrong direction. Make the guard fail-closed: scan for `.field(` / `.series(`
*not* followed by a quote, and assert that set is empty, with a message naming
file and line and stating that the key is built dynamically and is therefore
invisible to this test. A key the guard cannot read must turn it red, not
invisible — the same rung already ratified for symbol citations: decay must fail
loud. Triage of the 13 into catalogue or register is a separate question and is
not pre-judged here.

### Why this is a class, not an incident

Third specimen of one failure mode on this branch: `page-claims`' coverage list,
F16's corpus glob, and now this matcher. None of the three is *wrong*. All three
are *narrow*. A narrow checker does not fail — it passes, credibly, and certifies
the territory it cannot see. A guard's assertion is tested by its suite; its
**scope** is tested by nothing.

### A method note against myself

The first pass of this finding counted `.field()` **call sites** and concluded
`throughput.js` was 6/6 permanently blank — that the headline panel rendered
nothing. That was false, and it was killed before it was sent. Deduped, the panel
has 3 distinct `field()` keys, and the hero number does not use `field()` at all:
`throughput.js:100` and `:127` read `telemetryStore.rate()` and `.rateSeries()`
on `metrics.tokens_generated_total`, both live. The dead surface is the latency
row, not the panel.

The falsification was run *because the number was big enough to be quotable*. The
operative rule was therefore *check hardest when the finding is most quotable* —
which is the reverse of the usual failure direction, where the pessimistic finding
goes unchallenged because nobody wants to argue for good news.

## F15 — closed in half, changed in character

The denominator fix landed correctly: `scheduling.js:113` now binds
`batch.capacity`, the catalogue resolves it, and `field-keys.test.js` is 4 pass /
0 fail. The **typo** half of F15 is closed.

The **substance** half is not; it has been reclassified rather than fixed.
`scheduler.running`, `scheduler.waiting`, `kv.allocation_failures` and
`queue.depth_peak` now sit in `NOT_YET_PUBLISHED` — 4 of scheduling's 9 fields,
and 10 of `kv-memory.js`'s 13, are permanently em-dashed by declaration.

That is the right mechanism, and it is an improvement on a silent blank. The
residual concern is governance, not code: the register holds 26 entries, carries
no owner and no expiry, and nothing re-asks whether an entry is still true. It
converts an unresolved red into a permanent green. That is a gate question.

---

## F18 (MAJOR, new) — a duplicate catalogue key is undetectable by any runtime test

`telemetry-provenance.js` briefly carried two definitions of `'batch.capacity'`
(found by another reviewer at `bc8ef473`; already repaired by `185d6720`, and the
survivor is the symbol-anchored entry, which is the right outcome).

The finding is not the duplicate. It is that **no test could have caught it, and
none ever can.** Established by mutation in a clean worktree at `185d6720`,
`--porcelain` 0, by re-introducing the defect:

| | catalogue entries | suite |
| --- | --- | --- |
| baseline | 37 | 520 pass / 0 fail |
| duplicate re-introduced | 37 | 520 pass / 0 fail |

A duplicate key in an object literal produces one key. `Object.keys().length` is
invariant, so no counting assertion can observe it, and no property of the parsed
object differs in any way. A guard written against the runtime object would be
green against the defect as readily as against the fix — its green would carry no
information.

The corollary is the uncomfortable half: **the repair is equally unverifiable.**
Nobody can demonstrate the duplicate is gone by executing anything. Defect and
repair have identical runtime signatures — the same shape as a commit that
silently fails and a grep that matches nothing.

**Fix:** a source-level check. Parse the object literal (or scan textually for
repeated `'key':` at the catalogue's top level) and assert each key appears once.
This must be a text check, deliberately, and the reason should be written into the
assertion message so the next reader does not "improve" it into a runtime check
and silently disarm it.

The cost of the live defect was never runtime behaviour — both entries agreed on
`source`, `path` and `classification`. The cost was that the file stated one
field's provenance twice and the reader could not tell which the program believed.
That is the exact defect class this product exists to refuse, located in the
provenance table itself.

## F15 — CLOSED

Both halves resolved. The denominator binds `batch.capacity`, registered against
`/v1/status` with `path: 'batch_capacity'`, carrying a symbol-anchored citation to
`AppConfig::effective_batch_capacity()` and a comment preserving the reason
`max_batch` alone is wrong. Verified at `185d6720`: `scheduler.running`,
`scheduler.waiting` and `kv.allocation_failures` are all declared in
`NOT_YET_PUBLISHED`.

Four of `scheduling.js`'s nine fields and ten of `kv-memory.js`'s thirteen are
permanently em-dashed **by declaration**. That is an honest state, not a defect,
and F15 is closed. The residual is governance, not code: the register holds 26
entries with no owner and no expiry, and nothing re-asks whether an entry is still
true — it converts an unresolved red into a permanent green.

## Note on F17's fix, deliberately not landed

The fail-closed matcher for F17 is eight lines and is ready. It is not being
landed, because it would turn a green tree red at demo time over 13 keys in a file
this reviewer does not own, and the triage of those 13 belongs to that owner. A
reviewer who reddens a shared tree to prove a point has stopped reviewing.

---

## F19 (MINOR, new) — the in-flight fixture repair left its affordance armed

The last suite failure (`telemetry-store.test.js`, `the in-flight gauge is NEVER
exposed as the engine batch size`, `2 !== 8`) is fixed at `712cc39b`: 60 pass /
0 fail in a clean detached worktree, `--porcelain` 0.

The fix is good. `batch.in_flight` migrated from `/metrics` to `/v1/status`; the
test now injects through `statusBody({ batch_in_flight: 8 })`, all four assertions
survive verbatim including `assert.match(reason, /does not report/i)`, and the
comment was updated to describe the new path *and* to record the trap for the next
reader.

**The residue:** `telemetry-store.test.js:121` still declares
`metricsBody({ inFlight = 3, ... })` and `:127` still emits
`onnx_genai_batch_size_current ${inFlight}`. Measured at `712cc39b`:

| | count |
| --- | --- |
| callers passing `inFlight` | 0 |
| catalogue entries reading that metric | 0 |
| prometheus names injected / consumed | 8 / 7 |
| orphaned | 1 |

So the next person who needs to set in-flight will find `metricsBody({ inFlight })`
— the discoverable, natural-looking door — and it will silently inject nothing,
and their assertion will read the `/v1/status` fixture's value instead. That is
the identical bug that just held the gate, still loaded, with zero callers.

**Fix:** delete the `inFlight` parameter and the two lines it feeds. Zero callers,
zero risk. `statusBody({ batch_in_flight: N })` becomes the only door.

A comment warning about a trap is strictly weaker than not having the trap. The
comment only reaches someone reading the *repaired* test; the next person will be
writing a *new* one and meets the parameter first. When a function accepts an
argument that can only produce a wrong result, the fix is to stop accepting it.

### Governance ruling: a third permitted category

This edit was challenged against the rule recorded earlier in this review —
*editing a test because it is inconvenient is banned; editing it because the
requirement it encodes was superseded is required.* Neither clause fits. Here
**neither the requirement nor the evidence changed — only the wire address did.**
The edit changed the input path and touched no assertion. That is a third,
clearly permitted category, and it leaves the test's meaning entirely intact.

### A retraction against myself, and a broken instrument

Suspicion that the comment's mechanism clause (`incremented per HTTP generation
and decremented on drop`) had been carried across the migration and become false:
**wrong.** `metrics.rs:211` reads `current_batch_size` from `REGISTRY.batch_size`,
the same process-global static backing `onnx_genai_batch_size_current`, and
`admin.rs:167` documents it. One counter, two endpoints. The clause is true of
both.

The first blast-radius probe reported **8 of 8 prometheus names orphaned**. It was
entirely false: it matched `path: '<name>'`, but METRICS entries use
`metric: '<name>'`, so the extractor matched nothing and every name fell through
as an orphan. The corrected answer is 1. A finding that indicts everything is
usually indicting the instrument — 8-of-8 is the signature of a broken matcher,
not of a defect.

### Measured scope for the citation conversion

`telemetry-provenance.js` carries **38** `file.rs:NNN` line-anchored citations,
including `metrics.ttft` (`metrics.rs:119-123`) and `metrics.e2e_latency`
(`metrics.rs:141-144`). The `batch.capacity` entry is already symbol-anchored and
is the template.

### Well done, and worth naming

`crates/onnx-genai-server/src/tests.rs:3959-3975` documents what the test *cannot*
prove and why it stopped trying: `REGISTRY.batch_size` is process-global, `cargo
test` runs concurrently, the old absolute assertion passed alone and failed
roughly one full-suite run in three, and a lock could not fix it because the other
mutators would have to volunteer to take it. It then names the deterministic test
that proves the property at the arithmetic seam instead. A flaky assertion
correctly retired, with its replacement cited. That is the standard.

---

## F20 (MAJOR, new) — the canonical suite is defined by a phrase that resolves two ways

Measured in one pinned clean worktree at `ac7c7412`, `--porcelain` 0, node
v25.6.1, both numbers taken in the same invocation:

| scope | result |
| --- | --- |
| `examples/serving-dashboard/dashboard` | 289 pass / 0 fail |
| `examples/serving-dashboard` | 532 pass / 0 fail |
| **delta** | **243 tests across 25 files** |

The canon is currently stated as "the full dashboard directory". That phrase
resolves to either scope, and both are honest readings — so two careful people
will run two different suites and both will correctly report "canonical, green."
The published count being stale is benign; the suite has grown all session
(435 → 463 → 507 → 520 → 532). The ambiguity is the defect.

It is not pedantry, because of what falls in the 243:

```
telemetry-store.test.js          60   <- the gate's last red lived here
check-source-citations.test.js    5   <- gate item 5 is certified on this file
check-perf-claims.test.js         7       page-claims.test.js             10
check-readme-claims.test.js       4       prefix-counters-forbidden.js     3
denominator-binding.test.js       3   <- F15's guard
provenance-expiry.test.js         5       register-completeness.test.js    8
check-docstring-drift.test.js     5       never-bind.test.js               3
```

Every test enforcing the honesty bar lives outside `dashboard/`. Under the
narrower reading, the single failure that held gate item 2 red would have been
invisible and item 2 would have been certified green throughout.

**Fix:** state the canon as a command, not a place —
`cd examples/serving-dashboard && node --test`, 532 green at `ac7c7412`, node
v25.6.1. A command cannot resolve two ways. The node version is load-bearing and
not optional: v25 recurses into subdirectories and older releases do not, so on an
older runtime the *same command* silently collapses to roughly the 289 subset.
**The scope is decided by the runtime as well as by the path.**

### Fourth specimen of one class, now reaching the gate

`page-claims`' coverage list (F-class, another reviewer), F16's corpus glob, F17's
key matcher, and now the canon's own directory. None is wrong; all are narrow. A
narrow scope does not fail — it passes, credibly, and certifies the territory it
cannot see. A test's assertion is checked by its suite; its scope is checked by
nothing, and the gate was the last place that was still true of.

---

## F19 confirmed by mutation, and the guard it sits in is load-bearing

A vacuity objection was raised against the repaired in-flight test: with the
`/metrics` injection removed, does its green still mean anything? Two mutations in
a clean pinned worktree at `e160ac6f`, `--porcelain` 0, settle it in opposite
directions.

**Mutation 1 — the catalogue binding.** Bind `batch.effective_size` to the
in-flight gauge, which is precisely the hazard the test's title forbids:

```
metric: null                  -> 'onnx_genai_batch_size_current'
classification: 'NOT_PLUMBED' -> 'MEASURED'
git diff --numstat            -> 2  2        (the mutation landed)

✖ the in-flight gauge is NEVER exposed as the engine batch size
   59 pass / 1 fail
```

The guard fires, by name, on the exact defect. `batch.effective_size` is
`metric: null` / `NOT_PLUMBED` by construction, and the moment anyone backfills it
the test reddens. That is a regression guard on a *deliberate absence* — the
hardest kind to write and the easiest to mistake for vacuity.

**Mutation 2 — the fixture value.** Change `metricsBody`'s `inFlight` default from
3 to 999: **60 pass / 0 fail, entirely unchanged.** Nothing parses
`onnx_genai_batch_size_current`, so no value of it can discriminate anything.

F19 therefore stands and is strengthened. The proposal to *keep* the `inFlight`
parameter so the test "asserts something real" does not work: the parameter makes
nothing true. Zero callers, zero readers, zero discriminating power, and it
remains a discoverable door that silently injects into the void. Delete it. The
title is made true by `NOT_PLUMBED` in the catalogue, which survives the deletion
untouched.

### The reusable distinction

*A fixture that matches production has no discriminating power* is true only for
values something **reads**. For an unread key, **no** fixture value has
discriminating power, and choosing two different numbers buys nothing while
looking rigorous. Only mutating the fixture and watching tells the two cases
apart — and the answer here was that the discriminating power lives in the
catalogue binding, not in the fixture at all.

## Suite state

`cd examples/serving-dashboard && node --test`, node v25.6.1, clean detached
worktree at `e160ac6f`, `--porcelain` 0: **534 pass / 0 fail.**

Per F20 this is the wide scope. The narrow `dashboard/` scope is 289 and excludes
`telemetry-store.test.js` entirely — the file this section is about.

**This is the dashboard suite only.** F1 is a Rust blocker in `driver.rs` and no
node suite at any scope will ever surface it. A green board here is not a green
product.

## F21 (MAJOR, new) — one quantity, two namespaces, neither side knowing the other exists

Locate by symbol, not by line: `git grep -nF "'metrics.e2e_latency'" -- telemetry-provenance.js`.

```
telemetry-provenance.js   'metrics.e2e_latency'   classification: 'MEASURED'
                          metric: 'onnx_genai_e2e_request_latency_seconds'
  consumers in shipped JS ..................................... 0
  (the only hit is its own definition)

dashboard/throughput.js   asks for prefix 'latency.e2e_server'
  occurrences in the catalogue ................................ 0
```

Executed against three live origins (`:9451`, `:9452`, `:8133`): **17 series each, with real
observations** — `:8133` reports `_count 46`, `_sum 1461.96`, a mean of ~31.8 s. The metric is
plumbed, served, and carrying data. The panel renders an em-dash over it.

**A producer with no consumer and a consumer with no producer, describing the same quantity, in
one repository.** This is the only defect found in this review that fails in *our* favour — every
other one over-claimed; this one hides a real measurement. It was findable only because someone
asked the question in the other direction: we had searched the consumer's vocabulary and concluded
the producer did not exist.

### Do not "just rename the key"

`metrics.e2e_latency` is a **full-generation** latency — the catalogue says so itself: *observed in
`Drop for GenerationMetrics`, so it covers the full generation lifetime.* The row that would
consume it sits in a panel captioned `Throughput & latency` beside TTFT/ITL/TPOT, which are
per-token quantities in the millisecond range. Wiring the two together puts `31.78 s` next to
numbers ~1000× smaller, correctly formatted, under a caption that makes it false.

That is the same caption-versus-value defect as `Directory` and `Batch limit`, and it would be the
worst instance of the three: the others show a wrong *label*, which a reader can dispute. This one
shows a plausible *number*, which a reader cannot. The audience sees a slow server.

The correct fix is its own row with its own caption. The catalogue's existing label —
`'End-to-end latency (mean)'` — is already right.

## F22 (MINOR, new) — `fetchWithDeadline` promises pass-through and silently overwrites `signal`

`request-deadline.js` is the best code reviewed in this pass, and this is the one seam in it. Its
JSDoc describes the options object as passed through to `fetch`. It is — except for `signal`, which
the wrapper constructs from its own `AbortController` and writes over any caller-supplied value.

No current caller passes one, so this is latent, not live. It is filed because the failure mode is
silent and arrives at the worst moment: the first caller that wants *both* a deadline and its own
cancellation gets the deadline only, with no error and no warning, and the bug surfaces as a
request that would not cancel. Either compose the two signals or document that `signal` is owned by
the wrapper. Naming it costs a line; discovering it costs an afternoon.

## B1 — a regression caught after the review tag, and fixed inside eight minutes

Recorded because the *shape* is reusable, not because the bug is interesting.

`telemetry-store.test.js` was **60 pass / 0 fail at `review-0` (`6ecd9183`)** and **5 fail** eight
minutes later. The commit between them was a genuinely good refactor — it hoisted three hand-rolled
`{source, unit}` literals into one `catalogueMeta()` helper, fixing a real defect where every field
on a no-data frame dropped its `label` and rendered as the literal word `"value"`. For
never-measurable fields there is no second frame, so a documented zero announced itself as `"value"`
permanently.

The collateral was that the per-origin catalogue resolution changed which arm the
`pending` / never-measured ternary selected. The four surviving failures were not four bugs:

```
:639   the same zero means opposite things on the two servers
:926   a hit rate with zero lookups is undefined, not 0%
:949   a hit rate with real lookups and no hits IS a measured 0%
:1386  the same field IS pending on the first frame of the dynamic server
```

**One invariant, asserted from four directions by an author who knew how fragile it was** —
`:926`/`:949` a matched pair on undefined-zero versus measured-zero, `:639`/`:1386` a matched pair
on same-bytes-two-origins. That is the same defect the Rust server documents as *"a zero ratio and
'no batch has ever run' are the same bytes"*, and the same one the router re-creates on
deserialisation with bare `serde(default)`. **Three layers, three languages, three authors, one
solved problem re-created each time.** Nobody anywhere can spell *I have no value* without reaching
for a number that already means something.

Resolved at 64 pass / 0 fail — fixed *and* the suite grew 60 → 62 → 64. The fix added coverage
rather than deleting the assertions that caught it, which is the right direction and worth naming.

### The process finding, which outlives the bug

The regression author added **68 lines of new, passing tests to the very file they broke**. They
validated the behaviour they added and never ran the file they edited. The runner that catches this
(`run-tests.sh`) had already landed. It was not run.

`review-0` was green while the branch was red, and nothing re-scores a tag when the branch moves
past it. Reviewing from a tag solves drift and creates this. One line closes it: **run
`bash run-tests.sh` before landing and put its `PASS:` line in the report.**

## Correction: C2 is closed, and two reviewers' boards say otherwise

Measured by outcome, not by marker — a marker tied to one implementation cannot detect a different
and better implementation of the same outcome:

```
app.js               :18  import { fetchWithDeadline } from './request-deadline.js';
                    :189  await fetchWithDeadline(new URL('/health', location.origin), {…})
telemetry-store.js   :44 / :448   same helper, same import
bare fetch( in either file ................................... 0
control: request-deadline.js exists at HEAD ✅ · its own suite 7/7 ✅
```

`app.js:180` is not bare and has not been since `6ecd9183`. Any verdict still carrying it as a
blocker is scoring a fix that landed.

## Self-audit of this document, and the instrument that flattered it

Run at another reviewer's instruction: `grep -cF` six of this review's findings against this file.
The instrument returned 5 of 6. Opening every hit returned **3 confirmed, 1 false positive, 1
unverified, 1 genuinely absent**.

The false positive was `e2e_latency`, which appears in a list of catalogue keys — byte-identical to
the same string appearing in a finding. **A `grep -c` audit of a frame-blind corpus is itself
frame-blind, and it fails toward *you are fine*.** A narrow red gets argued down by the next reader;
a narrow green stops the next reader looking.

F21 and F22 above close the two real gaps that audit exposed.

## F1 — the prescribed test landed, and it is better than what was asked for

Confirmed at another reviewer's request, because a prescription is not satisfied until its author
says the delivered thing is the thing they meant.

I asked for a race-free unit test that calls the classifier directly rather than inferring
applicability through a live engine. Locate it by symbol:
`git grep -nF 'fn paged_kv_applicability_is_a_conjunction_of_two_independent_facts'`.

```rust
use crate::driver::{KvApplicabilityDecision, classify_kv_applicability};
…
classify_kv_applicability(paged, batching),
```

It exceeds the prescription in the way that matters. I asked for a direct call; what landed is a
**four-row total truth table**, and its doc comment states the property that makes it
mutation-proof:

> *Two of the four rows below distinguish that rule from the correct one, and each fails it in a
> different direction, so no single-row test could have caught both.*

That is the anti-vacuity argument written into the test, by its author, before anyone asked. A
second guard in the same file scans the source for `classify_kv_applicability(` and asserts the
capability is never re-derived from the absence of another — **a guard against the defect class,
not against the defect instance.** That is the distinction this review has been asking for all
session, and it was delivered without being requested.

**F1 is STRUCK in code.** What remained live was the citation trail, corrected below.

### The residual, which is mine and is about attribution rather than fabrication

The seven-arm `.is_ok()` collapse survives upstream of the classifier. It no longer fabricates a
measurement — the enum makes that unrepresentable — but it still attributes the absence to a single
mechanism when seven distinct failures reach the same arm. **Downgraded from "reports a false fact"
to "reports a true fact with the wrong cause."** Non-blocking, and worth naming precisely because a
good refactor relocated it: the bool used to be visible at the call site and is now one hop away
through a well-named total function. **A good refactor can move a defect somewhere that reads as
deliberate.**

## Correction: what "the Rust was never run" should have said

I asked for the brief to record that the Rust was never executed this session. **That is false and
I withdraw it.** The server suite was run by its author — 183 passing, ten consecutive identical
runs, three named mutations. The true statement is narrower and still worth printing:

> **No gate item measures the Rust, and none of the JavaScript tests can reach `driver.rs`.**

The difference matters. "Never run" impugns an author who did the work. "Not covered by the gate"
names a hole in *our* process. I made the broader claim because I had not looked, and a reviewer
asserting absence carries the same burden as a reviewer asserting presence.

## Correction: a third status table existed, and my strike did not reach it

Recorded because it is the clearest instance of a class this review filed against other people.

I struck F1 in two status tables and a section heading and reported the strike complete. **There was
a third table** — a dated historical snapshot — whose every row read `UNCHANGED — still live` in the
present tense. Its heading said *superseded*. That was true and it was not enough:

> **The qualifier lived in the heading; the claim lived in the cell. Every operation we perform on
> a document — a grep, a quote, a table extraction, a summary — strips the heading and keeps the
> cell.**

Three reviewers independently re-derived F1 as live from those cells while the Rust had been clean
for over an hour. **The defect was fixed in code and kept alive by my artefact.** It is now repaired
the only way that survives extraction: **each cell restates its own tense and names the commit that
struck it**, so no row can be quoted into a claim about the tree.

Ancillary, and the same lesson one level down: my own citation to the surviving doc comment named a
line number that had drifted by thirty lines. Both instances are now symbol-anchored. **A line
number is a citation that rots silently; a symbol either resolves or it does not.**

## Re-verification #3 — every row carries the SHA it was observed at

Ordered by the Project Lead, who asked the right question: *are these findings, or are they
photographs of an older tree?* **A review table is a findings list with no expiry column**, and this
one now has one. Measured in a detached worktree at `review-0` = `6ecd9183`, **porcelain 0**, by
execution — not by reading the previous row.

| Finding | Status | Observed at | Evidence |
| --- | --- | --- | --- |
| **F1** | 🟡 **LATENT — fix landing** | `6ecd9183` | Re-scoped, see below. Not a shipped fabrication. |
| **F3** | 🔴 **LIVE** | `6ecd9183` | All three stacks ship: `format.js`, `telemetry-field.js`, `dashboard/field-state.js`. The alias is real, not a substring match: `field-state.js` → `ok: RENDER_STATES.OK`. Control: `'measured'` appears 6× in the same file, so the search is not vacuous. |
| **F4-revised** | 🔴 **LIVE** | `6ecd9183` | `dashboard/prefix-cache.js` still declares `staleCeilingMs: null`, and `dashboard/field-state.js` documents that null is silently handed a 10-second expiry. Control: `system.js` declares an explicit `30000`, so the field is genuinely read. |
| **F11** | ⚪ **RETRACTED** (unchanged) | `6ecd9183` | Already retracted in this document. Re-measured anyway: 2 `role="group"` sites, 0 with a nearby `aria-label`. **I am not re-opening it on that number** — a JSX/template-built role is invisible to this grep, so the measurement is a floor, not a census. |
| **F12** | 🟢 **STRUCK** | `6ecd9183` | `repair-citations.test.js` now exists — 6,145 bytes, **6 tests, 6 pass**. It tests the exact property I filed: `it('REFUSES a cited file with uncommitted changes')` asserting `/DECLINED/`. The tool no longer computes from the dirty tree; it declines. |
| **F15** | 🟢 **CLOSED** (unchanged) | `6ecd9183` | Already closed in this document; no contrary evidence found. |
| **F16** | 🔻 **LARGELY CLOSED — my LIVE was wrong three ways** | re-measured at `1bca52a8` | **This row named the wrong file.** F16's body is about `check-perf-claims.test.js` (a *scope* defect); the row described `check-field-states.test.js` (a *dead-fallback* claim). They are different findings and I fused them. **Both halves then failed re-measurement.** ① Scope: 7 tracked `.md` files carry the withdrawn figure and the corpus now reaches **5 of them** — `demo-spec.md` and `design/` are the only exempt carriers, both as documented promises, and `check-perf-claims.test.js:240` now *detects exemptions that outlive their reason*. @fc8b5d97 retired one exemption with a stated 0-occurrence measurement. My "every instance is unreachable" is **stale**. ② Dead fallback: **mutation-disproven, see below.** |

### Two instrument notes from this pass, both of which nearly cost a wrong row

**A false LIVE, caught.** My first F3 measurement was `grep -c "ok"`, which matches `token`,
`broken` and `look`. It returned 8 and I nearly banked it. The corrected pattern found the real
alias and the row survives — **but it survives on evidence I did not have when I first wrote the
number.** A right answer from a wrong instrument is still a wrong instrument.

**A false STUB, caught by the runner.** I counted `repair-citations.test.js`'s tests with
`grep -c "^test("` and got **0** on a file with **6 passing tests** — it uses `describe`/`it`.
Had I trusted the zero I would have called a real, targeted test file a stub and left F12 open.
**The thing that saved it was executing the file instead of describing it**, which is the same
correction this review has issued to four other people tonight.

### F1, re-scoped on someone else's better evidence

The wire reports `page_size=16`, which can only originate from `kv_model.tensor_config`. **Both
demo origins are therefore genuinely paged, and the `!continuous_batch_supported` inference reaches
the right answer on the shipped configuration.** F1 was never a shipped fabrication and this
document should not have implied one.

It is **latent, not live**: the native backend hardcodes `kv_model: None`, so the predicate is one
configuration change away from lying, and the fix is landing anyway. That distinction is
load-bearing for anyone deciding whether the product misrepresented itself. **It did not.** What it
had was a correct output produced by reasoning that does not guarantee correctness — which is worth
fixing precisely because nothing would have told us when it started being wrong.

**Recorded because I got this wrong in the other direction:** I carried F1 as a live blocker on the
strength of a code shape without measuring the runtime values it produces. The shape was genuinely
bad. The claim I attached to it — that we were reporting a false capability to a visitor — was not
verified, and it was the more serious half.

---

## Suite observation #4 — a red I caught, a flake I nearly invented, and P1 closing

**Measured at `b04c6e8f` → `5e1a843d`, canonical runner, five confirming runs.**

I ran `run-tests.sh` for a closing number and got **red twice** (2 fail, then 1 fail), then green,
then **five consecutive greens at 646 tests / 98 suites / 0 fail / exit 0**.

**I was one sentence from publishing "the canonical gate instrument is nondeterministic."** That
would have been the most damaging false claim available tonight: it invalidates every green anyone
has reported, including the Lead's `608/0` and my own. **It was false, and the thing that caught it
was checking `HEAD` before and after rather than after only** — the tree moved `b04c6e8f` →
`5e1a843d` underneath the experiment. *Three runs on "the same tree" were three runs on three trees.*

**The reds were real, and both are explained by commits landing mid-experiment:**

| observation | cause | not |
|---|---|---|
| `no shell module reads a never-bind field off a response body` | `f025ae58` added `server.model_path` to `NEVER_BIND` while a render site still read it. **The guard fired exactly as designed, during the landing it was written for.** | not a flake, not a regression |
| `register-completeness.test.js` file-level ✖ | `5e1a843d` was rewriting that file. | not a misattribution by the runner |

**I also had a wrong theory and killed it before filing.** The failing test name lives in
`never-bind.test.js`, not in `register-completeness.test.js`, and I was ready to file the runner for
misattributing a failure to the wrong file. **There were simply two independent failures**, and one
of them was a whole file. *A single confusing output is not evidence of an instrument defect; check
whether it is two ordinary things before filing one extraordinary one.*

### P1 (`server.model_path`) is CLOSED, and the fix is better than the one that was ordered

The board spent hours on this as "two line deletions, zero dissent, no owner." Predicate run
unchanged at HEAD, with the published positive control:

```
server.model_path  (non-test, non-provenance)  ->  1 file
   telemetry-store.js:684  — a COMMENT recording the deletion. The epitaph, not the defect.
CONTROL server.model_id -> 3 files ✅ the instrument still reaches the tree
dashboard/system.js 'model directory' -> GONE      ui/model-card.js 'Directory' -> GONE
```

**And they did more than delete two rows: `projectServedModel()` is deleted outright.** Its only
consumer was that one row, and the comment states the reason for removing the *mechanism* rather
than the *row* — it "lifted the absolute path out of a list nobody addresses and pinned it at a
fixed, guessable location on the parsed body." **Deleting a render site removes what is painted;
deleting the projection removes what is reachable.** That is the difference between fixing the
instance and closing the class, and it is the right call.

---

## Tooling #1 — `git add <path> && git commit` does not scope a commit, and the boundary is narrower than reported

@c7a654ed proved a path-scoped `git add` does not scope the *commit*, after it captured 68 lines of
someone else's `run-tests.sh` under their message. **I used that exact pattern for all six of my
commits tonight.** Audited from committed bytes — `cf7c7717`, `0e8734ed`, `a84718fb`, `76596e2e`,
`b04c6e8f`, `13fb70c3` — **every one contains exactly one file, mine.** I was not careful; I was lucky,
and the difference is worth writing down because the crew is adopting a rule from this.

**Reproduced in a scratch repo. The hazard requires the other agent to have STAGED, not merely edited:**

| case | other agent's file | my command | result |
|---|---|---|---|
| 1 | **modified, unstaged** | `git add mine && git commit -m` | ✅ **mine only** |
| 2 | **staged** | `git add mine && git commit -m` | ⛔ **both captured** |
| 3 | **staged** | `git commit -m -- mine` | ✅ **mine only** |

**Case 1 is why my six are clean.** The 3–12 dirty paths present all session were working-tree
modifications, not index entries. **The rule "never `git add -A`" was never the protection anyone
thought it was — it is `git add` versus `git commit` that matters, not `-A` versus a pathspec.**

**And case 3 has a property nobody has stated, which is the reason to prefer it rather than merely
tolerate it: `theirs.txt` is STILL STAGED afterwards.** `git commit -- <path>` does not consume, drop
or disturb another agent's staged work — it commits your path and leaves their index entry intact for
them to commit themselves. *A safe form that destroys someone else's staging would just relocate the
damage.* This one does not.

## Tooling #2 — the citation writer's refusal, verified against my own file by hash

@e00032a4 disabled `migrate_citations.py --apply` at `26cef372` after measuring that it would perform
**93 silent rewrites, 76 of them in this document.** That is the largest blast radius aimed at my
deliverable tonight, so I verified the safety rather than accepting it:

```
md5 IMPLEMENTATION-REVIEW.md  before : 320c80826274a7bba38a43df041ddf4e
python3 scripts/migrate_citations.py --apply <this file>   -> EXIT 2
md5 IMPLEMENTATION-REVIEW.md  after  : 320c80826274a7bba38a43df041ddf4e   ✅ IDENTICAL
git status --porcelain <this file>                          -> empty       ✅
```

**The flag is not refused at runtime — it no longer exists**, which is strictly stronger, and the
refusal exits **2** with the reason stated: *"this is NOT a finding about the document."* **A refusal
that exits 1 is indistinguishable from a defect in the thing it declined to touch**, and that
distinction is the difference between a tool that stops and a tool that accuses.

**Why this mattered specifically to this document:** it declares its own `file:NNN` citations to be
*hints, not addresses*. The tool would have resolved those hints against today's files and emitted
confident, present-tense, symbol-anchored citations that nobody wrote — **converting an honest
disclaimer into 76 assertions and leaving my name on them.** The most dangerous input to a citation
repairer is a document that is deliberately imprecise and says so.

---

## 🛑 Review-anchor #1 — the `review-0` tag MOVED, and `/tmp/review-0` is now the banned vehicle

**Measured 04:26 from `/Users/justinc/Documents/GitHub/onnx-genai-demo`, HEAD `f7116dbe`.**
**Two independent failures of the review anchor, both landed in the last ten minutes.**

### ① The tag moved 60 commits

The Lead pinned the review with *"REVIEW SHA: tag `review-0` = `6ecd9183`"*, and reviewers have been
stamping findings against it all night.

```
git rev-parse review-0        ->  0aac6bb1        ⛔ NOT 6ecd9183
git cat-file -t 6ecd9183      ->  commit          (the old commit still exists)
merge-base --is-ancestor      ->  6ecd9183 IS an ancestor of review-0
git rev-list --count 6ecd9183..review-0  ->  60   ⛔ THE TAG MOVED 60 COMMITS FORWARD
for-each-ref                  ->  `commit`, i.e. a LIGHTWEIGHT tag — re-pointable with `tag -f`
```

**A lightweight tag is a mutable pointer.** Anyone who re-runs `git show review-0:<path>` now reads
bytes 60 commits newer than the ones every finding tonight was taken from. **The name did not change,
the meaning did, and nothing anywhere reports a conflict.**

**This is the strongest possible argument for the rule that was already ratified, so I am claiming no
credit for it — only evidence:** *cite a SHA, not a name.* My Re-verification #3 table survives this
intact **only because it spells `6ecd9183` in every row** rather than saying "at review-0". Had I
written the tag name, forty rows would have silently re-pointed. **A tag is a citation that someone
else can edit.**

### ② The directory is no longer a worktree — it is the vehicle the Lead banned by name

```
/private/tmp/review-0/.git   ->  ABSENT.  ⛔ NOT A WORKTREE. Created 04:16.
git -C /tmp/review-0 rev-parse HEAD  ->  fatal: not a git repository
git worktree list            ->  review-0 is NOT listed
```

**The Lead measured this exact hazard and banned it:** *"NOT `git archive` — it has no `.git`, and
every self-inspecting test dies with `fatal: not a git repository`, an exit code indistinguishable
from a real finding."* **It is back, and at least one reviewer has been citing measurements taken
inside it.**

**Demonstrated rather than asserted — one guard, same committed bytes, two locations:**

| location | result |
|---|---|
| `/private/tmp/review-0/examples/serving-dashboard` | **1 test, 0 pass** — `fatal: not a git repository` |
| `…/onnx-genai-demo/examples/serving-dashboard` (control) | **6 tests, 6 pass, 0 fail** |

**Note the count, not just the colour: 6 discovered collapses to 1.** The failure does not merely
redden a guard — **it shrinks the denominator**, which is the one number reviewers quote and the one
this crew has spent all night learning to publish first. A suite run in there reports a smaller,
confident total with no indication that five sixths of the checks never existed.

**Anything measured in `/tmp/review-0` after 04:16 needs re-taking in a real checkout.**

---

## F24 — the P1 path guard has a false positive and a false negative

**Status:** 🟡 open · not a blocker · the guard is correct for the shipped launcher
**Subject:** `telemetry-store.test.js`, test `the absolute model directory is not
addressable through the store at all`
**Measured at** `1384f7aa`, clean detached worktree, `porcelain 0`, node v25.6.1,
raw unpiped exit codes, 65 tests / 65 pass / 0 fail on the unmutated control.

This test replaced the two assertions I flagged as a landmine (`:1299` and
`:1352`, the second found by `@c0de4c2e` widening my census from 1 to 2). The
replacement is better than the repair either of us proposed and I want that on
the record before the criticism: it asserts on the **value**, not the key name,
so renaming the row is not a way to satisfy it; it puts a **positive control
first**, deliberately, because the test it replaces *went green while its subject
was being deleted*; and its comment says so in those words. That is the
vacuous-pass class this crew has chased all night, caught and documented by the
author against their own work.

**Mutation-proved, not read.** Injecting a path into a *differently named* field
(`server.execution_provider`) fails the test naming that field — so the
"whatever it is called" claim is real:

```
mutated   -> raw exit 1, fail 1/65, leaking == ['server.execution_provider']
unmutated -> raw exit 0, pass 65/65
```

**The defect is the predicate.** It bans any string value containing `/`:

| value | source | current | should be |
|---|---|---|---|
| `/Users/someone/.../models/qwen2.5-0.5b` | the threat | BAN ✅ | BAN |
| `Qwen/Qwen2.5-0.5B-Instruct` | `scripts/build_qwen.sh:25` | **BAN ⛔ false positive** | pass |
| `roneneldan/TinyStories-33M` | `scripts/bench_speculative.sh:6` | **BAN ⛔ false positive** | pass |
| `C:\Users\someone\models` | a Windows operator | **pass ⛔ false negative** | BAN |
| `qwen-scatter` | `run-demo.sh:236` | pass ✅ | pass |

`server.model_id` is `/health`.`model` = the registry's `default_id`, which is
the operator-supplied `--model-id`. **There is no character validation on it.**

<!-- cite: crates/onnx-genai-server/src/routes/admin.rs:8 = "model: state" -->
<!-- cite: crates/onnx-genai-server/src/cli.rs:37 = "pub model_id" -->
<!-- cite: scripts/build_qwen.sh:25 = "MODEL_ID" -->
<!-- cite: scripts/bench_speculative.sh:6 = "TARGET_MODEL_ID" -->
<!-- cite: examples/serving-dashboard/run-demo.sh:236 = "model-id" --> The guard is green
today only because the shipped launcher happens to pass slash-free ids; this
repository's own scripts default to slash-bearing ones. `--model-id
Qwen/Qwen2.5-0.5B-Instruct` is a legitimate invocation that turns the suite red
with a message accusing the operator of leaking their home directory.

I verified this the wrong way twice first, and the test caught me both times:
my first two attempts reported red, and **both reds were my own broken positive
control**, not the path guard. The slash-bearing id never reached a field until I
put it in `/health`. A red for the wrong reason is the failure I nearly published
an hour ago and it is the failure this test is built to prevent — it prevented it
on me.

**The false negative is the one that matters**, and it is the sharper half: a
Windows absolute path contains no forward slash at all, so the guard that closes
P1 does not fire on it. The test's own message promises "no field may carry a
filesystem path". On Windows that promise does not hold.

**Proposed predicate — tested in both directions, not proposed in the abstract.**
Ban values that are *absolute paths* rather than values that *contain a slash*:

```js
/^([A-Za-z]:)?[\\/]/.test(value)
```

This is strictly better on all five rows above: it keeps every true positive,
drops both false positives, and adds the Windows case. The threat is an
**absolute** path — a namespaced model id is relative, and that is the property
that separates them.

**Why this is worth fixing rather than tolerating:** a guard that reddens on a
legitimate configuration gets loosened by whoever hits it, and the loosening most
likely to be reached for is weakening the path ban itself. That is
`@bb2ee824`'s and `@e00032a4`'s law — *a safeguard that bans a legal layout gets
loosened within a day* — aimed at the single test now holding P1 closed.

**Not my file. Not landed by me.** The author is active and this is their
construction; I am reporting a tested patch, not editing their work.

---

## F25 — my prescription for `ARCHITECTURE.md` was stale, and my own document is the weaker of the two

**Status:** ⚪ retracted (mine) · finding redirected at this file

I told `@e00032a4` their nine rotted diagram coordinates needed the same
document-wide disclaimer I use here. **Both halves of that were wrong and I am
striking it.**

**Stale.** The fix predates the SHA I measured at. Verified with a control
proving the instrument can say no:

```
git merge-base --is-ancestor 1c6082d3 b54437df   -> exit 0   (fix is an ancestor)
git merge-base --is-ancestor HEAD 1c6082d3       -> exit 1   CONTROL
ARCHITECTURE.md at HEAD: positional coords 6, ALL 6 inside <!-- cite: --> anchors.
  Checked by IDENTITY, not by count: all six are lines 779-784. Zero rotted prose.
```

**And wrong for their file even if it had been fresh**, which is the part worth
keeping. Same predicate, both files, at HEAD:

| file | `<!-- cite: -->` anchored | positional |
|---|---|---|
| `docs/ARCHITECTURE.md` | 6 | 6 *(the same 6)* |
| `IMPLEMENTATION-REVIEW.md` (this file) | 0 | **108** |

Their formulation is the one to keep, and it is the mirror of `@086345a5`'s
credibility-transfer law: *a partial audit lends credibility to the rows it does
not cover; a blanket disclaimer strips credibility from the rows that earned it.*
Both are the same defect — **the scope of the statement not matching the scope of
the evidence.** A document-wide claim is safe only where the citations are
uniform. Mine are uniform. Theirs are not.

**One correction to their figure, offered as a missing denominator and not as a
dispute** — this is the fourth reconciliation of this shape tonight and by their
own diagnosis: they said a disclaimer would demote **180** symbol-anchored
machine-checked citations. I cannot reproduce 180 under any predicate I tried:
`<!-- cite: ` = 6, `= "..."` = 6, backticked paths = 24, any path mention = 229.
**The symbol-anchored, machine-checked population is 6.** Their argument survives
this completely — 6 anchored citations are still strictly stronger than hints and
a blanket disclaimer would still demote them — but the cost is 6, not 180.

**The finding I am keeping is against this file, and it is the uncomfortable
one.** My disclaimer — *every `file:NNN` here is a hint, not an address* — is
costless in my document because **I have 108 positional citations and zero
anchored ones**. That is not a virtue. It is a symptom. I spent the entire
session proving that line numbers rot, watched my own citations rot inward and
outward, and then **chose the disclaimer over the repair, 108 times.** Their
anchor form carries the symbol text, so it is *self-repairing*: when the file
moves, the citation can be re-resolved mechanically. Mine cannot be re-resolved
at all; they can only be re-taken by hand.

**Adopted, not just conceded.** F24's five citations are now anchored, each
verified to resolve at HEAD with a negative control proving the checker misses:

```
cli.rs:37 = "pub model_id"              ✅      admin.rs:8 = "model: state"      ✅
build_qwen.sh:25 = "MODEL_ID"           ✅      bench_speculative.sh:6 = "TAR…"  ✅
run-demo.sh:236 = "model-id"            ✅      CONTROL "ZZNOSUCHSYMBOL"         ⛔ misses
```

The remaining 103 are a named, owned, unfinished job — **not a hint I am content
with.** Anchored citations in this file: **0 → 5.**

---

## C2 — CLOSED. Scored against my own fixture, not against anyone's word.

`@f6527cc9` asked me to score my one remaining blocker against the discriminating
fixture I specified — *a socket that accepts and never answers, because
`connection refused` is green before and after.* I built it and ran it. Clean
detached worktree, `porcelain 0`, raw unpiped exit **0**:

```
BLACKHOLE (accepts, never answers)  REJECTED  RequestTimeoutError  @2019 ms  ✅
CONTROL   (normal server)           RESOLVED  status=200           @  13 ms  ✅
                                    ^ differs in exactly ONE respect: it replies
boot probe   await fetchWithDeadline(new URL('/health', …))   found by STRING
bare fetch( in non-test dashboard js   0     POSITIVE CONTROL any fetch   43
merge-base --is-ancestor 6ecd9183 HEAD -> 0   control -> 1
```

**C2 is CLOSED. My blocking set is empty and was already empty when I last said
so; this makes it measured rather than asserted.**

**And the boot-probe citation rotted while this was being written.** `@f6527cc9`
read it at `:189` roughly two minutes before I read it at `:191`. Three boards
carried `:180`, which is now the comment explaining the fix. **One line, four
addresses, inside one hour** — and every reading was honest at the moment it was
taken. This is the entire argument for anchoring citations to symbols, and it
happened to the citation for the finding about citations rotting.

## 🛑 Review-anchor #2 — `/private/tmp/review-0` contains no commit at all

**This supersedes my Review-anchor #1 and it is worse.** `@086345a5` reported the
directory as a real worktree at `6ecd9183`, `porcelain 0`. I could not reproduce
any of that. Measured `04:38`, positive control included:

```
git -C onnx-genai-demo        --is-inside-work-tree -> true    ✅ instrument CAN say yes
git -C onnx-genai-spec-capture --is-inside-work-tree -> true    ✅
git -C /private/tmp/review-0   --is-inside-work-tree -> fatal: not a git repository
git worktree list  ->  5 worktrees, review-0 IS NOT AMONG THEM
```

My earlier control was `/tmp`, which is also not a repository — **a control that
differs from its subject in zero respects, which is `@f6527cc9`'s confessed
defect and I repeated it.** The line above is the real one: two directories that
answer `true` prove the `fatal` is a finding, not an instrument failure.

**Then the part that matters more than the missing `.git`.** I checked which
commit the directory's *bytes* correspond to:

| file | extract | `6ecd9183` | HEAD | verdict |
|---|---|---|---|---|
| `model-card.js` | `e01b5a` | `a28f66` | `e01b5a` | **= HEAD** |
| `app.js` | `af489a` | `a59a74` | `af489a` | **= HEAD** |
| `system.js` | `3be06c` | `657863` | `3be06c` | **= HEAD** |
| `telemetry-store.js` | `8c4401` | `a21d79` | `a065b5` | **= NEITHER** |
| `telemetry-provenance.js` | `64b1c9` | `66c4c2` | `37fcf3` | **= NEITHER** |
| `telemetry-store.test.js` | `5f2c29` | `0599cc` | `362699` | **= NEITHER** |

**Zero of six match `6ecd9183` — the SHA it is named for and cited as. Three match
HEAD. Three match neither.** A directory whose files match three different states
is not a snapshot of any commit: **it is a copy of somebody's working tree taken
mid-flight, mixing committed and uncommitted bytes, wearing a SHA's name.**

That is strictly worse than the archive problem I reported at 04:16, and it is
worse in the specific way this session has been teaching all night: it is not
merely wrong, **it is wrong while displaying the exact label that would make a
careful reader trust it.** Every message banner reading `extract /private/tmp/review-0
HEAD 6ecd9183` asserts a pin that does not exist.

**What survives and what does not**, stated so nobody over-corrects: findings of
the form *this defect is present* read there are probably still sound, because
the bytes are at-or-ahead of `6ecd9183` and three instruments agreed on the P1.
**What does not survive is any `porcelain`, any `is-inside-work-tree`, any test
execution, and any claim of the form *verified at `6ecd9183`*.** Re-take those
against `git show 6ecd9183:<path>`, which needs no directory at all.

---

## 🔻 Review-anchor #2 — RETRACTED. I measured a worktree while it was being built.

**I published, forty minutes ago, that `/private/tmp/review-0` "contains no commit
at all" and was "a copy of somebody's working tree taken mid-flight, mixing
committed and uncommitted bytes, wearing a SHA's name." That conclusion is
wrong and I am withdrawing it.** Measured `04:41`:

```
git -C /private/tmp/review-0 rev-parse --is-inside-work-tree -> true
git -C /private/tmp/review-0 rev-parse --short HEAD          -> 0aac6bb1
git -C /private/tmp/review-0 status --porcelain              -> 0 lines
git worktree list  ->  review-0 IS REGISTERED, detached at 0aac6bb1
model-card.js       now e01b5a == 0aac6bb1 e01b5a   (6ecd9183 was a28f66)
telemetry-store.js  now 8c4401 == 0aac6bb1 8c4401   (6ecd9183 was a21d79)
```

**Every byte now matches `0aac6bb1` exactly.** The three files I scored as
matching "NEITHER" were partially materialised at the instant I sampled them.
There was no corruption. **There was a worktree being created, and I photographed
it halfway.**

`@086345a5`'s reading — `is-inside-work-tree=true`, `porcelain 0` — is correct
and reproducible. I said I could not reproduce one word of it. **I could not
reproduce it because I sampled a three-minute window during which it was being
rebuilt, and I reported the transient as a property of the artefact.**

**This is precisely the defect I have spent the session catching in other people**
— `@c7a654ed`'s suite counted mid-flight, `@12e42da8`'s dirty-tree read, my own
three-runs-on-three-trees — and it is the second time tonight I have taken a
reading of a moving object and published its blur as a shape. `@e00032a4` named
the general form in the same minute and it covers my case exactly: **a porcelain
line is the most perishable reading any of us takes, and we all append it to a
broadcast as a throwaway footer.** The same is true of a directory listing.

**What survives, and it is now stronger because git itself asserts it:** that
worktree is at **`0aac6bb1`, sixty commits ahead of `6ecd9183`** — confirmed by
`merge-base --is-ancestor` (exit 0) and `rev-list --count` (60). It is pinned to
the tag, and **the tag moved.** So the directory is real, clean, and correctly
built, and **it is still not what its name promises to anyone who wrote
`6ecd9183` in a banner.** The failure was never the directory. It was always the
mutable tag, and that half of Review-anchor #1 stands unmodified.

**The rule I am taking out of this, stated against myself:** I twice reported a
`/tmp` directory's state without asking whether anything was *writing* to it.
I would never quote a suite count without a SHA, and I quoted a filesystem
without a clock. **A directory is a mutable object shared by fourteen agents,
exactly like the tree — and it deserves the same two-sided read: measure, act,
measure again, and if the two disagree, the object moved and neither reading is
a property.**

**One clearance owed:** `@e00032a4` refers to a checkout `wt-73`. **No such
worktree exists** — `git worktree list` shows nine, none carrying my id, and no
`/tmp` path matches `*73*` beyond IPC sockets that are not mine. My standing
certification holds: **I hold zero worktrees and have reaped every one I created.**
