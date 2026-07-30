# Readability Review — serving-dashboard

Reviewer: readability lane (naming, organisation, documentation freshness, simplicity,
consistency, co-location). Implementation correctness and test quality belong to the Code
Reviewer; architecture and security belong to the Critical Reviewer. This document does not
duplicate their findings.

**Provenance.** Every finding below was verified by execution or by reading the file, in the
worktree `/Users/justinc/Documents/GitHub/onnx-genai-demo`, on branch `feat/genai-demo-dashboard`,
at `346763a0`, 02:11. Five checkouts of this repository exist on this machine and the demo exists
in exactly one of them, so a bare path is not a citation. Findings name a **symbol** and **quote
the text**; line numbers are a hint and may have rotted by the time you read this.

---

## The root cause behind the longest argument of the session

The five field states are spelled by hand in **15 non-test files** and derived from the enum in
exactly **one shipping module** (`format.js`). Six test files derive correctly — which is why the
suite agreed with whoever ran it, at every commit, before *and* after a rename: derived tests are
green on both sides of the change, so the one mechanism that could have flagged a divergence was
structurally incapable of seeing it.

That is why a one-word vocabulary change required a sixteen-file atomic commit, and why the
failure mode is silent: a missed CSS site does not throw, it renders unstyled.

> A constant is only a single source of truth for the files that import it. The enum was never
> the source of truth — it was the sixteenth copy, and it happened to be the one we argued about.

**The guard already exists and is better than what I first specified.** `state-treatments.test.js`
derives in *both* directions and excludes the four connection states (`connected`, `connecting`,
`no-model`, `unreachable`) explicitly, because they share the `data-state` attribute and belong to
a different vocabulary. Its failure message is the best prose in the repository:

> *"a rule that never fires is worse than a missing one, because it reads as covered."*

Run it before committing any vocabulary change and the fatal half-landed edit cannot reach the
tree.

---

## Live findings

### R1 — `mount` typedef in `dashboard/index.js` promises a method that does not exist

```js
 * @property {(root, store) => {destroy: () => void}} mount     // the contract
 * @returns  {{unmount: () => void, ...}}                       // the reality
```

The typedef says a panel's `mount` returns `{destroy}`. The mounting code stores
`handle: {unmount}` and calls `unmount()`. A panel author autocompleting from the type writes
`return { destroy }`, `unmount()` silently no-ops, and panels leak their subscriptions on every
mode switch — with no error, ever. `CONTRACT.md` was corrected; this typedef was not carried with
it. One-word fix.

*Signature-level lesson: what would a new team member assume from the type alone? Here the type
forbids the only method the runtime calls.*

### R6 — `SERVER_CLASSES.DYNAMIC` docstring asserts a feature that was cut

```js
/** Dynamic model. Paged KV and the prefix cache engage; batching does not. */
DYNAMIC: 'dynamic',
```

This is a **claim position**, not an explanation position, on the enum member that defines what
the dynamic server *is*. It is also the root cause of the register defect found in the browser:
the three prefix counters are still classified `MEASURED` on that origin, the classification
agrees with this docstring, and both disagree with the measurement. Fix the classification alone
and the next reader re-derives `MEASURED` from the prose, correctly, forever. **Fix both in one
commit.**

### R9 — `ui/model-card.js` writes `data-state` raw, bypassing the normaliser

```js
element.dataset.state = field.state;   // renderCardField — not normalised
```

Every other `data-state` write in shipping JS routes through `renderStateOf` or writes a frozen
enum member. This is the single seam where the text channel and the style channel derive from
different sources and can disagree. Latent today (the root store emits only `FIELD_STATES.*`), and
untested — no test asserts model-card's `data-state` at all. Independently found by the Code
Reviewer, who owns the severity call; listed here because it is the one place the two-vocabulary
split is physically visible.

### R10 — a withdrawn measurement propagated faster than its retraction

The `7.0% slower` prefix result was withdrawn by its author (the re-run came back with the
opposite sign, on a machine where a byte-identical binary swung 9.8% from ambient load). The
retraction reached the discussion four times before it reached the tree once.

**This finding deliberately carries no count.** An earlier draft of it said "ten sites"; a sweep
landed minutes later and the number was false while the defect it described was still real.
Carry the predicate instead:

```
grep -riE '7\.0%|7% slower|1341|1254|prefix reuse absent' $(git ls-files)
```

Two flags that cost a measurement when omitted: `-i`, because the live text is `7% SLOWER` in
caps, and the raw values `1341`/`1254`, because some sites quote the measurement without quoting
the percentage. A search missing either returns a clean, confident, wrong zero.

**At the time of writing the shipping test files have been swept, and the surviving carrier in
shipping JS is `CUT_SCENARIOS` in `scenario-origins.js`.** Its fix is surgical rather than a
substitution, because the paragraph holds both a withdrawn claim and the argument's real
evidence:

```
"...measured and found ABSENT on both execution paths (QA, n=6 per arm with a
 sensitivity control: shared-prefix requests ran 7% SLOWER than a zero-sharing   <- WITHDRAWN
 control, while the engine's hit counter fired on every request including all
 six controls)."                                                                 <- LOAD-BEARING
```

Delete the timing clause; keep the counter clause. Counter arithmetic cannot be overturned by a
re-run on a quieter machine; a stopwatch can. **The argument gets stronger without it.**

**Two sites must not be swept.** `check-readme-claims.test.js` asserts the *disagreement* — "one
controlled run put the shared arm 7% SLOWER, another put it 17% faster" — so its claim was never
the effect size; it is still true and it is the template the others should become.
`check-perf-claims.test.js` is the guard that detects this entire class, and a guard necessarily
quotes what it forbids. A blind substitution breaks the only two sites that were already right.

*This document is itself a carrier, three times, and that is the correct state — a retraction has
to name what it retracts. The distinction that matters is quoted-as-history versus
asserted-as-finding, which is the same distinction R11 turns on.*

### R11 — the design record describes the enum in the present tense, and it is the regeneration source

`design/demo-ux.md` §53 (D159/D160) still reads, as statements about the code *now*:

```
"FIELD_STATES currently exports OK: 'ok' AND MEASURED: 'ok'"
"Twelve modules and every [data-state='ok'] selector already agree"
```

Both were true when written. Neither is true at HEAD, and one is actively forbidden:

```
telemetry-field.js       MEASURED: 'measured'            <- the alias is gone
[data-state='ok']        ZERO occurrences in styles/     <- control: 9 data-state
                                                            selectors found, incl.
                                                            'measured'
state-channel.test.js    assert !shellCss.includes("[data-state='ok']")
                         "A selector matching nothing is the half-migration this
                          assertion exists to make impossible."
```

**This is the same defect as R6, one level up, and it is why the enum question kept
reopening for two hours.** A reader consulting the design record to decide the question
correctly derives that `'ok'` is the current value and that the cost ledger favours keeping
it. The ledger actually runs the other way — the selectors agree on `measured`, zero say
`ok`, so `'ok'` is the direction requiring CSS, writer and test edits. The document is not
wrong about the *decision* it recorded; it is wrong about the *tense*.

**The fix is not to rewrite the ruling — it is to date it.** And the repository already
contains the correct pattern, three times, for this exact fact:

| where the same fact appears | how it is marked |
|---|---|
| `telemetry-field.js` | "disagreed **once**" — past |
| `dashboard/state-vocabulary.test.js` | "has **flipped between** … more than once tonight" — past |
| `state-channel.test.js` | an executable guard forbidding the old value |
| `design/demo-ux.md` §53 | "**currently** exports", "**already** agree" — **present** |

Three code sites narrate it as history. The one prose site asserts it as current — and the
prose site is the one people read before deciding.

> A decision record written in the present tense stops being a record and becomes a claim.
> Records need dates; claims need checkers. §53 has neither.

**The same pattern, done right, landed while this review was being written** — worth naming
because it is the counter-example rather than another complaint. `dashboard/scheduling.js`
rebound away from a key with no producers, and its comment explains the *dead* binding in the
past tense: *"`scheduler.max_batch` had ZERO producers … the suite was green because the tests
handed the panel the value the server could not."* `scheduler.max_batch` now has **zero** code
readers; both remaining occurrences are prose describing its own removal. That is exactly the
tense discipline §53 is missing, from the same night and the same crew.

*One consequence worth flagging for anyone auditing that fix:* `grep scheduler.max_batch` still
returns hits, in the file most likely to be cited, and every hit is a comment about the repair.
A reader who greps for the symptom finds the symptom and concludes it is unfixed. **The honest
past-tense comment and the live defect are indistinguishable to a text search** — so verify this
class by asking whether anything *reads* the key, not whether anything *mentions* it.

**Related, smaller, same root:** 11 deictic references — *"see above"*, *"see below"*, *"the
line above"* — across 7 tracked documents. A deictic is a citation with no resolvable target:
`file:line` can be checked by a script, "above" cannot, and it silently breaks whenever
anything is inserted between pointer and referent. Notably **zero** appear in 6,390 lines of
JS doc comments — code comments sit beside their referent and never need to point. The
unowned prose surface is *cleaner* on this axis than every owned document, including this one,
which shipped with one.

### R12 — the provenance catalogue defines one field twice, and the weaker entry is the live one

`PROVENANCE` in `telemetry-provenance.js` contains **two** `'batch.capacity'` entries, out of 37
keys. A duplicate key in a JavaScript object literal is not an error: no syntax error, no
warning, no lint. The last definition silently wins.

Confirmed by executing the module rather than reading it:

```
import('./telemetry-provenance.js') -> PROVENANCE['batch.capacity']
  LIVE evidence: 'admin.rs:178 (batch_capacity, from state.config.effective_batch_capacity()).'
  LIVE label   : 'Batch limit'
```

**The entry that loses is the better one, on the axis this branch has spent all night
ratifying.** It is anchored to a *symbol* and it carries the semantics:

```
DEAD  '...admin.rs — `batch_capacity` is serialised from AppConfig::effective_batch_capacity(),
       which state.rs defines as max_batch.min(max_queue_depth). Genuinely computed from
       configuration; no stub.'
LIVE  '...admin.rs:178 (batch_capacity, from state.config.effective_batch_capacity()).'
```

The `min(max_batch, max_queue_depth)` explanation is the one the crew explicitly decided must not
be un-learned — raw `max_batch` overstates the ceiling, so a saturated server can draw as 25%
busy. That explanation is in the dead half. The surviving half is anchored to a line number, the
least durable thing this file could cite.

**So the fix is a merge, not a deletion, and getting that backwards loses something either way:**
the dead entry uniquely holds the evidence prose and the denominator comment; the live entry
uniquely holds `label: 'Batch limit'`. Keep the symbol-anchored evidence, keep the label, delete
one key.

> Two definitions of one fact do not disagree loudly — they resolve, silently, and the file
> still reads as if both are in force. A reader who scrolls to the first entry, finds an
> exemplary symbol-anchored explanation, and stops reading has read dead code that is
> indistinguishable from live code.

This is the defect the product exists to refuse — an absence that renders identically to a value
— sitting inside the provenance table itself. **A cheap guard closes the class:** extract the key
literals from the source and assert the list equals its own `Set`. Today that check goes red:

```
git show HEAD:telemetry-provenance.js | grep -oE "^  '[a-z_]+\.[a-z_]+':" | sort | uniq -d
  ->  'batch.capacity':
```

*Do not write that guard by comparing a source count against `Object.keys(PROVENANCE).length`.*
I tried it, and the two numbers are drawn from different populations — the pattern above matches
a subset of the object's keys, so the counts agree while a duplicate is present and the check
proves nothing. Compare a list against its own deduplication, in one population.

---

## Withdrawn by me

Recording these because a review that only accumulates findings is not measuring itself.

| | status |
|---|---|
| `panel.title` missing from the registry | **Withdrawn** — `PANELS` lifts `title` out of `meta`, with a comment documenting that exact bug as already fixed. |
| `[data-state='not-applicable']` takes the wrong token | **Withdrawn** — corroborated from token values and a contrast formula; refuted by measuring the rendered pixels. The pixel outranks the declaration. |
| "build a derived enum→CSS check" | **Withdrawn** — it already existed, and my own command output had listed it. I read my summary instead of my list. |

---

## Patterns worth carrying past this branch

**Prose inside a code file is the worst of both surfaces.** It inherits code's authority, travels
in code review, is trusted like code, and is executed by nothing. Five instances were found here
by four reviewers, and every one was a *type or vocabulary claim* — a docstring saying what a
value means. That is the one category of prose a machine could check, and nothing checks it. The
prose surfaces we did assign owners to were all *documents*; `.js` doc comments are the surface
where the prose sits inches from the code it contradicts.

**Form-checking cannot detect endorsement decay.** An audit that asks whether a number is properly
sourced — measurement, method, control arm, explanation position — awards full marks to a
withdrawn measurement in immaculate form. The better-written the false statement, the higher it
scores. *Well-cited is not still-true.*

**Propagation debt is proportional to credibility.** The best measurement gets copied to the most
places, so it is the most expensive one to retract. Nothing here connects a withdrawal to its
copies.

**Read your own output before you summarise it.** A count is a claim about a list; the list is the
evidence. Twice in this review I held correct, current output and reasoned from the aggregate I
had just built from it.

**Every zero needs a positive control.** One finding here initially measured zero from a directory
that does not exist. A grep over a missing path and a clean verified absence are byte-identical.
