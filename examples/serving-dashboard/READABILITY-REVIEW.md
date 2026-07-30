# Readability Review — serving-dashboard

Reviewer: readability lane (naming, organisation, documentation freshness, simplicity,
consistency, co-location). Implementation correctness and test quality belong to the Code
Reviewer; architecture and security belong to the Critical Reviewer. This document does not
duplicate their findings.

**Provenance.** Every finding below was verified by execution or by reading the file, in the
worktree `/Users/justinc/Documents/GitHub/onnx-genai-demo`, on branch `feat/genai-demo-dashboard`,
first at `346763a0`, 02:11; **every status line below was re-verified at `484cda07`, 04:06.**
Five checkouts of this repository exist on this machine and the demo exists
in exactly one of them, so a bare path is not a citation. Findings name a **symbol** and **quote
the text**; line numbers are a hint and may have rotted by the time you read this.

> ⚠️ **A review is a measurement, not a document, and it decays at the rate the tree moves.**
> This file spent roughly ninety minutes asserting five findings in the present tense after
> the crew had fixed all five. That is worse than the findings were: a stale *open* finding
> sends someone to repair something already repaired, or convinces them the repair never
> landed. Every heading below now carries its status and the commit that closed it. **If you
> are reading this at a SHA later than `484cda07`, treat every 🔴 as unverified — the 🟢 rows
> name a commit and can be checked; the red ones only name a moment.**

**Reading the status column.** 🔴 LIVE = re-measured open at `484cda07`. 🟢 FIXED = re-measured
closed at `484cda07`, with the closing commit named. Each status was taken with a **positive
control** (an expression that must return non-zero if the instrument reaches the file) and a
**negative control** (a string that must return zero), because a search that matches nothing and
a tree with no defects are byte-identical from here.

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

### R1 — 🔴 LIVE at `484cda07` — `mount` typedef in `dashboard/index.js` promises a method that does not exist

> **Re-verified:** `destroy: () => void` → **2**; control `unmount` → **7**. The typedef promises
> `destroy`, the code returns `unmount`. **One word, and the signature is the only thing a caller
> reads.** Cheapest finding on this board and it has survived every sweep tonight.

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

### R6 — 🟢 FIXED — `SERVER_CLASSES.DYNAMIC` docstring asserted a feature that was cut

> **Status re-verified at `484cda07`, and this row nearly went the wrong way.** A keyword search
> for the offending phrase still returns **1** hit — and the hit is the *fix*:
> `"Deliberately does NOT say the prefix cache engages here."` The replacement comment names the
> old error, negates it, and explains why it was dangerous. **A grep cannot see a negation, so it
> scored an exemplary fix identically to the defect.** The law this produced is the most
> load-bearing thing in this review: **the better the fix, the more likely it trips the keyword
> guard** — a lazy fix deletes the sentence and greps clean; a good fix quotes the bug it killed.
> Every keyword instrument on this branch systematically penalises the documentation we say we
> want. The control that saved it: `"deliberately does not"` → 1.

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

### R9 — 🔴 LIVE at `484cda07` — `ui/model-card.js` writes `data-state` raw, bypassing the normaliser

> **Re-verified:** `dataset.state = ` → **1**; control `dataset` → **4**. The normaliser exists
> and this one site goes around it, so the invariant is discipline rather than construction.

```js
element.dataset.state = field.state;   // renderCardField — not normalised
```

Every other `data-state` write in shipping JS routes through `renderStateOf` or writes a frozen
enum member. This is the single seam where the text channel and the style channel derive from
different sources and can disagree. Latent today (the root store emits only `FIELD_STATES.*`), and
untested — no test asserts model-card's `data-state` at all. Independently found by the Code
Reviewer, who owns the severity call; listed here because it is the one place the two-vocabulary
split is physically visible.

### R10 — 🔴 LIVE at `484cda07`, and the count has grown from 5 files to 9 — a withdrawn measurement propagated faster than its retraction

> **Re-verified from the repository root with no pathspec: `7.0%` appears in 9 tracked files.**
> An earlier pass reported 5. **The denominator moved and the conclusion did not**, which is the
> strongest form this finding can take.
>
> ⚠️ **The first re-measurement of this row returned `0`, and it was wrong.** The pathspec
> `-- 'examples/serving-dashboard'` was supplied from *inside* `examples/serving-dashboard`, so
> git resolved it relative to the current directory, matched nothing, and exited **0** with no
> output. **A confident, clean, false zero, deterministically reproducible.** The only reason it
> was caught is that a zero on a figure known to be widespread is an asymmetry that should not
> exist, and the rule is now: **when a negative surprises you, widen the corpus before you
> believe it.** Dropping the pathspec took nine seconds and changed 0 into 9. *Note that this
> document is itself one of the nine, and that is the correct state — a retraction has to name
> what it retracts.*

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

### R11 — 🟢 FIXED by @0837fdf9 (D278) — the design record described the enum in the present tense

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

### R12 — 🟢 FIXED at `185d6720` — the provenance catalogue defined one field twice, and the weaker entry was the live one

> **⚠️ Correction, and it runs against this reviewer.** This finding was **true**, and for most of
> the session it was recorded as *withdrawn as my own error*. It was not an error. The duplicate
> genuinely existed: `'batch.capacity'` appears **twice** at `13ba68a7` and `13d9214b`, and
> **once** from `185d6720` — *"make the provenance register agree with the page it certifies"*
> — onward. Verified at `484cda07`: 1 occurrence, and **0** duplicate keys anywhere in the
> catalogue.
>
> **How the false retraction happened, because the mechanism matters more than the row.** I
> re-measured after the fix landed, found one key where I had reported two, and concluded *I
> was wrong* instead of *it was fixed*. That is precisely the rule this crew ratified tonight in
> someone else's words — **a clean zero has two explanations: my instrument was narrow, or I
> measured after the fix, and they are indistinguishable from the result.** I applied that rule
> to other people's findings all night and never once applied it to my own withdrawal.
>
> **And the direction is the dangerous part.** Measure a defect gone and call it *fixed*, and
> you credit the person who fixed it. Measure it gone and call it *my mistake*, and you delete
> the evidence that the fix was ever needed — while collecting credit for candour. This session
> built enormous social reward for retracting your own findings. **I harvested that reward for
> retracting a finding that was right.** A retraction is a claim like any other and needs the
> same control: before withdrawing a finding, check whether the tree changed under it.
> `git log -S'<the thing>' -- <file>` answers it in one command and I did not run it for hours.

`PROVENANCE` in `telemetry-provenance.js` contained **two** `'batch.capacity'` entries, out of 37
keys. A duplicate key in a JavaScript object literal is not an error: no syntax error, no
warning, no lint. The last definition silently won.

Confirmed at the time by executing the module rather than reading it:

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

**What actually shipped, verified at `484cda07` — the fix was the merge, and it was better than the
one recommended here.** The surviving entry keeps the symbol-anchored evidence
(`routes/admin.rs — \`batch_capacity\` is serialised from AppConfig::effective_batch_capacity()`)
and the denominator comment, and it **renames the label from `'Batch limit'` to
`'Effective batch capacity'`** — which closes a *separate* caption finding filed independently by
another reviewer, because `batch limit` is the name of raw `max_batch`, not of the clamped
minimum actually served. The bare `admin.rs:178` string that this section quotes twice above no
longer exists anywhere in the catalogue (`grep -c 'admin.rs:178'` → **0**; control `admin.rs`
→ 28). *Those two quotations are left unedited on purpose: they are quotations of a historical
state, and rewriting a number inside a quotation to satisfy a citation checker falsifies the
quote.*

**And the coordinate never resolved, which is measured, not assumed — `crates/onnx-genai-server/src/routes/admin.rs:178`
is `"paused_sessions",`.** The real `batch_capacity` sites in that file are `:122`, `:231`, `:232`
and `:240`; the file is 714 lines, so `:178` is **in range and simply wrong**. That is the
strongest possible argument against "repairing" it: re-anchoring the number would invent an
address for a string the source no longer contains, and would convert an honest quotation of a
dead entry into a false claim about a live file. **A citation that cannot be repaired truthfully
must be captioned, not corrected.**

> **This reviewer's own scorecard, stated because it is unflattering and because the instrument
> rewards the wrong thing.** This document has the *fewest* `file:NNN` citations of the three
> reviewer deliverables — 2, against 90 and 43. That was published as evidence of cleanliness.
> It is not. **Both of this document's citations are wrong, so its citation hit rate is 0 of 2.**
> The census that scored it well counts citations; it has never validated one — neither bounds
> nor content. *A document with almost no coordinates cannot rot and cannot be checked either.
> Low citation density is a different trade, not rigour, and here it bought the worst hit rate
> on the branch behind the best-looking number.*

**And the guard recommended below was built, with a control I did not think to ask for.**
`provenance-expiry.test.js` extracts the key literals and asserts the list equals its own `Set`
— and it carries an **anti-vacuity** assertion, because a regex that matched nothing would
report zero duplicates and pass. Mutation-proven this session: raw exit **0** clean, raw exit
**1** with a duplicate injected, file restored, 0 dirty paths.


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

### R13 — 🟢 FIXED — the launcher advertised, as a clickable URL, the one scenario the code says must not be addressable

> **Re-verified at `484cda07`:** the `run-demo.sh` banner now emits exactly two scenario links —
> `scenario=continuous-batching` and `scenario=paged-kv`. Prefix-cache links: **0**.

This is the first prose an operator ever reads — before the page loads, before any panel renders,
outside every honesty mechanism this project built. `run-demo.sh`'s success banner prints three
headline scenario URLs:

```
  continuous batching   ${SCATTER_ORIGIN}/demo/?${TOPOLOGY}&scenario=continuous-batching
  paged KV block table  ${DYNAMIC_ORIGIN}/demo/?${TOPOLOGY}&scenario=paged-kv
  prefix caching        ${DYNAMIC_ORIGIN}/demo/?${TOPOLOGY}&scenario=prefix-cache
```

`CUT_SCENARIOS` in `scenario-origins.js` says the opposite, and explains itself:

> *"Keyed only, with no `id:` field, and that is deliberate: an `id` is what makes a scenario
> addressable, **and this one must not be addressable.**"*

The code went to deliberate lengths to make that route unaddressable — omitting the `id` field
specifically so the scenario could not be linked — and the launcher links to it anyway. The URL
returns **200**, identically to the working control, so nothing announces the discrepancy.

Two more claims in the same file assert the same cut capability:

```
"The paged KV allocator and the prefix cache live on the dynamic path, so this one drives those scenarios."
printf 'starting the dynamic server on :%s (paged KV, prefix caching)\n'
```

The second is the third independent copy of R6's false belief — the docstring, the classification,
and now the launcher, none of which can see the other two.

**Why this surface survived when four sibling claims were fixed:** `run-demo.sh` is *heavily*
tested. Six suites read it — `check-launcher`, `check-cli-flags`, `check-endpoint-registration`,
`check-launch-command`, `scenario-origins`, `scenario-switcher`. Every one of them tests
**mechanism**: that it launches the right servers, passes `--demo-assets-dir`, uses an absolute
path. None tests **prose**. And the guard that *does* check claims scans exactly three paths:

```
page-claims.test.js -> './index.html'  './scenario-origins.js'  './design/demo-ux.md'
```

`run-demo.sh` is not among them.

> A file can be the most-tested artifact in the repository and still have an entirely unexamined
> surface, because coverage is counted per *file* and claims live per *sentence*. The launcher
> passed six suites while advertising a feature the code it launches declares cut.

The fix is one line of deletion and one line of coverage: drop the third URL and the two prose
claims, and add `run-demo.sh` to the claim guard's path list so the operator-facing surface is
scanned by the same rule as the visitor-facing one.

### R14 — 🟢 FIXED at the type level — the silent substitution was not a policy choice; the signature made honesty unrepresentable

> **Re-verified at `484cda07`.** This finding's complaint was that the return type could not
> express *what was asked for* beside *what was shown*, so a caller had no way to be honest even
> if it wanted to be. The fix is in the **signature**, which is the right place: the documented
> return now carries `requested` and `substitution` alongside the resolved id (`requested` → 12
> occurrences in `scenario-origins.js`). The whole chain was walked rather than spot-checked —
> exported, imported by `ui/scenario-switcher.js`, called, and plumbed into `app.js`. **A fix
> verified only at its definition is a fix verified nowhere; the interesting failure is always
> the caller that never adopted it.**

@376a0297 executed the shipping resolver on the launcher's own URL and found `prefix-cache`
resolves to `paged-kv` with no notice. They asked whether an unrecognised `scenario=` should fall
back silently at all, and routed the call here. **It should not — but the reason is structural, not
a preference, and that is the finding.**

```
export function currentScenarioId(href, selfClasses = []) {
  if (requested && Object.hasOwn(SCENARIOS, requested)) return requested;
  ...falls through to the local default
}
```

Two defects, both visible in the signature alone:

**1. The name says accessor; the body is a decision.** `current…Id` reads as a read of existing
state — the scenario we are on. It is really an adjudicator: it receives a *request* and may
overrule it. The give-away is the vocabulary the file needs to explain itself — `requested`
vs `current` — two adjectives on one noun, which is this review's recurring signal that the noun is
doing two jobs. `requestedScenarioId` and `resolveScenarioId(href)` are two different functions and
we have one name covering both.

**2. The return type is `string`, so "I substituted" has nowhere to live.** This is the load-bearing
half. Downstream, `ui/scenario-switcher.js` accepts `currentScenarioId` as a plain string
parameter and renders it as the active tab. By the time the value reaches the UI, *the fact that a
substitution occurred no longer exists in the program.* Not hidden — absent. No caller could
disclose it however much it wanted to.

> The Lead's rule is: if the signature implies something the design forbids, the signature is
> misleading. This is the sharper inverse — **the signature forbids something the product
> requires.** Our entire honesty layer is built on `{value, state, reason}`: never a bare number,
> always the number plus what we know about it. This function returns a bare id. It is the one
> place in the codebase where we went back to returning the value alone, and it is the place that
> decides what the visitor is looking at before a single field is fetched.

The fix follows the pattern the rest of the codebase already uses, and needs no new concept:
return `{id, requested, substituted}` — the same shape as every telemetry field. The switcher then
*can* say `prefix-cache is not a scenario on this build — showing paged-kv`, because the
information finally exists at the point of rendering. Rename to `resolveScenarioId` so the name
admits it decides.

That is also why R13 and R14 must be fixed together: deleting the launcher URL removes today's
one known bad link, but any typo, stale bookmark, or README edit re-creates a silent substitution
tomorrow. R13 is the instance; R14 is the reason instances are silent.

### R15 — 🟢 FIXED, and inverted — the QA plan instructed the tester to perform the silent substitution and record it as a pass

> **Re-verified at `484cda07`.** The plan no longer merely omits the hazard — it **warns about
> the false pass it once invited**. That is the strongest shape a documentation fix can take:
> the document that caused the error is the document that now prevents it, so the correction is
> co-located with the instruction that needed it rather than filed in a separate errata nobody
> opens.
>
> *A caveat on this reviewer's own instrument, disclosed because it changes how the row should be
> read:* an earlier keyword pass over this plan returned **0** and was wrong — the search was
> case-sensitive and the word is sentence-initial (`Substitution`). The zero was refused rather
> than banked, and only because a zero on a document known to discuss the topic is the asymmetry
> that should not exist. **Case is the sixth thing a grep cannot see, after arrays, tense, line
> breaks inside an expression, negation, and string-concatenation boundaries.**

`QA-PLAN.md`, item **B1**, already ticked `[x]` and marked **RESOLVED**:

> *"**Test deep-linking directly**: paste `?scenario=prefix-cache` into a fresh tab and confirm it
> opens on the right panel against the right origin."*

The tester will paste it. They will get a page that loads, renders completely, and is honest in
every field — the paged-KV scenario, with correct provenance badges and correct states. It looks
exactly like a pass, so they will tick it.

**This instruction is incapable of failing.** It asks for confirmation of "the right panel"
without saying which panel is right, against a route the code deliberately refuses to make
addressable (R13) via a resolver that cannot report having substituted (R14).

The distinction from R13 matters, and it is why this is filed separately rather than as another
site:

| | who reads it | what it costs |
|---|---|---|
| R13 | the operator | one misleading link |
| R15 | **the verifier** | a signed-off pass certifying the defect as correct behaviour |

R13 misleads someone. R15 recruits our own quality process into ratifying the thing the product
exists to refuse. It is the one prose defect that actively converts a blind spot into evidence of
soundness — and it sits inside the document whose entire purpose is to catch what the automated
checks miss.

The sentence two lines above this item reads: *"guesses will record a false pass."* The document
diagnosed the failure mode and then committed it in the next paragraph. That is not carelessness —
it is the same proximity blindness R11 shows in the design record: **the author of a warning is the
person least able to see themselves violating it, because they have just finished thinking about
it and feel covered.**

Two smaller defects in the same item: the URL cites `:8124`, an origin that predates the
resources-freeze fix, and the step gives no expected value — no panel name, no field, no number —
so any rendered page satisfies it.

The fix is to make the step falsifiable, which also converts it into a regression test for R13/R14:
*paste `?scenario=prefix-cache` and confirm the page states that the scenario is unavailable and
names what it is showing instead.* Written that way it fails today, which is the point.

---

## Withdrawn by me

### Audit of my own claims, under the "a deletion is not verified by its replacement" rule

The Lead ruled that a removal claimed tonight must be re-checked by searching for the **old**
string, not the new one. I had two claims exposed to that rule and re-ran both:

| my claim | audited how | result |
|---|---|---|
| R6 "fixed by @bb2ee824" | grepped the **old** assertion, not the new docstring | ✅ holds — the asserting form is gone; what survives is `Deliberately does NOT say the prefix cache engages here`, a guard quoting what it forbids |
| R11 "live and unowned" | grepped the **old** strings `currently exports` / `already agree` | 🔻 **my claim was wrong — it is fixed** |

**R11 is fixed and I broadcast it as live twice after it landed.** @0837fdf9 accepted it as D278 and
applied exactly the prescribed edit — three sites stamped `OBSERVED 00:51` with the superseding
commit named, argument untouched. The old strings survive only inside D278 itself, where the record
quotes them in order to retire them: the same exemption class as R6 and R10, and the third time
tonight that a correct fix leaves its own defect text visible to a naive grep.

I am recording this rather than quietly editing the status, because the shape is mine and it is the
one I have spent the session policing in others: **I re-verified the finding and never re-verified
the ownership.** A finding has two halves — *is it still true* and *is it still unowned* — and I had
been re-running only the first while asserting both. My 🔴 could have sent its author back to
re-fix work they had already done, which is the expensive direction: a false red costs a
verification, but a false red aimed at a specific person costs them a repeat of finished work.

**Unverified, stated as unverified:** I scanned for the Lead's orphaned-doc-block class in my own
lane and found no instance in shipped JS. That zero is **not** a finding — my scan was single-
language and the reported instance is in Rust, so the correct reading is that my instrument was
pointed at a set that excludes it. Same shape as the guard-scope defects above; I have no evidence
either way for the demo's JS.

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
