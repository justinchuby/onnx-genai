# Readability Review — serving-dashboard

Reviewer: readability lane (naming, organisation, documentation freshness, simplicity,
consistency, co-location). Implementation correctness and test quality belong to the Code
Reviewer; architecture and security belong to the Critical Reviewer. This document does not
duplicate their findings.

**Provenance.** Every finding below was verified by execution or by reading the file, in the
worktree `/Users/justinc/Documents/GitHub/onnx-genai-demo`, on branch `feat/genai-demo-dashboard`,
first at `346763a0`, 02:11; **every status line below was re-verified at `484cda07`, 04:06.**
Findings name a **symbol** and **quote
the text**; line numbers are a hint and may have rotted by the time you read this.

> ## 📍 Where these paths resolve — read this before checking any citation
>
> **Every unqualified path in this document is relative to
> `examples/serving-dashboard/` in the repository whose toplevel is
> `/Users/justinc/Documents/GitHub/onnx-genai-demo`.**
>
> That single sentence is the highest-leverage fix in this review, and the reason is
> arithmetic: **it repairs every citation in the file without editing any of them, and it
> cannot rot, because it names no line, no symbol and no file.** Every other citation repair
> we performed tonight had to be redone when the tree moved.
>
> It is necessary because **there is no second project — there is the same project twice.**
> `onnx-genai` and `onnx-genai-demo` share one object store; `crates/admin.rs`,
> `crates/driver.rs` and `crates/cli.rs` exist at *identical* paths in both. A
> fully-qualified path — our most rigorous-looking citation form — resolves cleanly in the
> tree where the defect is absent. **The reviewer gets a clean result, from the wrong tree,
> with no error and no clue.**
>
> **⛔ Do not use `cd $(git rev-parse --show-toplevel)` to protect yourself.** It normalises
> *depth* and preserves *repository*, which is the wrong half. Assert the destination:
>
> ```sh
> [ "$(git rev-parse --show-toplevel)" = "/Users/justinc/Documents/GitHub/onnx-genai-demo" ] || exit 2
> ```

**Path audit of this document, run at `876a9cd7`.** 37 distinct file paths are cited here;
**34 resolve against `git ls-files` (control: 2154 tracked files).** The three that do not are
absent *by construction* and are named rather than hidden: `scenario-reachability.test.js` and
`scenario-substitution-notice.test.js` are names this review **proposes**, and `ui/sparkline.js`
is quoted **as an example of a path that does not exist**.

> **⚠️ That is a limit of the audit, not a clean bill.** A path-existence scanner cannot tell a
> *citation* from a *mention of a non-existent path*, so all three of its hits are false
> positives — the same shape as the 147 negative-declarative assertion messages recorded below,
> which are false by construction in a green suite. **Both belong in a presence scanner and
> neither belongs in a truth scanner.** The audit's real result is narrower and worth stating
> exactly: **this document contains no phantom path of the `dashboard/telemetry-store.js`
> kind** — a plausible prefix whose parent directory resolves and whose leaf does not.

<!-- Machine-checked by check-review-freshness.test.js. Raw hex, never a ref name:
     `review-0` named 6ecd9183 at 03:57 and 0aac6bb1 at 04:21 -- 60 commits apart,
     re-pointed silently, because a tag is a mutable pointer to an immutable object. -->
MEASURED-AT: 8230060c

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

---

## Findings that existed only in chat

These six were published as broadcasts and never landed in this file. That is itself the
defect this document is about: **a finding that exists only in a conversation did not survive
the session**, and several of these were the most load-bearing things measured. Each carries
its status, its predicate, and both controls.

### R16 🔴 `'…/**/*.js'` reaches 36 of 74 tracked files, and exits 0

Measured at `8230060c`, **from the repository toplevel**. `git grep -l '' HEAD -- 'examples/serving-dashboard/**/*.js'` → **36** files; the same command with
`'examples/serving-dashboard/*.js'` → **75**. Positive control: `fetch` reaches 17 files under
the working form. Negative control: `zzz_never_written` → 1, and that one hit is this document
talking about the control, which is the honest answer rather than a suspicious zero.

> ⚠️ **This finding is stated with a predicate that exhibits the finding.** Run either command
> from inside `examples/serving-dashboard/` instead of the toplevel and **both return 0**, because
> a `git grep` pathspec resolves relative to the current directory. So the sentence above is only
> true from one place on disk — which is R18's defect, in R16's evidence. **Every measurement must
> print its container**, and this one now does. It is the third time this session that a
> repo-root pathspec issued from a subdirectory turned a real count into a confident zero.

**The `**` form is NARROWER than the `*` form, which is the opposite of what every reader
assumes.** In a git pathspec `*` already matches `/`, so `dir/*.js` reaches every depth, while
`**/*.js` demands an extra path segment — it drops every file sitting *directly* in
`serving-dashboard/`. That is where `telemetry-store.js`, `telemetry-provenance.js` and `app.js`
live: **the most-edited files on the branch are exactly the ones it cannot see.**

This is the general form worth keeping: *a decorative-looking token silently changed what was
measured, and exited 0.* Its siblings this session were `| tail` (which replaced a suite's
verdict with the pipe's), a `git grep` inheriting a cwd, and a line break inside an expression.
**All four look like formatting. All four are part of the measurement. None of them warn.**

### R17 🔴 The documented test command runs a subset, and reports success

The command in the README executed **282 of 587** tests. The missing files were named rather
than counted, and the list is the finding: `field-keys.test.js` and `stylesheet.test.js` — at
the time, the two files that had just been hardened to enforce the honesty layer. **Their new
guards executed zero times under the documented command.** A reviewer would run it, see
`282 pass, exit 0`, and have verified none of the layer the demo exists to demonstrate.

**Publish the list, not the percentage.** The same defect reported as *255 of 543* persuaded
nobody, because 47% reads as a big chunk. As a list of file names it reads as *the entire
honesty layer is untested*. **A shortfall expressed as a count hides which items are missing,
and the identity of the missing items is the whole finding.**

### R18 🔴 `README.md` and `CONTRACT.md` §7 open with the same clause and end in opposite claims

Both sentences begin *"The script resolves its own directory, so"*. CONTRACT §7 finishes
*"its **results** never depend on where you stand. **This path does.**"* — correct. The README
finishes *"so **it runs from anywhere**"* — false, with the repo-root-relative invocation printed
four lines above it. Adjudicated by running it: from the repo root, raw exit 0, 49 files
discovered.

**A grep on the shared prefix returns two hits that are indistinguishable for six words.**
Seventh member of the *grep cannot see it* family, after arrays, tense, line breaks, negation,
string-concatenation boundaries, and sentence-initial capitals. **Grep sees bytes; meaning
lives in the frame around them, and a subordinate clause is a frame.**

### R19 🟡 `entry.path` is one noun doing two jobs, and it makes a correct guard impossible

`never-bind.test.js` bans the token `path` to keep the model directory off the page. Every
match it fires on is `entry.path` — the catalogue's JSON pointer, an unrelated concept sharing
a noun. **The fix is the rename `entry.path` → `entry.pointer`, not an exemption.** An
exemption teaches the guard to ignore the exact string it exists to catch; the rename removes
the collision. **When a guard needs a whitelist, suspect the vocabulary before the guard.**

### R20 🔴 `review-0` moved 60 commits, and three reviewers pinned verdicts to the name

At 03:57 `review-0` resolved to `6ecd9183`. At 04:21 it resolved to `0aac6bb1` — **60 commits
later**, re-pointed by `eb2abbd3`, announced in a commit message, and quoted in its old form by
three separate verdicts afterwards.

**A tag is a mutable pointer to an immutable object, and the object's immutability is exactly
what hides the move.** Nothing errors; every old SHA still resolves; only the mapping changed.
And `review-1` is timestamped 04:02 while `review-0` is 04:16 — **the numbering implies an
order the timestamps deny, and readers sort by the name without looking.**

Guarded by `check-review-freshness.test.js`, which requires a raw hex SHA and rejects a ref
name. Mutation-proven on four arms — tag name, agent id, unresolvable token, marker deleted —
each raw exit 1, restored raw exit 0.

### R21 🔴 Our agent ids are eight hex characters, and so are our short SHAs

`IMPLEMENTATION-REVIEW.md:3` presents `73e77d95` three lines above a real commit SHA. One is an
agent, one is a commit, and **no regex can separate them — same bytes, same alphabet, same
length.** Only asking git whether the object exists *and is a commit* distinguishes them. Any
citation checker that validates SHAs by shape will accept every agent id in the corpus as a
verified anchor.

---

## The error signature of this document, stated because it predicts the next one

**Both of the false claims this review made were claims of COMPLETENESS, never of SEVERITY.**
That is not a coincidence and it generalises past this file.

A severity over-call gets argued down — someone disagrees, the argument is public, the row
survives at a lower grade. **A completeness over-call stops the next reader looking.** It
closes a search rather than opening one, and it does it silently, because the reader who was
deterred never files anything and never knows they were deterred.

It predicts the shape of the session's other misses: a `NOT_YET_PUBLISHED` list that was taken
as exhaustive, a performance corpus whose four exemptions were read as none, and four
independent reviewers who converged on a test command that missed a third directory — **not
because they disagreed, but because they all read the same incomplete map.** Four experts
agreeing is not evidence when the agreement has one source.

The unified law, owed jointly to the architect who supplied its mirror: **a partial audit
over-credits the rows it omits; a blanket disclaimer under-credits the rows that earned better;
and a count over-credits every row it never examined.** One defect — *the scope of the claim
not matching the scope of the evidence.* All three are invisible because all three are
literally true. **The dishonesty is entirely in the quantifier, and a quantifier is the one
part of a sentence that no instrument here can see.**

The remedy this document now ships is the per-row form: every heading carries its own status,
its own SHA, and both controls. **A document-level claim in either direction is the thing to
avoid — including a document-level disclaimer, which costs nothing in a file with no anchors
and strips 180 machine-checked ones in a file that has them.**

### This review's own citation hit rate is 0 of 2

Stated here because it is unflattering and because the instrument rewards the wrong thing. This
document has the **fewest** `file:NNN` citations of the three reviewer deliverables — 2, against
90 and 43 — and that was published as evidence of cleanliness. **Both are wrong.** The census
that scored it well counts citations; it has never validated one, neither bounds nor content.
*Low citation density is a different trade, not rigour.*

### R22 🔴 `git cat-file -e` cannot verify that a tag names a commit, and was used to

A verdict published tonight reads: **`review-0` = `6ecd9183` · verified by `git cat-file -e`.**
That instrument cannot answer that question. `git cat-file -e 6ecd9183` exits 0 because the
commit exists. `git cat-file -e review-0` exits 0 because *whatever review-0 points at* exists.
**Both succeed no matter what the tag names, so the pair proves existence twice and identity
never.** Measured at `8230060c`: `review-0` resolves to `0aac6bb1`, not `6ecd9183` — the two are
60 commits apart, and four separate agents published the old mapping after it stopped being true.

The correct predicate is a comparison, not an existence check:
`[ "$(git rev-parse review-0)" = "$(git rev-parse 6ecd9183^{commit})" ]`.

**The general form, which is this session's most repeated defect and now has five specimens:**
*an instrument answered a question adjacent to the one asked, and the adjacent answer was
true.* A histogram that is registered but never observed serves the same 19 lines as one with
real data. A control proves the instrument reached the subject and says nothing about whether
the subject is the quantity you meant. **A true answer to the wrong question is far more
dangerous than a false one, because it survives every check aimed at falsehood.**

### R23 🔴 A review tag freezes the artifact and leaves the verdicts floating

Two reviewers formed blocking verdicts four minutes and two minutes before a tag was cut, on
SHAs that are ancestors of it. Both were true when written; both were false of the tagged tree;
neither reviewer could have noticed, because **nothing re-scores a finding when the branch moves
past it, and a finding does not know a tag was cut after it.**

This document had the same defect: it declared `484cda07`, sixteen commits behind the review
point. It is now guarded — `check-review-freshness.test.js` fails when any review document's
`MEASURED-AT` is an ancestor of the newest review tag, and the failure names the document, the
SHA and the boundary.

**Two details in that guard are the finding rather than the implementation.** It picks the
boundary tag by **commit date, not by name**, because `review-1` is timestamped 04:02 while
`review-0` is 04:16 — the numbering implies an order the timestamps deny, and readers sort by
the name without looking. And it **resolves the tag once and prints the SHA it resolved to**,
because the obvious spelling of this check — `merge-base --is-ancestor <mine> review-0` —
re-introduces the exact defect the marker exists to block: **it anchors a freshness decision to
a mutable name, so the set of documents it condemns changes silently, and the run that
condemned you cannot be reproduced from its own output.**

### R24 🟢 Which tree reviewers extract was never written down, and my own guard guessed wrong

Three `review-*` tags exist and nothing in the repository said which one was authoritative. At
`33c7a77c` three parties held three different answers, all defensible: the lead declared
`review-1` (`fca13038`); a secretary's board carried `review-0` (`6ecd9183`, correct when
measured at 03:57); and **`check-review-freshness.test.js` — my own guard, shipped an hour
earlier — inferred `review-2` (`0bc86726`) by taking the newest tag by commit date.**

**The guard's answer was the most dangerous of the three precisely because it was automatic.**
It would have enforced a boundary nobody chose, on every review document, silently, with a
plausible justification attached. *A wrong answer a human states can be argued with. A wrong
answer a test computes gets obeyed.*

Fixed by `REVIEW-POINT.md`, which records the declaration, and by making the guard **read it
and fail while naming the candidates when it is absent, rather than picking one.** Refusing to
answer is a valid measurement; inventing a denominator is not. Mutation-proven on three arms —
declaration deleted, SHA given as a tag name, review point moved ahead of this document — each
raw exit 1, restored raw exit 0.

Two properties of the tags made this worse and both are worth fixing at the source. **The
numbering does not sort by time** (`review-1` 04:02, `review-0` 04:16, `review-2` 04:19), and
**all three are lightweight tags** — `git cat-file -t review-1` returns `commit`, not `tag`. A
lightweight tag carries no tagger, no date, no message and no reflog, which is exactly why
`review-0` could move 60 commits leaving no evidence in the repository that it had moved. **An
annotated tag would have carried the designation in the object itself and this file would be
unnecessary.**

---

## Triple review — pass 1, readability arm, at `review-1` = `fca13038`

Three findings. R1 is **fixed in this commit**; R25 and R26 are new and are the
reason two other tasks in my queue were *not* done.

### R1 🟢 FIXED — the `PanelModule` typedef was the lone outlier, and the brief's reason for it was wrong

`dashboard/index.js:59` declared:

```js
 * @property {(root: HTMLElement, store: object) => {destroy: () => void}} mount
```

Every panel module returns `unmount`. Read, not grepped — six of six:

```
kv-memory.js:159    return {  / :160 unmount() {
prefix-cache.js:177 return {  / :178 unmount() {
requests.js:132     return {  / :133 unmount() {
scheduling.js:166   return {  / :167 unmount() {
system.js:138       return {  / :139 unmount() {
throughput.js:144   return {  / :145 unmount() {
```

One word changed. No behaviour, no test touched.

**⛔ But the justification I was handed was false, and acting on it would have been a
disaster.** The brief said *"every implementation returns `unmount`, so the typedef is the
lone outlier"* — the second clause is true, the first is not a statement about the codebase.
`destroy` is **legitimately live** in the very same file for *different* objects:
`entry.roving.destroy()` at `:175` and `adapter.destroy()` at `:182`. Across the dashboard,
8 modules expose `destroy()` and 7 expose `unmount()`. A reader who accepted *"every
implementation returns unmount"* and reached for a global rename would have broken the
roving-focus and adapter teardown paths.

> **The defect was exactly one typedef. The brief described it as a vocabulary.** A correct
> fix arrived with a reason that, if believed one step further than the fix, does damage.
> **Scope of the fix and scope of the reason are separate quantities, and only the fix was checked.**

**And the instrument nearly hid it.** My first probe was
`grep -E "return \{[^}]*unmount"` across the panels — it returned **nothing**, and its
positive control *also* returned nothing, because these returns span two lines and grep is
line-oriented. Had I run only the negative arm I would have published *"no panel returns
`unmount`"* — the exact inverse of the truth, with a clean-looking zero behind it.
**The control is what converted an inverted finding into a correct one.**

### R25 🔴 NEW — deriving a review document's `MEASURED-AT` from its last commit manufactures freshness

I was ordered to apply the `MEASURED-AT` guard to `IMPLEMENTATION-REVIEW.md` and
`REVIEWER-BRIEF.md`. **I did not, and this row is why.**

The obvious mechanical source for the marker is the document's last-modifying commit.
For `IMPLEMENTATION-REVIEW.md` that is `d2219ea8` — which is **at-or-after `review-1`, so
the guard would have gone green.** But that document states its own measurement point in
prose, at line 4:

```
Originally reviewed at: `24d831a2`
```

```
git rev-list --count 24d831a2..review-1   ->  222
```

**222 commits.** The author's declared measurement point predates the review point by 222
commits; the honest marker for that file is **red**. Had I stamped the last-modified SHA I
would have certified a 222-commit-stale review as fresh — **automatically, in its author's
name, using the very instrument built to detect staleness.**

> **A file's last-commit SHA records when it was *written*, never when its claims were
> *checked*.** Documents get touched to fix a typo, add a row, repair a citation — none of
> which re-verifies the rows already there. **Editing is not re-measuring, and only the
> author knows which it was.**

So the marker **cannot be conscripted**. It has to be self-declared or it is a forgery with
a passing test attached. What I can honestly supply is the measurement, which is done:

| document | own declared point | vs `review-1` | one-line fix, for its owner |
|---|---|---|---|
| `IMPLEMENTATION-REVIEW.md` | `24d831a2` (line 4, prose) | ⛔ 222 behind | re-measure, then `MEASURED-AT: <new sha>` |
| `ARCHITECTURE-SECURITY-REVIEW.md` | none declared | unknown | `MEASURED-AT: <sha you measured at>` |
| `REVIEWER-BRIEF.md` | none declared | unknown | `MEASURED-AT: <sha you measured at>` |

**Note the second-order finding: `IMPLEMENTATION-REVIEW.md` already had the convention.**
It declared its measurement point in prose at line 4 and had done so all night. No machine
read it, so it decayed 222 commits without a single warning. **The convention was never
missing — only its enforcement was.** That is the whole argument for the marker in four
words: *prose is not a predicate.*

### R26 🟡 CONFIRMED — colour literals, and the empty cell that nearly banked it as clean

The co-location convention holds: shipped dashboard JS carries no colour literals, with one
exception — **`dashboard/sparkline.js`, 8 literals in 622 lines**, sole violator, confirmed
by a sweep with no pathspec at all.

**⛔ The brief named the file `ui/sparkline.js`. That path does not exist at the tag.**
`git show review-1:ui/sparkline.js | grep -c '#...'` returns **`0`** — and a `0` from a
missing file is byte-identical to a `0` from a clean file. I was one step from recording
*"co-location convention: clean"* against a file that isn't there.

> **This is the fourth time tonight the same shape has caught me** (repo-root pathspec from
> a subdirectory ×3, now a nonexistent path). Every instance produced a **confident zero and
> exit 0**. It is the single highest-yield defect class in this review and it has exactly one
> cure: **never accept a zero without a control that proves the instrument could have
> returned non-zero.** Not a rule about care — a second command.

---

**Suite at the time of the R1 fix:** `bash run-tests.sh`, raw exit **1**, `tests 663 ·
suites 103 · pass 660 · fail 3`, from
`/Users/justinc/Documents/GitHub/onnx-genai-demo/examples/serving-dashboard`, head `ffaef0cd`.
**The 3 failures are not mine and not this change**: they reproduce identically with my edit
stashed (control run), `dashboard/index.js` appears **0** times in the failure output, and the
failing assertion is `no [data-state] rule is unqualified` — a CSS rule test. Five other
agents' files were dirty in the shared tree at the time.

---

## R14 disclosure case — 🟢 CLOSED by execution, and the reason it survived is a new defect

**Retired at `ee8542d2`.** R14's type-level half was already recorded fixed. Its *live*
remainder — does the UI actually **tell the visitor** it substituted a scenario? — is now
closed. @1cb42f0e found the proof; `ui/scenario-switcher.test.js` landed at `f7b884b0`
(`git merge-base --is-ancestor f7b884b0 HEAD` → ancestor). Executed at HEAD, **raw exit 0,
5 tests / 5 pass / 0 fail**:

```
:46  renders a notice naming the rejected id and what is shown instead
:59  announces it, because the visitor is looking at the panels and not at this
:71  renders NOTHING when we showed what was asked for          <- the control
:85  keeps the substitution notice separate from the contradiction notice
:108 escapes nothing into markup, because the id comes off the query string
```

`:71` is the arm that makes the other four mean anything — a notice that always renders
discloses nothing. **The author shipped the negative control inside the suite.**

### R27 🔴 NEW — two unrelated suites share a basename, and "duplicate" is the wrong word for it

This finding was invisible for hours because it lives in `ui/`, a **third** test directory
that the two-glob command four reviewers independently converged on cannot reach. It has
been described on the channel as a *duplicate file* that the runner merely warns about.
**It is not a duplicate.** At HEAD the two blobs differ (`db7b3b06` vs `93bd9a47`) and they
test entirely disjoint things:

| file | tests | subject |
|---|---|---|
| `scenario-switcher.test.js` | 10 | reachability, peer servers, `describe()`, CSS-class coverage |
| `ui/scenario-switcher.test.js` | 5 | the substitution **disclosure** — R14's open half |

Both raw exit 0. **Neither is redundant, and deleting either to silence the warning would
delete real coverage** — which is what "duplicate" invites a reader to do.

> **A colliding basename is not a duplicate; it is two things wearing one name.** The word
> chosen for a defect decides what the next reader does about it, and *duplicate* prescribes
> deletion. The fix is a rename that says what each covers — `scenario-reachability.test.js`
> and `scenario-substitution-notice.test.js` — after which no warning exists to suppress.

**And the mechanism that hid the cure is the sharpest instance of my own doctrine yet.** I
have been writing that a completeness over-call stops the next reader looking. Here the
over-call was **the instrument's, not any reviewer's**: a two-glob runner reported a large
green number and exit 0 while never reaching `ui/`. Nobody misread it. **It was a true
report from an incomplete instrument — and a true report is the one artefact no honesty
check can catch.**

The cure is already shipped and must not be restated: `run-tests.sh` **discovers** rather
than enumerates. Cite the script, do not copy its internals.

> **⚠️ Cite it by name, not by line.** The order that sent me here placed the discovery call
> at line 56; at HEAD line 56 is `echo "pwd:    $(pwd)"` and the call is at **66**. Ten lines
> of drift inside one night. This is the fourth line-citation in this review that moved
> between the read and the check. **A line number is a guess about a moving file; a symbol,
> or a quoted string, is an address.**

**Runner state at this measurement:** `discovered: 52 test files`, raw exit **1**, with the
first failure being its own tracking check — `1 test file(s) ran here but are not committed:
shipping-tree.test.js`. Not mine, and not a test failure at all: the runner is refusing to
report a number that describes this desk rather than the branch. **That refusal is the
single most valuable line it prints**, and it is the same discipline as declining to invent
a denominator.

---

## For the brief: the two laws about comments, and why this review distrusts them

These are documentation findings, which is why they sit in the readability lane rather than
the correctness one. **Both say the same thing from opposite ends of a defect's life.**

### A prediction in a comment is a test that never runs

`telemetry-provenance.js` contains a `reason` field, committed *before* the incident, that
describes tonight's P1 disclosure in the **future tense** — naming the endpoint, the
mechanism and the blast radius. It was right. It was specific. It was ignored, because
**nothing executes a `reason` string.**

### The obituary pattern: a well-fixed bug leaves a searchable description of itself

The complement, and the more expensive of the two. A past-tense comment narrating a dead
defect and a present-tense defect are **the same bytes to every grep, every scanner, every
dashboard and every agent.** This produced a false re-dispatch loop against an item that had
been closed for over an hour, and it is why *a rising count is a prompt to read, never
evidence of a regression* — **the better the fix, the more prose it leaves at the scene, and
the higher the counter climbs.**

> **Together: our comments knew about the defects that were already fixed and the ones that
> were still coming, and not one comment ever stopped a defect from shipping.** The
> prescription is not *write fewer comments*. It is: **if a comment states a checkable
> property, the comment is a specification and it belongs in a test. If it cannot be
> checked, say so in the comment itself** — the way `request-deadline.js` prices its own
> limits rather than overselling them, which is the best paragraph in this codebase.

### Instrument-failure catalogue — the readability-lane entries

Six classes were catalogued crew-wide tonight. **Five of the six fail toward green**, and
that asymmetry is not chance: *a tool that errs toward red is fixed within minutes because
somebody is blocked by it; a tool that errs toward green survives the whole project.*

Three are mine and all three are naming or organisation defects wearing an instrument's
costume:

1. **`' +` concatenation boundaries defeat phrase-grep.** 96 in `telemetry-provenance.js`,
   279 across the branch — **concentrated in exactly the honesty files everyone was
   grepping.** A sentence split across a string concatenation is invisible to a search for
   the sentence. **The prose is correct and unfindable, which for a reviewer is the same as
   absent.**
2. **A git pathspec of the form `dir/**/*.js` is NARROWER than `dir/*.js`** — 36 files
   versus 75. In a git pathspec `*` already crosses `/`, so `**/*.js` demands an extra path
   segment and silently drops every file sitting directly in the directory.
3. **A repo-root pathspec issued from a subdirectory returns a confident `0` and exits `0`.**
   This caught me three separate times, including once while stating finding 2 — **my
   finding that a pathspec silently under-reaches was itself measured with a pathspec that
   silently under-reached.**

> **The single rule that covers all three, and the one I would put above every finding in
> this document: never accept a zero without a control that proves the instrument could have
> returned non-zero.** Not a rule about care — a second command. Every zero in this review
> ships with one.

### My error signature, as a crew-wide rule

Both of my false claims this session were claims of **completeness**, never of **severity**.
That distinction is the whole finding:

> **A severity over-call gets argued down by the next reader. A completeness over-call stops
> the next reader looking** — silently, because the deterred reader never files anything and
> so never appears in any count. **A narrow red gets challenged; a narrow green closes the
> question.**

The operational form is three words: **publish the list, not the percentage.** A list can be
checked row by row by someone who was not there. A percentage cannot be checked at all, and
it over-credits every row it never examined. My own instrument scored me 5-of-6 when the
truth was 3-of-6, and it flattered me precisely because it reported a ratio.

---

## R28 🔴 NEW — a pinned gate SHA turns every document in the tree into a stale read, and a freshness guard cannot save you from it

**Measured at `8eaebfb8`.** The release gate scores items 3–10 at `1bca52a8` and its
reasoning is **correct** — I checked it before writing this, because the claim was travelling
on my own commit:

```
git merge-base --is-ancestor 1133a874 1bca52a8   -> YES   (04:12:26 -> 04:12:41, 15 s)
reverse                                          -> NO    strict order, no ambiguity
delta                                            -> 1 file, READABILITY-REVIEW.md, 0 .js
```

The suite result transfers because **the delta cannot reach the thing measured** — not
because the gap is short. That is the right test and I have nothing to add to it.

**⛔ The defect is downstream of a correct decision.** `1bca52a8` is a commit to *my review
document*. Naming it on the board makes it the canonical read-point for the whole tree, and a
reader who checks it out to reproduce the score also reads every document there:

| | at `1bca52a8` | at HEAD |
|---|---|---|
| lines in `READABILITY-REVIEW.md` | 566 | **1077** |
| R25 · R26 · R27 · repo preamble · comment laws | **0** | present |
| `MEASURED-AT` marker | **absent** | `8230060c` |

**Half the document did not exist yet.** A reader arriving there sees a shorter, cleaner,
entirely plausible review with no indication that anything is missing — **and absence is the
one defect no reading can detect.**

> **A score is a property of a commit. A review is a property of the tree it describes.**
> Pinning both to one SHA serves the first correctly and silently breaks the second. The two
> artefacts have opposite freshness requirements and the board gives them one coordinate.

**☠️ And the part that indicts my own guard.** `check-review-freshness.test.js` exists to stop
exactly this. **It cannot fire here, because at `1bca52a8` the guard does not exist either** —
and neither does the `MEASURED-AT` marker it reads. The reader gets no warning, no red, no
marker: **the guard fails *absent* rather than *red*, and absent is indistinguishable from
clean.** That is the sixth instance tonight of *a scan that matches nothing and a tree with no
defects are byte-identical from here* — arriving, this time, on the instrument built to
prevent it.

> **A freshness guard is only ever as fresh as the commit you read it from.** It protects the
> branch tip and nothing else. Any pinned, archived, extracted or detached read is outside its
> reach **by construction**, and those are precisely the reads reviewers perform.

**✅ The fix is one line on the board, and it costs nothing:** when a gate cites a SHA, say
what it is a claim about. *"Items 3–10 scored at `1bca52a8`; **documents in that checkout are
stale — read prose at the branch tip.**"* The score stays reproducible; the reader stops
inheriting a six-hour-old review as though it were current.

**⚖️ To be scrupulous about what this is not:** it is not a criticism of the gate, whose
delta-transfer discriminator is the best measurement instrument published tonight, and it is
not an argument against pinning. **It is the cost of pinning, stated once, so nobody pays it
without knowing.**

---

## R29 🟡 NEW — an alarm can be correct about the hazard and wrong about every piece of its evidence

**Measured at `29870de6`, toplevel asserted.** I was sent an urgent warning that
`/private/tmp/review-0` had been reaped, that my banners had cited it all session, and that
twelve honesty guards were dead with sixty-six tests missing from the denominator. **The
underlying hazard is real and I have adopted the fix. Three of the four factual claims are
false, and I checked all four before replying.**

```
① "you have bannered extract=/private/tmp/review-0 on every broadcast"
   git worktree list | grep -c 086            ->  0     I HAVE CREATED NONE, ALL SESSION.
② "git -C /private/tmp/review-0 rev-parse HEAD -> fatal: not a git repository"
   ACTUAL                                     ->  0aac6bb1d34688c0…   IT IS ALIVE.
③ "12 failures, 66 tests that never run"      ->  DESCRIBES THE BROKEN TREE, NOT THIS ONE.
   check-*.test.js in the real tree: 12 files · 96 tests · 11 of 12 raw exit 0.
   The single red is check-source-citations (4/6) — the KNOWN coverage ratchet,
   red because the de-rot repair PROMOTED line citations to symbols. Not a worktree fault.
④ "6ecd9183 is still an ancestor, findings re-derivable"   ->  ✅ TRUE. VERIFIED.
```

**⚖️ And yet the warning was worth sending, because its *core* is right and I have taken it:**
`review-0` resolves to `0aac6bb1` now and resolved to `6ecd9183` earlier. **The directory
`/private/tmp/review-0` currently holds `0aac6bb1` — so it did not die, it *moved*, which is
strictly worse: a dead worktree announces itself and a re-pointed one does not.**

> **A directory name is a coordinate, and it ages exactly like a line number.**
> `/private/tmp/review-0` was a true name when it was created. The tag moved, the checkout
> moved, and the *name* could not — names cannot rot, which is precisely why they go on
> lying. This is `file:NNN` wearing a filesystem path, and it is the seventh member of the
> family this review has catalogued.

**✅ Adopted: pin the SHA in the banner, never the tag name.** The string `review-0` has
denoted two different commits tonight and no banner containing it can distinguish them.

### The part that is a recommendation, not a complaint

**This is the fifth claim tonight attached to my name or my SHA that I had to check before I
could refuse it** — and the checking is not the expensive part. **The expensive part is that a
claim wearing a named reviewer's evidence reads as already-verified, so the next reader does
not check it at all.** That is my own completeness law arriving from the outside: *a narrow
green stops the next reader looking.* **The remedy is the one already ratified and it is free:
cite the bytes, not the agent.**

**And the reason my own work needs no re-qualification tonight is an instrument choice, not
care.** Every finding in this document was taken with `git show <ref>:<path>`. I created zero
worktrees, all session, deliberately:

> **`git show <ref>:<path>` reads the object store. A worktree materialises a working tree.**
> The object store is immutable, shared, and cannot be reaped, re-pointed, half-created,
> `cd`-missed or filled to 99%. A working tree can be all six, and tonight it was four of
> them. **Reviewers downgraded their own greens because the tree their evidence lived in
> became unattestable — the measurements were fine; the custody was not.**

**➡️ The prescription, and it is the single cheapest thing in this review: prefer the
instrument with no state to lose.** A worktree is only required to *execute*. For reading
source — which is the whole readability lane — it buys nothing and can silently cost
everything.

---

## R30 🔴 NEW — the review tags do not sort, and the review point moved under this document

**Measured at `7b177e32`.** `REVIEW-POINT.md` has been re-pointed to `review-0` = `0aac6bb1`,
superseding `review-1` = `fca13038`. The earlier declaration was **not wrong — it was spent**;
`fca13038` is a strict ancestor, so everything measured at it is re-derivable.

**The naming defect found while re-pointing is the one worth keeping.** All six pairwise
ancestries, tested:

```
04:02:36  fca13038  review-1     ⬅ FIRST
04:16:22  0aac6bb1  review-0     ⬅ SECOND
04:19:23  0bc86726  review-2     ⬅ THIRD

review-1 is an ancestor of review-0     ⬅ THE EXACT INVERSE OF ITS NAME
review-1 is an ancestor of review-2     ✅ as named
review-0 is an ancestor of review-2     ✅ as named
```

**Two of three comparisons match the numbering and one is exactly inverted — the worst
possible ratio.** Enough agreement to confirm the assumption, enough disagreement to make it
false. **A name that is *usually* monotonic is more dangerous than one that never is**,
because the exceptions arrive as surprises instead of as habits. And nobody will ever catch
it, because **no reader runs an ancestry check to discover whether `review-0` precedes
`review-1`.**

> **A sequential name is a claim about ordering, and this repository does not keep it.** These
> are lightweight tags: any of them can be re-pointed with `-f`, and `review-0` was, twice.
> **The tag name is a nickname, not an address — and it does not even sort.**

### R28 and the extract defect are one finding, and I had only half of it

I filed R28 as *a pinned SHA makes every document in that checkout a stale read*. The gate
secretary independently filed the opposite polarity: **a stale extract freezes a defect that
has since died** — both P1 render sites are present in the older tree and absent in the newer,
so a reviewer working from the old extract files a real defect **correctly for the tree in
front of them and wrongly for the tree we ship.**

**These are the same defect and neither of us had both halves:**

| | what the reader gets | direction |
| --- | --- | --- |
| R28 (mine) | findings that **do not exist yet** | under-reads — misses live defects |
| the extract defect (theirs) | findings that **no longer exist** | over-reads — files dead defects |

> **The unified statement: an extract removes *drift*, not *staleness*, and we adopted it
> believing those were one property.** Drift is the tree moving **under** you; staleness is the
> tree having moved **before** you. **Freezing a coordinate cures the first and *guarantees*
> the second.** Every argument made tonight for pinning was an argument against drift, and not
> one of them was an argument against staleness — **so we bought a cure for the half we had
> noticed and paid for it with the half we had not.**

**The prescription is the same one line in both directions, and it is free:** a pinned
coordinate must state **what it is a claim about**. Code at the extract; prose at the branch
tip; and the tip re-checked before any finding is filed from a frozen tree.

---

## R31 🔴 NEW — pass 2's pinned artifact excludes the fixes filed for pass 1, and both poles of R30 are live in it at once

**Measured at `35aaa6e2`, toplevel asserted.** `review-2` = `0bc86726` (04:19:23) is pinned as
the scoring and review artifact. **It does not contain any of my six landed commits**, tested
individually with `git merge-base --is-ancestor`:

```
ee8542d2  R1 typedef fix      ⛔ NOT in review-2
32bf88e7  R27 basename        ⛔ NOT in review-2
608de6c2  repo preamble       ⛔ NOT in review-2
889eb00e  R28                 ⛔ NOT in review-2
d7660e15  R29                 ⛔ NOT in review-2
c1f215fe  R30                 ⛔ NOT in review-2

GROUND TRUTH, review-2 tree, dashboard/index.js:59:
  * @property {(… ) => {destroy: () => void}} mount     ⬅ THE DEFECT, STILL THERE
```

**✅ First the good news, measured before the complaint, because the gate item is what matters
most:** the P1 is **genuinely closed in `review-2`** — `1133a874` and `f025ae58` are both
ancestors, both render sites read **0**, and the control `server.model_id` reads **1** in the
same file. The pinned artifact is sound on the thing it was pinned for.

**⛔ The defect is that R30's two poles are now live simultaneously in one artifact**, which is
the case I described as hypothetical forty minutes ago:

| pole | what pass 2 gets at `review-2` | consequence |
| --- | --- | --- |
| over-read | `index.js:59` still says `destroy` | **a reviewer re-derives R1 as LIVE — correctly for that tree, wrongly for the branch** |
| under-read | my document lacks R25–R31 and the preamble | **the correction that would have warned them is not in the tree they are reading** |

> **The frozen tree contains the defect and not its retraction, because a fix and its
> write-up land in the same minute and a pin can fall between them.** The artifact is not
> merely stale — **it is stale asymmetrically, and it decays in the direction that
> manufactures work.**

**✅ What pass 2 should do, concretely, so nobody spends a cycle on it:**

1. **Score the *suite* at `review-2`.** That is what a pinned tree is good for and the number
   is real there.
2. **Read *prose* at the branch tip.** Code at the extract, prose at the tip — the same split
   the gate secretary applied to their own brief when they noted a document cannot contain the
   announcement of its own SHA.
3. **Do not re-file `dashboard/index.js:59`.** It is fixed at `ee8542d2`, verified by presence
   of the bytes in `HEAD`. If your pass finds it, **your tree is older than the fix, not newer
   than the finding.**

**⚖️ And the runner's duplicate-filename warning is R27, already filed and committed at
`32bf88e7`** — with one correction that matters, because the word chosen decides what the next
reader does: **the two files are not duplicates.** Different blobs, disjoint subjects, 10 tests
versus 5, both raw exit 0. **Deleting either to silence the warning deletes real coverage**, and
*duplicate* is a word that prescribes deletion. The fix is a rename. **The guard that caught it
deserves the credit it was given — it is the only instrument on this branch that found a
problem without a human aiming it.**

### Self-audit on R31's own landing — my verification control failed, twice, in the way I catalogued

**The commit was sound; my check of it was not.** After landing `06647d2c` I confirmed
presence with `grep -c` on a phrase that turned out to be **wrapped across two lines** in the
committed file. It returned **0** — indistinguishable from *the commit did not land*.

```
verify  "older than the fix, not newer than the finding"   -> 0   ⛔ FALSE NEGATIVE
cause   line 1300 ends "...not newer" / line 1301 begins "than the finding."
truth   "R31" -> 2 · "pinned artifact excludes" -> 1 · "stale asymmetrically" -> 1
        · "manufactures work" -> 1        ✅ IT LANDED
repair  a multiline probe: tr '\n' ' ' then grep   ->  0  ⛔ THE FIX ALSO FAILED
        and its negative control also 0 — so that probe distinguishes nothing
```

**This is instrument-failure class 1 from my own catalogue — *a sentence split across a line
boundary is invisible to a line-oriented search* — landing on the command I use to verify
every commit I make.** Third time tonight a defect I documented has bitten the hand that
documented it.

> **Two things are worth keeping, and the second is the one I did not expect.**
> **① A false negative from a verification control is the most dangerous direction it can
> fail in**, because the honest response to *"my commit did not land"* is to commit again —
> **so this defect's natural remedy is a duplicate commit.** It fails toward doing more work,
> which is why it never looks like a defect.
> **② The clever repair was worse than the simple one.** The multiline probe failed *and its
> negative control failed identically*, so it distinguished nothing — while four boring
> single-line anchors settled the question immediately. **When a control fails, prefer more
> controls over a smarter control: a broken instrument and a clean tree are byte-identical,
> and a *sophisticated* broken instrument is merely harder to doubt.**

**Corrected practice, adopted for the rest of this review: verify presence with a short
anchor that cannot wrap** — a finding id, a heading, a four-word phrase — **never a sentence.**

---

## Caption precedence (`panel-kit.js:266`) 🟢 CLOSED — CLEAN. @c0de4c2e was right not to score it, and I was wrong to suspect it.

**Re-derived at `7b12c962`, toplevel asserted.** @c0de4c2e listed this row `VERIFIED AT 6ecd9183 · NOT
RE-RUN · NOT SCORING IT`. **Declining to score a row you did not measure is the correct call and I
want it on the record as such.** I have now re-run it, and **the finding does not survive.**

```
panel-kit.js:266   const label = options.label ?? field?.label ?? 'value';
JSDoc :260         @param {string} [options.label] Overrides `field.label` for the aria sentence.
```

**I went looking for a signature-vs-design mismatch — my own brief's sharpest test — and found the
opposite.** The doc says the override is *for the aria sentence*; `label` reaches **exactly five
sites, all five `aria-label`, zero visible-text surfaces.** The doc is precise, including the word
*aria*, which a looser writer would have omitted.

**🎖️ And the code around it is the best-documented function I have read on this branch.** `:434–437`
explains *why* the age must be in the accessible name — *"announcing 'queue depth 41' while the screen
says '41 · 12s old' hands the number to a screen-reader user stripped of the one qualifier that makes
it honest"* — and `:440–441` gives the unit decision as *"the same defect, heard instead of seen."*
**Both are WHY, not WHAT. Neither restates the code. Both name the visitor who gets hurt if it
regresses.** This is the standard the rest of the file should be measured against.

## R32 🔴 NEW — a window-bounded measurement's positive control proves the window is *non-empty*, never that it is *complete*

**This is how I nearly filed the false finding above, and the near-miss is worth more than the row.**

```
WINDOW I DREW      renderField = lines 264..430          ⬅ GUESSED THE END
COUNT              aria-label sites -> 4                  ⬅ ALL FOUR IN PLACEHOLDER BRANCHES
POSITIVE CONTROL   'wrapper' in same window -> 38         ✅ HEALTHY
NEGATIVE CONTROL   'zzqqxx' in same window  -> 0          ✅ HEALTHY
CONCLUSION I HAD DRAFTED: 'the value branch emits no aria sentence, so options.label
                           is a NO-OP whenever the field has a value'   ⛔ FALSE

TRUE BOUNDARY  renderField = 264..454  (next decl: metricRow at :455)
COUNT          aria-label -> **5**.  The fifth is :438, in the VALUE branch, using ${label}.
```

> **Both controls were green and the answer was still wrong, because the controls validated the
> wrong property.** `'wrapper' -> 38` proves the window **reached the file and is non-empty**. It
> says nothing about whether the window **covers the function** — and a truncated window is
> non-empty by definition. **The control could not fail in the direction the measurement was wrong
> in.**

**This is @12e42da8's doctrine correction ③ — *a control must vary the instrument, not just the
subject* — arriving one hour later on a different lane.** My control varied the *subject* (a
different token) inside the *same* window. **To vary the instrument I had to vary the window**, and
the only honest way to bound a function is to **find the next declaration**, never to guess a
round number.

**⛔ And it fails in the direction that manufactures work, like five of the six failures already in
my catalogue: a window that is too narrow reports a MISSING thing, which reads as a defect.** A
window too wide reports a spurious extra, which someone investigates and dismisses. **Narrow windows
produce confident false findings; wide windows produce noise. We under-draw windows because tidy
numbers look deliberate.**

**✅ Adopted, and it is one command, not a resolution:** bound a function by measuring to the **next
top-level declaration**, and state the boundary and how it was found in the finding itself —
`renderField = 264..454, next decl metricRow at :455`. **A window with an unstated end is an
unfalsifiable measurement.**

## R33 🔴 NEW — `run-tests.sh:3` calls itself *"the one way to run this suite"* and runs zero Rust

**@12e42da8's correction — *we have two suites and one word for them* — has a specific, one-line
site, and it is a naming defect, which puts it in my lane.**

```
run-tests.sh:3   # The one way to run this suite.
  'cargo' in run-tests.sh -> 0        'rust' -> 0        POSITIVE CONTROL 'test' -> 49 ✅
THE OTHER HALF   cargo test -p onnx-genai-server -> 188 tests, 185 pass, 1 FAIL, 2 ignored
```

**The definite article does the damage.** *"**The** one way to run **this** suite"* is true of the
JavaScript corpus and false of the project, and **the sentence gives a reader no way to tell which
noun it means.** Every agent tonight who said *"the suite is green"* read this file's number.
**Nobody misread it — the name is what they read, and the name is unscoped.**

**⚠️ The bitter part: this file is otherwise the most careful instrument on the branch.** It
documents three ways `node --test` lies, reconciles the disk list against `HEAD` in both directions,
and treats *no files matched* as a failure with the comment **"This is the failure that looks like
success."** **An author who thought that hard about false greens still shipped a name that produces
one.** ➡️ ***Rigour inside a tool does not audit the tool's own name, because the name is the one
part the author never has to read.***

**✅ Concrete fix, two lines, no behaviour change:**
```
- # The one way to run this suite.
+ # The one way to run the JavaScript suite. This runner does NOT build or test
+ # the Rust crates -- run `cargo test -p onnx-genai-server` for those, and quote
+ # both denominators. "The suite is green" is not a sentence this repo can parse.
```
**Co-location argument, which is why the comment belongs here and not in a README: the only reader
who needs this warning is the one already running this file.** A scope caveat kept anywhere other
than the tool it scopes is a caveat that arrives after the number.

---

## R34 🔴 NEW — a document that *adjudicates* an accusation contains the accusation's vocabulary, so grepping for the defect finds the refutation and scores it as a hit

**Two agents filed against me in the same ten minutes. Both were right when they measured. Both
are false now, and the two mechanisms are different — which is why both are worth recording.**

### ① @0837fdf9: *"`check-review-freshness.test.js`, 6,493 bytes, `??` untracked — any bare `node --test` total is inflated by two."*

**Correct when taken, to the byte, and now cured. The arithmetic settles it with no appeal to anyone's word:**

```
THEIR TIP   f7116dbe   04:24:16
MY COMMIT   110abb0c   04:28:53   <- FOUR MINUTES THIRTY-SEVEN SECONDS LATER
merge-base --is-ancestor 110abb0c f7116dbe  ->  NO   ⬅ IT REALLY WAS ABSENT
size of the guard AT 110abb0c  ->  **6493 bytes**    ⬅ **THEIR EXACT NUMBER**

AT HEAD NOW:  tracked ✅ · in HEAD tree ✅ · 10440 bytes in HEAD == 10440 on disk
              · porcelain clean · NEG CONTROL (impossible name) -> untracked ✅
```

**🎖️ The byte count is the part that deserves credit rather than defence.** `6493` is *exactly* the
size the file had in the commit that added it — **so they measured the finished file in the last
minutes before it landed, not a half-written draft.** Their inference was sound and their remedy
(*don't quote a bare `node --test` total from the shared desk*) was right for the whole crew, not
just for my file. **They reported an author's file as unshipped 4m37s before the author shipped it.
There is no version of this where either of us should have done something different.**

### ② @73e77d95: *"your last two broadcasts banner `/private/tmp/review-0`, and your counts came from a tree whose provenance is unverifiable."*

**Right about the string. Wrong about what the string is doing — and I can only show that by
quoting my own line numbers, because *"trust me, they're quotations"* is exactly the move I file
against other people.**

```
'/private/tmp/review-0' in my committed doc  ->  5 occurrences
  :1141 :1147 :1149 :1160 :1164   ⬅ **ALL FIVE INSIDE R29**, the section that
                                    ADJUDICATES this accusation. :1147 and :1149
                                    are me QUOTING the alarm's own two claims.
'git archive' in my doc  ->  **0**        'git show' (my stated method)  ->  4 ✅
worktrees registered under my id  ->  **0** (I have created none, all session)
```

> **⛔ This is the obituary pattern — which I filed weeks of session-hours ago against other
> people's fixes — arriving on my own review document.** A well-written adjudication **quotes the
> claim it is adjudicating**, so the refutation and the defect are byte-identical to a
> line-oriented search. **A reviewer grepping my document for the banned artifact finds five hits,
> and every one of them is me refusing it.**
>
> **The countermeasure is not "read more carefully." It is a second command:** a hit inside a
> document is only evidence if it is **outside** that document's quotation and retraction blocks.
> **`grep` cannot see quotation marks any more than it can see negation** — the third member of
> that set, after *negation* and *line breaks*.

### ③ And the substantive contradiction underneath, which is neither agent's error

**They ran `git -C /private/tmp/review-0 rev-parse HEAD` and got `fatal: not a git repository`.
I ran it and got `0aac6bb1`. I re-ran it just now, at `725b10ab`:**

```
directory exists ✅ · .git PRESENT ✅ · registered worktree: 1 ✅ · rev-parse -> 0aac6bb1
```

**One path has been three different KINDS of object tonight:** a worktree at `6ecd9183`, then a
`git archive` extract with **no `.git`** (@c7a654ed, 04:16), then a worktree at `0aac6bb1`.
**Neither of us measured wrongly. We measured different objects through the same name.**

> **⚠️ And this is strictly worse than a stale line number, which is the finding.** A stale
> coordinate resolves to *different content of the same type* — you get a wrong answer in a
> familiar shape. **A path whose object TYPE changed underneath it fails with a
> category error (`not a git repository`) that reads as an indictment of the person who cited
> it, not as drift.** ➡️ ***Content drift produces a wrong answer; type drift produces a wrong
> accusation.***

**✅ What I am adopting, and it costs one line:** when citing any path outside the repository,
**record what KIND of thing it was and how that was established** — `worktree (git worktree list: 1)`,
not `/tmp/review-0`. **A path plus a SHA still does not say whether the thing at that path is
capable of having a SHA.**

---

## R35 🔴 NEW — **a negative control decays the moment you publish it.** Mine fired today, and a second agent poisoned the same token independently.

**I ran a routine negative control at `fa1fd425` and it came back `2`. A negative control that
fires is a dead instrument, so I stopped and traced it.**

```
git grep -lE 'zzqq' HEAD -- examples/serving-dashboard   ->  **2**   ⛔ MUST BE 0

  READABILITY-REVIEW.md:1378   NEG CONTROL 'zzqq..' in same window -> 0  ✅ HEALTHY
  demo-spec.md:2233            NEG CONTROL 'zzqq_field' -> 0 ✅
   ⬆ MINE, written 90 min ago  ⬆ SOMEBODY ELSE'S, written independently

FRESH TOKEN, NEVER PUBLISHED BY ANYONE  ->  0  ✅
POSITIVE ARM, same instrument, 'renderField' -> 21 ✅   (the instrument is fine)
```

> **☠️ The artifact certifying that the control was clean is the thing that made the control
> dirty.** I wrote that line *as evidence of rigour* — a record proving my instrument could say
> no. **The record is now the reason it cannot.**

**⚡ And the part that makes this a class and not an anecdote: two agents, working separately,
reached for the same "obviously impossible" string.** Nobody coordinated `zzq…`. **Control tokens
are drawn from a tiny shared vocabulary of keyboard-mashes, so independent authors collide — which
means the contamination rate rises with the number of careful people on the team.** ➡️ ***The more
agents who document their controls, the faster everyone's controls rot.*** This is the only defect
tonight whose incidence is **proportional to how rigorously the crew behaves.**

**⛔ Which direction does it fail?** A poisoned control returns non-zero, which reads as *your
instrument is broken* — so it fails **loud**, and I caught it in one step. **But it is loud only if
you actually read the control's NUMBER.** @12e42da8 required exactly that an hour ago — *print the
control's number, not the words "control passed"* — **and this is the case that rule was written
for. Anyone reporting "negative control ✅" from a remembered token has published a control they
did not run.**

**✅ Three rules, and the third is the one that costs nothing:**
1. **Generate the control token fresh per measurement.** It must be unguessable, not merely
   improbable.
2. **Never publish the token verbatim.** Publish *that a fresh token returned 0*. **I have
   deliberately not spelled a new one in this document, because doing so would burn it.**
3. **A control token is write-once.** Reusing a published one is reusing a key you posted.

**⚠️ I am NOT striking `:1378`.** It was true when written and it is now the evidence for this
finding. **Repair the instrument, not the evidence** — @c8d9a40e's ruling, and it binds me here.

## R10 🔻 REPAIRED — my own row published a drifting TOTAL. Replaced with the invariant, both measurements SHA-anchored.

**@c7a654ed's ruling — *a retraction that lives anywhere except beside the retracted string has
been filed, not applied* — applied to my own document, on my own row.**

R10's heading claimed the carriers of the withdrawn `7.0% slower` figure **"grew from 5 files to
9."** Re-derived just now, and **I cannot reproduce the 9 with any instrument I can construct:**

```
PATTERN            @484cda07   @HEAD        PATTERN        @484cda07  @HEAD
'7\.0% slower'         5          5         '7%'              11        13
'slower'              10         10         '9\.8'            13        11
'prefix cache'        22         22         '7 ?%'            13        15
                                            ⬅ **NOTHING YIELDS 9, AT EITHER SHA.**
```

**The defensible statement, and it is the invariant rather than the total:** the withdrawn figure
survives in **5 files at both SHAs — stable, not growing.** One of the five is *this document*,
which is the correct state (a retraction must name what it retracts, `:175`, `:215`). **The four
non-self carriers, named rather than counted:** `check-perf-claims.test.js`,
`check-readme-claims.test.js`, `demo-spec.md`, `design/demo-ux.md`.

> **@bb2ee824's law, which I praised and then failed to apply to my own row: PUBLISH THE INVARIANT,
> NOT THE TOTAL. A total rots in minutes on this branch; an invariant does not.** My heading carried
> a growth claim — the most alarming shape a number can take — **and growth is precisely the claim
> that requires two measurements, which is why it is the one that should never be published without
> both SHAs.**

**✅ And the one shipped-JS carrier of the retracted 9.8 % floor is named and owned:**
`scenario-origins.js:94` — @bb2ee824's declared, expiring deferral. **Not mine, and correctly
disclosed by its owner before I found it.**

## ✅ @c7a654ed's co-location ruling, tested against my own document — it PASSES, and here is the measurement rather than my word

```
'panel.title'        total 1  | outside the withdrawal section: **0**
'not-applicable'     total 1  | outside the withdrawal section: **0**
'derived enum'       total 1  | outside the withdrawal section: **0**
POSITIVE CONTROL 'readability' -> 8   ✅ the instrument reads the document
```

**Every withdrawn claim appears exactly once, and that once is beside its withdrawal.** There is no
orphaned original for a grep to land on. **This is true by construction, not by care: those three
were chat-only findings that were withdrawn INTO the table rather than stated live and corrected
later** — which is the general cure. ➡️ ***A finding that was never stated in two places cannot be
retracted in only one.***
