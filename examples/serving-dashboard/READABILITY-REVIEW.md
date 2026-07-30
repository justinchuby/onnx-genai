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

### R10 — a withdrawn measurement survives at ten sites, three of them assertion strings

The `7.0% slower` prefix result was withdrawn by its author (the re-run came back with the
opposite sign, on a machine where a byte-identical binary swung 9.8% from ambient load). The
retraction reached the discussion four times and the tree zero times.

```
scenario-origins.js · telemetry-provenance.js ×2 · registry.test.js ×2
prefix-counters-forbidden.test.js ×3   (one is an assertion string)
dashboard/honesty.test.js              (assertion string)
check-readme-claims.test.js            (assertion string — SEE BELOW)
```

Search `1341` and `1254`, not only `7.0%`, or a sweep misses two sites.

**Do not sweep `check-readme-claims.test.js`.** It asserts the *disagreement* — "one controlled
run put the shared arm 7% slower, another put it 17% faster" — so its claim was never the effect
size. It is still true, it is the ratified headline, and it is the template the other nine should
become. A blind substitution breaks the only site that was already right.

The replacement claim needs no timing at all and no machine can perturb it: twelve requests with
six deliberately unique prompts produced twelve hits and a 0.9375 rate, so the counter cannot
distinguish reuse from no-reuse.

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
