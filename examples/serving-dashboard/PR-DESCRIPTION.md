# Serving dashboard demo: an honest view of a live inference server

> **MEASURED-AT: 1e809173**
> Every number in this document was re-derived at that revision. Any claim you
> cannot re-run from that revision alone is a defect in this document, not a
> matter of trust. No tag names appear here: a tag is a mutable pointer to an
> immutable object, and the object's immutability is exactly what hides the move.

---

## 1. What shipped

Two capabilities that `onnxruntime-genai` does not have, plus the thing that
turned out to be more interesting than either of them.

### Continuous batching

Multiple in-flight generations share a single decode step. The scheduler admits
and retires sequences between steps rather than draining a batch to empty
before accepting new work.

**Verify by mechanism, not by a ratio:**

```
GET /v1/status
  batch_driver         continuous_batch   <- the scheduler is the batched one
  batch_capacity       4                  <- generations sharing each decode step
  batch_driver_detail  "continuous batching is active: up to 4 generations
                        share each decode step"
```

### The mechanism on the wire, as counts

Counts, not durations. Durations on this hardware are not resolvable (see below);
concurrency is.

```
SAME BINARY - SAME FLAGS - SAME PROBE - 4 concurrent requests - only the model differs

  batched arm      MAX active_batch_size = 4    418 samples   distinct {0, 4}
  one-row arm      MAX active_batch_size = 1    577 samples   distinct {0, 1}
  COMPLETIONS      4 of 4 on both arms.  ZERO errors.   <- the denominator
```

`distinct {0, 4}` with no intermediate value is the mechanism itself: all four
generations were admitted to a single decode step, not four that happened to
overlap. Overlapping requests would show 2s and 3s.

**Each arm is the other's positive control.** A `1` from an instrument that has
never returned anything else is a dead needle. The same probe, in the same
minute, returned `4` on the other origin — so `1` is a measurement rather than a
flatline. This is the only reason the negative result is admissible.

### KV cache paging

The cache is a page table over a fixed pool, not a contiguous per-sequence
allocation. Sequences claim and release pages as they grow and finish.

```
GET /v1/debug/kv    -> nine counters, live, per-page
```

### The capability the demo exists to prove is that these two are mutually exclusive

This is the headline and it is a *negative* result, which is why it is stated
first rather than buried. On the model whose execution provider does not report
fixed-capacity present binding, continuous batching is refused — and the server
says so, in the response, in words:

```
GET /v1/status   (the one-row engine)
  batch_driver         per_request
  batch_capacity       1
  batch_driver_detail  "continuous batching is INACTIVE; this engine decodes one
                        generation at a time. The engine refused it: continuous
                        batching requires a shared KV buffer, and this
                        past/present model is not using one: the execution
                        provider did not report fixed-capacity present binding,
                        or it was not opted into at launch"
```

**Two servers, two answers, and the disagreement is the evidence.** A single
server reporting `batch_capacity: 4` proves nothing — it is one number with
nothing to contradict it. Two servers built from one binary, reporting `4` and
`1` on the same field, prove the field is derived from the engine rather than
from a constant. `batch_capacity` and `batch_driver` are computed from one
variable, so they cannot disagree again without a compile error.

### No hero number

**We are not shipping a speedup ratio, and the reason is measurement, not
modesty.** On this hardware, a control effect that is zero by construction
measured a **+58.41% worst-case swing**, with a coefficient of variation of
**19.83%** against a clean-tree baseline of **1.98%**, while load average moved
from 30 to 212 mid-run.

Three significant figures is not a thing this hardware can produce. If you need
a magnitude for a slide, the honest envelope is **roughly 1.6x to 3.9x**, quoted
with that variance beside it — never as a point.

An earlier draft of this work carried a specific figure. It is withdrawn:
~~2.46x~~ — withdrawn, not refined. The measurement was real; the precision was
an artifact of the instrument, and a withdrawn number that keeps circulating
without its withdrawal is the single most expensive object this project
produced.

---

## 2. The honesty layer — the actual novelty

A dashboard that renders a number it cannot justify is worse than one that
renders nothing, because a wrong number is indistinguishable from a right one at
a glance. This layer exists so the page can never do that.

### The provenance catalogue

Every field the dashboard can display is declared, in one place, with where its
value comes from. A field with no catalogue entry cannot render a value. This
inverts the usual default: the page cannot show you something merely because
some object happened to have the key.

### `NEVER_BIND` — display permission as a deny-by-default axis

Some values exist in the process and must never reach a screen. `NEVER_BIND`
makes that a declared property of the field rather than an accident of which
renderer someone wrote. It is a separate axis from "do we have this value":
a field can be present, correct, fresh, and still forbidden.

The alternative — remembering not to render something — failed in this
repository during development, which is why the axis exists.

### `MISATTRIBUTED` — "asked, answered, and answering something else"

The state that has no name in any other dashboard. A request succeeded, a value
came back, and the value describes a different question than the one the panel
asks. Without this state, such a value renders as `measured` and is completely
convincing.

### Five render states, and the distinction that carries the story

```
measured          a real observation, with a timestamp
pending           not yet, and it could arrive
stale             it arrived, and it is now too old to show as a number
unavailable       we asked and could not get it; expect it later
not-applicable    this code path can never produce it; do not expect it
```

`unavailable` and `not-applicable` are deliberately not collapsed. Collapsing
them would flatten the single most interesting thing the demo has to say — the
mutual exclusivity above is told entirely through that distinction.

### Guards that state the condition that retires them

Each guard in this branch names what would make it obsolete. A guard that cannot
say why it exists gets deleted by the next person who finds it inconvenient, and
takes its real check with it.

---

## 3. What is measured, and how

**Both suites, both denominators, one revision, raw unpiped exit codes.**

We had two test suites and one word for them for most of this project's life.
That is fixed here, permanently, by never printing a bare `X/Y` again — the
slash meant `passed/total` in one report and `tests/suites` in another, and no
instrument we own can grep a punctuation mark for its schema.

### JavaScript suite

```
REVISION            37d0d72e
RUNNER              node --test  (node v25.6.1)
FILES DISCOVERED    56
FILES EXECUTED      56          <- reconciled; a dropped file exits 0 silently
TESTS               749
SUITES              115
PASS                749
FAIL                0
SKIPPED             0
TODO                0
RAW EXIT            0           <- captured with $?, never through a pipe
ANTI-VACUITY        864 assertion lines, 0 'not ok'
```

**Status: carried, not re-derived by this author.** Reproduced independently
twice — two detached worktrees, two different hours, four matching numbers, one
of them taken without reading the other first. This document does not re-run it
because the machine is at 100% disk and creating a worktree currently fails
part-way and silently, which is the exact condition that manufactures a
plausible wrong number.

### Rust suite

```
REVISION            37d0d72e
RUNNER              cargo test
PASS                264
FAIL                0
IGNORED             4
RAW EXIT            0
```

**Status: carried.** Note the structural limit honestly: `cargo test` prints one
`test result:` line per test binary and never sums them, so any total is an
addition someone performed. Without `--no-fail-fast` the denominator shrinks
precisely when something is wrong.

**Coverage gap, stated plainly:** the Rust tests are executed by their author,
not by the gate, and no JavaScript test can reach the scheduler. That is a
coverage gap, not an unexecuted claim.

### How to verify

Use the canonical runner. Do not hand-roll a pipeline; every hand-rolled one in
this project's history laundered an exit code.

```
# from the repository root -- this path is not cwd-independent
./examples/serving-dashboard/run-tests.sh
```

The runner prints its own container — working directory, branch, head revision,
uncommitted file count, node version, and discovered file count — **above** the
results, so a measurement cannot be quoted without the tree it came from.

Name the revision explicitly via `REVIEW_SHA` or `SHIPPING_TREE_REF`. Never a
tag name.

---

## 4. What we know we do not know

Stated without softening. Every item here is re-runnable at `1e809173`.

### The vocabulary in the types and the vocabulary in the code disagree

The render-state union is documented as
`measured | pending | stale | unavailable | not-applicable`.

```
production sites emitting  state: 'ok'        6      <- all in the store adapter
production sites emitting  state: 'measured'  0      <- [positive control]
```

`'ok'` is a retired token. It still works because the state map carries
`ok -> OK` **and** `measured -> OK` as aliases, so both spellings resolve and
nothing ever fails. **The type says one thing, the code does another, and a
compatibility alias guarantees no test will ever notice.** This is the whole
disease of this project in six lines of a real file.

### The static asset surface is guarded by a MIME allowlist doing secrecy work

```rust
const SERVABLE_EXTENSIONS: [&str; 9] = [
    "html", "js", "mjs", "css", "json", "svg", "png", "ico", "woff2",
];
```

The demo asset directory is a **source tree**, not a build output. Files in it
are refused by *extension*, not by a decision about whether they should be
public. Copying a refused file to a servable extension serves it — that has been
demonstrated by execution, not argued.

The allowlist was never designed for secrecy, is not reviewed as if it were, and
nothing tells a future contributor that naming a file `.json` publishes it. The
source comment already names the class for a different instance: *a refusal by
coincidence*.

### The percent-encoding bypass

The asset guard reads the raw request path and refuses any path segment
beginning with a dot. The static file service downstream then decodes.

```
GET /demo/%2Evscode/settings.json
  segments seen by the guard:  "%2Evscode"  "settings.json"
    leading-dot rule ....... neither segment starts with '.'  -> PASSES
    extension allowlist .... final segment is .json           -> PASSES
  downstream decode -> .vscode/settings.json -> served
```

```
percent/decode/urlencode tokens in the asset guard ....... 0
[controls] raw-path reads 3 · leading-dot guard 1 · allowlist 2   (all fire)
```

The two rules do not compose: the one that fails closed is evaluated on a
different segment from the one that fails open. **This is live at `1e809173`.**
It is not exploitable on a machine with no dotted directory in the dashboard
folder — today there are none — and that is luck, which is the thing the guard
exists to stop relying on.

The fix is a rejection of the character before any decode, not a second decoder.
A second decoder must agree with the first one forever, and that is a
maintenance obligation nobody signed.

### The path disclosure, wire half

Closed in the shipped source: the model directory does not leave the process.
Live only in already-running processes started before the fix. Loopback-bound.
Remedied by restart, never by a commit — **a source read cannot refute a running
process, and a source edit cannot remedy one either.**

The launcher rebuilds only when the binary is *absent*, never when it is
*stale*, so a rebuild is not automatic.

### Session artifacts ship in the diff

QA evidence files under `examples/qa-evidence/` are tracked at this revision and
contain a reference to the development harness's session path. **They are not
reachable over HTTP** — that directory is a sibling of the served root, verified
by request including a traversal attempt — so this is a repository-contents
issue, not a disclosure. It is disclosed here rather than fixed quietly.

### Absence states are separated by 5–7 of 255

```
--og-unavail-fg  #72869d      the four absence states are legible individually
--og-pending-fg  #788ca2      (each meets WCAG AA 1.4.3 for contrast against
--og-stale-fg    #7f91a6       the raised background) and are near-identical to
--og-na-fg       #8597ab       each other -- adjacent pairs differ by 5-7/255
```

Contrast against the *background* was verified. Contrast between the *states*
was not, and it is the one that matters for telling them apart. The design
compensates with distinct glyphs, so the information is not colour-dependent —
but a reader scanning for "which of these is stale" is working from the glyph
alone, and that was never a deliberate decision.

### The path-ban guard's predicate is wrong in both directions

One test inspects field *values* — not field names — and asserts none of them
looks like a filesystem path. That design is right: renaming a row must not
satisfy the ban. The predicate is not.

```js
.filter(([, field]) => typeof field.value === 'string' && field.value.includes('/'))
```

```
Qwen/Qwen2.5-0.5B-Instruct   a legal model id     -> CONTAINS '/' -> BANNED    (false positive)
C:\Users\someone\models      a Windows path       -> NO '/'      -> PERMITTED (false negative)
```

Both directions are provable from the predicate alone; neither needs a run. A
related ban elsewhere keys on a `/Users/`-shaped string, which is blind on every
Linux box and CI container this would ever run in.

**A count of files that mention a property is not a count of guards over it.**
Four files reference this property and two of them are comments; one is a
styling fixture holding a relative path on purpose. The real guard is one file,
and its predicate has a proven false negative.

### Specification debt

The specification is 2,857 lines carrying 218 acceptance criteria. **9 of its
115 cited source paths do not resolve at this revision.** They are published
unfixed and are not exempted from the citation guard. An exemption granted to
the document that defines the quality bar is the worst possible place to grant
one.

The specification carries a document-level header stating that it is a
historical record, that its imperatives are not live orders, and that its line
citations are valid only at the revision they were taken at. It needs that
header because instructions inside it generated three separate incorrect actions
by three different readers, none of whom misremembered — all three read it.

**The citation guard's corpus is 1 of 4 review documents.** A guard reading a
quarter of its corpus is a guard that passes by sampling.

`AC196`, `AC199`, `AC200` and `AC201` are named, mechanisable, and unwritten.

### A label can be accurate and still misinform

The name is not being changed in this PR, and that is a decision rather than an
omission:

```
occurrences of the arm's directory name, tree-wide ....... 34
of those, inside recorded JSON captures .................. 2
[negative control, freshly generated] .................... 0
```

Renaming would edit two committed capture files whose entire evidentiary value
is that they record what a real run emitted. **We would cure a comprehension
defect by manufacturing a provenance one.** The remedy chosen instead is a
caption at the point of display, where the reader actually is.


The two model directories are named for their attention implementation. One of
them is, as a consequence, the arm that structurally cannot batch — that is the
whole point of the comparison above.

Nothing in the name says so. Several people on this project benchmarked the
non-batching arm expecting batching numbers, and every instrument agreed with
them: the label was accurate, the server was healthy, the requests succeeded,
and the results were meaningless for the question being asked.

**No test can catch this.** A name that is true is indistinguishable from a name
that is informative to every checker in this repository. The only defence is a
reader who knows what the arm is for, which is precisely the defence that does
not scale — and it is why the refusal reason is now printed in the response body
rather than left for the operator to infer from a number.

### Attribution

This repository cannot tell you who wrote any of this. All contributors shared
one git identity, so authorship is not establishable from the history — only
corroborated by timing and file ownership. Any credit or blame in the commit log
is a guess with a cryptographic hash attached to it.

---

## 5. Method — the transferable part

These are not aphorisms. Each one was paid for.

**A completion is a re-runnable predicate over its subject plus a positive
control — never a narrative.** "I fixed it" is not checkable. "This command
returns 0, and this control returns non-zero" is.

**A predicate must key on what the defect does, never on what it is called.**
Vocabulary belongs to the domain, not to the failure. A detector keyed on the
word `refus` fired on a passing test whose subject was refusals.

**Substring containment standing in for equality is a defect until proven
otherwise.** `includes`, `startsWith` and `endsWith` in a guard each shipped a
bypass here. When you mutate a guard to test it, mutate the *direction* as well
as the value — every fixture author reached for the same geometry, so twelve
tests confirmed one shape.

**A mutation that does not apply is byte-identical to a test that does not
fire.** Prove the mutation landed — print the diffstat — before believing the
red.

**A negative result has two causes and one symbol.** *It is not there* and *I
could not see it* are byte-identical in every instrument here: an exit code of
0, an empty grep count, a false ancestry check. We check alarms and we do not
check reassurances, so the reassuring symbol is the one that never gets a second
command.

**Presence is the one property every wrong answer also has.** A stale value, a
misattributed value, and a correct value are all present.

**A move is an absence at the source and a presence at the destination.**
Verifying a deletion by the path it used to occupy confirms the move, not the
removal. Locate by basename with no path filter before trusting any zero.

**Publish the instrument, not the reading.** A result whose command is not shown
cannot be refuted, and everything here that was overturned was overturned in
seconds because its author printed the command that produced it.

**And the law that outranks the rest:**

> **Almost nothing here was ever false. Every expensive error was a true
> statement that outlived its tree.**

We built many controls against falsehood and none against staleness. The missing
field was never rigour — it was an expiry date. That is why every claim in this
document carries the revision it was measured at, and why the ones this author
could not re-derive are labelled *carried* rather than quietly adopted.
