# Serving dashboard demo: an honest view of a live inference server

> **MEASURED-AT: 37d0d72e**
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

**Revision ruled as the review pin. Status: re-derived by this author, at that
revision, with the canonical command.** The earlier version of this paragraph
said *carried, not re-derived*, and gave a reason: the machine was at 100% disk,
and a worktree that fails part-way and silently is the exact condition that
manufactures a plausible wrong number. That reason expired when someone reclaimed
8.2 GB. The figures above are now execution, not testimony.

```
VEHICLE   git worktree add --detach 37d0d72e   (a commit has no untracked files,
                                                so the hazard below is unreachable)
CORPUS    node --test $(git ls-files '*.test.js')
          56 files enumerated by git, not walked on disk
          [vacuity guard] corpus > 0 asserted before running
          [neg control]   git ls-files '*.zzq9x' -> 0
RESULT    tests 749 · suites 115 · pass 749 · fail 0 · RAW EXIT 0
```

**Why the corpus is enumerated by `git` and not by the runner.** A bare
`node --test` walks the directory, so it executes whatever is lying there —
including files no one has committed. That defect landed four separate times on
this branch and defeated two purpose-built canaries, because the countermeasure
was *remember to check for untracked files* and remembering failed every time.
Letting `git ls-files` name the corpus makes it unreachable instead of unlikely.
**Every suite total in this document's history that was produced by a bare walk
is withdrawn**, including ones that happened to be correct: a number that could
not have detected the defect is not evidence merely because it escaped it.

**And the instrument that nearly cost me this measurement.** My first read of the
run reported a clean exit and an empty summary — I had grepped for `# tests`,
which is TAP, and this runner emits `ℹ tests`. **The exit code was 0 and the
summary was blank, which is indistinguishable from a suite that never loaded.**
Had I published on the exit code I would have published a true number supported
by a run I had not confirmed happened. A pass count cannot be produced by a file
failing to import; an exit code can.

### What this file's freshness stamp does and does not mean

`MEASURED-AT` names the revision **every figure below was re-derived at**. It does
not claim this file existed there — it did not; a description of a reviewed
revision is necessarily written after it. The distinction matters because the two
readings are indistinguishable in the marker itself, and only one of them is true.

### This file cannot be read at a review pin, and that is not a pin-choice problem

A reviewer checked whether the tree the code is scored at also contains the
*reviews*, and found a third of their own review missing from it. The same
question asked of this file returns a worse answer, because this file is the
pull request:

```
PIN CANDIDATE   CUT AT     THIS FILE THERE   MY LAST COMMIT IN IT?
1f9fc70b        05:02:09      256 lines      NO   (reverse: YES — strictly newer)
42c15622        05:17:24      256 lines      NO   (reverse: YES)
37d0d72e        05:48:55      402 lines      NO   (reverse: YES)
HEAD                          684 lines      —
[CONTROL] demo-spec.md at the same four: 2284 · 2414 · 2457 · 2857  — varies, so
          the reader is reading four different trees and not one tree four times
[NEG CTL] a path that never existed: fatal, not a quiet zero
```

**62% of this document does not exist at the cut review pin, and every one of its
31 sections is absent there.** That includes both suite blocks, all fourteen
known gaps, and this sentence.

**The structural point, which no choice of pin fixes.** A description of a merge
is written after the thing it describes. It is therefore *always* newer than any
revision it could be scored at, and the gap is widest at exactly the moment the
work is most complete. Pinning is the correct discipline for code, for fixtures,
and for the reviews — all of them are claims *about* a tree, and reading them at
that tree is what makes them checkable. **This file is a claim about the
difference between two trees, so there is no single revision at which it is
true.** Freezing it to one is not conservative; it silently substitutes a draft.

**So the rule this document asks for is the opposite of the rule everything else
gets: read it at the tip, and never from a tag.** If that is inconvenient, the
inconvenience is the honest shape of the artifact and not a defect to be
engineered away.

**And this retires an ambiguity in this file's own stamp rather than merely
noting it.** The section below observes that `MEASURED-AT` cannot mean *this file
existed at this revision*. The magnitude is now measured: at the stamped
revision this file is 402 lines, and it is 684 here. **282 lines of the document
carrying the stamp postdate the stamp.** The figures were re-derived at that
revision and the marker is honest about them; it says nothing whatever about the
prose around them, and a reader has no way to see where one ends and the other
begins.

### The guard that enforces that marker cannot see this file

```
freshness-guard corpus predicate ......... /(REVIEW|BRIEF)/ && .md
matches 'PR-DESCRIPTION.md' ..............  0
matches 'READABILITY-REVIEW.md' ..........  1   <- control, fires
documents in the corpus / documents present ... 4 of 15
positional citations in the 11 it cannot open .. 598
```

This file adopted the marker the guard enforces and receives no protection from
it, and nothing anywhere reports that. **An exemption ledger cannot record what
the corpus filter already excluded** — the skips it documents are the ones that
got past the filter, so a ledger reading empty means *no exemptions among the
files I chose to open*. The largest document in the tree complied and was
excluded for its name.

The fix is one predicate: discover by the marker, not by the filename. Any
document containing the marker is in scope, so a document opts in by making a
dateable claim rather than by being christened correctly. **A corpus selected by
filename encodes what documents happened to be called on the day the guard was
written.**

### A dated document that cites an undated one is not dated

It is a fresh wrapper around an unknown. *(Project Lead)*

This is why the freshness stamp on this file names the shipping revision rather
than the revision its author happened to be sitting on. A marker pointing at the
author's desk certifies the author, not the tree. Every figure in this document
has been re-derived at the stamped revision, and one of them changed when it was:
a count taken on a working desk read 112 where the shipping tree holds 98. Both
readings were honest and only one describes what a reader will check out.

### Where the variance lives

We do not claim this suite is deterministic. We claim something more specific
and more useful, because it tells a maintainer what to do rather than only what
to fear.

```
SHARED CHECKOUT     non-reproducible results observed by three authors
DETACHED WORKTREE   flakes observed by anyone, all session ................ 0
```

Every non-reproducible result this session occurred in a shared checkout where
many processes commit and save concurrently. Two mechanisms were identified: a
test that resolves `HEAD` twice can straddle another commit landing between the
two reads, and a test that reads a file can catch it mid-save. **Score the gate
only in a detached worktree at a pinned revision.**

The honest limit: the detached-run denominator is small — a handful of runs
across three authors, not the many dozens the shared-tree figure rests on. This
says where the variance lives. It does not say variance is absent.

One caveat that matters more at the end of a quiet period than at the start:
when concurrent editing stops, the shared checkout starts returning green too.
**That green means the writing stopped, not that the race was fixed** — and it
arrives exactly when a reader is most inclined to believe it.

### Rust suite

```
REVISION            37d0d72e
RUNNER              cargo test
PASS                264
FAIL                0
IGNORED             4
RAW EXIT            0
```

**Revision ruled as the review pin. Status: carried, and reproduced three
times independently across six binaries.** Note the structural limit honestly: `cargo test` prints one
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

### RETRACTED — the vocabulary claim below was mine, and it was wrong

**I compared code written against one vocabulary to the union of a different
one.** There are two state vocabularies in this dashboard, and only one of them
retired `'ok'`:

```
FIELD states   telemetry-field.js  measured|pending|stale|unavailable|not-applicable
                                   'ok' RETIRED — and it was a real landmine:
                                   the constant once read MEASURED: 'ok'
SERIES states  store-adapter.js:38 @property {'ok'|'unavailable'} state
                                   'ok' DECLARED, CURRENT, CORRECT
```

```
state: 'ok' in store-adapter.js .................. 5   <- I published 6
state: 'ok' anywhere else in shipped non-test JS .. 0
[POS CTL] any state: '…' in that file ............ 11  <- the grep can count
[NEG CTL] state: 'zzq' ...........................  0  <- and can still say no
```

**Every site I counted is in the file that declares the union it satisfies.**
The code and its type agree. The sentence below — *the type says one thing, the
code does another* — was produced by measuring real occurrences correctly and
checking them against the wrong specification, which is this document's most
frequent failure and now its author's fourth instance. **My count was also wrong
by one in the direction that made the finding sound larger.**

**Why this is retracted in place rather than deleted.** A false gap in a document
whose subject is honesty is worse than the same error anywhere else, and deleting
it would leave the strongest evidence for that claim invisible. **The last
surviving copy of a withdrawn figure is usually inside its own retraction**, so a
retraction written in the language of the claim is indistinguishable from the
claim — four people hit that property in four files tonight. The paragraph below
is kept, marked, and must not be quoted as a finding.

**What survives, and belongs to someone else.** A reviewer found the real
divergence in the neighbouring vocabulary: the ratified source classes include
`simulated`, the canonical enum is short that member, and the badge lookup falls
back to `derived` — so an unknown class renders as *arithmetic on measured
inputs*, which is a stronger claim than the truth. **That is a live finding, it
is theirs, and it is not the one written below.**

### ~~The vocabulary in the types and the vocabulary in the code disagree~~ (RETRACTED — see above)

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

### Two guards in this directory disagree about that allowlist, and the count in the failing one is not the exposure

The exposure ratchet is the only failing test on this branch. Its message is
exact, and its class breakdown is where the interesting part is:

```
94 tracked files are fetchable at /demo/ that the page never loads (was 91).
BY CLASS: TEST 64 · DESIGN 3 · INTERNAL_DOC 14 · TOOLING 10 · FIXTURE 3
```

**`INTERNAL_DOC` is `/\.md$/`, and the server refuses `.md`.** Those fourteen
files are counted as fetchable by a guard that sits four directories from a
guard proving they are not. Both are committed, both run in the same suite, and
the suite is green on the pair.

The ratchet is not stale by accident — it records its own provenance, and the
record is what dates it: *111 of 111 served, 0 refused … there is no allowlist,
no extension check*. **That was true when it was written.** The allowlist landed
afterwards. The comment is not a stale opinion, it is a **stale execution
record**, which is the most credible kind of wrong thing in a repository,
because it carries a measurement.

**And the ratchet is still right, for a reason its own number does not express.**
`.md` is refused by extension, so renaming one of those fourteen to `.json`
serves it — demonstrated above. Its bytes are also in the repository regardless
of what any route does, so a `git mv` out of the served directory turns this
guard green without making a single byte private. **It counts the fetchable set;
the exposure is the tracked set.** Those two are different, and only one of them
has a test.

So the number is not the exposure, in both directions at once: it counts
fourteen files that are not currently served, and it stops counting a file the
moment it moves — while the bytes stay exactly as public as they were.

**The reason this is a gap and not a fix.** The honest repair is to give the
ratchet the server's own allowlist rather than a private guess at it, so the two
cannot drift again. That is an edit to a guard this author does not own, at the
end of a freeze, and a wrong patch handed to an owner is worse than no patch —
it arrives carrying the sender's authority. This author has already made that
exact mistake once tonight, on a different file: a diagnosis that was correct
and a remedy that was prose where the defect was a build instruction. It was
caught by a third party, not by the sender or the forwarder. **A handed-over fix
is a claim, and it needs a control like any other claim.** The census above is
the control this one has; the patch is left to the owner.

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
different segment from the one that fails open.

**This is no longer a source argument. It was proven on the wire, from a binary
built at the scored pin**, by the reviewer who had earlier withdrawn the same
claim for lack of provenance and then went and produced it properly:

```
[POS CTL] /demo/index.html ................. 200   the server can serve
[NEG CTL] /demo/nonexistent.json ........... 404   it refuses properly
BASELINE  /demo/.secretdir/settings.json ... 404   THE DOTFILE RULE FIRES
C19       /demo/%2Esecretdir/settings.json . 200   BYPASSED — canary body returned
```

**And this retires the hedge that used to end this section.** It read: *not
exploitable on a machine with no dotted directory in the dashboard folder — today
there are none.* That was true and it was worthless, because the condition it
rested on is one `mkdir` away and belongs to the operator, not to us. **A defence
whose premise is the current contents of a directory is not a defence.**

**The instrument note, because a second reviewer probed the same thing and got a
clean 404 that meant nothing.** They requested an encoded path to a file that did
not exist. A 404 is then correct regardless of whether authorisation was
bypassed, so the probe could not distinguish the two outcomes it appeared to
decide. **They voided it themselves rather than reporting it.** The proof that
counts used a real file and confirmed the byte count on the wire — *a negative
result against a subject that does not exist is the vacuous-guard defect wearing
an HTTP status code.*

### The verdict "blocking set empty" rests on a default, not on a property of the code

The reviewer holding the security lane published the exact condition that would
flip their verdict to blocking, so that it could be run rather than interpreted:
*if the demo ever binds anything but loopback, I block* — because a fault-induced
500 then hands the presenter's filesystem layout to every device on the network,
and demos fail in public by nature.

**I ran their predicate. It passes, and what it tests is a default:**

```
run-demo.sh:29   BIND_HOST="${BIND_HOST:-127.0.0.1}"
  identical at the scored pin and at HEAD
  BIND_HOST mentions in that file ....... 6
  lines validating its value ............ 0
  '0.0.0.0' in any tracked .sh .......... 0    [POS CTL] '127.0.0.1' in 5 files
                                               [NEG CTL] fabricated token in 0
  guards asserting the bind is loopback . ~~0~~  **FALSE — SEE BELOW. THE REAL
                                          ANSWER IS 1, AND IT MAKES THE POINT
                                          BETTER THAN MY ZERO DID.**
```

**I published that zero and it was wrong.** `check-launch-command.test.js:425`
does assert it:

```js
test('both servers bind loopback by default', () => {
  assert.match(runDemoCode, /BIND_HOST:-127\.0\.0\.1/,
    'run-demo.sh must bind loopback by default');
});
```

**How I got it wrong is the part worth keeping.** I ran a search for test files
mentioning `BIND_HOST` or `--addr`. It returned four files. **I then reported how
many of them *assert* the property without opening any of them.** The search
answered *which files mention the token*; I published an answer to *which files
assert the bound*. **Same instrument, adjacent question, and the arithmetic in
between was flawless.** That is the third time tonight I have caught this exact
substitution in my own work, and it has never once looked like a mistake while I
was making it.

**And the guard I missed is better evidence for my finding than the absence I
claimed.** It reads `runDemoCode` — the *source text* of the shell script — and
asserts the literal string `BIND_HOST:-127.0.0.1`. An environment override
changes the value at runtime and leaves that text untouched. **So the guard is
green in every world where the demo is serving on `0.0.0.0`.** It is not weak;
it is measuring the only thing a source-level test can measure.

**Its own title carries the limitation: `bind loopback by default`.** The author
named the gap precisely and put it in the test description. **What decays is the
reading**: a green scrolls past as *binds loopback*, and the two words doing all
the work are the ones a passing test stops showing you. **An honest test name is
not a guard — nobody reads the name of a test that passes.**

So the corrected finding is narrower and firmer than the one I published: **the
default is asserted, the override is unconstrained, and no guard can see the
difference from the source alone.**

**`:-` is a default, not a constraint.** `BIND_HOST=0.0.0.0` reaches the blocking
condition without editing one tracked byte, without a code review, and without a
warning anywhere in the run. **So the empty blocking set is true, and it is a
statement about how the demo is usually started rather than about what the demo
permits.** The honest form of the verdict names the assumption: *no blocking
issues, given a loopback bind that nothing enforces.*

**This is the same shape as the allowlist above and the ratchet earlier.** In all
three, a real safety property is delivered by something that was chosen for a
different reason — a MIME list, a file layout, a shell default — and each holds
until the day someone changes it for that different reason. **Safe by luck is not
a criticism of anyone's work here; it is a description of where the guard is
missing, and in this case the guard is one assertion.**

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

### The server-side model path field was deleted, not redacted — do not re-add one

Stated here because this document is where someone will land, and because the
weaker design is written in the present tense in an internal review that has not
been corrected yet.

```
fn model_path_for_display   -> 0   [control] fn resolve_model -> 6
file_name() in admin.rs     -> 0
```

An earlier design returned the path's last segment instead of the whole path.
**That was tried and rejected, and the code says why in its own comment: a
basename is the final segment of an operator-chosen path, so its contents are
unbounded — safe on this machine by luck, not by construction.** The field was
removed outright, which is strictly stronger.

The hazard worth naming is not the stale sentence, it is its grammatical mood.
**A stale fact gets corrected; a stale instruction gets executed.** Prose that
describes a defensive mechanism in the present tense reads as a specification to
restore it. I read exactly such a line earlier and proposed putting a basename
back; the engineer who owned the file refused, correctly. The next reader might
not, so the refusal belongs here rather than in a conversation.

### The model path can still reach a client through the error channel

The dashboard no longer renders the model path, and the server no longer sends
it in the models response. A third route is open, and it is the one nobody
looks at, because it is not a display surface at all — it is a 500 body.

```
registry.rs:283, :430   .with_context(|| format!("failed to load model '{}' from '{}'",
                                                 spec.id, spec.path.display()))
routes/mod.rs:750       .map_err(|err| ApiError::internal(
routes/admin.rs:505         format!("failed to load model '{id}': {err}")))
```

`with_context` makes the path string the *outermost* context, and plain `{err}`
prints exactly that. **`{err:#}` would be the loud mistake; `{err}` looks
restrained and discloses anyway.** Request a configured-but-unloadable model and
the error body carries an absolute filesystem path.

This is read from source and **has not been put on a wire**. One command settles
it against a running server:

```
curl -s localhost:$PORT/v1/models/<configured-but-broken-id> | grep -c /Users
```

The correct pattern already exists 41 lines above the leak, in the same file:
`map_registry_error` logs the detail for the operator and returns a **static**
string to the client. It *replaces* rather than *wraps*. Two sibling functions,
one error family, opposite disclosure policies, and the reachable path is the
leaky one.

The structural fix is a signature, not a sweep: make `internal` take
`&'static str` so it cannot interpolate at all, and leave `bad_request` dynamic.
The split is principled — `bad_request` describes the caller's own input back to
them and discloses nothing they did not send; `internal` describes our state,
which is entirely new information to a stranger. **That makes the class
unrepresentable and lets the compiler catch the site nobody reviewed.**

### Our prose is not searchable by phrase

```
lines ending in a string-concatenation join, telemetry-provenance.js ..... 98
file length ............................................................ 1044
[negative control, freshly generated token] ............................. 0
```

Long explanations hit the line-length limit and get split across string
concatenations, so a phrase that reads perfectly on screen returns zero to a
search. **The density is highest in the file whose entire job is human-readable
explanations of why we trust each number** — the defect is proportional to the
virtue. Template literals keep the sentence contiguous at identical runtime cost.

This one is worth stating plainly because it silently weakens every audit
performed against this tree: a phrase-search zero here is not evidence of
absence. Token searches are unaffected — a single token never spans a join.

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
