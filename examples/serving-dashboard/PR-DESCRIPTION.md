# Serving dashboard demo: an honest view of a live inference server

MEASURED-AT: 37d0d72e

> The line above is deliberately undecorated: bare lowercase hex, column 1,
> nothing before or after it. It carried blockquote and bold markup until the
> convention's author published the grammar, at which point I ran their parser
> against my own file. The capture group takes `\S+`, so the decorated form
> yielded `37d0d72e**`, and `git cat-file -t` answers
> `fatal: Not a valid object name`. **My freshness stamp was unreadable by the
> tool that checks freshness stamps, and it was unreadable because I had made it
> look important.** The most emphatic form of a machine-readable field is the one
> the machine cannot read.
>
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

## What ships

**The headline is a count, not a ratio: four requests served concurrently by one
model instance, against one at a time.** A count has no noise floor. You can see
it happen, and you can re-run it.

**The `2.46x` aggregate speedup is withdrawn and stays withdrawn.** An
interleaved A/B/A/B run against a zero effect produced `+58.41%` in the worst
case with a coefficient of variation of `19.83%`, against a `1.98%` clean
baseline. **That rules out the precision, not the existence.** Roughly `1.6x` to
`3.9x` survives the measurement; `2.46x` does not, and a demo that shows two
decimal places is claiming a resolution this harness cannot deliver.

**Two admissions stay on the page, because removing a number is not the same as
removing a disclosure:**

- **Batching does not make any single request faster.** The throughput gain is
  aggregate, and any individual request is slower under load than it would be
  alone.
- **Each stream runs at roughly `0.62x` the speed it achieves by itself.** That
  is the price of the aggregate win, and it belongs beside the aggregate win.

**A big number with a small caveat is still the lie, told more quietly.** The
per-stream cost is not a footnote to the speedup; it is the other half of the
same measurement, and the panel never renders one without the other.

## Open, and unowned

**Nothing in this section is fixed. It is here because a release note that omits
its own open set is a marketing document.**

- **16 `latency.*` keys are unclassified**, deliberately. The reviewer who owned
  that pass declined to label them and gave the reason in nine words:
  **"I did not establish that, so I am not calling it anything."** That is the
  correct answer and it is the project's character in one sentence — the
  alternative was a confident `NOT_PLUMBED` that would have been right for
  fourteen of them and a fabrication for the fifteenth.
- **One latency field was measured, certified, and never wired to the panel that
  promises latency.** Found while overturning a ruling of mine that rested on a
  regex which could not match the key it was looking for.
- **The exposure ratchet is red**, at 91 against an actual 94, and it was left
  red on purpose rather than raised to green by an author who noticed it.
- **A guard's corpus filter reads 4 of 15 documents**, so a document that
  complied fully with the freshness convention has never been opened by the tool
  that checks compliance. **An exemption ledger cannot record what the corpus
  filter already excluded.**
- **Two in-file duplicate symbols** are cited by name where the name is defined
  more than once in its own file, which a file-qualified symbol anchor cannot
  disambiguate.

## The command channel had no guard

**This is recorded at the project lead's explicit instruction, in their words,
unsoftened, because the alternative is a release note in which only the code has
faults.**

Over one session the lead issued a regression as an order after reading a
past-tense fix comment as a live defect; raised a false index emergency that
invited precisely the destructive commands they had banned thirty minutes
earlier, while an untracked test sat in zero commits; dispatched a one-hour-stale
test count inside the order responding to a staleness finding; and sent two
reviewers to a retired tag.

**Four agents refused orders with evidence. All four refusals were correct.**

> **Every deliverable on this branch has a guard, a reviewer, and a mutation
> test. The command channel has none — and it is the one artefact nobody is
> allowed to refuse on sight.**

An instruction arrives with the authority of the person sending it and none of
the verification we demand of a one-line code change. The routing protocol and
the standing rule that an order may be refused with evidence exist because of
that asymmetry, and they were written during the session that discovered it.

**The generalisation is not about any individual.** Orders are built from
measurements, measurements go stale in minutes at this commit rate, and **an
order is the only artefact in the system that carries a measurement forward
without carrying its clock.**

## Authorship, disclosed

**I wrote 14 of the acceptance criteria in the specification this demo is built
against.** I did not write the rest, and the credit for the specification is not
mine to take.

**And the denominator moved while I was quoting it.** I have published this
disclosure five times as *14 of 208*. Measured directly:

```
distinct ACs in demo-spec.md @ 37d0d72e (what ships) ... 213
distinct ACs in demo-spec.md @ HEAD .................... 218
the number I have been repeating ....................... 208
```

**All three were true when spoken.** The specification is append-only, so it grew
underneath a figure I had verified once and then carried. **The correct form is
`14 of 213 at 37d0d72e`** — a fraction with a revision attached, because without
one it is not a fraction, it is a fraction-shaped claim about an unspecified
document.

**This is the append-only tension arriving inside my own disclosure of good
faith**, which is where it was always most likely to land: the number I repeated
most often is the number I re-checked least, precisely because repeating it felt
like candour.

### A path you typed is a hypothesis; a path git printed is a fact

**The single most productive defect of the session was a missing directory
segment.** It hit at least five people in unrelated files, and it produced a
retracted security guard, four phantom deleted tests, a near-accusation, and
several near-misses including two of mine.

The shape never varies:

```
SEARCHED   examples/serving-dashboard/dashboard/never-bind.test.js  ->  0
ACTUAL     examples/serving-dashboard/never-bind.test.js            ->  1
```

**`dashboard/` is a real directory, so the path resolves as far as its parent and
only the leaf is wrong.** Nothing errors. The zero is clean, immediate, and
indistinguishable from a genuine absence — and it arrives with all the confidence
of a search that ran correctly, because it did.

> **Resolve the path, never type it:
> `git ls-tree -r --name-only <sha> -- <dir> | grep <basename>`.
> A path you typed is a hypothesis. A path git printed is a fact.**

**This subsumes a defect recorded earlier in this document from the other
direction.** I probed a filename that did not exist and never saw
`fatal: path … does not exist` because I had suppressed stderr for tidiness. **One
of us lost the error message; the others never generated one.** Both produce a
zero that no amount of care distinguishes from an answer, and **only one habit
prevents both: let the tool produce the path, and read everything it says.**

**A live case, and the reason this section exists rather than being a footnote.**
A ban-list guard for values that must never reach a screen was declared missing
and its citations were ordered struck — from a security review, an implementation
review, and this document. **The file exists.** Searched by resolved path it is
one line; the guard runs green; the identifier appears in 13 tracked files at the
revision I checked and 11 at the earlier one another reviewer checked. **Both
counts are correct and the tree grew between them.**

**Its citation stands here, verified by resolved path and not by relay.** Had the
strike been executed, three documents would have deleted true statements about a
live security control, and the deletion would have looked like diligence in every
one of them.

### KNOWN GAP — C19, percent-encoded dot segments, unfixed and shipping

**This is a named, unfixed defect in the code this PR ships.** It is stated here
because it is real, because it is loopback-only, and because shipping it silently
would be the exact failure this document spends 1,400 lines describing.

```
CONFIRMED LIVE ON ALL FOUR DEMO ORIGINS, 06:39:24, by the agent who reproduced it
  /demo/…/.vite/…/results.json      404   <- the dotfile rule works
  /demo/…/%2Evite/…/results.json    200   <- 88 bytes, byte-identical to disk
  /demo/…/%2evite/…/results.json    200   <- lowercase too
  [POS CTL] /demo/app.js            200   <- the asset layer is alive
  [NEG CTL] /demo/zzz-no-such.js    404   <- a missing file really does 404,
                                             so the decoded 404 is a REFUSAL,
                                             not an absence
```

**The guard compares raw bytes; `ServeDir` decodes first.** A segment spelled
`%2Evite` does not begin with `.`, so the dotfile rule passes it, and the server
then decodes it to `.vite` and serves the file. **Remedy, proposed by the reviewer
who found it: refuse any `/demo/` path containing `%` at all — 91 servable assets,
none of which need encoding.** That removes the differential rather than policing
two parsers, so it cannot drift back.

**And it composes with a second gap, which is the first time two findings on this
branch multiply rather than merely coexist.** Measured directly:

```
UNDER THE SERVED ROOT
  tracked by git ......... 125
  present on disk ........ 126        <- the difference is ONE file
  [POS CTL] index.html tracked: 1

THAT FILE
  node_modules/.vite/vitest/…/results.json
  git check-ignore ....... IGNORED     <- invisible to `git status`
  tracked at HEAD ........ NO          <- invisible to `ls-tree` and `ls-files`
  extension .............. json        <- ON the nine-entry servable allowlist
```

**Three independent barriers — the tracked-file corpus, `.gitignore`, and the
dotfile rule — and the union of their blind spots is not empty.** Each looks like
defence in depth on its own.

> **`ServeDir` walks the filesystem. Every guard we own walks git. The blind spot
> is exactly `.gitignore`, which is a list of things we decided not to look at.**

**The file is a build-tool cache, and it is not stable.** It held 88 bytes when it
was first measured and **266 bytes when this section was written** — the same file,
the same path, within the hour. **Nobody predicted that; it was observed while
verifying the claim.** The argument for treating it as a real exposure was
originally *nothing structural keeps it empty*; the file then changed size on its
own, which converts that prediction into a measurement.

**The correct repair is one word wider than the obvious one.** A guard that scans
every *tracked* file under the served root would not see this file. **The corpus
has to be a disk walk of the served root, because that is the corpus the server
uses** — a guard whose corpus differs from the server's has a blind spot exactly
the size of the difference, and nobody measures that difference, because both
numbers look correct alone.

**Attribution, since none of this is mine:** found by the reviewer holding the
security lane, reproduced on production origins by the secretary, and the corpus
composition measured by the security reviewer. **The 125/126 gap, the ignore
status, and the byte-size change above are the only parts I measured myself.**

### The bypass does not reach this document, and here is why it cannot

**The percent-encoding bypass above is real and it is unfixed. It is also
narrower than the section preceding this one might leave a reader believing**, and
the reviewer who found it published that bound against their own finding rather
than letting it stand at its most impressive width.

**The reach:** dotfiles and dot-directories whose contents carry a *servable*
extension — `.vscode/settings.json`, `.secret.json`. That is genuine, and the
wire proof used `.json` for exactly that reason.

**What it does not reach: the extension allowlist, and therefore not one of the
14 markdown documents on this branch — including this one.** Verified against the
source rather than relayed:

```
demo_assets.rs:188   let Some((_, extension)) = name.rsplit_once('.') else {
demo_assets.rs:189       return false;                      <- FAILS CLOSED
demo_assets.rs:191   SERVABLE_EXTENSIONS.contains(extension.to_ascii_lowercase())

"md" in SERVABLE_EXTENSIONS   -> 0
"html" in SERVABLE_EXTENSIONS -> 1     [positive control: the probe reaches]
```

Every raw spelling that decodes to this file's name is refused, and the two
refusal paths are exhaustive:

```
PR-DESCRIPTION.md        ext = md      not in allowlist         REFUSED
PR-DESCRIPTION%2Emd      no literal dot -> rsplit_once -> None   REFUSED
PR-DESCRIPTION.m%64      ext = m%64    not in allowlist         REFUSED
PR-DESCRIPTION.%6Dd      ext = %6Dd    not in allowlist         REFUSED
%50R-DESCRIPTION.md      ext = md      not in allowlist         REFUSED
```

> **Encoding can only ever make a raw extension *more* mangled. It can never turn
> `md` into `js`. Either it destroys the literal dot and the `else` denies, or it
> corrupts the extension characters and the allowlist denies.**

**So an allowlist that fails closed turned out to be immune to a class of attack
its author never considered, and a denylist in the same position would have
fallen to the first `%2E`.** That is the whole argument for allowlists, and this
branch earned it empirically instead of asserting it.

**The structural finding is sharper than either the bypass or the bound.** Both
guards are `let-else`, in one function, 27 lines apart, written by one author:

```
:161  let Some(rest) = path.strip_prefix("/demo/") else {
:162      return true;      <- FAIL-OPEN.   cannot parse -> SERVE IT.   the bypass
:188  let Some((_, extension)) = name.rsplit_once('.') else {
:189      return false;     <- FAIL-CLOSED. cannot parse -> REFUSE IT.  holds
```

**Same idiom, same function, same author, opposite safety — and the author
demonstrably knew the fail-closed form, because they wrote it 27 lines later.**
The correct framing of the bypass is therefore not *add a percent check*. It is
**the default is allow and it should be deny**, and the file already contains its
own counter-example.

### A guard whose every failure mode is green

**The freshness guard that polices the review documents on this branch passes,
and its passing is close to uninformative.** Its corpus is discovered by filename
pattern on a non-recursive directory read, so it covers **4 of 8** citation-bearing
documents and **21 %** of the citations — an upper bound, since the denominator was
itself counted with a hand-written extension list. Its staleness boundary is a
SHA stored in a file, **210 commits behind the shipping pin**.

**It has three degradation paths — a document renamed out of the corpus, a
document moved into a subdirectory, and a boundary left behind. Every one of them
makes it greener. Not one of them can make it red.**

> **A guard whose every failure mode is green is not a guard. It is a logging
> statement with a checkmark.**

**And the boundary is the worse half, because it does not decay gradually.** The
assertion is *no document was measured before the boundary*. A boundary that ages
backwards becomes **more permissive with every commit**, blesses more stale
documents the older it gets, and never once turns red while doing it.

**This is not a coding error, and the author refuted the obvious fix in a comment
before anyone raised it:** a derived boundary would re-score every document on
every commit and enforce a line nobody chose. They were right; a declared
boundary is the better design. **But a declared value is a value with an owner,
and this one has none. We shipped the stored form without shipping the obligation
that makes it safe.**

> **Staleness is not an accident that befalls a correct value. It is the
> guaranteed end state of every value we store instead of derive.**

**That sentence indicts this document more than it indicts the guard.** The
acceptance-criteria count in it moved 208 → 213 → 218 while the specification sat
frozen — the artefact did not change, the number I had stored beside it was
simply never derived again. **The measurement marker at line 3 is a stored value.
So is the pin. So is every count in the table above.** Each is correct at the
moment it was written and each is on the same trajectory. **Eleven controls were
built on this branch against falsehood and none against staleness**, and the
distinction we kept failing to make is that a false value has an adversary while a
stale one merely has an author who has moved on.

### Corroboration must be verified at the source, not at the number

**An agent was credited with a test run they never performed.** Two independent
figures were reported to agree; the figure had one source. The credited agent had
been present the whole time and **nobody asked them whether they had run it.**

> **Before citing agreement, name the other party's command, revision, working
> tree and clock. If you cannot produce all four, you are citing your own number
> twice.**

**Two matching figures feel like independent confirmation and are the cheapest
false positive available.** An instrument that was never run agrees with
everybody, and the agreement here was manufactured by attribution rather than by
measurement.

**The asymmetry that keeps this class alive is worth stating, because it is the
mirror of a false accusation.** A false red has an adversary — its target checks
it. **A false credit has none.** The only person with standing to refuse it is the
recipient, and refusing reads as modesty rather than as a correction, so it gets
acknowledged and then re-quoted unchanged. One agent here had **three** findings
filed under their name tonight and had to decline each individually.

**Applied to this document:** every figure is either re-derived by me at a stated
revision, or attributed to one named measurer with their command and their
revision. **No number here is presented as corroborated by two parties**, because
in no case did I hold both commands.

### A third species of control: shape, not just liveness

Two kinds of control were used all session. One asks *is my instrument alive* —
run the matcher against something it must hit. One asks *am I pointed at the
right corpus* — assert the corpus is non-empty before counting in it. **A defect
in this document needed a third, and it cost a ruling.**

I searched for latency fields with the pattern `latency\.` and got zero. **The
zero was real.** I controlled it with the key `queue.`, and the control fired, so
the matcher was demonstrably alive. I then ruled on the result.

The field exists. It is `metrics.e2e_latency`, classified `MEASURED`, backed by
`onnx_genai_e2e_request_latency_seconds` in the server's metrics module.

```
metrics.e2e_latency   my pattern: MISS      shape-matched pattern: HIT
metrics.latency_ms    my pattern: MISS      shape-matched pattern: HIT
latency.p50           my pattern: HIT       shape-matched pattern: HIT
queue.depth  (my control)  ..... token in PREFIX position — a namespace
metrics.e2e_latency  (target)  .. token in LEAF position   — a suffix
```

**My pattern required `latency` to be a namespace. The field has `latency` as a
leaf.** And my control had the same shape as my pattern rather than the same
shape as my target — so **it could not fail in the way the target fails.** It
proved the matcher runs. It could not prove the matcher asks the right question.

> **A positive control must share the shape of the subject, not merely the
> liveness of the instrument.** A control drawn from the same assumption as the
> query confirms the assumption instead of testing it. Mine was live, correct,
> and pointed at a differently-shaped key — and every field of the report around
> it passed inspection.

**Two consequences were avoided by the author who overturned it, and both were
one step away.** Retitling the panel to *latency (not yet measured)* would have
printed a caption on the projector denying a value the system measures and
certifies — the honesty layer apologising for a number it has, which is the
inverse of the defect the honesty layer exists to prevent. And filing the field
with the batch's shared *not plumbed* reason would have been correct for fourteen
entries and a fabrication for the fifteenth. **A reason that is true of a group is
not thereby true of each member.**

**The coordinate they sent was 27 lines stale; the symbol name found it
immediately.** That is the fourth time a bare `path:NNN` rotted while the property
it named survived, and it is why the standard here is a quoted expected string
rather than a line number.

**Worth recording how it surfaced.** The author wrote that file, edited it that
session, and read it end-to-end four times without seeing the field. What found it
was a checklist entry marked UNRUN. **Familiarity with a file is not coverage of
it — an unrun check that is written down outranks a file you are sure you know.**

### The corruption that produced grammatical English

The single most alarming thing found this session was not in the server. It was
in the process of writing about the server.

An author wrote a commit message with `git commit -m "…"`. The message contained
a backtick — unavoidable, since the subject was code. In a double-quoted shell
string a backtick is command substitution. **Two things happened, and only one of
them is the scary one.**

First, the shell **executed the enclosed text as a command**, in a tree the crew
had agreed not to build in. That is bad and it is loud.

Second — and this is the one that belongs in a document about honesty — **the
substitution deleted the phrase from the sentence and left the sentence
standing.** The text was confessing a defect involving a pipe to `tail`. The name
of the defect was consumed, and what got committed was fluent English with the
subject silently removed. I reproduced the mechanism with a harmless command:

```
intended : the sentence confessing the | tail defect
survived : the sentence confessing the  defect
           -> no syntax error, no warning, no diagnostic. Just prose.
```

> **A corruption that throws a syntax error is a gift. That one produced a
> sentence.** Every other failure mode in this document announces itself as a
> wrong value, a red test, or a missing file. This one announces itself as
> *readable text*, and the only reader positioned to catch it is the author, who
> already knows what the sentence was supposed to say and will read it there.

**And it lands on me, which is why it is here rather than in a style guide.**
Every commit in this file used `git commit -F <msgfile>`, which is immune. **My
last six commit messages contain five backticks.** Five silent mutations did not
happen to a document whose entire purpose is being accurate about what was
measured.

**I did not adopt `-F` for safety. I adopted it because my messages are long and
multi-line.** So this document's integrity was protected by a formatting
preference — which is precisely the *safe by luck* pattern it complains about in
the MIME allowlist, the ratchet, and the loopback bind, now found in my own
tooling by someone else paying the bill for it.

### An append-only document is safest to write and most dangerous to freeze

This file, the spec, and every review document here are append-only: nobody
rewrites anyone's paragraph, so fourteen authors committed at high frequency all
session and **not one person's text was ever swept by another's**. It is the
reason the concurrency worked.

**It is also the reason a retraction can sit hundreds of lines and many minutes
away from the claim it kills.** Corrections land as new text, so a reader
encountering the original first gets the withdrawn version with no local signal
that it is dead — and if a snapshot is taken between the claim and its
retraction, the snapshot contains a confident falsehood and no trace of doubt.

> **The discipline that makes a document safe to write concurrently is the same
> discipline that makes it unsafe to freeze at an arbitrary moment.** These are
> not two policies in tension by accident; they are one property seen from the
> write side and the read side.

**The mitigation used here is to strike in place rather than append elsewhere**:
withdrawn claims stay where they were, struck through, with the correction beside
them. It costs more lines and it is the only form where finding the claim
guarantees finding its retraction.

### The two anchors this document is written against

**The code and the suites are pinned at `37d0d72e`. The review documents are
current to a later commit.** Those are different anchors and stating both is
cheaper than reconciling them later.

This matters concretely rather than pedantically: at least one reviewer confirmed
findings that are **live in the code at the pin** while the **write-up of those
findings is not in the pinned document**, because their review file grew 256 lines
after the cut. A reviewer scoring that arm at `37d0d72e` reads a review that does
not mention two things that are demonstrably there.

**That is not an argument to move the pin, and nobody involved made it one.** The
pin is the only revision carrying a dual denominator, which is worth more than any
amount of review prose. It is an argument for naming two anchors instead of
implying one — the same correction this document already makes about itself.

### Auditing the zeros, and the phantom the audit produced

A colleague drew the sharpest bound of the session on wrong-tree measurement:

> *It can only ever corrupt a zero. You cannot obtain 125 dashboard files from a
> tree that has none — a non-zero count is self-authenticating, a zero is not.
> So audit your zeros and leave your counts alone.*

**They also separated two species of control we had been conflating.** Every
control built here asks *is my instrument alive* — and in the wrong tree the
instrument is perfectly alive. The missing kind asks *am I pointed at the right
body*, and it is one line: assert the corpus is non-empty before counting in it.

**I ran it against this document. Five published zeros; corpus check first
(125 dashboard files, so: right tree); all five hold**, including one whose
positive control fires on the two files the guard was built for.

**And the audit manufactured a fresh false zero inside thirty seconds**, which is
the part worth recording:

```
git show <sha>:…/review-freshness.test.js       -> ZERO MATCHES
git show <sha>:…/check-review-freshness.test.js -> the file that exists
```

I probed a filename that does not exist. **The command printed
`fatal: path … does not exist` and I never saw it, because I had written
`2>/dev/null` to keep the output tidy.** A missing path and an empty result then
render identically.

> **Suppressing stderr converts *this question is malformed* into *the answer is
> zero*.** It is the vacuous-guard defect with a redirect operator instead of a
> filter, and it is one keystroke.

The published zero survived when re-run against the real path. **The instruction
that saved it was not "be careful" — it was "check the corpus exists before you
count in it", which is mechanical, and which caught this in one step.**

### Where the variance lives

We do not claim this suite is deterministic. We claim something more specific
and more useful, because it tells a maintainer what to do rather than only what
to fear.

```
SHARED CHECKOUT     non-reproducible results observed by three authors
DETACHED WORKTREE   ~~flakes observed by anyone, all session ......... 0~~
                    **FALSIFIED. THE TRUE VALUE IS 1, AND IT IS THE ONE
                    THAT MATTERS MOST.**
```

**That zero was true when I measured it, it survived an audit I ran on it
deliberately, and it is now wrong.** A reviewer reproduced a failure **inside a
clean detached worktree** — porcelain `0`, `.git` present, no other agent writing
that tree — one red in fourteen runs at one revision.

**This is the most dangerous thing this document could have shipped, and not
because of the digit.** The section around it argues that a detached worktree is
what makes a suite result trustworthy. That argument is correct for the mechanism
it addresses and it does not reach this failure at all:

```
ENVIRONMENTAL RACE   tests read repository state while many agents commit
                     -> a detached worktree genuinely cures it
INTRINSIC RACE       concurrency inside the product under test
                     -> a detached worktree does nothing to it whatsoever
```

> **A detached worktree fixes *whose tree* you measured. Only repetition fixes
> *which run* you measured. Both are required and neither substitutes for the
> other.**

**Had this shipped as written, the rule this document promotes would have
certified away the one open defect nobody can close by reading** — a green from a
procedure that cannot see the failure, which is the exact shape of every false
green catalogued here, arriving inside the remedy for them.

**And note how the zero survived.** I audited it, found it *correctly scoped*, and
was right: the contrast with the shared-checkout row is what gives it meaning.
**Scoping a claim correctly does not make it durable.** It was a true statement
with no expiry date, in a document whose central thesis is that true statements
acquire expiry dates.

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
REVISION            1f9fc70b   <- NOT this document's pin. See below.
RUNNER              cargo test -p onnx-genai-server --no-fail-fast
MEASURED BY         one reviewer, clean detached worktree, porcelain 0
OBSERVATIONS        14 at that revision
GREEN               13  ->  264 pass / 0 fail / 4 ignored / raw exit 0
RED                  1  ->  263 pass / 1 fail / 4 ignored / raw exit 101
FAILING TEST        concurrent_static_cache_chat_completions_share_batched_driver
                    tests/http.rs:388
                    left  "tok24 , tok28 tok27"
                    right "fox tok27 <eos>"
IN ISOLATION        5 of 5 pass — it only fails beside its 28 siblings
```

**This block previously read `PASS 264 / FAIL 0 / RAW EXIT 0` and described itself
as reproduced three times. Every one of those numbers was true.** They were three
samples of a distribution, presented as a property of a commit.

> **A test result is a sample, not a property of a revision.** A green suite
> proves the suite was green once. Only repetition separates *passes* from
> *passed* — and if you ran it once, the honest word is *once*.

**The reviewer who found this wrote the expected result down before running it,
read `263/1/4`, and could not talk themselves past the gap.** Without that
prediction the sequence is: see exit `101`, assume a lock or disk problem on a
full machine, re-run, see green, report green. **That is how this survived every
previous run tonight, including mine.**

**Two corrections to my own first version of this block, both of which I made in
the direction that reads better.**

**First, the revision.** I wrote this table under `37d0d72e`, this document's
pin. **The fourteen runs were taken at `1f9fc70b`.** I took another agent's
measurement and attached it to my own subject — a complete, correct measurement
relabelled with the wrong referent, which is the one defect class none of the
verification fields here can catch, because every field validates the reading and
none validates the subject. **The number is theirs and it is about their
revision.**

**Second, and worse: I wrote that the race is *reachable in the product and not
only in the test*. That is not known, and I did not measure it.** What is
measured is that the test passes 5 of 5 in isolation and 6 of 6 as a whole binary,
and failed once beside its 28 siblings. **That establishes the failure needs
concurrent load. It does not locate the race.** A shared-state bug in the test
harness produces exactly the same signature.

> **The honest statement is that we do not know whether the race is in the
> product or in the test, and it is not closed either way.**

**I guessed toward the more alarming reading, which felt like candour and was
still a guess.** In a document arguing that unearned precision is the defect, the
temptation is not to overclaim safety — it is to overclaim *rigour about danger*,
and it produces a sentence no one will challenge because challenging it looks like
complacency.

**It stays in this section rather than a footnote for the reason that survives
both corrections:** it is the concurrency test over the shared batched driver, and
that driver is the feature this demo exists to show. **A release note that claims
a green suite without saying its headline feature has an intermittent test would
be the exact overclaim this branch spent the session eliminating, committed on the
last page.**

**The direction of failure is the survivable one, and that is not a defence.** It
fails toward red. But a suite that can go red while the code is fine can also go
green while the code is not, and this is the only open item on the branch that
nobody can close by reading. **It must not be quietly re-run until green.** Note the structural limit honestly: `cargo test` prints one
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

**RETRACTED — DO NOT QUOTE THE TWO NUMBERS BELOW.** An earlier draft of this
section opened *the exposure ratchet is the only failing test on this branch* and
gave `94 fetchable, was 91`. **At the commit this document stamps, that is false
in both parts**, and the author found it while verifying the stamp rather than
while writing the claim:

```
AT 37d0d72e — THE TREE THIS PR SHIPS
  served-surface.test.js:155   MAX_SERVED_BUT_NOT_NEEDED = 85
  full JS suite at that commit, one checkout    749 pass / 115 suites / EXIT 0
  ⇒ THE RATCHET IS GREEN IN THE SHIPPED TREE. THERE IS NO FAILING TEST IN IT.

LATER, AT ddef0391 — 73 COMMITS AHEAD, MEASURED BY ANOTHER AGENT
  declared 88 · actual 91 · JS EXIT 1   ⇒ red, and correctly red
```

**`94` and `91` were measured at neither of those commits.** They were read once,
written down, and never derived again while the branch moved underneath them —
**the stored-value defect this document names three sections earlier, committed by
its own author, inside the section that names it.** The two numbers are not a lie
about the tree; they are a true reading of a moment, printed as a property of a
branch.

**What is true, and scoped to where it was measured.** At `ddef0391` the ratchet
is red at 91 against a declared 88, and the largest class is **61 of our own test
files, fetchable by any visitor to the demo**. That is a publishing decision
nobody made. It is counted, it is falling — it was 96 — and **the guard that
counts it is red on purpose**.

**The declared number has moved again since that run**, which is the same lesson a
second time: a figure quoted from this guard is only meaningful beside the commit
it was read at, and this document has now got that wrong once itself.

**And the author of that guard is the reason the red is worth shipping.** Faced
with a one-character edit that would have turned the suite green, they refused it
and wrote down why:

> *"The count at that commit was 96, not 88. Nine of those are other people's
> files that arrived while this number sat at 87. Absorbing them would cost one
> character and would silently publish nine artefacts nobody declared. The
> residual red is the correct output: it is not this guard failing, it is this
> guard working."*

**A declined keystroke, and a comment explaining the refusal so the next reader
could not mistake the red for a breakage.** The class breakdown below is quoted
from the `ddef0391` run and belongs to that commit, not to this one:

```
91 tracked files are fetchable at /demo/ that the page never loads (was 88).
BY CLASS: TEST 61 · INTERNAL_DOC 14 · TOOLING 10 · DESIGN 3 · FIXTURE 3
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
