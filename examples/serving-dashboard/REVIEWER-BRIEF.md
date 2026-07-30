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

### 12. A stale reading and a narrow instrument look identical, and the flattering diagnosis is the wrong one

I was told a key was declared twice in `telemetry-provenance.js`. I searched for
duplicates with `^  '[a-z_.]+':` and found none — 36 declared, 36 distinct. I
nearly published *cannot reproduce*, and the reason I did not is that I had
misdiagnosed my own result once already that hour.

Both explanations fit a clean zero equally well:

```
(a) MY INSTRUMENT WAS TOO NARROW  -> widen it
(b) I MEASURED AFTER THE FIX      -> timestamp it
```

I assumed (a), widened the search, and immediately found evidence for it: one
entry, `metrics.e2e_latency`, is declared **unquoted**, so my pattern really was
blind to 1 of 37 keys. That felt like the answer. **It was not the answer.** The
actual duplicate was `'batch.capacity'`, declared at two-space indent, quoted, at
lines `:497` and `:637` — **a shape my original narrow pattern would have matched
perfectly.** It was live for exactly two commits and was fixed one commit before
the sha I measured at, which `git merge-base --is-ancestor` settles in one
command.

> **The report was right and I was late. Nothing was wrong with the instrument
> that had anything to do with this defect.** The Lead's own distinction is the
> remedy and I failed to apply it to myself: *wrong needs a correction, stale
> needs a timestamp.* Ask which one you have **before** you reach for the fix.

Prefer (a) and you will feel diligent while missing it, because widening an
instrument is real work that produces real findings — I did find a genuine
inconsistency — and none of it touches the reason you got a zero. **The
flattering diagnosis is that your tool failed. The useful one is usually that the
world moved.** At one commit every 45 seconds, (b) is the prior.

And widening has a cost nobody bills. My widened pattern reported three further
duplicates: `byOrigin`, `dynamic`, `scatter`. All three are **false** — nested
keys under different parents, at `:435`, `:463` and `:688`. Had I published, I
would have filed a duplicate-key alarm against a file that is correct, on a
branch under freeze, in the same message as a retraction.

> **Trading a false negative for three false positives is not an improvement,
> and a flat pattern cannot see structure. If you must widen, widen against a
> parser or against the runtime — the only check that actually decides this is
> `declared` versus `surviving`, and it needs no regex at all.**

Count the entries the source declares, count the keys the object has after it
loads, and compare. A collision is exactly `declared > surviving`. It cannot hide
from that pair and it is invisible to either number alone: **36 alone looks
perfect; 37 alone looks perfect.**

### 13. A withdrawal must state the question it answered, not just its verdict

A reviewer filed a finding, re-checked it, and withdrew it correctly. The
withdrawal read, in substance, *withdrawn — `dataset.state` is a real
vocabulary.* That is true. Ninety minutes later a second reviewer examined the
same six lines and found a live defect in them.

The two are not in conflict, and they read as though they are:

```
QUESTION ONE   is 'stale' a raw enum leaking to a user-facing surface?
               ANSWER: no — it is a member of the ruled vocabulary.   WITHDRAWN, correctly.

QUESTION TWO   do these writes go through the chokepoint that THROWS on an
               unknown state?
               ANSWER: no — all three assign the attribute directly.  LIVE.
```

A correct value written through an unchecked path is the one configuration in
which this class survives review, **because every reader who spot-checks the
value finds it correct and stops.** The first reviewer *was* that reader, and
their stopping is now a committed document that reads like a clearance.

> **Nobody re-derives the scope of a withdrawal.** *Already fixed* is the one
> disposal nobody re-checks; *already withdrawn* is the one finding nobody
> re-scopes. A retracted finding gets cited as proof the area is clean, and the
> retraction is more authoritative than the original claim ever was — because
> retracting costs the author something, and readers price that in.

The fix costs one clause. Not *R9 withdrawn*, but **R9 withdrawn as a leak; the
write path is unexamined.** The second version hands the next reviewer the live
finding instead of nearly burying it.

This document contains withdrawals. Read every one of them as scoped to the
question in it, and if the question is not stated, treat the area as unexamined
rather than as cleared.

### 14. `git show HEAD:` is absolute; `git grep … HEAD` is scoped to where you stand

We escalated all night from *read the file* to *read `HEAD`*, precisely to take
the desk out of the measurement. One of those two commands does that. The other
does not, and it is the one that looks more thorough:

```
IDENTICAL COMMAND · IDENTICAL COMMIT · ONLY THE DIRECTORY DIFFERS

  from repo root                     git grep -c batch_telemetry HEAD -- '*.rs'  ->  4 files
  from examples/serving-dashboard    (the same command, character for character) ->  0 hits

  git show HEAD:crates/onnx-genai-server/src/lib.rs   from that same subdirectory
                                                       ->  174 lines. UNAFFECTED.
```

`git grep <pat> HEAD -- <glob>` does not search the commit. **It searches the
commit intersected with your current working directory**, and the `HEAD`
argument makes that invisible, because `HEAD` names a commit rather than a
place. `git show HEAD:<path>` takes a root-relative path and is immune.

This produced a live order to delete a file that 32 committed references depend
on, including `lib.rs:25: mod batch_telemetry;`. The zero was real, reproducible,
and taken from a directory containing no Rust at all. **Deleting the file yields
`error[E0583]` and the crate does not compile.**

> **Print `pwd` beside `toplevel`, or your measurement has two unstated
> parameters instead of one.** Most of this crew works from
> `examples/serving-dashboard`, and every `git grep … HEAD -- '*.rs'` run from
> there returns a confident, well-formed, correctly-formatted zero about the
> entire Rust codebase.

And the control failed for a reason worth more than the defect. The finding used
a **glob** pathspec; the control used an **explicit root-relative path**. A root-
relative path behaves differently under a subdirectory cwd than a bare glob does,
so the control exercised an instrument configuration the finding never used and
returned a healthy number from the wrong machine.

> **A control must differ from the finding in exactly one respect: the expected
> answer.** Change the command's *shape* — its pathspec form, its anchoring, its
> flags — and you have validated a different instrument and learned nothing about
> yours.

### A deduplication can produce a value that was in neither half

`'batch.capacity'` was declared twice in `telemetry-provenance.js`, at `:497` and
`:637`. JavaScript keeps the **last** definition silently, so the terse entry was
the one that shipped and the fuller one was discarded with no error. It was fixed,
and the fix was verified — someone asked *which of the two survived*, which is a
step almost nobody takes.

Asking it was still not enough, because the honest answer is **neither**:

```
                 evidence                              label
  HALF A :497    symbol-anchored, full derivation      (none)
  HALF B :637    positional — 'admin.rs:178'           'Batch limit'
  ----------------------------------------------------------------------
  AT HEAD        A's evidence                          'Effective batch capacity'
                                                        ^ IN NEITHER HALF
```

The result is better than both inputs — the positional citation is gone
(`admin.rs:178` is now zero occurrences) and the label was corrected rather than
merely chosen. But *which half won* was the wrong question, and it is the question
the careful reviewer asks. **Diff the result against both inputs, not against your
expectation of which one deserved to win.** A merge can introduce a third value
silently, and nothing here made that value better except the judgement of whoever
performed it.

### The misnomer was retired in the catalogue and still ships in the renderer

This is the same fix, half-landed, and it is live at `HEAD`:

```
dashboard/scheduling.js:123   const maxBatch = telemetryStore.field('batch.capacity');
dashboard/scheduling.js:210   renderField(maxBatch, { label: 'Batch limit' })   <- ON SCREEN

telemetry-provenance.js:497   'batch.capacity': { … label: 'Effective batch capacity' }
```

Same key, two labels, and **the renderer is the one a visitor reads.** The
catalogue's label is not consulted by this panel at all — `scheduling.js`
hardcodes nine `label:` strings of its own — so correcting the catalogue changed
the audit trail and changed nothing on the page.

The distinction matters because the value is not a batch limit. It is
`effective_batch_capacity`, which `state.rs` defines as
`max_batch.min(max_queue_depth)`; the catalogue was renamed *precisely because*
`'Batch limit'` names the wrong quantity. **The renderer therefore ships the exact
caption the catalogue abandoned, and the rename stands as evidence that the crew
already agreed it was wrong.**

> Treat *the misnomer was fixed* as a claim about a surface, never about a
> repository. A caption lives wherever it is written, and a correction lands only
> where it is applied. This one was reported closed and is on the projector.

### An orphan is a private stale belief; a tracked file is a published one

Thirteen untracked artefacts were found late in the session and there was
pressure to land them all. Landing one imported a **retracted** measurement into
the branch four times over, in a commit whose message described it as evidence a
reviewer must read.

The mechanism is worth stating because it inverts the obvious instinct. An
append-only working document supersedes its own claims internally — the author
knows which paragraphs are dead. **Committing it wholesale republishes every
withdrawn finding in the present tense, with the authority of a tracked file.**

> For any document whose claims have been withdrawn, committing it is strictly
> worse than leaving it an orphan. Presence is the right first question. It is not
> the answer. **Distil, do not copy — commit a consolidated current state with
> retractions marked as retractions.**

A 32-kilobyte artefact was subsequently, deliberately left uncommitted on exactly
this reasoning: it predates three retractions, and landing it would have moved
staleness inside the perimeter where it inherits the tree's authority.

### What the citation harness does and does not certify

Gate items and review claims lean on `scripts/check_citations.py`. Its own author
put the limit in the verdict line, and it belongs beside any green that cites it:

> *A passing run means **only** that each pointer lands on something real. It does
> **not** mean the surrounding prose is correct.*

A citation harness can never verify that a pointer is the one the sentence is
about. It can only refuse to let an author believe the question was asked and
answered. Fourteen citations in the architecture document are flagged **ambiguous**
and are deliberately not claimed as fixed — the alternative was to infer them from
surrounding context, which is a checker fabricating the evidence it exists to
audit.

### 15. A full disk produces a file that exists, and a pipe produces an exit code that passes

Two laundering mechanisms were found within minutes of each other, and both
convert a failure into the exact signal our checks accept as success.

**The disk filled to 148 MiB free, machine-wide.** It did not announce itself as
a full disk — it surfaced as `OSError: problem writing element 2814976 to file`
from inside a numpy `tofile`, which reads as an application bug in serialization
code. If you are chasing a truncated artefact, a corrupt file, or a commit that
would not land, run `df -h` before you debug your own correct code.

> **A full disk produces a file that exists and is incomplete — precisely the
> artefact class an existence check certifies as present.** `ls` and `test -f`
> cannot see it. Anything written during that window needs a size *and a parse*,
> not a listing.

This gate had an item resting on exactly that: an artefact scored by whether
`tokenizer_config.json` was there. Re-scored by parsing it, and by parsing the
7 MB and 11 MB tokenizers beside it, on both served models. **Both are whole.**
The item is green on a parse, and it would have been green on a listing whether or
not it was true.

**And the second mechanism is punctuation.** A shell pipeline returns the *last*
command's status:

```bash
./scripts/build_qwen.sh 2>&1 | tail -40      # build FAILS, shell reports exit 0
```

Every one of us pipes build and test output through `tail`, `head` or `grep` to
keep it readable — so the modification you make for legibility is the one that
destroys the result. Use `set -o pipefail`, or read `${PIPESTATUS[0]}`.

> **If a green build or test run was reported through a pipe, it is not evidence.**
> The test command in this document is deliberately bare and unpiped for this
> reason; if you add `| tail` to make its output readable, you have discarded the
> only thing it was measuring.

Both of these belong to the family this document keeps meeting, and they are its
purest members yet: the failure and the success are **byte-identical at the point
of observation**, and in both cases the observer introduced the ambiguity while
trying to be careful.

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

Measured, rather than assumed — `git worktree list` reports **six checkouts on
five branches**, and `git rev-parse --git-common-dir` from this directory points
at *another checkout's* `.git`:

```
/Users/justinc/…/onnx-genai                justinchu/demo              <- the primary
/Users/justinc/…/onnx-genai-demo           feat/genai-demo-dashboard   <- YOU ARE HERE
/Users/justinc/…/onnx-genai-pris-multiturn squad/multiturn-session-api
/Users/justinc/…/onnx-genai-spec-capture   feat/speculative-conformance-capture
/Users/justinc/…/onnx-genai-unify-session  squad/unify-session-api
/private/tmp/…                             (detached, transient)
```

They are **one repository with one object store**, not sibling clones — a
distinction I got wrong in conversation before measuring it, and it has a
consequence worth more than the correction:

> **Every commit is resolvable from every checkout.** `git show <SHA>:<path>`
> gives identical bytes no matter which of the six directories you stand in, so a
> citation carrying a sha is portable across all of them. `HEAD` is the only
> ambiguous word, and it is the one every command uses by default.

So the failure mode is narrower and more treatable than *you might be in the
wrong repository*: you are in the right repository and on the wrong branch, and
the fix is to name the sha rather than to hunt for the correct directory.

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

### The honesty layer is the delivery mechanism, not the missing guard

The instinct on reading the path defect is *our honesty apparatus failed*. It did
not. It worked exactly as designed and that is why the path is on screen.

`telemetry-provenance.js` classifies `server.model_path` as `NOT_PLUMBED`, with
evidence reading *"No route returns the model directory today."* That was true
when written. It is false now — the route returns it, verified live on both
origins. The store detects precisely this condition: a field the catalogue says
is absent arrives carrying a value. Its documented remedy is **display the value
and warn**, on the reasoning that *hiding a real number is the exact failure this
branch exists to catch*.

That reasoning is correct, and it is written for **numbers**. Applied to the one
field on this dashboard whose value is an arbitrary operator-controlled string,
it publishes a home directory and a username to whoever is looking at the
projector.

> **So the property nobody designed: drift in the catalogue *promotes* a field
> onto the page.** The branch fires exactly when our documentation is wrong —
> which is exactly when nobody has reviewed the field. It is an unreviewed render
> path for arbitrary server strings, and its trigger condition guarantees it only
> ever fires on fields no one has thought about yet.

Two consequences a reviewer should not have to derive:

**Do not "fix" this by reclassifying the row to `MEASURED`.** That silences the
only instrument that noticed and leaves the path exactly where it is. The stale
classification is the symptom; the disclosure is the defect. Reclassify *after*
the render is fixed, never instead.

**Err-toward-truth is the right default for a gauge and the wrong default for a
free-text string.** We asked, everywhere in this document, what happens when a
value is *wrong*. We never once asked what happens when a value is *sensitive*.

The render fix is smaller than it looks: `ui/model-card.js` already lists
`{ key: 'server.model_id', label: 'Model' }` on the line directly above
`{ key: 'server.model_path', label: 'Directory' }`. The identifier row has
shipped all along. And note the caption forbids the obvious substitution — a
field labelled `Directory` cannot host an identifier, so renaming the value under
it produces a caption that lies. There is a second render site in
`dashboard/system.js`, and a third copy of the value inside the same loop as the
visible text: it is also written to `aria-label`, which the one instrument that
finally beat thirty review findings — looking at the page — cannot see.

### What the other review artefacts do not cover

Volunteered by their own authors, recorded here because a document should not be
the thing that vouches for itself:

- **`FIELD-MEANING-AUDIT.md` is scoped to numeric fields.** Measured: 15 table
  rows, **zero** mentioning `server.model_path`. The field in tonight's
  disclosure defect is a string, so it fell outside the column headings. The
  table is complete and honest about the set it chose; the set is the part that
  needed review.
- **That same table certifies `batch.capacity` as honest**, and it was scored by
  reading the catalogue and the render path without ever querying a server that
  structurally cannot batch. Both arms serve `batch_capacity: 4`.
- **`READABILITY-REVIEW.md` carries a known-false line naming an agent to act.**
  Left in place deliberately under the freeze and flagged rather than edited,
  matching the precedent set for the other known-false line above.

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
nine-way confirmation used a `404` control soundly: it was always `404` for the
missing file and `200` for a sibling in the same directory in the same second.

> **Pick a control that no commit can repair.** That sweep used `prefix-cache.js`
> as its missing file. The panel was *restored* at `42d9a3e5` and now returns
> `200` on both origins, so all nine confirmations expired at once -- not wrong
> when taken, invalidated by a feature landing that none of them could have
> anticipated. A control that depends on a file being **absent** depends on a
> product decision; a control that depends on an origin having **no catch-all**
> depends on a mechanism. Use a path nobody can ever create --
> `/demo/definitely-not-a-real-file-30685.js` -- which still `404`s on both
> origins today and cannot be restored by anyone.

Compare bytes when the answer is *present*; a sibling control is sufficient when
the answer is *absent*.

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

### 8.12 The latency table is empty, and the guard that would say so cannot see it

Ruled into this brief by the Project Lead as a measured, named gap. **The
mechanism he named is exact and permanent. The count in the ruling had already
expired when it was issued, and both halves matter.**

`dashboard/throughput.js:272-273` composes its field keys:

```js
for (const percentile of ['p50', 'p95', 'max']) {
  const field = telemetryStore.field(`${definition.prefix}_${percentile}`);
```

Five prefixes are declared at `:253-257`, so this builds **fifteen** latency
keys. Our field-key guard extracts keys by matching quoted string literals, and
**no character class at any width matches a backtick** -- so the guard could
never see any of the fifteen. The two it did know about were the two somebody
happened to type by hand elsewhere in the same file. *A coincidence, recorded in
the same notation as a decision.*

Verify all of it from `$(git rev-parse --show-toplevel)`:

```
git grep -nE 'latency\.[a-z0-9_]+_(p50|p95|max)' HEAD -- examples/serving-dashboard
  dashboard/field-keys.test.js   15 unique   <- the exemption list
  dashboard/panels.test.js        2 unique
  dashboard/throughput.js         2 unique   <- the only two typed in the panel

git grep -lE 'latency\.' HEAD -- examples/serving-dashboard
  -> throughput.js, three *.test.js, and four *.md.
  -> telemetry-provenance.js  ABSENT.   telemetry-store.js  ABSENT.
```

**Two facts, and they point opposite ways. Read both before you score this.**

**The exemption list is no longer two of fifteen -- it is fifteen of fifteen.**
It was hand-written, with a reason per key and a note explaining that hand-listing
is deliberate, and it landed at `59355c9c`, roughly twenty-five minutes before the
ruling that describes it as two. Do not file this half. It is closed, and its
author closed it while explicitly refusing the one-keystroke version that would
have generated the list from the panel sources -- *an exemption list that
maintains itself is not an inventory, it is a mirror; it would exempt whatever
the panels ask for and could never go red.*

**The product gap behind it is open and is larger than the guard defect.**
Neither the catalogue nor the store mentions `latency.` at all: **fifteen keys
declared, zero producers.** Every cell of the latency table renders as an em-dash.
That is an *honest* absence and not a lie, so it is not a P1 -- but a reviewer who
sees fifteen empty cells should know it is unplumbed by construction rather than
broken at runtime, and no test we own distinguishes those two.

The remedy the Lead attached is the durable one and it is worth more than fifteen
allowlist lines: **make the extractor fail on any binding whose argument is not a
string literal, so it declares what it cannot see.** The next dynamic binding is
then caught too, without anyone remembering to list it.

> **An exemption list derived from an instrument's output makes an unsurveyed gap
> look surveyed.** If your allowlist was built from what your tool reported, it
> documents your tool, not your code.

And note what the ruling's own expiry demonstrates, because it is this brief's
recurring subject arriving on the order to write this brief: **the ruling was
correct when reasoned and stale when issued, and executing it verbatim would have
published a closed finding as an open one.** Rule 5 says rulings decay. It does
not exempt the ruling that tells you to write down a decayed ruling.

### 8.13 The mtime signal that convicted the model will convict every correct model

Gate item 9 ("the model is rebuildable") was closed twice, independently, and the
second close is the stronger one: this brief closed it by **parsing** the
companion files; `1cb42f0e` closed it by **executing** a clean scratch build into
an empty `OUT_DIR` and running `scripts/verify_model.sh` to exit 0, with the
generation terminating on its own (`finish_reason=stop`). *Parsing shows the file
is well-formed. Executing shows the artefact it belongs to actually works.*

**They then reported that the instrument used in the first close is about to
invert, and they are right.** `shutil.copy2` preserves the source's mtime, so
companion files copied into a model directory inherit dates from the upstream
snapshot. The "July-12 companions beside a July-29 model" split -- which this
brief cited as evidence of staleness -- **appears identically in a model built
correctly minutes ago.** The signal that correctly convicted tonight's model will
convict every correct model from now on. That is @086345a5's and @e00032a4's
*guard shaped like the incident*, except it flips to false **positives**, which is
the failure mode that gets an instrument deleted rather than merely distrusted.

**Their remedy is right, and measuring it makes it righter than the reason given.
Do not replace the mtime table with a written-vs-copied table.** Across all
nineteen `qwen2.5-0.5b*` directories:

```
tokenizer_config.json   19 dirs   1 DISTINCT HASH   (5b5d4f65…)
vocab.json              19 dirs   1 DISTINCT HASH
merges.txt              19 dirs   1 DISTINCT HASH
tokenizer.json          19 dirs   2 DISTINCT HASHES  -> c0382117 x18
                                                        3fd16973 x1  (scatter-v2)
```

`tokenizer.json` was proposed as a member of the *written* set whose dates are
therefore real. **It is genuinely per-build on one directory out of nineteen.** On
the other eighteen it is byte-identical with an inherited July-12 date --
indistinguishable, by any test, from a copy.

> **Written-versus-copied is a property of the build run, not of the filename.**
> The same file is written by one build and copied by the next. Any table that
> classifies files by name is correct on one model here and wrong on eighteen.

So the fix is not a better classification -- it is **not needing one**. Hash the
companions against a fresh helper run. That removes the file-identity question
entirely, and it is why the worry retires while the finding stands: the leftover
companions are **byte-identical to what a build writes today**, so the provenance
was accidental and the content was always correct. *No number measured tonight
needs revisiting on account of the stop token.*

And note the discipline in how that conclusion was reached, because it is the
rarest move in this document: they found a mechanism that appeared to refute the
staleness finding, and then **killed their own refutation with a date** -- the
model was built at 22:39 and the helper that would explain its companions did not
land until 23:42. *A mechanism that postdates the artefact cannot explain it.*
One file survives all of this untouched: `inference_metadata.yaml` at 23:33 is
uniquely hashed on that model, so its timestamp is real evidence and the flag on
it stands.

### 8.14 The demo server serves the desk, not the branch

**This is the most consequential instrument finding in this document, because it
applies to the acceptance standard itself.** Reported by `fc8b5d97` against their
own prior claim; reproduced here independently, on two files:

```
curl :8133/demo/<file>  vs  git show HEAD:<file>  vs  the file on disk

  telemetry-store.js        served 49558   HEAD 48436   disk 49558   -> SERVES DISK
  dashboard/scheduling.js   served 13939   HEAD 13513   disk 13939   -> SERVES DISK
  app.js                    served 15582   HEAD 15582   disk 15582   -> indistinguishable
```

`--demo-assets-dir` points at the working tree, so the demo server executes
whatever is on the author's disk **at the instant of the request** -- including a
file another agent is part-way through writing. A reviewer clones. **Every browser
observation taken against these origins is a measurement of somebody's desk.**

Note the third row, because it is the trap: `app.js` is clean, so served, HEAD and
disk agree and the check cannot tell you anything. **A comparison that passes on a
clean file proves nothing about the mechanism** -- it must be run on a file that is
known dirty, or it is a control that differs from its subject in the one axis being
tested. Two of the three rows above are load-bearing; the third is decoration.

**What this costs the gate, stated plainly.** Item 10 was worded *one browser
load*, and that wording is now insufficient: a browser load against a
working-tree-backed origin certifies the desk. The item is re-worded, and the
change is a construction rather than a discipline:

> **Item 10 (revised).** Open the page against an origin whose `--demo-assets-dir`
> is a **detached worktree pinned to the frozen sha**, and prove the pin in the
> same invocation by comparing served bytes against `git show <sha>:<path>` **on a
> file that differs from the working tree**. A load against the shared tree does
> not close this item, however green it looks.

**And the honest accounting of what survives.** The two browser-confirmed P1s in
§0.9 were measured against `ui/model-card.js`, `dashboard/field-state.js`,
`telemetry-provenance.js` and `dashboard/system.js`. All four are byte-identical
to `HEAD` right now, so **both P1s stand, unmodified, on committed bytes.**

> But they stand by **luck, not by construction.** Nothing in the method that
> produced them checked whether those files were clean; they simply happened to be,
> while six other dashboard files were not. *An observation that would have been
> invalidated by a neighbouring edit is not made sound by the edit having landed
> elsewhere.* The finding survives. The method that produced it does not, and it is
> replaced above.

This is the same shape as this brief's own §8.10 correction and as the `git grep`
cwd defect: **an instrument reporting honestly about the wrong subject.** It has
now appeared on a document, a search, a test count, a control, and finally on the
acceptance standard. *If a measurement does not name its subject, it is not a
measurement -- it is an anecdote with a number in it.*

### 8.15 A retraction travels slower than the error it retracts

**The claim now circulating on three boards -- that this brief's author proved
`served.path` does not resolve, and that the model-path P1 therefore needs
re-scoring -- is a position this document retracted before it was ever repeated.**
§0.9 has said since its first draft:

> *I refuted this P1 in public and **I was wrong**.* The shipping store calls
> `projectServedModel()` first, which synthesises a `served` key from the `data[]`
> list, so at runtime it resolves to the full path.

The original error was broadcast. The retraction was broadcast too, and committed
into this file, and neither travelled. **The wrong half is now cited by name in a
reviewer's committed lane board as grounds for parking a live P1** -- *not mine to
re-adjudicate tonight* -- which is the most expensive possible outcome: the defect
is neither fixed nor argued, it is **suspended pending a fact that was withdrawn.**

Measured at HEAD, from the repository root, both glob forms:

```
git grep -n 'server\.model_path' HEAD -- '…/*.js' '…/**/*.js' | grep -v '\.test\.js'
  ui/model-card.js:25          { key: 'server.model_path', label: 'Directory' }   PRESENT
  dashboard/system.js:89       definition('model directory', …)                   PRESENT
  telemetry-provenance.js:150  the catalogue definition                           KEEP
```

**Two render sites, unchanged, while the task DAG records the deletion as
completed.** Both facts are true at the same commit.

> **A completion is a claim about a task identifier. Every human-readable field
> attached to it -- the summary, the sha, the test count -- describes whatever the
> agent actually did.** Nothing cross-checks the two, so an agent can do excellent
> work, report it with complete honesty, and close a task they never touched, with
> every field in the record accurate.

The remedy is the one already ratified for deletions, pointed at the ledger
instead of at the diff: **a completion must carry a predicate that goes false when
the work lands.** For this item it is the grep above, and it must reach one hit --
the catalogue definition -- not three.

And the general form, which is this session's most under-priced hazard:

> **An error and its retraction do not propagate at the same speed.** The error is
> new information and gets repeated; the retraction is a correction to something
> the reader already filed, and gets skipped. So a withdrawn finding keeps
> accumulating citations *after* it dies, and each citation makes it more
> load-bearing. **If you cite a colleague's finding, re-derive it or name the
> message you are citing -- because the one thing you cannot tell from a quotation
> is whether its author still believes it.**

*Corollary, and it is why this section sits in this file rather than in a
broadcast: a retraction that lives only in chat has the same lifespan as the
channel. This one is in the artefact, and the artefact is what a reviewer opens.*

### 8.16 The suite is red, and this brief published it as green

**Run this before you believe anything in §7, and run it exactly this way:**

```
node --test *.test.js dashboard/*.test.js ui/*.test.js > /tmp/raw.txt 2>&1 ; echo $?
```

**No pipe.** At the sha this section was written the answer is **`1`** --
`tests 621 · pass 613 · fail 8 · skipped 0`.

This document previously recorded the suite as **608/608 green**, and that number
was obtained through `| tail -18`. A pipeline returns the exit status of its *last*
command, so that run reported the exit status of `tail`, which succeeds at printing
anything at all.

**Two separate things follow, and conflating them would discard good work:**

> **The count survives; the completion does not.** `pass 608` is text the runner
> emitted, and it is still evidence of what the runner printed. What the pipe
> destroyed is the proof that the run *terminated* -- a suite that dies half-way
> still prints a green-looking summary for the files it reached.

**And the honest second half, which cuts against the alarm:** the pipe did **not**
conceal these eight. They did not exist when 608 was measured. Seven of the eight
live in test files committed **within five minutes** of this section, and the
eighth in one committed ninety minutes earlier. All eight are committed and clean
against HEAD -- none is a scratch edit. **They are red tests landing ahead of their
fixes, which is the discipline this team asked for.** The failing names are the
authorised work items almost verbatim: the missing classification text, the
degraded debug endpoint, the poll-flood suppression, the wrong-server detection,
the first-frame pending state, and two README perf claims.

**None of that makes the suite green.**

> **A red suite for a good reason is still a red suite.** The gate scores the
> artefact, not the intention behind it. An item that reads *the suite passes*
> cannot be satisfied by *the suite fails in ways we understand and endorse* --
> because every red suite in history was understood and endorsed by the person
> looking at it.

So gate item 2 is **RED**, and this document's own headline number was wrong in the
direction that ships. The lesson is not about `tail`:

> **The plumbing of a command silently overrode its semantics, and the plumbing was
> added for readability.** Nobody has ever thought of `| tail` as part of a
> measurement -- it is the formatting. It replaced the verdict. That is the fourth
> instance this session of a decorative-looking token changing what was measured,
> alongside a narrowing glob, an inherited working directory, and a line break
> hiding a string. **All four were silent and all four exited zero.**
