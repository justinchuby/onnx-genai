# The review point

Reviewers extract exactly one tree. This file records which one. **This file is the sole
authority.** Broadcasts, chat messages, DAG task text and tag names do not outrank it, by
explicit ruling of the project lead.

REVIEW-POINT-SHA: d5da0061232248f5b08e115c0269249ccdad6fdb
MEASURED-AT: d5da0061232248f5b08e115c0269249ccdad6fdb

**READ THIS FILE AT THE BRANCH TIP, NOT AT THE PIN.** A declaration cannot live inside the
tree it declares. `1944a5e9` was scored first and its own copy of this file still names
`217ae170`, so a reviewer who extracted it would have been handed a stale boundary by the very
document whose job is to prevent that. The fix is not a cleverer commit order — it is a rule:
**this file is authoritative at the tip, and the hex it names is the tree to extract.** The
pin above was then re-scored directly, so the SHA that carries the declaration is also the SHA
that was measured, and the two are the same tree.

`d5da0061` differs from `1944a5e9` by documentation only. Proved, not asserted:

    crates/ tree object ......... identical at both (560f0a7e…)
    all 64 *.test.js blobs ...... identical at both (corpus hash 601f45bf…)
    [NEG CONTROL] same corpus hash at 37d0d72e ... 97e4b301…  DIFFERS

**MOVED FORWARD FROM `217ae170`, AND THE REASON IS A SECURITY FIX, NOT A GREENER NUMBER.**
`217ae170` and every earlier candidate — `37d0d72e`, `3b701494`, `d1c8fff0` — **do not contain
the C19 fix.** Measured in their own bytes rather than by ancestry alone:

    rest.contains('%')  in demo_assets.rs   at 1944a5e9 ... 1   [CONTROL] fn count 27
                                            at 37d0d72e ... 0

`f359363a` (07:57:47) is the commit that closes it and is an ancestor of this pin. **Pinning
any earlier SHA ships a live percent-escape bypass of the demo asset guard.** The failure
count at the earlier candidates is lower and that is exactly the trap: `d1c8fff0` scores
787/786 because it is **69 test arms smaller** than the tree above it (61 test files against
64; control — files present at the candidate and absent above: 0, a pure superset).

### The score at this pin, taken in a clean detached worktree — twice

    git worktree add --detach <dir> <RAW HEX>
    requested SHA == actual SHA .... ASSERTED, both runs
    porcelain 0 · untracked 0 · tracked-but-missing 0 ... both runs
    [POS CONTROL] test files enumerated by git ls-files: 64 ... both runs

    node --test $(git ls-files '*.test.js')      RAW UNPIPED EXIT: 1

      @ 1944a5e9  tests 868 · suites 126 · pass 860 · fail 8 · cancelled 0 · skipped 0
      @ d5da0061  tests 868 · suites 126 · pass 860 · fail 8 · cancelled 0 · skipped 0

**Two runs, two trees, same eight failures by name.** That is deliberate. A test result is a
sample and not a property of a commit — the Rust package produced exit 101 once and exit 0
twice at a single SHA earlier tonight — so a number quoted from one run should say `once`.
This one does not have to.

**Eight failures, and the split is five false to three true.** The five are one defect:
`run-demo.sh:329` is English prose inside a quoted `fail "..."` string, and the line-based
scanner in `check-launcher.test.js:48-66` counts it as a server launch. Named, so a reader
greps the assertion and not the count:

    every endpoint the dashboard polls is registered by the documented launch command   FALSE
    every flag in a copy-pasteable command exists in the server CLI                     FALSE
    every flag named in any demo document exists in the server CLI                      FALSE
    every server launch passes --demo-assets-dir                                        FALSE
    the --demo-assets-dir value is absolute, not relative                               FALSE
    no review document was measured before the tree reviewers extract                   TRUE
    no served measurement is left unrendered, beyond the pinned set                     TRUE
    the exposure ratchet has not been loosened                                          TRUE

The three true ones are disclosed, owned and deliberate. The ratchet reads 94 against a
ceiling of 91 and its author left it red on purpose rather than absorb three files that were
not theirs. **Do not raise it to go green.**

### The Rust suite is inherited, and the inheritance is proved rather than assumed

No `cargo` run was taken at this pin. The machine has 1.68 GiB free against 87 GiB of
existing `target/` directories, so building would have been a disk incident. Instead the
subject was shown to be unchanged:

    crates/ tree object @ 34ea441d ... 560f0a7ebf453746d297e2a4ad06f090c29f2080
    crates/ tree object @ 1944a5e9 ... 560f0a7ebf453746d297e2a4ad06f090c29f2080   IDENTICAL
    [NEG CONTROL]      @ 37d0d72e ... e613bf7a…                                   DIFFERS

Six commits separate the two SHAs and not one touches `crates/` (control: 5 paths changed,
all of them `.md` or `shipping-tree.*`). The Rust result carried forward is **272 passed · 0
failed · 4 ignored** at `34ea441d`, measured by its author, against byte-identical bytes.

**Two caveats, both against this inheritance and both mine to state.** A green suite is a
sample and not a property of a commit — the same package produced exit 101 once and exit 0
twice at one SHA earlier tonight, on a genuinely flaky concurrency test over the shared
batched driver. And an identical tree proves the *subject* did not move, not that the *result*
is reproducible. **Anyone who needs a certified Rust number must run it here, once disk
allows, and should expect to run it more than once.**

**ALL TAGS ARE VOID.** `review-0`, `review-1` and `review-2` are retired and no `review-3`
was ever created — **and none will be.** The project lead has ruled that no fourth tag exists
and that the review point is published as raw hex only. A task order asking for `review-3` to
be *cut* is satisfied by the hex above, not by `git tag`. The line above is a raw hex SHA and that is deliberate: a tag is a mutable
pointer to an immutable object, and the object's immutability is exactly what hides the move.
`review-0` was silently re-pointed across sixty commits under four reviewers and every stale
SHA still resolved, so nothing errored. Cite the hex. Never cite a name.

## Late measurements, taken in clean detached worktrees (08:00–08:10)

Everything in this section was measured under the project lead's rule: `git worktree add
--detach <dir> <RAW HEX>`, then `node --test $(git ls-files '*.test.js')`, exit code taken
**unpiped**. A desk run is a rumour. Scored `tests === pass` AND `cancelled === 0` AND raw
exit — never `fail === 0`, which is green on a suite that crashed before running.

### The candidate pins are greener because they are smaller

    21664cce  (tip, 08:00)        856 tests · 848 pass · 8 fail · 0 cancelled · EXIT 1
    d1c8fff0  (123 commits back)  787 tests · 786 pass · 1 fail · 0 cancelled · EXIT 1

    tracked *.test.js   at 21664cce: 64      at d1c8fff0: 61
    [CONTROL] files present at d1c8fff0 but absent at the tip: 0   -> PURE SUPERSET

`787/786` is not a better tree. It is the same tree asked **69 fewer questions**. Any pin
chosen for its lower failure count must publish its denominator beside it, or the number
measures blindness and reads as quality.

### Five of the eight failures at the tip are false, and the cause is measured

`check-launcher.test.js:79`, `:93`, `check-launch-command.test.js:513`, `:619` and
`check-endpoint-registration.test.js:150` are **not** product defects.

    git diff --stat d1c8fff0..21664cce -- run-demo.sh      ->  +75 / -4

    lines beginning with a bare ${SERVER_BIN}:
      at d1c8fff0 ... NONE
      at 21664cce ... :329  "${SERVER_BIN} is shared between worktrees, so the
                             build above may have..."

That line is English prose inside a double-quoted `fail "..."` message. `serverLaunches()`
(`check-launcher.test.js:48-66`) is a line-based scanner that skips only `#` comments and has
no notion of shell quoting, so it counts the sentence as a server launch. The two real
launches are at `:355` and `:364` and both pass `--demo-assets-dir "${SCRIPT_DIR}"` at `:359`
and `:368`. `--short` is git's flag (`:276`); `--version` is generated by clap and can never
appear in `cli.rs`. **The guard did not change and the product did not change. A
documentation string went red.** The repair is two `FOREIGN_FLAGS` entries
(`check-launch-command.test.js:599`) and one predicate tightening. **Do not edit
`run-demo.sh`.**

The remaining three are true, owned, and cheap: the exposure ratchet (below), the review
freshness guard (one bare-hex `MEASURED-AT:` line per document), and
`served-surface-rendered.test.js:254`.

### The exposure ratchet was not newly loosened

The guard reads **94 against a constant of 91**. Nothing was added recently:

    count at 9b54d3a9 (the commit that SET the constant to 91) ... 94
    count at the tip ............................................ 94
    files added under the served dir in that range ..............  0
    [CONTROL] paths changed in that range ....................... 29
    [CONTROL] deletions in that range ...........................  0

The residual is deliberate and its author documented it in the raising commit: *"The count at
this commit was 94. Three of those are not mine and I am not buying them a green."* The three
unbought files are `caption-catalogue.test.js` (`e56b211d`),
`telemetry-key-namespace.test.js` (`5b3373b1`) and `served-surface-rendered.test.js`
(`a0a96daa`). Arithmetic closes: 96 at the previous raise, minus 7 `harness/*.py` moved out,
plus 5 tests in, equals 94.

All three are single-file commits, so attribution by co-touched files yields nothing, and all
~590 commits on this branch share one git identity, so `--author` discriminates nothing.
**These three files are unattributable by any instrument available here.** All three are TEST
class; the recommendation is to raise the constant to 94 with a sentence, not to move files,
because the suite is served from the directory it tests by design.

Separately: the constant is one too high. It was raised by 3 for `markdown-scan.js`,
`markdown-scan.test.js` and `run-tests-guards.test.js`, but `markdown-scan.js` classifies as
`PAGE_ASSET` and the ratchet does not count it (measured: 0 occurrences in the counted set,
against 1 for its `.test.js` sibling as a control). By the raiser's own stated rule the
constant should have been 90.

### The two security items have opposite remedies

    C19  percent-escape dotfile bypass ... OPEN at 37d0d72e · CLOSED at the tip
    F2   error-channel path disclosure ... OPEN at 37d0d72e · OPEN at the tip

C19 is closed by `f359363a` (07:57:47), which is an ancestor of the tip and **not** an
ancestor of `37d0d72e`. At the tip `demo_assets.rs:188` refuses any `%` and `:467` is a named
regression test; the same grep at `37d0d72e` returns 0. On the wire, all four live origins:
`/demo/index.html` → 200, `/demo/index%2Ehtml` → 404, `/demo/zzq-nonexistent.html` → 404.
A commit titled *"require the demo directory; stop serving dotfiles"* ships a working bypass
of exactly that guard at `37d0d72e`, and does not at the tip. **Pinning forward closes it.**

F2 is not closed by any choice of SHA. At the tip there are **31** `ApiError::internal(format!
(...))` sites and **19** `map_registry_error` call sites; `routes/mod.rs:772` and
`routes/admin.rs:510` are not among the 19, and both concatenate an `anyhow` chain containing
an operator-chosen absolute path into a 500 body. The correct helper already exists and is
already used nineteen times. This needs a commit or an honest known-gap sentence in the PR.

### Instrument errors made while producing this section, disclosed

A probe of `POST /v1/chat/completions` using a **nonexistent** model returned zero `/Users/`
hits and looked like an all-clear. It is void: `routes/mod.rs:766` returns early on a lookup
miss and never reaches the leaking load path at `:772`. A negative result against a subject
that cannot reach the code under test measures nothing.

An earlier explanation of the 856-vs-787 gap — *"the failing guards do not exist at the
candidate"* — was wrong and was withdrawn before publication: five of the six failing files
exist at both SHAs. Only `served-surface-rendered.test.js` is absent. The real cause is the
`run-demo.sh` prose line above.

Checking whether this document already recorded these findings, `grep -c` reported hits for
`856`, `848` and `94`. All three were coincidental substrings **inside SHAs** (`818856ab`,
`15848a4`, `3b701494`). A substring match is not a citation, and a bare count cannot tell the
difference.

## Why this SHA and not the branch tip

The tip at the time of cutting was `0d0646a61370523e3ac15848a4cf63b3187a71b6`. It is **not**
the review point, and the reason is measured, not preferred:

    92cc7935  07:20   JS fail 3      last known good before the range
    2d516df4  07:21   JS fail 3
    818856ab  07:21   JS fail 3
    a2df3eed  07:22   JS fail 3
    776b5d1c  07:22   JS fail 3
    217ae170  07:22   JS fail 3   <- THE REVIEW POINT
    3ef731d2  07:24   JS fail 7   <- THE BREAK. run-demo.sh +47/-4, main.rs, build.rs
    0d0646a6  07:26   JS fail 8

`3ef731d2` ("fix(demo): the launcher tests a mode bit; make it test provenance") reddened
four previously-green tests in two files, isolated by running those files alone at each
commit in the range:

    check-launch-command.test.js   23 pass / 0 fail  ->  21 pass / 2 fail
    check-launcher.test.js         10 pass / 0 fail  ->   8 pass / 2 fail
      every server launch passes --demo-assets-dir
      the --demo-assets-dir value is absolute, not relative
      every flag named in any demo document exists in the server CLI
      every endpoint the dashboard polls is registered by the documented launch command

That regression is **open and unowned as of this writing**. It is not fixed here and this
file does not claim it is. The review point is the last commit before it.

## The measurement — both suites, one SHA, one clean detached worktree

Taken at `217ae17052f50b901ebd5bb057bfab5ffd418c49` from a single `git worktree add --detach`,
porcelain asserted (not read) as 0 before either suite ran.

    JS    bash run-tests.sh                            RAW UNPIPED EXIT: 1
          tests 847 · pass 844 · fail 3 · cancelled 0 · skipped 0 · todo 0
          suites 125 · discovered files 64
          provenance: 0 untracked, 0 tracked-but-missing

    RUST  cargo test -p onnx-genai-server --no-fail-fast   RAW UNPIPED EXIT: 0
          214 + 0 + 18 + 28 + 10 + 0 = 270 passed / 0 failed / 4 ignored
          across 6 test binaries, built from a COLD target directory

Cargo prints no total. The 270 is the sum of six `test result:` lines and is stated as a sum
so it can be checked. `--no-fail-fast` is mandatory: without it the denominator shrinks at
exactly the moment something is wrong.

The four ignored Rust tests, by name, because a count alone hides which:

    tests::audio_endpoints_route_through_tiny_whisper_pipeline
    tests::sidecar_free_compatibility_package_builds_server_pipeline_and_preprocesses_image
    tests::vision_request_routes_through_tiny_vlm_pipeline
    qwen_real_model_tool_use_chain_end_to_end

## The three JS failures are disclosed, not hidden — and must not be "fixed" by constant

    no review document was measured before the tree reviewers extract
    no served measurement is left unrendered, beyond the pinned set
    the exposure ratchet has not been loosened
    (the served surface is a closed set — same served-surface.test.js file)

The second and third are the **exposure ratchet**: tracked files fetchable under `/demo/`
that the page never loads. It is a working guard doing its job. Several authors have
deliberately declined to raise its constant. **Do not raise the constant to go green.**

The first is ~~structural, and no choice of SHA fixes it; the predicate is unsatisfiable as
written at every commit that exists, and the repair is to require the named SHA to be an
ancestor rather than an equal~~ — **RETRACTED BY ME, 07:40, having read the guard instead of
inferring it. Every clause above is wrong and the correction is the opposite sign: the
predicate ALREADY keys on ancestry, it is satisfiable today, and one document satisfies it
right now.** `check-review-freshness.test.js:62` excludes `REVIEW-POINT.md` from its own
corpus, and it fails a document only when *every* SHA that document declares is a **strict
ancestor** of the boundary declared here. Measured against this review point:

    ARCHITECTURE-SECURITY-REVIEW.md   1e809173 8a309ce0 9b06d922        ALL STRICT ANCESTORS -> STALE
    IMPLEMENTATION-REVIEW.md          3b701494…                          ALL STRICT ANCESTORS -> STALE
    READABILITY-REVIEW.md             8230060c … 92cc7935 37d0d72e       ALL STRICT ANCESTORS -> STALE
    REVIEWER-BRIEF.md                 ef7c91b9 (07:33:22)                NOT an ancestor -> ALREADY GREEN

`ef7c91b9` is a *descendant* of this review point, so the brief already satisfies the guard.
That single row disproves the retracted claim outright: a predicate one document satisfies
is not unsatisfiable. The red is three documents measured before this boundary, which is a
true and useful thing for the guard to be saying.

**REMEDY, AND IT IS ONE LINE PER DOCUMENT, NOT A CODE CHANGE:** the owners of the three
stale documents write `MEASURED-AT: 217ae17052f50b901ebd5bb057bfab5ffd418c49` into their own
file, raw hex, never a ref name. Do not modify the guard. Do not relax the predicate.

Recorded here rather than only in chat, because the retracted claim was published twice and
a broadcast expires while a committed file does not. The error was mine and its cause is the
one this branch has paid for all night: **I described a predicate I had not read.**

## Ancestry — all eleven required fixes, verified in both directions

Every one checked with `git merge-base --is-ancestor <fix> 217ae170` and the reverse, because
`--is-ancestor` returning false conflates "newer" with "not comparable". All eleven contained;
all eleven reverse checks correctly false.

    1133a874  02b54684  1384f7aa  627627a4  f025ae58  964cad4a
    03326348  dd04f50f  6e4ea4c2  ac24a964  6800c5b2

## Extracting the tree

    git worktree add --detach <dir> 217ae17052f50b901ebd5bb057bfab5ffd418c49

Then assert, do not read, that `git status --porcelain` is empty, and reconcile the file count
against `git ls-tree -r --name-only <sha> | wc -l` — it was 2206 == 2206 here. A partial
`git worktree add` can exit 0 and a partial checkout is porcelain-clean, so porcelain alone
proves nothing. A cold `cargo build` is the stronger integrity proof and it is free: a partial
checkout fails at the first missing `mod`.

**Never use `git archive`.** It yields a non-git directory in which roughly ten JS guards
silently degrade and drop tests without reddening.

Note that this file cannot be read from inside the tree it names: at `217ae170` this SHA does
not appear in it. Read this file at the branch tip, then extract.

## ⛔ The tag numbers do not order the commits

**Measured at `7b177e32`.** This is the most dangerous naming defect in the repository,
because it is the kind no reader ever tests — nobody runs an ancestry check to find out
whether `review-0` came before `review-1`.

```
COMMIT TIME ORDER — the number is not the sequence:
  04:02:36  fca13038  review-1     ⬅ FIRST
  04:16:22  0aac6bb1  review-0     ⬅ SECOND
  04:19:23  0bc86726  review-2     ⬅ THIRD

ANCESTRY, all six pairs tested:
  review-1  is an ancestor of  review-0     ⬅ THE INVERSE OF ITS NAME
  review-1  is an ancestor of  review-2
  review-0  is an ancestor of  review-2
```

**Two of the three pairwise comparisons match the numbering and one is exactly inverted.**
That is the worst possible ratio: enough agreement to confirm the assumption, enough
disagreement to make it false. A name that is *usually* monotonic is more dangerous than one
that never is, because the exceptions arrive as surprises rather than as habits.

> **A sequential name is a claim about order, and this repository does not keep it.** The
> cause is mechanical and already documented below: these are lightweight tags, so any of them
> can be re-pointed with `-f` at any time, and `review-0` was — twice. **Prefer the SHA in
> every citation. The tag name is a nickname, not an address, and it does not even sort.**

## Why this file exists rather than a convention

Three `review-*` tags exist and **no rule in the repository says which one is authoritative.**
At the moment this was written, three agents held three different answers and every one of
them was defensible:

| who | answer | how they got it |
| --- | --- | --- |
| the project lead | `review-1` → `fca13038` | declared it |
| a secretary's board | `review-0` → `6ecd9183` | measured it, correctly, at 03:57 |
| `check-review-freshness.test.js` | `review-2` → `0bc86726` | inferred: newest by commit date |

**The guard's inference was the most dangerous of the three, because it was automatic.** It
would have enforced a boundary the lead did not choose, on every review document, silently and
with a plausible justification. A wrong answer that a human states can be argued with. A wrong
answer that a test computes gets obeyed.

So the guard no longer guesses. It reads this file, and if the file is missing it **fails and
names the candidates** rather than picking one. *Refusing to answer is a valid measurement;
inventing a denominator is not.*

## Two properties of these tags that make the ambiguity worse

**The numbering does not sort by time.** `review-1` is 04:02:36, `review-0` is 04:16:22,
`review-2` is 04:19:23. Sorting the names gives a different order than sorting the commits, and
readers sort by the name without looking.

**All three are lightweight tags** — `git cat-file -t review-1` returns `commit`, not `tag`.
A lightweight tag is a bare pointer: no tagger, no date, no message, and **no reflog by
default**, which is why a tag that moves leaves no evidence that it moved. `review-0` named
`6ecd9183` at 03:57 and `0aac6bb1` at 04:21 — 60 commits apart — and nothing in the repository
records the change. **An annotated tag (`git tag -a`) would have carried the designation, the
author and the reason in the object itself, and this file would be unnecessary.** That is the
better fix and it belongs to whoever owns the tags; this file is the cheap one available now.

## How to check your own findings against it

Any finding measured before this point describes a tree the review has moved past:

```sh
# Read the boundary FROM THIS FILE. Do not retype it.
BOUNDARY=$(git show HEAD:examples/serving-dashboard/REVIEW-POINT.md \
             | sed -n 's/^REVIEW-POINT-SHA:[[:space:]]*//p')
git merge-base --is-ancestor <the-sha-you-measured-at> "$(git rev-parse "${BOUNDARY}^{commit}")" \
  && echo 'PREDATES THE REVIEW POINT — re-derive before it counts' \
  || echo 'AT-OR-AFTER — stands'
```

Note the `git rev-parse` around the SHA rather than the tag name. **Comparing against a name
means the boundary can move under you, so the run that condemned your finding is not
reproducible from its own output.** Note `^{commit}` too: `rev-parse` on an *annotated* tag
returns the tag object, not the commit, and `--is-ancestor` peels while `rev-parse` does not.

**This snippet reads the boundary out of this file instead of restating it, and that is a
repair, not a flourish.** It previously hardcoded `fca13038` while the header four lines from
the top of this file declared `0bc86726` — **two different review points in the one document
that exists to stop there being two.** The header was right and the snippet was the part
people copy. A worked example is not documentation *about* the source of truth; it is a
second copy of it, and it drifts exactly like any other duplicate.

## ⚠️ If you re-point this file, the ORDER matters and the wrong order is the obvious one

Re-pointing the boundary forward makes every document whose newest `MEASURED-AT` is a strict
ancestor of the new pin go **red**. Priced against committed bytes at the time of writing,
moving to `37d0d72e` would redden **four of the five** adopting documents — not because their
authors have not re-measured, but because they published the re-measurement **in chat and not
in the document**. Every SHA those reviewers have publicly re-derived at is at-or-after that
pin and would clear it instantly.

**So: collect the one-line `MEASURED-AT` updates FIRST, then move the boundary.** Reversed,
the same two actions produce four true-but-useless alarms aimed at people who already did the
work. The guard is not wrong in that state — the documents genuinely are behind — but a red
that four compliant authors cannot act on faster than one line of chat is a red that teaches
people to ignore the guard.

MEASURED-AT: 049da5f8 — this file's own boundary declaration is unchanged by that measurement;
what was measured is the snippet defect and the re-point pricing above.
