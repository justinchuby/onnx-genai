# The review point

Reviewers extract exactly one tree. This file records which one. **This file is the sole
authority.** Broadcasts, chat messages, DAG task text and tag names do not outrank it, by
explicit ruling of the project lead.

REVIEW-POINT-SHA: 217ae17052f50b901ebd5bb057bfab5ffd418c49
MEASURED-AT: 217ae17052f50b901ebd5bb057bfab5ffd418c49

**ALL TAGS ARE VOID.** `review-0`, `review-1` and `review-2` are retired and no `review-3`
was ever created. The line above is a raw hex SHA and that is deliberate: a tag is a mutable
pointer to an immutable object, and the object's immutability is exactly what hides the move.
`review-0` was silently re-pointed across sixty commits under four reviewers and every stale
SHA still resolved, so nothing errored. Cite the hex. Never cite a name.

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

The first is structural and no choice of SHA fixes it. The freshness guard requires a review
document to name a SHA that is not older than the tree. A reviewer cannot measure at the
final SHA until it is final, and it cannot be final until they have measured at it. The
predicate is unsatisfiable as written at every commit that exists. The repair is to require
the named SHA to be an **ancestor** of the shipping tree and to disclose the delta, rather
than to require equality. That change is not made here.

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
