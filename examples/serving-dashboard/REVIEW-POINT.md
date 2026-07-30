# The review point

Reviewers extract exactly one tree. This file records which one. **This file is the sole
authority.** Broadcasts, chat messages and tag names do not outrank it, by explicit ruling
of the project lead.

REVIEW-POINT-SHA: 219307afcb7f3a27989e4f9d246a6ffa890f885e

**That is a raw hex SHA and it is deliberately not a tag name.** Three tags existed
(`review-0` `review-1` `review-2`); all three are retired. `review-0` was silently
re-pointed across sixty commits, and the tag numbers did not order the commits — see the
section below, which is retained precisely because it is the argument for this line's form.

> **A tag is a mutable pointer to an immutable object, and the object's immutability is
> exactly what hides the move. Every stale SHA still resolves. Nothing ever errors.**

## Measured state at this SHA

Both suites were run at **one SHA**, from **one clean detached worktree**
(`porcelain 0`), with **raw unpiped exit codes**. Neither number is inherited from
another agent's report.

| Suite | Result | Raw exit |
|---|---|---|
| JavaScript (`bash run-tests.sh`) | **831 tests · 829 pass · 2 fail** · 0 cancelled · 0 skipped · 0 todo · 123 suites · 64 discovered files | **1** |
| Rust (`cargo test -p onnx-genai-server --no-fail-fast`) | **270 passed · 0 failed · 4 ignored** across 6 test binaries | **0** |

**The JavaScript suite is RED at this SHA and that is not an oversight.** The two failures
are the exposure ratchet in `served-surface.test.js`:

- `no served measurement is left unrendered, beyond the pinned set`
- `the exposure ratchet has not been loosened` — *94 tracked files are fetchable at
  `/demo/` that the page never loads (was 91)*

This guard is working as designed. It fails because files were added inside the served
asset directory without anyone making a publishing decision about them. **Raising the
constant would turn the gate green in one character and ship the undecided files.** It has
been deliberately left red by more than one author. Do not raise it to go green.

**The Rust number is a COLD-TARGET build.** The worktree had no `target/` directory
before the run, so this also demonstrates the tree builds from nothing. Earlier Rust
numbers quoted tonight shared a warm `target/` and did not carry that property.

`cargo` prints no total — it prints one `test result:` line per test binary, six of them.
The 270 is their sum. Without `--no-fail-fast` a failure in the first binary aborts the
rest, so **the denominator shrinks exactly when something is wrong**; the flag is
mandatory, not stylistic.

### The 4 ignored Rust tests, by name

1. `tests::audio_endpoints_route_through_tiny_whisper_pipeline` — synthetic Whisper-contract smoke test; run explicitly for audio validation.
2. `tests::sidecar_free_compatibility_package_builds_server_pipeline_and_preprocesses_image` — missing fixture `vlm-executable/vision.onnx`; `.gitignore` skips `*.onnx` and nobody force-added one.
3. `tests::vision_request_routes_through_tiny_vlm_pipeline` — requires gitignored `models/tiny-vlm`.
4. `qwen_real_model_tool_use_chain_end_to_end` — requires the gitignored `models/qwen2.5-0.5b` fixture.

**All four are missing-fixture gates, not disabled assertions.** None is ignored because it fails.

## Extraction recipe — use this exact form

```sh
git worktree add --detach /tmp/review-tree 219307afcb7f3a27989e4f9d246a6ffa890f885e
cd /tmp/review-tree/examples/serving-dashboard
SHIPPING_TREE_REF=219307afcb7f3a27989e4f9d246a6ffa890f885e bash run-tests.sh
cd /tmp/review-tree && cargo test -p onnx-genai-server --no-fail-fast
```

> **⛔ Never use `git archive`.** It produces a directory that is not a git repository.
> Ten JavaScript guards resolve their corpus through git and silently degrade there,
> dropping tests **without reddening anything**. A smaller green suite and a correct green
> suite are byte-identical in a report.

`SHIPPING_TREE_REF` makes every guard read one immutable tree instead of whatever happens
to be on the reviewer's desk.

## Reading the numbers above

- **Verify by presence, never by absence of an error.** A missing file, an unapplied
  change and a filter that matches nothing all produce the same bytes.
- **The runner emits its counters as `ℹ tests`, not `# tests`.** Node v25 changed the
  prefix. A `grep '^# tests'` returns empty, which is indistinguishable from a clean run.
  This cost at least three agents a false measurement tonight, including the author of
  this file, who reported the runner as emitting no counters at all. **It emits all six.**
- **`node --test .` is a phantom red** — it prints `Could not find '.'`, exits 1, and emits
  zero counters. Use `node --test` with no path argument, or the canonical runner.
- Both instruments were run here and **agree exactly**: 831/829/2 from the runner and
  831/829/2 from a bare `node --test`.

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
