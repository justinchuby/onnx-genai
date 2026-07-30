# The review point

Reviewers extract exactly one tree. This file records which one, because until now that
designation existed only in chat.

REVIEW-POINT: review-0
REVIEW-POINT-SHA: 0aac6bb1

Re-declared by the gate secretary at 04:19, superseding the lead's 04:30 declaration of
`review-1` = `fca13038`. **The earlier declaration was not wrong — it was spent.**
`fca13038` (04:02:36) is a strict ancestor of `0aac6bb1` (04:16:22), so every finding
measured at it is re-derivable, not void.

> **⛔ Re-extraction is mandatory, not cosmetic, and the reason is the reverse of the one we
> built extracts for.** Both P1 render sites are *present* in the older tree and *absent* in
> this one. A reviewer working from a stale extract files a live defect **correctly for the
> tree in front of them and wrongly for the tree we ship.**
>
> **An extract removes drift. It does not remove staleness. We adopted it believing those
> were one property.** Drift is the tree moving *under* you; staleness is the tree having
> moved *before* you. Freezing a coordinate cures the first and *guarantees* the second.

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
git merge-base --is-ancestor <the-sha-you-measured-at> "$(git rev-parse fca13038)" \
  && echo 'PREDATES THE REVIEW POINT — re-derive before it counts' \
  || echo 'AT-OR-AFTER — stands'
```

Note the `git rev-parse` around the SHA rather than the tag name. **Comparing against a name
means the boundary can move under you, so the run that condemned your finding is not
reproducible from its own output.**
