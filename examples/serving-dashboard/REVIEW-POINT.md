# The review point

Reviewers extract exactly one tree. This file records which one, because until now that
designation existed only in chat.

REVIEW-POINT: review-1
REVIEW-POINT-SHA: fca13038

Declared by the project lead, 04:30, verbatim: *"REVIEW SHA UPDATED — `review-1` =
`fca13038`. THIS IS THE ARTIFACT THE THREE REVIEWERS WILL READ."*

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
