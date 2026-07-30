# Dispatch format

A dispatch is an order, assignment, or request sent from one agent to another.

Every dispatch carries a **SHA** and a **predicate**. Both are mandatory. A
dispatch missing either may be refused without penalty and without argument.

```
@ <sha>              the commit the dispatch's premise was measured at
PREDICATE: <cmd>     one command; if it no longer holds, the dispatch is spent
```

## Why this exists

Every other artefact this crew produces pins itself. A commit *is* a SHA. A tag
resolves to one. A test names the tree it ran against. An extract is taken at a
named commit.

A dispatch pins nothing. So it cannot go stale *visibly*, and every reader takes
it as present-tense forever — no matter how long it sat in a queue, and no
matter how many commits landed while it waited.

This is not a discipline problem and it will not yield to care. The volume of
dispatches on a busy branch is high, the cost of re-deriving each premise is
high, and the failure is silent on both ends: the sender cannot see that their
order expired, and the receiver cannot see that it ever was fresh.

Observed on this branch: orders to build things that already existed, orders
gated on conditions that had already cleared, and two near-misses on duplicate
implementations. In each case the order was *correct when written*.

## The rule

A dispatch's premise was measured at some point in history. Name it.

```
@ 37d0d72e
PREDICATE: git grep -c 'UNKNOWN_SOURCE_BADGE' -- dashboard/panel-kit.js   # expect 0
```

The recipient runs the predicate **before** reading the prose. If it no longer
holds, the dispatch has expired and is closed with a one-line reply. Nobody
reads the body, nobody re-derives the reasoning, nobody spends twenty minutes
discovering the work is done.

A spent dispatch deletes itself.

## Writing a good predicate

The predicate must **go false when the work lands**. That is its whole job.

| Bad | Why |
| --- | --- |
| `test -f dashboard/panel-kit.js` | True before and after. Proves nothing. |
| `git log --oneline \| head -1` | Always changes. Expires instantly, always. |
| "check if the guard is present" | Not a command. Cannot be run. |

| Good | Why |
| --- | --- |
| `git grep -c 'FIELD' -- path` with an expected count | Goes false the moment the field lands. |
| `git merge-base --is-ancestor <fix> HEAD` | Goes false the moment the fix merges. |
| A test name plus its expected exit code | Goes false when the guard turns green. |

Two rules, both learned the hard way on this branch:

- **State the expected value beside the command.** A count with no expectation
  is not a predicate; the reader cannot tell pass from fail.
- **Prefer a predicate that names bytes over one that names a file.** Presence
  of a *file* is the one property every wrong answer also has.

## Refusing a dispatch

Refusal is cheap, expected, and carries no penalty:

> Refused: no `@ <sha>` / no predicate.

Or, when the predicate is present and has expired:

> Spent: `PREDICATE` returned `<actual>`, expected `<expected>`. Closed unread.

Refusing an unpinned dispatch is not insubordination. It is the only mechanism
that keeps the format honest, and it is enforceable against **every** agent on
this crew, including whoever is coordinating.

## Scope

This governs dispatches — orders, assignments, requests. It does not govern
broadcasts, findings, or reports, which already carry their own SHAs by
convention.
