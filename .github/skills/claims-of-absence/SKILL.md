---
name: claims-of-absence
description: How to write, test and maintain a claim that something is NOT available — unplumbed telemetry fields, deferred keys, allowlists of "not yet landed" work. Use when adding an entry to a deferral list, when a dashboard field renders an em-dash, when writing a guard that scans source files for names, or when a test reads git HEAD instead of the working tree. Explains why undefined and unavailable must look different, and why a scanner must strip comments.
---

# Claims of absence

A claim that data is missing is a claim about the world, and it rots exactly
like any other. Unlike a claim of presence, **nothing fails when it goes
stale** — the field simply keeps rendering nothing, which is what it rendered
when the claim was true.

This repo has been bitten by this hard: ten `kv.*` keys sat in an allowlist
saying "block-table endpoint, not yet landed" while `/v1/debug/kv/blocks` was
live, registered, and *advertised on the wire by an endpoint the page was
already polling*. The panel em-dashed over data the server was sending.

## 1. Never let "undefined" and "unavailable" render the same

The root defect is always the same shape:

```js
telemetryStore.field('kv.block_size')   // key not in the catalogue
// -> returns a total, explained, `unavailable` field
// -> renders an em-dash
// -> identical to "the server genuinely has no data"
```

A total lookup (never throws, always returns something) is good design and is
*precisely* why a typo and a real gap are indistinguishable. So the distinction
has to be made somewhere else:

- **A test that reconciles call sites against the catalogue.** Parse every
  `field('...')` / `series('...')` literal out of the panel sources and assert
  each one resolves. See `examples/serving-dashboard/dashboard/field-keys.test.js`.
- **Distinct states, not one placeholder.** This project uses
  `measured / pending / unavailable / not-applicable / stale`. "No server could
  ever supply this" is `not-applicable` **with a reason**; "the server should
  have sent this and didn't" is `unavailable`. Do not invent a sixth state.

## 2. Prefer the server's own words

If the response carries its own state, use it verbatim. `BlockTableResponse`
has `applicable: bool` plus `FieldUnavailable { code, detail }`, and the codes
are already this project's vocabulary. The sentence a visitor reads should be
written by the people who know why, not guessed client-side from a missing key.

```jsonc
// GET /v1/debug/kv/blocks on a static-cache model
{ "applicable": false,
  "unavailable": { "code": "not-applicable",
                   "detail": "this model's KV cache holds no paged tensor storage, ..." } }
```

## 3. Every claim of absence must name a checkable wire spelling

A deferral entry whose reason is free prose ("not yet landed") can never go red.
Require the entry to declare **the names the server WOULD serve it under**, then
assert those names do not appear in the server sources:

```js
'kv.block_size': {
  reason: 'The block table endpoint is not registered.',
  absentWireNames: ['page_size', 'block_size'],   // <- the falsifier
}
```

Now landing the feature turns the list red and forces it to be updated. Pair it
with **keyset equality** against the list it mirrors, so neither can drift.

Same idea for a row that names no endpoint at all (`source: null`). That is the
strongest claim in a catalogue — *nothing could ever serve this* — and it is the
one claim no response body can refute, because there is no path to read. Its
falsifiability must **move** to a source scan, not disappear. Do not mark it
"unfalsifiable" and move on.

## 4. Strip comments before scanning source, or prose will lie to you

A scanner searching Rust for `uptime` reported it as present. It was present —
in a doc comment explaining **why a rate must not be derived from uptime**. The
scanner read prose *documenting an absence* as evidence of a *presence*.

Strip comments first, and keep a permanent regression control that asserts both
halves at once:

```js
// `uptime` IS in the raw file, and is NOT found by the scanner.
assert.match(rawAdminRs, /uptime/);
assert.equal(sourcesMentioning('uptime').length, 0);
```

Also pick wire names that cannot collide. `ttft` appears in `metrics.rs` as a
registry field; `ttft_client` does not. A short token is a false positive
waiting to happen.

## 5. Anti-vacuity controls are mandatory

A scan that reaches nothing passes. Every scanner here asserts:

- a **non-zero, plausible** number of call sites / keys were extracted;
- a **positive control** — a name known to be present IS found;
- a **negative control** — a deliberately bogus name is reported missing.

Green must mean "checked and clean", never "found nothing to check".

## 6. Some guards read `git show HEAD:`, not the working tree — on purpose

`served-surface.test.js`, `caption-catalogue.test.js`, `check-perf-claims.test.js`
and others read committed bytes, because a reviewer clones HEAD and so does CI.
Consequences when working on this repo:

- **They cannot go green before you commit.** Do not "fix" them in a loop
  pre-commit; commit, then re-run.
- **Re-run the suite with a clean `git status --porcelain`**, or you are reading
  one tree and reasoning about another.

## 7. Ratchets move in both directions

Pins like `KNOWN_UNSERVABLE_KEYS` and `MAX_SERVED_BUT_NOT_NEEDED` assert *exact
set equality* or an exact count. Repairing something **fails the test** until
the pin is shrunk in the same commit. That is the feature: it stops a list
draining quietly and keeps the stated size true.

House rule observed here for raising a ceiling: you may buy a green for a file
**you** added, with a sentence saying why, and never for anybody else's. Check
that the residual gap stays the same size — if raising the ceiling shrank the
disclosure, it was not honest.

## 8. Re-record fixtures, and read the diff

Re-capturing revealed the committed captures predated the shipped binary:
`/v1/status` had grown `build_sha`/`build_dirty` and no fixture knew. Stale
fixtures mean guards reason about a response shape the server stopped sending.
The capture recorder derives its endpoint list from `Object.values(ENDPOINTS)`,
so adding an endpoint needs no recorder change — but the fixtures still have to
be re-recorded and the diff actually read.
