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

Also pick wire names that cannot collide. `ttft` appears in
`crates/onnx-genai-server/src/metrics.rs` as a registry field; `ttft_client` does
not. A short token is a false positive waiting to happen.

Cite the **repo-relative path**, never the bare basename. Two crates ship a file
named metrics.rs, and a citation that resolves to either sends the reader to the
wrong one with full confidence.

## 5. Anti-vacuity controls are mandatory

A scan that reaches nothing passes. Every scanner here asserts:

- a **non-zero, plausible** number of call sites / keys were extracted;
- a **positive control** — a name known to be present IS found;
- a **negative control** — a deliberately bogus name is reported missing.

Green must mean "checked and clean", never "found nothing to check".

Corollary, learned the expensive way: **measure the hazard before building the
detector.** A text-similarity check for this very document was written, measured,
and thrown away — verbatim overlap with the guards was already near zero because
the duplication was semantic. It would have shipped permanently green.

## 6. Some guards read committed bytes, not the working tree — on purpose

They read the shipping ref because a reviewer clones a commit and so does CI.
Consequences when working on this repo:

- **They cannot go green before you commit.** Do not "fix" them in a loop
  pre-commit; commit, then re-run.
- **Re-run the suite with a clean `git status --porcelain`**, or you are reading
  one tree and reasoning about another.

Do not guess which guards these are from the filename — check for a
`shipped()` / `SHIPPING_REF` read. The two lists below are **verified in both
directions** by `examples/serving-dashboard/check-skill-drift.test.js`: every
file on the first line must read the shipping ref, and every file on the second
must not.

- Reads committed bytes: `examples/serving-dashboard/served-surface-rendered.test.js`,
  `examples/serving-dashboard/caption-catalogue.test.js`,
  `examples/serving-dashboard/check-perf-claims.test.js`
- Reads the working tree: `examples/serving-dashboard/served-surface.test.js`

Those first two are near-namesakes and they differ. An earlier revision of this
document confused them and told readers to commit in order to fix a red that
committing could not fix — which is why the working-tree line is asserted rather
than simply omitted. A counter-example nobody checks decays into an exception.

The house form resolves a **SHA** into `SHIPPING_REF` rather than spelling
`HEAD` at each call site, so that one run cannot read several different trees.

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

## 9. Where each rule is enforced — and which text wins

This document is **not** the authority on how these rules are enforced here. The
guards are: they sit beside the code, they run, and they fail. Prose cannot.

So the division is deliberate, and it is the whole reason this section exists:

| Concern | Authority | This document's job |
| --- | --- | --- |
| How a rule is enforced in *this* repo | the guard | cite it |
| The shape of the mistake, in code not yet written | this document | state it |

When the two overlap, **the skill cites and the guard states.** Restating a
guard's rationale here creates two independent wordings of one rule, and only
one of them can ever go red — so the copy that cannot fail is the copy that
rots, in front of the audience least able to check it.

The enforced statements:

| Rule | Enforced by | Mechanism |
| --- | --- | --- |
| §1 call sites resolve | `examples/serving-dashboard/dashboard/field-keys.test.js` | `KEY_LITERAL` |
| §3 deferrals name a falsifier | `examples/serving-dashboard/check-unplumbed-claims.test.js` | `absentWireNames` |
| §4 scanners strip comments | `examples/serving-dashboard/check-unplumbed-claims.test.js` | `COMMENT_ONLY_CONTROL` |
| §7 ratchets shrink on repair | `examples/serving-dashboard/telemetry-key-namespace.test.js` | `KNOWN_UNSERVABLE_KEYS` |
| §7 counted ratchet | `examples/serving-dashboard/served-surface.test.js` | `MAX_SERVED_BUT_NOT_NEEDED` |
| this section's own citations | `examples/serving-dashboard/check-skill-drift.test.js` | resolves every path and symbol above |

That last row is the point. Every path and symbol in this document is checked
against the shipping ref, so a mechanism that is renamed or deleted turns this
document **red** instead of leaving it quietly wrong. A skill is loaded *instead
of* reading the code; an uncheckable citation is not a broken link, it is advice
about a mechanism the reader will assume is still there.
