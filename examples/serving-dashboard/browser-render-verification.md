
# §1 — AC52: The Five State Treatments, Seen In A Browser

**Method.** Real Chrome 150 (CDP), the assembled page from `GET /demo/`, both origins.
Specimens rendered into one fixed 260×56 box at 2× and captured with
`Page.captureScreenshot`, then compared **byte-for-byte**. Same box, same font,
same paint pipeline — the only variable is `data-state`.

Assets are served from `--demo-assets-dir` (disk), so CSS/JS here are at HEAD
even though the running binary is older. This distinction matters for behaviour
claims; it does not affect a render claim.

## 1.1 panels.css is not merely linked — it is PARSED AND APPLIED

Both origins, identical:

    <link> hrefs   ./styles/tokens.css  ./styles/shell.css  ./styles/panels.css
    CSSOM            2 rules  tokens.css
                    76 rules  shell.css
                   137 rules  panels.css      <- live in the CSSOM
    ES modules      21 loaded
    tokens resolve  --og-na-fg=#7e8fa0  --og-unavail-fg=#758493

The orphaned-stylesheet defect is **closed with the evidence class that would
have caught it**: not a 404 (a never-referenced sheet is never requested), but
137 rules resolved in the document's own CSSOM and its custom properties
resolving on `:root`. Verified on **both** origins.

## 1.2 All ten pairs of the five ruled states are distinct

    measured         #e6edf3   normal   no border
    pending          #748494   italic   no border
    stale            #7a8794   normal   1px dashed #4a5560
    unavailable      #758493   normal   1px dotted #3d4855
    not-applicable   #758493   normal   3px double #3d4855

Ten of ten pairs distinct. ✅

## 1.3 🟢 D232 REFUTED — not-applicable does NOT render in unavailable's tokens

They share a foreground colour and are separated by **border style**:
`1px dotted` vs `3px double`. Pixel diff: **different**. The concern was
reasonable from the colour tokens alone; the border rule resolves it.

## 1.4 🟢 D233 REFUTED — italic on `···` is NOT inert. And I got this wrong first.

My first instrument measured **bounding-box width**: `···` normal 12.66px vs
italic 12.66px, delta 0.00px — apparently inert. **I nearly shipped that.**
The control saved it: `123.4` normal 37.41 vs italic 37.45, delta **0.05px** —
so the instrument reported "inert" for the case that is visibly italic too.
**Width cannot detect slant.** A synthesised oblique shears glyphs in place
and need not change advance width at all.

Pixel diff, which measures the actual thing: `pending(···)` **637 bytes** vs
plain `···` **643 bytes** — **visibly different**. Italic renders.

*This is the third time tonight a control arm inverted my own verdict, and the
first where the control was the only reason I didn't publish a false RED.*

## 1.5 🔴 CONFIRMED, P1 — an UNKNOWN or ABSENT state renders PIXEL-IDENTICALLY to `measured`

    measured        2219 bytes ┐
    UNKNOWN-STATE   2219 bytes ├─ all three byte-for-byte identical
    NO-STATE-ATTR   2219 bytes ┘

@376a0297's finding is **confirmed at the pixel level, on the shipped page.**

The severity is directional, and that is the whole point of this dashboard:
the failure does not degrade toward caution, it degrades toward **confidence**.
A field whose state is a typo, a renamed constant, a state this build has never
heard of, or simply never set at all, renders as **a bright, unqualified,
fully-trusted measurement.** There is no border, no muting, no badge, nothing
to notice — and nothing to notice is exactly what `measured` looks like.

Every other defect this project has hunted tonight is a display that cannot
distinguish "no signal" from "signal of zero." This is the same defect **in the
mechanism built to prevent it**: the honesty layer's own default is the
maximally-dishonest one.

Recommendation: `.value:not([data-state])` and any unrecognised value must
render in a distinct, obviously-wrong treatment. It must never be the treatment
that means *trust this number*.

# §2 — Gate item (A): the module graph, in a browser, on both origins

Real Chrome 150 over CDP. `GET /demo/` on both origins, full query string.
Not a harness, not `file://`, not a module opened by URL.

                              :8123 scatter    :8124 dynamic
    responses                      160              160
    .js requests                    21               21
    .css requests                    3                3
    NON-2xx                          0                0
    Network.loadingFailed            0                0
    js served as javascript      21/21            21/21
    uncaught exceptions              0                0
    visible panels                26/27            20/21
    rendered innerText          9,839 ch         9,473 ch
    [data-state] nodes              59               50

**No module-resolution failure exists. The page is not blank on either origin.**
The one zero-height element is `failure-state__panel` — the "this page has to be
served by the onnx-genai server" fallback — correctly collapsed to 0×0 because
the page *is* properly served. That is the fallback working, not a defect.

## 2.1 The positive control — because a green here is worthless without one

A network check that finds nothing looks exactly like a network check that
*looks* at nothing. So I injected a module import for a file that does not exist:

    ./dashboard/this-module-does-not-exist-qa-control.js
      -> 404 captured by the harness                       ✅
      -> "[Log.error] Failed to load resource: 404"         ✅

**The detector demonstrably fires on the exact failure mode being ruled out.**
The two greens above are therefore meaningful rather than merely quiet.

## 2.2 🟡 P2 — the one real console warning, and it is the honesty layer WORKING

Identical on both origins, verbatim (paths differ per model):

> `[telemetry-store] provenance table is stale: "server.model_path" is
> classified NOT_PLUMBED in telemetry-provenance.js, which expects no value at
> all from /v1/models. The server sent "…/models/qwen2.5-0.5b-scatter-v2". That
> means this field is now genuinely measured and the provenance table is out of
> date. … The value is being displayed rather than hidden, because suppressing a
> real measurement is the failure this check exists to catch -- but it is
> flagged until the table is fixed.`

This is the single best-behaved thing I have seen tonight. The table went stale,
the client **detected its own staleness at runtime**, chose the honest failure
direction, and said why. It is the mirror image of the night's recurring defect.
Action is a one-line reclassification, not a fix.

## 2.3 🔴 P1 (new) — an absolute filesystem path is rendered on the demo page

    /Users/justinc/Documents/GitHub/onnx-genai-demo/../onnx-genai/models/qwen2.5-0.5b-scatter-v2

Rendered in visible `innerText`, both origins. In front of an audience this
displays a developer's home directory, local username and an unresolved `..`
segment. Display the model **id**, not its resolved path.

# §3 — Gate item (B): the three §48 render checks

Rasterized at **14px, deviceScaleFactor = 1** — a real display, not a 2× capture
that would flatter thin borders. Screenshots decoded back through a canvas and
read as luminance rows, so these are measured pixels, not CSS readings.

## 3.1 ✅ Does 3px double smear into a thick solid?  NO.

    stale           y=33  ###.####.####.####.####.####.####.###..###
    unavailable     y=33  ##.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#
    not-applicable  y=32  ##########################################
                    y=33  (gap — no ink)
                    y=34  ##########################################

**Two solid rules separated by a clean 1px gap row.** The double survives
rasterization intact. And the three patterns are structurally different in kind,
not merely in degree: **dashed = period-5 runs, dotted = period-2 dots,
double = two rules + gap.**

*I got this wrong on the first pass twice and both were instrument errors.*
First I scanned fixed 30px bands and read each band's top row — which is the
**previous** specimen's border, so I was comparing dashed-against-dotted while
labelling it dotted-against-double. Then my verdict heuristic demanded ≥3 inked
rows and called the correct 2-rows-plus-gap rendering "smeared to solid." **The
raster was right both times; my reading of it was wrong.** Isolating one
specimen per capture fixed both.

## 3.2 ✅ Do the absence glyphs hold their reserved width beside a number?  YES.

    measured        '123.4'  42.00px
    unavailable     '—'      42.00px   delta 0.00
    not-applicable  'n/a'    42.00px   delta 0.00
    pending         '···'    42.00px   delta 0.00
    .value          min-width 45px · font-variant-numeric: tabular-nums

Zero reflow when a value drops out. `tabular-nums` plus an explicit `min-width`
is exactly the right mechanism and it is already in place.

## 3.3 ✅ Does any of it survive a compressed screenshot?  JPEG q40: all 3 pairs survive.

No pair collapsed to an identical raster signature. Stated conservatively:
**no pair became identical.** Compression does inject noise into these 1px
features, so this establishes the distinction is not *destroyed* — it does not
establish it stays comfortably legible in a heavily recompressed slide.

## 3.4 Consequence for the grayscale finding

The four absence tokens measuring 1.001:1 in grayscale is confirmed as serious —
colour carries nothing. But the border channel **does** carry the whole signal
and it holds up: three structurally distinct patterns, surviving both real-display
rasterization and JPEG q40, with `measured`/`pending` carrying no border at all.
**The five-state system is sound. Its DEFAULT (§1.5) is the defect.**

# §4 — What actually mounts, counted on the running page, both origins

Method: `import('./dashboard/index.js')` **executed inside the loaded page**, plus
a DOM census. Not a grep, not a worktree read — the count, from the page a
visitor gets. Per the lead's own acceptance criterion.

## 4.1 ✅ panels: 5. The prefix-cache panel is NOT registered, on EITHER origin.

    import('./dashboard/index.js').PANELS
      :8123 scatter  ->  5  [throughput, scheduling, kv-memory, requests, system]
      :8124 dynamic  ->  5  [throughput, scheduling, kv-memory, requests, system]

**The `panels: 6` alarm is false at this HEAD.** There is no prefix-cache import,
no registration, no mount, on either origin. The five deletions are already done.
**Nobody should delete anything.**

## 4.2 ✅ The meta description no longer advertises prefix caching.

> "Live demonstration of continuous batching and paged KV block allocation in
> onnx-genai. Every number carries its own provenance."

Both origins. Also closed.

## 4.3 ✅ No fabricated `0`/`0.0` rendered at any state, either origin.

Answering the direct ask: every `[data-state]` node whose text is `0` or `0.0` —
**zero matches.** Field-state census: scatter `measured:14, unavailable:43,
pending:1`; dynamic `measured:8, unavailable:40, pending:1`.

# §5 🔴 P1 — THE PANEL WAS CUT. THE HONESTY REGISTER STILL VOUCHES FOR THE FEATURE.

The page renders a table captioned:

> *"Every value this page can display, where it comes from, and whether the
> server genuinely measures it."*

45 rows. Columns: Metric · Source · Status · Evidence. **Three of those rows are
prefix-cache metrics, and all three carry Status = "Measured by the server", in
visible text, on BOTH origins.**

    Prefix cache hits       /v1/debug/kv   Measured by the server
    Prefix-cache hits       /metrics       Measured by the server
    Prefix-cache hit rate   /v1/debug/kv   Measured by the server

**The justification below carries NO timing number, deliberately.** An earlier
draft of this section cited a shared-prefix slowdown percentage and my own
Scenario B TTFT delta. Both are struck. The slowdown figure was **withdrawn by
its author** — an interleaved re-run came back with the **opposite sign** — and
this machine's ambient noise floor is **9.8% on a byte-identical binary**, which
exceeds either effect. I took my own TTFT figure at loadavg 30.40 while
refusing to publish timings elsewhere in this same report. *A number I would not
accept from someone else does not become admissible because I measured it.*

**The verdict rests entirely on structural evidence, which no re-run on a
quieter machine can overturn:**

1. **The register's own evidence column**, rendered on screen: the rate *"emits
   a literal 0.0 when `lookups == 0`, so an undefined rate and a genuine 0% are
   the same bytes."* A defect visible in source, with no stopwatch.
2. **The counter cannot distinguish reuse from no-reuse**: twelve requests with
   six deliberately unique prompts produced twelve hits and a 0.9375 rate.
3. **The mechanism, traced in source**: the hit path returns a length without
   setting the loaded prefix, so prefill recomputes everything regardless.

These are claims about **the instrument**, not about **the world** — and that is
why they hold at any load average.

**Everyone hunted the panel. The panel is gone. The claim survived in the
register — and the register is the artefact a sceptical visitor consults
precisely because they don't trust the panels.** Cutting the panel removed the
number; it did not remove the certification. **This is D111 one more time: we
deleted the thing that looked like the defect.**

## 5.1 And row 3 refutes itself INSIDE THE ROW

    STATUS   : Measured by the server
    EVIDENCE : "...hits/lookups, but emits a literal 0.0 when lookups == 0, so an
                undefined rate and a genuine 0% are the same bytes. The store
                corrects this where the denominator is still in scope."

**The Status column certifies what the Evidence column refutes, in the same row,
on screen, at the same instant.** Row 1 is the same shape and its prose is
genuinely excellent — *"The counter is honest; the path simply never hits it"* —
but it sits under **"Measured by the server."** A reader who scans the Status
column (which is what a status column is *for*) gets the exact opposite of what
a reader who reads the Evidence column gets.

**Fix: Status must become `not-applicable` for all three, with the existing
evidence prose kept verbatim — it is already correct and already written.**

## 5.2 🟡 P2 — 24 cells render `file:LINE` citations to a visitor

The register renders 24 `<td>` cells containing `admin.rs:132`,
`metrics.rs:136-138`, `admin.rs:126-130` and similar — **user-visible line
numbers on the shipped page.** Line-number citations rotted eight times tonight
in files we control and re-read constantly; these are frozen into a rendered
artefact nobody re-reads. Cite `file` + `symbol` — the register already names
the symbols (`snapshot.prefix_cache_hits`, `prefix_cache_hit_len`), so the
durable half is present and the rotting half can simply be dropped.

## 5.3 🟡 P2 — a raw enum token leaks into user-facing text

Status census across the register: `Measured by the server` ×25,
`Exists in the process, not yet exposed over HTTP` ×10, `—` ×5, and
**`STRUCTURALLY_BYPASSED` ×1** — an internal classification constant rendered
verbatim to a visitor beside nine carefully-written English statuses.

## 5.4 🟡 P2 — `scheduling` is registered but does not mount on the dynamic origin

    scatter :8123   scheduling panel present, 554px tall
    dynamic :8124   scheduling panel ABSENT from the DOM

Registered 5, mounted 5 on scatter, mounted 4 on dynamic. This may be correct
(scheduling is a batching concept), but it is **undeclared**: a panel that
vanishes entirely is the one absence the honesty layer never gets to narrate.
Per the project's own D207 — every absence keeps its frame and states its
reason — a missing panel should say why it is missing.

# §6 — D232 RE-OPENED, AGAINST MY OWN REFUTATION

@c0de4c2e asked one narrow question about my §1.3 refutation: *do
`not-applicable` and `unavailable` render at DIFFERENT foreground colours on
screen?* They were right to ask, and **the answer is in my own §1.2 table, where
I failed to read it:**

    unavailable      color = rgb(117, 132, 147)
    not-applicable   color = rgb(117, 132, 147)   <- BYTE-IDENTICAL

**They are the same colour. Not similar — identical, to the last of 8 bits.**

So the honest disposition splits, and I was wrong to state it as a flat
refutation:

- ❌ **The claim "`not-applicable` renders pixel-identically to `unavailable`"
  is FALSE.** The border channel separates them: `1px dotted` vs `3px double`,
  confirmed in real pixels and surviving JPEG q40.
- ✅ **The claim "`not-applicable` takes `--og-unavail-*` instead of its own
  `--og-na-*`" is TRUE, and my measurement confirms it rather than refuting it.**
  `--og-na-fg` resolves to `#7e8fa0` on `:root` — the token exists, it is
  brighter, and **the state's own bare selector does not use it.**

**I answered "are they distinguishable" when the filed defect was "does the
state use its own token."** Those are different questions and I collapsed them.
The state *is* distinguishable — through **one channel only**, the border — and
the redundancy the design intended is genuinely absent.

**That makes the fix worth doing, on my measurement rather than despite it.**
With the four absence tokens at 1.001:1 in grayscale, the border is not
reinforcement — it is the entire signal, single-channel, no redundancy. Pointing
`[data-state='not-applicable']` at `--og-na-*` restores the second channel and
costs one line.

> **The method lesson is mine and it is the same one I have caught twice tonight
> in my own instruments: a measurement refutes the claim it was aimed at, not
> every claim in the neighbourhood. I aimed at "indistinguishable", hit it
> cleanly, and reported a verdict on a broader statement than I had tested.
> @c0de4c2e's narrow question was the control I had not run.**

**@086345a5 withdrew D232 citing my pixels before this correction landed.
That withdrawal was made on my error and should be reversed — the colour half
of their finding was correct all along.**

---

## §7. AC189 — the scheduling panel, measured against HEAD. **P1, intermittent, reproduced 5 of 6.**

**Asked by @376a0297:** *"load `/demo/`, fire four concurrent requests, and the scheduling
panel must read `n of 4` — not `n in flight`, not an em-dash. If it reads wrong I would
rather hear it from you than from a reviewer."* They asked to be falsified. They are.

### §7.1 The result

```
server   :9351  binary ada195f5  governor=3 (post-fix)  --demo-assets-dir = CLEAN HEAD TREE
assets   /tmp/qa-head-assets @ 5893ce3c   porcelain 0
identity served telemetry-store.js == git show HEAD:  ✅ and != worktree ✅ (discriminating)

WIRE, sampled in the SAME evaluate call as the panel text:
    batch_in_flight = 4      batch_capacity = 4      active_batch_size = 4
    sustained 21 s, 4/4 completions, every sample

PANEL, same instant:
    "··· requests  sequences  —
     The count is real; the limit isn't reported, so there is no
     percentage to show. Without a limit you are watching a queue
     length, not batch occupancy."

RUNS ON HEAD ASSETS:  STUCK 5  /  RESOLVED 1
```

**The failure is not an em-dash. It is a fabricated explanation.** The panel states
*"the limit isn't reported"* while `batch_capacity: 4` is in the very response the page
fetched, 4 times a second, 117 times in 30 s, with zero failures and zero console errors.

> **This is the honesty layer's own defect class, committed by the honesty layer, in its own
> voice. An em-dash says "I don't know." This says "the server didn't tell me" — a specific,
> confident, checkable claim ABOUT THE SERVER, and it is false. A wrong number invites doubt;
> a wrong *reason* closes the question, because it already explains itself.**

### §7.2 What is NOT the cause — each excluded by measurement, not by argument

| candidate | measurement | verdict |
|---|---|---|
| server doesn't serve it | `batch_capacity` 4, `batch_in_flight` 4, `active_batch_size` 4 | ❌ excluded |
| page polls the wrong origin | 117/117 requests to the page's own origin, hosts preserved | ❌ excluded |
| poller dies or backs off | 4.0 req/s flat across six 5 s windows, 30 s, no decay | ❌ excluded |
| fetch failures / JS errors | `Network.loadingFailed` 0, console errors 0 | ❌ excluded |
| catalogue mis-binding | executed: `batch.capacity`→`/v1/status`,`batch_capacity`,MEASURED | ❌ excluded |
| duplicate `batch.capacity` key | now **1** occurrence in HEAD; symbol-anchored entry survived | ❌ fixed |
| my under-priming | reproduced at 6 s **and** 20 s priming | ❌ excluded |

**It is a race.** Two byte-identical runs — same binary, same HEAD assets, same URL, same
timing — gave `measured 3 / pending 11` and `measured 16 / pending 1`. When it loses,
~13 fields never leave `pending` and the panel prints the false reason.

### §7.3 🔻 A retraction of my own, and it invalidates browser evidence taken by anyone tonight

I have stated, and broadcast, that **"assets are read from disk per request, so CSS/JS are
always at HEAD even under a stale binary."** **That is false and I am withdrawing it.**

```
served by :8133   48,529 B   ==  WORKING TREE   (byte-identical)
git show HEAD:                45,832 B          (53 uncommitted lines behind)
```

`--demo-assets-dir` points at the **working tree**. The demo server therefore executes
**whatever is on the author's disk at the instant of the request**, including files another
agent is mid-edit. `telemetry-store.js` was dirty throughout my first six runs, which is
precisely why they contradicted each other — I was measuring a file being written while I read it.

> **A DEMO SERVER SERVES THE DESK, NOT THE BRANCH. Every browser verification tonight —
> mine, and everyone's — certifies an uncommitted tree. The reviewer clones and gets HEAD.**
> This is @1cb42f0e's `readFileSync` finding and @c7a654ed's clean-vs-dirty ruling arriving on
> the one surface we all agreed was the acceptance standard. **The fix is construction, not
> discipline: point `--demo-assets-dir` at a detached worktree pinned to a SHA, and check the
> served bytes against `git show HEAD:` in the same invocation — an asset identity check,
> exactly parallel to @1cb42f0e's binary identity check, and it must be DISCRIMINATING (differ
> from the worktree) or it proves nothing.** Every §7 number above was taken that way.

### §7.4 Two smaller findings from the same session

- **`batch.in_flight` has ZERO consumers in shipped dashboard code.** The raw uncllamped
  in-flight count was ordered, served, and registered in the catalogue — and no panel reads it.
  The fix landed at three layers of four. `scheduling.js` still takes its numerator from
  `batch.active_size` (`/v1/debug/kv`).
- **`.value__num--not-applicable` is never emitted.** Zero instances across 3 origins × 3
  scenarios (`:8133`, `:8134`, `:9242` × continuous-batching, paged-kv, memory-pressure).
  **@376a0297's D266 test cannot be run as specified, and its unrunnability is the answer:**
  neither they nor @0837fdf9 is right, because the rule is live in CSS and nothing renders an
  element matching it. That is @12e42da8's *styled-but-never-emitted* gap, second instance,
  measured. The only `not-applicable` element on the page is a `scenario-switcher__note`.

---

## §8 — Scenario switching across origins, and the missing-assets failure that is not a page

Assigned directly by @12e42da8: *"START BOTH, SWITCH SCENARIOS, CONFIRM A DASHBOARD AND NOT
THE 404 PAGE."* Run at HEAD `0c387cf2`, Chrome 150.0.7871.187 over CDP, loadavg **37.85**.
Servers: `:9452` (`qwen-scatter`) and `:9451` (`qwen-dynamic`), both on the post-`d08d44b8`
binary, both launched with `--demo-assets-dir` pointed at a detached worktree pinned to
`fb718b2c`. Harness `/tmp/qa_switch3.mjs`.

### 8.1 Result — the gate item PASSES on a correctly-launched pair

| tab | lands on | panels | body len | verdict |
|---|---|---|---|---|
| Continuous batching | `127.0.0.1:9452` | 36 | 12455 | DASHBOARD |
| Paged KV block table | `127.0.0.1:9451` | 36 | 12136 | DASHBOARD |
| Memory pressure | `127.0.0.1:9451` | 36 | 12076 | DASHBOARD |

Two of the three tabs are **cross-origin** — they navigate to the *dynamic* origin, not the
one serving the page. That is the mechanism behind the hazard: the landing pane is served by
the scatter origin and looks perfect regardless of how the dynamic origin was launched.

### 8.2 The control, and why the first two versions of this test were worthless

**Arm B replaced the dynamic origin with `:8151`, which was launched WITHOUT the asset flag:**

| tab | lands on | panels | verdict |
|---|---|---|---|
| Continuous batching | `127.0.0.1:9452` | 36 | DASHBOARD |
| Paged KV block table | `null` | **0** | **NOT-A-DASHBOARD** |
| Memory pressure | `null` | **0** | **NOT-A-DASHBOARD** |

**My first two attempts at this test were green in BOTH arms — including the arm engineered to
fail.** I constructed each scenario URL by hand from the topology template instead of clicking
the tab. Because I wrote the origin into the URL myself, every probe landed on the flagged
scatter origin and the broken origin was never contacted. The test named *scenario switching*
and measured *string formatting*.

> That is @086345a5's **R15** reproduced exactly, by me, in my own lane, ninety minutes after I
> read it. R15 objects to `QA-PLAN.md` B1 instructing the tester to paste `?scenario=…` and
> "confirm it opens on the right panel" — because pasting the URL **silently substitutes for
> the navigation** and cannot fail. I now have that as a measurement rather than an argument:
> **URL-constructed = 3/3 green on a origin that 404s; tab-clicked = 2/3 red on the same
> origin.** Same servers, same scenarios, same minute. **The substitution does not weaken the
> check, it inverts the result.**
> **A control that does not fire has not passed. It has abstained, and it looks identical.**

### 8.3 🔴 P2 — there is no missing-assets page. There is nothing at all.

`--demo-assets-dir` absent does not produce an explanatory in-product page:

```
GET http://127.0.0.1:8151/demo/    ->  HTTP/1.1 404 Not Found
                                       content-length: 0        <- ZERO BYTES
GET http://127.0.0.1:9452/demo/    ->  HTTP/1.1 200 OK          content-length: 5917
```

With a zero-byte body the **browser** supplies the page. `location.origin` reads `null`; the
audience sees Chrome's built-in network-error screen, **in the operating system's locale** —
on this machine it rendered in Chinese — with the full raw URL and query string printed on
screen, no product branding, no explanation, and **no route back to the dashboard**.

The standing description of this failure ("shows the missing-assets page") implies the product
explains itself. It does not. **Severity is P2 and not higher only because `run-demo.sh` gets
it right; on a hand-started server it is the worst-looking thing in the demo.**

### 8.4 What is NOT claimed

- I did **not** find a defect in `run-demo.sh`. It passes `--demo-assets-dir` on **both**
  launches (`:238`, `:247`), and `SCRIPT_DIR` (`:31`, `cd … && pwd`) is **absolute**, so
  @12e42da8's absolute-path requirement is **already satisfied in the script**. The README is
  a separate artifact and I did not check it — @732c7548's item.
- I did not test a launch where the **scatter** origin lacks the flag. That failure is not
  silent — it appears on the landing page — so it is the less dangerous direction.
- Arm A proves the tabs work between **two origins that both have assets**. It says nothing
  about whether the two panes should differ in batching behaviour (see `perf-baseline.md` §11).
