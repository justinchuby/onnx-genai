
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

---

## §9 — The batch caption, read off the screen

@086345a5 reported that `telemetry-provenance.js` was fixed to `'Effective batch capacity'`
while `scheduling.js` hardcoded a `'Batch limit'` override at the render site, with
`honesty.test.js` pinning the wrong string. Three agents had verified the label by importing
the module; all three readings were correct and none of them was about the screen.

**Nobody had looked at the pixel. This is that measurement.**

Method: fresh detached worktree pinned to `664b3721` at `porcelain 0`, a server launched with
`--demo-assets-dir` pointed at it (so the bytes served are the committed bytes, not a working
tree eleven agents are writing to), Chrome 150 over CDP, `continuous batch driver enabled
max_batch=4`.

```
RENDERED document.body.innerText — 12,400 chars
  "Batch limit"                 ->  0 occurrences
  "Effective batch capacity"    ->  1 occurrence
  CONTROL "requests"            ->  6 occurrences   <- the search works, the page is populated
  wire /v1/status               ->  batch_capacity=4  batch_in_flight=0
```

**The caption defect is closed at the layer that ships.** `Batch limit` is painted nowhere.

The control is not decoration: a zero from a string search on a page that failed to render is
byte-identical to a zero from a page that rendered correctly. Six hits on a word I did not
care about is what separates *the defect is gone* from *nothing loaded*.

### 9.1 A detail worth keeping — the panel does not use a caption here at all

The scheduling panel renders capacity as a **denominator in a phrase** — `… of 4 requests` —
not as a labelled field. `Effective batch capacity` renders once, on the provenance surface
(`/v1/status` · *Measured by the server* · `crates/onnx-genai-server/src/…`).

That is stronger than the fix that was asked for. @0837fdf9's D275 hazard was that a caption
naming the *raw ceiling* sits beside a value that is `min(max_batch, max_queue_depth)`, so a
saturated server reads as 25% busy under a caption asserting it is at its limit. **A
denominator cannot make that claim.** `X of 4 requests` states a ratio and names no ceiling.

### 9.2 What is NOT claimed

- `max_queue_depth >= max_batch` on this server (`batch_capacity=4` = `max_batch=4`), so I did
  **not** exercise the clamped case where the two values diverge. I verified the caption, not
  the arithmetic under a low queue depth.
- One rendered occurrence of the string is consistent with the catalogue being read; I did not
  prove the panel *sources* its caption from the catalogue rather than coinciding with it.
- Measured at `664b3721`. The tree moves at roughly a commit a minute; this is a snapshot.

---

## 10. The full browser pass on the committed launcher

Assigned as the last gate item: four agents independently closed their reports with the same
sentence — *nobody has opened this dashboard in a browser* — and all three reviewers refused to
let their sign-off substitute for it. This section is that pass. Observations, not verdicts.

### 10.0 The stale-binary trap is real, and the launcher cannot see it

`run-demo.sh:207` builds **only if the binary is missing**. It never checks whether the binary is
older than the source:

```sh
SERVER_BIN="${CARGO_TARGET_DIR:-${REPO_ROOT}/target}/release/onnx-genai-server"
```

Two release binaries existed, 36 minutes and 44 KB apart. The one the launcher resolves was built
at 03:31; **six Rust commits landed after it**, including `1e1b2a82` (disclose which decode driver
is running) and `4a059a93` (publish the pool size) — both change served fields. A stale binary
launches, binds, and answers `200` while omitting fields entirely, which is indistinguishable from
a front-end bug. I rebuilt (24.2 s) before measuring anything; every number below is from a binary
timestamped 04:10:57 with zero commits after it.

**P2, and it belongs to the launcher, not the dashboard:** the freshness check should be
`-nt` against the source tree, not `-f` against the binary path.

### 10.1 Both servers, dashboard not 404

Launched via the committed `run-demo.sh` with `SCATTER_PORT=9123 DYNAMIC_PORT=9124`.
Both `/demo/` and `/v1/status` return `200`. Dashboard renders on both; **not** the 404 page.

The distinction the Lead asked for: a *mistyped asset directory* and *not-configured* both return
404, and they look identical. They are separable on the wire — the not-configured arm returns
`404` with **`content-length: 0`** (verified on `:8151`, launched without `--demo-assets-dir`).
A mistyped directory returns a 404 with a body. Neither occurred here.

### 10.2 Four of the five ruled states are exercised; one I could not induce

⚠️ **This subsection replaces an earlier, wrong version of itself. See §10.9 for the retraction.**

The ruled vocabulary is `measured, pending, stale, unavailable, not-applicable`
(`dashboard/state-vocabulary.test.js:28`). Counts over **all** state-bearing elements
(`n=94`, `n=93`, `n=94` for the three scenarios), not a filtered subset:

| state | continuous-batching | paged-kv | memory-pressure |
|---|---|---|---|
| `measured` | 30 | 23 | 30 |
| `unavailable` | 57 | 58 | 57 |
| `pending` | 6 | 11 | 6 |
| `stale` | 0 | 0 | 0 |
| `not-applicable` | **0** | **0** | **0** |

`stale` is 0 in all three **because a healthy server never produces it** — that is correct
behaviour, not an absence. **I induced it and it works** (§10.9): killing the origin the page
polls moves `connected` 1→0, `stale` 0→2, `measured` 30→21, and surfaces
*"Disconnected — last reading 20s ago"*. The vocabulary is live.

**`not-applicable` is the one state I never observed and could not induce.** The source comments
call it *"an intentional gap"*. I have no scenario that declares one, so I cannot say whether it
is reachable or dead — only that **no scenario a reviewer will open ever renders it**.

### 10.3 Two absence states are distinguished by glyph only, not by colour

| state | glyph | colour |
|---|---|---|
| `pending` | `···` | `rgb(120,140,162)` |
| `unavailable` | `—` | `rgb(114,134,157)` |

**The colour delta is 6/255 on each channel — imperceptible.** The entire distinction rests on the
glyph. This is not a bug; it is a design fact nobody had seen, and it means any future change that
normalises the glyph silently collapses two states into one. `pending` is also transient (gone by
~2 s), so in practice a viewer sees one absence glyph.

### 10.4 Numbers move — and it took three broken instruments to establish it

Under four concurrent 300-token generations, with the wire at `inflight=4 util=1.0`:

```
-  "256"      +  "252"     free KV blocks
-  "0"        +  "4"       in-flight
-  "0"        +  "4"       batch occupancy
```

The page polls hard and correctly: **131 non-asset requests in a 15 s window**
(`/v1/status`, `/health`, `/v1/debug/kv` at ~4 Hz). Panels mount, values track the wire,
nothing white-screens.

**⚠️ The methodology warning, which is the most useful thing I learned tonight.** I reached
"the dashboard is frozen — P1" three separate times, on three different instruments:

1. `[data-field]` selector — matched **3** elements, all static server-identity fields that
   *should* never move. `CHANGED=0`.
2. In-page `fetch()` load — never reached the server; wire stayed `inflight=0`. `CHANGED=0`.
3. `[data-state]` selector, 51 cells, **with a positive control that passed** — I injected a
   sentinel into a cell and the sampler reported exactly 1 change. `CHANGED=0/51` at `util=1.0`.

The third is the dangerous one. **The positive control proved the sampler could detect a change in
a cell it had already selected. It did not prove the sampler selected the cells that carry live
values.** A positive control validates the *detector*; it says nothing about the *corpus*. Passing
it made me more confident in a conclusion that was still wrong.

What broke the tie was abandoning selectors entirely and diffing `document.body.innerText` — a
measurement with no corpus assumption at all. **If I had filed at any of the three stops, I would
have reported a P0 against a working dashboard**, and the fix would have gone to @c8d9a40e or
@bb2ee824 for a defect that does not exist.

### 10.5 Scenario switching by clicking the real tabs

Following actual `href`s (not URLs I construct — that is the R15 error that gave me a false green
in §8):

| tab | origin | panels | body |
|---|---|---|---|
| Continuous batching | `:9123` | ✅ | 11,920 chars |
| Paged KV block table | `:9124` (cross-origin) | ✅ | 11,811 chars |
| Memory pressure | `:9124` (cross-origin) | ✅ | 11,811 chars |

3/3 render. 2 of 3 are genuinely cross-origin — the case that fails when only one server is up.

### 10.6 Console and network

**0 console errors, 0 failed requests** on all three scenarios, across every probe in this section.
No refused module, no unknown-state warning, no 404 on an asset.

### 10.7 The `file://` fallback is correctly hidden

`display: none`, bounding rect `0×0`, on both instances. It does **not** leak onto an HTTP-served
page. One cosmetic note: the fallback's `<h1>` is **first in DOM order**, so
`document.querySelector('h1')` returns *"This page has to be served by the onnx-genai server"*
rather than the visible *"onnx-genai"*. Hidden from users and from screen readers
(`display:none` is not announced), but it will mislead any tooling that reads the first `h1`.
**P3.**

### 10.8 Continuous batching is inactive, and the engine now says why

The launcher log states the cause directly, rather than leaving it to be inferred from the model's
name as I had to in `perf-baseline.md` §11:

```
continuous batching is INACTIVE ... The engine refused it: continuous batching
requires a STATIC-CACHE or shared-buffer past/present model
```

This closes the causal half I explicitly declined to assert there.

### What this section does NOT claim

- Rendered on one engine (Chrome/CDP) at one viewport. No cross-browser or responsive check.
- `not-applicable` is shown **unreached in these three scenarios and uninducible by me**. I did not
  prove it is unreachable in principle, and I did not read the emitter to find out.
- The colour delta is measured from computed style, not from a screenshot; I did not test against
  any contrast standard.
- Numbers were verified to *move and track the wire*, not to be *arithmetically correct*.
- Idle steady state is ~42 em-dashes to ~6 measured values. Whether that is the intended density
  for a demo is a design question, not a defect I can assert.

---

## 10.9 Retraction: I reported the five-state model as three. It is four, and the fifth is not a defect.

**⛔ WITHDRAWN — the claim in the first committed version of §10.2 (`643f1e72`), that the
five-state model "renders as three" and that the two invisible states were `not-applicable` and
`bypass`.** Both halves were wrong. This is the fifth instrument artifact in one evening's work on
the same page, and it is the only one that reached a commit and a report to the Lead.

**What was wrong, precisely:**

1. **`bypass` is not one of the ruled five.** The ruled vocabulary is
   `measured, pending, stale, unavailable, not-applicable`. I substituted a styled class I had
   seen in CSS for a state I had never checked, and never counted `stale` at all.
2. **The density figures were a filter artifact.** I reported `measured=2, unavailable=42` from a
   sampler that excluded elements with more than two children. The true counts are
   `measured=30, unavailable=57` of `n=94`.
3. **`stale` is emitted, and the disconnect path works well.** Killing the origin the page polls:

   ```
   connected  1 -> 0
   stale      0 -> 2
   measured  30 -> 21      (nine values correctly demoted)
   unavailable 57 -> 64
   banner:   "Disconnected — last reading 20s ago"
   polls continue (8 -> 14 attempts), so the loop survives its own failures
   ```

**How the error happened, because the mechanism matters more than the correction.** My first
attempt at this experiment killed **the wrong process**. `lsof -ti tcp:9123` returns *every* socket
on the port, including established client connections; the first line was a client holder
(`21707`), not the listener (`53418`). I killed a socket, the server never went down, the dashboard
correctly kept showing `measured` and `connected`, **and I wrote that up as a P1: "the dashboard
does not detect server loss."**

What caught it was not insight — it was a **stray inconsistency**: the launcher refused to restart
with *"port 9123 is already in use"* after I believed I had killed it. The finding was already
written.

**The control I had skipped is one line**: confirm from *outside the browser* that the server
actually refuses connections before interpreting anything the browser shows.

```
CONTROL curl after kill  -> 000 REFUSED
CONTROL listener present -> NONE
```

With that control in place the experiment inverted completely and the product came out **well** —
not merely not-broken, but with a demotion path and a human-readable banner that nobody had seen
until now.

**The pattern across all five artifacts tonight was identical: I kept validating the detector and
never validating the target.** The sampler could see an injected change (but not the live cells).
The load generator ran (but never reached the server). The kill command succeeded (but not against
the listener). **Every one of those is a green light on the instrument and silence about the
subject.** For a browser check specifically: *before believing any absence on screen, prove the
condition you think you created actually exists on the wire.*

**Net effect on the gate: the dashboard is in better shape than my first report said.** One state
(`not-applicable`) is unobserved and uninducible by me; four of five are exercised; the disconnect
path is genuinely good.

---

## 10.10 Re-run on a re-armed binary, with identity asserted — and the definitive state-appearance answer

The Lead re-issued the browser order with a guardrail I had **not** satisfied: *assert an identity
marker in the payload before trusting any reading*. Re-running the whole pass exposed that
**guardrail 1 had re-armed while I worked**.

### The binary went stale underneath a passing result

My §10 measurements ran on a binary built at `04:10:57`. While I was writing them up,
`f110647c` landed at `04:36:36` — *"kv: separate an absent block mirror from an empty block
window"*, i.e. **precisely a field-state distinction, which is what §10 reports on**. Two binaries
again differed: `04:10:57 / 29,495,696 B` in the demo tree and `04:26:15 / 29,521,056 B` in the
sibling checkout. Everything below is on `04:38:39 / 29,520,576 B`, which post-dates every
`crates/` commit.

**The lesson is not "rebuild". It is that a passing browser result has a shelf life measured in
minutes on this branch, and nothing in the harness announces its expiry.**

### Identity, asserted rather than assumed

A port answering `200` proves *a* server is there, not *which*. Chain now closed:

| | `:9611` | `:9612` |
|---|---|---|
| LISTEN pid | 89947 | 89948 |
| executing | `…/onnx-genai-demo/target/release/onnx-genai-server` | same |
| same inode as my build (`-ef`) | ✅ | ✅ |
| `node_id` | `node-3c7e9f3fbadc991a` | `node-591ce8fb5f4f237d` |
| `model_id` | `qwen-scatter` | `qwen-dynamic` |

Distinct `node_id`s and distinct models — so this is not one server measured twice, and not
someone else's. Ports moved to 9611/9612 to stay clear of @c8d9a40e.

**Guardrail 3 reproduced exactly:** `/status` → **404**, `/models` → **404**;
`/v1/status`, `/v1/models`, `/v1/resources` → **200**. The reviewer's near-miss is real: an
unversioned path 404s in a way that looks like "this server has no API".

### 🔑 Do the five states look different on screen? Four do, and by TEXT — not by colour.

Forced-rendered under the real stylesheet, then confirmed against naturally-occurring elements:

| state | rendered as | colour |
|---|---|---|
| `measured` | the value — `qwen-dynamic` | `rgb(230,237,243)` |
| `stale` | **`qwen-dynamic · 12s old`** — keeps the value, appends age | `rgb(127,145,166)` |
| `pending` | `···` | `rgb(120,140,162)` |
| `unavailable` | `—` | `rgb(114,134,157)` |
| `not-applicable` | **never rendered in any scenario** | `rgb(133,151,171)` |

**The four observable states are genuinely distinguishable, and well designed for it.** `stale` is
the best of them: it does not blank the value, it keeps it and timestamps it, plus a page banner
*"Disconnected — last reading 12s ago"*. A reader can tell all four apart at a glance.

**⚠️ But the discrimination is carried entirely by text. Colour does almost nothing:** the four
absence states span `rgb(114–133, 134–151, 157–171)` — adjacent pairs differ by as little as
**4–7/255**. `pending` vs `stale` is 7/5/4. **Any future change that normalises the glyph or drops
the age suffix collapses three states into one indistinguishable grey**, and
`state-vocabulary.test.js` would stay green, because it asserts the five renderings are *distinct
values*, not that they are *perceptible*. That is the gap the Lead asked me to look for — it is
real, but it is latent, not present.

**So: the model is five in the vocabulary and four on screen — not because two look alike, but
because `not-applicable` never renders.** I could not induce it; the source calls it *"an
intentional gap"*.

### Everything else, re-confirmed on the current binary

| check | result |
|---|---|
| Dashboard, not 404 | ✅ title `onnx-genai — serving dashboard`, visible `h1` `onnx-genai`, 11,920 chars |
| Scenario tabs (clicked via real `href`) | ✅ 3/3 mount, 2 cross-origin to `:9612`, **0 white-screens** |
| Numbers move | ✅ `256→252` free KV, `0→4` in-flight, `0→4` occupancy at `util=1.0` |
| Console | ✅ **0 errors** |
| Network | ✅ **0 failed requests** |

No refused module, no unknown-state warning, no asset 404.

## §11 — The path-disclosure P1, verified fixed at HEAD, and the limit of my own instrument

Scope: the Lead's P1 — *an absolute filesystem path rendered in visible text on the
demo page, on both origins*. Verified at HEAD `dd04f50f`. Servers: `:9611`
(qwen-scatter) / `:9612` (qwen-dynamic), both launched with an **absolute**
`--model` and an **absolute** `--demo-assets-dir`, so the raw material for the
defect was present in the launch.

### §11.1 Six independent channels, all clean

| channel | method | result |
|---|---|---|
| served HTML | `curl /demo/` grep `/Users/` | 0 |
| every referenced asset | follow each `src`/`href`, grep | 0 |
| `/v1/status`, `/v1/models`, `/v1/resources` | path-shaped values | 0 |
| rendered DOM, visible leaves | CDP, `getBoundingClientRect().height > 0` | 0 |
| rendered DOM, **hidden** leaves | same sweep, height `=== 0` | 0 |
| every attribute value | `/Users/` over all attributes | 0 |

Both origins, with and without the topology query string.

**Positive control, because a zero is worthless without one.** Injected a real
absolute path into the live page as a visible node and a `display:none` node.
Detector: visible `0 -> 1`, hidden `0 -> 1`. Both channels fire. The zeros above
are readings, not blindness.

### §11.2 Two layers, and both are real

**Server (Rust).** `routes/admin.rs:37` redacts with `path.file_name()`. Measured
on the wire: launched with `--model /Users/justinc/.../models/qwen2.5-0.5b-scatter-v2`,
`/v1/models` answers `"path": "qwen2.5-0.5b-scatter-v2"`. Basename only. Pinned by
`tests.rs:4230 model_paths_never_disclose_more_than_the_basename`, on every bind
address.

**Client (JS).** `dashboard/model-path-disclosure.test.js` — it stops asking for
the field at all.

### §11.3 The guard has teeth — mutation-tested, not assumed

Three guards on this branch passed green while blind to the thing they protect.
This one is not one of them. Run against **committed bytes** (`git archive HEAD`),
reintroducing the disclosure by two different routes:

```
BASELINE                          pass 5  fail 0
M1  visible definition            pass 3  fail 2   RED
M3  screen-reader sentence only   pass 4  fail 1   RED
RESTORED                          pass 5  fail 0
```

M3 is the one that matters: a leak that reaches only the spoken summary, which no
screenshot review and no visible-text sweep would ever catch. The guard sweeps
attribute values precisely because the original defect was written three times per
field (`textContent`, `title`, `aria-label`).

### §11.4 A suspicion of mine, tested and withdrawn

I suspected the guard could be neutered by making its fixture realistic — the real
wire carries a basename, so a "realistic" fixture would contain nothing to find,
and the guard would go green with the disclosure code live. **Measured, and it is
false:**

```
disclosure LIVE, fixture = absolute path   pass 3  fail 2
disclosure LIVE, fixture = real basename   pass 2  fail 3   <- MORE red, not green
```

`:155` asserts *the `/Users/` predicate must be able to match*. The guard refuses
to run against a fixture that cannot expose the defect, so sanitising it fails
loudly instead of silently. That is the anti-vacuity property the rest of us keep
discovering we lack, already built. Withdrawn.

### §11.5 ⚠️ The finding that is actually mine: my browser pass cannot see this regression

The two layers are each independently sufficient today, and that is the hazard.
**Because the server now redacts to a basename, an absolute path is no longer
available anywhere in the browser.** If the client fix were reverted tomorrow, the
page would render `qwen2.5-0.5b-scatter-v2` — harmless — and **every browser check
I own would stay green.**

So: the client-side guard is *load-bearing and is the only detector*, and a browser
pass is **not** evidence for this class of defect. Four agents have ranked the
browser above their own findings tonight; this is one place it must not be trusted.
Same shape as the guard-corpus problems found elsewhere on this branch, except the
instrument with the missing corpus is mine.

Neither layer may be removed on the grounds that the other covers it. Removing
either leaves zero margin, the full suite green, and the browser green.
