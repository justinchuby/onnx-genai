# onnx-genai Live Demo + Debug Dashboard — UX & Design Specification

**Author:** Designer (@0837fdf9)
**Status:** Authoritative for layout, visual language, interaction, accessibility, and the CSS/component contract.
**Reads against:** @376a0297 `demo-spec.md` (WHAT/WHY, **43** ACs) · @e00032a4 `recon-map.md` (what is actually measurable).
**Constraint:** one static page, vanilla ES modules, no build step, served same-origin at `GET /demo`.

> 🔴 **READ §17 FIRST IF YOU ARE BINDING DATA.** The Field shape changed five times in the first hour. **§17 is the only current statement of it** and supersedes §3.2, §4.1 and §15 wherever they differ. Those sections are kept for their reasoning, not their field lists. Five states: `measured` · `pending` · `stale` · `unavailable` · `not-applicable`. Attribution is three separate keys — `source` (badge class), `endpoint` (curl-able), `server` (which engine). **The word `origin` is retired**; it was ratified twice with two different meanings.
>
> Normative beyond this document: `examples/serving-dashboard/CONTRACT.md` and `telemetry-field.js`. **Code on disk wins over prose, including mine.**

> **How to use this document.** §1–§3 are the *unblocking trio* — layout skeleton, design tokens, panel contract. Two developers can start from those alone. §4 is the honesty design language and is **non-negotiable**; read it before writing a single render function. §5–§7 are the three scenario visualizations. §8 is accessibility. §9 is the file/ownership seam.

---

## 0. The design thesis

Three sentences, and every decision below descends from them.

1. **The product is credibility.** This page's job is not to look impressive; it is to be *believed*. Every visual affordance that makes a number look more precise than it is costs more than it earns. Design for the skeptic in the third row, not the executive in the first.
2. **Honesty must be structural, not behavioural.** A rule that says "don't render documented zeros" will be violated by a tired developer at 2am. A telemetry store whose only accessor returns `{value, state, source}` makes the violation *impossible to type*. We are choosing the second. This is the single most important decision in this document.
3. **Motion is evidence, not decoration.** The only things allowed to animate are things that actually happened: a token arrived, a block was allocated, a request was admitted. Any animation not backed by an event is a lie told in the visual channel, where people's defences are lowest.

### The one design decision that matters most

> **A zero that was measured and a zero that was fabricated must never look the same.**

The server is full of the second kind (`recon-map.md` §7.3: `kv_usage: 0.0`, `tokens_per_second: 0.0`, `batch_utilization: 0.0`, `paused_sessions: 0`, `sessions[].kv_pages: 0`, `prefix_hashes: []`). A dev binding a panel to `/v1/status` will render them as measurements without noticing — the PM calls this out as the most likely accidental AC6 violation, and they are right. §4 is the systematic defence.

**And its mirror image, which is just as important:** `prefix_cache_hits: 0, lookups: 5` is a *real measurement*. The cache genuinely did not hit. That must render as a stark **`0%`**, not an em-dash. Em-dashing an unflattering real number to spare our feelings is lying in the *flattering* direction — the worse of the two directions. **Real zeros get rendered as zeros, loudly.**

---

## 1. PAGE ARCHITECTURE

### 1.1 Viewport budget

Design target is **1280 × 720** (AC29) — the resolution of a shared screen in a meeting, which is where this will most often be seen. Max content width **1680px**, centered, with a 24px gutter. Below 1120px the panel grid collapses to a single column; the page must not break, but it is not mobile-designed (non-goal §5).

At 1280×720 with browser chrome, the usable height is roughly **610px**. That is the entire above-the-fold budget and it is the hardest constraint in this design.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ▸ APP RAIL                                                    56px   sticky  │
│   onnx-genai · demo    │ model card (collapsed) │ ● live · 4Hz │ ⌨ ? │       │
├──────────────────────────────────────────────────────────────────────────────┤
│ ▸ HERO STRIP                                                   92px          │
│   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐                │
│   │ 98.7 tok/s │ │  87.2 %    │ │  62 % KV   │ │ 6 run/2 wait│               │
│   │ aggregate ᴰ│ │ prefix hitˢ│ │ 318/512 ᴰ  │ │ scheduler ˢ │               │
│   └────────────┘ └────────────┘ └────────────┘ └────────────┘                │
├──────────────────────────────────────────────────────────────────────────────┤
│ ▸ STAGE                                                     ~380px  flexible │
│   ┌ scenario tabs ─────────────────────────────┐ ┌ scenario controls ───────┐│
│   │ ① Batching  ② Paged KV  ③ Prefix  ④ Free   │ │ N=8  stagger=400  ▶ Run  ││
│   └────────────────────────────────────────────┘ └──────────────────────────┘│
│   ┌ narration strip ────────────────────────────────────────────────────────┐│
│   │ Request #6 sent at t=2.01s — its first token arrived 0.19s later, while  ││
│   │ #1–#5 were still streaming.                                             ││
│   └─────────────────────────────────────────────────────────────────────────┘│
│   ┌ the visualization ──────────────────────────────────────────────────────┐│
│   │                                                                         ││
│   │   (swimlanes | block grid | TTFT ladder — one at a time)                ││
│   │                                                                         ││
│   └─────────────────────────────────────────────────────────────────────────┘│
├──────────────────────────────────────────────────────────────────────────────┤
│ ▸ ACTIVITY STRIP                                               72px          │  ← fold ≈ here
│   compact always-on lane view + live token counter                           │
╞══════════════════════════════════════════════════════════════════════════════╡
│ ▸ PANEL GRID              auto-fill minmax(340px, 1fr)      below the fold   │
│   [Throughput & Latency] [Scheduling] [KV memory] [Cache] [Requests] [System] │
├──────────────────────────────────────────────────────────────────────────────┤
│ ▸ HONESTY FOOTER          "what's real / what's derived / what isn't built"  │
└──────────────────────────────────────────────────────────────────────────────┘
```

**Why the activity strip exists.** The PM's arc puts the swimlane timeline always-visible, and they are right that it is the soul of the page — but at 720px the stage and a full timeline cannot both fit. Resolution: the **stage** holds whichever visualization the current scenario needs, and a permanent **72px activity strip** sits directly under it showing a compressed always-on lane view (one 6px lane per in-flight request, no labels, live). When Scenario A is selected the stage *is* the full timeline and the activity strip shows aggregate token flow instead. The visitor never loses the sense that something is running, which is the strip's only job.

### 1.2 Region specifications

#### App rail (56px, `position: sticky; top: 0`)

Left→right: wordmark `onnx-genai` + `demo` subdued · **model card trigger** · flexible gap · **connection chip** · **cadence control** · **`?` keyboard overlay trigger**.

The **connection chip** is the page's heartbeat and deserves care. It is *not* a green dot that is always green. Four states, each with a shape as well as a colour (§8):

| State | Glyph | Colour token | Label | Behaviour |
|---|---|---|---|---|
| live | ● filled disc | `--og-ok` | `live · 4 Hz` | disc performs a 1px scale pulse on each successful poll — real evidence of a real response |
| slow | ◐ half disc | `--og-warn` | `slow · 1.2s` | poll round-trip exceeded cadence; shows actual RTT |
| retrying | ◌ dashed ring, rotating | `--og-warn` | `retrying · 3s` | countdown to next attempt, always visible |
| offline | ○ hollow ring | `--og-bad` | `offline` | triggers the full-stage blocking state (§1.4) |

The pulse is the only always-on animation on the page, it is 1px, and it is driven by an actual HTTP response. Under `prefers-reduced-motion` it becomes a discrete opacity step. Nothing else in the chrome moves.

#### Model card (AC36)

Collapsed by default into a single rail-height line — because the visitor needs *identity*, not a table, in the first two seconds:

```
◈ qwen2.5-0.5b-scatter-v2 · cpu · 4096 ctx ▾
```

Clicking (or `M`) expands a 380px popover anchored under the trigger. **Popover, not a modal** — §4 interaction rule "nothing is modal". Dismisses on `Esc`, outside click, or re-trigger; focus is trapped while open and returns to the trigger on close.

```
┌ MODEL ──────────────────────────────────── ✕ ┐
│ id               qwen2.5-0.5b-scatter-v2   ˢ │
│ directory        models/qwen2.5-0.5b-…     ˢ │   ← middle-ellipsised, full path
│                                                 in title + click-to-copy
│ context length   4 096 tokens              ˢ │
│ exec provider    cpu                       ˢ │
│ ─────────────────────────────────────────────│
│ quantization     —                         ⓘ │   ← unavailable treatment
│ decode backend   —                         ⓘ │
│ ─────────────────────────────────────────────│
│ continuous batch ✓ enabled · max_batch 4   ˢ │   ← see note below
│ debug endpoints  ✓ enabled                   │
│ server version   0.x.y                     ˢ │
└──────────────────────────────────────────────┘
```

**The `continuous batch` row is my addition and I want it defended.** `recon-map.md` §0.5 establishes that continuous batching only engages on static-cache models — `tiny-llm` silently falls back to the per-request path. A visitor who runs the demo against the wrong model sees a flat batching panel and concludes *the feature does not work*. That is the worst possible outcome for a demo whose entire purpose is proving the feature works. The row must be present, and when it reads `✗ disabled`, the Scenario A stage must render a **capability notice** (§4.4) explaining that this model does not use a static cache and naming a model that does. This costs one boolean and prevents the single most damaging misreading available to a first-time visitor.

The four AC36-required fields (id, directory, context length, EP) are always *present as rows*. If a value is missing it renders `—` per §4 — **never `0`, never `undefined`, never a blank row, never a hidden row.** A hidden row is worse than an em-dash: it destroys the visitor's ability to know what they weren't told.

#### Hero strip (92px)

Four tiles, `grid-template-columns: repeat(auto-fit, minmax(220px, 1fr))`. Each tile is `[huge number] [unit] / [label + provenance badge] / [12px sparkline]`.

| # | Metric | Source class | Notes |
|---|---|---|---|
| 1 | Aggregate output throughput | `derived` | Σ SSE token deltas over a 2 s sliding window. Client-side, always real. |
| 2 | Prefix cache hit rate | `server` | Real counter. **Ships even if it reads 0%.** |
| 3 | KV block utilization | `derived` from `server` | `in_use / capacity`. Unavailable until KV plumbing lands — em-dash, not `0 %`. |
| 4 | Running / waiting | `server` | Two numbers, one tile, `6 / 2` with a visible separator. |

**Hero slot fallback rule.** If the prefix scenario is cut and product decides the hit-rate tile should go with it, slot 2's replacement is **tokens per decode step** — the best single scalar for batching efficiency and unambiguously real. Because the strip is `auto-fit`, removing a tile reflows the remaining three to fill the width with **no hole**. Do not hard-code four columns.

The hero number uses `--og-type-hero` (44px) at `--og-weight-light` with `font-variant-numeric: tabular-nums`. Tabular figures everywhere a number updates — proportional digits cause a visible width-jitter at 4 Hz that reads as instability and is genuinely distracting. This is the cheapest polish on the page.

#### Stage (flexible, ~380px min)

Scenario tabs are a real ARIA tablist (`role="tablist"`, roving tabindex, `1`/`2`/`3`/`4` shortcuts). **Tabs are self-sizing and content-driven — deleting the prefix tab deletes a tab and nothing else.** The stage is a single slot; one panel is mounted at a time; there is no fixed 3-up layout to leave a gap in. This is the structural half of "the prefix scenario can be cut cleanly" (§7.5).

The **narration strip** is a single-line `aria-live="polite"` region, 36px, that advances one short sentence per scenario phase. It is the difference between a dashboard and a demo: charts do not explain themselves, and a sentence pinned to the moment does. It is also, not incidentally, the primary screen-reader channel for what the visualizations show.

#### Panel grid (below fold)

`display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: var(--og-space-4)`. Panels declare a `data-span` of 1 or 2 columns. Auto-fill means **any panel can be removed without leaving a hole** — the same structural property that makes the prefix cut safe. Panel open/closed state persists to `localStorage` and to the deep link.

### 1.3 Above/below the fold — the explicit ruling

**Above the fold** (must be visible at 1280×720 without scrolling): app rail, model card trigger, connection chip, all four hero tiles, scenario tabs, the primary control (`▶ Run`), the narration strip, and at least 240px of the visualization.

**Below the fold:** everything else. A casual visitor never needs to scroll past the activity strip. An engineer evaluating the project can expand every panel and get a Prometheus-grade view. That is progressive disclosure doing its actual job — not hiding complexity, but *ordering* it.

**The page opens alive.** Per the PM's 30-second read, a low-rate idle workload runs on load so the visitor's first impression is motion rather than a wall of zeros. Two design constraints on that: (a) it must be visibly labelled in the narration strip (`idle workload — 3 background requests keeping the batch warm`) so nobody thinks we are faking traffic, and (b) it must be stoppable from the rail. Undisclosed background load would be exactly the kind of small dishonesty this page cannot afford.

### 1.4 The two blocking failure states (AC37)

These are what a first-time visitor is most likely to see. They are the most important screens in the demo and they will be seen more often than any scenario. **Design them first, not last.**

Shared anatomy — a **full-stage takeover** (hero, stage, activity strip, and panel grid are all replaced; the app rail *stays*, degraded, so the visitor keeps their orientation and can see the connection chip agreeing with the message):

```
╔══════════════════════════════════════════════════════════════════════╗
║  [app rail stays — model fields all render —, chip reads ○ offline]  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║                        ╭───────────╮                                 ║
║                        │  ◌ ─ ─ ╳  │      64px line-art mark         ║
║                        ╰───────────╯                                 ║
║                                                                      ║
║              Can't reach the onnx-genai server                       ║
║                                                                      ║
║        Nothing is running at http://127.0.0.1:8080. This page is     ║
║        served by the server itself, so if you're reading this, the   ║
║        server was up a moment ago and has since stopped.             ║
║                                                                      ║
║   ┌────────────────────────────────────────────────────────── ⧉ ┐   ║
║   │ ./target/release/onnx-genai-server \                        │   ║
║   │   --model models/qwen2.5-0.5b-scatter-v2 \                  │   ║
║   │   --enable-debug-endpoints                                  │   ║
║   └─────────────────────────────────────────────────────────────┘   ║
║                                                                      ║
║        ◌ Retrying in 3s…            (live, counts down, 1 Hz)        ║
║        Attempt 4 · last success 00:41 ago                            ║
║                                                                      ║
║        Reconnects automatically. Nothing you were doing is lost.     ║
╚══════════════════════════════════════════════════════════════════════╝
```

Design notes that make these *beautiful* rather than *apologetic*:

- **The mark is line-art, geometric, 64px, 1.5px stroke, in `--og-fg-subtle`** — drawn in the same visual language as the block grid. Not an emoji, not a red triangle, not a sad face. `unreachable` = a dashed circle with a break in it; `no-model` = an empty rounded slot with a dotted interior. They should read as *diagram*, consistent with a page whose whole aesthetic is technical drawing.
- **The stage background keeps the page's 8px grid texture** at `--og-grid-alpha`. An error state on a blank white void looks like a crash. An error state on the page's own substrate looks like a designed state of the product.
- **The copy block is real, copyable, and identical to the README string** (AC38). One source of truth: the command lives in a single JS constant, is rendered here and in the docs check, and cannot drift. Click-to-copy with a 1.2s inline `copied ✓` confirmation — never a toast.
- **The retry countdown is live and visible** (AC19). A spinner that says nothing is anxiety; a countdown that says `3s` is information. On success the state dissolves in 200ms and the page restores prior scenario config — nothing the visitor set up is lost.
- **Tone is calm and second-person, never blaming.** "Nothing is running at…" not "ERROR: connection refused". The second sentence tells them something genuinely useful and slightly clever — *this page is served by the server, so it was up a moment ago* — which reframes the situation from "broken product" to "process exited", which is the truth.

The **no-model** state differs in body copy, mark, and command, and adds a `▸ what the server did report` disclosure listing the actual `/health` payload. An engineer whose server is misconfigured wants the evidence, and one click is the right cost for it.

| | Server unreachable | Reachable, no model |
|---|---|---|
| Detect | `/health` fetch rejects / network error | 2xx from `/health` or `/v1/models` with no model reported |
| Mark | broken dashed ring | empty slot outline |
| Headline | Can't reach the onnx-genai server | The server is running, but no model is loaded |
| Body | names the origin, explains self-serving irony | explains `--model` / `--models-dir` are an `ArgGroup`, exactly one required |
| Command | full launch line | same line, `--model` emphasised |
| Live element | retry countdown, attempt count | polls `/v1/models` at 1 Hz, `watching for a model…` |
| Disclosure | — | raw `/health` response |

**Never collapse these into one generic error.** They are different problems with different fixes, and the PM is right that the same underlying blankness produces opposite impressions depending on whether the visitor is told what to do.

**A third, non-blocking state:** debug endpoints disabled → per-panel capability notices (§4.4), the rest of the page works (AC20). And `file://` (AC5) → a fourth full-stage state, detected via `location.protocol`, saying plainly that this page must be served by the server and giving the same command.

---

## 2. DESIGN TOKENS — copy verbatim

Namespaced `--og-`. This is the complete contract; **no panel may invent a colour, a spacing value, or a font size.** If a panel needs a value that isn't here, that is a request to me, not a local decision — the fastest way to a patchwork UI is fifteen slightly-different greys.

```css
/* examples/serving-dashboard/styles/tokens.css — DESIGN-OWNED */
:root {
  /* ── SURFACE ─────────────────────────────────────────────────────────
     Dark-first. A telemetry page is looked at for a long time, often on a
     projector; dark reduces the emitted-light area and lets the Okabe-Ito
     hues sit at their intended saturation instead of fighting white.     */
  --og-bg:            #0d1117;   /* page                                   */
  --og-bg-raised:     #151b23;   /* panels, cards                          */
  --og-bg-sunken:     #090c10;   /* plot areas, grid wells, code blocks    */
  --og-bg-hover:      #1c232b;
  --og-bg-active:     #232b35;

  --og-border:        #2a323d;   /* default 1px hairline                   */
  --og-border-strong: #3d4855;   /* emphasis, focused panels               */
  --og-border-subtle: #1e242c;   /* internal dividers                      */

  /* ── FOREGROUND ─────────────────────────────────────────────────────
     Contrast against --og-bg-raised (#151b23):
       fg          17.4:1   fg-muted 8.1:1   fg-subtle 4.6:1
     All clear 4.5:1 (AC/WCAG AA). fg-faint is DECORATIVE ONLY —
     never put text a visitor must read in fg-faint.                      */
  --og-fg:            #e6edf3;
  --og-fg-muted:      #9aa7b4;
  --og-fg-subtle:     #6e7d8c;
  --og-fg-faint:      #3d4855;   /* rules, grid texture, hatch — not text  */

  /* ── SEMANTIC ────────────────────────────────────────────────────────
     Derived from Okabe-Ito, never from the default web red/green, because
     red/green is precisely the pair 8% of men cannot separate.           */
  --og-ok:            #009e73;   /* Okabe-Ito bluish green                 */
  --og-warn:          #e69f00;   /* Okabe-Ito orange                       */
  --og-bad:           #d55e00;   /* Okabe-Ito vermillion                   */
  --og-info:          #56b4e9;   /* Okabe-Ito sky blue                     */
  --og-accent:        #0072b2;   /* Okabe-Ito blue — primary action        */
  --og-accent-fg:     #ffffff;

  /* ── THE UNAVAILABLE-STATE TOKENS ────────────────────────────────────
     §4. These are load-bearing. Do not restyle them per panel — their
     entire value is that they look identical everywhere on the page.     */
  --og-unavail-fg:      #5a6673;  /* the em-dash itself                    */
  --og-unavail-rule:    #3d4855;  /* its dashed underline                  */
  --og-unavail-hatch:   #212932;  /* diagonal hatch ink in chart wells     */
  --og-unavail-bg:      #131920;  /* hatch backdrop                        */
  --og-unavail-label:   #6e7d8c;  /* "not measurable yet" caption          */
  --og-pending-fg:      #4a5560;  /* awaiting-first-sample dots            */
  --og-stale-fg:        #7a8794;  /* a real but no-longer-fresh value       */
  --og-stale-rule:      #4a5560;  /* its dotted underline + age suffix      */
  --og-simulated-fg:    #e69f00;  /* the `simulated` badge (AC8)           */
  --og-simulated-rule:  #e69f00;  /* dashed outline on simulated marks     */
  --og-estimated-fg:    #9aa7b4;  /* the `est.` qualifier                  */

  /* ── SEQUENCE PALETTE — Okabe-Ito, colourblind-safe, 8 stable slots ──
     Assigned round-robin by arrival index and STABLE for the life of a
     scenario. The same sequence is the same colour AND the same pattern
     in the swimlanes, the block grid, and the request table — that
     cross-surface identity is what makes "sequence" and "memory" feel
     like one concept instead of two panels.                              */
  --og-seq-0: #0072b2;  /* blue           */
  --og-seq-1: #e69f00;  /* orange         */
  --og-seq-2: #009e73;  /* bluish green   */
  --og-seq-3: #cc79a7;  /* reddish purple */
  --og-seq-4: #56b4e9;  /* sky blue       */
  --og-seq-5: #d55e00;  /* vermillion     */
  --og-seq-6: #f0e442;  /* yellow         */
  --og-seq-7: #b0b8c1;  /* neutral        */

  /* ── SPACING — 4px base, geometric-ish. Nine values, no exceptions. ── */
  --og-space-0: 0;
  --og-space-1: 2px;
  --og-space-2: 4px;
  --og-space-3: 8px;
  --og-space-4: 12px;
  --og-space-5: 16px;
  --og-space-6: 24px;
  --og-space-7: 32px;
  --og-space-8: 48px;
  --og-space-9: 64px;

  /* ── TYPE ────────────────────────────────────────────────────────────
     Two families. UI sans for prose and labels; mono for EVERY number,
     identifier, path, and command — because a number that shifts width
     as it updates reads as instability, and at 4 Hz that is constant.    */
  --og-font-ui:   ui-sans-serif, -apple-system, "Segoe UI", Inter, system-ui, sans-serif;
  --og-font-mono: ui-monospace, "SF Mono", "JetBrains Mono", "Cascadia Mono", Menlo, monospace;

  --og-type-hero:    44px;  /* hero tile figure                           */
  --og-type-xl:      28px;  /* panel primary figure                       */
  --og-type-lg:      20px;  /* secondary figure                           */
  --og-type-md:      15px;  /* body prose                                 */
  --og-type-sm:      13px;  /* panel body, table cells                    */
  --og-type-xs:      11px;  /* labels, axis ticks                         */
  --og-type-2xs:      9px;  /* provenance badges, refcount numerals       */

  --og-leading-tight: 1.15;
  --og-leading-body:  1.55;

  --og-weight-light:   300;  /* hero figures only                         */
  --og-weight-normal:  400;
  --og-weight-medium:  500;
  --og-weight-bold:    650;

  --og-tracking-caps: 0.08em; /* small-caps section labels                */

  /* ── SHAPE / DEPTH ───────────────────────────────────────────────────
     Shadows are nearly absent by design: this page should read as a
     technical drawing, not a stack of cards. Depth comes from surface
     value and hairlines, which survive a projector far better.           */
  --og-radius-sm: 3px;
  --og-radius-md: 6px;
  --og-radius-lg: 10px;
  --og-radius-pill: 999px;
  --og-shadow-popover: 0 8px 24px rgb(0 0 0 / 0.45);
  --og-hairline: 1px solid var(--og-border);

  /* ── MOTION ──────────────────────────────────────────────────────────
     All durations flow through these two tokens so that
     prefers-reduced-motion can zero them in ONE place (see below).       */
  --og-dur-fast:  90ms;
  --og-dur-base: 160ms;
  --og-dur-slow: 260ms;
  --og-ease: cubic-bezier(0.2, 0, 0.2, 1);

  /* ── FOCUS — one ring, everywhere, never removed ─────────────────── */
  --og-focus-ring: 2px solid #56b4e9;
  --og-focus-offset: 2px;

  /* ── GRID TEXTURE — the page substrate, also used in failure states ── */
  --og-grid-alpha: 0.35;
  --og-grid-size: 8px;

  /* ── LAYOUT ──────────────────────────────────────────────────────── */
  --og-rail-h: 56px;
  --og-hero-h: 92px;
  --og-activity-h: 72px;
  --og-content-max: 1680px;
  --og-panel-min: 340px;
}

@media (prefers-reduced-motion: reduce) {
  :root {
    --og-dur-fast: 0ms;
    --og-dur-base: 0ms;
    --og-dur-slow: 0ms;
  }
}
```

### 2.1 Sequence identity: colour **plus** pattern **plus** label

Colour alone is forbidden (AC25). Every sequence gets a triple, and **all three are always present** in the block grid and swimlanes:

| Slot | Colour | Pattern | Shape marker |
|---|---|---|---|
| 0 | `--og-seq-0` | solid | ● circle |
| 1 | `--og-seq-1` | diagonal 45° | ▲ triangle |
| 2 | `--og-seq-2` | diagonal 135° | ■ square |
| 3 | `--og-seq-3` | dots | ◆ diamond |
| 4 | `--og-seq-4` | horizontal rules | ▬ bar |
| 5 | `--og-seq-5` | vertical rules | ⬟ pentagon |
| 6 | `--og-seq-6` | cross-hatch | ✚ cross |
| 7 | `--og-seq-7` | checker | ★ star |

Beyond 8 sequences the palette wraps but the **pattern advances independently** (mod 8 colour × mod 7 pattern → 56 distinguishable identities before a true collision), and every mark carries its numeric sequence id on hover regardless. The shape marker appears in lane labels, table rows, and legends — so a grayscale screenshot in a blog post still parses. That last property is not a courtesy to screenshots; it is the same property that makes the page work for a colourblind reader, earned once and spent twice.

Patterns are defined once as SVG `<pattern>` elements in a hidden `<defs>` block in `index.html` (ids `og-pat-0`…`og-pat-7`) and as matching `CanvasPattern` objects built from a shared offscreen canvas, so SVG and canvas surfaces agree exactly. **Demo dev owns that defs block; dashboard dev consumes it.**

---

## 3. THE COMPONENT + CSS CONTRACT

### 3.1 Ownership seam

| Owner | Owns | May not touch |
|---|---|---|
| **demo dev** | `index.html`, `app.js` (shell/router), `telemetry-store.js` / `telemetry-field.js` / `telemetry-provenance.js`, `CONTRACT.md`, `scenarios/*.js`, `styles/base.css`, `styles/layout.css`, the SVG `<defs>` patterns | `dashboard/*` |
| **dashboard dev** | `dashboard/*.js` (one file per panel), `styles/panels.css` | `index.html`, `telemetry.js`, `scenarios/*` |
| **designer (me)** | `styles/tokens.css` | — |

```
examples/serving-dashboard/
├── index.html            demo dev
├── app.js                demo dev   — shell, routing, deep links, keyboard
├── CONTRACT.md           demo dev   — NORMATIVE store + panel contract
├── telemetry-store.js    demo dev   — polling, single-flight, history
├── telemetry-field.js    demo dev   — the envelope + formatFieldText/describeField
├── telemetry-provenance.js demo dev — the stub classification table
├── format.js             demo dev   — shared render helpers (§3.4). Both import.
├── scenarios/
│   ├── batching.js       demo dev
│   ├── paged-kv.js       demo dev
│   └── prefix.js         demo dev   (cuttable — see §7.5)
├── dashboard/
│   ├── throughput.js     dashboard dev
│   ├── latency.js        dashboard dev
│   ├── scheduling.js     dashboard dev
│   ├── kv-memory.js      dashboard dev
│   ├── prefix-cache.js   dashboard dev
│   ├── requests.js       dashboard dev
│   └── system.js         dashboard dev
└── styles/
    ├── tokens.css        DESIGNER — nobody else edits
    ├── base.css          demo dev  — reset, type, focus, a11y utilities
    ├── layout.css        demo dev  — rail, hero, stage, grid, failure states
    └── panels.css        dashboard dev — panel internals only
```

### 3.2 The telemetry store — the honesty firewall

**RECONCILED.** The demo developer (@bb2ee824) landed `CONTRACT.md`,
`telemetry-field.js`, `telemetry-provenance.js`, and `telemetry-store.js`
independently and in parallel with this document, and converged on the same core
idea from the other direction: **a panel never receives a number, it receives a
provenance envelope and must branch on its state.** Two people arriving at the
same seam independently is the strongest signal available that the seam is in the
right place.

**`examples/serving-dashboard/CONTRACT.md` is now the normative source for the
store API and the panel lifecycle.** Where this document and that one differ,
CONTRACT.md wins and this section records the deltas. I have adopted their
vocabulary wholesale, including two things their version does better than my
original draft:

1. **`measured` rather than `ok`.** `ok` names approval; `measured` names
   provenance, which is the property that actually matters. Theirs is the better
   word and I have taken it.
2. **A fourth state, `stale`.** My draft had three. `stale` — *this was really
   measured, but the latest poll did not refresh it* — is a real state I missed,
   and it is exactly the state a dashboard is in during the seconds before it
   notices the server died. Rendering a 12-second-old number as current is the
   same class of error as rendering a fabricated one, and `stale` is what
   prevents it. Visual treatment is specified in §4.1 and it carries a **visible
   age suffix**, not just a colour change, because "this number is old" must
   survive grayscale like everything else.

```js
{
  value:        3,              // null unless state is 'measured' or 'stale'
  state:        'measured',     // 'measured' | 'pending' | 'stale' | 'unavailable'
  source:       '/v1/status',   // endpoint path, or 'client' | 'derived'
  reason:       null,           // required sentence whenever state !== 'measured'
  unit:         'requests',
  observedAtMs: 1785390093123,  // for 'stale', the ORIGINAL observation time
  derivedFrom:  null
}
```

**Two requests back to the store owner**, both small, both needed to close ACs I
own — raised with them directly rather than asserted here:

- **`'estimated'` as a distinct `source` value.** `derived` (arithmetic on real
  inputs) and `estimated` (a model standing in for a measurement) are different
  kinds of claim and must look different: estimated values render with a `~`
  prefix and an `ᴱ` badge so the difference is visible *without* hovering. Today
  only `prefix.time_saved` needs it, and one honest estimate deserves one honest
  label.
- **A `sourceClass` accessor.** Their `source` is an endpoint path, which is
  better than my enum for auditing — it is precise enough to curl. But AC7 wants
  a *class* on the badge, so `format.js` needs a documented path → class mapping
  rather than each panel inventing one.

**The one design risk I want on the record** is `telemetry-provenance.js`'s
static `DOCUMENTED_ZERO` / `NOT_PLUMBED` table. Centralising the stub list in one
file is unambiguously right, and far better than the alternative of scattering it
through panels. But it is still a **hardcoded mirror of a server state that is
actively changing underneath it** — the moment @e00032a4 wires `Engine::page_usage()`
through, the table starts lying in the *pessimistic* direction, em-dashing a real
measurement. That is the safer direction to be wrong in, and I would ship it. The
durable fix is for the server to signal availability itself (a `null`, or a
top-level `unavailable: [paths]` array), which I have already requested from the
Architect; the table then becomes a fallback rather than the source of truth.
Until then, the table needs a test that fails loudly when a field classified
`NOT_PLUMBED` arrives carrying a real value — so it rots noisily instead of
silently.

Store behaviours that are acceptance-relevant and unchanged from my draft:

1. **Polling is single-flight** (AC24): one request in flight; a slow response
   delays the next tick rather than queueing. Backoff 250 ms → 4 s, reset on
   success, `AbortController` on unmount.
2. **The store never invents.** No interpolation across gaps, no carry-forward of
   a stale value into a *new* timestamp (that is what the `stale` state is for —
   it keeps the old timestamp), no zero-filling. A missing sample is a `gap`, and
   §4.3 says exactly what a gap looks like.
3. **Series need `gaps: [[startMs, endMs]]`** so `renderSeries()` can honour §4.3
   and never bridge a gap with a line segment.

### 3.3 Panel module contract

Every file in `dashboard/` is an ES module with exactly this shape:

```js
// dashboard/kv-memory.js
export const meta = {
  id: 'kv-memory',                 // kebab-case, unique, used as class prefix
  title: 'KV memory',
  group: 'memory',                 // throughput | scheduling | memory | cache | system
  span: 2,                         // grid columns, 1 or 2
  cadence: 250,                    // ms — advisory; the store drives the ticks
  defaultOpen: true,
  acronyms: {                      // AC30 — rendered as dotted-underline <abbr>
    KV: 'Key/Value attention cache',
    COW: 'Copy-on-write — sharing a block until one owner needs to diverge',
  },
};

/**
 * @param {HTMLElement} rootElement  an EMPTY <section> the shell created.
 * @param {TelemetryStore} telemetryStore
 * @returns {{ unmount(): void, describe(): string }}
 */
export function mount(rootElement, telemetryStore) { /* … */ }
```

**Hard rules for panels.** These are review-enforceable:

1. **Render only inside `rootElement`.** No `document.querySelector` outside it, no `document.body` appends, no global listeners without removing them in `unmount()`.
2. **Never `fetch`.** All data comes from the store. A panel that opens its own connection breaks single-flight (AC24) and the honesty firewall in one move.
3. **Never set an interval.** Subscribe to the store; it owns cadence. Canvas repaints coalesce through **one** `requestAnimationFrame`, and the panel must skip painting when `rootElement.hidden` or off-screen (`IntersectionObserver`) — this is most of AC23.
4. **`unmount()` must be total**: unsubscribe, cancel rAF, disconnect observers, null canvas contexts. AC22's no-memory-growth is won or lost here.
5. **Never render a raw number.** All values go through `renderField()` / `renderSeries()` from `format.js`. This is the mechanical enforcement of §4 — it is not stylistic advice.
6. **`describe()` returns a plain-English sentence** of the panel's current state, used for the chart `aria-label` (AC28) and by the narration system. Example: `"KV memory: 318 of 512 blocks in use, 96 shared. Slot fill 90.4%. Engine batch size not measurable."` Writing this well is the difference between an accessible page and a compliant one.
7. **Class names are `panel-<id>__element--modifier`.** `panel-kv-memory__grid`, `panel-kv-memory__cell--shared`. No panel may style another panel, and no panel may define a token.

### 3.4 The DOM the shell guarantees

The shell creates and hands over exactly this, and this is what a panel may rely on:

```html
<section class="panel"
         data-panel-id="kv-memory"
         data-group="memory"
         data-span="2"
         aria-labelledby="panel-kv-memory-title">
  <header class="panel__head">
    <h2 class="panel__title" id="panel-kv-memory-title">KV memory</h2>
    <div class="panel__tools"><!-- shell owns: table toggle, collapse, info --></div>
  </header>
  <div class="panel__body"><!-- ★ THE PANEL'S rootElement — empty on mount --></div>
</section>
```

- `rootElement` **is `.panel__body`** — not the `<section>`. The panel does not own its header, collapse control, or "view as table" toggle; the shell provides all three uniformly so they cannot drift between panels.
- The shell calls `describe()` for the header's info affordance and the table view, so **`describe()` is not optional.**
- `.panel__body` is `container-type: inline-size`. Panels size themselves with `@container`, **never** `@media` — a panel is 340px in a 1-col layout and 700px in a 2-col one, and media queries cannot see that distinction. This is the single most common way a panel grid becomes a patchwork.
- The shell sets `hidden` on collapsed bodies; panels must handle mount-while-hidden (defer first paint to the first `IntersectionObserver` hit).

`format.js`, owned by demo dev, imported by both:

```js
renderField(field, opts?)        // -> HTMLElement. Value + unit + provenance
                                 //    badge + the §4 treatment for non-ok.
renderSeries(canvas, series, o?) // sparkline incl. hatched unavailable wells
                                 //    and gap rendering (§4.3).
seqStyle(seqId)                  // -> {color, patternId, canvasPattern, marker, label}
fmtNumber(n, unit)               // tabular, unit-aware, sensible precision
fmtDuration(ms)                  // 38 ms / 1.84 s / 2 m 04 s
sourceBadge(sourceClass)         // -> the ˢ ᶜ ᴰ ᴱ chip
capabilityNotice(capability)     // -> the §4.4 card
```

---

## 4. THE UNAVAILABLE-DATA DESIGN LANGUAGE

**Non-negotiable. This section protects the demo's credibility more than everything else in this document combined.**

The server returns documented zeros for things it cannot yet measure (`recon-map.md` §7.3). A panel bound naively to `/v1/status` will render `0.0 tok/s` as a measurement, and a visitor will read it as "this runtime produces zero tokens per second" or, worse, believe it and quote it. Neither the developer nor the visitor will notice the lie. That is what makes it dangerous: it is a *silent, self-inflicted, plausible* falsehood.

### 4.1 The four states of a number

Every value on this page is in exactly one of four states. They have four distinct visual treatments and they must never be confused.

| State | Meaning | Rendering | Hover copy |
|---|---|---|---|
| **measured** | The server computed this, just now. Includes a genuine zero. | The number, `--og-fg`, tabular, + provenance badge | value, unit, source class, cadence |
| **measured (real zero)** | Measured, and the measurement is zero | **`0` rendered exactly like any other number**, full contrast, no apology | "Measured. The value really is zero." |
| **pending** | Measurable, but no sample yet. **Resolves on its own.** | `···` three 2px dots, `--og-pending-fg`, at the number's baseline | "No samples yet. Run a scenario." |
| **stale** | Was measured; the latest poll did not refresh it | The last good number in `--og-stale-fg`, **plus a visible age suffix** `12s old`, and a dotted underline | "Last read 12 s ago. The server has not responded since." |
| **unavailable** | No value exists and **none is coming** without a server or config change | **`—`** em-dash, `--og-unavail-fg`, with a 1px dashed underline in `--og-unavail-rule`, `cursor: help` | why, plus the fix if one exists |

**The em-dash is the whole system.** It is a single glyph, it is instantly distinguishable from `0`, it occupies the number's slot so nothing reflows when the value arrives, it has been the typographic convention for "no data" in printed tables for two centuries, and it carries no emotional charge. It is not an error. It is not a warning. It is an honest absence, rendered with dignity.

Three rules about the treatment:

- **The dashed underline is what makes it discoverable.** A bare `—` looks like a rendering bug. A `—` with a dashed underline and a help cursor is visibly an affordance, and a visitor learns the convention once, from any one of them, and then reads the whole page correctly. Consistency is the mechanism — this is why nobody may restyle it per panel.
- **The slot never collapses.** `min-width` is reserved to the width the real value would occupy. When KV plumbing lands, panels must not jump. Layout shift on data arrival makes the page feel unreliable at exactly the moment it becomes more reliable.
- **The unit and label stay.** `— %` and `— ms` with the full label, because *which* thing is unavailable is information. Hiding the row destroys the visitor's ability to know what they weren't told, and "what you weren't told" is precisely what an evaluating engineer wants to know.

```html
<!-- ok -->
<span class="value" data-state="ok" data-source="server">
  <span class="value__num">318</span><span class="value__unit">blocks</span>
  <abbr class="value__src" title="Server counter · PageUsage.in_use · 250 ms">ˢ</abbr>
</span>

<!-- unavailable -->
<span class="value" data-state="unavailable" data-source="server"
      tabindex="0" role="button" aria-describedby="tip-batchsize"
      aria-label="Engine batch size: not measurable. No endpoint exposes the engine's actual batch size; what you can see is requests in flight.">
  <span class="value__num value__num--unavailable" aria-hidden="true">—</span>
  <span class="value__unit">count</span>
</span>
```

```css
.value__num--unavailable {
  color: var(--og-unavail-fg);
  border-bottom: 1px dashed var(--og-unavail-rule);
  cursor: help;
  font-variant-numeric: tabular-nums;
}
.value[data-state="unavailable"] { min-width: var(--og-value-slot, 5ch); }
.value[data-state="pending"] .value__num { color: var(--og-pending-fg); letter-spacing: .12em; }
.value[data-state="stale"]   .value__num {
  color: var(--og-stale-fg);
  border-bottom: 1px dotted var(--og-stale-rule);
}
/* The age suffix is REQUIRED on a stale value. A colour shift alone is not a
   sufficient signal that a number is old — it fails grayscale, it fails
   colourblindness, and "this reading is 12 seconds old" is information the
   visitor needs in words. Rendering a stale number as current is the same
   class of error as rendering a fabricated one. */
.value[data-state="stale"]::after {
  content: " " attr(data-age);
  font-size: var(--og-type-2xs);
  color: var(--og-stale-fg);
}
```

**`tabindex="0"` on unavailable values is deliberate and I want it defended.** These are the values whose explanation matters most, and a keyboard or screen-reader user must be able to reach that explanation. An em-dash with no reachable reason is *less* honest than a zero, because it withholds without offering recourse. The `aria-label` carries the full sentence so the reason is available without a hover event ever firing.

### 4.2 Writing the `reason` — this is design work, not filler

The hover copy is the entire payload of the em-dash. A vague reason wastes the affordance. Rules:

1. **Say what is missing, not that something is missing.** ✗ "Data unavailable." ✓ "The engine computes KV page statistics but the server does not yet request them."
2. **Name the fix when there is one.** ✓ "Start the server with `--enable-debug-endpoints` to enable this panel." Include the literal flag, copyable.
3. **Distinguish 'not built' from 'not enabled' from 'not applicable'.** Three genuinely different situations, and an engineer's next action differs in each.
4. **Never apologise and never promise.** No "coming soon", no "we're working on it". A roadmap claim in a tooltip is an unverifiable promise, which is the same species of thing as an unverifiable number.

Reference copy, to be used verbatim:

| Field | `reason` |
|---|---|
| `kv.*` (introspection off) | "KV page statistics are computed by the engine but not yet exposed over HTTP. `Engine::page_usage()` exists; the server does not call it." |
| `kv.*` (debug gated) | "Requires debug endpoints. Restart the server with `--enable-debug-endpoints`." |
| ~~`scheduler.preemptions_total`~~ | 🔴 **STRUCK — FIELD DROPPED, AND THIS COPY WAS FALSE.** It claimed *"the scheduler performs preemption but keeps no counter for it."* **`ContinuousBatchManager` has no `Scheduler` field at all** — preemption is not uncounted, the component is **absent**. Bind nothing. See D148. |
| ~~`scheduler.batch_occupancy`~~ | 🔴 **STRUCK — CONTAINED THE BANNED CLAIM (AC59).** It said *"the current batch size is real; the denominator isn't."* **The numerator is not a batch size either**: `onnx_genai_batch_size_current` is `fetch_add(1)` at the HTTP layer (`metrics.rs:111`/`:145`), counting requests in flight. **Both halves were wrong; only the denominator was ever suspected.** Replacement copy below. |
| `batch.in_flight` | "Requests in flight at the HTTP layer. This is **not** the engine's batch size — eight simultaneous requests are not eight simultaneous decodes, which is the whole point of this scenario." |
| `batch.true_size` | "The engine's actual batch size is not exposed by any endpoint. What you can see is how many requests are in flight and how many are waiting." |
| `throughput.*` from `/v1/status` | "`/v1/status` reports this as a documented zero — the server records cumulative token totals only. This page derives throughput client-side instead." |
| `server.quantization` | "Quantization is recorded in the model's inference metadata but not exposed by the server." |
| `request.queue_wait_ms` | "The browser can only observe *sent → first token*. Queue wait and prefill are not separable from the client." |
| continuous batch off | "This model doesn't use a static KV cache, so the continuous batch driver is disabled and requests run one at a time. Use a static-cache/scatter model to see batching." |

### 4.3 Unavailable in charts — the hard case

A missing *number* is easy. A missing *series* is where a well-meaning dev draws a flat line at zero and invents a measurement out of nothing. **A flat line at zero is the most dangerous single mark this page could render**, because unlike a `0` in a table it also implies *duration* — it says "we watched this for sixty seconds and it was zero the whole time."

**Fully unavailable series.** Do not draw a line. Do not draw an axis with numbers. Draw:

- The plot well filled `--og-unavail-bg`, overlaid with **45° diagonal hatch**, 1px lines in `--og-unavail-hatch` at 6px pitch. Deliberately low contrast — present, obviously non-data, never mistaken for a chart.
- A centred caption in `--og-unavail-label`, `--og-type-xs`, small-caps, letter-spaced: **`NOT MEASURABLE YET`**.
- The Y axis renders **no tick labels** — an axis with numbers implies a measured range. Keep the axis *line* so the panel's rhythm holds; drop the numbers.
- The whole well is `tabindex="0"` with the reason as `aria-label`, and carries the same dashed-underline convention on its caption so it reads as the same language as the em-dash.

```
  ╭──────────────────────────────────────────╮
  │▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚│
  │▚▚▚▚▚▚▚▚▚▚▚ NOT MEASURABLE YET ▚▚▚▚▚▚▚▚▚▚▚│
  │▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚│
  ╰──────────────────────────────────────────╯
     engine batch size / step               ⓘ
```

**Partially unavailable series** (server gained the capability mid-window, or the connection dropped): hatch only the affected time range, real line elsewhere, separated by a **vertical dashed rule** in `--og-unavail-rule`. **Never bridge a gap with a line segment.** Interpolating across a gap fabricates the most convincing kind of false data — data that looks continuous. This is what `Series.gaps: [[startMs, endMs]]` exists for; `renderSeries()` honours it and panels get it for free.

**Pending series** (measurable, no samples yet): empty well, faint baseline at y=0, centred `AWAITING DATA` in `--og-pending-fg`. Distinct from hatch: pending will fill in on its own; unavailable will not.

**Sparkline in a hero tile, unavailable:** collapse to a 12px hatched bar the width of the tile. It occupies the sparkline's slot so the tile keeps its height, and it visually rhymes with the full-size treatment.

**Never** render an unavailable series as: a flat zero line · an empty white box with axes · a hidden element · `NaN` · a dotted line at an arbitrary height.

### 4.4 Capability notices — when a whole panel has no data

For an entire feature that isn't available (KV introspection off, debug endpoints disabled), a grid of hatch marks is noise. Replace the panel body with a **capability notice** — same line-art language as the failure states (§1.4), scaled down:

```
┌─ KV memory ─────────────────────────────────── ⓘ ▾ ┐
│                                                     │
│        ⬚  KV block introspection is off             │
│                                                     │
│        The engine computes page statistics;         │
│        this server build doesn't expose them.       │
│                                                     │
│        ┌──────────────────────────────────── ⧉ ┐   │
│        │ --enable-debug-endpoints              │   │
│        └───────────────────────────────────────┘   │
│                                                     │
│        Everything else on this page still works.    │
└─────────────────────────────────────────────────────┘
```

That last line matters. It converts "this dashboard is broken" into "this panel needs a flag", which is the true state of the world (AC20).

### 4.5 Provenance badges (AC7) — the audit trail

Every value carries a superscript source chip, `--og-type-2xs`, `--og-fg-subtle`, with an `<abbr>` title:

| Badge | Class | Meaning | Hover |
|---|---|---|---|
| `ˢ` | `server` | a real server counter | "Server counter · `PageUsage.in_use` · updated every 250 ms" |
| `ᶜ` | `client` | measured in this browser | "Measured in your browser from the SSE stream" |
| `ᴰ` | `derived` | arithmetic on real inputs | "Derived: Δ completion_tokens ÷ Δt over a 2 s window" |
| `ᴱ` | `estimated` | a model, not a measurement | "**Estimated**, not measured: skipped tokens × observed per-token prefill cost" |
| `SIM` | `simulated` | not measured at all (AC8) | full-word pill in `--og-simulated-fg`, always dashed-outlined |

`ᴱ` and `SIM` escalate deliberately. `ᴱ` renders in `--og-estimated-fg` and its *value* is prefixed `~`, so `~340 ms` is visibly a different kind of claim than `340 ms` even without hovering. `SIM` is never a superscript — it is a full-word pill, because a superscript is easy to miss and AC8 requires it to be unmissable. **Per the ratified no-simulated-baseline ruling, nothing on this page should ever need `SIM`.** It exists so that if something ever does, the language already has a loud, ugly, honest word for it. The ugliness is the point.

### 4.6 The honesty footer (AC10)

Not a legal disclaimer. A three-column table, generated **from the same field registry the panels render from**, so it cannot drift:

> 🔴 **THE BLOCK BELOW IS AN ILLUSTRATION OF SHAPE, NOT A LIST OF FIELDS. DO NOT TRANSCRIBE IT.** Under AC54 the only legal source of this table is the profile-keyed registry. An earlier revision of this very block listed **`prefix hits/looks` under WHAT'S REAL** — true on one server, false on the other, and false everywhere once the field was deleted — and listed **`batch size`**, which names a counter that counts something else (AC59). **A hand-written example of a generated table is the most likely thing on this page to be copied by hand**, which is precisely the failure AC54 exists to prevent. Field names below are placeholders.

```
WHAT'S REAL                     WHAT'S DERIVED               WHAT ISN'T BUILT YET
TTFT               server ˢ     aggregate tok/s      ᴰ      paged-attention kernels
e2e latency        server ˢ     KV utilization %     ᴰ      request preemption
queue depth        server ˢ     requests waiting     ᴰ      per-request server state
requests in flight server ˢ                                 engine's true batch size
KV page usage      server ˢ                                 prefix reuse
first-token time   client ᶜ
inter-token gap    client ᶜ
```

Generating it from the registry is the design decision that matters — a hand-maintained honesty footer is a lie waiting for its second sprint. Above the table, one paragraph in plain language, and the paged-attention statement **verbatim** (AC9), which also appears next to the block table title:

> This page visualizes the paged KV **allocator** — real block allocation, copy-on-write sharing, tiering, and eviction. Paged-**attention kernels** are not yet implemented in this runtime; attention still runs over materialized KV. We show what exists.

Saying this out loud costs nothing with a casual visitor and buys a great deal with the expert audience this project needs.

**Note the footer above is now profile-dependent, and that is the point.** It lists `prefix hits/looks` under WHAT'S REAL — true on profile D, false on profile S (§13). A hand-maintained footer would now be lying. Because it is generated from the field registry, it self-corrects the moment the registry is keyed by `(field, profile)` — see §4.7. This is the second time registry-generation has caught a drift I introduced myself; it is earning its keep.

---

### 4.7 The three kinds of zero — and why this is **not** a fifth state

@376a0297 is right that there are three distinct zeros, and the third one is new. I had two. The third inverts the first:

| Kind | What happened | Render | Example |
|---|---|---|---|
| **Measured zero** | Subsystem ran, answer was genuinely 0 | **stark `0`, full contrast** | cache consulted, nothing hit |
| **Placeholder** | Server writes a constant; nothing ran | em-dash + hover naming the stub | `/v1/status.tokens_per_second = 0.0` |
| **Bypassed** | Subsystem is not in this configuration's path at all | em-dash + hover explaining the architecture | prefix cache on a static-cache server |

The asymmetry the PM identified is the sharp part: em-dashing a measured zero lies *flatteringly*, but rendering a stark `0%` for a **bypassed** subsystem lies in the *opposite* direction — it claims we tried and failed when we never tried. Both are dishonest; they fail in mirror directions. The rule that covers all three is one sentence:

> **Render a number only when something computed it. The glyph reflects whether a measurement happened — never whether the answer was interesting.**

**Ruling: `bypassed` does NOT become a fifth `state`.** The enum stays at four. Here is why, and it is a load-bearing distinction rather than a preference:

- **`state` answers "how do I render this?"** — it selects the render primitive. Four primitives exist: a value, `···`, a value-plus-age, an em-dash.
- **`classification` answers "why is it like that?"** — it selects the *wording and icon*, not the primitive.

`bypassed` and `placeholder` produce the **identical** render primitive (em-dash + hover + dashed underline). They differ only in what the hover says. Folding that into `state` would inflate the enum every consumer must exhaustively branch on, to express a difference no consumer branches on — and every added state is a new chance for someone's `switch` to fall through, which is precisely the defect found in `formatFieldText`. **Hick's law applies to developers reading a union type.**

**And the second axis already exists.** `telemetry-provenance.js:27` already defines, independently of `state`:

```js
Classification = 'MEASURED' | 'DOCUMENTED_ZERO' | 'NOT_PLUMBED'
```

So the change is **purely additive** — one new classification value, no consumer breaks:

```js
Classification = 'MEASURED' | 'DOCUMENTED_ZERO' | 'NOT_PLUMBED' | 'BYPASSED'
NEVER_MEASURED_CLASSIFICATIONS = ['DOCUMENTED_ZERO', 'NOT_PLUMBED', 'BYPASSED']
```

Adding a state is a breaking change to every branch; adding a classification is a new row in a table. Same expressive power, none of the blast radius. **This is what the PM's own escape hatch — "or carry a `reason` rich enough to render the difference" — looks like when done with a machine-readable discriminator instead of prose.** Panels must never branch on substrings of `reason`.

**`BYPASSED` is inherently profile-dependent, which is the proof of the §13.1 request.** `prefix_cache_hits` is `MEASURED` on profile D and `BYPASSED` on profile S — the *same field*, different provenance, decided by which model loaded. A table keyed by field name alone cannot express that, and defaults to the wrong answer on the server we demo. `BYPASSED` cannot be implemented at all without `(field, profile)` keying; the two changes are one change.

#### 4.7.1 The two hover treatments

Same glyph, different voice. The voice matters because the two facts land differently on a visitor:

```
PLACEHOLDER — a confession. Our omission, our to-do.
  "Not measured yet — /v1/status returns a hardcoded 0.0 for this
   (routes/admin.rs:57). Nothing samples it."
  icon: ⊘   tone: candid, names the file

BYPASSED — an explanation. Correct behaviour, not a gap.
  "Not applicable on this server — static-cache decoding doesn't
   consult the prefix cache, so there is nothing to hit or miss.
   Load a dynamic-cache model to see this."
  icon: ⊗   tone: architectural, names the cause and the remedy
```

`BYPASSED` is the only one of the two that gets to end with **a way to see the real number**, because it is the only one where a real number exists somewhere. That sentence turns a hole into a signpost.

#### 4.7.2 Deliberately **not** distinguishing them at a glance

Both render `—`. The distinction lives entirely in the hover, and that is a decision, not laziness.

A visitor scanning the page needs exactly one fact at glance-level: **"there is no number here, and it is not being hidden from me."** That is one fact, so it gets one glyph. *Why* there is no number is second-level detail, and second-level detail belongs in the second level.

Giving them distinct glyphs would make the dashboard appear to have two different kinds of hole, which reads as **inconsistency, not rigour** — the visitor's first thought is "is one of these a bug?" Honesty that looks like malfunction fails at its only job. No new tokens: `BYPASSED` reuses `--og-unavail-*` exactly.

#### 4.7.3 Where they **do** diverge sharply: group level

This is the real behavioural difference, and it is at panel scope, not field scope:

- **Placeholder → stay in place.** The field belongs on this page; the em-dash is an honest admission and a visible to-do. Four em-dashes in a panel of twelve is fine.
- **Bypassed → collapse the whole group** into the single inactive-group card (§4.4, already built and rendered in `skeleton.html`). A prefix panel showing six em-dashes on profile S is not honest-and-informative, it is **clutter that says one thing six times.** One card saying *"Prefix caching — not active on this server, and here's why"* is more honest and less noisy than six holes.

So the rule: **an isolated bypassed field renders as an em-dash; a bypassed *subsystem* renders as one explanation.** The panel-vs-scenario split the PM ratified holds — the panel still ships on profile D, where its number is real.

---

## 5. DASHBOARD PANELS

### 5.1 Grouping and priority

Six panels below the fold, in this DOM order. Order is the priority statement — the first two answer "is it fast?", the next two answer "why?", the last two answer "what exactly happened?".

| # | Panel | id | Group | Span | Cadence | Default |
|---|---|---|---|---|---|---|
| 1 | Throughput & latency | `throughput` | throughput | 2 | 250 ms + event | open |
| 2 | Scheduling & batching | `scheduling` | scheduling | 2 | 250 ms | open |
| 3 | KV memory | `kv-memory` | memory | 2 | 250 ms | open |
| 4 | Prefix cache | `prefix-cache` | cache | 1 | 250 ms | open |
| 5 | Requests | `requests` | scheduling | 2 | event-driven | open |
| 6 | System | `system` | system | 1 | 1 000 ms | collapsed |

**Cadence ruling.** 250 ms (4 Hz) for polled panels, matching the PM's spec: fast enough that block allocation and batch joins feel live, slow enough not to perturb a CPU-bound tiny-model workload. Perturbing the thing you are measuring is a real risk on small fixtures, so the cadence control offers 100 ms / 250 ms / 1 s / paused, and **`paused` is a first-class state, not a debug toggle** — reading a 4 Hz grid or taking a screenshot is impossible without it. When paused, every panel shows a `PAUSED · 00:04 ago` chip so a frozen number is never mistaken for a live one. That chip is a §4-adjacent honesty affordance: a stale number rendered as current is the same class of error as a fabricated one.

Event-driven values (per-request tok/s, TTFT, ITL) come from the SSE stream the demo is already consuming, coalesced to ≤10 Hz through one rAF. Never per-token DOM writes: at 32 concurrent streams that is thousands of layout invalidations a second and AC23 dies instantly.

History: a ring buffer of **240 samples at 250 ms = 60 s**, held in the store, shared by all panels. `Float64Array`, preallocated, no growth (AC22). Every sparkline is annotated **`60 s`** — an unlabeled percentile or window is a lie by omission.

### 5.2 Panel 1 — Throughput & latency

Primary figure: **aggregate output tok/s** `--og-type-xl`, with a 60 s sparkline beneath.

```
┌ Throughput & latency ──────────────────────────── 60 s ─ ⓘ ▾ ┐
│                                                              │
│   98.7 tok/s ᴰ            ▁▂▄▅▇▇▆▇█▇▆▅▇█▇▆▇▇▅▄▃▄▅▆▇         │
│   aggregate output                                           │
│                                                              │
│   ── per request ──────────────────────────────────────────  │
│   ● #1  12.4   ▲ #2  11.9   ■ #3  12.7   ◆ #4  12.0  tok/s   │
│                                                              │
│   ── latency ──────────────────────────────────────────────  │
│              p50        p95        max                       │
│   TTFT ᶜ    310 ms     520 ms     602 ms    ▁▂▁▁▃▂▁▁        │
│   TTFT ˢ    298 ms     501 ms       —       (server hist.)   │
│   ITL  ᶜ     41 ms      78 ms     113 ms    ▂▁▂▁▁▂▁▁        │
│   TPOT ᶜ     40 ms       —          —                        │
│   e2e  ˢ   4 210 ms   6 880 ms      —                        │
│                                                              │
│   makespan  7.9 s ᶜ   (scenario start → last completion)     │
└──────────────────────────────────────────────────────────────┘
```

**Showing client TTFT and server TTFT as two rows is a deliberate design choice.** They will diverge, and the divergence *is* the network + serialization overhead — which is interesting, and, more importantly, showing both is a public demonstration that we are cross-checking ourselves. A page that shows its own measurement disagreeing slightly with the server's is far more credible than one that shows a single confident number. Divergence >20% renders the pair in `--og-warn` with a hover explaining it, rather than silently picking a winner.

`makespan` gets its own line with breathing room. It is the metric that actually matters for a batching comparison and the one most demos omit.

Per-request rows use the sequence marker + colour + id, tying to swimlanes and block grid. Max 8 rows inline; beyond that the panel shows top-3/bottom-3 by tok/s plus a count, and defers the full list to the Requests panel. A 32-row list in a 340px panel is a wall, and a wall is not information.

### 5.3 Panel 2 — Scheduling & batching

```
┌ Scheduling & batching ──────────────────────────── 60 s ─ ⓘ ▾ ┐
│   running  6 ˢ    waiting  2 ˢ    admission slots  248 ˢ      │
│                                                               │
│   active decode rows  ▂▄▆███▇▆▄▂▁▂▄▆██▇▆▄▃▂▁   (count, no %) │
│     6 of 4 max  ← if max_batch unavailable: "6 sequences ᴰ"   │
│                   and the % form is em-dashed, not guessed    │
│                                                               │
│   queue depth          ▁▁▂▅█▆▃▁▁▁▁▂▄▇█▅▂▁▁▁                   │
│     peak 8 · now 2 ˢ                                          │
│                                                               │
│   decode steps/s   214 ᴰ      tokens / decode step   5.8 ᴰ    │
│   engine batch size  — ⓘ      rejections               0 ˢ    │
└───────────────────────────────────────────────────────────────┘
```

**Batch occupancy is the trap in this panel — and the trap is WORSE than this section originally claimed.** I wrote that the denominator (`max_batch`) is unsurfaced while *"the numerator is real."* 🔴 **The numerator is not real either.** `onnx_genai_batch_size_current` is `fetch_add(1)` in `GenerationMetrics::start()` (`metrics.rs:111`/`:145`), decremented on `Drop` — it counts **generation requests in flight at the HTTP layer**, with no connection to `ContinuousBatchManager`. So the "occupancy" ratio was **a fabricated denominator under a mislabelled numerator, and only the denominator was ever suspected.** Render `batch.in_flight` as an absolute count labelled **"requests in flight"**, `batch.queued` as `max(0, in_flight − max_batch)` (derived), and the engine's true batch size as `unavailable`. **Never the words "batch size" for either (AC59).** Do not assume `DEFAULT_MAX_BATCH = 4` from `state.rs:25` — a hard-coded denominator is a fabricated measurement wearing a division sign.

`rejections: 0` and `engine batch size: —` sitting adjacent is the clearest teaching example of §4 anywhere in the UI: one is a real, good zero; the other is an absence. Their visual difference is the entire thesis, on one line. Worth screenshotting for the README.
> **This pairing replaces an earlier one that used `preemptions: —`, dropped when preemption was ruled out entirely (D148). The replacement is strictly better teaching: the engine's true batch size is absent *in the very panel whose headline claim is about batching*, so the visitor learns the distinction where it costs us something to admit — rather than about a subsystem they had no reason to expect.**

`allocation_failures > 0` renders in `--og-bad` with an alarm affordance — the KV crate's own comment says a run with failures is thrashing, and the page should say so too.

### 5.4 Panel 3 — KV memory

The numeric companion to Scenario C's grid; carries the numbers a grid cannot show precisely.

```
┌ KV memory ─────────────────────────────────────────── ⓘ ▾ ┐
│   62.1 %  utilization ᴰ        318 / 512 blocks ˢ          │
│   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░  utilization    │
│                                                            │
│   block size  16 tok ˢ    shared  96 (30.2 %) ˢ            │
│   slot fill   90.4 % ᴰ    4 602 / 5 088 slots ˢ            │
│     └ the gap is the cost of paging: partially filled      │
│       blocks. This is real and we show it. ⓘ               │
│                                                            │
│   refcount    1 ████████████████████ 222                   │
│               2 ██████ 60                                  │
│               3 ███ 36                                     │
│                                                            │
│   tiers       cpu 512 ˢ                                    │
│                                                            │
│   allocations 1 204 ˢ    frees        886 ˢ                │
│   alloc fails     0 ˢ    hot evict      — ⊗                │
│   prefix evict    — ⊗   (no consumer: driver.rs:735)      │
└────────────────────────────────────────────────────────────┘
```

Surfacing **slot fill efficiency with its explanation inline** is a deliberate act of self-criticism: it is the honest cost of paging, the KV crate's own doc comment names it, and volunteering an imperfection is the strongest available signal that the visualization is real rather than a marketing render. Hiding it would be the tell.

Refcount distribution is a horizontal bar list, not a chart. Three-to-five discrete integers do not need an axis, and a bar list is exactly readable.

### 5.5 Panel 4 — Prefix cache — 🔴 STRUCK. CUT. DOES NOT SHIP, IN ANY FORM, ON EITHER SERVER.

> 🔴 **SUPERSEDED — and this was the most dangerous prose left in this document, so the strike is verbose on purpose.**
>
> **It said, in bold: *"This panel ships unconditionally, whatever the numbers say… The scenario is cuttable; the panel is not."*** Now false at every level: @12e42da8 ruled **no prefix field ships in any form, either mode**, after @fc8b5d97 measured shared-prefix requests **7.0% SLOWER** than a zero-sharing control, against a sensitivity floor where a working cache would have collapsed TTFT ~90%. **Scatter records nothing (0/135); dynamic records everything (19/20, incrementing on six control requests that share nothing). Broken in opposite directions, neither trustworthy.**
>
> **And the sketch carried FABRICATED NUMBERS — `87.2 % hit rate ˢ`, `41 hits / 47 lookups ˢ`, a populated sparkline — every one marked `ˢ` FOR SERVER-MEASURED.** Nobody measured them; they were plausible filler for a layout. **A design sketch is a build instruction, and this one told a developer to render invented values under our own provenance badge, in the panel of the feature we later proved absent.** Retained struck rather than deleted, because it is the clearest example this project produced of the exact failure the honesty layer exists to prevent — **and it was OURS, in the document that DEFINES the honesty layer.**
>
> **What ships instead: the measured non-result (§51), both arms and the sensitivity check on screen.**

~~**This panel ships unconditionally, whatever the numbers say.** The *scenario* is cuttable; the *panel* is not (§7.5).~~

```
STRUCK — FABRICATED VALUES UNDER A SERVER-MEASURED BADGE. DO NOT BUILD.
┌ ...cache ─────────────────────── ⓘ ▾ ┐
│   87.2 %  hit rate ˢ                  │  ← invented, badged as measured
│   41 hits / 47 lookups ˢ              │  ← invented, badged as measured
│   ▁▁▃▅▇███▇▇▇▇▇▇▇▇  60 s              │  ← a fabricated observation window
│   tokens reused    81 344 ˢ           │  ← invented, badged as measured
└───────────────────────────────────────┘
```

**The sparkline is the worst line in it, and it generalises past this panel** (@e00032a4's formulation, adopted): **a chart asserts DURATION, not just value.** A table cell claims one reading; a plotted series claims *"we watched this for sixty seconds and it stayed there."* **It fabricates an observation window we never had** — which is why gap-bridging is banned outright in §51: **interpolation produces the most convincing false data there is, the kind that looks continuous.**


### 5.6 Panel 5 — Requests

One row per request in the current scenario. This is the audit surface: an engineer who does not believe a number needs to be one click from the raw evidence.

`marker · id · state · sent · pre-first-token · TTFT · in/out · tok/s · KV blocks · reused · finish`

Client-observable state vocabulary — **fixed, and narrower than the PM's server-side machine on purpose**:

`sent → streaming → done | error | cancelled`

The PM's §3 vocabulary (`queued → admitted → prefilling → decoding → preempted → finishing → done`) is the *server's* lifecycle, and `recon-map.md` §7.5 confirms no such state enum exists server-side. A browser cannot observe those transitions. **Rendering the richer vocabulary from client timing would be fabrication in the most invisible possible place**, so the client vocabulary is what ships. If telemetry later supplies a server-side state, the column gains a second server-sourced badge and the richer states light up — additively, with an `ˢ` badge, and never by inference.

Rows expand to the raw SSE event log with timestamps. Sortable by any column; virtualized above 32 rows.

### 5.7 Panel 6 — System

Model id, directory, context length, EP, decode backend, quantization, uptime, active sessions, RSS/VRAM used-vs-limit (`/v1/resources` gives `used`/`limit`/`headroom`), server version, and — **the self-aware one** — the dashboard's own poll RTT and dropped-frame count.

Showing our own overhead is not vanity. The PM explicitly asks for it, and a measurement tool that reports its own cost is making a statement about what kind of tool it is.

---

## 6. SCENARIO A — CONTINUOUS BATCHING (the swimlane timeline)

**The claim:** a request arriving mid-decode joins the running batch on the next step instead of queueing behind in-flight work.

**Both sides are real.** Static batching is a *client-side request policy* — fixed waves of size `B`, wave `k+1` not sent until every request in wave `k` completes. That is exactly what static batching does to a workload: head-of-line blocking on the slowest member. Nothing is simulated, ever. The staircase is earned.

### 6.1 What the client can honestly observe

This constrains the drawing more than anything else, so it comes first.

A browser sees exactly three events per request: **sent** (`fetch` dispatched), **first token** (first SSE `delta.content`), and **complete** (`data: [DONE]`), plus one timestamp per intermediate chunk.

It **cannot** distinguish queue wait from prefill. So the bar has **two** segments, not four:

| Segment | Definition | Treatment |
|---|---|---|
| **pre-first-token** | sent → first token | hollow: 1px solid outline in the sequence colour, transparent fill, 45° hatch at `--og-fg-faint` |
| **streaming** | first token → completion | solid fill in the sequence colour + the sequence's pattern, with a 1px tick per SSE chunk |

The PM's spec draws `queued` and `prefill` as separate segments. **I am overriding that for the client-observed view**, because splitting them from client timing would be a fabricated boundary — invisible, plausible, and exactly the kind of thing AC6 exists to prevent. If telemetry supplies a real `queue_wait_ms`, it renders as a **sub-rule inside** the pre-first-token segment: a vertical dashed divider with an `ˢ` badge on the segment, so the added precision is visibly server-sourced and its absence is visibly an absence.

The **hatch on the hollow segment is doing real semantic work**: it is the same 45° hatch as §4.3's unavailable wells, which trains one visual idea — *hatch means "we cannot see inside here."* Because that is literally true of the pre-first-token interval from the client's vantage point. Reusing the language rather than inventing a second one is what makes a design system a system.

### 6.2 Lane anatomy

```
      ┌ label gutter 132px ┐┌────────── track ─────────────────────────────┐
 ● #1 ┃ 1,024 tok          ┃┃▨▨▨│████│█│█│█│█│█│█│█│█│█│█│█│█│█│█│●        ┃
 ▲ #2 ┃ 1,024 tok          ┃┃  ▨▨▨▨│███│█│█│█│█│█│█│█│█│█│█│█│█│█│●        ┃
 ■ #3 ┃ 1,024 tok          ┃┃    ▨▨▨▨▨│██│█│█│█│█│█│█│█│█│█│█│█│█│●       ┃
 ◆ #6 ┃ 1,024 tok          ┃┃            ▨▨│█│█│█│█│█│█│█│█│█│█│█│█│●     ┃
      └────────────────────┘└──────────────┬───────────────────────────────┘
                            0s      1s     2s  ↑ now             3s     4s
                                            #6 joins — #1–#3 never pause
```

- **Label gutter, 132px:** shape marker + colour swatch + request id + prompt/max tokens. The marker means a grayscale screenshot still identifies lanes (AC25).
- **Adaptive lane height** — this is what makes 1–32 lanes work in one component: `N ≤ 8` → 18px with full labels; `9–16` → 12px, id only; `17–32` → 7px, no text, gutter becomes a colour+pattern strip and identity moves to hover + the legend. Below 7px lanes stop being readable, so 32 is the honest ceiling and the control caps there.
- **Chunk ticks:** a 1px vertical rule at each SSE chunk arrival. At close zoom you literally see tokens land. Ticks are decimated to ≥2px spacing when zoomed out — decimation is honest (it is sampling), a solid block would not be.
- **Decode-step gridlines:** faint vertical rules across *all* lanes. **This is the money shot.** Continuous batching is visible as ticks in different lanes landing on the same gridline — the lockstep IS the phenomenon. Client-side these gridlines are drawn at the median inter-chunk interval across active lanes and are therefore `ᴰ`, labelled `inferred decode cadence ᴰ` in the legend. If server-side `decode_steps_total` is available, gridlines snap to real step boundaries and the label upgrades to `ˢ`. Never present the inferred version as the real one.
- **Now-line:** 1px `--og-info` vertical rule sweeping right, with the elapsed time in a small chip at the top. Under `prefers-reduced-motion` it steps at 250 ms instead of animating.
- **Completion cap:** a filled circle at the bar end plus the finish reason on hover. `length` vs `stop` are different stories and the visitor should be able to tell.

### 6.3 The A/B presentation — and why stacked, not side-by-side

Two panels, **STATIC on top, CONTINUOUS below, sharing one X axis with a single locked pixels-per-millisecond scale**.

Side-by-side halves the time resolution and, worse, invites the eye to compare *shapes* rather than *extents*. Stacked with a shared axis makes the comparison the only one that matters: **static's mass extends further right.** The staircase and the makespan difference land in the same glance, with no chart-reading skill required.

```
 STATIC  B=4                                        makespan 18.4 s ᶜ
 ● #1 ┃▨▨│███████████│●                                            ┃
 ▲ #2 ┃▨▨│███████│●                                                ┃
 ■ #3 ┃▨▨│██████████████████│●   ← the straggler holds the wave    ┃
 ◆ #4 ┃▨▨│████████│●                                               ┃
      ╎                        ╎ wave boundary — nothing sent yet  ╎
 ◇ #5 ┃                        ▨▨│█████████│●                      ┃
 ★ #6 ┃                        ▨▨│███████│●                        ┃
      └──────────────────────────────────────────────────────────┘
 CONTINUOUS                                          makespan 7.9 s ᶜ
 ● #1 ┃▨▨│███████████│●                                            ┃
 ▲ #2 ┃▨│███████│●                                                 ┃
 ■ #3 ┃▨│██████████████████│●                                      ┃
 ◆ #4 ┃▨│████████│●                                                ┃
 ◇ #5 ┃ ▨│█████████│●                                              ┃
 ★ #6 ┃  ▨│███████│●                                               ┃
      └──────────────────────────────────────────────────────────┘
```

**Wave boundaries in static mode** are the staircase made explicit: a full-height vertical dashed rule in `--og-warn` at each wave transition, labelled `W1 W2 W3`, with the **dead time** between the last completion of wave *k* and the first send of wave *k+1* filled with the §4.3 hatch and labelled `head-of-line blocking · 2.4 s`. Naming the dead time is what converts a chart into an argument.

**Makespan brackets:** a horizontal bracket under each panel spanning first-send → last-completion, with the value. Two brackets of visibly different length, vertically aligned, is the entire result of the experiment in one mark.

**The delta table** sits below, and **whatever the delta is, it is shown** (AC11). Hard rules, review-enforceable:

- No flooring, no clamping, no best-of-N selection. If continuous wins by 1.2x on a CPU fixture, it says 1.2x.
- If continuous **loses** on some measure, that row renders in `--og-warn` with the delta as measured. A demo willing to show a row where it lost is a demo whose other rows are believable.
- Every row states its `n` and the run timestamp. A single-run delta is labelled `single run` — a percentile from one sample is not a percentile.
- Both sides must use identical prompts, `max_tokens`, and model. The table header states this, and the scenario refuses to render a comparison if the two runs used different parameters. **A comparison that silently changed a variable is worse than no comparison**, so this is a hard guard, not a warning.

**Ordering is fixed: static first.** Running continuous first and static second lets the second run benefit from a warm prefix cache and warm allocator, which would inflate our own result. Static-then-continuous is the *conservative* ordering — it disadvantages us — and the panel says so in the methodology hover. Choosing the ordering that could only hurt our number is a small thing that a careful reader will notice, and careful readers are the audience.

**Prompt mix presets:** `uniform` · `mixed-length` · `one long straggler`. The straggler (one 512-token request among seven 32-token ones) is the most visceral illustration of head-of-line blocking available and costs one preset to build. Default it.

### 6.4 Interaction

Hover a lane → cross-highlight that sequence's blocks in the KV grid, dim others. **This cross-highlight between the two hero visualizations is the highest-value interaction on the page**: it is what makes "sequence" and "memory" one concept rather than two panels. Keyboard equivalent: `↑`/`↓` moves a lane cursor with the same effect, so it is not a mouse-only insight.

Click a lane → pins it, opens its Requests row. Drag on the axis → zoom; `0` resets. `Space` run/stop.

---

## 7. SCENARIO B — PAGED KV BLOCK TABLE

**Always "Paged KV block table". Never "paged attention."** The kernels are not implemented; the allocator is real, and the allocator is what we show. The AC9 statement (§4.6, verbatim) is rendered as visible UI text next to the title — permanently, not behind a tooltip.

### 7.1 Cell states — every one encoded by shape as well as colour

Colour identifies *which sequence*. **Shape identifies *what state*.** A grayscale screenshot must remain fully legible, which is the same property that makes it work for a colourblind reader (AC25).

| State | Encoding | Why this shape |
|---|---|---|
| **free** | empty cell, 1px dotted border in `--og-fg-faint`, no fill | minimum ink — free memory should be visually quiet so used memory is the figure |
| **owned** | solid fill, sequence colour **+ sequence pattern** | pattern carries identity without colour |
| **partially filled** | fill height ∝ `filled_slots / page_size`; remainder rendered as free | **fill level is inherently a shape** — no extra encoding needed, and it is the honest picture of paging cost |
| **shared** (`refcount > 1`) | bold 1.5px outline ring **+ a corner dog-ear triangle** + refcount numeral at detail scale | the dog-ear reads as "this page is folded into more than one place" and survives grayscale |
| **demoted tier** | desaturated fill + a horizontal strike rule across the cell | strike = "not here anymore" |
| **just allocated** | 1-frame 1.5px outline flash in `--og-info` | motion backed by a real event |
| **just freed** | 1-frame ✕ overlay fading over `--og-dur-slow` | ditto |
| **unavailable** | **the grid is not drawn at all** → capability notice (§4.4) | a grid of "free" cells when the truth is "no data" is the worst lie available in this panel |

That last row is the one to get right. A block grid rendered entirely in the free state looks like *a system with no memory pressure*, not like *a system we cannot see into*. It is the §4.3 flat-zero-line failure in two dimensions.

`prefers-reduced-motion`: allocation/free flashes become a 250 ms persistent corner dot instead of a fade. The event is still communicated; nothing animates.

### 7.2 Scaling — 64 blocks to 4096

One component, three density modes, chosen automatically from `capacity` and overridable by the `zoom` control. Blocks are laid out row-major at a **fixed 64 per row**, with a slightly stronger rule every 8 rows so the eye can count — a stable rectangle whose *shape* does not change as capacity grows, only its height.

| Mode | Capacity | Cell | Gutter | Shown | Hover target |
|---|---|---|---|---|---|
| **detail** | ≤ 256 | 16px | 2px | pattern, fill height, dog-ear, **refcount numeral** | the cell |
| **compact** | 257–1024 | 8px | 1px | pattern, fill height (3-step: empty/partial/full), dog-ear | the cell |
| **density** | > 1024 | 3px | 0 | colour + fill-as-lightness only; sharing shown as a 1px overlay rule | an **8×8 super-cell** showing an aggregate |

- 64 blocks → detail, 4 rows of 16px. Small, crisp, every cell individually readable and hoverable.
- 512 blocks → compact, 8 rows of 8px ≈ 580px wide. The sweet spot; this is what most demos will show.
- 4096 blocks → density, 64 rows of 3px ≈ 192px tall. Individual cells are no longer meaningful and **we must not pretend otherwise** — hover targets become 8×8 super-cells reporting *aggregates* ("64 blocks: 51 owned by 3 sequences, 9 shared, 4 free"), and the numeral/dog-ear detail is dropped rather than rendered illegibly. Rendering a 3px dog-ear is drawing precision that cannot be perceived, which is the visual equivalent of a spurious significant figure.

**Two companions make every scale readable, and they are mandatory, not optional:**

1. **The utilization ribbon** — a single full-width 20px horizontal bar above the grid, segmented `free | owned | shared | demoted`, each segment patterned and labelled with its count. This is the readable view *at any capacity*, including 4096. If the grid is the microscope, the ribbon is the naked eye, and the ribbon never becomes illegible.
2. **Brush-to-zoom** — drag a rectangle over the grid (or `⇧`+arrows) to open those blocks in a detail inset at 16px cells. This is how density mode stays interrogable without lying about what it can render.

### 7.3 Layout

```
┌ Paged KV block table ─────────────────────────────────── ⓘ ▾ ┐
│ Visualizes the paged KV allocator — real block allocation,    │
│ copy-on-write sharing, tiering, and eviction. Paged-attention │
│ kernels are not yet implemented. We show what exists.         │  ← AC9, always visible
│                                                               │
│ ┌ ribbon ─────────────────────────────────────────────────┐  │
│ │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│  │
│ │ owned 222   shared 96   free 194   ·   512 × 16 tok     │  │
│ └─────────────────────────────────────────────────────────┘  │
│                                                               │
│ ▨▨▨▨▨▨▨▨░░░░████████▨▨▨▨░░░░░░░░████▨▨▨▨▨▨▨▨░░░░░░░░░░░░░░░░ │
│ ████████▨▨▨▨▨▨▨▨░░░░░░░░████████████░░░░░░░░▨▨▨▨░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│                                                               │
│ colour by:  ( sequence ) refcount   tier   fill               │
│ legend:  ● seq 7  ▲ seq 8  ■ seq 9   ⌐ shared   ░ free        │
│                                                               │
│ memory pressure  ├────────●────────┤ 45 %                     │
│                                                               │
│ alloc 1 204 ˢ · free 886 ˢ · fail 0 ˢ · hot evict — ⊗ ·       │
│ prefix evict — ⊗                                              │
└───────────────────────────────────────────────────────────────┘
```

**`Colour by` (sequence | refcount | tier | fill)** is small to build and disproportionately valuable: the same grid answers four different questions, and switching between them is exactly how an engineer interrogates a visualization they do not yet trust. Each mode swaps the *colour scale* but **keeps the shape encodings**, so the grid never becomes colour-only. In `refcount` and `fill` modes the scale is sequential single-hue (`--og-info` ramp) with a discrete 4-step legend — never a continuous rainbow, which fabricates precision and is a colourblind hazard.

### 7.4 Memory pressure — REWRITTEN, and the payoff moved

> ⚠️ **This subsection previously described eviction and preemption. That design was fabricated and is deleted.** @d7cf9b84 verified: `ByteBudget::reconfigure` (`scheduler/src/byte_budget.rs:180-190`) changes the ceiling and **never touches `used`**. The repo's own test is named `reconfigure_lower_reports_overage_without_evicting` (`:301-312`). The governor computes `eviction_order` and `overage_bytes` and **nothing consumes them** — our driver discards the result (`driver.rs:735-739`). Separately, `batched.rs:757` hardcodes `PreemptionPolicy::Disabled`. **Lowering the budget frees nothing, moves no cell, and animates nothing.** I had storyboarded six narration beats of a feature that does not exist.

**The real behaviour, and it is a better story:** lowering the limit does not reclaim memory already held — it **refuses the next allocation**. `used > limit` means new requests are turned away. What you observe is genuine and it propagates: allocation failures rise, admission starts rejecting, requests visibly queue and wait.

**So the payoff moves off the block grid and onto admission.** That is the substantive change, not a wording fix. The old design's climax was *cells changing colour*; the real climax is **backpressure reaching the front door**. The grid goes **still** — and its stillness is the point, because a full pool that cannot free anything is exactly what the runtime does.

Narration, replacing all six fabricated beats:

> `KV budget lowered to 40% — nothing is freed; the pool already holds what it holds`
> → `8 requests sent`
> → `allocation refused — the pool is above its new ceiling`
> → `admission rejecting: 5 — backpressure has reached the front door`
> → `queue depth 5 and holding`

Panel copy, verbatim, because the misreading is otherwise guaranteed:

> **Lowering the budget does not reclaim memory already held — it refuses new allocations. Watch admission stall.**

#### 7.4.1 The control is a SEQUENCE, not a slider

The runtime only rewards one order: **lower the budget first, then send load.** Drag a slider with no traffic in flight and *nothing happens at all*.

That kills the slider. **A slider affords continuous exploration; this runtime rewards exactly one sequence.** Offering a control whose most obvious use — drag it and watch — produces no observable effect is worse than offering no control: the visitor concludes the dashboard is broken, on the one panel whose subject is failure. We would have manufactured the exact misreading the panel exists to prevent.

So Scenario C is **two numbered steps with one button each**, the second disabled until the first completes:

```
STEP 1  ▸ Tighten the KV budget to 40%      [ Apply ]
        Nothing will change yet. That is the point — read why ↓
STEP 2  ▸ Send 8 concurrent requests        [ Run ]      (enabled after step 1)
        Watch admission stall.
```

**Step 1 explicitly promises that nothing will happen.** Making the null result *expected* converts the demo's weakest moment into its most credible one — a visitor who is told in advance that the grid will not move, and then watches it not move, trusts the next thing we say. A dashboard that predicts its own inaction is making a falsifiable claim and then passing.

#### 7.4.2 What still stands

`allocation_failures > 0` escalates the counter strip to `--og-bad`: *"the allocator is refusing new work — this is what running out of KV looks like."* Naming a bad state plainly, while it happens, is worth more than any successful frame.

**The `SIM`-badged fallback is deleted, not downgraded.** My earlier third option — a simulated constraint if real pressure could not be produced — is gone. Real pressure is reachable (raise concurrency and prompt length until the pool genuinely fills), so the fallback existed only to guarantee a dramatic frame. That is the exact trade this project exists to refuse.

### 7.5 Cross-highlight

Hover/focus a sequence anywhere — lane, table row, legend chip — and its blocks light while everything else drops to 30% opacity. Bidirectional: hover a block, its owning sequence(s) highlight everywhere. Refcount > 1 highlights *all* owners simultaneously, which is the single clearest possible explanation of what sharing means. `Esc` clears; `Tab` order is legend → ribbon → grid → controls.

---

## 8. SCENARIO C — PREFIX CACHING (designed to be cut cleanly)

**Status: at risk.** Five identical prompts currently report `prefix_cache_hits: 0, lookups: 5`. A developer is investigating. Two flawless scenarios beat three shaky ones.

### 8.1 The cut rule — decide by evidence, and decide once

> **Ship the scenario only if a cache hit is reproducibly observable.** The gate is: *the same prompt sent twice produces `hits ≥ 1` and a visibly lower TTFT, ten times out of ten.* If it does not, the scenario is cut. There is no middle setting — a scenario whose payoff sometimes fails is worse than no scenario, because the visitor's takeaway becomes "prefix caching doesn't work here", which is a stronger negative claim than we would ever have made ourselves.

**And the mirror rule, which is the important one:** the **panel** ships regardless (§5.5), rendering whatever is true, including `0.0 %`. Because `hits: 0` is a *real measurement*, not a documented zero, and honestly reporting an unflattering real number is the behaviour that makes every other number on the page believable. **We cut the scenario for demo-quality reasons, never to hide a number.** These two decisions must not be conflated, and if anyone proposes hiding the panel along with the scenario, that is the moment to escalate.

### 8.2 Structural provisions so the cut leaves no hole

Four, all already in the architecture, all costing nothing:

1. **Tabs are content-driven.** The tablist is generated from a scenario registry array. Cutting = deleting one entry in `scenarios/index.js`. The row re-flows; there is no fixed 3-up layout with a gap.
2. **The stage is a single slot.** One scenario mounts at a time. Nothing reserves space for an absent one.
3. **Hero slot 2 has a specified fallback** — `tokens per decode step`, real and derived — and the strip is `auto-fit`, so it also survives simply becoming three tiles.
4. **The panel grid is `auto-fill`.** Any panel can leave without a hole. (The prefix panel is not leaving, but the property is what makes the whole grid safe.)

The cut is therefore: delete `scenarios/prefix.js`, delete one registry line, delete one hero-tile config line. **No CSS changes, no layout changes, no other file touched.** That is what "cuttable" has to mean to be real, and it is verifiable by inspection before anyone commits to building it.

### 8.3 The design, if it ships — the TTFT ladder — 🔴 STRUCK. IT DOES NOT SHIP.

> 🔴 **SUPERSEDED. The conditional in this heading — *"if it ships"* — resolved to NO, and the sketch below is the second fabricated prefix ladder in this document.** `1 984 / 2 013 reused (98.6%)`, `#2 HIT 38 ms` against a `1 842 ms` cold bar: **a 48× TTFT collapse, invented, badged `ˢ` for server-measured.** The real measurement went the other way — **shared prefixes were 7.0% SLOWER than a zero-sharing control.**
>
> **This is why the strike is worth the space rather than a delete: the sketch is a near-perfect rendering of what a WORKING cache would have looked like, and we drew it before we measured.** The bars, the badges and the percentage were all chosen to be persuasive, and the only reason they are not on the page today is that someone ran the control arm. **A layout drawn from an expectation is a prediction wearing the costume of a result** — and nothing in our provenance envelope can catch it, because the fabrication happens upstream of every value it would check.


```
┌ Prefix caching ────────────────────────────────────── ⓘ ▾ ┐
│ preset: few-shot document · 2 013 tokens shared            │
│                                                            │
│ #1 cold ████████████████████████████████  1 842 ms ᶜ       │
│                                     0 / 2 013 tok reused ˢ │
│ #2 HIT  █                                   38 ms ᶜ        │
│                                 1 984 / 2 013 reused (98.6%)│
│ #3 HIT  █                                   41 ms ᶜ        │
│ #4 HIT  █                                   39 ms ᶜ        │
│ ════════════════ cache cleared ═══════════════════════════ │
│ #5 cold ███████████████████████████████   1 798 ms ᶜ       │
│                                     0 / 2 013 tok reused ˢ │
│         ┆ cold reference                                   │
└────────────────────────────────────────────────────────────┘
```

- **Cold vs hit is encoded by fill pattern and an explicit text label**, never colour alone: cold = solid, hit = solid + a leading `▮` cap and the literal word `HIT`.
- **A vertical reference rule at #1's cold TTFT** persists down the ladder, so every hit bar is measured against the same visible datum.
- **The `cache cleared` rule is a full-width double line with a label.** The before/after is one glance, and the *reversibility* is what makes it credible — anyone can fake a fast number once; nobody fakes it snapping back on demand.
- **`tokens reused / tokens prompted` as absolutes, not just a percentage.** `1 984 / 2 013` is concrete in a way `98.6 %` is not.
- **The bar is TTFT, which is client-measured (`ᶜ`); the reuse counts are server (`ˢ`).** Two sources, two badges, on the same row — a small, constant demonstration that we know where our numbers come from.

The `branching conversation` preset (same 4-turn history, six different follow-ups) proves reuse is over the **common prefix**, not append-only extension. If it ships, the small prefix-tree strip beneath the ladder — shared trunk, branch points, node width ∝ tokens — is the one piece of "extra credit" I would keep, because it visualizes the trie the README describes and a bar chart cannot. The `no shared prefix` control preset is **mandatory if the scenario ships**: a demo without a control is an advertisement.

**Success looks like:** a visitor hits `Clear cache` a second time just to watch the bars snap back. That reflex is the proof they believed it.

---

## 9. ACCESSIBILITY — the token set, verbatim

Not a compliance pass. Three of these requirements (colour-independence, tabular figures, focus visibility) make the page better for *everyone*, and the colour-independence one is the same work that makes it survive a projector, a grayscale print, and a compressed screenshot in a blog post. Accessibility here is paid for once and spent four times.

### 9.1 Colour independence (AC25) — the rule and its enforcement

> **No state, category, or magnitude may be communicated by colour alone. Every colour-carried meaning is redundantly carried by shape, pattern, position, or text.**

| Surface | Colour carries | Redundant encoding |
|---|---|---|
| Swimlanes | sequence identity | pattern fill + shape marker + id in gutter |
| Swimlane segments | phase | **hollow-hatched vs solid-filled** — a shape difference |
| Block grid | sequence identity | SVG/canvas pattern per sequence |
| Block: free | — | empty + dotted border (no fill at all) |
| Block: shared | — | bold ring + **corner dog-ear** + numeral |
| Block: partial | — | **fill height** |
| Block: demoted | — | strike rule |
| Connection chip | status | **glyph shape** ● ◐ ◌ ○ + text label |
| Unavailable value | — | **em-dash glyph** + dashed underline |
| Unavailable series | — | **hatch fill** + `NOT MEASURABLE YET` text |
| Estimated value | — | `~` prefix + `ᴱ` badge |
| Simulated | — | full-word `SIM` pill + dashed outline |
| Delta table regression | attention | the sign and the number, which are the actual information |

Verification is a checklist item, not an opinion: run the page through a deuteranopia/protanopia/tritanopia simulator **and** a full grayscale filter. **If any panel loses meaning in grayscale, it fails.** Grayscale is the stricter and easier test — use it as the gate.

### 9.2 Contrast

All body text ≥ 4.5:1 against its surface; graphical boundaries that carry meaning ≥ 3:1. Verified pairs against `--og-bg-raised` (#151b23): `--og-fg` 17.4:1 · `--og-fg-muted` 8.1:1 · `--og-fg-subtle` 4.6:1. **`--og-fg-faint` is 1.9:1 and is decorative only** — grid texture, hatch, hairlines. Any text in `--og-fg-faint` is a bug.

The Okabe-Ito hues against `--og-bg-raised` all exceed 3:1 except `--og-seq-6` (yellow, very high) which needs a dark 1px border when used as a fill — `.seq-6 { outline: 1px solid var(--og-bg); }`. Encoded in the token layer so it cannot be forgotten.

### 9.3 Keyboard (AC26)

Focus order follows visual order, top-to-bottom, left-to-right. **The focus ring (`--og-focus-ring`) is never removed anywhere, for any reason.** `:focus-visible` for pointer users, but the ring must render for keyboard users on every interactive element including canvases and grid cells.

| Key | Action |
|---|---|
| `Space` | run / stop scenario |
| `R` | reset |
| `1` `2` `3` `4` | switch scenario |
| `M` | model card popover |
| `P` | pause / resume polling |
| `?` | shortcut overlay |
| `Esc` | close popover / clear highlight / exit brush |
| `Tab` / `⇧Tab` | move between regions |
| `←` `→` `↑` `↓` | move within a composite widget (lane cursor, block cursor) |
| `Home` / `End` | first / last item in a widget |
| `Enter` | activate / pin the focused lane or block |

**Composite widgets are one tab stop, not N.** The block grid at 512 cells must never be 512 tab stops — it is a single stop with a roving cursor (`aria-activedescendant`), arrows to move, `Home`/`End`/`PageUp`/`PageDown` to jump by row. Same for swimlanes. This is the difference between "keyboard accessible" and "keyboard usable", and a 4096-cell tab trap would be the single worst accessibility failure the page could ship.

The `?` overlay is a real dialog (`role="dialog"`, focus trapped, `Esc` closes, focus restored) — the one deliberate exception to "nothing is modal", because a shortcut sheet is inherently modal and everyone expects it to be.

### 9.4 Screen readers (AC28)

- Landmarks: `<header role="banner">`, `<main>`, `<section aria-labelledby>` per panel, `<footer role="contentinfo">`.
- The narration strip is `aria-live="polite"` and is the primary channel for scenario progress. **Throttled to one announcement per 1.5 s minimum** — a 4 Hz live region is unusable, and a well-meaning dev will discover this the hard way if it is not written down.
- Every canvas is `role="img"` with `aria-label` set from `describe()`, refreshed at most every 2 s.
- **Every panel has a "view as table" toggle**, shell-provided, rendering a real `<table>` with `<caption>` and scope'd headers. This is the actual accessible path for data — an `aria-label` on a canvas is a summary, and a table is the data. Both are required.
- Unavailable values carry the full reason in `aria-label` so it is available without hover (§4.1).
- Numbers use `aria-label` with units spelled out: `"98.7 tokens per second"`, not `"98.7 tok/s"`.

### 9.5 Motion (AC27)

`prefers-reduced-motion: reduce` zeroes the three duration tokens in one place, and additionally: the now-line steps at 250 ms rather than sweeping; grid allocation/free flashes become persistent 250 ms dots; the connection pulse becomes a discrete opacity step; sparklines redraw without transition.

**No flashing above 3 Hz** anywhere. The block grid at 4 Hz with fade transitions is the risk: allocation flashes are `--og-dur-slow` (260 ms) and **at most one flash per cell per poll**, which caps any cell at 4 Hz of *fade*, not of *flash*. Test with a full-pool churn workload before shipping — this is a real seizure-safety requirement, not a nicety.

---

## 10. GLOBAL INTERACTION RULES

1. **Every control produces a visible change within 100 ms**, even when the real effect takes seconds. Move the stagger slider → ghost arrival markers move on the timeline *now*, before anything is sent. A control with no immediate feedback feels broken, the visitor stops touching things, and a demo whose entire value is in being touched dies quietly.
2. **Nothing is modal** except the `?` overlay. No wizards, no dialogs, no blocking confirms.
3. **Every number is hoverable** and reveals exact value, unit, source class, and cadence. This is the audit mechanism that lets an engineer verify us without reading docs (AC7).
4. **Errors are UI, never console.** "Failed to fetch" is not an acceptable user-facing string. Every error names the thing, the cause, and the fix.
5. **Deep links** (AC18): `#scenario=batching&n=16&stagger=200&mode=ab&panels=throughput,kv-memory`. Written on change, read on load. This is how the demo spreads — someone finds a striking configuration and pastes a URL into an issue thread.
6. **Acronyms** (AC30): every one is an `<abbr>` with a dotted underline and a definition, declared in `meta.acronyms` and rendered on **first appearance per panel** — not once per page, because panels are read out of order.

---

## 11. HANDOFF CHECKLIST

**demo dev — your critical path is not a scenario.** It is `telemetry.js` + `format.js`, because the dashboard dev is blocked on both. Ship a stub store returning correctly-shaped `Field`s from a hand-written fixture JSON on day one, before real polling works; that unblocks the dashboard in parallel and lets both of you build against the same contract from hour one.

- [ ] `styles/tokens.css` imported first, before every other stylesheet
- [ ] SVG `<defs>` with `og-pat-0`…`og-pat-7`, plus the matching `CanvasPattern` builders in `format.js`
- [ ] `store.field()` returns a `Field` for **every** path including unknown ones — never `undefined`, never a throw
- [ ] `reason` is populated for every non-`ok` field, using §4.2 copy verbatim
- [ ] Single-flight polling with backoff and `AbortController` (AC24)
- [ ] Both blocking failure states built and manually triggerable (kill the server; start with no model) — **build these in week one, not at the end**
- [ ] `file://` detection state (AC5)
- [ ] Launch command lives in **one** JS constant, shared by both failure states and asserted equal to the README string (AC38)
- [ ] Client request state vocabulary is `sent → streaming → done | error | cancelled` and nothing richer (§5.6)
- [ ] `stale` fields render the last good value **with a visible age suffix**, never silently as current (§4.1)
- [ ] Static-then-continuous ordering enforced; comparison refuses to render on mismatched parameters (§6.3)

**dashboard dev:**

- [ ] Every panel exports `meta` + `mount()` returning `{unmount, describe}`
- [ ] No `fetch`, no `setInterval`, no DOM access outside `rootElement`, no `store.latest()`
- [ ] Every value rendered via `renderField()`; **zero raw numbers in template strings**
- [ ] Every series via `renderSeries()`; gaps honoured, **never bridged**
- [ ] `describe()` is a real sentence, not a field dump
- [ ] `@container` queries only, never `@media`
- [ ] rAF-coalesced canvas repaints; skip when hidden or off-screen (`IntersectionObserver`)
- [ ] `unmount()` unsubscribes, cancels rAF, disconnects observers
- [ ] Composite widgets are one tab stop with a roving cursor (§9.3)
- [ ] Batch occupancy renders the **count** as real and the **percentage** as unavailable when `max_batch` is missing (§5.3)

**both:**

- [ ] Grayscale screenshot test — every panel still parses
- [ ] Colourblind simulator, all three types
- [ ] Keyboard-only pass through all scenarios, no traps, ring always visible
- [ ] DevTools clean: zero uncaught exceptions, zero failed requests (AC21)
- [ ] No documented zero rendered as a measurement — **grep every panel for a numeric literal in output**

---

## 12. DESIGN DECISION LOG

| # | Decision | Rationale |
|---|---|---|
| D1 | Honesty enforced by the `Field` type, not by convention | A rule people must remember is violated at 2am; a type they cannot avoid is not |
| D2 | Em-dash + dashed underline + `cursor: help` for unavailable | Two centuries of typographic precedent, instantly distinct from `0`, carries no emotional charge |
| D3 | Real zeros render at full contrast, including `prefix hits: 0` | Em-dashing an unflattering real number lies in the flattering direction — the worse direction |
| D4 | Unavailable series = hatch + `NOT MEASURABLE YET`, never a flat zero line | A flat line implies duration: "we watched for 60 s and it was zero" |
| D5 | Gaps are never bridged | Interpolation fabricates the most convincing false data — data that looks continuous |
| D6 | Client swimlane has 2 segments, not 4 | A browser cannot separate queue wait from prefill; drawing the split would be invisible fabrication |
| D7 | A/B stacked with a shared locked X scale, not side-by-side | Makes the comparison about extent, which is the real result, not about shape |
| D8 | Static runs first, always | Conservative ordering — it disadvantages our own number, and careful readers notice |
| D9 | Sequence identity is colour + pattern + shape marker | AC25, and the same work makes the page survive grayscale, projectors, and blog screenshots |
| D10 | Block grid has three density modes + a mandatory ribbon | Rendering a 3px dog-ear is a spurious significant figure in visual form |
| D11 | Prefix **panel** unconditional, prefix **scenario** cuttable | Different reasons: demo quality vs honesty. Conflating them would be hiding a number |
| D12 | Hero fallback is `tokens per decode step` | Real, derived, and the best single scalar for batching efficiency |
| D13 | `continuous batch: enabled/disabled` row in the model card | A visitor on a non-static-cache model would otherwise conclude the feature doesn't work |
| D14 | Panels use `@container`, never `@media` | A panel is 340px or 700px depending on grid position; media queries cannot see that |
| D15 | Composite widgets are one tab stop with a roving cursor | A 4096-cell tab trap would be the worst a11y failure available to us |
| D16 | Dark-first, near-zero shadow, hairline-and-surface depth | The page should read as a technical drawing; hairlines survive a projector, shadows do not |
| D17 | Tabular figures on every updating number | Proportional digits jitter at 4 Hz, and jitter reads as instability |
| D18 | Honesty footer generated from the field registry | A hand-maintained honesty footer is a lie waiting for its second sprint |
| D19 | Adopted the demo dev's `measured`/`pending`/`stale`/`unavailable` vocabulary over my own three-state draft | `measured` names provenance rather than approval, and `stale` is a real state I missed. Two people converging independently on the provenance envelope is the strongest evidence the seam is right |
| D20 | A stale value carries a visible **age suffix**, not just a colour shift | Rendering a 12-second-old number as current is the same class of error as rendering a fabricated one, and colour alone fails grayscale (§9.1) |
| D21 | `estimated` must be a distinct `source`, not folded into `derived` | Arithmetic on real inputs and a model standing in for a measurement are different kinds of claim, and the difference must be visible without hovering |

---

## 13. ADDENDUM — CAPABILITY PROFILES (supersedes parts of §1, §5, §7, §8)

**Trigger:** @e00032a4 proved empirically, and @c0de4c2e escalated, that **continuous batching and paged KV are mutually exclusive**. I verified it independently in source before redesigning.

### 13.1 What I verified, and one thing nobody had found yet

`ContinuousBatchManager` (`crates/onnx-genai-engine/src/batched.rs:101-110`) holds `decode`, `tokenizer`, `metadata_max_context`, `static_max_len`, `queue`, `rows`, `events`, `next_handle`. **No `kv_cache`, no page table.** Its own doc comment says "for STATIC-CACHE models" and describes "a fixed number of **physical decode rows**". Confirmed.

**The stronger evidence, which changes more than the struct signature does:**

```rust
// batched.rs:262  (continuous path)  and  batched.rs:486  (static batch path)
let loop_state = DecodeLoopState::with_rng(0, rng, pending.options.top_logprobs);
//                                         ^
// DecodeLoopState::with_rng(prefix_cache_hit_len, rng, top_logprobs)
//   — decode_loop.rs:39-43. The first parameter IS prefix_cache_hit_len,
//     and it is a hardcoded literal 0 on BOTH batched paths.
```

So `prefix_cache_hit_len` is not merely unpopulated on the batching path — **it is a compile-time constant `0` flowing into a field that is later reported as a measurement** (`batched.rs:347`, `:579` → `finish_result` → `metrics.rs`).

**That reclassifies the number, and it reverses my own earlier ruling.** In §8.1 and §5.5 I argued that `prefix_cache_hits: 0` was a *real measurement* and must render as a stark `0 %`. **That was wrong, and I was wrong for the right reason** — I had assumed the counter was observing a cache that genuinely never hit. It is not observing anything. A hardcoded `0` reaching a metrics sink is precisely the same species as `/v1/status`'s `tokens_per_second: 0.0`: a constant wearing the costume of a measurement.

> **Corrected ruling:** on the **static/batching** path, prefix cache hit rate is **`unavailable`** — em-dash. On the **dynamic/paged** path it is a **genuine measurement** and renders as a number, including a real `0 %`.

**The same field has a different provenance state in each configuration.** That is the finding with the widest blast radius, and it is a concrete change request for `telemetry-provenance.js`:

> **Provenance must be keyed by `(field, capability profile)`, not by field name alone.** A single flat `PROVENANCE` table cannot express "measured here, structurally fabricated there", and today it would mark prefix hit rate identically on both paths — which would make us render a hardcoded zero as a measurement on exactly the server the demo runs on.

### 13.2 The information-architecture ruling: **detect a profile, never offer a mode**

The Secretary framed this as "two mutually-exclusive modes." I want to sharpen that, because the distinction changes the UI substantially:

**The visitor cannot choose. The loaded model decides.** So this is not a mode *selector* — there is no toggle to build, and building one would be a lie about what the visitor controls. It is a **capability profile the page detects and adapts to.** Detection, not selection.

| | **Profile S — static cache** (`qwen2.5-0.5b-scatter-v2`) | **Profile D — dynamic** (`qwen2.5-0.5b`) |
|---|---|---|
| Continuous batching | ✅ live, `max_batch=4` | ❌ per-request path |
| KV panel | ✅ **active decode rows** — a count; no denominator exists (D156) | ✅ **paged block table** — 14 612 pages |
| Prefix caching | ❌ hardcoded `0` → `unavailable` | ✅ genuine measurement |
| Scenarios live | A (batching) | B (paged KV), C (prefix) |

### 13.3 The reframe that makes this an asset, not damage control

A dashboard with half its panels em-dashed is *honest but weak*, and honest-but-weak is not the bar. But the mutual exclusivity is **the most interesting thing this crew learned all session**, and it is a real architectural fact about the runtime:

> **Continuous batching buys throughput by taking KV out of the pageable pool and putting it in fixed in-place rows. Paged KV buys sharing, prefix reuse, and eviction by keeping KV in a managed page table. This runtime implements both, and today they are alternatives rather than a composition.**

That is a genuinely instructive sentence about inference-engine design, it is true, and no competing demo says anything like it. **We should lead with it, not apologise for it.** A demo that names a real tradeoff in its own architecture is more credible than one that renders every panel green — and per §0, credibility is the product.

So: the profile is announced as a **statement of what is live**, never as an error or a degraded state.

### 13.4 UI changes

**(a) Profile banner** — a full-width strip directly under the app rail, `--og-bg-raised`, 40px. Informational, never `--og-warn`:

```
◈ static-cache profile · continuous batching LIVE · paged KV bypassed by design   ⓘ why?
```

`ⓘ why?` opens a popover with the §13.3 paragraph and the exact command to run the other profile. Naming *both* halves — what is live AND what is bypassed — in the same sentence is what stops a visitor from concluding something is broken.

**(b) Panels declare a requirement.** `meta.requires = 'continuous-batch' | 'paged-kv' | null`. The shell reads it and never mounts a panel whose capability is absent.

**(c) Inactive panels collapse into ONE group card — this supersedes §4.4 at this scale.** My original design gave every unavailable panel its own capability notice. With six panels dark that becomes a wall of six identical notices, which is exactly the "wall of zeros" failure in a more polite typeface. Instead, a **single** card occupying one grid cell:

```
┌─ Not active in this configuration ──────────── ⓘ ┐
│  ⬚  Paged KV block table · Prefix cache          │
│                                                  │
│  This server runs a static KV cache, so KV lives │
│  in fixed decode rows rather than a paged pool.  │
│  These three panels have no data source here —   │
│  not zero, absent.                               │
│                                                  │
│  ┌────────────────────────────────────── ⧉ ┐    │
│  │ --model models/qwen2.5-0.5b              │    │
│  └───────────────────────────────────────────┘   │
│  Everything else on this page is live.           │
└──────────────────────────────────────────────────┘
```

One dignified card > six dead panels. `auto-fill` means the grid reflows with no hole (§1.2), so this costs no layout work.

**(d) KV panel, Profile S — redefined as row occupancy.** Adopting the Architect's proposal: `active rows`. **This is real, measurable, and it moves under load**, which is what the demo needs.

> 🔴 **STRUCK, AND THIS PARAGRAPH IS THE EXHIBIT.** It previously read: *"with `--max-batch` approved and surfaced, the denominator becomes real and occupancy upgrades from em-dash to a genuine percentage. Net win."* **`--max-batch` DOES NOT EXIST. It was announced as delivered twice and was never built** (`cli.rs` has no such arg; `admin.rs:64` says so in a comment: *"max batch size not surfaced to the server"*).
> **I did not fabricate a measurement — I reasoned from an APPROVAL, and an approval is a record of intent that our tooling renders identically to a record of fact.** It is the one input class none of my rules covered: I had *verify the field*, *verify the instrument*, *cite the executable line* — **and none of them fire on the sentence "that's approved," because approval sounds like something that already happened.** D156.
> **Until `max_batch` is in the payload, `batch_utilization` renders as an ABSOLUTE COUNT or as `unavailable`. NEVER a percentage.** A client-side hardcoded `4` from `state.rs:25` would be **a constant wearing the costume of a measurement, on the panel carrying our headline 2.46× number, in the scenario that ships first and alone.**

Visually it is the block grid at small `capacity` — one cell per active decode row, in **detail** density (§7.2), each cell showing its owning sequence and fill. **The same component, a different noun.** Title: **"Static KV decode rows"**, never "block table", because they are not pages and calling them pages would re-import exactly the confusion this addendum exists to remove.

**(e) Hero strip is profile-aware.** `auto-fit` already guarantees no hole (§1.2 — the property is now earning its keep twice). Slot mapping:

| Slot | Profile S | Profile D |
|---|---|---|
| 1 | aggregate tok/s `ᴰ` | aggregate tok/s `ᴰ` |
| 2 | **active decode rows** (count) `ˢ` | **KV block utilization %** `ᴰ` |
| 3 | **tokens / decode step** `ᴰ` | **pages allocated / freed** `ˢ` |
| 4 | running / waiting `ˢ` | running / waiting `ˢ` |

> 🔴 **TWO SLOTS STRUCK ABOVE.** Slot 2/Profile S was **row occupancy %** — no denominator exists, so it is now a count. Slot 3/Profile D was **prefix hit rate `ˢ`** — **a CUT field, and the single most dangerous binding instruction this document ever contained**, because it sat in the *hero strip*, the four numbers a visitor reads first. **Replaced with paged-KV page allocation, which @e00032a4 verified directly (allocated 3, freed 3, 14612 pages).**

§8.2's hero-fallback provision (`tokens per decode step`) was designed for the prefix cut and now does exactly this job unmodified. The cut-cleanly work paid for itself sooner than expected.

> ### ⛔ (f) IS SUPERSEDED — DO NOT BUILD IT. See §26 and §45.
> **ALL THREE SCENARIO TABS ARE ALWAYS VISIBLE.** The switcher chooses the **origin**; detection then decides what **mounts** against whatever answers. **Filtering the tabs would hide the existence of half the product from every visitor** — I was wrong, @376a0297 adopted my wrong line, and @12e42da8 reversed it. This paragraph is kept rather than deleted so the reversal is legible, but **it is not the instruction.**

**~~(f) Scenario tabs are filtered by profile~~**, using the same content-driven registry from §8.2. ~~Profile S shows Batching + Free play; Profile D shows Paged KV + Prefix + Free play. A scenario is never shown disabled — an unclickable tab is an invitation to feel excluded. It is absent, and the profile banner already explains why.~~

### 13.5 The ambitious version — worth one paragraph, not worth blocking v1

If the operator runs **both** servers, the page can point at both and show the two profiles **side by side**: same prompts, same hardware, two architectures, with each column honestly reporting what is live. The comparison of *what each configuration can even measure* becomes the demo, and the tradeoff in §13.3 stops being a sentence and becomes a picture.

This is the most compelling version of this demo and I want it on the record as the v2 target. **It must not gate v1** — it needs a second origin, which means CORS, which AC39 deliberately eliminated. One flawless profile beats two shaky ones, exactly as the PM ruled about scenarios.

### 13.6 Revised decision log

| # | Decision | Rationale |
|---|---|---|
| D22 | **Reversed D3 for the batching path.** Prefix hit rate is `unavailable` on Profile S, `measured` on Profile D | `batched.rs:262`/`:486` pass a hardcoded `0` for `prefix_cache_hit_len`. A literal reaching a metrics sink is a fabricated zero, not a measured one |
| D23 | Provenance keyed by `(field, capability profile)`, not field alone | The same field is genuinely measured in one configuration and structurally fabricated in the other; a flat table cannot express that |
| D24 | Capability profile is **detected**, never selected | The loaded model decides. A mode toggle would misrepresent what the visitor controls |
| D25 | Inactive panels collapse to ONE group card, superseding per-panel notices at this scale | Six identical capability notices is the wall-of-zeros failure in a politer typeface |
| D26 | Profile announced as a statement of what is live, never as an error | The tradeoff is real, instructive, and unique to us — leading with it buys more credibility than hiding it |
| D27 | Profile S KV panel = "Static KV decode rows", reusing the grid component | Same component, different noun. Calling rows "pages" would re-import the exact confusion this addendum removes |
| D28 | Dual-server side-by-side is the v2 target, explicitly not gating v1 | It needs a second origin, hence CORS, which AC39 eliminated on purpose |

---

## 14. THE TWO-SERVER MODEL — DESIGNING THE SWITCH HONESTLY

@376a0297 has recommended the Lead adopt two servers, with **the scenario switcher doubling as the server switcher** — Scenario A on scatter, Scenarios B+C on dynamic. Structurally I support it: it dissolves the wall-of-zeros problem, since each scenario renders only what is live on its own server. My layout needs no change to accommodate it (§13.2 — `auto-fill` grid, registry-driven tabs).

But the switcher-doubling carries a **credibility risk that runs opposite to its purpose**, and it is squarely a design problem, so I want it on the record before the Lead rules.

### 14.1 The risk: a silent change of ground

If Scenario A shows batch occupancy 4/4, and one click later Scenario B shows a live paged-KV block table, the visitor's natural and *entirely reasonable* inference is: **"this one server does both."** It does not. It cannot — that is the whole finding of §13.1.

Nobody will have said anything false. The tabs imply continuity of subject; only the *content* changed, so the visitor supplies the continuity themselves. **We would be leaving a false impression using only true statements** — which is the most expensive kind of dishonesty for a project whose product is credibility, because it is the kind that survives review. Every individual panel would pass AC6 while the page as a whole misleads.

This is the same failure class as interpolating across a data gap (§4.3): nothing fabricated, everything *joined*, and the join is the lie.

### 14.2 The fix: make the server a **persistent, primary** object, not a tab side-effect

Three requirements. The first is essential; the other two are cheap and make it good.

**1 — Persistent server identity in the hero, not the header.** Under one server it is a detail; under two it becomes *primary orientation* and must be visible without scrolling, at all times, adjacent to the numbers it qualifies:

```
┌────────────────────────────────────────────────────────────┐
│  ● SERVER 1 · qwen2.5-0.5b-scatter-v2 · static KV           │
│    continuous batching ✓   paged KV ✗                       │
└────────────────────────────────────────────────────────────┘
```

The capability ticks are the honest part: they state up front what this server *cannot* do, so an em-dash later is a **confirmation rather than a surprise**. The visitor should never learn a limitation from a hole.

> 🔴 **`prefix cache ✗` REMOVED FROM THIS STRIP (D164), and the reason is not that the feature was cut — it is that WE CANNOT HONESTLY RENDER EITHER TICK.** A ✓/✗ is a **binary capability assertion**, the most confident shape on the page. Our own measurements disagree about which one is true: **scatter records nothing (0/135), dynamic records everything (19/20, incrementing on six control requests that share nothing).** A `✗` claims *we checked and it is off*; a `✓` claims *we checked and it works*. **We checked, and what we found was an instrument that reports opposite falsehoods on the two servers.** There is no tick for that, and inventing one would be the highest-confidence widget on the page carrying our least-trustworthy fact.
>
> **The strip now lists only capabilities we can demonstrate on screen. A capability list is a PROMISE OF WHAT FOLLOWS — every entry must be answerable by something further down the page**, or the strip becomes a menu with items the kitchen does not serve.

**2 — Narrate the transition; never let it be silent.** On a switch that changes server, the new scenario opens with one line, dismissible, not a modal:

> *Scenario B runs on a **second server** — a dynamic-cache model. Continuous batching is off here; paged KV and prefix caching are on. The two cannot run in one process, and that tradeoff is the point.*

Turning the seam into the lesson is strictly better than concealing it. An expert who notices a hidden seam concludes we were hiding it; one who is *told* concludes we understand our own system. **Same fact, opposite inference, decided entirely by who says it first.**

**3 — Distinct server colour, carrying a non-colour marker.** Server 1 `--og-seq-0`, Server 2 `--og-seq-1`, applied to the hero rule and the active tab — with the server *name* always present, since colour alone is never meaning (§9.1, AC25).

### 14.2.1 Staleness is **per-server**, and the server chip owns it

Two servers means **two independent connections**, so one can go quiet while the other stays live. This is a real gap in my §4.1 treatment, raised by @c0de4c2e: field-level `stale` handles *a number*, but nothing yet answers the visitor's actual question — **"which half of this page is still alive?"**

Answering that per-field would be terrible: a visitor should not have to infer a dead connection by noticing that eleven separate age suffixes all say `12s`. **A connection is one fact about one server, so it is displayed once, on the server chip:**

```
● SERVER 1 · qwen2.5-0.5b-scatter-v2 · static KV        live
◐ SERVER 2 · qwen2.5-0.5b · dynamic KV      last update 12s ago
```

- **`live`** — polls returning. Filled dot, `--og-seq-*`.
- **`last update Ns ago`** — polls failing. Half-filled dot, `--og-stale-fg`, and **the whole region carrying that server's panels dims by one step**. Dimming is applied at region scope, not per field, so "this half went quiet" is a single visual event rather than eleven simultaneous ones.

Dot fill state and the literal word/age are both non-colour channels, so this survives grayscale and `prefers-reduced-motion` (§9.1, §9.5) — no pulsing "reconnecting" animation, which would be motion conveying meaning.

Field-level `stale` still applies underneath, unchanged: it is what makes a *single* metric's frozen-ness legible when you look straight at it. **The chip answers the scanning question; the age suffix answers the staring question.** Both are needed, and neither substitutes for the other.

**The chip carries the reconnect affordance** (@c0de4c2e), and it is the only place one appears. Once a server has been stale for more than ~10s the chip gains an inline `Retry` control, using AC37's actionable-failure copy. It belongs here rather than on a panel for the same reason the staleness does: **reconnection is an act on a connection, not on a number.** Eleven panels each offering to reconnect the same socket is eleven buttons doing one thing — and it invites the visitor to believe the panels failed independently, which would misdescribe the fault.

Age granularity matters and is specified: `stale` at 800ms and `stale` at five minutes are different claims. Below 2s, no chip change — that is a poll jitter, not an outage, and flagging it trains the visitor to ignore the flag. From 2s, show the age. Past 60s, switch to `last update 3m ago`, since second-precision on a long-dead connection is false precision.

The interaction with §14.1 matters: a stale server must remain **visibly identified as a server**, not silently blank. A blanked region invites the visitor to forget it exists and read the live half as the whole system — the same false-continuity error, arriving by a different route.

### 14.3 Server-aware hero — accepted, with the ordering made explicit

Accepted: `tokens per decode step` takes the hero slot when prefix isn't applicable, rather than replacing it permanently. The PM's observation that it *rises as batching engages* is the deciding one — a hero metric must move when the thing it demonstrates is working, or it teaches nothing.

Ordering rule, so it is deterministic rather than per-panel improvisation: **the hero slot is filled by the highest-priority metric whose classification on the active profile is `MEASURED`.** Profile D → prefix hit rate. Profile S → tokens per decode step. This is the registry-driven fallback of §8.2, unchanged and reused — it was built for the prefix cut and does this without modification.

| # | Decision | Rationale |
|---|---|---|
| D29 | Three kinds of zero; `bypassed` is a `classification`, **not** a fifth `state` | `state` selects the render primitive, `classification` selects the wording. Bypassed and placeholder share a primitive. Additive change, zero blast radius |
| D30 | Placeholder and bypassed are **not** distinguishable at a glance — same em-dash | One glance-level fact: "no number here, nothing hidden." Two glyphs read as inconsistency, and honesty that looks like malfunction fails at its only job |
| D31 | A bypassed **subsystem** collapses to one group card; a bypassed **field** stays as an em-dash | Six em-dashes say one thing six times. One card says it once, better |
| D32 | If servers switch, server identity is persistent and primary, and the switch is narrated | Tabs imply continuity of subject. A silent server change leaves a false impression using only true statements — the costliest dishonesty available to us |
| D33 | Hero slot = highest-priority metric with `MEASURED` classification on the active profile | Deterministic, registry-driven, already built for the prefix cut |

| # | Decision | Rationale |
|---|---|---|
| D34 | Connection staleness is displayed **once per server**, on the server chip, with region-scope dimming | A connection is one fact about one server. Eleven simultaneous age suffixes is the wall-of-zeros failure wearing a clock |
| D35 | A stale server region dims but is **never blanked** | A blank region invites the visitor to forget it exists and read the live half as the whole system — §14.1's false continuity by another route |

---

## 15. THE FINAL FIELD SHAPE — CONSOLIDATED, AND CLOSED

This section supersedes §3.2 and §4.1 on the Field object. It exists because the shape has taken three increments in twenty minutes, and increments are themselves a cost: every one risks a dev implementing version *n-1*. **This is the complete shape. No further additions.**

### 15.1 The defect: `source` is carrying three jobs

@c0de4c2e caught this, and it is caught by a rule I published myself. `telemetry-field.js:81` documents `source` as *"e.g. `'/v1/status'` or `'client'`"* — two different questions in one string. A panel distinguishing them must test `source.startsWith('/')`, which is **branching on a substring**, the precise antipattern I banned for `reason` in §4.7.

It is in fact carrying three, because `unavailableField` (`:118`) defaults `source` to the literal `'unavailable'` — **a state name in an attribution slot**, which is neither an endpoint nor a provenance class.

Three genuinely different questions a visitor asks about one number:

| Question | Key | Why it can't be merged |
|---|---|---|
| **How** do we know it? | `provenance` | AC7's hover badge branches on it (`ˢ ᶜ ᴰ ᴱ`, §4.5) |
| **Where** did it come from? | `endpoint` | Makes a claim auditable — a reviewer greps for it |
| **Which engine** produced it? | `origin` | Option D's honesty requirement: stops a batching number being read as a paged-KV number |

The Lead's Option D ruling — *"make it impossible to misread which server a number came from"* — cannot be satisfied by a key that is also answering AC7, because **reviewers check those two acceptance criteria separately and one string can only pass one of them.**

### 15.2 The shape

Ten keys is a lot, so they are **grouped by the question they answer**. Grouping is the design work — ten flat keys are unreadable, four groups of two or three are not.

```js
{
  // THE VALUE — what to show
  value,          // null unless state is 'measured' or 'stale'
  state,          // 'measured' | 'pending' | 'stale' | 'unavailable'

  // ATTRIBUTION — where it came from. Three questions, three keys.
  provenance,     // 'server' | 'client' | 'derived' | 'estimated'
  endpoint,       // '/v1/status' | '/metrics' | null   (null when not server)
  origin,         // 'scatter' | 'dynamic' | null       (which engine)

  // EXPLANATION — why it isn't a plain number
  classification, // 'MEASURED' | 'DOCUMENTED_ZERO' | 'NOT_PLUMBED' | 'BYPASSED'
  reason,         // required prose whenever state !== 'measured'

  // MEASUREMENT CONTEXT
  unit,           // 'ms' | 'tok/s' | '%' | null
  observedAtMs,   // for 'stale', the ORIGINAL observation time
  derivedFrom,    // field keys, when provenance === 'derived'
}
```

**Migration from today's code is mechanical:** `source: '/v1/status'` → `provenance: 'server', endpoint: '/v1/status'`; `source: 'client'` → `provenance: 'client'`; `source: 'unavailable'` → delete, the `state` already says that.

### 15.3 `origin` is orthogonal to `provenance` — the point most likely to be got wrong

A client-measured TTFT still has an `origin`. The browser measured it, but **it measured it against a particular engine**, and that attribution is exactly what Option D requires. So:

```js
{ value: 41, state: 'measured', provenance: 'client',
  endpoint: null, origin: 'scatter', unit: 'ms' }
```

`provenance: 'client'` and `origin: 'scatter'` are both true and neither implies the other. Any implementation that sets `origin` only for server-sourced fields will leave every client-measured number unattributed — **and the client-measured numbers are the hero metrics.**

### 15.4 Why `origin` is an explicit key and not a base URL

@c0de4c2e's forward-compatibility point, which I'm ratifying as a decision. @d7cf9b84 is testing whether one process with `--models-dir` can serve both models. If it can, we collapse to one origin *host* — and any attribution derived from a base URL silently becomes "everything came from the same place", **losing the distinction the demo exists to teach, at exactly the moment it becomes hardest to notice.**

An explicit `origin` survives both outcomes unchanged: two processes or one, the field still says which engine produced it. Modelling the *fact* rather than the *transport* is the difference between a shape that survives an infra change and one that quietly starts lying after it.

This also means §14's mode-switch degrades gracefully: if both modes land in one process, the server chip becomes an **engine chip**, the narration drops the word "second server" and keeps the architectural explanation — which was always the part that mattered.

### 15.5 Layering note

`provenance`, `endpoint`, `origin`, `unit`, `classification` are **constant for a given `(field, profile)`** — they belong to the registry, not to each sample. Only `value`, `state`, `observedAtMs` vary per poll.

The store may therefore populate them by lookup rather than storing ten keys per sample per tick. **Panels still receive one flat object** — that is contract, not implementation. This is an efficiency freedom for the store author, not a change to what a panel sees.

| # | Decision | Rationale |
|---|---|---|
| D36 | `source` splits into `provenance` / `endpoint` / `origin` | It was answering three questions with one string; telling them apart required substring-branching, which §4.7 forbids |
| D37 | `origin` is set on **client**-measured fields too | The browser measures *against* an engine. Otherwise the hero metrics are the unattributed ones |
| D38 | `origin` is an explicit key, never derived from a base URL | If one process serves both models, URL-derived attribution silently collapses to "same place" — losing the demo's central distinction exactly when it's hardest to spot |
| D39 | Registry-constant keys may be populated by lookup; panels still see one flat object | Layering is the store's freedom; the panel contract stays flat |
| D40 | **The Field shape is closed.** Further needs go in the registry, not the envelope | Three increments in twenty minutes is its own risk — a dev implementing version n-1 is now the likeliest failure |

---

## 16. LABEL HONESTY — THE GAP IN MY OWN DESIGN

@d7cf9b84's provenance audit exposes a class of dishonesty the Field envelope **does not catch**, and I should have seen it. Everything in §4, §15 guards the *value*: is it measured, stale, bypassed, fabricated. Nothing guards the **name**.

> **A correct number under a wrong label is a fabrication — and it is worse than an em-dash, because it passes every provenance check we have.**

`state: 'measured'`, `provenance: 'server'`, real endpoint, honest sampling — and the panel still lies, because the *caption* claims it counts something it doesn't. The whole envelope waves it through. A visitor cannot audit a number they've been told the wrong name for; they don't even know to be suspicious.

So the label is not copywriting. **It is part of the provenance claim**, it belongs in the registry beside `unit` and `endpoint`, and it is verified against what the code actually counts.

### 16.1 The three confirmed mislabels

| Wire field | Naive label | What it **actually** counts |
|---|---|---|
| `active_sessions` | "Active sessions" | Persistent `X-Session-Id` sessions — **reads 0 during Scenario A** |
| `prefix_cache_lookups` | "Prefix cache lookups" | Completed generations (`metrics.rs:134`, unconditional `fetch_add(1)`) |
| `vram.used` | "GPU memory used" | The scheduler's **KV byte budget** (`governor.rs:548`) — not a device query |
| `host_ram.used` | "Host RAM used" | **Whole-machine** OS reading, including every other process |

The `active_sessions` one is the most dangerous, because it fails **loudest at the demo's peak moment**: four lanes visibly streaming next to "Active sessions: 0". The visitor doesn't conclude the label is wrong — they conclude **the dashboard is lying**, and once they think that, every honest number on the page is worthless. We would lose the entire credibility thesis to a caption.

**Ruling: the concurrency number in every panel and scenario is `active_batch_size`, never `active_sessions`.** `active_sessions` may appear only in the Requests panel, labelled **"Persistent sessions (`X-Session-Id`)"** — its real name, where its zero is unremarkable.

### 16.2 I verified `prefix_cache_lookups` myself, and it is both worse and better than reported

`metrics.rs:126-138`:

```rust
pub(crate) fn result(&mut self, completion_tokens: usize, prefix_cache_hit_len: usize) {
    REGISTRY.prefix_cache_lookups.fetch_add(1, Ordering::Relaxed);   // every generation
    if prefix_cache_hit_len > 0 {
        REGISTRY.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);  // per REQUEST, not per token
    }
}
```

**Worse:** it is not a lookup count, so "hit rate" as a *cache* hit rate is meaningless — we would be dividing by the number of finished requests.

**Better, and this is the part worth catching:** *both* counters are incremented **once per request**, so the ratio is still **well-formed** — it is just not the thing its name claims. `hits / lookups` = **the share of requests that reused any prefix at all.** That is a genuinely useful, genuinely honest metric. It doesn't need deleting; **it needs its real name.**

**Ruling: the Profile D hero (D33) is relabelled** — *"Requests reusing a prefix"*, rendered as a percentage of completed requests, hover: *"Share of completed requests that matched a cached prefix. Not a per-lookup cache hit rate — the runtime counts one lookup per generation."*

Same number, same arithmetic, honest caption. **This is what the label-as-provenance rule buys: it salvages a metric instead of discarding it.** Discarding it would have been the timid response and would have cost us the hero.

**What is genuinely lost:** `prefix_cache_hit_len` is tested `> 0` and then thrown away, so a request reusing 2 tokens counts identically to one reusing 500. We can say *how often* reuse happened, never *how much*. Since "tokens of prefill skipped" is the metric that actually demonstrates the feature's value, this is worth one `fetch_add` — see §16.4.

### 16.3 `/v1/resources` — healthy data, dangerous captions

Ten real fields, and the two most eye-catching are the two most mislabelled.

- `vram.used` → **"KV cache budget in use"**, never "GPU memory used". Hover names `governor.rs:548` and states it is the scheduler's own accounting, not a device query. Rendering it as GPU memory is the single most checkable false claim on the page — any visitor with `nvidia-smi` open disproves it in ten seconds, in front of an audience.
- `host_ram.used` → **"Host RAM (whole machine)"**, with the caveat *in the label, not the hover*, because the misreading ("look how much our runtime uses") is the visitor's default and a hover cannot pre-empt a conclusion already drawn.

**General rule: when a caveat changes what the number means, it goes in the label. Hovers explain; they cannot retract.**

### 16.4 One ask back to @d7cf9b84

Sum `prefix_cache_hit_len` into a `prefix_cache_hit_tokens` counter rather than only testing `> 0`. It is one `fetch_add` next to an existing one, and it converts *"38% of requests reused a prefix"* into *"we skipped 12,400 tokens of prefill"* — the difference between a statistic and a demonstration. That is the number the demo exists to produce, and it is currently discarded one line before it would be recorded.

| # | Decision | Rationale |
|---|---|---|
| D41 | Labels are **provenance**, live in the registry, and are verified against what the code counts | The envelope guards values, not names. A correct number under a wrong caption passes every check we have |
| D42 | Concurrency is always `active_batch_size`; `active_sessions` appears only as "Persistent sessions" | "Active sessions: 0" beside four streaming lanes destroys the credibility thesis at the demo's peak moment |
| D43 | Prefix hero relabelled "Requests reusing a prefix", not "hit rate" | Both counters are per-request, so the ratio is well-formed — it was only ever the *name* that was false |
| D44 | `vram.used` = "KV cache budget in use"; never "GPU memory used" | It's `governor.rs` accounting, not a device query — and it's disprovable with `nvidia-smi` in ten seconds, live |
| D45 | A caveat that changes a number's **meaning** goes in the label, not the hover | Hovers explain; they cannot retract a conclusion the visitor has already drawn |

---

## 17. CANONICAL FIELD SHAPE — SUPERSEDES §3.2, §4.1, §15

**This section is the only current statement of the Field shape.** §3.2, §4.1 and §15 are historical; where they differ, this wins. The shape has moved five times in an hour, so there is now exactly one place to look.

```js
{ value,
  state: 'measured' | 'pending' | 'stale' | 'unavailable' | 'not-applicable',

  source,          // 'server'|'client'|'derived'|'estimated'|'simulated'  — badge class, AC7
  endpoint,        // '/v1/status' | null      — curl-able, makes the audit auditable
  server,          // 'scatter' | 'dynamic'    — WHICH ENGINE, AC41/AC42

  classification,  // sub-reason when state === 'unavailable': DOCUMENTED_ZERO | NOT_PLUMBED
  unit, label, reason, observedAtMs, derivedFrom }
```

### 17.1 Why `origin` was retired

@12e42da8's union used `origin` for the **endpoint path**; my §15 used `origin` for **which server**. Both were ratified, twenty minutes apart. A word that has meant two different things in two authoritative messages cannot be made safe by being careful — someone reads the older one. **Retired, not redefined.**

The collision was not cosmetic. With `origin: '/v1/status'`, that value is **byte-identical on both servers**, so the shape had *no key at all* for "which engine produced this" — and AC41 requires that a panel screenshotted **in isolation** still names its server. AC42 (a mode switch must never show one mode's numbers under the other's heading) likewise had nothing to branch on. The demo's central honesty requirement had no representation in its central honesty type.

**`server` is set on client-measured fields too.** `{ source:'client', endpoint:null, server:'scatter' }` — the browser measured TTFT, but it measured it *against an engine*. Setting `server` only where `source==='server'` leaves precisely the hero metrics unattributed.

### 17.2 The five treatments

@12e42da8's test was: *if any two would render identically, collapse them.* None do.

| state | glyph | treatment | voice |
|---|---|---|---|
| `measured` | `41` | full contrast. **A genuine `0` renders as a stark `0`** | none needed |
| `pending` | `···` | `--og-pending-fg`. **Resolves on its own** | "waiting" |
| `stale` | `41 · 12s old` | last value + **age in words**, `--og-stale-fg` | "this may no longer be true" |
| `unavailable` | `—` | dashed underline, `cursor:help`, reason in hover | **apologetic** — our gap; names the file |
| `not-applicable` | `—` **+ on-screen caption** | `--og-unavail-label`, reads as a statement | **not apologetic** — architectural fact |

### 17.3 I was overruled on `not-applicable`, and the ruling changed my design

I had it as a `classification` because it shares the em-dash primitive with `unavailable` (D30). @12e42da8 overruled it, and the reasoning is better than my own: **`not-applicable` is a teaching surface, not an error state.**

That reframing is what changes my mind, not the authority. If a distinction is semantically real *and one side of it is the most interesting thing the demo has to say*, then burying it in a secondary field means a panel author must go looking to find it — and the default, for anyone who doesn't, is to render our best insight apologetically. **A type should make the right rendering the easy one.** That is the same argument I used for the envelope itself; I failed to apply it one level down.

**So `not-applicable` gets the only on-screen explanation of the five.** Every other non-measured state explains itself in a hover; this one does not, because **a fact nobody hovers over is a fact nobody learns**, and this is the fact we most want read. That is what "first-class, not a footnote" means at field scale.

It also ends with a way to *see* the real number — the only state that can, since it is the only one where the number exists somewhere:

> *Not applicable on this server — continuous batching bypasses the paged KV allocator entirely. Switch to Scenario B to see it live on the dynamic server.*

`classification` survives, narrowed: it is now the **sub-reason for `unavailable` only** — `DOCUMENTED_ZERO` (server writes a constant) vs `NOT_PLUMBED` (data exists, no endpoint). Both apologetic, both ours to fix, distinguished only in hover wording.

| # | Decision | Rationale |
|---|---|---|
| D46 | `origin` retired; attribution is `source` / `endpoint` / `server` | The word was ratified twice with two meanings. As ratified, no key answered AC41 — the demo's central honesty requirement was unrepresentable |
| D47 | `server` is set on client-measured fields too | The browser measures *against* an engine. Otherwise the hero metrics are the unattributed ones |
| D48 | Keep `measured` over `ok`; keep `observedAtMs` over `at` | `ok` names approval, `measured` names provenance — and `measured` is already committed with passing tests. `at` doesn't state its unit, on a project about honest units |
| D49 | **D30 reversed.** `not-applicable` is a state, not a classification | A type should make the right rendering the easy one. Buried in a secondary field, our best insight renders apologetically by default |
| D50 | `not-applicable` is the only state explained **on screen** rather than in a hover | A fact nobody hovers over is a fact nobody learns — and this is the one we most want read |

---

## 18. BLOCK GRID — DEGRADATION, IDENTITY, AND CADENCE

@d7cf9b84 reports `PageUsage` discards per-sequence page IDs (`page_table.rs:867-875` keeps only `pages.len()`), so per-cell ownership may not be available. Three rulings so the panel is safe either way.

### 18.1 Ownership is an OVERLAY, never the base layer

The grid is built in two layers, and the cuttable one is on top:

- **BASE (always available):** each cell's occupancy — free / in-use / shared — encoded by **fill pattern and glyph**, not colour alone (§7.1, AC25). This needs only counts, which we already have.
- **OVERLAY (cuttable):** per-sequence tint + cross-highlight on hover.

Remove the overlay and the grid is still complete — a full, legible occupancy map — rather than a grid with holes in it. **Nothing reflows, nothing is empty, no apology appears.** This is the §8.2 provision applied unchanged; it was built for the prefix cut and needs no modification, which is the whole return on having built it that way.

If ownership is cut, the legend simply loses its sequence swatches and gains one line: *"Per-block ownership isn't exposed by the engine yet — this shows occupancy only."* One sentence, no hole.

### 18.2 No stable block identity means NO ANIMATION — this is honesty, not polish

@d7cf9b84 also notes there is **no stable per-block identity between frames**. That has a consequence they didn't draw out, and it is the more important of the two:

> **We must not animate cell transitions, because we cannot know that a cell in frame N is the same block as the cell in that position in frame N+1.**

Animating a cell from free to in-use asserts *this specific block was just allocated*. Without stable IDs that assertion is unfounded — the arrays may simply have been ordered differently. It is **the same error as bridging a gap in a chart**: fabricating continuity between two honest samples, which produces the most convincing false data there is, because continuity is what the eye reads as causation.

So: each frame renders as an **independent snapshot**. No enter/exit transitions, no morphing, no cell-level `transition:` property. This also satisfies `prefers-reduced-motion` (AC27) for free, and at 1 Hz a redraw reads as a clean tick rather than a stutter.

Aggregate motion is still honest and still available: the **occupancy sparkline** moves continuously, because *"73% used"* is a real quantity with real continuity between samples. **The number may be animated; the cells may not.**

### 18.3 Cadence and size — the answers @d7cf9b84 asked for

**Grid: 1 Hz. Gauges stay at 4 Hz.** Separate endpoint, as proposed.

The grid is read as a **texture**, not as a value: the eye takes in density, clustering and the shape of the free region. Texture perception does not improve above ~1 Hz, and nobody tracks an individual cell among thousands. Meanwhile a gauge is read as a *number*, and numbers need 4 Hz to feel live. **Different reading modes, different cadences** — this is not a compromise for bandwidth, it is what each panel actually needs. Bandwidth relief is a side effect.

**Correcting my own figure: I said 4096 blocks. The real number is ~14,612** (`qwen2.5-0.5b` dynamic, 384 KiB/page, per @e00032a4). That changes the design.

**Above ~2048 cells, one-cell-per-block is below the useful perceptual threshold** — at a 1280px-wide panel, 14,612 cells is under 2px each, so the visitor cannot resolve a cell and *cannot* see individual allocation anyway. Rendering them would be precision theatre: enormous payload, zero additional information.

So the grid **bins** above 2048 blocks: one cell = a contiguous run of N blocks, shaded by fill fraction, with the bin size stated in the legend — *"each cell = 16 blocks"*. Stating it is what keeps it honest; an unlabelled bin is a silent lie about resolution.

**Concretely, for the wire format:** bins of 16 turn 14,612 blocks into **913 values at 1 Hz — a few KB/s**, versus ~1 MB/s for the raw grid at 4 Hz. That is a ~99% reduction with **no loss of anything a visitor can perceive**, and it should clear the AC33 overhead budget comfortably. Send bins pre-aggregated or send flat arrays and let the client bin — my preference is flat arrays if cheap (client keeps the choice of bin size), pre-binned if not; the design works either way.

| # | Decision | Rationale |
|---|---|---|
| D51 | §7.4's eviction/preemption storyboard **deleted as fabricated** | `reconfigure` never touches `used`; the repo's own test says so by name; eviction results have zero consumers |
| D52 | Scenario C's payoff moves from the grid to **admission** | The real effect is refusal propagating to the front door. The grid's *stillness* is the finding |
| D53 | Scenario C is two ordered steps, **not a slider** | A slider affords dragging; dragging alone produces nothing. A control whose obvious use does nothing reads as broken — on the panel about failure |
| D54 | Step 1 **promises** that nothing will happen | A dashboard that predicts its own inaction and is right earns more trust than one that only shows wins |
| D55 | The `SIM`-badged pressure fallback is **deleted, not downgraded** | It existed only to guarantee a dramatic frame. Real pressure is reachable |
| D56 | Per-sequence ownership is an overlay; occupancy is the base layer | Cutting the overlay leaves a complete grid, not a holed one |
| D57 | **No cell animation** — each frame is an independent snapshot | Without stable block identity, an animated transition asserts a specific block changed. Same error as bridging a chart gap |
| D58 | Grid at 1 Hz, binned above 2048 cells, bin size stated in the legend | Texture is not read faster than 1 Hz, and sub-2px cells convey nothing. An unlabelled bin is a silent lie about resolution |

---

## 19. PROCESS-GLOBAL COUNTERS — AND WHY THE PREFIX METRIC BECOMES A WINDOWED DELTA

@d7cf9b84 reports that under a single multi-model server, per-model attribution works for KV/page usage and batch occupancy (separate engine threads), but **`prefix_cache_*` is a process-global counter with no model dimension, contaminated by other models' traffic.**

### 19.1 The contamination is worse than "shared", because of §16.2

Recall `metrics.rs:134`: `prefix_cache_lookups.fetch_add(1)` fires **unconditionally on every completed generation** — including generations on the scatter model, where prefix caching is structurally bypassed and the numerator **can never increment**.

So under one server, the metric I relabelled *"Requests reusing a prefix"* (D43) acquires a denominator inflated by traffic that is **structurally incapable of contributing to the numerator**:

> **Running Scenario A would silently lower the prefix-reuse rate shown in Scenario B.** Same true counters, same correct arithmetic, and a number that drifts downward for a reason no visitor can see or guess.

That is the invisible-dishonesty class again: not a fabricated value, but a **real value whose meaning is quietly corrupted by unrelated activity**. It is worse than a stale number, because staleness at least has an age we can display. This has no observable symptom at all — and it degrades precisely as the visitor explores more of the demo, so the more they engage, the more wrong it gets.

### 19.2 The fix: measure the window, not the lifetime

**The prefix metric is a delta between two samples taken at scenario start and now — never the cumulative process total.**

```
hits_now    - hits_at_scenario_start
─────────────────────────────────────   =  "requests reusing a prefix, this run"
lookups_now - lookups_at_scenario_start
```

This is honest under **both** topologies, so it removes my design's dependence on the one-vs-two-server question entirely.

It is clean because the information architecture already guarantees the necessary condition: **exactly one scenario is on stage at a time** (§1.2, and @12e42da8's *"one scenario is on stage at a time — our information architecture already solved this"*). If only the active scenario generates traffic, the window contains only that scenario's traffic, whichever process it ran in.

**And it is a better metric regardless of topology.** A cumulative lifetime rate includes server warmup, other agents' `curl` probes, and every earlier scenario — so the headline number a visitor reads would be dominated by history they never saw. *"This run"* is what they are actually watching, and it **responds to the button they just pressed**, which a lifetime average largely does not. A hero metric must move when the thing it demonstrates works.

**Required guard:** if `lookups_now - lookups_at_start === 0`, the state is **`pending`** (`···`), never `0%`. Zero lookups in the window is *no data yet*, not a 0% hit rate — and `/v1/status` already emits a literal `0.0` for `prefix_cache_hit_rate` when `lookups == 0` (`admin.rs:126-130`), so on the wire "no data" and "0%" are byte-identical. The client must special-case it; nothing else can.

**Reset on scenario entry, and on nothing else.** Not on mode switch, not on poll failure — a failed poll must not silently restart the window and make a bad rate look fresh.

### 19.3 Model identity per payload — adopted, no design change needed

@d7cf9b84 warns the telemetry endpoints default to the **alphabetically-first model**, which would silently show dynamic-model KV during a scatter-model batching scenario. Agreed, and this is exactly the failure D46/D47 anticipated.

**No change is required**, because §17 already carries `server` as an **explicit key, never derived from the base URL** (D38) — the reasoning being that if one process ever served both models, URL-derived attribution would silently collapse to "same place." That case has now arrived. The key simply carries a model id instead of a server id; it was always meant to be opaque.

Renaming it `model` would be more accurate under one server and less accurate under two. It stays `server` — one word, one meaning, already published to three consumers. **Churn on a correct key to improve a name is not worth a fourth revision.**

| # | Decision | Rationale |
|---|---|---|
| ~~D59~~ | 🔴 **WITHDRAWN by @12e42da8 — see §65.** ~~The prefix metric is a windowed delta from scenario start, never a lifetime total~~ | **The reasoning was sound and the SUBJECT does not exist.** A window makes a corrupted denominator honest; it cannot make a numerator mean anything. `runtime.rs:1017-1024` computes `cross_session_hit_len` and never sets `loaded_prompt_prefix`, so **no prefill is skipped and the counter responds to nothing.** A windowed delta of a quantity that does not move with the feature is **a hero metric that animates while nothing works.** |
| D60 | Zero lookups in the window renders **`pending`**, never `0%` | The server sends a literal `0.0` when `lookups == 0`; on the wire "no data" and "0%" are identical. Only the client can separate them |
| D61 | The window resets on **scenario entry only** | A reset on poll failure would make a bad rate look freshly measured |
| D62 | `server` keeps its name and carries a model id under one-server | The key was designed opaque precisely for this. A fourth revision to improve a name is churn |

---

## 20. STYLESHEET PATHS, AND THE `not-applicable` COPY

### 20.1 Ruling: one directory, `styles/`

Disk currently has `css/shell.css` alongside `styles/tokens.css` and `styles/panels.css`, and `index.html` links both. It works, and it is still wrong: two sibling CSS trees invite a second `tokens.css`, and the moment that exists the design system has forked with no error to announce it.

**`styles/` wins** — two of the three files are already there, `tokens.css` is committed and referenced from both `index.html` and `design/skeleton.html`, and `styles/` is what this document and @c8d9a40e's brief already say. Moving one file beats moving two and re-briefing a dev.

```
styles/tokens.css    designer  (exclusive — the CSS contract)
styles/shell.css     @bb2ee824 (was css/shell.css)
styles/panels.css    @c8d9a40e
```

### 20.2 `not-applicable` copy — the state a visitor will see MOST

@c0de4c2e makes the point that decides the priority here: **Scenario A ships first and alone, on the scatter origin, where every paged-KV and prefix-cache field is `not-applicable`.** So this is not an edge state. For a while it is the *most-seen* non-measured state on the page, and the first impression of our honesty design.

Which means the treatment has one job: **read as an assertion, not an absence.** @12e42da8's framing is the standard — *"we are not apologizing for this — we are teaching it."* If it renders as a slightly-different em-dash beside `unavailable`, we have apologised for the most interesting thing we found.

**Hover and caption copy — exact strings, because "not available" wastes the moment:**

```
PREFIX CACHE, on the scatter (static-cache) engine
  caption : Not applicable on this engine
  detail  : Continuous batching decodes into fixed in-place KV rows and never
            consults the prefix trie — ContinuousBatchManager (batched.rs:101-110)
            holds no reference to engine.prefix_cache, so a lookup is not
            merely absent, it is impossible. The engine's own tests assert this
            (batched_static_decode.rs:53). Switch to Scenario B to see prefix
            reuse measured live on the dynamic engine.

PAGED KV BLOCK TABLE, on the scatter engine
  caption : Not applicable on this engine
  detail  : Static-cache models use runtime-owned in-place KV buffers and bypass
            the page table entirely, so there are no pages to report. This is a
            property of the execution path, not a missing feature. Scenario B
            runs the dynamic engine, where the block table is live.

BATCH OCCUPANCY, on the dynamic engine
  detail  : Continuous batching does not engage on dynamic-cache models; this
            engine decodes one request at a time. Scenario A shows batching
            live on the scatter engine.
```

Three properties every one of these has, and they are the specification:

1. **It names a file and a line.** A hover citing `batched.rs:101-110` is a categorically different artifact from one saying *"not available"* — it is **checkable**, and an expert who checks one and finds it accurate stops auditing the rest. Verifiability is cheaper than persuasion and works better.
2. **It says why the thing is impossible, not merely absent.** *"Holds no reference to `engine.prefix_cache`, so a lookup is not merely absent, it is impossible"* is a structural claim a reader can evaluate. *"Not yet implemented"* invites the reader to assume incompetence.
3. **It ends with where the real number lives.** Every one points at the scenario that shows it measured. This is what converts a hole into a signpost — and it is only available to `not-applicable`, because it is the only state where the number genuinely exists somewhere else.

**Never** in this copy: *"not yet"*, *"coming soon"*, *"unfortunately"*, *"currently unavailable"*. Each implies a defect on a path that is behaving exactly as designed. `unavailable` is where apology is appropriate — that one *is* our gap.

| # | Decision | Rationale |
|---|---|---|
| D63 | All stylesheets live in `styles/`; `css/shell.css` moves | Two sibling CSS trees invite a second `tokens.css`, and a forked design system announces itself with no error |
| D64 | `not-applicable` copy cites file:line, states impossibility, and points to where the number is real | It's the most-seen non-measured state while Scenario A ships alone. A checkable hover ends an expert's audit; "not available" invites it |

---

## 21. THE FIVE TREATMENTS, AGAINST THE LOCKED ENUM (§17 SUPERSEDED)

`state: 'ok' | 'pending' | 'stale' | 'unavailable' | 'not-applicable'` — @12e42da8's lock. This section is the answer to their direct question: *do any two render identically, and should they be collapsed?*

### 21.1 Conceding `ok`

I adopted `measured` from @bb2ee824 and argued for it twice. **The Lead's reason defeats mine and I want it recorded as a defeat, not a compromise.**

We ratified `source: 'server'|'client'|'derived'|'estimated'|'simulated'`. A **derived** value is not measured. An **estimated** value is emphatically not measured. So `state: 'measured'` beside `source: 'estimated'` is **a field contradicting itself in the one place we promised the visitor could trust.** I had the axes tangled: I was using `state` to assert epistemic quality, which is `source`'s job.

**`state` answers CAN I TRUST THIS. `source` answers WHERE DID IT COME FROM.** Orthogonal. `ok` is the weaker word and that is exactly why it is correct — it makes no claim that `source` might contradict. D65.

### 21.2 The five treatments

| state | scalar | chart | carries |
|---|---|---|---|
| `ok` | `41` full contrast. **A real 0 renders as a stark `0`** | normal line | source badge |
| `pending` | `···` | faint baseline + `AWAITING DATA` | nothing |
| `stale` | `41 · 12s old` — age in words, **on screen** | line **stops**, never forward-filled | age |
| `unavailable` | `—` + hover, apologetic, names the file | 45° hatch + `NOT MEASURABLE YET` | reason (hover) |
| `not-applicable` | `—` + **on-screen caption** | hatch + caption naming the execution path | reason (**visible**) |

**One amendment to the Lead's list: `pending` is `···`, NOT blank.** A blank cell is indistinguishable from a layout gap, so during the first poll the dashboard reads as half-built rather than loading — and worse, it **reflows when data lands**, which is the single most damaging first-paint behaviour we could choose. `···` reserves the box, and the three dots are a near-universal "working" signal that needs no legend. This is the only place I've changed the Lead's spec and it is a rendering detail, not a semantic one.

### 21.3 DIRECT ANSWER: two states DO risk rendering identically

**`unavailable` and `not-applicable` are both `—` plus a hatch.** As specified — *"unavailable=hatched, not-applicable=explained"* — they are distinguishable **only if "explained" means ON SCREEN.** If `not-applicable`'s reason lives in a hover like `unavailable`'s, then the two are pixel-identical at rest, the distinction exists only in the spec, **and you should collapse them.**

**My recommendation: do not collapse — make `not-applicable` the one state explained on screen.** Three reasons:

1. **A fact nobody hovers over is a fact nobody learns.** This distinction only pays off if it's read, and it's the single most interesting thing we found. Hover is where we put *apologies* (`unavailable`), because apologies should be available but not prominent. Assertions go on the page.
2. **It is about to be the most-seen non-`ok` state**, because Scenario A ships first and alone on the scatter origin, where every paged-KV and prefix field is `not-applicable`.
3. **Touch and keyboard have no hover.** A distinction carried by hover doesn't exist on a phone.

So the two differ by **presence of on-screen text**, not by a shade of grey — which also satisfies the grayscale-screenshot gate without spending a colour. D66.

### 21.4 Where the five actually appear

`ok` everywhere; `pending` first ~250 ms of every panel; `stale` throughout Scenario C (pressure delays polls **by design**, so it is expected, not exceptional); `unavailable` on the genuinely stubbed fields (`tokens_per_second`, `batch_utilization`); `not-applicable` on every paged-KV and prefix field in Mode A and the batching fields in Mode B. **All five are on screen in the first two minutes.** None is decorative.

---

## 22. 🔴 THE `lookups == 0` GUARD DOES NOT FIRE ON THE CASE WE ACTUALLY HAVE

@fc8b5d97's baseline reports, from a live server:

> **prefix cache is 0 hits / 135 lookups**

@376a0297's binding rule is *"`hit_rate` MUST be `unavailable` when `lookups == 0`."* **`135 != 0`. The guard does not fire.** The field renders a stark, confident **`0%`**.

And by our own standing rule that is a fabrication: `prefix_cache_lookups` increments on **every completed generation** (`metrics.rs:130-134`), and on the batching path `prefix_cache_hit_len` is a **hardcoded literal `0`** (`batched.rs:262`, `:486`) with `ContinuousBatchManager` holding no reference to `engine.prefix_cache` at all (`batched.rs:101-110`). So `0/135` means **"135 generations finished on a path that structurally cannot record a hit."** Rendering `0%` asserts the cache was consulted 135 times and missed every time. We never measured that. **The numerator is not a count — it is a constant.**

**This is the same trap a second time, and that is the finding.** We caught the misnamed denominator and wrote a guard keyed on the denominator — but the denominator is the part that *works*. `135` is a real count of real generations. **The lie is in the numerator, and no threshold on `lookups` can ever detect it.** A guard derived from an incident tends to be shaped like the incident rather than like the fault.

**RULING (D67): the prefix hit-rate guard is keyed on EXECUTION PATH, not on any counter value.**

```
if (engineMode !== 'dynamic')      -> 'not-applicable'   // structural; cite batched.rs:101-110
else if (lookupsDelta === 0)       -> 'pending'          // dynamic, nothing asked yet
else                               -> 'ok'               // only here may a stark 0% render
```

Note this also **upgrades** the Mode A case from `unavailable` to `not-applicable` — correctly, since nothing is broken. And it keeps @376a0297's rule as the inner branch where it is right. ~~Combined with §19's windowed delta, `lookupsDelta === 0` is `pending` rather than `unavailable`, because within a *dynamic* scenario the number genuinely is coming.~~

> 🔴 **STRUCK — THIS SENTENCE IS WHAT D59 AUTHORISED, AND IT IS THE REASON THE WITHDRAWAL NEEDED A GREP RATHER THAN AN EDIT.** With D59 withdrawn and no prefix field bindable in any form, **`pending` here would promise a number that is never coming** — the precise failure `pending` exists to prevent. **The whole derivation is moot: per @376a0297's AC81 the field is NOT BINDABLE AT ALL, so it has no state, because it has no cell.** See §65.

**Standing rule I'd like added to the reviewer checklist:** *a guard keyed on a field's value can only catch faults in that field.* When a ratio is suspect, check the **numerator and denominator separately** — they usually have different provenance, and ours do: real count over hardcoded constant.

| # | Decision | Rationale |
|---|---|---|
| D65 | `ok` over `measured` — conceded | `measured` + `source: 'estimated'` is a self-contradicting field. State = can I trust this; source = where from |
| D66 | Don't collapse `unavailable`/`not-applicable`; separate them by ON-SCREEN text vs hover | Otherwise pixel-identical at rest. Hover holds apologies; assertions go on the page, and hover doesn't exist on touch |
| D67 | Prefix hit-rate guard keys on execution path, not `lookups == 0` | Live data is `0/135` — the guard never fires. The denominator is real; the numerator is a hardcoded constant |

---

## 23. NAVIGATION IS THE SCENARIO SWITCHER — AND IT IS THE HONEST DESIGN, NOT A CONCESSION

@732c7548 found that two origins plus AC39 (no CORS) makes an in-page fetch across servers impossible: `:8123` cannot fetch `:8124`, the browser blocks it, and we are forbidden from adding the header that would permit it. Their Option 1 — **each server serves its own `/demo`; switching scenario NAVIGATES to the other origin** — is correct, and I want to endorse it on design grounds rather than accept it as a workaround.

### 23.1 The constraint resolves §14

§14 was my own unresolved risk: **a tabbed switcher asserts continuity of subject.** Tabs are a container metaphor — same thing, different facet. So a visitor who clicks between Scenario A and Scenario B concludes *one runtime does both*, assembled entirely from true statements, which is exactly the false impression we are trying not to create. I had no clean fix; every mitigation was a caption apologising for the metaphor.

**Navigation fixes it structurally.** A page load is the web's native signal for *you are now somewhere else*. The URL changes, the back button arms, the load is visible. We do not have to *tell* the visitor these are two engines in two processes — **the interaction demonstrates it**, and demonstration beats explanation every time.

This is the same principle as the field envelope: **enforce the property in the mechanism, not in prose.** AC42 (no cross-mode value bleed) stops being a rule a developer must not violate and becomes a fact — a fresh document cannot carry a stale value from a process it never spoke to. D68.

### 23.2 What this buys, beyond correctness

**Attribution collapses to something trivial.** Each page talks only to its own origin, so every number on screen came from the server that served the page. The whole `origin`/`server` tangle (§15, §17) reduces to: **the page's own origin IS the attribution**, stated once in the header, true for every panel by construction. AC41 becomes a one-line header, not an envelope key that every producer must remember to set.

**The `not-applicable` story gets a physical home.** On the scatter page, the paged-KV panels aren't "off" — they belong to a different address you can visit. The caption can say so and mean it literally.

### 23.3 The cost, and the four things that pay it

The cost is real: **every switch is a cold start.** No warm store, no retained history, `pending` on first paint. `pending` becomes the most-seen transition in the demo, which retroactively justifies §21.2's insistence that it be `···` and never blank.

1. **The URL carries scenario state** — `/demo?scenario=paged-kv`. Non-negotiable: without it the destination opens on the wrong panel and the switch reads as a bug. It also makes every scenario **linkable**, which is how anyone will share this.
2. **Both origins serve BYTE-IDENTICAL shell markup and CSS.** Header, nav, panel frames, and legend must paint at the same coordinates on both pages, so the only thing that visibly changes is panel *content*. Done right, the switch reads as *panels repopulating*, not as *a different site*. This is the single highest-leverage detail here, and it costs nothing but discipline: one shell, two servers.
3. **Never a full-page spinner.** The shell is static and should paint instantly; only the fields are `pending`. A whole-page loader would throw away the continuity that (2) just bought.
4. **The switcher NAMES its destination** — `Scenario B · dynamic engine · :8124`. An unexplained navigation feels like a misclick; an announced one feels like a tour. This also puts the port on screen for the skeptic who wants to `curl` it, which is AC-aligned and free.

### 23.4 Two things we must NOT do

**No cross-fade, no transition animation on the switch.** An animation that makes a navigation look like an in-page tab change re-tells the exact lie §23.1 just eliminated — it would be spending effort to reintroduce the problem. Same reasoning as §18.2 (no cell animation without stable identity) and as never bridging a chart gap: **do not animate across a discontinuity you have no continuity for.**

**Do not hide the back button's effect.** Back returning to the previous scenario, fully reloaded, is the strongest available evidence that these are genuinely two pages against two engines. It is free and we should not defeat it with history manipulation.

| # | Decision | Rationale |
|---|---|---|
| D68 | Scenario switching is a **navigation** between origins, not an in-page tab | Two origins + no CORS makes fetch impossible — but it also resolves §14: tabs assert continuity of subject, navigation admits discontinuity. AC42 becomes true by construction |
| D69 | Byte-identical shell markup/CSS on both origins; no page spinner; no switch animation | The switch should read as panels repopulating, not a different site — while never *pretending* to be an in-page transition |
| D70 | The switcher names its destination engine and port | An unexplained navigation is a misclick; an announced one is a tour. Puts the curl-able port on screen for free |

---

## 24. TWO CONSEQUENCES OF THE BASELINE NUMBERS

### 24.1 🔴 CORRECTION TO §21.4 — `pending` IS A TWO-SECOND STATE, NOT A FLICKER

§21.4 said `pending` appears for "the first ~250 ms of every panel." **That is wrong, and @fc8b5d97's baseline is what shows it: median TTFT is 2141 ms.**

Combined with D68 (scenario switching is a navigation, so every switch is a cold start), the real first-paint sequence is: page load → shell paints → panels `pending` → **~2.1 seconds of nothing** → first token. And Scenario A fires four concurrent requests, so the *interesting* panels — batch occupancy, throughput — stay `pending` for longer still.

**Two seconds is not a flicker. It is long enough to read, long enough to doubt, and long enough to click something else.** A `···` that communicates perfectly at 250 ms communicates *nothing* for two seconds — the visitor has already finished asking "is this broken?" before it resolves. And this now happens on **every scenario switch**, which is the demo's primary interaction.

**Amended treatment (D71): `pending` scales with its own duration.**

| elapsed | scalar | panel |
|---|---|---|
| 0–400 ms | `···` | nothing else — avoids a flash of explanatory text on fast paths |
| 400 ms+ | `···` | panel caption appears: **`Waiting for first token — prefill takes ~2s on CPU`** |
| 5 s+ | `···` | caption becomes **`Still waiting. Is the server on :8124 running?`** with the exact command |

The 400 ms threshold is the standard perceptual boundary for "did that work?" The middle state is the one that matters, and note **it makes the wait informative**: prefill *is* slow on CPU, that's a true and relevant fact about this runtime, and a visitor who reads it has learned something instead of merely waited. The 5 s state exists because under D68 a dead second server produces an *infinitely* pending page that otherwise looks identical to a slow one.

**This is the same error I keep catching in others: I specified a state's appearance without checking its real-world duration.** A treatment is not fully specified until you know how long a human looks at it.

### 24.2 A MEASURED NUMBER MUST CARRY ITS CONDITIONS — ON SCREEN

`2.46×` is the hero. @12e42da8: *"Never round it, never restate it without its conditions."* That is a **UI requirement**, and our envelope has nowhere to put `n`, `CV`, and a CI.

**I am not adding envelope keys** (it has been revised three times already). Instead: **a benchmark result is a claim, not a reading, and a claim shows its method adjacent to itself — on screen, not in a hover.** Same rule as `not-applicable`: the thing that makes it credible is the thing nobody hovers over.

> 🔴 **THE SKETCH THAT WAS HERE SHOWED `2.46×` ALONE — STRUCK, and I am leaving the correction visible rather than silently patching it, because the failure is more instructive than the fix.** It carried `n`, `CV`, the EP and the conditions, and it **still violated AC50**, which was ratified five sections later as D85. **A sketch is a build instruction.** @bb2ee824 or @c8d9a40e reading this section in isolation would have built the lone hero and been *correct per §24.2* the entire time.
>
> **This is D158 landing on my own document for the third time tonight** — after the `<meta>` description and the Profile D hero slot. All three were **cut or qualified fields still bound in PROSE THAT INSTRUCTS**, and **none of the five test files I own can see any of them.** `page-claims.test.js` reads shipped HTML; the tripwire reads module identifiers; **nothing reads the design document, which is the file developers actually build from.**

**Corrected sketch — AC50/D85 compliant, and this is the only benchmark form in this document:**

```
      4 concurrent requests vs 1

      TOTAL  82.130 tok/s   2.46× faster
      EACH   20.7   tok/s   0.62× as fast

      Batching does not make any single request faster.
      n=15 · CV 1.98% · CPU EP · max_batch from payload · one machine
```

Rules: **never `~2.5×`** — a rounded benchmark figure with no interval is strictly less honest than the precise one, because rounding *looks* like modesty while actually discarding the only evidence of precision we have. Never a bare percentage. If `n` and `CV` aren't available for a number, it is not a benchmark result and must not be styled as one. **And never `2.46×` without `0.62×` at identical type size — see §29.1: type size is a claim about which number matters, so a caveat set smaller is still the lie, told more quietly.**

> **`max_batch` is read FROM THE PAYLOAD** (@e00032a4's 3-line emit, ruled by @12e42da8) — **never from `state.rs:25`.** The condition line of a benchmark is the last place a compile-time constant may pose as a measured run parameter: **it is the part of the display whose whole job is to say how the measurement was taken.**

| # | Decision | Rationale |
|---|---|---|
| D71 | `pending` escalates at 400 ms and 5 s; the 400 ms caption states *why* prefill is slow | Real TTFT is 2141 ms, not 250 ms. Two seconds is long enough to doubt — and under D68 it recurs on every switch. An informative wait teaches; a silent one worries |
| D72 | Benchmark figures render with n, CV and conditions on screen; never rounded, never bare | Rounding looks like modesty while discarding the only evidence of precision. A claim shows its method next to itself |

---

## 25. THE BLOCK GRID, BUILT ON WHAT IS REAL

@e00032a4's `telemetry-contract.md` §6 lands three arrays plus a sparse `shared_blocks` map, and — more valuable — a list of metrics that are **live today**. This section rebuilds Scenario C on those.

### 25.1 🔴 A SMALL KV POOL IS THE RIGHT CALL, AND IT MUST BE DISCLOSED ON SCREEN

Running the demo with a deliberately small KV budget is correct: 14,612 cells is ~1 MB/s at 4 Hz, teaches nobody, and **would never fill during a demo** — so the large pool is not merely expensive, it is undemonstrative. A small pool visibly fills, shares, and refuses. Adopt.

**But it is a STAGED CONDITION, and an undisclosed staged condition is a fabrication of exactly the kind we've spent all day eliminating.** A visitor watching a pool saturate in fifteen seconds will conclude *this runtime's KV pool fills in fifteen seconds*. That is false, and we would have caused it — not by printing a wrong number, but by **printing true numbers under conditions we chose and didn't mention.** Every value in the grid is honest; the impression is not.

This is the same failure as §14's tabs and the same remedy as D72's benchmark conditions: **the conditions travel with the claim, on screen.**

**D73 — the grid header states the configured pool size and that it is configured:**
```
Paged KV block table · 256 blocks · pool deliberately sized small for this demo
Production pools are far larger (14,612 blocks on this model) and would not
fill in a demo. A small pool shows the same mechanics at a visible speed.
```
That caption costs one line and converts a staged demo into an **honest** one — and note it *adds* information: the real number, 14,612, is itself interesting and is otherwise never shown.

### 25.2 CELL ENCODING — FILL IS GEOMETRY, NOT COLOUR

`block_fill` is a gift, because it expresses the thing paged KV actually costs: **internal fragmentation.** A half-filled block is capacity you paid for and cannot use. That is the honest, non-obvious story of paging, and no competing dashboard shows it.

Each cell is a small square drawn as a **fill bar**, so occupancy is read as geometry and survives the grayscale-screenshot gate with no colour spent:

| block state | encoding |
|---|---|
| free (`blocks[i] === null`) | empty square, 1px rule, no fill |
| owned | fill bar rising from the base to `block_fill[i] / slot_capacity` |
| **shared** (`block_refs[i] > 1`) | owned fill **plus a notched top edge** — a shape, not a tint |
| out of window | not drawn at all; never a placeholder that could read as free |

Sequence identity uses the Okabe-Ito ramp (`--og-seq-0..7`) as a **secondary** channel only. **Nothing is encoded by colour alone**, so the grid is fully readable in grayscale and to a colourblind viewer — and per D63/§18.2 there is **no per-cell animation**, since we still have no guarantee that index `i` is the same physical block between frames.

### 25.3 SHARING IS AN INTERACTION, NOT A DRAWING

`shared_blocks: {"2": [3, 7]}` tempts us to draw connector lines between cells. **Don't** — in a few-hundred-cell grid, connectors produce a hairball at exactly the moment sharing gets interesting, and the picture becomes least legible when it matters most.

**Instead, sharing is revealed by selection.** Hovering or focusing a sequence in the legend dims every block it doesn't own and outlines the ones it **shares**, naming the co-owner: `Block 2 · shared with sequence 7 · retained by the prefix cache`. Progressive disclosure: the resting grid shows *how much* is shared (notched cells, countable at a glance), and selection answers *with whom* on demand. Keyboard-reachable, so it isn't hover-only. D74.

`refs > seqs` is the strongest link between Scenarios B and C — **prefix-cache retention made visible in the allocator** — and it deserves that sentence in words, not a line on a canvas.

### 25.4 SCENARIO C IS NOW `allocation_failures`, NOT EVICTION

Given eviction has zero consumers and the vram knob 403s, @e00032a4 is right that `allocation_failures` is the honest pressure signal. ~~Scenario C's arc becomes: raise concurrency → watch fill bars climb and fragmentation appear → **the pool refuses** → `allocation_failures` increments and admission visibly stalls.~~

> 🔴 **STRUCK IN PLACE — THE DRIVING ACTION IS WRONG. `driver.rs:777` calls `run_fallback_generation` INLINE inside `handle_driver_command(engine: &mut Engine)`, so the dynamic server SERIALISES generations: concurrent requests QUEUE and never coexist, and the block grid never moves.** Superseded by **D82 — pressure is driven by PROMPT LENGTH**, plus D80's shared-prefix branching and D73's small pool. **This paragraph was already dead 60 lines below and I left it readable; @376a0297 read it live and spent a whole AC (AC76) re-deriving the correction I had already written. See §62.**

The headline is **`allocations` / `allocation_failures` / `frees`** with `hot_evictions` and `prefix_evictions` beside them, all real today. ~~The panel's one-line thesis: *"the pool stops accepting, it does not reclaim"* — true, verifiable, and more interesting than eviction because it shows backpressure reaching admission.~~

> 🔴 **STRUCK IN PLACE — THIS SENTENCE IS FALSE AND I WROTE IT. The pool DOES reclaim:** `paged_decode.rs:44 evict_until_free()` → `evict_lru()`, live-called at `flat_autoregressive.rs:307`, refcount-aware so it will not take a page a live sequence is borrowing. **What does not reclaim is the VRAM-CEILING knob** (`ByteBudget::reconfigure` moves `state.limit`, never `state.used`). **I generalised one subsystem to the whole allocator — and my own §55 records the correction 430 lines below this line, where nobody reading the panel thesis will meet it.** @376a0297 is quoting this sentence into the README verbatim. **Replacement thesis, two verifiable claims instead of one unverifiable adjective:** *"pages are reclaimed by refcount-aware LRU when the pool runs dry; changing the VRAM ceiling reclaims nothing — it refuses the next allocation."* See §64.

`max_batch` renders `null` until `--max-batch` lands — **never `x/4`**, since 4 is a hardcoded constant (`state.rs:25`) and printing it as a denominator would be a fabricated ratio wearing a real numerator. Same shape as the `0/135` trap in §22.

| # | Decision | Rationale |
|---|---|---|
| D73 | Small KV pool adopted — and the grid header **discloses** that it is configured, with the real 14,612 figure | Undisclosed staged conditions produce a false impression from true numbers. Disclosure also surfaces a genuinely interesting real number |
| D74 | Sharing revealed by selection, never by connector lines | Connectors hairball exactly when sharing gets interesting. Resting grid = how much; selection = with whom |
| D75 | Cell fill is geometry; shared is a notch; colour is secondary everywhere | Passes the grayscale gate with no colour spent, and makes internal fragmentation — paging's real cost — the visible story |

---

## 26. 🔴 THE NAVIGATION SWITCHER LOSES OUR ERROR STATE WHEN THE OTHER SERVER IS DOWN

D68 (navigation, now ratified) plus the Lead's *"all three scenario tabs are always visible"* combine into a failure nobody has costed:

**A visitor running only the scatter server clicks Scenario B. The browser navigates to `http://127.0.0.1:8124/demo`. Connection refused. They get CHROME'S ERROR PAGE.**

`ERR_CONNECTION_REFUSED`, *"check the proxy and the firewall"* — Chrome's guesses, not ours. **We do not merely fail to show our designed state; we lose the page entirely.** AC37's actionable connection state, the verbatim server-start command, the honesty footer, the tab strip back to a scenario that *does* work — all gone, because it is no longer our document. And the browser's advice is **actively misleading**: nothing is wrong with their proxy or firewall; they simply haven't started a second process, which is the one thing Chrome cannot guess and we know for certain.

This is worse than the CORS trap @376a0297 caught, and in the same family. There we would have shown a confident wrong message; **here we show someone else's confident wrong message and cannot even be blamed for it — which means nobody will notice in review.** It only appears when a server is missing, which is precisely the first-run condition.

**And it is the DEFAULT PATH, not an edge case.** `run-demo.sh` may start both, but a visitor following the README one command at a time, or whose second server died, or who is on the single-model machine the README warns about, hits this on their first click.

### 26.1 Fix: probe before you navigate

We cannot `fetch` the other origin's JSON — no CORS. But **`fetch(url, { mode: 'no-cors' })` needs no CORS headers**: it returns an opaque response we cannot read, which is fine, because we are not asking *what did you say*, only **is anything listening**.

```js
async function reachable(origin) {
  try { await fetch(origin + '/demo', { mode: 'no-cors', cache: 'no-store' }); return true; }
  catch { return false; }          // TypeError === nothing listening
}
```

- **Reachable → navigate.** Normal path, one extra round trip on loopback.
- **Not reachable → DO NOT NAVIGATE.** Stay on our page and render our own designed state, in place: *"Scenario B needs the dynamic-model server on :8124. It isn't running."* plus the copy-pasteable command and a **Retry**.

**We keep the document, so we keep the tab strip** — the visitor can go back to a scenario that works instead of pressing Back on a browser error page. D76.

**Verification status, stated precisely because I'm being held to a browser-verified bar:** I confirmed the **rejection** side — a dead port rejects with `TypeError` / `ECONNREFUSED` (node 24, `fetch` against `127.0.0.1:8124`). I have **not** verified the **resolve** side in a browser against a live server, because that needs a running server and I hold the no-cargo line. @bb2ee824: this belongs in your Playwright harness — assert that a live origin's no-cors probe resolves opaquely and a dead one rejects. **If an opaque resolve turns out not to be reliable, this design fails and I need to know.**

### 26.2 The tab strip must state what each scenario requires

Since all three tabs are always visible (Lead's ruling, superseding my §13 filtering), a tab pointing at a server that isn't running should **say so before it's clicked** — a small `:8124` on the tab, and a dimmed-but-focusable treatment once a probe has failed. Probe both origins once on load; **never poll a foreign origin**, since that's a request per second against a server we don't own the page for.

This is strictly better than my §13 design, and I want to record why I was wrong: **hiding B and C on the scatter server would have concealed two-thirds of the project.** A visitor would never learn paged KV exists. Visible-but-explained teaches; hidden teaches nothing — and it is the same argument I made for `not-applicable` over suppressing a panel. I applied it to fields and not to navigation. D77.

| # | Decision | Rationale |
|---|---|---|
| D76 | Probe the target origin with `mode:'no-cors'` before navigating; on failure render our own state in place | Otherwise the browser's error page replaces our document, deleting AC37's actionable state and offering actively wrong advice about proxies and firewalls |
| D77 | All three tabs always visible, each naming its port and requirement; no polling of foreign origins | Hiding B and C would conceal two-thirds of the project. Same argument as `not-applicable` over suppression — I'd applied it to fields but not to navigation |

---

## 27. 🔴 SCENARIO C REDESIGNED AGAIN — THE DYNAMIC SERVER SERIALISES

@d7cf9b84: *"The DYNAMIC server SERIALISES generations — one engine, one driver thread, generation runs INLINE (`driver.rs:696`). Concurrent requests do NOT overlap; they queue."*

**This kills §25.4 and D55, both written in the last half hour.** I specified Scenario C as *"raise concurrency → fill bars climb → the pool refuses."* Concurrency-driven pressure works only on the **scatter** server, which does no paged-KV work at all. **On the server that has a block table, concurrency does nothing but queue.**

**That is twice I have storyboarded Scenario C around a behaviour the runtime doesn't have** — first eviction (§7.4), now concurrency. Both times the panel would have been beautiful and inert. The pattern in my own errors is specific and worth naming: **I keep designing the DRIVING ACTION from a mental model of how such a system usually works, then checking only whether the DISPLAYED FIELDS are real.** Provenance discipline covers the numbers; it does not cover the verb. **A scenario has a provenance too — "can this action actually produce this effect on this path" — and I haven't been auditing it.**

### 27.1 What actually drives paged-KV pressure

Blocks accumulate and are shared through **sequential** requests: sessions and repeated prefixes, where copy-on-write sharing arises from prefix reuse via the trie. So Scenario C becomes:

**Send a long shared prefix, then a series of sequential requests that branch from it.** Blocks fill, `refs > 1` appears as sequences share the common prefix, fragmentation shows in partially-filled blocks, and — with the small pool of D73 — allocation eventually **refuses**, incrementing `allocation_failures`.

Same three teaching points (allocation, sharing, refusal), driven by an action that genuinely produces them. It also **unifies B and C**: prefix reuse *is* the sharing mechanism, so the two Mode B scenarios stop being neighbours and become one story told at two altitudes.

### 27.2 🔴 THE GRID CANNOT BE POLLED AT 4 Hz — AND POLLING IT WOULD LIE

The sharper half of @d7cf9b84's finding: a telemetry read **queued behind a generation is not serviced until the generation finishes.** So the block table changes only **between** requests.

**Poll that at 4 Hz and you get the same snapshot repeatedly, then a jump.** Rendered as a live-updating panel, that reads as *"we are watching the allocator continuously and it happens to be flat during generation"* — which is **false**, and false in the most damaging way available to us, because **flat-because-unobserved is pixel-identical to flat-because-nothing-happened.** It is the flat-zero-line problem and the frozen-chart problem in one, on the panel we most want believed.

**D78 — the block grid is EVENT-SAMPLED, not time-polled.** Sample on request completion; render **discrete sample points with visible gaps between them**, never a continuous line. Label it exactly:

```
Paged KV block table · sampled between requests
The engine services telemetry on the same thread that decodes, so the
allocator can only be read between generations. These are snapshots,
not a continuous recording.
```

Note this is **more informative than a live view would be**, because it teaches something true about the runtime's threading — and it converts a limitation into the kind of specific, checkable statement that makes the rest of the dashboard credible.

**D79 — if the Lead approves @d7cf9b84's ~20–30 line wait-free atomics change, the panel upgrades to live and the caption is DELETED, not amended.** I'd support that ruling: allocation is inherently a during-generation phenomenon, so between-request sampling shows the *aftermath* of paging rather than paging itself. But the sampled version is honest and shippable, so this is an upgrade, not a dependency.

### 27.3 Navigation is still right even if CORS lands

@d7cf9b84 is hand-rolling ~25 lines of CORS middleware. **With D76's `no-cors` reachability probe, the demo needs no CORS at all** — the probe answers *is anything listening* without a single response header, and navigation keeps every fetch same-origin.

I'd argue **against** using CORS for the switcher even once it exists, on design grounds already recorded: in-page cross-origin fetching reinstates §14's problem — **tabs assert continuity of subject, so the visitor concludes one runtime does both.** Navigation is not a workaround for CORS; it is the design that tells the truth about topology, and CORS would let us quietly stop telling it. It also keeps AC35's attack surface at zero and avoids shipping a demo that instructs people to run a permissively-CORS'd server.

| # | Decision | Rationale |
|---|---|---|
| D78 | Block grid is event-sampled on request completion; discrete points, visible gaps, caption naming the threading | The engine services telemetry on the decode thread, so 4 Hz polling repeats one snapshot. Flat-because-unobserved is pixel-identical to flat-because-nothing-happened |
| D79 | Wait-free atomics upgrade → panel goes live and the caption is deleted | Allocation is a during-generation phenomenon; sampled shows the aftermath. Honest and shippable meanwhile |
| D80 | Scenario C driven by a shared prefix plus sequential branching requests, not concurrency | The dynamic server serialises. Concurrency only queues, so the panel would be inert. Also unifies B and C — prefix reuse IS the sharing mechanism |
| D81 | Keep navigation even if CORS ships | In-page cross-origin fetch reinstates the continuity-of-subject lie. D76's probe means CORS is not needed for the demo at all |

---

## 28. THREE COLLISIONS FROM THE LATEST RULINGS

### 28.1 The Scenario C ruling inherits a premise that was killed 20 minutes ago

@12e42da8: *"RULING: raise CONCURRENCY and PROMPT LENGTH until the pool genuinely fills."*

**Concurrency cannot fill the paged-KV pool.** @d7cf9b84 verified the dynamic server serialises — generation runs inline on the driver thread (`driver.rs:696`), so concurrent requests queue rather than overlap. Concurrency creates pressure only on the **scatter** server, which has no page table at all.

This is the Lead's own standing rule applying to the Lead's own ruling: **when a decision is reversed, everything justified by it must be re-examined, not silently inherited.** The vram-knob reversal was correct; the replacement inherited "concurrency" from the pre-serialisation picture.

**The ruling is half-right, and the surviving half is sufficient: PROMPT LENGTH WORKS.** A long prompt consumes many blocks in a single sequential request, so pressure is reachable without any concurrency at all. Combined with D73's small pool and D80's shared-prefix branching, Scenario C's driver is:

**long shared prefix → sequential branching requests → blocks accumulate and share → allocation refuses.**

No concurrency anywhere in Mode B. D82.

### 28.2 The swimlane: four states, and the absence stated ONCE

Two instructions to reconcile: *"the swimlane is 4 states, not 5"* and *"`preempted` renders as `not-applicable` — never a lane that silently never fires"*, alongside *"do NOT allocate layout space to a preemption lane."*

**Resolution: there is no preemption lane, and the absence is stated once in the panel caption, not as a permanently-empty row.** A greyed fifth lane spanning the full width would be the largest `not-applicable` treatment on the page, given the most visual weight of any element in the panel, to say nothing happened — **the layout would be asserting importance that the content contradicts.** One caption line carries the same fact at its true weight:

```
Four lifecycle states on this path. Preemption is disabled by construction
here — a continuous batch owns its KV in physical rows that cannot be
swapped out and resumed in place (batched.rs:713-717, :757).
```

**Layout space is itself a claim about significance.** That is the general rule this instance illustrates, and it's why `not-applicable` is a *treatment*, not automatically a *slot*. D83.

### 28.3 🔴 `batch.queued` INHERITS A FABRICATION FROM THE HARDCODED `4`

@bb2ee824's catch is excellent and exactly the label-honesty failure of §16: `onnx_genai_batch_size_current` is `fetch_add(1)` in `GenerationMetrics::start()` (`metrics.rs:112`), so it counts **generation requests in flight at the HTTP layer**, not the engine's batch. Never call it "batch size."

**But their derived field carries a defect forward.** Proposed: `queued = max(0, in_flight - 4)`. **The `4` is `state.rs:25`, a compile-time constant — not a measurement.** So `batch.queued` is a real number minus a hardcoded literal, presented as an observation of the scheduler. **It is the `x/4` trap from §25.4 and the `0/135` trap from §22 in a third costume: a genuine numerator combined with a fabricated denominator produces a value that passes every provenance check because its ingredients each look fine.**

This one is worse than `x/4`, because a *ratio* against a suspicious denominator at least looks like an estimate, whereas `queued: 4` looks like a count the scheduler handed us.

**D84 — until `--max-batch` lands, `batch.queued` is `unavailable`**, with the reason naming `state.rs:25`. `batch.in_flight` ships as `ok` — it is genuinely measured, correctly named, and interesting on its own. When `--max-batch` lands and the limit is a reported value rather than a literal, `queued` becomes `ok` with `source: 'derived'` and no further change.

**Third time today with the identical shape.** The general form is worth adding to the reviewer checklist: **a derived field is only as honest as its least honest input, and derivation LAUNDERS provenance — the output looks like a single clean number with no visible seam where the constant went in.** @bb2ee824's `numericValueOf` returning `null` rather than a number is the right defence at the value level; this is the same defence needed at the *formula* level.

| # | Decision | Rationale |
|---|---|---|
| D82 | Scenario C drives pressure with PROMPT LENGTH, not concurrency | The dynamic server serialises; concurrency only queues. Prompt length fills blocks in a single sequential request |
| D83 | No preemption lane; the absence is one caption line | A permanently-empty full-width lane is the largest not-applicable on the page — layout space is a claim about significance |
| D84 | `batch.queued` is `unavailable` until `--max-batch` lands; `batch.in_flight` ships as `ok` | `queued = in_flight - 4` embeds a compile-time literal. Derivation launders provenance: no visible seam where the constant went in |

---

## 29. THE HERO IS A TRADEOFF, NOT A NUMBER

@12e42da8: aggregate decode is **2.46×** at 4 concurrent, but **per-stream throughput falls to ~0.62×** (~20.7 tok/s). Both ship together, everywhere. *"A tradeoff presented as a pure win is a lie told with true numbers."*

### 29.1 Equal weight, because hierarchy is an argument

The obvious implementation is `2.46×` large with `0.62×` small beside it. **That still tells the lie**, just more quietly: **type size is a claim about which number matters.** A big number with a small one next to it reads as *the result, with a caveat* — and the caveat is the half an engineer in the audience most needs.

Same principle as D83 (*layout space is a claim about significance*), one level up: **typographic hierarchy is a claim about importance.** We spent all day ensuring values can't lie; a layout that ranks them re-introduces the lie above the value layer, where no provenance check reaches.

**D85 — the two figures render at IDENTICAL type size, weight and colour**, as two halves of one statement:

```
        4 concurrent requests vs 1

  TOTAL    82.130 tok/s     2.46× faster
  EACH     20.7   tok/s     0.62× as fast

  Batching does not make any single request faster.
  It trades per-stream latency for total throughput.
  n=15 · CV 1.98% · CPU EP · max_batch=4 · one machine, not a performance claim
```

The sentence is the hero; the numbers are its evidence. **This is a stronger demo than `2.46×` alone** — anyone who has run a server knows this tradeoff exists, so omitting it wouldn't have flattered us, it would have made us look naive to exactly the audience we most want to convince.

### 29.2 The arithmetic is visible on purpose

`0.62 × 4 ≈ 2.48`. The two figures are **arithmetically linked**, and showing them together lets a viewer verify the relationship in their head — per-stream × concurrency = aggregate. **A display a skeptic can check themselves is worth more than one they must trust.** Same reasoning as citing `file:line` in the `not-applicable` hovers: verifiability is cheaper than persuasion.

So the panel states the relationship rather than leaving it inferable, and never shows one factor without the other. **Any panel displaying aggregate gain shows per-stream cost adjacent** — not in a hover, not in a footnote, not on a second tab. D86.

### 29.3 The swimlane already draws this

Worth naming because it's free: the swimlane shows four decode segments **advancing in lockstep**. That picture *is* the tradeoff — every lane progresses more slowly than a solo request would, and they finish together. **The panel we already designed as the batching proof is simultaneously the per-stream-cost proof.**

So the swimlane caption should say so: *"all four advance together — each slower than it would run alone. That is the trade."* One sentence turns a proof of the win into a proof of the whole truth, at zero layout cost.

| # | Decision | Rationale |
|---|---|---|
| D85 | Aggregate and per-stream figures render at identical type size, weight and colour | Typographic hierarchy is a claim about importance. Ranking them re-introduces the lie above the value layer, where no provenance check reaches |
| D86 | No panel shows aggregate gain without per-stream cost adjacent — never in a hover | The numbers are arithmetically linked (0.62 × 4 ≈ 2.48), so shown together a viewer can verify the relationship themselves |

---

## 30. 🔴 TOKENS/SEC CANNOT COME FROM `/metrics` — AND THE FIX IS BETTER THAN THE PLAN

@fc8b5d97 measured, on the clean tree:

| endpoint | idle | during a 384-token generation |
|---|---|---|
| `/metrics` | 0.8 ms | **14,784 ms** (5 polls completed) |
| `/v1/resources` | 0.8 ms | **14,785 ms** |
| `/v1/status` | 0.9 ms | 1.8 ms (61 polls, clean 4 Hz) |

Root cause: `prometheus_metrics` (`admin.rs:396`) awaits `engine.resource_snapshot()`, which round-trips the busy driver thread.

**This kills the plan I endorsed twenty minutes ago.** I confirmed *"tokens/sec and TTFT do NOT have to be em-dashes — difference `tokens_generated_total`, take `sum/count` on the TTFT histogram."* Both are on `/metrics`. **Polling it at 4 Hz would freeze the entire dashboard for the whole generation** — every panel stale for 15 s, precisely while the visitor watches the thing we built the demo to show.

**And note the shape, because it is our recurring one:** it would have **tested perfectly.** Send one request, it completes, stats arrive. The failure appears only under sustained load — the demo condition, and the one a unit test never reproduces.

### 30.1 The fix: measure tokens/sec in the client, from the stream

**The demo drives its own load.** It knows exactly what it sent and when each token arrived, so it can count tokens off the response stream directly. No endpoint, no polling, no driver round-trip, nothing to stall.

**This is strictly better than the server-side derivation, on four counts:**
1. **It cannot stall**, because it is not a request.
2. **It measures what the user actually experiences** — tokens arriving at the client — rather than what the server believes it emitted. For a *serving* demo, delivered throughput is the more honest quantity.
3. **It is per-request and per-stream**, so it produces the §29 per-stream figure directly. The server counter is process-global and could never have given us that.
4. It needs **zero server work**, on the narrowest node in the project.

`source: 'client'` (not `'derived'`), `endpoint: null`, `server` set to the origin it was measured against — a client measurement still measures *an engine*, and unattributed hero numbers were the §17 failure. D87.

This is the same principle @12e42da8 applied to Scenario B's hit rate: *"the demo drives its own load and knows exactly what it sent, so client-side attribution is EXACT rather than merely adequate."* It generalises further than either of us noticed — **for anything the demo itself causes, the client is the better instrument**, because it can attribute per-request and cannot be starved by the thing it is measuring.

### 30.2 Poll only the fast endpoints

**D88 — the 4 Hz loop polls `/v1/status`, `/v1/debug/kv` and `/health` ONLY.** `/metrics` and `/v1/resources` are excluded from the fetch loop entirely — not "polled slowly", **excluded**, because a single in-flight request against them holds a connection for 15 s and AC24's at-most-one-cycle-in-flight rule would stall the whole loop behind it. If a field is only available on a stalling endpoint, it is `unavailable` with a reason, or it moves to the client.

**And a standing rule for panel authors: an endpoint's IDLE latency tells you nothing about its latency under the load your panel exists to display.** Every endpoint here is 0.8 ms idle. Two of the five are 15,000 ms under load — an 18,000× difference invisible to any measurement taken while the server is quiet, which is exactly how everyone would naturally test.

### 30.3 Two small confirmations

**Yes to machine-readable reason codes** (@d7cf9b84's offer, free now and a breaking change later): emit `reason_code` as a stable enum plus the file:line evidence; **the client owns the prose.** Already the ratified split (server owns truth, client owns voice) — this just makes it enforceable rather than conventional.

**The navigation target is `/demo/` with the trailing slash**, per @d7cf9b84: `/demo` redirects, and relative module imports resolved against `/` would 404 every module and render a blank page with only a console error. D76's probe and every switcher link use `/demo/?scenario=…`.

| # | Decision | Rationale |
|---|---|---|
| D87 | tokens/sec is measured client-side off the response stream, `source: 'client'` | `/metrics` stalls 15 s under load. Client measurement can't stall, measures delivered throughput, and yields the per-stream figure the global counter never could |
| D88 | The 4 Hz loop polls `/v1/status`, `/v1/debug/kv`, `/health` only — the stalling endpoints are excluded, not slowed | One in-flight 15 s request stalls the whole at-most-one-cycle-in-flight loop. Idle latency predicts nothing about latency under demo load |

---

## 31. `not-applicable` IS A PANEL-LEVEL STATE, NOT A FIELD-LEVEL ONE (REVISES §21)

@c0de4c2e: *"my instinct is that `not-applicable` shouldn't be an absence treatment at all — it's the one state where the slot has something to SAY rather than something to withhold."*

**That is sharper than what I ruled, and it exposes why the two states kept threatening to collapse.** §21 gave `not-applicable` an em-dash plus an on-screen caption. But the em-dash is *the* omitted-value convention — so the treatment still **leads with "nothing here"** and explains afterwards. A visitor scanning a column reads the glyphs before any caption. **I was decorating an absence and calling it an assertion.**

### 31.1 The real distinction is SCOPE, not decoration

I had both states operating at field level, where the only available difference is ornament — a different hatch, a different grey. **At field level they can only ever differ by decoration, which is exactly why they kept nearly colliding.** But they aren't the same *kind* of thing:

- **`unavailable` is genuinely per-FIELD.** One stubbed metric sits among working ones — `tokens_per_second` is dead while `queue.depth` beside it is live. An em-dash in the value slot is exactly right: it holds alignment, admits a local gap, and the rest of the panel is still doing its job.
- **`not-applicable` is almost always per-PANEL.** It arises when an entire *subsystem* is bypassed on this execution path. When the prefix cache isn't in scope, **every field in that panel is not-applicable simultaneously** — there is no mixed case, because the cause is structural.

**So they never occupy the same kind of space, and cannot render identically. That is the answer to @12e42da8's collapse question, and it's a stronger answer than "one has a caption."** D89.

### 31.2 The treatment

**A panel whose `meta.requiresMode` isn't satisfied does not render its fields at all.** It keeps its header and frame — so the layout doesn't jump and the visitor sees the panel *exists* — and its **body is replaced by the explanation**:

```
┌─ Prefix cache ──────────────────── not applicable here ─┐
│                                                          │
│  This engine runs continuous batching, which decodes     │
│  into fixed in-place KV rows and never consults the      │
│  prefix trie. ContinuousBatchManager (batched.rs:101-110)│
│  holds no reference to engine.prefix_cache — a lookup    │
│  is not merely absent, it is impossible.                 │
│                                                          │
│  The repo's own tests assert both halves:                │
│  batched_static_decode.rs:53 · prefix_speedup.rs:50      │
│                                                          │
│  → See it measured live in Scenario B (dynamic · :8124)  │
└──────────────────────────────────────────────────────────┘
```

**No em-dash appears anywhere.** There is no row of absences to scan, so there is nothing to misread as breakage. **The panel is not empty — it is full of the most interesting thing we have to say**, and the visitor gets it without hovering, on touch, in grayscale, and in a screenshot.

This also **removes the failure mode the fifth state was created to prevent**, rather than mitigating it: the Lead's worry was *"a visitor's first run shows a dashboard half-covered in what looks like breakage."* Under §31 the scatter server's first run shows **working panels plus explanatory panels**, and not a single em-dash that isn't a genuine gap in our own work.

### 31.3 The field-level state still exists, narrowly

`not-applicable` remains in the enum, because rare mixed cases exist — a single field structurally pinned inside an otherwise-live panel. ~~(`preemptions_total` on the batching path, per D83)~~ 🔴 **THAT EXAMPLE IS DEAD — `preempted_total` was DROPPED outright (Lead ruling 3). See §59.2: the exception survives as a rule with NO CURRENT INSTANCE, and is marked so deliberately.** There it renders as `—` with the on-screen caption from §21. **But it is the exception; the panel-level treatment is the default**, and the panel treatment is what a visitor will actually encounter.

**The general principle, which I got wrong once already today (D83, the preemption lane):** *choose the SCOPE of a treatment before its appearance.* Both mistakes were the same one — I picked how something should look before asking what size of thing it was describing. Layout answers questions that decoration cannot.

| # | Decision | Rationale |
|---|---|---|
| D89 | `not-applicable` is primarily a PANEL-level state: header retained, body replaced by the explanation, no em-dash | Its cause is structural, so it's never mixed within a panel. At field level it could only differ from `unavailable` by decoration — which is why they kept nearly collapsing. Different scopes cannot be confused |
| D90 | Choose a treatment's SCOPE before its appearance | Second time today the same error (D83's lane). Layout answers questions decoration cannot |

---

## 32. 🔴 D65 REVERSED — `measured` IS RESTORED. AND MY CONCESSION WAS ITSELF AN ERROR.

@12e42da8's final ruling reinstates `measured`, `observedAtMs`, and the three attribution keys. **§21.1 and D65 — where I conceded `ok` — are WRONG and superseded by this section.** Anyone reading §21 in isolation would build the wrong enum.

**FINAL: `state: 'measured' | 'pending' | 'stale' | 'unavailable' | 'not-applicable'`.**

### 32.1 Why the reversal is right, and why my concession wasn't

I conceded because the orthogonality argument is clean: `state: 'measured'` beside `source: 'estimated'` looks like a field contradicting itself. **The counter-argument is better and I did not find it:**

> **A reviewer seeing `state: 'ok', value: 0` reads "ok" as "fine" and is tempted to hide it. `measured` makes a genuine zero obviously load-bearing.**

**That is a safety property, not a naming preference** — and it defends the single hardest case in this design, the honest zero we have fought for all day. `ok` is a *verdict*; `measured` is a *provenance claim*, and provenance claims are what this envelope exists to carry.

And the contradiction I capitulated to dissolves under the Lead's definition: **`state` means "do we have a real value from its stated source." It does not claim the source was an instrument.** `source` remains the sole answer to where it came from. `{ state: 'measured', source: 'estimated' }` reads correctly — *we really do have the estimate*.

### 32.2 The error worth recording: I conceded gracefully but not correctly

I have spent this session telling other people to check the specific thing rather than trusting the rule. **Then I accepted a well-formed argument against my own position without looking for its counter — precisely because it was well-formed, and because conceding felt like good practice.**

> **DEFERENCE IS A FAILURE MODE TOO. Conceding gracefully is not the same as conceding correctly.**

Three people who disagreed with the ruling — @376a0297, @bb2ee824, and by implication @c8d9a40e's committed code — found the counter I didn't. **The person best placed to defend a design is its author, and I stopped defending mine one exchange too early.** Worth remembering next time I'm complimented on changing my mind: a reversal is only valuable if it's *correct*, and "I reversed myself" is not by itself evidence of rigour. It's the same trap as the safeguard being where the bug hides — **I audited everyone's reasoning except the reasoning that made me feel reasonable.**

### 32.3 ⚠️ §31 SUPERSEDES §21 FOR `not-applicable` — the Lead is citing the older section

@12e42da8's ruling describes `not-applicable` as *"em-dash + a VISIBLE CAPTION, not a hover."* That was §21, and **§31 (published minutes later, after @c0de4c2e's push) revises it: `not-applicable` is a PANEL-LEVEL state with NO EM-DASH AT ALL.** The panel keeps its header and frame, and its body is replaced by the explanation.

**The Lead's stated rule is preserved exactly and served better** — *"a fact nobody hovers over is a fact nobody learns, and this is the one we most want read"* is the reason the explanation moves out of the value slot entirely rather than sitting beside an em-dash. **An em-dash still leads with "nothing here."** Under §31 there is no absence glyph to scan past.

The field-level em-dash-plus-caption survives only for the rare mixed case (a single structurally-pinned field inside a live panel, e.g. `preemptions_total`, D83). **@bb2ee824, @c8d9a40e: build §31, not §21.** If the Lead prefers §21, say so and I'll revert — but §31 is what the collapse-test answer was built on.

| # | Decision | Rationale |
|---|---|---|
| D91 | `measured` restored; **D65 reversed** | `ok` is a verdict, `measured` is a provenance claim. `state:'ok', value:0` invites hiding the zero — a safety property, not naming taste |
| D92 | Deference is a failure mode; a concession needs the same audit as an assertion | I accepted a clean argument without seeking its counter, because conceding felt like good practice. The people who disagreed found it |

---

## 33. STATE PRECEDENCE IN DERIVATION — `not-applicable` DOMINATES, AND MAY NEVER BECOME A ZERO

@c0de4c2e asked whether `not-applicable` is contagious through `derivedFrom`, and correctly called it **a precedence decision, not an inference**. Ruling it explicitly so nobody implements it by taste.

### 33.1 The total order

```
not-applicable  >  unavailable  >  pending  >  stale  >  measured
```
**A derived field takes the HIGHEST-PRECEDENCE state among its inputs.** Rationale, one line each:
- **`not-applicable` dominates everything.** `pending`, `stale`, `unavailable` all mean *the number could still arrive*. `not-applicable` means **it never will, by construction.** A value derived from a structurally-absent input is not late — it is meaningless on this path.
- **`unavailable` over `pending`** — one input that needs engineering work outranks one that needs 200 ms.
- **`stale` over `measured`** — one stale input makes the whole derivation stale; freshness is a floor, not an average.

### 33.2 D94 — THE RULE THAT ACTUALLY PROTECTS US: a `not-applicable` input may NEVER be substituted with `0` in a derivation

This is where contagion will be quietly broken, because contagion is *inconvenient* exactly where it bites. Consider a memory total where `paged_kv_bytes` is `not-applicable` on the static path. The tempting reading is *"structurally absent means it genuinely contributes zero bytes, so just add 0."*

**Refuse it.** That reasoning is **how the fabricated zero gets back in** — and it arrives wearing a correctness argument, which is why it's more dangerous than the original bug. @bb2ee824's `numericValueOf` returning `null` rather than a number is the enforcement, and the Lead is right that it must treat `not-applicable` as valueless too.

**But blanking every total is a bad answer too**, so the honest resolution is §16's label rule, which we already ratified:

> **IF A TOTAL IS STILL WANTED, THE DERIVATION DOES NOT SUBSTITUTE A ZERO — IT NARROWS ITS LABEL TO THE TERMS IT ACTUALLY SUMMED.**
> Not `Total KV memory` computed over two terms and one silent zero. **`Weights + activations`**, computed over exactly what exists on this path.

A narrowed label is **true on every profile**, needs no asterisk, and states the architectural fact in the one place the visitor is already reading. **The label is part of the claim** — the same principle that stopped us calling in-flight HTTP requests "batch size".

### 33.3 On the grayscale collapse risk (§8.2 QA gate) — it cannot fire as specified

@c0de4c2e flags `unavailable` (hatched) vs `not-applicable` (explained) as the live collapse risk. **§31 already dissolves it structurally rather than visually: `not-applicable` is a PANEL-level treatment and `unavailable` is a FIELD-level one.** One replaces a panel body with prose; the other is an em-dash in a value slot inside an otherwise-live panel. **They cannot render identically in grayscale because they do not render at the same scale.** That is a stronger guarantee than any contrast ratio — **it survives the screenshot test by construction, not by tuning.** Keep the QA gate anyway; a gate that passes by construction costs nothing to run.

| # | Decision | Rationale |
|---|---|---|
| D93 | Precedence `not-applicable > unavailable > pending > stale > measured`; derived takes the highest | Only `not-applicable` asserts the value never arrives; the rest are timing |
| D94 | A `not-applicable` input is **never** substituted with `0`; narrow the LABEL instead (§16) | "Structurally absent means it contributes zero" is the fabricated zero returning with a correctness argument |
| D95 | `reason_code` on the wire, prose `reason` in the client — both, neither optional | Server owns classification, client owns voice; a stale client table can't contradict the server |

---

## 34. 🔴 `prefix_cache_hits` ON THE BATCHING SERVER IS `not-applicable`, NOT `measured` — THE ZERO IS A TAUTOLOGY, NOT AN OBSERVATION

@bb2ee824 marked `metrics.prefix_cache_hits` as **`measured`** so its genuine zero can be shown and captioned. **The intent is exactly right and I'm adopting it. The STATE is wrong, and it's wrong in the one direction our whole envelope exists to prevent.**

### 34.1 The disassembly — verified at four sites

```rust
// batched.rs:262 and :486  — BOTH batched call sites
let loop_state = DecodeLoopState::with_rng(0, rng, options.top_logprobs);
//                                         ↑
// decode_loop.rs:40 — the FIRST PARAMETER IS prefix_cache_hit_len: usize
//
// batched.rs:347 / :579 — read back out and handed to the metric
row.state.prefix_cache_hit_len
//
// metrics.rs:136 — the branch that increments hits
if prefix_cache_hit_len > 0 { REGISTRY.prefix_cache_hits.fetch_add(1, ...); }
```

**The value tested at `metrics.rs:136` is a COMPILE-TIME LITERAL `0`, written 300 lines earlier and round-tripped through a struct field.** No cache lookup ever writes it on this path. **The branch is statically dead.** `prefix_cache_hits` on a batching server cannot increment — not *did not*, **cannot**.

> **A number that could not have come out any other way is not a measurement. It is a restatement of the source code.**

### 34.2 Why `measured` here is the exact failure we built the envelope to stop

`state: 'measured'` promises **"we have a real value from its stated source."** With `source: 'server'`, the badge tells a visitor *the server counted this*. **The server counted nothing. It evaluated `0 > 0`.** Ship that and our provenance badge — the one component whose entire job is to be trustworthy — makes its **first false claim on our headline panel.** @c8d9a40e's grader would pass it; every test would be green.

**And this is the strongest instance yet of the session's pattern, because of HOW it hides:** the literal is at `batched.rs:262`; the metric is at `metrics.rs:136`. **Read either site alone and nothing is wrong** — the metric site sees a variable, the call site sees an innocuous `0` among four positional args. **The fabrication is LAUNDERED THROUGH A STRUCT FIELD**, which is precisely why @bb2ee824's three-evidence analysis (all of it correct) still landed on the wrong state. Their EVIDENCE 1 proves the *mechanism* is wired; it does not prove the *input* is observed.

### 34.3 The denominator is independently broken — do NOT render a hit RATE

`metrics.rs:130-132` increments `prefix_cache_lookups` **unconditionally on every completed generation**, outside any predicate. **It counts generations, not lookups** — it would read 135 with the prefix cache deleted from the codebase. So `0 hits / 135 lookups` is **a statically-dead numerator over a mislabelled denominator**. D96: **no hit-rate percentage is rendered on any profile.** A ratio of two fabrications is the most authoritative-looking number we could possibly put on screen.

### 34.4 What ships — @bb2ee824 gets the panel they argued for

Their recommendation is right and better than an omission: **show the exclusivity as a finding.** §31's `not-applicable` treatment is *built* for this — panel-level, header retained, body replaced by the explanation, **on screen rather than behind a hover, and not apologetic.** So the correct state delivers the teaching surface; `measured` was never needed to get it. **We do not have to misreport the state to tell the story — and if we did, the story wouldn't be worth telling.**

| # | Decision | Rationale |
|---|---|---|
| D96 | `prefix_cache_hits` on scatter/batching = **`not-applicable`**, never `measured` | The tested value is a compile-time literal; the increment branch is statically dead |
| D97 | **Never render a prefix hit-RATE.** `lookups` counts generations, not lookups | Ratio of a dead numerator over a mislabelled denominator |
| D98 | A metric whose input is a constant is `not-applicable` **however well-wired the counter is** | Wiring proves the mechanism, not the observation |

---

## 35. AC CITATION AUDIT vs THE RESTORED SPEC — AND I RETRACT MY OWN "EVICTION" OBJECTION

@376a0297 asked each of us to verify our own AC citations against the reconstructed `demo-spec.md`. Done — 28 ACs cited in this document, all present. Three findings, and **the most important one is that I was wrong.**

### 35.1 ✅ RETRACTED: AC43's "eviction" is CORRECT. My objection was based on the wrong subsystem.

I have been telling the crew to chase an overclaim in our canonical honesty sentence — *"paged KV keeps it in a managed page table for **sharing and eviction**"* — on the grounds that the allocator never evicts. **I verified before escalating, and eviction is real:**

```rust
// pipeline/paged_decode.rs:44   — called from flat_autoregressive.rs:307
pub(crate) fn evict_until_free(&mut self, wanted_pages: usize) {
    ...
    self.prefix.evict_lru(wanted_pages - free, &mut self.cache.page_table);
}
```
LRU prefix eviction, wired, with a doc comment explaining the exact failure it prevents. **My evidence was `ByteBudget::reconfigure` never evicting — a different subsystem on a different axis (a VRAM-budget knob, not the page pool).** I generalised from one component to the whole allocator and carried it for hours.

> **I ALMOST TALKED THE CREW INTO WEAKENING A TRUE CLAIM.** Every other correction today made a statement *more* conservative; this one would have made the demo's headline sentence **less accurate in the name of honesty.** An honesty process that only ever ratchets toward understating is not calibrated — **it is just a different bias, and it is harder to notice because each individual step feels virtuous.** D99.

**AC43 stands verbatim. Nobody edits that sentence.**

### 35.2 🔴 AC39 CARRIES A STALE VERIFICATION THAT IS NOW FALSE AT HEAD

AC39 reads: *"Verified today: the current binary returns **404 on `/demo`**, zero `access-control-*` headers, and **405** to an `OPTIONS` preflight."*

**`GET /demo` has since SHIPPED** (`demo_assets.rs`, confirmed by @e00032a4). That sentence was true when written and is **false now**, but it is phrased in the present tense as a live verification. **A reconstruction is exactly where a stale fact gets laundered into a current one** — it survives the rewrite as prose while the world moves. And AC39 is on the never-cut list (*"without it nothing runs in a browser"*).

**Also: AC39 asserts `GET /demo` byte-exact three times, but the Lead ruled the URL is `/demo/`** — `/demo` is a temporary redirect (`lib.rs:82-84`), and without the trailing slash relative module imports resolve against `/` and every `<script type="module">` 404s, presenting as a blank page. **@376a0297: re-date that verification and use `/demo/` wherever the URL is asserted exactly.**

### 35.3 ⚠️ THE SPEC CONTAINS AC1–AC61; THE ANNOUNCEMENT SAID 46

`grep` reports 61 contiguous ACs against an announced *"46 ACs, AC1–AC46 contiguous."* Probably continued appending after the announcement — but **anyone binding to the count (a reviewer checking coverage, a QA plan asserting completeness) is working from 46.** Worth one line of confirmation.

| # | Decision | Rationale |
|---|---|---|
| D99 | An honesty process that only ratchets toward understating is **not calibrated, it is inversely biased** | Each understating step feels virtuous, so nothing flags the drift. Verify before weakening a claim, exactly as before strengthening one |
| D100 | AC43 stands **verbatim** — eviction verified at `paged_decode.rs:44` ← `flat_autoregressive.rs:307` | LRU prefix eviction is wired and reachable |

---

## 36. THREE LIVE CONTRADICTIONS BETWEEN RATIFIED ARTIFACTS — the spec disagrees with the ruling, and with ITSELF

Verified on disk, not remembered. All three change what somebody builds in the next few minutes.

### 36.1 🔴 THE SPEC MANDATES `ok`; THE RULING AND THE CODE SAY `measured` — AND THE SPEC CONTRADICTS ITSELF ABOUT IT

`demo-spec.md:185` rules **`ok`, NOT `measured`** — carrying the orthogonality rationale (*"a field whose state read `measured` while its source badge read `estimated` would contradict itself"*). **That argument is MINE, and it was overturned.** @12e42da8 yielded and I reversed my own D65 in §32. The final enum is `measured`.

**And the spec's own AC49, 189 lines later, treats `ok` as the STALE value:**
> *"a module built against the older shape passing `state: 'ok'` renders a **live-looking number**."*

**So the spec simultaneously MANDATES `ok` in §3.1 and cites `ok` as the canonical example of the enum-migration bug in AC49.** A dev binding to §3.1 builds precisely the defect AC49 exists to prevent.

**This is the reconstruction hazard doing exactly what we predicted, one level up.** The spec is a *record* of rulings; a record rebuilt from a ruling that has since been reversed **re-enters the document wearing the reconstruction's fresh timestamp** and outranks the newer decision by looking more authoritative. **@376a0297 — §3.1 must say `measured`, and the retired rationale should be kept with an explicit "superseded, and here is why" rather than deleted**, or someone re-derives it in three weeks exactly as I did.

### 36.2 🔴 DOES THE BLOCK GRID HAVE PER-CELL OWNERSHIP? TWO RULINGS SAY OPPOSITE THINGS

- **@376a0297:** per-cell ownership *"isn't reachable from the server crate"* — `SequenceUsage` consumes the `Vec<PageId>` into a length (`page_table.rs:867-875`). AC15 is caveated to a degraded form, **and the hover-linked swimlane↔KV highlight is declared to fall with it.**
- **@12e42da8 (QA plan §0/B3, APPROVED AND IN FLIGHT):** @d7cf9b84 owns an accessor returning `Vec<PageBlock { id, ref_count, filled_slots, tier, **owner** }>`.

**`owner` is exactly the field the PM says is unreachable.** Both are right about their own premise — the PM read *today's* server crate; the Lead approved *an engine change that adds it.* **The contradiction is a TENSE problem, not a facts problem**, and tense is invisible in a spec written in the present.

**It is load-bearing for me in two places, so I need it resolved, not averaged:** §25's **sharing-by-selection** (select a sequence, its blocks light up) requires `owner` **and** a stable `id` across frames; without both it degrades to an anonymous fill meter. **D101: I design to the DEGRADED form and treat `owner` as an enhancement that lights up when the engine change lands.** A grid that is honest and good without ownership, and better with it, is the only version that can't be wrong — and it means **a slip in @d7cf9b84's engine work costs a feature, not a panel.**

### 36.3 GRID CADENCE — the PM's 1 Hz and my event-sampling agree; take the stricter one

@376a0297 ruled **1 Hz, not 4 Hz** (4096 cells × 4 Hz ≈ 1 MB/s of JSON straight into AC33's budget). §27/D78 already ruled the grid **event-sampled** rather than polled at all. **These aren't in conflict — event-sampling is strictly cheaper than 1 Hz, and their justification is the better articulation of why:**

> **Gauges and sparklines need 4 Hz because the eye reads them as MOTION. The block grid is read as a STATE YOU INSPECT, not a flow you watch.**

**D102: event-sampled is canonical; 1 Hz is the fallback ceiling if event hooks aren't available. Never 4 Hz.** And their cut-order call is right and I'm adopting it verbatim: **if cadence must be traded against fidelity, keep the blocks and slow the refresh.**

### 36.4 ✅ Confirmed, no action: no `preempted` lane

@d7cf9b84 asks for no preemption lane. **Already ruled in §28/D83 — the swimlane is 4 states with no reserved lane**, decided when their first blocker landed. Their four-independent-blockers finding doesn't change the design, it just makes it unarguable. **A reserved-but-never-filled lane is a fabricated zero in layout form** — it occupies space, implies a category, and reads as *"this never happens"* rather than *"this cannot happen."*

| # | Decision | Rationale |
|---|---|---|
| D101 | Design the block grid to the **degraded** (no-`owner`) form; ownership is an enhancement | A slip in the engine change then costs a feature, not a panel |
| D102 | Grid is **event-sampled**; 1 Hz fallback ceiling; never 4 Hz | The grid is a state you inspect, not a flow you watch |
| D103 | A reserved-but-unfillable lane is **a fabricated zero in layout form** | It implies a category and reads as "never happens", not "cannot happen" |

---

## 37. 🔴 THE NAMING LIE IS A **DESIGN** DEFECT, AND OUR ENVELOPE CANNOT CATCH IT BY CONSTRUCTION

@12e42da8 has now named six fields whose **values are correct and whose names lie**. This is the one failure class the Field envelope gives **no signal on**, and I want to be precise about why, because it isn't an oversight — **it's structural.**

### 37.1 Why every check passes

Take `prefix_cache_hit_rate` on the dynamic profile:
```js
{ value: 0.5, state: 'measured', source: 'server', endpoint: '/v1/debug/kv', server: 'dynamic' }
```
**Every field in that envelope is TRUE.** It really was measured. It really came from the server. The endpoint really is curl-able. `numericValueOf` returns a number because there genuinely is one. **A reviewer checking provenance end-to-end finds nothing wrong, because nothing IS wrong — with the provenance.**

> **PROVENANCE ANSWERS *WHERE DID THIS NUMBER COME FROM*. IT NEVER ANSWERS *WHAT IS THIS NUMBER*. We built an elaborate apparatus for the first question and none at all for the second — and the second is the one a visitor actually reads off the screen.**

**And the failure is strictly worse than a stub, for a reason worth stating plainly: a stub is greppable and inert; a misnamed field is LIVE. It moves when you exercise the feature, which is the strongest possible confirmation signal a developer can get.** Watching the number respond to your actions *feels* like verification. It is the opposite.

### 37.2 The defence is the LABEL, and the label is mine

There is exactly one place this can be caught: **the words on screen.** So the rules, all cheap:

- **D104 — `label` IS MANDATORY, AUTHORED BY US, AND NEVER DERIVED FROM THE WIRE FIELD NAME.** No `titleCase(fieldName)`, no prettifier, no fallback to the key. **An automatic label launders the upstream name into a claim.** A missing label is an authoring bug and must **throw**, exactly like an unknown state.
- **D105 — TRIPWIRE: no rendered UI string may equal or normalise to a server field identifier.** A test asserting `label !== fieldName` for every bound field. That's the same pattern @c8d9a40e used on `tokens_per_second`, generalised: **when you remove a trap, leave a tripwire.**
- **D106 — WHERE THE NAME AND THE QUANTITY DIVERGE, THE HOVER STATES WHAT THE COUNTER ACTUALLY COUNTS**, in one sentence, in the visitor's words. Not the field name. Not the file:line — that's for `unavailable`.

### 37.3 THE HONEST LABEL TABLE — my copy for all six, ready to paste

| Wire field | What it actually counts | ❌ Never render | ✅ Render |
|---|---|---|---|
| `prefix_cache_lookups` | completed generations, no predicate | "Cache lookups" | **"Generations completed"** |
| `prefix_cache_hit_rate` | hits ÷ generations | "Cache hit rate" | **"Generations with a prefix hit"** |
| `active_sessions` | persistent `X-Session-Id` sessions | "Active requests" | **"Named sessions"** |
| `vram.used` | KV byte-budget accounting | "VRAM used" | **"KV budget in use"** |
| `host_ram.used` | whole-machine RAM | "Memory used" | **"System RAM (whole machine)"** |
| `batch_size_current` | in-flight HTTP requests | "Batch size" | **"Requests in flight"** |

**Note what the ✅ column has in common: every one is LONGER and LESS PUNCHY than the lie.** "Batch size" is a better piece of UI copy than "Requests in flight" by every conventional measure — shorter, familiar, scannable. **That is exactly why it wins arguments, and exactly why this class of defect will keep recurring after tonight: the dishonest label is always the more elegant one.** D107: **when a label is being shortened, that is the moment to re-check it against the counter.** Concision is the pressure that produces these.

### 37.4 ✅ Block-grid ownership: resolved, no rework

@e00032a4 confirms `PageUsage` consumes `Vec<PageId>` into a length (`page_table.rs:867-875`) and that per-block ownership needs the return type widened in `onnx-genai-kv`. **That resolves §36.2 in favour of the degraded form — which D101 already designed to**, so @c8d9a40e builds ownership as a nullable input and nothing is wasted either way. Their framing matches mine exactly: **present → colour by sequence; absent → honest fill with ownership `unavailable`.**

| # | Decision | Rationale |
|---|---|---|
| D104 | `label` mandatory, authored, **never** derived from the field name; missing label throws | An automatic label launders the upstream name into a claim |
| D105 | Tripwire test: no rendered string equals a server field identifier | Remove a trap, leave a tripwire |
| D106 | Where name ≠ quantity, the hover says what the counter actually counts | The only place the second question can be answered |
| D107 | **Shortening a label is the trigger to re-verify it** | The dishonest label is reliably the more elegant one — concision is the pressure that creates this defect |

---

## 38. 🔴 SCENARIO B — THE MECHANISM BEHIND QA'S RED RESULT. `prefix_cache_hit_len` ON DYNAMIC MEANS "TOKENS IN COMMON", NOT "PREFILL SKIPPED"

@fc8b5d97 measured Scenario B red: shared-prefix warm requests **+7.0% SLOWER** than completely-unshared controls, with a sensitivity control proving a real effect would have been a ~90% TTFT collapse — **proven absent, not merely unobserved** — while the hit counter read **95%**. I traced the mechanism, because "cut it" and "re-scope it" are different decisions and only the mechanism distinguishes them.

### 38.1 There are TWO prefix branches and only one of them reuses anything

`engine/runtime.rs:997` `prepare_session_prefix` — **taken by both the FCFS path (`:1209`) and the callback path (`:435`)**, i.e. both server entry points:

```rust
if started_empty && state.decode_state.uses_token_prefix_cache() {
    cross_session_hit_len = self.token_prefix_cache.iter()
        .map(|cached| common_prefix_len(cached, prompt_tokens).min(cached.len()))
        .filter(|&len| len > 0).max().unwrap_or(0);          // ← BRANCH 1
} else if started_empty && state.decode_state.use_kv && self.kv_model.is_some()
          && self.kv_cache.page_table.tensor_config.is_some() {
    let matched = self.prefix_cache.lookup_shared(...);       // ← BRANCH 2
    // ...materializes pages, genuinely skips prefill
}
```

**BRANCH 1 COMPUTES A STRING OVERLAP AND RESTORES NOTHING.** No page table access, no KV materialization, **no prefill skipped**. It sets `cross_session_hit_len` — which becomes `prefix_cache_hit_len`, which feeds the metric — **purely from `common_prefix_len`.** It is a *measurement of textual similarity being reported as a cache hit.*

**And branch 1 WINS FIRST:** `uses_token_prefix_cache()` is `has_runner() || is_windowed()` (`decode/state.rs:206-208`). Any model using a decode runner takes branch 1 and **never evaluates branch 2's conditions at all.**

### 38.2 This explains every number QA measured, exactly

| QA observation | Mechanism |
|---|---|
| hit rate pins ~95–100% from the first request | any **nonzero** overlap counts; the chat template shares the first few tokens |
| ARM B controls counted as hits despite differing from token 0 | they still share the template prefix |
| TTFT unchanged (+7%) | **branch 1 skips no prefill — there is nothing to speed up** |
| the ~90% collapse never appears | branch 2, the only branch that materializes pages, is never reached |

**QA proved the absence behaviourally; this is the same finding at file:line.** Two independent methods, same conclusion — which is the standard we've been holding everyone to.

### 38.3 D108 — THE SEVENTH MISNAMED FIELD, AND THE MOST CONSEQUENTIAL

> **`prefix_cache_hit_len` ON THE DYNAMIC SERVER MEANS "THE LONGEST TOKEN OVERLAP WITH ANY CACHED PROMPT". IT DOES NOT MEAN "PREFILL WORK WAS SKIPPED".**

It is a **real, honestly-computed, live, responsive number** measuring a quantity nobody wants. It moves when you exercise the feature. It is exactly §37's class — and note it defeated the strongest thing we had: **the scatter zero was suspicious enough to investigate, so the FABRICATED value got caught while the LIVE one nearly shipped as our headline.** Suspicion tracks implausibility, not falsehood.

### 38.4 What I recommend for Scenario B — re-scope, don't cut, and don't dress it up

**Cutting loses a real finding; showcasing "prefix caching is broken" makes a product claim we haven't earned** (branch 2 is real, tested by `prefix_speedup.rs`, and reachable — just not from either server path as configured). Both extremes are wrong.

**D109 — SCENARIO B BECOMES THE COLD-vs-WARM TTFT PAIR, HONESTLY LABELLED, WITH NO HIT-RATE FIELD ANYWHERE.** We show two measured client-side TTFTs and state plainly that on this execution path the second request is **not** faster. The panel's `prefix_cache_*` fields render **`not-applicable`** with the on-screen caption naming branch 1.

**D110 — SCENARIO B IS DEMOTED FROM HEADLINE.** Scenario A (batching, 2.46× aggregate, measured) and Scenario C (paged-KV block grid + admission backpressure) carry the demo. **A negative result is worth showing and is not worth leading with** — leading with it invites "so your prefix cache doesn't work," which is **a stronger claim than our evidence supports** and it isn't the story we set out to tell. **@376a0297/@12e42da8 own this call; it is a product decision and I'm recommending, not ruling.**

| # | Decision | Rationale |
|---|---|---|
| D108 | `prefix_cache_hit_len` on dynamic = longest token overlap, **not** prefill skipped | `runtime.rs:1016-1023`, branch 1 restores nothing |
| D109 | Scenario B = honest cold/warm TTFT pair; **no hit-rate field in any form** | Both counters are untrustworthy in opposite directions |
| D110 | Scenario B **demoted from headline**; recommend, PM/Lead rule | A negative result is worth showing, not worth leading with |
| D111 | **Suspicion tracks implausibility, not falsehood** — so a live plausible lie outranks a fabricated zero for danger | The zero got investigated; the 95% nearly shipped |

---

## 39. 🛑 STOP THE EVICTION EDIT — THE ALLOCATOR **DOES** EVICT. TWO MECHANISMS, BOTH TESTED.

@12e42da8 has instructed @376a0297 to remove "eviction" from the AC43 honesty string, citing my framing. **I RETRACTED THAT OBJECTION IN §35 (D100) TWENTY MINUTES AGO AFTER VERIFYING IT WAS WRONG.** The correction did not propagate. Re-verified now, harder:

**MECHANISM 1 — prefix LRU eviction, releases pages back to the pool:**
```
onnx-genai-kv/src/prefix_cache.rs:151   pub fn evict_lru(&mut self, target_pages, page_table) -> Vec<PageId>
  ← engine/pipeline/paged_decode.rs:53  evict_until_free()
  ← engine/pipeline/flat_autoregressive.rs:307
  tests: prefix_cache.rs:329, :351, :356, :357
```
**MECHANISM 2 — hot-tier LRU demotion, GPU → CPU:**
```
onnx-genai-kv/src/page_table.rs:1068    pub fn evict_lru_hot(&mut self, exclude) -> Result<PageId, KvError>
  selects min_by_key(last_access), sets victim.device = Device::Cpu,
  increments self.stats.hot_evictions
  ← page_table.rs:913, :1055
```

**The Lead's evidence — `eviction_order` / `overage_bytes` in `governor.rs` having zero consumers — is TRUE and is about a DIFFERENT SUBSYSTEM: the VRAM byte-budget governor behind `ByteBudget::reconfigure`.** That governor really does compute a plan and discard it. **The paged KV allocator, which is what AC43's sentence is about, evicts by LRU in two places with tests.** AC43 stands verbatim.

### 39.1 The lesson, and it's about how this crew communicates rather than about eviction

I made this exact conflation, held it for hours, verified it, and retracted it in a broadcast. **What propagated was not my finding and not my retraction — it was my PHRASE**, *"an overclaim inside our own honesty disclaimer is the most expensive error available on the page."* The Lead cited that framing while acting on the claim it was originally attached to.

> **D112 — A MEMORABLE FRAMING OUTRUNS THE FACT IT WAS ATTACHED TO, AND OUTLIVES ITS RETRACTION.** The better the sentence, the further it travels beyond the evidence. **When you retract something you said well, retract the SENTENCE by name, not just the conclusion** — because the sentence is what people kept.

This is the counterpart to D99. **D99: an honesty process that only ratchets toward understating is inversely biased. D112: the rhetoric that makes a correction persuasive is exactly what makes it survive being wrong.** Both failures point the same way — **toward confidently removing true claims** — and neither is caught by any check we have, because every check we built looks for fabricated *additions*.

### 39.2 What must NOT happen

**@376a0297 — do not edit the AC43 string. @e00032a4 — do not weaken ARCHITECTURE.md's eviction language.** If it had gone through: the demo's single most-scrutinised sentence would have been edited to **understate a capability the runtime genuinely has**, in a paragraph whose entire purpose is precision, **and we'd have called it an honesty fix.**

| # | Decision | Rationale |
|---|---|---|
| D112 | Retract the **sentence by name**, not just the conclusion | The phrase is what propagated; it outlived the fact and the retraction |
| D113 | AC43 "sharing and eviction" is **verified true**; the string is frozen | `prefix_cache.rs:151` and `page_table.rs:1068`, both wired, both tested |

---

## 40. `MEASURED: 'ok'` IS A LIVE BUG · `--max-batch` EXISTS · AND OCCUPANCY MUST NEVER BE A PERCENTAGE

### 40.1 🔴 CONFIRMED — the constant's NAME and its VALUE disagree

`telemetry-field.js:89-95` on disk:
```js
export const FIELD_STATES = Object.freeze({
  MEASURED: 'ok',            // ← name says measured, wire value says ok
  PENDING: 'pending', STALE: 'stale',
  UNAVAILABLE: 'unavailable', NOT_APPLICABLE: 'not-applicable',
});
```
@c0de4c2e found this and it's mine to settle, because I'm the one who conceded `ok` and then reversed. **The ratified wire value is `measured`.** Today `field.state === 'measured'` is `false` for **every measured field** — and because `formatFieldText` falls through, that field then **renders as a plain number anyway.** *The check fails silently while the output looks correct*, which is the worst pairing available: **a broken guard that produces a healthy-looking screen.**

**D114: `MEASURED: 'measured'`.** One line. And **`:160-163` still emits `sourceClass` and `origin`** — both retired; three keys, `source`/`endpoint`/`server`.

**Note the shape of how it got here:** the name was written when the value was right, the value changed, and the name kept vouching for it. **A constant's name is documentation with no test coverage** — which is the same defect class as `:20`'s stale JSDoc sitting directly above the constant it describes. @c0de4c2e's line on that is the best statement of it: **a doc comment inherits the authority of code while carrying none of the guarantees.**

### 40.2 ✅ REVERSED: `--max-batch` DOES exist — but the denominator still shouldn't be used as one

@c0de4c2e reports the flag is absent. **It landed:** `cli.rs:77 pub max_batch: usize`, plumbed at `:127`; `DEFAULT_MAX_BATCH = 4` at `state.rs:28`. So §5.3's missing denominator is **available**.

**I am still ruling against a percentage, and now for a better reason than not having one.**

### 40.3 🔴 D115 — BATCH OCCUPANCY RENDERS AS `3 of 4`, NEVER `75%`

With `max_batch = 4`, the occupancy metric has exactly **five reachable values**: 0, 1, 2, 3, 4. Rendering that as a percentage produces `0% · 25% · 50% · 75% · 100%` — a scale whose **form implies 101 possible values when only 5 exist.**

> **A PERCENTAGE OVER A SMALL INTEGER DENOMINATOR FABRICATES RESOLUTION. The number is correct; the PRECISION IS INVENTED.**

This is D85's principle — *type size is a claim about which number matters* — one level down: **the FORMAT is a claim about how finely the quantity can be known.** `75%` invites a visitor to expect `76%`, and to read a jump from 50% to 75% as a smooth 25-point move rather than **one sequence entering the batch**.

`3 of 4` is strictly better on every axis: it shows the numerator **and** the denominator (so `max_batch` never has to be hunted for), it makes the granularity self-evident, it can't imply absent precision, and **it stays honest if `--max-batch` changes** — `75%` silently means something different at `max_batch=8`. It also satisfies the Lead's *"a ratio invents a numerator — name both terms"* rule **by construction, because both terms are on screen.**

**D116: this generalises — any ratio whose denominator is a small integer renders as `n of m`.** Batch occupancy, decode rows in use, sessions against a cap. **Percentages are for quantities with genuine continuum**, i.e. the block grid (~14,612 pages), never for counts of four.

| # | Decision | Rationale |
|---|---|---|
| D114 | `MEASURED: 'measured'`; drop `sourceClass`/`origin` for `source`/`endpoint`/`server` | Name and wire value disagree; the guard fails silently while output looks right |
| D115 | Batch occupancy renders **`3 of 4`**, never a percentage | Five reachable values presented on a 101-point scale invents precision |
| D116 | **Any ratio over a small integer denominator renders `n of m`** | The format is a claim about how finely a quantity can be known |

---

## 41. 🔴 THE ENUM FORK DISSOLVES — THE SAFETY PROPERTY LIVES IN THE CONSTANT NAME, NOT THE WIRE STRING. D114 PARTIALLY REVERSED.

The enum has genuinely forked and **I caused most of it** by conceding `ok`, reversing to `measured`, then ruling `MEASURED: 'measured'`. Current truth on disk:

| Artifact | Says |
|---|---|
| `demo-spec.md` (@376a0297) | five states, **`ok`** |
| `telemetry-field.js:90` (@bb2ee824, `baf18736`, 52/52 green) | **`MEASURED: 'ok'`** |
| Lead's last group ruling | **`measured`** |

### 41.1 The migration is NOT one line — I measured it

```
dashboard/prefix-cache.js:227    state: 'ok',            ← raw literal
dashboard/field-state.js:46      OK: 'ok',
dashboard/field-state.js:28      @typedef {'ok'|'pending'|'stale'|'unavailable'}   ← FOUR states
dashboard/store-adapter.js:194   return { state: 'ok', ... }
```
**@bb2ee824's scope is clean (constants only); `dashboard/` carries raw string literals.** So a rename lands in the scope that uses raw strings, **during an active build, against a known silent fall-through** — and a missed literal doesn't throw, it **renders the value as a plain number.** That's a real risk for a word change.

### 41.2 The dissolution: the Lead's safety argument is already satisfied

The argument for `measured` was: *a reviewer seeing `state: 'ok', value: 0` reads "ok" as "fine" and is tempted to hide the zero.* **That is a claim about what a DEVELOPER READS.** And a developer reading or writing this never types the wire string — they type:

```js
FIELD_STATES.MEASURED       // ← the name is ALREADY 'MEASURED'
```

> **THE SAFETY PROPERTY LIVES IN THE IDENTIFIER DEVELOPERS TYPE, NOT IN THE WIRE STRING THEY NEVER SEE. `MEASURED: 'ok'` DELIVERS BOTH — the honest name in the code, the shipped value on the wire, and ZERO MIGRATION.**

**So `MEASURED: 'ok'` is not a bug. It is the correct design, and I mis-ruled it in D114** because I read the mismatch against a spec value rather than asking *who ever reads this string.* @c0de4c2e was right that name and value disagreed; **the fix is to ratify the value, not to churn it.**

### 41.3 D117 — WHAT MAKES IT SAFE: NO MODULE MAY USE A RAW STATE LITERAL

The mismatch is only dangerous if someone writes `field.state === 'measured'` (or `'ok'`) by hand. **Ban it and both risks vanish at once — this fork AND the leak class @bb2ee824 flagged at `prefix-cache.js:266`.**

- **D117:** every state comparison goes through `FIELD_STATES.*` or a helper (`hasValue`, `numericValueOf`). **A test asserts no module contains a raw state string.** Same tripwire pattern as `tokens_per_second`.
- **D118:** `dashboard/field-state.js:28`'s `RenderState` typedef lists **four** states — **add `not-applicable`.** A stale typedef is the `:20` JSDoc failure again: **documentation with the authority of code and none of its guarantees.**

**MY RECOMMENDATION TO @12e42da8, and it costs nothing to overrule: RATIFY `ok` AS THE WIRE VALUE.** Spec and code already agree; your safety property is delivered by the constant name; the migration risk lands in the scope least able to absorb it. **I argued for `measured` and I'm withdrawing that — the ambiguity window is now more expensive than either word, and this is the option that closes it in zero edits.**

### 41.4 What I got wrong, plainly

I reversed on this enum three times. **My §21 four-state line, my `ok` concession, my `measured` reversal, and D114's rename each looked locally correct and collectively produced a fork that cost two devs real time.** The lesson isn't "rule earlier" — evidence genuinely kept arriving. It's narrower:

> **D119 — WHEN YOU HAVE REVERSED YOURSELF TWICE ON THE SAME QUESTION, YOU ARE NO LONGER THE RIGHT PERSON TO RULE ON IT. HAND IT TO THE OWNER WITH THE EVIDENCE AND STOP PUBLISHING.** Each of my reversals was defensible; the *sequence* was the damage, and the sequence is invisible from inside it.

| # | Decision | Rationale |
|---|---|---|
| D117 | No raw state literals anywhere; comparisons via `FIELD_STATES.*`/helpers, enforced by test | Kills the fork risk and the leak class together |
| D118 | `field-state.js:28` typedef must list five states | A stale typedef is code-authority without code-guarantees |
| D119 | **Two reversals on one question = hand it to the owner and stop ruling** | Each reversal was locally defensible; the sequence was the damage |

---

## 42. §41 IS RETRACTED ON EVIDENCE, NOT AUTHORITY — THE STYLESHEET ALREADY REQUIRES `measured`. PLUS: THE FIND-REPLACE TRAP THAT WOULD BREAK HEALTH DETECTION.

@12e42da8 re-ruled `measured` minutes after I published §41 recommending `ok`. **I am not re-arguing — but the reason I was wrong is concrete and worth more than the compliance.**

### 42.1 The CSS contract already says `measured`, and has all along

```
css/shell.css:154   [data-state='measured'] { color: var(--og-fg); }
```
Under a header reading **"THE MOST IMPORTANT RULES IN THIS FILE."** Meanwhile `panel-kit.test.js:39` asserts the DOM emits `data-state="ok"`.

> **THAT SELECTOR MATCHES NOTHING TODAY.**

**And it is invisible for the worst possible reason: `measured` is styled as `color: var(--og-fg)` — the inherited default — so a rule that matches nothing looks exactly like a rule that matches everything.** The one state whose treatment is *the absence of a treatment* is the one state whose selector can rot undetected. My own §21 note ("`measured` is not a treatment, it is the ABSENCE of one") is what made this undetectable.

**So `measured` is not the Lead's preference winning a tie — it is what the stylesheet on disk has required from the beginning.** §41's core claim, that `ok` costs zero edits, was **false**: it costs a silent CSS mismatch in the file that guards the whole honesty system. **I checked the JS and the spec and never checked the CSS — my own layer.** D119 said hand it to the owner after two reversals; I should have handed it over *and gone to look at my own files*, which is where the answer was.

### 42.2 🔴 D120 — THE MIGRATION MUST NOT BE A FIND-REPLACE. `'ok'` MEANS TWO UNRELATED THINGS.

~50 raw `'ok'` literals are on disk. **They are not all field states, and a global replace corrupts a subsystem that has nothing to do with telemetry:**

```
telemetry-store.test.js:148   { status: 'ok', model: 'qwen-scatter' }   ← HTTP HEALTH PAYLOAD
origin-integration.test.js:33 '/health': { status: 'ok' }               ← HTTP HEALTH PAYLOAD
shell.css:221                 .connection-indicator[data-state='connected']  ← DIFFERENT ENUM
```
**`status: 'ok'` is the server's health response. Renaming it to `measured` breaks connection detection — and it breaks it INTO the "unreachable" state, so the page would render as a dead server while the server is fine.** The connection indicator is a *third* `data-state` vocabulary (`connected`/`connecting`/`no-model`/`unreachable`) that must not be touched either.

- **D120:** the rename is **`FIELD_STATES.MEASURED: 'ok' → 'measured'` plus field-state call sites ONLY.** Never a project-wide replace of the string `'ok'`. Three distinct vocabularies share that token.
- **D121:** `dashboard/state-vocabulary.test.js:28` hardcodes `RULED_STATES = ['ok', …]` — **it pins the wrong value and will fail loudly. Good.** That test is the migration's tripwire; update it *last*, so it stays red until every call site is done.
- **D122 (supersedes §41's D117 framing, keeps its substance):** after the rename, ban raw state literals — comparisons via `FIELD_STATES.*`/`hasValue()`. `store-adapter.js:410` currently reads `field?.state !== 'ok' && field?.state !== 'measured'` — **someone already wrote defensive code accepting both.** That line is the fork made visible; it should end up as one constant.

### 42.3 AC15 — @d7cf9b84's question answered directly: I would REPLACE the grid, not degrade it

*"If the ~10-line engine accessor slips, is grid-minus-per-cell-ownership a degraded panel you'd still ship, or one you'd rather replace?"* — **Replace it.**

**Without per-cell `refs`/`seqs`, every cell carries exactly one bit: allocated or free. A grid whose cells encode a single scalar is a progress bar rendered as 400 squares** — strictly worse than a progress bar, because the grid *form* promises per-cell identity and invites the viewer to hunt for structure that does not exist. That is the visual equivalent of a fabricated zero: **a container implying more information than it holds.**

- **D123:** if ownership data is unavailable, **the block grid is replaced by a single occupancy bar plus the numeric fields** — not thinned, not greyed. And **the panel is renamed: "KV cache occupancy", NEVER "Paged KV block table."** *Block table* promises blocks with identity; **the sharing story — two sequences pointing at one physical block — IS the pedagogical content of pillar 2, and a grid that cannot show it is a title making a claim its pixels cannot support.** @e00032a4's finding that `page_table.rs:617 pub sequences` is already public makes this contingency unlikely, and I'd spend the ~10 lines rather than take the fallback.

### 42.4 D124 — `created` IS A CLOCK. And it defeats my honesty treatments BY CONSTRUCTION.

@376a0297/@d7cf9b84's `created: now_unix()` (`routes/admin.rs:29`) is the sharpest finding of the session against *my* layer specifically, and I want to name why:

> **EVERY VISUAL HONESTY MECHANISM I DESIGNED USES ABSENCE OF MOTION AS THE CUE FOR DOUBT.** `stale`'s dashed underline, the age suffix, AC20's *no metric that cannot move* — **all of them treat a frozen number as the suspicious one. A fabrication that ticks is invisible to every one of them, and reads as MORE trustworthy than a real value that happens to be constant.**

- **D124:** the model card **must not display `created`**, and more generally **no absolute server-supplied timestamp is ever rendered.** Times are shown only as **ages relative to an event the client itself observed** (`observedAtMs`), because that is the only clock whose provenance we can state. A server timestamp has no field state that describes it honestly — it is neither `measured` nor `unavailable`; it is *plausible*, which is the one thing our vocabulary has no word for.

| # | Decision | Rationale |
|---|---|---|
| D120 | Rename `FIELD_STATES.MEASURED` value + field-state call sites ONLY; never a global `'ok'` replace | `status:'ok'` is the health payload; renaming it fakes an unreachable server |
| D121 | `state-vocabulary.test.js:28` is the migration tripwire; update it LAST | Keeps a loud red signal until every call site is converted |
| D122 | Ban raw state literals post-rename; collapse `store-adapter.js:410`'s dual check | Defensive both-values code is the fork made visible |
| D123 | No ownership data ⇒ occupancy bar, and rename to "KV cache occupancy" | A grid encoding one bit per cell promises structure it doesn't have |
| D124 | Never render absolute server timestamps; only client-observed ages | A fabrication that MOVES defeats every motion-based honesty cue |

---

## 43. AC CITATION AUDIT COMPLETE — ALL 34 RESOLVE. BUT THE SPEC IS 78 ACs, NOT 46 OR 50, AND TWO OF MY OWN RULINGS NOW CONTRADICT IT.

@12e42da8 asked every artifact owner to open `demo-spec.md` and confirm cited ACs still say what we built against. Done — mechanically, not by eye.

### 43.1 Result: zero dangling citations

```
DEFINED:  78 ACs, AC1..AC78, contiguous, ZERO gaps
CITED by demo-ux.md:  34 ACs
ACs I cite that are NOT defined:  (none)
```
**Every AC number in my contract resolves.** Method — `grep -oE '^- \[[ x]\] \*\*AC[0-9]+'` for definitions vs all `AC[0-9]+` tokens in mine, `comm`'d. Anyone can re-run it; **the first regex I tried reported 6 definitions and I nearly published that — the definition format is `- [ ] **ACnn**`, and a citation audit that silently under-matches produces a clean bill of health, which is the worst possible failure for an audit.**

### 43.2 🔴 THE COUNT IS WRONG IN BOTH DIRECTIONS, ASSERTED MINUTES APART

- @12e42da8: *"IT IS 46 ACs — AC1–AC46 contiguous, no duplicates."*
- @376a0297: *"Spec: 50 ACs, contiguous, backup current."*
- **The file: AC1–AC78, contiguous, no gaps.**

Both were stated *while instructing everyone else to open the file*. **This isn't a nit: the Lead's own standing rule is NEVER CITE AN AC FROM MEMORY, and the count is itself a citation.** More practically — **anyone who reads "46 ACs" and finds their AC number above 46 will conclude their artifact is stale and start editing.** AC48, AC57, AC61 and AC76 are all live and all cited in binding rulings from tonight.

### 43.3 ✅ AC43 SURVIVED INTACT — my §39 stop worked

AC43 now carries my copy substantially verbatim and still reads **"a managed page table for sharing and eviction."** The instruction to strike "eviction" did not land. **Closing that open item — and noting I nearly caused real damage by attaching a memorable framing to a claim I later retracted (D112).**

### 43.4 🔴 MY D123 CONTRADICTS AC15. THE SPEC IS THE OLDER TEXT; I DEFER, WITH ONE AMENDMENT.

AC15's degraded form mandates *"capacity, in-use, free, shared count, refcount histogram, and per-sequence block counts — but no per-cell ownership and no stable block identity."* **That is still a GRID.** Twenty minutes ago I ruled (D123) that the ownership-less fallback should be **replaced** by an occupancy bar, because a grid whose cells each carry one bit is a progress bar drawn as 400 squares.

**I'm not overriding a ratified AC from my own doc.** But the amendment is small and I'd ask @376a0297 to take it:

> **D125 — AC15's degraded form is accepted, with the naming clause: if per-cell ownership is cut, the panel MUST be retitled from "Paged KV block table" to "KV cache occupancy."** The listed fields (capacity, in-use, free, refcount histogram) are genuinely useful and I withdraw the objection to shipping them. **What I will not withdraw is the title.** *Block table* promises blocks with identity; **the sharing story — two sequences pointing at one physical block — is the entire pedagogical payload of pillar 2, and a title claiming it while the pixels cannot show it is the fabrication problem in typography rather than in numbers.** A refcount histogram *does* preserve the sharing story in aggregate, which is why the panel is worth shipping — it just isn't a table of blocks.

### 43.5 🔴 D126 — AC20 IS DEFEATED BY `created`, AND THE FIX IS ONE CLAUSE

AC20 reads: **"No metric is displayed that cannot move."** `created: now_unix()` (`routes/admin.rs:29`) **moves on every single call** and is pure fabrication. It passes AC20 perfectly.

- **D126:** AC20 needs its inverse — **"and no metric is displayed that moves for a reason unrelated to the thing it names. Ask what would have to happen IN THE ENGINE for this number to change; if the answer is 'nothing,' it is a clock."** As written, AC20 tests for one failure mode and **silently certifies its mirror image.** @376a0297's §3.2 rule already says this; **it needs to be in AC20 itself, because AC20 is the one people will grep for.**

| # | Decision | Rationale |
|---|---|---|
| D125 | AC15's degraded grid accepted; **the retitle to "KV cache occupancy" is not** negotiable | A title claiming block identity the pixels can't show is fabrication in typography |
| D126 | AC20 gains the inverse clause: motion unrelated to the named quantity | AC20 currently certifies `created: now_unix()` as compliant |

---

## 44. 🚨 THE FIVE-STATE ENUM HAS A HOLE, AND `prefix_cache_hits` IS SITTING IN IT. NO SIXTH STATE — AN UNBINDABLE-FIELD REGISTRY INSTEAD. (D127–D130)

@fc8b5d97's §7 addendum is the most dangerous finding of the session and it lands squarely on my contract. It also **independently confirms my §38 trace** — arrived at from measurement while I arrived from source, and we agree on the mechanism line-for-line:

> `prepare_session_prefix` (`runtime.rs:997`) forks on `uses_token_prefix_cache()` = `has_runner() || is_windowed()` (`decode/state.rs:206`). **Branch A (`:1017-1024`) is REPORTING-ONLY** — computes `common_prefix_len`, loads no KV, never sets `loaded_prompt_prefix`, so prefill recomputes every token. And the value is `common_prefix_len(...).filter(|&len| len > 0)` — **any ONE shared leading token scores a hit.** Every `/v1/chat/completions` request shares the chat-template preamble. **⇒ every request reports a hit, forever.**

### 44.1 D127 — MY ENUM CANNOT EXPRESS THIS, AND THAT IS THE FINDING

All five states assume **that if a value is present, it is true.** `measured`/`stale` carry true values; `unavailable`/`not-applicable`/`pending` carry none.

> **THERE IS NO STATE FOR *WE HAVE A NUMBER AND IT LIES*. `prefix_cache_hits` at 95% is exactly that, and it is the ONE case the whole apparatus was built to stop.**

**This is my D111 arriving as a live defect rather than a lesson:** *suspicion tracks implausibility, not falsehood.* We spent the session hardening against fabricated **zeros** — seven mechanisms — and **a fabricated 95% walks through every one of them.** It is `measured` by every test we have: computed from real inputs, no hardcoded literal, no dead branch, greppable as clean, and it *moves*. It passes AC6, AC20, AC44 and the provenance audit simultaneously.

### 44.2 D128 — DO NOT ADD A SIXTH STATE. THE FIELD MUST BE UNBINDABLE.

The tempting fix is a `misleading` state with its own treatment. **Reject it, on design grounds:**

> **A STATE IS A RENDERING INSTRUCTION, AND THERE IS NO HONEST RENDERING OF A NUMBER THAT LIES.** A badge, a hatch and a hover do not undo a **95%** set in 48px. Every treatment I designed works by making *absence* legible; **none of them can make a present, plausible, wrong number safe.** Shipping `misleading` would mean the design system had formally licensed displaying a known falsehood, provided it wore the right underline.

- **D128:** fields known to misreport are removed from the **binding surface**, not given a state. A `FORBIDDEN_FIELDS` registry in the store throws at **bind time** with the reason and the source line. **The panel cannot render it, because the panel cannot obtain it.**
- **This is the project's own thesis applied to itself** — *the honesty bar is enforced by the API shape, not developer discipline.* We enforced it for every field except the one that needed it most. **A rule in a doc saying "never bind `prefix_cache_hits`" is exactly the discipline-based safeguard we rejected everywhere else**, and there are now three separate documents carrying that instruction, which is three chances to miss it.
- **D129:** entry one is `prefix_cache_hits` and every rate derived from it, **on BOTH servers.** Not `measured` on dynamic — **@376a0297's `hit_rate 0→0.5` is Branch A's always-true counter, not evidence of reuse.**

### 44.3 🔴 D130 — TWO MEASUREMENTS OF THE SAME THING DISAGREE IN OPPOSITE DIRECTIONS. DO NOT RESOLVE THIS BY PICKING ONE.

- **@376a0297:** dynamic path, e2e **1.53s → 1.22s** — a 20% *speed-up*, cited as the reason Scenario B ships.
- **@fc8b5d97:** shared-prefix arm **+7.0% SLOWER**, n=20, with six controls, and a source mechanism explaining why a speed-up is impossible.

**These cannot both describe the same code path.** One is n=1 without a stated control; the other is n=20 with controls and a line-numbered mechanism. **I am not qualified to adjudicate the measurement and I am not going to — but the design consequence holds under BOTH: the panel cannot bind `hits` either way**, because under the PM's numbers the counter is still always-true, and under QA's it is also slower. **The scenario's fate is @12e42da8's call; the binding is mine, and it is closed.**

- **D130:** where two agents' measurements of one quantity disagree in **direction**, neither may be cited as established until reconciled. **The flattering one is the one that will get cited by default, precisely because nobody investigates a pleasant result** — the mechanism that nearly shipped the 95%.

### 44.4 What this costs, honestly

Scenario B loses its headline number. **I would rather ship two pillars that are true than three where the most quotable one is invented** — and per D107, the honest version is always the less punchy one. The remaining honest surface is real: **`not-applicable` on the batching path is still the best teaching moment we have**, and it does not depend on `hits` being trustworthy — it depends on `hits` being *structurally frozen*, which QA's finding **strengthens** rather than undermines.

| # | Decision | Rationale |
|---|---|---|
| D127 | The five states cannot express "present and false"; that is the hole | Every state assumes a present value is true |
| D128 | **No sixth state.** Misreporting fields become unbindable at the store | There is no honest rendering of a number that lies |
| D129 | `prefix_cache_hits` + derived rates forbidden on BOTH servers | Branch A's counter is always-true, not measured |
| D130 | Direction-conflicting measurements: neither is established until reconciled | The flattering result is the one nobody audits |

---

## 45. THE SWITCHER, CONSOLIDATED — AND WHY MY APPEND-ONLY DISCIPLINE JUST FAILED (D131–D134)

@e00032a4 told @bb2ee824 and me to **build the switcher**. Before that could happen safely I had to fix my own document, because it was still issuing a superseded instruction.

### 45.1 🔴 D131 — AN APPEND-ONLY DOCUMENT CANNOT CORRECT ITSELF

All session I appended. Never edited. That kept the artifact and in-repo copies byte-identical and every reversal auditable, and I was pleased with the discipline. **It has a failure mode I walked straight into:**

> **§13(f) — *"scenario tabs are filtered by profile… a scenario is never shown disabled"* — was still sitting there reading as live guidance. Its reversal is in §26, SEVEN HUNDRED LINES LATER.** A developer who greps `scenario tabs` hits the wrong instruction first and has no reason to keep reading.

**I originated that line. @376a0297 adopted it. @12e42da8 reversed it. And the wrong version stayed authoritative in the contract the devs build from** — while I spent the session ruling that *a memorable framing outruns its retraction* (D112) and that *the gaps between owners are where defects live.* **Mine was a gap inside a single file I owned outright.**

- **D131:** a reversal must be applied **at the point of use**, not only recorded at the point of decision. §13(f) is now struck in place with a pointer forward. **Append-only is right for the DECISION LOG and wrong for the INSTRUCTIONS** — the log is history, the instructions are executable, and history must never be executable.
- **The struck text is kept, not deleted.** Deleting it would make the reversal invisible, which is @e00032a4's *"a gaps list that silently loses entries is indistinguishable from an abandoned document."*

### 45.2 THE SWITCHER SPEC — ONE PLACE, SINCE IT IS CURRENTLY SPREAD ACROSS SIX BROADCASTS

Consolidated so nobody reconstructs it from chat. Every line traces to a ruling already made:

1. **All three scenario tabs are ALWAYS VISIBLE.** Never filtered, never hidden. (Lead, superseding §13(f).)
2. **The tab is a NAVIGATION, not a fetch** — `http://127.0.0.1:<port>/demo/?scenario=<id>`, **trailing slash mandatory** (`/demo` redirects, and relative module imports resolved against `/` 404 every module → blank page, console-only error).
3. **The scenario travels in the query string.** Without it the visitor lands on the destination's default view and **the click reads as having failed.** (@732c7548.)
4. **The tab NAMES its destination** — `Scenario B · dynamic engine · :8124`. An unexplained navigation is a misclick; an announced one is a tour. Puts the curl-able port on screen for free. (D70.)
5. **Capability is DETECTED from whatever answers, never assumed from the tab.** The tab picks the origin; the origin declares its own profile. **A visitor pointing this at their own server must not get a lying dashboard** — that, not our topology, is the real reason for detection. (@376a0297.)
6. **Detect ONCE, on load. Build no re-evaluation lifecycle.** A navigation *is* a page load: fresh document, fresh JS context, heap destroyed. **There is no origin-switch event to subscribe to and no teardown that could ever fire.** (@e00032a4 — and this deletes work, not adds it.)
7. **Probe both origins once on load; NEVER poll a foreign origin.** A tab whose server is down gets a dimmed-but-**focusable** treatment and says so *before* it is clicked. (§26.)
8. **The URL bar is the label.** A full navigation to a visibly different origin is a stronger provenance signal than any in-page badge, because the browser renders it and we cannot get it wrong.

### 45.3 D132 — THE DIMMED TAB MUST NOT USE THE FIELD-STATE TREATMENTS

A tab pointing at a down server is **not** `unavailable` in the field sense. Reusing the dotted underline and `--og-unavail-fg` would be **a third vocabulary colliding with the first two** — and the connection indicator already owns a fourth (`connected`/`connecting`/`no-model`/`unreachable`).

- **D132:** the down-tab treatment is **reduced opacity + an explicit `:8124 not responding` sublabel**, never the field-state palette. **Field states describe VALUES. A tab is not a value.** My new `state-channel.test.js` asserts the five field states are pairwise distinct **once colour is removed** — a tab borrowing that palette would be legible but would mean something entirely different, which is worse than illegible.

### 45.4 D133 — THE SWITCHER IS THE ONE INTERACTION THAT MUST SURVIVE A DEAD SERVER

Per AC52's non-negotiable path (kill the server mid-scenario, restart, full recovery, no manual refresh): **the switcher is the only control whose failure is unrecoverable in-page**, because a failed navigation replaces the document with the browser's own error page — **our error states never render, our recovery never runs, and the visitor is looking at Chrome's dinosaur.**

- **D133:** **never navigate to an origin whose probe has not succeeded.** Clicking an unresponsive tab must keep the visitor on the current page and surface our own actionable error (`the dynamic server isn't running — start it with: …`), reusing @d7cf9b84's byte-identical launch command. **A navigation is irreversible from our side; the probe is the last moment we still control the page.**

### 45.5 D134 — ACCESSIBILITY: THE TABS ARE NOT A TABLIST ANY MORE

§8 specified `role="tablist"` with roving tabindex and arrow-key traversal. **That was correct when tabs swapped in-page panels. It is now WRONG and actively harmful:** ARIA tabs promise same-document panel switching, so a screen-reader user pressing → expects focus to move within the page — and instead **the document is destroyed and replaced.** Arrow keys must never trigger navigation.

- **D134:** the switcher is a **`<nav>` containing ordinary links**, with `aria-current="page"` on the active scenario. Ordinary links because **that is what they are** — they navigate. This also gets middle-click, ⌘-click, "open in new tab" and the browser's own affordances for free, all of which a `role="tab"` div silently breaks. **The visual design does not change at all; only the semantics, and only because the semantics became a lie when the mechanism changed underneath them.**

| # | Decision | Rationale |
|---|---|---|
| D131 | Reversals applied at the point of use; append-only for the log, not the instructions | History must never be executable |
| D132 | Down-tab uses opacity + sublabel, never the field-state palette | Field states describe values; a tab is not a value |
| D133 | Never navigate to an unprobed origin | A failed navigation replaces our document with the browser's error page |
| D134 | `<nav>` + links + `aria-current`, not `role="tablist"` | ARIA tabs promise in-page panels; arrow-key navigation destroys the document |

---

## 46. 🔒 TWO VOCABULARIES, ONE IDEA — RULED. `state` IS DERIVED FROM `classification`, TOTALLY, AND PANELS NEVER READ `classification`. (D135–D138)

@c0de4c2e found that the same distinction is encoded **twice, in two files, by the same author**, and asked me to collapse it. Verified — and the verification turned up a **live bug**, which is why this could not stay a naming question.

```js
telemetry-field.js:~94      state:          measured | pending | stale | unavailable | not-applicable   (5)
telemetry-provenance.js:31  classification: MEASURED | DOCUMENTED_ZERO | NOT_PLUMBED | STRUCTURALLY_BYPASSED   (4)
```

### 46.1 🔴 THE LIVE BUG THIS EXPOSES — `app.js:245`

```js
entry.classification === 'MEASURED' ? FIELD_STATES.MEASURED : FIELD_STATES.UNAVAILABLE;
```

> **FOUR CLASSIFICATIONS COLLAPSED INTO TWO STATES. `STRUCTURALLY_BYPASSED` RENDERS AS `unavailable`.**

That is precisely the outcome @12e42da8 ruled against an hour ago — *"`unavailable` PROMISES A NUMBER THAT CANNOT EXIST, and it lies in the FLATTERING direction: it implies we're behind on measurement rather than that we made an architectural choice."* **It is live, in the provenance audit view — the one screen whose entire purpose is to state our honesty policy — and it is currently the screen that misstates it.** A ternary is how a five-state vocabulary quietly becomes a two-state one.

### 46.2 D135 — THEY ARE NOT SYNONYMS AND MUST NOT BE MERGED. THEY ARE DIFFERENT AXES.

@c0de4c2e proposed collapsing them to one. **I'm declining the merge and taking the harder half of their point, because the two answer genuinely different questions:**

- **`classification` is STATIC** — a property of the field's *implementation*, true at HEAD regardless of whether the server is running. It is the provenance audit's verdict. It changes only when someone changes Rust.
- **`state` is RUNTIME** — a property of *this poll*. `pending` and `stale` have no classification equivalent and cannot: they are facts about the network, not about the code.

**Merging them would delete `pending`/`stale`, or would make a static audit verdict change every 250 ms. Both are worse.** But @c0de4c2e is right that **two vocabularies a panel can reach for is a divergence generator** — so:

### 46.3 🔒 D136 — `state` IS A TOTAL PURE FUNCTION OF `classification` PLUS LIVENESS. NEVER ASSIGNED INDEPENDENTLY.

| `classification` | ⇒ `state` | Why |
|---|---|---|
| `MEASURED` | `measured` → `stale` → `pending` **by liveness only** | The only classification whose state can vary at runtime |
| `NOT_PLUMBED` | **`unavailable`** | Data exists in-process; an endpoint could expose it. Plumbing closes this. |
| `DOCUMENTED_ZERO` | **`unavailable`** | The server writes a constant. Plumbing could replace it with a measurement. |
| `STRUCTURALLY_BYPASSED` | **`not-applicable`** | This execution path never consults the subsystem. **Nobody will ever instrument it, because there is nothing to instrument.** |

- **D136:** this mapping is **one exported function**, total over all four inputs, **throwing on an unrecognised classification** (AC49's rule, applied one layer up). No module may compute a state from a classification any other way. **`app.js:245` is the first caller to fix.**
- **The distinction that decides the last two rows is the Lead's, and it is the whole reason the fifth state exists:** `unavailable` is **a promise** — someone will do this work. `not-applicable` is **an architectural fact.** `DOCUMENTED_ZERO` and `NOT_PLUMBED` are both promises; `STRUCTURALLY_BYPASSED` is not.

### 46.4 🔒 D137 — PANELS BRANCH ON `state`. NEVER ON `classification`.

- **D137:** `classification` is **audit metadata**. It may be *displayed* (the audit table, the hover text explaining *why*), and it is the input to D136's function. **It is never a rendering branch in a panel.** A panel that reads `classification` has re-implemented the mapping, and two implementations of one mapping is exactly the divergence @c0de4c2e is warning about — **the second copy is always the one that misses the fifth state, because the fifth state was added last.**
- **Enforceable the same way as the raw-literal ban (D122):** a test asserting no file under `dashboard/` references `classification` outside the audit view.

### 46.5 D138 — FOUR NAMES FOR ONE IDEA: `not-applicable` WINS, AND MY OWN NAME LOSES

Live right now: `not-applicable` (ruled, in code) · `STRUCTURALLY_BYPASSED` (in code, different file) · **`Bypassed` (mine, §4.7)** · em-dash-plus-hover (the treatment).

- **D138:** **`not-applicable` is the only name that appears in UI, docs or panel code.** `STRUCTURALLY_BYPASSED` survives **only** as a `classification` value, because it names the *cause* and `not-applicable` names the *consequence* — and the audit table is the one place the cause is the point. **My §4.7 "Bypassed" is retired outright; strike it on sight.** A word I coined that now has three better-specified rivals is not worth defending — **the cost of a synonym is paid by every future reader, and I'd be the only one who thinks in it.**

### 46.6 Two corrections to @c0de4c2e, offered because they were right to push

Both of their closing items are **already closed**, and I'd rather they spend the time elsewhere: **`styles/panels.css` IS linked** — `index.html:29`, committed `3af5c8d7`, with `stylesheet.test.js`'s reachability assertion now green. And **derivation contagion was ruled in §33** — `not-applicable > unavailable > pending > stale > measured`, which is the precedence they proposed, for the reason they gave. **Their read of the mechanism was right in both cases; only the timestamps were stale.** The `MEASURED: 'ok'` mismatch they flag is real and is @12e42da8's open call (§42).

| # | Decision | Rationale |
|---|---|---|
| D135 | `state` and `classification` are different axes; do not merge | One is static implementation truth, one is per-poll liveness |
| D136 | `state` is a **total pure function** of classification + liveness, throwing on unknown | Two ways to compute one value is a divergence generator |
| D137 | Panels branch on `state` only; `classification` is audit metadata | A second copy of the mapping always misses the newest state |
| D138 | `not-applicable` is the only public name; my `Bypassed` is retired | A synonym's cost is paid by every future reader |

---

## 47. 🔴 SWITCHER REVIEW — THE FILTERING CAME BACK UNDER A DIFFERENT PREDICATE (D139–D141)

`ui/scenario-switcher.js` landed minutes after §45. **Most of it is right, and two things in it are better than my spec.** One thing reintroduces a reversed ruling, and it is invisible from inside the file.

### 47.1 ✅ What is right, including where the author beat the spec

- **D134 satisfied exactly.** `<a>` elements, `aria-current="page"`, `aria-label="Scenarios"`, **no `role="tab"`, no keydown traversal.** The comment gives the correct reason unprompted: *"Announcing them as tabs would promise in-place panel switching that this control genuinely does not do."*
- **The remote hint is marked in TEXT, not colour** — `on the dynamic server` — with the reason written in the file. That is D21 applied without being asked.
- **Better than my spec:** the `title` explains *"The two servers are separate processes, so this is a page load rather than a panel switch."* **I specified naming the destination; they explained the mechanism, which is the part that stops the navigation feeling like a misclick.**

### 47.2 🔴 D139 — UNREACHABLE SCENARIOS ARE NOT RENDERED AS TABS AT ALL

```js
const reachable = plans.filter(({ plan }) => plan.available);
for (const { id, plan } of reachable) list.append(buildTab(...));   // ← only reachable
```
Everything unreachable is moved into a grouped `<aside>`. **So a scenario whose server is not running DISAPPEARS FROM THE TAB LIST.**

**This is the ruling @12e42da8 reversed and I struck from §13(f) forty minutes ago — returning under a different predicate.** The reversal was about filtering by **profile**; this filters by **reachability**. **The author has almost certainly not violated any rule they were aware of.** But the visitor-facing effect is identical and lands harder:

> **THE MOST COMMON FIRST-RUN STATE IS ONE SERVER RUNNING.** A visitor who starts only the scatter server sees a product with **one scenario**, and **never learns that paged KV or prefix caching were ever part of the demo.** @376a0297's words for exactly this: *filtering the tabs would have hidden the existence of half the product from every visitor.*

**And I must own the other half of why this happened: my own §31 says inactive PANELS collapse into ONE group card**, because six identical notices is the wall-of-zeros failure in a politer typeface. **The author applied my panel rule to the tabs, which is a completely reasonable reading of my contract.** The distinction I never wrote down:

- **D139:** **panels display VALUES; tabs advertise CAPABILITIES.** Collapsing an empty panel hides a *number the visitor can see is missing*. **Collapsing a tab hides the EXISTENCE of a feature — the visitor cannot miss what was never named.** Grouping is right for the first and wrong for the second. **The two rules read alike and point opposite ways, which is my failure to distinguish them, not the author's to infer it.**

### 47.3 🔒 D140 — THE RESOLUTION KEEPS BOTH RULINGS, AND KEEPS THEIR NOTE

The tension is real and the author's instinct was sound; it does not require choosing.

- **All three tabs render, always.** An unreachable one is **present, dimmed, and FOCUSABLE**, carrying `:8124 not responding` **on screen** — not in a `title`, because a hover is not a channel for a keyboard or touch visitor, and per §45.2 the port belongs on screen for the skeptic who wants to `curl` it.
- **It does not navigate** (D133 preserved — their removal solved this correctly, and interception solves it while keeping the tab). Activating it reveals the launch command **inline**.
- **`buildUnreachableNote` STAYS** — as the single grouped *explanation*, which is the right shape and satisfies §31. **It is a supplement to the tabs, never a substitute for them.**
- **D141:** the dimmed tab uses **opacity + the on-screen sublabel**, never `--og-unavail-*` or the dotted underline (D132). Four vocabularies are live; a tab wearing the field-state palette would be **legible and mean the wrong thing, which is worse than illegible.**

### 47.4 The transferable part

**This is the first defect tonight caused by a rule of mine being applied CORRECTLY somewhere it didn't belong.** Every other one was a stale premise, a wrong tree, or an unenforced claim. **A rule that generalises further than its author intended is indistinguishable, from the reader's side, from a rule that was meant to.** The fix is not more prose — it is to state the *boundary* alongside the rule. §31 now carries D139's distinction inline.

| # | Decision | Rationale |
|---|---|---|
| D139 | Panels display values and may group; tabs advertise capabilities and may not | A visitor cannot miss a feature that was never named |
| D140 | All three tabs always render; unreachable = present, dimmed, focusable, non-navigating; the grouped note stays as explanation | Preserves the always-visible ruling, D133, and §31 together |
| D141 | Dimmed tab uses opacity + on-screen sublabel, never the field-state palette | Legible-but-wrong-meaning is worse than illegible |

---

## 48. AC52 — THE GRAYSCALE CHECK, COMPUTED (D142–D144)

I have owed an AC52 verification of the five treatments all session and kept deferring it to "when there's a browser." **There is now a browser-servable page — and it turns out the most important half of AC52 never needed one.** The question *"can `not-applicable` be told from `unavailable` without colour?"* is answered by arithmetic on the tokens, and arithmetic is the stronger instrument here: an eyeball on one monitor at one gamma is a sample of one.

**EVIDENCE CLASS: COMPUTED from `styles/tokens.css`, not OBSERVED in a browser.** Stated plainly because §43 made evidence class a first-class property of every claim in this document, and because the remaining half of AC52 — that the treatments *survive a real render* — is still unobserved.

### 48.1 The measurement

Relative luminance (WCAG formula) of the four absence tokens, and the grayscale-equivalent contrast between each pair:

| state | hex | rel. luminance | 8-bit gray |
|---|---|---|---|
| `unavailable` | `#758493` | 0.2239 | **129** |
| `pending` | `#748494` | 0.2235 | **129** |
| `stale` | `#7a8794` | 0.2360 | 132 |
| `not-applicable` | `#7e8fa0` | 0.2662 | 140 |
| `measured` | `#e6edf3` | 0.8386 | 235 |

| pair | grayscale contrast |
|---|---|
| `unavailable` vs `pending` | **1.001 : 1** |
| `unavailable` vs `stale` | 1.044 : 1 |
| `pending` vs `stale` | 1.046 : 1 |
| `not-applicable` vs `stale` | 1.105 : 1 |
| `unavailable` vs `not-applicable` | 1.154 : 1 |

### 48.2 🔒 D142 — COLOUR IS NOT A WEAK CHANNEL HERE. IT IS NOT A CHANNEL AT ALL.

**`unavailable` and `pending` are the SAME GRAY — 129 and 129, 1.001:1.** Every absence pair sits under 1.16:1, where 3:1 is the floor for non-text UI distinction. **In grayscale, on a projector, or to a visitor with achromatopsia, the four absence states are ONE state.**

This is the design working as intended — §21 chose near-identical foregrounds deliberately so that pattern would carry the meaning — **but I had been describing the second channel as REINFORCEMENT, and the numbers say it is the ENTIRE SIGNAL.** That is not a nuance:

- **D142:** for the absence states, **the non-colour channel is load-bearing on its own.** Any state whose border/decoration is missing is not *harder* to identify — it is **exactly indistinguishable** from two others. There is no partial credit and no graceful degradation.

### 48.3 D143 — WHAT THIS RETROSPECTIVELY PROVES ABOUT THE MUTATION TEST

Earlier I deleted every `border-bottom` from the four non-default states and reported the suite stayed green while the states "rendered identically." **I wrote that from reading the CSS and I was understating it by accident.** These numbers show the mutation did not *degrade* the distinction — **it collapsed four states into one, completely, with a 1.001:1 residue.** A visitor would have had no way, by any means, to tell a value that is *coming* from one that is *structurally impossible*.

- **D143:** **a test that guards the only channel is not a style test, it is a correctness test.** `state-channel.test.js` is now the single thing standing between the page and a total loss of absence semantics, and it should be treated with the seriousness of the provenance envelope itself — **not as a lint.**

### 48.4 D144 — WHAT REMAINS GENUINELY UNOBSERVED

Computation settles *separation*. It cannot settle:
1. whether `3px double` and `1px dotted` are **telling apart at 14px on a real display** — sub-pixel rendering can turn `double` into a smear;
2. whether the em-dash and `n/a` **hold their reserved width** when a number arrives beside them;
3. whether any of it survives the **compressed screenshot** in AC8.

- **D144:** these three go to @fc8b5d97's browser pass as named checks, not as "eyeball the states." **I am not marking AC52 satisfied on the strength of arithmetic, and this section is not a sign-off — it is the half I could do without a browser, labelled as such.**

| # | Decision | Rationale |
|---|---|---|
| D142 | For absence states the non-colour channel is the ENTIRE signal, not reinforcement | All pairs ≤1.16:1 grayscale; `unavailable`/`pending` are 1.001:1 |
| D143 | `state-channel.test.js` is a correctness test, not a lint | It guards the only channel that carries absence semantics |
| D144 | AC52 stays OPEN; three named render checks go to the browser pass | Arithmetic settles separation, not legibility |

---

## 49. 🔒 THE ONE-LINE RULING PANEL AUTHORS ACTUALLY NEED (D145–D147)

@c0de4c2e is right that four live names for one concept is divergence *inside the honesty layer*, and right that asking me who was correct is worthless next to asking me what to type. So the ruling is a rule about **which question a panel is allowed to ask**, not about which word wins.

### 49.1 🔒 D145 — PANELS BRANCH ON `state`. ONLY ON `state`. EVER.

```js
// ✅ the only shape a panel may use
if (field.state === FIELD_STATES.NOT_APPLICABLE) { ... }

// ❌ forbidden in any panel, for any purpose
if (field.classification === 'STRUCTURALLY_BYPASSED') { ... }
```

**Nothing "wins", because they were never competing — they answer different questions, and only one of them is a rendering instruction.**

| name | axis | answers | who reads it |
|---|---|---|---|
| `state` | **runtime**, this poll | *how do I render this cell?* | **panels — exclusively** |
| `classification` | **static**, implementation truth at HEAD | *why is it like that?* | the registry, `reason` copy, AC54's generated table |
| `Bypassed` (§4.7) | — | — | **retired, D138. My term. Gone.** |
| em-dash / `n/a` | presentation | the glyph `state` resolves to | `formatFieldText`, nothing else |

`state` is already a **total pure function of `classification` + liveness** (D136), so **every question a panel could answer from `classification` is already answered, correctly and earlier, by `state`.** A panel reading `classification` is not getting more information — **it is re-deriving a mapping that has already been made, and it will drift the moment a classification is added.**

- **D145:** **`classification` is INPUT to the envelope; `state` is its OUTPUT. A panel consuming an input has reached around the contract.** That is the whole of the ruling, and it is enforceable by grep.

### 49.2 D146 — WHY THIS IS THE SAFE DIRECTION EVEN THOUGH `classification` IS RICHER

The tempting objection is that `classification` carries more detail, so surely a panel wanting nuance should read it. **That is exactly backwards, because the nuance is not renderable.** §21 gives five treatments; there is no sixth glyph waiting for `DOCUMENTED_ZERO` that `NOT_PLUMBED` shouldn't get. **The extra detail exists to be READ BY A CONTRIBUTOR deciding whether to plumb an endpoint or delete a field — it is a maintenance signal, not a visitor-facing one.** Routing it to a panel converts a maintenance fact into a rendering decision, which is how a five-state vocabulary quietly becomes a nine-state one.

### 49.3 🔴 D147 — AND THE ORIGIN-KEYING IS 100% ON FIELDS THAT WERE DELETED

@c0de4c2e reported 3 of 35 entries carry `byOrigin`. At HEAD it is **5 of 46** — and I checked *which five*:

```
:332 prefix_cache.hits          :359 prefix_cache.lookups
:541 metrics.prefix_cache_hits  :567 metrics.prefix_cache_lookups
:599 prefix_cache.hit_rate
```

**All five are prefix-cache fields. Every one of them is a field the Lead removed from the demo entirely on the RED verification.** So the per-origin override machinery is **fully built, correct, tested — and applied exclusively to the fields that must never render.** `kv.*` and `batch.*`, which *will* ship and *do* differ by server, carry **no override at all.**

Two corrections to the report, because the difference changes the fix:
- **`batch.*` is not INVERTED — it is ABSENT.** No `byOrigin` key exists on it, so there is nothing set backwards. That is better news (nothing to un-break) and worse news (nothing to notice): **an inverted flag is a bug someone eventually sees; a missing dimension renders a plausible number on both servers forever.**
- The machinery is **not** the missing piece. `provenanceFor(key, origin)` works (`:755-756`) and is proven by five entries. **This is a data gap, not an engineering gap** — which is why it is finishable tonight.

- **D147:** whoever adds `byOrigin` to `kv.*` and `batch.*` must **derive the direction from the incrementing code, never from the field name** (QA-PLAN §7). Getting it backwards renders Scenario A's headline metric as `n/a` **on the exact server carrying the 2.46× number**, and — per §48 — a wrongly-`n/a` field is *indistinguishable from three other absence states without its border*. **Two of our failure modes would have to be believed at once, and both are silent.**

| # | Decision | Rationale |
|---|---|---|
| D145 | Panels branch on `state` only; `classification` is registry/copy-only | `classification` is input, `state` is output; consuming an input reaches around the contract |
| D146 | The extra classification detail is a CONTRIBUTOR signal, not a visitor one | There is no sixth glyph; routing it to panels grows the vocabulary silently |
| D147 | `byOrigin` on `kv.*`/`batch.*` is a DATA gap, not an engineering gap; direction from the incrementing code | All 5 existing overrides sit on deleted prefix fields; a missing dimension never announces itself |

---

## 50. APPLYING THE DROP RULINGS AT THE POINT OF USE (D148–D149)

Two rulings landed that invalidate copy inside this document: **`preempted_total`/`sessions.paused` DROPPED, final** (`ContinuousBatchManager` has no `Scheduler` field — preemption is not disabled, the component is *absent*), and **AC59: never the words "batch size" in UI copy**, because `onnx_genai_batch_size_current` is `fetch_add(1)` at the HTTP layer.

**Per D131 I applied these AT THE POINT OF USE, not as a section at the end.** Appending a correction to a 2,700-line spec and calling it done is how §13(f) shipped a reversed instruction for four hours. Six sites edited in place: the verbatim `reason` table (§4.2), the treatment markup example, the unavailable-series sketch, the honesty footer block, the Scenario A panel, and the `describe()` example.

### 50.1 🔴 D148 — THE STRUCK COPY WAS NOT MERELY STALE, IT WAS FALSE

The dropped row in §4.2 read: *"The scheduler performs preemption but keeps no counter for it."* **That sentence asserts a capability that does not exist.** It was written to explain an absence and it invented a *different* absence — a scheduler that preempts silently, rather than no scheduler at all.

- **D148:** **copy that explains why a number is missing is itself an unverified claim about the system, and it is the LEAST likely claim on the page to be audited** — because it sits in the `reason` field, which reviewers read as an apology rather than as an assertion. **A wrong number gets challenged; a wrong explanation gets sympathy.** Every `reason` string must cite the same way a value does.

That table is headed *"Reference copy, to be used verbatim."* **Prose marked for verbatim reuse is executable — it is the one kind of writing in this document that a developer is instructed not to think about.** It must be held to the standard of code, and it was not.

### 50.2 🔒 D149 — A NARROW DIVERGENCE FROM AC59'S LITERAL WORDING, DECLARED RATHER THAN TAKEN QUIETLY

AC59 says **never the words "batch size" in UI copy.** The page now renders `engine batch size  — ⓘ`, which contains those words. **That is deliberate and I am flagging it rather than letting a reviewer find it.**

- **The ban's target is the phrase used as a LABEL FOR A VALUE.** `batch size: 8` is the lie, because the 8 is in-flight HTTP requests.
- **Naming the absent quantity is the opposite act.** Enforced literally, AC59 makes it impossible to say *"we cannot show you the engine's batch size"* — and that sentence is one the demo needs, because the gap between **requests in flight** and **actual batch** is the single most important idea in Scenario A. **Suppressing the term would leave the visitor with only the misleading number and no name for the honest one.**
- **D149:** the phrase is banned **beside a number** and required **beside an em-dash**. Mechanically: `"batch size"` may appear in the DOM only within an element whose `data-state` is `unavailable`. **That is greppable, so it is enforceable** — @c0de4c2e, this belongs in QA-PLAN §7 as the check, and @376a0297, AC59 wants this one clause or it forbids its own purpose.

### 50.3 The replacement teaching pair is better than the one it replaced

§20's honesty pairing was `rejections: 0` beside `preemptions: —`. It is now `rejections: 0` beside `engine batch size: —`. **The absence now sits in the very panel whose headline claim is about batching** — so the visitor learns the good-zero-vs-absence distinction at the point where admitting it costs us something, instead of about a subsystem they had no reason to expect. **An absence we would rather not mention teaches the thesis better than a convenient one.**

| # | Decision | Rationale |
|---|---|---|
| D148 | `reason` copy is an unverified claim and must be cited like a value | The struck line invented a scheduler that preempts silently; a wrong number gets challenged, a wrong explanation gets sympathy |
| D149 | "batch size" banned beside a number, REQUIRED beside an em-dash; greppable via `data-state="unavailable"` | Enforced literally, AC59 would forbid the page from naming the honest quantity it exists to distinguish |

---

## 51. 🔬 THE NULL RESULT PANEL — SHIPPING THE EXPERIMENT THAT DIDN'T WORK (D150–D155)

@12e42da8 re-scoped Scenario B to **(a)** paged-KV page allocation and **(b)** *the measured non-result, shipped and explained*, and assigned me the treatment. **(b) is the hardest thing on this page and the most valuable**, because it is the only panel whose entire persuasive force comes from us reporting something we would rather not.

### 51.1 🔴 D150 — THE HEADLINE IS NOT "7% SLOWER". OUR OWN NOISE IS BIGGER THAN THAT.

The measured numbers: ARM A (one shared ~900-token prefix ×6) **1341 ms**; ARM B control (six prefixes differing from token 0) **1254 ms**. Shared is **6.9% slower**.

**Before designing anything I checked that number against the only noise measurement we have.** @fc8b5d97 re-ran a **byte-identical binary on the same machine 75 minutes later**: 33.415 → 30.151 tok/s, **9.8% drift from background load alone.**

> **OUR MEASUREMENT NOISE (9.8%) IS LARGER THAN OUR MEASURED EFFECT (6.9%). SO "PREFIX CACHING MAKES IT SLOWER" IS NOT A CLAIM WE ARE ENTITLED TO MAKE.**

- **D150:** the panel reports **"no effect detected"**, never "slower". **Rendering the 6.9% as a finding would be the exact error we spent the night catching — a real number, correctly computed, meaning something it does not mean.** It would also be *more* interesting than the truth, which is precisely why it must be resisted (D107). The 6.9% appears only inside the noise band, where its being smaller than the band is the point.

### 51.2 🔒 D151 — THE DETECTION FLOOR IS THE HERO, NOT THE RESULT

A null result is worthless without evidence the test could have found the effect. Prefill is ~90% of long-prompt TTFT, so a working cache collapses **1380 ms → ~140 ms: an 89.9% drop.**

**That is the number that gets the largest type on the card.** It converts *"we didn't see it"* into *"it is not there"* — the difference between an absence of evidence and evidence of absence, and the only honest basis for the claim.

```
┌─ EXPERIMENT · recorded, not live ────────────────────────────┐
│                                                              │
│  Does prefix caching reduce time-to-first-token?             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │              NO EFFECT DETECTED                        │  │
│  │  This test could detect a 90% improvement.             │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│   shared prefix ×6   ████████████████████████ 1341 ms        │
│   control, no sharing████████████████████████ 1254 ms        │
│   ░░░░░░░ measurement noise on this machine: ±9.8% ░░░░░░░   │
│                                                              │
│   ▚▚▚▚▚▚▚▚▚▚ where a working cache would land ▚▚▚▚▚▚▚▚▚▚     │
│   ~140 ms  ▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚▚     │
│                                                              │
│   Recorded 2026-07-30 00:12 · n=20 · reproduce:              │
│   $ ./scripts/prefix-control.sh --arms both --n 20           │
└──────────────────────────────────────────────────────────────┘
```

- **D151:** **the two observed bars must be drawn on the same axis as the hatched target region, and must conspicuously fail to reach it.** A visitor should see the gap before reading a word. **Two bars alone say "these are about the same" — which is *un*convincing, because that is also what a broken test looks like.** The target region is what makes the flatness meaningful.

### 51.3 🔒 D152 — THIS IS THE ONLY RECORDED PANEL ON A PAGE THAT OTHERWISE RE-MEASURES, AND THAT MUST BE UNMISSABLE

Every other panel polls. **This one cannot** — it needs a controlled 20-request protocol that would take a minute and perturb everything else. So it ships **recorded**, and the honesty footer's own argument (*a stored baseline is a claim about a machine that no longer exists*) applies to us.

- **D152:** the card carries a **different chrome from every live panel**: no connection dot, no cadence chip, no `ˢ`/`ᴰ` badge, and a persistent **`recorded · <timestamp>`** in the header. **Per D142 this distinction cannot be colour** — it is the *presence of a timestamp* and the *absence of the live dot*, both structural.
- **Its `source` is `simulated`? No — `measured`, with `observedAtMs` far in the past.** The field envelope already carries exactly the right vocabulary for this and it should be used rather than invented: **a recorded measurement is a `stale` measurement with an honest age**, which is the case §21's `stale` treatment exists for. **The most rigorous panel on the page renders in the state we built for staleness, and that is correct, not a demotion.**

### 51.4 D153–D154 — TWO THINGS THE PANEL MUST NOT DO

- **D153: it must not name a single banned field.** No `prefix_cache_hits`, `_lookups`, `_hit_rate`, `prefix_hashes` — the tripwire (`prefix-counters-forbidden.test.js`) covers modules, and this panel's copy is where the names would most naturally creep back, because the panel is *about* prefix caching. **The panel discusses the mechanism and reports TTFT; it never reports a counter.** The counter is the thing that lied.
- **D154: it must not editorialise about the runtime.** *"onnxruntime-genai can't do this either"* is not on the page. **We measured our own engine and found a gap in our own engine.** A null result about ourselves is credible precisely because it costs us something; the moment it is turned into a competitive claim it becomes marketing and the credibility is spent.

### 51.5 🔒 D155 — WHY THIS PANEL IS THE THESIS, STATED FOR THE README

- **D155:** **anyone can ship a number going up. Almost nobody ships the experiment that didn't work, next to the control proving the test could have detected it.** Every other panel asks the visitor to trust that we would have told them if something were wrong. **This one is the proof that we would** — and it costs us a headline feature to make. **It should be the panel we point at first, not the one we bury at the bottom.**

| # | Decision | Rationale |
|---|---|---|
| D150 | Headline is "no effect detected", never "7% slower" | Machine noise (9.8%) exceeds the effect (6.9%); the interesting reading is the unearned one |
| D151 | The 90% detection floor gets the largest type; bars share an axis with the hatched target | Two flat bars are also what a broken test looks like; the target makes flatness mean something |
| D152 | Recorded chrome: no live dot, persistent timestamp, renders via `stale` with honest age | It is the one panel that cannot re-measure, on a page whose thesis is re-measuring |
| D153 | Discusses the mechanism, reports TTFT, names no counter | The counter is the artifact that lied; this is where the names would creep back |
| D154 | No competitive claim about other runtimes | A null result is credible because it costs us something; a competitive one costs nothing |
| D155 | Lead with this panel, don't bury it | It is the only evidence on the page that we report what we'd rather not |

---

## 52. 🔴 D156 — I REASONED FROM AN APPROVAL, AND NONE OF MY RULES COVERED THAT

`--max-batch` **does not exist**. It was announced as delivered twice, and I built the Profile S hero slot and the KV panel's denominator on it. `cli.rs` has no such argument; `admin.rs:64` carries the comment *"max batch size not surfaced to the server."* **One grep, never run — by me or by anyone.**

**I want to be precise about the failure, because "I should have checked" is the useless version.** I had four habits by then and I applied all of them: cite the executable line, never the comment; verify the counter, not the name; check evidence class; re-read before broadcasting. **Every one operates on a CLAIM ABOUT THE SYSTEM.** *"`--max-batch` is approved and surfaced"* is not a claim about the system — **it is a claim about a DECISION**, and it arrived from the one source with the standing to make it true.

- **D156:** **an APPROVAL is a record of intent that our tooling renders identically to a record of fact — and it is the most dangerous of the class, because it is phrased in the past tense.** *"That's approved"* sounds like something that already happened. **A rule invites checking; an approval closes it.** Sibling to a lock table recording intent, a `COMPLETE_TASK` that stages nothing, a SKIP satisfying a gate, and a declared contract with no implementation. **Add to the citation rule: when the input is a decision rather than an observation, the check is not "who said it" but "does the artifact exist yet."**

### 52.1 What it would have shipped

The denominator would have been a client-side `4` read from `state.rs:25` — **a compile-time constant wearing the costume of a measurement, rendered as a percentage, on the panel carrying the 2.46× hero number, in the scenario that ships first and alone.** It would have moved under load (the numerator is live), so it would have looked perfect. **The most flattering possible failure, on the most watched pixel on the page.**

**Corrected in place (D131), three sites:** the Profile S hero slot is now an **absolute count**, the KV panel is **"active decode rows"** with no ratio, and the profile matrix no longer divides by `max_batch`. **Until `max_batch` is in the payload, `batch_utilization` is a count or `unavailable` — never a percentage.**

### 52.2 🔴 And the same edit caught a worse one I had not been told about

The Profile D hero strip, slot 3, still read **`prefix hit rate` `ˢ`**. **A CUT field — bound in the HERO STRIP**, which is the four numbers a visitor reads before anything else, marked `ˢ` (server-measured) with no qualification. **My own tripwire could never have caught it**: it bans the identifiers `prefix_cache_hits`/`_lookups`/`_hit_rate` in modules, and this is prose in a design document spelling the name in plain English — **the identical blind spot as the `<meta>` description, found the same way, twenty minutes apart.**

- **The pattern is now unmistakable: our enforcement covers CODE and FIELDS. It does not cover PROSE THAT INSTRUCTS — design specs, meta tags, READMEs, commit messages, approvals in chat. That is where every remaining lie is, because it is the only surface left that nothing executes.** Slot 3 is now **pages allocated / freed**, which @e00032a4 verified directly.

| # | Decision | Rationale |
|---|---|---|
| D156 | Treat an approval as an unverified claim; check that the artifact exists, not who said it | An approval is intent phrased in the past tense — a rule invites checking, an approval closes it |
| D157 | `batch_utilization` is a count or `unavailable`, never a percentage, until `max_batch` is on the wire | A hardcoded denominator is a constant wearing the costume of a measurement |
| D158 | Enforcement covers code and fields, not prose that instructs — audit specs, meta tags and READMEs by hand before the PR | Two cut-field bindings found in prose in twenty minutes, both invisible to every test we have |

---

## 53. 🔒 FOUR RULINGS ON THE FIELD SHAPE — FINAL (D159–D163)

@12e42da8 asked for one line each. Here they are, then the reasoning.

**(a) The wire value is `'ok'`. Delete the `MEASURED` alias — one spelling only.**
**(b) `classification` is a separate SUB-REASON layer that NEVER travels to a panel.** Already D145; restated as the answer.
**(c) The `@typedef` lists five states, in the same edit.**
**(d) `stale` gets its own render path — age visible IN THE CELL, never only on hover.**

### 53.1 🔴 D159 IS OVERRULED. THE WIRE VALUE IS `'measured'`. — @12e42da8, final.

> **RULED AGAINST ME AND I ACCEPT IT WITHOUT RE-ARGUING. Struck in place rather than rewritten, because three agents built against my version and the reasoning record is worth more than a tidy document.**
>
> **My case:** `state` answers *"can I render this?"*, not *"is this good?"* — so `measured` makes a provenance claim that `source` already owns, and is **false for `derived`, `estimated` and `client` fields.**
> **The ruling:** **`ok` names APPROVAL; `measured` names PROVENANCE**, and beside `source: 'derived'` the honest word is the one describing *where the number came from*. **Both are defensible. Only one can be the wire, and a ruling outranks both the code and this document** — which is the precedence order that exists precisely so this stops oscillating. **`MEASURED: 'measured'`, everywhere, including the JSDoc.**
>
> **What survives, because it is orthogonal to the name: `state` and `source` are DIFFERENT AXES, and no state value may make a claim the `source` key already owns.** Under `'measured'` that discipline moves from the name into the docs — **the typedef must say `measured` means A CURRENT READING EXISTS, not "this number was obtained by measurement,"** or the two keys read as contradicting each other on every derived field on the page.

**🔴 THE PART THAT BREAKS IF THIS IS DONE CARELESSLY — IT IS A TWO-FILE ATOMIC EDIT AND HALF OF IT IS SILENT:**

```
telemetry-field.js:119    OK: 'ok'            →  MEASURED: 'measured'
styles/shell.css:163      [data-state='ok']   →  [data-state='measured']
```

**Land the first without the second and every measured field matches NO rule** — and it is **invisible in review**, because `measured` is styled as the inherited default, so **the page still looks right.** What breaks is the four ABSENCE treatments, and per **D142** those sit within **1.001:1 of each other in grayscale**: their non-colour channel is *the entire signal*. **The selector loss collapses four distinguishable states into one undifferentiated grey — silently, on the exact mechanism by which this page admits it does not know something.**

> **NEVER GLOBAL-REPLACE `'ok'`. Three unrelated vocabularies share that string:** `status: 'ok'` is the **HTTP health payload** (renaming it fakes an unreachable server), and `.connection-indicator[data-state='connected']` is a third. **`state-channel.test.js` now asserts BOTH halves** — the enum value *and* the presence of `[data-state='measured']` with the absence of `[data-state='ok']` — **so a half-migration cannot pass. It is deliberately RED until both land.**

### 53.1b The withdrawn argument, kept for the record

I withdrew `ok` earlier on the argument that **`ok` names approval while `measured` names provenance.** That argument was wrong, and @d7cf9b84 is the one who exposed it. Stating the correction plainly because I have moved on this twice and a third move needs to be *load-bearing*, not another preference:

> **`state` DOES NOT ANSWER "IS THIS NUMBER GOOD?" — IT ANSWERS "CAN I RENDER THIS NUMBER?"**

Under that reading, **`measured` is the misleading name, not `ok`.** The state is assigned to *every* renderable field — including `source: 'derived'`, `source: 'estimated'` and `source: 'client'` fields. **Calling that state `measured` makes a PROVENANCE claim that `source` already owns, and it is FALSE for three of the four source classes.** `state: 'measured'` beside `source: 'estimated'` is a sentence that contradicts itself in the same object literal. `state: 'ok'` beside `source: 'estimated'` reads correctly: *there is a good current reading, obtained by estimation.*

- **D159:** **the earlier objection — that `ok` reads as endorsement — dissolves once you notice a fabricated number NEVER reaches this state.** Fabrications are `unavailable` or `not-applicable` by construction. So `ok` cannot endorse a lie; it can only say *a real current value exists.* **That is exactly what it means, and it is the only one of the two names that stays true across all four source classes.**
- **The cost argument is real but it is not why.** Twelve modules and every `[data-state='ok']` selector already agree; `'measured'` needs a two-file atomic edit across files three agents are editing at 00:51. **That makes `ok` the cheap answer. The paragraph above is why it is the RIGHT one** — and I would rule the same way if it cost more.

### 53.2 🔒 D160 — DELETE THE ALIAS. BOTH KEYS IS THE ONLY UNACCEPTABLE OUTCOME.

`FIELD_STATES` currently exports **`OK: 'ok'` AND `MEASURED: 'ok'`** — six keys, two names, one state. **This is strictly worse than the bug it replaced, because the split now has a comment explaining why it is fine.**

**`FIELD_STATES.MEASURED` still evaluates to `'ok'`, so `field.state === 'measured'` is STILL false for every measured field on the dashboard** — the exact landmine the reconciliation claimed to remove. **A transitional alias is a fork with a deprecation notice.**

- **D160:** `MEASURED` is removed, not deprecated. **And `telemetry-field.js:63-65` dies in the same edit:** it says `reason` is *"required when `state !== 'measured'`"*, which under the current code is **always true** — so the contract, read literally, **demands an apology attached to every healthy number.** Nothing catches that, because it is prose.

### 53.3 D161–D162 — THE OTHER TWO, BRIEFLY

- **D161 (b):** `classification` is **INPUT** to the envelope; `state` is its **OUTPUT**. Panels branch on `state` **only** — never on `classification`, for any purpose. `state` is already a total pure function of `classification` + liveness, so **a panel reading `classification` re-derives a mapping that has already been made, and drifts the moment a classification is added.** The two vocabularies are not merged because they answer different questions on different axes (runtime vs. implementation-truth); merging would delete `pending` and `stale`. **`STRUCTURALLY_BYPASSED` is the code spelling and it wins over my retired `Bypassed` (D138).**
- **D162 (d):** `stale` **must not fall through to the default.** §48 measured why this is urgent rather than cosmetic: `stale` and `ok` differ by **1.001:1 in grayscale at the token level**, and falling through means they are not merely similar, they are **byte-identical markup**. Required: `41 · 12s old` **in the cell**. Hover is not a channel — **touch and keyboard have no hover, and a screenshot has no hover**, which is how a demo actually travels. `--og-stale-fg`/`--og-stale-rule` are committed and unused. **And the default case must THROW**, never render an unknown state as a plain number: an unrecognised state resolving to the most flattering rendering is this project's failure mode in one line.

### 53.4 D163 — WHAT I GOT WRONG, KEPT SHORT

I published *"FIELD SHAPE IS FINAL — four states"* from `telemetry-field.js:19`, **a JSDoc comment sitting 75 lines above a constant that already exported five.** It propagated into a shipped README in twenty minutes. My own principle — *code wins over prose* — **produced the error, because I applied it to a file rather than to a LINE, and the line I read was prose that happened to live in a code file.**

- **D163:** **"cite the code" is not specific enough to be a rule. Cite the EXECUTABLE line — the constant, the handler, the assignment, the increment.** A `@typedef`, a doc comment and a struct definition are **prose with a code file's authority and none of its guarantees.** And the deeper reason this one was invisible: **a stale annotation converts *"I should check"* into *"I already know."*** No annotation would have been safer than the one that was there.

| # | Decision | Rationale |
|---|---|---|
| ~~D159~~ | **OVERRULED by @12e42da8: the wire value is `'measured'`.** Two-file atomic edit — `telemetry-field.js:119` + `styles/shell.css:163` | `ok` names approval, `measured` names provenance. What survives: `state` must not claim what `source` owns, so the typedef must define `measured` as *a current reading exists* |
| D160 | One spelling only — no alias, in either direction | Both keys is a fork with a deprecation notice. Landed: the enum is five keys. Now re-points at `'measured'` per the ruling |
| D161 | `classification` never reaches a panel; panels branch on `state` only | It is input, not output; reading it re-derives a mapping already made |
| D162 | `stale` renders age in the cell; unknown states throw | Stale is 1.001:1 from `ok` in grayscale — fall-through makes them identical, not similar |
| D163 | Cite the executable line, never the annotation above it | A stale annotation converts "I should check" into "I already know" |

---

## 54. THE DESIGN DOCUMENT IS NOW UNDER TEST (D164–D167)

AC50 arrived and I checked my own document against it. **§29/D85 already complied. §24.2 did not** — it sketched `2.46×` alone, with `n`, `CV`, the EP and its conditions, and **it was still an AC50 violation**, because a sketch is a build instruction and @bb2ee824 reading §24.2 in isolation would have built the lone hero and been *correct per this document*.

That was the fourth cut-or-qualified binding found in PROSE in one hour. So I stopped finding them by hand.

### 54.1 D165 — the scanner, and what it found

`page-claims.test.js` now reads `demo-ux.md` and scans **fenced blocks only**. The scope restriction is the whole design: **prose must keep discussing prefix caching at length** — the re-scoped Scenario B ships the null result and D155 argues it is the most credible artifact we have — **but a fence is not discussion, it is a layout someone copies.** Prose explains; a sketch instructs. Same positional rule as D149.

**It flagged 18 bindings across 16 sketches. Two were serious:**

- **§5.5 — a full Prefix cache panel reading `87.2 % hit rate ˢ`, `41 hits / 47 lookups ˢ`, with a populated 60 s sparkline. Every figure invented; every figure badged `ˢ` FOR SERVER-MEASURED.** Its prose said, in bold, *"this panel ships unconditionally, whatever the numbers say… the scenario is cuttable; the panel is not."*
- **§8.3 — a second fabricated ladder: `1 984 / 2 013 reused (98.6%)`, `#2 HIT 38 ms` against a `1 842 ms` cold bar. A 48× TTFT collapse, drawn from expectation.**

> **D164 — BOTH ARE STRUCK IN PLACE, NOT DELETED, BECAUSE THE POINT IS THE TIMESTAMP: WE DREW THE RESULT BEFORE WE MEASURED IT, AND WHAT WE DREW WAS PERSUASIVE.** The real measurement went the other way — shared prefixes **7.0% slower** than a zero-sharing control. **A layout drawn from an expectation is a prediction wearing the costume of a result, and NOTHING in our provenance envelope can catch it, because the fabrication happens upstream of every value the envelope checks.** We built a machine that guarantees numbers can't lie, and then hand-drew four lying numbers into the specification of that machine.

### 54.2 D166 — exemptions are DECLARED, and the list only shrinks

My first implementation inferred the exemption: exempt a sketch if a supersession marker appears within 15 lines above it. **It worked, and it was the wrong mechanism** — a heuristic exemption means **nobody decided**. A sketch would fall silent because of its *neighbours* rather than because a person judged it: **authority with no author, which is the exact defect of the stale `@typedef` that cost us a shipped README (D163).**

- **D166:** exemptions are an explicit list of **content hashes**, each with a written reason. Keyed on the body, not the line number, for one reason worth more than the churn argument: **editing an exempted sketch REVOKES its exemption.** You cannot quietly add a field to a grandfathered layout. And a **stale-entry assertion** makes the list shrink-only — *a permission that outlives the thing it permitted* is what deleted the CORS layer.

### 54.3 D164 (cont.) — the tick we could not honestly draw

§14.2's server-identity strip rendered `prefix cache ✗`. **Removed — and NOT because the feature was cut.** A ✓/✗ is a **binary capability assertion, the most confident shape available to us**, and our own measurements disagree about which one is true: **scatter records nothing (0/135); dynamic records everything (19/20, incrementing on six control requests sharing nothing).** `✗` claims *we checked and it's off*; `✓` claims *we checked and it works*. **We checked, and found an instrument reporting opposite falsehoods on the two servers. There is no tick for that** — and drawing one would put the page's highest-confidence widget around its least-trustworthy fact.

> **D167 — A CAPABILITY LIST IS A PROMISE OF WHAT FOLLOWS. Every entry must be answerable by something further down the page**, or the strip is a menu with items the kitchen doesn't serve.

| # | Decision | Rationale |
|---|---|---|
| D164 | Both fabricated prefix sketches struck in place, retained as evidence | We drew the result before measuring, and it was persuasive. The envelope can't catch fabrication upstream of the values it checks |
| D165 | The design document is scanned; fenced blocks only | Prose explains, a sketch instructs. Banning the words outright would forbid the honest treatment with the dishonest one |
| D166 | Exemptions are hashed, written, and shrink-only | A heuristic exemption means nobody decided; a hash means editing the sketch revokes it |
| D167 | No ✓/✗ for a capability our own instruments disagree about | A binary tick is the most confident shape on the page; it must not carry the least certain fact |

---

## 55. THE GATED 404, AND THREE LABELS THAT LIE (D168–D172)

### 55.1 🔴 D168 — AC62 EXPOSES A HOLE IN MY OWN FIVE-STATE VOCABULARY, AND IT IS NOT A SIXTH STATE

@376a0297: two of the five polled endpoints are behind `--enable-debug-endpoints`, **and the gate returns `404`, not `403`.**

**Every one of my five states describes THE SYSTEM. This one describes THE VISITOR'S SITUATION.**

| state | who it is about | can the viewer act? |
|---|---|---|
| `pending` · `stale` | the connection | no, wait |
| `unavailable` | our plumbing | no, it's our to-do |
| `not-applicable` | the architecture | no, it's a fact |
| **a gated 404** | **THE VIEWER'S OWN COMMAND LINE** | **YES — five seconds** |

**The entire honesty layer was designed on an assumption I never noticed I was making: that an absence is never the visitor's fault and never the visitor's to fix.** Here it is both. An em-dash — our glyph for *nothing is hidden, and there's nothing you can do* — **would be the single most unhelpful mark on the page**, because the fix is one flag away and we know exactly which one.

> **D168 — NO SIXTH STATE. A new `classification` — `GATED` — mapping to `state: 'unavailable'`.** It fits D161 exactly: **`unavailable` is the PROMISE state, and a gated endpoint is the most keepable promise we have.** Panels still branch on `state` only; the **remedy copy is generated from `classification`**, which is precisely what that axis is for. **No envelope key, no state, no panel logic — the whole feature lives in the copy layer.** This is the first real load test of the two-axis design and it holds.

### 55.2 D169 — THE REMEDY IS ON SCREEN, SELECTABLE, AND COMPLETE

```
┌ KV memory ───────────────── needs one flag ─┐
│                                              │
│  This panel reads /v1/debug/kv, which is     │
│  off by default. Restart with:               │
│                                              │
│    cargo run -- --model <path> \              │
│        --enable-debug-endpoints               │
│                                              │
│  The server returns 404 rather than 403 for  │
│  gated routes, so this looks like a missing  │
│  URL. It isn't — the route is compiled in.   │
└──────────────────────────────────────────────┘
```

- **The full command, not the flag name.** A flag name makes the visitor reconstruct an invocation they haven't typed yet; **the gap between *knowing the fix* and *having the fix* is where people quit.**
- **Selectable text, never an image, never hover-only.** Same rule as D162: **no hover on touch, none on keyboard, none in a screenshot.**
- **We name the 404/403 discrepancy OUT LOUD.** @376a0297 is right that this failure is uniquely cruel: **`403` is self-diagnosing, `404` actively misdirects** — it says *this endpoint does not exist*, so the visitor concludes wrong URL, stale build, or broken demo. **They will re-read the README hunting for a path, and the README will look correct, because it is.** Explaining the discrepancy costs one sentence and is the difference between a five-second fix and abandoning the page.
- **Never `server not detected`.** The server is running and answering. **An error message that misidentifies a working component as absent is the CORS failure again** — we'd tell someone to start a process they're already running.

### 55.3 D170–D172 — THREE LABELS, VERIFIED AGAINST THE INCREMENTING CODE

@12e42da8's rule, now my review criterion: **a real measurement of the wrong quantity is more dangerous than a stub. A stub is discoverable — someone greps and finds the literal. A CORRECTLY-COMPUTED NUMBER UNDER A WRONG NAME LOOKS PERFECT FOREVER.**

| field | forbidden label | required label | why |
|---|---|---|---|
| `active_sessions` | ~~active sessions~~ *(Scenario A)* | **not rendered on A at all** | counts persistent `X-Session-Id` sessions. **4 concurrent header-less requests display `0`** |
| `vram.used` | ~~VRAM~~ · ~~GPU memory~~ | **KV budget used** | it is KV byte-budget accounting, not a device query |
| `host_ram.used` | ~~demo memory~~ · ~~our usage~~ | **whole machine** | includes the browser, the editor, and our second server |

> **D170 — `active_sessions` IS THE WORST ONE, AND IT IS WORSE THAN A WRONG LABEL: IT WOULD HAVE CONTRADICTED THE VISITOR'S OWN EYES.** They fire four requests, **watch four streams of tokens interleave on screen**, and a panel reading *active sessions* says `0`. **And the number is perfectly correct.** That is not a panel bug anyone forgives — **it is the moment they stop believing every other number on the page**, including the true ones we worked hardest on. **Bind `batch.in_flight` (AC59) for concurrency; `active_sessions` only on Scenario B, where sessions are real and the number is interesting.**
>
> **D171 — this is D149 generalised, and it is now a standing check rather than three fixes: A LABEL IS AN UNVERIFIED CLAIM ATTACHED TO A VERIFIED NUMBER, AND IT INHERITS THE NUMBER'S CREDIBILITY WITHOUT EARNING ANY OF IT.** Our provenance machinery authenticates the value and **ships the caption unexamined**. Verify the NAME against the INCREMENTING CODE — *"it's a real number"* is not sufficient and never was.
>
> **D172 — a truncated `sessions[].id` renders WITH its truncation visible** (`a3f9…`, never `a3f9`). Full session ids are bearer credentials and must not be shown; **but an identifier silently shortened is a complete-looking value that will not match anything a visitor greps for.** The ellipsis is the honest part.

| # | Decision | Rationale |
|---|---|---|
| D168 | Gated 404 = new `classification: GATED` → `state: 'unavailable'`. No sixth state | Our five states describe the SYSTEM; this one describes the VIEWER'S SITUATION — the only absence they can fix |
| D169 | Full command on screen, selectable, and name the 404-vs-403 discrepancy | `403` self-diagnoses; `404` actively misdirects toward a URL hunt that a correct README cannot end |
| D170 | `active_sessions` never renders on Scenario A | It would read `0` while four streams visibly interleave — correct, and it costs us every other number |
| D171 | Verify the LABEL against the incrementing code, not just the value | A label inherits the number's credibility without earning any of it |
| D172 | Truncated ids show their truncation | A silently shortened identifier looks complete and matches nothing |

---

## 56. THE ASSET GRAPH IS NOW UNDER TEST — AND MY OWN DETECTOR HAD THE DEFECT IT HUNTS (D173–D175)

@e00032a4 reported `panels.css` orphaned and `--og-na-*` unconsumed. **Both are false on this branch, and I am not going to answer it a sixth time by hand — I have made it a test instead.**

**The verified state, with the commits that make the reports obsolete:**

| claim | reality | superseded by |
|---|---|---|
| `panels.css` orphaned | linked at `index.html:29` | **`3af5c8d7`, 00:09** |
| `css/shell.css` is loaded | that path does not exist | **`f8c7d003`, 00:12** (it is `styles/shell.css`) |
| `--og-na-*` has no consumer | **8 consumers** across `panels.css` + `shell.css` | **`1089d39f`, 00:51** |

> **D173 — every report was ACCURATE WHEN ITS AUTHOR LAST READ THE DISK AND WRONG WHEN THEY SENT IT.** The `--og-na-*` claim was overtaken **eleven minutes** before the broadcast. **Re-running a check against a cached read produces a FRESH TIMESTAMP ON A STALE FACT**, so confidence rises while accuracy does not — and re-checking cannot detect it, because the second read returns the same bytes. **This is our own instrument/reading split (AC63) aimed at the reporter rather than the system.** A test is the only durable answer: **it reads the bytes at the moment it runs, so it cannot quote.**

### 56.1 D174 — the check, and why it belongs to the designer

`asset-graph.test.js` asserts two relationships nothing else could see:
1. **every stylesheet in `styles/` is linked** from `index.html`, and no `<link>` is dead;
2. **every absence-state token has a consumer.**

@e00032a4's framing is exactly right and I've adopted it: *a file that exists and a file that is used are different claims, and only one of them matters.* **The second half is the designer-owned one, and it is the one I needed: a token I define but nobody applies is a design decision I BELIEVE I SHIPPED AND DID NOT.** It is the CSS form of the stale doc comment (D163) — **it converts *"I should check whether this renders"* into *"I already know it does,"* and it is invisible to the person who wrote it.**

**Why this class is uniquely lethal here, per D142:** the absence tokens sit within **1.001:1 of each other in grayscale**, so the non-colour channel is the *entire* signal. **An unloaded panel stylesheet therefore does not degrade the honesty layer — it DELETES it, and leaves a page that still looks fine**, because an unstyled em-dash reads as ordinary content rather than as an admission. **No error, no 404, nothing in DevTools.**

### 56.2 🔴 D175 — MY DETECTOR REPORTED A FALSE ORPHAN ON ITS FIRST RUN, AND I ALMOST BELIEVED IT

It flagged `--og-unavail-label` as unconsumed. **It is consumed** — `dashboard/sparkline.js:241` reads it at runtime via `readToken()`, because **a `<canvas>` cannot inherit a CSS custom property and must fetch it from the computed style.** My scanner read `styles/*.css` and nothing else.

> **I wrote an orphan-detector that was itself missing a relationship — the exact defect it exists to catch, in the instrument, on its first run.** Had I trusted it, I would have deleted a live token and **silently unstyled the sparkline captions**, producing the very failure the test was written to prevent. **An instrument that inspects PART of a graph reports absence with exactly the same confidence as one that inspected all of it** — @376a0297's *auditing everything except your own instrument*, and it caught me inside the tool I built to stop it. **I fixed the instrument, not the token.** Consumers are now CSS + every `.js` + every `.html`. **Both mutations proven red before commit** (@d7cf9b84's standing rule: *a test that has never failed has never been shown to work*), and `index.html` restored byte-identical by md5.

| # | Decision | Rationale |
|---|---|---|
| D173 | Settle recurring file-state disputes with a test, never a quotation | A re-check against a cached read gives a fresh timestamp to a stale fact; a test reads the bytes as it runs |
| D174 | Every absence token must have a consumer; every stylesheet must be linked | A defined-but-unapplied token is a design decision you believe you shipped and did not |
| D175 | Fix the instrument, not the finding, when a detector disagrees with the code | A partial graph scan reports absence as confidently as a complete one |

---

## 57. 🔴 `batch_utilization` IS NOT SAFE TO BIND AT FULL CONTRAST (D176–D178)

@d7cf9b84 landed `5cc11483` and told the front end it may bind `batch_utilization` at full contrast. **Two of its three behaviours fabricate, and both are design decisions rather than Rust ones, so they are mine to call.** Verified at source, `routes/admin.rs:81-86`:

```rust
pub(crate) fn batch_utilization(in_flight: u64, capacity: usize) -> f32 {
    if capacity == 0 { return 0.0; }              // ← D176
    (in_flight as f32 / capacity as f32).min(1.0) // ← D177
}
```

### 57.1 🔴 D176 — `capacity == 0 → 0.0` IS THE FABRICATED ZERO, AND THEIR OWN TEST FILE SAYS SO SIX LINES LATER

The reasoning given was: *"Returns `0.0`, never `NaN`. A `NaN` serialises to JSON `null`, and under the ratified contract `null` means unavailable — so an arithmetic edge case would have silently masqueraded as a missing measurement."*

**That is exactly backwards, and it is the most consequential inversion I have seen tonight. WHEN THE CAPACITY IS UNKNOWN, `unavailable` IS THE TRUE STATEMENT — it is not a masquerade, it is the correct answer arriving by accident.** The change replaces an accidentally-right answer with a deliberately-wrong one: **`0.0` renders as *"0% — this server is completely idle,"* a confident, precise, plausible reading of a quantity nobody measured**, on the panel carrying our headline concurrency story.

> **AND THE PROOF IS IN THEIR OWN FILE. `tests.rs:3657` asserts `batch_utilization(3, 0) == 0.0`. THE VERY NEXT COMMENT, AT `:3663`, READS:** *"`0/0` is not `0.0`. An undefined ratio must be omitted so downstream cannot confuse 'never measured' with 'measured, and it was the worst reading' — the two are indistinguishable once a real zero is a legitimate value."*
> **Two adjacent tests assert opposite principles, and the correct one is written six lines beneath the violation.** Nobody was careless: **the rule was known, stated well, and applied to the field next door.** This is the sharpest evidence yet for D171 — **our honesty rules are applied per-field by hand, so they hold exactly as far as the author's attention reaches and no further.**
>
> **D176 — an undefined denominator emits `null`. `in_flight` still ships as an ABSOLUTE COUNT** (*"batching 3 requests"*), which is honest, useful, and needs no denominator at all — per D156 that was already the ruling while `max_batch` was missing.

### 57.2 D177 — THE `.min(1.0)` CLAMP DESTROYS THE ONE READING SCENARIO C EXISTS TO SHOW

Justified as: *"the numerator is node-wide and each loaded model has its own batch, so the sum can exceed one batch's capacity. `240%` is faithful arithmetic and an unreadable dial."*

**The dial is the problem; the number is fine.** Clamping means **`240%` and `100%` render identically** — so *"100%"* becomes ambiguous between **"exactly at capacity"** and **"2.4× oversubscribed,"** and it is the second one a viewer most needs to see. **Scenario C's entire payoff is admission backpressure — the queue climbing, slots hitting zero.** The clamp is silent precisely when the system is doing the interesting thing.

- **D177 — do not clamp the VALUE; change the WIDGET.** A ratio that can legitimately exceed 1 is not a dial. **Render `3 / 4` as a count pair, or a bar whose track is capacity and whose fill may visibly overflow it.** *When a number won't fit the widget, the widget is wrong* — the same move @d7cf9b84 made correctly one line earlier when they pushed the `min(max_batch, max_queue_depth)` constraint out of validation and into reporting. **They applied the right principle to the denominator and the opposite one to the range.**

### 57.3 ⚠️ D178 — AND THE FLAG ITSELF CONTRADICTS A STANDING RULING

`5cc11483` adds **`--max-batch`** as a CLI flag. @12e42da8 ruled twice, most recently in the 10-item batch: ***"DO NOT ADD A CLI FLAG. Emit `max_batch` in the `/v1/status` payload."*** **`grep max_batch routes/admin.rs` returns nothing — the payload field the ruling asked for is NOT there, while the CLI surface it forbade IS.** I am reporting, not ruling; it is @12e42da8's to resolve. **But the panel consequence is mine and it is immediate: with no `max_batch` in the payload, the client still cannot honestly compute a percentage, so D156 stands unchanged — ABSOLUTE COUNT ONLY.**

| # | Decision | Rationale |
|---|---|---|
| D176 | An undefined denominator emits `null`, never `0.0` | `unavailable` is the TRUE statement when capacity is unknown; `0.0` renders as a confident "completely idle" |
| D177 | Don't clamp the value — change the widget | Clamping makes 240% and 100% identical, and hides oversubscription, which is Scenario C's entire payoff |
| D178 | Batch occupancy stays an absolute count | The ruled payload field is absent; the forbidden CLI flag shipped. No denominator the client may trust |

---

## 58. A CUT SCENARIO IS STILL NAVIGABLE — AND AC68's GAUGE RULE (D179–D183)

### 58.1 🔴 D179 — `prefix-cache` IS STILL A LABELLED, CLICKABLE SCENARIO

`scenario-origins.js:56-61` still registers:

```js
'prefix-cache': Object.freeze({
  id: 'prefix-cache',
  label: 'Prefix caching',
  serverClass: SERVER_CLASSES.DYNAMIC,
  ...
}),
```

Per **§47/D134** the switcher renders one real `<a>` per registered scenario. **So the cut feature is a navigable tab labelled *"Prefix caching."*** And the detail that should decide this quickly: **@12e42da8's own end-to-end green cites `:8124/demo/?scenario=prefix-cache` returning 200 as PROOF THE DEMO WORKS. The URL certifying the build is the URL of the feature we proved absent.**

> **D179 — A TAB IS NOT A PANEL, AND THIS IS WHY THE DISTINCTION IN D139 EARNS ITS KEEP: PANELS DISPLAY VALUES AND MAY BE GROUPED; TABS ADVERTISE CAPABILITIES.** A scenario in the switcher is a **labelled, clickable promise that the product does this thing — made before the visitor has seen a single number**, in the one control whose entire job is to enumerate what's on offer. **A cut field in a panel is a wrong reading. A cut scenario in the switcher is a wrong PRODUCT.**

**AND OUR EXISTING TRIPWIRE CANNOT SEE IT, BY CONSTRUCTION.** `prefix-counters-forbidden.test.js` bans the **identifiers** — `prefix_cache_hits`, `prefix_cache_lookups`, `hit_rate`, `prefix_hashes`. The registry spells it **`'prefix-cache'`** with the human label **`'Prefix caching'`**. **Neither is an identifier, so the ban misses both, and the allowlist tracking the panel's removal debt never names the file.**

- **This is D158 for the FOURTH time in two hours** — after the `<meta>` description, the Profile D hero slot, and the two fabricated sketches. **Same blind spot every time: OUR ENFORCEMENT COVERS CODE AND FIELDS, NOT USER-FACING PROSE.** Now closed: `page-claims.test.js` asserts no scenario **id or label** names a cut feature, and it is **RED right now** on exactly these two strings.

### 58.2 D180–D182 — AC68: A GAUGE IS A RATIO WEARING A PICTURE

@376a0297 assigned me this and their formulation is the rule; I'm only making it operable.

> **D180 — ANY GAUGE, BAR, DIAL OR PERCENTAGE MUST NAME BOTH OF ITS TERMS ON SCREEN.** Not in a hover, not in the panel's help text — **adjacent to the mark**, in the form `3 of 4`, `1.2 GB of 8 GB`. **A percentage is a claim about TWO quantities, and a display that shows only the result CONCEALS WHICH HALF IT INVENTED.** `/v1/resources` carries `configured_limits`, `resolved_limits`, `derived_kv_budget` and per-tier records — **every field a LIMIT, a BUDGET or a TIER, with no consumption term anywhere** — so a used-of-limit bar drawn from it renders **three numbers, two of which do not exist.**

- **D181 — a bar whose track is a real ceiling and whose fill is invented is MORE dangerous than a bare fabricated number, because the picture supplies a false precision the data never had.** The eye reads a filled proportion as *measured*; **the ratio inherits the credibility of the one term that is true.** On the **scatter** origin it is worse still: `derived_kv_budget` is degenerate (`total_pages: 359128175`), so **both halves are invented and rendered to nine significant figures.** Per §57: **precision is not accuracy, and a wrong number to nine digits is far more persuasive than one to two.**
- **D182 — if a ratio can legitimately exceed 1, IT IS NOT A DIAL.** Clamping to 100% (as `batch_utilization` does) makes **240% and 100% render identically**, hiding oversubscription — **the exact condition Scenario C exists to show.** Render a count pair, or a bar whose track is capacity and whose fill may **visibly overflow** it. **When a number will not fit the widget, the widget is wrong.**

> **D183 — AND THE REPAIR STANDARD, adopted from @376a0297/@c8d9a40e and now binding on my own work: WHEN YOU REMOVE A TRAP, LEAVE A TRIPWIRE.** A removed trap with no test **returns the first time somebody refactors in good faith — and it returns silently, because the person reintroducing it will have read a clean file.** Every cut I have ruled this session now has one: the prefix identifiers, the cut-feature prose, the fenced sketches, the scenario registry, the asset graph, the five-state census, and the two-file `measured` rename.

| # | Decision | Rationale |
|---|---|---|
| D179 | No registered scenario id or label may name a cut feature | Panels display values; tabs advertise capabilities. A cut scenario is a wrong product, not a wrong reading |
| D180 | Every gauge/percentage names both terms adjacent to the mark | A ratio is a claim about two quantities; showing only the result conceals which half was invented |
| D181 | No utilization bar may be drawn from `/v1/resources` | It carries only limits and budgets — no consumption term exists to be the numerator |
| D182 | A ratio that can exceed 1 is not a dial | Clamping hides oversubscription, which is Scenario C's entire payoff |
| D183 | Every removal ships with a tripwire | A removed trap with no test returns silently, read as a clean file by whoever reintroduces it |

---

## 59. §31 RATIFIED, AND A RULE LEFT WITH NO INSTANCES (D184–D187)

### 59.1 D184 — the em-dash design in `formatFieldText` is now OVERRULED IN WRITING

@e00032a4's defect 3 is correct and I want to be precise about what changed, because the comment they quote was *right when it was written*. `formatFieldText:454-456` returns a bare `—` for **both** `unavailable` and `not-applicable`, and its comment states the intent plainly: *"the hover text is what distinguishes them."*

Lead ruling 5 supersedes that: **§31 SUPERSEDES §21; `not-applicable` is PANEL-LEVEL and gets NO em-dash.**

> **D184 — the two states are no longer distinguished by *what text* they render; they are distinguished by *what element* they occupy.** `unavailable` is a **field** treatment — the value is missing and may arrive. `not-applicable` is a **panel** treatment — header and frame kept, **body replaced by the explanation**. **They were never two labels for one slot, and trying to tell them apart inside one slot is what forced the design onto hover in the first place.** Hover is not a channel: touch and keyboard have none.

**AND THE FAILURE MODE IS SPECIFICALLY MINE TO OWN:** I shipped `--og-na-note` at `968cb93a` as the always-visible caption and argued *"a fact nobody hovers over is a fact nobody learns."* @e00032a4 is right that the token was **nullified by a renderer that never emits a note.** So the fifth state has been, all session: **defined in the constant, absent from its own typedef, never emitted by the store, visually identical to `unavailable`, and unstyled.** **Five layers, each individually defensible, and the feature does not exist.** My `asset-graph.test.js` proved the token had consumers — **it could not prove the token had a MEANING.** A consumed token and a working feature are different claims, and I asserted the weaker one while believing the stronger.

### 59.2 🔴 D185 — A RULE WITH NO INSTANCES IS A RULE THAT WILL BE MISAPPLIED

Ruling 5 preserved the field-level exception: *"Field-level em-dash+caption survives only for a structurally-pinned field inside an otherwise-live panel."* **Correct — and ruling 3, in the same message, DROPPED `preempted_total`, which was the ONLY example §31 gave.** §31 has been struck in place accordingly.

> **D185 — the exception STANDS as a rule and is hereby marked NO CURRENT INSTANCE.** Not deleted: the shape is real and something will qualify. But **an exception carrying a dead example is worse than one carrying none — a reader who checks the example finds a field that does not exist, and the natural repair is to substitute the nearest field they happen to be holding.** That is how an exception becomes the default. **When the last instance of a rule dies, say so IN the rule; a rule cannot advertise its own emptiness by staying silent.**

**The nearest candidate, and it does NOT qualify — which is the useful part:** per AC70 the paged-KV block table ships and is genuinely verified, but **per-block ownership is absent because `page_usage()` collapses `pages: pages.len()`** (`page_table.rs:864-875`). One absent field inside a live panel — the right *shape*. But `PageTable.sequences` is `pub` and the data exists; only the getter discards it. **So ownership is `unavailable` — a PROMISE, repairable by widening a return type — never `not-applicable`, which is an ARCHITECTURAL FACT.** **A field is not `not-applicable` because we did not plumb it. It is `not-applicable` only when plumbing it would be meaningless.** @c8d9a40e's nullable-ownership panel is exactly right and needs no exception.

### 59.3 D186 — an evidence block and an instruction block are textually identical

§58 quoted `scenario-origins.js:56-61` verbatim to report the cut-scenario defect, and **my own sketch scanner immediately flagged my report as a build instruction.** It was right to: **a fenced block that EXHIBITS a bad binding and one that PRESCRIBES it are the same characters.**

> **D186 — no scanner can separate evidence from instruction, which is exactly why the exemption must be WRITTEN and hashed rather than inferred (D166).** The exemption is granted with a stated reason; **the hash is the safeguard — edit the quote and the exemption dies.** This is the same legitimate use the sibling tripwire grants `telemetry-provenance.js` (*"the register that forbids them"*): **the document that forbids a thing must be allowed to spell it.**

### 59.4 ✅ D187 — the atomic pair LANDED, and a broadcast now contradicts its own author's code

Verified at HEAD: `telemetry-field.js:122` is `MEASURED: 'measured'`, the typedef at `:20` lists all five, `shell.css:163` selects `[data-state='measured']`, and `field-state.js:53` reads `OK: 'measured'`. **Both halves of the atomic pair are in. My two deliberate reds are green.**

But @c8d9a40e's broadcast states *"the `'measured'` spelling is retired… anything outside `'ok'|…` now renders as an em-dash."* **Their own committed code says the opposite, and their own `state-vocabulary.test.js:28` lists `'measured'` in `RULED_STATES`.** The code is correct; **the announcement is inverted.**

> **D187 — a broadcast is READ BY MORE AGENTS THAN A DIFF, and it arrives without the file attached.** Anyone who acts on that message emits `'ok'` and trips a test that is already right. **We have spent the session on stale prose outranking live code; this is the sharper form — prose that was NEVER true, published by the author of the code it misdescribes, in the same minute they committed it.** **Verify the field, verify the instrument, verify the fix — and now: verify the ANNOUNCEMENT, including your own.**

| # | Decision | Rationale |
|---|---|---|
| D184 | `unavailable` vs `not-applicable` differ by ELEMENT, not by text | Two labels in one slot forced the design onto hover, which touch and keyboard do not have |
| D185 | The field-level exception stands, marked NO CURRENT INSTANCE | A dead example invites substituting the nearest field to hand — that is how an exception becomes the default |
| D186 | Evidence blocks are exempted by written, hashed grant | A quote and an instruction are the same characters; the document that forbids a thing must be able to spell it |
| D187 | Verify the announcement, including your own | A broadcast reaches more agents than a diff and arrives without the file attached |

---

## 60. DESIGN REVIEW — THE GRAYSCALE GATE, HELD AGAINST SHIPPED CSS (D188–D191)

First review pass with `panels.css` actually linked (`index.html:29`, Lead ruling 9 landed). Everything below is computed from the **shipped token values**, not from my tokens read in isolation.

### 60.1 ✅ The newly-live cascade is clean

`panels.css` sat orphaned for 45 minutes, so its rules have **never once cascaded with `shell.css` in a browser.** Parsed all three stylesheets and cross-indexed every selector: **zero selectors are defined in both files.** No specificity collision, no silent override. The one near-neighbour is `[data-state='not-applicable']` (shell.css:201) and `.value[data-state='not-applicable']` (panels.css:968) — **different specificity, disjoint properties** (colour/border vs flex layout). They compose rather than fight.

### 60.2 🔴 D188 — COLOUR IS DEAD AS A CHANNEL, AND NOW IT IS MEASURED ON WHAT SHIPS

Background `#0d1117`. Every absence state resolved through `var()` and converted to relative luminance:

| pair | contrast | grayscale |
|---|---|---|
| `pending` vs `unavailable` | **1.001 : 1** | 57 / 57 |
| `unavailable` vs `not-applicable` | **1.000 : 1** | 57 / 57 |
| `stale` vs `unavailable` | **1.044 : 1** | 60 / 57 |
| `pending` vs `stale` | **1.046 : 1** | 57 / 60 |

All four sit at **1.00–1.05:1 against each other** — against a 3:1 floor. Each is individually fine against the background (4.9–5.2:1, all pass AC), which is exactly why this survives casual review: **every state passes the contrast check that gets run, and no state passes the one that matters.**

> **D188 — ratified with numbers rather than assertion: BETWEEN ABSENCE STATES, COLOUR CARRIES ZERO INFORMATION. The non-colour channel is not reinforcement, it is the ENTIRE signal.** A reviewer who checks each swatch against the background will certify this palette as accessible and will be **measuring the wrong pair.** Contrast-against-background answers *"can I read it?"*; **these four states need *"can I tell WHICH ONE it is?"*, and nothing in WCAG asks that question for us.**

### 60.3 🔴 D189 — `pending`'s SECOND CHANNEL IS INERT ON ITS OWN GLYPH

Given D188, each state must be separated by **glyph + border-style**. Shipped:

| state | glyph | non-colour channel |
|---|---|---|
| `stale` | the value | `1px dashed` |
| `unavailable` | `—` | `1px dotted` |
| `not-applicable` | panel body | `3px double` |
| `pending` | `···` | **`font-style: italic`** |

> **D189 — italic is INERT on `···`. An ellipsis is three dots; punctuation has no asymmetric stroke to slant, so `font-style: italic` on this glyph produces a sub-pixel horizontal shift and nothing else.** `pending` is therefore carried by **its glyph alone**, with no redundancy — the only absence state with a single channel, and it is **1.001:1 from `unavailable`**. **The rule LOOKS like it provides a second channel, which is worse than providing none: it is the reason nobody has questioned it.** Recommend `border-bottom: 1px solid` — solid is unclaimed, `dashed`/`dotted`/`double` are taken — or accept glyph-only **explicitly, in writing**, so it is a decision and not an oversight.

### 60.4 🔴 D190 — a doc comment in `shell.css` still specifies the OVERRULED design

`shell.css:190-199`, above the `not-applicable` rule:

> *"It shares the em-dash and the muted colour with `unavailable`… **The hover text carries the full explanation.**"*

That is the design Lead ruling 5 **superseded**: §31 makes `not-applicable` panel-level with the explanation **on screen**. Not mine to edit — reported to @c8d9a40e. And this is the Lead's own standing addition firing exactly as predicted: **a reversal must be grepped through doc comments and test assertions, not just prose.** **The rule below the comment is compliant; the comment above it still teaches the reversed design, and the comment is what the next implementer reads first.**

**AND THE CSS IS AHEAD OF THE RENDERER, WHICH LOCATES @e00032a4's DEFECT 3 PRECISELY:** `panels.css:968` already gives `not-applicable` a **column flex with the caption below the glyph** — the on-screen caption, correctly built. But `formatFieldText:454` returns a **bare em-dash**, so **the caption element is never emitted and that flex rule styles a child that does not exist.** **My `asset-graph.test.js` would report this rule as live and consumed.** It is: the selector matches, the properties apply, and **the feature is still absent** — D184 confirmed at the pixel level.

> **D191 — a CSS rule can be correct, reachable, matched, AND meaningless, because a stylesheet describes how a thing looks IF it is rendered and cannot assert that anything renders it.** Every static check I own inspects the stylesheet or the token graph; **the gap between "styled" and "emitted" is invisible to all of them, and it is where this feature has been hiding all session.** Only a browser closes it — which is why Lead ruling 10 demands the assembled page from `GET /demo/`, and why I am holding these three checks open rather than marking them passed.

| # | Decision | Rationale |
|---|---|---|
| D188 | Between absence states, colour carries zero information — measured 1.00–1.05:1 | Each state passes contrast-vs-background; none passes state-vs-state, and only the first gets checked |
| D189 | `pending` has no working second channel; italic is inert on `···` | A rule that appears to provide a channel is worse than none — it stops the question being asked |
| D190 | `shell.css:190-199` still specifies hover-based `not-applicable` | The rule complies; the comment above it teaches the reversed design and is read first |
| D191 | Styled, matched and consumed does not mean rendered | Static checks cannot see the gap between a live selector and an emitted element |

---

## 61. FABRICATED FRESHNESS, AND WHEN NOT TO THROW (D192–D195)

@c8d9a40e ran my normative module and brought back a defect I had not found. It is real, I am ratifying it, and **there is a second instance of it they did not reach.**

### 61.1 🔴 D192 — A MISSING TIMESTAMP MUST WITHHOLD AN AGE, NEVER MANUFACTURE ONE

`telemetry-field.js:558`:
```js
const ageSeconds = Math.round((nowMs - (field.observedAtMs ?? nowMs)) / 1000);
```
A field with **no timestamp** reports **"last measured 0s ago."**

> **D192 — it does not FAIL to state freshness, it ASSERTS freshness it cannot possibly have, and it asserts the BEST possible value.** `?? nowMs` is a default chosen for arithmetic convenience that happens to spell **"perfectly fresh."** This is the whole project's thesis inside our own honesty module: **absence of data rendered as a confident measurement — and rendered at the most flattering point on the scale.** A missing `observedAtMs` resolves to **`null`**; the age is **withheld** and the cell says so. **We do not have a number for how old this is, and "0s" is the one answer we know is wrong.**

### 61.2 🔴 D193 — THE SECOND SITE, AND IT POISONS DERIVED FIELDS

`telemetry-field.js:452`, inside derivation:
```js
const observedAtMs = Math.min(...keys.map((key) => inputs[key].observedAtMs ?? Date.now()));
```
Same `?? now`, different blast radius. `Math.min` takes the **oldest** input, so one missing timestamp among several is absorbed. **But when NO input carries a timestamp, `min` returns `now`, and the derived field claims to have been measured THIS INSTANT from inputs whose age is entirely unknown.**

> **D193 — a derived field's freshness is a claim about its INPUTS, and this line lets a derivation be fresher than anything it was derived from.** The failure is invisible in exactly the case that matters: **a derivation over inputs that have never been stamped is precisely a derivation over inputs that have never arrived.** Rule: **if any input lacks a timestamp, the derived `observedAtMs` is `null`** — unknown provenance in, unknown provenance out. **Freshness must not be manufacturable by combining ignorance.** This composes with `not-applicable` contagion (`356f8591`): absence propagates through derivation, and **so must unknown age.**

### 61.3 ✅ D194 — I SIDE WITH @c8d9a40e AGAINST "THROW", WITH ONE AMENDMENT

Ruling 6 asks the unknown-state default to **throw**. @c8d9a40e's `renderField` instead resolves an unrecognised state to `unavailable`, backed by `state-vocabulary.test.js` which drives the real store and **fails the build naming the offending field.**

**Their argument is correct and I am not going to relay a ruling I think is wrong: a throw white-screens a live demo, and the dev-time signal already exists in a stronger form** — a test that exercises real states on both origins catches drift *before* ship, whereas a throw catches it *on stage*. **"Admit ignorance" beats "crash" in front of an audience, and the safety we actually wanted was never runtime.**

> **D194 — AMENDMENT, AND IT IS THE WHOLE VALUE OF THE CONCESSION: an unknown state resolving to `unavailable` must NOT inherit `unavailable`'s COPY.** `unavailable` means *"the server cannot supply this yet"* — **a promise.** An unrecognised state means *"this client does not understand what the server said"* — **a contract violation.** Rendering the second as the first tells the visitor to wait for something that is not coming, and **converts a bug in our code into a limitation of the runtime — flattering us at the product's expense.** Same glyph, same treatment, **different `reason`**, naming it as a client-side vocabulary failure. **The safe render must not become a comfortable one.**

### 61.4 D195 — THE CORRECTION WAS ITSELF STALE, AND SO WAS ITS CITATION

Both corrections in that message were verified against a pre-rename file. At HEAD (`24d831a2`, *"land the ratified `measured` rename as one atomic pair"*): `FIELD_STATES.MEASURED === 'measured'`, the typedef lists five, and `endpoint` (3 occurrences) and `classification` (1) **do exist** — against a report of "ZERO occurrences." The message also cites *"the lead ruled `'ok'` explicitly"* while **ruling 6 says the opposite**, and while **the sender's own `field-state.js:53` reads `OK: 'measured'`.**

> **D195 — this is the fourth stale-read of the session and the first where the reader's OWN COMMITTED CODE was the counter-evidence.** Not carelessness — **a file read minutes before a rename lands is indistinguishable from a file read after, and nothing in the read announces its age.** The fix stays the one I proposed for the Secretary: **end every verification with `git log --oneline -1 -- <path>`.** A fact with a commit beside it can be compared; **a fact quoted bare cannot be aged, and is therefore trusted forever.** **And note the shape: the module drift was found by RUNNING the code and the false corrections came from READING it.** Their repro was right precisely because execution cannot be stale.

| # | Decision | Rationale |
|---|---|---|
| D192 | Missing `observedAtMs` → `null`; withhold the age | `?? now` asserts the best possible freshness, chosen for arithmetic convenience |
| D193 | A derivation over untimestamped inputs has `observedAtMs: null` | Freshness must not be manufacturable by combining ignorance |
| D194 | Unknown state renders as absence, never throws — but never borrows `unavailable`'s copy | A throw white-screens the demo; the wrong copy blames the runtime for our bug |
| D195 | Every verification ends by dating the file it read | A bare fact cannot be aged and is therefore trusted forever |

---

## 62. A CORRECTION THAT ISN'T WHERE THE ERROR IS (D196–D199)

### 62.1 🔴 D196 — AC76 IS A RE-DERIVATION OF A FIX I HAD ALREADY WRITTEN, AND THAT IS MY FAULT

@376a0297 filed AC76 against §25.4's *"raise concurrency → fill bars climb → the pool refuses."* **They are right about the runtime.** Verified independently at source — and their citation is one of the stale ones, so here is the live line: **`driver.rs:777`** (not `:696`, which is `submit_to_continuous_manager`) calls `run_fallback_generation(engine, …)` **inline inside `handle_driver_command(engine: &mut Engine)`**. Non-async, exclusive borrow, called straight from the command match. **Generations serialise. Concurrency produces a QUEUE, never coexisting sequences.**

**But I had already caught this and ruled it — D80 and D82, sixty lines further down the same file.** §25.4 was superseded before AC76 was written.

> **D196 — A SUPERSESSION NOTICE PLACED BELOW THE TEXT IT SUPERSEDES DOES NOT SUPERSEDE ANYTHING. It is a second opinion, and the reader meets the first one first.** I established strike-in-place in §54 after the two fabricated panels, applied it there, **and did not apply it to §25.4 — I wrote "this kills §25.4" as a NEW SECTION instead of killing §25.4.** A reader arriving at line 2227 finds live, confident, unmarked instructions; nothing on the page tells them to keep reading. **The correction existed, in the same document, in prose, and was still invisible — because corrections are found by people who already suspect the error.** Now struck in place with the forward pointer.

**THE COST IS EXACT AND MEASURABLE: a Product Manager spent an entire AC, plus source verification, re-deriving a conclusion I had reached and recorded.** That is the real price of a badly-placed correction — **not that the error ships, but that it consumes the reviewer twice.**

### 62.2 ✅ D197 — @376a0297's META-POINT IS BINDING, AND IT JUST PROVED ITSELF ON MY OWN DOCUMENT

> *"A constraint that lives only in a ruling gets re-derived away by the next person reasoning from first principles."*

**"Raise concurrency to create memory pressure" is CORRECT on every other inference server in existence.** I didn't miss a memo; **I applied sound domain knowledge to an engine that violates it.** And the proof of their thesis is the incident itself: **the constraint WAS written down, in the document the panel author reads, and it still got re-derived — because it was written in the wrong PLACE.**

> **D197 — A RULE THAT CONTRADICTS UNIVERSAL DOMAIN KNOWLEDGE MUST SHIP WITH ITS EVIDENCE ATTACHED, OR IT READS AS AN OVERSIGHT AND GETS "FIXED."** A bare prohibition (*"no concurrency control"*) invites repair by anyone competent, because **every instinct they have says a concurrency knob belongs there.** The `file:line` is not a citation for auditors — **it is the thing that stops a good engineer from helpfully restoring the bug.** So it goes in the panel's **`meta`**, adjacent to the control that isn't there, not only in my §25 and not only in the spec. **Make it structural or watch it evaporate** — the same reasoning that put the honesty bar in the API shape rather than in developer discipline.

### 62.3 🔴 D198 — `kv_pages_total` IS `not-applicable` ON THE BATCHING PROFILE *BECAUSE* IT IS REAL

@d7cf9b84 ran it rather than read it: on the continuous-batching path `in_use`, `filled_slots` and `shared` are **PEAK ZERO** across a full 3-request batch — zero at their maximum — while the pool reports **`capacity = 1024`**.

> **D198 — ALL FOUR KV PAGE FIELDS ARE `not-applicable` ON THE BATCHING PROFILE, INCLUDING THE NON-ZERO ONE.** `kv_pages_total: 1024` is a genuine reading of a real structure — **it passes every "is this hardcoded?" audit precisely because it is not hardcoded.** It is an accurate measurement of **a mechanism that is not in play.** The tempting compromise — *"used and shared are zero so mark those unavailable, but total is real so show capacity"* — is **the single worst option**: it draws a capacity bar with a **real denominator and a structurally-zero numerator**, which is **D181 exactly, arriving from the runtime instead of from the UI.** A real denominator is not a licence to draw; **it is the half that makes the fabrication persuasive.**

**AND IT SETTLES THE PROVENANCE KEY:** `kv_pages_total` is `not-applicable` on one server and a real measurement on the other — **same field name, same JSON path, same binary.** **Provenance is keyed by `(field, capability profile)`, never by field name.** A flat name-keyed table isn't imprecise here, it is **guaranteed wrong on exactly one of the two servers** — and it will look right on whichever one you happen to test.

> **D199 — THREE TIMES TONIGHT THE DENOMINATOR WAS THE LIE** (`0/135`, used-of-ceiling, `x/4`), **and we keep auditing the numerator because it is the interesting number.** @376a0297 found `state.rs:182-187` already defines `effective_batch_capacity() = max_batch.min(max_queue_depth)`, **with a doc comment saying `max_batch` alone OVERSTATES capacity whenever admission is tighter — the authors documented our bug before we arrived.** Any batch ratio uses `effective_batch_capacity()` **surfaced on the wire**, never a client-side `min()` reimplementation, per AC70: a duplicated invariant diverges silently.

| # | Decision | Rationale |
|---|---|---|
| D196 | Corrections are struck IN PLACE; a notice below the error corrects nobody | The reader meets the error first, and corrections are found only by those who already suspect one |
| D197 | A rule contradicting universal domain knowledge ships with its evidence, in the panel's `meta` | A bare prohibition invites repair by anyone competent; the file:line is what stops the helpful fix |
| D198 | All four KV page fields are `not-applicable` on the batching profile, including `kv_pages_total` | A real denominator is not a licence to draw — it is the half that makes the fabrication persuasive |
| D199 | Provenance is keyed by (field, capability profile), never by field name | The same field is a measurement on one server and structurally absent on the other |

---

## 63. ACCESSIBLE PARITY FOR THE HONESTY LAYER (D200–D203)

### 63.1 ✅ D200 — THE CONTRACT LINE @bb2ee824 CAUGHT IS WRONG, AND WRONG TWICE

They report that my *"value is null unless `state === 'ok'`"* is false, because **`stale` carries the last good value — that is the entire point of the state.** Correct, ratified: **`value` is non-null when `state` is `measured` OR `stale`.**

> **D200 — and the line was carrying TWO errors, which is the interesting part: the literal was stale (`'ok'` → `'measured'`, landed `24d831a2`) AND the state list was incomplete.** The correction offered fixes one while its premise would have restored the other. **A line with two independent defects is not twice as likely to be caught — it is LESS likely, because whoever finds one reports it and stops looking.** Fixing a line is the moment to re-derive it, not to patch it.

**On the `'ok'` premise:** verified with dates, because that is now the standing rule. `telemetry-field.js:129` reads `MEASURED: 'measured'` at HEAD. **The rename is `24d831a2` at 01:17 — authored by @bb2ee824 themselves. Their message describes `ab0a08ee` at 00:12, sixty-five minutes earlier.** Not a disagreement: **an agent's own uncommitted-then-committed work outran their own message.** Third stale premise of the session, and the second where the sender's own commit is the counter-evidence.

### 63.2 🔴 D201 — A TEST CAN CERTIFY THE DATA EXISTS WHILE THE CONSUMER READS A DIFFERENT PATH

Their AC28 report is **real, with a stale line.** Not `dashboard/index.js:140` (that is `panelById`) — the live site is **`dashboard/index.js:179`**:
```js
const roving = createRovingGroup(root, { label: panel.title });   // undefined
```
`registry.test.js:22` asserts `panel.module.meta.title` — **so the correct path is `panel.module.meta.title`, and `panel.title` is `undefined`. Every roving-group gets an empty accessible label.**

> **D201 — AND THE TEST IS WHAT MAKES IT INVISIBLE. `registry.test.js:22` proves every panel HAS a title; `index.js:179` reads it from the wrong path. The suite is green, the data is present, and the label is empty.** A test that validates the SOURCE certifies nothing about the CONSUMER. **This is the exact shape of D191 one layer up — there I found CSS that was matched but never emitted; here is data that exists but is never reached.** Both are the gap between *a thing is correct* and *a thing is used*, and **no static check I own spans it.** Reported, not edited — @c8d9a40e's file.

**Their `store-adapter` report is already fixed:** `:211` handles `not-applicable` and `unavailable` together. Verified at HEAD.

### 63.3 🔴 D202 — EVERY HONESTY TREATMENT MUST HAVE AN ACCESSIBLE EQUIVALENT, AND THE GAUGE RULES APPLY IN AUDIO

Everything §21/§31/§48 specify — em-dash, dotted vs double underline, grayscale separation, the on-screen caption — is **VISUAL**. A screen-reader user receives **none of it.** `panel-kit.js` is already right (`:290/:330/:353` produce *"not applicable here"*, *"too old to show"*, *"not measurable yet"*), and `kv-memory.js:229` already announces **"X percent, N of M blocks"** — **D180's name-both-terms rule, satisfied in audio before I ruled it.** Credit where due.

> **D202 — THE ABSENCE STATES ARE THE PART THAT MUST NOT DEGRADE, BECAUSE A SIGHTED USER SEES A GLYPH THAT LOOKS ODD AND INVESTIGATES, WHILE A SCREEN-READER USER HEARS A NUMBER AND MOVES ON.** Absence is *conspicuous* visually and *silent* in audio: an unstyled `—` still reads as "dash", but a field that announces its value with no state word is **indistinguishable from a live measurement.** So the accessible name is not a translation of the visual treatment — **it is the only channel that carries the state at all, and it must lead with the state, never append it.**

**TWO SITES WHERE THE GAUGE RULES LEAK, both in `kv-memory.js`:**
1. **`:229` — `numericValueOf(x) ?? '?'` announces `"0.0 percent, ? of ? blocks"`.** Per D180 a percentage whose terms are unknown must not be announced **at all**; `'?'` is the audio equivalent of drawing a bar and inventing half of it. **If either term is `null`, the field is `unavailable` — not a percentage with a shrug.**
2. **`:227-228` — `aria-valuemin: 0`, `aria-valuemax: 100` on a `progressbar`.** That is **D182 in audio, and worse than on screen:** a clamped bar at least looks pinned, but a screen reader flatly announces **"100 percent"** for a 240% oversubscription. **`role="progressbar"` is a dial by definition, so a ratio that can exceed 1 must not use it.**

> **D203 — AN ARIA ATTRIBUTE IS A CLAIM WITH THE SAME PROVENANCE OBLIGATIONS AS A RENDERED NUMBER, AND IT IS AUDITED BY NOBODY.** We have spent the session on what panels *draw*. `aria-valuenow` is a machine-readable assertion of a measured quantity, **consumed by assistive tech that cannot see the dotted underline, the caption, or the badge that would have qualified it.** Every rule in §58 and §60 applies verbatim to the accessible name — **and this is the one surface where no reviewer, no screenshot and no test of mine has ever looked.**

| # | Decision | Rationale |
|---|---|---|
| D200 | `value` is non-null for `measured` AND `stale` | Stale carrying the last good value is the point of the state; the line held two defects and one report closed the search |
| D201 | A test validating the source certifies nothing about the consumer | `registry.test.js` proves the title exists while `index.js:179` reads the wrong path — green suite, empty label |
| D202 | Absence states must lead the accessible name, never append it | Absence is conspicuous visually and silent in audio; a value announced with no state word reads as measured |
| D203 | ARIA attributes carry the same provenance obligations as rendered numbers | `aria-valuenow` asserts a measurement to a consumer who cannot see any qualifier we drew |

---

## 64. THE UNDERCLAIM IS THE SAME BUG WITH THE SIGN FLIPPED (D204–D207)

### 64.1 🔴 D204 — MY OWN PANEL THESIS IS FALSE, AND IT IS BEING COPIED INTO THE README RIGHT NOW

@e00032a4 named the failure mode; **I then found an instance of it in my own document, in the sentence @376a0297 is quoting approvingly into the README.** §25 carried:

> *"the pool stops accepting, it does not reclaim"*

**Verified at source, and it is wrong.** `paged_decode.rs:44 evict_until_free()` calls `evict_lru()`, live-called at `flat_autoregressive.rs:307`, and its doc comment states the invariant: *"Only prefixes no live sequence is borrowing can go."* **Refcount-aware LRU eviction, running, on the dynamic path.** What does *not* reclaim is the **VRAM-ceiling knob** — `ByteBudget::reconfigure` moves `state.limit` and never `state.used`.

> **D204 — CORRECTING AN OVERCLAIM BY INSTALLING THE OPPOSITE UNDERCLAIM IS NOT A FIX. IT IS THE SAME ERROR WITH THE SIGN FLIPPED, AND IT IS STRICTLY HARDER TO CATCH, BECAUSE IT SOUNDS MODEST.** Every reflex this crew built tonight fires on *claiming too much*. **We have built nothing that fires on disclaiming something true** — no test, no badge, no envelope. The provenance system is **structurally blind to it**: an underclaim renders as `unavailable` or `not-applicable`, which are the two states we have taught everyone to read as *honest*. **A false negative wears the costume of integrity.**

**AND THE SPECIFIC DANGER IS THE AUDIENCE:** we invite the skeptical expert to check us. *"It does not reclaim"* is **disproved in thirty seconds by anyone who opens `paged_decode.rs`** — and it is sitting in the paragraph that underwrites every other claim on the page. **An underclaim in the honesty layer costs more credibility than an overclaim in a panel, because the honesty layer is the thing the reader is using to decide whether to trust the panels.**

### 64.2 D205 — NAME THE MECHANISM, DON'T PICK A SIDE

Both *"the allocator evicts"* and *"the allocator does not evict"* are wrong, because **"the allocator" is two subsystems on two axes.** The repair is not a better adjective:

> **D205 — WHEN A CLAIM IS FALSE IN BOTH DIRECTIONS, THE WORD BEING ARGUED OVER IS AMBIGUOUS AND THE FIX IS TO REPLACE IT WITH THE MECHANISM.** Ship two verifiable claims instead of one unverifiable adjective: *"pages are reclaimed by refcount-aware LRU when the pool runs dry; changing the VRAM ceiling reclaims nothing — it refuses the next allocation."* **This keeps the genuinely impressive part (LRU that will not evict a live sequence) while surrendering the part we do not have (ceiling-driven reclamation)** — and it is checkable line by line, which a single adjective never is. **A sentence a reader can verify in halves is worth more than one they must accept whole.**

### 64.3 🔴 D206 — THE CORRECTION WAS ALREADY IN MY DOCUMENT, 430 LINES AWAY

§55 records my self-correction — *"eviction is real; I generalised from one component to the whole allocator and carried it for hours"* — **written before @e00032a4's broadcast, from independent verification.** And the false thesis stayed live at line 2231 the whole time.

> **D206 — I FOUND THE TRUTH, WROTE IT DOWN, AND DID NOT GO BACK AND FIX WHAT IT FALSIFIED. This is D196 recurring inside a single author's own document within the hour** — my correction, my error, my file, and it still propagated to the PM and is now inbound to the README. **Learning a fact does not retract the things you said before you knew it; nothing does that automatically, and memory reliably reports the opposite.** **Standing rule for me: when I correct myself, `grep` my own document for what the old belief authorised — before writing the new section.**

### 64.4 D207 — RESOLUTION BY OMISSION IS A UI FAILURE, NOT ONLY A PROCESS ONE

@c7a654ed: *"the risk isn't someone deliberately deleting the panel; it's someone reading two correct messages, inferring a conflict, and resolving it by omission. That failure needs no bad intent and leaves no trace."*

> **D207 — THAT IS ALSO THE DEFAULT BEHAVIOUR OF EVERY UI WE ARE BUILDING, AND IT IS WHY §31 MATTERS MORE THAN IT LOOKED.** When a panel cannot resolve what to show, the cheap outcome is to render nothing — and **a missing panel is indistinguishable from a panel that was never specified.** Omission is the one outcome that leaves no artifact to review: **there is no empty slot, no badge, no reason string, nothing to grep.** So the rule that `not-applicable` keeps its **header and frame** and replaces only the **body** is not a stylistic preference — **it is the mechanism that converts an omission into a visible, reviewable statement.** A panel that says *"this cannot exist here, and here is why"* can be checked; a panel that isn't there cannot. **Every absence must leave a trace, or absence becomes the safest place to hide a decision nobody made.**

| # | Decision | Rationale |
|---|---|---|
| D204 | An underclaim is the same defect as an overclaim, and harder to catch | It renders as `unavailable`/`not-applicable` — the two states we taught everyone to read as honest |
| D205 | When a claim is false in both directions, replace the adjective with the mechanism | Two claims a reader can verify in halves beat one they must accept whole |
| D206 | On self-correction, grep your own document for what the old belief authorised | Learning a fact does not retract what you said before you knew it |
| D207 | Every absence keeps its frame and states its reason | Omission is the only outcome that leaves nothing to review |

---

## 65. D59 WITHDRAWN — AND THE FIRST RUN OF MY OWN RETRACTION RULE (D208–D211)

@12e42da8 withdrew **D59** (windowed prefix delta). Accepted without argument. **I wrote D206 twenty minutes ago — *"when I correct myself, grep my own document for what the old belief authorised"* — so this is its first live test, and it found something a strike alone would have missed.**

### 65.1 D208 — THE WITHDRAWAL TOOK TWO EDITS, NOT ONE

`grep D59` returns **one** line. But D59 had a **dependant** 130 lines away: §55's AC69 derivation read *"Combined with §19's windowed delta, `lookupsDelta === 0` is `pending` rather than `unavailable`, because within a dynamic scenario the number genuinely is coming."*

> **D208 — WITH D59 GONE, THAT SENTENCE PROMISES A NUMBER THAT IS NEVER COMING — WHICH IS EXACTLY WHAT `pending` EXISTS TO PREVENT.** Withdrawing the premise silently **inverted** the conclusion, from "honest, the value is en route" to "a spinner for a field that will never fill." **Striking only the line named in the ruling would have left the more harmful half live**, and it does not contain the string "D59." **A retraction is a graph traversal, not a text edit — and the dependants are exactly the places where the old belief has been converted into an instruction, which is where it does damage.** Both struck in place.

### 65.2 🔴 D209 — THE LEAD'S REASON IS BETTER THAN MY DESIGN, AND IT IS ABOUT THE SUBJECT, NOT THE SHAPE

D59's logic was correct: a process-global denominator is inflated by traffic that cannot contribute to the numerator, so window it. **The flaw is that a window repairs a DENOMINATOR and D59's numerator means nothing.** @376a0297 verified `runtime.rs:1017-1024`: branch A computes `cross_session_hit_len` and **never sets `loaded_prompt_prefix`** — the full prompt is queued and prefill recomputes every token. **The counter has no compute saving behind it.**

> **D209 — A WINDOW MAKES A CORRUPTED DENOMINATOR HONEST; IT CANNOT MAKE A NUMERATOR MEAN ANYTHING.** And the failure would have been **maximally convincing**: a windowed delta *moves*, it *resets per scenario*, it *responds to traffic* — **it would have animated beautifully while the feature did nothing.** @12e42da8's phrasing is the one to keep: **a hero metric that moves when nothing works is the worst artifact we could ship.** My repair was aimed at the half I could see, and **the half I could see was the half that was fine** — AC69's lesson (*a guard derived from an incident is shaped like the incident*) applied to my own fix.

### 65.3 D210 — WE HAD A DETECTOR FOR STILLNESS AND NONE FOR SPURIOUS MOTION

@376a0297's `created: now_unix()` finding closes the loop: a field that changes **every call** and is a **wall clock**.

> **D210 — EVERY REFLEX THIS CREW BUILT TREATS MOTION AS EVIDENCE OF LIFE.** `stale` exists because a frozen value is suspicious; AC20 forbids a metric that cannot move; §48's treatments all trigger on absence. **A reviewer hunting fabrications scans for a literal that never changes — and both of tonight's worst fields change constantly.** The mirror rule, adopted verbatim in substance: **ask what would have to happen IN THE ENGINE for this number to move. If the answer is "nothing," it is a clock, not a measurement.** **Design consequence: a sparkline is the strongest liveness claim on the page, so no sparkline may be drawn over a series whose motion we cannot attribute to the mechanism it is captioned with.** Motion is a claim, and it is the one claim we never made anyone justify.

### 65.4 D211 — "NOT BINDABLE" IS A DESIGN STATE, NOT AN ABSENCE OF ONE

Adopting AC81: `prefix_cache_hits` is **not a zero to be fixed — it is a nonzero to be distrusted**, and it must NOT become a sixth enum value.

> **D211 — @376a0297's reason is the design argument and I want it recorded in my own vocabulary: A `misleading` STATE WOULD STILL PUT THE NUMBER ON SCREEN WEARING A BADGE.** Every one of my five states resolves to **a cell that renders something**; the envelope's whole premise is that a field always has a presentation. **A field that must never be spoken has no state because it has no cell** — it is removed at the registry, not styled at the panel. **The five states describe fields we DISPLAY; "not bindable" is a decision made one layer above the enum, and pretending it is a state would smuggle the value back onto the page through the very system built to keep it honest.** Enforced by tripwire, per D183 — and this is why my `prefix-counters-forbidden.test.js` allowlist must reach zero rather than gain a `misleading` exemption.

**AND THE STRONGEST THING IN THAT BROADCAST IS AC82, WHICH IS ABOUT TESTS, NOT FIELDS:** `prefix_speedup.rs` asserts `hit_len > 0` and that greedy output matches. **Neither asserts prefill got shorter.** Under this defect **both pass while zero work is saved** — an always-nonzero counter trivially satisfies `> 0`, and recomputing a prefix trivially yields identical output. **The shipped test for prefix speedup cannot detect the absence of prefix speedup: it tests the REPORTING of the optimisation instead of its EFFECT, and being green it was stronger evidence than no test at all.** This is my own standard — *a test that has never been shown to fail has never been shown to work* — arriving from the repo rather than from us, and it is why every mechanical check I ship gets its mutation proven red first.

| # | Decision | Rationale |
|---|---|---|
| D208 | A retraction is a graph traversal, not a text edit | The dependants are where the old belief became an instruction, and they never contain its name |
| D209 | A window repairs a denominator; it cannot make a numerator mean anything | A windowed delta of a dead counter animates convincingly while the feature does nothing |
| D210 | No sparkline over a series whose motion we cannot attribute to its captioned mechanism | Motion is our only liveness heuristic and the one claim we never made anyone justify |
| D211 | "Not bindable" is decided above the enum, never inside it | Every state renders a cell; a sixth state would smuggle the value back on screen wearing a badge |

---

## 66. AGREEMENT BETWEEN COPIES IS NOT CORROBORATION (D212–D214)

Third message this session proposing to revert the ratified `'measured'` wire value, each verified against a working tree that the sender's **own later commit** superseded. HEAD, checked just now: `telemetry-field.js:153` = `MEASURED: 'measured'`, `shell.css:163` = `[data-state='measured']`, my suite 7/7. The cited fix `39f88fd2` is **00:19**; the rename `24d831a2` is **01:17** — **58 minutes later, same author.**

### 66.1 🔴 D212 — "TWO LAYERS AGREE, SO IT'S 2-TO-1" IS AN EPISTEMIC ERROR, NOT A COUNTING ONE

The argument offered was: `telemetry-field.js` says `'ok'`, `field-state.js` says `'ok'`, **CSS is the lone outlier — therefore the CSS is a typo.**

> **D212 — TWO FILES IMPLEMENTING ONE SUPERSEDED DECISION ARE NOT TWO WITNESSES. THEY ARE ONE DECISION, COPIED.** Agreement between derivatives measures **propagation**, not truth. **And the arithmetic is backwards in the way that matters: the more thoroughly a wrong value has spread, the more corroborated it looks** — so the outlier is systematically read as the error, and **the correct file is the one that gets "fixed."** That is precisely what was proposed here: the CSS had been updated toward the ruling, and the tally nominated it for reversion. **Independence is a property of ORIGIN, never of location — and in a codebase with no build step, copies are the normal way a value travels.**

### 66.2 ✅ D213 — THEIR TEST IS THE ANSWER TO THEIR OWN QUESTION

`state-treatments.test.js` parses `shell.css` and asserts **both directions**: every `FIELD_STATES` value has a rule, **and no rule names a state nothing emits.** The second direction is the one that bit them.

> **D213 — THAT TEST MAKES THE TWO-FILE PAIR ATOMIC BY CONSTRUCTION, WHICH IS WHY THE VALUE NEVER NEEDS RELITIGATING AGAIN.** Their diagnosis is exactly right and is the best structural observation in the message: **the bug lived only in the GAP between JS and CSS — a gap with zero coverage in a project that deliberately has no build step and no type checker.** JS was internally consistent; CSS was internally consistent; **nothing owned the seam.** **And "a selector that never fires reads as coverage" is D191 arriving independently from the implementation side** — a rule can be present, valid and matched-by-nobody, and every static check reports it healthy. The test now pins whichever value is ratified, so **the migration they offered is unnecessary rather than merely unwanted.**

**It also mechanises two accessibility requirements that were previously checkable only by eye:** that the three absence states differ by **border pattern** rather than colour, and that every `SOURCE_CLASS` has a `[data-source]` hook. **Per §60 the colour channel between absence states measures 1.00–1.05:1 — so the border pattern is not reinforcement, it is the entire signal, and it was until now defended by nothing but my memory.** This is the outcome I want: **the treatment set enforced rather than remembered.**

### 66.3 D214 — THE STANDING FIX, STATED AS A COMMAND

Three stale reports from one author on one subject inside ninety minutes is not carelessness — it is a **workflow** producing them.

> **D214 — A MESSAGE COMPOSED WHILE WORK IS IN FLIGHT DESCRIBES A TREE THAT NO LONGER EXISTS BY THE TIME IT IS READ, AND NOTHING IN THE MESSAGE CARRIES ITS OWN TIMESTAMP.** The fix is one line before sending any claim about a file: `git log --oneline -1 -- <path>`. **Quote the SHA with the claim.** A fact with a commit beside it can be aged by its reader; **a bare fact cannot, so it is trusted indefinitely — including by its own author, who is the one person certain to have moved on from it.**

| # | Decision | Rationale |
|---|---|---|
| D212 | Agreement between copies is propagation, not corroboration | The more widely a wrong value spread, the more corroborated it looks — so the corrected file is nominated as the outlier |
| D213 | The CSS↔JS seam test makes the atomic pair atomic by construction | The bug lived only in the gap; both sides were internally consistent and nothing owned the seam |
| D214 | Quote the SHA with any claim about a file | Nothing in a message carries its own age, so a bare fact is trusted indefinitely — first of all by its author |

---

## 67. THE 250 ms CADENCE IS AN ASPIRATION — GLOBAL STALENESS, AND THE FINDING CARD (D215–D218)

Two assignments: @376a0297's **AC84** (global staleness detection, ruled for option (a)) and @12e42da8's *"design it as a finding card, not an empty panel."*

### 67.1 D215 — ONE POLL LOOP, SO ONE PAGE-LEVEL STATE

Verified: `app.js:32` and `telemetry-store.js:84` both hold `250`, **one loop, six-to-seven endpoints, every panel.** @fc8b5d97 measured `/metrics` and `/v1/resources` blocking **14,784 ms** during generation. So on the dynamic server **~59 consecutive polls return the same snapshot and then everything jumps at once.**

> **🔴 D215's PREMISE IS STRUCK — SEE §69. THE ~59-FROZEN-POLLS PREMISE CAME FROM D78's THREADING CLAIM, WHICH IS FALSE, AND FROM A MEASUREMENT OF A BUG THAT HAS SINCE BEEN FIXED. NO `sampled between requests` CAPTION SHIPS.** The rule below stands **only for staleness that is actually OBSERVED**, never asserted.
>
> **D215 — STALENESS IS A PROPERTY OF THE POLL LOOP, NOT OF A PANEL, SO IT IS ANNOUNCED ONCE IN THE HEADER AND NEVER REPEATED PER PANEL.** Six identical captions is not six disclosures — **it is noise that trains the reader to skip the one place it will eventually matter.** The header carries `sampled between requests`, and every time-series switches from a connected line to **discrete points with visible gaps.** The panels do not each decide; **they read one page-level fact**, which is also what keeps it out of the mode→field table AC64 forbids.

### 67.2 🔴 D216 — THE DETECTOR HAS THE EXACT AMBIGUITY IT WAS BUILT TO RESOLVE, AND THE CLIENT ALREADY HOLDS THE DISAMBIGUATOR

AC84 detects *"identical payload for N consecutive polls."* **But a genuinely idle server also returns identical payloads, forever.** So the signal means **"unobserved OR nothing is happening"** — which is **precisely the pair D78 says are pixel-identical.** A detector whose output is ambiguous in the same way as its input has moved the problem, not solved it.

> **D216 — THE DISAMBIGUATOR IS THE ONE FACT THE CLIENT MEASURES RATHER THAN READS: DID *I* ISSUE A GENERATION THAT HAS NOT YET COMPLETED?** Frozen payloads **with** a request in flight ⇒ **unobserved** — the server is busy and cannot answer, so render gaps and say so. Frozen payloads **with no request in flight** ⇒ **genuinely idle** — a flat line is TRUE and must be drawn as a normal connected line, because **treating a real idle period as a measurement gap is the underclaim of D204, and would make our own server look broken while it is working correctly.** **The client is not guessing about the runtime's threading here; it is reporting its own outbound state, which is the single thing on this page it knows with certainty.** That keeps AC84 observation-driven and self-correcting: **if @d7cf9b84's atomics land and polls stop freezing, the condition simply stops firing — nothing to un-hardcode.**

### 67.3 🔴 D217 — A CONNECTING LINE IS AN ASSERTION ABOUT THE INTERVAL, NOT THE ENDPOINTS

> **D217 — NEVER INTERPOLATE ACROSS AN UNOBSERVED GAP. A line segment between two samples asserts that the quantity passed through every value in between, and across a 14.8-SECOND HOLE THAT IS A CLAIM ABOUT ~59 MEASUREMENTS WE DO NOT HAVE** — drawn in the smoothest, most confident form the chart language offers. Points we measured render as points; the interval renders as **nothing**, because nothing is what we know about it. **This is D210's twin: there I banned motion we cannot attribute, here I ban CONTINUITY we cannot attribute. A chart makes two claims — the values AND the shape between them — and we have only ever audited the first.**

### 67.4 D218 — THE FINDING CARD: A NULL RESULT IS A RESULT, SO IT GETS A RESULT'S LAYOUT

The prefix-cache non-result must **not** be an empty panel or a `not-applicable` body. Those say *"nothing here."* This is the opposite: **the most evidenced claim on the page.**

> **D218 — THE FINDING CARD USES THE SAME TYPE SCALE, FRAME AND BADGE WEIGHT AS A LIVE PANEL, BECAUSE DEMOTING ITS TYPOGRAPHY WOULD RE-INTRODUCE THE LIE ABOVE THE VALUE LAYER — AC88 EXACTLY: type size is a claim about which number matters.** A finding rendered in caption grey next to full-size panels reads as a footnote **whatever the words say.** Structure, in this order:
> 1. **The prediction**, stated as we held it: *"we expected shared prefixes to cut TTFT."*
> 2. ~~**The two arms at IDENTICAL size** — shared **1341 ms** · zero-sharing control **1254 ms**.~~ **🔴 STRUCK ~90 SECONDS AFTER I WROTE IT — SEE §68. @fc8b5d97 DOWNGRADED THIS PAIR TO INCONCLUSIVE: a warm interleaved re-run put the shared prefix 16.98% FASTER — OPPOSITE SIGN — at `load average 22.56`, where a byte-identical binary swung −9.8% from load alone.** The hero is now the **mechanism** (`engine/runtime.rs:1083`); the timing pair appears **below it, as corroboration, carrying its own inconclusive verdict.**
> 3. **The sensitivity check**, which is what converts this from *unobserved* to *absent*: prefill is 1241 ms of a 1380 ms TTFT, so a working cache would show **~140 ms** — **a ~90% drop we could not have missed.**
> 4. **The conclusion**, and it must be stated in the DIRECTION THE EVIDENCE SUPPORTS: *"no prefix reuse on either execution path."* **Never "prefix caching is broken"** — that is an overclaim about a subsystem we did not audit — and **never "we could not measure it,"** which is the D204 underclaim and throws away the entire result.

**AND THE CAPTION THAT MAKES IT PERSUASIVE RATHER THAN APOLOGETIC:** *"This is the only claim on this page backed by evidence that it could have detected the opposite."* **A negative result with a control arm outranks a positive result without one, and no competing dashboard shows either.** Per @376a0297's AC83 — **which is my own scenario-provenance finding, and it applies here as the card's own justification: the sensitivity check IS the driving-action provenance, proving the action could have moved the number.**

| # | Decision | Rationale |
|---|---|---|
| D215 | Staleness is announced once at page level, never per panel | Six identical captions train the reader to skip the one that matters |
| D216 | Disambiguate frozen payloads by the client's own in-flight request state | Idle and unobserved are pixel-identical to the detector too; only the client's outbound state separates them |
| D217 | Never interpolate across an unobserved gap | A line asserts the interval, not just the endpoints — ~59 measurements we do not have |
| D218 | The finding card gets a live panel's type scale, frame and weight | Demoting its typography reinstates the lie above the value layer, where no check reaches |

---

## 68. A GREEN TEST IS A CLAIM NOBODY READS (D219–D222)

### 68.1 🔴 D219 — I REPORTED `panels.css` UNLINKED FOUR TIMES WHILE OWNING A GREEN TEST THAT CERTIFIES IT LINKED

@c7a654ed is right and the correction is accepted without qualification: `index.html:29` links `styles/panels.css`, landed **~25 minutes ago in `3af5c8d7`**, and the `css/` → `styles/` migration is finished. **The part that indicts me is not that I was stale.** It is that `asset-graph.test.js` — **my file, my assertion, running green in every commit I have made since** — asserts *every stylesheet is linked and no `<link>` is dead. **It has been passing. Passing MEANT `panels.css` was linked. I had a mechanical, 200-millisecond answer inside my own suite and relayed a memory instead — four times, into a blocker list two other agents are carrying.**

> **D219 — A GREEN TEST IS A CLAIM NOBODY READS. We consult a suite only when it is RED; when green it is silent, so the fact it certifies never enters anyone's head — including the head of the person who wrote the assertion.** Every stale report tonight has been answered by *someone re-running a command*. **Mine was answered by a command that was ALREADY RUNNING, on every commit, and reporting success into a channel I had stopped listening to.** Verification is a timestamp (AC86) — **and a green suite is the one timestamp on this project that refreshes itself continuously and is read by no one.**
>
> **THE FIX IS THE CHEAPEST ON THIS PAGE: before reporting any blocker that a test covers, RUN THE TEST AND QUOTE ITS NAME.** Not "panels.css is still unlinked" but *"`asset-graph.test.js › every stylesheet is linked` — pass, so it is linked."* **That converts a green test from a silent daemon into a citable witness, which is the only form in which any of us actually consume evidence.**

**AND THE SAME GREEN TEST REFUTES THE TWO RESIDUALS IN THE MESSAGE THAT CORRECTED ME**, which is why this is a mechanism and not a mea culpa. Verified at HEAD just now:

| Claim | Status | Evidence |
|---|---|---|
| `--og-na-*` has zero consumers, reuses `--og-unavail-*` | ❌ **FALSE** | **10 consumers** — `panels.css:900,920,927,967,969,975,985` + `shell.css:594,596,597,616` |
| `--og-na-*` is not the brighter colour committed at `968cb93a` | ❌ **FALSE** | `tokens.css:87` `#7e8fa0` vs `unavail-fg` `#758493` — **distinct and brighter, as committed** |
| `telemetry-store.js` has ZERO occurrences of `not-applicable` — no path can set it | ❌ **FALSE** | `:1079` `notApplicableField(entry.reason, meta)` — **a live production call** |

**The reachability chain is COMPLETE end to end, and I traced it rather than trusting the call site's existence, per @e00032a4's harness lesson:** `telemetry-provenance.js` **5 real classification sites** (`:347`, `:377`, `:555`, `:590`, `:613`) → `neverMeasuredField` (`telemetry-store.js:1077`) → `notApplicableField` → `field-state.js:111` → `panel-kit.js:281/615` → `[data-state='not-applicable']` **3px double**. **§31 renders.**

### 68.2 🔴 D220 — WHEN A MEASUREMENT AND A MECHANISM AGREE, THE MECHANISM IS THE CLAIM

@fc8b5d97 withdrew the timing verdict: the interleaved warm re-run put the shared prefix **16.98% FASTER** — opposite sign to the +7.0% — on a box at **load average 22.56 across 10 cores**, where a **byte-identical binary swung −9.8% from load alone.** The effect and the noise floor are the same size. **"PROVEN ABSENT" has been ruled three times on evidence that no longer stands.**

**What DOES stand is the source, and @c7a654ed found the decisive line I had missed:**
```
engine/runtime.rs:1014  let mut loaded_prompt_prefix = 0;
engine/runtime.rs:1018  branch 1 sets ONLY cross_session_hit_len
engine/runtime.rs:1075  loaded_prompt_prefix = materialized_len;   ← branch 2 ONLY
engine/runtime.rs:1083  .extend_from_slice(&prompt_tokens[loaded_prompt_prefix..]);
```
> **`:1083` IS THE WHOLE PROOF, AND IT IS A SLICE INDEX RATHER THAN AN INFERENCE: the prompt is sliced by `loaded_prompt_prefix`, the reachable branch never assigns it, so it stays `0` and THE ENTIRE PROMPT IS RE-PREFILLED WHILE A HIT IS REPORTED.**
>
> **D220 — A STRUCTURAL FACT IS TRUE AT n=1 AND CANNOT BE OVERTURNED BY MACHINE LOAD. A timing result is a sample of a distribution we do not control; a slice index is a property of the program.** So the finding card leads with **`:1083`** and demotes the timings to corroboration **carrying their own inconclusive verdict.** We had these two facts in the right order for the wrong reason: **the benchmark felt stronger because it produced a NUMBER, and a number looks like a measurement even when it is a sample of the machine's mood.**

### 68.3 D221 — THIS IS D217 AT THE POINT LEVEL, WHICH MEANS I SHIPPED THE BUG I HAD JUST BANNED

I wrote D217 — *never draw a line across an interval you did not observe* — and **one paragraph later set two timing numbers side by side whose difference is inside the noise floor.** Same error, different axis.

> **D221 — AN ERROR BAR IS THE INTERPOLATION PROBLEM AT A SINGLE POINT. A bare `1341 ms` asserts a precision the sample does not have, exactly as a connecting line asserts a continuity the series does not have. BOTH DRAW CONFIDENCE THAT WAS NOT MEASURED, and our envelope guards neither: `state: 'measured'` is entirely true of a number that is 90% machine load.** Any timing on this page ships **with its spread and its load conditions**, or it does not ship as a number.

### 68.4 D222 — AND THE CAPTION I WAS PROUDEST OF WAS THE FALSEST THING IN §67

I captioned the card *"the only claim on this page backed by evidence that could have detected the opposite."* **That sentence was FALSE AT THE MOMENT I WROTE IT** — the sensitivity check establishes the instrument's resolution, but at load 22.56 the noise floor had already swallowed the effect size. **I claimed a detection capability while the detector was saturated.**

> **D222 — THE MOST RHETORICALLY SATISFYING SENTENCE IN ANY ARTIFACT IS THE ONE TO CHECK FIRST. It is the least likely to be questioned, because it is the one that makes the reader — and the author — feel most rigorous.** This is @c7a654ed's D111 (*suspicion tracks implausibility, not falsehood*) with its most uncomfortable corollary: **an epistemically humble claim is CAMOUFLAGE, because humility reads as diligence already performed. And their harder corollary stands — it argues for re-checking the 2.46×, the number we feel best about, taken on the machine QA says cannot be trusted right now.**

| # | Decision | Rationale |
|---|---|---|
| D219 | Before reporting a blocker a test covers, run it and quote the test name | A green suite is a continuously-refreshed timestamp that nobody reads |
| D220 | Mechanism outranks measurement when both agree | A slice index is true at n=1; a timing is a sample of the machine's mood |
| D221 | No timing ships without its spread and load conditions | An error bar is D217's interpolation problem at a single point |
| D222 | Audit the most rhetorically satisfying sentence first | Humility reads as diligence already performed, so it is never questioned |

---

## 69. THREE MECHANISMS, ONE REMEDY — AND A TEST THAT PASSED WHILE THE BUG IT NAMES WAS LIVE (D223–D226)

### 69.1 ✅ @376a0297 IS RIGHT AND I VERIFIED THE BOUNDARY RATHER THAN THE LINE NUMBER

Their cited sites (`admin.rs:205`, `:448`) have drifted to **`:266`** and **`:509`**, and `:266` *looks* like it falls inside `debug_kv`. **It does not — I checked the handler boundaries instead of trusting proximity:**
```
admin.rs:232-254  debug_kv          ← NO resource_snapshot
admin.rs:256-270  resources         ← :266 lives HERE
```
**Their line numbers were stale; their conclusion is CORRECT. `/v1/debug/kv` never round-trips the driver.** Worth stating plainly because tonight's reflex is to assume a stale citation invalidates its argument — **it usually doesn't, and reporting only the drift would have destroyed a true finding.**

### 69.2 🔴 D223 — BUT NEITHER MECHANISM IS THE CURRENT TRUTH: THE STALL WAS A BUG, IT IS FIXED, AND THE TREE SAYS SO IN TWO TESTS

Nobody has cited **`crates/onnx-genai-server/src/tests.rs:3531`**:
```rust
async fn resource_snapshots_are_answered_during_a_batch_not_deferred() {
    assert!(deferred.is_none(),
      "a resource snapshot was pushed to the deferred queue; /v1/resources will \
       appear to hang until every in-flight generation completes");
```
and its sibling, whose docstring **names the exact symptom in the past tense**: *"the intake regression … that hung `/v1/resources` for **7.9s** under real sustained load"* — `run_static_engine_driver` drained every pending command into a `deferred` queue, then entered a batch loop reading only `rx`, so **a parked `ResourceSnapshot` waited for a batch that never goes idle under backfilled load.**

> **D223 — THE 14,784 ms WAS REAL, AND IT WAS A DEFECT, NOT AN ARCHITECTURE. Three explanations have now been advanced for one measurement — my threading claim (D78), the two-handlers claim (AC91), and the deferred-queue intake bug — AND ONLY THE THIRD IS WRITTEN DOWN IN THE TREE WITH A TEST HOLDING IT SHUT.** Both of the first two were **reverse-engineered from a stopwatch reading by people who did not open the driver.** The engine team had already found it, fixed it, and pinned it with an assertion **quoting the failure mode in the message.**
>
> **THE CONSEQUENCE IS THAT AC91's "whole fix" — TAKING `/metrics` AND `/v1/resources` OFF THE LOOP — MAY BE REMOVING PANELS TO ROUTE AROUND A BUG THAT IS NO LONGER THERE. Nobody should re-rule this from source, INCLUDING ME. It is a stopwatch question and it now takes @fc8b5d97 ninety seconds: re-run the 14.8 s measurement at HEAD.** Source told us three stories tonight; **the endpoint will tell us one.**

### 69.3 🔴 D224 — THE FIRST TEST PASSED WHILE THE 7.9-SECOND HANG WAS LIVE, AND ITS AUTHOR WROTE DOWN WHY

The sibling's docstring is the most valuable sentence in the repo tonight: the helper-level test ***"structurally cannot see"*** the regression, because *"the test fixtures generate in under a millisecond, so a racing integration test cannot land a request inside the batch window and **passes whether or not the bug is present.**"*

> **D224 — A TEST THAT PASSES WHETHER OR NOT THE BUG IS PRESENT IS NOT WEAK COVERAGE; IT IS A GREEN CLAIM ABOUT NOTHING, AND IT IS INDISTINGUISHABLE ON THE DASHBOARD FROM THE STRONGEST TEST IN THE SUITE.** This is D219 at its sharpest — **not a green test nobody read, but a green test that COULD NOT HAVE BEEN RED**, sitting in the same summary line as tests that could. **@fc8b5d97's standard — ask whether the instrument could detect the effect before reporting the result — is here proven necessary by a case where the instrument existed, was correct, was green, and was blind.** The fix was not a better assertion but a **different observation point**: assert after the batch RETURNS, where the parked command is still sitting unanswered. **The bug was never invisible; the test was standing in the wrong place.**

### 69.4 ✅ D225 — THREE WRONG MECHANISMS, ONE UNCHANGED REMEDY: THAT IS THE ARGUMENT, NOT A CONSOLATION

@376a0297 ruled **against** a hardcoded per-panel freeze list and **for** observation-driven detection. **Had we hardcoded on D78, we would have shipped permanent gaps in panels that update fine.** My D216 independently required the same: **detect, never assert**, and treat *frozen + nothing in flight* as genuinely idle rendering a normal line.

> **D225 — THE REMEDY SURVIVED THREE SUCCESSIVE MECHANISM ERRORS BY THREE DIFFERENT AGENTS BECAUSE IT NEVER DEPENDED ON THE MECHANISM. A rule that self-corrects under a wrong premise outranks a rule that is merely correct today** — and every premise offered tonight was wrong, including both of mine. **This is the strongest available argument for the whole provenance envelope: we have been wrong about the runtime repeatedly, and the design held anyway.**

**AC91 ACCEPTED on the caption, and the reason is stronger than the ruling states:** printing *"the engine services telemetry on the same thread that decodes"* would be **a false claim about our own architecture, rendered in the most technically confident sentence on the page**, drawing gaps where continuous data exists. **A FALSE ADMISSION OF IGNORANCE IS STILL A FALSE STATEMENT — and it is HARDER to catch than a false number, because self-deprecation reads as rigour.** That is D222 (audit the most satisfying sentence first) pointed at humility rather than confidence, **and it closes the mirror-image gap our ACs never covered: every rule we wrote guards against overstating what we know.**

### 69.5 ⚠️ D226 — AND THE JSDOC CLAIM IS FALSE; THIS IS THE FOURTH `'ok'` REPORT TONIGHT

@376a0297 reports `telemetry-field.js:19` reads `'ok' | 'pending' | …`. **Verified at HEAD, thirty seconds ago:**
```
telemetry-field.js:19  @typedef {'measured' | 'pending' | 'stale' | 'unavailable' | 'not-applicable'} FieldState
```
> **D226 — `'measured'` IS CORRECT AT HEAD AND NOBODY SHOULD "FIX" IT. Four agents have now independently reported `'ok'`, each from a tree superseded by `24d831a2` (01:17).** Per D212 this is **propagation, not corroboration** — and the count is now high enough that **the reports themselves have become the most persuasive evidence for the wrong value.** The mechanical guard already exists: `state-channel.test.js › gives the measured state a constant whose name equals its wire value`. **Per D219, quote that test name instead of re-reading the file — it is the only citation in this dispute that cannot go stale.**

| # | Decision | Rationale |
|---|---|---|
| D223 | The 14.8 s stall was a fixed intake bug, not an architecture; re-measure before dropping endpoints | Three mechanisms were reverse-engineered from a stopwatch; only the tree's version has a test |
| D224 | A test that passes with the bug present is a green claim about nothing | It shares a summary line with tests that could have failed |
| D225 | Prefer remedies that self-correct under a wrong premise | The remedy outlived three mechanism errors by three agents |
| D226 | `'measured'` stands; cite the test, not the file | Four independent stale reports look like corroboration and are one copy |

---

## 70. A MACHINE-GENERATED ARTIFACT HAS NO EXPIRY DATE EITHER (D227–D230)

### 70.1 🔴 D227 — THE FIFTH `'ok'` REPORT ARRIVED AS A JSON DUMP, WHICH IS THE MOST AUTHORITATIVE-LOOKING EVIDENCE FORMAT WE HAVE

@bb2ee824 wrote *"you are reading a stale copy — **the disk does not say what you think it says**"* and pasted:
```
{"MEASURED":"ok", …}
```
**That is not a remembered claim. It is real program output, machine-generated, in the format we all trust most.** Settled by **importing the module rather than grepping it**, because a grep can hit a comment and a paste can be old:
```
$ node -e "import('./telemetry-field.js').then(m=>console.log(JSON.stringify(m.FIELD_STATES)))"
{"MEASURED":"measured","PENDING":"pending","STALE":"stale","UNAVAILABLE":"unavailable","NOT_APPLICABLE":"not-applicable"}
```
Their cited commit `bac149a2` is **00:23**. The rename `24d831a2` is **01:17** — **fifty-four minutes later**, and `telemetry-field.js` has been touched **three more times since** (`356f8591`, `5ffa85b4`, `50d412be`).

> **D227 — MACHINE OUTPUT IS NOT FRESHER THAN MEMORY; IT IS ONLY MORE CONVINCING. A JSON dump is a photograph, and once pasted into a message it is exactly as frozen as a recollection — but it LOOKS unfalsifiable, so it ends the conversation instead of starting one.** @376a0297's AC86 said verification is a timestamp; **this is the corollary that hurts, because it revokes the one evidence format we had all silently exempted: the more mechanically-produced the artifact, the more freshness it CONNOTES and the less it CARRIES.** Every paste of command output on this project should carry the wall-clock time it was produced, for the same reason every citation now carries a SHA.

### 70.2 🔴 D228 — FIVE ROUNDS, AND THE COMMON FACTOR IS THAT EACH AGENT WAS CITING THEIR OWN WORK

Five reports that the wire value is `'ok'`. **In every case the reporter cited a commit they themselves authored** — and in this case the reporter authored the rename that superseded it too.

> **D228 — AN AGENT'S OWN COMMITS ARE THE SOURCE THEY ARE LEAST LIKELY TO RE-VERIFY, BECAUSE AUTHORSHIP FEELS LIKE KNOWLEDGE. Everyone here re-checks a claim from someone else and trusts a claim from themselves — so the staleness concentrates precisely where confidence is highest.** It is @376a0297's retraction in a different costume (*"I believed my recollection over two witnesses"*) and it is why this string has cost five rounds while genuinely contested questions cost one.

**🔒 THE STOP RULE, so this ends mechanically rather than by seniority:** `'measured'` is correct at HEAD. **No further re-litigation without BOTH** (a) the output of a **runtime import** — never a grep, never a recollection of a commit — **and** (b) `git log --format='%h %ad' -1 -- <path>` showing a commit **later than `24d831a2` (01:17)** that changed it. **Anything else is answered by one citation that cannot go stale: `state-channel.test.js › gives the measured state a constant whose name equals its wire value` — green at every commit since.**

### 70.3 🏅 D229 — `"32,768 tokens tokens"` SHIPPED THROUGH 109 GREEN TESTS, AND THAT IS THE THIRD BLIND-INSTRUMENT CASE TONIGHT

@bb2ee824 reports that migrating the model card produced **`"32,768 tokens tokens"` on the live page while all 109 tests passed** — a custom formatter appended the unit, then `formatField` appended it again. **Caught by looking at the browser.**

> **D229 — THE RENDER LAYER IS A SEPARATE INSTRUMENT CLASS, NOT A WEAKER FORM OF UNIT TESTING. A unit test asserts what a function RETURNS; the defect here lives in the COMPOSITION of two correct functions, and composition is only observable at the surface where the strings meet — the DOM.** With D219 (a green test nobody reads) and D224 (a green test that could not have been red), this is the third failure tonight of the same shape, **and it is the one squarely in my remit.** It is the standing argument for the design-review-by-screenshot pass: **not taste, but coverage of a layer no assertion in this repo reaches.** Their derived rule — **a custom formatter owns the whole string, units included** — is correct and should read as an API invariant rather than a bugfix: **the double-unit is not a formatting slip, it is TWO COMPONENTS BOTH BELIEVING THEY OWN THE SAME SUFFIX**, which is the panel-vs-field ownership question of §31 arriving one layer down.

### 70.4 ✅ D230 — DELETING THE UNSAFE PATH BEATS FIXING IT, AND THE TEST FEEDS IT THE STALE SPEC'S OWN OUTPUT

They deleted `formatFieldText` rather than patch it, moved its one caller, and gave `format.js` a terminal branch that logs and renders an em-dash for an unrecognised state — **refusing to put an unverified value on screen.** The test feeds it state `'measured'`… **which is precisely what a stale spec produces**, and asserts `999` never reaches the DOM.

> **D230 — A TEST WHOSE FIXTURE IS THE EXACT ARTIFACT A STALE READER WOULD GENERATE IS WORTH MORE THAN ONE USING A SYNTHETIC BAD VALUE, because it fails in the shape the failure will actually arrive in.** This is the correct resolution of the §21/§31 unknown-state question: **not a throw at render time, but a REFUSAL TO DISPLAY** — the value is withheld, the frame is kept, and the reason is stated. Per D211, *"not bindable"* is decided above the enum, and `format.js` is now where that decision is enforced rather than described.

| # | Decision | Rationale |
|---|---|---|
| D227 | Pasted command output carries a wall-clock time or it is a recollection | A dump is a photograph; mechanical production connotes freshness it does not carry |
| D228 | An agent's own commits are their least-verified source | Authorship feels like knowledge, so staleness concentrates where confidence is highest |
| D229 | Render-layer review is a distinct instrument, not weaker unit testing | Composition defects are only observable where the strings meet |
| D230 | Unknown state refuses to display rather than throwing | Withhold the value, keep the frame, state the reason |

---

## 71. THE SERVED-BYTES RENDER PASS — AND I OVER-REFUTED A CORRECT REPORT (D231–D234)

**BOTH ORIGINS ARE UP.** First render review of this project against **bytes fetched from the server**, not files read from disk:
```
:8123/health 200   :8123/demo/ 200      :8124/health 200   :8124/demo/ 200
served styles/tokens.css 200 11,273b · shell.css 200 18,037b · panels.css 200 30,510b
curl :8124/demo/  vs  disk index.html  →  IDENTICAL
```
**`panels.css` is not merely linked, it is SERVED, 30,510 bytes.** That blocker is closed in the strongest available form and **the demo-serving task is DONE** — `GET /demo/` returns the page on **both** origins, answering @376a0297's open question. **Served-vs-disk being byte-identical also validates `asset-graph.test.js`, which reasons about disk: that equality was an assumption until this fetch, and it is now a measurement.**

### 71.1 🔴 D231 — I DISPROVED THE LETTER OF @c7a654ed's REPORT AND DISMISSED ITS SUBSTANCE, ONE SECTION AFTER WARNING @376a0297 AGAINST EXACTLY THAT

In §68 I marked *"`--og-na-*` has zero consumers; `not-applicable` reuses `--og-unavail-*`"* as **❌ FALSE**, on 10 consumers. **The consumer count was right and my verdict was wrong**, because I never checked *which rule* consumes them:
```
shell.css:201  [data-state='not-applicable'] {
                 color:         var(--og-unavail-fg);      ← UNAVAILABLE
                 border-bottom: 3px double var(--og-unavail-rule);   ← UNAVAILABLE
panels.css:886 .value[data-state='not-applicable'] { color: var(--og-na-fg); }   ← NA
shell.css:594  .scenario-switcher__note { … var(--og-na-rule) … }   ← NOT A STATE RULE AT ALL
```
**The PRIMARY state treatment — the bare attribute selector that cascades to every `not-applicable` element on the page — reaches for the UNAVAILABLE tokens.** Their substantive claim was **TRUE**; only their "zero consumers" wording was false. **And one of the four shell consumers I counted in their defence is the unreachable-scenario note, not a field state at all — I counted a token's occurrences and reported it as a state's treatment.**

> **D231 — DISPROVING A REPORT'S EVIDENCE IS NOT DISPROVING ITS CLAIM, AND A CRISP `❌ FALSE` IN A TABLE FORECLOSES THE INVESTIGATION FAR HARDER THAN A PARAGRAPH WOULD.** In §69.1 I wrote that *"stale citation ⇒ argument dead"* is usually wrong and that reporting only the drift would have destroyed a true finding. **I had already committed that error in the previous section, in the opposite direction: they cited a wrong MECHANISM for a real DEFECT, and I refuted the mechanism and closed the case.** The grep that vindicated me is the grep that stopped me looking.

### 71.2 🔴 D232 — ONE STATE, TWO COLOURS, DECIDED BY SELECTOR SPECIFICITY

Because `.value[data-state='not-applicable']` (panels) is more specific than `[data-state='not-applicable']` (shell), **a `not-applicable` value inside `.value` renders `--og-na-fg` `#7e8fa0` and every other `not-applicable` element on the page renders `--og-unavail-fg` `#758493`.** `panels.css:896` even carries the comment *"`--og-na-*`, NOT `--og-unavail-*`. The na set is deliberately BRIGHTER"* — **a comment in one file asserting a rule the other file contradicts.**

> **D232 — A DESIGN STATE THAT RESOLVES TO DIFFERENT COLOURS DEPENDING ON WHICH ELEMENT IT LANDS ON IS NOT A STATE, IT IS A COINCIDENCE OF SPECIFICITY. The visitor sees `not-applicable` in two brightnesses and has no way to learn that both mean the same thing** — worse, the dimmer of the two is **pixel-identical to `unavailable`**, which is the ONE distinction §31 exists to make. **RULING: `[data-state='not-applicable']` in `shell.css` takes `--og-na-fg` / `--og-na-rule`.** One state, one treatment, and the brightness gap committed at `968cb93a` finally reaches the screen. **@c8d9a40e / @bb2ee824 — shell.css is not mine; this is a report, not an edit.**

### 71.3 🔴 D233 — `pending` IS COLOUR-ONLY IN THE SERVED ARTIFACT, AND ITS COLOUR IS 1.00:1 AGAINST THE OTHERS

Confirmed against served CSS and the shipped glyph:
```
format.js:61  export const PENDING_TEXT = '···';
served CSS    [data-state='pending'] → font-style: italic   ← ONLY non-colour channel
```
**Italic applied to `···` produces no perceptible change** — three middots have no ascenders, descenders or stroke asymmetry to slant. So `pending`'s second channel is **decorative, not informational**, and per §60 its grey is **1.00–1.05:1 against `stale`, `unavailable` and `not-applicable`.**

> **D233 — `pending` IS THE ONLY ABSENCE STATE WITH NO WORKING SECOND CHANNEL, AND IT IS THE ONE A VISITOR SEES FIRST, ON EVERY FIRST FRAME.** In greyscale or for a colourblind viewer it is **indistinguishable from a permanent architectural absence** — *"loading"* and *"this can never have a value"* rendered identically at the exact moment the page is making its first impression. **`border-bottom: 1px solid var(--og-pending-rule)` — solid is the one underline pattern no state has claimed**, it is unambiguous against dashed/dotted/double, and it costs one line. **My `state-channel.test.js › gives every non-default state a second, non-colour channel` PASSES on this**, because it checks that a declaration EXISTS, not that it renders — **my own instrument answering "is there a property?" when the question is "can anyone see it?", which is D224's blind test in my own suite.**

### 71.4 ✅ D234 — AC95 IS RIGHT AND I WAS ONE OF THE THREE WHO ENDORSED THE WIRING

@376a0297 notes the `/metrics` wiring was endorsed **by @12e42da8, by me, and by them** — on a derivation that the data was real and the endpoint sound. **It was, and it stalled 14,784 ms anyway.**

> **D234 — A FIX IS NOT A MEASUREMENT, AND THREE CORRECT REVIEWERS ARE NOT A STOPWATCH. Re-inclusion of `/metrics` on the poll loop is gated on a measured number under load, never on the diff looking right** — which is my own D223 reached from the product side, and the second time tonight two agents have converged on one rule from opposite directions. **@fc8b5d97's *"write-side push, not read-side request"* and @e00032a4's *"publish, don't request"* are the same sentence derived from a stopwatch and from source with no contact between them. That is the only form of agreement this session has earned the right to trust** — and per D212 it is corroboration precisely BECAUSE the two derivations are independent in origin, not merely in author.

| # | Decision | Rationale |
|---|---|---|
| D231 | Disproving a report's evidence is not disproving its claim | A crisp ❌ in a table forecloses the investigation harder than prose |
| D232 | `[data-state='not-applicable']` must use the na tokens in shell.css | One state resolving to two colours by specificity is not a state |
| D233 | `pending` needs `border-bottom: 1px solid`; italic on `···` is inert | The first-frame state is currently indistinguishable from permanent absence |
| D234 | `/metrics` re-inclusion gated on a measured number, not a reviewed diff | Three correct reviewers endorsed the wiring that stalled 14.8 s |

---

## 72. AGE OF THE SAMPLE vs AGE OF THE VALUE — AND A STAND-DOWN SENT TO THE WRONG AGENT (D235–D238)

### 72.1 🔴 D235 — MY OWN D216 DETECTOR FIRES FALSELY ON A PERFECTLY-POLLED PANEL, AND @376a0297's AC45 CORRECTION IS WHAT EXPOSED IT

Their distinction is the sharpest technical gift I've received tonight and it lands directly on §67:

> ***"Staleness must measure the age of the SAMPLE, not the age of the VALUE."***

**The block grid is polled continuously and successfully — @fc8b5d97 clocked `/v1/debug/kv` at 1.9 ms across 61 clean polls DURING a 384-token generation.** Its **content** legitimately does not change while a generation runs. **And D216 says: identical payload for N consecutive polls, with a request in flight ⇒ render `unobserved`, with gaps.**

> **D235 — THAT IS EXACTLY WRONG FOR THIS PANEL. THE GRID WOULD BE MARKED UNOBSERVED WHILE BEING SAMPLED SUCCESSFULLY EVERY 250 ms, BECAUSE MY DETECTOR INFERS THE HEALTH OF THE SAMPLING FROM THE MOTION OF THE DATA.** It cannot tell *"we failed to look"* from *"we looked, repeatedly, and it genuinely hadn't moved"* — **which is D78's flat-because-unobserved ambiguity reappearing INSIDE the instrument I built to resolve it, for the third time in one design.**
>
> **THE FIX IS TO STOP INFERRING AND START RECORDING: `observedAtMs` is already in the field shape, and it is set when a poll SUCCEEDS — regardless of whether the value changed.** Staleness is then `now − observedAtMs` against the poll interval: **a direct measurement of when we last looked, which is a fact the client owns outright, exactly as D216's in-flight state was.** Payload comparison is deleted from the design. **An unchanged value with a fresh `observedAtMs` is `measured` and draws a normal connected line — a flat line that is TRUE.** This also retires the last piece of AC84 that depended on anyone understanding the runtime: **three mechanisms were proposed for the stall and all three were wrong, but nobody has to be right about threading to record a timestamp when a fetch resolves.**

### 72.2 ✅ D236 — AC97: PER-FIELD HONESTY CANNOT SEE A PER-CONNECTION FAILURE

Ratified as they state it, and it is the most dangerous residual defect in the design:

> **A whole-origin stall is AC6 arriving through the TRANSPORT layer. Every panel is individually honest and the page still lies, because honesty was enforced PER-FIELD while the failure is PER-CONNECTION.** Under two origins the dead half **keeps showing its last good frame forever, beside a live half — which makes it MORE convincing, not less**, because the working half supplies the credibility the dead half is spending.

**D235's fix is also the fix here, which is why I'd rather have one mechanism than two:** with `observedAtMs` stamped per successful poll, **a dead origin's fields age visibly and automatically** — no reachability probe, no heartbeat panel, no new endpoint. **The half that stopped answering stops claiming to be fresh, by construction.**

### 72.3 🔴 D237 — THE STAND-DOWN WAS SENT TO THE WRONG AGENT, WHICH MEANS THE RIGHT ONE DIDN'T GET IT

@376a0297 asked me to stand down from a `demo-spec.md` reconstruction. **I have never edited `demo-spec.md`.** Verified:
```
$ git log --oneline -- '*demo-spec.md'          → 1 commit, and it is not mine
$ git log --oneline -- '…/design/demo-ux.md'    → 42 commits, all mine
```
**My work this session is `demo-ux.md`, four test files, and `tokens.css`. Nothing else.**

> **D237 — A CORRECTION DELIVERED TO THE WRONG RECIPIENT IS WORSE THAN ONE NEVER SENT, BECAUSE IT IS MARKED DELIVERED. The sender now believes a hazard is contained; the agent actually rebuilding 29 ACs from memory has heard nothing and is still typing.** Every failure tonight has been about **whether a fact is still true**; this one is about **whether it reached the person it constrains** — a new axis, and the more agents there are, the cheaper the misroute and the more expensive the silence. **When an instruction says STOP, name the artifact and the commit, not just the agent** — *"stand down on the `demo-spec.md` reconstruction"* is checkable by any recipient in one `git log` and self-cancels when it lands on the wrong desk. **@12e42da8 @376a0297 — please re-issue it to the crew, not to me.**

### 72.4 ✅ D238 — AND THE BACKUP WARNING IS KIND, SINCERE, AND BUILT ON AN ASSUMPTION ABOUT MY SETUP

They warned that `demo-ux.md` is *"2,500 lines, scratch dir, no git, appended all session with `cat >>` — one truncated write from oblivion."* **Verified just now:**
```
git ls-files → TRACKED · 42 commits · 452,318 bytes · 4,443 lines · working tree CLEAN
+ a byte-identical twin in the artifacts dir, diff -q verified after every commit
```
> **D238 — IT IS COMMITTED AFTER EVERY SECTION AND MIRRORED; THE FAILURE THEY FEARED CANNOT HAPPEN. But their instinct deserves the credit, not the correction: they extrapolated from a real catastrophe they personally survived tonight, and warning someone about your own worst hour is the right reflex even when it misses.** The general form is worth keeping: **the warning was about MY setup, derived from THEIR setup, and unlike every other stale claim tonight it was never checkable by its author** — they had no way to run `git ls-files` on my behalf. **A claim about another agent's environment is the one class of fact our verify-before-you-cite rule cannot reach, because the evidence lives where the speaker isn't.**

| # | Decision | Rationale |
|---|---|---|
| D235 | Staleness reads `observedAtMs` per successful poll; payload comparison is deleted | Inferring sampling health from data motion cannot tell "didn't look" from "looked, unchanged" |
| D236 | The same timestamp covers a dead origin | A per-connection failure is invisible to per-field honesty; the live half lends credibility to the dead half |
| D237 | A STOP instruction names the artifact and commit, not just the agent | A misrouted correction is marked delivered while the real recipient keeps typing |
| D238 | A claim about another agent's environment is unverifiable by its author | The evidence lives where the speaker isn't |

---

## 73. THE HARDCODE IS NOW CORRECT, WHICH IS WHY IT MUST STILL BE BANNED (D239–D242)

`--max-batch` exists (@e00032a4, verified in HEAD). I measured the two defaults that decide the design:
```
cli.rs:71  --max-queue-depth  default_value_t = 256
cli.rs:76  --max-batch        default_value_t = 4
state.rs:205  effective_batch_capacity() = max_batch.min(max_queue_depth)  →  min(4, 256) = 4
```

### 73.1 🔴 D239 — THE FORBIDDEN CONSTANT AND THE CORRECT VALUE ARE NOW THE SAME NUMBER ON EVERY MACHINE WE WILL EVER DEMO ON

@12e42da8 ruled a client-side hardcoded `4` **forbidden** — *"it asserts a capacity no endpoint confirms."* **That ruling was enforceable when `--max-batch` didn't exist. It is now unenforceable by observation:** with stock defaults `effective_batch_capacity()` **is 4**, so a panel that hardcodes `4` and a panel that reads the served denominator **render identical pixels, in every screenshot, on every machine in this demo.**

> **D239 — A BAN WHOSE VIOLATION IS PIXEL-IDENTICAL TO COMPLIANCE CANNOT BE ENFORCED BY REVIEW, BY SCREENSHOT, OR BY ANY RENDER-LAYER CHECK — INCLUDING THE ONE I ARGUED FOR IN D229 THIRTY MINUTES AGO.** My own instrument goes blind here, and I want that stated rather than discovered later: **the render layer catches composition defects (`"tokens tokens"`) precisely because they are VISIBLE, and this one is invisible by construction.** It diverges only under `--max-batch 8`, or under `--max-queue-depth 2`, **which is the source's OWN doc-comment example** — *"with `max_batch=4` and `max_queue_depth=1` … reporting `1/4 = 25%` would show a fully saturated server as three-quarters idle."*
>
> **THIS IS THE NIGHT'S PATTERN IN ITS FINAL FORM: RIGHT ANSWER, WRONG DERIVATION, AND NO OBSERVATION CAN SEPARATE THEM.** Every other case tonight was a wrong value we could eventually see. **Here the value is correct and the defect is entirely in the provenance — which is the exact thing this dashboard exists to make visible, failing inside the dashboard's own implementation.** A mechanical test is not belt-and-braces here; **it is the only detector that can exist.**

**🔒 THE TRIPWIRE, with its mutation stated per AC85:** the batch panel's module must **read `batch_capacity` from the payload** and must contain **no numeric literal in a batch-denominator position.** *Mutation:* replace the bound denominator with the literal `4` — **the panel still renders `3 of 4` correctly, every visual check passes, and the test must go RED.** That inversion — **a mutation that is invisible on screen and loud in the suite** — is the clearest statement of why this test has to exist.

### 73.2 🔴 D240 — THE HARDCODE HASN'T HAPPENED YET, BUT THE INSTRUCTION TO WRITE IT IS ALREADY IN THE COMMENTS

I checked for a live violation and **found none in executing code.** What I found instead:
```
telemetry-provenance.js:505  "With max_batch pinned at 4, firing 8 concurrent requests…"
field-state.js:338           "…the block grid is plainly continuous, `max_batch=4`…"
field-state.js:371           "`3 of 4` shows both terms…"
telemetry-store.test.js:109  "max_batch of 4, which is the case that exposes the naming trap"
```
**And `batch_capacity` appears in exactly ONE place client-side — `telemetry-provenance.js:250`, inside a PROSE STRING describing the field. No module reads it.**

> **D240 — THE CONSTANT IS ALREADY IN THE EXPLANATORY TEXT, AND EXPLANATORY TEXT IS WHAT THE NEXT IMPLEMENTER READS. The panel is unbuilt; whoever builds it will read `3 of 4` in a design comment and type `4`, and will be RIGHT, and will have introduced the defect while following the documentation correctly.** This is §58's finding in its most consequential instance: **our enforcement covers identifiers and never covers prose, so the prose is where the next bug is currently incubating** — **already written, already reviewed, and waiting for someone to obey it.** The comments must say **`3 of batch_capacity`**, never `3 of 4`.

### 73.3 ✅ D241 — @12e42da8's "ABSOLUTE COUNT, NEVER A PERCENTAGE" IS SUPERSEDED, AND THE SERVER WENT FURTHER THAN THE RULING ASKED

`admin.rs:169-178` emits **all three terms**, and the source comments state the reasoning the ruling was reaching for:
```
batch_utilization: batch_utilization(current_batch_size, effective_batch_capacity())
batch_in_flight:   the raw numerator, unclamped
batch_capacity:    "so the client never hardcodes a capacity no endpoint confirms"
```
> **D241 — THE PERCENTAGE IS NOW PERMITTED, BECAUSE THE DENOMINATOR IS SERVED AND IS THE RIGHT QUANTITY.** The ruling said *emit `max_batch` as the denominator, non-negotiable*; **the code declined and was correct to** — raw `max_batch` **overstates capacity**, so obeying that instruction would have shipped a plausible wrong quantity. **Per @12e42da8's own precedence rule, a ruling beats code on DECISIONS and code beats prose on FACTS — and "what the denominator physically is" is a fact.** The unclamped numerator also settles my D84/AC87: **`queued` derives from `batch_in_flight − batch_capacity`, both served, never from a literal and never from `max_batch`.**

### 73.4 ✅ D242 — ANTI-VACUITY, AUDITED ON MY OWN SUITES RATHER THAN RECOMMENDED TO OTHERS

@e00032a4's guards (*the parser must find ≥10 flags; ≥1 real invocation must be audited*) are the correct generalisation of tonight's `problems 0`. **I audited my four suites for the same property rather than praising the idea:** `page-claims` **6** non-emptiness assertions, `state-channel` **3**, `prefix-counters-forbidden` **2**, `asset-graph` **1**.

> **D242 — EVERY SUITE I OWN GUARDS ITS OWN INPUT, BUT `asset-graph`'s SINGLE GUARD IS THE THIN ONE AND IT IS THE SUITE THAT MATTERS MOST — it is the file that certified `panels.css` linked while I reported it unlinked four times (D219).** A vacuity guard answers *"did the parser find anything?"*; **D233 showed my other blind spot is subtler — `state-channel` asserts a DECLARATION EXISTS, not that it RENDERS, so it passes on `pending`'s inert italic.** Both are the same failure at different depths: **the test measured a proxy that is cheap to satisfy. Non-emptiness is the floor, not the standard.**

| # | Decision | Rationale |
|---|---|---|
| D239 | The batch denominator needs a mechanical tripwire; review cannot catch it | Hardcoded `4` and served `4` are pixel-identical on every demo machine |
| D240 | Design comments must say `3 of batch_capacity`, never `3 of 4` | The instruction to write the hardcode is already in the prose the next implementer reads |
| D241 | Percentage permitted; denominator is `batch_capacity`, never `max_batch` | The served denominator is the right quantity; raw `max_batch` overstates capacity |
| D242 | Non-emptiness is the floor, not the standard | A guard that asks "did we find anything" passes on a proxy that is cheap to satisfy |
