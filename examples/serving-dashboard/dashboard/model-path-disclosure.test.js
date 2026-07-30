// Copyright (c) Microsoft Corporation.
//
// The model directory must not reach a visitor's screen.
//
// The server answers /v1/models with an ABSOLUTE path. On the machine this demo
// is presented from, that path begins with the presenter's home directory, so
// rendering it puts a personal filesystem layout on a projector. It was found
// by opening the page in a browser, past ~30 review findings and a green suite,
// because it is invisible to every test we had: the shared fake-store fixture
// spells the path `models/qwen2.5-0.5b-scatter-v2` -- RELATIVE, with no machine
// in it. A fixture cannot contain the one thing this defect is made of.
//
// WHY THE FIX IS A DELETION AND NOT A BASENAME, settled by execution rather
// than by preference. The catalogue classifies `server.model_path` as
// NOT_PLUMBED, and it is tempting to think reclassifying it would suppress the
// render. IT WOULD NOT. Driving the committed store against a live origin, the
// field comes back state='measured' with the full path and an accompanying
// drift warning, because the store deliberately DISPLAYS a value the catalogue
// did not expect -- suppressing a real measurement is the exact failure that
// check exists to catch. So the classification does not gate the render at all:
// the only thing that stops the path reaching a screen is not asking for it.
//
// The store is right and stays as it is. `telemetry-store.test.js` asserts the
// projection picks the correct model entry and that the drift warning fires;
// those tests are about honesty on the wire, they do not pin this defect, and
// they are untouched. This file pins the other axis, the one the field schema
// has no word for: MAY THIS BE SHOWN.

import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readdirSync, readFileSync } from 'node:fs';
import os from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore } from './testing/fake-store.js';
import { FIELD_STATES } from '../telemetry-field.js';
import { allFieldKeys, PROVENANCE } from '../telemetry-provenance.js';
import { findAbsolutePaths } from '../absolute-path.mjs';
import { REPO_ROOT } from '../shipping-tree.mjs';

// Built here rather than with fake-store's `measured()` helper ON PURPOSE: that
// helper still emits the retired `state: 'ok'`, which format.js refuses to
// render. A fixture in a state the product rejects would make every negative
// assertion below pass for the wrong reason -- the path would be absent because
// NOTHING renders, not because we stopped asking for it. This file states the
// state it means. See the broadcast on fake-store.js:26.
function measuredField(value, options = {}) {
  return {
    value,
    state: FIELD_STATES.MEASURED,
    source: options.source ?? 'server',
    unit: options.unit ?? '',
    label: options.label ?? '',
    at: Date.now(),
  };
}

// A path shaped like the real one: absolute, with a home directory in it. The
// test is worthless if this looks like the sanitised fixture everywhere else.
const HOME_PATH = '/Users/presenter/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b';
const NAMESPACED_MODEL_ID = 'Qwen/Qwen2.5-0.5B-Instruct';

// The two surfaces the defect was actually found on. These stay named because
// they are the regression, and they get the STRICT predicate below: not one
// mention of the key survives in either, in any form.
const SOURCES = Object.freeze([
  ['dashboard/system.js', new URL('./system.js', import.meta.url)],
  ['ui/model-card.js', new URL('../ui/model-card.js', import.meta.url)],
]);

// WHY THIS FILE DISCOVERS ITS CORPUS INSTEAD OF LISTING IT.
//
// `SOURCES` above is the two files where the disclosure was last found, and for
// a while it was also the entire corpus of the sweep. That made the guard pin
// the INSTANCE and miss the CLASS: a binding appended to a third shipped module
// left the suite fully green at 646/646, proven by mutation, not argued. A
// hardcoded list stops covering whatever was added last, and the render surface
// that leaks tomorrow is by definition the one nobody has written down yet.
//
// The same defect was found the same night in `run-tests.sh`, where four
// independent reviewers converged on a two-glob test command and all four
// missed a third test directory. Agreement between readers of the same
// incomplete map is not corroboration.


/**
 * Every shipped dashboard module, from the index rather than from the disk.
 *
 * `git ls-files` and not `readdirSync`: the question this guard answers is
 * "what do we SHIP", and an untracked scratch file on one agent's desk is not
 * shipped while a tracked file someone deleted locally still is. Both pathspec
 * forms are required together -- `:(top)` because a bare path resolves relative
 * to CWD and silently matches nothing when the runner starts in a subdirectory,
 * and `--full-name` because otherwise the names print `../`-prefixed and every
 * downstream join is wrong.
 */
function shippedDashboardModules() {
  const listed = execFileSync(
    'git',
    ['ls-files', '--full-name', '--', ':(top)examples/serving-dashboard/**/*.js', ':(top)examples/serving-dashboard/*.js'],
    { cwd: REPO_ROOT, encoding: 'utf8' },
  );
  return listed
    .split('\n')
    .filter(Boolean)
    .filter((path) => !path.includes('/node_modules/'))
    // Tests and their fixtures MUST be excluded: this very file, and the fake
    // store it drives, both spell the key on purpose. A guard that reddens on
    // its own fixture teaches the next author to weaken the predicate.
    .filter((path) => !path.endsWith('.test.js') && !path.includes('/testing/'));
}

/**
 * Source with comments removed, so a tombstone cannot be mistaken for a defect.
 *
 * Two shipped modules carry a comment naming this key -- `telemetry-provenance`
 * records where the row used to live, `telemetry-store` records why the row is
 * now a ban. Both are correct and both should stay. A text scanner cannot tell
 * a warning about a defect from the defect, so the scanner has to stop reading
 * the prose rather than the prose being made to hide from the scanner.
 */
function stripComments(source) {
  return source.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/.*$/gm, '$1');
}

// A BINDING, not a mention. `{ key: 'server.model_path' }` and
// `field('server.model_path')` are the two shapes that put the value on a
// screen; `NEVER_BIND` listing the same string is the opposite of the defect.
// Matching the bare key would make the ban that prevents this bug indis-
// tinguishable from the bug.
const BINDS_MODEL_PATH = /(?:key\s*:\s*|field\s*\(\s*)['"`]server\.model_path['"`]/;

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

function storeReportingHomePath(modelId = 'qwen-scatter') {
  return createFakeStore({
    fields: {
      'server.model_path': measuredField(HOME_PATH, { source: 'server' }),
      'server.model_id': measuredField(modelId, { source: 'server' }),
      'server.context_length': measuredField(32768, { source: 'server' }),
      'server.execution_provider': measuredField('CPU', { source: 'server' }),
    },
  });
}

/**
 * Every string a visitor could reach: rendered text AND attribute values.
 * Attributes matter specifically -- the disclosure was written three times per
 * field (textContent, title, aria-label), and the aria-label copy is invisible
 * to a browser screenshot, so a text-only sweep would score this clean while a
 * screen-reader user still heard the path.
 */
function visibleStrings(node, collected = []) {
  if (node?.attributes) {
    collected.push(...Object.values(node.attributes));
  }
  if (typeof node?.ownText === 'string') {
    collected.push(node.ownText);
  }
  for (const child of node?.children ?? []) {
    visibleStrings(child, collected);
  }
  return collected;
}

async function mountAndCollect(mount, modelId) {
  const root = document.createElement('div');
  const store = storeReportingHomePath(modelId);
  const mounted = mount(root, store);
  // Surfaces paint placeholders on mount and fill in on the first poll tick.
  // Collecting before this would sweep a card of "···" and score it clean.
  store.tick();
  await flushAnimationFrames();
  const strings = visibleStrings(root);
  mounted?.unmount?.();
  return { strings, text: strings.join(' ') };
}

describe('the model directory never reaches a rendered surface', () => {
  it('the system panel does not render the absolute model path', async () => {
    const { mount } = await import('./system.js');
    const { strings, text } = await mountAndCollect(mount);

    // Anti-vacuity: a mount that rendered nothing would pass the assertion
    // below for the wrong reason, which is how a deleted panel scores green.
    assert.ok(strings.length > 5, `system panel rendered almost nothing (${strings.length} strings)`);
    assert.ok(text.includes('qwen-scatter'), 'control: the model id must still render');

    assert.ok(!text.includes(HOME_PATH), 'the system panel rendered the absolute model path');
    assert.deepEqual(
      findAbsolutePaths(text),
      [],
      'the system panel rendered an absolute filesystem path',
    );
  });

  it('the model card does not render the absolute model path', async () => {
    const { mountModelCard } = await import('../ui/model-card.js');
    const { strings, text } = await mountAndCollect(mountModelCard);

    assert.ok(strings.length > 3, `model card rendered almost nothing (${strings.length} strings)`);
    assert.ok(text.includes('qwen-scatter'), 'control: the model id must still render');

    assert.ok(!text.includes(HOME_PATH), 'the model card rendered the absolute model path');
    assert.deepEqual(
      findAbsolutePaths(text),
      [],
      'the model card rendered an absolute filesystem path',
    );
  });

  it('keeps a legitimate namespaced model id intact on both protected surfaces', async () => {
    const { mount: mountSystem } = await import('./system.js');
    const { mountModelCard } = await import('../ui/model-card.js');

    for (const [surface, mount] of [
      ['system panel', mountSystem],
      ['model card', mountModelCard],
    ]) {
      const { text } = await mountAndCollect(mount, NAMESPACED_MODEL_ID);
      assert.ok(text.includes(NAMESPACED_MODEL_ID), `${surface} rendered: ${text}`);
    }
  });

  it('the detector can actually fire, so a clean run means something', async () => {
    // The known-failing control. Both assertions above are negative, and a
    // negative assertion over a broken collector passes forever. Here the same
    // collector is aimed at markup that genuinely contains the path -- once in
    // text and once in an attribute, because those are two different bugs.
    const root = document.createElement('div');
    const textNode = document.createElement('dd');
    textNode.textContent = HOME_PATH;
    const attrNode = document.createElement('dd');
    attrNode.setAttribute('aria-label', `Directory ${HOME_PATH}`);
    root.append(textNode, attrNode);

    const strings = visibleStrings(root);
    const found = strings.filter((value) => value.includes(HOME_PATH));
    assert.equal(found.length, 2, 'the collector must see BOTH the text and the attribute copy');

    // And the detector must fire on paths from operating systems other than the
    // one this file was written on. The previous predicate was the literal
    // `/Users/`, and it was certified by a mutation injecting HOME_PATH -- a
    // constant defined in this file. The probe came from the detector's own
    // literal, so it could only ever pass. Measured at b63f0a82, a /home, a
    // C:\ and a /var disclosure all rendered with this suite at 5/5 green.
    for (const foreign of [
      '/home/presenter/models/qwen2.5-0.5b',
      'C:\\Users\\presenter\\models\\qwen',
      '/var/lib/onnx-genai/models/qwen',
    ]) {
      const alien = document.createElement('dd');
      alien.setAttribute('aria-label', `Directory ${foreign}`);
      const host = document.createElement('div');
      host.append(alien);
      assert.ok(
        findAbsolutePaths(visibleStrings(host).join(' ')).length > 0,
        `the detector cannot see ${foreign} -- it is shaped like this desk, not like a filesystem`,
      );
    }
  });

  it('no shipped render path asks the store for server.model_path', () => {
    for (const [label, url] of SOURCES) {
      const source = readFileSync(fileURLToPath(url), 'utf8');

      // Control first: if this file stopped mentioning model_id, the search is
      // reading something other than what we think and the zero below is noise.
      assert.ok(
        source.includes('server.model_id'),
        `${label}: control failed -- server.model_id is absent, so this file is not what this test thinks it is`,
      );

      assert.ok(
        !source.includes('server.model_path'),
        `${label} binds server.model_path again. The store will hand back the absolute path with state='measured' -- reclassifying the catalogue entry does NOT suppress it, so the binding is the only thing standing between a home directory and a projector.`,
      );
    }
  });

  it('NO shipped dashboard module binds server.model_path, discovered not enumerated', () => {
    const modules = shippedDashboardModules();

    // ANTI-VACUITY FLOOR. A universal claim over an empty corpus is true and
    // worthless, and an empty corpus is exactly what a mis-resolved pathspec
    // produces -- silently, with exit 0. This floor is not ceremony: the first
    // run of this sweep returned 0 files because the pathspec resolved against
    // the runner's CWD instead of the repo root.
    assert.ok(
      modules.length >= 8,
      `the shipped-module sweep found ${modules.length} files, which means the pathspec is wrong, not that the dashboard is small`,
    );

    // SUBJECT REQUIREMENT. The corpus must contain the files the defect was
    // actually found on. A sweep that reaches many files but not these two
    // would go green while pinning nothing that ever broke.
    for (const [label] of SOURCES) {
      assert.ok(
        modules.some((path) => path.endsWith(label)),
        `${label} is missing from the discovered corpus, so this sweep cannot see the surface the bug shipped on`,
      );
    }

    // POSITIVE CONTROL FOR THE COMMENT STRIPPER, and it is the reason this
    // guard can afford to ignore comments at all. `telemetry-provenance.js`
    // carries a tombstone naming the key. If the raw read stops finding it the
    // reader is broken; if the stripped read still finds it the stripper is.
    // Both halves are asserted, because either one alone passes on a corpse.
    const tombstone = readFileSync(join(REPO_ROOT, 'examples/serving-dashboard/telemetry-provenance.js'), 'utf8');
    assert.ok(
      tombstone.includes('server.model_path'),
      'control failed: the tombstone comment is gone, so a zero below proves nothing about the reader',
    );
    assert.ok(
      !stripComments(tombstone).includes('server.model_path'),
      'control failed: the comment stripper left the tombstone behind, so every result below is a false positive waiting to happen',
    );

    // POSITIVE CONTROL FOR THE PREDICATE. A regex that matches nothing passes
    // every negative assertion forever. This proves it still recognises the
    // shape of the bug that shipped.
    assert.ok(
      BINDS_MODEL_PATH.test("{ key: 'server.model_path', label: 'Directory' }"),
      'control failed: the binding predicate no longer matches the exact line that reached a projector',
    );

    const offenders = modules.filter((path) =>
      BINDS_MODEL_PATH.test(stripComments(readFileSync(join(REPO_ROOT, path), 'utf8'))),
    );
    assert.deepEqual(
      offenders,
      [],
      `these shipped modules bind server.model_path: ${offenders.join(', ')}. The store hands back the absolute path with state='measured', so the binding is the only thing between a presenter's home directory and a projector. Reclassifying the catalogue entry does NOT suppress it.`,
    );
  });
});

// ---------------------------------------------------------------------------
// THE CLASS, not the field. Everything above pins `server.model_path` by name,
// which closes the instance a browser found and leaves the mechanism that
// produced it fully armed. @73e77d95 traced that mechanism: telemetry-store's
// stale-provenance branch shows ANY value the catalogue did not expect, and its
// stated reason -- "hiding a real number is the exact failure this branch
// exists to catch" -- is a rule written for GAUGES being applied to STRINGS.
// It fires precisely when our documentation is wrong about a field, which is
// precisely when nobody has reviewed that field. So the one branch guaranteed
// to render an unreviewed value is the one that renders it unreviewed.
//
// A per-field tripwire cannot cover that: the next disclosure will arrive under
// a key nobody has written down yet. This sweeps EVERY key the register knows.

// WHICH FIELDS CAN REACH THIS, decided by the wire and not by my judgement.
// Injecting a path into `sessions.active` proves nothing: the server types that
// as an integer, so no response can carry a string there and a red from it is a
// fixture artefact. But the opposite exclusion is the trap that would gut this
// guard -- `server.model_path`, the field that actually reached a projector,
// was classified NOT_PLUMBED and rendered anyway. Filtering to "string on the
// wire today" would therefore be blind to precisely the defect that started
// this file, because arrival is exactly when the catalogue is stale.
//
// So the rule fails closed: a key is exempt ONLY if a capture has PROVEN it
// numeric or boolean. Never observed, absent, null, unplumbed -- all stay in.
const CAPTURES = ['scatter', 'dynamic'].map((name) =>
  JSON.parse(readFileSync(new URL(`../fixtures/captures/${name}.json`, import.meta.url), 'utf8')),
);

function digPath(body, path) {
  return String(path).split('.').reduce((node, part) => (node == null ? node : node[part]), body);
}

/** Keys a real response has been observed to type as a number or boolean. */
function provenNonString(key) {
  const entry = PROVENANCE[key];
  if (!entry || typeof entry.path !== 'string') return false;
  return CAPTURES.some((capture) => {
    const endpoint = capture.endpoints?.[entry.source];
    if (!endpoint) return false;
    const value = digPath(endpoint.body ?? endpoint.json ?? endpoint, entry.path);
    return typeof value === 'number' || typeof value === 'boolean';
  });
}

const undeclaredDetail = new Map();

describe('no field can put an absolute home path on screen, whatever its name', () => {
  it('renders no absolute path for any key the wire has not proven numeric', async () => {
    const { mount } = await import('./system.js');
    const { mountModelCard } = await import('../ui/model-card.js');
    const PANEL_MOUNTS = [
      ['dashboard/system.js', mount],
      ['ui/model-card.js', mountModelCard],
    ];

    const all = allFieldKeys();
    assert.ok(all.length > 20, `Only ${all.length} keys enumerated; the register did not load.`);
    const swept = all.filter((key) => !provenNonString(key));
    assert.ok(
      swept.length > 5,
      `Only ${swept.length} keys survived the numeric filter; it is over-excluding and this sweep is hollow.`,
    );

    const offenders = new Set();
    let mountsObserved = 0;

    for (const key of swept) {
      for (const [label, mountPanel] of PANEL_MOUNTS) {
        const root = document.createElement('div');
        const fields = {
          [key]: measuredField(HOME_PATH, { source: 'server' }),
        };
        if (key !== 'server.model_id') {
          // Keep a visible positive control without overwriting the value under
          // test. The old unconditional assignment made the server.model_id
          // iteration assert against qwen-scatter instead of attacker input.
          fields['server.model_id'] = measuredField('qwen-scatter', { source: 'server' });
        }
        const store = createFakeStore({
          fields,
        });
        assert.equal(
          store.field(key).value,
          HOME_PATH,
          `${key}: the fixture overwrote the attacker-controlled value before rendering`,
        );
        const mounted = mountPanel(root, store);
        store.tick();
        await flushAnimationFrames();
        const text = visibleStrings(root).join(' ');
        mounted?.unmount?.();

        if (text.includes('qwen-scatter')) mountsObserved += 1;
        if (text.includes(HOME_PATH) || findAbsolutePaths(text).length > 0) {
          offenders.add(key);
          undeclaredDetail.set(key, label);
        }
      }
    }

    assert.ok(
      mountsObserved > 10,
      `The control string rendered in only ${mountsObserved} mounts; this sweep would pass vacuously.`,
    );

    assert.deepEqual(
      [...offenders],
      [],
      `${[...offenders].map((k) => `${k} (${undeclaredDetail.get(k)})`).join(', ')}\n\n` +
        'A panel painted a filesystem path from a field that is not server.model_path. ' +
        'The per-field guard above cannot see this: the store promotes any unexpected ' +
        'value to state="measured", so the next disclosure arrives under whichever key ' +
        'the catalogue is stale about next. Apply the display-safety boundary to the field.',
    );
  });
});

// ---------------------------------------------------------------------------
// THE SERVED TREE, not just the two renderers above.
//
// The header of this file says "a fixture cannot contain the one thing this
// defect is made of". THAT WAS FALSE, and it was measured false at HEAD:
// `/demo/fixtures/captures/scatter.json` answered 200 carrying
// `.endpoints./v1/models.body.data[0].path` = the presenter's real home
// directory, and `dynamic.json` the same. A CAPTURED FIXTURE FREEZES A PRE-FIX
// RESPONSE, so the server-side basename redaction cannot reach it and the
// SOURCES sweep above never looks at it.
//
// The discriminator is os.homedir(), deliberately: this file, and the two
// sibling disclosure suites, contain INVENTED paths (/Users/presenter,
// /Users/someone) that must keep passing. Only the machine actually running
// the suite can be disclosed by it, and only that is a defect.
describe('no file the launcher serves contains THIS machine home directory', () => {
  const HOME = os.homedir();
  const PKG = fileURLToPath(new URL('../', import.meta.url));

  // Measured against a live origin rather than assumed: .md and *.test.js
  // answer 404, while .js/.css/.html/.json answer 200.
  const isServed = (rel) =>
    /\.(js|mjs|css|html|json)$/.test(rel) && !/\.test\.js$/.test(rel);

  const walk = (dir, base = '') => {
    const out = [];
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name.startsWith('.')) continue;
      const rel = base ? `${base}/${entry.name}` : entry.name;
      if (entry.isDirectory()) out.push(...walk(`${dir}/${entry.name}`, rel));
      else if (isServed(rel)) out.push(rel);
    }
    return out;
  };

  it('the sweep reaches a real, non-trivial set of served files', () => {
    const files = walk(PKG);
    assert.ok(
      files.length > 20,
      `only ${files.length} served files found -- the walk is broken, and a broken ` +
        'walk makes the next assertion vacuously green',
    );
    assert.ok(files.includes('app.js'), 'app.js must be in the swept set');
    assert.ok(
      files.some((f) => f.startsWith('fixtures/')),
      'fixtures/ must be swept -- that is where the defect this block exists for lived',
    );
  });

  it('no served file discloses the presenter home directory', () => {
    const offenders = walk(PKG)
      .map((rel) => ({ rel, text: readFileSync(`${PKG}${rel}`, 'utf8') }))
      .filter(({ text }) => text.includes(HOME))
      .map(({ rel }) => rel);

    assert.deepEqual(
      offenders,
      [],
      `${offenders.join(', ')} contain(s) this machine's home directory and is served ` +
        "at /demo/. Replace it with the basename the server actually emits -- do NOT " +
        'exempt the file, because the visitor fetches it either way.',
    );
  });

  it('the detector can fire, so the green above means something', () => {
    // Without this, a homedir() that returned '' would make the sweep pass forever.
    assert.ok(HOME.length > 1, 'os.homedir() must be a real path for the sweep to mean anything');
    const planted = `{"path": "${HOME}/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b"}`;
    assert.ok(planted.includes(HOME), 'the predicate must match a planted disclosure');
    assert.equal(
      findAbsolutePaths(planted).length > 0,
      true,
      'the shared absolute-path detector must also see it',
    );
    // ...and must NOT fire on the invented paths this repository deliberately keeps.
    assert.equal(
      '/Users/presenter/Documents/GitHub/onnx-genai/models/qwen2.5-0.5b'.includes(HOME),
      false,
      'an invented fixture path must not be reported as a disclosure of this machine',
    );
  });
});
