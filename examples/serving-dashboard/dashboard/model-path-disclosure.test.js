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
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';
import { createFakeStore } from './testing/fake-store.js';
import { FIELD_STATES } from '../telemetry-field.js';
import { allFieldKeys, PROVENANCE } from '../telemetry-provenance.js';
import { findAbsolutePaths } from '../absolute-path.mjs';

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

const SOURCES = Object.freeze([
  ['dashboard/system.js', new URL('./system.js', import.meta.url)],
  ['ui/model-card.js', new URL('../ui/model-card.js', import.meta.url)],
]);

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

function storeReportingHomePath() {
  return createFakeStore({
    fields: {
      'server.model_path': measuredField(HOME_PATH, { source: 'server' }),
      'server.model_id': measuredField('qwen-scatter', { source: 'server' }),
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

async function mountAndCollect(mount) {
  const root = document.createElement('div');
  const store = storeReportingHomePath();
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

// A string-typed field that is NOT_PLUMBED today and IS bound to a surface.
// It is the model_path profile exactly: the wire does not carry it, so nothing
// proves it numeric, and if the server starts echoing it the store's stale-
// provenance branch promotes it to state='measured' and paints it. Declared
// rather than fixed because removing a field a visitor can see is a product
// call under freeze, not a developer's. Listed alone, and the anti-rot check
// below deletes this entry the moment it stops offending.
const DECLARED_STRING_DISCLOSURE = Object.freeze(['server.execution_provider']);

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
        const store = createFakeStore({
          fields: {
            [key]: measuredField(HOME_PATH, { source: 'server' }),
            // The identifier stays real so a panel that rendered NOTHING cannot
            // be mistaken for a panel that refused the path. @73e77d95's
            // positive control, applied on every iteration rather than once.
            'server.model_id': measuredField('qwen-scatter', { source: 'server' }),
          },
        });
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

    const undeclared = [...offenders].filter((key) => !DECLARED_STRING_DISCLOSURE.includes(key));
    assert.deepEqual(
      undeclared,
      [],
      `${undeclared.map((k) => `${k} (${undeclaredDetail.get(k)})`).join(', ')}\n\n` +
        'A panel painted a filesystem path from a field that is not server.model_path. ' +
        'The per-field guard above cannot see this: the store promotes any unexpected ' +
        'value to state="measured", so the next disclosure arrives under whichever key ' +
        'the catalogue is stale about next. Either stop binding the field or declare it.',
    );

    // Anti-rot, and the half that is usually omitted: a declaration that has
    // stopped being true is a permanent hole with a comment on it.
    const stale = DECLARED_STRING_DISCLOSURE.filter((key) => !offenders.has(key));
    assert.deepEqual(
      stale,
      [],
      `${stale.join(', ')} no longer discloses. Delete the declaration -- an exemption ` +
        'outliving its defect silently exempts whatever inherits that key.',
    );
  });
});
