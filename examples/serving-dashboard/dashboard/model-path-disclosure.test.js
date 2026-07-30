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
    assert.ok(!text.includes('/Users/'), 'the system panel rendered a home directory');
  });

  it('the model card does not render the absolute model path', async () => {
    const { mountModelCard } = await import('../ui/model-card.js');
    const { strings, text } = await mountAndCollect(mountModelCard);

    assert.ok(strings.length > 3, `model card rendered almost nothing (${strings.length} strings)`);
    assert.ok(text.includes('qwen-scatter'), 'control: the model id must still render');

    assert.ok(!text.includes(HOME_PATH), 'the model card rendered the absolute model path');
    assert.ok(!text.includes('/Users/'), 'the model card rendered a home directory');
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
    assert.ok(strings.join(' ').includes('/Users/'), 'the /Users/ predicate must be able to match');
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
