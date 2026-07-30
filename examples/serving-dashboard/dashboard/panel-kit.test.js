// Copyright (c) Microsoft Corporation.
//
// Tests for the shared panel rendering helpers.
//
// The centre of gravity is `renderField`. It is the function every displayed
// number on this page passes through, so it is the one place where the
// difference between "the server measured zero" and "the server cannot measure
// this" is either preserved or destroyed. These tests hold that line.

import assert from 'node:assert/strict';
import { after, before, describe, it } from 'node:test';

import { flushAnimationFrames, installFakeDom } from './testing/fake-dom.js';

let uninstallDom;
before(() => {
  uninstallDom = installFakeDom();
});
after(() => uninstallDom());

const {
  capabilityNotice,
  createRepaintScheduler,
  describeFieldText,
  formatDuration,
  formatNumber,
  metricRow,
  renderField,
  sourceBadge,
  withAcronyms,
} = await import('./panel-kit.js');

describe('renderField — a measured zero and an unmeasurable value must look different', () => {
  it('renders a real zero as a full-contrast number, with no apology', () => {
    // demo-ux.md §4.1: "Measured. The value really is zero." `rejections: 0` is
    // real and good news; hiding or dimming it would misrepresent the server.
    const node = renderField({ value: 0, state: 'ok', source: 'server', unit: 'count', label: 'Rejections' });

    assert.equal(node.getAttribute('data-state'), 'ok');
    assert.equal(node.findByClass('value__num').textContent, '0');
    assert.equal(node.findByClass('value__num--unavailable'), null);
  });

  it('renders an unavailable value as an em-dash and never as a number', () => {
    const node = renderField({
      value: null,
      state: 'unavailable',
      source: 'server',
      unit: 'count',
      label: 'Preemptions',
      reason: 'The scheduler performs preemption but keeps no counter for it.',
    });

    assert.equal(node.getAttribute('data-state'), 'unavailable');
    assert.equal(node.findByClass('value__num--unavailable').textContent, '—');
    assert.doesNotMatch(node.textContent, /\d/, 'an unavailable field must contain no digits at all');
  });

  it('keeps the unit when the value is unavailable, because WHICH thing is missing is information', () => {
    const node = renderField({
      value: null,
      state: 'unavailable',
      unit: 'ms',
      label: 'Queue wait',
      reason: 'The browser can only observe sent to first token.',
    });

    assert.equal(node.findByClass('value__unit').textContent, 'ms');
  });

  it('makes the reason reachable by keyboard and screen reader, not hover only', () => {
    const reason = 'KV page statistics are computed by the engine but not yet exposed over HTTP.';
    const node = renderField({ value: null, state: 'unavailable', label: 'KV blocks', reason });

    // Focusable, but as a stop on a roving cursor rather than its own tab stop
    // (AC29). createRovingGroup promotes one of these to tabindex 0.
    assert.equal(node.getAttribute('tabindex'), '-1', 'the explanation must be focusable');
    assert.equal(node.hasAttribute('data-roving-item'), true, 'must join the roving cursor');
    assert.match(node.getAttribute('aria-label'), /KV blocks: not measurable yet/);
    assert.match(node.getAttribute('aria-label'), /not yet exposed over HTTP/);
    assert.equal(node.getAttribute('title'), reason);
  });

  it('renders pending distinctly from unavailable', () => {
    const node = renderField({ value: null, state: 'pending', unit: 'tok/s', label: 'Throughput' });

    assert.equal(node.getAttribute('data-state'), 'pending');
    assert.equal(node.findByClass('value__num--pending').textContent, '···');
    assert.match(node.getAttribute('aria-label'), /no samples yet/i);
  });

  it('degrades an unknown state to an em-dash rather than printing the value', () => {
    const node = renderField({ value: 99, state: 'freshly-invented', label: 'Anything' });

    assert.equal(node.getAttribute('data-state'), 'unavailable');
    assert.doesNotMatch(node.textContent, /99/);
  });

  it('shows a stale value with its age instead of presenting it as current', () => {
    const node = renderField({
      value: 42,
      state: 'stale',
      source: 'server',
      unit: 'count',
      label: 'Queue depth',
      at: Date.now() - 8000,
    });

    assert.equal(node.getAttribute('data-stale'), 'true');
    assert.match(node.findByClass('value__stale').textContent, /^\d+s ago$/);
  });
});

describe('renderField — AC7 provenance badges', () => {
  it('badges a server counter, a client measurement and a derivation differently', () => {
    const server = renderField({ value: 1, state: 'ok', source: 'server', label: 'a' });
    const client = renderField({ value: 1, state: 'ok', source: 'client', label: 'b' });
    const derived = renderField({ value: 1, state: 'ok', source: 'derived', label: 'c' });

    assert.equal(server.findByClass('value__src').textContent, 'ˢ');
    assert.equal(client.findByClass('value__src').textContent, 'ᶜ');
    assert.equal(derived.findByClass('value__src').textContent, 'ᴰ');
  });

  it('treats an endpoint path as a server source, so either Field vocabulary badges correctly', () => {
    const node = renderField({ value: 7, state: 'measured', source: '/v1/status', label: 'Queue depth' });

    assert.equal(node.findByClass('value__src').textContent, 'ˢ');
    assert.match(node.findByClass('value__src').getAttribute('title'), /\/v1\/status/);
  });

  it('prefixes an estimated value with a tilde so it reads as a different kind of claim', () => {
    const node = renderField({ value: 340, state: 'ok', source: 'estimated', unit: 'ms', label: 'Time saved' });

    assert.match(node.findByClass('value__num').textContent, /^~/);
    assert.equal(node.findByClass('value__src').textContent, 'ᴱ');
  });

  it('spells SIMULATED out in full, because a superscript is easy to miss', () => {
    assert.equal(sourceBadge('simulated').textContent, 'SIM');
    assert.match(sourceBadge('simulated').getAttribute('title'), /not measured at all/i);
  });
});

describe('formatNumber / formatDuration', () => {
  it('picks precision by magnitude so the same quantity reads the same everywhere', () => {
    assert.equal(formatNumber(0), '0');
    assert.equal(formatNumber(7.25), '7.25');
    assert.equal(formatNumber(98.74), '98.7');
    assert.equal(formatNumber(318), '318');
    assert.equal(formatNumber(4602), '4,602');
  });

  it('appends a percent sign rather than a spaced unit', () => {
    assert.equal(formatNumber(62.1, '%'), '62.1%');
  });

  it('returns an em-dash for a non-finite number instead of NaN or Infinity', () => {
    assert.equal(formatNumber(NaN), '—');
    assert.equal(formatNumber(Infinity), '—');
  });

  it('scales duration units at human thresholds', () => {
    assert.equal(formatDuration(0.4), '0.40 ms');
    assert.equal(formatDuration(38), '38 ms');
    assert.equal(formatDuration(1840), '1.84 s');
    assert.equal(formatDuration(124_000), '2 m 04 s');
  });
});

describe('metricRow', () => {
  it('labels the value for a screen reader even when it is an em-dash', () => {
    const row = metricRow('Preemptions', {
      value: null,
      state: 'unavailable',
      reason: 'No counter exists.',
    });

    assert.equal(row.findByClass('metric-row__label').textContent, 'Preemptions');
    assert.match(row.findByClass('value').getAttribute('aria-label'), /^Preemptions: not measurable/);
  });
});

describe('capabilityNotice', () => {
  it('names the fix and reassures that the rest of the page still works', () => {
    const notice = capabilityNotice({
      title: 'KV block introspection is off',
      body: "The engine computes page statistics; this server build doesn't expose them.",
      command: '--enable-debug-endpoints',
    });

    assert.match(notice.textContent, /--enable-debug-endpoints/);
    assert.match(
      notice.textContent,
      /Everything else on this page still works\./,
      'AC20: a missing flag is not a broken dashboard, and the copy must say so',
    );
  });

  it('gives the copy button an accessible name that says what it copies', () => {
    const notice = capabilityNotice({ title: 't', body: 'b', command: '--enable-admin-endpoints' });

    assert.equal(
      notice.findByClass('capability-notice__copy').getAttribute('aria-label'),
      'Copy --enable-admin-endpoints to the clipboard',
    );
  });
});

describe('withAcronyms — AC30', () => {
  it('defines each acronym on its first appearance only', () => {
    const fragment = withAcronyms('TTFT is the time to first token; TTFT is measured per request.', {
      TTFT: 'Time to first token',
    });
    const host = document.createElement('div');
    host.append(fragment);

    const abbreviations = host.findAllByTag('abbr');
    assert.equal(abbreviations.length, 1, 'a definition on every occurrence is noise, not help');
    assert.equal(abbreviations[0].getAttribute('title'), 'Time to first token');
    assert.equal(host.textContent, 'TTFT is the time to first token; TTFT is measured per request.');
  });

  it('leaves text untouched when there are no acronyms to define', () => {
    const host = document.createElement('div');
    host.append(withAcronyms('Plain sentence.', {}));

    assert.equal(host.textContent, 'Plain sentence.');
    assert.equal(host.findAllByTag('abbr').length, 0);
  });
});

describe('createRepaintScheduler — AC23', () => {
  it('coalesces many requests in one frame into a single paint', () => {
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    });

    scheduler.request();
    scheduler.request();
    scheduler.request();
    flushAnimationFrames();

    assert.equal(paints, 1, 'three requests in one frame must produce one paint, not three');
  });

  it('does not paint while the panel is hidden, and catches up when it returns', () => {
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    });

    scheduler.setVisible(false);
    scheduler.request();
    flushAnimationFrames();
    assert.equal(paints, 0, 'an off-screen panel must not burn frames');

    scheduler.setVisible(true);
    flushAnimationFrames();
    assert.equal(paints, 1, 'and it must repaint once it is on screen again');
  });

  it('respects the shell setting `hidden` on a collapsed panel body', () => {
    const root = document.createElement('div');
    root.hidden = true;
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    });

    scheduler.request();
    flushAnimationFrames();

    assert.equal(paints, 0);
  });

  it('cancel() prevents a queued paint, so destroy() cannot leave a frame in flight', () => {
    const root = document.createElement('div');
    let paints = 0;
    const scheduler = createRepaintScheduler(root, () => {
      paints += 1;
    });

    scheduler.request();
    scheduler.cancel();
    flushAnimationFrames();

    assert.equal(paints, 0);
  });
});

describe('describeFieldText — the sentence panels build describe() from', () => {
  it('says a value is not measurable rather than reporting a number', () => {
    assert.equal(
      describeFieldText('Preemptions', { value: null, state: 'unavailable' }),
      'Preemptions is not measurable yet',
    );
  });

  it('distinguishes "no samples yet" from "not measurable"', () => {
    assert.equal(describeFieldText('Hit rate', { state: 'pending' }), 'Hit rate has no samples yet');
  });

  it('includes the unit for a measured value', () => {
    assert.equal(
      describeFieldText('Queue depth', { value: 3, state: 'ok', unit: 'requests' }),
      'Queue depth 3 requests',
    );
  });
});
