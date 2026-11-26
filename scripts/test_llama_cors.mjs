#!/usr/bin/env node
// Actual Chrome, two browser origins and a real Native CPU server. No inference proxy/stub.
// Usage: node scripts/test_llama_cors.mjs [playwright/index.mjs] [server-binary]
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { createServer } from 'node:http';
import { resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

const { chromium } = await import(process.argv[2] ? pathToFileURL(process.argv[2]).href : 'playwright');
const binary = resolve(process.argv[3] ?? 'target/debug/onnx-genai-server');
const key = 'cors-native-fixture-key';
async function pageServer() {
  const server = createServer((req, res) => {
    res.writeHead(200, { 'Content-Type': 'text/html' });
    res.end('<!doctype html><title>Native cross-origin fetch proof</title>');
  });
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  return { server, origin: `http://127.0.0.1:${server.address().port}` };
}
const pages = [];
let child;
let browser;
let diagnostics = '';
let interrupted;
function terminateServer(signal) {
  if (!child?.pid) return;
  try { if (process.platform !== 'win32') process.kill(-child.pid, signal); else child.kill(signal); }
  catch (error) { if (error.code !== 'ESRCH') throw error; }
}
function interrupt(signal) {
  interrupted = new Error(`CORS browser proof interrupted by ${signal}`);
  terminateServer('SIGTERM');
  if (browser) void browser.close();
}
const onSigint = () => interrupt('SIGINT');
const onSigterm = () => interrupt('SIGTERM');
process.on('SIGINT', onSigint);
process.on('SIGTERM', onSigterm);
try {
  pages.push(await pageServer(), await pageServer());
  if (interrupted) throw interrupted;
  const reservation = await pageServer();
  const backend = reservation.origin;
  const closed = once(reservation.server, 'close');
  reservation.server.close();
  await closed;
  const env = { ...process.env };
  for (const name of Object.keys(env)) {
    if (name.startsWith('LLAMA_ARG_CORS_') || name === 'LLAMA_ARG_NO_CORS_CREDENTIALS' || name === 'LLAMA_API_KEY' || name === 'LLAMA_ARG_API_KEY_FILE') delete env[name];
  }
  child = spawn(binary, ['tests/fixtures/tiny-llm', '--backend', 'native', '--addr', new URL(backend).host,
    '--api-key', key, '--cors-origins', pages[0].origin, '--max-output-tokens', '32'],
  { env, detached: process.platform !== 'win32', stdio: ['ignore', 'pipe', 'pipe'] });
  let spawnError;
  child.on('error', error => { spawnError = error; });
  for (const stream of [child.stdout, child.stderr]) stream.on('data', data => { diagnostics = (diagnostics + data).slice(-16000); });
  const deadline = Date.now() + 60000;
  while (true) {
    if (interrupted) throw interrupted;
    if (spawnError) throw spawnError;
    assert(child.exitCode === null && child.signalCode === null, `server exited: ${diagnostics}`);
    try { if ((await fetch(`${backend}/health`, { signal: AbortSignal.timeout(1000) })).ok) break; } catch {}
    assert(Date.now() < deadline, `Native server startup timed out: ${diagnostics}`);
    await delay(100);
  }
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage();
  await page.goto(pages[0].origin);
  const preflights = [];
  const preflightIds = new Set();
  const network = await page.context().newCDPSession(page);
  await network.send('Network.enable');
  network.on('Network.requestWillBeSent', event => {
    if (event.request.method === 'OPTIONS' && event.request.url.startsWith(backend)) preflightIds.add(event.requestId);
  });
  network.on('Network.responseReceived', event => {
    if (preflightIds.has(event.requestId)) preflights.push(event.response.status);
  });
  const result = await page.evaluate(async ({ backend, key }) => {
    const request = async (stream, authorized = true, malformed = false) => {
      const response = await fetch(`${backend}/v1/completions`, {
        method: 'POST', credentials: 'include', signal: AbortSignal.timeout(30000),
        headers: { 'Content-Type': 'application/json', ...(authorized ? { Authorization: `Bearer ${key}` } : {}) },
        body: malformed ? 'not JSON' : JSON.stringify({ prompt: 'hello', max_tokens: 2, temperature: 0, logit_bias: { '5': 1000 }, stream })
      });
      // Read the actual streamed body incrementally, as a frontend does.
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let wire = '';
      while (true) { const { done, value } = await reader.read(); if (done) break; wire += decoder.decode(value, { stream: true }); }
      wire += decoder.decode();
      return { status: response.status, wire };
    };
    return { unauthorized: await request(false, false), malformed: await request(false, true, true), json: await request(false), sse: await request(true) };
  }, { backend, key });
  assert(preflights.includes(200), `Chrome did not observe a successful unauthenticated preflight: ${JSON.stringify(preflights)}`);
  assert.equal(result.unauthorized.status, 401, JSON.stringify(result));
  assert.equal(result.malformed.status, 400, JSON.stringify(result));
  assert.equal(result.json.status, 200, result.json.wire);
  const json = JSON.parse(result.json.wire);
  assert.equal(json.usage.completion_tokens, 2);
  assert(json.choices[0].text.length > 0);
  assert.equal(result.sse.status, 200, result.sse.wire);
  assert(result.sse.wire.includes('data: [DONE]'));
  const events = result.sse.wire.split('\n').filter(line => line.startsWith('data: ') && line !== 'data: [DONE]').map(line => JSON.parse(line.slice(6)));
  assert.equal(events.map(event => event.choices?.[0]?.text ?? '').join(''), json.choices[0].text);
  assert(events.some(event => event.choices?.[0]?.finish_reason));
  // Browser CORS failures are expected here; the actual denied POST must also fail closed server-side.
  await page.goto(pages[1].origin);
  const rejected = await page.evaluate(async ({ backend, key }) => {
    try {
      await fetch(`${backend}/v1/completions`, { method: 'POST', signal: AbortSignal.timeout(30000), headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${key}` }, body: '{}' });
      return false;
    } catch (error) { return error instanceof TypeError; }
  }, { backend, key });
  assert.equal(rejected, true, 'disallowed browser origin could read the server');
  const denied = await fetch(`${backend}/v1/completions`, { method: 'POST', headers: { Origin: pages[1].origin, Authorization: `Bearer ${key}`, 'Content-Type': 'application/json' }, body: '{}' });
  assert.equal(denied.status, 403);
  assert.equal(denied.headers.get('access-control-allow-origin'), null);
  // Check the real embedded frontend still loads under the restricted policy.
  await page.goto(backend, { waitUntil: 'networkidle' });
  assert((await page.title()).length > 0);
  console.log(JSON.stringify({ nativeBackend: true, crossOriginJson: true, crossOriginSse: true,
    readableUnauthorized: true, readableMalformed: true, forbiddenOrigin: true, embeddedUiLoads: true, preflightStatuses: preflights }));
} catch (error) {
  console.error(diagnostics);
  throw error;
} finally {
  if (browser) await browser.close();
  if (child?.pid) {
    const exited = child.exitCode !== null || child.signalCode !== null ? Promise.resolve() : once(child, 'exit');
    terminateServer('SIGTERM');
    await Promise.race([exited, delay(5000, undefined, { ref: false })]);
    terminateServer('SIGKILL');
    await exited;
  }
  process.off('SIGINT', onSigint);
  process.off('SIGTERM', onSigterm);
  for (const { server } of pages) {
    server.closeAllConnections();
    await new Promise(resolve => server.close(resolve));
  }
}
