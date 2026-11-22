#!/usr/bin/env node
// Real-browser smoke test of the bundled upstream UI against a tiny ONNX model.
// Usage: node scripts/test_llama_chat.mjs [absolute path to playwright/index.mjs] [screenshot.png]
// Requires a built target/debug/onnx-genai-server and Chrome. Run under the
// host's shared workload lock when other sessions are measuring performance.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

const { chromium } = await import(process.argv[2] ? pathToFileURL(process.argv[2]).href : 'playwright');
const port = 18080;
const origin = `http://127.0.0.1:${port}`;
const server = spawn('target/debug/onnx-genai-server', [
  'tests/fixtures/tiny-llm', '--addr', `127.0.0.1:${port}`, '--max-output-tokens', '32'
], { stdio: ['ignore', 'ignore', 'pipe'] });
let diagnostics = '';
server.stderr.on('data', data => { diagnostics += data; });
let browser;
try {
  let ready = false;
  for (let attempt = 0; attempt < 100; attempt++) {
    if (server.exitCode !== null || server.signalCode !== null) throw new Error(`server exited ${server.exitCode}: ${diagnostics}`);
    try { ready = (await fetch(`${origin}/health`)).ok; } catch {}
    if (ready) break;
    await delay(100);
  }
  assert(ready, `server did not become ready: ${diagnostics}`);
  console.log('Tiny-fixture server ready');
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
  const errors = [];
  const completions = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (new URL(response.url()).pathname.endsWith('/v1/chat/completions')) completions.push(response);
  });
  await page.goto(origin, { waitUntil: 'networkidle' });
  console.log('Initial UI:', (await page.locator('body').innerText()).slice(0, 1800));
  const input = page.getByRole('textbox').last();
  await input.waitFor({ state: 'visible', timeout: 15000 });
  await input.fill('hello');
  const responsePromise = page.waitForResponse(response => new URL(response.url()).pathname.endsWith('/v1/chat/completions'), { timeout: 30000 });
  await input.press('Enter');
  const firstResponse = await responsePromise;
  assert(firstResponse.ok(), `chat request failed: ${firstResponse.status()} ${await firstResponse.text()}`);
  await firstResponse.finished();
  const payload = firstResponse.request().postDataJSON();
  assert(!payload.tools?.length, 'UI attached tools to a model that explicitly declares no tool protocol');
  const wireAnswer = (await firstResponse.text()).split('\n')
    .filter(line => line.startsWith('data: ') && line !== 'data: [DONE]')
    .flatMap(line => JSON.parse(line.slice(6)).choices ?? [])
    .map(choice => choice.delta?.content ?? '').join('');
  assert(wireAnswer.trim(), 'server streamed no answer content');
  await page.getByLabel('Assistant message with actions', { exact: true }).first().waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden', timeout: 30000 });
  assert(completions.some(response => response.ok()), 'UI did not submit a successful chat request');
  for (const response of completions) {
    assert(response.ok(), `chat request failed: ${response.status()} ${await response.text()}`);
  }
  const answer = await page.getByLabel('Assistant message with actions', { exact: true }).first().innerText();
  assert(answer.includes(wireAnswer), 'UI did not render the streamed answer');
  await page.reload({ waitUntil: 'networkidle' });
  await page.getByLabel('User message with actions', { exact: true }).first().waitFor({ timeout: 15000 });
  assert((await page.locator('body').innerText()).includes('hello'), 'conversation was not restored after reload');
  const conversationId = (await firstResponse.request().allHeaders())['x-conversation-id'];
  assert(conversationId, 'UI did not identify its conversation for resumable streaming');
  const originalWire = await firstResponse.text();
  const replay = await page.evaluate(async ({ origin, conversationId }) => {
    const response = await fetch(`${origin}/v1/stream?conv_id=${encodeURIComponent(conversationId)}&from=0`);
    return { status: response.status, body: await response.text() };
  }, { origin, conversationId });
  assert.equal(replay.status, 200, 'completed conversation was not available for replay');
  assert.equal(replay.body, originalWire, 'browser replay changed the original streamed bytes');
  assert.deepEqual(errors, [], 'browser runtime errors');
  if (process.argv[3]) await page.screenshot({ path: process.argv[3], fullPage: true });
  console.log(JSON.stringify({ result: 'passed', successfulChatRequests: completions.length, answer, pageErrors: errors }));
} finally {
  if (browser) await browser.close();
  server.kill('SIGTERM');
  await new Promise(resolve => { if (server.exitCode !== null || server.signalCode !== null) resolve(); else server.once('exit', resolve); });
}
