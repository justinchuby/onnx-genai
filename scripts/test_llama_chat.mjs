#!/usr/bin/env node
// Real-browser chat, network recovery and active-reload tests against ONNX fixtures.
// Usage: node scripts/test_llama_chat.mjs [absolute path to playwright/index.mjs] [screenshot.png]
// Requires a built target/debug/onnx-genai-server and Chrome. Run under the
// host's shared workload lock when other sessions are measuring performance.
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';
import { mkdtemp, readFile, writeFile, copyFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { createServer, request as httpRequest } from 'node:http';

// Expand only the position table of this known fixture, deriving its name and
// dimensions from the graph. Runtime code and the maintained fixture stay intact.
async function longContextFixture() {
  const directory = await mkdtemp(join(tmpdir(), 'nxrt-chat-reconnect-'));
  try {
    let graph = await readFile('tests/fixtures/tiny-llm/model.onnx.textproto', 'utf8');
    const nodes = [...graph.matchAll(/^  node \{\n[\s\S]*?^  \}/gm)].map(match => match[0]);
    const gather = nodes.find(node => node.includes('input: "position_ids"') && node.includes('op_type: "Gather"'));
    assert(gather, 'fixture must declare its position lookup');
    const tableName = [...gather.matchAll(/input: "([^"]+)"/g)][0][1];
    let context;
    graph = graph.replace(/^  initializer \{\n[\s\S]*?^  \}/gm, block => {
      if (!block.includes(`name: "${tableName}"`)) return block;
      const rows = Number(block.match(/dims: (\d+)/)[1]);
      context = rows * 2048;
      assert(Number.isSafeInteger(context) && context <= 65536, 'fixture context expansion exceeds the bounded smoke workload; update the test fixture');
      const raw = block.match(/raw_data: "(.*)"\n/);
      assert(raw && raw[1].length * 2048 <= 16 * 1024 * 1024,
        'fixture needs a bounded inline position table; update the reconnect fixture builder');
      return block.replace(/dims: \d+/, `dims: ${context}`)
        .replace(/(raw_data: ")(.*)("\n)/, (_, start, bytes, end) => start + bytes.repeat(2048) + end);
    });
    assert(context, 'fixture position initializer was not found');
    graph = graph.replace(/^  value_info \{\n[\s\S]*?^  \}/gm, block =>
      block.includes(`name: "${tableName}"`) ? block.replace(/dim_value: \d+/, `dim_value: ${context}`) : block);
    const metadata = (await readFile('tests/fixtures/tiny-llm/inference_metadata.yaml', 'utf8'))
      .replace(/max_sequence_length: \d+/, `max_sequence_length: ${context}`);
    await writeFile(join(directory, 'model.onnx.textproto'), graph);
    await writeFile(join(directory, 'inference_metadata.yaml'), metadata);
    await copyFile('tests/fixtures/tiny-llm/tokenizer.json', join(directory, 'tokenizer.json'));
    return { directory, outputTokens: context / 2 };
  } catch (error) {
    await rm(directory, { recursive: true, force: true });
    throw error;
  }
}

async function waitForServer(origin, process, diagnostics) {
  const deadline = Date.now() + 60000;
  let lastReport = Date.now();
  while (Date.now() < deadline) {
    if (process.exitCode !== null || process.signalCode !== null) {
      throw new Error(`fixture server exited before ${origin}/health became ready: ${diagnostics()}`);
    }
    try {
      if ((await fetch(`${origin}/health`, { signal: AbortSignal.timeout(1000) })).ok) return;
    } catch {}
    if (Date.now() - lastReport >= 5000) {
      console.log(`Waiting for fixture startup at ${origin}: ${diagnostics().slice(-800)}`);
      lastReport = Date.now();
    }
    await delay(100);
  }
  throw new Error(`fixture server did not become ready at ${origin}/health within 60s: ${diagnostics()}`);
}

async function stopServer(process) {
  if (process.exitCode !== null || process.signalCode !== null) return;
  const exited = once(process, 'exit');
  process.kill('SIGTERM');
  await Promise.race([exited, delay(5000, undefined, { ref: false })]);
  if (process.exitCode === null && process.signalCode === null) process.kill('SIGKILL');
  await exited;
}

// A real network fault, not an inference stub: end only the first browser
// reader after its first SSE chunk. The backend producer continues independently.
async function disconnectingProxy() {
  let disconnected = 0;
  const proxy = createServer((incoming, outgoing) => {
    const upstream = httpRequest({ hostname: '127.0.0.1', port: 18081,
      path: incoming.url, method: incoming.method, headers: incoming.headers }, response => {
      outgoing.writeHead(response.statusCode, response.headers);
      outgoing.flushHeaders();
      const interrupt = disconnected === 0 && incoming.method === 'POST'
        && new URL(incoming.url, 'http://127.0.0.1').pathname === '/v1/chat/completions'
        && incoming.headers['x-conversation-id'] && response.statusCode === 200
        && String(response.headers['content-type']).startsWith('text/event-stream');
      if (interrupt) {
        response.once('data', bytes => {
          disconnected++;
          outgoing.end(bytes);
          response.destroy();
        });
      } else {
        response.pipe(outgoing);
      }
      outgoing.on('close', () => response.destroy());
    });
    upstream.on('error', error => {
      if (outgoing.destroyed || outgoing.writableEnded) return;
      if (!outgoing.headersSent) outgoing.writeHead(502, { 'content-type': 'text/plain' });
      outgoing.end(`browser fault-proxy upstream error: ${error.message}`);
    });
    incoming.pipe(upstream);
  });
  proxy.listen(18082, '127.0.0.1');
  await once(proxy, 'listening');
  return { server: proxy, disconnectCount: () => disconnected };
}

async function longGenerationRequests(page, outputTokens) {
  await page.route('**/v1/chat/completions', async route => {
    if (!route.request().headers()['x-conversation-id']) return await route.continue();
    const body = route.request().postDataJSON();
    delete body.max_completion_tokens;
    Object.assign(body, { max_tokens: outputTokens, temperature: 0, ignore_eos: true, logit_bias: { '5': 100 } });
    await route.continue({ postData: JSON.stringify(body) });
  });
}

async function stopActiveGeneration(browser, origin, outputTokens) {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  try {
    await longGenerationRequests(page, outputTokens);
    await page.goto(origin, { waitUntil: 'networkidle' });
    const started = page.waitForResponse(response => new URL(response.url()).pathname === '/v1/chat/completions'
      && response.request().headers()['x-conversation-id'], { timeout: 30000 });
    await page.getByRole('textbox').last().fill('Stop this generation while it is computing');
    await page.getByRole('textbox').last().press('Enter');
    const response = await started;
    assert(response.ok(), 'Stop test generation was not admitted');
    const conversationId = response.request().headers()['x-conversation-id'];
    const activeTurns = async () => {
      const response = await fetch(`${origin}/v1/status`, { signal: AbortSignal.timeout(3000) });
      assert(response.ok);
      const workers = (await response.json()).workers;
      assert(workers.length > 0, 'status must expose actual inference workers');
      return workers.reduce((sum, worker) => sum + worker.active_turns, 0);
    };
    const stop = page.getByRole('button', { name: 'Stop', exact: true });
    await stop.waitFor({ state: 'visible' });
    assert.equal(await activeTurns(), 1, 'generation must own a real worker before Stop');
    const cancelled = page.waitForResponse(response => new URL(response.url()).pathname === '/v1/stream'
      && response.request().method() === 'DELETE', { timeout: 10000 });
    await stop.click();
    assert.equal((await cancelled).status(), 204, 'frontend Stop must explicitly cancel its producer');
    await stop.waitFor({ state: 'hidden', timeout: 10000 });
    const deadline = Date.now() + 5000;
    while (await activeTurns()) {
      assert(Date.now() < deadline, 'cancelled inference did not release its worker within 5s');
      await delay(25);
    }
    const replay = await fetch(`${origin}/v1/stream?conv_id=${encodeURIComponent(conversationId)}&from=0`);
    assert.equal(replay.status, 404, 'Stop must discard the cancelled replay');
    assert.deepEqual(errors, [], 'browser runtime errors while stopping generation');
    console.log(JSON.stringify({ activeStop: 'passed', workerReleased: true, cancelledReplay: replay.status, pageErrors: errors }));
  } finally {
    await page.close();
  }
}

async function activeReload(browser) {
  const fixture = await longContextFixture();
  const backendOrigin = 'http://127.0.0.1:18081';
  const origin = 'http://127.0.0.1:18082';
  const server = spawn('target/debug/onnx-genai-server', [fixture.directory,
    '--model-id', 'browser-reconnect-fixture', '--addr', '127.0.0.1:18081',
    '--max-output-tokens', String(fixture.outputTokens), '--replay-buffer-bytes', String(32 * 1024 * 1024)
  ], { stdio: ['ignore', 'pipe', 'pipe'] });
  let diagnostics = '';
  for (const stream of [server.stdout, server.stderr]) stream.on('data', data => { diagnostics = (diagnostics + data).slice(-32000); });
  let page, proxy;
  try {
    await waitForServer(backendOrigin, server, () => diagnostics);
    proxy = await disconnectingProxy();
    page = await browser.newPage({ viewport: { width: 1360, height: 900 } });
    const errors = [], posts = [], replayRequests = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => {
      const path = new URL(request.url()).pathname;
      if (path === '/v1/chat/completions') posts.push(request);
      if (path === '/v1/stream' && request.method() === 'GET') replayRequests.push(request.url());
    });
    await longGenerationRequests(page, fixture.outputTokens);
    await page.goto(origin, { waitUntil: 'networkidle' });
    const networkRecovered = page.waitForResponse(response => {
      const url = new URL(response.url());
      return url.pathname === '/v1/stream' && Number(url.searchParams.get('from')) > 0 && response.ok();
    }, { timeout: 30000 });
    networkRecovered.catch(() => {}); // cleanup may close a page before this awaited phase
    const input = page.getByRole('textbox').last();
    const started = page.waitForResponse(response => new URL(response.url()).pathname === '/v1/chat/completions');
    await input.fill('hello');
    await input.press('Enter');
    const initial = await started;
    assert(initial.ok(), `live request rejected: ${initial.status()} ${initial.ok() ? '' : await initial.text()}`);
    const conversationId = (await initial.request().allHeaders())['x-conversation-id'];
    assert(conversationId);
    const bareId = conversationId.split('::')[0];
    console.log('Live browser stream started', { conversationId, url: page.url() });
    await page.waitForURL(url => url.href.includes(bareId), { waitUntil: 'domcontentloaded' });
    await page.waitForFunction(bareId => Object.keys(localStorage).some(key => {
      if (!key.includes(bareId)) return false;
      try { return typeof JSON.parse(localStorage.getItem(key)).bytesReceived === 'number'; } catch { return false; }
    }), bareId);
    const lookup = async () => {
      const response = await fetch(`${backendOrigin}/v1/streams/lookup`, { method: 'POST',
        headers: { 'content-type': 'application/json' }, body: JSON.stringify({ conversation_ids: [conversationId] }) });
      assert(response.ok);
      return (await response.json()).find(row => row.conversation_id === conversationId);
    };
    const recovery = await networkRecovered;
    assert.equal(proxy.disconnectCount(), 1, 'proxy must disconnect exactly the initial browser reader');
    assert(Number(new URL(recovery.url()).searchParams.get('from')) > 0, 'network recovery did not use its persisted byte offset');
    assert.equal((await lookup())?.is_done, false, 'inference must still be running before reload');
    console.log('Reloading a genuinely active generation', { conversationId, outputTokens: fixture.outputTokens });
    const resumed = page.waitForResponse(response => new URL(response.url()).pathname === '/v1/stream' && response.ok(), { timeout: 30000 });
    resumed.catch(() => {});
    await page.reload({ waitUntil: 'domcontentloaded' });
    const resumedResponse = await resumed;
    assert.equal((await lookup())?.is_done, false, 'generation finished before automatic browser reconnection could be tested');
    assert(replayRequests.length > 0, 'frontend did not automatically request replay');
    assert.equal(posts.filter(request => request.headers()['x-conversation-id']).length, 1, 'reload resubmitted inference instead of resuming the original generation');
    // Open an independent observer only AFTER the old browser reader has been
    // torn down and automatic reattachment has succeeded. It cannot keep the
    // original reader alive and mask a disconnect-cancels regression.
    const retained = await fetch(`${backendOrigin}/v1/stream?conv_id=${encodeURIComponent(conversationId)}&from=0`, { signal: AbortSignal.timeout(90000) });
    assert(retained.ok);
    const wire = await retained.text();
    assert(wire.includes('data: [DONE]'), `generation did not finish: ${wire.slice(-1500)} ${diagnostics}`);
    const values = wire.split('\n').filter(line => line.startsWith('data: ') && line !== 'data: [DONE]')
      .map(line => JSON.parse(line.slice(6)));
    const answer = values.flatMap(row => row.choices ?? []).map(choice => choice.delta?.content ?? '').join('');
    assert.equal(answer.split(' ').length, fixture.outputTokens, 'generation did not run its actual output budget');
    await page.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden', timeout: 90000 });
    const messages = page.getByLabel('Assistant message with actions', { exact: true });
    await messages.last().waitFor({ timeout: 30000 });
    assert.equal(await messages.count(), 1, 'reload created duplicate assistant messages');
    const rendered = await messages.last().innerText();
    assert(rendered.includes(answer), 'resumed UI lost generated text');
    assert.equal((rendered.match(/\bworld\b/g) ?? []).length, fixture.outputTokens, 'resumed UI duplicated output tokens');
    assert.equal((await lookup())?.is_done, true);
    assert.deepEqual(errors, [], 'runtime errors while restoring active generation');
    assert.equal(posts.filter(request => request.headers()['x-conversation-id']).length, 1);
    assert.equal(await resumedResponse.finished(), null, 'automatic replay reader failed');
    console.log(JSON.stringify({ activeReload: 'passed', networkRecovery: 'passed', outputTokens: fixture.outputTokens,
      generationPosts: posts.filter(request => request.headers()['x-conversation-id']).length,
      automaticReplayRequests: replayRequests.length, pageErrors: errors }));
    await stopActiveGeneration(browser, backendOrigin, fixture.outputTokens);
  } catch (error) {
    throw new Error(`${error.message}\n${diagnostics}`, { cause: error });
  } finally {
    if (page) await page.close();
    if (proxy) {
      const closed = once(proxy.server, 'close');
      proxy.server.close();
      proxy.server.closeAllConnections();
      await closed;
    }
    await stopServer(server);
    await rm(fixture.directory, { recursive: true, force: true });
  }
}

const { chromium } = await import(process.argv[2] ? pathToFileURL(process.argv[2]).href : 'playwright');
const port = 18080;
const origin = `http://127.0.0.1:${port}`;
const server = spawn('target/debug/onnx-genai-server', [
  'tests/fixtures/tiny-llm', '--addr', `127.0.0.1:${port}`, '--max-output-tokens', '32'
], { stdio: ['ignore', 'pipe', 'pipe'] });
let diagnostics = '';
for (const stream of [server.stdout, server.stderr]) stream.on('data', data => { diagnostics = (diagnostics + data).slice(-32000); });
let browser;
try {
  await waitForServer(origin, server, () => diagnostics);
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
  await activeReload(browser);
} finally {
  if (browser) await browser.close();
  await stopServer(server);
}
