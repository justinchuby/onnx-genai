#!/usr/bin/env node
// Real browser API-key onboarding and Native chat. Run under the shared host lock.
// Usage: node scripts/test_llama_chat_auth.mjs [playwright/index.mjs] [server binary]
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, writeFile, rm, access } from 'node:fs/promises';
import { constants } from 'node:fs';
import { createServer } from 'node:net';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';
import { pathToFileURL } from 'node:url';

const { chromium } = await import(process.argv[2] ? pathToFileURL(process.argv[2]).href : 'playwright');
const binary = resolve(process.argv[3] ?? 'target/debug/onnx-genai-server');
await access(binary, constants.X_OK);
const reservation = createServer();
await new Promise(resolve => reservation.listen(0, '127.0.0.1', resolve));
const port = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const origin = `http://127.0.0.1:${port}`;
const directory = await mkdtemp(join(tmpdir(), 'nxrt-ui-auth-'));
const key = 'nxrt-browser-fixture-key';
const keyFile = join(directory, 'keys.txt');
await writeFile(keyFile, `# Browser fixture key\n${key}\n`, { mode: 0o600 });
const env = { ...process.env };
delete env.LLAMA_API_KEY;
delete env.LLAMA_ARG_API_KEY_FILE;
const server = spawn(binary, ['tests/fixtures/tiny-llm', '--backend', 'native', '--addr', `127.0.0.1:${port}`,
  '--max-output-tokens', '32', '--api-key-file', keyFile], { env, stdio: ['ignore', 'pipe', 'pipe'] });
let diagnostics = '';
let startupError;
server.on('error', error => { startupError = error; });
for (const stream of [server.stdout, server.stderr]) stream.on('data', data => { diagnostics = (diagnostics + data).slice(-16000); });
let browser;
let page;
try {
  const deadline = Date.now() + 60000;
  while (true) {
    if (startupError) throw startupError;
    assert(server.exitCode === null && server.signalCode === null, `fixture server exited: ${diagnostics}`);
    try {
      if ((await fetch(`${origin}/health`, { signal: AbortSignal.timeout(1000) })).ok) break;
    } catch {}
    assert(Date.now() < deadline, `fixture server startup timed out: ${diagnostics}`);
    await delay(100);
  }
  console.log('Authenticated Native fixture server ready');
  browser = await chromium.launch({ channel: 'chrome', headless: true });
  const context = await browser.newContext();
  page = await context.newPage();
  page.setDefaultTimeout(15000);
  const errors = [];
  const completions = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (new URL(response.url()).pathname === '/v1/chat/completions') completions.push(response);
  });
  await page.goto(origin, { waitUntil: 'networkidle' });
  await page.getByRole('button', { name: 'Enter API Key', exact: true }).click();
  const input = page.getByLabel('API Key', { exact: true });
  await input.fill('wrong-fixture-key');
  const wrong = page.waitForResponse(response => new URL(response.url()).pathname === '/props'
    && response.request().headers()['authorization'] === 'Bearer wrong-fixture-key');
  await page.getByRole('button', { name: 'Save & Retry', exact: true }).click();
  assert.equal((await wrong).status(), 401);
  await page.getByText('Invalid API key - please check and try again', { exact: true }).waitFor();
  assert.equal(completions.length, 0, 'invalid credentials must not start chat inference');
  console.log('Incorrect key rejected through the actual login UI');
  await input.fill(key);
  const accepted = page.waitForResponse(response => new URL(response.url()).pathname === '/props'
    && response.request().headers()['authorization'] === `Bearer ${key}`);
  await page.getByRole('button', { name: 'Save & Retry', exact: true }).click();
  assert.equal((await accepted).status(), 200);
  await input.waitFor({ state: 'hidden' });
  const chat = page.locator('.chat-form-editor [contenteditable="true"]').last();
  await chat.waitFor();
  console.log('Valid key accepted; chat editor ready');
  await chat.fill('hello');
  const [response] = await Promise.all([
    page.waitForResponse(response => new URL(response.url()).pathname === '/v1/chat/completions', { timeout: 30000 }),
    chat.press('Enter')
  ]);
  assert.equal(response.status(), 200, await response.text());
  await response.finished();
  const headers = await response.request().allHeaders();
  assert.equal(headers.authorization, `Bearer ${key}`);
  assert(!response.request().postDataJSON().tools?.length, 'a model without tool protocol must not receive UI tools');
  const wire = await response.text();
  const answer = wire.split('\n').filter(line => line.startsWith('data: ') && line !== 'data: [DONE]')
    .flatMap(line => JSON.parse(line.slice(6)).choices ?? []).map(choice => choice.delta?.content ?? '').join('');
  assert(answer.trim(), 'Native inference streamed no answer');
  const rendered = page.getByLabel('Assistant message with actions', { exact: true }).first();
  await rendered.waitFor({ timeout: 30000 });
  await page.getByRole('button', { name: 'Stop', exact: true }).waitFor({ state: 'hidden', timeout: 30000 });
  assert((await rendered.innerText()).includes(answer), 'UI omitted streamed answer content');
  const conversation = headers['x-conversation-id'];
  assert(conversation, 'UI did not send its replay conversation ID');
  const replayUrl = `${origin}/v1/stream?conv_id=${encodeURIComponent(conversation)}&from=0`;
  assert.equal((await fetch(replayUrl, { signal: AbortSignal.timeout(5000) })).status, 401);
  const replay = await fetch(replayUrl, { headers: { Authorization: `Bearer ${key}` }, signal: AbortSignal.timeout(5000) });
  assert.equal(replay.status, 200);
  assert.equal(await replay.text(), wire, 'authenticated replay must preserve the original streamed bytes');
  const persisted = page.waitForResponse(result => new URL(result.url()).pathname === '/props'
    && result.request().headers()['authorization'] === `Bearer ${key}`);
  await page.reload({ waitUntil: 'networkidle' });
  assert.equal((await persisted).status(), 200);
  await page.getByLabel('User message with actions', { exact: true }).first().waitFor();
  assert((await page.locator('body').innerText()).includes('hello'));
  await page.getByRole('button', { name: 'Enter API Key', exact: true }).waitFor({ state: 'hidden' });
  const fresh = await browser.newContext();
  const freshPage = await fresh.newPage();
  await freshPage.goto(origin, { waitUntil: 'networkidle' });
  await freshPage.getByRole('button', { name: 'Enter API Key', exact: true }).waitFor();
  await fresh.close();
  assert.equal(completions.length, 1, 'login or reload duplicated Native generation');
  assert.deepEqual(errors, [], 'browser runtime errors');
  console.log(JSON.stringify({ result: 'passed', authenticatedChats: completions.length,
    wrongKeyRejected: true, loginPersisted: true, replayProtected: true, pageErrors: errors }));
} catch (error) {
  if (page && !page.isClosed()) {
    console.error('Browser state at failure:', JSON.stringify({ url: page.url(), body: (await page.locator('body').innerText()).slice(0, 5000),
      editors: await page.getByRole('textbox').evaluateAll(nodes => nodes.map(node => ({ tag: node.tagName, label: node.getAttribute('aria-label'), placeholder: node.getAttribute('placeholder'), editable: node.getAttribute('contenteditable') }))) }));
  }
  throw error;
} finally {
  if (browser) await browser.close();
  if (server.pid && server.exitCode === null && server.signalCode === null) {
    const exited = once(server, 'exit');
    server.kill('SIGTERM');
    await Promise.race([exited, delay(5000, undefined, { ref: false })]);
    if (server.exitCode === null && server.signalCode === null) server.kill('SIGKILL');
    await exited;
  }
  await rm(directory, { recursive: true, force: true });
}
