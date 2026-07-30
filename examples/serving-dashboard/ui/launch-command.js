// Copyright (c) Microsoft Corporation.
//
// The launch command, defined ONCE.
//
// The spec requires that the command shown in the README and the command shown
// in the failure states are the same string, so they cannot drift. Two copies of
// a command that must match is a bug waiting for a rename; one exported constant
// is not. The README quotes this file.
//
// The flags are not decoration:
//   --model         the server needs a model directory; without it the demo
//                   reaches the "reachable but no model" blocking state.
//   -scatter-v2     continuous batching engages ONLY on static-cache (-scatter)
//                   models. On a plain model the server silently falls back to
//                   the per-request path and the batching scenario goes flat —
//                   the demo appears to disprove the exact claim it exists to
//                   demonstrate. This is why the model is not a placeholder.
//   --enable-debug-endpoints
//                   /v1/debug/kv and /v1/debug/config carry the KV, prefix-cache
//                   and context-length fields. Without it those panels degrade
//                   to "unavailable" and the model card loses context length.

/**
 * The demo runs TWO servers, because continuous batching and paged KV are
 * mutually exclusive in this runtime — one process cannot demonstrate both.
 * Keyed by the SAME `serverClass` vocabulary scenario-origins.js uses, so a
 * scenario can name the server it needs without a second mapping to drift.
 *
 * A single set of constants could only ever describe one of the two, which is
 * how "start the other server" ended up as generic advice: the page had no way
 * to say WHICH server or WITH WHICH MODEL, and a visitor cannot act on that.
 */
export const SERVERS = Object.freeze({
  scatter: Object.freeze({
    serverClass: 'scatter',
    address: '127.0.0.1:8123',
    modelDir: 'models/qwen2.5-0.5b-scatter-v2',
    modelId: 'qwen-scatter',
    label: 'static-cache server',
    // What it is FOR, in the visitor's terms rather than the engine's.
    demonstrates: 'continuous batching',
  }),
  dynamic: Object.freeze({
    serverClass: 'dynamic',
    address: '127.0.0.1:8124',
    modelDir: 'models/qwen2.5-0.5b',
    modelId: 'qwen-dynamic',
    label: 'dynamic-model server',
    demonstrates: 'paged KV block allocation',
  }),
});

/**
 * The exact, copy-pasteable launch command for one server.
 *
 * @param {keyof typeof SERVERS} serverClass
 * @returns {string}
 */
export function launchCommandFor(serverClass) {
  const server = SERVERS[serverClass];
  if (!server) {
    // Loud rather than a blank command block: a failure state that renders an
    // empty <pre> looks like the page is still loading, and the visitor waits.
    throw new RangeError(
      `Unknown server class "${serverClass}". Known: ${Object.keys(SERVERS).join(', ')}.`,
    );
  }
  return [
    // ANCHOR THE WORKING DIRECTORY FIRST. Every path below is repo-relative,
    // and a visitor pastes this into whatever directory they happen to be in.
    // Without this line the command is silently environment-dependent: it works
    // for whoever wrote it and fails for everyone else, with a cargo error that
    // names a manifest rather than the real problem.
    //
    // `git rev-parse --show-toplevel` rather than a hardcoded absolute path,
    // because the page is served from someone else's checkout, not ours.
    'cd "$(git rev-parse --show-toplevel)"',
    '',
    'cargo build --release -p onnx-genai-server',
    '',
    `ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server \\`,
    `  --model ${server.modelDir} \\`,
    `  --model-id ${server.modelId} \\`,
    `  --addr ${server.address} \\`,
    // The server resolves demo assets against its WORKING DIRECTORY
    // (crates/onnx-genai-server/src/demo_assets.rs). Passing this explicitly
    // means /demo still serves after the `cd` above is edited away, and it is
    // the flag @732c7548's check-launch-command.test.js already requires of
    // README.md and run-demo.sh -- this file was the disclosed exception.
    '  --demo-assets-dir examples/serving-dashboard \\',
    '  --enable-debug-endpoints',
  ].join('\n');
}

/** Default address the server binds, matching the documented launch command. */
export const DEFAULT_SERVER_ADDRESS = SERVERS.scatter.address;

/** The model directory the demo is built and verified against. */
export const RECOMMENDED_MODEL_DIR = SERVERS.scatter.modelDir;

/**
 * The exact, copy-pasteable launch command. Shown verbatim in both blocking
 * failure states and quoted in the README.
 */
export const LAUNCH_COMMAND = launchCommandFor('scatter');

/**
 * Where the visitor opens the demo once the server is up.
 *
 * The trailing slash is load-bearing, not cosmetic. `/demo` is served as a
 * redirect to `/demo/`, so the short form costs a round-trip AND breaks any
 * byte-exact comparison between this constant, the README and index.html.
 */
export const DEMO_URL = `http://${DEFAULT_SERVER_ADDRESS}/demo/`;

/**
 * Why the `-scatter-v2` model specifically. Shown in the failure states so a
 * visitor who substitutes a different model knows what they will lose before
 * they spend an hour debugging a flat chart.
 */
export const MODEL_CHOICE_NOTE =
  'Continuous batching engages only on static-cache (-scatter) models. With a plain model the ' +
  'server falls back to the per-request path and the batching scenario shows no overlap.';
