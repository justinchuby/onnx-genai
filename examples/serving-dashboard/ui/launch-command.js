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

/** Default address the server binds, matching the documented launch command. */
export const DEFAULT_SERVER_ADDRESS = '127.0.0.1:8123';

/** The model directory the demo is built and verified against. */
export const RECOMMENDED_MODEL_DIR = 'models/qwen2.5-0.5b-scatter-v2';

/**
 * The exact, copy-pasteable launch command. Shown verbatim in both blocking
 * failure states and quoted in the README.
 */
export const LAUNCH_COMMAND = [
  'cargo build --release -p onnx-genai-server',
  '',
  `ONNX_GENAI_EP=cpu ./target/release/onnx-genai-server \\`,
  `  --model ${RECOMMENDED_MODEL_DIR} \\`,
  '  --model-id qwen-scatter \\',
  `  --addr ${DEFAULT_SERVER_ADDRESS} \\`,
  '  --enable-debug-endpoints',
].join('\n');

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
