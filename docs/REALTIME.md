# Realtime audio implementation

`/v1/realtime` upgrades to a WebSocket for packages declaring the native quantum
ABI below. It supports turn-based text/audio, committed-audio transcription and
input-driven duplex sessions. Ordinary buffered transcription packages do not
automatically acquire realtime capability. Remaining conformance work is listed
at the end; this is not a claim of complete llama-server parity.

## Native quantum ABI v1

Schema v1.13 introduces a `kind: realtime`, `version: '1'` task profile with a
`realtime` block. A package declares its supported `default_mode`, a model-authored
`prompt_template`, voice/language selection policies, supported `modes` (`realtime`,
`transcription`, `duplex`), positive `input_sample_rate_hz`, positive
`input_frame_samples`, optional positive `output_sample_rate_hz`, and four
required opaque application inputs:

| Input binding | Tensor | Meaning |
| --- | --- | --- |
| `inputs.output_token_budget` | int64 scalar | Remaining response output-token allowance, or -1 for unlimited |
| `inputs.operation` | int64 scalar | Control operation below |
| `inputs.audio` | float32 symbolic vector | Mono normalized PCM, empty except audio operations |
| `inputs.tokens` | int64 symbolic vector | Package-assembled prompt/text/tool-result IDs, empty when unused |
| `inputs.token_mask` (optional) | bool symbolic vector | Empty for unconstrained selection, otherwise one admission bit per tokenizer ID; apply before native sampling |

Output bindings live in `profile.outputs`. `tokens` is an int64 symbolic vector;
`audio`, when declared, is a float32 symbolic vector at
`output_sample_rate_hz`. Both vectors may be empty. `done` and `needs_input` are bool scalars. `needs_input` explicitly suspends
polling until new input; empty publications never imply readiness or completion.
`prompt_tokens` and `output_tokens` are nonnegative int64 scalars reporting this
quantum's actual usage, including codec/token expansion. All are materialized
`role: tensor` outputs. The workflow must emit every declared output on every
quantum, including control operations. Input PCM is normalized to [-1, 1]. Output PCM must be finite; PCM16
encoding saturates finite model peaks using llama-server's symmetric 32767 scale.

| Operation | int64 value | Meaning |
| --- | --- | --- |
| StartRealtime | 0 | Initialize turn-based realtime state using initial prompt IDs |
| StartTranscription | 1 | Initialize transcription state using initial prompt IDs |
| StartDuplex | 2 | Initialize input-driven duplex state using initial prompt IDs |
| Audio | 3 | Accept 1 through `input_frame_samples` PCM samples |
| Commit | 4 | Finalize the current input audio turn |
| Text | 5 | Accept package-assembled conversation text or external tool results |
| Generate | 6 | Begin a requested turn-based response using the supplied header token IDs |
| Poll | 7 | Advance one bounded model quantum without new input |
| Cancel | 8 | Cancel the current response and publish its terminal state |

`done` ends a response, not the native session. Empty audio/text output does not
mean end of response. Duplex polling is not governed by ordinary generation's
`max_new_tokens` default. Ordinary request-sourced/runtime inputs are rejected
for this ABI so generation defaults cannot silently alter quantum execution.
The remaining response allowance is passed unchanged to each invocation; it is
not a per-quantum token cap. A constrained quantum still emits at most one text
token, independently of that allowance. Reported output usage must cover every
emitted text token so an underreported counter cannot evade the response limit.
A model may finish an unfinished grammar when the actual response allowance is
exhausted; premature natural completion remains a contract error.

The workflow must obey `output_token_budget`; excess reported output is a
contract violation that requires native reset or connection closure. The model workflow owns its codec, temporal/acoustic
steps, delay buffers and response state. It must declare session-scoped state
and `publication_mode: commit_only`: each quantum commits independently, so a
connection does not retain an uncommitted whole-session transaction.

`Engine::open_realtime_session` resolves admission before allocating a native
session. Its returned owner exclusively borrows the engine and closes the
session on drop, including initialization failure and disconnect unwinding.
Transport code cannot supply an arbitrary native session identifier.

The audio resampler retains a rational clock and bounded filter history across
incoming messages. Only end of an audio turn flushes its tail. Buffered WAV
resampling uses the same implementation. Sample rates come from the admitted
package and session configuration.

The `realtime` fixture is a deterministic native state-continuity test, not an
ASR, codec or speech model and not evidence of recognition quality or speed.

The server's native connection task reserves an otherwise idle worker for the
connection lifetime. Ordinary turns and session placement exclude that worker;
existing persistent sessions prevent a new exclusive reservation. A bounded
single-producer mailbox serializes native quanta. The command owns the worker
and capacity guards until native session cleanup completes. Disconnect probes
cancel an in-flight quantum; worker shutdown independently wakes the connection
and joins the owner thread without waiting for its client.

The wire encoder emits response creation, item/part additions, text/audio and
function-argument deltas, completion events and actual output-token usage.
Content indices follow first appearance; repeated canonical tool-call snapshots
do not duplicate items or argument bytes. A consumed terminal encoder cannot
emit another `response.done`. Audio samples are not retained in the response
object. PCM16 output follows llama-server's symmetric 32767 scaling; input is
little-endian PCM16 normalized by 32768. Invalid odd byte counts are rejected
with a contextual input error.

The PCM framer bounds both queued source bytes and resampling expansion before
allocating output, including the filter's pending tail. Message boundaries keep
the same rational sampling clock; commit flushes the final partial native frame.
The native integration fixture publishes an audio delta before input commit,
proving that this path does not wait for a complete audio turn to execute.
Response text/tool arguments, individual audio events and final transcription
text have explicit caller-supplied budgets.

Session configuration is parsed into typed defaults for mode, PCM input rate,
modalities, token budget, tool choice and voice/language conditioning. Invalid
updates are atomic. First audio/message permanently freezes instructions,
tools, session type and output voice/language; later responses do not unfreeze
that prompt state. Per-response overrides leave session defaults intact.
Realtime and Responses function definitions share one canonical lowering and
the package's existing tool protocol admission.

Conversation bookkeeping distinguishes input turns from active responses and
outstanding external tool calls. Auto-commit consumes its input turn once;
stale response completions and repeated/unmatched tool results fail explicitly.
Duplex opens from audio with no ordinary generation cap. Transcription requires
a committed audio item and uses its own response lifecycle. These transitions drive the native conditioning bridge and WebSocket runner.

`realtime.prompt_template` uses the canonical bounded Jinja renderer. Its context
includes `operation` (`session_start`, `user_text`, `tool_result`,
`response_start`), `text`, `call_id`, `mode`, `instructions`, `voice`, `language`,
`transcription_language`, `transcription_prompt`, and a session-settings object.
The template owns all model control/header spellings; stream tokenization adds
no BOS/EOS automatically. Voice/language policies explicitly declare
`{kind: unsupported}`, `{kind: any}`, or `{kind: choices, values: [...]}`.
Configuration changes reach a real native reset/start before first data, and
independent transcription turns start with fresh native conditioning.

HTTP upgrade and framing use Hyper and tungstenite. Independent bounded reader,
writer and native protocol tasks preserve backpressure. Native `needs_input`
prevents busy polling while a duplex model waits for audio. Writer deadlines,
disconnect cancellation and worker shutdown bound cleanup. An ordinary GET does
not reserve a worker or impersonate a WebSocket handshake. Transport/prompt
budgets are configurable through `ServerConfig.realtime` / `RealtimeLimits`.

## Native constrained sampling contract

A workflow declaring `inputs.token_mask` consumes the host grammar mask before
selecting its text token. A constrained quantum emits at most one text token;
the next quantum receives a mask advanced from the actual accepted token.
The host verifies mask membership and grammar completion before publishing
output, and a contract violation requires native state reset. Audio codec
state remains workflow-owned. An empty mask means unconstrained selection.
Logical text constraints check the decoded text directly for completion. Raw
token grammars require declared EOS IDs. After a text ending, an all-false mask
permits codec-only quanta to finish without admitting further text.

## Response termination

A token-limited response ends with upstream-compatible `status: completed`;
an explicit client cancellation uses `status: cancelled`. Both preserve the
connection for later turns. An unfinished tool envelope at either intentional
stop is not admitted as a completed tool call. Complete calls still pass name
and parameter validation, and a natural model stop must satisfy required-tool
postconditions. Raw partial function-call fragments are not yet projected onto
wire call items; completed batches retain atomic admission.

`response.cancel` is idempotent: without an active response it emits no error
or acknowledgement, including before the first turn and after completion. An
active response still executes native cancellation and publishes its terminal
events. A repeated cancel cannot cancel a later response that has not started.

Realtime error envelopes match the pinned llama-server classification: an empty
`response.create` uses `error.type: conversation_empty`, and an empty audio
commit uses `error.type: input_audio_buffer_commit_empty`. Other rejected client
events use `invalid_request_error`; native failures use `server_error` and close
the connection after flushing the error. `error.code` and `error.param` are null.
Every envelope retains its actionable message and generated top-level event ID.
The native text WebSocket regression checks both specialized errors, repeated
idle cancellation before/after a response, ordinary invalid requests and a
successful budgeted native response on the same connection.

## Remaining conformance work

Turn-based tagged-json tool forcing is connected for workflows declaring
`inputs.token_mask`, using canonical decoder text and declared EOS roles.
Profiles without that binding retain protocol/parameter postconditions; native
forcing for other protocols and duplex tool responses remains open. Tool
parameter schemas compile through the shared llguidance byte-grammar backend
before configuration is installed, and complete call arguments must match that
grammar before publication. This uses the backend's supported schema language,
not a claim of general JSON Schema validation. Protocol parsing, parameter
admission and required-call postconditions do not substitute for
constraint-guided sampling.
Expand wire acceptance for tools, malformed/pressure cases, overlapping turns,
rate epochs, multimodel isolation and native device execution. The current
synthetic CPU fixture proves protocol/data flow and budgets, not recognition,
synthesis quality or GPU performance. Input-rate changes are rejected while an active transcription has queued
next-turn PCM; retry after its completed event. Duplex cancellation discards
old queued audio and resets native conditioning before the next response,
matching the upstream stream/KV lifecycle.

### Executed malformed-input recovery

`websocket_rejected_frames_and_pcm_pressure_preserve_native_transcription`
opens a real TCP WebSocket against a Native CPU ONNX workflow. Each case below
must return a distinct `error` event with `invalid_request_error`, after which
the same connection remains usable:

| Rejected input | Required diagnostic |
| --- | --- |
| Truncated JSON text | Valid JSON required |
| Binary WebSocket frame | JSON text frames required |
| JSON null without an event type | String `type` required |
| Unknown event type | Recognized realtime event required |
| Invalid audio base64 | Valid base64 PCM16 required |
| One decoded audio byte | Even PCM16 byte count required |
| 642 decoded bytes with a 640-byte queue budget | Remaining PCM queue budget exceeded |

After all rejections, changing the session to transcription must still succeed.
A valid 640-byte PCM message then fills the configured budget and commits to an
actual native transcription, with matching item IDs, deltas and terminal text.
This proves rejection did not consume input quota or freeze conditioning. It
does not establish oversized WebSocket frame closure, slow-reader backpressure,
all rate epochs or device execution conformance.

### Executed model isolation and reconnect

`websocket_models_isolate_native_state_and_release_closed_worker` loads two
independent Native CPU model entries from the same authored duplex fixture and
opens both routed WebSocket sessions simultaneously. Each handshake identifies
its selected model and a distinct session. An additional connection to the
occupied first model receives HTTP 503 before upgrade; it cannot steal that
worker or prevent the second model from executing.

The fixture accumulates ingested audio in native state. Zero audio in the first
model and nonzero audio in the second produce independent PCM output. Closing
and reopening the first connection releases its worker and creates fresh native
state, while the second connection preserves its previous accumulated state.
The test checks complete native frame sizes and actual output bytes after
reconnect. The close handshake and retry-on-busy cleanup wait are bounded. This
establishes routed CPU state isolation and reconnect cleanup, rather than device
isolation, arbitrary worker-pool sizing or recognition/synthesis quality.

Tool execution belongs exclusively to the external harness. This transport
accepts definitions/results and emits model calls; it provides no tool runtime.
A Native CPU WebSocket regression covers two tool-call rounds with an external
`function_call_output`, and rejects wrong or repeated call IDs. A separate wire
regression requires a schema-invalid model call to produce an error immediately
after response creation, with no call item or argument delta sent to the harness. Its scripted
fixture verifies protocol/data flow rather than model tool-selection quality.
A completed call batch is checked against declared function names and fully
rendered within its resource budget before any call in that batch is sent to
the harness. A late invalid call cannot publish an earlier partial batch.
