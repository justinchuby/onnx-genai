### 2026-08-18: Reasoning content is sent to the client; the privacy gate is withdrawn
**By:** Squad (Coordinator), on the owner's instruction (@justinchuby)
**What:** The server must stream and return reasoning content to callers. The
"keep it private" gate proposed in #1224 is cancelled. Any test asserting that
reasoning is *absent* from a response (e.g. `reasoning_never_streams`) is
asserting the wrong behaviour and must be inverted, covering the streamed
deltas and the final non-streamed message as separate paths.
**Why:** Owner decision. Agent clients are expected to see the reasoning turn.
Note the budget half of #1224 ("give a reasoning turn room to finish") is
unaffected and still wanted.
