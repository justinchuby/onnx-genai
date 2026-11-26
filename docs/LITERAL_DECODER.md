# Fused literal replacement and constrained output

The resolved byte-stream contract admits nonempty `Replace.pattern.String`
after ByteLevel, or after ByteFallback followed by an explicit Fuse. Replacement
content may be empty. The saved tokenizer declaration determines stage order;
there is no model-name or token-glyph inference. Regex and empty replacement
patterns need their own streaming proof and remain explicit capability errors.
Existing token-local Replace/Strip before the byte stage keeps its semantics.

The transformation implements leftmost, nonoverlapping, nonrecursive literal
replacement. A request retains only the unmatched literal prefix, and EOF
releases an unmatched prefix unchanged. Replacement output feeds subsequent
Replace/Strip stages in declaration order; flushing an earlier stage goes
through later stages before those stages finish. A failed stage does not commit
any earlier stage's candidate state.

ByteLevel and ByteFallback are different canonical decoder contracts. ByteLevel
uses standard lossy UTF-8 decoding and retains incomplete scalar bytes. ByteFallback
retains an entire consecutive run of byte tokens: when a run ends or EOF arrives,
it emits the whole valid UTF-8 run, or one U+FFFD for **each byte token** if the
run is invalid. A valid early prefix therefore cannot be published before a
later invalid byte makes the whole run invalid. Ordinary and preserved special
tokens end a run; skipped special tokens do not. Vocabulary piece kinds are
resolved after token-local transforms, before conversion to bytes. Irreparably
invalid runs have the same future semantics at the same run length, so those
states can merge without guessing from their original payloads.

Decoded-content JSON/JSON Schema/Regex/Lark constraints use isolated candidate
forks of this same decoder, its actual token vocabulary, response-channel
projection and grammar cursor. EOF checks the actual decoded terminal output.
A held replacement prefix is not assumed to be an inevitable output suffix.
The finite continuation search deduplicates equal typed vocabulary effects and
checks emitted text before committing a candidate. Raw GBNF/RawLark continue to
recognize the raw vocabulary bytes; publication transformations do not change
the raw grammar's language. Lazy raw replay still requires its existing identity
contract.

Composed transform specifications and finite Strip/literal retention have a
4 MiB contract ceiling. Whole ByteFallback runs use the request's explicit byte
budget, rather than claiming constant UTF-8 lookbehind. Every stage checks
output capacity before allocation, including expansion that a later stage might
delete. A shared work meter charges intermediate scans, counting passes, copies,
replacement emission and EOF work. The continuation frontier additionally limits
states, transitions, memory and byte work; exhaustion is an actionable resource
error, never a fabricated unsatisfiable grammar result. Complex valid decoder /
vocabulary products may still exceed these bounded search capabilities.

Engine-owned requests reserve contextual-decoder workspace from the actual host
memory governor before plan compilation. The processor chain retains the lease
through ordinary generation, returned per-token cursors and continuous-batch
rows. Realtime publication is prepared on the native owner before response creation
and returned to the transport with its own lease at the configured text limit.
Publication-only admission does not reserve an unused constraint frontier. Replacement grammar
admission retains the previous grammar/lease until the new plan and first mask
succeed. Standalone pure validators have explicit local budgets and make no
Engine-governor admission claim. Server-owned requests defer this particular
compilation to the worker's existing pre-SSE admission handshake; schema / decoder
configuration failures remain typed request errors, while governor refusals keep
their resource error classification.

The runtime still uses its owned llguidance fork. Upstream submissions
[#460](https://github.com/guidance-ai/llguidance/pull/460) (typed unsatisfiability)
and [#461](https://github.com/guidance-ai/llguidance/pull/461) (root key policy)
remain pending; they do not replace the fork until a published release passes
the complete runtime regressions.

Executable definitions include independent canonical tokenizers oracles for
split Unicode, overlaps, deletion, invalid byte groups, specials and EOF; a
50,000-piece equivalent vocabulary case; intermediate expansion/work and atomic
failure checks; and real Native generation, cancellation/retry, raw-grammar,
JSON/SSE, usage and failed-admission regressions. Their source presence does not
establish a fresh passing result, all-model/device acceptance or performance.

Canonical dependency semantics are defined by tokenizers
[Replace](https://github.com/huggingface/tokenizers/blob/master/tokenizers/src/normalizers/replace.rs),
[Fuse](https://github.com/huggingface/tokenizers/blob/master/tokenizers/src/decoders/fuse.rs),
[ByteFallback](https://github.com/huggingface/tokenizers/blob/master/tokenizers/src/decoders/byte_fallback.rs)
and [ByteLevel](https://github.com/huggingface/tokenizers/blob/master/tokenizers/src/pre_tokenizers/byte_level.rs).
The implementation audit used the workspace's tokenizers 0.23.2 source.

Declared literal specification sizes are checked using borrowed serialization
before plan allocation or governor admission, including the composed 4 MiB
limit. Allocator and memory-governor failures retain their resource/internal
classification. HTTP host-quota refusals use the existing `429` response,
`Retry-After: 1`, and sanitized `resource_limit_error` body before SSE opens.
