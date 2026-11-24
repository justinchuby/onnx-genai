# Native session state files

The Native engine can save and restore its committed host-owned decoder state,
including across engine instances and server restarts. Enable
`EngineConfig.native_state_files = NativeStateFilePolicy::Required` when loading
the model. `WhenSupported` enables files only for supported decoders and lets
unrelated workflow/reference models load normally; `Disabled` is the default. This opts into a
load-time SHA-256 content identity; the default remains disabled and does not
hash model weights for this feature. Model package artifacts must remain
immutable during loading, as required by the runtime's weight mappings.

`Engine::write_native_session_state(session, writer, limits)` writes the active
conversation's complete executable cache and logical history.
`Engine::restore_native_session_state(destination, reader, limits)` verifies a
file and restores it into an existing destination. Both return
`NativeStateFileSummary` with separate logical and materialized token counts and
the encoded byte count. A selected final token can still be pending; loading a
file preserves that distinction and does not evaluate the saved prompt again.

The content identity covers the ONNX model, referenced external weight regions
(including nested graphs, sparse tensors and local functions), external compiled
context payloads resolved by the loader, semantic metadata, derived decoder I/O,
the loaded tokenizer and EOS facts, and decode precision. It is fixed when the
model loads. Saving never re-identifies an already-loaded model from subsequently
changed files. Equivalent tokenizer object key order does not change identity.
An incompatible model, tokenizer, metadata or precision is rejected before state
publication. Opaque in-memory `NativeSessionSnapshot` captures remain bound to
one engine ownership generation; verified files can cross those boundaries.

## Bounds and restore behavior

`NativeStateFileLimits` defaults to 4 GiB per file, 16 MiB per manifest, and
1,048,576 logical tokens. The engine also applies its model context ceiling.
Callers can choose explicit tighter or larger file bounds; host memory admission
still applies. Input metadata, payload copies and CPU row descriptors reserve
memory before allocation. Retained imported state owns its leases until restored
or dropped. Insufficient headroom fails without changing live state.

A restore validates the complete tensor inventory, names, dtypes, ranks, static
geometry, KV cursors, declared compressed-record/carry contracts, hidden width
and token vocabulary. The payload checksum and exact file boundary must verify
before the engine installs any state. Existing state is captured under a
separate undo lease before physical writes. A failed restore rolls back the
previous physical owner; failed undo poisons affected conversations until reset.
CPU in-place KV storage can grow to fit saved live rows while preserving its
previous content. Unused capacity is not serialized.

The codec requires a host-state Native decoder on a little-endian machine.
Device-owned state and interpreted workflow state do not yet expose the required
portable participant. Enabling the flag for an unsupported executor fails at
load; it is never silently ignored. An inactive logical token history is not an
executable cache and cannot be exported through this API.

## File and transport ownership

The engine accepts `Read` and `Write`, not paths. The caller controls path
admission, storage quota, concurrent access and timeouts. Stage a temporary file,
finish the write and any required sync, then publish it atomically. An I/O error
can leave a partial writer output, while the live conversation is unchanged.
The reader must supply one finite file; trailing data is rejected.

The v1 encoding is `NXRTST01`, a 32-byte model identity, little-endian u64 manifest
and payload lengths, a strict JSON layout manifest, contiguous tensor/live-row
payloads, and a SHA-256 checksum over everything preceding the checksum. This is
an nxrt format; llama.cpp binary prompt-cache files are not interchangeable.
The checksum detects corruption; model identity checks compatibility, not an
artifact producer's identity.

The server exposes `GET /slots` for an admitted single-sequence Native decoder.
Its slot ID is 0; queue capacity and logical conversation count do not create
additional slots. `is_processing` follows the owner thread, and `n_past` is the
last completed task's retained physical cursor. `fail_on_no_slot=1` returns 503
while that executor is busy. Unsupported executors return 501 and advertise
`endpoint_slots=false` in model properties.

Start with `--slot-save-path DIRECTORY` to enable
`POST /slots/0?action=save|restore|erase`. Save/restore take a JSON `filename`;
erase takes no filename. Actions queue on the model's owner thread, so a save
waits for an active generation to finish. Restores populate the private cache
used by stateless requests; named conversation histories remain independent.
The file codec remains nxrt-specific, not llama.cpp binary-compatible.

Filenames must be plain names within the opened capability directory. Atomic
staging preserves an existing file on failure; files are private on Unix.
One server owns each directory through a process lock. Defaults bound each
file to 4 GiB, the directory to 16 GiB and entries to 128; configure these with
`--slot-save-max-file-bytes`, `--slot-save-max-total-bytes` and
`--slot-save-max-files`. Atomic replacements need space for both old and staged
files. These bounds include other directory entries, so use a dedicated
directory. A process crash may leave a staging file, counted against the bounds;
remove it while the server is stopped after checking its provenance.
