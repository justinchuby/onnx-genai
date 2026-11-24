# Stdio MCP tools for Llama Chat

`nxrt serve MODEL --mcp-servers-config mcp.json` discovers and invokes tools from
operator-configured stdio MCP servers. The same configuration can be supplied
with `--mcp-servers-json`; the environment variables are `NXRT_MCP_SERVERS_CONFIG`
and `NXRT_MCP_SERVERS_JSON`. These options execute configured programs with the
server's privileges. Use trusted configuration and clients, as with host tools.

```json
{
  "mcpServers": {
    "files": {
      "command": "/absolute/path/to/your-mcp-server",
      "args": [],
      "env": { "PROJECT_ROOT": "/path/to/project" },
      "cwd": "/path/to/project",
      "timeout_ms": 30000
    }
  }
}
```

`command` and its `args` are passed directly, without a shell. Environment entries
override the inherited process environment. Relative `cwd` values resolve against
the nxrt process's starting directory. The file is read before inline JSON;
duplicate server names retain the first entry. Entries without a stdio command
are skipped. Combined configuration is limited to 1 MiB and 32 servers; syntax,
field types, invalid environment entries and timeout errors identify their source.

MCP configuration is independent of `--tools`: no host built-ins are enabled by
these options. Use `--tools read_file,grep_search` or `--tools all` separately when
those privileges are wanted. HTTP MCP proxying remains a separate opt-in feature
using `--ui-mcp-proxy`.

## Discovery and calls

The server performs bounded tool discovery before accepting HTTP requests.
Library users constructing `app(state)` directly get the same discovery on first
use of `/tools`. Discovery never resolves or loads a model. Unavailable MCP
servers are logged with bounded stderr diagnostics and omitted from the catalog;
a host-credit refusal remains a retriable failure rather than caching an empty
catalog. Each server's discovery is capped at ten seconds, including pagination.

`GET /tools` includes definitions named `<server>_<tool>` with `type: "mcp"`,
`uses_cwd: false` and `permissions.write: false`, matching upstream metadata.
For example, a `read_file` tool advertised by `files` becomes `files_read_file`.
Use the returned function schema when constructing `params`:

```sh
curl http://127.0.0.1:8080/tools
curl http://127.0.0.1:8080/tools \
  -H 'Content-Type: application/json' \
  -d '{"tool":"files_read_file","params":{"path":"README.md"}}'
```

Enabled built-ins win name collisions, followed by the first discovered MCP
definition. MCP tool calls do not support `stream:true`. Text content parts are
joined with newlines into `plain_text_response`; non-text parts are omitted,
and `isError` and JSON-RPC application errors become ordinary tool `error`
results. This follows the pinned llama.cpp stdio integration's normalization.

As with built-ins, model-supplied `cwd`, `runtime` and `resp_type` fields are
removed. Explicit `x-tool-cwd` and `x-resp-type` frontend headers are forwarded as
MCP argument controls; they do not change the configured process directory.
The trusted-origin and JSON content-type policy in [host tools](SERVER_TOOLS.md)
also applies to MCP. Unsupported `x-tool-runtime` requests fail explicitly.

## Lifetime and resource bounds

The official Rust MCP SDK handles the `2024-11-05` initialization lifecycle,
request IDs, notifications and peer requests. Discovery sessions are closed and
reaped; execution lazily starts a new session and reuses it. Calls serialize per
configured server, while independent servers use independent lanes. `timeout_ms`
defaults to 30000 and includes queued time. Expired queued calls do not execute.
Application errors keep a healthy session; transport failures close it and apply
a five-second restart cooldown. Host-credit failures can retry without cooldown.

Disconnect and timeout send a correlated cancellation notification, allow a
bounded ping barrier, then stop the session. Initialization is never cancelled
with a protocol notification. Unix teardown stops the process group even if its
parent already exited. Admission and connection credit remain owned until the
child is reaped. The node owns a separate runtime and joins its actors and reaper
at shutdown, including when the caller's Tokio runtime has already ended.

`--mcp-max-message-bytes` defaults to 262144 (256 KiB), and can be raised to the
upstream 8388608-byte (8 MiB) incoming stdio line limit. Outgoing arguments retain
the HTTP request admission and 32 MiB request-body bound. Larger incoming messages require additional
`--tools-max-host-bytes`. At the default message limit, a connection reserves
32 MiB for SDK/session staging; each incoming request/notification additionally
holds at most 8 MiB plus control credit until its SDK handler releases it.
Discovery temporarily reserves 128 MiB for a catalog, reducing that to its
retained size afterwards. Using the full 8 MiB message limit needs approximately
1.5 GiB or more of host-tool credit; multiple live sessions need more.

The shared host-tool governor also covers configuration parsing, immutable
catalogs, queued request arguments and response/error ownership. Credit follows
response byte clones through slow consumers and survives node shutdown. An
accepted call's buffers remain owned through cancellation. Arbitrary child
process RSS is outside this I/O budget. The SDK response cache is disabled;
incoming notification tasks retain their own canonical leases.

Each server's catalog is limited to 4 MiB / 512 tools / 128 pages, with repeated
pagination cursors refused. The combined exposed catalog is limited to 4 MiB.
These limits fail explicitly; they do not fabricate successful tool execution.
Container/SSH tool runtimes remain separate compatibility work.
