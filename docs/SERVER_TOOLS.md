# Host tools for Llama Chat

Start with `nxrt serve MODEL --tools all`, or select a comma-separated list such
as `--tools read_file,file_glob_search,grep_search`. These tools run with the
server's filesystem and process privileges. Enable them for trusted clients;
use an authenticated reverse proxy when exposing the server outside a trusted
network. `--tools` is an execution capability, not an operating-system sandbox.

`GET /tools` returns llama.cpp-compatible definitions, display names, write
permissions and working-directory support. `POST /tools` invokes an enabled tool:

```json
{"tool":"read_file","params":{"path":"src/main.rs","start_line":1,"end_line":20,"append_loc":true}}
```

Seven built-ins are available:

| Tool | Result |
| --- | --- |
| `read_file` | `plain_text_response`; optional 1-based inclusive line range and line-number prefixes |
| `file_glob_search` | matching `entries`, absolute `base` and `plain_text_response` |
| `grep_search` | matching text, paths, optional line numbers and context |
| `exec_shell_command` | combined stdout/stderr and exit/timeout information |
| `write_file` | written path and byte count; creates parent directories |
| `edit_file` | exact replacements against the original file; rejects ambiguous/overlapping edits |
| `get_info` | operating-system information and effective working directory |

`x-tool-cwd` selects the working directory. Relative paths resolve against it,
with leading `~` expansion and lexical `.`/`..` normalization. `x-resp-type:
base64` selects binary output from `read_file`, returning `base64` and
`size_bytes`. The similarly named fields inside model-supplied `params` are
ignored: these controls belong to the frontend headers.

`edit_file` supports the upstream fallback normalization of smart quotes,
dashes, spaces and trailing whitespace. Replacements are checked together
before publication. Untouched lines retain their original bytes. Files are
published through a sibling staging file and retain existing permissions.

File searches use Git's tracked/untracked listing and ignore rules when
available. Repository fsmonitor hooks are explicitly disabled for this read-only
operation. Directory listings use a bounded filesystem walk; common generated
and VCS directories are reported but not traversed. A basename glob matches at
any depth; slash-containing globs are anchored with `**/` unless explicitly
anchored. Grep supports Unicode expressions, backreferences and lookarounds,
with bounded compiled size and backtracking work. Literal search bypasses regex
interpretation.

## Streaming and resources

`exec_shell_command` accepts `stream:true` and emits SSE JSON objects containing
`chunk`, followed by `{"done":true}` or a terminal `error`. This is the upstream
tool stream format, without an OpenAI `[DONE]` sentinel. UTF-8 fragments are
retained across reads. Disconnect and operation timeout stop the child process;
on Unix they also stop its process group, including when the shell has already
exited. Child cleanup retains admission until the process is reaped.

Default limits are eight in-flight calls and 256 MiB of host-tool I/O credit:
`--tools-max-concurrent` and `--tools-max-host-bytes` configure them. Request
admission precedes reading its JSON body. Parsing and edit descriptors reserve
16 times the request byte bound plus 256 KiB of control credit; large requests
can require raising the host budget even below the 32 MiB wire limit. After
parsing, unused body credit is released before file-search staging.
Credit follows response bytes until
the last owner releases them; slow consumers continue to occupy admission.
These limits cover this service's Rust I/O buffers, not arbitrary child-process
memory. Filesystem workers retain their admission if a system call cannot yet
return after cancellation.

Tool requests, base64 reads, edits and writes have a 32 MiB size limit. Plain
file output and command output are capped at 16 KiB; larger text files need an
explicit line range. Searches retain at most 65,536 entries / 8 MiB of paths,
return at most 100 matches, and report truncation. Grep input files are bounded
at 32 MiB, output at 8 MiB, patterns at 64 KiB, compiled regular components sharing 2 MiB of NFA and DFA budget, and
backtracking at one million steps. Backtracking expressions additionally limit
expression/input work to 65,536 units; recursive subroutines and absent-operator
extensions are refused. Searches have a 15-second operation deadline;
commands default to 10 seconds and cap at 60. Other calls have a 60-second
operation deadline. Unreadable/oversized search inputs are reported as truncated
results rather than complete searches.

## HTTP policy and remaining compatibility

Tools are disabled by default. Browser origins are limited to loopback by
default; `--tools-allow-origin https://trusted.example` adds an exact trusted
origin. Credentialed CORS applies only to `/tools`. Non-browser clients may
omit `Origin`. JSON content type is required before any side effect.

Disabled tools or untrusted origins return 403, unknown tools and unsupported
streaming return 404, invalid JSON/arguments return 400, oversized requests
return 413, and saturated admission/host credit returns 503 with `Retry-After`.
Filesystem/process failures are ordinary tool result objects containing `error`.
`/props` reports `nxrt_capabilities.server_tools` independently of the model's
ability to generate tool calls. Discovery and execution do not load a model.

This increment supplies the seven host built-ins. MCP stdio discovery/invocation
and container/SSH tool runtimes remain separate compatibility work.
`x-tool-runtime` is refused explicitly; a request for an isolate never falls back
to host execution. The HTTP MCP proxy is a separate opt-in transport.
