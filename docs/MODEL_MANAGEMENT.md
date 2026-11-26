# Managed model packages

Start a model catalog with an operator-owned cache:

```sh
nxrt serve --model-cache-dir ./model-cache
```

The cache can also accompany a positional model, `--models-dir`, or
`--models-config`. Downloading and deleting files remain disabled without
`--model-cache-dir`. This option enables the llama-server `/models` management
protocol; it does not enable the separate `/v1/admin/*` API.

The current managed source is a Hugging Face **nxrt package** containing its
inference metadata, tokenizer and declared model artifacts. The package may use
`inference_metadata.yaml`, `.yml`, `.json`, or the supported GenAI package
configuration, or the supported ORT model-package `manifest.json`. Loading still
uses the configured backend and the package's explicit workflow. Native packages
can bind GGUF/SafeTensors containers through `graph_weights`; containers alone
do not declare an executable graph. Named quantization/variant selectors require
an authored manifest entry. Selective GGUF sidecars remain unsupported;
nonempty `sidecars` or `sidecar` selectors fail explicitly rather than deleting
or downloading a different selection. `/props` advertises
`nxrt_capabilities.model_management` and `managed_model_formats`.

## Download, pause and resume

```http
POST /models
Content-Type: application/json

{"model":"organization/native-package","sidecars":[]}
```

An optional `revision` selects a Git branch, tag or commit; it defaults to
`main`. The server resolves that revision once, validates the complete sized
file plan, reserves cache capacity, persists the plan, and acknowledges with
`{"success":true}`. File transfer then runs in the background. Files are fetched
from that immutable commit; LFS files are checked against their SHA-256. Unsafe
paths, oversized responses, truncated files and mismatched ranges fail before
publication.

`GET /models` includes active transfers as `downloading` and paused or failed
transfers as `incomplete`. An incomplete model is not available for inference;
requests for it receive an actionable unavailable response. After publication,
`in_cache` and `can_remove` are true and `source` is `cache`. It can be loaded
with `POST /models/load` or through ordinary lazy inference.

Pause a running download with:

```http
POST /models/unload
Content-Type: application/json

{"model":"organization/native-package"}
```

Re-post the original `/models` request to resume. A paused or failed transfer
keeps its immutable plan and existing file prefixes, including across a server
restart. Resume uses HTTP Range when a prefix exists. If the source returns a
complete response instead, the server restarts that owned file; an inconsistent
`Content-Range` is rejected without appending to the prefix. To select a different
revision, delete the old entry first.

## Explicit package variants

```http
POST /models
Content-Type: application/json

{"model":"organization/native-package:Q4_K_M","sidecars":[]}
```

The suffix names the exact `manifest.json` variant key. No filename matching,
architecture inference, quantization conversion or fallback to a different
variant occurs. Named selectors use portable identifier characters (letters,
digits, `.`, `_`, `-`). A flat package rejects a named selector.

The server reads the package index at the resolved immutable commit, using the
same component and variant selection rules as local loading. Inline component
indexes and external `component.json` references are supported. The transfer
contains the index documents, the selected `variant_directory`, standard
`shared_assets/` content and explicitly mapped shared-asset roots. Unrelated
variant directories are excluded. Limits and reservations apply to this selected
transfer; each JSON index is bounded to 4 MiB and checked for exact size and any
LFS SHA-256 before planning. All remote file names are validated before index
requests, including unselected names.

The full selector is the model ID for progress, load, inference, resume and
deletion. Its separate variant field persists in cache-manifest **version 2**;
older cache versions are explicitly refused. Use a separate cache directory or a
compatible build to access an older cache; the server does not rewrite or delete
operator files to migrate it. Variants are independent cache entries, including
their shared payloads and quotas. Deleting one entry does not delete another.

For installed packages, `--models-config` accepts the same explicit selection:

```toml
[[models]]
id = "my-model"
path = "/models/native-package"
package_variant = "Q4_K_M"
```

The selected graph, workflow, metadata, tokenizer and serving templates use that
variant consistently. Shared tokenizer assets are supported, with variant-local
prompt assets taking precedence. A contradictory workflow tokenizer declaration
is rejected. Runtime loading validates every selected reference and canonical
path; an explicit full-package audit still checks all declared variants, even
ones that have not been installed.

## Progress and deletion

`GET /models/sse` emits the upstream `{model,event,data}` envelopes. Events
include `model_status`, `status_change`, `download_progress`,
`download_finished`, `download_failed`, `model_remove` and `models_reload`.
Progress uses actual per-file URL counters:

```json
{"model":"organization/native-package","event":"download_progress","data":{"progress":{"https://huggingface.co/organization/native-package/resolve/<commit>/model.onnx":{"done":4096,"total":8192}}}}
```

Every progress snapshot includes the complete file plan, so summing its file
totals gives the package total before later files start. Progress notifications
coalesce while a client is slow. A lagging subscriber
receives a reload signal and a current snapshot. A user-requested pause reports
`download_failed` with `data.stopped: true`; a transfer error retains its files
for retry.

```http
DELETE /models?model=organization%2Fnative-package
```

Deletion pauses any download, removes its persisted partials, or stops an
installed model's workers before removing its published files. An admitted
deletion finishes even if the requesting client disconnects. Operator-configured
model directories cannot be deleted through this API, and configuring a second
model path inside the managed cache is rejected. Interrupted publication and
deletion are recovered from the cache manifests on restart.

## Resource bounds

| Flag | Default | Bound |
| --- | --- | --- |
| `--model-cache-max-bytes` | 64 GiB | Published packages, incomplete reservations and manifests |
| `--model-cache-max-files` | 4096 | All planned package files, including manifests |
| `--model-cache-max-models` | 128 | Published and incomplete models |
| `--model-downloads` | 2 | Concurrent metadata resolutions and transfers |

All bounds are positive. Incomplete transfers reserve their entire planned
capacity, so concurrent downloads cannot overcommit the cache. One server owns
a cache directory at a time. Failed physical cleanup keeps its capacity charged;
repairing permissions and restarting retries the committed cleanup before
admitting new transfers.
