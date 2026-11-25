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
configuration. Loading still uses the configured backend and the package's
explicit workflow. A GGUF repository or quantization selector is not converted
into an ONNX workflow automatically. Selective GGUF sidecars remain unsupported;
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
