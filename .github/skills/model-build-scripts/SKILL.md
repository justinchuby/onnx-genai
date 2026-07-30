---
name: model-build-scripts
description: Conventions and traps for the shell scripts under scripts/ that export models with Mobius (build_qwen.sh, build_real_model.sh). Read this before writing or editing any bash in this repo, before adding a Mobius-based build script, or when a build script fails with "unbound variable" or "No module named mobius".
---

# Model build scripts

## Write bash that runs on bash 3.2

macOS still ships **bash 3.2.57** as `/bin/bash`. A `#!/usr/bin/env bash`
shebang picks up Homebrew bash 5 *only if* it is on PATH, so a script that
works on the author's machine can be broken for everyone else. Always test
with `/bin/bash` explicitly.

The trap that actually broke `build_qwen.sh`: under `set -u`, bash 3.2 treats
expansion of an **empty** array as an unbound variable and aborts.

```bash
set -euo pipefail
args=()
echo "${args[@]}"        # bash 3.2: "args[@]: unbound variable" -> exit 1
echo ${args[@]+"${args[@]}"}   # correct on 3.2 and 5.x
```

`${#args[@]}` (the count) is safe on empty arrays; only value expansion is not.

Also unavailable in 3.2: associative arrays (`declare -A`), `mapfile`/
`readarray`, and `${var^^}` / `${var,,}` case conversion (use `tr`).

Note that `while ... done < <(cmd)` and `done <<EOF` run the loop body in the
*current* shell, so assignments and `return` inside them work. Piping into a
`while` does not — the body runs in a subshell and assignments are lost.

## Getting Mobius

[Mobius](https://github.com/onnxruntime/mobius) is the exporter that turns a
HuggingFace checkpoint into the ONNX package this runtime loads. Two traps:

1. **The repo is `onnxruntime/mobius`.** `justinchuby/mobius` 404s.
2. **Never `pip install mobius`.** That name belongs to an unrelated project on
   PyPI. Ours is distributed as `mobius-ai`, is *not* published to PyPI, and
   must be installed from source:
   ```bash
   pip install "git+https://github.com/onnxruntime/mobius"
   ```
   If someone installs the squatter, `import mobius` succeeds but
   `python -m mobius` fails — `scripts/lib/mobius_env.sh` detects this by
   checking for `mobius/__main__.py` and says so explicitly.

## Reuse the discovery helper

Do not hardcode a Mobius path and do not assume `python` exists (many machines
have only `python3`). Source the shared helper:

```bash
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source-path=SCRIPTDIR
# shellcheck source=lib/mobius_env.sh
. "$ROOT/scripts/lib/mobius_env.sh"

mobius_resolve "$ROOT"          # exits non-zero with install instructions
"$MOBIUS_PYTHON" -m mobius build ...
```

It sets `MOBIUS_PYTHON`, `MOBIUS_PYTHONPATH` and `MOBIUS_SOURCE`, trying an
explicit `MOBIUS_DIR`, then an installed package, then a sibling checkout. It
picks the interpreter **by whether it can import mobius**, not by name, so
conda/venv setups keep working. An explicit `MOBIUS_DIR` that is invalid is a
hard error — it must never silently fall back to a different Mobius than the
one the user asked for.

Probing uses `importlib.util.find_spec`, which locates `torch`/`transformers`
without importing them. Keep it that way; importing torch makes preflight take
seconds.

## Script conventions

- Validate inputs and print a fix, not a traceback. Every error should say what
  was wrong and give a copy-pasteable command.
- Support `DRY_RUN=1` to print the command without building. This is what makes
  the scripts testable without a multi-GB download.
- Verify the output package after building (`genai_config.json`, `model.onnx`,
  `tokenizer.json`). A partial export otherwise fails much later inside the
  runtime with a confusing message.
- Keep `--help` accurate; it is the discoverable documentation for the env vars.

## Testing

`scripts/build_qwen_test.sh` is the model for this: it runs the script with
`DRY_RUN=1` under **both** `/bin/bash` (3.2) and the default bash, asserts on
the generated command line, and simulates a fresh clone using an interpreter
without Mobius plus an empty `HOME`. Run it, and `shellcheck -x`, after any
change:

```bash
scripts/build_qwen_test.sh
shellcheck -x scripts/build_qwen.sh scripts/lib/mobius_env.sh
```

## Which model does the demo need?

`STATIC_CACHE=1` produces `models/qwen2.5-0.5b-scatter`, whose graph contains
`TensorScatter` and pre-allocated `key_cache.N`/`value_cache.N` inputs. Only
these `-scatter` static-cache models engage continuous batching; a plain
dynamic-cache model silently falls back to the per-request path. If a batching
panel reads flat, check the model before debugging anything else.
