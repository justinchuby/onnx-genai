#!/usr/bin/env bash
# Shared Mobius environment discovery for the model-build scripts.
#
# Mobius (https://github.com/onnxruntime/mobius) is the exporter that turns a
# HuggingFace checkpoint into the ONNX package this runtime loads. It is a
# separate repository, so every build script has to answer the same question:
# "which Python interpreter can `import mobius`, and what PYTHONPATH does it
# need?". This file answers that once.
#
# Source it, then call `mobius_resolve`:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/lib/mobius_env.sh"
#     mobius_resolve
#     "$MOBIUS_PYTHON" -m mobius build ...
#
# On success `mobius_resolve` exports:
#   MOBIUS_PYTHON      - interpreter that can import mobius
#   MOBIUS_PYTHONPATH  - PYTHONPATH to run it with (may be empty)
#   MOBIUS_SOURCE      - human-readable description of where mobius came from
#
# On failure it prints actionable install instructions and returns non-zero.
#
# Compatibility: must run on bash 3.2 (the stock macOS /bin/bash) under
# `set -euo pipefail`. That rules out associative arrays, `mapfile`, `${x^^}`,
# and bare `"${arr[@]}"` expansion of a possibly-empty array.

MOBIUS_REPO_URL="https://github.com/onnxruntime/mobius"

# The PyPI distribution is `mobius-ai`; the import name is `mobius`. An
# unrelated project already owns the name `mobius` on PyPI, so `pip install
# mobius` silently installs the wrong library. Always install from source.
MOBIUS_INSTALL_HINT="pip install \"git+$MOBIUS_REPO_URL\""

mobius_log() {
  printf '%s\n' "$*" >&2
}

# Print the list of directories to search for a sibling Mobius checkout.
# Callers pass the repository root.
mobius_candidate_dirs() {
  local root="$1"
  printf '%s\n' \
    "$root/../mobius" \
    "$root/../../mobius" \
    "$HOME/mobius" \
    "$HOME/Documents/GitHub/mobius"
}

# A directory is a usable Mobius checkout if it has an importable src/mobius.
mobius_is_checkout() {
  [ -f "$1/src/mobius/__init__.py" ]
}

# Probe one interpreter for a working Mobius. Echoes a diagnosis on stdout:
#   ok                      - mobius importable, entry point present, deps present
#   missing                 - no mobius module at all
#   wrong-package           - a module named mobius exists but is not mobius-ai
#   missing-deps:<a,b,c>    - mobius found but required third-party deps absent
# Returns 0 if it could run the interpreter at all, 1 otherwise.
#
# Uses importlib.util.find_spec throughout so that heavy dependencies (torch,
# transformers) are located but never actually imported. This keeps the
# preflight fast and lets us report *all* missing deps in one message.
mobius_probe_python() {
  local python_bin="$1"
  local pythonpath="$2"

  command -v "$python_bin" >/dev/null 2>&1 || return 1

  PYTHONPATH="$pythonpath" "$python_bin" - <<'PYPROBE' 2>/dev/null
import importlib.util
import os
import sys

# find_spec on a top-level name locates the module without executing it.
try:
    spec = importlib.util.find_spec("mobius")
except (ImportError, ValueError):
    spec = None

if spec is None:
    print("missing")
    sys.exit(0)

# Distinguish the real exporter (mobius-ai) from the unrelated PyPI package
# that squats the `mobius` import name. Ours ships a __main__ entry point.
locations = list(spec.submodule_search_locations or [])
if not any(os.path.isfile(os.path.join(p, "__main__.py")) for p in locations):
    print("wrong-package")
    sys.exit(0)

required = [
    "huggingface_hub",
    "numpy",
    "onnx_ir",
    "onnxscript",
    "safetensors",
    "torch",
    "transformers",
]
missing = []
for name in required:
    try:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    except (ImportError, ValueError):
        missing.append(name)

if missing:
    print("missing-deps:" + ",".join(missing))
else:
    print("ok")
PYPROBE
}

# Resolve MOBIUS_PYTHON / MOBIUS_PYTHONPATH / MOBIUS_SOURCE.
#
# Discovery order, most explicit first:
#   1. MOBIUS_DIR or MOBIUS_ROOT, if set  (hard error when invalid - an
#      explicit request that cannot be honoured must never fall back silently)
#   2. an already-importable mobius (pip install, editable install, venv)
#   3. a sibling ../mobius checkout
mobius_resolve() {
  local root="${1:-$ROOT}"
  local explicit_dir="${MOBIUS_DIR:-${MOBIUS_ROOT:-}}"
  local candidates_pythonpath=""
  local source_desc=""

  if [ -n "$explicit_dir" ]; then
    if ! mobius_is_checkout "$explicit_dir"; then
      mobius_log "error: MOBIUS_DIR is set to '$explicit_dir', but that is not a Mobius checkout."
      mobius_log "       Expected to find: $explicit_dir/src/mobius/__init__.py"
      mobius_log ""
      mobius_log "Fix it by pointing MOBIUS_DIR at a clone of $MOBIUS_REPO_URL:"
      mobius_log "    git clone $MOBIUS_REPO_URL"
      mobius_log "    MOBIUS_DIR=\$PWD/mobius $0"
      mobius_log ""
      mobius_log "Or unset MOBIUS_DIR to auto-detect an installed Mobius."
      return 1
    fi
    candidates_pythonpath="$explicit_dir/src"
    source_desc="MOBIUS_DIR=$explicit_dir"
  fi

  # Interpreter candidates. An explicit PYTHON wins; otherwise try python3
  # before python, but accept whichever one can actually see mobius. Picking
  # by capability rather than by name keeps existing conda/venv setups working.
  local python_candidates
  if [ -n "${PYTHON:-}" ]; then
    python_candidates="$PYTHON"
  else
    python_candidates="python3 python"
  fi

  # Path candidates, newline-delimited. An empty entry means "try the
  # interpreter as-is", which is how an installed/editable Mobius is found.
  local path_candidates
  if [ -n "$candidates_pythonpath" ]; then
    path_candidates="$candidates_pythonpath"
  else
    path_candidates=""
    local dir
    while IFS= read -r dir; do
      if mobius_is_checkout "$dir"; then
        # Normalise to an absolute path so PYTHONPATH survives a cd.
        path_candidates="$path_candidates
$(cd "$dir" && pwd)/src"
      fi
    done < <(mobius_candidate_dirs "$root")
  fi

  local best_diagnosis=""
  local best_python=""
  local candidate_path python_bin diagnosis effective_path

  while IFS= read -r candidate_path; do
    for python_bin in $python_candidates; do
      # Preserve any PYTHONPATH the caller already had.
      if [ -n "$candidate_path" ]; then
        effective_path="$candidate_path${PYTHONPATH:+:$PYTHONPATH}"
      else
        effective_path="${PYTHONPATH:-}"
      fi

      diagnosis="$(mobius_probe_python "$python_bin" "$effective_path")" || continue
      case "$diagnosis" in
        ok)
          MOBIUS_PYTHON="$python_bin"
          MOBIUS_PYTHONPATH="$effective_path"
          if [ -n "$source_desc" ]; then
            MOBIUS_SOURCE="$source_desc"
          elif [ -n "$candidate_path" ]; then
            MOBIUS_SOURCE="checkout at ${candidate_path%/src}"
          else
            MOBIUS_SOURCE="installed package"
          fi
          export MOBIUS_PYTHON MOBIUS_PYTHONPATH MOBIUS_SOURCE
          return 0
          ;;
        missing-deps:*|wrong-package)
          # Remember the most informative failure to report if nothing works.
          if [ -z "$best_diagnosis" ] || [ "$best_diagnosis" = "wrong-package" ]; then
            best_diagnosis="$diagnosis"
            best_python="$python_bin"
          fi
          ;;
      esac
    done
  done <<EOF
$path_candidates
EOF

  mobius_report_failure "$best_diagnosis" "$best_python" "$root"
  return 1
}

# Turn the best diagnosis we collected into an actionable error message.
mobius_report_failure() {
  local diagnosis="$1"
  local python_bin="$2"
  local root="$3"

  case "$diagnosis" in
    missing-deps:*)
      local deps="${diagnosis#missing-deps:}"
      mobius_log "error: Mobius is importable but its dependencies are missing."
      mobius_log "       Missing: $(printf '%s' "$deps" | tr ',' ' ')"
      mobius_log ""
      mobius_log "Install them into the same interpreter ($python_bin):"
      mobius_log "    $python_bin -m $MOBIUS_INSTALL_HINT"
      ;;
    wrong-package)
      mobius_log "error: found a Python module named 'mobius', but it is not the Mobius exporter."
      mobius_log ""
      mobius_log "There is an unrelated package named 'mobius' on PyPI. The exporter this"
      mobius_log "repository needs is distributed as 'mobius-ai' and is not published to"
      mobius_log "PyPI, so it must be installed from source:"
      mobius_log "    pip uninstall -y mobius"
      mobius_log "    $MOBIUS_INSTALL_HINT"
      ;;
    *)
      mobius_log "error: could not find Mobius, the exporter that builds the ONNX model package."
      mobius_log ""
      mobius_log "Mobius lives in a separate repository and is not published to PyPI."
      mobius_log "Do one of the following, then re-run this script:"
      mobius_log ""
      mobius_log "  1. Install it (recommended):"
      mobius_log "         $MOBIUS_INSTALL_HINT"
      mobius_log ""
      mobius_log "  2. Point at an existing checkout:"
      mobius_log "         MOBIUS_DIR=/path/to/mobius $0"
      mobius_log ""
      mobius_log "  3. Clone it next to this repository ($(cd "$root/.." && pwd)/mobius):"
      mobius_log "         git clone $MOBIUS_REPO_URL $(cd "$root/.." && pwd)/mobius"
      mobius_log ""
      mobius_log "Note: do NOT run 'pip install mobius' - that name belongs to an"
      mobius_log "unrelated project on PyPI. The distribution is 'mobius-ai'."
      mobius_log "See README.md, section 'Build a model with Mobius', for the"
      mobius_log "full prerequisites."
      ;;
  esac
}
