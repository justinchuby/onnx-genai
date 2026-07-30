#!/usr/bin/env bash
#
# Locating the models directory, shared by anything that needs a real model.
#
# Models are gitignored and are not part of a clone, and they commonly live in
# a primary checkout rather than in a worktree. Every script that wants one
# therefore has to search, and searching wrong is not a loud failure -- it is a
# check that silently does not run.
#
# THE RULE THIS FILE EXISTS TO ENFORCE: a candidate must CONTAIN the model, not
# merely exist. `models/` is a shared dumping ground; `.hf_cache` lives there
# and torchinductor creates `.scratch` on its own, so an empty-but-present
# `models/` is the NORMAL state of a fresh worktree. Testing `[[ -d ]]` on the
# directory selects it and defeats the fallback entirely. That is not
# hypothetical: examples/serving-dashboard/run-demo.sh ran green, then began
# failing with no edit to it, because an unrelated tool created models/.scratch
# and the worktree directory started winning.
#
# The candidate LIST is name-based (a sibling checkout called onnx-genai), but
# no candidate is ever ACCEPTED on its name -- acceptance is a content test.
# Names generate guesses; content decides. A wrong guess costs nothing.
#
# Usage:
#   . "$(dirname "$0")/lib/models_dir.sh"
#   models_root="$(resolve_models_dir "$REPO_ROOT")"
#   if scatter="$(resolve_model_dir "$REPO_ROOT" qwen2.5-0.5b-scatter-v2)"; then ...

# models_dir_candidates <repo-root>
#
# Prints the directories to search, most-preferred first, one per line. Paths
# are normalised when they exist, because these strings are shown to a human in
# the skip banner and `repo/../onnx-genai/models` is materially harder to act
# on than the real path it denotes.
models_dir_candidates() {
  local raw
  for raw in "$1/models" "$1/../onnx-genai/models"; do
    if [ -d "$raw" ]; then
      (cd "$raw" && pwd -P)
    else
      printf '%s\n' "$raw"
    fi
  done
}

# models_dir_contains_model <dir>
#
# True when <dir> looks like a models directory that actually holds a model.
# A model is a directory containing model.onnx -- the file every consumer
# needs. Checking for the weights rather than for the folder is what makes an
# empty-but-present candidate lose.
models_dir_contains_model() {
  local dir="$1" entry
  [ -d "$dir" ] || return 1
  for entry in "$dir"/*/; do
    [ -f "${entry}model.onnx" ] && return 0
  done
  return 1
}

# resolve_model_dir <repo-root> <model-name>
#
# Prints the path to <model-name> in the first candidate root that actually
# holds it, and returns 0. Returns 1 and prints nothing when no candidate has
# it. MODELS_DIR, when set, is the only place searched -- an explicit override
# must never silently fall through to somewhere the caller did not name.
#
# Resolving per MODEL rather than per DIRECTORY matters: a checkout may hold
# the scatter model but not the dynamic one, and a whole-directory match would
# then send both lookups to a root that satisfies only one of them.
resolve_model_dir() {
  local root="$1" model="$2" candidate
  if [ -n "${MODELS_DIR:-}" ]; then
    [ -f "${MODELS_DIR}/${model}/model.onnx" ] || return 1
    printf '%s' "${MODELS_DIR}/${model}"
    return 0
  fi
  while IFS= read -r candidate; do
    if [ -f "${candidate}/${model}/model.onnx" ]; then
      printf '%s' "${candidate}/${model}"
      return 0
    fi
  done <<EOF
$(models_dir_candidates "$root")
EOF
  return 1
}

# resolve_models_dir <repo-root>
#
# Prints the models ROOT to use. Falls back to the in-repo path when nothing is
# found, so callers produce a sensible error message naming the expected
# location rather than an empty string.
resolve_models_dir() {
  local root="$1" candidate
  if [ -n "${MODELS_DIR:-}" ]; then
    printf '%s' "${MODELS_DIR}"
    return 0
  fi
  while IFS= read -r candidate; do
    if models_dir_contains_model "$candidate"; then
      printf '%s' "$candidate"
      return 0
    fi
  done <<EOF
$(models_dir_candidates "$root")
EOF
  printf '%s' "$root/models"
}
