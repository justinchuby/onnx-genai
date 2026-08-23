#!/usr/bin/env bash
# Capture one revision's observable generation behaviour on the pinned packages.
#
# Every package produces lines, and a package that cannot run produces lines
# saying so. A missing directory or a failed load is recorded, not skipped: a
# comparison whose corpus quietly emptied would diff clean and prove nothing,
# which is the failure mode this guards against.
set -u

OUT="$1"; shift
PROBE="$1"; shift

: "${ONNX_GENAI_ORT_LIB:=/home/justinchu/.ort129/onnxruntime/capi/libonnxruntime.so.1.29.0}"
export ONNX_GENAI_ORT_LIB

PINS_ROOT="${EVIDENCE_PINS:-/datadisks/disk1/justinchu/relocated/evidence-pins}"

# package label -> directory. Each is pinned to an exact revision recorded in
# REAL_MODEL_EVIDENCE.md, so a rerun compares the same bytes.
LABELS=(qwen3_0_6b gemma4_e2b gemma4_e2b_speculative)
DIRS=(
  "$PINS_ROOT/qwen3-0.6b-onnx-genai"
  "$PINS_ROOT/gemma4-e2b"
  "$PINS_ROOT/gemma4-e2b-speculative"
)

: > "$OUT"
covered=0
for index in "${!LABELS[@]}"; do
  label="${LABELS[$index]}"
  dir="${DIRS[$index]}"
  if [ ! -d "$dir" ]; then
    printf '%s\tMISSING_DIR\t%s\n' "$label" "$dir" >> "$OUT"
    continue
  fi
  covered=$((covered + 1))
  timeout 1800 "$PROBE" "$dir" "Hello" 16 2>/dev/null \
    | sed "s|^|${label}\t|" >> "$OUT" \
    || printf '%s\tprobe\tERROR exit=%s\n' "$label" "$?" >> "$OUT"
done

sort -o "$OUT" "$OUT"
printf 'COVERED=%s\n' "$covered"
wc -l < "$OUT"
