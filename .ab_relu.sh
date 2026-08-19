#!/usr/bin/env bash
# Interleaved A/B: before = /workspace/dev/resch-base (origin/main c55a3fab3),
# after = /workspace/dev/resch-inline-dims (+ AVX2 Relu/Clip).
set -u
BEFORE=/workspace/dev/resch-base
AFTER=/workspace/dev/resch-inline-dims
CASE="${CASE:-_f32_1m}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-30}"
THREADS="${THREADS:-1}"
PIN="${PIN:-8-15}"
ROUNDS="${ROUNDS:-5}"
OUT="${OUT:-$AFTER/.ab_relu.csv}"

: > "$OUT"
run_arm() {
  local dir="$1" tag="$2" round="$3"
  ( cd "$dir" && \
    NXRT_MM_BENCH=1 NXRT_MM_BENCH_CASE="$CASE" NXRT_MM_BENCH_ITERS="$ITERS" \
    NXRT_MM_BENCH_WARMUP="$WARMUP" NXRT_MM_BENCH_THREADS="$THREADS" \
    ONNX_GENAI_MLAS_THREADPOOL_THREADS="$THREADS" RAYON_NUM_THREADS="$THREADS" \
    taskset -c "$PIN" cargo test --release -q -p onnx-runtime-ep-cpu-plugin \
      --test plugin_ort_e2e plugin_path_ab -- --nocapture --ignored 2>/dev/null ) \
  | grep -E "^bench_" | sed "s/^/$tag,$round,/" >> "$OUT"
}
for r in $(seq 1 "$ROUNDS"); do
  run_arm "$BEFORE" before "$r"
  run_arm "$AFTER"  after  "$r"
  echo "round $r done" >&2
done
echo "wrote $OUT" >&2
