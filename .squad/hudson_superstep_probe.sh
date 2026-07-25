#!/usr/bin/env bash
# Hudson in-situ barrier probe A/B. Persistent SPMD pool forced (32 workers).
# Round-robins ONNX_GENAI_DECODE_SUPERSTEP={0,1,4} so drift cancels; load-gates
# < THRESH before every generation; median-of-N. Also records generated_token_ids
# per config so token-exactness across N can be audited (must be identical).
set -u
cd /home/justinchu/onnx-genai-cpu-superstep
export LD_LIBRARY_PATH=$(find target -type d -path '*onnx-genai-ort-sys*/out/ort-prebuilt/lib' 2>/dev/null | head -1):${LD_LIBRARY_PATH:-}
BIN=./target/release/profile_native
MODEL=/home/justinchu/.foundry/cache/models/Microsoft/qwen3-0.6b-generic-cpu-4/v4
PROMPT="The capital of France is"
OUT=.squad/hudson_runs
mkdir -p "$OUT"
THRESH=${THRESH:-10}
ROUNDS=${ROUNDS:-5}
THREADS=${THREADS:-32}
CPUSET=${CPUSET:-0-47}

wait_for_load() {
  while :; do
    L=$(awk '{print $1}' /proc/loadavg)
    awk "BEGIN{exit !($L < $THRESH)}" && return 0
    echo "  [load-wait] $L >= $THRESH ($(date -u +%H:%M:%S))" >&2
    sleep 20
  done
}

run() { # N -> prints "tok/s|token_ids"
  local n=$1
  wait_for_load
  taskset -c $CPUSET env ONNX_GENAI_CPU_DECODE_PERSISTENT_POOL=1 \
    ONNX_GENAI_CPU_DECODE_THREADS=$THREADS ONNX_GENAI_DECODE_SUPERSTEP=$n \
    $BIN --model "$MODEL" --tokens 64 --warmups 1 --runs 1 --steady \
    --decode-skip 8 --ep cpu --backend native --prompt "$PROMPT" 2>/dev/null \
    | awk '
        /steady_median/{for(i=1;i<=NF;i++) if($i ~ /throughput=/){sub("throughput=","",$i); t=$i}}
        /generated_token_ids/{ids=$0}
        END{print t "|" ids}'
}

declare -a S0 S1 S4
IDS0=""; IDS1=""; IDS4=""
echo "[probe] warmup..." >&2
run 0 >/dev/null; run 1 >/dev/null; run 4 >/dev/null
for r in $(seq 1 $ROUNDS); do
  o0=$(run 0); o1=$(run 1); o4=$(run 4)
  S0+=("${o0%%|*}"); S1+=("${o1%%|*}"); S4+=("${o4%%|*}")
  IDS0="${o0#*|}"; IDS1="${o1#*|}"; IDS4="${o4#*|}"
  echo "[probe] round $r: N0=${o0%%|*}  N1=${o1%%|*}  N4=${o4%%|*}" | tee -a "$OUT/probe.log"
done
median() { printf '%s\n' "$@" | sort -n | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
echo "[probe] === MEDIANS (n=$ROUNDS, 32 workers) ===" | tee -a "$OUT/probe.log"
echo "[probe] N=0 (0 extra/step)   = $(median "${S0[@]}") tok/s   samples: ${S0[*]}" | tee -a "$OUT/probe.log"
echo "[probe] N=1 (~196 extra/step)= $(median "${S1[@]}") tok/s   samples: ${S1[*]}" | tee -a "$OUT/probe.log"
echo "[probe] N=4 (~784 extra/step)= $(median "${S4[@]}") tok/s   samples: ${S4[*]}" | tee -a "$OUT/probe.log"
echo "[probe] token-exact N0==N1: $([ "$IDS0" = "$IDS1" ] && echo YES || echo NO)  N0==N4: $([ "$IDS0" = "$IDS4" ] && echo YES || echo NO)" | tee -a "$OUT/probe.log"
