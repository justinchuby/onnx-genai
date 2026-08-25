set -u
BIN=./target/release/bench_decode_gap
M=.seb_fixt/decode_shaped.onnx
echo "##### CHECK 1: concurrent arm (--sessions 2) must attribute, not cry 'dead instrument'"
$BIN --model $M --iters 256 --arm null --native-threads 2 --sessions 2 2>&1 \
  | grep -E "steady window|counter window|ATTRIBUTION|route:|native pool|WARNING"
echo "##### CHECK 2: single session, counter window vs sample window"
$BIN --model $M --iters 256 --arm null --native-threads 2 2>&1 \
  | grep -E "steady window|counter window|ATTRIBUTION|route:|native pool|WARNING"
