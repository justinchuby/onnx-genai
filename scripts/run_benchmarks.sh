#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL=0
if [[ "${1:-}" == "--model" ]]; then
  MODEL=1
elif [[ -n "${1:-}" ]]; then
  echo "usage: $0 [--model]" >&2
  exit 2
fi

cargo bench -p onnx-genai-bench --bench no_model
if [[ "$MODEL" == "1" ]]; then
  cargo bench -p onnx-genai-bench --features bench-ort --bench model
fi

CPU="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || awk -F: '/model name/{print $2; exit}' /proc/cpuinfo 2>/dev/null || echo unknown)"
CORES="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo unknown)"
OS="$(uname -srmo 2>/dev/null || uname -srm)"

printf '\nMachine: %s | cores: %s | OS: %s | %s\n\n' "$CPU" "$CORES" "$OS" "$(rustc --version)"

python3 - "$MODEL" <<'PY'
import json
import pathlib
import sys

include_model = sys.argv[1] == "1"
root = pathlib.Path("target/criterion")
rows = []
for estimate in root.glob("**/new/estimates.json"):
    relative = estimate.relative_to(root)
    scenario = "/".join(relative.parts[:-2])
    if scenario.startswith(("report/", "change/")):
        continue
    is_model = scenario.startswith("model_")
    if is_model != include_model and is_model:
        continue
    if not include_model and is_model:
        continue
    data = json.loads(estimate.read_text())
    mean_ns = data["mean"]["point_estimate"]
    benchmark = estimate.parent / "benchmark.json"
    throughput = None
    if benchmark.exists():
        metadata = json.loads(benchmark.read_text())
        throughput = metadata.get("throughput")
    metric = f"{mean_ns:,.1f} ns/iter"
    if throughput:
        kind, count = next(iter(throughput.items()))
        rate = count * 1e9 / mean_ns
        if kind == "Elements":
            if scenario.startswith(("tokenization/", "model_e2e/", "model_batch/")):
                unit = "tokens/s"
            elif scenario.startswith("kv_cache/"):
                unit = "pages/s"
            else:
                unit = "steps/s"
        else:
            unit = {"Bytes": "bytes/s"}.get(kind, f"{kind.lower()}/s")
        metric = f"{rate:,.2f} {unit} ({mean_ns:,.1f} ns/iter)"
    rows.append((scenario, metric))

print("| scenario | metric |")
print("|---|---:|")
for scenario, metric in sorted(rows):
    print(f"| {scenario} | {metric} |")
PY
