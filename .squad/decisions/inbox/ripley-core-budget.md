### 2026-07-25: Keep peak default and add an explicit CPU decode budget
**By:** Ripley
**What:** `onnx-genai generate` and `onnx-genai run` expose `--cpu-cores N`, mapped to the native decode worker-count mechanism with precedence CLI > `ONNX_GENAI_CPU_DECODE_THREADS` > AUTO. The uncapped automatic worker count is unchanged.
**Why:** Shared-machine users need a first-class good-citizen control, while the measured 48-worker default remains the best dedicated-host peak. Persistent workers already pin one worker per selected allowed CPU, so the budget bounds their affinity footprint without requiring a hand-written `taskset`.
