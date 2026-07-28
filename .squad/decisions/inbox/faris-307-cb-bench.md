### 2026-07-28: Scheduler-driven continuous-batch benchmark design
**By:** Faris
**What:** Benchmark scheduler-driven continuous batching against sequential generation at 1, 2, 4, and 8 physical rows, with two requests per row to include queue admission and backfill. Materialize the committed scatter fixture as `model.onnx` so the model benchmark and its CI smoke check are runnable.
**Why:** This measures the serving path introduced by #303 under concurrent decode while retaining a deterministic, small CPU fixture and a direct throughput baseline. The pinned-core measurement shows the expected gain at concurrent load (51.01K vs 44.26K tok/s at 2 rows; 75.09K vs 58.35K at 4 rows).
