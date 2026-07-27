### 2026-07-27: Conservative graph-level compute-in-place slice for #85
**By:** Holden
**What:** Added `Kernel::can_run_in_place(input_index)` (default false) and executor dead-input transfer for one-output, equal-shape/dtype, contiguous, exclusively owned inputs. CPU unary elementwise opts in. `ONNX_GENAI_COMPUTE_IN_PLACE=0` keeps the out-of-place reference path.
**Why:** The executor's structural last-use plan plus runtime exclusions for graph/external outputs, views, pinned/shared/sequence storage, external bindings, and layout mismatch keeps aliasing conservative. Binary, normalization, CUDA, and capture-aware persistent-buffer planning remain deferred until each kernel proves identical-range read/write safety.
