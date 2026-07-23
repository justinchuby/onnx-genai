# DeepSeek MLA fix merged

DeepSeek-V2-Lite real-weight garbage decode was root-caused to a CUDA `dtod` stream-ordering race and fixed in `1fe314f`, now merged to `origin/main`. Holden reviewed green; Marsten confirmed GLM-4-9b dense native decode remains coherent and cuda-graph active.
