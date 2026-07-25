### 2026-07-25: Treat CUDA-targeted rows as the clean native-vs-ORT comparison
**By:** Ripley
**What:** The uncontended H200 sweep records all four requested three-way
measurements, but uses Qwen2.5-0.5B and DeepSeek-R1-Distill-Qwen-1.5B as the
clean competitive native-vs-ORT rows. Phi-3.5-mini and Qwen2.5-Coder-7B ratios
remain explicitly artifact-specific because their generic-CPU exports caused
ORT to insert 67/57 memcpy nodes and partially assign the CUDA EP.
**Why:** GPU 6 was idle throughout, making absolute CUDA rates trustworthy, but
an idle GPU does not remove graph-export and execution-provider assignment
confounds. The distinction preserves the credible 1.556× and 1.421× native
wins without overstating the much larger generic-CPU-artifact ratios.
