# Native CUDA EP versus ORT headline

- DeepSeek-V2-Lite native whole-step capture vs eager: block-32 **79.2 vs 44.5 tok/s (1.78×)**; block-128 **84.6 vs 46.8 tok/s (1.81×)**.
- Foundry-local CUDA A/B, same harness and exact token parity: Qwen2.5-0.5B native **902 vs 584 tok/s (1.55×)**; Phi-4-mini native **322 vs 238 tok/s (1.35×)**.
- Native capture worked. ORT capture failed with `ort_value must contain a constructed tensor`, so ORT ran eager.
- `profile_native --backend native|ort|auto` (`d03261c7`) enables repeatable same-harness comparisons while retaining the native header contract.
