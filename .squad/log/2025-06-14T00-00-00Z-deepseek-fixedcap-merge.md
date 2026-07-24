# DeepSeek MLA fixed-capacity merge — 2025-06-14T00-00-00Z

`53afab0` is merged to `origin/main`, replacing default-domain Attention's capacity-bound KV growth with fixed-slot append while retaining logical valid-length bounds. It improves eager DeepSeek MLA decode by roughly 3–6%, preserves deterministic output and Qwen GQA capture, and adds a padding-garbage regression test.

Rachael's follow-up assessment found the full CUDA-graph capture path is reachable in-engine: replace the growing `Unsqueeze_18` causal/pad mask island with fixed-capacity, device-valid-length-driven kernel handling. No Mobius/export change is required; a device valid-length ABI is the prerequisite.
