# Explicit PEFT LoRA loading

A package can supply `format: hf_peft`, loader capability
`onnx-genai.adapters.hf-peft@1`, `config_location` and `scale_encoding:
alpha_over_rank` in its existing adapter artifact manifest. The runtime selects
the first declared supported JSON/HF PEFT/GGUF source; it never retries another
format after a selected source fails validation.

Each `bindings[].weight_key` is the exact full module prefix in the saved PEFT
file. For example, `base_model.model.explicit.projection` selects
`base_model.model.explicit.projection.lora_A.weight` and the corresponding
`lora_B.weight`. This is an explicit source-to-target mapping: the existing
`target_manifest` still owns the ONNX component, initializer, node/output,
input/output dimensions and any output slice. No architecture, layer count,
initializer naming convention or model family is inferred.

This implementation accepts ordinary linear LoRA saved as F32, F16 or BF16
SafeTensors. The artifact manifest `dtype` names factor storage, not the
base model activation or computation dtype; both A and B must match it exactly.
F16/BF16 values are widened to F32 using the existing IEEE half conversion
library before entering the governed F32 cache. Mixed storage, F64, quantized
factors and dtype guessing are rejected. Portable JSON remains F32-only.

The portable overlay still requires host F32 inputs and base projection outputs;
its factor cache, arithmetic and output remain F32. Half storage support does
not imply F16/BF16 base computation or device-fused LoRA. SafeTensors must have
A shape `[r, input_features]` and B shape `[output_width, r]`, contain finite
values, and all saved tensors must be bound. `peft_type` must be `LORA`, and
config `r`/`lora_alpha` must match the effective artifact/binding rank and alpha.
The existing overlay applies `request_scale * alpha / r * B(Ax)` to the actual
base projection. Bias must be `none`. Inference disables training dropout;
boolean/gaussian initialization descriptions and inference/training mode flags
do not alter saved factor inference. Other initialization variants are rejected.

Explicit `target_modules` name lists are checked against the saved module
prefixes; they do not discover graph targets. Regex or `all-linear` discovery,
embedding/Conv factors, named-adapter tensor spellings, per-target rank/alpha
patterns, modules-to-save, nonstandard bias, weight tying, DoRA/RSLoRA,
quantized/activated/routed/other numeric variants and
unrecognized configuration fields remain unsupported and produce errors.
Known disabled variant defaults are accepted only in their disabled form.
See the upstream [PEFT LoRA configuration](https://github.com/huggingface/peft/blob/v0.22.0/src/peft/tuners/lora/config.py).

Loading reuses the strict SafeTensors parser already used by the native ONNX
weight loader. File/config parsing staging and normalized factor buffers use
the model's existing host memory authority before allocating factor vectors.
Live factor leases charge four bytes per normalized element even for two-byte
source storage; the held parsing staging lease also covers transient converted
factors while they are copied into the live cache.
Both source paths must resolve under the package root. The configuration's
content SHA-256 participates in plan identity and is rechecked before a cached
PEFT activation. Changing any config bytes requires restoring the immutable
file or reloading the package with a new adapter version; a cached tensor hit
cannot conceal an `r`, alpha or variant change. Loaded factors retain their
existing immutable cache lifetime; in-place weight replacement is not a hot
reload API.

`peft_lora` contains synthetic real SafeTensors/Native CPU regressions with a
nonidentity ONNX MatMul, positive/negative/zero runtime scales, configuration
mutation after cache load, unsupported variants, malformed/unmapped/nonfinite
weights and subsequent base/valid-request recovery. Hand-authored half-bit
patterns have independent dyadic expected results for rank-two/multiple-row
Native inference, opposite and zero scales, declared/mixed/unsupported storage,
NaN/Inf, staging-budget refusal and cached-valid-adapter recovery. These fixtures do not prove
large-model, CUDA-fused, distributed or packaged importer acceptance.

The HTTP catalogue shares the engine's portable container selector; a PEFT-only
artifact no longer needs a JSON duplicate. Explicit GGUF LoRA support and its
separate bounded F32 capability are described in [GGUF LoRA](GGUF_LORA.md).
