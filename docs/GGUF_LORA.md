# Explicit GGUF LoRA factor storage

Schema `v1.19` adds `format: gguf` with loader capability
`onnx-genai.adapters.gguf@1`. A package declares the source beneath its adapter
root and explicitly selects `scale_encoding: alpha_over_rank` or `baked`.
No `config_location` is used.
Each `bindings[].weight_key` is the exact saved parameter prefix: the loader
selects only its standard `.lora_a` and `.lora_b` tensors. The existing target
manifest names the actual ONNX component, initializer, node/output and optional
output slice. It supplies rank and input/output geometry. There is no model,
architecture, layer-name or target discovery from GGUF names.

This implementation admits little-endian GGUF v2/v3 and unquantized
GGML F32, F16 (wire type 1) and BF16 (wire type 30) factors. Both factors
must exactly match the manifest storage dtype; mixed storage is rejected.
F16/BF16 values widen to F32 using the same saved-factor classifier and
little-endian conversion as PEFT. Base weights, inputs, live normalized factors
and portable computation remain F32; JSON factors remain F32-only. The canonical
container reader resolves GGML contiguous-first dimensions into row-major
A `[rank, input_features]` and B `[output_width, rank]`; every factor must be
finite and every saved tensor must have an exact manifest binding. Embedding,
norm and other extra tensors cannot silently disappear.

Required typed metadata is `general.type: STRING = adapter`, `adapter.type:
STRING = lora`. For `alpha_over_rank`, `adapter.lora.alpha: FLOAT32`
is required. Alpha must be finite, positive
and exactly agree with every effective manifest/binding alpha. Re-export or
correct the explicit manifest when these facts disagree; the runtime does not
infer or override numeric semantics. The existing portable overlay computes
`request_scale * alpha/rank * B(Ax)` using host F32 inputs, base outputs, factors
and arithmetic. Negative and zero request scales are supported.

For an upstream unity-scale source, `scale_encoding: baked` must be explicitly
declared: the header alpha must be missing or finite FLOAT32 zero (including
negative zero). The existing Baked contract applies no static alpha/rank factor;
positive manifest alpha/target policies remain validated but do not participate
in Baked arithmetic. The loader never infers Baked from a header or rewrites
factors/manifest alpha. Nonzero alpha with Baked, zero/missing alpha with
AlphaOverRank and wrong wire types reject. Negative source alpha is unsupported.

Missing, wrong-type and duplicate metadata are distinct errors. Other
`adapter.*` fields are rejected, including aLoRA activation semantics. General
provenance fields can be present; a `general.architecture` label does not choose
or construct an executable graph. The package author owns the exact source-to-
graph mapping, including feature ordering. Negative source alpha,
DoRA/aLoRA, quantized factors, arbitrary embedding/norm
adapters, automatic target/base-model discovery and device-fused execution
remain unsupported. This is not bare-GGUF model import.

Native unity-scale tests cover all three storage types, independently calculated
rank-two signed/zero row outputs, zero/missing header cache changes and rejection
recovery; HTTP tests cover actual generation and raw embedding without applying
the deliberately nonunity manifest alpha. These definitions still require fresh
execution on the final source revision.

The wire facts and suffix convention follow the upstream
[GGUF LoRA reader](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-adapter.cpp)
and its [scale operation](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-adapter.h).
The upstream [converter](https://github.com/ggml-org/llama.cpp/blob/master/convert_lora_to_gguf.py)
provides explicit F32/F16/BF16 output formats. These references describe the format;
they do not establish parity with every upstream model or its float rounding.

Loading reuses the same GGUF parser as graph initializer binding. File parsing,
normalized live factors and transient copies retain governed leases before
allocation. The loaded whole-container SHA-256 and byte length participate in
cache planning. Each cached activation rechecks the actual file with an 8 KiB
governed streaming scratch buffer; changing embedded alpha or factor bytes
requires restoring the immutable source or reloading a new adapter version.
Failure cannot publish a partial selection. No second binary parser is used.

The server catalogue and actual loader use one engine-owned supported-container
selector. It preserves declared source order and never retries another format
after the selected source fails. HF PEFT-only and GGUF-only artifacts therefore
need no JSON duplicate to be selectable; their paths are reported as declared.

The `gguf_lora` Native CPU tests use independently authored GGUF bytes and a
non-square ONNX MatMul, rank-two/multiple-row signed scales, cache metadata and
payload mutations, invalid metadata/shape/storage/truncation and initial staging
budget refusal with recovery. Independently specified half bit patterns cover
F16/BF16 rectangular rank-two row composition, finite-value and storage rejection,
same-length cached payload changes, initial staging denial and recovery. `lora_container_http` exercises actual Native
generation and raw embeddings for HF PEFT-only/GGUF-only F32/F16/BF16 sources, including
catalogue/default/request scales and invalid-selection recovery. These sources
are executable test definitions, not a fresh passing result or full GPU/model/
streaming acceptance claim. Cache-hit scratch-denial under external host pressure
is a separate acceptance case from initial container staging denial.
