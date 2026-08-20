//! Decode backend selection and native-request routing.

use super::*;

pub(crate) fn resolve_decode_backend(
    model_path: &Path,
    requested: EngineDecodeBackend,
) -> anyhow::Result<EngineDecodeBackend> {
    let requested = requested_decode_backend(requested)?;
    match requested {
        EngineDecodeBackend::Ort => Ok(EngineDecodeBackend::Ort),
        EngineDecodeBackend::Native => {
            #[cfg(feature = "native-backend")]
            {
                Ok(EngineDecodeBackend::Native)
            }
            #[cfg(not(feature = "native-backend"))]
            {
                let _ = model_path;
                anyhow::bail!(
                    "native decoder backend was requested, but this binary was not built with \
                     native decoder support. Rebuild the CLI with \
                     `cargo build -p onnx-genai-cli --features native-backend` for the native \
                     CPU/backend path, or `--features native-cuda` when you need the native CUDA \
                     EP. To run this model on ONNX Runtime instead, pass --backend ort, set \
                     decode_backend = EngineDecodeBackend::Ort, or set ONNX_GENAI_BACKEND=ort."
                )
            }
        }
        EngineDecodeBackend::Auto => {
            if model_requires_native_backend(model_path)? {
                #[cfg(feature = "native-backend")]
                {
                    return Ok(EngineDecodeBackend::Native);
                }
                #[cfg(not(feature = "native-backend"))]
                {
                    anyhow::bail!(
                        "model contains native-only operators (pkg.nxrt::BlockQuantizedMatMul); \
                         rebuild the CLI with --features native-backend (or --features \
                         native-cuda for the native CUDA EP) and select \
                         decode_backend = EngineDecodeBackend::Native \
                         (or ONNX_GENAI_BACKEND=native)"
                    );
                }
            }
            Ok(EngineDecodeBackend::Ort)
        }
    }
}

pub(crate) fn parse_backend_env(value: &str) -> anyhow::Result<EngineDecodeBackend> {
    match value.trim().to_ascii_lowercase().as_str() {
        "auto" => Ok(EngineDecodeBackend::Auto),
        "ort" => Ok(EngineDecodeBackend::Ort),
        "native" => Ok(EngineDecodeBackend::Native),
        _ => anyhow::bail!(
            "invalid ONNX_GENAI_BACKEND={value:?}; expected one of: auto, ort, native"
        ),
    }
}

pub(crate) fn requested_decode_backend_with_env(
    requested: EngineDecodeBackend,
    env_lookup: impl FnOnce() -> anyhow::Result<Option<String>>,
) -> anyhow::Result<EngineDecodeBackend> {
    if requested != EngineDecodeBackend::Auto {
        return Ok(requested);
    }
    env_lookup()?.map_or(Ok(EngineDecodeBackend::Auto), |value| {
        parse_backend_env(&value)
    })
}

pub(crate) fn requested_decode_backend(
    requested: EngineDecodeBackend,
) -> anyhow::Result<EngineDecodeBackend> {
    requested_decode_backend_with_env(requested, || match std::env::var("ONNX_GENAI_BACKEND") {
        Ok(value) => Ok(Some(value)),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(anyhow::anyhow!(
            "failed to read ONNX_GENAI_BACKEND: {error}"
        )),
    })
}

pub(crate) fn ort_to_native_hint() -> &'static str {
    "ONNX Runtime could not load this model; if it requires native execution, \
     set decode_backend = EngineDecodeBackend::Native (or ONNX_GENAI_BACKEND=native)"
}

pub(crate) fn native_to_ort_hint() -> &'static str {
    "if this model uses operators unsupported by the native backend, \
     set decode_backend = EngineDecodeBackend::Ort (or ONNX_GENAI_BACKEND=ort) \
     to run this model on ONNX Runtime"
}

pub(crate) fn augment_backend_error<T>(
    result: anyhow::Result<T>,
    backend: EngineDecodeBackend,
) -> anyhow::Result<T> {
    let hint = match backend {
        EngineDecodeBackend::Ort => ort_to_native_hint(),
        EngineDecodeBackend::Native => native_to_ort_hint(),
        EngineDecodeBackend::Auto => unreachable!("the selected backend cannot be Auto"),
    };
    result.with_context(|| hint)
}

pub(crate) fn model_requires_native_backend(model_path: &Path) -> anyhow::Result<bool> {
    #[cfg(feature = "native-backend")]
    {
        use prost::Message;

        let bytes = onnx_runtime_loader::read_model_binary(model_path).with_context(|| {
            format!(
                "Failed to inspect model '{}' for native operators",
                model_path.display()
            )
        })?;
        let model = onnx_runtime_loader::proto::ModelProto::decode(bytes.as_slice())
            .context("Failed to parse ONNX model while selecting decoder backend")?;
        Ok(model_proto_requires_native_backend(&model))
    }
    #[cfg(not(feature = "native-backend"))]
    {
        let _ = model_path;
        Ok(false)
    }
}

#[cfg(feature = "native-backend")]
pub(crate) fn model_proto_requires_native_backend(
    model: &onnx_runtime_loader::proto::ModelProto,
) -> bool {
    const DOMAIN: &str = "pkg.nxrt";
    const OP_TYPE: &str = "BlockQuantizedMatMul";
    const OPSET_VERSION: i64 = 1;

    let supports_native_opset = model
        .opset_import
        .iter()
        .any(|opset| opset.domain == DOMAIN && opset.version == OPSET_VERSION);
    supports_native_opset
        && model.graph.as_ref().is_some_and(|graph| {
            graph
                .node
                .iter()
                .any(|node| node.domain == DOMAIN && node.op_type == OP_TYPE)
        })
}

#[cfg(feature = "native-backend")]
pub(crate) fn reject_native_request_speculation(options: &GenerateOptions) -> anyhow::Result<()> {
    // Prompt-lookup is now implemented on the native path (WP2); only the
    // not-yet-ported proposer families are rejected.
    let unsupported = match options.speculative_mode.as_ref() {
        None | Some(SpeculativeMode::None) | Some(SpeculativeMode::PromptLookup { .. }) => None,
        Some(SpeculativeMode::DraftModel) => Some("draft-model"),
        Some(SpeculativeMode::Mtp(_)) => Some("MTP"),
        Some(SpeculativeMode::Eagle3(_)) => Some("EAGLE-3"),
        Some(SpeculativeMode::SharedKv(_)) => None,
    };
    if let Some(mode) = unsupported {
        anyhow::bail!(
            "native decoder backend does not yet support per-request {mode} speculative decoding (only prompt-lookup is implemented)"
        );
    }
    // `num_speculative_tokens` only has meaning alongside an implemented native
    // speculative mode; reject it when no such mode selects native speculation.
    if options.num_speculative_tokens.is_some()
        && !matches!(
            options.speculative_mode.as_ref(),
            Some(SpeculativeMode::PromptLookup { .. } | SpeculativeMode::SharedKv(_))
        )
    {
        anyhow::bail!(
            "native decoder backend does not support the per-request num_speculative_tokens option without a prompt-lookup speculative_mode"
        );
    }
    Ok(())
}

/// Prompt-lookup speculation parameters resolved for a native request.
#[cfg(feature = "native-backend")]
pub(crate) struct NativeSpeculationPlan {
    pub(crate) kind: NativeSpeculationKind,
    pub(crate) width: usize,
}

#[cfg(feature = "native-backend")]
#[derive(Clone, Copy)]
pub(crate) enum NativeSpeculationKind {
    PromptLookup { ngram: usize, max_tokens: usize },
    SharedKv,
}

/// Decide whether a native request should run through the speculative driver.
///
/// Returns `Some` only for an implemented, greedy prompt-lookup request with no
/// processor chain and no logprobs — the exact regime in which host-argmax
/// acceptance reproduces plain greedy selection. Every other request (including
/// non-greedy, processor-chain, logprobs, or the default `None` mode) returns
/// `None` and stays on the untouched plain fast path.
#[cfg(feature = "native-backend")]
pub(crate) fn native_speculation_plan(
    options: &GenerateOptions,
    chain: &crate::logits::ProcessorChain,
) -> Option<NativeSpeculationPlan> {
    let (kind, default_width) = match options.speculative_mode.as_ref()? {
        SpeculativeMode::PromptLookup { ngram, max_tokens } => (
            NativeSpeculationKind::PromptLookup {
                ngram: *ngram,
                max_tokens: *max_tokens,
            },
            *max_tokens,
        ),
        SpeculativeMode::SharedKv(config) => (
            NativeSpeculationKind::SharedKv,
            config.num_speculative_tokens.saturating_add(1),
        ),
        _ => return None,
    };
    let greedy = options.selects_greedily();
    if !greedy || !chain.is_empty() || options.top_logprobs.is_some() {
        return None;
    }
    let width = options
        .num_speculative_tokens
        .map(|value| {
            value.saturating_add(usize::from(matches!(kind, NativeSpeculationKind::SharedKv)))
        })
        .unwrap_or(default_width)
        .max(1);
    Some(NativeSpeculationPlan { kind, width })
}
