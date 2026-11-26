//! Canonical, explicitly mapped positive-alpha unquantized GGUF LoRA factors.
use super::{Holder, JsonTarget, MemoryGovernor, factor_storage::FactorStorage};
use anyhow::{Context, ensure};
use onnx_genai_metadata::{
    AdapterArtifact, AdapterScaleEncoding, AdapterServiceContract, AdapterWeightArtifact,
};
use onnx_runtime_loader::{GgufMetadataValue, GgufViews};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    io::Read,
    path::Path,
};

#[derive(Clone, Copy, Serialize)]
pub(super) struct ContainerIdentity {
    pub digest: [u8; 32],
    pub bytes: u64,
}

fn string_fact<'a>(views: &GgufViews<'a>, key: &str) -> anyhow::Result<&'a str> {
    match views.adapter_metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(value),
        Some(value) => anyhow::bail!(
            "GGUF LoRA metadata {key:?} must have STRING wire type, got {value:?}; repair the adapter container"
        ),
        None => anyhow::bail!(
            "GGUF LoRA metadata {key:?} is missing; supply an explicit adapter container rather than a base-model GGUF"
        ),
    }
}
pub(super) fn load(
    artifact: &AdapterArtifact,
    service: &AdapterServiceContract,
    weight: &AdapterWeightArtifact,
    bytes: &[u8],
) -> anyhow::Result<HashMap<String, JsonTarget>> {
    let storage = FactorStorage::declared(artifact)?;
    ensure!(
        weight.scale_encoding == AdapterScaleEncoding::AlphaOverRank,
        "GGUF LoRA requires alpha_over_rank; zero/missing alpha or baked scaling requires separate explicit support"
    );
    let views = onnx_runtime_loader::gguf_views(bytes).with_context(|| {
        format!(
            "parse GGUF LoRA {}@{} source {:?}",
            artifact.identity, artifact.version, weight.location
        )
    })?;
    let general_type = string_fact(&views, "general.type")?;
    ensure!(
        general_type == "adapter",
        "GGUF general.type must be 'adapter', got {general_type:?}; use a LoRA adapter rather than a model container"
    );
    let adapter_type = string_fact(&views, "adapter.type")?;
    ensure!(
        adapter_type == "lora",
        "GGUF adapter.type must be 'lora', got {adapter_type:?}; other adapter algorithms require explicit support"
    );
    let alpha = match views.adapter_metadata.get("adapter.lora.alpha") {
        Some(GgufMetadataValue::Float32(alpha)) => f64::from(*alpha),
        Some(value) => anyhow::bail!(
            "GGUF adapter.lora.alpha must have FLOAT32 wire type, got {value:?}; re-export the LoRA alpha explicitly"
        ),
        None => anyhow::bail!(
            "GGUF adapter.lora.alpha is missing; this loader requires explicit positive alpha and alpha_over_rank scaling"
        ),
    };
    ensure!(
        alpha.is_finite() && alpha > 0.0,
        "GGUF adapter.lora.alpha must be finite and positive, got {alpha}; zero/no-alpha scaling is unsupported, re-export an ordinary positive-alpha adapter"
    );
    for key in views.adapter_metadata.keys() {
        ensure!(
            matches!(*key, "general.type" | "adapter.type" | "adapter.lora.alpha"),
            "GGUF adapter metadata {key:?} has unsupported semantics; aLoRA/DoRA and other numeric/activation variants cannot be silently treated as ordinary LoRA"
        );
    }
    let mut used = HashSet::new();
    let mut targets = HashMap::new();
    for binding in &artifact.bindings {
        let rank = binding.rank.unwrap_or(artifact.rank);
        ensure!(
            binding.alpha.unwrap_or(artifact.alpha) == alpha,
            "GGUF adapter.lora.alpha={alpha} disagrees with {}@{} binding {:?} alpha={}; correct the authoritative manifest instead of changing the numeric scale",
            artifact.identity,
            artifact.version,
            binding.target,
            binding.alpha.unwrap_or(artifact.alpha)
        );
        let target=service.target_manifest.targets.iter().find(|target|target.id==binding.target).with_context(||format!("GGUF adapter {}@{} source {:?} binding target {:?} is undeclared; add its exact component/initializer geometry to target_manifest",artifact.identity,artifact.version,weight.location,binding.target))?;
        let width = target
            .output_slice
            .as_ref()
            .map_or(target.output_features, |slice| slice.width);
        let mut factors = Vec::new();
        for (suffix, shape) in [
            ("lora_a", [rank, target.input_features]),
            ("lora_b", [width, rank]),
        ] {
            let name = format!("{}.{}", binding.weight_key, suffix);
            ensure!(
                used.insert(name.clone()),
                "GGUF saved tensor {name:?} is bound more than once; provide unique exact source prefixes"
            );
            let view=views.tensors.get(&name).with_context(||format!("GGUF factor {name:?} is missing; binding.weight_key must be the exact saved parameter prefix for the standard .lora_a/.lora_b pair"))?;
            ensure!(
                view.dtype == storage.dtype() && view.packed_format.is_none(),
                "GGUF factor {name:?} dtype {:?}/packing {:?} is unsupported; saved A/B factors must exactly match the declared storage dtype; quantized blocks are unsupported",
                view.dtype,
                view.packed_format
            );
            ensure!(
                view.shape == shape,
                "GGUF factor {name:?} canonical row-major shape {:?} must be {shape:?}; correct rank/features/slice in the explicit target binding (GGML dimensions are already reversed by the container loader)",
                view.shape
            );
            let mut factor = Vec::new();
            factor.try_reserve_exact(view.data.len()/storage.width()).with_context(||format!("allocate governed F32 GGUF factor {name:?} from {:?}; free host memory or export smaller factor tensors",weight.location))?;
            for value in view.data.chunks_exact(storage.width()) {
                let value = storage.decode(value);
                ensure!(
                    value.is_finite(),
                    "GGUF factor {name:?} contains nonfinite saved values; export finite ordinary LoRA factors"
                );
                factor.push(value);
            }
            factors.push(factor);
        }
        let b = factors.pop().unwrap();
        let a = factors.pop().unwrap();
        targets.insert(binding.weight_key.clone(), JsonTarget { a, b });
    }
    ensure!(
        views.tensors.keys().all(|name| used.contains(name)),
        "GGUF LoRA contains tensors without exact manifest bindings; map every linear factor pair and reject embedding/norm/unknown tensor semantics"
    );
    Ok(targets)
}

/// Recheck embedded metadata and factors without allocating another container
/// vector on a cache hit. The fixed hashing scratch is governed before use.
pub(super) fn verify_identity(
    root: &Path,
    weight: &AdapterWeightArtifact,
    expected: ContainerIdentity,
    authority: &dyn MemoryGovernor,
) -> anyhow::Result<()> {
    let root = root
        .canonicalize()
        .context("resolve GGUF package root for cached source identity")?;
    let path = root
        .join(&weight.location)
        .canonicalize()
        .with_context(|| format!("resolve cached GGUF source {:?}", weight.location))?;
    ensure!(
        path.starts_with(&root),
        "cached GGUF source {} resolves outside package {}; repair the source path",
        path.display(),
        root.display()
    );
    let file = std::fs::File::open(&path)
        .with_context(|| format!("open cached GGUF source {}", path.display()))?;
    let metadata=file.metadata().with_context(||format!("inspect cached GGUF source {}; restore a readable regular container or reload the package",path.display()))?;
    ensure!(
        metadata.is_file() && metadata.len() == expected.bytes,
        "GGUF adapter source {} changed after loading; restore its immutable container or reload the package with a new adapter version",
        path.display()
    );
    let _lease = authority
        .reserve(
            onnx_runtime_memory_api::Tier::Host,
            8192,
            Holder::CpuDerivedWeights.role(),
            Holder::CpuDerivedWeights.id(),
        )
        .with_context(|| {
            format!(
                "reserve bounded cached GGUF digest scratch for {}",
                path.display()
            )
        })?;
    let mut file = file.take(
        expected
            .bytes
            .checked_add(1)
            .context("GGUF source identity length overflow")?,
    );
    let mut scratch = [0u8; 8192];
    let mut digest = Sha256::new();
    let mut total = 0u64;
    loop {
        let count = file
            .read(&mut scratch)
            .with_context(|| format!("hash cached GGUF source {}", path.display()))?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .context("GGUF digest byte count overflow")?;
        digest.update(&scratch[..count]);
    }
    let actual: [u8; 32] = digest.finalize().into();
    ensure!(
        total == expected.bytes && actual == expected.digest,
        "GGUF adapter source {} changed after loading; restore its immutable container or reload the package with a new adapter version before selecting it",
        path.display()
    );
    Ok(())
}
