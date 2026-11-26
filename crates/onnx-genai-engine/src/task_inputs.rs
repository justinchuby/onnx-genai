//! Shared task-owned encoded-media staging with retained memory ownership.
use crate::Engine;
use anyhow::Context;
use onnx_genai_ort::{DataType, Value};
use std::sync::Arc;
impl Engine {
    pub(crate) fn encoded_media_value(
        &self,
        endpoint: &str,
        bytes: &[u8],
    ) -> anyhow::Result<Value> {
        let size = i64::try_from(bytes.len()).with_context(|| {
            format!("encoded media '{endpoint}' exceeds tensor dimensions; reduce the media size")
        })?;
        let holder = crate::engine::memory_plan::Holder::Activations;
        let lease = onnx_runtime_memory_governor::MemoryGovernor::reserve(
            self.governor.memory(), onnx_runtime_memory_api::Tier::Host,
            u64::try_from(bytes.len())?, holder.role(), holder.id(),
        ).with_context(|| format!("reserve encoded media '{endpoint}' input staging; reduce media bytes or increase the host memory budget"))?;
        let mut owned = Vec::new();
        owned.try_reserve_exact(bytes.len()).with_context(|| format!("allocate encoded media '{endpoint}' input staging; reduce media bytes or free host memory"))?;
        owned.extend_from_slice(bytes);
        Value::from_raw_bytes(owned, &[size], DataType::Uint8)?
            .with_owner(Arc::new(lease))
            .map_err(Into::into)
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn encoded_image_lease_survives_tensor_aliases() {
        let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/onnx_genai_workflows/decision_images");
        let engine = Engine::from_dir(&root, crate::fixture_engine_config()).unwrap();
        let bytes = std::fs::read(root.join("white.png")).unwrap();
        let tier = onnx_runtime_memory_api::Tier::Host;
        let before = engine.governor().leased_bytes_on(tier);
        let value = engine
            .encoded_media_value("request.picture", &bytes)
            .unwrap();
        let alias = value
            .try_alias_clone()
            .expect("owned image must retain its lease through aliases")
            .unwrap();
        assert_eq!(
            engine.governor().leased_bytes_on(tier),
            before + bytes.len() as u64
        );
        drop(value);
        assert_eq!(
            engine.governor().leased_bytes_on(tier),
            before + bytes.len() as u64
        );
        assert_eq!(alias.as_raw_bytes().unwrap(), bytes.as_slice());
        drop(alias);
        assert_eq!(engine.governor().leased_bytes_on(tier), before);
    }
}
