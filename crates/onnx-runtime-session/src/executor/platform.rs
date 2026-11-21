use super::*;

/// Instantiate and initialize the Phase-1 CPU execution provider (§20.7,
/// CPU-only auto-detection). A GPU/accelerator EP would be prepended here in a
/// later phase; for Phase 1 the CPU EP is the sole, always-available backend.
pub(crate) fn auto_detect_cpu_ep() -> Result<Arc<dyn ExecutionProvider>> {
    let mut ep = CpuExecutionProvider::new();
    ep.initialize(&Default::default())?;
    Ok(Arc::new(ep))
}

pub(super) fn validate_cpu_fallback(provider: &dyn ExecutionProvider) -> Result<()> {
    if provider.device_type() != DeviceType::Cpu {
        return Err(SessionError::InvalidFallbackProvider {
            provider: provider.name().to_owned(),
            device: provider.device_id(),
        });
    }
    Ok(())
}

pub(super) fn resolve_cpu_fallback(
    configured: Option<&Arc<dyn ExecutionProvider>>,
) -> Result<Arc<dyn ExecutionProvider>> {
    match configured {
        Some(provider) => {
            validate_cpu_fallback(provider.as_ref())?;
            Ok(Arc::clone(provider))
        }
        None => auto_detect_cpu_ep(),
    }
}
