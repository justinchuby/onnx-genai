use std::path::Path;

use onnx_genai_engine::{
    CompressedStatePathStats, NativeDecodeCudaOptions, NativeDecodeDevice, NativeDecodeSession,
    compressed_state_map_lookups, native_cuda_provider_construction_attempts,
};
use onnx_runtime_session::DeviceBindingTransferStats;

fn fixture_model() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/tiny-native-scalar-gqa/model.onnx.textproto")
}

#[test]
#[cfg_attr(
    not(feature = "gpu-tests"),
    ignore = "requires a CUDA device; enable the gpu-tests feature on a CUDA runner"
)]
fn default_off_state_is_zero_work_during_real_cuda_capture_and_replay() -> anyhow::Result<()> {
    const WARM_DECODE_REQUESTS: u64 = 1;
    const CAPTURE_REQUESTS: u64 = 1;
    const REPLAY_REQUESTS: u64 = 8;
    const TOTAL_REQUESTS: u64 = WARM_DECODE_REQUESTS + CAPTURE_REQUESTS + REPLAY_REQUESTS;

    let provider_before = native_cuda_provider_construction_attempts();
    let map_lookups_before = compressed_state_map_lookups();
    let mut session = NativeDecodeSession::load_with_cuda_options(
        fixture_model(),
        NativeDecodeDevice::Cuda { index: Some(0) },
        NativeDecodeCudaOptions {
            decode_batch: None,
            kv_max_len: Some(32),
            metadata_max_len: None,
            graph_capture: Some(true),
            weight_offload_enabled: None,
            weight_offload_stable_va: None,
        },
    )?;
    assert_eq!(
        native_cuda_provider_construction_attempts() - provider_before,
        1,
        "the governed fixture must construct exactly one production CUDA provider"
    );
    let before = session
        .cuda_kv_debug_stats()
        .expect("CUDA session must expose device execution stats");
    assert!(
        before.graph.enabled,
        "fixture must request CUDA graph capture"
    );
    assert_eq!(before.graph.captures, 0);
    assert_eq!(before.graph.replays, 0);
    assert_eq!(before.cuda_decode_submissions, 0);
    assert!(
        !before.device_ptrs.is_empty(),
        "real CUDA KV bindings required"
    );

    for request in 0..TOTAL_REQUESTS {
        session.decode(&[request as u32 % 4], session.current_len())?;
    }

    let after = session
        .cuda_kv_debug_stats()
        .expect("CUDA session must expose post-replay execution stats");
    assert_eq!(session.current_len(), TOTAL_REQUESTS as usize);
    assert_eq!(after.device_ptrs, before.device_ptrs);
    assert_eq!(after.graph.captures, CAPTURE_REQUESTS);
    assert_eq!(after.graph.replays, REPLAY_REQUESTS);
    assert_eq!(after.graph.fallbacks, 0);
    assert_eq!(after.graph.invalidations, 0);
    assert_eq!(after.cuda_decode_submissions, TOTAL_REQUESTS);
    assert!(
        after.graph.allocation_counts.allocations > 0,
        "a real CUDA provider must own device allocations"
    );
    assert_eq!(after.kv_transfers, DeviceBindingTransferStats::default());
    assert_eq!(
        session.compressed_state_path_stats(),
        CompressedStatePathStats::default(),
        "Absent/Disabled compressed state must perform zero lookup, allocation, copy, sync, \
         telemetry, metadata-clone, and environment-read work"
    );
    drop(session);
    assert_eq!(
        compressed_state_map_lookups(),
        map_lookups_before,
        "default-off load/prefill/warm/capture/replay/teardown must not probe state maps"
    );

    eprintln!(
        "default-off CUDA proof: requests={TOTAL_REQUESTS} warm={WARM_DECODE_REQUESTS} \
         captures={} replays={} cuda_submissions={} state={:?}",
        after.graph.captures,
        after.graph.replays,
        after.cuda_decode_submissions,
        CompressedStatePathStats::default()
    );
    Ok(())
}
