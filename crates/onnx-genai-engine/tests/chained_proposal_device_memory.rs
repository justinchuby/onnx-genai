//! A device-resident proposal chain gives its device memory back.
//!
//! Every step of a chain running on a device allocates: a gathered embedding
//! row, a narrowed carry aliasing the proposer's output binding, a truncated
//! state cell on a rejection. Each of those keeps its owner alive exactly as
//! long as it is used, and the proof that the ownership is right is that the
//! allocations go back.
//!
//! # Why this is its own binary
//!
//! The assertion is a *process-global* live-allocation count, which is what
//! makes it attributable: unlike device-wide free memory it cannot be moved by
//! another process on the same GPU. But it also means any sibling test in the
//! same binary that builds or drops a CUDA engine while this one is looping
//! shifts the count under it. Cargo gives each integration test file its own
//! process, so a file of one test is the scope this measurement needs.

use onnx_genai_engine::{Engine, EngineConfig, EngineDecodeBackend, NativeDecodeDevice};

#[path = "common/chained.rs"]
mod chained;

/// Free device memory this test tolerates another *process* on the GPU moving.
///
/// Attributable retention is caught exactly by the allocation count below. This
/// second reading is a coarse backstop for retention the count cannot see — an
/// execution-provider arena that grows without bound, above all — so its
/// tolerance is set to absorb a neighbour rather than to resolve kilobytes.
const SHARED_GPU_NOISE_BYTES: usize = 64 * 1024 * 1024;

fn native_cuda_engine(root: &std::path::Path) -> anyhow::Result<Engine> {
    Engine::from_dir(
        root,
        EngineConfig {
            decode_backend: EngineDecodeBackend::Native,
            native_device: Some(NativeDecodeDevice::Cuda { index: Some(0) }),
            ..EngineConfig::default()
        },
    )
}

#[test]
fn repeated_proposals_do_not_retain_device_memory_native_cuda() -> anyhow::Result<()> {
    let root = chained::fixture_root();
    let mut native = chained::ChainedFixture::new(native_cuda_engine(&root)?)?;

    // Warm up: the first rounds allocate the session's arena, the embedding
    // table's device mirror, and the fused input buffer, none of which are
    // per-proposal.
    for _ in 0..8 {
        native.propose(chained::PROMPT_TOKENS, 4)?;
    }
    let settled_allocations = onnx_genai_ort::cuda_rt::live_allocations();
    let settled_free = onnx_genai_ort::cuda_rt::device_memory_info(0)?.free_bytes;

    for _ in 0..100 {
        native.propose(chained::PROMPT_TOKENS, 4)?;
    }

    // 100 proposals of width 4 make 400 fused-input buffers, 400 gathered
    // embedding rows and 400 narrowed carries. A single one of them outliving
    // its step moves this number by hundreds.
    assert_eq!(
        onnx_genai_ort::cuda_rt::live_allocations(),
        settled_allocations,
        "100 proposals left device allocations behind that the first eight did not; an owning \
         value is outliving the step that made it"
    );
    let free = onnx_genai_ort::cuda_rt::device_memory_info(0)?.free_bytes;
    assert!(
        free + SHARED_GPU_NOISE_BYTES >= settled_free,
        "100 proposals consumed {} bytes of device memory, beyond the {SHARED_GPU_NOISE_BYTES} \
         this test allows a neighbouring process on the same GPU to move; something outside this \
         process's own tracked allocations is retaining device memory",
        settled_free - free
    );
    Ok(())
}
