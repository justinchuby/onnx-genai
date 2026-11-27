//! Strict CUDA MaxRoiPool conformance, with no CPU fallback.

mod common;
#[path = "fixtures/max_roi_pool.rs"]
mod max_roi_pool;

#[test]
#[cfg_attr(not(feature = "gpu-tests"), ignore = "requires a CUDA device")]
fn max_roi_pool_strict_cuda_conformance() {
    max_roi_pool::strict_cuda_conformance();
}
