//! Native CPU MaxRoiPool conformance without the GPU feature.

mod common;
#[path = "fixtures/max_roi_pool.rs"]
mod max_roi_pool;

#[test]
fn max_roi_pool_cpu_conformance() {
    max_roi_pool::cpu_conformance();
}
