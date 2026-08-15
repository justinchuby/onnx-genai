//! Placement cost model for the ORT 2.0 runtime (`docs/architecture/ORT2.md`
//! §6).
//!
//! # The organising principle
//!
//! A cost is the ratio of a kernel's **structure** to a machine's **rates**:
//!
//! ```text
//! cost = structure / rate
//! ```
//!
//! * **Structure** — bytes moved, FLOPs, launch count — is a property of the
//!   kernel and its shapes. It is portable: the same EP binary reports the same
//!   structure on any host. It lives in [`KernelStructure`] and is produced by
//!   execution providers at kernel-match time (`onnx_runtime_ep_api::Cost`'s
//!   `bytes_moved` and `Kernel::estimated_flops`).
//! * **Rates** — memory bandwidth, link bandwidth, FLOP/s, launch latency — are
//!   properties of the machine. They are *not* portable and must be **measured
//!   or supplied**, never hardcoded. They live in [`DeviceProfile`] and
//!   [`TransferProfile`].
//!
//! This crate owns the rate side and the formulas that combine the two. It
//! deliberately does **not** implement §7's ILP placement — that is a separate
//! issue that depends on this crate existing first.
//!
//! # Two non-negotiable invariants
//!
//! 1. **Unknown must be representable** (the #947 lesson). Every rate is
//!    `Option`-shaped and every formula returns `Option<Cost>`. A rate that was
//!    never measured is absent, and a cost that depends on it is `None` — never
//!    a fabricated default that looks confident and is wrong.
//! 2. **Memory bandwidth is parallelism-qualified.** It is stored as a curve
//!    over thread count ([`MemoryBandwidth`]), because measured DRAM bandwidth
//!    spans a 5× range with load; a scalar would be wrong by that factor.
//!
//! # Offline planning
//!
//! [`PlacementCostModel::save`] / [`PlacementCostModel::load`] round-trip the
//! model through JSON so a user can calibrate on their own hardware, save the
//! artifact, and pass it back in for offline cost and plan computation — an
//! explicit product requirement (issue #995).

pub mod calibration;
pub mod device;
pub mod model;
pub mod profile;
pub mod serde_util;
pub mod structure;
pub mod transfer;

pub use device::DeviceKey;
pub use model::{CostModelError, PlacementCostModel};
pub use profile::{ComputeThroughput, DeviceProfile, MemoryBandwidth};
pub use structure::{Cost, KernelStructure};
pub use transfer::{
    HostMemoryKind, MeasuredLinkState, TransferCostMatrix, TransferKey, TransferProfile,
};

#[cfg(test)]
mod integration_tests {
    use super::*;
    use calibration::{
        TransferDirection, TransferFit, apply_memory_bandwidth, apply_transfer_fits,
    };
    use std::time::Duration;
    /// End-to-end: calibrate a model purely from probe measurements, save it,
    /// reload it, and confirm the reloaded model prices ops and transfers
    /// identically — and still returns `None` for what was never measured.
    #[test]
    fn calibrate_save_load_roundtrip() {
        let mut model = PlacementCostModel::new();

        // Host DRAM from roofline_bandwidth (#995 box).
        apply_memory_bandwidth(
            &mut model,
            DeviceKey::cpu(),
            "host",
            [(1, 10.07), (4, 26.50), (8, 37.46), (20, 49.28)],
        );
        // A GPU profile with a single measured DRAM point.
        {
            let gpu = model.device_profile_mut(DeviceKey::cuda(0));
            gpu.name = "RTX 4060 Laptop".to_string();
            gpu.memory_bandwidth = MemoryBandwidth::from_samples([(1, 200.0e9)]);
        }
        // Host<->device link from roofline_transfer (#995 box, idle run).
        apply_transfer_fits(
            &mut model,
            DeviceKey::cpu(),
            DeviceKey::cuda(0),
            &[
                TransferFit {
                    direction: TransferDirection::HostToDevice,
                    pinned: true,
                    latency_base_us: 11.961,
                    bandwidth_gb_s: 11.737,
                },
                TransferFit {
                    direction: TransferDirection::HostToDevice,
                    pinned: false,
                    latency_base_us: 8.294,
                    bandwidth_gb_s: 6.075,
                },
            ],
            true,
            MeasuredLinkState::Active,
        );

        // Price the CPU lm_head GEMV read (389,283,840 B) at 20-thread DRAM.
        let structure = KernelStructure::from_bytes(389_283_840);
        let cpu_cost = model
            .op_cost(&structure, &DeviceKey::cpu(), 20)
            .expect("cpu op cost");
        // 389,283,840 / 49.28e9 ≈ 7.9 ms — the #995 corrected estimate.
        assert!(
            (cpu_cost.time.as_secs_f64() - 7.9e-3).abs() < 0.5e-3,
            "{cpu_cost:?}"
        );

        // Price a pinned H2D upload of the same bytes.
        let up = model
            .transfer_cost(
                389_283_840,
                &DeviceKey::cpu(),
                &DeviceKey::cuda(0),
                HostMemoryKind::Pinned,
            )
            .expect("h2d cost");
        assert!(up.time > Duration::ZERO);

        // Round-trip through JSON.
        let json = model.to_json().unwrap();
        let reloaded = PlacementCostModel::from_json(&json).unwrap();
        assert_eq!(model, reloaded);

        let cpu_cost2 = reloaded.op_cost(&structure, &DeviceKey::cpu(), 20).unwrap();
        assert_eq!(cpu_cost, cpu_cost2);

        // What was never measured stays None after a round-trip.
        assert!(
            reloaded
                .transfer_cost(
                    1024,
                    &DeviceKey::cpu(),
                    &DeviceKey::cuda(0),
                    HostMemoryKind::Pageable
                )
                .is_some() // pageable H2D *was* measured above
        );
        assert!(
            reloaded
                .transfer_cost(
                    1024,
                    &DeviceKey::cuda(0),
                    &DeviceKey::cpu(),
                    HostMemoryKind::Pinned
                )
                .is_none() // D2H was never measured
        );
    }
}
