//! # `onnx-runtime-memory-api`
//!
//! Dependency-free primitive types shared by memory mechanisms, governors, and
//! execution providers.
//!
//! This crate is the lowest layer of the runtime memory stack. It owns only
//! types that describe placement, allocation ranges, mapped-byte outcomes, and
//! opaque shared-prefix backing. It does not own allocation policy, accounting,
//! synchronization, or runtime selection.
//!
//! ## Types extracted in Phase 1
//!
//! * [`Tier`] and [`DeviceKey`] describe memory placement.
//! * [`AllocationCommitRange`] and [`MappedAllocation`] describe existing
//!   allocation and commit results.
//! * [`SharedDevicePrefix`] and [`SharedPrefixCommitInfo`] describe the existing
//!   opaque shared-prefix mechanism.
//!
//! ## Types that remain in `onnx-runtime-memory-governor`
//!
//! * `DeviceAllocator` and `HostAllocator`, because the current allocator trait
//!   accepts governor-owned capacity tokens and errors.
//! * `MemoryRole`, `MemoryError`, authority and holder identities, ledgers,
//!   capacity tokens and grants, leases, pressure responders, and governor
//!   traits, because they express accounting or governance.
//! * The large-allocation cache and prefix-shareability analysis, because they
//!   are built over the current governor-owned allocator and admission model.
//!
//! Existing `onnx-runtime-memory-governor` paths re-export the moved types for
//! source compatibility.

pub mod allocator;

pub use allocator::{
    AllocationCommitRange, DeviceKey, MappedAllocation, SharedDevicePrefix, SharedPrefixCommitInfo,
};

/// Where the bytes physically live.
///
/// Ordered from fastest to slowest, which is also the demotion order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Tier {
    /// Accelerator memory (VRAM).
    Device,
    /// Host RAM.
    Host,
    /// Spill file on disk.
    Disk,
}

impl Tier {
    /// Every tier, fastest first.
    pub const ALL: [Tier; 3] = [Tier::Device, Tier::Host, Tier::Disk];

    /// Human-facing name used in error messages.
    pub const fn name(self) -> &'static str {
        match self {
            Tier::Device => "device",
            Tier::Host => "host",
            Tier::Disk => "disk",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_order_and_names_are_stable() {
        assert_eq!(Tier::ALL, [Tier::Device, Tier::Host, Tier::Disk]);
        assert_eq!(Tier::Device.name(), "device");
        assert_eq!(Tier::Host.name(), "host");
        assert_eq!(Tier::Disk.name(), "disk");
    }
}
