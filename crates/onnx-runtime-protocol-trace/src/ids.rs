//! Stable identity newtypes shared across protocol families.
//!
//! Conformance traces must identify entities by stable identities, never by
//! pointers or vector positions (`specs/tla/REFINEMENT.md` § "Required Trace
//! Envelope"). These newtypes are deliberately small, `Copy`, and totally
//! ordered so they can key ledgers and be compared in an independent checker.

macro_rules! id_newtype {
    ($(#[$meta:meta])* $name:ident, $inner:ty) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
        pub struct $name($inner);

        impl $name {
            /// Wraps a raw value.
            #[inline]
            pub const fn new(value: $inner) -> Self {
                Self(value)
            }

            /// Returns the raw value.
            #[inline]
            pub const fn get(self) -> $inner {
                self.0
            }
        }

        impl From<$inner> for $name {
            #[inline]
            fn from(value: $inner) -> Self {
                Self(value)
            }
        }

        impl core::fmt::Display for $name {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                write!(f, "{}({})", stringify!($name), self.0)
            }
        }
    };
}

id_newtype!(
    /// Identifies the producer of a trace event stream. `source_sequence` is
    /// monotonic within a single `ProtocolSourceId`.
    ProtocolSourceId,
    u64
);

id_newtype!(
    /// Stable identity of a single pressure request/ticket. Never reused within
    /// a process; queue index or address are *not* stable identities.
    PressureRequestId,
    u64
);

id_newtype!(
    /// Identifies the HostGovernor configuration generation (not an individual
    /// request). Reconfiguration increments it.
    PressureGeneration,
    u64
);

id_newtype!(
    /// Process-unique, never-reused physical host allocation identity. Because
    /// it is never reused it discharges the ABA-prevention obligation without a
    /// separate exposed generation counter (`REFINEMENT.md` § "Buffer
    /// ownership").
    PhysicalAllocationId,
    u64
);

id_newtype!(
    /// A machine-local device (GPU) owning a host-memory charge.
    LocalDeviceId,
    u32
);

impl PressureGeneration {
    /// Returns the next generation.
    #[inline]
    pub const fn next(self) -> Self {
        Self(self.0 + 1)
    }
}
