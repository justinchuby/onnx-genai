//! Test helpers for constructing deliberately broken nxrt plugins.
//!
//! These are intended for negative-testing (Pris's fixture variants):
//! wrong major version, unknown capability bits, factory errors, zero devices,
//! and panicking plugins. Consumers use these to build custom `NxrtNegotiate`
//! and `NxrtCreateEpFactories` implementations without duplicating the ABI.
//!
//! # Usage
//!
//! ```rust,ignore
//! use onnx_runtime_ep_nxrt_abi::testing::NxrtNegotiateOverride;
//! use onnx_runtime_ep_nxrt_abi::export_nxrt_ep_negotiate_custom;
//!
//! // Report wrong major version:
//! export_nxrt_ep_negotiate_custom!(NxrtNegotiateOverride::wrong_major(99));
//! ```

use crate::status::{NxrtStatus, NxrtStatusCode};
use crate::version::{
    NXRT_ABI_VERSION_MAJOR, NXRT_ABI_VERSION_MINOR, NxrtNegotiateRequest, NxrtNegotiateResponse,
    NxrtVersionRange,
};

/// Configuration for a custom `NxrtNegotiate` response.
///
/// Allows constructing deliberately broken or unusual negotiation responses
/// for negative testing.
#[derive(Debug, Clone)]
pub struct NxrtNegotiateOverride {
    /// Major version to report in the response.
    pub major: u32,
    /// Minor version to report in the response.
    pub minor: u32,
    /// Capability flags to report.
    pub capability_flags: u64,
    /// If `Some`, force this status code instead of normal logic.
    pub force_status: Option<NxrtStatusCode>,
    /// If true, panic inside negotiate (for panic-containment testing).
    pub panic: bool,
}

impl NxrtNegotiateOverride {
    /// Normal negotiation (equivalent to the real implementation).
    pub const fn normal() -> Self {
        Self {
            major: NXRT_ABI_VERSION_MAJOR,
            minor: NXRT_ABI_VERSION_MINOR,
            capability_flags: crate::version::NXRT_CAP_DEVICE_ENUMERATION,
            force_status: None,
            panic: false,
        }
    }

    /// Reports a wrong major version so the host rejects.
    pub const fn wrong_major(major: u32) -> Self {
        Self {
            major,
            minor: 0,
            capability_flags: 0,
            force_status: None,
            panic: false,
        }
    }

    /// Reports unknown capability bits (host must reject, fail closed).
    pub const fn unknown_caps(flags: u64) -> Self {
        Self {
            major: NXRT_ABI_VERSION_MAJOR,
            minor: NXRT_ABI_VERSION_MINOR,
            capability_flags: flags,
            force_status: None,
            panic: false,
        }
    }

    /// The negotiate function will panic (for containment testing).
    pub const fn panicking() -> Self {
        Self {
            major: 0,
            minor: 0,
            capability_flags: 0,
            force_status: None,
            panic: true,
        }
    }

    /// Execute this override, writing the response. This is the logic that
    /// a custom `NxrtNegotiate` symbol implementation should call.
    ///
    /// # Safety
    ///
    /// Both pointers must be valid and non-null.
    pub unsafe fn execute(
        &self,
        _request: *const NxrtNegotiateRequest,
        response_out: *mut NxrtNegotiateResponse,
    ) -> NxrtStatus {
        if self.panic {
            panic!("NxrtNegotiate: deliberate panic for testing");
        }
        if _request.is_null() || response_out.is_null() {
            return NxrtStatus::from_code_with_message(
                NxrtStatusCode::InvalidArgument,
                "null pointer in negotiate",
            );
        }
        let resp = unsafe { &mut *response_out };
        resp.struct_size = std::mem::size_of::<NxrtNegotiateResponse>() as u32;
        resp.agreed_major = self.major;
        resp.agreed_minor = self.minor;
        resp.plugin_range = NxrtVersionRange {
            major_min: self.major,
            major_max: self.major,
            minor_max: self.minor,
        };
        resp.capability_flags = self.capability_flags;

        if let Some(code) = self.force_status {
            return NxrtStatus::from_code(code);
        }
        NxrtStatus::ok()
    }
}

/// Configuration for a custom `NxrtCreateEpFactories` behavior.
#[derive(Debug, Clone)]
pub struct NxrtCreateFactoriesOverride {
    /// If `Some`, return this error status immediately.
    pub force_error: Option<NxrtStatusCode>,
    /// If true, report zero factories (num_devices = 0 scenario).
    pub zero_factories: bool,
    /// If true, panic inside create (for containment testing).
    pub panic: bool,
}

impl NxrtCreateFactoriesOverride {
    /// A factory that immediately returns an error.
    pub const fn error(code: NxrtStatusCode) -> Self {
        Self {
            force_error: Some(code),
            zero_factories: false,
            panic: false,
        }
    }

    /// A factory that reports zero factories.
    pub const fn zero() -> Self {
        Self {
            force_error: None,
            zero_factories: true,
            panic: false,
        }
    }

    /// A factory that panics.
    pub const fn panicking() -> Self {
        Self {
            force_error: None,
            zero_factories: false,
            panic: true,
        }
    }

    /// Execute this override.
    ///
    /// # Safety
    ///
    /// Pointers must be valid per ABI contract.
    pub unsafe fn execute(
        &self,
        _out_factories: *mut *mut crate::NxrtEpFactoryVtable,
        _max_factories: usize,
        out_num: *mut usize,
    ) -> NxrtStatus {
        if self.panic {
            panic!("NxrtCreateEpFactories: deliberate panic for testing");
        }
        if let Some(code) = self.force_error {
            if !out_num.is_null() {
                unsafe { *out_num = 0 };
            }
            return NxrtStatus::from_code_with_message(code, "deliberate error for testing");
        }
        if self.zero_factories {
            if !out_num.is_null() {
                unsafe { *out_num = 0 };
            }
            return NxrtStatus::ok();
        }
        // Unreachable in override-only mode; real factories use the macro.
        if !out_num.is_null() {
            unsafe { *out_num = 0 };
        }
        NxrtStatus::ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiate_override_wrong_major() {
        let over = NxrtNegotiateOverride::wrong_major(99);
        let req = NxrtNegotiateRequest::current();
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { over.execute(&req, &mut resp) };
        assert!(status.is_ok());
        assert_eq!(resp.agreed_major, 99);
    }

    #[test]
    fn negotiate_override_unknown_caps() {
        let over = NxrtNegotiateOverride::unknown_caps(1 << 63 | 1 << 62);
        let req = NxrtNegotiateRequest::current();
        let mut resp = NxrtNegotiateResponse::zeroed();
        let status = unsafe { over.execute(&req, &mut resp) };
        assert!(status.is_ok());
        // Host should reject these flags via validate_negotiation
        let result = crate::version::validate_negotiation(&NxrtVersionRange::current(), &resp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unknown capability flags"));
    }

    #[test]
    fn negotiate_override_panicking() {
        let over = NxrtNegotiateOverride::panicking();
        let req = NxrtNegotiateRequest::current();
        let mut resp = NxrtNegotiateResponse::zeroed();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            over.execute(&req, &mut resp)
        }));
        assert!(result.is_err(), "must panic");
    }

    #[test]
    fn create_override_error() {
        let over = NxrtCreateFactoriesOverride::error(NxrtStatusCode::DeviceError);
        let mut num: usize = 99;
        let status = unsafe { over.execute(std::ptr::null_mut(), 0, &mut num) };
        assert_eq!(status.status_code(), Some(NxrtStatusCode::DeviceError));
        assert_eq!(num, 0);
    }

    #[test]
    fn create_override_zero_factories() {
        let over = NxrtCreateFactoriesOverride::zero();
        let mut num: usize = 99;
        let status = unsafe { over.execute(std::ptr::null_mut(), 0, &mut num) };
        assert!(status.is_ok());
        assert_eq!(num, 0);
    }

    #[test]
    fn create_override_panicking() {
        let over = NxrtCreateFactoriesOverride::panicking();
        let mut num: usize = 0;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
            over.execute(std::ptr::null_mut(), 0, &mut num)
        }));
        assert!(result.is_err());
    }
}
