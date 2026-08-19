//! Loading an nxmem plugin, negotiating the contract, and pinning the module.
//!
//! # Pinning
//!
//! Everything reachable from a loaded plugin — factories, allocators, queued
//! releases — holds an [`Arc<PluginModule>`]. [`PluginModule`] declares its
//! [`libloading::Library`] **last**, so the library is the final field to drop.
//! Code pages therefore stay mapped until every plugin object built on top of
//! them is gone.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use onnx_runtime_memory_abi::{
    NXMEM_CAP_ALLOCATOR, NXMEM_SYMBOL_CREATE_ALLOCATOR_FACTORIES, NXMEM_SYMBOL_NEGOTIATE,
    NXMEM_SYMBOL_QUERY_UNLOAD_READINESS, NxmemAllocatorFactoryVtable,
    NxmemCreateAllocatorFactoriesFn, NxmemNegotiateFn, NxmemNegotiateRequest,
    NxmemNegotiateResponse, NxmemQueryUnloadReadinessFn, NxmemUnloadReport, NxmemVersionRange,
    validate_negotiation,
};
use onnx_runtime_memory_api::DeviceKey;

use crate::allocator::{HostReclaim, PluginAllocator, open_allocator};
use crate::error::PluginError;

/// The most factories a single plugin may publish.
///
/// A fixed ceiling means the host never allocates on behalf of a plugin's
/// unvalidated count, and a runaway plugin cannot exhaust host memory during
/// enumeration.
pub const MAX_FACTORIES: usize = 64;

/// What the host and plugin agreed on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegotiatedAbi {
    /// The agreed major version. Calls are only legal within one major.
    pub major: u32,
    /// The agreed minor version. This is the **ceiling**: no vtable may claim
    /// a higher level, and every vtable is read at its own declared level.
    pub minor: u32,
    /// The capability bits the plugin claims and the host offered.
    pub capability_flags: u64,
}

/// A loaded plugin module.
///
/// Field order is load-bearing: `library` is last so it unmaps after every
/// other field has been dropped. Rust drops struct fields in declaration
/// order, so any field that might still call into plugin code during its own
/// drop must be declared before `library`.
#[derive(Debug)]
pub struct PluginModule {
    path: PathBuf,
    negotiated: NegotiatedAbi,
    query_unload: NxmemQueryUnloadReadinessFn,
    /// Allocators the host has opened and not yet dropped.
    live_allocators: AtomicU64,
    /// Allocations the host believes are live.
    live_allocations: AtomicU64,
    /// Deferred releases the host has queued and not seen retire.
    queued_releases: AtomicU64,
    /// Declared last: the library must outlive everything above it.
    library: libloading::Library,
}

impl PluginModule {
    /// Where the module was loaded from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The agreed contract.
    pub fn negotiated(&self) -> NegotiatedAbi {
        self.negotiated
    }

    /// The library handle, for tests that want to reach a raw symbol.
    pub fn library(&self) -> &libloading::Library {
        &self.library
    }

    pub(crate) fn allocator_opened(&self) {
        self.live_allocators.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn allocator_closed(&self) {
        self.live_allocators.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn allocation_opened(&self) {
        self.live_allocations.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn allocation_closed(&self) {
        self.live_allocations.fetch_sub(1, Ordering::AcqRel);
    }

    pub(crate) fn release_queued(&self) {
        self.queued_releases.fetch_add(1, Ordering::AcqRel);
    }

    pub(crate) fn release_retired(&self) {
        self.queued_releases.fetch_sub(1, Ordering::AcqRel);
    }

    /// The host's own view of what is still live.
    ///
    /// This is deliberately independent of the plugin's report: unload is
    /// gated on both, so a bug on one side cannot unmap code the other side is
    /// about to enter.
    pub fn host_live_counts(&self) -> HostLiveCounts {
        HostLiveCounts {
            allocators: self.live_allocators.load(Ordering::Acquire),
            allocations: self.live_allocations.load(Ordering::Acquire),
            queued_releases: self.queued_releases.load(Ordering::Acquire),
        }
    }

    /// Ask the plugin what it still owns.
    pub fn unload_report(&self) -> Result<NxmemUnloadReport, PluginError> {
        let mut report = NxmemUnloadReport::zeroed();
        // SAFETY: `report` is a valid, writable, correctly aligned local. The
        // symbol was resolved from this still-mapped library.
        let status = unsafe { (self.query_unload)(&raw mut report) };
        if !status.is_ok() {
            return Err(PluginError::call("NxmemQueryUnloadReadiness", status));
        }
        Ok(report)
    }
}

/// The host's own tally of live plugin-backed objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct HostLiveCounts {
    pub allocators: u64,
    pub allocations: u64,
    pub queued_releases: u64,
}

impl HostLiveCounts {
    /// Total live objects tracked by the host.
    pub const fn total(&self) -> u64 {
        self.allocators
            .saturating_add(self.allocations)
            .saturating_add(self.queued_releases)
    }
}

/// One mechanism a plugin publishes.
///
/// # Ownership
///
/// The host owns the factory and calls its `release` slot exactly once, from
/// [`Drop`]. Per the contract, releasing a factory must not invalidate
/// allocators already opened from it, so an allocator outliving its factory is
/// legal and is exercised by the test suite.
#[derive(Debug)]
pub struct PluginFactory {
    name: String,
    device: DeviceKey,
    capability_flags: u64,
    vtable: NxmemAllocatorFactoryVtable,
    module: Arc<PluginModule>,
}

// SAFETY: the vtable is a plain-data copy of the plugin's function pointers
// and an opaque `ctx`. The nxmem contract requires every slot to be callable
// from any host thread and requires the plugin to synchronise its own state.
unsafe impl Send for PluginFactory {}
// SAFETY: as above; `&PluginFactory` only ever reads the copied vtable.
unsafe impl Sync for PluginFactory {}

impl PluginFactory {
    /// The mechanism name, as published by the plugin.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The device this mechanism serves.
    pub fn device(&self) -> DeviceKey {
        self.device
    }

    /// The capabilities this mechanism claims.
    pub fn capability_flags(&self) -> u64 {
        self.capability_flags
    }

    /// The module this factory came from.
    pub fn module(&self) -> &Arc<PluginModule> {
        &self.module
    }

    /// Open an allocator from this mechanism.
    ///
    /// `reclaim` is the host's reclaim hook. It may be invoked reentrantly
    /// from inside a plugin call and from plugin-owned threads, so it must not
    /// block indefinitely and must not be called with any governance lock
    /// held.
    pub fn open(
        &self,
        required_capability_flags: u64,
        reclaim: Option<Arc<dyn HostReclaim>>,
    ) -> Result<PluginAllocator, PluginError> {
        open_allocator(self, required_capability_flags, reclaim)
    }

    pub(crate) fn vtable(&self) -> &NxmemAllocatorFactoryVtable {
        &self.vtable
    }
}

impl Drop for PluginFactory {
    fn drop(&mut self) {
        if let Some(release) = self.vtable.release {
            // SAFETY: `ctx` came from this factory's vtable and `release` is
            // called exactly once, here. No host lock is held.
            unsafe { release(self.vtable.ctx) };
        }
    }
}

/// A loaded plugin and the mechanisms it publishes.
#[derive(Debug)]
pub struct MemoryPlugin {
    /// Declared before `module` so factories release before the module (and
    /// therefore the library) can possibly go away.
    factories: Vec<PluginFactory>,
    module: Arc<PluginModule>,
}

impl MemoryPlugin {
    /// Load a plugin using this host's current supported version range.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, PluginError> {
        Self::load_with_host_range(path, NxmemVersionRange::current())
    }

    /// Load a plugin while pretending the host only supports `range`.
    ///
    /// This is the honest way to test both negotiation directions: an older
    /// host meeting a newer plugin, and a version mismatch, both driven
    /// through the real plugin entry point and the real loader rather than a
    /// parallel re-implementation.
    pub fn load_with_host_range(
        path: impl AsRef<Path>,
        range: NxmemVersionRange,
    ) -> Result<Self, PluginError> {
        let path = path.as_ref().to_path_buf();
        let display = path.display().to_string();

        // SAFETY: loading a dynamic library runs its initialisers. The caller
        // is trusted to name a library it intends to load, exactly as for any
        // other plugin surface in this workspace.
        let library =
            unsafe { libloading::Library::new(&path) }.map_err(|source| PluginError::Open {
                path: display.clone(),
                source,
            })?;

        let negotiate = resolve::<NxmemNegotiateFn>(&library, NXMEM_SYMBOL_NEGOTIATE, &display)?;
        let create_factories = resolve::<NxmemCreateAllocatorFactoriesFn>(
            &library,
            NXMEM_SYMBOL_CREATE_ALLOCATOR_FACTORIES,
            &display,
        )?;
        let query_unload = resolve::<NxmemQueryUnloadReadinessFn>(
            &library,
            NXMEM_SYMBOL_QUERY_UNLOAD_READINESS,
            &display,
        )?;

        let request = NxmemNegotiateRequest::with_range(range);
        let mut response = NxmemNegotiateResponse::zeroed();
        // SAFETY: both pointers address valid, aligned, writable locals that
        // outlive the call. The plugin may not retain either.
        let status = unsafe { negotiate(&raw const request, &raw mut response) };
        if !status.is_ok() {
            return Err(PluginError::Negotiation {
                path: display,
                reason: status.describe(),
            });
        }
        validate_negotiation(&range, &response).map_err(|rejection| PluginError::Negotiation {
            path: display.clone(),
            reason: rejection.reason,
        })?;

        if response.capability_flags & NXMEM_CAP_ALLOCATOR == 0 {
            return Err(PluginError::Negotiation {
                path: display,
                reason: String::from(
                    "the plugin does not claim the allocator capability; a memory plugin that \
                     cannot allocate has nothing to contribute",
                ),
            });
        }

        let negotiated = NegotiatedAbi {
            major: response.agreed_major,
            minor: response.agreed_minor,
            capability_flags: response.capability_flags,
        };

        let module = Arc::new(PluginModule {
            path,
            negotiated,
            query_unload,
            live_allocators: AtomicU64::new(0),
            live_allocations: AtomicU64::new(0),
            queued_releases: AtomicU64::new(0),
            library,
        });

        let factories = enumerate_factories(&module, create_factories, &display)?;
        Ok(Self { factories, module })
    }

    /// The agreed contract.
    pub fn negotiated(&self) -> NegotiatedAbi {
        self.module.negotiated
    }

    /// The loaded module.
    pub fn module(&self) -> &Arc<PluginModule> {
        &self.module
    }

    /// Every mechanism the plugin published.
    pub fn factories(&self) -> &[PluginFactory] {
        &self.factories
    }

    /// Find a mechanism by name.
    pub fn factory(&self, name: &str) -> Result<&PluginFactory, PluginError> {
        self.factories
            .iter()
            .find(|factory| factory.name == name)
            .ok_or_else(|| PluginError::UnknownMechanism {
                path: self.module.path.display().to_string(),
                name: name.to_string(),
                available: self
                    .factories
                    .iter()
                    .map(|factory| factory.name.as_str())
                    .collect::<Vec<_>>()
                    .join(", "),
            })
    }

    /// Attempt to unload the plugin.
    ///
    /// Unload is refused while **either** side still reports live objects. On
    /// refusal the plugin is handed back, so a caller can retire the
    /// outstanding work and try again — this is a deferral, not a leak.
    pub fn try_unload(self) -> Result<(), UnloadRejection> {
        let host = self.module.host_live_counts();
        // Ask the plugin first, unconditionally, so every rejection carries
        // both tallies. A refusal that reports only one side leaves the caller
        // guessing about which objects to retire.
        let report = match self.module.unload_report() {
            Ok(report) => report,
            Err(error) => {
                return Err(UnloadRejection {
                    reason: format!(
                        "the plugin could not report unload readiness, so unloading it would be \
                         a guess: {error}"
                    ),
                    report: NxmemUnloadReport::zeroed(),
                    host,
                    plugin: Some(self),
                });
            }
        };

        if host.total() != 0 {
            return Err(UnloadRejection {
                reason: format!(
                    "the host still holds {} allocator(s), {} allocation(s) and {} queued \
                     release(s); retire them before unloading",
                    host.allocators, host.allocations, host.queued_releases
                ),
                report,
                host,
                plugin: Some(self),
            });
        }

        if report.total() != 0 {
            return Err(UnloadRejection {
                reason: format!(
                    "the plugin still owns {} allocator(s), {} allocation(s), {} view(s), {} \
                     capability object(s) and {} queued release(s)",
                    report.live_allocators,
                    report.live_allocations,
                    report.live_views,
                    report.live_capabilities,
                    report.queued_releases
                ),
                report,
                host,
                plugin: Some(self),
            });
        }

        // Both sides agree nothing is live. Release the factories, then check
        // that nothing else still pins the module before letting it unmap.
        let Self {
            factories, module, ..
        } = self;
        drop(factories);

        match Arc::try_unwrap(module) {
            Ok(module) => {
                drop(module);
                Ok(())
            }
            Err(module) => {
                // Something outside this handle still pins the module. The
                // factories are gone, so the plugin cannot be handed back —
                // but the module stays mapped, which is the safe direction.
                let reason = format!(
                    "{} handle(s) outside this plugin still pin the module, so it stays mapped",
                    Arc::strong_count(&module)
                );
                Err(UnloadRejection {
                    reason,
                    report,
                    host,
                    plugin: None,
                })
            }
        }
    }
}

/// Why an unload was refused or deferred.
#[derive(Debug)]
pub struct UnloadRejection {
    /// A human-readable explanation naming what is still live.
    pub reason: String,
    /// The plugin's own report at the moment of refusal.
    pub report: NxmemUnloadReport,
    /// The host's own tally at the moment of refusal.
    pub host: HostLiveCounts,
    /// The plugin, handed back so the caller can retire work and retry.
    ///
    /// `None` only in the terminal case where the module is pinned by a handle
    /// this loader does not own; the module then simply stays mapped.
    pub plugin: Option<MemoryPlugin>,
}

impl UnloadRejection {
    /// Recover the plugin so the caller can retire outstanding work and retry.
    pub fn into_plugin(self) -> Option<MemoryPlugin> {
        self.plugin
    }
}

impl std::fmt::Display for UnloadRejection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "cannot unload the memory plugin: {}", self.reason)
    }
}

impl std::error::Error for UnloadRejection {}

fn resolve<T: Copy>(
    library: &libloading::Library,
    symbol: &[u8],
    path: &str,
) -> Result<T, PluginError> {
    // SAFETY: the symbol is looked up by name and immediately copied out of
    // the `Symbol` guard. The resulting function pointer stays valid because
    // `PluginModule` keeps the library mapped for as long as anything can call
    // it.
    let found = unsafe { library.get::<T>(symbol) }.map_err(|_| PluginError::MissingSymbol {
        path: path.to_string(),
        symbol: String::from_utf8_lossy(symbol).into_owned(),
    })?;
    Ok(*found)
}

fn enumerate_factories(
    module: &Arc<PluginModule>,
    create: NxmemCreateAllocatorFactoriesFn,
    path: &str,
) -> Result<Vec<PluginFactory>, PluginError> {
    let mut raw = [core::ptr::null::<NxmemAllocatorFactoryVtable>(); MAX_FACTORIES];
    let mut count: u64 = 0;
    // SAFETY: `raw` has exactly `MAX_FACTORIES` writable slots and `count` is a
    // valid writable local.
    let status = unsafe { create(raw.as_mut_ptr(), MAX_FACTORIES as u64, &raw mut count) };
    if !status.is_ok() {
        return Err(PluginError::call("NxmemCreateAllocatorFactories", status));
    }
    if count > MAX_FACTORIES as u64 {
        return Err(PluginError::Contract {
            path: path.to_string(),
            reason: format!(
                "the plugin reported {count} factories after being told it may write at most \
                 {MAX_FACTORIES}; the extra entries were never written and cannot be read"
            ),
        });
    }

    let minor = module.negotiated.minor;
    let mut factories = Vec::with_capacity(count as usize);
    for slot in raw.iter().take(count as usize) {
        // SAFETY: the plugin wrote this pointer in response to the call above.
        // `read_prefix` null-checks, alignment-checks, and size-checks it
        // before reading any field, and copies rather than borrowing.
        let vtable = unsafe { NxmemAllocatorFactoryVtable::read_prefix(*slot, minor) }
            .map_err(|status| PluginError::call("factory vtable", status))?;

        let device = device_key(vtable.device).ok_or_else(|| PluginError::Contract {
            path: path.to_string(),
            reason: format!(
                "a factory declared tier code {} which this host does not know; tiers are never \
                 guessed",
                vtable.device.tier
            ),
        })?;

        // SAFETY: the contract requires `name` to be a NUL-terminated UTF-8
        // string that stays valid until the factory's final `release`, which
        // this host has not called yet.
        let name = unsafe { read_c_string(vtable.name) }.ok_or_else(|| PluginError::Contract {
            path: path.to_string(),
            reason: String::from("a factory published a null or non-UTF-8 name"),
        })?;

        factories.push(PluginFactory {
            name,
            device,
            capability_flags: vtable.capability_flags,
            vtable,
            module: Arc::clone(module),
        });
    }
    Ok(factories)
}

/// Translate an ABI device id into the internal key, refusing unknown tiers.
pub(crate) fn device_key(device: onnx_runtime_memory_abi::NxmemDeviceId) -> Option<DeviceKey> {
    use onnx_runtime_memory_abi::{NXMEM_TIER_DEVICE, NXMEM_TIER_DISK, NXMEM_TIER_HOST};
    use onnx_runtime_memory_api::Tier;
    let tier = match device.tier {
        NXMEM_TIER_DEVICE => Tier::Device,
        NXMEM_TIER_HOST => Tier::Host,
        NXMEM_TIER_DISK => Tier::Disk,
        _ => return None,
    };
    Some(DeviceKey {
        tier,
        index: device.index,
    })
}

/// Translate an internal key into an ABI device id.
pub(crate) fn device_id(device: DeviceKey) -> onnx_runtime_memory_abi::NxmemDeviceId {
    use onnx_runtime_memory_abi::{
        NXMEM_TIER_DEVICE, NXMEM_TIER_DISK, NXMEM_TIER_HOST, NxmemDeviceId,
    };
    use onnx_runtime_memory_api::Tier;
    let tier = match device.tier {
        Tier::Device => NXMEM_TIER_DEVICE,
        Tier::Host => NXMEM_TIER_HOST,
        Tier::Disk => NXMEM_TIER_DISK,
    };
    NxmemDeviceId {
        tier,
        index: device.index,
    }
}

/// Read a NUL-terminated UTF-8 string published by a plugin.
///
/// # Safety
///
/// `ptr` must be null or point to a NUL-terminated byte string that stays
/// valid for the duration of the call.
unsafe fn read_c_string(ptr: *const u8) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: delegated to this function's contract.
    let c_str = unsafe { core::ffi::CStr::from_ptr(ptr.cast()) };
    c_str.to_str().ok().map(str::to_owned)
}

#[cfg(test)]
mod tests {
    use super::*;
    use onnx_runtime_memory_abi::NxmemDeviceId;
    use onnx_runtime_memory_api::Tier;

    #[test]
    fn device_ids_round_trip_through_both_directions() {
        for key in [
            DeviceKey::HOST,
            DeviceKey::device(0),
            DeviceKey::device(7),
            DeviceKey {
                tier: Tier::Disk,
                index: 3,
            },
        ] {
            assert_eq!(device_key(device_id(key)), Some(key));
        }
    }

    #[test]
    fn an_unknown_tier_code_is_refused_rather_than_guessed() {
        let unknown = NxmemDeviceId {
            tier: 4_242,
            index: 0,
        };
        assert_eq!(device_key(unknown), None);
    }

    #[test]
    fn host_live_counts_sum_every_axis() {
        let counts = HostLiveCounts {
            allocators: 1,
            allocations: 2,
            queued_releases: 3,
        };
        assert_eq!(counts.total(), 6);
        assert_eq!(HostLiveCounts::default().total(), 0);
    }

    #[test]
    fn a_null_name_is_refused() {
        // SAFETY: a null pointer is the documented "no name" case.
        assert_eq!(unsafe { read_c_string(core::ptr::null()) }, None);
    }

    #[test]
    fn a_valid_name_is_read_back() {
        let raw = c"eager";
        // SAFETY: `raw` is a NUL-terminated literal alive for the whole call.
        let name = unsafe { read_c_string(raw.as_ptr().cast()) };
        assert_eq!(name.as_deref(), Some("eager"));
    }
}
