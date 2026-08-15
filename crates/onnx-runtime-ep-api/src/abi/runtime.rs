use std::collections::HashMap;
use std::ffi::{CStr, c_char, c_void};
use std::path::Path;
use std::ptr;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};

use onnx_genai_ort_sys as ort;

use super::ffi_helpers::MAX_PLUGIN_THREAD_STATES;
use super::host::{
    HostKernelContext, check_compute_status, check_status, ort_api_base, release_status,
};
use crate::error::{EpError, Result};
use crate::kernel::{ARG_BYTES, ARG_DEVICE, ARG_FLOPS, CAT_KERNEL_WORKER, Kernel};
use crate::tensor::{TensorMut, TensorView};

pub(super) struct PluginRuntime {
    pub(super) path: std::path::PathBuf,
    // Kept as a lifetime anchor for every function pointer and plugin object released in Drop.
    #[allow(dead_code)]
    pub(super) lib: libloading::Library,
    pub(super) factory: *mut ort::OrtEpFactory,
    pub(super) ep: *mut ort::OrtEp,
    pub(super) release_factory:
        Option<unsafe extern "C" fn(*mut ort::OrtEpFactory) -> *mut ort::OrtStatus>,
    pub(super) compute_infos: Vec<*mut ort::OrtNodeComputeInfo>,
}

unsafe impl Send for PluginRuntime {}
unsafe impl Sync for PluginRuntime {}

impl PluginRuntime {
    pub(super) fn load(library_path: &Path, registration_name: Option<&CStr>) -> Result<Self> {
        // SAFETY: The caller explicitly selected this ORT plugin library. The
        // handle is stored in PluginRuntime and outlives every resolved symbol.
        let lib = unsafe { libloading::Library::new(library_path) }.map_err(|err| {
            EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: format!(
                    "failed to open plugin dynamic library ({err}); fix by building the plugin dylib and passing the correct absolute path"
                ),
            }
        })?;
        type CreateEpFactories = unsafe extern "C" fn(
            *const c_char,
            *const ort::OrtApiBase,
            *const ort::OrtLogger,
            *mut *mut ort::OrtEpFactory,
            usize,
            *mut usize,
        ) -> *mut ort::OrtStatus;
        type ReleaseEpFactory = unsafe extern "C" fn(*mut ort::OrtEpFactory) -> *mut ort::OrtStatus;
        // SAFETY: Symbol types match ONNX Runtime's plugin EP C ABI.
        let create = unsafe { lib.get::<CreateEpFactories>(b"CreateEpFactories") }.map_err(|err| {
            EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: format!(
                    "CreateEpFactories symbol was not found ({err}); fix by using an ONNX Runtime plugin-EP library built against the plugin EP C ABI"
                ),
            }
        })?;
        // SAFETY: Optional release symbol from the same plugin ABI.
        let release_factory = unsafe {
            lib.get::<ReleaseEpFactory>(b"ReleaseEpFactory")
                .ok()
                .map(|symbol| *symbol)
        };
        let mut factories: [*mut ort::OrtEpFactory; 1] = [ptr::null_mut()];
        let mut num_factories = 0usize;
        let name_ptr = registration_name.map_or(ptr::null(), CStr::as_ptr);
        // SAFETY: Output pointers reference live stack storage and API base is static.
        let status = unsafe {
            create(
                name_ptr,
                ort_api_base(),
                ptr::null(),
                factories.as_mut_ptr(),
                factories.len(),
                &mut num_factories,
            )
        };
        check_status(library_path, "CreateEpFactories", status)?;
        if num_factories == 0 || factories[0].is_null() {
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "CreateEpFactories returned no factories; fix by checking that the plugin supports this platform and ORT API version".into(),
            });
        }
        let factory = factories[0];
        let supported_version = unsafe { (*factory).ort_version_supported };
        if supported_version == 0 || supported_version > ort::ORT_API_VERSION {
            if let Some(release_factory) = release_factory {
                // SAFETY: The factory was returned by CreateEpFactories and the
                // optional release callback was resolved from that same library.
                let status = unsafe { release_factory(factory) };
                release_status(status);
            }
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: format!(
                    "plugin factory requires ORT API version {supported_version}, but this host supports version {}; fix by using a plugin built for a compatible ORT plugin-EP ABI",
                    ort::ORT_API_VERSION
                ),
            });
        }
        let mut ep: *mut ort::OrtEp = ptr::null_mut();
        // SAFETY: The factory pointer came from the plugin and CreateEp writes
        // the out-pointer before returning a null status.
        let status = unsafe {
            let create_ep = (*factory).CreateEp.ok_or_else(|| EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "OrtEpFactory.CreateEp is null; fix by using a complete plugin EP factory"
                    .into(),
            })?;
            create_ep(
                factory,
                ptr::null(),
                ptr::null(),
                1,
                ptr::null(),
                ptr::null(),
                &mut ep,
            )
        };
        check_status(library_path, "OrtEpFactory.CreateEp", status)?;
        if ep.is_null() {
            return Err(EpError::EpLoadFailed {
                path: library_path.to_path_buf(),
                reason: "OrtEpFactory.CreateEp returned a null EP; fix by checking plugin device requirements and options".into(),
            });
        }
        Ok(Self {
            path: library_path.to_path_buf(),
            lib,
            factory,
            ep,
            release_factory,
            compute_infos: Vec::new(),
        })
    }
}

impl Drop for PluginRuntime {
    fn drop(&mut self) {
        // SAFETY: Release callbacks belong to the plugin objects this runtime
        // owns, and the dynamic library is still loaded while they run.
        unsafe {
            if !self.compute_infos.is_empty()
                && let Some(release_infos) = (*self.ep).ReleaseNodeComputeInfos
            {
                release_infos(
                    self.ep,
                    self.compute_infos.as_mut_ptr(),
                    self.compute_infos.len(),
                );
            }
            if let Some(release_ep) = (*self.factory).ReleaseEp {
                release_ep(self.factory, self.ep);
            }
            if let Some(release_factory) = &self.release_factory {
                let st = release_factory(self.factory);
                if !st.is_null() {
                    release_status(st);
                }
            }
        }
    }
}

pub(super) struct PluginKernelShared {
    pub(super) runtime: Arc<PluginRuntime>,
    pub(super) info: *mut ort::OrtNodeComputeInfo,
    pub(super) create_state: unsafe extern "C" fn(
        *mut ort::OrtNodeComputeInfo,
        *mut ort::OrtNodeComputeContext,
        *mut *mut c_void,
    ) -> *mut ort::OrtStatus,
    pub(super) compute: unsafe extern "C" fn(
        *mut ort::OrtNodeComputeInfo,
        *mut c_void,
        *mut ort::OrtKernelContext,
    ) -> *mut ort::OrtStatus,
    pub(super) release_state:
        Option<unsafe extern "C" fn(*mut ort::OrtNodeComputeInfo, *mut c_void)>,
    /// Plugin state, one per thread that has executed this kernel.
    ///
    /// Per-thread because a plugin may be thread-affine -- the MLX one is, and
    /// reusing its state from a second thread fails on the next token. The
    /// entries are bounded by [`MAX_PLUGIN_THREAD_STATES`]: the decode path
    /// uses a small fixed set of threads, so exceeding it means threads are
    /// being created per call, and quietly accumulating plugin state for each
    /// would be a leak that only shows up as memory growth much later.
    pub(super) states: Mutex<HashMap<std::thread::ThreadId, *mut c_void>>,
    pub(super) index: usize,
    pub(super) calls: AtomicU64,
    pub(super) device_label: Arc<str>,
}

unsafe impl Send for PluginKernelShared {}
unsafe impl Sync for PluginKernelShared {}

impl Drop for PluginKernelShared {
    /// Release every per-thread state.
    ///
    /// Known limitation: these are released on whichever thread drops the
    /// session, not on the thread that created each one. For a plugin that is
    /// merely thread-affine for *execution* that is fine, and it is what the
    /// one plugin exercised here does; for a plugin whose teardown is also
    /// thread-affine it is not, and the fix is to own the executing threads so
    /// each state can be released on its own. Left as is because the executor
    /// does not own those threads today, and leaking state instead would be a
    /// worse trade.
    fn drop(&mut self) {
        if let Some(release_state) = self.release_state
            && let Ok(states) = self.states.get_mut()
        {
            for (_, state) in states.drain() {
                // SAFETY: each state was returned by this compute-info's
                // CreateState and is released before PluginRuntime releases
                // the compute infos.
                unsafe { release_state(self.info, state) };
            }
        }
    }
}

/// A native-runtime kernel backed by an ORT plugin `OrtNodeComputeInfo`.
pub struct PluginCompiledKernel {
    pub(super) shared: Arc<PluginKernelShared>,
}

impl Kernel for PluginCompiledKernel {
    fn execute(&self, inputs: &[TensorView], outputs: &mut [TensorMut]) -> Result<()> {
        let mut states = self.shared.states.lock().map_err(|_| {
            EpError::KernelFailed(
                "plugin fused-subgraph state mutex was poisoned; recreate the session".into(),
            )
        })?;
        let thread_id = std::thread::current().id();
        let state = if let Some(&state) = states.get(&thread_id) {
            state
        } else {
            // Bounded, because each entry holds plugin-side resources until the
            // session ends. Decode runs on a small fixed set of threads, so
            // passing this means threads are being created per call and the map
            // would grow without limit -- better to say so than to leak.
            if states.len() >= MAX_PLUGIN_THREAD_STATES {
                return Err(EpError::KernelFailed(format!(
                    "the execution provider plugin has been asked to run fused subgraph {} from \
                     more than {MAX_PLUGIN_THREAD_STATES} threads, and it holds per-thread state \
                     for each. Why: plugin state is created per executing thread because plugins \
                     may be thread-affine, so a caller creating a fresh thread per call would \
                     grow it without bound. Fix by running generation from a bounded thread pool, \
                     or set ONNX_GENAI_BACKEND=ort to run without the plugin",
                    self.shared.index
                )));
            }
            let mut state: *mut c_void = ptr::null_mut();
            // SAFETY: `info` was allocated by the plugin and remains owned by
            // `shared.runtime`; the native bridge does not expose a compute
            // context to CreateState. Creating the state on the executing thread
            // preserves plugins whose sessions are thread-affine.
            let status = unsafe {
                (self.shared.create_state)(self.shared.info, ptr::null_mut(), &mut state)
            };
            check_status(
                &self.shared.runtime.path,
                "OrtNodeComputeInfo.CreateState",
                status,
            )?;
            states.insert(thread_id, state);
            state
        };
        // Released before calling into the plugin. The map only guards *state
        // lookup*; holding it across Compute would serialise every fused
        // subgraph in the process behind one lock, which is the opposite of
        // why the state is per-thread in the first place.
        drop(states);
        let mut context = HostKernelContext::new(inputs, outputs)?;
        let bytes = context.byte_size();
        let _span = onnx_runtime_tracer::global_context().map(|trace| {
            let span = trace
                .span(
                    format!("plugin_fused_{}", self.shared.index),
                    CAT_KERNEL_WORKER,
                )
                .without_source();
            onnx_runtime_tracer::annotate_current_span_with(|| {
                onnx_runtime_tracer::Args::new()
                    .with(ARG_DEVICE, self.shared.device_label.to_string())
                    .with(ARG_BYTES, bytes as u64)
                    .with(ARG_FLOPS, 0_u64)
            });
            span
        });
        // SAFETY: The kernel context is a stack-owned HostKernelContext whose
        // OrtValue pointers borrow the input/output TensorViews for this call.
        // The plugin must not retain them after Compute returns (ORT contract).
        let status = unsafe {
            (self.shared.compute)(
                self.shared.info,
                state,
                (&mut context as *mut HostKernelContext).cast::<ort::OrtKernelContext>(),
            )
        };
        // Not `check_status`: that reports a load failure and tells the reader
        // to check plugin compatibility and ORT_API_VERSION, which is exactly
        // wrong once the plugin has already loaded, compiled and run. A
        // failure here is about this call.
        check_compute_status(&self.shared.runtime.path, self.shared.index, status)?;
        self.shared.calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn supports_strided_input(&self, _input_idx: usize) -> bool {
        false
    }
}
