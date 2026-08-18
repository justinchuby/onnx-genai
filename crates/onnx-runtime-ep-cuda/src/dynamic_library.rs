//! Cross-platform CUDA shared-library discovery.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use libloading::Library;
use onnx_genai_cuda_version_guard::{HostOs, cudart_candidates_for};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CudaLibrary {
    Driver,
    Runtime,
    // Kept for the crate-local loader surface; kernels currently load cuBLASLt instead.
    #[allow(dead_code)]
    Cublas,
    CublasLt,
    Cudnn,
    Nvrtc,
    // Kept for the crate-local loader surface; CUPTI loading currently lives in the tracer crate.
    #[allow(dead_code)]
    Cupti,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetOs {
    // Kept for cross-target library discovery; constructed only in Linux builds and tests.
    #[cfg_attr(not(target_os = "linux"), allow(dead_code))]
    Linux,
    // Kept for cross-target library discovery; constructed only in macOS builds and tests.
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Macos,
    Windows,
    // Kept for cross-target library discovery; constructed only on unsupported target OSes and tests.
    #[cfg_attr(
        any(target_os = "linux", target_os = "macos", target_os = "windows"),
        allow(dead_code)
    )]
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TargetArch {
    Aarch64,
    Other,
}

#[cfg(target_os = "linux")]
fn target_os() -> TargetOs {
    TargetOs::Linux
}

#[cfg(target_os = "macos")]
fn target_os() -> TargetOs {
    TargetOs::Macos
}

#[cfg(target_os = "windows")]
fn target_os() -> TargetOs {
    TargetOs::Windows
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn target_os() -> TargetOs {
    TargetOs::Other
}

fn target_arch() -> TargetArch {
    if cfg!(target_arch = "aarch64") {
        TargetArch::Aarch64
    } else {
        TargetArch::Other
    }
}

/// Map this crate's [`TargetOs`] onto the version-guard [`HostOs`] selector used
/// by the canonical `cudart` candidate table.
fn host_os(os: TargetOs) -> HostOs {
    match os {
        TargetOs::Linux => HostOs::Linux,
        TargetOs::Macos => HostOs::Macos,
        TargetOs::Windows => HostOs::Windows,
        TargetOs::Other => HostOs::Other,
    }
}

pub(crate) fn candidates(library: CudaLibrary) -> &'static [&'static str] {
    candidates_for(target_os(), library)
}

fn candidates_for(os: TargetOs, library: CudaLibrary) -> &'static [&'static str] {
    match (os, library) {
        // The CUDA runtime (`cudart`) candidate list is the single canonical
        // table in `onnx-genai-cuda-version-guard`, shared with `onnx-genai-ort`'s
        // dynamically-loaded cudart shim so the two loaders can never drift
        // (see issue #1180).
        (os, CudaLibrary::Runtime) => cudart_candidates_for(host_os(os)),

        (TargetOs::Linux, CudaLibrary::Driver) => &["libcuda.so.1", "libcuda.so"],
        (TargetOs::Linux, CudaLibrary::Cublas) => {
            &["libcublas.so.13", "libcublas.so.12", "libcublas.so"]
        }
        (TargetOs::Linux, CudaLibrary::CublasLt) => {
            &["libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so"]
        }
        (TargetOs::Linux, CudaLibrary::Cudnn) => &["libcudnn.so.9", "libcudnn.so"],
        (TargetOs::Linux, CudaLibrary::Nvrtc) => {
            &["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"]
        }
        (TargetOs::Linux, CudaLibrary::Cupti) => {
            &["libcupti.so.13", "libcupti.so.12", "libcupti.so"]
        }

        (TargetOs::Macos, CudaLibrary::Driver) => &["libcuda.dylib"],
        (TargetOs::Macos, CudaLibrary::Cublas) => &["libcublas.dylib"],
        (TargetOs::Macos, CudaLibrary::CublasLt) => &["libcublasLt.dylib"],
        (TargetOs::Macos, CudaLibrary::Cudnn) => &["libcudnn.dylib"],
        (TargetOs::Macos, CudaLibrary::Nvrtc) => &["libnvrtc.dylib"],
        (TargetOs::Macos, CudaLibrary::Cupti) => &["libcupti.dylib"],

        (TargetOs::Windows, CudaLibrary::Driver) => &["nvcuda.dll"],
        (TargetOs::Windows, CudaLibrary::Cublas) => {
            &["cublas64_13.dll", "cublas64_12.dll", "cublas.dll"]
        }
        (TargetOs::Windows, CudaLibrary::CublasLt) => {
            &["cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt.dll"]
        }
        (TargetOs::Windows, CudaLibrary::Cudnn) => &["cudnn64_9.dll", "cudnn64_8.dll", "cudnn.dll"],
        (TargetOs::Windows, CudaLibrary::Nvrtc) => &[
            "nvrtc64_130_0.dll",
            "nvrtc64_120_0.dll",
            "nvrtc64_13.dll",
            "nvrtc64_12.dll",
            "nvrtc.dll",
        ],
        (TargetOs::Windows, CudaLibrary::Cupti) => {
            &["cupti64_13.dll", "cupti64_12.dll", "cupti.dll"]
        }

        (TargetOs::Other, _) => &[],
    }
}

fn nvidia_component(library: CudaLibrary) -> Option<&'static str> {
    match library {
        CudaLibrary::Driver => None,
        CudaLibrary::Runtime => Some("cuda_runtime"),
        CudaLibrary::Cublas | CudaLibrary::CublasLt => Some("cublas"),
        CudaLibrary::Cudnn => Some("cudnn"),
        CudaLibrary::Nvrtc => Some("cuda_nvrtc"),
        CudaLibrary::Cupti => Some("cuda_cupti"),
    }
}

fn wheel_library_directory(os: TargetOs) -> &'static str {
    match os {
        TargetOs::Windows => "bin",
        TargetOs::Linux | TargetOs::Macos | TargetOs::Other => "lib",
    }
}

fn wheel_candidates_for(root: &Path, os: TargetOs, library: CudaLibrary) -> Vec<PathBuf> {
    if !root.is_absolute() {
        return Vec::new();
    }
    let Some(component) = nvidia_component(library) else {
        return Vec::new();
    };
    let directory = root
        .join("nvidia")
        .join(component)
        .join(wheel_library_directory(os));
    candidates_for(os, library)
        .iter()
        .map(|name| directory.join(name))
        .collect()
}

/// Wheel roots the process environment already implies.
///
/// The Python bindings call [`set_wheel_search_paths`] with `site-packages`,
/// which is how `nxrt[cuda]` finds NVIDIA's wheels without a system CUDA
/// install. Nothing on the pure-Rust path did that, so `cargo test` found
/// `nvcuda.dll` — which ships with the display driver — but not cuBLAS, NVRTC
/// or the CUDA headers, and every GPU test skipped on a machine that could run
/// them perfectly well.
///
/// Two sources, in order of authority:
///
/// * `NXRT_CUDA_WHEEL_ROOTS`, for saying so explicitly. Same syntax as `PATH`.
/// * The platform's own loader path. Anyone who can load these wheels already
///   has `<root>/nvidia/<component>/{bin,lib}` on it, and the layout is fixed,
///   so the root is recoverable from any single entry. This generalises a
///   heuristic that already existed for `LD_LIBRARY_PATH` but had no Windows
///   counterpart, where the directory is `bin` rather than `lib`.
///
/// Relative entries are dropped for the same reason [`set_wheel_search_paths`]
/// drops them: a wheel must never be loaded relative to the process CWD.
fn wheel_roots_from_environment() -> Vec<PathBuf> {
    let mut roots = Vec::new();

    if let Some(value) = std::env::var_os("NXRT_CUDA_WHEEL_ROOTS") {
        roots.extend(std::env::split_paths(&value));
    }

    for variable in ["PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"] {
        let Some(value) = std::env::var_os(variable) else {
            continue;
        };
        for entry in std::env::split_paths(&value) {
            if let Some(root) = wheel_root_of(&entry) {
                roots.push(root);
            }
        }
    }

    roots.retain(|path| path.is_absolute());
    // `Vec::dedup` would only drop *consecutive* repeats, and one root is
    // reached through every component directory on the path, which need not be
    // adjacent. Duplicates are not incorrect, only repeated failed probes and a
    // noisier candidate list on failure -- but the explicit path already rejects
    // them this way, so both behave alike.
    let mut unique = Vec::with_capacity(roots.len());
    for root in roots {
        if !unique.contains(&root) {
            unique.push(root);
        }
    }
    unique
}

/// The root a `<root>/nvidia/<component>/{bin,lib}` entry belongs to.
fn wheel_root_of(entry: &Path) -> Option<PathBuf> {
    let library_directory = entry.file_name()?;
    if library_directory != "bin" && library_directory != "lib" {
        return None;
    }
    let component = entry.parent()?;
    let nvidia = component.parent()?;
    if nvidia.file_name()? != "nvidia" {
        return None;
    }
    Some(nvidia.parent()?.to_path_buf())
}

fn wheel_search_paths() -> &'static Mutex<Vec<PathBuf>> {
    static PATHS: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    PATHS.get_or_init(|| Mutex::new(wheel_roots_from_environment()))
}

fn loaded_libraries() -> &'static Mutex<Vec<(CudaLibrary, Library)>> {
    static LIBRARIES: OnceLock<Mutex<Vec<(CudaLibrary, Library)>>> = OnceLock::new();
    LIBRARIES.get_or_init(|| Mutex::new(Vec::new()))
}

/// How many entries at the front of the search list were configured
/// explicitly rather than inferred from the environment.
fn explicit_root_count() -> &'static std::sync::atomic::AtomicUsize {
    static COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    &COUNT
}

/// Insert `paths` into `roots` at `at`, skipping relative and duplicate
/// entries, and answer how many were added.
///
/// Split out from [`set_wheel_search_paths`] so the precedence rule can be
/// tested without touching process-global state.
fn insert_explicit_roots(
    roots: &mut Vec<PathBuf>,
    at: usize,
    paths: impl IntoIterator<Item = PathBuf>,
) -> usize {
    let mut added = 0;
    for path in paths {
        if path.is_absolute() && !roots.contains(&path) {
            roots.insert(at + added, path);
            added += 1;
        }
    }
    added
}

/// Add roots such as Python's `site-packages` directory to the CUDA wheel search
/// path. NVIDIA's pip wheels install component libraries beneath
/// `nvidia/<component>/{lib,bin}` relative to these roots. Relative roots are
/// rejected so wheel libraries are never loaded relative to the process CWD.
///
/// These take precedence over roots inferred from the environment. The Python
/// bindings pass the interpreter's own `site-packages`, and a different CUDA
/// major version sitting on the machine's `PATH` must not outrank the
/// environment the caller actually selected — loading components from two
/// toolkits at once is worse than finding none.
pub fn set_wheel_search_paths(paths: impl IntoIterator<Item = PathBuf>) {
    use std::sync::atomic::Ordering;

    let mut configured = wheel_search_paths()
        .lock()
        .expect("CUDA wheel search-path lock poisoned");
    let at = explicit_root_count().load(Ordering::Relaxed);
    let added = insert_explicit_roots(&mut configured, at, paths);
    explicit_root_count().fetch_add(added, Ordering::Relaxed);
}

/// CUDA header directories owned by configured NVIDIA wheels.
///
/// Two components, because the headers NVRTC needs are split across two wheels.
/// `nvidia-cuda-runtime` carries `cuda_fp16.h`, `cuda_bf16.h` and `mma.h`; the
/// `crt/` tree that `mma.h` itself includes ships in `nvidia-cuda-nvcc`.
/// Offering only the first gets as far as `mma.h` and then fails with
/// `cannot open source file "crt/mma.h"`, which is a long way from naming the
/// wheel that is missing.
pub(crate) fn wheel_cuda_include_paths() -> Vec<PathBuf> {
    let roots = wheel_search_paths()
        .lock()
        .expect("CUDA wheel search-path lock poisoned")
        .clone();
    roots
        .into_iter()
        .flat_map(|root| {
            let nvidia = root.join("nvidia");
            ["cuda_runtime", "cuda_nvcc"]
                .into_iter()
                .map(move |component| nvidia.join(component).join("include"))
        })
        .collect()
}

fn load_library(library: CudaLibrary) -> Result<Library, Vec<String>> {
    let os = target_os();
    let roots = wheel_search_paths()
        .lock()
        .expect("CUDA wheel search-path lock poisoned")
        .clone();
    let wheel_candidates = roots
        .iter()
        .flat_map(|root| wheel_candidates_for(root, os, library))
        .collect::<Vec<_>>();
    let mut tried = Vec::new();

    for path in wheel_candidates {
        if !path.is_absolute() {
            continue;
        }
        tried.push(path.display().to_string());
        // SAFETY: paths are supplied by nxrt's installed package layout and the
        // handle is retained for the lifetime of the process below.
        if let Ok(handle) = unsafe { Library::new(&path) } {
            // A wheel component is a self-contained set of libraries that load
            // each other *by base name* at runtime: NVRTC opens
            // `nvrtc-builtins` when it compiles, cuDNN opens its own engine
            // libraries when a handle is created. Those go through the process
            // search order, which does not include the directory the caller was
            // loaded from -- so an absolute path to the entry point does not
            // help with the loads it makes on its own behalf.
            //
            // Loading the siblings here, by absolute path, puts modules of
            // those base names in the process, and the component's own requests
            // resolve to them.
            //
            // Chosen over the alternatives deliberately. Mutating the process
            // environment is unsound while another thread reads it, and on
            // glibc it is also a no-op -- `LD_LIBRARY_PATH` is snapshotted at
            // startup, so a later `dlopen` never sees the change.
            // `AddDllDirectory` requires `SetDefaultDllDirectories`, which
            // changes resolution for every library in the process, CUDA or not.
            if let Some(directory) = path.parent() {
                preload_component_siblings(directory, &path);
            }
            return Ok(handle);
        }
    }
    for name in candidates(library) {
        tried.push((*name).to_string());
        // SAFETY: these are trusted NVIDIA libraries. The returned handle is
        // retained for the process lifetime by `require`.
        if let Ok(handle) = unsafe { Library::new(name) } {
            return Ok(handle);
        }
    }
    Err(tried)
}

pub(crate) fn is_available(library: CudaLibrary) -> bool {
    require(library).is_ok()
}

/// Load the other libraries that live beside `loaded` in its wheel component.
///
/// An NVIDIA wheel component is a self-contained set whose members load each
/// other by base name at runtime -- NVRTC opens `nvrtc-builtins` when it
/// compiles, cuDNN opens its engine libraries when a handle is created. Those
/// requests go through the process search order, which does not include the
/// directory the caller was loaded from, so an absolute path to the entry point
/// does not help with them. Loading the siblings puts modules of those base
/// names in the process, and the requests resolve to them.
///
/// Best effort by design. A sibling that will not load is not necessarily
/// needed, and if it is, the failure arrives later with the component's own
/// message naming the file -- more precise than anything this could say in
/// advance. What it must not do is fail the load that just succeeded.
fn preload_component_siblings(directory: &Path, loaded: &Path) {
    static SEEN: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    let mut seen = SEEN
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("CUDA sibling-preload lock poisoned");
    if seen.iter().any(|existing| existing == directory) {
        return;
    }
    seen.push(directory.to_path_buf());

    let Ok(entries) = std::fs::read_dir(directory) else {
        return;
    };
    let suffix = if cfg!(windows) { ".dll" } else { ".so" };
    for entry in entries.flatten() {
        let path = entry.path();
        if path == loaded {
            continue;
        }
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if !name.contains(suffix) {
            continue;
        }
        // SAFETY: an NVIDIA component from the same wheel directory as the
        // library that was just loaded from it. The handle is leaked on
        // purpose: these modules must outlive every use of that component,
        // which is the life of the process.
        if let Ok(handle) = unsafe { Library::new(&path) } {
            std::mem::forget(handle);
        }
    }
}

fn cuda_supported(os: TargetOs, arch: TargetArch) -> bool {
    !(os == TargetOs::Windows && arch == TargetArch::Aarch64)
}

pub(crate) fn require(library: CudaLibrary) -> Result<(), String> {
    if !cuda_supported(target_os(), target_arch()) {
        return Err(
            "CUDA is unavailable on Windows ARM64 because NVIDIA ships x64-only CUDA libraries"
                .into(),
        );
    }
    let mut loaded = loaded_libraries()
        .lock()
        .expect("CUDA loaded-library lock poisoned");
    if loaded
        .iter()
        .any(|(loaded_library, _)| *loaded_library == library)
    {
        return Ok(());
    }
    let handle = load_library(library)
        .map_err(|tried| format!("CUDA {library:?} library not found; tried {tried:?}"))?;
    loaded.push((library, handle));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_linux_cuda_names() {
        assert_eq!(
            candidates_for(TargetOs::Linux, CudaLibrary::Runtime),
            onnx_genai_cuda_version_guard::CUDART_CANDIDATES_LINUX
        );
        assert_eq!(
            candidates_for(TargetOs::Linux, CudaLibrary::Cupti),
            ["libcupti.so.13", "libcupti.so.12", "libcupti.so"]
        );
    }

    #[test]
    fn generates_macos_cuda_names() {
        assert_eq!(
            candidates_for(TargetOs::Macos, CudaLibrary::Cublas),
            ["libcublas.dylib"]
        );
        assert_eq!(
            candidates_for(TargetOs::Macos, CudaLibrary::Nvrtc),
            ["libnvrtc.dylib"]
        );
    }

    #[test]
    fn generates_windows_cuda_names() {
        assert_eq!(
            candidates_for(TargetOs::Windows, CudaLibrary::Runtime),
            onnx_genai_cuda_version_guard::CUDART_CANDIDATES_WINDOWS
        );
        assert!(candidates_for(TargetOs::Windows, CudaLibrary::Cudnn).contains(&"cudnn64_9.dll"));
        assert!(
            candidates_for(TargetOs::Windows, CudaLibrary::Nvrtc).contains(&"nvrtc64_130_0.dll")
        );
        assert!(candidates_for(TargetOs::Windows, CudaLibrary::Cupti).contains(&"cupti64_13.dll"));
    }

    #[test]
    fn locates_wheel_libraries_beneath_absolute_package_root() {
        let root = std::env::current_dir()
            .expect("current directory should be available")
            .join("site-packages");
        assert_eq!(
            wheel_candidates_for(&root, TargetOs::Linux, CudaLibrary::CublasLt),
            vec![
                root.join("nvidia/cublas/lib/libcublasLt.so.13"),
                root.join("nvidia/cublas/lib/libcublasLt.so.12"),
                root.join("nvidia/cublas/lib/libcublasLt.so"),
            ]
        );
        assert_eq!(
            wheel_candidates_for(&root, TargetOs::Windows, CudaLibrary::Nvrtc),
            vec![
                root.join("nvidia/cuda_nvrtc/bin/nvrtc64_130_0.dll"),
                root.join("nvidia/cuda_nvrtc/bin/nvrtc64_120_0.dll"),
                root.join("nvidia/cuda_nvrtc/bin/nvrtc64_13.dll"),
                root.join("nvidia/cuda_nvrtc/bin/nvrtc64_12.dll"),
                root.join("nvidia/cuda_nvrtc/bin/nvrtc.dll"),
            ]
        );
    }

    #[test]
    fn empty_python_search_path_produces_no_relative_candidates() {
        for root in ["", "   ", "site-packages"] {
            let candidates =
                wheel_candidates_for(Path::new(root), TargetOs::Linux, CudaLibrary::CublasLt);
            assert!(
                candidates.is_empty(),
                "relative sys.path entry {root:?} produced dlopen candidates: {candidates:?}"
            );
        }
    }

    #[test]
    fn driver_is_not_expected_in_a_python_wheel() {
        let root = std::env::current_dir()
            .expect("current directory should be available")
            .join("site-packages");
        assert!(wheel_candidates_for(&root, TargetOs::Linux, CudaLibrary::Driver).is_empty());
    }

    #[test]
    fn unsupported_platform_has_no_candidates() {
        assert!(candidates_for(TargetOs::Other, CudaLibrary::Driver).is_empty());
    }

    /// A wheel library directory on the loader path identifies its root.
    ///
    /// This is what lets the pure-Rust path find NVIDIA's wheels at all. Only
    /// the Python bindings ever called `set_wheel_search_paths`, so `cargo
    /// test` found `nvcuda.dll` -- which ships with the display driver -- but
    /// not cuBLAS, NVRTC or the CUDA headers, and 44 GPU test files skipped on
    /// a machine that could run them.
    ///
    /// Split by platform because `Path` parses with host semantics: a
    /// backslash is an ordinary filename character on Unix, so a Windows
    /// literal here would assert nothing there.
    #[test]
    #[cfg(windows)]
    fn a_wheel_library_directory_identifies_its_root() {
        // Windows wheels put their libraries in `bin`.
        assert_eq!(
            wheel_root_of(Path::new(r"C:\py\site-packages\nvidia\cuda_nvrtc\bin")),
            Some(PathBuf::from(r"C:\py\site-packages"))
        );
        // Forward slashes are also separators on Windows.
        assert_eq!(
            wheel_root_of(Path::new("C:/py/site-packages/nvidia/cublas/lib")),
            Some(PathBuf::from("C:/py/site-packages"))
        );
    }

    /// See the Windows counterpart above.
    #[test]
    #[cfg(not(windows))]
    fn a_wheel_library_directory_identifies_its_root() {
        assert_eq!(
            wheel_root_of(Path::new("/opt/venv/site-packages/nvidia/cublas/lib")),
            Some(PathBuf::from("/opt/venv/site-packages"))
        );
        assert_eq!(
            wheel_root_of(Path::new("/opt/venv/site-packages/nvidia/cuda_nvrtc/bin")),
            Some(PathBuf::from("/opt/venv/site-packages"))
        );
    }

    /// Directories that merely resemble the layout are not treated as roots.
    ///
    /// The loader path is full of unrelated `bin` and `lib` directories. Taking
    /// a grandparent from one of those would add a bogus search root and make
    /// every later failure report the wrong path.
    #[test]
    fn a_directory_that_is_not_a_wheel_component_is_not_a_root() {
        for entry in [
            "/usr/local/lib",
            "/usr/bin",
            "/opt/nvidia/lib",
            "/opt/site-packages/nvidia/cublas",
            "/opt/site-packages/nvidia/cublas/lib64",
            "/opt/notnvidia/cublas/lib",
        ] {
            assert_eq!(
                wheel_root_of(Path::new(entry)),
                None,
                "{entry} is not a wheel component directory"
            );
        }
    }

    /// An explicitly configured root is searched before any the environment
    /// merely implied.
    ///
    /// The Python bindings pass the interpreter's own `site-packages`. A
    /// different CUDA major version sitting on the machine's `PATH` must not
    /// take precedence over the environment the caller actually selected,
    /// because mixing the two loads components from two different toolkits.
    #[test]
    fn an_explicit_root_outranks_one_derived_from_the_environment() {
        // Built from an absolute base because "absolute" is platform-specific:
        // a leading slash is not absolute on Windows.
        let base = std::env::current_dir().expect("a current directory");
        let derived = base.join("derived-from-path");
        let explicit = base.join("chosen-by-the-caller");
        let second = base.join("also-chosen");

        let mut roots = vec![derived.clone()];
        assert_eq!(insert_explicit_roots(&mut roots, 0, [explicit.clone()]), 1);
        assert_eq!(roots.first(), Some(&explicit));

        // A second explicit root follows the first, still ahead of the derived one.
        assert_eq!(insert_explicit_roots(&mut roots, 1, [second.clone()]), 1);
        assert_eq!(roots, vec![explicit.clone(), second, derived]);

        // A repeat is not added again, and reports that it added nothing so the
        // insertion point does not drift past the end.
        assert_eq!(insert_explicit_roots(&mut roots, 2, [explicit]), 0);

        // A relative root is refused, so a wheel is never loaded relative to
        // the process working directory.
        assert_eq!(
            insert_explicit_roots(&mut roots, 2, [PathBuf::from("site-packages")]),
            0
        );
    }

    #[test]
    fn windows_arm64_is_an_explicit_cpu_only_target() {
        assert!(!cuda_supported(TargetOs::Windows, TargetArch::Aarch64));
        assert!(cuda_supported(TargetOs::Windows, TargetArch::Other));
    }
}
