//! Cross-platform CUDA shared-library discovery.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use libloading::Library;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CudaLibrary {
    Driver,
    // Kept for the crate-local loader surface; no production caller probes cudart yet.
    #[allow(dead_code)]
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

pub(crate) fn candidates(library: CudaLibrary) -> &'static [&'static str] {
    candidates_for(target_os(), library)
}

fn candidates_for(os: TargetOs, library: CudaLibrary) -> &'static [&'static str] {
    match (os, library) {
        (TargetOs::Linux, CudaLibrary::Driver) => &["libcuda.so.1", "libcuda.so"],
        (TargetOs::Linux, CudaLibrary::Runtime) => {
            &["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
        }
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
        (TargetOs::Macos, CudaLibrary::Runtime) => &["libcudart.dylib"],
        (TargetOs::Macos, CudaLibrary::Cublas) => &["libcublas.dylib"],
        (TargetOs::Macos, CudaLibrary::CublasLt) => &["libcublasLt.dylib"],
        (TargetOs::Macos, CudaLibrary::Cudnn) => &["libcudnn.dylib"],
        (TargetOs::Macos, CudaLibrary::Nvrtc) => &["libnvrtc.dylib"],
        (TargetOs::Macos, CudaLibrary::Cupti) => &["libcupti.dylib"],

        (TargetOs::Windows, CudaLibrary::Driver) => &["nvcuda.dll"],
        (TargetOs::Windows, CudaLibrary::Runtime) => {
            &["cudart64_13.dll", "cudart64_12.dll", "cudart.dll"]
        }
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

fn wheel_search_paths() -> &'static Mutex<Vec<PathBuf>> {
    static PATHS: OnceLock<Mutex<Vec<PathBuf>>> = OnceLock::new();
    PATHS.get_or_init(|| Mutex::new(Vec::new()))
}

fn loaded_libraries() -> &'static Mutex<Vec<(CudaLibrary, Library)>> {
    static LIBRARIES: OnceLock<Mutex<Vec<(CudaLibrary, Library)>>> = OnceLock::new();
    LIBRARIES.get_or_init(|| Mutex::new(Vec::new()))
}

/// Add roots such as Python's `site-packages` directory to the CUDA wheel search
/// path. NVIDIA's pip wheels install component libraries beneath
/// `nvidia/<component>/{lib,bin}` relative to these roots. Relative roots are
/// rejected so wheel libraries are never loaded relative to the process CWD.
pub fn set_wheel_search_paths(paths: impl IntoIterator<Item = PathBuf>) {
    let mut configured = wheel_search_paths()
        .lock()
        .expect("CUDA wheel search-path lock poisoned");
    for path in paths {
        if path.is_absolute() && !configured.contains(&path) {
            configured.push(path);
        }
    }
}

/// CUDA runtime-header directories owned by configured NVIDIA wheels.
pub(crate) fn wheel_cuda_include_paths() -> Vec<PathBuf> {
    let roots = wheel_search_paths()
        .lock()
        .expect("CUDA wheel search-path lock poisoned")
        .clone();
    roots
        .into_iter()
        .map(|root| root.join("nvidia").join("cuda_runtime").join("include"))
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
            ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
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
            ["cudart64_13.dll", "cudart64_12.dll", "cudart.dll"]
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

    #[test]
    fn windows_arm64_is_an_explicit_cpu_only_target() {
        assert!(!cuda_supported(TargetOs::Windows, TargetArch::Aarch64));
        assert!(cuda_supported(TargetOs::Windows, TargetArch::Other));
    }
}
