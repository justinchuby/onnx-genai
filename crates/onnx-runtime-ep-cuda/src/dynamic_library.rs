//! Cross-platform CUDA shared-library discovery.

use libloading::Library;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) enum CudaLibrary {
    Driver,
    Runtime,
    Cublas,
    CublasLt,
    Cudnn,
    Nvrtc,
    Cupti,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum TargetOs {
    Linux,
    Macos,
    Windows,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
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

pub(crate) fn is_available(library: CudaLibrary) -> bool {
    candidates(library).iter().any(|name| {
        // SAFETY: these are trusted NVIDIA libraries; the handle is used only as
        // a loader probe and is dropped immediately.
        unsafe { Library::new(name) }.is_ok()
    })
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
    if is_available(library) {
        Ok(())
    } else {
        Err(format!(
            "CUDA {library:?} library not found; tried {:?}",
            candidates(library)
        ))
    }
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
    fn unsupported_platform_has_no_candidates() {
        assert!(candidates_for(TargetOs::Other, CudaLibrary::Driver).is_empty());
    }

    #[test]
    fn windows_arm64_is_an_explicit_cpu_only_target() {
        assert!(!cuda_supported(TargetOs::Windows, TargetArch::Aarch64));
        assert!(cuda_supported(TargetOs::Windows, TargetArch::Other));
    }
}
