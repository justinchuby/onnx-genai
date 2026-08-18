#[cfg(all(
    feature = "cuda",
    not(any(
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))
))]
compile_error!(
    "onnx-genai CUDA build: no CUDA version selected. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000."
);

#[cfg(any(
    all(feature = "cuda-12060", feature = "cuda-12080"),
    all(feature = "cuda-12060", feature = "cuda-12090"),
    all(feature = "cuda-12060", feature = "cuda-13000"),
    all(feature = "cuda-12080", feature = "cuda-12090"),
    all(feature = "cuda-12080", feature = "cuda-13000"),
    all(feature = "cuda-12090", feature = "cuda-13000")
))]
compile_error!(
    "onnx-genai CUDA build: multiple CUDA versions selected; cudarc bindings cannot compile with more than one. Enable exactly one of cuda-12060 | cuda-12080 | cuda-12090 | cuda-13000 (and set default-features = false on inter-crate deps if you override the default)."
);

/// Host operating system selector for [`cudart_candidates_for`].
///
/// The CUDA runtime (`cudart`) ships under different file names per platform, so
/// the canonical candidate list is keyed on the host OS rather than a single
/// flat cross-platform list.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HostOs {
    /// Linux (`libcudart.so*`).
    Linux,
    /// macOS (`libcudart.dylib`).
    Macos,
    /// Windows (`cudart64_*.dll`).
    Windows,
    /// Any other / unsupported target OS (no candidates).
    Other,
}

/// Linux CUDA runtime (`cudart`) shared-library candidate names, most specific
/// first. Newer CUDA majors are listed ahead of older ones so a host with
/// several runtimes prefers the newest, and the bare `libcudart.so` lets the
/// platform loader resolve whatever is already on the search path.
pub const CUDART_CANDIDATES_LINUX: &[&str] =
    &["libcudart.so.13", "libcudart.so.12", "libcudart.so"];

/// macOS CUDA runtime (`cudart`) shared-library candidate names.
pub const CUDART_CANDIDATES_MACOS: &[&str] = &["libcudart.dylib"];

/// Windows CUDA runtime (`cudart`) shared-library candidate names, most specific
/// first. Windows ships versioned DLLs (`cudart64_13.dll` for CUDA 13.x,
/// `cudart64_12.dll` for 12.x, older `cudart64_120.dll`); the bare `cudart.dll`
/// lets the platform loader resolve a name already on the search path.
pub const CUDART_CANDIDATES_WINDOWS: &[&str] = &[
    "cudart64_13.dll",
    "cudart64_12.dll",
    "cudart64_120.dll",
    "cudart.dll",
];

/// Canonical CUDA runtime (`cudart`) shared-library candidate names for `os`,
/// most specific first.
///
/// This is the single source of truth every `cudart` loader in the workspace
/// reads. It must cover every CUDA major version any loader accepts: when two
/// loaders disagree, one loads and the other reports the *last* candidate it
/// tried (e.g. a Linux `.so` name failing on Windows), which reads like "this
/// machine has no CUDA" rather than "this list is stale" (see issue #1180).
pub const fn cudart_candidates_for(os: HostOs) -> &'static [&'static str] {
    match os {
        HostOs::Linux => CUDART_CANDIDATES_LINUX,
        HostOs::Macos => CUDART_CANDIDATES_MACOS,
        HostOs::Windows => CUDART_CANDIDATES_WINDOWS,
        HostOs::Other => &[],
    }
}

/// Canonical CUDA runtime (`cudart`) candidate names for the current build
/// target's OS. See [`cudart_candidates_for`].
pub fn cudart_candidates() -> &'static [&'static str] {
    #[cfg(target_os = "linux")]
    {
        cudart_candidates_for(HostOs::Linux)
    }
    #[cfg(target_os = "macos")]
    {
        cudart_candidates_for(HostOs::Macos)
    }
    #[cfg(target_os = "windows")]
    {
        cudart_candidates_for(HostOs::Windows)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
    {
        cudart_candidates_for(HostOs::Other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linux_lists_cuda_13_and_12() {
        assert_eq!(
            cudart_candidates_for(HostOs::Linux),
            ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
        );
    }

    #[test]
    fn windows_lists_cuda_13_and_12() {
        let names = cudart_candidates_for(HostOs::Windows);
        assert!(names.contains(&"cudart64_13.dll"));
        assert!(names.contains(&"cudart64_12.dll"));
    }

    #[test]
    fn unsupported_os_has_no_candidates() {
        assert!(cudart_candidates_for(HostOs::Other).is_empty());
    }

    #[test]
    fn current_target_candidates_are_nonempty_on_supported_hosts() {
        if cfg!(any(
            target_os = "linux",
            target_os = "macos",
            target_os = "windows"
        )) {
            assert!(!cudart_candidates().is_empty());
        }
    }
}
