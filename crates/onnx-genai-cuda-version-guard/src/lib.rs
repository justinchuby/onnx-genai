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

// ─────────────────────────────────────────────────────────────────────────────
// Wheel layout
//
// NVIDIA's redistributable wheels are how this project gets CUDA without a
// toolkit, and more than one crate has to find libraries inside them: the CUDA
// EP resolves cuBLASLt/NVRTC/cudart, and the tracer resolves CUPTI. Both were
// building `<root>/nvidia/<component>/{bin,lib}` by hand, so when NVIDIA
// republished the wheels under a consolidated layout both went looking in a
// directory that no longer exists — the same bug, twice, found once.
//
// The layout is a property of the wheels, not of any one consumer, so it lives
// here beside the library names rather than in whichever crate noticed first.
// ─────────────────────────────────────────────────────────────────────────────

/// CUDA major versions the consolidated wheels are published under.
///
/// Ordered newest first so a root carrying more than one prefers the newer,
/// which is also the one a default build can use — `cudarc` derives the library
/// names it will `dlopen` from its build-time CUDA version, so a CUDA 13 build
/// never looks for a CUDA 12 file.
pub const WHEEL_CUDA_MAJORS: &[&str] = &["cu13", "cu12"];

/// Directory holding loadable libraries inside a wheel component, per OS.
///
/// Windows keeps DLLs beside executables in `bin`; the Unix platforms use `lib`.
pub const fn wheel_library_directory(os: HostOs) -> &'static str {
    match os {
        HostOs::Windows => "bin",
        HostOs::Linux | HostOs::Macos | HostOs::Other => "lib",
    }
}

/// Architecture sub-directory used by the consolidated wheel layout.
///
/// The per-component wheels put libraries directly in `bin`/`lib`; the
/// consolidated ones add an architecture level below it, spelled differently
/// for binaries (`x86_64`) and import libraries (`x64`) on Windows.
pub const fn wheel_arch_directories(os: HostOs) -> &'static [&'static str] {
    match os {
        HostOs::Windows => &["x86_64", "x64"],
        HostOs::Linux => &["x86_64", "sbsa"],
        HostOs::Macos | HostOs::Other => &[],
    }
}

/// Relative directories, newest layout first, that may hold `component`'s
/// libraries inside a wheel root.
///
/// Two layouts are in circulation and a machine can carry both, because the
/// package names differ:
///
/// * consolidated — `nvidia/cu13/{bin,lib}/<arch>/`, one wheel per CUDA line
///   (`nvidia-cuda-nvrtc`, where the *version pin* selects the CUDA major);
/// * per-component — `nvidia/<component>/{bin,lib}/`, the older `-cu12`
///   spelling.
///
/// Returned relative so the caller joins them onto whatever roots it knows,
/// which differ per consumer (`sys.path` entries, `NXRT_CUDA_WHEEL_ROOTS`, the
/// loaded extension's directory).
pub fn wheel_component_directories(component: &str, os: HostOs) -> Vec<std::path::PathBuf> {
    use std::path::PathBuf;

    let library_directory = wheel_library_directory(os);
    let mut directories = Vec::new();

    for major in WHEEL_CUDA_MAJORS {
        let base = PathBuf::from("nvidia").join(major).join(library_directory);
        for arch in wheel_arch_directories(os) {
            directories.push(base.join(arch));
        }
        // Some consolidated wheels omit the architecture level entirely.
        directories.push(base);
    }

    directories.push(
        PathBuf::from("nvidia")
            .join(component)
            .join(library_directory),
    );
    directories
}

/// The wheel root a library directory belongs to, if it is one.
///
/// Both layouts share one shape:
///
/// ```text
/// <root>/nvidia/<component-or-cuda-major>/<bin|lib>[/<arch>]
/// ```
///
/// so rather than enumerating them, walk it: an optional architecture level,
/// the library directory, the component level, then `nvidia`.
///
/// The component level is load-bearing, not decoration. Without it a plain
/// system install like `/opt/nvidia/lib` would report `/opt` as a wheel root,
/// and every later failure would name a path nothing was ever installed to.
pub fn wheel_root_of(entry: &std::path::Path) -> Option<std::path::PathBuf> {
    fn is_library_directory(path: &std::path::Path) -> bool {
        matches!(
            path.file_name().and_then(|name| name.to_str()),
            Some("bin") | Some("lib")
        )
    }

    // Skip at most one architecture level, which only the consolidated layout has.
    let library_directory = if is_library_directory(entry) {
        entry
    } else {
        let parent = entry.parent()?;
        if !is_library_directory(parent) {
            return None;
        }
        parent
    };

    let component = library_directory.parent()?;
    let nvidia = component.parent()?;
    if nvidia.file_name()? != "nvidia" {
        return None;
    }
    Some(nvidia.parent()?.to_path_buf())
}

#[cfg(test)]
mod wheel_layout_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn the_consolidated_layout_is_searched_before_the_per_component_one() {
        let directories = wheel_component_directories("cuda_nvrtc", HostOs::Windows);
        assert_eq!(
            directories.first(),
            Some(&PathBuf::from("nvidia/cu13/bin/x86_64"))
        );
        assert!(directories.contains(&PathBuf::from("nvidia/cuda_nvrtc/bin")));
    }

    #[test]
    fn both_layouts_recover_their_root() {
        let root = Path::new("/venv/site-packages");
        // Consolidated, with and without the architecture level.
        assert_eq!(
            wheel_root_of(&root.join("nvidia/cu13/bin/x86_64")),
            Some(root.to_path_buf())
        );
        assert_eq!(
            wheel_root_of(&root.join("nvidia/cu13/bin")),
            Some(root.to_path_buf())
        );
        // Per-component.
        assert_eq!(
            wheel_root_of(&root.join("nvidia/cuda_cupti/lib")),
            Some(root.to_path_buf())
        );
    }

    #[test]
    fn a_directory_without_a_component_level_is_not_a_wheel_root() {
        // A plain system install. Claiming `/opt` here would send every later
        // failure to a path nothing was installed to.
        assert_eq!(wheel_root_of(Path::new("/opt/nvidia/lib")), None);
        assert_eq!(wheel_root_of(Path::new("/opt/nvidia/bin")), None);
        // Not under `nvidia` at all.
        assert_eq!(wheel_root_of(Path::new("/opt/notnvidia/cublas/lib")), None);
        // More than one level below the library directory.
        assert_eq!(
            wheel_root_of(Path::new("/venv/nvidia/cu13/bin/x86_64/extra")),
            None
        );
    }
}
