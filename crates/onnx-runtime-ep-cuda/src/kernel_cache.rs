//! On-disk cache for NVRTC-compiled kernels.
//!
//! Every NVRTC module in this EP is compiled from source the first time it is
//! launched and then kept in a process-local `HashMap`. That map dies with the
//! process, so a fresh run recompiles every kernel — and because compilation is
//! lazy, the decode-only kernels are all compiled inside the *first decode
//! step*, which showed up as a single ~480 ms inter-token stall on Muse Glimmer
//! 30B (p50 23 ms, max 481 ms).
//!
//! This module persists the compiler output next to the user's cache directory
//! so the second and later runs skip NVRTC entirely.
//!
//! Design constraints:
//!
//! * **The cache is never load-bearing.** Every failure path (unreadable
//!   directory, truncated file, permission error) falls through to compiling
//!   from source. There is no error type here on purpose.
//! * **Keys cover everything that can change the output**: the module source,
//!   the target architecture, the include paths, the artifact kind, and the
//!   NVRTC version. A changed key is a miss, never a stale hit.
//! * **Writes are atomic.** Artifacts are written to a unique temporary file in
//!   the same directory and renamed into place, so a concurrent reader either
//!   sees the whole artifact or no file at all.
//!
//! Set `ONNX_GENAI_KERNEL_CACHE=0` to disable, or `ONNX_GENAI_KERNEL_CACHE_DIR`
//! to relocate it.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Which compiler artifact a cache entry holds.
///
/// PTX and CUBIN are cached separately because they come from different NVRTC
/// invocations targeting different architectures (`compute_80` vs `sm_80`) and
/// are loaded through different driver paths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArtifactKind {
    Ptx,
    Cubin,
}

impl ArtifactKind {
    fn tag(self) -> &'static str {
        match self {
            Self::Ptx => "ptx",
            Self::Cubin => "cubin",
        }
    }
}

/// Everything that can change what NVRTC emits for a module.
///
/// Anything omitted here would be a correctness bug: two different compilations
/// would collide on one filename and the second run would load the wrong code.
pub(crate) struct CacheKey<'a> {
    pub(crate) module_key: &'a str,
    pub(crate) source: &'a str,
    pub(crate) arch: &'a str,
    pub(crate) include_paths: &'a [String],
    pub(crate) kind: ArtifactKind,
}

impl CacheKey<'_> {
    /// A filename that is stable across processes and machines with the same
    /// toolkit.
    ///
    /// `std::hash::DefaultHasher` is deliberately not used: its output is only
    /// guaranteed stable within one Rust release, which would silently empty
    /// the cache on every toolchain bump. FNV-1a is fixed by its specification.
    fn file_name(&self) -> String {
        let mut hash = Fnv128::new();
        hash.field(self.module_key.as_bytes());
        hash.field(self.source.as_bytes());
        hash.field(self.arch.as_bytes());
        hash.field(self.kind.tag().as_bytes());
        for path in self.include_paths {
            hash.field(path.as_bytes());
        }
        hash.field(nvrtc_version_tag().as_bytes());
        // The module key is kept in the clear so a human looking at the cache
        // directory can tell which kernel an entry belongs to.
        let sanitized: String = self
            .module_key
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
            .collect();
        format!("{sanitized}-{:032x}.{}", hash.finish(), self.kind.tag())
    }
}

/// FNV-1a, 128-bit, with length-prefixed fields.
///
/// Length prefixing is what stops `("ab", "c")` and `("a", "bc")` from hashing
/// alike, which for a cache key would mean loading another kernel's code.
struct Fnv128(u128);

impl Fnv128 {
    const OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
    const PRIME: u128 = 0x0000000001000000000000000000013b;

    fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn field(&mut self, bytes: &[u8]) {
        self.write(&(bytes.len() as u64).to_le_bytes());
        self.write(bytes);
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u128::from(byte);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    fn finish(&self) -> u128 {
        self.0
    }
}

/// NVRTC's own version, so a toolkit upgrade cannot serve last toolkit's PTX.
fn nvrtc_version_tag() -> &'static str {
    static TAG: OnceLock<String> = OnceLock::new();
    TAG.get_or_init(|| {
        let mut major = 0i32;
        let mut minor = 0i32;
        // SAFETY: both out-params are live for the duration of the call. A
        // failed call leaves them at zero, which is a valid (if less specific)
        // tag; the cache only needs the value to change when NVRTC does.
        let ok = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }
            == cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS;
        if ok {
            format!("nvrtc{major}.{minor}")
        } else {
            "nvrtc-unknown".to_string()
        }
    })
    .as_str()
}

/// Time spent inside NVRTC this process, and how much the cache avoided.
///
/// Kept as process-global counters rather than runtime fields because the
/// interesting question ("did this run compile kernels?") is per-process, and
/// because tests want to read it without owning a `CudaRuntime`.
static COMPILED_NS: AtomicU64 = AtomicU64::new(0);
static COMPILED_COUNT: AtomicU64 = AtomicU64::new(0);
static HIT_COUNT: AtomicU64 = AtomicU64::new(0);

/// Compile statistics for the current process.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelCompileStats {
    /// Modules this process compiled with NVRTC.
    pub compiled: u64,
    /// Modules this process loaded from the on-disk cache instead.
    pub cache_hits: u64,
    /// Wall time spent inside NVRTC.
    pub compile_time: Duration,
}

/// Read the process-wide NVRTC compile counters.
///
/// A run with a warm cache reports `compiled == 0`, which is the signal that
/// the first-token stall this cache exists to remove is actually gone.
pub fn kernel_compile_stats() -> KernelCompileStats {
    KernelCompileStats {
        compiled: COMPILED_COUNT.load(Ordering::Relaxed),
        cache_hits: HIT_COUNT.load(Ordering::Relaxed),
        compile_time: Duration::from_nanos(COMPILED_NS.load(Ordering::Relaxed)),
    }
}

pub(crate) fn record_compile(elapsed: Duration) {
    COMPILED_COUNT.fetch_add(1, Ordering::Relaxed);
    COMPILED_NS.fetch_add(elapsed.as_nanos() as u64, Ordering::Relaxed);
}

fn record_hit() {
    HIT_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// Directory holding the cache, resolved once.
///
/// `None` means "no cache this process": either the user disabled it or no
/// writable location could be established. Both are handled the same way —
/// compile from source, as if this module did not exist.
fn cache_dir() -> Option<&'static Path> {
    static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
    DIR.get_or_init(|| {
        if let Ok(value) = std::env::var("ONNX_GENAI_KERNEL_CACHE") {
            let value = value.trim().to_ascii_lowercase();
            if matches!(value.as_str(), "0" | "off" | "false" | "no") {
                return None;
            }
        }
        let dir = match std::env::var_os("ONNX_GENAI_KERNEL_CACHE_DIR") {
            Some(dir) => PathBuf::from(dir),
            None => base_cache_dir()?.join("onnx-genai").join("nvrtc"),
        };
        std::fs::create_dir_all(&dir).ok()?;
        Some(dir)
    })
    .as_deref()
}

fn base_cache_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        let xdg = PathBuf::from(xdg);
        if xdg.is_absolute() {
            return Some(xdg);
        }
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache"))
}

/// Look up a previously compiled artifact.
pub(crate) fn load(key: &CacheKey<'_>) -> Option<Vec<u8>> {
    let bytes = load_in(cache_dir()?, key)?;
    record_hit();
    Some(bytes)
}

fn load_in(dir: &Path, key: &CacheKey<'_>) -> Option<Vec<u8>> {
    let bytes = std::fs::read(dir.join(key.file_name())).ok()?;
    // A zero-length artifact can only come from a bug or a truncated write;
    // treat it as a miss so the caller recompiles and overwrites it.
    if bytes.is_empty() {
        return None;
    }
    Some(bytes)
}

/// Publish a freshly compiled artifact for the next process.
///
/// Best effort: a failure here costs the next run a recompile and nothing else.
pub(crate) fn store(key: &CacheKey<'_>, bytes: &[u8]) {
    let Some(dir) = cache_dir() else { return };
    store_in(dir, key, bytes);
}

fn store_in(dir: &Path, key: &CacheKey<'_>, bytes: &[u8]) {
    if bytes.is_empty() {
        return;
    }
    let final_path = dir.join(key.file_name());
    // Unique per writer so two processes compiling the same kernel at once
    // cannot interleave into one another's temporary file.
    let temp_path = dir.join(format!(
        ".{}.{}.{:x}.tmp",
        key.file_name(),
        std::process::id(),
        temp_counter()
    ));
    if std::fs::write(&temp_path, bytes).is_err() {
        let _ = std::fs::remove_file(&temp_path);
        return;
    }
    // `rename` within a directory is atomic, so a concurrent reader sees either
    // the old artifact or the complete new one — never a partial file.
    if std::fs::rename(&temp_path, &final_path).is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }
}

fn temp_counter() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key<'a>(module: &'a str, source: &'a str, includes: &'a [String]) -> CacheKey<'a> {
        CacheKey {
            module_key: module,
            source,
            arch: "compute_80",
            include_paths: includes,
            kind: ArtifactKind::Ptx,
        }
    }

    #[test]
    fn identical_inputs_hash_to_the_same_file() {
        let includes = vec!["/usr/include".to_string()];
        assert_eq!(
            key("gemv", "__global__ void k() {}", &includes).file_name(),
            key("gemv", "__global__ void k() {}", &includes).file_name()
        );
    }

    #[test]
    fn every_key_component_changes_the_file_name() {
        let includes = vec!["/usr/include".to_string()];
        let base = key("gemv", "src", &includes).file_name();

        assert_ne!(base, key("other", "src", &includes).file_name(), "module");
        assert_ne!(base, key("gemv", "src2", &includes).file_name(), "source");

        let other_includes = vec!["/opt/include".to_string()];
        assert_ne!(
            base,
            key("gemv", "src", &other_includes).file_name(),
            "include paths"
        );

        let mut arch = key("gemv", "src", &includes);
        arch.arch = "compute_90";
        assert_ne!(base, arch.file_name(), "architecture");

        let mut kind = key("gemv", "src", &includes);
        kind.kind = ArtifactKind::Cubin;
        assert_ne!(base, kind.file_name(), "artifact kind");
    }

    #[test]
    fn field_boundaries_are_not_ambiguous() {
        // Without length prefixing these two would hash identically, and one
        // kernel would be served the other's compiled code.
        let includes: Vec<String> = Vec::new();
        assert_ne!(
            key("ab", "c", &includes).file_name(),
            key("a", "bc", &includes).file_name()
        );
    }

    #[test]
    fn file_name_keeps_the_module_readable_and_the_path_safe() {
        let includes: Vec<String> = Vec::new();
        let name = key("fused/attention softmax", "src", &includes).file_name();
        assert!(name.starts_with("fused_attention_softmax-"), "{name}");
        assert!(!name.contains('/'), "{name}");
        assert!(name.ends_with(".ptx"), "{name}");
    }

    #[test]
    fn hashing_is_specified_not_inherited_from_the_toolchain() {
        // Pins FNV-1a so a Rust upgrade cannot silently invalidate every
        // cached kernel on every user's machine.
        let mut hash = Fnv128::new();
        hash.write(b"a");
        assert_eq!(hash.finish(), 0xd228cb696f1a8caf78912b704e4a8964);
    }

    /// A scratch directory that removes itself, so these tests never touch the
    /// real cache and never depend on the process-global `cache_dir()`.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new(tag: &str) -> Self {
            let dir = std::env::temp_dir().join(format!(
                "onnx-genai-kernel-cache-{tag}-{}-{:x}",
                std::process::id(),
                temp_counter()
            ));
            std::fs::create_dir_all(&dir).expect("scratch cache directory");
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn a_stored_artifact_comes_back_verbatim() {
        let dir = TempDir::new("roundtrip");
        let includes: Vec<String> = Vec::new();
        let key = key("gemv", "src", &includes);

        assert_eq!(load_in(&dir.0, &key), None, "must miss before a store");
        store_in(&dir.0, &key, b".version 8.0\n");

        assert_eq!(
            load_in(&dir.0, &key).as_deref(),
            Some(&b".version 8.0\n"[..])
        );
    }

    #[test]
    fn a_different_source_does_not_hit_another_kernels_entry() {
        let dir = TempDir::new("distinct");
        let includes: Vec<String> = Vec::new();
        store_in(&dir.0, &key("gemv", "src", &includes), b"compiled");

        assert_eq!(load_in(&dir.0, &key("gemv", "other src", &includes)), None);
    }

    #[test]
    fn storing_leaves_no_temporary_files_behind() {
        // A leaked `.tmp` per compile would grow without bound in a directory
        // nothing ever prunes.
        let dir = TempDir::new("tempfiles");
        let includes: Vec<String> = Vec::new();
        store_in(&dir.0, &key("gemv", "src", &includes), b"compiled");

        let stray: Vec<_> = std::fs::read_dir(&dir.0)
            .expect("scratch dir")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".tmp"))
            .collect();
        assert!(stray.is_empty(), "{stray:?}");
    }

    #[test]
    fn an_empty_artifact_is_never_written_or_served() {
        // NVRTC returning nothing must not poison the cache with a file that
        // would load as an empty module forever after.
        let dir = TempDir::new("empty");
        let includes: Vec<String> = Vec::new();
        let key = key("gemv", "src", &includes);
        store_in(&dir.0, &key, b"");

        assert_eq!(load_in(&dir.0, &key), None);
        assert!(!dir.0.join(key.file_name()).exists());
    }
}
