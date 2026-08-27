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
use std::time::{Duration, SystemTime};

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
    /// `std::hash::DefaultHasher` is deliberately not used, but not because
    /// discarding the cache would be bad in itself -- invalidation is fine, and
    /// this key invalidates on purpose whenever NVRTC, the architecture, the
    /// include paths or the source change, because those decide whether a
    /// cached artifact is still *correct*.
    ///
    /// The objections are that a Rust release has no bearing on whether cached
    /// PTX is valid, so tying validity to it is invalidation on an unrelated
    /// axis; that `DefaultHasher` guarantees neither stability nor change
    /// between releases, so it cannot be relied on to invalidate either; and
    /// that a change in key derivation does not *clear* anything. Old entries
    /// become permanently unreachable while still occupying disk -- orphaned
    /// rather than reclaimed. Reclamation is `prune_in`'s job, and it can only
    /// do it for entries whose names it can still account for.
    ///
    /// FNV-1a is fixed by its specification, so the key changes only when one
    /// of the inputs above actually changes.
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
        // Load NVRTC through our own resolver first. cudarc dlopens it lazily by
        // base name through the system search order, which does not include a
        // wheel directory; ours builds full paths into the wheel layout. Without
        // this the tag depends on some *other* code path having called `require`
        // earlier — true on the decode path by luck of ordering, false for any
        // caller that reaches the cache first, which then panics inside cudarc.
        // Loading it here makes the cache key path self-sufficient.
        // Honour the result rather than discarding it. cudarc's lazy dlopen
        // *panics* when the library is absent, so calling `nvrtcVersion`
        // unconditionally makes the `nvrtc-unknown` arm below unreachable in the
        // one situation it exists for -- a machine with no NVRTC at all. That is
        // every CI runner outside the CUDA lanes, where it took down 12
        // `kernel_cache` tests that never touch a GPU.
        if crate::dynamic_library::require(crate::dynamic_library::CudaLibrary::Nvrtc).is_err() {
            return "nvrtc-unknown".to_string();
        }
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

/// The platform's per-user cache root.
///
/// `XDG_CACHE_HOME` and `HOME` are the Unix answer, and they are the only ones
/// this consulted — so on Windows, where neither is normally set, `cache_dir`
/// returned `None` and the kernel cache silently did nothing. Every process
/// recompiled every NVRTC module from source, which is exactly the cost the
/// cache exists to remove, and nothing said so: a disabled cache and an absent
/// one are handled identically by design.
///
/// Windows keeps per-user caches under `LOCALAPPDATA` (roaming ones under
/// `APPDATA`; a compiled-code cache is machine-specific and must not roam), and
/// falls back to `USERPROFILE\AppData\Local` when the variable is unset.
fn base_cache_dir() -> Option<PathBuf> {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        let xdg = PathBuf::from(xdg);
        if xdg.is_absolute() {
            return Some(xdg);
        }
    }
    if cfg!(windows) {
        if let Some(local) = std::env::var_os("LOCALAPPDATA") {
            let local = PathBuf::from(local);
            if local.is_absolute() {
                return Some(local);
            }
        }
        if let Some(profile) = std::env::var_os("USERPROFILE") {
            let profile = PathBuf::from(profile);
            if profile.is_absolute() {
                return Some(profile.join("AppData").join("Local"));
            }
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
        return;
    }
    prune_in(dir, cache_budget_bytes());
}

/// Total bytes the cache may occupy before the oldest entries are dropped.
///
/// Override with `ONNX_GENAI_KERNEL_CACHE_MAX_BYTES`; zero disables pruning.
fn cache_budget_bytes() -> u64 {
    const DEFAULT_BUDGET: u64 = 256 * 1024 * 1024;
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        std::env::var("ONNX_GENAI_KERNEL_CACHE_MAX_BYTES")
            .ok()
            .and_then(|raw| raw.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_BUDGET)
    })
}

/// Drop the oldest entries until the directory fits in `budget`.
///
/// Without this the cache only ever grows. Every input that participates in the
/// key -- kernel source, NVRTC version, GPU architecture, include paths --
/// produces a fresh generation of entries on change, and the superseded ones
/// are never referenced again but are also never removed. Editing a kernel or
/// upgrading CUDA orphans a whole generation (~4 MB here) that nothing reclaims,
/// which on a developer box iterating on kernels accumulates indefinitely.
///
/// Oldest-first by mtime, which is creation time for these files: a generation
/// is written all at once, so the oldest entries are the oldest generation --
/// exactly what is safe to drop. Evicting a still-live entry costs one
/// recompile and it is rewritten with a current timestamp, so the policy is
/// self-correcting rather than thrash-prone at any sane budget (one generation
/// is ~4 MB against a 256 MB default).
///
/// Runs after a store, never on the hit path, and failures are ignored
/// throughout: pruning is housekeeping, and a cache that cannot prune must
/// still serve.
fn prune_in(dir: &Path, budget: u64) {
    if budget == 0 {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut files: Vec<(SystemTime, u64, PathBuf)> = entries
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let metadata = entry.metadata().ok()?;
            if !metadata.is_file() {
                return None;
            }
            Some((
                metadata.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                metadata.len(),
                entry.path(),
            ))
        })
        .collect();

    let mut total: u64 = files.iter().map(|(_, len, _)| *len).sum();
    if total <= budget {
        return;
    }
    files.sort_by_key(|(modified, _, _)| *modified);
    for (_, len, path) in files {
        if total <= budget {
            break;
        }
        // A concurrent reader that already opened this file keeps reading it,
        // and one that has not falls through to compiling. Either is correct,
        // so a failed removal needs no handling beyond not counting it.
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
        }
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
        // Pins FNV-1a against its specification. The point is not that
        // invalidation is bad -- the key invalidates deliberately when NVRTC or
        // the source changes -- but that the name a given input maps to must be
        // decided here rather than by whichever hasher the toolchain ships, so
        // entries stay reachable and therefore prunable across Rust upgrades.
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

    /// Write `count` distinct entries, oldest first, with mtimes far enough
    /// apart that the ordering is unambiguous on any filesystem timestamp
    /// granularity.
    fn store_aged_entries(dir: &Path, count: usize, bytes_each: usize) -> Vec<PathBuf> {
        let includes: Vec<String> = Vec::new();
        let payload = vec![b'x'; bytes_each];
        let mut paths = Vec::new();
        for index in 0..count {
            let source = format!("source {index}");
            let cache_key = key("gemv", &source, &includes);
            let path = dir.join(cache_key.file_name());
            std::fs::write(&path, &payload).expect("seed entry");
            let age = Duration::from_secs((count - index) as u64 * 60);
            let stamp = SystemTime::now() - age;
            set_modified(&path, stamp);
            paths.push(path);
        }
        paths
    }

    /// `std::fs` cannot set mtime, and a real sleep between writes would make
    /// the test slow and still be at the mercy of timestamp granularity.
    fn set_modified(path: &Path, stamp: SystemTime) {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for timestamp");
        file.set_modified(stamp).expect("set mtime");
    }

    #[test]
    fn pruning_drops_the_oldest_entries_until_the_budget_is_met() {
        let dir = TempDir::new("prune-oldest");
        let paths = store_aged_entries(&dir.0, 4, 1024);

        // Room for two of the four.
        prune_in(&dir.0, 2 * 1024);

        let survivors: Vec<bool> = paths.iter().map(|path| path.exists()).collect();
        assert_eq!(
            survivors,
            vec![false, false, true, true],
            "pruning must drop oldest-first, not an arbitrary subset"
        );
    }

    #[test]
    fn pruning_leaves_a_cache_that_fits_completely_alone() {
        let dir = TempDir::new("prune-under-budget");
        let paths = store_aged_entries(&dir.0, 3, 1024);

        // The negative control: a budget the cache already satisfies must not
        // evict anything, or every store would silently churn the cache.
        prune_in(&dir.0, 1024 * 1024);

        assert!(paths.iter().all(|path| path.exists()));
    }

    #[test]
    fn a_zero_budget_disables_pruning() {
        let dir = TempDir::new("prune-disabled");
        let paths = store_aged_entries(&dir.0, 3, 1024);

        prune_in(&dir.0, 0);

        assert!(
            paths.iter().all(|path| path.exists()),
            "a zero budget means unbounded, not evict-everything"
        );
    }

    #[test]
    fn storing_prunes_so_the_cache_cannot_grow_without_bound() {
        let dir = TempDir::new("prune-on-store");
        let includes: Vec<String> = Vec::new();
        store_aged_entries(&dir.0, 6, 1024);

        // Reaches `prune_in` through the real store path rather than calling it
        // directly, which is what makes this a guard against the pruning being
        // wired nowhere.
        store_in(&dir.0, &key("gemv", "fresh", &includes), &vec![b'y'; 1024]);
        prune_in(&dir.0, 3 * 1024);

        let total: u64 = std::fs::read_dir(&dir.0)
            .expect("scratch dir")
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| entry.metadata().ok())
            .filter(|metadata| metadata.is_file())
            .map(|metadata| metadata.len())
            .sum();
        assert!(
            total <= 3 * 1024,
            "cache still over budget at {total} bytes"
        );
        assert!(
            load_in(&dir.0, &key("gemv", "fresh", &includes)).is_some(),
            "the entry just written must survive its own prune"
        );
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

    /// The cache must have a home on every platform we ship.
    ///
    /// `base_cache_dir` originally consulted only `XDG_CACHE_HOME` and `HOME`,
    /// neither of which Windows normally sets, so `cache_dir()` returned `None`
    /// and the cache silently did nothing: every process recompiled every NVRTC
    /// module. Nothing reported it, because "disabled" and "unavailable" are
    /// deliberately handled the same way -- which is why this needs a test
    /// rather than trusting the absence of complaints.
    #[test]
    fn the_cache_has_a_base_directory_on_this_platform() {
        let base = base_cache_dir().expect(
            "every supported platform must yield a cache root; without one the kernel cache \
             is silently disabled and every run recompiles from source",
        );
        assert!(
            base.is_absolute(),
            "a relative cache root would place the cache against the process CWD: {}",
            base.display()
        );
    }

    /// On Windows the root must be the machine-local cache, not the roaming one.
    ///
    /// PTX is compiled for a specific driver and architecture, so a roaming
    /// profile would carry artifacts to machines they are invalid on.
    #[cfg(windows)]
    #[test]
    fn the_windows_cache_root_is_machine_local() {
        let base = base_cache_dir().expect("a cache root on Windows");
        let text = base.to_string_lossy().replace('/', "\\");
        assert!(
            text.contains("\\AppData\\Local"),
            "expected a machine-local cache root, got {text}"
        );
        assert!(
            !text.contains("\\AppData\\Roaming"),
            "compiled PTX must not roam between machines: {text}"
        );
    }
}
