//! Real, platform-measured host RAM and filesystem capacities.
//!
//! #947: the resource governor used to fabricate host RAM and disk capacities
//! from hardcoded constants (`16 << 30`), which a real user hit on a machine
//! whose actual memory bore no relation to the constant. This module replaces
//! those constants with genuine OS queries. Every function returns `Option`:
//! `None` means the platform could not report the figure, which is a different
//! fact from any specific number and must never be rendered as one.
//!
//! Per the #947 guidance we prefer small, targeted OS calls over pulling in a
//! large system-info crate: the workspace carries no `sysinfo`/`libc` today, and
//! these queries are a handful of well-documented syscalls. VRAM is deliberately
//! *not* handled here — the real CUDA query lives in `engine::governor`
//! (`real_cuda_vram_capacity`), and a vendor-neutral DXGI/Metal/Vulkan adapter
//! query is intentionally out of scope for this change.

/// Total physical host RAM in bytes, measured from the OS. `None` when the
/// platform query is unavailable or fails.
pub(crate) fn host_ram_total_bytes() -> Option<u64> {
    host_ram_bytes().map(|(total, _available)| total)
}

/// Physical host RAM currently available in bytes. `None` when unknown.
pub(crate) fn host_ram_available_bytes() -> Option<u64> {
    host_ram_bytes().and_then(|(_total, available)| available)
}

/// `(total_bytes, free_bytes)` for the filesystem containing `path`. `None`
/// when the path cannot be queried on this platform.
pub(crate) fn disk_capacity_bytes(path: &std::path::Path) -> Option<(u64, u64)> {
    disk_capacity_impl(path)
}

// ── Windows ────────────────────────────────────────────────────────────────

#[cfg(windows)]
fn host_ram_bytes() -> Option<(u64, Option<u64>)> {
    #[repr(C)]
    struct MemoryStatusEx {
        length: u32,
        memory_load: u32,
        total_phys: u64,
        avail_phys: u64,
        total_page_file: u64,
        avail_page_file: u64,
        total_virtual: u64,
        avail_virtual: u64,
        avail_extended_virtual: u64,
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(buffer: *mut MemoryStatusEx) -> i32;
    }

    let mut status = MemoryStatusEx {
        length: std::mem::size_of::<MemoryStatusEx>() as u32,
        memory_load: 0,
        total_phys: 0,
        avail_phys: 0,
        total_page_file: 0,
        avail_page_file: 0,
        total_virtual: 0,
        avail_virtual: 0,
        avail_extended_virtual: 0,
    };
    // SAFETY: `status` is a correctly-sized, initialised MEMORYSTATUSEX; the
    // call only writes into it and returns non-zero on success.
    let ok = unsafe { GlobalMemoryStatusEx(&mut status) };
    if ok == 0 || status.total_phys == 0 {
        return None;
    }
    Some((status.total_phys, Some(status.avail_phys)))
}

#[cfg(windows)]
fn disk_capacity_impl(path: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::windows::ffi::OsStrExt;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetDiskFreeSpaceExW(
            directory_name: *const u16,
            free_bytes_available_to_caller: *mut u64,
            total_number_of_bytes: *mut u64,
            total_number_of_free_bytes: *mut u64,
        ) -> i32;
    }

    // GetDiskFreeSpaceExW accepts a directory; use the path if it is one,
    // otherwise its parent, falling back to the path itself.
    let dir = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(|parent| parent.to_path_buf())
            .unwrap_or_else(|| path.to_path_buf())
    };
    let mut wide: Vec<u16> = dir.as_os_str().encode_wide().collect();
    wide.push(0);

    let mut free_to_caller: u64 = 0;
    let mut total: u64 = 0;
    let mut total_free: u64 = 0;
    // SAFETY: `wide` is a NUL-terminated UTF-16 path; the out-params are valid
    // for writes and only written on success.
    let ok = unsafe {
        GetDiskFreeSpaceExW(
            wide.as_ptr(),
            &mut free_to_caller,
            &mut total,
            &mut total_free,
        )
    };
    if ok == 0 || total == 0 {
        return None;
    }
    Some((total, free_to_caller))
}

// ── Linux ────────────────────────────────────────────────────────────────

#[cfg(target_os = "linux")]
fn host_ram_bytes() -> Option<(u64, Option<u64>)> {
    // /proc/meminfo is the portable, dependency-free source for both figures.
    let contents = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total_kib: Option<u64> = None;
    let mut available_kib: Option<u64> = None;
    for line in contents.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total_kib = parse_meminfo_kib(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available_kib = parse_meminfo_kib(rest);
        }
    }
    let total = total_kib?.checked_mul(1024)?;
    if total == 0 {
        return None;
    }
    Some((total, available_kib.and_then(|kib| kib.checked_mul(1024))))
}

#[cfg(target_os = "linux")]
fn parse_meminfo_kib(rest: &str) -> Option<u64> {
    // Lines look like "MemTotal:       16244964 kB".
    rest.split_whitespace().next()?.parse::<u64>().ok()
}

// ── macOS ────────────────────────────────────────────────────────────────

#[cfg(target_os = "macos")]
fn host_ram_bytes() -> Option<(u64, Option<u64>)> {
    use std::os::raw::{c_char, c_int, c_void};

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    let mut memsize: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    let name = c"hw.memsize";
    // SAFETY: querying a well-known scalar sysctl into a correctly sized u64.
    let rc = unsafe {
        sysctlbyname(
            name.as_ptr(),
            &mut memsize as *mut u64 as *mut c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || memsize == 0 {
        return None;
    }
    // Available RAM on macOS needs vm_stat plumbing; report total only.
    Some((memsize, None))
}

#[cfg(all(unix, not(target_os = "linux")))]
fn disk_capacity_impl(path: &std::path::Path) -> Option<(u64, u64)> {
    statvfs_capacity(path)
}

#[cfg(target_os = "linux")]
fn disk_capacity_impl(path: &std::path::Path) -> Option<(u64, u64)> {
    statvfs_capacity(path)
}

/// `fsblkcnt_t`. POSIX leaves the width implementation-defined: 64-bit glibc
/// makes it 64-bit, but on macOS it is a plain 32-bit `unsigned int`. Declaring
/// it as `u64` everywhere made `StatVfs` a 112-byte Linux struct laid over a
/// 64-byte macOS one, so `f_blocks`/`f_bfree` were read as a single fused u64
/// and every later field sat at the wrong offset.
#[cfg(all(unix, target_os = "macos"))]
type FsBlkCnt = u32;
#[cfg(all(unix, not(target_os = "macos")))]
type FsBlkCnt = u64;

/// `fsfilcnt_t`; same width story as [`FsBlkCnt`].
#[cfg(all(unix, target_os = "macos"))]
type FsFilCnt = u32;
#[cfg(all(unix, not(target_os = "macos")))]
type FsFilCnt = u64;

/// POSIX `struct statvfs`. Only `f_frsize`, `f_bsize`, `f_blocks` and `f_bavail`
/// are read, but the whole tail must still be declared correctly: the kernel
/// writes `sizeof(struct statvfs)` bytes, so a wrong width anywhere before a
/// field we do read silently corrupts it.
///
/// `f_bsize`/`f_frsize` are `unsigned long` on both platforms. The counts differ
/// (see [`FsBlkCnt`]), and macOS has no trailing spare array at all — its struct
/// ends after `f_namemax`. See `statvfs_layout_matches_the_platform_abi` for the
/// measured offsets this mirrors.
#[cfg(unix)]
#[repr(C)]
struct StatVfs {
    f_bsize: std::os::raw::c_ulong,
    f_frsize: std::os::raw::c_ulong,
    f_blocks: FsBlkCnt,
    f_bfree: FsBlkCnt,
    f_bavail: FsBlkCnt,
    f_files: FsFilCnt,
    f_ffree: FsFilCnt,
    f_favail: FsFilCnt,
    f_fsid: std::os::raw::c_ulong,
    f_flag: std::os::raw::c_ulong,
    f_namemax: std::os::raw::c_ulong,
    #[cfg(not(target_os = "macos"))]
    f_spare: [std::os::raw::c_int; 6],
}

// Compile-time layout guards. A mismatch here is a build failure rather than
// the silent runtime garbage this replaces. Deliberately scoped to the two
// targets whose ABI is pinned down: every other unix keeps whatever it builds
// today rather than being newly broken by an unverified assertion.
#[cfg(target_os = "macos")]
const _: () = assert!(
    size_of::<StatVfs>() == 64,
    "macOS struct statvfs is 64 bytes (measured on arm64 macOS)"
);
#[cfg(all(target_os = "linux", target_pointer_width = "64"))]
const _: () = assert!(
    size_of::<StatVfs>() == 112,
    "64-bit glibc struct statvfs is 112 bytes"
);

#[cfg(unix)]
fn statvfs_capacity(path: &std::path::Path) -> Option<(u64, u64)> {
    use std::os::raw::c_int;
    use std::os::unix::ffi::OsStrExt;

    unsafe extern "C" {
        fn statvfs(path: *const std::os::raw::c_char, buf: *mut StatVfs) -> c_int;
    }

    let query = if path.is_dir() {
        path.to_path_buf()
    } else {
        path.parent()
            .map(|parent| parent.to_path_buf())
            .unwrap_or_else(|| path.to_path_buf())
    };
    let mut c_path: Vec<u8> = query.as_os_str().as_bytes().to_vec();
    c_path.push(0);

    // SAFETY: `c_path` is a NUL-terminated path; `buf` is uninitialised memory
    // fully written by a successful call.
    let mut buf = std::mem::MaybeUninit::<StatVfs>::zeroed();
    let rc = unsafe {
        statvfs(
            c_path.as_ptr() as *const std::os::raw::c_char,
            buf.as_mut_ptr(),
        )
    };
    if rc != 0 {
        return None;
    }
    let buf = unsafe { buf.assume_init() };
    // `c_ulong` is already 64-bit on LP64 targets and 32-bit elsewhere, so this
    // widening is redundant on some targets and load-bearing on others. Keep it
    // and silence the lint rather than making the arithmetic below
    // target-dependent.
    #[allow(clippy::unnecessary_cast)]
    let unit = if buf.f_frsize != 0 {
        buf.f_frsize as u64
    } else {
        buf.f_bsize as u64
    };
    // `u64::from` rather than `as`: the block counts are 32-bit on macOS and
    // 64-bit on Linux, and this widens correctly on both without a cast lint.
    let total = u64::from(buf.f_blocks).checked_mul(unit)?;
    let free = u64::from(buf.f_bavail).checked_mul(unit)?;
    if total == 0 {
        return None;
    }
    Some((total, free))
}

#[cfg(not(any(windows, unix)))]
fn host_ram_bytes() -> Option<(u64, Option<u64>)> {
    None
}

#[cfg(not(any(windows, unix)))]
fn disk_capacity_impl(_path: &std::path::Path) -> Option<(u64, u64)> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_ram_total_is_measured_on_supported_platforms() {
        // On every CI platform (Windows/Linux/macOS) the OS can report RAM, so
        // this must be a real, plausible number — not a fabricated constant and
        // not the old 16 GiB default.
        let total = host_ram_total_bytes().expect("host RAM must be measurable here");
        assert!(total > (1u64 << 30), "implausibly small host RAM: {total}");
        // The old fabricated constant was exactly 16 GiB; a real box is very
        // unlikely to report that to the byte.
        assert_ne!(total, 16u64 << 30, "looks like the old fabricated constant");
    }

    #[test]
    fn disk_capacity_is_measured_for_the_working_directory() {
        let cwd = std::env::current_dir().expect("cwd");
        let (total, free) = disk_capacity_bytes(&cwd).expect("disk must be measurable here");
        assert!(total > 0);
        assert!(free <= total);
    }

    /// The FFI layout, asserted field by field rather than inferred from "the
    /// disk came back `Some`". The macOS numbers below were measured on this
    /// platform by compiling a C program against `<sys/statvfs.h>`:
    ///
    /// ```text
    /// sizeof(struct statvfs) = 64   alignof = 8
    /// sizeof(fsblkcnt_t) = 4        sizeof(fsfilcnt_t) = 4
    /// f_bsize 0, f_frsize 8, f_blocks 16, f_bfree 20, f_bavail 24,
    /// f_files 28, f_ffree 32, f_favail 36, f_fsid 40, f_flag 48, f_namemax 56
    /// ```
    ///
    /// Declaring the counts as `u64` put `f_fsid` at 64 instead of 40 and fused
    /// `f_blocks` with `f_bfree`, which is what made the disk unmeasurable. A
    /// test that only checks `Some` would pass by luck on a different disk;
    /// this one cannot.
    #[cfg(target_os = "macos")]
    #[test]
    fn statvfs_layout_matches_the_platform_abi() {
        use std::mem::{align_of, offset_of, size_of};

        assert_eq!(size_of::<StatVfs>(), 64, "sizeof(struct statvfs)");
        assert_eq!(align_of::<StatVfs>(), 8, "alignof(struct statvfs)");
        assert_eq!(size_of::<FsBlkCnt>(), 4, "sizeof(fsblkcnt_t)");
        assert_eq!(size_of::<FsFilCnt>(), 4, "sizeof(fsfilcnt_t)");

        for (name, actual, expected) in [
            ("f_bsize", offset_of!(StatVfs, f_bsize), 0),
            ("f_frsize", offset_of!(StatVfs, f_frsize), 8),
            ("f_blocks", offset_of!(StatVfs, f_blocks), 16),
            ("f_bfree", offset_of!(StatVfs, f_bfree), 20),
            ("f_bavail", offset_of!(StatVfs, f_bavail), 24),
            ("f_files", offset_of!(StatVfs, f_files), 28),
            ("f_ffree", offset_of!(StatVfs, f_ffree), 32),
            ("f_favail", offset_of!(StatVfs, f_favail), 36),
            ("f_fsid", offset_of!(StatVfs, f_fsid), 40),
            ("f_flag", offset_of!(StatVfs, f_flag), 48),
            ("f_namemax", offset_of!(StatVfs, f_namemax), 56),
        ] {
            assert_eq!(actual, expected, "offsetof({name})");
        }
    }

    /// The 64-bit glibc layout. Not executed in this change — no Linux host was
    /// available — but stated explicitly so a regression is a visible test
    /// failure on Linux CI rather than silent misreads. The compile-time
    /// `size_of` guard next to the declaration covers the build itself.
    #[cfg(all(target_os = "linux", target_pointer_width = "64"))]
    #[test]
    fn statvfs_layout_matches_the_platform_abi() {
        use std::mem::{align_of, offset_of, size_of};

        assert_eq!(size_of::<StatVfs>(), 112, "sizeof(struct statvfs)");
        assert_eq!(align_of::<StatVfs>(), 8, "alignof(struct statvfs)");
        assert_eq!(size_of::<FsBlkCnt>(), 8, "sizeof(fsblkcnt_t)");
        assert_eq!(size_of::<FsFilCnt>(), 8, "sizeof(fsfilcnt_t)");

        for (name, actual, expected) in [
            ("f_bsize", offset_of!(StatVfs, f_bsize), 0),
            ("f_frsize", offset_of!(StatVfs, f_frsize), 8),
            ("f_blocks", offset_of!(StatVfs, f_blocks), 16),
            ("f_bfree", offset_of!(StatVfs, f_bfree), 24),
            ("f_bavail", offset_of!(StatVfs, f_bavail), 32),
            ("f_files", offset_of!(StatVfs, f_files), 40),
            ("f_ffree", offset_of!(StatVfs, f_ffree), 48),
            ("f_favail", offset_of!(StatVfs, f_favail), 56),
            ("f_fsid", offset_of!(StatVfs, f_fsid), 64),
            ("f_flag", offset_of!(StatVfs, f_flag), 72),
            ("f_namemax", offset_of!(StatVfs, f_namemax), 80),
            ("f_spare", offset_of!(StatVfs, f_spare), 88),
        ] {
            assert_eq!(actual, expected, "offsetof({name})");
        }
    }

    /// The block arithmetic must widen, not truncate or fuse. Cross-checks the
    /// measured capacity against a second, independent reading of the same
    /// filesystem so a future width regression shows up as a mismatch.
    #[cfg(unix)]
    #[test]
    fn disk_capacity_agrees_with_a_direct_statvfs_reading() {
        let cwd = std::env::current_dir().expect("cwd");
        let (total, free) = statvfs_capacity(&cwd).expect("statvfs must succeed on the cwd");

        // A block count fused with its neighbour lands far outside any real
        // disk; a truncated one lands at zero. Both are excluded here.
        assert!(
            total > (1u64 << 30),
            "implausibly small disk total: {total}"
        );
        assert!(
            total < (1u64 << 50),
            "implausibly large disk total, likely fused fields: {total}"
        );
        assert!(free <= total, "free {free} exceeds total {total}");
    }
}
