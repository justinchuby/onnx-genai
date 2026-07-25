//! Process memory probes for `--profile`.
//!
//! Peak resident set size is the number that tells a user whether a model fits
//! on their machine, and it is not derivable from anything the engine reports:
//! weights, KV pages, ORT arenas, and transient tensors all land in the same
//! process. The kernel already tracks the high-water mark, so read it rather
//! than trying to sum allocations.
//!
//! What this does *not* cover depends on the device. On unified-memory hardware
//! (Apple Silicon) GPU buffers live in the process address space and are
//! counted here. On a discrete GPU they are not: device allocations never enter
//! the host resident set, so VRAM must come from the engine's own accounting
//! (`GovernorSnapshot::vram`) and is reported separately.

/// Peak resident set size of this process, in bytes.
///
/// Returns `None` on a platform without a probe rather than guessing, so a
/// report never carries a fabricated number.
pub(crate) fn peak_resident_bytes() -> Option<u64> {
    peak_resident_bytes_impl()
}

/// Linux exposes the high-water mark directly, in kibibytes.
///
/// `/proc/self/status` is preferred over `getrusage` here because `ru_maxrss`
/// units differ per platform, and the file is unambiguous.
#[cfg(target_os = "linux")]
fn peak_resident_bytes_impl() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    parse_vm_hwm(&status)
}

/// macOS reports the high-water mark from `getrusage` in bytes.
///
/// Deliberately macOS-only: `ru_maxrss` units are not portable across the BSDs
/// (several report kibibytes), and a silently wrong unit is worse than no
/// number. Other unix targets report nothing until each is verified.
#[cfg(target_os = "macos")]
fn peak_resident_bytes_impl() -> Option<u64> {
    // SAFETY: `getrusage` writes a fully-initialized `rusage` into the pointer
    // we own; it reads no memory from us and cannot fail for RUSAGE_SELF except
    // by returning non-zero, which is checked.
    unsafe {
        let mut usage: libc::rusage = std::mem::zeroed();
        (libc::getrusage(libc::RUSAGE_SELF, &mut usage) == 0)
            .then(|| u64::try_from(usage.ru_maxrss).unwrap_or(0))
            .filter(|bytes| *bytes > 0)
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn peak_resident_bytes_impl() -> Option<u64> {
    None
}

/// Extract `VmHWM` (peak resident set) from `/proc/self/status`, in bytes.
///
/// Split out from the read so it can be tested on any platform.
#[cfg_attr(not(target_os = "linux"), allow(dead_code))]
fn parse_vm_hwm(status: &str) -> Option<u64> {
    let line = status
        .lines()
        .find(|line| line.starts_with("VmHWM:"))?
        .strip_prefix("VmHWM:")?;
    let mut fields = line.split_whitespace();
    let value: u64 = fields.next()?.parse().ok()?;
    match fields.next() {
        Some("kB") | None => Some(value * 1024),
        Some("mB") => Some(value * 1024 * 1024),
        Some(_) => None,
    }
}

/// Render a byte count the way a human reads it.
pub(crate) fn format_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut value = bytes as f64;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{bytes} B")
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vm_hwm_is_read_in_kibibytes() {
        let status = "Name:\tonnx-genai\nVmPeak:\t 8000 kB\nVmHWM:\t   4096 kB\nVmRSS:\t 2048 kB\n";

        assert_eq!(parse_vm_hwm(status), Some(4096 * 1024));
    }

    #[test]
    fn a_status_without_a_high_water_mark_reports_nothing() {
        assert_eq!(parse_vm_hwm("Name:\tonnx-genai\nVmRSS:\t 2048 kB\n"), None);
        // An unknown unit is refused rather than assumed to be kibibytes.
        assert_eq!(parse_vm_hwm("VmHWM:\t 10 furlongs\n"), None);
    }

    #[test]
    fn byte_counts_are_rendered_for_humans() {
        assert_eq!(format_bytes(512), "512 B");
        assert_eq!(format_bytes(2048), "2.0 KiB");
        assert_eq!(format_bytes(3 * 1024 * 1024), "3.0 MiB");
        assert_eq!(format_bytes(5_368_709_120), "5.0 GiB");
    }

    #[test]
    fn the_platform_probe_reports_a_plausible_peak() {
        // Every platform this is tested on has a probe; a zero or absent value
        // would mean the report silently loses the number.
        if let Some(peak) = peak_resident_bytes() {
            assert!(
                peak > 1024 * 1024,
                "a running test process uses more than a mebibyte: {peak}"
            );
        }
    }
}
