//! Calibration (`docs/architecture/ORT2.md` §6.4): fill a
//! [`PlacementCostModel`]'s rates from **hardware probes**, never from literals.
//!
//! The two probes that feed this live in `onnx-genai-bench`:
//!   * `roofline_bandwidth` — host DRAM read bandwidth vs. thread count, which
//!     becomes a device's parallelism-qualified [`MemoryBandwidth`].
//!   * `roofline_transfer` — host<->device H2D/D2H bandwidth for pinned and
//!     pageable host memory plus a fitted `latency_base`, which becomes the
//!     [`TransferProfile`]s in the transfer matrix.
//!
//! Both structured ingestion (from typed samples) and text ingestion (parsing
//! the probes' CSV output directly) are provided, so the calibration path
//! literally consumes what the probes emit. Anything a probe did not measure
//! stays unknown; nothing is defaulted to a plausible number.

use crate::device::DeviceKey;
use crate::model::PlacementCostModel;
use crate::profile::MemoryBandwidth;
use crate::transfer::TransferProfile;
use std::time::Duration;

/// Direction of a host<->device transfer measurement.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransferDirection {
    /// Host to device (upload).
    HostToDevice,
    /// Device to host (download).
    DeviceToHost,
}

/// One fitted row of `roofline_transfer` output: a direction × host-memory-kind
/// measurement with its fitted `latency_base` and bandwidth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransferFit {
    /// Transfer direction.
    pub direction: TransferDirection,
    /// Whether the host buffer was pinned (`true`) or pageable (`false`).
    pub pinned: bool,
    /// Fitted fixed latency (microseconds).
    pub latency_base_us: f64,
    /// Fitted sustained bandwidth (GB/s, i.e. 1e9 bytes/sec).
    pub bandwidth_gb_s: f64,
}

/// Ingest a memory-bandwidth curve from `roofline_bandwidth`-style
/// `(threads, gb_s)` samples and apply it to a device profile.
///
/// GB/s are converted to bytes/sec (`* 1e9`, matching the probe's decimal-giga
/// convention). The device profile is created if absent; its name is set only
/// if provided non-empty so repeated calibration does not clobber a good name.
pub fn apply_memory_bandwidth(
    model: &mut PlacementCostModel,
    device: DeviceKey,
    name: &str,
    samples: impl IntoIterator<Item = (u32, f64)>,
) {
    let profile = model.device_profile_mut(device);
    if !name.is_empty() {
        profile.name = name.to_string();
    }
    profile.memory_bandwidth =
        MemoryBandwidth::from_samples(samples.into_iter().map(|(t, gb_s)| (t, gb_s * 1.0e9)));
}

/// Ingest fitted host<->device transfer measurements and apply them to the
/// transfer matrix for the `host`<->`device` pair.
///
/// Fits are grouped into H2D (`host → device`) and D2H (`device → host`)
/// profiles, merging the pinned and pageable bandwidths for each direction.
/// `latency_base` is taken as the min over the direction's fits (the pinned fit
/// is the cleaner latency estimate). Unmeasured host-memory kinds stay `None`.
pub fn apply_transfer_fits(
    model: &mut PlacementCostModel,
    host: DeviceKey,
    device: DeviceKey,
    fits: &[TransferFit],
    is_async_capable: bool,
) {
    let build = |direction: TransferDirection| -> Option<TransferProfile> {
        let matching: Vec<&TransferFit> =
            fits.iter().filter(|f| f.direction == direction).collect();
        if matching.is_empty() {
            return None;
        }
        let mut profile = TransferProfile {
            is_async_capable,
            ..TransferProfile::unknown()
        };
        let mut min_latency_us: Option<f64> = None;
        for f in matching {
            let bw = (f.bandwidth_gb_s * 1.0e9).max(0.0);
            let bw = (bw > 0.0).then_some(bw);
            if f.pinned {
                profile.pinned_bandwidth = bw;
            } else {
                profile.pageable_bandwidth = bw;
            }
            if f.latency_base_us.is_finite() && f.latency_base_us >= 0.0 {
                min_latency_us =
                    Some(min_latency_us.map_or(f.latency_base_us, |m| m.min(f.latency_base_us)));
            }
        }
        profile.latency_base = min_latency_us.map(Duration::from_secs_f64_us);
        Some(profile)
    };

    if let Some(h2d) = build(TransferDirection::HostToDevice) {
        model.set_transfer_profile(host.clone(), device.clone(), h2d);
    }
    if let Some(d2h) = build(TransferDirection::DeviceToHost) {
        model.set_transfer_profile(device, host, d2h);
    }
}

/// Small extension so microsecond latencies read clearly at the call site.
trait DurationFromUs {
    fn from_secs_f64_us(us: f64) -> Duration;
}
impl DurationFromUs for Duration {
    fn from_secs_f64_us(us: f64) -> Duration {
        Duration::from_secs_f64((us / 1.0e6).max(0.0))
    }
}

/// Parse the `threads,gb_s,...` data rows of `roofline_bandwidth` output into
/// `(threads, gb_s)` samples, ignoring the header and any non-data lines.
pub fn parse_memory_bandwidth_csv(text: &str) -> Vec<(u32, f64)> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') || line.starts_with("roofline") {
            continue;
        }
        let mut cols = line.split(',');
        let (Some(a), Some(b)) = (cols.next(), cols.next()) else {
            continue;
        };
        if let (Ok(threads), Ok(gb_s)) = (a.trim().parse::<u32>(), b.trim().parse::<f64>()) {
            out.push((threads, gb_s));
        }
    }
    out
}

/// Parse the fitted `TransferProfile` block of `roofline_transfer` output
/// (`direction,host,latency_base_us,fit_bandwidth_gb_s,peak_sustained_gb_s`)
/// into [`TransferFit`]s. Header, comment, and raw-sweep rows are ignored.
pub fn parse_transfer_fits_csv(text: &str) -> Vec<TransferFit> {
    let mut out = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        let cols: Vec<&str> = line.split(',').map(str::trim).collect();
        if cols.len() < 4 {
            continue;
        }
        let direction = match cols[0] {
            "h2d" => TransferDirection::HostToDevice,
            "d2h" => TransferDirection::DeviceToHost,
            _ => continue,
        };
        let pinned = match cols[1] {
            "pinned" => true,
            "pageable" => false,
            _ => continue,
        };
        // Distinguish the fit block from the raw sweep: the fit block's third
        // column is latency_base_us and fourth is a GB/s bandwidth. The raw
        // sweep's third column is a byte count (a large integer). Require both
        // remaining columns to parse as floats, and reject the raw sweep by
        // checking the row is the 5-column fit shape.
        if cols.len() != 5 {
            continue;
        }
        let (Ok(latency_base_us), Ok(bandwidth_gb_s)) =
            (cols[2].parse::<f64>(), cols[3].parse::<f64>())
        else {
            continue;
        };
        // Raw-sweep rows carry the byte size in col 2 (>= 1024) and a per-copy
        // time in col 6; the fit block's latency is a small microsecond value.
        // The 5-vs-7 column count already separates them, but guard anyway.
        out.push(TransferFit {
            direction,
            pinned,
            latency_base_us,
            bandwidth_gb_s,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transfer::HostMemoryKind;

    #[test]
    fn memory_bandwidth_from_probe_samples() {
        let mut model = PlacementCostModel::new();
        apply_memory_bandwidth(
            &mut model,
            DeviceKey::cpu(),
            "host",
            [(1, 10.07), (4, 26.50), (8, 37.46), (20, 49.28)],
        );
        let p = model.device_profile(&DeviceKey::cpu()).unwrap();
        assert_eq!(p.name, "host");
        assert_eq!(p.memory_bandwidth.single_thread(), Some(10.07e9));
        assert_eq!(p.memory_bandwidth.peak(), Some(49.28e9));
    }

    #[test]
    fn transfer_fits_populate_both_directions() {
        let mut model = PlacementCostModel::new();
        let fits = [
            TransferFit {
                direction: TransferDirection::HostToDevice,
                pinned: true,
                latency_base_us: 12.0,
                bandwidth_gb_s: 11.7,
            },
            TransferFit {
                direction: TransferDirection::HostToDevice,
                pinned: false,
                latency_base_us: 8.3,
                bandwidth_gb_s: 6.1,
            },
            TransferFit {
                direction: TransferDirection::DeviceToHost,
                pinned: true,
                latency_base_us: 7.8,
                bandwidth_gb_s: 12.0,
            },
        ];
        apply_transfer_fits(
            &mut model,
            DeviceKey::cpu(),
            DeviceKey::cuda(0),
            &fits,
            true,
        );
        let h2d = model
            .transfer_matrix()
            .get(&DeviceKey::cpu(), &DeviceKey::cuda(0))
            .unwrap();
        assert_eq!(h2d.bandwidth(HostMemoryKind::Pinned), Some(11.7e9));
        assert_eq!(h2d.bandwidth(HostMemoryKind::Pageable), Some(6.1e9));
        // latency = min(12.0, 8.3) us.
        assert_eq!(h2d.latency_base, Some(Duration::from_secs_f64(8.3e-6)));
        let d2h = model
            .transfer_matrix()
            .get(&DeviceKey::cuda(0), &DeviceKey::cpu())
            .unwrap();
        assert_eq!(d2h.bandwidth(HostMemoryKind::Pinned), Some(12.0e9));
        // D2H pageable was not measured — stays unknown.
        assert_eq!(d2h.bandwidth(HostMemoryKind::Pageable), None);
    }

    #[test]
    fn parses_memory_bandwidth_csv() {
        let text = "roofline_bandwidth: buffer=1024 MiB\nthreads,gb_s,elapsed_s\n1,10.070,3.0\n4,26.500,3.0\n20,49.280,3.0\n";
        let samples = parse_memory_bandwidth_csv(text);
        assert_eq!(samples, vec![(1, 10.07), (4, 26.50), (20, 49.28)]);
    }

    #[test]
    fn parses_transfer_fit_block_and_ignores_sweep() {
        let text = "\
direction,host,bytes,gb_s_median,gb_s_min,gb_s_max,per_copy_us
h2d,pinned,262144,9.526,5.883,10.422,27.520

# Fitted TransferProfile (time = latency_base + bytes / bandwidth)
direction,host,latency_base_us,fit_bandwidth_gb_s,peak_sustained_gb_s
h2d,pageable,8.294,6.075,6.108
h2d,pinned,11.961,11.737,11.737
d2h,pinned,7.830,11.988,12.003
";
        let fits = parse_transfer_fits_csv(text);
        assert_eq!(fits.len(), 3, "{fits:?}");
        assert!(fits.iter().all(|f| f.bandwidth_gb_s < 20.0));
    }
}
