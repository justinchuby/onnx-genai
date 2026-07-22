//! NUMA-aware CPU affinity for the bounded M=1 decode thread pool.
//!
//! Decode streams the full int4 weight set every token, so its throughput is
//! dominated by memory bandwidth and by the fork-join barrier that closes each
//! of the ~141 per-token `MatMulNBits` projections. When the decode workers are
//! free to float across both sockets of a multi-node host, every barrier pays a
//! cross-socket cache-coherency round trip and the streamed weights land on
//! whatever node first touched them. Pinning the workers to the CPUs of a single
//! NUMA node keeps both the barrier traffic and the weight reads node-local.
//!
//! Topology is *queried* from the running machine (`/sys/devices/system/node`
//! on Linux), never hardcoded, and the behaviour is selected through an explicit
//! environment switch so it stays inspectable.

use std::collections::BTreeMap;

/// Selects how the decode pool binds its workers to CPUs.
///
/// * unset / `off` -- no pinning; workers float on the OS scheduler (default).
/// * `compact` -- pin the workers, one per CPU, to the CPUs of a single NUMA
///   node (the smallest-index node whose CPU count covers the pool size, so all
///   workers, their fork-join barriers, and their weight reads stay node-local).
/// * `node:<index>` -- pin the workers to the CPUs of the named NUMA node.
pub const DECODE_AFFINITY_ENV: &str = "ONNX_GENAI_CPU_DECODE_AFFINITY";

/// A parsed [`DECODE_AFFINITY_ENV`] request.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DecodeAffinity {
    /// Leave the workers unpinned.
    Off,
    /// Pin the pool to a single node chosen to cover the worker count.
    Compact,
    /// Pin the pool to the CPUs of the named NUMA node.
    Node(usize),
}

impl DecodeAffinity {
    /// Parse the affinity request, returning a clear error for malformed input.
    pub fn parse(raw: Option<&str>) -> std::result::Result<Self, String> {
        let Some(raw) = raw else {
            return Ok(Self::Off);
        };
        let trimmed = raw.trim();
        match trimmed {
            "" | "off" | "0" => Ok(Self::Off),
            "compact" => Ok(Self::Compact),
            other => {
                if let Some(index) = other.strip_prefix("node:") {
                    index
                        .trim()
                        .parse::<usize>()
                        .map(Self::Node)
                        .map_err(|_| {
                            format!(
                                "{DECODE_AFFINITY_ENV}=`{raw}` is not a valid NUMA node selector; \
                                 expected `node:<index>` with a non-negative integer index"
                            )
                        })
                } else {
                    Err(format!(
                        "{DECODE_AFFINITY_ENV}=`{raw}` is not a recognized affinity mode; \
                         expected `off`, `compact`, or `node:<index>`"
                    ))
                }
            }
        }
    }

    /// Parse the affinity request from the process environment.
    pub fn from_env() -> std::result::Result<Self, String> {
        Self::parse(std::env::var(DECODE_AFFINITY_ENV).ok().as_deref())
    }
}

/// The CPU membership of the host's NUMA nodes, keyed by node index.
///
/// Queried from the operating system so no socket or core counts are baked in.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NumaTopology {
    nodes: BTreeMap<usize, Vec<usize>>,
}

impl NumaTopology {
    /// Detect the NUMA topology of the running host, returning `None` when the
    /// platform exposes no node information (e.g. non-Linux or a single node).
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(not(target_os = "linux"))]
        {
            None
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Option<Self> {
        let mut nodes = BTreeMap::new();
        let entries = std::fs::read_dir("/sys/devices/system/node").ok()?;
        for entry in entries.flatten() {
            let file_name = entry.file_name();
            let name = file_name.to_str()?;
            let Some(index) = name.strip_prefix("node") else {
                continue;
            };
            let Ok(index) = index.parse::<usize>() else {
                continue;
            };
            let cpulist = entry.path().join("cpulist");
            let Ok(contents) = std::fs::read_to_string(&cpulist) else {
                continue;
            };
            let cpus = parse_cpu_list(&contents);
            if !cpus.is_empty() {
                nodes.insert(index, cpus);
            }
        }
        (nodes.len() > 1).then_some(Self { nodes })
    }

    /// The number of NUMA nodes discovered.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The CPUs belonging to `node`, if it exists.
    pub fn cpus_for_node(&self, node: usize) -> Option<&[usize]> {
        self.nodes.get(&node).map(Vec::as_slice)
    }

    /// Choose the CPU set to pin `worker_count` decode workers to.
    ///
    /// For [`DecodeAffinity::Compact`], pick the smallest-index node whose CPU
    /// count covers the worker count so every worker stays on one node; if no
    /// single node is large enough, fall back to the largest node. For
    /// [`DecodeAffinity::Node`], return that node's CPUs (an unknown index is a
    /// clear error). Returns `None` for [`DecodeAffinity::Off`].
    pub fn cpus_for(
        &self,
        affinity: &DecodeAffinity,
        worker_count: usize,
    ) -> std::result::Result<Option<Vec<usize>>, String> {
        match affinity {
            DecodeAffinity::Off => Ok(None),
            DecodeAffinity::Node(index) => match self.cpus_for_node(*index) {
                Some(cpus) => Ok(Some(cpus.to_vec())),
                None => {
                    let available = self
                        .nodes
                        .keys()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", ");
                    Err(format!(
                        "{DECODE_AFFINITY_ENV}=`node:{index}` names an unknown NUMA node; \
                         available nodes are [{available}]"
                    ))
                }
            },
            DecodeAffinity::Compact => {
                let fitting = self
                    .nodes
                    .values()
                    .filter(|cpus| cpus.len() >= worker_count)
                    .min_by_key(|cpus| cpus.len());
                let chosen = fitting.or_else(|| self.nodes.values().max_by_key(|cpus| cpus.len()));
                Ok(chosen.map(|cpus| cpus.to_vec()))
            }
        }
    }
}

/// Parse a Linux `cpulist` string such as `"0-3,8,10-11"` into CPU indices.
fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) = (start.trim().parse::<usize>(), end.trim().parse::<usize>())
            {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Pin the calling thread to a single CPU. Best-effort: a failure to set
/// affinity (e.g. a restricted cgroup) is reported so the caller can log it,
/// but never aborts decode.
#[cfg(target_os = "linux")]
pub fn pin_current_thread_to_cpu(cpu: usize) -> std::result::Result<(), String> {
    // SAFETY: `cpu_set_t` is a plain bitset; we zero it, set one valid bit, and
    // hand a correctly sized pointer to `sched_setaffinity` for the current
    // thread (`pid == 0`). No Rust invariants are involved.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(cpu, &mut set);
        let result = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if result == 0 {
            Ok(())
        } else {
            Err(format!(
                "sched_setaffinity(cpu={cpu}) failed: {}",
                std::io::Error::last_os_error()
            ))
        }
    }
}

#[cfg(not(target_os = "linux"))]
pub fn pin_current_thread_to_cpu(_cpu: usize) -> std::result::Result<(), String> {
    Err("thread affinity is only implemented on Linux".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_affinity_modes() {
        assert_eq!(DecodeAffinity::parse(None).unwrap(), DecodeAffinity::Off);
        assert_eq!(
            DecodeAffinity::parse(Some("off")).unwrap(),
            DecodeAffinity::Off
        );
        assert_eq!(
            DecodeAffinity::parse(Some("compact")).unwrap(),
            DecodeAffinity::Compact
        );
        assert_eq!(
            DecodeAffinity::parse(Some("node:1")).unwrap(),
            DecodeAffinity::Node(1)
        );
        assert!(DecodeAffinity::parse(Some("node:x")).is_err());
        assert!(DecodeAffinity::parse(Some("bogus")).is_err());
    }

    #[test]
    fn parses_cpu_lists() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0-3,8,10-11"), vec![0, 1, 2, 3, 8, 10, 11]);
        assert_eq!(parse_cpu_list(" 5 "), vec![5]);
        assert!(parse_cpu_list("").is_empty());
    }

    #[test]
    fn compact_picks_smallest_covering_node() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..48).collect::<Vec<_>>());
        nodes.insert(1, (48..96).collect::<Vec<_>>());
        let topology = NumaTopology { nodes };
        let cpus = topology
            .cpus_for(&DecodeAffinity::Compact, 32)
            .unwrap()
            .unwrap();
        assert_eq!(cpus.len(), 48);
        assert_eq!(cpus[0], 0);
    }

    #[test]
    fn node_selector_reports_unknown_node() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, vec![0, 1]);
        nodes.insert(1, vec![2, 3]);
        let topology = NumaTopology { nodes };
        assert!(
            topology
                .cpus_for(&DecodeAffinity::Node(7), 2)
                .unwrap_err()
                .contains("unknown NUMA node")
        );
        let cpus = topology
            .cpus_for(&DecodeAffinity::Node(1), 2)
            .unwrap()
            .unwrap();
        assert_eq!(cpus, vec![2, 3]);
    }
}
