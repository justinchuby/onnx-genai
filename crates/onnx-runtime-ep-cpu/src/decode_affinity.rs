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

/// The complete set of accepted affinity modes, named in every diagnostic so a
/// rejected value always sees the full menu of valid options.
const ACCEPTED_MODES: &str = "`off`, `compact`, `node:<index>`";

/// Render the discovered available-node list for a diagnostic, or state plainly
/// that topology is unavailable, so every invalid value reports the same three
/// facts: what was rejected, what is accepted, and what nodes actually exist.
fn available_nodes_clause(topology: Option<&NumaTopology>) -> String {
    match topology {
        Some(topology) if !topology.nodes.is_empty() => {
            let list = topology
                .nodes
                .keys()
                .map(usize::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("available NUMA nodes are [{list}]")
        }
        _ => "NUMA topology is unavailable on this host (single NUMA node or non-Linux), \
              so no node selector can be honored"
            .to_string(),
    }
}

/// Build the single, consistent diagnostic used for every invalid affinity
/// value: it always names the rejected value, all accepted modes, and the
/// discovered available-node list (or an explicit topology-unavailable note).
fn invalid_selector_error(value: &str, topology: Option<&NumaTopology>) -> String {
    format!(
        "{DECODE_AFFINITY_ENV}=`{value}` is not a usable decode-affinity selector; \
         accepted modes are {ACCEPTED_MODES}; {clause}",
        clause = available_nodes_clause(topology),
    )
}

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

    /// Parse the affinity request from the process environment, validating it
    /// against the host's discovered NUMA topology so every invalid value gets a
    /// single, consistent, actionable diagnostic (see [`Self::resolve`]).
    pub fn from_env() -> std::result::Result<Self, String> {
        let raw = std::env::var(DECODE_AFFINITY_ENV).ok();
        Self::resolve(raw.as_deref(), NumaTopology::detect().as_ref())
    }

    /// Parse `raw` and validate it against `topology`, producing one consistent
    /// diagnostic for every invalid value.
    ///
    /// A malformed mode, a non-integer node index, an unknown node index, and a
    /// node index requested on a host without discoverable topology all fail the
    /// same way: the error names the rejected value, all accepted modes, and the
    /// available-node list (or states that topology is unavailable). `compact`
    /// and `off` are always accepted; a `compact` request on a host with no
    /// discoverable topology is honored as "leave unpinned" by the caller.
    pub fn resolve(
        raw: Option<&str>,
        topology: Option<&NumaTopology>,
    ) -> std::result::Result<Self, String> {
        match Self::parse(raw) {
            Ok(Self::Node(index)) => {
                if topology
                    .and_then(|topology| topology.cpus_for_node(index))
                    .is_some()
                {
                    Ok(Self::Node(index))
                } else {
                    Err(invalid_selector_error(&format!("node:{index}"), topology))
                }
            }
            Ok(affinity) => Ok(affinity),
            Err(_) => Err(invalid_selector_error(
                raw.unwrap_or_default().trim(),
                topology,
            )),
        }
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
                None => Err(format!(
                    "{DECODE_AFFINITY_ENV}=`node:{index}` names an unknown NUMA node; \
                     accepted modes are {ACCEPTED_MODES}; {clause}",
                    clause = available_nodes_clause(Some(self)),
                )),
            },
            DecodeAffinity::Compact => {
                // `nodes` is a BTreeMap, so `.values()` walks nodes in ascending
                // index order; `find` therefore returns the smallest-index node
                // whose CPU count covers the pool, matching the documented policy.
                let fitting = self.nodes.values().find(|cpus| cpus.len() >= worker_count);
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

/// Build a kernel CPU affinity mask (an array of `unsigned long` words, the
/// exact layout `sched_setaffinity` expects) with only `cpu`'s bit set.
///
/// The mask is sized from the runtime CPU index, so a `cpu` at or above the
/// fixed `CPU_SETSIZE` (1024) grows the allocation to cover it instead of
/// writing out of bounds (the flaw the reviewer flagged in the fixed
/// `cpu_set_t`). Returns `None` only when `cpu` is so large that the word count
/// overflows `usize`, in which case the caller falls back to unpinned rather
/// than attempting a nonsensical allocation.
///
/// A hand-built `Vec` is used instead of `CPU_ALLOC`/`CPU_SET_S` because those
/// helpers are not exposed by the `libc` crate for the `*-linux-gnu` target;
/// this owns its buffer in safe Rust, so the only remaining `unsafe` is the
/// syscall itself, and the sizing is directly unit-testable.
#[cfg(target_os = "linux")]
fn build_cpu_mask(cpu: usize) -> Option<Vec<libc::c_ulong>> {
    let bits_per_word = 8 * std::mem::size_of::<libc::c_ulong>();
    let word_index = cpu / bits_per_word;
    let bit = cpu % bits_per_word;
    let len = word_index.checked_add(1)?;
    let mut mask = vec![0 as libc::c_ulong; len];
    mask[word_index] = (1 as libc::c_ulong) << bit;
    Some(mask)
}

/// Pin the calling thread to a single CPU. Best-effort: a failure to set
/// affinity (e.g. a restricted cgroup) is reported so the caller can log it,
/// but never aborts decode.
#[cfg(target_os = "linux")]
pub fn pin_current_thread_to_cpu(cpu: usize) -> std::result::Result<(), String> {
    // Size the mask from `cpu` itself so a large CPU index can never index past
    // a fixed 1024-bit `cpu_set_t`; on overflow we fall back to unpinned.
    let mask = build_cpu_mask(cpu)
        .ok_or_else(|| format!("cpu index {cpu} is too large to build a CPU affinity mask"))?;
    let byte_len = mask.len() * std::mem::size_of::<libc::c_ulong>();
    // SAFETY: `mask` is a live, `byte_len`-sized allocation of `unsigned long`
    // words in the exact layout `sched_setaffinity` expects; we pass its true
    // byte length and set affinity for the current thread (`pid == 0`). The
    // pointer is only read for the call and the `Vec` outlives it.
    let result =
        unsafe { libc::sched_setaffinity(0, byte_len, mask.as_ptr() as *const libc::cpu_set_t) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!(
            "sched_setaffinity(cpu={cpu}) failed: {}",
            std::io::Error::last_os_error()
        ))
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

    #[test]
    fn resolve_reports_consistent_diagnostics_for_invalid_values() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, vec![0, 1]);
        nodes.insert(2, vec![4, 5]);
        let topology = NumaTopology { nodes };

        // Every invalid path names the rejected value, all accepted modes, and
        // the available-node list.
        for (raw, needle) in [
            ("bogus", "bogus"),         // malformed mode
            ("node:x", "node:x"),       // non-integer node index
            ("node:9", "node:9"),       // unknown node index
        ] {
            let err = DecodeAffinity::resolve(Some(raw), Some(&topology)).unwrap_err();
            assert!(err.contains(needle), "names rejected value `{needle}`: {err}");
            assert!(
                err.contains("`off`")
                    && err.contains("`compact`")
                    && err.contains("`node:<index>`"),
                "lists all accepted modes: {err}"
            );
            assert!(
                err.contains("available NUMA nodes are [0, 2]"),
                "lists available nodes: {err}"
            );
        }
    }

    #[test]
    fn resolve_reports_topology_unavailable_for_node_without_topology() {
        // A node selector on a host with no discoverable topology is reported,
        // not silently treated as fallback, and the message says topology is
        // unavailable while still listing the accepted modes and rejected value.
        let err = DecodeAffinity::resolve(Some("node:1"), None).unwrap_err();
        assert!(err.contains("node:1"), "{err}");
        assert!(
            err.contains("`off`") && err.contains("`compact`") && err.contains("`node:<index>`"),
            "{err}"
        );
        assert!(err.contains("NUMA topology is unavailable"), "{err}");

        // `compact` and `off` remain acceptable without topology.
        assert_eq!(
            DecodeAffinity::resolve(Some("compact"), None).unwrap(),
            DecodeAffinity::Compact
        );
        assert_eq!(
            DecodeAffinity::resolve(None, None).unwrap(),
            DecodeAffinity::Off
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn build_cpu_mask_sizes_beyond_cpu_setsize_without_oob() {
        let bits = 8 * std::mem::size_of::<libc::c_ulong>();

        // Low index: a single word with exactly the right bit set.
        let mask = build_cpu_mask(3).unwrap();
        assert_eq!(mask.len(), 1);
        assert_eq!(mask[0], (1 as libc::c_ulong) << 3);

        // At the fixed CPU_SETSIZE (1024) the mask must grow beyond a fixed
        // `cpu_set_t` instead of indexing out of bounds, with the correct bit in
        // the correct word and every earlier word left zero.
        let cpu = 1024;
        let mask = build_cpu_mask(cpu).unwrap();
        let word = cpu / bits;
        assert_eq!(mask.len(), word + 1);
        assert!(
            mask.len() * std::mem::size_of::<libc::c_ulong>()
                > std::mem::size_of::<libc::cpu_set_t>(),
            "cpu {cpu} must need a mask larger than a fixed cpu_set_t"
        );
        assert_eq!(mask[word], (1 as libc::c_ulong) << (cpu % bits));
        assert!(mask[..word].iter().all(|&w| w == 0));

        // Far beyond CPU_SETSIZE stays sound (graceful, no panic / OOB).
        let cpu = 5000;
        let mask = build_cpu_mask(cpu).unwrap();
        assert_eq!(mask.len(), cpu / bits + 1);
        assert_eq!(mask[cpu / bits], (1 as libc::c_ulong) << (cpu % bits));
    }

    #[test]
    fn compact_prefers_smallest_index_not_fewest_cpus() {
        // node 0 has MORE CPUs than node 1; both cover the pool. The documented
        // policy selects the smallest index (node 0). The previous
        // `min_by_key(len)` would have wrongly picked node 1 (fewest CPUs).
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..64).collect::<Vec<_>>());
        nodes.insert(1, (64..96).collect::<Vec<_>>());
        let topology = NumaTopology { nodes };
        let cpus = topology
            .cpus_for(&DecodeAffinity::Compact, 16)
            .unwrap()
            .unwrap();
        assert_eq!(cpus[0], 0);
        assert_eq!(cpus.len(), 64);
    }
}
