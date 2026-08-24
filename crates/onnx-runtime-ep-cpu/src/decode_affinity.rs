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
//! Topology is *queried* from the running machine, never hardcoded:
//! `/sys/devices/system/node` on Linux, `GetLogicalProcessorInformationEx`
//! (`RelationNumaNode`) on Windows. macOS exposes no thread-to-core affinity, so
//! pinning there is a documented, logged no-op; any other target falls back to
//! unpinned decode. The behaviour is selected through an explicit environment
//! switch so it stays inspectable, and — when that switch is unset — an
//! auto-enable policy pins `compact` on multi-node hosts where it is safe (see
//! [`plan_decode_affinity`]).
//!
//! ## Cross-processor CPU identity
//!
//! CPU indices in [`NumaTopology`] are opaque, OS-defined handles, never assumed
//! to be dense or socket-ordered. On Linux they are the kernel's logical CPU
//! ids. On Windows, where a host with more than 64 logical CPUs is partitioned
//! into *processor groups* of up to 64 CPUs each, a CPU is encoded as
//! `group * 64 + bit`, so pinning recovers the group and in-group mask exactly
//! (`group = cpu / 64`, `mask = 1 << (cpu % 64)`); this handles >64-CPU hosts
//! correctly instead of silently truncating to one group.
//!
//! ## Container / cgroup / taskset safety
//!
//! Before pinning, the OS-discovered topology is intersected with the process's
//! actual allowed CPU set (Linux `sched_getaffinity`, Windows
//! `GetThreadGroupAffinity`), so a container or `taskset`-restricted host never
//! tries to pin to a CPU it is not permitted to run on. If that intersection
//! leaves fewer than two NUMA nodes, auto-enable declines and decode stays
//! unpinned (logged once).

use std::collections::BTreeMap;

/// Selects how the decode pool binds its workers to CPUs.
///
/// * unset / `off` -- no pinning; workers float on the OS scheduler (default).
/// * `compact` -- pin the workers, one per CPU, to the CPUs of a single NUMA
///   node (the smallest-index node whose CPU count covers the pool size, so all
///   workers, their fork-join barriers, and their weight reads stay node-local).
/// * `node:<index>` -- pin the workers to the CPUs of the named NUMA node.
/// * `numa-split` -- spread the workers across *every* NUMA node as node-pinned
///   sub-pools and shard each M=1 projection's output rows across them, so both
///   sockets' memory bandwidth is used while each per-op barrier stays
///   node-local (see [`crate::decode_numa`]).
pub const DECODE_AFFINITY_ENV: &str = "ONNX_GENAI_CPU_DECODE_AFFINITY";

/// The complete set of accepted affinity modes, named in every diagnostic so a
/// rejected value always sees the full menu of valid options.
const ACCEPTED_MODES: &str = "`off`, `compact`, `node:<index>`, `numa-split`";

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
        _ => "NUMA topology is unavailable on this host (single NUMA node or a \
              platform without discoverable NUMA topology), so no node selector \
              can be honored"
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
    /// Shard M=1 decode across per-node sub-pools spanning every NUMA node.
    NumaSplit,
}

/// One NUMA node's share of a [`DecodeAffinity::NumaSplit`] decode layout:
/// which node, which CPUs it may pin to, and how many workers it receives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NodeShard {
    /// The NUMA node index this shard pins to.
    pub index: usize,
    /// The CPUs of the node the shard's workers may run on.
    pub cpus: Vec<usize>,
    /// The number of decode workers assigned to this node.
    pub workers: usize,
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
            "numa-split" => Ok(Self::NumaSplit),
            other => {
                if let Some(index) = other.strip_prefix("node:") {
                    index.trim().parse::<usize>().map(Self::Node).map_err(|_| {
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
    /// platform exposes no multi-node information (a single node, or a target
    /// without discoverable NUMA topology such as macOS or an unsupported OS).
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::detect_linux()
        }
        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }
        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // macOS exposes no thread-to-core affinity and, in practice, a
            // single NUMA node; every other target has no discoverable topology
            // here. Report "no multi-node topology" so decode stays unpinned.
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

    /// Detect NUMA topology on Windows via `GetLogicalProcessorInformationEx`
    /// with `RelationNumaNode`, returning `None` when the host has a single node
    /// or the API is unavailable.
    ///
    /// Each `NUMA_NODE_RELATIONSHIP` carries a `GROUP_AFFINITY` naming the
    /// processor group and a 64-bit CPU mask within it, so a CPU is encoded as
    /// `group * 64 + bit`. This spans processor groups correctly, so a host with
    /// more than 64 logical CPUs is not silently truncated to its first group.
    #[cfg(target_os = "windows")]
    fn detect_windows() -> Option<Self> {
        let nodes = windows_imp::numa_nodes()?;
        let nodes: BTreeMap<usize, Vec<usize>> = nodes
            .into_iter()
            .filter(|(_, cpus)| !cpus.is_empty())
            .collect();
        (nodes.len() > 1).then_some(Self { nodes })
    }

    /// Intersect every node's CPU list with `allowed`, dropping any CPU the
    /// process may not run on and any node left empty. Returns the restricted
    /// topology (which may have fewer nodes, or be empty).
    ///
    /// `allowed == None` means the allowed set could not be determined, so the
    /// topology is returned unchanged (we do not guess a restriction). This is
    /// the cgroup/cpuset/taskset safety gate: pinning only ever targets CPUs the
    /// intersection proves the process is permitted to use.
    pub fn restrict_to_allowed(&self, allowed: Option<&[usize]>) -> Self {
        let Some(allowed) = allowed else {
            return self.clone();
        };
        let allowed: std::collections::BTreeSet<usize> = allowed.iter().copied().collect();
        let nodes = self
            .nodes
            .iter()
            .filter_map(|(&index, cpus)| {
                let kept: Vec<usize> = cpus
                    .iter()
                    .copied()
                    .filter(|c| allowed.contains(c))
                    .collect();
                (!kept.is_empty()).then_some((index, kept))
            })
            .collect();
        Self { nodes }
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
            DecodeAffinity::NumaSplit => {
                // NumaSplit does not pin a single flat pool; the per-node
                // sub-pools are built from `split_workers`. Leaving the single
                // fallback pool unpinned keeps the escape hatch well-defined.
                Ok(None)
            }
        }
    }

    /// Spread `total_workers` across *every* NUMA node for the
    /// [`DecodeAffinity::NumaSplit`] layout, returning one [`NodeShard`] per node
    /// that receives at least one worker.
    ///
    /// Workers are distributed as evenly as possible (nodes with a lower index
    /// absorb the remainder first) and a node never receives more workers than
    /// it has CPUs. Returns `None` when fewer than two nodes would receive a
    /// worker, since a single-node layout has no cross-socket bandwidth to gain
    /// and the caller should fall back to the flat single-node path.
    pub fn split_workers(&self, total_workers: usize) -> Option<Vec<NodeShard>> {
        if total_workers == 0 || self.nodes.len() < 2 {
            return None;
        }
        let node_count = self.nodes.len();
        let base = total_workers / node_count;
        let remainder = total_workers % node_count;
        let mut shards = Vec::with_capacity(node_count);
        for (position, (&index, cpus)) in self.nodes.iter().enumerate() {
            let requested = base + usize::from(position < remainder);
            let workers = requested.min(cpus.len());
            if workers == 0 {
                continue;
            }
            shards.push(NodeShard {
                index,
                cpus: cpus.clone(),
                workers,
            });
        }
        (shards.len() >= 2).then_some(shards)
    }
}

/// Parse a Linux `cpulist` string such as `"0-3,8,10-11"` into CPU indices.
#[cfg(any(target_os = "linux", test))]
fn parse_cpu_list(list: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in list.trim().split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((start, end)) = part.split_once('-') {
            if let (Ok(start), Ok(end)) =
                (start.trim().parse::<usize>(), end.trim().parse::<usize>())
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

#[cfg(target_os = "windows")]
pub fn pin_current_thread_to_cpu(cpu: usize) -> std::result::Result<(), String> {
    windows_imp::pin_current_thread_to_cpu(cpu)
}

/// macOS exposes no public thread-to-core affinity API (the old
/// `thread_policy_set(THREAD_AFFINITY_POLICY)` is an advisory hint the scheduler
/// is free to ignore and is unavailable on Apple Silicon), so pinning is a
/// documented no-op. It reports an error string so the caller logs the no-op
/// once and continues with an unpinned decode pool. In practice this path is not
/// reached, because [`pinning_supported`] is `false` on macOS, so auto-enable
/// never builds a pinning pool there.
#[cfg(target_os = "macos")]
pub fn pin_current_thread_to_cpu(cpu: usize) -> std::result::Result<(), String> {
    Err(format!(
        "thread-to-core affinity is not supported on macOS; \
         decode worker for cpu {cpu} runs unpinned (no-op)"
    ))
}

/// Any other target: no affinity mechanism is implemented, so pinning is a
/// graceful, logged no-op and decode runs unpinned. Like the macOS path this is
/// not reached in practice because [`pinning_supported`] gates it off.
#[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
pub fn pin_current_thread_to_cpu(cpu: usize) -> std::result::Result<(), String> {
    Err(format!(
        "thread-to-core affinity is not implemented on this platform; \
         decode worker for cpu {cpu} runs unpinned (no-op)"
    ))
}

/// Whether this OS can actually pin a thread to a CPU. Drives the auto-enable
/// policy: on a platform that cannot pin, auto-enable stays `off` rather than
/// building a pool whose pinning would be a silent no-op.
pub const fn pinning_supported() -> bool {
    cfg!(any(target_os = "linux", target_os = "windows"))
}

/// Build a kernel CPU affinity mask with a bit set for every CPU in `cpus`.
///
/// Multi-CPU sibling of [`build_cpu_mask`]: the mask is sized from the largest
/// CPU index so a CPU at or above the fixed `CPU_SETSIZE` (1024) grows the
/// allocation to cover it instead of writing out of bounds. Returns `None` when
/// `cpus` is empty or the required word count overflows `usize`.
#[cfg(target_os = "linux")]
fn build_cpu_mask_multi(cpus: &[usize]) -> Option<Vec<libc::c_ulong>> {
    let bits_per_word = 8 * std::mem::size_of::<libc::c_ulong>();
    let max_cpu = *cpus.iter().max()?;
    let len = (max_cpu / bits_per_word).checked_add(1)?;
    let mut mask = vec![0 as libc::c_ulong; len];
    for &cpu in cpus {
        mask[cpu / bits_per_word] |= (1 as libc::c_ulong) << (cpu % bits_per_word);
    }
    Some(mask)
}

/// Confine the calling thread (and, because threads inherit their creator's
/// affinity mask on Linux, every thread it later spawns) to the CPUs in `cpus`.
///
/// This is the process-wide "good citizen" mask applied once, early, before any
/// worker pool is built, so the Rayon global pool and the SPMD decode workers
/// all inherit the restricted set. Best-effort: a failure (e.g. a restricted
/// cgroup) is reported so the caller can log it, but never aborts inference.
#[cfg(target_os = "linux")]
pub fn set_current_thread_affinity(cpus: &[usize]) -> std::result::Result<(), String> {
    if cpus.is_empty() {
        return Err("cannot set CPU affinity to an empty CPU set".to_string());
    }
    let mask = build_cpu_mask_multi(cpus)
        .ok_or_else(|| "CPU indices are too large to build a CPU affinity mask".to_string())?;
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
            "sched_setaffinity(cpus={cpus:?}) failed: {}",
            std::io::Error::last_os_error()
        ))
    }
}

/// Non-Linux hosts have no process-wide CPU affinity mask applied here (Windows
/// per-thread pinning is handled by the SPMD pool; macOS and others cannot pin),
/// so this is a documented no-op that reports the reason for the caller to log.
#[cfg(not(target_os = "linux"))]
pub fn set_current_thread_affinity(cpus: &[usize]) -> std::result::Result<(), String> {
    let _ = cpus;
    Err("process-wide CPU affinity masking is only implemented on Linux (no-op)".to_string())
}

/// Whether the user explicitly requested a decode affinity via
/// [`DECODE_AFFINITY_ENV`] (a non-empty value). When set, the automatic
/// good-citizen process mask stands down so the user's affinity choice wins.
pub fn explicit_decode_affinity_requested() -> bool {
    std::env::var(DECODE_AFFINITY_ENV)
        .ok()
        .is_some_and(|value| !value.trim().is_empty())
}

/// Order `pool` so that one CPU per physical core comes first, then the SMT
/// siblings, and keep the first `count`.
///
/// This is the whole point of [`choose_budget_cpus`] on an SMT host: a budget of
/// `count` CPUs taken as the `count` lowest indices lands on `count / 2`
/// physical cores whenever the kernel numbers siblings adjacently (`0-1`,
/// `2-3`, … on AMD EPYC and on every Intel host since Skylake-SP). Two threads
/// sharing one core's front end do not add a core's worth of throughput, so the
/// naive mask halves the machine and then burns twice the CPU proving it.
///
/// Leaders come from [`CoreTopology::leaders_within`], which only ever returns
/// CPUs already in `pool`, so this reorders — it never widens the mask. When the
/// topology is unknown every CPU is its own leader and the result is exactly the
/// ascending prefix the old code produced.
fn scatter_across_cores(
    pool: &[usize],
    count: usize,
    cores: Option<&crate::core_topology::CoreTopology>,
) -> Vec<usize> {
    let leaders: Vec<usize> = match cores {
        Some(cores) => cores.leaders_within(pool),
        None => pool.to_vec(),
    };
    let leader_set: std::collections::BTreeSet<usize> = leaders.iter().copied().collect();
    let mut ranked = leaders;
    ranked.extend(pool.iter().copied().filter(|cpu| !leader_set.contains(cpu)));
    ranked.truncate(count);
    ranked.sort_unstable();
    ranked
}

/// How many logical CPUs a NUMA node must hold to supply `count` distinct
/// physical cores: `count × threads-per-core`, or `count` itself when the SMT
/// map is unknown.
///
/// The node search in [`NumaTopology::cpus_for`] is sized in *logical* CPUs, so
/// asking it for `count` would accept a node that has only `count / 2` cores and
/// force the cross-node top-up for budgets that could have been served by one
/// core-rich node.
fn smt_scaled_request(count: usize, cores: Option<&crate::core_topology::CoreTopology>) -> usize {
    let factor = cores
        .filter(|c| c.core_count() > 0)
        .map(|c| c.logical_count().div_ceil(c.core_count()))
        .unwrap_or(1)
        .max(1);
    count.saturating_mul(factor)
}

/// Pick `count` CPUs to confine the whole process to, preferring CPUs packed on
/// a single NUMA node for memory-bandwidth and barrier locality and spread one
/// per physical core for throughput.
///
/// Pure and platform-independent so it is unit-testable without touching the
/// kernel. `topology` is expected to already be restricted to the process's
/// allowed CPU set; `allowed` is the raw allowed list used both to keep the
/// result within the process cpuset and to top up the selection when no single
/// node covers `count`; `cores` is the SMT sibling map used to rank the survivors
/// (see [`scatter_across_cores`]). The result never exceeds `count`, never
/// contains a CPU outside `allowed` (when `allowed` is known), and is returned in
/// ascending order. Returns `None` when `count` is zero or no usable CPU can be
/// chosen.
fn choose_budget_cpus(
    topology: Option<&NumaTopology>,
    allowed: Option<&[usize]>,
    count: usize,
    cores: Option<&crate::core_topology::CoreTopology>,
) -> Option<Vec<usize>> {
    if count == 0 {
        return None;
    }
    let allowed_set: Option<std::collections::BTreeSet<usize>> =
        allowed.map(|a| a.iter().copied().collect());
    let mut pool: Vec<usize> = Vec::new();
    if let Some(topology) = topology {
        // Prefer a node that can supply `count` distinct *cores*; only if no node
        // is that large does the search fall back to sizing in logical CPUs, so a
        // budget that fits on one node still stays on one node.
        let wide = smt_scaled_request(count, cores);
        if wide > count
            && let Ok(Some(node_cpus)) = topology.cpus_for(&DecodeAffinity::Compact, wide)
            && node_cpus.len() >= wide
        {
            pool = node_cpus;
        }
        if pool.is_empty()
            && let Ok(Some(node_cpus)) = topology.cpus_for(&DecodeAffinity::Compact, count)
        {
            pool = node_cpus;
        }
    }
    if let Some(set) = &allowed_set {
        pool.retain(|cpu| set.contains(cpu));
    }
    pool.sort_unstable();
    pool.dedup();
    // Top up *before* ranking: a second node's CPUs are new physical cores, and
    // they must compete for leader slots against the first node's SMT siblings
    // rather than be appended after a mask that is already full.
    if pool.len() < count
        && let Some(allowed) = allowed
    {
        let mut extra: Vec<usize> = allowed
            .iter()
            .copied()
            .filter(|cpu| !pool.contains(cpu))
            .collect();
        extra.sort_unstable();
        pool.extend(extra);
    }
    let selected = scatter_across_cores(&pool, count, cores);
    (!selected.is_empty()).then_some(selected)
}

/// Resolve the `count` CPUs the process should be confined to on the running
/// host: detect NUMA topology, intersect it with the process's allowed CPU set,
/// and pick CPUs packed on a single node where possible (see
/// [`choose_budget_cpus`]). Returns `None` when no usable CPU set can be
/// determined (e.g. the allowed set is unknown and no topology is discoverable).
pub fn select_budget_cpus(count: usize) -> Option<Vec<usize>> {
    let allowed = allowed_cpus();
    let restricted = NumaTopology::detect().map(|t| t.restrict_to_allowed(allowed.as_deref()));
    choose_budget_cpus(
        restricted.as_ref(),
        allowed.as_deref(),
        count,
        crate::core_topology::host(),
    )
}

/// The CPUs the current process is actually permitted to run on, or `None` when
/// the allowed set cannot be determined on this platform.
///
/// This is the cgroup/cpuset/taskset safety input: intersecting it with the
/// discovered topology (see [`NumaTopology::restrict_to_allowed`]) guarantees we
/// never try to pin to a CPU outside the process's cpuset. `None` means "do not
/// restrict" — we could not learn the mask, so we do not guess one.
pub fn allowed_cpus() -> Option<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        linux_allowed_cpus()
    }
    #[cfg(target_os = "windows")]
    {
        windows_imp::allowed_cpus()
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        None
    }
}

/// What asking the **calling thread** which CPUs it may run on returned.
///
/// Three states, not two, and deliberately not `Option<Vec<usize>>`: a caller
/// that cannot tell "this target has no affinity query" from "the query ran and
/// failed" from "here is the mask" will eventually collapse the first two into
/// the third's success path. That collapse is exactly how a pool comes to report
/// a placement nothing observed -- the defect behind #1792 -- so the type makes
/// it unrepresentable rather than merely discouraged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ObservedAffinity {
    /// The kernel reported this thread's effective CPU set.
    Cpus(Vec<usize>),
    /// The query is implemented for this target but failed at runtime (a
    /// seccomp filter, an emulator without the syscall, a hardened container).
    QueryFailed(String),
    /// This target has no per-thread affinity query at all, so no amount of
    /// retrying would answer. macOS is the live example: `thread_policy_set`
    /// is an advisory hint and there is nothing to read back.
    Unsupported,
}

impl ObservedAffinity {
    /// The observed CPU set, or `None` when the mask was never obtained.
    ///
    /// Callers that only want the happy path use this; callers that must *fail
    /// closed* match on the variants, because `None` here is two very different
    /// facts glued together.
    pub fn cpus(&self) -> Option<&[usize]> {
        match self {
            Self::Cpus(cpus) => Some(cpus),
            Self::QueryFailed(_) | Self::Unsupported => None,
        }
    }

    /// `Some(cpu)` when the thread is confined to exactly one CPU.
    ///
    /// A mask with more than one bit is not a pin, however it was requested:
    /// the scheduler is free to use any of those CPUs, so a multi-bit mask
    /// cannot support a one-worker-per-core claim.
    pub fn pinned_cpu(&self) -> Option<usize> {
        match self.cpus()? {
            [cpu] => Some(*cpu),
            _ => None,
        }
    }
}

/// Whether this target can report a thread's *effective* affinity mask.
///
/// A `cfg!` constant, and that is the point: it is a property of the target,
/// so it cannot be falsified by the very failure a fail-closed check exists to
/// catch. Keying a guard on `observe_current_thread_cpus().cpus().is_some()`
/// would switch the guard off in precisely the case it is there to report.
pub const fn affinity_observation_supported() -> bool {
    cfg!(any(target_os = "linux", target_os = "windows"))
}

/// Ask the **calling thread** which CPUs it is currently permitted to run on.
///
/// This is the read-back half of [`pin_current_thread_to_cpu`], and it is the
/// only evidence that a pin actually happened. A pin call's return value says
/// the request was accepted; it does not say the kernel enforced anything, and
/// a build where the syscall is stubbed, filtered, or emulated away returns
/// `Ok(())` while every thread stays free to roam. Ask the kernel instead.
///
/// Per-thread on every supported target: Linux `sched_getaffinity(0, ..)` and
/// Windows `GetThreadGroupAffinity(GetCurrentThread(), ..)` both scope to the
/// caller, so this must be called *on* the thread being asked about.
pub fn observe_current_thread_cpus() -> ObservedAffinity {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        match linux_thread_cpus() {
            Ok(cpus) => ObservedAffinity::Cpus(cpus),
            Err(message) => ObservedAffinity::QueryFailed(message),
        }
    }
    #[cfg(all(target_os = "linux", miri))]
    {
        // Miri has no scheduler affinity to report and would abort on the
        // syscall shim. Unsupported is the honest answer, and it fails closed.
        ObservedAffinity::Unsupported
    }
    #[cfg(target_os = "windows")]
    {
        windows_imp::allowed_cpus().map_or_else(
            || {
                ObservedAffinity::QueryFailed(
                    "GetThreadGroupAffinity did not report a mask".to_string(),
                )
            },
            ObservedAffinity::Cpus,
        )
    }
    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        ObservedAffinity::Unsupported
    }
}

/// Read the calling thread's CPU affinity mask via `sched_getaffinity` and
/// return the set CPU indices. The mask buffer is grown on `EINVAL` (the kernel
/// signalling it is too small for the host's CPU count), so hosts with more than
/// 1024 CPUs are handled without a fixed `cpu_set_t`.
///
/// `pid == 0` means the *calling thread* on Linux, not the process: threads have
/// independent masks, which is what makes per-worker pinning observable at all.
#[cfg(target_os = "linux")]
fn linux_thread_cpus() -> std::result::Result<Vec<usize>, String> {
    let bits_per_word = 8 * std::mem::size_of::<libc::c_ulong>();
    // Start comfortably above a typical core count and grow on EINVAL.
    let mut words = 128 / bits_per_word.max(1);
    words = words.max(16);
    loop {
        let mut mask = vec![0 as libc::c_ulong; words];
        let byte_len = words * std::mem::size_of::<libc::c_ulong>();
        // SAFETY: `mask` is a live `byte_len`-sized buffer of `unsigned long`
        // words in the `cpu_set_t` layout the syscall fills; we pass its true
        // byte length and query the calling thread (`pid == 0`). The buffer
        // outlives the call and is only written by the kernel up to `byte_len`.
        let result = unsafe {
            libc::sched_getaffinity(0, byte_len, mask.as_mut_ptr() as *mut libc::cpu_set_t)
        };
        if result == 0 {
            let mut cpus = Vec::new();
            for (word_index, &word) in mask.iter().enumerate() {
                for bit in 0..bits_per_word {
                    if word & ((1 as libc::c_ulong) << bit) != 0 {
                        cpus.push(word_index * bits_per_word + bit);
                    }
                }
            }
            if cpus.is_empty() {
                // A runnable thread always has at least one allowed CPU, so an
                // empty mask means the read did not describe reality.
                return Err("sched_getaffinity returned an empty mask".to_string());
            }
            return Ok(cpus);
        }
        let err = std::io::Error::last_os_error();
        if err.raw_os_error() == Some(libc::EINVAL) {
            // Buffer too small for the host's CPU count: double and retry, with a
            // sane ceiling so a persistent EINVAL cannot loop forever.
            words = words
                .checked_mul(2)
                .ok_or_else(|| "cpu mask size overflowed while growing".to_string())?;
            if words * bits_per_word > 1 << 20 {
                return Err("sched_getaffinity kept rejecting the mask size".to_string());
            }
            continue;
        }
        return Err(format!("sched_getaffinity failed: {err}"));
    }
}

/// The CPUs the current thread is permitted to run on, or `None` when the mask
/// could not be read at all. The `Option`-shaped view of [`linux_thread_cpus`]
/// for the cpuset/taskset safety input, which genuinely wants "do not restrict"
/// on any failure and does not need to distinguish the reasons.
#[cfg(target_os = "linux")]
fn linux_allowed_cpus() -> Option<Vec<usize>> {
    linux_thread_cpus().ok()
}

/// A resolved decode-affinity plan: the CPUs to pin the pool's workers to (round
/// robin), plus an optional one-time human-readable explanation of the chosen
/// policy for the caller to log at info.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodePlan {
    /// `Some(cpus)` -> pin each worker to `cpus[worker % cpus.len()]`.
    /// `None` -> leave the pool unpinned.
    pub cpus: Option<Vec<usize>>,
    /// A single line describing the auto-enable/decline decision, or `None` when
    /// the behaviour is fully explicit (an operator-set mode) and needs no note.
    pub log: Option<String>,
}

/// Decide the effective decode affinity, applying the auto-enable policy.
///
/// `raw` is the raw environment value (`None` = unset). `restricted` is the
/// topology already intersected with the process's allowed CPU set (so it may
/// have fewer nodes than the host). `host_is_multinode` records whether the
/// *unrestricted* host had more than one NUMA node, so a drop to a single node
/// can be attributed to a cpuset restriction. `pinning_supported` reports
/// whether this OS can pin at all.
///
/// Policy:
/// * env unset + pinning unsupported -> `off` (logged once).
/// * env unset + >=2 restricted nodes -> `compact` auto-enabled (logged once).
/// * env unset + host multi-node but restricted to <2 nodes -> `off`, cpuset
///   declines auto-enable (logged once).
/// * env unset + single-node host -> `off`, unchanged, no log.
/// * env `off` -> `off`, explicit opt-out, no auto log.
/// * any other explicit mode -> parsed and validated against `restricted`.
fn decide_affinity(
    raw: Option<&str>,
    restricted: Option<&NumaTopology>,
    host_is_multinode: bool,
    pinning_supported: bool,
) -> std::result::Result<(DecodeAffinity, Option<String>), String> {
    let restricted_nodes = restricted.map(NumaTopology::node_count).unwrap_or(0);
    // Treat an unset or empty/whitespace value as "unset" for auto-enable.
    let is_unset = raw.map(|value| value.trim().is_empty()).unwrap_or(true);
    if is_unset {
        if !pinning_supported {
            return Ok((
                DecodeAffinity::Off,
                Some(format!(
                    "{DECODE_AFFINITY_ENV} unset and CPU pinning is not supported on this OS; \
                     decode pool left unpinned"
                )),
            ));
        }
        if restricted_nodes >= 2 {
            return Ok((
                DecodeAffinity::Compact,
                Some(format!(
                    "{DECODE_AFFINITY_ENV} unset and host has {restricted_nodes} usable NUMA \
                     nodes; auto-enabling `compact` (pin the decode pool to one node for \
                     bandwidth/barrier locality). Set {DECODE_AFFINITY_ENV}=off to opt out, or \
                     any explicit mode ({ACCEPTED_MODES}) to override"
                )),
            ));
        }
        if host_is_multinode {
            return Ok((
                DecodeAffinity::Off,
                Some(format!(
                    "{DECODE_AFFINITY_ENV} unset; host is multi-node but the process cpuset spans \
                     fewer than two NUMA nodes, so auto-enable declines and the decode pool is \
                     left unpinned (safe under container/taskset restriction)"
                )),
            ));
        }
        // Single-node host: unchanged behaviour, nothing noteworthy to log.
        return Ok((DecodeAffinity::Off, None));
    }

    // Explicit value: `off` opts out; anything else is validated against the
    // (allowed-restricted) topology, so an explicit mode also never pins outside
    // the process cpuset.
    match DecodeAffinity::resolve(raw, restricted)? {
        DecodeAffinity::Off => Ok((DecodeAffinity::Off, None)),
        affinity => Ok((affinity, None)),
    }
}

/// Resolve the full decode-affinity plan for the running host: detect topology,
/// intersect it with the process's allowed CPU set, apply the auto-enable
/// policy, and return the CPUs to pin `worker_count` workers to plus a one-time
/// log line. This is the single entry point the decode-pool builder uses.
pub fn plan_decode_affinity(worker_count: usize) -> std::result::Result<DecodePlan, String> {
    let raw = std::env::var(DECODE_AFFINITY_ENV).ok();
    let full = NumaTopology::detect();
    let host_is_multinode = full.as_ref().map(|t| t.node_count() >= 2).unwrap_or(false);
    let allowed = allowed_cpus();
    let restricted = full
        .as_ref()
        .map(|t| t.restrict_to_allowed(allowed.as_deref()));
    let usable = restricted.as_ref().filter(|t| t.node_count() >= 1);

    let (affinity, log) = decide_affinity(
        raw.as_deref(),
        usable,
        host_is_multinode,
        pinning_supported(),
    )?;

    let cpus = match usable {
        Some(topology) => topology
            .cpus_for(&affinity, worker_count)?
            .map(|cpus| order_pin_targets(&cpus, crate::core_topology::host()))
            .filter(|cpus| !cpus.is_empty()),
        None => None,
    };
    Ok(DecodePlan { cpus, log })
}

/// Order a decode pool's CPU list so that worker `i` lands on a distinct
/// physical core for as long as cores last, and only then starts doubling up on
/// SMT siblings.
///
/// [`crate::kernels::matmul_nbits`]'s `build_decode_pool` pins worker `i` to
/// `cpus[i % cpus.len()]`, so the *order* of this list — not just its contents —
/// decides whether an 8-worker pool on a 16-core/32-thread node occupies 8 cores
/// or 4. Same set, same cpuset guarantees; only the ranking changes. With no
/// discoverable SMT map this is the identity.
///
/// # Who uses it, and a reversal
///
/// Applied by the fork-join pool, where a worker handed a share of the
/// arithmetic wants its own core's front end, **and now also by the persistent
/// SPMD pool** ([`crate::decode_spmd`]'s shard builder).
///
/// The SPMD half is a reversal. This doc previously said the spinning pool was
/// deliberately left compact, on the strength of the 0.133 ms vs 0.079 ms
/// experiment recorded in [`crate::core_topology`]'s module docs. What that
/// experiment did not separate is the *dispatcher*: the inline dispatcher
/// spin-waits on the completion counters, and with every physical core taken by
/// a worker it has nowhere to run but some worker's SMT sibling, making that
/// worker the straggler the whole barrier waits for on every op. Pinning only
/// the dispatcher and changing nothing else measured 2.1x (19.58 ms/token on a
/// CPU inside the worker set against 8.77 ms on a free core), with 600
/// involuntary context switches per token in the contended case. So the
/// spinning pool is spread *and* [`crate::decode_spmd`] reserves a core for the
/// dispatcher; see [`crate::core_topology`]'s module docs for what survives of
/// the compact-mask result and what does not.
///
/// Still **not** applied to the `numa-split` sub-pools
/// ([`crate::decode_numa`]), whose per-node reserve frees a logical CPU rather
/// than a core and so does not yet give an on-node dispatcher a free core to
/// land on.
pub(crate) fn order_pin_targets(
    cpus: &[usize],
    cores: Option<&crate::core_topology::CoreTopology>,
) -> Vec<usize> {
    let Some(cores) = cores else {
        return cpus.to_vec();
    };
    let leaders = cores.leaders_within(cpus);
    let leader_set: std::collections::BTreeSet<usize> = leaders.iter().copied().collect();
    let mut ordered = leaders;
    ordered.extend(cpus.iter().copied().filter(|cpu| !leader_set.contains(cpu)));
    ordered
}

/// Windows NUMA topology discovery and per-thread pinning.
///
/// All Win32 calls here are bounded (fixed-size or single-call-sized buffers),
/// null-checked, and fall back gracefully (`None` / `Err`) instead of panicking
/// so an unavailable API never aborts decode. CPU indices use the crate-wide
/// `group * 64 + bit` encoding (see the module docs), which round-trips through
/// `GROUP_AFFINITY { Group, Mask }` and handles hosts with more than 64 logical
/// CPUs (multiple processor groups) correctly.
#[cfg(target_os = "windows")]
mod windows_imp {
    use std::mem::{size_of, zeroed};

    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationNumaNode, SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };
    use windows_sys::Win32::System::Threading::{
        GetCurrentThread, GetThreadGroupAffinity, SetThreadGroupAffinity,
    };

    /// Bits per processor group mask (`KAFFINITY` is 64-bit on x64/arm64 Windows).
    const GROUP_BITS: usize = 64;

    /// Expand a `GROUP_AFFINITY` (a group number and its 64-bit CPU mask) into the
    /// crate-wide `group * 64 + bit` CPU indices it names.
    fn cpus_from_mask(group: u16, mask: usize) -> Vec<usize> {
        let base = group as usize * GROUP_BITS;
        (0..GROUP_BITS)
            .filter(|bit| mask & (1usize << bit) != 0)
            .map(|bit| base + bit)
            .collect()
    }

    /// Discover NUMA nodes as `(node_index, cpu_indices)` pairs via
    /// `GetLogicalProcessorInformationEx(RelationNumaNode)`. Returns `None` if the
    /// API cannot be queried; an empty/one-node result is left for the caller to
    /// reject.
    pub(super) fn numa_nodes() -> Option<Vec<(usize, Vec<usize>)>> {
        // First call with a null buffer to learn the required byte length.
        let mut len: u32 = 0;
        // SAFETY: passing a null buffer with length 0 is the documented "size
        // query" form; it only writes `len` and returns FALSE with
        // ERROR_INSUFFICIENT_BUFFER.
        unsafe {
            GetLogicalProcessorInformationEx(RelationNumaNode, std::ptr::null_mut(), &mut len);
        }
        if len == 0 {
            return None;
        }
        // Over-allocate as `u8` and keep the buffer alive for the whole walk.
        let mut buffer = vec![0u8; len as usize];
        // SAFETY: `buffer` is a live `len`-byte allocation; we pass its true
        // length. On success the OS fills it with a packed sequence of
        // variable-length `SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX` records.
        let ok = unsafe {
            GetLogicalProcessorInformationEx(
                RelationNumaNode,
                buffer.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
                &mut len,
            )
        };
        if ok == 0 {
            return None;
        }

        let mut nodes: Vec<(usize, Vec<usize>)> = Vec::new();
        let mut offset = 0usize;
        let end = len as usize;
        while offset + size_of::<u32>() * 2 <= end {
            // Read an owned copy with `read_unaligned`: the records live in a
            // `Vec<u8>` (alignment 1), so forming a `&SYSTEM_LOGICAL_PROCESSOR_
            // INFORMATION_EX` reference would be unaligned-reference UB. The
            // struct is `Copy`, so a by-value read is sound and cheap.
            // SAFETY: `offset` leaves at least the header (`Relationship`+`Size`)
            // inside the filled buffer, and the source is a valid, initialized
            // record the OS wrote; `read_unaligned` needs no alignment guarantee.
            let record = unsafe {
                std::ptr::read_unaligned(
                    buffer.as_ptr().add(offset) as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX
                )
            };
            let size = record.Size as usize;
            if size == 0 || offset + size > end {
                break;
            }
            if record.Relationship == RelationNumaNode {
                // SAFETY: the record's relationship is RelationNumaNode, so the
                // `NumaNode` union arm is the active one.
                let numa = unsafe { record.Anonymous.NumaNode };
                let node_index = numa.NodeNumber as usize;
                // A NUMA node names its CPUs by a single `GroupMask` group
                // affinity; recover the (group, mask) and expand to CPU indices.
                // SAFETY: reading the `GroupMask` arm of the group-mask union is
                // valid for a NUMA-node record.
                let group_mask = unsafe { numa.Anonymous.GroupMask };
                let cpus = cpus_from_mask(group_mask.Group, group_mask.Mask as usize);
                if !cpus.is_empty() {
                    nodes.push((node_index, cpus));
                }
            }
            offset += size;
        }
        Some(nodes)
    }

    /// Pin the calling thread to `cpu` (encoded `group * 64 + bit`) via
    /// `SetThreadGroupAffinity`, which is group-aware and therefore correct on
    /// hosts with more than 64 logical CPUs. Best-effort: a failure is returned
    /// as a message so the caller logs it once and continues unpinned.
    pub(super) fn pin_current_thread_to_cpu(cpu: usize) -> std::result::Result<(), String> {
        let group = (cpu / GROUP_BITS) as u16;
        let bit = cpu % GROUP_BITS;
        // SAFETY: `GROUP_AFFINITY` is plain-old-data; zeroing it and setting the
        // single group + mask we want is a valid initialization.
        let mut affinity =
            unsafe { zeroed::<windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY>() };
        affinity.Group = group;
        affinity.Mask = 1usize << bit;
        // SAFETY: `GetCurrentThread` returns a pseudo-handle valid for the call;
        // `affinity` is a live, fully-initialized `GROUP_AFFINITY`; passing a null
        // previous-affinity out-param is documented as "not returned".
        let ok =
            unsafe { SetThreadGroupAffinity(GetCurrentThread(), &affinity, std::ptr::null_mut()) };
        if ok != 0 {
            Ok(())
        } else {
            Err(format!(
                "SetThreadGroupAffinity(group={group}, cpu={cpu}) failed: {}",
                std::io::Error::last_os_error()
            ))
        }
    }

    /// The current process's allowed CPUs, derived from the calling thread's
    /// group affinity (`GetThreadGroupAffinity`). A Windows thread's affinity is
    /// always within a single processor group, so this reports the allowed CPUs
    /// of that group — the common case, and always complete on hosts with <= 64
    /// CPUs. Returns `None` on failure so the caller treats the set as unknown
    /// (unrestricted) rather than guessing.
    pub(super) fn allowed_cpus() -> Option<Vec<usize>> {
        // SAFETY: zero-initialize the POD out-param, then let the OS fill it.
        let mut affinity =
            unsafe { zeroed::<windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY>() };
        // SAFETY: `GetCurrentThread` pseudo-handle is valid for the call and
        // `affinity` is a live, writable `GROUP_AFFINITY`.
        let ok = unsafe { GetThreadGroupAffinity(GetCurrentThread(), &mut affinity) };
        if ok == 0 {
            return None;
        }
        let cpus = cpus_from_mask(affinity.Group, affinity.Mask as usize);
        (!cpus.is_empty()).then_some(cpus)
    }
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
        assert_eq!(
            DecodeAffinity::parse(Some("numa-split")).unwrap(),
            DecodeAffinity::NumaSplit
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
            ("bogus", "bogus"),   // malformed mode
            ("node:x", "node:x"), // non-integer node index
            ("node:9", "node:9"), // unknown node index
        ] {
            let err = DecodeAffinity::resolve(Some(raw), Some(&topology)).unwrap_err();
            assert!(
                err.contains(needle),
                "names rejected value `{needle}`: {err}"
            );
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

    #[test]
    fn split_workers_spreads_evenly_across_nodes() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..48).collect::<Vec<_>>());
        nodes.insert(1, (48..96).collect::<Vec<_>>());
        let topology = NumaTopology { nodes };

        let shards = topology.split_workers(32).unwrap();
        assert_eq!(shards.len(), 2);
        assert_eq!(shards[0].index, 0);
        assert_eq!(shards[0].workers, 16);
        assert_eq!(shards[0].cpus[0], 0);
        assert_eq!(shards[1].index, 1);
        assert_eq!(shards[1].workers, 16);
        assert_eq!(shards[1].cpus[0], 48);

        // Odd totals give the remainder to the lower-index node.
        let odd = topology.split_workers(33).unwrap();
        assert_eq!(odd[0].workers, 17);
        assert_eq!(odd[1].workers, 16);
    }

    #[test]
    fn split_workers_caps_workers_at_node_cpu_count() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, vec![0, 1]);
        nodes.insert(1, vec![2, 3]);
        let topology = NumaTopology { nodes };
        // Requesting 8 workers over 2x2 CPUs caps each node at its 2 CPUs.
        let shards = topology.split_workers(8).unwrap();
        assert_eq!(shards.len(), 2);
        assert!(shards.iter().all(|shard| shard.workers == 2));
    }

    #[test]
    fn split_workers_needs_two_populated_nodes() {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, vec![0, 1]);
        let topology = NumaTopology { nodes };
        assert!(topology.split_workers(4).is_none());
        assert!(topology.split_workers(0).is_none());
    }

    /// The exact conditions [`crate::decode_numa::build_from_env`] treats as
    /// "cannot split -> fall back to the flat single-node decode path": a
    /// single-node host, and a multi-node host whose process cpuset leaves fewer
    /// than two usable nodes. Both must yield `None` from the topology so the
    /// numa-split builder declines and decode stays correct (flat path) instead
    /// of pinning a degenerate single-node "split".
    #[test]
    fn numa_split_declines_when_topology_cannot_be_split() {
        // Single-node host: nothing to split across.
        let mut single = BTreeMap::new();
        single.insert(0, vec![0, 1, 2, 3]);
        let single = NumaTopology { nodes: single };
        assert!(
            single.split_workers(8).is_none(),
            "single-node host must decline the split"
        );

        // Multi-node host restricted (cpuset/taskset) to a single node: after
        // `restrict_to_allowed` only one node survives, so the split declines.
        let host = two_node_topology();
        let restricted = host.restrict_to_allowed(Some(&[0, 1, 2]));
        assert_eq!(restricted.node_count(), 1);
        assert!(
            restricted.split_workers(8).is_none(),
            "cpuset confined to one node must decline the split"
        );

        // A usable two-node topology still splits, so the decline is specific to
        // un-splittable hosts, not a blanket refusal.
        assert!(host.split_workers(8).is_some());
    }

    fn two_node_topology() -> NumaTopology {
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..8).collect::<Vec<_>>());
        nodes.insert(1, (8..16).collect::<Vec<_>>());
        NumaTopology { nodes }
    }

    #[test]
    fn restrict_to_allowed_filters_cpus_and_drops_empty_nodes() {
        let topology = two_node_topology();

        // Unknown allowed set (None) leaves the topology unchanged.
        assert_eq!(topology.restrict_to_allowed(None), topology);

        // A cpuset that spans both nodes keeps only the permitted CPUs per node.
        let allowed = vec![1, 2, 9, 10];
        let restricted = topology.restrict_to_allowed(Some(&allowed));
        assert_eq!(restricted.node_count(), 2);
        assert_eq!(restricted.cpus_for_node(0).unwrap(), &[1, 2]);
        assert_eq!(restricted.cpus_for_node(1).unwrap(), &[9, 10]);

        // A cpuset confined to one node drops the other node entirely.
        let allowed = vec![0, 1, 2];
        let restricted = topology.restrict_to_allowed(Some(&allowed));
        assert_eq!(restricted.node_count(), 1);
        assert_eq!(restricted.cpus_for_node(0).unwrap(), &[0, 1, 2]);
        assert!(restricted.cpus_for_node(1).is_none());

        // A cpuset naming no discovered CPU leaves an empty topology.
        let restricted = topology.restrict_to_allowed(Some(&[999]));
        assert_eq!(restricted.node_count(), 0);
    }

    #[test]
    fn auto_enable_pins_compact_on_multi_node_host() {
        // Env unset + >= 2 usable nodes + pinning supported -> auto `compact`.
        let topology = two_node_topology();
        let (affinity, log) = decide_affinity(None, Some(&topology), true, true).unwrap();
        assert_eq!(affinity, DecodeAffinity::Compact);
        let log = log.expect("auto-enable logs its decision");
        assert!(log.contains("auto-enabling `compact`"), "{log}");
        assert!(log.contains("off"), "mentions the opt-out: {log}");
    }

    #[test]
    fn auto_enable_declines_on_single_node_host() {
        // Single-node host (no multi-node topology): unchanged, quiet.
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..8).collect::<Vec<_>>());
        let single = NumaTopology { nodes };
        let (affinity, log) = decide_affinity(None, Some(&single), false, true).unwrap();
        assert_eq!(affinity, DecodeAffinity::Off);
        assert!(log.is_none(), "single-node auto-decision is quiet: {log:?}");

        // No topology at all behaves the same.
        let (affinity, log) = decide_affinity(None, None, false, true).unwrap();
        assert_eq!(affinity, DecodeAffinity::Off);
        assert!(log.is_none());
    }

    #[test]
    fn auto_enable_declines_when_cpuset_restricts_to_one_node() {
        // Host is multi-node, but the allowed set left < 2 usable nodes: decline
        // and log the container/taskset-safe reason.
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..4).collect::<Vec<_>>());
        let restricted = NumaTopology { nodes };
        let (affinity, log) = decide_affinity(None, Some(&restricted), true, true).unwrap();
        assert_eq!(affinity, DecodeAffinity::Off);
        let log = log.expect("cpuset decline logs its decision");
        assert!(log.contains("cpuset"), "{log}");
        assert!(log.contains("fewer than two"), "{log}");
    }

    #[test]
    fn auto_enable_declines_when_pinning_unsupported() {
        // Multi-node host but the OS cannot pin (e.g. macOS): stay off, log why.
        let topology = two_node_topology();
        let (affinity, log) = decide_affinity(None, Some(&topology), true, false).unwrap();
        assert_eq!(affinity, DecodeAffinity::Off);
        let log = log.expect("unsupported-OS decision logs why");
        assert!(log.contains("not supported on this OS"), "{log}");
    }

    #[test]
    fn explicit_off_opts_out_without_auto_log() {
        // An explicit `off` on a multi-node host disables pinning and adds no
        // auto-policy note (the operator asked for it).
        let topology = two_node_topology();
        let (affinity, log) = decide_affinity(Some("off"), Some(&topology), true, true).unwrap();
        assert_eq!(affinity, DecodeAffinity::Off);
        assert!(log.is_none());
    }

    #[test]
    fn explicit_modes_are_honored_and_validated_against_restricted_topology() {
        let topology = two_node_topology();

        let (affinity, log) =
            decide_affinity(Some("compact"), Some(&topology), true, true).unwrap();
        assert_eq!(affinity, DecodeAffinity::Compact);
        assert!(log.is_none(), "explicit modes carry no auto note");

        let (affinity, _) = decide_affinity(Some("node:1"), Some(&topology), true, true).unwrap();
        assert_eq!(affinity, DecodeAffinity::Node(1));

        // An explicit node index absent from the (restricted) topology is a clear
        // error, so an explicit mode never silently pins outside the cpuset.
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..4).collect::<Vec<_>>());
        let restricted = NumaTopology { nodes };
        let err = decide_affinity(Some("node:1"), Some(&restricted), true, true).unwrap_err();
        assert!(err.contains("node:1"), "{err}");
    }

    #[test]
    fn pinning_supported_matches_target_os() {
        // Compile-time sanity: the constant tracks the OSes that can actually pin.
        assert_eq!(
            pinning_supported(),
            cfg!(any(target_os = "linux", target_os = "windows"))
        );
    }

    #[test]
    fn choose_budget_cpus_prefers_a_single_node() {
        let topology = two_node_topology();
        let allowed: Vec<usize> = (0..16).collect();
        // Four CPUs fit inside node 0, so all four come from the smallest-index
        // node that covers the count -- packed on one node for locality.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 4, None).unwrap();
        assert_eq!(chosen, vec![0, 1, 2, 3]);
    }

    #[test]
    fn choose_budget_cpus_tops_up_across_nodes_when_no_node_covers_count() {
        let topology = two_node_topology();
        let allowed: Vec<usize> = (0..16).collect();
        // Twelve exceeds either 8-CPU node, so the largest node fills first and
        // the rest are topped up from the allowed set, never exceeding count.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 12, None).unwrap();
        assert_eq!(chosen.len(), 12);
        assert!(chosen.iter().all(|cpu| allowed.contains(cpu)));
    }

    #[test]
    fn choose_budget_cpus_never_exceeds_allowed_set() {
        let topology = two_node_topology();
        // The process is only allowed on four CPUs; a larger request is clamped
        // to exactly those CPUs and never pins outside the cpuset.
        let allowed = vec![8, 9, 10, 11];
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 8, None).unwrap();
        assert_eq!(chosen, vec![8, 9, 10, 11]);
    }

    #[test]
    fn choose_budget_cpus_without_topology_uses_allowed_prefix() {
        let allowed = vec![2, 3, 5, 7, 11];
        let chosen = choose_budget_cpus(None, Some(&allowed), 3, None).unwrap();
        assert_eq!(chosen, vec![2, 3, 5]);
    }

    #[test]
    fn choose_budget_cpus_returns_none_without_any_usable_cpu() {
        assert!(choose_budget_cpus(None, None, 4, None).is_none());
        assert!(choose_budget_cpus(Some(&two_node_topology()), Some(&[0]), 0, None).is_none());
    }

    /// 16 logical CPUs, siblings adjacent (`0-1`, `2-3`, …) — the layout of
    /// every AMD EPYC and post-Skylake Intel host we run on.
    fn adjacent_smt_topology() -> crate::core_topology::CoreTopology {
        crate::core_topology::CoreTopology::from_sibling_groups(
            (0..8).map(|core| vec![core * 2, core * 2 + 1]),
        )
    }

    #[test]
    fn budget_cpus_take_one_thread_per_core_before_doubling_up() {
        let topology = two_node_topology();
        let allowed: Vec<usize> = (0..16).collect();
        let smt = adjacent_smt_topology();
        // Without the SMT map a budget of 4 is `0,1,2,3` = two physical cores.
        // With it, the same four-CPU budget covers four distinct cores. Measured
        // on a 16-core EPYC 9V74: 4 threads on 4 cores ran an 8B QKV prefill in
        // 33.5 ms against 54.3 ms for 4 threads on 2 cores, using less CPU time.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 4, Some(&smt)).unwrap();
        assert_eq!(chosen, vec![0, 2, 4, 6]);
        assert_eq!(smt.physical_cores_within(&chosen), 4);
    }

    #[test]
    fn budget_cpus_fall_back_to_siblings_once_every_core_is_taken() {
        let topology = two_node_topology();
        let allowed: Vec<usize> = (0..16).collect();
        let smt = adjacent_smt_topology();
        // Node 0 holds 8 CPUs = 4 cores. A 6-CPU budget therefore takes node 0's
        // 4 leaders plus the 2 lowest siblings, and stays inside one node.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 6, Some(&smt)).unwrap();
        assert_eq!(chosen.len(), 6);
        assert!(chosen.iter().all(|cpu| (0..8).contains(cpu)));
        assert!([0, 2, 4, 6].iter().all(|cpu| chosen.contains(cpu)));
    }

    #[test]
    fn budget_cpus_stay_inside_the_cpuset_when_ranked_by_core() {
        let topology = two_node_topology();
        let smt = adjacent_smt_topology();
        // A taskset of one full core plus one lone sibling: ranking must never
        // invent CPU 4 just because it is a leader the process cannot use.
        let allowed = vec![2, 3, 5];
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 3, Some(&smt)).unwrap();
        assert_eq!(chosen, vec![2, 3, 5]);
        let two = choose_budget_cpus(Some(&topology), Some(&allowed), 2, Some(&smt)).unwrap();
        assert_eq!(
            two,
            vec![2, 5],
            "the two leaders reachable inside the cpuset"
        );
    }

    #[test]
    fn budget_of_the_whole_machine_is_unchanged_by_core_ranking() {
        let topology = two_node_topology();
        let allowed: Vec<usize> = (0..16).collect();
        let smt = adjacent_smt_topology();
        // Ranking only reorders; asking for every CPU must still return every
        // CPU, so the full-width case cannot regress.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 16, Some(&smt)).unwrap();
        assert_eq!(chosen, allowed);
    }

    #[test]
    fn smt_scaled_request_sizes_the_node_search_in_cores() {
        let smt = adjacent_smt_topology();
        assert_eq!(smt_scaled_request(4, Some(&smt)), 8);
        assert_eq!(smt_scaled_request(4, None), 4);
        let no_smt =
            crate::core_topology::CoreTopology::from_sibling_groups((0..8).map(|cpu| vec![cpu]));
        assert_eq!(smt_scaled_request(4, Some(&no_smt)), 4);
    }

    #[test]
    fn a_core_rich_node_wins_over_a_smaller_node_that_only_covers_the_thread_count() {
        // node 0: 4 CPUs = 2 cores. node 1: 8 CPUs = 4 cores.
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..4).collect::<Vec<_>>());
        nodes.insert(1, (4..12).collect::<Vec<_>>());
        let topology = NumaTopology { nodes };
        let allowed: Vec<usize> = (0..12).collect();
        let smt = crate::core_topology::CoreTopology::from_sibling_groups(
            (0..6).map(|core| vec![core * 2, core * 2 + 1]),
        );
        // Sized in threads, node 0 "covers" a budget of 4 -- but it is two cores.
        // Sized in cores the search needs 8 CPUs, which only node 1 has, so the
        // budget lands on four distinct cores of one node instead of two.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 4, Some(&smt)).unwrap();
        assert_eq!(chosen, vec![4, 6, 8, 10]);
        assert_eq!(smt.physical_cores_within(&chosen), 4);
        // Without the SMT map the old behaviour stands: the smallest-index node
        // that covers the thread count.
        let flat = choose_budget_cpus(Some(&topology), Some(&allowed), 4, None).unwrap();
        assert_eq!(flat, vec![0, 1, 2, 3]);
    }

    #[test]
    fn topped_up_cpus_compete_for_core_slots_instead_of_being_appended() {
        // Two 6-CPU nodes on a 16-CPU host: no node covers a budget of 8, so the
        // selection is topped up from the allowed set.
        let mut nodes = BTreeMap::new();
        nodes.insert(0, (0..6).collect::<Vec<_>>());
        nodes.insert(1, (6..12).collect::<Vec<_>>());
        let topology = NumaTopology { nodes };
        let allowed: Vec<usize> = (0..16).collect();
        let smt = adjacent_smt_topology();
        // Ranking *after* the top-up spends all eight slots on eight distinct
        // cores. Appending the top-up to an already-full mask (the old order)
        // returned `[0, 1, 6, 7, 8, 9, 10, 11]` -- four cores for eight threads.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 8, Some(&smt)).unwrap();
        assert_eq!(chosen, vec![0, 2, 4, 6, 8, 10, 12, 14]);
        assert_eq!(smt.physical_cores_within(&chosen), 8);
    }

    #[test]
    fn a_budget_that_fits_one_node_does_not_spill_across_nodes_for_cores() {
        let topology = two_node_topology();
        let allowed: Vec<usize> = (0..16).collect();
        let smt = adjacent_smt_topology();
        // Node 0 holds 8 CPUs = 4 cores. A budget of 8 could reach 8 cores by
        // spanning both nodes, but NUMA locality wins: the whole budget stays on
        // node 0 and simply uses both threads of each of its four cores.
        let chosen = choose_budget_cpus(Some(&topology), Some(&allowed), 8, Some(&smt)).unwrap();
        assert_eq!(chosen, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn pin_targets_are_ordered_one_per_core_then_siblings() {
        let smt = adjacent_smt_topology();
        let node: Vec<usize> = (0..8).collect();
        // The pool builder pins worker `i` to `cpus[i % len]`, so the first four
        // workers of an 8-CPU node must land on four different cores.
        let ordered = order_pin_targets(&node, Some(&smt));
        assert_eq!(ordered, vec![0, 2, 4, 6, 1, 3, 5, 7]);
        assert_eq!(order_pin_targets(&node, None), node);
    }
}
